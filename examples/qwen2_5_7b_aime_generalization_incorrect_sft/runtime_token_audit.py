#!/usr/bin/env python3
"""Dump and verify exact shifted-label supervision before/inside a real canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from examples.qwen2_5_7b_aime_generalization_incorrect_sft.data_utils import (
    assert_record,
    atomic_json,
)
from examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.weighted_sft_rollout import (
    generate_rollout,
)
from slime.utils.types import Sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--rollout-data", type=Path, action="append", default=[])
    parser.add_argument("--train-data", type=Path, action="append", default=[])
    parser.add_argument("--max-sequence-length", type=int, default=32768)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--data-parallel-size", type=int, required=True)
    return parser.parse_args()


def compact_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def source_records(path: Path, max_sequence_length: int) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            assert_record(row, max_sequence_length)
            trace_id = str(row["metadata"]["trace_id"])
            if trace_id in records:
                raise ValueError(f"duplicate trace_id at line {line_number}: {trace_id}")
            records[trace_id] = row
    if not records:
        raise ValueError(f"empty training dataset: {path}")
    return records


class FixedBuffer:
    def __init__(self, samples: list[list[Sample]]) -> None:
        self.samples = samples

    def get_samples(self, count: int) -> list[list[Sample]]:
        if count != len(self.samples):
            raise ValueError(f"adapter requested {count}, prepared {len(self.samples)}")
        return self.samples


def realize_with_adapter(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = [[Sample(prompt=list(row["input_ids"]), metadata=dict(row["metadata"]))] for row in records]
    result = generate_rollout(
        SimpleNamespace(
            rollout_global_dataset=True,
            rollout_batch_size=len(groups),
        ),
        0,
        FixedBuffer(groups),
        evaluation=False,
    )
    return [group[0].to_dict() for group in result]


def to_list(value: Any, cast: type[int] | type[float]) -> list[int] | list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1).tolist()
    return [cast(item) for item in value]


def validate_runtime_sample(
    sample: dict[str, Any], sources: dict[str, dict[str, Any]]
) -> tuple[str, list[int], list[float]]:
    metadata = sample.get("metadata") or {}
    trace_id = str(metadata.get("trace_id", ""))
    if trace_id not in sources:
        raise ValueError(f"runtime sample has unknown trace_id: {trace_id!r}")
    tokens = to_list(sample["tokens"], int)
    weights = to_list(sample["loss_mask"], float)
    source = sources[trace_id]
    if tokens != [int(value) for value in source["input_ids"]]:
        raise ValueError(f"runtime token drift: {trace_id}")
    if int(sample["response_length"]) != int(source["metadata"]["response_length"]):
        raise ValueError(f"runtime response boundary drift: {trace_id}")
    if weights != [float(value) for value in source["metadata"]["loss_weights"]]:
        raise ValueError(f"runtime loss-weight drift: {trace_id}")
    return trace_id, tokens, weights


def token_table(
    tokenizer: Any,
    trace_id: str,
    tokens: list[int],
    response_weights: list[float],
) -> dict[str, Any]:
    response_length = len(response_weights)
    prompt_length = len(tokens) - response_length
    if prompt_length < 1:
        raise ValueError(f"invalid prompt length: {trace_id}")
    position_weights = [0.0] * (prompt_length - 1) + response_weights + [0.0]
    if len(position_weights) != len(tokens):
        raise AssertionError("shifted loss-weight alignment is not token-length exact")
    entries = []
    for input_position, input_id in enumerate(tokens):
        label_position = input_position + 1
        label_id = tokens[label_position] if label_position < len(tokens) else None
        entries.append(
            {
                "input_decoded": tokenizer.decode([input_id], clean_up_tokenization_spaces=False),
                "input_position": input_position,
                "input_token": tokenizer.convert_ids_to_tokens(input_id),
                "input_token_id": input_id,
                "label_decoded": (
                    tokenizer.decode([label_id], clean_up_tokenization_spaces=False) if label_id is not None else None
                ),
                "label_position": label_position if label_id is not None else None,
                "label_token": (tokenizer.convert_ids_to_tokens(label_id) if label_id is not None else None),
                "label_token_id": label_id,
                "loss_weight": position_weights[input_position],
            }
        )
    return {
        "full_decoded": tokenizer.decode(tokens, skip_special_tokens=False),
        "prompt_length": prompt_length,
        "response_length": response_length,
        "supervised_weight_mass": sum(response_weights),
        "token_count": len(tokens),
        "token_rows": entries,
        "trace_id": trace_id,
    }


def load_rollout_samples(path: Path) -> tuple[int, list[dict[str, Any]]]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    samples = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"real rollout dump has no samples: {path}")
    return int(payload["rollout_id"]), samples


def jsonable(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def validate_train_dumps(
    paths: list[Path],
    runtime_by_rollout: dict[int, dict[int, tuple[tuple[int, ...], tuple[float, ...]]]],
    tensor_parallel_size: int,
    data_parallel_size: int,
) -> dict[str, Any]:
    import torch

    model_rank_sample_count = 0
    unique_dp_sample_count = 0
    reports = []
    by_rollout: dict[int, dict[int, tuple[Path, dict[str, Any]]]] = {}
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        data = payload.get("rollout_data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            raise ValueError(f"actor train dump has no rollout_data: {path}")
        rollout_id = int(payload["rollout_id"])
        rank = int(payload["rank"])
        if rank in by_rollout.setdefault(rollout_id, {}):
            raise ValueError(f"duplicate actor rank {rank} for rollout {rollout_id}")
        by_rollout[rollout_id][rank] = (path, data)
    if set(by_rollout) != set(runtime_by_rollout):
        raise ValueError("actor/rollout evidence IDs differ")
    world_size = tensor_parallel_size * data_parallel_size
    if tensor_parallel_size < 1 or data_parallel_size < 1:
        raise ValueError("tensor/data parallel sizes must both be positive")
    expected_ranks = set(range(world_size))
    for rollout_id, rank_payloads in sorted(by_rollout.items()):
        if set(rank_payloads) != expected_ranks:
            raise ValueError(
                f"rollout {rollout_id} actor ranks must be exactly 0..{world_size - 1}; "
                f"found={sorted(rank_payloads)}"
            )
        realized_indices: set[int] = set()
        for data_parallel_rank in range(data_parallel_size):
            first_tensor_rank = data_parallel_rank * tensor_parallel_size
            tensor_ranks = range(first_tensor_rank, first_tensor_rank + tensor_parallel_size)
            representative_path, representative_data = rank_payloads[first_tensor_rank]
            for rank in tensor_ranks:
                if jsonable(rank_payloads[rank][1]) != jsonable(representative_data):
                    raise ValueError(
                        f"TP rank {rank} differs from representative rank {first_tensor_rank} "
                        f"for rollout {rollout_id}"
                    )
            data = representative_data
            tokens = data.get("tokens")
            masks = data.get("loss_masks")
            lengths = data.get("response_lengths")
            sample_indices = data.get("sample_indices")
            if not all(isinstance(value, list) for value in (tokens, masks, lengths, sample_indices)):
                raise ValueError(f"actor train dump has malformed fields: {representative_path}")
            if not (len(tokens) == len(masks) == len(lengths) == len(sample_indices)):
                raise ValueError(f"actor train dump field-length mismatch: {representative_path}")
            for token_values, mask_values, response_length, sample_index in zip(
                tokens, masks, lengths, sample_indices, strict=True
            ):
                token_list = to_list(token_values, int)
                mask_list = to_list(mask_values, float)
                if len(mask_list) != int(response_length):
                    raise ValueError(f"actor response/mask mismatch: {representative_path}")
                sample_index = int(sample_index)
                expected_pair = runtime_by_rollout[rollout_id].get(sample_index)
                actual_pair = (tuple(token_list), tuple(mask_list))
                if expected_pair != actual_pair:
                    raise ValueError(f"actor sample {sample_index} differs from rollout {rollout_id}")
                if sample_index in realized_indices:
                    raise ValueError(f"DP ranks duplicate sample {sample_index} in rollout {rollout_id}")
                realized_indices.add(sample_index)
                unique_dp_sample_count += 1
            reports.append(
                {
                    "path": str(representative_path),
                    "representative_rank": first_tensor_rank,
                    "rollout_id": rollout_id,
                    "samples": len(tokens),
                }
            )
        expected_indices = set(runtime_by_rollout[rollout_id])
        if realized_indices != expected_indices or len(realized_indices) != 64:
            raise ValueError(
                f"DP representatives do not cover exactly 64 runtime samples: "
                f"rollout={rollout_id} missing={sorted(expected_indices - realized_indices)} "
                f"extra={sorted(realized_indices - expected_indices)}"
            )
        model_rank_sample_count += sum(len(rank_payloads[rank][1]["tokens"]) for rank in range(world_size))
    return {
        "data_parallel_size": data_parallel_size,
        "model_rank_samples_including_tp_replicas": model_rank_sample_count,
        "representative_dp_payloads": reports,
        "tensor_parallel_size": tensor_parallel_size,
        "unique_dp_samples": unique_dp_sample_count,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    sources = source_records(args.data, args.max_sequence_length)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    first = next(iter(sources.values()))
    longest = max(sources.values(), key=lambda row: len(row["input_ids"]))
    selected = [first] if first is longest else [first, longest]
    adapter_samples = realize_with_adapter(selected)
    intended_tables = []
    for sample in adapter_samples:
        trace_id, tokens, weights = validate_runtime_sample(sample, sources)
        intended_tables.append(token_table(tokenizer, trace_id, tokens, weights))

    runtime_reports = []
    runtime_by_rollout: dict[int, dict[int, tuple[tuple[int, ...], tuple[float, ...]]]] = {}
    for path in args.rollout_data:
        rollout_id, samples = load_rollout_samples(path)
        if rollout_id in runtime_by_rollout:
            raise ValueError(f"duplicate rollout dump ID: {rollout_id}")
        realized = []
        for sample in samples:
            trace_id, tokens, weights = validate_runtime_sample(sample, sources)
            sample_index = int(sample["index"])
            runtime_by_rollout.setdefault(rollout_id, {})[sample_index] = (
                tuple(tokens),
                tuple(weights),
            )
            realized.append((trace_id, tokens, weights))
        if len(runtime_by_rollout[rollout_id]) != len(realized):
            raise ValueError(f"duplicate sample indices in rollout {rollout_id}")
        first_realized = realized[0]
        longest_realized = max(realized, key=lambda item: len(item[1]))
        selected_realized = (
            [first_realized] if first_realized is longest_realized else [first_realized, longest_realized]
        )
        runtime_reports.append(
            {
                "path": str(path),
                "rollout_id": rollout_id,
                "samples": len(realized),
                "token_tables": [
                    token_table(tokenizer, trace_id, tokens, weights)
                    for trace_id, tokens, weights in selected_realized
                ],
            }
        )
    if args.train_data and not runtime_by_rollout:
        raise ValueError("actor train dumps require at least one real rollout dump")
    train_report = validate_train_dumps(
        args.train_data,
        runtime_by_rollout,
        args.tensor_parallel_size,
        args.data_parallel_size,
    )
    artifact = {
        "artifact_schema_version": 1,
        "data_path": str(args.data),
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "intended_adapter_token_tables": intended_tables,
        "phase": args.phase,
        "real_runtime_batches": runtime_reports,
        "source_rows": len(sources),
        "train_actor_realization": train_report,
    }
    artifact["evidence_sha256"] = compact_hash(artifact)
    atomic_json(args.output, artifact)
    print(
        "AIME_RUNTIME_TOKEN_AUDIT_OK "
        f"phase={args.phase} source_rows={len(sources)} "
        f"runtime_batches={len(runtime_reports)} "
        f"actor_samples={train_report['unique_dp_samples']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
