#!/usr/bin/env python3
"""Audit every realized canary token, causal target, and shifted loss weight."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from examples.qwen_correct_only_openr1_math220k_sft_2k.data_utils import (
    atomic_json,
    atomic_jsonl,
)
from slime.utils.mask_utils import MultiTurnLossMaskGenerator


IM_START = 151644
IM_END = 151645
END_OF_TEXT = 151643


def plain(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def expected_custom(path: Path) -> dict[tuple[int, ...], dict[str, Any]]:
    result: dict[tuple[int, ...], dict[str, Any]] = {}
    for row in load_jsonl(path):
        tokens = tuple(map(int, row["input_ids"]))
        metadata = row["metadata"]
        result[tokens] = {
            "assistant_token_count": int(metadata["assistant_token_count"]),
            "trace_id": str(metadata["trace_id"]),
            "response_length": int(metadata["response_length"]),
            "weights": list(map(float, metadata["loss_weights"])),
        }
    if len(result) != 64:
        raise ValueError("custom canary token sequences are not 64 unique rows")
    return result


def expected_stock(
    path: Path,
    tokenizer: Any,
    loss_mask_type: str,
    template_kwargs: dict[str, Any],
) -> dict[tuple[int, ...], dict[str, Any]]:
    generator = MultiTurnLossMaskGenerator(
        tokenizer,
        tokenizer_type=loss_mask_type,
        apply_chat_template_kwargs=template_kwargs,
    )
    result: dict[tuple[int, ...], dict[str, Any]] = {}
    for row in load_jsonl(path):
        tokens, full_mask = generator.get_loss_mask(row["messages"])
        response_length = generator.get_response_lengths([full_mask])[0]
        result[tuple(map(int, tokens))] = {
            "assistant_token_count": None,
            "trace_id": str(row["metadata"]["trace_id"]),
            "response_length": int(response_length),
            "weights": list(map(float, full_mask[-response_length:])),
        }
    if len(result) != 64:
        raise ValueError("stock canary token sequences are not 64 unique rows")
    return result


def representative_dp_payloads(path: Path, dp_size: int, tensor_parallel_size: int) -> list[dict[str, Any]]:
    import torch

    if dp_size <= 0 or tensor_parallel_size <= 0:
        raise ValueError("DP and tensor-parallel sizes must be positive")
    files = sorted((path / "train_data").glob("*_*.pt"))
    world_size = dp_size * tensor_parallel_size
    if len(files) != world_size:
        raise ValueError(f"expected {world_size} model-rank dumps in {path}, found {len(files)}")
    payloads: dict[int, dict[str, Any]] = {}
    for file in files:
        payload = torch.load(file, map_location="cpu", weights_only=False)
        rank = int(payload["rank"])
        if rank in payloads:
            raise ValueError(f"duplicate model-rank dump rank={rank} in {path}")
        payloads[rank] = payload
    expected_ranks = set(range(world_size))
    if set(payloads) != expected_ranks:
        raise ValueError(
            f"model-rank dump inventory drift in {path}: "
            f"expected={sorted(expected_ranks)} actual={sorted(payloads)}"
        )

    representatives: list[dict[str, Any]] = []
    for dp_rank in range(dp_size):
        ranks = range(
            dp_rank * tensor_parallel_size,
            (dp_rank + 1) * tensor_parallel_size,
        )
        representative = payloads[ranks.start]
        expected_rollout_id = int(representative["rollout_id"])
        expected_data = plain(representative["rollout_data"])
        for rank in ranks:
            replica = payloads[rank]
            if int(replica["rollout_id"]) != expected_rollout_id or plain(replica["rollout_data"]) != expected_data:
                raise ValueError(f"tensor-parallel replica drift in {path}: " f"dp_rank={dp_rank} model_rank={rank}")
        representatives.append(representative)
    return representatives


def token_rows(
    path_name: str,
    rank: int,
    sample_order: int,
    trace_id: str,
    tokens: list[int],
    weights: list[float],
    response_length: int,
    tokenizer: Any,
):
    prompt_length = len(tokens) - response_length
    shifted_weights = [0.0] * (prompt_length - 1) + weights + [0.0]
    if len(shifted_weights) != len(tokens):
        raise AssertionError("shifted loss weights do not align to the causal stream")
    for index, token_id in enumerate(tokens):
        label_id = tokens[index + 1] if index + 1 < len(tokens) else None
        yield {
            "path": path_name,
            "rollout_id": 0,
            "rank": rank,
            "sample_order": sample_order,
            "trace_id": trace_id,
            "token_index": index,
            "input_token_id": token_id,
            "input_piece": tokenizer.decode([token_id], skip_special_tokens=False),
            "label_token_id": label_id,
            "label_piece": (
                tokenizer.decode([label_id], skip_special_tokens=False)
                if label_id is not None
                else None
            ),
            "shifted_loss_weight": shifted_weights[index],
        }


def audit_path(
    name: str,
    details: Path,
    expected: dict[tuple[int, ...], dict[str, Any]],
    tokenizer: Any,
    dp_size: int,
    tensor_parallel_size: int,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    seen: Counter[str] = Counter()
    microbatch_histogram: Counter[int] = Counter()
    sequence_lengths: list[int] = []
    response_lengths: list[int] = []
    total_supervised_weight = 0.0
    for payload in representative_dp_payloads(details, dp_size, tensor_parallel_size):
        if int(payload["rollout_id"]) != 0:
            raise ValueError(f"{name} contains a nonzero canary rollout")
        rank = int(payload["rank"])
        microbatches = [int(value) for value in plain(payload["rollout_data"]["num_microbatches"])]
        if len(microbatches) != 1:
            raise ValueError(f"{name} rank={rank} has unexpected update count: {microbatches}")
        microbatch_histogram[microbatches[0]] += 1
        data = payload["rollout_data"]
        for sample_order, (raw_tokens, raw_response_length, raw_weights) in enumerate(
            zip(
                data["tokens"],
                plain(data["response_lengths"]),
                data["loss_masks"],
                strict=True,
            )
        ):
            tokens = list(map(int, plain(raw_tokens)))
            response_length = int(raw_response_length)
            weights = list(map(float, plain(raw_weights)))
            record = expected.get(tuple(tokens))
            if record is None:
                raise ValueError(f"{name} runtime tokens do not match a frozen source record")
            if response_length != record["response_length"] or weights != record["weights"]:
                raise ValueError(f"{name} runtime weights drifted for {record['trace_id']}")
            prompt_length = len(tokens) - response_length
            if END_OF_TEXT in tokens:
                raise ValueError(f"{name} inserted synthetic endoftext for {record['trace_id']}")
            if not tokenizer.decode(tokens[:prompt_length], skip_special_tokens=False).endswith(
                "<|im_start|>assistant\n"
            ):
                raise ValueError(f"{name} prompt does not terminate at assistant entry")
            if name == "custom_pretokenized":
                assistant_count = int(record["assistant_token_count"])
                response = tokens[prompt_length:]
                if response[-2:] != [IM_END, 198] or weights[-2:] != [1.0, 0.0]:
                    raise ValueError(f"custom terminal weights drifted for {record['trace_id']}")
                if any(value != 1.0 for value in weights[:assistant_count]):
                    raise ValueError(f"custom assistant weight drifted for {record['trace_id']}")
                if any(token in (IM_START, IM_END, END_OF_TEXT) for token in response[:assistant_count]):
                    raise ValueError(f"custom assistant contains a chat boundary")
            trace_id = record["trace_id"]
            seen[trace_id] += 1
            sequence_lengths.append(len(tokens))
            response_lengths.append(response_length)
            total_supervised_weight += sum(weights)
            rows.extend(
                token_rows(
                    name,
                    rank,
                    sample_order,
                    trace_id,
                    tokens,
                    weights,
                    response_length,
                    tokenizer,
                )
            )
    expected_seen = Counter({record["trace_id"]: 1 for record in expected.values()})
    if seen != expected_seen:
        raise ValueError(f"{name} did not train every canary trace exactly once")
    return {
        "runtime_samples": sum(seen.values()),
        "runtime_rollouts": {"0": sum(seen.values())},
        "microbatches_per_dp_rank_histogram": dict(sorted(microbatch_histogram.items())),
        "total_supervised_weight": total_supervised_weight,
        "min_sequence_length": min(sequence_lengths),
        "max_sequence_length": max(sequence_lengths),
        "min_response_length": min(response_lengths),
        "max_response_length": max(response_lengths),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--loss-mask-type", choices=["qwen", "qwen3"], required=True)
    parser.add_argument("--template-kwargs", default="{}")
    parser.add_argument("--dp-size", type=int, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--custom-source", type=Path, required=True)
    parser.add_argument("--stock-source", type=Path, required=True)
    parser.add_argument("--custom-details", type=Path, required=True)
    parser.add_argument("--stock-details", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-dump", type=Path, required=True)
    args = parser.parse_args()
    for destination in (args.output, args.token_dump):
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite {destination}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True, local_files_only=True
    )
    template_kwargs = json.loads(args.template_kwargs)
    rows: list[dict[str, Any]] = []
    result = {
        "custom_pretokenized": audit_path(
            "custom_pretokenized",
            args.custom_details,
            expected_custom(args.custom_source),
            tokenizer,
            args.dp_size,
            args.tensor_parallel_size,
            rows,
        ),
        "stock_messages": audit_path(
            "stock_messages",
            args.stock_details,
            expected_stock(
                args.stock_source, tokenizer, args.loss_mask_type, template_kwargs
            ),
            tokenizer,
            args.dp_size,
            args.tensor_parallel_size,
            rows,
        ),
        "causal_alignment": (
            "position_i_predicts_token_i_plus_1; response weights left padded by "
            "prompt_length_minus_1 and right padded by one"
        ),
    }
    atomic_jsonl(args.token_dump, rows)
    atomic_json(args.output, result)
    print(
        f"RUNTIME_BATCH_AUDIT_VALID samples=128 decoded_token_rows={len(rows)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
