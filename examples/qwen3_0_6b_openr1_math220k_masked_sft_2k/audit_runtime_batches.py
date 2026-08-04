#!/usr/bin/env python3
"""Audit every realized canary token, causal target, and shifted loss weight."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.data_utils import (
    atomic_json,
    atomic_jsonl,
)
from slime.utils.mask_utils import MultiTurnLossMaskGenerator


IM_START = 151644
IM_END = 151645
END_OF_TEXT = 151643


def plain(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
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
            "trace_id": str(metadata["trace_id"]),
            "response_length": int(metadata["response_length"]),
            "weights": list(map(float, metadata["loss_weights"])),
            "messages": None,
        }
    if len(result) != 32:
        raise ValueError("custom canary token sequences are not 32 unique rows")
    return result


def expected_stock(path: Path, tokenizer: Any) -> dict[tuple[int, ...], dict[str, Any]]:
    generator = MultiTurnLossMaskGenerator(
        tokenizer,
        tokenizer_type="qwen3",
        apply_chat_template_kwargs={"enable_thinking": True},
    )
    result: dict[tuple[int, ...], dict[str, Any]] = {}
    for row in load_jsonl(path):
        messages = row["messages"]
        tokens, full_mask = generator.get_loss_mask(messages)
        response_length = generator.get_response_lengths([full_mask])[0]
        trace = str(row.get("metadata", {}).get("trace_id") or str(row["id"]).split("/", 1)[-1])
        result[tuple(map(int, tokens))] = {
            "trace_id": trace,
            "response_length": int(response_length),
            "weights": list(map(float, full_mask[-response_length:])),
            "messages": messages,
        }
    if len(result) != 32:
        raise ValueError("stock canary token sequences are not 32 unique rows")
    return result


def dump_files(path: Path) -> list[Path]:
    files = sorted((path / "train_data").glob("*_*.pt"))
    if len(files) != 8:
        raise ValueError(f"expected 2 rollouts x 4 DP dumps in {path}, found {len(files)}")
    return files


def token_rows(
    path_name: str,
    rollout_id: int,
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
        target_id = tokens[index + 1] if index + 1 < len(tokens) else None
        yield {
            "path": path_name,
            "rollout_id": rollout_id,
            "rank": rank,
            "sample_order": sample_order,
            "trace_id": trace_id,
            "token_index": index,
            "input_token_id": token_id,
            "input_piece": tokenizer.decode([token_id], skip_special_tokens=False),
            "label_token_id": target_id,
            "label_piece": (
                tokenizer.decode([target_id], skip_special_tokens=False)
                if target_id is not None
                else None
            ),
            "shifted_loss_weight": shifted_weights[index],
        }


def audit_path(
    name: str,
    details: Path,
    expected: dict[tuple[int, ...], dict[str, Any]],
    tokenizer: Any,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    import torch

    seen: Counter[str] = Counter()
    rollout_counts: Counter[int] = Counter()
    supervised_mass = 0.0
    sequence_lengths: list[int] = []
    response_lengths: list[int] = []
    newline_weights: list[float] = []
    for file in dump_files(details):
        payload = torch.load(file, map_location="cpu", weights_only=False)
        data = payload["rollout_data"]
        rollout_id = int(payload["rollout_id"])
        rank = int(payload["rank"])
        microbatches = [int(value) for value in plain(data["num_microbatches"])]
        if microbatches != [1]:
            raise ValueError(f"{name} rollout={rollout_id} rank={rank} microbatches={microbatches}")
        tokens_batch = data["tokens"]
        lengths_batch = [int(value) for value in plain(data["response_lengths"])]
        weights_batch = data["loss_masks"]
        for sample_order, (raw_tokens, response_length, raw_weights) in enumerate(
            zip(tokens_batch, lengths_batch, weights_batch, strict=True)
        ):
            tokens = list(map(int, plain(raw_tokens)))
            weights = list(map(float, plain(raw_weights)))
            record = expected.get(tuple(tokens))
            if record is None:
                raise ValueError(f"{name} runtime tokens do not match a frozen source record")
            if response_length != record["response_length"] or weights != record["weights"]:
                raise ValueError(f"{name} runtime response weights drifted for {record['trace_id']}")
            prompt_length = len(tokens) - response_length
            response = tokens[prompt_length:]
            assistant_count = len(response) - 2 if name == "custom_pretokenized" else None
            if END_OF_TEXT in tokens:
                raise ValueError(f"{name} inserted synthetic endoftext for {record['trace_id']}")
            if not tokenizer.decode(tokens[:prompt_length], skip_special_tokens=False).endswith(
                "<|im_start|>assistant\n"
            ):
                raise ValueError(f"{name} prompt does not terminate at assistant entry")
            if name == "custom_pretokenized":
                if response[-2:] != [IM_END, 198] or weights[-2:] != [1.0, 0.0]:
                    raise ValueError(f"custom terminal weights drifted for {record['trace_id']}")
                if any(token in (IM_START, IM_END, END_OF_TEXT) for token in response[:assistant_count]):
                    raise ValueError(f"custom assistant contains an embedded chat boundary")
            else:
                if response[-2:] != [IM_END, 198] or weights[-2] != 1.0:
                    raise ValueError(f"stock assistant terminal drifted for {record['trace_id']}")
            trace_id = record["trace_id"]
            seen[trace_id] += 1
            rollout_counts[rollout_id] += 1
            supervised_mass += sum(weights)
            sequence_lengths.append(len(tokens))
            response_lengths.append(response_length)
            newline_weights.append(weights[-1])
            rows.extend(
                token_rows(
                    name,
                    rollout_id,
                    rank,
                    sample_order,
                    trace_id,
                    tokens,
                    weights,
                    response_length,
                    tokenizer,
                )
            )
    if seen != Counter({record["trace_id"]: 1 for record in expected.values()}):
        raise ValueError(f"{name} did not train every canary trace exactly once: {seen}")
    if rollout_counts != Counter({0: 16, 1: 16}):
        raise ValueError(f"{name} runtime global batches drifted: {rollout_counts}")
    return {
        "runtime_samples": sum(seen.values()),
        "runtime_rollouts": dict(sorted(rollout_counts.items())),
        "microbatches_per_dp_rank_per_update": 1,
        "total_supervised_weight": supervised_mass,
        "min_sequence_length": min(sequence_lengths),
        "max_sequence_length": max(sequence_lengths),
        "min_response_length": min(response_lengths),
        "max_response_length": max(response_lengths),
        "terminal_newline_weights": dict(Counter(newline_weights)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
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
    custom = expected_custom(args.custom_source)
    stock = expected_stock(args.stock_source, tokenizer)
    rows: list[dict[str, Any]] = []
    result = {
        "custom_pretokenized": audit_path(
            "custom_pretokenized", args.custom_details, custom, tokenizer, rows
        ),
        "stock_messages": audit_path(
            "stock_messages", args.stock_details, stock, tokenizer, rows
        ),
        "causal_alignment": "position_i_predicts_token_i_plus_1; response weights left padded by prompt_length_minus_1 and right padded by one",
    }
    atomic_jsonl(args.token_dump, rows)
    atomic_json(args.output, result)
    print(f"RUNTIME_BATCH_AUDIT_VALID samples=64 decoded_token_rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
