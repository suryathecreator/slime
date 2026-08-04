#!/usr/bin/env python3
"""Compare deterministic base/custom/stock canary completions before full training."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.data_utils import atomic_json


def load_prompts(path: Path, count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows.append(
                    {
                        "trace_id": str(row.get("metadata", {}).get("trace_id") or row["id"]),
                        "user": row["messages"][0]["content"],
                    }
                )
                if len(rows) == count:
                    break
    if len(rows) != count:
        raise ValueError(f"expected {count} generation prompts, found {len(rows)}")
    return rows


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_generation_inputs(encoded: Any, device: str) -> dict[str, Any]:
    """Move tokenizer output to a device and expose model.generate keyword inputs."""
    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise ValueError("tokenizer generation output is missing input_ids")
        inputs = {
            key: encoded[key].to(device)
            for key in ("input_ids", "attention_mask")
            if key in encoded
        }
        return inputs
    if not hasattr(encoded, "shape") or not hasattr(encoded, "to"):
        raise TypeError(f"unsupported tokenizer generation output {type(encoded).__name__}")
    return {"input_ids": encoded.to(device)}


def summarize_copy_regressions(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    base = results["base"]
    comparisons = []
    for label in (
        "custom_pretokenized_after_two_updates",
        "stock_messages_after_two_updates",
    ):
        for metric in ("role_copy_starts", "prompt_copy_starts"):
            base_count = int(base[metric])
            trained_count = int(results[label][metric])
            comparisons.append(
                {
                    "model": label,
                    "metric": metric,
                    "base_count": base_count,
                    "trained_count": trained_count,
                    "regression": trained_count > base_count,
                }
            )
    return {
        "policy": "diagnostic_only_does_not_block_training",
        "detected": any(item["regression"] for item in comparisons),
        "comparisons": comparisons,
    }


def generate_model(label: str, path: Path, prompts: list[dict[str, Any]], tokenizer: Any) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        local_files_only=True,
    ).to("cuda")
    model.eval()
    rows = []
    for prompt in prompts:
        generation_inputs = normalize_generation_inputs(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt["user"]}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
                return_tensors="pt",
            ),
            "cuda",
        )
        prompt_length = int(generation_inputs["input_ids"].shape[1])
        with torch.inference_mode():
            output = model.generate(
                **generation_inputs,
                do_sample=False,
                max_new_tokens=512,
                eos_token_id=[151645, 151643],
                pad_token_id=151643,
            )
        completion = tokenizer.decode(output[0, prompt_length:], skip_special_tokens=False)
        normalized = compact(completion)
        user = compact(prompt["user"])
        role_copy = normalized.startswith(
            ("<|im_start|>user", "<|im_start|>system", "<|im_start|>assistant")
        )
        prompt_copy = bool(user[:96]) and normalized.startswith(user[:96])
        rows.append(
            {
                "trace_id": prompt["trace_id"],
                "completion": completion,
                "role_copy_start": role_copy,
                "prompt_copy_start": prompt_copy,
                "valid_think_start": normalized.startswith("<think>"),
                "generated_tokens": int(output.shape[1] - prompt_length),
            }
        )
    del model
    torch.cuda.empty_cache()
    return {
        "label": label,
        "model": str(path),
        "rows": rows,
        "role_copy_starts": sum(row["role_copy_start"] for row in rows),
        "prompt_copy_starts": sum(row["prompt_copy_start"] for row in rows),
        "valid_think_starts": sum(row["valid_think_start"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--custom", type=Path, required=True)
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.base, trust_remote_code=True, local_files_only=True
    )
    prompts = load_prompts(args.prompts, args.count)
    results = {
        label: generate_model(label, path, prompts, tokenizer)
        for label, path in (
            ("base", args.base),
            ("custom_pretokenized_after_two_updates", args.custom),
            ("stock_messages_after_two_updates", args.stock),
        )
    }
    copy_regression = summarize_copy_regressions(results)
    atomic_json(
        args.output,
        {
            "generation": results,
            "copy_regression": copy_regression,
            "completion_storage": "verbatim decoded completion including special tokens",
        },
    )
    print(
        "CANARY_GENERATION_VALID "
        f"models=3 prompts={args.count} "
        f"copy_regression_detected={int(copy_regression['detected'])} "
        "copy_regression_policy=diagnostic_only",
        flush=True,
    )


if __name__ == "__main__":
    main()
