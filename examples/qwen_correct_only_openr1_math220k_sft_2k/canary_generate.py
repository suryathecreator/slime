#!/usr/bin/env python3
"""Store verbatim base/custom/stock canary completions and copy diagnostics."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from examples.qwen_correct_only_openr1_math220k_sft_2k.data_utils import atomic_json


def load_prompts(path: Path, count: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows.append(
                    {
                        "trace_id": str(row["metadata"]["trace_id"]),
                        "user": str(row["messages"][0]["content"]),
                    }
                )
                if len(rows) == count:
                    break
    if len(rows) != count:
        raise ValueError(f"expected {count} prompts, found {len(rows)}")
    return rows


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_generation_inputs(encoded: Any, device: str) -> dict[str, Any]:
    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise ValueError("tokenizer generation output is missing input_ids")
        return {
            key: encoded[key].to(device)
            for key in ("input_ids", "attention_mask")
            if key in encoded
        }
    if not hasattr(encoded, "to"):
        raise TypeError(f"unsupported tokenizer output {type(encoded).__name__}")
    return {"input_ids": encoded.to(device)}


def generate_model(
    label: str,
    path: Path,
    prompts: list[dict[str, str]],
    tokenizer: Any,
    template_kwargs: dict[str, Any],
) -> dict[str, Any]:
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
                return_tensors="pt",
                **template_kwargs,
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
        rows.append(
            {
                "trace_id": prompt["trace_id"],
                "completion": completion,
                "role_copy_start": normalized.startswith(
                    ("<|im_start|>user", "<|im_start|>system", "<|im_start|>assistant")
                ),
                "prompt_copy_start": bool(user[:96]) and normalized.startswith(user[:96]),
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
    parser.add_argument("--template-kwargs", default="{}")
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
    template_kwargs = json.loads(args.template_kwargs)
    generation = {
        label: generate_model(label, path, prompts, tokenizer, template_kwargs)
        for label, path in (
            ("base", args.base),
            ("custom_pretokenized_after_one_update", args.custom),
            ("stock_messages_after_one_update", args.stock),
        )
    }
    comparisons = []
    for label in (
        "custom_pretokenized_after_one_update",
        "stock_messages_after_one_update",
    ):
        for metric in ("role_copy_starts", "prompt_copy_starts"):
            comparisons.append(
                {
                    "model": label,
                    "metric": metric,
                    "base_count": int(generation["base"][metric]),
                    "trained_count": int(generation[label][metric]),
                    "regression": int(generation[label][metric])
                    > int(generation["base"][metric]),
                }
            )
    copy_regression = {
        "policy": "diagnostic_only_does_not_block_training",
        "detected": any(item["regression"] for item in comparisons),
        "comparisons": comparisons,
    }
    atomic_json(
        args.output,
        {
            "generation": generation,
            "copy_regression": copy_regression,
            "completion_storage": "verbatim decoded completion including special tokens",
        },
    )
    print(
        "CANARY_GENERATION_VALID models=3 "
        f"prompts={args.count} copy_regression_detected={int(copy_regression['detected'])} "
        "copy_regression_policy=diagnostic_only",
        flush=True,
    )


if __name__ == "__main__":
    main()
