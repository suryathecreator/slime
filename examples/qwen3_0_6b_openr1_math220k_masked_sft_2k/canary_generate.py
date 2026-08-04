#!/usr/bin/env python3
"""Compare deterministic base/custom/stock canary completions before full training."""

from __future__ import annotations

import argparse
import json
import re
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
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt["user"]}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
            return_tensors="pt",
        ).to("cuda")
        with torch.inference_mode():
            output = model.generate(
                ids,
                do_sample=False,
                max_new_tokens=512,
                eos_token_id=[151645, 151643],
                pad_token_id=151643,
            )
        completion = tokenizer.decode(output[0, ids.shape[1] :], skip_special_tokens=False)
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
                "generated_tokens": int(output.shape[1] - ids.shape[1]),
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
    for label in ("custom_pretokenized_after_two_updates", "stock_messages_after_two_updates"):
        atomic_json(
            args.output,
            {
                "generation": results,
                "gate": "each trained role-copy and prompt-copy count must not exceed its base count",
            },
        )
        for metric in ("role_copy_starts", "prompt_copy_starts"):
            if results[label][metric] > results["base"][metric]:
                raise ValueError(
                    f"{label} worsened {metric}: {results[label][metric]} > {results['base'][metric]}"
                )
    print("CANARY_GENERATION_VALID models=3 prompts=8", flush=True)


if __name__ == "__main__":
    main()
