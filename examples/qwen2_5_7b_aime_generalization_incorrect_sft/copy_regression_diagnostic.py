#!/usr/bin/env python3
"""Small nonblocking before/after diagnostic for role-boundary prompt copying."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.qwen2_5_7b_aime_generalization_incorrect_sft.data_utils import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        local_files_only=True,
    )
    model.eval()
    problems = []
    with args.problems.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                problems.append(json.loads(line))
            if len(problems) == 2:
                break
    if len(problems) != 2:
        raise ValueError("copy diagnostic requires two prepared AIME problems")

    rows = []
    for problem in problems:
        prompt_value = problem.get("prompt", problem.get("query"))
        if not isinstance(prompt_value, str) or not prompt_value:
            raise ValueError("copy diagnostic row has neither a nonempty prompt nor query")
        prompt = prompt_value
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda:0")
        with torch.no_grad():
            output = model.generate(
                rendered,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(output[0, rendered.shape[-1] :], skip_special_tokens=False)
        stripped = completion.lstrip()
        rows.append(
            {
                "completion": completion,
                "generated_tokens": int(output.shape[-1] - rendered.shape[-1]),
                "problem_id": str(problem["problem_id"]),
                "prompt_copy_start": stripped.startswith(prompt),
                "system_copy_start": stripped.startswith("You are a helpful assistant."),
            }
        )
    atomic_json(
        args.output,
        {
            "artifact_schema_version": 1,
            "blocking": False,
            "model": str(args.model),
            "phase": args.phase,
            "rows": rows,
        },
    )
    print(
        "AIME_COPY_DIAGNOSTIC_COMPLETE "
        f"phase={args.phase} copy_starts={sum(row['prompt_copy_start'] or row['system_copy_start'] for row in rows)}/2",
        flush=True,
    )


if __name__ == "__main__":
    main()
