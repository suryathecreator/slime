#!/usr/bin/env python3
"""Load one transferred checkpoint and run a short real vLLM generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--eval-file", required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt

    with Path(args.eval_file).open(encoding="utf-8") as handle:
        row = json.loads(next(line for line in handle if line.strip()))
    rendered = str(row["rendered_prompt"])
    if hashlib.sha256(rendered.encode("utf-8")).hexdigest() != row["rendered_prompt_sha256"]:
        raise RuntimeError(f"rendered prompt hash mismatch: {row['problem_id']}")
    if "<|im_start|>system" in rendered:
        raise RuntimeError(f"rendered prompt contains a system turn: {row['problem_id']}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True, use_fast=True)
    # Verbatim bytes: the chat template is deliberately not applied.
    input_ids = [int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)]
    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=32768,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.92,
        max_num_seqs=1,
        seed=0,
        enable_prefix_caching=True,
    )
    output = llm.generate(
        [TokensPrompt(prompt_token_ids=input_ids)],
        sampling_params=[
            SamplingParams(
                n=1,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                min_p=0.0,
                max_tokens=64,
                seed=1234,
                stop_token_ids=[151643, 151645],
                ignore_eos=False,
            )
        ],
        use_tqdm=False,
    )
    if len(output) != 1 or len(output[0].outputs) != 1:
        raise RuntimeError("vLLM canary returned an unexpected output shape")
    print(
        f"SPLIT_AIME_H200_CANARY_OK problem={row['problem_id']} prompt_tokens={len(input_ids)} "
        f"generated_tokens={len(output[0].outputs[0].token_ids or [])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
