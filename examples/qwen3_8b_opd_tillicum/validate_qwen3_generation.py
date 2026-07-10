#!/usr/bin/env python3
"""Validate Qwen3 control-token ids and explicit thinking prompt rendering."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--im-end-token-id", type=int, required=True)
    parser.add_argument("--endoftext-token-id", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    from transformers import AutoTokenizer

    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    expected = {
        "<|im_end|>": args.im_end_token_id,
        "<|endoftext|>": args.endoftext_token_id,
    }
    actual = {token: tokenizer.convert_tokens_to_ids(token) for token in expected}
    if actual != expected:
        raise RuntimeError(f"Qwen3 token id mismatch: expected={expected} actual={actual}")

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Qwen3 generation preflight"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if "<|im_start|>assistant" not in prompt:
        raise RuntimeError(f"Qwen3 thinking prompt is missing assistant generation header: {prompt!r}")
    print(f"QWEN3_GENERATION_PREFLIGHT_OK token_ids={actual} enable_thinking=True prompt={prompt!r}")


if __name__ == "__main__":
    main()
