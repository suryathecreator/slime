#!/usr/bin/env python3
"""Validate Qwen3 control-token ids and explicit thinking prompt rendering."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any


def normalize_token_ids(rendered: Any) -> list[int]:
    """Return one flat token-id sequence from tokenizer/template output."""
    if isinstance(rendered, Mapping):
        if "input_ids" not in rendered:
            raise ValueError("Tokenizer output is missing input_ids")
        rendered = rendered["input_ids"]
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    if isinstance(rendered, Sequence) and not isinstance(rendered, (str, bytes)):
        if len(rendered) == 1 and isinstance(rendered[0], Sequence) and not isinstance(rendered[0], (str, bytes)):
            rendered = rendered[0]
        if all(isinstance(token_id, Integral) for token_id in rendered):
            return [int(token_id) for token_id in rendered]
    raise ValueError(f"Expected one flat token-id sequence, got {type(rendered)!r}")


def semantic_terminal_audit(tokenizer: Any, token_ids: Any, terminal_token_id: int) -> dict[str, Any]:
    """Require the final semantic token to be terminal, allowing whitespace after it."""
    normalized = normalize_token_ids(token_ids)
    positions = [index for index, token_id in enumerate(normalized) if token_id == terminal_token_id]
    if not positions:
        raise ValueError(f"Token sequence does not contain terminal token id {terminal_token_id}")
    terminal_position = positions[-1]
    trailing_token_ids = normalized[terminal_position + 1 :]
    trailing_text = tokenizer.decode(trailing_token_ids, skip_special_tokens=False)
    if trailing_text.strip():
        raise ValueError(
            f"Non-whitespace content follows terminal token id {terminal_token_id}: {trailing_text!r}"
        )
    return {
        "input_ids": normalized,
        "terminal_token_id": terminal_token_id,
        "terminal_token_position": terminal_position,
        "trailing_token_ids": trailing_token_ids,
        "trailing_text": trailing_text,
    }


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
