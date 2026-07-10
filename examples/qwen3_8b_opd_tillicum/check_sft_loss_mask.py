#!/usr/bin/env python3
"""Validate that SFT loss masking preserves cleaned Qwen3 thinking traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.processing_utils import load_tokenizer


def assistant_text(messages: list[dict]) -> str:
    return "\n".join(str(message.get("content", "")) for message in messages if message.get("role") == "assistant")


def compact(text: str, width: int = 900) -> str:
    text = text.replace("\r", "")
    if len(text) <= width:
        return text
    head = width // 2
    tail = width - head
    return text[:head] + "\n...\n" + text[-tail:]


def iter_rows(path: Path, limit: int):
    with path.open() as f:
        for index, line in enumerate(f):
            if index >= limit:
                break
            yield index, json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument("--loss-mask-type", required=True)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--min-mask-raw-ratio", type=float, default=0.80)
    parser.add_argument("--require-think", action="store_true")
    parser.add_argument("--print-snippets", action="store_true")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    template_kwargs = {"enable_thinking": True} if args.loss_mask_type == "qwen3" else {}
    mask_generator = MultiTurnLossMaskGenerator(
        tokenizer,
        tokenizer_type=args.loss_mask_type,
        apply_chat_template_kwargs=template_kwargs,
    )
    default_mask_generator = (
        MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=args.loss_mask_type)
        if args.loss_mask_type == "qwen3"
        else None
    )

    checked = 0
    for row_index, row in iter_rows(args.data, args.num_samples):
        messages = row["messages"]
        assistant = assistant_text(messages)
        raw_assistant_tokens = tokenizer(assistant, add_special_tokens=False)["input_ids"]
        token_ids, loss_mask = mask_generator.get_loss_mask(messages)
        if default_mask_generator is not None:
            default_token_ids, default_loss_mask = default_mask_generator.get_loss_mask(messages)
            if token_ids != default_token_ids or loss_mask != default_loss_mask:
                raise SystemExit(
                    f"row {row_index}: explicit enable_thinking=True changed Qwen3 SFT token ids or loss mask"
                )
        response_length = mask_generator.get_response_lengths([loss_mask])[0]
        if response_length <= 0:
            raise SystemExit(f"row {row_index}: no trainable response tokens produced")
        train_target_ids = token_ids[-response_length:]
        train_target = tokenizer.decode(train_target_ids, skip_special_tokens=False)
        loss_tokens = int(sum(loss_mask))
        ratio = loss_tokens / max(len(raw_assistant_tokens), 1)

        print(
            "SFT_LOSS_MASK_CHECK "
            f"row={row_index} loss_mask_type={args.loss_mask_type} "
            f"raw_assistant_tokens={len(raw_assistant_tokens)} "
            f"token_ids={len(token_ids)} response_length={response_length} "
            f"loss_tokens={loss_tokens} loss_to_raw_ratio={ratio:.4f}"
        )

        if ratio < args.min_mask_raw_ratio:
            raise SystemExit(
                f"row {row_index}: loss mask only covers {ratio:.2%} of raw assistant tokens; "
                f"expected at least {args.min_mask_raw_ratio:.2%}"
            )
        if args.require_think and ("<think>" not in train_target or "</think>" not in train_target):
            raise SystemExit(f"row {row_index}: decoded train target does not contain complete think tags")
        if args.print_snippets:
            print(f"SFT_LOSS_MASK_SNIPPET row={row_index}")
            print(compact(train_target))
            print("SFT_LOSS_MASK_SNIPPET_END")
        checked += 1

    if checked == 0:
        raise SystemExit(f"No rows checked in {args.data}")
    print(f"SFT_LOSS_MASK_CHECK_OK checked={checked} loss_mask_type={args.loss_mask_type}")


if __name__ == "__main__":
    main()
