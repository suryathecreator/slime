#!/usr/bin/env python3
"""Validate a pinned base architecture, weight inventory, and chat boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.qwen_correct_only_openr1_math220k_sft_2k.data_utils import normalize_token_ids
from examples.qwen_correct_only_openr1_math220k_sft_2k.experiment import load_contract


def validate_weight_inventory(model: Path) -> list[str]:
    index_path = model / "model.safetensors.index.json"
    if not index_path.is_file():
        single = model / "model.safetensors"
        if not single.is_file() or single.stat().st_size <= 0:
            raise ValueError(f"model has no materialized safetensors weights: {model}")
        return [single.name]
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("safetensors index has no weight map")
    shard_names = sorted({str(name) for name in weight_map.values()})
    for name in shard_names:
        shard = model / name
        if not shard.is_file() or shard.stat().st_size <= 0:
            raise ValueError(f"missing or empty weight shard: {shard}")
    return shard_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    args = parser.parse_args()
    contract = load_contract()
    try:
        spec = contract["models"][args.model_key]
    except KeyError as error:
        raise ValueError(f"unknown model key: {args.model_key}") from error
    config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    for key, expected in spec["architecture"].items():
        if config.get(key) != expected:
            raise ValueError(
                f"{args.model_key} architecture drift {key}: {config.get(key)!r} != {expected!r}"
            )
    shards = validate_weight_inventory(args.model)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    added = {str(token): int(token_id) for token, token_id in tokenizer.get_added_vocab().items()}
    for token, token_id in {
        "<|endoftext|>": 151643,
        "<|im_start|>": 151644,
        "<|im_end|>": 151645,
    }.items():
        if added.get(token) != token_id:
            raise ValueError(f"{args.model_key} control-token drift for {token}: {added.get(token)}")
    if tokenizer.eos_token_id != 151643 or tokenizer.pad_token_id != 151643:
        raise ValueError(f"{args.model_key} EOS/pad IDs drifted")
    rendered = normalize_token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "x"}],
            tokenize=True,
            add_generation_prompt=True,
            **spec["apply_chat_template_kwargs"],
        )
    )
    if not tokenizer.decode(rendered, skip_special_tokens=False).endswith(
        "<|im_start|>assistant\n"
    ):
        raise ValueError(f"{args.model_key} generation prompt does not end at assistant entry")
    print(
        f"BASE_MODEL_VALID model={args.model_key} shards={len(shards)} "
        f"model_type={config['model_type']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
