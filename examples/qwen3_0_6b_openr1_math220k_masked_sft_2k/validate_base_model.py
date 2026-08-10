#!/usr/bin/env python3
"""Validate the pinned Qwen3-0.6B-Base architecture and chat boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.data_utils import (
    normalize_token_ids,
)


EXPECTED_CONFIG = {
    "model_type": "qwen3",
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "tie_word_embeddings": True,
    "rope_theta": 1000000.0,
}
EXPECTED_ADDED = {
    "<|endoftext|>": 151643,
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
    "<|object_ref_start|>": 151646,
    "<|object_ref_end|>": 151647,
    "<|box_start|>": 151648,
    "<|box_end|>": 151649,
    "<|quad_start|>": 151650,
    "<|quad_end|>": 151651,
    "<|vision_start|>": 151652,
    "<|vision_end|>": 151653,
    "<|vision_pad|>": 151654,
    "<|image_pad|>": 151655,
    "<|video_pad|>": 151656,
    "<tool_call>": 151657,
    "</tool_call>": 151658,
    "<|fim_prefix|>": 151659,
    "<|fim_middle|>": 151660,
    "<|fim_suffix|>": 151661,
    "<|fim_pad|>": 151662,
    "<|repo_name|>": 151663,
    "<|file_sep|>": 151664,
    "<tool_response>": 151665,
    "</tool_response>": 151666,
    "<think>": 151667,
    "</think>": 151668,
}


def validate_weight_inventory(model: Path) -> list[str]:
    index_path = model / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("safetensors index has no weight map")
        names = sorted({str(value) for value in weight_map.values()})
    else:
        names = ["model.safetensors"]
    for name in names:
        path = model / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"missing or empty model weight file: {path}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    for key, expected in EXPECTED_CONFIG.items():
        if config.get(key) != expected:
            raise ValueError(f"Qwen3-0.6B config drift {key}: {config.get(key)!r} != {expected!r}")
    if int(config.get("max_position_embeddings", 0)) < 32768:
        raise ValueError("Qwen3-0.6B native context is below the 32768-token contract")
    weight_files = validate_weight_inventory(args.model)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    actual_added = {
        str(token): int(token_id)
        for token, token_id in tokenizer.get_added_vocab().items()
    }
    if actual_added != EXPECTED_ADDED:
        raise ValueError(f"Qwen3 added-token inventory drift: {actual_added}")
    if tokenizer.convert_tokens_to_ids("<|im_end|>") != 151645:
        raise ValueError("Qwen3 im_end token ID drifted")
    rendered = normalize_token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "x"}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    )
    if rendered[-3:] != [151644, 77091, 198]:
        raise ValueError(f"thinking generation boundary drifted: {rendered[-8:]}")
    print(
        "QWEN3_0P6B_BASE_VALID architecture=Qwen3ForCausalLM "
        f"experiment_context=32768 added_controls={len(actual_added)} weight_files={len(weight_files)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
