#!/usr/bin/env python3
"""Validate the pinned Qwen2.5-7B architecture and tokenizer inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_CONFIG = {
    "model_type": "qwen2",
    "hidden_size": 3584,
    "intermediate_size": 18944,
    "num_hidden_layers": 28,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "vocab_size": 152064,
    "max_position_embeddings": 131072,
    "tie_word_embeddings": False,
    "rope_theta": 1000000.0,
}
EXPECTED_ADDED = {
    "<|endoftext|>": 151643, "<|im_start|>": 151644, "<|im_end|>": 151645,
    "<|object_ref_start|>": 151646, "<|object_ref_end|>": 151647,
    "<|box_start|>": 151648, "<|box_end|>": 151649,
    "<|quad_start|>": 151650, "<|quad_end|>": 151651,
    "<|vision_start|>": 151652, "<|vision_end|>": 151653,
    "<|vision_pad|>": 151654, "<|image_pad|>": 151655, "<|video_pad|>": 151656,
    "<tool_call>": 151657, "</tool_call>": 151658,
    "<|fim_prefix|>": 151659, "<|fim_middle|>": 151660,
    "<|fim_suffix|>": 151661, "<|fim_pad|>": 151662,
    "<|repo_name|>": 151663, "<|file_sep|>": 151664,
}
EXPECTED_HASHES = {
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "c91efca15ceff6e9ee9424db58a6f59cd41294e550a86cbd07e3c1fb500b34f9",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    for key, expected in EXPECTED_CONFIG.items():
        if config.get(key) != expected:
            raise ValueError(f"Qwen2.5 config drift {key}: {config.get(key)!r} != {expected!r}")
    for name, expected in EXPECTED_HASHES.items():
        actual = sha256(args.model / name)
        if actual != expected:
            raise ValueError(f"pinned tokenizer hash drift {name}: {actual}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    actual_added = {
        str(token): int(token_id)
        for token, token_id in tokenizer.get_added_vocab().items()
    }
    if actual_added != EXPECTED_ADDED:
        raise ValueError(f"Qwen2.5 added-token inventory drift: {actual_added}")
    if tokenizer.eos_token_id != 151643 or tokenizer.pad_token_id != 151643:
        raise ValueError("Qwen2.5 EOS/pad IDs drifted")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if list(map(int, rendered))[-1] != 198:
        raise ValueError("Qwen2.5 generation prompt no longer ends at the assistant newline")
    print(
        "QWEN25_BASE_VALID architecture=Qwen2ForCausalLM native_context=131072 "
        "experiment_context=32768 added_controls=22",
        flush=True,
    )


if __name__ == "__main__":
    main()
