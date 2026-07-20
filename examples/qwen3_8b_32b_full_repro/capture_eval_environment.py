#!/usr/bin/env python3
"""Capture and validate the separate vLLM evaluation runtime."""

from __future__ import annotations

import importlib.metadata
import json
import os
import tempfile
from pathlib import Path


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(value)
        temp = Path(handle.name)
    temp.replace(path)


def main() -> None:
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    versions = {
        "vllm": importlib.metadata.version("vllm"),
        "transformers": importlib.metadata.version("transformers"),
        "torch": importlib.metadata.version("torch"),
        "math_verify": importlib.metadata.version("math-verify"),
        "cuda": torch.version.cuda,
    }
    expected = {"vllm": "0.24.0", "transformers": "5.13.0", "math_verify": "0.9.0"}
    for name, value in expected.items():
        if versions[name] != value:
            raise SystemExit(f"Evaluation runtime mismatch for {name}: {versions[name]} != {value}")
    tokenizer = AutoTokenizer.from_pretrained(os.environ["STUDENT_HF_DIR"], trust_remote_code=True)
    token_ids = {token: tokenizer.convert_tokens_to_ids(token) for token in ("<|endoftext|>", "<|im_end|>")}
    if token_ids != {"<|endoftext|>": 151643, "<|im_end|>": 151645}:
        raise SystemExit(f"Evaluation tokenizer mismatch: {token_ids}")
    manifest = {
        "schema_version": 1,
        "versions": versions,
        "vllm_module_version": vllm.__version__,
        "api_probe": {"LLM": LLM.__name__, "SamplingParams": SamplingParams.__name__},
        "token_ids": token_ids,
        "sampling": {"temperature": 0.8, "top_p": 0.7, "top_k": -1, "n": 1, "max_new_tokens": 16384},
    }
    output = Path(os.environ["MANIFEST_ROOT"]) / "evaluation_environment.json"
    atomic_text(output, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
