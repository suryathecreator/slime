#!/usr/bin/env python3
"""Capture immutable runtime and source metadata before any training."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temp = Path(handle.name)
    temp.replace(path)


def package(name: str) -> str:
    return importlib.metadata.version(name)


def optional_package(name: str) -> str | None:
    try:
        return package(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=os.environ["SLIME_REPO_ROOT"], text=True).strip()


def require_cached_ref(relative: str, expected: str) -> dict[str, str]:
    path = Path(os.environ["HF_HOME"]) / "hub" / relative / "refs" / "main"
    if not path.is_file():
        raise SystemExit(f"Missing cached source revision marker: {path}")
    actual = path.read_text().strip()
    if actual != expected:
        raise SystemExit(f"Cached source revision mismatch at {path}: {actual} != {expected}")
    return {"path": str(path), "revision": actual, "sha256": sha256_file(path)}


def main() -> None:
    import torch
    from transformers import AutoConfig, AutoTokenizer

    contract_path = Path(os.environ["CONTRACT_FILE"])
    contract = json.loads(contract_path.read_text())
    commit = git("rev-parse", "HEAD")
    implementation_commit = contract["repository_commit"]
    if implementation_commit != contract["slime_commit"] or "TO_BE_PINNED" in implementation_commit:
        raise SystemExit(f"Contract implementation commit is not pinned consistently: {implementation_commit}")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", implementation_commit, commit],
        cwd=os.environ["SLIME_REPO_ROOT"],
        check=False,
    ).returncode == 0
    if not ancestor:
        raise SystemExit(f"Contract implementation commit {implementation_commit} is not an ancestor of current HEAD {commit}")
    tokenizer = AutoTokenizer.from_pretrained(os.environ["STUDENT_HF_DIR"], trust_remote_code=True)
    token_ids = {token: tokenizer.convert_tokens_to_ids(token) for token in ("<|endoftext|>", "<|im_end|>", "<think>", "</think>")}
    expected = {"<|endoftext|>": 151643, "<|im_end|>": 151645, "<think>": 151667, "</think>": 151668}
    if token_ids != expected:
        raise SystemExit(f"Tokenizer contract mismatch: {token_ids} != {expected}")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": os.environ["EVAL_PROMPT"]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    student_config = AutoConfig.from_pretrained(os.environ["STUDENT_HF_DIR"], trust_remote_code=True)
    teacher_config = AutoConfig.from_pretrained(os.environ["TEACHER_HF_DIR"], trust_remote_code=True)
    cached_refs = {
        "student": require_cached_ref("models--Qwen--Qwen3-8B-Base", os.environ["STUDENT_HF_REVISION"]),
        "teacher": require_cached_ref("models--Qwen--Qwen3-32B", os.environ["TEACHER_HF_REVISION"]),
        "openthoughts3": require_cached_ref("datasets--open-thoughts--OpenThoughts3-1.2M", os.environ["OT3_REVISION"]),
        "math500": require_cached_ref("datasets--HuggingFaceH4--MATH-500", os.environ["MATH500_REVISION"]),
    }
    manifest = {
        "schema_version": 1,
        "contract_sha256": sha256_file(contract_path),
        "contract_hash_prefix": os.environ["CONTRACT_HASH"],
        "implementation_commit": implementation_commit,
        "launch_commit": commit,
        "branch": git("branch", "--show-current"),
        "megatron_lm_commit": contract["megatron_lm_commit"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: package(name)
            for name in ("torch", "transformers", "ray", "sglang", "flash-attn", "math-verify", "lingua-language-detector")
        },
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nccl_package": optional_package("nvidia-nccl-cu12"),
        "external_vllm_version": "0.24.0",
        "token_ids": token_ids,
        "student_context": int(student_config.max_position_embeddings),
        "teacher_context": int(teacher_config.max_position_embeddings),
        "rendered_prompt": rendered,
        "rendered_prompt_token_ids": tokenizer.encode(rendered, add_special_tokens=False),
        "source_paths": {
            "student": str(Path(os.environ["STUDENT_HF_DIR"]).resolve()),
            "teacher": str(Path(os.environ["TEACHER_HF_DIR"]).resolve()),
            "hf_cache": str(Path(os.environ["HF_HOME"]).resolve()),
        },
        "cached_source_refs": cached_refs,
        "source_config_hashes": {
            "student_config": sha256_file(Path(os.environ["STUDENT_HF_DIR"]) / "config.json"),
            "student_tokenizer_config": sha256_file(Path(os.environ["STUDENT_HF_DIR"]) / "tokenizer_config.json"),
            "teacher_config": sha256_file(Path(os.environ["TEACHER_HF_DIR"]) / "config.json"),
            "teacher_tokenizer_config": sha256_file(Path(os.environ["TEACHER_HF_DIR"]) / "tokenizer_config.json"),
        },
    }
    output = Path(os.environ["MANIFEST_ROOT"]) / "environment.json"
    atomic_text(output, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
