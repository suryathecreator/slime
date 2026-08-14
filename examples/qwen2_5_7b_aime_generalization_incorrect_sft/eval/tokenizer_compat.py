#!/usr/bin/env python3
"""Build manifest-preserving tokenizer overlays for transferred checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

OVERLAY_SCHEMA_VERSION = 1
NORMALIZATION_VERSION = "qwen2.5-transformers-4.57-extra-special-tokens-v1"
TOKENIZER_ASSETS = (
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
)
QWEN_SPECIAL_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _normalized_config(model: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = model / "tokenizer_config.json"
    tokenizer_path = model / "tokenizer.json"
    if not config_path.is_file() or not tokenizer_path.is_file():
        raise FileNotFoundError(f"missing required tokenizer files: {model}")
    config = load_json(config_path)
    original_extra = config.get("extra_special_tokens")
    existing_additional = config.get("additional_special_tokens")
    action = "none"
    if isinstance(original_extra, list):
        if not original_extra or not all(isinstance(token, str) and token for token in original_extra):
            raise RuntimeError(f"invalid legacy extra_special_tokens list: {config_path}")
        if len(set(original_extra)) != len(original_extra):
            raise RuntimeError(f"duplicate legacy extra_special_tokens: {config_path}")
        if existing_additional not in (None, original_extra):
            raise RuntimeError(f"conflicting additional_special_tokens: {config_path}")
        config["additional_special_tokens"] = original_extra
        config["extra_special_tokens"] = {}
        action = "legacy_list_to_additional_special_tokens"
    elif original_extra is None:
        config["extra_special_tokens"] = {}
    elif not isinstance(original_extra, dict):
        raise RuntimeError(f"unsupported extra_special_tokens value: {config_path}")

    embedded = config.get("chat_template")
    embedded_template = embedded if isinstance(embedded, str) and embedded else None
    external_path = model / "chat_template.jinja"
    external_template = external_path.read_text(encoding="utf-8") if external_path.is_file() else None
    if embedded_template is None and external_template is None:
        raise RuntimeError(f"checkpoint has no Qwen chat template: {model}")
    if embedded_template is not None and external_template is not None and embedded_template != external_template:
        raise RuntimeError(f"embedded and external chat templates differ: {model}")
    effective_template = external_template or embedded_template
    assert effective_template is not None
    config["chat_template"] = effective_template
    return config, {
        "chat_template_sha256": sha256_bytes(effective_template.encode()),
        "normalization_action": action,
        "source_tokenizer_config_sha256": sha256_file(config_path),
        "source_tokenizer_json_sha256": sha256_file(tokenizer_path),
    }


def overlay_spec(model: Path, checkpoint_identity: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(checkpoint_identity) != 64:
        raise RuntimeError(f"invalid checkpoint identity: {checkpoint_identity}")
    normalized_config, source = _normalized_config(model)
    normalized_text = canonical_json(normalized_config)
    identity_input = {
        "checkpoint_identity": checkpoint_identity,
        "normalization_version": NORMALIZATION_VERSION,
        "normalized_tokenizer_config_sha256": sha256_bytes(normalized_text.encode()),
        **source,
    }
    overlay_identity = sha256_bytes(canonical_json(identity_input).encode())
    metadata = {
        "artifact_schema_version": OVERLAY_SCHEMA_VERSION,
        "checkpoint_identity": checkpoint_identity,
        "normalization_version": NORMALIZATION_VERSION,
        "normalized_tokenizer_config_sha256": identity_input["normalized_tokenizer_config_sha256"],
        "overlay_identity": overlay_identity,
        "source_model": str(model.resolve()),
        **source,
    }
    return normalized_config, metadata


def expected_overlay_path(root: Path, target: str, model: Path, checkpoint_identity: str) -> Path:
    _, metadata = overlay_spec(model, checkpoint_identity)
    return root / target / str(metadata["overlay_identity"])


def validate_overlay(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    actual = load_json(path / "overlay_metadata.json")
    if actual != expected:
        raise RuntimeError(f"tokenizer overlay metadata changed: {path}")
    config_path = path / "tokenizer_config.json"
    if sha256_file(config_path) != expected["normalized_tokenizer_config_sha256"]:
        raise RuntimeError(f"tokenizer overlay config changed: {path}")
    if not (path / "tokenizer.json").is_file():
        raise RuntimeError(f"tokenizer overlay lacks tokenizer.json: {path}")
    return actual


def prepare_overlay(
    root: Path,
    target: str,
    model: Path,
    checkpoint_identity: str,
) -> tuple[Path, dict[str, Any]]:
    normalized_config, metadata = overlay_spec(model, checkpoint_identity)
    destination = root / target / str(metadata["overlay_identity"])
    if destination.is_dir():
        return destination, validate_overlay(destination, metadata)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        (temporary / "tokenizer_config.json").write_text(canonical_json(normalized_config), encoding="utf-8")
        for name in TOKENIZER_ASSETS:
            source = model / name
            if source.is_file():
                (temporary / name).symlink_to(source.resolve())
        (temporary / "overlay_metadata.json").write_text(canonical_json(metadata), encoding="utf-8")
        try:
            os.rename(temporary, destination)
        except FileExistsError:
            pass
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
    return destination, validate_overlay(destination, metadata)
