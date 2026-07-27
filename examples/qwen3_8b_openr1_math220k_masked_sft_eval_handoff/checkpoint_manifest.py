#!/usr/bin/env python3
"""Create and verify portable, content-addressed HF checkpoint manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
IDENTITY_SCHEME = "hf_checkpoint_all_regular_files_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temp = Path(handle.name)
        temp.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temp is not None and temp.exists():
            temp.unlink()


def checkpoint_files(checkpoint: Path) -> list[dict[str, Any]]:
    checkpoint = checkpoint.resolve()
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"checkpoint has no config.json: {checkpoint}")
    paths = sorted(path for path in checkpoint.rglob("*") if path.is_file())
    if not paths:
        raise RuntimeError(f"checkpoint has no files: {checkpoint}")
    if any(path.is_symlink() for path in checkpoint.rglob("*")):
        raise RuntimeError(
            "checkpoint manifests require materialized files, not symlinks"
        )
    files = [
        {
            "name": path.relative_to(checkpoint).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    weight_files = [
        item
        for item in files
        if item["name"].endswith((".safetensors", ".bin"))
    ]
    if not weight_files:
        raise RuntimeError(f"checkpoint has no model weight files: {checkpoint}")
    return files


def content_identity(files: list[dict[str, Any]]) -> str:
    canonical = json.dumps(
        files, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def git_commit(repo_root: Path | None) -> str | None:
    if repo_root is None:
        return None
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint).resolve()
    files = checkpoint_files(checkpoint)
    return {
        "artifact_schema_version": SCHEMA_VERSION,
        "checkpoint_manifest_sha256": content_identity(files),
        "contract_hash": args.contract_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "family": args.family,
        "files": files,
        "final_iteration": args.final_iteration,
        "full_sft": args.full_sft,
        "identity_scheme": IDENTITY_SCHEME,
        "repo_commit": git_commit(
            Path(args.repo_root).resolve() if args.repo_root else None
        ),
        "source_checkpoint": str(checkpoint),
        "variant": args.variant,
    }


def validate_manifest(value: dict[str, Any]) -> None:
    if value.get("artifact_schema_version") != SCHEMA_VERSION:
        raise RuntimeError("checkpoint manifest schema mismatch")
    if value.get("identity_scheme") != IDENTITY_SCHEME:
        raise RuntimeError("checkpoint identity scheme mismatch")
    files = value.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("checkpoint manifest has no file inventory")
    if [item["name"] for item in files] != sorted(item["name"] for item in files):
        raise RuntimeError("checkpoint manifest file order is not canonical")
    expected = content_identity(files)
    if value.get("checkpoint_manifest_sha256") != expected:
        raise RuntimeError("checkpoint manifest content identity mismatch")


def verify(manifest_path: Path, checkpoint: Path) -> str:
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(value)
    actual_files = checkpoint_files(checkpoint)
    if actual_files != value["files"]:
        expected = {item["name"]: item for item in value["files"]}
        actual = {item["name"]: item for item in actual_files}
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(
            name
            for name in set(expected) & set(actual)
            if expected[name] != actual[name]
        )
        raise RuntimeError(
            "checkpoint copy differs from manifest: "
            f"missing={missing[:20]} extra={extra[:20]} changed={changed[:20]}"
        )
    identity = content_identity(actual_files)
    if identity != value["checkpoint_manifest_sha256"]:
        raise RuntimeError("checkpoint content identity mismatch")
    return identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record")
    record.add_argument("--checkpoint", required=True)
    record.add_argument("--output", required=True)
    record.add_argument("--family", choices=["shared", "40k", "2k"], required=True)
    record.add_argument("--variant", required=True)
    record.add_argument("--final-iteration", type=int)
    record.add_argument("--contract-hash")
    record.add_argument("--repo-root")
    record.add_argument(
        "--full-sft",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--manifest", required=True)
    verify_parser.add_argument("--checkpoint", required=True)

    identity = subparsers.add_parser("identity")
    identity.add_argument("--manifest", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "record":
        output = Path(args.output)
        if output.exists():
            existing = json.loads(output.read_text(encoding="utf-8"))
            validate_manifest(existing)
            actual = checkpoint_files(Path(args.checkpoint))
            if existing["files"] != actual:
                raise RuntimeError(
                    f"refusing to overwrite changed checkpoint manifest: {output}"
                )
            expected_metadata = {
                "contract_hash": args.contract_hash,
                "family": args.family,
                "final_iteration": args.final_iteration,
                "full_sft": args.full_sft,
                "variant": args.variant,
            }
            for key, expected in expected_metadata.items():
                if existing.get(key) != expected:
                    raise RuntimeError(
                        f"existing checkpoint manifest {key} mismatch: "
                        f"expected={expected!r} found={existing.get(key)!r}"
                    )
            print(existing["checkpoint_manifest_sha256"], flush=True)
            return
        value = build_manifest(args)
        atomic_json(output, value)
        print(value["checkpoint_manifest_sha256"], flush=True)
        return
    manifest = Path(args.manifest)
    if args.command == "verify":
        value = verify(manifest, Path(args.checkpoint))
        print(f"CHECKPOINT_VERIFIED sha256={value}", flush=True)
        return
    value = json.loads(manifest.read_text(encoding="utf-8"))
    validate_manifest(value)
    print(value["checkpoint_manifest_sha256"], flush=True)


if __name__ == "__main__":
    main()
