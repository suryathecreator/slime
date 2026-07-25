#!/usr/bin/env python3
"""Build a self-validating portable bundle from one completed MATH-500 eval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


EXPECTED_ROWS = 500
SCORER = "math500_strict_boxed_scorer_v3"


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


def manifest_identity(checkpoint_manifest: dict[str, Any]) -> str:
    files = checkpoint_manifest["files"]
    canonical = json.dumps(
        files, sort_keys=True, separators=(",", ":")
    ).encode()
    actual = hashlib.sha256(canonical).hexdigest()
    if actual != checkpoint_manifest.get("checkpoint_manifest_sha256"):
        raise RuntimeError("checkpoint manifest identity mismatch")
    return actual


def validate_evaluation(
    evaluation_dir: Path,
    stage: str,
    checkpoint_identity: str,
) -> tuple[dict[str, Any], list[Path]]:
    summary_path = evaluation_dir / "summary.json"
    summary_sha_path = evaluation_dir / "summary.sha256"
    if not summary_path.is_file() or not summary_sha_path.is_file():
        raise FileNotFoundError("evaluation needs summary.json and summary.sha256")
    expected_summary_hash = summary_sha_path.read_text(encoding="utf-8").strip()
    if sha256_file(summary_path) != expected_summary_hash:
        raise RuntimeError("summary.sha256 does not match summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("stage") != stage:
        raise RuntimeError("summary stage mismatch")
    if summary.get("scorer_version") != SCORER:
        raise RuntimeError("summary scorer mismatch")
    if int(summary.get("total", -1)) != EXPECTED_ROWS:
        raise RuntimeError("summary must contain exactly 500 examples")
    problem_paths = sorted((evaluation_dir / "problems").glob("problem_*.json"))
    if len(problem_paths) != EXPECTED_ROWS:
        raise RuntimeError(
            f"expected 500 raw problem files, found {len(problem_paths)}"
        )
    raw_by_index = {
        int(item["eval_index"]): item for item in summary.get("raw_artifacts", [])
    }
    if set(raw_by_index) != set(range(EXPECTED_ROWS)):
        raise RuntimeError("summary raw artifact manifest is incomplete")
    policy_hashes: set[str] = set()
    for index, path in enumerate(problem_paths):
        row = json.loads(path.read_text(encoding="utf-8"))
        if int(row.get("eval_index", -1)) != index:
            raise RuntimeError(f"problem row order mismatch at {path}")
        if row.get("stage") != stage:
            raise RuntimeError(f"problem stage mismatch at {path}")
        if (
            row.get("model_checkpoint_manifest_sha256")
            != checkpoint_identity
        ):
            raise RuntimeError(f"checkpoint identity mismatch at {path}")
        if row.get("scorer_trace", {}).get("scorer_version") != SCORER:
            raise RuntimeError(f"problem scorer mismatch at {path}")
        policy_hashes.add(str(row.get("policy_sha256")))
        if sha256_file(path) != raw_by_index[index].get("sha256"):
            raise RuntimeError(f"raw hash mismatch at {path}")
    if policy_hashes != {summary.get("policy_sha256")}:
        raise RuntimeError("evaluation contains mixed policies")
    return summary, problem_paths


def copy_or_link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def validate_existing_bundle(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "bundle_manifest.json"
    if not manifest_path.is_file():
        raise FileExistsError(
            f"incomplete bundle directory already exists: {output_dir}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts", [])
    actual_paths = sorted(
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "bundle_manifest.json"
    )
    if actual_paths != [item["path"] for item in artifacts]:
        raise RuntimeError("existing bundle has missing or extra artifacts")
    for item in artifacts:
        path = output_dir / item["path"]
        if path.stat().st_size != int(item["bytes"]) or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"existing bundle artifact changed: {path}")
    canonical = json.dumps(
        artifacts, sort_keys=True, separators=(",", ":")
    ).encode()
    if hashlib.sha256(canonical).hexdigest() != manifest.get(
        "bundle_content_sha256"
    ):
        raise RuntimeError("existing bundle identity mismatch")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--eval-policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_manifest = json.loads(
        args.checkpoint_manifest.read_text(encoding="utf-8")
    )
    checkpoint_identity = manifest_identity(checkpoint_manifest)
    _, problems = validate_evaluation(
        args.evaluation_dir, args.stage, checkpoint_identity
    )
    if args.output_dir.exists():
        existing = validate_existing_bundle(args.output_dir)
        if (
            existing.get("stage") != args.stage
            or existing.get("checkpoint_manifest_sha256")
            != checkpoint_identity
        ):
            raise RuntimeError("existing bundle belongs to a different run")
        print(
            f"RESULT_BUNDLE_ALREADY_COMPLETE stage={args.stage} "
            f"sha256={existing['bundle_content_sha256']} path={args.output_dir}",
            flush=True,
        )
        return
    args.output_dir.mkdir(parents=True)
    for source_name, target_name in (
        ("summary.json", "summary.json"),
        ("summary.sha256", "summary.sha256"),
    ):
        copy_or_link(
            args.evaluation_dir / source_name,
            args.output_dir / target_name,
        )
    copy_or_link(
        args.checkpoint_manifest,
        args.output_dir / "checkpoint_manifest.json",
    )
    copy_or_link(args.eval_policy, args.output_dir / "eval_policy.json")
    for source in problems:
        copy_or_link(
            source,
            args.output_dir / "problems" / source.name,
        )
    artifact_paths = sorted(
        path
        for path in args.output_dir.rglob("*")
        if path.is_file() and path.name != "bundle_manifest.json"
    )
    artifacts = [
        {
            "path": path.relative_to(args.output_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in artifact_paths
    ]
    canonical = json.dumps(
        artifacts, sort_keys=True, separators=(",", ":")
    ).encode()
    bundle = {
        "artifact_schema_version": 1,
        "artifacts": artifacts,
        "bundle_content_sha256": hashlib.sha256(canonical).hexdigest(),
        "checkpoint_manifest_sha256": checkpoint_identity,
        "eval_policy_file_sha256": sha256_file(
            args.output_dir / "eval_policy.json"
        ),
        "problem_count": EXPECTED_ROWS,
        "stage": args.stage,
        "summary_sha256": sha256_file(args.output_dir / "summary.json"),
    }
    atomic_json(args.output_dir / "bundle_manifest.json", bundle)
    print(
        f"RESULT_BUNDLE_COMPLETE stage={args.stage} "
        f"sha256={bundle['bundle_content_sha256']} path={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
