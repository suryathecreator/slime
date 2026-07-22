#!/usr/bin/env python3
"""Copy only compact completed audits/results into Git for a reviewed final commit."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, tempfile.NamedTemporaryFile("wb", dir=target.parent, delete=False) as handle:
        for chunk in iter(lambda: reader.read(8 << 20), b""):
            handle.write(chunk)
        temp = Path(handle.name)
    temp.replace(target)


def atomic_text(target: Path, value: str) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
        handle.write(value)
        temp = Path(handle.name)
    temp.replace(target)


def main() -> None:
    repo_dir = Path(__file__).resolve().parent
    output_root = Path(os.environ["OUTPUT_ROOT"])
    data_root = Path(os.environ["DATA_ROOT"])
    sources = {
        data_root / "cleanup_audit.json": repo_dir / "DATA_CLEANUP_AUDIT.json",
        data_root / "cleanup_audit.md": repo_dir / "DATA_CLEANUP_AUDIT.md",
        data_root / "cleanup_and_split_audit.json": repo_dir / "DATA_SPLIT_MANIFEST.json",
        Path(os.environ["MANIFEST_ROOT"]) / "environment.json": repo_dir / "ENVIRONMENT_MANIFEST.json",
        Path(os.environ["MANIFEST_ROOT"]) / "evaluation_environment.json": repo_dir / "EVALUATION_ENVIRONMENT_MANIFEST.json",
        Path(os.environ["MANIFEST_ROOT"]) / "submission.json": repo_dir / "SLURM_SUBMISSION.json",
        Path(os.environ["MANIFEST_ROOT"]) / "resubmission_after_opd_compiler_and_32k_proxy.json": repo_dir / "SLURM_FINAL_CHAIN.json",
        output_root / "checkpoint_reports" / "sft_manifest.json": repo_dir / "SFT_CHECKPOINT_MANIFEST.json",
        output_root / "checkpoint_reports" / "opd_manifest.json": repo_dir / "OPD_CHECKPOINT_MANIFEST.json",
        output_root / "final_report" / "results.json": repo_dir / "results.json",
        output_root / "final_report" / "RESULTS.md": repo_dir / "RESULTS.md",
        output_root / "final_report" / "STOPPING_OBJECTIVE_AUDIT.json": repo_dir / "STOPPING_OBJECTIVE_AUDIT.json",
        output_root / "final_report" / "STOPPING_OBJECTIVE_AUDIT.md": repo_dir / "STOPPING_OBJECTIVE_AUDIT.md",
        output_root / "final_report" / "opd_learning_dynamics.json": repo_dir / "OPD_LEARNING_DYNAMICS.json",
        output_root / "final_report" / "OPD_LEARNING_DYNAMICS.md": repo_dir / "OPD_LEARNING_DYNAMICS.md",
        output_root / "final_report" / "CAP_HIT_32K_ATTEMPT.json": repo_dir / "CAP_HIT_32K_ATTEMPT.json",
        output_root / "final_report" / "CAP_HIT_32K_ATTEMPT.md": repo_dir / "CAP_HIT_32K_ATTEMPT.md",
    }
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise SystemExit("Missing completed compact artifacts:\n" + "\n".join(missing))
    for source, target in sources.items():
        atomic_copy(source, target)
        print(f"SYNCED {source} -> {target}")
    combined_checkpoints = {
        "schema_version": 1,
        "sft": json.loads((repo_dir / "SFT_CHECKPOINT_MANIFEST.json").read_text()),
        "opd": json.loads((repo_dir / "OPD_CHECKPOINT_MANIFEST.json").read_text()),
    }
    atomic_text(repo_dir / "CHECKPOINT_MANIFEST.json", json.dumps(combined_checkpoints, indent=2, sort_keys=True) + "\n")
    manifest = {str(target.relative_to(repo_dir)): str(source) for source, target in sources.items()}
    atomic_text(repo_dir / "SYNCED_ARTIFACT_SOURCES.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
