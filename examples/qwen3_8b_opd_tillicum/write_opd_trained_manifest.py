#!/usr/bin/env python3
"""Write an OPD trained-data manifest from actual rollout debug artifacts."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import torch


ROLLOUT_RE = re.compile(r"rollout_(\d+)\.pt$")


def sample_to_dict(sample: Any) -> dict[str, Any]:
    if isinstance(sample, dict):
        return sample
    if hasattr(sample, "to_dict"):
        return sample.to_dict()
    if hasattr(sample, "__dict__"):
        return dict(sample.__dict__)
    raise TypeError(f"Unsupported rollout sample type: {type(sample)!r}")


def rollout_id_from_path(path: Path) -> int | None:
    match = ROLLOUT_RE.search(path.name)
    return int(match.group(1)) if match else None


def load_rows_from_dir(path: Path, *, origin: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for rollout_path in sorted(path.glob("rollout_*.pt"), key=lambda p: rollout_id_from_path(p) or -1):
        rollout_id = rollout_id_from_path(rollout_path)
        payload = torch.load(rollout_path, map_location="cpu", weights_only=False)
        for sample in payload.get("samples", []):
            item = sample_to_dict(sample)
            metadata = item.get("metadata") or {}
            if metadata.get("source_row_id") is None:
                raise RuntimeError(f"Missing source_row_id in {rollout_path}")
            rows.append(
                {
                    "origin": origin,
                    "rollout_id": rollout_id,
                    "sample_index": item.get("index"),
                    "group_index": item.get("group_index"),
                    "source_row_id": int(metadata["source_row_id"]),
                    "prompt_sha256": metadata.get("prompt_sha256"),
                    "response_length": item.get("response_length"),
                    "status": item.get("status"),
                }
            )
    return rows


def main() -> None:
    env = os.environ
    previous_dir = Path(env.get("OPD_PREVIOUS_ROLLOUT_LOG_DIR", ""))
    current_dir = Path(env["OPD_ROLLOUT_LOG_DIR"])
    manifest_path = Path(env["OPD_TRAINED_MANIFEST"])
    hf_dir = Path(env["OPD_FINAL_HF_DIR"])
    save_dir = Path(env["OPD_SAVE_DIR"])
    final_rollout_id = int(env["OPD_FINAL_ROLLOUT_ID"])
    effective_samples = int(env["OPD_EFFECTIVE_TRAIN_SAMPLES"])
    already_trained_samples = int(env.get("OPD_ALREADY_TRAINED_SAMPLES", "0"))

    previous_rows = load_rows_from_dir(previous_dir, origin="previous_opd") if already_trained_samples else []
    current_rows = load_rows_from_dir(current_dir, origin="current_opd")
    rows = previous_rows + current_rows
    if len(previous_rows) != already_trained_samples:
        raise SystemExit(
            f"Expected {already_trained_samples} previous OPD rows from {previous_dir}, found {len(previous_rows)}"
        )
    if len(rows) != effective_samples:
        raise SystemExit(f"Expected {effective_samples} total trained OPD rows, found {len(rows)}")

    source_ids = [row["source_row_id"] for row in rows]
    if len(set(source_ids)) != len(source_ids):
        raise SystemExit("Duplicate source_row_ids detected in trained OPD manifest rows")

    dataset_state = save_dir / "rollout" / f"global_dataset_state_dict_{final_rollout_id}.pt"
    if not dataset_state.exists():
        raise SystemExit(f"Missing rollout dataset state for continuation: {dataset_state}")

    manifest = {
        "run_label": env["OPD_RUN_LABEL"],
        "opd_data_pool": env["OPD_JSONL"],
        "split_metadata": env.get("SPLIT_METADATA"),
        "continuation_metadata": env.get("OPD_CONTINUATION_METADATA"),
        "previous_run_label": env.get("OPD_PREVIOUS_RUN_LABEL"),
        "previous_save_dir": env.get("OPD_PREVIOUS_SAVE_DIR"),
        "previous_rollout_log_dir": str(previous_dir) if previous_rows else None,
        "current_rollout_log_dir": str(current_dir),
        "opd_train_size": int(env["OPD_TRAIN_SIZE"]),
        "opd_effective_samples": effective_samples,
        "already_trained_samples": already_trained_samples,
        "newly_trained_samples": len(current_rows),
        "rollout_start_id": int(env.get("OPD_START_ROLLOUT_ID", "0") or 0),
        "rollout_final_id": final_rollout_id,
        "num_rollout": int(env["OPD_NUM_ROLLOUT"]),
        "rollout_batch_size": int(env["OPD_ROLLOUT_BATCH_SIZE"]),
        "global_batch_size": int(env["OPD_GLOBAL_BATCH_SIZE"]),
        "n_samples_per_prompt": int(env["OPD_N_SAMPLES_PER_PROMPT"]),
        "resume_from_full_optim_checkpoint": str(save_dir / f"iter_{final_rollout_id:07d}"),
        "initial_load_mode": env.get("OPD_INITIAL_LOAD_MODE"),
        "initial_load_dir": env.get("OPD_INITIAL_LOAD_DIR"),
        "prompt_contract_version": env.get("OPD_PROMPT_CONTRACT_VERSION"),
        "ref_load_dir": env.get("OPD_REF_LOAD_DIR"),
        "resume_from_hf_snapshot": str(hf_dir),
        "rollout_dataset_state": str(dataset_state),
        "trained_rows": rows,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    manifest_path.write_text(text, encoding="utf-8")
    hf_dir.mkdir(parents=True, exist_ok=True)
    (hf_dir / "opd_trained_manifest.json").write_text(text, encoding="utf-8")
    print(f"Wrote OPD trained manifest from rollout logs: {manifest_path}")
    print(f"Copied OPD trained manifest to: {hf_dir / 'opd_trained_manifest.json'}")


if __name__ == "__main__":
    main()
