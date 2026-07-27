#!/usr/bin/env python3
"""Validate an external MATH-500 bundle and import compact repo results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


EXPECTED_ROWS = 500
SCORER = "math500_strict_boxed_scorer_v3"
HANDOFF_RELATIVE = Path(
    "examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
            temp = Path(handle.name)
        temp.replace(path)
    finally:
        if temp is not None and temp.exists():
            temp.unlink()


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def load_and_validate_bundle(bundle: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = bundle / "bundle_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise RuntimeError("bundle manifest has no artifact inventory")
    if [item["path"] for item in artifacts] != sorted(
        item["path"] for item in artifacts
    ):
        raise RuntimeError("bundle artifact order is not canonical")
    actual_paths = sorted(
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file() and path.name != "bundle_manifest.json"
    )
    expected_paths = [item["path"] for item in artifacts]
    if actual_paths != expected_paths:
        raise RuntimeError("bundle contains missing or untracked files")
    for item in artifacts:
        path = bundle / item["path"]
        if path.stat().st_size != int(item["bytes"]):
            raise RuntimeError(f"bundle size mismatch: {path}")
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"bundle hash mismatch: {path}")
    canonical = json.dumps(
        artifacts, sort_keys=True, separators=(",", ":")
    ).encode()
    if (
        hashlib.sha256(canonical).hexdigest()
        != manifest.get("bundle_content_sha256")
    ):
        raise RuntimeError("bundle content identity mismatch")
    summary = json.loads((bundle / "summary.json").read_text(encoding="utf-8"))
    summary_hash = sha256_file(bundle / "summary.json")
    if summary_hash != manifest.get("summary_sha256"):
        raise RuntimeError("bundle summary hash mismatch")
    if (bundle / "summary.sha256").read_text(encoding="utf-8").strip() != summary_hash:
        raise RuntimeError("summary.sha256 mismatch")
    if int(manifest.get("problem_count", -1)) != EXPECTED_ROWS:
        raise RuntimeError("bundle manifest problem count mismatch")
    problems = sorted((bundle / "problems").glob("problem_*.json"))
    if len(problems) != EXPECTED_ROWS:
        raise RuntimeError("bundle must contain all 500 problem files")
    stage = str(manifest.get("stage"))
    policy_hashes: set[str] = set()
    for index, path in enumerate(problems):
        row = json.loads(path.read_text(encoding="utf-8"))
        if int(row.get("eval_index", -1)) != index:
            raise RuntimeError(f"problem row order mismatch: {path}")
        if row.get("stage") != stage:
            raise RuntimeError(f"problem stage mismatch: {path}")
        if row.get("scorer_trace", {}).get("scorer_version") != SCORER:
            raise RuntimeError(f"problem scorer mismatch: {path}")
        if (
            row.get("model_checkpoint_manifest_sha256")
            != manifest.get("checkpoint_manifest_sha256")
        ):
            raise RuntimeError(f"problem checkpoint identity mismatch: {path}")
        policy_hashes.add(str(row.get("policy_sha256")))
    if policy_hashes != {summary.get("policy_sha256")}:
        raise RuntimeError("bundle contains mixed evaluation policies")
    if summary.get("stage") != stage or summary.get("scorer_version") != SCORER:
        raise RuntimeError("summary stage or scorer mismatch")
    if int(summary.get("total", -1)) != EXPECTED_ROWS:
        raise RuntimeError("summary row count mismatch")
    return manifest, summary


def validate_scientific_policy(bundle: Path, repo_root: Path) -> None:
    expected_path = repo_root / HANDOFF_RELATIVE / "eval_policy.json"
    if (bundle / "eval_policy.json").read_bytes() != expected_path.read_bytes():
        raise RuntimeError("external eval_policy.json differs from the repo policy")
    policy = json.loads(expected_path.read_text(encoding="utf-8"))
    problem = json.loads(
        (bundle / "problems/problem_0000.json").read_text(encoding="utf-8")
    )
    generated = problem["policy"]
    checks = {
        "dataset": policy["dataset"]["name"],
        "revision": policy["dataset"]["revision"],
        "temperature": policy["decoding"]["temperature"],
        "top_p": policy["decoding"]["top_p"],
        "top_k": policy["decoding"]["top_k"],
        "n": policy["decoding"]["n"],
        "seed": policy["decoding"]["seed"],
        "ignore_eos": policy["decoding"]["ignore_eos"],
        "max_new_tokens": policy["response_budget"][
            "configured_max_new_tokens"
        ],
        "stop_token_ids": policy["decoding"]["stop_token_ids"],
        "prompt_instruction": policy["prompt"]["instruction"],
        "enable_thinking": policy["prompt"]["enable_thinking"],
        "target_total_context_tokens": policy["response_budget"][
            "target_total_context_tokens"
        ],
        "safety_margin": policy["response_budget"]["safety_margin"],
        "scorer": policy["scoring"]["scorer"],
    }
    for key, expected in checks.items():
        if generated.get(key) != expected:
            raise RuntimeError(
                f"generated policy mismatch for {key}: "
                f"expected={expected!r} found={generated.get(key)!r}"
            )
    if generated.get("engine") != {
        key: value
        for key, value in policy["engine"].items()
        if key != "vllm_version"
    }:
        raise RuntimeError("vLLM engine policy mismatch")
    if generated.get("response_budget_mode") != (
        "min(configured_max_new_tokens, target_context - "
        "rendered_prompt_tokens - safety_margin)"
    ):
        raise RuntimeError("dynamic response budget mode mismatch")


def select_destination(
    bundle_manifest: dict[str, Any],
    checkpoint_manifest: dict[str, Any],
    inventory: dict[str, Any],
) -> dict[str, Any]:
    stage = bundle_manifest["stage"]
    matches = [item for item in inventory["checkpoints"] if item["id"] == stage]
    if len(matches) != 1:
        raise RuntimeError(f"stage is not in checkpoint_sources.json: {stage}")
    item = matches[0]
    expected = {
        "family": item["family"],
        "variant": item["variant"],
        "final_iteration": item["final_iteration"],
        "full_sft": item["full_sft"],
    }
    for key, value in expected.items():
        if checkpoint_manifest.get(key) != value:
            raise RuntimeError(
                f"checkpoint manifest {key} mismatch: "
                f"expected={value!r} found={checkpoint_manifest.get(key)!r}"
            )
    if (
        checkpoint_manifest.get("checkpoint_manifest_sha256")
        != bundle_manifest.get("checkpoint_manifest_sha256")
    ):
        raise RuntimeError("bundle and checkpoint manifest identities differ")
    return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    if not (repo_root / ".git").exists():
        raise RuntimeError(f"not a Git worktree: {repo_root}")
    manifest, summary = load_and_validate_bundle(args.bundle)
    validate_scientific_policy(args.bundle, repo_root)
    checkpoint_manifest = json.loads(
        (args.bundle / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    inventory = json.loads(
        (repo_root / HANDOFF_RELATIVE / "checkpoint_sources.json").read_text(
            encoding="utf-8"
        )
    )
    destination_spec = select_destination(
        manifest, checkpoint_manifest, inventory
    )
    destination = repo_root / destination_spec["result_dir"]
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite existing imported results: {destination}"
        )
    destination.mkdir(parents=True)
    compact = {
        key: summary[key]
        for key in (
            "bootstrap_95_ci",
            "cap_hits",
            "correct",
            "eligible_final_answers",
            "finish_reasons",
            "length",
            "parseable",
            "pass_at_1",
            "policy_sha256",
            "scorer_version",
            "stage",
            "stop_tokens",
            "think_closed",
            "total",
        )
    }
    compact["checkpoint_manifest_sha256"] = manifest[
        "checkpoint_manifest_sha256"
    ]
    compact["per_problem"] = summary["per_problem"]
    provenance = {
        "bundle_content_sha256": manifest["bundle_content_sha256"],
        "bundle_summary_sha256": manifest["summary_sha256"],
        "checkpoint_manifest_sha256": manifest[
            "checkpoint_manifest_sha256"
        ],
        "eval_policy_file_sha256": manifest["eval_policy_file_sha256"],
        "raw_problem_count": EXPECTED_ROWS,
        "raw_results_location": "external_bundle",
        "stage": manifest["stage"],
    }
    markdown = (
        f"# {manifest['stage']} MATH-500\n\n"
        f"- Pass@1: {summary['correct']}/{summary['total']} "
        f"({summary['pass_at_1']:.2%})\n"
        f"- Cap hits: {summary['cap_hits']}\n"
        f"- Scorer: `{summary['scorer_version']}`\n"
        f"- Checkpoint: `{manifest['checkpoint_manifest_sha256']}`\n"
        f"- Raw bundle: `{manifest['bundle_content_sha256']}`\n"
    )
    atomic_text(destination / "RESULTS.json", canonical_json(compact))
    atomic_text(destination / "RESULTS.md", markdown)
    atomic_text(
        destination / "PROVENANCE.json", canonical_json(provenance)
    )
    atomic_text(
        destination / "CHECKPOINT.json",
        canonical_json(checkpoint_manifest),
    )
    atomic_text(
        destination / "EVAL_POLICY.json",
        (args.bundle / "eval_policy.json").read_text(encoding="utf-8"),
    )
    print(
        f"RESULT_IMPORT_COMPLETE stage={manifest['stage']} "
        f"destination={destination}",
        flush=True,
    )


if __name__ == "__main__":
    main()
