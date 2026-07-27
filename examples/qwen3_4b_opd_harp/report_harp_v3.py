#!/usr/bin/env python3
"""Write a compact, reproducible HARP V2-versus-V3 correction report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-dir", required=True)
    parser.add_argument("--v3-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    v2_dir = Path(args.v2_dir)
    v3_dir = Path(args.v3_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    v2_metrics_path = v2_dir / "metrics.json"
    v3_metrics_path = v3_dir / "metrics.json"
    if not v3_metrics_path.is_file():
        raise RuntimeError("V3 has no authoritative metrics.json; inspect metrics_blocked.json and needs_review.jsonl")
    v2_metrics = read_json(v2_metrics_path)
    v3_metrics = read_json(v3_metrics_path)
    if v2_metrics.get("scorer_version") != "harp_answer_v2" or v3_metrics.get("scorer_version") != "harp_answer_v3":
        raise ValueError("Correction report received the wrong scorer versions")
    if v2_metrics.get("data_sha256") != v3_metrics.get("data_sha256"):
        raise ValueError("V2 and V3 do not use the same HARP source split")
    v3_rows = read_jsonl(v3_dir / "predictions.jsonl")
    official_positive_rejected = sum(
        row.get("grade") == "incorrect"
        and bool((((row.get("verifier_evidence") or {}).get("official_checker") or {}).get("output") or {}).get("is_correct"))
        for row in v3_rows
        if not row.get("excluded")
    )
    audit = read_json(v3_dir / "rescore_audit.json")
    payload = {
        "title": "Qwen3-4B HARP-500 scorer correction: V2 versus V3",
        "same_generation_artifact": True,
        "data_sha256": v3_metrics["data_sha256"],
        "source_predictions_sha256": v3_metrics["source_predictions_sha256"],
        "v2_metrics_sha256": sha256_file(v2_metrics_path),
        "v3_metrics_sha256": sha256_file(v3_metrics_path),
        "v2": v2_metrics,
        "v3": v3_metrics,
        "accuracy_delta": v3_metrics["accuracy"] - v2_metrics["accuracy"],
        "correct_count_delta": v3_metrics["correct_count"] - v2_metrics["correct_count"],
        "correctness_flips": audit["correctness_flips"],
        "official_positive_typed_rejections": official_positive_rejected,
    }
    write_json(output_dir / "report.json", payload)

    lines = [
        "# Qwen3-4B HARP-500 scorer correction: V2 versus V3",
        "",
        "Both rows score the same stored 500 greedy generations; only answer extraction and verification changed.",
        "",
        "| Scorer | Correct | Accuracy | Cap hits | Cap-hit accuracy | Reviews |",
        "|---|---:|---:|---:|---:|---:|",
        f"| `harp_answer_v2` | {v2_metrics['correct_count']}/500 | {v2_metrics['accuracy']:.3%} | {v2_metrics['cap_hit_count']} | {v2_metrics['cap_hit_accuracy']:.3%} | n/a |",
        f"| `harp_answer_v3` | {v3_metrics['correct_count']}/{v3_metrics['included_count']} | {v3_metrics['accuracy']:.3%} | {v3_metrics['cap_hit_count']} | {v3_metrics['cap_hit_accuracy']:.3%} | {v3_metrics['review_count']} |",
        "",
        f"Correction: {payload['correct_count_delta']:+d} answers, {payload['accuracy_delta']:+.3%} absolute.",
        f"Correctness flips: {len(payload['correctness_flips'])}. Official-checker positives rejected by typed verification: {official_positive_rejected}.",
        "",
        f"- HARP split SHA-256: `{payload['data_sha256']}`",
        f"- Stored generation SHA-256: `{payload['source_predictions_sha256']}`",
        f"- V2 metrics SHA-256: `{payload['v2_metrics_sha256']}`",
        f"- V3 metrics SHA-256: `{payload['v3_metrics_sha256']}`",
        f"- Clean split SHA-256: `{v3_metrics['clean_split_sha256']}` ({v3_metrics['source_count']} source, {v3_metrics['excluded_count']} excluded)",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
