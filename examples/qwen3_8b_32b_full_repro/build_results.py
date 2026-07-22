#!/usr/bin/env python3
"""Build immutable scratch result and stopping-audit summaries after final eval."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


STAGES = ("qwen3_8b_base", "qwen3_32b_teacher", "sft_200000", "opd_100pct")
CAP_PROXY_STAGES = {
    "qwen3_8b_base": "qwen3_8b_base_32k_proxy",
    "qwen3_32b_teacher": "qwen3_32b_teacher_32k_proxy",
    "sft_200000": "sft_200000_32k_proxy",
    "opd_100pct": "opd_100pct_32k_proxy",
}


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temp = Path(handle.name)
    temp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_stage(root: Path, stage: str) -> dict[str, Any]:
    path = root / stage / "summary.json"
    if not path.is_file():
        raise SystemExit(f"Missing required final evaluation: {path}")
    value = json.loads(path.read_text())
    value["summary_path"] = str(path)
    value["summary_sha256"] = sha256_file(path)
    return value


def stopping_audit(root: Path) -> dict[str, Any]:
    rollout_audit = Path(os.environ.get("OPD_ROLLOUT_AUDIT_JSON", ""))
    if rollout_audit.is_file():
        audit = json.loads(rollout_audit.read_text())
        audit["status"] = "complete"
        audit["source"] = str(rollout_audit)
        atomic_text(root / "STOPPING_OBJECTIVE_AUDIT.json", json.dumps(audit, indent=2, sort_keys=True) + "\n")
        source_md = Path(os.environ["OPD_ROLLOUT_AUDIT_MD"])
        atomic_text(root / "STOPPING_OBJECTIVE_AUDIT.md", source_md.read_text())
        return audit
    sanity_dir = Path(os.environ["OPD_SANITY_REPORT_DIR"])
    reports = [json.loads(path.read_text()) for path in sorted(sanity_dir.glob("rollout_*.json"))]
    if not reports:
        return {"status": "missing", "reports": 0, "note": "No OPD sanity windows were found."}
    weighted_n = sum(int(report.get("n", 0)) for report in reports)
    weighted = lambda key: sum(float(report.get(key, 0)) * int(report.get("n", 0)) for report in reports) / max(1, weighted_n)
    audit = {
        "status": "complete",
        "reports": len(reports),
        "samples": weighted_n,
        "mean_response_tokens": weighted("avg_response_tokens"),
        "max_response_tokens": max(int(report.get("max_response_tokens", 0)) for report in reports),
        "cap_hit_rate": weighted("cap_hit_rate"),
        "completed_rate": weighted("completed_rate"),
        "eligible_final_answer_rate": weighted("final_answer_rate"),
        "windows_with_threshold_violations": sum(bool(report.get("violations")) for report in reports),
        "behavior_thresholds_were_fatal": False,
        "interpretation": "The 16K cutoff bounds compute and does not teach stopping; capped trajectories were retained in the faithful loss.",
    }
    atomic_text(root / "STOPPING_OBJECTIVE_AUDIT.json", json.dumps(audit, indent=2, sort_keys=True) + "\n")
    atomic_text(
        root / "STOPPING_OBJECTIVE_AUDIT.md",
        "# Stopping-objective audit\n\n"
        f"- Rollout windows: {audit['reports']}\n"
        f"- Trajectories observed: {audit['samples']}\n"
        f"- Mean/max response tokens: {audit['mean_response_tokens']:.1f}/{audit['max_response_tokens']}\n"
        f"- 16K cap-hit rate: {audit['cap_hit_rate']:.2%}\n"
        f"- Completed rate: {audit['completed_rate']:.2%}\n"
        f"- Eligible final-answer rate: {audit['eligible_final_answer_rate']:.2%}\n"
        f"- Windows with diagnostic threshold violations: {audit['windows_with_threshold_violations']}\n\n"
        "The 16K cutoff bounded compute but was not interpreted as a stopping target. Capped trajectories stayed in the faithful OPD loss, and behavior thresholds were nonfatal.\n",
    )
    return audit


def main() -> None:
    eval_root = Path(os.environ["EVAL_OUTPUT_ROOT"])
    report_root = Path(os.environ["OUTPUT_ROOT"]) / "final_report"
    values = {stage: load_stage(eval_root, stage) for stage in STAGES}
    proxy_values = {
        source_stage: load_stage(eval_root, proxy_stage)
        for source_stage, proxy_stage in CAP_PROXY_STAGES.items()
    }
    sft = values["sft_200000"]
    opd = values["opd_100pct"]
    sft_points = 100 * float(sft["pass_at_1"])
    opd_points = 100 * float(opd["pass_at_1"])
    criteria = {
        "sft_target": 76.0,
        "opd_target": 94.0,
        "target_delta": 18.0,
        "sft_within_3_points": abs(sft_points - 76.0) <= 3.0,
        "opd_within_3_points": abs(opd_points - 94.0) <= 3.0,
        "opd_gain_at_least_12_points": opd_points - sft_points >= 12.0,
        "observed_gain_points": opd_points - sft_points,
    }
    audit = stopping_audit(report_root)
    result = {
        "schema_version": 1,
        "contract_sha256": sha256_file(Path(os.environ["CONTRACT_FILE"])),
        "evaluations": values,
        "cap_hit_32k_context_proxies": proxy_values,
        "reproduction_criteria": criteria,
        "stopping_audit": audit,
    }
    atomic_text(report_root / "results.json", json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    lines = ["# Qwen3 8B -> 32B reproduction results", ""]
    for stage in STAGES:
        value = values[stage]
        lines.append(f"- {stage}: {value['correct']}/500 ({100 * value['pass_at_1']:.2f}%)")
    lines.extend(["", "## Cap-hit-only 32K total-context diagnostics", ""])
    for source_stage, proxy_stage in CAP_PROXY_STAGES.items():
        value = proxy_values[source_stage]
        gate = value["exactness_gate"]
        lines.append(
            f"- {source_stage}: {value['correct']}/500 ({100 * value['pass_at_1']:.2f}%), "
            f"reran {gate['source_cap_hits_selected']} cap hits with "
            f"{gate['prefixes_exact']}/{gate['prefixes_checked']} exact saved prefixes"
        )
    lines.extend(
        [
            "",
            f"Observed OPD gain: {criteria['observed_gain_points']:.2f} points. Public targets were 76% SFT and 94% OPD; exact observed values are reported without seed or scorer selection.",
            "",
            "Primary reproduction criteria use only the fixed-16K-output evaluations above. The diagnostics reuse natural stops and rerun only cap-hit prompts with a dynamic response budget up to 32K total context; they are exact proxies only because every saved 16K token prefix is required to match.",
        ]
    )
    atomic_text(report_root / "RESULTS.md", "\n".join(lines) + "\n")
    print(json.dumps({"report_root": str(report_root), "criteria": criteria}, indent=2))


if __name__ == "__main__":
    main()
