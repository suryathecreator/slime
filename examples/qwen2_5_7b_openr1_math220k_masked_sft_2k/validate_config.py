#!/usr/bin/env python3
"""Fail-closed repository and artifact preflight for the Qwen2.5 suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

BRANCH = "qwen2.5-7b-openr1-masked-sft-2k"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example-dir", type=Path, required=True)
    parser.add_argument("--require-pushed-clean", action="store_true")
    args = parser.parse_args()
    example = args.example_dir.resolve()
    repo = example.parents[1]
    contract = json.loads((example / "config/experiment_contract.json").read_text())
    expected_variants = [
        "correct_only", "unmasked", "random_mask_70", "inverse_tau_0p20",
        "inverse_tau_0p05", "random_mask_25", "random_mask_50",
        "random_mask_80", "random_mask_90", "margin_mask", "prob_ratio_mask",
    ]
    if contract["variants"] != expected_variants:
        raise ValueError("variant order drift")
    base = contract["base_model"]
    if base != {
        "hf_repo": "Qwen/Qwen2.5-7B",
        "revision": "d149729398750b98c0af14eb82c78cfe92750796",
        "tokenizer_files": {
            "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
            "tokenizer_config.json": "c91efca15ceff6e9ee9424db58a6f59cd41294e550a86cbd07e3c1fb500b34f9",
        },
    }:
        raise ValueError("base model pin drift")
    if contract["supervision"] != {
        "assistant_added_token_ids": "all tokenizer.get_added_vocab values",
        "embedded_chat_boundaries": "fail_closed",
        "im_end": "exact_weight_1",
        "no_synthetic_endoftext": True,
        "padding_outside_logical_sequence": "weight_0",
        "prompt_tokens": "context_only_outside_response_loss",
        "protected_control_markup": ["<think>", "</think>"],
        "protected_positions": "excluded_from_every_masking_and_downweighting_decision",
        "template_newline_after_im_end": "weight_0",
    }:
        raise ValueError("protected supervision contract drift")
    source_root = Path(
        "/gpfs/scrubbed/suryadv/repos/axolotl-masked-sft/runs/"
        "2026-06-23_openr1_220k_2k_balanced_full_sft_axolotl_harp_vllm_prob_ratio_mask/"
        "unpaired/seed_42/datasets"
    )
    selection = contract["selection"]
    source_hashes = {
        source_root / "correct_only.jsonl": selection["source_correct_only_sha256"],
        source_root / "correct_wrong_unmasked.jsonl": selection["source_correct_wrong_unmasked_sha256"],
        source_root / "stats.json": selection["source_stats_sha256"],
    }
    for path, expected in source_hashes.items():
        if sha256(path) != expected:
            raise ValueError(f"frozen Axolotl source hash mismatch: {path}")
    axolotl_repo = Path(
        "/gpfs/scrubbed/suryadv/repos/axolotl-masked-sft/repos/Axolotl-Masked-SFT"
    )
    if git(axolotl_repo, "rev-parse", "HEAD") != selection["source_repository_commit"]:
        raise ValueError("Axolotl source commit drift")
    required_source = {
        repo / "scripts/models/qwen2.5-7B.sh": "--num-layers 28",
        repo / "examples/qwen3_8b_opd_tillicum/04_run_sft_100k_8xh200.sbatch": "SFT_MODEL_ARGS_SCRIPT",
        repo / "examples/qwen3_8b_opd_tillicum/container_exec.sh": "SFT_APPLY_CHAT_TEMPLATE_KWARGS_JSON",
        example / "build_variants.py": "if index in protected:",
        example / "weighted_sft_rollout.py": "weights[index] != 1.0",
    }
    for path, needle in required_source.items():
        if needle not in path.read_text(encoding="utf-8"):
            raise ValueError(f"required implementation missing in {path}")
    for script in sorted(example.glob("*.sbatch")) + sorted(example.glob("*.sh")):
        subprocess.run(["bash", "-n", str(script)], check=True)
    if args.require_pushed_clean:
        if git(repo, "branch", "--show-current") != BRANCH:
            raise ValueError(f"submission must run from {BRANCH}")
        if git(repo, "status", "--porcelain"):
            raise ValueError("submission requires a clean worktree")
        upstream = git(repo, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
        if git(repo, "rev-parse", "HEAD") != git(repo, "rev-parse", upstream):
            raise ValueError("local HEAD is not the pushed upstream commit")
    print("CONFIG_QWEN25_2K_VALID variants=11 full_sft=1 protected_controls=fail_closed", flush=True)


if __name__ == "__main__":
    main()
