#!/usr/bin/env python3
"""Fail-closed repository and source preflight for the Qwen3-0.6B suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


BRANCH = "qwen3-0.6b-openr1-masked-sft-2k"
EXPECTED_VARIANTS = [
    "correct_only",
    "unmasked",
    "random_mask_70",
    "inverse_tau_0p20",
    "inverse_tau_0p05",
    "random_mask_25",
    "random_mask_50",
    "random_mask_80",
    "random_mask_90",
    "margin_mask",
    "prob_ratio_mask",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def require_text(path: Path, needles: tuple[str, ...]) -> None:
    text = path.read_text(encoding="utf-8")
    for needle in needles:
        if needle not in text:
            raise ValueError(f"required implementation missing in {path}: {needle}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example-dir", type=Path, required=True)
    parser.add_argument("--require-pushed-clean", action="store_true")
    args = parser.parse_args()
    example = args.example_dir.resolve()
    repo = example.parents[1]
    contract = json.loads((example / "config/experiment_contract.json").read_text())
    if contract["variants"] != EXPECTED_VARIANTS:
        raise ValueError("variant order drift")
    if contract["base_model"] != {
        "hf_repo": "Qwen/Qwen3-0.6B-Base",
        "revision": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
    }:
        raise ValueError("base model pin drift")
    if contract["training"] != {
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_epsilon": 1e-8,
        "bf16": True,
        "context_length": 32768,
        "data_parallel_size": 4,
        "effective_gradient_accumulation_steps": 1,
        "epochs": 1,
        "full_parameter_sft": True,
        "global_batch_size": 16,
        "gradient_clip_norm": 1.0,
        "learning_rate": 1e-5,
        "lr_decay_style": "cosine",
        "max_tokens_per_gpu": 32768,
        "min_learning_rate": 0.0,
        "optimizer": "AdamW",
        "optimizer_updates": 125,
        "rows": 2000,
        "tensor_parallel_size": 1,
        "warmup_fraction": 0.05,
        "warmup_style": "linear_from_zero",
        "weight_decay": 1e-4,
    }:
        raise ValueError("training hyperparameter contract drift")
    source_root = Path(
        "/gpfs/scrubbed/suryadv/repos/axolotl-masked-sft/runs/"
        "2026-06-23_openr1_220k_2k_balanced_full_sft_axolotl_harp_vllm_prob_ratio_mask/"
        "unpaired/seed_42/datasets"
    )
    selection = contract["selection"]
    for path, expected in {
        source_root / "correct_only.jsonl": selection["source_correct_only_sha256"],
        source_root / "correct_wrong_unmasked.jsonl": selection["source_correct_wrong_unmasked_sha256"],
        source_root / "stats.json": selection["source_stats_sha256"],
    }.items():
        if sha256(path) != expected:
            raise ValueError(f"frozen Axolotl source hash mismatch: {path}")
    axolotl_repo = Path(
        "/gpfs/scrubbed/suryadv/repos/axolotl-masked-sft/repos/Axolotl-Masked-SFT"
    )
    if git(axolotl_repo, "rev-parse", "HEAD") != selection["source_repository_commit"]:
        raise ValueError("Axolotl source commit drift")

    require_text(repo / "scripts/models/qwen3-0.6B.sh", ("--num-layers 28", "--hidden-size 1024"))
    require_text(
        repo / "examples/qwen3_8b_opd_tillicum/04_run_sft_100k_8xh200.sbatch",
        (
            '--lr-decay-style "${SFT_LR_DECAY_STYLE}"',
            '--lr-warmup-fraction "${SFT_LR_WARMUP_FRACTION}"',
            '--lr-decay-iters "${SFT_LR_DECAY_ITERS}"',
            '--weight-decay "${SFT_WEIGHT_DECAY}"',
            '--adam-eps "${SFT_ADAM_EPS}"',
            'SFT_ARGS+=(--start-rollout-id 0)',
        ),
    )
    require_text(
        repo / "examples/qwen3_8b_opd_tillicum/container_exec.sh",
        ("SFT_LR_WARMUP_FRACTION", "SFT_ADAM_EPS", "SFT_WEIGHT_DECAY"),
    )
    require_text(
        repo / "slime/backends/megatron_utils/arguments.py",
        ("args.bf16 = not args.fp16",),
    )
    require_text(
        example / "audit_runtime_batches.py",
        ("shifted_weights = [0.0] * (prompt_length - 1) + weights + [0.0]",),
    )
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
    print(
        "CONFIG_QWEN3_0P6B_2K_VALID variants=11 full_sft=1 gbs=16 grad_accum=1 seq=32768 updates=125",
        flush=True,
    )


if __name__ == "__main__":
    main()
