#!/usr/bin/env python3
"""Validate the Qwen2.5-7B correct-recipe masked-SFT contract and launch surface."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

VARIANTS = (
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
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example-dir", type=Path, required=True)
    parser.add_argument("--require-source-artifacts", action="store_true")
    parser.add_argument("--require-pushed-clean", action="store_true")
    args = parser.parse_args()
    example = args.example_dir.resolve()
    repo = example.parents[1]
    contract = json.loads((example / "config/experiment_contract.json").read_text(encoding="utf-8"))
    if tuple(contract["variants"]) != VARIANTS or "correct_only" in contract["variants"]:
        raise ValueError("variant contract drift")
    if contract["selection"]["correct_rows"] != 1000 or contract["selection"]["wrong_rows"] != 1000:
        raise ValueError("balanced row-count contract drift")
    expected_training = {
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_epsilon": 1e-8,
        "bf16": True,
        "data_parallel_size": 2,
        "epochs": 4,
        "full_parameter_sft": True,
        "global_batch_size": 64,
        "gradient_clip_norm": 1.0,
        "learning_rate": 5e-6,
        "lr_decay_iters": 125,
        "lr_decay_style": "cosine",
        "max_sequence_length": 10240,
        "max_tokens_per_gpu": 16384,
        "min_learning_rate": 1e-6,
        "optimizer": "AdamW",
        "optimizer_cpu_offload": True,
        "optimizer_updates": 125,
        "rollout_batch_size": 64,
        "tensor_parallel_size": 2,
        "trace_token_cap": 8192,
        "warmup_fraction": 0.03,
        "warmup_init": 0.0,
        "weight_decay": 1e-4,
        "zero_stage": 1,
    }
    for key, value in expected_training.items():
        if contract["training"].get(key) != value:
            raise ValueError(f"training contract drift for {key}")

    env = (example / "env.sh").read_text(encoding="utf-8")
    required_env = (
        "export SFT_NUM_ROLLOUT=125",
        "export SFT_FINAL_ROLLOUT_ID=124",
        "export SFT_GLOBAL_BATCH_SIZE=64",
        "export SFT_LR=5e-6",
        "export SFT_MIN_LR=1e-6",
        "export SFT_TENSOR_MODEL_PARALLEL_SIZE=2",
        "export SFT_MAX_TOKENS_PER_GPU=16384",
        "export SFT_OPTIMIZER_CPU_OFFLOAD=1",
        "VARIANTS=(",
    )
    for value in required_env:
        if value not in env:
            raise ValueError(f"missing environment contract: {value}")
    submit = (example / "submit_training_only.sh").read_text(encoding="utf-8")
    if 'for variant in "${VARIANTS[@]}"' not in submit:
        raise ValueError("submission does not consume the canonical variant array")
    if "SFT_VARIANT=correct_only" in submit or "train_correct_only" in submit:
        raise ValueError("submission must not train correct-only")
    if '--dependency="afterok:${tail_job}"' not in submit:
        raise ValueError("submission is not strict serial afterok")
    if "DRY_RUN_COMPLETE jobs=14" not in submit:
        raise ValueError("submission job-count contract drift")

    finalize = (example / "finalize.py").read_text(encoding="utf-8")
    rsync = (example / "rsync_to_klone.sh").read_text(encoding="utf-8")
    if '"checkpoint_count": len(transfer_records)' not in finalize:
        raise ValueError("finalizer does not isolate the new transfer records")
    if '"comparison_checkpoint_count": len(comparison_targets)' not in finalize:
        raise ValueError("finalizer omits the complete comparison inventory")
    if '[[ "${#records[@]}" -eq 10 ]]' not in rsync:
        raise ValueError("rsync checkpoint-count contract drift")
    for batch_script in sorted(example.glob("*.sbatch")):
        text = batch_script.read_text(encoding="utf-8")
        if text.count("#SBATCH --time=02:00:00") != 1:
            raise ValueError(f"two-hour wall-time cap drift: {batch_script.name}")

    if args.require_source_artifacts:
        source_root = Path("/gpfs/scrubbed/suryadv/slime-qwen-correct-only-openr1-sft-2k/" "v1/3568736a3743c319")
        pinned = {
            source_root
            / "data/selected/8k_shared.jsonl": contract["selection"]["source_correct_only_selected_sha256"],
            source_root
            / "data/datasets/qwen2_5_7b_8k.jsonl": contract["selection"]["source_correct_only_dataset_sha256"],
            source_root
            / "data/selection_and_tokenization_stats.json": contract["selection"][
                "source_correct_only_prep_stats_sha256"
            ],
            source_root
            / "handoff/checkpoints/base_qwen2_5_7b.json": contract["source_baselines"]["base_manifest_sha256"],
            source_root
            / "handoff/checkpoints/qwen2_5_7b_8k.json": contract["source_baselines"]["correct_only_manifest_sha256"],
            source_root / "TRAINING_STATUS.json": contract["source_baselines"]["correct_only_status_sha256"],
        }
        for path, expected in pinned.items():
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f"source artifact hash drift: {path}")

    if args.require_pushed_clean:
        branch = subprocess.check_output(["git", "-C", str(repo), "branch", "--show-current"], text=True).strip()
        if branch != "qwen-correct-only-openr1-sft-2k":
            raise ValueError(f"unexpected submission branch: {branch}")
        if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip():
            raise ValueError("repository is not clean")
        local = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        upstream = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "@{upstream}"], text=True).strip()
        if local != upstream:
            raise ValueError("submission commit is not pushed")
    print(
        "CORRECT_RECIPE_MASKED_CONFIG_VALID " "model=qwen2.5-7b runs=10 rows=2000 updates=125",
        flush=True,
    )


if __name__ == "__main__":
    main()
