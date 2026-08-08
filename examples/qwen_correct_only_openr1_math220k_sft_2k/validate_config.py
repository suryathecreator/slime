#!/usr/bin/env python3
"""Fail-closed static and Git preflight for the correct-only suite."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from examples.qwen_correct_only_openr1_math220k_sft_2k.experiment import (
    MODEL_ORDER,
    RUN_ORDER,
)


BRANCH = "qwen-correct-only-openr1-sft-2k"


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
    contract = json.loads((example / "config/experiment_contract.json").read_text(encoding="utf-8"))
    if tuple(contract["models"]) != MODEL_ORDER:
        raise ValueError("model order drift")
    if tuple(contract["runs"]) != (
        "qwen2_5_3b_16k",
        "qwen2_5_3b_8k",
        "qwen2_5_7b_8k",
        "qwen3_4b_8k",
        "qwen3_8b_8k",
    ):
        raise ValueError("canonical JSON run order drift")
    if set(contract["runs"]) != set(RUN_ORDER):
        raise ValueError("run inventory drift")
    if contract["variants"] != ["base", "correct_only"]:
        raise ValueError("only base and correct_only are allowed")
    expected_training = {
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_epsilon": 1e-8,
        "bf16": True,
        "context_parallel_size": 1,
        "epochs": 4,
        "full_parameter_sft": True,
        "global_batch_size": 64,
        "gradient_clip_norm": 1.0,
        "learning_rate": 5e-6,
        "lr_decay_iters": 125,
        "lr_decay_style": "cosine",
        "min_learning_rate": 1e-6,
        "optimizer": "AdamW",
        "optimizer_updates": 125,
        "pipeline_parallel_size": 1,
        "recompute_loss_function": True,
        "rollout_batch_size": 64,
        "warmup_fraction": 0.03,
        "warmup_init": 0.0,
        "weight_decay": 1e-4,
        "zero_stage": 1,
    }
    if contract["training"] != expected_training:
        raise ValueError("training hyperparameter contract drift")
    shared_runs = [key for key in RUN_ORDER if key.endswith("_8k")]
    if any(contract["runs"][key]["selection"] != "8k_shared" for key in shared_runs):
        raise ValueError("8K runs do not share one selection")
    if contract["selection"]["shared_8k_policy"] != (
        "same ordered 2000 trace IDs and verbatim texts for every 8K model, "
        "sampled from the cross-tokenizer eligibility intersection"
    ):
        raise ValueError("shared 8K comparison policy drift")
    if contract["runs"]["qwen2_5_3b_16k"]["max_sequence_length"] != 18432:
        raise ValueError("16K training context drift")
    expected_runtime = {
        "qwen2_5_3b_8k": "1:16384:0",
        "qwen2_5_3b_16k": "1:16384:0",
        "qwen2_5_7b_8k": "2:16384:1",
        "qwen3_4b_8k": "1:16384:0",
        "qwen3_8b_8k": "2:16384:1",
    }
    env_command = (
        'source "$1"; '
        "for run_key in qwen2_5_3b_8k qwen2_5_3b_16k qwen2_5_7b_8k "
        "qwen3_4b_8k qwen3_8b_8k; do "
        'resolve_run "${run_key}"; '
        "printf '%s=%s:%s:%s\\n' \"${run_key}\" "
        '"${SFT_TENSOR_MODEL_PARALLEL_SIZE}" "${SFT_MAX_TOKENS_PER_GPU}" '
        '"${SFT_OPTIMIZER_CPU_OFFLOAD}"; done; '
        "printf 'schedule_audit_dir=%s\\n' \"${SCHEDULE_AUDIT_DIR}\""
    )
    resolved = dict(
        line.split("=", 1)
        for line in subprocess.check_output(
            ["bash", "-c", env_command, "bash", str(example / "env.sh")], text=True
        ).splitlines()
    )
    if not resolved.pop("schedule_audit_dir", "").endswith(
        "/schedule_audits/memory_r1"
    ):
        raise ValueError("schedule-audit memory profile drift")
    if resolved != expected_runtime:
        raise ValueError(f"runtime memory profile drift: {resolved}")
    for model_key, script in {
        "qwen2_5_3b": "scripts/models/qwen2.5-3B.sh",
        "qwen2_5_7b": "scripts/models/qwen2.5-7B.sh",
        "qwen3_4b": "scripts/models/qwen3-4B.sh",
        "qwen3_8b": "scripts/models/qwen3-8B.sh",
    }.items():
        if contract["models"][model_key]["model_args_script"] != script or not (repo / script).is_file():
            raise ValueError(f"model argument script drift for {model_key}")
    require_text(
        repo / "examples/qwen3_8b_opd_tillicum/04_run_sft_100k_8xh200.sbatch",
        (
            '--lr-decay-style "${SFT_LR_DECAY_STYLE}"',
            '--lr-warmup-fraction "${SFT_LR_WARMUP_FRACTION}"',
            '--lr-decay-iters "${SFT_LR_DECAY_ITERS}"',
            '--adam-eps "${SFT_ADAM_EPS}"',
            'SFT_ARGS+=(--start-rollout-id 0)',
        ),
    )
    require_text(
        repo / "slime/backends/megatron_utils/arguments.py",
        ("args.bf16 = not args.fp16",),
    )
    require_text(
        example / "correct_only_sft_rollout.py",
        (
            "weights[-2:] != [1.0, 0.0]",
            "any(value != 1.0 for value in weights[:assistant_count])",
        ),
    )
    require_text(
        example / "canary_generate.py",
        ('"policy": "diagnostic_only_does_not_block_training"',),
    )
    require_text(
        example / "02_canary.sbatch",
        ('--tensor-parallel-size "${SFT_TENSOR_MODEL_PARALLEL_SIZE}"',),
    )
    require_text(
        example / "02b_resume_canary_audit.sbatch",
        (
            'readonly CANARY_INPUT_ROOT="${OUTPUT_ROOT}/canaries/${RUN_KEY}"',
            'readonly AUDIT_ROOT="${CANARY_INPUT_ROOT}/attempts/${CANARY_AUDIT_ATTEMPT}"',
            '--tensor-parallel-size "${SFT_TENSOR_MODEL_PARALLEL_SIZE}"',
            'echo "CANARY_AUDIT_RESUME_COMPLETE',
        ),
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
        "CORRECT_ONLY_CONFIG_VALID models=4 runs=5 variants=base,correct_only "
        "gbs=64 epochs=4 updates=125 shared_8k_rows=2000",
        flush=True,
    )


if __name__ == "__main__":
    main()
