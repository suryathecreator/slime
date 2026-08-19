#!/usr/bin/env python3
"""Fail-closed static, source-artifact, base-model, and pushed-revision preflight."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from examples.qwen2_5_7b_aime_2009_2024_split_sft_6k.data_utils import VARIANTS, sha256_file


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def validate_contract(contract: dict[str, Any]) -> None:
    require(
        contract.get("family") == "qwen2_5_7b_aime_2009_2024_split_sft_6k",
        "experiment family drift",
    )
    require(tuple(contract.get("variants", ())) == VARIANTS, "variant order drift")
    training = contract["training"]
    expected = {
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_epsilon": 1e-8,
        "bf16": True,
        "context_parallel_size": 1,
        "data_parallel_size": 1,
        "effective_presentations": 12032,
        "epochs": 2,
        "final_iteration": 187,
        "global_batch_size": 64,
        "gradient_clip_norm": 1.0,
        "learning_rate": 5e-6,
        "log_probs_chunk_size": 2048,
        "lr_decay_style": "cosine",
        "max_sequence_length": 32896,
        "max_tokens_per_gpu": 16384,
        "min_learning_rate": 1e-6,
        "optimizer": "AdamW",
        "optimizer_checkpoint_interval_updates": 24,
        "optimizer_updates": 188,
        "pipeline_parallel_size": 1,
        "rollout_batch_size": 64,
        "tensor_parallel_size": 4,
        "warmup_fraction": 0.03,
        "warmup_init": 0.0,
        "weight_decay": 1e-4,
        "zero_stage": 0,
    }
    for key, expected_value in expected.items():
        require(training.get(key) == expected_value, f"training contract drift: {key}")
    require(training.get("full_parameter_sft") is True, "full-SFT contract drift")
    require("synchronous train.py" in training["train_entrypoint"], "entrypoint drift")
    selection = contract["selection"]
    require(selection["training_rows_per_variant"] == 6000, "6K expansion drift")
    require(selection["seed"] == 42, "selection seed drift")
    require(selection["split_problem_count"] == 106, "mixed split problem-count drift")
    require(selection["common_source_trace_count"] == 508, "common source trace-count drift")
    require("No source trace is filtered or truncated" in selection["length_policy"], "length policy drift")
    require("exact same problem-ID set" in selection["correct_control"], "matched-control policy drift")
    require("only the 212 problems" in selection["mixed_outcome_policy"], "mixed-only policy drift")
    audit = selection["realized_source_audit"]
    require(audit["maximum_fully_rendered_tokens"] == 32774, "realized maximum-length audit drift")
    for split, correct_pool, incorrect_pool in (("held_in", 1188, 508), ("held_out", 1105, 591)):
        require(audit[split]["correct_rows"] == 508, f"{split} correct-row audit drift")
        require(audit[split]["incorrect_rows"] == 508, f"{split} incorrect-row audit drift")
        require(audit[split]["correct_problem_count"] == 106, f"{split} correct-problem audit drift")
        require(audit[split]["incorrect_problem_count"] == 106, f"{split} incorrect-problem audit drift")
        require(audit[split]["source_correct_pool_rows"] == correct_pool, f"{split} correct-pool drift")
        require(audit[split]["source_incorrect_pool_rows"] == incorrect_pool, f"{split} incorrect-pool drift")
        require(
            audit[split]["correct_problem_ids_sha256"] == audit[split]["incorrect_problem_ids_sha256"],
            f"{split} exact problem identity drift",
        )
    require(set(audit["dataset_sha256"]) == set(VARIANTS), "realized dataset hashes missing")
    require(set(audit["eval_sha256"]) == {"held_in", "held_out"}, "realized eval hashes missing")
    require(contract["formatting"]["system"] is None, "training system-message drift")
    require(contract["formatting"]["uses_apply_chat_template"] is False, "manual prompt-format drift")
    source = contract["source_dataset"]
    require(source["revision"] == "c34143c92856d9b90fdb9c808c7e66070dcab9a1", "source revision drift")
    require(source["rows"] == 7680 and source["problem_count"] == 480, "source shape drift")
    require(len(source["files"]) == 16, "source file inventory drift")
    require(contract["submission"]["job_count"] == 10, "job-count contract drift")


def validate_static(package: Path, repo_root: Path) -> None:
    required = (
        "README.md",
        "env.sh",
        "config/experiment_contract.json",
        "prepare_data.py",
        "validate_data.py",
        "validate_config.py",
        "finalize.py",
        "01_prepare_data.sbatch",
        "02_canary.sbatch",
        "03_train.sbatch",
        "05_finalize.sbatch",
        "submit_training_only.sh",
        "rsync_to_klone.sh",
    )
    for relative in required:
        require((package / relative).is_file(), f"missing package file: {relative}")
    env = (package / "env.sh").read_text(encoding="utf-8")
    for needle in (
        "SFT_TRAIN_ENTRYPOINT=train.py",
        "SFT_SEQ_LENGTH=32896",
        "SFT_TENSOR_MODEL_PARALLEL_SIZE=4",
        "SFT_GLOBAL_BATCH_SIZE=64",
        'SFT_NUM_ROLLOUT="${AIME_SPLIT_6K_OPTIMIZER_UPDATES}"',
        'SFT_FINAL_ROLLOUT_ID="${AIME_SPLIT_6K_FINAL_ITERATION}"',
        "SFT_LR=5e-6",
        "SFT_MIN_LR=1e-6",
        "SFT_SAVE_INTERVAL=24",
        'SFT_MANIFEST_FAMILY="aime_2009_2024_split"',
        "unset SFT_INITIAL_HF_DIR",
    ):
        require(needle in env, f"env contract missing: {needle}")
    for relative, walltime, gpu in (
        ("01_prepare_data.sbatch", "02:00:00", "h200:1"),
        ("02_canary.sbatch", "02:00:00", "h200:4"),
        ("03_train.sbatch", "06:00:00", "h200:4"),
        ("05_finalize.sbatch", "02:00:00", "h200:1"),
    ):
        text = (package / relative).read_text(encoding="utf-8")
        require(f"#SBATCH --time={walltime}" in text, f"walltime drift: {relative}")
        require(f"#SBATCH --gres=gpu:{gpu}" in text, f"GPU request drift: {relative}")
    shared_train = (repo_root / "examples/qwen2_5_7b_aime_generalization_incorrect_sft/03_train.sbatch").read_text(
        encoding="utf-8"
    )
    require("SFT_EXPERIMENT_ENV_FILE" in shared_train, "shared train wrapper lacks env interface")
    require("SFT_MANIFEST_FAMILY" in shared_train, "shared train wrapper lacks family interface")
    shared_canary = (
        repo_root / "examples/qwen2_5_7b_aime_generalization_incorrect_sft/02_stage1_canary.sbatch"
    ).read_text(encoding="utf-8")
    require("SFT_CANARY_VARIANT" in shared_canary, "shared canary lacks variant interface")
    manifest_tool = (
        repo_root / "examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff/checkpoint_manifest.py"
    ).read_text(encoding="utf-8")
    require("aime_2009_2024_split" in manifest_tool, "checkpoint family missing")
    submitter = (package / "submit_training_only.sh").read_text(encoding="utf-8")
    require("git worktree add --detach" in submitter, "immutable execution worktree missing")
    require('--chdir="${EXECUTION_WORKTREE}"' in submitter, "Slurm worktree chdir missing")


def validate_sources(contract: dict[str, Any]) -> None:
    source_root = Path(
        "/gpfs/scrubbed/suryadv/datasets/suryadv_qwen3-8b-aime-2009-2024-16x/"
        "c34143c92856d9b90fdb9c808c7e66070dcab9a1/data"
    )
    for filename, expected in contract["source_dataset"]["files"].items():
        path = source_root / filename
        require(path.is_file(), f"missing staged source: {path}")
        require(sha256_file(path) == expected, f"source hash drift: {filename}")
    model_root = Path("/gpfs/scrubbed/suryadv/slime-qwen-correct-only-openr1-sft-2k/v1/3568736a3743c319")
    model = model_root / "models/Qwen2.5-7B"
    for filename, expected in contract["base_model"]["tokenizer_files"].items():
        require(sha256_file(model / filename) == expected, f"base tokenizer drift: {filename}")
    model_config = load_json(model / "config.json")
    tensor_parallel_size = int(contract["training"]["tensor_parallel_size"])
    for field in ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads", "vocab_size"):
        require(
            int(model_config[field]) % tensor_parallel_size == 0,
            f"base model {field} is not divisible by TP={tensor_parallel_size}",
        )
    torch_dist = model_root / "models/Qwen2.5-7B_torch_dist"
    require(
        (torch_dist / "latest_checkpointed_iteration.txt").read_text().strip() == "release",
        "base torch_dist tracker drift",
    )
    require((torch_dist / "release/.metadata").is_file(), "base torch_dist incomplete")
    base_manifest = model_root / "handoff/checkpoints/base_qwen2_5_7b.json"
    require(
        load_json(base_manifest)["checkpoint_manifest_sha256"] == contract["base_model"]["checkpoint_identity"],
        "base manifest identity drift",
    )


def validate_pushed_clean(repo_root: Path) -> None:
    require(not command("git", "-C", str(repo_root), "status", "--porcelain"), "Git worktree is not clean")
    head = command("git", "-C", str(repo_root), "rev-parse", "HEAD")
    upstream = command("git", "-C", str(repo_root), "rev-parse", "@{upstream}")
    require(head == upstream, f"HEAD is not synchronized with upstream: {head} != {upstream}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--require-source-artifacts", action="store_true")
    parser.add_argument("--require-pushed-clean", action="store_true")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    contract = load_json(args.contract)
    package = args.contract.resolve().parent.parent
    validate_contract(contract)
    validate_static(package, repo_root)
    if args.require_source_artifacts:
        validate_sources(contract)
    if args.require_pushed_clean:
        validate_pushed_clean(repo_root)
    print(
        "AIME_SPLIT_6K_CONFIG_VALID "
        f"sources={int(args.require_source_artifacts)} pushed={int(args.require_pushed_clean)}"
    )


if __name__ == "__main__":
    main()
