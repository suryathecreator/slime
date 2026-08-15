#!/usr/bin/env python3
"""Fail-closed static, source-artifact, and pushed-revision preflight."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from examples.qwen2_5_7b_nous_aime_mixed_outcome_sft.data_utils import (
    VARIANTS,
    sha256_file,
)


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
        contract.get("family") == "qwen2_5_7b_nous_aime_mixed_outcome_sft",
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
        "effective_presentations": 3008,
        "epochs": 1,
        "final_iteration": 46,
        "global_batch_size": 64,
        "gradient_clip_norm": 1.0,
        "learning_rate": 5e-6,
        "log_probs_chunk_size": 2048,
        "lr_decay_style": "cosine",
        "max_sequence_length": 32768,
        "max_tokens_per_gpu": 16384,
        "min_learning_rate": 1e-6,
        "optimizer": "AdamW",
        "optimizer_checkpoint_interval_updates": 24,
        "optimizer_updates": 47,
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
    require("synchronous train.py" in training["train_entrypoint"], "entrypoint drift")
    selection = contract["selection"]
    require(selection["training_rows_per_variant"] == 3000, "3K expansion drift")
    require("take every eligible scorer-incorrect" in selection["rule"], "selection rule drift")
    require(contract["evaluation_handoff"]["draws_per_prompt"] == 16, "Monte Carlo draw drift")


def validate_static(package: Path, repo_root: Path) -> None:
    required = (
        "README.md",
        "env.sh",
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
        "SFT_SEQ_LENGTH=32768",
        "SFT_TENSOR_MODEL_PARALLEL_SIZE=4",
        "SFT_GLOBAL_BATCH_SIZE=64",
        "SFT_NUM_ROLLOUT=47",
        "SFT_FINAL_ROLLOUT_ID=46",
        "SFT_LR=5e-6",
        "SFT_MIN_LR=1e-6",
        "SFT_SAVE_INTERVAL=24",
        'SFT_MANIFEST_FAMILY="nous_aime_mixed_outcome"',
    ):
        require(needle in env, f"env contract missing: {needle}")
    for relative, walltime, gpu in (
        ("01_prepare_data.sbatch", "02:00:00", "h200:1"),
        ("02_canary.sbatch", "02:00:00", "h200:4"),
        ("03_train.sbatch", "04:00:00", "h200:4"),
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
    require("nous_aime_mixed_outcome" in manifest_tool, "checkpoint family missing")


def validate_sources(contract: dict[str, Any]) -> None:
    source_root = Path(
        "/gpfs/scrubbed/suryadv/datasets/NousResearch_eval-Qwen3-235B-A22B-reasoning/"
        "67bba1eb0642bc9a39d6971fd6c149e6bbbbb0c8"
    )
    dataset = contract["selection"]["dataset"]
    for year in ("aime24", "aime25"):
        path = source_root / year / "conversations.parquet"
        require(path.is_file(), f"missing staged source: {path}")
        require(sha256_file(path) == dataset[f"{year}_sha256"], f"source hash drift: {year}")
    model_root = Path("/gpfs/scrubbed/suryadv/slime-qwen-correct-only-openr1-sft-2k/v1/" "3568736a3743c319")
    model = model_root / "models/Qwen2.5-7B"
    for filename, expected in contract["base_model"]["tokenizer_files"].items():
        require(sha256_file(model / filename) == expected, f"base tokenizer drift: {filename}")
    model_config = load_json(model / "config.json")
    tensor_parallel_size = int(contract["training"]["tensor_parallel_size"])
    for field in (
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
    ):
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
        "NOUS_AIME_CONFIG_VALID "
        f"sources={int(args.require_source_artifacts)} pushed={int(args.require_pushed_clean)}"
    )


if __name__ == "__main__":
    main()
