#!/usr/bin/env python3
"""Fail-closed static, source-artifact, and pushed-revision preflight."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from examples.qwen2_5_7b_aime_generalization_incorrect_sft.data_utils import (
    ALL_TRAINED_VARIANTS,
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
        contract.get("family") == "qwen2_5_7b_aime_generalization_incorrect_sft",
        "experiment family drift",
    )
    require(tuple(contract.get("variants", ())) == ALL_TRAINED_VARIANTS, "variant order drift")
    training = contract["training"]
    expected = {
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_epsilon": 1e-8,
        "bf16": True,
        "child_final_iteration": 93,
        "child_optimizer_updates": 94,
        "context_parallel_size": 1,
        "data_parallel_size": 2,
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
        "pipeline_parallel_size": 1,
        "rollout_batch_size": 64,
        "stage1_final_iteration": 187,
        "stage1_optimizer_updates": 188,
        "tensor_parallel_size": 2,
        "warmup_fraction": 0.03,
        "warmup_init": 0.0,
        "weight_decay": 1e-4,
        "zero_stage": 1,
    }
    for key, expected_value in expected.items():
        require(
            training.get(key) == expected_value,
            f"training contract drift: {key}={training.get(key)!r}",
        )
    require("synchronous train.py" in training["train_entrypoint"], "resume-safe entrypoint drift")
    openr1 = contract["selection"]["openr1"]
    require(openr1["correct_rows"] == openr1["wrong_rows"] == 1500, "OpenR1 row count drift")
    require(openr1["source_id_audit"]["unique_problem_texts"] == 93733, "semantic ID audit drift")
    require(openr1["aime_question_exclusion"]["source_overlaps"] == 94, "AIME leakage gate drift")
    require(contract["selection"]["aime"]["stage1_rows"] == 3000, "stage1 row count drift")
    require(contract["submission"]["external_afterok_job"] == "222947", "external gate drift")


def validate_static(package: Path, repo_root: Path, contract: dict[str, Any]) -> None:
    required = (
        "README.md",
        "env.sh",
        "prepare_data.py",
        "build_variants.py",
        "validate_data.py",
        "runtime_token_audit.py",
        "runtime_repo_gate.py",
        "parent_checkpoint_gate.py",
        "finalize.py",
        "01_prepare_data.sbatch",
        "02_stage1_canary.sbatch",
        "03_train.sbatch",
        "04_continuation_canary.sbatch",
        "05_finalize.sbatch",
        "submit_training_only.sh",
    )
    for relative in required:
        require((package / relative).is_file(), f"missing package file: {relative}")
    env = (package / "env.sh").read_text(encoding="utf-8")
    for needle in (
        "SFT_TRAIN_ENTRYPOINT=train.py",
        'SFT_INITIAL_HF_DIR="${STAGE1_FINAL_HF_DIR}"',
        "SFT_SEQ_LENGTH=32768",
        "SFT_MAX_TOKENS_PER_GPU=16384",
        "SFT_LOG_PROBS_CHUNK_SIZE=2048",
        "SFT_GLOBAL_BATCH_SIZE=64",
        "SFT_LR=5e-6",
        "SFT_MIN_LR=1e-6",
        "SFT_SAVE_INTERVAL=24",
    ):
        require(needle in env, f"env contract missing: {needle}")
    train = (package / "03_train.sbatch").read_text(encoding="utf-8")
    for needle in (
        "#SBATCH --time=04:00:00",
        "#SBATCH --gres=gpu:h200:4",
        "#SBATCH --requeue",
        'parent_checkpoint_gate.py" check',
        "prune_hf_snapshots 1",
        "--fresh-optimizer",
    ):
        require(needle in train, f"training wrapper missing: {needle}")
    for relative in (
        "01_prepare_data.sbatch",
        "02_stage1_canary.sbatch",
        "04_continuation_canary.sbatch",
        "05_finalize.sbatch",
    ):
        text = (package / relative).read_text(encoding="utf-8")
        require("#SBATCH --time=02:00:00" in text, f"2h cap drift: {relative}")
        require("AIME_EXPECTED_GIT_COMMIT" in text, f"runtime commit gate missing: {relative}")
    require("AIME_EXPECTED_GIT_COMMIT" in train, "training runtime commit gate missing")
    submitter = (package / "submit_training_only.sh").read_text(encoding="utf-8")
    require("AIME_EXPECTED_GIT_COMMIT=${submit_commit}" in submitter, "submitter does not export pinned commit")
    runner = repo_root / "examples/qwen3_8b_opd_tillicum/04_run_sft_100k_8xh200.sbatch"
    runner_text = runner.read_text(encoding="utf-8")
    require(
        "SFT_TRAIN_ENTRYPOINT" in runner_text and '"${SFT_TRAIN_ENTRYPOINT}"' in runner_text,
        "shared runner lacks scoped entrypoint",
    )
    require("SFT_INITIAL_HF_DIR" in runner_text, "shared runner lacks continuation HF init")


def validate_sources(repo_root: Path, contract: dict[str, Any]) -> None:
    env_root = Path("/gpfs/scrubbed/suryadv/slime-qwen-correct-only-openr1-sft-2k/v1/3568736a3743c319")
    aime = Path(
        "/gpfs/scrubbed/suryadv/datasets/sxiong_AIME-trajectory/"
        "7fc7a89aaae52d114a7a1819088e048a7336cc02/train.jsonl"
    )
    require(aime.is_file(), f"missing staged AIME source: {aime}")
    require(
        sha256_file(aime) == contract["selection"]["aime"]["dataset"]["source_jsonl_sha256"],
        "staged AIME source hash drift",
    )
    model = env_root / "models/Qwen2.5-7B"
    for filename, expected in contract["base_model"]["tokenizer_files"].items():
        require(sha256_file(model / filename) == expected, f"base tokenizer hash drift: {filename}")
    torch_dist = env_root / "models/Qwen2.5-7B_torch_dist"
    require(
        (torch_dist / "latest_checkpointed_iteration.txt").read_text().strip() == "release",
        "base torch_dist tracker drift",
    )
    require((torch_dist / "release/.metadata").is_file(), "base torch_dist is incomplete")
    cache = Path(
        "/gpfs/scrubbed/suryadv/slime-qwen-correct-only-openr1-sft-2k/hf_home/"
        "open-r1___open_r1-math-220k/default/0.0.0/"
        "dc748648036c1ed619b020e056dc4b603eb39817"
    )
    arrow = sorted(cache.glob("open_r1-math-220k-train-*.arrow"))
    require(len(arrow) == 10 and all(path.stat().st_size > 0 for path in arrow), "pinned OpenR1 cache is incomplete")
    upstream_path = Path(contract["submission"]["external_chain_manifest"])
    require(
        sha256_file(upstream_path) == contract["submission"]["external_chain_manifest_sha256"],
        "external chain manifest hash drift",
    )
    upstream = load_json(upstream_path)
    require(str(upstream["jobs_in_dependency_order"][-1]["job_id"]) == "222947", "external chain tail drift")
    base_manifest = env_root / "handoff/checkpoints/base_qwen2_5_7b.json"
    base_value = load_json(base_manifest)
    require(
        base_value["checkpoint_manifest_sha256"] == contract["base_model"]["checkpoint_identity"],
        "base manifest identity drift",
    )
    tool = repo_root / "examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff/checkpoint_manifest.py"
    subprocess.run(
        ["python3", str(tool), "verify", "--manifest", str(base_manifest), "--checkpoint", str(model)],
        check=True,
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
    package = args.contract.resolve().parent.parent
    contract = load_json(args.contract)
    validate_contract(contract)
    validate_static(package, repo_root, contract)
    if args.require_source_artifacts:
        validate_sources(repo_root, contract)
    if args.require_pushed_clean:
        validate_pushed_clean(repo_root)
    print(
        "AIME_GENERALIZATION_CONFIG_VALID "
        f"sources={int(args.require_source_artifacts)} pushed={int(args.require_pushed_clean)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
