#!/usr/bin/env python3
"""Fail-closed preflight for the 40K recovery plus 2K Slime suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


BRANCH = "qwen3-8b-openr1-masked-sft-40k"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example-dir", type=Path, required=True)
    parser.add_argument("--require-pushed-clean", action="store_true")
    args = parser.parse_args()
    example = args.example_dir.resolve()
    repo = example.parents[1]
    contract = json.loads(
        (example / "config/experiment_contract.json").read_text(encoding="utf-8")
    )
    expected_variants = [
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
    if contract["variants"] != expected_variants:
        raise ValueError("variant order drift")
    training = contract["training"]
    expected_training = {
        "attention_dropout": 0.0,
        "full_sft": True,
        "global_batch_size": 200,
        "hidden_dropout": 0.0,
        "learning_rate": 1e-06,
        "lr_schedule": "constant",
        "max_sequence_length": 32768,
        "num_epochs": 1,
        "optimizer": "adam",
        "rollout_batch_size": 200,
        "weight_decay": 0.1,
    }
    if training != expected_training:
        raise ValueError("speedy full-SFT recipe drift")
    evaluation = contract["evaluation"]
    if (
        evaluation["decoding"] != "greedy"
        or evaluation["dynamic_response_budget"]
        != "32768 - rendered_prompt_tokens - 64"
        or evaluation["stop_token_ids"] != [151643, 151645]
        or evaluation["scorer"] != "math500_strict_boxed_scorer_v3"
    ):
        raise ValueError("evaluation policy drift")

    source_root = Path(
        "/gpfs/scrubbed/suryadv/repos/axolotl-masked-sft/runs/"
        "2026-06-23_openr1_220k_2k_balanced_full_sft_axolotl_harp_vllm_prob_ratio_mask/"
        "unpaired/seed_42/datasets"
    )
    selection = contract["selection"]
    if (
        selection["correct_only_trace_order_sha256"]
        != "95c199c745005797627f210b39078058161f3e27531b7af91db5ba4a6cc1bd94"
        or selection["mixed_trace_order_sha256"]
        != "a6020e731bc64f93114d398e4d868f92f0505e7942f52d19fb26aa93b302f46d"
        or selection["source_structure"]
        != {
            "correct_only_duplicate_close": 0,
            "correct_only_missing_close": 1,
            "mixed_duplicate_close": 1,
            "mixed_missing_close": 56,
        }
    ):
        raise ValueError("canonical trace fingerprint or structure audit drift")
    source_hashes = {
        source_root / "correct_only.jsonl": selection[
            "source_correct_only_sha256"
        ],
        source_root / "correct_wrong_unmasked.jsonl": selection[
            "source_correct_wrong_unmasked_sha256"
        ],
        source_root / "stats.json": selection["source_stats_sha256"],
    }
    for path, expected in source_hashes.items():
        if sha256(path) != expected:
            raise ValueError(f"canonical trace source hash mismatch: {path}")
    axolotl_repo = Path(
        "/gpfs/scrubbed/suryadv/repos/axolotl-masked-sft/repos/Axolotl-Masked-SFT"
    )
    if git(axolotl_repo, "rev-parse", "HEAD") != selection[
        "source_repository_commit"
    ]:
        raise ValueError("Axolotl source commit drift")

    model_root = Path("/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/models")
    for path in (
        model_root / "Qwen3-8B-Base/config.json",
        model_root / "Qwen3-8B-Base_torch_dist/latest_checkpointed_iteration.txt",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    required_source = {
        repo
        / "slime/backends/megatron_utils/cp_utils.py": "TOKEN_WEIGHT_FIXED_POINT_SCALE = 256",
        repo
        / "slime/backends/megatron_utils/loss.py": "get_fixed_point_token_normalizer",
        repo / "examples/qwen3_8b_32b_full_repro/evaluate_math500.py": "AsyncLLM",
        repo
        / "examples/qwen3_8b_32b_full_repro/math500_scorer.py": 'SCORER_VERSION = "math500_strict_boxed_scorer_v3"',
    }
    for path, needle in required_source.items():
        if needle not in path.read_text(encoding="utf-8"):
            raise ValueError(f"required implementation missing in {path}")
    scripts = sorted(example.glob("*.sbatch")) + sorted(example.glob("*.sh"))
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)
    handoff = (
        repo
        / "examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff"
    )
    subprocess.run(
        ["bash", "-n", str(handoff / "run_one_h200.sbatch")], check=True
    )
    policy = json.loads((handoff / "eval_policy.json").read_text())
    if (
        policy["decoding"]["temperature"] != 0.0
        or policy["decoding"]["top_p"] != 1.0
        or policy["decoding"]["top_k"] != -1
        or policy["decoding"]["stop_token_ids"] != [151643, 151645]
        or policy["response_budget"]["formula"]
        != "32768 - rendered_prompt_tokens - 64"
        or policy["persistence"]["commit_unit"]
        != "one_completed_problem"
        or policy["persistence"]["shards"] != 1
    ):
        raise ValueError("portable MATH-500 handoff policy drift")
    inventory = json.loads((handoff / "checkpoint_sources.json").read_text())
    if (
        len(inventory["checkpoints"]) != 15
        or sum(
            item["full_sft"] is True
            for item in inventory["checkpoints"]
        )
        != 14
    ):
        raise ValueError("checkpoint handoff inventory drift")
    training_launcher = (example / "submit_training_only.sh").read_text()
    if (
        "04_eval_math500.sbatch" in training_launcher
        or "05_report.sbatch" in training_launcher
        or training_launcher.index("submit_serial prepare_2k_traces")
        > training_launcher.index(
            "submit_serial train_40k_weighted_tau_0p20"
        )
    ):
        raise ValueError("training-only launch scope or order drift")

    if args.require_pushed_clean:
        if git(repo, "branch", "--show-current") != BRANCH:
            raise ValueError(f"submission must run from {BRANCH}")
        if git(repo, "status", "--porcelain"):
            raise ValueError("submission requires a clean worktree")
        upstream = git(
            repo,
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        )
        if git(repo, "rev-parse", "HEAD") != git(repo, "rev-parse", upstream):
            raise ValueError("local HEAD is not the pushed upstream commit")
    print(
        "CONFIG_2K_VALID full_sft=1 traces=2000 unpaired=1 "
        "eval=greedy_dynamic_32k scorer=v3",
        flush=True,
    )


if __name__ == "__main__":
    main()
