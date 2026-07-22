#!/usr/bin/env python3
"""Fail-closed static validation for the full reproduction contract."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--require-pinned-commit", action="store_true")
    args = parser.parse_args()
    root = args.root
    contract = load(root / "config/experiment_contract.json")
    sft = load(root / "config/sft_config.json")
    opd = load(root / "config/opd_config.json")
    evaluation = load(root / "config/eval_config.json")
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(contract["branch"] == "qwen3-8b-32b-full-repro", "wrong experiment branch")
    if args.require_pinned_commit:
        require("TO_BE_PINNED" not in contract["repository_commit"], "repository commit is not pinned")
        require(contract["repository_commit"] == contract["slime_commit"], "SLIME/repository commits differ")
    require(sft["examples"] == 200_000 and sft["epochs"] == 1, "SFT exposure must be 200K x one epoch")
    require(sft["optimizer"]["lr"] == 1e-6 and sft["scheduler"] == "constant", "SFT LR/schedule mismatch")
    require(sft["native_context_length"] == 32768 and "never truncate" in sft["overlength_policy"], "SFT native-context policy mismatch")
    require(sft["dynamic_packing_max_tokens_per_gpu"] == 16384, "SFT dynamic packing target must remain 16,384 tokens/GPU")
    require(sft["hardware"]["gpus"] == 4, "SFT must use exactly four GPUs")
    require(sft["parallelism"] == {"tp": 2, "dp": 2, "pp": 1, "cp": 1}, "SFT parallelism mismatch")
    require(sft["checkpoint_exposure_rows"] == list(range(25_000, 200_001, 25_000)), "SFT checkpoint cadence mismatch")
    require(opd["prompt_rows"] == 1472 and opd["samples_per_prompt"] == 4 and opd["trajectories"] == 5888, "OPD size mismatch")
    require(opd["prompt_batch_size"] == 16 and opd["global_batch_size"] == 64 and opd["updates"] == 92, "OPD batch geometry mismatch")
    require(opd["sampling"] == {"temperature": 0.8, "top_p": 1.0, "top_k": -1}, "OPD sampling mismatch")
    require(opd["response_cap"] == 16384 and opd["rollout_context"] == 16380, "OPD 16K context mismatch")
    require(opd["rollout_context"] % (2 * opd["colocation"]["cp"]) == 0, "OPD context is not CP-compatible")
    require(opd["objective"] == {"opd_kl_coef": 1.0, "base_kl_coef": 0.0, "entropy_coef": 0.0, "scalar_reward": 0.0}, "OPD objective mismatch")
    require(opd["colocation"]["physical_gpus"] == 4 and opd["colocation"]["teacher_gpu"] == 3, "OPD must use colocated four-GPU layout")
    require(opd["student"] == "final_200k_sft", "OPD actor/reference initialization must be the final SFT weights")
    require(evaluation["sampling"] == {"temperature": 0.8, "top_p": 0.7, "top_k": -1, "n": 1, "seed": 1234}, "evaluation sampling mismatch")
    require(evaluation["max_new_tokens"] == 16384, "evaluation cap mismatch")
    require(evaluation["automatic_checkpoints"] == ["qwen3_8b_base", "qwen3_32b_teacher", "sft_200000", "opd_100pct"], "evaluation checkpoint policy mismatch")
    proxy = evaluation["cap_hit_context_proxy"]
    require(proxy["status"] == "diagnostic_not_primary_protocol", "32K proxy must remain diagnostic")
    require(proxy["target_total_context_tokens"] == 32768, "32K proxy context mismatch")
    require(proxy["safety_margin"] == evaluation["safety_margin"], "32K proxy safety margin mismatch")
    require(proxy["selection"].startswith("only rows with cap_hit=true"), "32K proxy must select only cap hits")
    require("exact prefix" in proxy["publication_gate"], "32K proxy must fail closed on prefix drift")
    require(
        proxy["serial_order"] == ["opd_100pct", "sft_200000", "qwen3_32b_teacher", "qwen3_8b_base"],
        "32K proxy serial order mismatch",
    )
    require(evaluation["dataset_policy"].startswith("load read-only"), "MATH-500 must be read-only")
    cleanup = contract["cleanup"]
    require(cleanup["version"] == "ot3_assistant_cjk_v1", "cleanup version mismatch")
    require(
        cleanup["assistant_character_filter"]
        == "reject any Han, Hiragana, Katakana, Hangul, or Bopomofo Unicode letter in assistant responses only",
        "assistant-only CJK filter mismatch",
    )
    require(cleanup["unicode_category"] == "Letter", "CJK filter must exclude punctuation and symbols")
    require(cleanup["legacy_non_english_reference_reused_for_assistant_cjk"] is True, "legacy assistant-CJK gate mapping must remain explicit")
    require(cleanup["relative_count_epsilon"] == 0.05, "cleanup reference tolerance must remain 5%")
    require(cleanup["minimum_eligible_rows"] == 300_000, "cleanup must retain at least 300K eligible rows")
    require(cleanup["prompt_grouping"] is False, "prompt grouping must stay disabled")
    require(cleanup["prompt_deduplication"] is False, "prompt deduplication must stay disabled")
    require(cleanup["math500_cross_contamination_check"] is False, "MATH-500 cross-contamination check must stay disabled")
    require(contract["failure_gates"]["cleanup_validation_gate"] == "fatal", "cleanup validation gate must stay fatal")
    require(contract["failure_gates"]["opd_behavior_metrics"] == "nonfatal_diagnostic", "OPD behavior metrics must be nonfatal")
    if os.environ.get("FULL_REPRO_DIR"):
        require(os.environ.get("EVAL_PROMPT") == evaluation["prompt_instruction"], "runtime prompt differs from eval contract")
        require(int(os.environ.get("SFT_SIZE", "0")) == sft["examples"], "runtime SFT row count differs")
        require(int(os.environ.get("SFT_SEQ_LENGTH", "0")) == sft["native_context_length"], "runtime SFT context differs")
        require(
            int(os.environ.get("SFT_MAX_TOKENS_PER_GPU", "0")) == sft["dynamic_packing_max_tokens_per_gpu"],
            "runtime SFT dynamic packing target differs",
        )
        require(int(os.environ.get("OPD_TRAIN_SIZE", "0")) == opd["prompt_rows"], "runtime OPD prompt count differs")
        require(int(os.environ.get("OPD_N_SAMPLES_PER_PROMPT", "0")) == opd["samples_per_prompt"], "runtime OPD samples/prompt differs")
        require(int(os.environ.get("OPD_MAX_RESPONSE_LEN", "0")) == opd["response_cap"], "runtime OPD response cap differs")
        require(int(os.environ.get("EVAL_MAX_RESPONSE_LEN", "0")) == evaluation["max_new_tokens"], "runtime eval response cap differs")
        require(
            int(os.environ.get("EVAL_CAP_PROXY_TARGET_CONTEXT", "0")) == proxy["target_total_context_tokens"],
            "runtime 32K proxy context differs",
        )
        for slurm in root.glob("*.sbatch"):
            text = slurm.read_text()
            require("#SBATCH --gres=gpu:h200:4" in text, f"{slurm.name} does not request exactly four H200s")
            cpu_match = re.search(r"^#SBATCH --cpus-per-task=(\d+)$", text, re.M)
            require(cpu_match is not None, f"{slurm.name} has no explicit CPU request")
            if cpu_match is not None:
                require(int(cpu_match.group(1)) <= 32, f"{slurm.name} exceeds the site limit of eight CPUs per requested GPU")
            time_match = re.search(r"^#SBATCH --time=(\d+):(\d+):(\d+)$", text, re.M)
            require(time_match is not None, f"{slurm.name} has no explicit wall-time request")
            if time_match is not None:
                hours, minutes, seconds = map(int, time_match.groups())
                require(hours * 3600 + minutes * 60 + seconds <= 86_400, f"{slurm.name} exceeds the normal-QoS one-day wall limit")
    if errors:
        raise SystemExit("CONFIG_VALIDATION_FAILED\n" + "\n".join(f"- {error}" for error in errors))
    print("CONFIG_VALIDATION_OK")


if __name__ == "__main__":
    main()
