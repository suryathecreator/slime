from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import pytest

from examples.qwen2_5_7b_openr1_math220k_correct_recipe_masked_sft_2k import (
    prepare_data,
)
from examples.qwen2_5_7b_openr1_math220k_correct_recipe_random_mask_replicates_sft_2k.build_random_masks import (
    EXPECTED_VARIANTS as REPLICATE_VARIANTS,
    load_variant_specs,
    wrong_assistant_weights,
)
from examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.build_variants import (
    wrong_weights,
)

NUM_GPUS = 0
EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[1] / "examples" / "qwen2_5_7b_openr1_math220k_correct_recipe_masked_sft_2k"
)
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
PROVENANCE_DIR = EXPERIMENT_DIR / "provenance/v1/07291674d43cf3da"
REPLICATE_DIR = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "qwen2_5_7b_openr1_math220k_correct_recipe_random_mask_replicates_sft_2k"
)


def load_contract() -> dict:
    return json.loads((EXPERIMENT_DIR / "config/experiment_contract.json").read_text(encoding="utf-8"))


def test_contract_copies_the_successful_qwen25_7b_recipe_exactly():
    contract = load_contract()
    training = contract["training"]
    assert tuple(contract["variants"]) == VARIANTS
    assert "correct_only" not in contract["variants"]
    assert contract["base_model"] == {
        "hf_repo": "Qwen/Qwen2.5-7B",
        "revision": "d149729398750b98c0af14eb82c78cfe92750796",
    }
    assert training == {
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_epsilon": 1e-8,
        "bf16": True,
        "context_parallel_size": 1,
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
        "pipeline_parallel_size": 1,
        "recompute_loss_function": True,
        "rollout_batch_size": 64,
        "tensor_parallel_size": 2,
        "trace_token_cap": 8192,
        "warmup_fraction": 0.03,
        "warmup_init": 0.0,
        "weight_decay": 1e-4,
        "zero_stage": 1,
    }
    assert 2000 * training["epochs"] // training["global_batch_size"] == 125


def test_selection_reuses_the_frozen_correct_subset_and_one_shared_mixed_order():
    selection = load_contract()["selection"]
    assert selection["correct_rows"] == selection["wrong_rows"] == 1000
    assert selection["correct_subset"] == ("first_1000_of_frozen_seed_42_correct_only_order")
    assert selection["wrong_problem_exclusion"] == ("all_2000_frozen_correct_only_problem_ids")
    assert selection["mixed_order"] == "correct_then_wrong"
    prep = (EXPERIMENT_DIR / "prepare_data.py").read_text(encoding="utf-8")
    assert "frozen[: args.correct_rows]" in prep
    assert "mixed = correct + wrong" in prep
    assert '"same_order_required_for_every_variant": True' in prep


def test_seeded_wrong_reservoir_is_repeatable_and_excludes_frozen_problems(
    monkeypatch,
):
    dataset = [{"uuid": f"p{index}"} for index in range(12)]

    def fake_eligible(source, row_index, *args, **kwargs):
        del args, kwargs
        return {
            "trace_id": f"{source['uuid']}:trace",
            "problem_id": source["uuid"],
        }

    monkeypatch.setattr(prepare_data, "eligible_wrong", fake_eligible)
    first, _, _ = prepare_data.select_wrong_rows(dataset, {"p0", "p1"}, object(), 5, 10240, 8192, 42)
    second, _, _ = prepare_data.select_wrong_rows(dataset, {"p0", "p1"}, object(), 5, 10240, 8192, 42)
    assert [row["trace_id"] for row in first] == [row["trace_id"] for row in second]
    assert not ({row["problem_id"] for row in first} & {"p0", "p1"})


def test_mask_formulas_and_protected_tokens_match_the_prior_variant_recipe(
    monkeypatch,
):
    monkeypatch.setattr(
        "examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.build_variants.stable_unit_interval",
        lambda seed, trace_id, index: 0.5,
    )
    margins = [-0.2, 0.0, math.log(4.0)]
    assert wrong_weights("margin_mask", margins, "t", 42, [2]) == [0.0, 0.0, 1.0]
    assert wrong_weights("inverse_tau_0p20", margins, "t", 42, [2]) == pytest.approx([0.5, 1.0, 1.0])
    assert wrong_weights("inverse_tau_0p05", margins, "t", 42, [2]) == pytest.approx([0.2, 1.0, 1.0])
    assert wrong_weights("prob_ratio_mask", margins, "t", 42, [2]) == [0.0, 0.0, 1.0]


def test_runtime_environment_resolves_the_exact_qwen25_7b_values():
    names = (
        "SFT_GLOBAL_BATCH_SIZE SFT_ROLLOUT_BATCH_SIZE SFT_NUM_EPOCH "
        "SFT_NUM_ROLLOUT SFT_FINAL_ROLLOUT_ID SFT_SEQ_LENGTH "
        "SFT_TENSOR_MODEL_PARALLEL_SIZE SFT_MAX_TOKENS_PER_GPU "
        "SFT_LR SFT_MIN_LR SFT_LR_DECAY_STYLE SFT_LR_WARMUP_FRACTION "
        "SFT_LR_DECAY_ITERS SFT_ADAM_BETA1 SFT_ADAM_BETA2 SFT_ADAM_EPS "
        "SFT_WEIGHT_DECAY SFT_GRAD_CLIP SFT_ZERO_STAGE "
        "SFT_OPTIMIZER_CPU_OFFLOAD SFT_RECOMPUTE_LOSS_FUNCTION"
    )
    command = 'source "$1"; shift; for name in "$@"; do ' 'printf "%s=%s\\n" "$name" "${!name}"; done'
    output = subprocess.check_output(
        [
            "bash",
            "-c",
            command,
            "bash",
            str(EXPERIMENT_DIR / "env.sh"),
            *names.split(),
        ],
        text=True,
    )
    assert dict(line.split("=", 1) for line in output.splitlines()) == {
        "SFT_GLOBAL_BATCH_SIZE": "64",
        "SFT_ROLLOUT_BATCH_SIZE": "64",
        "SFT_NUM_EPOCH": "4",
        "SFT_NUM_ROLLOUT": "125",
        "SFT_FINAL_ROLLOUT_ID": "124",
        "SFT_SEQ_LENGTH": "10240",
        "SFT_TENSOR_MODEL_PARALLEL_SIZE": "2",
        "SFT_MAX_TOKENS_PER_GPU": "16384",
        "SFT_LR": "5e-6",
        "SFT_MIN_LR": "1e-6",
        "SFT_LR_DECAY_STYLE": "cosine",
        "SFT_LR_WARMUP_FRACTION": "0.03",
        "SFT_LR_DECAY_ITERS": "125",
        "SFT_ADAM_BETA1": "0.9",
        "SFT_ADAM_BETA2": "0.95",
        "SFT_ADAM_EPS": "1.0e-8",
        "SFT_WEIGHT_DECAY": "1e-4",
        "SFT_GRAD_CLIP": "1.0",
        "SFT_ZERO_STAGE": "1",
        "SFT_OPTIMIZER_CPU_OFFLOAD": "1",
        "SFT_RECOMPUTE_LOSS_FUNCTION": "1",
    }


def test_submission_is_strict_serial_and_never_retrains_correct_only():
    submit = (EXPERIMENT_DIR / "submit_training_only.sh").read_text(encoding="utf-8")
    train = (EXPERIMENT_DIR / "04_train.sbatch").read_text(encoding="utf-8")
    assert '--dependency="afterok:${tail_job}"' in submit
    assert 'for variant in "${VARIANTS[@]}"' in submit
    assert "DRY_RUN_COMPLETE jobs=14" in submit
    assert "SFT_VARIANT=correct_only" not in submit
    assert "train_correct_only" not in submit
    assert 'SFT_PARQUET="$(variant_data_path "${SFT_VARIANT}")"' in train
    assert "iter_0000124" in train


def test_handoff_transfers_ten_new_checkpoints_but_compares_twelve():
    finalize = (EXPERIMENT_DIR / "finalize.py").read_text(encoding="utf-8")
    rsync = (EXPERIMENT_DIR / "rsync_to_klone.sh").read_text(encoding="utf-8")
    assert '"checkpoint_count": len(transfer_records)' in finalize
    assert '"comparison_checkpoint_count": len(comparison_targets)' in finalize
    assert "SOURCE_REMOTE_EXPERIMENT" in finalize
    assert '[[ "${#records[@]}" -eq 10 ]]' in rsync
    assert "selection_and_tokenization_stats.json" in rsync
    assert "comparison_sources.json" in rsync


def test_shell_launch_surfaces_parse():
    paths = sorted(EXPERIMENT_DIR.glob("*.sh")) + sorted(EXPERIMENT_DIR.glob("*.sbatch"))
    for path in paths:
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_every_slurm_stage_has_a_strict_two_hour_cap():
    sbatch_paths = sorted(EXPERIMENT_DIR.glob("*.sbatch"))
    assert len(sbatch_paths) == 5
    for path in sbatch_paths:
        text = path.read_text(encoding="utf-8")
        assert text.count("#SBATCH --time=02:00:00") == 1, path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_published_provenance_is_complete_and_self_consistent():
    publication = json.loads((PROVENANCE_DIR / "PROVENANCE_PUBLICATION.json").read_text(encoding="utf-8"))
    expected_contract_hash = file_sha256(EXPERIMENT_DIR / "config/experiment_contract.json")[:16]
    assert publication["contract_hash"] == expected_contract_hash == "07291674d43cf3da"
    files = publication["published_files"]
    assert publication["published_file_count_excluding_self"] == len(files) == 21
    actual_paths = {
        path.relative_to(PROVENANCE_DIR).as_posix()
        for path in PROVENANCE_DIR.rglob("*")
        if path.is_file() and path.name != "PROVENANCE_PUBLICATION.json"
    }
    assert actual_paths == {item["path"] for item in files}
    for item in files:
        path = PROVENANCE_DIR / item["path"]
        assert path.stat().st_size == item["bytes"]
        assert file_sha256(path) == item["sha256"]
        if item["source_path"] is not None:
            assert item["sha256"] == item["source_sha256"]

    status = json.loads((PROVENANCE_DIR / "TRAINING_STATUS.json").read_text(encoding="utf-8"))
    inventory = json.loads((PROVENANCE_DIR / "handoff/checkpoint_sources.json").read_text(encoding="utf-8"))
    comparison = json.loads((PROVENANCE_DIR / "handoff/comparison_sources.json").read_text(encoding="utf-8"))
    submission = json.loads((PROVENANCE_DIR / "submission/training_chain.json").read_text(encoding="utf-8"))
    assert status["trained_checkpoints"] == inventory["checkpoint_count"] == 10
    assert status["comparison_checkpoint_count"] == comparison["comparison_checkpoint_count"] == 12
    assert len(submission["jobs_in_dependency_order"]) == 14
    assert [item["id"] for item in inventory["checkpoints"]] == list(VARIANTS)
    for variant in VARIANTS:
        manifest = json.loads((PROVENANCE_DIR / f"handoff/checkpoints/{variant}.json").read_text(encoding="utf-8"))
        assert manifest["final_iteration"] == 124
        assert manifest["checkpoint_manifest_sha256"] == status["checkpoint_identities"][variant]

    baseline_manifests = {
        "base_qwen2_5_7b": "base_qwen2_5_7b.json",
        "qwen2_5_7b_8k_correct_only": "qwen2_5_7b_8k_correct_only.json",
    }
    for checkpoint_id, filename in baseline_manifests.items():
        manifest = json.loads((PROVENANCE_DIR / "handoff/baseline_manifests" / filename).read_text(encoding="utf-8"))
        assert manifest["checkpoint_manifest_sha256"] == status["checkpoint_identities"][checkpoint_id]

    trace_rows = [
        json.loads(line)
        for line in (PROVENANCE_DIR / "data/selected_trace_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(trace_rows) == publication["trace_identity_rows"] == 2000
    assert len({row["trace_id"] for row in trace_rows}) == 2000
    assert sum(row["is_correct"] for row in trace_rows) == 1000
    assert all(row["is_correct"] for row in trace_rows[:1000])
    assert not any(row["is_correct"] for row in trace_rows[1000:])
    assert not ({"problem", "answer", "assistant_trace"} & set(trace_rows[0]))


def test_replicate_contract_reuses_exact_recipe_and_frozen_source():
    contract_path = REPLICATE_DIR / "config/experiment_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert tuple(load_variant_specs(contract_path)) == REPLICATE_VARIANTS
    assert contract["training"] == load_contract()["training"]
    assert contract["source_experiment"] == {
        "comparison_sources_sha256": "4683fa2db3c45cd64b7ca801377c049219d0ccd4c0f0fcacb61b6965627a401e",
        "contract_hash": "07291674d43cf3da",
        "contract_sha256": "07291674d43cf3dab1675c187dbd0c1c389a5b4002702e3c8b8a6a4d1abfabc7",
        "family": "qwen2_5_7b_openr1_math220k_correct_recipe_masked_sft_2k",
        "ordered_mixed_trace_ids_sha256": "f3f51daa4e63ee6bc33ba91e97b05c195f9a40ddc641de7abb60408d06bd904a",
        "schedule_audit_sha256": "d10105f0262396f46165887f52f94db90e95bb6c752ad2c1c7edf8e30130fe40",
        "selection_and_tokenization_stats_sha256": "57fd47fd7cfac7fbbed3d3b369d78f6cd6ea3109824a413e0ad39ec44e66066d",
        "tokenizer_control_inventory_sha256": "8413031275892990bc57a1f2cefd4b35951fbfad122fcbeb2383efae830eae85",
        "training_status_sha256": "044d3dc982225c6a6120bceb02b3e1ecc00c7d1b23aec022d682eeacaa02595d",
        "unmasked_dataset_sha256": "b7c13714061fb5078848943d915d64639264eebcad76023e43637191e1080234",
    }


def test_replicate_random_masks_are_deterministic_nested_and_seeded():
    protected = [2, 9]
    five_42 = wrong_assistant_weights(1000, protected, 42, 0.05, "trace")
    fifteen_42 = wrong_assistant_weights(1000, protected, 42, 0.15, "trace")
    five_43 = wrong_assistant_weights(1000, protected, 43, 0.05, "trace")
    assert five_42 == wrong_assistant_weights(1000, protected, 42, 0.05, "trace")
    assert {index for index, value in enumerate(five_42) if value == 0.0} <= {
        index for index, value in enumerate(fifteen_42) if value == 0.0
    }
    assert five_42 != five_43
    assert all(weights[index] == 1.0 for weights in (five_42, fifteen_42, five_43) for index in protected)


def test_replicate_runtime_environment_copies_training_recipe_exactly():
    names = (
        "SFT_NUM_EPOCH SFT_ROLLOUT_BATCH_SIZE SFT_GLOBAL_BATCH_SIZE "
        "SFT_NUM_ROLLOUT SFT_FINAL_ROLLOUT_ID SFT_SAVE_INTERVAL "
        "SFT_ACTOR_GPUS SFT_TENSOR_MODEL_PARALLEL_SIZE "
        "SFT_CONTEXT_PARALLEL_SIZE SFT_PIPELINE_MODEL_PARALLEL_SIZE "
        "SFT_ZERO_STAGE SFT_CKPT_FORMAT SFT_INPUT_KEY SFT_SEQ_LENGTH "
        "SFT_GRAD_CLIP SFT_OPTIMIZER_CPU_OFFLOAD "
        "SFT_RECOMPUTE_LOSS_FUNCTION SFT_OPTIMIZER SFT_LR SFT_MIN_LR "
        "SFT_LR_DECAY_STYLE SFT_LR_WARMUP_FRACTION SFT_LR_WARMUP_INIT "
        "SFT_LR_DECAY_ITERS SFT_WEIGHT_DECAY SFT_ADAM_BETA1 "
        "SFT_ADAM_BETA2 SFT_ADAM_EPS SFT_MAX_TOKENS_PER_GPU"
    ).split()
    command = 'source "$1"; shift; for name in "$@"; do ' 'printf "%s=%s\\n" "$name" "${!name}"; done'

    def resolved(env_path: Path) -> dict[str, str]:
        output = subprocess.check_output(["bash", "-c", command, "bash", str(env_path), *names], text=True)
        return dict(line.split("=", 1) for line in output.splitlines())

    assert resolved(REPLICATE_DIR / "env.sh") == resolved(EXPERIMENT_DIR / "env.sh")


def test_replicate_launch_is_six_trainings_in_one_nine_job_serial_chain():
    submit = (REPLICATE_DIR / "submit_training_only.sh").read_text(encoding="utf-8")
    train = (REPLICATE_DIR / "03_train.sbatch").read_text(encoding="utf-8")
    finalize = (REPLICATE_DIR / "finalize.py").read_text(encoding="utf-8")
    rsync = (REPLICATE_DIR / "rsync_to_klone.sh").read_text(encoding="utf-8")
    assert '--dependency="afterok:${tail_job}"' in submit
    assert 'for variant in "${VARIANTS[@]}"' in submit
    assert "DRY_RUN_COMPLETE jobs=9" in submit
    assert "submit_serial correct_only" not in submit
    assert 'SFT_PARQUET="$(variant_data_path "${SFT_VARIANT}")"' in train
    assert "iter_0000124" in train
    assert "len(transfer_records) != 6 or len(comparison_targets) != 18" in finalize
    assert '[[ "${#records[@]}" -eq 6 ]]' in rsync


def test_replicate_shell_surfaces_parse_and_keep_two_hour_caps():
    paths = sorted(REPLICATE_DIR.glob("*.sh")) + sorted(REPLICATE_DIR.glob("*.sbatch"))
    for path in paths:
        subprocess.run(["bash", "-n", str(path)], check=True)
    sbatch_paths = sorted(REPLICATE_DIR.glob("*.sbatch"))
    assert len(sbatch_paths) == 4
    for path in sbatch_paths:
        assert path.read_text(encoding="utf-8").count("#SBATCH --time=02:00:00") == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
