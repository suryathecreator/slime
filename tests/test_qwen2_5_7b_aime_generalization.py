from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from examples.qwen2_5_7b_aime_generalization_incorrect_sft import build_variants, prepare_data, validate_data
from examples.qwen2_5_7b_aime_generalization_incorrect_sft.data_utils import (
    CHILD_VARIANTS,
    MASK_VARIANTS,
    stable_hex,
)

from slime.rollout.data_source import RolloutDataSource

NUM_GPUS = 0
ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "examples/qwen2_5_7b_aime_generalization_incorrect_sft"


def contract() -> dict:
    return json.loads((EXPERIMENT / "config/experiment_contract.json").read_text(encoding="utf-8"))


def test_contract_pins_sources_variants_and_exact_training_recipe() -> None:
    value = contract()
    assert value["base_model"] == {
        "checkpoint_identity": "b62a56261fb3642627086c1a083e453e298ba683d3cefd4196245e11c857de8f",
        "hf_repo": "Qwen/Qwen2.5-7B",
        "revision": "d149729398750b98c0af14eb82c78cfe92750796",
        "tokenizer_files": {
            "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
            "tokenizer_config.json": "c91efca15ceff6e9ee9424db58a6f59cd41294e550a86cbd07e3c1fb500b34f9",
        },
    }
    assert tuple(value["variants"]) == ("aime_correct_3000", *CHILD_VARIANTS)
    assert tuple(value["masking"]["probabilities"]) == tuple(rate / 100 for rate in range(10, 100, 10))
    training = value["training"]
    assert {
        key: training[key]
        for key in (
            "adam_beta1",
            "adam_beta2",
            "adam_epsilon",
            "global_batch_size",
            "gradient_clip_norm",
            "learning_rate",
            "max_sequence_length",
            "max_tokens_per_gpu",
            "min_learning_rate",
            "warmup_fraction",
            "weight_decay",
        )
    } == {
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_epsilon": 1e-8,
        "global_batch_size": 64,
        "gradient_clip_norm": 1.0,
        "learning_rate": 5e-6,
        "max_sequence_length": 32768,
        "max_tokens_per_gpu": 16384,
        "min_learning_rate": 1e-6,
        "warmup_fraction": 0.03,
        "weight_decay": 1e-4,
    }
    assert training["stage1_optimizer_updates"] == 188
    assert training["stage1_final_iteration"] == 187
    assert training["child_optimizer_updates"] == 94
    assert training["child_final_iteration"] == 93


def test_contract_pins_aime_audits_and_openr1_leakage_exclusion() -> None:
    selection = contract()["selection"]
    aime = selection["aime"]
    assert aime["dataset"]["revision"] == "7fc7a89aaae52d114a7a1819088e048a7336cc02"
    assert aime["dataset"]["source_jsonl_sha256"] == (
        "8f1ea33b8c209479a7bf84bc6838a7cadb9b53b37906d0caf2f46cecf1b00515"
    )
    assert (aime["held_in_problems"], aime["held_out_problems"], aime["unused_problems"]) == (
        400,
        400,
        61,
    )
    assert aime["source_render_audit"] == {
        "eligible_problems": 861,
        "eligible_rows": 1227,
        "longest_eligible_total_tokens": 32246,
        "max_problem_min_total_tokens": 31901,
        "over_32768_rows": 10,
    }
    openr1 = selection["openr1"]
    assert openr1["source_id_audit"]["rows"] == 93733
    assert openr1["source_id_audit"]["invalid_raw_id_rows"] == 664
    assert openr1["source_id_audit"]["unique_problem_texts"] == 93733
    assert openr1["aime_question_exclusion"] == {
        "canonicalization": "SHA256(UTF-8 of whitespace-collapsed exact question text)",
        "held_in_overlaps": 41,
        "held_out_problem_overlaps": 45,
        "source_overlaps": 94,
        "unused_overlaps": 8,
        "selected_overlaps_allowed": 0,
    }


def test_aime_trajectory_selection_inserts_prompt_verbatim_and_uses_min_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def fake_build(
        tokenizer,
        *,
        user_prompt,
        assistant_text,
        metadata,
        max_sequence_length,
    ):
        del tokenizer, assistant_text, max_sequence_length
        captured.append(user_prompt)
        return {
            "input_ids": [1, 2, 3],
            "metadata": {
                **metadata,
                "assistant_token_count": 1,
                "assistant_token_sha256": hashlib.sha256(b"x").hexdigest(),
            },
        }

    monkeypatch.setattr(prepare_data, "build_prompt_record", fake_build)
    rows = [
        {
            "answer": 1,
            "content": r"Final \boxed{1}",
            "correct": True,
            "id": trace_id,
            "model": "source",
            "prompt": f"VERBATIM PROMPT {trace_id}",
            "reasoning_content": "reasoning",
        }
        for trace_id in ("trace-a", "trace-b")
    ]
    selected, _ = prepare_data.select_aime_trajectory(object(), "problem", rows, 42, 32768, Counter())
    expected = min(
        rows,
        key=lambda row: (
            stable_hex("aime-trajectory-selection-v1", "seed=42", row["id"]),
            row["id"],
        ),
    )
    assert selected["source_sample_id"] == expected["id"]
    assert selected["prompt"] == expected["prompt"]
    assert captured == [row["prompt"] for row in rows]


def test_openr1_problem_identity_ignores_unreliable_uuid_and_leakage_hash_is_canonical() -> None:
    left, text = prepare_data.semantic_openr1_problem_id({"uuid": "NaN", "problem": "A  problem\nwith spacing"})
    right, _ = prepare_data.semantic_openr1_problem_id({"uuid": "different", "problem": "A  problem\nwith spacing"})
    assert left == right == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert prepare_data.normalized_question_sha256("A  problem\nwith spacing") == (
        prepare_data.normalized_question_sha256(" A problem with\tspacing ")
    )


def test_random_masks_are_deterministic_nested_and_preserve_protected_and_terminal_tokens() -> None:
    source = {
        "input_ids": list(range(102)),
        "metadata": {
            "assistant_token_count": 100,
            "is_correct": False,
            "loss_weights": [1.0] * 101 + [0.0],
            "protected_assistant_positions": [0, 7, 99],
            "trace_id": "wrong-trace",
        },
    }
    ten, _ = build_variants.masked_row(source, "continue_random_mask_10", 42)
    ninety, _ = build_variants.masked_row(source, "continue_random_mask_90", 42)
    ten_masked = {index for index, weight in enumerate(ten["metadata"]["loss_weights"][:100]) if weight == 0.0}
    ninety_masked = {index for index, weight in enumerate(ninety["metadata"]["loss_weights"][:100]) if weight == 0.0}
    assert ten_masked <= ninety_masked
    assert ten == build_variants.masked_row(source, "continue_random_mask_10", 42)[0]
    assert all(row["metadata"]["loss_weights"][index] == 1.0 for row in (ten, ninety) for index in (0, 7, 99))
    assert ten["metadata"]["loss_weights"][-2:] == [1.0, 0.0]
    assert ninety["metadata"]["loss_weights"][-2:] == [1.0, 0.0]


def test_real_dp_scheduler_covers_four_epochs_plus_exact_wrap() -> None:
    stage = validate_data.realized_dynamic_schedule(
        [{"input_ids": [0] * 128} for _ in range(3000)],
        updates=188,
        seed=42,
        global_batch_size=64,
        max_tokens_per_gpu=16384,
    )
    child = validate_data.realized_dynamic_schedule(
        [{"input_ids": [0] * 128} for _ in range(1500)],
        updates=94,
        seed=42,
        global_batch_size=64,
        max_tokens_per_gpu=16384,
    )
    assert stage["exposure_histogram"] == {"4": 2968, "5": 32}
    assert child["exposure_histogram"] == {"4": 1484, "5": 16}
    assert stage["exposures"] == 188 * 64
    assert child["exposures"] == 94 * 64


def resolved_environment(variant: str) -> dict[str, str]:
    names = (
        "SFT_SIZE SFT_NUM_ROLLOUT SFT_FINAL_ROLLOUT_ID SFT_SEQ_LENGTH "
        "SFT_GLOBAL_BATCH_SIZE SFT_ROLLOUT_BATCH_SIZE SFT_ACTOR_GPUS "
        "SFT_TENSOR_MODEL_PARALLEL_SIZE SFT_CONTEXT_PARALLEL_SIZE "
        "SFT_PIPELINE_MODEL_PARALLEL_SIZE SFT_MAX_TOKENS_PER_GPU "
        "SFT_LR SFT_MIN_LR SFT_LR_DECAY_STYLE SFT_LR_WARMUP_FRACTION "
        "SFT_LR_WARMUP_INIT SFT_LR_DECAY_ITERS SFT_ADAM_BETA1 "
        "SFT_ADAM_BETA2 SFT_ADAM_EPS SFT_WEIGHT_DECAY SFT_GRAD_CLIP "
        "SFT_ZERO_STAGE SFT_OPTIMIZER_CPU_OFFLOAD SFT_RECOMPUTE_LOSS_FUNCTION "
        "SFT_INITIAL_HF_DIR SFT_TRAIN_ENTRYPOINT SFT_SAVE_INTERVAL "
        "CHECKPOINT_REQUIRE_ROLLOUT_STATE"
    ).split()
    command = (
        'source "$1"; configure_sft_variant "$2"; shift 2; '
        'for name in "$@"; do printf "%s=%s\\n" "$name" "${!name-}"; done'
    )
    output = subprocess.check_output(
        [
            "bash",
            "-c",
            command,
            "bash",
            str(EXPERIMENT / "env.sh"),
            variant,
            *names,
        ],
        text=True,
    )
    return dict(line.split("=", 1) for line in output.splitlines())


def test_runtime_environment_stage_and_children_share_recipe_but_restart_schedule() -> None:
    stage = resolved_environment("aime_correct_3000")
    child = resolved_environment("continue_random_mask_90")
    shared = {
        "SFT_SEQ_LENGTH": "32768",
        "SFT_GLOBAL_BATCH_SIZE": "64",
        "SFT_ROLLOUT_BATCH_SIZE": "64",
        "SFT_ACTOR_GPUS": "4",
        "SFT_TENSOR_MODEL_PARALLEL_SIZE": "2",
        "SFT_CONTEXT_PARALLEL_SIZE": "1",
        "SFT_PIPELINE_MODEL_PARALLEL_SIZE": "1",
        "SFT_MAX_TOKENS_PER_GPU": "16384",
        "SFT_LR": "5e-6",
        "SFT_MIN_LR": "1e-6",
        "SFT_LR_DECAY_STYLE": "cosine",
        "SFT_LR_WARMUP_FRACTION": "0.03",
        "SFT_LR_WARMUP_INIT": "0.0",
        "SFT_ADAM_BETA1": "0.9",
        "SFT_ADAM_BETA2": "0.95",
        "SFT_ADAM_EPS": "1.0e-8",
        "SFT_WEIGHT_DECAY": "1e-4",
        "SFT_GRAD_CLIP": "1.0",
        "SFT_ZERO_STAGE": "1",
        "SFT_OPTIMIZER_CPU_OFFLOAD": "1",
        "SFT_RECOMPUTE_LOSS_FUNCTION": "1",
        "SFT_TRAIN_ENTRYPOINT": "train.py",
        "CHECKPOINT_REQUIRE_ROLLOUT_STATE": "1",
    }
    for name, expected in shared.items():
        assert stage[name] == child[name] == expected
    assert (stage["SFT_SIZE"], stage["SFT_NUM_ROLLOUT"], stage["SFT_FINAL_ROLLOUT_ID"]) == (
        "3000",
        "188",
        "187",
    )
    assert stage["SFT_LR_DECAY_ITERS"] == "188"
    assert stage["SFT_SAVE_INTERVAL"] == "24"
    assert stage["SFT_INITIAL_HF_DIR"] == ""
    assert (child["SFT_SIZE"], child["SFT_NUM_ROLLOUT"], child["SFT_FINAL_ROLLOUT_ID"]) == (
        "1500",
        "94",
        "93",
    )
    assert child["SFT_LR_DECAY_ITERS"] == "94"
    assert child["SFT_SAVE_INTERVAL"] == "24"
    assert child["SFT_INITIAL_HF_DIR"].endswith("/outputs/training/aime_correct_3000/weights/iter_0000187")


def test_submitter_is_strict_serial_revision_pinned_and_training_only() -> None:
    submit = (EXPERIMENT / "submit_training_only.sh").read_text(encoding="utf-8")
    assert 'for variant in "${CHILD_VARIANTS[@]}"' in submit
    assert "AIME_TRAINING_DRY_RUN_OK" in submit
    assert "jobs=16" in submit
    assert "AIME_EXPECTED_GIT_COMMIT=${submit_commit}" in submit
    assert '--dependency="${dependency}"' in submit
    assert "submit_hyak" not in submit
    assert "rsync_to_klone" not in submit
    for name in (
        "01_prepare_data.sbatch",
        "02_stage1_canary.sbatch",
        "03_train.sbatch",
        "04_continuation_canary.sbatch",
        "05_finalize.sbatch",
    ):
        text = (EXPERIMENT / name).read_text(encoding="utf-8")
        assert "runtime_repo_gate.py" in text, name
        assert "AIME_EXPECTED_GIT_COMMIT" in text, name
    train = (EXPERIMENT / "03_train.sbatch").read_text(encoding="utf-8")
    assert "recover_final_hf_from_full_state" in train
    assert "tools/convert_torch_dist_to_hf_bridge.py" in train
    assert 'HF_GATE=(python3 "${SCRIPT_DIR}/hf_checkpoint_gate.py")' in train
    assert "checkpoint_highest_durable_iteration" in train
    assert "MALFORMED_TRACKER_RECOVERY" in train
    assert "TORN_TRACKER_ROLLED_BACK" in train
    assert "NO_DURABLE_CHECKPOINT_RESTART_FROM_ZERO" in train
    assert "RESUME_FORWARD_STATE_QUARANTINED" in train
    assert "REUSE_COMPLETE_RECOVERED_HF" in train
    assert "manifest_already_present" in train
    assert "Variant is already finalized; refusing" not in train


def test_checkpoint_pruning_waits_for_rollout_state_and_torn_tracker_rolls_back(
    tmp_path: Path,
) -> None:
    save_dir = tmp_path / "full_state"
    rollout_dir = save_dir / "rollout"
    rollout_dir.mkdir(parents=True)
    prior = save_dir / "iter_0000023"
    current = save_dir / "iter_0000047"
    prior.mkdir()
    current.mkdir()
    (prior / ".metadata").write_bytes(b"prior")
    (current / ".metadata").write_bytes(b"current")
    (rollout_dir / "global_dataset_state_dict_23.pt").write_bytes(b"prior-state")
    (save_dir / "latest_checkpointed_iteration.txt").write_text("47\n", encoding="utf-8")

    utilities = ROOT / "examples/qwen3_8b_opd_tillicum/checkpoint_utils.sh"
    selected = subprocess.check_output(
        [
            "bash",
            "-c",
            'source "$1"; checkpoint_highest_durable_iteration "$2" 47',
            "bash",
            str(utilities),
            str(save_dir),
        ],
        text=True,
    ).strip()
    assert selected == "23"

    (save_dir / "latest_checkpointed_iteration.txt").write_bytes(b"")
    selected = subprocess.check_output(
        [
            "bash",
            "-c",
            'source "$1"; checkpoint_highest_durable_iteration "$2" 93',
            "bash",
            str(utilities),
            str(save_dir),
        ],
        text=True,
    ).strip()
    assert selected == "23"
    (save_dir / "latest_checkpointed_iteration.txt").write_text("47\n", encoding="utf-8")

    subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; export CHECKPOINT_REQUIRE_ROLLOUT_STATE=1; ' 'checkpoint_prune_old test "$2" background',
            "bash",
            str(utilities),
            str(save_dir),
        ],
        check=True,
    )
    assert prior.is_dir()
    assert current.is_dir()

    (rollout_dir / "global_dataset_state_dict_47.pt").write_bytes(b"current-state")
    selected = subprocess.check_output(
        [
            "bash",
            "-c",
            'source "$1"; checkpoint_highest_durable_iteration "$2" 47',
            "bash",
            str(utilities),
            str(save_dir),
        ],
        text=True,
    ).strip()
    assert selected == "47"

    empty_save = tmp_path / "empty_full_state"
    (empty_save / "rollout").mkdir(parents=True)
    (empty_save / "latest_checkpointed_iteration.txt").write_bytes(b"")
    no_candidate = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; checkpoint_highest_durable_iteration "$2" 93',
            "bash",
            str(utilities),
            str(empty_save),
        ],
        check=False,
    )
    assert no_candidate.returncode == 1


def test_rollout_dataset_state_is_published_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = RolloutDataSource.__new__(RolloutDataSource)
    source.args = SimpleNamespace(rollout_global_dataset=True, save=str(tmp_path))
    source.sample_offset = 64
    source.epoch_id = 1
    source.sample_group_index = 64
    source.sample_index = 64
    source.metadata = {"contract": "aime"}

    state_dir = tmp_path / "rollout"
    state_dir.mkdir()
    state_path = state_dir / "global_dataset_state_dict_47.pt"
    torch.save({"prior": True}, state_path)
    original_save = torch.save

    def interrupted_save(_state: dict, file_object) -> None:
        file_object.write(b"nonempty-but-truncated")
        file_object.flush()
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(torch, "save", interrupted_save)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        source.save(47)
    assert original_save is not torch.save
    assert torch.load(state_path, weights_only=False) == {"prior": True}
    assert not list(state_dir.glob(".global_dataset_state_dict_47.pt.*.tmp"))

    monkeypatch.setattr(torch, "save", original_save)
    source.save(47)
    saved = torch.load(state_path, weights_only=False)
    assert saved == {
        "sample_offset": 64,
        "epoch_id": 1,
        "sample_group_index": 64,
        "sample_index": 64,
        "metadata": {"contract": "aime"},
    }


def test_training_handoff_is_final_only_and_never_transfers_training_data() -> None:
    finalize = (EXPERIMENT / "finalize.py").read_text(encoding="utf-8")
    transfer = (EXPERIMENT / "rsync_to_klone.sh").read_text(encoding="utf-8")
    assert "trained_checkpoints=12" in transfer
    assert "eval_jsonl=2" in transfer
    assert "training_jsonl=0" in transfer
    assert "optimizer_state=0" in transfer
    assert '"training_datasets_transferred": False' in finalize
    assert "continue_wrong_unmasked_1500.jsonl" not in transfer
    assert "continue_correct_only_1500.jsonl" not in transfer
    assert "outputs/training/{variant}/weights/iter_" in finalize


def test_shell_surfaces_parse_and_walltimes_are_bounded() -> None:
    paths = sorted(EXPERIMENT.glob("*.sh")) + sorted(EXPERIMENT.glob("*.sbatch"))
    for path in paths:
        subprocess.run(["bash", "-n", str(path)], check=True)
    walltimes = {
        path.name: next(
            line.split("=", 1)[1]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("#SBATCH --time=")
        )
        for path in EXPERIMENT.glob("*.sbatch")
    }
    assert walltimes["03_train.sbatch"] == "04:00:00"
    assert all(value in {"02:00:00", "04:00:00"} for value in walltimes.values())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
