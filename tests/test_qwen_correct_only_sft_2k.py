from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.qwen_correct_only_openr1_math220k_sft_2k.audit_runtime_batches import (
    representative_dp_payloads,
)
from examples.qwen_correct_only_openr1_math220k_sft_2k.correct_only_sft_rollout import (
    generate_rollout,
)
from examples.qwen_correct_only_openr1_math220k_sft_2k.data_utils import (
    build_training_record,
)
from examples.qwen_correct_only_openr1_math220k_sft_2k.experiment import (
    RUN_ORDER,
    load_contract,
)
from examples.qwen_correct_only_openr1_math220k_sft_2k.finalize import (
    canary_evidence_root,
    trained_checkpoint,
)
from examples.qwen_correct_only_openr1_math220k_sft_2k.prepare_data import (
    deterministic_order,
    eligible_ref,
)
from examples.qwen_correct_only_openr1_math220k_sft_2k.validate_schedule import (
    main as validate_schedule,
)


NUM_GPUS = 0
EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "qwen_correct_only_openr1_math220k_sft_2k"
)


class FakeTokenizer:
    def __init__(self, reject_long: bool = False):
        self.reject_long = reject_long

    def apply_chat_template(self, messages, **kwargs):
        del messages, kwargs
        return [151644, 77091, 198]

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == "<|im_end|>\n":
            return [151645, 198]
        if self.reject_long and "LONG" in text:
            return [17] * 8193
        return [17] * 12

    def decode(self, tokens, skip_special_tokens=False):
        del skip_special_tokens
        if tokens == [151644, 77091, 198]:
            return "<|im_start|>assistant\n"
        return "x"


def selected() -> dict:
    return {
        "assistant_trace": "<think>work</think>\\boxed{1}",
        "generation_index": 0,
        "problem": "1+0?",
        "problem_id": "p1",
        "source_row_index": 4,
        "trace_id": "p1__row_0000004__gen_000",
    }


def test_contract_has_only_base_and_correct_only_with_shared_8k_selection():
    contract = load_contract()
    assert contract["variants"] == ["base", "correct_only"]
    assert contract["training"]["global_batch_size"] == 64
    assert contract["training"]["epochs"] == 4
    assert contract["training"]["optimizer_updates"] == 125
    assert contract["training"]["learning_rate"] == 5e-6
    assert contract["training"]["min_learning_rate"] == 1e-6
    eight_k = [key for key in RUN_ORDER if key.endswith("_8k")]
    assert len(eight_k) == 4
    assert {contract["runs"][key]["selection"] for key in eight_k} == {"8k_shared"}
    assert "same ordered 2000 trace IDs" in contract["selection"]["shared_8k_policy"]
    assert contract["runs"]["qwen2_5_3b_16k"]["max_sequence_length"] == 18432


def test_exact_token_record_fully_supervises_assistant_and_im_end():
    record = build_training_record(
        FakeTokenizer(), selected(), {}, 10240, 8192, "qwen2_5_3b"
    )
    metadata = record["metadata"]
    assert record["input_ids"][:3] == [151644, 77091, 198]
    assert record["input_ids"][-2:] == [151645, 198]
    assert metadata["assistant_token_count"] == 12
    assert metadata["response_length"] == 14
    assert metadata["loss_weights"] == [1.0] * 13 + [0.0]
    assert 151643 not in record["input_ids"]


def test_shared_eligibility_rejects_a_trace_that_fails_any_model_tokenizer():
    contract = load_contract()
    row = {
        "uuid": "problem-1",
        "problem": "problem",
        "answer": "1",
        "generations": [
            "<think>LONG work</think>\\boxed{1}",
            "<think>short work</think>\\boxed{1}",
        ],
        "correctness_math_verify": [True, True],
        "is_reasoning_complete": [True, True],
    }
    tokenizers = {
        "qwen2_5_3b": FakeTokenizer(reject_long=True),
        "qwen2_5_7b": FakeTokenizer(),
        "qwen3_4b": FakeTokenizer(),
        "qwen3_8b": FakeTokenizer(),
    }
    from collections import Counter

    ref = eligible_ref(
        row,
        0,
        "8k_shared",
        tuple(tokenizers),
        tokenizers,
        contract,
        8192,
        10240,
        42,
        Counter(),
    )
    assert ref is not None
    assert ref["generation_index"] == 1
    assert set(ref["per_model_lengths"]) == set(tokenizers)


def test_seeded_generation_order_is_reproducible():
    assert deterministic_order([0, 1, 2, 3], 42, 17, "8k_shared") == deterministic_order(
        [0, 1, 2, 3], 42, 17, "8k_shared"
    )
    assert deterministic_order([0, 1, 2, 3], 42, 17, "8k_shared") != deterministic_order(
        [0, 1, 2, 3], 43, 17, "8k_shared"
    )


def test_rollout_adapter_preserves_full_correct_only_weights():
    record = build_training_record(
        FakeTokenizer(), selected(), {}, 10240, 8192, "qwen2_5_3b"
    )
    sample = SimpleNamespace(prompt=record["input_ids"], metadata=record["metadata"])

    class Buffer:
        def get_samples(self, size):
            assert size == 1
            return [[sample]]

    args = SimpleNamespace(rollout_global_dataset=True, rollout_batch_size=1)
    result = generate_rollout(args, 0, Buffer())
    assert result == [[sample]]
    assert sample.tokens == record["input_ids"]
    assert sample.loss_mask == record["metadata"]["loss_weights"]
    assert sample.reward == 0


def test_four_epoch_schedule_resolves_to_125_global_updates(tmp_path, monkeypatch):
    data = tmp_path / "data.jsonl"
    data.write_text(
        "".join(json.dumps({"input_ids": list(range(32))}) + "\n" for _ in range(2000)),
        encoding="utf-8",
    )
    output = tmp_path / "schedule.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_schedule.py",
            "--data",
            str(data),
            "--output",
            str(output),
            "--dp-size",
            "4",
            "--max-tokens-per-gpu",
            "128",
        ],
    )
    validate_schedule()
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["global_examples"] == 8000
    assert value["optimizer_updates"] == 125
    assert value["examples_per_global_update"] == 64
    assert value["microbatches_per_rank"] == 500
    assert value["microbatches_per_update_histogram"] == {"4": 125}


def test_gpu_jobs_respect_cluster_resource_policy():
    for sbatch_path in EXPERIMENT_DIR.glob("*.sbatch"):
        text = sbatch_path.read_text(encoding="utf-8")
        gpu_match = re.search(r"^#SBATCH --gres=gpu:h200:(\d+)$", text, re.MULTILINE)
        assert gpu_match is not None
        cpu_match = re.search(r"^#SBATCH --cpus-per-task=(\d+)$", text, re.MULTILINE)
        assert cpu_match is not None
        gpu_count = int(gpu_match.group(1))
        assert int(cpu_match.group(1)) <= 8 * gpu_count, sbatch_path
        memory_match = re.search(r"^#SBATCH --mem=(\d+)(?:G)?$", text, re.MULTILINE)
        assert memory_match is not None
        memory_gb = int(memory_match.group(1))
        assert memory_gb == 0 or memory_gb <= 240 * gpu_count, sbatch_path


def test_submission_preflights_every_unique_sbatch_before_creating_manifest_root():
    text = (EXPERIMENT_DIR / "submit_training_only.sh").read_text(encoding="utf-8")
    assert "sbatch --test-only" in text
    assert text.index("sbatch --test-only") < text.index('mkdir -p "${MANIFEST_ROOT}"')


def test_data_repair_reuses_artifacts_and_rewires_only_first_blocked_job():
    data_job = (EXPERIMENT_DIR / "01_prepare_data.sbatch").read_text(encoding="utf-8")
    repair = (EXPERIMENT_DIR / "resubmit_after_data_schedule_audit.sh").read_text(
        encoding="utf-8"
    )
    assert 'REUSE_PREPARED_DATA="${REUSE_PREPARED_DATA:-0}"' in data_job
    assert "DATA_PREP_REUSE_EXISTING" in data_job
    assert "readonly FAILED_DATA_JOB=212331" in repair
    assert "readonly BLOCKED_CANARY_JOB=212332" in repair
    assert '--export=ALL,REUSE_PREPARED_DATA=1' in repair
    assert 'JobId="${BLOCKED_CANARY_JOB}"' in repair
    assert 'Dependency="afterok:${replacement_job}"' in repair
    assert 'scancel "${replacement_job}"' in repair


def test_runtime_memory_profile_resolves_for_every_run():
    command = (
        'source "$1"; shift; '
        'for run_key in "$@"; do resolve_run "${run_key}"; '
        "printf '%s=%s:%s:%s\\n' \"${run_key}\" "
        '"${SFT_TENSOR_MODEL_PARALLEL_SIZE}" "${SFT_MAX_TOKENS_PER_GPU}" '
        '"${SFT_OPTIMIZER_CPU_OFFLOAD}"; done; '
        "printf 'schedule_audit_dir=%s\\n' \"${SCHEDULE_AUDIT_DIR}\""
    )
    output = subprocess.check_output(
        ["bash", "-c", command, "bash", str(EXPERIMENT_DIR / "env.sh"), *RUN_ORDER],
        text=True,
    )
    resolved = dict(line.split("=", 1) for line in output.splitlines())
    assert resolved.pop("schedule_audit_dir").endswith("/schedule_audits/memory_r3")
    assert resolved == {
        "qwen2_5_3b_8k": "1:10240:1",
        "qwen2_5_3b_16k": "1:16384:1",
        "qwen2_5_7b_8k": "2:16384:1",
        "qwen3_4b_8k": "1:9216:0",
        "qwen3_8b_8k": "2:16384:1",
    }


def test_canary_oom_repair_isolates_retry_and_rewires_only_first_blocked_job():
    canary = (EXPERIMENT_DIR / "02_canary.sbatch").read_text(encoding="utf-8")
    train = (EXPERIMENT_DIR / "03_train.sbatch").read_text(encoding="utf-8")
    repair = (EXPERIMENT_DIR / "resubmit_after_canary_oom.sh").read_text(
        encoding="utf-8"
    )
    assert '/attempts/${CANARY_ATTEMPT}' in canary
    assert '${SCHEDULE_AUDIT_DIR}/${RUN_KEY}.json' in train
    assert "readonly FAILED_CANARY_JOB=212332" in repair
    assert "readonly BLOCKED_CANARY_JOB=212333" in repair
    assert "REUSE_PREPARED_DATA=1" in repair
    assert "CANARY_ATTEMPT=oom_r1" in repair
    assert 'JobId="${BLOCKED_CANARY_JOB}"' in repair
    assert 'Dependency="afterok:${replacement_canary_job}"' in repair
    assert 'scancel "${replacement_canary_job}"' in repair


def test_tp_runtime_audit_validates_replicas_and_selects_one_payload_per_dp_rank(
    tmp_path,
):
    import torch

    details = tmp_path / "details"
    train_data = details / "train_data"
    train_data.mkdir(parents=True)

    def write_payload(rank, tokens):
        torch.save(
            {
                "rank": rank,
                "rollout_id": 0,
                "rollout_data": {
                    "num_microbatches": [2],
                    "tokens": tokens,
                    "response_lengths": [1],
                    "loss_masks": [[1.0]],
                },
            },
            train_data / f"0_{rank}.pt",
        )

    write_payload(0, [[10, 11]])
    write_payload(1, [[10, 11]])
    write_payload(2, [[20, 21]])
    write_payload(3, [[20, 21]])
    representatives = representative_dp_payloads(details, dp_size=2, tensor_parallel_size=2)
    assert [payload["rank"] for payload in representatives] == [0, 2]

    write_payload(1, [[10, 12]])
    with pytest.raises(ValueError, match="tensor-parallel replica drift"):
        representative_dp_payloads(details, dp_size=2, tensor_parallel_size=2)


def test_tp_audit_repair_reuses_checkpoints_and_rewires_only_first_blocked_job():
    canary = (EXPERIMENT_DIR / "02_canary.sbatch").read_text(encoding="utf-8")
    resume = (EXPERIMENT_DIR / "02b_resume_canary_audit.sbatch").read_text(encoding="utf-8")
    repair = (EXPERIMENT_DIR / "resubmit_after_tp_audit_failure.sh").read_text(encoding="utf-8")
    assert '--tensor-parallel-size "${SFT_TENSOR_MODEL_PARALLEL_SIZE}"' in canary
    assert 'readonly CANARY_INPUT_ROOT="${OUTPUT_ROOT}/canaries/${RUN_KEY}"' in resume
    assert 'readonly AUDIT_ROOT="${CANARY_INPUT_ROOT}/attempts/${CANARY_AUDIT_ATTEMPT}"' in resume
    assert "audit_runtime_batches.py" in resume
    assert "canary_generate.py" in resume
    assert "04_run_sft_100k_8xh200.sbatch" not in resume
    assert "readonly FAILED_CANARY_JOB=212334" in repair
    assert "readonly BLOCKED_CANARY_JOB=212335" in repair
    assert 'CANARY_AUDIT_ATTEMPT="${AUDIT_ATTEMPT}"' in repair
    assert 'JobId="${BLOCKED_CANARY_JOB}"' in repair
    assert 'Dependency="afterok:${replacement_job}"' in repair
    assert 'scancel "${replacement_job}"' in repair


def test_qwen3_4b_oom_repair_isolates_retry_and_rewires_only_blocked_job():
    repair = (EXPERIMENT_DIR / "resubmit_after_qwen3_4b_oom.sh").read_text(
        encoding="utf-8"
    )
    assert "readonly FAILED_CANARY_JOB=212335" in repair
    assert "readonly BLOCKED_CANARY_JOB=212336" in repair
    assert "readonly NEXT_JOB=212337" in repair
    assert '[[ "${SFT_MEMORY_PROFILE}" == "memory_r2" ]]' in repair
    assert "1:9216:0" in repair
    assert "REUSE_PREPARED_DATA=1" in repair
    assert "CANARY_ATTEMPT=oom_r1" in repair
    assert 'JobId="${BLOCKED_CANARY_JOB}"' in repair
    assert 'Dependency="afterok:${replacement_canary_job}"' in repair
    assert 'scancel "${replacement_canary_job}"' in repair


def test_qwen3_8b_audit_repair_reuses_canary_and_replaces_stale_training_tail():
    resume = (EXPERIMENT_DIR / "02b_resume_canary_audit.sbatch").read_text(
        encoding="utf-8"
    )
    train = (EXPERIMENT_DIR / "03_train.sbatch").read_text(encoding="utf-8")
    repair = (
        EXPERIMENT_DIR / "resubmit_after_qwen3_8b_audit_failure.sh"
    ).read_text(encoding="utf-8")
    assert '--tensor-parallel-size "${SFT_TENSOR_MODEL_PARALLEL_SIZE}"' in resume
    assert "04_run_sft_100k_8xh200.sbatch" not in resume
    assert 'test -f "${SCHEDULE_AUDIT_DIR}/${RUN_KEY}.json"' in train
    assert "readonly FAILED_CANARY_JOB=212336" in repair
    assert "readonly AUDIT_ATTEMPT=snapshot_audit_r1" in repair
    assert "readonly -a STALE_JOBS=(212337 212338 212339 212340 212341 212342)" in repair
    assert 'CANARY_AUDIT_ATTEMPT="${AUDIT_ATTEMPT}"' in repair
    assert '"${SCRIPT_DIR}/02b_resume_canary_audit.sbatch"' in repair
    assert '"${SCRIPT_DIR}/03_train.sbatch"' in repair
    assert "for run_key in \"${RUN_KEYS[@]}\"" in repair
    assert repair.index("replacement_chain_verified=1") < repair.index(
        'scancel "${STALE_JOBS[@]}"'
    )
    assert "stale_retirement_started=1" in repair
    assert '"${SCRIPT_DIR}/04_finalize.sbatch"' in repair


def test_shared_runner_honors_explicit_rollout_count():
    runner = (
        EXPERIMENT_DIR.parent / "qwen3_8b_opd_tillicum/04_run_sft_100k_8xh200.sbatch"
    ).read_text(encoding="utf-8")
    assert 'if [[ -n "${SFT_NUM_ROLLOUT:-}" ]]' in runner
    assert 'SFT_ARGS+=(--num-rollout "${SFT_NUM_ROLLOUT}")' in runner
    assert 'SFT_ARGS+=(--num-epoch "${SFT_NUM_EPOCH}")' in runner


def test_training_prefix_canary_replays_five_full_dataset_updates():
    prefix = (EXPERIMENT_DIR / "02c_training_prefix_canary.sbatch").read_text(
        encoding="utf-8"
    )
    assert "qwen2_5_3b_8k|qwen2_5_3b_16k" in prefix
    assert "export SFT_NUM_ROLLOUT=5" in prefix
    assert "export SFT_FINAL_ROLLOUT_ID=4" in prefix
    assert 'export SFT_PARQUET=' not in prefix
    assert "prepare_canaries.py" not in prefix
    assert "canary_generate.py" not in prefix


def test_training_retry_is_isolated_and_finalizer_uses_manifest_provenance():
    train = (EXPERIMENT_DIR / "03_train.sbatch").read_text(encoding="utf-8")
    finalize = (EXPERIMENT_DIR / "finalize.py").read_text(encoding="utf-8")
    finalize_job = (EXPERIMENT_DIR / "04_finalize.sbatch").read_text(encoding="utf-8")
    assert 'TRAINING_ATTEMPT="${TRAINING_ATTEMPT:-}"' in train
    assert 'export TRAINING_ROOT="${CORRECT_ONLY_ROOT}/attempts/${TRAINING_ATTEMPT}"' in train
    assert '"${TRAINED_HANDOFF_MANIFEST}"' in train
    assert 'checkpoint = Path(value["source_checkpoint"]).resolve()' in finalize
    assert 'evidence_root = canary_evidence_root(args.experiment_root, run_key)' in finalize
    assert '"qwen3_8b_8k": ("qwen3_8b_audit_repair.json", "audit_root")' in finalize
    assert '--memory-profile "${SFT_MEMORY_PROFILE}"' in finalize_job


def test_finalizer_accepts_recorded_retry_paths_but_rejects_escapes(tmp_path):
    experiment_root = tmp_path / "experiment"
    manifest = experiment_root / "handoff/checkpoints/qwen2_5_3b_8k.json"
    manifest.parent.mkdir(parents=True)
    checkpoint = (
        experiment_root
        / "outputs/training/qwen2_5_3b_8k/correct_only/attempts/oom_r1"
        / "weights/iter_0000124"
    )
    manifest.write_text(json.dumps({"source_checkpoint": str(checkpoint)}))
    assert trained_checkpoint(manifest, experiment_root, "qwen2_5_3b_8k") == checkpoint

    manifest.write_text(
        json.dumps({"source_checkpoint": str(tmp_path / "outside/weights/iter_0000124")})
    )
    with pytest.raises(ValueError, match="escapes run root"):
        trained_checkpoint(manifest, experiment_root, "qwen2_5_3b_8k")


def test_finalizer_resolves_repaired_canary_evidence_but_rejects_escapes(tmp_path):
    experiment_root = tmp_path / "experiment"
    repair_manifest = experiment_root / "manifests/canary_oom_repair.json"
    repair_manifest.parent.mkdir(parents=True)
    retry_root = experiment_root / "outputs/canaries/qwen2_5_3b_8k/attempts/oom_r1"
    repair_manifest.write_text(json.dumps({"retry_root": str(retry_root)}))
    assert canary_evidence_root(experiment_root, "qwen2_5_3b_8k") == retry_root

    repair_manifest.write_text(json.dumps({"retry_root": str(tmp_path / "outside")}))
    with pytest.raises(ValueError, match="escapes run root"):
        canary_evidence_root(experiment_root, "qwen2_5_3b_8k")


def test_qwen2_5_3b_oom_repair_replays_prefixes_and_replaces_frozen_tail():
    repair = (EXPERIMENT_DIR / "resubmit_after_qwen2_5_3b_oom.sh").read_text(
        encoding="utf-8"
    )
    assert "readonly COMPLETED_AUDIT_JOB=212698" in repair
    assert "readonly FAILED_TRAINING_JOB=212699" in repair
    assert "readonly PREFIX_ATTEMPT=optimizer_offload_r1" in repair
    assert "readonly -a STALE_JOBS=(212700 212701 212702 212703 212704)" in repair
    assert "SFT max tokens per GPU: 16384" in repair
    assert "Tried to allocate 4.64 GiB" in repair
    assert "1:10240:1" in repair
    assert "1:16384:1" in repair
    assert '"${SCRIPT_DIR}/02c_training_prefix_canary.sbatch"' in repair
    assert "TRAINING_ATTEMPT=oom_r1" in repair
    assert repair.index("replacement_chain_verified=1") < repair.index(
        'scancel "${STALE_JOBS[@]}"'
    )
    assert '"${SCRIPT_DIR}/04_finalize.sbatch"' in repair


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
