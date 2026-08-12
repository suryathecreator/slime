"""Regression tests for training-only launch and portable eval handoff."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from argparse import Namespace
from pathlib import Path

import pytest
from examples.qwen3_8b_openr1_math220k_masked_sft_eval_handoff import (
    axolotl_math500_eval,
    build_result_bundle,
)
from examples.qwen3_8b_openr1_math220k_masked_sft_eval_handoff import checkpoint_manifest as manifests
from examples.qwen3_8b_openr1_math220k_masked_sft_eval_handoff import (
    import_result_bundle,
)

HANDOFF = Path(__file__).resolve().parents[1]
EXAMPLES = HANDOFF.parent
TWO_K = EXAMPLES / "qwen3_8b_openr1_math220k_masked_sft_2k"
SFT_RUNNER = EXAMPLES / "qwen3_8b_opd_tillicum/04_run_sft_100k_8xh200.sbatch"
CONTAINER_EXEC = EXAMPLES / "qwen3_8b_opd_tillicum/container_exec.sh"


def test_checkpoint_identity_survives_a_path_change(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text('{"model_type":"qwen3"}\n')
    (source / "model.safetensors").write_bytes(b"weights")
    files = manifests.checkpoint_files(source)
    identity = manifests.content_identity(files)
    value = {
        "artifact_schema_version": manifests.SCHEMA_VERSION,
        "identity_scheme": manifests.IDENTITY_SCHEME,
        "checkpoint_manifest_sha256": identity,
        "files": files,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(value))
    destination = tmp_path / "different-system" / "checkpoint"
    shutil.copytree(source, destination)
    assert manifests.verify(manifest_path, destination) == identity


def test_checkpoint_verification_fails_on_changed_weight(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n")
    (checkpoint / "model.safetensors").write_bytes(b"original")
    files = manifests.checkpoint_files(checkpoint)
    value = {
        "artifact_schema_version": manifests.SCHEMA_VERSION,
        "identity_scheme": manifests.IDENTITY_SCHEME,
        "checkpoint_manifest_sha256": manifests.content_identity(files),
        "files": files,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(value))
    (checkpoint / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="differs from manifest"):
        manifests.verify(manifest_path, checkpoint)


def test_checkpoint_manifest_records_aime_continuation_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    output = tmp_path / "manifest.json"
    parent_identity = "a" * 64
    argv = [
        "checkpoint_manifest.py",
        "record",
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--family",
        "aime_generalization",
        "--variant",
        "random_mask_10",
        "--final-iteration",
        "46",
        "--contract-hash",
        "contract",
        "--parent-checkpoint-manifest-sha256",
        parent_identity,
        "--fresh-optimizer",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    manifests.main()
    value = json.loads(output.read_text())
    assert value["family"] == "aime_generalization"
    assert value["parent_checkpoint_manifest_sha256"] == parent_identity
    assert value["fresh_optimizer"] is True

    # Recording the same checkpoint and provenance remains idempotent.
    manifests.main()


def test_shared_sft_runner_guards_explicit_hf_initialization() -> None:
    runner = SFT_RUNNER.read_text()
    resume_branch = runner.index('if [[ -f "${SFT_SAVE_DIR}/latest_checkpointed_iteration.txt" ]]')
    initial_branch = runner.index('elif [[ -n "${SFT_INITIAL_HF_DIR:-}" ]]')
    assert resume_branch < initial_branch
    assert 'CKPT_HF_CHECKPOINT="${STUDENT_HF_DIR}"' in runner
    assert 'CKPT_HF_CHECKPOINT="${SFT_LOAD_DIR}"' in runner
    assert '--hf-checkpoint "${CKPT_HF_CHECKPOINT}"' in runner
    assert 'SFT_LOAD_KIND="initial_hf_weights"' in runner
    assert "base_megatron_torch_dist_initial|initial_hf_weights)" in runner
    assert "HF architecture differs from canonical" in runner
    assert "HF index references an invalid, missing, or empty shard" in runner
    assert "A fresh SFT optimizer will be initialized" in runner
    assert "SFT_INITIAL_HF_DIR" in CONTAINER_EXEC.read_text()


def test_inventory_covers_base_and_fourteen_trained_checkpoints() -> None:
    inventory = json.loads((HANDOFF / "checkpoint_sources.json").read_text())
    checkpoints = inventory["checkpoints"]
    assert len(checkpoints) == 15
    assert sum(item["full_sft"] is True for item in checkpoints) == 14
    assert {item["id"] for item in checkpoints if item["family"] == "40k"} == {
        "40k_correct_only",
        "40k_weighted_tau_0p20",
        "40k_unmasked",
    }
    assert len([item for item in checkpoints if item["family"] == "2k"]) == 11
    assert all(item["source_checkpoint"].startswith("/gpfs/") for item in checkpoints)
    assert all(item["source_manifest"].startswith("/gpfs/") for item in checkpoints)


def test_training_only_launcher_has_exact_order_and_no_eval_submission() -> None:
    text = (TWO_K / "submit_training_only.sh").read_text()
    first_2k = text.index("submit_serial prepare_2k_traces")
    last_2k = text.index('submit_serial "train_2k_${variant}"')
    first_40k = text.index("submit_serial train_40k_weighted_tau_0p20")
    last_40k = text.index("submit_serial train_40k_unmasked")
    assert first_2k < last_2k < first_40k < last_40k
    assert "04_eval_math500.sbatch" not in text
    assert "05_report.sbatch" not in text
    assert "train_40k_correct_only" not in text
    assert 'args+=(--dependency="afterok:${tail_job}")' in text


def test_retired_launcher_fails_closed() -> None:
    text = (TWO_K / "submit_completion_chain.sh").read_text()
    assert "retired" in text
    assert "exit 2" in text
    assert "sbatch" not in text


def test_one_h200_wrapper_keeps_exact_eval_policy() -> None:
    text = (HANDOFF / "run_one_h200.sbatch").read_text()
    assert "#SBATCH --gpus=h200:1" in text
    assert "#SBATCH --partition=ckpt-all" in text
    assert "--num-shards 1" in text
    assert "--temperature 0.0" in text
    assert "--top-p 1.0" in text
    assert "--top-k -1" in text
    assert "--target-context 32768" in text
    assert "--safety-margin 64" in text
    assert "--stop-token-ids 151643 151645" in text
    assert "--resume" in text


def test_available_axolotl_inventory_is_exactly_the_eleven_local_2k_runs() -> None:
    inventory = json.loads((HANDOFF / "available_eval_manifest.json").read_text())
    assert inventory["base_model"]["hf_repo"] == "Qwen/Qwen3-8B-Base"
    assert inventory["base_model"]["revision"] == ("49e3418fbbbca6ecbdf9608b4d22e5a407081db4")
    assert inventory["base_model"]["tokenizer_files"] == (axolotl_math500_eval.EXPECTED_BASE_TOKENIZER_FILES)
    assert inventory["axolotl"] == {
        "commit": "6b8f0e3314e3d162260cdc35d84741c3da163f30",
        "evaluator": "scripts/openr1_axolotl/evaluate_math_vllm.py",
        "repository": "../Axolotl-Masked-SFT",
    }
    assert inventory["sharding"] == {
        "chunk_size": 32,
        "count": 4,
        "layout": "contiguous",
        "problems_per_shard": 125,
    }
    assert {item["variant"] for item in inventory["checkpoints"]} == (axolotl_math500_eval.EXPECTED_VARIANTS)
    assert len(inventory["checkpoints"]) == 11
    assert all(
        item["model"].endswith(f"/outputs/training/{item['variant']}/weights/iter_0000009")
        for item in inventory["checkpoints"]
    )


def test_axolotl_worker_uses_direct_fast_checkpointable_path() -> None:
    worker = (HANDOFF / "run_axolotl_h200.sbatch").read_text()
    submitter = (HANDOFF / "submit_available_axolotl.sh").read_text()
    policy = json.loads((HANDOFF / "axolotl_eval_policy.json").read_text())
    assert "scripts/openr1_axolotl/evaluate_math_vllm.py" in worker
    assert "--max_tokens 32768" in worker
    assert "--max_model_len 32768" in worker
    assert "--chunk_size 32" in worker
    assert "--max_num_seqs 8" in worker
    assert "--gpu_memory_utilization 0.92" in worker
    assert "--require_math_verify" in worker
    assert "path --field tokenizer" in worker
    assert '--model_name "$TOKENIZER_DIR"' in worker
    assert "check-preflight" in worker
    assert "repair-shard" in worker
    assert "#SBATCH --gpus=h200:1" in worker
    assert "#SBATCH --requeue" in worker
    assert "#SBATCH --signal=B:USR1@600" in worker
    assert "#SBATCH --mail-user=suryadv@cs.washington.edu" in worker
    assert "#SBATCH --mail-type=END,FAIL" in worker
    assert "--array=0-3" in submitter
    assert "--array=0-3%" not in submitter
    assert policy["engine"]["enforce_eager"] is False
    assert policy["persistence"]["layout"] == ("four_contiguous_125_problem_shards")


def test_repair_shard_retains_only_paired_completed_rows(tmp_path: Path) -> None:
    eval_file = tmp_path / "eval.jsonl"
    output = tmp_path / "output"
    output.mkdir()
    expected = [
        {
            "problem_id": f"p{index:03d}",
            "problem": f"problem {index}",
            "correct_answer": str(index),
        }
        for index in range(125)
    ]
    eval_file.write_text(axolotl_math500_eval.jsonl_text(expected))
    predictions = [{"problem_id": f"p{index:03d}", "value": f"pred-{index}"} for index in range(5)]
    predictions.append({"problem_id": "not-in-this-shard", "value": "ignore"})
    raw = [{"problem_id": f"p{index:03d}", "value": f"raw-{index}"} for index in range(3)]
    raw.append({"problem_id": "p004", "value": "orphan"})
    (output / "predictions.jsonl").write_text(axolotl_math500_eval.jsonl_text(predictions))
    (output / "raw_generations.jsonl").write_text(axolotl_math500_eval.jsonl_text(raw))
    axolotl_math500_eval.repair_shard(Namespace(eval_file=str(eval_file), output_dir=str(output)))
    assert [row["problem_id"] for row in axolotl_math500_eval.read_jsonl(output / "predictions.jsonl")] == [
        "p000",
        "p001",
        "p002",
        "p004",
    ]
    assert [row["problem_id"] for row in axolotl_math500_eval.read_jsonl(output / "raw_generations.jsonl")] == [
        "p000",
        "p001",
        "p002",
        "p004",
    ]


def test_native_shard_validator_recomputes_axolotl_metrics(
    tmp_path: Path,
) -> None:
    eval_file = tmp_path / "eval.jsonl"
    output = tmp_path / "output"
    output.mkdir()
    eval_rows = [
        {
            "problem_id": f"p{index:03d}",
            "problem": f"problem {index}",
            "correct_answer": str(index),
        }
        for index in range(125)
    ]
    eval_file.write_text(axolotl_math500_eval.jsonl_text(eval_rows))
    predictions = [
        {
            "cap_hit": index % 10 == 0,
            "correct_answer": str(index),
            "finish_reason": "length" if index % 10 == 0 else "stop",
            "generated_token_count": index + 1,
            "is_correct": index % 2 == 0,
            "predicted_answer": None if index % 25 == 0 else str(index),
            "problem_id": f"p{index:03d}",
            "variant": "correct_only",
        }
        for index in range(125)
    ]
    raw = [
        {
            "generation": f"answer {index}",
            "problem_id": f"p{index:03d}",
            "variant": "correct_only",
        }
        for index in range(125)
    ]
    correct = sum(row["is_correct"] for row in predictions)
    cap_hits = sum(row["cap_hit"] for row in predictions)
    parse_failures = sum(row["predicted_answer"] is None for row in predictions)
    tokens = sum(row["generated_token_count"] for row in predictions)
    (output / "predictions.jsonl").write_text(axolotl_math500_eval.jsonl_text(predictions))
    (output / "raw_generations.jsonl").write_text(axolotl_math500_eval.jsonl_text(raw))
    (output / "metrics.json").write_text(
        json.dumps(
            {
                "accuracy": correct / 125,
                "average_generated_length": tokens / 125,
                "benchmark": "math500",
                "cap_hit_rate": cap_hits / 125,
                "eval_file": str(eval_file),
                "evaluator": "vllm",
                "load_mode": "full_sft",
                "num_eval_problems": 125,
                "parse_failure_rate": parse_failures / 125,
                "variant": "correct_only",
            }
        )
    )
    validated_predictions, validated_raw, _ = axolotl_math500_eval.validate_native_shard(
        eval_file=eval_file,
        output_dir=output,
        variant="correct_only",
    )
    assert len(validated_predictions) == 125
    assert len(validated_raw) == 125


def test_bundle_round_trip_writes_compact_repo_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    checkpoint_files = manifests.checkpoint_files(checkpoint)
    checkpoint_identity = manifests.content_identity(checkpoint_files)
    checkpoint_value = {
        "artifact_schema_version": manifests.SCHEMA_VERSION,
        "checkpoint_manifest_sha256": checkpoint_identity,
        "contract_hash": "5c3230a293f7acb2",
        "created_at": "2026-07-24T00:00:00+00:00",
        "family": "2k",
        "files": checkpoint_files,
        "final_iteration": 9,
        "full_sft": True,
        "identity_scheme": manifests.IDENTITY_SCHEME,
        "repo_commit": "a" * 40,
        "source_checkpoint": str(checkpoint),
        "variant": "random_mask_70",
    }
    checkpoint_manifest_path = tmp_path / "checkpoint_manifest.json"
    checkpoint_manifest_path.write_text(json.dumps(checkpoint_value))

    eval_dir = tmp_path / "eval"
    problem_dir = eval_dir / "problems"
    problem_dir.mkdir(parents=True)
    frozen_policy = json.loads((HANDOFF / "eval_policy.json").read_text())
    generated_policy = {
        "dataset": frozen_policy["dataset"]["name"],
        "revision": frozen_policy["dataset"]["revision"],
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "n": 1,
        "seed": 1234,
        "ignore_eos": True,
        "max_new_tokens": 32768,
        "stop_token_ids": [151643, 151645],
        "prompt_instruction": frozen_policy["prompt"]["instruction"],
        "enable_thinking": True,
        "target_total_context_tokens": 32768,
        "safety_margin": 64,
        "scorer": "math500_strict_boxed_scorer_v3",
        "response_budget_mode": (
            "min(configured_max_new_tokens, target_context - " "rendered_prompt_tokens - safety_margin)"
        ),
        "engine": {key: value for key, value in frozen_policy["engine"].items() if key != "vllm_version"},
    }
    policy_sha = hashlib.sha256(
        json.dumps(generated_policy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    raw_artifacts = []
    per_problem = []
    for index in range(500):
        row = {
            "eval_index": index,
            "model_checkpoint_manifest_sha256": checkpoint_identity,
            "policy": generated_policy,
            "policy_sha256": policy_sha,
            "scorer_trace": {
                "is_correct": index % 2 == 0,
                "scorer_version": "math500_strict_boxed_scorer_v3",
            },
            "stage": "2k_random_mask_70",
        }
        path = problem_dir / f"problem_{index:04d}.json"
        path.write_text(json.dumps(row, sort_keys=True) + "\n")
        raw_artifacts.append(
            {
                "eval_index": index,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": build_result_bundle.sha256_file(path),
            }
        )
        per_problem.append(
            {
                "eval_index": index,
                "is_correct": index % 2 == 0,
            }
        )
    summary = {
        "bootstrap_95_ci": [0.45, 0.55],
        "cap_hits": 0,
        "correct": 250,
        "eligible_final_answers": 500,
        "finish_reasons": {"stop": 500},
        "length": {"max": 100, "mean": 50.0, "median": 50, "p90": 90, "p95": 95},
        "parseable": 500,
        "pass_at_1": 0.5,
        "per_problem": per_problem,
        "policy_sha256": policy_sha,
        "raw_artifacts": raw_artifacts,
        "scorer_version": "math500_strict_boxed_scorer_v3",
        "stage": "2k_random_mask_70",
        "stop_tokens": {"151645": 500},
        "think_closed": 500,
        "total": 500,
    }
    summary_path = eval_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n")
    (eval_dir / "summary.sha256").write_text(build_result_bundle.sha256_file(summary_path) + "\n")
    bundle = eval_dir / "bundle"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_result_bundle.py",
            "--evaluation-dir",
            str(eval_dir),
            "--stage",
            "2k_random_mask_70",
            "--checkpoint-manifest",
            str(checkpoint_manifest_path),
            "--eval-policy",
            str(HANDOFF / "eval_policy.json"),
            "--output-dir",
            str(bundle),
        ],
    )
    build_result_bundle.main()

    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    (fake_repo / ".git").write_text("gitdir: elsewhere\n")
    fake_handoff = fake_repo / HANDOFF.relative_to(EXAMPLES.parent)
    fake_handoff.mkdir(parents=True)
    shutil.copy2(HANDOFF / "eval_policy.json", fake_handoff)
    shutil.copy2(HANDOFF / "checkpoint_sources.json", fake_handoff)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "import_result_bundle.py",
            "--bundle",
            str(bundle),
            "--repo-root",
            str(fake_repo),
        ],
    )
    import_result_bundle.main()
    imported = fake_repo / "examples/qwen3_8b_openr1_math220k_masked_sft_2k/results/random_mask_70"
    assert json.loads((imported / "RESULTS.json").read_text())["correct"] == 250
    assert json.loads((imported / "PROVENANCE.json").read_text())["raw_problem_count"] == 500
    assert not (imported / "problems").exists()
