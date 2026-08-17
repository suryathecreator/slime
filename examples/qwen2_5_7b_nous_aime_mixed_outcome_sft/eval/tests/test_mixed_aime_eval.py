from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pytest

from examples.qwen2_5_7b_nous_aime_mixed_outcome_sft.eval import mixed_aime_eval

from examples.qwen2_5_7b_nous_aime_mixed_outcome_sft.eval.generate_mixed_aime import (
    recover_records,
    response_budget,
    sample_seed,
)
from examples.qwen2_5_7b_nous_aime_mixed_outcome_sft.eval.mixed_aime_eval import (
    CONTRACT_ID,
    DATASET_SHA256,
    PROMPTS,
    TARGETS,
    base_reuse_manifest,
    contract_id,
    final_iteration,
    metrics,
    normalized_rows,
    paired_delta,
    scorer_function,
    target_paths,
    validate_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ROOT = REPO_ROOT / "checkpoints" / "qwen2_5_7b_nous_aime_mixed_outcome_sft" / "v1" / CONTRACT_ID
FOUR_EPOCH_ROOT = (
    REPO_ROOT / "checkpoints" / "qwen2_5_7b_nous_aime_mixed_outcome_sft" / "v1" / "2f2b580a2f52b1b7"
)
PUBLISHED_RESULTS = (
    REPO_ROOT / "examples/qwen2_5_7b_nous_aime_mixed_outcome_sft/results/aime_mixed_outcome_sampled_v1"
)


def args() -> argparse.Namespace:
    return argparse.Namespace(repo_root=REPO_ROOT, experiment_root=EXPERIMENT_ROOT)


def four_epoch_args() -> argparse.Namespace:
    return argparse.Namespace(repo_root=REPO_ROOT, experiment_root=FOUR_EPOCH_ROOT)


def test_seed_contract_endpoints() -> None:
    assert sample_seed(0, 0) == 1234
    assert sample_seed(0, 59) == 1293
    assert sample_seed(15, 0) == 2134
    assert sample_seed(15, 59) == 2193
    with pytest.raises(ValueError):
        sample_seed(16, 0)


def test_dynamic_response_budget() -> None:
    assert response_budget(1) == 32767
    assert response_budget(858) == 31910
    with pytest.raises(RuntimeError):
        response_budget(32768)


def test_contract_identity_and_final_iteration_are_root_relative(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    config = root / "config"
    config.mkdir(parents=True)
    contract = config / "experiment_contract.json"
    contract.write_text('{"training":{"final_iteration":187}}\n')
    local_args = argparse.Namespace(experiment_root=root)
    assert contract_id(local_args) != CONTRACT_ID
    assert len(contract_id(local_args)) == 16
    assert final_iteration(local_args) == 187


def test_transferred_contract_and_rows_are_canonical() -> None:
    contract = validate_contract(args())
    rows = normalized_rows(args())
    assert contract["generation_count"] == 4800
    assert len(rows) == PROMPTS
    assert [row["global_eval_index"] for row in rows] == list(range(PROMPTS))
    assert {row["year"] for row in rows} == {2024, 2025}
    assert DATASET_SHA256 == {
        "aime24": "0c838ee6351d9636a7fdb62dbf9a1123c3fbf1cd13e61325aa892ac4343f6538",
        "aime25": "195dd659d4c20a3900875cfe7e83acc9a69c16a7c73ad6d6eb5f61c9fd30f70e",
    }


def test_all_checkpoint_paths_resolve() -> None:
    for target in TARGETS:
        model, manifest, identity = target_paths(args(), target)
        assert model.is_dir()
        assert manifest.is_file()
        assert len(identity) == 64


def test_four_epoch_base_reuse_is_exact_and_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = base_reuse_manifest(four_epoch_args(), EXPERIMENT_ROOT, PUBLISHED_RESULTS)
    assert manifest["destination_contract_id"] == "2f2b580a2f52b1b7"
    assert manifest["record_count"] == 960
    assert manifest["source_artifacts"] == {
        "final_audit_sha256": "4a1e5eef6979e3e84f9eb8ba84ccef5f83052414db6004a56770857078b02989",
        "scores_sha256": "4e5574ff922a898fd980347aed6eba2479c95f010cb0137c6ba91dd686bfb350",
        "summary_sha256": "c3045004010bbbbcaeb850c246c82075c687eb3ef45310f466f32e839d424495",
    }

    destination = tmp_path / "results"
    local_args = four_epoch_args()
    local_args.source_experiment_root = EXPERIMENT_ROOT
    local_args.source_result_root = PUBLISHED_RESULTS
    monkeypatch.setattr(mixed_aime_eval, "result_root", lambda _: destination)
    mixed_aime_eval.reuse_base(local_args)
    mixed_aime_eval.reuse_base(local_args)

    target = destination / "targets/base_qwen2_5_7b"
    assert mixed_aime_eval.sha256_file(target / "scores.jsonl") == manifest["source_artifacts"]["scores_sha256"]
    assert mixed_aime_eval.sha256_file(target / "summary.json") == manifest["source_artifacts"]["summary_sha256"]
    assert json.loads((target / "REUSE_PROVENANCE.json").read_text()) == manifest


def test_four_epoch_submitter_never_generates_or_finalizes_base() -> None:
    submitter = (REPO_ROOT / "examples/qwen2_5_7b_nous_aime_mixed_outcome_sft/eval/submit_hyak.sh").read_text()
    assert 'GENERATED_TARGETS=("${TARGETS[@]:1}")' in submitter
    assert 'for target in "${GENERATED_TARGETS[@]}"' in submitter
    assert "--reuse-target base_qwen2_5_7b" in submitter
    assert "reuse_base_cpu.sbatch" in submitter


def test_four_epoch_submission_metadata_separates_generated_and_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_args = four_epoch_args()
    local_args.array_job = [f"{target}=array-{index}" for index, target in enumerate(TARGETS[1:])]
    local_args.finalizer_job = [f"{target}=final-{index}" for index, target in enumerate(TARGETS[1:])]
    local_args.preflight_job = "preflight"
    local_args.canary_job = "canary"
    local_args.audit_job = "audit"
    local_args.git_commit = "a" * 40
    local_args.reuse_job = "reuse"
    local_args.reuse_target = TARGETS[0]
    local_args.reuse_source_experiment_root = EXPERIMENT_ROOT
    local_args.reuse_source_result_root = PUBLISHED_RESULTS
    monkeypatch.setattr(mixed_aime_eval, "control_root", lambda _: tmp_path)

    mixed_aime_eval.record_submission(local_args)
    metadata = json.loads((tmp_path / "submission.json").read_text())
    assert metadata["generation_count"] == 4800
    assert metadata["generated_completion_count"] == 3840
    assert metadata["reused_completion_count"] == 960
    assert metadata["generated_targets"] == list(TARGETS[1:])
    assert metadata["reused_targets"] == [TARGETS[0]]
    assert TARGETS[0] not in metadata["array_jobs"]
    assert TARGETS[0] not in metadata["finalizer_jobs"]
    assert metadata["reuse"]["job"] == "reuse"
    audit_edge = next(edge for edge in metadata["dependency_edges"] if edge["stage"] == "final_audit")
    assert audit_edge["afterok"] == ["final-0", "final-1", "final-2", "final-3", "reuse"]


def test_contracted_scorer_uses_last_complete_nonempty_box() -> None:
    score = scorer_function(args())
    correct = score(
        r"First $\boxed{12}$. Therefore $\boxed{\text{113}}$.",
        113,
        finish_reason="stop",
        stop_token_id=151645,
        cap_hit=False,
        problem_id="test",
    )
    assert correct["is_correct"] is True
    assert correct["official_candidate_raw"] == r"\text{113}"

    invalid_last = score(
        r"$\boxed{113}$ then $\boxed{x}$",
        113,
        finish_reason="stop",
        stop_token_id=151645,
        cap_hit=False,
        problem_id="test",
    )
    assert invalid_last["is_correct"] is False
    assert invalid_last["official_decision_reason"] == "integer_parse_failure"

    cap_hit = score(
        r"done $\boxed{113}$ trailing repetition",
        113,
        finish_reason="length",
        stop_token_id=None,
        cap_hit=True,
        problem_id="test",
    )
    assert cap_hit["is_correct"] is True
    assert cap_hit["cap_status"] is True


def make_row(*, correct: bool, tokens: int, draw: int, index: int) -> dict:
    return {
        "cap_hit": tokens == 32700,
        "draw_index": draw,
        "generated_token_count": tokens,
        "global_eval_index": index,
        "is_correct": correct,
        "is_scorable": True,
        "split": "held_in",
        "year": 2024,
    }


def test_metrics_and_paired_delta_use_independent_draws() -> None:
    left = [
        make_row(correct=False, tokens=10, draw=0, index=0),
        make_row(correct=True, tokens=20, draw=0, index=1),
    ]
    right = [
        make_row(correct=True, tokens=30, draw=0, index=0),
        make_row(correct=True, tokens=40, draw=0, index=1),
    ]
    summary = metrics(left)
    assert summary["accuracy"] == 0.5
    assert summary["n"] == 2
    assert summary["completion_tokens"]["mean"] == 15
    delta = paired_delta(left, right)
    assert delta["delta"] == 0.5
    assert math.isclose(delta["delta_mc_se"], 0.5)


def test_recover_records_repairs_only_truncated_tail(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_bytes(b'{"global_eval_index": 0}\n{"global_eval_index":')
    assert recover_records(path) == [{"global_eval_index": 0}]
    assert path.read_text() == '{"global_eval_index": 0}\n'

    path.write_text('{"global_eval_index":\n{"global_eval_index": 1}\n')
    with pytest.raises(RuntimeError):
        recover_records(path)
