from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.qwen3_8b_32b_full_repro import evaluate_math500 as evaluator
from examples.qwen3_8b_32b_full_repro.math500_scorer import SCORER_VERSION


def trace(correct: bool, text: str) -> dict:
    digest = evaluator.hashlib.sha256(text.encode()).hexdigest()
    return {
        "is_correct": correct,
        "official_extracted_candidate": "1",
        "official_candidate_sha256": digest,
        "generated_text_sha256": digest,
        "parser_verifier_exception": None,
        "think_closure": True,
        "answer_anywhere_diagnostic": "1",
        "answer_replacement_count": 0,
    }


def write_source(tmp_path: Path) -> tuple[Path, Path, list[dict]]:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n")
    source_dir = tmp_path / "source"
    problems = source_dir / "problems"
    problems.mkdir(parents=True)
    source_policy = {
        "model": str(model.resolve()),
        "seed": 1234,
        "scorer": SCORER_VERSION,
    }
    source_policy_sha = evaluator.policy_hash(source_policy)
    rows = []
    for index, cap_hit in enumerate((False, True, True)):
        ids = [10 + index, 20 + index] if cap_hit else [1]
        row = {
            "artifact_schema_version": 1,
            "stage": "source",
            "eval_index": index,
            "problem": f"problem {index}",
            "gold_answer": "1",
            "rendered_prompt": f"prompt {index}",
            "rendered_prompt_sha256": evaluator.hashlib.sha256(f"prompt {index}".encode()).hexdigest(),
            "input_token_ids": [100, 101, 102],
            "input_tokens": 3,
            "generated_token_ids": ids,
            "generated_tokens": len(ids),
            "response": "source",
            "finish_reason": "length" if cap_hit else "stop",
            "stop_token_id": None if cap_hit else 151645,
            "cap_hit": cap_hit,
            "seed": 1234 + index,
            "policy": source_policy,
            "policy_sha256": source_policy_sha,
            "scorer_trace": trace(not cap_hit, "source"),
        }
        path = problems / f"problem_{index:04d}.json"
        path.write_text(json.dumps(row, sort_keys=True) + "\n")
        rows.append(row)
    raw_artifacts = [
        {
            "eval_index": index,
            "path": str(problems / f"problem_{index:04d}.json"),
            "bytes": (problems / f"problem_{index:04d}.json").stat().st_size,
            "sha256": evaluator.sha256_file(problems / f"problem_{index:04d}.json"),
        }
        for index in range(3)
    ]
    summary = {
        "stage": "source",
        "policy_sha256": source_policy_sha,
        "correct": 1,
        "total": 3,
        "pass_at_1": 1 / 3,
        "cap_hits": 2,
        "raw_artifacts": raw_artifacts,
    }
    (source_dir / "summary.json").write_text(json.dumps(summary, sort_keys=True) + "\n")
    return source_dir, model, rows


def write_reruns(source_dir: Path, output_dir: Path, model: Path, rows: list[dict], *, mismatch: bool) -> None:
    output_problems = output_dir / "problems"
    output_problems.mkdir(parents=True)
    source_summary_sha = evaluator.sha256_file(source_dir / "summary.json")
    policy = evaluator.cap_proxy_policy(
        model=str(model),
        source_summary_sha256=source_summary_sha,
        source_policy_sha256=rows[0]["policy_sha256"],
        source_stage="source",
        target_context=20,
        safety_margin=2,
    )
    fingerprint = evaluator.policy_hash(policy)
    for row in rows[1:]:
        index = row["eval_index"]
        generated = list(row["generated_token_ids"]) + [30 + index]
        if mismatch and index == 1:
            generated[0] += 1
        source_path = source_dir / "problems" / f"problem_{index:04d}.json"
        rerun = {
            **row,
            "artifact_schema_version": 2,
            "stage": "proxy",
            "generated_token_ids": generated,
            "generated_tokens": len(generated),
            "response": "proxy",
            "finish_reason": "stop",
            "stop_token_id": 151645,
            "cap_hit": False,
            "effective_max_new_tokens": 15,
            "policy": policy,
            "policy_sha256": fingerprint,
            "source_artifact": {"sha256": evaluator.sha256_file(source_path)},
            "prefix_validation": {"exact_match": True},
            "scorer_trace": trace(True, f"proxy {index}"),
        }
        (output_problems / f"problem_{index:04d}.json").write_text(
            json.dumps(rerun, sort_keys=True) + "\n"
        )


def aggregate_args(source_dir: Path, output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        source_dir=str(source_dir),
        output_dir=str(output_dir),
        stage="proxy",
        expected=3,
        target_context=20,
        safety_margin=2,
        bootstrap_seed=1234,
    )


def test_dynamic_budget_and_prefix_gate() -> None:
    assert evaluator.dynamic_response_budget(100, 32768, 64) == 32604
    evaluator.require_exact_prefix([1, 2], [1, 2, 3], index=7)
    with pytest.raises(RuntimeError, match="token offset 1"):
        evaluator.require_exact_prefix([1, 2], [1, 9, 3], index=7)


def test_primary_policy_records_dynamic_target_context() -> None:
    args = SimpleNamespace(
        model="/tmp/model",
        dataset="MATH-500",
        revision="revision",
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        seed=1234,
        max_new_tokens=32768,
        target_context=32768,
        safety_margin=64,
        stop_token_ids=[151645, 151643],
    )
    policy = evaluator.policy(args, 40960)
    assert policy["target_total_context_tokens"] == 32768
    assert policy["model_native_context_tokens"] == 40960
    assert policy["temperature"] == 0.0
    assert policy["stop_token_ids"] == [151643, 151645]
    assert "target_context - rendered_prompt_tokens" in policy[
        "response_budget_mode"
    ]


def test_cap_proxy_reuses_natural_stops_and_publishes_after_exact_prefixes(tmp_path: Path) -> None:
    source_dir, model, rows = write_source(tmp_path)
    output_dir = tmp_path / "proxy"
    write_reruns(source_dir, output_dir, model, rows, mismatch=False)
    evaluator.aggregate_cap_proxy(aggregate_args(source_dir, output_dir))
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["correct"] == 3
    assert summary["exactness_gate"] == {
        "natural_stop_rows_reused": 1,
        "prefix_mismatches": 0,
        "prefixes_checked": 2,
        "prefixes_exact": 2,
        "publication_refused_on_any_prefix_mismatch": True,
        "source_cap_hits_selected": 2,
    }
    assert [row["provenance"] for row in summary["per_problem"]] == [
        "source_natural_stop",
        "cap_rerun",
        "cap_rerun",
    ]


def test_cap_proxy_refuses_to_publish_on_prefix_drift(tmp_path: Path) -> None:
    source_dir, model, rows = write_source(tmp_path)
    output_dir = tmp_path / "proxy"
    write_reruns(source_dir, output_dir, model, rows, mismatch=True)
    with pytest.raises(RuntimeError, match="prefix mismatch"):
        evaluator.aggregate_cap_proxy(aggregate_args(source_dir, output_dir))
    assert not (output_dir / "summary.json").exists()
