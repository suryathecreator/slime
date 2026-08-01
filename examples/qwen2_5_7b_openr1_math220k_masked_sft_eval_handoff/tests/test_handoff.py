"""Static handoff contract checks independent of checkpoint availability."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from examples.qwen2_5_7b_openr1_math220k_masked_sft_eval_handoff import (
    axolotl_math500_eval as qwen2_native,
    llama_math500_eval as strict,
)


HANDOFF = Path(__file__).resolve().parents[1]


def test_available_manifest_covers_base_plus_eleven_variants():
    value = json.loads((HANDOFF / "available_eval_manifest.json").read_text())
    variants = [item["variant"] for item in value["checkpoints"]]
    assert len(variants) == len(set(variants)) == 12
    assert variants[0] == "base_7b"
    assert value["base_model"]["revision"] == "d149729398750b98c0af14eb82c78cfe92750796"


def test_eval_policy_is_greedy_and_uses_both_qwen_stops():
    policy = json.loads((HANDOFF / "axolotl_eval_policy.json").read_text())
    assert policy["decoding"]["temperature"] == 0.0
    assert policy["decoding"]["stop_token_ids"] == [151643, 151645]
    assert policy["decoding"]["max_tokens"] == 32768
    assert policy["engine"]["max_model_len"] == 32768


def test_qwen2_overlay_uses_the_pinned_base_tokenizer():
    assert (
        qwen2_native.controller.prepare_tokenizer_overlay
        is qwen2_native.prepare_qwen2_tokenizer_overlay
    )
    source = (HANDOFF / "axolotl_math500_eval.py").read_text()
    assert 'item_for_variant(inventory, "base_7b")' in source
    assert "ignored_checkpoint_tokenizers" in source


def test_source_and_destination_inventories_match():
    source = json.loads((HANDOFF / "checkpoint_sources.json").read_text())
    destination = json.loads((HANDOFF / "available_eval_manifest.json").read_text())
    assert {item["variant"] for item in source["checkpoints"]} == {
        item["variant"] for item in destination["checkpoints"]
    }


def test_llama_gate_policy_is_pinned_greedy_fp8_weights_bf16_kv_and_single_h200_sized():
    policy = json.loads((HANDOFF / "llama_gate_policy.json").read_text())
    assert policy["model"] == {
        "hf_repo": "nvidia/Llama-3.3-70B-Instruct-FP8",
        "revision": "68579f6750089fc4ccf3d3b58d9f4648d1eed615",
    }
    assert policy["gate"]["temperature"] == 0.0
    assert policy["gate"]["response_format"] == "json_schema"
    assert policy["gate"]["adjudication_max_tokens"] == 256
    assert policy["gate"]["extraction_max_tokens"] == 128
    assert policy["engine"]["quantization"] == "modelopt"
    assert policy["engine"]["kv_cache_dtype"] == "auto"
    assert policy["engine"]["calculate_kv_scales"] is False
    assert policy["engine"]["tensor_parallel_size"] == 1
    assert policy["runtime"]["pyparsing"] == "3.3.2"
    assert policy["runtime"]["xgrammar"] == "0.1.23"


def test_gate_prompt_has_no_gold_or_correctness_judgment():
    prompt = (HANDOFF / "llama_gate_prompt.txt").read_text()
    lowered = prompt.lower()
    assert "{{problem}}" in lowered and "{{model_response}}" in lowered
    assert "gold answer" not in lowered
    assert "correct_answer" not in lowered
    assert "not judging whether the mathematical answer is correct" in lowered
    assert "do not generate or copy answer text" in lowered


def test_gate_input_schema_is_an_exact_no_gold_allow_list():
    assert strict.ALLOWED_GATE_INPUT_FIELDS == {
        "problem",
        "problem_id",
        "response",
        "response_sha256",
        "shard_index",
        "variant",
    }
    assert not any("gold" in field or "answer" in field for field in strict.ALLOWED_GATE_INPUT_FIELDS)


def test_definitive_gate_validation_rejects_null_commitment_fields_cleanly():
    decision = {
        "definitive": True,
        "final_answer_raw": None,
        "boxed_final_answer": True,
        "format_compliant": True,
        "answer_form": "scalar",
        "supporting_quote": None,
        "later_conflict_quote": None,
        "reason_code": "single_committed_answer",
    }
    with pytest.raises(RuntimeError, match="lacks answer or supporting quote"):
        strict.validate_decision(decision, r"Final: \\boxed{2}")


def test_span_selector_builds_answer_and_quote_only_from_exact_source_spans():
    response = "Reasoning.\nSo the final answer is:\n\\boxed{(3, \\frac{\\pi}{2})}\n"
    regions = strict.response_regions(response)
    region = next(region for region in regions if r"\boxed" in region["text"])
    candidates = strict.answer_candidates(response, region)
    candidate = next(value for value in candidates if value["boxed"])
    boundaries = strict.lexer_boundaries(response, region, candidates)
    adjudication = {
        "definitive": True,
        "boxed_final_answer": True,
        "format_compliant": True,
        "answer_form": "tuple_or_list",
        "evidence_region_id": region["id"],
        "reason_code": "single_committed_answer",
    }
    extraction = strict.selected_extraction_span(
        {
            "extraction_mode": "candidate",
            "candidate_id": candidate["id"],
            "start_boundary_id": None,
            "end_boundary_id": None,
        },
        response,
        adjudication,
        candidates,
        boundaries,
    )
    decision = strict.terminal_decision(response, adjudication, regions, extraction)
    assert decision["final_answer_raw"] == r"(3, \frac{\pi}{2})"
    assert decision["supporting_quote"] in response
    assert decision["final_answer_raw"] == response[extraction["start"] : extraction["end"]]


def test_constrained_stages_have_no_answer_text_output_fields():
    assert "final_answer_raw" not in strict.ADJUDICATION_FIELDS
    assert "final_answer_raw" not in strict.EXTRACTION_FIELDS
    assert "supporting_quote" not in strict.ADJUDICATION_FIELDS
    assert "candidate_id" in strict.EXTRACTION_FIELDS


def test_box_extraction_removes_only_wrapper_and_preserves_inner_whitespace():
    response = r"Final: \boxed{  \frac{1}{8}  }."
    region = strict.response_regions(response)[0]
    candidate = next(
        value for value in strict.answer_candidates(response, region) if value["boxed"]
    )
    assert candidate["text"] == r"  \frac{1}{8}  "


def test_gate_validation_never_repairs_the_extracted_answer():
    response = r"Final: \boxed{\frac{1}{2}}"
    decision = {
        "definitive": True,
        "final_answer_raw": r"\dfrac{1}{2}",
        "boxed_final_answer": True,
        "format_compliant": True,
        "answer_form": "expression",
        "supporting_quote": response,
        "later_conflict_quote": None,
        "reason_code": "single_committed_answer",
    }
    with pytest.raises(RuntimeError, match="final_answer_raw is not copied exactly"):
        strict.validate_decision(decision, response)


def test_llama_runtime_caches_are_isolated_per_array_task(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "12345")
    for name in ("VLLM_CACHE_ROOT", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))

    strict.isolate_vllm_runtime_env(tmp_path, "base_7b", 2)
    for name in ("VLLM_CACHE_ROOT", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        isolated = Path(os.environ[name])
        assert isolated.name == "12345_2"
        assert isolated.is_dir()

    strict.isolate_vllm_runtime_env(tmp_path, "base_7b", 2)
    assert Path(os.environ["VLLM_CACHE_ROOT"]).name == "12345_2"
    assert os.environ["VLLM_NO_USAGE_STATS"] == "1"


def test_llama_gate_wrapper_exports_pythonpath_before_controller_lookup():
    wrapper = (HANDOFF / "run_llama_gate_h200.sbatch").read_text()
    assert wrapper.index('export PYTHONPATH="$REPO_ROOT:') < wrapper.index(
        'GATE_ROOT="$("$PYTHON_BIN" "$CONTROL"'
    )


def test_gold_classifier_routes_all_required_structures():
    assert strict.classify_gold("id", "Enter the ordered pair.", r"(3,\frac{\pi}{2})")[
        "gold_type"
    ] == "ordered_tuple"
    assert strict.classify_gold("id", "Find all roots.", r"3,5,7")["gold_type"] == "finite_set"
    assert strict.classify_gold("id", "Use interval notation.", r"(5,\infty)")[
        "gold_type"
    ] == "interval"
    assert strict.classify_gold("id", "Find the equation.", "y=2x+3")["gold_type"] == "equation"
    assert strict.classify_gold("test/precalculus/819.json", "Choose.", r"\text{(C)}")[
        "gold_type"
    ] == "multiple_choice"


def test_structural_guards_reject_reordered_tuple_set_subset_and_interval_closure(monkeypatch):
    monkeypatch.setattr(
        strict,
        "equivalent_math",
        lambda gold, candidate: (True, gold.replace(" ", "") == candidate.replace(" ", "")),
    )
    tuple_result = strict.verify_structured_answer(
        {"gold_type": "ordered_tuple", "parts": ["1", "2"], "correct_answer": "(1,2)"},
        "(2,1)",
    )
    assert tuple_result["structure_matches"]
    assert not tuple_result["mathematically_equivalent"]

    set_result = strict.verify_structured_answer(
        {"gold_type": "finite_set", "parts": ["1", "2"], "correct_answer": "1,2"},
        "1",
    )
    assert not set_result["structure_matches"]

    interval_result = strict.verify_structured_answer(
        {
            "gold_type": "interval",
            "correct_answer": "(1,2]",
            "parts": [
                {
                    "left_boundary": "(",
                    "left_endpoint": "1",
                    "right_endpoint": "2",
                    "right_boundary": "]",
                }
            ],
        },
        "[1,2]",
    )
    assert not interval_result["structure_matches"]


def test_equation_guard_requires_a_complete_relation(monkeypatch):
    monkeypatch.setattr(strict, "equivalent_math", lambda gold, candidate: (True, True))
    result = strict.verify_structured_answer(
        {"gold_type": "equation", "parts": ["x", "5"], "correct_answer": "x=5"}, "5"
    )
    assert not result["structure_matches"]
    assert not result["mathematically_equivalent"]


def test_unboxed_correct_answer_is_semantic_success_and_format_failure(monkeypatch):
    monkeypatch.setattr(
        strict,
        "verify_structured_answer",
        lambda _gold, _answer: {
            "mathematically_equivalent": True,
            "structure_matches": True,
            "symbolic_parse_succeeded": True,
        },
    )
    gate = {
        "decision": {
            "answer_form": "scalar",
            "boxed_final_answer": False,
            "definitive": True,
            "final_answer_raw": "5",
            "format_compliant": False,
            "later_conflict_quote": None,
            "reason_code": "single_committed_unboxed_answer",
            "supporting_quote": "final answer is 5",
        },
        "gate_status": "resolved",
        "problem_id": "p",
        "row_key": "key",
        "shard_index": 0,
        "variant": "base_7b",
    }
    row = strict.score_row(
        gate,
        {"gold_type": "math", "correct_answer": "5", "parts": ["5"]},
        {"cap_hit": False, "is_correct": True},
    )
    assert row["correct"] is True
    assert row["format_compliant"] is False


def test_cap_hit_sensitivity_reports_verified_and_extreme_counts():
    rows = [
        {"correct": True, "definitive": True, "symbolic_parse_succeeded": True, "format_compliant": True, "native_cap_hit": True, "native_correct": False},
        {"correct": False, "definitive": True, "symbolic_parse_succeeded": True, "format_compliant": False, "native_cap_hit": True, "native_correct": False},
        {"correct": True, "definitive": True, "symbolic_parse_succeeded": True, "format_compliant": False, "native_cap_hit": False, "native_correct": True},
        {"correct": False, "definitive": False, "symbolic_parse_succeeded": False, "format_compliant": False, "native_cap_hit": False, "native_correct": False},
    ]
    value = strict.metrics(rows)
    assert value["cap_hit_percent"] == 50.0
    assert value["cap_hits_correct"] == 1
    assert value["correct_if_all_cap_hits_wrong"] == 1
    assert value["correct_if_all_cap_hits_right"] == 3
    assert value["correct"] == 2


def test_unresolved_rows_hold_accuracy_open_and_publish_cap_bounds():
    rows = [
        {"correct": True, "definitive": True, "symbolic_parse_succeeded": True, "format_compliant": True, "native_cap_hit": False, "native_correct": True},
        {"correct": None, "definitive": None, "symbolic_parse_succeeded": None, "format_compliant": None, "native_cap_hit": False, "native_correct": False},
        {"correct": None, "definitive": True, "symbolic_parse_succeeded": False, "format_compliant": True, "native_cap_hit": True, "native_correct": False},
    ]
    value = strict.metrics(rows)
    assert value["accuracy"] is None
    assert value["scoring_coverage"] == pytest.approx(1 / 3)
    assert value["accuracy_lower_bound"] == pytest.approx(1 / 3)
    assert value["accuracy_upper_bound"] == 1.0
    assert value["cap_hits_correct"] is None
    assert value["correct_if_all_cap_hits_wrong"] is None
    assert value["correct_if_all_cap_hits_wrong_lower_bound"] == 1
    assert value["correct_if_all_cap_hits_wrong_upper_bound"] == 2


def test_pinned_math_verify_routes_symbolic_equations_and_text():
    pytest.importorskip("math_verify")
    assert strict.equivalent_math(r"\frac{1}{2}", "0.5") == (True, True)
    assert strict.equivalent_math("x=5", "5=x") == (True, True)
    assert strict.equivalent_text(r"\text{Evelyn}", "Evelyn") == (True, True)
    assert strict.equivalent_text(r"\text{(C)}", "C") == (True, True)


def test_real_structural_verifier_requires_complete_multi_answers():
    pytest.importorskip("math_verify")
    complete_set = strict.verify_structured_answer(
        {"gold_type": "finite_set", "parts": ["3", "5", "7"], "correct_answer": "3,5,7"},
        "7,3,5",
    )
    assert complete_set == {
        "mathematically_equivalent": True,
        "structure_matches": True,
        "symbolic_parse_succeeded": True,
    }
    subset = strict.verify_structured_answer(
        {"gold_type": "finite_set", "parts": ["3", "5", "7"], "correct_answer": "3,5,7"},
        "3",
    )
    assert subset["structure_matches"] is False


def test_gate_journal_repairs_duplicate_and_torn_suffix(tmp_path):
    gate_policy_sha = "a" * 64
    source = {
        "problem": "What is 2+3?",
        "problem_id": "p",
        "response": "The final answer is 5.",
        "response_sha256": strict.sha256_text("The final answer is 5."),
        "shard_index": 0,
        "variant": "base_7b",
    }
    decision = {
        "answer_form": "scalar",
        "boxed_final_answer": False,
        "definitive": True,
        "final_answer_raw": "5",
        "format_compliant": False,
        "later_conflict_quote": None,
        "reason_code": "single_committed_unboxed_answer",
        "supporting_quote": "final answer is 5",
    }
    regions = strict.response_regions(source["response"])
    adjudication = {
        "answer_form": "scalar",
        "boxed_final_answer": False,
        "definitive": True,
        "evidence_region_id": regions[-1]["id"],
        "format_compliant": False,
        "reason_code": "single_committed_unboxed_answer",
    }
    row = {
        "adjudication": adjudication,
        "attempts": {"adjudication": 1, "extraction": 1},
        "decision": decision,
        "extraction": {
            "boxed_content": False,
            "candidate_id": "C000",
            "end": source["response"].rfind("5") + 1,
            "end_boundary_id": None,
            "extraction_mode": "candidate",
            "start": source["response"].rfind("5"),
            "start_boundary_id": None,
        },
        "gate_status": "resolved",
        "model_revision": strict.policy()["model"]["revision"],
        "policy_sha256": gate_policy_sha,
        "problem_id": "p",
        "response_sha256": source["response_sha256"],
        "row_key": strict.stable_row_key("base_7b", "p", source["response_sha256"], gate_policy_sha),
        "shard_index": 0,
        "unresolved": None,
        "variant": "base_7b",
    }
    output = tmp_path / "gate.jsonl"
    valid = json.dumps(row, sort_keys=True)
    output.write_text(valid + "\n" + valid + "\n" + '{"torn":')
    repaired = strict.repair_gate_journal(output, [source], gate_policy_sha)
    assert list(repaired) == ["base_7b\0" + "0\0p"]
    assert output.read_text() == valid + "\n"


def test_slurm_contract_uses_48_generation_and_48_gate_h200_tasks():
    submit = (HANDOFF / "submit_available_axolotl.sh").read_text()
    gate = (HANDOFF / "run_llama_gate_h200.sbatch").read_text()
    assert "--array=0-3" in submit
    assert "gate_array_jobs" in submit
    assert "llama_gate_shards=48" in submit
    assert "#SBATCH --gpus=h200:1" in gate
    assert "#SBATCH --time=00:30:00" in gate
    assert "#SBATCH --signal=B:USR1@180" in gate
    assert re.search(r'--dependency="afterok:\$gate_canary_job"', submit)


def test_preflight_recovery_rewires_only_generation_arrays():
    recovery = (HANDOFF / "recover_preflight.sh").read_text()
    assert '[[ "${#array_jobs[@]}" -eq 12 ]]' in recovery
    assert 'scontrol update JobId="$array_job" Dependency="afterok:$replacement_job"' in recovery
    assert "record-preflight-recovery" in recovery
