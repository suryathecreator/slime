from __future__ import annotations

from examples.qwen3_8b_opd_tillicum.prepare_openr1_math220k import (
    build_opd_rows,
    build_sft_rows,
    correct_generation_candidates,
    select_records,
    stable_trace_choice,
)
from examples.qwen3_8b_opd_tillicum.validate_qwen3_generation import (
    normalize_token_ids,
    semantic_terminal_audit,
)


class FakeTokenizer:
    def decode(self, token_ids, *, skip_special_tokens=False):
        assert not skip_special_tokens
        return "".join({198: "\n", 220: " ", 42: "content"}.get(token_id, f"<{token_id}>") for token_id in token_ids)


def make_row(index: int, *, llama_only: bool = False, invalid_first: bool = False):
    generations = [
        f"<think>wrong or incomplete {index}",
        f"<think>correct reasoning {index}</think>\\boxed{{{index}}}",
        f"<think>also correct {index}</think>answer {index}",
    ]
    return {
        "uuid": f"uuid-{index}",
        "messages": [{"role": "user", "content": f"Problem {index}"}, {"role": "assistant", "content": "unused"}],
        "generations": generations,
        "is_reasoning_complete": [False, True, True],
        "correctness_math_verify": [False, not llama_only, True],
        "correctness_llama": [False, llama_only, False],
        "answer": str(index),
        "source": "fixture",
        "problem_type": "Algebra",
        "question_type": "math-word-problem",
    }


def test_correct_trace_uses_math_verify_or_llama_and_requires_complete_think():
    normal = correct_generation_candidates(make_row(1))
    llama = correct_generation_candidates(make_row(2, llama_only=True))
    assert normal == [(1, ("math_verify",)), (2, ("math_verify",))]
    assert llama == [(1, ("llama_judge",)), (2, ("math_verify",))]


def test_trace_choice_is_seeded_and_order_independent():
    candidates = [(1, ("math_verify",)), (2, ("llama_judge",))]
    assert stable_trace_choice(1234, "uuid-9", candidates) == stable_trace_choice(1234, "uuid-9", candidates)


def test_selection_is_exact_disjoint_and_reproducible():
    rows = [make_row(i) for i in range(12)]
    first = select_records(rows, seed=1234, sft_size=6, opd_size=4)
    second = select_records(rows, seed=1234, sft_size=6, opd_size=4)
    sft, opd, _ = first
    assert first == second
    assert len(sft) == 6
    assert len(opd) == 4
    assert {row["uuid"] for row in sft}.isdisjoint({row["uuid"] for row in opd})
    assert {row["source_row_id"] for row in sft}.isdisjoint({row["source_row_id"] for row in opd})
    assert {row["prompt_sha256"] for row in sft}.isdisjoint({row["prompt_sha256"] for row in opd})

    sft_rows = build_sft_rows(sft, 1234)
    opd_rows = build_opd_rows(opd, 1234)
    assert sft_rows[0]["messages"][1]["content"].startswith("<think>")
    assert opd_rows[0]["reference_response"].startswith("<think>")
    assert opd_rows[0]["metadata"]["correctness_sources"]


def test_normalize_token_ids_accepts_batch_encoding_shape_and_plain_list():
    expected = [10, 151645, 198]
    assert normalize_token_ids({"input_ids": expected}) == expected
    assert normalize_token_ids(expected) == expected
    assert normalize_token_ids({"input_ids": [expected]}) == expected


def test_semantic_terminal_accepts_template_whitespace():
    audit = semantic_terminal_audit(FakeTokenizer(), {"input_ids": [10, 151645, 198]}, 151645)
    assert audit["terminal_token_position"] == 1
    assert audit["trailing_token_ids"] == [198]
    assert audit["trailing_text"] == "\n"


def test_semantic_terminal_rejects_missing_or_followed_by_content():
    for token_ids in ([10, 198], [10, 151645, 42]):
        try:
            semantic_terminal_audit(FakeTokenizer(), token_ids, 151645)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected semantic terminal validation failure for {token_ids}")
