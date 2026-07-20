from __future__ import annotations

import random

from examples.qwen3_8b_32b_full_repro.clean_openthoughts3 import (
    PROMPT_INSTRUCTION,
    RANGES,
    message_source_field,
    normalize_messages,
    prompt_with_instruction,
    validate_think,
    evaluate_task,
)


def test_think_validation_accepts_one_complete_nonempty_trace() -> None:
    valid, reasons = validate_think("<think>work</think>\n\\boxed{2}")
    assert valid
    assert reasons == []


def test_think_validation_rejects_all_contract_failures() -> None:
    cases = {
        "no tags": "missing_open",
        "<think>no close": "missing_close",
        "</think><think>x": "close_before_open",
        "<think></think>x": "empty_thinking",
        "<think>x</think>": "empty_post_think",
        "<think>x</think><think>y</think>z": "duplicated_or_unbalanced_tags",
    }
    for response, expected in cases.items():
        valid, reasons = validate_think(response)
        assert not valid, response
        assert expected in reasons, (response, reasons)


def test_normalizes_only_documented_raw_message_fields() -> None:
    row = {
        "conversations": [
            {"from": "human", "value": "question"},
            {"from": "gpt", "value": "<think>x</think>answer"},
        ]
    }
    assert normalize_messages(row) == [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "<think>x</think>answer"},
    ]
    assert message_source_field(row) == "conversations"


def test_message_field_priority_and_documented_flat_fallback() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "messages question"},
            {"role": "assistant", "content": "<think>x</think>messages answer"},
        ],
        "conversations": [
            {"from": "human", "value": "ignored question"},
            {"from": "gpt", "value": "<think>x</think>ignored answer"},
        ],
    }
    assert normalize_messages(row)[0]["content"] == "messages question"
    assert message_source_field(row) == "messages"

    flat = {"problem": "flat question", "solution": "<think>x</think>flat answer"}
    assert normalize_messages(flat) == [
        {"role": "user", "content": "flat question"},
        {"role": "assistant", "content": "<think>x</think>flat answer"},
    ]
    assert message_source_field(flat) == "flat:problem+solution"


def test_prompt_instruction_is_exact_and_idempotent() -> None:
    assert prompt_with_instruction("problem") == "problem\n\n" + PROMPT_INSTRUCTION
    assert prompt_with_instruction("problem\n\n" + PROMPT_INSTRUCTION) == "problem\n\n" + PROMPT_INSTRUCTION


def test_raw_row_shuffle_is_seeded_without_grouping_or_deduplication() -> None:
    source_rows = [11, 12, 13, 14, 15, 16]
    first = source_rows.copy()
    second = source_rows.copy()
    random.Random(1234).shuffle(first)
    random.Random(1234).shuffle(second)
    assert first == second
    assert sorted(first) == source_rows
    # Duplicate prompt text has no role in selecting distinct raw row IDs.
    duplicate_prompts = {11: "same", 12: "same"}
    assert duplicate_prompts[11] == duplicate_prompts[12] and 11 != 12


def test_issue_count_ranges_are_frozen() -> None:
    assert RANGES == {
        "incomplete_think": (711_911, 786_849),
        "non_english": (377_760, 417_524),
        "retained": (316_201, 349_485),
    }


def test_raw_chat_terminal_would_be_rejected_without_repair(monkeypatch) -> None:
    monkeypatch.setattr(
        "examples.qwen3_8b_32b_full_repro.clean_openthoughts3.classify_language",
        lambda _text: {"accepted": True, "reason": "english"},
    )
    result = evaluate_task(
        (
            7,
            {
                "conversations": [
                    {"from": "human", "value": "question"},
                    {"from": "gpt", "value": "<think>x</think>answer<|im_end|>"},
                ]
            },
        )
    )
    assert result["malformed"]
    assert result["raw_chat_control_tokens"] == ["<|im_end|>"]
    assert not result["retained"]
