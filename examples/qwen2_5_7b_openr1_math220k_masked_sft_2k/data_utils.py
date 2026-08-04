"""Exact-token helpers and fail-closed supervision invariants for Qwen2.5."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable


INSTRUCTION = r"Please reason step by step, and put your final answer within \boxed{}."
CHAT_BOUNDARY_LITERALS = ("<|endoftext|>", "<|im_start|>", "<|im_end|>")
THINK_LITERALS = ("<think>", "</think>")
TRACE_LOCATION = re.compile(r"__row_(\d+)__gen_(\d+)$")


def normalize_token_ids(value: Any) -> list[int]:
    if isinstance(value, dict):
        value = value["input_ids"]
    elif hasattr(value, "input_ids") and not isinstance(value, (list, tuple)):
        value = value.input_ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = list(value[0])
    if not isinstance(value, list):
        raise ValueError(f"unsupported tokenizer output {type(value).__name__}")
    return [int(token) for token in value]


def stable_unit_interval(seed: int, trace_id: str, token_index: int) -> float:
    payload = "%d\0%s\0%d" % (int(seed), str(trace_id), int(token_index))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def source_location(trace_id: str) -> tuple[int, int]:
    match = TRACE_LOCATION.search(trace_id)
    if match is None:
        raise ValueError(f"trace ID has no source location: {trace_id}")
    return int(match.group(1)), int(match.group(2))


def find_all_subsequence_positions(sequence: list[int], needle: list[int]) -> list[int]:
    if not needle:
        raise ValueError("protected token subsequence is empty")
    covered: list[int] = []
    start = 0
    while start <= len(sequence) - len(needle):
        if sequence[start : start + len(needle)] == needle:
            covered.extend(range(start, start + len(needle)))
            start += len(needle)
        else:
            start += 1
    return covered


def tokenizer_control_inventory(tokenizer: Any) -> dict[str, Any]:
    added = {str(token): int(token_id) for token, token_id in tokenizer.get_added_vocab().items()}
    if len(added) != len(set(added.values())):
        raise ValueError("tokenizer added-vocabulary IDs are not unique")
    return {
        "added_vocabulary": dict(sorted(added.items(), key=lambda item: item[1])),
        "protected_added_token_ids": sorted(added.values()),
        "think_literal_token_ids": {
            literal: normalize_token_ids(
                tokenizer.encode(literal, add_special_tokens=False)
            )
            for literal in THINK_LITERALS
        },
    }


def protected_assistant_positions(
    tokenizer: Any,
    assistant_ids: list[int],
    assistant_text: str | None = None,
) -> tuple[list[int], dict[str, list[str]], dict[str, list[int]]]:
    """Protect all added-token occurrences and all literal think-tag pieces."""
    inventory = tokenizer_control_inventory(tokenizer)
    added_by_id = {
        int(token_id): token
        for token, token_id in inventory["added_vocabulary"].items()
    }
    reasons: dict[int, set[str]] = {}
    for index, token_id in enumerate(assistant_ids):
        if token_id in added_by_id:
            reasons.setdefault(index, set()).add(f"added_token:{added_by_id[token_id]}")
    literal_positions: dict[str, list[int]] = {}
    if assistant_text is None:
        for literal, token_ids in inventory["think_literal_token_ids"].items():
            literal_positions[literal] = find_all_subsequence_positions(
                assistant_ids, token_ids
            )
    else:
        encoded = tokenizer(
            assistant_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        contextual_ids = normalize_token_ids(encoded)
        if contextual_ids != assistant_ids:
            raise ValueError("offset tokenizer IDs differ from assistant token IDs")
        offsets = encoded["offset_mapping"]
        if hasattr(offsets, "tolist"):
            offsets = offsets.tolist()
        for literal in THINK_LITERALS:
            spans: list[tuple[int, int]] = []
            cursor = 0
            while True:
                start = assistant_text.find(literal, cursor)
                if start < 0:
                    break
                spans.append((start, start + len(literal)))
                cursor = start + len(literal)
            covered = {
                index
                for index, (token_start, token_end) in enumerate(offsets)
                for span_start, span_end in spans
                if int(token_end) > span_start and int(token_start) < span_end
            }
            literal_positions[literal] = sorted(covered)
    for literal, positions_for_literal in literal_positions.items():
        for index in positions_for_literal:
            reasons.setdefault(index, set()).add(f"literal_markup:{literal}")
    if not any(
        reason.startswith("literal_markup:")
        for position_reasons in reasons.values()
        for reason in position_reasons
    ):
        raise ValueError("assistant trace contains no literal think-tag tokens")
    positions = sorted(reasons)
    return (
        positions,
        {str(index): sorted(reasons[index]) for index in positions},
        literal_positions,
    )


def user_content(problem: str) -> str:
    return f"{INSTRUCTION}\n\nProblem:\n{problem}"


def chat_prefix_suffix_ids(
    tokenizer: Any,
    content: str,
    apply_chat_template_kwargs: dict[str, Any] | None = None,
) -> tuple[list[int], list[int]]:
    template_kwargs = dict(apply_chat_template_kwargs or {})
    prefix = normalize_token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            **template_kwargs,
        )
    )
    suffix = normalize_token_ids(
        tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
    )
    return prefix, suffix


def token_sha256(token_ids: Iterable[int]) -> str:
    payload = ",".join(str(int(token)) for token in token_ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def trace_set_sha256(trace_ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(trace_ids) + "\n").encode("utf-8")).hexdigest()


def build_training_record(
    tokenizer: Any,
    selected: dict[str, Any],
    assistant_weights: list[float] | None,
    max_sequence_length: int,
    apply_chat_template_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prefix, suffix = chat_prefix_suffix_ids(
        tokenizer,
        user_content(str(selected["problem"])),
        apply_chat_template_kwargs,
    )
    assistant_ids = normalize_token_ids(
        tokenizer.encode(str(selected["assistant_trace"]), add_special_tokens=False)
    )
    if not assistant_ids:
        raise ValueError(f"empty assistant tokens for {selected['trace_id']}")
    if token_sha256(assistant_ids) != selected["assistant_token_sha256"]:
        raise ValueError(f"assistant token hash drift for {selected['trace_id']}")
    if len(prefix) + len(assistant_ids) + len(suffix) > max_sequence_length:
        raise ValueError(f"selected trace no longer fits: {selected['trace_id']}")

    protected, protected_reasons, literal_positions = protected_assistant_positions(
        tokenizer, assistant_ids, str(selected["assistant_trace"])
    )
    if assistant_weights is None:
        assistant_weights = [1.0] * len(assistant_ids)
    if len(assistant_weights) != len(assistant_ids):
        raise ValueError(f"assistant weight length mismatch for {selected['trace_id']}")
    assistant_weights = [float(value) for value in assistant_weights]
    if any(assistant_weights[index] != 1.0 for index in protected):
        raise ValueError(f"protected assistant token is not fully supervised: {selected['trace_id']}")

    im_end_id = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    eos_id = int(tokenizer.convert_tokens_to_ids("<|endoftext|>"))
    im_start_id = int(tokenizer.convert_tokens_to_ids("<|im_start|>"))
    if (eos_id, im_start_id, im_end_id) != (151643, 151644, 151645):
        raise ValueError(
            f"unexpected Qwen boundaries eos={eos_id} im_start={im_start_id} im_end={im_end_id}"
        )
    if any(token_id in assistant_ids for token_id in (eos_id, im_start_id, im_end_id)):
        raise ValueError(f"embedded chat boundary in assistant tokens: {selected['trace_id']}")
    im_end_positions = [index for index, token in enumerate(suffix) if token == im_end_id]
    if im_end_positions != [0] or eos_id in suffix:
        raise ValueError(f"unexpected assistant suffix IDs: {suffix}")
    suffix_weights = [1.0] + [0.0] * (len(suffix) - 1)
    response_weights = assistant_weights + suffix_weights
    metadata = {
        "assistant_token_count": len(assistant_ids),
        "assistant_token_sha256": selected["assistant_token_sha256"],
        "correctness_source": selected["correctness_source"],
        "is_correct": bool(selected["is_correct"]),
        "loss_weights": response_weights,
        "problem_id": selected["problem_id"],
        "protected_assistant_positions": protected,
        "protected_literal_positions": literal_positions,
        "protected_token_reasons": protected_reasons,
        "response_length": len(assistant_ids) + len(suffix),
        "source_generation_index": int(selected["generation_index"]),
        "source_row_index": int(selected["source_row_index"]),
        "trace_id": selected["trace_id"],
        "train_terminal_policy": "assistant_plus_fully_supervised_im_end_no_synthetic_endoftext",
    }
    if selected.get("wrong_index") is not None:
        metadata["wrong_index"] = int(selected["wrong_index"])
    return {"input_ids": prefix + assistant_ids + suffix, "metadata": metadata}


def atomic_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(target)


def atomic_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        Path(temporary).replace(target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
