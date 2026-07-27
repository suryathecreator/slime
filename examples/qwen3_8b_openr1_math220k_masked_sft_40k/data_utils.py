"""Shared exact-token helpers for the OpenR1 masked full-SFT pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


INSTRUCTION = r"Please reason step by step, and put your final answer within \boxed{}."
CHAT_CONTROL_TOKENS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
CORRECTNESS_PRIORITY = (
    "correctness_math_verify",
    "correctness_llama",
    "math_verify_reparsed_answer",
    "math_verify_answer",
    "llama_verification",
)


def value_at(value: Any, index: int) -> Any:
    """Return a scalar or generation-indexed value."""
    if isinstance(value, (list, tuple)):
        return value[index] if index < len(value) else None
    return value if index == 0 else None


def normalize_bool(value: Any) -> bool | None:
    """Normalize common correctness encodings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "correct"}:
            return True
        if lowered in {"false", "no", "0", "wrong", "incorrect"}:
            return False
    return None


def correctness_at(row: dict[str, Any], generation_index: int) -> tuple[bool | None, str | None]:
    """Resolve correctness using the Axolotl experiment's priority order."""
    for key in CORRECTNESS_PRIORITY:
        if key in row:
            result = normalize_bool(value_at(row.get(key), generation_index))
            if result is not None:
                return result, key
    nested = row.get("correctness")
    if isinstance(nested, dict):
        for key in CORRECTNESS_PRIORITY:
            result = normalize_bool(value_at(nested.get(key), generation_index))
            if result is not None:
                return result, f"correctness.{key}"
    return None, None


def usable_text(value: Any) -> str | None:
    """Return stripped non-null text."""
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("content") or value.get("text")
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan", "[]"}:
        return None
    return text


def problem_id(row: dict[str, Any], row_index: int) -> str:
    """Return the stable OpenR1 problem identifier."""
    value = row.get("uuid") or row.get("problem_id") or row.get("id")
    return str(value).strip() if value is not None and str(value).strip() else f"row_{row_index:07d}"


def user_content(problem: str, correct_answer: str | None = None) -> str:
    """Build the ordinary or answer-conditioned Axolotl OpenR1 prompt."""
    base = f"{INSTRUCTION}\n\nProblem:\n{problem}"
    if correct_answer is None:
        return base
    return f"{base}\n\nKnown correct answer: {correct_answer}"


def normalize_token_ids(value: Any) -> list[int]:
    """Normalize tokenizer output wrappers to a flat integer list."""
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


def chat_prefix_ids(tokenizer: Any, content: str) -> list[int]:
    """Tokenize a user prompt through the Qwen chat generation boundary."""
    return normalize_token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    )


def chat_prefix_suffix_ids(tokenizer: Any, content: str) -> tuple[list[int], list[int]]:
    """Return the Qwen generation prefix and exact assistant terminal suffix.

    Qwen3's template renders an empty assistant as an additional empty think
    block. That is not a neutral suffix for an already-tokenized OpenR1 trace,
    so construct only the literal assistant terminator here.
    """
    prefix = chat_prefix_ids(tokenizer, content)
    suffix = normalize_token_ids(tokenizer.encode("<|im_end|>\n", add_special_tokens=False))
    return prefix, suffix


def token_sha256(token_ids: Iterable[int]) -> str:
    """Hash a token sequence without tokenizer-dependent text decoding."""
    payload = ",".join(str(int(token)) for token in token_ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def trace_structure_ok(trace: str) -> tuple[bool, str]:
    """Require one ordered, non-empty think block and a non-empty answer tail."""
    if any(token in trace for token in CHAT_CONTROL_TOKENS):
        return False, "embedded_chat_control"
    if trace.count("<think>") != 1 or trace.count("</think>") != 1:
        return False, "think_tag_count"
    start = trace.index("<think>") + len("<think>")
    end = trace.index("</think>")
    if start >= end or not trace[start:end].strip():
        return False, "empty_think"
    if not trace[end + len("</think>") :].strip():
        return False, "empty_post_think"
    return True, "ok"


def build_training_record(
    tokenizer: Any,
    selected: dict[str, Any],
    assistant_weights: list[float] | None,
    max_sequence_length: int,
) -> dict[str, Any]:
    """Create one exact-token Slime SFT record with response-side weights."""
    content = user_content(str(selected["problem"]))
    prefix, suffix = chat_prefix_suffix_ids(tokenizer, content)
    assistant_ids = tokenizer.encode(str(selected["assistant_trace"]), add_special_tokens=False)
    assistant_ids = [int(token) for token in assistant_ids]
    if not assistant_ids:
        raise ValueError(f"empty assistant tokens for {selected['trace_id']}")
    if token_sha256(assistant_ids) != selected["assistant_token_sha256"]:
        raise ValueError(f"assistant token hash drift for {selected['trace_id']}")
    if len(prefix) + len(assistant_ids) + len(suffix) > max_sequence_length:
        raise ValueError(f"selected trace no longer fits: {selected['trace_id']}")
    if assistant_weights is None:
        assistant_weights = [1.0] * len(assistant_ids)
    if len(assistant_weights) != len(assistant_ids):
        raise ValueError(f"assistant weight length mismatch for {selected['trace_id']}")

    im_end_id = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    eos_id = int(tokenizer.convert_tokens_to_ids("<|endoftext|>"))
    if im_end_id != 151645 or eos_id != 151643:
        raise ValueError(f"unexpected Qwen terminal IDs im_end={im_end_id} eos={eos_id}")
    im_end_positions = [index for index, token in enumerate(suffix) if int(token) == im_end_id]
    if len(im_end_positions) != 1:
        raise ValueError(f"expected exactly one im_end in assistant suffix, got {im_end_positions}")
    if eos_id in suffix:
        raise ValueError("assistant suffix unexpectedly contains synthetic endoftext")
    suffix_weights = [1.0 if index in im_end_positions else 0.0 for index in range(len(suffix))]
    response_weights = [float(value) for value in assistant_weights] + suffix_weights
    input_ids = prefix + assistant_ids + suffix
    response_length = len(assistant_ids) + len(suffix)
    metadata = {
        "assistant_token_count": len(assistant_ids),
        "assistant_token_sha256": selected["assistant_token_sha256"],
        "correctness_source": selected["correctness_source"],
        "is_correct": bool(selected["is_correct"]),
        "loss_weights": response_weights,
        "problem_id": selected["problem_id"],
        "response_length": response_length,
        "source_generation_index": int(selected["generation_index"]),
        "source_row_index": int(selected["source_row_index"]),
        "trace_id": selected["trace_id"],
        "train_terminal_policy": "assistant_plus_im_end_no_synthetic_endoftext",
    }
    if selected.get("wrong_index") is not None:
        metadata["wrong_index"] = int(selected["wrong_index"])
    return {"input_ids": input_ids, "metadata": metadata}


def atomic_json(path: str | Path, value: Any) -> None:
    """Atomically write a formatted JSON file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        temp = Path(handle.name)
    temp.replace(target)


def atomic_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Atomically stream JSONL rows to a destination."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        Path(temporary).replace(target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL artifact."""
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: str | Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
