"""Exact-token construction helpers for unmasked correct-only SFT."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


INSTRUCTION = r"Please reason step by step, and put your final answer within \boxed{}."
CHAT_CONTROLS = ("<|endoftext|>", "<|im_start|>", "<|im_end|>")
END_OF_TEXT = 151643
IM_START = 151644
IM_END = 151645
TERMINAL_NEWLINE = 198


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
        raise ValueError(f"unsupported tokenizer output: {type(value).__name__}")
    return [int(token) for token in value]


def user_content(problem: str) -> str:
    return f"{INSTRUCTION}\n\nProblem:\n{problem}"


def chat_prefix(tokenizer: Any, problem: str, template_kwargs: dict[str, Any]) -> list[int]:
    return normalize_token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content(problem)}],
            tokenize=True,
            add_generation_prompt=True,
            **template_kwargs,
        )
    )


def assistant_ids(tokenizer: Any, trace: str) -> list[int]:
    return normalize_token_ids(tokenizer.encode(trace, add_special_tokens=False))


def token_sha256(tokens: Iterable[int]) -> str:
    payload = ",".join(str(int(token)) for token in tokens).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def rendered_lengths(
    tokenizer: Any,
    problem: str,
    trace: str,
    template_kwargs: dict[str, Any],
) -> tuple[int, int]:
    prefix = chat_prefix(tokenizer, problem, template_kwargs)
    response = assistant_ids(tokenizer, trace)
    return len(response), len(prefix) + len(response) + 2


def build_training_record(
    tokenizer: Any,
    selected: dict[str, Any],
    template_kwargs: dict[str, Any],
    max_sequence_length: int,
    trace_token_cap: int,
    model_key: str,
) -> dict[str, Any]:
    trace = str(selected["assistant_trace"])
    if any(control in trace for control in CHAT_CONTROLS):
        raise ValueError(f"embedded chat control in {selected['trace_id']}")
    prefix = chat_prefix(tokenizer, str(selected["problem"]), template_kwargs)
    response = assistant_ids(tokenizer, trace)
    suffix = normalize_token_ids(tokenizer.encode("<|im_end|>\n", add_special_tokens=False))
    if suffix != [IM_END, TERMINAL_NEWLINE]:
        raise ValueError(f"terminal tokenization drift for {model_key}: {suffix}")
    if not tokenizer.decode(prefix, skip_special_tokens=False).endswith("<|im_start|>assistant\n"):
        raise ValueError(f"assistant entry drift for {model_key}")
    if not response or len(response) > trace_token_cap:
        raise ValueError(f"assistant length outside contract for {selected['trace_id']}")
    if any(token in (END_OF_TEXT, IM_START, IM_END) for token in response):
        raise ValueError(f"assistant boundary token in {selected['trace_id']}")
    input_ids = prefix + response + suffix
    if len(input_ids) > max_sequence_length:
        raise ValueError(f"rendered sequence too long for {selected['trace_id']}")
    response_length = len(response) + len(suffix)
    loss_weights = [1.0] * len(response) + [1.0, 0.0]
    return {
        "input_ids": input_ids,
        "metadata": {
            "assistant_token_count": len(response),
            "assistant_token_sha256": token_sha256(response),
            "is_correct": True,
            "loss_weights": loss_weights,
            "model_key": model_key,
            "problem_id": str(selected["problem_id"]),
            "response_length": response_length,
            "source_generation_index": int(selected["generation_index"]),
            "source_row_index": int(selected["source_row_index"]),
            "trace_id": str(selected["trace_id"]),
            "trace_text_sha256": hashlib.sha256(trace.encode("utf-8")).hexdigest(),
            "train_terminal_policy": "assistant_and_im_end_weight_1_template_newline_weight_0_no_eot",
        },
    }


def message_record(selected: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(selected["trace_id"]),
        "messages": [
            {"role": "user", "content": user_content(str(selected["problem"]))},
            {"role": "assistant", "content": str(selected["assistant_trace"])},
        ],
        "metadata": {"trace_id": str(selected["trace_id"])},
    }


def atomic_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(target)


def atomic_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        Path(temporary_name).replace(target)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
