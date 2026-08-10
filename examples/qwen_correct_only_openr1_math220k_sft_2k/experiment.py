"""Static contract access for the correct-only experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


EXAMPLE_DIR = Path(__file__).resolve().parent
CONTRACT_PATH = EXAMPLE_DIR / "config" / "experiment_contract.json"
RUN_ORDER = (
    "qwen2_5_3b_8k",
    "qwen2_5_3b_16k",
    "qwen2_5_7b_8k",
    "qwen3_4b_8k",
    "qwen3_8b_8k",
)
MODEL_ORDER = ("qwen2_5_3b", "qwen2_5_7b", "qwen3_4b", "qwen3_8b")
SHARED_8K_MODELS = MODEL_ORDER


def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_spec(run_key: str, contract: dict[str, Any] | None = None) -> dict[str, Any]:
    value = contract or load_contract()
    try:
        return value["runs"][run_key]
    except KeyError as error:
        raise ValueError(f"unknown run key: {run_key}") from error


def model_spec(model_key: str, contract: dict[str, Any] | None = None) -> dict[str, Any]:
    value = contract or load_contract()
    try:
        return value["models"][model_key]
    except KeyError as error:
        raise ValueError(f"unknown model key: {model_key}") from error


def model_for_run(run_key: str, contract: dict[str, Any] | None = None) -> str:
    return str(run_spec(run_key, contract)["model"])


def ordered_trace_sha256(trace_ids: list[str]) -> str:
    import hashlib

    return hashlib.sha256(("\n".join(trace_ids) + "\n").encode("utf-8")).hexdigest()
