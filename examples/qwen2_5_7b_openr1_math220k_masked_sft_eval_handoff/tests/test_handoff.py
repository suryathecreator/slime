"""Static handoff contract checks independent of checkpoint availability."""

from __future__ import annotations

import json
from pathlib import Path


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


def test_source_and_destination_inventories_match():
    source = json.loads((HANDOFF / "checkpoint_sources.json").read_text())
    destination = json.loads((HANDOFF / "available_eval_manifest.json").read_text())
    assert {item["variant"] for item in source["checkpoints"]} == {
        item["variant"] for item in destination["checkpoints"]
    }
