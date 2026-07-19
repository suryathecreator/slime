from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from unittest import skipUnless


ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "examples" / "qwen3_1_7b_opd_aime2026" / "aime_answer_v2.py"
SPEC = importlib.util.spec_from_file_location("aime_answer_v2_existing", MODULE_PATH)
assert SPEC and SPEC.loader
SCORER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCORER)

ARTIFACT_ROOT = Path(
    os.environ.get(
        "AIME_V1_ARTIFACT_ROOT",
        "/gpfs/scrubbed/suryadv/slime-qwen3-1.7b-opd-aime2026/outputs",
    )
)
BASE = ARTIFACT_ROOT / "aime2026_qwen3_1.7b_base" / "predictions.jsonl"
TEACHER = ARTIFACT_ROOT / "aime2026_qwen3_8b_teacher_single" / "predictions.jsonl"


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def rescore(rows):
    return [
        SCORER.score_answer(
            row["generated_text"],
            row["correct_answer"],
            row["problem"],
            row["problem_id"],
            row["cap_hit"],
            row.get("finish_reason"),
        )
        for row in rows
    ]


@skipUnless(BASE.is_file() and TEACHER.is_file(), "stored AIME v1 generations are unavailable")
def test_existing_510_generations_have_the_expected_v1_to_v2_delta():
    base_v1 = read_jsonl(BASE)
    teacher_v1 = read_jsonl(TEACHER)
    assert len(base_v1) == 480
    assert len(teacher_v1) == 30

    base_v2 = rescore(base_v1)
    teacher_v2 = rescore(teacher_v1)
    assert sum(row["is_correct"] for row in base_v1) == 183
    assert sum(row["is_correct"] for row in base_v2) == 184
    assert sum(row["is_correct"] for row in teacher_v1) == 21
    assert sum(row["is_correct"] for row in teacher_v2) == 21

    flips = [
        (before["problem_id"], before["sample_index"], after["candidate"])
        for before, after in zip(base_v1, base_v2)
        if bool(before["is_correct"]) != bool(after["is_correct"])
    ]
    assert flips == [("aime_2026_21", 2, "41 + 9 = 50")]
    assert not any(
        bool(before["is_correct"]) != bool(after["is_correct"])
        for before, after in zip(teacher_v1, teacher_v2)
    )

    capped_with_close = [
        (before, after)
        for before, after in zip(base_v1, base_v2)
        if before["cap_hit"] and "</think>" in before["generated_text"]
    ]
    assert len(capped_with_close) == 6
    assert all(not after["is_correct"] for _, after in capped_with_close)
    assert all(after["region_policy"] == "final_post_think_region" for _, after in capped_with_close)
