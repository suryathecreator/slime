from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import build_corpus  # noqa: E402
import pipeline  # noqa: E402


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = pipeline.load_corpus()

    def test_exact_shape_and_order(self) -> None:
        self.assertEqual(len(self.rows), 480)
        self.assertEqual(
            [row["problem_id"] for row in self.rows], build_corpus.expected_ids()
        )
        for year in range(2009, 2025):
            self.assertEqual(sum(row["year"] == year for row in self.rows), 30)

    def test_sources_and_special_gold(self) -> None:
        counts = {}
        for row in self.rows:
            kind = row["source"]["kind"]
            counts[kind] = counts.get(kind, 0) + 1
        self.assertEqual(
            counts,
            {"primary_huggingface": 455, "supplement_live_poshenloh": 25},
        )
        special = next(row for row in self.rows if row["problem_id"] == "2022-II-8")
        self.assertEqual(special["accepted_answer_integers"], [80, 81])


class PolicyTests(unittest.TestCase):
    def test_prompt_is_exact(self) -> None:
        self.assertEqual(
            pipeline.prompt_for_problem("X"),
            "Please reason step by step, and put your final answer within \\boxed{}.\n\nProblem:\nX",
        )

    def test_task_map_is_complete_and_contiguous(self) -> None:
        assignments = [pipeline.task_assignment(index) for index in range(64)]
        self.assertEqual(assignments[0], (0, 0, 120))
        self.assertEqual(assignments[3], (0, 360, 480))
        self.assertEqual(assignments[4], (1, 0, 120))
        self.assertEqual(assignments[-1], (15, 360, 480))
        pairs = {
            (draw, eval_index)
            for task in range(64)
            for draw, start, end in [pipeline.task_assignment(task)]
            for eval_index in range(start, end)
        }
        self.assertEqual(len(pairs), 7680)

    def test_seed_map_is_unique(self) -> None:
        seeds = {
            pipeline.seed_for(draw, eval_index)
            for draw in range(16)
            for eval_index in range(480)
        }
        self.assertEqual(len(seeds), 7680)
        self.assertEqual(min(seeds), 1234)
        self.assertEqual(max(seeds), 1234 + 7679)


class ScorerTests(unittest.TestCase):
    def score(self, response: str, accepted: list[int]) -> dict:
        return pipeline.score_accepted_answers(
            response,
            accepted,
            finish_reason="stop",
            stop_token_id=151645,
            cap_hit=False,
            problem_id="test",
        )

    def test_last_box_only_and_no_unboxed_fallback(self) -> None:
        trace = self.score(r"first \boxed{7}; finally \boxed{8}", [8])
        self.assertTrue(trace["is_correct"])
        self.assertEqual(trace["parsed_integer"], 8)
        trace = self.score("final answer is 8", [8])
        self.assertFalse(trace["is_scorable"])

    def test_both_officially_accepted_answers(self) -> None:
        self.assertTrue(self.score(r"\boxed{080}", [80, 81])["is_correct"])
        trace = self.score(r"\boxed{081}", [80, 81])
        self.assertTrue(trace["is_correct"])
        self.assertEqual(trace["matched_gold_integer"], 81)

    def test_invalid_box_contents_are_not_repaired(self) -> None:
        self.assertFalse(self.score(r"\boxed{80 or 81}", [80, 81])["is_scorable"])
        self.assertFalse(self.score(r"\boxed{-1}", [999])["is_scorable"])


if __name__ == "__main__":
    unittest.main()
