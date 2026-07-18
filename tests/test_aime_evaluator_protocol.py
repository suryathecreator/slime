from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path


EXAMPLE_DIR = Path(__file__).parents[1] / "examples" / "qwen3_1_7b_opd_aime2026"
sys.path.insert(0, str(EXAMPLE_DIR))
from evaluate_aime_vllm import (  # noqa: E402
    BASE_SEED,
    MIN_P,
    NUM_GENERATIONS,
    NUM_PROBLEMS,
    NUM_SHARDS,
    REQUESTS_PER_SHARD,
    SAMPLES_PER_PROBLEM,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    request_grid,
)


def canonical_rows():
    return [
        {"problem_idx": index, "problem_id": f"aime_2026_{index:02d}"}
        for index in range(1, NUM_PROBLEMS + 1)
    ]


def test_exact_sampling_policy():
    assert (NUM_PROBLEMS, SAMPLES_PER_PROBLEM, NUM_GENERATIONS) == (30, 16, 480)
    assert (TEMPERATURE, TOP_P, TOP_K, MIN_P, BASE_SEED) == (0.6, 0.95, 20, 0.0, 42)


def test_request_grid_has_unique_deterministic_seeds():
    requests = request_grid(canonical_rows())
    assert len(requests) == NUM_GENERATIONS
    assert [request["flat_index"] for request in requests] == list(range(NUM_GENERATIONS))
    assert [request["seed"] for request in requests] == list(range(BASE_SEED, BASE_SEED + NUM_GENERATIONS))
    counts = Counter(request["problem_position"] for request in requests)
    assert set(counts.values()) == {SAMPLES_PER_PROBLEM}


def test_each_real_batch_has_120_requests_and_four_samples_per_problem():
    requests = request_grid(canonical_rows())
    for shard_index in range(NUM_SHARDS):
        shard = [request for request in requests if request["flat_index"] % NUM_SHARDS == shard_index]
        assert len(shard) == REQUESTS_PER_SHARD == 120
        counts = Counter(request["problem_position"] for request in shard)
        assert set(counts.values()) == {4}
