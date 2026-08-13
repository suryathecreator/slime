#!/usr/bin/env python3
"""Reuse the frozen Qwen2.5 sampled MATH-500 generator."""

from examples.qwen2_5_7b_openr1_math220k_correct_recipe_masked_sft_eval_handoff.generate_math500 import *  # noqa: F403
from examples.qwen2_5_7b_openr1_math220k_correct_recipe_masked_sft_eval_handoff.generate_math500 import parse_args, run


if __name__ == "__main__":
    run(parse_args())
