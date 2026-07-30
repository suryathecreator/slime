#!/usr/bin/env python3
"""Qwen2.5 configuration of the shared, audited Axolotl MATH-500 controller."""

from pathlib import Path

from examples.qwen3_8b_openr1_math220k_masked_sft_eval_handoff import (
    axolotl_math500_eval as controller,
)

HANDOFF = Path(__file__).resolve().parent
controller.HANDOFF = HANDOFF
controller.DEFAULT_MANIFEST = HANDOFF / "available_eval_manifest.json"
controller.POLICY = HANDOFF / "axolotl_eval_policy.json"
controller.EXPECTED_VARIANTS = {
    "base_7b",
    "correct_only",
    "unmasked",
    "random_mask_25",
    "random_mask_50",
    "random_mask_70",
    "random_mask_80",
    "random_mask_90",
    "inverse_tau_0p20",
    "inverse_tau_0p05",
    "margin_mask",
    "prob_ratio_mask",
}
controller.EXPECTED_CHECKPOINT_COUNT = len(controller.EXPECTED_VARIANTS)
controller.APPROVED_BASE_REVISION = "d149729398750b98c0af14eb82c78cfe92750796"
controller.EXPECTED_BASE_TOKENIZER_FILES = {
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "c91efca15ceff6e9ee9424db58a6f59cd41294e550a86cbd07e3c1fb500b34f9",
}


if __name__ == "__main__":
    controller.main()
