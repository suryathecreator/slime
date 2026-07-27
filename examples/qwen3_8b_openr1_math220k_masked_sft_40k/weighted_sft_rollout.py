"""Rollout adapter for exact pre-tokenized full SFT with float loss weights."""

from __future__ import annotations

import logging
import math

__all__ = ["generate_rollout"]

logger = logging.getLogger(__name__)
_PRINTED = False


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """Load exact tokens and response weights from the global dataset.

    The prompt field is the complete pre-tokenized chat sequence. Metadata
    stores weights only for the response suffix, which is what Slime expects.
    """
    del rollout_id
    if evaluation:
        raise ValueError("weighted SFT rollout is train-only")
    if not args.rollout_global_dataset:
        raise ValueError("weighted SFT requires --rollout-global-dataset")

    global _PRINTED
    samples = data_buffer.get_samples(args.rollout_batch_size)
    for group in samples:
        if len(group) != 1:
            raise ValueError("full SFT requires one sample per prompt")
        sample = group[0]
        token_ids = [int(value) for value in sample.prompt]
        response_length = int(sample.metadata["response_length"])
        weights = [float(value) for value in sample.metadata["loss_weights"]]
        if not 0 < response_length <= len(token_ids):
            raise ValueError(f"invalid response length {response_length} for {len(token_ids)} tokens")
        if len(weights) != response_length:
            raise ValueError(f"loss-weight length {len(weights)} != response length {response_length}")
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in weights):
            raise ValueError("loss weights must be finite and in [0, 1]")
        if sum(weights) <= 0.0:
            raise ValueError("sample has zero supervised weight")

        sample.tokens = token_ids
        sample.response_length = response_length
        sample.loss_mask = weights
        sample.reward = 0
        if not _PRINTED:
            logger.info(
                "weighted_sft sample total_tokens=%d response_tokens=%d weight_sum=%.6f trace_id=%s",
                len(token_ids),
                response_length,
                sum(weights),
                sample.metadata.get("trace_id"),
            )
            _PRINTED = True
    return samples
