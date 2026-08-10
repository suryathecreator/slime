"""Rollout adapter for exact pre-tokenized SFT with protected controls."""

from __future__ import annotations

import logging
import math

__all__ = ["generate_rollout"]
logger = logging.getLogger(__name__)
_PRINTED = False


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
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
        metadata = sample.metadata
        response_length = int(metadata["response_length"])
        assistant_count = int(metadata["assistant_token_count"])
        weights = [float(value) for value in metadata["loss_weights"]]
        if not 0 < assistant_count < response_length <= len(token_ids):
            raise ValueError("invalid assistant/response boundary")
        if len(weights) != response_length:
            raise ValueError("loss-weight length does not equal response length")
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in weights):
            raise ValueError("loss weights must be finite and in [0,1]")
        protected = {int(index) for index in metadata["protected_assistant_positions"]}
        if not protected or any(index >= assistant_count or weights[index] != 1.0 for index in protected):
            raise ValueError("protected assistant controls are not exactly supervised")
        response_ids = token_ids[-response_length:]
        if response_ids[assistant_count] != 151645 or weights[assistant_count] != 1.0:
            raise ValueError("im_end is not the exactly supervised terminal")
        if 151643 in token_ids or any(value != 0.0 for value in weights[assistant_count + 1:]):
            raise ValueError("synthetic EOS or supervised template suffix")
        if sum(weights) <= 0.0:
            raise ValueError("sample has zero supervised weight")
        sample.tokens = token_ids
        sample.response_length = response_length
        sample.loss_mask = weights
        sample.reward = 0
        if not _PRINTED:
            logger.info(
                "qwen25_weighted_sft total_tokens=%d response_tokens=%d protected=%d trace_id=%s",
                len(token_ids), response_length, len(protected), metadata.get("trace_id"),
            )
            _PRINTED = True
    return samples
