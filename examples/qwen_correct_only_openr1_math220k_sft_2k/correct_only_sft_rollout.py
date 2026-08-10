"""Train-only adapter for exact pretokenized, fully supervised correct traces."""

from __future__ import annotations

import logging


__all__ = ["generate_rollout"]
logger = logging.getLogger(__name__)
_PRINTED = False


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    del rollout_id
    if evaluation:
        raise ValueError("correct-only SFT rollout is train-only")
    if not args.rollout_global_dataset:
        raise ValueError("correct-only SFT requires --rollout-global-dataset")
    global _PRINTED
    samples = data_buffer.get_samples(args.rollout_batch_size)
    for group in samples:
        if len(group) != 1:
            raise ValueError("full SFT requires one sample per prompt")
        sample = group[0]
        tokens = [int(token) for token in sample.prompt]
        metadata = sample.metadata
        response_length = int(metadata["response_length"])
        assistant_count = int(metadata["assistant_token_count"])
        weights = [float(value) for value in metadata["loss_weights"]]
        if not 0 < assistant_count < response_length <= len(tokens):
            raise ValueError("invalid assistant/response boundary")
        if len(weights) != response_length:
            raise ValueError("loss-weight length does not match response length")
        response = tokens[-response_length:]
        if response[-2:] != [151645, 198] or weights[-2:] != [1.0, 0.0]:
            raise ValueError("assistant terminal supervision drift")
        if any(value != 1.0 for value in weights[:assistant_count]):
            raise ValueError("correct-only assistant token is not fully supervised")
        if any(token in (151643, 151644, 151645) for token in response[:assistant_count]):
            raise ValueError("assistant trace contains a chat boundary token")
        sample.tokens = tokens
        sample.response_length = response_length
        sample.loss_mask = weights
        sample.reward = 0
        if not _PRINTED:
            logger.info(
                "correct_only_sft total_tokens=%d assistant_tokens=%d trace_id=%s",
                len(tokens),
                assistant_count,
                metadata.get("trace_id"),
            )
            _PRINTED = True
    return samples
