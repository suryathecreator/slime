import json
import os
import re
from pathlib import Path

import aiohttp
import torch

from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample

_FINAL_ANSWER_RE = re.compile(
    r"(\\boxed\s*\{|(?:final\s+answer|answer)\s*(?:is|:)|therefore[, ]+(?:the\s+)?answer)",
    re.IGNORECASE,
)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _response_answer_segment(response: str) -> str:
    if "</think>" in response:
        return response.rsplit("</think>", 1)[-1]
    if "###Response" in response:
        return response.rsplit("###Response", 1)[-1]
    return response


def _has_final_answer(response: str) -> bool:
    return bool(_FINAL_ANSWER_RE.search(_response_answer_segment(response or "")))


def _rollout_id_for_samples(samples: list[Sample]) -> int | None:
    rollout_ids = {sample.rollout_id for sample in samples if sample.rollout_id is not None}
    if len(rollout_ids) == 1:
        return next(iter(rollout_ids))
    return None


def summarize_rollout_sanity(args, samples: list[Sample]) -> dict:
    max_response_len = int(getattr(args, "rollout_max_response_len", 0) or 0)
    n = len(samples)
    response_lengths = [int(sample.response_length or 0) for sample in samples]
    sample_indices = [sample.index for sample in samples if sample.index is not None]
    cap_hits = [
        sample.status == Sample.Status.TRUNCATED
        or (max_response_len > 0 and int(sample.response_length or 0) >= max_response_len)
        for sample in samples
    ]
    final_answer_hits = [_has_final_answer(sample.response or "") for sample in samples]
    completed = [sample.status == Sample.Status.COMPLETED for sample in samples]

    return {
        "rollout_id": _rollout_id_for_samples(samples),
        "sample_index_min": min(sample_indices) if sample_indices else None,
        "sample_index_max": max(sample_indices) if sample_indices else None,
        "n": n,
        "avg_response_tokens": (sum(response_lengths) / n) if n else 0.0,
        "max_response_tokens": max(response_lengths) if response_lengths else 0,
        "cap_hit_rate": (sum(cap_hits) / n) if n else 0.0,
        "completed_rate": (sum(completed) / n) if n else 0.0,
        "final_answer_rate": (sum(final_answer_hits) / n) if n else 0.0,
        "example_tails": [
            {
                "index": sample.index,
                "rollout_id": sample.rollout_id,
                "response_length": int(sample.response_length or 0),
                "status": sample.status.value,
                "has_final_answer": final_answer_hits[i],
                "tail": (sample.response or "")[-500:],
            }
            for i, sample in enumerate(samples[:3])
        ],
    }


def maybe_check_rollout_sanity(args, samples: list[Sample]) -> None:
    if not _env_bool("OPD_SANITY_CHECK_ENABLED", False):
        return
    rollout_id = _rollout_id_for_samples(samples)
    max_rollout_id = int(os.environ.get("OPD_SANITY_MAX_ROLLOUT_ID", "0"))
    if rollout_id is not None and rollout_id > max_rollout_id:
        return

    summary = summarize_rollout_sanity(args, samples)
    report_dir = os.environ.get("OPD_SANITY_REPORT_DIR")
    if report_dir:
        if rollout_id is not None:
            report_name = f"rollout_{rollout_id}"
        else:
            report_name = f"samples_{summary['sample_index_min']}_{summary['sample_index_max']}"
        path = Path(report_dir) / f"{report_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"OPD_SANITY {json.dumps(summary, ensure_ascii=False)}", flush=True)

    violations = []
    max_cap_hit_rate = float(os.environ.get("OPD_SANITY_MAX_CAP_HIT_RATE", "1.0"))
    max_avg_tokens = float(os.environ.get("OPD_SANITY_MAX_AVG_RESPONSE_TOKENS", "inf"))
    min_final_answer_rate = float(os.environ.get("OPD_SANITY_MIN_FINAL_ANSWER_RATE", "0.0"))

    if summary["cap_hit_rate"] > max_cap_hit_rate:
        violations.append(f"cap_hit_rate {summary['cap_hit_rate']:.3f} > {max_cap_hit_rate:.3f}")
    if summary["avg_response_tokens"] > max_avg_tokens:
        violations.append(f"avg_response_tokens {summary['avg_response_tokens']:.1f} > {max_avg_tokens:.1f}")
    if summary["final_answer_rate"] < min_final_answer_rate:
        violations.append(
            f"final_answer_rate {summary['final_answer_rate']:.3f} < {min_final_answer_rate:.3f}"
        )

    if violations and _env_bool("OPD_SANITY_FAIL_ON_COLLAPSE", True):
        raise RuntimeError("OPD sanity check failed: " + "; ".join(violations))


async def reward_func(args, sample, **kwargs):
    payload = {
        # "text": sample.prompt + sample.response,
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }

    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        image_data = sample.multimodal_inputs["images"]
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    session_kwargs = {}
    async with aiohttp.ClientSession(**session_kwargs) as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Process rewards from teacher model and extract teacher log probabilities.

    This function:
    1. Extracts teacher log-probs from the reward response (which contains sglang's logprob output)
    2. Trims them to match the response length
    3. Stores them in sample.teacher_log_probs for OPD KL penalty computation
    4. Returns scalar rewards (0.0 for pure distillation) compatible with GRPO/PPO

    Note: The reward_func calls the teacher server which returns token-level log-probs.
    For pure on-policy distillation without task rewards, we return 0.0 for each sample.
    The actual learning signal comes from the OPD KL penalty applied in compute_advantages_and_returns.
    """
    maybe_check_rollout_sanity(args, samples)

    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]

    # Extract teacher log-probs from the sglang response
    teacher_log_probs = [
        torch.tensor([item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]], dtype=torch.float32)
        for reward in raw_rewards
    ]
    teacher_log_probs = [
        t_log_prob[-response_length:]
        for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=False)
    ]

    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=False):
        sample.teacher_log_probs = t_log_probs

    # Return scalar rewards for GRPO/PPO advantage estimator
    # For pure on-policy distillation, we use 0.0 as the task reward.
    # The learning signal comes entirely from the OPD KL penalty.
    # If you have task rewards, you can add them here.
    scalar_rewards = [0.0] * len(samples)

    return scalar_rewards, scalar_rewards
