from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


eval_vllm = load_script("eval_math500_vllm_test", "examples/qwen3_8b_opd_tillicum/eval_math500_vllm.py")
summarize_eval = load_script("summarize_eval_test", "examples/qwen3_8b_opd_tillicum/summarize_eval.py")


class FakeTokenizer:
    def __init__(self):
        self.template_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.template_kwargs = kwargs
        return "<|im_start|>user\nquestion<|im_end|>\n<|im_start|>assistant\n"

    def encode(self, text, add_special_tokens=False):
        return [10, 11, 12]

    def decode(self, token_ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        mapping = {
            151667: "<think>",
            42: "The answer is \\boxed{4}.",
            151668: "</think>",
            151645: "<|im_end|>",
            151643: "<|endoftext|>",
        }
        if skip_special_tokens:
            return "".join(mapping[token] for token in token_ids if token not in {151667, 151668, 151645, 151643})
        return "".join(mapping[token] for token in token_ids)

    def convert_tokens_to_ids(self, token):
        return {"<|im_end|>": 151645, "<|endoftext|>": 151643}[token]


def args(tmp_path: Path) -> Namespace:
    return Namespace(
        model=str(tmp_path / "model"),
        stage="sft_025000",
        train_samples=25000,
        max_response_len=31744,
        max_context_len=32768,
        max_model_len=32768,
        dtype="bfloat16",
        enable_thinking=True,
        stop_token_ids=[151645, 151643],
        im_end_token_id=151645,
        endoftext_token_id=151643,
        preserve_special_tokens=True,
        gpu_memory_utilization=0.97,
        max_num_seqs=24,
        max_num_batched_tokens=16384,
        kv_cache_dtype="auto",
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        async_scheduling=True,
        safetensors_load_strategy="prefetch",
        speculative_method="ngram_gpu",
        num_speculative_tokens=8,
        prompt_lookup_min=5,
        prompt_lookup_max=5,
    )


def test_sample_from_output_preserves_special_text_and_raw_token_ids(tmp_path):
    tokenizer = FakeTokenizer()
    prepared, _ = eval_vllm.prepare_chunk_rows(
        tokenizer,
        [{"_eval_index": 0, "prompt": "2+2?", "label": "4", "metadata": {}}],
        args(tmp_path),
    )
    cap, prompt_ids, base = prepared[0]
    completion = SimpleNamespace(
        token_ids=[151667, 42, 151668],
        finish_reason="stop",
        stop_reason=151645,
    )
    sample = eval_vllm.sample_from_output(tokenizer, base, cap, SimpleNamespace(outputs=[completion]))

    assert tokenizer.template_kwargs["enable_thinking"] is True
    assert prompt_ids == sample["prefill_token_ids"] == [10, 11, 12]
    assert sample["generated_token_ids"] == [151667, 42, 151668]
    assert sample["terminal_stop_token_id"] == 151645
    assert sample["generated_token_ids_with_terminal"] == [151667, 42, 151668, 151645]
    assert sample["response"].endswith("</think><|im_end|>")
    assert sample["response_plain"] == "The answer is \\boxed{4}."


def test_chunk_policy_rejects_legacy_and_accepts_matching(tmp_path):
    policy = eval_vllm.generation_policy(args(tmp_path))
    path = tmp_path / "chunk.pt"
    torch.save({"samples": [{"eval_index": 0}]}, path)
    assert eval_vllm.load_valid_chunk(path, {0}, policy) is None

    torch.save(
        {
            "samples": [{"eval_index": 0}],
            "generation_policy": policy,
            "generation_policy_fingerprint": eval_vllm.generation_policy_fingerprint(policy),
        },
        path,
    )
    assert eval_vllm.load_valid_chunk(path, {0}, policy) == [{"eval_index": 0}]


def save_artifact(path: Path, policy: dict, indices: list[int]) -> None:
    torch.save(
        {
            "samples": [{"eval_index": index} for index in indices],
            "generation_policy": policy,
            "generation_policy_fingerprint": eval_vllm.generation_policy_fingerprint(policy),
        },
        path,
    )


def test_resume_discovers_old_chunks_and_new_per_sample_artifacts(tmp_path):
    policy = eval_vllm.generation_policy(args(tmp_path))
    save_artifact(tmp_path / "debug_eval_chunk_shard0_start0_end4.pt", policy, [0, 4])
    save_artifact(tmp_path / "debug_eval_chunk_shard1_start1_end1.pt", policy, [1])

    incompatible = dict(policy)
    incompatible["max_response_len"] = 100
    save_artifact(tmp_path / "debug_eval_chunk_shard2_start2_end2.pt", incompatible, [2])

    completed = eval_vllm.discover_completed_samples(tmp_path, policy, set(range(5)))
    assert set(completed) == {0, 1, 4}


def test_resume_rejects_same_policy_duplicate_indices(tmp_path):
    policy = eval_vllm.generation_policy(args(tmp_path))
    save_artifact(tmp_path / "debug_eval_chunk_shard0_start0_end0.pt", policy, [0])
    save_artifact(tmp_path / "debug_eval_chunk_shard1_start0_end4.pt", policy, [0, 4])

    with pytest.raises(RuntimeError, match="Duplicate eval_index=0"):
        eval_vllm.discover_completed_samples(tmp_path, policy, set(range(5)))


def test_optimized_operational_config_is_not_semantic_policy(tmp_path):
    eval_args = args(tmp_path)
    runtime = eval_vllm.operational_config(eval_args, "0.24.0")

    assert runtime["max_num_seqs"] == 24
    assert runtime["max_num_batched_tokens"] == 16384
    assert runtime["offline_llm"] is True
    assert runtime["async_scheduling"] is False
    assert runtime["speculative_config"] is None
    assert "max_num_seqs" not in eval_vllm.generation_policy(eval_args)


def test_scorer_sanitizes_terminal_control_tokens():
    response = "<think>work</think>Final answer is \\boxed{4}.<|im_end|>"
    assert summarize_eval.scoring_response(response).endswith("\\boxed{4}.")
    assert summarize_eval.extract_prediction(summarize_eval.scoring_response(response)) == "4"
