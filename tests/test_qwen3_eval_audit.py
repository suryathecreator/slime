from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

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


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeTokensPrompt:
    def __init__(self, *, prompt_token_ids):
        self.prompt_token_ids = prompt_token_ids


class FakeLLM:
    def __init__(self):
        self.prompts = None
        self.sampling_params = None

    def generate(self, prompts, sampling_params, use_tqdm):
        self.prompts = prompts
        self.sampling_params = sampling_params
        completion = SimpleNamespace(
            token_ids=[151667, 42, 151668],
            finish_reason="stop",
            stop_reason=151645,
            text="The answer is \\boxed{4}.",
        )
        return [SimpleNamespace(outputs=[completion])]


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
    )


def test_run_chunk_preserves_special_text_and_raw_token_ids(monkeypatch, tmp_path):
    fake_vllm = SimpleNamespace(SamplingParams=FakeSamplingParams, TokensPrompt=FakeTokensPrompt)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    tokenizer = FakeTokenizer()
    llm = FakeLLM()

    samples = eval_vllm.run_chunk(
        llm,
        tokenizer,
        [{"_eval_index": 0, "prompt": "2+2?", "label": "4", "metadata": {}}],
        args(tmp_path),
    )

    sample = samples[0]
    assert tokenizer.template_kwargs["enable_thinking"] is True
    assert llm.prompts[0].prompt_token_ids == sample["prefill_token_ids"] == [10, 11, 12]
    assert llm.sampling_params.kwargs["stop_token_ids"] == [151645, 151643]
    assert llm.sampling_params.kwargs["skip_special_tokens"] is False
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


def test_scorer_sanitizes_terminal_control_tokens():
    response = "<think>work</think>Final answer is \\boxed{4}.<|im_end|>"
    assert summarize_eval.scoring_response(response).endswith("\\boxed{4}.")
    assert summarize_eval.extract_prediction(summarize_eval.scoring_response(response)) == "4"
