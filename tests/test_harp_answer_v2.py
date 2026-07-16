from __future__ import annotations

import importlib.util
import json
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/qwen3_4b_opd_harp"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def official_check(candidate, correct, **_kwargs):
    normalize = lambda value: str(value).strip().strip("$").replace(" ", "")
    return {"is_correct": normalize(candidate) == normalize(correct), "predict": candidate}


harp_package = types.ModuleType("harp_official")
harp_checker = types.ModuleType("harp_official.latex_answer_check")
harp_checker.check_one_latex_answer = official_check
sys.modules.setdefault("harp_official", harp_package)
sys.modules.setdefault("harp_official.latex_answer_check", harp_checker)
sys.path.insert(0, str(EXAMPLE))

prompt_contract = load_module("harp_prompt_contract_test", EXAMPLE / "prompt_contract.py")
scorer = load_module("harp_answer_v2_test", EXAMPLE / "harp_answer_v2.py")
sys.modules["harp_answer_v2"] = scorer
sys.modules["prompt_contract"] = prompt_contract
evaluator = load_module("evaluate_harp_vllm_test", EXAMPLE / "evaluate_harp_vllm.py")


def test_prompt_contract_is_shared_and_idempotent():
    expected = "Please reason step by step, and put your final answer within \\boxed{}.\n\nProblem:\n2+2?"
    assert prompt_contract.canonical_problem_prompt("2+2?") == expected
    assert prompt_contract.canonical_problem_prompt(expected) == expected
    row = prompt_contract.apply_prompt_contract({"prompt": "2+2?", "metadata": {}})
    assert row["prompt"] == expected
    assert row["metadata"]["prompt_contract_version"] == "qwen3_harp_boxed_v1"


def test_final_region_think_states_and_special_tokens():
    assert scorer.isolate_final_region("<think>work</think> \\boxed{4}<|im_end|>") == ("\\boxed{4}", "closed")
    assert scorer.isolate_final_region("<think>work and \\boxed{4}") == ("work and \\boxed{4}", "unclosed")
    assert scorer.isolate_final_region("answer: 4") == ("answer: 4", "tagless")
    assert scorer.isolate_final_region("<|endoftext|>") == ("", "missing")


def test_balanced_boxes_ignore_incomplete_and_keep_nested_content():
    boxes = scorer.scan_complete_boxes(r"bad \boxed{3 then \boxed{\frac{1}{2}}")
    assert [box["content"] for box in boxes] == [r"\frac{1}{2}"]
    nested = scorer.scan_complete_boxes(r"\boxed{1+\fbox{2}}")
    assert [box["content"] for box in nested] == [r"1+\fbox{2}"]


def test_last_box_and_trailing_collection_rules():
    collection = scorer.extract_candidate(r"\boxed{0}, \boxed{1}, \boxed{2}, and \boxed{4}")
    assert collection["components"] == ["0", "1", "2", "4"]
    assert collection["extraction_method"] == "box_collection"
    authoritative = scorer.extract_candidate(r"draft \boxed{1}. Correction: \boxed{2}")
    assert authoritative["candidate"] == "2"


def test_marker_priority_and_no_last_number_fallback():
    extracted = scorer.extract_candidate("Therefore 3.\nAnswer: 4.\nThus 5.")
    assert extracted["candidate"] == "4."
    assert extracted["extraction_method"] == "explicit_marker"
    long_unmarked = "This is a long discussion with many words and no explicit answer marker ending in 17"
    assert scorer.extract_candidate(long_unmarked)["candidate"] is None


def test_multi_answer_matching_is_order_independent():
    result = scorer.score_answer(r"</think>\boxed{2}, and \boxed{1}", "1 and 2")
    assert result["is_correct"] is True
    assert result["verifier_method"] == "multi_symbolic_match"


def test_cap_hit_metrics_are_scored_not_discarded():
    rows = [
        {
            "is_correct": True,
            "cap_hit": True,
            "parse_failure": False,
            "generated_token_count": 10,
            "extraction_method": "boxed",
            "verifier_method": "official_checker",
        },
        {
            "is_correct": False,
            "cap_hit": False,
            "parse_failure": True,
            "generated_token_count": 2,
            "extraction_method": "none",
            "verifier_method": "not_run",
        },
    ]
    metrics = scorer.metrics_from_predictions(rows)
    assert metrics["accuracy"] == 0.5
    assert metrics["cap_hit_accuracy"] == 1.0
    assert metrics["non_cap_accuracy"] == 0.0
    assert metrics["parse_failures"] == 1


def test_offline_worker_builds_one_engine_and_generates_whole_125_row_shard(tmp_path, monkeypatch):
    class FakeTokenizer:
        eos_token_id = 151645
        unk_token_id = -1
        pad_token_id = 151645
        eos_token = "<|im_end|>"

        def convert_tokens_to_ids(self, token):
            return {"<|im_end|>": 151645, "<|endoftext|>": 151643}[token]

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["add_generation_prompt"] is True
            assert kwargs["enable_thinking"] is True
            return messages[0]["content"] + "<assistant>"

        def encode(self, text, add_special_tokens=False):
            return list(range(max(1, len(text) // 20)))

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: FakeTokenizer())

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeCompletion:
        text = r"</think>\boxed{0}"
        token_ids = [1, 2, 3]
        finish_reason = "stop"
        stop_reason = 151645

    class FakeRequestOutput:
        outputs = [FakeCompletion()]

    class FakeLLM:
        instances = 0
        generate_calls = []

        def __init__(self, **_kwargs):
            FakeLLM.instances += 1

        def generate(self, prompts, sampling_params, use_tqdm):
            FakeLLM.generate_calls.append((len(prompts), len(sampling_params), use_tqdm))
            return [FakeRequestOutput() for _ in prompts]

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeSamplingParams
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    data = tmp_path / "harp.jsonl"
    with data.open("w", encoding="utf-8") as handle:
        for index in range(500):
            handle.write(json.dumps({"problem_id": f"p{index:03d}", "problem": f"Problem {index}", "correct_answer": "0"}) + "\n")
    args = Namespace(
        data=str(data),
        output_dir=str(tmp_path / "output"),
        model=str(tmp_path / "model"),
        tokenizer=str(tmp_path / "tokenizer"),
        shard_index=0,
        num_shards=4,
        expected_samples=500,
        max_model_len=32768,
        max_response_len=31744,
        max_num_seqs=8,
        max_num_batched_tokens=16384,
        gpu_memory_utilization=0.92,
        enforce_eager=False,
        benchmark=False,
        benchmark_total_size=64,
        benchmark_max_tokens=4096,
        resume=False,
        ready_file=None,
    )
    evaluator.run_shard(args)
    assert FakeLLM.instances == 1
    assert FakeLLM.generate_calls == [(125, 125, True)]
    predictions = evaluator.read_jsonl(Path(args.output_dir) / "shard_00_predictions.jsonl")
    assert len(predictions) == 125
    assert all(row["scorer_version"] == "harp_answer_v2" for row in predictions)
    for shard_index in (1, 2, 3):
        shard_args = Namespace(**vars(args))
        shard_args.shard_index = shard_index
        evaluator.run_shard(shard_args)
    merged_dir = tmp_path / "merged"
    evaluator.merge_production(Namespace(stage_dir=args.output_dir, data=str(data), output_dir=str(merged_dir)))
    merged = evaluator.read_jsonl(merged_dir / "predictions.jsonl")
    assert [row["problem_id"] for row in merged] == [f"p{index:03d}" for index in range(500)]
    metrics = json.loads((merged_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["num_eval_problems"] == 500
    assert metrics["correct_count"] == 500
