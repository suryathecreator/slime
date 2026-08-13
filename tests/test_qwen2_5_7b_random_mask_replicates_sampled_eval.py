from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.qwen2_5_7b_openr1_math220k_correct_recipe_random_mask_replicates_sft_eval_handoff import (
    generate_math500 as generation,
)
from examples.qwen2_5_7b_openr1_math220k_correct_recipe_random_mask_replicates_sft_eval_handoff import (
    math500_eval as control,
)


NUM_GPUS = 0
ROOT = Path(__file__).resolve().parents[1]
HANDOFF = (
    ROOT
    / "examples/qwen2_5_7b_openr1_math220k_correct_recipe_random_mask_replicates_sft_eval_handoff"
)
EXPECTED_TARGETS = (
    "random_mask_05_seed42",
    "random_mask_15_seed42",
    "random_mask_05_seed43",
    "random_mask_15_seed43",
    "random_mask_05_seed44",
    "random_mask_15_seed44",
)


def test_inventory_is_exact_mask_rate_by_training_seed_product() -> None:
    inventory = control.load_inventory(ROOT)
    assert control.TARGETS == EXPECTED_TARGETS
    assert tuple(item["id"] for item in inventory["checkpoints"]) == EXPECTED_TARGETS
    assert [
        (item["mask_probability"], item["training_mask_seed"])
        for item in inventory["checkpoints"]
    ] == [
        (0.05, 42),
        (0.15, 42),
        (0.05, 43),
        (0.15, 43),
        (0.05, 44),
        (0.15, 44),
    ]
    assert inventory["contract_hash"] == "9c5acfdc3d3a6b44"
    assert inventory["sharding"]["array_tasks_per_checkpoint"] == 12


def test_every_transferred_checkpoint_manifest_and_tokenizer_is_present() -> None:
    inventory = control.load_inventory(ROOT)
    expected_tokenizer_hashes = {
        "chat_template.jinja": inventory["tokenizer"]["chat_template_sha256"],
        "tokenizer.json": inventory["tokenizer"]["tokenizer_json_sha256"],
        "tokenizer_config.json": inventory["tokenizer"]["tokenizer_config_sha256"],
    }
    for item in inventory["checkpoints"]:
        model = control.absolute(ROOT, item, "model")
        manifest_path = control.absolute(ROOT, item, "manifest")
        manifest = control.load_json(manifest_path)
        assert model.is_dir()
        assert manifest["variant"] == f"qwen2_5_7b_8k.{item['id']}"
        assert manifest["contract_hash"] == "9c5acfdc3d3a6b44"
        assert manifest["family"] == "2k"
        assert manifest["final_iteration"] == 124
        assert len(list(model.glob("model-*-of-00004.safetensors"))) == 4
        for name, expected_hash in expected_tokenizer_hashes.items():
            assert control.sha256_file(model / name) == expected_hash


def test_prior_aggregate_is_hash_pinned_and_policy_identical() -> None:
    inventory = control.load_inventory(ROOT)
    prior = control.validate_prior_results(ROOT, inventory)
    assert tuple(prior) == control.PRIOR_TARGETS
    assert inventory["prior_results"]["sha256"] == (
        "2d466abb65afe6c32357247d7af11c7c871cb3905c341d3aeef679e11fbdd6b3"
    )
    assert prior["base_qwen2_5_7b"]["sampled_accuracy"]["values"] == [
        0.556,
        0.556,
        0.558,
    ]
    assert prior["qwen2_5_7b_8k"]["sampled_accuracy"]["values"] == [
        0.704,
        0.726,
        0.73,
    ]


def test_primary_and_replicate_table_membership_is_frozen() -> None:
    assert control.PRIMARY_TARGETS == (
        "base_qwen2_5_7b",
        "qwen2_5_7b_8k",
        "unmasked",
        "random_mask_05_seed42",
        "random_mask_15_seed42",
        "random_mask_25",
        "random_mask_50",
        "random_mask_70",
        "random_mask_80",
        "random_mask_90",
        "inverse_tau_0p20",
        "inverse_tau_0p05",
        "margin_mask",
        "prob_ratio_mask",
    )
    assert control.TARGETS == EXPECTED_TARGETS
    assert set(control.PRIMARY_TARGETS) - set(control.PRIOR_TARGETS) == {
        "random_mask_05_seed42",
        "random_mask_15_seed42",
    }


def test_array_layout_is_three_complete_passes_per_checkpoint() -> None:
    assert [control.pass_spec(index) for index in range(12)] == [
        *(("sampled", repeat, shard) for repeat in range(3) for shard in range(4))
    ]
    assert len(control.TARGETS) * 12 == 72
    with pytest.raises(ValueError):
        control.pass_spec(12)


def test_sampled_decoding_seed_dynamic_context_and_stops_are_unchanged() -> None:
    assert generation.decoding() == {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": -1,
    }
    assert generation.sample_seed(0, 0) == 1234
    assert generation.sample_seed(1, 0) == 1734
    assert generation.sample_seed(2, 499) == 2733
    assert generation.effective_response_budget(512) == 32256
    assert generation.STOP_TOKENS == {
        "<|endoftext|>": 151643,
        "<|im_end|>": 151645,
    }
    with pytest.raises(RuntimeError):
        generation.effective_response_budget(32768)


def test_prompt_engine_and_last_boxed_scorer_contract_are_frozen() -> None:
    policy = json.loads((HANDOFF / "eval_policy.json").read_text())
    assert generation.rendered_messages("What is 1+1?") == [
        {
            "role": "user",
            "content": (
                "Please reason step by step, and put your final answer within "
                "\\boxed{}.\n\nProblem:\nWhat is 1+1?"
            ),
        }
    ]
    assert policy["response_budget"]["formula"] == "32768 - rendered_prompt_tokens"
    assert policy["engine"] == {
        "chunk_size": 32,
        "dtype": "bfloat16",
        "enable_prefix_caching": True,
        "gpu_memory_utilization": 0.9,
        "max_model_len": 32768,
        "max_num_seqs": 8,
        "tensor_parallel_size": 1,
    }
    assert policy["scoring"] == {
        "cap_hits": "score saved response identically to natural stops",
        "candidate_selection": "last complete nonempty balanced boxed group only",
        "non_boxed_fallback": None,
        "scorer": "math500_last_boxed_symmetric_scorer_v6",
    }


def test_submission_shape_push_gate_and_job_local_caches_are_frozen() -> None:
    submit = (HANDOFF / "submit.sh").read_text()
    worker = (HANDOFF / "run_h200.sbatch").read_text()
    canary = (HANDOFF / "canary_h200.sbatch").read_text()
    control_source = (HANDOFF / "math500_eval.py").read_text()
    assert "--array=0-11" in submit
    assert "targets=6 tasks=72" in submit
    assert '"eval_task_count": 72' in control_source
    assert "Evaluator commit is not pushed" in submit
    assert "origin/$BRANCH" in submit
    assert "qwen2_5:random_mask_05_seed42:sampled:0" in canary
    assert "TRITON_CACHE_DIR" in worker
    assert "VLLM_CACHE_ROOT" in worker
    assert "--time=02:00:00" in submit
    assert "--exclude=g3130" in submit


def test_sampled_statistic_is_sample_standard_deviation() -> None:
    result = control.mean_sd([0.50, 0.52, 0.54])
    assert result["mean"] == pytest.approx(0.52)
    assert result["sample_std"] == pytest.approx(0.02)
    with pytest.raises(ValueError):
        control.mean_sd([0.50, 0.52])


def test_audit_emits_primary_and_six_row_replicate_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result_directory = tmp_path / "results"
    control_directory = tmp_path / "control"
    inventory = {
        "checkpoints": [
            {
                "id": target,
                "mask_probability": 0.05 if "_05_" in target else 0.15,
                "training_mask_seed": int(target[-2:]),
            }
            for target in EXPECTED_TARGETS
        ],
        "prior_results": {"path": "prior.json", "sha256": "0" * 64},
    }

    def metric(values: list[float]) -> dict[str, object]:
        return control.mean_sd(values)

    prior = {
        target: {
            "cap_hit_correct": 0,
            "cap_hits": 0,
            "generated_length_mean": metric([10.0, 10.0, 10.0]),
            "parse_failures": 0,
            "sampled_accuracy": metric([0.5, 0.5, 0.5]),
            "sampled_cap_hit_rate": metric([0.0, 0.0, 0.0]),
            "sampled_valid_box_rate": metric([1.0, 1.0, 1.0]),
            "source": "generated",
            "target": target,
        }
        for target in control.PRIOR_TARGETS
    }
    for target in EXPECTED_TARGETS:
        repeats = [
            {
                "accuracy": 0.6,
                "cap_hit_correct": 0,
                "cap_hit_rate": 0.0,
                "cap_hits": 0,
                "generated_length_mean": 11.0,
                "mode": "sampled",
                "parse_failure_count": 0,
                "parse_failure_rate": 0.0,
                "repeat": repeat,
                "scorer_version": control.SCORER_VERSION,
                "target": target,
                "total": 500,
                "valid_box_rate": 1.0,
            }
            for repeat in range(3)
        ]
        result = {
            "sampled": {
                "aggregate": {
                    field: metric([float(row[field]) for row in repeats])
                    for field in (
                        "accuracy",
                        "cap_hit_rate",
                        "generated_length_mean",
                        "parse_failure_rate",
                        "valid_box_rate",
                    )
                },
                "repeats": repeats,
            },
            "scorer_version": control.SCORER_VERSION,
            "target": target,
        }
        target_directory = result_directory / target
        target_directory.mkdir(parents=True)
        (target_directory / "RESULTS.json").write_text(json.dumps(result))

    monkeypatch.setattr(control, "repo_root", lambda _args: tmp_path)
    monkeypatch.setattr(control, "load_inventory", lambda *_args: inventory)
    monkeypatch.setattr(control, "validate_prior_results", lambda *_args: prior)
    monkeypatch.setattr(control, "result_root", lambda _root: result_directory)
    monkeypatch.setattr(control, "control_root", lambda _root: control_directory)
    control.audit(SimpleNamespace(manifest="unused", repo_root=str(tmp_path)))

    aggregate = json.loads((result_directory / "RESULTS.json").read_text())
    assert [row["target"] for row in aggregate["primary_results"]] == list(
        control.PRIMARY_TARGETS
    )
    assert [row["target"] for row in aggregate["replicate_results"]] == list(
        EXPECTED_TARGETS
    )
    assert len(aggregate["comparisons"]) == 12
    assert len(json.loads((result_directory / "PRIMARY_RESULTS.json").read_text())["results"]) == 14
    assert len(json.loads((result_directory / "REPLICATE_RESULTS.json").read_text())["results"]) == 6
    markdown = (result_directory / "RESULTS.md").read_text()
    assert "Primary comparison (training mask seed 42)" in markdown
    assert "5%/15% training-mask seed replicates" in markdown


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
