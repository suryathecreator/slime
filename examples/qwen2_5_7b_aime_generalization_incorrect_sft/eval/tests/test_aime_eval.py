from __future__ import annotations

import inspect
import json
import struct
from argparse import Namespace
from pathlib import Path

import pytest
from examples.qwen2_5_7b_aime_generalization_incorrect_sft.eval import aime_eval, generate_aime
from examples.qwen2_5_7b_aime_generalization_incorrect_sft.eval.generate_aime import (
    DECODING,
    effective_response_budget,
    generation_policy,
    read_jsonl_recover_truncated_tail,
    sample_seed,
)
from examples.qwen2_5_7b_aime_generalization_incorrect_sft.eval.tokenizer_compat import (
    prepare_overlay,
)
from examples.qwen2_5_7b_aime_generalization_incorrect_sft.eval.verify_transfer import (
    validate_hf_checkpoint,
)

NUM_GPUS = 0
PROMPT_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def write_split(path: Path, split: str, prefix: str) -> None:
    rows = []
    for index in range(400):
        question = f"Question {prefix} {index}"
        rows.append(
            {
                "answer": index % 1000,
                "part": "I" if index % 2 == 0 else "II",
                "problem_id": f"{prefix}-{index}",
                "problem_number": index % 15 + 1,
                "prompt": question + PROMPT_SUFFIX,
                "question": question,
                "split": split,
                "year": 2000 + index % 20,
            }
        )
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def args(tmp_path: Path) -> Namespace:
    return Namespace(
        repo_root=str(Path(__file__).resolve().parents[4]),
        experiment_root=str(tmp_path / "experiment"),
    )


def test_sampling_and_context_policy_are_exact() -> None:
    assert DECODING == {"temperature": 0.7, "top_p": 0.8, "top_k": -1, "n": 1}
    assert sample_seed(0, 0) == 1234
    assert sample_seed(2, 399) == 2433
    assert effective_response_budget(123) == 32645
    with pytest.raises(RuntimeError, match="no response budget"):
        effective_response_budget(32768)
    policy = generation_policy(
        target="base_qwen2_5_7b",
        dataset="held_in",
        repeat=2,
        checkpoint_identity="a" * 64,
        eval_file_sha256="b" * 64,
        tokenizer_overlay_identity="c" * 64,
    )
    assert policy["decoding"] == DECODING
    assert policy["seed_formula"] == "1234 + 400 * repeat + eval_index"
    assert policy["stop_token_ids"] == [151643, 151645]
    assert policy["response_budget"] == "32768 - rendered_prompt_tokens"
    assert policy["target_context"] == 32768
    assert policy["tokenizer_overlay_identity"] == "c" * 64


def test_generator_forwards_transferred_prompt_verbatim() -> None:
    source = inspect.getsource(generate_aime.run)
    assert '[{"role": "user", "content": str(row["prompt"])}]' in source
    assert "PROMPT_SUFFIX" not in source
    assert "Problem:" not in source


def test_generation_resume_recovers_only_one_unterminated_trailing_record(
    tmp_path: Path,
) -> None:
    valid = json.dumps({"eval_index": 0}) + "\n"
    records = tmp_path / "records.jsonl"
    records.write_bytes(valid.encode() + b'{"eval_index":')
    assert read_jsonl_recover_truncated_tail(records) == [{"eval_index": 0}]
    assert records.read_text(encoding="utf-8") == valid

    records.write_bytes(valid.encode() + b'{"broken":\n' + valid.encode())
    with pytest.raises(RuntimeError, match="invalid JSONL"):
        read_jsonl_recover_truncated_tail(records)

    records.write_bytes(valid.encode() + b'{"broken":\n')
    with pytest.raises(RuntimeError, match="invalid JSONL"):
        read_jsonl_recover_truncated_tail(records)


def test_eval_config_names_paths_and_keys_are_explicit() -> None:
    text = aime_eval.EVAL_CONFIG.read_text(encoding="utf-8")
    assert "name: held_in" in text
    assert "path: ${oc.env:HELD_IN_EVAL_JSONL}" in text
    assert "name: held_out_problem" in text
    assert "path: ${oc.env:HELD_OUT_EVAL_JSONL}" in text
    assert text.count("input_key: prompt") == 3
    assert text.count("label_key: answer") == 3


def test_prepare_allows_contest_overlap_but_rejects_problem_overlap(tmp_path: Path) -> None:
    held_in = tmp_path / "held_in.jsonl"
    held_out = tmp_path / "held_out.jsonl"
    write_split(held_in, "held_in", "in")
    write_split(held_out, "held_out_problem", "out")
    namespace = args(tmp_path)
    namespace.held_in = str(held_in)
    namespace.held_out = str(held_out)
    aime_eval.prepare(namespace)
    metadata = json.loads((aime_eval.control_root(namespace) / "eval_benchmarks/metadata.json").read_text())
    assert metadata["problem_id_overlap"] == 0
    assert metadata["contest_overlap_count"] == 20
    for split in aime_eval.SPLITS:
        assert len(aime_eval.read_jsonl(aime_eval.benchmark_full(namespace, split))) == 400
        assert all(
            len(aime_eval.read_jsonl(aime_eval.benchmark_shard(namespace, split, shard))) == 100 for shard in range(4)
        )

    rows = aime_eval.read_jsonl(held_out)
    rows[0]["problem_id"] = "in-0"
    held_out.write_text(aime_eval.jsonl_text(rows), encoding="utf-8")
    with pytest.raises(RuntimeError, match="problem IDs overlap"):
        aime_eval.prepare(namespace)


def test_prepare_rejects_noncanonical_prompt_and_2024(tmp_path: Path) -> None:
    source = tmp_path / "held_in.jsonl"
    write_split(source, "held_in", "in")
    rows = aime_eval.read_jsonl(source)
    rows[0]["prompt"] = "changed"
    source.write_text(aime_eval.jsonl_text(rows), encoding="utf-8")
    with pytest.raises(RuntimeError, match="noncanonical prompt"):
        aime_eval.normalize_rows(source, "held_in")
    rows[0]["prompt"] = rows[0]["question"] + PROMPT_SUFFIX
    rows[0]["year"] = 2024
    source.write_text(aime_eval.jsonl_text(rows), encoding="utf-8")
    with pytest.raises(RuntimeError, match="excluded year"):
        aime_eval.normalize_rows(source, "held_in")


def test_held_out_source_label_is_exact(tmp_path: Path) -> None:
    source = tmp_path / "held_out.jsonl"
    write_split(source, "held_out", "out")
    with pytest.raises(RuntimeError, match="invalid source split"):
        aime_eval.normalize_rows(source, "held_out_problem")
    write_split(source, "held_out_problem", "out")
    assert len(aime_eval.normalize_rows(source, "held_out_problem")) == 400


def test_target_contract_and_array_shape() -> None:
    assert len(aime_eval.TARGETS) == 13
    assert aime_eval.TARGETS[0:2] == (
        "base_qwen2_5_7b",
        "aime_correct_3000",
    )
    assert aime_eval.TARGETS[-1] == "continue_correct_only_1500"
    assert aime_eval.INTERIM_TARGETS == (
        "base_qwen2_5_7b",
        "aime_correct_3000",
        "continue_wrong_unmasked",
        "continue_correct_only_1500",
        "continue_random_mask_50",
    )
    assert len(aime_eval.DEFERRED_TARGETS) == 8
    assert set(aime_eval.INTERIM_TARGETS).isdisjoint(aime_eval.DEFERRED_TARGETS)
    assert set(aime_eval.INTERIM_TARGETS) | set(aime_eval.DEFERRED_TARGETS) == set(aime_eval.TARGETS)
    assert len(aime_eval.TARGETS) * len(aime_eval.SPLITS) * 3 * 400 == 31200


def test_checkpoint_subprocess_reuses_pinned_python() -> None:
    source = inspect.getsource(aime_eval.verify_checkpoints)
    assert "sys.executable" in source
    assert '["python3"' not in source


def test_tokenizer_template_may_be_embedded_or_external(tmp_path: Path) -> None:
    template = "{{ messages[0]['content'] }}"
    embedded = tmp_path / "embedded"
    external = tmp_path / "external"
    embedded.mkdir()
    external.mkdir()
    (embedded / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (external / "tokenizer.json").write_text("{ }\n", encoding="utf-8")
    (embedded / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "chat_template": template,
                "eos_token": "<|endoftext|>",
                "pad_token": "<|endoftext|>",
            }
        ),
        encoding="utf-8",
    )
    (external / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "chat_template": None,
                "eos_token": "<|endoftext|>",
                "pad_token": "<|endoftext|>",
            }
        ),
        encoding="utf-8",
    )
    (external / "chat_template.jinja").write_text(template, encoding="utf-8")
    embedded_contract = aime_eval.tokenizer_file_contract(embedded)
    external_contract = aime_eval.tokenizer_file_contract(external)
    assert embedded_contract["chat_template_sha256"] == external_contract["chat_template_sha256"]
    assert embedded_contract["chat_template_external_sha256"] is None
    assert external_contract["chat_template_embedded_sha256"] is None


def test_legacy_tokenizer_config_gets_manifest_preserving_overlay(tmp_path: Path) -> None:
    model = tmp_path / "checkpoint"
    model.mkdir()
    template = "{{ messages[0]['content'] }}"
    legacy_tokens = ["<|im_start|>", "<|im_end|>"]
    source_config = {
        "chat_template": None,
        "eos_token": "<|endoftext|>",
        "extra_special_tokens": legacy_tokens,
        "tokenizer_class": "Qwen2Tokenizer",
    }
    (model / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (model / "tokenizer_config.json").write_text(json.dumps(source_config), encoding="utf-8")
    (model / "chat_template.jinja").write_text(template, encoding="utf-8")

    overlay, metadata = prepare_overlay(tmp_path / "overlays", "trained", model, "a" * 64)
    normalized = json.loads((overlay / "tokenizer_config.json").read_text())
    assert normalized["extra_special_tokens"] == {}
    assert normalized["additional_special_tokens"] == legacy_tokens
    assert normalized["chat_template"] == template
    assert metadata["normalization_action"] == "legacy_list_to_additional_special_tokens"
    assert (overlay / "tokenizer.json").resolve() == (model / "tokenizer.json").resolve()
    assert json.loads((model / "tokenizer_config.json").read_text()) == source_config


def test_tokenizer_overlay_rejects_conflicting_special_token_lists(tmp_path: Path) -> None:
    model = tmp_path / "checkpoint"
    model.mkdir()
    (model / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (model / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "additional_special_tokens": ["<different>"],
                "chat_template": "{{ messages }}",
                "extra_special_tokens": ["<legacy>"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="conflicting additional_special_tokens"):
        prepare_overlay(tmp_path / "overlays", "trained", model, "a" * 64)


def test_transfer_accepts_only_final_hf_checkpoint_files(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (checkpoint / name).write_text("{}\n", encoding="utf-8")
    header = json.dumps(
        {"weight": {"data_offsets": [0, 4], "dtype": "F32", "shape": [1]}},
        separators=(",", ":"),
    ).encode()
    header += b" " * (-len(header) % 8)
    weight = checkpoint / "model.safetensors"
    weight.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
    files = [
        {"bytes": path.stat().st_size, "name": path.name, "sha256": "0" * 64} for path in sorted(checkpoint.iterdir())
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"checkpoint_manifest_sha256": "a" * 64, "files": files}),
        encoding="utf-8",
    )
    validate_hf_checkpoint(checkpoint, manifest, "unit", "a" * 64)
    weight.write_bytes(weight.read_bytes()[:-1])
    with pytest.raises(ValueError, match="offsets out of bounds"):
        validate_hf_checkpoint(checkpoint, manifest, "unit", "a" * 64)
    weight.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
    (checkpoint / "optimizer_state.pt").write_bytes(b"not allowed")
    with pytest.raises(RuntimeError, match="non-HF training state"):
        validate_hf_checkpoint(checkpoint, manifest, "unit", "a" * 64)


def test_every_hyak_job_is_revision_pinned_and_results_stay_outside_repo() -> None:
    submit = (aime_eval.EVAL_DIR / "submit_hyak.sh").read_text(encoding="utf-8")
    assert "AIME_EXPECTED_GIT_COMMIT=${SUBMIT_COMMIT}" in submit
    assert "PYTHONDONTWRITEBYTECODE=1" in submit
    assert '--chdir="${REPO_ROOT}"' in submit
    assert "ALL,REPO_ROOT=${REPO_ROOT}" in submit
    for name in (
        "preflight_cpu.sbatch",
        "canary_h200.sbatch",
        "run_h200.sbatch",
        "finalize_cpu.sbatch",
        "interim_cpu.sbatch",
        "audit_cpu.sbatch",
    ):
        text = (aime_eval.EVAL_DIR / name).read_text(encoding="utf-8")
        assert "runtime_repo_gate.py" in text, name
        assert "AIME_EXPECTED_GIT_COMMIT" in text, name
    namespace = args(Path("/tmp/aime-result-root-test"))
    assert aime_eval.result_root(namespace) == (
        Path(namespace.experiment_root).resolve() / "results/aime_generalization_sampled_v1"
    )


def test_submission_script_encodes_priority_and_interim_dependencies() -> None:
    submit = (aime_eval.EVAL_DIR / "submit_hyak.sh").read_text(encoding="utf-8")
    assert "PRIORITY_TARGETS=(" in submit
    assert "DEFERRED_TARGETS=(" in submit
    assert 'submit_target "${BASE_TARGET}" "${canary_job}"' in submit
    assert 'priority_predecessor="${TARGET_FINALIZER_JOB}"' in submit
    assert '--dependency="afterok:${base_finalizer_job}:${priority_predecessor}"' in submit
    assert 'submit_target "${target}" "${interim_job}"' in submit
    assert '--interim-job "${interim_job}"' in submit
    assert '"${MODE}" == "--resubmit"' in submit
    assert 'scancel "${ACTIVE_PRIOR_JOBS[@]}"' in submit
    canary = (aime_eval.EVAL_DIR / "canary_h200.sbatch").read_text(encoding="utf-8")
    assert "CANARY_TARGET=aime_correct_3000" in canary


def test_merge_rejects_generation_policy_drift(tmp_path: Path) -> None:
    held_in = tmp_path / "held_in.jsonl"
    held_out = tmp_path / "held_out.jsonl"
    write_split(held_in, "held_in", "in")
    write_split(held_out, "held_out_problem", "out")
    namespace = args(tmp_path)
    namespace.held_in = str(held_in)
    namespace.held_out = str(held_out)
    aime_eval.prepare(namespace)

    identity = "a" * 64
    model = tmp_path / "model"
    model.mkdir()
    (model / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (model / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages[0]['content'] }}", "extra_special_tokens": {}}),
        encoding="utf-8",
    )
    inventory = {
        "comparison_checkpoint_count": len(aime_eval.TARGETS),
        "targets": [
            {"checkpoint_identity": identity, "id": target, "source_checkpoint": str(model)}
            for target in aime_eval.TARGETS
        ],
    }
    path = aime_eval.inventory_path(namespace)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(aime_eval.canonical_json(inventory), encoding="utf-8")

    target = "base_qwen2_5_7b"
    _, overlay_metadata = prepare_overlay(
        aime_eval.tokenizer_overlay_root(namespace), target, model, identity
    )
    split = "held_in"
    repeat = 0
    for shard in range(aime_eval.SHARDS):
        eval_file = aime_eval.benchmark_shard(namespace, split, shard)
        policy = generation_policy(
            target=target,
            dataset=split,
            repeat=repeat,
            checkpoint_identity=identity,
            eval_file_sha256=aime_eval.sha256_file(eval_file),
            tokenizer_overlay_identity=overlay_metadata["overlay_identity"],
        )
        fingerprint = aime_eval.sha256_bytes(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode())
        output = aime_eval.pass_output(namespace, target, split, repeat) / "shards" / f"shard_{shard:02d}"
        output.mkdir(parents=True, exist_ok=True)
        (output / "generation_policy.json").write_text(aime_eval.canonical_json(policy), encoding="utf-8")
        records = []
        for row in aime_eval.read_jsonl(eval_file):
            records.append(
                {
                    "answer": row["answer"],
                    "cap_hit": False,
                    "checkpoint_manifest_sha256": identity,
                    "dataset": split,
                    "effective_max_response_tokens": 32758,
                    "eval_index": row["eval_index"],
                    "finish_reason": "stop",
                    "generated_token_count": 0,
                    "generated_token_ids": [],
                    "policy_sha256": fingerprint,
                    "problem_id": row["problem_id"],
                    "prompt_sha256": aime_eval.sha256_bytes(row["prompt"].encode()),
                    "rendered_prompt_tokens": 10,
                    "repeat": repeat,
                    "response": "",
                    "response_sha256": aime_eval.sha256_bytes(b""),
                    "seed": sample_seed(repeat, row["eval_index"]),
                    "target": target,
                    "total_context_limit": 32768,
                    "tokenizer_overlay_identity": overlay_metadata["overlay_identity"],
                }
            )
        (output / "records.jsonl").write_text(aime_eval.jsonl_text(records), encoding="utf-8")
    assert len(aime_eval.records_for_pass(namespace, target, split, repeat)) == 400

    first_policy = aime_eval.pass_output(namespace, target, split, repeat) / "shards/shard_00/generation_policy.json"
    changed = json.loads(first_policy.read_text(encoding="utf-8"))
    changed["decoding"]["temperature"] = 0.0
    first_policy.write_text(aime_eval.canonical_json(changed), encoding="utf-8")
    with pytest.raises(RuntimeError, match="generation policy mismatch"):
        aime_eval.records_for_pass(namespace, target, split, repeat)


def test_interim_and_final_audits_are_distinct(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(aime_eval, "PACKAGE", tmp_path / "package")
    namespace = args(tmp_path)
    common = {
        "dataset_hashes": {"held_in": "a" * 64, "held_out_problem": "b" * 64},
        "eval_config_sha256": "c" * 64,
        "generation_contract_sha256": "d" * 64,
        "scorer_commit": "e" * 40,
        "scorer_sha256": "f" * 64,
        "scorer_version": aime_eval.SCORER_VERSION,
    }
    for target_index, target in enumerate(aime_eval.TARGETS):
        splits = {}
        for split_index, split in enumerate(aime_eval.SPLITS):
            values = [0.10 + 0.001 * target_index + 0.01 * split_index] * 3
            repeats = [
                {
                    "cap_hit_correct": 0,
                    "cap_hits": 0,
                    "parse_failure_count": 1,
                }
                for _ in range(3)
            ]
            splits[split] = {
                "aggregate": {
                    "accuracy": aime_eval.mean_sd(values),
                    "cap_hit_rate": aime_eval.mean_sd([0.0, 0.0, 0.0]),
                    "generated_length_mean": aime_eval.mean_sd([100.0, 100.0, 100.0]),
                    "valid_box_rate": aime_eval.mean_sd([0.9, 0.9, 0.9]),
                },
                "repeats": repeats,
            }
        destination = aime_eval.result_root(namespace) / target / "RESULTS.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            aime_eval.canonical_json({**common, "splits": splits, "target": target}),
            encoding="utf-8",
        )
    aime_eval.interim_audit(namespace)
    interim_path = aime_eval.result_root(namespace) / "INTERIM_RESULTS.json"
    interim = json.loads(interim_path.read_text())
    assert interim["status"] == "interim"
    assert interim["included_targets"] == list(aime_eval.INTERIM_TARGETS)
    assert interim["pending_targets"] == [target for target in aime_eval.TARGETS if target not in aime_eval.INTERIM_TARGETS]
    interim_report = (aime_eval.result_root(namespace) / "INTERIM_RESULTS.md").read_text(encoding="utf-8")
    for target in aime_eval.INTERIM_TARGETS:
        assert f"| {target} |" in interim_report
    for target in aime_eval.DEFERRED_TARGETS:
        assert f"| {target} |" not in interim_report

    aime_eval.audit(namespace)
    report = (aime_eval.result_root(namespace) / "RESULTS.md").read_text(encoding="utf-8")
    assert "| Target | Held-in accuracy | Held-out accuracy |" in report
    assert "| Target | Split | Valid box | Cap hit |" in report
    for target in aime_eval.TARGETS:
        assert f"| {target} |" in report
    assert json.loads(interim_path.read_text()) == interim
    assert json.loads((aime_eval.control_root(namespace) / "FINAL_AUDIT.json").read_text())["status"] == "complete"


def test_submission_metadata_records_staged_dependency_graph(tmp_path: Path) -> None:
    namespace = args(tmp_path)
    namespace.preflight_job = "100"
    namespace.canary_job = "101"
    namespace.array_job = [f"{target}={200 + index}" for index, target in enumerate(aime_eval.TARGETS)]
    namespace.finalizer_job = [f"{target}={300 + index}" for index, target in enumerate(aime_eval.TARGETS)]
    namespace.interim_job = "400"
    namespace.audit_job = "500"
    namespace.git_commit = "a" * 40
    namespace.attempt = 1
    namespace.supersedes_journal = None
    namespace.failure_kind = None
    namespace.failure_log = None
    aime_eval.record_submission(namespace)
    submission = json.loads((aime_eval.control_root(namespace) / "submission.json").read_text())
    assert submission["artifact_schema_version"] == 3
    assert submission["attempt"] == 1
    assert submission["supersedes"] is None
    assert submission["target_groups"]["priority"] == list(aime_eval.PRIORITY_TARGETS)
    assert submission["target_groups"]["deferred"] == list(aime_eval.DEFERRED_TARGETS)
    interim = next(edge for edge in submission["dependency_edges"] if edge["stage"] == "interim_aggregation")
    assert interim["job"] == "400"
    assert interim["afterok"] == [
        submission["finalizer_jobs"][aime_eval.BASE_TARGET],
        submission["finalizer_jobs"][aime_eval.PRIORITY_TARGETS[-1]],
    ]
    final = next(edge for edge in submission["dependency_edges"] if edge["stage"] == "final_audit")
    assert final["afterok"] == [submission["finalizer_jobs"][target] for target in aime_eval.DEFERRED_TARGETS]


def test_resubmission_requires_exact_failure_and_records_supersession(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    namespace = args(tmp_path)
    namespace.preflight_job = "100"
    namespace.canary_job = "101"
    namespace.array_job = [f"{target}={200 + index}" for index, target in enumerate(aime_eval.TARGETS)]
    namespace.finalizer_job = [f"{target}={300 + index}" for index, target in enumerate(aime_eval.TARGETS)]
    namespace.interim_job = "400"
    namespace.audit_job = "500"
    namespace.git_commit = "a" * 40
    namespace.attempt = 1
    namespace.supersedes_journal = None
    namespace.failure_kind = None
    namespace.failure_log = None
    aime_eval.record_submission(namespace)

    journal = aime_eval.control_root(namespace) / "submission.json"
    failure_log = aime_eval.control_root(namespace) / "slurm_logs/q25-aime-preflight-100.out"
    failure_log.parent.mkdir(parents=True)
    failure_log.write_text(
        "_set_model_specific_special_tokens\nAttributeError: 'list' object has no attribute 'keys'\n",
        encoding="utf-8",
    )
    retry = args(tmp_path)
    retry.journal = str(journal)
    retry.failure_log = str(failure_log)
    aime_eval.validate_resubmission(retry)
    assert len(capsys.readouterr().out.splitlines()) == 30

    archive = aime_eval.control_root(namespace) / "submission_attempts/attempt-1"
    archive.mkdir(parents=True)
    archived_journal = archive / "submission.json"
    journal.replace(archived_journal)
    namespace.preflight_job = "600"
    namespace.canary_job = "601"
    namespace.array_job = [f"{target}={700 + index}" for index, target in enumerate(aime_eval.TARGETS)]
    namespace.finalizer_job = [f"{target}={800 + index}" for index, target in enumerate(aime_eval.TARGETS)]
    namespace.interim_job = "900"
    namespace.audit_job = "901"
    namespace.attempt = 2
    namespace.supersedes_journal = str(archived_journal)
    namespace.failure_kind = "legacy_extra_special_tokens_list"
    namespace.failure_log = str(failure_log)
    aime_eval.record_submission(namespace)
    replacement = json.loads(journal.read_text())
    assert replacement["attempt"] == 2
    assert replacement["supersedes"]["failure_kind"] == "legacy_extra_special_tokens_list"
    assert replacement["supersedes"]["journal"] == str(archived_journal.resolve())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
