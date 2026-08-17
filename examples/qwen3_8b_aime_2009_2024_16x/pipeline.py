#!/usr/bin/env python3
"""Control plane, strict scoring wrapper, and audit for the Qwen3 AIME run."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from typing import Any, Iterable

import build_corpus


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
POLICY_PATH = HERE / "eval_policy.json"
CORPUS_PATH = HERE / "aime_2009_2024.jsonl"
CORPUS_MANIFEST_PATH = HERE / "corpus_manifest.json"
SUBMISSION_PATH = HERE / "SUBMISSION.json"
EXPECTED_PROBLEMS = 480
EXPECTED_DRAWS = 16
EXPECTED_ROWS = EXPECTED_PROBLEMS * EXPECTED_DRAWS


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_policy() -> dict[str, Any]:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    required = {
        "model": {
            "id": "Qwen/Qwen3-8B",
            "revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "variant": "post-trained",
        },
        "generation": {
            "completions_per_problem": 16,
            "enable_thinking": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
        },
        "stop": {"accepted_token_ids": [151643, 151645]},
    }
    for section, expected in required.items():
        for key, value in expected.items():
            if policy.get(section, {}).get(key) != value:
                raise RuntimeError(f"policy mismatch at {section}.{key}")
    if policy["context"]["max_model_len"] != 32768:
        raise RuntimeError("only the approved 32K context policy is supported")
    if policy["corpus"]["problem_count"] != EXPECTED_PROBLEMS:
        raise RuntimeError("policy problem count changed")
    return policy


def scorer_path(policy: dict[str, Any]) -> Path:
    return REPO_ROOT / policy["scorer"]["file"]


def model_root(policy: dict[str, Any]) -> Path:
    return REPO_ROOT / policy["model"]["local_dir"]


def contract_sha256(policy: dict[str, Any] | None = None) -> str:
    value = policy or load_policy()
    payload = {
        "corpus_sha256": sha256_file(CORPUS_PATH),
        "policy": value,
        "scorer_sha256": sha256_file(scorer_path(value)),
    }
    return sha256_text(canonical_json(payload))


def output_root(policy: dict[str, Any] | None = None) -> Path:
    value = policy or load_policy()
    return (
        REPO_ROOT
        / "checkpoints/qwen3_8b_aime_2009_2024_16x/v1"
        / contract_sha256(value)[:16]
    )


def log_root(policy: dict[str, Any] | None = None) -> Path:
    return output_root(policy) / "slurm_logs"


def sample_path(root: Path, draw_index: int, eval_index: int) -> Path:
    shard = eval_index // 120
    return (
        root
        / "raw"
        / f"draw_{draw_index:02d}"
        / f"shard_{shard:02d}"
        / f"eval_{eval_index:04d}.json"
    )


def load_corpus() -> list[dict[str, Any]]:
    policy = load_policy()
    actual_hash = sha256_file(CORPUS_PATH)
    if actual_hash != policy["corpus"]["sha256"]:
        raise RuntimeError(f"corpus hash mismatch: {actual_hash}")
    rows = build_corpus.read_jsonl(CORPUS_PATH)
    build_corpus.verify_rows(rows)
    return rows


def prompt_for_problem(problem: str, policy: dict[str, Any] | None = None) -> str:
    value = policy or load_policy()
    template = value["prompt"]["template"]
    if template.count("{problem}") != 1:
        raise RuntimeError("prompt template must contain exactly one {problem} marker")
    return template.replace("{problem}", problem)


def seed_for(draw_index: int, eval_index: int, policy: dict[str, Any] | None = None) -> int:
    value = policy or load_policy()
    return int(value["generation"]["seed_base"]) + EXPECTED_PROBLEMS * draw_index + eval_index


def task_assignment(task_id: int) -> tuple[int, int, int]:
    if not 0 <= task_id < 64:
        raise ValueError(f"task ID must be in [0, 63], found {task_id}")
    draw_index = task_id // 4
    shard_index = task_id % 4
    return draw_index, shard_index * 120, (shard_index + 1) * 120


def import_scorer(policy: dict[str, Any] | None = None) -> Any:
    value = policy or load_policy()
    path = scorer_path(value)
    actual_hash = sha256_file(path)
    if actual_hash != value["scorer"]["sha256"]:
        raise RuntimeError(f"scorer hash mismatch: {actual_hash}")
    spec = importlib.util.spec_from_file_location("pinned_aime_scorer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import scorer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if module.SCORER_VERSION != value["scorer"]["version"]:
        raise RuntimeError("scorer version mismatch")
    return module


def score_accepted_answers(
    response: str,
    accepted_answers: list[int],
    *,
    finish_reason: str | None,
    stop_token_id: int | None,
    cap_hit: bool,
    problem_id: str,
    scorer: Any | None = None,
) -> dict[str, Any]:
    if not accepted_answers:
        raise RuntimeError(f"no accepted answers for {problem_id}")
    module = scorer or import_scorer()
    traces = [
        module.score_response(
            response,
            gold,
            finish_reason=finish_reason,
            stop_token_id=stop_token_id,
            cap_hit=cap_hit,
            problem_id=problem_id,
        )
        for gold in accepted_answers
    ]
    invariant_keys = (
        "generated_text_sha256",
        "is_scorable",
        "official_candidate_raw",
        "official_candidate_sha256",
        "official_extracted_source_raw",
        "official_extracted_source_span",
        "parsed_integer",
        "selected_box_event_id",
    )
    for key in invariant_keys:
        if len({canonical_json(trace.get(key)) for trace in traces}) != 1:
            raise RuntimeError(f"scorer extraction changed by gold at {problem_id}: {key}")
    matches = [
        int(gold)
        for gold, trace in zip(accepted_answers, traces, strict=True)
        if trace["is_correct"]
    ]
    return {
        "accepted_gold_decisions": [
            {
                "gold_integer": int(gold),
                "is_correct": bool(trace["is_correct"]),
                "reason": trace["official_decision_reason"],
            }
            for gold, trace in zip(accepted_answers, traces, strict=True)
        ],
        "accepted_gold_integers": [int(value) for value in accepted_answers],
        "base_scorer_trace": traces[0],
        "is_correct": bool(matches),
        "is_scorable": bool(traces[0]["is_scorable"]),
        "matched_gold_integer": matches[0] if matches else None,
        "official_candidate_raw": traces[0]["official_candidate_raw"],
        "official_decision_reason": (
            "correct_any_officially_accepted_integer"
            if matches
            else traces[0]["official_decision_reason"]
        ),
        "parsed_integer": traces[0]["parsed_integer"],
        "scorer_version": module.SCORER_VERSION,
        "wrapper_version": "aime_multi_accepted_integer_wrapper_v1",
    }


def verify_contract() -> dict[str, Any]:
    policy = load_policy()
    rows = load_corpus()
    actual_scorer_hash = sha256_file(scorer_path(policy))
    if actual_scorer_hash != policy["scorer"]["sha256"]:
        raise RuntimeError("scorer hash changed")
    manifest = json.loads(CORPUS_MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["corpus_sha256"] != policy["corpus"]["sha256"]:
        raise RuntimeError("corpus manifest disagrees with policy")
    if manifest["source_counts"] != {
        "primary_huggingface": 455,
        "supplement_live_poshenloh": 25,
    }:
        raise RuntimeError("corpus source counts changed")
    return {
        "contract_sha256": contract_sha256(policy),
        "corpus_rows": len(rows),
        "corpus_sha256": sha256_file(CORPUS_PATH),
        "output_root": str(output_root(policy)),
        "scorer_sha256": actual_scorer_hash,
    }


def download_and_verify_model() -> dict[str, Any]:
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer

    policy = load_policy()
    target = model_root(policy)
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=policy["model"]["id"],
        revision=policy["model"]["revision"],
        local_dir=str(target),
    )
    config_path = target / "config.json"
    if sha256_file(config_path) != policy["model"]["config_sha256"]:
        raise RuntimeError("downloaded model config hash mismatch")
    config = AutoConfig.from_pretrained(target, local_files_only=True)
    if config.model_type != "qwen3" or int(config.max_position_embeddings) < 32768:
        raise RuntimeError("downloaded model is not the required Qwen3 32K model")
    tokenizer = AutoTokenizer.from_pretrained(target, local_files_only=True)
    expected_tokens = {"<|endoftext|>": 151643, "<|im_end|>": 151645}
    for token, expected in expected_tokens.items():
        actual = tokenizer.convert_tokens_to_ids(token)
        if actual != expected:
            raise RuntimeError(f"tokenizer mismatch for {token}: {actual}")
    first = load_corpus()[0]
    messages = [{"role": "user", "content": prompt_for_problem(first["problem"], policy)}]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if "<|im_start|>system" in rendered:
        raise RuntimeError("chat template unexpectedly injected a system message")
    if not rendered.endswith("<|im_start|>assistant\n"):
        raise RuntimeError("chat template generation prefix changed")
    files: list[dict[str, Any]] = []
    tensor_bytes = 0
    for path in sorted(target.rglob("*")):
        if not path.is_file() or ".cache" in path.parts:
            continue
        relative = path.relative_to(target).as_posix()
        size = path.stat().st_size
        item: dict[str, Any] = {"name": relative, "size": size}
        if path.suffix != ".safetensors":
            item["sha256"] = sha256_file(path)
        else:
            tensor_bytes += size
        files.append(item)
    if tensor_bytes < 14_000_000_000:
        raise RuntimeError(f"model snapshot appears incomplete: {tensor_bytes} tensor bytes")
    return {
        "chat_template_probe_sha256": sha256_text(rendered),
        "files": files,
        "model_id": policy["model"]["id"],
        "model_revision": policy["model"]["revision"],
        "model_root": str(target),
        "tensor_bytes": tensor_bytes,
    }


def preflight() -> None:
    contract = verify_contract()
    policy = load_policy()
    runtime = {
        name: importlib.metadata.version(name)
        for name in ("transformers", "vllm", "huggingface-hub")
    }
    if runtime["transformers"] != policy["runtime"]["transformers"]:
        raise RuntimeError(f"transformers runtime changed: {runtime}")
    if runtime["vllm"] != policy["runtime"]["vllm"]:
        raise RuntimeError(f"vLLM runtime changed: {runtime}")
    model = download_and_verify_model()
    result = {
        **contract,
        "completed_at": now_iso(),
        "model": model,
        "runtime": runtime,
        "status": "valid",
    }
    destination = output_root(policy) / "preflight.json"
    atomic_text(destination, pretty_json(result))
    print(
        f"QWEN3_AIME_PREFLIGHT_VALID contract={contract['contract_sha256']} "
        f"model_bytes={model['tensor_bytes']} path={destination}",
        flush=True,
    )


def compatible_sample(path: Path, contract: str, draw: int, eval_index: int) -> bool:
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        row.get("contract_sha256") == contract
        and row.get("draw_index") == draw
        and row.get("eval_index") == eval_index
        and row.get("response_sha256") == sha256_text(str(row.get("response", "")))
    )


def rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def metric_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(bool(row["scorer_trace"]["is_correct"]) for row in rows)
    scorable = sum(bool(row["scorer_trace"]["is_scorable"]) for row in rows)
    complete_box = sum(
        row["scorer_trace"]["base_scorer_trace"]["selected_box_event_id"] is not None
        for row in rows
    )
    cap_hits = sum(bool(row["cap_hit"]) for row in rows)
    output_lengths = [int(row["output_tokens"]) for row in rows]
    accuracy = rate(correct, total)
    return {
        "accuracy": accuracy,
        "accuracy_standard_error": math.sqrt(accuracy * (1.0 - accuracy) / total) if total else 0.0,
        "cap_hit_count": cap_hits,
        "cap_hit_rate": rate(cap_hits, total),
        "complete_box_count": complete_box,
        "complete_box_rate": rate(complete_box, total),
        "correct_count": correct,
        "mean_output_tokens": statistics.fmean(output_lengths) if output_lengths else 0.0,
        "row_count": total,
        "scorable_count": scorable,
        "scorable_rate": rate(scorable, total),
    }


def audit() -> None:
    policy = load_policy()
    corpus = load_corpus()
    scorer = import_scorer(policy)
    contract = contract_sha256(policy)
    root = output_root(policy)
    rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    missing: list[str] = []
    for draw in range(EXPECTED_DRAWS):
        for eval_index, source in enumerate(corpus):
            path = sample_path(root, draw, eval_index)
            if not path.is_file():
                missing.append(str(path))
                continue
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("contract_sha256") != contract:
                raise RuntimeError(f"contract mismatch in {path}")
            if row.get("draw_index") != draw or row.get("eval_index") != eval_index:
                raise RuntimeError(f"sample identity mismatch in {path}")
            if row.get("problem_id") != source["problem_id"]:
                raise RuntimeError(f"problem identity mismatch in {path}")
            if row.get("seed") != seed_for(draw, eval_index, policy):
                raise RuntimeError(f"seed mismatch in {path}")
            if row.get("prompt") != prompt_for_problem(source["problem"], policy):
                raise RuntimeError(f"prompt mismatch in {path}")
            if row.get("response_sha256") != sha256_text(str(row.get("response", ""))):
                raise RuntimeError(f"response hash mismatch in {path}")
            recomputed = score_accepted_answers(
                row["response"],
                source["accepted_answer_integers"],
                finish_reason=row.get("finish_reason"),
                stop_token_id=row.get("stop_token_id"),
                cap_hit=bool(row["cap_hit"]),
                problem_id=source["problem_id"],
                scorer=scorer,
            )
            if canonical_json(recomputed) != canonical_json(row.get("scorer_trace")):
                raise RuntimeError(f"saved scorer trace differs on CPU rescore: {path}")
            rows.append(row)
            artifacts.append(
                {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}
            )
    if missing:
        raise RuntimeError(f"missing {len(missing)} of {EXPECTED_ROWS} outputs; first={missing[0]}")
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"expected {EXPECTED_ROWS} rows, found {len(rows)}")

    grouped_year: dict[int, list[dict[str, Any]]] = defaultdict(list)
    grouped_draw: dict[int, list[dict[str, Any]]] = defaultdict(list)
    grouped_contest: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_year[int(row["year"])].append(row)
        grouped_draw[int(row["draw_index"])].append(row)
        grouped_contest[f"{row['year']}-{row['part']}"] .append(row)
    summary = {
        "artifact_schema_version": "qwen3_aime_audit_v1",
        "completed_at": now_iso(),
        "contract_sha256": contract,
        "corpus_sha256": policy["corpus"]["sha256"],
        "decision_counts": dict(
            sorted(Counter(row["scorer_trace"]["official_decision_reason"] for row in rows).items())
        ),
        "model": policy["model"],
        "overall": metric_block(rows),
        "per_contest": {
            key: metric_block(value) for key, value in sorted(grouped_contest.items(), reverse=True)
        },
        "per_draw": {
            str(key): metric_block(value) for key, value in sorted(grouped_draw.items())
        },
        "per_year": {
            str(key): metric_block(value) for key, value in sorted(grouped_year.items(), reverse=True)
        },
        "policy_sha256": sha256_file(POLICY_PATH),
        "row_count": len(rows),
        "scorer": policy["scorer"],
        "status": "complete",
    }
    atomic_text(root / "artifact_manifest.jsonl", "".join(canonical_json(item) + "\n" for item in artifacts))
    atomic_text(root / "summary.json", pretty_json(summary))
    overall = summary["overall"]
    markdown = (
        "# Qwen3-8B AIME 2009–2024 16x results\n\n"
        f"- Rows: {overall['row_count']:,}\n"
        f"- Accuracy: {100 * overall['accuracy']:.2f}% "
        f"± {100 * overall['accuracy_standard_error']:.2f}% SE\n"
        f"- Scorable last-box rate: {100 * overall['scorable_rate']:.2f}%\n"
        f"- Complete-box rate: {100 * overall['complete_box_rate']:.2f}%\n"
        f"- 32K cap-hit rate: {100 * overall['cap_hit_rate']:.2f}%\n"
        f"- Mean output tokens: {overall['mean_output_tokens']:.1f}\n"
    )
    atomic_text(root / "summary.md", markdown)
    print(
        f"QWEN3_AIME_AUDIT_COMPLETE rows={len(rows)} "
        f"accuracy={overall['accuracy']:.6f} path={root / 'summary.json'}",
        flush=True,
    )


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args], text=True).strip()


def record_submission(args: argparse.Namespace) -> None:
    policy = load_policy()
    metadata = {
        "contract_sha256": contract_sha256(policy),
        "dependency_chain": {
            "preflight": args.preflight_job,
            "canary_afterok_preflight": args.canary_job,
            "generation_array_afterok_canary": args.generation_job,
            "audit_afterok_generation": args.audit_job,
            "huggingface_publish_afterok_audit": args.publish_job,
        },
        "expected_completions": EXPECTED_ROWS,
        "expected_h200_tasks": 65,
        "generation_array": "0-63",
        "git_branch": git_value("branch", "--show-current"),
        "scripts_git_commit": git_value("rev-parse", "HEAD"),
        "submitted_at": now_iso(),
        "target_huggingface_dataset": policy["huggingface_publication"]["repo_id"],
    }
    atomic_text(Path(args.output), pretty_json(metadata))
    print(f"QWEN3_AIME_SUBMISSION_RECORDED path={args.output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify")
    subparsers.add_parser("preflight")
    subparsers.add_parser("audit")
    path_parser = subparsers.add_parser("path")
    path_parser.add_argument(
        "--field",
        required=True,
        choices=("contract_sha256", "log_root", "model_root", "output_root"),
    )
    record_parser = subparsers.add_parser("record-submission")
    record_parser.add_argument("--preflight-job", required=True)
    record_parser.add_argument("--canary-job", required=True)
    record_parser.add_argument("--generation-job", required=True)
    record_parser.add_argument("--audit-job", required=True)
    record_parser.add_argument("--publish-job", required=True)
    record_parser.add_argument("--output", default=str(SUBMISSION_PATH))
    args = parser.parse_args()

    if args.command == "verify":
        print(pretty_json(verify_contract()), end="")
    elif args.command == "preflight":
        preflight()
    elif args.command == "audit":
        audit()
    elif args.command == "record-submission":
        record_submission(args)
    else:
        policy = load_policy()
        values = {
            "contract_sha256": contract_sha256(policy),
            "log_root": log_root(policy),
            "model_root": model_root(policy),
            "output_root": output_root(policy),
        }
        print(values[args.field])


if __name__ == "__main__":
    main()
