#!/usr/bin/env python3
"""Validate, score, aggregate, and record the AIME split-control evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from examples.qwen2_5_7b_aime_generalization_incorrect_sft.eval.tokenizer_compat import prepare_overlay
from examples.qwen2_5_7b_aime_generalization_incorrect_sft.hf_checkpoint_gate import validate_checkpoint

CONTRACT_ID = "0b22cc069c941ec5"
EVAL_NAME = "aime_split_control_sampled_v1"
SUBMISSION_METADATA_RELATIVE = (
    "examples/qwen2_5_7b_aime_2009_2024_split_sft_6k/eval/submission_metadata.json"
)
BASE_TARGET = "base_qwen2_5_7b"
TARGETS = (
    BASE_TARGET,
    "held_in_correct_6000",
    "held_in_incorrect_6000",
    "held_out_correct_6000",
    "held_out_incorrect_6000",
)
TRAINED_TARGETS = TARGETS[1:]
SPLITS = ("held_in", "held_out")
DRAWS = 4
PROMPTS_PER_SPLIT = 106
PROMPTS = PROMPTS_PER_SPLIT * len(SPLITS)
SEED_BASE = 1234
SEED_SPAN = 480
FINAL_ITERATION = 187
EVAL_CONTRACT_SHA256 = "e7b4d14ab32512d5d23a0820e032fd5418a8c2d9cbd391af0e3016acab0e01f7"
SCORER_SHA256 = "c6b10ee5a5667a21df095f9ef87a04c2499cf8dbe56cff11f3bf9836d8262256"
SPLIT_SHA256 = {
    "held_in": "44f0ed73482564e0f59343c913fad6b6e526bef5287f6216a495cba00b013333",
    "held_out": "4e1b724926f7a50b4ee6e05710c18dd750e9d196dd4596e2aad2c36521e6fefa",
}
STOP_TOKENS = {"<|endoftext|>": 151643, "<|im_end|>": 151645}
DECODING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "n": 1}
# Which split each trained target saw during SFT; the base saw neither.
TRAINED_ON = {
    "held_in_correct_6000": "held_in",
    "held_in_incorrect_6000": "held_in",
    "held_out_correct_6000": "held_out",
    "held_out_incorrect_6000": "held_out",
}
SUPERVISION = {
    "held_in_correct_6000": "correct",
    "held_in_incorrect_6000": "incorrect",
    "held_out_correct_6000": "correct",
    "held_out_incorrect_6000": "incorrect",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


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
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise RuntimeError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(value)
    return rows


def sample_seed(draw_index: int, eval_index: int) -> int:
    if draw_index not in range(DRAWS):
        raise ValueError(f"draw_index must be 0..{DRAWS - 1}: {draw_index}")
    if eval_index not in range(SEED_SPAN):
        raise ValueError(f"eval_index must be 0..{SEED_SPAN - 1}: {eval_index}")
    return SEED_BASE + SEED_SPAN * draw_index + eval_index


def experiment_root(args: argparse.Namespace) -> Path:
    return args.experiment_root.resolve()


def eval_root(args: argparse.Namespace) -> Path:
    return experiment_root(args) / "outputs" / "eval" / EVAL_NAME


def result_root(args: argparse.Namespace) -> Path:
    # Scratch, not the package: the runtime repo gate requires a clean worktree
    # (including untracked files) on every delayed job, so scoring must not write
    # into examples/. Publishing into the package is a separate, deliberate step.
    return experiment_root(args) / "results" / EVAL_NAME


def control_root(args: argparse.Namespace) -> Path:
    return eval_root(args) / "_control"


def submission_metadata_path(args: argparse.Namespace) -> Path:
    return args.repo_root.resolve() / SUBMISSION_METADATA_RELATIVE


def eval_file(args: argparse.Namespace, split: str) -> Path:
    return experiment_root(args) / "data" / "eval" / f"{split}_{PROMPTS_PER_SPLIT}.jsonl"


def comparison(args: argparse.Namespace) -> list[dict[str, Any]]:
    inventory = load_json(experiment_root(args) / "handoff" / "comparison_sources.json")
    targets = inventory.get("targets")
    if (
        inventory.get("contract_hash") != CONTRACT_ID
        or inventory.get("comparison_checkpoint_count") != len(TARGETS)
        or not isinstance(targets, list)
        or tuple(row.get("id") for row in targets) != TARGETS
    ):
        raise RuntimeError("comparison inventory does not match the five-target contract")
    return targets


def safe_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"unsafe relative path: {value}")
    return path


def target_paths(args: argparse.Namespace, target: str) -> tuple[Path, Path, str]:
    row = next(item for item in comparison(args) if item["id"] == target)
    root = args.repo_root.resolve() / safe_relative(str(row["remote_experiment_relative"]))
    model = root / safe_relative(str(row["remote_checkpoint_relative"]))
    manifest = root / safe_relative(str(row["remote_manifest_relative"]))
    return model, manifest, str(row["checkpoint_identity"])


def normalized_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Load both splits, verifying pinned hashes and per-row prompt integrity."""
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        path = eval_file(args, split)
        if sha256_file(path) != SPLIT_SHA256[split]:
            raise RuntimeError(f"evaluation split hash changed: {path}")
        split_rows = read_jsonl(path)
        if len(split_rows) != PROMPTS_PER_SPLIT:
            raise RuntimeError(f"expected {PROMPTS_PER_SPLIT} rows in {path}, found {len(split_rows)}")
        for row in split_rows:
            rendered = row.get("rendered_prompt")
            accepted = row.get("accepted_answer_integers")
            if not isinstance(rendered, str) or sha256_text(rendered) != row.get("rendered_prompt_sha256"):
                raise RuntimeError(f"rendered prompt hash mismatch: {row.get('problem_id')}")
            if row.get("split") != split:
                raise RuntimeError(f"row split mismatch: {row.get('problem_id')}")
            if (
                not isinstance(accepted, list)
                or not accepted
                or any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 999 for v in accepted)
            ):
                raise RuntimeError(f"invalid accepted answers: {row.get('problem_id')}")
        rows.extend(split_rows)
    indices = [int(row["eval_index"]) for row in rows]
    if len(set(indices)) != PROMPTS or min(indices) < 0 or max(indices) >= SEED_SPAN:
        raise RuntimeError("eval indices are not unique across the two splits")
    rows.sort(key=lambda row: int(row["eval_index"]))
    return rows


def validate_contract(args: argparse.Namespace) -> dict[str, Any]:
    path = experiment_root(args) / "handoff" / "eval_contract.json"
    if sha256_file(path) != EVAL_CONTRACT_SHA256:
        raise RuntimeError(f"evaluation contract hash changed: {path}")
    contract = load_json(path)
    sources = {str(item["split"]): item for item in contract.get("evaluation_sources", [])}
    if (
        tuple(contract.get("checkpoints", ())) != TARGETS
        or contract.get("cap_hits_scored") is not True
        or contract.get("scorer", {}).get("sha256") != SCORER_SHA256
        or contract.get("scorer", {}).get("version") != "aime_last_boxed_integer_scorer_v1"
        or contract.get("prompt", {}).get("uses_qwen2_5_default_chat_template") is not False
        or contract.get("prompt", {}).get("rendered_prompt_field_used_directly") is not True
        or contract.get("prompt", {}).get("system") is not None
        or contract.get("split_counts") != {split: PROMPTS_PER_SPLIT for split in SPLITS}
        or set(sources) != set(SPLITS)
        or any(sources[split].get("sha256") != SPLIT_SHA256[split] for split in SPLITS)
        or any(sources[split].get("rows") != PROMPTS_PER_SPLIT for split in SPLITS)
    ):
        raise RuntimeError("evaluation contract content changed")
    return contract


def validate_checkpoint_files(model: Path, manifest_path: Path, identity: str, full: bool) -> dict[str, Any]:
    if not model.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError(f"missing checkpoint or manifest: {model}")
    validate_checkpoint(model)
    manifest = load_json(manifest_path)
    if manifest.get("checkpoint_manifest_sha256") != identity:
        raise RuntimeError(f"checkpoint identity mismatch: {model}")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError(f"empty checkpoint manifest: {manifest_path}")
    expected = {str(item["name"]): (int(item["bytes"]), str(item["sha256"])) for item in files}
    actual = {path.relative_to(model).as_posix(): path.stat().st_size for path in model.rglob("*") if path.is_file()}
    if actual != {name: size_hash[0] for name, size_hash in expected.items()}:
        raise RuntimeError(f"checkpoint size inventory mismatch: {model}")
    if full:
        for name, (_, expected_hash) in expected.items():
            if sha256_file(model / name) != expected_hash:
                raise RuntimeError(f"checkpoint file hash mismatch: {model / name}")
    return {"files": len(expected), "identity": identity, "model": str(model), "full_hashes": full}


def scorer_function(args: argparse.Namespace) -> Callable[..., dict[str, Any]]:
    path = args.repo_root.resolve() / "examples/qwen2_5_7b_aime_generalization_incorrect_sft/eval/aime_scorer.py"
    if sha256_file(path) != SCORER_SHA256:
        raise RuntimeError("refusing scorer with unexpected hash")
    spec = importlib.util.spec_from_file_location("contracted_aime_scorer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import scorer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if module.SCORER_VERSION != "aime_last_boxed_integer_scorer_v1":
        raise RuntimeError("scorer version changed")
    return module.score_response


def score_accepted(
    score_response: Callable[..., dict[str, Any]],
    record: dict[str, Any],
    accepted: list[int],
) -> dict[str, Any]:
    """Run the unchanged scorer against every officially accepted integer.

    A response counts as correct when it matches any of them. The returned trace
    is the matching one, or the first-listed answer's trace when none match, so
    the recorded parse diagnostics stay well defined either way.
    """
    traces = [
        score_response(
            str(record["response"]),
            int(answer),
            finish_reason=str(record.get("finish_reason") or ""),
            stop_token_id=record.get("stop_reason") if isinstance(record.get("stop_reason"), int) else None,
            cap_hit=bool(record["cap_hit"]),
            problem_id=str(record["problem_id"]),
        )
        for answer in accepted
    ]
    for trace in traces:
        if trace["is_correct"]:
            return trace
    return traces[0]


def preflight(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    validate_contract(args)
    rows = normalized_rows(args)
    sources = load_json(experiment_root(args) / "handoff" / "checkpoint_sources.json")
    if (
        sources.get("contract_hash") != CONTRACT_ID
        or sources.get("checkpoint_count") != len(TRAINED_TARGETS)
        or tuple(str(row.get("id")) for row in sources.get("checkpoints", [])) != TRAINED_TARGETS
    ):
        raise RuntimeError("checkpoint transfer inventory changed")
    transferred = {row["id"]: row for row in sources["checkpoints"]}
    for row in comparison(args)[1:]:
        source = transferred[row["id"]]
        for field in (
            "checkpoint_identity",
            "remote_checkpoint_relative",
            "remote_manifest_relative",
            "source_checkpoint",
            "source_manifest",
        ):
            if source.get(field) != row.get(field):
                raise RuntimeError(f"checkpoint/comparison inventory mismatch: {row['id']}/{field}")
    status = load_json(experiment_root(args) / "TRAINING_STATUS.json")
    if (
        status.get("contract_hash") != CONTRACT_ID
        or status.get("trained_checkpoints") != len(TRAINED_TARGETS)
        or status.get("full_parameter_sft") is not True
        or status.get("final_iterations") != {target: FINAL_ITERATION for target in TRAINED_TARGETS}
    ):
        raise RuntimeError("training status is incomplete")

    checkpoint_status: dict[str, Any] = {}
    prompt_id_reference: dict[int, list[int]] = {}
    overlays = control_root(args) / "tokenizer_overlays"
    for target in TARGETS:
        print(f"SPLIT_AIME_PREFLIGHT_TARGET_START target={target}", flush=True)
        model, manifest, identity = target_paths(args, target)
        if target in TRAINED_TARGETS and status.get("checkpoint_identities", {}).get(target) != identity:
            raise RuntimeError(f"training/comparison identity mismatch: {target}")
        checkpoint_status[target] = validate_checkpoint_files(model, manifest, identity, args.full)
        overlay_path, overlay_metadata = prepare_overlay(overlays, target, model, identity)
        tokenizer = AutoTokenizer.from_pretrained(overlay_path, trust_remote_code=True, use_fast=True)
        actual_stops = {token: int(tokenizer.convert_tokens_to_ids(token)) for token in STOP_TOKENS}
        if actual_stops != STOP_TOKENS or int(tokenizer.eos_token_id) != STOP_TOKENS["<|endoftext|>"]:
            raise RuntimeError(f"stop-token mismatch: {target}")
        lengths: list[int] = []
        for row in rows:
            rendered = str(row["rendered_prompt"])
            # The contract forbids the default chat template: the rendered prompt
            # carries exactly one user turn and no system turn.
            if not rendered.startswith("<|im_start|>user\n") or not rendered.endswith(
                "<|im_end|>\n<|im_start|>assistant\n"
            ):
                raise RuntimeError(f"rendered prompt shape changed: {row['problem_id']}")
            if "<|im_start|>system" in rendered:
                raise RuntimeError(f"rendered prompt contains a system turn: {row['problem_id']}")
            ids = [int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)]
            if not ids or len(ids) >= 32768:
                raise RuntimeError(f"invalid rendered prompt length: {target}/{row['problem_id']}")
            index = int(row["eval_index"])
            if target == BASE_TARGET:
                prompt_id_reference[index] = ids
            elif ids != prompt_id_reference[index]:
                raise RuntimeError(f"tokenizer overlay changed prompt IDs: {target}/{row['problem_id']}")
            lengths.append(len(ids))
        checkpoint_status[target]["overlay_identity"] = overlay_metadata["overlay_identity"]
        checkpoint_status[target]["prompt_tokens"] = {"min": min(lengths), "max": max(lengths)}
        print(
            f"SPLIT_AIME_PREFLIGHT_TARGET_OK target={target} prompt_tokens={min(lengths)}-{max(lengths)}",
            flush=True,
        )

    artifact = {
        "artifact_schema_version": 1,
        "checkpoints": checkpoint_status,
        "contract_id": CONTRACT_ID,
        "decoding": DECODING,
        "draws": DRAWS,
        "eval_contract_sha256": EVAL_CONTRACT_SHA256,
        "generation_count": len(TARGETS) * PROMPTS * DRAWS,
        "prompt_count": len(rows),
        "scorer_sha256": SCORER_SHA256,
        "split_sha256": SPLIT_SHA256,
        "status": "valid",
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_text(control_root(args) / "preflight.json", canonical_json(artifact))
    print(f"SPLIT_AIME_PREFLIGHT_OK targets={len(TARGETS)} prompts={len(rows)} full={args.full}")


def draw_output(args: argparse.Namespace, target: str, split: str, draw: int) -> Path:
    if target not in TARGETS or split not in SPLITS or draw not in range(DRAWS):
        raise ValueError(f"invalid draw coordinate: {target}/{split}/{draw}")
    return eval_root(args) / target / split / f"draw_{draw:02d}"


def validate_generation_policy(
    args: argparse.Namespace,
    target: str,
    split: str,
    draw: int,
    identity: str,
    overlay_identity: str,
) -> str:
    path = draw_output(args, target, split, draw) / "generation_policy.json"
    policy = load_json(path)
    if (
        policy.get("checkpoint_identity") != identity
        or policy.get("decoding") != DECODING
        or policy.get("draw_index") != draw
        or policy.get("eval_contract_sha256") != EVAL_CONTRACT_SHA256
        or policy.get("split") != split
        or policy.get("split_sha256") != SPLIT_SHA256[split]
        or policy.get("target") != target
        or policy.get("target_context") != 32768
        or policy.get("tokenizer_overlay_identity") != overlay_identity
        or policy.get("seed_formula") != "1234 + 480 * draw_index + eval_index"
        or policy.get("prompt", {}).get("uses_qwen2_5_default_chat_template") is not False
        or sorted(policy.get("stop_token_ids", ())) != sorted(STOP_TOKENS.values())
    ):
        raise RuntimeError(f"generation policy drifted: {path}")
    return sha256_text(json.dumps(policy, sort_keys=True, separators=(",", ":")))


def score_target(args: argparse.Namespace) -> None:
    score_response = scorer_function(args)
    target = args.target
    model, _, expected_identity = target_paths(args, target)
    _, overlay_metadata = prepare_overlay(control_root(args) / "tokenizer_overlays", target, model, expected_identity)
    expected_overlay_identity = str(overlay_metadata["overlay_identity"])
    canonical = {int(row["eval_index"]): row for row in normalized_rows(args)}
    scored: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for split in SPLITS:
        for draw in range(DRAWS):
            policy_sha256 = validate_generation_policy(
                args, target, split, draw, expected_identity, expected_overlay_identity
            )
            path = draw_output(args, target, split, draw) / "records.jsonl"
            rows = read_jsonl(path)
            if len(rows) != PROMPTS_PER_SPLIT:
                raise RuntimeError(f"draw is incomplete: {target}/{split}/{draw} ({len(rows)})")
            for record in rows:
                index = int(record.get("eval_index", -1))
                coordinate = (draw, index)
                source = canonical.get(index, {})
                if (
                    coordinate in seen
                    or record.get("target") != target
                    or record.get("split") != split
                    or record.get("draw_index") != draw
                    or record.get("checkpoint_identity") != expected_identity
                    or record.get("seed") != sample_seed(draw, index)
                    or record.get("policy_sha256") != policy_sha256
                    or record.get("tokenizer_overlay_identity") != expected_overlay_identity
                    or record.get("accepted_answer_integers") != source.get("accepted_answer_integers")
                    or record.get("problem_id") != source.get("problem_id")
                    or record.get("rendered_prompt_sha256") != source.get("rendered_prompt_sha256")
                    or record.get("split") != source.get("split")
                    or record.get("year") != source.get("year")
                    or sha256_text(str(record.get("response", ""))) != record.get("response_sha256")
                ):
                    raise RuntimeError(f"invalid generation coordinate: {target}/{split}/{coordinate}")
                seen.add(coordinate)
                accepted = [int(value) for value in record["accepted_answer_integers"]]
                trace = score_accepted(score_response, record, accepted)
                scored.append(
                    {
                        "accepted_answer_integers": accepted,
                        "cap_hit": bool(record["cap_hit"]),
                        "draw_index": draw,
                        "eval_index": index,
                        "finish_reason": record.get("finish_reason"),
                        "generated_token_count": int(record["generated_token_count"]),
                        "is_correct": bool(trace["is_correct"]),
                        "is_scorable": bool(trace["is_scorable"]),
                        "official_candidate_raw": trace["official_candidate_raw"],
                        "official_decision_reason": trace["official_decision_reason"],
                        "official_extracted_source_span": trace["official_extracted_source_span"],
                        "parsed_integer": trace["parsed_integer"],
                        "problem_id": str(record["problem_id"]),
                        "rendered_prompt_tokens": int(record["rendered_prompt_tokens"]),
                        "response_sha256": str(record["response_sha256"]),
                        "scorer_version": trace["scorer_version"],
                        "seed": int(record["seed"]),
                        "split": split,
                        "stop_reason": record.get("stop_reason"),
                        "target": target,
                        "trace_schema_version": trace["trace_schema_version"],
                        "year": int(record["year"]),
                    }
                )
    expected = {(draw, int(index)) for draw in range(DRAWS) for index in canonical}
    if seen != expected:
        raise RuntimeError(f"target has missing generation coordinates: {target}")
    scored.sort(key=lambda row: (row["draw_index"], row["eval_index"]))
    output = result_root(args) / "targets" / target / "scores.jsonl"
    atomic_text(output, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in scored))
    summary = {name: metrics([row for row in scored if selector(row)]) for name, selector in slice_selectors().items()}
    atomic_text(output.parent / "summary.json", canonical_json({"target": target, "slices": summary}))
    print(f"SPLIT_AIME_TARGET_FINALIZED target={target} scores={len(scored)}")


def validate_scored_records(args: argparse.Namespace, target: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if target not in TARGETS or len(rows) != PROMPTS * DRAWS:
        raise RuntimeError(f"invalid scored target bundle: {target}/{len(rows)}")
    canonical = {int(row["eval_index"]): row for row in normalized_rows(args)}
    for row in rows:
        draw = row.get("draw_index")
        index = row.get("eval_index")
        if (
            isinstance(draw, bool)
            or not isinstance(draw, int)
            or isinstance(index, bool)
            or not isinstance(index, int)
        ):
            raise RuntimeError(f"invalid scored coordinate types: {target}/{draw}/{index}")
        source = canonical.get(index, {})
        if (
            row.get("target") != target
            or row.get("seed") != sample_seed(draw, index)
            or row.get("accepted_answer_integers") != source.get("accepted_answer_integers")
            or row.get("problem_id") != source.get("problem_id")
            or row.get("split") != source.get("split")
            or row.get("year") != source.get("year")
            or row.get("scorer_version") != "aime_last_boxed_integer_scorer_v1"
            or row.get("trace_schema_version") != "aime_last_boxed_integer_trace_v1"
            or not isinstance(row.get("cap_hit"), bool)
            or not isinstance(row.get("is_correct"), bool)
            or not isinstance(row.get("is_scorable"), bool)
            or (row.get("is_correct") is True and row.get("is_scorable") is not True)
            or isinstance(row.get("generated_token_count"), bool)
            or not isinstance(row.get("generated_token_count"), int)
            or int(row.get("generated_token_count", -1)) < 0
            or isinstance(row.get("rendered_prompt_tokens"), bool)
            or not isinstance(row.get("rendered_prompt_tokens"), int)
            or int(row.get("rendered_prompt_tokens", -1)) <= 0
            or len(str(row.get("response_sha256", ""))) != 64
        ):
            raise RuntimeError(f"invalid scored record: {target}/{draw}/{index}")
    coordinates = [(row["draw_index"], row["eval_index"]) for row in rows]
    expected = sorted((draw, index) for draw in range(DRAWS) for index in canonical)
    if coordinates != sorted(coordinates) or sorted(coordinates) != expected:
        raise RuntimeError(f"scored coordinates are not canonical: {target}")
    return rows


def slice_selectors() -> dict[str, Callable[[dict[str, Any]], bool]]:
    return {
        "held_in": lambda row: row["split"] == "held_in",
        "held_out": lambda row: row["split"] == "held_out",
        "all_212": lambda row: True,
    }


def sample_se(values: list[float]) -> float:
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty percentile")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def draw_se(rows: list[dict[str, Any]]) -> float:
    """Standard error across the per-draw accuracies of a slice."""
    by_draw: dict[int, list[float]] = {}
    for row in rows:
        by_draw.setdefault(int(row["draw_index"]), []).append(float(bool(row["is_correct"])))
    accuracies = [statistics.fmean(values) for _, values in sorted(by_draw.items())]
    return sample_se(accuracies)


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("cannot aggregate an empty slice")
    correct = [float(bool(row["is_correct"])) for row in rows]
    lengths = [int(row["generated_token_count"]) for row in rows]
    return {
        "accuracy": statistics.fmean(correct),
        "accuracy_draw_se": draw_se(rows),
        "accuracy_mc_se": sample_se(correct),
        "cap_hit_rate": statistics.fmean(float(bool(row["cap_hit"])) for row in rows),
        "completion_tokens": {
            "max": max(lengths),
            "mean": statistics.fmean(lengths),
            "p50": percentile(lengths, 0.50),
            "p90": percentile(lengths, 0.90),
            "p95": percentile(lengths, 0.95),
            "p99": percentile(lengths, 0.99),
            "std": statistics.stdev(lengths) if len(lengths) > 1 else 0.0,
        },
        "correct": sum(bool(row["is_correct"]) for row in rows),
        "draws": len({int(row["draw_index"]) for row in rows}),
        "n": len(rows),
        "valid_box_rate": statistics.fmean(float(bool(row["is_scorable"])) for row in rows),
    }


def paired_delta(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    left_map = {(row["draw_index"], row["eval_index"]): row for row in left}
    right_map = {(row["draw_index"], row["eval_index"]): row for row in right}
    if left_map.keys() != right_map.keys():
        raise RuntimeError("paired delta coordinates differ")
    values = [
        float(bool(right_map[key]["is_correct"])) - float(bool(left_map[key]["is_correct"]))
        for key in sorted(left_map)
    ]
    return {"delta": statistics.fmean(values), "delta_mc_se": sample_se(values), "n": len(values)}


def primary_table(records: dict[str, list[dict[str, Any]]], trained_split: str) -> list[dict[str, Any]]:
    """Rows for the pair trained on ``trained_split``: in-distribution, then unseen."""
    unseen = "held_out" if trained_split == "held_in" else "held_in"
    correct_target = f"{trained_split}_correct_6000"
    incorrect_target = f"{trained_split}_incorrect_6000"
    selectors = slice_selectors()
    rows = [
        (f"{trained_split} (trained distribution)", trained_split),
        (f"{unseen} (unseen problems)", unseen),
        ("all 212", "all_212"),
    ]
    result: list[dict[str, Any]] = []
    for label, slice_name in rows:
        selected = {
            target: [row for row in values if selectors[slice_name](row)] for target, values in records.items()
        }
        result.append(
            {
                "base": metrics(selected[BASE_TARGET]),
                "correct": metrics(selected[correct_target]),
                "correct_minus_base": paired_delta(selected[BASE_TARGET], selected[correct_target]),
                "incorrect": metrics(selected[incorrect_target]),
                "incorrect_minus_base": paired_delta(selected[BASE_TARGET], selected[incorrect_target]),
                "incorrect_minus_correct": paired_delta(selected[correct_target], selected[incorrect_target]),
                "label": label,
                "slice": slice_name,
            }
        )
    return result


def fmt_rate(value: float, se: float | None = None) -> str:
    base = f"{100 * value:.2f}%"
    return base if se is None else f"{base} ± {100 * se:.2f}%"


def fmt_delta(value: dict[str, Any]) -> str:
    return f"{100 * value['delta']:+.2f}% ± {100 * value['delta_mc_se']:.2f}%"


def markdown_table(title: str, rows: list[dict[str, Any]]) -> str:
    lines = [
        f"## {title}",
        "",
        "| Slice | Base | Correct-trained | Incorrect-trained | Correct − base "
        "| Incorrect − base | Incorrect − correct |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {base} | {correct} | {incorrect} | {cb} | {ib} | {ic} |".format(
                label=row["label"],
                base=fmt_rate(row["base"]["accuracy"], row["base"]["accuracy_draw_se"]),
                correct=fmt_rate(row["correct"]["accuracy"], row["correct"]["accuracy_draw_se"]),
                incorrect=fmt_rate(row["incorrect"]["accuracy"], row["incorrect"]["accuracy_draw_se"]),
                cb=fmt_delta(row["correct_minus_base"]),
                ib=fmt_delta(row["incorrect_minus_base"]),
                ic=fmt_delta(row["incorrect_minus_correct"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def markdown_diagnostics(diagnostics: dict[str, dict[str, Any]]) -> str:
    lines = [
        "## All-212 diagnostics",
        "",
        "Valid box means the contracted scorer parsed a complete boxed AIME integer.",
        "",
        "| Target | Accuracy | Valid box | Cap hit | Mean tokens | Std | P50 | P90 | P95 | P99 | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target in TARGETS:
        value = diagnostics[target]
        tokens = value["completion_tokens"]
        lines.append(
            "| {target} | {accuracy} | {box} | {cap} | {mean:.1f} | {std:.1f} | "
            "{p50:.1f} | {p90:.1f} | {p95:.1f} | {p99:.1f} | {maximum} |".format(
                target=target,
                accuracy=fmt_rate(value["accuracy"], value["accuracy_draw_se"]),
                box=fmt_rate(value["valid_box_rate"]),
                cap=fmt_rate(value["cap_hit_rate"]),
                mean=tokens["mean"],
                std=tokens["std"],
                p50=tokens["p50"],
                p90=tokens["p90"],
                p95=tokens["p95"],
                p99=tokens["p99"],
                maximum=tokens["max"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def audit(args: argparse.Namespace) -> None:
    validate_contract(args)
    records: dict[str, list[dict[str, Any]]] = {}
    for target in TARGETS:
        path = result_root(args) / "targets" / target / "scores.jsonl"
        records[target] = validate_scored_records(args, target, read_jsonl(path))
    selectors = slice_selectors()
    diagnostics = {
        target: metrics([row for row in rows if selectors["all_212"](row)]) for target, rows in records.items()
    }
    tables = {split: primary_table(records, split) for split in SPLITS}
    payload = {
        "artifact_schema_version": 1,
        "contract_id": CONTRACT_ID,
        "decoding": DECODING,
        "diagnostics": diagnostics,
        "draws": DRAWS,
        "eval_contract_sha256": EVAL_CONTRACT_SHA256,
        "eval_name": EVAL_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prompts": PROMPTS,
        "scorer_sha256": SCORER_SHA256,
        "slices": {
            target: {name: metrics([row for row in rows if selector(row)]) for name, selector in selectors.items()}
            for target, rows in records.items()
        },
        "supervision": SUPERVISION,
        "tables": tables,
        "trained_on": TRAINED_ON,
    }
    atomic_text(result_root(args) / "RESULTS.json", canonical_json(payload))

    document = [
        f"# Qwen2.5-7B AIME split-control evaluation ({EVAL_NAME})",
        "",
        f"Four sampled generations per problem at temperature {DECODING['temperature']}, "
        f"top-p {DECODING['top_p']}, top-k {DECODING['top_k']}, min-p {DECODING['min_p']}, "
        "reused verbatim from the Qwen3-8B 16x source-generation run.",
        "",
        "These numbers sit on the same decoding distribution as the traces the checkpoints were",
        "trained on. Earlier Qwen2.5 AIME bundles in this repository used temperature 0.7 /",
        "top-p 0.8 / top-k -1 and are **not** directly comparable to this table.",
        "",
        "Every completion is scored, including cap hits. Accuracy is the mean over all",
        f"{PROMPTS * DRAWS} completions per target; ± is the standard error across the {DRAWS} draws.",
        "",
    ]
    for split in SPLITS:
        document.append(markdown_table(f"Trained on {split}", tables[split]))
    document.append(markdown_diagnostics(diagnostics))
    atomic_text(result_root(args) / "RESULTS.md", "\n".join(document))
    print(f"SPLIT_AIME_AUDIT_OK targets={len(TARGETS)} rows={PROMPTS * DRAWS}")


def path_command(args: argparse.Namespace) -> None:
    if args.field == "control":
        print(control_root(args))
        return
    if args.field == "eval-root":
        print(eval_root(args))
        return
    if args.field == "result-root":
        print(result_root(args))
        return
    if args.field == "eval-file":
        if args.split is None:
            raise SystemExit("--split is required for eval-file")
        print(eval_file(args, args.split))
        return
    if args.target is None:
        raise SystemExit(f"--target is required for {args.field}")
    model, manifest, identity = target_paths(args, args.target)
    if args.field == "model":
        print(model)
    elif args.field == "manifest":
        print(manifest)
    elif args.field == "identity":
        print(identity)
    elif args.field == "tokenizer":
        overlay_path, _ = prepare_overlay(control_root(args) / "tokenizer_overlays", args.target, model, identity)
        print(overlay_path)
    elif args.field == "draw-output":
        if args.split is None or args.draw is None:
            raise SystemExit("--split and --draw are required for draw-output")
        print(draw_output(args, args.target, args.split, args.draw))


def parse_assignments(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        target, _, job = value.partition("=")
        if target not in TARGETS or not job.isdigit() or target in result:
            raise RuntimeError(f"invalid job assignment: {value}")
        result[target] = job
    if set(result) != set(TARGETS):
        raise RuntimeError("job assignments do not cover every target")
    return result


def record_submission(args: argparse.Namespace) -> None:
    arrays = parse_assignments(args.array_job)
    finalizers = parse_assignments(args.finalizer_job)
    edges: list[dict[str, Any]] = [{"afterok": [args.preflight_job], "job": args.canary_job, "stage": "canary"}]
    for target in TARGETS:
        edges.append(
            {"afterok": [args.canary_job], "job": arrays[target], "stage": "generation", "target": target}
        )
        edges.append(
            {"afterok": [arrays[target]], "job": finalizers[target], "stage": "finalize", "target": target}
        )
    edges.append(
        {"afterok": [finalizers[target] for target in TARGETS], "job": args.audit_job, "stage": "audit"}
    )
    metadata = {
        "array_jobs": arrays,
        "artifact_schema_version": 1,
        "audit_job": args.audit_job,
        "canary_job": args.canary_job,
        "contract_id": CONTRACT_ID,
        "decoding": DECODING,
        "dependency_edges": edges,
        "draws": DRAWS,
        "eval_contract_sha256": EVAL_CONTRACT_SHA256,
        "eval_name": EVAL_NAME,
        "finalizer_jobs": finalizers,
        "generation_count": len(TARGETS) * PROMPTS * DRAWS,
        "git_commit": args.git_commit,
        "log_root": str(control_root(args) / "slurm_logs"),
        "output_root": str(eval_root(args)),
        "preflight_job": args.preflight_job,
        "prompts": PROMPTS,
        "scorer_sha256": SCORER_SHA256,
        "splits": list(SPLITS),
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "targets": list(TARGETS),
        "tasks_per_target": DRAWS * len(SPLITS),
    }
    journal = control_root(args) / "submission.json"
    if journal.exists():
        raise RuntimeError(f"submission journal already exists: {journal}")
    atomic_text(journal, canonical_json(metadata))
    # The committed copy is the one delayed jobs allow as a clean descendant path.
    atomic_text(submission_metadata_path(args), canonical_json(metadata))
    print(journal)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo-root", type=Path, required=True)
    result.add_argument("--experiment-root", type=Path, required=True)
    commands = result.add_subparsers(dest="command", required=True)
    preflight_parser = commands.add_parser("preflight")
    preflight_parser.add_argument("--full", action="store_true")
    preflight_parser.set_defaults(func=preflight)
    path_parser = commands.add_parser("path")
    path_parser.add_argument(
        "--field",
        required=True,
        choices=(
            "control",
            "eval-root",
            "result-root",
            "eval-file",
            "model",
            "manifest",
            "identity",
            "tokenizer",
            "draw-output",
        ),
    )
    path_parser.add_argument("--target", choices=TARGETS)
    path_parser.add_argument("--split", choices=SPLITS)
    path_parser.add_argument("--draw", type=int)
    path_parser.set_defaults(func=path_command)
    finalize_parser = commands.add_parser("finalize-target")
    finalize_parser.add_argument("--target", required=True, choices=TARGETS)
    finalize_parser.set_defaults(func=score_target)
    audit_parser = commands.add_parser("audit")
    audit_parser.set_defaults(func=audit)
    record = commands.add_parser("record-submission")
    record.add_argument("--preflight-job", required=True)
    record.add_argument("--canary-job", required=True)
    record.add_argument("--array-job", action="append", required=True)
    record.add_argument("--finalizer-job", action="append", required=True)
    record.add_argument("--audit-job", required=True)
    record.add_argument("--git-commit", required=True)
    record.set_defaults(func=record_submission)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.repo_root.resolve() not in experiment_root(args).parents:
        raise RuntimeError("experiment root must be inside the repository")
    args.func(args)


if __name__ == "__main__":
    main()
