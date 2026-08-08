#!/usr/bin/env python3
"""Select two correct-only trace sets and materialize exact-token datasets."""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Any

from examples.qwen3_8b_openr1_math220k_masked_sft_40k.data_utils import (
    correctness_at,
    problem_id,
    trace_structure_ok,
    usable_text,
    value_at,
)
from examples.qwen_correct_only_openr1_math220k_sft_2k.data_utils import (
    atomic_json,
    atomic_jsonl,
    build_training_record,
    message_record,
    rendered_lengths,
    sha256_file,
)
from examples.qwen_correct_only_openr1_math220k_sft_2k.experiment import (
    MODEL_ORDER,
    RUN_ORDER,
    SHARED_8K_MODELS,
    load_contract,
    ordered_trace_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def deterministic_order(indices: list[int], seed: int, row_index: int, policy: str) -> list[int]:
    result = list(indices)
    random.Random(f"{seed}:{row_index}:{policy}").shuffle(result)
    return result


def eligible_ref(
    row: dict[str, Any],
    row_index: int,
    policy: str,
    model_keys: tuple[str, ...],
    tokenizers: dict[str, Any],
    contract: dict[str, Any],
    trace_cap: int,
    sequence_cap: int,
    seed: int,
    counts: Counter[str],
) -> dict[str, Any] | None:
    problem = usable_text(row.get("problem"))
    answer = usable_text(row.get("answer"))
    generations = row.get("generations")
    if problem is None or answer is None or not isinstance(generations, (list, tuple)):
        counts["row_missing_required_fields"] += 1
        return None
    candidates: list[tuple[int, str]] = []
    for generation_index in range(len(generations)):
        correct, source = correctness_at(row, generation_index)
        if correct is not True:
            continue
        if value_at(row.get("is_reasoning_complete"), generation_index) is not True:
            counts["trace_incomplete_source_flag"] += 1
            continue
        trace = usable_text(generations[generation_index])
        if trace is None:
            counts["trace_empty"] += 1
            continue
        structure_ok, reason = trace_structure_ok(trace)
        if not structure_ok:
            counts[f"trace_rejected_{reason}"] += 1
            continue
        candidates.append((generation_index, str(source)))
    by_index = dict(candidates)
    for generation_index in deterministic_order(
        list(by_index), seed, row_index, policy
    ):
        trace = str(generations[generation_index]).strip()
        per_model: dict[str, dict[str, int]] = {}
        fits = True
        for model_key in model_keys:
            model = contract["models"][model_key]
            trace_tokens, total_tokens = rendered_lengths(
                tokenizers[model_key],
                problem,
                trace,
                model["apply_chat_template_kwargs"],
            )
            per_model[model_key] = {
                "assistant_tokens": trace_tokens,
                "total_tokens": total_tokens,
            }
            if trace_tokens > trace_cap or total_tokens > sequence_cap:
                counts[f"trace_too_long_{model_key}"] += 1
                fits = False
                break
        if fits:
            pid = problem_id(row, row_index)
            return {
                "correctness_source": by_index[generation_index],
                "generation_index": generation_index,
                "per_model_lengths": per_model,
                "problem_id": pid,
                "source_row_index": row_index,
                "trace_id": f"{pid}__row_{row_index:07d}__gen_{generation_index:03d}",
            }
    counts["problem_no_eligible_correct_trace"] += 1
    return None


def reservoir_select(
    dataset: Any,
    target: int,
    policy: str,
    model_keys: tuple[str, ...],
    tokenizers: dict[str, Any],
    contract: dict[str, Any],
    trace_cap: int,
    sequence_cap: int,
    seed: int,
) -> tuple[list[dict[str, Any]], Counter[str], int]:
    rng = random.Random(f"{seed}:{policy}:reservoir")
    reservoir: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    eligible_seen = 0
    seen_problem_ids: set[str] = set()
    for row_index, row in enumerate(dataset):
        counts["source_rows_seen"] += 1
        pid = problem_id(row, row_index)
        if pid in seen_problem_ids:
            counts["duplicate_problem_id"] += 1
            continue
        seen_problem_ids.add(pid)
        ref = eligible_ref(
            row,
            row_index,
            policy,
            model_keys,
            tokenizers,
            contract,
            trace_cap,
            sequence_cap,
            seed,
            counts,
        )
        if ref is None:
            continue
        eligible_seen += 1
        if len(reservoir) < target:
            reservoir.append(ref)
        else:
            slot = rng.randrange(eligible_seen)
            if slot < target:
                reservoir[slot] = ref
        if counts["source_rows_seen"] % 5000 == 0:
            print(
                f"SELECTION_PROGRESS policy={policy} rows={counts['source_rows_seen']} "
                f"eligible={eligible_seen} reservoir={len(reservoir)}",
                flush=True,
            )
    if len(reservoir) != target:
        raise RuntimeError(f"wanted {target} rows for {policy}, found {len(reservoir)}")
    random.Random(f"{seed}:{policy}:final-order").shuffle(reservoir)
    return reservoir, counts, eligible_seen


def materialize(dataset: Any, refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ref in refs:
        source = dataset[int(ref["source_row_index"])]
        trace = str(source["generations"][int(ref["generation_index"])]).strip()
        row = dict(ref)
        row.update(
            {
                "answer": str(source["answer"]),
                "assistant_trace": trace,
                "is_correct": True,
                "problem": str(source["problem"]),
                "source": source.get("source"),
                "source_is_reasoning_complete": True,
            }
        )
        rows.append(row)
    if len({row["problem_id"] for row in rows}) != len(rows):
        raise RuntimeError("selected problem IDs are not unique")
    if len({row["trace_id"] for row in rows}) != len(rows):
        raise RuntimeError("selected trace IDs are not unique")
    return rows


def load_tokenizers(model_root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    result = {}
    for model_key in MODEL_ORDER:
        model_dir = model_root / contract["models"][model_key]["hf_dir_name"]
        result[model_key] = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True, local_files_only=True
        )
    return result


def main() -> None:
    args = parse_args()
    contract = load_contract()
    stats_path = args.data_root / "selection_and_tokenization_stats.json"
    outputs = [
        stats_path,
        args.data_root / "selected" / "8k_shared.jsonl",
        args.data_root / "selected" / "16k_qwen2_5_3b.jsonl",
    ]
    outputs.extend(args.data_root / "datasets" / f"{key}.jsonl" for key in RUN_ORDER)
    outputs.extend(args.data_root / "messages" / f"{key}.jsonl" for key in RUN_ORDER)
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite preparation artifacts: {existing}")

    from datasets import load_dataset

    dataset_spec = contract["dataset"]
    dataset = load_dataset(
        dataset_spec["hf_repo"],
        dataset_spec["config"],
        split=dataset_spec["split"],
        revision=dataset_spec["revision"],
        cache_dir=args.cache_dir,
    )
    tokenizers = load_tokenizers(args.model_root, contract)
    shared_refs, shared_counts, shared_seen = reservoir_select(
        dataset,
        args.rows,
        "8k_shared",
        SHARED_8K_MODELS,
        tokenizers,
        contract,
        8192,
        10240,
        args.seed,
    )
    long_refs, long_counts, long_seen = reservoir_select(
        dataset,
        args.rows,
        "16k_qwen2_5_3b",
        ("qwen2_5_3b",),
        tokenizers,
        contract,
        16384,
        18432,
        args.seed,
    )
    shared = materialize(dataset, shared_refs)
    long_rows = materialize(dataset, long_refs)
    selected_paths = {
        "8k_shared": args.data_root / "selected" / "8k_shared.jsonl",
        "16k_qwen2_5_3b": args.data_root / "selected" / "16k_qwen2_5_3b.jsonl",
    }
    atomic_jsonl(selected_paths["8k_shared"], shared)
    atomic_jsonl(selected_paths["16k_qwen2_5_3b"], long_rows)

    run_artifacts: dict[str, dict[str, Any]] = {}
    for run_key in RUN_ORDER:
        run = contract["runs"][run_key]
        model_key = run["model"]
        selected = shared if run["selection"] == "8k_shared" else long_rows
        dataset_path = args.data_root / "datasets" / f"{run_key}.jsonl"
        messages_path = args.data_root / "messages" / f"{run_key}.jsonl"
        atomic_jsonl(
            dataset_path,
            (
                build_training_record(
                    tokenizers[model_key],
                    row,
                    contract["models"][model_key]["apply_chat_template_kwargs"],
                    int(run["max_sequence_length"]),
                    int(run["trace_token_cap"]),
                    model_key,
                )
                for row in selected
            ),
        )
        atomic_jsonl(messages_path, (message_record(row) for row in selected))
        run_artifacts[run_key] = {
            "dataset": str(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "messages": str(messages_path),
            "messages_sha256": sha256_file(messages_path),
            "ordered_trace_sha256": ordered_trace_sha256(
                [str(row["trace_id"]) for row in selected]
            ),
            "rows": len(selected),
        }

    shared_hash = ordered_trace_sha256([str(row["trace_id"]) for row in shared])
    for run_key in ("qwen2_5_3b_8k", "qwen2_5_7b_8k", "qwen3_4b_8k", "qwen3_8b_8k"):
        if run_artifacts[run_key]["ordered_trace_sha256"] != shared_hash:
            raise RuntimeError(f"8K trace-set drift for {run_key}")
    stats = {
        "contract": {
            "dataset": dataset_spec,
            "rows_per_selection": args.rows,
            "seed": args.seed,
        },
        "invariants": {
            "correct_only": True,
            "no_truncation": True,
            "one_trace_per_unique_problem": True,
            "same_ordered_2k_across_all_8k_models": True,
            "shared_8k_requires_cross_tokenizer_eligibility": True,
            "synthetic_endoftext_appended": False,
        },
        "selection": {
            "8k_shared": {
                "eligible_problems_seen": shared_seen,
                "ordered_trace_sha256": shared_hash,
                "path": str(selected_paths["8k_shared"]),
                "sha256": sha256_file(selected_paths["8k_shared"]),
                "skip_counts": dict(shared_counts),
            },
            "16k_qwen2_5_3b": {
                "eligible_problems_seen": long_seen,
                "ordered_trace_sha256": ordered_trace_sha256(
                    [str(row["trace_id"]) for row in long_rows]
                ),
                "path": str(selected_paths["16k_qwen2_5_3b"]),
                "sha256": sha256_file(selected_paths["16k_qwen2_5_3b"]),
                "skip_counts": dict(long_counts),
            },
        },
        "runs": run_artifacts,
    }
    atomic_json(stats_path, stats)
    print(
        f"CORRECT_ONLY_PREP_COMPLETE shared_8k_rows={len(shared)} "
        f"qwen25_3b_16k_rows={len(long_rows)} shared_trace_sha256={shared_hash}",
        flush=True,
    )


if __name__ == "__main__":
    main()
