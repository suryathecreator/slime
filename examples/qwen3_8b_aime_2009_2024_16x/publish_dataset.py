#!/usr/bin/env python3
"""Build audited Parquet shards and publish the completed dataset to HF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import pipeline


def publication_row(
    sample: dict[str, Any], corpus_row: dict[str, Any]
) -> dict[str, Any]:
    trace = sample["scorer_trace"]
    base = trace["base_scorer_trace"]
    return {
        "sample_id": sample["sample_id"],
        "eval_index": sample["eval_index"],
        "draw_index": sample["draw_index"],
        "seed": sample["seed"],
        "problem_id": sample["problem_id"],
        "year": sample["year"],
        "part": sample["part"],
        "contest": sample["contest"],
        "problem_number": sample["problem_number"],
        "problem": sample["problem"],
        "has_asy": bool(corpus_row["has_asy"]),
        "prompt": sample["prompt"],
        "messages_json": pipeline.canonical_json(sample["messages"]),
        "rendered_prompt": sample["rendered_prompt"],
        "response": sample["response"],
        "gold_answer_raw": sample["answer_raw"],
        "gold_answer_canonical": sample["answer_canonical"],
        "accepted_answer_integers": sample["accepted_answer_integers"],
        "extracted_answer_raw": trace["official_candidate_raw"],
        "extracted_answer_integer": trace["parsed_integer"],
        "is_scorable": trace["is_scorable"],
        "is_correct": trace["is_correct"],
        "matched_gold_integer": trace["matched_gold_integer"],
        "decision_reason": trace["official_decision_reason"],
        "complete_box_found": base["selected_box_event_id"] is not None,
        "cap_hit": sample["cap_hit"],
        "finish_reason": sample["finish_reason"],
        "stop_condition": sample["stop_condition"],
        "stop_token_id": sample["stop_token_id"],
        "input_tokens": sample["input_tokens"],
        "output_tokens": sample["output_tokens"],
        "effective_max_new_tokens": sample["effective_max_new_tokens"],
        "model_id": sample["model_id"],
        "model_revision": sample["model_revision"],
        "generation_parameters_json": pipeline.canonical_json(
            sample["generation_parameters"]
        ),
        "scorer_version": trace["scorer_version"],
        "scorer_trace_json": pipeline.canonical_json(trace),
        "source_kind": sample["source"]["kind"],
        "source_dataset": sample["source"]["dataset"],
        "source_revision": sample["source"]["revision"],
        "source_url": sample["source"]["url"],
        "source_record_sha256": sample["source_record_sha256"],
        "rendered_prompt_sha256": sample["rendered_prompt_sha256"],
        "response_sha256": sample["response_sha256"],
        "contract_sha256": sample["contract_sha256"],
    }


def readme(policy: dict[str, Any], summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    return f"""---
license: other
task_categories:
- text-generation
language:
- en
tags:
- aime
- qwen3
- reasoning
- chain-of-thought
- sampled-generations
pretty_name: Qwen3-8B AIME 2009-2024 16x
size_categories:
- 1K<n<10K
---

# Qwen3-8B AIME 2009–2024, 16 sampled completions per problem

This dataset contains {overall['row_count']:,} audited completions from the official
post-trained `{policy['model']['id']}` model at revision
`{policy['model']['revision']}`. It covers AIME I and AIME II from 2009 through
2024: exactly 30 problems per year, 480 unique prompts, and 16 independently
seeded completions per prompt.

## Generation contract

- Thinking is enabled through Qwen's chat template (`enable_thinking=True`).
- There is one user message and no system message.
- Total context is 32,768 tokens; each response budget is
  `32768 - rendered_prompt_tokens`.
- Sampling: temperature 0.6, top-p 0.95, top-k 20, min-p 0.0.
- Generation stops on `<|endoftext|>` (151643) or `<|im_end|>` (151645), and
  the stop control token is not included in `response`.
- Seed: `1234 + 480 * draw_index + eval_index`.

The exact user message is:

```text
Please reason step by step, and put your final answer within \\boxed{{}}.

Problem:
{{problem}}
```

## Problems and provenance

The primary source is
[`gneubig/aime-1983-2024`](https://huggingface.co/datasets/gneubig/aime-1983-2024)
at revision `5d610df981dec508dd93d0a16333a029ac1739d8`; it supplies 455 of the
480 rows. That source is incomplete for the requested years. The other 25
problem statements are exact English MAA text exposed by
[LIVE by Po-Shen Loh](https://live.poshenloh.com/past-contests/aime), whose
pages state that the problems are used with official legal permission. Each
supplement row includes its page URL and frozen page-content hash. The build
also checked 95 overlapping answers across the two sources to validate contest
ordering. `data/problems.jsonl`, `corpus_manifest.json`, and
`source_inputs.json` preserve the canonical source manifest.

The problem statements remain copyrighted by their respective owner(s); this
dataset therefore uses `license: other`. Preserve the included attribution and
consult the linked sources before redistribution or commercial use.

## Scoring

Every completion, including a 32K cap hit, is scored by the included
`aime_scorer.py` (`aime_last_boxed_integer_scorer_v1`). The scorer scans the
entire visible assistant response for balanced, complete, nonempty
`\\boxed{{...}}` expressions and selects the last such box. It removes only the
outer box plus fixed presentation wrappers/math delimiters, then requires the
remaining content to be exactly an optional `+` followed by digits whose
integer value is in 0–999. It does not use an unboxed-answer fallback or any
heuristic extraction. The 2022 AIME II problem 8 row accepts either 080 or 081,
because both answers were officially accepted; the unchanged scorer is run
against both accepted gold integers and the completion is correct if either
matches.

Overall audit snapshot:

- Accuracy: {100 * overall['accuracy']:.2f}% ± {100 * overall['accuracy_standard_error']:.2f}% standard error
- Scorable last-box rate: {100 * overall['scorable_rate']:.2f}%
- Complete-box rate: {100 * overall['complete_box_rate']:.2f}%
- 32K cap-hit rate: {100 * overall['cap_hit_rate']:.2f}%
- Mean completion length: {overall['mean_output_tokens']:.1f} tokens

`summary.json` contains per-draw, per-year, and per-contest aggregates.

## Files and row schema

- `data/draw-00.parquet` through `data/draw-15.parquet`: one row per completion.
- `data/problems.jsonl`: the canonical 480-problem manifest.
- `aime_scorer.py`: the exact scorer used for every generation.
- `eval_policy.json`: model, prompt, sampling, stop, context, and sharding pins.
- `summary.json`: audited aggregate statistics.

The Parquet rows include the problem and exact prompt, raw assistant response,
raw/canonical/accepted gold answers, extracted last-box answer, parse/correct
flags, finish and cap status, token counts, seed/draw, source provenance,
model/scorer/contract identities, and a JSON serialization of the complete
scorer trace.
"""


def build_staging() -> tuple[Path, dict[str, Any]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    policy = pipeline.load_policy()
    root = pipeline.output_root(policy)
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError("audited summary is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or summary.get("row_count") != 7680:
        raise RuntimeError("audit is not a complete 7,680-row result")
    corpus = pipeline.load_corpus()
    staging = root / "hf_staging"
    data_dir = staging / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    schema_text = None
    for draw in range(16):
        rows = []
        for eval_index, corpus_row in enumerate(corpus):
            sample = json.loads(
                pipeline.sample_path(root, draw, eval_index).read_text(encoding="utf-8")
            )
            rows.append(publication_row(sample, corpus_row))
        table = pa.Table.from_pylist(rows)
        schema_text = str(table.schema)
        pq.write_table(
            table,
            data_dir / f"draw-{draw:02d}.parquet",
            compression="zstd",
            compression_level=9,
        )
    shutil.copy2(pipeline.CORPUS_PATH, data_dir / "problems.jsonl")
    shutil.copy2(pipeline.POLICY_PATH, staging / "eval_policy.json")
    shutil.copy2(pipeline.CORPUS_MANIFEST_PATH, staging / "corpus_manifest.json")
    shutil.copy2(pipeline.HERE / "source_inputs.json", staging / "source_inputs.json")
    shutil.copy2(pipeline.scorer_path(policy), staging / "aime_scorer.py")
    shutil.copy2(summary_path, staging / "summary.json")
    (staging / "README.md").write_text(readme(policy, summary), encoding="utf-8")
    (staging / "SCHEMA.txt").write_text(str(schema_text) + "\n", encoding="utf-8")
    files = []
    for path in sorted(staging.rglob("*")):
        if path.is_file() and path.name != "PUBLICATION_MANIFEST.json":
            files.append(
                {
                    "path": path.relative_to(staging).as_posix(),
                    "sha256": pipeline.sha256_file(path),
                    "size": path.stat().st_size,
                }
            )
    manifest = {
        "contract_sha256": pipeline.contract_sha256(policy),
        "files": files,
        "generated_at": pipeline.now_iso(),
        "repo_id": policy["huggingface_publication"]["repo_id"],
        "row_count": 7680,
    }
    (staging / "PUBLICATION_MANIFEST.json").write_text(
        pipeline.pretty_json(manifest), encoding="utf-8"
    )
    return staging, manifest


def publish() -> None:
    from huggingface_hub import HfApi

    policy = pipeline.load_policy()
    staging, manifest = build_staging()
    repo_id = policy["huggingface_publication"]["repo_id"]
    api = HfApi()
    identity = api.whoami()
    if identity.get("name") != repo_id.split("/", 1)[0]:
        raise RuntimeError(
            f"cached HF identity {identity.get('name')!r} cannot publish {repo_id}"
        )
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=policy["huggingface_publication"]["private"],
        exist_ok=True,
    )
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(staging),
        commit_message=(
            "Publish audited Qwen3-8B AIME 2009-2024 16x generations "
            f"({manifest['contract_sha256'][:12]})"
        ),
    )
    publication = {
        "commit": str(commit),
        "completed_at": pipeline.now_iso(),
        "contract_sha256": manifest["contract_sha256"],
        "repo_id": repo_id,
        "status": "published",
    }
    destination = pipeline.output_root(policy) / "hf_publication.json"
    pipeline.atomic_text(destination, pipeline.pretty_json(publication))
    print(f"QWEN3_AIME_HF_PUBLISHED repo={repo_id} commit={commit}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "publish"))
    args = parser.parse_args()
    staging, manifest = build_staging() if args.command == "build" else (None, None)
    if args.command == "build":
        print(
            f"QWEN3_AIME_HF_STAGING_READY path={staging} "
            f"files={len(manifest['files'])}",
            flush=True,
        )
    else:
        publish()


if __name__ == "__main__":
    main()
