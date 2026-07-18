# Qwen3-1.7B OPD with sampled AIME 2026 evaluation

This isolated replacement chain starts from the post-trained `Qwen/Qwen3-1.7B`
student, uses post-trained `Qwen/Qwen3-8B` as the OPD teacher, and evaluates the
student base, teacher base, OPD-1K, and cumulative OPD-5.1K checkpoints on the
30 problems in `MathArena/aime_2026`. It does not overwrite the earlier HARP or
Math-500 artifacts.

The dataset is pinned to revision
`d2de22f3c656b4f56cf8981212186377d1e23bc3`. Setup downloads its parquet file,
normalizes exactly problem indices 1 through 30, and records both raw and
normalized SHA-256 hashes. The pinned parquet hash is
`d91db799651b4cc1f0734f52792a695c9cc60dac342524b3d8e5b2ff31c3e957` and
the canonical JSONL hash is
`6e07e802a416526fe52559216e6e3f13f35db1c169443a39c7ae2790c8b5ca2b`.
Model downloads are pinned to revisions
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (student) and
`b968826d9c46dd6066d109eabc6255188de91218` (teacher).

## Evaluation protocol

Every checkpoint receives exactly 480 independent requests: 16 samples for
each of 30 problems. Sampling uses thinking mode, temperature 0.6, top-p 0.95,
top-k 20, min-p 0, a 32,768-token model context, and a 31,744-token response
cap. The deterministic seed for zero-based problem position `p` and sample
index `s` is `42 + 16*p + s`.

Four H200 workers each receive 120 requests in one synchronous
`llm.generate(list_of_prompts, sampling_params=list_of_params)` call. Each shard
therefore has a continuously populated real request queue; `max_num_seqs` is
20 for the 1.7B student and 12 for the 8B teacher. Prefix caching, chunked
prefill, and CUDA graphs are enabled, with eager fallback only if graph engine
initialization fails.

Per-problem pass@1 is the fraction correct among its 16 generations. The
benchmark value is the mean of the 30 fractions, equivalently total correct
divided by 480. This is a 16-sample Monte Carlo estimate of single-sample
accuracy, not pass@16 and not majority-vote accuracy.

## Scoring

`aime_answer_v1.py` is numerical-only. Natural stops use the final post-think
region. The scorer prefers the chronologically last balanced box or explicit
answer marker. If neither exists, it accepts a final integer only when it is a
standalone answer-like line or conclusion, rejecting equation operands and
incidental intermediate numbers. Cap hits scan the unfinished trace with the
same strong-event preference and conservative terminal-number heuristic. Every
answer event, span, candidate, selection method, parsed integer, and decision is
stored with the generation.

## Chain

OPD uses three colocated student/rollout GPUs and one 8B-teacher GPU. The first
stage trains on 1,024 prompts. The second consumes 4,096 disjoint additional
prompts while resuming the full optimizer, RNG, and training state, producing
the cumulative 5,120-prompt checkpoint. OPD rollout decoding remains its
established temperature-1/top-p-1 policy; the new temperature-0.6 protocol is
for benchmark evaluation.

```text
setup → 1.7B base eval → 8B teacher eval → report
                                             → OPD-1K → eval → report
                                                                      → OPD-5.1K → eval → final report
```

Dependencies are strict serial `afterok`, so no more than four GPUs are active
at once. Submit from the repository root with:

```bash
bash examples/qwen3_1_7b_opd_aime2026/submit_chain.sh
```

Artifacts are rooted at
`/gpfs/scrubbed/suryadv/slime-qwen3-1.7b-opd-aime2026`.
