# Strict-English v2 SFT -> OPD experiment

## Why this run exists

The first cleaned run removed incomplete thinking traces but retained substantially more multilingual/noisy material than the comparison implementation.

| Category | First cleaner | Comparison | Strict-v2 gate |
|---|---:|---:|---:|
| Incomplete think | 743,814 | 749,380 | within +/-1% of comparison |
| Non-English/mixed | 94,920 | 397,642 | at least 377,760 |
| Overall valid | 429,713 | 332,843 | at most 349,486 |

Categories overlap. Over-filtering is accepted; under-filtering relative to these gates blocks training.

## Filtering contract

The reusable all-domain corpus requires a complete `<think>...</think>` trace, strict English in the prompt/reasoning/final-answer sections, and basic noise hygiene. Language detection masks LaTeX, equations, code, URLs, numbers, operators, and isolated mathematical Greek symbols. It then rejects non-English Latin-script prose, meaningful foreign spans inside English text, non-Latin prose, multilingual responses, and obvious web/gibberish noise. The implementation uses `lingua-language-detector==2.2.0` in low-memory all-language mode.

The strict math subset additionally requires `domain == math`, exactly one ordered think-tag pair, a nonempty answer after `</think>`, math-heavy content, and no repeated-line, repeated 8-gram, duplicate-sentence, or repeated-suffix loops. Source shards are atomic and resumable. The complete accepted corpus is retained as sharded Parquet with source IDs and hashes.

Candidates are shuffled once with seed `1234`. Up to 50,000 are selected. With at least 10,000 candidates, the final 5,000 are the OPD reserve and the remainder are SFT; otherwise the selection is split 50/50. SFT is truncated only to a complete 250-row batch. OPD is truncated only to a complete 128-row batch: a 5,000-row reserve therefore trains on 4,992 rows, as 1,024 then 3,968 disjoint continuation rows. Metadata proves source-ID and prompt-hash disjointness.

## Training and evaluation

- SFT starts from Qwen3-8B-Base: 4 H200s, TP=2, CP=1, DP=2, Megatron distributed optimizer, Qwen3 loss mask, one epoch, LR `1e-6`, global batch 250, and 16,384 max packed tokens/GPU. Model snapshots are emitted every 5,000 effective samples; only the newest complete checkpoint retains optimizer state.
- OPD uses the validated colocated 4-H200 layout: actor/rollout/teacher `3/3/1`, TP=1, CP=3, 32,766 actor sequence length, per-prompt dynamic response cap, packing 2,048, logprob chunks 512, zero train margin, CPU/offload/recompute, one pass per phase, and LR `1e-6`.
- Actor and fixed diagnostic KL reference both initialize from the new final SFT snapshot.
- Generation explicitly enables Qwen3 thinking, stops on `<|im_end|>` (`151645`) or `<|endoftext|>` (`151643`), preserves special-token text and raw IDs, and uses schema-2 resumable eval fingerprints.
- MATH-500 is run after SFT, OPD-1k, and final OPD. The valid historical base summary from job `165035` is reused.

## Old-job cleanup

The prior corrected-data tail was superseded and canceled on 2026-07-11: `167285` through `167290`. Job `167285` had resumable eval chunks covering 128 examples; those chunks, all prior checkpoints, and all diagnostics remain on disk. No artifacts were deleted. They are not reused in strict-v2 because the training dataset and model revision are different.

## Scheduling

Initial submission is language setup -> resumable cleaner -> dynamic dispatcher. The dispatcher reads the validated split metadata and submits the exact batch-aligned SFT/eval/OPD chain with strict `afterok` dependencies. No job requests more than four H200s and the chain is serialized. Walltimes are 6h setup, 24h cleaning, 24h each training/eval stage, and 6h report.
