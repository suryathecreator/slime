# Qwen2.5-7B base plus 2K suite handoff

This handoff transfers and evaluates the pinned Qwen2.5-7B base and all eleven
2K full-SFT variants. Tillicum produces content-addressed checkpoint manifests;
Klone verifies every file before inference.

The evaluation is the established Axolotl MATH-500 pipeline at commit
`6b8f0e3314e3d162260cdc35d84741c3da163f30`: greedy decoding, 32K model
length, a request for 32K new tokens, Qwen EOS and `<|im_end|>` stop IDs, vLLM
0.10.2, transformers 4.57.6, torch 2.8.0, and math-verify 0.9.0. Each of the 12
checkpoints uses four contiguous 125-problem shards, for 48 one-H200 tasks.

The currently published provisional rescore is
`math500_llama33_definitive_strict_v2`. It uses pinned
`nvidia/Llama-3.3-70B-Instruct-FP8` revision
`68579f6750089fc4ccf3d3b58d9f4648d1eed615` only as a definitive-answer gate
and source-span extractor. The gate sees the problem and the complete raw
assistant completion, but no gold answer and no native correctness fields.
It runs greedily with JSON-schema constrained decoding (XGrammar when the
schema is supported, with vLLM's LLGuidance fallback), ModelOpt W8A8 FP8 weights, BF16 KV
cache, and one H200. BF16 KV avoids uncalibrated FP8-cache scales, while the
67.7-GiB FP8 weights plus a full 131K BF16 cache fit within one H200. Like
generation, the gate is split into four contiguous
125-response shards per model: another 48 one-H200 tasks. Every decision is
flushed and fsynced before the next row, and every shard safely resumes after
requeue. Each H200 shard requests 30 minutes and checkpoints on a three-minute
warning. Llama first emits only adjudication labels and an evidence-region ID.
If definitive, a second constrained pass emits only a candidate ID or two
LaTeX-aware lexer-boundary IDs. `final_answer_raw` and quotes are then sliced
mechanically from the original response, so Llama cannot generate answer text.
Each stage gets three total attempts. Exhaustion becomes a durable unresolved
row and the shard continues instead of failing or repeatedly requeuing.

The CPU scorer passes only the extracted final answer to Math-Verify 0.9.0.
It keeps the heuristic gold-type route, evaluates structurally compatible
fallback routes when needed, and leaves conflicting route outcomes unresolved.
Explicit guards require ordered tuple arity/order, exact
finite-set cardinality with one-to-one matching, exact interval boundaries,
and complete equations. Semantic correctness and boxing compliance are
reported separately. Each result also reports native cap-hit count/percent,
verified-correct cap hits, and strict-correct counts/rates under the two
extremes where all cap hits are treated as wrong or all are treated as right.
If any gate or verifier row remains unresolved, final accuracy and affected
cap metrics stay null. The result still publishes coverage, provisional
correct count, and lower/upper bounds. In particular it reports cap-hit
percent, cap hits known correct (or bounds), and counts plus rates for treating
all cap hits as wrong or all as right.

## Region-verbatim v3 scorer

The checked-in strict-v2 results remain immutable historical snapshots. The
previously proposed Llama-only-definitiveness replacement was not adopted.
The settled `math500_llama33_region_verbatim_strict_v3` policy retains the
unchanged adjudication call, including its evidence-region selection. A second
call sees only that selected region, the exact boxed contents, and other exact
heuristic candidates. It returns constrained JSON containing a selection mode
and `final_answer_raw` copied verbatim. It does not return candidate IDs or
lexer-boundary IDs.

The modes are `boxed_exact`, `heuristic_exact`, and `verbatim_region`.
JSON-schema guided decoding constrains the object shape and mode but cannot by
itself guarantee source fidelity. Post-decoding validation therefore requires
the text to be, respectively, one complete balanced box content span, one
eligible heuristic span, or one non-box contiguous substring of the selected
region. Boxing compliance is derived from the validated source mode rather
than trusted from the adjudication labels. Three failed attempts produce a
durable unresolved row and processing continues.

The motivation is narrower than replacing Llama extraction wholesale.
Selecting opaque candidate and especially lexer-boundary IDs introduced
avoidable errors, while many exact boxed and heuristic selections were valid.
V3 keeps those exact texts as guidance but lets the model copy the answer in
the natural representation needed by the symbolic verifier.

For the current 6,000 immutable generations, preparation performs a one-off
reuse of the 5,991 valid v2 adjudications only after pinning the v2 policy,
response hashes, and region catalog. It reruns v3 extraction for every one of
the 2,196 definitive adjudications, including old candidate-ID successes, and
does not reuse any old answer extraction. The 3,795 nondefinitive rows need no
second call; the nine rows that already exhausted adjudication remain
unresolved. Future v3 evaluations run both calls normally, so this import is a
compute optimization for the existing corpus rather than a scoring exception.

Unresolved gate or verifier rows are shown as strict lower–upper bounds: the
lower endpoint treats each unresolved row as wrong and the upper endpoint
treats it as right. They are also written to an unresolved-review manifest for
versioned review by a stronger model reasoning over the full trace or by a
human. Native generations and strict-v2 results are never overwritten.

## Source-mapped v4 scorer

The subsequent `math500_llama33_source_mapped_strict_v4` policy corrects the
v3 extraction contract. The unchanged first call still decides definitiveness
and selects one evidence region. In the second call, Llama generates the answer
content and a mode. For `boxed_exact` and `heuristic_exact`, presentation-only
differences are used only to identify a candidate; the exact candidate text
sliced from the response is what reaches the symbolic verifier. This prevents
constrained decoding artifacts such as full-width reverse slashes, Unicode
radicals, or added math delimiters from changing the scored candidate.

`verbatim_region` is the fallback beyond both candidate catalogs. Its generated
answer is retained for symbolic verification only if its content occurs in the
selected non-box source region after a versioned, non-solving presentation
normalization. That normalization handles wrappers, whitespace, Unicode/LaTeX
glyph presentation, font/size commands, simple fraction/radical notation, and
encoding artifacts observed across the prior extraction attempts. It does not
use the gold answer or mathematical equivalence to establish source presence.
Candidate modes cannot silently fall back to a shorter source substring.

For the existing 6,000 generations, v4 is a one-off CPU-only replay. It reuses
the immutable v2 adjudications, every v3 decoded-attempt journal entry in its
original order, and every accepted v3 extraction. The first attempt that the v4
validator would have accepted is selected; no generation or Llama inference is
rerun. All symbolic and structural checks are then recomputed into new v4
artifacts. Future evaluations run this validation directly and retain decoded
failure attempts so later presentation-normalization revisions remain
post-hoc and generation-fidelity preserving.

Every comparison table reports both the strict lower–upper range and a separate
`midpoint ± eval noise` column. The midpoint is the center of the unresolved
range and the displayed noise is its half-width; percentages express the
half-width in percentage points. The strict endpoints remain the authoritative
known-wrong/known-right extremes.

The tokenizer overlay is byte-identical to the pinned base
`Qwen/Qwen2.5-7B@d149729398750b98c0af14eb82c78cfe92750796`; Qwen2.5 needs no
metadata compatibility rewrite.

The eleven training bundles contain stray Qwen3 tokenizer artifacts even
though every config and weight set is Qwen2.5-7B. The Qwen2-specific
controller rejects any non-Qwen2 model config, ignores those bundled tokenizer
files, and builds the inference overlay only from the hash-pinned `base_7b`
tokenizer. `TOKENIZER_PROVENANCE.json` records both the selected hashes and the
ignored per-checkpoint hashes.

After every training job and manifest has finished:

```bash
bash examples/qwen2_5_7b_openr1_math220k_masked_sft_eval_handoff/rsync_to_klone.sh --check
bash examples/qwen2_5_7b_openr1_math220k_masked_sft_eval_handoff/rsync_to_klone.sh --transfer
bash examples/qwen2_5_7b_openr1_math220k_masked_sft_eval_handoff/rsync_to_klone.sh --verify
```

Remote verification uses the sibling Axolotl checkout's `.venv/bin/python`,
not Klone's legacy system `python3`. Set `REMOTE_PYTHON` only if that venv is
located elsewhere.

On Klone, with this branch checked out and the sibling Axolotl repository at
the pinned commit:

```bash
bash examples/qwen2_5_7b_openr1_math220k_masked_sft_eval_handoff/setup_llama_gate_runtime.sh
bash examples/qwen2_5_7b_openr1_math220k_masked_sft_eval_handoff/submit_available_axolotl.sh --test-only-shapes
bash examples/qwen2_5_7b_openr1_math220k_masked_sft_eval_handoff/submit_available_axolotl.sh --submit
```

The setup command creates an isolated ignored venv under `checkpoints/runtime`
with exact package pins and downloads the exact FP8 model revision into
`checkpoints/models`. It is lock-protected and resume-safe. No sibling Axolotl
environment is modified.

The submission DAG is:

```text
native preflight -> 48 generation tasks -> 12 merges -> native audit
  -> redacted gate inputs -> v2 edge-case canary -> 48 Llama gate tasks
  -> strict CPU score -> final or provisional audit
```

Gate inputs live below the no-gold `_llama_gate_v2_no_gold` directory. The gold
structure manifest lives separately below `_llama_strict_scoring_v2`; the GPU
command has no code path that opens it. Compact strict results are written to
`results/<variant>/rescored/math500_llama33_definitive_strict_v2/` alongside,
without replacing, the native results.

When native generation is already complete, submit only the clean v2 recovery
DAG with:

```bash
bash examples/qwen2_5_7b_openr1_math220k_masked_sft_eval_handoff/resubmit_llama_gate_v2.sh
```

To reuse the completed adjudications and submit the extraction-only v3 DAG for
the current corpus, run:

```bash
bash examples/qwen2_5_7b_openr1_math220k_masked_sft_eval_handoff/submit_llama_gate_v3_reuse.sh
```

That DAG is preparation/import → v3 H200 canary → 48 extraction-only one-H200
tasks → CPU scoring/table generation → audit. Each extraction shard requests
20 minutes and remains row-checkpointed and requeue-safe.

The gate identity hashes the model/runtime pins, both prompt files, both
constrained schemas, and the span-catalog implementation version. An unchanged
gate policy can later backfill only unresolved rows with `run-gate
--overlay-id <id>`; score journals detect the changed gate-record hash and
recompute those rows. A changed extraction policy gets a new gate version over
all 6,000 responses. A changed verifier gets a new scorer/metric version over
all existing gate rows, without changing or rerunning native generations.

`checkpoint_sources.json` names every Tillicum source. The Hyak-relative
runtime inventory is `available_eval_manifest.json`.
