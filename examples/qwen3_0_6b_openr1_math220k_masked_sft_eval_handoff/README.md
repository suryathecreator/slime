# Qwen3-0.6B masked-SFT MATH-500 evaluation

This handoff evaluates the transferred `base_0p6b` model and all eleven final
iteration-124 Full-SFT checkpoints.  It uses the same Axolotl/vLLM generation
path as the earlier masked-SFT evaluations and the versioned
`math500_last_boxed_symmetric_scorer_v5` policy.

## Generation contract

- `HuggingFaceH4/MATH-500` test split at revision
  `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`, in its original 500-row order.
- User text is the boxed-answer instruction, two newlines, `Problem:`, and the
  original problem.  The tokenizer chat template adds the assistant prefix.
- Greedy BF16 generation, TP=1, temperature 0, one response, 32,768-token
  model context/output request, and EOS/`<|im_end|>` stopping.
- Four contiguous 125-row shards per checkpoint, 32-row durable chunks,
  Slurm requeue, and one H200 per shard.
- Every H200 task has an exclusive persistent vLLM, TorchInductor, Triton, XDG,
  and temporary cache namespace keyed by submission and shard.  The namespace
  survives requeue but cannot collide with another concurrent worker.
- The approved runtime is vLLM 0.10.2, Transformers 4.57.6, Torch 2.8.0,
  Math-Verify 0.9.0, and Axolotl commit
  `6b8f0e3314e3d162260cdc35d84741c3da163f30`.

The trained tokenizer overlay is used uniformly.  Preflight proves that all
base vocabulary token IDs are unchanged, the base and trained chat templates
are identical, and the only trained-only entries are `<think>`, `</think>`,
`<tool_response>`, and `</tool_response>`.

## Exact scoring contract

The complete saved assistant generation is scanned for balanced
`\boxed{...}` expressions.  The last complete, nonempty box is selected.
Nearby boxes are one multipart group only when at most 80 characters of
whitespace, punctuation, LaTeX spacing, or `and`/`or` separate them.  A prose
or display-math boundary breaks the group.  Every selected interior and its
source span are retained verbatim; normalized verification copies are stored
separately.

There is no unboxed fallback, final-line heuristic, answer-marker heuristic,
Llama call, or continuation.  Cap hits are scored from their saved response
exactly like natural stops.  All 500 rows remain in the denominator; missing
boxes, malformed-only boxes, parser failures, and mismatches are incorrect.

Candidate and gold are independently passed through the same fixed
interpreter.  No parser, type, or row override is chosen from the gold:

1. The LaTeX lane wraps the normalized value in `$...$` and calls `parse`
   with only `LatexExtractionConfig`, `fallback_mode="no_fallback"`,
   `extraction_mode="first_match"`, a five-second timeout, and raised errors.
2. The expression lane calls the same API on the unwrapped value with only
   `ExprExtractionConfig` and the same failure settings.
3. Only lanes returning exactly one object survive.  Same-type objects are
   checked using `verify(strict=True, allow_set_relation_comp=False,
   timeout_seconds=5)`.
4. Text uses fixed whole-value categorical/MCQ normalization.  It never seeds
   `StringExtractionConfig` from the gold.

The symmetric interpreter emits all applicable representations.  In
particular, `(a,b)` emits both ordered-pair and open-interval interpretations.
Structural guards require tuple order/arity, matrix shape, exact set or
multi-answer cardinality with one-to-one matching, interval endpoints and
boundaries, exact interval-union components, and complete relation objects.
It also handles top-level `\pm`, numeric thousands separators, categorical
text, MCQ spellings, and recognized presentation units.  Unit normalization
is symmetric and terminal-only; arbitrary text such as `\text{Evelyn}` is
not removed.

Preflight requires all 500 canonical gold strings to receive at least one
interpretation and verify against themselves under this fixed pipeline.

## Submit

The submission is a CPU preflight, one-H200 generation/scoring canary, twelve
unthrottled arrays of `0-3` (48 H200 tasks), one CPU finalizer per checkpoint,
and a final audit:

```bash
bash examples/qwen3_0_6b_openr1_math220k_masked_sft_eval_handoff/submit.sh \
  --test-only-shapes
bash examples/qwen3_0_6b_openr1_math220k_masked_sft_eval_handoff/submit.sh \
  --submit
```

If a submitted DAG must be superseded after a verified infrastructure failure,
run the same launcher with `--resubmit`.  That mode validates the recorded
failure, cancels the old DAG, archives its journal and canary, and submits the
full standard DAG again.  Durable complete shard chunks remain in place and
are reused; only incomplete or missing rows are generated again.

Each H200 request is two hours, 8 CPUs, and 96 GB, with a ten-minute requeue
signal.  The launcher refuses a dirty or unpushed SLIME branch, a changed
Axolotl commit, incomplete checkpoints, or a stale benchmark.  A normal submit
also refuses a duplicate journal; a guarded resubmit archives it and rolls back
any new jobs if the replacement DAG cannot be recorded completely.

Final tables report correct/500, accuracy, valid-box rate, cap-hit count/rate,
correct cap hits, cap-hit accuracy, and parse failures.  Raw generations and
full traces remain under the ignored checkpoint tree; compact results are
installed under the 0.6B example results directory.
