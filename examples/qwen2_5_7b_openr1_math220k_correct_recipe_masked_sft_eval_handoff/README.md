# Correct-recipe masked Qwen2.5-7B MATH-500 evaluation

This handoff evaluates the ten transferred iteration-124 checkpoints from
contract `07291674d43cf3da`. It runs sampled pass@1 only: three independent
complete MATH-500 passes at temperature 0.7, top-p 0.8, and top-k disabled.
For repeat `r` and benchmark row `i`, the seed is
`1234 + 500*r + i`. The reported result is the mean and sample standard
deviation of the three 500-problem pass accuracies; it is not majority vote or
pass@k.

The already-finalized Qwen2.5-7B base and 8K correct-only sampled results are
reused as comparison rows. This is an exact-policy reuse, not a rescore: the
preflight pins the result and policy file hashes, checkpoint identities,
dataset revision, scorer version, three-repeat structure, and the sampled
decoding, prompt, context-budget, stop, engine, and scoring contracts. The ten
new checkpoints are generated and scored in this run. The final audit labels
every row as `generated` or `reused` and reports paired repeat deltas from both
reference rows.

The 32,768-token limit is the total rendered prompt plus response context. For
each request, the worker computes `32768 - rendered_prompt_tokens`; there is no
fixed response cap and no safety reserve. Generation stops only on tokenizer
EOS `<|endoftext|>` (ID 151643), Qwen chat terminator `<|im_end|>` (ID 151645),
or exhaustion of the request's remaining context. `</think>`, boxed answers,
and answer text do not stop generation. All saved responses, including context
cap hits, are scored.

Scoring uses `math500_last_boxed_symmetric_scorer_v6`. It extracts only the
last complete, balanced, nonempty `\boxed{...}` group; there is no unboxed or
heuristic fallback. Candidate and gold independently enter the same fixed
interpretation ensemble, so routing never inspects the gold type. The ensemble
handles categorical MCQ/text, scalars and symbolic expressions, full
relations, ordered tuples, finite sets, unordered multi-answer lists,
intervals and interval unions, and matrices. Mathematical leaves are parsed
as whole explicitly wrapped LaTeX values with Math-Verify and checked with
strict symmetric verification (`float_rounding=15`, `numeric_precision=30`,
set/relation coercion disabled). Tuple order/arity, set cardinality and
one-to-one membership, interval endpoints/boundaries, relation structure, and
matrix shape are guarded explicitly. Fixed ambiguity interpretations are
tried symmetrically on both candidate and gold; exact decimals are represented
as exact fractions, terminal percent is presentation, and categorical
normalization stays in its own lane.

The GPU DAG has one CPU preflight, one real H200 sampled canary, ten independent
12-task H200 arrays (three repeats times four contiguous 125-row shards), ten
CPU finalizers, and one CPU audit: 120 generation tasks total. Each generation
task uses one H200, BF16, tensor parallelism 1, 0.90 GPU-memory utilization,
eight maximum concurrent sequences, 32-request chunks, a two-hour requeueable
allocation, and a ten-minute requeue signal. Node `g3130` is excluded. Every
GPU job uses job-local XDG, vLLM, TorchInductor, Triton, and temporary caches;
it does not depend on the user's pip or Triton home caches.

The audit reports sampled accuracy, valid-box rate, cap-hit rate, the number of
correct cap hits, parse failures, generated length, all repeat accuracies, and
paired accuracy deltas against both reused references.

Submission order is enforced by `submit.sh`: evaluator code must be committed,
pushed, clean, and equal to `origin/qwen-correct-only-openr1-sft-2k` before the
DAG can be submitted. After submission, `submission_metadata.json` is written
for a separate metadata commit.

```bash
bash examples/qwen2_5_7b_openr1_math220k_correct_recipe_masked_sft_eval_handoff/submit.sh --test-only-shapes
bash examples/qwen2_5_7b_openr1_math220k_correct_recipe_masked_sft_eval_handoff/submit.sh --submit
```
