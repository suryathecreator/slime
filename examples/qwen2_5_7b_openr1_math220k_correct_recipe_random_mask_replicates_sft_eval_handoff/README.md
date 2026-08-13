# Qwen2.5-7B random-mask replicate MATH-500 evaluation

This handoff evaluates the six transferred iteration-124 checkpoints from
contract `9c5acfdc3d3a6b44`: random masking at 5% and 15% for training mask
seeds 42, 43, and 44. It runs sampled pass@1 only, with three independent
complete MATH-500 passes per checkpoint at temperature 0.7, top-p 0.8, and
top-k disabled. Evaluation sampling is independent of the training mask seed:
for repeat `r` and benchmark row `i`, the generation seed is
`1234 + 500*r + i`. The reported result is the mean and sample standard
deviation of the three 500-problem accuracies, not majority vote or pass@k.

The completed 12-row masked-checkpoint aggregate is reused one-off and pinned
by SHA-256; none of those generations or scores are rerun. The primary table
inserts only `random_mask_05_seed42` and `random_mask_15_seed42` into that
comparison because the earlier random-mask sweep used training mask seed 42.
A separate replicate table reports all six new mask-rate × training-seed
checkpoints as individual rows.

The 32,768-token limit is the total rendered prompt plus response context.
Each request receives `32768 - rendered_prompt_tokens` response tokens, with
no fixed response cap or reserve. Generation stops only at tokenizer EOS
`<|endoftext|>` (151643), Qwen chat terminator `<|im_end|>` (151645), or
context exhaustion. All saved responses, including cap hits, are scored.

Scoring is unchanged: `math500_last_boxed_symmetric_scorer_v6` extracts only
the last complete, balanced, nonempty `\\boxed{...}`. There is no unboxed
or heuristic fallback. Candidate and gold independently enter the same fixed
interpretation ensemble; routing never inspects the gold type. The scorer
supports the categorical, scalar, symbolic, relation, tuple, set, unordered
multi-answer, interval/union, and matrix forms present in MATH-500, with
explicit structural guards and strict symmetric Math-Verify/SymPy checks.

The DAG is CPU preflight → one real H200 canary → six independent 12-task H200
arrays → six CPU finalizers → one CPU audit: 72 generation tasks total. Each
generation task uses one H200, BF16, TP1, eight maximum sequences, 32-request
chunks, 0.90 GPU-memory utilization, a two-hour requeueable allocation, and
job-local caches. Node `g3130` is excluded.

Submission requires committed, pushed, clean evaluator code equal to
`origin/qwen-correct-only-openr1-sft-2k`. Submission metadata is then written
for a separate metadata commit.

Attempt 1 completed its full checkpoint preflight, but its H200 canary failed
before generation because TorchInductor resolved a non-executable `nvcc`.
GPU jobs now source `gpu_runtime.sh`, which pins the executable CUDA 12.8.1
toolkit matching the PyTorch `+cu128` build. `submit_retry.sh` reuses that
completed preflight, preserves the attempt-1 record, and writes separate
attempt-2 control and tracked metadata.

```bash
bash examples/qwen2_5_7b_openr1_math220k_correct_recipe_random_mask_replicates_sft_eval_handoff/submit.sh --test-only-shapes
bash examples/qwen2_5_7b_openr1_math220k_correct_recipe_random_mask_replicates_sft_eval_handoff/submit.sh --submit
bash examples/qwen2_5_7b_openr1_math220k_correct_recipe_random_mask_replicates_sft_eval_handoff/submit_retry.sh --test-only-shapes
bash examples/qwen2_5_7b_openr1_math220k_correct_recipe_random_mask_replicates_sft_eval_handoff/submit_retry.sh --submit
```
