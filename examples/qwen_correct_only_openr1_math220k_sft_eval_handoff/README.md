# Correct-only Qwen MATH-500 evaluation

This is submission attempt 2 and writes to `math500_greedy_sampled_v2`.
Attempt 1 is retained as provenance but produced no generation records: its
preflight completed, then the Qwen2.5 canary exhausted its requeues when the
vLLM architecture-inspection child process imported `mistral-common` and found
the transitive `pycountry` dependency missing. The original submission and
startup-intervention metadata are preserved in the `attempt_01_*` files; its
failed `math500_greedy_sampled_v1` output is not reused.

This handoff evaluates four pretrained bases and five final iteration-124
`correct_only` checkpoints transferred under contract `3568736a3743c319`.
The Qwen2.5-3B base is generated once and used for both its 8K and 16K
comparisons, so there are nine unique generation targets and five comparisons.

Each target receives one greedy MATH-500 pass and three independent sampled
passes. Greedy decoding uses temperature 0 and top-p 1. Sampled decoding uses
temperature 0.7 and top-p 0.8. Sample `r` for benchmark row `i` uses seed
`1234 + 500*r + i`; the identical seed schedule is used for every checkpoint.
Reported sampled accuracy is the mean and sample standard deviation across the
three complete 500-problem passes, not majority vote or pass@k.

The 32,768-token limit is total rendered prompt plus response context. The
worker explicitly computes `32768 - rendered_prompt_tokens` for each request
and records both values. There is no safety reserve. Exactly two tokens stop
generation: tokenizer EOS `<|endoftext|>` (`151643`) and Qwen chat terminator
`<|im_end|>` (`151645`). `</think>`, boxed answers, and answer text do not stop
generation.

The exact runtime is recorded in `runtime_pins.txt` and in the evaluation
manifest. In addition to vLLM 0.10.2, Transformers 4.57.6, Torch 2.8.0,
Math-Verify 0.9.0, Tokenizers 0.22.2, Triton 3.4.0, and NumPy 2.2.6, it pins
the complete import chain that failed in attempt 1: mistral-common 1.11.7,
pydantic-extra-types 2.11.1, pycountry 26.2.16, SoundFile 0.14.0, soxr 1.1.0,
cffi 2.0.0, and pycparser 3.0. The tracked runtime launcher binds those
packages to the cluster's intact Python 3.11.5 executable; this avoids the
installed coenv Python 3.11.9 build, whose standard library is missing
`_ctypes`.

Before submission and again in the CPU preflight, the controller checks every
exact package version, runs `pip check`, and starts a fresh Python subprocess
that imports the vLLM generation surface plus both `Qwen2ForCausalLM` and
`Qwen3ForCausalLM`. The 30-minute H200 canary then performs one real greedy
Qwen2.5 generation and one real sampled Qwen3 generation before any arrays are
released. H200 jobs exclude node `g3130`, which preempted two attempt-1 canary
runs before Python startup; generation arrays retain the two-hour/requeue
allocation.

Scoring does not inspect the gold answer to choose a parser or route. Candidate
and gold independently enter the same fixed interpretation ensemble. It
recognizes MCQ/text, scalar or symbolic expressions, complete relations,
ordered tuples, finite sets, unordered multi-answer lists, intervals and
interval unions, and matrices. Mathematical leaves are each parsed as one
whole explicitly wrapped LaTeX value with Math-Verify, then compared with
strict symmetric verification (`float_rounding=15`, `numeric_precision=30`,
set/relation coercion disabled). Ordered values preserve arity and order; sets
and unordered answers require equal cardinality and a one-to-one match;
interval endpoints and open/closed boundaries must match; relations must parse
as relations; matrices must retain shape. Ambiguous parentheses are evaluated
through every fixed applicable interpretation on both sides, rather than by
looking at the gold type. Exact decimals are rewritten to exact fractions for
parsing, terminal percent notation is treated as presentation, and MCQ/text is
normalized only within its own categorical lane.

Every saved response, including a length-capped response, is scored with
`math500_last_boxed_symmetric_scorer_v6`. It selects only the last complete,
nonempty balanced boxed group and has no unboxed fallback. GPU generation is
immutable and CPU scoring is post-hoc.

After implementation is committed and pushed:

```bash
bash examples/qwen_correct_only_openr1_math220k_sft_eval_handoff/submit.sh --test-only-shapes
bash examples/qwen_correct_only_openr1_math220k_sft_eval_handoff/submit.sh --submit
```
