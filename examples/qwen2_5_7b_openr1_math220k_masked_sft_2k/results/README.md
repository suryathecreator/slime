# MATH-500 result snapshots

This directory publishes the results obtained so far for the Qwen2.5-7B base
model and all eleven 2K trained checkpoints. Every variant directory contains
the native Axolotl MATH-500 result and its checkpoint, evaluation-policy, and
provenance manifests. The `rescored/math500_llama33_definitive_strict_v2`
subdirectory contains the corresponding provisional Llama-gated result and its
checksum.

The strict-v2 files are historical, reproducible snapshots rather than the
settled metric. We will likely switch to using Llama only for the binary
definitive-answer-state decision. For the checks already completed, rescoring
will reuse only the existing `definitive` decision and will ignore subsequent
Llama-produced extraction fields. Future checks will ask Llama only for that
decision. Deterministic heuristics will then extract the final committed answer
from the exact saved assistant response and pass it to the structural and
symbolic verifier.

This planned scorer change is entirely post-hoc. It will be versioned and run
uniformly over all existing and future saved generations, so it does not change
generation fidelity or silently rewrite the result snapshots published here.

See the [evaluation handoff](../../qwen2_5_7b_openr1_math220k_masked_sft_eval_handoff/README.md)
for the exact runtime, gate, scoring, cap-hit, and recovery policies.

The [provisional analysis](ANALYSIS.md) records the comparison table, cap-hit
sensitivity, observed failure modes, and manual Llama-gate audit that motivated
the planned scorer change.
