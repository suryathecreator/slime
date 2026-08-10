# MATH-500 result snapshots

This directory publishes the results obtained so far for the Qwen2.5-7B base
model and all eleven 2K trained checkpoints. Every variant directory contains
the native Axolotl MATH-500 result and its checkpoint, evaluation-policy, and
provenance manifests. The `rescored/math500_llama33_definitive_strict_v2`
subdirectory contains the corresponding provisional Llama-gated result and its
checksum.

The strict-v2 files are historical, reproducible snapshots rather than the
settled metric. The previously proposed switch to using Llama only for the
binary definitive-answer-state decision was not adopted. The replacement
`math500_llama33_region_verbatim_strict_v3` scorer keeps two Llama calls: the
unchanged first call decides definitiveness and selects one evidence region;
the second sees only that region and copies the committed answer verbatim in
constrained JSON. It may copy an exact boxed candidate, an exact heuristic
candidate, or an otherwise exact contiguous region substring. It no longer
selects candidate or lexer-boundary IDs.

For this existing 6,000-response corpus only, v3 imports the 5,991 saved valid
first-call adjudications after checking the exact v2 policy, response hashes,
and region catalog. It reruns the new second call for all 2,196 responses that
were adjudicated definitive; no v2 extraction is reused. This is a one-off
stage reuse, not a different future scoring rule. Future v3 evaluations run
both calls normally. The nine v2 rows that exhausted first-call adjudication
remain unresolved instead of receiving a fourth attempt.

Identifier selection proved to be an avoidably difficult extraction task,
especially for lexer-boundary IDs. Exact boxed and heuristic spans were often
usable, so v3 retains their text as guidance while asking Llama to copy the
answer naturally. JSON-schema decoding constrains the output shape; a
mechanical validator separately requires the returned text to be an eligible
exact span of the selected region. After three failed attempts the row remains
unresolved rather than accepting generated or repaired text.

This scorer change is entirely post-hoc and does not rerun or alter any model
generation. Strict-v2 remains immutable. Strict-v3 writes a separate versioned
result, reports unresolved rows as a lower–upper range (all unresolved wrong
versus all unresolved right), and queues them for versioned review by a
stronger model reasoning over the trace or by a human.

See the [evaluation handoff](../../qwen2_5_7b_openr1_math220k_masked_sft_eval_handoff/README.md)
for the exact runtime, gate, scoring, cap-hit, and recovery policies.

The [provisional analysis](ANALYSIS.md) records the comparison table, cap-hit
sensitivity, observed failure modes, and manual Llama-gate audit that motivated
the v3 scorer change.
