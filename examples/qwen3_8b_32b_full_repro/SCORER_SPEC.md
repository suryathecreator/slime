# MATH-500 scorer specification

`math500_scorer_v1.py` preserves the generation-time
`math500_event_scorer_v1`. `math500_scorer_v2.py` preserves the audited
`math500_event_scorer_v2`. The active `math500_scorer.py` implements
`math500_strict_boxed_scorer_v3`. All three use pinned `math_verify==0.9.0`,
and no saved generation is modified or continued during rescoring.

## Extraction

V3 always searches the entire assistant response, regardless of think tags.
Think-tag shape is recorded only as a formatting diagnostic. The scorer finds
complete balanced `\boxed{...}` expressions, groups adjacent boxes as one
multipart answer, and selects the chronologically last complete boxed group.

The selection is purely structural. Later reasoning, doubt, retractions,
examples, assertions, conclusions, and prose do not alter it. Unboxed answers,
standalone final lines, arbitrary terminal numbers, empty boxes, and malformed
boxes are never fallbacks. If the selected box cannot be parsed or verified,
the row is unscorable and remains incorrect in the fixed 500-row denominator.

## Cap policy and diagnostics

Every generation marked as a cap hit is officially incorrect. V3 still runs
the same whole-response last-box extraction and typed verifier and stores that
counterfactual result as a diagnostic. The diagnostic never changes official
correctness.

Reports include cap hits marked wrong, cap-hit diagnostic correctness,
non-cap unscorable rows, official decision reasons, and these think-tag
formatting categories: complete, none, open without close, close without open,
and malformed or repeated.

Think-tag counts are not reasoning-quality measurements. The model is still
learning consistent formatting while its reasoning improves, and substantially
more instruction-tuning data is likely needed to teach stable tag behavior.

## Typed verification and publication

Scalar and ordered expressions use the pinned mathematical verifier.
Collections split only at top-level separators, require exact cardinality, and
match components without order. Ordered tuples, intervals, matrices, equations,
fractions, radicals, complex values, text answers, degrees, `\pm`, and adjacent
multipart boxes retain their appropriate structure.

Every rescored row records immutable source and response hashes, think-tag
statistics, all complete boxes and groups, the selected span and candidate,
typed-verifier evidence, official decision reason, cap diagnostic, failures,
and timeouts. Draft output cannot publish. Final publication additionally
requires exact 2,000-row validation, a locked aggregate and row-decision digest,
zero unresolved reviews, and matching compact artifact hashes.
