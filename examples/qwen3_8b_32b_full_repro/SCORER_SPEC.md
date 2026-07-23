# MATH-500 scorer specification

`math500_scorer_v1.py` preserves the exact scorer used at generation time as
`math500_event_scorer_v1`. It is retained unchanged for provenance.
`math500_scorer.py` is the audited `math500_event_scorer_v2`. Both use the
pinned `math_verify==0.9.0`; V2 changes extraction, event selection, and
collection verification, not the saved generations.

## Eligible answer region

When `</think>` is present, only the final post-`</think>` region is eligible.
For a cap hit without a closed think region, V2 scans the unfinished trace only
for balanced boxes and explicit final-answer assertions. Natural conclusions
and standalone-line fallbacks are excluded from cap-hit scoring.

For natural stops, the chronologically last complete strong answer event wins.
Adjacent boxes are grouped as one multipart event. A concise standalone final
line is allowed only when no complete box or explicit assertion exists.
Structural display delimiters, empty boxes, malformed boxes, incidental
numbers, and hypothetical/example boxes are not answers.

## Replacement and cap policy

A later distinct complete event replaces the active answer. An equivalent
repeat is a reaffirmation. Doubt and checking language alone do not erase a
complete answer. Only an explicit, local, unreplaced rejection makes the active
answer unresolved and incorrect.

At a generation cap, a terminal strict prefix of the previous complete scalar
or multipart answer is recorded as a truncated repetition and suppressed. This
is why a final `5` does not replace a prior complete `59`, and why a partial
repetition does not replace a complete multipart answer. V2 never uses an
arbitrary last-number fallback.

## Verification and publication

Scalar and ordered expressions are passed to the pinned mathematical verifier.
Collections are split only at top-level separators, compared component by
component with exact cardinality, and matched without order. A failed parse is
never equivalent to another failed parse. Standalone natural-language answer
sentences are verified as scalar evidence rather than being split at prose
commas.

Every rescored row records its immutable source hash, response hash, selected
candidate, all answer-event spans and candidates, grouping, invalidations,
replacement/reaffirmation/retraction decisions, typed-verifier evidence,
finish/cap/think state, parse failures, and timeouts. The rescore report refuses
publication unless every source artifact validates, all 2,000 rows are
decided, review count is zero, and the audited aggregate transition gate
passes.
