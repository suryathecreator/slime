# MATH-500 scorer specification

`math500_scorer.py` is immutable scorer version `math500_event_scorer_v1` and
uses `math_verify==0.9.0` for mathematical equivalence.

If a final `</think>` exists, only the text after that close is eligible. If no
close exists because generation is capped or unfinished, the complete response
is scanned. Ordered answer events include balanced boxes, explicit final-answer
assertions, conclusion statements, and strongly indicated final math lines.
Incidental numbers are ignored.

The state machine records corrections, replacements, reaffirmations, doubt, and
retractions. A later confident answer supersedes an earlier answer. A doubtful
event does not become official; selection walks backward to the most recent
confident candidate that remains unretracted and unsuperseded. If none exists,
the scorer returns no answer.

Every row records the official candidate, region and rule, complete event
history, invalidation flags, boxed and last-confident diagnostics,
answer-anywhere diagnostic, parse and verifier results, finish reason, observed
stop token, cap status, think closure, replacement count, exceptions/timeouts,
and hashes. Parse failures and timeouts score incorrect and are never guessed.
