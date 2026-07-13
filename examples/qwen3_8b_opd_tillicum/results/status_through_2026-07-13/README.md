# Experiment Status Through 2026-07-13

This directory freezes the valid eval summaries and incomplete/invalid run
status before switching the data source from OpenThoughts3 to
OpenR1-Math-220k. The source summaries are copied verbatim under `summaries/`;
`metrics.csv` normalizes their main fields.

## Valid Full MATH-500

| Experiment | Samples | Accuracy | Accuracy on parseable | Parse failure | Cap hit | Avg generated tokens |
|---|---:|---:|---:|---:|---:|---:|
| Base, corrected vLLM contract (`165035`) | 500 | 0.612 | 0.7427 | 0.176 | 0.040 | 1880.558 |
| Base, earlier SGLang contract | 500 | 0.638 | 0.7595 | 0.160 | 0.036 | 1686.402 |
| Original OpenThoughts SFT 5,120 | 500 | 0.664 | 0.8949 | 0.258 | 0.318 | 10543.942 |
| Original OpenThoughts SFT 10,240 | 500 | 0.734 | 0.7646 | 0.040 | 0.534 | 17383.034 |
| Original OpenThoughts SFT 15,360 | 500 | 0.746 | 0.7803 | 0.044 | 0.694 | 22238.692 |
| Original OpenThoughts SFT 20,480 | 500 | 0.730 | 0.7783 | 0.062 | 0.716 | 22928.606 |
| Original OpenThoughts SFT 24,832 | 500 | 0.750 | 0.7812 | 0.040 | 0.576 | 18574.302 |
| Corrected SFT-loaded OPD 1,024 | 500 | 0.724 | 0.7686 | 0.058 | 0.832 | 26567.282 |

The two base rows used different generation backends/contracts and are retained
as separate measurements, not averaged together.

## Val100 Continuation Diagnostics

| Cumulative OPD samples | Accuracy | Accuracy on parseable | Parse failure | Cap hit | Avg generated tokens |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.700 | 0.7527 | 0.070 | 0.840 | 26924.53 |
| 2,048 | 0.720 | 0.8182 | 0.120 | 0.980 | 31365.73 |
| 3,072 | 0.770 | 0.8280 | 0.070 | 0.980 | 31352.81 |
| 4,096 | 0.740 | 0.8043 | 0.080 | 1.000 | 31744.00 |

These are 100-example measurements and must not be presented as full
MATH-500 results.

## Invalid and Incomplete Runs

- The accidental base-to-OPD run scored `0.000` with parse failure `0.998` and
  cap hit `0.882`. It is preserved as a diagnostic only: SGLang rollout/ref
  weights and the trainable Megatron actor did not represent the same model.
- The first OpenThoughts cleaner counted 1,200,000 source rows, 743,814
  incomplete-think rows, 94,920 mixed-language rows, and 429,713 valid rows.
- SFT job `165036` used the wrong `qwen` loss mask and is diagnostic only.
- Corrected Qwen3-mask SFT job `165695` trained successfully, but its full
  MATH-500 eval never completed. Job `167285` left schema-2 chunks covering
  128 examples; no 500-example accuracy is claimed.
- Strict-English setup `168162` completed. Cleaner `168163` timed out after
  390,000 rows and falsely classified all 390,000 as non-English because the
  Lingua microspan rule rejected tiny low-confidence spans. It produced zero
  valid rows. Dispatcher `168164` was canceled after becoming
  `DependencyNeverSatisfied`. All shards and scripts remain on scratch/disk.

## Direction Change

The OpenThoughts experiments showed useful math gains, but low-data SFT also
increased cap hits, redundant reasoning, and inconsistent termination/formatting.
Strict cleanup is expensive, and a small cleaned subset may not generalize
without substantially more SFT data. We are therefore preserving this work and
moving the next isolated reproduction to the more consistently formatted,
verified OpenR1-Math-220k traces. Nothing from the previous paths is deleted.
