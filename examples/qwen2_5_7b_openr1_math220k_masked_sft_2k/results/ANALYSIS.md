# Provisional MATH-500 analysis

Snapshot date: 2026-08-01.

This report summarizes the native results and the
`math500_llama33_definitive_strict_v2` snapshots currently in this directory.
The native figures are final for the native evaluator. The strict figures are
provisional because the extraction stage will likely be replaced. A range
means unresolved rows are counted wrong at the lower end and right at the
upper end. Every rate below divides by the full 500-problem dataset, not only
the resolved subset.

## Provisional comparison

| Variant | Native | Strict range | Strict resolved | Definitive | Cap hits |
|---|---:|---:|---:|---:|---:|
| base_7b | 265/500 (53.0%) | 304–307/500 (60.8–61.4%) | 497/500 | 453/500 | 11/500 (2.2%) |
| correct_only | 123/500 (24.6%) | 137–139/500 (27.4–27.8%) | 498/500 | 175/500 | 30/500 (6.0%) |
| inverse_tau_0p05 | 76/500 (15.2%) | 87–89/500 (17.4–17.8%) | 498/500 | 113/500 | 37/500 (7.4%) |
| inverse_tau_0p20 | 94/500 (18.8%) | 107–109/500 (21.4–21.8%) | 498/500 | 139/500 | 30/500 (6.0%) |
| margin_mask | 80/500 (16.0%) | 91–93/500 (18.2–18.6%) | 498/500 | 113/500 | 39/500 (7.8%) |
| prob_ratio_mask | 219/500 (43.8%) | 255–285/500 (51.0–57.0%) | 470/500 | 405/500 | 341/500 (68.2%) |
| random_mask_25 | 87/500 (17.4%) | 101–102/500 (20.2–20.4%) | 499/500 | 129/500 | 31/500 (6.2%) |
| random_mask_50 | 71/500 (14.2%) | 82–84/500 (16.4–16.8%) | 498/500 | 104/500 | 39/500 (7.8%) |
| random_mask_70 | 57/500 (11.4%) | 70/500 (14.0%) | 500/500 | 96/500 | 35/500 (7.0%) |
| random_mask_80 | 60/500 (12.0%) | 73–77/500 (14.6–15.4%) | 496/500 | 99/500 | 65/500 (13.0%) |
| random_mask_90 | 119/500 (23.8%) | 135–153/500 (27.0–30.6%) | 482/500 | 193/500 | 192/500 (38.4%) |
| unmasked | 92/500 (18.4%) | 104–106/500 (20.8–21.2%) | 498/500 | 137/500 | 33/500 (6.6%) |

The strict scorer raises every model's absolute score relative to the native
evaluator, largely because it recognizes definitive answers in more forms.
It does not reverse the main result: the base model's 60.8% lower bound remains
above every trained checkpoint's upper bound. `prob_ratio_mask` is closest,
but its 68.2% cap-hit rate makes that comparison especially unstable.

The weak trained results are not explained by native parsing alone. Native
parse failures range from 4 to 42 among the trained models, while the loss
relative to the base is much larger. Most low-cap trained variants stop before
the token limit but are definitive on only 96 to 175 of 500 responses. This
points to a final-answer commitment or response-quality shift, rather than a
simple evaluator-parser failure.

The result also does not isolate wrong-trace supervision as the sole cause:
`correct_only`, which used 2,000 correct traces, remains far below the base.
The common factors across all eleven runs—full-parameter SFT on a small 2K
trace set, the shared training recipe, and the trace/output distribution—are
therefore stronger candidates. That is a hypothesis, not a causal finding.
A useful follow-up is to compare training examples and held-out generations
for answer closure, repetition, and style shift before attributing the drop to
any particular masking rule.

## Cap-hit sensitivity

“All cap hits wrong” removes credit from every length-capped response. “All
cap hits right” gives credit to every capped response. The rate is the resulting
correct count divided by 500. Ranges remain where non-cap or cap rows are still
unresolved.

| Variant | Verified-correct cap hits | All cap hits wrong | All cap hits right |
|---|---:|---:|---:|
| base_7b | 0/11 | 304–307/500 (60.8–61.4%) | 315–318/500 (63.0–63.6%) |
| correct_only | 0–1/30 | 137–138/500 (27.4–27.6%) | 167–168/500 (33.4–33.6%) |
| inverse_tau_0p05 | 0–2/37 | 87/500 (17.4%) | 124/500 (24.8%) |
| inverse_tau_0p20 | 0–1/30 | 107–108/500 (21.4–21.6%) | 137–138/500 (27.4–27.6%) |
| margin_mask | 0–2/39 | 91/500 (18.2%) | 130/500 (26.0%) |
| prob_ratio_mask | 164–192/341 | 91–93/500 (18.2–18.6%) | 432–434/500 (86.4–86.8%) |
| random_mask_25 | 0/31 | 101–102/500 (20.2–20.4%) | 132–133/500 (26.4–26.6%) |
| random_mask_50 | 0–1/39 | 82–83/500 (16.4–16.6%) | 121–122/500 (24.2–24.4%) |
| random_mask_70 | 0/35 | 70/500 (14.0%) | 105/500 (21.0%) |
| random_mask_80 | 2–6/65 | 71/500 (14.2%) | 136/500 (27.2%) |
| random_mask_90 | 66–83/192 | 69–70/500 (13.8–14.0%) | 261–262/500 (52.2–52.4%) |
| unmasked | 0–2/33 | 104/500 (20.8%) | 137/500 (27.4%) |

`prob_ratio_mask` and `random_mask_90` are qualitatively different from the
other runs. Their average generated lengths are 22,469 and 12,704 tokens,
respectively, versus 1,225 for the base and roughly 2,192–4,431 for the other
trained variants. Their apparent score depends heavily on how capped loops
are handled. Even granting every cap hit as correct does not let
`random_mask_90` reach the base lower bound. `prob_ratio_mask` cannot be ranked
confidently until its capped responses are extracted and verified reliably.

## Llama-gate audit

The audit below was computed from the 6,000 saved strict-v2 gate records and a
manual skim of representative generations:

- Llama produced a resolved gate record for 5,951/6,000 responses. There were
  49 final gate-unresolved rows: 40 at extraction and 9 at adjudication.
- Of the 5,951 resolved gate records, 2,156 were definitive. Their reason codes
  were 2,086 `single_committed_answer` and 70
  `single_committed_unboxed_answer`.
- The final boxing flag says 147 of those 2,156 definitive responses were
  unboxed, not 70. Of those 147, 124 were cap hits. This mismatch is one reason
  not to rely on the Llama-produced post-definitiveness fields.
- Candidate-ID extraction handled 2,135/2,156 resolved definitive rows.
  Boundary-ID extraction handled the remaining 21. Several boundary results
  are visibly not complete final answers—for example `fractions`, `\(a`, `*`,
  `:`, and an explanatory sentence. These are span-selection failures, not
  generated answer text, but they can still corrupt downstream verification.
- There were 111 attempt-level validation-failure records: 93 during
  extraction and 18 during adjudication. Retries recovered some cases, so 111
  is not the final unresolved count. The most common failure was selecting a
  boxed region that did not map to one complete boxed-content candidate.
- The strict verifier left another 19 rows unresolved after the gate, yielding
  68 unresolved strict-score rows in total and 5,932/6,000 scoring coverage.
- The current strict results prove at least 232 capped responses correct, with
  an upper bound of 290 while unresolved rows remain. A manual spot-check of 30
  cap hits that the current scorer credited found no obvious false-positive
  credit. That is reassuring but is only a qualitative sample.

The skim found ordinary boxed commitments and obvious no-answer responses
generally well classified. The concentrated weakness was Llama's extraction
and field consistency on long, repetitive, or capped responses. This supports
retaining Llama as the definitive-answer gate while removing it from answer
span selection.

## Planned scorer interpretation

The likely next metric will use only Llama's binary `definitive` decision. For
the 6,000 checks already run, it will reuse only that decision from the saved
gate records and ignore the Llama-selected region, span, extracted answer,
answer form, and boxing fields. Future gate requests will ask Llama only about
definitiveness.

For both existing and future responses, deterministic heuristics will extract
the final committed answer and boxing status from the immutable raw response.
The structural guards and symbolic verifier will then run on that extracted
answer. This scorer change is post-hoc and versionable: it can be applied
uniformly without changing or regenerating any model completion. The files in
this directory remain the exact native and strict-v2 snapshots obtained so
far, rather than being silently overwritten.
