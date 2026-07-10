# Cleaned SFT Loss-Mask Incident

Recorded: 2026-07-08 PDT

The first cleaned OpenThoughts SFT job, Slurm `165036`, completed 25k rows but
used Slime's default `loss_mask_type=qwen`. With the Qwen3 tokenizer this path
rendered the assistant message mostly as the post-`</think>` answer text,
dropping most of the intended thinking trace from the trainable target.

This checkpoint is diagnostic only and must not be used as the intended
cleaned SFT -> OPD starting point.

## Affected SFT Runs

| Slurm Job | Run | Loss Mask | Status |
| --- | --- | --- | --- |
| `151633` | Original non-cleaned SFT | `qwen` | Diagnostic for this issue; it did not use the Qwen3 thinking-trace mask. |
| `165036` | First cleaned SFT | `qwen` | Invalid for the intended cleaned full-thinking-trace experiment. |
| `165695` | Corrected cleaned SFT rerun | `qwen3` | Intended rerun; preflight verified full `<think>...</think>` targets. |

The `qwen`-masked runs can still be useful as historical/diagnostic artifacts,
but they should not be interpreted as Qwen3 full-thinking-trace SFT.

## Evidence

For cleaned SFT JSONL row `0`:

| Path | Total Tokens | Train/Loss Tokens | Notes |
| --- | ---: | ---: | --- |
| Raw assistant text | 15,299 | 15,299 | Complete `<think>...</think>` trace exists in the JSONL. |
| `loss_mask_type=qwen` | 730 | 668 | This was used by job `165036`; it stripped most thinking content. |
| `loss_mask_type=qwen3` | 15,364 | 15,302 | This preserves the full Qwen3 thinking trace and is the corrected setting. |

The completed base vLLM eval is still valid and reused:

| Job | Stage | Accuracy | Parse Failure Rate | Cap Hit Rate |
| --- | --- | ---: | ---: | ---: |
| `165035` | base | 0.612 | 0.176 | 0.040 |

## Preservation And Correction

- Preserved valid artifacts: cleaned data, model conversion, DP optimizer smoke,
  and base eval `165035`.
- Preserved bad SFT artifacts from `165036` as diagnostics only.
- Canceled invalid downstream jobs `165037`-`165042` so OPD cannot start from
  the bad SFT checkpoint.
- Corrected rerun uses `SFT_LOSS_MASK_TYPE=qwen3`.
- Corrected outputs use a `qwen3mask` tag in SFT, OPD, eval, report, and
  checkpoint-report directories so they cannot accidentally load `165036`.

## Corrected Validation

The corrected SFT launcher runs a preflight check before training:

- reads the cleaned SFT JSONL,
- tokenizes raw assistant text,
- builds the train target using the configured loss-mask type,
- verifies the loss target covers at least 80% of raw assistant tokens,
- requires decoded train targets to contain both `<think>` and `</think>`,
- prints compact decoded snippets for inspection.
