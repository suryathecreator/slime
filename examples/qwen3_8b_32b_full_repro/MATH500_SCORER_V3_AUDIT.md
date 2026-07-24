# MATH-500 strict boxed scorer V3 saved-generation audit

All 2,000 immutable generations were rescored without inference or continuation.
Cap hits are officially wrong; their strict-box result is retained only as a diagnostic.

| Stage | V1 | V2 | V3 strict | Cap wrong | Cap diagnostic correct | Non-cap unscorable |
|---|---:|---:|---:|---:|---:|---:|
| qwen3_8b_base | 235/500 | 373/500 | 371/500 (74.20%) | 11 | 0 | 5 |
| qwen3_32b_teacher | 476/500 | 482/500 | 480/500 (96.00%) | 14 | 1 | 0 |
| sft_200000 | 304/500 | 434/500 | 403/500 (80.60%) | 87 | 3 | 0 |
| opd_100pct | 429/500 | 440/500 | 428/500 (85.60%) | 58 | 6 | 0 |

## Think-tag formatting diagnostics

| Stage | Complete | None | Open/no close | Close/no open | Malformed/repeated |
|---|---:|---:|---:|---:|---:|
| qwen3_8b_base | 0 | 500 | 0 | 0 | 0 |
| qwen3_32b_teacher | 487 | 0 | 13 | 0 | 0 |
| sft_200000 | 264 | 0 | 181 | 0 | 55 |
| opd_100pct | 0 | 135 | 0 | 359 | 6 |

The model is still learning consistent think-tag behavior while its reasoning is improving. These counts measure formatting, not reasoning quality; substantially more instruction-tuning data is likely needed to teach consistent formatting.

Think tags never affect extraction. Every row uses the whole response, and the scorer takes the chronologically last complete boxed group and does not interpret later reasoning, doubt, retraction, examples, or prose.

Review count: 0. Aggregate gate: passed.
