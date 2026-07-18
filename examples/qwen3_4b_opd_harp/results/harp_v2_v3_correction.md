# Qwen3-4B HARP-500 scorer correction: V2 versus V3

Slurm report job `177946` rescored the stored outputs from evaluation job
`177135`. Both rows below use the exact same 500 greedy generations; only
answer extraction and verification changed.

| Scorer | Correct | Accuracy | Cap hits | Cap-hit accuracy | Reviews |
|---|---:|---:|---:|---:|---:|
| `harp_answer_v2` | 414/500 | 82.800% | 36 | 8.333% | n/a |
| `harp_answer_v3` | 418/500 | 83.600% | 36 | 8.333% | 0 |

Correction: **+4 answers, +0.800 percentage points**. There were six
correctness flips: V3 corrected five V2 false negatives and rejected one
cap-hit V2 false positive that had no eligible answer event.

- HARP source and clean-split SHA-256: `30a43db8e669d7a6cc8fbb8a93a84018d5440b110632372467183d707a30c5ab`
- Stored-generation SHA-256: `fbd1cdfff57a414bd14093755e39f806d311a6ed41ba51901d9186b9e60e6b74`
- V2 metrics SHA-256: `6f2a481ba2ae09c7fa0bb9d17516934319db88a334619e3cfbc7123a14194192`
- V3 predictions SHA-256: `57d5b6666bf9ed6821136bf1d5bf791804b846f143b69e73fda52488ee56ddbc`
- V3 metrics SHA-256: `2aac88ef124621fce306f3fa91ff3b8d450cd9dbe77895fab7f541d7ea3ad231`
- Reproducible report JSON SHA-256: `56c33981fbc3395876d3062cf97fcb43fd6836420acce0ae09d18c1ef0bf6ce1`

The strict V3 run used all 500 source rows, excluded none, and emitted no
`needs_review` decisions. The full flip-level audit is in
`harp_v2_v3_correction.json`.
