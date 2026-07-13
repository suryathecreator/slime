# Tillicum Qwen3 OPD Results

- [Strict-English v2 experiment](strict_en_v2_experiment.md): stricter multilingual/noise filtering, count gates, a dynamic 45k/5k split, and old-job cleanup.
- [Status through 2026-07-13](status_through_2026-07-13/README.md): frozen full-eval summaries, val100 diagnostics, invalid/incomplete run status, and the OpenR1 direction change.

## Accidental Base -> OPD Salvage

These files record the completed salvage eval for the accidental `1k_32k`
base -> OPD run:

- `accidental_base_opd_salvage_summary.json`
- `accidental_base_opd_salvage_summary.csv`

This result is diagnostic only. The run used the old broken loader/fallback
path and should not be interpreted as the corrected SFT -> OPD experiment.

| Stage | Accuracy | Accuracy On Parseable | Parse Failure | Cap Hit |
|---|---:|---:|---:|---:|
| Base | 0.638 | 0.7595 | 0.160 | 0.036 |
| Accidental OPD 1024 | 0.000 | 0.0000 | 0.998 | 0.882 |
