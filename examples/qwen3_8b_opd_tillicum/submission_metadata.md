# Tillicum Qwen3 OPD Submission Metadata

Recorded: 2026-07-01 17:28 PDT

## Accidental Base -> OPD Cleanup

- Purpose: complete a valid final MATH-500 report for the accidental base -> OPD
  `1k_32k` run. The earlier eval reached 491/500 samples but timed out before
  Slime wrote `debug_eval_0.pt`, so this cleanup reruns the full final eval
  stage rather than appending partial responses.
- Code commit submitted: `d25a6f4` (`Add 2GPU base OPD cleanup eval`)
- Job id: `157809`
- Submit time: `2026-07-01T17:28:20-07:00`
- Dependency policy: none. This job is independent of corrected SFT-loaded jobs
  `157036`-`157039`.
- Slurm request: `gpu:h200:2`, `cpus-per-task=16`, `time=18:00:00`,
  account `raivn`, partition `gpu-h200`, QOS `normal`.
- Mail: `MailUser=suryadv@cs.washington.edu`, `MailType=END,FAIL`.
- Cleanup eval settings: `EVAL_ROLLOUT_NUM_GPUS=2`,
  `EVAL_ROLLOUT_BATCH_SIZE=64`, `EVAL_SGLANG_SERVER_CONCURRENCY=6`,
  `EVAL_TENSOR_MODEL_PARALLEL_SIZE=1`,
  `EVAL_CONTEXT_PARALLEL_SIZE=1`.
- Run label: `1k_32k`.
- OPD checkpoint evaluated:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/qwen3_8b_sft_25k_opd_1k_32k_eval_snapshots/iter_0000007`
- OPD eval output:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/math500_eval_opd_1k_32k_final/opd_001024`
- Base eval output:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/math500_eval_base_25k_opd_1k_32k/base`
- Combined base -> OPD report output:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/math500_eval_combined_25k_opd_1k_32k`
- Expected runtime validation:
  `debug_eval_0.pt` and `summary.json` are written for `opd_001024`;
  `base/summary.json` is written if missing; the combined report includes only
  base and accidental OPD points, labels OPD at `1024` samples, and states that
  this is base -> OPD, not SFT -> OPD.
- OOM fallback policy: if this cleanup OOMs before a complete debug file is
  produced, resubmit the same job with `EVAL_SGLANG_SERVER_CONCURRENCY=4` and
  record the replacement job here.

## Corrected SFT-Loaded 4-GPU OPD

- Purpose: corrected 1k/32k OPD run that loads the final SFT full Megatron
  checkpoint instead of the HF snapshot/base fallback path.
- Code commit at submission: `7c138b7` (`Fix SFT-loaded 4GPU OPD retry`).
  Later cleanup/report-compatible commit `d25a6f4` is also pushed to
  `origin/opd-reproduction`.
- Submit time: `2026-07-01T02:16:35-07:00`.
- Initial dependency policy: OPD train submitted with
  `afterany:156277:156278:156279`; after the stale old jobs were cleared, job
  `157036` has `Dependency=(null)`. Downstream corrected jobs remain `afterok`
  dependencies.
- OPD train job: `157036`, `slime-qwen3-opd1k-sft4g`,
  `gpu:h200:4`, `cpus-per-task=32`, `time=18:00:00`.
- OPD final eval job: `157037`, `slime-qwen3-opd1k-sft4g-eval`,
  `gpu:h200:4`, `cpus-per-task=32`, `time=05:00:00`,
  dependency `afterok:157036`.
- Base maybe-eval job: `157038`, `slime-qwen3-base-math500-maybe`,
  `gpu:h200:4`, `cpus-per-task=32`, `time=05:00:00`,
  dependency `afterok:157037`.
- Final report job: `157039`, `slime-qwen3-final-report-sft4g`,
  `gpu:h200:1`, `cpus-per-task=4`, `time=00:30:00`,
  dependency `afterok:157038`.
- Mail for all four jobs: `MailUser=suryadv@cs.washington.edu`,
  `MailType=END,FAIL`.
- OPD save dir:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/qwen3_8b_sft_25k_opd_1k_32k_sft_offload4_full_optim`
- OPD eval output:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/math500_eval_opd_1k_32k_sft_offload4_final`
- Base eval output:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/math500_eval_base_25k_opd_1k_32k_sft_offload4`
- Combined final report output:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/math500_eval_combined_25k_opd_1k_32k_sft_offload4`
- Corrected eval settings: `EVAL_ROLLOUT_NUM_GPUS=4`,
  `EVAL_ROLLOUT_BATCH_SIZE=128`,
  `EVAL_SGLANG_SERVER_CONCURRENCY=4`.
- Expected runtime validation: OPD train log shows SFT full checkpoint load at
  iteration `96` and no base fallback; actor/rollout/teacher split is `2/1/1`;
  actor `TP=1`, `CP=2`; eval log shows 4 one-GPU SGLang engines with
  concurrency 4; final report compares base, final SFT, and corrected final OPD.
