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

## Corrected SFT-Loaded 4-GPU OPD Retry

- Reason for retry: train job `157036` failed on `2026-07-01` after 2m58s,
  before rollout or actor training, with
  `/usr/bin/bash: line 98: OPD_INITIAL_LOAD_DIR: unbound variable`.
- Root cause: `OPD_INITIAL_LOAD_DIR` and `OPD_OPTIMIZER_CPU_OFFLOAD` were set
  in the host Slurm environment but were not forwarded by `container_exec.sh`
  through Apptainer `--cleanenv`.
- Progress preservation: no corrected OPD checkpoint, HF snapshot, rollout
  dataset state, or debug rollout file existed, so the retry restarts from the
  final SFT full checkpoint at
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/qwen3_8b_sft_25k_full_optim/iter_0000096`.
- Patch commit: `0b952db` (`Forward corrected OPD load env into container`).
- Stale jobs canceled: `157037`, `157038`, `157039`.
- Independent cleanup job `157809` was left running.
- Submit time: `2026-07-01T20:30:20-07:00`.
- Dependency policy: replacement train has no dependency; downstream jobs use
  `afterok`.
- OPD train job: `157935`, `slime-qwen3-opd1k-sft4g`,
  `gpu:h200:4`, `cpus-per-task=32`, `time=18:00:00`,
  `Dependency=(null)`.
- OPD final eval job: `157936`, `slime-qwen3-opd1k-sft4g-eval`,
  `gpu:h200:4`, `cpus-per-task=32`, `time=05:00:00`,
  dependency `afterok:157935`.
- Base maybe-eval job: `157937`, `slime-qwen3-base-math500-maybe`,
  `gpu:h200:4`, `cpus-per-task=32`, `time=05:00:00`,
  dependency `afterok:157936`.
- Final report job: `157938`, `slime-qwen3-final-report-sft4g`,
  `gpu:h200:1`, `cpus-per-task=4`, `time=00:30:00`,
  dependency `afterok:157937`.
- Mail for all replacement jobs: `MailUser=suryadv@cs.washington.edu`,
  `MailType=END,FAIL`.
- OPD initial load:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/qwen3_8b_sft_25k_full_optim`
- OPD optimizer CPU offload: `1`.
- OPD save dir:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/qwen3_8b_sft_25k_opd_1k_32k_sft_offload4_full_optim`
- OPD eval output:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/math500_eval_opd_1k_32k_sft_offload4_final`
- Base eval output:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/math500_eval_base_25k_opd_1k_32k_sft_offload4`
- Combined final report output:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/math500_eval_combined_25k_opd_1k_32k_sft_offload4`
- Expected runtime validation: train log shows
  `Starting OPD from full SFT optimizer checkpoint`, validates Megatron load dir
  at iteration `96`, prints `OPD optimizer CPU offload: 1`, reaches rollout id
  `0`, and does not print `OPD_INITIAL_LOAD_DIR: unbound variable` or base
  fallback checkpoint messages.

## Corrected SFT-Weights 4-GPU OPD Retry

- Reason for retry: train job `157935` failed on `2026-07-01` after 4m45s,
  before rollout or actor training, with a Megatron distributed optimizer
  checkpoint topology error:
  `TP, PP mismatch after resume ((1, 1) vs (2, 1) from checkpoint)`.
- Root cause: the final SFT full optimizer checkpoint was saved with tensor
  parallel `2`, but the corrected 4-GPU OPD actor uses tensor parallel `1` and
  context parallel `2`. Megatron cannot optimizer-resume that SFT checkpoint
  across the TP mismatch because it was not saved with fully-parallel checkpoint
  support.
- Progress preservation: no corrected OPD checkpoint, HF snapshot, rollout
  dataset state, or debug rollout file existed. The retry initializes from the
  final SFT HF weights snapshot and creates fresh OPD optimizer state; after
  OPD writes its own full checkpoint, continuation should resume from
  `$OPD_SAVE_DIR`, not from the SFT checkpoint.
- Patch commit: `3e9c216` (`Fix corrected OPD HF initialization`), pushed to
  `origin/opd-reproduction`.
- Stale jobs canceled: `157936`, `157937`, `157938`.
- Dependency policy: replacement train has no dependency; downstream jobs use
  `afterok`.
- Submit time: `2026-07-01T23:29:56-07:00`.
- OPD train job: `158041`, `slime-qwen3-opd1k-sft4g`,
  `gpu:h200:4`, `cpus-per-task=32`, `time=18:00:00`,
  `Dependency=(null)`, pending for resources at submission.
- OPD final eval job: `158042`, `slime-qwen3-opd1k-sft4g-eval`,
  `gpu:h200:4`, `cpus-per-task=32`, `time=05:00:00`,
  dependency `afterok:158041`.
- Base maybe-eval job: `158043`, `slime-qwen3-base-math500-maybe`,
  `gpu:h200:4`, `cpus-per-task=32`, `time=05:00:00`,
  dependency `afterok:158042`.
- Final report job: `158044`, `slime-qwen3-final-report-sft4g`,
  `gpu:h200:1`, `cpus-per-task=4`, `time=00:30:00`,
  dependency `afterok:158043`.
- Mail for all replacement jobs: `MailUser=suryadv@cs.washington.edu`,
  `MailType=END,FAIL`.
- OPD initial load mode: `hf`.
- OPD initial load:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/qwen3_8b_sft_25k_eval_snapshots/iter_0000096`
- SFT full checkpoint retained for provenance:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/qwen3_8b_sft_25k_full_optim/iter_0000096`
- OPD optimizer CPU offload: `1`.
- OPD actor/rollout/teacher/ray GPUs: `2/1/1/3`; actor `TP=1`, `CP=2`;
  `OPD_MAX_RESPONSE_LEN=31744`; `OPD_SEQ_LENGTH=32768`;
  `OPD_MAX_TOKENS_PER_GPU=16384`.
- OPD save dir:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/qwen3_8b_sft_25k_opd_1k_32k_sft_offload4_full_optim`
- OPD eval output:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/math500_eval_opd_1k_32k_sft_offload4_final`
- Base eval output:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/math500_eval_base_25k_opd_1k_32k_sft_offload4`
- Combined final report output:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/math500_eval_combined_25k_opd_1k_32k_sft_offload4`
- Expected runtime validation: train log shows `OPD_INITIAL_LOAD_MODE=hf`,
  validates the SFT HF snapshot, logs
  `Load checkpoint from HuggingFace model into Megatron`, does not print the
  TP/PP mismatch error, does not fall back to base, teacher `/health_generate`
  passes, and OPD reaches rollout id `0`.
