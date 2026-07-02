# Tillicum Qwen3 OPD Submission Metadata

Recorded: 2026-07-01 17:28 PDT

## Accidental Base -> OPD Cleanup

- Purpose: complete a valid final MATH-500 report for the accidental base -> OPD
  `1k_32k` run. The earlier eval reached 491/500 samples but timed out before
  Slime wrote `debug_eval_0.pt`, so this cleanup reruns the full final eval
  stage rather than appending partial responses.
- Code commit submitted: `d25a6f4` (`Add 2GPU base OPD cleanup eval`)
- Job id: `157809`
- Final state: `FAILED`, exit `2:0`, elapsed `03:20:56`, from
  `2026-07-01T20:06:33-07:00` to `2026-07-01T23:27:29-07:00`.
- Failure point: the accidental OPD eval completed and wrote `debug_eval_0.pt`
  plus `summary.json`, then the nested eval wrapper returned
  `container_exec.sh: line 250: unexpected EOF while looking for matching '"'`
  before base eval or combined report generation.
- Preserved OPD eval result: `accuracy=0.0`, `accuracy_on_parseable=0.0`,
  `parse_failure_rate=0.998`, `cap_hit_rate=0.882`,
  `avg_generated_tokens=30137.598`, `n=500`.
- Salvage decision: preserve the complete OPD final eval artifact, rerun only
  missing base eval/report pieces, and skip OPD eval on replacement unless the
  OPD summary is missing.
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

## Accidental Base -> OPD Cleanup Salvage Retry

- Purpose: finish the accidental base -> OPD report while preserving the
  complete OPD eval artifact from failed job `157809`.
- Patch commit: `a99c041` (`Fix base OPD cleanup salvage`), pushed to
  `origin/opd-reproduction`.
- Key patch behavior: `06_eval_math500_greedy_1x.sbatch` now invokes a tracked
  inner helper instead of a large inline `bash -lc` payload, and
  `09_cleanup_base_opd_2gpu.sbatch` skips completed summaries and treats a
  nested nonzero eval as salvaged if the expected `summary.json` exists.
- Job id: `158065`
- Submit time: `2026-07-01T23:44:26-07:00`.
- Dependency policy: none. This job is independent of corrected SFT -> OPD jobs
  `158041`-`158044`.
- Slurm request: `gpu:h200:2`, `cpus-per-task=16`, `time=18:00:00`,
  account `raivn`, partition `gpu-h200`, QOS `normal`.
- Initial scheduler state: `PENDING`, reason `Priority`, `Dependency=(null)`.
- Mail: `MailUser=suryadv@cs.washington.edu`, `MailType=END,FAIL`.
- Expected runtime behavior: skip existing
  `math500_eval_opd_1k_32k_final/opd_001024/summary.json`, run fresh base eval
  into `math500_eval_base_25k_opd_1k_32k/base`, then write the base -> OPD-only
  combined report under `math500_eval_combined_25k_opd_1k_32k`.
- Runtime validation target: no `unexpected EOF while looking for matching '"'`;
  combined report includes only base and accidental OPD points with SFT omitted.

## Accidental Base -> OPD Cleanup Final Result

- Final state: job `158065` completed successfully on `2026-07-02`.
- Patch commit used by job: `a99c041` (`Fix base OPD cleanup salvage`).
- Base summary:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/math500_eval_base_25k_opd_1k_32k/base/summary.json`
- Accidental OPD summary:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/math500_eval_opd_1k_32k_final/opd_001024/summary.json`
- Combined report:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/math500_eval_combined_25k_opd_1k_32k`
- Base result: `accuracy=0.638`, `accuracy_on_parseable=0.7595238095238095`,
  `parse_failure_rate=0.16`, `cap_hit_rate=0.036`,
  `avg_generated_tokens=1686.402`, `n=500`.
- Accidental base -> OPD result: `accuracy=0.0`,
  `accuracy_on_parseable=0.0`, `parse_failure_rate=0.998`,
  `cap_hit_rate=0.882`, `avg_generated_tokens=30137.598`, `n=500`.
- Interpretation note: the base eval and accidental OPD rollout used the same
  base HF revision, but the OPD run used the broken/accidental loader path and
  should not be treated as a clean planned base -> OPD baseline.

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

## Corrected SFT-Weights 4-GPU OPD Retry With Sanity Guard

- Reason for retry: train job `158041` loaded the final SFT HF snapshot and
  reached rollout `0`, then failed during the first actor train step with CUDA
  OOM. The failing allocation was `192.00 MiB`; GPU 1 had only `68.19 MiB`
  free and `139.63 GiB` in use.
- Progress preservation: no corrected OPD checkpoint, HF snapshot, or rollout
  dataset state existed. The diagnostic rollout file was preserved as
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/opd_1k_32k_sft_offload4_rollout_logs/rollout_0.failed_158041.pt`.
- Rollout sanity from the preserved artifact: `n=128`,
  `avg_response_tokens=1325.5234375`, `max_response_tokens=17370`,
  `cap_hit_rate=0.0`, `completed_rate=1.0`,
  `final_answer_rate=0.6953125`.
- Patch commit: `4e8f653` (`Guard corrected OPD against rollout collapse`),
  pushed to `origin/opd-reproduction`.
- Key patch behavior:
  - Adds OPD rollout sanity summaries before actor updates.
  - Writes JSON reports under `$OPD_SANITY_REPORT_DIR`.
  - Fails fast on extreme cap-hit / no-final-answer collapse signals.
  - Lowers corrected 4-GPU actor dynamic train packing to
    `OPD_MAX_TOKENS_PER_GPU=8192` while keeping
    `OPD_MAX_RESPONSE_LEN=31744`.
  - Repairs the current pycache-only Apptainer sandbox via
    `00_pull_or_load_container.sh`; final dry check passed
    `RUN_CONTAINER_CHECKS=1`.
- Stale jobs canceled: `158042`, `158043`, `158044`.
- Submit time: `2026-07-02T14:41:54-07:00`.
- Dependency policy: replacement train has no dependency; downstream jobs use
  `afterok`.
- OPD train job: `158665`, `slime-qwen3-opd1k-sft4g`,
  `gpu:h200:4`, `time=18:00:00`, `Dependency=(null)`, pending for priority
  at submission.
- OPD final eval job: `158666`, `slime-qwen3-opd1k-sft4g-eval`,
  `gpu:h200:4`, `time=05:00:00`, dependency `afterok:158665`.
- Base maybe-eval job: `158667`, `slime-qwen3-base-math500-maybe`,
  `gpu:h200:4`, `time=05:00:00`, dependency `afterok:158666`.
- Final report job: `158668`, `slime-qwen3-final-report-sft4g`,
  `gpu:h200:1`, `time=00:30:00`, dependency `afterok:158667`.
- Mail for all replacement jobs: `MailUser=suryadv@cs.washington.edu`,
  `MailType=END,FAIL`.
- OPD initial load mode: `hf`.
- OPD initial load:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/qwen3_8b_sft_25k_eval_snapshots/iter_0000096`
- OPD actor/rollout/teacher/ray GPUs: `2/1/1/3`; actor `TP=1`, `CP=2`;
  `OPD_MAX_RESPONSE_LEN=31744`; `OPD_SEQ_LENGTH=32768`;
  `OPD_MAX_TOKENS_PER_GPU=8192`; optimizer CPU offload enabled.
- OPD sanity guard:
  `OPD_SANITY_CHECK_ENABLED=1`, `OPD_SANITY_MAX_CAP_HIT_RATE=0.50`,
  `OPD_SANITY_MAX_AVG_RESPONSE_TOKENS=16000`,
  `OPD_SANITY_MIN_FINAL_ANSWER_RATE=0.02`.
- OPD save dir:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/qwen3_8b_sft_25k_opd_1k_32k_sft_offload4_full_optim`
- OPD eval output:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/math500_eval_opd_1k_32k_sft_offload4_final`
- Base eval output:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/math500_eval_base_25k_opd_1k_32k_sft_offload4`
- Combined final report output:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/math500_eval_combined_25k_opd_1k_32k_sft_offload4`
- Expected runtime validation: train log shows `OPD_INITIAL_LOAD_MODE=hf`,
  the SFT HF snapshot load, no base fallback, no TP mismatch, OPD sanity JSON
  for rollout `0`, and actor training proceeds past the previous first-step
  OOM.

## Corrected SFT-Weights 4-GPU OPD Colocate Retry

- Result snapshot commit: `f7f79b5` (`Record accidental base OPD salvage
  results`), pushed to `origin/opd-reproduction`.
- Tracked salvage result files:
  `examples/qwen3_8b_opd_tillicum/results/accidental_base_opd_salvage_summary.json`
  and
  `examples/qwen3_8b_opd_tillicum/results/accidental_base_opd_salvage_summary.csv`.
- Superseded retry: non-colocated jobs `158665`, `158666`, `158667`, and
  `158668` were canceled before start. That canceled chain used the separated
  `2/1/1` layout and is not the current corrected run.
- Patch commit: `7852d74` (`Switch corrected OPD to colocate4`), pushed to
  `origin/opd-reproduction`.
- Key patch behavior:
  - New corrected default run label: `1k_32k_sft_colocate4`.
  - Actor and student rollout are colocated on Ray GPUs `0,1,2`; teacher is on
    physical GPU `3`.
  - Actor/rollout/teacher GPU counts: `3/3/1`.
  - Actor `TP=1`, `CP=3`, `OPD_SEQ_LENGTH=32766`,
    `OPD_MAX_RESPONSE_LEN=31744`, `OPD_MAX_TOKENS_PER_GPU=11264`.
  - Enables `--colocate`, `--offload-train`, `--offload-rollout`,
    optimizer CPU offload, and `--recompute-loss-function`.
  - Keeps the OPD sanity guard from commit `4e8f653`.
  - Removes the `8192` train-packing default from the current corrected path;
    `8192` remains only as a documented fallback if the colocated CP=3 retry
    still OOMs.
- Container repair/validation: `00_pull_or_load_container.sh` was rerun before
  submission. It materialized `0` additional stdlib `.pyc` files and validated
  both direct Apptainer imports and `container_exec.sh` imports for
  `encodings, os, site, sglang, torch`.
- Static validation before submission:
  - `bash -n` on edited shell/sbatch scripts.
  - `python3 -m py_compile` on touched Python helpers.
  - `git diff --check`.
  - Full colocate4 dry check with `RUN_CONTAINER_CHECKS=1` passed.
- Submit time: `2026-07-02T15:29:12-07:00`.
- Dependency policy: replacement train has no dependency; downstream jobs use
  `afterok`.
- OPD train job: `158697`, `slime-qwen3-opd1k-sft4g`,
  `gpu:h200:4`, `time=18:00:00`, `Dependency=(null)`, pending for resources
  at submission.
- OPD final eval job: `158698`, `slime-qwen3-opd1k-sft4g-eval`,
  `gpu:h200:4`, `time=05:00:00`, dependency `afterok:158697`.
- Base maybe-eval job: `158699`, `slime-qwen3-base-math500-maybe`,
  `gpu:h200:4`, `time=05:00:00`, dependency `afterok:158698`.
- Final report job: `158700`, `slime-qwen3-final-report-sft4g`,
  `gpu:h200:1`, `time=00:30:00`, dependency `afterok:158699`.
- Mail for all replacement jobs: `MailUser=suryadv@cs.washington.edu`,
  `MailType=END,FAIL`.
- OPD initial load mode: `hf`.
- OPD initial load:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/qwen3_8b_sft_25k_eval_snapshots/iter_0000096`
- OPD save dir:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/qwen3_8b_sft_25k_opd_1k_32k_sft_colocate4_full_optim`
- OPD eval output:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/math500_eval_opd_1k_32k_sft_colocate4_final`
- Base eval output:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/math500_eval_base_25k_opd_1k_32k_sft_colocate4`
- Base reuse source:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/math500_eval_base_25k_opd_1k_32k`
- Combined final report output:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/math500_eval_combined_25k_opd_1k_32k_sft_colocate4`
- Expected runtime validation: train log shows `OPD_INITIAL_LOAD_MODE=hf`,
  SFT HF snapshot as the Megatron HF load path, `--colocate`,
  `--offload-train`, `--offload-rollout`, actor/rollout/teacher `3/3/1`,
  `TP=1`, `CP=3`, rollout `0` reaches and completes actor train step `0`
  without CUDA OOM, `train_rollout_logprob_abs_diff` is small rather than the
  old pathological value, and final `iter_0000007` checkpoint/HF snapshot plus
  trained-data manifest are written.
