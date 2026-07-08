# Tillicum Qwen3 OPD Submission Metadata

Recorded: 2026-07-01 17:28 PDT

## Planned Cleaned OpenThoughts SFT + OPD vLLM Chain

- Purpose:
  - Restart the experiment from cleaned OpenThoughts3 thinking traces because
    high cap-hit behavior suggested the earlier SFT data may have taught
    undesirable long-generation habits.
  - Preserve and push all current tracked result snapshots before submitting
    the new chain.
- Cleaning/sampling:
  - Remove rows without complete `<think>...</think>` assistant traces.
  - Remove mixed-language rows with the local script-count heuristic.
  - Deterministically shuffle valid rows with seed `1234`.
  - Split shuffled valid rows into first `2/3` SFT reserve and final `1/3`
    OPD reserve.
  - SFT uses the first `25,000` SFT-reserve rows.
  - OPD-1k uses the first `1,024` OPD-reserve rows.
  - OPD-5k continuation uses the next `4,096` OPD-reserve rows.
  - Metadata records selected `source_row_id`s and proves no overlap between
    selected SFT rows and used OPD rows.
- Training:
  - SFT: 4 H200s, `TP=2`, `CP=1`, `DP=2`, `SFT_ZERO_STAGE=3`,
    `SFT_MAX_TOKENS_PER_GPU=16384`, learning rate `1e-6`, exactly one epoch.
  - OPD: 4 H200s, actor/rollout/teacher `3/3/1`, `TP=1`, `CP=3`,
    `OPD_SEQ_LENGTH=32766`, `OPD_MAX_RESPONSE_LEN=31744`,
    `OPD_MAX_TOKENS_PER_GPU=2048`, `OPD_LOG_PROBS_CHUNK_SIZE=512`,
    `OPD_TRAIN_MEMORY_MARGIN_BYTES=0`, learning rate `1e-6`, exactly one pass
    over OPD-1k then one continuation pass over the next 4,096 rows.
  - SFT ZeRO stages `1`, `2`, and `3` are implemented; a tiny smoke job runs
    before full SFT, while the submitted full SFT uses stage `3`.
- Checkpointing:
  - SFT saves model snapshots every 5k effective samples.
  - OPD saves model snapshots every roughly 1k effective samples.
  - The newest checkpoint keeps full optimizer state; older completed
    checkpoints may be reduced to model snapshots after a newer checkpoint is
    validated. In-progress checkpoint dirs are never pruned.
- Eval:
  - Only four full MATH-500 evals run: base, SFT-25k, OPD-1k, OPD-5k.
  - Eval backend is the Tillicum-local vLLM path.
  - vLLM settings: 4 independent 1-GPU workers, greedy decoding,
    `VLLM_EVAL_MAX_MODEL_LEN=32768`, GPU memory utilization `0.92`,
    `VLLM_EVAL_MAX_NUM_SEQS=16`, and
    `VLLM_EVAL_MAX_NUM_BATCHED_TOKENS=131072`.
- Dynamic caps:
  - OPD train cap: `min(31744, OPD_ROLLOUT_MAX_CONTEXT_LEN - prompt_tokens)`,
    with `OPD_ROLLOUT_MAX_CONTEXT_LEN=32766`.
  - vLLM eval cap: `min(31744, EVAL_MAX_CONTEXT_LEN - prompt_tokens)`, with
    `EVAL_MAX_CONTEXT_LEN=32768`.
  - Eval/debug samples record prompt length, requested cap, effective cap, and
    clamp flag.
- Estimated/reserved walltimes:
  - vLLM setup + smoke: expected `20-45m`, reserve `2h`.
  - Clean/split data: expected `3.5-5h`, reserve `8h`.
  - Each full MATH-500 vLLM eval: expected `35-75m`, cap-heavy `2-3h`,
    reserve `4h`.
  - SFT 25k: expected `10-12h`, reserve `16h`.
  - OPD 1k: expected `4.5-5.5h`, reserve `8h`.
  - OPD +4k: expected `16-18h`, reserve `24h`.
- Pre-submit validation:
  - `python3 -m py_compile` on touched Python helpers: passed.
  - `bash -n` on touched shell/sbatch scripts: passed.
  - `git diff --check`: passed.
  - `RUN_CONTAINER_CHECKS=1 bash examples/qwen3_8b_opd_tillicum/run_all_dry_check.sh`: passed.

## Submitted Cleaned OpenThoughts SFT + OPD vLLM Chain

- Implementation commit: `3fb1372` (`Add cleaned OpenThoughts vLLM OPD chain`),
  pushed to `origin/opd-reproduction` before submission.
- Submit time/log: `2026-07-07 21:42 PDT`,
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/slurm_logs/submit_cleaned_sft_opd_vllm_20260707_214245.txt`.
- Dependency policy:
  - Strict `afterok` chain; no train/eval/report job from this chain is
    intentionally eligible to run in parallel with another train/eval/report
    job from the same chain.
  - No job requests more than 4 H200s.
- Job IDs:
  - vLLM setup: `163539` (`gpu:h200:1`, `02:00:00`), started immediately on
    `g013`.
  - Clean data: `163540` (`gpu:h200:1`, `08:00:00`), `afterok:163539`.
  - Model prep/convert: `163541` (`gpu:h200:4`, `02:00:00`),
    `afterok:163540`.
  - SFT ZeRO smoke: `163542` (`gpu:h200:4`, `02:00:00`),
    `afterok:163541`.
  - Base vLLM MATH-500 eval: `163543` (`gpu:h200:4`, `04:00:00`),
    `afterok:163542`.
  - SFT 25k train: `163544` (`gpu:h200:4`, `16:00:00`),
    `afterok:163543`.
  - SFT-25k vLLM MATH-500 eval: `163545` (`gpu:h200:4`, `04:00:00`),
    `afterok:163544`.
  - OPD-1k train: `163546` (`gpu:h200:4`, `08:00:00`),
    `afterok:163545`.
  - OPD-1k vLLM MATH-500 eval: `163547` (`gpu:h200:4`, `04:00:00`),
    `afterok:163546`.
  - OPD +4k train to 5,120 total OPD samples: `163548` (`gpu:h200:4`,
    `24:00:00`), `afterok:163547`.
  - OPD-5k vLLM MATH-500 eval: `163549` (`gpu:h200:4`, `04:00:00`),
    `afterok:163548`.
  - Final report: `163550` (`gpu:h200:1`, `00:30:00`), `afterok:163549`.
- Scheduler validation after submission:
  - `scontrol` shows `MailUser=suryadv@cs.washington.edu` and
    `MailType=END,FAIL` for every job.
  - Representative `ReqTRES` values are `cpu=8,gres/gpu:h200=1` for 1-GPU
    setup/clean/report jobs and `cpu=32,gres/gpu:h200=4` for 4-GPU
    train/eval jobs.
  - No submitted cleaned-chain job was `DependencyNeverSatisfied` at the
    post-submit check.
- Output paths:
  - Cleaned metadata:
    `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/data/openthoughts3_cleaned_cleaned_think_sft25k_opd5k_vllm_metadata.json`.
  - SFT data:
    `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/data/openthoughts3_cleaned_cleaned_think_sft25k_opd5k_vllm_sft_25000.jsonl`.
  - OPD-1k data:
    `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/data/openthoughts3_cleaned_cleaned_think_sft25k_opd5k_vllm_opd_001024.jsonl`.
  - OPD next-4k data:
    `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/data/openthoughts3_cleaned_cleaned_think_sft25k_opd5k_vllm_opd_next_004096.jsonl`.
  - SFT save/snapshots:
    `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/qwen3_8b_cleaned_sft_25k_full_optim`,
    `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/qwen3_8b_cleaned_sft_25k_eval_snapshots`.
  - OPD-1k save/snapshots:
    `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/qwen3_8b_cleaned_sft_25k_opd_1k_32k_full_optim`,
    `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/qwen3_8b_cleaned_sft_25k_opd_1k_32k_eval_snapshots`.
  - OPD-5k save/snapshots:
    `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/qwen3_8b_cleaned_sft_25k_opd_5k_32k_full_optim`,
    `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/qwen3_8b_cleaned_sft_25k_opd_5k_32k_eval_snapshots`.
  - Eval/report dirs:
    `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/math500_eval_cleaned_base_vllm`,
    `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/math500_eval_cleaned_sft_25k_vllm`,
    `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/math500_eval_cleaned_opd_1k_5k_vllm`,
    `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/math500_eval_cleaned_combined_vllm`.
- Runtime validation targets:
  - `163539` validates or installs `vllm` in
    `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/vllm_eval_venv`.
  - `163540` writes cleaned counts near the expected ballpark and records
    selected `source_row_id`s with no SFT/OPD overlap.
  - `163544` logs `SFT_LR=1e-6`, `SFT_ZERO_STAGE=3`, `SFT_NUM_EPOCH=1`, and
    writes final full checkpoint/HF snapshot `iter_0000099`.
  - `163546` logs OPD-1k one-pass settings and writes `iter_0000007`.
  - `163548` loads OPD-1k full optimizer checkpoint, starts the fresh next-4k
    OPD pool at offset 0, and writes `iter_0000039`.
  - vLLM eval jobs write exactly 500 samples each, with dynamic cap metadata
    (`prompt_tokens`, requested/effective cap, clamp flag).

## Failed Cleaned vLLM Setup and Patch

- Failed job: `163539`.
- Failure:
  - The container Python could not create a venv because `ensurepip` is not
    available:
    `The virtual environment was not created successfully because ensurepip is not available`.
  - No data prep, training, eval, checkpoint, or report progress was produced.
- Patch:
  - `10_setup_vllm_eval_env.sbatch` now attempts `python3 -m venv` first and
    falls back to a scratch-local `pip --target` install under
    `$VLLM_EVAL_SITE` when venv support is unavailable.
  - `06_eval_math500_vllm.sbatch` can run either from the venv Python or from
    system `python3` with `$VLLM_EVAL_SITE` prepended to `PYTHONPATH`.
  - `VLLM_EVAL_SITE` and `VLLM_EVAL_PYTHON` are forwarded into the container
    and covered by the dry-check env forwarding guard.
- Preservation/resubmission policy:
  - Stale downstream jobs `163540` through `163550` should be canceled.
  - Resubmit the cleaned chain from the start after pushing this patch.

## Resubmitted Cleaned OpenThoughts Chain After vLLM Setup Patch

- Patch commit: `a10ac61` (`Add vLLM eval target install fallback`), pushed
  to `origin/opd-reproduction` before replacement submission.
- Canceled stale jobs from the first failed chain:
  `163540 163541 163542 163543 163544 163545 163546 163547 163548 163549 163550`.
- Replacement submit time/log: `2026-07-07 21:47 PDT`,
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/slurm_logs/submit_cleaned_sft_opd_vllm_20260707_214730.txt`.
- Replacement job IDs:
  - vLLM setup: `163564` (`gpu:h200:1`, `02:00:00`), started immediately on
    `g013`.
  - Clean data: `163565` (`gpu:h200:1`, `08:00:00`), `afterok:163564`.
  - Model prep/convert: `163566` (`gpu:h200:4`, `02:00:00`),
    `afterok:163565`.
  - SFT ZeRO smoke: `163567` (`gpu:h200:4`, `02:00:00`),
    `afterok:163566`.
  - Base vLLM eval: `163568` (`gpu:h200:4`, `04:00:00`),
    `afterok:163567`.
  - SFT 25k train: `163569` (`gpu:h200:4`, `16:00:00`),
    `afterok:163568`.
  - SFT-25k vLLM eval: `163570` (`gpu:h200:4`, `04:00:00`),
    `afterok:163569`.
  - OPD-1k train: `163571` (`gpu:h200:4`, `08:00:00`),
    `afterok:163570`.
  - OPD-1k vLLM eval: `163572` (`gpu:h200:4`, `04:00:00`),
    `afterok:163571`.
  - OPD +4k train to 5,120 total OPD samples: `163573` (`gpu:h200:4`,
    `24:00:00`), `afterok:163572`.
  - OPD-5k vLLM eval: `163574` (`gpu:h200:4`, `04:00:00`),
    `afterok:163573`.
  - Final report: `163575` (`gpu:h200:1`, `00:30:00`), `afterok:163574`.
- Scheduler validation after replacement submission:
  - `163564` was `RUNNING`; downstream jobs were pending on strict `afterok`
    dependencies.
  - `scontrol` showed `MailUser=suryadv@cs.washington.edu` and
    `MailType=END,FAIL` for every replacement job.
  - `ReqTRES` showed `gpu:h200:1` for setup/clean/report and `gpu:h200:4` for
    convert, smoke, train, and eval jobs.
  - No replacement job was `DependencyNeverSatisfied` at the post-submit check.

## Failed Cleaned vLLM Setup Partial-Venv Retry and Patch

- Failed job: `163564`.
- Failure:
  - The previous failed setup left a partial
    `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/vllm_eval_venv/bin/python`
    without `pip`.
  - The fallback patch saw the executable and tried to use it, then failed
    with `No module named pip`.
- Patch:
  - `10_setup_vllm_eval_env.sbatch` now verifies that a found venv Python can
    run `-m pip --version`; if not, it ignores the partial venv and falls back
    to `$VLLM_EVAL_SITE`.
  - `06_eval_math500_vllm.sbatch` prefers the scratch target install when
    `$VLLM_EVAL_SITE/vllm` exists, so a partial venv cannot shadow the valid
    target install.

## Resubmitted Cleaned OpenThoughts Chain After Partial-Venv Patch

- Patch commit: `e5dd3c3` (`Ignore partial vLLM eval venvs`), pushed to
  `origin/opd-reproduction` before replacement submission.
- Canceled stale jobs from the second failed chain:
  `163565 163566 163567 163568 163569 163570 163571 163572 163573 163574 163575`.
- Replacement submit time/log: `2026-07-07 21:52 PDT`,
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/slurm_logs/submit_cleaned_sft_opd_vllm_20260707_215244.txt`.
- Replacement job IDs:
  - vLLM setup: `163589` (`gpu:h200:1`, `02:00:00`), started immediately on
    `g013`.
  - Clean data: `163590` (`gpu:h200:1`, `08:00:00`), `afterok:163589`.
  - Model prep/convert: `163591` (`gpu:h200:4`, `02:00:00`),
    `afterok:163590`.
  - SFT ZeRO smoke: `163592` (`gpu:h200:4`, `02:00:00`),
    `afterok:163591`.
  - Base vLLM eval: `163593` (`gpu:h200:4`, `04:00:00`),
    `afterok:163592`.
  - SFT 25k train: `163594` (`gpu:h200:4`, `16:00:00`),
    `afterok:163593`.
  - SFT-25k vLLM eval: `163595` (`gpu:h200:4`, `04:00:00`),
    `afterok:163594`.
  - OPD-1k train: `163596` (`gpu:h200:4`, `08:00:00`),
    `afterok:163595`.
  - OPD-1k vLLM eval: `163597` (`gpu:h200:4`, `04:00:00`),
    `afterok:163596`.
  - OPD +4k train to 5,120 total OPD samples: `163598` (`gpu:h200:4`,
    `24:00:00`), `afterok:163597`.
  - OPD-5k vLLM eval: `163599` (`gpu:h200:4`, `04:00:00`),
    `afterok:163598`.
  - Final report: `163600` (`gpu:h200:1`, `00:30:00`), `afterok:163599`.
- Scheduler/log validation after replacement submission:
  - `163589` was `RUNNING`; downstream jobs were pending on strict `afterok`
    dependencies.
  - The setup log showed:
    `Ignoring incomplete vLLM venv without pip` and
    `Falling back to scratch-local pip --target install`.
  - No replacement job was `DependencyNeverSatisfied` at the post-submit check.

## Failed Cleaned vLLM Setup Target-Library Import and Patch

- Failed job: `163589`.
- Failure:
  - The scratch target install completed, but final `import vllm` failed
    because the dynamic linker could not find target-installed CUDA libraries:
    `ImportError: libcudart.so.13: cannot open shared object file`.
  - The log also showed SGLang sitecustomize conflicts during vLLM-only Python
    startup.
- Patch:
  - vLLM setup/eval now prepend `$VLLM_EVAL_SITE/nvidia/*/lib` directories to
    `LD_LIBRARY_PATH` when using the scratch target install.
  - vLLM setup/eval set `SLIME_SGLANG_PATCH_SITE=0` for vLLM-only Python so
    SGLang startup patches do not run during vLLM import.
  - `container_exec.sh` forwards `LD_LIBRARY_PATH`, and the dry check verifies
    that forwarding entry exists.

## Resubmitted Cleaned OpenThoughts Chain After Target-Library Patch

- Patch commit: `84074df` (`Forward vLLM target library paths`), pushed to
  `origin/opd-reproduction` before replacement submission.
- Canceled stale jobs from the third failed chain:
  `163590 163591 163592 163593 163594 163595 163596 163597 163598 163599 163600`.
- Replacement submit time/log: `2026-07-07 22:10 PDT`,
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/slurm_logs/submit_cleaned_sft_opd_vllm_20260707_221054.txt`.
- Replacement job IDs:
  - vLLM setup: `163642` (`gpu:h200:1`, `02:00:00`), completed successfully
    on `g015`.
  - Clean data: `163643` (`gpu:h200:1`, `08:00:00`), started after
    `163642`.
  - Model prep/convert: `163644` (`gpu:h200:4`, `02:00:00`),
    `afterok:163643`.
  - SFT ZeRO smoke: `163645` (`gpu:h200:4`, `02:00:00`),
    `afterok:163644`.
  - Base vLLM eval: `163646` (`gpu:h200:4`, `04:00:00`),
    `afterok:163645`.
  - SFT 25k train: `163647` (`gpu:h200:4`, `16:00:00`),
    `afterok:163646`.
  - SFT-25k vLLM eval: `163648` (`gpu:h200:4`, `04:00:00`),
    `afterok:163647`.
  - OPD-1k train: `163649` (`gpu:h200:4`, `08:00:00`),
    `afterok:163648`.
  - OPD-1k vLLM eval: `163650` (`gpu:h200:4`, `04:00:00`),
    `afterok:163649`.
  - OPD +4k train to 5,120 total OPD samples: `163651` (`gpu:h200:4`,
    `24:00:00`), `afterok:163650`.
  - OPD-5k vLLM eval: `163652` (`gpu:h200:4`, `04:00:00`),
    `afterok:163651`.
  - Final report: `163653` (`gpu:h200:1`, `00:30:00`), `afterok:163652`.
- Runtime validation observed:
  - Setup log showed fallback to scratch target, then `vllm 0.24.0`, then
    `Finished vLLM eval environment setup`.
  - `163643` was running after `163642` completed, confirming the strict
    `afterok` chain advanced past setup.

## Canceled Older OPD Continuation Chain

- Cleanup request: remove the older OPD continuation/val100 chain so it does
  not run alongside the new cleaned OpenThoughts experiment.
- Canceled jobs:
  - Running train job: `163315` (`slime-qwen3-opd-cont-05120`).
  - Pending continuation/eval/report jobs:
    `163316 163317 163318 163319 163320 163321 163322 163323 163324 163325
    163326 163327 163328 163329 163330 163331 163332 163333 163334 163335
    163336 163337 163338 163339 163340 163341 163342 163343 163344 163345
    163346 163347 163348 163349 163350 163351 163352 163353 163354 163355
    163356 163357 163358 163359 163360`.
- Verification:
  - Post-cancel `squeue -u suryadv` showed no older
    `slime-qwen3-opd-cont-*`, `slime-qwen3-val100-*`,
    `slime-qwen3-opd12544-full500`, or `slime-qwen3-opd25k-val-report` jobs.
  - The active cleaned chain was left untouched: `163643` was running and
    cleaned downstream jobs `163644` through `163653` remained pending on the
    cleaned dependency chain.

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

## Corrected SFT-Weights Colocate Retry After Env-Forwarding Fix

- Failed job: `158697`, `slime-qwen3-opd1k-sft4g`, failed after `00:00:05`
  on `2026-07-02T16:30:09`.
- Root cause: `container_exec.sh` forwarded `REPORT_EXPERIMENT_NOTE` through
  repeated Apptainer `--env` flags. The note included `GPUs 0,1,2`; Apptainer
  split the comma-separated value and rejected `1` because it was not
  formatted as `key=value`.
- Progress decision: no corrected OPD progress exists. The full optimizer save
  dir and HF snapshot dir are empty top-level dirs, and no rollout/debug
  artifact was created, so the fidelity-safe restart point remains the final
  SFT HF snapshot at `iter_0000096`.
- Patch commit: `1a56828` (`Fix Apptainer env forwarding for report notes`),
  pushed to `origin/opd-reproduction`.
- Key patch behavior:
  - `PASS_ENV` variables are forwarded with Apptainer/Singularity
    prefix-based env vars instead of repeated `--env key=value` arguments.
  - Newline-containing env values fail fast with a clear wrapper error.
  - Container launch failures from Apptainer CLI/env parsing are reported
    separately from Python stdlib import failures.
  - `run_all_dry_check.sh` now verifies that a comma-bearing
    `REPORT_EXPERIMENT_NOTE` survives into the container unchanged.
- Static validation before replacement submission:
  - `bash -n` on changed shell scripts passed.
  - `git diff --check` passed.
  - Full colocate4 dry check with `RUN_CONTAINER_CHECKS=1` passed, including
    Slurm `sbatch --test-only`, container imports, and the comma-env probe.
- Canceled stale dependent jobs: `158698`, `158699`, `158700`.
- Replacement submit time: `2026-07-02T17:26:11-07:00`.
- Dependency policy: replacement train has no dependency; downstream jobs use
  `afterok`.
- OPD train job: `158772`, `slime-qwen3-opd1k-sft4g`,
  `gpu:h200:4`, `time=18:00:00`, `Dependency=(null)`, pending for resources
  at submission.
- OPD final eval job: `158773`, `slime-qwen3-opd1k-sft4g-eval`,
  `gpu:h200:4`, `time=05:00:00`, dependency `afterok:158772`.
- Base maybe-eval job: `158774`, `slime-qwen3-base-math500-maybe`,
  `gpu:h200:4`, `time=05:00:00`, dependency `afterok:158773`.
- Final report job: `158775`, `slime-qwen3-final-report-sft4g`,
  `gpu:h200:1`, `time=00:30:00`, dependency `afterok:158774`.
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
- Expected runtime validation: new OPD log should have no
  `invalid argument ... for "--env" flag`, should pass container preflight,
  should load the SFT HF snapshot at `iter_0000096`, should show
  colocate/offload enabled with actor/rollout/teacher `3/3/1`, and should reach
  teacher `/health_generate`, rollout `0`, and actor train.

## Corrected SFT-Weights Colocate Retry With CUDA Graphs Disabled

- Superseded retry: `158772`, `slime-qwen3-opd1k-sft4g`, started on
  `2026-07-02T17:27:40` and was canceled on `2026-07-02T18:04:46` after
  teacher startup repeatedly failed before `/health_generate`.
- Root cause: the minimal Apptainer sandbox contains SGLang/NVCC but does not
  contain the full CUDA development/header and GCC toolchain needed for fresh
  SGLang CUDA-graph/JIT compilation. The teacher log failed on
  `gcc: No such file or directory` while compiling a fused-rope CUDA kernel,
  and direct probes showed CUDA runtime headers were also absent.
- Progress decision: no corrected OPD rollout, checkpoint, HF snapshot, or
  dataset state was produced by `158772`; restart from final SFT HF weights
  remains fidelity-safe.
- Patch commit: `06a6b10` (`Disable SGLang cuda graphs for Tillicum OPD`),
  pushed to `origin/opd-reproduction`.
- Key patch behavior:
  - Direct Qwen3-32B teacher server gets `--disable-cuda-graph`.
  - Student rollout SGLang engines get `--sglang-disable-cuda-graph`.
  - Eval SGLang engines also get `--sglang-disable-cuda-graph`.
  - The OPD teacher wait loop now exits if the teacher process dies before
    `/health_generate`, instead of tailing a dead server log until walltime.
  - The earlier experimental host-GCC bind attempt was not kept.
- Static validation before replacement submission:
  - `bash -n` on changed shell/sbatch scripts passed.
  - `git diff --check` passed.
  - Full colocate4 dry check with `RUN_CONTAINER_CHECKS=1` passed, including
    Slurm `sbatch --test-only`, container imports, and the comma-env probe.
- Canceled stale jobs: `158772`, `158773`, `158774`, `158775`.
- Replacement submit time: `2026-07-02T18:05:22-07:00`.
- Dependency policy: replacement train has no dependency; downstream jobs use
  `afterok`.
- OPD train job: `158786`, `slime-qwen3-opd1k-sft4g`,
  `gpu:h200:4`, `time=18:00:00`, `Dependency=(null)`, running on `g001` at
  submission verification.
- OPD final eval job: `158787`, `slime-qwen3-opd1k-sft4g-eval`,
  `gpu:h200:4`, `time=05:00:00`, dependency `afterok:158786`.
- Base maybe-eval job: `158788`, `slime-qwen3-base-math500-maybe`,
  `gpu:h200:4`, `time=05:00:00`, dependency `afterok:158787`.
- Final report job: `158789`, `slime-qwen3-final-report-sft4g`,
  `gpu:h200:1`, `time=00:30:00`, dependency `afterok:158788`.
- Mail for all replacement jobs: `MailUser=suryadv@cs.washington.edu`,
  `MailType=END,FAIL`.
- OPD/EVAL CUDA graph disable flags: `OPD_DISABLE_CUDA_GRAPH=1`,
  `EVAL_DISABLE_CUDA_GRAPH=1`.
- Expected runtime validation: new OPD log should show
  `OPD disable cuda graph: 1`, teacher args with `disable_cuda_graph=True`,
  no SGLang fused-rope NVCC/GCC failure, teacher `/health_generate`, SFT HF
  snapshot load at `iter_0000096`, rollout `0`, and actor train.

## Corrected SFT-Weights Colocate Retry With Native SGLang RoPE Path

- Superseded retry: `158786`, `slime-qwen3-opd1k-sft4g`, started on
  `2026-07-02T18:05:22` and failed before teacher `/health_generate`.
- Root cause: disabling CUDA graphs was not sufficient. The Qwen3 teacher still
  entered SGLang's fused RoPE JIT path during warmup and failed because the
  sandbox lacks `gcc` / CUDA development headers:
  `gcc: No such file or directory` and
  `nvcc fatal: Failed to preprocess host compiler properties`.
- Progress decision: no corrected OPD rollout, checkpoint, HF snapshot, or
  dataset state was produced by `158786`; restart from final SFT HF weights
  remains fidelity-safe.
- Patch behavior:
  - Direct Qwen3-32B teacher server gets
    `--rl-on-policy-target fsdp`.
  - Student rollout SGLang engines get
    `--sglang-rl-on-policy-target fsdp`.
  - Eval SGLang engines get
    `--sglang-rl-on-policy-target fsdp`.
  - `OPD_SGLANG_RL_ON_POLICY_TARGET` and
    `EVAL_SGLANG_RL_ON_POLICY_TARGET` are forwarded through Apptainer and
    recorded in submit logs.
  - The setting uses SGLang's on-policy/native path for Qwen rotary embeddings,
    avoiding runtime fused RoPE NVCC compilation in this minimal sandbox.
- Static validation before replacement submission:
  - `bash -n` on changed shell/sbatch scripts passed.
  - `git diff --check` passed.
  - Full colocate4 dry check with `RUN_CONTAINER_CHECKS=1` passed, including
    Slurm `sbatch --test-only`, container imports, and the comma-env probe.
- Patch commit: `7c357c1` (`Use SGLang native path for Tillicum RoPE`),
  pushed to `origin/opd-reproduction`.
- Canceled stale jobs: `158786`, `158787`, `158788`, `158789`.
- Replacement submit time: `2026-07-02T18:13:42-07:00`.
- Dependency policy: replacement train has no dependency; downstream jobs use
  `afterok`.
- OPD train job: `158799`, `slime-qwen3-opd1k-sft4g`,
  `gpu:h200:4`, `time=18:00:00`, `Dependency=(null)`, running on `g001` at
  submission verification.
- OPD final eval job: `158800`, `slime-qwen3-opd1k-sft4g-eval`,
  `gpu:h200:4`, `time=05:00:00`, dependency `afterok:158799`.
- Base maybe-eval job: `158801`, `slime-qwen3-base-math500-maybe`,
  `gpu:h200:4`, `time=05:00:00`, dependency `afterok:158800`.
- Final report job: `158802`, `slime-qwen3-final-report-sft4g`,
  `gpu:h200:1`, `time=00:30:00`, dependency `afterok:158801`.
- Mail for all replacement jobs: `MailUser=suryadv@cs.washington.edu`,
  `MailType=END,FAIL`.
- OPD/EVAL SGLang native path flags:
  `OPD_SGLANG_RL_ON_POLICY_TARGET=fsdp`,
  `EVAL_SGLANG_RL_ON_POLICY_TARGET=fsdp`.
- Expected runtime validation: new OPD log should show
  `OPD SGLang rl-on-policy target: fsdp`, teacher args with
  `rl_on_policy_target='fsdp'`, no fused-RoPE NVCC/GCC failure, teacher
  `/health_generate`, SFT HF snapshot load at `iter_0000096`, rollout `0`, and
  actor train.

## Corrected SFT-Weights Colocate Retry With Narrow Native-RoPE Shim

- Superseded retry: `158799`, `slime-qwen3-opd1k-sft4g`, started on
  `2026-07-02T18:13:46` and failed before teacher `/health_generate`.
- Root cause: `--rl-on-policy-target fsdp` avoided the fused RoPE NVCC path but
  also enabled deterministic batch-invariant SGLang ops. During teacher warmup,
  SGLang replaced matmul with a Triton persistent matmul and failed because the
  sandbox still lacks a C compiler:
  `RuntimeError: Failed to find C compiler`.
- Progress decision: no corrected OPD rollout, checkpoint, HF snapshot, or
  dataset state was produced by `158799`; restart from final SFT HF weights
  remains fidelity-safe.
- Patch behavior:
  - Added `slime.backends.sglang_utils.native_rope`, a narrow shim controlled
    by `SLIME_SGLANG_FORCE_NATIVE_ROPE=1`.
  - The shim patches SGLang `RotaryEmbedding` instances to use
    `forward_native`, avoiding fused RoPE JIT compilation without setting
    `rl_on_policy_target`.
  - Added a tracked teacher launcher
    `examples/qwen3_8b_opd_tillicum/sglang_launch_native_rope.py`.
  - Patched Slime's Ray SGLang engine child process target so rollout/eval
    SGLang servers apply the same shim after Python `spawn`.
  - Reset OPD/EVAL `SGLANG_RL_ON_POLICY_TARGET` defaults to empty so
    deterministic batch-invariant matmul is not enabled by default.
- Static validation before replacement submission:
  - `bash -n` on changed shell/sbatch scripts passed.
  - `python3 -m py_compile` on touched Python files passed.
  - `git diff --check` passed.
  - Full colocate4 dry check with `RUN_CONTAINER_CHECKS=1` passed, including
    Slurm `sbatch --test-only`, container imports, and the comma-env probe.
- Patch commit: `9ab04c1` (`Force native SGLang RoPE without deterministic
  mode`), pushed to `origin/opd-reproduction`.
- Canceled stale jobs: `158800`, `158801`, `158802`.
- Replacement submit time: `2026-07-02T18:21:41-07:00`.
- Dependency policy: replacement train has no dependency; downstream jobs use
  `afterok`.
- OPD train job: `158815`, `slime-qwen3-opd1k-sft4g`,
  `gpu:h200:4`, `time=18:00:00`, `Dependency=(null)`, running on `g017` at
  submission verification.
- OPD final eval job: `158816`, `slime-qwen3-opd1k-sft4g-eval`,
  `gpu:h200:4`, `time=05:00:00`, dependency `afterok:158815`.
- Base maybe-eval job: `158817`, `slime-qwen3-base-math500-maybe`,
  `gpu:h200:4`, `time=05:00:00`, dependency `afterok:158816`.
- Final report job: `158818`, `slime-qwen3-final-report-sft4g`,
  `gpu:h200:1`, `time=00:30:00`, dependency `afterok:158817`.
- Mail for all replacement jobs: `MailUser=suryadv@cs.washington.edu`,
  `MailType=END,FAIL`.
- OPD/EVAL SGLang native-RoPE flags:
  `SLIME_SGLANG_FORCE_NATIVE_ROPE=1`,
  `OPD_SGLANG_RL_ON_POLICY_TARGET=<empty>`,
  `EVAL_SGLANG_RL_ON_POLICY_TARGET=<empty>`.
- Expected runtime validation: new OPD log should show
  `SLIME SGLang force native RoPE: 1`, no deterministic
  `rl_on_policy_target='fsdp'`, no fused-RoPE NVCC/GCC failure, no Triton
  `Failed to find C compiler`, teacher `/health_generate`, SFT HF snapshot
  load at `iter_0000096`, rollout `0`, and actor train.

## Corrected SFT-Weights Colocate Retry With Native RoPE And Activation

- Superseded retry: `158815`, `slime-qwen3-opd1k-sft4g`, started on
  `2026-07-02T18:21:45-07:00` and failed before teacher
  `/health_generate`.
- Root cause: the narrow native-RoPE patch avoided the fused RoPE compiler
  path, but SGLang's Qwen MLP activation still tried to JIT-compile
  `silu_and_mul` during warmup. The runtime sandbox lacks `gcc`, so NVCC
  failed with `gcc: No such file or directory` and
  `nvcc fatal: Failed to preprocess host compiler properties`.
- Progress decision: no corrected OPD rollout, checkpoint, HF snapshot, or
  dataset state was produced by `158815`; restart from final SFT HF weights
  remains fidelity-safe.
- Patch behavior:
  - Extended `slime.backends.sglang_utils.native_rope` so
    `SLIME_SGLANG_FORCE_NATIVE_ROPE=1` patches both SGLang
    `RotaryEmbedding` and `SiluAndMul` objects to call their existing
    `forward_native` implementations.
  - This still leaves `OPD_SGLANG_RL_ON_POLICY_TARGET` and
    `EVAL_SGLANG_RL_ON_POLICY_TARGET` empty, avoiding the deterministic
    batch-invariant matmul path that previously failed under Triton.
- Static validation before replacement submission:
  - `python3 -m py_compile slime/backends/sglang_utils/native_rope.py`
    passed.
  - `git diff --check` passed.
  - Targeted container regression passed: with SGLang global server args set,
    a new `SiluAndMul` instance uses `forward_native`.
  - Full colocate4 dry check with `RUN_CONTAINER_CHECKS=1` passed, including
    Slurm `sbatch --test-only`, container imports, and the comma-env probe.
- Patch commit: `a48f652` (`Force native SGLang activation without
  deterministic mode`), pushed to `origin/opd-reproduction`.
- Canceled stale jobs: `158816`, `158817`, `158818`.
- Replacement submit time: `2026-07-02T18:37:18-07:00`.
- Dependency policy: replacement train has no dependency; downstream jobs use
  `afterok`.
- OPD train job: `158832`, `slime-qwen3-opd1k-sft4g`,
  `gpu:h200:4`, `time=18:00:00`, `Dependency=(null)`, pending for resources
  at submission verification.
- OPD final eval job: `158833`, `slime-qwen3-opd1k-sft4g-eval`,
  `gpu:h200:4`, `time=05:00:00`, dependency `afterok:158832`.
- Base maybe-eval job: `158834`, `slime-qwen3-base-math500-maybe`,
  `gpu:h200:4`, `time=05:00:00`, dependency `afterok:158833`.
- Final report job: `158835`, `slime-qwen3-final-report-sft4g`,
  `gpu:h200:1`, `time=00:30:00`, dependency `afterok:158834`.
- Mail for all replacement jobs: `MailUser=suryadv@cs.washington.edu`,
  `MailType=END,FAIL`.
- Expected runtime validation: new OPD log should show the updated shim message
  `patched SGLang RotaryEmbedding and SiluAndMul`, no deterministic
  `rl_on_policy_target='fsdp'`, no fused RoPE or activation NVCC/GCC failure,
  teacher `/health_generate`, SFT HF snapshot load at `iter_0000096`,
  rollout `0`, and actor train.

## Corrected SFT-Weights Colocate Retry With Native Clamp Position

- Superseded retry: `158832`, `slime-qwen3-opd1k-sft4g`, started on
  `2026-07-02T18:55:38-07:00` and failed during the first teacher
  `/health_generate` request.
- Root cause: the native RoPE and native SiLU patch worked, but SGLang then
  tried to JIT-compile `sglang.jit_kernel.clamp_position` during scheduler
  warmup. The runtime sandbox lacks `gcc`, so NVCC failed with
  `gcc: No such file or directory` and
  `nvcc fatal: Failed to preprocess host compiler properties`.
- Progress decision: no corrected OPD rollout, checkpoint, HF snapshot, sanity
  report, or dataset state was produced by `158832`; restart from final SFT HF
  weights remains fidelity-safe.
- Patch behavior:
  - Extended `slime.backends.sglang_utils.native_rope` so
    `SLIME_SGLANG_FORCE_NATIVE_ROPE=1` also patches
    `sglang.srt.model_executor.forward_batch_info.clamp_position` to SGLang's
    own `_clamp_position_native`.
  - Added `--disable-piecewise-cuda-graph` for the direct teacher SGLang server
    and `--sglang-disable-piecewise-cuda-graph` for rollout/eval SGLang
    engines whenever the existing CUDA graph disable flags are enabled.
  - This still leaves `OPD_SGLANG_RL_ON_POLICY_TARGET` and
    `EVAL_SGLANG_RL_ON_POLICY_TARGET` empty, avoiding the deterministic
    batch-invariant matmul path that previously failed under Triton.
- Static validation before replacement submission:
  - `python3 -m py_compile slime/backends/sglang_utils/native_rope.py`
    passed.
  - `bash -n` on touched shell/sbatch scripts passed.
  - `git diff --check` passed.
  - Targeted container regression passed:
    `forward_batch_info.clamp_position is _clamp_position_native` and
    `[0, 1, 5]` maps to `tensor([0, 0, 4])`.
  - Full colocate4 dry check with `RUN_CONTAINER_CHECKS=1` passed, including
    Slurm `sbatch --test-only`, container imports, and the comma-env probe.
- Patch commit: `74a2e0f` (`Patch SGLang clamp fallback for Tillicum`),
  pushed to `origin/opd-reproduction`.
- Canceled stale jobs: `158833`, `158834`, `158835`.
- Replacement submit time: `2026-07-02T19:19:10-07:00`.
- Dependency policy: replacement train has no dependency; downstream jobs use
  `afterok`.
- OPD train job: `158872`, `slime-qwen3-opd1k-sft4g`,
  `gpu:h200:4`, `time=18:00:00`, `Dependency=(null)`, pending for resources
  at submission verification.
- OPD final eval job: `158873`, `slime-qwen3-opd1k-sft4g-eval`,
  `gpu:h200:4`, `time=05:00:00`, dependency `afterok:158872`.
- Base maybe-eval job: `158874`, `slime-qwen3-base-math500-maybe`,
  `gpu:h200:4`, `time=05:00:00`, dependency `afterok:158873`.
- Final report job: `158875`, `slime-qwen3-final-report-sft4g`,
  `gpu:h200:1`, `time=00:30:00`, dependency `afterok:158874`.
- Mail for all replacement jobs: `MailUser=suryadv@cs.washington.edu`,
  `MailType=END,FAIL`.
- Expected runtime validation: new OPD log should show the updated shim message
  including `clamp_position`, teacher args with
  `disable_piecewise_cuda_graph=True`, no clamp-position NVCC/GCC failure,
  teacher `/health_generate`, SFT HF snapshot load at `iter_0000096`,
  rollout `0`, and actor train.

## Corrected SFT-Weights Colocate Retry With SGLang Allocator Scoping

- Superseded retry: `158872`, `slime-qwen3-opd1k-sft4g`, started on
  `2026-07-02T19:40:17-07:00` and failed after `00:03:23`.
- Root cause: the direct teacher and SGLang native-op shims worked, but the
  colocated rollout SGLang engines inherited
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` while using SGLang
  `TorchMemorySaver`. SGLang memory saver rejects expandable segments and
  aborted during rollout engine model load with
  `TorchMemorySaver is disabled for the current process because
  expandable_segments is not supported yet`.
- Progress decision: no corrected OPD rollout, checkpoint, HF snapshot, sanity
  report, or dataset state was produced by `158872`; restart from final SFT HF
  weights remains fidelity-safe.
- Patch behavior:
  - Added `SLIME_SGLANG_PYTORCH_CUDA_ALLOC_CONF`, default
    `max_split_size_mb:128`, and forwarded it through the Tillicum container
    wrapper and Ray runtime env.
  - Scoped allocator override to SGLang rollout actors whose memory saver is
    enabled, leaving the global Megatron/Ray driver
    `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` unchanged.
  - Added a runtime log line showing the SGLang actor allocator override.
  - Added a focused regression test for the allocator helper.
- Static validation before replacement submission:
  - `python3 -m py_compile` on touched Python files passed.
  - `bash -n` on touched shell/sbatch scripts passed.
  - `git diff --check` passed.
  - Focused pytest file was added; host `pytest` was unavailable in the login
    environment, so it was syntax-checked with `py_compile` but not executed
    there.
  - Full colocate4 dry check with `RUN_CONTAINER_CHECKS=1` passed, including
    Slurm `sbatch --test-only`, container imports, and the comma-env probe.
- Patch commit: `a0f5b33` (`Scope SGLang memory saver allocator env`),
  pushed to `origin/opd-reproduction`.
- Canceled stale jobs: `158873`, `158874`, `158875`.
- Replacement submit time: `2026-07-03T01:08:25-07:00`.
- Dependency policy: replacement train has no dependency; downstream jobs use
  `afterok`.
- OPD train job: `159090`, `slime-qwen3-opd1k-sft4g`,
  `gpu:h200:4`, `time=18:00:00`, `Dependency=(null)`, pending for resources
  at submission verification with estimated start
  `2026-07-03T03:47:20-07:00`.
- OPD final eval job: `159091`, `slime-qwen3-opd1k-sft4g-eval`,
  `gpu:h200:4`, `time=05:00:00`, dependency `afterok:159090`.
- Base maybe-eval job: `159092`, `slime-qwen3-base-math500-maybe`,
  `gpu:h200:4`, `time=05:00:00`, dependency `afterok:159091`.
- Final report job: `159093`, `slime-qwen3-final-report-sft4g`,
  `gpu:h200:1`, `time=00:30:00`, dependency `afterok:159092`.
- Mail for all replacement jobs: `MailUser=suryadv@cs.washington.edu`,
  `MailType=END,FAIL`.
- Expected runtime validation: new OPD log should show
  `SGLang engine rank=... uses memory saver; PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`,
  no `TorchMemorySaver ... expandable_segments` failure, teacher
  `/health_generate`, SFT HF snapshot load at `iter_0000096`, rollout `0`,
  and actor train.

## Corrected SFT-Weights Colocate Retry With Method-Level Native RoPE

- Superseded retry: `159090`, `slime-qwen3-opd1k-sft4g`, started on
  `2026-07-03T03:48:08-07:00` and failed after `00:04:55`.
- Root cause: the allocator-scoping patch worked, but rollout SGLang engines
  still entered `RotaryEmbedding.forward_cuda` during warmup. That path tried
  to JIT-compile fused RoPE with NVCC/ninja; the runtime sandbox lacks `gcc`,
  so NVCC failed with `gcc: No such file or directory` and
  `ninja exited with status 1`.
- Progress decision: no corrected OPD rollout, checkpoint, HF snapshot, sanity
  report, or dataset state was produced by `159090`; restart from final SFT HF
  weights remains fidelity-safe.
- Patch behavior:
  - Moved the rollout SGLang HTTP server import until after
    `maybe_force_native_rope()` runs.
  - Extended the native shim to patch `RotaryEmbedding.forward_cuda` itself to
    call `forward_native`, not only the constructor's `_forward_method`.
  - Updated already-cached RoPE objects in SGLang's `_ROPE_DICT` whenever the
    shim runs.
  - Kept the existing native `SiluAndMul`, `clamp_position`, and rollout
    memory-saver allocator patches.
- Static validation before replacement submission:
  - `python3 -m py_compile slime/backends/sglang_utils/native_rope.py
    slime/backends/sglang_utils/sglang_engine.py` passed.
  - `git diff --check` passed.
  - Targeted container regression passed: with
    `SLIME_SGLANG_FORCE_NATIVE_ROPE=1`, `RotaryEmbedding.forward_cuda` was
    replaced, one cached RoPE object was repaired, and a fresh RoPE object used
    `forward_native`.
  - Full colocate4 dry check with `RUN_CONTAINER_CHECKS=1` passed, including
    Slurm `sbatch --test-only`, container imports, and the comma-env probe.
- Patch commit: `072357b` (`Force native SGLang RoPE method path`), pushed to
  `origin/opd-reproduction`.
- Canceled stale jobs: `159091`, `159092`, `159093`.
- Replacement submit time: `2026-07-03T04:18:54-07:00`.
- Dependency policy: replacement train has no dependency; downstream jobs use
  `afterok`.
- OPD train job: `159189`, `slime-qwen3-opd1k-sft4g`,
  `gpu:h200:4`, `time=18:00:00`, `Dependency=(null)`, running on `g024` at
  submission verification.
- OPD final eval job: `159190`, `slime-qwen3-opd1k-sft4g-eval`,
  `gpu:h200:4`, `time=05:00:00`, dependency `afterok:159189`.
- Base maybe-eval job: `159191`, `slime-qwen3-base-math500-maybe`,
  `gpu:h200:4`, `time=05:00:00`, dependency `afterok:159190`.
- Final report job: `159192`, `slime-qwen3-final-report-sft4g`,
  `gpu:h200:1`, `time=00:30:00`, dependency `afterok:159191`.
- Mail for all replacement jobs: `MailUser=suryadv@cs.washington.edu`,
  `MailType=END,FAIL`.
- Expected runtime validation: new OPD log should show the method-level native
  RoPE shim message, memory-saver allocator override, no fused-RoPE
  NVCC/ninja failure, teacher `/health_generate`, SFT HF snapshot load at
  `iter_0000096`, rollout `0`, and actor train.

## Corrected SFT-Weights Colocate Retry With Startup Native RoPE Hook

- Superseded retry: `159189`, `slime-qwen3-opd1k-sft4g`, started on
  `2026-07-03T04:18:54-07:00` and failed after `00:03:43`.
- Root cause: the method-level native RoPE patch was active in the Ray actor
  process, but SGLang launched a fresh scheduler/model-worker Python process
  during rollout engine startup. That child process did not call the shim
  before importing SGLang RoPE code, so it still entered
  `RotaryEmbedding.forward_cuda` and hit the no-compiler fused-RoPE JIT path
  (`gcc: No such file or directory`, `ninja exited with status 1`).
- Progress decision: no corrected OPD rollout, checkpoint, HF snapshot, sanity
  report, or dataset state was produced by `159189`; restart from final SFT HF
  weights remains fidelity-safe.
- Patch behavior:
  - Added a gated repo-root `sitecustomize.py` startup hook. When
    `SLIME_SGLANG_PATCH_SITE=1`, every Python process with the repo on
    `PYTHONPATH` applies `maybe_force_native_rope()` before user code imports
    SGLang.
  - Defaulted `SLIME_SGLANG_PATCH_SITE` to
    `SLIME_SGLANG_FORCE_NATIVE_ROPE` in the Tillicum environment and forwarded
    it through the Apptainer wrapper.
  - Passed `SLIME_SGLANG_PATCH_SITE` through Ray runtime env and SGLang actor
    env so rollout scheduler/model-worker subprocesses inherit it.
  - Updated the direct teacher launcher to set the startup-hook env for any
    SGLang child processes it creates.
- Static validation before replacement submission:
  - `python3 -m py_compile sitecustomize.py
    slime/backends/sglang_utils/native_rope.py
    slime/backends/sglang_utils/sglang_engine.py slime/ray/rollout.py
    examples/qwen3_8b_opd_tillicum/sglang_launch_native_rope.py` passed.
  - `bash -n` on touched shell/sbatch scripts passed.
  - `git diff --check` passed.
  - Targeted container regression passed: with
    `SLIME_SGLANG_PATCH_SITE=1` and `SLIME_SGLANG_FORCE_NATIVE_ROPE=1`,
    `RotaryEmbedding.forward_cuda` was patched at Python startup before normal
    code ran.
  - Cached/fresh RoPE container regression passed: cached `_ROPE_DICT` objects
    and newly constructed RoPE objects both used `forward_native`.
  - Full colocate4 dry check with `RUN_CONTAINER_CHECKS=1` passed, including
    Slurm `sbatch --test-only`, container imports, startup native shim, and the
    comma-env probe.
- Patch commit: `ddf36d8` (`Apply SGLang patches at Python startup`), pushed
  to `origin/opd-reproduction`.
- Canceled stale jobs: `159190`, `159191`, `159192`.
- Replacement submit time: `2026-07-03T13:46:33-07:00`.
- Dependency policy: replacement train has no dependency; downstream jobs use
  `afterok`.
- OPD train job: `159392`, `slime-qwen3-opd1k-sft4g`,
  `gpu:h200:4`, `time=18:00:00`, `Dependency=(null)`, running on `g018` at
  submission verification.
- OPD final eval job: `159393`, `slime-qwen3-opd1k-sft4g-eval`,
  `gpu:h200:4`, `time=05:00:00`, dependency `afterok:159392`.
- Base maybe-eval job: `159394`, `slime-qwen3-base-math500-maybe`,
  `gpu:h200:4`, `time=05:00:00`, dependency `afterok:159393`.
- Final report job: `159395`, `slime-qwen3-final-report-sft4g`,
  `gpu:h200:1`, `time=00:30:00`, dependency `afterok:159394`.
- Mail for all replacement jobs: `MailUser=suryadv@cs.washington.edu`,
  `MailType=END,FAIL`.
- Expected runtime validation: new OPD log should show
  `SLIME_SGLANG_PATCH_SITE=1: applied SGLang native compatibility patches at
  Python startup`, memory-saver allocator override, no fused-RoPE NVCC/ninja
  failure, teacher `/health_generate`, SFT HF snapshot load at
  `iter_0000096`, rollout `0`, and actor train.

## Corrected SFT-Weights Colocate Retry With Ray Control-Plane Scoping

- Superseded retry: `159392`, `slime-qwen3-opd1k-sft4g`, started on
  `2026-07-03T13:46:34-07:00` and failed after `00:09:34`.
- Root cause: the startup native RoPE hook fixed the earlier SGLang
  fused-RoPE compiler failure, but it was still enabled globally for the Ray
  control plane. `ray start` succeeded, then `ray job submit` failed through
  the dashboard with `RuntimeError: Request failed with status code 504`.
- Progress decision: the job failed before the Ray training job was submitted,
  so no corrected OPD rollout, checkpoint, HF snapshot, sanity report, or
  dataset state was produced. Restart from final SFT HF weights remains
  fidelity-safe.
- Patch behavior:
  - Run `ray start` and `ray job submit` with `SLIME_SGLANG_PATCH_SITE=0` so
    Ray dashboard/CLI/control-plane processes do not import SGLang/Torch via
    `sitecustomize.py`.
  - Keep `SLIME_SGLANG_PATCH_SITE=1` in the submitted Ray runtime env for OPD
    and eval workers, so SGLang rollout/eval subprocesses still get the native
    RoPE startup hook.
  - Applied the same Ray-control-plane scoping to the SFT sbatch script to
    avoid this failure mode in future SFT reruns.
- Static validation before replacement submission:
  - `bash -n` on touched shell/sbatch scripts passed.
  - `git diff --check` passed.
  - Full colocate4 dry check with `RUN_CONTAINER_CHECKS=1` passed, including
    Slurm `sbatch --test-only`, container imports, startup native shim, and the
    comma-env probe.
- Patch commit: `ce640a4` (`Scope SGLang startup hook away from Ray control
  plane`), pushed to `origin/opd-reproduction`.
- Canceled stale jobs: `159393`, `159394`, `159395`.
- Replacement submit time: `2026-07-03T14:04:05-07:00`.
- Dependency policy: replacement train has no dependency; downstream jobs use
  `afterok`.
- OPD train job: `159409`, `slime-qwen3-opd1k-sft4g`,
  `gpu:h200:4`, `time=18:00:00`, `Dependency=(null)`, running on `g018` at
  submission verification.
- OPD final eval job: `159410`, `slime-qwen3-opd1k-sft4g-eval`,
  `gpu:h200:4`, `time=05:00:00`, dependency `afterok:159409`.
- Base maybe-eval job: `159411`, `slime-qwen3-base-math500-maybe`,
  `gpu:h200:4`, `time=05:00:00`, dependency `afterok:159410`.
- Final report job: `159412`, `slime-qwen3-final-report-sft4g`,
  `gpu:h200:1`, `time=00:30:00`, dependency `afterok:159411`.
- Mail for all replacement jobs: `MailUser=suryadv@cs.washington.edu`,
  `MailType=END,FAIL`.
- Expected runtime validation: new OPD log should show Ray starts and accepts
  the job submission without a dashboard `504`, then SGLang worker processes
  show the startup native RoPE hook, memory-saver allocator override, no
  fused-RoPE NVCC/ninja failure, SFT HF snapshot load at `iter_0000096`,
  rollout `0`, and actor train.

## Corrected SFT-Weights Colocate Retry With Train-Actor Allocator Scoping

- Superseded retry: `159409`, `slime-qwen3-opd1k-sft4g`, started on
  `2026-07-03T14:04:05-07:00` and failed after `00:05:45`.
- Root cause: the Ray control-plane scoping patch worked: the Ray job was
  submitted successfully and the rollout SGLang engines started. The next
  failure was in Megatron train actor initialization. Because `--offload-train`
  preloads `torch_memory_saver`, train actors cannot inherit
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; `torch_memory_saver`
  aborted with `TorchMemorySaver is disabled for the current process because
  expandable_segments is not supported yet`.
- Progress decision: the job failed during actor initialization before any OPD
  rollout/training/checkpoint/HF snapshot/dataset state was produced. Restart
  from final SFT HF weights remains fidelity-safe.
- Patch behavior:
  - Train actors now apply the same non-expandable allocator compatibility
    override used by SGLang memory-saver rollout actors whenever
    `offload_train` is enabled for Megatron.
  - The top-level Ray job can keep `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`;
    the train actor runtime env overrides it only for actors that preload
    `torch_memory_saver`.
- Static validation before replacement submission:
  - `python3 -m py_compile slime/ray/actor_group.py slime/ray/utils.py`
    passed.
  - `git diff --check` passed.
  - Focused pytest could not run in the login environment because `pytest` is
    not installed there.
  - Full colocate4 dry check with `RUN_CONTAINER_CHECKS=1` passed, including
    Slurm `sbatch --test-only`, container imports, startup native shim, and the
    comma-env probe.
- Patch commit: `519f5cd` (`Scope allocator for train memory saver actors`),
  pushed to `origin/opd-reproduction`.
- Canceled stale jobs: `159410`, `159411`, `159412`.
- Replacement submit time: `2026-07-03T14:19:28-07:00`.
- Dependency policy: replacement train has no dependency; downstream jobs use
  `afterok`.
- OPD train job: `159422`, `slime-qwen3-opd1k-sft4g`,
  `gpu:h200:4`, `time=18:00:00`, `Dependency=(null)`, running on `g018` at
  submission verification.
- OPD final eval job: `159423`, `slime-qwen3-opd1k-sft4g-eval`,
  `gpu:h200:4`, `time=05:00:00`, dependency `afterok:159422`.
- Base maybe-eval job: `159424`, `slime-qwen3-base-math500-maybe`,
  `gpu:h200:4`, `time=05:00:00`, dependency `afterok:159423`.
- Final report job: `159425`, `slime-qwen3-final-report-sft4g`,
  `gpu:h200:1`, `time=00:30:00`, dependency `afterok:159424`.
- Mail for all replacement jobs: `MailUser=suryadv@cs.washington.edu`,
  `MailType=END,FAIL`.
- Expected runtime validation: new OPD log should show Ray job submission,
  SGLang rollout engine memory-saver allocator override, train actors no longer
  failing on expandable segments, SFT HF snapshot load at `iter_0000096`,
  rollout `0`, and actor train.

## Corrected SFT-Weights Colocate Retry With 6144 Actor Packing

- Superseded retry: `159422`, `slime-qwen3-opd1k-sft4g`, started on
  `2026-07-03T14:19:28-07:00` and failed after rollout `0` reached actor
  training.
- Root cause: the previous SGLang/compiler/container/Ray allocator fixes
  worked, and the job completed rollout generation, reference logprobs, and
  actor logprobs. Actor training then hit CUDA OOM in fused cross-entropy
  backward while allocating a `(3754, 1, 151936)` bf16 tensor: `1.06 GiB`
  requested with only about `1.45 GiB` free on GPU 1.
- Progress decision: no optimizer step, OPD checkpoint, HF snapshot, or
  dataset-state checkpoint completed. `rollout_0.pt` and
  `samples_0_127.json` are diagnostics only and must not be used as training
  progress. Restart from final SFT HF weights remains fidelity-safe.
- Patch behavior:
  - Lower only the colocate4 actor dynamic packing default from
    `OPD_MAX_TOKENS_PER_GPU=11264` to `OPD_MAX_TOKENS_PER_GPU=6144`.
  - Keep the 4-GPU corrected experiment shape unchanged: run label
    `1k_32k_sft_colocate4`, actor/rollout/teacher `3/3/1`, actor `TP=1`,
    `CP=3`, `OPD_SEQ_LENGTH=32766`, `OPD_MAX_RESPONSE_LEN=31744`, SFT HF
    initialization from `iter_0000096`, colocate/offload/recompute enabled,
    and native SGLang startup patches unchanged.
- Archived diagnostics before replacement submission:
  - `opd_1k_32k_sft_colocate4_rollout_logs/rollout_0.pt.failed_159422`
  - `opd_1k_32k_sft_colocate4_sanity/samples_0_127.json.failed_159422`
- Static validation before replacement submission:
  - `bash -n` on touched shell/sbatch scripts.
  - `git diff --check`.
  - Full colocate4 dry check with `RUN_CONTAINER_CHECKS=1` passed, including
    Slurm `sbatch --test-only`, container imports, startup native shim, and the
    comma-env probe.
- Patch commit: `050a961` (`Lower colocate4 OPD actor packing`), pushed to
  `origin/opd-reproduction`.
- Canceled stale jobs: `159423`, `159424`, `159425`.
- Replacement submit time: `2026-07-03T22:27:34-07:00`.
- Dependency policy: replacement train has no dependency; downstream jobs use
  `afterok`.
- OPD train job: `159763`, `slime-qwen3-opd1k-sft4g`,
  `gpu:h200:4`, `time=18:00:00`, `Dependency=(null)`, running on `g006` at
  submission verification.
- OPD final eval job: `159764`, `slime-qwen3-opd1k-sft4g-eval`,
  `gpu:h200:4`, `time=05:00:00`, dependency `afterok:159763`.
- Base maybe-eval job: `159765`, `slime-qwen3-base-math500-maybe`,
  `gpu:h200:4`, `time=05:00:00`, dependency `afterok:159764`.
- Final report job: `159766`, `slime-qwen3-final-report-sft4g`,
  `gpu:h200:1`, `time=00:30:00`, dependency `afterok:159765`.
- Mail for all replacement jobs: `MailUser=suryadv@cs.washington.edu`,
  `MailType=END,FAIL`.
- Expected runtime validation: new OPD log should show
  `OPD_MAX_TOKENS_PER_GPU=6144`, no previous SGLang native RoPE/gcc/clamp or
  allocator failures, SFT HF snapshot load at `iter_0000096`, rollout `0`,
  and actor train completing rollout `0` without the fused cross-entropy CUDA
  OOM.

## Corrected SFT-Weights Colocate Retry With 4096 Actor Packing

- Superseded retry: `159763`, `slime-qwen3-opd1k-sft4g`, started on
  `2026-07-03T22:27:35-07:00` and failed after rollout `0` reached actor
  training.
- Root cause: the `6144` actor packing patch worked enough to get farther:
  container/SGLang startup passed, SFT HF-load path was entered, rollout
  generation completed, reference logprobs and actor logprobs completed, and
  actor train reached microbatch `8/83`. Fused cross-entropy backward then hit
  CUDA OOM while allocating about `1.00 GiB` with about `1.91 GiB` free on GPU
  1 and the train memory-saver margin still at `1073741824` bytes.
- Progress decision: no optimizer step, OPD checkpoint, HF snapshot,
  dataset-state checkpoint, or trained-data manifest completed. `rollout_0.pt`
  and `samples_0_127.json` are diagnostics only and must not be used as
  training progress. Restart from final SFT HF weights remains fidelity-safe.
- Patch behavior:
  - Lower only the colocate4 actor dynamic packing default from
    `OPD_MAX_TOKENS_PER_GPU=6144` to `OPD_MAX_TOKENS_PER_GPU=4096`.
  - Keep `train-memory-margin-bytes` unchanged for this retry.
  - Keep the 4-GPU corrected experiment shape unchanged: run label
    `1k_32k_sft_colocate4`, actor/rollout/teacher `3/3/1`, actor `TP=1`,
    `CP=3`, `OPD_SEQ_LENGTH=32766`, `OPD_MAX_RESPONSE_LEN=31744`, SFT HF
    initialization from `iter_0000096`, colocate/offload/recompute enabled,
    and native SGLang startup patches unchanged.
- Archived diagnostics before replacement submission:
  - `opd_1k_32k_sft_colocate4_rollout_logs/rollout_0.pt.failed_159763`
  - `opd_1k_32k_sft_colocate4_sanity/samples_0_127.json.failed_159763`
- Static validation before replacement submission:
  - `bash -n` on touched shell/sbatch scripts.
  - `git diff --check`.
  - Full colocate4 dry check with `RUN_CONTAINER_CHECKS=1` passed, including
    Slurm `sbatch --test-only`, container imports, startup native shim, and the
    comma-env probe.
- Patch commit: `8e0b414` (`Lower colocate4 OPD packing to 4096`), pushed to
  `origin/opd-reproduction`.
- Canceled stale jobs: `159764`, `159765`, `159766`.
- Replacement submit time: `2026-07-04T01:34:18-07:00`.
- Dependency policy: replacement train has no dependency; downstream jobs use
  `afterok`.
- OPD train job: `159941`, `slime-qwen3-opd1k-sft4g`,
  `gpu:h200:4`, `time=18:00:00`, `Dependency=(null)`, pending for resources
  with scheduler node list `g012` at submission verification.
- OPD final eval job: `159942`, `slime-qwen3-opd1k-sft4g-eval`,
  `gpu:h200:4`, `time=05:00:00`, dependency `afterok:159941`.
- Base maybe-eval job: `159943`, `slime-qwen3-base-math500-maybe`,
  `gpu:h200:4`, `time=05:00:00`, dependency `afterok:159942`.
- Final report job: `159944`, `slime-qwen3-final-report-sft4g`,
  `gpu:h200:1`, `time=00:30:00`, dependency `afterok:159943`.
- Mail for all replacement jobs: `MailUser=suryadv@cs.washington.edu`,
  `MailType=END,FAIL`.
- Expected runtime validation: new OPD log should show
  `OPD_MAX_TOKENS_PER_GPU=4096`, SFT HF load from `iter_0000096`, no silent
  base fallback, no previous SGLang native RoPE/gcc/clamp or allocator
  failures, rollout `0`, and actor train completing rollout `0` without fused
  cross-entropy CUDA OOM.

## Corrected SFT-Weights Colocate Retry With Zero Train Memory Margin

- Superseded retry: `159941`, `slime-qwen3-opd1k-sft4g`, started on
  `2026-07-04T02:06:35-07:00` and failed after rollout `0` reached actor
  training.
- Root cause: `OPD_MAX_TOKENS_PER_GPU=4096` was applied, but actor training
  still hit fused cross-entropy backward CUDA OOM at microbatch `2/82`.
  Torch-memory-saver repeatedly refused allocations with
  `memory_margin_bytes=1073741824`, then PyTorch failed a `1.00 GiB`
  allocation with about `1.68 GiB` free on GPU 0.
- Progress decision: no optimizer step, OPD checkpoint, HF snapshot,
  dataset-state checkpoint, or trained-data manifest completed. `rollout_0.pt`
  and `samples_0_127.json` are diagnostics only and must not be used as
  training progress. Restart from final SFT HF weights remains fidelity-safe.
- Patch behavior:
  - Add `OPD_TRAIN_MEMORY_MARGIN_BYTES`, globally defaulting to Megatron's
    current `1073741824` byte train memory-saver margin.
  - Set `OPD_TRAIN_MEMORY_MARGIN_BYTES=0` only in the colocate4 submitter.
  - Forward the variable through the container and pass it to training as
    `--train-memory-margin-bytes`.
  - Keep the 4-GPU corrected experiment shape unchanged: run label
    `1k_32k_sft_colocate4`, actor/rollout/teacher `3/3/1`, actor `TP=1`,
    `CP=3`, `OPD_SEQ_LENGTH=32766`, `OPD_MAX_RESPONSE_LEN=31744`,
    `OPD_MAX_TOKENS_PER_GPU=4096`, SFT HF initialization from
    `iter_0000096`, colocate/offload/recompute enabled, and native SGLang
    startup patches unchanged.
- Archived diagnostics before replacement submission:
  - `opd_1k_32k_sft_colocate4_rollout_logs/rollout_0.pt.failed_159941`
  - `opd_1k_32k_sft_colocate4_sanity/samples_0_127.json.failed_159941`
- Static validation before replacement submission:
  - `bash -n` on touched shell/sbatch scripts passed.
  - `git diff --check` passed.
  - Full colocate4 dry check with `RUN_CONTAINER_CHECKS=1` passed, including
    Slurm `sbatch --test-only`, container imports, startup native shim, and the
    comma-env probe.
- Patch commit: `ac6663d` (`Set zero train memory margin for colocate4 OPD`),
  pushed to `origin/opd-reproduction`.
- Canceled stale jobs: `159942`, `159943`, `159944`.
- Replacement submit time: `2026-07-04T11:25:40-07:00`.
- Dependency policy: replacement train has no dependency; downstream jobs use
  `afterok`.
- OPD train job: `160260`, `slime-qwen3-opd1k-sft4g`,
  `gpu:h200:4`, `time=18:00:00`, `Dependency=(null)`, running on `g003` at
  submission verification.
- OPD final eval job: `160261`, `slime-qwen3-opd1k-sft4g-eval`,
  `gpu:h200:4`, `time=05:00:00`, dependency `afterok:160260`.
- Base maybe-eval job: `160262`, `slime-qwen3-base-math500-maybe`,
  `gpu:h200:4`, `time=05:00:00`, dependency `afterok:160261`.
- Final report job: `160263`, `slime-qwen3-final-report-sft4g`,
  `gpu:h200:1`, `time=00:30:00`, dependency `afterok:160262`.
- Mail for all replacement jobs: `MailUser=suryadv@cs.washington.edu`,
  `MailType=END,FAIL`.
- Initial runtime check: `160260` log shows `OPD train memory margin bytes: 0`;
  deeper runtime validation is still pending while the job runs.
- Expected runtime validation: new OPD log should show
  `OPD_MAX_TOKENS_PER_GPU=4096`, `OPD_TRAIN_MEMORY_MARGIN_BYTES=0`, parsed
  `train_memory_margin_bytes ....................... 0`, SFT HF load from
  `iter_0000096`, no silent base fallback, no previous SGLang native
  RoPE/gcc/clamp or allocator failures, rollout `0`, and actor train
  completing rollout `0` without fused cross-entropy CUDA OOM.

## Corrected SFT-Weights Colocate Retry With 512 Logprob Chunking

- Superseded retry: `160260`, `slime-qwen3-opd1k-sft4g`, started on
  `2026-07-04T11:25:40-07:00` and failed after rollout `0` reached actor
  training.
- Root cause: `OPD_TRAIN_MEMORY_MARGIN_BYTES=0` worked, but actor training
  still hit fused vocab cross-entropy/logprob CUDA OOM at microbatch `23/88`.
  The failing temporary was a `float32` full-vocab buffer that tried to allocate
  `2.00 GiB` with about `1.64 GiB` free on GPU 2.
- Progress decision: no optimizer step, OPD checkpoint, HF snapshot,
  dataset-state checkpoint, or trained-data manifest completed. `rollout_0.pt`
  and `samples_0_127.json` are diagnostics only and must not be used as
  training progress. Restart from final SFT HF weights remains fidelity-safe.
- Patch behavior:
  - Add `OPD_LOG_PROBS_CHUNK_SIZE`, globally defaulting to the previous
    `4096`.
  - Set `OPD_LOG_PROBS_CHUNK_SIZE=512` only in the colocate4 submitter and
    pass it as `--log-probs-chunk-size`.
  - Lower colocate4 `OPD_MAX_TOKENS_PER_GPU` from `4096` to `2048`.
  - Keep `OPD_TRAIN_MEMORY_MARGIN_BYTES=0`.
  - Keep the 4-GPU corrected experiment shape unchanged: run label
    `1k_32k_sft_colocate4`, actor/rollout/teacher `3/3/1`, actor `TP=1`,
    `CP=3`, `OPD_SEQ_LENGTH=32766`, `OPD_MAX_RESPONSE_LEN=31744`, SFT HF
    initialization from `iter_0000096`, colocate/offload/recompute enabled,
    and native SGLang startup patches unchanged.
- Archived diagnostics before replacement submission:
  - `opd_1k_32k_sft_colocate4_rollout_logs/rollout_0.pt.failed_160260`
  - `opd_1k_32k_sft_colocate4_sanity/samples_0_127.json.failed_160260`
- Static validation before replacement submission:
  - `bash -n` on touched shell/sbatch scripts passed.
  - `git diff --check` passed.
  - Full colocate4 dry check with `RUN_CONTAINER_CHECKS=1` passed, including
    Slurm `sbatch --test-only`, container imports, startup native shim, and the
    comma-env probe.
- Patch commit: `9d89b0d` (`Reduce colocate4 OPD logprob memory`), pushed to
  `origin/opd-reproduction`.
- Canceled stale jobs: `160261`, `160262`, `160263`.
- Replacement submit time: `2026-07-04T13:07:15-07:00`.
- Dependency policy: replacement train has no dependency; downstream jobs use
  `afterok`.
- OPD train job: `160279`, `slime-qwen3-opd1k-sft4g`,
  `gpu:h200:4`, `time=18:00:00`, `Dependency=(null)`, pending for resources
  with scheduler node list `g002` and projected start
  `2026-07-05T03:10:18` at submission verification.
- OPD final eval job: `160280`, `slime-qwen3-opd1k-sft4g-eval`,
  `gpu:h200:4`, `time=05:00:00`, dependency `afterok:160279`.
- Base maybe-eval job: `160281`, `slime-qwen3-base-math500-maybe`,
  `gpu:h200:4`, `time=05:00:00`, dependency `afterok:160280`.
- Final report job: `160282`, `slime-qwen3-final-report-sft4g`,
  `gpu:h200:1`, `time=00:30:00`, dependency `afterok:160281`.
- Mail for all replacement jobs: `MailUser=suryadv@cs.washington.edu`,
  `MailType=END,FAIL`.
- Expected runtime validation: new OPD log should show
  `OPD_MAX_TOKENS_PER_GPU=2048`, `OPD_LOG_PROBS_CHUNK_SIZE=512`, actual
  `--log-probs-chunk-size 512`, parsed `train_memory_margin_bytes=0`, SFT HF
  load from `iter_0000096`, no silent base fallback, rollout `0`, and actor
  train completing rollout `0` without fused cross-entropy/logprob CUDA OOM.

## Corrected SFT-Weights Colocate Retry With Non-Fatal Sanity

- Superseded retry: `160279`, `slime-qwen3-opd1k-sft4g`, started on
  `2026-07-04` and reached corrected SFT-loaded colocate4 OPD training.
- Progress observed: rollouts `0..3` completed actor training. Rollout `4`
  generated successfully but did not train because the old OPD sanity guard
  hard-stopped before reward postprocessing/training.
- Trigger: rollout `4` had `avg_response_tokens=16131.8125`,
  `median_response_tokens=16579`, `min_response_tokens=2`,
  `max_response_tokens=31744`, `cap_hit_rate=0.1640625`,
  `completed_rate=0.8359375`, and `final_answer_rate=0.984375`.
- Healthy loader/sync signal before the hard-stop: Megatron-vs-SGLang absolute
  logprob differences for rollouts `0..3` were `0.0216`, `0.0232`, `0.0226`,
  and `0.0217`, not the older accidental-run pathological `~11.9`.
- Progress decision: no final OPD optimizer checkpoint, HF snapshot,
  dataset-state checkpoint, or trained-data manifest completed. Rollout and
  sanity JSON files from `160279` are diagnostics only and must not be used as
  training progress. Restart from final SFT HF weights remains fidelity-safe.
- Patch behavior:
  - Keep `OPD_SANITY_CHECK_ENABLED=1` but set corrected colocate4
    `OPD_SANITY_FAIL_ON_COLLAPSE=0`, so violations are logged/reported and do
    not stop training.
  - Add thresholds, `violations`, and `sanity_passed` to every `OPD_SANITY`
    JSON record.
  - Add `summarize_opd_sanity.py`, which writes
    `opd_sanity_summary.{json,csv,md}` from sanity JSON and train-log metrics.
    Its tables define columns and state that logprob/KL/loss diagnostics are
    response-token-weighted rollout means; response lengths are sample
    statistics; cap/completion/final-answer rates are sample fractions.
  - Run the sanity summary helper best-effort from the OPD train wrapper on
    exit, so a table is produced even if training later fails.
  - Set corrected colocate4 `OPD_REF_LOAD_DIR` to the final SFT HF snapshot
    `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/qwen3_8b_sft_25k_eval_snapshots/iter_0000096`.
    The reference model for logged `train/kl_loss` therefore matches final SFT.
    `kl_loss_coef=0.00`, so this changes diagnostics only, not gradients.
  - Copy/link the sanity summary into the combined report output when report
    aggregation runs.
- Documentation added:
  `examples/qwen3_8b_opd_tillicum/results/corrected_colocate4_diagnostics.md`
  records the `160279` trigger table, dataset facts used in the long-trace
  analysis, the accidental base -> OPD loader/ref-sync issue, metric averaging
  rules, and the OPD pipeline/debugging semantics.
- Patch commit: `91233ff` (`Make colocate4 sanity nonfatal`), pushed to
  `origin/opd-reproduction` before replacement submission.
- Static validation before replacement submission:
  - `python3 -m py_compile` on touched Python files passed.
  - `bash -n` on touched shell/sbatch scripts passed.
  - `git diff --check` passed.
  - Full colocate4 dry check with `RUN_CONTAINER_CHECKS=1` passed, including
    Slurm `sbatch --test-only`, container imports, native SGLang shim, and the
    comma-env forwarding probe.
  - `summarize_opd_sanity.py` successfully regenerated a `160279` sanity
    table in `/tmp/opd_sanity_160279_check` and marked rollout `4` violated
    but nonfatal.
- Archived diagnostics before replacement submission:
  - `opd_1k_32k_sft_colocate4_rollout_logs/rollout_0.pt.failed_160279`
  - `opd_1k_32k_sft_colocate4_rollout_logs/rollout_1.pt.failed_160279`
  - `opd_1k_32k_sft_colocate4_rollout_logs/rollout_2.pt.failed_160279`
  - `opd_1k_32k_sft_colocate4_rollout_logs/rollout_3.pt.failed_160279`
  - `opd_1k_32k_sft_colocate4_rollout_logs/rollout_4.pt.failed_160279`
  - `opd_1k_32k_sft_colocate4_sanity/samples_0_127.json.failed_160279`
  - `opd_1k_32k_sft_colocate4_sanity/samples_128_255.json.failed_160279`
  - `opd_1k_32k_sft_colocate4_sanity/samples_256_383.json.failed_160279`
  - `opd_1k_32k_sft_colocate4_sanity/samples_384_511.json.failed_160279`
  - `opd_1k_32k_sft_colocate4_sanity/samples_512_639.json.failed_160279`
- Canceled stale downstream jobs: `160280`, `160281`, and `160282`.
- Replacement submit time: `2026-07-04T20:42:36-07:00`.
- Replacement dependency policy: train has no dependency; downstream jobs use
  `afterok`.
- OPD train job: `160424`, `slime-qwen3-opd1k-sft4g`, `gpu:h200:4`,
  `time=18:00:00`, `Dependency=(null)`, running on `g018` at submission
  verification.
- OPD final eval job: `160425`, `slime-qwen3-opd1k-sft4g-eval`,
  `gpu:h200:4`, `time=05:00:00`, dependency `afterok:160424`.
- Base maybe-eval job: `160426`, `slime-qwen3-base-math500-maybe`,
  `gpu:h200:4`, `time=05:00:00`, dependency `afterok:160425`.
- Final report job: `160427`, `slime-qwen3-final-report-sft4g`,
  `gpu:h200:1`, `time=00:30:00`, dependency `afterok:160426`.
- Mail for all replacement jobs: `MailUser=suryadv@cs.washington.edu`,
  `MailType=END,FAIL`.
- OPD initial load and reference load:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/qwen3_8b_sft_25k_eval_snapshots/iter_0000096`.
- OPD save dir:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/qwen3_8b_sft_25k_opd_1k_32k_sft_colocate4_full_optim`.
- OPD eval output:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/math500_eval_opd_1k_32k_sft_colocate4_final`.
- Combined final report output:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/math500_eval_combined_25k_opd_1k_32k_sft_colocate4`.
- Expected runtime validation: new OPD log should show
  `OPD_SANITY_CHECK_ENABLED=1`, `OPD_SANITY_FAIL_ON_COLLAPSE=0`,
  `OPD_REF_LOAD_DIR` pointing at final SFT HF `iter_0000096`, SFT HF actor
  load from `iter_0000096`, logged SFT-reference KL rather than base-reference
  KL, small Megatron-vs-SGLang abs diff, and continuation past any sanity
  violation instead of a hard-stop.

## Corrected SFT-Weights Colocate Final Result

- Jobs completed successfully on `2026-07-05`:
  - OPD train `160424`: `COMPLETED`, `0:0`, elapsed `05:40:59`.
  - OPD eval `160425`: `COMPLETED`, `0:0`, elapsed `04:33:04`.
  - maybe-base eval `160426`: `COMPLETED`, `0:0`, elapsed `00:00:05`.
  - final report `160427`: `COMPLETED`, `0:0`, elapsed `00:00:03`.
- Final full OPD checkpoint:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/qwen3_8b_sft_25k_opd_1k_32k_sft_colocate4_full_optim/iter_0000007`.
- Final OPD HF snapshot:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/qwen3_8b_sft_25k_opd_1k_32k_sft_colocate4_eval_snapshots/iter_0000007`.
- Trained-data manifest:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/qwen3_8b_sft_25k_opd_1k_32k_sft_colocate4_full_optim/opd_trained_manifest.json`.
- Checkpoint size reported by the wrapper:
  `114672812172` bytes for `iter_0000007`.
- Final MATH-500 points:
  - Base: `accuracy=0.638`, `accuracy_on_parseable=0.7595238095`,
    `parse_failure_rate=0.160`, `cap_hit_rate=0.036`,
    `avg_generated_tokens=1686.402`.
  - Final SFT: `accuracy=0.750`, `accuracy_on_parseable=0.78125`,
    `parse_failure_rate=0.040`, `cap_hit_rate=0.576`,
    `avg_generated_tokens=18574.302`.
  - SFT + OPD 1024: `accuracy=0.724`,
    `accuracy_on_parseable=0.7685774947`, `parse_failure_rate=0.058`,
    `cap_hit_rate=0.832`, `avg_generated_tokens=26567.282`.
- OPD rollout diagnostics:
  - Mean response tokens over rollout means: `15845.8916`
    (`min=14218.9375`, `max=18577.8125`).
  - Mean `opd_reverse_kl`: `0.485068`.
  - Mean Megatron actor vs SGLang rollout abs logprob diff: `0.022237`.
  - Mean logged SFT-reference KL loss: `0.001396`.
  - Rollouts `5` and `7` exceeded the `16000` average-token sanity threshold
    but continued/trained because `OPD_SANITY_FAIL_ON_COLLAPSE=0`.
- Tracked result artifacts were added under
  `examples/qwen3_8b_opd_tillicum/results/corrected_sft_opd_colocate4_1k_32k/`,
  including `summary_all.json`, `combined_accuracy_curve.csv`,
  `combined_accuracy_curve.svg`, `opd_001024_summary.json`, and
  `opd_sanity_summary.{json,csv,md}`.

## Planned 1k -> 25k OPD Continuation With Val100

- Purpose: continue the completed corrected SFT -> OPD colocate4 run from the
  full optimizer checkpoint at OPD rollout `7` / `1,024` samples to rollout
  `194` / `24,960` total OPD samples.
- Submitter: `submit_opd_continue_1k_to_25k_val100_chain.sh`.
- Starting full optimizer checkpoint:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/qwen3_8b_sft_25k_opd_1k_32k_sft_colocate4_full_optim/iter_0000007`.
- Starting HF snapshot for the first val100 eval:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/qwen3_8b_sft_25k_opd_1k_32k_sft_colocate4_eval_snapshots/iter_0000007`.
- Continuation policy: load model/optimizer from the old OPD full checkpoint,
  but skip the old rollout dataset-state load for the first continuation
  segment so the new no-repeat continuation pool starts at offset `0`.
  Subsequent segments resume from the new continuation save dir and load the
  new dataset state normally.
- No-repeat data policy: extract the actual `1,024` trained OPD source row ids
  from the successful rollout debug files, then generate continuation OPD rows
  excluding both those rows and all 25k SFT source row ids.
- Eval policy: run seeded MATH-500 val100 on `opd_001024` first, then after
  every 1k-ish endpoint through `opd_024960`; run a parallel 1-GPU full
  MATH-500 eval at `opd_012544`.
- Resource policy: OPD continuation jobs use `gpu:h200:4`; val100 and
  midpoint full eval jobs use `gpu:h200:1`; final report uses `gpu:h200:1`.

## Submitted 1k -> 25k OPD Continuation With Val100

- Implementation commit: `24359e6` (`Add OPD 25k continuation val100 chain`),
  pushed to `origin/opd-reproduction` before submission.
- Submit time/log: `2026-07-05 23:55 PDT`,
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/slurm_logs/submit_opd_continue_1k_to_25k_val100_20260705_235549.txt`.
- Scheduler validation after submission: data job pending for `Priority`;
  downstream sampled jobs pending on expected dependencies; no sampled job is
  `DependencyNeverSatisfied`; sampled jobs have
  `MailUser=suryadv@cs.washington.edu` and `MailType=END,FAIL`.
- Pre-submit data validation: container extractor read exactly `1,024` unique
  old OPD trained `source_row_id`s from
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/opd_1k_32k_sft_colocate4_rollout_logs`.
- Output paths:
  - Continuation data:
    `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/data/openthoughts3_math_opd_continue_1k_to_25k_seed1234.jsonl`.
  - Continuation metadata:
    `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/data/openthoughts3_math_sft_25000_opd_continue_1k_to_25k_seed1234_metadata.json`.
  - Val100 eval:
    `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/math500_eval_opd_25k_32k_sft_colocate4_continue_val100`.
  - Midpoint full500 eval:
    `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/math500_eval_opd_25k_32k_sft_colocate4_continue_mid_full500`.
  - Continuation save dir:
    `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/qwen3_8b_sft_25k_opd_25k_32k_sft_colocate4_continue_full_optim`.
  - Continuation HF snapshots:
    `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/qwen3_8b_sft_25k_opd_25k_32k_sft_colocate4_continue_eval_snapshots`.
- Job IDs:
  - Data prep: `161481`.
  - Current 1k val100: `161482`.
  - Train/eval pairs:
    `2048=(161483,161484)`, `3072=(161485,161486)`,
    `4096=(161487,161488)`, `5120=(161489,161490)`,
    `6144=(161491,161492)`, `7168=(161493,161494)`,
    `8192=(161495,161496)`, `9216=(161497,161498)`,
    `10240=(161499,161500)`, `11264=(161501,161502)`,
    `12288=(161503,161504)`, `12544=(161505,161506)`,
    `13312=(161508,161509)`, `14336=(161510,161511)`,
    `15360=(161512,161513)`, `16384=(161514,161515)`,
    `17408=(161516,161517)`, `18432=(161518,161519)`,
    `19456=(161520,161521)`, `20480=(161522,161523)`,
    `21504=(161524,161525)`, `22528=(161526,161527)`,
    `23552=(161528,161529)`, `24576=(161530,161531)`,
    `24960=(161532,161533)`.
  - Midpoint full500 at `12,544`: `161507`.
  - Final report: `161534`, dependency `afterany:161533:161507`.
- Runtime validation targets:
  - `161481` writes val100 JSONL/config/metadata and continuation data with
    `SFT ∩ new_OPD = ∅` and `old_1k_OPD ∩ new_OPD = ∅`.
  - `161482` writes `opd_001024/summary.json` with exactly 100 samples.
  - `161483` logs old full optimizer load from `iter_0000007`,
    `OPD_SKIP_ROLLOUT_DATA_STATE_LOAD=1`, and rollout start at `8`.
  - Later train jobs log normal dataset-state resume from the continuation
    save dir.

## OPD 25k Continuation Resume Patch and Replacement Chain

- Latest failure: first continuation train job `161483` failed before rollout
  or training while loading the completed 1k OPD full optimizer checkpoint.
- Fatal error:
  `OptimizerParamScheduler: class input value 24960 and checkpointvalue 1024 for total number of iterations do not match`.
- Root cause:
  - The old 1k OPD optimizer checkpoint persisted an LR scheduler horizon for
    `1,024` samples.
  - The continuation submitter had also leaked final-run derived env values
    through `--export=ALL`, so the first segment logged the final rollout
    horizon instead of endpoint-specific values.
- Preserved progress:
  - Data prep `161481` completed and is reused.
  - Current 1k val100 `161482` completed and is reused:
    `accuracy=0.700`, `accuracy_on_parseable=0.7526881720`,
    `parse_failure_rate=0.070`, `cap_hit_rate=0.840`.
  - No output from failed train `161483` is used as progress; the continuation
    save dir still had no `latest_checkpointed_iteration.txt`.
- Patch commit: `05e238b` (`Fix OPD continuation segmented resume`), pushed to
  `origin/opd-reproduction` before replacement submission.
- Patch behavior:
  - Continuation train jobs set
    `OPD_ALLOW_OPT_PARAM_SCHEDULER_MISMATCH=1`, which passes Megatron's
    `--override-opt-param-scheduler`; model weights and optimizer state still
    load from the 1k full checkpoint, while the freshly constructed scheduler
    uses the current continuation segment horizon.
  - The submitter now explicitly exports per-endpoint derived values:
    `OPD_TRAIN_SIZE`, `OPD_NUM_ROLLOUT`, `OPD_FINAL_ROLLOUT_ID`,
    `OPD_EFFECTIVE_TRAIN_SAMPLES`, `OPD_FINAL_HF_DIR`,
    `OPD_MILESTONE_ROLLOUT_IDS`, and `OPD_SANITY_MAX_ROLLOUT_ID`.
  - The submitter can reuse existing data/current-val artifacts; for this retry
    `OPD_CONTINUE_SKIP_PREP_AND_CURRENT_VAL=1`.
- Validation before submission:
  - `bash -n` on touched shell/sbatch scripts: passed.
  - `python3 -m py_compile` on touched/related Python helpers: passed.
  - `git diff --check`: passed.
  - `RUN_CONTAINER_CHECKS=1 bash examples/qwen3_8b_opd_tillicum/run_all_dry_check.sh`: passed.
  - Targeted container CLI check confirmed Megatron exposes
    `--override-opt-param-scheduler`.
- Canceled stale downstream jobs from the failed chain:
  `161484` through `161534`.
- Replacement submit time/log: `2026-07-06 12:38 PDT`,
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/slurm_logs/submit_opd_continue_1k_to_25k_val100_20260706_123823.txt`.
- Replacement job IDs:
  - Preserved data/current val:
    `data=preserved_existing_data`,
    `val100_current_001024=preserved_existing_opd_001024_val100`.
  - Train/eval pairs:
    `2048=(161751,161752)`, `3072=(161753,161754)`,
    `4096=(161755,161756)`, `5120=(161757,161758)`,
    `6144=(161759,161760)`, `7168=(161761,161762)`,
    `8192=(161763,161764)`, `9216=(161765,161766)`,
    `10240=(161767,161768)`, `11264=(161769,161770)`,
    `12288=(161771,161772)`, `12544=(161773,161774)`,
    `13312=(161776,161777)`, `14336=(161778,161779)`,
    `15360=(161780,161781)`, `16384=(161782,161783)`,
    `17408=(161784,161785)`, `18432=(161786,161787)`,
    `19456=(161788,161789)`, `20480=(161790,161791)`,
    `21504=(161792,161793)`, `22528=(161794,161795)`,
    `23552=(161796,161797)`, `24576=(161798,161799)`,
    `24960=(161800,161801)`.
  - Midpoint full500 at `12,544`: `161775`.
  - Final report: `161802`, dependency `afterany:161801:161775`.
- Scheduler validation after replacement submission:
  - First replacement train `161751` is pending for `Priority` with no
    dependency and estimated start `2026-07-06 16:39 PDT`.
  - `scontrol show job -o 161751` shows endpoint-specific exports:
    `OPD_TRAIN_SIZE=2048`, `OPD_NUM_ROLLOUT=16`,
    `OPD_FINAL_ROLLOUT_ID=15`, `OPD_EFFECTIVE_TRAIN_SAMPLES=2048`,
    `OPD_SKIP_ROLLOUT_DATA_STATE_LOAD=1`, and
    `OPD_ALLOW_OPT_PARAM_SCHEDULER_MISMATCH=1`.
  - Representative jobs `161751`, `161752`, `161775`, and `161802` have
    `MailUser=suryadv@cs.washington.edu` and `MailType=END,FAIL`.
  - No replacement job is `DependencyNeverSatisfied` at submission check.
- Runtime validation targets:
  - `161751` log shows full optimizer load from old OPD `iter_0000007`.
  - `161751` log shows
    `Allowing optimizer-parameter scheduler horizon changes via --override-opt-param-scheduler`.
  - `161751` starts rollout id `8` and writes first continuation checkpoint/HF
    snapshot `iter_0000015`.
  - `161775` writes the 1-GPU full MATH-500 midpoint summary for
    `opd_012544`.
  - `161801` writes the final val100 summary for `opd_024960`.

## Planned Serialized 4-GPU Val100/Eval Replacement for OPD 25k Continuation

- Reason for patch:
  - First replacement train `161751` completed and produced the valid
    continuation checkpoint/HF snapshot for `opd_002048` at rollout `15`.
  - Follow-up val100 `161752` was running as a 1-GPU eval and had not yet
    written `opd_002048/summary.json` or a complete `debug_eval_0.pt` when
    inspected, so there was no fidelity-safe partial artifact to append.
  - Downstream jobs `161753` through `161802` were still the old serialized
    train/1-GPU-val chain.
- Patch behavior:
  - Continuation val100 and midpoint full MATH-500 eval jobs now default to
    `gpu:h200:4`, `EVAL_ROLLOUT_NUM_GPUS=4`,
    `EVAL_ROLLOUT_BATCH_SIZE=64`, eval `TP=1`, eval `CP=1`, and
    `EVAL_SGLANG_SERVER_CONCURRENCY=2`.
  - Report-only jobs use `gpu:h200:1`.
  - The submitter detects a completed continuation endpoint checkpoint. If its
    val100 summary is missing, it preserves the train checkpoint and submits
    only that endpoint's val100. If the summary already exists, it skips the
    endpoint entirely.
  - The midpoint full MATH-500 eval at `opd_012544` now depends on
    `val100_012544`, and later training depends on the full eval, preventing
    any intentional overlap between training, val100, full eval, and report
    jobs.
- Preservation policy for `161752`:
  - If `opd_002048/summary.json` appears before cancellation, preserve it.
  - Else if a complete `debug_eval_0.pt` appears, summarize and preserve it.
  - Otherwise cancel `161752`; partial in-memory generations are not reused.
- Resource policy after patch:
  - OPD train: `gpu:h200:4`.
  - Val100 and midpoint full MATH-500 eval: `gpu:h200:4`.
  - Report: `gpu:h200:1`.
  - Dependencies are strict enough that no chain job should run in parallel
    with another chain train/eval/report job.

## Submitted Serialized 4-GPU Val100/Eval Replacement for OPD 25k Continuation

- Patch commit: `13ef32c` (`Serialize OPD continuation 4-GPU evals`), pushed
  to `origin/opd-reproduction` before replacement submission.
- Validation before submission:
  - `bash -n examples/qwen3_8b_opd_tillicum/submit_opd_continue_1k_to_25k_val100_chain.sh`: passed.
  - `git diff --check`: passed.
  - `RUN_CONTAINER_CHECKS=1 bash examples/qwen3_8b_opd_tillicum/run_all_dry_check.sh`: passed.
- Salvage outcome:
  - `opd_002048/summary.json` and `opd_002048/debug_eval_0.pt` were still
    absent immediately before cancellation.
  - Canceled incomplete 1-GPU val100 `161752` and stale downstream jobs
    `161753` through `161802`.
  - Preserved completed `opd_002048` train checkpoint/HF snapshot:
    `iter_0000015`.
- Replacement submit time/log: `2026-07-06 20:17 PDT`,
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/slurm_logs/submit_opd_continue_1k_to_25k_val100_20260706_201734.txt`.
- Replacement job IDs:
  - Preserved data/current val:
    `data=preserved_existing_data`,
    `val100_current_001024=preserved_existing_opd_001024_val100`.
  - Preserved 2k train:
    `train_2048=preserved_existing_train_2048`.
  - Val/train chain:
    `val100_2048=162217`, `3072=(162218,162219)`,
    `4096=(162220,162221)`, `5120=(162222,162223)`,
    `6144=(162224,162225)`, `7168=(162226,162227)`,
    `8192=(162228,162229)`, `9216=(162230,162231)`,
    `10240=(162232,162233)`, `11264=(162234,162235)`,
    `12288=(162236,162237)`, `12544=(162238,162239)`,
    `13312=(162241,162242)`, `14336=(162243,162244)`,
    `15360=(162245,162246)`, `16384=(162247,162248)`,
    `17408=(162249,162250)`, `18432=(162251,162252)`,
    `19456=(162253,162254)`, `20480=(162255,162256)`,
    `21504=(162257,162258)`, `22528=(162259,162260)`,
    `23552=(162261,162262)`, `24576=(162263,162264)`,
    `24960=(162265,162266)`.
  - Midpoint full500 at `12,544`: `162240`.
  - Final report: `162267`, dependency `afterany:162266:162240`.
- Scheduler validation after submission:
  - First replacement job `162217` is a 4-GPU val100 with no dependency,
    pending for `Resources`, with scheduler start estimate
    `2026-07-07 00:15 PDT`.
  - `162218` waits on `afterok:162217`, so `opd_003072` training cannot start
    until the 2k val100 finishes.
  - `162240` waits on `afterok:162239`, and `162241` waits on
    `afterok:162240`; the midpoint full500 eval is serialized between
    `val100_012544` and later training.
  - `162217`, `162218`, `162239`, `162240`, and `162241` each request
    `gpu:h200:4`; final report `162267` requests `gpu:h200:1`.
  - Representative jobs have `MailUser=suryadv@cs.washington.edu` and
    `MailType=END,FAIL`.
  - No replacement job was `DependencyNeverSatisfied` at submission check.
- Runtime validation targets:
  - `162217` log shows `EVAL_ROLLOUT_NUM_GPUS=4`, eval `TP=1`, eval `CP=1`,
    `EVAL_ROLLOUT_BATCH_SIZE=64`, and
    `EVAL_SGLANG_SERVER_CONCURRENCY=2`.
  - `162217` writes `opd_002048/summary.json` with exactly 100 samples.
  - `162218` starts `opd_003072`; it must not rerun the completed
    `opd_002048` train segment.
  - `162240` writes the 4-GPU full MATH-500 midpoint summary for
    `opd_012544`.
  - `162266` writes the final val100 summary for `opd_024960`, then `162267`
    refreshes the report.

## Planned OPD 25k Continuation Context-Overflow Patch

- Latest continuation failure: train job `162222` for endpoint `opd_005120`
  failed during rollout generation after `03:44:29`.
- Fatal SGLang error:
  `Requested token count exceeds the model's maximum context length of 32768 tokens. You requested a total of 32893 tokens: 1149 tokens from the input messages and 31744 tokens for the completion.`
- Root cause:
  - OPD requested the global `OPD_MAX_RESPONSE_LEN=31744` for every prompt.
  - A continuation prompt with `1,149` tokens needed a per-sample response cap
    of at most `32766 - 1149 = 31617` to stay inside the actor rollout context
    ceiling aligned with `OPD_SEQ_LENGTH=32766`.
  - The router then returned HTTP `400`, and later requests degraded into
    `503 no_available_workers`.
- Preserved progress:
  - Full optimizer checkpoint:
    `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/qwen3_8b_sft_25k_opd_25k_32k_sft_colocate4_continue_full_optim/iter_0000031`.
  - HF eval snapshot:
    `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/qwen3_8b_sft_25k_opd_25k_32k_sft_colocate4_continue_eval_snapshots/iter_0000031`.
  - Val100 summaries through `opd_004096`: `0.70`, `0.72`, `0.77`, `0.74`.
- Non-progress diagnostics:
  - Failed-job rollout logs `rollout_32.pt` through `rollout_37.pt`.
  - Failed-job sanity JSONs `samples_3072_3199.json` through
    `samples_3712_3839.json`.
  - These are not used as training progress because no matching full optimizer
    checkpoint was written after `iter_0000031`.
- Patch behavior:
  - `slime/rollout/sglang_rollout.py` clamps each sample's
    `max_new_tokens` to `rollout_max_context_len - prompt_tokens` when
    `--rollout-max-context-len` is set.
  - `OPD_ROLLOUT_MAX_CONTEXT_LEN` defaults to `OPD_SEQ_LENGTH` and is passed to
    OPD train jobs as `--rollout-max-context-len`.
  - Short prompts still receive the global `OPD_MAX_RESPONSE_LEN=31744`; only
    prompts that would overflow are capped lower.
- Planned replacement:
  - Cancel stale downstream jobs `162223` through `162267`.
  - Archive the failed diagnostics above with `.failed_162222` suffixes.
  - Resubmit with `OPD_CONTINUE_SKIP_PREP_AND_CURRENT_VAL=1` and
    `OPD_CONTINUE_ALLOW_EXISTING_SAVE=1`.
  - Auto-resume should preserve endpoints through `4096`; first replacement
    train starts endpoint `5120` from `iter_0000031` and rollout id `32`.

## OPD 25k Continuation Context-Overflow Submitter Correction

- First patch commit: `745f7a1` (`Clamp OPD rollout caps to context`), pushed
  to `origin/opd-reproduction`.
- Validation before the first replacement attempt:
  - `python3 -m py_compile slime/rollout/sglang_rollout.py`: passed.
  - Focused in-container clamp regression: prompt `1149`, requested cap
    `31744`, context limit `32766` produced effective cap `31617`; a short
    prompt kept `31744`.
  - `bash -n` on touched shell/sbatch scripts: passed.
  - `git diff --check`: passed.
  - `RUN_CONTAINER_CHECKS=1 bash examples/qwen3_8b_opd_tillicum/run_all_dry_check.sh`: passed.
- Archival/cancel actions:
  - Archived failed rollout diagnostics `rollout_32.pt` through
    `rollout_37.pt` with `.failed_162222` suffixes.
  - Archived failed sanity JSON diagnostics `samples_3072_3199.json` through
    `samples_3712_3839.json` with `.failed_162222` suffixes.
  - Canceled stale downstream jobs `162223` through `162267`.
- Canceled replacement attempt:
  - Submitted then immediately canceled jobs `163252` through `163302`.
  - Reason: the submitter required endpoint-specific full checkpoint dirs to
    recognize preserved endpoints. Because full checkpoint pruning kept only
    latest `iter_0000031`, the submitter would have rerun earlier endpoints
    `2048` and `3072` despite their completed val100 summaries.
- Submitter correction:
  - The submitter now reads the latest continuation checkpoint marker and
    treats endpoints at or below that sample count as preserved when their
    val100 summary exists.
  - With `latest_checkpointed_iteration.txt = 31`, the latest preserved
    endpoint is `4096`, so replacement submission should skip `2048`, `3072`,
    and `4096` and start new training at `5120`.
- Validation after submitter correction:
  - `bash -n examples/qwen3_8b_opd_tillicum/submit_opd_continue_1k_to_25k_val100_chain.sh`: passed.
  - `git diff --check`: passed.
  - `RUN_CONTAINER_CHECKS=1 bash examples/qwen3_8b_opd_tillicum/run_all_dry_check.sh`: passed.

## Submitted OPD 25k Continuation Context-Overflow Replacement

- Submitter correction commit: `471a8e2` (`Fix OPD continuation preserve detection`),
  pushed to `origin/opd-reproduction`.
- Replacement submit time/log: `2026-07-07 18:25 PDT`,
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/slurm_logs/submit_opd_continue_1k_to_25k_val100_20260707_182511.txt`.
- Submitter detected latest existing continuation checkpoint:
  `latest_checkpointed_iteration.txt = 31`, endpoint `4096`.
- Preserved endpoints in replacement submit:
  - `train_2048=preserved_existing_train_2048`,
    `val100_2048=preserved_existing_val100_2048`.
  - `train_3072=preserved_existing_train_3072`,
    `val100_3072=preserved_existing_val100_3072`.
  - `train_4096=preserved_existing_train_4096`,
    `val100_4096=preserved_existing_val100_4096`.
- Replacement job IDs:
  - `5120=(163315,163316)`, `6144=(163317,163318)`,
    `7168=(163319,163320)`, `8192=(163321,163322)`,
    `9216=(163323,163324)`, `10240=(163325,163326)`,
    `11264=(163327,163328)`, `12288=(163329,163330)`,
    `12544=(163331,163332)`, `13312=(163334,163335)`,
    `14336=(163336,163337)`, `15360=(163338,163339)`,
    `16384=(163340,163341)`, `17408=(163342,163343)`,
    `18432=(163344,163345)`, `19456=(163346,163347)`,
    `20480=(163348,163349)`, `21504=(163350,163351)`,
    `22528=(163352,163353)`, `23552=(163354,163355)`,
    `24576=(163356,163357)`, `24960=(163358,163359)`.
  - Midpoint full500 at `12,544`: `163333`.
  - Final report: `163360`, dependency `afterany:163359:163333`.
- Scheduler validation after submission:
  - `163315` started immediately on `g011` with no dependency.
  - `163316` waits on `afterok:163315`.
  - `163333` waits on `afterok:163332`.
  - `163360` waits on `afterany:163359:163333`.
  - Representative jobs have `MailUser=suryadv@cs.washington.edu` and
    `MailType=END,FAIL`.
  - Train/eval jobs request `gpu:h200:4`; final report requests `gpu:h200:1`.
  - No replacement job was `DependencyNeverSatisfied` at scheduler check.
- Runtime validation observed after `163315` started:
  - Log shows `OPD_ROLLOUT_MAX_CONTEXT_LEN=32766` as
    `OPD rollout max context len: 32766`.
  - Log shows endpoint-specific values for `opd_005120`:
    `OPD train rows requested: 5120`, `OPD effective samples: 5120`,
    final rollout id `39`.
  - Log shows the preserved latest checkpoint report for `iter=31`.
- Remaining runtime validation targets:
  - `163315` loads/resumes the full optimizer checkpoint from `iter_0000031`.
  - Rollout generation starts at rollout id `32`.
  - Long prompts log per-sample response-cap clamps instead of SGLang HTTP
    `400 Requested token count exceeds the model's maximum context length`.
  - First replacement checkpoint/HF snapshot writes endpoint `opd_005120`,
    rollout `iter_0000039`.

## Cleaned Chain ZeRO Smoke Timeout

- Failed job: `163645` (`slime-qwen3-sft-zero-smoke`), state `TIMEOUT`,
  elapsed `02:00:28`.
- Root cause: the smoke wrapper set `SFT_SIZE=4` and one rollout in the
  environment, but still passed the full cleaned SFT JSONL as `SFT_PARQUET`.
  Slime therefore trained on the real 25k cleaned SFT file instead of a tiny
  smoke file.
- Observed state:
  - Only ZeRO stage 1 ran; stages 2 and 3 never started.
  - Stage 1 was training and saving normally, reaching
    `latest_checkpointed_iteration.txt = 509`.
  - The accidental smoke output tree reached `8.0T`.
- Cleanup decision:
  - The old `outputs/sft_zero_smoke/stage_1_*` artifacts are not intended
    experiment checkpoints and are not fidelity-preserving progress.
  - They are safe to delete after this note because the actual cleaned SFT
    run will start fresh from base on the cleaned 25k SFT split.
- Cleanup performed:
  - Deleted only `outputs/sft_zero_smoke/stage_1_full_optim`,
    `stage_1_hf`, `stage_1_details`, and `stage_1_reports`.
  - Remaining `outputs/sft_zero_smoke` footprint after cleanup: `1.0K`.
- Patch:
  - The smoke job now creates
    `outputs/sft_zero_smoke/sft_zero_smoke_000004.jsonl` from the first
    four cleaned SFT rows and passes that file as `SFT_PARQUET`.
  - Smoke outputs now use fresh `tiny_stage_{1,2,3}_*` directories.
  - Each stage skips if `iter_0000000/.metadata` and HF
    `iter_0000000/config.json` already exist.
  - The cleaned-chain submitter supports `CLEANED_RESUME_AFTER_CONVERT=1`,
    reusing completed setup/data/convert artifacts from jobs `163642`,
    `163643`, and `163644`.

## Cleaned Chain ZeRO Smoke Replacement

- Patch commit before replacement submission: `ed35917`
  (`Fix cleaned ZeRO smoke resume`), pushed to `origin/opd-reproduction`.
- Static validation:
  - `bash -n` on edited smoke/submitter/SFT wrapper scripts: passed.
  - `git diff --check`: passed.
  - `RUN_CONTAINER_CHECKS=1 bash examples/qwen3_8b_opd_tillicum/run_all_dry_check.sh`: passed.
- Preserved completed artifacts:
  - vLLM setup job `163642`.
  - cleaned data job `163643`.
  - model conversion job `163644`.
  - cleaned data row counts: SFT `25,000`, OPD-1k `1,024`,
    OPD-next-4k `4,096`.
- Canceled stale downstream jobs: `163646` through `163653`.
- Replacement submit time/log: `2026-07-08 10:55 PDT`,
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/slurm_logs/submit_cleaned_sft_opd_vllm_20260708_105541.txt`.
- Replacement job IDs:
  - ZeRO smoke: `164126` (`gpu:h200:4`, `02:00:00`), no dependency.
  - Base vLLM eval: `164127` (`gpu:h200:4`, `04:00:00`),
    `afterok:164126`.
  - SFT 25k: `164128` (`gpu:h200:4`, `16:00:00`),
    `afterok:164127`.
  - SFT vLLM eval: `164129` (`gpu:h200:4`, `04:00:00`),
    `afterok:164128`.
  - OPD 1k: `164130` (`gpu:h200:4`, `08:00:00`),
    `afterok:164129`.
  - OPD 1k vLLM eval: `164131` (`gpu:h200:4`, `04:00:00`),
    `afterok:164130`.
  - OPD 5k: `164132` (`gpu:h200:4`, `24:00:00`),
    `afterok:164131`.
  - OPD 5k vLLM eval: `164133` (`gpu:h200:4`, `04:00:00`),
    `afterok:164132`.
  - Final report: `164134` (`gpu:h200:1`, `00:30:00`),
    `afterok:164133`.
- Scheduler validation:
  - `164126` started immediately on `g019`.
  - Downstream jobs use strict `afterok` dependencies and none were
    `DependencyNeverSatisfied` at scheduler check.
  - Representative jobs showed
    `MailUser=suryadv@cs.washington.edu MailType=END,FAIL`.
  - No replacement job requests more than 4 H200s.
- Runtime validation observed:
  - Smoke data file
    `outputs/sft_zero_smoke/sft_zero_smoke_000004.jsonl` has exactly
    `4` rows.
  - Live `164126` log shows SFT loading that smoke data file and writing to
    `outputs/sft_zero_smoke/tiny_stage_1_*`.
  - Live `164126` log shows rollout `0` only so far, not the old rollout
    `500+` behavior.
- Follow-up hardening:
  - The SFT wrapper now honors `SFT_SKIP_LOG_REDIRECT=1`, and the smoke job
    sets it for nested SFT calls so future smoke reruns keep the outer
    row-count/stage markers in the Slurm log instead of truncating them.

## Cleaned Chain Stable Smoke Resubmission

- Superseded replacement:
  - Smoke job `164126` completed tiny ZeRO stage 1 artifacts
    (`tiny_stage_1_full_optim/iter_0000000/.metadata` and
    `tiny_stage_1_hf/iter_0000000/config.json`) but exited `FAILED 2:0`.
  - Cause: the SFT wrapper was edited while `164126` was still executing it,
    which made bash read an inconsistent script tail and report
    `unexpected EOF while looking for matching '"'`.
  - Canceled stale downstream jobs `164127` through `164134`.
- Stable code commit: `7479671` (`Record cleaned smoke replacement`), pushed
  to `origin/opd-reproduction`.
- Stable replacement submit time/log: `2026-07-08 11:00 PDT`,
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/slurm_logs/submit_cleaned_sft_opd_vllm_20260708_110057.txt`.
- Stable replacement job IDs:
  - ZeRO smoke: `164148` (`gpu:h200:4`, `02:00:00`), no dependency.
  - Base vLLM eval: `164149` (`gpu:h200:4`, `04:00:00`),
    `afterok:164148`.
  - SFT 25k: `164150` (`gpu:h200:4`, `16:00:00`), `afterok:164149`.
  - SFT vLLM eval: `164151` (`gpu:h200:4`, `04:00:00`),
    `afterok:164150`.
  - OPD 1k: `164152` (`gpu:h200:4`, `08:00:00`), `afterok:164151`.
  - OPD 1k vLLM eval: `164153` (`gpu:h200:4`, `04:00:00`),
    `afterok:164152`.
  - OPD 5k: `164154` (`gpu:h200:4`, `24:00:00`), `afterok:164153`.
  - OPD 5k vLLM eval: `164155` (`gpu:h200:4`, `04:00:00`),
    `afterok:164154`.
  - Final report: `164156` (`gpu:h200:1`, `00:30:00`),
    `afterok:164155`.
- Runtime validation observed for stable smoke:
  - `164148` started immediately on `g011`.
  - Log shows `SFT ZeRO smoke data row count: 4`.
  - Log shows stage 1 skipped from the complete tiny artifact and stage 2
    started on the same 4-row smoke data file.
  - Downstream jobs use strict `afterok` dependencies and none were
    `DependencyNeverSatisfied` at scheduler check.
  - `164148` shows
    `MailUser=suryadv@cs.washington.edu MailType=END,FAIL`; no replacement
    job requests more than 4 H200s.

## Cleaned Chain ZeRO Stage-2 Smoke Fixes

- Stable smoke job `164148` failed during ZeRO stage 2 before training:
  - Stage 1 was skipped from complete tiny artifacts.
  - Stage 2 started on the 4-row smoke JSONL.
  - Megatron-FSDP aborted with
    `FSDP always requires CUDA_DEVICE_MAX_CONNECTIONS value large than one`.
- Patch commit: `28d384c` (`Fix SFT FSDP CUDA connections`), pushed to
  `origin/opd-reproduction`.
  - SFT now defaults `CUDA_DEVICE_MAX_CONNECTIONS=8` for
    `SFT_ZERO_STAGE=2` or `3`, while keeping `1` for non-FSDP stages.
  - Canceled stale downstream jobs `164149` through `164156`.
- Replacement after CUDA-connection fix:
  - Submit time/log: `2026-07-08 11:14 PDT`,
    `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/slurm_logs/submit_cleaned_sft_opd_vllm_20260708_111429.txt`.
  - Job IDs: smoke `164173`, base eval `164174`, SFT `164175`,
    SFT eval `164176`, OPD-1k `164177`, OPD-1k eval `164178`,
    OPD-5k `164179`, OPD-5k eval `164180`, report `164181`.
- Smoke job `164173` failed during ZeRO stage 2 before training:
  - Log showed `SFT CUDA_DEVICE_MAX_CONNECTIONS=8`.
  - Megatron-FSDP then aborted with
    `Megatron FSDP only supports fsdp_dtensor checkpoint format`.
- Patch commit: `f23ad89` (`Fix ZeRO stage 2 smoke checkpoint format`),
  pushed to `origin/opd-reproduction`.
  - Smoke stages 2 and 3 now use `SFT_CKPT_FORMAT=fsdp_dtensor`.
  - Stage 1 remains `torch_dist`.
  - Canceled stale downstream jobs `164174` through `164181`.
- Replacement after checkpoint-format fix:
  - Submit time/log: `2026-07-08 11:21 PDT`,
    `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/slurm_logs/submit_cleaned_sft_opd_vllm_20260708_112100.txt`.
  - Job IDs: smoke `164211`, base eval `164212`, SFT `164213`,
    SFT eval `164214`, OPD-1k `164215`, OPD-1k eval `164216`,
    OPD-5k `164217`, OPD-5k eval `164218`, report `164219`.
- Smoke job `164211` failed during ZeRO stage 2 before training:
  - Log showed `SFT CUDA_DEVICE_MAX_CONNECTIONS=8`.
  - Log showed `--ckpt-format fsdp_dtensor`.
  - Megatron-FSDP then aborted while constructing PyTorch `DeviceMesh`:
    `RuntimeError: ProcessGroup name not set`.
  - This is a Megatron-FSDP/PyTorch process-group compatibility issue: the
    process groups are valid, but this container's PyTorch `DeviceMesh`
    requires `group.group_name` and these Megatron-created groups do not have
    a name set.

## Cleaned Chain DeviceMesh Smoke Replacement

- Patch commit: `7cf5d9a` (`Patch Megatron FSDP DeviceMesh group names`),
  pushed to `origin/opd-reproduction`.
- Patch details:
  - Added a narrow Megatron compatibility shim in
    `slime/backends/megatron_utils/megatron_patch/device_mesh_process_group_name_patch.py`.
  - If `DeviceMesh.from_group` raises `ProcessGroup name not set`, the shim
    rebuilds the `DeviceMesh` with deterministic synthetic names and stores the
    actual process groups in DeviceMesh's private registry.
  - This does not change cleaned data, SFT/OPD hyperparameters, checkpoint
    retention, or the actual training objective.
- Validation before resubmission:
  - `python3 -m py_compile` on the new patch module and patch package:
    passed.
  - `bash -n` on smoke/SFT/submitter scripts: passed.
  - `git diff --check`: passed.
  - `RUN_CONTAINER_CHECKS=1 bash examples/qwen3_8b_opd_tillicum/run_all_dry_check.sh`:
    passed with the known harmless Apptainer fuse-overlay cleanup warning.
  - Direct container import of the new patch via `container_exec.sh`: passed.
- Canceled stale downstream jobs from `164211`:
  - `164212` through `164219` were no longer present in `squeue` at check
    time; `scancel 164212 ... 164219` returned cleanly.
- Replacement submit time/log: `2026-07-08 11:31 PDT`,
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/slurm_logs/submit_cleaned_sft_opd_vllm_20260708_113151.txt`.
- Replacement job IDs:
  - ZeRO smoke: `164234` (`gpu:h200:4`, `02:00:00`), no dependency.
  - Base vLLM eval: `164235` (`gpu:h200:4`, `04:00:00`),
    `afterok:164234`.
  - SFT 25k: `164236` (`gpu:h200:4`, `16:00:00`), `afterok:164235`.
  - SFT vLLM eval: `164237` (`gpu:h200:4`, `04:00:00`),
    `afterok:164236`.
  - OPD 1k: `164238` (`gpu:h200:4`, `08:00:00`), `afterok:164237`.
  - OPD 1k vLLM eval: `164239` (`gpu:h200:4`, `04:00:00`),
    `afterok:164238`.
  - OPD 5k: `164240` (`gpu:h200:4`, `24:00:00`), `afterok:164239`.
  - OPD 5k vLLM eval: `164241` (`gpu:h200:4`, `04:00:00`),
    `afterok:164240`.
  - Final report: `164242` (`gpu:h200:1`, `00:30:00`),
    `afterok:164241`.
- Scheduler validation:
  - `164234` started immediately on `g011`.
  - Downstream jobs are pending on strict `afterok` dependencies, not
    `DependencyNeverSatisfied`.
  - All jobs show `MailUser=suryadv@cs.washington.edu` and
    `MailType=END,FAIL`.
  - No job requests more than 4 H200s; final report requests 1 H200.
- Runtime validation targets:
  - Smoke log should skip complete stage 1, run stage 2 with
    `CUDA_DEVICE_MAX_CONNECTIONS=8` and `fsdp_dtensor`, and no longer fail on
    `ProcessGroup name not set`.
  - ZeRO stages 2 and 3 should each write `iter_0000000/.metadata` plus HF
    `iter_0000000/config.json`.
  - No smoke stage should train past rollout `0`.

## Cleaned Chain Pivot Away From ZeRO/FSDP

- Latest ZeRO/FSDP smoke attempt: `164383`.
  - Stage 1 was already complete and skipped from the tiny smoke artifact.
  - Stage 2 loaded base weights, completed rollout `0`, completed actor
    backward, and reached checkpoint save.
  - It then failed inside Megatron `fsdp_dtensor` checkpoint saving:
    `AttributeError: 'Tensor' object has no attribute 'to_local'` from
    `megatron/core/transformer/fsdp_dtensor_checkpoint.py` while splitting
    SwiGLU FC1 tensors.
- Decision:
  - Stop using the Megatron-FSDP/ZeRO-2/3 path for the cleaned experiment.
  - Keep prior logs, scripts, and compatibility patches as debugging history;
    do not delete smoke artifacts or failure logs.
  - Use Megatron's distributed optimizer over `DP=2` for SFT instead
    (`SFT_ZERO_STAGE=1` in the legacy wrapper variable, checkpoint format
    `torch_dist`).
  - The tiny smoke job now validates only this Megatron distributed-optimizer
    path; the old FSDP stage-2/3 smoke loop is disabled in-script with a note
    explaining the fsdp_dtensor save incompatibility.
- Static validation before resubmission:
  - `bash -n` on touched shell/sbatch scripts: passed.
  - `git diff --check`: passed.
  - `RUN_CONTAINER_CHECKS=1 bash examples/qwen3_8b_opd_tillicum/run_all_dry_check.sh`:
    passed with the known harmless Apptainer fuse-overlay cleanup warning.
- Patch commit: `3b42f27`
  (`Pivot cleaned SFT from ZeRO to Megatron DP optimizer`), pushed to
  `origin/opd-reproduction`.
- Replacement submit time/log: `2026-07-08 12:33 PDT`,
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/slurm_logs/submit_cleaned_sft_opd_vllm_20260708_123304.txt`.
- Replacement job IDs:
  - SFT Megatron-DP optimizer smoke: `164409` (`gpu:h200:4`, `02:00:00`),
    no dependency.
  - Base vLLM eval: `164410` (`gpu:h200:4`, `04:00:00`),
    `afterok:164409`.
  - SFT 25k: `164411` (`gpu:h200:4`, `16:00:00`), `afterok:164410`.
  - SFT vLLM eval: `164412` (`gpu:h200:4`, `04:00:00`),
    `afterok:164411`.
  - OPD 1k: `164413` (`gpu:h200:4`, `08:00:00`), `afterok:164412`.
  - OPD 1k vLLM eval: `164414` (`gpu:h200:4`, `04:00:00`),
    `afterok:164413`.
  - OPD 5k: `164415` (`gpu:h200:4`, `24:00:00`), `afterok:164414`.
  - OPD 5k vLLM eval: `164416` (`gpu:h200:4`, `04:00:00`),
    `afterok:164415`.
  - Final report: `164417` (`gpu:h200:1`, `00:30:00`),
    `afterok:164416`.
- Scheduler validation:
  - `164409` completed in `00:00:04` by reusing the existing complete tiny
    Megatron distributed-optimizer smoke artifact.
  - Downstream jobs are strict `afterok` dependencies and were not
    `DependencyNeverSatisfied` at check time.
  - All replacement jobs have `MailUser=suryadv@cs.washington.edu` and
    `MailType=END,FAIL`.
  - No replacement job requests more than 4 H200s; final report requests
    1 H200.

## Cleaned Chain vLLM Eval Wrapper Fix

- Base eval job `164410` failed after the Megatron-DP pivot smoke completed:
  - Log reached `VLLM_EVAL_STAGE stage=base`.
  - It then failed inside the inner container shell with
    `/usr/bin/bash: line 4: seq: command not found` and
    `VLLM_EVAL_PYTHON_CMD: unbound variable`.
- Patch commit: `e7ac9aa`
  (`Fix vLLM eval container env forwarding`), pushed to
  `origin/opd-reproduction`.
  - Added `VLLM_EVAL_PYTHON_CMD` to the container env allowlist and dry-check
    required forwarding list.
  - Replaced the inner `seq` loop with a pure Bash arithmetic `while` loop.
- Validation:
  - `bash -n` on touched eval/container/dry-check scripts: passed.
  - `git diff --check`: passed.
  - `RUN_CONTAINER_CHECKS=1 bash examples/qwen3_8b_opd_tillicum/run_all_dry_check.sh`:
    passed with the known harmless Apptainer fuse-overlay cleanup warning.
- Canceled stale downstream jobs from `164410`:
  - `164411` through `164417`.
- Replacement submit time/log: `2026-07-08 12:36 PDT`,
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/slurm_logs/submit_cleaned_sft_opd_vllm_20260708_123643.txt`.
- Replacement job IDs:
  - SFT Megatron-DP optimizer smoke: `164431` (`gpu:h200:4`, `02:00:00`),
    no dependency; completed in `00:00:03` by reusing the existing smoke
    artifact.
  - Base vLLM eval: `164432` (`gpu:h200:4`, `04:00:00`),
    `afterok:164431`; running on `g011` at scheduler check.
  - SFT 25k: `164433` (`gpu:h200:4`, `16:00:00`), `afterok:164432`.
  - SFT vLLM eval: `164434` (`gpu:h200:4`, `04:00:00`),
    `afterok:164433`.
  - OPD 1k: `164435` (`gpu:h200:4`, `08:00:00`), `afterok:164434`.
  - OPD 1k vLLM eval: `164436` (`gpu:h200:4`, `04:00:00`),
    `afterok:164435`.
  - OPD 5k: `164437` (`gpu:h200:4`, `24:00:00`), `afterok:164436`.
  - OPD 5k vLLM eval: `164438` (`gpu:h200:4`, `04:00:00`),
    `afterok:164437`.
  - Final report: `164439` (`gpu:h200:1`, `00:30:00`),
    `afterok:164438`.
- Runtime check:
  - `164432` log no longer shows the old `seq` or
    `VLLM_EVAL_PYTHON_CMD` failure and proceeds past stage setup into shard
    execution.

## Cleaned Chain vLLM Native Sampler Fallback

- Base eval job `164432` failed during vLLM V1 engine startup, before any eval
  samples were written.
  - vLLM loaded the base Qwen3 shards, then failed during scheduler/profile
    warmup in `vllm.v1.worker.gpu.sample.gumbel`.
  - Fatal root cause: `RuntimeError: Failed to find C compiler. Please specify
    via CC environment variable...`; the follow-on merge found `0/500`
    samples because all shards died before writing outputs.
- Patch commit: `6ac3f36`
  (`Patch vLLM native sampler eval fallback`), pushed to
  `origin/opd-reproduction`.
  - Added `VLLM_EVAL_FORCE_NATIVE_SAMPLER=1`, forwarded it into Apptainer, and
    logged it in the vLLM eval job.
  - Patched the eval helper to replace vLLM V1 sampler temperature/gumbel calls
    with native PyTorch argmax before `LLM(...)` construction.
  - This is scoped to greedy MATH-500 eval (`temperature=0`, `top_p=1`), where
    argmax is the intended decoding rule.
- Validation:
  - `python3 -m py_compile examples/qwen3_8b_opd_tillicum/eval_math500_vllm.py`:
    passed.
  - `bash -n` on touched shell/sbatch scripts: passed.
  - `git diff --check`: passed.
  - Targeted container regression returned native argmax tokens `[1, 2]`.
  - `RUN_CONTAINER_CHECKS=1 bash examples/qwen3_8b_opd_tillicum/run_all_dry_check.sh`:
    passed with the known harmless Apptainer fuse-overlay cleanup warning.
- Canceled stale downstream jobs from `164432`:
  - `164433` through `164439`.
- Replacement submit time/log: `2026-07-08 13:21 PDT`,
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/slurm_logs/submit_cleaned_sft_opd_vllm_20260708_132143.txt`.
- Replacement job IDs:
  - SFT Megatron-DP optimizer smoke: `164485` (`gpu:h200:4`, `02:00:00`),
    no dependency; completed by reusing the existing smoke artifact.
  - Base vLLM eval: `164486` (`gpu:h200:4`, `04:00:00`),
    after smoke; configuring/running at scheduler check.
  - SFT 25k: `164487` (`gpu:h200:4`, `16:00:00`), `afterok:164486`.
  - SFT vLLM eval: `164488` (`gpu:h200:4`, `04:00:00`),
    `afterok:164487`.
  - OPD 1k: `164489` (`gpu:h200:4`, `08:00:00`), `afterok:164488`.
  - OPD 1k vLLM eval: `164490` (`gpu:h200:4`, `04:00:00`),
    `afterok:164489`.
  - OPD 5k: `164491` (`gpu:h200:4`, `24:00:00`), `afterok:164490`.
  - OPD 5k vLLM eval: `164492` (`gpu:h200:4`, `04:00:00`),
    `afterok:164491`.
  - Final report: `164493` (`gpu:h200:1`, `00:30:00`),
    `afterok:164492`.
- Scheduler validation:
  - Downstream jobs are strict `afterok` dependencies and not
    `DependencyNeverSatisfied`.
  - All replacement jobs have `MailUser=suryadv@cs.washington.edu` and
    `MailType=END,FAIL`.
  - No replacement job requests more than 4 H200s; final report requests
    1 H200.
- Runtime validation target:
  - `164486` log should show `vLLM force native sampler: 1` and
    `patched vLLM V1 sampler to native greedy argmax path`, then write
    `math500_eval_cleaned_base_vllm/base/debug_eval_0.pt` and `summary.json`
    with 500 samples.

## Cleaned Chain vLLM Spawned-Worker and V1 Bookkeeping Fallbacks

- Follow-up failures after the first native sampler patch:
  - `164486` failed because the parent-process monkeypatch did not reach vLLM
    V1 spawned EngineCore workers; those workers still entered the original
    Triton gumbel path and failed without a C compiler.
  - Patch commit `5d37220` (`Patch vLLM sampler in spawned workers`) added a
    startup `sitecustomize.py` import hook controlled by
    `SLIME_VLLM_PATCH_SITE=1`, so spawned workers also patch the greedy sampler.
  - Replacement base eval `164523` then got past gumbel sampling but failed in
    vLLM V1 `get_num_sampled_and_rejected`, another small Triton warmup helper.
  - Patch commit `6a2e184`
    (`Patch vLLM sampler warmup counter fallback`) replaced that helper with a
    native torch implementation for the greedy eval path.
  - Replacement base eval `164563` then got past both sampler helpers but failed
    in vLLM V1 `buffer_utils._apply_write_kernel` during request-state staged
    writes, again because the runtime has no usable compiler.
  - A host-GCC bind was probed and rejected: the host compiler/binutils are not
    ABI-clean inside this Apptainer image without replacing libc paths, and
    replacing those paths breaks container Python preflight.
- Latest patch commit: `5ac1431`
  (`Patch vLLM V1 bookkeeping fallbacks`), pushed to
  `origin/opd-reproduction`.
  - Broadens the startup hook to patch vLLM V1 greedy-eval bookkeeping helpers:
    staged writes, fused staged writes, block-table gather/slot mapping, and
    input-batch preparation/update helpers.
  - Keeps eval semantics unchanged for this experiment: four 1-GPU vLLM
    workers, greedy decoding, dynamic per-prompt cap, and the same scorer.
- Validation before resubmission:
  - `python3 -m py_compile` on touched Python files: passed.
  - `git diff --check`: passed.
  - Targeted Apptainer/vLLM regression confirmed sampler, sampled/rejected
    counter, staged-write, block-table, and input-batch methods are patched.
  - `RUN_CONTAINER_CHECKS=1 bash examples/qwen3_8b_opd_tillicum/run_all_dry_check.sh`:
    passed with the known harmless Apptainer fuse-overlay cleanup warning.
- Canceled stale downstream jobs:
  - From failed `164523`: `164524` through `164530`.
  - From failed `164563`: `164564` through `164570`.
- Replacement submit time/log: `2026-07-08 14:30 PDT`,
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/slurm_logs/submit_cleaned_sft_opd_vllm_20260708_143058.txt`.
- Replacement job IDs:
  - SFT Megatron-DP optimizer smoke: `164634` (`gpu:h200:4`, `02:00:00`),
    no dependency; reuses the existing complete smoke artifact.
  - Base vLLM eval: `164635` (`gpu:h200:4`, `04:00:00`),
    `afterok:164634`.
  - SFT 25k: `164636` (`gpu:h200:4`, `16:00:00`), `afterok:164635`.
  - SFT vLLM eval: `164637` (`gpu:h200:4`, `04:00:00`),
    `afterok:164636`.
  - OPD 1k: `164638` (`gpu:h200:4`, `08:00:00`), `afterok:164637`.
  - OPD 1k vLLM eval: `164639` (`gpu:h200:4`, `04:00:00`),
    `afterok:164638`.
  - OPD 5k: `164640` (`gpu:h200:4`, `24:00:00`), `afterok:164639`.
  - OPD 5k vLLM eval: `164641` (`gpu:h200:4`, `04:00:00`),
    `afterok:164640`.
  - Final report: `164642` (`gpu:h200:1`, `00:30:00`), `afterok:164641`.
- Scheduler validation:
  - Downstream jobs are strict `afterok` dependencies and none is
    `DependencyNeverSatisfied` at submission check time.
  - Replacement jobs have `MailUser=suryadv@cs.washington.edu` and
    `MailType=END,FAIL`.
  - No job requests more than 4 H200s; final report requests 1 H200.
- Runtime validation target:
  - `164635` should show `SLIME_VLLM_PATCH_SITE=1` in spawned workers, avoid
    the gumbel/counter/staged-write `Failed to find C compiler` failures, and
    write `math500_eval_cleaned_base_vllm/base/debug_eval_0.pt` plus
    `summary.json` with 500 samples.

## Cleaned Chain vLLM Penalties Fallback Retry

- Base eval job `164635` got past the prior gumbel, sampled/rejected-counter,
  and staged-write compiler failures, then failed in
  `vllm.v1.worker.gpu.sample.penalties.bincount`, another Triton bookkeeping
  helper reached during sampler staged writes.
- Patch commit: `501d447` (`Patch vLLM penalties fallback`), pushed to
  `origin/opd-reproduction`.
  - Adds native torch fallbacks for vLLM V1 penalties `bincount` and
    `apply_penalties`.
  - This remains scoped to the greedy vLLM eval path; default/no-op penalties
    should not alter decoding semantics.
- Validation before resubmission:
  - `python3 -m py_compile slime/backends/vllm_utils/native_sampler.py`: passed.
  - `git diff --check`: passed.
  - Targeted Apptainer/vLLM regression confirmed `native_bincount` and
    `native_apply_penalties` are installed and produce expected counts.
  - `RUN_CONTAINER_CHECKS=1 bash examples/qwen3_8b_opd_tillicum/run_all_dry_check.sh`:
    passed with the known harmless Apptainer fuse-overlay cleanup warning.
- Canceled stale downstream jobs from failed `164635`:
  - `164636` through `164642`.
- Replacement submit time/log: `2026-07-08 14:46 PDT`,
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/slurm_logs/submit_cleaned_sft_opd_vllm_20260708_144649.txt`.
- Replacement job IDs:
  - SFT Megatron-DP optimizer smoke: `164706` (`gpu:h200:4`, `02:00:00`),
    no dependency; reuses the existing complete smoke artifact.
  - Base vLLM eval: `164707` (`gpu:h200:4`, `04:00:00`),
    `afterok:164706`.
  - SFT 25k: `164708` (`gpu:h200:4`, `16:00:00`), `afterok:164707`.
  - SFT vLLM eval: `164709` (`gpu:h200:4`, `04:00:00`),
    `afterok:164708`.
  - OPD 1k: `164710` (`gpu:h200:4`, `08:00:00`), `afterok:164709`.
  - OPD 1k vLLM eval: `164711` (`gpu:h200:4`, `04:00:00`),
    `afterok:164710`.
  - OPD 5k: `164712` (`gpu:h200:4`, `24:00:00`), `afterok:164711`.
  - OPD 5k vLLM eval: `164713` (`gpu:h200:4`, `04:00:00`),
    `afterok:164712`.
  - Final report: `164714` (`gpu:h200:1`, `00:30:00`), `afterok:164713`.
- Scheduler validation:
  - `164707` started on `g020`; downstream jobs remain strict `afterok`.
  - Replacement jobs have `MailUser=suryadv@cs.washington.edu` and
    `MailType=END,FAIL`.
  - No job requests more than 4 H200s; final report requests 1 H200.

## Cleaned Chain vLLM Structured-Output and Logit-Bias Fallback Retries

- Follow-up vLLM no-compiler failures:
  - Base eval job `164707` got past the sampler, bookkeeping, and penalties
    fallbacks, then failed in vLLM V1 structured-output grammar bitmask
    application. This is another Triton helper reached during V1 sampler
    warmup/request bookkeeping in the no-compiler Apptainer runtime.
  - Patch commit `f2140bb` (`Patch vLLM structured output fallback`), pushed
    to `origin/opd-reproduction`, replaces `StructuredOutputsWorker.apply_grammar_bitmask`
    with a no-op for this greedy MATH-500 eval path. We do not request guided
    decoding/structured outputs for these evals.
  - Replacement base eval job `164741` then got past structured-output handling
    and failed in `vllm.v1.worker.gpu.sample.logit_bias.apply_logit_bias`, where
    vLLM launched `_bias_kernel` and hit `Failed to find C compiler`.
- Latest patch commit: `0e80336` (`Patch vLLM logit bias fallback`), pushed to
  `origin/opd-reproduction`.
  - Adds a native torch fallback for vLLM V1 logit-bias handling.
  - The fallback preserves allowed-token masks, explicit additive logit biases,
    and min-token stop-token suppression; ordinary greedy requests still skip
    the helper when vLLM reports no active logit-bias controls.
  - Eval semantics remain unchanged for the cleaned experiment: 4 one-GPU vLLM
    workers, greedy decoding, dynamic per-prompt cap, and the same
    parse-fail-wrong MATH scorer.
- Validation before resubmission:
  - `python3 -m py_compile slime/backends/vllm_utils/native_sampler.py`: passed.
  - `bash -n` on touched shell/sbatch scripts: passed.
  - `git diff --check`: passed.
  - Targeted Apptainer/vLLM regression confirmed
    `logit_bias.apply_logit_bias` is patched to `native_apply_logit_bias` and
    applies mask/bias/stop-token behavior on a toy tensor.
  - `RUN_CONTAINER_CHECKS=1 bash examples/qwen3_8b_opd_tillicum/run_all_dry_check.sh`:
    passed with the known harmless Apptainer fuse-overlay cleanup warning.
- Canceled stale downstream jobs:
  - From failed `164707`: `164708` through `164714`.
  - From failed `164741`: `164742` through `164748`.
- Replacement submit time/log: `2026-07-08 15:18 PDT`,
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/slurm_logs/submit_cleaned_sft_opd_vllm_20260708_151853.txt`.
- Replacement job IDs:
  - SFT Megatron-DP optimizer smoke: `164775` (`gpu:h200:4`, `02:00:00`),
    no dependency; reuses the existing complete smoke/data/convert artifacts.
  - Base vLLM eval: `164776` (`gpu:h200:4`, `04:00:00`),
    `afterok:164775`.
  - SFT 25k: `164777` (`gpu:h200:4`, `16:00:00`), `afterok:164776`.
  - SFT vLLM eval: `164778` (`gpu:h200:4`, `04:00:00`),
    `afterok:164777`.
  - OPD 1k: `164779` (`gpu:h200:4`, `08:00:00`), `afterok:164778`.
  - OPD 1k vLLM eval: `164780` (`gpu:h200:4`, `04:00:00`),
    `afterok:164779`.
  - OPD 5k: `164781` (`gpu:h200:4`, `24:00:00`), `afterok:164780`.
  - OPD 5k vLLM eval: `164782` (`gpu:h200:4`, `04:00:00`),
    `afterok:164781`.
  - Final report: `164783` (`gpu:h200:1`, `00:30:00`), `afterok:164782`.
- Scheduler validation:
  - `164776` started on `g019`; downstream jobs remain strict `afterok`.
  - `164776` has `MailUser=suryadv@cs.washington.edu` and
    `MailType=END,FAIL`; all replacement scripts use the same mail settings.
  - No job requests more than 4 H200s; final report requests 1 H200.
- Runtime validation target:
  - `164776` should avoid the structured-output and logit-bias
    `Failed to find C compiler` failures and write
    `math500_eval_cleaned_base_vllm/base/debug_eval_0.pt` plus `summary.json`
    with 500 samples.

## Cleaned Chain vLLM Bad-Words Fallback Retry

- Base eval job `164776` got past the structured-output and logit-bias
  fallbacks, initialized the four vLLM engines, loaded Qwen3-8B-Base weights,
  and reached warmup. It then failed in
  `vllm.v1.worker.gpu.sample.bad_words.apply_bad_words`, where
  `_bad_words_kernel` tried to launch Triton and failed with
  `Failed to find C compiler`.
- No eval samples were merged from `164776`, so there is no eval artifact to
  preserve.
- Patch commit: `5726e31` (`Patch vLLM bad words fallback`), pushed to
  `origin/opd-reproduction`.
  - Adds a native torch fallback for vLLM V1 bad-words handling.
  - The fallback preserves the bad-word semantics by comparing generated
    suffixes against bad-word prefixes and setting the matching final-token logit
    to `-inf`.
  - These MATH-500 greedy evals do not intentionally pass bad-word lists, but
    vLLM warmup exercises this path; the patch keeps the no-compiler runtime
    moving without changing requested decoding settings.
- Validation before resubmission:
  - `python3 -m py_compile slime/backends/vllm_utils/native_sampler.py`: passed.
  - `git diff --check`: passed.
  - Targeted Apptainer/vLLM regression confirmed
    `bad_words.apply_bad_words` is patched to `native_apply_bad_words` and
    suppresses the expected logits on a toy tensor.
  - `RUN_CONTAINER_CHECKS=1 bash examples/qwen3_8b_opd_tillicum/run_all_dry_check.sh`:
    passed with the known harmless Apptainer fuse-overlay cleanup warning.
- Canceled stale downstream jobs from failed `164776`:
  - `164777` through `164783`.
- Replacement submit time/log: `2026-07-08 15:28 PDT`,
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/slurm_logs/submit_cleaned_sft_opd_vllm_20260708_152821.txt`.
- Replacement job IDs:
  - SFT Megatron-DP optimizer smoke: `164824` (`gpu:h200:4`, `02:00:00`),
    no dependency; reuses the existing complete smoke/data/convert artifacts.
  - Base vLLM eval: `164825` (`gpu:h200:4`, `04:00:00`),
    `afterok:164824`.
  - SFT 25k: `164826` (`gpu:h200:4`, `16:00:00`), `afterok:164825`.
  - SFT vLLM eval: `164827` (`gpu:h200:4`, `04:00:00`),
    `afterok:164826`.
  - OPD 1k: `164828` (`gpu:h200:4`, `08:00:00`), `afterok:164827`.
  - OPD 1k vLLM eval: `164829` (`gpu:h200:4`, `04:00:00`),
    `afterok:164828`.
  - OPD 5k: `164830` (`gpu:h200:4`, `24:00:00`), `afterok:164829`.
  - OPD 5k vLLM eval: `164831` (`gpu:h200:4`, `04:00:00`),
    `afterok:164830`.
  - Final report: `164832` (`gpu:h200:1`, `00:30:00`), `afterok:164831`.
- Scheduler validation:
  - `164825` has `MailUser=suryadv@cs.washington.edu` and
    `MailType=END,FAIL`.
  - `164825` requests 4 H200s and waits on `afterok:164824`; downstream jobs
    remain a strict `afterok` chain.
  - No job requests more than 4 H200s; final report requests 1 H200.
- Runtime validation target:
  - `164825` should avoid the bad-words `Failed to find C compiler` failure,
    enter real request generation, and write
    `math500_eval_cleaned_base_vllm/base/debug_eval_0.pt` plus `summary.json`
    with 500 samples.

## Cleaned Chain vLLM Min-P Fallback Retry

- Base eval job `164825` got past the bad-words fallback and then failed during
  vLLM V1 warmup in `vllm.v1.worker.gpu.sample.min_p.apply_min_p`, where
  `_min_p_kernel` tried to launch Triton and failed with
  `Failed to find C compiler`.
- No eval samples were merged from `164825`, so there is no eval artifact to
  preserve.
- Patch commit: `a0933ee` (`Patch vLLM min-p fallback`), pushed to
  `origin/opd-reproduction`.
  - Adds a native torch fallback for vLLM V1 min-p filtering.
  - Also patches the already-imported `sample.states.apply_min_p` reference,
    because `states.py` imports `apply_min_p` directly.
  - For the cleaned MATH-500 greedy eval path, requested min-p is the default
    zero value; the fallback still preserves min-p masking if vLLM warmup or a
    future request sets it nonzero.
- Validation before resubmission:
  - `python3 -m py_compile slime/backends/vllm_utils/native_sampler.py`: passed.
  - `git diff --check`: passed.
  - Targeted Apptainer/vLLM regression confirmed both
    `min_p.apply_min_p` and `states.apply_min_p` are patched to
    `native_apply_min_p` and apply the expected toy threshold.
  - `RUN_CONTAINER_CHECKS=1 bash examples/qwen3_8b_opd_tillicum/run_all_dry_check.sh`:
    passed with the known harmless Apptainer fuse-overlay cleanup warning.
- Canceled stale downstream jobs from failed `164825`:
  - `164826` through `164832`.
- Replacement submit time/log: `2026-07-08 15:39 PDT`,
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/slurm_logs/submit_cleaned_sft_opd_vllm_20260708_153934.txt`.
- Replacement job IDs:
  - SFT Megatron-DP optimizer smoke: `164870` (`gpu:h200:4`, `02:00:00`),
    no dependency; reuses the existing complete smoke/data/convert artifacts.
  - Base vLLM eval: `164871` (`gpu:h200:4`, `04:00:00`),
    `afterok:164870`.
  - SFT 25k: `164872` (`gpu:h200:4`, `16:00:00`), `afterok:164871`.
  - SFT vLLM eval: `164873` (`gpu:h200:4`, `04:00:00`),
    `afterok:164872`.
  - OPD 1k: `164874` (`gpu:h200:4`, `08:00:00`), `afterok:164873`.
  - OPD 1k vLLM eval: `164875` (`gpu:h200:4`, `04:00:00`),
    `afterok:164874`.
  - OPD 5k: `164876` (`gpu:h200:4`, `24:00:00`), `afterok:164875`.
  - OPD 5k vLLM eval: `164878` (`gpu:h200:4`, `04:00:00`),
    `afterok:164876`.
  - Final report: `164879` (`gpu:h200:1`, `00:30:00`), `afterok:164878`.
- Scheduler validation:
  - `164871` has `MailUser=suryadv@cs.washington.edu` and
    `MailType=END,FAIL`.
  - `164871` requests 4 H200s and waits on `afterok:164870`; downstream jobs
    remain a strict `afterok` chain.
  - No job requests more than 4 H200s; final report requests 1 H200.
- Runtime validation target:
  - `164871` should avoid the min-p `Failed to find C compiler` failure, enter
    real request generation, and write
    `math500_eval_cleaned_base_vllm/base/debug_eval_0.pt` plus `summary.json`
    with 500 samples.

## Cleaned Chain vLLM Logprob Fallback Retry

- Base eval job `164871` got past the min-p fallback and then failed during
  vLLM V1 warmup in `vllm.v1.worker.gpu.sample.logprob.compute_token_logprobs`,
  where `_topk_log_softmax_kernel` tried to launch Triton and failed with
  `Failed to find C compiler`.
- No eval samples were merged from `164871`, so there is no eval artifact to
  preserve.
- Patch commit: `eeea463` (`Patch vLLM logprob fallback`), pushed to
  `origin/opd-reproduction`.
  - Adds native torch fallbacks for vLLM V1 `compute_token_logprobs` and
    `compute_topk_logprobs`.
  - The fallback computes selected-token logprobs via `torch.log_softmax`,
    selected-token ranks via tensor comparison, and supports the custom
    per-request `logprob_token_ids` path used by warmup.
  - The cleaned MATH-500 eval path does not request returned logprobs, but
    vLLM warmup exercises this path.
- Validation before resubmission:
  - `python3 -m py_compile slime/backends/vllm_utils/native_sampler.py`: passed.
  - `git diff --check`: passed.
  - Targeted Apptainer/vLLM regression confirmed
    `logprob.compute_topk_logprobs` is patched to `native_compute_topk_logprobs`
    and returns expected token ids, ranks, and logprobs on a toy tensor.
  - `RUN_CONTAINER_CHECKS=1 bash examples/qwen3_8b_opd_tillicum/run_all_dry_check.sh`:
    passed with the known harmless Apptainer fuse-overlay cleanup warning.
- Canceled stale downstream jobs from failed `164871`:
  - `164872` through `164876`, plus `164878` and `164879`.
- Replacement submit time/log: `2026-07-08 15:56 PDT`,
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/slurm_logs/submit_cleaned_sft_opd_vllm_20260708_155608.txt`.
- Replacement job IDs:
  - SFT Megatron-DP optimizer smoke: `164974` (`gpu:h200:4`, `02:00:00`),
    no dependency; reuses the existing complete smoke/data/convert artifacts.
  - Base vLLM eval: `164975` (`gpu:h200:4`, `04:00:00`),
    `afterok:164974`.
  - SFT 25k: `164976` (`gpu:h200:4`, `16:00:00`), `afterok:164975`.
  - SFT vLLM eval: `164977` (`gpu:h200:4`, `04:00:00`),
    `afterok:164976`.
  - OPD 1k: `164978` (`gpu:h200:4`, `08:00:00`), `afterok:164977`.
  - OPD 1k vLLM eval: `164979` (`gpu:h200:4`, `04:00:00`),
    `afterok:164978`.
  - OPD 5k: `164980` (`gpu:h200:4`, `24:00:00`), `afterok:164979`.
  - OPD 5k vLLM eval: `164981` (`gpu:h200:4`, `04:00:00`),
    `afterok:164980`.
  - Final report: `164982` (`gpu:h200:1`, `00:30:00`), `afterok:164981`.
- Scheduler validation:
  - `164975` has `MailUser=suryadv@cs.washington.edu` and
    `MailType=END,FAIL`.
  - `164975` requests 4 H200s and waits on `afterok:164974`; downstream jobs
    remain a strict `afterok` chain.
  - No job requests more than 4 H200s; final report requests 1 H200.
- Runtime validation target:
  - `164975` should avoid the logprob `Failed to find C compiler` failure,
    enter real request generation, and write
    `math500_eval_cleaned_base_vllm/base/debug_eval_0.pt` plus `summary.json`
    with 500 samples.

## Cleaned Chain vLLM Prompt-Logprob Fallback Retry

- Base eval job `164975` got past the sampled-token logprob fallback and then
  failed during vLLM V1 warmup in
  `vllm.v1.worker.gpu.sample.prompt_logprob.get_prompt_logprobs_token_ids`,
  where `_prompt_logprobs_token_ids_kernel` tried to launch Triton and failed
  with `Failed to find C compiler`.
- No eval samples were merged from `164975`, so there is no eval artifact to
  preserve.
- Patch commit: `098353c` (`Patch vLLM prompt logprob fallback`), pushed to
  `origin/opd-reproduction`.
  - Adds a native torch fallback for prompt-logprob token-id gathering.
  - Also updates the prompt-logprob module's direct imported
    `compute_topk_logprobs` reference to the native logprob fallback.
  - The cleaned MATH-500 eval path does not request prompt logprobs, but vLLM
    warmup exercises this path.
- Validation before resubmission:
  - `python3 -m py_compile slime/backends/vllm_utils/native_sampler.py`: passed.
  - `git diff --check`: passed.
  - Targeted Apptainer/vLLM regression confirmed
    `prompt_logprob.get_prompt_logprobs_token_ids` is patched to
    `native_get_prompt_logprobs_token_ids` and that its
    `compute_topk_logprobs` reference points at the native fallback.
  - `RUN_CONTAINER_CHECKS=1 bash examples/qwen3_8b_opd_tillicum/run_all_dry_check.sh`:
    passed with the known harmless Apptainer fuse-overlay cleanup warning.
- Canceled stale downstream jobs from failed `164975`:
  - `164976` through `164982`.
- Replacement submit time/log: `2026-07-08 16:04 PDT`,
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/outputs/slurm_logs/submit_cleaned_sft_opd_vllm_20260708_160404.txt`.
- Replacement job IDs:
  - SFT Megatron-DP optimizer smoke: `165034` (`gpu:h200:4`, `02:00:00`),
    no dependency; reuses the existing complete smoke/data/convert artifacts.
  - Base vLLM eval: `165035` (`gpu:h200:4`, `04:00:00`),
    `afterok:165034`.
  - SFT 25k: `165036` (`gpu:h200:4`, `16:00:00`), `afterok:165035`.
  - SFT vLLM eval: `165037` (`gpu:h200:4`, `04:00:00`),
    `afterok:165036`.
  - OPD 1k: `165038` (`gpu:h200:4`, `08:00:00`), `afterok:165037`.
  - OPD 1k vLLM eval: `165039` (`gpu:h200:4`, `04:00:00`),
    `afterok:165038`.
  - OPD 5k: `165040` (`gpu:h200:4`, `24:00:00`), `afterok:165039`.
  - OPD 5k vLLM eval: `165041` (`gpu:h200:4`, `04:00:00`),
    `afterok:165040`.
  - Final report: `165042` (`gpu:h200:1`, `00:30:00`), `afterok:165041`.
- Scheduler validation:
  - `165035` has `MailUser=suryadv@cs.washington.edu` and
    `MailType=END,FAIL`.
  - `165035` requests 4 H200s and waits on `afterok:165034`; downstream jobs
    remain a strict `afterok` chain.
  - No job requests more than 4 H200s; final report requests 1 H200.
- Runtime validation target:
  - `165035` should avoid the prompt-logprob `Failed to find C compiler`
    failure, enter real request generation, and write
    `math500_eval_cleaned_base_vllm/base/debug_eval_0.pt` plus `summary.json`
    with 500 samples.
