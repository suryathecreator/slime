# Infrastructure failures

No failures have yet occurred in this experiment. Append-only entries will
record job ID, stage, timestamp, exact error, preserved artifact paths, recovery
action, and regression test. Failed and capped rollouts are never deleted.

Historical infrastructure lessons are indexed in
`PREVIOUS_FIXES_CARRIED_FORWARD.md`; objective failures remain separate from
OOM, checkpoint, tokenizer, and server failures.

## Submission attempt 1: preserved and superseded

Contract `0984ccb69c314a9f` at commit `553a6a1` was submitted on 2026-07-19.
The site rejected cleanup and OPD because their 64-CPU requests exceeded the
eight-CPUs-per-GPU limit for four GPUs. A shell error-propagation defect then
submitted six jobs with an incomplete dependency graph. Jobs
`180853`–`180858` were canceled or had already failed; none remain queued.

No cleanup, training, checkpoint save, or scored MATH-500 problem completed.
The 186 KiB failed root is retained unchanged. Its logs additionally identified
a vLLM Unix-socket path overrun and inherited host cuDNN mismatch. Execution
revision 2 caps every job at 32 CPUs, uses a short job-local temporary path,
removes explicit host library inheritance before entering the training
container, records the bundled cuDNN package without initializing the unused
backend, and cancels all accepted partial jobs if any future `sbatch` call
fails. The Qwen jobs retain the previously successful FlashAttention path.
Exact job states and artifact hashes are in `INFRASTRUCTURE_FAILURES.json`.
