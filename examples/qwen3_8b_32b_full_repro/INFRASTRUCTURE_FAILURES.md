# Infrastructure failures

Entries record job ID, stage, timestamp, exact error, preserved artifact paths,
recovery action, and regression test. Failed and capped rollouts are never
deleted.

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
container, records the detected cuDNN package without initializing the unused
backend, and cancels all accepted partial jobs if any future `sbatch` call
fails. The Qwen jobs retain the previously successful FlashAttention path.
Exact job states and artifact hashes are in `INFRASTRUCTURE_FAILURES.json`.

## Submission attempt 2: preflight passed, QoS hold superseded

Contract `0e42ac6dd4a5557a` at commit `adf8691` submitted the complete serial
dependency chain. Preflight job `180863` completed in 2:19 and verified all
pinned sources, environments, tokenizer IDs, contexts, and 44 tests. Cleanup
job `180864` was then held at `QOSMaxWallDurationPerJobLimit`: the normal QoS
has a one-day maximum, while cleanup requested 48 hours. Cleanup and the six
remaining dependents were canceled before any scientific work began.

Execution revision 3 limits every job to at most 24 hours. Cleanup and
evaluation already support atomic resume/requeue; SFT and OPD now also request
a ten-minute pre-timeout signal and self-requeue from their newest validated
full state. The 147 KiB attempt-2 root, including the successful preflight
manifests, remains unchanged. Exact states and hashes are in the JSON audit.

## Submission attempt 3: evaluation compiler failure superseded

Contract `4095e6ac256b0dbf` completed preflight and cleanup, but base-eval job
`180970` failed before producing a scored problem because Triton could not find
a C compiler in the evaluation container. Its five pending descendants were
canceled. Commit `992a01c` pins a Zig compiler and Python headers, preflights
them, and makes all four eval shards use that compiler. The failed log remains
preserved under the contract root.

## Submission attempt 4: evaluation GPU collision superseded

Replacement base-eval job `181398` got past compilation but all four shard
processes mapped to physical GPU 0. It OOMed in 2:23; jobs `181399`–`181403`
were canceled without running. Commit `9205c4a` forwards
`CUDA_VISIBLE_DEVICES` through the container boundary and asserts each shard's
physical GPU mapping before model load.

## Submission attempt 5: base and teacher complete; SFT OOM

Base job `181406` completed at 235/500 (47.0%) and teacher job `181407`
completed at 476/500 (95.2%). SFT job `181408` passed its eight-row Qwen3
loss-mask preflight, then OOMed at microbatch 0/37 before the first optimizer
update. Its 32,768-token TP2 vocabulary-logit conversion requested 9.27 GiB
with only 3.64 GiB free. This is a capacity failure, not allocator
fragmentation: just 67.61 MiB was reserved but unallocated.

No SFT checkpoint exists. The failed log and 18,335,753-byte rollout tensor are
retained and hashed in the JSON audit. Recovery reduces only the dynamic
packing target to the previously successful 16,384 tokens per GPU; native
32,768-token context, overlength-row handling, and the full 200,000-row
exposure remain unchanged. Completed base and teacher evaluations will not be
rerun.
