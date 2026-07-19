# Final small-scale status before the 8B -> 32B pivot

Snapshot time: `2026-07-19T15:51:54-07:00`

This record closes the 1.7B -> 8B investigation without rewriting or deleting
its artifacts. It supplements `current_results_and_opd_audit_20260719.md` with
the completed 16K-cap continuation results and the live state of the final
480-generation AIME evaluation before it is canceled for the full reproduction.

## Completed results

| Experiment | Result | Cap hits | Mean generated tokens | Artifact |
|---|---:|---:|---:|---|
| Qwen3-1.7B base, AIME 2026, 16 samples/problem | 184/480 (38.33%) | 50/480 | 17,417.74 | immutable scorer-v2 rescore |
| Qwen3-8B teacher, AIME 2026, one sample/problem | 21/30 (70.00%) | 1/30 | 15,410.73 | immutable scorer-v2 rescore |
| Qwen3-1.7B OPD after 2,048 rows, original 31,744-cap run | 11/30 (36.67%) | 7/30 | 21,886.10 | `aime2026_qwen3_1.7b_opd_002048_single` |
| Qwen3-1.7B OPD after 2,048 rows, fresh 16K-cap run | 11/30 (36.67%) | 7/30 | 20,985.43 | `aime2026_qwen3_1.7b_opd_002048_r16k_single` |
| Qwen3-1.7B OPD after 5,120 rows, fresh 16K-cap run | 9/30 (30.00%) | 6/30 | 21,199.30 | `aime2026_qwen3_1.7b_opd_005120_r16k_single` |
| Qwen3-8B base, MATH-500 | 319/500 (63.8%) | 18/500 | 1,686.40 | preserved 8B result bundle |
| Qwen3-8B after corrected 24,832-row SFT, MATH-500 | 375/500 (75.0%) | 288/500 | 18,574.30 | preserved 8B result bundle |
| Qwen3-8B after corrected SFT + 1,024-row OPD, MATH-500 | 362/500 (72.4%) | 416/500 | 26,567.28 | preserved 8B result bundle |

The fresh 16K-cap 1.7B continuation therefore did not recover useful AIME
behavior: the one-sample result stayed at 11/30 after 2,048 rows and fell to
9/30 after 5,120 rows while mean evaluation length remained above 20K tokens.
This reinforces the prior finding that a rollout cutoff bounds compute but does
not teach stopping.

## Job chain

| Job | Stage | Final state at snapshot | Elapsed |
|---:|---|---|---:|
| 179845 | fresh 2,048-row 16K-cap OPD | COMPLETED | 01:38:33 |
| 179846 | 2,048-row single-sample AIME evaluation | COMPLETED | 01:43:30 |
| 179847 | continuation to 5,120 rows | COMPLETED | 04:30:51 |
| 179848 | 5,120-row single-sample AIME evaluation | COMPLETED | 01:33:13 |
| 179849 | 5,120-row, 480-generation AIME evaluation | RUNNING | 06:08:40 |

Job `179849` uses four H200s on `g004`. At the recorded snapshot, its four
shards had completed 51, 49, 49, and 52 of 120 prompts respectively: 201/480
generations had completed inside the workers. The evaluator writes final shard
JSONL files only when a worker finishes, so these in-memory partial generations
are not scoreable artifacts. The engine readiness files and worker logs are
preserved under:

`/gpfs/scrubbed/suryadv/slime-qwen3-1.7b-opd-aime2026/outputs/aime2026_qwen3_1.7b_opd_005120_r16k_mc16/shards`

The partial evaluation directory was 33 MiB at the snapshot. The evaluated HF
checkpoint occupied 12 GiB and its full OPD checkpoint occupied 23 GiB. No
artifact is deleted as part of the pivot.

## Immutable artifact hashes

- 2,048-row 16K metrics: `042c506377051b1fcaba3b3c06e918bb011c74fc408a6aef8e216ba2f10a163d`
- 2,048-row 16K predictions: `330002986d5714283413dc4f0b1d56e8825553a92ae970559b4647a353bd8f56`
- 5,120-row 16K metrics: `dffe35ed411c4e6463437e8553048aaeac5afd3f82b68c1198e2c68cc4a8d849`
- 5,120-row 16K predictions: `0aed41d4a480b51cb1adcdf58625cee52d601e7f5d350b5f6e251a5c3a44f342`

The live job-log hashes in the companion JSON are point-in-time hashes and are
expected to change until job `179849` exits. A post-cancel state and final log
hashes will be appended without rewriting the raw files.

## Conclusion

The smaller experiments were useful: they exposed loss-mask and checkpoint
loading bugs, validated terminal-token alignment, improved the scorer and
artifact pipeline, and showed that sampled reverse-KL can improve while useful
generation behavior and termination deteriorate. They are retained as
provenance. The project now pivots deliberately to the published Qwen3-8B-Base
-> OpenThoughts3 SFT -> Qwen3-32B OPD result.
