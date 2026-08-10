# MATH-500 runtime reference

Each 500-example pass was split into four contiguous 125-example shards. Each
shard requested one H200, so a fully concurrent pass used four H200s. The table
uses the longest cumulative shard runtime as the effective four-H200 pass time.
It includes retry/preemption compute but excludes time merely waiting in queue.

| Target | Greedy pass | Sampled pass, mean +/- SD | All three sampled passes, sequential equivalent |
|---|---:|---:|---:|
| Qwen2.5-3B base | 19 min | 16 +/- 2 min | 48 min |
| Qwen2.5-3B 8K | 63 min | 79 +/- 3 min | 3h 57m |
| Qwen2.5-3B 16K | 80 min | 76 +/- 11 min | 3h 48m |
| Qwen2.5-7B base | 10 min | 13 +/- 3 min | 38 min |
| Qwen2.5-7B 8K | 41 min | 30 +/- 3 min | 1h 31m |
| Qwen3-4B base | 20 min | 12 +/- 2 min | 37 min |
| Qwen3-4B 8K | 97 min | 68 +/- 8 min | 3h 23m |
| Qwen3-8B base | 19 min | 17 +/- 3 min | 52 min |
| Qwen3-8B 8K | 2h 02m | 1h 46m +/- 7 min | 5h 18m |

## Aggregate accounting

- Greedy: 26.83 H200-hours over nine passes.
- Sampling: 73.50 H200-hours over 27 passes.
- Mean effective pass time: 52 minutes greedy and 46 minutes sampled.
- Greedy wall-clock span: 10:39:01-16:07:24 PDT, or 5h 28m.
- Remaining sampled wave after greedy: 16:08:21-21:31:14 PDT, or 5h 23m.
- Entire evaluation: 10:39:01-21:31:14 PDT, or 10h 52m.

The wall-clock spans include queueing, the greedy-first barrier, holds, and
preemptions. H200-hours include every Slurm accounting record, including
preempted and requeued attempts (`sacct -X -D`), so they represent observed
resource consumption rather than idealized clean-run cost. Exact job IDs and
machine-readable accounting are in `RUNTIME.json`.
