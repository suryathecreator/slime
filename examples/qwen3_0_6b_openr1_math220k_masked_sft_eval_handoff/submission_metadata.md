# Submission metadata

## Compile-cache-isolated replacement — 2026-08-07

- Evaluator patch: `aa6461bf2b3bea9cc2c15891fb2c904dd11cf533`
  (`Isolate MATH-500 worker compile caches`), pushed before submission to
  `origin/qwen3-0.6b-openr1-masked-sft-2k`.
- Replacement submitted at `2026-08-07T07:55:34Z` on `ckpt-all` with account
  `raivn-ckpt`, QOS `ckpt`, one H200, 8 CPUs, 96 GB, and two hours per shard.
- Superseded attempt: preflight `38268373`, canary `38268374`, arrays
  `38268375`, `38268377`, `38268379`, `38268381`, `38268383`, `38268385`,
  `38268387`, `38268390`, `38268392`, `38268394`, `38268396`, and `38268398`;
  finalizers `38268376`, `38268378`, `38268380`, `38268382`, `38268384`,
  `38268386`, `38268389`, `38268391`, `38268393`, `38268395`, `38268397`, and
  `38268399`; and audit `38268400`.
- Failure cause: concurrent workers wrote the same vLLM/TorchInductor graph
  cache and failed during atomic rename with `BackendCompilerFailed` and
  `FileNotFoundError`.  Observed failed tasks were `38268381_3`
  (`random_mask_25`), `38268394_1` (`inverse_tau_0p05`), `38268396_2`
  (`margin_mask`), and `38268398_2` (`prob_ratio_mask`).
- The replacement assigns each worker a persistent cache under
  `tasks/array_<array-job>/<variant>/shard_<index>`; canary uses its own job
  namespace.  vLLM, TorchInductor, Triton, XDG, and temporary caches are all
  isolated while the offline Hugging Face cache remains shared.
- Durable shard output was preserved.  Complete generation chunks, including
  completed base shard 2, are reused verbatim; `repair-shard` removes only an
  incomplete trailing write before generation resumes.  Prompts, decoding,
  checkpoints, and last-box scoring are unchanged.

| Checkpoint | H200 array | CPU finalizer |
| --- | ---: | ---: |
| `base_0p6b` | `38278879` | `38278880` |
| `correct_only` | `38278881` | `38278882` |
| `unmasked` | `38278883` | `38278884` |
| `random_mask_25` | `38278885` | `38278886` |
| `random_mask_50` | `38278887` | `38278888` |
| `random_mask_70` | `38278889` | `38278890` |
| `random_mask_80` | `38278891` | `38278892` |
| `random_mask_90` | `38278893` | `38278894` |
| `inverse_tau_0p20` | `38278895` | `38278896` |
| `inverse_tau_0p05` | `38278897` | `38278898` |
| `margin_mask` | `38278899` | `38278900` |
| `prob_ratio_mask` | `38278901` | `38278902` |

- Replacement preflight: `38278877`.
- Replacement generation/scoring canary: `38278878`, dependency
  `afterok:38278877`.
- Every H200 array is `0-3`, unthrottled, and depends on
  `afterok:38278878`; each finalizer depends on its complete array.
- Final audit: `38278903`, dependent on all twelve finalizers.
- Initial scheduler validation found the expected resources and dependency
  graph, with no `DependencyNeverSatisfied` jobs.  Preflight was pending for
  priority and all downstream jobs were pending on valid dependencies.

## Short-IPC-path replacement — 2026-08-07

- Follow-up patch: `50dcc5106ae92c9c7a463d8a909e9e5b6e46328e`
  (`Shorten MATH-500 vLLM IPC paths`), pushed before submission.
- Reason for replacement: canary `38278878` failed while binding a ZeroMQ IPC
  socket.  Its persistent cache-based temporary path was 114 characters, over
  `zmq.IPC_PATH_MAX_LEN=107`; all twelve dependent arrays consequently became
  `DependencyNeverSatisfied` without running.
- Persistent vLLM, TorchInductor, Triton, and XDG caches remain isolated by
  submission, variant, and shard.  Only `TMPDIR` moved to a secure short
  node-local `mktemp` path under `/tmp/q06m5-...`.
- The guarded launcher classified the failure as `zmq_ipc_path_too_long`,
  canceled jobs `38278877`–`38278903` from that submission journal, archived
  the attempt, and preserved all durable generation chunks.
- Corrected replacement submitted at `2026-08-07T08:11:48Z` with the unchanged
  48-task H200 shape and scoring/generation contract.

| Checkpoint | H200 array | CPU finalizer |
| --- | ---: | ---: |
| `base_0p6b` | `38279369` | `38279370` |
| `correct_only` | `38279371` | `38279372` |
| `unmasked` | `38279373` | `38279374` |
| `random_mask_25` | `38279375` | `38279376` |
| `random_mask_50` | `38279377` | `38279378` |
| `random_mask_70` | `38279379` | `38279380` |
| `random_mask_80` | `38279381` | `38279382` |
| `random_mask_90` | `38279383` | `38279384` |
| `inverse_tau_0p20` | `38279385` | `38279386` |
| `inverse_tau_0p05` | `38279387` | `38279388` |
| `margin_mask` | `38279389` | `38279390` |
| `prob_ratio_mask` | `38279391` | `38279392` |

- Preflight `38279367` completed successfully in 1m12s.
- Canary `38279368` used persistent cache
  `tasks/job_38279368/canary` with temporary directory
  `/tmp/q06m5-38279368.K8siKr`; it compiled, generated, scored, wrote
  `CANARY_READY.json`, and completed successfully in 2m57s.
- The successful canary released every array to normal scheduler priority.
  Base shard `38279369_0` then used
  `/tmp/q06m5-38279369-0.XR3mvc` and wrote its first durable progress record,
  `completed=32 total=125`.
- Final audit: `38279393`, dependent on all twelve corrected finalizers.
