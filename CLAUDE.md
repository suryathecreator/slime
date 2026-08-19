# CLAUDE.md

## What this repo is

A research fork of THUDM/slime (`origin` = `git@github.com:suryathecreator/slime.git`).
Upstream framework code under `slime/`, `slime_plugins/`, `scripts/`, `docs/` is treated as
vendored: change it only when a run is actually blocked, and keep the change minimal.
All research lives in `examples/<experiment_name>/`, with matching unit tests in
`tests/test_<experiment_name>.py`.

The research line is SFT on reasoning traces — correct vs. incorrect supervision, token
masking schemes, and how each transfers off the trained problem distribution. Read the
experiment's own `README.md` first; each one states its scientific contract in prose.

## Two clusters

| | Tillicum | Hyak / Klone |
|---|---|---|
| Role | training (4x H200, Apptainer sandbox) | evaluation (vLLM) |
| Job IDs | ~2xx,xxx | ~38,6xx,xxx |
| Entry point | `submit_training_only.sh` | `eval/submit_hyak.sh` |
| Repo path | `/gpfs/scrubbed/suryadv/repos/slime` | `/mmfs1/gscratch/scrubbed/suryadv/repos/SLIME` |

Checkpoints cross the gap via `rsync_to_klone.sh`, never by re-training. Training packages
deliberately leave sampling/decoding parameters unpinned for the Hyak evaluator to define.

## Anatomy of an experiment package

New experiments are cloned from the most recent similar one, not written from scratch.
The current reference is `examples/qwen2_5_7b_aime_2009_2024_split_sft_6k/`.

- `config/experiment_contract.json` — machine-readable scientific contract. Its SHA-256
  prefix (16 hex chars) is `CONTRACT_HASH`, which *derives* the scratch root. Editing the
  contract moves the experiment to a new root; that is the intended way to version a redesign.
- `env.sh` — the executable projection of the contract. Sourced, never executed. All paths
  are derived from `CONTRACT_HASH`; never hardcode a scratch path.
- `NN_*.sbatch` — numbered stages wired into one strict serial `afterok` DAG:
  prepare -> canary(s) -> train -> finalize.
- `prepare_data.py`, `validate_data.py`, `validate_config.py`, `finalize.py`, `data_utils.py`.
- `submit_training_only.sh`, `write_submission_manifest.py`, `publish_provenance.py`,
  `rsync_to_klone.sh`.
- `SUBMISSION.json` — the committed record of a submitted job chain.
- `results/` — published immutable bundles.

Scratch layout: `/gpfs/scrubbed/suryadv/slime-<experiment>/v1/<CONTRACT_HASH>/` containing
`data/ outputs/ handoff/ manifests/ slurm_logs/ execution_repo/ TRAINING_STATUS.json`.

## Hard rules

- **Dry-run before submit.** `submit_training_only.sh --dry-run`, then `--submit`.
- **Clean worktree before submit.** Submitters export the exact 40-char git SHA into every
  delayed job, and each job re-verifies `HEAD` equality plus a clean worktree before writing
  any artifact. A later pull or edit must stop the chain, not silently mix revisions.
  Consequence: push the implementation first, submit, *then* commit `SUBMISSION.json`.
  Where a runtime gate allows post-submission metadata, it permits only a clean descendant
  commit touching that one metadata path.
- **Never mutate a published `results/` bundle or a superseded manifest.** A failed or
  replaced attempt is recorded (`SUBMISSION_ATTEMPT_NN.json` with `status: superseded`,
  `superseded_reason`, and the failure record) and archived — not deleted, not overwritten.
  `publish_provenance.py` refuses to overwrite an existing output root by design.
- **Fail closed.** Validation gates raise on drift rather than repairing or truncating.
  Preserve that; do not soften a gate to make a run pass.
- **Pin everything.** Model/dataset/tokenizer HF revisions, artifact SHA-256s, and the CUDA
  toolchain (`/sw/cuda/12.8.1`, executable `nvcc` first on `PATH` — a non-executable `nvcc`
  has broken TorchInductor warmup more than once).
- Runs are offline: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`. Weights are pre-staged.
- Git excludes weights, optimizer/full state, tokenized training data, raw source
  generations, and large canary payloads. Hashes of those go in Git; the bytes do not.

## Scoring

AIME work uses `aime_last_boxed_integer_scorer_v1`: last complete, nonempty, brace-balanced
`\boxed{...}`, exactly one integer 0–999, no unboxed or heuristic fallback. **Score cap hits
too** — do not silently drop length-capped completions. Report accuracy alongside valid-box
rate, cap-hit rate, and length diagnostics; cap-hit rates have been high enough to dominate
interpretation. MATH-500 scorers are versioned (V2/V3/V5/V6) and rescores are published as
new bundles beside the old ones.

## Style

Python 3.10 target, `black`/`isort` at line length 119, `ruff`. Run `pre-commit run --files ...`
on touched files. Experiment code is `from __future__ import annotations`, typed, and writes
JSON artifacts atomically (temp file + `fsync` + `replace`) with `sort_keys=True`.
