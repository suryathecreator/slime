# Reproduction contract

The machine-readable contract is `config/experiment_contract.json`. Its SHA-256
prefix names the immutable scratch root. Jobs refuse nonempty output directories
unless explicitly resuming a compatible atomic checkpoint or per-problem eval.

The target is Qwen3-8B-Base -> 200K cleaned OpenThoughts3 rows for one SFT epoch
-> 1,472 held-out raw prompt rows with four on-policy samples each and Qwen3-32B
teacher log probabilities. Public targets are 76% after SFT and 94% after OPD on
MATH-500, with a preregistered band of ±3 points and at least +12 points from
SFT to OPD. These are comparison targets, never score-forcing targets.

MATH-500 is loaded read-only at its pinned revision in its original 500-row
order. It is never rewritten, normalized, filtered, subsetted, reordered, or
resampled. No MATH-500/OpenThoughts3 contamination check is performed. The
boxed-answer instruction is added only to the rendered user prompt.

All GPU Slurm jobs request exactly four H200s. OPD health metrics are preserved
as nonfatal diagnostics and do not abort the faithful run. Infrastructure
failures, nonfinite training, token-alignment errors, corrupt checkpoints, and
the pre-SFT data count gate remain fatal.
