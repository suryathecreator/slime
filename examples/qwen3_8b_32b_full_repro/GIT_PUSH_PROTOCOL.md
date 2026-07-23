# Git commit and push protocol

The implementation commit includes only paths allowed by
`commit_allowlist.json`. Before each push, record `git status --short`,
`git diff --stat`, and the staged name list; scan the staged patch for secrets,
tokens, private keys, unintended private paths, datasets, and weight files.
Run Python compilation, JSON validation, shell syntax checks, static contract
validation, and the pinned-container regression suite.

Push `qwen3-8b-32b-full-repro` normally and never force-push or rewrite the
small-scale history. Record the implementation commit in the contract, and
require the launch commit to be pushed before `submit_chain.sh` accepts it.

Raw OpenThoughts rows, MATH-500 rows, generated responses, model weights,
optimizer states, and rollout tensors remain in the contract-hashed scratch
root. V2 per-problem traces also remain in scratch. Only compact audits,
decision-change rows, hashes, manifests, configs, scripts, and summaries enter
normal Git. No LFS object is currently required. If a future explicitly
approved artifact uses Git LFS and its upload fails, do not push the main
experiment ref.

After completion, `sync_completed_artifacts.py` atomically stages the compact
cleanup/split/environment/submission/checkpoint/result/stopping records for a
second reviewed commit and normal push. It does not copy raw data or weights.
The V2 rescore uses the same two-push pattern: push the reviewed implementation
first, submit only from that clean remote-matched commit, then sync and push
the completed one-H200 submission manifest, scorer audit, and corrected result
report.
