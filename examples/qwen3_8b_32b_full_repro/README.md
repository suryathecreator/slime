# Full Qwen3 8B -> 32B reproduction

This directory implements the pinned full reproduction. `config/` is the
machine-readable contract; `env.sh` is the executable projection of it.

The serial chain is:

```text
OpenThoughts3 raw rows
  -> cleanup audit and validation gate
  -> seed-1234 raw-row shuffle (200K SFT, next 100K OPD reserve)
  -> base and teacher MATH-500 evaluations
  -> 200K one-epoch SFT
  -> final SFT MATH-500 evaluation
  -> 1,472-prompt x four-sample colocated OPD
  -> final 100% OPD MATH-500 evaluation
  -> failed-closed 32K replay attempt on OPD cap hits (no score)
  -> results and stopping audit
  -> immutable V2 audit and whole-response strict-box V3 rescore
  -> corrected results and scorer audit
```

The training and inference stages request exactly four H200 GPUs. The
saved-generation rescore requests one H200 and performs no model inference.
MATH-500 is loaded read-only at the pinned revision in original order. It is
never used in cleanup or split selection, and there is no
OpenThoughts3/MATH-500 cross-contamination check.

## Key settings

- SFT: Qwen3-8B-Base, 200K rows, one epoch, BF16, native 32,768 context with
  overlength rejection and no truncation, batch 200, LR 1e-6 constant,
  AdamW 0.9/0.95, TP2/DP2.
- OPD: Qwen3-32B teacher, final SFT student, 1,472 raw prompt rows, four samples,
  temperature 0.8, top-p 1.0, top-k -1, response cap 16,384, CP-compatible
  16,380 context, batch 16/global 64, LR 1e-6, pure OPD objective.
- Evaluation: exact boxed-answer prompt, temperature 0.8, top-p 0.7, top-k -1,
  one sample, fixed 16,384 output-generation cap within the 32,768-token model
  context, EOS plus `<|im_end|>` stopping. The exact effective response budget
  is `min(16384, 32768 - rendered_prompt_tokens - 64)`.
- Automatic evaluation: base 8B, teacher 32B, final 200K SFT, final 100% OPD
  only. Intermediate weights remain available.

After the primary post-OPD evaluation, job `183039` attempted a diagnostic
32K-total-context replay of the 58 OPD rows that hit the primary cap. All four
shards completed generation, but seeded reruns diverged from their saved 16K
prefixes. The fail-closed gate therefore published zero rows and no 32K score.
Jobs `183040`, `183041`, and `183042` (SFT, teacher, and base proxies) were
cancelled without starting.

Exact sampled generation-to-generation equality is not guaranteed by a seed.
Dynamic batching, asynchronous scheduling, RNG consumption, kernel and
floating-point paths, engine/runtime details, and even the requested generation
budget can change the sampled tokens. The observed first failures occurred at
token offsets 95, 173, and 287; another seeded rerun stopped after 5,159 tokens
instead of reproducing its saved 16,384-token prefix.

Continuing from the saved 16K response would avoid replay drift, but it would
condition on an old trajectory while restarting sampler state. That is not the
same experiment as drawing a response from the original prompt with a 32K
budget, so it is not treated as a fidelity-preserving evaluation. A valid 32K
ablation would regenerate all 500 prompts from their original inputs under one
pinned protocol. Context length alone would still not guarantee perfect
reproduction of an external result unless the checkpoint, prompt template,
tokenizer, stop rules, sampling implementation, seed handling, scorer,
engine/runtime, batching, and hardware behavior were also aligned. Such a 32K
ablation, and other protocol ablations, can be run later if desired.

The completed primary rule is a fixed 16K output cap, subject to the 32K model
context and 64-token safety margin. The abandoned attempt used a dynamic 32K
total-context budget. No incomplete 32K attempt is included in the reported
reproduction metrics.

## Execution

Validate without launching work:

```bash
python3 examples/qwen3_8b_32b_full_repro/validate_config.py --require-pinned-commit
bash -n examples/qwen3_8b_32b_full_repro/*.sh
```

After the reviewed branch is pushed, submit the chain once:

```bash
bash examples/qwen3_8b_32b_full_repro/submit_chain.sh
```

The submitter refuses a dirty/unpushed branch, a prior submission manifest, a
nonempty output root, or any Slurm script that does not request four H200s.
Runtime data, raw generations, checkpoints, and models stay under the
contract-hashed scratch root and are never added to normal Git.

Expected wall times are roughly 6–24 hours cleanup, 4–12 hours per full
MATH-500 evaluation, 20–30 hours SFT, and 8–12 hours OPD. Expected peak durable
storage is about 250 GB for SFT weights plus latest full state, 180 GB for OPD
weights plus latest full state, and additional rollout/evaluation artifacts.
Every request stays within the normal QoS 24-hour limit; long cleanup, SFT, OPD,
and evaluation stages requeue and resume from atomic artifacts when needed.

After the chain completes, run `sync_completed_artifacts.py`, review the compact
diff and allowlist, run tests again, then commit and push the final audits and
results without force-pushing.

The original evaluation artifacts remain immutable under
`math500_event_scorer_v1`. To rescore all four saved 500-row stages with the
audited V2 policy, push a clean launch commit and run:

```bash
bash examples/qwen3_8b_32b_full_repro/submit_rescore.sh
```

The rescore submitter refuses a dirty or unpushed checkout. Publication is
fail-closed unless all 2,000 source hashes validate, every row is decided,
review count is zero, and all audited aggregate transition counts match.
