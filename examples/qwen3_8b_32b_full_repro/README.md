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
  -> serial exact-prefix 32K cap-hit proxies (OPD, SFT, teacher, base)
  -> results and stopping audit
```

Every Slurm stage requests exactly four H200 GPUs. MATH-500 is loaded read-only
at the pinned revision in original order. It is never used in cleanup or split
selection, and there is no OpenThoughts3/MATH-500 cross-contamination check.

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

After the primary post-OPD evaluation, the recovery chain runs a diagnostic
32K-total-context proxy in serial order: OPD, SFT, teacher, then base. It
regenerates only source rows that hit the primary cap, from the original saved
input token IDs and per-problem seed. Natural stops are reused. The dynamic
response budget is `32768 - rendered_prompt_tokens - 64`, where the rendered
prompt includes chat-template special tokens, the problem, instruction, and
assistant prefix; the generated response includes reasoning, think tags, answer
text, and any emitted stop token. There is no hidden reasoning stream outside
that accounting. Every rerun must reproduce the saved source output token IDs
as an exact prefix or the diagnostic refuses to publish.

The primary rule is a 16K output cap, while the proxy rule is a dynamic 32K
total-context budget. Prompt lengths are small enough that their effective
generation lengths remain near 16K and 32K respectively. Each rule is applied
consistently to every checkpoint, and proxy results never replace the primary
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
