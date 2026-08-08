# Correct-only length-filtered OpenR1 full-SFT

This experiment trains only `correct_only`. There are no wrong-trace,
unmasked-mixed, random-mask, margin-mask, or probability-ratio variants. Every
run compares its pinned pretrained base checkpoint with one full-parameter SFT
checkpoint.

## Scientific contract

OpenR1-Math-220k is pinned at
`dc748648036c1ed619b020e056dc4b603eb39817`. Selection requires a trace to be
correct, source-marked reasoning-complete, structurally valid, and free of chat
control text. Seed 42 samples one trace per unique problem without truncation.

The 8K experiment uses one shared ordered set of exactly 2,000 trace IDs and
verbatim trace texts for Qwen2.5-3B, Qwen2.5-7B, Qwen3-4B-Base, and
Qwen3-8B-Base. A trace enters the sampling pool only when every model tokenizer
places its assistant response at or below 8,192 tokens and its complete rendered
training sequence at or below 10,240 tokens. This shared-eligibility intersection
is required so the same 2K examples are used across models, giving a proper
cross-model comparison. Each model then retokenizes those exact texts with its
own native chat template.

The independent Qwen2.5-3B 16K experiment allows assistant traces through
16,384 tokens and complete rendered sequences through 18,432 tokens.

The ordinary prompt is preserved verbatim:

```text
Please reason step by step, and put your final answer within \boxed{}.

Problem:
<problem>
```

All assistant tokens and exactly one `<|im_end|>` have loss weight 1. Prompt
tokens and the terminal template newline have weight 0. No synthetic
`<|endoftext|>` is appended. The model parameters are fully updated in BF16.

## Training

All five runs use global batch size 64, four epochs over 2,000 rows, and 125
optimizer updates. AdamW uses LR `5e-6`, betas `(0.9, 0.95)`, epsilon `1e-8`,
weight decay `1e-4`, gradient clip `1.0`, 3% linear warmup from zero, then cosine
decay to `1e-6`. The final zero-based HF snapshot is `iter_0000124`.

Before each full run, a one-update custom-pretokenized versus stock-message
canary dumps every realized token, shifted label, and loss weight. Boundary or
weight drift blocks that run. Deterministic before/after completions are retained
verbatim; prompt-copy detection is diagnostic and never fails the training chain.

## Submission and Hyak handoff

```bash
bash examples/qwen_correct_only_openr1_math220k_sft_2k/submit_training_only.sh --dry-run
bash examples/qwen_correct_only_openr1_math220k_sft_2k/submit_training_only.sh
```

The submitted chain is strictly serial with `afterok`, uses at most four H200s,
and contains no evaluation jobs.

The original data job `212331` completed selection and tokenization but failed
in its dynamic-schedule audit. Its narrow repair reuses those hash-validated
artifacts, replaces only the data-audit stage, and rewires the first blocked
canary while preserving jobs `212332` through `212342`:

```bash
bash examples/qwen_correct_only_openr1_math220k_sft_2k/resubmit_after_data_schedule_audit.sh --dry-run
bash examples/qwen_correct_only_openr1_math220k_sft_2k/resubmit_after_data_schedule_audit.sh
```

The first Qwen2.5-3B 8K canary then exhausted H200 memory on its first training
microbatch at the original 32,768-token dynamic-batch cap. The `memory_r1`
runtime profile lowers the 3B and 4B caps to 16,384 tokens without changing the
dataset, global batch, optimizer, loss, or model topology. Fresh schedule audits
are versioned under that profile, and the retry is preserved separately as
attempt `oom_r1`; the failed canary remains verbatim at its original path.

```bash
bash examples/qwen_correct_only_openr1_math220k_sft_2k/resubmit_after_canary_oom.sh --dry-run
bash examples/qwen_correct_only_openr1_math220k_sft_2k/resubmit_after_canary_oom.sh
```

After the training chain finalizes, checkpoint transfer is manual and
content-verified:

```bash
bash examples/qwen_correct_only_openr1_math220k_sft_2k/rsync_to_klone.sh --check
bash examples/qwen_correct_only_openr1_math220k_sft_2k/rsync_to_klone.sh --transfer
bash examples/qwen_correct_only_openr1_math220k_sft_2k/rsync_to_klone.sh --verify
```

Four unique bases and five trained checkpoints are transferred. The
Qwen2.5-3B base is shared by its 8K and 16K comparisons.

The rsync helper copies checkpoints and manifests, not repository code. Before
remote `--verify` or evaluation, fetch and switch to
`qwen-correct-only-openr1-sft-2k` in the Hyak `SLIME` checkout so the matching
manifest verifier and experiment contract are present.
