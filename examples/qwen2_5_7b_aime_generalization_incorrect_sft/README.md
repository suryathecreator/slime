# Qwen2.5-7B AIME distribution generalization vs. incorrect traces

This experiment separates fitting held-in AIME problem IDs from generalizing to
unseen, disjoint AIME problem IDs. Contests may occur in both problem sets, so
this is a problem-disjoint rather than contest-disjoint comparison. It first full-finetunes
`Qwen/Qwen2.5-7B` on 3,000 repeated correct trajectories from 400 held-in AIME
problems. Eleven independent continuations then start from that exact HF
checkpoint: one uses 1,500 unmasked wrong OpenR1 traces, nine use the same wrong
traces with nested random masks from 10% through 90%, and one uses 1,500 correct
OpenR1 traces. Wrong and correct OpenR1 problem IDs are unique and disjoint.

The AIME source `prompt` field is preserved verbatim as the Qwen user content.
Its assistant content is rendered as
`<think>\n{reasoning_content}\n</think>\n\n{content}`. The tokenizer's Qwen2.5
chat template inserts the default `You are a helpful assistant.` system turn,
the user/assistant role boundaries, and the assistant generation marker. For
OpenR1, the existing standard user instruction and already structured
`<think>...</think>` generation are preserved. Every record ends in a supervised
`<|im_end|>` followed by an unsupervised template newline; prompts never receive
loss and no synthetic `<|endoftext|>` is added.

For each held-in problem, every eligible AIME trajectory receives the
SHA-256 pseudorandom key
`SHA256("aime-trajectory-selection-v1\\0seed=42\\0" + trace_id)`. The minimum
key wins, with `trace_id` as the exact collision tie-break. Under the seeded
random-key model, each of a problem's `n` eligible trajectories therefore has
probability `1/n` of selection. This AIME path deliberately differs from
OpenR1: AIME inserts the dataset `prompt` verbatim and constructs the assistant
think/final text from `reasoning_content` and `content`; OpenR1 uses our
standard instruction built from its `problem` field and preserves its already
structured generation verbatim. OpenR1 UUIDs are retained only as provenance:
because the pinned source has null-like and colliding UUIDs, selection and
correct/wrong disjointness use `SHA256(exact UTF-8 problem text)` as the
semantic problem identity. The preparation gate proves all 93,733 exact problem
texts are unique. It also excludes every OpenR1 problem whose whitespace-
normalized question matches any of the 861 eligible AIME questions. The pinned
source contains 94 such overlaps (41 held-in, 45 held-out, and 8 unused), so
this gate prevents direct AIME leakage into either continuation pool.

`--seq-length 32768` is a total rendered-sequence ceiling, not an assistant-only
allowance. It includes system and user text, chat boundaries, reasoning, final
answer, and terminal markup. Overlong examples are rejected rather than
truncated. Dynamic batching packs complete records by their realized lengths at
`--max-tokens-per-gpu 16384`; it does not change their contents.

The standard recipe is full BF16 SFT on four H200s (TP4/DP1, no optimizer
sharding), global batch 64,
AdamW at `5e-6`, 3% warmup from zero, cosine decay to `1e-6`, betas
`(0.9, 0.95)`, epsilon `1e-8`, weight decay `1e-4`, and gradient norm `1.0`.
Tokenwise log-probability loss is evaluated in 2,048-token chunks. This keeps
the FP32 vocabulary-loss temporary bounded for complete sequences near the 32K
ceiling without changing the examples, masks, token weights, or cross-entropy.
This topology is the fail-closed repair for two preserved stage-1 canary
failures: job 223263 exhausted memory in an 8.56-GiB unchunked vocabulary-loss
temporary for a 29,949-token sample; job 223574 reduced that immediate
allocation to 594 MiB with chunking but still exhausted aggregate TP2 backward
memory on a 30,444-token sample. TP4 halves each GPU's local vocabulary and
tensor-parallel model/activation share while retaining the same four H200s.
Job 223913 then passed all five TP4 updates and both checkpoint exports with at
least 62.96 GiB free, but its post-training audit failed because the audit still
grouped actor dumps as TP2/DP2. The topology-aware audit now derives TP and DP
groups from the runtime contract and compares every tensor replica before
counting one representative payload per data-parallel rank.
Stage 1 runs 188 updates and finishes at iteration 187. Every continuation runs
the same 94-update budget and finishes at iteration 93 with a fresh optimizer
and LR schedule. A continuation-local checkpoint still resumes its own optimizer
after preemption. Full optimizer plus rollout-dataset state is saved every 24
updates and at the forced final update; the latest complete state is retained
for requeue/resume. Periodic HF bridge snapshots are pruned in the background
and only `iter_0000187` for stage 1 or `iter_0000093` for a child enters the
handoff. This package selects synchronous `train.py`: unlike the asynchronous
entrypoint, it does not prefetch the next 64-row batch before saving the current
rollout-dataset state, so a resumed update consumes the exact next batch. A
signal before the first durable save moves the partial attempt to a preserved
quarantine and restarts from the beginning; it never pretends partial state is
resumable. If the final full state exists but its HF export is incomplete, the
HF snapshot is regenerated from that full state and passes the shard-index gate
before manifesting.

The chain is fail-closed and strictly serial. Before submission it proves that
job `222947`, the finalizer of the preceding random-mask-replicate chain,
completed with exit code `0:0`. If Slurm still knows that job, preparation is
also submitted with `afterok:222947`; if the controller has aged it out, the
verified `sacct` gate is recorded and no invalid dependency is attached.
Data preparation, two five-update canaries, stage 1, all eleven continuations,
and finalization are submitted; evaluation and transfer are not. Hyak later
evaluates base, stage 1, and all continuations with three samples on each of the
same 400 held-in and 400 held-out problem IDs. Each pass uses temperature 0.7,
top-p 0.8, top-k -1, one completion, and seed
`1234 + 400 * repeat + eval_index`. The transferred dataset `prompt` is again
the verbatim Qwen user content. Its response budget is
`32768 - rendered_prompt_tokens`, so system text and every Qwen boundary count
against the same total context limit.

The AIME scorer examines the whole saved completion, including cap hits. It
selects the last complete, nonempty, brace-balanced `\boxed{...}` group,
handles nested and escaped braces, removes only fixed presentation wrappers,
and accepts exactly one integer from 0 through 999 (leading zeros allowed).
There is no unboxed, prose, algebraic, or multipart fallback. The final report
compares all 13 targets on held-in and held-out accuracy, their generalization
gap, deltas from base and stage 1, and cap/parse/length diagnostics.

The submitter exports the exact 40-character Git SHA into every delayed job.
Before creating any experiment artifact, each job proves both `HEAD` equality
and a clean tracked/untracked worktree; later pulls or edits therefore stop the
chain instead of silently mixing code revisions.

```bash
bash examples/qwen2_5_7b_aime_generalization_incorrect_sft/submit_training_only.sh --dry-run
bash examples/qwen2_5_7b_aime_generalization_incorrect_sft/submit_training_only.sh --submit
```

The preparation report records per-split and per-training-set prompt, assistant,
response, and total-token mean/median/p90/p95/p99/max values; 32K rejection
counts; duplication exposures; actual mask rates; source, selection, and output
hashes; and every protected-token invariant.

After training, `rsync_to_klone.sh` transfers only the 12 final HF checkpoints,
the two 400-row evaluation JSONLs, their manifests, and compact audit/provenance
files. It deliberately does not transfer selected traces, tokenized training
JSONLs, intermediate HF snapshots, or full optimizer state. Hyak pulls this
same pinned repository revision for the evaluator; every delayed eval job also
fails on revision or worktree drift.

The Git provenance publication includes every optimizer update's realized loss,
gradient norm, learning rate, and global batch size, plus a human-readable
first/last-loss table. Generate it once from the immutable completed logs, then
run the guarded transfer sequence on Tillicum:

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 python3 \
  examples/qwen2_5_7b_aime_generalization_incorrect_sft/publish_provenance.py \
  --experiment-root /gpfs/scrubbed/suryadv/slime-qwen2-5-7b-aime-generalization-incorrect-sft/v1/ded3390ab073b15f \
  --contract examples/qwen2_5_7b_aime_generalization_incorrect_sft/config/experiment_contract.json \
  --contract-hash ded3390ab073b15f \
  --output-root examples/qwen2_5_7b_aime_generalization_incorrect_sft/provenance/v1/ded3390ab073b15f

bash examples/qwen2_5_7b_aime_generalization_incorrect_sft/rsync_to_klone.sh --check
bash examples/qwen2_5_7b_aime_generalization_incorrect_sft/rsync_to_klone.sh --transfer
bash examples/qwen2_5_7b_aime_generalization_incorrect_sft/rsync_to_klone.sh --verify
```

`--transfer` opens one authenticated SSH control connection and multiplexes all
checkpoint and metadata rsync processes through it, so it requires one Duo
login rather than one per file. It uses `--partial --append-verify`: rerunning
after an interruption skips completed files, verifies/resumes partial files,
and converges each dedicated final-checkpoint directory to its source. It does
not perform remote checkpoint hashing inline. Run `--verify` afterward; that
command uses one separate SSH login and verifies all 12 trained checkpoints,
the base checkpoint, both eval sets, and every compact provenance file in that
single remote session.

After pulling the pinned branch on Hyak, validate Slurm shapes and submit the
three-sample held-in/held-out evaluation chain:

```bash
bash examples/qwen2_5_7b_aime_generalization_incorrect_sft/eval/submit_hyak.sh --test-only-shapes
bash examples/qwen2_5_7b_aime_generalization_incorrect_sft/eval/submit_hyak.sh --submit
# Only after an audited infrastructure/runtime failure:
bash examples/qwen2_5_7b_aime_generalization_incorrect_sft/eval/submit_hyak.sh --resubmit
```

The evaluator passes each transferred row's decoded `prompt` value verbatim as
one user message, with no system message or added prefix/suffix, before applying
the Qwen chat template with `add_generation_prompt=true`. Sampling uses three
paired repeats at temperature 0.7 and top-p 0.8. The base reference and 3K
correct checkpoint start after the canary; trained checkpoints then proceed in
priority order through unmasked, 1.5K correct, and random-mask 50. A CPU interim
table is finalized for those five targets before masks 10, 20, 30, 40, 60, 70,
80, and 90 are released concurrently. Scoring uses the existing strict
`aime_last_boxed_integer_scorer_v1` implementation in `eval/aime_scorer.py`.

Transferred trained checkpoints retain a legacy tokenizer serialization in
which `extra_special_tokens` is a list. Pinned Transformers 4.57.6 expects that
field to be a mapping. Evaluation therefore builds a content-addressed tokenizer
overlay outside the checkpoint: it preserves the original `tokenizer.json` and
chat template, moves the legacy list to `additional_special_tokens`, and sets
`extra_special_tokens` to an empty mapping. Checkpoint files and manifest
identities remain unchanged. Preflight proves the rendered probe, vocabulary
size, EOS IDs, and every Qwen special-token ID agree across all 13 targets; the
H200 canary uses the affected 3K trained checkpoint, and both prompt rendering
and vLLM consume the same overlay. Preflight, canary, and generation shards also
pin `/sw/cuda/12.8.1`, put its executable `nvcc` first on `PATH`, and verify its
12.8 release banner before vLLM starts. This avoids inheriting a non-executable
compiler entry when Torch 2.8 serializes its Inductor graph. A guarded
`--resubmit` accepts only the known legacy-tokenizer failure, the preserved
metadata-publication worktree race, or that exact canary `nvcc` permission
failure; it archives the superseded journal, cancels its remaining dependency
graph, and records the replacement attempt and failure provenance.
Submission metadata must be published through an isolated Git index; the live
checkout stays clean and pinned to the exported runtime SHA while delayed jobs
execute their revision gates.
