# Qwen2.5-7B OpenR1-Math 2K masked full SFT

This experiment reproduces the existing eleven-run 2K masked full-SFT suite
from the base model `Qwen/Qwen2.5-7B` at revision
`d149729398750b98c0af14eb82c78cfe92750796`. It is full-parameter Slime SFT,
not LoRA and not an Axolotl training run.

The canonical Axolotl seed-42 trace identities and ordering are frozen. The
Qwen2.5-7B tokenizer rebuilds every exact-token training record, and the
Qwen2.5-7B base recomputes every conditioned-minus-unconditioned wrong-token
margin. No Qwen3 tokenized record or margin is reused.

Correct-only contains 2,000 correct traces. Every mixed variant uses the same
ordered 1,000 correct plus 1,000 wrong traces. The variants are `unmasked`,
random wrong-token masking at 25%, 50%, 70%, 80%, and 90%, inverse-margin
weighting at tau 0.20 and 0.05, positive-margin masking, and probability-ratio
masking. The training recipe is one epoch, global/rollout batch 200, LR
`1e-6`, constant schedule, weight decay 0.1, zero dropout, and a 32K context.

## Non-negotiable special-token supervision

Every assistant token whose ID belongs to the pinned tokenizer's added-token
vocabulary is protected. Every token piece in every literal `<think>` and
`</think>` occurrence is also protected, including occurrences in the frozen
missing- or duplicate-close anomalies. Protected positions are assigned loss
weight exactly 1.0 at record construction and are excluded from all random,
margin, inverse-margin, and probability-ratio decisions.

The protected added-token inventory includes:

- `<|endoftext|>`, `<|im_start|>`, and `<|im_end|>`;
- object, box, quad, and vision start/end controls and vision/image/video pads;
- `<tool_call>`, `</tool_call>`, FIM controls, `<|repo_name|>`, and
  `<|file_sep|>`.

Unexpected embedded chat boundaries or EOS in a source assistant trace fail
closed instead of being removed or masked. The generated training terminal is
one `<|im_end|>` at weight 1.0. No synthetic `<|endoftext|>` is appended, and
only the template newline after `<|im_end|>` has weight zero. Prompt tokens
remain intact context outside the response-loss boundary. Runtime padding
outside the logical sequence remains unsupervised.

The invariant is checked after tokenization, after variant construction, at
each train-job dataset preflight, and again inside the rollout adapter before
an optimizer step.

## Launch and handoff

`submit_training_only.sh` requires the exact pushed feature branch and submits
a strict serial chain: pinned model download/conversion, 2K preparation, a
one-step smoke train, margin scoring/variant construction, and all eleven
full-SFT jobs. The maximum concurrent allocation is four H200s. Tillicum
submits no evaluation jobs.

Each final HF checkpoint is `weights/iter_0000009` with a content-addressed
manifest. The base and all eleven checkpoints are transferred and evaluated
through
`../qwen2_5_7b_openr1_math220k_masked_sft_eval_handoff/`.

## One-time model-validation repair

Initial model-preparation job `198094` completed the pinned base download but
failed because its validator iterated `BatchEncoding` keys instead of
normalizing `input_ids`. After the repair commit is pushed,
`resubmit_after_model_validation.sh` submits one replacement model-preparation
job and rewires pending data job `198095`. Jobs `198095` through `198108`
remain the original strict downstream chain; the repair writes an immutable
manifest under the experiment manifest directory.
