# Qwen2.5-7B base plus 2K suite handoff

This handoff transfers and evaluates the pinned Qwen2.5-7B base and all eleven
2K full-SFT variants. Tillicum produces content-addressed checkpoint manifests;
Klone verifies every file before inference.

The evaluation is the established Axolotl MATH-500 pipeline at commit
`6b8f0e3314e3d162260cdc35d84741c3da163f30`: greedy decoding, 32K model
length, a request for 32K new tokens, Qwen EOS and `<|im_end|>` stop IDs, vLLM
0.10.2, transformers 4.57.6, torch 2.8.0, and math-verify 0.9.0. Each of the 12
checkpoints uses four contiguous 125-problem shards, for 48 one-H200 tasks.

The tokenizer overlay is byte-identical to the pinned base
`Qwen/Qwen2.5-7B@d149729398750b98c0af14eb82c78cfe92750796`; Qwen2.5 needs no
metadata compatibility rewrite.

After every training job and manifest has finished:

```bash
bash examples/qwen2_5_7b_openr1_math220k_masked_sft_eval_handoff/rsync_to_klone.sh --check
bash examples/qwen2_5_7b_openr1_math220k_masked_sft_eval_handoff/rsync_to_klone.sh --transfer
bash examples/qwen2_5_7b_openr1_math220k_masked_sft_eval_handoff/rsync_to_klone.sh --verify
```

Remote verification uses the sibling Axolotl checkout's `.venv/bin/python`,
not Klone's legacy system `python3`. Set `REMOTE_PYTHON` only if that venv is
located elsewhere.

On Klone, with this branch checked out and the sibling Axolotl repository at
the pinned commit:

```bash
bash examples/qwen2_5_7b_openr1_math220k_masked_sft_eval_handoff/submit_available_axolotl.sh --test-only-shapes
bash examples/qwen2_5_7b_openr1_math220k_masked_sft_eval_handoff/submit_available_axolotl.sh --submit
```

`checkpoint_sources.json` names every Tillicum source. The Hyak-relative
runtime inventory is `available_eval_manifest.json`.
