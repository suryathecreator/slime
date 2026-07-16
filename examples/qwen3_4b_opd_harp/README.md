# Qwen3-4B OPD with HARP Evaluator V2

This directory is the isolated replacement for the earlier Qwen3-8B-Base/Math500 sequence. It starts from the post-trained `Qwen/Qwen3-4B`, evaluates it on a pinned HARP-500 split, runs OPD for 1,024 and then 4,096 additional prompts, and evaluates the cumulative 5,120-prompt checkpoint. The existing `Qwen/Qwen3-32B` teacher receives a base evaluation on the same ordered HARP rows.

SFT is intentionally skipped. The post-trained student already has useful base capability and is more likely to terminate naturally, while changing the benchmark to HARP leaves sufficient accuracy headroom for direct OPD. HARP results from this sequence must not be compared numerically with historical Math500 results.

## Prompt contract

Evaluation, OPD, and any future SFT in this experiment must use exactly:

```text
Please reason step by step, and put your final answer within \boxed{}.

Problem:
{problem}
```

`prompt_contract.py` is the only prompt constructor. OPD preparation rewrites each source problem through it before Slime applies the native Qwen3 chat template with thinking enabled. A future SFT stage must run its prompt or message rows through the same module and retain `prompt_contract_version=qwen3_harp_boxed_v1`; legacy prompt text is not permitted.

## Evaluation

`evaluate_harp_vllm.py` uses direct synchronous offline `vllm.LLM`. Four processes each own one H200 and one tensor-parallel-1 engine. Each worker gives its entire deterministic 125-problem shard to one `llm.generate()` call, allowing vLLM to continuously batch internally without reloading the checkpoint or issuing one generation call per problem.

The model-specific tuning job tests CUDA-graph `max_num_seqs` values 4 and 8 on the same 64 prompts, falls back to 2 if necessary, and only tests eager mode when every graph configuration fails. Selection uses aggregate generated tokens per four-worker makespan. Production uses bfloat16, a 32,768-token model limit, 31,744-token response cap, 0.92 GPU-memory utilization, prefix caching, chunked prefill, and 16,384 batched tokens.

EOS and `<|im_end|>` are resolved from each tokenizer and recorded in every manifest. Length-cap generations are retained and scored.

## HARP V2

The source archive and seed-42 subset are pinned by SHA-256 in `prepare_experiment.py`. The official HARP checker under `harp_official/` is byte-identical to the audited vendored source; all new behavior lives in `harp_answer_v2.py`.

V2 isolates post-thinking text, performs balanced box extraction, supports trailing multi-box collections, applies deterministic marker fallbacks, bounds every symbolic attempt to ten seconds, and records complete per-answer audit data. Fresh merges require exactly one prediction for every benchmark ID in canonical order. Historical rescoring is available through the `rescore` subcommand and writes a separate `rescore_harp_answer_v2/` directory rather than editing source artifacts.

## OPD and resources

Both OPD stages use three colocated student/rollout GPUs and one 32B-teacher GPU. A two-update smoke job compares the established CP3/2,048/512 configuration with CP1/4,096/1,024 and CP1/8,192/2,048, rejecting OOM, non-finite, incomplete, or invalid-checkpoint runs. The additional 4,096 prompts resume the complete optimizer, RNG, and training state from the 1,024 checkpoint.

All jobs are connected by strict `afterok` dependencies and are serialized, so this sequence uses at most four GPUs at any time. The cluster requires every `gpu-h200` job to reserve a GPU, so each small report requests one GPU and completes before the next four-GPU stage starts.

```text
setup → 4B tune → 4B base eval → 32B tune → 32B base eval
                                            → base report → OPD tune → OPD-1K → eval
                                                                                  → OPD-5.1K → eval → final report
```

Submit from the repository root with:

```bash
bash examples/qwen3_4b_opd_harp/submit_chain.sh
```

Every stage writes to `/gpfs/scrubbed/suryadv/slime-qwen3-4b-opd-harp-v2`; earlier Math500 and 8B artifacts are never overwritten.
