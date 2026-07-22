# Results

Execution is partial. Base, teacher, and 200K-SFT MATH-500 evaluations completed
under contract `4095e6ac256b0dbf`. The first OPD attempt failed during engine
startup before rollout 0 because the 32B teacher process did not receive the
pinned C compiler; no OPD update or checkpoint was created.

| Checkpoint | Job | Correct | pass@1 | Bootstrap 95% CI | Parseable | Cap hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B-Base | 181406 | 235/500 | 47.0% | 42.6%–51.4% | 495/500 | 11/500 |
| Qwen3-32B teacher | 181407 | 476/500 | 95.2% | 93.2%–97.0% | 491/500 | 14/500 |
| 200K SFT | 181912 | 304/500 | 60.8% | 56.6%–65.0% | 445/500 | 87/500 |
| 100% OPD | 181913 | infrastructure failure | — | — | — | — |

The base response length was 920.32 tokens on average (median 453.5, p95
1286.05); the teacher response length was 4447.134 tokens on average (median
3180.5, p95 12225.3). The SFT response length was 6161.166 tokens on average
(median 3775, p90/p95 16384). Among its 413 naturally stopped rows, 300 were
correct (72.64%); among its 87 cap-hit rows, 4 were correct (4.60%). The
naturally stopped subset is selected by termination behavior and is therefore
not an alternative pass@1 estimate, but its 72–73% accuracy is within plausible
sampling noise of the reported post-SFT result before OPD.

The full immutable summaries remain outside Git. The base, teacher, and SFT
summary SHA-256 hashes are
`ba252b5bfa78e7e5ccc6f803c9af4a9d32df7121b0d87001a2761646fbf4fef9`,
`d1d8ee1fed7472f81c7524764b7a255e52779eabe137ee1a8ff393e31ad5d103`,
and `2f9937f2e6b49943fa0ea0d006ce80ea4526f436cf049718874225f2e120e180`.

SFT job `181408` passed the eight-row Qwen3 loss-mask preflight but OOMed at
microbatch 0/37 while converting the TP2 vocabulary logits to FP32. It tried to
allocate 9.27 GiB with only 3.64 GiB free on GPU 0. No optimizer update or
checkpoint completed. The failed log and rollout tensor are preserved and
hashed in `INFRASTRUCTURE_FAILURES.json`.

The 16,384-token packing retry `181873` passed the same preflight and completed
microbatch 1/77, proving that the original 32K logits OOM was removed. It then
failed in fused cross-entropy backward because TorchInductor's Ray workers did
not inherit a C compiler. It also completed no optimizer update or checkpoint.
Retry `181893` propagated the compiler but used an incorrect CPATH leaf, so
`Python.h` could not resolve its multiarch-prefixed `pyconfig.h`. An uncached
reproduction now passes with the exact proven eval CPATH
`python3.12:<include-root>`. All retries retain the 16K packing fix and have
completed no optimizer update or checkpoint.

The completed evaluation used the scripted fixed 16,384-token output-generation
cap, subject to the 32,768-token model context: the effective response budget
was `min(16384, 32768 - rendered_prompt_tokens - 64)`. Here the rendered prompt
contains the chat-template special tokens, problem, instruction, and assistant
prefix; the response contains all emitted reasoning, think tags, answer text,
and any emitted stop token. The final 64 tokens are reserved safety margin.
There is no separate hidden reasoning stream outside this accounting.

One hypothesis for the gap from the reported SFT score is that the reported run
was effectively evaluated with a 32K total context even though the released
scripts describe a 16K generation cap. The recovery chain will test that
hypothesis without changing this primary result: it will rerun only the 87
cap-hit prompts with a dynamic response budget of
`32768 - rendered_prompt_tokens - 64`, reuse all natural-stop generations, and
require every rerun's first 16K generated token IDs to match the saved source
generation exactly. This is an exact cap-hit-only proxy for the corresponding
32K-total-context evaluation if the prefix checks pass. Although one rule is a
fixed output cap and the other a dynamic total-context budget, prompts are short
enough that the effective lengths remain near the intended 16K and 32K limits;
the important requirement is to apply each rule consistently across stages.

Job `181913` loaded the 32B teacher and reached its first health generation,
where Triton raised `Failed to find C compiler`. The job exited after 3:03,
before rollout 0; its sanity audit contains zero rows. Recovery will forward the
same proven Zig `CC` and Python-header `CPATH` used by SFT and evaluation.

The final report will add OPD scores, paired transitions, bootstrap intervals,
McNemar tests, length and termination diagnostics, primary-versus-proxy context
results, checkpoint hashes, and all deviations. Results outside the 76% ±3 SFT
or 94% ±3 OPD targets, an OPD gain below 12 points, or termination collapse
will be reported unchanged.
