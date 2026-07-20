# Results

Execution is partial. Base and teacher MATH-500 evaluations completed under
contract `4095e6ac256b0dbf`; SFT failed before its first optimizer update, so no
SFT or OPD evaluation exists yet.

| Checkpoint | Job | Correct | pass@1 | Bootstrap 95% CI | Parseable | Cap hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B-Base | 181406 | 235/500 | 47.0% | 42.6%–51.4% | 495/500 | 11/500 |
| Qwen3-32B teacher | 181407 | 476/500 | 95.2% | 93.2%–97.0% | 491/500 | 14/500 |
| 200K SFT | — | pending | pending | — | — | — |
| 100% OPD | — | pending | pending | — | — | — |

The base response length was 920.32 tokens on average (median 453.5, p95
1286.05); the teacher response length was 4447.134 tokens on average (median
3180.5, p95 12225.3). The full immutable summaries remain outside Git. Their
SHA-256 hashes are
`ba252b5bfa78e7e5ccc6f803c9af4a9d32df7121b0d87001a2761646fbf4fef9`
and `d1d8ee1fed7472f81c7524764b7a255e52779eabe137ee1a8ff393e31ad5d103`.

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

The final report will add SFT and OPD scores, paired transitions, bootstrap
intervals, McNemar tests, length and termination diagnostics, checkpoint
hashes, and all deviations. Results outside the 76% ±3 SFT or 94% ±3 OPD
targets, an OPD gain below 12 points, or termination collapse will be reported
unchanged.
