# Cleaned SFT Eval Special-Token Incident

## What We Observed

Corrected cleaned SFT job `165695` trained the intended Qwen3 loss targets and
saved final snapshot `iter_0000099`. Its first resumable MATH-500 eval,
`166666`, exposed a separate generation-control problem.

At the inspection point, 12 chunks covered 192 of 500 examples. Every saved
response had all of the following properties:

| Field | Observed value |
| --- | ---: |
| Completed examples | 192 / 500 |
| Mean generated tokens | 31,744 |
| Minimum generated tokens | 31,744 |
| Maximum generated tokens | 31,744 |
| `finish_reason=length` | 192 / 192 |
| Cap-hit rate | 1.000 |
| Decoded `<think>` or `</think>` visible | 0 / 192 |

The old artifact format saved vLLM's decoded `completion.text` but discarded
the generated token ids. vLLM also defaults to skipping special tokens. The
absence of visible `<think>`, `</think>`, or `<|im_end|>` therefore does **not**
show that the model never emitted them. The old files cannot answer that
question.

The corrected SFT target traces end with `<|im_end|>` (token id `151645`),
while the old eval only reliably honored the model-config EOS
`<|endoftext|>` (token id `151643`). The leading eval-side hypothesis is that
the model emitted chat control tokens that were hidden and/or not treated as
terminal. This combines two problems: the evaluator was unauditable, and its
stop contract did not match the SFT chat target.

The 12 old chunks are preserved under a `pre_special_token_fix_166666`
diagnostic path. They must not be merged with corrected chunks because their
generation policy and schema differ.

## Corrected Generation Contract

All Qwen3 prompt-generation paths in this Tillicum experiment now explicitly
use `enable_thinking=True`. Future vLLM eval and SGLang OPD generation stop on
either:

| Token | ID | Reason |
| --- | ---: | --- |
| `<|im_end|>` | 151645 | Qwen3 chat-turn terminator used by the corrected SFT traces. |
| `<|endoftext|>` | 151643 | Base-model/config EOS and a valid terminal condition. |

Eval samples preserve special tokens in `response` and separately save exact
`prefill_token_ids`, exact vLLM `generated_token_ids`, any separately reported
`terminal_stop_token_id`, and `generated_token_ids_with_terminal`. The scorer
removes terminal control markers only from a temporary scoring view; it does
not rewrite the saved response.

Resumable chunks now carry a generation-policy fingerprint. A chunk is reused
only when its model, thinking mode, stop ids, decoding settings, context limits,
and artifact schema match exactly.

The preserved base result remains a useful reference:

| Stage | Accuracy | Parse failure | Cap hit |
| --- | ---: | ---: | ---: |
| Base (`165035`) | 0.612 | 0.176 | 0.040 |

The base checkpoint was not chat-SFT-trained around `<|im_end|>` and did not
show the SFT eval's 100% cap-hit pathology. It does not need to be rerun for
this comparison. Any future base eval will still accept both terminal tokens.

## Cleaning Gap Found During This Investigation

Our incomplete-thinking count is close to the reference implementation, while
our mixed-language filter is substantially less aggressive:

| Count | This run | Reference implementation |
| --- | ---: | ---: |
| Incomplete thinking trace | 743,814 | 749,380 |
| Mixed-language | 94,920 | 397,642 |
| Valid after both filters | 429,713 | 332,843 |

The removal categories overlap, so the rows cannot be derived by simply
subtracting both rejection counts from 1.2M. The main discrepancy is still
clear: the local script-ratio heuristic retains far more language noise.

For example, cleaned source row `655427` is mostly English but contains
fragments such as `Por lo tanto`, `因此`, and Korean text. Because the answer is
very long and English-dominated, those fragments remain below the current
non-Latin ratio threshold.

If corrected stopping/auditing does not resolve most of the behavior, the next
data pass should target:

- English answers with small non-English contamination;
- pure non-English answers written in Latin script;
- multilingual answers where one language dominates;
- noisy/web text that script ratios miss.

That pass must preserve LaTeX, equations, mathematical symbols, variable names,
and normal technical notation. Some data noise is acceptable; the goal is to
remove the worst contamination rather than make the corpus unnaturally sterile.
No new filtering is applied before the corrected eval establishes how much of
the observed failure was eval-side.

## Optional Later Diagnostic

If the corrected standard eval still does not expose a normal thinking trace,
run a separately labeled diagnostic that prefills the assistant with
`<think>\n`. If the model then reasons, emits `</think>`, and terminates, that
would distinguish failure to initiate thinking from failure to reason once
thinking has started. This is not part of the primary eval because it changes
the prompt contract.
