# Evaluation protocol

Evaluate only Qwen3-8B-Base, Qwen3-32B teacher, final 200K SFT, and final 100%
OPD. Quarter OPD weights are retained but not evaluated automatically.

MATH-500 stays read-only and in its original order. Each prompt adds exactly:
`Please reason step by step, and put your final answer within \boxed{}.` The
pinned thinking-mode chat template is saved with its token count.

Sampling is temperature 0.8, top-p 0.7, top-k -1, one sample, and fixed
per-problem seeds derived from 1234. The configured maximum is 16,384 generated
tokens, clamped to model context minus rendered input minus a 64-token safety
margin. EOS and `<|im_end|>` both stop generation; `</think>` does not.

Four one-GPU vLLM workers share each four-H200 allocation. Each problem is
atomically persisted, so requeue resumes compatible completed rows. Output
includes raw text and token IDs, prompts, scorer traces, pass@1, correct/500,
bootstrap intervals, paired transitions, McNemar tests, length percentiles,
cap/stop/think rates, parse failures, and exact checkpoint hashes.

The generation-time summaries remain immutable V1 evidence. The post-hoc V2
evaluation hashes and validates each saved per-problem artifact, then rescores
the same response text without loading a model, sampling, continuing a prefix,
or modifying any generation field. V2 outputs live under
`math500/rescored/math500_event_scorer_v2`; V1 remains alongside them as the
historical before-audit result. The final report includes both score sets and a
row-level decision-change audit.

Official references are calibration values, not tuning targets: post-trained
Qwen3-8B 97.4 and Qwen3-32B 97.2 on MATH-500; base Qwen3-8B 60.80 and base
Qwen3-32B 61.62 on the older MATH benchmark; Qwen3 8B ablation 92.4
off-policy, 94.8 after RL, and 97.0 after OPD. There is no official
Qwen3-8B-Base MATH-500 value.
