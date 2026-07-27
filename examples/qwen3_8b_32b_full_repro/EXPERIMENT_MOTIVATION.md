# Experiment motivation

We originally avoided a full 200K SFT run because it appeared likely to take
several days and a cheaper scaled reproduction could answer useful questions
first. We have now completed those smaller explorations: 1.7B -> 8B OPD, 50K
and smaller SFT initializations, 16K rollout-cap experiments, stopping and
scorer audits, and termination-failure analysis.

That work established two important findings. Rollout truncation does not teach
stopping, and pure sampled reverse-KL can improve its own metric while useful
generation behavior deteriorates. The 1.7B student became longer and less
likely to terminate, lost useful MATH/AIME behavior despite lower sampled
reverse-KL, may lack the capacity for the published gain, and would remain a
scaled analogue rather than the published result.

We now want the main result even if it takes several extra days. Evaluation and
artifact handling are more reliable than when the project began, and a faithful
8B -> 32B endpoint is a stronger foundation for the next OPD project. This is a
deliberate change of direction, not an erasure: all smaller experiments,
failures, fixes, and conclusions remain preserved as provenance.

The 200K SFT is intended to establish stronger long-form mathematical reasoning,
formatting, and teacher-compatible token distributions than the prior weak
initialization. Termination can still remain imperfect because `</think>`, the
final answer, and terminal chat tokens occupy very few positions in long
token-normalized sequences. The 16K OPD cutoff is retained because it matches
the public SLIME configuration and bounds compute; it is not treated as an EOS
target or stopping objective. Increasing the cutoff would not add correctness,
EOS, or terminal supervision and could merely permit longer loops.

## Public sources

- SLIME OPD example, retrieved 2026-07-19: https://github.com/THUDM/slime/tree/main/examples/on_policy_distillation
- Public Qwen3-8B OPD script, retrieved 2026-07-19: https://github.com/THUDM/slime/blob/main/examples/on_policy_distillation/run-qwen3-8B-opd.sh
- Contributor details in issue #1493, retrieved 2026-07-19: https://github.com/THUDM/slime/issues/1493
- Qwen3 technical report, retrieved 2026-07-19: https://arxiv.org/abs/2505.09388
- Thinking Machines OPD article, retrieved 2026-07-19: https://thinkingmachines.ai/blog/on-policy-distillation/
