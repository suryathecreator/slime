# OPD learning dynamics

Each row is a consecutive 23-rollout window containing 1,472 trajectories.

| Rollouts | Mean tokens | Cap hit | Think closed | Eligible final | `<|im_end|>` | Repetition | Sampled reverse-KL |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-22 | 13309.4 | 54.14% | 32.95% | 84.10% | 30.50% | 16.44% | 0.6422 |
| 23-45 | 13605.5 | 56.45% | 42.87% | 86.41% | 37.50% | 19.50% | 0.2164 |
| 46-68 | 13285.5 | 47.69% | 47.69% | 84.99% | 51.90% | 10.05% | 0.2109 |
| 69-91 | 12493.2 | 40.15% | 51.43% | 86.41% | 59.24% | 9.17% | 0.2013 |

From the first to final window:

- Think closure changed by +18.48 points.
- Natural stopping changed by +13.99 points.
- `<|im_end|>` termination changed by +28.74 points.
- Cap hits changed by -13.99 points.
- Mean response length changed by -816.2 tokens.
- Eligible final answers changed by +2.31 points.
- Repetition indicators changed by -7.27 points.
- Sampled reverse-KL changed by -0.4409.

Not measured: OpenThoughts rollout responses are not verified gold answers.
Windows contain different prompts, so changes are descriptive training-time trends, not a controlled within-prompt causal estimate.
The 16K cutoff bounded compute and supplied no synthetic terminal target. Think closure and natural stopping therefore report observed behavior, not an explicit length reward.
