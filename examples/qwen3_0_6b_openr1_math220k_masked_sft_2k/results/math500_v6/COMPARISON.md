# MATH-500 V5 to V6 scorer comparison

The saved generations and extracted boxed spans are unchanged. V6 reverses eight V5 false positives.

| Checkpoint | Problem | V5 | V6 | Extracted answer | Gold |
|---|---|---:|---:|---|---|
| base_0p6b | `test/number_theory/598.json` | 1 | 0 | `0.00006666666666666667` | `.0000672` |
| base_0p6b | `test/geometry/826.json` | 1 | 0 | `1 \frac{1}{2} \sqrt{5}` | `1\frac{4}{5}` |
| base_0p6b | `test/intermediate_algebra/1566.json` | 1 | 0 | `2a + 2b + 3c + 2k` | `2k` |
| random_mask_25 | `test/counting_and_probability/119.json` | 1 | 0 | `-125x^{12}` | `-125` |
| random_mask_50 | `test/prealgebra/1924.json` | 1 | 0 | `2x + 2` | `2k+2` |
| random_mask_70 | `test/precalculus/1291.json` | 1 | 0 | `-9 - 12i` | `1 - 12i` |
| inverse_tau_0p20 | `test/precalculus/1252.json` | 1 | 0 | `1 - i\sqrt{3}` | `1` |
| inverse_tau_0p05 | `test/precalculus/1291.json` | 1 | 0 | `-9 - 12i` | `1 - 12i` |
