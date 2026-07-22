# OpenThoughts3 cleanup audit

- Dataset: `open-thoughts/OpenThoughts3-1.2M@61bcf9d4eb38b30295efc2021227a63cc5bb34c8`
- Raw rows: 1,200,000
- Incomplete think: 749,552 (gate [711911, 786849])
- Assistant responses containing CJK: 401,806 (legacy non-English gate [377760, 417524])
- Retained: 331,418 (gate [316201, 349485])
- Gate passed: `True`
- Filter: any Han, Hiragana, Katakana, Hangul, or Bopomofo Unicode letter in the assistant response only
- Unicode database: 15.0.0
- Split: seeded raw-row shuffle; 200,000 SFT then 100,000 OPD; source-row-ID overlap only
- Prompt grouping/deduplication: disabled
- MATH-500 cross-contamination check: disabled

See the JSON audit and compressed per-row classifications for full overlap and CJK-match details.
