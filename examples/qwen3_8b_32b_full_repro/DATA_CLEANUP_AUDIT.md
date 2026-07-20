# Data cleanup audit

`clean_openthoughts3.py` reads the pinned 1.2M-row OpenThoughts3 revision and
records each rejection category independently so overlaps remain visible.
It checks the raw message fields in the fixed order `messages`, `conversations`,
then `conversation`; maps `role`/`from`/`speaker` and
`content`/`value`/`text`; and maps `human`/`prompt` to `user` and `gpt`/`model`
to `assistant`. If no message collection exists, it uses the first populated
field from `prompt`, `problem`, `question`, `instruction`, `input` and from
`response`, `completion`, `answer`, `solution`, `output`. The selected source
field is recorded for every row; no other schema inference is performed.
Source metadata fields `source`, `domain`, and `difficulty` are retained when
present.
Rows that already embed `<|im_start|>`, `<|im_end|>`, or `<|endoftext|>` in the
raw assistant text are rejected, not repaired, so the tokenizer template is the
only source of chat terminals and cannot double-append an in-data terminal.

An entry passes think validation only when the assistant response contains one
ordered `<think>...</think>` pair, nonempty thinking, nonempty content after the
close, and no obvious unclosed truncation. Invalid entries are never repaired.

Lingua 2.2.0 classifies the complete human prompt and assistant response
separately after whitespace normalization and masking large LaTeX/code spans.
Both must be confidently English. Raw predictions and confidence values are
recorded for every row; low-confidence rows remain an auditable rejection
bucket.

The independent issue #1493 reference counts and ±5% gates are:

| Category | Reference | Allowed range |
|---|---:|---:|
| incomplete think | 749,380 | 711,911–786,849 |
| non-English/mixed | 397,642 | 377,760–417,524 |
| retained | 332,843 | 316,201–349,485 |

After this pre-deduplication gate passes, accepted raw rows are shuffled with
seed 1234. No prompt grouping or prompt-text deduplication is performed. The
first 200,000 native-context-compatible rows become SFT and the next 100,000
become the OPD reserve. Only source-row-ID disjointness is enforced. The first
1,472 reserve rows form the headline OPD prompt file.

The generated JSON audit records source revision, raw count, independent
rejections, overlap matrix, accepted/rejected deterministic samples, language
classifications, overlength exclusions, row-ID hashes, output hashes, and exact
selection order. The committed audit is a template until the job completes;
large row-level classifications remain in scratch.
