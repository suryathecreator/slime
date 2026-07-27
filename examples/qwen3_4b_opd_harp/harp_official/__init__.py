"""Vendored official HARP answer-checking code.

This package contains the HARP short-answer checker copied from
`aadityasingh/HARP` under its MIT license. The HARP vLLM experiment imports
this package directly so reported HARP scores use the benchmark's own
answer-checking logic rather than this repo's historical `math_verify` or
string-normalization fallback.
"""

