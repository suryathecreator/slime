"""Optional process-startup patches for managed cluster runtimes."""

from __future__ import annotations

import os
import sys

_TRUTHY = {"1", "true", "yes", "on"}


def _enabled(name: str) -> bool:
    return os.environ.get(name, "0").lower() in _TRUTHY


if _enabled("SLIME_SGLANG_PATCH_SITE"):
    from slime.backends.sglang_utils.native_rope import maybe_force_native_rope

    if maybe_force_native_rope():
        print(
            "SLIME_SGLANG_PATCH_SITE=1: applied SGLang native compatibility patches at Python startup.",
            file=sys.stderr,
        )

if _enabled("SLIME_VLLM_PATCH_SITE"):
    from slime.backends.vllm_utils.native_sampler import install_native_sampler_import_hook

    if install_native_sampler_import_hook():
        print(
            "SLIME_VLLM_PATCH_SITE=1: registered vLLM V1 native sampler startup patch.",
            file=sys.stderr,
        )
