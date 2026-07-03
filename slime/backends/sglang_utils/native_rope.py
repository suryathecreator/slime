"""Small SGLang compatibility shim for clusters without runtime CUDA compilers."""

from __future__ import annotations

import os
import sys
from functools import wraps

_PATCHED = False
_TRUTHY = {"1", "true", "yes", "on"}


def force_native_rope_enabled() -> bool:
    return os.environ.get("SLIME_SGLANG_FORCE_NATIVE_ROPE", "0").lower() in _TRUTHY


def maybe_force_native_rope() -> bool:
    """Route SGLang RoPE through its native PyTorch path when requested.

    The Tillicum Apptainer sandbox intentionally avoids a full CUDA development
    toolchain. SGLang's default Qwen RoPE path JIT-compiles a fused CUDA kernel
    on first warmup, which fails without gcc/CUDA headers. This patch is narrow:
    it leaves SGLang server args alone and only changes RotaryEmbedding objects
    created after the patch to call ``forward_native``.
    """

    global _PATCHED
    if _PATCHED or not force_native_rope_enabled():
        return _PATCHED

    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding

    if getattr(RotaryEmbedding, "_slime_native_rope_patched", False):
        _PATCHED = True
        return True

    original_init = RotaryEmbedding.__init__

    @wraps(original_init)
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._forward_method = self.forward_native

    RotaryEmbedding.__init__ = patched_init
    RotaryEmbedding._slime_native_rope_patched = True
    _PATCHED = True
    print(
        "SLIME_SGLANG_FORCE_NATIVE_ROPE=1: patched SGLang RotaryEmbedding to use forward_native.",
        file=sys.stderr,
    )
    return True
