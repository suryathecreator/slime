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
    """Route selected SGLang ops through native PyTorch paths when requested.

    The Tillicum Apptainer sandbox intentionally avoids a full CUDA development
    toolchain. SGLang's default Qwen RoPE and SiLU-mul activation paths
    JIT-compile CUDA kernels on first warmup, which fail without gcc/CUDA
    headers. This patch is narrow: it leaves SGLang server args alone and only
    changes objects created after the patch to call their existing
    ``forward_native`` implementations.
    """

    global _PATCHED
    if _PATCHED or not force_native_rope_enabled():
        return _PATCHED

    from sglang.srt.layers.activation import SiluAndMul
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding

    rope_patched = getattr(RotaryEmbedding, "_slime_native_rope_patched", False)
    activation_patched = getattr(SiluAndMul, "_slime_native_activation_patched", False)
    if rope_patched and activation_patched:
        _PATCHED = True
        return True

    if not rope_patched:
        original_rope_init = RotaryEmbedding.__init__

        @wraps(original_rope_init)
        def patched_rope_init(self, *args, **kwargs):
            original_rope_init(self, *args, **kwargs)
            self._forward_method = self.forward_native

        RotaryEmbedding.__init__ = patched_rope_init
        RotaryEmbedding._slime_native_rope_patched = True

    if not activation_patched:
        original_activation_init = SiluAndMul.__init__

        @wraps(original_activation_init)
        def patched_activation_init(self, *args, **kwargs):
            original_activation_init(self, *args, **kwargs)
            self._forward_method = self.forward_native

        SiluAndMul.__init__ = patched_activation_init
        SiluAndMul._slime_native_activation_patched = True

    _PATCHED = True
    print(
        "SLIME_SGLANG_FORCE_NATIVE_ROPE=1: patched SGLang RotaryEmbedding and SiluAndMul to use forward_native.",
        file=sys.stderr,
    )
    return True
