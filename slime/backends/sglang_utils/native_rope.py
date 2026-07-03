"""Small SGLang compatibility shim for clusters without runtime CUDA compilers."""

from __future__ import annotations

import os
import sys
from functools import wraps

_PATCHED = False
_TRUTHY = {"1", "true", "yes", "on"}


def force_native_rope_enabled() -> bool:
    return os.environ.get("SLIME_SGLANG_FORCE_NATIVE_ROPE", "0").lower() in _TRUTHY


def _patch_cached_rope_instances(factory_module) -> int:
    patched_count = 0
    for rope in getattr(factory_module, "_ROPE_DICT", {}).values():
        if hasattr(rope, "forward_native"):
            rope._forward_method = rope.forward_native
            patched_count += 1
    return patched_count


def maybe_force_native_rope() -> bool:
    """Route selected SGLang ops through native PyTorch paths when requested.

    The Tillicum Apptainer sandbox intentionally avoids a full CUDA development
    toolchain. SGLang's default Qwen RoPE, SiLU-mul activation, and
    clamp-position paths JIT-compile CUDA kernels on first warmup, which fail
    without gcc/CUDA headers. This patch is narrow: it leaves SGLang server
    args alone and only routes these known compiler-sensitive operations to
    their existing native PyTorch implementations.
    """

    global _PATCHED
    if not force_native_rope_enabled():
        return _PATCHED
    if _PATCHED:
        import sglang.srt.layers.rotary_embedding.factory as rotary_factory

        _patch_cached_rope_instances(rotary_factory)
        return True

    from sglang.srt.layers.activation import SiluAndMul
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
    import sglang.srt.layers.rotary_embedding.factory as rotary_factory
    import sglang.srt.model_executor.forward_batch_info as forward_batch_info

    rope_patched = getattr(RotaryEmbedding, "_slime_native_rope_patched", False)
    rope_forward_cuda_patched = getattr(RotaryEmbedding, "_slime_native_rope_forward_cuda_patched", False)
    activation_patched = getattr(SiluAndMul, "_slime_native_activation_patched", False)
    clamp_patched = getattr(forward_batch_info, "_slime_native_clamp_position_patched", False)
    if rope_patched and rope_forward_cuda_patched and activation_patched and clamp_patched:
        _patch_cached_rope_instances(rotary_factory)
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

    if not rope_forward_cuda_patched:
        original_rope_forward_cuda = RotaryEmbedding.forward_cuda

        @wraps(original_rope_forward_cuda)
        def patched_rope_forward_cuda(self, *args, **kwargs):
            return self.forward_native(*args, **kwargs)

        RotaryEmbedding.forward_cuda = patched_rope_forward_cuda
        RotaryEmbedding._slime_native_rope_forward_cuda_patched = True

    if not activation_patched:
        original_activation_init = SiluAndMul.__init__

        @wraps(original_activation_init)
        def patched_activation_init(self, *args, **kwargs):
            original_activation_init(self, *args, **kwargs)
            self._forward_method = self.forward_native

        SiluAndMul.__init__ = patched_activation_init
        SiluAndMul._slime_native_activation_patched = True

    if not clamp_patched:
        forward_batch_info.clamp_position = forward_batch_info._clamp_position_native
        forward_batch_info._slime_native_clamp_position_patched = True

    cached_rope_count = _patch_cached_rope_instances(rotary_factory)

    _PATCHED = True
    print(
        "SLIME_SGLANG_FORCE_NATIVE_ROPE=1: patched SGLang RotaryEmbedding init/forward_cuda, "
        f"SiluAndMul, clamp_position, and {cached_rope_count} cached RoPE object(s) to use native paths.",
        file=sys.stderr,
    )
    return True
