"""No-compiler vLLM sampler fallback for greedy eval jobs."""

from __future__ import annotations

import os
import sys
from types import ModuleType
from typing import Any

_TRUTHY = {"1", "true", "yes", "on"}
_PATCH_INSTALLED = False
_IMPORT_HOOK_INSTALLED = False


def _enabled(name: str) -> bool:
    return os.environ.get(name, "0").lower() in _TRUTHY


def _native_functions() -> tuple[Any, Any]:
    import torch

    def native_apply_temperature(
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
    ) -> None:
        token_temperature = temperature[expanded_idx_mapping].to(torch.float32)
        needs_scale = (token_temperature != 0.0) & (token_temperature != 1.0)
        if torch.any(needs_scale):
            logits[needs_scale] = logits[needs_scale] / token_temperature[needs_scale].unsqueeze(-1)

    def native_gumbel_sample(
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        seed: torch.Tensor,
        pos: torch.Tensor,
        apply_temperature: bool,
        output_processed_logits: torch.Tensor | None = None,
        output_processed_logits_col: torch.Tensor | None = None,
        use_fp64: bool = False,
    ) -> torch.Tensor:
        del seed, pos, output_processed_logits_col, use_fp64
        if apply_temperature:
            native_apply_temperature(logits, expanded_idx_mapping, temperature)
        if output_processed_logits is not None:
            raise RuntimeError(
                "VLLM_EVAL_FORCE_NATIVE_SAMPLER supports greedy eval only, "
                "not processed-logit output."
            )
        return torch.argmax(logits, dim=-1).to(torch.int64)

    return native_apply_temperature, native_gumbel_sample


def _patch_loaded_modules() -> bool:
    global _PATCH_INSTALLED

    gumbel_mod = sys.modules.get("vllm.v1.worker.gpu.sample.gumbel")
    sampler_mod = sys.modules.get("vllm.v1.worker.gpu.sample.sampler")
    states_mod = sys.modules.get("vllm.v1.worker.gpu.sample.states")
    if gumbel_mod is None and sampler_mod is None and states_mod is None:
        return False

    native_apply_temperature, native_gumbel_sample = _native_functions()
    if isinstance(gumbel_mod, ModuleType):
        gumbel_mod.apply_temperature = native_apply_temperature
        gumbel_mod.gumbel_sample = native_gumbel_sample
    if isinstance(states_mod, ModuleType):
        states_mod.apply_temperature = native_apply_temperature
    if isinstance(sampler_mod, ModuleType):
        sampler_mod.gumbel_sample = native_gumbel_sample

    _PATCH_INSTALLED = True
    return True


def maybe_force_native_sampler() -> bool:
    if not _enabled("VLLM_EVAL_FORCE_NATIVE_SAMPLER"):
        return False

    from vllm.v1.worker.gpu.sample import gumbel as gumbel_mod
    from vllm.v1.worker.gpu.sample import sampler as sampler_mod
    from vllm.v1.worker.gpu.sample import states as states_mod

    del gumbel_mod, sampler_mod, states_mod
    return _patch_loaded_modules()


def install_native_sampler_import_hook() -> bool:
    global _IMPORT_HOOK_INSTALLED

    if _IMPORT_HOOK_INSTALLED:
        return False

    import builtins

    original_import = builtins.__import__

    def slime_vllm_import_hook(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        module = original_import(name, globals, locals, fromlist, level)
        if name.startswith("vllm.v1.worker.gpu.sample"):
            _patch_loaded_modules()
        return module

    builtins.__import__ = slime_vllm_import_hook
    _IMPORT_HOOK_INSTALLED = True
    _patch_loaded_modules()
    return True
