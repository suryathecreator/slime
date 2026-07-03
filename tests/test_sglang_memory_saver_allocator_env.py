import pytest

from slime.ray.utils import apply_sglang_memory_saver_allocator_env


def test_memory_saver_uses_non_expandable_default(monkeypatch):
    monkeypatch.delenv("SLIME_SGLANG_PYTORCH_CUDA_ALLOC_CONF", raising=False)

    env = apply_sglang_memory_saver_allocator_env({}, enable_memory_saver=True)

    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:128"
    assert "expandable_segments" not in env["PYTORCH_CUDA_ALLOC_CONF"]


def test_memory_saver_rejects_expandable_override(monkeypatch):
    monkeypatch.setenv("SLIME_SGLANG_PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    with pytest.raises(RuntimeError, match="memory saver is incompatible"):
        apply_sglang_memory_saver_allocator_env({}, enable_memory_saver=True)


def test_non_memory_saver_preserves_existing_allocator(monkeypatch):
    monkeypatch.setenv("SLIME_SGLANG_PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    existing = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

    env = apply_sglang_memory_saver_allocator_env(existing, enable_memory_saver=False)

    assert env == existing
