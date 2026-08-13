#!/usr/bin/env bash
# Shared H200 launch contract for the pinned PyTorch 2.8.0+cu128 runtime.

CUDA_TOOLKIT_ROOT=/sw/cuda/12.8.1
NVCC="$CUDA_TOOLKIT_ROOT/bin/nvcc"
[[ -x "$NVCC" ]] || {
  echo "Missing executable CUDA 12.8 compiler: $NVCC" >&2
  return 2
}
export CUDA_HOME="$CUDA_TOOLKIT_ROOT"
export CUDA_PATH="$CUDA_TOOLKIT_ROOT"
export PATH="$CUDA_TOOLKIT_ROOT/bin:$PATH"
nvcc_version="$("$NVCC" --version)"
[[ "$nvcc_version" =~ release[[:space:]]12\.8([,[:space:]]|$) ]] || {
  echo "CUDA compiler contract changed: $nvcc_version" >&2
  return 2
}
[[ "$(command -v nvcc)" == "$NVCC" ]] || {
  echo "CUDA compiler path was not pinned: $(command -v nvcc)" >&2
  return 2
}
