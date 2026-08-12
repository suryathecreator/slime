#!/usr/bin/env python3
"""Fail-closed structural completeness gate for a finalized HF checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


def safetensors_names(path: Path) -> set[str]:
    size = path.stat().st_size
    if size < 9:
        raise ValueError(f"safetensors shard is too short: {path}")
    with path.open("rb") as handle:
        header_size_bytes = handle.read(8)
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        if header_size == 0 or header_size > 100 * 1024 * 1024:
            raise ValueError(f"invalid safetensors header size: {path}/{header_size}")
        if 8 + header_size > size:
            raise ValueError(f"truncated safetensors header: {path}")
        header = json.loads(handle.read(header_size))
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header is not an object: {path}")
    tensor_items = {str(name): value for name, value in header.items() if name != "__metadata__"}
    if not tensor_items:
        raise ValueError(f"safetensors shard has no tensors: {path}")
    data_size = size - 8 - header_size
    intervals = []
    for name, value in tensor_items.items():
        if not isinstance(value, dict):
            raise ValueError(f"malformed safetensors tensor metadata: {path}/{name}")
        offsets = value.get("data_offsets")
        shape = value.get("shape")
        dtype = value.get("dtype")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(item, int) and not isinstance(item, bool) for item in offsets)
            or not isinstance(shape, list)
            or not all(
                isinstance(dimension, int) and not isinstance(dimension, bool) and dimension >= 0
                for dimension in shape
            )
            or dtype not in DTYPE_BYTES
        ):
            raise ValueError(f"malformed safetensors tensor entry: {path}/{name}")
        start, end = offsets
        if start < 0 or end < start or end > data_size:
            raise ValueError(f"safetensors offsets out of bounds: {path}/{name}")
        expected_bytes = math.prod(shape) * DTYPE_BYTES[dtype]
        if end - start != expected_bytes:
            raise ValueError(f"safetensors tensor byte size disagrees with shape/dtype: {path}/{name}")
        intervals.append((start, end, name))
    cursor = 0
    for start, end, name in sorted(intervals):
        if start != cursor:
            raise ValueError(
                f"safetensors data is gapped/overlapping: {path}/{name} "
                f"expected_start={cursor} actual_start={start}"
            )
        cursor = end
    if cursor != data_size:
        raise ValueError(f"safetensors final data offset differs from file size: {path}")
    return set(tensor_items)


def validate_checkpoint(checkpoint: Path) -> int:
    checkpoint = checkpoint.resolve()
    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        path = checkpoint / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"HF checkpoint missing nonempty {filename}: {checkpoint}")
        if not isinstance(json.loads(path.read_text(encoding="utf-8")), dict):
            raise ValueError(f"HF JSON asset is not an object: {path}")
    if any(path.is_symlink() for path in checkpoint.rglob("*")):
        raise ValueError(f"HF checkpoint contains symlinks: {checkpoint}")
    index = None
    for filename in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        candidate = checkpoint / filename
        if candidate.is_file():
            if index is not None:
                raise ValueError("HF checkpoint has multiple weight indexes")
            index = candidate
    if index is not None:
        value = json.loads(index.read_text(encoding="utf-8"))
        weight_map = value.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"HF weight index has no weight_map: {index}")
        shards = sorted(set(map(str, weight_map.values())))
        names_by_shard: dict[str, set[str]] = {}
        for shard in shards:
            relative = Path(shard)
            if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".safetensors":
                raise ValueError(f"unsafe HF shard path: {shard}")
            path = checkpoint / relative
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"HF index references missing/empty shard: {path}")
            names_by_shard[shard] = safetensors_names(path)
        for shard, actual_names in names_by_shard.items():
            expected_names = {str(name) for name, mapped_shard in weight_map.items() if str(mapped_shard) == shard}
            if actual_names != expected_names:
                raise ValueError(f"HF index/tensor-name mismatch: {checkpoint}/{shard}")
    else:
        weight = checkpoint / "model.safetensors"
        if not weight.is_file() or weight.stat().st_size == 0:
            raise ValueError(
                "unindexed HF checkpoint needs a nonempty model.safetensors; "
                "pickle .bin integrity is intentionally unsupported"
            )
        safetensors_names(weight)
        shards = [weight.name]
    return len(shards)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    shard_count = validate_checkpoint(args.checkpoint)
    print(
        "AIME_HF_CHECKPOINT_COMPLETE " f"checkpoint={args.checkpoint.resolve()} weight_shards={shard_count}",
        flush=True,
    )


if __name__ == "__main__":
    main()
