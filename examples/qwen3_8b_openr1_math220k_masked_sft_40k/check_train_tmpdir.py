#!/usr/bin/env python3
"""Fail fast if checkpoint multiprocessing cannot use the configured TMPDIR."""

from __future__ import annotations

import multiprocessing
import os
import tempfile
from pathlib import Path


MAX_TMPDIR_BYTES = 64


def main() -> None:
    configured = os.environ.get("TMPDIR")
    if not configured:
        raise SystemExit("TMPDIR is not set")
    resolved = tempfile.gettempdir()
    if resolved != configured:
        raise SystemExit(
            f"tempfile resolved {resolved!r}, expected configured TMPDIR {configured!r}"
        )
    if not resolved.startswith("/tmp/"):
        raise SystemExit(f"training TMPDIR must be node-local under /tmp: {resolved}")
    if len(os.fsencode(resolved)) > MAX_TMPDIR_BYTES:
        raise SystemExit(
            f"training TMPDIR exceeds {MAX_TMPDIR_BYTES} bytes: {resolved}"
        )
    if not Path(resolved).is_dir():
        raise SystemExit(f"training TMPDIR does not exist: {resolved}")

    # This is the same primitive that failed when torch_dist checkpointing
    # called multiprocessing.Manager().Queue() at iteration 49.
    with multiprocessing.Manager() as manager:
        queue = manager.Queue()
        queue.put("ready")
        if queue.get(timeout=5) != "ready":
            raise SystemExit("multiprocessing manager queue round trip failed")
    print(
        f"TRAIN_TMPDIR_PREFLIGHT_OK tmpdir={resolved} "
        f"bytes={len(os.fsencode(resolved))}",
        flush=True,
    )


if __name__ == "__main__":
    main()
