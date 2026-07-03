"""Launch SGLang after applying the optional native-RoPE compatibility shim."""

from __future__ import annotations

import os
import sys

from slime.backends.sglang_utils.native_rope import maybe_force_native_rope

if os.environ.get("SLIME_SGLANG_FORCE_NATIVE_ROPE", "0").lower() in {"1", "true", "yes", "on"}:
    os.environ.setdefault("SLIME_SGLANG_PATCH_SITE", "1")
maybe_force_native_rope()

from sglang.launch_server import run_server  # noqa: E402
from sglang.srt.plugins import load_plugins  # noqa: E402
from sglang.srt.server_args import prepare_server_args  # noqa: E402
from sglang.srt.utils import kill_process_tree  # noqa: E402
from sglang.srt.utils.common import suppress_noisy_warnings  # noqa: E402


def main() -> None:
    suppress_noisy_warnings()
    load_plugins()
    server_args = prepare_server_args(sys.argv[1:])
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
