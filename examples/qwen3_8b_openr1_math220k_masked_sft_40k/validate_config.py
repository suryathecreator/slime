#!/usr/bin/env python3
"""Validate the immutable experiment contract and local prerequisites."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


BRANCH = "qwen3-8b-openr1-masked-sft-40k"


def sha256(path: Path) -> str:
    """Hash a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    """Run a read-only Git query."""
    return subprocess.check_output(["git", *args], text=True).strip()


def main() -> None:
    """Fail closed on configuration drift before a Slurm submission."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example-dir", type=Path, required=True)
    parser.add_argument("--require-pushed-clean", action="store_true")
    args = parser.parse_args()
    example = args.example_dir.resolve()
    repo = example.parents[1]
    contract = json.loads((example / "config/experiment_contract.json").read_text(encoding="utf-8"))
    if contract["training"] != {
        "full_sft": True,
        "global_batch_size": 200,
        "learning_rate": 1e-06,
        "max_sequence_length": 32768,
        "num_epochs": 1,
        "tau": 0.2,
        "variants": ["correct_only", "weighted_tau_0p20", "unmasked"],
    }:
        raise ValueError("training contract drift")
    if contract["evaluation"]["stop_token_ids"] != [151643, 151645]:
        raise ValueError("evaluation must accept both endoftext and im_end")
    base_summary = Path(
        "/gpfs/scrubbed/suryadv/slime-qwen3-8b-32b-full-repro/v1/4095e6ac256b0dbf/outputs/math500/qwen3_8b_base/summary.json"
    )
    if sha256(base_summary) != contract["evaluation"]["base_summary_sha256"]:
        raise ValueError("reused base MATH-500 summary hash mismatch")
    model_root = Path("/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/models")
    for path in (
        model_root / "Qwen3-8B-Base/config.json",
        model_root / "Qwen3-8B-Base_torch_dist/latest_checkpointed_iteration.txt",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    dataset_cache = Path(
        "/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/hf_home/open-r1___open_r1-math-220k/default/0.0.0/"
        + contract["data"]["revision"]
    )
    shards = sorted(dataset_cache.glob("open_r1-math-220k-train-*-of-00010.arrow"))
    if len(shards) != 10:
        raise FileNotFoundError(f"expected ten cached OpenR1 shards under {dataset_cache}, found {len(shards)}")
    dtype_checks = {
        repo / "slime/ray/rollout.py": '"loss_masks": torch.float32',
        repo / "slime/backends/megatron_utils/actor.py": "dtype=torch.float32",
        repo / "slime/backends/megatron_utils/server/logprob_utils.py": '"loss_masks"], torch.float32',
    }
    for path, needle in dtype_checks.items():
        if needle not in path.read_text(encoding="utf-8"):
            raise ValueError(f"fractional loss-mask transport is not enabled in {path}")
    for script in sorted(example.glob("*.sbatch")) + [example / "submit_chain.sh", example / "container_exec.sh"]:
        subprocess.run(["bash", "-n", str(script)], check=True)
    if args.require_pushed_clean:
        if git("branch", "--show-current") != BRANCH:
            raise ValueError(f"submission must run from {BRANCH}")
        if git("status", "--porcelain"):
            raise ValueError("submission requires a clean worktree")
        upstream = git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
        if git("rev-parse", "HEAD") != git("rev-parse", upstream):
            raise ValueError("local HEAD is not the pushed upstream commit")
    print("CONFIG_VALID full_sft=1 variants=3 train_rows_each=40000 eval=MATH-500", flush=True)


if __name__ == "__main__":
    main()
