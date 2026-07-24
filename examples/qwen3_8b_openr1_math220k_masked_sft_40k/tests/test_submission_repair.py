"""Regression checks for checkpoint TMPDIR and the one-time chain repair."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT_40K = Path(__file__).resolve().parents[1]
ROOT_2K = ROOT_40K.parent / "qwen3_8b_openr1_math220k_masked_sft_2k"


def test_training_jobs_use_short_node_local_tmpdirs() -> None:
    expected = {
        ROOT_40K / "03_train.sbatch": "q8b-masksft-train-${SLURM_JOB_ID}",
        ROOT_2K / "03_train.sbatch": "q8b-sft2k-train-${SLURM_JOB_ID}",
    }
    for path, stem in expected.items():
        text = path.read_text()
        assert f'export TMPDIR="/tmp/${{USER:-suryadv}}/{stem}"' in text
        assert 'export RAY_TMPDIR="${TMPDIR}/ray"' in text
        assert 'mkdir -p "${TMPDIR}" "${RAY_TMPDIR}"' in text
        assert "check_train_tmpdir.py" in text
        assert text.index("check_train_tmpdir.py") < text.index(
            "04_run_sft_100k_8xh200.sbatch"
        )


def test_multiprocessing_manager_preflight_works_in_short_tmpdir() -> None:
    with tempfile.TemporaryDirectory(prefix="q8b-tmp-", dir="/tmp") as temp:
        environment = os.environ.copy()
        environment["TMPDIR"] = temp
        result = subprocess.run(
            [sys.executable, str(ROOT_40K / "check_train_tmpdir.py")],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
    assert "TRAIN_TMPDIR_PREFLIGHT_OK" in result.stdout


def test_recovery_reuses_blocked_eval_and_preserves_downstream_chain() -> None:
    text = (ROOT_40K / "resubmit_after_tmpdir.sh").read_text()
    assert "readonly FAILED_TRAIN_JOB=185501" in text
    assert "readonly BLOCKED_EVAL_JOB=185502" in text
    assert "readonly DOWNSTREAM_JOB=185503" in text
    assert 'mv -- "${FAILED_OUTPUT}" "${FAILED_ARCHIVE}"' in text
    assert 'scontrol update \\\n  JobId="${BLOCKED_EVAL_JOB}"' in text
    assert 'Dependency="afterok:${replacement_job}"' in text
    assert "scancel" in text  # only the fail-closed replacement cleanup
    assert "185503" not in "\n".join(
        line for line in text.splitlines() if line.lstrip().startswith("scancel")
    )
    assert text.index('raw="$(sbatch') < text.index("scontrol update")
