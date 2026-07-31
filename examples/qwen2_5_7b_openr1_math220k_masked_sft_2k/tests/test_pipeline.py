"""Focused tests for deterministic masking and protected-token bypass."""

from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.build_variants import wrong_weights
from examples.qwen2_5_7b_openr1_math220k_masked_sft_2k import (
    check_checkpoint_tmpdir,
)
from examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.data_utils import (
    protected_assistant_positions,
    stable_unit_interval,
)
from examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.prepare_from_axolotl import (
    canonical_trace_structure,
)
from examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.weighted_sft_rollout import (
    generate_rollout,
)
from examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.validate_base_model import (
    generation_prompt_ids,
)


class FakeTokenizer:
    def get_added_vocab(self):
        return {"<|control|>": 77}

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"<think>": [7, 8], "</think>": [9, 10]}[text]


class FakeChatTokenizer:
    def __init__(self, rendered):
        self.rendered = rendered

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert messages == [{"role": "user", "content": "x"}]
        assert tokenize is True
        assert add_generation_prompt is True
        return self.rendered


@pytest.mark.parametrize(
    "rendered",
    (
        [151644, 77091, 198],
        {"input_ids": [151644, 77091, 198], "attention_mask": [1, 1, 1]},
    ),
)
def test_generation_prompt_accepts_list_and_batch_encoding_shapes(rendered):
    assert generation_prompt_ids(FakeChatTokenizer(rendered)) == [
        151644,
        77091,
        198,
    ]


def test_stable_draw_is_repeatable():
    assert stable_unit_interval(42, "trace", 3) == stable_unit_interval(42, "trace", 3)
    assert stable_unit_interval(42, "trace", 3) != stable_unit_interval(42, "trace", 4)


def test_all_added_controls_and_think_pieces_are_protected():
    positions, reasons, literal_positions = protected_assistant_positions(
        FakeTokenizer(), [7, 8, 11, 77, 9, 10, 77]
    )
    assert positions == [0, 1, 3, 4, 5, 6]
    assert reasons["3"] == ["added_token:<|control|>"]
    assert reasons["0"] == ["literal_markup:<think>"]
    assert literal_positions == {"<think>": [0, 1], "</think>": [4, 5]}


@pytest.mark.parametrize(
    "variant",
    [
        "random_mask_25",
        "inverse_tau_0p20",
        "inverse_tau_0p05",
        "margin_mask",
        "prob_ratio_mask",
    ],
)
def test_protected_positions_bypass_every_masking_policy(monkeypatch, variant):
    def draw(seed, trace_id, index):
        if index in {0, 2}:
            raise AssertionError("protected position entered random policy")
        return 0.0

    monkeypatch.setattr(
        "examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.build_variants.stable_unit_interval",
        draw,
    )
    values = wrong_weights(variant, [-10.0, -10.0, -10.0], "t", 42, [0, 2])
    assert values[0] == 1.0
    assert values[2] == 1.0


def test_unprotected_margin_semantics():
    margins = [-0.20, 0.0, 0.20]
    assert wrong_weights("margin_mask", margins, "t", 42, []) == [0.0, 0.0, 1.0]
    assert wrong_weights("inverse_tau_0p20", margins, "t", 42, []) == pytest.approx(
        [0.5, 1.0, 1.0]
    )
    assert wrong_weights("inverse_tau_0p05", margins, "t", 42, []) == pytest.approx(
        [0.2, 1.0, 1.0]
    )


def test_probability_ratio_semantics(monkeypatch):
    monkeypatch.setattr(
        "examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.build_variants.stable_unit_interval",
        lambda seed, trace_id, index: 0.5,
    )
    assert wrong_weights(
        "prob_ratio_mask", [-1.0, 0.0, math.log(4.0)], "t", 42, []
    ) == [0.0, 0.0, 1.0]


def test_frozen_trace_anomalies_are_preserved_but_chat_controls_rejected():
    assert canonical_trace_structure("<think>x") == (True, "missing_close_preserved")
    assert canonical_trace_structure("<think>x</think></think>answer") == (
        True,
        "duplicate_close_preserved",
    )
    assert canonical_trace_structure("<think>x</think><|im_end|>answer") == (
        False,
        "embedded_chat_control",
    )


def test_rollout_rejects_a_downweighted_protected_control():
    sample = SimpleNamespace(
        prompt=[1, 2, 3, 151645, 198],
        metadata={
            "assistant_token_count": 3,
            "loss_weights": [1.0, 0.5, 1.0, 1.0, 0.0],
            "protected_assistant_positions": [1],
            "response_length": 5,
        },
    )
    buffer = SimpleNamespace(get_samples=lambda count: [[sample]])
    args = SimpleNamespace(rollout_global_dataset=True, rollout_batch_size=1)
    with pytest.raises(ValueError, match="protected assistant controls"):
        generate_rollout(args, 0, buffer)


def test_model_validation_repair_reuses_the_pending_chain():
    root = Path(__file__).resolve().parents[1]
    text = (root / "resubmit_after_model_validation.sh").read_text()
    assert "readonly FAILED_MODEL_JOB=198094" in text
    assert "readonly BLOCKED_DATA_JOB=198095" in text
    assert "readonly NEXT_SMOKE_JOB=198096" in text
    assert "readonly TAIL_TRAIN_JOB=198108" in text
    assert 'scontrol update \\\n  JobId="${BLOCKED_DATA_JOB}"' in text
    assert 'Dependency="afterok:${replacement_job}"' in text
    assert '"reused_jobs": list(range(198095, 198109))' in text
    assert '"${SCRIPT_DIR}/container_exec.sh" python3 \\' in text
    assert '"${SCRIPT_DIR}/validate_base_model.py" --model "${STUDENT_HF_DIR}"' in text
    scancel_lines = [
        line for line in text.splitlines() if line.lstrip().startswith("scancel")
    ]
    assert scancel_lines == ['      scancel "${replacement_job}" || true']


def test_checkpoint_tmpdir_preflight_round_trip(monkeypatch):
    class FakeQueue:
        def __init__(self):
            self.value = None

        def put(self, value):
            self.value = value

        def get(self, timeout):
            assert timeout == 5
            return self.value

    class FakeManager:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def Queue(self):
            return FakeQueue()

    configured = Path("/tmp") / f"q25-test-{os.getpid()}"
    configured.mkdir(exist_ok=True)
    monkeypatch.setenv("TMPDIR", str(configured))
    monkeypatch.setattr(
        check_checkpoint_tmpdir.tempfile, "gettempdir", lambda: str(configured)
    )
    monkeypatch.setattr(check_checkpoint_tmpdir.multiprocessing, "Manager", FakeManager)
    try:
        check_checkpoint_tmpdir.main()
    finally:
        configured.rmdir()


def test_model_prep_uses_short_checkpoint_tmpdir_and_preflight():
    root = Path(__file__).resolve().parents[1]
    text = (root / "00_prepare_model.sbatch").read_text()
    assert 'export TMPDIR="/tmp/${USER:-suryadv}/q25-model-${SLURM_JOB_ID}"' in text
    assert 'export RAY_TMPDIR="${TMPDIR}/ray"' in text
    assert '"${SCRIPT_DIR}/check_checkpoint_tmpdir.py"' in text


def test_conversion_tmpdir_repair_preserves_and_reuses_the_pending_chain():
    root = Path(__file__).resolve().parents[1]
    text = (root / "resubmit_after_conversion_tmpdir.sh").read_text()
    assert "readonly FAILED_MODEL_JOB=198594" in text
    assert "readonly BLOCKED_DATA_JOB=198095" in text
    assert "readonly NEXT_SMOKE_JOB=198096" in text
    assert "readonly TAIL_TRAIN_JOB=198108" in text
    assert "OSError: AF_UNIX path too long" in text
    assert 'mv -- "${PARTIAL_DIR}" "${FAILED_ARCHIVE}"' in text
    assert "--job-name=q25-7b-model-r2" in text
    assert 'Dependency="afterok:${replacement_job}"' in text
    assert '"reused_jobs": list(range(198095, 198109))' in text
    assert text.index("trap fail_closed EXIT") < text.index(
        'mv -- "${PARTIAL_DIR}" "${FAILED_ARCHIVE}"'
    )
    assert text.index('mv -- "${PARTIAL_DIR}" "${FAILED_ARCHIVE}"') < text.index(
        'raw="$(sbatch'
    )
    assert text.index('raw="$(sbatch') < text.index("scontrol update")
    scancel_lines = [
        line for line in text.splitlines() if line.lstrip().startswith("scancel")
    ]
    assert scancel_lines == ['      scancel "${replacement_job}" || true']


@pytest.mark.parametrize(
    ("initial", "expected"),
    (("{}", "{}"), ("", '{"enable_thinking": true}')),
)
def test_shared_sft_chat_template_json_resolution(initial, expected):
    command = r'''
set -euo pipefail
SFT_APPLY_CHAT_TEMPLATE_KWARGS_JSON="$1"
if [[ -z "${SFT_APPLY_CHAT_TEMPLATE_KWARGS_JSON:-}" ]]; then
  SFT_APPLY_CHAT_TEMPLATE_KWARGS_JSON='{"enable_thinking": true}'
fi
python3 - "${SFT_APPLY_CHAT_TEMPLATE_KWARGS_JSON}" <<'PY'
import json
import sys
value = json.loads(sys.argv[1])
if not isinstance(value, dict):
    raise SystemExit(2)
print(sys.argv[1])
PY
'''
    result = subprocess.run(
        ["bash", "-c", command, "bash", initial],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == expected


def test_shared_sft_rejects_malformed_chat_template_json():
    runner = (
        Path(__file__).resolve().parents[2]
        / "qwen3_8b_opd_tillicum/04_run_sft_100k_8xh200.sbatch"
    ).read_text()
    assert 'if [[ -z "${SFT_APPLY_CHAT_TEMPLATE_KWARGS_JSON:-}" ]]' in runner
    assert "json.loads(sys.argv[1])" in runner
    assert '${SFT_APPLY_CHAT_TEMPLATE_KWARGS_JSON:-{\\"enable_thinking\\": true}}' not in runner
    result = subprocess.run(
        [
            "python3",
            "-c",
            "import json, sys; value = json.loads(sys.argv[1]); "
            "assert isinstance(value, dict)",
            "{}}",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_smoke_json_repair_preserves_the_pending_chain():
    root = Path(__file__).resolve().parents[1]
    text = (root / "resubmit_after_smoke_json.sh").read_text()
    assert "readonly COMPLETED_MODEL_JOB=199182" in text
    assert "readonly COMPLETED_DATA_JOB=198095" in text
    assert "readonly FAILED_SMOKE_JOB=198096" in text
    assert "readonly BLOCKED_SCORE_JOB=198097" in text
    assert "readonly FIRST_TRAIN_JOB=198098" in text
    assert "readonly TAIL_TRAIN_JOB=198108" in text
    assert 'mv -- "${FAILED_OUTPUT}" "${FAILED_ARCHIVE}"' in text
    assert "--job-name=q25-7b-smoke-r1" in text
    assert 'Dependency="afterok:${replacement_job}"' in text
    assert '"reused_jobs": list(range(198097, 198109))' in text
    assert text.index("trap fail_closed EXIT") < text.index(
        'mv -- "${FAILED_OUTPUT}" "${FAILED_ARCHIVE}"'
    )
    assert text.index('raw="$(sbatch') < text.index("scontrol update")
    scancel_lines = [
        line for line in text.splitlines() if line.lstrip().startswith("scancel")
    ]
    assert scancel_lines == ['      scancel "${replacement_job}" || true']
