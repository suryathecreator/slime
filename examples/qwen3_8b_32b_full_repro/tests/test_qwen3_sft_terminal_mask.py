from __future__ import annotations

import os

import pytest

from examples.qwen3_8b_opd_tillicum.check_sft_loss_mask import find_subsequence
from slime.utils.mask_utils import MultiTurnLossMaskGenerator


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    model_dir = os.environ.get("STUDENT_HF_DIR", "/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/models/Qwen3-8B-Base")
    return transformers.AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)


def test_qwen3_assistant_only_mask_supervises_think_final_and_terminal(tokenizer) -> None:
    messages = [
        {"role": "user", "content": "What is one plus one?"},
        {"role": "assistant", "content": "<think>One plus one is two.</think>\nThe final answer is \\boxed{2}."},
    ]
    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3", apply_chat_template_kwargs={"enable_thinking": True})
    token_ids, loss_mask = generator.get_loss_mask(messages)
    assert len(token_ids) == len(loss_mask)
    for target in ("<think>", "</think>", "<|im_end|>"):
        needle = tokenizer(target, add_special_tokens=False)["input_ids"]
        starts = find_subsequence(token_ids, needle)
        assert starts, target
        assert any(all(loss_mask[pos] == 1 for pos in range(start, start + len(needle))) for start in starts), target
    supervised = tokenizer.decode(
        [token_id for token_id, weight in zip(token_ids, loss_mask, strict=True) if weight],
        skip_special_tokens=False,
    )
    assert "The final answer is \\boxed{2}." in supervised


def test_user_tokens_are_masked_and_terminal_is_not_double_appended(tokenizer) -> None:
    messages = [
        {"role": "user", "content": "UNIQUE_USER_TEXT_937"},
        {"role": "assistant", "content": "<think>x</think>answer"},
    ]
    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3", apply_chat_template_kwargs={"enable_thinking": True})
    token_ids, loss_mask = generator.get_loss_mask(messages)
    user_ids = tokenizer("UNIQUE_USER_TEXT_937", add_special_tokens=False)["input_ids"]
    for start in find_subsequence(token_ids, user_ids):
        assert all(loss_mask[pos] == 0 for pos in range(start, start + len(user_ids)))
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert token_ids.count(im_end) == 2
    assert loss_mask[token_ids.index(im_end)] == 0
    assert loss_mask[len(token_ids) - 1 - token_ids[::-1].index(im_end)] == 1


def test_stop_token_contract_is_eos_plus_im_end(tokenizer) -> None:
    assert tokenizer.convert_tokens_to_ids("<|endoftext|>") == 151643
    assert tokenizer.convert_tokens_to_ids("<|im_end|>") == 151645
    assert tokenizer.eos_token_id in {151643, 151645}
    assert tokenizer.convert_tokens_to_ids("</think>") not in {151643, 151645}
