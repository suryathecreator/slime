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


def _native_functions() -> tuple[Any, Any, Any]:
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

    def native_get_num_sampled_and_rejected(
        num_sampled: torch.Tensor,
        seq_lens: torch.Tensor,
        cu_num_logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        prefill_len: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_reqs = idx_mapping.shape[0]
        req_indices = idx_mapping.to(torch.long)
        valid_reqs = req_indices >= 0
        safe_req_indices = torch.clamp(req_indices, min=0)
        req_prefill_len = prefill_len[safe_req_indices]
        is_chunked_prefill = (~valid_reqs) | (seq_lens[:num_reqs] < req_prefill_len)

        sampled = num_sampled.clone()
        sampled = torch.where(is_chunked_prefill, torch.zeros_like(sampled), sampled)

        logits_start = cu_num_logits[:num_reqs]
        logits_end = cu_num_logits[1 : num_reqs + 1]
        num_logits = (logits_end - logits_start).to(sampled.dtype)
        rejected = num_logits - sampled
        rejected = torch.where(is_chunked_prefill, torch.zeros_like(rejected), rejected)
        return sampled, rejected

    return native_apply_temperature, native_gumbel_sample, native_get_num_sampled_and_rejected


def _copy_staged_values(gpu: Any, row_idx: int, start_idx: int, values: list[Any], dtype: Any) -> None:
    import torch

    if not values:
        return
    value_tensor = torch.as_tensor(values, dtype=dtype, device=gpu.device)
    if gpu.dim() == 1:
        gpu.narrow(0, row_idx + start_idx, len(values)).copy_(value_tensor)
    else:
        gpu[row_idx, start_idx : start_idx + len(values)].copy_(value_tensor)


def _patch_buffer_utils(buffer_utils_mod: ModuleType) -> None:
    if not hasattr(buffer_utils_mod, "StagedWriteTensor") or not hasattr(
        buffer_utils_mod, "FusedStagedWriter"
    ):
        return

    def native_staged_apply_write(self: Any) -> None:
        n = len(self._staged_write_indices)
        if n == 0:
            return

        prev_cu_len = 0
        for index, start, cu_len in zip(
            self._staged_write_indices,
            self._staged_write_starts,
            self._staged_write_cu_lens,
            strict=True,
        ):
            values = self._staged_write_contents[prev_cu_len:cu_len]
            _copy_staged_values(self.gpu, int(index), int(start), values, self.dtype)
            prev_cu_len = cu_len
        self.clear_staged_writes()

    def native_fused_apply(self: Any, tensors: Any, output_ptrs: Any, output_strides: Any) -> None:
        del output_ptrs, output_strides
        for tensor in tensors:
            native_staged_apply_write(tensor)

    buffer_utils_mod.StagedWriteTensor.apply_write = native_staged_apply_write
    buffer_utils_mod.FusedStagedWriter.apply = native_fused_apply


def _patch_block_table(block_table_mod: ModuleType) -> None:
    import torch

    if not hasattr(block_table_mod, "BlockTables"):
        return

    def native_gather_block_tables(
        self: Any,
        idx_mapping: torch.Tensor,
        num_reqs_padded: int,
    ) -> tuple[torch.Tensor, ...]:
        num_reqs = idx_mapping.shape[0]
        for group_id, (src_block_table, dst_block_table) in enumerate(
            zip(self.block_tables, self.input_block_tables, strict=True)
        ):
            dst_block_table[:num_reqs_padded].zero_()
            for batch_idx in range(num_reqs):
                req_idx = int(idx_mapping[batch_idx].item())
                num_blocks = int(self.num_blocks.gpu[group_id, req_idx].item())
                if num_blocks > 0:
                    dst_block_table[batch_idx, :num_blocks].copy_(
                        src_block_table.gpu[req_idx, :num_blocks]
                    )
        return tuple(bt[:num_reqs_padded] for bt in self.input_block_tables)

    def native_compute_slot_mappings(
        self: Any,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        num_tokens_padded: int,
    ) -> torch.Tensor:
        self.slot_mappings[:, :num_tokens_padded].fill_(block_table_mod.PAD_SLOT_ID)
        num_reqs = idx_mapping.shape[0]
        for group_id, block_table in enumerate(self.block_tables):
            block_size = int(self.kernel_block_sizes[group_id])
            for batch_idx in range(num_reqs):
                req_idx = int(idx_mapping[batch_idx].item())
                start = int(query_start_loc[batch_idx].item())
                end = int(query_start_loc[batch_idx + 1].item())
                if end <= start:
                    continue

                pos = positions[start:end]
                block_offsets = pos % (block_size * self.cp_size)
                block_indices = (pos // (block_size * self.cp_size)).to(torch.long)
                block_numbers = block_table.gpu[req_idx, block_indices]
                if self.cp_size == 1:
                    slot_ids = block_numbers * block_size + block_offsets
                else:
                    is_local = (block_offsets // self.cp_interleave) % self.cp_size == self.cp_rank
                    rounds = block_offsets // (self.cp_interleave * self.cp_size)
                    remainder = block_offsets % self.cp_interleave
                    local_offsets = rounds * self.cp_interleave + remainder
                    slot_ids = block_numbers * block_size + local_offsets
                    slot_ids = torch.where(
                        is_local,
                        slot_ids,
                        torch.full_like(slot_ids, block_table_mod.PAD_SLOT_ID),
                    )
                self.slot_mappings[group_id, start:end].copy_(slot_ids.to(torch.int64))
        return self.slot_mappings[:, :num_tokens_padded]

    block_table_mod.BlockTables.gather_block_tables = native_gather_block_tables
    block_table_mod.BlockTables.compute_slot_mappings = native_compute_slot_mappings


def _patch_input_batch(input_batch_mod: ModuleType) -> None:
    import torch

    if not hasattr(input_batch_mod, "prepare_prefill_inputs"):
        return

    def native_prepare_prefill_inputs(
        input_ids: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        all_token_ids: torch.Tensor,
        prefill_len: torch.Tensor,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        for batch_idx in range(idx_mapping.shape[0]):
            req_idx = int(idx_mapping[batch_idx].item())
            prefill = int(prefill_len[req_idx].item())
            computed = int(num_computed_tokens[req_idx].item())
            if computed >= prefill:
                continue
            start = int(query_start_loc[batch_idx].item())
            end = int(query_start_loc[batch_idx + 1].item())
            query_len = end - start
            input_ids[start:end].copy_(all_token_ids[req_idx, computed : computed + query_len])
            next_pos = computed + query_len
            if next_pos < prefill:
                next_prefill_tokens[req_idx] = all_token_ids[req_idx, next_pos]

    def native_prepare_pos_seq_lens(
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        pos: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        num_reqs = idx_mapping.shape[0]
        if seq_lens.shape[0] > num_reqs:
            seq_lens[num_reqs:].zero_()
        for batch_idx in range(num_reqs):
            req_idx = int(idx_mapping[batch_idx].item())
            computed = int(num_computed_tokens[req_idx].item())
            start = int(query_start_loc[batch_idx].item())
            end = int(query_start_loc[batch_idx + 1].item())
            query_len = end - start
            seq_lens[batch_idx] = computed + query_len
            if query_len > 0:
                pos[start:end].copy_(
                    torch.arange(computed, computed + query_len, device=pos.device, dtype=pos.dtype)
                )

    def native_combine_sampled_and_draft_tokens(
        input_ids: torch.Tensor,
        idx_mapping: torch.Tensor,
        last_sampled_tokens: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        prefill_len: torch.Tensor,
        draft_tokens: torch.Tensor,
        cu_num_logits: torch.Tensor,
        num_logits: int,
        num_new_sampled_tokens: int = 1,
    ) -> torch.Tensor:
        del num_logits
        logits_indices = torch.empty(
            int(cu_num_logits[-1].item()), dtype=torch.int64, device=input_ids.device
        )
        for batch_idx in range(idx_mapping.shape[0]):
            req_idx = int(idx_mapping[batch_idx].item())
            logits_start = int(cu_num_logits[batch_idx].item())
            logits_end = int(cu_num_logits[batch_idx + 1].item())
            req_num_logits = logits_end - logits_start
            query_end = int(query_start_loc[batch_idx + 1].item())
            if req_num_logits > 0:
                logits_indices[logits_start:logits_end].copy_(
                    torch.arange(
                        query_end - req_num_logits,
                        query_end,
                        dtype=torch.int64,
                        device=input_ids.device,
                    )
                )

            if int(seq_lens[batch_idx].item()) <= int(prefill_len[req_idx].item()):
                continue
            if num_new_sampled_tokens > 0:
                input_ids[query_end - req_num_logits] = last_sampled_tokens[req_idx, 0]
            num_draft_tokens = req_num_logits - num_new_sampled_tokens
            if num_draft_tokens > 0:
                input_ids[query_end - num_draft_tokens : query_end].copy_(
                    draft_tokens[req_idx, :num_draft_tokens]
                )
        return logits_indices

    def native_post_update(
        idx_mapping: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        last_sampled_tokens: torch.Tensor,
        output_bin_counts: torch.Tensor | None,
        sampled_tokens: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        query_start_loc: torch.Tensor | None,
        all_token_ids: torch.Tensor,
        total_len: torch.Tensor,
    ) -> None:
        for batch_idx in range(idx_mapping.shape[0]):
            req_idx = int(idx_mapping[batch_idx].item())
            if req_idx < 0:
                continue
            total = int(total_len[req_idx].item())
            sampled_count = int(num_sampled[batch_idx].item())
            if sampled_count > 0:
                last_sampled_tokens[req_idx, 0] = sampled_tokens[batch_idx, sampled_count - 1]
                total_len[req_idx] = total + sampled_count
            for i in range(sampled_count):
                token_id = sampled_tokens[batch_idx, i]
                all_token_ids[req_idx, total + i] = token_id
                if output_bin_counts is not None:
                    output_bin_counts[req_idx, int(token_id.item())] += 1
            query_len = 0
            if query_start_loc is not None:
                query_len = int(query_start_loc[batch_idx + 1].item()) - int(
                    query_start_loc[batch_idx].item()
                )
            computed_delta = query_len - int(num_rejected[batch_idx].item())
            if computed_delta != 0:
                num_computed_tokens[req_idx] += computed_delta

    def native_post_update_num_computed_tokens(
        idx_mapping: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> None:
        for batch_idx in range(idx_mapping.shape[0]):
            req_idx = int(idx_mapping[batch_idx].item())
            query_len = int(query_start_loc[batch_idx + 1].item()) - int(
                query_start_loc[batch_idx].item()
            )
            num_computed_tokens[req_idx] += query_len

    def native_expand_idx_mapping(
        idx_mapping: torch.Tensor,
        total_num_logits: int,
        cu_num_logits: torch.Tensor,
        max_expand_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del max_expand_len
        expanded_idx_mapping = idx_mapping.new_empty(total_num_logits)
        expanded_local_pos = torch.empty(
            total_num_logits, dtype=torch.int32, device=idx_mapping.device
        )
        for req_idx in range(idx_mapping.shape[0]):
            start = int(cu_num_logits[req_idx].item())
            end = int(cu_num_logits[req_idx + 1].item())
            if end > start:
                expanded_idx_mapping[start:end] = idx_mapping[req_idx]
                expanded_local_pos[start:end].copy_(
                    torch.arange(end - start, dtype=torch.int32, device=idx_mapping.device)
                )
        return expanded_idx_mapping, expanded_local_pos

    input_batch_mod.prepare_prefill_inputs = native_prepare_prefill_inputs
    input_batch_mod.prepare_pos_seq_lens = native_prepare_pos_seq_lens
    input_batch_mod.combine_sampled_and_draft_tokens = native_combine_sampled_and_draft_tokens
    input_batch_mod.post_update = native_post_update
    input_batch_mod.post_update_num_computed_tokens = native_post_update_num_computed_tokens
    input_batch_mod.expand_idx_mapping = native_expand_idx_mapping


def _patch_penalties(penalties_mod: ModuleType) -> None:
    import torch

    if not hasattr(penalties_mod, "bincount"):
        return

    def native_bincount(
        expanded_idx_mapping: torch.Tensor,
        all_token_ids: torch.Tensor,
        prompt_len: torch.Tensor,
        prefill_len: torch.Tensor,
        prompt_bin_mask: torch.Tensor,
        output_bin_counts: torch.Tensor,
        max_prefill_len: int,
    ) -> None:
        del max_prefill_len
        idx_long = expanded_idx_mapping.long()
        prompt_bin_mask.index_fill_(0, idx_long, 0)
        output_bin_counts.index_fill_(0, idx_long, 0)

        for req_idx_tensor in expanded_idx_mapping:
            req_idx = int(req_idx_tensor.item())
            prompt = int(prompt_len[req_idx].item())
            prefill = int(prefill_len[req_idx].item())

            for token in all_token_ids[req_idx, :prompt].to(torch.long).tolist():
                packed_idx = int(token) // 32
                bit_idx = int(token) % 32
                prompt_bin_mask[req_idx, packed_idx] |= 1 << bit_idx

            if prefill > prompt:
                output_tokens = all_token_ids[req_idx, prompt:prefill].to(torch.long)
                output_bin_counts[req_idx].scatter_add_(
                    0,
                    output_tokens,
                    torch.ones_like(output_tokens, dtype=output_bin_counts.dtype),
                )

    def native_apply_penalties(
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        token_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        repetition_penalty: torch.Tensor,
        frequency_penalty: torch.Tensor,
        presence_penalty: torch.Tensor,
        prompt_bin_mask: torch.Tensor,
        output_bin_counts: torch.Tensor,
    ) -> None:
        vocab_size = logits.shape[1]
        bit_offsets = torch.arange(32, dtype=torch.int32, device=logits.device)
        for token_idx in range(logits.shape[0]):
            req_idx = int(expanded_idx_mapping[token_idx].item())
            rep = float(repetition_penalty[req_idx].item())
            freq = float(frequency_penalty[req_idx].item())
            pres = float(presence_penalty[req_idx].item())
            if rep == 1.0 and freq == 0.0 and pres == 0.0:
                continue

            counts = output_bin_counts[req_idx].to(torch.float32)
            pos = int(expanded_local_pos[token_idx].item())
            if pos > 0:
                start = token_idx - pos
                prev_tokens = token_ids[start + 1 : start + pos + 1].to(torch.long)
                counts = counts.clone()
                counts.scatter_add_(0, prev_tokens, torch.ones_like(prev_tokens, dtype=counts.dtype))
            output_mask = counts > 0

            if rep != 1.0:
                packed = prompt_bin_mask[req_idx]
                prompt_mask = ((packed[:, None] >> bit_offsets[None, :]) & 1).reshape(-1)
                prompt_mask = prompt_mask[:vocab_size].to(torch.bool)
                seen_mask = prompt_mask | output_mask
                row = logits[token_idx]
                positive = row > 0
                row[seen_mask & positive] /= rep
                row[seen_mask & ~positive] *= rep
            if freq != 0.0:
                logits[token_idx] -= freq * counts.to(logits.dtype)
            if pres != 0.0:
                logits[token_idx] -= pres * output_mask.to(logits.dtype)

    penalties_mod.bincount = native_bincount
    penalties_mod.apply_penalties = native_apply_penalties


def _patch_structured_outputs(structured_outputs_mod: ModuleType) -> None:
    if not hasattr(structured_outputs_mod, "StructuredOutputsWorker"):
        return

    def native_apply_grammar_bitmask(
        self: Any,
        logits: Any,
        input_batch: Any,
        grammar_req_ids: list[str],
        grammar_bitmask: Any,
    ) -> None:
        del self, logits, input_batch, grammar_req_ids, grammar_bitmask
        return

    structured_outputs_mod.StructuredOutputsWorker.apply_grammar_bitmask = (
        native_apply_grammar_bitmask
    )


def _patch_logit_bias(logit_bias_mod: ModuleType) -> None:
    import torch

    if not hasattr(logit_bias_mod, "apply_logit_bias"):
        return

    def native_apply_logit_bias(
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        pos: torch.Tensor,
        num_allowed_token_ids: torch.Tensor,
        allowed_token_ids: torch.Tensor,
        num_logit_bias: torch.Tensor,
        logit_bias_token_ids: torch.Tensor,
        logit_bias: torch.Tensor,
        min_lens: torch.Tensor,
        num_stop_token_ids: torch.Tensor,
        stop_token_ids: torch.Tensor,
    ) -> None:
        for token_idx in range(logits.shape[0]):
            req_state_idx = int(expanded_idx_mapping[token_idx].item())

            allowed_count = int(num_allowed_token_ids[req_state_idx].item())
            if allowed_count > 0:
                ids = allowed_token_ids[req_state_idx, :allowed_count].to(torch.long)
                saved_logits = logits[token_idx, ids].clone()
                logits[token_idx].fill_(-float("inf"))
                logits[token_idx, ids] = saved_logits

            bias_count = int(num_logit_bias[req_state_idx].item())
            if bias_count > 0:
                ids = logit_bias_token_ids[req_state_idx, :bias_count].to(torch.long)
                bias = logit_bias[req_state_idx, :bias_count].to(logits.dtype)
                logits[token_idx, ids] += bias

            stop_count = int(num_stop_token_ids[req_state_idx].item())
            token_pos = int(pos[token_idx].item())
            min_len = int(min_lens[req_state_idx].item())
            if stop_count > 0 and token_pos + 1 < min_len:
                ids = stop_token_ids[req_state_idx, :stop_count].to(torch.long)
                logits[token_idx, ids] = -float("inf")

    logit_bias_mod.apply_logit_bias = native_apply_logit_bias


def _patch_bad_words(bad_words_mod: ModuleType) -> None:
    import torch

    if not hasattr(bad_words_mod, "apply_bad_words"):
        return

    def native_apply_bad_words(
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        bad_word_token_ids: torch.Tensor,
        bad_word_offsets: torch.Tensor,
        num_bad_words: torch.Tensor,
        all_token_ids: torch.Tensor,
        prompt_len: torch.Tensor,
        total_len: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        max_num_bad_words: int,
    ) -> None:
        del max_num_bad_words
        for token_idx in range(logits.shape[0]):
            req_state_idx = int(expanded_idx_mapping[token_idx].item())
            num_req_bad_words = int(num_bad_words[req_state_idx].item())
            if num_req_bad_words == 0:
                continue

            pos = int(expanded_local_pos[token_idx].item())
            cur_req_first_pos = token_idx - pos
            req_prompt_len = int(prompt_len[req_state_idx].item())
            req_total_len = int(total_len[req_state_idx].item())
            output_len = req_total_len - req_prompt_len
            effective_len = output_len + pos

            for bad_word_idx in range(num_req_bad_words):
                start = int(bad_word_offsets[req_state_idx, bad_word_idx].item())
                end = int(bad_word_offsets[req_state_idx, bad_word_idx + 1].item())
                bad_word_len = end - start
                if bad_word_len <= 0:
                    continue

                prefix_len = bad_word_len - 1
                if prefix_len > effective_len:
                    continue

                matched = True
                for prefix_idx in range(prefix_len):
                    expected = int(
                        bad_word_token_ids[req_state_idx, start + prefix_idx].item()
                    )
                    actual_pos = effective_len - prefix_len + prefix_idx
                    if actual_pos >= output_len:
                        spec_offset = actual_pos - output_len
                        actual = int(input_ids[cur_req_first_pos + spec_offset].item())
                    else:
                        actual = int(
                            all_token_ids[
                                req_state_idx,
                                req_prompt_len + actual_pos,
                            ].item()
                        )
                    if expected != actual:
                        matched = False
                        break

                if matched:
                    last_token = int(bad_word_token_ids[req_state_idx, end - 1].item())
                    logits[token_idx, last_token] = -float("inf")

    bad_words_mod.apply_bad_words = native_apply_bad_words


def _patch_min_p(min_p_mod: ModuleType, states_mod: ModuleType | None = None) -> None:
    import torch

    if not hasattr(min_p_mod, "apply_min_p"):
        return

    def native_apply_min_p(
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        min_p: torch.Tensor,
    ) -> None:
        for token_idx in range(logits.shape[0]):
            req_state_idx = int(expanded_idx_mapping[token_idx].item())
            req_min_p = float(min_p[req_state_idx].item())
            if req_min_p == 0.0:
                continue
            threshold = torch.max(logits[token_idx]).to(torch.float32) + torch.log(
                torch.tensor(req_min_p, dtype=torch.float32, device=logits.device)
            )
            logits[token_idx] = torch.where(
                logits[token_idx] < threshold.to(logits.dtype),
                torch.full_like(logits[token_idx], -float("inf")),
                logits[token_idx],
            )

    min_p_mod.apply_min_p = native_apply_min_p
    if isinstance(states_mod, ModuleType):
        states_mod.apply_min_p = native_apply_min_p


def _patch_loaded_modules() -> bool:
    global _PATCH_INSTALLED

    gumbel_mod = sys.modules.get("vllm.v1.worker.gpu.sample.gumbel")
    sampler_mod = sys.modules.get("vllm.v1.worker.gpu.sample.sampler")
    states_mod = sys.modules.get("vllm.v1.worker.gpu.sample.states")
    input_batch_mod = sys.modules.get("vllm.v1.worker.gpu.input_batch")
    buffer_utils_mod = sys.modules.get("vllm.v1.worker.gpu.buffer_utils")
    block_table_mod = sys.modules.get("vllm.v1.worker.gpu.block_table")
    penalties_mod = sys.modules.get("vllm.v1.worker.gpu.sample.penalties")
    structured_outputs_mod = sys.modules.get("vllm.v1.worker.gpu.structured_outputs")
    logit_bias_mod = sys.modules.get("vllm.v1.worker.gpu.sample.logit_bias")
    bad_words_mod = sys.modules.get("vllm.v1.worker.gpu.sample.bad_words")
    min_p_mod = sys.modules.get("vllm.v1.worker.gpu.sample.min_p")
    if (
        gumbel_mod is None
        and sampler_mod is None
        and states_mod is None
        and input_batch_mod is None
        and buffer_utils_mod is None
        and block_table_mod is None
        and penalties_mod is None
        and structured_outputs_mod is None
        and logit_bias_mod is None
        and bad_words_mod is None
        and min_p_mod is None
    ):
        return False

    native_apply_temperature, native_gumbel_sample, native_get_num_sampled_and_rejected = (
        _native_functions()
    )
    if isinstance(gumbel_mod, ModuleType):
        gumbel_mod.apply_temperature = native_apply_temperature
        gumbel_mod.gumbel_sample = native_gumbel_sample
    if isinstance(states_mod, ModuleType):
        states_mod.apply_temperature = native_apply_temperature
    if isinstance(sampler_mod, ModuleType):
        sampler_mod.gumbel_sample = native_gumbel_sample
        sampler_mod.get_num_sampled_and_rejected = native_get_num_sampled_and_rejected
    if isinstance(input_batch_mod, ModuleType):
        input_batch_mod.get_num_sampled_and_rejected = native_get_num_sampled_and_rejected
        _patch_input_batch(input_batch_mod)
    if isinstance(buffer_utils_mod, ModuleType):
        _patch_buffer_utils(buffer_utils_mod)
    if isinstance(block_table_mod, ModuleType):
        _patch_block_table(block_table_mod)
    if isinstance(penalties_mod, ModuleType):
        _patch_penalties(penalties_mod)
    if isinstance(structured_outputs_mod, ModuleType):
        _patch_structured_outputs(structured_outputs_mod)
    if isinstance(logit_bias_mod, ModuleType):
        _patch_logit_bias(logit_bias_mod)
    if isinstance(bad_words_mod, ModuleType):
        _patch_bad_words(bad_words_mod)
    if isinstance(min_p_mod, ModuleType):
        _patch_min_p(min_p_mod, states_mod if isinstance(states_mod, ModuleType) else None)

    _PATCH_INSTALLED = True
    return True


def maybe_force_native_sampler() -> bool:
    if not _enabled("VLLM_EVAL_FORCE_NATIVE_SAMPLER"):
        return False

    from vllm.v1.worker.gpu.sample import gumbel as gumbel_mod
    from vllm.v1.worker.gpu.sample import sampler as sampler_mod
    from vllm.v1.worker.gpu.sample import states as states_mod
    from vllm.v1.worker.gpu import input_batch as input_batch_mod
    from vllm.v1.worker.gpu import buffer_utils as buffer_utils_mod
    from vllm.v1.worker.gpu import block_table as block_table_mod
    from vllm.v1.worker.gpu import structured_outputs as structured_outputs_mod
    from vllm.v1.worker.gpu.sample import bad_words as bad_words_mod
    from vllm.v1.worker.gpu.sample import logit_bias as logit_bias_mod
    from vllm.v1.worker.gpu.sample import min_p as min_p_mod
    from vllm.v1.worker.gpu.sample import penalties as penalties_mod

    del (
        gumbel_mod,
        sampler_mod,
        states_mod,
        input_batch_mod,
        buffer_utils_mod,
        block_table_mod,
        penalties_mod,
        structured_outputs_mod,
        logit_bias_mod,
        bad_words_mod,
        min_p_mod,
    )
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
        if name.startswith("vllm.v1.worker.gpu"):
            _patch_loaded_modules()
        return module

    builtins.__import__ = slime_vllm_import_hook
    _IMPORT_HOOK_INSTALLED = True
    _patch_loaded_modules()
    return True
