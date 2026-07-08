from contextlib import contextmanager
import logging

try:
    from megatron.core.utils import unwrap_model
except ImportError:
    unwrap_model = None

logger = logging.getLogger(__name__)


def patch_hf_config_for_megatron_bridge(hf_config):
    configs = []
    seen_config_ids = set()

    def add_config(config):
        if config is None or id(config) in seen_config_ids:
            return
        seen_config_ids.add(id(config))
        configs.append(config)

    add_config(hf_config)
    add_config(getattr(hf_config, "config", None))

    for config in list(configs):
        add_config(getattr(config, "text_config", None))

    for config in configs:
        rope_params = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None)
        if isinstance(rope_params, dict) and "rope_theta" in rope_params and not hasattr(config, "rope_theta"):
            config.rope_theta = rope_params["rope_theta"]

    return hf_config


def patch_auto_bridge_hf_config(bridge):
    patch_bridge_tp_scatter_for_fsdp()

    hf_pretrained = getattr(bridge, "hf_pretrained", None)
    if hf_pretrained is not None:
        patch_hf_config_for_megatron_bridge(hf_pretrained)

    return bridge


def patch_bridge_tp_scatter_for_fsdp():
    try:
        import torch
        from megatron.bridge.models.conversion import param_mapping
    except ImportError:
        return

    mapping_cls = param_mapping.MegatronParamMapping
    if getattr(mapping_cls, "_slime_tp_scatter_patch_applied", False):
        return

    original_scatter_to_tp_ranks = mapping_cls.scatter_to_tp_ranks

    def _split_full_tensor_if_needed(full_tensor, expected_shape, tp_size):
        expected_shape = torch.Size(expected_shape)
        full_shape = torch.Size(full_tensor.shape)
        if full_shape == expected_shape:
            return None
        if len(full_shape) != len(expected_shape):
            return None

        for dim, (full_dim, expected_dim) in enumerate(zip(full_shape, expected_shape)):
            if expected_dim * tp_size != full_dim:
                continue
            other_dims_match = all(
                full_shape[i] == expected_shape[i]
                for i in range(len(expected_shape))
                if i != dim
            )
            if other_dims_match:
                return list(torch.chunk(full_tensor, tp_size, dim=dim))
        return None

    def scatter_to_tp_ranks(self, splits, output_shape, dtype, device, src_rank=0):
        if self.tp_size == 1:
            return splits[0].to(device=device, dtype=dtype) if splits else None

        output = torch.empty(output_shape, dtype=dtype, device=device)
        global_src = torch.distributed.get_global_rank(group=self.tp_group, group_rank=src_rank)

        scatter_list = None
        if self.tp_rank == src_rank and splits:
            scatter_list = [s.to(device=device, dtype=dtype).contiguous() for s in splits]
            expected_shape = torch.Size(output_shape)
            if any(torch.Size(t.shape) != expected_shape for t in scatter_list):
                candidates = scatter_list[:1]
                if len(scatter_list) == 1:
                    candidates = scatter_list
                fixed_splits = None
                for candidate in candidates:
                    fixed_splits = _split_full_tensor_if_needed(candidate, expected_shape, self.tp_size)
                    if fixed_splits is not None:
                        break
                if fixed_splits is not None:
                    scatter_list = [s.contiguous() for s in fixed_splits]
                    logger.warning(
                        "slime Megatron Bridge TP scatter patch split full tensor "
                        "from %s into %d shard(s) of %s",
                        tuple(candidate.shape),
                        len(scatter_list),
                        tuple(scatter_list[0].shape),
                    )

        torch.distributed.scatter(
            output,
            scatter_list,
            src=global_src,
            group=self.tp_group,
        )
        return output

    mapping_cls.scatter_to_tp_ranks = scatter_to_tp_ranks
    mapping_cls._slime_tp_scatter_patch_applied = True
    mapping_cls._slime_original_scatter_to_tp_ranks = original_scatter_to_tp_ranks

    column_cls = param_mapping.ColumnParallelMapping
    if not getattr(column_cls, "_slime_fsdp_dtensor_patch_applied", False):
        original_column_hf_to_megatron = column_cls.hf_to_megatron

        def column_hf_to_megatron(self, hf_weights, megatron_module):
            if self.tp_size == 1:
                return hf_weights

            normalized_param = self._normalize_expert_param_name(self.megatron_param)
            _, target_param = param_mapping.get_module_and_param_from_name(megatron_module, normalized_param)
            if not isinstance(target_param, param_mapping.DTensor):
                return original_column_hf_to_megatron(self, hf_weights, megatron_module)

            if self.tp_rank == 0:
                if hf_weights is None:
                    raise ValueError("hf_weights should not be None on rank 0")

                if hf_weights.dtype != target_param.dtype:
                    if not getattr(column_cls, "_dtype_warned", False):
                        column_cls._dtype_warned = True
                        logger.warning(
                            "Dtype mismatch: HF weights are %s but Megatron module uses %s. "
                            "Casting all mismatched weights to %s (further warnings suppressed).",
                            hf_weights.dtype,
                            target_param.dtype,
                            target_param.dtype,
                        )
                    hf_weights = hf_weights.to(target_param.dtype)

                actual_dim0_size = hf_weights.shape[0]
                expect_dim0_size = target_param.shape[0]
                if actual_dim0_size != expect_dim0_size:
                    assert self.megatron_param in {"embedding.word_embeddings.weight", "output_layer.weight"}, (
                        f"{hf_weights.shape=} {target_param.shape=} {self.tp_size=} "
                        f"{self.megatron_param=} {self.hf_param=}"
                    )
                    hf_weights = param_mapping._pad_right_dim0(
                        hf_weights,
                        pad_size=expect_dim0_size - actual_dim0_size,
                    )

                full_size = hf_weights.shape[0]
                if full_size % self.tp_size != 0:
                    raise ValueError(f"Cannot evenly split dimension 0 size {full_size} across {self.tp_size} TP ranks")
                splits = torch.chunk(hf_weights, self.tp_size, dim=0)
            else:
                splits = None

            output_shape = [target_param.shape[0] // self.tp_size, *target_param.shape[1:]]
            logger.debug(
                "slime Megatron Bridge ColumnParallel DTensor patch mapping %s to local shape %s",
                self.megatron_param,
                tuple(output_shape),
            )
            return self.scatter_to_tp_ranks(
                splits,
                output_shape,
                target_param.dtype,
                target_param.device,
            )

        column_cls.hf_to_megatron = column_hf_to_megatron
        column_cls._slime_fsdp_dtensor_patch_applied = True
        column_cls._slime_original_hf_to_megatron = original_column_hf_to_megatron


@contextmanager
def patch_megatron_model(model):
    unwrapped_model = unwrap_model(model)[0]
    model_config = unwrapped_model.config
    attribute_was_added = False
    if not hasattr(model_config, "share_embeddings_and_output_weights"):
        model_config.share_embeddings_and_output_weights = unwrapped_model.share_embeddings_and_output_weights
        attribute_was_added = True

    try:
        yield
    finally:
        if attribute_was_added:
            delattr(model_config, "share_embeddings_and_output_weights")
