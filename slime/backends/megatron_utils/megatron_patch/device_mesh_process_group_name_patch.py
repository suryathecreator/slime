# PyTorch 2.8's DeviceMesh.from_group assumes ProcessGroup.group_name is set.
# Some Megatron-created process groups in our container expose the property but
# raise "ProcessGroup name not set", which breaks Megatron-FSDP setup before
# training starts. Use DeviceMesh's existing private registry with deterministic
# synthetic names for those unnamed groups.

import logging
import warnings

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.distributed.device_mesh as device_mesh_module
    from torch.distributed import get_process_group_ranks

    _DeviceMesh = device_mesh_module.DeviceMesh
    _ProcessGroup = device_mesh_module.ProcessGroup
    _original_from_group = _DeviceMesh.from_group

    def _synthetic_group_name(group, index):
        return f"slime_unnamed_pg_{index}_{id(group)}"

    def _from_group_with_unnamed_pg(group, device_type, mesh=None, *, mesh_dim_names=None):
        try:
            return _original_from_group(
                group,
                device_type=device_type,
                mesh=mesh,
                mesh_dim_names=mesh_dim_names,
            )
        except RuntimeError as exc:
            if "ProcessGroup name not set" not in str(exc):
                raise

            if isinstance(group, _ProcessGroup):
                groups = [group]
                group_ranks = get_process_group_ranks(group)
                mesh_tensor = torch.tensor(group_ranks, device="cpu", dtype=torch.int)
            else:
                groups = list(group)
                mesh_tensor = (
                    mesh.detach().to(dtype=torch.int, device="cpu")
                    if isinstance(mesh, torch.Tensor)
                    else torch.tensor(mesh, device="cpu", dtype=torch.int)
                )

            names = []
            for index, process_group in enumerate(groups):
                try:
                    name = process_group.group_name
                except RuntimeError as name_exc:
                    if "ProcessGroup name not set" not in str(name_exc):
                        raise
                    name = _synthetic_group_name(process_group, index)
                names.append(name)

            device_mesh = _DeviceMesh(
                device_type,
                mesh_tensor,
                mesh_dim_names=mesh_dim_names,
                _init_backend=False,
            )
            device_mesh._dim_group_names = names
            for name, process_group in zip(names, groups):
                device_mesh._pg_registry[name] = process_group

            logger.warning(
                "slime DeviceMesh process-group-name patch used synthetic names for "
                "unnamed process groups: %s",
                names,
            )
            return device_mesh

    _DeviceMesh.from_group = staticmethod(_from_group_with_unnamed_pg)
    logger.info("slime DeviceMesh process-group-name patch applied.")

except ImportError as exc:
    warnings.warn(
        f"slime DeviceMesh process-group-name patch not applied; import failed ({exc!r}).",
        stacklevel=2,
    )
