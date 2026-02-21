import socket
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray import ObjectRef
from ray.actor import ActorHandle
from tqdm import tqdm

from slime.utils.distributed_utils import init_process_group

from .update_weight_from_remote import UpdateWeightFromRemote


class UpdateWeightFromDistributed(UpdateWeightFromRemote):
    """
    Update distributed engines via NCCL. Each PP rank: group "slime-pp_{pp_rank}",
    only DP=TP=0 broadcasts. Non-expert (TP) and expert (EP) params separate.

    When PP>1, source ranks synchronize per-bucket via all_gather_object so that
    global rank 0 dispatches Ray calls for ALL PP sources before any of them
    broadcast. This prevents deadlocks from engines joining NCCL groups in
    inconsistent order.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        """
        Initialize. Groups created in connect_rollout_engines.
        """
        super().__init__(
            args,
            model,
            weights_getter,
            model_name=model_name,
            quantization_config=quantization_config,
            weight_update_mode="nccl",
        )

        if self._is_source:
            assert self.transfer_plan.mode == "nccl", "Only NCCL supported currently."
            self._group_name = self.transfer_plan.get_nccl_group()
        # Indicates if the nccl group has been established.
        self._model_update_groups = None
        # Leader rank dispatches Ray calls and waits for engine completion.
        self._is_leader = dist.get_rank() == 0
        assert not self._is_leader or self._is_source, "Leader rank must be a source rank"
        # Ray ObjectRefs tracking engine-side weight update completion,
        # accumulated per phase and drained in _finish_phase_sync.
        self._pending_engine_refs = []

    def connect_rollout_engines(
        self, rollout_engines: Sequence[ActorHandle], rollout_engine_lock: ActorHandle
    ) -> None:
        """
        Create NCCL "slime-pp_{pp_rank}" if PP source (DP=TP=0).
        """
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock

        # For TP:
        #   1. AllGather paramters to rank 0
        #   2. Broadcast parameters from rank 0 to all sglang engines
        if self._is_source:
            if self._model_update_groups is not None:
                # Reestablish group if already connected, e.g. new instance has joined.
                disconnect_rollout_engines_from_distributed(
                    self.args, self._group_name, self._model_update_groups, self.rollout_engines
                )
            self._model_update_groups = connect_rollout_engines_from_distributed(
                self.args, self._group_name, rollout_engines
            )

    def _update_bucket_weights_from_remote(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        """
        Sync PP sources via all_gather_object (no-op for PP=1), rank 0 dispatches
        Ray calls for ALL PP sources, then each source broadcasts on its own
        NCCL group concurrently.
        """
        # Gather tensor metadata across PP sources; leader schedules engine updates
        self._gather_meta_and_schedule_engines(converted_named_tensors)
        # NCCL broadcast on own group
        handles = []
        for _, param in converted_named_tensors:
            handles.append(dist.broadcast(param.data, 0, group=self._model_update_groups, async_op=True))
        for handle in handles:
            handle.wait()

        converted_named_tensors.clear()
        pbar.update(1)

    def _gather_meta_and_schedule_engines(self, converted_named_tensors: list[tuple[str, torch.Tensor]]) -> None:
        """Gather tensor metadata from all PP sources to leader; leader schedules engine updates."""
        pp_group = mpu.get_pipeline_model_parallel_group()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        my_meta = (
            self._group_name,
            [name for name, _ in converted_named_tensors],
            [param.dtype for _, param in converted_named_tensors],
            [tuple(param.shape) for _, param in converted_named_tensors],
        )
        per_pp_meta = [None] * pp_size
        dist.all_gather_object(per_pp_meta, my_meta, group=pp_group)
        if self._is_leader:
            self._dispatch_from_gathered_meta(per_pp_meta)

    def _dispatch_from_gathered_meta(self, per_pp_meta: list) -> None:
        """Dispatch Ray weight-update calls for every PP source that has data."""
        for group_name, names, dtypes, shapes in per_pp_meta:
            if names:
                refs = [
                    engine.update_weights_from_distributed.remote(
                        names=names,
                        dtypes=dtypes,
                        shapes=shapes,
                        group_name=group_name,
                        weight_version=str(self.weight_version),
                    )
                    for engine in self.rollout_engines
                ]
                self._pending_engine_refs.extend(refs)

    def _finish_phase_sync(self) -> None:
        """
        After a transfer phase, sources that finished early participate in empty
        all_gather_object rounds so sources still broadcasting can synchronize.
        Exits once every PP source sends empty metadata. Then rank 0 waits for
        all accumulated engine refs before proceeding.
        """
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        if pp_size > 1:
            pp_group = mpu.get_pipeline_model_parallel_group()
            while True:
                empty_meta = (self._group_name, [], [], [])
                per_pp_meta = [None] * pp_size
                dist.all_gather_object(per_pp_meta, empty_meta, group=pp_group)
                if all(not meta[1] for meta in per_pp_meta):
                    break
                if self._is_leader:
                    self._dispatch_from_gathered_meta(per_pp_meta)

        if self._is_leader and self._pending_engine_refs:
            ray.get(self._pending_engine_refs)
        self._pending_engine_refs = []

    def _update_weights(self, named_params_and_buffers: Sequence[tuple[str, torch.Tensor]]) -> None:
        super()._update_weights(named_params_and_buffers)
        if self._is_source:
            self._finish_phase_sync()

    def _update_expert_weights(self, named_params_and_buffers: Sequence[tuple[str, torch.Tensor]]) -> None:
        super()._update_expert_weights(named_params_and_buffers)
        if self._is_source:
            self._finish_phase_sync()


def connect_rollout_engines_from_distributed(
    args: Namespace, group_name: str, rollout_engines: Sequence[ActorHandle]
) -> dist.ProcessGroup:
    """
    Create NCCL group: training rank 0 + all engine GPUs. Blocks until joined.
    """
    master_address = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
    world_size = len(rollout_engines) * args.rollout_num_gpus_per_engine + 1

    refs = [
        engine.init_weights_update_group.remote(
            master_address,
            master_port,
            i * args.rollout_num_gpus_per_engine + 1,
            world_size,
            group_name,
            backend="nccl",
        )
        for i, engine in enumerate(rollout_engines)
    ]
    model_update_groups = init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_address}:{master_port}",
        world_size=world_size,
        rank=0,
        group_name=group_name,
    )
    ray.get(refs)
    return model_update_groups


def disconnect_rollout_engines_from_distributed(args, group_name, model_update_groups, rollout_engines):
    """
    Destroy NCCL on training and engines.
    """
    refs = [engine.destroy_weights_update_group.remote(group_name) for engine in rollout_engines]
    dist.destroy_process_group(model_update_groups)
    ray.get(refs)


def update_weights_from_distributed(
    group_name: str,
    group: dist.ProcessGroup,
    weight_version: int,
    rollout_engines: Sequence[ActorHandle],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
) -> list[ObjectRef]:
    """
    Send metadata (Ray), broadcast tensors (NCCL rank 0 → engines).
    """
    refs = [
        engine.update_weights_from_distributed.remote(
            names=[name for name, _ in converted_named_tensors],
            dtypes=[param.dtype for _, param in converted_named_tensors],
            shapes=[param.shape for _, param in converted_named_tensors],
            group_name=group_name,
            weight_version=str(weight_version),
        )
        for engine in rollout_engines
    ]

    handles = []
    for _, param in converted_named_tensors:
        handles.append(dist.broadcast(param.data, 0, group=group, async_op=True))
    for handle in handles:
        handle.wait()

    return refs
