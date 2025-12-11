"""
Remote Transfer Plan - Abstract transfer planning for NCCL and RDMA weight updates.

This module provides a unified interface for determining transfer sources and planning
weight transfer tasks across different communication backends (NCCL, RDMA).
"""

import logging
from argparse import Namespace
from dataclasses import dataclass
from typing import Literal, Sequence
import torch
from megatron.core import mpu
from .common import named_params_and_buffers

logger = logging.getLogger(__name__)


@dataclass
class TransferTask:
    """
    Attributes:
        session: Session identifier (e.g., NCCL group name or Transfer Engine Session Id)
        names: Full list of tensor names to be transferred in this task
    """
    session: str
    names: list[str]


class RemoteTransferPlan:
    """
    Plans and manages remote weight transfers for both NCCL and RDMA backends, assuming static training and rollout placements. 
    
    The plan assumes an all-gather in the tp/ep dimension. 
    """

    def __init__(self, args: Namespace, mode: Literal["nccl", "rdma"] = "nccl") -> None:
        """
        Initialize the transfer plan.
        
        Args:
            args: Configuration namespace containing parallelism settings
            mode: Transfer backend mode - either "nccl" or "rdma"
        """
        self.args = args
        self.mode = mode
        
        self._pp_rank, self._pp_size = mpu.get_pipeline_model_parallel_rank(), mpu.get_pipeline_model_parallel_world_size()
        self._ep_rank, self._ep_soze = mpu.get_expert_model_parallel_rank(), mpu.get_expert_model_parallel_world_size()
        self._tp_rank, self._tp_size = mpu.get_tensor_model_parallel_rank(), mpu.get_tensor_model_parallel_world_size()
        self._dp_rank, self._dp_size = mpu.get_data_parallel_rank(with_context_parallel=True), mpu.get_data_parallel_world_size(with_context_parallel=True)
        
        logger.info(
            f"RemoteTransferPlan initialized: mode={mode}, pp_rank={self._pp_rank}/{self._pp_size}, tp_rank={self._tp_rank}/{self._tp_size}, "
            f"ep_rank={self._ep_rank}/{self._ep_soze}, dp_rank={self._dp_rank}/{self._dp_size}"
        )

    def is_source(self) -> bool:
        """
        Determine if the current rank needs to initiate weight transfer.
        
        Returns:
            A tuple of (is_source, group_name) where:
                - is_source: True if this rank should initiate transfers
                - group_name: The nccl group identifier (e.g., "slime-pp_0") or transfer engine session id
        """
        if self.mode == "nccl":
            # NCCL only load from DP=TP=0 PP ranks to all rollout engines.
            return (
                mpu.get_data_parallel_rank(with_context_parallel=True) == 0 
                and mpu.get_tensor_model_parallel_rank() == 0
            )
        raise NotImplementedError("rdma is not yet implemented.")
    
    def get_transfer_tasks(self, model: Sequence[torch.nn.Module]) -> Sequence[TransferTask]:
        # Generate session identifier based on mode
        if self.mode == "nccl":
            session = f"slime-pp_{self._pp_rank}"
            # In NCCL mode, the transfer is simply a broadcast from DP=TP=0 to all rollout engines.
            return [TransferTask(session=session, names=list([name for name, _ in named_params_and_buffers(self.args, model)]))]

        raise NotImplementedError("RDMA not implemented")