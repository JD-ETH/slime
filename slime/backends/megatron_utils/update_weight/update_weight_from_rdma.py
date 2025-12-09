from .update_weight_from_distributed import UpdateWeightFromDistributed

from megatron.core import mpu
from typing import Callable
import torch
from argparse import Namespace
from collections.abc import Mapping, Sequence
from ray.actor import ActorHandle
import time 
import ray 
from slime.backends.sglang_utils.sglang_training_rdma_p2p import P2PTrainingTransferEngine
import logging 
import socket
from tqdm import tqdm

logger = logging.getLogger(__name__)
class UpdateWeightFromRDMA(UpdateWeightFromDistributed):
    """
    Update distributed engines via P2P RDMA transfer. Each PP rank: group "slime-pp_{pp_rank}",
    only DP=TP=0 transfers. Uses P2PTrainingTransferEngine for RDMA communication.

    Inherits from UpdateWeightFromDistributed and overrides only the RDMA-specific methods.
    This reduces code duplication and improves maintainability.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
        vocab_size: int,
    ) -> None:
        """
        Initialize. P2PTrainingTransferEngine created in connect_rollout_engines.
        Calls parent constructor and adds P2P RDMA specific attributes.
        """
        # Call parent constructor to initialize all base attributes
        super().__init__(
            args, model, weights_getter,
            model_name=model_name,
            quantization_config=quantization_config,
            vocab_size=vocab_size
        )

        # P2P RDMA specific initialization
        self.training_p2p_transfer_engine = None
        self.master_addr = None
        self.master_port = None

    def connect_rollout_engines(
        self, rollout_engines: Sequence[ActorHandle], rollout_engine_lock: ActorHandle
    ) -> None:
        """
        Initialize P2PTrainingTransferEngine if PP source (DP=TP=0).
        Overrides parent method to use P2P RDMA instead of NCCL.
        """
        # Store rollout engines and lock (same as parent)
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock

        # Determine if this is a PP source rank (same logic as parent)
        self._is_pp_src_rank = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
        )
        pp_rank = mpu.get_pipeline_model_parallel_rank()

        if self._is_pp_src_rank:
            self._group_name = f"slime-pp_{pp_rank}"

            # Stop existing P2P engine if running (task requirement)
            if self._model_update_groups is not None:
                if self.training_p2p_transfer_engine is not None:
                    self.training_p2p_transfer_engine.stop()
                    self.master_addr = None  # Reset as per task requirement
                    self.master_port = None  # Reset as per task requirement
                self._model_update_groups = None

            # Get master address and port for P2P communication
            self.master_addr = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                self.master_port = sock.getsockname()[1]

            # Initialize P2PTrainingTransferEngine
            self.training_p2p_transfer_engine = P2PTrainingTransferEngine(
                master_ip=self.master_addr, # TODO(ask): Engine should be localhost?
                master_port=self.master_port,
                gpu_id=0,  ## TODO(ask): gpu_id works or not in this case?
                ib_device=None  # Auto-detect InfiniBand device
            )

            # Start the training transfer engine
            self.training_p2p_transfer_engine.start()

            # Indicate that P2P system is active (as per task requirement)
            # TODO(jsf):  check if `self._model_update_groups` is redundant
            self._model_update_groups = "p2p_active"  # Use as indicator

            logger.info(f"P2PTrainingTransferEngine started on {self.master_addr}:{self.master_port}")

    def _update_bucket_weights_from_distributed(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        """
        Register weights with P2PTrainingTransferEngine and wait for transfers to complete.
        Based on lines 518-545 in SGLang test: register_weights pattern.
        Overrides parent method to use P2P RDMA instead of NCCL broadcast.
        """
        if not self._is_pp_src_rank or not converted_named_tensors:
            return

        # Lock the rollout engines to prevent concurrent operations (same as parent)
        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            time.sleep(0.1)

        try:
            # Register all weights with the P2P training transfer engine
            # This follows the pattern from SGLang test lines 518-537
            for name, tensor in converted_named_tensors:
                self.training_p2p_transfer_engine.register_buffer(name, tensor)


            # Send metadata to rollout engines (similar to parent NCCL version)
            # NOTE: training node will broadcast all weights to all rollout nodes,
            # while different rollout engines may need different weights. We need
            # a mask list here inidicating the weights for different rollout engines.
            # TODO(JD): where the training_host<->rollout_nodes mapping happens?
            refs = [
                engine.update_weights_from_distributed.remote(
                    names=[name for name, _ in converted_named_tensors],
                    dtypes=[param.dtype for _, param in converted_named_tensors],
                    shapes=[param.shape for _, param in converted_named_tensors],
                    group_name=self._group_name,
                    weight_version=str(self.weight_version),
                    session_id=f"{self.master_addr}:{self.master_port}",  # Pass P2P session info
                )
                for engine in self.rollout_engines
            ]

            # Wait for all P2P transfers to complete
            ray.get(refs)
            converted_named_tensors.clear()

        finally:
            # Release the lock (same as parent)
            ray.get(self.rollout_engine_lock.release.remote())
            if pbar:
                pbar.update(1)