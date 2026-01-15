import dataclasses
import logging
import queue
import threading
import time
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence

import ray
import sglang.srt.distributed.parallel_state as sglang_parallel_state
import sglang.srt.layers.dp_attention as sglang_dp_attention
import sglang.srt.server_args as sglang_server_args
import torch
from mooncake.engine import TransferEngine
from ray.actor import ActorHandle
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_loader import get_model
from sglang.srt.model_loader.remote_instance_weight_loader_utils import register_memory_region_v2
from sglang.srt.server_args import ServerArgs
from torch_memory_saver import torch_memory_saver
from tqdm import tqdm

from slime.utils.memory_utils import print_memory

from .update_weight_from_remote import UpdateWeightFromRemote

logger = logging.getLogger(__name__)


def create_server_args_from_dict(data_dict: dict) -> ServerArgs:
    # Reconstruct Sglang ServerArgs from sglang Http query.
    valid_fields = {f.name for f in dataclasses.fields(ServerArgs)}
    filtered_data = {k: v for k, v in data_dict.items() if k in valid_fields}
    return ServerArgs(**filtered_data)


@dataclasses.dataclass
class RemoteWeightInfo:
    # Remote session and weight registration info.
    session_id: str
    weights_info: dict[str, tuple[int, int, int]]  # name -> (remote_address, numel, element_size)


@dataclasses.dataclass
class TransferTask:
    """Represents a transfer task to be executed asynchronously."""
    task_id: str
    transfer_bundle: "TransferBundle"
    updated_names: list[str]
    priority: int = 0  # Higher priority tasks execute first
    created_at: float = dataclasses.field(default_factory=time.time)
    is_ready: bool = False  # Flag to indicate when weight loading is complete
    is_completed: bool = False  # Flag to indicate when transfer is complete
    error: Exception | None = None  # Store any error that occurred during execution

    def __lt__(self, other):
        """Enable priority queue ordering - higher priority first, then by creation time"""
        if self.priority != other.priority:
            return self.priority > other.priority  # Higher priority first
        return self.created_at < other.created_at  # Earlier tasks first for same priority


class RDMATransferQueue:
    """Queue-based RDMA transfer task manager with priority support."""

    def __init__(self, max_concurrent_transfers: int = 1):
        self.task_queue = queue.PriorityQueue()
        self.completed_tasks: dict[str, TransferTask] = {}
        self.active_tasks: dict[str, TransferTask] = {}
        self.max_concurrent_transfers = max_concurrent_transfers
        self._lock = threading.Lock()
        self._next_task_id = 0
        self._shutdown = False

    def generate_task_id(self) -> str:
        """Generate unique task ID."""
        with self._lock:
            self._next_task_id += 1
            return f"transfer_task_{self._next_task_id}"

    def enqueue_task(self, transfer_bundle: "TransferBundle", updated_names: list[str], priority: int = 0) -> str:
        """Add a new transfer task to the queue."""
        task_id = self.generate_task_id()
        task = TransferTask(
            task_id=task_id,
            transfer_bundle=transfer_bundle,
            updated_names=updated_names.copy(),
            priority=priority,
            is_ready=True
        )

        self.task_queue.put(task)
        logger.debug(f"[RDMA Queue] Enqueued task {task_id} with {len(updated_names)} weights (priority {priority})")
        return task_id

    def execute_available_tasks(self) -> list[str]:
        """Execute tasks that can be run concurrently. Returns list of completed task IDs."""
        completed_task_ids = []

        # Start new tasks if we have capacity
        while (len(self.active_tasks) < self.max_concurrent_transfers and
               not self.task_queue.empty() and
               not self._shutdown):
            try:
                task = self.task_queue.get_nowait()
                if task.is_ready and not task.is_completed:
                    self.active_tasks[task.task_id] = task
                    self._execute_single_task(task)
            except queue.Empty:
                break

        # Check for completed tasks
        with self._lock:
            for task_id, task in list(self.active_tasks.items()):
                if task.is_completed:
                    completed_task_ids.append(task_id)
                    self.completed_tasks[task_id] = task
                    del self.active_tasks[task_id]

        return completed_task_ids

    def _execute_single_task(self, task: TransferTask) -> None:
        """Execute a single transfer task."""
        try:
            logger.debug(f"[RDMA Queue] Executing task {task.task_id}")

            # Ensure CUDA is synchronized before transfer
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Execute the transfer
            task.transfer_bundle.execute_each(task.updated_names)

            # Ensure transfer is fully completed
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            task.is_completed = True
            logger.debug(f"[RDMA Queue] Completed task {task.task_id} for {len(task.updated_names)} weights")

        except Exception as e:
            logger.error(f"[RDMA Queue] Error executing task {task.task_id}: {e}")
            task.error = e
            task.is_completed = True  # Mark as completed even on error to prevent hanging

            # Attempt to recover CUDA state
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except:
                    logger.error(f"[RDMA Queue] Failed to recover CUDA after error in task {task.task_id}")

    def wait_for_task(self, task_id: str, timeout: float = 30.0) -> bool:
        """Wait for a specific task to complete. Returns True if completed, False if timeout."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Execute available tasks to make progress
            self.execute_available_tasks()

            # Check if our task is completed
            if task_id in self.completed_tasks:
                task = self.completed_tasks[task_id]
                if task.error:
                    raise task.error
                return True

            time.sleep(0.01)  # Small sleep to prevent busy waiting

        logger.warning(f"[RDMA Queue] Task {task_id} timed out after {timeout}s")
        return False

    def wait_for_bundle(self, transfer_bundle: "TransferBundle", timeout: float = 30.0) -> bool:
        """Wait for all tasks associated with a specific transfer bundle to complete."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            self.execute_available_tasks()

            # Check if any active tasks are for this bundle
            bundle_tasks_active = any(
                task.transfer_bundle is transfer_bundle
                for task in self.active_tasks.values()
            )

            # Check if any queued tasks are for this bundle
            temp_queue = queue.PriorityQueue()
            bundle_tasks_queued = False

            while not self.task_queue.empty():
                try:
                    task = self.task_queue.get_nowait()
                    temp_queue.put(task)
                    if task.transfer_bundle is transfer_bundle and not task.is_completed:
                        bundle_tasks_queued = True
                except queue.Empty:
                    break

            # Restore queue
            while not temp_queue.empty():
                self.task_queue.put(temp_queue.get_nowait())

            if not bundle_tasks_active and not bundle_tasks_queued:
                return True

            time.sleep(0.01)

        logger.warning(f"[RDMA Queue] Bundle tasks timed out after {timeout}s")
        return False

    def drain_all_tasks(self, timeout: float = 60.0) -> bool:
        """Execute all remaining tasks in the queue. Returns True if all completed."""
        start_time = time.time()

        while (not self.task_queue.empty() or self.active_tasks) and time.time() - start_time < timeout:
            self.execute_available_tasks()
            time.sleep(0.01)

        success = self.task_queue.empty() and not self.active_tasks
        if not success:
            logger.warning(f"[RDMA Queue] Failed to drain all tasks within {timeout}s")
        else:
            logger.info(f"[RDMA Queue] Successfully drained all tasks")

        return success

    def get_queue_stats(self) -> dict:
        """Get statistics about the queue state."""
        return {
            "queued_tasks": self.task_queue.qsize(),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "max_concurrent": self.max_concurrent_transfers
        }

    def shutdown(self):
        """Shutdown the queue and prevent new tasks."""
        self._shutdown = True


@dataclasses.dataclass
class TransferBundle:
    model_replica: Sequence[torch.nn.Module]
    engine: TransferEngine
    weight_memory_registry: dict
    remote_weight_infos: list[RemoteWeightInfo]

    def add_remote_session(self, remote_info: RemoteWeightInfo) -> None:
        self.remote_weight_infos.append(remote_info)

    def execute_each(self, names: Sequence[str]) -> None:
        # Ensure CUDA synchronization before accessing memory pointers
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        try:
            # Find local pointers and lengths for the given names
            source_ptrs, source_lens = [], []
            for name in names:
                tensor_register = self.weight_memory_registry[name]
                data_ptr, numel, ele_size = tensor_register
                source_ptrs.append(data_ptr)
                source_lens.append(numel * ele_size)

            # Match with remote sessions and target pointers
            for remote_session in self.remote_weight_infos:
                session_id, remote_weights_info = remote_session.session_id, remote_session.weights_info
                target_ptrs = []
                for name in names:
                    target_ptrs.append(remote_weights_info[name][0])  # remote address

                # Batch transfer weights through RDMA
                ret = self.engine.batch_transfer_async_write(session_id, source_ptrs, target_ptrs, source_lens)
                if ret < 0:
                    logger.error(f"[RDMA] Async batch transfer failed with error code {ret}")

            # Ensure transfer operations are completed before returning
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        except Exception as e:
            logger.error(f"[RDMA] Error in execute_each: {e}")
            # Attempt to recover CUDA state
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except:
                    logger.error("[RDMA] Failed to synchronize CUDA after error")
            raise

    def execute(self) -> None:
        # Execute transfer for each target session using this replica.
        # Ensure CUDA synchronization before accessing memory pointers
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        try:
            for remote_session in self.remote_weight_infos:
                session_id, remote_weights_info = remote_session.session_id, remote_session.weights_info
                source_ptrs, target_ptrs, source_lens = [], [], []
                for name, tensor_register in self.weight_memory_registry.items():
                    data_ptr, numel, ele_size = tensor_register
                    source_ptrs.append(data_ptr)
                    target_ptrs.append(remote_weights_info[name][0])  # remote address
                    source_lens.append(numel * ele_size)

                # Batch transfer weights through RDMA
                ret = self.engine.batch_transfer_sync_write(session_id, source_ptrs, target_ptrs, source_lens)
                if ret < 0:
                    raise RuntimeError(f"Batch transfer weights via RDMA failed with error code {ret}.")

            # Ensure all transfer operations are completed
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        except Exception as e:
            logger.error(f"[RDMA] Error in execute: {e}")
            # Attempt to recover CUDA state
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except:
                    logger.error("[RDMA] Failed to synchronize CUDA after error")
            raise


class UpdateWeightFromRDMA(UpdateWeightFromRemote):
    """
    Update weights from RDMA using Transfer Engine.

    Similar to UpdateWeightFromNCCL but uses P2P RDMA transfer engine for the underlying weight transfer. Workflow
    consists of following steps:
    1. Based off the transfer plan, query the target rollout engines for remote session and weight info during connect_rollout_engines.
    2. Construct local model replica according to the plan and attach target session id and weight memory registry
    2. Do TP-EP all-gather for bucketed weights on parameters needing transfer from local just as in NCCL case.
    3. Convert the gathered HF tensor into target shape and register them with Engine.
    4. Call engine to batch transfer weights for each transfer task.
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
        super().__init__(
            args,
            model,
            weights_getter,
            model_name=model_name,
            quantization_config=quantization_config,
            weight_update_mode="rdma",
        )

        # For torch memory saver tagging
        self.tag = f"Model Replica {self.global_rank}"
        self._model_on_cpu = False
        self.pipelined_transfer = args.rdma_pipelined_transfer

        # Queue-based transfer system
        max_concurrent = getattr(args, 'rdma_max_concurrent_transfers', 1)
        self.transfer_queue = RDMATransferQueue(max_concurrent_transfers=max_concurrent)
        logger.info(f"[RDMA] Initialized transfer queue with max_concurrent={max_concurrent}")

    def _process_transfer_queue(self) -> None:
        """Process tasks in the transfer queue."""
        completed_task_ids = self.transfer_queue.execute_available_tasks()
        if completed_task_ids:
            logger.debug(f"[RDMA] Processed {len(completed_task_ids)} transfer tasks")

        # Log queue stats periodically for debugging
        stats = self.transfer_queue.get_queue_stats()
        if stats["queued_tasks"] > 0 or stats["active_tasks"] > 0:
            logger.debug(f"[RDMA] Queue stats: {stats}")

    def _wait_for_bundle_completion(self, transfer_bundle: TransferBundle, timeout: float = 30.0) -> None:
        """Wait for all transfers on this bundle to complete."""
        if not self.transfer_queue.wait_for_bundle(transfer_bundle, timeout):
            logger.warning(f"[RDMA] Timeout waiting for bundle transfers to complete")

    def connect_rollout_engines(
        self, rollout_engines: Sequence[ActorHandle], rollout_engine_lock: ActorHandle
    ) -> None:
        """
        Initialize P2PTrainingTransferEngine if serves as a source.
        """
        # Store rollout engines and lock
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock

        if self._is_source:
            # Query Engine session and weight info from rollout instances according to the transfer plan
            self.remote_weight_infos_by_session_id = {}
            targets = self.transfer_plan.plan_p2p()
            targets_to_query = set((target.engine_ind, target.engine_rank) for target in targets)
            targets_to_session_id, self.session_id_to_engine_rank = {}, {}
            self.session_id_to_server_args = {}
            for engine_ind, engine_rank in targets_to_query:
                session_id, weights_info = ray.get(
                    self.rollout_engines[engine_ind].get_remote_instance_transfer_engine_info.remote(rank=engine_rank)
                )
                self.session_id_to_engine_rank[session_id] = engine_rank
                self.session_id_to_server_args[session_id] = create_server_args_from_dict(
                    ray.get(self.rollout_engines[engine_ind].get_server_info.remote())
                )
                assert (
                    session_id is not None
                ), f"Failed to get session id from rollout engine {engine_ind} rank {engine_rank}"
                logger.info(
                    f"[RDMA] Obtained remote {session_id} info from rollout engine {engine_ind} rank {engine_rank}"
                )
                logger.info(f"[RDMA] Remote weight info has {len(weights_info)} tensors.")
                # logger.info(list(weights_info.keys()))
                self.remote_weight_infos_by_session_id[session_id] = weights_info
                targets_to_session_id[(engine_ind, engine_rank)] = session_id

            print_memory("[RDMA] After obtaining remote weight info")

            # Create local model replicas and transfer engines for each target rollout shard
            self.engines = {}
            # Associate transfer tasks based on obtained session and weight info
            with torch_memory_saver.region(tag=self.tag):
                for target in targets:
                    session_id = targets_to_session_id[(target.engine_ind, target.engine_rank)]
                    remote_info = RemoteWeightInfo(session_id, self.remote_weight_infos_by_session_id[session_id])
                    # Instantiate the local model replicas and a corresponding transfer engine with memory registry for each type of rollout shard.
                    # TODO verify:
                    # - if sglang dp is enabled, then attn_tp is equal to tp // dp
                    # - if sglang ep is enabled, then moe-tp is equal to tp // ep
                    # generally tp * pp should be equal to the world_size
                    if target.engine_rank not in self.engines:
                        transfer_engine = self._create_transfer_engine()
                        parallel_rank_dict = self.transfer_plan.tp_conversion(target.engine_rank)
                        logger.info(
                            f"[RDMA] Creating model replica for engine rank {target.engine_rank} with rank dict {parallel_rank_dict}"
                        )
                        model_replica = self._create_inference_replica(
                            self.args.hf_checkpoint,
                            pp_shard=target.source_shard,
                            target_rank=target.engine_rank,  # NOTE: here we assume that sglang_tp == world_size when pp_size == 1
                            target_tp=self.args.rollout_num_gpus_per_engine,
                            dp_rank=parallel_rank_dict["dp_rank"],
                            dp_size=self.transfer_plan._rollout_dp_size,
                            attn_tp_rank=parallel_rank_dict["attn_tp_rank"],
                            attn_tp_size=self.transfer_plan._rollout_attn_tp_size,
                            ep_rank=parallel_rank_dict["ep_rank"],
                            ep_size=self.transfer_plan._rollout_ep_size,
                            moe_tp_rank=parallel_rank_dict["moe_tp_rank"],
                            moe_tp_size=self.transfer_plan._rollout_moe_tp_size,
                            server_args=self.session_id_to_server_args[session_id],
                        )
                        print_memory(f"[RDMA] After model replica at {target.engine_rank}")
                        weight_memory_registry = self._register_replica_memory(
                            model_replica, self.remote_weight_infos_by_session_id[session_id], transfer_engine
                        )
                        self.engines[target.engine_rank] = TransferBundle(
                            model_replica, transfer_engine, weight_memory_registry, [remote_info]
                        )
                    else:
                        self.engines[target.engine_rank].add_remote_session(remote_info)

            print_memory("[RDMA] After Local Engine Replicas and engine Creation")

    def _register_replica_memory(self, model_replica, remote_weight_info, transfer_engine) -> dict:
        # Verify the 1-to-1 mapping between local replica and remote weights expected.
        for name, tensor in model_replica.named_parameters():
            if name not in remote_weight_info:
                raise RuntimeError(f"Local replica parameter {name} not found in remote replica.")
            remote_numel, remote_ele_size = remote_weight_info[name][1], remote_weight_info[name][2]
            if tensor.numel() != remote_numel or tensor.element_size() != remote_ele_size:
                raise RuntimeError(
                    f"Local replica parameter {name} numel {tensor.numel()} size {tensor.element_size()} does not match remote numel {remote_numel} size {remote_ele_size}."
                )
            if tensor.device.type != "cuda":
                raise RuntimeError(f"Local replica parameter {name} is not on CUDA device.")
        weight_memory_registry = register_memory_region_v2(model_replica, transfer_engine)

        logger.info(
            f"[RDMA] Registered {len(list(model_replica.named_parameters()))} tensors from replica with transfer engine."
        )
        return weight_memory_registry

    def _create_transfer_engine(self) -> TransferEngine:
        transfer_engine = TransferEngine()
        local_ip = ray._private.services.get_node_ip_address()
        transfer_engine.initialize(local_ip, "P2PHANDSHAKE", "rdma", "")

        logger.info(f"[RDMA] Local replica Transfer Engine initialized at port {transfer_engine.get_rpc_port()}")
        return transfer_engine

    def _create_inference_replica(
        self,
        model_path: str,
        pp_shard: int,
        target_rank: int,
        target_tp: int,
        dp_rank: int,
        dp_size: int,
        attn_tp_rank: int,
        attn_tp_size: int,
        ep_rank: int,
        ep_size: int,
        moe_tp_rank: int,
        moe_tp_size: int,
        server_args: ServerArgs,
    ):
        """
        Create model replica for target rank with correct tp settings.

        Uses MockSglangDistributedContext to avoid initializing actual distributed environment
        while ensuring the model weights have the correct shape for the target rank.
        """
        model_config = ModelConfig(model_path)
        load_config = LoadConfig(
            load_format="auto",
            tp_rank=target_rank,
            model_loader_extra_config=server_args.model_loader_extra_config,
            rl_quant_profile=server_args.rl_quant_profile,
        )
        device_config = DeviceConfig()

        # Mock the distributed environment to get correct weight shapes
        logger.info(
            f" Engine replica: {target_rank} tp {target_tp} pp_shard {pp_shard}, model pp sharding not implemented, "
            f" dp_rank {dp_rank}/{dp_size}, attn_tp_rank {attn_tp_rank}/{attn_tp_size}, "
            f" ep_rank {ep_rank}/{ep_size}, moe_tp_rank {moe_tp_rank}/{moe_tp_size} "
        )
        # TODO: should take attn_tp/ep/dp into account in the future.
        with MockSglangDistributedContext(
            tp_size=target_tp,
            tp_rank=target_rank,
            dp_rank=dp_rank,
            dp_size=dp_size,
            attn_tp_rank=attn_tp_rank,
            attn_tp_size=attn_tp_size,
            ep_rank=ep_rank,
            ep_size=ep_size,
            moe_tp_rank=moe_tp_rank,
            moe_tp_size=moe_tp_size,
            server_args=server_args,
        ):
            model = get_model(
                model_config=model_config,
                load_config=load_config,
                device_config=device_config,
            )
        device = next(model.parameters()).device
        logger.info(f" Model {device}, params: {sum(p.numel() for p in model.parameters())} ")
        return model

    def leader_post_update(self) -> None:
        ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        # Update weight version as we were write-only.
        ray.get(
            [
                engine.update_weight_version.remote(weight_version=str(self.weight_version))
                for engine in self.rollout_engines
            ]
        )
        return

    def _update_bucket_weights_from_remote(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        """
        The RDMA P2P weight update is implemented as a single side write, meaning the trainer writes its weights directly to the rollout engines' memory.
        Now uses queue-based system for managing transfer tasks.
        """

        if not self._is_source or not converted_named_tensors:
            return

        # Process any pending transfers from previous calls
        self._process_transfer_queue()

        if self._model_on_cpu:
            torch_memory_saver.resume(self.tag)
            self._model_on_cpu = False

        for transfer_bundle in self.engines.values():
            # Wait for any previous transfers on this bundle to complete before loading new weights
            self._wait_for_bundle_completion(transfer_bundle)

            # Load weights into the model replica (this must complete before transfer)
            updated_names = transfer_bundle.model_replica.load_weights(converted_named_tensors)

            if self.pipelined_transfer and updated_names:
                # Enqueue transfer task for async execution
                priority = 0  # Could be made configurable based on bundle importance
                task_id = self.transfer_queue.enqueue_task(
                    transfer_bundle=transfer_bundle,
                    updated_names=updated_names,
                    priority=priority
                )
                logger.debug(f"[RDMA] Queued transfer task {task_id} for {len(updated_names)} weights")

        converted_named_tensors.clear()

    def finish_transfer_task(self) -> None:
        if not self._is_source:
            return

        try:
            # Execute transfer for each engine replica if not using pipelined transfer
            if not self.pipelined_transfer:
                for transfer_bundle in self.engines.values():
                    transfer_bundle.execute()
            else:
                # For pipelined transfer, drain all remaining tasks from the queue
                logger.info("[RDMA] Draining all remaining transfer tasks from queue...")
                if not self.transfer_queue.drain_all_tasks(timeout=60.0):
                    logger.error("[RDMA] Failed to complete all transfer tasks within timeout")

            # Ensure all CUDA operations are completed before offloading
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception as e:
                    logger.error(f"[RDMA] Failed to synchronize CUDA after transfers: {e}")

            # Offload model replicas from memory after transfer.
            if not self._model_on_cpu:
                # Use safer memory printing that handles CUDA errors gracefully
                self._safe_print_memory("[RDMA] Before offloading model replica")
                torch_memory_saver.pause(self.tag)
                self._model_on_cpu = True
                self._safe_print_memory("[RDMA] After offloading model replica")

            # Final queue stats for debugging
            final_stats = self.transfer_queue.get_queue_stats()
            logger.info(f"[RDMA] Transfer session complete. Final queue stats: {final_stats}")

        except Exception as e:
            logger.error(f"[RDMA] Error in finish_transfer_task: {e}")
            # Attempt recovery
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except:
                    logger.error("[RDMA] Failed to recover CUDA state")
        return

    def _safe_print_memory(self, message: str) -> None:
        """Safely print memory usage, handling CUDA errors gracefully."""
        try:
            print_memory(message)
        except Exception as e:
            logger.warning(f"[RDMA] Could not print memory for '{message}': {e}")
            # Try to clear any pending CUDA errors
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except:
                    pass

    def __del__(self):
        """Cleanup method to ensure queue is properly shut down."""
        if hasattr(self, 'transfer_queue'):
            try:
                self.transfer_queue.shutdown()
            except:
                pass  # Ignore errors during cleanup


class MockSglangDistributedContext:
    def __init__(
        self,
        tp_size: int,
        tp_rank: int,
        dp_rank: int,
        dp_size: int,
        attn_tp_rank: int,
        attn_tp_size: int,
        ep_rank: int,
        ep_size: int,
        moe_tp_rank: int,
        moe_tp_size: int,
        server_args: ServerArgs,
    ):
        """
        TODO: Extend this to support ep, and dp attention?
        """
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.pp_size = 1
        self.pp_rank = 0
        self.attn_tp_size = attn_tp_size
        self.attn_tp_rank = attn_tp_rank
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.moe_tp_rank = moe_tp_rank
        self.moe_tp_size = moe_tp_size
        self.server_args = server_args
        # Store active patches for cleanup
        self._patches = []

    def __enter__(self):
        """Apply function-level mocks using unittest.mock.patch."""
        from unittest.mock import MagicMock, patch

        # Mock TP group
        mock_group = MagicMock()
        mock_group.world_size = self.tp_size
        mock_group.rank_in_group = self.tp_rank

        # Mock Attn TP group
        mock_attn_tp_group = MagicMock()
        mock_attn_tp_group.world_size = self.attn_tp_size
        mock_attn_tp_group.rank_in_group = self.attn_tp_rank

        # Mock PP group with proper attributes
        mock_pp_group = MagicMock()
        mock_pp_group.rank_in_group = self.pp_rank
        mock_pp_group.world_size = self.pp_size

        # Mock MoE EP group
        mock_ep_group = MagicMock()
        mock_ep_group.world_size = self.ep_size
        mock_ep_group.rank_in_group = self.ep_rank

        # Mock Moe-tp group
        mock_moe_tp_group = MagicMock()
        mock_moe_tp_group.world_size = self.moe_tp_size
        mock_moe_tp_group.rank_in_group = self.moe_tp_rank

        sglang_parallel_state._MOE_TP = mock_moe_tp_group
        sglang_parallel_state._MOE_EP = mock_ep_group

        # IMPORTANT: Set global variables FIRST, before any patches or model loading.
        # The get_attention_tp_rank() function reads from _ATTN_TP_RANK global variable.
        # Setting this BEFORE model loading ensures the correct value is used.
        sglang_server_args._global_server_args = self.server_args
        sglang_dp_attention._ATTN_TP_RANK = self.attn_tp_rank
        sglang_dp_attention._ATTN_TP_SIZE = self.attn_tp_size
        sglang_dp_attention._ATTN_DP_RANK = self.dp_rank
        sglang_dp_attention._ATTN_DP_SIZE = self.dp_size

        # Mock parallelism getters
        # IMPORTANT: We need to patch functions at BOTH locations:
        # 1. Where they are defined (sglang.srt.layers.dp_attention)
        # 2. Where they are imported and used (sglang.srt.models.qwen3, etc.)
        # This is because Python's import creates a local reference in the importing module.

        self._patches = [
            patch("sglang.srt.distributed.parallel_state.get_tp_group", return_value=mock_group),
            patch("sglang.srt.distributed.parallel_state.get_moe_expert_parallel_rank", return_value=self.ep_rank),
            patch(
                "sglang.srt.distributed.parallel_state.get_moe_expert_parallel_world_size", return_value=self.ep_size
            ),
            patch("sglang.srt.distributed.parallel_state.get_moe_tensor_parallel_rank", return_value=self.moe_tp_rank),
            patch(
                "sglang.srt.distributed.parallel_state.get_moe_tensor_parallel_world_size",
                return_value=self.moe_tp_size,
            ),
            patch(
                "sglang.srt.distributed.get_pp_group", return_value=mock_pp_group
            ),  # TODO: redundant. Delete pp group setting in the future
            patch("sglang.srt.distributed.get_moe_tp_group", return_value=mock_moe_tp_group),
            patch("sglang.srt.distributed.get_tp_group", return_value=mock_group),
            patch("sglang.srt.distributed.get_moe_expert_parallel_rank", return_value=self.ep_rank),
            patch("sglang.srt.distributed.get_moe_expert_parallel_world_size", return_value=self.ep_size),
            patch("sglang.srt.distributed.get_moe_tensor_parallel_rank", return_value=self.moe_tp_rank),
            patch("sglang.srt.distributed.get_moe_tensor_parallel_world_size", return_value=self.moe_tp_size),
            patch(
                "sglang.srt.distributed.parallel_state.get_tensor_model_parallel_world_size", return_value=self.tp_size
            ),
            patch("sglang.srt.distributed.parallel_state.get_tensor_model_parallel_rank", return_value=self.tp_rank),
            # Patch at definition location
            patch("sglang.srt.layers.dp_attention.get_attention_tp_rank", return_value=self.attn_tp_rank),
            patch("sglang.srt.layers.dp_attention.get_attention_tp_size", return_value=self.attn_tp_size),
            patch("sglang.srt.layers.dp_attention.get_attention_tp_group", return_value=mock_attn_tp_group),
            # Patch at import locations in model files - these are critical!
            patch("sglang.srt.models.qwen3.get_attention_tp_rank", return_value=self.attn_tp_rank),
            patch("sglang.srt.models.qwen3.get_attention_tp_size", return_value=self.attn_tp_size),
            patch("sglang.srt.models.qwen3.get_pp_group", return_value=mock_pp_group),
            # Patch at import locations in DeepSeek V2 model
            patch("sglang.srt.models.deepseek_v2.get_attention_tp_rank", return_value=self.attn_tp_rank),
            patch("sglang.srt.models.deepseek_v2.get_attention_tp_size", return_value=self.attn_tp_size),
            patch("sglang.srt.models.deepseek_v2.get_tensor_model_parallel_world_size", return_value=self.tp_size),
            patch("sglang.srt.models.deepseek_v2.get_pp_group", return_value=mock_pp_group),
            patch("sglang.srt.models.deepseek_v2.get_moe_expert_parallel_world_size", return_value=self.ep_size),
            # Patch moe layers
            patch(
                "sglang.srt.layers.moe.fused_moe_triton.layer.get_moe_expert_parallel_rank", return_value=self.ep_rank
            ),
            patch(
                "sglang.srt.layers.moe.fused_moe_triton.layer.get_moe_expert_parallel_world_size",
                return_value=self.ep_size,
            ),
            patch("sglang.srt.layers.moe.fused_moe_triton.layer.get_tp_group", return_value=mock_group),
            patch(
                "sglang.srt.layers.moe.fused_moe_triton.layer.get_moe_tensor_parallel_rank",
                return_value=self.moe_tp_rank,
            ),
            patch(
                "sglang.srt.layers.moe.fused_moe_triton.layer.get_moe_tensor_parallel_world_size",
                return_value=self.moe_tp_size,
            ),
            # Patch at import locations in MoE token dispatcher
            patch(
                "sglang.srt.layers.moe.token_dispatcher.standard.get_moe_expert_parallel_rank",
                return_value=self.ep_rank,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.standard.get_moe_expert_parallel_world_size",
                return_value=self.ep_size,
            ),
            patch("sglang.srt.layers.moe.token_dispatcher.standard.get_tp_group", return_value=mock_group),
            # Also patch in distributed module where get_tensor_model_parallel_rank may be imported
            patch("sglang.srt.distributed.get_tensor_model_parallel_rank", return_value=self.tp_rank),
            patch("sglang.srt.distributed.get_tensor_model_parallel_world_size", return_value=self.tp_size),
            patch("sglang.srt.distributed.get_moe_expert_parallel_world_size", return_value=self.ep_size),
        ]

        # Start all patches
        for p in self._patches:
            p.start()

        logger.info(
            f"[MockDist] Activated: TP={self.tp_rank}/{self.tp_size}, "
            f"PP={self.pp_rank}/{self.pp_size}, AttnTP={self.attn_tp_rank}/{self.attn_tp_size}"
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop all patches and restore original functions."""
        for p in self._patches:
            p.stop()
        sglang_server_args._global_server_args = None
        self._patches.clear()
        logger.info("[MockDist] Deactivated")
        return False  # Don't suppress exceptions
