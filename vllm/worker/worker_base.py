# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import inspect
import os
import statistics
import threading
import time
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import cloudpickle
import torch
import torch.nn as nn

from vllm.config import (ObservabilityConfig, VllmConfig,
                         set_current_vllm_config)
from vllm.distributed import broadcast_tensor_dict, get_pp_group, get_tp_group
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import ExecuteModelRequest, IntermediateTensors
from vllm.utils import (enable_trace_function_call_for_thread,
                        resolve_obj_by_qualname, run_method,
                        update_environment_variables,
                        warn_for_unimplemented_methods)
from vllm.worker.model_runner_base import (BroadcastableModelInput,
                                           ModelRunnerBase,
                                           ModelRunnerInputBase)

logger = init_logger(__name__)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


def _percentile(sorted_list: List[float], pct: float) -> float:
    # pct: 0~100
    if not sorted_list:
        return float("nan")
    k = (len(sorted_list) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_list) - 1)
    if f == c:
        return sorted_list[f]
    return sorted_list[f] * (c - k) + sorted_list[c] * (k - f)


@warn_for_unimplemented_methods
class WorkerBase:
    """Worker interface that allows vLLM to cleanly separate implementations for
    different hardware. Also abstracts control plane communication, e.g., to
    communicate request metadata to other workers.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config
        self.kv_transfer_config = vllm_config.kv_transfer_config
        self.compilation_config = vllm_config.compilation_config
        from vllm.platforms import current_platform
        self.current_platform = current_platform

    def init_device(self) -> None:
        """Initialize device state, such as loading the model or other on-device
        memory allocations.
        """
        raise NotImplementedError

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the KV cache with the given size in blocks.
        """
        raise NotImplementedError

    def get_model(self) -> nn.Module:
        raise NotImplementedError

    def load_model(self) -> None:
        """Load model onto target device."""
        raise NotImplementedError

    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> Optional[List[SamplerOutput]]:
        raise NotImplementedError

    async def execute_model_async(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
    ) -> Optional[List[SamplerOutput]]:
        """Async model execution entry point used by online engines."""
        return self.execute_model(execute_model_req)

    def start_worker_execution_loop(self) -> None:
        """Execute model loop in parallel worker.

        You can stop the loop by executing a driver worker with an empty output.
        See `stop_remote_worker_execution_loop` for more details.
        """
        with self.current_platform.inference_mode():
            while True:
                output = self.execute_model(execute_model_req=None)
                if output is None:
                    return None

    async def start_worker_execution_loop_async(self) -> None:
        """Async counterpart for online distributed worker loops."""
        with self.current_platform.inference_mode():
            while True:
                output = await self.execute_model_async(
                    execute_model_req=None)
                if output is None:
                    return None

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available blocks for the GPU KV cache and
        swappable CPU KV cache.

        The implementation may run profiling or other heuristics to determine
        the size of caches.

        Returns a Tuple[num_gpu_blocks, num_cpu_blocks], where num_gpu_blocks
        are blocks that are "active" on the device and can be appended to.
        num_cpu_blocks refers to "swapped" blocks in CPU memory and cannot be
        appended to.
        """
        raise NotImplementedError

    def get_cache_block_size_bytes(self) -> int:
        """Return the size of a single cache block, in bytes. Used in
        speculative decoding.
        """
        raise NotImplementedError

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def list_loras(self) -> Set[int]:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size from model configuration."""
        return self.model_config.get_vocab_size()


class DelegateWorkerBase(WorkerBase):
    """
    A class that delegates all methods to another WorkerBase instance. This is
    useful for creating a WorkerBase that wraps another WorkerBase instance,
    e.g. speculative decoding.
    """
    worker: WorkerBase

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        vllm_config: VllmConfig = kwargs.get("vllm_config")
        cls = resolve_obj_by_qualname(vllm_config.parallel_config.worker_cls)
        self.worker = cls(*args, **kwargs)

    def init_device(self) -> None:
        self.worker.init_device()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        return self.worker.determine_num_available_blocks()

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        self.worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)

    def load_model(self) -> None:
        """Load model onto target device."""
        self.worker.load_model()

    def get_model(self) -> nn.Module:
        return self.worker.get_model()

    def set_p_pool_preemption(self, enabled: bool) -> bool:
        return self.worker.set_p_pool_preemption(enabled)

    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> Optional[List[SamplerOutput]]:
        return self.worker.execute_model(execute_model_req)

    async def execute_model_async(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
    ) -> Optional[List[SamplerOutput]]:
        return await self.worker.execute_model_async(execute_model_req)

    def get_cache_block_size_bytes(self) -> int:
        return self.worker.get_cache_block_size_bytes()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.worker.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.worker.remove_lora(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        return self.worker.pin_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.worker.list_loras()

    def __getattr__(self, attr):
        return getattr(self.worker, attr)


class LoRANotSupportedWorkerBase(WorkerBase):
    """Partial implementation of WorkerBase that raises exceptions when LoRA
    methods are invoked.
    """

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise ValueError(f"{type(self)} does not support LoRA")

    def remove_lora(self, lora_id: int) -> bool:
        raise ValueError(f"{type(self)} does not support LoRA")

    def pin_lora(self, lora_id: int) -> bool:
        raise ValueError(f"{type(self)} does not support LoRA")

    def list_loras(self) -> Set[int]:
        raise ValueError(f"{type(self)} does not support LoRA")


@dataclasses.dataclass(frozen=True)
class WorkerInput:
    """Local inputs to each worker. May contain device-specific data. These
    fields should be broadcastable to other workers.
    """

    num_seq_groups: Optional[int] = None
    blocks_to_swap_in: Optional[torch.Tensor] = None
    blocks_to_swap_out: Optional[torch.Tensor] = None
    blocks_to_copy: Optional[torch.Tensor] = None
    virtual_engine: int = 0
    num_steps: int = 1

    @classmethod
    def from_broadcasted_tensor_dict(
        cls: Type["WorkerInput"],
        tensor_dict: Dict[str, Any],
    ) -> "WorkerInput":
        """
        Pop fields from the given tensor_dict and populate a new instance of
        WorkerInput.
        """
        return cls(
            num_seq_groups=tensor_dict.pop("num_seq_groups"),
            blocks_to_swap_in=tensor_dict.pop("blocks_to_swap_in"),
            blocks_to_swap_out=tensor_dict.pop("blocks_to_swap_out"),
            blocks_to_copy=tensor_dict.pop("blocks_to_copy"),
            virtual_engine=tensor_dict["virtual_engine"],
            num_steps=tensor_dict.pop("num_steps"),
        )

    def as_broadcastable_tensor_dict(
            self) -> Dict[str, Union[int, torch.Tensor]]:
        """
        Extract broadcastable fields.
        """
        tensor_dict = {
            "num_seq_groups": self.num_seq_groups,
            "blocks_to_swap_in": self.blocks_to_swap_in,
            "blocks_to_swap_out": self.blocks_to_swap_out,
            "blocks_to_copy": self.blocks_to_copy,
            "virtual_engine": self.virtual_engine,
            "num_steps": self.num_steps,
        }

        return tensor_dict


class LocalOrDistributedWorkerBase(WorkerBase):
    """
    Partial implementation of WorkerBase that has a default `execute_model`
    definition to perform metadata transfer between workers when in distributed
    mode. Subclasses of this interface should use model runners that inherit
    from ModelRunnerBase, and should only need to implement worker-local logic.
    If custom control plane logic is needed to transfer metadata, or if the
    model runner cannot inherit from ModelRunnerBase, use WorkerBase instead.
    """
    is_driver_worker: bool
    model_runner: ModelRunnerBase
    observability_config: Optional[ObservabilityConfig] = None

    @property
    @abstractmethod
    def do_metadata_broadcast(self) -> bool:
        """
        Used by the default `execute_model` to check whether broadcast is
        needed to transfer request inputs from the driver worker to other
        workers in the TP group. If WorkerBase subclass only supports
        single-worker execution, then this method should return False.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def kv_cache(self) -> Optional[List[List[torch.Tensor]]]:
        """
        Gets the list of kv caches to pass to the worker's model runner. Each
        element in the list is a kv cache corresponding to a particular virtual
        engine (PP stream). Used by the default `execute_model`. If the worker's
        model runner does not follow the ModelRunnerBase interface, then inherit
        from WorkerBase instead.
        """
        raise NotImplementedError

    @abstractmethod
    def prepare_worker_input(
            self, execute_model_req: ExecuteModelRequest) -> WorkerInput:
        """
        Prepare the inputs to WorkerBase.execute_worker from an execution
        request. This method may move data to the worker's local device. It is
        not allowed to communicate with other workers or devices.
        """
        raise NotImplementedError

    @abstractmethod
    def execute_worker(self, worker_input: WorkerInput) -> None:
        """
        Process an execution request.
        """
        raise NotImplementedError

    def _get_worker_input_from_broadcast(
        self
    ) -> Optional[Tuple[BroadcastableModelInput, WorkerInput, Dict[
            str, torch.Tensor]]]:
        """ Get the worker input from the broadcasted tensor dict. """
        assert self.do_metadata_broadcast
        assert not self.is_driver_worker
        broadcast_data = broadcast_tensor_dict(src=0)
        if not broadcast_data:
            return None

        worker_input = WorkerInput.from_broadcasted_tensor_dict(broadcast_data)
        model_input = (
            self.model_runner.make_model_input_from_broadcasted_tensor_dict(
                broadcast_data))

        kwargs = extract_previous_hidden_states(broadcast_data)

        return model_input, worker_input, kwargs

    def _get_driver_input_and_broadcast(
        self, execute_model_req: ExecuteModelRequest
    ) -> Tuple[BroadcastableModelInput, WorkerInput, Dict[str, torch.Tensor]]:
        """ Get the driver input and broadcast it to other workers.  """
        assert self.is_driver_worker

        worker_input: WorkerInput = self.prepare_worker_input(
            execute_model_req=execute_model_req)
        model_input: ModelRunnerInputBase = (
            self.model_runner.prepare_model_input(
                execute_model_req.seq_group_metadata_list,
                execute_model_req.virtual_engine,
                execute_model_req.finished_requests_ids))

        kwargs = extract_previous_hidden_states(execute_model_req)

        if self.do_metadata_broadcast:
            broadcast_data = worker_input.as_broadcastable_tensor_dict()
            broadcast_data.update(model_input.as_broadcastable_tensor_dict())
            broadcast_data.update(kwargs)
            broadcast_tensor_dict(broadcast_data, src=0)

        if execute_model_req.async_callback:
            model_input = dataclasses.replace(  # type: ignore
                model_input,
                async_callback=execute_model_req.async_callback)

        return model_input, worker_input, kwargs

    def prepare_input(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> Optional[Tuple[BroadcastableModelInput, WorkerInput, Dict[
            str, torch.Tensor]]]:
        """
        Prepare the inputs to ModelRunner and workers.
        """
        if self.is_driver_worker:
            if execute_model_req is None:
                if self.do_metadata_broadcast:
                    # This signals that there's no more requests to process for
                    # now. All workers are running infinite loop with
                    # broadcast_tensor_dict, and it stops the loop when the
                    # driver broadcasts an empty input. Send an empty input to
                    # notify all other workers to stop their execution loop.
                    broadcast_tensor_dict({}, src=0)
                return None
            return self._get_driver_input_and_broadcast(execute_model_req)
        else:
            return self._get_worker_input_from_broadcast()

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    def set_p_pool_preemption(self, enabled: bool) -> bool:
        model = self.get_model()
        inner_model = getattr(model, "model", None)
        if inner_model is None or not hasattr(inner_model, "preemption_event"):
            return False

        if inner_model.preemption_event is None:
            inner_model.preemption_event = threading.Event()

        if enabled:
            inner_model.preemption_event.set()
        else:
            inner_model.preemption_event.clear()
            if hasattr(inner_model, "preempted"):
                inner_model.preempted = False
        return True

    # Sentinel key used to signal preemption across PP stages.
    _PP_PREEMPTION_KEY = "__pp_preempted__"

    def _send_pp_preemption_signal(self) -> None:
        """Send a poison-pill tensor dict to the next PP stage so it
        unblocks from recv_tensor_dict and knows to abort."""
        poison = {self._PP_PREEMPTION_KEY: torch.tensor([1.0])}
        get_pp_group().send_tensor_dict(poison,
                                        all_gather_group=get_tp_group())

    def _is_pp_preemption_signal(
        self, tensor_dict: dict,
    ) -> bool:
        """Check whether a received tensor dict is a preemption signal."""
        return self._PP_PREEMPTION_KEY in tensor_dict

    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
    ) -> Optional[List[SamplerOutput]]:
        """Executes at least one model step on the given sequences, unless no
        sequences are provided."""
        start_time = time.perf_counter()

        inputs = self.prepare_input(execute_model_req)
        if inputs is None:
            return None

        model_input, worker_input, kwargs = inputs
        num_steps = worker_input.num_steps
        if execute_model_req is not None and execute_model_req.spec_step_idx:
            kwargs["spec_step_idx"] = execute_model_req.spec_step_idx

        self.execute_worker(worker_input)

        # If there is no input, we don't need to execute the model.
        if worker_input.num_seq_groups == 0:
            return []

        intermediate_tensors = None
        orig_model_execute_time = 0.0
        if not get_pp_group().is_first_rank:
            received = get_pp_group().recv_tensor_dict(
                all_gather_group=get_tp_group())
            # Check for preemption poison-pill from the upstream PP stage.
            if self._is_pp_preemption_signal(received):
                # Upstream was preempted. Propagate the signal downstream
                # (if we are not the last rank) so subsequent stages also
                # unblock, then return a non-None sentinel so that the
                # worker execution loop does NOT exit (None = stop loop).
                if not get_pp_group().is_last_rank:
                    self._send_pp_preemption_signal()
                # Return [None] keeps the execution loop alive; the driver
                # (first rank) will return real None to the engine.
                return [None]
            intermediate_tensors = IntermediateTensors(received)
            if (self.observability_config is not None
                    and self.observability_config.collect_model_execute_time):
                orig_model_execute_time = intermediate_tensors.tensors.get(
                    "model_execute_time", torch.tensor(0)).item()

        output = self.model_runner.execute_model(
            model_input=model_input,
            kv_caches=self.kv_cache[worker_input.virtual_engine]
            if self.kv_cache is not None else None,
            intermediate_tensors=intermediate_tensors,
            num_steps=num_steps,
            **kwargs,
        )

        model_execute_time = time.perf_counter() - start_time
        # ``None`` from the model runner means this step was preempted.  On a
        # non-driver TP rank it must not escape as ``None`` because the remote
        # execution loop reserves that value for the empty-broadcast shutdown
        # signal.  Keep the rank in the loop for the next driver step.
        if output is None and not self.is_driver_worker:
            return [None]
        if not get_pp_group().is_last_rank:
            # If model_runner returned None, the forward was preempted
            # mid-layer.  Send a preemption signal to unblock downstream
            # stages instead of asserting on IntermediateTensors.
            if output is None:
                self._send_pp_preemption_signal()
                return None
            assert isinstance(output, IntermediateTensors)
            if (self.observability_config is not None
                    and self.observability_config.collect_model_execute_time):
                output.tensors["model_execute_time"] = torch.tensor(
                    model_execute_time + orig_model_execute_time)
            get_pp_group().send_tensor_dict(output.tensors,
                                            all_gather_group=get_tp_group())
            return [None]
        if (self.observability_config is not None
                and self.observability_config.collect_model_execute_time
                and output is not None):
            for o in output:
                o.model_execute_time = (orig_model_execute_time +
                                        model_execute_time)

        # output is List[SamplerOutput]
        return output

    async def execute_model_async(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
    ) -> Optional[List[SamplerOutput]]:
        """Execute one online model step without a sync-to-async bridge."""
        start_time = time.perf_counter()

        inputs = self.prepare_input(execute_model_req)
        if inputs is None:
            return None

        model_input, worker_input, kwargs = inputs
        num_steps = worker_input.num_steps
        if execute_model_req is not None and execute_model_req.spec_step_idx:
            kwargs["spec_step_idx"] = execute_model_req.spec_step_idx

        self.execute_worker(worker_input)
        if worker_input.num_seq_groups == 0:
            return []

        intermediate_tensors = None
        orig_model_execute_time = 0.0
        if not get_pp_group().is_first_rank:
            received = get_pp_group().recv_tensor_dict(
                all_gather_group=get_tp_group())
            if self._is_pp_preemption_signal(received):
                if not get_pp_group().is_last_rank:
                    self._send_pp_preemption_signal()
                return [None]
            intermediate_tensors = IntermediateTensors(received)
            if (self.observability_config is not None
                    and self.observability_config.collect_model_execute_time):
                orig_model_execute_time = intermediate_tensors.tensors.get(
                    "model_execute_time", torch.tensor(0)).item()

        output = await self.model_runner.execute_model_async(
            model_input=model_input,
            kv_caches=self.kv_cache[worker_input.virtual_engine]
            if self.kv_cache is not None else None,
            intermediate_tensors=intermediate_tensors,
            num_steps=num_steps,
            **kwargs,
        )

        model_execute_time = time.perf_counter() - start_time
        if output is None and not self.is_driver_worker:
            return [None]
        if not get_pp_group().is_last_rank:
            if output is None:
                self._send_pp_preemption_signal()
                return None
            assert isinstance(output, IntermediateTensors)
            if (self.observability_config is not None
                    and self.observability_config.collect_model_execute_time):
                output.tensors["model_execute_time"] = torch.tensor(
                    model_execute_time + orig_model_execute_time)
            get_pp_group().send_tensor_dict(output.tensors,
                                            all_gather_group=get_tp_group())
            return [None]
        if (self.observability_config is not None
                and self.observability_config.collect_model_execute_time
                and output is not None):
            for item in output:
                item.model_execute_time = (orig_model_execute_time +
                                           model_execute_time)
        return output

    def _execute_model_spmd(
        self,
        execute_model_req: ExecuteModelRequest,
        intermediate_tensors: Optional[IntermediateTensors] = None
    ) -> Optional[List[SamplerOutput]]:
        """
        Execute model in Single Program Multiple Data (SPMD) fashion.
        All workers take the same request, prepare the input and
        execute the model.
        """
        assert execute_model_req is not None, (
            "_execute_model_spmd() requires each worker to take in an "
            "ExecuteModelRequest")
        worker_input: WorkerInput = self.prepare_worker_input(
            execute_model_req=execute_model_req)
        model_input: ModelRunnerInputBase = (
            self.model_runner.prepare_model_input(
                execute_model_req.seq_group_metadata_list))

        self.execute_worker(worker_input)

        # If there is no input, we don't need to execute the model.
        if worker_input.num_seq_groups == 0:
            return []

        kwargs = extract_previous_hidden_states(execute_model_req)

        return self.model_runner.execute_model(
            model_input=model_input,
            kv_caches=self.kv_cache[worker_input.virtual_engine]
            if self.kv_cache is not None else None,
            intermediate_tensors=intermediate_tensors,
            **kwargs,
        )

    def _maybe_get_block_size(self) -> int:
        # 兼容不同版本字段名：尽量从 cache_engine / cache_config 拿
        if hasattr(self, "cache_engine") and hasattr(self.cache_engine, "block_size"):
            return int(self.cache_engine.block_size)
        if hasattr(self, "cache_config") and hasattr(self.cache_config, "block_size"):
            return int(self.cache_config.block_size)
        # 实在没有就给个常见默认值（vLLM 常见是 16）
        return 16

    @torch.inference_mode()
    def profile_attn_linear(
        self,
        *,
        warmup: int = 3,
        repeat: int = 3,
        num_layer_to_run: int = 5,
    ) -> Optional[Dict[str, Any]]:

        block_size = self.cache_config.block_size
        import math
        import statistics
        GPU_BLOCKS = self.cache_config.num_gpu_blocks     
        UTIL = 0.65           

        def blocks_per_req(req_len: int, block_size: int) -> int:
            return (req_len + block_size - 1) // block_size

        def _run_one_attn(req_len: int, num_req: int, is_prompt: bool) -> float:
            execute_model_req = _build_execute_model_req_attn(
                req_len=req_len, num_req=num_req, is_prompt=is_prompt,
                block_size=block_size, virtual_engine=0
            )

            inputs = self.prepare_input(execute_model_req)
            if inputs is None:
                return float("nan")

            model_input, worker_input, kwargs = inputs
            self.execute_worker(worker_input)

            # print(f"execute_model_req seq group metadata list length: {len(execute_model_req.seq_group_metadata_list)}")
            # print(f"model_input input_tokens shape: {model_input.input_tokens.shape}")

            # PP：非首段需要 recv intermediate
            intermediate_tensors = None
            if not get_pp_group().is_first_rank:
                intermediate_tensors = IntermediateTensors(
                    get_pp_group().recv_tensor_dict(all_gather_group=get_tp_group())
                )

            try:
                elapsed_ms = self.model_runner.execute_model_attn_only(
                    model_input=model_input,
                    kv_caches=self.kv_cache[worker_input.virtual_engine] if self.kv_cache is not None else None,
                    intermediate_tensors=intermediate_tensors,
                    num_steps=1,
                    num_layer_to_run=num_layer_to_run,
                    **kwargs,
                )
                return elapsed_ms
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                print(f"  [SKIP] OOM/error in attn: {e}")
                torch.cuda.empty_cache()
                return float("inf")

        def _run_one_linear(num_token: int) -> float:
            # print(f"info of iteration: num_token={num_token}")
            execute_model_req = _build_execute_model_req_linear(
                num_token=num_token, block_size=block_size, virtual_engine=0
            )
            inputs = self.prepare_input(execute_model_req)
            if inputs is None:
                return float("nan")

            model_input, worker_input, kwargs = inputs
            self.execute_worker(worker_input)

            intermediate_tensors = None
            if not get_pp_group().is_first_rank:
                intermediate_tensors = IntermediateTensors(
                    get_pp_group().recv_tensor_dict(all_gather_group=get_tp_group())
                )

            try:
                elapsed_ms = self.model_runner.execute_model_linear_only(
                    model_input=model_input,
                    kv_caches=self.kv_cache[worker_input.virtual_engine] if self.kv_cache is not None else None,
                    intermediate_tensors=intermediate_tensors,
                    num_steps=1,
                    num_layer_to_run=num_layer_to_run,
                    **kwargs,
                )
                return elapsed_ms
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                print(f"  [SKIP] OOM/error in linear: {e}")
                torch.cuda.empty_cache()
                return float("inf")

        warmup_cases = [
            ("prefill", 128, 4, True),
            ("prefill", 2048, 12, True),
            ("decode", 128, 16, False),
            ("decode", 2048, 12, False),
        ]
        for _ in range(warmup):
            for kind, rl, nr, isp in warmup_cases:
                if kind in ("prefill", "decode"):
                    _ = _run_one_attn(rl, nr, isp)
            _ = _run_one_linear(256)

        results: Dict[str, Any] = {
            "meta": {
                "warmup": warmup,
                "repeat": repeat,
                "block_size": block_size,
                "rank": getattr(self, "rank", None),
                "is_driver": getattr(self, "is_driver_worker", None),
            },
            "prefill_attn_ms": [],  # list of dict rows
            "decode_attn_ms": [],
            "decode_full_layer_ms": [],  # decode full-layer (via forward_with_time_record)
            "full_layer_ms": [],  # full-layer calibration rows with layer/attn/linear split
            "linear_ms": [],
        }
        
        max_req_len = block_size * GPU_BLOCKS

        def _summarize_times(times):
            # 过滤 inf/nan
            valid = [t for t in times if isinstance(t, (int, float)) and math.isfinite(t)]
            if len(valid) == 0:
                return {
                    "oom": True,
                    "valid_count": 0,
                    "ms_mean": float("inf"),
                    "ms_median": float("inf"),
                    "ms_min": float("inf"),
                    "ms_max": float("inf"),
                    "each_ms": times,
                }

            return {
                "oom": False,
                "valid_count": len(valid),
                "ms_mean": float(sum(valid) / len(valid)),
                "ms_median": float(statistics.median(valid)),
                "ms_min": float(min(valid)),
                "ms_max": float(max(valid)),
                "each_ms": times,
            }

        # -------------------------
        # Prefill attn
        # -------------------------
        # prefill_req_lens = [1408]
        # prefill_num_reqs = [1, 2, 4, 8, 16, 32, 64, 128]
        prefill_req_lens = [i for i in range(64, 4097, 64)]
        prefill_num_reqs = list(range(1, 17, 1)) + list(range(20, 65, 4))
        max_prefill_tokens = 16384 
        min_prefill_tokens = 128

        for req_len in prefill_req_lens:
            for num_req in prefill_num_reqs:
                if req_len * num_req > max_prefill_tokens or req_len*num_req < min_prefill_tokens:
                    continue  
                times = []
                for _ in range(repeat): 
                    t = _run_one_attn(req_len, num_req, True)
                    times.append(t / num_layer_to_run)

                row = {"req_len": req_len, "num_req": num_req}
                row.update(_summarize_times(times))
                results["prefill_attn_ms"].append(row)

        # -------------------------
        # Decode attn
        # -------------------------
        # decode_req_lens = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
        # decode_num_reqs = [16, 32, 64, 128, 256, 512]
        decode_req_lens = list(range(256, 3073, 128))
        decode_num_reqs = list(range(1, 33, 1)) + list(range(36, 129, 4)) + list(range(136, 257, 8))
        max_blocks = int(GPU_BLOCKS * UTIL)

        for req_len in decode_req_lens:
            for num_req in decode_num_reqs:
                if blocks_per_req(req_len, block_size) * num_req > max_blocks:
                    continue 
                times = []
                for _ in range(repeat):
                    t = _run_one_attn(req_len, num_req, False)
                    times.append(t / num_layer_to_run)

                row = {"req_len": req_len, "num_req": num_req}
                row.update(_summarize_times(times))
                results["decode_attn_ms"].append(row)

        # -------------------------
        # Full-layer calibration (via forward_with_time_record)
        # -------------------------
        from vllm.worker.model_runner import set_forward_context

        def _run_full_forward(req_len, num_req, is_prompt):
            try:
                execute_model_req = _build_execute_model_req_attn(
                    req_len=req_len, num_req=num_req, is_prompt=is_prompt,
                    block_size=block_size, virtual_engine=0
                )
                inputs = self.prepare_input(execute_model_req)
                if inputs is None:
                    return float("nan"), float("nan"), float("nan")
                model_input, worker_input, kwargs = inputs
                self.execute_worker(worker_input)
                intermediate_tensors = None
                if not get_pp_group().is_first_rank:
                    intermediate_tensors = IntermediateTensors(
                        get_pp_group().recv_tensor_dict(all_gather_group=get_tp_group())
                    )
                self.model_runner.attn_state.begin_forward(model_input)
                layer_events = []
                attn_events = []
                with set_forward_context(model_input.attn_metadata, self.model_runner.vllm_config, model_input.virtual_engine):
                    self.model_runner.model.model.forward_with_time_record(
                        input_ids=model_input.input_tokens,
                        positions=model_input.input_positions,
                        intermediate_tensors=intermediate_tensors,
                        inputs_embeds=model_input.inputs_embeds,
                        layer_events=layer_events,
                        attn_events=attn_events,
                    )
                torch.cuda.synchronize()
                n_layers = min(len(layer_events), len(attn_events))
                if n_layers == 0:
                    return float("nan"), float("nan"), float("nan")
                layer_ms = sum(
                    s.elapsed_time(e) for s, e in layer_events[:n_layers]
                ) / n_layers
                attn_ms = sum(
                    s.elapsed_time(e) for s, e in attn_events[:n_layers]
                ) / n_layers
                return layer_ms, attn_ms, layer_ms - attn_ms
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                print(f"  [SKIP] OOM/error in full forward: {e}")
                torch.cuda.empty_cache()
                return float("inf"), float("inf"), float("inf")

        full_layer_configs = []
        prefill_full_req_lens = [128, 256, 512, 1024, 2048]
        prefill_full_num_reqs = [1, 2, 4, 8, 16]
        for req_len in prefill_full_req_lens:
            for num_req in prefill_full_num_reqs:
                if req_len * num_req <= max_prefill_tokens:
                    full_layer_configs.append(("prefill", req_len, num_req))

        decode_full_req_lens = list(range(256, 3073, 256))
        decode_full_num_reqs = (
            list(range(1, 17, 1))
            + list(range(20, 129, 8))
            + list(range(136, 257, 16))
        )
        for req_len in decode_full_req_lens:
            for num_req in decode_full_num_reqs:
                if blocks_per_req(req_len, block_size) * num_req <= max_blocks:
                    full_layer_configs.append(("decode", req_len, num_req))

        for mode, req_len, num_req in full_layer_configs:
            layer_times = []
            attn_times = []
            linear_times = []
            for _ in range(repeat):
                layer_ms, attn_ms, linear_ms = _run_full_forward(
                    req_len, num_req, mode == "prefill")
                layer_times.append(layer_ms)
                attn_times.append(attn_ms)
                linear_times.append(linear_ms)

            layer_summary = _summarize_times(layer_times)
            attn_summary = _summarize_times(attn_times)
            linear_summary = _summarize_times(linear_times)
            row = {
                "mode": mode,
                "req_len": req_len,
                "num_req": num_req,
                "layer_ms_median": layer_summary["ms_median"],
                "attn_ms_median": attn_summary["ms_median"],
                "linear_ms_median": linear_summary["ms_median"],
                "layer_each_ms": layer_summary["each_ms"],
                "attn_each_ms": attn_summary["each_ms"],
                "linear_each_ms": linear_summary["each_ms"],
            }
            results["full_layer_ms"].append(row)

            if mode == "decode":
                legacy_row = {"req_len": req_len, "num_req": num_req}
                legacy_row.update(layer_summary)
                results["decode_full_layer_ms"].append(legacy_row)

        # -------------------------
        # Linear
        # -------------------------
        for num_token in range(4,2049,4):
            times = []
            for _ in range(repeat):
                t = _run_one_linear(num_token)
                times.append(t / num_layer_to_run)

            row = {"num_token": num_token}
            row.update(_summarize_times(times))
            results["linear_ms"].append(row)

        
        import json
        from pathlib import Path

        def save_results_json(results: dict, out_path: str):
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        if self.is_driver_worker:
            save_results_json(results, "profile_bench_results.json")

        if not self.is_driver_worker:
            return None
            
        return results



    @torch.inference_mode()
    def profile_full_layer(
        self,
        *,
        warmup: int = 3,
        repeat: int = 5,
        num_layer_to_run: int = 5,
        configs: list = None,
    ) -> Optional[Dict[str, Any]]:
        """Profile full layer forward (layer_time = attn + linear) for validation."""
        block_size = self.cache_config.block_size
        import math
        import statistics

        def _run_full_forward(req_len, num_req, is_prompt):
            execute_model_req = _build_execute_model_req_attn(
                req_len=req_len, num_req=num_req, is_prompt=is_prompt,
                block_size=block_size, virtual_engine=0
            )
            inputs = self.prepare_input(execute_model_req)
            if inputs is None:
                return float("nan"), float("nan")
            model_input, worker_input, kwargs = inputs
            self.execute_worker(worker_input)

            intermediate_tensors = None
            if not get_pp_group().is_first_rank:
                intermediate_tensors = IntermediateTensors(
                    get_pp_group().recv_tensor_dict(all_gather_group=get_tp_group())
                )

            from vllm.worker.model_runner import set_forward_context
            self.model_runner.attn_state.begin_forward(model_input)
            layer_events = []
            attn_events = []
            with set_forward_context(model_input.attn_metadata, self.model_runner.vllm_config, model_input.virtual_engine):
                self.model_runner.model.model.forward_with_time_record(
                    input_ids=model_input.input_tokens,
                    positions=model_input.input_positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=model_input.inputs_embeds,
                    layer_events=layer_events,
                    attn_events=attn_events,
                )
            torch.cuda.synchronize()

            n_layers = min(len(layer_events), len(attn_events))
            if n_layers == 0:
                return float("nan"), float("nan")
            layer_ms = sum(s.elapsed_time(e) for s, e in layer_events[:n_layers]) / n_layers
            attn_ms = sum(s.elapsed_time(e) for s, e in attn_events[:n_layers]) / n_layers
            return layer_ms, attn_ms

        # Warmup
        for _ in range(warmup):
            _run_full_forward(256, 4, True)
            _run_full_forward(256, 16, False)

        if configs is None:
            configs = [
                ("prefill", 128, 1), ("prefill", 256, 1), ("prefill", 512, 1),
                ("prefill", 1024, 1), ("prefill", 2048, 1),
                ("prefill", 128, 4), ("prefill", 256, 4), ("prefill", 512, 4),
                ("prefill", 1024, 4),
                ("prefill", 128, 8), ("prefill", 256, 8), ("prefill", 512, 8),
                ("prefill", 128, 16), ("prefill", 256, 16),
                ("decode", 256, 1), ("decode", 512, 1), ("decode", 1024, 1),
                ("decode", 2048, 1), ("decode", 4096, 1),
                ("decode", 256, 8), ("decode", 512, 8), ("decode", 1024, 8),
                ("decode", 2048, 8),
                ("decode", 256, 32), ("decode", 512, 32), ("decode", 1024, 32),
                ("decode", 256, 64), ("decode", 512, 64),
                ("decode", 256, 128), ("decode", 512, 128),
            ]

        results = []
        for mode, req_len, num_req in configs:
            is_prompt = (mode == "prefill")
            layer_times = []
            attn_times = []
            for _ in range(repeat):
                lt, at = _run_full_forward(req_len, num_req, is_prompt)
                if math.isfinite(lt):
                    layer_times.append(lt)
                if math.isfinite(at):
                    attn_times.append(at)

            med_layer = statistics.median(layer_times) if layer_times else float("nan")
            med_attn = statistics.median(attn_times) if attn_times else float("nan")
            results.append({
                "mode": mode, "req_len": req_len, "num_req": num_req,
                "layer_ms": med_layer, "attn_ms": med_attn,
                "linear_ms": med_layer - med_attn if math.isfinite(med_layer) and math.isfinite(med_attn) else float("nan"),
            })
            print(f"  [{mode}] req_len={req_len}, num_req={num_req}: layer={med_layer:.4f}ms, attn={med_attn:.4f}ms")

        if not self.is_driver_worker:
            return None
        return results


    def profile_full_iteration(
        self,
        *,
        warmup: int = 3,
        repeat: int = 5,
        configs: list = None,
    ):
        """Profile full iteration (all layers via execute_model) wall-clock time."""
        import math, statistics
        block_size = self.cache_config.block_size
        GPU_BLOCKS = self.cache_config.num_gpu_blocks
        max_blocks = int(GPU_BLOCKS * 0.65)

        def blocks_per_req(req_len, bs):
            return math.ceil((req_len + 1) / bs)

        def _run_iteration(req_len, num_req, is_prompt):
            if blocks_per_req(req_len, block_size) * num_req > max_blocks:
                return None
            execute_model_req = _build_execute_model_req_attn(
                req_len=req_len, num_req=num_req, is_prompt=is_prompt,
                block_size=block_size, virtual_engine=0
            )
            inputs = self.prepare_input(execute_model_req)
            if inputs is None:
                return None
            model_input, worker_input, kwargs = inputs
            self.execute_worker(worker_input)
            intermediate_tensors = None
            if not get_pp_group().is_first_rank:
                intermediate_tensors = IntermediateTensors(
                    get_pp_group().recv_tensor_dict(all_gather_group=get_tp_group()))
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self.model_runner.execute_model(
                model_input=model_input,
                kv_caches=self.kv_cache[worker_input.virtual_engine] if self.kv_cache is not None else None,
                intermediate_tensors=intermediate_tensors,
                num_steps=1,
                **kwargs,
            )
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end)

        if configs is None:
            configs = [
                ("prefill", 128, 1), ("prefill", 256, 1), ("prefill", 512, 1),
                ("prefill", 1024, 1), ("prefill", 2048, 1),
                ("prefill", 128, 4), ("prefill", 256, 4), ("prefill", 512, 4),
                ("prefill", 1024, 4),
                ("prefill", 128, 8), ("prefill", 256, 8), ("prefill", 512, 8),
                ("prefill", 128, 16), ("prefill", 256, 16),
                ("decode", 256, 1), ("decode", 512, 1), ("decode", 1024, 1),
                ("decode", 2048, 1),
                ("decode", 256, 4), ("decode", 512, 4), ("decode", 1024, 4),
                ("decode", 2048, 4),
                ("decode", 256, 8), ("decode", 512, 8), ("decode", 1024, 8),
                ("decode", 2048, 8),
                ("decode", 256, 16), ("decode", 512, 16), ("decode", 1024, 16),
                ("decode", 256, 32), ("decode", 512, 32), ("decode", 1024, 32),
                ("decode", 256, 64), ("decode", 512, 64),
                ("decode", 256, 128), ("decode", 512, 128),
            ]

        # Warmup
        for _ in range(warmup):
            _run_iteration(256, 4, True)
            _run_iteration(256, 16, False)

        results = []
        for mode, req_len, num_req in configs:
            is_prompt = (mode == "prefill")
            times = []
            for _ in range(repeat):
                t = _run_iteration(req_len, num_req, is_prompt)
                if t is not None:
                    times.append(t)
            if not times:
                continue
            med = statistics.median(times)
            results.append({"mode": mode, "req_len": req_len, "num_req": num_req, "iter_ms": med})
            print(f"  [{mode}] req_len={req_len}, B={num_req}: iter={med:.2f}ms")

        if not self.is_driver_worker:
            return None
        return results

    @torch.inference_mode()
    def profile_qwen2_module_times(
        self,
        *,
        warmup: int = 3,
        repeat: int = 5,
        warmup_per_config: int = 0,
        configs: list = None,
        profile_num_layers: int = None,
        profile_layer_start: int = 0,
        profile_kv_cache_fraction: float = 0.65,
    ) -> Optional[Dict[str, Any]]:
        """Profile Qwen2 module latencies using the regular forward path."""
        import math
        import statistics

        from vllm.worker.model_runner import set_forward_context

        block_size = self.cache_config.block_size
        gpu_blocks = self.cache_config.num_gpu_blocks
        max_blocks = int(gpu_blocks * float(profile_kv_cache_fraction))
        total_model_layers = getattr(
            getattr(self.model_runner.model, "config", None),
            "num_hidden_layers",
            None,
        )
        event_groups = {
            "stateless_linear": [
                "input_layernorm",
                "self_attn.qkv_proj",
                "self_attn.qkv_split",
                "self_attn.rotary_emb",
                "self_attn.o_proj",
                "post_attention_layernorm",
                "mlp.gate_up_proj",
                "mlp.act_fn",
                "mlp.down_proj",
            ],
            "stateful_attention": [
                "self_attn.attn",
            ],
        }

        def blocks_per_req(req_len, is_prompt):
            total_len = req_len if is_prompt else req_len + 1
            return math.ceil(total_len / block_size)

        def feature_values(mode, req_len, num_req):
            if mode == "prefill":
                n_tokens = req_len * num_req
                sum_context = num_req * req_len * (req_len + 1.0) / 2.0
            else:
                n_tokens = num_req
                sum_context = num_req * req_len
            return int(n_tokens), float(sum_context)

        def summarize(values):
            valid = [v for v in values if isinstance(v, (int, float))
                     and math.isfinite(v)]
            if not valid:
                return {
                    "valid_count": 0,
                    "mean": float("nan"),
                    "median": float("nan"),
                    "mid3": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                    "each": values,
                }
            sv = sorted(valid)
            lo = (len(sv) - 3) // 2
            mid = sv[lo:lo + 3] if len(sv) >= 3 else sv
            mid3 = float(sum(mid) / len(mid))
            return {
                "valid_count": len(valid),
                "mean": float(sum(valid) / len(valid)),
                "median": float(statistics.median(valid)),
                "mid3": mid3,
                "min": float(min(valid)),
                "max": float(max(valid)),
                "each": values,
            }

        def run_one(mode, req_len, num_req):
            is_prompt = mode == "prefill"
            if blocks_per_req(req_len, is_prompt) * num_req > max_blocks:
                return None

            execute_model_req = _build_execute_model_req_attn(
                req_len=req_len,
                num_req=num_req,
                is_prompt=is_prompt,
                block_size=block_size,
                virtual_engine=0,
            )
            inputs = self.prepare_input(execute_model_req)
            if inputs is None:
                return None
            model_input, worker_input, _ = inputs
            self.execute_worker(worker_input)

            # For profiling with profile_num_layers, PP communication is
            # skipped (model returns early without sending intermediate
            # tensors).  Each PP rank profiles its own layers independently
            # using synthetic intermediate tensors when necessary.
            intermediate_tensors = None
            if not get_pp_group().is_first_rank:
                if profile_num_layers is not None:
                    # Create synthetic intermediate tensors for non-first
                    # ranks so they can profile independently.
                    hidden_size = getattr(
                        getattr(self.model_runner.model, "config", None),
                        "hidden_size", 4096)
                    n_tokens = (req_len * num_req if is_prompt
                                else num_req)
                    device = next(
                        self.model_runner.model.parameters()).device
                    intermediate_tensors = IntermediateTensors({
                        "hidden_states": torch.zeros(
                            n_tokens, hidden_size, device=device,
                            dtype=torch.float16),
                        "residual": torch.zeros(
                            n_tokens, hidden_size, device=device,
                            dtype=torch.float16),
                    })
                else:
                    intermediate_tensors = IntermediateTensors(
                        get_pp_group().recv_tensor_dict(
                            all_gather_group=get_tp_group()))

            self.model_runner.attn_state.begin_forward(model_input)
            if model_input.attn_metadata is not None:
                _ = model_input.attn_metadata.prefill_metadata
                _ = model_input.attn_metadata.decode_metadata
            module_events = {}
            torch.cuda.synchronize()
            total_start = torch.cuda.Event(enable_timing=True)
            total_end = torch.cuda.Event(enable_timing=True)
            total_start.record()
            with set_forward_context(model_input.attn_metadata,
                                     self.model_runner.vllm_config,
                                     model_input.virtual_engine):
                self.model_runner.model(
                    input_ids=model_input.input_tokens,
                    positions=model_input.input_positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=model_input.inputs_embeds,
                    module_events=module_events,
                    profile_num_layers=profile_num_layers,
                    profile_layer_start=profile_layer_start,
                )
            total_end.record()
            torch.cuda.synchronize()

            module_ms = {}
            module_total_ms = {}
            module_counts = {}
            for name, events in module_events.items():
                elapsed = [s.elapsed_time(e) for s, e in events]
                module_ms[name] = float(sum(elapsed) / len(elapsed))
                module_total_ms[name] = float(sum(elapsed))
                module_counts[name] = len(elapsed)

            group_total_ms = {}
            group_per_layer_ms = {}
            group_counts = {}
            for group_name, event_names in event_groups.items():
                present = [n for n in event_names if n in module_total_ms]
                if not present:
                    continue
                total_ms = sum(module_total_ms[n] for n in present)
                layer_count = max(module_counts[n] for n in present)
                group_total_ms[group_name] = float(total_ms)
                group_per_layer_ms[group_name] = float(total_ms / layer_count)
                group_counts[group_name] = int(layer_count)

            grouped_event_names = {
                n for names in event_groups.values() for n in names
            }
            ungrouped = [
                n for n in module_total_ms if n not in grouped_event_names
            ]
            if ungrouped:
                total_ms = sum(module_total_ms[n] for n in ungrouped)
                layer_count = max(module_counts[n] for n in ungrouped)
                group_total_ms["unmodeled_other"] = float(total_ms)
                group_per_layer_ms["unmodeled_other"] = float(
                    total_ms / layer_count)
                group_counts["unmodeled_other"] = int(layer_count)

            return {
                "total_forward_ms": float(total_start.elapsed_time(total_end)),
                "module_ms": module_ms,
                "module_total_ms": module_total_ms,
                "module_counts": module_counts,
                "group_total_ms": group_total_ms,
                "group_per_layer_ms": group_per_layer_ms,
                "group_counts": group_counts,
            }

        if configs is None:
            configs = [
                ("prefill", 64, 1), ("prefill", 128, 1),
                ("prefill", 256, 1), ("prefill", 512, 1),
                ("prefill", 1024, 1), ("prefill", 2048, 1),
                ("prefill", 128, 4), ("prefill", 256, 4),
                ("prefill", 512, 4), ("prefill", 1024, 4),
                ("decode", 256, 1), ("decode", 512, 1),
                ("decode", 1024, 1), ("decode", 2048, 1),
                ("decode", 256, 8), ("decode", 512, 8),
                ("decode", 1024, 8), ("decode", 2048, 8),
                ("decode", 256, 32), ("decode", 512, 32),
                ("decode", 1024, 32),
            ]

        max_model_len = max(1, int(getattr(self.model_config, "max_model_len",
                                           8192)))
        configs = [
            (mode, int(req_len), int(num_req))
            for mode, req_len, num_req in configs
            if int(req_len) <= max_model_len
        ]
        if not configs:
            return {
                "meta": {
                    "profile_num_layers": profile_num_layers,
                    "profile_layer_start": profile_layer_start,
                    "total_model_layers": total_model_layers,
                    "max_model_len": max_model_len,
                },
                "rows": [],
            }

        warmup_configs = []
        for mode in ("prefill", "decode"):
            for candidate in configs:
                if candidate[0] == mode:
                    warmup_configs.append(candidate)
                    break

        for _ in range(warmup):
            for mode, req_len, num_req in warmup_configs:
                run_one(mode, req_len, num_req)

        rows = []
        for mode, req_len, num_req in configs:
            for _ in range(warmup_per_config):
                run_one(mode, int(req_len), int(num_req))
            per_repeat = []
            for _ in range(repeat):
                out = run_one(mode, int(req_len), int(num_req))
                if out is not None:
                    per_repeat.append(out)
            if not per_repeat:
                continue

            component_names = sorted({
                name for one in per_repeat for name in one["module_ms"]
            })
            module_summary = {}
            module_total_summary = {}
            for name in component_names:
                vals = [one["module_ms"].get(name, float("nan"))
                        for one in per_repeat]
                module_summary[name] = summarize(vals)
                total_vals = [one["module_total_ms"].get(name, float("nan"))
                              for one in per_repeat]
                module_total_summary[name] = summarize(total_vals)
            group_names = sorted({
                name for one in per_repeat for name in one["group_total_ms"]
            })
            group_total_summary = {}
            group_per_layer_summary = {}
            group_estimated_model_total_summary = {}
            group_counts = {}
            for name in group_names:
                total_vals = [one["group_total_ms"].get(name, float("nan"))
                              for one in per_repeat]
                group_total_summary[name] = summarize(total_vals)
                per_layer_vals = [
                    one["group_per_layer_ms"].get(name, float("nan"))
                    for one in per_repeat
                ]
                group_per_layer_summary[name] = summarize(per_layer_vals)
                if total_model_layers is not None:
                    estimated_vals = [
                        v * int(total_model_layers) for v in per_layer_vals
                    ]
                    group_estimated_model_total_summary[name] = summarize(
                        estimated_vals)
                group_counts[name] = max(
                    one["group_counts"].get(name, 0) for one in per_repeat)
            n_tokens, sum_context = feature_values(mode, int(req_len),
                                                   int(num_req))
            row = {
                "mode": mode,
                "req_len": int(req_len),
                "num_req": int(num_req),
                "n_tokens": n_tokens,
                "sum_context": sum_context,
                "total_forward_ms": summarize(
                    [one["total_forward_ms"] for one in per_repeat]),
                "module_ms": module_summary,
                "module_total_ms": module_total_summary,
                "module_counts": per_repeat[-1]["module_counts"],
                "group_total_ms": group_total_summary,
                "group_per_layer_ms": group_per_layer_summary,
                "group_estimated_model_total_ms":
                    group_estimated_model_total_summary,
                "group_counts": group_counts,
            }
            rows.append(row)
            if (self.is_driver_worker
                    and _env_truthy("VLLM_TIME_PREDICTOR_VERBOSE")):
                total_med = row["total_forward_ms"]["median"]
                print(f"  [{mode}] req_len={req_len}, B={num_req}: "
                      f"forward={total_med:.3f}ms",
                      flush=True)
                print("    grouped events:", flush=True)
                for name in group_names:
                    stats = group_per_layer_summary[name]
                    total_stats = group_total_summary[name]
                    count = group_counts.get(name, 0)
                    print(
                        f"      {name}: "
                        f"per_layer_median={stats['median']:.4f}ms, "
                        f"total_median={total_stats['median']:.4f}ms, "
                        f"estimated_"
                        f"{total_model_layers or count}layer_median="
                        f"{stats['median'] * (total_model_layers or count):.4f}ms, "
                        f"mean={stats['mean']:.4f}ms, "
                        f"min={stats['min']:.4f}ms, "
                        f"max={stats['max']:.4f}ms, "
                        f"count={count}",
                        flush=True,
                    )

        if not self.is_driver_worker:
            return None
        return {
            "meta": {
                "warmup": warmup,
                "repeat": repeat,
                "block_size": block_size,
                "gpu_blocks": gpu_blocks,
                "profile_kv_cache_fraction": profile_kv_cache_fraction,
                "profile_max_blocks": max_blocks,
                "profile_num_layers": profile_num_layers,
                "profile_layer_start": profile_layer_start,
                "total_model_layers": total_model_layers,
                "rank": getattr(self, "rank", None),
                "is_driver": getattr(self, "is_driver_worker", None),
                "note": "Qwen2 regular forward with optional module_events",
            },
            "rows": rows,
        }


    def profile_mixed_iterations(
        self,
        *,
        trace_path: str,
        predictor_path: str,
        num_iterations: int = 32,
        warmup: int = 2,
        repeat: int = 3,
        prefill_chunk_size: int = 512,
        prefill_batch_size: int = 1,
        decode_batch_size: int = 32,
        max_trace_requests: int = 64,
    ):
        """Replay mixed chunk-prefill/decode iterations from a trace."""
        import csv
        import math

        from vllm.engine.time_predictor import TimePredictor
        from vllm.sequence import SequenceData, SequenceGroupMetadata

        block_size = self.cache_config.block_size
        num_layers = getattr(
            getattr(self.model_config, "hf_text_config",
                    self.model_config.hf_config),
            "num_hidden_layers",
            getattr(self.model_config.hf_config, "num_layers", 32),
        )
        predictor = TimePredictor(predictor_path, num_layers=num_layers)

        def _load_trace():
            prefill, decode = [], []
            with open(trace_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    if idx >= max_trace_requests:
                        break
                    req_type = row.get("RequestType", "")
                    prompt_len = min(
                        int(row["ContextTokens"]),
                        self.model_config.max_model_len - 1)
                    output_len = max(1, int(row["GeneratedTokens"]))
                    item = {
                        "req_id": int(row.get("req_id", idx)),
                        "prompt_len": prompt_len,
                        "output_len": output_len,
                    }
                    if "prefill" in req_type:
                        prefill.append({**item, "computed": 0})
                    elif "decode" in req_type:
                        decode.append({**item, "generated": 0})
            if not prefill or not decode:
                raise ValueError("trace must contain both RT_prefill and RT_decode rows")
            return prefill, decode

        prefill_states, decode_states = _load_trace()
        prefill_cursor = 0
        decode_cursor = 0
        block_cursor = 0

        def _block_table(total_len: int):
            nonlocal block_cursor
            nblocks = max(1, (total_len + block_size - 1) // block_size)
            start = block_cursor
            block_cursor += nblocks
            return {0: list(range(start, start + nblocks))}

        def _next_prefill_chunk():
            nonlocal prefill_cursor
            for _ in range(len(prefill_states)):
                state = prefill_states[prefill_cursor % len(prefill_states)]
                prefill_cursor += 1
                if state["computed"] >= state["prompt_len"]:
                    state["computed"] = 0
                remaining = state["prompt_len"] - state["computed"]
                if remaining <= 0:
                    continue
                query_len = min(prefill_chunk_size, remaining)
                cache_len = state["computed"]
                context_len = cache_len + query_len
                state["computed"] = context_len
                return {
                    "req_id": state["req_id"],
                    "query_len": query_len,
                    "context_len": context_len,
                    "cache_len": cache_len,
                }
            return None

        def _next_decode_step():
            nonlocal decode_cursor
            for _ in range(len(decode_states)):
                state = decode_states[decode_cursor % len(decode_states)]
                decode_cursor += 1
                if state["generated"] >= state["output_len"]:
                    state["generated"] = 0
                context_len = min(
                    state["prompt_len"] + state["generated"],
                    self.model_config.max_model_len - 1)
                state["generated"] += 1
                return {
                    "req_id": state["req_id"],
                    "query_len": 1,
                    "context_len": context_len,
                }
            return None

        def _build_iteration_plan():
            prefill_reqs = []
            decode_reqs = []
            for _ in range(prefill_batch_size):
                chunk = _next_prefill_chunk()
                if chunk is not None:
                    prefill_reqs.append(chunk)
            for _ in range(decode_batch_size):
                step = _next_decode_step()
                if step is not None:
                    decode_reqs.append(step)
            return prefill_reqs, decode_reqs

        def _build_execute_model_req(prefill_reqs, decode_reqs):
            sgm_list = []
            sampling_params = SamplingParams()

            for i, req in enumerate(prefill_reqs):
                prompt_ids = _fake_token_ids(req["context_len"], token_id=1)
                seq_data = SequenceData.from_seqs(prompt_ids)
                if req["cache_len"] > 0:
                    seq_data.update_num_computed_tokens(req["cache_len"])
                block_tables = _block_table(req["context_len"])
                sgm_list.append(SequenceGroupMetadata(
                    request_id=f"mixed-prefill-{req['req_id']}-{i}",
                    is_prompt=True,
                    seq_data={0: seq_data},
                    sampling_params=sampling_params,
                    block_tables=block_tables,
                    do_sample=False,
                    token_chunk_size=req["query_len"],
                ))

            for i, req in enumerate(decode_reqs):
                prompt_ids = _fake_token_ids(req["context_len"], token_id=1)
                seq_data = SequenceData.from_seqs(
                    prompt_ids, output_token_ids=[2])
                seq_data.update_num_computed_tokens(req["context_len"])
                block_tables = _block_table(req["context_len"] + 1)
                sgm_list.append(SequenceGroupMetadata(
                    request_id=f"mixed-decode-{req['req_id']}-{i}",
                    is_prompt=False,
                    seq_data={0: seq_data},
                    sampling_params=sampling_params,
                    block_tables=block_tables,
                    do_sample=False,
                    token_chunk_size=1,
                ))

            return ExecuteModelRequest(
                seq_group_metadata_list=sgm_list,
                virtual_engine=0,
                num_steps=1,
                blocks_to_swap_in=[],
                blocks_to_swap_out=[],
                blocks_to_copy=[],
            )

        def _run_iteration(prefill_reqs, decode_reqs):
            nonlocal block_cursor
            block_cursor = 0
            execute_model_req = _build_execute_model_req(prefill_reqs, decode_reqs)
            inputs = self.prepare_input(execute_model_req)
            if inputs is None:
                return float("nan")
            model_input, worker_input, kwargs = inputs
            self.execute_worker(worker_input)
            intermediate_tensors = None
            if not get_pp_group().is_first_rank:
                intermediate_tensors = IntermediateTensors(
                    get_pp_group().recv_tensor_dict(all_gather_group=get_tp_group()))
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self.model_runner.execute_model(
                model_input=model_input,
                kv_caches=self.kv_cache[worker_input.virtual_engine]
                if self.kv_cache is not None else None,
                intermediate_tensors=intermediate_tensors,
                num_steps=1,
                **kwargs,
            )
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end)

        plans = [_build_iteration_plan()
                 for _ in range(num_iterations + warmup)]

        for prefill_reqs, decode_reqs in plans[:warmup]:
            _run_iteration(prefill_reqs, decode_reqs)

        results = []
        for iteration_id, (prefill_reqs, decode_reqs) in enumerate(plans[warmup:]):
            actual_times = []
            for _ in range(repeat):
                t = _run_iteration(prefill_reqs, decode_reqs)
                if math.isfinite(t):
                    actual_times.append(t)
            if not actual_times:
                continue

            prefill_predict_reqs = [
                (r["query_len"], r["context_len"]) for r in prefill_reqs
            ]
            decode_predict_reqs = [
                (r["query_len"], r["context_len"]) for r in decode_reqs
            ]
            pred_ms = predictor.predict_forward_ms(
                prefill_predict_reqs, decode_predict_reqs)
            actual_ms = statistics.median(actual_times)
            error_pct = (abs(pred_ms - actual_ms) / actual_ms * 100
                         if actual_ms > 0 else float("nan"))
            results.append({
                "iteration": iteration_id,
                "prefill_count": len(prefill_reqs),
                "decode_count": len(decode_reqs),
                "prefill_tokens": sum(r["query_len"] for r in prefill_reqs),
                "decode_tokens": sum(r["query_len"] for r in decode_reqs),
                "prefill_context_sum": sum(r["context_len"] for r in prefill_reqs),
                "decode_context_sum": sum(r["context_len"] for r in decode_reqs),
                "actual_ms": actual_ms,
                "pred_ms": pred_ms,
                "error_pct": error_pct,
                "actual_each_ms": actual_times,
                "prefill_reqs": prefill_predict_reqs,
                "decode_reqs": decode_predict_reqs,
            })

        if not self.is_driver_worker:
            return None
        return {
            "meta": {
                "trace_path": trace_path,
                "predictor_path": predictor_path,
                "num_layers": num_layers,
                "num_iterations": num_iterations,
                "warmup": warmup,
                "repeat": repeat,
                "prefill_chunk_size": prefill_chunk_size,
                "prefill_batch_size": prefill_batch_size,
                "decode_batch_size": decode_batch_size,
                "max_trace_requests": max_trace_requests,
            },
            "iterations": results,
        }


def _cuda_time_ms(fn, sync: bool = True) -> float:
    """Measure GPU time (ms) for a callable that launches CUDA kernels."""
    if sync:
        torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)

def _inclusive_range(start: int, stop: int, step: int) -> List[int]:
    # 你给的 (128, 4097, 128) 这种写法更像 range(start, stop, step)
    # 但 linear 的 (16,4096,16) 你很可能希望包含 4096
    vals = list(range(start, stop, step))
    if (stop - start) % step == 0:
        # stop 是“上界”，且刚好落在步长上：把 stop 加进去
        if (start + ((stop - start) // step) * step) == stop:
            if len(vals) == 0 or vals[-1] != stop:
                vals.append(stop)
    return vals

def _fake_token_ids(length: int, token_id: int = 1) -> List[int]:
    # 用常数 token id 就行；profile kernel shape 为主
    return [token_id] * length

def _make_block_tables(req_len: int, is_prompt: bool, num_req: int,
                       block_size: int,
                       base_block_id: int = 0) -> Tuple[List[Dict[int, List[int]]], int]:

    # prefill: len = req_len
    # decode:  len = req_len + 1（通常会把新 token 也算进 KV）
    total_len = req_len if is_prompt else (req_len + 1)
    nblocks = (total_len + block_size - 1) // block_size

    tables = []
    cur = base_block_id
    for i in range(num_req):
        seq_id = 0  # 每个 SequenceGroupMetadata 里我们只放 1 条 seq，用 0 即可
        block_ids = list(range(cur, cur + nblocks))
        cur += nblocks
        tables.append({seq_id: block_ids})
    return tables, cur

def _build_execute_model_req_attn(req_len: int, num_req: int, is_prompt: bool,
                                 block_size: int,
                                 virtual_engine: int = 0) -> "ExecuteModelRequest":
    from vllm.sequence import ExecuteModelRequest, SequenceGroupMetadata, SequenceData

    sampling_params = SamplingParams()  # 或 SamplingParams() 也行，但 seed=0 更稳

    block_tables_list, _ = _make_block_tables(
        req_len=req_len, is_prompt=is_prompt, num_req=num_req,
        block_size=block_size, base_block_id=0
    )

    sgm_list = []
    for i in range(num_req):
        # 每个 request 做成一个独立的 sequence group（最贴近 scheduler 的输出）
        prompt_ids = _fake_token_ids(req_len, token_id=1)

        if is_prompt:
            seq_data = {0: SequenceData.from_seqs(prompt_ids)}
            token_chunk_size = req_len
        else:
            # decode：prompt 是历史上下文，output_token_ids 放 1 个新 token
            seq_data = {0: SequenceData.from_seqs(prompt_ids, output_token_ids=[2])}
            seq_data[0].update_num_computed_tokens(req_len)

            token_chunk_size = 1

        sgm = SequenceGroupMetadata(
            request_id=f"prof-attn-{'p' if is_prompt else 'd'}-{req_len}-{i}",
            is_prompt=is_prompt,
            seq_data=seq_data,
            sampling_params=sampling_params,
            block_tables=block_tables_list[i],
            do_sample=False,
            token_chunk_size=token_chunk_size,
        )
        # state 由 default_factory 初始化；num_steps=1 默认就 ok
        sgm_list.append(sgm)

    return ExecuteModelRequest(
        seq_group_metadata_list=sgm_list,
        virtual_engine=virtual_engine,
        num_steps=1,
        blocks_to_swap_in=[],
        blocks_to_swap_out=[],
        blocks_to_copy=[],
    )

# def _build_execute_model_req_linear(num_token: int,
#                                    block_size: int,
#                                    virtual_engine: int = 0) -> "ExecuteModelRequest":
#     from vllm.sequence import ExecuteModelRequest, SequenceGroupMetadata, SequenceData

#     req_len = 1
#     num_req = num_token
#     block_tables_list, _ = _make_block_tables(
#         req_len=req_len, is_prompt=False, num_req=num_req,
#         block_size=block_size, base_block_id=0
#     )

#     sgm_list = []
#     for i in range(num_req):
#         prompt_ids = _fake_token_ids(req_len, token_id=1)
#         seq_data = {0: SequenceData.from_seqs(prompt_ids, output_token_ids=[2])}

#         sgm = SequenceGroupMetadata(
#             request_id=f"prof-linear-{num_token}-{i}",
#             is_prompt=False,
#             seq_data=seq_data,
#             sampling_params=SamplingParams(),
#             block_tables=block_tables_list[i],
#             do_sample=False,
#             token_chunk_size=1,
#         )
#         sgm_list.append(sgm)

#     return ExecuteModelRequest(
#         seq_group_metadata_list=sgm_list,
#         virtual_engine=virtual_engine,
#         num_steps=1,
#         blocks_to_swap_in=[],
#         blocks_to_swap_out=[],
#         blocks_to_copy=[],
#     )


def _build_execute_model_req_linear(num_token: int,
                                    block_size: int,
                                    virtual_engine: int = 0) -> "ExecuteModelRequest":
    from vllm.sequence import ExecuteModelRequest, SequenceGroupMetadata, SequenceData
    from vllm.sampling_params import SamplingParams

    # 1. 直接构造一个长度为 num_token 的 prompt
    prompt_ids = [1] * num_token
    seq_data = {0: SequenceData.from_seqs(prompt_ids)}

    # 2. 生成真实的 mock block tables，避免 builder 解析和算 slot mapping 时报错
    block_tables_list, _ = _make_block_tables(
        req_len=num_token, is_prompt=True, num_req=1,
        block_size=block_size, base_block_id=0
    )
    block_tables = block_tables_list[0]

    # 3. 构造单一的 Prefill SequenceGroup
    sgm = SequenceGroupMetadata(
        request_id=f"prof-linear-prefill-{num_token}",
        is_prompt=True,               # 设为 Prefill 阶段
        seq_data=seq_data,
        sampling_params=SamplingParams(),
        block_tables=block_tables,    # <- 传入真实尺寸的 mock块表
        do_sample=False,
        token_chunk_size=num_token,   # 确保 Chunking 机制一次性把 num_token 全部吃进去
    )

    # 4. 返回 Request
    return ExecuteModelRequest(
        seq_group_metadata_list=[sgm],
        virtual_engine=virtual_engine,
        num_steps=1,
        blocks_to_swap_in=[],
        blocks_to_swap_out=[],
        blocks_to_copy=[],
    )


class WorkerWrapperBase:
    """
    This class represents one process in an executor/engine. It is responsible
    for lazily initializing the worker and handling the worker's lifecycle.
    We first instantiate the WorkerWrapper, which remembers the worker module
    and class name. Then, when we call `update_environment_variables`, and the
    real initialization happens in `init_worker`.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        rpc_rank: int = 0,
    ) -> None:
        """
        Initialize the worker wrapper with the given vllm_config and rpc_rank.
        Note: rpc_rank is the rank of the worker in the executor. In most cases,
        it is also the rank of the worker in the distributed group. However,
        when multiple executors work together, they can be different.
        e.g. in the case of SPMD-style offline inference with TP=2,
        users can launch 2 engines/executors, each with only 1 worker.
        All workers have rpc_rank=0, but they have different ranks in the TP
        group.
        """
        self.rpc_rank = rpc_rank
        self.worker: Optional[WorkerBase] = None
        self.vllm_config: Optional[VllmConfig] = None
        # do not store this `vllm_config`, `init_worker` will set the final
        # one. TODO: investigate if we can remove this field in
        # `WorkerWrapperBase`, `init_cached_hf_modules` should be
        # unnecessary now.
        if vllm_config.model_config is not None:
            # it can be None in tests
            trust_remote_code = vllm_config.model_config.trust_remote_code
            if trust_remote_code:
                # note: lazy import to avoid importing torch before initializing
                from vllm.utils import init_cached_hf_modules
                init_cached_hf_modules()

    def adjust_rank(self, rank_mapping: Dict[int, int]) -> None:
        """
        Adjust the rpc_rank based on the given mapping.
        It is only used during the initialization of the executor,
        to adjust the rpc_rank of workers after we create all workers.
        """
        if self.rpc_rank in rank_mapping:
            self.rpc_rank = rank_mapping[self.rpc_rank]

    def update_environment_variables(self, envs_list: List[Dict[str,
                                                                str]]) -> None:
        envs = envs_list[self.rpc_rank]
        key = 'CUDA_VISIBLE_DEVICES'
        if key in envs and key in os.environ:
            # overwriting CUDA_VISIBLE_DEVICES is desired behavior
            # suppress the warning in `update_environment_variables`
            del os.environ[key]
        update_environment_variables(envs)

    def init_worker(self, all_kwargs: List[Dict[str, Any]]) -> None:
        """
        Here we inject some common logic before initializing the worker.
        Arguments are passed to the worker class constructor.
        """
        kwargs = all_kwargs[self.rpc_rank]
        self.vllm_config = kwargs.get("vllm_config", None)
        assert self.vllm_config is not None, (
            "vllm_config is required to initialize the worker")
        enable_trace_function_call_for_thread(self.vllm_config)

        from vllm.plugins import load_general_plugins
        load_general_plugins()

        if isinstance(self.vllm_config.parallel_config.worker_cls, str):
            worker_class = resolve_obj_by_qualname(
                self.vllm_config.parallel_config.worker_cls)
        else:
            logger.warning(
                "passing worker_cls as a class object is strongly deprecated,"
                " as the serialization of class objects can be tricky and"
                " error-prone. To be safe, please keep the class in a separate"
                " module and pass the qualified name of the class as a string."
            )
            assert isinstance(self.vllm_config.parallel_config.worker_cls,
                              bytes)
            worker_class = cloudpickle.loads(
                self.vllm_config.parallel_config.worker_cls)
        if self.vllm_config.parallel_config.worker_extension_cls:
            worker_extension_cls = resolve_obj_by_qualname(
                self.vllm_config.parallel_config.worker_extension_cls)
            extended_calls = []
            if worker_extension_cls not in worker_class.__bases__:
                # check any conflicts between worker and worker_extension_cls
                for attr in dir(worker_extension_cls):
                    if attr.startswith("__"):
                        continue
                    assert not hasattr(worker_class, attr), (
                        f"Worker class {worker_class} already has an attribute"
                        f" {attr}, which conflicts with the worker"
                        f" extension class {worker_extension_cls}.")
                    if callable(getattr(worker_extension_cls, attr)):
                        extended_calls.append(attr)
                # dynamically inherit the worker extension class
                worker_class.__bases__ = worker_class.__bases__ + (
                    worker_extension_cls, )
                logger.info(
                    "Injected %s into %s for extended collective_rpc calls %s",
                    worker_extension_cls, worker_class, extended_calls)
        with set_current_vllm_config(self.vllm_config):
            # To make vLLM config available during worker initialization
            self.worker = worker_class(**kwargs)
            assert self.worker is not None

    def initialize_from_config(self, kv_cache_configs: List[Any]) -> None:
        kv_cache_config = kv_cache_configs[self.rpc_rank]
        with set_current_vllm_config(self.vllm_config):
            self.worker.initialize_from_config(kv_cache_config)  # type: ignore

    def init_device(self):
        with set_current_vllm_config(self.vllm_config):
            # To make vLLM config available during device initialization
            self.worker.init_device()  # type: ignore

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        try:
            # method resolution order:
            # if a method is defined in this class, it will be called directly.
            # otherwise, since we define `__getattr__` and redirect attribute
            # query to `self.worker`, the method will be called on the worker.
            return run_method(self, method, args, kwargs)
        except Exception as e:
            # if the driver worker also execute methods,
            # exceptions in the rest worker may cause deadlock in rpc like ray
            # see https://github.com/vllm-project/vllm/issues/3455
            # print the error and inform the user to solve the error
            msg = (f"Error executing method {method!r}. "
                   "This might cause deadlock in distributed execution.")
            logger.exception(msg)
            raise e

    async def execute_method_async(self, method: Union[str, bytes], *args,
                                   **kwargs):
        """Resolve an RPC method and await it when it is asynchronous."""
        try:
            result = run_method(self, method, args, kwargs)
            if inspect.isawaitable(result):
                return await result
            return result
        except Exception as e:
            msg = (f"Error executing async method {method!r}. "
                   "This might cause deadlock in distributed execution.")
            logger.exception(msg)
            raise e

    def __getattr__(self, attr):
        return getattr(self.worker, attr)


def extract_previous_hidden_states(
        data: Union[ExecuteModelRequest, Dict[str, torch.Tensor]]) -> \
            Dict[str, torch.Tensor]:
    """If data contains previous_hidden_states, extract it. This returns a dict
    which can be used directly as additional kwargs in any following 
    execute_model calls. This is used in draft models like EAGLE."""
    output = {}

    # When called from non-driver worker, data is dict but when called from
    # driver worker, data is ExecuteModelRequest.
    if isinstance(data, dict):
        if "previous_hidden_states" in data:
            output["previous_hidden_states"] = data["previous_hidden_states"]
    elif data.previous_hidden_states is not None:
        output["previous_hidden_states"] = data.previous_hidden_states\
            .hidden_states

    return output
