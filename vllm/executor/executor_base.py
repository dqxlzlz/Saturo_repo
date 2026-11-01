# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time
from abc import ABC, abstractmethod
from typing import (Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple,
                    Union)

import torch.nn as nn
from typing_extensions import TypeVar

import vllm.platforms
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sequence import ExecuteModelRequest, PoolerOutput
from vllm.utils import make_async
from vllm.worker.worker_base import WorkerBase

logger = init_logger(__name__)

_R = TypeVar("_R", default=Any)


class ExecutorBase(ABC):
    """Base class for all executors.

    An executor is responsible for executing the model on one device,
    or it can be a distributed executor 
    that can execute the model on multiple devices.
    """

    uses_ray: bool  # whether the executor uses Ray for orchestration.

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
        self._init_executor()
        self.is_sleeping = False
        self.sleeping_tags: set[str] = set()

    @abstractmethod
    def _init_executor(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def collective_rpc(self,
                       method: Union[str, Callable[..., _R]],
                       timeout: Optional[float] = None,
                       args: Tuple = (),
                       kwargs: Optional[Dict[str, Any]] = None) -> List[_R]:
        """
        Execute an RPC call on all workers.

        Args:
            method: Name of the worker method to execute, or a callable that
                is serialized and sent to all workers to execute.

                If the method is a callable, it should accept an additional
                `self` argument, in addition to the arguments passed in `args`
                and `kwargs`. The `self` argument will be the worker object.
            timeout: Maximum time in seconds to wait for execution. Raises a
                [`TimeoutError`][] on timeout. `None` means wait indefinitely.
            args: Positional arguments to pass to the worker method.
            kwargs: Keyword arguments to pass to the worker method.

        Returns:
            A list containing the results from each worker.
        
        Note:
            It is recommended to use this API to only pass control messages,
            and set up data-plane communication to pass data.
        """
        raise NotImplementedError

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available blocks for the GPU KV cache and
        swappable CPU KV cache.

        Normally, this should simply delegate to the underlying Worker. Some
        ExecutorBase may require modification of the result, e.g. to ensure the
        selected cache sizes are compatible with all workers.

        Returns a Tuple[num_gpu_blocks, num_cpu_blocks], where num_gpu_blocks
        are blocks that are "active" on the device and can be appended to.
        num_cpu_blocks refers to "swapped" blocks in CPU memory and cannot be
        appended to.
        """
        results = self.collective_rpc("determine_num_available_blocks")
        a = min([r[0] for r in results])
        b = min([r[1] for r in results])
        return a, b

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks) -> None:
        """Initialize the KV cache by invoking the underlying worker.
        """
        # NOTE: This is logged in the executor because there can be >1 workers.
        logger.info("# %s blocks: %d, # CPU blocks: %d",
                    vllm.platforms.current_platform.device_name,
                    num_gpu_blocks, num_cpu_blocks)
        max_concurrency = (num_gpu_blocks * self.cache_config.block_size /
                           self.model_config.max_model_len)
        logger.info("Maximum concurrency for %s tokens per request: %.2fx",
                    self.model_config.max_model_len, max_concurrency)

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        self.collective_rpc("initialize_cache",
                            args=(num_gpu_blocks, num_cpu_blocks))

    def apply_model(self, func: Callable[[nn.Module], _R]) -> list[_R]:
        """
        Run a function directly on the model inside each worker,
        returning the result for each of them.
        """

        def rpc_func(worker: WorkerBase) -> _R:
            return func(worker.get_model())

        return self.collective_rpc(rpc_func)

    def set_p_pool_preemption(self, enabled: bool) -> None:
        self.collective_rpc("set_p_pool_preemption", args=(enabled, ))

    def execute_model(
        self, execute_model_req: ExecuteModelRequest
    ) -> Optional[List[Union[SamplerOutput, PoolerOutput]]]:
        output = self.collective_rpc("execute_model",
                                     args=(execute_model_req, ))
        return output[0]

    def profile_attn_linear(self, warmup: int = 20, repeat: int = 3, num_layer_to_run:int=5) -> Dict[str, Any]:
        output = self.collective_rpc("profile_attn_linear", kwargs={"warmup":warmup, "repeat":repeat, "num_layer_to_run": num_layer_to_run})
        return output[0]

    def profile_full_layer(self, warmup: int = 3, repeat: int = 5, num_layer_to_run: int = 5) -> Dict[str, Any]:
        output = self.collective_rpc("profile_full_layer", kwargs={"warmup": warmup, "repeat": repeat, "num_layer_to_run": num_layer_to_run})
        return output[0]
    def profile_full_iteration(self, warmup: int = 3, repeat: int = 5) -> Dict[str, Any]:
        output = self.collective_rpc("profile_full_iteration", kwargs={"warmup": warmup, "repeat": repeat})
        return output[0]

    def profile_mixed_iterations(
        self,
        trace_path: str,
        predictor_path: str,
        num_iterations: int = 32,
        warmup: int = 2,
        repeat: int = 3,
        prefill_chunk_size: int = 512,
        prefill_batch_size: int = 1,
        decode_batch_size: int = 32,
        max_trace_requests: int = 64,
    ) -> Dict[str, Any]:
        output = self.collective_rpc(
            "profile_mixed_iterations",
            kwargs={
                "trace_path": trace_path,
                "predictor_path": predictor_path,
                "num_iterations": num_iterations,
                "warmup": warmup,
                "repeat": repeat,
                "prefill_chunk_size": prefill_chunk_size,
                "prefill_batch_size": prefill_batch_size,
                "decode_batch_size": decode_batch_size,
                "max_trace_requests": max_trace_requests,
            },
        )
        return output[0]

    def profile_qwen2_module_times(
        self,
        warmup: int = 3,
        repeat: int = 5,
        warmup_per_config: int = 0,
        configs: Optional[List[Tuple[str, int, int]]] = None,
        profile_num_layers: Optional[int] = None,
        profile_layer_start: int = 0,
        profile_kv_cache_fraction: float = 0.65,
    ) -> Dict[str, Any]:
        output = self.collective_rpc(
            "profile_qwen2_module_times",
            kwargs={
                "warmup": warmup,
                "repeat": repeat,
                "warmup_per_config": warmup_per_config,
                "configs": configs,
                "profile_num_layers": profile_num_layers,
                "profile_layer_start": profile_layer_start,
                "profile_kv_cache_fraction": profile_kv_cache_fraction,
            })
        return output[0]


    def stop_remote_worker_execution_loop(self) -> None:
        """Releases parallel workers from model loop."""
        return

    def add_lora(self, lora_request: LoRARequest) -> bool:
        assert lora_request.lora_int_id > 0, "lora_id must be greater than 0."
        return all(self.collective_rpc("add_lora", args=(lora_request, )))

    def remove_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return all(self.collective_rpc("remove_lora", args=(lora_id, )))

    def pin_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return all(self.collective_rpc("pin_lora", args=(lora_id, )))

    def list_loras(self) -> Set[int]:
        sets = self.collective_rpc("list_loras")
        for s in sets:
            assert s == sets[0], "All workers should have the same LORAs."
        return sets[0]

    def add_prompt_adapter(
            self, prompt_adapter_request: PromptAdapterRequest) -> bool:
        assert prompt_adapter_request.prompt_adapter_id > 0, \
            "prompt_adapter_id must be greater than 0."
        return all(
            self.collective_rpc("add_prompt_adapter",
                                args=(prompt_adapter_request, )))

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        assert prompt_adapter_id > 0, \
            "prompt_adapter_id must be greater than 0."
        return all(
            self.collective_rpc("remove_prompt_adapter",
                                args=(prompt_adapter_id, )))

    def pin_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        assert prompt_adapter_id > 0, \
            "prompt_adapter_id must be greater than 0."
        return all(
            self.collective_rpc("pin_prompt_adapter",
                                args=(prompt_adapter_id, )))

    def list_prompt_adapters(self) -> Set[int]:
        sets = self.collective_rpc("list_prompt_adapters")
        for s in sets:
            assert (s == sets[0]
                    ), "All workers should have the same prompt adapters."
        return sets[0]

    def start_profile(self) -> None:
        self.collective_rpc("start_profile")

    def stop_profile(self) -> None:
        self.collective_rpc("stop_profile")

    def sleep(self, level: int = 1):
        if self.is_sleeping:
            logger.warning("Executor is already sleeping.")
            return
        time_before_sleep = time.perf_counter()
        self.collective_rpc("sleep", kwargs=dict(level=level))
        time_after_sleep = time.perf_counter()
        self.sleeping_tags = {"weights", "kv_cache"}
        self.is_sleeping = True
        logger.info("It took %.6f seconds to fall asleep.",
                    time_after_sleep - time_before_sleep)

    def wake_up(self, tags: Optional[list[str]] = None):
        if not self.is_sleeping:
            logger.warning("Executor is not sleeping.")
            return
        if tags:
            for tag in tags:
                if tag not in self.sleeping_tags:
                    logger.warning("Tag %s is not in sleeping tags %s", tag,
                                   self.sleeping_tags)
                    return
        time_before_wakeup = time.perf_counter()
        self.collective_rpc("wake_up", kwargs=dict(tags=tags))
        time_after_wakeup = time.perf_counter()
        logger.info("It took %.6f seconds to wake up tags %s.",
                    time_after_wakeup - time_before_wakeup,
                    tags if tags is not None else self.sleeping_tags)
        if tags:
            for tag in tags:
                self.sleeping_tags.remove(tag)
        else:
            self.sleeping_tags.clear()
        if not self.sleeping_tags:
            self.is_sleeping = False

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        self.collective_rpc("save_sharded_state",
                            kwargs=dict(path=path,
                                        pattern=pattern,
                                        max_size=max_size))

    @abstractmethod
    def check_health(self) -> None:
        """Checks if the executor is healthy. If not, it should raise an
        exception."""
        raise NotImplementedError

    def shutdown(self) -> None:
        """Shutdown the executor."""
        return

    def __del__(self):
        self.shutdown()

    async def execute_model_async(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        """Executes one model step on the given sequences."""
        output = await make_async(self.execute_model)(execute_model_req)
        return output

    async def stop_remote_worker_execution_loop_async(self) -> None:
        """Releases parallel workers from model loop."""
        return

    async def check_health_async(self) -> None:
        """Checks if the executor is healthy. If not, it should raise an
        exception."""
        self.check_health()


class DistributedExecutorBase(ExecutorBase):
    """Abstract superclass of distributed executor implementations."""

    def __init__(self, *args, **kwargs):
        # This is non-None when the execute model loop is running
        # in the parallel workers. It's a coroutine in the AsyncLLMEngine case.
        self.parallel_worker_tasks: Optional[Union[Any, Awaitable[Any]]] = None

        super().__init__(*args, **kwargs)

    def execute_model(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> Optional[List[SamplerOutput]]:
        # TODO: unify into collective_rpc
        if self.parallel_worker_tasks is None:
            self.parallel_worker_tasks = self._run_workers(
                "start_worker_execution_loop",
                async_run_tensor_parallel_workers_only=True)

        # Only the driver worker returns the sampling results.
        driver_outputs = self._driver_execute_model(execute_model_req)
        return driver_outputs

    def set_p_pool_preemption(self, enabled: bool) -> None:
        if self.parallel_worker_tasks is None:
            self._run_workers("set_p_pool_preemption", enabled)
            return

        # Remote TP workers are inside start_worker_execution_loop while a
        # forward pass is active, so queueing another RPC would run too late and
        # can block the engine. Set the driver immediately; TP ranks observe the
        # driver flag at the next layer boundary via model-level all_reduce.
        self.driver_worker.set_p_pool_preemption(enabled)

    def stop_remote_worker_execution_loop(self) -> None:
        if self.parallel_worker_tasks is None:
            return

        self._driver_execute_model(execute_model_req=None)
        parallel_worker_tasks = self.parallel_worker_tasks
        self.parallel_worker_tasks = None
        # Ensure that workers exit model loop cleanly
        # (this will raise otherwise)
        self._wait_for_tasks_completion(parallel_worker_tasks)

    @abstractmethod
    def _driver_execute_model(
        self, execute_model_req: Optional[ExecuteModelRequest]
    ) -> Optional[List[SamplerOutput]]:
        """Run execute_model in the driver worker.

        Passing None will cause the driver to stop the model execution loop
        running in each of the remote workers. In this case, this method
        returns None. Otherwise, this method returns the model output.
        """
        raise NotImplementedError

    def collective_rpc(self,
                       method: Union[str, Callable],
                       timeout: Optional[float] = None,
                       args: Tuple = (),
                       kwargs: Optional[Dict] = None) -> List[Any]:
        return self._run_workers(method, *args, **(kwargs or {}))

    @abstractmethod
    def _run_workers(
        self,
        method: Union[str, Callable],
        *args,
        async_run_tensor_parallel_workers_only: bool = False,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers.

        Args:
            async_run_tensor_parallel_workers_only: If True the method will be
                run only in the remote TP workers, not the driver worker.
                It will also be run asynchronously and return a list of futures
                rather than blocking on the results.
        
        # TODO: simplify and merge with collective_rpc
        """
        raise NotImplementedError

    @abstractmethod
    def _wait_for_tasks_completion(self, parallel_worker_tasks: Any) -> None:
        """Wait for futures returned from _run_workers() with
        async_run_remote_workers_only to complete."""
        raise NotImplementedError

    async def execute_model_async(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        if self.parallel_worker_tasks is None:
            # Start model execution loop running in the parallel workers
            self.parallel_worker_tasks = asyncio.create_task(
                self._start_worker_execution_loop())
            # The driver worker performs blocking collectives while preparing
            # its first model input. Give the newly-created task one event-loop
            # turn so remote TP ranks enter their execution loops before rank 0
            # reaches the first broadcast; otherwise TP>1 deadlocks with the
            # driver waiting in NCCL and remote workers waiting for their RPC.
            await asyncio.sleep(0)

        # Only the driver worker returns the sampling results.
        return await self._driver_execute_model_async(execute_model_req)

    async def stop_remote_worker_execution_loop_async(self) -> None:
        if self.parallel_worker_tasks is None:
            return

        await self._driver_execute_model_async()
        parallel_worker_tasks = self.parallel_worker_tasks
        self.parallel_worker_tasks = None
        # Ensure that workers exit model loop cleanly
        # (this will raise otherwise)
        await parallel_worker_tasks

    @abstractmethod
    async def _driver_execute_model_async(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
    ) -> List[SamplerOutput]:
        """Execute the model asynchronously in the driver worker.

        Passing None will cause the driver to stop the model execution
        loop running in each of the remote workers.
        """
        raise NotImplementedError

    @abstractmethod
    async def _start_worker_execution_loop(self):
        """Run execution loop on all workers. It guarantees all workers run
        the loop or None of them is running the loop. Loop can be stopped by
        `stop_remote_worker_execution_loop`.
        The API is idempotent (guarantee only 1 loop run at any moment)."""
        raise NotImplementedError


    def profile_attn_linear(self, warmup: int = 20, repeat: int = 10, num_layer_to_run:int = 5) -> Dict[str, Any]:
        # Run profile_attn_linear on ALL workers so that NCCL collective operations
        # (broadcast in prepare_input, all-reduce in QKV/O proj) are properly
        # synchronized across all TP ranks. The previous implementation only ran
        # on the driver worker while non-driver workers were in
        # start_worker_execution_loop, causing NCCL mismatch and timeout.
        #
        # _run_workers submits work to non-driver workers first (async via queue),
        # then runs the driver synchronously. All workers execute the same
        # profiling loop, participating in the same NCCL collectives.
        # Non-driver workers return None; the driver returns the profiling results.
        outputs = self._run_workers(
            "profile_attn_linear",
            warmup=warmup,
            repeat=repeat,
            num_layer_to_run=num_layer_to_run,
        )

        # outputs[0] is the driver worker's result (the profiling results dict)
        driver_out = outputs[0]
        assert driver_out is not None, "Driver worker returned None from profile_attn_linear"
        return driver_out

    def profile_full_layer(self, warmup: int = 3, repeat: int = 5, num_layer_to_run: int = 5) -> Dict[str, Any]:
        outputs = self._run_workers(
            "profile_full_layer",
            warmup=warmup,
            repeat=repeat,
            num_layer_to_run=num_layer_to_run,
        )
        driver_out = outputs[0]
        assert driver_out is not None, "Driver worker returned None from profile_full_layer"
        return driver_out

    def profile_mixed_iterations(
        self,
        trace_path: str,
        predictor_path: str,
        num_iterations: int = 32,
        warmup: int = 2,
        repeat: int = 3,
        prefill_chunk_size: int = 512,
        prefill_batch_size: int = 1,
        decode_batch_size: int = 32,
        max_trace_requests: int = 64,
    ) -> Dict[str, Any]:
        outputs = self._run_workers(
            "profile_mixed_iterations",
            trace_path=trace_path,
            predictor_path=predictor_path,
            num_iterations=num_iterations,
            warmup=warmup,
            repeat=repeat,
            prefill_chunk_size=prefill_chunk_size,
            prefill_batch_size=prefill_batch_size,
            decode_batch_size=decode_batch_size,
            max_trace_requests=max_trace_requests,
        )
        driver_out = outputs[0]
        assert driver_out is not None, (
            "Driver worker returned None from profile_mixed_iterations")
        return driver_out

    def profile_qwen2_module_times(
        self,
        warmup: int = 3,
        repeat: int = 5,
        warmup_per_config: int = 0,
        configs: Optional[List[Tuple[str, int, int]]] = None,
        profile_num_layers: Optional[int] = None,
        profile_layer_start: int = 0,
        profile_kv_cache_fraction: float = 0.65,
    ) -> Dict[str, Any]:
        outputs = self._run_workers(
            "profile_qwen2_module_times",
            warmup=warmup,
            repeat=repeat,
            warmup_per_config=warmup_per_config,
            configs=configs,
            profile_num_layers=profile_num_layers,
            profile_layer_start=profile_layer_start,
            profile_kv_cache_fraction=profile_kv_cache_fraction,
        )
        driver_out = outputs[0]
        assert driver_out is not None, (
            "Driver worker returned None from profile_qwen2_module_times")
        return driver_out
