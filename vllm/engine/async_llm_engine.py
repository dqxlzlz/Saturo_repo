# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from collections import defaultdict
import copy
import csv
import os
import time
import traceback
import weakref
from functools import partial
from typing import (Any, AsyncGenerator, Callable, Dict, Iterable, List,
                    Mapping, Optional, Set, Tuple, Type, Union)
from weakref import ReferenceType

import torch
import vllm.envs as envs
from saturo.request_metadata import is_decode_phase, is_rt_request
from vllm.config import (DecodingConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, VllmConfig)
from vllm.core.scheduler import SchedulerOutputs
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_timeout import asyncio_timeout
from vllm.engine.llm_engine import LLMEngine, SchedulerOutputState
from vllm.engine.metrics_types import StatLoggerBase
from vllm.engine.protocol import EngineClient
from vllm.executor.executor_base import ExecutorBase
from vllm.inputs import PromptType
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.guided_decoding import (
    get_guided_decoding_logits_processor)
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.sequence import ExecuteModelRequest
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Device, weak_bind

from vllm.global_experiment_config import settings

logger = init_logger(__name__)
ENGINE_ITERATION_TIMEOUT_S = envs.VLLM_ENGINE_ITERATION_TIMEOUT_S


def _emit_optional_request_stage(engine: Any, request_id: str,
                                 stage: str) -> None:
    """Notify an optional embedding-specific admission diagnostics hook."""
    callback = getattr(engine, "_saturo_request_stage_callback", None)
    if not callable(callback):
        return
    try:
        callback(request_id, stage, time.perf_counter())
    except Exception:
        logger.exception("Saturo request-stage callback failed")


class AsyncEngineDeadError(RuntimeError):
    pass


def _log_task_completion(task: asyncio.Task,
                         error_callback: Callable[[Exception], None]) -> None:
    """This function is only intended for the `engine.run_engine_loop()` task.

    In particular, that task runs a `while True` loop that can only exit if
    there is an exception.
    """

    exception = None
    try:
        return_value = task.result()
        raise AssertionError(
            f"The engine background task should never finish without an "
            f"exception. {return_value}")
    except asyncio.exceptions.CancelledError:
        # We assume that if the task is cancelled, we are gracefully shutting
        # down. This should only happen on program exit.
        logger.info("Engine is gracefully shutting down.")
    except Exception as e:
        exception = e
        logger.error("Engine background task failed", exc_info=e)
        error_callback(exception)
        raise AsyncEngineDeadError(
            "Task finished unexpectedly. This should never happen! "
            "Please open an issue on GitHub. See stack trace above for the "
            "actual cause.") from e


STOP_ITERATION = Exception()  # Sentinel


class AsyncStream:
    """A stream of RequestOutputs or PoolingRequestOutputs for a request
    that can be iterated over asynchronously via an async generator."""

    def __init__(self, request_id: str, cancel: Callable[[str], None]) -> None:
        self.request_id = request_id
        self._cancel = cancel
        self._queue: asyncio.Queue = asyncio.Queue()
        self._finished = False

    def put(self, item: Union[RequestOutput, PoolingRequestOutput,
                              Exception]) -> None:
        if not self._finished:
            if (os.environ.get("SATURO_STEP_TIMING", "").strip().lower()
                    in ("1", "true", "yes", "on")):
                try:
                    setattr(item, "_saturo_stream_put_s", time.perf_counter())
                except Exception:
                    pass
            self._queue.put_nowait(item)

    def finish(
        self,
        exception: Optional[Union[BaseException, Type[BaseException]]] = None,
    ) -> None:
        if not self._finished:
            self._finished = True
            self._queue.put_nowait(
                exception if self._is_raisable(exception) else STOP_ITERATION)

    @property
    def finished(self) -> bool:
        return self._finished

    async def generator(
        self
    ) -> AsyncGenerator[Union[RequestOutput, PoolingRequestOutput], None]:
        try:
            while True:
                result = await self._queue.get()
                if self._is_raisable(result):
                    if result == STOP_ITERATION:
                        return
                    raise result
                if (os.environ.get("SATURO_STEP_TIMING", "").strip().lower()
                        in ("1", "true", "yes", "on")):
                    try:
                        setattr(result, "_saturo_stream_get_s",
                                time.perf_counter())
                    except Exception:
                        pass
                yield result
        except GeneratorExit:
            print(f"GeneratorExit Error")
            self._cancel(self.request_id)
            raise asyncio.CancelledError from None

    @staticmethod
    def _is_raisable(value: Any):
        return isinstance(value, BaseException) or \
                (isinstance(value, type) and \
                 issubclass(value, BaseException))


class RequestTracker:
    """Synchronous abstraction for tracking requests."""

    def __init__(self) -> None:
        self._request_streams: Dict[str, AsyncStream] = {}
        self._aborted_requests: asyncio.Queue[str] = asyncio.Queue()
        self._new_requests: asyncio.Queue[Tuple[AsyncStream,
                                                dict]] = asyncio.Queue()
        self.new_requests_event = asyncio.Event()

    def __contains__(self, item):
        return item in self._request_streams

    def __len__(self) -> int:
        return len(self._request_streams)

    def propagate_exception(self,
                            exc: Exception,
                            request_id: Optional[str] = None) -> None:
        """Propagate an exception to request streams
        (all if request_id is None)."""
        if request_id is not None:
            self.abort_request(request_id, exception=exc)
        else:
            # NB: tuple() used here because self.abort_request pops the stream
            # out of self._request_streams, so we can't iterate on it directly
            for rid in tuple(self._request_streams.keys()):
                self.abort_request(rid, exception=exc)

    def process_request_output(self,
                               request_output: Union[RequestOutput,
                                                     PoolingRequestOutput],
                               *,
                               verbose: bool = False) -> None:
        """Process a request output from the engine."""
        request_id = request_output.request_id
        finished = request_output.finished

        if finished:
            stream = self._request_streams.pop(request_id, None)
        else:
            stream = self._request_streams.get(request_id)
        # Guard against a KeyError which can occur if the request was aborted
        # while the output was generated
        if stream is not None:
            stream.put(request_output)
            if finished:
                stream.finish()

        if verbose and finished:
            logger.info("Finished request %s.", request_id)

    def process_exception(self,
                          request_id: str,
                          exception: BaseException,
                          *,
                          verbose: bool = False) -> None:
        """Propagate an exception from the engine."""
        print(f"abort request: {request_id}")
        if verbose:
            logger.info("Finished request %s.", request_id)
        self.abort_request(request_id, exception=exception)

    def add_request(self,
                    request_id: str,
                    *,
                    verbose: bool = False,
                    **engine_add_request_kwargs) -> AsyncStream:
        """Add a request to be sent to the engine on the next background
        loop iteration."""
        if request_id in self._request_streams:
            raise KeyError(f"Request {request_id} already exists.")

        abort_request = partial(self.abort_request, verbose=verbose)
        stream = AsyncStream(request_id, abort_request)
        self._new_requests.put_nowait((stream, {
            "request_id": request_id,
            **engine_add_request_kwargs
        }))

        self.new_requests_event.set()

        if verbose:
            logger.info("Added request %s.", request_id)

        return stream

    def abort_request(self,
                      request_id: str,
                      *,
                      exception: Optional[Union[BaseException,
                                                Type[BaseException]]] = None,
                      verbose: bool = False) -> None:
        """Abort a request during next background loop iteration."""
        if verbose:
            logger.info("Aborted request %s.", request_id)
        print(f"aborted request: {request_id}")
        self._aborted_requests.put_nowait(request_id)

        stream = self._request_streams.pop(request_id, None)
        if stream is not None:
            stream.finish(exception=exception)

    def get_new_and_aborted_requests(self) -> Tuple[List[Dict], Set[str]]:
        """Get the new requests and finished requests to be
        sent to the engine."""
        new_requests: List[Dict] = []
        finished_requests: Set[str] = set()

        while not self._aborted_requests.empty():
            request_id = self._aborted_requests.get_nowait()
            finished_requests.add(request_id)

        while not self._new_requests.empty():
            stream, new_request = self._new_requests.get_nowait()
            request_id = stream.request_id
            if request_id in finished_requests:
                # The request has already been aborted.
                stream.finish(asyncio.CancelledError)
                finished_requests.discard(request_id)
            else:
                self._request_streams[request_id] = stream
                new_requests.append(new_request)

        return new_requests, finished_requests

    async def wait_for_new_requests(self):
        if not self.has_new_requests():
            await self.new_requests_event.wait()
        self.new_requests_event.clear()

    def has_new_requests(self):
        return not self._new_requests.empty()


def append_free_gpu_block(csv_dir: str, virtual_engine: int, free_blocks: int) -> str:
    os.makedirs(csv_dir, exist_ok=True)
    path = os.path.join(csv_dir, f"free_gpu_block_ve{virtual_engine}.csv")

    file_exists = os.path.exists(path) and os.path.getsize(path) > 0

    with open(path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["ts_unix", "free_gpu_blocks"])
        writer.writerow([int(time.time()), free_blocks])

    return path

class _AsyncLLMEngine(LLMEngine):
    """Extension of LLMEngine to add async methods."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.free_gpu_block_iteration:dict[int, List[int]] = defaultdict(list)
        self.num_iteration = 0
        self.cnt = 0
        self._saturo_step_timing = (
            os.environ.get("SATURO_STEP_TIMING", "").strip().lower()
            in ("1", "true", "yes", "on"))
        self._saturo_async_output_phase_logs: List[Dict[str, Any]] = []
        self._saturo_stream_dispatch_logs: List[Dict[str, Any]] = []

    def _process_model_outputs(self, *args, **kwargs) -> None:
        """Optionally time vLLM's asynchronous output-processing callback."""
        if not getattr(self, "_saturo_step_timing", False):
            return super()._process_model_outputs(*args, **kwargs)
        ctx = kwargs.get("ctx")
        queue_depth_before = len(ctx.output_queue) if ctx is not None else -1
        dispatch_start = len(self._saturo_stream_dispatch_logs)
        started_s = time.perf_counter()
        try:
            return super()._process_model_outputs(*args, **kwargs)
        finally:
            finished_s = time.perf_counter()
            dispatch_logs = self._saturo_stream_dispatch_logs[dispatch_start:]
            self._saturo_async_output_phase_logs.append({
                "started_s": started_s,
                "finished_s": finished_s,
                "total_ms": (finished_s - started_s) * 1000.0,
                "stream_dispatch_ms": sum(
                    float(item.get("total_ms", 0.0))
                    for item in dispatch_logs),
                "stream_dispatch_outputs": sum(
                    int(item.get("outputs", 0)) for item in dispatch_logs),
                "queue_depth_before": queue_depth_before,
            })

    async def step_async(
        self, virtual_engine: int
    ) -> List[Union[RequestOutput, PoolingRequestOutput]]:
        """Performs one decoding iteration and returns newly generated results.
        The workers are ran asynchronously if possible.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        # these are cached outputs from previous iterations. None if on first
        # iteration

        step_wall_start = time.perf_counter()
        phase_cursor_s = step_wall_start
        phase_times_ms: Dict[str, float] = {}
        async_output_log_start = len(self._saturo_async_output_phase_logs)
        stream_dispatch_log_start = len(self._saturo_stream_dispatch_logs)

        def finish_phase(name: str) -> None:
            nonlocal phase_cursor_s
            now_s = time.perf_counter()
            phase_times_ms[name] = (now_s - phase_cursor_s) * 1000.0
            phase_cursor_s = now_s

        scheduler_wall_ms = 0.0
        current_iteration_log = None
        cached_outputs = self.cached_scheduler_outputs[virtual_engine]
        seq_group_metadata_list = cached_outputs.seq_group_metadata_list
        scheduler_outputs = cached_outputs.scheduler_outputs
        allow_async_output_proc = cached_outputs.allow_async_output_proc

        ctx = self.scheduler_contexts[virtual_engine]
            
        # Clear outputs for each new scheduler iteration
        ctx.request_outputs.clear()

        # skip the scheduler if there are any remaining steps in the seq groups.
        # This ensures that the scheduler is only called again when the current
        # batch has completed.
        if not self._has_remaining_steps(seq_group_metadata_list):

            # Schedule iteration
            scheduler_wall_start = time.perf_counter()
            phase_times_ms["pre_schedule_ms"] = (
                scheduler_wall_start - phase_cursor_s) * 1000.0
            (seq_group_metadata_list, scheduler_outputs,
             allow_async_output_proc
             ) = self.scheduler[virtual_engine].schedule()
            scheduler_wall_ms = (
                time.perf_counter() - scheduler_wall_start) * 1000.0
            phase_cursor_s = time.perf_counter()
            phase_times_ms["schedule_ms"] = scheduler_wall_ms

            request_id_list = []
            for seq in seq_group_metadata_list:
                request_id_list.append(seq.request_id)

            ctx.seq_group_metadata_list = seq_group_metadata_list
            ctx.scheduler_outputs = scheduler_outputs

            if not scheduler_outputs.is_empty():
                # this will cause mamba_cache/minimax_cache failed
                # to release finished_requests_ids of the last steps
                finished_requests_ids = self.scheduler[
                    virtual_engine].get_and_reset_finished_requests_ids()

            # Maybe switch from async mode to sync mode
            if not allow_async_output_proc and len(ctx.output_queue) > 0:
                self._process_model_outputs(ctx=ctx)

            if (self.scheduler_config.is_multi_step
                    and scheduler_outputs.num_lookahead_slots > 0):
                # cache the scheduler outputs for the next iteration if we have
                # lookahead slots
                self._cache_scheduler_outputs_for_multi_step(
                    virtual_engine, seq_group_metadata_list, scheduler_outputs,
                    allow_async_output_proc)
        else:
            finished_requests_ids = list()

        assert seq_group_metadata_list is not None
        assert scheduler_outputs is not None

        if not scheduler_outputs.is_empty():

            # Check if we have a cached last_output from the previous iteration.
            # For supporting PP this is probably the best way to pass the
            # sampled_token_ids, as a separate broadcast over all the PP stages
            # will cause one virtual engine's microbatch to block the pipeline.
            last_sampled_token_ids = \
                self._get_last_sampled_token_ids(virtual_engine)

            execute_model_req = ExecuteModelRequest(
                seq_group_metadata_list=seq_group_metadata_list,
                blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy=scheduler_outputs.blocks_to_copy,
                virtual_engine=virtual_engine,
                num_lookahead_slots=scheduler_outputs.num_lookahead_slots,
                running_queue_size=scheduler_outputs.running_queue_size,
                finished_requests_ids=finished_requests_ids,
                # We use ExecuteModelRequest to pass the last sampled_token_ids
                # to each of the non-last PP stages for in-place prepare_input.
                last_sampled_token_ids=last_sampled_token_ids)

            if allow_async_output_proc:
                execute_model_req.async_callback = self.async_callbacks[
                    virtual_engine]

            # 这里根据scheduler的输出预测当前step的执行时间，结果要放到 predict_forward_ms 中作为参数，
            # 其中 prefill_reqs: List[Tuple[int, int]], decode_reqs: List[Tuple[int, int]]，xxx_reqs中每一项为 context_len, query_len，分别表示需要多少的上下文和生成多少的token。scheduler_outputs中应该包含了这些信息，可以直接传入time_predictor进行时间预测。
            # 对于prefill而言，context_len应该是当前chunk-prefill中已经算完的token数量，querylen为token-chunk这次token的数量；对于decode而言，context_len上下文的数量，query_len为当前的query token数量（一般情况下为1）
            prefill_reqs = []
            decode_reqs = []
            schedule_counts = {
                "rt_prefill_reqs": 0,
                "be_prefill_reqs": 0,
                "rt_decode_reqs": 0,
                "be_decode_reqs": 0,
                "be_dummy_handoff_reqs": 0,
            }
            schedule_tokens = {
                "rt_prefill_tokens": 0,
                "be_prefill_tokens": 0,
                "rt_decode_tokens": 0,
                "be_decode_tokens": 0,
                "be_dummy_handoff_tokens": 0,
            }
            for scheduled_seq_group, metadata in zip(
                    scheduler_outputs.scheduled_seq_groups,
                    seq_group_metadata_list):
                seq_group = scheduled_seq_group.seq_group
                chunk_size = scheduled_seq_group.token_chunk_size
                query_len = chunk_size or 1
                is_rt = is_rt_request(seq_group)
                is_prefill = metadata.is_prompt and not is_decode_phase(
                    seq_group)
                extra_args = (getattr(
                    seq_group.sampling_params, "extra_args", None) or {})
                is_dummy_handoff = (
                    not is_rt and is_prefill
                    and extra_args.get("saturo_dummy_kv_decode", False))
                req_key = ("be_dummy_handoff_reqs" if is_dummy_handoff else
                           ("rt_" if is_rt else "be_") + (
                               "prefill_reqs" if is_prefill
                               else "decode_reqs"))
                schedule_counts[req_key] += 1
                for seq in seq_group.get_seqs():
                    computed_tokens = seq.get_num_computed_tokens()
                    if is_prefill:
                        prefill_reqs.append(
                            (query_len, computed_tokens + query_len))
                    else:
                        decode_reqs.append((query_len, computed_tokens))
                    token_key = (
                        "be_dummy_handoff_tokens" if is_dummy_handoff else
                        ("rt_" if is_rt else "be_") + (
                            "prefill_tokens" if is_prefill
                            else "decode_tokens"))
                    schedule_tokens[token_key] += int(query_len)

            predicted_time = self._predict_p_pool_forward_ms(prefill_reqs,
                                                              decode_reqs)
            import os as _os
            _dbg_f = _os.environ.get("SATURO_DEBUG_KV_RECV_FILE", "")
            if _dbg_f:
                _n_rt_pf = _n_rt_pf_flag = _n_be_pf = _n_dec = 0
                for _ssg, _md in zip(
                        scheduler_outputs.scheduled_seq_groups,
                        seq_group_metadata_list):
                    _sg = _ssg.seq_group
                    _rt = is_rt_request(_sg)
                    _pf = _md.is_prompt and not is_decode_phase(_sg)
                    _ea = (getattr(
                        _sg.sampling_params, "extra_args", None) or {})
                    _flag = bool(_ea.get("saturo_dummy_kv_decode", False))
                    if _pf and _rt:
                        _n_rt_pf += 1
                        if _flag:
                            _n_rt_pf_flag += 1
                    elif _pf:
                        _n_be_pf += 1
                    else:
                        _n_dec += 1
                if _n_rt_pf > 0:
                    with open(_dbg_f, "a") as _fh:
                        _fh.write(
                            "ENGINE_ITER rt_prefill=%d rt_prefill_with_flag=%d "
                            "be_prefill=%d decode=%d pure_prefill_batch=%s\n"
                            % (_n_rt_pf, _n_rt_pf_flag, _n_be_pf, _n_dec,
                               _n_dec == 0))
            # print(f"predicted_time: {predicted_time} ms, prefill_reqs: {prefill_reqs}, decode_reqs: {decode_reqs}")

            is_be_batch = self._is_be_batch(scheduler_outputs)
            if is_be_batch:
                self._setup_be_preemption(predicted_time or 0.0)

            # Execute the model.
            try:
                finish_phase("prepare_execute_ms")
                pre_cuda_sync_started_s = time.perf_counter()
                torch.cuda.synchronize()
                pre_cuda_sync_finished_s = time.perf_counter()
                phase_times_ms["pre_execute_cuda_sync_ms"] = (
                    pre_cuda_sync_finished_s - pre_cuda_sync_started_s
                ) * 1000.0
                iter_start = torch.cuda.Event(enable_timing=True)
                iter_end = torch.cuda.Event(enable_timing=True)
                iter_start.record()
                execute_await_started_s = time.perf_counter()
                outputs = await self.model_executor.execute_model_async(
                    execute_model_req)
                execute_await_finished_s = time.perf_counter()
                phase_times_ms["execute_model_await_ms"] = (
                    execute_await_finished_s - execute_await_started_s
                ) * 1000.0
                iter_end.record()
                post_cuda_sync_started_s = time.perf_counter()
                torch.cuda.synchronize()
                post_cuda_sync_finished_s = time.perf_counter()
                phase_times_ms["post_execute_cuda_sync_ms"] = (
                    post_cuda_sync_finished_s - post_cuda_sync_started_s
                ) * 1000.0
                phase_cursor_s = post_cuda_sync_finished_s
                actual_time = iter_start.elapsed_time(iter_end)
            finally:
                if is_be_batch:
                    self._teardown_be_preemption()

            if outputs is None:
                logger.info("P-pool: BE chunk preempted mid-forward (async), "
                            "rolling back BE scheduler state")
                self._abort_p_pool_scheduled_be(
                    virtual_engine, scheduler_outputs, ctx)
                return []

            module_profile_times = next((
                dict(times)
                for output in outputs
                if (times := getattr(output, "module_profile_times", None))
            ), None)
            prefill_tokens = sum(q for q, _ in prefill_reqs)
            decode_tokens = sum(q for q, _ in decode_reqs)
            scheduler = self.scheduler[virtual_engine]
            num_gpu_blocks_total = int(self.cache_config.num_gpu_blocks or 0)
            try:
                num_free_gpu_blocks = int(
                    scheduler.block_manager.get_num_free_gpu_blocks())
            except Exception:
                num_free_gpu_blocks = num_gpu_blocks_total
            decision = getattr(
                getattr(scheduler, "decode_saturation", None),
                "last_decision",
                None,
            )
            decision_payload = None
            if decision is not None:
                decision_payload = {
                    "enabled": bool(decision.enabled),
                    "t_rate_ms": float(decision.t_rate_ms),
                    "target_interval_ms": float(
                        decision.target_interval_ms),
                    "rt_only_tbt_ms": float(decision.rt_only_tbt_ms),
                    "predicted_peak_rt_blocks": int(
                        decision.predicted_peak_rt_blocks),
                    "current_rt_blocks": int(decision.current_rt_blocks),
                    "current_be_blocks": int(decision.current_be_blocks),
                    "be_block_budget": int(decision.be_block_budget),
                    "max_be_prefill_tokens": int(
                        decision.max_be_prefill_tokens),
                }
            current_iteration_log = {
                "iter_id": len(self.iteration_logs),
                "n_prefill_reqs": len(prefill_reqs),
                "n_decode_reqs": len(decode_reqs),
                **schedule_counts,
                **schedule_tokens,
                "n_rt_prefill_reqs": schedule_counts["rt_prefill_reqs"],
                "n_be_prefill_reqs": schedule_counts["be_prefill_reqs"],
                "n_rt_decode_reqs": schedule_counts["rt_decode_reqs"],
                "n_be_decode_reqs": schedule_counts["be_decode_reqs"],
                "prefill_tokens": prefill_tokens,
                "decode_tokens": decode_tokens,
                "total_tokens": prefill_tokens + decode_tokens,
                "prefill_sum_ctx": sum(
                    q * max(0, c - q) + q * (q + 1.0) / 2.0
                    for q, c in prefill_reqs),
                "decode_sum_ctx": sum(q * c for q, c in decode_reqs),
                "prefill_reqs": list(prefill_reqs),
                "decode_reqs": list(decode_reqs),
                "actual_time_ms": actual_time,
                "module_profile_times": module_profile_times,
                "predicted_time_ms": predicted_time,
                "scheduler_wall_ms": scheduler_wall_ms,
                "num_gpu_blocks_total": num_gpu_blocks_total,
                "num_free_gpu_blocks": num_free_gpu_blocks,
                "num_used_gpu_blocks": max(
                    0, num_gpu_blocks_total - num_free_gpu_blocks),
                "preempted": int(scheduler_outputs.preempted),
                "blocks_to_swap_in": len(
                    scheduler_outputs.blocks_to_swap_in),
                "blocks_to_swap_out": len(
                    scheduler_outputs.blocks_to_swap_out),
                "decode_saturation_decision": decision_payload,
                "planner_timing": (getattr(getattr(scheduler, "decode_saturation", None),
                                           "last_phase_timings", None)
                                   if self._saturo_step_timing else None),
                "scheduler_running_queue_size": (
                    scheduler_outputs.running_queue_size),
                "scheduler_swapped_queue_size": len(scheduler.swapped),
                "scheduler_waiting_queue_size": len(scheduler.waiting),
                "scheduler_waiting_rt_decode_admission_size": len(
                    scheduler.waiting_rt_decode_admission),
                "scheduler_waiting_be_decode_admission_size": len(
                    scheduler.waiting_be_decode_admission),
                "scheduler_num_prefill_groups": (
                    scheduler_outputs.num_prefill_groups),
            }
            self.iteration_logs.append(current_iteration_log)
            finish_phase("iteration_log_build_ms")

            # we need to do this here so that last step's sampled_token_ids can
            # be passed to the next iteration for PP.
            if self.scheduler_config.is_multi_step:
                self._update_cached_scheduler_output(virtual_engine, outputs)
        else:
            if len(ctx.output_queue) > 0:
                self._process_model_outputs(ctx=ctx)
            outputs = []

        # Finish the current step for all the sequence groups.
        if self.scheduler_config.is_multi_step:
            for seq_group in seq_group_metadata_list:
                seq_group.finish_step()

        if not self._has_remaining_steps(seq_group_metadata_list):
            # Clear the cache if we have finished all the steps
            if self.scheduler_config.is_multi_step:
                self.cached_scheduler_outputs[
                    virtual_engine] = SchedulerOutputState()

            # is_first_step_output is True only when the num_steps of all
            # the sequences are 1. When the num_steps > 1,
            # multi_step_model_runner does the first-step output append.
            is_first_step_output: bool = False if not seq_group_metadata_list \
                else seq_group_metadata_list[0].state.num_steps == 1

            finish_phase("post_model_state_ms")
            ctx.append_output(outputs=outputs,
                              seq_group_metadata_list=seq_group_metadata_list,
                              scheduler_outputs=scheduler_outputs,
                              is_async=allow_async_output_proc,
                              is_last_step=True,
                              is_first_step_output=is_first_step_output)
            finish_phase("append_output_ms")

            if outputs and allow_async_output_proc:
                assert len(
                    outputs
                ) == 1, "Async postprocessor expects only a single output set"
                self._advance_to_next_step(
                    outputs[0], seq_group_metadata_list,
                    scheduler_outputs.scheduled_seq_groups)
            finish_phase("advance_to_next_step_ms")

            if not allow_async_output_proc:
                self._process_model_outputs(ctx=ctx)

                # Log stats.
                self.do_log_stats(scheduler_outputs, outputs)

                # Tracing
                self.do_tracing(scheduler_outputs)
            finish_phase("sync_output_process_and_stats_ms")

        else:
            # Multi-step case
            return ctx.request_outputs

        if not self.has_unfinished_requests():
            # Drain async postprocessor (if exists)
            if len(ctx.output_queue) > 0:
                self._process_model_outputs(ctx=ctx)
            assert len(ctx.output_queue) == 0
        
        if current_iteration_log is not None:
            finish_phase("step_finalize_ms")
            step_wall_ms = (
                time.perf_counter() - step_wall_start) * 1000.0
            current_iteration_log["step_wall_time_ms"] = step_wall_ms
            current_iteration_log["runtime_overhead_ms"] = max(
                0.0,
                step_wall_ms
                - float(current_iteration_log.get("actual_time_ms", 0.0)),
            )
            if self._saturo_step_timing:
                callback_logs = self._saturo_async_output_phase_logs[
                    async_output_log_start:]
                dispatch_logs = self._saturo_stream_dispatch_logs[
                    stream_dispatch_log_start:]
                current_iteration_log.update({
                    "step_started_s": step_wall_start,
                    "step_finished_s": time.perf_counter(),
                    **phase_times_ms,
                    "async_output_callback_count": len(callback_logs),
                    "async_output_callback_ms": sum(
                        float(item.get("total_ms", 0.0))
                        for item in callback_logs),
                    "async_output_core_ms": sum(
                        max(0.0,
                            float(item.get("total_ms", 0.0))
                            - float(item.get("stream_dispatch_ms", 0.0)))
                        for item in callback_logs),
                    "stream_dispatch_calls": len(dispatch_logs),
                    "stream_dispatch_outputs": sum(
                        int(item.get("outputs", 0))
                        for item in dispatch_logs),
                    "stream_dispatch_ms": sum(
                        float(item.get("total_ms", 0.0))
                        for item in dispatch_logs),
                })
        return ctx.request_outputs

    async def stop_remote_worker_execution_loop_async(self) -> None:
        """Stop the remote worker execution loop."""
        await self.model_executor.stop_remote_worker_execution_loop_async()

    async def get_tokenizer_async(self,
                                  lora_request: Optional[LoRARequest] = None
                                  ) -> AnyTokenizer:
        return await (
            self.get_tokenizer_group().get_lora_tokenizer_async(lora_request))

    async def add_request_async(
        self,
        request_id: str,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
        data_parallel_rank: Optional[int] = None,
    ) -> None:
        """
        Async version of
        [`add_request`][vllm.engine.llm_engine.LLMEngine.add_request].
        """
        if lora_request is not None and not self.lora_config:
            raise ValueError(f"Got lora_request {lora_request} but LoRA is "
                             "not enabled!")
        if priority != 0 and not self.scheduler_config.policy == "priority":
            raise ValueError(f"Got priority {priority} but "
                             "Priority scheduling is not enabled.")
        if arrival_time is None:
            arrival_time = time.time()

        if data_parallel_rank is not None:
            raise ValueError("Targeting data_parallel_rank only supported "
                             "in v1 client.")

        if (isinstance(prompt, dict)
                and prompt.get("prompt_embeds", None) is not None
                and not prompt.get("prompt_token_ids", None)):
            # We use the -2 dimension (instead of 0) in case a batched input
            # of batch size 1 is passed in.
            prompt["prompt_token_ids"] = [0
                                          ] * prompt["prompt_embeds"].shape[-2]

        processed_inputs = await self.input_preprocessor.preprocess_async(
            prompt,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
        )

        if isinstance(params, SamplingParams) and \
            params.guided_decoding is not None:
            # Guided decoding has an async implementation for building logits
            # processors in a separate threadpool.
            # We want to invoke that here instead of using the blocking
            # implementation in the LLMEngine
            params = await build_guided_decoding_logits_processor_async(
                sampling_params=params,
                tokenizer=await self.get_tokenizer_async(lora_request),
                default_guided_backend=self.decoding_config.backend,
                reasoning_backend=self.decoding_config.reasoning_backend,
                model_config=self.model_config)

        self._add_processed_request(
            request_id=request_id,
            processed_inputs=processed_inputs,
            params=params,
            arrival_time=arrival_time,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            trace_headers=trace_headers,
            priority=priority,
        )

    async def check_health_async(self) -> None:
        self.model_executor.check_health()

    async def collective_rpc_async(self,
                                   method: str,
                                   timeout: Optional[float] = None,
                                   args: tuple = (),
                                   kwargs: Optional[dict] = None):
        raise NotImplementedError


async def build_guided_decoding_logits_processor_async(
        sampling_params: SamplingParams, tokenizer: AnyTokenizer,
        default_guided_backend: str, reasoning_backend: Optional[str],
        model_config: ModelConfig) -> SamplingParams:
    """Constructs logits processors based on the guided_decoding,
    logits_bias, and allowed_token_ids fields in sampling_params. Deletes
    those fields and adds the constructed logits processors to the
    logits_processors field. Modifies sampling params in-place and returns
    the modified sampling params."""
    if sampling_params.guided_decoding is None:
        return sampling_params

    # Defensively copy sampling params since guided decoding logits
    # processors can have different state for each request
    sampling_params = copy.copy(sampling_params)
    guided_decoding = sampling_params.guided_decoding

    logger.debug(
        "Building guided decoding logits processor. "
        "guided_decoding: %s%s", guided_decoding,
        f", reasoning_backend: {reasoning_backend}"
        if reasoning_backend is not None else "")

    guided_decoding.backend = guided_decoding.backend or default_guided_backend

    processor = await get_guided_decoding_logits_processor(
        guided_params=guided_decoding,
        tokenizer=tokenizer,
        reasoning_backend=reasoning_backend,
        model_config=model_config)

    if processor:
        if sampling_params.logits_processors is None:
            sampling_params.logits_processors = []
        sampling_params.logits_processors.append(processor)

    # Unset guided decoding params after constructing the lp from them
    sampling_params.guided_decoding = None

    return sampling_params


class AsyncLLMEngine(EngineClient):
    """An asynchronous wrapper for [`LLMEngine`][vllm.LLMEngine].

    This class is used to wrap the [`LLMEngine`][vllm.LLMEngine] class to
    make it asynchronous. It uses asyncio to create a background loop that keeps
    processing incoming requests. The [`LLMEngine`][vllm.LLMEngine] is kicked
    by the generate method when there are requests in the waiting queue. The
    generate method yields the outputs from the [`LLMEngine`][vllm.LLMEngine]
    to the caller.

    Args:
        log_requests: Whether to log the requests.
        start_engine_loop: If True, the background task to run the engine
            will be automatically started in the generate call.
        *args: Arguments for [`LLMEngine`][vllm.LLMEngine].
        **kwargs: Arguments for [`LLMEngine`][vllm.LLMEngine].
    """

    _engine_class: Type[_AsyncLLMEngine] = _AsyncLLMEngine

    def __init__(self,
                 *args,
                 log_requests: bool = True,
                 start_engine_loop: bool = True,
                 **kwargs) -> None:
        if envs.VLLM_USE_V1:
            raise ValueError(
                "Using V0 AsyncLLMEngine, but envs.VLLM_USE_V1=True. "
                "This should not happen. As a workaround, try using "
                "AsyncLLMEngine.from_vllm_config(...) or explicitly set "
                "VLLM_USE_V1=0 or 1 and report this issue on Github.")

        self.log_requests = log_requests
        self.engine = self._engine_class(*args, **kwargs)

        # This ensures quick processing of request outputs
        # so the append to asyncio queues is not delayed,
        # especially for multi-step.
        self.use_process_request_outputs_callback = (
            self.engine.model_config.use_async_output_proc)

        if self.use_process_request_outputs_callback:
            self.engine.process_request_outputs_callback = \
                weak_bind(self.process_request_outputs)

        self.background_loop: Optional[asyncio.Future] = None
        # We need to keep a reference to unshielded
        # task as well to prevent it from being garbage
        # collected
        self._background_loop_unshielded: Optional[asyncio.Task] = None
        self.start_engine_loop = start_engine_loop
        self._errored_with: Optional[BaseException] = None
        self._saturo_last_engine_step_finished_s: Optional[float] = None

        # Lazy initialized fields
        self._request_tracker: RequestTracker

    def __del__(self):
        if rt := getattr(self, "request_tracker", None):
            # Wake up engine loop so that it will exit cleanly
            rt.new_requests_event.set()

    @classmethod
    def _get_executor_cls(cls,
                          engine_config: VllmConfig) -> Type[ExecutorBase]:
        return LLMEngine._get_executor_cls(engine_config)

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[dict[str, StatLoggerBase]] = None,
        disable_log_requests: bool = False,
        disable_log_stats: bool = False,
    ) -> "AsyncLLMEngine":
        """Create an AsyncLLMEngine from the EngineArgs."""

        return cls(
            vllm_config=vllm_config,
            executor_class=cls._get_executor_cls(vllm_config),
            start_engine_loop=start_engine_loop,
            log_requests=not disable_log_requests,
            log_stats=not disable_log_stats,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
    ) -> "AsyncLLMEngine":
        """Creates an async LLM engine from the engine arguments."""

        vllm_config = engine_args.create_engine_config(usage_context)

        async_engine_cls = cls
        if envs.VLLM_USE_V1:
            from vllm.v1.engine.async_llm import AsyncLLM as V1AsyncLLMEngine
            async_engine_cls = V1AsyncLLMEngine

        return async_engine_cls.from_vllm_config(
            vllm_config=vllm_config,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            disable_log_stats=engine_args.disable_log_stats,
            disable_log_requests=engine_args.disable_log_requests,
        )

    @property
    def is_running(self) -> bool:
        return (self.background_loop is not None
                and self._background_loop_unshielded is not None
                and not self._background_loop_unshielded.done())

    @property
    def is_stopped(self) -> bool:
        return self.errored or (self.background_loop is not None and
                                self._background_loop_unshielded is not None
                                and self._background_loop_unshielded.done())

    @property
    def errored(self) -> bool:
        return self._errored_with is not None

    @property
    def dead_error(self) -> BaseException:
        return AsyncEngineDeadError(
            "Background loop is not running. If it was running, "
            "inspect the output to find the stacktrace of the "
            "error that caused the background loop to stop "
            "(AsyncEngineDeadError).")

    def set_errored(self, exc: Exception) -> None:
        self._errored_with = exc

    def _error_callback(self, exc: Exception) -> None:
        self.set_errored(exc)
        self._request_tracker.propagate_exception(exc)

    async def get_input_preprocessor(self) -> InputPreprocessor:
        return self.engine.input_preprocessor

    async def get_tokenizer(
        self,
        lora_request: Optional[LoRARequest] = None,
    ) -> AnyTokenizer:
        return await self.engine.get_tokenizer_async(lora_request)

    def start_background_loop(self) -> None:
        """Start the background loop."""
        if self.errored:
            raise AsyncEngineDeadError(
                "Background loop has errored already.") from self._errored_with
        if self.is_running:
            raise RuntimeError("Background loop is already running.")
        # Initialize the RequestTracker here so it uses the right event loop.
        self._request_tracker = RequestTracker()

        self._background_loop_unshielded = asyncio.get_event_loop(
        ).create_task(self.run_engine_loop(weakref.ref(self)))
        self._background_loop_unshielded.add_done_callback(
            partial(_log_task_completion, error_callback=self._error_callback))
        self.background_loop = asyncio.shield(self._background_loop_unshielded)

    def shutdown_background_loop(self) -> None:
        """
        Shut down the background loop.

        This method needs to be called during cleanup to remove
        references to `self` and properly GC the resources held
        by the async LLM engine (e.g., the executors as well as
        their resources).
        """
        if self._background_loop_unshielded is not None:
            self._background_loop_unshielded.cancel()
            self._background_loop_unshielded = None
        self.background_loop = None

    async def engine_step(self, virtual_engine: int) -> bool:
        """Kick the engine to process the waiting requests.

        Returns True if there are in-progress requests."""
        timing_enabled = getattr(self.engine, "_saturo_step_timing", False)
        outer_started_s = time.perf_counter()
        previous_finished_s = self._saturo_last_engine_step_finished_s
        loop_gap_ms = (
            (outer_started_s - previous_finished_s) * 1000.0
            if timing_enabled and previous_finished_s is not None else None)
        iteration_log_start = len(self.engine.iteration_logs)
        new_requests, aborted_requests = (
            self._request_tracker.get_new_and_aborted_requests())
        for new_request in new_requests:
            # Add the request into the vLLM engine's waiting queue.
            request_id = str(new_request["request_id"])
            _emit_optional_request_stage(
                self, request_id, "request_tracker_dequeued")
            try:
                await self.engine.add_request_async(**new_request)
                _emit_optional_request_stage(
                    self, request_id, "scheduler_added")
            except ValueError as e:
                # TODO: use a vLLM specific error for failed validation
                print("error:", repr(e))          # repr 比 str 信息更全一些
                traceback.print_exc()             # 打印完整堆栈到控制台

                self._request_tracker.process_exception(
                    new_request["request_id"],
                    e,
                    verbose=self.log_requests,
                )

        if aborted_requests:
            await self._engine_abort(aborted_requests)
        admission_finished_s = time.perf_counter()
        request_outputs = await self.engine.step_async(virtual_engine)
        step_async_finished_s = time.perf_counter()
        
        # Put the outputs into the corresponding streams.
        # If used as a callback, then already invoked inside
        # LLMEngine's _process_model_outputs
        if not self.use_process_request_outputs_callback:
            all_finished = self.process_request_outputs(request_outputs)
        else:
            # For callback case, we only need to detect when all
            # requests are finished
            all_finished = all(request_output.finished
                               for request_output in request_outputs)
        dispatch_finished_s = time.perf_counter()
        if timing_enabled and len(self.engine.iteration_logs) > iteration_log_start:
            iteration_log = self.engine.iteration_logs[-1]
            iteration_log.update({
                "outer_loop_gap_ms": loop_gap_ms,
                "outer_admission_ms": (
                    admission_finished_s - outer_started_s) * 1000.0,
                "outer_step_async_ms": (
                    step_async_finished_s - admission_finished_s) * 1000.0,
                "outer_stream_dispatch_ms": (
                    dispatch_finished_s - step_async_finished_s) * 1000.0,
                "outer_engine_step_total_ms": (
                    dispatch_finished_s - outer_started_s) * 1000.0,
                "outer_request_outputs": len(request_outputs),
            })
        self._saturo_last_engine_step_finished_s = dispatch_finished_s
        return not all_finished

    def process_request_outputs(self, request_outputs) -> bool:
        # Put the outputs into the corresponding streams.
        timing_enabled = getattr(self.engine, "_saturo_step_timing", False)
        dispatch_started_s = time.perf_counter() if timing_enabled else 0.0
        all_finished = True
        for request_output in request_outputs:
            self._request_tracker.process_request_output(
                request_output, verbose=self.log_requests)
            all_finished = all_finished and request_output.finished

        if timing_enabled:
            dispatch_finished_s = time.perf_counter()
            self.engine._saturo_stream_dispatch_logs.append({
                "started_s": dispatch_started_s,
                "finished_s": dispatch_finished_s,
                "total_ms": (
                    dispatch_finished_s - dispatch_started_s) * 1000.0,
                "outputs": len(request_outputs),
            })

        return all_finished

    async def _engine_abort(self, request_ids: Iterable[str]):
        self.engine.abort_request(request_ids)

    @staticmethod
    async def run_engine_loop(engine_ref: ReferenceType):
        """We use a weakref to the engine so that the running loop
        doesn't prevent the engine being garbage collected."""
        engine: Optional[AsyncLLMEngine] = engine_ref()
        if not engine:
            return

        pipeline_parallel_size = \
                engine.engine.parallel_config.pipeline_parallel_size
        has_requests_in_progress = [False] * pipeline_parallel_size
        while True:
            if not any(has_requests_in_progress):
                logger.debug("Waiting for new requests...")
                # Stop the execute model loop in parallel workers until there
                # are more requests to process. This avoids waiting
                # indefinitely in torch.distributed ops which may otherwise
                # timeout, and unblocks the RPC thread in the workers so that
                # they can process any other queued control plane messages,
                # such as add/remove lora adapters.
                await engine.engine.stop_remote_worker_execution_loop_async()
                request_tracker = engine._request_tracker
                # Allow engine to be garbage collected while
                # waiting for new requests
                del engine
                await asyncio.sleep(0)
                if engine_ref() is None:
                    return
                await request_tracker.wait_for_new_requests()
                engine = engine_ref()
                if not engine:
                    return
                logger.debug("Got new requests!")
                requests_in_progress = [
                    asyncio.create_task(engine.engine_step(ve))
                    for ve in range(pipeline_parallel_size)
                ]
                has_requests_in_progress = [True] * pipeline_parallel_size

            # Abort if iteration takes too long due to unrecoverable errors
            # (eg. NCCL timeouts).
            try:
                async with asyncio_timeout(ENGINE_ITERATION_TIMEOUT_S):
                    done, _ = await asyncio.wait(
                        requests_in_progress,
                        return_when=asyncio.FIRST_COMPLETED)
                    for _ in range(pipeline_parallel_size):
                        await asyncio.sleep(0)
                for task in done:
                    result = task.result()
                    virtual_engine = requests_in_progress.index(task)
                    has_unfinished_requests = (
                        engine.engine.
                        has_unfinished_requests_for_virtual_engine(
                            virtual_engine))
                    if result or has_unfinished_requests:
                        requests_in_progress[virtual_engine] = (
                            asyncio.create_task(
                                engine.engine_step(virtual_engine)))
                        has_requests_in_progress[virtual_engine] = True
                    else:
                        has_requests_in_progress[virtual_engine] = False
            except asyncio.TimeoutError as exc:
                logger.error(
                    "Engine iteration timed out. This should never happen!")
                engine.set_errored(exc)
                raise
            await asyncio.sleep(0)

    async def add_request(
        self,
        request_id: str,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
        data_parallel_rank: Optional[int] = None,
    ) -> AsyncGenerator[Union[RequestOutput, PoolingRequestOutput], None]:
        if not self.is_running:
            if self.start_engine_loop:
                self.start_background_loop()
            else:
                raise AsyncEngineDeadError(
                    "Background loop is not running. If it was running, "
                    "inspect the output to find the stacktrace of the "
                    "error that caused the background loop to stop "
                    "(AsyncEngineDeadError).")

        if (priority != 0
                and not self.engine.scheduler_config.policy == "priority"):
            raise ValueError(f"Got priority {priority} but "
                             "Priority scheduling is not enabled.")

        # An RT request must be able to interrupt an active P-pool BE forward
        # at its actual arrival time.  Waiting for engine_step() to move the
        # request from RequestTracker into _AsyncLLMEngine.add_request_async()
        # is too late: the active BE model step has already completed by then.
        # Notify the core engine before enqueueing the request so it can set the
        # model runner's thread-safe layer-boundary preemption event.
        p_pool_enabled = getattr(self.engine, "_p_pool_enabled", None)
        if (callable(p_pool_enabled) and p_pool_enabled()
                and isinstance(params, SamplingParams)
                and is_rt_request(params)
                and getattr(self.engine, "_executing_be", False)):
            prompt_len = None
            if isinstance(prompt, Mapping):
                prompt_token_ids = prompt.get("prompt_token_ids")
                if prompt_token_ids is not None:
                    prompt_len = len(prompt_token_ids)
                else:
                    prompt_embeds = prompt.get("prompt_embeds")
                    if prompt_embeds is not None:
                        prompt_len = int(prompt_embeds.shape[-2])
            self.engine._check_and_preempt_be(params, prompt_len=prompt_len)

        stream = self._request_tracker.add_request(
            request_id,
            verbose=self.log_requests,
            prompt=prompt,
            params=params,
            arrival_time=arrival_time or time.time(),
            lora_request=lora_request,
            trace_headers=trace_headers,
            prompt_adapter_request=prompt_adapter_request,
            priority=priority,
            data_parallel_rank=data_parallel_rank,
        )
        _emit_optional_request_stage(
            self, request_id, "request_tracker_enqueued")

        return stream.generator()

    async def generate(
        self,
        prompt: PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
        data_parallel_rank: Optional[int] = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Generate outputs for a request.

        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.

        Args:
            prompt: The prompt to the LLM. See
                [`PromptType`][vllm.inputs.PromptType] for more details about
                the format of each input.
            sampling_params: The sampling parameters of the request.
            request_id: The unique id of the request.
            lora_request: LoRA request to use for generation, if any.
            trace_headers: OpenTelemetry trace headers.
            prompt_adapter_request: Prompt Adapter request to use
                                            for generation, if any.
            priority: The priority of the request.
                Only applicable with priority scheduling.
            data_parallel_rank: The (global) data parallel rank that must
                handle this request. Only applicable if DP is enabled.
        Yields:
            The output `RequestOutput` objects from the LLMEngine
            for the request.

        Details:
            - If the engine is not running, start the background loop,
              which iteratively invokes
              [`engine_step`][vllm.engine.async_llm_engine.AsyncLLMEngine.engine_step]
              to process the waiting requests.
            - Add the request to the engine's `RequestTracker`.
              On the next background loop, this request will be sent to
              the underlying engine.
              Also, a corresponding `AsyncStream` will be created.
            - Wait for the request outputs from `AsyncStream` and yield them.

        Example:
            >>> # Please refer to entrypoints/api_server.py for
            >>> # the complete example.
            >>>
            >>> # initialize the engine and the example input
            >>> # note that engine_args here is AsyncEngineArgs instance
            >>> engine = AsyncLLMEngine.from_engine_args(engine_args)
            >>> example_input = {
            >>>     "prompt": "What is LLM?",
            >>>     "stream": False, # assume the non-streaming case
            >>>     "temperature": 0.0,
            >>>     "request_id": 0,
            >>> }
            >>>
            >>> # start the generation
            >>> results_generator = engine.generate(
            >>>    example_input["prompt"],
            >>>    SamplingParams(temperature=example_input["temperature"]),
            >>>    example_input["request_id"])
            >>>
            >>> # get the results
            >>> final_output = None
            >>> async for request_output in results_generator:
            >>>     if await request.is_disconnected():
            >>>         # Abort the request if the client disconnects.
            >>>         await engine.abort(request_id)
            >>>         # Return or raise an error
            >>>         ...
            >>>     final_output = request_output
            >>>
            >>> # Process and return the final output
            >>> ...
        """
        try:
            async for output in await self.add_request(
                    request_id,
                    prompt,
                    sampling_params,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    prompt_adapter_request=prompt_adapter_request,
                    priority=priority,
                    data_parallel_rank=data_parallel_rank,
            ):
                yield LLMEngine.validate_output(output, RequestOutput)
        except asyncio.CancelledError:
            print(f"asyncio.CancelledError")
            await self.abort(request_id)
            raise

    async def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        """Generate outputs for a request from a pooling model.

        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.

        Args:
            prompt: The prompt to the LLM. See
                [`PromptType`][vllm.inputs.PromptType] for more details about
                the format of each input.
            pooling_params: The pooling parameters of the request.
            request_id: The unique id of the request.
            lora_request: LoRA request to use for generation, if any.
            trace_headers: OpenTelemetry trace headers.
            priority: The priority of the request.
                Only applicable with priority scheduling.

        Yields:
            The output `PoolingRequestOutput` objects from the LLMEngine
            for the request.

        Details:
            - If the engine is not running, start the background loop,
                which iteratively invokes
                [`vllm.engine.async_llm_engine.AsyncLLMEngine.engine_step`][]
                to process the waiting requests.
            - Add the request to the engine's `RequestTracker`.
                On the next background loop, this request will be sent to
                the underlying engine.
                Also, a corresponding `AsyncStream` will be created.
            - Wait for the request outputs from `AsyncStream` and yield them.

        Example:
        ```
        # Please refer to entrypoints/api_server.py for
        # the complete example.
    
        # initialize the engine and the example input
        # note that engine_args here is AsyncEngineArgs instance
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        example_input = {
            "input": "What is LLM?",
            "request_id": 0,
        }
    
        # start the generation
        results_generator = engine.encode(
        example_input["input"],
        PoolingParams(),
        example_input["request_id"])
    
        # get the results
        final_output = None
        async for request_output in results_generator:
            if await request.is_disconnected():
                # Abort the request if the client disconnects.
                await engine.abort(request_id)
                # Return or raise an error
                ...
            final_output = request_output
    
        # Process and return the final output
        ...
        ```
        """
        try:
            async for output in await self.add_request(
                    request_id,
                    prompt,
                    pooling_params,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=priority,
            ):
                yield LLMEngine.validate_output(output, PoolingRequestOutput)
        except asyncio.CancelledError:
            await self.abort(request_id)
            raise

    async def abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        if not self.is_running:
            raise AsyncEngineDeadError(
                "Background loop is not running. If it was running, "
                "inspect the output to find the stacktrace of the "
                "error that caused the background loop to stop "
                "(AsyncEngineDeadError).")

        return self._abort(request_id)

    def _abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        self._request_tracker.abort_request(request_id,
                                            exception=asyncio.CancelledError,
                                            verbose=self.log_requests)

    async def get_vllm_config(self) -> VllmConfig:
        """Get the vllm configuration of the vLLM engine."""
        return self.engine.get_vllm_config()

    async def get_model_config(self) -> ModelConfig:
        """Get the model configuration of the vLLM engine."""
        return self.engine.get_model_config()

    async def get_parallel_config(self) -> ParallelConfig:
        """Get the parallel configuration of the vLLM engine."""
        return self.engine.get_parallel_config()

    async def get_decoding_config(self) -> DecodingConfig:
        """Get the decoding configuration of the vLLM engine."""
        return self.engine.get_decoding_config()

    async def get_scheduler_config(self) -> SchedulerConfig:
        """Get the scheduling configuration of the vLLM engine."""
        return self.engine.get_scheduler_config()

    async def get_lora_config(self) -> LoRAConfig:
        """Get the lora configuration of the vLLM engine."""
        return self.engine.get_lora_config()

    async def do_log_stats(
            self,
            scheduler_outputs: Optional[SchedulerOutputs] = None,
            model_output: Optional[List[SamplerOutput]] = None) -> None:
        self.engine.do_log_stats()

    async def check_health(self) -> None:
        """Raises an error if engine is unhealthy."""
        t = time.perf_counter()
        logger.debug("Starting health check...")
        if self.is_stopped:
            raise AsyncEngineDeadError("Background loop is stopped.")

        await self.engine.check_health_async()
        logger.debug("Health check took %fs", time.perf_counter() - t)

    async def is_tracing_enabled(self) -> bool:
        return self.engine.is_tracing_enabled()

    def add_logger(self, logger_name: str, logger: StatLoggerBase) -> None:
        self.engine.add_logger(logger_name=logger_name, logger=logger)

    def remove_logger(self, logger_name: str) -> None:
        self.engine.remove_logger(logger_name=logger_name)

    async def start_profile(self) -> None:
        self.engine.start_profile()

    async def stop_profile(self) -> None:
        self.engine.stop_profile()

    async def reset_mm_cache(self) -> None:
        self.engine.reset_mm_cache()

    async def reset_prefix_cache(self,
                                 device: Optional[Device] = None) -> None:
        self.engine.reset_prefix_cache(device)

    async def sleep(self, level: int = 1) -> None:
        self.engine.sleep(level)

    async def wake_up(self, tags: Optional[list[str]] = None) -> None:
        self.engine.wake_up(tags)

    async def is_sleeping(self) -> bool:
        return self.engine.is_sleeping()

    async def add_lora(self, lora_request: LoRARequest) -> None:
        self.engine.add_lora(lora_request)

    async def collective_rpc(self,
                             method: str,
                             timeout: Optional[float] = None,
                             args: tuple = (),
                             kwargs: Optional[dict] = None):
        """
        Perform a collective RPC call to the given path.
        """
        return await self.engine.collective_rpc_async(method, timeout, args,
                                                      kwargs)


# TODO(v1): Remove this class proxy when V1 goes default.
if envs.is_set("VLLM_USE_V1") and envs.VLLM_USE_V1:
    from vllm.v1.engine.async_llm import AsyncLLM

    AsyncLLMEngine = AsyncLLM  # type: ignore
