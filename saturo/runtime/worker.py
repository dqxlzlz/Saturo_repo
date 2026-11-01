"""Saturo's V0 AsyncLLMEngine worker and Ray actor definitions."""

from __future__ import annotations

import argparse
import asyncio
import copy
import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from saturo.runtime.workload import BE, DECODE, PREFILL, RT, TPUT
from saturo.runtime.pool_controls import pool_accepts_be
from saturo.runtime.kv_handoff import CONNECTOR, real_handoff


def _is_direct_decode_command(
    worker_role: str,
    cmd: dict[str, Any],
) -> bool:
    """Return whether a command represents prefilled D-pool work."""
    return bool(
        worker_role == DECODE
        and cmd["kind"] in (RT, BE)
        and cmd["phase"] == DECODE
    )


@dataclass
class WorkerState:
    worker_id: int
    role: str
    host: str
    config_name: str
    tensor_parallel_size: int
    pipeline_parallel_size: int
    stderr_file: Any
    actor: Any = None
    pending_commands: list[dict[str, Any]] = field(default_factory=list)
    active_attempts: dict[str, str] = field(default_factory=dict)
    preempting_attempts: set[str] = field(default_factory=set)
    ready: bool = False
    retired: bool = False

    # Kept for placement/result compatibility. Ray owns the real GPU binding.
    gpu_id: str = ""
    backend: str = "ray"
    runtime_profile_path: str = ""
    num_gpu_blocks: int = 0
    max_model_len: int | None = None
    block_size: int = 16
    max_num_seqs: int = 0
    role_switching_enabled: bool = False
    borrowed_role: bool = False
    decode_reservations: set[str] = field(default_factory=set)
    expert_parallel_size: int = 0


@dataclass(frozen=True)
class WorkerPlacement:
    role: str
    host: str
    gpu_id: str
    config_name: str = "tp1_pp1"
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 0  # 0 means disabled


def _configure_worker_environment(disable_inductor: bool) -> None:
    """Configure V0 before importing any vLLM engine modules."""
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    if project_root not in existing_pythonpath.split(":"):
        os.environ["PYTHONPATH"] = (f"{project_root}:{existing_pythonpath}"
                                    if existing_pythonpath else project_root)

    # Assignment is intentional: this worker is specifically a V0 worker.
    os.environ["VLLM_USE_V1"] = "0"
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if disable_inductor:
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")


def _build_engine_config(
    worker_args: dict[str, Any], ) -> tuple[dict[str, Any], type[Any], bool]:
    """Build the shared role-specific V0 engine configuration."""
    from vllm import SamplingParams
    from vllm.config import CompilationConfig, KVTransferConfig

    role = str(worker_args["worker_role"])
    model = str(worker_args["model"])
    if not model:
        raise ValueError("missing --model for Saturo worker")

    # A fixed D-pool does no request-scoped prefill preemption. Its decode
    # forward can safely replay CUDA graphs, removing native-op launch costs.
    # Workers that may switch roles retain the eager path used by P/T-pools.
    static_decode = (role == DECODE
                     and not worker_args.get("adaptive_role_switching", False))
    enforce_eager = (
        os.environ.get("SATURO_ENFORCE_EAGER", "0" if static_decode else "1")
        .lower() not in ("0", "false", "no"))
    decode_graphs = static_decode and not enforce_eager

    engine_kwargs: dict[str, Any] = {
        "model": model,
        "trust_remote_code": True,
        "skip_tokenizer_init": True,
        "enforce_eager": enforce_eager,
        "tensor_parallel_size": int(worker_args["tensor_parallel_size"]),
        "pipeline_parallel_size": int(worker_args["pipeline_parallel_size"]),
        "load_format": worker_args["load_format"],
        "dtype": worker_args["dtype"],
        "gpu_memory_utilization": float(worker_args["gpu_memory_utilization"]),
        "disable_custom_all_reduce": True,
        "scheduling_policy": "priority",
        "enable_chunked_prefill": role == PREFILL,
        "disable_log_requests": True,
        "disable_log_stats": True,
    }
    if decode_graphs:
        # Keep output handling identical to the measured eager path. Graph
        # replay is independent of asynchronous output processing.
        engine_kwargs["disable_async_output_proc"] = True
    hf_config_path = worker_args.get("hf_config_path")
    if hf_config_path:
        engine_kwargs["hf_config_path"] = str(hf_config_path)
    for optional_key in (
            "max_model_len",
            "max_num_seqs",
            "max_num_batched_tokens",
            "swap_space",
    ):
        optional_value = worker_args.get(optional_key)
        if optional_value is not None:
            engine_kwargs[optional_key] = optional_value

    expert_parallel_size = int(worker_args.get("expert_parallel_size", 0))
    if expert_parallel_size > 0:
        engine_kwargs["enable_expert_parallel"] = True

    profile = str(worker_args.get("runtime_profile") or "")
    if role == DECODE:
        enable_decode_saturation = bool(
            worker_args.get("enable_decode_saturation", True))
        engine_kwargs.update({
            "enable_decode_saturation":
            enable_decode_saturation,
        })
        if enable_decode_saturation:
            from saturo.predictors.qrf import load_required_qrf, resolve_qrf_path, validate_qrf_workload
            qrf_path = resolve_qrf_path(worker_args.get("model_len_predictor"))
            qrf = load_required_qrf(qrf_path)  # Fail before constructing the engine.
            validate_qrf_workload(qrf, worker_args)
            engine_kwargs.update({
                "decode_saturation_tbt_ms":
                float(worker_args.get("rt_tbt_slo_ms", 100.0)),
                "decode_saturation_lookahead_s":
                float(worker_args.get(
                    "decode_saturation_lookahead_s", 60.0)),
                "decode_saturation_require_model_len_predictor":
                True,
                "time_predictor_profile":
                profile or None,
            })
            engine_kwargs["decode_saturation_model_len_predictor"] = str(qrf_path)
        # Ordinary BE requests may still run local chunked prefill in the
        # D-pool. Decode-only requests bypass prefill through direct scheduler
        # admission and do not require a KV connector hand-off iteration.
        engine_kwargs["enable_chunked_prefill"] = enable_decode_saturation

    p_pool_be_chunk_ms = worker_args.get("p_pool_be_chunk_ms")
    if role == PREFILL:
        engine_kwargs["enable_p_pool_saturation"] = bool(worker_args.get("enable_prefill_saturation", True))
    if (role == PREFILL and worker_args.get("enable_prefill_saturation", True)
            and p_pool_be_chunk_ms is not None):
        engine_kwargs.update({
            "enable_p_pool_saturation":
            True,
            "p_pool_be_chunk_ms":
            float(p_pool_be_chunk_ms),
            "p_pool_preemption_margin":
            float(worker_args.get("p_pool_preemption_margin", 0.0)),
        })
        # Fixed token-size BE prefill chunk. When > 0 the scheduler admits
        # exactly this many BE prefill tokens per step and skips the
        # latency (ms) binary search (see Scheduler._find_be_prefill_chunk).
        p_pool_be_chunk_size = worker_args.get("p_pool_be_chunk_size")
        if p_pool_be_chunk_size is not None:
            engine_kwargs["p_pool_be_chunk_size"] = int(p_pool_be_chunk_size)

    if profile:
        engine_kwargs["time_predictor_profile"] = profile
        engine_kwargs["p_pool_time_predictor_profile"] = profile

    if worker_args.get("adaptive_role_switching", False):
        # Keep partial BE prefills valid across subsequent role changes.
        engine_kwargs["enable_chunked_prefill"] = True

    if bool(worker_args.get("disable_inductor", False)):
        engine_kwargs["compilation_config"] = CompilationConfig(
            level=0,
            custom_ops=["none"],
            # Leave capture sizes unset so vLLM covers max_num_seqs using
            # its native padding rules rather than a fixed batch-size cap.
            use_cudagraph=decode_graphs,
            use_inductor=False,
        )

    # P/T-pool validation can preload a request-scoped decode-ready population
    # through DummyConnector. D-pool decode-only requests use direct scheduler
    # admission instead and never execute a synthetic prefill hand-off.
    p_pool_dummy_decode = bool(
        worker_args.get("enable_p_pool_dummy_decode", False))
    t_pool_dummy_decode = bool(
        worker_args.get("enable_t_pool_dummy_decode", False))
    use_dummy_kv = (
        (role == PREFILL and p_pool_dummy_decode)
        or (role == TPUT and t_pool_dummy_decode)
    )
    if real_handoff(worker_args):
        if use_dummy_kv:
            raise ValueError("Real KV handoff cannot be combined with dummy-KV pool populations")
        if worker_args.get("load_format") == "dummy":
            raise ValueError("Real KV handoff requires real model weights")
        if not worker_args.get("kv_transfer_dir") or not worker_args.get("kv_transfer_namespace"):
            raise ValueError("Real KV handoff requires a shared directory and run namespace")
        engine_kwargs["kv_transfer_config"] = KVTransferConfig(
            kv_connector=CONNECTOR, kv_role="kv_both",
            kv_connector_extra_config={
                "directory": worker_args["kv_transfer_dir"],
                "namespace": worker_args["kv_transfer_namespace"],
                "timeout_s": worker_args.get("kv_transfer_timeout_s", 30),
            })
    elif use_dummy_kv:
        engine_kwargs["kv_transfer_config"] = KVTransferConfig(
            kv_connector="DummyConnector",
            kv_role="kv_consumer",
        )

    return engine_kwargs, SamplingParams, use_dummy_kv


def _build_async_engine(
    worker_args: dict[str, Any], ) -> tuple[Any, type[Any], bool]:
    """Create the role-specific V0 AsyncLLMEngine."""
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    engine_kwargs, sampling_params, use_dummy_kv = _build_engine_config(
        worker_args)
    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(**engine_kwargs),
        start_engine_loop=True,
    )
    return engine, sampling_params, use_dummy_kv


def _validate_p_pool_budget(core, role: str) -> None:
    """Reject a budget that can never schedule a single-token BE tail."""
    if role != PREFILL:
        return
    config = getattr(core, "scheduler_config", None)
    if not getattr(config, "enable_p_pool_saturation", True):
        return
    target_ms = float(getattr(config, "p_pool_be_chunk_ms", 0) or 0)
    predictor = getattr(core, "time_predictor", None)
    if (target_ms > 0 and predictor is not None
            and getattr(predictor, "available", False)
            and not getattr(config, "p_pool_be_chunk_size", 0)):
        one_token_ms = predictor.predict_forward_ms([(1, 1)], [])
        if one_token_ms > target_ms:
            raise ValueError(
                f"P-pool BE budget {target_ms:.3f}ms cannot schedule a "
                f"single-token prefill ({one_token_ms:.3f}ms predicted). "
                "Recalibrate the timing profile and regenerate the allocation "
                "profile; this configuration would stall BE requests.")


class RayEngineWorker:
    """A Ray async actor hosting one V0 ``AsyncLLMEngine``.

    Every accepted request owns an asyncio task that consumes
    ``AsyncLLMEngine.generate``. The vLLM background loop performs scheduling
    and model steps independently; callers only submit work and drain events.
    """

    def __init__(self, worker_args: dict[str, Any]):
        _configure_worker_environment(
            bool(worker_args.get("disable_inductor", False)))

        self.worker_id = int(worker_args["worker_id"])
        self.worker_role = str(worker_args["worker_role"])
        self.model = str(worker_args["model"])
        self.rt_prefill_deadline_s = float(
            worker_args["rt_prefill_deadline_s"])

        self.engine, self.SamplingParams, self._use_dummy_kv = (
            _build_async_engine(worker_args))
        _validate_p_pool_budget(self.engine.engine, self.worker_role)
        self.active: dict[str, dict[str, Any]] = {}
        self._request_tasks: dict[str, asyncio.Task[None]] = {}
        self._abort_requested: set[str] = set()
        self._events: list[dict[str, Any]] = []
        self._last_iteration_log_idx = 0
        self._shutting_down = False
        self._admission_diagnostics = (
            os.environ.get("SATURO_ADMISSION_DIAGNOSTICS", "").strip().lower()
            in ("1", "true", "yes", "on"))
        self._admission_snapshot_interval_s = max(float(os.environ.get(
            "SATURO_ADMISSION_SNAPSHOT_INTERVAL_S", "1.0")), 0.05)
        self._last_admission_snapshot_s = 0.0
        self._role_args = dict(worker_args)
        self._pool_layout_generation = int(worker_args.get("pool_layout_generation", 0))
        self._history_version = 0
        self._prepared_history = None
        self._role_switching_enabled = bool(
            worker_args.get("adaptive_role_switching", False))
        self._decode_role_initialized = (
            self.worker_role == DECODE
            and bool(worker_args.get("enable_decode_saturation", True)))
        if self._role_switching_enabled:
            self._warm_decode_role()
            self._install_role_barrier()
        if self._admission_diagnostics:
            self.engine._saturo_request_stage_callback = (
                self._on_engine_request_stage)
        self._configure_native_updates(warm=True)
        self._initialize_static_decode_gc()

    def _configure_native_updates(self, *, warm=False, history_ready=True):
        """Apply the shared protocol before traffic, or at a role/epoch boundary."""
        args = getattr(self, "_role_args", {})
        for scheduler in getattr(self.engine.engine, "scheduler", ()):
            policy = getattr(scheduler, "decode_saturation", None)
            configure = getattr(policy, "configure_update_protocol", None)
            if configure is None:
                continue
            if getattr(scheduler, "time_predictor", None) is not None:
                policy.time_predictor = scheduler.time_predictor
            configure(
                async_enabled=(self.worker_role == DECODE
                               and args.get("enable_decode_saturation", True)
                               and args.get("decode_target_async", False)),
                update_interval_s=float(args.get("saturation_update_interval_s", 40)),
                update_timeout_s=float(args.get("saturation_update_timeout_s", 30)),
                lookahead_window_s=float(args.get("decode_saturation_lookahead_s", 40)),
                epoch=getattr(self, "_pool_layout_generation", 0),
                history_ready=history_ready)
            if warm and self.worker_role == DECODE:
                initialize = getattr(policy, "warm_update_protocol", None)
                if initialize is not None:
                    initialize(timeout_s=float(args.get("saturation_update_timeout_s", 30)))

    def _reset_native_history(self):
        for scheduler in getattr(self.engine.engine, "scheduler", ()):
            policy = getattr(scheduler, "decode_saturation", None)
            provider = getattr(policy, "provider", None)
            reset = getattr(provider, "reset_history_observations", None)
            if reset is not None:
                reset()

    def _p_history_budget_required(self):
        """Disabled P stages participate in versions without budget changes."""
        return self.worker_role == PREFILL and any(
            getattr(scheduler.scheduler_config, "enable_p_pool_saturation", True)
            for scheduler in getattr(self.engine.engine, "scheduler", ()))

    def _history_control(self, cmd):
        operation = cmd["cmd"]
        epoch = int(cmd.get("pool_layout_generation", 0))
        current = getattr(self, "_pool_layout_generation", 0)
        version = int(cmd.get("version", 0))
        reply = {"operation": operation, "version": version,
                 "pool_layout_generation": epoch}
        if operation == "update_epoch":
            if epoch < current:
                return {**reply, "accepted": False, "reason": "stale_layout"}
            if epoch > current:
                self._pool_layout_generation = epoch
                self._history_version = 0
                self._prepared_history = None
                self._reset_native_history()
                self._configure_native_updates(history_ready=False)
            return {**reply, "accepted": True}
        if epoch != current:
            return {**reply, "accepted": False, "reason": "layout_mismatch"}
        applied = getattr(self, "_history_version", 0)
        if operation == "history_prepare":
            if version <= applied:
                return {**reply, "accepted": version == applied, "already_applied": version == applied,
                        "reason": "stale_version" if version < applied else None}
            prepared = getattr(self, "_prepared_history", None)
            if prepared is not None and version < prepared["version"]:
                return {**reply, "accepted": False, "reason": "stale_version"}
            if self._p_history_budget_required():
                try:
                    value = float(cmd.get("p_chunk_ms", 0))
                except (TypeError, ValueError):
                    return {**reply, "accepted": False, "reason": "invalid_p_budget"}
                if not math.isfinite(value) or value <= 0:
                    return {**reply, "accepted": False, "reason": "invalid_p_budget"}
            if self.worker_role == DECODE:
                stats = cmd.get("statistics", {})
                for name in ("arrival_rate", "mean_prompt_len", "mean_output_len", "valid_until_s", "completed_requests"):
                    if name not in stats or not math.isfinite(float(stats[name])):
                        return {**reply, "accepted": False, "reason": "invalid_statistics"}
                if float(stats["arrival_rate"]) < 0 or min(float(stats["mean_prompt_len"]), float(stats["mean_output_len"])) <= 0:
                    return {**reply, "accepted": False, "reason": "invalid_statistics"}
                if float(stats["completed_requests"]) <= 0:
                    return {**reply, "accepted": False, "reason": "incomplete_history"}
                if float(stats["valid_until_s"]) <= time.time():
                    return {**reply, "accepted": False, "reason": "expired_history"}
            self._prepared_history = copy.deepcopy(cmd)
            return {**reply, "accepted": True}
        if operation != "history_commit":
            raise ValueError("unknown history operation")
        if version == applied:
            return {**reply, "accepted": True, "already_applied": True}
        prepared = getattr(self, "_prepared_history", None)
        if prepared is None or prepared["version"] != version:
            return {**reply, "accepted": False, "reason": "not_prepared"}
        p_chunk_ms = None
        if self._p_history_budget_required():
            try:
                p_chunk_ms = float(prepared.get("p_chunk_ms", 0))
            except (TypeError, ValueError):
                return {**reply, "accepted": False, "reason": "invalid_p_budget"}
            if not math.isfinite(p_chunk_ms) or p_chunk_ms <= 0:
                return {**reply, "accepted": False, "reason": "invalid_p_budget"}
        if (self.worker_role == DECODE
                and float(prepared["statistics"]["valid_until_s"]) <= time.time()):
            self._prepared_history = None
            return {**reply, "accepted": False, "reason": "expired_history"}
        # Preparation validated all inputs before mutating any PP stage.
        for scheduler in getattr(self.engine.engine, "scheduler", ()):
            if (self.worker_role == PREFILL
                    and getattr(scheduler.scheduler_config,
                                "enable_p_pool_saturation", True)):
                scheduler.scheduler_config.p_pool_be_chunk_ms = p_chunk_ms
            elif self.worker_role == DECODE:
                scheduler.decode_saturation.provider.set_history_statistics(prepared["statistics"])
        if p_chunk_ms is not None and hasattr(self, "_role_args"):
            self._role_args["p_pool_be_chunk_ms"] = p_chunk_ms
        self._history_version = version
        self._prepared_history = None
        self._configure_native_updates(history_ready=True)
        self._emit({"event": "history_applied", **reply, "time_s": time.perf_counter()})
        return {**reply, "accepted": True}

    def _initialize_static_decode_gc(self) -> None:
        """Keep the initialized actor-lifetime heap out of serving GC scans.

        This runs once, before ready and before any requests. New request
        cycles retain ordinary collection. A static D actor never replaces
        its engine, and retirement terminates its dedicated Ray process.
        """
        if getattr(self, "_static_decode_gc_initialized", False):
            return
        self._static_decode_gc_initialized = True
        self._static_decode_gc_frozen = False
        self._static_decode_gc_frozen_count = 0
        self._static_decode_gc_startup_ms = 0.0
        if (self.worker_role != DECODE or self._role_switching_enabled
                or not self._decode_role_initialized
                or os.environ.get("SATURO_STATIC_DECODE_GC_FREEZE", "1")
                .strip().lower() in ("0", "false", "no", "off")
                or not gc.isenabled()):
            return
        started = time.perf_counter()
        gc.collect(2)
        gc.freeze()
        self._static_decode_gc_frozen = True
        self._static_decode_gc_frozen_count = gc.get_freeze_count()
        self._static_decode_gc_startup_ms = (
            time.perf_counter() - started) * 1000

    async def info(self) -> dict[str, Any]:
        core = self.engine.engine
        cache = getattr(core, "cache_config", None)
        parallel = getattr(core, "parallel_config", None)
        predictors = []
        prediction_calls = 0
        for scheduler in getattr(core, "scheduler", ()):
            provider = getattr(getattr(scheduler, "decode_saturation", None),
                               "provider", None)
            predictor = getattr(provider, "model_len_predictor", None)
            if predictor is not None:
                prediction_calls += getattr(provider, "prediction_calls", 0)
                predictors.append({
                    "type": predictor.model_type,
                    "path": provider.model_len_predictor_path,
                    "quantile": provider.model_len_predictor_eta,
                    "feature_mode": getattr(getattr(predictor, "config", None), "feature_mode", None),
                })
        original_prediction_calls = prediction_calls
        update_diagnostics = [p.update_protocol_diagnostics()
            for s in getattr(core, "scheduler", ())
            for p in [getattr(s, "decode_saturation", None)]
            if callable(getattr(p, "update_protocol_diagnostics", None))]
        background_prediction_calls = sum(int(d.get("background_qrf_prediction_calls", 0))
                                          for d in update_diagnostics)
        prediction_calls += background_prediction_calls
        return {
            "event": "ready",
            "worker_id": self.worker_id,
            "role": self.worker_role,
            "engine": type(self.engine).__name__,
            "vllm_v1": os.environ.get("VLLM_USE_V1"),
            "gpu_id": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "pid": os.getpid(),
            "static_decode_gc_frozen": getattr(
                self, "_static_decode_gc_frozen", False),
            "static_decode_gc_frozen_count": getattr(
                self, "_static_decode_gc_frozen_count", 0),
            "static_decode_gc_startup_ms": getattr(
                self, "_static_decode_gc_startup_ms", 0.0),
            "remaining_length_predictors": predictors,
            "remaining_length_prediction_calls": prediction_calls,
            "serving_provider_prediction_calls": original_prediction_calls,
            "background_qrf_prediction_calls": background_prediction_calls,
            "runtime_profile_path": str(getattr(
                getattr(core, "time_predictor", None), "profile_path", "")
                or ""),
            "num_gpu_blocks": int(getattr(cache, "num_gpu_blocks", 0) or 0)
            // max(int(getattr(parallel, "pipeline_parallel_size", 1)), 1),
            "max_model_len": getattr(
                getattr(core, "model_config", None), "max_model_len", None),
            "block_size": int(getattr(cache, "block_size", 16)),
            "max_num_seqs": int(getattr(
                getattr(core, "scheduler_config", None), "max_num_seqs", 0)),
            "role_switching_enabled": getattr(
                self, "_role_switching_enabled", False),
            "pool_layout_generation": getattr(self, "_pool_layout_generation", 0),
            "history_version": getattr(self, "_history_version", 0),
            "saturation_update_interval_s": getattr(self, "_role_args", {}).get("saturation_update_interval_s", 40),
            "decode_update_protocol": update_diagnostics,
        }

    def _build_decode_role_policy(self, scheduler):
        config = copy.copy(scheduler.scheduler_config)
        config.enable_decode_saturation = True
        config.decode_saturation_tbt_ms = float(
            self._role_args.get("rt_tbt_slo_ms", 100))
        config.decode_saturation_lookahead_s = float(
            self._role_args.get("decode_saturation_lookahead_s", 60))
        config.decode_saturation_model_len_predictor = (
            self._role_args.get("model_len_predictor"))
        policy = scheduler.decode_saturation
        return type(policy)(block_size=policy.block_size,
                            total_gpu_blocks=policy.total_gpu_blocks,
                            scheduler_config=config)

    def _warm_decode_role(self):
        """Load QRF before traffic, keeping the current P/T mode disabled."""
        if (self._decode_role_initialized
                or not self._role_args.get("enable_decode_saturation", True)):
            return
        schedulers = list(self.engine.engine.scheduler)
        policies = [self._build_decode_role_policy(s) for s in schedulers]
        for scheduler, policy in zip(schedulers, policies):
            policy.enabled = False
            scheduler.decode_saturation = policy
        self._decode_role_initialized = True

    def _install_role_barrier(self) -> None:
        """Pause new iterations only while changing all PP stages' roles."""
        self._role_condition = asyncio.Condition()
        self._role_changing = False
        self._role_iterations = 0
        original = self.engine.engine_step

        async def guarded_step(*args, **kwargs):
            async with self._role_condition:
                await self._role_condition.wait_for(
                    lambda: not self._role_changing)
                self._role_iterations += 1
            try:
                return await original(*args, **kwargs)
            finally:
                async with self._role_condition:
                    self._role_iterations -= 1
                    self._role_condition.notify_all()

        self.engine.engine_step = guarded_step

    async def switch_role(self, role: str) -> dict[str, Any]:
        """Reconfigure scheduling at an iteration boundary; retain BE/KV."""
        if role not in (PREFILL, DECODE, TPUT):
            raise ValueError(f"invalid worker role: {role!r}")
        if (not self._role_switching_enabled or self._shutting_down
                or self._use_dummy_kv):
            return {"accepted": False, "role": self.worker_role}
        async with self._role_condition:
            if self._role_changing:
                return {"accepted": False, "role": self.worker_role}
            self._role_changing = True
            try:
                await self._role_condition.wait_for(
                    lambda: self._role_iterations == 0)
                if any(meta["kind"] == RT for meta in self.active.values()):
                    return {"accepted": False, "role": self.worker_role}
                if (not pool_accepts_be(role, self._role_args)
                        and any(meta["kind"] == BE for meta in self.active.values())):
                    return {"accepted": False, "role": self.worker_role,
                            "reason": "target pool excludes BE in this ablation"}
                core = self.engine.engine
                schedulers = list(core.scheduler)
                if any(getattr(getattr(stage, "_decode_reservations", None),
                               "tokens", {}) for stage in schedulers):
                    return {"accepted": False, "role": self.worker_role}
                decode = (role == DECODE and bool(
                    self._role_args.get("enable_decode_saturation", True)))
                # Prepare every policy before mutating any live PP stage.
                policies = []
                for scheduler in schedulers:
                    policy = scheduler.decode_saturation
                    if decode and not self._decode_role_initialized:
                        policy = self._build_decode_role_policy(scheduler)
                    policies.append(policy)
                configs = [core.scheduler_config] + [
                    scheduler.scheduler_config for scheduler in schedulers]
                for config in configs:
                    config.enable_p_pool_saturation = (role == PREFILL and bool(
                        self._role_args.get("enable_prefill_saturation", True)))
                    config.enable_decode_saturation = decode
                    config.decode_saturation_tbt_ms = float(
                        self._role_args.get("rt_tbt_slo_ms", 100))
                    config.decode_saturation_lookahead_s = float(
                        self._role_args.get(
                            "decode_saturation_lookahead_s", 60))
                    config.p_pool_be_chunk_ms = float(
                        self._role_args.get("p_pool_be_chunk_ms", 100))
                    config.p_pool_preemption_margin = float(
                        self._role_args.get("p_pool_preemption_margin", 0))
                for scheduler, policy in zip(schedulers, policies):
                    policy.enabled = decode
                    scheduler.decode_saturation = policy
                self._decode_role_initialized |= decode
                self.worker_role = role
                self._prepared_history = None
                self._reset_native_history()
                self._configure_native_updates(history_ready=False)
                return {"accepted": True, "role": role}
            finally:
                self._role_changing = False
                self._role_condition.notify_all()

    def _emit(self, payload: dict[str, Any]) -> None:
        payload.setdefault("worker_id", self.worker_id)
        self._events.append(payload)

    def _emit_request_stage(self, attempt_id: str, stage: str,
                            event_time_s: float | None = None) -> None:
        if not self._admission_diagnostics:
            return
        meta = self.active.get(attempt_id)
        self._emit({
            "event": "request_stage",
            "stage": stage,
            "attempt_id": attempt_id,
            "request_id": (
                str(meta["request_id"]) if meta is not None else attempt_id),
            "kind": (str(meta["kind"]) if meta is not None else ""),
            "phase": (str(meta["phase"]) if meta is not None else ""),
            "time_s": (
                time.perf_counter()
                if event_time_s is None else float(event_time_s)),
        })

    def _on_engine_request_stage(self, attempt_id: str, stage: str,
                                 event_time_s: float) -> None:
        self._emit_request_stage(attempt_id, stage, event_time_s)

    def emit_admission_snapshot(self) -> None:
        if not self._admission_diagnostics:
            return

        tracker = getattr(self.engine, "_request_tracker", None)
        tracker_new_queue = -1
        tracker_streams = -1
        if tracker is not None:
            new_requests = getattr(tracker, "_new_requests", None)
            if new_requests is not None:
                tracker_new_queue = int(new_requests.qsize())
            tracker_streams = len(
                getattr(tracker, "_request_streams", {}))

        schedulers = getattr(self.engine.engine, "scheduler", ())
        scheduler_running = 0
        scheduler_waiting = 0
        scheduler_waiting_rt_admission = 0
        scheduler_swapped = 0
        for scheduler in schedulers:
            scheduler_running += len(getattr(scheduler, "running", ()))
            scheduler_waiting += len(getattr(scheduler, "waiting", ()))
            scheduler_waiting_rt_admission += len(getattr(
                scheduler, "waiting_rt_decode_admission", ()))
            scheduler_swapped += len(getattr(scheduler, "swapped", ()))

        active_meta = tuple(self.active.values())
        self._emit({
            "event": "admission_snapshot",
            "time_s": time.perf_counter(),
            "worker_active_requests": len(active_meta),
            "worker_request_tasks": len(self._request_tasks),
            "worker_tasks_done": sum(
                task.done() for task in self._request_tasks.values()),
            "worker_tasks_not_started": sum(
                not bool(meta.get("consume_started"))
                for meta in active_meta),
            "worker_active_zero_token": sum(
                int(meta.get("last_seen", 0)) == 0
                for meta in active_meta),
            "request_tracker_new_queue": tracker_new_queue,
            "request_tracker_streams": tracker_streams,
            "scheduler_running": scheduler_running,
            "scheduler_waiting": scheduler_waiting,
            "scheduler_waiting_rt_decode_admission":
            scheduler_waiting_rt_admission,
            "scheduler_swapped": scheduler_swapped,
        })

    def _request_inputs(
        self,
        cmd: dict[str, Any],
    ) -> tuple[dict[str, list[int]], Any, int]:
        real = real_handoff(getattr(self, "_role_args", {}))
        handoff = cmd.get("kv_handoff")
        direct_decode = (_is_direct_decode_command(self.worker_role, cmd)
                         and (not real or handoff is not None))
        if real and cmd["kind"] == RT and handoff and "prompt_token_ids" not in cmd:
            raise ValueError("Real KV handoff requires the actual prompt/output token IDs")
        params_kwargs: dict[str, Any] = {
            "temperature": 0.0,
            # Direct D-pool admission seeds one synthetic output token as the
            # first RT/BE decode input. Add one to the vLLM cap, then hide that
            # seed from benchmark token accounting in _consume_request().
            "max_tokens": int(cmd["max_tokens"]) + int(direct_decode),
            "ignore_eos": True,
            "detokenize": False,
        }
        if cmd["kind"] == RT:
            params_kwargs["is_realtime"] = True
            params_kwargs["prefill_ddl"] = (time.time() +
                                            self.rt_prefill_deadline_s)
            if "absolute_prefill_deadline_s" in cmd:
                params_kwargs["prefill_ddl"] = min(
                    params_kwargs["prefill_ddl"], float(cmd["absolute_prefill_deadline_s"]))
        if direct_decode:
            params_kwargs["extra_args"] = {
                "saturo_direct_decode": True,
            }
        elif cmd["phase"] == DECODE and self._use_dummy_kv:
            params_kwargs["extra_args"] = {
                "saturo_dummy_kv_decode": True,
            }

        if cmd["kind"] == RT:
            extra = dict(params_kwargs.get("extra_args") or {})
            if str(cmd.get("request_id", "")).startswith("__warmup__"):
                extra["saturo_skip_traffic_observation"] = True
            if "source_arrival_time_s" in cmd:
                extra["saturo_source_arrival_time_s"] = float(
                    cmd["source_arrival_time_s"])
                extra["saturo_source_request_id"] = str(
                    cmd.get("source_request_id", cmd["request_id"]))
            if extra:
                params_kwargs["extra_args"] = extra

        if real and handoff:
            extra = dict(params_kwargs.get("extra_args") or {})
            extra["saturo_kv_handoff"] = dict(handoff,
                operation="recv" if cmd["phase"] == DECODE else "send")
            params_kwargs["extra_args"] = extra
        if (direct_decode and not real) or (cmd["phase"] == DECODE and self._use_dummy_kv):
            prompt_token_ids = [1] * int(cmd.get("prompt_len", 1))
        elif "prompt_token_ids" in cmd:
            prompt_token_ids = list(cmd["prompt_token_ids"])
        else:
            prompt_token_ids = [0] * int(cmd["prompt_len"])

        if cmd.get("reservation_id"):
            extra = dict(params_kwargs.get("extra_args") or {})
            extra["saturo_decode_reservation_id"] = cmd["reservation_id"]
            params_kwargs["extra_args"] = extra
        priority = 0 if cmd["kind"] == RT else 10
        return ({
            "prompt_token_ids": prompt_token_ids
        }, self.SamplingParams(**params_kwargs), priority)

    async def submit(
        self,
        cmd: dict[str, Any],
        *,
        _yield_after: bool = True,
    ) -> dict[str, Any]:
        """Accept one request and start consuming its asynchronous stream."""
        if self._shutting_down:
            raise RuntimeError("worker is shutting down")
        if (self._engine_failed()
                and cmd.get("cmd") not in ("abort", "release_decode")):
            from vllm.engine.async_llm_engine import AsyncEngineDeadError
            raise AsyncEngineDeadError("Saturo worker's engine has failed")
        if cmd.get("cmd") in ("update_epoch", "history_prepare", "history_commit"):
            return self._history_control(cmd)
        if cmd.get("cmd") in ("reserve_decode", "release_decode"):
            if self.worker_role != DECODE:
                raise ValueError("decode reservation sent to a non-D worker")
            schedulers = getattr(self.engine.engine, "scheduler", ())
            if not schedulers:
                raise RuntimeError("D worker has no native scheduler")
            reservation_id = str(cmd["reservation_id"])
            try:
                for scheduler in schedulers:
                    if cmd["cmd"] == "reserve_decode":
                        scheduler.reserve_decode_request(
                            reservation_id, int(cmd["prompt_len"]))
                    else:
                        scheduler.release_decode_reservation(reservation_id)
            except ValueError as exc:
                for scheduler in schedulers:
                    scheduler.release_decode_reservation(reservation_id)
                return {"accepted": False, "reason": str(exc)}
            return {"accepted": True, "operation": cmd["cmd"]}
        if cmd.get("cmd") == "history":
            for scheduler in getattr(self.engine.engine, "scheduler", ()):
                if self.worker_role == PREFILL and "p_chunk_ms" in cmd:
                    scheduler.scheduler_config.p_pool_be_chunk_ms = float(
                        cmd["p_chunk_ms"])
                    if hasattr(self, "_role_args"):
                        self._role_args["p_pool_be_chunk_ms"] = float(
                            cmd["p_chunk_ms"])
                if self.worker_role == DECODE and "statistics" in cmd:
                    policy = getattr(scheduler, "decode_saturation", None)
                    provider = getattr(policy, "provider", None)
                    setter = getattr(provider, "set_history_statistics", None)
                    if setter is not None:
                        setter(cmd["statistics"])
            return {"accepted": True, "operation": "history"}
        if cmd.get("cmd", "add") == "abort":
            aborted = await self.abort(str(cmd["attempt_id"]))
            return {"accepted": aborted, "operation": "abort"}
        if cmd.get("cmd", "add") != "add":
            raise ValueError(f"unsupported worker command: {cmd.get('cmd')!r}")

        if getattr(self, "_role_switching_enabled", False):
            async with self._role_condition:
                await self._role_condition.wait_for(
                    lambda: not self._role_changing)

        attempt_id = str(cmd["attempt_id"])
        if attempt_id in self.active or attempt_id in self._request_tasks:
            raise ValueError(f"duplicate attempt_id: {attempt_id}")

        max_tokens = int(cmd["max_tokens"])
        if max_tokens <= 0:
            if cmd.get("reservation_id"):
                for scheduler in getattr(self.engine.engine, "scheduler", ()):
                    scheduler.release_decode_reservation(cmd["reservation_id"])
            self._emit({
                "event": "finished",
                "attempt_id": attempt_id,
                "request_id": cmd["request_id"],
                "kind": cmd["kind"],
                "phase": cmd["phase"],
                "token_count": 0,
                "global_token_count": int(cmd["base_tokens"]),
                "time_s": time.perf_counter(),
            })
            return {"accepted": True, "attempt_id": attempt_id}

        prompt, sampling_params, priority = self._request_inputs(cmd)
        self.active[attempt_id] = {
            "request_id": str(cmd["request_id"]),
            "kind": str(cmd["kind"]),
            "phase": str(cmd["phase"]),
            "last_seen": 0,
            "base_tokens": int(cmd["base_tokens"]),
            "consume_started": False,
            "reservation_id": cmd.get("reservation_id"),
            "hidden_prefix_tokens": int(bool(
                (getattr(sampling_params, "extra_args", None) or {}).get(
                    "saturo_direct_decode", False))),
        }
        self._emit_request_stage(attempt_id, "worker_submitted")
        task = asyncio.create_task(
            self._consume_request(
                attempt_id,
                prompt,
                sampling_params,
                priority,
            ),
            name=f"saturo-request-{attempt_id}",
        )
        self._request_tasks[attempt_id] = task

        # Match benchmark_serving's create_task pattern: yield once so the task
        # can register with AsyncLLMEngine before the submission RPC returns.
        if _yield_after:
            await asyncio.sleep(0)
        return {"accepted": True, "attempt_id": attempt_id}

    async def submit_batch(
        self,
        commands: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        acknowledgements = []
        for cmd in commands:
            acknowledgements.append(
                await self.submit(cmd, _yield_after=False))
        # Register the whole arrival batch before allowing the synchronous
        # engine loop to run. Yielding after every command can otherwise turn
        # a 128-request initial BE population into 128 model iterations and
        # starve later RT arrivals in the stdio pipe.
        await asyncio.sleep(0)
        return acknowledgements

    async def _consume_request(
        self,
        attempt_id: str,
        prompt: dict[str, list[int]],
        sampling_params: Any,
        priority: int,
    ) -> None:
        finished = False
        try:
            meta = self.active.get(attempt_id)
            if meta is not None:
                meta["consume_started"] = True
            self._emit_request_stage(attempt_id, "worker_task_started")
            async for output in self.engine.generate(prompt,
                                                     sampling_params,
                                                     attempt_id,
                                                     priority=priority):
                self._emit_new_iteration_logs()
                meta = self.active.get(attempt_id)
                if meta is None:
                    continue

                if meta.get("reservation_id"):
                    # A yielded decode output confirms actual admission.
                    # Release conservative claims on other virtual engines.
                    for scheduler in getattr(self.engine.engine, "scheduler", ()):
                        scheduler.release_decode_reservation(meta["reservation_id"])
                    meta["reservation_id"] = None
                token_count = int(meta["last_seen"])
                if output.outputs:
                    raw_token_count = max(
                        len(candidate.token_ids)
                        for candidate in output.outputs)
                    token_count = max(
                        raw_token_count
                        - int(meta.get("hidden_prefix_tokens", 0)), 0)
                sampled_ids = None
                if real_handoff(getattr(self, "_role_args", {})) and meta["kind"] == RT and output.outputs:
                    candidate = max(output.outputs, key=lambda x: len(x.token_ids))
                    sampled_ids = list(candidate.token_ids)[int(meta.get("hidden_prefix_tokens", 0)):]
                delta = token_count - int(meta["last_seen"])
                now_s = time.perf_counter()
                if delta > 0:
                    meta["last_seen"] = token_count
                    stream_put_s = getattr(
                        output, "_saturo_stream_put_s", None)
                    stream_get_s = getattr(
                        output, "_saturo_stream_get_s", None)
                    self._emit({
                        "event":
                        "token",
                        "attempt_id":
                        attempt_id,
                        "request_id":
                        meta["request_id"],
                        "kind":
                        meta["kind"],
                        "phase":
                        meta["phase"],
                        "delta":
                        delta,
                        "attempt_token_count":
                        token_count,
                        "global_token_count":
                        (int(meta["base_tokens"]) + token_count),
                        "sampled_token_ids": sampled_ids,
                        "base_tokens": int(meta["base_tokens"]),
                        "stream_queue_wait_ms": (
                            (float(stream_get_s) - float(stream_put_s)) * 1000.0
                            if stream_put_s is not None
                            and stream_get_s is not None else None),
                        "stream_put_to_worker_ms": (
                            (now_s - float(stream_put_s)) * 1000.0
                            if stream_put_s is not None else None),
                        "stream_get_to_worker_ms": (
                            (now_s - float(stream_get_s)) * 1000.0
                            if stream_get_s is not None else None),
                        "time_s":
                        now_s,
                    })

                if output.finished:
                    finished = True
                    self._emit({
                        "event":
                        "finished",
                        "attempt_id":
                        attempt_id,
                        "request_id":
                        meta["request_id"],
                        "kind":
                        meta["kind"],
                        "phase":
                        meta["phase"],
                        "token_count":
                        token_count,
                        "global_token_count":
                        (int(meta["base_tokens"]) + token_count),
                        "sampled_token_ids": sampled_ids,
                        "base_tokens": int(meta["base_tokens"]),
                        "time_s":
                        now_s,
                    })
                    # Let AsyncLLMEngine.generate() exhaust its stream
                    # naturally. Breaking here closes the async generator
                    # with GeneratorExit after a successful final output,
                    # which makes vLLM log the request as aborted.
            if (not finished and attempt_id in self.active
                    and attempt_id not in self._abort_requested):
                meta = self.active[attempt_id]
                self._emit({
                    "event": "error",
                    "attempt_id": attempt_id,
                    "request_id": meta["request_id"],
                    "kind": meta["kind"],
                    "phase": meta["phase"],
                    "error_type": "IncompleteStreamError",
                    "engine_fatal": self._engine_failed(),
                    "message": "AsyncLLMEngine stream ended before completion",
                    "time_s": time.perf_counter(),
                })
        except asyncio.CancelledError:
            self._emit_aborted(attempt_id)
        except Exception as exc:
            meta = self.active.get(attempt_id)
            if meta is not None:
                self._emit({
                    "event": "error",
                    "attempt_id": attempt_id,
                    "request_id": meta["request_id"],
                    "kind": meta["kind"],
                    "phase": meta["phase"],
                    "error_type": type(exc).__name__,
                    "engine_fatal": (self._engine_failed()
                                     or type(exc).__name__ == "AsyncEngineDeadError"),
                    "message": str(exc),
                    "time_s": time.perf_counter(),
                })
        finally:
            meta = self.active.get(attempt_id)
            if meta is not None and meta.get("reservation_id"):
                for scheduler in getattr(self.engine.engine, "scheduler", ()):
                    scheduler.release_decode_reservation(meta["reservation_id"])
            self._emit_new_iteration_logs()
            if not finished and attempt_id in self._abort_requested:
                self._emit_aborted(attempt_id)
            self.active.pop(attempt_id, None)
            self._abort_requested.discard(attempt_id)
            current = asyncio.current_task()
            if self._request_tasks.get(attempt_id) is current:
                self._request_tasks.pop(attempt_id, None)

    def _engine_failed(self) -> bool:
        # A stream receives the original background exception (e.g. the first
        # AssertionError), not necessarily an AsyncEngineDeadError wrapper.
        return bool(getattr(self.engine, "errored", False)
                    or getattr(self.engine, "is_stopped", False))

    def _emit_aborted(self, attempt_id: str) -> None:
        meta = self.active.get(attempt_id)
        if meta is None or meta.get("abort_emitted"):
            return
        meta["abort_emitted"] = True
        self._emit({
            "event": "aborted",
            "attempt_id": attempt_id,
            "request_id": meta["request_id"],
            "kind": meta["kind"],
            "phase": meta["phase"],
            "time_s": time.perf_counter(),
        })

    async def abort(self, attempt_id: str) -> bool:
        if attempt_id not in self.active:
            return False
        self._abort_requested.add(attempt_id)
        task = self._request_tasks.get(attempt_id)
        try:
            if self.engine.is_running:
                await self.engine.abort(attempt_id)
            elif task is not None:
                task.cancel()
        except Exception:
            if task is not None:
                task.cancel()
        await asyncio.sleep(0)
        return True

    def _emit_new_iteration_logs(self) -> None:
        core_engine = self.engine.engine
        logs = getattr(core_engine, "iteration_logs", [])
        while self._last_iteration_log_idx < len(logs):
            payload = dict(logs[self._last_iteration_log_idx])
            payload["event"] = "iteration_log"
            self._emit(payload)
            self._last_iteration_log_idx += 1

    async def drain_events(self) -> list[dict[str, Any]]:
        now_s = time.perf_counter()
        if (self._admission_diagnostics
                and now_s - self._last_admission_snapshot_s
                >= self._admission_snapshot_interval_s):
            self.emit_admission_snapshot()
            self._last_admission_snapshot_s = now_s
        self._emit_new_iteration_logs()
        events = self._events
        self._events = []
        return events

    async def shutdown(self) -> dict[str, Any] | None:
        if self._shutting_down:
            return
        self._shutting_down = True

        for attempt_id in list(self.active):
            await self.abort(attempt_id)
        tasks = list(self._request_tasks.values())
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        optimizers_closed = True
        for scheduler in getattr(self.engine.engine, "scheduler", ()):
            claims = scheduler.__dict__.get("_decode_reservations")
            if claims is not None:
                claims.tokens.clear()
            close = getattr(getattr(scheduler, "decode_saturation", None), "close", None)
            if close is not None:
                optimizers_closed = bool(close(timeout_s=1.0)) and optimizers_closed
        self.engine.shutdown_background_loop()
        shutdown_executor = getattr(getattr(self.engine.engine, "model_executor", None), "shutdown", None)
        if shutdown_executor is not None:
            shutdown_executor()
        return {"shutdown": True, "async_target_threads_joined": optimizers_closed}


class OfflineEngineWorker:
    """Explicit synchronous worker for step-driven offline microbenchmarks.

    This class is intentionally excluded from the Ray/global-scheduler path.
    It preserves manual iteration control for profiling experiments that need
    to inspect every V0 scheduler step.
    """

    def __init__(self, worker_args: dict[str, Any]):
        if real_handoff(worker_args):
            raise ValueError("Real KV handoff uses RayEngineWorker; the offline worker is dummy-only")
        _configure_worker_environment(
            bool(worker_args.get("disable_inductor", False)))
        from vllm.engine.arg_utils import EngineArgs
        from vllm.engine.llm_engine import LLMEngine

        self.worker_id = int(worker_args["worker_id"])
        self.worker_role = str(worker_args["worker_role"])
        self.model = str(worker_args["model"])
        self.rt_prefill_deadline_s = float(
            worker_args["rt_prefill_deadline_s"])
        engine_kwargs, self.SamplingParams, self._use_dummy_kv = (
            _build_engine_config(worker_args))
        engine_kwargs.pop("disable_log_requests", None)
        engine_kwargs.pop("disable_log_stats", None)
        self.engine = LLMEngine.from_engine_args(EngineArgs(**engine_kwargs))
        self.active: dict[str, dict[str, Any]] = {}
        self._events: list[dict[str, Any]] = []
        self._last_iteration_log_idx = 0

    def _emit(self, payload: dict[str, Any]) -> None:
        payload.setdefault("worker_id", self.worker_id)
        self._events.append(payload)

    def _request_inputs(
        self,
        cmd: dict[str, Any],
    ) -> tuple[dict[str, list[int]], Any, int]:
        real = real_handoff(getattr(self, "_role_args", {}))
        handoff = cmd.get("kv_handoff")
        direct_decode = (_is_direct_decode_command(self.worker_role, cmd)
                         and (not real or handoff is not None))
        if real and cmd["kind"] == RT and handoff and "prompt_token_ids" not in cmd:
            raise ValueError("Real KV handoff requires the actual prompt/output token IDs")
        params_kwargs: dict[str, Any] = {
            "temperature": 0.0,
            "max_tokens": int(cmd["max_tokens"]) + int(direct_decode),
            "ignore_eos": True,
            "detokenize": False,
        }
        if cmd["kind"] == RT:
            params_kwargs["is_realtime"] = True
            params_kwargs["prefill_ddl"] = (time.time() +
                                            self.rt_prefill_deadline_s)
        if direct_decode:
            params_kwargs["extra_args"] = {
                "saturo_direct_decode": True,
            }
        elif cmd["phase"] == DECODE and self._use_dummy_kv:
            params_kwargs["extra_args"] = {
                "saturo_dummy_kv_decode": True,
            }
        if real and handoff:
            extra = dict(params_kwargs.get("extra_args") or {})
            extra["saturo_kv_handoff"] = dict(handoff,
                operation="recv" if cmd["phase"] == DECODE else "send")
            params_kwargs["extra_args"] = extra
        if (direct_decode and not real) or (cmd["phase"] == DECODE and self._use_dummy_kv):
            prompt_token_ids = [1] * int(cmd.get("prompt_len", 1))
        elif "prompt_token_ids" in cmd:
            prompt_token_ids = list(cmd["prompt_token_ids"])
        else:
            prompt_token_ids = [0] * int(cmd["prompt_len"])
        priority = 0 if cmd["kind"] == RT else 10
        return ({
            "prompt_token_ids": prompt_token_ids
        }, self.SamplingParams(**params_kwargs), priority)

    def add(self, cmd: dict[str, Any]) -> None:
        attempt_id = str(cmd["attempt_id"])
        max_tokens = int(cmd["max_tokens"])
        if max_tokens <= 0:
            self._emit({
                "event": "finished",
                "attempt_id": attempt_id,
                "request_id": cmd["request_id"],
                "kind": cmd["kind"],
                "phase": cmd["phase"],
                "token_count": 0,
                "global_token_count": int(cmd["base_tokens"]),
                "time_s": time.perf_counter(),
            })
            return
        prompt, sampling_params, priority = self._request_inputs(cmd)
        self.engine.add_request(
            attempt_id,
            prompt,
            sampling_params,
            priority=priority,
        )
        self.active[attempt_id] = {
            "request_id": str(cmd["request_id"]),
            "kind": str(cmd["kind"]),
            "phase": str(cmd["phase"]),
            "last_seen": 0,
            "base_tokens": int(cmd["base_tokens"]),
            "hidden_prefix_tokens": int(bool(
                (getattr(sampling_params, "extra_args", None) or {}).get(
                    "saturo_direct_decode", False))),
        }

    def abort(self, attempt_id: str) -> None:
        meta = self.active.pop(attempt_id, None)
        if meta is None:
            return
        self.engine.abort_request(attempt_id)
        self._emit({
            "event": "aborted",
            "attempt_id": attempt_id,
            "request_id": meta["request_id"],
            "kind": meta["kind"],
            "phase": meta["phase"],
            "time_s": time.perf_counter(),
        })

    def _emit_new_iteration_logs(self) -> None:
        logs = getattr(self.engine, "iteration_logs", [])
        while self._last_iteration_log_idx < len(logs):
            payload = dict(logs[self._last_iteration_log_idx])
            payload["event"] = "iteration_log"
            self._emit(payload)
            self._last_iteration_log_idx += 1

    def _step_once(self) -> None:
        if not self.engine.has_unfinished_requests():
            return
        outputs = self.engine.step()
        self._emit_new_iteration_logs()
        now_s = time.perf_counter()
        for output in outputs:
            meta = self.active.get(output.request_id)
            if meta is None:
                continue
            token_count = int(meta["last_seen"])
            if output.outputs:
                raw_token_count = max(
                    len(candidate.token_ids) for candidate in output.outputs)
                token_count = max(
                    raw_token_count
                    - int(meta.get("hidden_prefix_tokens", 0)), 0)
            delta = token_count - int(meta["last_seen"])
            if delta > 0:
                meta["last_seen"] = token_count
                self._emit({
                    "event":
                    "token",
                    "attempt_id":
                    output.request_id,
                    "request_id":
                    meta["request_id"],
                    "kind":
                    meta["kind"],
                    "phase":
                    meta["phase"],
                    "delta":
                    delta,
                    "attempt_token_count":
                    token_count,
                    "global_token_count":
                    (int(meta["base_tokens"]) + token_count),
                    "time_s":
                    now_s,
                })
            if output.finished:
                self.active.pop(output.request_id, None)
                self._emit({
                    "event":
                    "finished",
                    "attempt_id":
                    output.request_id,
                    "request_id":
                    meta["request_id"],
                    "kind":
                    meta["kind"],
                    "phase":
                    meta["phase"],
                    "token_count":
                    token_count,
                    "global_token_count":
                    (int(meta["base_tokens"]) + token_count),
                    "time_s":
                    now_s,
                })

    def drain_events(self) -> list[dict[str, Any]]:
        self._emit_new_iteration_logs()
        events = self._events
        self._events = []
        return events

    def drive(self, commands: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for command in commands:
            if command.get("cmd") == "abort":
                self.abort(str(command["attempt_id"]))
            else:
                self.add(command)
        self._step_once()
        return self.drain_events()

    def shutdown(self) -> None:
        return None


async def _stdio_worker_main(args: argparse.Namespace) -> None:
    """Standalone JSON-lines frontend backed by the same async worker."""
    worker = RayEngineWorker(vars(args))

    def emit(payload: dict[str, Any]) -> None:
        print(json.dumps(payload, separators=(",", ":")), flush=True)

    emit(await worker.info())

    async def event_pump() -> None:
        while True:
            for event in await worker.drain_events():
                emit(event)
            await asyncio.sleep(0.002)

    pump_task = asyncio.create_task(event_pump(), name="saturo-event-pump")
    try:
        while True:
            raw_line = await asyncio.to_thread(sys.stdin.readline)
            if not raw_line:
                break
            line = raw_line.strip()
            if not line:
                continue
            command = json.loads(line)
            if command.get("cmd") == "shutdown":
                break
            if command.get("cmd") == "add_batch":
                await worker.submit_batch(list(command.get("commands", ())))
            else:
                await worker.submit(command)
    finally:
        await worker.shutdown()
        for event in await worker.drain_events():
            emit(event)
        pump_task.cancel()
        await asyncio.gather(pump_task, return_exceptions=True)


def worker_main(args: argparse.Namespace) -> None:
    asyncio.run(_stdio_worker_main(args))
