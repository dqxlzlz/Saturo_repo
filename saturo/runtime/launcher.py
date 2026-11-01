"""Worker launch and lifecycle helpers for the Saturo scheduler (Ray-only)."""

from __future__ import annotations

import json
import time
from typing import Any

from saturo.runtime.worker import RayEngineWorker, WorkerPlacement, WorkerState
from saturo.runtime.workload import DECODE, PREFILL, TPUT, repo_root


class WorkerLauncherMixin:

    def _start_initial_workers(self) -> None:
        while self.worker_plan:
            self._start_worker(self.worker_plan.popleft())

    # ------------------------------------------------------------------
    # Ray helpers
    # ------------------------------------------------------------------

    def _ensure_ray(self) -> Any:
        import ray
        if not ray.is_initialized():
            address = getattr(self.args, "ray_address", "auto") or ""
            if address in ("", "local"):
                ray.init(address="local", include_dashboard=False)  # Ignore inherited RAY_ADDRESS.
            else:
                ray.init(address=address)
        return ray

    def _ray_node_options(self, placement: WorkerPlacement) -> dict[str, Any]:
        """Build Ray actor options for the given placement.

        num_gpus tells Ray how many GPUs to reserve for this actor (and
        therefore for the vLLM engine it hosts, including TP workers that
        vLLM spawns internally).  For multi-node clusters the host is used
        to constrain scheduling to a specific node via a custom resource
        label (set up by running ``ray start --resources='{"node:HOSTNAME":1}``
        on each node, or via the cluster YAML).
        """
        num_gpus = max(
            1,
            placement.tensor_parallel_size * placement.pipeline_parallel_size)
        options: dict[str, Any] = {
            "num_gpus": num_gpus,
            # Ray async actors multiplex submit/drain RPCs on one event loop.
            # Request execution itself is managed by AsyncLLMEngine tasks.
            "max_concurrency": 1024,
        }
        host = placement.host or ""
        if host and host not in ("local", "localhost", "127.0.0.1"):
            options["resources"] = {f"node:{host}": 0.001}
        # Ensure remote workers use the project PYTHONPATH and env vars
        import os
        project_root = str(repo_root())
        pythonpath = os.pathsep.join(dict.fromkeys(
            [project_root] + [
                path for path in os.environ.get("PYTHONPATH", "").split(os.pathsep)
                if path
            ]))
        options["runtime_env"] = {
            "env_vars": {
                "PYTHONPATH": pythonpath,
                "VLLM_USE_V1": "0",
                "VLLM_TARGET_DEVICE": "cuda",
            },
        }
        return options

    def _worker_args_dict(self, placement: WorkerPlacement,
                          worker_id: int) -> dict[str, Any]:
        # Prefer the optimized BE chunk latency from the resource profile;
        # fall back to the CLI arg.
        p_pool_be_chunk_ms = self._resolve_p_pool_be_chunk_ms(
            placement.config_name)
        p_pool_be_chunk_ms = getattr(self, "_deployment_start_p_budgets", {}).get(
            placement.config_name, p_pool_be_chunk_ms)
        worker_args = {
            "worker_id": worker_id,
            "kv_handoff": getattr(self.args, "kv_handoff", "dummy"),
            "kv_transfer_dir": getattr(self.args, "kv_transfer_dir", ""),
            "kv_transfer_namespace": getattr(self, "_kv_transfer_namespace", ""),
            "kv_transfer_timeout_s": getattr(self.args, "kv_transfer_timeout_s", 30),
            "worker_role": placement.role,
            "tensor_parallel_size": placement.tensor_parallel_size,
            "pipeline_parallel_size": placement.pipeline_parallel_size,
            "expert_parallel_size": placement.expert_parallel_size,
            "model": str(self.args.model),
            "gpu_memory_utilization": self.args.gpu_memory_utilization,
            "load_format": self.args.load_format,
            "dtype": self.args.dtype,
            "rt_prefill_deadline_s": self.args.rt_prefill_deadline_s,
            "disable_inductor": self.args.disable_inductor,
            "rt_tbt_slo_ms": self.args.rt_tbt_slo_ms,
            "enable_prefill_saturation": getattr(self.args, "enable_prefill_saturation", True),
            "enable_decode_saturation": getattr(
                self.args, "enable_decode_saturation", True),
            "decode_saturation_lookahead_s":
            self.args.decode_saturation_lookahead_s,
            "p_pool_be_chunk_ms": p_pool_be_chunk_ms,
            "p_pool_preemption_margin": self.args.p_pool_preemption_margin,
            "runtime_profile": self.args.runtime_profile,
            "adaptive_role_switching": getattr(
                self.args, "adaptive_role_switching", True),
            "decode_target_async": getattr(self.args, "decode_target_async", False),
            "saturation_update_interval_s": getattr(self.args, "saturation_update_interval_s", 40),
            "saturation_update_timeout_s": getattr(self.args, "saturation_update_timeout_s", 30),
            "pool_layout_generation": getattr(self, "_pool_layout_generation", 0),
        }
        from saturo.predictors.qrf import resolve_qrf_path
        predictor = resolve_qrf_path()
        if predictor:
            worker_args["model_len_predictor"] = predictor
        for optional_key in (
                "max_model_len",
                "max_num_seqs",
                "max_num_batched_tokens",
                "swap_space",
        ):
            optional_value = getattr(self.args, optional_key, None)
            if optional_value is not None:
                worker_args[optional_key] = optional_value
        return worker_args

    def _resolve_p_pool_be_chunk_ms(self, config_name: str) -> float:
        """Look up P-pool BE chunk latency from resource profile."""
        profile_map: dict[str, float] = getattr(self,
                                                "_p_pool_be_chunk_ms_map", {})
        if config_name in profile_map:
            return profile_map[config_name]
        # Profile should always provide this value; fall back to a
        # conservative default only when the profile is missing.
        return 100.0

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------

    def _start_worker(self, placement: WorkerPlacement) -> WorkerState:
        state, ready_ref = self._begin_worker_start(placement)
        ray = self._ensure_ray()
        try:
            info = ray.get(ready_ref, timeout=self.args.worker_start_timeout_s)
            self._finish_worker_start(state, info)
        except BaseException:
            ray.kill(state.actor, no_restart=True)
            state.retired = True
            state.stderr_file.close()
            raise
        return state

    def _begin_worker_start(self, placement: WorkerPlacement):
        """Allocate an actor and return its readiness future without waiting."""
        ray = self._ensure_ray()
        worker_id = self.next_worker_id
        self.next_worker_id += 1
        role = placement.role

        stderr_path = (
            self.worker_log_dir /
            f"worker_{worker_id}_{role}_{placement.host}_tp{placement.tensor_parallel_size}.log"
        )
        stderr_file = stderr_path.open("w")

        actor_cls = ray.remote(RayEngineWorker)
        actor = actor_cls.options(**self._ray_node_options(placement)).remote(
            self._worker_args_dict(placement, worker_id))

        state = WorkerState(
            worker_id=worker_id,
            role=role,
            host=placement.host,
            config_name=placement.config_name,
            tensor_parallel_size=placement.tensor_parallel_size,
            pipeline_parallel_size=placement.pipeline_parallel_size,
            stderr_file=stderr_file,
            actor=actor,
            gpu_id=placement.gpu_id,
            expert_parallel_size=placement.expert_parallel_size,
            ready=False,
        )
        self.workers[worker_id] = state
        return state, actor.info.remote()

    def _finish_worker_start(self, state, info):
        if info.get("worker_id") != state.worker_id or info.get("role") != state.role:
            raise RuntimeError("Worker readiness identity does not match deployment")
        state.runtime_profile_path = str(info.get("runtime_profile_path", ""))
        state.num_gpu_blocks = int(info.get("num_gpu_blocks", 0))
        state.max_model_len = info.get("max_model_len")
        state.block_size = int(info.get("block_size", 16))
        state.max_num_seqs = int(info.get("max_num_seqs", 0))
        state.role_switching_enabled = bool(info.get("role_switching_enabled", False))
        state.ready = True
        print(f"[scheduler] Worker ready: {json.dumps(info, sort_keys=True)}", flush=True)
        state.stderr_file.write(json.dumps(info, sort_keys=True) + "\n")
        state.stderr_file.flush()

    def _has_live_worker(self) -> bool:
        return any(not worker.retired for worker in self.workers.values())

    def _reap_dead_workers(self) -> None:
        # Ray actor failures are caught as exceptions in _service_workers
        # and result in worker retirement there.  Nothing to poll here.
        pass

    def _retire_worker(self,
                       worker: WorkerState,
                       requeue_active: bool = False) -> None:
        if worker.retired:
            return
        # Exclude the failed worker before choosing destinations for requeues.
        worker.retired = True
        self._invalidate_decode_reservations(worker)
        if requeue_active:
            for attempt_id, request_id in list(worker.active_attempts.items()):
                req = self._request_by_id(request_id)
                if req is None or req.is_finished:
                    continue
                req.active_attempt_id = None
                req.assigned_worker_id = None
                if ":p" in attempt_id:
                    self._release_decode_reservation(req)
                    phase = PREFILL
                elif ":d" in attempt_id:
                    phase = DECODE
                else:
                    phase = TPUT
                self._requeue(req, phase)
            worker.active_attempts.clear()
            worker.preempting_attempts.clear()
        try:
            ray = self._ensure_ray()
            ray.get(worker.actor.shutdown.remote(), timeout=5.0)
        except Exception:
            pass
        try:
            self._ensure_ray().kill(worker.actor, no_restart=True)
        except Exception:
            pass
        try:
            worker.stderr_file.close()
        except Exception:
            pass

    def _shutdown_workers(self) -> None:
        if not self.workers:
            return
        ray = self._ensure_ray()
        shutdown_refs: dict[Any, WorkerState] = {}
        for worker in self.workers.values():
            if worker.retired:
                continue
            try:
                info = ray.get(worker.actor.info.remote(), timeout=5.0)
                (self.worker_log_dir / f"worker_{worker.worker_id}_final.json").write_text(
                    json.dumps(info, indent=2))
            except Exception:
                pass
            try:
                shutdown_refs[worker.actor.shutdown.remote()] = worker
            except Exception:
                pass

        for ref, worker in shutdown_refs.items():
            try:
                ray.get(ref, timeout=10.0)
            except Exception:
                pass

        for worker in list(self.workers.values()):
            if not worker.retired:
                try:
                    ray.kill(worker.actor, no_restart=True)
                except Exception:
                    pass
            worker.retired = True
            try:
                worker.stderr_file.close()
            except Exception:
                pass
