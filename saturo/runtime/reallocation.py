"""CPU-only, versioned resource reprofiling and deployment coordination.

The serving thread only polls this service.  It owns all worker RPCs and must
execute a proposal through a drain/restart transaction before committing it.
Timing collection is intentionally absent from this module: a missing or
incompatible measured cache preserves the current deployment.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping
import uuid


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


@dataclasses.dataclass(frozen=True)
class PlacementSpec:
    role: str
    tensor_parallel_size: int
    pipeline_parallel_size: int
    expert_parallel_size: int = 0
    gpu_id: str = ""
    config_name: str = ""

    @property
    def gpu_count(self) -> int:
        return self.tensor_parallel_size * self.pipeline_parallel_size


@dataclasses.dataclass(frozen=True)
class ReallocationProposal:
    proposal_id: str
    base_generation: int
    profile_path: str
    plan_path: str
    manifest_path: str
    placements: tuple[PlacementSpec, ...]
    plan_json: str
    manifest_sha256: str

    def placement_dicts(self) -> list[dict[str, Any]]:
        return [dataclasses.asdict(item) for item in self.placements]

    def plan(self) -> dict[str, Any]:
        return json.loads(self.plan_json)

    def verify(self) -> None:
        """Detect modified staged inputs/results before any worker is stopped."""
        if _sha(self.manifest_path) != self.manifest_sha256:
            raise ValueError("proposal manifest changed after publication")
        manifest = json.loads(Path(self.manifest_path).read_text())
        for row in manifest["files"] + manifest.get("sources", []):
            if _sha(row["path"]) != row["sha256"]:
                raise ValueError(f"proposal artifact changed: {row['path']}")
        plan = json.loads(Path(self.plan_path).read_text())
        if _json(plan) != self.plan_json:
            raise ValueError("proposal plan changed")
        if plan.get("placements") != self.placement_dicts():
            raise ValueError("proposal placements differ from the immutable plan")
        if plan.get("base_generation", self.base_generation) != self.base_generation:
            raise ValueError("proposal generation differs from the immutable plan")


class InsufficientHistory(ValueError):
    pass


def freeze_history(snapshot: Any, *, min_completed: int = 2) -> dict[str, Any]:
    """Copy only observations available at the history end, never output caps."""
    start, end = float(snapshot.start_s), float(snapshot.end_s)
    if not math.isfinite(start + end) or end <= start:
        raise InsufficientHistory("history has no positive observed coverage")
    selected = []
    for original in snapshot.records:
        row = json.loads(_json(original))
        arrival = float(row["arrival_time_s"])
        if not start <= arrival <= end:
            continue
        output = row.get("output_len")
        finished = row.get("finished_time_s")
        # An incomplete request's requested max_tokens is not a completion.
        if finished is None or float(finished) > end:
            output = None
        prompt = int(row["prompt_len"])
        if prompt < 1 or (output is not None and int(output) < 1):
            raise ValueError("observed lengths must be positive")
        selected.append({"run_id": str(row.get("run_id", "")),
                         "request_id": str(row["request_id"]),
                         "arrival_time_s": arrival, "prompt_len": prompt,
                         "output_len": None if output is None else int(output),
                         "finished_time_s": None if output is None else float(finished)})
    selected.sort(key=lambda row: (row["arrival_time_s"], row["run_id"], row["request_id"]))
    identities = [(r["run_id"], r["request_id"]) for r in selected]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate request IDs in observed history")
    completed = sum(r["output_len"] is not None for r in selected)
    if completed < min_completed:
        raise InsufficientHistory(f"observed_completed={completed} below minimum={min_completed}")
    metadata = json.loads(_json(snapshot.metadata()))
    result = {"key": str(snapshot.key), "model": str(snapshot.model),
              "start_s": start, "end_s": end, "records": selected,
              "metadata": metadata, "statistics": {
                  "arrival_rate": len(selected) / (end - start),
                  "arrival_requests": len(selected), "completed_requests": completed,
                  "effective_window_s": end - start}}
    result["fingerprint"] = hashlib.sha256(_json(result).encode()).hexdigest()
    return result


def _normal_dtype(value: Any) -> str:
    value = str(value).lower().removeprefix("torch.")
    return {"half": "float16", "fp16": "float16", "bf16": "bfloat16",
            "float": "float32"}.get(value, value)


def validate_measured_timing(path: Path, *, model: str, config: Any,
                             dtype: str = "auto", gpu_name: str = "",
                             quantization: str = "") -> dict[str, Any]:
    """Require cache identity and measurement evidence, not filename guesses."""
    data = json.loads(path.read_text())
    inner = data.get("time_predictor", data)
    meta = inner.get("meta", {})
    identity = inner.get("profile_identity") or meta.get("profile_identity") or {}
    saved_model = str(identity.get("model", "")).rstrip("/")
    wanted_model = model.rstrip("/")
    if saved_model != wanted_model:
        # Same HF repo and its local checkout are compatible, arbitrary same
        # basenames (which may identify different weights) are not.
        remote, local = ((wanted_model, saved_model) if saved_model.startswith("/")
                         else (saved_model, wanted_model))
        if (remote.startswith("/") or len(remote.split("/")) < 2 or
                not local.endswith("/" + remote)):
            raise ValueError("cached timing model identity mismatch")
    for key, expected in (("tensor_parallel_size", config.tp),
                          ("pipeline_parallel_size", config.pp)):
        if int(identity.get(key, 0)) != expected:
            raise ValueError(f"cached timing {key} mismatch")
    if dtype not in ("", "auto") and _normal_dtype(identity.get("dtype")) != _normal_dtype(dtype):
        raise ValueError("cached timing dtype mismatch")
    if gpu_name and identity.get("gpu_name") != gpu_name:
        raise ValueError("cached timing GPU identity mismatch")
    saved_quantization = str(identity.get("quantization", "unknown"))
    if (quantization or "unknown") != saved_quantization:
        raise ValueError("cached timing quantization mismatch")
    if inner.get("model_type") == "iteration_level":
        # Existing full-model calibrations contain sample/shape provenance
        # rather than per-layer warmup/repeat fields. Keep those original
        # schemas intact; neither is manufactured to imitate the other.
        shapes = inner.get("calibration_shapes", [])
        features = ["prefill_tokens", "prefill_sum_ctx", "n_decode_reqs", "decode_sum_ctx"]
        coef = inner.get("coef", [])
        if (int(inner.get("n_samples", 0)) <= 0 or not shapes or
                inner.get("features") != features or len(coef) != 4 or
                not all(math.isfinite(float(x)) for x in coef + [inner.get("intercept", float("nan"))]) or
                not all(isinstance(shape, list) and len(shape) == 2 and
                        all(int(value) > 0 for value in shape) for shape in shapes)):
            raise ValueError("cached iteration timing has no measured sample/shape provenance")
        measurement = {"schema": "measured_full_model_iteration_fit",
                       **{key: inner[key] for key in ("n_samples", "calibration_shapes", "features", "coef", "intercept")},
                       "meta": meta}
    else:
        row_counts = meta.get("pass_row_counts", {})
        if int(meta.get("repeat", 0)) <= 0 or not row_counts or not all(
                int(row_counts.get(name, 0)) > 0 for name in ("linear", "prefill_attn", "decode_attn")):
            raise ValueError("cached timing has no measured row provenance")
        measurement = {"schema": "measured_per_layer_modules", "meta": meta}
    block = data.get("kv_cache_block", {})
    if int(block.get("block_size", 0)) <= 0 or int(block.get("total_gpu_blocks", 0)) <= 0:
        raise ValueError("cached timing lacks KV capacity")
    if int(inner.get("num_layers", 0)) <= 0:
        raise ValueError("cached timing lacks model layer count")
    return {"identity": identity, "measurement": measurement, "kv_cache_block": block}


def _source_manifest(root: Path) -> list[dict[str, str]]:
    # Cover every Python dependency of simulation/solver/QRF, not only this
    # service.  Hashing is in the background builder, never the serving loop.
    files = sorted((root / "saturo").rglob("*.py"))
    native = root / "vllm" / "core" / "scheduler.py"
    if native.exists():
        files.append(native)
    return [{"path": str(path), "sha256": _sha(path)} for path in files]


def _stable_copy(source: Path, target: Path) -> dict[str, str]:
    before = _sha(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    if _sha(target) != before or _sha(source) != before:
        raise ValueError(f"input changed while snapshotting: {source}")
    return {"source": str(source.resolve()), "source_sha256": before,
            "path": str(target.resolve()), "sha256": before}


def build_reallocation(spec_path: str | Path) -> Path:
    """Subprocess entry point: rebuild mu/phi from observed history on CPU."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("reallocation builder requires CUDA_VISIBLE_DEVICES='' ")
    spec_path = Path(spec_path).resolve()
    spec = json.loads(spec_path.read_text())
    directory = spec_path.parent
    args = argparse.Namespace(**spec["args"])
    history = spec["history"]
    root = Path(__file__).resolve().parents[2]
    sources = _source_manifest(root)
    from saturo.profiling.real_profile_collector import (
        TraceRequest, _profile_p_pool, _resource_rps_values, build_allocation_profile,
        parse_parallel_config, write_history_profile)
    from saturo.predictors.qrf import resolve_qrf_path
    from saturo.predictors.time_predictor import TimePredictor
    from saturo.resource_allocation.planner import SaturoResourcePlanner, parse_config_names

    args.output_dir = str(directory)
    args.target_rps = float(history["statistics"]["arrival_rate"])
    args.num_req = int(history["statistics"]["completed_requests"])
    args._history_metadata = dict(history["metadata"],
                                  observed_fingerprint=history["fingerprint"],
                                  statistics=history["statistics"])
    configs = [parse_parallel_config(name) for name in sorted(
        parse_config_names(args.resource_config_filter))]
    if not configs:
        raise ValueError("resource_config_filter must name cached timing configurations")
    budget = int(args.total_gpus)
    if budget <= 0:
        raise ValueError("periodic reallocation requires an explicit total_gpus budget")
    if any(config.gpus > budget for config in configs):
        raise ValueError("requested timing configuration exceeds GPU budget")
    inputs = []
    # Freeze independent BE/QRF inputs; never train QRF on future completion
    # labels or silently substitute a different length model.
    be_path = Path(str(getattr(args, "be_workload_csv", ""))).expanduser()
    if not be_path.is_file():
        raise ValueError("cached CPU rebuild requires a BE workload CSV")
    inputs.append(_stable_copy(be_path, directory / "inputs" / "be.csv"))
    args.be_workload_csv = inputs[-1]["path"]
    qrf = resolve_qrf_path(getattr(args, "model_len_predictor", None))
    if not qrf or not Path(qrf).is_file():
        raise ValueError("cached CPU rebuild requires an existing QRF artifact")
    inputs.append(_stable_copy(Path(qrf), directory / "inputs" / "qrf.joblib"))
    args.model_len_predictor = inputs[-1]["path"]
    candidates = spec.get("timing_profiles", {})
    cache_root = Path(str(getattr(args, "runtime_profile", "saturo/results/runtime_profiles")))
    cache_paths = ([cache_root] if cache_root.is_file() else
                   sorted(cache_root.rglob("time_predictor*.json")))
    summaries = {}
    origin = history["start_s"]
    requests = [TraceRequest(f"{r['run_id']}:{r['request_id']}",
                             r["arrival_time_s"] - origin,
                             r["prompt_len"], r["output_len"])
                for r in history["records"] if r["output_len"] is not None]
    for config in configs:
        paths = ([Path(candidates[config.name])] if config.name in candidates else cache_paths)
        accepted = []
        reasons = []
        for path in paths:
            if path.name.endswith(".raw_profile.json"):
                continue
            try:
                provenance = validate_measured_timing(
                    path, model=str(args.model), config=config,
                    dtype=str(getattr(args, "dtype", "auto")),
                    gpu_name=str(getattr(args, "resource_reallocation_gpu_name", "")),
                    quantization=str(getattr(args, "quantization", "") or ""))
                accepted.append((path, provenance))
            except (ValueError, KeyError, TypeError, OSError) as exc:
                reasons.append(f"{path}: {exc}")
        if not accepted:
            raise ValueError(f"no compatible measured cache for {config.name}: {reasons}")
        # Explicit mapping is preferable. Otherwise select deterministically
        # by mtime then path and record the exact chosen bytes.
        path, provenance = max(accepted, key=lambda item: (item[0].stat().st_mtime_ns, str(item[0])))
        target = directory / "inputs" / (config.name + ".timing.json")
        copied = _stable_copy(path, target)
        copied["provenance"] = provenance
        inputs.append(copied)
        predictor = TimePredictor(str(target))  # fail closed, no analytic fallback
        rows = []
        for request in requests:
            prefill_ms = max(float(predictor.predict_forward_ms(
                [(request.prompt_len, request.prompt_len)], [])), 0.1)
            rows.append({"request_id": request.request_id,
                         "arrival_time_s": request.arrival_time_s,
                         "prompt_len": request.prompt_len, "output_len": request.output_len,
                         "ContextTokens": request.prompt_len, "GeneratedTokens": request.output_len,
                         "prefill_time_s": prefill_ms / 1000,
                         "ttft_s": prefill_ms / 1000, "generated_tokens": request.output_len})
        write_history_profile(directory / config.name / "history_profile.csv", rows)
        summaries[config.name] = {"runtime_profile": str(target), "runtime_profile_available": True}
    payload = build_allocation_profile(args=args, configs=configs, summaries=summaries,
                                       profile_root=directory, rps_values=_resource_rps_values(args))
    profile_path = directory / "allocation_profiles.json"
    _write(profile_path, payload)
    planner = SaturoResourcePlanner(args, lambda role, tp, pp, ep, **kw: PlacementSpec(
        role, tp, pp, ep, **kw))
    plan = planner.solve_staged_profile(profile_path, total_gpus=budget,
                                       target_rps=args.target_rps)
    selected = plan["allocation"]["p"]
    selected_config = next(config for config in configs if config.name == selected["config"])
    selected_rps = float(selected["per_instance_rt_workload"])
    selected_p = _profile_p_pool(args=args, config=selected_config, summaries=summaries,
        history_profile=directory / selected_config.name / "history_profile.csv",
        rps_values=[selected_rps], config_dir=directory / selected_config.name)
    chunk_ms = float(selected_p.get("be_chunk_latency_ms") or 0)
    if getattr(args, "enable_prefill_saturation", True) and chunk_ms <= 0:
        raise ValueError("selected P allocation has no positive optimized BE budget")
    plan["deployment_controls"] = {
        "p_budgets": {selected_config.name: {"chunk_ms": chunk_ms, "rt_rps": selected_rps,
            "method": "post_solver_cpu_optimization_at_selected_per_instance_rps"}},
        "selected_p_profile": selected_p,
        "capacity_objective_scope": "sampled_mu_phi_table_with_interpolation; selected_P_control_optimized_separately"}
    plan["proposal_id"] = spec["proposal_id"]
    plan["base_generation"] = spec["base_generation"]
    plan_path = directory / "resource_worker_plan.json"
    _write(plan_path, plan)
    # Reject a build straddling a code update; an old deployment must never
    # receive a table generated by an unidentifiable mixture of source code.
    for row in sources:
        if _sha(row["path"]) != row["sha256"]:
            raise ValueError(f"source changed during rebuild: {row['path']}")
    artifacts = [spec_path, profile_path, plan_path]
    artifacts.extend(directory.glob("*/history_profile.csv"))
    artifacts.extend(directory.glob("*/*.json"))
    artifacts.extend(Path(row["path"]) for row in inputs)
    files = [{"path": str(p.resolve()), "sha256": _sha(p)} for p in sorted(set(artifacts))]
    manifest = {"proposal_id": spec["proposal_id"], "base_generation": spec["base_generation"],
                "history": history, "inputs": inputs, "sources": sources,
                "files": files, "method": "cached_measured_timing_cpu_simulation_then_solver",
                "gpu_profiling": False, "created_s": time.time()}
    _write(directory / "manifest.json", manifest)
    return directory / "manifest.json"


class PeriodicReallocationService:
    """One background snapshot/build at a time; poll never waits for it."""

    def __init__(self, args: argparse.Namespace, snapshot_loader: Callable[..., Any], *,
                 output_dir: str | Path | None = None,
                 timing_profiles: Mapping[str, str] | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 wall_clock: Callable[[], float] = time.time):
        # Serialize a deep copy now: later scheduler mutations cannot change
        # an in-flight build. Unsupported (non-JSON) private runtime objects
        # are deliberately excluded.
        self.args = {k: v for k, v in vars(args).items() if not k.startswith("_")}
        self.args = json.loads(_json(self.args))
        self.snapshot_loader = snapshot_loader
        self.clock, self.wall_clock = clock, wall_clock
        self.interval_s = float(getattr(args, "resource_reallocation_interval_s", 1800))
        self.timeout_s = float(getattr(args, "resource_reallocation_build_timeout_s", 300))
        self.min_completed = int(getattr(args, "resource_reallocation_min_completed", 8))
        self.history_window_s = float(getattr(args, "resource_reallocation_history_window_s", 1800))
        if self.interval_s <= 0 or self.timeout_s <= 0 or self.min_completed < 1 or self.history_window_s <= 0:
            raise ValueError("reallocation interval/timeout/minimum history must be positive")
        self.output_dir = Path(output_dir or Path(args.output_dir) / "reallocation").resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timing_profiles = dict(timing_profiles or {})
        self._next_due = self.clock() + self.interval_s
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._results: queue.Queue = queue.Queue()
        self._events: queue.Queue = queue.Queue()

    def _event(self, kind: str, **fields: Any) -> None:
        event = {"event": kind, "time_s": self.wall_clock(), **fields}
        self._events.put(event)
        # Append from builder thread only (poll's stale check is queued too).
        with self._lock:
            with (self.output_dir / "events.jsonl").open("a") as stream:
                stream.write(_json(event) + "\n")

    def drain_events(self) -> list[dict[str, Any]]:
        result = []
        while True:
            try:
                result.append(self._events.get_nowait())
            except queue.Empty:
                return result

    def poll(self, pool_layout_generation: int) -> ReallocationProposal | None:
        if self._closed.is_set():
            return None
        proposal = None
        try:
            ready = self._results.get_nowait()
            if ready.base_generation == pool_layout_generation:
                proposal = ready
            else:
                self._events.put({"event": "proposal_stale", "proposal_id": ready.proposal_id,
                                  "base_generation": ready.base_generation,
                                  "current_generation": pool_layout_generation})
        except queue.Empty:
            pass
        now = self.clock()
        if proposal is None and now >= self._next_due and (
                self._thread is None or not self._thread.is_alive()):
            self._next_due = now + self.interval_s
            self._thread = threading.Thread(target=self._build, args=(int(pool_layout_generation),),
                                            daemon=True, name="saturo-reallocation")
            self._thread.start()
        return proposal

    def _build(self, generation: int) -> None:
        proposal_id = f"{int(self.wall_clock() * 1e6)}_{uuid.uuid4().hex[:8]}"
        directory = self.output_dir / proposal_id
        directory.mkdir()
        self._event("build_started", proposal_id=proposal_id, base_generation=generation)
        try:
            snapshot = self.snapshot_loader(window_s=self.history_window_s, end_s=self.wall_clock())
            history = freeze_history(snapshot, min_completed=self.min_completed)
            if self._closed.is_set():
                self._event("build_cancelled", proposal_id=proposal_id,
                            reason="service_shutdown", stage="history_snapshot")
                return
            spec = {"proposal_id": proposal_id, "base_generation": generation,
                    "args": self.args, "history": history, "timing_profiles": self.timing_profiles}
            spec_path = directory / "build_spec.json"
            _write(spec_path, spec)
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", VLLM_USE_V1="0")
            root = str(Path(__file__).resolve().parents[2])
            environment["PYTHONPATH"] = root + os.pathsep + environment.get("PYTHONPATH", "")
            with (directory / "build.log").open("w") as log:
                process = subprocess.Popen([sys.executable, "-m", "saturo.runtime.reallocation",
                                            "--build", str(spec_path)], cwd=root,
                                           env=environment, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                with self._lock:
                    self._process = process
                if self._closed.is_set():
                    self._terminate(process)
                try:
                    returncode = process.wait(timeout=self.timeout_s)
                except subprocess.TimeoutExpired:
                    self._terminate(process)
                    raise TimeoutError(f"CPU resource rebuild exceeded {self.timeout_s}s")
                finally:
                    with self._lock:
                        self._process = None
            if self._closed.is_set():
                self._event("build_cancelled", proposal_id=proposal_id,
                            reason="service_shutdown", stage="child_wait", returncode=returncode)
                return
            if returncode:
                raise RuntimeError(f"CPU resource rebuild exited {returncode}; see {directory / 'build.log'}")
            manifest_path = directory / "manifest.json"
            plan_path = directory / "resource_worker_plan.json"
            plan = json.loads(plan_path.read_text())
            proposal = ReallocationProposal(proposal_id, generation,
                str(directory / "allocation_profiles.json"), str(plan_path), str(manifest_path),
                tuple(PlacementSpec(**row) for row in plan["placements"]), _json(plan), _sha(manifest_path))
            proposal.verify()
            if not self._closed.is_set():
                self._results.put(proposal)
                self._event("proposal_ready", proposal_id=proposal_id,
                            base_generation=generation, manifest_path=str(manifest_path),
                            manifest_sha256=proposal.manifest_sha256,
                            history_fingerprint=history["fingerprint"],
                            target_rps=history["statistics"]["arrival_rate"],
                            placements=proposal.placement_dicts())
            else:
                self._event("build_cancelled", proposal_id=proposal_id,
                            reason="service_shutdown", stage="proposal_publication", returncode=returncode)
        except InsufficientHistory as exc:
            self._event("build_skipped", proposal_id=proposal_id, reason="insufficient_observed_history", detail=str(exc))
        except Exception as exc:
            if self._closed.is_set():
                self._event("build_cancelled", proposal_id=proposal_id,
                            reason="service_shutdown", stage="build_exception", detail=repr(exc))
            else:
                self._event("build_failed", proposal_id=proposal_id, error=repr(exc))

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=2)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()

    def close(self) -> None:
        self._closed.set()
        with self._lock:
            process = self._process
        if process is not None:
            self._terminate(process)
        # Shutdown callers drain the event queue immediately after close.
        # Give the builder a bounded chance to publish cancellation first.
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)


@dataclasses.dataclass(frozen=True)
class DrainCensus:
    active_attempts: int = 0
    local_pending_requests: int = 0
    queued_commands: int = 0
    inflight_rpcs: int = 0
    reservations: int = 0
    retained_local_be: int = 0

    @property
    def drained(self) -> bool:
        values = dataclasses.astuple(self)
        if any(value < 0 for value in values):
            raise ValueError("negative drain census")
        # Only same-engine, BE-eligible in-place deployments may populate
        # this diagnostic field. Their resident BE/KV remains on that engine.
        return not any((self.active_attempts, self.local_pending_requests,
                        self.queued_commands, self.inflight_rpcs, self.reservations))


class DeploymentTransaction:
    """Pure safety state machine; actions are executed by the serving pump.

    Buffered, unassigned arrivals stay outside DrainCensus and retain their
    original timestamps.  Existing P→D transitions and resident BE continue
    draining.  No partially generated request is ever moved to a new engine.
    """
    TERMINAL = frozenset({"committed", "rolled_back", "resumed_old", "failed"})

    def __init__(self, proposal: ReallocationProposal, *, current_generation: int,
                 gpu_budget: int, drain_timeout_s: float = 180,
                 start_timeout_s: float = 300, clock: Callable[[], float] = time.monotonic,
                 in_place: bool = False, old_placements: tuple[PlacementSpec, ...] = (),
                 controls_only: bool = False):
        if proposal.base_generation != current_generation:
            raise ValueError("stale proposal generation")
        if sum(p.gpu_count for p in proposal.placements) > gpu_budget or gpu_budget <= 0:
            raise ValueError("new placement exceeds GPU budget")
        if any(p.tensor_parallel_size <= 0 or p.pipeline_parallel_size <= 0 or
               p.expert_parallel_size < 0 or p.role not in ("prefill", "decode", "throughput")
               for p in proposal.placements):
            raise ValueError("invalid placement width")
        if drain_timeout_s <= 0 or start_timeout_s <= 0:
            raise ValueError("deployment timeouts must be positive")
        self.in_place = bool(in_place)
        self.controls_only = bool(controls_only)
        if in_place:
            def signature(item):
                return (item.tensor_parallel_size, item.pipeline_parallel_size,
                        item.expert_parallel_size, item.gpu_id)
            if sorted(map(signature, old_placements)) != sorted(map(signature, proposal.placements)):
                raise ValueError("in-place deployment requires identical engine/GPU configuration multiset")
        if controls_only:
            def control_signature(item):
                return (item.role, item.config_name, item.tensor_parallel_size,
                        item.pipeline_parallel_size, item.expert_parallel_size, item.gpu_id)
            if not in_place or sorted(map(control_signature, old_placements)) != sorted(
                    map(control_signature, proposal.placements)):
                raise ValueError("controls-only requires identical roles and engine/GPU configurations")
        proposal.verify()
        self.proposal = proposal
        self.generation = current_generation
        self.gpu_budget = gpu_budget
        self.clock = clock
        self.drain_timeout_s, self.start_timeout_s = drain_timeout_s, start_timeout_s
        self.phase = "verifying_controls" if controls_only else "draining"
        self.deadline = clock() + drain_timeout_s
        self.events = [{"phase": self.phase, "action": "verify_unchanged_layout" if controls_only else "quiesce_new_work"}]
        self.error: str | None = None

    def _transition(self, phase: str, action: str) -> str:
        self.phase = phase
        self.deadline = self.clock() + self.start_timeout_s
        self.events.append({"phase": phase, "action": action, "error": self.error})
        return action

    def commit_failed(self, reason: str) -> str:
        """An executor-side publication failure still requires rollback."""
        if self.phase != "committed":
            raise ValueError("commit_failed requires an uncompleted commit action")
        self.generation -= 1
        self.error = str(reason)
        if self.controls_only:
            return self._transition("restoring_controls", "restore_controls")
        return self._transition("restoring_roles" if self.in_place else "stopping_new",
                                "restore_roles" if self.in_place else "stop_new")

    def abort(self, reason: str) -> str:
        """Fail closed when the old layout itself changes during deployment."""
        self.error = str(reason)
        return self._transition("failed", "fail")

    def advance(self, census: DrainCensus = DrainCensus(), *,
                old_gpu_claims: int | None = None, new_gpu_claims: int | None = None,
                new_ready: bool = False, rollback_ready: bool = False,
                roles_ready: bool = False,
                controls_ready: bool = False,
                failure: str | None = None) -> str:
        """Return wait/stop_old/start_new/stop_new/start_old/commit/resume_old/fail.

        The pump acknowledges actor release with GPU claims=0, including after
        partial startup failure.  Neither new nor rollback actors are started
        while the other layout can still hold GPU allocations.
        """
        if self.phase in self.TERMINAL:
            return "wait"
        if self.phase == "verifying_controls":
            if failure or self.clock() >= self.deadline:
                self.error = failure or "control_apply_timeout"
                return self._transition("restoring_controls", "restore_controls")
            if controls_ready:
                self.generation += 1
                return self._transition("committed", "commit")
        elif self.phase == "restoring_controls":
            if rollback_ready:
                return self._transition("rolled_back", "resume_old")
            if failure or self.clock() >= self.deadline:
                self.error = failure or "control_restore_timeout"
                return self._transition("failed", "fail")
        elif self.phase == "draining":
            if failure or self.clock() >= self.deadline:
                self.error = failure or "drain_timeout"
                return self._transition("resumed_old", "resume_old")
            if census.drained and (self.in_place or not census.retained_local_be):
                if self.in_place:
                    return self._transition("applying_roles", "apply_roles")
                return self._transition("stopping_old", "stop_old")
        elif self.phase == "applying_roles":
            if failure or self.clock() >= self.deadline:
                self.error = failure or "role_apply_timeout"
                return self._transition("restoring_roles", "restore_roles")
            if roles_ready:
                self.generation += 1
                return self._transition("committed", "commit")
        elif self.phase == "restoring_roles":
            if rollback_ready:
                return self._transition("rolled_back", "resume_old")
            if failure or self.clock() >= self.deadline:
                self.error = failure or "role_rollback_timeout"
                return self._transition("failed", "fail")
        elif self.phase == "stopping_old":
            if old_gpu_claims == 0:
                if failure:
                    self.error = failure
                    return self._transition("starting_old", "start_old")
                return self._transition("starting_new", "start_new")
            if failure or self.clock() >= self.deadline:
                self.error = failure or "old_stop_timeout"
                return self._transition("failed", "fail")
        elif self.phase == "starting_new":
            if failure or self.clock() >= self.deadline:
                self.error = failure or "new_start_timeout"
                return self._transition("stopping_new", "stop_new")
            expected = sum(p.gpu_count for p in self.proposal.placements)
            if new_ready and new_gpu_claims == expected:
                self.generation += 1
                return self._transition("committed", "commit")
        elif self.phase == "stopping_new":
            if new_gpu_claims == 0:
                return self._transition("starting_old", "start_old")
            if self.clock() >= self.deadline:
                self.error = "partial_new_stop_timeout"
                return self._transition("failed", "fail")
        elif self.phase == "starting_old":
            if rollback_ready:
                return self._transition("rolled_back", "resume_old")
            if failure or self.clock() >= self.deadline:
                self.error = failure or "rollback_start_timeout"
                return self._transition("failed", "fail")
        return "wait"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", required=True)
    build_reallocation(parser.parse_args().build)
