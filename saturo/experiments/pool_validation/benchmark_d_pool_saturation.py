#!/usr/bin/env python3
"""Simulate, then drive one isolated Ray-hosted D-pool AsyncLLMEngine."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any


REQUEST_MODE_SPLIT_PHASE = "split-phase"
REQUEST_MODE_LOCAL = "local"


if hasattr(argparse, "BooleanOptionalAction"):
    BooleanOptionalAction = argparse.BooleanOptionalAction
else:
    class BooleanOptionalAction(argparse.Action):
        def __init__(self, option_strings, dest, default=None, **kwargs):
            options = []
            for option in option_strings:
                options.append(option)
                if option.startswith("--"):
                    options.append("--no-" + option[2:])
            super().__init__(
                option_strings=options,
                dest=dest,
                nargs=0,
                default=default,
                **kwargs,
            )

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest,
                    not str(option_string).startswith("--no-"))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(q * len(ordered))))
    return float(ordered[idx])


class RayWorkerClient:
    """Direct Ray client for one isolated ``RayEngineWorker`` actor."""

    def __init__(
        self,
        worker_args: dict[str, Any],
        *,
        ray_address: str,
        ray_node_id: str,
        actor_env: dict[str, str],
        num_gpus: int,
        rpc_timeout_s: float,
    ) -> None:
        import ray
        from ray.util.scheduling_strategies import (
            NodeAffinitySchedulingStrategy,
        )
        from saturo.runtime.worker import RayEngineWorker

        self._ray = ray
        self._owns_ray = not ray.is_initialized()
        if self._owns_ray:
            address = str(ray_address or "auto")
            try:
                if address.lower() == "local":
                    ray.init(log_to_driver=False)
                else:
                    ray.init(address=address, log_to_driver=False)
            except ConnectionError:
                if address != "auto":
                    raise
                # Keep the standalone benchmark usable when no external Ray
                # cluster has been started.
                ray.init(log_to_driver=False)

        target_node_id = (
            str(ray_node_id).strip()
            or str(ray.get_runtime_context().get_node_id()))
        actor_cls = ray.remote(RayEngineWorker)
        self.actor = actor_cls.options(
            num_cpus=1,
            num_gpus=int(num_gpus),
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=target_node_id,
                soft=False,
            ),
            runtime_env={"env_vars": actor_env},
        ).remote(worker_args)
        self.rpc_timeout_s = float(rpc_timeout_s)
        self._reader_stop = threading.Event()
        self._reader_thread: threading.Thread | None = None
        self._reader_error = ""
        self._submit_refs: set[Any] = set()

    @property
    def error(self) -> str:
        return self._reader_error

    def info(self, timeout_s: float) -> dict[str, Any]:
        return self._ray.get(
            self.actor.info.remote(), timeout=float(timeout_s))

    def submit(self, command: dict[str, Any]) -> dict[str, Any]:
        return self._ray.get(
            self.actor.submit.remote(command),
            timeout=self.rpc_timeout_s,
        )

    def submit_batch(self, commands: list[dict[str, Any]]) -> None:
        if not commands:
            return
        self._submit_refs.add(self.actor.submit_batch.remote(commands))
        self.poll_submissions()

    @property
    def pending_submissions(self) -> int:
        return len(self._submit_refs)

    def poll_submissions(self) -> None:
        if not self._submit_refs:
            return
        candidates = list(self._submit_refs)[:256]
        ready, _ = self._ray.wait(
            candidates,
            num_returns=len(candidates),
            timeout=0,
        )
        for ref in ready:
            self._submit_refs.discard(ref)
            try:
                self._ray.get(ref)
            except Exception as exc:
                self._reader_error = f"{type(exc).__name__}: {exc}"
                return

    def start_event_reader(
        self,
        events: queue.Queue[dict[str, Any]],
        *,
        poll_interval_s: float = 0.01,
    ) -> None:
        def run() -> None:
            while not self._reader_stop.is_set():
                try:
                    batch = self._ray.get(
                        self.actor.drain_events.remote(),
                        timeout=self.rpc_timeout_s,
                    )
                except Exception as exc:
                    self._reader_error = f"{type(exc).__name__}: {exc}"
                    events.put({
                        "event": "ray_worker_error",
                        "message": self._reader_error,
                        "time_s": time.perf_counter(),
                    })
                    return
                for event in batch:
                    events.put(event)
                self._reader_stop.wait(max(float(poll_interval_s), 0.001))

        self._reader_thread = threading.Thread(target=run, daemon=True)
        self._reader_thread.start()

    def flush_events(self, events: queue.Queue[dict[str, Any]]) -> None:
        for event in self._ray.get(
                self.actor.drain_events.remote(),
                timeout=self.rpc_timeout_s):
            events.put(event)

    def close(self) -> None:
        self._reader_stop.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2)
        try:
            self._ray.get(
                self.actor.shutdown.remote(), timeout=min(
                    self.rpc_timeout_s, 30.0))
        except Exception:
            pass
        try:
            self._ray.kill(self.actor, no_restart=True)
        except Exception:
            pass
        if self._owns_ray:
            self._ray.shutdown()

    def __enter__(self) -> "RayWorkerClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def be_request_command(
    request_id: str,
    *,
    request_mode: str,
    phase: str,
    prompt_len: int,
    output_len: int,
) -> dict[str, Any]:
    """Build one split-phase or ordinary local-lifecycle BE command."""
    if request_mode == REQUEST_MODE_LOCAL:
        wire_phase = "prefill"
        max_tokens = output_len
    else:
        wire_phase = "decode" if phase == "decode" else "prefill"
        max_tokens = output_len if wire_phase == "decode" else 1
    return {
        "cmd": "add",
        "request_id": request_id,
        "attempt_id": request_id,
        "kind": "be",
        "phase": wire_phase,
        "prompt_len": prompt_len,
        "max_tokens": max_tokens,
        "base_tokens": 0,
    }


def load_length_trace(
    csv_path: str,
    *,
    fallback_prompt_len: int,
    fallback_output_len: int,
    max_source: int,
    max_model_len: int | None,
    input_max_percentile: float | None = None,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    if not csv_path:
        return [(fallback_prompt_len, fallback_output_len)], {
            "source_rows": 0,
            "usable_rows": 1,
            "skipped_prompt_too_long": 0,
            "clamped_output_len": 0,
        }

    source: list[tuple[int, int]] = []
    source_rows = 0
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if max_source > 0 and source_rows >= max_source:
                break
            source_rows += 1
            prompt_len = max(1, int(float(row["prompt_len"])))
            output_len = max(1, int(float(row["output_len"])))
            source.append((prompt_len, output_len))

    input_max_tokens_applied = None
    if input_max_percentile is not None:
        q = float(input_max_percentile)
        if not 0.0 < q <= 1.0:
            raise ValueError("--be-input-max-percentile must be in (0, 1]")
        input_max_tokens_applied = int(percentile(
            [prompt_len for prompt_len, _ in source], q))
        source = [
            item for item in source
            if item[0] <= input_max_tokens_applied
        ]

    rows: list[tuple[int, int]] = []
    skipped_prompt_too_long = 0
    clamped_output_len = 0
    for prompt_len, output_len in source:
        if max_model_len is not None:
            if prompt_len >= max_model_len:
                skipped_prompt_too_long += 1
                continue
            max_output = max_model_len - prompt_len
            if output_len > max_output:
                output_len = max(1, max_output)
                clamped_output_len += 1
        rows.append((prompt_len, output_len))

    if not rows:
        raise ValueError(
            f"No usable rows in {csv_path}; try a larger --max-model-len or "
            "a less long-prompt-heavy workload.")
    return rows, {
        "source_rows": source_rows,
        "usable_rows": len(rows),
        "input_max_percentile": input_max_percentile,
        "input_max_tokens_applied": input_max_tokens_applied,
        "skipped_prompt_too_long": skipped_prompt_too_long,
        "clamped_output_len": clamped_output_len,
    }


def load_rt_arrival_trace(
    csv_path: str,
    *,
    num_req: int,
    target_rps: float,
    max_model_len: int | None,
) -> tuple[list[tuple[float, int, int]], dict[str, Any]]:
    """Load RT lengths together with the scaled dataset arrival timeline."""
    if not csv_path:
        raise ValueError(
            "--rt-workload-csv is required: RT arrivals must come from a "
            "dataset trace")

    from saturo.runtime.workload import load_rt_workload_csv
    from saturo.tools.trace_scaler import compute_trace_rps

    raw_arrivals: list[float] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            raw_arrivals.append(float(row.get("arrival_offset_s") or 0.0))
    if not raw_arrivals:
        raise ValueError(f"Empty RT workload CSV: {csv_path}")

    original_rps = compute_trace_rps(raw_arrivals)
    loaded = load_rt_workload_csv(
        csv_path, num_req=num_req, target_rps=target_rps)

    rows: list[tuple[float, int, int]] = []
    skipped_prompt_too_long = 0
    clamped_output_len = 0
    for req in loaded:
        arrival_s = float(req.arrival_offset_s)
        prompt_len = max(1, int(req.prompt_len))
        output_len = max(1, int(req.output_len))
        if max_model_len is not None:
            if prompt_len >= max_model_len:
                skipped_prompt_too_long += 1
                continue
            max_output = max_model_len - prompt_len
            if output_len > max_output:
                output_len = max(1, max_output)
                clamped_output_len += 1
        rows.append((arrival_s, prompt_len, output_len))

    if not rows:
        raise ValueError(f"No usable RT arrivals in {csv_path}")
    if len(rows) != num_req:
        raise ValueError(
            f"only {len(rows)} of num_req={num_req} RT requests fit "
            f"max_model_len={max_model_len}")
    from saturo.tools.trace_scaler import fixed_num_window_s
    window_s = fixed_num_window_s(num_req, target_rps)
    return rows, {
        "arrival_mode": "fixed_num_scaled_dataset_trace",
        "source_rows": len(raw_arrivals),
        "num_req": num_req,
        "original_trace_rps": original_rps,
        "target_rps": target_rps,
        "window_s": window_s,
        "scaled_first_arrival_s": rows[0][0],
        "scaled_last_arrival_s": rows[-1][0],
        "skipped_prompt_too_long": skipped_prompt_too_long,
        "clamped_output_len": clamped_output_len,
    }


def run_warmup(
    worker: RayWorkerClient,
    events: queue.Queue[dict[str, Any]],
    timeout_s: float,
    *,
    rt_decode_only: bool = False,
) -> None:
    request_id = "__warmup_rt_decode" if rt_decode_only else "__warmup_be"
    pending = {request_id}
    worker.submit({
        "cmd": "add",
        "request_id": request_id,
        "attempt_id": request_id,
        "kind": "rt" if rt_decode_only else "be",
        "phase": "decode" if rt_decode_only else "throughput",
        "prompt_len": 32,
        "max_tokens": 4 if rt_decode_only else 2,
        "base_tokens": 0,
    })
    deadline = time.perf_counter() + timeout_s
    while pending and time.perf_counter() < deadline:
        try:
            event = events.get(timeout=0.1)
        except queue.Empty:
            if worker.error:
                raise RuntimeError(
                    f"Ray worker failed during warmup: {worker.error}")
            continue
        if event.get("request_id") in pending and event.get("event") == "finished":
            pending.remove(str(event["request_id"]))
    if pending:
        raise TimeoutError(f"warmup timed out: {sorted(pending)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/models/Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument(
        "--hf-config-path",
        default=None,
        help="Optional Hugging Face config path, independent of the model "
             "identity used for runtime-profile matching.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-wall-s", type=float, default=300.0)
    parser.add_argument("--num-req", type=int, required=True)
    parser.add_argument("--target-rps", type=float, required=True)
    parser.add_argument(
        "--rt-workload-csv",
        required=True,
        help="RT workload CSV containing the original arrival_offset_s "
             "timeline.",
    )
    parser.add_argument("--be-workload-csv", default="")
    parser.add_argument("--max-source-be", type=int, default=0)
    parser.add_argument(
        "--be-input-max-percentile",
        type=float,
        default=None,
        help="Keep BE requests at or below this prompt-length percentile, "
             "e.g. 0.85 for the aligned P85 workload.",
    )
    parser.add_argument("--be-prompt-len", type=int, default=32)
    parser.add_argument("--be-output-len", type=int, default=100)
    parser.add_argument("--be-inflight", type=int, default=128)
    parser.add_argument(
        "--request-mode",
        choices=(REQUEST_MODE_SPLIT_PHASE, REQUEST_MODE_LOCAL),
        default=REQUEST_MODE_LOCAL,
        help="'split-phase' keeps independent prefill-only and decode-ready "
             "BE populations. 'local' submits ordinary end-to-end requests "
             "whose own prefill must finish before they enter decode.")
    parser.add_argument(
        "--enable-be-decode",
        action=BooleanOptionalAction,
        default=True,
        help="Split the constant BE population 50/50 between decode-ready "
             "requests and prefill-only requests. Disable for a prefill-only "
             "BE population.")
    parser.add_argument(
        "--be-decode-inflight",
        type=int,
        default=None,
        help="Optional manual override for the decode-ready inflight count. "
             "When set, --be-inflight becomes the prefill-only target and this "
             "value is the (independent) decode-ready population. When unset "
             "(default), decode-ready is driven to saturation and the "
             "scheduler's KV block budget (fraction*blocks - RT peak - "
             "BE-prefill reserve) decides the actual concurrency.")
    parser.add_argument(
        "--be-decode-block-fraction",
        type=float,
        default=0.95,
        help="Watermark fraction of physical KV blocks the BE-decode budget "
             "may fill. The rest is reserved for the RT peak and the longest "
             "BE-prefill input. Applied on both the real worker and the "
             "simulator so their operating points match.")
    parser.add_argument(
        "--kv-block-size",
        type=int,
        default=16,
        help="KV cache block size, used to convert the longest BE-prefill "
             "prompt into a reserved block count.")
    parser.add_argument(
        "--sim-total-gpu-blocks",
        type=int,
        default=None,
        help="Override the simulator's KV block count so it matches the real "
             "machine (e.g. 5963). Required for a block-matched sim-vs-real "
             "comparison; the worker prints its num_gpu_blocks at startup.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--load-format", default="auto")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--runtime-profile", default="")
    parser.add_argument(
        "--run-simulator",
        action=BooleanOptionalAction,
        default=True,
        help="Run the CPU D-pool simulator before starting the real worker.")
    parser.add_argument(
        "--sim-warmup-s",
        type=float,
        default=10.0,
        help="Warmup excluded from simulator TBT statistics.")
    parser.add_argument(
        "--sim-chunk-sizes",
        default="auto",
        help="BE prefill chunk candidates passed to the D-pool simulator.")
    parser.add_argument("--disable-inductor",
                        action=BooleanOptionalAction,
                        default=True)
    parser.add_argument("--worker-start-timeout-s", type=float, default=600.0)
    parser.add_argument("--worker-warmup-timeout-s", type=float, default=600.0)
    parser.add_argument(
        "--ray-address",
        default=os.environ.get("RAY_ADDRESS", "auto"),
        help="Ray cluster address. Use 'local' to start an isolated local "
             "runtime; 'auto' connects to an existing cluster and falls back "
             "to local when none exists.")
    parser.add_argument(
        "--ray-node-id",
        default="",
        help="Node id for the single D-pool actor. Empty pins it to the "
             "driver's current Ray node.")
    parser.add_argument("--ray-rpc-timeout-s", type=float, default=600.0)
    parser.add_argument("--ray-event-poll-s", type=float, default=0.01)
    parser.add_argument("--rt-tbt-slo-ms", type=float, default=100.0)
    parser.add_argument(
        "--enable-decode-saturation",
        action=BooleanOptionalAction,
        default=True,
        help="Disable for a pure RT-Decode baseline with the ordinary vLLM "
             "decode scheduler and no BE filling.")
    parser.add_argument(
        "--admission-diagnostics",
        action=BooleanOptionalAction,
        default=False,
        help="Emit per-request admission stages and periodic worker/engine/"
             "scheduler queue snapshots.")
    parser.add_argument(
        "--saturation-strategy",
        choices=("oracle", "progressive", "greedy"),
        default="progressive",
        help="Decode-pool saturation forecast provider. Oracle consumes an "
             "explicit future RT schedule generated from this replay; greedy "
             "uses current state only (zero look-ahead).")
    parser.add_argument(
        "--oracle-lookahead-s",
        type=float,
        default=60.0,
        help="Exact future-RT horizon exposed to Oracle saturation.")
    parser.add_argument("--rt-prefill-deadline-s", type=float, default=3600.0)
    args = parser.parse_args()
    if args.num_req <= 0 or args.target_rps <= 0:
        parser.error("--num-req and --target-rps must be positive")
    return args


def run_decode_simulation(
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    """Predict D-pool BE saturation before starting the real worker."""
    request_mode = getattr(
        args, "request_mode", REQUEST_MODE_SPLIT_PHASE)
    if request_mode == REQUEST_MODE_LOCAL:
        from saturo.simulators.decode_simulator.run_local_decode_simulator import (
            run_local_decode_profile as run_decode_profile,
        )
    else:
        from saturo.simulators.decode_simulator.run_decode_simulator import (
            run_decode_profile,
        )

    if not args.runtime_profile:
        raise ValueError(
            "--run-simulator requires --runtime-profile; pass a "
            "time_predictor JSON or use --no-run-simulator")
    from saturo.tools.trace_scaler import fixed_num_window_s
    window_s = fixed_num_window_s(args.num_req, args.target_rps)
    if not 0 <= args.sim_warmup_s < window_s:
        raise ValueError(
            "--sim-warmup-s must be nonnegative and smaller than "
            "num_req / target_rps")
    saturation_strategy = getattr(
        args, "saturation_strategy", "progressive")
    lookahead_s = (
        0.0 if saturation_strategy == "greedy"
        else float(getattr(args, "oracle_lookahead_s", 5.0))
    )

    sim_args = argparse.Namespace(
        rt_trace=str(Path(args.rt_workload_csv).expanduser().resolve()),
        runtime_profile=str(
            Path(args.runtime_profile).expanduser().resolve()),
        be_workload_csv=str(
            Path(args.be_workload_csv).expanduser().resolve()),
        num_req=int(args.num_req),
        warmup_s=float(args.sim_warmup_s),
        target_rps=float(args.target_rps),
        rt_tbt_slo_ms=float(args.rt_tbt_slo_ms),
        rt_max_prompt_len=args.max_model_len,
        rt_max_output_len=args.max_model_len,
        lookahead_window_s=lookahead_s,
        model_len_quantile=0.95,
        model_len_predictor=None,
        model_len_predictor_type=None,
        min_interval_ms=5.0,
        memory_watermark=float(args.be_decode_block_fraction),
        be_prefill_reserve_blocks=int(
            getattr(args, "reserved_be_prefill_blocks", 0)),
        chunk_sizes=str(args.sim_chunk_sizes),
        strategies=str(saturation_strategy),
        total_gpu_blocks=args.sim_total_gpu_blocks,
        block_size=None,
    )
    payload = run_decode_profile(sim_args, print_summaries=False)
    rps_results = payload.get("results", {})
    if not rps_results:
        raise ValueError("D-pool simulator returned no RPS result")
    strategy_results = next(iter(rps_results.values()))
    selected = strategy_results.get(saturation_strategy)
    if not selected:
        raise ValueError(
            "D-pool simulator returned no "
            f"{saturation_strategy} result")

    simulation_path = output_dir / "simulation.json"
    simulation_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plan = {
        "path": str(simulation_path),
        "num_req": int(args.num_req),
        "window_s": window_s,
        "warmup_s": float(args.sim_warmup_s),
        "target_rps": float(args.target_rps),
        "predicted_be_saturation_req_per_s": float(
            selected.get("be_saturation_req_per_s", 0.0) or 0.0),
        "predicted_be_prefill_req_per_s": float(
            selected.get("be_prefill_req_per_s", 0.0) or 0.0),
        "predicted_be_decode_req_per_s": float(
            selected.get("be_decode_req_per_s", 0.0) or 0.0),
        "rt_tbt_violation_rate": float(
            selected.get("rt_tbt_violation_rate", 0.0) or 0.0),
        "rt_slo_violations_total": int(
            selected.get("rt_slo_violations_total", 0) or 0),
        "rps_infeasible": bool(
            selected.get("rps_infeasible", False)),
    }
    if plan["rps_infeasible"]:
        print("[sim] WARNING: simulator judged this RPS INFEASIBLE for the "
              "current config (RT KV demand exceeds usable capacity even "
              "with all swappable BE reclaimed).", flush=True)
    print(f"[sim] D-pool {saturation_strategy} result: "
          f"BE saturation={plan['predicted_be_saturation_req_per_s']:.6f} "
          "req/s, "
          f"TBT violation rate={plan['rt_tbt_violation_rate']:.6f}; "
          f"wrote {simulation_path}", flush=True)
    return plan


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rt_trace, rt_trace_stats = load_rt_arrival_trace(
        args.rt_workload_csv,
        num_req=args.num_req,
        target_rps=args.target_rps,
        max_model_len=args.max_model_len,
    )
    from saturo.tools.trace_scaler import fixed_num_window_s
    args.window_s = fixed_num_window_s(args.num_req, args.target_rps)
    oracle_schedule_path: Path | None = None
    if args.enable_decode_saturation and args.saturation_strategy == "oracle":
        oracle_schedule_path = (
            output_dir / "rt_oracle_schedule.json").resolve()
        oracle_schedule_path.write_text(json.dumps({
            "schema_version": 1,
            "lookahead_window_s": float(args.oracle_lookahead_s),
            "requests": [
                {
                    "arrival_offset_s": float(arrival_s),
                    "prompt_len": int(prompt_len),
                    "output_len": int(output_len),
                }
                for arrival_s, prompt_len, output_len in rt_trace
            ],
        }, indent=2), encoding="utf-8")
    be_trace, be_trace_stats = load_length_trace(
        args.be_workload_csv,
        fallback_prompt_len=args.be_prompt_len,
        fallback_output_len=args.be_output_len,
        max_source=args.max_source_be,
        max_model_len=args.max_model_len,
        input_max_percentile=args.be_input_max_percentile,
    )
    block_size = max(int(args.kv_block_size), 1)
    if args.request_mode == REQUEST_MODE_LOCAL or args.be_inflight <= 0:
        max_be_prompt = None
        args.reserved_be_prefill_blocks = 0
    else:
        max_be_prompt = max((int(p) for p, _ in be_trace), default=1)
        args.reserved_be_prefill_blocks = math.ceil(
            max_be_prompt / block_size)
    max_prompt_log = (
        "not-inspected" if max_be_prompt is None else str(max_be_prompt))
    print(
        f"BE request mode={args.request_mode}; "
        f"BE-decode KV budget: block_fraction={args.be_decode_block_fraction} "
        f"reserved_be_prefill_blocks={args.reserved_be_prefill_blocks} "
        f"(max_be_prompt={max_prompt_log}, block_size={block_size})",
        flush=True)
    simulation_plan = None
    if args.run_simulator:
        print("=== Running D-pool simulator before worker startup ===",
              flush=True)
        simulation_plan = run_decode_simulation(args, output_dir)

    worker_env = {
        key: value
        for key, value in os.environ.items()
        if (key.startswith("SATURO_")
            or key.startswith("VLLM_")
            or key.startswith("TORCH")
            or key in ("LD_LIBRARY_PATH", "PYTHONPATH"))
    }
    worker_env["PYTHONPATH"] = str(Path.cwd())
    worker_env["SATURO_DECODE_BLOCK_FRACTION"] = str(
        args.be_decode_block_fraction)
    worker_env["SATURO_DECODE_BE_PREFILL_RESERVE_BLOCKS"] = str(
        int(getattr(args, "reserved_be_prefill_blocks", 0)))
    if oracle_schedule_path is not None:
        worker_env["SATURO_DECODE_ORACLE_SCHEDULE"] = str(
            oracle_schedule_path)
    else:
        worker_env.pop("SATURO_DECODE_ORACLE_SCHEDULE", None)
    if args.admission_diagnostics:
        worker_env["SATURO_ADMISSION_DIAGNOSTICS"] = "1"
    else:
        worker_env.pop("SATURO_ADMISSION_DIAGNOSTICS", None)
    worker_env["VLLM_USE_V1"] = "0"
    worker_env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if args.disable_inductor:
        worker_env.setdefault("TORCHDYNAMO_DISABLE", "1")
        worker_env.setdefault("TORCH_COMPILE_DISABLE", "1")

    worker_args = vars(args).copy()
    worker_args.update({
        "worker_id": 0,
        "worker_role": "decode",
        "decode_saturation_lookahead_s": (
            0.0 if args.saturation_strategy == "greedy"
            else args.oracle_lookahead_s),
    })

    events: queue.Queue[dict[str, Any]] = queue.Queue()
    required_gpus = (
        int(args.tensor_parallel_size) * int(args.pipeline_parallel_size))
    with RayWorkerClient(
        worker_args,
        ray_address=args.ray_address,
        ray_node_id=args.ray_node_id,
        actor_env=worker_env,
        num_gpus=required_gpus,
        rpc_timeout_s=args.ray_rpc_timeout_s,
    ) as worker:
        try:
            worker_info = worker.info(args.worker_start_timeout_s)
            (output_dir / "ray_worker_info.json").write_text(
                json.dumps(worker_info, indent=2), encoding="utf-8")
            worker.start_event_reader(
                events, poll_interval_s=args.ray_event_poll_s)
            run_warmup(
                worker,
                events,
                args.worker_warmup_timeout_s,
                rt_decode_only=not args.enable_decode_saturation,
            )

            active_be = {
                "decode": set(),
                "prefill": set(),
                "local": set(),
            }
            active_rt: set[str] = set()
            rt_arrivals: dict[str, float] = {}
            rt_token_times: dict[str, list[float]] = {}
            be_phase_by_request: dict[str, str] = {}
            be_target_tokens_by_request: dict[str, int] = {}
            be_window_tokens_by_request: dict[str, int] = {}
            rt_request_stage_times: dict[str, dict[str, float]] = {}
            admission_snapshots: list[dict[str, Any]] = []
            event_rows: list[dict[str, Any]] = []
            be_tokens = 0
            be_prefill_tokens = 0
            be_decode_tokens = 0
            be_dummy_handoff_tokens = 0
            rt_decode_tokens = 0
            be_completed = 0
            be_completed_in_window = 0
            be_submitted = 0
            be_completed_by_phase = {
                "decode": 0,
                "prefill": 0,
                "local": 0,
            }
            be_completed_in_window_by_phase = {
                "decode": 0,
                "prefill": 0,
                "local": 0,
            }
            be_submitted_by_phase = {
                "decode": 0,
                "prefill": 0,
                "local": 0,
            }
            be_trace_idx = {"decode": 0, "prefill": 0, "local": 0}
            rt_submitted = 0
            rt_completed = 0
            rt_trace_idx = 0
            next_be_id = 0
            crash_error = ""
            eviction_steps = 0
            total_preempted = 0
            max_swapped = 0
            max_running = 0
            max_pending = 0
            target_tbt_ms: list[float] = []
            actual_iteration_ms: list[float] = []
            predicted_iteration_ms: list[float] = []
            scheduler_wall_ms: list[float] = []
            runtime_overhead_ms: list[float] = []
            step_phase_keys = (
                "pre_schedule_ms", "schedule_ms", "prepare_execute_ms",
                "pre_execute_cuda_sync_ms", "execute_model_await_ms",
                "post_execute_cuda_sync_ms", "iteration_log_build_ms",
                "post_model_state_ms", "append_output_ms",
                "advance_to_next_step_ms",
                "sync_output_process_and_stats_ms", "step_finalize_ms",
                "async_output_callback_ms", "async_output_core_ms",
                "stream_dispatch_ms",
                "outer_loop_gap_ms", "outer_admission_ms",
                "outer_step_async_ms", "outer_stream_dispatch_ms",
                "outer_engine_step_total_ms",
            )
            step_phase_samples: dict[str, list[float]] = {
                key: [] for key in step_phase_keys
            }
            rt_stream_queue_wait_ms: list[float] = []
            rt_stream_put_to_worker_ms: list[float] = []
            rt_stream_get_to_worker_ms: list[float] = []
            started = time.perf_counter()
            deadline = started + args.max_wall_s

            def rel_now() -> float:
                return time.perf_counter() - started

            def drain_events() -> None:
                while True:
                    try:
                        event = events.get_nowait()
                    except queue.Empty:
                        break
                    record_event(event)

            def record_event(event: dict[str, Any]) -> None:
                nonlocal be_tokens, be_prefill_tokens, be_decode_tokens
                nonlocal be_dummy_handoff_tokens
                nonlocal rt_decode_tokens, be_completed
                nonlocal be_completed_in_window, rt_completed
                nonlocal eviction_steps, total_preempted, max_swapped
                nonlocal max_running, max_pending
                request_id = str(event.get("request_id", ""))
                event_time = float(event.get("time_s") or time.perf_counter()) - started
                row = dict(event)
                row["relative_time_s"] = event_time
                event_rows.append(row)

                if (event.get("event") == "request_stage"
                        and request_id.startswith("rt_")):
                    stage = str(event.get("stage", ""))
                    if stage:
                        rt_request_stage_times.setdefault(
                            request_id, {})[stage] = event_time

                if event.get("event") == "admission_snapshot":
                    admission_snapshots.append(row)

                if event.get("event") == "iteration_log":
                    if event_time <= args.window_s:
                        be_prefill_tokens += int(
                            event.get("be_prefill_tokens", 0) or 0)
                        be_decode_tokens += int(
                            event.get("be_decode_tokens", 0) or 0)
                        be_dummy_handoff_tokens += int(
                            event.get("be_dummy_handoff_tokens", 0) or 0)
                        rt_decode_tokens += int(
                            event.get("rt_decode_tokens", 0) or 0)
                        actual_iteration_ms.append(float(
                            event.get("actual_time_ms", 0.0) or 0.0))
                        predicted = event.get("predicted_time_ms")
                        if predicted is not None:
                            predicted_iteration_ms.append(float(predicted))
                        scheduler_wall_ms.append(float(
                            event.get("scheduler_wall_ms", 0.0) or 0.0))
                        overhead = event.get("runtime_overhead_ms")
                        if overhead is not None:
                            runtime_overhead_ms.append(float(overhead))
                        for key in step_phase_keys:
                            value = event.get(key)
                            if value is not None:
                                step_phase_samples[key].append(float(value))
                        decision = event.get(
                            "decode_saturation_decision") or {}
                        if decision.get("target_interval_ms") is not None:
                            target_tbt_ms.append(float(
                                decision["target_interval_ms"]))
                    preempted = int(event.get("preempted", 0) or 0)
                    if preempted > 0:
                        eviction_steps += 1
                        total_preempted += preempted
                    swapped_size = event.get(
                        "scheduler_swapped_queue_size",
                        event.get("num_swapped_reqs", 0),
                    )
                    max_swapped = max(max_swapped, int(swapped_size or 0))
                    running_size = event.get(
                        "scheduler_running_queue_size",
                        event.get("running_queue_size", 0),
                    )
                    max_running = max(max_running, int(running_size or 0))
                    waiting_size = event.get(
                        "scheduler_waiting_queue_size",
                        event.get("num_waiting_reqs", 0),
                    )
                    waiting_size = int(waiting_size or 0)
                    waiting_size += int(event.get(
                        "scheduler_waiting_rt_decode_admission_size", 0) or 0)
                    waiting_size += int(event.get(
                        "scheduler_waiting_be_decode_admission_size", 0) or 0)
                    max_pending = max(max_pending, waiting_size)

                if event.get("event") == "token":
                    delta = int(event.get("delta", 0))
                    if request_id.startswith("be_") and event_time <= args.window_s:
                        be_tokens += delta
                        be_window_tokens_by_request[request_id] = (
                            be_window_tokens_by_request.get(request_id, 0)
                            + delta)
                    elif request_id.startswith("rt_"):
                        for field, samples in (
                            ("stream_queue_wait_ms",
                             rt_stream_queue_wait_ms),
                            ("stream_put_to_worker_ms",
                             rt_stream_put_to_worker_ms),
                            ("stream_get_to_worker_ms",
                             rt_stream_get_to_worker_ms),
                        ):
                            value = event.get(field)
                            if value is not None:
                                samples.append(float(value))
                        token_times = rt_token_times.setdefault(request_id, [])
                        if not token_times:
                            rt_request_stage_times.setdefault(
                                request_id, {})["first_token"] = event_time
                        token_times.append(event_time)

                if event.get("event") == "finished":
                    if request_id.startswith("be_"):
                        if request_id.startswith("be_local_"):
                            phase = "local"
                        else:
                            phase = str(event.get("phase", "prefill"))
                            phase = (
                                "decode" if phase == "decode" else "prefill")
                        active_be[phase].discard(request_id)
                        be_completed += 1
                        be_completed_by_phase[phase] += 1
                        if event_time <= args.window_s:
                            be_completed_in_window += 1
                            be_completed_in_window_by_phase[phase] += 1
                    elif request_id.startswith("rt_"):
                        active_rt.discard(request_id)
                        rt_completed += 1
                        rt_request_stage_times.setdefault(
                            request_id, {})["finished"] = event_time

            def send_timed_batch(commands: list[dict[str, Any]]) -> None:
                send_started_s = time.perf_counter()
                if args.admission_diagnostics:
                    for command in commands:
                        request_id = str(command.get("request_id", ""))
                        if not request_id.startswith("rt_"):
                            continue
                        record_event({
                            "event": "request_stage",
                            "stage": "driver_send_started",
                            "request_id": request_id,
                            "attempt_id": str(
                                command.get("attempt_id", request_id)),
                            "kind": str(command.get("kind", "")),
                            "phase": str(command.get("phase", "")),
                            "time_s": send_started_s,
                        })
                worker.submit_batch(commands)
                if not args.admission_diagnostics:
                    return
                # ``.remote()`` has returned an ObjectRef: Ray owns the RPC,
                # while worker_submitted records actual actor admission.
                enqueued_at_s = time.perf_counter()
                for command in commands:
                    request_id = str(command.get("request_id", ""))
                    if not request_id.startswith("rt_"):
                        continue
                    record_event({
                        "event": "request_stage",
                        "stage": "ray_rpc_enqueued",
                        "request_id": request_id,
                        "attempt_id": str(
                            command.get("attempt_id", request_id)),
                        "kind": str(command.get("kind", "")),
                        "phase": str(command.get("phase", "")),
                        "time_s": enqueued_at_s,
                    })

            def build_be(phase: str) -> dict[str, Any]:
                nonlocal next_be_id, be_submitted
                if args.request_mode == REQUEST_MODE_LOCAL:
                    phase = "local"
                else:
                    phase = "decode" if phase == "decode" else "prefill"
                request_id = f"be_{phase}_{next_be_id}"
                next_be_id += 1
                be_submitted += 1
                be_submitted_by_phase[phase] += 1
                active_be[phase].add(request_id)
                trace_idx = be_trace_idx[phase]
                prompt_len, output_len = be_trace[trace_idx % len(be_trace)]
                be_trace_idx[phase] += 1
                be_phase_by_request[request_id] = phase
                if phase in ("decode", "local"):
                    be_target_tokens_by_request[request_id] = output_len
                else:
                    be_target_tokens_by_request[request_id] = prompt_len
                be_window_tokens_by_request[request_id] = 0
                return be_request_command(
                    request_id,
                    request_mode=args.request_mode,
                    phase=phase,
                    prompt_len=prompt_len,
                    output_len=output_len,
                )

            def build_rt(req: tuple[float, int, int]) -> dict[str, Any]:
                nonlocal rt_submitted, rt_trace_idx
                arrival_s, prompt_len, output_len = req
                request_id = f"rt_{rt_submitted}"
                rt_submitted += 1
                active_rt.add(request_id)
                rt_arrivals[request_id] = arrival_s
                rt_token_times[request_id] = []
                rt_trace_idx += 1
                return {
                    "cmd": "add",
                    "request_id": request_id,
                    "attempt_id": request_id,
                    "kind": "rt",
                    "phase": "decode",
                    "prompt_len": prompt_len,
                    "max_tokens": output_len,
                    "base_tokens": 0,
                }

            try:
                if args.request_mode == REQUEST_MODE_LOCAL:
                    decode_inflight = 0
                    prefill_inflight = 0
                    be_inflight_targets = {
                        "local": max(1, int(args.be_inflight)),
                    }
                else:
                    saturating_decode = (
                        int(args.max_num_seqs) if args.max_num_seqs else 128)
                    if args.be_decode_inflight is not None:
                        decode_inflight = max(
                            0, int(args.be_decode_inflight))
                        prefill_inflight = args.be_inflight
                    elif args.enable_be_decode:
                        # Saturate: keep a large decode-ready pool queued and
                        # let the scheduler's KV budget decide concurrency.
                        decode_inflight = saturating_decode
                        prefill_inflight = args.be_inflight
                    else:
                        decode_inflight = 0
                        prefill_inflight = args.be_inflight
                    be_inflight_targets = {
                        "decode": decode_inflight,
                        "prefill": prefill_inflight,
                    }
                # Preserve the trace boundary at t=0.  Submitting a large BE
                # population first can keep the async worker busy registering
                # BE tasks for many forward steps before rt_0 reaches the
                # engine, creating an artificial multi-second RT startup lag.
                initial_commands: list[dict[str, Any]] = []
                initial_elapsed = rel_now()
                while (rt_trace_idx < len(rt_trace)
                       and rt_trace[rt_trace_idx][0]
                       <= initial_elapsed + 1e-9):
                    initial_commands.append(build_rt(rt_trace[rt_trace_idx]))
                for phase, target in be_inflight_targets.items():
                    while len(active_be[phase]) < target:
                        initial_commands.append(build_be(phase))
                send_timed_batch(initial_commands)

                while time.perf_counter() < deadline:
                    elapsed = rel_now()
                    due_commands: list[dict[str, Any]] = []
                    while (rt_trace_idx < len(rt_trace)
                           and rt_trace[rt_trace_idx][0] <= elapsed + 1e-9):
                        due_commands.append(build_rt(rt_trace[rt_trace_idx]))

                    if elapsed < args.window_s:
                        for phase, target in be_inflight_targets.items():
                            while len(active_be[phase]) < target:
                                due_commands.append(build_be(phase))
                    send_timed_batch(due_commands)

                    worker.poll_submissions()
                    if worker.error:
                        crash_error = worker.error
                        break

                    try:
                        event = events.get(timeout=0.01)
                    except queue.Empty:
                        if elapsed >= args.window_s and not active_rt:
                            break
                        continue
                    record_event(event)
            except Exception as exc:
                crash_error = f"{type(exc).__name__}: {exc}"
            finally:
                worker.poll_submissions()
                try:
                    worker.flush_events(events)
                except Exception as exc:
                    if not crash_error:
                        crash_error = f"{type(exc).__name__}: {exc}"
                drain_events()

            rt_tbts_ms = []
            rt_ttfts_ms = []
            rt_request_tbt_rows = []
            for request_id, token_times in rt_token_times.items():
                if token_times:
                    rt_ttfts_ms.append(
                        max(token_times[0] - rt_arrivals[request_id], 0.0) * 1000.0)
                request_tbts_ms = [
                    (right - left) * 1000.0
                    for left, right in zip(token_times, token_times[1:])
                ]
                rt_tbts_ms.extend(request_tbts_ms)
                rt_request_tbt_rows.append({
                    "request_id": request_id,
                    "token_count": len(token_times),
                    "tbt_sample_count": len(request_tbts_ms),
                    "tbt_mean_ms": (
                        sum(request_tbts_ms) / max(len(request_tbts_ms), 1)),
                    "tbt_p50_ms": percentile(request_tbts_ms, 0.50),
                    "tbt_p90_ms": percentile(request_tbts_ms, 0.90),
                    "tbt_p95_ms": percentile(request_tbts_ms, 0.95),
                    "tbt_p99_ms": percentile(request_tbts_ms, 0.99),
                    "tbt_max_ms": max(request_tbts_ms, default=0.0),
                    "tbt_slo_violations": sum(
                        value > args.rt_tbt_slo_ms
                        for value in request_tbts_ms),
                })
            rt_tbt_violations = sum(
                1 for item in rt_tbts_ms if item > args.rt_tbt_slo_ms)
            rt_tbt_violation_rate = (
                rt_tbt_violations / max(len(rt_tbts_ms), 1))

            admission_stages = (
                "driver_send_started",
                "ray_rpc_enqueued",
                "worker_submitted",
                "worker_task_started",
                "request_tracker_enqueued",
                "request_tracker_dequeued",
                "scheduler_added",
                "first_token",
                "finished",
            )
            rt_request_stage_rows: list[dict[str, Any]] = []
            admission_terminal_counts = {
                "diagnostics_disabled": 0,
                "not_driver_send_started": 0,
                "ray_rpc_not_enqueued": 0,
                "ray_rpc_enqueued_missing_worker_stage": 0,
                "worker_task_not_started": 0,
                "before_request_tracker_enqueue": 0,
                "in_request_tracker_queue": 0,
                "in_add_request_async": 0,
                "scheduler_added_no_token": 0,
                "first_token_not_finished": 0,
                "finished": 0,
            }

            def delay_ms(stages: dict[str, float], left: str,
                         right: str) -> float | None:
                if left not in stages or right not in stages:
                    return None
                return max(stages[right] - stages[left], 0.0) * 1000.0

            for request_idx in range(rt_submitted):
                request_id = f"rt_{request_idx}"
                stages = rt_request_stage_times.get(request_id, {})
                if not args.admission_diagnostics:
                    terminal = "diagnostics_disabled"
                elif "finished" in stages:
                    terminal = "finished"
                elif "first_token" in stages:
                    terminal = "first_token_not_finished"
                elif "scheduler_added" in stages:
                    terminal = "scheduler_added_no_token"
                elif "request_tracker_dequeued" in stages:
                    terminal = "in_add_request_async"
                elif "request_tracker_enqueued" in stages:
                    terminal = "in_request_tracker_queue"
                elif "worker_task_started" in stages:
                    terminal = "before_request_tracker_enqueue"
                elif "worker_submitted" in stages:
                    terminal = "worker_task_not_started"
                elif "ray_rpc_enqueued" in stages:
                    terminal = "ray_rpc_enqueued_missing_worker_stage"
                elif "driver_send_started" in stages:
                    terminal = "ray_rpc_not_enqueued"
                else:
                    terminal = "not_driver_send_started"
                admission_terminal_counts[terminal] += 1

                row: dict[str, Any] = {
                    "request_id": request_id,
                    "trace_arrival_s": rt_arrivals.get(request_id),
                    "terminal_stage": terminal,
                }
                for stage in admission_stages:
                    row[f"{stage}_s"] = stages.get(stage)
                row.update({
                    "driver_lateness_ms": (
                        max(stages["driver_send_started"]
                            - rt_arrivals[request_id], 0.0) * 1000.0
                        if ("driver_send_started" in stages
                            and request_id in rt_arrivals) else None),
                    "ray_submit_rpc_ms": delay_ms(
                        stages, "driver_send_started", "ray_rpc_enqueued"),
                    "send_start_to_worker_ms": delay_ms(
                        stages, "driver_send_started", "worker_submitted"),
                    "worker_task_start_ms": delay_ms(
                        stages, "worker_submitted", "worker_task_started"),
                    "tracker_enqueue_ms": delay_ms(
                        stages, "worker_task_started",
                        "request_tracker_enqueued"),
                    "tracker_queue_ms": delay_ms(
                        stages, "request_tracker_enqueued",
                        "request_tracker_dequeued"),
                    "engine_add_request_ms": delay_ms(
                        stages, "request_tracker_dequeued",
                        "scheduler_added"),
                    "scheduler_to_first_token_ms": delay_ms(
                        stages, "scheduler_added", "first_token"),
                })
                rt_request_stage_rows.append(row)

            admission_stage_counts = {
                stage: sum(
                    stage in rt_request_stage_times.get(f"rt_{idx}", {})
                    for idx in range(rt_submitted))
                for stage in admission_stages
            }
            if not args.admission_diagnostics:
                admission_stage_counts = {}
            admission_snapshot_fields = (
                "worker_active_requests",
                "worker_request_tasks",
                "worker_tasks_done",
                "worker_tasks_not_started",
                "worker_active_zero_token",
                "request_tracker_new_queue",
                "request_tracker_streams",
                "scheduler_running",
                "scheduler_waiting",
                "scheduler_waiting_rt_decode_admission",
                "scheduler_swapped",
            )
            admission_snapshot_max = {
                field: max(
                    (int(float(row[field]))
                     for row in admission_snapshots
                     if row.get(field) not in (None, "")),
                    default=0,
                )
                for field in admission_snapshot_fields
            }
            admission_snapshot_last = (
                {field: admission_snapshots[-1].get(field)
                 for field in ("relative_time_s",) + admission_snapshot_fields}
                if admission_snapshots else None)

            avg_be_prompt_len = (
                sum(prompt for prompt, _ in be_trace) / len(be_trace))
            avg_be_output_len = (
                sum(output for _, output in be_trace) / len(be_trace))
            be_prefill_token_per_s = (
                be_prefill_tokens / max(args.window_s, 1e-9))
            be_decode_token_per_s = (
                be_decode_tokens / max(args.window_s, 1e-9))
            be_prefill_req_per_s = (
                be_prefill_token_per_s / max(avg_be_prompt_len, 1.0))
            be_decode_token_normalized_req_per_s = (
                be_decode_token_per_s / max(avg_be_output_len, 1.0))
            be_decode_fractional_completed = sum(
                min(
                    be_window_tokens_by_request.get(request_id, 0)
                    / max(target_tokens, 1),
                    1.0,
                )
                for request_id, target_tokens
                in be_target_tokens_by_request.items()
                if be_phase_by_request.get(request_id) == "decode"
            )
            be_decode_req_per_s = (
                be_decode_fractional_completed
                / max(args.window_s, 1e-9))
            be_work_token_per_s = (
                (be_prefill_tokens + be_decode_tokens)
                / max(args.window_s, 1e-9))
            be_e2e_req_per_s = (
                be_completed_in_window / max(args.window_s, 1e-9))
            if args.request_mode == REQUEST_MODE_LOCAL:
                be_throughput_req_per_s = be_e2e_req_per_s
                be_saturation_req_per_s = be_e2e_req_per_s
            elif args.be_inflight <= 0:
                # A phase-isolated decode pool has no Prefill leg on this
                # instance. Its useful request rate is therefore the Decode
                # token-flow rate, not min(0, decode_req_per_s).
                be_throughput_req_per_s = be_decode_req_per_s
                be_saturation_req_per_s = be_decode_req_per_s
            else:
                be_throughput_req_per_s = min(
                    be_prefill_req_per_s, be_decode_req_per_s)
                be_saturation_req_per_s = be_throughput_req_per_s

            summary = {
                "model": args.model,
                "hf_config_path": args.hf_config_path,
                "load_format": args.load_format,
                "dtype": args.dtype,
                "runtime_profile": (
                    str(Path(args.runtime_profile).expanduser().resolve())
                    if args.runtime_profile else None),
                "pool": "decode",
                "request_mode": args.request_mode,
                "saturation_strategy": args.saturation_strategy,
                "decode_saturation_enabled": args.enable_decode_saturation,
                "admission_diagnostics_enabled":
                args.admission_diagnostics,
                "admission_stage_counts": admission_stage_counts,
                "admission_terminal_counts": admission_terminal_counts,
                "admission_snapshot_count": len(admission_snapshots),
                "admission_snapshot_max": admission_snapshot_max,
                "admission_snapshot_last": admission_snapshot_last,
                "oracle_lookahead_s": (
                    float(args.oracle_lookahead_s)
                    if args.saturation_strategy == "oracle" else None),
                "oracle_schedule_path": (
                    str(oracle_schedule_path)
                    if oracle_schedule_path is not None else None),
                "instances": 1,
                "gpu_id": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "tensor_parallel_size": args.tensor_parallel_size,
                "pipeline_parallel_size": args.pipeline_parallel_size,
                "num_req": args.num_req,
                "window_s": args.window_s,
                "elapsed_s": rel_now(),
                "max_model_len": args.max_model_len,
                "max_num_seqs": args.max_num_seqs,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "rt_workload_csv": args.rt_workload_csv,
                "be_workload_csv": args.be_workload_csv,
                "rt_trace": rt_trace_stats,
                "be_trace": be_trace_stats,
                "avg_be_prompt_len": avg_be_prompt_len,
                "avg_be_output_len": avg_be_output_len,
                "target_rps": args.target_rps,
                "rt_tbt_slo_ms": args.rt_tbt_slo_ms,
                "rt_submitted": rt_submitted,
                "rt_completed": rt_completed,
                "rt_ttft_ms_p50": percentile(rt_ttfts_ms, 0.50),
                "rt_ttft_ms_p95": percentile(rt_ttfts_ms, 0.95),
                "rt_tbt_ms_p50": percentile(rt_tbts_ms, 0.50),
                "rt_tbt_ms_mean": (
                    sum(rt_tbts_ms) / max(len(rt_tbts_ms), 1)),
                "rt_tbt_ms_p90": percentile(rt_tbts_ms, 0.90),
                "rt_tbt_ms_p95": percentile(rt_tbts_ms, 0.95),
                "rt_tbt_ms_p99": percentile(rt_tbts_ms, 0.99),
                "rt_tbt_sample_count": len(rt_tbts_ms),
                "rt_tbt_slo_violations": rt_tbt_violations,
                "rt_tbt_slo_violation_rate": rt_tbt_violation_rate,
                "rt_tbt_slo_attainment": 1.0 - rt_tbt_violation_rate,
                "be_submitted": be_submitted,
                "be_completed": be_completed,
                "be_completed_in_window": be_completed_in_window,
                "be_submitted_decode": be_submitted_by_phase["decode"],
                "be_submitted_prefill": be_submitted_by_phase["prefill"],
                "be_submitted_local": be_submitted_by_phase["local"],
                "be_completed_decode": be_completed_by_phase["decode"],
                "be_completed_prefill": be_completed_by_phase["prefill"],
                "be_completed_local": be_completed_by_phase["local"],
                "be_completed_local_in_window": (
                    be_completed_in_window_by_phase["local"]),
                "be_generated_tokens": be_tokens,
                "be_throughput_tokens_per_s": (
                    be_tokens / max(args.window_s, 1e-9)),
                "be_work_throughput_tokens_per_s": be_work_token_per_s,
                "be_prefill_tokens": be_prefill_tokens,
                "be_decode_tokens": be_decode_tokens,
                "be_dummy_handoff_tokens": be_dummy_handoff_tokens,
                "be_prefill_throughput_tokens_per_s": (
                    be_prefill_token_per_s),
                "be_decode_throughput_tokens_per_s": (
                    be_decode_token_per_s),
                "be_prefill_throughput_req_per_s": be_prefill_req_per_s,
                "be_decode_throughput_req_per_s": be_decode_req_per_s,
                "be_decode_fractional_completed_requests": (
                    be_decode_fractional_completed),
                "be_decode_token_normalized_req_per_s": (
                    be_decode_token_normalized_req_per_s),
                "be_throughput_req_per_s": be_throughput_req_per_s,
                "be_saturation_req_per_s": be_saturation_req_per_s,
                "rt_decode_tokens": rt_decode_tokens,
                "avg_target_tbt_ms": (
                    sum(target_tbt_ms) / max(len(target_tbt_ms), 1)),
                "avg_actual_iteration_ms": (
                    sum(actual_iteration_ms)
                    / max(len(actual_iteration_ms), 1)),
                "avg_predicted_iteration_ms": (
                    sum(predicted_iteration_ms)
                    / max(len(predicted_iteration_ms), 1)),
                "avg_scheduler_wall_ms": (
                    sum(scheduler_wall_ms) / max(len(scheduler_wall_ms), 1)),
                "scheduler_wall_ms_p50": percentile(
                    scheduler_wall_ms, 0.50),
                "scheduler_wall_ms_p95": percentile(
                    scheduler_wall_ms, 0.95),
                "scheduler_wall_ms_p99": percentile(
                    scheduler_wall_ms, 0.99),
                "scheduler_wall_total_s": (
                    sum(scheduler_wall_ms) / 1000.0),
                "scheduler_wall_fraction_of_window": (
                    sum(scheduler_wall_ms)
                    / max(args.window_s * 1000.0, 1e-9)),
                "avg_runtime_overhead_ms": (
                    sum(runtime_overhead_ms)
                    / max(len(runtime_overhead_ms), 1)),
                "step_phase_ms": {
                    key: {
                        "count": len(values),
                        "mean": sum(values) / max(len(values), 1),
                        "p50": percentile(values, 0.50),
                        "p95": percentile(values, 0.95),
                        "p99": percentile(values, 0.99),
                        "max": max(values, default=0.0),
                    }
                    for key, values in step_phase_samples.items()
                },
                "rt_stream_delivery_ms": {
                    key: {
                        "count": len(values),
                        "mean": sum(values) / max(len(values), 1),
                        "p50": percentile(values, 0.50),
                        "p95": percentile(values, 0.95),
                        "p99": percentile(values, 0.99),
                        "max": max(values, default=0.0),
                    }
                    for key, values in {
                        "queue_wait": rt_stream_queue_wait_ms,
                        "put_to_worker": rt_stream_put_to_worker_ms,
                        "get_to_worker": rt_stream_get_to_worker_ms,
                    }.items()
                },
                "iteration_count": len(actual_iteration_ms),
                "active_be_remaining": sum(map(len, active_be.values())),
                "active_be_decode_remaining": len(active_be["decode"]),
                "active_be_prefill_remaining": len(active_be["prefill"]),
                "active_be_local_remaining": len(active_be["local"]),
                "be_inflight": args.be_inflight,
                "enable_be_decode": args.enable_be_decode,
                "be_decode_inflight": be_inflight_targets.get("decode", 0),
                "be_prefill_inflight": be_inflight_targets.get(
                    "prefill", 0),
                "be_local_inflight": be_inflight_targets.get("local", 0),
                "simulator": simulation_plan,
                "eviction_steps": eviction_steps,
                "total_preempted": total_preempted,
                "max_swapped_reqs": max_swapped,
                "max_running_queue_size": max_running,
                "max_waiting_reqs": max_pending,
                "worker_backend": "ray_actor",
                "worker_pending_submission_rpcs":
                worker.pending_submissions,
                "worker_returncode": None,
                "crash_error": crash_error,
            }

            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
            with (output_dir / "events.csv").open("w", newline="") as f:
                keys = sorted({key for row in event_rows for key in row})
                if not keys:
                    keys = ["event"]
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(event_rows)
            with (output_dir / "rt_request_tbt.csv").open(
                    "w", newline="") as f:
                keys = list(rt_request_tbt_rows[0]) if rt_request_tbt_rows else [
                    "request_id"]
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rt_request_tbt_rows)
            with (output_dir / "rt_request_stages.csv").open(
                    "w", newline="") as f:
                keys = (
                    list(rt_request_stage_rows[0])
                    if rt_request_stage_rows else ["request_id"])
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rt_request_stage_rows)
            with (output_dir / "be_request_progress.csv").open(
                    "w", newline="") as f:
                keys = (
                    "request_id",
                    "phase",
                    "target_tokens",
                    "generated_tokens_in_window",
                    "completed_request_fraction",
                )
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for request_id in sorted(be_target_tokens_by_request):
                    target_tokens = max(
                        be_target_tokens_by_request[request_id], 1)
                    generated_tokens = be_window_tokens_by_request.get(
                        request_id, 0)
                    writer.writerow({
                        "request_id": request_id,
                        "phase": be_phase_by_request[request_id],
                        "target_tokens": target_tokens,
                        "generated_tokens_in_window": generated_tokens,
                        "completed_request_fraction": min(
                            generated_tokens / target_tokens, 1.0),
                    })
            print(json.dumps(summary, indent=2, sort_keys=True))
        finally:
            worker.poll_submissions()


if __name__ == "__main__":
    main()
