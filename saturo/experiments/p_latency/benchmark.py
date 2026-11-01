#!/usr/bin/env python3
"""Benchmark P-pool (prefill) RT latency under varying RPS — RT-only, no BE.

This script launches a single Saturo P-pool engine and injects RT requests
at configurable request rates (RPS). It measures **per-request**:

* **TTFT** (time-to-first-token) — wall-clock time from injection to first token
* **Queueing delay** — how long the request waited before its prefill step began
* **Execution latency** — the GPU step time of the iteration that processed it

The output is a JSON file with per-request records and two matplotlib charts:

1. **Latency-over-time scatter** — one subplot per RPS level, showing each
   request's TTFT vs its arrival time.  This visualises the queue build-up
   pattern that is characteristic of saturation.
2. **Saturation summary** — TTFT percentiles (p50 / p95 / p99 / max) vs RPS
   with a log-scale y-axis, plus the ideal (no-queue) baseline.

Usage
-----

.. code-block:: bash

    # Quick test on GPU 7 with the 0.5B model
    CUDA_VISIBLE_DEVICES=7 \\
        python experiments/saturo/benchmark_p_pool_rt_latency.py \\
        --model /root/huggingface_config/Qwen2.5-0.5B \\
        --load-format dummy \\
        --num-req 600 --target-rps 1,2,4,8,16,32 \\
        --warmup-s 5

    # Production run on 7B model
    CUDA_VISIBLE_DEVICES=7 \\
        python experiments/saturo/benchmark_p_pool_rt_latency.py \\
        --model /root/huggingface_config/Qwen2.5-7B \\
        --load-format dummy \\
        --num-req 600 --target-rps 1,2,4,8,16,32,64,128 \\
        --warmup-s 10 \\
        --output-dir saturo/results/p_pool_rt_latency

    # Or use the shell wrapper
    ./experiments/saturo/run_p_pool_rt_latency.sh
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TraceRequest:
    request_id: str
    prompt_len: int
    output_len: int
    arrival_offset_s: float = 0.0


@dataclass
class PerRequestRecord:
    """One row in the per-request CSV / JSON output."""
    request_id: str
    rps: float
    arrival_s: float          # wall-clock offset from run start
    first_token_s: float      # wall-clock offset of first token event
    finish_s: float           # wall-clock offset of finished event
    ttft_ms: float            # first_token_s - arrival_s
    step_latency_ms: float    # actual_time_ms of the iteration that processed this request
    queueing_ms: float        # approximate: ttft_ms - step_latency_ms
    prompt_len: int
    in_measurement_window: bool


@dataclass
class RunResult:
    target_rps: float
    requests: list[PerRequestRecord] = field(default_factory=list)
    window_s: float = 0.0
    rt_injected: int = 0
    rt_completed: int = 0
    total_steps: int = 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="P-pool RT latency benchmark — RT only, no BE.")

    m = p.add_argument_group("model and engine")
    m.add_argument("--model", default="",
                   help="Path or HuggingFace ID of the model. "
                        "Not required with --plot-only.")
    m.add_argument("--dtype", default="auto")
    m.add_argument("--load-format", default="auto",
                   help="'dummy' for random weights (no download needed); "
                        "'auto' to load real weights from disk.")
    m.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    m.add_argument("--max-model-len", type=int, default=None)
    m.add_argument("--max-num-seqs", type=int, default=256)
    m.add_argument("--max-num-batched-tokens", type=int, default=None)
    m.add_argument("--disable-inductor", action=argparse.BooleanOptionalAction,
                   default=True)
    m.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction,
                   default=True)

    w = p.add_argument_group("workload and measurement")
    w.add_argument("--rt-workload-csv", default="data/datasets/rt_workload.csv",
                   help="CSV with prompt_len, output_len columns. "
                        "Only prompt_len is used (max_tokens=1 for RT prefill).")
    w.add_argument("--num-req", type=int, required=True)
    w.add_argument("--target-rps", required=True,
                   help="Comma-separated RPS levels to sweep.")
    w.add_argument("--warmup-s", type=float, default=10.0,
                   help="Warmup duration before measurement starts.")
    w.add_argument("--max-wall-s", type=float, default=1200.0,
                   help="Hard timeout for the whole run.")

    hw = p.add_argument_group("hardware")
    hw.add_argument("--gpu-ids", default="",
                    help="Comma-separated GPU IDs (sets CUDA_VISIBLE_DEVICES).")
    hw.add_argument("--tensor-parallel-size", type=int, default=1)
    hw.add_argument("--pipeline-parallel-size", type=int, default=1)

    o = p.add_argument_group("output")
    o.add_argument("--output-dir", default="",
                   help="Directory for output files. "
                        "Default: saturo/results/p_pool_rt_latency/<timestamp>/")
    o.add_argument("--no-plot", action="store_true",
                   help="Skip matplotlib chart generation.")
    o.add_argument("--plot-only", action="store_true",
                   help="Only generate charts from an existing result JSON.")
    o.add_argument("--result-json", default="",
                   help="Path to an existing result JSON (for --plot-only).")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return 0.0
    rank = int(math.ceil(q * len(ordered))) - 1
    return ordered[min(len(ordered) - 1, max(0, rank))]


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    data = [float(v) for v in values]
    if not data:
        return {"count": 0, "mean": 0.0, "median": 0.0,
                "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "count": len(data),
        "mean": round(statistics.fmean(data), 3),
        "median": round(statistics.median(data), 3),
        "p95": round(percentile(data, 0.95), 3),
        "p99": round(percentile(data, 0.99), 3),
        "max": round(max(data), 3),
    }


def load_trace(path: Path, num_req: int,
               target_rps: float) -> list[TraceRequest]:
    from saturo.runtime.workload import load_rt_workload_csv

    if not path.is_file():
        raise FileNotFoundError(f"RT workload CSV not found: {path}")
    loaded = load_rt_workload_csv(
        str(path), num_req=num_req, target_rps=target_rps)
    trace = [TraceRequest(str(r.request_id), int(r.prompt_len),
                          max(1, int(r.output_len)),
                          float(r.arrival_offset_s)) for r in loaded]
    if not trace:
        raise ValueError(f"workload CSV is empty: {path}")
    return trace


def detect_gpu_name() -> str:
    """Return the GPU model name (e.g. 'GB200', 'A100-SXM4-80GB').

    Uses ``nvidia-smi`` which does not create a CUDA context in this process.
    Falls back to a generic tag if the query fails.
    """
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        names = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        if names:
            return names[0]
    except Exception:
        pass
    return "GPU"


def _slug(text: str) -> str:
    """Turn an arbitrary label into a filesystem-friendly slug."""
    text = text.replace("NVIDIA", "").strip()
    out: list[str] = []
    for ch in text:
        if ch.isalnum() or ch in "-.":
            out.append(ch)
        elif ch in " _":
            out.append("-")
    return "".join(out).strip("-") or "x"


def build_run_name(args: argparse.Namespace, rps_points: list[float]) -> str:
    """Compose a human-readable run directory name from hardware, model,
    parallel config and the RPS sweep — e.g.
    ``GB200_Qwen2.5-0.5B_tp1pp1_rps2-4``.
    """
    hw = _slug(detect_gpu_name())
    model_tag = _slug(Path(args.model).name or "model")
    par = f"tp{args.tensor_parallel_size}pp{args.pipeline_parallel_size}"
    rps_tag = "rps" + "-".join(
        (f"{r:g}" for r in rps_points)) if rps_points else "rps"
    return f"{hw}_{model_tag}_{par}_{rps_tag}"


def sanitize(trace: list[TraceRequest], max_model_len: int) -> list[TraceRequest]:
    usable: list[TraceRequest] = []
    for r in trace:
        if r.prompt_len >= max_model_len:
            continue
        out = min(r.output_len, max_model_len - r.prompt_len)
        usable.append(TraceRequest(r.request_id, r.prompt_len, max(1, out)))
    if not usable:
        raise ValueError(f"no request fits max_model_len={max_model_len}")
    return usable


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def worker_args(args: argparse.Namespace) -> dict[str, Any]:
    """Build the worker args dict for a TPUT-role engine.

    We use TPUT (throughput) role rather than PREFILL to avoid enabling
    ``enable_p_pool_saturation``, which would inject BE chunks.  This gives
    us a clean prefill-only engine.
    """
    from saturo.runtime.workload import TPUT

    return {
        "worker_id": 0,
        "worker_role": TPUT,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "expert_parallel_size": 0,
        "model": args.model,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "load_format": args.load_format,
        "dtype": args.dtype,
        "rt_prefill_deadline_s": 3600.0,
        "disable_inductor": args.disable_inductor,
        "rt_tbt_slo_ms": 100.0,
        "p_pool_be_chunk_ms": 0.0,           # disable BE chunk
        "p_pool_preemption_margin": 0.0,      # disable BE preemption
        "runtime_profile": "",
    }


class _MonkeyPatchLimits:
    """Context manager that monkey-patches LLMEngine.from_engine_args to inject
    scheduler limits before engine creation."""

    def __init__(self, args: argparse.Namespace):
        self._args = args
        self._original = None

    def __enter__(self):
        from vllm.engine.llm_engine import LLMEngine
        self._original = LLMEngine.from_engine_args

        args = self._args

        def patched(cls, engine_args, *pos, **kw):
            engine_args.max_num_seqs = args.max_num_seqs
            if args.max_num_batched_tokens is not None:
                engine_args.max_num_batched_tokens = args.max_num_batched_tokens
            if args.max_model_len is not None:
                engine_args.max_model_len = args.max_model_len
            if args.enforce_eager:
                engine_args.enforce_eager = True
            return self._original(engine_args, *pos, **kw)

        LLMEngine.from_engine_args = classmethod(patched)

    def __exit__(self, *exc):
        from vllm.engine.llm_engine import LLMEngine
        LLMEngine.from_engine_args = self._original


def create_worker(args: argparse.Namespace) -> Any:
    from saturo.runtime.worker import OfflineEngineWorker

    with _MonkeyPatchLimits(args):
        worker = OfflineEngineWorker(worker_args(args))
    return worker


def rt_command(req: TraceRequest, serial: int) -> dict[str, Any]:
    """Build an add command for an RT prefill request (max_tokens=1)."""
    from saturo.runtime.workload import RT, TPUT

    aid = f"rt_{serial}:{req.request_id}"
    return {
        "cmd": "add",
        "request_id": aid,
        "attempt_id": aid,
        "kind": RT,
        "phase": TPUT,
        "prompt_len": req.prompt_len,
        "max_tokens": 1,          # only measure prefill → 1 token then finish
        "base_tokens": 0,
    }


def drain_worker(worker: Any, max_wall_s: float) -> None:
    """Abort all active requests and step until idle."""
    start = time.perf_counter()
    aborts = [{"cmd": "abort", "attempt_id": aid}
              for aid in list(worker.active.keys())]
    worker.drive(aborts)
    while worker.engine.has_unfinished_requests():
        if time.perf_counter() - start > max_wall_s:
            break
        worker.drive([])
    worker.drain_events()


# ---------------------------------------------------------------------------
# Per-RPS measurement
# ---------------------------------------------------------------------------


def run_one_rps(
    worker: Any,
    rt_trace: list[TraceRequest],
    target_rps: float,
    args: argparse.Namespace,
) -> RunResult:
    """Inject RT at *target_rps* on a real clock and collect per-request timing."""
    result = RunResult(target_rps=target_rps)
    rt_serial = 0
    rt_idx = 0

    # Per-attempt tracking
    submit_time: dict[str, float] = {}          # attempt_id → wall-clock inject time
    seen_first: set[str] = set()                # attempts that emitted ≥1 token
    # Map attempt_id → iteration's actual_time_ms (from iteration_log)
    step_latency: dict[str, float] = {}

    run_trace = rt_trace
    scaled_arrivals = [req.arrival_offset_s for req in run_trace]

    pending: list[dict[str, Any]] = []
    run_start = time.perf_counter()
    meas_start = run_start + args.warmup_s
    from saturo.tools.trace_scaler import fixed_num_window_s
    send_window_s = fixed_num_window_s(args.num_req, target_rps)
    meas_end = run_start + send_window_s
    deadline = run_start + args.max_wall_s

    while True:
        now = time.perf_counter()
        if now >= deadline:
            raise TimeoutError(f"exceeded --max-wall-s={args.max_wall_s}")

        # Inject RT requests whose scaled arrival time has passed
        elapsed = now - run_start
        while (rt_idx < len(run_trace)
                and scaled_arrivals[rt_idx] <= elapsed):
            req = run_trace[rt_idx]
            rt_idx += 1
            cmd = rt_command(req, rt_serial)
            rt_serial += 1
            submit_time[cmd["attempt_id"]] = time.perf_counter()
            pending.append(cmd)

        events = worker.drive(pending)
        pending = []
        observed = time.perf_counter()

        # Track which iteration_log's actual_time_ms to attribute to requests
        iter_latency_ms = 0.0
        iter_prefill_reqs: list[str] = []

        for ev in events:
            etype = ev.get("event")

            if etype == "iteration_log":
                iter_latency_ms = float(ev.get("actual_time_ms", 0.0))
                iter_prefill_reqs = [str(r) for r in ev.get("prefill_reqs", [])]
                # Attribute this step's latency to each request in the batch
                for rid in iter_prefill_reqs:
                    step_latency[rid] = iter_latency_ms
                if observed >= meas_start:
                    result.total_steps += 1

            elif etype == "token":
                aid = str(ev.get("attempt_id", ""))
                kind = ev.get("kind")
                if kind == "rt" and aid not in seen_first:
                    seen_first.add(aid)
                    t0 = submit_time.get(aid)
                    if t0 is not None:
                        ttft_s = float(ev.get("time_s", observed)) - t0
                        in_window = meas_start <= t0 < meas_end
                        if in_window:
                            result.rt_injected += 1
                        st_lat = step_latency.get(aid, iter_latency_ms)
                        result.requests.append(PerRequestRecord(
                            request_id=aid,
                            rps=target_rps,
                            arrival_s=t0 - run_start,
                            first_token_s=float(ev.get("time_s", observed)) - run_start,
                            finish_s=float(ev.get("time_s", observed)) - run_start,
                            ttft_ms=ttft_s * 1000.0,
                            step_latency_ms=st_lat,
                            queueing_ms=max(0.0, ttft_s * 1000.0 - st_lat),
                            prompt_len=int(ev.get("prompt_len", 0) or 0),
                            in_measurement_window=in_window,
                        ))

            elif etype == "finished":
                aid = str(ev.get("attempt_id", ""))
                kind = ev.get("kind")
                if kind == "rt":
                    # RT that finished without emitting a token
                    t0 = submit_time.get(aid)
                    if aid not in seen_first and t0 is not None:
                        finish_s = float(ev.get("time_s", observed))
                        ttft_s = finish_s - t0
                        in_window = meas_start <= t0 < meas_end
                        if in_window:
                            result.rt_injected += 1
                        st_lat = step_latency.get(aid, 0.0)
                        result.requests.append(PerRequestRecord(
                            request_id=aid,
                            rps=target_rps,
                            arrival_s=t0 - run_start,
                            first_token_s=finish_s - run_start,
                            finish_s=finish_s - run_start,
                            ttft_ms=ttft_s * 1000.0,
                            step_latency_ms=st_lat,
                            queueing_ms=max(0.0, ttft_s * 1000.0 - st_lat),
                            prompt_len=0,
                            in_measurement_window=in_window,
                        ))
                    result.rt_completed += 1

        if (not worker.engine.has_unfinished_requests()
                and not pending
                and rt_idx >= len(run_trace)):
            break

    result.window_s = send_window_s - args.warmup_s
    return result


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------


def plot_gantt(run: dict[str, Any], output_dir: Path) -> None:
    """Draw a per-request Gantt chart for one RPS level.

    Layout mirrors the reference figure:

    * A thin **GPU Util.** strip on top marks the wall-clock intervals during
      which the engine was executing a prefill step (union of execution bars).
    * The main panel shows one horizontal row per request (ordered by arrival):
      a light **Queueing Time** bar followed by a dark **Execution Time** bar.

    Time is measured in seconds relative to the first request's arrival.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rps = float(run["target_rps"])
    reqs = list(run.get("requests", []))
    if not reqs:
        return
    reqs.sort(key=lambda r: r["arrival_s"])

    t0 = min(r["arrival_s"] for r in reqs)

    queue_color = "#AECBEB"   # light blue
    exec_color = "#3B7FBF"    # dark blue

    fig, (ax_util, ax_main) = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True,
        gridspec_kw={"height_ratios": [1, 9]})

    exec_intervals: list[tuple[float, float]] = []
    for i, r in enumerate(reqs):
        arrival = r["arrival_s"] - t0
        first_tok = r["first_token_s"] - t0
        exec_s = max(0.0, float(r.get("step_latency_ms", 0.0)) / 1000.0)
        exec_start = max(arrival, first_tok - exec_s)
        queue_w = max(0.0, exec_start - arrival)
        ax_main.broken_barh([(arrival, queue_w)], (i - 0.4, 0.8),
                            facecolors=queue_color)
        ax_main.broken_barh([(exec_start, exec_s)], (i - 0.4, 0.8),
                            facecolors=exec_color)
        exec_intervals.append((exec_start, exec_s))

    # GPU utilization strip: every execution interval as a busy bar.
    ax_util.broken_barh(exec_intervals, (0, 1), facecolors=exec_color)
    ax_util.set_yticks([])
    ax_util.set_ylabel("GPU Util.", fontsize=10)
    ax_util.tick_params(axis="x", labelbottom=False)

    ax_main.set_xlabel("Time (s)")
    ax_main.set_ylabel("Request")
    ax_main.set_ylim(-1, len(reqs))
    ax_main.set_xlim(left=0)
    ax_main.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax_main.legend(handles=[
        Patch(color=queue_color, label="Queueing Time(s)"),
        Patch(color=exec_color, label="Execution Time(s)")],
        loc="upper left", fontsize=10)

    fig.suptitle(f"P-pool RT Prefill Timeline  (RPS = {rps:g}, RT-only)",
                 fontsize=13)
    fig.tight_layout()
    path = output_dir / f"gantt_rps{rps:g}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Chart] {path}", flush=True)


def plot_results(all_runs: list[dict[str, Any]], output_dir: Path) -> None:
    """Generate two charts from the collected latency data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # ── unpack data ──────────────────────────────────────────────────────
    rps_list: list[float] = []
    p50_vals: list[float] = []
    p95_vals: list[float] = []
    p99_vals: list[float] = []
    max_vals: list[float] = []
    mean_queue: list[float] = []
    p95_queue: list[float] = []
    per_rps_ttft: dict[float, list[float]] = {}
    per_rps_times: dict[float, list[float]] = {}
    ideal_latency_ms: float = 0.0

    for run in all_runs:
        rps = float(run["target_rps"])
        requests = run.get("requests", [])
        # Only use requests within the measurement window
        ttfts = [r["ttft_ms"] for r in requests if r.get("in_measurement_window", True)]
        queues = [r["queueing_ms"] for r in requests if r.get("in_measurement_window", True)]
        arrivals = [r["arrival_s"] for r in requests if r.get("in_measurement_window", True)]

        if not ttfts:
            continue

        rps_list.append(rps)
        p50_vals.append(percentile(ttfts, 0.50))
        p95_vals.append(percentile(ttfts, 0.95))
        p99_vals.append(percentile(ttfts, 0.99))
        max_vals.append(max(ttfts))
        mean_queue.append(statistics.fmean(queues) if queues else 0.0)
        p95_queue.append(percentile(queues, 0.95))
        per_rps_ttft[rps] = ttfts
        per_rps_times[rps] = arrivals

        # Use the lowest-RPS median as the "ideal" (no-queue) baseline
        if ideal_latency_ms == 0.0 or rps < min(rps_list):
            ideal_latency_ms = percentile(ttfts, 0.50)

    if not rps_list:
        print("No data to plot.", flush=True)
        return

    # Per-RPS Gantt timeline (queueing + execution bars, GPU util strip).
    for run in all_runs:
        plot_gantt(run, output_dir)

    # Sort by RPS
    order = sorted(range(len(rps_list)), key=lambda i: rps_list[i])
    rps_list = [rps_list[i] for i in order]
    p50_vals = [p50_vals[i] for i in order]
    p95_vals = [p95_vals[i] for i in order]
    p99_vals = [p99_vals[i] for i in order]
    max_vals = [max_vals[i] for i in order]
    mean_queue = [mean_queue[i] for i in order]
    p95_queue = [p95_queue[i] for i in order]

    # ── Chart 1: Latency-over-time scatter per RPS ───────────────────────
    n_rps = len(rps_list)
    cols = min(4, n_rps)
    rows = math.ceil(n_rps / cols)
    fig1, axes1 = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows),
                               squeeze=False)
    fig1.suptitle("P-pool RT Latency over Time by RPS", fontsize=14, y=1.01)

    for idx, rps in enumerate([rps_list[i] for i in order]):
        ax = axes1[idx // cols][idx % cols]
        ttfts = per_rps_ttft.get(rps, [])
        arrivals = per_rps_times.get(rps, [])
        if ttfts and arrivals:
            colors = ['#2196F3' if t < ideal_latency_ms * 2 else '#FF5722'
                      for t in ttfts]
            ax.scatter(arrivals, ttfts, c=colors, s=8, alpha=0.6,
                       edgecolors='none')
            ax.axhline(y=ideal_latency_ms, color='green', linestyle='--',
                       alpha=0.7, label=f'ideal ≈ {ideal_latency_ms:.0f} ms')
            ax.axhline(y=percentile(ttfts, 0.95), color='orange', linestyle=':',
                       alpha=0.7, label=f'p95 = {percentile(ttfts, 0.95):.0f} ms')
        ax.set_title(f"RPS = {rps:.1f}  (n={len(ttfts)})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("TTFT (ms)")
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_rps, rows * cols):
        axes1[idx // cols][idx % cols].set_visible(False)

    fig1.tight_layout()
    chart1_path = output_dir / "latency_over_time.png"
    fig1.savefig(chart1_path, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"[Chart] {chart1_path}", flush=True)

    # ── Chart 2: Saturation summary (latency percentiles vs RPS) ─────────
    fig2, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8),
                                           sharex=True,
                                           gridspec_kw={'height_ratios': [2, 1]})

    # Top: TTFT percentiles
    ax_top.fill_between(rps_list, p50_vals, p95_vals, alpha=0.3,
                        color='#2196F3', label='p50–p95 band')
    ax_top.fill_between(rps_list, p95_vals, p99_vals, alpha=0.2,
                        color='#FF9800', label='p95–p99 band')
    ax_top.plot(rps_list, p50_vals, 'o-', color='#1976D2', linewidth=2,
                markersize=6, label='p50')
    ax_top.plot(rps_list, p95_vals, 's--', color='#F57C00', linewidth=1.5,
                markersize=5, label='p95')
    ax_top.plot(rps_list, p99_vals, '^:', color='#D32F2F', linewidth=1.5,
                markersize=5, label='p99')
    ax_top.plot(rps_list, max_vals, 'x-', color='#B71C1C', linewidth=1,
                markersize=4, alpha=0.7, label='max')
    ax_top.axhline(y=ideal_latency_ms, color='green', linestyle='--',
                   alpha=0.6, label=f'ideal ({ideal_latency_ms:.0f} ms)')
    ax_top.set_yscale("log")
    ax_top.set_ylabel("TTFT (ms, log scale)")
    ax_top.set_title("P-pool RT Prefill Latency vs RPS  (RT-only, no BE)")
    ax_top.legend(loc='upper left', fontsize=8)
    ax_top.grid(True, alpha=0.3, which='both')

    # Bottom: Queueing delay
    ax_bot.fill_between(rps_list, 0, mean_queue, alpha=0.3,
                        color='#4CAF50', label='mean queueing')
    ax_bot.fill_between(rps_list, mean_queue, p95_queue, alpha=0.2,
                        color='#FF9800', label='p95 queueing')
    ax_bot.plot(rps_list, mean_queue, 'o-', color='#388E3C', linewidth=2,
                markersize=6, label='mean queue')
    ax_bot.plot(rps_list, p95_queue, 's--', color='#F57C00', linewidth=1.5,
                markersize=5, label='p95 queue')
    ax_bot.set_xlabel("Offered RPS")
    ax_bot.set_ylabel("Queueing delay (ms)")
    ax_bot.legend(loc='upper left', fontsize=8)
    ax_bot.grid(True, alpha=0.3)

    fig2.tight_layout()
    chart2_path = output_dir / "saturation_summary.png"
    fig2.savefig(chart2_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[Chart] {chart2_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # --plot-only mode: skip benchmark, just generate charts
    if args.plot_only:
        if not args.no_plot:
            pass  # plotting is the point of --plot-only
        result_path = Path(args.result_json) if args.result_json else None
        if not result_path or not result_path.is_file():
            # auto-find latest
            candidates = sorted(
                Path("saturo/results/p_pool_rt_latency").glob("**/result.json"),
                key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                result_path = candidates[0]
                print(f"Auto-selected result: {result_path}", flush=True)
            else:
                print("ERROR: no result JSON found and --result-json not set.",
                      file=sys.stderr)
                sys.exit(1)
        data = json.loads(result_path.read_text(encoding="utf-8"))
        out_dir = result_path.parent
        plot_results(data["runs"], out_dir)
        return

    # ---- validate required args for benchmark mode ----------------------
    if not args.model:
        print("ERROR: --model is required (unless using --plot-only).",
              file=sys.stderr)
        sys.exit(1)

    # ---- setup -----------------------------------------------------------
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("VLLM_TARGET_DEVICE", "cuda")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    if args.num_req <= 0:
        raise ValueError("--num-req must be positive")
    rps_points = [
        float(x) for x in str(args.target_rps).split(",") if x.strip()]
    if not rps_points:
        raise ValueError("--target-rps had no values")
    if min(rps_points) <= 0:
        raise ValueError("--target-rps must contain positive values")
    from saturo.tools.trace_scaler import fixed_num_window_s
    if any(args.warmup_s >= fixed_num_window_s(args.num_req, rps)
           for rps in rps_points):
        raise ValueError(
            "--warmup-s must be smaller than num_req / target_rps")

    # ---- output directory ------------------------------------------------
    # Named by hardware / model / parallel config / RPS so results are easy
    # to find without decoding a timestamp.  A short time suffix is appended
    # only when a same-named directory already exists (to avoid clobbering).
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        run_name = build_run_name(args, rps_points)
        base = PROJECT_ROOT / "saturo" / "results" / "p_pool_rt_latency"
        output_dir = base / run_name
        if output_dir.exists():
            stamp = datetime.now().astimezone().strftime("%H%M%S")
            output_dir = base / f"{run_name}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    rt_path = Path(args.rt_workload_csv).expanduser().resolve()
    print(f"Using the first {args.num_req} RT rows.", flush=True)
    print(f"RPS sweep: {rps_points}", flush=True)
    print(f"Model: {args.model}  |  TP={args.tensor_parallel_size}  "
          f"|  load_format={args.load_format}", flush=True)
    print(f"Warmup: {args.warmup_s}s; send windows derive from "
          f"num_req / target_rps.", flush=True)
    print(f"Output: {output_dir}", flush=True)

    # ---- create worker ---------------------------------------------------
    print("Initializing engine ...", flush=True)
    worker = None
    all_runs: list[dict[str, Any]] = []

    try:
        worker = create_worker(args)
        max_model_len = int(worker.engine.model_config.max_model_len)
        cache = worker.engine.cache_config
        print(f"Engine ready: num_gpu_blocks={cache.num_gpu_blocks}, "
              f"block_size={cache.block_size}, max_model_len={max_model_len}",
              flush=True)

        for rps in rps_points:
            rt_trace = sanitize(
                load_trace(rt_path, args.num_req, rps), max_model_len)
            if len(rt_trace) != args.num_req:
                raise ValueError(
                    f"only {len(rt_trace)} of num_req={args.num_req} RT "
                    f"requests fit max_model_len={max_model_len}")
            print(f"\n{'='*60}")
            print(f"  RPS = {rps:.1f}  (interval = {1000.0/rps:.1f} ms)")
            print(f"{'='*60}", flush=True)

            result = run_one_rps(worker, rt_trace, rps, args)

            # Collect window-only requests for summary statistics
            win_reqs = [r for r in result.requests if r.in_measurement_window]
            win_ttfts = [r.ttft_ms for r in win_reqs]
            win_queues = [r.queueing_ms for r in win_reqs]

            run_dict = {
                "target_rps": result.target_rps,
                "window_s": round(result.window_s, 3),
                "rt_injected": result.rt_injected,
                "rt_completed": result.rt_completed,
                "total_steps": result.total_steps,
                "ttft_ms": summarize(win_ttfts),
                "queueing_ms": summarize(win_queues),
                "requests": [
                    {
                        "request_id": r.request_id,
                        "rps": r.rps,
                        "arrival_s": round(r.arrival_s, 6),
                        "first_token_s": round(r.first_token_s, 6),
                        "ttft_ms": round(r.ttft_ms, 3),
                        "step_latency_ms": round(r.step_latency_ms, 3),
                        "queueing_ms": round(r.queueing_ms, 3),
                        "prompt_len": r.prompt_len,
                        "in_measurement_window": r.in_measurement_window,
                    }
                    for r in result.requests
                ],
            }
            all_runs.append(run_dict)

            summary = run_dict["ttft_ms"]
            q_summary = run_dict["queueing_ms"]
            print(f"  TTFT  p50={summary['median']:.1f}  "
                  f"p95={summary['p95']:.1f}  p99={summary['p99']:.1f}  "
                  f"max={summary['max']:.1f} ms  (n={summary['count']})",
                  flush=True)
            print(f"  Queue p50={q_summary['median']:.1f}  "
                  f"p95={q_summary['p95']:.1f}  "
                  f"mean={q_summary['mean']:.1f} ms",
                  flush=True)

            drain_worker(worker, args.max_wall_s)

        # ---- save result JSON --------------------------------------------
        output = {
            "config": {
                "model": args.model,
                "tensor_parallel_size": args.tensor_parallel_size,
                "load_format": args.load_format,
                "dtype": args.dtype,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "max_model_len": max_model_len,
                "num_gpu_blocks": int(cache.num_gpu_blocks),
                "block_size": int(cache.block_size),
                "warmup_s": args.warmup_s,
                "num_req": args.num_req,
                "rt_workload_csv": str(rt_path),
                "rps_points": rps_points,
                "max_num_seqs": args.max_num_seqs,
                "max_num_batched_tokens": args.max_num_batched_tokens,
            },
            "runs": all_runs,
        }
        result_path = output_dir / "result.json"
        result_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"\n[Result] {result_path}", flush=True)

        # ---- plot --------------------------------------------------------
        if not args.no_plot:
            plot_results(all_runs, output_dir)

    finally:
        if worker is not None:
            try:
                worker.shutdown()
            except Exception:
                pass
            del worker
        gc.collect()

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
