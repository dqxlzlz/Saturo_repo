#!/usr/bin/env python3
"""Benchmark a single Saturo T-pool (throughput) instance under BE saturation.

T-pool is the pure-throughput pool — it only processes BE (best-effort) requests
with no RT traffic. By default, its validation population mirrors P/D: half is
real prefill-only work and half is decode-ready work restored through
DummyConnector. ``--request-mode=vllm`` instead submits ordinary end-to-end
requests, lets vLLM move each request from Prefill to Decode, and counts only
completed end-to-end requests as throughput.

Before starting the engine, the benchmark uses ``tput_simulator`` to choose the
best chunk-size (``max_num_batched_tokens``) and predict its BE saturation
throughput. It then measures achieved throughput on the real engine using that
chunk-size. ``--chunk-size-source=sweep`` retains the old real-engine sweep.

The real engine is the V0 ``AsyncLLMEngine`` (via ``RayEngineWorker``), the
same engine the online scheduler path uses. This also enables pipeline
parallelism (e.g. TP4 x PP2), which the synchronous V0 ``LLMEngine`` rejects.

Example::

    CUDA_VISIBLE_DEVICES=0 python \
        saturo/experiments/pool_validation/benchmark_t_pool_saturation.py \
        --model /root/huggingface_config/Qwen2.5-7B \
        --runtime-profile /path/to/time_predictor.json \
        --be-workload-csv data/datasets/be_workload.csv \
        --chunk-sizes 512,1024,2048,4096 --be-inflight 64,128 \
        --warmup-s 10 --duration-s 30
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REQUEST_MODE_SPLIT_PHASE = "split-phase"
REQUEST_MODE_VLLM = "vllm"


@dataclass(frozen=True)
class TraceRequest:
    request_id: str
    prompt_len: int
    output_len: int


@dataclass
class RunResult:
    chunk_size: int
    be_inflight: int
    request_mode: str = REQUEST_MODE_SPLIT_PHASE
    avg_be_input_len: float = 1.0
    avg_be_output_len: float = 1.0
    # BE metrics
    be_completed: int = 0
    be_output_tokens: int = 0
    be_prefill_tokens: int = 0
    be_decode_tokens: int = 0
    be_dummy_handoff_tokens: int = 0
    be_submitted: int = 0
    be_e2e_completed: int = 0
    be_prefill_completed: int = 0
    be_decode_completed: int = 0
    # diagnostics
    decode_batch_sizes: list[float] = field(default_factory=list)
    prefill_batch_sizes: list[float] = field(default_factory=list)
    iteration_count: int = 0
    iteration_actual_ms: list[float] = field(default_factory=list)
    iteration_predicted_ms: list[float] = field(default_factory=list)
    scheduler_wall_ms: list[float] = field(default_factory=list)
    runtime_overhead_ms: list[float] = field(default_factory=list)
    used_gpu_blocks: list[float] = field(default_factory=list)
    running_queue_sizes: list[float] = field(default_factory=list)
    waiting_queue_sizes: list[float] = field(default_factory=list)
    preempted_total: int = 0
    blocks_to_swap_in_total: int = 0
    blocks_to_swap_out_total: int = 0
    window_s: float = 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Measure achieved BE throughput on one Saturo T-pool instance."))

    m = p.add_argument_group("model and engine")
    m.add_argument("--model", required=True)
    m.add_argument("--dtype", default="auto")
    m.add_argument("--load-format", default="auto",
                   help="dummy = random weights (profiling only, no weights "
                        "needed); auto = load real weights.")
    m.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    m.add_argument("--max-model-len", type=int, default=None)
    m.add_argument("--max-num-seqs", type=int, default=256,
                   help="Keep high so KV memory, not the seq cap, bounds the "
                        "BE decode batch.")
    m.add_argument("--max-num-batched-tokens", type=int, default=None)
    m.add_argument("--disable-inductor", action=argparse.BooleanOptionalAction,
                   default=True)
    m.add_argument(
        "--runtime-profile", default="",
        help="TimePredictor profile used by both the engine diagnostics and "
             "the matching simulator.")

    w = p.add_argument_group("workload and measurement")
    w.add_argument("--be-workload-csv", required=True,
                   help="CSV with prompt_len, output_len columns.")
    w.add_argument("--chunk-sizes", default="auto",
                   help="'auto' searches 128..8192 in steps of 128 in "
                        "simulator mode; otherwise use comma-separated "
                        "max_num_batched_tokens values. Each value is the "
                        "total per-iteration token budget (prefill + decode).")
    w.add_argument(
        "--chunk-size-source",
        choices=("sim", "sweep"),
        default="sim",
        help="'sim' runs the T-pool simulator before engine startup and uses "
             "only its optimal chunk size; 'sweep' preserves the old real "
             "engine sweep over every --chunk-sizes value.")
    w.add_argument("--sim-duration-s", type=float, default=600.0,
                   help="T-pool simulator horizon before engine startup.")
    w.add_argument("--sim-warmup-s", type=float, default=0.0,
                   help="T-pool simulator warmup horizon.")
    w.add_argument("--sim-plan-json", default="",
                   help="Simulator plan output. Default: "
                        "<output-json>.sim_plan.json.")
    w.add_argument("--be-inflight", default="64,128",
                   help="Comma-separated BE concurrency levels to sweep.")
    w.add_argument(
        "--request-mode",
        choices=(REQUEST_MODE_SPLIT_PHASE, REQUEST_MODE_VLLM),
        default=REQUEST_MODE_VLLM,
        help="'split-phase' (explicit legacy mode) preserves the existing independent "
             "Prefill-only/Decode-ready injection with DummyConnector; "
             "'vllm' submits ordinary end-to-end requests and lets vLLM "
             "schedule Prefill -> Decode with chunked prefill.")
    w.add_argument(
        "--enable-be-decode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Split the constant BE population 50/50 between independent "
             "prefill-only and DummyConnector decode-ready requests.")
    w.add_argument(
        "--be-decode-kv-utilization",
        type=float,
        default=0.95,
        help="Fraction of total GPU KV blocks considered usable before "
             "reserving the longest BE-prefill prompt (default: 0.95).")
    w.add_argument("--warmup-s", type=float, default=10.0)
    w.add_argument("--duration-s", type=float, default=300.0)
    w.add_argument("--max-wall-s", type=float, default=1200.0)
    w.add_argument("--max-source-be", type=int, default=0,
                   help="Cap on BE rows read from CSV (0 = unlimited).")
    # BE length filters (mirror trace_loader / dataset knobs; default off).
    w.add_argument("--be-input-drop-frac", type=float, default=0.0)
    w.add_argument("--be-input-max-percentile", type=float, default=None,
                   help="Drop BE requests whose prompt length is above this "
                        "dataset percentile, e.g. 0.80 keeps <= P80.")
    w.add_argument("--be-output-keep-frac", type=float, default=None)
    w.add_argument("--be-output-max-len", type=int, default=None)
    # T-pool specific: optional RT decode hand-off simulation.
    # When > 0, a synthetic RT decode population consumes KV slots,
    # modelling the scenario where T-pool serves overflow RT decode.
    w.add_argument("--synthetic-rt-decode", type=int, default=0,
                   help="Number of synthetic RT decode sequences to keep "
                        "active (consumes KV slots, reduces BE capacity).")

    hw = p.add_argument_group("hardware")
    hw.add_argument("--gpu-ids", default="")
    hw.add_argument("--tensor-parallel-size", type=int, default=1)
    hw.add_argument("--pipeline-parallel-size", type=int, default=1)
    hw.add_argument("--expert-parallel-size", type=int, default=0)

    p.add_argument("--output-json", default="",
                   help="Where to write the result JSON. Default: "
                        "experiments/saturo/results/t_pool_sat_<ts>.json")
    return p.parse_args()


def default_output_path() -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return (PROJECT_ROOT / "experiments" / "saturo" / "results" /
            f"t_pool_sat_{stamp}.json")


def run_sweep_subprocesses(
    out_path: Path,
    chunk_sizes: list[int],
) -> None:
    """Run every real-engine chunk in a fresh Python process.

    V0's TP multiprocessing executor leaves the driver's distributed state
    initialized after shutdown, so constructing the next TP engine in the same
    process can hang in ``init_device``. A process boundary is also a cleaner
    benchmark boundary: CUDA, NCCL, KV cache, and worker state are all fresh.
    """
    child_payloads: list[dict[str, Any]] = []
    child_paths: list[Path] = []
    for chunk_size in chunk_sizes:
        child_path = out_path.with_name(
            f"{out_path.stem}.chunk{chunk_size}{out_path.suffix}")
        child_paths.append(child_path)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *sys.argv[1:],
            "--chunk-sizes",
            str(chunk_size),
            "--output-json",
            str(child_path),
        ]
        child_env = os.environ.copy()
        child_env["SATURO_T_POOL_SWEEP_CHILD"] = "1"
        print("\n=== Starting isolated chunk process: "
              f"chunk_size={chunk_size} ===", flush=True)
        subprocess.run(command, env=child_env, check=True)
        child_payloads.append(json.loads(
            child_path.read_text(encoding="utf-8")))

    combined = dict(child_payloads[0])
    combined["config"] = dict(combined["config"])
    combined["config"]["chunk_size_candidates"] = chunk_sizes
    combined["config"]["selected_chunk_size"] = None
    combined["config"]["isolated_chunk_processes"] = True
    combined["config"]["num_gpu_blocks"] = None
    combined["config"]["num_gpu_blocks_by_chunk"] = {
        str(payload["runs"][0]["chunk_size"]):
        int(payload["config"]["num_gpu_blocks"])
        for payload in child_payloads
    }
    combined["chunk_outputs"] = [str(path) for path in child_paths]
    combined["runs"] = [
        run
        for payload in child_payloads
        for run in payload.get("runs", [])
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"\n[Output] wrote combined sweep {out_path}", flush=True)


def percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return 0.0
    rank = int(math.ceil(q * len(ordered))) - 1
    return ordered[min(len(ordered) - 1, max(0, rank))]


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    data = [float(v) for v in values]
    if not data:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0,
                "p99": 0.0, "max": 0.0}
    return {"count": len(data), "mean": statistics.fmean(data),
            "median": statistics.median(data), "p95": percentile(data, 0.95),
            "p99": percentile(data, 0.99), "max": max(data)}


def load_trace(path: Path, max_requests: int) -> list[TraceRequest]:
    from saturo.runtime.workload import BE, load_workload_csv

    if not path.is_file():
        raise FileNotFoundError(f"workload CSV not found: {path}")
    loaded = load_workload_csv(str(path), BE)
    if max_requests > 0:
        loaded = loaded[:max_requests]
    trace = [TraceRequest(str(r.request_id), int(r.prompt_len),
                          max(1, int(r.output_len))) for r in loaded]
    if not trace:
        raise ValueError(f"workload CSV is empty: {path}")
    return trace


def filter_be(trace: list[TraceRequest], args: argparse.Namespace
              ) -> list[TraceRequest]:
    """Apply the same BE length filters as trace_loader.load_be_trace."""
    items = list(trace)
    args.be_input_max_tokens_applied = None
    if args.be_input_max_percentile is not None:
        q = float(args.be_input_max_percentile)
        if not 0.0 < q <= 1.0:
            raise ValueError("--be-input-max-percentile must be in (0, 1]")
        cutoff = int(percentile((r.prompt_len for r in items), q))
        args.be_input_max_tokens_applied = cutoff
        items = [r for r in items if r.prompt_len <= cutoff]
    if args.be_input_drop_frac:
        if not 0.0 < args.be_input_drop_frac < 1.0:
            raise ValueError("--be-input-drop-frac must be in (0, 1)")
        items.sort(key=lambda r: r.prompt_len)
        items = items[int(args.be_input_drop_frac * len(items)):]
    if args.be_output_max_len is not None:
        items = [r for r in items if r.output_len <= args.be_output_max_len]
    elif args.be_output_keep_frac is not None:
        if not 0.0 < args.be_output_keep_frac <= 1.0:
            raise ValueError("--be-output-keep-frac must be in (0, 1]")
        items.sort(key=lambda r: r.output_len)
        items = items[:max(1, int(args.be_output_keep_frac * len(items)))]
    if not items:
        raise ValueError("BE length filter removed all rows")
    return items


def sanitize(trace: list[TraceRequest], max_model_len: int
             ) -> list[TraceRequest]:
    usable: list[TraceRequest] = []
    for r in trace:
        if r.prompt_len >= max_model_len:
            continue
        out = min(r.output_len, max_model_len - r.prompt_len)
        usable.append(TraceRequest(r.request_id, r.prompt_len, max(1, out)))
    if not usable:
        raise ValueError(
            f"no request fits max_model_len={max_model_len}")
    return usable


def infer_model_max_len(args: argparse.Namespace) -> Optional[int]:
    """Best-effort model-length lookup used before GPU engine creation."""
    explicit = getattr(args, "max_model_len", None)
    if explicit is not None:
        return int(explicit)
    model = str(getattr(args, "model", "") or "")
    config_path = Path(model).expanduser() / "config.json"
    if not config_path.is_file():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    keys = (
        "max_position_embeddings",
        "n_positions",
        "max_seq_len",
        "seq_length",
        "model_max_length",
        "max_sequence_length",
        "max_seq_length",
        "seq_len",
    )
    values = [int(config[key]) for key in keys
              if isinstance(config.get(key), (int, float))
              and int(config[key]) > 0]
    return min(values) if values else None


def be_decode_kv_plan(
    total_gpu_blocks: int,
    block_size: int,
    be_trace: list[TraceRequest],
    utilization: float,
) -> dict[str, int | float]:
    """Reserve one maximum BE prefill below a fixed KV utilization ceiling."""
    if not 0.0 < utilization <= 1.0:
        raise ValueError("--be-decode-kv-utilization must be in (0, 1]")
    total_gpu_blocks = int(total_gpu_blocks)
    block_size = max(int(block_size), 1)
    usable_gpu_blocks = int(math.floor(total_gpu_blocks * utilization))
    max_be_prompt = max((req.prompt_len for req in be_trace), default=1)
    reserved_be_prefill_blocks = int(math.ceil(max_be_prompt / block_size))
    be_decode_block_limit = usable_gpu_blocks - reserved_be_prefill_blocks
    if be_decode_block_limit <= 0:
        raise ValueError(
            "no KV blocks remain for BE decode after reservation: "
            f"floor({total_gpu_blocks} * {utilization}) - "
            f"ceil({max_be_prompt} / {block_size}) = "
            f"{be_decode_block_limit}")
    return {
        "total_gpu_blocks": total_gpu_blocks,
        "kv_utilization": float(utilization),
        "usable_gpu_blocks": usable_gpu_blocks,
        "max_be_prompt": max_be_prompt,
        "reserved_be_prefill_blocks": reserved_be_prefill_blocks,
        "be_decode_block_limit": be_decode_block_limit,
    }


def build_sim_plan(
    args: argparse.Namespace,
    chunk_sizes: list[int],
    be_trace: list[TraceRequest],
    plan_path: Path,
) -> dict[str, Any]:
    """Choose the T-pool chunk size before creating a real engine."""
    from saturo.simulators.tput_simulator.config import (
        BEConfig,
        ModelConfig,
        SimulatorConfig,
    )
    from saturo.simulators.tput_simulator.simulator import (
        ThroughputPoolSimulator,
    )

    if not args.runtime_profile:
        raise ValueError(
            "--chunk-size-source=sim requires --runtime-profile; pass a "
            "time_predictor JSON or use --chunk-size-source=sweep")
    profile_path = Path(args.runtime_profile).expanduser().resolve()
    if not profile_path.is_file():
        raise FileNotFoundError(f"runtime profile does not exist: {profile_path}")
    if args.sim_duration_s <= 0:
        raise ValueError("--sim-duration-s must be positive")
    if not 0 <= args.sim_warmup_s < args.sim_duration_s:
        raise ValueError(
            "--sim-warmup-s must be nonnegative and smaller than "
            "--sim-duration-s")

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    kv = profile.get("kv_cache_block") or {}
    total_gpu_blocks = int(kv.get("total_gpu_blocks", 0) or 0)
    block_size = int(kv.get("block_size", 16) or 16)
    if total_gpu_blocks <= 0:
        raise ValueError(
            "runtime profile has no positive "
            "kv_cache_block.total_gpu_blocks")

    avg_input_len = max(1, int(round(statistics.fmean(
        req.prompt_len for req in be_trace))))
    avg_output_len = max(1, int(round(statistics.fmean(
        req.output_len for req in be_trace))))
    decode_kv = be_decode_kv_plan(
        total_gpu_blocks,
        block_size,
        be_trace,
        float(getattr(args, "be_decode_kv_utilization", 0.95)),
    )
    be_decode_blocks = (int(decode_kv["be_decode_block_limit"])
                        if getattr(args, "enable_be_decode", True)
                        else total_gpu_blocks)
    inflight_levels = [
        int(value)
        for value in str(getattr(args, "be_inflight", "128")).split(",")
        if value.strip()
    ]
    if not inflight_levels or any(value <= 0 for value in inflight_levels):
        raise ValueError("--be-inflight must contain positive integers")
    request_mode = str(getattr(
        args, "request_mode", REQUEST_MODE_SPLIT_PHASE))
    split_phase = request_mode == REQUEST_MODE_SPLIT_PHASE
    # split-phase uses independent Prefill-only and Decode-ready populations;
    # vllm replays complete Prefill-to-Decode lifecycles.
    candidate_payloads: list[dict[str, Any]] = []
    sim_pp_size = max(int(getattr(args, "pipeline_parallel_size", 1) or 1), 1)
    sim_inflight_levels = inflight_levels if split_phase else inflight_levels[:1]
    simulator_mode = "split_trace" if split_phase else "trace"
    for sim_be_inflight in sim_inflight_levels:
        sim_config = SimulatorConfig(
            duration_s=float(args.sim_duration_s),
            warmup_s=float(args.sim_warmup_s),
            latency_model_path=str(profile_path),
            model=ModelConfig(block_size=block_size),
            pipeline_parallel_size=sim_pp_size,
            be=BEConfig(
                avg_input_len=avg_input_len,
                avg_output_len=avg_output_len,
                total_gpu_blocks=total_gpu_blocks,
                be_decode_blocks=(be_decode_blocks
                                  if split_phase else total_gpu_blocks),
                be_inflight=sim_be_inflight,
                max_num_seqs=int(getattr(args, "max_num_seqs", 256)),
                enable_be_decode=bool(
                    getattr(args, "enable_be_decode", True)),
                chunk_size_candidates=chunk_sizes,
                prefill_chunk_candidates=chunk_sizes,
            ),
        )
        simulator = ThroughputPoolSimulator(sim_config)
        result = (simulator.run_split_trace(be_trace)
                  if split_phase else simulator.run_trace(be_trace))
        candidate_payloads.extend(
            candidate.to_dict() for candidate in result.candidates)

    best = max(
        candidate_payloads,
        key=lambda candidate: (
            float(candidate["be_saturation_req_per_s"]),
            float(candidate["raw_tokens_per_s"]),
            -int(candidate["chunk_size"]),
        ),
    )
    selected_chunk_size = int(best["chunk_size"])
    payload = {
        "pool": "T",
        "mode": simulator_mode,
        "config": {
            "request_mode": request_mode,
            "runtime_profile": str(profile_path),
            "pipeline_parallel_size": sim_pp_size,
            "be_workload_csv": str(
                Path(args.be_workload_csv).expanduser().resolve()),
            "duration_s": float(args.sim_duration_s),
            "warmup_s": float(args.sim_warmup_s),
            "avg_input_len": avg_input_len,
            "avg_output_len": avg_output_len,
            "total_gpu_blocks": total_gpu_blocks,
            "block_size": block_size,
            "enable_be_decode": bool(
                getattr(args, "enable_be_decode", True)),
            "be_inflight_candidates": inflight_levels,
            "max_num_seqs": int(getattr(args, "max_num_seqs", 256)),
            "be_decode_kv": decode_kv,
            "chunk_size_candidates": chunk_sizes,
        },
        "best": best,
        "candidates": candidate_payloads,
    }
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[sim] T-pool best: "
          f"chunk_size={selected_chunk_size}, "
          "predicted BE saturation="
          f"{float(best['be_saturation_req_per_s']):.6f} req/s; "
          f"wrote {plan_path}", flush=True)
    return payload


def worker_args(args: argparse.Namespace) -> dict[str, Any]:
    from saturo.runtime.workload import TPUT

    return {
        "worker_id": 0,
        "worker_role": TPUT,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "expert_parallel_size": args.expert_parallel_size,
        "model": args.model,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "load_format": args.load_format,
        "dtype": args.dtype,
        "rt_prefill_deadline_s": 3600.0,   # unused by TPUT role
        "disable_inductor": args.disable_inductor,
        "rt_tbt_slo_ms": 100.0,            # unused by TPUT role
        "p_pool_be_chunk_ms": 100.0,       # unused by TPUT role
        "p_pool_preemption_margin": 0.0,   # unused by TPUT role
        "runtime_profile": args.runtime_profile,
        "enable_t_pool_dummy_decode": (
            args.request_mode == REQUEST_MODE_SPLIT_PHASE
            and args.enable_be_decode),
    }


def create_worker(args: argparse.Namespace) -> Any:
    """Create a Saturo T-pool async worker, forwarding scheduler limits.

    Uses ``RayEngineWorker`` (in-process, no Ray actor) so the benchmark runs
    the same V0 ``AsyncLLMEngine`` as the online scheduler path. This is also
    required for pipeline parallelism (e.g. TP4 x PP2): the synchronous V0
    ``LLMEngine`` only supports PP through ``AsyncLLMEngine``.
    """
    from saturo.runtime.worker import RayEngineWorker
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    original_descriptor = AsyncLLMEngine.__dict__["from_engine_args"]
    original_factory = AsyncLLMEngine.from_engine_args

    def from_engine_args_with_limits(cls, engine_args, *pos, **kwargs):
        engine_args.max_num_seqs = args.max_num_seqs
        if args.max_num_batched_tokens is not None:
            engine_args.max_num_batched_tokens = args.max_num_batched_tokens
        if args.max_model_len is not None:
            engine_args.max_model_len = args.max_model_len
        # The TPUT role leaves chunked prefill disabled by default, but the
        # chunk-size sweep interprets max_num_batched_tokens as a per-iteration
        # (prefill + decode) token budget. That only works with chunked prefill
        # on; otherwise vLLM rejects max_num_batched_tokens < max_model_len for
        # the long BE prompts. Enable it so the sweep mirrors the tput_simulator.
        engine_args.enable_chunked_prefill = True
        return original_factory(engine_args, *pos, **kwargs)

    AsyncLLMEngine.from_engine_args = classmethod(from_engine_args_with_limits)
    try:
        worker = RayEngineWorker(worker_args(args))
    finally:
        AsyncLLMEngine.from_engine_args = original_descriptor
    return worker


def be_command(req: TraceRequest, serial: int, phase: str) -> dict[str, Any]:
    from saturo.runtime.workload import BE, DECODE

    phase_name = "decode" if phase == DECODE else "prefill"
    aid = f"be_{phase_name}_{serial}:{req.request_id}"
    return {"cmd": "add", "request_id": aid, "attempt_id": aid, "kind": BE,
            "phase": phase, "prompt_len": req.prompt_len,
            # Prefill-only requests stop on the first sampled token. Decode
            # requests restore prompt KV through DummyConnector and then run
            # the dataset output length.
            "max_tokens": req.output_len if phase == DECODE else 1,
            "base_tokens": 0}


def be_lifecycle_command(req: TraceRequest, serial: int) -> dict[str, Any]:
    """Create one ordinary vLLM request with a full Prefill->Decode lifecycle."""
    from saturo.runtime.workload import BE, PREFILL

    aid = f"be_vllm_{serial}:{req.request_id}"
    return {
        "cmd": "add",
        "request_id": aid,
        "attempt_id": aid,
        "kind": BE,
        # PREFILL means that no external KV is supplied. vLLM itself changes
        # the sequence to Decode after all prompt chunks have run.
        "phase": PREFILL,
        "prompt_len": req.prompt_len,
        "max_tokens": req.output_len,
        "base_tokens": 0,
    }


def synthetic_rt_decode_command(serial: int, prompt_len: int,
                                output_len: int) -> dict[str, Any]:
    """Create a synthetic RT decode sequence that consumes KV slots."""
    from saturo.runtime.workload import DECODE, RT

    aid = f"synth_rt_{serial}"
    return {"cmd": "add", "request_id": aid, "attempt_id": aid, "kind": RT,
            "phase": DECODE, "prompt_len": prompt_len,
            "max_tokens": output_len, "base_tokens": 0}


async def drain_worker(worker: Any, max_wall_s: float) -> None:
    """Abort all active attempts and wait until the engine is empty."""
    start = time.perf_counter()
    for aid in list(worker.active.keys()):
        await worker.abort(aid)
    while worker.active or worker._request_tasks:
        if time.perf_counter() - start > max_wall_s:
            break
        await asyncio.sleep(0.002)
    await worker.drain_events()


async def run_one_config(worker: Any, be_trace: list[TraceRequest],
                         chunk_size: int, be_inflight: int,
                         synthetic_rt: int,
                         args: argparse.Namespace) -> RunResult:
    """Run one (chunk_size, be_inflight) configuration point."""
    result = RunResult(
        chunk_size=chunk_size,
        be_inflight=be_inflight,
        request_mode=args.request_mode,
        avg_be_input_len=statistics.fmean(
            req.prompt_len for req in be_trace),
        avg_be_output_len=statistics.fmean(
            req.output_len for req in be_trace),
    )
    be_serial = 0
    from saturo.runtime.workload import DECODE, PREFILL
    be_idx = {PREFILL: 0, DECODE: 0}
    lifecycle_idx = 0

    def next_be(phase: str) -> dict[str, Any]:
        nonlocal be_serial
        req = be_trace[be_idx[phase] % len(be_trace)]
        be_idx[phase] += 1
        cmd = be_command(req, be_serial, phase)
        be_serial += 1
        return cmd

    def next_lifecycle_be() -> dict[str, Any]:
        nonlocal be_serial, lifecycle_idx
        req = be_trace[lifecycle_idx % len(be_trace)]
        lifecycle_idx += 1
        cmd = be_lifecycle_command(req, be_serial)
        be_serial += 1
        return cmd

    if args.request_mode == REQUEST_MODE_VLLM:
        # A normal vLLM population. Each request starts with real prompt
        # chunks, becomes decode-ready internally, and is refilled only after
        # the entire output finishes.
        pending = [next_lifecycle_be() for _ in range(be_inflight)]
    else:
        # Preserve the original independent constant Prefill and Decode
        # populations. Replenishing the completed phase keeps both saturated.
        decode_inflight = (round(be_inflight * 0.5)
                           if args.enable_be_decode else 0)
        prefill_inflight = be_inflight - decode_inflight
        pending = [
            *(next_be(DECODE) for _ in range(decode_inflight)),
            *(next_be(PREFILL) for _ in range(prefill_inflight)),
        ]
    result.be_submitted += len(pending)
    for i in range(synthetic_rt):
        # Synthetic RT decode sequences: medium-length prompts to occupy KV.
        pending.append(synthetic_rt_decode_command(
            i, prompt_len=2048, output_len=1024))

    # Register the initial saturated population before starting the warmup
    # clock. Large populations are submitted cooperatively and can take tens
    # of seconds; counting that one-time queue fill inside the measurement
    # window would systematically understate steady-state throughput.
    if pending:
        await worker.submit_batch(pending)
        pending = []

    run_start = time.perf_counter()
    meas_start = run_start + args.warmup_s
    meas_end = meas_start + args.duration_s
    deadline = run_start + args.max_wall_s

    while True:
        now = time.perf_counter()
        if now >= meas_end:
            break
        if now >= deadline:
            raise TimeoutError(f"exceeded --max-wall-s={args.max_wall_s}")

        if pending:
            await worker.submit_batch(pending)
            pending = []
        # Let the AsyncLLMEngine background loop make progress, then
        # harvest whatever events the per-request tasks have emitted.
        await asyncio.sleep(0.002)
        events = await worker.drain_events()
        observed = time.perf_counter()
        in_window = meas_start <= observed < meas_end
        finished_be_phases: list[str] = []

        for ev in events:
            etype = ev.get("event")
            if etype == "iteration_log":
                if not in_window:
                    continue
                n_dec = int(ev.get("n_decode_reqs", 0))
                n_pre = int(ev.get("n_prefill_reqs", 0))
                if n_dec > 0:
                    result.decode_batch_sizes.append(float(n_dec))
                if n_pre > 0:
                    result.prefill_batch_sizes.append(float(n_pre))
                be_prefill_tokens = int(
                    ev.get("be_prefill_tokens", 0) or 0)
                be_decode_tokens = int(
                    ev.get("be_decode_tokens", 0) or 0)
                result.be_prefill_tokens += be_prefill_tokens
                result.be_decode_tokens += be_decode_tokens
                result.be_dummy_handoff_tokens += int(
                    ev.get("be_dummy_handoff_tokens", 0) or 0)
                actual = float(ev.get("actual_time_ms", 0.0) or 0.0)
                if actual > 0.0:
                    result.iteration_actual_ms.append(actual)
                predicted = ev.get("predicted_time_ms")
                if predicted is not None:
                    result.iteration_predicted_ms.append(float(predicted))
                result.scheduler_wall_ms.append(float(
                    ev.get("scheduler_wall_ms", 0.0) or 0.0))
                overhead = ev.get("runtime_overhead_ms")
                if overhead is not None:
                    result.runtime_overhead_ms.append(float(overhead))
                used_gpu_blocks = ev.get("num_used_gpu_blocks")
                if used_gpu_blocks is not None:
                    result.used_gpu_blocks.append(float(used_gpu_blocks))
                running_queue_size = ev.get("scheduler_running_queue_size")
                if running_queue_size is not None:
                    result.running_queue_sizes.append(
                        float(running_queue_size))
                waiting_queue_size = ev.get("scheduler_waiting_queue_size")
                if waiting_queue_size is not None:
                    result.waiting_queue_sizes.append(
                        float(waiting_queue_size))
                result.preempted_total += int(ev.get("preempted", 0) or 0)
                result.blocks_to_swap_in_total += int(
                    ev.get("blocks_to_swap_in", 0) or 0)
                result.blocks_to_swap_out_total += int(
                    ev.get("blocks_to_swap_out", 0) or 0)
                result.iteration_count += 1
            elif etype == "finished":
                kind = ev.get("kind")
                if kind == "be":
                    phase = str(ev.get("phase", PREFILL))
                    phase = DECODE if phase == DECODE else PREFILL
                    finished_be_phases.append(phase)
                    if in_window:
                        result.be_completed += 1
                        result.be_output_tokens += int(
                            ev.get("token_count", 0))
                        if args.request_mode == REQUEST_MODE_VLLM:
                            result.be_e2e_completed += 1
                        elif phase == DECODE:
                            result.be_decode_completed += 1
                        else:
                            result.be_prefill_completed += 1
            elif etype == "token":
                kind = ev.get("kind")
                if kind == "be" and in_window:
                    pass  # token counts come from finished events

        # Keep the BE population saturated.
        for phase in finished_be_phases:
            if args.request_mode == REQUEST_MODE_VLLM:
                pending.append(next_lifecycle_be())
            else:
                pending.append(next_be(phase))
            result.be_submitted += 1

        if not worker.active and not pending and not finished_be_phases:
            # All done, nothing left to process.
            if time.perf_counter() >= meas_start:
                break

    result.window_s = max(0.0, min(time.perf_counter(), meas_end) - meas_start)
    return result


def result_to_dict(r: RunResult) -> dict[str, Any]:
    be_rps = r.be_completed / r.window_s if r.window_s > 0 else 0.0
    be_tok = r.be_output_tokens / r.window_s if r.window_s > 0 else 0.0
    prefill_tok_s = (
        r.be_prefill_tokens / r.window_s if r.window_s > 0 else 0.0)
    decode_tok_s = (
        r.be_decode_tokens / r.window_s if r.window_s > 0 else 0.0)
    prefill_req_s = prefill_tok_s / max(r.avg_be_input_len, 1.0)
    decode_req_s = decode_tok_s / max(r.avg_be_output_len, 1.0)
    decode_batch = summarize(r.decode_batch_sizes)
    prefill_batch = summarize(r.prefill_batch_sizes)
    saturation_req_s = (
        be_rps
        if r.request_mode == REQUEST_MODE_VLLM
        else min(prefill_req_s, decode_req_s)
    )
    return {
        "request_mode": r.request_mode,
        "chunk_size": r.chunk_size,
        "be_inflight": r.be_inflight,
        "window_s": round(r.window_s, 3),
        "be_throughput_req_per_s": round(be_rps, 4),
        "be_saturation_req_per_s": round(saturation_req_s, 4),
        "be_prefill_req_per_s": round(prefill_req_s, 4),
        "be_decode_req_per_s": round(decode_req_s, 4),
        "avg_be_input_len": round(r.avg_be_input_len, 3),
        "avg_be_output_len": round(r.avg_be_output_len, 3),
        "be_throughput_token_per_s": round(be_tok, 1),
        "be_completed": r.be_completed,
        "be_e2e_completed": r.be_e2e_completed,
        "be_submitted": r.be_submitted,
        "be_prefill_completed": r.be_prefill_completed,
        "be_decode_completed": r.be_decode_completed,
        "be_prefill_tokens": r.be_prefill_tokens,
        "be_decode_tokens": r.be_decode_tokens,
        "be_dummy_handoff_tokens": r.be_dummy_handoff_tokens,
        "be_prefill_token_per_s": round(
            prefill_tok_s, 1),
        "be_decode_token_per_s": round(
            decode_tok_s, 1),
        "decode_batch_size": {k: round(float(v), 3)
                              for k, v in decode_batch.items()},
        "prefill_batch_size": {k: round(float(v), 3)
                               for k, v in prefill_batch.items()},
        "iteration_count": r.iteration_count,
        "iteration_actual_ms": {
            k: round(float(v), 3)
            for k, v in summarize(r.iteration_actual_ms).items()
        },
        "iteration_predicted_ms": {
            k: round(float(v), 3)
            for k, v in summarize(r.iteration_predicted_ms).items()
        },
        "scheduler_wall_ms": {
            k: round(float(v), 3)
            for k, v in summarize(r.scheduler_wall_ms).items()
        },
        "runtime_overhead_ms": {
            k: round(float(v), 3)
            for k, v in summarize(r.runtime_overhead_ms).items()
        },
        "used_gpu_blocks_per_virtual_engine": {
            k: round(float(v), 3)
            for k, v in summarize(r.used_gpu_blocks).items()
        },
        "scheduler_running_queue_size": {
            k: round(float(v), 3)
            for k, v in summarize(r.running_queue_sizes).items()
        },
        "scheduler_waiting_queue_size": {
            k: round(float(v), 3)
            for k, v in summarize(r.waiting_queue_sizes).items()
        },
        "preempted_total": r.preempted_total,
        "blocks_to_swap_in_total": r.blocks_to_swap_in_total,
        "blocks_to_swap_out_total": r.blocks_to_swap_out_total,
    }


async def run_chunk(args: argparse.Namespace, chunk_size: int,
                    inflight_levels: list[int],
                    be_trace: list[TraceRequest],
                    runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Create one async engine for a chunk size and sweep inflight levels.

    The whole engine lifecycle lives inside a single event loop because the
    AsyncLLMEngine background loop is bound to the loop that first drives it.
    """
    args.max_num_batched_tokens = chunk_size
    worker = create_worker(args)
    try:
        core_engine = worker.engine.engine
        max_model_len_val = int(core_engine.model_config.max_model_len)
        cache = core_engine.cache_config
        num_gpu_blocks = int(cache.num_gpu_blocks)
        block_size = int(cache.block_size)
        be_trace_sanitized = sanitize(be_trace, max_model_len_val)
        split_decode = (
            args.request_mode == REQUEST_MODE_SPLIT_PHASE
            and args.enable_be_decode
        )
        decode_kv_plan = (
            be_decode_kv_plan(
                num_gpu_blocks,
                block_size,
                be_trace_sanitized,
                args.be_decode_kv_utilization,
            )
            if split_decode else None
        )
        if split_decode:
            os.environ["SATURO_BE_DECODE_BLOCK_LIMIT"] = str(
                decode_kv_plan["be_decode_block_limit"])
        else:
            os.environ.pop("SATURO_BE_DECODE_BLOCK_LIMIT", None)
        print(f"Engine ready (chunk_size={chunk_size}): "
              f"num_gpu_blocks={num_gpu_blocks}, "
              f"block_size={block_size}, max_model_len={max_model_len_val}",
              flush=True)
        if split_decode:
            print("BE-decode KV limit: "
                  f"floor({num_gpu_blocks} * "
                  f"{args.be_decode_kv_utilization}) - "
                  f"{decode_kv_plan['reserved_be_prefill_blocks']} "
                  "reserved BE-prefill blocks "
                  f"(max_prompt={decode_kv_plan['max_be_prompt']}) = "
                  f"{decode_kv_plan['be_decode_block_limit']}",
                  flush=True)
        elif args.request_mode == REQUEST_MODE_SPLIT_PHASE:
            print("BE-decode disabled: prefill-only BE population.",
                  flush=True)
        else:
            print("Request mode: ordinary vLLM Prefill -> Decode lifecycle; "
                  "DummyConnector disabled.", flush=True)

        for inflight in inflight_levels:
            print(f"\n=== chunk_size={chunk_size}, be_inflight={inflight}: "
                  f"warmup {args.warmup_s}s + measure "
                  f"{args.duration_s}s ===", flush=True)
            result = await run_one_config(
                worker, be_trace_sanitized, chunk_size, inflight,
                args.synthetic_rt_decode, args)
            payload = result_to_dict(result)
            payload["be_decode_kv"] = decode_kv_plan
            runs.append(payload)
            print(json.dumps(payload, indent=2), flush=True)
            await drain_worker(worker, args.max_wall_s)
    finally:
        try:
            await worker.shutdown()
        except Exception:
            pass
        del worker
        gc.collect()
    return {
        "num_gpu_blocks": num_gpu_blocks,
        "block_size": block_size,
        "max_model_len": max_model_len_val,
        "decode_kv_plan": decode_kv_plan,
    }


def main() -> None:
    args = parse_args()
    if (args.request_mode == REQUEST_MODE_VLLM
            and args.synthetic_rt_decode > 0):
        raise ValueError(
            "--synthetic-rt-decode requires --request-mode=split-phase; "
            "vllm mode intentionally disables the DummyConnector")
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("VLLM_TARGET_DEVICE", "cuda")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    chunk_sizes_value = str(args.chunk_sizes).strip()
    if chunk_sizes_value.lower() == "auto":
        if args.chunk_size_source == "sweep":
            raise ValueError(
                "--chunk-size-source=sweep requires explicit --chunk-sizes")
        chunk_sizes = list(range(128, 8193, 128))
    else:
        chunk_sizes = [int(x) for x in chunk_sizes_value.split(",")
                       if x.strip()]
    inflight_levels = [int(x) for x in str(args.be_inflight).split(",")
                       if x.strip()]
    if not chunk_sizes:
        raise ValueError("--chunk-sizes had no values")
    if not inflight_levels:
        raise ValueError("--be-inflight had no values")

    out_path = (Path(args.output_json).expanduser().resolve()
                if args.output_json else default_output_path())
    if (args.chunk_size_source == "sweep"
            and len(chunk_sizes) > 1
            and os.environ.get("SATURO_T_POOL_SWEEP_CHILD") != "1"):
        run_sweep_subprocesses(out_path, chunk_sizes)
        return
    be_path = Path(args.be_workload_csv).expanduser().resolve()
    be_trace = filter_be(load_trace(be_path, args.max_source_be), args)
    print(f"Loaded {len(be_trace)} BE rows (after filters).", flush=True)

    sim_be_trace = be_trace
    inferred_max_model_len = infer_model_max_len(args)
    if inferred_max_model_len is not None:
        sim_be_trace = sanitize(be_trace, inferred_max_model_len)
        print("Simulator workload sanitized with "
              f"max_model_len={inferred_max_model_len}: "
              f"{len(sim_be_trace)} rows.", flush=True)

    simulation_plan = None
    selected_chunk_sizes = chunk_sizes
    sim_plan_path = None
    if args.chunk_size_source == "sim":
        sim_plan_path = (Path(args.sim_plan_json).expanduser().resolve()
                         if args.sim_plan_json else out_path.with_suffix(
                             out_path.suffix + ".sim_plan.json"))
        print("=== Running T-pool simulator before engine startup ===",
              flush=True)
        simulation_plan = build_sim_plan(
            args, chunk_sizes, sim_be_trace, sim_plan_path)
        selected_chunk_sizes = [int(simulation_plan["best"]["chunk_size"])]

    print("Initializing one Saturo T-pool async engine: "
          f"model={args.model}, TP={args.tensor_parallel_size}, "
          f"PP={args.pipeline_parallel_size}, load_format={args.load_format}, "
          f"request_mode={args.request_mode}",
          flush=True)

    runs: list[dict[str, Any]] = []
    num_gpu_blocks = 0
    block_size = 16
    max_model_len_val = 2048
    decode_kv_plan: Optional[dict[str, int | float]] = None

    for chunk_size in selected_chunk_sizes:
        # One event loop per engine lifecycle: the AsyncLLMEngine background
        # loop cannot outlive the asyncio.run() that first drove it.
        engine_info = asyncio.run(
            run_chunk(args, chunk_size, inflight_levels, be_trace, runs))
        num_gpu_blocks = int(engine_info["num_gpu_blocks"])
        block_size = int(engine_info["block_size"])
        max_model_len_val = int(engine_info["max_model_len"])
        decode_kv_plan = engine_info["decode_kv_plan"]

    output = {
        "config": {
            "model": args.model,
            "tensor_parallel_size": args.tensor_parallel_size,
            "pipeline_parallel_size": args.pipeline_parallel_size,
            "expert_parallel_size": args.expert_parallel_size,
            "load_format": args.load_format,
            "runtime_profile": args.runtime_profile,
            "max_num_seqs": args.max_num_seqs,
            "be_inflight_candidates": inflight_levels,
            "warmup_s": args.warmup_s,
            "duration_s": args.duration_s,
            "num_gpu_blocks": num_gpu_blocks,
            "block_size": block_size,
            "max_model_len": max_model_len_val,
            "be_workload_csv": str(be_path),
            "request_mode": args.request_mode,
            "be_input_drop_frac": args.be_input_drop_frac,
            "be_input_max_percentile": args.be_input_max_percentile,
            "be_input_max_tokens_applied":
                args.be_input_max_tokens_applied,
            "be_output_keep_frac": args.be_output_keep_frac,
            "be_output_max_len": args.be_output_max_len,
            "enable_be_decode": args.enable_be_decode,
            "be_decode_kv_utilization": args.be_decode_kv_utilization,
            "be_decode_kv": decode_kv_plan,
            "synthetic_rt_decode": args.synthetic_rt_decode,
            "chunk_size_source": args.chunk_size_source,
            "chunk_size_candidates": chunk_sizes,
            "selected_chunk_size": selected_chunk_sizes[0]
            if len(selected_chunk_sizes) == 1 else None,
        },
        "simulator": ({
            "path": str(sim_plan_path),
            "best": simulation_plan["best"],
        } if simulation_plan is not None else None),
        "runs": runs,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\n[Output] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
