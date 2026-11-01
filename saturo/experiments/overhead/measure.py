"""CPU-only measurements of the actual optimization functions, never fake timings."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def path(value):
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT / p


def csv_rows(value):
    with path(value).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty trace: " + str(value))
    for r in rows:
        for key in ("prompt_len", "output_len"):
            if int(r[key]) <= 0:
                raise ValueError("Nonpositive length in " + str(value))
    return rows


def runtime_file(value):
    p = path(value)
    if p.is_dir():
        matches = [q for q in p.rglob("time_predictor_*.json")
                   if not q.name.endswith(".raw_profile.json") and "synthetic" not in q.name]
        if len(matches) != 1:
            raise ValueError("Pass one explicit measured runtime profile, found " + str(len(matches)))
        p = matches[0]
    payload = json.loads(p.read_text())
    timing = payload.get("time_predictor", payload)
    if str(timing.get("meta", {}).get("note", "")).upper().startswith("SYNTHETIC"):
        raise ValueError("Synthetic timing profile is not accepted")
    from saturo.predictors.time_predictor import TimePredictor
    predictor = TimePredictor(str(p))
    if not predictor.available:
        raise ValueError("Unavailable timing predictor; no fixed-ms fallback is allowed")
    return p, predictor, payload.get("kv_cache_block", {})


def window_rows(value, window_s):
    rows = csv_rows(value)
    offsets = [float(r["arrival_offset_s"]) for r in rows]
    origin = min(offsets)
    coverage = max(offsets) - origin
    if coverage + 1e-6 < window_s:
        raise ValueError(f"Trace covers {coverage:.3f}s, shorter than requested K={window_s}s")
    selected = [dict(r, arrival_offset_s=str(offset - origin))
                for r, offset in zip(rows, offsets) if offset - origin < window_s]
    if not selected:
        raise ValueError("History window has no requests")
    return selected


def p_budget(case):
    cfg, params = case["config"], case["study_parameters"]
    k = float(params["history_window_s"])
    source = params.get("history_csv")
    if not source:
        raise ValueError("An independent history_csv is required; do not seed from future evaluation requests")
    rows = window_rows(source, k)
    profile, _, blocks = runtime_file(params.get("pool_runtime_profile") or cfg["runtime_profile"])
    from saturo.simulators.prefill_simulator.run_prefill_simulator import run_prefill_profile
    return run_prefill_profile(
        rt_trace=str(path(source)), rt_rows=rows, runtime_profile=str(profile),
        be_trace=str(path(cfg["be_workload_csv"])), num_req=len(rows), target_rps=len(rows) / k,
        ttft_slo=cfg.get("rt_ttft_slo_ms", 1000) / 1000,
        max_rt_violation_rate=cfg.get("rt_max_violation_rate", .05),
        max_candidates=cfg.get("resource_profile_max_candidates", 64),
        memory_watermark=cfg.get("resource_memory_watermark", .9),
        max_prefill_chunk=cfg.get("resource_max_prefill_chunk", 8192),
        prefill_search_granularity=cfg.get("resource_prefill_search_granularity", 1),
        max_model_len=cfg.get("max_model_len"),
        total_gpu_blocks=int(blocks["total_gpu_blocks"]),
    )


def build_operation(case, kind):
    cfg, params = case["config"], case["study_parameters"]
    if kind == "allocator":
        from saturo.resource_allocation.profiles import load_profiles
        from saturo.resource_allocation.solver import ResourceAllocationSolver, Workload
        # The microbenchmark consumes an EXISTING table; it must not trigger GPU profiling.
        profiles = load_profiles(path(cfg["resource_profile"]))
        solver = ResourceAllocationSolver(profiles)
        workload = Workload(cfg["target_rps"], cfg["target_rps"])
        budget = int(params["gpu_budget"])
        def operation():
            allocations = solver.solve(total_gpus=budget, workload=workload, top_k=1)
            if not allocations:
                raise ValueError("No feasible allocation for measured input table and workload")
            return allocations[0].to_dict()
        return operation, "solver.solve only; profile loading excluded"
    if kind == "p":
        # Validate all inputs before the warmup/sample loop.
        window_rows(params.get("history_csv", ""), float(params["history_window_s"]))
        runtime_file(params.get("pool_runtime_profile") or cfg["runtime_profile"])
        return lambda: p_budget(case), "P optimizer API including CPU profile setup; trace source is a frozen CSV"
    from saturo.predictors.qrf import load_required_qrf
    from saturo.saturation.decode_saturation_core import (
        DecodeSaturationEngine, SaturationParams, RequestForecast)
    _, predictor, blocks = runtime_file(params.get("pool_runtime_profile") or cfg["runtime_profile"])
    qrf = load_required_qrf(str(path(cfg["model_len_predictor"])))
    if not params.get("history_csv"):
        raise ValueError("D overhead requires independent completed history_csv")
    rows = csv_rows(params["history_csv"])
    count = int(params["batch_size"])
    if len(rows) < count:
        raise ValueError("Not enough distinct historical requests for requested RT batch")
    rows = rows[:count]
    engine = DecodeSaturationEngine(
        SaturationParams(tbt_slo_ms=cfg.get("rt_tbt_slo_ms", 150),
                         lookahead_window_s=cfg.get("decode_saturation_lookahead_s", 60),
                         block_size=int(blocks["block_size"]),
                         total_gpu_blocks=int(blocks["total_gpu_blocks"])),
        predictor.predict_forward_ms)
    eta = cfg.get("resource_decode_model_len_quantile", .95)
    def operation():
        forecasts = []
        occupied = 0
        for i, row in enumerate(rows):
            prompt = int(row["prompt_len"])
            generated = max(0, int(row["output_len"]) // 2)
            remaining = qrf.predict_remaining(
                eta=eta, prompt_len=prompt, generated_len=generated)
            forecasts.append(RequestForecast(str(i), prompt, generated, remaining, 0))
            occupied += math.ceil((prompt + generated) / int(blocks["block_size"]))
        decision = engine.compute_decision(
            active_rt=forecasts, future_rt=[], current_rt_blocks=occupied,
            current_be_blocks=0, now_s=0)
        return {"target_interval_ms": decision.target_interval_ms,
                "predicted_peak_rt_blocks": decision.predicted_peak_rt_blocks}
    return operation, "QRF + synchronous D target optimization on historical partial states; not GPU async critical-path overhead"


def percentile(values, q):
    values = sorted(values)
    i = (len(values) - 1) * q
    lo, hi = math.floor(i), math.ceil(i)
    return values[lo] + (values[hi] - values[lo]) * (i - lo)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=("p", "d", "allocator"), required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--repeats", type=int, default=30)
    args = p.parse_args()
    if args.repeats < 1:
        p.error("--repeats must be positive")
    case = json.loads(path(args.case).read_text())
    operation, boundary = build_operation(case, args.kind)
    operation()  # Untimed warmup; imports and artifact loads are not first samples.
    cpu, wall = [], []
    for _ in range(args.repeats):
        c0, w0 = time.process_time_ns(), time.perf_counter_ns()
        last = operation()
        wall.append((time.perf_counter_ns() - w0) / 1e6)
        cpu.append((time.process_time_ns() - c0) / 1e6)
    result = {"kind": args.kind, "measurement_boundary": boundary, "repeats": args.repeats,
              "cpu_ms": cpu, "wall_ms": wall,
              "cpu_median_ms": statistics.median(cpu), "cpu_p95_ms": percentile(cpu, .95),
              "wall_median_ms": statistics.median(wall), "wall_p95_ms": percentile(wall, .95),
              "last_result": last, "status": "measured"}
    output = path(case["directory"]); output.mkdir(parents=True, exist_ok=True)
    (output / "measurement.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

