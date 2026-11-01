#!/usr/bin/env python3
"""Compare T-pool simulator (run_trace) vs real benchmark across chunk sizes.

Usage:
  python saturo/experiments/pool_validation/compare_tput_sim_real.py \
      <config_name> <profile_json> <real_glob>
E.g.:
  python saturo/experiments/pool_validation/compare_tput_sim_real.py \
      tp2_pp1 <profile.json> \
      "saturo/results/t_pool_sat/sweep_perchunk/tp2_chunk*/t_pool_sat_*.json"
"""
import sys, csv, json, glob, math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from saturo.simulators.tput_simulator.config import (
    BEConfig, ModelConfig, SimulatorConfig)
from saturo.simulators.tput_simulator.simulator import (
    ThroughputPoolSimulator, TraceRequest)

BE_CSV = 'data/datasets/be_workload.csv'
MAX_MODEL_LEN = 32768


def load_trace():
    rows = []
    for r in csv.DictReader(open(BE_CSV)):
        p = int(float(r['prompt_len'])); o = int(float(r['output_len']))
        if p >= MAX_MODEL_LEN:
            continue
        rows.append((p, o))
    return rows


def kv_info(profile):
    kv = json.load(open(profile)).get('kv_cache_block', {})
    return kv.get('total_gpu_blocks', 0), kv.get('block_size', 16)


def real_results(real_glob):
    """Return real points and the fixed BE inflight levels they used."""
    out = {}
    inflight_levels = set()
    for path in glob.glob(real_glob):
        d = json.load(open(path))
        for run in d.get('runs', []):
            c = int(run['chunk_size'])
            inflight_levels.add(int(run.get('be_inflight', 128)))
            req = run.get(
                'be_saturation_req_per_s', run['be_throughput_req_per_s'])
            iters = run.get('iteration_count', 0)
            win = run.get('window_s', 0) or 0
            avg_ms = (win * 1000.0 / iters) if iters else 0.0
            out[c] = (req, avg_ms, iters, run.get('be_completed'))
    return out, inflight_levels


def main():
    config_name, profile, real_glob = sys.argv[1], sys.argv[2], sys.argv[3]
    blocks, bs = kv_info(profile)
    rows = load_trace()
    real, inflight_levels = real_results(real_glob)
    if len(inflight_levels) != 1:
        raise ValueError(
            'comparison requires exactly one be_inflight level; got %s'
            % sorted(inflight_levels))
    be_inflight = next(iter(inflight_levels))
    chunk_sizes = sorted(real)
    ai = sum(p for p, _ in rows) / len(rows)
    ao = sum(o for _, o in rows) / len(rows)
    decode_block_limit = (
        math.floor(blocks * 0.95)
        - math.ceil(max(p for p, _ in rows) / bs)
    )
    cfg = SimulatorConfig(
        duration_s=300, warmup_s=10, latency_model_path=profile,
        model=ModelConfig(block_size=bs),
        be=BEConfig(avg_input_len=int(round(ai)), avg_output_len=int(round(ao)),
                    avg_decode_context_len=int(round(ai + ao / 2)),
                    total_gpu_blocks=blocks,
                    be_decode_blocks=decode_block_limit,
                    be_inflight=be_inflight,
                    chunk_size_candidates=chunk_sizes))
    sim = ThroughputPoolSimulator(cfg)
    trace = [TraceRequest(p, o, 'be_%d' % i) for i, (p, o) in enumerate(rows)]
    tr = sim.run_split_trace(trace)
    sim_by_budget = {}
    for c in tr.candidates:
        d = c.to_dict()
        b = d.get('total_token_budget', d.get('chunk_size'))
        sim_by_budget[int(b)] = (
            d.get('be_saturation_req_per_s', d.get('be_throughput_req_per_s')),
            d.get('avg_iteration_latency_ms', 0.0))

    print(f"\n==== {config_name} | profile blocks={blocks} bs={bs} ====")
    print(f"{'chunk':>6} | {'real req/s':>10} {'sim req/s':>10} {'err%':>7} | "
          f"{'real ms':>8} {'sim ms':>8}")
    print('-' * 62)
    maxerr = 0.0
    for c in sorted(real):
        rreq, rms, _, _ = real[c]
        sreq, sms = sim_by_budget.get(c, (float('nan'), float('nan')))
        err = (sreq - rreq) / rreq * 100.0 if rreq else float('nan')
        maxerr = max(maxerr, abs(err))
        print(f"{c:>6} | {rreq:>10.4f} {sreq:>10.4f} {err:>+6.1f}% | "
              f"{rms:>8.1f} {sms:>8.1f}")
    print('-' * 62)
    print(f"max |err| = {maxerr:.1f}%  ({'PASS <20%' if maxerr < 20 else 'FAIL'})")


if __name__ == '__main__':
    main()
