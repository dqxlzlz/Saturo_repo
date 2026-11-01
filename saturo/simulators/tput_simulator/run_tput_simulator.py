"""CLI entry point for the T-pool BE-only throughput simulator.

Aligns with ``saturo/cli/global_scheduler.py`` conventions:
* ``--runtime-profile`` directory (instead of a hard-coded latency-model path).
* ``--be-workload-csv`` supplies the BE length distribution.
* Internal search grids (decode/chunk/prefill candidates) are NOT exposed
  as CLI flags — they use sensible defaults shared with the resource
  allocation path.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from saturo.simulators.tput_simulator.config import (  # noqa: E402
    BEConfig,
    ModelConfig,
    SimulatorConfig,
)
from saturo.simulators.tput_simulator.simulator import (  # noqa: E402
    ThroughputPoolSimulator,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _find_time_predictor(runtime_profile: str, config_name: str) -> Path:
    """Locate the newest time_predictor JSON matching *config_name*."""
    root = Path(runtime_profile)
    if not root.is_dir():
        raise SystemExit(
            f"--runtime-profile is not a directory: {root}")
    needle = f"_{config_name}_"
    candidates: list[Path] = []
    for path in sorted(root.rglob("time_predictor_*.json")):
        if path.name.endswith(".raw_profile.json"):
            continue
        if needle in path.name:
            candidates.append(path)
    if not candidates:
        raise SystemExit(
            f"No time_predictor matching {config_name} found in {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _read_be_workload_avg_lengths(
    csv_path: str,
) -> tuple[int, int, int]:
    """Return (avg_input_len, avg_output_len, avg_decode_context_len)."""
    prompts: list[int] = []
    outputs: list[int] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            prompts.append(int(float(row["prompt_len"])))
            outputs.append(int(float(row["output_len"])))
    if not prompts:
        raise SystemExit(f"BE workload CSV is empty: {csv_path}")
    avg_in = int(round(sum(prompts) / len(prompts)))
    avg_out = int(round(sum(outputs) / len(outputs)))
    avg_ctx = max(1, int(avg_in + avg_out / 2))
    return avg_in, avg_out, avg_ctx


def _read_kv_block_info(latency_model_path: str) -> tuple[int, int]:
    """Return (total_gpu_blocks, block_size) from the time-predictor JSON."""
    with open(latency_model_path) as f:
        data = json.load(f)
    kv = data.get("kv_cache_block", {})
    return kv.get("total_gpu_blocks", 0), kv.get("block_size", 16)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Saturo T-pool BE-only throughput simulator")
    parser.add_argument(
        "--runtime-profile",
        required=True,
        help="Directory containing TimePredictor profile JSON files "
             "(e.g. saturo/results/runtime_profiles).",
    )
    parser.add_argument(
        "--config-name",
        default="tp1_pp1",
        help="Parallel config name used to select the right time_predictor "
             "inside --runtime-profile (e.g. tp1_pp1).",
    )
    parser.add_argument(
        "--be-workload-csv",
        required=True,
        help="BE workload CSV. Columns: prompt_len, output_len.",
    )
    parser.add_argument(
        "--be-decode-blocks",
        type=int,
        default=0,
        help="Per-GPU KV-cache blocks. 0 = read from time_predictor profile.",
    )
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--warmup", type=float, default=0.0)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--print-all", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# config builders
# ---------------------------------------------------------------------------

def build_config(
    args: argparse.Namespace,
    latency_model_path: str,
    avg_in: int,
    avg_out: int,
    avg_decode_ctx: int,
    be_decode_blocks: int,
    block_size: int,
) -> SimulatorConfig:
    return SimulatorConfig(
        duration_s=args.duration,
        warmup_s=args.warmup,
        latency_model_path=latency_model_path,
        model=ModelConfig(block_size=block_size),
        be=BEConfig(
            avg_input_len=avg_in,
            avg_output_len=avg_out,
            avg_decode_context_len=avg_decode_ctx,
            be_decode_blocks=be_decode_blocks,
        ),
    )


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def run_tput_profile(args: argparse.Namespace) -> dict:
    # 1. Resolve time predictor JSON from runtime-profile directory
    latency_model = _find_time_predictor(
        args.runtime_profile, args.config_name)

    # 2. Read average request lengths from the BE workload CSV
    avg_in, avg_out, avg_decode_ctx = _read_be_workload_avg_lengths(
        args.be_workload_csv)

    # 3. Resolve KV-cache block budget (explicit arg or from profile)
    be_decode_blocks = args.be_decode_blocks
    block_size = args.block_size
    if be_decode_blocks <= 0:
        profile_blocks, profile_bs = _read_kv_block_info(str(latency_model))
        be_decode_blocks = profile_blocks or be_decode_blocks
        block_size = profile_bs or block_size

    config = build_config(args, str(latency_model), avg_in, avg_out,
                          avg_decode_ctx, be_decode_blocks, block_size)
    simulator = ThroughputPoolSimulator(config)
    result = simulator.run()
    best = result.best.to_dict()
    be_throughput = best.get(
        "be_saturation_req_per_s",
        best.get("be_throughput_req_per_s", 0.0),
    )

    payload = {
        "pool": "T",
        "mode": result.mode,
        "config_name": args.config_name,
        "config": {
            "duration_s": args.duration,
            "warmup_s": args.warmup,
            "runtime_profile": args.runtime_profile,
            "latency_model": str(latency_model),
            "be_workload_csv": args.be_workload_csv,
            "avg_input_len": avg_in,
            "avg_output_len": avg_out,
            "avg_decode_context_len": result.avg_decode_context_len,
            "be_decode_blocks": be_decode_blocks,
            "max_decode_batch_size_by_memory": (
                result.max_decode_batch_size_by_memory),
            "block_size": block_size,
        },
        "best": best,
        "allocation_profile_entry": {
            "saturation": [{
                "rt_rps": 0.0,
                "be_throughput": round(float(be_throughput), 9),
            }],
        },
        "candidates": (
            [item.to_dict() for item in result.candidates]
            if args.print_all else []
        ),
    }
    return payload


def print_summary(payload: dict) -> None:
    best = payload["best"]
    cfg = payload["config"]
    print("\n[T-pool BE-only]")
    print(f"  config              : {payload['config_name']}")
    print(f"  mode                : {payload['mode']}")
    print(f"  workload csv        : {cfg['be_workload_csv']}")
    print(f"  avg input/output    : "
          f"{cfg['avg_input_len']}/{cfg['avg_output_len']} tokens")
    print(f"  avg decode context  : "
          f"{cfg['avg_decode_context_len']} tokens")
    print(f"  max decode by memory: "
          f"{cfg['max_decode_batch_size_by_memory']}")
    print(f"  runtime profile     : {cfg['runtime_profile']}")
    print(f"  latency model       : {cfg['latency_model']}")
    print("\n[Best]")
    print(f"  BE throughput       : "
          f"{best['be_saturation_req_per_s']:.9f} req/s")
    if payload["mode"] == "steady_state":
        print(f"  prefill req tput    : {best['prefill_req_per_s']:.9f} req/s")
        print(f"  decode req tput     : {best['decode_req_per_s']:.9f} req/s")
        print(f"  iteration latency   : {best['iteration_latency_ms']:.6f} ms")
    else:
        print(f"  completed requests  : "
              f"{best['completed_requests']}/{best['total_requests']}")
        print(f"  measured time       : {best['measured_time_s']:.6f} s")
        print(f"  avg step latency    : "
              f"{best['avg_iteration_latency_ms']:.6f} ms")
        print(f"  p90 step latency    : "
              f"{best['p90_iteration_latency_ms']:.6f} ms")
        print(f"  swap out/in         : "
              f"{best.get('swap_out_count', 0)}/"
              f"{best.get('swap_in_count', 0)}")
    print(f"  decode/max seqs D   : {best['decode_batch_size']}")
    if "chunk_size" in best:
        print(f"  chunk size budget   : {best['chunk_size']}")
    print(f"  prefill tokens C    : {best['prefill_chunk_size']}")
    if "total_token_budget" in best:
        print(f"  total token budget  : {best['total_token_budget']}")
    print(f"  raw token tput      : {best['raw_tokens_per_s']:.3f} tok/s")
    if "decode_blocks" in best:
        print(f"  decode blocks       : {best['decode_blocks']}")
    if "peak_kv_blocks" in best:
        print(f"  peak KV blocks      : {best['peak_kv_blocks']}")


def main() -> None:
    args = parse_args()
    payload = run_tput_profile(args)
    print_summary(payload)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[Output] wrote {args.output}")
    else:
        print(json.dumps(payload if args.print_all else {
            "pool": payload["pool"],
            "config_name": payload["config_name"],
            "best": payload["best"],
            "allocation_profile_entry": payload["allocation_profile_entry"],
        }, indent=2))


if __name__ == "__main__":
    main()
