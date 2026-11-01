#!/usr/bin/env python3
"""Run the concrete BE trace through one TP TimePredictor."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from saturo.simulators.tput_simulator.config import BEConfig, ModelConfig, SimulatorConfig
from saturo.simulators.tput_simulator.simulator import ThroughputPoolSimulator, TraceRequest


TP_SIZE = int(os.environ.get("TP_SIZE", "2"))
PROFILE_HASH = {
    1: "7cf6d80ff959485c",
    2: "830481095bd45b35",
    4: "bc11c59e400ce84e",
}[TP_SIZE]
DEFAULT_PROFILE = (
    PROJECT_ROOT
    / "saturo/results/runtime_profiles/root-models-qwen-qwen2.5-72b-instruct"
    / (
        "time_predictor_root-models-qwen-qwen2.5-72b-instruct_"
        f"gpu-nvidia-gb200_tp{TP_SIZE}_pp1_torch.float16_{PROFILE_HASH}.json"
    )
)
PROFILE = Path(os.environ.get("PROFILE_PATH", str(DEFAULT_PROFILE)))
OUTPUT_SUFFIX = os.environ.get("OUTPUT_SUFFIX", "launchblocking")
CHUNK_SIZES = [
    int(value)
    for value in os.environ.get("CHUNK_SIZES", "256,512,768,1024").split(",")
    if value.strip()
]
SIM_DURATION_S = float(os.environ.get("SIM_DURATION_S", "310"))
SIM_WARMUP_S = float(os.environ.get("SIM_WARMUP_S", "10"))
BE_INFLIGHT = int(os.environ.get("BE_INFLIGHT", "128"))
BE_DECODE_KV_UTILIZATION = float(os.environ.get(
    "BE_DECODE_KV_UTILIZATION", "0.95"))
OUTPUT = Path(os.environ.get(
    "OUTPUT_PATH",
    str(PROJECT_ROOT / "saturo/results/t_pool_sat" /
        f"tp{TP_SIZE}_run_trace_{OUTPUT_SUFFIX}_20260715.json"),
))


def main() -> None:
    rows: list[TraceRequest] = []
    with (PROJECT_ROOT / "data/datasets/be_workload.csv").open(newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            prompt = int(float(row["prompt_len"]))
            output = int(float(row["output_len"]))
            if prompt >= 32768:
                continue
            rows.append(TraceRequest(prompt, output, f"be_{index}"))

    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    kv = profile.get("kv_cache_block", {})
    prompts = [request.prompt_len for request in rows]
    outputs = [request.output_len for request in rows]
    avg_input = round(sum(prompts) / len(prompts))
    avg_output = round(sum(outputs) / len(outputs))
    total_gpu_blocks = int(kv["total_gpu_blocks"])
    block_size = int(kv.get("block_size", 16))
    decode_block_limit = (
        math.floor(total_gpu_blocks * BE_DECODE_KV_UTILIZATION)
        - math.ceil(max(prompts) / block_size)
    )
    if decode_block_limit <= 0:
        raise ValueError("no BE decode KV blocks remain after prefill reserve")
    config = SimulatorConfig(
        duration_s=SIM_DURATION_S,
        warmup_s=SIM_WARMUP_S,
        latency_model_path=str(PROFILE),
        model=ModelConfig(block_size=block_size),
        be=BEConfig(
            avg_input_len=avg_input,
            avg_output_len=avg_output,
            avg_decode_context_len=round(avg_input + avg_output / 2),
            total_gpu_blocks=total_gpu_blocks,
            be_decode_blocks=decode_block_limit,
            be_inflight=BE_INFLIGHT,
            chunk_size_candidates=CHUNK_SIZES,
        ),
    )
    result = ThroughputPoolSimulator(config).run_split_trace(rows)
    payload = {
        "tp": TP_SIZE,
        "duration_s": config.duration_s,
        "warmup_s": config.warmup_s,
        "profile": str(PROFILE),
        "trace_requests": len(rows),
        "be_inflight": BE_INFLIGHT,
        "be_decode_kv_utilization": BE_DECODE_KV_UTILIZATION,
        "be_decode_block_limit": decode_block_limit,
        "result": result.to_dict(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
