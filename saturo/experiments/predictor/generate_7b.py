#!/usr/bin/env python3
"""Generate a 7B iteration-level time predictor from wall-clock measurements.

Builds a linear model ``latency_ms = a*prefill_tokens + b*decode_n + c*decode_sum_ctx + d``
by directly timing ``worker.drive()`` calls under controlled batch compositions.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from saturo.predictors.time_predictor import (
    TimePredictorProfileIdentity,
    time_predictor_cache_path,
)
from saturo.runtime.worker import OfflineEngineWorker
from saturo.runtime.workload import BE, TPUT
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine


def add_be_cmd(prompt_len: int, output_len: int, serial: int) -> dict:
    aid = f"prof_{serial}"
    return {
        "cmd": "add", "request_id": aid, "attempt_id": aid,
        "kind": BE, "phase": TPUT,
        "prompt_len": prompt_len, "max_tokens": output_len,
        "base_tokens": 0,
    }


def measure_iteration_ms(
    worker: OfflineEngineWorker,
    pending: list[dict],
    n_iters: int = 5,
    skip_first: int = 2,
) -> float:
    """Return median iteration wall-clock time in ms."""
    times = []
    for i in range(n_iters):
        t0 = time.perf_counter()
        events = worker.drive(pending)
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= skip_first:
            # Get actual batch composition from iteration_logs
            for ev in events:
                if ev.get("event") == "iteration_log":
                    times.append(dt)
        pending.clear()
    if not times:
        return 0.0
    return float(np.median(times))


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "/root/huggingface_config/Qwen2.5-7B"
    gpu = sys.argv[2] if len(sys.argv) > 2 else "0"
    mlen = int(sys.argv[3]) if len(sys.argv) > 3 else 2048

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("VLLM_TARGET_DEVICE", "cuda")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

    w_args = {
        "worker_id": 0, "worker_role": TPUT,
        "tensor_parallel_size": 1, "pipeline_parallel_size": 1,
        "expert_parallel_size": 0,
        "model": model,
        "gpu_memory_utilization": 0.90,
        "load_format": "dummy", "dtype": "float16",
        "rt_prefill_deadline_s": 3600.0,
        "disable_inductor": True,
        "rt_tbt_slo_ms": 100.0,
        "p_pool_be_chunk_ms": 100.0,
        "p_pool_preemption_margin": 0.0,
        "runtime_profile": "",
    }

    original_descriptor = LLMEngine.__dict__["from_engine_args"]
    original_factory = LLMEngine.from_engine_args

    def patched_from_engine_args(cls, engine_args, *pos, **kwargs):
        engine_args.max_num_seqs = 256
        engine_args.max_model_len = mlen
        engine_args.max_num_batched_tokens = 8192
        return original_factory(engine_args, *pos, **kwargs)

    LLMEngine.from_engine_args = classmethod(patched_from_engine_args)

    worker = None
    serial = 0
    measurements: list[dict] = []

    try:
        worker = OfflineEngineWorker(w_args)
    finally:
        LLMEngine.from_engine_args = original_descriptor

    real_mlen = int(worker.engine.model_config.max_model_len)
    block_size = int(worker.engine.cache_config.block_size)
    total_blocks = int(worker.engine.cache_config.num_gpu_blocks)
    max_active = total_blocks // max(1, real_mlen // block_size)
    print(f"num_gpu_blocks={total_blocks} block_size={block_size} "
          f"max_model_len={real_mlen} max_active_est={max_active}")

    # Helper: drain all active requests
    def drain():
        for aid in list(worker.active.keys()):
            worker.abort(aid)
        while worker.engine.has_unfinished_requests():
            worker.drive([])

    # Helper: fill decode requests to a target count and warm up
    def fill_decode(target: int, ctx_len: int = 1024) -> int:
        nonlocal serial
        needed = max(0, target - len(worker.active))
        pending = []
        for _ in range(needed):
            pending.append(add_be_cmd(ctx_len, 256, serial))
            serial += 1
        if pending:
            worker.drive(pending)
        # Drive until all are in decode phase.
        for _ in range(ctx_len + 200):
            if not worker.engine.has_unfinished_requests():
                break
            worker.drive([])
        return len(worker.active)

    # ---- Phase 1: Pure decode at varying batch sizes ----
    for n_decode in [8, 16, 24, 32, 48, 64, 72, 80]:
        if n_decode > max_active * 0.95:
            continue
        drain()
        actual = fill_decode(n_decode, ctx_len=1024)
        if actual < n_decode * 0.8:
            continue
        # Measure pure decode iterations.
        for _ in range(15):  # drive through
            worker.drive([])
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            worker.drive([])
            dt = (time.perf_counter() - t0) * 1000.0
            times.append(dt)
        med = float(np.median(times))
        measurements.append({
            "prefill_tokens": 0, "prefill_sum_ctx": 0,
            "decode_n": actual, "decode_sum_ctx": actual * 1024,
            "latency_ms": med,
        })
        print(f"  decode_only: n={actual:>3} ctx=1024 → {med:.2f}ms")

    # ---- Phase 2: Pure decode at longer context ----
    for n_decode in [16, 32, 48]:
        if n_decode > max_active * 0.95:
            continue
        drain()
        actual = fill_decode(n_decode, ctx_len=2047)
        if actual < n_decode * 0.8:
            continue
        for _ in range(15):
            worker.drive([])
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            worker.drive([])
            dt = (time.perf_counter() - t0) * 1000.0
            times.append(dt)
        med = float(np.median(times))
        measurements.append({
            "prefill_tokens": 0, "prefill_sum_ctx": 0,
            "decode_n": actual, "decode_sum_ctx": actual * 2047,
            "latency_ms": med,
        })
        print(f"  decode_only: n={actual:>3} ctx=2047 → {med:.2f}ms")

    # ---- Phase 3: Mixed prefill + decode ----
    for n_decode in [32, 48, 64]:
        if n_decode > max_active * 0.95:
            continue
        for prefill_tokens in [512, 1024, 1536, 2047]:
            drain()
            actual_decode = fill_decode(n_decode, ctx_len=1024)
            if actual_decode < n_decode * 0.8:
                continue

            # Add one prefill request and measure latency.
            prefill_cmd = add_be_cmd(prefill_tokens, 128, serial)
            serial += 1
            times = []
            for i in range(10):
                t0 = time.perf_counter()
                worker.drive([prefill_cmd] if i == 0 else [])
                dt = (time.perf_counter() - t0) * 1000.0
                if i >= 1:  # skip first (has prefill queued)
                    times.append(dt)
                prefill_cmd = None  # only inject once
            med = float(np.median(times)) if times else 0.0

            # The iteration that processed the prefill.
            t0 = time.perf_counter()
            events = worker.drive([add_be_cmd(prefill_tokens, 128, serial)])
            serial += 1
            prefill_dt = (time.perf_counter() - t0) * 1000.0

            # Get actual batch composition from the prefill iteration log.
            p_tok = prefill_tokens
            p_ctx = prefill_tokens  # chunked prefill: context ≈ query_len
            d_n = 0
            d_ctx = 0
            for ev in events:
                if ev.get("event") == "iteration_log":
                    p_tok = float(ev.get("prefill_tokens", prefill_tokens))
                    d_n = float(ev.get("n_decode_reqs", 0))
                    d_ctx = float(ev.get("decode_sum_ctx", 0))
                    # prefill_sum_ctx is ctx_tokens_before + causal mask area
                    p_ctx = float(ev.get("prefill_sum_ctx", prefill_tokens))

            measurements.append({
                "prefill_tokens": p_tok, "prefill_sum_ctx": p_ctx,
                "decode_n": d_n, "decode_sum_ctx": d_ctx,
                "latency_ms": prefill_dt,
            })
            print(f"  mixed: decode={actual_decode} prefill={prefill_tokens} "
                  f"→ p_tok={p_tok:.0f} d_n={d_n:.0f} {prefill_dt:.2f}ms")

    # ---- Fit model ----
    if len(measurements) < 10:
        raise RuntimeError(f"Too few measurements: {len(measurements)}")

    x = np.array([[m["prefill_tokens"], m["prefill_sum_ctx"],
                   m["decode_n"], m["decode_sum_ctx"], 1.0]
                  for m in measurements], dtype=float)
    y = np.array([m["latency_ms"] for m in measurements], dtype=float)

    # Non-negative least squares for physically meaningful coefficients.
    coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    feature_coef = np.maximum(coef[:4], 0.0)
    intercept = float(np.median(y - (x[:, :4] @ feature_coef)))
    coef_final = np.concatenate([feature_coef, [max(intercept, 0.0)]])
    pred = x @ coef_final
    residuals = y - pred
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    mape = float(np.mean(np.abs(residuals) / np.maximum(y, 1e-9)) * 100.0)

    print(f"\n=== Fitted model ===")
    print(f"  coef = {[float(v) for v in coef_final[:4]]}")
    print(f"  intercept = {float(coef_final[4]):.3f}")
    print(f"  R² = {r2:.4f}, MAPE = {mape:.1f}%")
    for m in measurements:
        feats = np.array([m["prefill_tokens"], m["prefill_sum_ctx"],
                          m["decode_n"], m["decode_sum_ctx"], 1.0])
        pred_ms = float(feats @ coef_final)
        err = m["latency_ms"] - pred_ms
        print(f"    actual={m['latency_ms']:6.1f}ms pred={pred_ms:6.1f}ms "
              f"err={err:+5.1f}ms "
              f"ptok={m['prefill_tokens']:5.0f} d_n={m['decode_n']:3.0f}")

    # Write time predictor JSON.
    identity = TimePredictorProfileIdentity.from_engine_configs(
        model_config=worker.engine.model_config,
        parallel_config=worker.engine.parallel_config,
    )
    num_layers = _get_num_layers(worker.engine)
    out_dir = Path("saturo/results/time_predictors")
    profile_path = time_predictor_cache_path(identity, out_dir)
    profile_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "time_predictor": {
            "model_type": "iteration_level",
            "description": f"7B wall-clock profiling on {identity.gpu_name}",
            "features": ["prefill_tokens", "prefill_sum_ctx",
                         "n_decode_reqs", "decode_sum_ctx"],
            "coef": [float(v) for v in coef_final[:4]],
            "intercept": float(coef_final[4]),
            "r2": float(r2),
            "mape": mape,
            "n_samples": int(len(measurements)),
            "profile_identity": identity.to_dict(),
            "profile_key": identity.digest,
            "num_layers": int(num_layers or 28),
            "meta": {"total_model_layers": int(num_layers or 28)},
        },
        "kv_cache_block": {
            "total_gpu_blocks": total_blocks,
            "block_size": block_size,
        },
    }

    tmp_path = profile_path.with_suffix(profile_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    tmp_path.replace(profile_path)
    print(f"\nWrote: {profile_path}")

    # Cleanup
    worker.shutdown()


def _get_num_layers(engine):
    cfg = getattr(engine.model_config, "hf_config", None)
    if cfg:
        for name in ("num_hidden_layers", "n_layer", "num_layers"):
            v = getattr(cfg, name, None)
            if v is not None:
                return int(v)
    return 28


if __name__ == "__main__":
    main()
