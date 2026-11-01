#!/usr/bin/env python3
"""Minimal 7B time predictor: measure decode-only and prefill-only latencies.

Runs simple batch compositions and times ``worker.drive()`` calls directly.
Each request uses large max_tokens to keep it alive during measurement.
"""

from __future__ import annotations

import json, os, sys, time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from saturo.predictors.time_predictor import (
    TimePredictorProfileIdentity, time_predictor_cache_path,
)
from saturo.runtime.worker import OfflineEngineWorker
from saturo.runtime.workload import BE, TPUT
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "/root/huggingface_config/Qwen2.5-7B"
    gpu = sys.argv[2] if len(sys.argv) > 2 else "0"
    mlen = int(sys.argv[3]) if len(sys.argv) > 3 else 2048

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    for k in ("VLLM_USE_V1", "VLLM_TARGET_DEVICE"):
        os.environ.setdefault(k, "0" if k == "VLLM_USE_V1" else
                              os.environ.get(k, "cuda"))

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

    orig_desc = LLMEngine.__dict__["from_engine_args"]
    orig_fact = LLMEngine.from_engine_args

    def patched(cls, ea, *a, **kw):
        ea.max_num_seqs = 256
        ea.max_model_len = mlen
        ea.max_num_batched_tokens = 8192
        return orig_fact(ea, *a, **kw)

    LLMEngine.from_engine_args = classmethod(patched)
    try:
        worker = OfflineEngineWorker(w_args)
    finally:
        LLMEngine.from_engine_args = orig_desc

    block_size = int(worker.engine.cache_config.block_size)
    total_blocks = int(worker.engine.cache_config.num_gpu_blocks)
    max_active = total_blocks // max(1, mlen // block_size)
    print(f"blocks={total_blocks} bs={block_size} mlen={mlen} max_active≈{max_active}")

    # Use very large max_tokens so requests stay alive.
    HUGE = 99999

    def drive_n(n: int, pending: list = None) -> float:
        """Drive N steps, return median iter time in ms."""
        times = []
        p = pending or []
        for i in range(n):
            t0 = time.perf_counter()
            worker.drive(p)
            dt = (time.perf_counter() - t0) * 1000.0
            if i >= 2:  # skip warmup
                times.append(dt)
            p = []
        return float(np.median(times)) if times else 0.0

    def drain():
        for aid in list(worker.active.keys()):
            worker.abort(aid)
        while worker.engine.has_unfinished_requests():
            worker.drive([])

    measurements = []

    # ---- Phase 1: pure decode at varying batch sizes ----
    print("Phase 1: pure decode")
    for ctx in [1024, 1536, 2047]:
        for n_req in [8, 16, 24, 32, 48, 64]:
            if n_req > max_active * 0.95:
                continue
            drain()
            # Add N decode requests.
            cmds = []
            for i in range(n_req):
                cmds.append({
                    "cmd": "add", "request_id": f"d_{ctx}_{i}",
                    "attempt_id": f"d_{ctx}_{i}",
                    "kind": BE, "phase": TPUT,
                    "prompt_len": ctx, "max_tokens": HUGE, "base_tokens": 0,
                })
            # Drive enough steps to get all into decode phase.
            for _ in range(ctx + 100):
                worker.drive(cmds)
                cmds = []
                if not worker.engine.has_unfinished_requests():
                    break
            actual = len(worker.active)
            if actual < n_req * 0.8:
                print(f"  SKIP decode ctx={ctx} n={n_req}: only {actual} active")
                continue
            # Measure.
            med = drive_n(20)
            measurements.append({
                "prefill_tokens": 0, "prefill_sum_ctx": 0,
                "decode_n": actual, "decode_sum_ctx": actual * ctx,
                "latency_ms": med,
            })
            print(f"  decode ctx={ctx:>4} n={actual:>3} → {med:.2f}ms")

    # ---- Phase 2: prefill at varying sizes (w/ fixed decode background) ----
    print("Phase 2: prefill + decode mixed")
    bg_decode = 32  # background decode population
    for prefill_tok in [512, 1024, 1536, 2047]:
        for n_pre in [1, 2]:
            # Only do single prefill for large sizes (to stay in budget).
            if prefill_tok > 1500 and n_pre > 1:
                continue
            drain()
            # Fill background decode.
            cmds = []
            for i in range(bg_decode):
                cmds.append({
                    "cmd": "add", "request_id": f"bg_{i}",
                    "attempt_id": f"bg_{i}",
                    "kind": BE, "phase": TPUT,
                    "prompt_len": 1024, "max_tokens": HUGE, "base_tokens": 0,
                })
            for _ in range(1200):
                worker.drive(cmds)
                cmds = []
                if not worker.engine.has_unfinished_requests():
                    break
            bg_actual = len(worker.active)
            if bg_actual < bg_decode * 0.7:
                print(f"  SKIP prefill tok={prefill_tok}: bg only {bg_actual}")
                continue

            # Warm up background.
            drive_n(10)

            # Inject prefill and measure.
            for pi in range(n_pre):
                drain()
                # Re-fill background.
                cmds = []
                for i in range(bg_decode):
                    cmds.append({
                        "cmd": "add", "request_id": f"bg2_{i}",
                        "attempt_id": f"bg2_{i}",
                        "kind": BE, "phase": TPUT,
                        "prompt_len": 1024, "max_tokens": HUGE, "base_tokens": 0,
                    })
                for _ in range(1200):
                    worker.drive(cmds)
                    cmds = []

                # Now add prefill and time the NEXT drive call.
                t0 = time.perf_counter()
                worker.drive([{
                    "cmd": "add", "request_id": f"p_{prefill_tok}_{pi}",
                    "attempt_id": f"p_{prefill_tok}_{pi}",
                    "kind": BE, "phase": TPUT,
                    "prompt_len": prefill_tok, "max_tokens": HUGE, "base_tokens": 0,
                }])
                dt = (time.perf_counter() - t0) * 1000.0

                # Read iteration log to get actual composition.
                logs = list(getattr(worker.engine, "iteration_logs", []))
                if logs:
                    last = logs[-1]
                    p_tok = float(last.get("prefill_tokens", prefill_tok))
                    p_ctx = float(last.get("prefill_sum_ctx", prefill_tok))
                    d_n = float(last.get("n_decode_reqs", 0))
                    d_ctx = float(last.get("decode_sum_ctx", 0))
                else:
                    p_tok, p_ctx = float(prefill_tok), float(prefill_tok)
                    d_n, d_ctx = float(bg_actual), float(bg_actual * 1024)

                measurements.append({
                    "prefill_tokens": p_tok, "prefill_sum_ctx": p_ctx,
                    "decode_n": d_n, "decode_sum_ctx": d_ctx,
                    "latency_ms": dt,
                })
                print(f"  prefill tok={prefill_tok:>4} n_pre={pi+1} "
                      f"ptok={p_tok:.0f} dn={d_n:.0f} → {dt:.2f}ms")

        # ---- Fit ----
        if len(measurements) < 6:
            raise RuntimeError(f"Too few: {len(measurements)}")

    x = np.array([[m["prefill_tokens"], m["prefill_sum_ctx"],
                   m["decode_n"], m["decode_sum_ctx"], 1.0]
                  for m in measurements], dtype=float)
    y = np.array([m["latency_ms"] for m in measurements], dtype=float)

    # Use non-negative least squares (scipy if available, otherwise clip).
    try:
        from scipy.optimize import nnls
        coef_feat, _ = nnls(x[:, :4], y)
        intercept = float(np.median(y - x[:, :4] @ coef_feat))
        coef = np.concatenate([coef_feat, [max(intercept, 0.0)]])
    except ImportError:
        coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        coef[:4] = np.maximum(coef[:4], 0.0)
        intercept = float(np.median(y - (x[:, :4] @ coef[:4])))
        coef[4] = max(intercept, 0.0)

    pred = x @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    mape = float(np.mean(np.abs(y - pred) / np.maximum(y, 1e-9)) * 100.0)

    print(f"\n=== Fitted model ===")
    print(f"  coef = {[float(v) for v in coef[:4]]}")
    print(f"  intercept = {float(coef[4]):.3f}")
    print(f"  R² = {r2:.4f}, MAPE = {mape:.1f}%")
    for m in measurements:
        feats = np.array([m["prefill_tokens"], m["prefill_sum_ctx"],
                          m["decode_n"], m["decode_sum_ctx"], 1.0])
        p = float(feats @ coef)
        e = m["latency_ms"] - p
        print(f"    actual={m['latency_ms']:6.1f} pred={p:6.1f} err={e:+5.1f}ms "
              f"ptok={m['prefill_tokens']:5.0f} dn={m['decode_n']:3.0f}")

    # Write JSON.
    identity = TimePredictorProfileIdentity.from_engine_configs(
        model_config=worker.engine.model_config,
        parallel_config=worker.engine.parallel_config,
    )
    num_layers = 28
    hf = getattr(worker.engine.model_config, "hf_config", None)
    if hf:
        for n in ("num_hidden_layers", "n_layer", "num_layers"):
            v = getattr(hf, n, None)
            if v: num_layers = int(v); break

    out_dir = Path("saturo/results/time_predictors")
    pp = time_predictor_cache_path(identity, out_dir)
    pp.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "time_predictor": {
            "model_type": "iteration_level",
            "description": f"7B direct-measure on {identity.gpu_name}",
            "features": ["prefill_tokens","prefill_sum_ctx","n_decode_reqs","decode_sum_ctx"],
            "coef": [float(v) for v in coef[:4]],
            "intercept": float(coef[4]),
            "r2": float(r2), "mape": mape, "n_samples": len(measurements),
            "profile_identity": identity.to_dict(),
            "profile_key": identity.digest,
            "num_layers": num_layers,
            "meta": {"total_model_layers": num_layers},
        },
        "kv_cache_block": {"total_gpu_blocks": total_blocks, "block_size": block_size},
    }
    pp.with_suffix(pp.suffix + ".tmp").write_text(json.dumps(payload, indent=2))
    pp.with_suffix(pp.suffix + ".tmp").replace(pp)
    print(f"\nWrote: {pp}")

    worker.shutdown()


if __name__ == "__main__":
    main()
