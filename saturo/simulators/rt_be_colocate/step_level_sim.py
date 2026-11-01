#!/usr/bin/env python3
"""Step-level RT-BE colocation simulator with full P/D lifecycles.

Unlike the event-driven prefill-only ColocateSimulator, this models the
vLLM engine at *step* granularity: every iteration shares one token budget
(``chunk_size`` = max_num_batched_tokens) between RT decode, RT prefill,
BE decode and BE prefill.  Both RT and BE requests are individual entities
with their own prompt_len / output_len, so BE end-to-end completion rate
(req/s) is a first-class output — not just prefill token throughput.

Step latency is an affine model calibrated from the BE-saturated chunk
benchmark (experiments/results/be_chunk_benchmark):

    step_ms = LAT_A + LAT_B * scheduled_tokens

Two scheduling policies:

``vanilla``
    vLLM default semantics.  Decodes first (running queue priority), then
    in-flight prefills continue (a 9k-token BE prefill keeps eating the
    whole budget for ~6 consecutive steps -> head-of-line blocking), then
    waiting requests admitted FCFS by arrival, subject to max_num_seqs.

``rt_priority``
    Chunk-boundary preemption.  Every step the budget is re-allocated:
    RT decode -> RT prefill (admission exempt from the seat cap) ->
    BE decode -> BE prefill.  An in-flight BE prefill simply gets no
    budget while RT work exists, so RT blocking is bounded by one step.

Example::

    python3 saturo/simulators/rt_be_colocate/step_level_sim.py \
        --rt-trace data/datasets/rt_workload_2.csv \
        --be-trace data/datasets/be_workload.csv \
        --chunk-size 1536 --target-rps 0.5,1,1.5,2,2.5,3 \
        --policies vanilla,rt_priority --window-s 300 \
        --output output/step_level_sim/step_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Step-latency calibration (BE-saturated full steps, be_chunk_benchmark)
# ---------------------------------------------------------------------------

_CALIB_POINTS = [(1024, 103.930), (2048, 163.141), (4096, 298.036)]


def _fit_latency(points) -> tuple:
    n = len(points)
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    b = (sum((x - mx) * (y - my) for x, y in points)
         / sum((x - mx) ** 2 for x, y in points))
    a = my - b * mx
    return a, b


LAT_A, LAT_B = _fit_latency(_CALIB_POINTS)


def build_latency_model(calibration_csv: Optional[str], chunk_size: int,
                        full_chunk_ms: Optional[float] = None) -> dict:
    """沿用三档完整 step 拟合；可将曲线整体缩放到指定满块耗时。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正数")
    points = list(_CALIB_POINTS)
    if calibration_csv:
        # 与原 Qwen 方法一致，仅使用 1024、2048、4096 三档。
        wanted = {x for x, _ in _CALIB_POINTS}
        with open(calibration_csv, newline="", encoding="utf-8") as stream:
            points = [(int(r["chunk_size"]), float(r["step_actual_ms_mean"]))
                      for r in csv.DictReader(stream)
                      if int(r["chunk_size"]) in wanted]
        if len(points) != len(wanted) or {x for x, _ in points} != wanted:
            raise ValueError("延迟 CSV 必须包含各一行 1024、2048、4096 标定点")
        points.sort()
    if any(not math.isfinite(y) or y <= 0 for _, y in points):
        raise ValueError("标定耗时必须为有限正数")
    a, b = _fit_latency(points)
    if a < 0 or b <= 0:
        raise ValueError("延迟拟合必须具有非负截距和正斜率")
    scale = 1.0
    if full_chunk_ms is not None:
        if not math.isfinite(full_chunk_ms) or full_chunk_ms <= 0:
            raise ValueError("full_chunk_ms 必须为有限正数")
        scale = full_chunk_ms / (a + b * chunk_size)
    return {
        "a_ms": a * scale,
        "b_ms_per_tok": b * scale,
        "calibration_points": points,
        "calibration_csv": calibration_csv,
        "scale": scale,
        "full_chunk_ms": (a + b * chunk_size) * scale,
    }


# ---------------------------------------------------------------------------
# Request entity
# ---------------------------------------------------------------------------


@dataclass
class Seq:
    kind: str                     # "rt" | "be"
    prompt_len: int
    output_len: int
    arrival_s: float
    prefill_done: int = 0
    dec_emitted: int = 0          # output tokens emitted (incl. first token)
    ttft_s: Optional[float] = None
    last_emit_s: Optional[float] = None
    tbt_s: List[float] = field(default_factory=list)
    finish_s: Optional[float] = None

    @property
    def phase(self) -> str:
        return "prefill" if self.prefill_done < self.prompt_len else "decode"

    @property
    def done(self) -> bool:
        return (self.prefill_done >= self.prompt_len
                and self.dec_emitted >= self.output_len)


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------


def load_rt(csv_path: str, target_rps: float, window_s: float) -> List[Seq]:
    rows = list(csv.DictReader(open(csv_path)))
    span = float(rows[-1]["arrival_offset_s"]) - float(rows[0]["arrival_offset_s"])
    orig_rps = len(rows) / span
    scale = orig_rps / target_rps
    out: List[Seq] = []
    for r in rows:
        t = float(r["arrival_offset_s"]) * scale
        if t > window_s:
            break
        out.append(Seq("rt", int(r["prompt_len"]), int(r["output_len"]), t))
    return out


def load_be_pool(csv_path: str) -> List[tuple]:
    rows = list(csv.DictReader(open(csv_path)))
    return [(int(r["prompt_len"]), int(r["output_len"])) for r in rows]


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def simulate(rt_reqs: List[Seq], be_pool: List[tuple], *, policy: str,
             chunk_size: int, window_s: float, max_num_seqs: int,
             be_inflight: int, overhead_ms: float,
             jitter_cv: float = 0.0, seed: int = 0,
             latency_a_ms: float = LAT_A,
             latency_b_ms_per_tok: float = LAT_B) -> dict:
    rng = random.Random(seed)
    now = 0.0
    rt_idx = 0
    be_next = 0
    waiting: List[Seq] = []       # FCFS by injection order
    running: List[Seq] = []
    done_rt: List[Seq] = []
    done_be: List[Seq] = []
    step_tokens_sum = 0
    step_count = 0
    be_prefill_tok = 0
    be_decode_tok = 0

    def inflight_be() -> int:
        return (sum(1 for s in waiting if s.kind == "be")
                + sum(1 for s in running if s.kind == "be"))

    while now < window_s:
        # ---- admit arrivals -------------------------------------------
        while rt_idx < len(rt_reqs) and rt_reqs[rt_idx].arrival_s <= now:
            waiting.append(rt_reqs[rt_idx])
            rt_idx += 1
        if be_pool and be_inflight > 0:
            while inflight_be() < be_inflight:
                p, o = be_pool[be_next % len(be_pool)]
                be_next += 1
                waiting.append(Seq("be", p, o, now))

        # ---- idle fast-forward ----------------------------------------
        if not running and not waiting:
            if rt_idx >= len(rt_reqs):
                break
            now = rt_reqs[rt_idx].arrival_s
            continue

        # ---- allocate the step budget ---------------------------------
        budget = chunk_size
        decode_grant: List[Seq] = []
        prefill_grant: List[tuple] = []   # (seq, tokens)

        def grant_decode(seqs):
            nonlocal budget
            for s in seqs:
                if budget <= 0:
                    break
                budget -= 1
                decode_grant.append(s)

        def grant_prefill(seqs):
            nonlocal budget
            for s in seqs:
                if budget <= 0:
                    break
                t = min(budget, s.prompt_len - s.prefill_done)
                budget -= t
                prefill_grant.append((s, t))

        def admit(kinds, seat_exempt=False):
            nonlocal budget
            i = 0
            while i < len(waiting) and budget > 0:
                s = waiting[i]
                if s.kind not in kinds:
                    i += 1
                    continue
                if not seat_exempt and len(running) >= max_num_seqs:
                    break
                waiting.pop(i)
                running.append(s)
                t = min(budget, s.prompt_len)
                budget -= t
                prefill_grant.append((s, t))

        if policy == "vanilla":
            # Real benchmark.py "vanilla" = vLLM priority scheduling:
            # running queue first (decodes, then in-flight prefills — a
            # 9k-token BE prefill keeps its budget until done: bounded
            # head-of-line blocking), then waiting by priority (RT=0
            # ahead of every waiting BE=10).
            grant_decode([s for s in running if s.phase == "decode"])
            grant_prefill([s for s in running if s.phase == "prefill"])
            admit(("rt",))
            admit(("be",))
        elif policy == "rt_priority":
            grant_decode([s for s in running
                          if s.kind == "rt" and s.phase == "decode"])
            grant_prefill([s for s in running
                           if s.kind == "rt" and s.phase == "prefill"])
            admit(("rt",), seat_exempt=True)
            grant_decode([s for s in running
                          if s.kind == "be" and s.phase == "decode"])
            grant_prefill([s for s in running
                           if s.kind == "be" and s.phase == "prefill"])
            admit(("be",))
        else:
            raise ValueError(f"unknown policy {policy}")

        step_tokens = chunk_size - budget
        if step_tokens <= 0:
            # nothing runnable this step (e.g. seats full of stalled seqs)
            if rt_idx < len(rt_reqs):
                now = rt_reqs[rt_idx].arrival_s
                continue
            break

        step_s = (latency_a_ms + latency_b_ms_per_tok * step_tokens
                  + overhead_ms) / 1000.0
        if jitter_cv > 0.0:
            # multiplicative log-normal jitter calibrated from the measured
            # step_actual p95/median ratio (~1.15 -> cv ~0.10)
            step_s *= rng.lognormvariate(0.0, jitter_cv)
        now += step_s
        step_tokens_sum += step_tokens
        step_count += 1

        # ---- apply results at step end --------------------------------
        for s, t in prefill_grant:
            s.prefill_done += t
            if s.kind == "be":
                be_prefill_tok += t
            if s.prefill_done >= s.prompt_len:
                # prefill completion emits the first output token
                s.ttft_s = now - s.arrival_s
                s.dec_emitted = 1
                s.last_emit_s = now
        for s in decode_grant:
            s.dec_emitted += 1
            if s.kind == "be":
                be_decode_tok += 1
            if s.last_emit_s is not None:
                s.tbt_s.append(now - s.last_emit_s)
            s.last_emit_s = now
        for s in list(running):
            if s.done:
                s.finish_s = now
                running.remove(s)
                (done_rt if s.kind == "rt" else done_be).append(s)

    # ------------------------------------------------------------------
    # Metrics (strict stop at window end, like the real benchmark)
    # ------------------------------------------------------------------
    injected = rt_reqs[:rt_idx]

    def pct(vals, q):
        if not vals:
            return 0.0
        v = sorted(vals)
        return v[min(len(v) - 1, max(0, int(math.ceil(q * len(v))) - 1))]

    ttfts = [s.ttft_s for s in injected if s.ttft_s is not None]
    ttft_ok = sum(1 for t in ttfts if t <= TTFT_SLO_S)
    joint = 0
    tbt_req_ok = 0
    for s in done_rt:
        t_ok = s.ttft_s is not None and s.ttft_s <= TTFT_SLO_S
        b_ok = all(b <= TBT_SLO_S for b in s.tbt_s)
        tbt_req_ok += 1 if b_ok else 0
        joint += 1 if (t_ok and b_ok) else 0
    all_tbt = [b for s in done_rt for b in s.tbt_s]

    n_inj = max(1, len(injected))
    return {
        "policy": policy,
        "chunk_size": chunk_size,
        "window_s": window_s,
        "rt_injected": len(injected),
        "rt_completed": len(done_rt),
        "joint_attainment_pct": 100.0 * joint / n_inj,
        "ttft_attainment_pct": 100.0 * ttft_ok / n_inj,
        "tbt_request_attainment_pct": 100.0 * tbt_req_ok / n_inj,
        "ttft_p50_ms": pct(ttfts, 0.50) * 1000,
        "ttft_p95_ms": pct(ttfts, 0.95) * 1000,
        "ttft_p99_ms": pct(ttfts, 0.99) * 1000,
        "tbt_mean_ms": (sum(all_tbt) / len(all_tbt) * 1000) if all_tbt else 0.0,
        "tbt_p99_ms": pct(all_tbt, 0.99) * 1000,
        "be_completed": len(done_be),
        "be_completed_req_s": len(done_be) / window_s,
        "be_e2e_p50_s": pct([s.finish_s - s.arrival_s for s in done_be], 0.5),
        "be_prefill_tok_s": be_prefill_tok / window_s,
        "be_decode_tok_s": be_decode_tok / window_s,
        "step_count": step_count,
        "step_tokens_mean": step_tokens_sum / max(1, step_count),
        "step_ms_mean": (latency_a_ms + latency_b_ms_per_tok
                         * step_tokens_sum / max(1, step_count)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

TTFT_SLO_S = 1.0
TBT_SLO_S = 0.150


def main() -> None:
    global TTFT_SLO_S, TBT_SLO_S
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--rt-trace", default="data/datasets/rt_workload_2.csv")
    p.add_argument("--be-trace", default="data/datasets/be_workload.csv")
    p.add_argument("--chunk-size", type=int, default=1536)
    p.add_argument("--target-rps", default="0.5,1,1.5,2,2.5,3")
    p.add_argument("--policies", default="vanilla,rt_priority")
    p.add_argument("--window-s", type=float, default=300.0)
    p.add_argument("--max-num-seqs", type=int, default=128)
    p.add_argument("--be-inflight", type=int, default=128)
    p.add_argument("--ttft-slo-ms", type=float, default=1000.0)
    p.add_argument("--tbt-slo-ms", type=float, default=150.0)
    p.add_argument("--step-overhead-ms", type=float, default=0.0)
    p.add_argument("--step-jitter-cv", type=float, default=0.0,
                   help="lognormal sigma for step-time jitter (0=off).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--latency-calibration-csv", default=None,
                   help="模型专属完整 step 标定 CSV；默认沿用 Qwen 三点。")
    p.add_argument("--full-chunk-ms", type=float, default=None,
                   help="整体缩放拟合曲线，使满 chunk 的基础耗时等于此值。")
    p.add_argument("--output", default="")
    p.add_argument("--summary-output", default="",
                   help="仅导出 rt_priority 的 RPS、TTFT 达成率、BE E2E req/s。")
    args = p.parse_args()
    try:
        latency_model = build_latency_model(
            args.latency_calibration_csv, args.chunk_size, args.full_chunk_ms)
    except (OSError, KeyError, ValueError) as exc:
        p.error(str(exc))
    latency_a = latency_model["a_ms"]
    latency_b = latency_model["b_ms_per_tok"]

    TTFT_SLO_S = args.ttft_slo_ms / 1000.0
    TBT_SLO_S = args.tbt_slo_ms / 1000.0

    be_pool = load_be_pool(args.be_trace)
    rps_list = [float(x) for x in args.target_rps.split(",") if x.strip()]
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]

    if args.summary_output and policies != ["rt_priority"]:
        p.error("--summary-output 要求 --policies rt_priority")

    print(f"step-latency model: step_ms = {latency_a:.2f} + {latency_b:.5f} * tokens "
          f"(chunk {args.chunk_size} -> "
          f"{latency_a + latency_b * args.chunk_size:.1f} ms)")

    rows = []
    for policy in policies:
        for rps in rps_list:
            rt = load_rt(args.rt_trace, rps, args.window_s)
            res = simulate(
                rt, be_pool, policy=policy, chunk_size=args.chunk_size,
                window_s=args.window_s, max_num_seqs=args.max_num_seqs,
                be_inflight=args.be_inflight,
                overhead_ms=args.step_overhead_ms,
                jitter_cv=args.step_jitter_cv, seed=args.seed,
                latency_a_ms=latency_a, latency_b_ms_per_tok=latency_b)
            res["target_rps"] = rps
            rows.append(res)
            print(f"  {policy:12s} rps={rps:4.1f}  "
                  f"joint={res['joint_attainment_pct']:5.1f}%  "
                  f"ttft={res['ttft_attainment_pct']:5.1f}%  "
                  f"tbt_req={res['tbt_request_attainment_pct']:5.1f}%  "
                  f"ttft_p95={res['ttft_p95_ms']:7.0f}ms  "
                  f"be={res['be_completed_req_s']:.3f} req/s  "
                  f"be_pf={res['be_prefill_tok_s']:7.0f} tok/s")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        fields = ["policy", "target_rps"] + [
            k for k in rows[0] if k not in ("policy", "target_rps")]
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        out.with_suffix(".json").write_text(json.dumps({
            "config": vars(args),
            "latency_model": latency_model,
            "rows": rows}, indent=2))
        print(f"\n[output] {out}")

    if args.summary_output:
        out = Path(args.summary_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["rt_rps", "rt_slo_attainment_pct", "be_e2e_req_s"])
            for row in rows:
                writer.writerow([row["target_rps"],
                                 f'{row["ttft_attainment_pct"]:.2f}',
                                 f'{row["be_completed_req_s"]:.3f}'])
        print(f"[summary] {out}")


if __name__ == "__main__":
    main()
