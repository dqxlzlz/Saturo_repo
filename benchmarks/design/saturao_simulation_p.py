
import argparse
import sys
from typing import Optional, Tuple

import pandas as pd

from vllm.engine.time_predictor import TimePredictor


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Saturo BE-Chunk-Size Simulation-driven Optimizer")

    # Trace / profile
    parser.add_argument("--profile", type=str, required=True,
                        help="Path to the RT trace CSV file")
    parser.add_argument("--json-profile", type=str,
                        default="profile_bench_results.json",
                        help="Path to the ML execution-time JSON profile")

    # Scaling / window
    parser.add_argument("--rps", type=float, default=None,
                        help="Target RT requests per second (per instance, "
                             "used to scale trace intervals)")
    parser.add_argument("--duration-time", type=float, default=60.0,
                        help="Simulation time window in seconds")
    parser.add_argument("--num-instances", type=int, default=1,
                        help="Number of P-pool instances; arrival intervals "
                             "are multiplied by this factor")

    # SLO & model
    parser.add_argument("--slo", type=float, default=1.0,
                        help="TTFT SLO constraint in seconds")
    parser.add_argument("--num-layers", type=int, default=32,
                        help="Number of layers for TimePredictor")

    # Memory / BE request shape
    parser.add_argument("--free-gpu-blocks", type=int, default=7200,
                        help="Free GPU blocks available for BE")
    parser.add_argument("--block-size", type=int, default=16,
                        help="Tokens per block")
    parser.add_argument("--avg-be-length", type=int, default=8000,
                        help="Average BE prefill input length (tokens)")
    parser.add_argument("--avg-be-output", type=int, default=100,
                        help="Average BE decode output length (tokens)")

    # Chunk size search space
    parser.add_argument("--c-min", type=int, default=1,
                        help="Minimum BE chunk size to explore")
    parser.add_argument("--c-max", type=int, default=8192,
                        help="Maximum BE chunk size (upper bound driven by "
                             "compute saturation)")
    parser.add_argument("--c-step", type=int, default=16,
                        help="Fallback step size when slack-driven expansion "
                             "fails to grow c (avoid dead loop)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------
def load_and_preprocess_trace(filepath: str,
                              target_rps: float,
                              num_instances: int,
                              duration_time: float) -> pd.DataFrame:
    """Load RT trace, scale the per-instance arrival intervals and clip to
    the simulation window.

    The trace must contain at least ``TIMESTAMP`` and ``forward_time`` columns.
    """
    df = pd.read_csv(filepath)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    df = df.sort_values('TIMESTAMP').reset_index(drop=True)

    t0 = df['TIMESTAMP'].iloc[0]
    df['arrival_time'] = (df['TIMESTAMP'] - t0).dt.total_seconds()
    print(f"origin arrivival time: {df['arrival_time'][:10]}")
    # (a) rescale to target global RPS if provided
    if target_rps is not None and len(df) > 1:
        total_time = df['arrival_time'].iloc[-1]
        if total_time > 0:
            original_rps = len(df) / total_time
            print(f"original_rps:{original_rps}")
            scale_factor = original_rps / target_rps
            df['arrival_time'] = df['arrival_time'] * scale_factor

    print(f"after deal arrivival time: {df['arrival_time'][:10]}")


    df = df[df['arrival_time'] <= duration_time].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# BE chunk primitives
# ---------------------------------------------------------------------------
def get_c_breakdown(c: int, args) -> Tuple[int, int]:
    """Split a chunk budget ``c`` into (c_d, c_p) where c_d is the decode
    batch size saturating memory and c_p is prefill chunk size using the
    residual compute."""
    available_context_tokens = args.free_gpu_blocks * args.block_size
    max_decode_num_req = available_context_tokens // max(1, args.avg_be_length)
    c_d = min(c, int(max_decode_num_req))
    c_p = max(0, c - c_d)
    return c_d, c_p


def predict_t_be_seconds(c: int, args, predictor: TimePredictor) -> float:
    """Predict BE chunk execution time (seconds) for a given ``c``."""
    if c <= 0:
        return 0.0
    c_d, c_p = get_c_breakdown(c, args)
    if c_d == 0 and c_p == 0:
        return 0.0

    # Decode: each request has query_len=1, context_len=avg_be_length
    decode_reqs = [(1, args.avg_be_length)] * c_d if c_d > 0 else []
    # Prefill: single pseudo request with query_len=c_p, context_len=c_p
    prefill_reqs = [(c_p, c_p)] if c_p > 0 else []

    return predictor.predict_forward_ms(prefill_reqs, decode_reqs) / 1000.0


def find_max_c_for_budget(budget_sec: float,
                          args,
                          predictor: TimePredictor,
                          lo: Optional[int] = None,
                          hi: Optional[int] = None) -> int:
    """Binary search the largest chunk size whose predicted BE time does not
    exceed ``budget_sec`` seconds."""
    if budget_sec <= 0:
        return 0
    left = args.c_min if lo is None else max(args.c_min, lo)
    right = args.c_max if hi is None else min(args.c_max, hi)
    if right < left:
        return 0
    best = 0
    while left <= right:
        mid = (left + right) // 2
        t_be = predict_t_be_seconds(mid, args, predictor)
        if t_be <= budget_sec:
            best = mid
            left = mid + 1
        else:
            right = mid - 1
    return best


# ---------------------------------------------------------------------------
# Step 1: Pure-RT slack
# ---------------------------------------------------------------------------
def compute_pure_rt_slacks(trace_df: pd.DataFrame, slo: float):
    """Sequentially compute each RT request's slack under a *pure* RT pipeline
    (no BE interference). Returns a list of slacks aligned with trace order."""
    slacks = []
    current_time = 0.0
    for _, row in trace_df.iterrows():
        t_arr = float(row['arrival_time'])
        t_rt = float(row['forward_time'])

        start_time = max(current_time, t_arr)
        t_queue = start_time - t_arr
        slack = slo - t_queue - t_rt
        slacks.append(slack)

        current_time = start_time + t_rt
    return slacks


# ---------------------------------------------------------------------------
# Step 3 helper: simulate the mixed pipeline under a given chunk size c
# ---------------------------------------------------------------------------
def simulate_under_chunk(c: int,
                         trace_df: pd.DataFrame,
                         args,
                         predictor: TimePredictor):
    """Simulate the mixed RT + BE pipeline under a fixed chunk size ``c``.

    The schedule follows the paper description:

      - If no pending RT request, the instance immediately launches a BE
        chunk of size ``c`` (decode saturating memory, prefill using the
        residual).
      - When an RT request arrives in the middle of a BE chunk, the
        scheduler decides whether to preempt that chunk based on
        ``T_rem + T_queue + T_RT <= SLO``. If the inequality holds the
        chunk finishes (BE work is accounted as useful). Otherwise the
        ongoing BE chunk is preempted (its tokens are discarded).

    Returns
    -------
    dict with:
        tput_req_s           : min(prefill_req/s, decode_req/s) over the window
        prefill_tput         : prefill request throughput (req/s)
        decode_tput          : decode request throughput (req/s)
        preemptions          : number of BE preemptions happened
        finished_be_chunks   : number of BE chunks that completed successfully
        t_be                 : predicted BE chunk execution time (s)
        slacks               : per-RT residual slack after this chunk size
                               (SLO - T_queue_actual - T_RT)
    """
    c_d, c_p = get_c_breakdown(c, args)
    t_be = predict_t_be_seconds(c, args, predictor)

    if t_be <= 0.0:
        return {
            "tput_req_s": 0.0,
            "prefill_tput": 0.0,
            "decode_tput": 0.0,
            "preemptions": 0,
            "finished_be_chunks": 0,
            "t_be": 0.0,
            "slacks": [],
        }

    current_time = 0.0
    finished_chunks = 0
    preemptions = 0
    total_prefill_tokens = 0
    total_decode_tokens = 0
    slacks = []

    for _, row in trace_df.iterrows():
        t_arr = float(row['arrival_time'])
        t_rt = float(row['forward_time'])

        # ------------------------------------------------------------------
        # (1) Try to greedily launch BE chunks before this RT arrives.
        #     Before EACH launch we verify the SLO preemption condition
        #     T_rem + T_queue + T_RT <= SLO against THIS upcoming RT.
        #     T_queue = 0 here because the RT queue is empty during the
        #     idle gap; T_rem is how much of the BE chunk would still be
        #     running by the time the RT arrives.
        # ------------------------------------------------------------------
        while current_time < t_arr:
            be_finish = current_time + t_be
            # How much of this BE chunk overlaps with the RT's service?
            t_rem = max(0.0, be_finish - t_arr)
            # SLO feasibility check for the about-to-be-launched BE chunk.
            if t_rem + t_rt > args.slo:
                # Launching this BE would violate the RT SLO -> don't
                # launch it. This is a "prevented preemption": no tokens
                # wasted, but we also don't keep packing BE.
                break
            # Safe to launch: account tokens and advance the clock.
            finished_chunks += 1
            total_prefill_tokens += c_p
            total_decode_tokens += c_d
            current_time = be_finish
            # If this BE overlaps RT arrival (finished "on top of" the
            # RT), the RT still has to wait `t_rem`; stop packing further.
            if be_finish > t_arr:
                break

        # ------------------------------------------------------------------
        # (2) Handle the state when the RT arrives.
        #     Case A (current_time <= t_arr): we either idled (prevented
        #         preemption in step 1) or a BE finished exactly at t_arr.
        #         Nothing to do here.
        #     Case B (current_time  > t_arr): a BE chunk was mid-flight
        #         when the RT arrived. If the remaining BE time still
        #         satisfies T_rem + T_RT <= SLO we let it finish (already
        #         accounted in step 1). Otherwise we preempt: roll back
        #         the BE chunk's tokens and reset the clock to t_arr.
        # ------------------------------------------------------------------
        if current_time > t_arr:
            t_rem_actual = current_time - t_arr
            if t_rem_actual + t_rt > args.slo:
                # Abort the overlapping BE chunk; its tokens are wasted.
                preemptions += 1
                finished_chunks -= 1
                total_prefill_tokens -= c_p
                total_decode_tokens -= c_d
                current_time = t_arr
            # else: chunk is allowed to finish; current_time stays.

        # ------------------------------------------------------------------
        # (3) Serve the RT request. Any residual queueing here comes from
        #     back-to-back RT overload (the previous RT's tail), NOT from
        #     BE chunks, so it should not be counted as a BE preemption.
        # ------------------------------------------------------------------
        start_time = max(current_time, t_arr)
        t_queue = start_time - t_arr
        slacks.append(args.slo - t_queue - t_rt)
        current_time = start_time + t_rt

    # (4) After the last RT request, keep packing BE until the simulation
    #     window is exhausted.
    while current_time + t_be <= args.duration_time:
        finished_chunks += 1
        total_prefill_tokens += c_p
        total_decode_tokens += c_d
        current_time += t_be

    window = args.duration_time if args.duration_time > 0 else 1.0
    prefill_req = total_prefill_tokens / max(1, args.avg_be_length)
    decode_req = total_decode_tokens / max(1, args.avg_be_output)
    prefill_tput = prefill_req / window
    decode_tput = decode_req / window

    return {
        "tput_req_s": min(prefill_tput, decode_tput),
        "prefill_tput": prefill_tput,
        "decode_tput": decode_tput,
        "preemptions": preemptions,
        "finished_be_chunks": finished_chunks,
        "t_be": t_be,
        "slacks": slacks,
    }


# ---------------------------------------------------------------------------
# Main optimization loop
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    trace_df = load_and_preprocess_trace(args.profile,
                                         args.rps,
                                         args.num_instances,
                                         args.duration_time)
    print(f"[Trace] Loaded {len(trace_df)} RT requests within "
          f"{args.duration_time}s window "
          f"(rps={args.rps}, num_instances={args.num_instances}).")

    if len(trace_df) == 0:
        print("[Error] Empty trace after filtering. Abort.")
        sys.exit(1)

    predictor = TimePredictor(args.json_profile, num_layers=args.num_layers)

    # ------------------------------------------------------------------
    # Step 1: per-request pure-RT slack
    # ------------------------------------------------------------------
    pure_slacks = compute_pure_rt_slacks(trace_df, args.slo)
    for idx,slack in enumerate(pure_slacks):
        print(f"idx and slack: {idx, slack}")
    
    # print(f"trace df arrival: {trace_df['arrival_time']}")
    # print(f"pure_slacks:{pure_slacks}")

    min_pure_slack = min(pure_slacks) if pure_slacks else 0.0
    print(f"\n[Step 1] Pure-RT min slack = {min_pure_slack*1000:.2f} ms, "
          f"mean slack = {sum(pure_slacks)/len(pure_slacks)*1000:.2f} ms")

    if min_pure_slack <= 0.0:
        print("[Step 1] Warning: min pure-RT slack is <= 0, meaning the RT "
              "pipeline itself already violates SLO. Falling back to c_min.")
        c0 = args.c_min
    else:
        # ------------------------------------------------------------------
        # Step 2: safe (0-preemption) chunk size via binary search on the
        # tightest pure-RT slack
        # ------------------------------------------------------------------
        c0 = find_max_c_for_budget(min_pure_slack, args, predictor)
        c0 = max(args.c_min, c0)
    print(f"[Step 2] Zero-preemption safe chunk size c0 = {c0} "
          f"(t_be = {predict_t_be_seconds(c0, args, predictor)*1000:.2f} ms)")

    # ------------------------------------------------------------------
    # Step 3: slack-driven iterative expansion
    # ------------------------------------------------------------------
    print("\n[Step 3] Slack-driven iterative expansion")
    header = (f"{'c':<8} | {'t_be(ms)':<10} | {'preempt':<8} | "
              f"{'pref(r/s)':<10} | {'dec(r/s)':<10} | "
              f"{'tput(r/s)':<10} | {'min_slack(ms)':<14}")
    print(header)
    print("-" * len(header))

    best = {
        "c": c0,
        "tput": -1.0,
        "preemptions": 0,
        "prefill_tput": 0.0,
        "decode_tput": 0.0,
        "t_be": 0.0,
    }

    c = c0
    visited = set()
    # Floor of slack we still treat as useful; tighter than 1e-4 is noise.
    EPS = 1e-4

    while c <= args.c_max and c not in visited:
        visited.add(c)
        res = simulate_under_chunk(c, trace_df, args, predictor)

        min_slack_ms = (min(res["slacks"]) * 1000.0) if res["slacks"] else 0.0
        print(f"{c:<8} | {res['t_be']*1000:<10.2f} | "
              f"{res['preemptions']:<8} | "
              f"{res['prefill_tput']:<10.4f} | "
              f"{res['decode_tput']:<10.4f} | "
              f"{res['tput_req_s']:<10.4f} | "
              f"{min_slack_ms:<14.2f}")

        if res["tput_req_s"] > best["tput"]:
            best.update({
                "c": c,
                "tput": res["tput_req_s"],
                "preemptions": res["preemptions"],
                "prefill_tput": res["prefill_tput"],
                "decode_tput": res["decode_tput"],
                "t_be": res["t_be"],
            })

        # Decide the next c to probe: extend t_be by the tightest residual
        # positive slack so the BE chunk grows without breaking SLO.
        positive = [s for s in res["slacks"] if s > EPS]
        if not positive:
            # Slack fully exhausted under current c -> further growth would
            # inevitably cause preemptions, stop exploring.
            break

        target_budget = res["t_be"] + min(positive)
        next_c = find_max_c_for_budget(target_budget, args, predictor,
                                       lo=c + 1)
        if next_c <= c:
            # Predictor granularity too coarse to bump c via slack alone;
            # fall back to a fixed step to avoid a dead loop.
            next_c = c + args.c_step
        c = next_c

    print("-" * len(header))

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("\n[Result] Saturo Optimal BE Chunk")
    print(f"  c*                 : {best['c']}")
    c_d, c_p = get_c_breakdown(best['c'], args)
    print(f"  c_d (decode batch) : {c_d}")
    print(f"  c_p (prefill size) : {c_p}")
    print(f"  t_be               : {best['t_be']*1000:.2f} ms")
    print(f"  preemptions        : {best['preemptions']}")
    print(f"  prefill throughput : {best['prefill_tput']:.4f} req/s")
    print(f"  decode  throughput : {best['decode_tput']:.4f} req/s")
    print(f"  BE saturation tput : {best['tput']:.4f} req/s")


if __name__ == "__main__":
    main()
