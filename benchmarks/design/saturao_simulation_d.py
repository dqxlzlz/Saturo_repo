
"""Saturo D-pool Saturation Simulator.

For each D-pool instance, Saturo co-locates Best-Effort (BE) requests with
real-time (RT) decode requests, following a *decode-prioritized* policy:

  Per iteration:
    1. Admit ALL active RT-Decode requests (highest priority).
    2. Pack BE-Decode requests, capped at a memory watermark
       (default 90% of free GPU memory) -> Proactive Cache Swapping.
    3. Append the largest BE-Prefill chunk whose predicted iteration time
       does NOT exceed the target iteration time T^target.

Target iteration time is jointly bounded by
    * the per-token TBT SLO        (T_SLO), and
    * the long-term RT decode service-rate constraint
        T^rate = C^RT / (lambda * L_out)
  i.e.   T^target = min(T_SLO, T^rate)

Given T^target, the simulator replays decode iterations across a window and
reports the saturated BE throughput (BE-Decode req/s and BE-Prefill req/s).

CLI conventions follow ``saturao_simulation_p.py``.
"""

import argparse
import sys
from typing import List, Optional, Tuple

import pandas as pd

from vllm.engine.time_predictor import TimePredictor


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Saturo D-pool Saturation Simulator")

    # Trace / profile
    parser.add_argument("--profile", type=str, required=True,
                        help="Path to the RT-Decode trace CSV file. Must "
                             "contain TIMESTAMP and output_len columns "
                             "(context_len/prompt_len optional).")
    parser.add_argument("--json-profile", type=str,
                        default="profile_bench_results.json",
                        help="Path to the ML execution-time JSON profile")

    # Scaling / window
    parser.add_argument("--rps", type=float, default=None,
                        help="Target RT requests per second (per instance, "
                             "used to scale trace intervals). If omitted, "
                             "the trace's native RPS is used.")
    parser.add_argument("--duration-time", type=float, default=60.0,
                        help="Simulation time window in seconds")
    parser.add_argument("--num-instances", type=int, default=1,
                        help="Number of D-pool instances; arrival intervals "
                             "are multiplied by this factor")

    # SLO & model
    parser.add_argument("--slo", type=float, default=0.1,
                        help="TBT (per-token) SLO constraint in seconds. "
                             "Caps T^target.")
    parser.add_argument("--num-layers", type=int, default=32,
                        help="Number of layers for TimePredictor")

    # Memory / RT shape
    parser.add_argument("--free-gpu-blocks", type=int, default=7200,
                        help="Free GPU blocks available for BE+RT decode "
                             "KV cache (per instance)")
    parser.add_argument("--block-size", type=int, default=16,
                        help="Tokens per KV block")
    parser.add_argument("--mem-watermark", type=float, default=0.9,
                        help="Memory occupancy cap when packing BE-Decode "
                             "(reserves a buffer for RT bursts to avoid "
                             "swap-induced admission delay).")

    # RT-Decode workload shape (used both for trace fallback and C^RT)
    parser.add_argument("--avg-rt-context", type=int, default=2000,
                        help="Average RT-Decode context length (tokens). "
                             "Used to estimate per-request KV footprint "
                             "when the trace lacks per-request context_len.")
    parser.add_argument("--avg-rt-output", type=int, default=128,
                        help="Average RT-Decode output length L_out (tokens). "
                             "Used in T^rate when trace lacks output_len.")

    # BE workload shape
    parser.add_argument("--avg-be-context", type=int, default=2000,
                        help="Average BE-Decode context length (tokens). "
                             "Used for the per-decode KV footprint and "
                             "BE-Decode latency prediction.")
    parser.add_argument("--avg-be-prompt", type=int, default=8000,
                        help="Average BE-Prefill prompt length (tokens). "
                             "Used to convert BE-Prefill tokens into "
                             "request-level throughput.")
    parser.add_argument("--avg-be-output", type=int, default=100,
                        help="Average BE-Decode output length (tokens). "
                             "Used to convert BE-Decode iterations into "
                             "request-level throughput.")

    # BE prefill chunk search space
    parser.add_argument("--c-min", type=int, default=0,
                        help="Minimum BE-Prefill chunk size to explore "
                             "(0 means: it is OK to skip prefill entirely "
                             "when the iteration is already saturated).")
    parser.add_argument("--c-max", type=int, default=8192,
                        help="Maximum BE-Prefill chunk size")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------
def load_and_preprocess_trace(filepath: str,
                              target_rps: Optional[float],
                              num_instances: int,
                              duration_time: float,
                              avg_rt_output: int,
                              avg_rt_context: int) -> pd.DataFrame:
    """Load RT-Decode trace and produce columns:
        arrival_time   : seconds since the first arrival
        output_len     : remaining decode tokens for the request
        context_len    : initial context length (used for KV-footprint est.)

    The CSV must contain ``TIMESTAMP``. If ``output_len``/``context_len`` are
    missing, the fixed averages from CLI are used instead.
    """
    df = pd.read_csv(filepath)

    if 'TIMESTAMP' in df.columns:
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
        df = df.sort_values('TIMESTAMP').reset_index(drop=True)
        t0 = df['TIMESTAMP'].iloc[0]
        df['arrival_time'] = (df['TIMESTAMP'] - t0).dt.total_seconds()
    else:
        # Fallback: synthesize uniform arrivals using target_rps.
        rate = target_rps if target_rps and target_rps > 0 else 1.0
        df = df.reset_index(drop=True)
        df['arrival_time'] = df.index.to_series().astype(float) / rate

    print(f"[Trace] origin arrival_time head: "
          f"{df['arrival_time'].head(5).tolist()}")

    # Rescale to the requested per-instance RPS.
    if target_rps is not None and len(df) > 1:
        total_time = df['arrival_time'].iloc[-1]
        if total_time > 0:
            original_rps = len(df) / total_time
            print(f"[Trace] original_rps = {original_rps:.4f} req/s")
            scale_factor = original_rps / target_rps
            df['arrival_time'] = df['arrival_time'] * scale_factor

    # Multi-instance dilation: spread arrivals across instances.
    if num_instances > 1:
        df['arrival_time'] = df['arrival_time'] * float(num_instances)

    print(f"[Trace] scaled arrival_time head: "
          f"{df['arrival_time'].head(5).tolist()}")

    # Fill workload-shape columns.
    if 'output_len' not in df.columns:
        df['output_len'] = avg_rt_output
    df['output_len'] = df['output_len'].fillna(avg_rt_output).astype(int)

    if 'context_len' in df.columns:
        df['context_len'] = df['context_len'].fillna(avg_rt_context).astype(int)
    elif 'prompt_len' in df.columns:
        df['context_len'] = df['prompt_len'].fillna(avg_rt_context).astype(int)
    else:
        df['context_len'] = avg_rt_context

    # Clip to simulation window.
    df = df[df['arrival_time'] <= duration_time].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Capacity / target-time helpers
# ---------------------------------------------------------------------------
def compute_kv_blocks_per_request(context_len: int, block_size: int) -> int:
    """KV blocks required for ONE request whose KV cache covers context_len
    tokens."""
    if context_len <= 0:
        return 0
    return (context_len + block_size - 1) // block_size


def compute_rt_capacity(args, avg_rt_context: int) -> int:
    """C^RT = floor(M / m_KV).

    M is the GPU memory budget for KV cache (expressed in blocks here);
    m_KV is the per-RT-decode KV footprint in blocks.
    """
    blocks_per_rt = compute_kv_blocks_per_request(avg_rt_context,
                                                  args.block_size)
    if blocks_per_rt <= 0:
        return args.free_gpu_blocks
    return max(1, args.free_gpu_blocks // blocks_per_rt)


def compute_target_iteration_time(args,
                                  trace_df: pd.DataFrame,
                                  predictor: TimePredictor) -> Tuple[float, float, int]:
    """Compute T^target = min(T_SLO, T^rate).

    Returns (T_target, T_rate, C^RT).
    """
    # Per-instance arrival rate (rps) measured from the (already-scaled) trace.
    if len(trace_df) > 1:
        span = max(1e-6, float(trace_df['arrival_time'].iloc[-1]
                               - trace_df['arrival_time'].iloc[0]))
        lam = len(trace_df) / span
    else:
        lam = args.rps if args.rps else 0.0

    avg_out = (float(trace_df['output_len'].mean())
               if 'output_len' in trace_df.columns and len(trace_df) > 0
               else float(args.avg_rt_output))
    avg_ctx = (float(trace_df['context_len'].mean())
               if 'context_len' in trace_df.columns and len(trace_df) > 0
               else float(args.avg_rt_context))

    c_rt = compute_rt_capacity(args, int(avg_ctx))

    decode_token_rate = max(1e-6, lam * avg_out)
    t_rate = c_rt / decode_token_rate
    t_target = min(args.slo, t_rate)

    print(f"[Capacity] lambda = {lam:.4f} req/s, "
          f"L_out_avg = {avg_out:.1f} tok, "
          f"L_ctx_avg = {avg_ctx:.1f} tok")
    print(f"[Capacity] C^RT = {c_rt} resident RT-decode reqs")
    print(f"[Capacity] decode-token rate r = {decode_token_rate:.4f} tok/s")
    print(f"[Capacity] T^rate = {t_rate*1000:.2f} ms, "
          f"T_SLO = {args.slo*1000:.2f} ms "
          f"-> T^target = {t_target*1000:.2f} ms")
    return t_target, t_rate, c_rt


# ---------------------------------------------------------------------------
# Iteration-time prediction
# ---------------------------------------------------------------------------
def predict_iteration_ms(active_rt: List[Tuple[int, int]],
                         be_decode_count: int,
                         be_decode_ctx: int,
                         be_prefill_chunk: int,
                         predictor: TimePredictor) -> float:
    """Predict the latency (ms) of ONE decode iteration that contains:
        * len(active_rt)         RT-Decode reqs   (query=1, context=ctx_i)
        * be_decode_count        BE-Decode reqs   (query=1, context=be_decode_ctx)
        * be_prefill_chunk tokens of BE-Prefill   (query=c, context=c)
    """
    decode_reqs: List[Tuple[int, int]] = []
    decode_reqs.extend(active_rt)
    if be_decode_count > 0:
        decode_reqs.extend([(1, be_decode_ctx)] * be_decode_count)

    prefill_reqs: List[Tuple[int, int]] = []
    if be_prefill_chunk > 0:
        prefill_reqs.append((be_prefill_chunk, be_prefill_chunk))

    if not decode_reqs and not prefill_reqs:
        return 0.0
    return predictor.predict_forward_ms(prefill_reqs, decode_reqs)


def find_max_be_prefill_chunk(budget_sec: float,
                              active_rt: List[Tuple[int, int]],
                              be_decode_count: int,
                              args,
                              predictor: TimePredictor) -> Tuple[int, float]:
    """Binary-search the largest BE-Prefill chunk size whose predicted
    iteration latency stays within ``budget_sec``.

    Returns (chunk_size, predicted_iter_seconds).
    """
    # Latency of an iteration that already contains RT + BE-Decode but no
    # BE-Prefill. This is the lower bound; if it already exceeds the budget
    # we cannot afford any BE-Prefill.
    base_ms = predict_iteration_ms(active_rt, be_decode_count,
                                   args.avg_be_context, 0, predictor)
    base_sec = base_ms / 1000.0
    if base_sec > budget_sec:
        return 0, base_sec

    left = max(0, args.c_min)
    right = max(left, args.c_max)
    best_c = 0
    best_t = base_sec
    while left <= right:
        mid = (left + right) // 2
        t_ms = predict_iteration_ms(active_rt, be_decode_count,
                                    args.avg_be_context, mid, predictor)
        t_sec = t_ms / 1000.0
        if t_sec <= budget_sec:
            best_c = mid
            best_t = t_sec
            left = mid + 1
        else:
            right = mid - 1
    return best_c, best_t


# ---------------------------------------------------------------------------
# RT request bookkeeping
# ---------------------------------------------------------------------------
class RTRequest:
    """Active RT-Decode request resident in the instance."""
    __slots__ = ("arrival", "remaining", "context_len", "kv_blocks")

    def __init__(self, arrival: float, output_len: int,
                 context_len: int, block_size: int):
        self.arrival = arrival
        self.remaining = max(1, int(output_len))
        self.context_len = max(1, int(context_len))
        self.kv_blocks = compute_kv_blocks_per_request(self.context_len,
                                                       block_size)

    def step(self) -> None:
        self.remaining -= 1
        # Each decode step adds one token to context, optionally crossing a
        # block boundary; we keep kv_blocks roughly in sync.
        self.context_len += 1


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------
def simulate_d_pool(args,
                    trace_df: pd.DataFrame,
                    predictor: TimePredictor,
                    t_target: float):
    """Replay decode iterations across the simulation window.

    At every iteration we:
        1. Admit any RT request whose arrival_time has been reached.
        2. Compute remaining KV-block budget AFTER RT (capped by the memory
           watermark) -> determines how many BE-Decode requests we can keep.
        3. Search for the largest BE-Prefill chunk whose predicted iteration
           time fits within ``t_target``.
        4. Advance the clock by the predicted iteration time, decrement the
           remaining output length of every active RT request, retire
           finished ones.
    """
    block_size = args.block_size
    total_blocks = args.free_gpu_blocks
    watermark_blocks = int(total_blocks * args.mem_watermark)

    blocks_per_be_decode = compute_kv_blocks_per_request(args.avg_be_context,
                                                         block_size)
    blocks_per_be_decode = max(1, blocks_per_be_decode)

    # Sort arrivals.
    arrivals = trace_df.sort_values('arrival_time').reset_index(drop=True)
    arrival_times = arrivals['arrival_time'].tolist()
    arrival_outs = arrivals['output_len'].tolist()
    arrival_ctxs = arrivals['context_len'].tolist()

    next_arr_idx = 0
    active: List[RTRequest] = []

    current_time = 0.0
    iters = 0
    rt_iter_tokens = 0          # RT decode tokens served (== sum of iters per RT)
    be_decode_iter_tokens = 0   # BE decode tokens served
    be_prefill_tokens = 0       # BE prefill tokens served
    slo_violations = 0
    finished_rt = 0

    # Track tail iteration latencies for diagnostics.
    iter_latencies: List[float] = []

    while current_time < args.duration_time:
        # ---- (1) admit new RT arrivals up to current_time ----
        while (next_arr_idx < len(arrival_times)
               and arrival_times[next_arr_idx] <= current_time):
            active.append(RTRequest(arrival_times[next_arr_idx],
                                    arrival_outs[next_arr_idx],
                                    arrival_ctxs[next_arr_idx],
                                    block_size))
            next_arr_idx += 1

        # If no work to do, jump to the next RT arrival (idle gap).
        if not active:
            if next_arr_idx >= len(arrival_times):
                break
            # Optional: still fire BE-only iterations to harvest the gap.
            t_idle_end = min(arrival_times[next_arr_idx], args.duration_time)
            be_decode_count = min(
                watermark_blocks // blocks_per_be_decode,
                # No real cap from RT memory either, but keep it sane.
                args.free_gpu_blocks // blocks_per_be_decode,
            )
            if be_decode_count <= 0 and args.c_min <= 0:
                current_time = t_idle_end
                continue
            chunk, t_iter = find_max_be_prefill_chunk(
                t_target, [], be_decode_count, args, predictor)
            if t_iter <= 0.0:
                current_time = t_idle_end
                continue
            # Run as many BE-only iterations as fit in this idle gap.
            n_iters = max(0, int((t_idle_end - current_time) // t_iter))
            iters += n_iters
            be_decode_iter_tokens += n_iters * be_decode_count
            be_prefill_tokens += n_iters * chunk
            current_time += n_iters * t_iter
            iter_latencies.extend([t_iter] * n_iters)
            # Tick clock to t_idle_end if we still have residual gap.
            if current_time < t_idle_end:
                current_time = t_idle_end
            continue

        # ---- (2) RT memory footprint & BE-Decode capacity ----
        rt_blocks = sum(r.kv_blocks for r in active)
        free_for_be = max(0, watermark_blocks - rt_blocks)
        be_decode_count = free_for_be // blocks_per_be_decode

        # ---- (3) BE-Prefill chunk fitting T^target ----
        active_rt_tuples = [(1, r.context_len) for r in active]
        chunk, t_iter = find_max_be_prefill_chunk(
            t_target, active_rt_tuples, be_decode_count, args, predictor)

        if t_iter <= 0.0:
            # Predictor returned 0 (no-op iteration); guard against infinite loop.
            t_iter = max(t_target, 1e-3)

        # ---- (4) advance state ----
        # SLO accounting: each RT in this iteration experiences t_iter as TBT.
        if t_iter > args.slo + 1e-9:
            slo_violations += len(active)

        iters += 1
        iter_latencies.append(t_iter)
        rt_iter_tokens += len(active)
        be_decode_iter_tokens += be_decode_count
        be_prefill_tokens += chunk
        current_time += t_iter

        # Decrement remaining output lengths; retire finished RTs.
        survivors: List[RTRequest] = []
        for r in active:
            r.step()
            if r.remaining > 0:
                survivors.append(r)
            else:
                finished_rt += 1
        active = survivors

    window = args.duration_time if args.duration_time > 0 else 1.0
    be_decode_req = be_decode_iter_tokens / max(1, args.avg_be_output)
    be_prefill_req = be_prefill_tokens / max(1, args.avg_be_prompt)
    be_decode_tput = be_decode_req / window
    be_prefill_tput = be_prefill_req / window
    rt_decode_tput = finished_rt / window  # finished RT requests per second

    if iter_latencies:
        avg_iter_ms = sum(iter_latencies) / len(iter_latencies) * 1000.0
        max_iter_ms = max(iter_latencies) * 1000.0
    else:
        avg_iter_ms = 0.0
        max_iter_ms = 0.0

    return {
        "iters": iters,
        "avg_iter_ms": avg_iter_ms,
        "max_iter_ms": max_iter_ms,
        "rt_iter_tokens": rt_iter_tokens,
        "finished_rt": finished_rt,
        "rt_decode_tput": rt_decode_tput,
        "be_decode_tokens": be_decode_iter_tokens,
        "be_prefill_tokens": be_prefill_tokens,
        "be_decode_tput": be_decode_tput,
        "be_prefill_tput": be_prefill_tput,
        "be_saturation_tput": min(be_decode_tput, be_prefill_tput),
        "slo_violations": slo_violations,
        "leftover_active_rt": len(active),
        "unadmitted_rt": max(0, len(arrival_times) - next_arr_idx),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    trace_df = load_and_preprocess_trace(args.profile,
                                         args.rps,
                                         args.num_instances,
                                         args.duration_time,
                                         args.avg_rt_output,
                                         args.avg_rt_context)
    print(f"[Trace] Loaded {len(trace_df)} RT-Decode requests within "
          f"{args.duration_time}s window "
          f"(rps={args.rps}, num_instances={args.num_instances}).")

    if len(trace_df) == 0:
        print("[Error] Empty trace after filtering. Abort.")
        sys.exit(1)

    predictor = TimePredictor(args.json_profile, num_layers=args.num_layers)

    # ------------------------------------------------------------------
    # Step 1: derive T^target = min(T_SLO, T^rate)
    # ------------------------------------------------------------------
    t_target, t_rate, c_rt = compute_target_iteration_time(
        args, trace_df, predictor)

    # ------------------------------------------------------------------
    # Step 2: iteration-level replay
    # ------------------------------------------------------------------
    print("\n[Step 2] Replaying decode iterations...")
    res = simulate_d_pool(args, trace_df, predictor, t_target)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("\n[Result] Saturo D-pool Saturation")
    print(f"  T^target            : {t_target*1000:.2f} ms "
          f"(T_SLO={args.slo*1000:.2f} ms, T^rate={t_rate*1000:.2f} ms)")
    print(f"  C^RT (resident cap) : {c_rt}")
    print(f"  Iterations replayed : {res['iters']}")
    print(f"  Avg iter latency    : {res['avg_iter_ms']:.2f} ms")
    print(f"  Max iter latency    : {res['max_iter_ms']:.2f} ms")
    print(f"  SLO violations      : {res['slo_violations']} "
          f"(RT-iter level; iter exceeded T_SLO)")
    print(f"  Finished RT reqs    : {res['finished_rt']}")
    print(f"  RT decode tput      : {res['rt_decode_tput']:.4f} req/s")
    print(f"  BE-Decode tokens    : {res['be_decode_tokens']}")
    print(f"  BE-Prefill tokens   : {res['be_prefill_tokens']}")
    print(f"  BE-Decode tput      : {res['be_decode_tput']:.4f} req/s")
    print(f"  BE-Prefill tput     : {res['be_prefill_tput']:.4f} req/s")
    print(f"  BE saturation tput  : {res['be_saturation_tput']:.4f} req/s "
          f"(min(BE-decode, BE-prefill))")
    if res['leftover_active_rt'] or res['unadmitted_rt']:
        print(f"  [Warn] {res['leftover_active_rt']} RT still active and "
              f"{res['unadmitted_rt']} RT unadmitted at window end.")


if __name__ == "__main__":
    main()
