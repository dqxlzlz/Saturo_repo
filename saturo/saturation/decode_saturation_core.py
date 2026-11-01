# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Core decode saturation engine and vLLM scheduler policy.

This module contains:
- Utility functions shared by runtime and simulator (sampling_offsets, etc.)
- DecodeSaturationEngine: pure-algorithm class with no SequenceGroup dependency
- DecodeSaturationPolicy: vLLM scheduler integration wrapping the engine
"""

import importlib
import json
import math
import os
import time
from dataclasses import replace
from bisect import bisect_right
from collections import deque
from pathlib import Path
from typing import (
    Callable,
    Deque,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

import numpy as np

from saturo.predictors.request_arima_arrival_predictor import ARIMAArrivalPredictor
from saturo.request_metadata import (
    is_be_request,
    is_decode_phase,
    is_rt_request,
)

from .types import DecodeSaturationDecision, RequestForecast, SaturationParams

# ---------------------------------------------------------------------------
# Environment helpers (used by Policy layer only)
# ---------------------------------------------------------------------------


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


# ---------------------------------------------------------------------------
# Shared utility functions
# ---------------------------------------------------------------------------


def sampling_offsets(window_s: float, tau_s: float) -> List[float]:
    """Return tau-step offsets for look-ahead constraint checks.

    For a fixed candidate tau, Saturo evaluates the RT memory and RT-only TBT
    constraints at 0, tau, 2*tau, ... within the look-ahead window.
    """
    if window_s <= 0:
        return [0.0]
    tau_s = max(tau_s, 1e-6)
    token_steps = int(math.ceil(window_s / tau_s))
    offsets = [min(window_s, i * tau_s) for i in range(token_steps + 1)]
    if offsets[-1] < window_s:
        offsets.append(window_s)
    # Round to 9 decimals to suppress floating-point noise and deduplicate.
    return sorted({round(x, 9) for x in offsets})


def forecast_length_after_elapsed(
    forecast: RequestForecast,
    elapsed_s: float,
    tau_s: float,
) -> int:
    """Predict the sequence length of a request after *elapsed_s* seconds.

    Returns 0 if the request would have finished by that time.
    """
    remaining = max(forecast.remaining_tokens, 0)
    if remaining <= 0:
        return 0
    decode_steps = int(math.floor(max(elapsed_s, 0.0) / tau_s + 1e-9))
    if decode_steps > remaining:
        return 0
    appended_tokens = min(decode_steps, remaining)
    return forecast.prompt_len + forecast.generated_tokens + appended_tokens


def blocks_for_tokens(num_tokens: int, block_size: int) -> int:
    """Number of KV-cache blocks needed for *num_tokens*."""
    return max(1, int(math.ceil(max(num_tokens, 1) / max(block_size, 1))))


# ---------------------------------------------------------------------------
# DecodeSaturationEngine — pure algorithm, no SequenceGroup dependency
# ---------------------------------------------------------------------------

# Type alias for the time-predictor callback.
# Signature: (prefill_reqs: List[(chunk_size, ctx_len)],
#             decode_reqs: List[(1, ctx_len)]) -> latency_ms
TimePredictorFn = Callable[
    [List[Tuple[int, int]], List[Tuple[int, int]]], float
]
AggregateTimePredictorFn = Callable[..., float]


def _identical_cohort_statistics(arrivals,start,remaining,offsets,tau,block):
    """Exact counts, contexts and blocks for an identical-length cohort.

    For arrival a_i/tau = q_i + f_i and offset t/tau + eps = k + f,
    decode_steps = k - q_i - I(f_i > f), apart from FP boundary ties.
    Alive arrivals form a sorted interval; prefix sums account for q_i,
    block quotients and residues. Recompute ambiguous ties using the original
    subtraction/division/floor expression, including phase wraparound.
    """
    arrivals=np.sort(arrivals)
    unique,weights=np.unique(arrivals,return_counts=True)
    prefix_weights=np.r_[np.int64(0),np.cumsum(weights,dtype=np.int64)]
    threshold=offsets-(remaining+1-1e-9)*tau
    ids=np.searchsorted(unique,threshold,side="right")
    n_unique=len(unique)
    for _ in range(64):
        prev=np.maximum(ids-1,0);after=np.minimum(ids,n_unique-1)
        prev_steps=np.floor((offsets-unique[prev])/tau+1e-9)
        next_steps=np.floor((offsets-unique[after])/tau+1e-9)
        too_many=(ids>0)&(prev_steps<remaining+1)
        too_few=(ids<n_unique)&(next_steps>=remaining+1)
        if not (too_many.any() or too_few.any()):break
        ids[too_many]-=1;ids[too_few]+=1
    else:raise ArithmeticError("phase expiry correction did not converge")
    left=prefix_weights[ids]
    right=np.searchsorted(arrivals,offsets,side="right")
    count=right-left
    quotient=arrivals/tau
    whole=np.floor(quotient).astype(np.int64)
    phase=quotient-whole
    base=start-whole
    base_floor=base//block
    residue=base%block
    whole_prefix=np.r_[np.int64(0),np.cumsum(whole,dtype=np.int64)]
    floor_prefix=np.r_[np.int64(0),np.cumsum(base_floor,dtype=np.int64)]
    histogram=np.zeros((len(arrivals)+1,block),dtype=np.int64)
    histogram[np.arange(1,len(arrivals)+1),residue]=1
    np.cumsum(histogram,axis=0,out=histogram)
    point=offsets/tau+1e-9
    k=np.floor(point).astype(np.int64);fraction=point-k
    columns=np.arange(len(arrivals))
    living=(columns[None,:]>=left[:,None])&(columns[None,:]<right[:,None])
    after=(phase[None,:]>fraction[:,None])&living
    after_count=after.sum(axis=1,dtype=np.int64)
    context=(start+k)*count-(whole_prefix[right]-whole_prefix[left])-after_count
    k_block,k_residue=np.divmod(k,block)
    hist=histogram[right]-histogram[left]
    blocks=count*k_block+(floor_prefix[right]-floor_prefix[left])
    blocks+=(hist*((np.arange(block)[None,:]+k_residue[:,None]+block-1)//block)).sum(axis=1,dtype=np.int64)
    target=(1-k_residue)%block
    blocks-=(after&(residue[None,:]==target[:,None])).sum(axis=1,dtype=np.int64)
    # Quotient splitting can differ by one at an integer boundary. Correct
    # those pairs with the exact original subtraction/division/floor order.
    phase_order=np.argsort(phase,kind="stable");sorted_phase=phase[phase_order]
    tolerance=16*np.finfo(np.float64).eps*np.maximum(np.maximum(1,np.abs(offsets/tau)),np.max(np.abs(quotient)))
    ranges=[(np.maximum(0,fraction-tolerance),np.minimum(1,fraction+tolerance))]
    ranges.append((np.where(fraction<tolerance,1+fraction-tolerance,2),np.ones(len(offsets))))
    ranges.append((np.zeros(len(offsets)),np.where(fraction+tolerance>1,fraction+tolerance-1,-1)))
    for low,high in ranges:
        first=np.searchsorted(sorted_phase,low,side="left")
        last=np.searchsorted(sorted_phase,high,side="right")
        for row in np.flatnonzero(last>first):
            indices=phase_order[first[row]:last[row]]
            indices=indices[(indices>=left[row])&(indices<right[row])]
            if not len(indices):continue
            actual=np.floor((offsets[row]-arrivals[indices])/tau+1e-9).astype(np.int64)
            np.maximum(actual,0,out=actual);np.minimum(actual,remaining,out=actual)
            predicted=k[row]-whole[indices]-(phase[indices]>fraction[row])
            context[row]+=int((actual-predicted).sum())
            blocks[row]+=int((((start+actual+block-1)//block)-((start+predicted+block-1)//block)).sum())
    return count,context,blocks


class DecodeSaturationEngine:
    """Core D-pool saturation decision engine.

    This class implements the binary-search-based algorithm that finds the
    largest target TBT (tau) such that RT memory and RT-only latency both
    stay within budget over a future look-ahead window.

    It is intentionally free of vLLM-specific types so that both the runtime
    scheduler and the offline simulator can use it directly.
    """

    def __init__(
        self,
        params: SaturationParams,
        time_predictor_fn: Optional[TimePredictorFn] = None,
        aggregate_time_predictor_fn: Optional[
            AggregateTimePredictorFn] = None,
        aggregate_batch_time_predictor_fn: Optional[Callable[..., np.ndarray]] = None,
    ) -> None:
        self.params = params
        self.time_predictor_fn = time_predictor_fn
        self.aggregate_time_predictor_fn = aggregate_time_predictor_fn
        self.aggregate_batch_time_predictor_fn = aggregate_batch_time_predictor_fn

    def compute_decision(
        self,
        active_rt: List[RequestForecast],
        future_rt: List[RequestForecast],
        current_rt_blocks: int,
        current_be_blocks: int,
        now_s: float,
        unreclaimable_be_blocks: int = 0,
    ) -> DecodeSaturationDecision:
        """Compute a full saturation decision from RT forecasts.

        ``unreclaimable_be_blocks`` is the BE occupancy that swap cannot
        reclaim (the in-progress BE prefill).  The RT feasibility bound is
        the capacity actually available to decode, not the raw total.
        """
        all_forecasts = active_rt + future_rt

        # KV blocks that RT decode can actually reach: everything except the
        # pinned in-progress BE prefill. Swappable BE decode is reclaimable,
        # so it is intentionally NOT subtracted here.
        rt_block_capacity = max(
            self.params.total_gpu_blocks
            - max(int(unreclaimable_be_blocks), 0), 0)

        # Per-decision cache: each unique tau is evaluated once for both
        # peak_blocks and max_tbt via the merged vectorized path.
        evaluation_cache: dict[float, "tuple[int, float]"] = {}
        forecast_arrays = self._forecast_arrays(all_forecasts, now_s)

        def eval_tau(tau_ms: float) -> "tuple[int, float]":
            key = float(tau_ms)
            if key not in evaluation_cache:
                evaluation_cache[key] = self._evaluate_tau_from_arrays(
                    forecast_arrays, tau_ms)
            return evaluation_cache[key]

        def candidate_feasible(tau_ms: float) -> bool:
            peak, tbt = eval_tau(tau_ms)
            if peak > rt_block_capacity:
                return False
            return tbt <= self.params.tbt_slo_ms

        t_rate_ms = self._find_max_safe_interval(
            active_rt,
            all_forecasts,
            now_s,
            candidate_feasible_fn=candidate_feasible,
        )
        search_feasible = (
            not all_forecasts
            or candidate_feasible(t_rate_ms)
        )
        target_interval_ms = min(self.params.tbt_slo_ms, t_rate_ms)
        predicted_peak_rt_blocks, rt_only_tbt_ms = eval_tau(t_rate_ms)

        be_block_budget = max(
            int(math.floor(self.params.total_gpu_blocks
                           * self.params.decode_block_fraction))
            - predicted_peak_rt_blocks
            - self.params.be_prefill_reserve_blocks, 0)
        available_be_blocks = max(be_block_budget - current_be_blocks, 0)

        # Optional transition debug: whenever the target jumps between two
        # consecutive decisions, dump the whole tau sweep so the shape of the
        # feasibility landscape (peak-vs-cap and tbt-vs-slo per tau) is
        # visible. Enabled via SATURO_SATURATION_DEBUG=1; capped to the first
        # 12 transitions to keep logs bounded.
        if _env_bool("SATURO_SATURATION_DEBUG"):
            last_target = getattr(self, "_debug_last_target_ms", None)
            dumps_done = getattr(self, "_debug_dumps", 0)
            if (last_target is not None
                    and abs(target_interval_ms - last_target) > 20.0
                    and dumps_done < 12):
                self._debug_dumps = dumps_done + 1
                rt_only_floor = max(
                    self.params.min_interval_ms,
                    self._predict_rt_only_tbt_now_ms(active_rt))
                lines = [
                    f"[sat-debug] now={now_s:.3f}s "
                    f"target {last_target:.2f} -> {target_interval_ms:.2f}ms "
                    f"cap={rt_block_capacity} "
                    f"(total={self.params.total_gpu_blocks} "
                    f"pinned={int(unreclaimable_be_blocks)}) "
                    f"active={len(active_rt)} future={len(future_rt)} "
                    f"rt_only_floor={rt_only_floor:.2f}ms "
                    f"search_feasible={search_feasible}"
                ]
                sweep = sorted({
                    5.0, 20.0, 40.0, round(rt_only_floor, 2), 60.0, 70.0,
                    80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0,
                    self.params.tbt_slo_ms,
                })
                for tau in sweep:
                    peak, tbt = eval_tau(tau)
                    mem_ok = peak <= rt_block_capacity
                    tbt_ok = tbt <= self.params.tbt_slo_ms
                    searchable = tau >= rt_only_floor - 1e-9
                    lines.append(
                        f"[sat-debug]   tau={tau:6.1f}ms "
                        f"peak={peak:5d}blk[{'fit ' if mem_ok else 'OVER'}] "
                        f"win_tbt={tbt:7.2f}ms[{'ok  ' if tbt_ok else 'VIOL'}] "
                        f"{'searchable' if searchable else 'below-floor'} "
                        f"-> {'FEASIBLE' if mem_ok and tbt_ok else 'infeasible'}")
                print("\n".join(lines), flush=True)
            self._debug_last_target_ms = target_interval_ms

        return DecodeSaturationDecision(
            enabled=True,
            t_rate_ms=t_rate_ms,
            target_interval_ms=target_interval_ms,
            rt_only_tbt_ms=rt_only_tbt_ms,
            predicted_peak_rt_blocks=predicted_peak_rt_blocks,
            current_rt_blocks=current_rt_blocks,
            current_be_blocks=current_be_blocks,
            be_block_budget=be_block_budget,
            max_be_prefill_tokens=available_be_blocks * self.params.block_size,
            search_feasible=search_feasible,
        )

    # ------------------------------------------------------------------
    # Internal algorithm
    # ------------------------------------------------------------------

    def _find_max_safe_interval(
        self,
        active_rt: List[RequestForecast],
        all_forecasts: List[RequestForecast],
        now_s: float,
        candidate_feasible_fn: Optional[Callable[[float], bool]] = None,
    ) -> float:
        if not all_forecasts:
            return self.params.tbt_slo_ms

        rt_only_tbt = self._predict_rt_only_tbt_now_ms(active_rt)
        if rt_only_tbt > self.params.tbt_slo_ms:
            return max(rt_only_tbt, self.params.tbt_slo_ms)

        lo = max(self.params.min_interval_ms, rt_only_tbt)
        hi = max(lo, self.params.tbt_slo_ms)
        best = lo
        is_feasible = candidate_feasible_fn or (
            lambda tau_ms: self._candidate_feasible(
                all_forecasts, tau_ms, now_s))
        # The upper bound is the requested outcome.  Most serviceable
        # operating points can use the full SLO, so avoid eight redundant
        # binary-search evaluations when ``hi`` is already feasible.
        if is_feasible(hi):
            return float(hi)
        while hi - lo > self.params.search_epsilon_ms:
            tau_ms = (lo + hi) / 2.0
            if is_feasible(tau_ms):
                best = tau_ms
                lo = tau_ms
            else:
                hi = tau_ms
        return float(best)

    def _candidate_feasible(
        self,
        forecasts: List[RequestForecast],
        tau_ms: float,
        now_s: float,
    ) -> bool:
        peak, tbt = self._evaluate_tau_vectorized(forecasts, tau_ms, now_s)
        if peak > self.params.total_gpu_blocks:
            return False
        if tbt > self.params.tbt_slo_ms:
            return False
        return True

    def _predict_rt_only_tbt_now_ms(
        self,
        active_rt: List[RequestForecast],
    ) -> float:
        if self.time_predictor_fn is None:
            return self.params.min_interval_ms
        decode_reqs = [
            (1, r.prompt_len + r.generated_tokens) for r in active_rt
            if r.remaining_tokens > 0
        ]
        if not decode_reqs:
            return self.params.min_interval_ms
        return self.time_predictor_fn([], decode_reqs)

    # ------------------------------------------------------------------
    # Vectorized merged evaluation
    # ------------------------------------------------------------------

    # Minimum number of offsets per decision to guarantee sufficient
    # TBT resolution.  When the event-based sampling produces fewer than
    # this, we interpolate a uniform grid as a safety net.
    _MIN_TBT_OFFSETS: int = 20

    @staticmethod
    def _forecast_arrays(
        forecasts: List[RequestForecast],
        now_s: float,
    ) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
        """Precompute immutable NumPy arrays from forecast list."""
        n = len(forecasts)
        arrival_offsets = np.empty(n, dtype=np.float64)
        start_lens = np.empty(n, dtype=np.int64)
        remaining = np.empty(n, dtype=np.int64)
        for i, r in enumerate(forecasts):
            arrival_offsets[i] = max(0.0, r.arrival_time_s - now_s)
            start_lens[i] = r.prompt_len + r.generated_tokens
            remaining[i] = max(r.remaining_tokens, 0)
        return arrival_offsets, start_lens, remaining

    @staticmethod
    def _event_offsets(
        arrival_offsets: np.ndarray,
        remaining: np.ndarray,
        tau_s: float,
        window_s: float,
    ) -> np.ndarray:
        """Compute the minimal set of offsets needed for correctness.

        Peak blocks and max TBT are both monotonic non-decreasing within any
        interval where no request arrives or finishes.  Therefore the global
        maximum must lie at one of:

        * the end of the look-ahead window, or
        * the decode step *immediately before* a request finishes
          (``completion - 0.5*tau_s``).

        This produces at most *N+1* offsets instead of the tau-step
        ``ceil(window_s/tau_s)+1`` that the brute-force scan used.
        """
        pre_finish = arrival_offsets + (remaining - 0.5) * tau_s
        valid = (
            (remaining > 0)
            & (arrival_offsets <= window_s)
            & (pre_finish >= arrival_offsets)
            & (pre_finish >= 0.0)
            & (pre_finish <= window_s)
        )
        return np.unique(np.round(np.concatenate([
            np.array([0.0, window_s], dtype=np.float64),
            pre_finish[valid],
        ]), 9))

    def _evaluate_tau_vectorized(
        self,
        forecasts: List[RequestForecast],
        tau_ms: float,
        now_s: float,
    ) -> "tuple[int, float]":
        """Return ``(peak_blocks, max_tbt_ms)`` in one NumPy pass.

        Compared to the two independent :meth:`_iter_predicted_length_snapshots`
        traversals this method:

        1. **Merges** the two passes into one (halves traversal).
        2. **Vectorises** the inner loop with NumPy broadcasting (eliminates
           Python-level per-request iteration).
        3. **Coarsens** offset sampling to *event boundaries* — arrivals and
           completions — which are the only points where the monotonic
           block/TBT functions can change direction.  This reduces offsets
           from O(window/tau) to O(N).
        """
        if not forecasts:
            return 0, 0.0

        return self._evaluate_tau_from_arrays(
            self._forecast_arrays(forecasts, now_s), tau_ms)

    def _evaluate_tau_from_arrays(self,arrays,tau_ms):
        arr,start,remaining=arrays
        tau=max(tau_ms/1000,1e-6);window=self.params.lookahead_window_s
        block=max(self.params.block_size,1)
        # Keep normal cases, arbitrary callbacks, huge domains and memory
        # requirements on the original evaluator. No request is discarded.
        if (not 1024<=len(arr)<=4096 or not 1<=block<=128
                or self.time_predictor_fn is None
                or self.aggregate_time_predictor_fn is None
                or not math.isfinite(tau) or not math.isfinite(window)
                or window<0 or not np.isfinite(arr).all()
                or np.any(arr<0) or np.any(start<0)
                or np.any(start>2**30) or np.any(remaining<0) or np.any(remaining>2**30)
                or max(float(arr.max(initial=0)),window)/tau>2**30):
            return self._evaluate_tau_broadcast(arrays,tau_ms)
        _,inverse,counts=np.unique(np.column_stack([start,remaining]),axis=0,
                                   return_inverse=True,return_counts=True)
        label=int(counts.argmax())
        if counts[label]<1024:
            return self._evaluate_tau_broadcast(arrays,tau_ms)
        group=inverse==label
        offsets=self._event_offsets(arr,remaining,tau,window)
        if len(offsets)<self._MIN_TBT_OFFSETS:
            grid=np.unique(np.round(np.linspace(0,window,max(self._MIN_TBT_OFFSETS,2)),9))
            offsets=np.unique(np.round(np.concatenate([offsets,grid]),9))
        offsets=np.sort(offsets)
        if len(offsets)>4098:
            return self._evaluate_tau_broadcast(arrays,tau_ms)
        try:
            alive,context,blocks=_identical_cohort_statistics(arr[group],int(start[group][0]),
                int(remaining[group][0]),offsets,tau,block)
        except ArithmeticError:
            # An extreme rounding plateau may span more than64 distinct
            # arrival values. Fall back before invoking any predictor callback.
            return self._evaluate_tau_broadcast(arrays,tau_ms)
        rest_arr=arr[~group];rest_start=start[~group];rest_remaining=remaining[~group]
        for left in range(0,len(offsets),128):
            if not len(rest_arr):break
            at=offsets[left:left+128]
            elapsed=at[:,None]-rest_arr[None,:]
            steps=np.floor(elapsed/tau+1e-9).astype(np.int64)
            living=(elapsed>=0)&(steps<=rest_remaining[None,:])
            np.maximum(steps,0,out=steps);np.minimum(steps,rest_remaining[None,:],out=steps)
            lengths=steps+rest_start[None,:]
            alive[left:left+128]+=living.sum(axis=1,dtype=np.int64)
            context[left:left+128]+=np.where(living,lengths,0).sum(axis=1,dtype=np.int64)
            blocks[left:left+128]+=np.where(living,(lengths+block-1)//block,0).sum(axis=1,dtype=np.int64)
        peak=int(blocks.max(initial=0));maximum=0.0
        batch=self.aggregate_batch_time_predictor_fn
        if callable(batch):
            nonempty=alive>0
            if nonempty.any():
                values=np.asarray(batch(prefill_tokens=0.0,prefill_sum_ctx=0.0,
                    decode_tokens=alive[nonempty].astype(np.float64),
                    decode_sum_ctx=context[nonempty].astype(np.float64),
                    prefill_request_count=0,decode_request_count=alive[nonempty]))
                maximum=float(np.max(values,where=values>0,initial=0))
        else:
            for count,ctx in zip(alive,context):
                if not count:continue
                value=self.aggregate_time_predictor_fn(prefill_tokens=0.0,prefill_sum_ctx=0.0,
                    decode_tokens=float(count),decode_sum_ctx=float(ctx),
                    prefill_request_count=0,decode_request_count=int(count))
                if value>maximum:maximum=value
        return peak,maximum

    def _evaluate_tau_broadcast(
        self,
        forecast_arrays: "tuple[np.ndarray, np.ndarray, np.ndarray]",
        tau_ms: float,
    ) -> "tuple[int, float]":
        """Evaluate one tau using batched NumPy snapshots.

        Runtime Oracle policies additionally use the time predictor's
        sufficient-statistics entry point, making each snapshot prediction
        O(1) instead of rebuilding an O(N) tuple list.
        """
        arr_off, start_lens, remaining = forecast_arrays
        if len(arr_off) == 0:
            return 0, 0.0

        tau_s = max(tau_ms / 1000.0, 1e-6)
        window_s = self.params.lookahead_window_s
        block_size = max(self.params.block_size, 1)

        # --- event-driven offsets (correctness-preserving sparse sampling) ---
        offsets = self._event_offsets(arr_off, remaining, tau_s, window_s)

        # Safety-net: when the forecast set is tiny and all requests survive
        # the full window, the event set may be just {0, window}.  Blend in a
        # uniform grid so the TBT predictor still sees enough resolution.
        if len(offsets) < self._MIN_TBT_OFFSETS:
            n_grid = max(self._MIN_TBT_OFFSETS, 2)
            grid = np.linspace(0.0, window_s, n_grid, dtype=np.float64)
            # Round to suppress floating-point noise before merging
            grid = np.unique(np.round(grid, 9))
            offsets = np.unique(np.round(np.concatenate([offsets, grid]), 9))
        offsets = np.sort(offsets)

        peak_blocks = 0
        max_tbt = 0.0
        has_predictor = self.time_predictor_fn is not None

        # Bound temporary matrices when an overloaded replay exposes thousands
        # of overdue arrivals to the Oracle.
        offset_batch_size = 128
        for left in range(0, len(offsets), offset_batch_size):
            batch_offsets = offsets[left:left + offset_batch_size]
            elapsed = batch_offsets[:, None] - arr_off[None, :]
            active = elapsed >= 0.0
            decode_steps = np.floor(
                elapsed / tau_s + 1e-9).astype(np.int64)
            alive = active & (decode_steps <= remaining[None, :])
            np.maximum(decode_steps, 0, out=decode_steps)
            np.minimum(
                decode_steps, remaining[None, :], out=decode_steps)
            lengths = decode_steps + start_lens[None, :]

            blocks = (lengths + block_size - 1) // block_size
            total_blocks = np.where(alive, blocks, 0).sum(
                axis=1, dtype=np.int64)
            if len(total_blocks):
                peak_blocks = max(peak_blocks, int(total_blocks.max()))

            if not has_predictor:
                continue
            alive_counts = alive.sum(axis=1, dtype=np.int64)
            sum_contexts = np.where(alive, lengths, 0).sum(
                axis=1, dtype=np.int64)
            for row_idx, n_alive_raw in enumerate(alive_counts):
                n_alive = int(n_alive_raw)
                if n_alive == 0:
                    continue
                if self.aggregate_time_predictor_fn is not None:
                    tbt = self.aggregate_time_predictor_fn(
                        prefill_tokens=0.0,
                        prefill_sum_ctx=0.0,
                        decode_tokens=float(n_alive),
                        decode_sum_ctx=float(sum_contexts[row_idx]),
                        prefill_request_count=0,
                        decode_request_count=n_alive,
                    )
                else:
                    alive_lens = lengths[row_idx][alive[row_idx]]
                    decode_reqs = [(1, int(length))
                                   for length in alive_lens]
                    tbt = self.time_predictor_fn([], decode_reqs)
                if tbt > max_tbt:
                    max_tbt = tbt

        return peak_blocks, max_tbt

    # ------------------------------------------------------------------
    # Legacy methods (delegate to vectorized path for compatibility)
    # ------------------------------------------------------------------

    def _predict_max_rt_only_tbt_ms(
        self,
        forecasts: List[RequestForecast],
        tau_ms: float,
        now_s: float,
    ) -> float:
        _, tbt = self._evaluate_tau_vectorized(forecasts, tau_ms, now_s)
        return tbt

    def _predict_peak_rt_blocks(
        self,
        forecasts: List[RequestForecast],
        tau_ms: float,
        now_s: float,
    ) -> int:
        peak, _ = self._evaluate_tau_vectorized(forecasts, tau_ms, now_s)
        return peak

    def _iter_predicted_length_snapshots(
        self,
        forecasts: List[RequestForecast],
        tau_ms: float,
        now_s: float,
    ) -> Iterable[Tuple[float, List[int]]]:
        """Legacy generator kept for external callers.

        New code should call :meth:`_evaluate_tau_vectorized` instead.
        """
        tau_s = max(tau_ms / 1000.0, 1e-6)
        offsets = sampling_offsets(self.params.lookahead_window_s, tau_s)

        for offset_s in offsets:
            lengths: List[int] = []
            for request in forecasts:
                arrival_offset = max(0.0, request.arrival_time_s - now_s)
                elapsed_s = offset_s - arrival_offset
                if elapsed_s < 0:
                    continue
                length = forecast_length_after_elapsed(
                    request, elapsed_s=elapsed_s, tau_s=tau_s)
                if length > 0:
                    lengths.append(length)
            yield offset_s, lengths


# ---------------------------------------------------------------------------
# DecodeSaturationStateProvider Protocol & Default Implementation
# ---------------------------------------------------------------------------


class DecodeSaturationStateProvider(Protocol):
    """Interface for Q_eta, future-arrival, and RT observation providers."""

    def get_remaining_tokens(self, seq_group) -> Optional[int]:
        ...

    def get_future_arrivals(
        self,
        now_s: float,
        window_s: float,
    ) -> Iterable[RequestForecast]:
        ...

    def observe_rt_arrival(self, seq_group, now_s: float) -> None:
        ...

    def observe_rt_finish(self, seq_group) -> None:
        ...


class DefaultDecodeSaturationStateProvider:
    """QRF-only active-RT lengths and cached-history future arrivals.

    Enabled runtime policies require a trained QRF artifact at construction.
    Missing models and prediction failures are errors. Generation max_tokens
    is never used as a remaining-length estimate or prediction clamp.
    Legacy require/strict arguments cannot enable a fallback.
    """

    def __init__(
        self,
        load_predictor: bool = True,
        require_model_len_predictor: Optional[bool] = None,
        require_length_predictor: Optional[bool] = None,
        predictor_path: Optional[str] = None,
        model_len_predictor_eta: Optional[float] = None,
        model_len_predictor_strict: Optional[bool] = None,
        lookahead_window_s: float = 5.0,
    ) -> None:
        if require_model_len_predictor is None:
            require_model_len_predictor = require_length_predictor

        self.model_len_predictor = None
        self.prediction_calls = 0
        self.model_len_predictor_eta = (
            float(model_len_predictor_eta)
            if model_len_predictor_eta is not None else _env_float(
                "VLLM_DECODE_SATURATION_MODEL_LEN_ETA",
                _env_float("VLLM_DECODE_SATURATION_LENGTH_ETA", 0.95),
            ))
        from saturo.predictors.qrf import load_required_qrf, resolve_qrf_path
        self.model_len_predictor_strict = True
        self.require_model_len_predictor = True
        self.model_len_predictor_path = resolve_qrf_path(predictor_path)
        if load_predictor:
            self.model_len_predictor = load_required_qrf(
                self.model_len_predictor_path)

        # ARIMA arrival predictor state
        self._arima_arrival_predictor = ARIMAArrivalPredictor()
        self._rt_arrival_observations: dict[str, tuple[float, int]] = {}
        self._rt_arrival_watermark: Optional[tuple[float, str]] = None
        self._lookahead_window_s = float(lookahead_window_s)
        self._prompt_len_sum = 0.0
        self._prompt_len_count = 0
        self._output_len_sum = 0.0
        self._output_len_count = 0
        self._history_statistics = None
        self._history_revision = 0
        self._traffic_revision = 0

    # Backward-compatible aliases ─────────────────────────────────────────────

    @property
    def length_predictor(self):
        """Backward-compatible alias for model_len_predictor."""
        return self.model_len_predictor

    @length_predictor.setter
    def length_predictor(self, predictor):
        self.model_len_predictor = predictor

    @property
    def length_predictor_eta(self) -> float:
        """Backward-compatible alias for model_len_predictor_eta."""
        return self.model_len_predictor_eta

    @length_predictor_eta.setter
    def length_predictor_eta(self, eta: float) -> None:
        self.model_len_predictor_eta = float(eta)

    @property
    def length_predictor_strict(self) -> bool:
        """Backward-compatible alias for model_len_predictor_strict."""
        return self.model_len_predictor_strict

    @length_predictor_strict.setter
    def length_predictor_strict(self, strict: bool) -> None:
        self.model_len_predictor_strict = bool(strict)

    @property
    def require_length_predictor(self) -> bool:
        """Backward-compatible alias for require_model_len_predictor."""
        return self.require_model_len_predictor

    @require_length_predictor.setter
    def require_length_predictor(self, require: bool) -> None:
        self.require_model_len_predictor = bool(require)

    # Core logic ──────────────────────────────────────────────────────────────

    def get_remaining_tokens(self, seq_group) -> int:
        from saturo.predictors.qrf import require_qrf
        require_qrf(self.model_len_predictor)
        # Prediction failures must surface; sampling caps belong to generation
        # stopping logic and never substitute for the paper's Q_eta.
        predicted = max(self._predict_remaining_tokens(seq_group), 0)
        self.prediction_calls += 1
        return predicted

    def get_remaining_tokens_batch(self, seq_groups: Sequence) -> List[int]:
        """Batch repeated pending-state queries without dropping RT requests."""
        if not seq_groups:
            return []
        from saturo.predictors.qrf import require_qrf
        predictor = require_qrf(self.model_len_predictor)
        predict_batch = getattr(predictor, "predict_remaining_batch", None)
        if not callable(predict_batch):
            # Custom predictors retain their scalar behavior and failures.
            return [self.get_remaining_tokens(group) for group in seq_groups]
        feature_mode = getattr(getattr(predictor, "config", None),
                               "feature_mode", None)
        predictions = []
        # Bound temporary feature/token storage independently of queue size.
        for left in range(0, len(seq_groups), 64):
            groups = seq_groups[left:left + 64]
            rows = [self._remaining_features(
                group, include_tokens=feature_mode != "len") for group in groups]
            values = list(predict_batch(rows, eta=self.model_len_predictor_eta))
            if len(values) != len(groups):
                raise RuntimeError("QRF remaining-length batch has invalid row count")
            predictions.extend(max(0, int(value)) for value in values)
            self.prediction_calls += len(groups)
        return predictions

    def predict_remaining_feature_rows(self, rows: Sequence[dict]) -> List[int]:
        """Predict immutable feature rows on a background-owned provider."""
        from saturo.predictors.qrf import require_qrf
        predictor = require_qrf(self.model_len_predictor)
        batch = getattr(predictor, "predict_remaining_batch", None)
        predictions = []
        for left in range(0, len(rows), 64):
            chunk = rows[left:left + 64]
            values = (list(batch(chunk, eta=self.model_len_predictor_eta))
                      if callable(batch) else [predictor.predict_remaining(
                          eta=self.model_len_predictor_eta, **row) for row in chunk])
            if len(values) != len(chunk):
                raise RuntimeError("QRF remaining-length batch has invalid row count")
            predictions.extend(max(0, int(value)) for value in values)
            self.prediction_calls += len(chunk)
        return predictions

    def _predict_remaining_tokens(self, seq_group) -> int:
        return int(self.model_len_predictor.predict_remaining(
            eta=self.model_len_predictor_eta,
            **self._remaining_features(seq_group)))

    def _remaining_features(self, seq_group, *, include_tokens=True) -> dict:
        seqs = seq_group.get_seqs()
        seq = max(seqs, key=lambda candidate: candidate.get_output_len())
        prompt_len = seq.get_prompt_len()
        generated_len = seq.get_output_len()
        return dict(
            prompt_len=prompt_len, generated_len=generated_len,
            prompt_token_ids=(self._token_ids(seq, "get_prompt_token_ids")
                              if include_tokens else None),
            generated_token_ids=(self._token_ids(seq, "get_output_token_ids")
                                 if include_tokens else None))

    @staticmethod
    def _token_ids(seq, method_name: str) -> Optional[Tuple[int, ...]]:
        method = getattr(seq, method_name, None)
        if method is None:
            return None
        return tuple(int(token_id) for token_id in method())

    @staticmethod
    def _traffic_metadata(seq_group) -> dict:
        params = getattr(seq_group, "sampling_params", None)
        return getattr(params, "extra_args", None) or {}

    def _skip_traffic_observation(self, seq_group) -> bool:
        return (str(getattr(seq_group, "request_id", "")).startswith("__warmup__")
                or bool(self._traffic_metadata(seq_group).get(
                    "saturo_skip_traffic_observation", False)))

    def observe_rt_arrival(self, seq_group, now_s: float) -> None:
        """Observe source traffic, independent of P/RPC completion batches.

        Only already-observed source arrivals are supplied by the runtime.
        Rebuild a bounded, sorted history so a late P completion can insert an
        older observation without turning transport order into inter-arrivals.
        Stable source IDs deduplicate retries; the eviction watermark prevents
        old retries from reentering after their IDs leave the bounded history.
        """
        if self._skip_traffic_observation(seq_group):
            return
        metadata = self._traffic_metadata(seq_group)
        arrival_time_s = float(metadata.get("saturo_source_arrival_time_s", now_s))
        if not math.isfinite(arrival_time_s) or arrival_time_s > float(now_s):
            raise ValueError("RT source arrival must be finite and already observed")
        source_id = str(metadata.get("saturo_source_request_id",
                        getattr(seq_group, "request_id", f"object:{id(seq_group)}")))
        observations = self._rt_arrival_observations
        key = (arrival_time_s, source_id)
        if (source_id in observations
                or (self._rt_arrival_watermark is not None
                    and key <= self._rt_arrival_watermark)):
            return
        seqs = seq_group.get_seqs()
        prompt_len = max(seq.get_prompt_len() for seq in seqs)
        observations[source_id] = (arrival_time_s, prompt_len)
        ordered = sorted((stamp, rid) for rid, (stamp, _) in observations.items())
        limit = self._arima_arrival_predictor.max_history_intervals + 1
        if len(ordered) > limit:
            evicted = ordered[:-limit]
            self._rt_arrival_watermark = evicted[-1]
            for _, rid in evicted:
                del observations[rid]
            ordered = ordered[-limit:]
        self._arima_arrival_predictor.seed_from_arrival_times(
            stamp for stamp, _ in ordered)
        self._prompt_len_sum = float(sum(length for _, length in observations.values()))
        self._prompt_len_count = len(observations)
        self._traffic_revision = getattr(self, "_traffic_revision", 0) + 1

    def observe_rt_finish(self, seq_group) -> None:
        """Feed an RT finish observation to update avg output length."""
        if self._skip_traffic_observation(seq_group):
            return
        seqs = seq_group.get_seqs()
        generated = max(seq.get_output_len() for seq in seqs)
        if generated > 0:
            self._output_len_sum += float(generated)
            self._output_len_count += 1
            self._traffic_revision = getattr(self, "_traffic_revision", 0) + 1

    def get_future_arrivals(
        self,
        now_s: float,
        window_s: float,
    ) -> Iterable[RequestForecast]:
        """Use cached history statistics, falling back to local observations."""
        lookahead = window_s if window_s > 0 else self._lookahead_window_s
        history = self._history_statistics
        if (history is not None and now_s <= history["valid_until_s"]
                and history["completed_requests"] > 0):
            rate = history["arrival_rate"]
            arrival_offsets = (
                [i / rate for i in range(1, int(lookahead * rate) + 1)]
                if rate > 0 else [])
            prompt_len = max(1, int(round(history["mean_prompt_len"])))
            output_len = max(1, int(round(history["mean_output_len"])))
            return [RequestForecast(
                request_id=f"history_rt_{i}", prompt_len=prompt_len,
                generated_tokens=0, remaining_tokens=output_len,
                arrival_time_s=now_s + offset, is_rt=True,
            ) for i, offset in enumerate(arrival_offsets)]
        arrival_offsets = self._arima_arrival_predictor.predict_arrival_offsets(
            lookahead, now_s=now_s)
        if not arrival_offsets:
            return ()

        prompt_len = max(1, int(round(self._avg_prompt_len())))
        output_len = max(1, int(round(self._avg_output_len())))
        return [
            RequestForecast(
                request_id=f"predicted_rt_{i}",
                prompt_len=prompt_len,
                generated_tokens=0,
                remaining_tokens=output_len,
                arrival_time_s=now_s + offset_s,
                is_rt=True,
            )
            for i, offset_s in enumerate(arrival_offsets, start=1)
        ]

    def set_history_statistics(self, statistics: dict) -> None:
        # Control-plane updates replace one snapshot atomically. This method
        # and the per-iteration forecast perform no Redis or file I/O.
        updated = dict(statistics) if statistics is not None else None
        if updated != self._history_statistics:
            self._history_statistics = updated
            self._history_revision = getattr(self, "_history_revision", 0) + 1

    def reset_history_observations(self) -> None:
        """Discard old pool history at a versioned layout transition."""
        self._history_statistics = None
        self._rt_arrival_observations = {}
        self._rt_arrival_watermark = None
        self._arima_arrival_predictor.seed_from_arrival_times(())
        self._prompt_len_sum = self._output_len_sum = 0.
        self._prompt_len_count = self._output_len_count = 0
        self._history_revision = getattr(self, "_history_revision", 0) + 1
        self._traffic_revision = getattr(self, "_traffic_revision", 0) + 1

    def _avg_prompt_len(self) -> float:
        if self._prompt_len_count == 0:
            return 1024.0
        return self._prompt_len_sum / self._prompt_len_count

    def _avg_output_len(self) -> float:
        if self._output_len_count == 0:
            from saturo.predictors.qrf import require_qrf
            predictor = require_qrf(self.model_len_predictor)
            if getattr(getattr(predictor, "config", None), "feature_mode", "len") != "len":
                # Future arrivals have no prompt yet. Use training-history
                # statistics until actual completions are observed; do not
                # invoke a content predictor with invented/missing tokens.
                mean = getattr(predictor, "metadata", {}).get("mean_output_len")
                if mean is None or not math.isfinite(float(mean)) or float(mean) <= 0:
                    raise RuntimeError("Content QRF cold start requires mean_output_len training metadata")
                return float(mean)
            return float(predictor.predict_remaining(
                eta=self.model_len_predictor_eta,
                prompt_len=max(1, int(round(self._avg_prompt_len()))),
                generated_len=0))
        return self._output_len_sum / self._output_len_count


class OracleDecodeSaturationStateProvider:
    """Exact future-arrival and remaining-length provider for trace replay.

    The schedule is intentionally an explicit runtime artifact rather than a
    reference to the benchmark's in-memory trace.  Its timestamps are offsets
    from the first RT arrival; the first observed RT request anchors those
    offsets to the worker's wall clock.
    """

    def __init__(
        self,
        schedule_path: str,
        *,
        lookahead_window_s: float = 60.0,
    ) -> None:
        payload = json.loads(
            Path(schedule_path).expanduser().resolve().read_text(
                encoding="utf-8"))
        raw_requests = payload.get("requests", payload)
        if not isinstance(raw_requests, list):
            raise ValueError(
                "decode saturation oracle schedule must contain a requests "
                "list")

        requests: list[tuple[float, int, int]] = []
        for row in raw_requests:
            if not isinstance(row, dict):
                raise ValueError(
                    "decode saturation oracle request entries must be objects")
            offset_s = float(
                row.get("arrival_offset_s", row.get("arrival_time_s", 0.0)))
            prompt_len = max(int(row["prompt_len"]), 1)
            output_len = max(int(row["output_len"]), 1)
            requests.append((max(offset_s, 0.0), prompt_len, output_len))
        requests.sort(key=lambda item: item[0])

        self.schedule_path = str(Path(schedule_path).expanduser().resolve())
        self.requests = requests
        self._arrival_offsets = [request[0] for request in requests]
        self.lookahead_window_s = float(
            payload.get("lookahead_window_s", lookahead_window_s))
        self._anchor_time_s: Optional[float] = None
        self._observed_arrivals = 0

    @property
    def is_oracle(self) -> bool:
        return True

    def get_remaining_tokens(self, seq_group) -> Optional[int]:
        sampling_params = seq_group.sampling_params
        if sampling_params is None:
            return None
        max_tokens = getattr(sampling_params, "max_tokens", None)
        if max_tokens is None:
            return None
        generated = max(
            seq.get_output_len() for seq in seq_group.get_seqs())
        return max(int(max_tokens) - int(generated), 0)

    def observe_rt_arrival(self, seq_group, now_s: float) -> None:
        del seq_group
        if self._observed_arrivals < len(self.requests):
            offset_s = self.requests[self._observed_arrivals][0]
            if self._anchor_time_s is None:
                self._anchor_time_s = float(now_s) - offset_s
        elif self._anchor_time_s is None:
            self._anchor_time_s = float(now_s)
        self._observed_arrivals += 1

    def observe_rt_finish(self, seq_group) -> None:
        del seq_group

    def get_future_arrivals(
        self,
        now_s: float,
        window_s: float,
    ) -> Iterable[RequestForecast]:
        if self._anchor_time_s is None:
            return ()
        lookahead = (
            float(window_s) if window_s > 0 else self.lookahead_window_s)
        end_s = float(now_s) + lookahead
        future: list[RequestForecast] = []
        end_offset_s = end_s - self._anchor_time_s
        end_index = bisect_right(
            self._arrival_offsets,
            end_offset_s,
            lo=min(self._observed_arrivals, len(self.requests)),
        )
        for index in range(self._observed_arrivals, end_index):
            offset_s, prompt_len, output_len = self.requests[index]
            arrival_time_s = self._anchor_time_s + offset_s
            future.append(RequestForecast(
                request_id=f"oracle_rt_{index}",
                prompt_len=prompt_len,
                generated_tokens=0,
                remaining_tokens=output_len,
                arrival_time_s=arrival_time_s,
                is_rt=True,
            ))
        return future


def _load_provider() -> DecodeSaturationStateProvider:
    provider_path = os.environ.get("VLLM_DECODE_SATURATION_PROVIDER")
    if not provider_path:
        return DefaultDecodeSaturationStateProvider()
    module_name, sep, attr_name = provider_path.partition(":")
    if not sep:
        raise ValueError(
            "VLLM_DECODE_SATURATION_PROVIDER must be 'module:factory'")
    module = importlib.import_module(module_name)
    factory = getattr(module, attr_name)
    return factory()


# ---------------------------------------------------------------------------
# DecodeSaturationPolicy — full vLLM scheduler integration
# ---------------------------------------------------------------------------


class DecodeSaturationPolicy:
    """High-level decode saturation policy for the vLLM V0 scheduler.

    Wraps :class:`DecodeSaturationEngine` with SequenceGroup integration
    and remaining-length estimation via the configured state provider.
    ARIMA arrival prediction is handled by the provider.
    """

    def __init__(
        self,
        block_size: int,
        total_gpu_blocks: int,
        scheduler_config=None,
    ) -> None:
        config_enabled = getattr(
            scheduler_config, "enable_decode_saturation", None)
        self.enabled = bool(config_enabled) or _env_bool(
            "VLLM_DECODE_SATURATION", False)
        self.block_size = max(int(block_size), 1)
        self.total_gpu_blocks = max(int(total_gpu_blocks or 0), 0)
        self.tbt_slo_ms = float(
            getattr(scheduler_config, "decode_saturation_tbt_ms", 100.0)
            if scheduler_config is not None else _env_float(
                "VLLM_DECODE_SATURATION_TBT_MS", 100.0))
        self.lookahead_window_s = float(
            getattr(scheduler_config, "decode_saturation_lookahead_s", 5.0)
            if scheduler_config is not None else _env_float(
                "VLLM_DECODE_SATURATION_LOOKAHEAD_S", 5.0))
        self.min_interval_ms = float(
            getattr(scheduler_config, "decode_saturation_min_interval_ms", 1.0)
            if scheduler_config is not None else _env_float(
                "VLLM_DECODE_SATURATION_MIN_INTERVAL_MS", 1.0))
        self.search_epsilon_ms = float(
            getattr(scheduler_config, "decode_saturation_epsilon_ms", 0.5)
            if scheduler_config is not None else _env_float(
                "VLLM_DECODE_SATURATION_EPSILON_MS", 0.5))
        self.decode_block_fraction = float(
            getattr(scheduler_config, "decode_saturation_block_fraction", None)
            if (scheduler_config is not None and getattr(
                scheduler_config, "decode_saturation_block_fraction", None)
                is not None)
            else _env_float("SATURO_DECODE_BLOCK_FRACTION", 1.0))
        self.be_prefill_reserve_blocks = int(
            getattr(scheduler_config,
                    "decode_saturation_be_prefill_reserve_blocks", None)
            if (scheduler_config is not None and getattr(
                scheduler_config,
                "decode_saturation_be_prefill_reserve_blocks", None)
                is not None)
            else _env_float("SATURO_DECODE_BE_PREFILL_RESERVE_BLOCKS", 0.0))
        if self.enabled:
            if os.environ.get("SATURO_DECODE_ORACLE_SCHEDULE", "").strip():
                raise ValueError(
                    "Saturo runtime decode saturation requires QRF; oracle "
                    "schedules are only supported by explicit replay baselines")
            self.provider = DefaultDecodeSaturationStateProvider(
                predictor_path=getattr(
                    scheduler_config, "decode_saturation_model_len_predictor",
                    None),
                model_len_predictor_eta=getattr(
                    scheduler_config, "decode_saturation_model_len_eta", None),
                require_model_len_predictor=True,
                model_len_predictor_strict=True,
                lookahead_window_s=self.lookahead_window_s,
            )
        else:
            self.provider = DefaultDecodeSaturationStateProvider(
                load_predictor=False,
                lookahead_window_s=self.lookahead_window_s)
        self.time_predictor = None

        self.last_decision: Optional[DecodeSaturationDecision] = None

        # Build the core engine (lazily rebuilt if time_predictor changes)
        self._engine: Optional[DecodeSaturationEngine] = None
        self.async_target_enabled = False
        self.update_interval_s = 40.0
        self.update_timeout_s = 30.0
        self._target_epoch = self._target_version = 0
        self._external_target_epoch = 0
        self._target_history_ready = True
        self._async_optimizer = self._async_computer = None
        self._async_computer_key = self._async_signature = None
        self._async_applied = None
        self._async_last_submit_s = float("-inf")
        self._async_last_error = None
        self._async_totals = dict(submitted=0, completed=0, discarded=0,
                                  coalesced=0, qrf_prediction_calls=0)
        self.configure_update_protocol(
            async_enabled=bool(getattr(
                scheduler_config, "decode_saturation_async_target", False)),
            update_interval_s=float(getattr(
                scheduler_config, "decode_saturation_update_interval_s", 40)),
            update_timeout_s=float(getattr(
                scheduler_config, "decode_saturation_update_timeout_s", 30)))

    def configure_update_protocol(
        self, *, async_enabled=None, update_interval_s=None,
        update_timeout_s=None, lookahead_window_s=None, epoch=None,
        tbt_slo_ms=None, history_ready=None,
    ) -> None:
        """Configure an idle/iteration-boundary policy; invalidate old epochs.

        A layout/history transaction must pass its new epoch before publishing
        replacement history. Disabling async closes its worker (P/T roles).
        No model copying or prediction happens in this method.
        """
        enabled = (self.async_target_enabled if async_enabled is None
                   else bool(async_enabled))
        interval = (self.update_interval_s if update_interval_s is None
                    else float(update_interval_s))
        timeout = (self.update_timeout_s if update_timeout_s is None
                   else float(update_timeout_s))
        horizon = (self.lookahead_window_s if lookahead_window_s is None
                   else float(lookahead_window_s))
        slo = self.tbt_slo_ms if tbt_slo_ms is None else float(tbt_slo_ms)
        if (any(not math.isfinite(v) or v <= 0 for v in (interval, timeout))
                or any(not math.isfinite(v) or v < 0 for v in (horizon, slo))):
            raise ValueError("Target update interval/timeout/horizon/SLO must be positive")
        ready = (self._target_history_ready if history_ready is None
                 else bool(history_ready))
        external = self._external_target_epoch if epoch is None else int(epoch)
        if external < self._external_target_epoch or external < 0:
            raise ValueError("Target epoch must not move backwards")
        changed = (enabled != self.async_target_enabled
                   or interval != self.update_interval_s
                   or timeout != self.update_timeout_s
                   or horizon != self.lookahead_window_s
                   or slo != self.tbt_slo_ms
                   or ready != self._target_history_ready
                   or external != self._external_target_epoch)
        kernel_parameters_changed = (horizon != self.lookahead_window_s
                                     or slo != self.tbt_slo_ms)
        if not changed:
            if not enabled and self._async_optimizer is not None:
                if not self.close(timeout_s=min(timeout, 1.)):
                    raise RuntimeError("Previous async target worker is still closing")
            return
        self.async_target_enabled = enabled
        self.update_interval_s, self.update_timeout_s = interval, timeout
        self.lookahead_window_s, self.tbt_slo_ms = horizon, slo
        self._external_target_epoch = external
        self._target_history_ready = ready
        if kernel_parameters_changed:
            self._engine = None
        # History/layout metadata invalidates decisions, not fitted artifacts.
        # Reuse the one private model graph/cache across epochs and role changes
        # rather than repeatedly cloning a process-lifetime frozen startup heap.
        self._invalidate_async_target(reset_computer=kernel_parameters_changed)
        if not enabled and self._async_optimizer is not None:
            if not self.close(timeout_s=min(timeout, 1.)):
                raise RuntimeError("Previous async target worker is still closing")

    def _invalidate_async_target(self, *, reset_computer=False):
        self._target_epoch += 1
        self._target_version += 1
        self._async_applied = self._async_signature = None
        self._async_last_submit_s = float("-inf")
        if reset_computer:
            self._async_computer = self._async_computer_key = None
        if self._async_optimizer is not None:
            self._async_optimizer.invalidate(self._target_epoch, self._target_version)

    def close(self, timeout_s: float = 1.) -> bool:
        """Stop and join the target worker; safe to call more than once."""
        self._async_applied = None
        self._async_signature = None
        if self._async_optimizer is None:
            return True
        closed = self._async_optimizer.close(timeout_s)
        if closed:
            stats = self._async_optimizer.diagnostics()
            for key in self._async_totals:
                self._async_totals[key] += stats.get(key, 0)
            self._async_optimizer = None
        return closed

    def update_protocol_diagnostics(self) -> dict:
        stats = (self._async_optimizer.diagnostics()
                 if self._async_optimizer is not None else
                 dict(active=0, pending=0, thread_alive=False))
        for key, previous in self._async_totals.items():
            stats[key] = stats.get(key, 0) + previous
        stats["background_target_computations"] = stats["completed"]
        stats["background_qrf_prediction_calls"] = stats["qrf_prediction_calls"]
        return dict(stats, async_enabled=self.async_target_enabled,
                    epoch=self._target_epoch, version=self._target_version,
                    external_epoch=self._external_target_epoch,
                    history_ready=self._target_history_ready,
                    update_interval_s=self.update_interval_s,
                    update_timeout_s=self.update_timeout_s,
                    lookahead_window_s=self.lookahead_window_s,
                    applied_version=(self._async_applied.version
                                     if self._async_applied else None),
                    kernel_initialized=bool(self._async_computer is not None
                                            and self._async_computer.initialized),
                    last_error=self._async_last_error)

    def _ensure_async_computer(self):
        from .async_target import AsyncTargetOptimizer, TargetComputer
        if not isinstance(self.provider, DefaultDecodeSaturationStateProvider):
            raise TypeError("Async target requires a snapshot-capable default provider")
        if (self.time_predictor is None
                or not getattr(self.time_predictor, "available", True)):
            raise RuntimeError("Async target requires a fitted time predictor")
        predictor = self.provider.model_len_predictor
        key = (id(predictor), getattr(predictor, "_prediction_revision", 0),
               id(self.time_predictor), self.provider.model_len_predictor_eta)
        if self._async_computer_key != key:
            self._invalidate_async_target(reset_computer=True)
            self._engine = None
            self._async_computer = TargetComputer(
                predictor, self.time_predictor, self._get_engine().params,
                self.provider.model_len_predictor_eta)
            self._async_computer_key = key
        if self._async_optimizer is None:
            self._async_optimizer = AsyncTargetOptimizer()

    def _freeze_async_snapshot(self, groups, now_s, rt_blocks, be_blocks, pinned):
        from .async_target import ArrivalState, RequestFeatures, TargetSnapshot
        provider = self.provider
        content = getattr(getattr(provider.model_len_predictor, "config", None),
                          "feature_mode", None) != "len"
        requests = []
        for group in groups:
            seqs = group.get_seqs()
            seq = max(seqs, key=lambda item: item.get_output_len())
            # Native token getters return immutable tuples. tuple(tuple) shares
            # that stable value; embedding callers returning lists are copied.
            prompt = tuple(seq.get_prompt_token_ids()) if content else None
            generated = tuple(seq.get_output_token_ids()) if content else None
            requests.append(RequestFeatures(
                str(group.request_id), int(seq.get_prompt_len()),
                int(seq.get_output_len()), float(group.arrival_time or now_s),
                prompt, generated, max(s.get_prompt_len() for s in seqs)))
        history = provider._history_statistics
        if history is not None and any(
                not isinstance(value, (str, int, float, bool, type(None)))
                for value in history.values()):
            raise TypeError("Async history snapshot must contain scalar values")
        arrival = provider._arima_arrival_predictor
        state = ArrivalState(
            tuple(sorted(history.items())) if history is not None else None,
            tuple(arrival._inter_arrivals_s), arrival._last_arrival_time_s,
            arrival.min_history_intervals, arrival.max_history_intervals,
            arrival.min_interval_s, provider._prompt_len_sum,
            provider._prompt_len_count, provider._output_len_sum,
            provider._output_len_count)
        return TargetSnapshot(
            self._target_version, self._target_epoch, now_s, tuple(requests),
            state, rt_blocks, be_blocks, pinned, time.perf_counter())

    def _accept_async_result(self, result):
        if result is None:
            return
        if (result.epoch != self._target_epoch
                or result.version != self._target_version):
            return
        if result.error_type is not None:
            self._async_applied = None
            self._async_last_error = dict(type=result.error_type,
                                          message=result.error_message,
                                          version=result.version)
            raise RuntimeError("Async decode target failed: "
                               f"{result.error_type}: {result.error_message}")
        self._async_applied = result
        self._async_last_error = None

    def warm_update_protocol(self, timeout_s: float = 30.) -> dict:
        """Initialize private model state before actor ready/GC freeze only."""
        if not self.async_target_enabled or not self.enabled:
            return self.update_protocol_diagnostics()
        self._ensure_async_computer()
        self._target_version += 1
        snapshot = self._freeze_async_snapshot([], time.time(), 0, 0, 0)
        self._async_optimizer.submit(snapshot, self._async_computer)
        result = self._async_optimizer.wait(timeout_s)
        if result is None:
            self.close(timeout_s=min(timeout_s, 1.))
            raise TimeoutError("Async decode target startup warmup timed out")
        try:
            self._accept_async_result(result)
        except Exception:
            self.close(timeout_s=min(timeout_s, 1.))
            raise
        # The empty startup state cannot be applied to later native requests.
        self._async_applied = self._async_signature = None
        return self.update_protocol_diagnostics()

    def _make_async_decision(self, running, pending_decode_admission, now_s):
        started = time.perf_counter()
        rt_groups = ([sg for sg in running if self.is_rt(sg)] +
                     [sg for sg in pending_decode_admission if self.is_rt(sg)])
        rt_blocks = sum(self._seq_group_blocks(sg) for sg in running if self.is_rt(sg))
        be_blocks = sum(self._seq_group_blocks(sg) for sg in running if self.is_be(sg))
        pinned = sum(self._seq_group_blocks(sg) for sg in running
                     if self.is_be(sg) and not is_decode_phase(sg))
        if not self._target_history_ready:
            decision = self._async_fallback(
                rt_blocks, be_blocks, now_s, "awaiting_history")
            self.last_phase_timings = {
                "async_main_wall_ms": (time.perf_counter()-started)*1000,
                "async_triggered": 0, "async_applied": 0,
                "async_target_version": self._target_version,
                "async_target_epoch": self._target_epoch}
            self.last_decision = decision
            return decision
        self._ensure_async_computer()
        signature = (tuple(str(sg.request_id) for sg in rt_groups), pinned,
                     getattr(self.provider, "_history_revision", 0),
                     getattr(self.provider, "_traffic_revision", 0))
        trigger = (signature != self._async_signature
                   or now_s - self._async_last_submit_s >= self.update_interval_s)
        if trigger:
            self._target_version += 1
            self._async_applied = None
            self._async_signature = signature
            self._async_last_submit_s = now_s
            snapshot = self._freeze_async_snapshot(
                rt_groups, now_s, rt_blocks, be_blocks, pinned)
            self._async_optimizer.submit(snapshot, self._async_computer)
        snapshot_finished = time.perf_counter()
        self._accept_async_result(self._async_optimizer.poll())
        if self._async_optimizer.active_age_s() > self.update_timeout_s:
            self._async_applied = None
            self._async_last_error = dict(type="TimeoutError",
                                          message="Async target update timed out")
            raise TimeoutError("Async decode target update exceeded its timeout")
        result = self._async_applied
        valid = (result is not None and result.epoch == self._target_epoch
                 and result.version == self._target_version
                 and 0 <= now_s - result.captured_time_s < self.update_interval_s)
        if valid:
            decision = result.decision
            # Live block ownership is authoritative while a result is reused.
            peak = max(decision.predicted_peak_rt_blocks, rt_blocks)
            budget = max(int(self.total_gpu_blocks * self.decode_block_fraction)
                         - peak - self.be_prefill_reserve_blocks, 0)
            decision = replace(decision, current_rt_blocks=rt_blocks,
                current_be_blocks=be_blocks, predicted_peak_rt_blocks=peak,
                be_block_budget=budget,
                max_be_prefill_tokens=max(budget-be_blocks, 0)*self.block_size,
                target_version=result.version, target_epoch=result.epoch,
                target_snapshot_time_s=result.captured_time_s,
                target_status="async_ready")
        else:
            # RT still executes. No old target may admit or execute BE while
            # an invalidated snapshot is pending (including resident BE).
            decision = self._async_fallback(rt_blocks, be_blocks, now_s, "pending")
        self.last_phase_timings = {
            "async_snapshot_ms": (snapshot_finished-started)*1000,
            "async_main_wall_ms": (time.perf_counter()-started)*1000,
            "async_triggered": int(trigger), "async_applied": int(valid),
            "async_target_version": self._target_version,
            "async_target_epoch": self._target_epoch}
        if valid:
            self.last_phase_timings.update(
                {"async_"+key: value for key, value in result.timings})
            self.last_phase_timings.update(
                async_worker_started_perf_s=result.started_perf_s,
                async_worker_completed_perf_s=result.completed_perf_s,
                async_result_age_ms=(now_s-result.captured_time_s)*1000)
        diagnostics = self._async_optimizer.diagnostics()
        for field in ("active", "pending", "discarded", "coalesced"):
            self.last_phase_timings["async_"+field] = diagnostics[field]
        self.last_decision = decision
        return decision

    def _async_fallback(self, rt_blocks, be_blocks, now_s, status):
        return DecodeSaturationDecision(
            True, self.min_interval_ms, self.min_interval_ms, 0.,
            rt_blocks, rt_blocks, be_blocks, 0, 0, False,
            self._target_version, self._target_epoch, now_s, status)

    @property
    def uses_oracle(self) -> bool:
        return bool(getattr(self.provider, "is_oracle", False))

    def _get_engine(self) -> DecodeSaturationEngine:
        """Lazily build or return the core engine."""
        if self._engine is not None:
            return self._engine
        params = SaturationParams(
            tbt_slo_ms=self.tbt_slo_ms,
            lookahead_window_s=self.lookahead_window_s,
            min_interval_ms=self.min_interval_ms,
            search_epsilon_ms=self.search_epsilon_ms,
            block_size=self.block_size,
            total_gpu_blocks=self.total_gpu_blocks,
            decode_block_fraction=self.decode_block_fraction,
            be_prefill_reserve_blocks=self.be_prefill_reserve_blocks,
        )
        predictor_fn: Optional[TimePredictorFn] = None
        aggregate_predictor_fn: Optional[AggregateTimePredictorFn] = None
        aggregate_batch_predictor_fn = None
        if self.time_predictor is not None:
            predictor_fn = self.time_predictor.predict_forward_ms
            aggregate_predictor_fn = getattr(
                self.time_predictor,
                "predict_forward_from_aggregates_ms",
                None,
            )
            aggregate_batch_predictor_fn = getattr(
                self.time_predictor,
                "predict_forward_from_aggregates_batch_ms", None)
        self._engine = DecodeSaturationEngine(
            params,
            predictor_fn,
            aggregate_predictor_fn,
            aggregate_batch_predictor_fn,
        )
        return self._engine

    # ------------------------------------------------------------------
    # Public API used by the V0 scheduler
    # ------------------------------------------------------------------

    def is_rt(self, seq_group) -> bool:
        return is_rt_request(seq_group)

    def is_be(self, seq_group) -> bool:
        return is_be_request(seq_group)

    def observe_rt_arrival(self, seq_group, now_s: float) -> None:
        if not self.enabled or not self.is_rt(seq_group):
            return
        self.provider.observe_rt_arrival(seq_group, now_s)

    def observe_rt_finish(self, seq_group) -> None:
        if not self.enabled or not self.is_rt(seq_group):
            return
        self.provider.observe_rt_finish(seq_group)

    def make_decision(
        self,
        running: Sequence,
        waiting: Sequence,
        pending_decode_admission: Sequence,
        now_s: float,
    ) -> DecodeSaturationDecision:
        if self.enabled and getattr(self, "async_target_enabled", False):
            return self._make_async_decision(
                running, pending_decode_admission, now_s)
        if not self.enabled:
            decision = DecodeSaturationDecision(
                enabled=False,
                t_rate_ms=self.tbt_slo_ms,
                target_interval_ms=self.tbt_slo_ms,
                rt_only_tbt_ms=0.0,
                predicted_peak_rt_blocks=0,
                current_rt_blocks=0,
                current_be_blocks=0,
                be_block_budget=self.total_gpu_blocks,
                max_be_prefill_tokens=0,
            )
            self.last_decision = decision
            return decision

        timing = _env_bool("SATURO_STEP_TIMING")
        planner_started = time.perf_counter() if timing else 0.0
        rt_groups = ([sg for sg in running if self.is_rt(sg)]
                     + [sg for sg in pending_decode_admission if self.is_rt(sg)])
        batch_remaining = getattr(self.provider, "get_remaining_tokens_batch", None)
        if callable(batch_remaining):
            remaining = list(batch_remaining(rt_groups))
            if len(remaining) != len(rt_groups):
                raise RuntimeError("RT remaining-length batch has invalid row count")
            active_rt = []
            for group, predicted in zip(rt_groups, remaining):
                if predicted is None:
                    raise RuntimeError("Decode saturation provider returned no "
                                       "remaining-length prediction for active RT "
                                       f"request {group.request_id!r}.")
                active_rt.append(self._state_from_seq_group(
                    group, now_s, remaining_tokens=predicted))
        else:
            active_rt = [self._state_from_seq_group(sg, now_s) for sg in rt_groups]
        # D-pool decode-only RT requests are represented by
        # ``pending_decode_admission``. Ordinary ``waiting`` contains local
        # prefills only and must not be counted as decode-ready state.

        state_finished = time.perf_counter() if timing else 0.0
        future_rt = list(self._future_arrivals(now_s))
        forecast_finished = time.perf_counter() if timing else 0.0
        current_rt_blocks = sum(self._seq_group_blocks(sg) for sg in running
                                if self.is_rt(sg))
        current_be_blocks = sum(self._seq_group_blocks(sg) for sg in running
                                if self.is_be(sg))
        # Blocks held by BE groups still in prefill phase: swap cannot
        # reclaim them, so they are off-limits to the RT feasibility bound.
        pinned_prefill_blocks = sum(
            self._seq_group_blocks(sg) for sg in running
            if self.is_be(sg) and not is_decode_phase(sg))

        blocks_finished = time.perf_counter() if timing else 0.0
        engine = self._get_engine()
        decision = engine.compute_decision(
            active_rt=active_rt,
            future_rt=future_rt,
            current_rt_blocks=current_rt_blocks,
            current_be_blocks=current_be_blocks,
            now_s=now_s,
            unreclaimable_be_blocks=pinned_prefill_blocks,
        )
        if timing:
            finished = time.perf_counter()
            self.last_phase_timings = {
                "active_state_ms": (state_finished - planner_started) * 1000,
                "future_forecast_ms": (forecast_finished - state_finished) * 1000,
                "block_accounting_ms": (blocks_finished - forecast_finished) * 1000,
                "target_search_ms": (finished - blocks_finished) * 1000,
                "planner_total_ms": (finished - planner_started) * 1000,
                "active_rt_count": len(active_rt),
                "future_rt_count": len(future_rt),
            }
            predictor = getattr(self.provider, "model_len_predictor", None)
            cache_info = getattr(predictor, "remaining_batch_cache_info", None)
            if callable(cache_info):
                info = cache_info()
                self.last_phase_timings.update({
                    "qrf_hits": info["hits"], "qrf_misses": info["misses"],
                    "qrf_model_calls": info["model_calls"],
                    "qrf_model_rows": info["model_rows"],
                    "qrf_cache_entries": info["entries"],
                    "qrf_cache_feature_bytes": info["feature_bytes"],
                })
        self.last_decision = decision
        return decision

    def order_running_queue(self, running: Deque) -> None:
        if not self.enabled:
            return
        rt_decode = []
        rt_other = []
        be_decode = []
        be_other = []
        for seq_group in running:
            is_decode = is_decode_phase(seq_group)
            if self.is_rt(seq_group):
                (rt_decode if is_decode else rt_other).append(seq_group)
            elif is_decode:
                be_decode.append(seq_group)
            else:
                be_other.append(seq_group)
        running.clear()
        running.extend(rt_decode + rt_other + be_decode + be_other)

    def can_admit_be(
        self,
        seq_group,
        decision: DecodeSaturationDecision,
        projected_be_blocks: int,
    ) -> bool:
        if not self.enabled or self.is_rt(seq_group):
            return True
        return (projected_be_blocks + self._seq_group_blocks(seq_group)
                <= decision.be_block_budget)

    def seq_group_blocks(self, seq_group) -> int:
        return self._seq_group_blocks(seq_group)

    def predict_decode_batch_ms(self, seq_groups: Sequence) -> float:
        if self.time_predictor is None:
            return 0.0
        decode_reqs = []
        for seq_group in seq_groups:
            if not is_decode_phase(seq_group):
                continue
            for seq in seq_group.get_seqs():
                decode_reqs.append((1, seq.get_num_computed_tokens()))
        if not decode_reqs:
            return 0.0
        return self.time_predictor.predict_forward_ms([], decode_reqs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _future_arrivals(self, now_s: float) -> Iterable[RequestForecast]:
        return self.provider.get_future_arrivals(
            now_s, self.lookahead_window_s)

    def _state_from_seq_group(
        self, seq_group, now_s: float, remaining_tokens: Optional[int] = None,
    ) -> RequestForecast:
        seqs = seq_group.get_seqs()
        prompt_len = max(seq.get_prompt_len() for seq in seqs)
        generated = max(seq.get_output_len() for seq in seqs)
        remaining = (self.provider.get_remaining_tokens(seq_group)
                     if remaining_tokens is None else remaining_tokens)
        if remaining is None:
            raise RuntimeError(
                "Decode saturation provider returned no remaining-length "
                f"prediction for active RT request {seq_group.request_id!r}.")
        return RequestForecast(
            request_id=seq_group.request_id,
            prompt_len=prompt_len,
            generated_tokens=generated,
            remaining_tokens=remaining,
            arrival_time_s=seq_group.arrival_time or now_s,
            is_rt=self.is_rt(seq_group),
        )

    def _seq_group_blocks(self, seq_group) -> int:
        total = 0
        for seq in seq_group.get_seqs():
            total += blocks_for_tokens(seq.get_len(), self.block_size)
        return total
