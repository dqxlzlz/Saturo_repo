"""Local BE lifecycle P-pool simulator.

Unlike ``simulator.py`` + ``optimizer.py`` (which analytically pick a
flow-balanced BE chunk and compute throughput in closed form -- effectively
*assuming* a steady decode batch), this module runs a full discrete-event
simulation of the real engine behaviour:

* RT requests are replayed FIFO on a real clock (same as the analytical model).
* BE requests are **real objects** drawn from a workload trace, each carrying a
  ``prompt_len`` (must be prefilled, possibly over many chunks) and an
  ``output_len`` (decoded one token per engine step). A BE request can only
  enter the decode phase *after* its prefill completes -- no cheating with a
  pre-known decode batch.
* Whenever the GPU is idle (no RT queued), the engine runs mixed BE steps.
  Each step is bounded by a ``be_chunk_ms`` latency budget: it advances every
  currently-decoding BE request by one token and, in the remaining budget,
  processes a chunked-prefill slice of the head prefill request (Sarathi-style
  single-prefill + running-decodes mix).
* The decode batch is therefore **emergent**: it grows as prefills complete and
  shrinks as decodes finish / KV blocks are freed, capped only by the profiled
  KV capacity.
* Preemption granularity equals one step: before starting a BE step the engine
  checks whether finishing it would blow the TTFT SLO of any RT arriving during
  the step; if so it yields to RT instead (that step is not started).

Throughput is measured as completed BE requests per second over the run, so
sweeping ``be_chunk_ms`` and running one full simulation per value reveals the
*achieved* optimum rather than an analytical balance point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from saturo.predictors.time_predictor import TimePredictor

from .request import RTRequest


@dataclass
class BERequest:
    """A best-effort request with a real prefill/decode lifecycle."""

    prompt_len: int
    output_len: int
    prefilled: int = 0          # prompt tokens processed so far
    generated: int = 0          # decode tokens emitted so far
    blocks: int = 0             # KV blocks reserved at admission

    @property
    def in_decode(self) -> bool:
        return self.prefilled >= self.prompt_len

    @property
    def decode_done(self) -> bool:
        return self.generated >= self.output_len

    @property
    def context_len(self) -> int:
        # During prefill the resident context is what has been prefilled; once
        # decoding it is the full prompt plus generated tokens.
        if self.in_decode:
            return self.prompt_len + self.generated
        return max(1, self.prefilled)


@dataclass
class LocalBEResult:
    be_chunk_ms: float
    chunk_tokens: int = 0
    window_s: float = 0.0
    completed_be: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    steps: int = 0
    preemptions: int = 0
    rt_total: int = 0
    rt_violations: int = 0
    rt_slacks_s: List[float] = field(default_factory=list)
    busy_be_time_s: float = 0.0
    idle_time_s: float = 0.0
    max_decode_batch_seen: int = 0
    avg_decode_batch: float = 0.0
    # Per-executed-step BE decode batch size (len(decodes) at step start),
    # for distribution analysis; parallel to the executed-step order.
    decode_batch_sizes: List[int] = field(default_factory=list)

    @property
    def be_req_per_s(self) -> float:
        return self.completed_be / max(self.window_s, 1e-9)

    @property
    def be_token_per_s(self) -> float:
        return (self.prefill_tokens + self.decode_tokens) / max(
            self.window_s, 1e-9)

    @property
    def rt_violation_rate(self) -> float:
        return self.rt_violations / max(1, self.rt_total)


class LocalBESimulator:
    """Discrete-event P-pool simulator with a real BE prefill/decode loop."""

    def __init__(
        self,
        rt_requests: Sequence[RTRequest],
        be_supply: Sequence[Tuple[int, int]],
        predictor: TimePredictor,
        *,
        ttft_slo_s: float,
        usable_kv_blocks: int,
        block_size: int,
        max_prefill_chunk: int = 8192,
        prefill_search_granularity: int = 16,
    ):
        self.rt = sorted(rt_requests, key=lambda r: r.arrival_time_s)
        self.be_supply = list(be_supply)
        self.predictor = predictor
        self.ttft_slo_s = ttft_slo_s
        self.usable_blocks = int(usable_kv_blocks)
        self.block_size = int(block_size)
        self.max_prefill_chunk = int(max_prefill_chunk)
        self.gran = max(1, int(prefill_search_granularity))
        # Memoize the mixed-step latency: keyed on (decode_batch, prefill_chunk)
        # rounded to the search granularity so repeated steps are cheap.
        self._lat_cache: dict = {}

    # ---- latency helpers ---------------------------------------------------
    def _step_latency_s(self, decode_batch: int, prefill_chunk: int,
                        decode_ctx: int, prefill_ctx: int) -> float:
        key = (decode_batch, prefill_chunk, decode_ctx // 256,
               prefill_ctx // 256)
        cached = self._lat_cache.get(key)
        if cached is not None:
            return cached
        decode_reqs = ([(1, max(1, decode_ctx))] * decode_batch
                       if decode_batch > 0 else [])
        prefill_reqs = ([(prefill_chunk, max(prefill_chunk, prefill_ctx))]
                        if prefill_chunk > 0 else [])
        if not decode_reqs and not prefill_reqs:
            self._lat_cache[key] = 0.0
            return 0.0
        ms = self.predictor.predict_forward_ms(prefill_reqs, decode_reqs)
        val = max(0.0, ms) / 1000.0
        self._lat_cache[key] = val
        return val

    def _max_prefill_chunk_for_budget(
        self, budget_s: float, decode_batch: int, decode_ctx: int,
        prefill_ctx: int, remaining_prefill: int,
    ) -> Tuple[int, float]:
        """Largest prefill chunk whose mixed-step latency fits the budget."""
        base = self._step_latency_s(decode_batch, 0, decode_ctx, 0)
        if base > budget_s or remaining_prefill <= 0:
            return 0, base
        hi = min(self.max_prefill_chunk, remaining_prefill)
        hi = (hi // self.gran) * self.gran
        lo = 0
        best_chunk, best_lat = 0, base
        while lo <= hi:
            mid = ((lo + hi) // (2 * self.gran)) * self.gran
            if mid <= 0:
                break
            lat = self._step_latency_s(decode_batch, mid, decode_ctx,
                                       prefill_ctx)
            if lat <= budget_s:
                best_chunk, best_lat = mid, lat
                lo = mid + self.gran
            else:
                hi = mid - self.gran
        if best_chunk == 0:
            # Even the smallest granule overflows the budget; take one granule
            # so prefill still makes progress (a real engine admits >=1 token).
            gran_chunk = min(self.gran, remaining_prefill)
            best_lat = self._step_latency_s(decode_batch, gran_chunk,
                                            decode_ctx, prefill_ctx)
            best_chunk = gran_chunk
        return best_chunk, best_lat

    # ---- main loop ---------------------------------------------------------
    def run(self, be_chunk_ms: float, avg_decode_ctx: int,
            chunk_tokens: Optional[int] = None) -> LocalBEResult:
        """Simulate the window.

        Two sweep units are supported:
        * latency mode (``chunk_tokens is None``): each step packs the largest
          prefill slice whose predicted latency fits ``be_chunk_ms``;
        * token mode (``chunk_tokens`` given): plain chunked prefill -- every
          step processes exactly ``chunk_tokens`` prompt tokens (or the
          remainder) and the step latency is whatever the predictor says for
          that mix. ``be_chunk_ms`` is ignored for planning in this mode.
        """
        budget_s = be_chunk_ms / 1000.0
        res = LocalBEResult(be_chunk_ms=be_chunk_ms,
                            chunk_tokens=int(chunk_tokens or 0),
                            rt_total=len(self.rt))

        t = 0.0
        i = 0                       # next RT index
        n = len(self.rt)
        supply_idx = 0

        decodes: List[BERequest] = []
        prefill_req: Optional[BERequest] = None
        used_blocks = 0
        decode_batch_accum = 0.0
        decode_batch_samples = 0

        def admit_prefill():
            nonlocal prefill_req, used_blocks, supply_idx
            if prefill_req is not None or supply_idx >= len(self.be_supply):
                return
            plen, olen = self.be_supply[supply_idx]
            need = math.ceil(max(1, plen + olen) / self.block_size)
            if used_blocks + need > self.usable_blocks:
                return
            prefill_req = BERequest(prompt_len=plen, output_len=olen,
                                    blocks=need)
            used_blocks += need
            supply_idx += 1

        while i < n:
            # Serve any RT that has already arrived (FIFO, one at a time).
            if self.rt[i].arrival_time_s <= t:
                req = self.rt[i]
                start = max(t, req.arrival_time_s)
                wait = start - req.arrival_time_s
                slack = self.ttft_slo_s - wait - req.runtime_s
                res.rt_slacks_s.append(slack)
                if slack < 0.0:
                    res.rt_violations += 1
                t = start + req.runtime_s
                i += 1
                continue

            next_arrival = self.rt[i].arrival_time_s

            # Nothing to run for BE -> idle straight to the next arrival.
            admit_prefill()
            if not decodes and prefill_req is None:
                res.idle_time_s += max(0.0, next_arrival - t)
                t = next_arrival
                continue

            # Plan one mixed BE step.
            d = len(decodes)
            if prefill_req is not None:
                remaining = prefill_req.prompt_len - prefill_req.prefilled
                if chunk_tokens is not None:
                    # Plain chunked prefill: fixed token slice per step.
                    chunk = min(int(chunk_tokens), remaining)
                    step_lat = self._step_latency_s(
                        d, chunk, avg_decode_ctx, prefill_req.context_len)
                else:
                    chunk, step_lat = self._max_prefill_chunk_for_budget(
                        budget_s, d, avg_decode_ctx, prefill_req.context_len,
                        remaining)
            else:
                chunk, step_lat = 0, self._step_latency_s(
                    d, 0, avg_decode_ctx, 0)

            if step_lat <= 0.0:
                res.idle_time_s += max(0.0, next_arrival - t)
                t = next_arrival
                continue

            # Preemption check: if finishing this step delays RTs arriving
            # during it beyond their TTFT SLO, yield to RT (step not started).
            step_end = t + step_lat
            if next_arrival < step_end:
                queued_runtime = 0.0
                would_violate = False
                j = i
                while j < n and self.rt[j].arrival_time_s < step_end:
                    r = self.rt[j]
                    remaining_be = step_end - r.arrival_time_s
                    if remaining_be + queued_runtime + r.runtime_s > \
                            self.ttft_slo_s:
                        would_violate = True
                        break
                    queued_runtime += r.runtime_s
                    j += 1
                if would_violate:
                    res.preemptions += 1
                    # Idle to the arrival and let the serve-RT branch handle it.
                    res.idle_time_s += max(0.0, next_arrival - t)
                    t = next_arrival
                    continue

            # Execute the step.
            res.steps += 1
            res.busy_be_time_s += step_lat
            t = step_end
            decode_batch_accum += d
            decode_batch_samples += 1
            res.decode_batch_sizes.append(d)
            res.max_decode_batch_seen = max(res.max_decode_batch_seen, d)

            # Advance decodes (one token each).
            survivors: List[BERequest] = []
            for br in decodes:
                br.generated += 1
                res.decode_tokens += 1
                if br.decode_done:
                    res.completed_be += 1
                    used_blocks -= br.blocks
                else:
                    survivors.append(br)
            decodes = survivors

            # Advance the prefill slice.
            if prefill_req is not None and chunk > 0:
                prefill_req.prefilled += chunk
                res.prefill_tokens += chunk
                if prefill_req.in_decode:
                    if prefill_req.output_len > 0:
                        decodes.append(prefill_req)
                    else:
                        res.completed_be += 1
                        used_blocks -= prefill_req.blocks
                    prefill_req = None

            admit_prefill()

        res.window_s = t
        res.avg_decode_batch = (decode_batch_accum
                                / max(1, decode_batch_samples))
        return res
