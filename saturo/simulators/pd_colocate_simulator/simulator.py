"""Steady-state solver and chunk-size optimizer for PD-aggregated colocation.

The core is a fixed point in the per-iteration latency ``t_iter``:

    N_rt_d = rps * Lo_rt * t_iter          # RT decode residency (Little's law)
    P_rt   = rps * Lp_rt * t_iter          # RT prefill tokens / iter to sustain rps
    (BE prefill/decode split maximizes their minimum sustainable request rate)
    t_iter = predict(prefill, decode)      # depends on the batch above

Because N_rt_d and P_rt grow with t_iter, and t_iter grows with the batch, the
iteration is a monotone map. If it converges the RPS point is feasible; if
``t_iter`` runs away it means RT alone is over capacity for this chunk_size.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

from .config import PDColocateConfig
from .latency import LatencyPredictor


@dataclass
class ChunkPoint:
    """Result of solving one (chunk_size, rps) configuration."""
    chunk_size: int
    rps: float
    feasible: bool
    infeasible_reason: str = ""

    t_iter_ms: float = 0.0
    # RT
    rt_decode_batch: float = 0.0
    rt_prefill_tokens: float = 0.0
    rt_decode_blocks: float = 0.0
    rt_prefill_req_per_s: float = 0.0
    rt_decode_req_per_s: float = 0.0
    # BE
    be_decode_batch: float = 0.0
    be_decode_capacity_batch: float = 0.0
    be_prefill_tokens: float = 0.0
    be_decode_blocks: float = 0.0
    be_prefill_req_per_s: float = 0.0
    be_decode_req_per_s: float = 0.0
    be_prefill_tok_per_s: float = 0.0
    be_decode_tok_per_s: float = 0.0
    be_goodput_req_per_s: float = 0.0     # min(BE-P, BE-D)
    be_rate_gap_req_per_s: float = 0.0
    # bookkeeping
    kv_block_budget: int = 0
    kv_blocks_used: float = 0.0

    def to_dict(self) -> dict:
        return {
            "chunk_size": self.chunk_size,
            "rps": round(self.rps, 4),
            "feasible": self.feasible,
            "infeasible_reason": self.infeasible_reason,
            "t_iter_ms": round(self.t_iter_ms, 3),
            "rt_decode_batch": round(self.rt_decode_batch, 2),
            "rt_prefill_tokens": round(self.rt_prefill_tokens, 1),
            "rt_decode_blocks": round(self.rt_decode_blocks, 1),
            "rt_prefill_req_per_s": round(self.rt_prefill_req_per_s, 4),
            "rt_decode_req_per_s": round(self.rt_decode_req_per_s, 4),
            "be_decode_batch": round(self.be_decode_batch, 2),
            "be_decode_capacity_batch": round(
                self.be_decode_capacity_batch, 2),
            "be_prefill_tokens": round(self.be_prefill_tokens, 1),
            "be_decode_blocks": round(self.be_decode_blocks, 1),
            "be_prefill_req_per_s": round(self.be_prefill_req_per_s, 4),
            "be_decode_req_per_s": round(self.be_decode_req_per_s, 4),
            "be_prefill_tok_per_s": round(self.be_prefill_tok_per_s, 1),
            "be_decode_tok_per_s": round(self.be_decode_tok_per_s, 1),
            "be_goodput_req_per_s": round(self.be_goodput_req_per_s, 4),
            "be_rate_gap_req_per_s": round(
                self.be_rate_gap_req_per_s, 4),
            "kv_block_budget": self.kv_block_budget,
            "kv_blocks_used": round(self.kv_blocks_used, 1),
        }


@dataclass
class OptimizeResult:
    rps: float
    best: Optional[ChunkPoint]
    all_points: List[ChunkPoint] = field(default_factory=list)


class PDColocateSimulator:
    """Solve BE throughput vs chunk-size for a PD-aggregated RT+BE engine."""

    def __init__(self, config: PDColocateConfig,
                 latency: LatencyPredictor):
        self.config = config
        self.latency = latency
        w = config.workload
        bs = config.model.block_size
        # Average resident context for a decoding request ~ prompt + half output.
        self._ctx_rt_d = w.rt_prompt_len + w.rt_output_len / 2.0
        self._ctx_be_d = w.be_prompt_len + w.be_output_len / 2.0
        self._blocks_rt_d = math.ceil(self._ctx_rt_d / bs)
        self._blocks_be_d = math.ceil(self._ctx_be_d / bs)

    def _plan_be_work(
        self,
        chunk_size: int,
        n_rt_d: float,
        p_rt: float,
        blocks_rt: float,
    ):
        """Choose the BE decode/prefill split for one fixed-point iterate.

        For each candidate decode batch ``D``, all remaining token budget is
        assigned to BE prefill. The useful steady-state BE request rate is
        ``min(D / (t * avg_out), P / (t * avg_in))``. As in the P-pool
        simulator, binary search finds the P/D rate crossover and the nearby
        candidates plus capacity endpoints are scored explicitly.
        """
        w = self.config.workload
        sc = self.config.solver
        budget = self.config.model.kv_block_budget
        rem_kv = max(0.0, budget - blocks_rt)
        decode_capacity = (
            rem_kv / self._blocks_be_d if self._blocks_be_d else 0.0)
        rem_tokens = max(0.0, chunk_size - n_rt_d - p_rt)
        d_max = max(0.0, min(decode_capacity, rem_tokens))

        def evaluate(decode_batch: float):
            decode_batch = min(max(float(decode_batch), 0.0), d_max)
            prefill_tokens = max(0.0, rem_tokens - decode_batch)
            latency_s = self.latency.predict_iter_s(
                n_rt_d,
                self._ctx_rt_d,
                decode_batch,
                self._ctx_be_d,
                p_rt,
                prefill_tokens,
            )
            latency_s = max(latency_s, 1e-12)
            decode_rate = (
                decode_batch / (latency_s * w.be_output_len)
                if w.be_output_len > 0 else 0.0)
            prefill_rate = (
                prefill_tokens / (latency_s * w.be_prompt_len)
                if w.be_prompt_len > 0 else 0.0)
            return {
                "decode_batch": decode_batch,
                "prefill_tokens": prefill_tokens,
                "latency_s": latency_s,
                "decode_req_per_s": decode_rate,
                "prefill_req_per_s": prefill_rate,
                "goodput_req_per_s": min(decode_rate, prefill_rate),
                "decode_capacity_batch": d_max,
            }

        if not sc.balance_be_throughput or d_max <= 0.0:
            return evaluate(d_max)

        # Find the smallest D where BE-D request throughput reaches BE-P.
        lo, hi = 0.0, d_max
        epsilon = max(float(sc.be_balance_search_epsilon), 1e-9)
        while hi - lo > epsilon:
            mid = (lo + hi) / 2.0
            point = evaluate(mid)
            if (point["decode_req_per_s"]
                    >= point["prefill_req_per_s"]):
                hi = mid
            else:
                lo = mid

        crossover = (lo + hi) / 2.0
        candidates = {
            0.0,
            d_max,
            min(max(crossover - epsilon, 0.0), d_max),
            min(max(crossover, 0.0), d_max),
            min(max(crossover + epsilon, 0.0), d_max),
        }
        points = [evaluate(candidate) for candidate in candidates]
        return max(
            points,
            key=lambda point: (
                point["goodput_req_per_s"],
                -abs(point["decode_req_per_s"]
                     - point["prefill_req_per_s"]),
                point["prefill_tokens"],
            ),
        )

    # ------------------------------------------------------------------
    # Single (chunk_size, rps) fixed-point solve
    # ------------------------------------------------------------------

    def solve(self, chunk_size: int, rps: float) -> ChunkPoint:
        w = self.config.workload
        sc = self.config.solver
        budget = self.config.model.kv_block_budget
        point = ChunkPoint(chunk_size=chunk_size, rps=rps, feasible=False,
                           kv_block_budget=budget)

        # Decode-only baseline latency to detect divergence.
        base_s = self.latency.predict_iter_s(1, self._ctx_rt_d, 0, 0, 0, 0)
        cap_s = base_s * sc.divergence_factor

        t = base_s
        state = None
        for _ in range(sc.max_iters):
            n_rt_d = rps * w.rt_output_len * t
            p_rt = rps * w.rt_prompt_len * t

            # RT-only token budget check: RT decode + RT prefill must fit.
            if n_rt_d + p_rt > chunk_size:
                point.infeasible_reason = "rt_exceeds_chunk_budget"
                return point
            blocks_rt = n_rt_d * self._blocks_rt_d
            if blocks_rt > budget:
                point.infeasible_reason = "rt_exceeds_kv_budget"
                return point

            be_plan = self._plan_be_work(
                chunk_size, n_rt_d, p_rt, blocks_rt)
            n_be_d = be_plan["decode_batch"]
            p_be = be_plan["prefill_tokens"]
            t_new = be_plan["latency_s"]
            if t_new > cap_s:
                point.infeasible_reason = "iteration_latency_diverged"
                return point

            state = (
                n_rt_d, p_rt, blocks_rt, n_be_d, p_be,
                be_plan["decode_capacity_batch"])
            if abs(t_new - t) <= sc.tol_s:
                t = t_new
                # Keep reported occupancy/throughput consistent with the
                # converged iteration latency instead of the previous fixed-
                # point iterate.  The residual is bounded by ``tol_s``.
                n_rt_d = rps * w.rt_output_len * t
                p_rt = rps * w.rt_prompt_len * t
                blocks_rt = n_rt_d * self._blocks_rt_d
                be_plan = self._plan_be_work(
                    chunk_size, n_rt_d, p_rt, blocks_rt)
                n_be_d = be_plan["decode_batch"]
                p_be = be_plan["prefill_tokens"]
                t = be_plan["latency_s"]
                state = (
                    n_rt_d, p_rt, blocks_rt, n_be_d, p_be,
                    be_plan["decode_capacity_batch"])
                break
            t = (1.0 - sc.damping) * t + sc.damping * t_new
        else:
            point.infeasible_reason = "not_converged"
            return point

        (n_rt_d, p_rt, blocks_rt, n_be_d, p_be,
         n_be_d_capacity) = state
        point.feasible = True
        point.t_iter_ms = t * 1000.0
        point.rt_decode_batch = n_rt_d
        point.rt_prefill_tokens = p_rt
        point.rt_decode_blocks = blocks_rt
        # RT sustained by construction: one full request cohort per second.
        point.rt_prefill_req_per_s = (p_rt / w.rt_prompt_len / t
                                      if w.rt_prompt_len > 0 else 0.0)
        point.rt_decode_req_per_s = (n_rt_d / w.rt_output_len / t
                                     if w.rt_output_len > 0 else 0.0)

        point.be_decode_batch = n_be_d
        point.be_decode_capacity_batch = n_be_d_capacity
        point.be_prefill_tokens = p_be
        point.be_decode_blocks = n_be_d * self._blocks_be_d
        point.be_prefill_tok_per_s = p_be / t
        point.be_decode_tok_per_s = n_be_d / t
        point.be_prefill_req_per_s = (p_be / w.be_prompt_len / t
                                      if w.be_prompt_len > 0 else 0.0)
        point.be_decode_req_per_s = (n_be_d / w.be_output_len / t
                                     if w.be_output_len > 0 else 0.0)
        point.be_goodput_req_per_s = min(
            point.be_prefill_req_per_s, point.be_decode_req_per_s)
        point.be_rate_gap_req_per_s = abs(
            point.be_prefill_req_per_s - point.be_decode_req_per_s)
        point.kv_blocks_used = blocks_rt + point.be_decode_blocks
        return point

    # ------------------------------------------------------------------
    # Chunk-size sweep / optimize
    # ------------------------------------------------------------------

    def optimize(self, rps: float,
                 chunk_sizes: Optional[List[int]] = None) -> OptimizeResult:
        sizes = chunk_sizes or self.config.chunk_sizes
        points = [self.solve(int(c), rps) for c in sorted(set(sizes))]
        feasible = [p for p in points if p.feasible]
        best = None
        if feasible:
            best = max(feasible, key=lambda p: p.be_goodput_req_per_s)
        return OptimizeResult(rps=rps, best=best, all_points=points)
