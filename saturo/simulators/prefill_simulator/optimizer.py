"""Throughput-driven optimizer for P-pool BE chunks."""

from dataclasses import dataclass, field
import math
import warnings
from typing import List, Optional

from .config import SimulatorConfig
from .latency import LatencyPredictor
from .request import BEChunk, RTRequest, SimulationResult
from .simulator import PrefillSimulator
from .trace_loader import compute_pure_rt_slacks


@dataclass
class OptimizationResult:
    best_result: SimulationResult
    all_results: List[SimulationResult] = field(default_factory=list)
    pure_rt_slacks_s: List[float] = field(default_factory=list)
    decode_only_latency_s: float = 0.0
    max_candidate_latency_s: Optional[float] = None
    # Pure-RT (BE-free) TTFT-SLO violation rate for this RPS point.
    rt_violation_rate: float = 0.0
    # True when the pure-RT violation rate exceeds the configured threshold,
    # i.e. RT alone is already over capacity and BE saturation was skipped.
    rt_infeasible: bool = False
    initial_budget_s: float = 0.0
    evaluated_candidates: int = 0
    stop_reason: str = ""
    budget_history_s: List[float] = field(default_factory=list)


class ChunkLatencyOptimizer:
    """Search BE latency budgets using replayed RT slack feedback."""

    def __init__(self, config: SimulatorConfig,
                 latency_predictor: LatencyPredictor):
        self.config = config
        self.latency_predictor = latency_predictor

    def optimize(
        self,
        rt_requests: List[RTRequest],
        max_candidates: int = 128,
        budget_granularity_ms: Optional[float] = None,
        *,
        max_iterations: int = 1024,
    ) -> OptimizationResult:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if budget_granularity_ms is not None:
            warnings.warn(
                "budget_granularity_ms is deprecated and ignored; "
                "BE budgets now advance by simulated RT slack",
                DeprecationWarning,
                stacklevel=2,
            )
        pure_slacks = compute_pure_rt_slacks(
            rt_requests, self.config.slo.ttft_slo_s)
        if not all(math.isfinite(slack) for slack in pure_slacks):
            raise ValueError("RT slacks must be finite")
        decode_batch = self.config.be.decode_batch_size
        decode_only_latency_s = self.latency_predictor.predict_be_chunk_s(
            decode_batch, 0, self.config.be.avg_be_decode_context)
        if not math.isfinite(decode_only_latency_s) or decode_only_latency_s < 0:
            raise ValueError("decode-only latency must be finite and non-negative")

        # Overload short-circuit: if RT alone already violates its TTFT SLO on
        # more than the tolerated fraction of requests, this RPS point is
        # beyond RT capacity. BE saturation is undefined here, so skip the
        # search and report saturation=0 (flagged rt_infeasible).
        num_slacks = len(pure_slacks)
        rt_violations = sum(1 for s in pure_slacks if s < 0.0)
        rt_violation_rate = (
            rt_violations / num_slacks if num_slacks else 0.0)
        if (num_slacks > 0
                and rt_violation_rate > self.config.slo.max_rt_violation_rate):
            idle_chunk = BEChunk(
                target_latency_s=0.0,
                latency_s=0.0,
                decode_batch_size=decode_batch,
                prefill_chunk_size=0,
            )
            infeasible = SimulationResult(
                chunk=idle_chunk,
                duration_s=self.config.duration_s,
                total_rt_requests=num_slacks,
                completed_rt_requests=num_slacks,
                rt_slo_violations=rt_violations,
                rt_slacks_s=list(pure_slacks),
            )
            return OptimizationResult(
                best_result=infeasible,
                all_results=[infeasible],
                pure_rt_slacks_s=pure_slacks,
                decode_only_latency_s=decode_only_latency_s,
                max_candidate_latency_s=None,
                rt_violation_rate=rt_violation_rate,
                rt_infeasible=True,
                evaluated_candidates=1,
                stop_reason="rt_infeasible",
                budget_history_s=[0.0],
            )

        simulator = PrefillSimulator(
            rt_requests=rt_requests,
            ttft_slo_s=self.config.slo.ttft_slo_s,
            duration_s=self.config.duration_s,
            be_config=self.config.be,
            stop_after_last_rt=self.config.run_to_completion,
        )

        # Start at the minimum pure-RT slack, then use feedback from each
        # replay to choose the next budget. A tolerated negative slack starts
        # at zero because an execution budget cannot be negative. Without RT
        # history there is no slack feedback; retain a decode-only fallback.
        decode_capacity = self.config.be.decode_batch_size
        initial_budget_s = max(0.0, min(pure_slacks, default=decode_only_latency_s))
        budget_s = initial_budget_s

        all_results: List[SimulationResult] = []
        best: Optional[SimulationResult] = None
        previous_throughput: Optional[float] = None
        stop_reason = "iteration_limit"
        for _ in range(max_iterations):
            if budget_s == 0.0:
                chunk = BEChunk(0.0, 0.0, 0, 0)
            else:
                balanced = self.latency_predictor.balance_decode_batch(
                    budget_s, self.config.be, max_decode_batch=decode_capacity)
                chunk = BEChunk(
                    target_latency_s=budget_s,
                    latency_s=balanced["latency_s"],
                    decode_batch_size=int(balanced["decode_batch"]),
                    prefill_chunk_size=int(balanced["prefill_chunk"]),
                )
            result = simulator.run(chunk)
            if (not math.isfinite(result.saturation_req_per_s)
                    or not all(math.isfinite(s) for s in result.rt_slacks_s)):
                raise ValueError("simulated throughput and RT slacks must be finite")
            all_results.append(result)
            if best is None or self._is_better(result, best):
                best = result

            throughput = result.saturation_req_per_s
            if previous_throughput is not None and throughput <= previous_throughput:
                stop_reason = "throughput_not_improved"
                break
            positive_slacks = [s for s in result.rt_slacks_s if s > 0.0]
            if not positive_slacks:
                stop_reason = "no_positive_slack"
                break
            next_budget_s = budget_s + min(positive_slacks)
            if not math.isfinite(next_budget_s) or next_budget_s <= budget_s:
                stop_reason = "no_budget_progress"
                break
            previous_throughput = throughput
            budget_s = next_budget_s

        assert best is not None
        evaluated_candidates = len(all_results)
        budget_history_s = [r.chunk.target_latency_s for r in all_results]
        max_latency_s = all_results[-1].chunk.target_latency_s

        if max_candidates > 0:
            all_results = self._compact_results(all_results, max_candidates, best)

        return OptimizationResult(
            best_result=best,
            all_results=all_results,
            pure_rt_slacks_s=pure_slacks,
            decode_only_latency_s=decode_only_latency_s,
            max_candidate_latency_s=max_latency_s,
            rt_violation_rate=rt_violation_rate,
            rt_infeasible=False,
            initial_budget_s=initial_budget_s,
            evaluated_candidates=evaluated_candidates,
            stop_reason=stop_reason,
            budget_history_s=budget_history_s,
        )

    def _compact_results(
        self,
        results: List[SimulationResult],
        max_results: int,
        best: SimulationResult,
    ) -> List[SimulationResult]:
        if len(results) <= max_results:
            return results

        keep = {0, len(results) - 1, results.index(best)}
        for idx in self._interesting_result_indices(results):
            keep.add(idx)

        if len(keep) < max_results:
            stride = max(1, len(results) // max_results)
            keep.update(range(0, len(results), stride))

        selected = sorted(keep)[:max_results]
        if results.index(best) not in selected:
            selected[-1] = results.index(best)
            selected.sort()
        return [results[idx] for idx in selected]

    def _interesting_result_indices(
        self,
        results: List[SimulationResult],
    ) -> List[int]:
        indices = []
        last_violations = None
        last_preemptions = None
        for idx, result in enumerate(results):
            if result.rt_slo_violations != last_violations:
                indices.append(idx)
                last_violations = result.rt_slo_violations
            if result.preemptions != last_preemptions:
                indices.append(idx)
                last_preemptions = result.preemptions
        return indices

    def _is_better(
        self,
        left: SimulationResult,
        right: SimulationResult,
    ) -> bool:
        # A candidate is "acceptable" when its RT TTFT-SLO violation rate stays
        # within the configured tolerance. Acceptable candidates beat those that
        # exceed the tolerance; ties are then broken by higher BE saturation
        # throughput, then fewer preemptions. Only when both candidates exceed
        # the tolerance do we fall back to minimizing the raw violation count.
        threshold = self.config.slo.max_rt_violation_rate
        left_ok = left.rt_violation_rate <= threshold
        right_ok = right.rt_violation_rate <= threshold
        if left_ok != right_ok:
            return left_ok
        if not left_ok and left.rt_slo_violations != right.rt_slo_violations:
            return left.rt_slo_violations < right.rt_slo_violations
        if left.saturation_req_per_s != right.saturation_req_per_s:
            return left.saturation_req_per_s > right.saturation_req_per_s
        return left.preemptions < right.preemptions
