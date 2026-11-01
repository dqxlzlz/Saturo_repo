"""Event-driven simulator for P-pool BE chunks with agile preemption."""

from collections import deque
from typing import Deque, List, Optional

from .config import BEConfig
from .request import BEChunk, RTRequest, SimulationResult


class PrefillSimulator:
    """Replay one P-pool instance over a fixed time window."""

    def __init__(
        self,
        rt_requests: List[RTRequest],
        ttft_slo_s: float,
        duration_s: float,
        be_config: BEConfig,
        stop_after_last_rt: bool = False,
    ):
        self.rt_requests = sorted(rt_requests, key=lambda r: r.arrival_time_s)
        self.ttft_slo_s = ttft_slo_s
        self.duration_s = duration_s
        self.be_config = be_config
        # Fixed-count replay: ignore the duration window, serve every RT
        # request, stop right after the last one completes (BE only fills the
        # gaps inside the RT span) and normalize throughput by the makespan.
        self.stop_after_last_rt = stop_after_last_rt

    def run(self, chunk: BEChunk) -> SimulationResult:
        result = SimulationResult(
            chunk=chunk,
            duration_s=self.duration_s,
            total_rt_requests=len(self.rt_requests),
        )
        if self.duration_s <= 0:
            return result

        current_time_s = 0.0
        next_arrival_idx = 0
        rt_queue: Deque[RTRequest] = deque()

        while self._has_work_in_window(current_time_s, next_arrival_idx, rt_queue):
            next_arrival_idx = self._enqueue_arrived(
                next_arrival_idx, current_time_s, rt_queue)

            if rt_queue:
                req = rt_queue.popleft()
                current_time_s = self._serve_rt(req, current_time_s, result)
                continue

            next_req = self._peek_next_request(next_arrival_idx)
            if next_req is None:
                if self.stop_after_last_rt:
                    break
                if not self._run_be_chunk_to_window_end(
                    chunk, current_time_s, result):
                    break
                current_time_s += chunk.latency_s
                continue

            if not self.stop_after_last_rt and current_time_s >= self.duration_s:
                break

            if current_time_s < next_req.arrival_time_s:
                current_time_s, next_arrival_idx = self._run_be_until_arrival(
                    chunk=chunk,
                    current_time_s=current_time_s,
                    next_arrival_idx=next_arrival_idx,
                    rt_queue=rt_queue,
                    result=result,
                )
                continue

            next_arrival_idx = self._enqueue_arrived(
                next_arrival_idx, current_time_s, rt_queue)

        self._finalize_throughput(result, current_time_s)
        return result

    def _run_be_until_arrival(
        self,
        chunk: BEChunk,
        current_time_s: float,
        next_arrival_idx: int,
        rt_queue: Deque[RTRequest],
        result: SimulationResult,
    ):
        if chunk.latency_s <= 0.0:
            next_req = self._peek_next_request(next_arrival_idx)
            if next_req is None:
                return self.duration_s, next_arrival_idx
            horizon = (
                next_req.arrival_time_s if self.stop_after_last_rt
                else min(next_req.arrival_time_s, self.duration_s))
            idle = max(0.0, horizon - current_time_s)
            result.idle_time_s += idle
            return current_time_s + idle, next_arrival_idx

        be_start_s = current_time_s
        be_end_s = current_time_s + chunk.latency_s
        next_req = self._peek_next_request(next_arrival_idx)

        if next_req is None or next_req.arrival_time_s >= be_end_s:
            if self.stop_after_last_rt or be_end_s <= self.duration_s:
                self._complete_be_chunk(chunk, result)
                return be_end_s, next_arrival_idx
            result.idle_time_s += max(0.0, self.duration_s - current_time_s)
            return self.duration_s, next_arrival_idx

        queued_rt_runtime_s = 0.0
        while next_arrival_idx < len(self.rt_requests):
            req = self.rt_requests[next_arrival_idx]
            if req.arrival_time_s >= be_end_s:
                break

            if (not self.stop_after_last_rt
                    and req.arrival_time_s > self.duration_s):
                break

            remaining_be_s = max(0.0, be_end_s - req.arrival_time_s)
            if (
                remaining_be_s + queued_rt_runtime_s + req.runtime_s
                > self.ttft_slo_s
            ):
                result.preemptions += 1
                result.wasted_be_time_s += max(0.0, req.arrival_time_s - be_start_s)
                rt_queue.append(req)
                next_arrival_idx += 1
                return req.arrival_time_s, next_arrival_idx

            rt_queue.append(req)
            queued_rt_runtime_s += req.runtime_s
            next_arrival_idx += 1

        self._complete_be_chunk(chunk, result)
        return be_end_s, next_arrival_idx

    def _run_be_chunk_to_window_end(
        self,
        chunk: BEChunk,
        current_time_s: float,
        result: SimulationResult,
    ) -> bool:
        if chunk.latency_s <= 0.0:
            result.idle_time_s += max(0.0, self.duration_s - current_time_s)
            return False
        if current_time_s + chunk.latency_s > self.duration_s:
            result.idle_time_s += max(0.0, self.duration_s - current_time_s)
            return False
        self._complete_be_chunk(chunk, result)
        return True

    def _serve_rt(
        self,
        req: RTRequest,
        current_time_s: float,
        result: SimulationResult,
    ) -> float:
        start_s = max(current_time_s, req.arrival_time_s)
        queue_s = start_s - req.arrival_time_s
        slack_s = self.ttft_slo_s - queue_s - req.runtime_s
        result.rt_slacks_s.append(slack_s)
        if slack_s < 0.0:
            result.rt_slo_violations += 1
        result.completed_rt_requests += 1
        return start_s + req.runtime_s

    def _complete_be_chunk(self, chunk: BEChunk, result: SimulationResult) -> None:
        result.completed_be_chunks += 1
        result.useful_be_time_s += chunk.latency_s
        result.total_prefill_tokens += chunk.useful_prefill_tokens
        result.total_decode_tokens += chunk.useful_decode_tokens

    def _enqueue_arrived(
        self,
        next_arrival_idx: int,
        current_time_s: float,
        rt_queue: Deque[RTRequest],
    ) -> int:
        while next_arrival_idx < len(self.rt_requests):
            req = self.rt_requests[next_arrival_idx]
            if req.arrival_time_s > current_time_s:
                break
            if self.stop_after_last_rt or req.arrival_time_s <= self.duration_s:
                rt_queue.append(req)
            next_arrival_idx += 1
        return next_arrival_idx

    def _peek_next_request(self, next_arrival_idx: int) -> Optional[RTRequest]:
        if next_arrival_idx >= len(self.rt_requests):
            return None
        return self.rt_requests[next_arrival_idx]

    def _has_work_in_window(
        self,
        current_time_s: float,
        next_arrival_idx: int,
        rt_queue: Deque[RTRequest],
    ) -> bool:
        if rt_queue:
            return True
        if self.stop_after_last_rt:
            return self._peek_next_request(next_arrival_idx) is not None
        if current_time_s < self.duration_s:
            return True
        next_req = self._peek_next_request(next_arrival_idx)
        return next_req is not None and next_req.arrival_time_s <= self.duration_s

    def _finalize_throughput(
        self,
        result: SimulationResult,
        end_time_s: float,
    ) -> None:
        if self.stop_after_last_rt:
            window_s = max(end_time_s, 1e-9)
            result.duration_s = window_s
        else:
            window_s = max(self.duration_s, 1e-9)
        result.prefill_req_per_s = (
            result.total_prefill_tokens
            / max(1, self.be_config.avg_be_length)
            / window_s
        )
        result.decode_req_per_s = (
            result.total_decode_tokens
            / max(1, self.be_config.avg_be_output_len)
            / window_s
        )
        result.saturation_req_per_s = min(
            result.prefill_req_per_s, result.decode_req_per_s)
        result.be_token_per_s = (
            result.total_prefill_tokens + result.total_decode_tokens) / window_s
