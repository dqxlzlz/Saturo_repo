"""Small control-plane policy for borrowing and reclaiming P/D/T workers."""

from __future__ import annotations

import math
from collections import OrderedDict, deque
from dataclasses import replace
from pathlib import Path

from saturo.runtime.workload import DECODE, PREFILL, RT, TPUT


ROLE_POOL = {PREFILL: "P", DECODE: "D", TPUT: "T"}


class AdaptiveRoleSwitcher:
    """Use RT token/KV estimates and cached timing/capacity profiles.

    No request migration, TP/PP changes, or background profile collection.
    Reclamation only considers RT-idle workers and keeps one P and one D.
    """

    def __init__(self, scheduler, profiles):
        self.scheduler = scheduler
        self.profiles = profiles
        self.interval_s = getattr(scheduler.args, "role_switch_interval_s", 60)
        self.window_s = getattr(scheduler.args, "role_switch_window_s", 60)
        self.arrivals = deque()
        self.last_rebalance_s = 0.0
        self.events = []
        self._predictors = {}
        self._length_predictor = None
        self._remaining_cache = OrderedDict()
        self._remaining_cache_bytes = 0
        self.admission_prediction_calls = 0
        self.admission_prediction_cache_hits = 0

    def observe_arrival(self, now_s):
        self.arrivals.append(now_s)

    def current_rps(self, now_s):
        while self.arrivals and self.arrivals[0] <= now_s - self.window_s:
            self.arrivals.popleft()
        if now_s < self.window_s:
            return max(self.scheduler.args.target_rps,
                       len(self.arrivals) / max(now_s, 1.0))
        return len(self.arrivals) / self.window_s

    def rt_requests(self, worker):
        requests = []
        for attempt, request_id in worker.active_attempts.items():
            req = self.scheduler._request_by_id(request_id)
            if (req is not None and req.kind == RT and not req.is_finished
                    and req.active_attempt_id == attempt):
                requests.append(req)
        for reservation in worker.decode_reservations:
            request_id = reservation.rsplit(":prefill:", 1)[0]
            req = self.scheduler._request_by_id(request_id)
            if (req is not None and not req.is_finished
                    and req.decode_reservation_id == reservation
                    and not any(item is req for item in requests)):
                requests.append(req)
        return requests

    def load(self, worker, role):
        requests = self.rt_requests(worker)
        if role == PREFILL:
            # Conservatively retain all prompt tokens until prefill finishes.
            return sum(req.prompt_len for req in requests)
        return sum(math.ceil((req.decode_prompt_len + 1) / worker.block_size)
                   for req in requests)

    def _profile(self, worker, role):
        return self.profiles.pools.get(ROLE_POOL[role], {}).get(
            worker.config_name)

    def _predictor(self, worker):
        path = worker.runtime_profile_path
        if path not in self._predictors:
            predictor = None
            if path and Path(path).is_file():
                from saturo.predictors.time_predictor import TimePredictor
                try:
                    predictor = TimePredictor(path)
                    if not predictor.available:
                        predictor = None
                except (OSError, ValueError, KeyError):
                    pass
            self._predictors[path] = predictor
        return self._predictors[path]

    def _predicted_remaining(self, requests):
        """Use observed content/progress only, never a trace's future output.

        Memoize unchanged observations before TF-IDF extraction. The cache is
        bounded independently of the full workload and invalidates on changed
        prompt/prefix content, progress, quantile or model training revision.
        """
        if self._length_predictor is None:
            from saturo.predictors.qrf import load_required_qrf, validate_qrf_workload
            self._length_predictor = load_required_qrf(
                getattr(self.scheduler.args, "model_len_predictor", None))
            validate_qrf_workload(self._length_predictor, self.scheduler.args)
        predictor = self._length_predictor
        eta = float(getattr(self.scheduler.args, "resource_decode_model_len_quantile", .95))
        if not math.isfinite(eta) or not 0 < eta <= 1:
            raise ValueError("Global admission QRF quantile must be in (0,1]")
        content = getattr(getattr(predictor, "config", None), "feature_mode", "len") != "len"
        results, missing, features = [None] * len(requests), [], []
        for index, req in enumerate(requests):
            prompt = tuple(req.prompt_token_ids) if content and req.prompt_token_ids is not None else None
            generated = tuple(req.generated_token_ids) if content else None
            if content and (prompt is None or len(prompt) != req.prompt_len
                            or len(generated) != req.observed_tokens):
                raise ValueError("Global admission content QRF requires actual prompt and observed-prefix token IDs")
            key = (req.prompt_len, req.observed_tokens, prompt, generated, eta,
                   getattr(predictor, "_prediction_revision", 0))
            cached = self._remaining_cache.get(req.request_id)
            if cached is not None and cached[0] is predictor and cached[1] == key:
                results[index] = cached[2]
                self._remaining_cache.move_to_end(req.request_id)
                self.admission_prediction_cache_hits += 1
                continue
            missing.append((index, req.request_id, key))
            features.append(dict(prompt_len=req.prompt_len, generated_len=req.observed_tokens,
                                 prompt_token_ids=prompt, generated_token_ids=generated))
        if missing:
            batch = getattr(predictor, "predict_remaining_batch", None)
            predictions = (list(batch(features, eta=eta)) if callable(batch) else
                           [predictor.predict_remaining(eta=eta, **row) for row in features])
            if len(predictions) != len(missing):
                raise RuntimeError("Global admission QRF returned an invalid prediction count")
            for (index, request_id, key), prediction in zip(missing, predictions):
                if not math.isfinite(float(prediction)):
                    raise ValueError("Global admission QRF returned a nonfinite prediction")
                remaining = max(1, math.ceil(float(prediction)))
                results[index] = remaining
                self.admission_prediction_calls += 1
                old = self._remaining_cache.pop(request_id, None)
                if old is not None:
                    self._remaining_cache_bytes -= old[3]
                size = 128 + 8 * (len(key[2] or ()) + len(key[3] or ()))
                if size <= 4 * 1024 * 1024:
                    self._remaining_cache[request_id] = (predictor, key, remaining, size)
                    self._remaining_cache_bytes += size
                    while len(self._remaining_cache) > 1024 or self._remaining_cache_bytes > 4 * 1024 * 1024:
                        _, evicted = self._remaining_cache.popitem(last=False)
                        self._remaining_cache_bytes -= evicted[3]
        return results

    def admission_diagnostics(self):
        return dict(length_source="qrf_prediction", uses_trace_output_len=False,
                    feature_mode=getattr(getattr(self._length_predictor, "config", None), "feature_mode", None),
                    quantile=float(getattr(self.scheduler.args, "resource_decode_model_len_quantile", .95)),
                    prediction_calls=self.admission_prediction_calls,
                    cache_hits=self.admission_prediction_cache_hits,
                    cache_entries=len(self._remaining_cache), cache_bytes=self._remaining_cache_bytes)

    def can_accept(self, worker, role, request, *, check_slo=True):
        profile = self._profile(worker, role)
        if profile is None or not profile.mu or profile.mu <= 0:
            return False
        requests = self.rt_requests(worker)
        if not any(req is request for req in requests):
            requests.append(request)
        if worker.max_num_seqs and len(requests) > worker.max_num_seqs:
            return False
        if role == PREFILL:
            lengths = [req.prompt_len + 1 for req in requests]
        else:
            # Generation budgets remain stopping controls, not forecasts.
            lengths = [req.decode_prompt_len + predicted
                       for req, predicted in zip(requests, self._predicted_remaining(requests))]
        if worker.max_model_len and max(lengths) > worker.max_model_len:
            return False
        blocks = sum(math.ceil(length / worker.block_size)
                     for length in lengths)
        watermark = self.scheduler.args.resource_memory_watermark
        if blocks > math.floor(worker.num_gpu_blocks * watermark):
            return False
        if not check_slo:
            return True
        predictor = self._predictor(worker)
        if role == PREFILL:
            slack_ms = min((req.arrival_offset_s
                            + self.scheduler.args.rt_ttft_slo_ms / 1000
                            - self.scheduler._elapsed()) * 1000
                           for req in requests)
            if predictor is None:
                runtime_ms = len(requests) / profile.mu * 1000
            else:
                runtime_ms = sum(predictor.predict_forward_ms(
                    [(req.prompt_len, req.prompt_len)], []) for req in requests)
            # One already-running BE chunk may need to complete first.
            if len(worker.active_attempts) > len(requests) - 1:
                runtime_ms += self.scheduler._resolve_p_pool_be_chunk_ms(
                    worker.config_name)
            return runtime_ms <= slack_ms
        if predictor is None:
            # Missing timing data: serialize RT decode rather than guessing
            # safe batch size. The engine's RT priority still applies.
            return len(requests) == 1
        runtime_ms = predictor.predict_forward_ms(
            [], [(1, req.decode_prompt_len) for req in requests])
        return runtime_ms <= self.scheduler.args.rt_tbt_slo_ms

    def select(self, role, request):
        workers = [w for w in self.scheduler._live_workers() if w.ready]
        eligible = [w for w in workers if w.role == role
                    and self.can_accept(w, role, request)]
        if eligible:
            return min(eligible,
                       key=lambda w: (self.load(w, role), w.worker_id))
        candidates = [w for w in workers if w.role == TPUT
                      and w.role_switching_enabled
                      and self.can_accept(w, role, request)]
        candidates.sort(key=lambda w: (len(w.active_attempts), w.worker_id))
        for worker in candidates:
            if self.scheduler._switch_worker_role(
                    worker, role, borrowed=True, reason="rt_burst"):
                return worker
        if (role == PREFILL and self.scheduler._elapsed()
                >= request.arrival_offset_s
                + self.scheduler.args.rt_ttft_slo_ms / 1000):
            # An already-missed TTFT must not leave a request stuck forever.
            # Serve best effort while still enforcing memory/context limits.
            candidates = [w for w in workers if w.role == role
                          and self.can_accept(w, role, request,
                                              check_slo=False)]
            if candidates:
                return min(candidates, key=lambda w: self.load(w, role))
            # P may be absent after a failure or in a BE-only starting layout.
            # Expired deadlines must not prevent a memory-safe T borrow.
            candidates = [w for w in workers if w.role == TPUT
                          and w.role_switching_enabled
                          and self.can_accept(w, role, request,
                                              check_slo=False)]
            for worker in candidates:
                if self.scheduler._switch_worker_role(
                        worker, role, borrowed=True, reason="rt_burst"):
                    return worker
        return None

    def _throughput(self, workers, rps):
        throughput = 0.0
        for role in (PREFILL, DECODE, TPUT):
            pool = [w for w in workers if w.role == role]
            if role != TPUT and not pool:
                return None
            profiles = [self._profile(w, role) for w in pool]
            if any(profile is None for profile in profiles):
                return None
            capacity = sum(profile.mu or 0 for profile in profiles)
            if role != TPUT and (capacity <= 0 or capacity + 1e-9 < rps):
                return None
            for profile in profiles:
                # Capacity-weighted load supports mixed TP/PP configurations
                # after borrowing, and reduces to W/n for homogeneous pools.
                load = 0 if role == TPUT else rps * (profile.mu or 0) / capacity
                value = profile.phi(load)
                if value is None:
                    return None
                throughput += value
        return throughput

    def maintain(self):
        now = self.scheduler._elapsed()
        # Return borrowed workers as soon as their RT attempts drain. Pending
        # RT of the same role keeps them available for the next loop turn.
        for worker in self.scheduler._live_workers():
            pending = (self.scheduler.pending_prefill if worker.role == PREFILL
                       else self.scheduler.pending_decode)
            if (worker.borrowed_role and not pending
                    and not worker.decode_reservations
                    and not self.rt_requests(worker)):
                self.scheduler._switch_worker_role(
                    worker, TPUT, borrowed=False, reason="rt_drained")
        if now - self.last_rebalance_s < self.interval_s:
            return
        self.last_rebalance_s = now
        if self.scheduler.pending_prefill or self.scheduler.pending_decode:
            return
        workers = [w for w in self.scheduler._live_workers() if w.ready]
        rps = self.current_rps(now)
        before = self._throughput(workers, rps)
        if before is None:
            return
        # One idle native instance per control interval avoids oscillation.
        choices = []
        for worker in workers:
            if (worker.role == TPUT or worker.borrowed_role
                    or not worker.role_switching_enabled
                    or worker.decode_reservations
                    or self.rt_requests(worker)):
                continue
            after_workers = [replace(w, role=TPUT) if w is worker else w
                             for w in workers]
            after = self._throughput(after_workers, rps)
            if after is not None and after > before + 1e-9:
                choices.append((after - before, worker))
        choices.sort(key=lambda item: (-item[0], item[1].worker_id))
        for _, worker in choices:
            if self.scheduler._switch_worker_role(
                    worker, TPUT, borrowed=False, reason="be_gain"):
                break
