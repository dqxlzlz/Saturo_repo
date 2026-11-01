from types import SimpleNamespace

import pytest

from saturo.runtime.scheduler import SaturoGlobalScheduler
from saturo.runtime.worker import RayEngineWorker, WorkerState
from saturo.runtime.workload import DECODE, RT, SaturoRequest
from saturo.saturation.decode_saturation_core import (
    DefaultDecodeSaturationStateProvider)


def provider():
    result = DefaultDecodeSaturationStateProvider(load_predictor=False)
    result.model_len_predictor = SimpleNamespace(
        model_type='qrf', predict_remaining=lambda **kwargs: 128)
    return result


def request(source_id, timestamp, *, attempt=1, prompt_len=16):
    return SimpleNamespace(
        request_id=f'{source_id}:d{attempt}', arrival_time=150.0,
        sampling_params=SimpleNamespace(extra_args={
            'saturo_source_request_id': source_id,
            'saturo_source_arrival_time_s': timestamp}),
        get_seqs=lambda: [SimpleNamespace(
            get_prompt_len=lambda: prompt_len, get_output_len=lambda: 8)])


def test_transport_batching_and_completion_order_do_not_change_source_cadence():
    ordered, batched = provider(), provider()
    times = [100.0 + i * .25 for i in range(16)]
    for i, stamp in enumerate(times):
        ordered.observe_rt_arrival(request(f'rt{i}', stamp), stamp + 1)
    # P finishes out of order and sends several requests in each RPC. Local
    # submission clocks are nearly identical, but source arrivals are not.
    delivery = list(range(0, 16, 2)) + list(reversed(range(1, 16, 2)))
    for i in delivery:
        batched.observe_rt_arrival(request(f'rt{i}', times[i]), 110 + i / 1000)
    assert ordered._arima_arrival_predictor._inter_arrivals_s == [.25] * 15
    assert batched._arima_arrival_predictor._inter_arrivals_s == [.25] * 15
    first = list(ordered.get_future_arrivals(now_s=111, window_s=60))
    second = list(batched.get_future_arrivals(now_s=111, window_s=60))
    assert len(first) == len(second) == 240
    assert [r.arrival_time_s for r in first] == [r.arrival_time_s for r in second]


def test_retry_attempts_do_not_add_arrivals_or_change_source_timing():
    state = provider()
    for i in range(8):
        state.observe_rt_arrival(request(f'rt{i}', 100 + .5 * i), 110)
    before = list(state._arima_arrival_predictor._inter_arrivals_s)
    state.observe_rt_arrival(request('rt3', 101.5, attempt=9), 120)
    assert state._arima_arrival_predictor._inter_arrivals_s == before
    assert state._prompt_len_count == 8


def test_bounded_history_keeps_latest_source_times_independent_of_delivery_order():
    ascending, descending = provider(), provider()
    for state in (ascending, descending):
        state._arima_arrival_predictor.max_history_intervals = 3
    for state, ids in [(ascending, range(12)), (descending, reversed(range(12)))]:
        for i in ids:
            state.observe_rt_arrival(
                request(f'rt{i}', 100 + .5 * i, prompt_len=16 + i), 120)
        # This old retry must stay evicted rather than becoming a fresh event.
        state.observe_rt_arrival(request('rt0', 100, attempt=99), 130)
        assert len(state._rt_arrival_observations) == 4
        assert state._arima_arrival_predictor._inter_arrivals_s == [.5] * 3
        assert state._arima_arrival_predictor._last_arrival_time_s == 105.5
        assert state._avg_prompt_len() == 25.5
    assert ascending._rt_arrival_observations == descending._rt_arrival_observations


@pytest.mark.parametrize('timestamp', [float('nan'), float('inf'), 200.0])
def test_source_arrival_must_be_finite_and_already_observed(timestamp):
    state = provider()
    with pytest.raises(ValueError, match='source arrival'):
        state.observe_rt_arrival(request('rt0', timestamp), now_s=150)
    assert state._prompt_len_count == 0


@pytest.mark.parametrize('skip_marker', [False, True])
def test_warmup_does_not_seed_traffic_arrivals_or_output_lengths(skip_marker):
    state = provider()
    warmup = request('warmup', 100)
    if skip_marker:
        warmup.sampling_params.extra_args['saturo_skip_traffic_observation'] = True
    else:
        warmup.request_id = '__warmup__:0'
    state.observe_rt_arrival(warmup, 150)
    state.observe_rt_finish(warmup)
    assert state._arima_arrival_predictor._inter_arrivals_s == []
    assert state._arima_arrival_predictor._last_arrival_time_s is None
    assert state._prompt_len_count == state._output_len_count == 0


def test_external_requests_without_source_metadata_still_use_observed_clock():
    state = provider()
    for i in range(4):
        raw = request(f'rt{i}', 100)
        raw.sampling_params.extra_args = {}
        state.observe_rt_arrival(raw, 110 + i * .5)
    assert state._arima_arrival_predictor._inter_arrivals_s == [.5] * 3


def test_source_metadata_does_not_rewrite_execution_arrival_time():
    state = provider()
    seq_group = request('rt', 100)
    state.observe_rt_arrival(seq_group, 150)
    assert seq_group.arrival_time == 150
    assert state._arima_arrival_predictor._last_arrival_time_s == 100


def test_global_handoff_keeps_original_source_time_across_decode_retries():
    scheduler = SaturoGlobalScheduler.__new__(SaturoGlobalScheduler)
    scheduler._source_arrival_epoch_s = 1000.0
    scheduler._elapsed = lambda: 20.0
    req = SaturoRequest('rt0', RT, 16, 8, 10.0, observed_tokens=1)
    worker = WorkerState(0, DECODE, 'local', 'tp1_pp1', 1, 1, None, ready=True)
    scheduler._assign(worker, req, DECODE)
    first = worker.pending_commands[-1]
    req.active_attempt_id = None
    scheduler._elapsed = lambda: 40.0
    scheduler._assign(worker, req, DECODE)
    second = worker.pending_commands[-1]
    assert first['attempt_id'] != second['attempt_id']
    assert first['source_request_id'] == second['source_request_id'] == 'rt0'
    assert first['source_arrival_time_s'] == second['source_arrival_time_s'] == 1010


def test_worker_carries_source_metadata_beside_direct_decode_and_reservation():
    worker = RayEngineWorker.__new__(RayEngineWorker)
    worker.worker_role = DECODE
    worker._use_dummy_kv = False
    worker.rt_prefill_deadline_s = 1
    worker.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
    _, params, _ = worker._request_inputs(dict(
        request_id='rt0', kind=RT, phase=DECODE, max_tokens=7, prompt_len=17,
        source_request_id='rt0', source_arrival_time_s=1010,
        reservation_id='rt0:p1'))
    assert params.extra_args == dict(
        saturo_source_request_id='rt0', saturo_source_arrival_time_s=1010,
        saturo_direct_decode=True, saturo_decode_reservation_id='rt0:p1')


def test_worker_marks_rt_warmup_for_traffic_exclusion():
    worker = RayEngineWorker.__new__(RayEngineWorker)
    worker.worker_role = DECODE
    worker._use_dummy_kv = False
    worker.rt_prefill_deadline_s = 1
    worker.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
    _, params, _ = worker._request_inputs(dict(
        request_id='__warmup__:0', kind=RT, phase='throughput',
        max_tokens=2, prompt_len=16))
    assert params.extra_args['saturo_skip_traffic_observation'] is True
