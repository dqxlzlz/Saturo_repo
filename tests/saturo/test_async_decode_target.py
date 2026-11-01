import threading
import time
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from saturo.saturation.async_target import (
    ArrivalState, AsyncTargetOptimizer, TargetSnapshot)
from saturo.saturation.decode_saturation_core import DecodeSaturationPolicy
from saturo.saturation.types import DecodeSaturationDecision


def snapshot(version=1, epoch=1):
    arrivals = ArrivalState(None, (), None, 4, 1024, .001, 0., 0, 0., 0)
    return TargetSnapshot(version, epoch, 10., (), arrivals, 0, 0, 0,
                          time.perf_counter())


def decision():
    return DecodeSaturationDecision(True, 100., 100., 5., 1, 1, 0, 99, 100)


def test_optimizer_runs_one_active_and_only_latest_pending():
    optimizer = AsyncTargetOptimizer()
    entered, release = threading.Event(), threading.Event()
    versions = []

    def compute(item):
        versions.append(item.version)
        if item.version == 1:
            entered.set()
            assert release.wait(2)
        return decision(), ()

    try:
        optimizer.submit(snapshot(1), compute)
        assert entered.wait(1)
        optimizer.submit(snapshot(2), compute)
        optimizer.submit(snapshot(3), compute)
        stats = optimizer.diagnostics()
        assert stats['active'] == stats['pending'] == 1
        assert stats['coalesced'] == 1 and optimizer.poll() is None
        release.set()
        result = optimizer.wait(2)
        assert result.version == 3 and versions == [1, 3]
        assert optimizer.diagnostics()['discarded'] == 1
    finally:
        release.set()
        assert optimizer.close(2)


def test_optimizer_catches_current_error_without_retaining_traceback():
    optimizer = AsyncTargetOptimizer()
    try:
        def fail(item):
            raise ValueError('bad forecast')
        optimizer.submit(snapshot(), fail)
        result = optimizer.wait(1)
        assert result.error_type == 'ValueError'
        assert result.error_message == 'bad forecast' and result.decision is None
    finally:
        assert optimizer.close(1)


def test_close_discards_pending_and_late_result_with_bounded_join():
    optimizer = AsyncTargetOptimizer()
    entered, release = threading.Event(), threading.Event()
    called = []

    def compute(item):
        called.append(item.version)
        entered.set()
        assert release.wait(2)
        return decision(), ()

    optimizer.submit(snapshot(), compute)
    assert entered.wait(1)
    optimizer.submit(snapshot(2), compute)
    assert not optimizer.close(0)
    release.set()
    assert optimizer.close(2)
    assert called == [1] and optimizer.poll() is None
    assert not optimizer.diagnostics()['thread_alive']
    with pytest.raises(RuntimeError, match='closed'):
        optimizer.submit(snapshot(3), compute)


def test_epoch_invalidation_discards_an_active_old_result():
    optimizer = AsyncTargetOptimizer()
    entered, release = threading.Event(), threading.Event()

    def compute(item):
        entered.set()
        assert release.wait(2)
        return decision(), ()

    try:
        optimizer.submit(snapshot(), compute)
        assert entered.wait(1)
        optimizer.invalidate(2, 2)
        release.set()
        assert optimizer.wait(.05) is None
        assert optimizer.diagnostics()['discarded'] == 1
    finally:
        release.set()
        optimizer.close(2)


def test_stale_background_error_cannot_kill_newest_valid_request():
    optimizer = AsyncTargetOptimizer()
    entered, release = threading.Event(), threading.Event()
    def compute(item):
        if item.version == 1:
            entered.set()
            assert release.wait(2)
            raise ValueError('obsolete model state')
        return decision(), ()
    try:
        optimizer.submit(snapshot(1), compute)
        assert entered.wait(1)
        optimizer.submit(snapshot(2), compute)
        release.set()
        result = optimizer.wait(2)
        assert result.version == 2 and result.error_type is None
        assert optimizer.diagnostics()['discarded'] == 1
    finally:
        release.set()
        optimizer.close(2)


def test_optimizer_rejects_nonincreasing_versions_without_replacing_work():
    optimizer = AsyncTargetOptimizer()
    try:
        optimizer.invalidate(2, 3)
        with pytest.raises(ValueError, match='increase'):
            optimizer.submit(snapshot(2, 1), lambda item: (decision(), ()))
        with pytest.raises(ValueError, match='advance'):
            optimizer.invalidate(2, 3)
        assert optimizer.diagnostics()['submitted'] == 0
    finally:
        optimizer.close(1)


class Controls:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.rows = []
        self.threads = []
        self.error = None
        self.copies = 0


class PrivateQRF:
    model_type = 'qrf'

    def __init__(self, controls, mode='len', private=False):
        self.controls = controls
        self.config = SimpleNamespace(feature_mode=mode)
        self.private = private

    def __deepcopy__(self, memo):
        self.controls.copies += 1
        return type(self)(self.controls, self.config.feature_mode, True)

    def predict_remaining_batch(self, rows, eta=.95):
        assert self.private, 'QRF must not execute on the serving provider'
        self.controls.rows.extend(dict(row) for row in rows)
        self.controls.threads.append(threading.get_ident())
        self.controls.entered.set()
        assert self.controls.release.wait(2)
        if self.controls.error is not None:
            raise self.controls.error
        return [8] * len(rows)

    def predict_remaining(self, **row):
        return self.predict_remaining_batch([row])[0]


class Latency:
    def predict_forward_ms(self, prefill, decode):
        return 5. + len(decode) * .1 + sum(context for _, context in decode) * .001


class Sequence:
    def __init__(self, prompt=16, generated=2):
        self.prompt = list(range(prompt))
        self.generated = list(range(generated))

    def get_prompt_len(self):
        return len(self.prompt)

    def get_output_len(self):
        return len(self.generated)

    def get_len(self):
        return self.get_prompt_len() + self.get_output_len()

    def get_prompt_token_ids(self):
        return self.prompt

    def get_output_token_ids(self):
        return self.generated


class Group:
    def __init__(self, name='rt', seq=None):
        self.request_id = name
        self.seq = seq or Sequence()
        self.arrival_time = 1.
        self.is_realtime = True
        self.sampling_params = SimpleNamespace(extra_args={})

    def get_seqs(self):
        return [self.seq]

    def is_prefill(self):
        return False


@pytest.fixture
def policy_factory(monkeypatch):
    policies, controls_list = [], []

    def make(mode='len', **protocol):
        controls = Controls()
        controls_list.append(controls)
        model = PrivateQRF(controls, mode)
        monkeypatch.setattr('saturo.predictors.qrf.load_required_qrf', lambda *a: model)
        config = SimpleNamespace(enable_decode_saturation=True,
            decode_saturation_tbt_ms=100., decode_saturation_lookahead_s=1.)
        policy = DecodeSaturationPolicy(16, 100, config)
        policy.time_predictor = Latency()
        policy.configure_update_protocol(async_enabled=True,
            update_interval_s=1., update_timeout_s=1., **protocol)
        policies.append(policy)
        return policy, controls

    yield make
    for controls in controls_list:
        controls.release.set()
    for policy in policies:
        assert policy.close(2)


def await_ready(policy, groups, now=10.):
    deadline = time.perf_counter() + 2
    while time.perf_counter() < deadline:
        result = policy.make_decision(groups, [], [], now)
        if result.target_status == 'async_ready':
            return result
        time.sleep(.001)
    raise AssertionError('background target did not complete')


def test_qrf_and_search_run_privately_in_background_while_main_continues(policy_factory, monkeypatch):
    policy, controls = policy_factory()
    controls.release.clear()
    from saturo.saturation.decode_saturation_core import DecodeSaturationEngine
    original = DecodeSaturationEngine.compute_decision
    search_threads = []

    def search(self, *args, **kwargs):
        search_threads.append(threading.get_ident())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DecodeSaturationEngine, 'compute_decision', search)
    group = Group()
    pending = policy.make_decision([group], [], [], 10.)
    assert controls.entered.wait(1)
    assert pending.target_status == 'pending' and pending.be_block_budget == 0
    assert not pending.search_feasible and not search_threads
    # The model remains blocked while native/main-thread work can continue.
    assert policy.make_decision([group], [], [], 10.).target_status == 'pending'
    controls.release.set()
    ready = await_ready(policy, [group])
    assert ready.search_feasible and ready.target_version == pending.target_version
    assert controls.threads and search_threads
    assert all(t != threading.get_ident() for t in controls.threads + search_threads)
    timing = policy.last_phase_timings
    assert timing['async_worker_wall_ms'] >= timing['async_qrf_prediction_ms']
    assert timing['async_worker_completed_perf_s'] >= timing['async_worker_started_perf_s']


def test_snapshot_freezes_tokens_history_and_never_passes_live_groups(policy_factory):
    policy, controls = policy_factory('token_ids')
    group = Group()
    controls.release.clear()
    policy.make_decision([group], [], [], 10.)
    assert controls.entered.wait(1)
    group.seq.prompt[:] = [999]
    group.seq.generated.append(777)
    row = controls.rows[0]
    assert row['prompt_token_ids'] == tuple(range(16))
    assert row['generated_token_ids'] == (0, 1)
    assert row['generated_len'] == 2 and row['prompt_len'] == 16
    controls.release.set()
    await_ready(policy, [group])
    assert policy.provider.prediction_calls == 0
    assert policy.update_protocol_diagnostics()['qrf_prediction_calls'] == 1


def test_frozen_snapshot_rejects_assignment_and_copies_arrival_inputs():
    item = snapshot()
    with pytest.raises(FrozenInstanceError):
        item.version = 2
    assert isinstance(item.requests, tuple) and isinstance(item.arrivals.intervals, tuple)


def test_membership_and_history_updates_invalidate_old_applied_target(policy_factory):
    policy, controls = policy_factory()
    group = Group()
    policy.make_decision([group], [], [], 10.)
    first = await_ready(policy, [group])
    controls.release.clear()
    second_group = Group('second')
    pending = policy.make_decision([group, second_group], [], [], 10.1)
    assert pending.target_status == 'pending' and pending.be_block_budget == 0
    assert pending.target_version > first.target_version
    controls.release.set()
    second = await_ready(policy, [group, second_group], 10.1)
    controls.release.clear()
    policy.provider.set_history_statistics(dict(arrival_rate=1., mean_prompt_len=32,
        mean_output_len=8, completed_requests=1, valid_until_s=20.))
    changed = policy.make_decision([group, second_group], [], [], 10.2)
    assert changed.target_version > second.target_version and not changed.search_feasible
    controls.release.set()


def test_interval_reuses_fresh_target_then_expires_to_rt_only(policy_factory):
    policy, controls = policy_factory()
    group = Group()
    policy.make_decision([group], [], [], 10.)
    ready = await_ready(policy, [group])
    before = policy.make_decision([group], [], [], 10.999)
    assert before.target_version == ready.target_version and before.target_status == 'async_ready'
    controls.release.clear()
    expired = policy.make_decision([group], [], [], 11.)
    assert expired.target_version > ready.target_version
    assert expired.target_status == 'pending' and expired.be_block_budget == 0
    controls.release.set()


def test_layout_epoch_waits_for_complete_history_bundle(policy_factory):
    policy, controls = policy_factory()
    group = Group()
    policy.make_decision([group], [], [], 10.)
    ready = await_ready(policy, [group])
    submitted = policy.update_protocol_diagnostics()['submitted']
    policy.configure_update_protocol(epoch=4, history_ready=False)
    policy.provider.reset_history_observations()
    held = policy.make_decision([group], [], [], 10.1)
    assert held.target_status == 'awaiting_history' and held.be_block_budget == 0
    assert held.target_epoch != ready.target_epoch
    assert policy.update_protocol_diagnostics()['submitted'] == submitted
    policy.provider.set_history_statistics(dict(arrival_rate=0., mean_prompt_len=32,
        mean_output_len=8, completed_requests=1, valid_until_s=20.))
    policy.configure_update_protocol(epoch=4, history_ready=True)
    policy.make_decision([group], [], [], 10.2)
    new = await_ready(policy, [group], 10.2)
    assert new.target_epoch > ready.target_epoch


def test_disable_closes_worker_and_warmup_initializes_before_traffic(policy_factory):
    policy, controls = policy_factory()
    info = policy.warm_update_protocol(1.)
    assert info['kernel_initialized'] and info['active'] == info['pending'] == 0
    assert not controls.rows  # No synthetic request/QRF feature is invented.
    epoch = info['epoch']
    policy.configure_update_protocol(async_enabled=True)
    assert policy.update_protocol_diagnostics()['epoch'] == epoch
    policy.configure_update_protocol(async_enabled=False)
    assert not policy.update_protocol_diagnostics()['thread_alive']
    policy.configure_update_protocol(async_enabled=True, epoch=2)
    policy.warm_update_protocol(1.)
    assert policy.update_protocol_diagnostics()['thread_alive']


def test_startup_can_initialize_private_kernel_while_history_is_unready(policy_factory):
    policy, controls = policy_factory()
    policy.configure_update_protocol(epoch=3, history_ready=False)
    info = policy.warm_update_protocol(1.)
    assert info['kernel_initialized'] and not info['history_ready']
    result = policy.make_decision([Group()], [], [], 10.)
    assert result.target_status == 'awaiting_history' and not result.search_feasible


def test_background_counters_are_preserved_after_close_and_role_reenable(policy_factory):
    policy, controls = policy_factory()
    group = Group()
    policy.make_decision([group], [], [], 10.)
    await_ready(policy, [group])
    assert policy.update_protocol_diagnostics()['background_qrf_prediction_calls'] == 1
    policy.configure_update_protocol(async_enabled=False)
    closed = policy.update_protocol_diagnostics()
    assert closed['background_target_computations'] == 1
    assert closed['background_qrf_prediction_calls'] == 1 and not closed['thread_alive']
    policy.configure_update_protocol(async_enabled=True)
    policy.make_decision([group], [], [], 10.)
    await_ready(policy, [group])
    assert policy.update_protocol_diagnostics()['background_target_computations'] == 2
    assert policy.update_protocol_diagnostics()['background_qrf_prediction_calls'] == 2
    assert controls.copies == 1


def test_history_epoch_invalidates_decision_without_recloning_frozen_model(policy_factory):
    policy, controls = policy_factory()
    policy.warm_update_protocol(1.)
    assert controls.copies == 1
    policy.configure_update_protocol(epoch=2, history_ready=False)
    policy.provider.reset_history_observations()
    policy.warm_update_protocol(1.)
    assert controls.copies == 1
    policy.configure_update_protocol(epoch=2, history_ready=True)
    policy.make_decision([Group()], [], [], 10.)
    assert await_ready(policy, [Group()]).target_status == 'async_ready'
    assert controls.copies == 1


def test_real_qrf_async_decision_matches_sync_for_identical_frozen_state(monkeypatch):
    import numpy as np
    from saturo.predictors.request_length_predictor import (
        QRFRemainingLengthPredictor, build_feature_matrix)
    model = QRFRemainingLengthPredictor(n_estimators=4, min_samples_leaf=1, n_jobs=1)
    rows = [dict(prompt_len=16+i, generated_len=i) for i in range(8)]
    model.fit(build_feature_matrix(rows, feature_mode='len', hash_bins=128),
              np.full(8, 8., dtype=np.float32))
    monkeypatch.setattr('saturo.predictors.qrf.load_required_qrf', lambda *args: model)
    config = SimpleNamespace(enable_decode_saturation=True,
        decode_saturation_tbt_ms=100., decode_saturation_lookahead_s=1.)
    sync = DecodeSaturationPolicy(16, 40, config)
    asynchronous = DecodeSaturationPolicy(16, 40, config)
    history = dict(arrival_rate=50., mean_prompt_len=16, mean_output_len=8,
                   completed_requests=10, valid_until_s=20.)
    groups = [Group('one'), Group('two')]
    try:
        for policy in (sync, asynchronous):
            policy.time_predictor = Latency()
            policy.provider.set_history_statistics(history)
        asynchronous.configure_update_protocol(async_enabled=True,
            update_interval_s=1., update_timeout_s=1.)
        expected = sync.make_decision(groups, [], [], 10.)
        asynchronous.make_decision(groups, [], [], 10.)
        actual = await_ready(asynchronous, groups)
        assert 1 < expected.target_interval_ms < 100
        fields = [name for name in expected.__dataclass_fields__
                  if name not in ('target_version', 'target_epoch',
                                  'target_snapshot_time_s', 'target_status')]
        assert {name: getattr(actual, name) for name in fields} == {
            name: getattr(expected, name) for name in fields}
        assert asynchronous.update_protocol_diagnostics()['qrf_prediction_calls'] == 2
    finally:
        sync.close(1.)
        asynchronous.close(1.)


def test_arrival_snapshot_contains_only_frozen_copied_values(policy_factory):
    policy, controls = policy_factory()
    policy.provider.set_history_statistics(dict(arrival_rate=2., mean_prompt_len=32,
        mean_output_len=8, completed_requests=2, valid_until_s=20.))
    policy.provider._arima_arrival_predictor._inter_arrivals_s = [.1, .2]
    frozen = policy._freeze_async_snapshot([Group()], 10., 1, 0, 0)
    policy.provider._history_statistics['mean_prompt_len'] = 999
    policy.provider._arima_arrival_predictor._inter_arrivals_s.append(10.)
    assert dict(frozen.arrivals.history)['mean_prompt_len'] == 32
    assert frozen.arrivals.intervals == (.1, .2)


def test_serving_provider_forecast_and_prediction_methods_are_not_called(policy_factory):
    policy, controls = policy_factory()
    def forbidden(*args):
        raise AssertionError('live provider used on native thread')
    policy.provider.get_future_arrivals = forbidden
    policy.provider.get_remaining_tokens_batch = forbidden
    group = Group()
    policy.make_decision([group], [], [], 10.)
    assert await_ready(policy, [group]).search_feasible


def test_invalid_time_profile_and_old_layout_epoch_fail_before_job_submission(policy_factory):
    policy, controls = policy_factory()
    policy.configure_update_protocol(epoch=4)
    with pytest.raises(ValueError, match='backwards'):
        policy.configure_update_protocol(epoch=3)
    policy.time_predictor.available = False
    with pytest.raises(RuntimeError, match='fitted time predictor'):
        policy.warm_update_protocol(1.)
    assert policy.update_protocol_diagnostics()['submitted'] == 0


def test_native_default_retains_synchronous_embedding_behavior(monkeypatch):
    config = SimpleNamespace(enable_decode_saturation=False)
    policy = DecodeSaturationPolicy(16, 100, config)
    assert not policy.async_target_enabled
    assert not policy.make_decision([], [], [], 1.).enabled
    assert not policy.update_protocol_diagnostics()['thread_alive']


def test_current_qrf_failure_is_explicit_and_never_reuses_old_target(policy_factory):
    policy, controls = policy_factory()
    controls.error = ValueError('QRF prediction failed')
    group = Group()
    policy.make_decision([group], [], [], 10.)
    result = policy._async_optimizer.wait(1.)
    with pytest.raises(RuntimeError, match='QRF prediction failed'):
        policy._accept_async_result(result)
    assert policy._async_applied is None
    assert policy.update_protocol_diagnostics()['last_error']['type'] == 'ValueError'


def test_active_target_timeout_is_explicit_without_creating_more_workers(policy_factory):
    policy, controls = policy_factory()
    policy.configure_update_protocol(update_timeout_s=.01)
    controls.release.clear()
    group = Group()
    policy.make_decision([group], [], [], 10.)
    assert controls.entered.wait(1.)
    time.sleep(.02)
    with pytest.raises(TimeoutError, match='exceeded'):
        policy.make_decision([group], [], [], 10.)
    stats = policy.update_protocol_diagnostics()
    assert stats['active'] == 1 and stats['pending'] == 0
    assert stats['submitted'] == 1 and policy._async_applied is None
    controls.release.set()


@pytest.mark.parametrize('field,value', [('update_interval_s', 0),
    ('update_timeout_s', float('nan')), ('epoch', -1)])
def test_invalid_protocol_is_rejected_before_mutation(policy_factory, field, value):
    policy, controls = policy_factory()
    before = policy.update_protocol_diagnostics()
    with pytest.raises(ValueError):
        policy.configure_update_protocol(**{field: value})
    assert policy.update_protocol_diagnostics() == before
