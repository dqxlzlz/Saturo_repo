"""Global static-D leases cover the P→D interval without changing RT clocks."""
from collections import deque
from types import SimpleNamespace

import pytest

from saturo.runtime.scheduler import SaturoGlobalScheduler
from saturo.runtime.worker import WorkerState
from saturo.runtime.workload import BE, DECODE, PREFILL, RT, SaturoRequest


def scheduler(limit=1, decode_workers=1):
    s = SaturoGlobalScheduler.__new__(SaturoGlobalScheduler)
    s.clock = SimpleNamespace(now=1.0)
    s._elapsed = lambda: s.clock.now
    s.args = SimpleNamespace(rt_ttft_slo_ms=2000, rt_tbt_slo_ms=100,
                             window_s=60)
    s._source_arrival_epoch_s = 1000.0
    s.trace_recorder = None
    s.all_rt, s.all_be = [], []
    s.pending_prefill, s.pending_decode, s.pending_be = deque(), deque(), deque()
    s.prefill_handoffs = s.request_errors = 0
    s.rt_tokens = s.be_tokens = s.be_tokens_in_window = 0
    s.rt_ttft_samples_ms, s.rt_tbt_samples_ms = [], []
    s.rt_ttft_violations = s.rt_tbt_violations = 0
    s.role_switcher = None
    s.workers = {i: WorkerState(i, PREFILL if i == 0 else DECODE,
        'local', 'tp1_pp1', 1, 1, None, ready=True, max_num_seqs=limit)
        for i in range(decode_workers + 1)}
    return s


def request(s, name, kind=RT, arrival=.25):
    req = SaturoRequest(name, kind, 32, 4, arrival)
    (s.all_rt if kind == RT else s.all_be).append(req)
    return req


def activate(s, req, worker_id=1):
    req.decode_attempts = 1
    req.active_attempt_id = f'{req.request_id}:d1'
    req.assigned_worker_id = worker_id
    s.workers[worker_id].active_attempts[req.active_attempt_id] = req.request_id


def ack(s, req):
    w = s.workers[req.reserved_decode_worker_id]
    cmd = next(c for c in w.pending_commands if c.get('cmd') == 'reserve_decode'
               and c['reservation_id'] == req.decode_reservation_id)
    s._acknowledge_decode_reservations(w, [cmd], [{'accepted': True}])


def finish(s, req, phase, tokens):
    s._handle_event(dict(event='finished', worker_id=req.assigned_worker_id,
        request_id=req.request_id, attempt_id=req.active_attempt_id,
        phase=phase, global_token_count=tokens))


def test_static_pool_stops_before_p_when_native_rt_slots_are_promised():
    s = scheduler()
    first, second = request(s, 'first'), request(s, 'second')
    assert s._assign(s.workers[0], first, PREFILL)
    assert not s._assign(s.workers[0], second, PREFILL)
    assert second.prefill_attempts == 0 and second.prefill_submitted_s is None
    assert second.active_attempt_id is None and second.arrival_offset_s == .25
    assert s._decode_rt_slot_occupants(s.workers[1]) == {'first'}


def test_d_active_rt_prevents_a_new_prefill_but_be_does_not():
    s = scheduler()
    be = request(s, 'be', BE); activate(s, be)
    first = request(s, 'first')
    assert s._reserve_decode_for_prefill(first)
    s._release_decode_reservation(first)
    activate(s, first)
    assert not s._reserve_decode_for_prefill(request(s, 'second'))


def test_valid_claim_is_idempotent_even_at_capacity():
    s = scheduler(); req = request(s, 'first')
    assert s._reserve_decode_for_prefill(req)
    claim = req.decode_reservation_id
    commands = len(s.workers[1].pending_commands)
    assert s._reserve_decode_for_prefill(req)
    assert req.decode_reservation_id == claim
    assert len(s.workers[1].pending_commands) == commands


def test_claim_and_active_attempt_are_one_request_not_two_slots():
    s = scheduler(limit=2); first = request(s, 'first')
    assert s._reserve_decode_for_prefill(first)
    activate(s, first)
    assert s._decode_rt_slot_occupants(s.workers[1]) == {'first'}
    assert s._reserve_decode_for_prefill(request(s, 'second'))
    assert not s._reserve_decode_for_prefill(request(s, 'third'))


def test_stale_attempts_and_claims_do_not_consume_current_slots():
    s = scheduler(); w = s.workers[1]
    old = request(s, 'old'); activate(s, old)
    old.active_attempt_id = 'old:d2'
    w.active_attempts['missing:d1'] = 'missing'
    w.decode_reservations.update({'missing:prefill:1:claim:0',
                                  'old:prefill:1:claim:1'})
    old.decode_reservation_id = 'old:prefill:1:claim:2'
    old.reserved_decode_worker_id = 2
    assert s._decode_rt_slot_occupants(w) == set()
    assert s._reserve_decode_for_prefill(request(s, 'new'))


def test_lost_claim_cannot_bypass_a_full_worker_via_idempotence_fast_path():
    s = scheduler(); old = request(s, 'old')
    assert s._reserve_decode_for_prefill(old)
    s.workers[1].decode_reservations.clear()
    active = request(s, 'active'); activate(s, active)
    assert not s._reserve_decode_for_prefill(old)
    assert old.decode_reservation_id is None


def test_finished_request_claim_and_active_metadata_do_not_hold_slot():
    s = scheduler(); req = request(s, 'done')
    assert s._reserve_decode_for_prefill(req)
    activate(s, req); req.observed_tokens = req.output_len
    assert s._reserve_decode_for_prefill(request(s, 'new'))


def test_static_selection_uses_other_worker_without_overbooking_first():
    s = scheduler(decode_workers=2)
    first, second, third = [request(s, name) for name in ('first','second','third')]
    assert s._reserve_decode_for_prefill(first)
    assert s._reserve_decode_for_prefill(second)
    assert {first.reserved_decode_worker_id, second.reserved_decode_worker_id} == {1,2}
    assert not s._reserve_decode_for_prefill(third)


def test_reclaimable_be_does_not_hide_the_d_with_more_available_rt_slots(monkeypatch):
    s = scheduler(limit=16, decode_workers=2)
    for i in range(15):
        activate(s, request(s, f'rt1_{i}'), worker_id=1)
        activate(s, request(s, f'be2_{i}', BE), worker_id=2)
    activate(s, request(s, 'rt2'), worker_id=2)
    calls = []
    original = s._decode_rt_slot_occupants
    def count(worker):
        calls.append(worker.worker_id)
        return original(worker)
    monkeypatch.setattr(s, '_decode_rt_slot_occupants', count)
    new = request(s, 'new')
    assert s._reserve_decode_for_prefill(new)
    assert new.reserved_decode_worker_id == 2
    assert calls == [1,2]  # Filter/ranking share one immutable count per candidate.


def test_handoff_replaces_claim_with_active_attempt_without_free_slot_gap(monkeypatch):
    s = scheduler(); first, second = request(s, 'first'), request(s, 'second')
    assert s._assign(s.workers[0], first, PREFILL); ack(s, first)
    finish(s, first, PREFILL, 1)
    release = s._release_decode_reservation
    observed = []
    def check_release(req, *, notify=True):
        if not notify:
            assert req.active_attempt_id == 'first:d1'
            observed.append(s._decode_rt_slot_occupants(s.workers[1]))
        release(req, notify=notify)
        if not notify:
            observed.append(s._decode_rt_slot_occupants(s.workers[1]))
    monkeypatch.setattr(s, '_release_decode_reservation', check_release)
    s._drain_pending_decode()
    assert observed == [{'first'}, {'first'}]
    assert not first.decode_reservation_id
    assert not s._reserve_decode_for_prefill(second)
    finish(s, first, DECODE, first.output_len)
    assert s._reserve_decode_for_prefill(second)


@pytest.mark.parametrize('event', ['aborted','error'])
def test_failed_or_preempted_p_replaces_lease_and_stale_ack_cannot_confirm(event):
    s = scheduler(); req = request(s, 'first')
    assert s._assign(s.workers[0], req, PREFILL)
    old_claim = req.decode_reservation_id
    s._handle_event(dict(event=event, worker_id=0, request_id=req.request_id,
        attempt_id=req.active_attempt_id, phase=PREFILL, error_type='TestError'))
    assert req.prefill_attempts == 2 and req.decode_reservation_id != old_claim
    assert s._decode_rt_slot_occupants(s.workers[1]) == {'first'}
    s._acknowledge_decode_reservations(s.workers[1],
        [{'cmd':'reserve_decode','request_id':'first','reservation_id':old_claim}],
        [{'accepted':True}])
    assert not req.decode_reservation_confirmed
    assert not s._reserve_decode_for_prefill(request(s, 'second'))


@pytest.mark.parametrize('event', ['aborted','error'])
def test_d_retry_obtains_new_lease_before_resubmission(event):
    s = scheduler(); req = request(s, 'first'); req.observed_tokens = 1
    activate(s, req)
    s._handle_event(dict(event=event, worker_id=1, request_id=req.request_id,
        attempt_id=req.active_attempt_id, phase=DECODE, error_type='TestError'))
    assert list(s.pending_decode) == [req]
    s._drain_pending_decode()
    assert req.decode_reservation_id and req.active_attempt_id is None
    assert not s._reserve_decode_for_prefill(request(s, 'second'))
    ack(s, req); s._drain_pending_decode()
    assert req.active_attempt_id is not None
    assert s._decode_rt_slot_occupants(s.workers[1]) == {'first'}


def test_retired_worker_claim_migrates_without_reusing_dead_capacity():
    s = scheduler(decode_workers=2); req = request(s, 'first')
    assert s._reserve_decode_for_prefill(req)
    old_claim = req.decode_reservation_id
    s.workers[1].retired = True
    s._invalidate_decode_reservations(s.workers[1])
    assert req.reserved_decode_worker_id == 2
    assert not s.workers[1].decode_reservations
    assert s._decode_rt_slot_occupants(s.workers[2]) == {'first'}
    s._acknowledge_decode_reservations(s.workers[1],
        [{'cmd':'reserve_decode','request_id':'first','reservation_id':old_claim}],
        [{'accepted':True}])
    assert not req.decode_reservation_confirmed
    assert not s._reserve_decode_for_prefill(request(s, 'second'))


def test_explicit_claim_release_makes_slot_available():
    s = scheduler(); req = request(s, 'first')
    assert s._reserve_decode_for_prefill(req)
    s._release_decode_reservation(req)
    assert s._reserve_decode_for_prefill(request(s, 'second'))


def test_wait_for_slot_remains_part_of_source_arrival_ttft():
    s = scheduler(); old, req = request(s, 'old'), request(s, 'new', arrival=.25)
    activate(s, old)
    assert not s._assign(s.workers[0], req, PREFILL)
    old.finished_s = 2.0
    s.clock.now = 4.0
    assert s._assign(s.workers[0], req, PREFILL)
    s.clock.now = 4.1
    s._handle_event(dict(event='token', worker_id=0, request_id='new',
        attempt_id=req.active_attempt_id, phase=PREFILL, global_token_count=1))
    assert req.arrival_offset_s == .25 and req.first_token_s == 4.1
    assert s.rt_ttft_samples_ms == [pytest.approx(3850)]
    assert s.rt_ttft_violations == 1


def test_unspecified_legacy_capacity_and_adaptive_policy_are_preserved():
    s = scheduler(limit=0)
    for i in range(20):
        assert s._reserve_decode_for_prefill(request(s, f'old_{i}'))
    s = scheduler(); chosen = s.workers[1]
    activate(s, request(s, 'old'))
    selected = []
    s.role_switcher = SimpleNamespace(select=lambda role, req: selected.append(
        (role,req.request_id)) or chosen)
    assert s._reserve_decode_for_prefill(request(s, 'new'))
    assert selected == [(DECODE,'new')]
