from types import SimpleNamespace

from saturo.runtime.request_metadata import (
    PoolRole,
    RequestQoS,
    get_prefill_deadline,
    is_be_request,
    is_decode_phase,
    is_rt_request,
    request_qos,
)


def test_rt_be_helpers_use_is_realtime_attribute():
    rt = SimpleNamespace(is_realtime=True, prefill_ddl=123.0)
    be = SimpleNamespace(is_realtime=False, prefill_ddl=float("inf"))
    missing = SimpleNamespace()

    assert is_rt_request(rt)
    assert not is_be_request(rt)
    assert request_qos(rt) == RequestQoS.REAL_TIME
    assert get_prefill_deadline(rt) == 123.0

    assert not is_rt_request(be)
    assert is_be_request(be)
    assert request_qos(be) == RequestQoS.BEST_EFFORT

    assert not is_rt_request(missing)
    assert is_be_request(missing)


def test_decode_phase_helper_uses_is_prefill():
    post_prefill = SimpleNamespace(is_prefill=lambda: False)
    prefill = SimpleNamespace(is_prefill=lambda: True)

    assert is_decode_phase(post_prefill)
    assert not is_decode_phase(prefill)


def test_pool_role_names_are_stable_for_resource_allocation():
    assert PoolRole.PREFILL.value == "P"
    assert PoolRole.DECODE.value == "D"
    assert PoolRole.THROUGHPUT.value == "T"
