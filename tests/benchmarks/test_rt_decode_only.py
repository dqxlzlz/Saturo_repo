from types import SimpleNamespace

from vllm.core.scheduler import Scheduler


def test_decode_saturation_advances_waiting_rt_prefill():
    request = SimpleNamespace(is_prefill=lambda: True)
    policy = SimpleNamespace(is_rt=lambda _request: True)
    scheduler = SimpleNamespace(
        decode_saturation=policy,
        waiting=[request],
        _schedule_default=lambda: "scheduled",
    )

    assert Scheduler._schedule_decode_saturation(scheduler) == "scheduled"
