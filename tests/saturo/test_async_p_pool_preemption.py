import asyncio
from types import SimpleNamespace

from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams


class _PendingTask:

    @staticmethod
    def done() -> bool:
        return False


class _CoreEngine:

    def __init__(self) -> None:
        self.scheduler_config = SimpleNamespace(policy="priority")
        self._executing_be = True
        self.preemption_checks = []

    @staticmethod
    def _p_pool_enabled() -> bool:
        return True

    def _check_and_preempt_be(self, params, prompt_len=None) -> None:
        self.preemption_checks.append((params, prompt_len))


class _Stream:

    async def generator(self):
        if False:
            yield None


class _RequestTracker:

    def __init__(self, core: _CoreEngine) -> None:
        self.core = core
        self.add_calls = []

    def add_request(self, request_id, **kwargs):
        # The arrival hook must run before the request is deferred to the
        # background engine loop via RequestTracker.
        assert len(self.core.preemption_checks) == 1
        self.add_calls.append((request_id, kwargs))
        return _Stream()


def test_rt_arrival_checks_preemption_before_request_tracker_enqueue():

    async def run_test() -> None:
        engine = AsyncLLMEngine.__new__(AsyncLLMEngine)
        engine.engine = _CoreEngine()
        engine.log_requests = False
        engine.start_engine_loop = True
        engine.background_loop = object()
        engine._background_loop_unshielded = _PendingTask()
        engine._request_tracker = _RequestTracker(engine.engine)
        params = SamplingParams(is_realtime=True, prefill_ddl=1.0)

        await engine.add_request(
            "rt-preemption-probe",
            {"prompt_token_ids": [1, 2, 3, 4]},
            params,
        )

        assert engine.engine.preemption_checks == [(params, 4)]
        assert len(engine._request_tracker.add_calls) == 1

    asyncio.run(run_test())
