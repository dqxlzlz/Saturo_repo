import asyncio
import inspect

from vllm.executor.uniproc_executor import UniProcExecutor
from vllm.executor.executor_base import DistributedExecutorBase
from vllm.worker.model_runner import ModelRunner
from vllm.worker.worker_base import WorkerWrapperBase


class _AsyncWorker:

    def __init__(self) -> None:
        self.calls = []

    async def execute_model_async(self, request):
        await asyncio.sleep(0)
        self.calls.append(request)
        return [request]


def test_sync_model_runner_does_not_start_an_event_loop():
    sync_source = inspect.getsource(ModelRunner.execute_model)

    assert "asyncio.run" not in sync_source
    assert "forward_async" not in sync_source


def test_uniproc_executor_awaits_worker_async_entrypoint():

    async def run_test() -> None:
        executor = UniProcExecutor.__new__(UniProcExecutor)
        executor.driver_worker = _AsyncWorker()

        result = await executor.execute_model_async("request")

        assert result == ["request"]
        assert executor.driver_worker.calls == ["request"]

    asyncio.run(run_test())


def test_worker_wrapper_awaits_async_rpc_method():

    async def run_test() -> None:
        wrapper = WorkerWrapperBase.__new__(WorkerWrapperBase)
        wrapper.worker = _AsyncWorker()

        result = await wrapper.execute_method_async("execute_model_async",
                                                    "request")

        assert result == ["request"]
        assert wrapper.worker.calls == ["request"]

    asyncio.run(run_test())


def test_distributed_executor_starts_remote_loop_before_driver_collective():

    class _DistributedProbe:

        def __init__(self) -> None:
            self.parallel_worker_tasks = None
            self.remote_started = False

        async def _start_worker_execution_loop(self):
            self.remote_started = True

        async def _driver_execute_model_async(self, request):
            assert self.remote_started
            return [request]

    async def run_test() -> None:
        executor = _DistributedProbe()
        result = await DistributedExecutorBase.execute_model_async(
            executor, "request")
        assert result == ["request"]
        await executor.parallel_worker_tasks

    asyncio.run(run_test())
