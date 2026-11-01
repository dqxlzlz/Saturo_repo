"""GPU verification of vLLM worker -> CacheEngine -> native KV swap kernels."""
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_native_worker_copies_kv_to_cpu_and_restores_to_different_gpu_blocks():
    from vllm.attention.backends.flash_attn import FlashAttentionBackend
    from vllm.worker.cache_engine import CacheEngine
    from vllm.worker.worker import Worker

    cache = CacheEngine.__new__(CacheEngine)
    cache.num_attention_layers = 2
    cache.attn_backend = FlashAttentionBackend
    shape = (2, 4, 16, 2, 64)
    cache.gpu_cache = [torch.randn(shape, device='cuda', dtype=torch.float16)
                       for _ in range(2)]
    cache.cpu_cache = [torch.zeros(shape, dtype=torch.float16, pin_memory=True)
                       for _ in range(2)]
    expected = [layer[:, 1].clone() for layer in cache.gpu_cache]
    worker = Worker.__new__(Worker)
    worker.cache_engine = [cache]
    worker.execute_worker(SimpleNamespace(
        virtual_engine=0, blocks_to_swap_in=None,
        blocks_to_swap_out=torch.tensor([[1, 0]], dtype=torch.int64),
        blocks_to_copy=None))
    torch.cuda.synchronize()
    for cpu, reference in zip(cache.cpu_cache, expected):
        assert torch.equal(cpu[:, 0], reference.cpu())
    # Corrupt the original GPU source so reallocation without a real copy
    # cannot satisfy the test. Restore into a different physical GPU block.
    for layer in cache.gpu_cache:
        layer[:, 1].fill_(float('nan'))
        layer[:, 2].zero_()
    worker.execute_worker(SimpleNamespace(
        virtual_engine=0, blocks_to_swap_in=torch.tensor([[0, 2]], dtype=torch.int64),
        blocks_to_swap_out=None, blocks_to_copy=None))
    torch.cuda.synchronize()
    for gpu, reference in zip(cache.gpu_cache, expected):
        assert torch.equal(gpu[:, 2], reference)
