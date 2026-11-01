# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exact physical cache positions across allocator reuse and native swaps."""
import pytest

from vllm.core.block.cpu_gpu_block_allocator import CpuGpuBlockAllocator
from vllm.core.block.naive_block import NaiveBlock, NaiveBlockAllocator
from vllm.core.block.prefix_caching_block import PrefixCachingBlockAllocator
from vllm.utils import Device


def make_allocator(kind, ids):
    if kind == 'naive':
        return NaiveBlockAllocator(create_block=NaiveBlock,
                                   num_blocks=len(ids), block_size=16,
                                   block_ids=ids)
    return PrefixCachingBlockAllocator(num_blocks=len(ids), block_size=16,
                                       block_ids=ids)


@pytest.mark.parametrize('kind', ['naive', 'prefix_caching'])
@pytest.mark.parametrize('ids', [list(range(8)), [99, 7, -4, 0, 42],
                                [1, 2, 3], []])
def test_physical_rank_matches_sorted_universe_and_survives_reuse(kind, ids):
    allocator = make_allocator(kind, ids)
    expected = sorted(ids)
    for _ in range(3):
        assert [allocator.get_physical_block_id(i) for i in ids] == [
            expected.index(i) for i in ids]
        blocks = [allocator.allocate_mutable_block(prev_block=None)
                  for _ in ids]
        for block in reversed(blocks):
            allocator.free(block)
        assert allocator.all_block_ids == frozenset(ids)


@pytest.mark.parametrize('kind', ['naive', 'prefix_caching'])
@pytest.mark.parametrize('value', [-100, 999, None, 'missing', [], [2]])
def test_unknown_id_preserves_list_index_error(kind, value):
    allocator = make_allocator(kind, [42, 0, 7])
    with pytest.raises(ValueError) as expected:
        sorted(allocator.all_block_ids).index(value)
    with pytest.raises(ValueError) as actual:
        allocator.get_physical_block_id(value)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize('kind', ['naive', 'prefix_caching'])
@pytest.mark.parametrize('value', [False, True, 0.0, 1.0])
def test_equal_numeric_ids_preserve_list_index_semantics(kind, value):
    allocator = make_allocator(kind, [42, 0, 1, 7])
    assert allocator.get_physical_block_id(value) == sorted(
        allocator.all_block_ids).index(value)


@pytest.mark.parametrize('kind', ['naive', 'prefix_caching'])
def test_native_cpu_gpu_roundtrip_uses_device_local_sorted_positions(kind):
    gpu = make_allocator(kind, [41, 7, 100, 23])
    cpu = make_allocator(kind, [-2, -9, 55, 300])
    allocator = CpuGpuBlockAllocator(cpu_block_allocator=cpu,
                                     gpu_block_allocator=gpu)
    blocks = [allocator.allocate_mutable_block(prev_block=None,
                                               device=Device.GPU)
              for _ in range(2)]
    expected_tokens = [[11, 12], [21, 22, 23]]
    for block, tokens in zip(blocks, expected_tokens):
        block.append_token_ids(tokens)
    original_ids = [block.block_id for block in blocks]
    gpu_cache = {gpu.get_physical_block_id(block.block_id): tuple(tokens)
                 for block, tokens in zip(blocks, expected_tokens)}
    cpu_cache = {}
    out_mapping = allocator.swap(blocks, Device.GPU, Device.CPU)
    assert set(out_mapping) == set(original_ids)
    for source, target in out_mapping.items():
        cpu_cache[allocator.get_physical_block_id(Device.CPU, target)] = (
            gpu_cache[allocator.get_physical_block_id(Device.GPU, source)])
    assert [block.token_ids for block in blocks] == expected_tokens
    assert allocator.get_num_free_blocks(Device.GPU) == 4
    assert allocator.get_num_free_blocks(Device.CPU) == 2

    cpu_ids = [block.block_id for block in blocks]
    in_mapping = allocator.swap(blocks, Device.CPU, Device.GPU)
    assert set(in_mapping) == set(cpu_ids)
    for source, target in in_mapping.items():
        gpu_cache[allocator.get_physical_block_id(Device.GPU, target)] = (
            cpu_cache[allocator.get_physical_block_id(Device.CPU, source)])
    assert [gpu_cache[gpu.get_physical_block_id(block.block_id)]
            for block in blocks] == [tuple(tokens) for tokens in expected_tokens]
    assert [block.token_ids for block in blocks] == expected_tokens
    assert allocator.get_num_free_blocks(Device.GPU) == 2
    assert allocator.get_num_free_blocks(Device.CPU) == 4
