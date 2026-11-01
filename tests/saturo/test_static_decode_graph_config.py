"""CPU checks for fixed-D CUDA graphs and eager role-switch boundaries."""
from types import SimpleNamespace

import pytest

from saturo.runtime.worker import _build_engine_config
from saturo.runtime.workload import DECODE, PREFILL, TPUT


def config(role=DECODE, **changes):
    args = dict(worker_role=role, model='model', tensor_parallel_size=1,
                pipeline_parallel_size=1, load_format='auto', dtype='float16',
                gpu_memory_utilization=.2, max_num_seqs=16,
                enable_decode_saturation=False, disable_inductor=True)
    args.update(changes)
    return _build_engine_config(args)[0]


@pytest.mark.parametrize('role,adaptive,eager', [
    (DECODE, False, False), (DECODE, True, True),
    (PREFILL, False, True), (PREFILL, True, True),
    (TPUT, False, True), (TPUT, True, True),
])
def test_only_fixed_decode_changes_default(monkeypatch, role, adaptive, eager):
    monkeypatch.delenv('SATURO_ENFORCE_EAGER', raising=False)
    cfg = config(role, adaptive_role_switching=adaptive)
    assert cfg['enforce_eager'] is eager
    assert cfg['compilation_config'].use_cudagraph is (not eager)
    assert cfg['compilation_config'].level == 0
    assert cfg['compilation_config'].custom_ops == ['none']
    assert not cfg['compilation_config'].use_inductor
    if eager:
        assert 'disable_async_output_proc' not in cfg
    else:
        assert cfg['disable_async_output_proc'] is True


@pytest.mark.parametrize('value,eager', [
    ('1', True), ('true', True), ('yes', True),
    ('0', False), ('false', False), ('no', False), ('FALSE', False),
])
@pytest.mark.parametrize('role,adaptive', [
    (DECODE, False), (DECODE, True), (PREFILL, False), (TPUT, False),
])
def test_explicit_global_override_is_preserved(monkeypatch, value, eager,
                                                role, adaptive):
    monkeypatch.setenv('SATURO_ENFORCE_EAGER', value)
    cfg = config(role, adaptive_role_switching=adaptive)
    assert cfg['enforce_eager'] is eager
    # The D-specific compilation/output settings never leak into P/T or
    # adaptive workers, even when an operator explicitly disables eager.
    graph_path = role == DECODE and not adaptive and not eager
    assert cfg['compilation_config'].use_cudagraph is graph_path
    assert ('disable_async_output_proc' in cfg) is graph_path


@pytest.mark.parametrize('max_seqs,expected', [
    (1, [1]), (3, [4, 2, 1]), (16, [16, 8, 4, 2, 1]),
    (17, [24, 16, 8, 4, 2, 1]),
    (32, [32, 24, 16, 8, 4, 2, 1]),
])
def test_native_capture_sizes_cover_configured_capacity(monkeypatch,
                                                       max_seqs, expected):
    import vllm.envs as envs
    from vllm.config import VllmConfig
    monkeypatch.delenv('SATURO_ENFORCE_EAGER', raising=False)
    monkeypatch.setattr(envs, 'VLLM_USE_V1', False)
    cfg = config(max_num_seqs=max_seqs)
    compilation = cfg['compilation_config']
    assert compilation.cudagraph_capture_sizes is None
    fake = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=cfg['enforce_eager']),
        scheduler_config=SimpleNamespace(max_num_seqs=cfg['max_num_seqs']),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        compilation_config=compilation)
    VllmConfig._set_cudagraph_sizes(fake)
    assert compilation.cudagraph_capture_sizes == expected
    assert all(batch <= compilation.bs_to_padded_graph_size[batch]
               for batch in range(1, max_seqs + 1))


def test_enabling_graphs_does_not_enable_disabled_inductor(monkeypatch):
    monkeypatch.delenv('SATURO_ENFORCE_EAGER', raising=False)
    cfg = config(disable_inductor=False)
    assert cfg['enforce_eager'] is False
    assert cfg['disable_async_output_proc'] is True
    assert 'compilation_config' not in cfg
