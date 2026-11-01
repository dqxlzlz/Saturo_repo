from types import SimpleNamespace

import torch

from vllm.worker.model_runner import ModelRunner


def test_dummy_kv_receive_requires_request_scoped_batch_marker():
    runner = ModelRunner.__new__(ModelRunner)
    runner.vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(is_kv_consumer=True),
    )
    model_input = SimpleNamespace(
        attn_metadata=SimpleNamespace(
            prefill_metadata=object(),
            num_prefill_tokens=4,
        ),
        input_tokens=torch.ones(4, dtype=torch.long),
        saturo_dummy_kv_decode_batch=False,
    )
    kv_caches = [torch.ones(1)]

    assert not runner.need_recv_kv(model_input, kv_caches)

    model_input.saturo_dummy_kv_decode_batch = True
    assert runner.need_recv_kv(model_input, kv_caches)


def test_dummy_kv_receive_rejects_mixed_prefill_decode_batch():
    runner = ModelRunner.__new__(ModelRunner)
    runner.vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(is_kv_consumer=True),
    )
    model_input = SimpleNamespace(
        attn_metadata=SimpleNamespace(
            prefill_metadata=object(),
            num_prefill_tokens=4,
        ),
        input_tokens=torch.ones(5, dtype=torch.long),
        saturo_dummy_kv_decode_batch=True,
    )

    assert not runner.need_recv_kv(model_input, [torch.ones(1)])
