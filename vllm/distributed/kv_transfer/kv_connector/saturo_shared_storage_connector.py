# SPDX-License-Identifier: Apache-2.0
"""Real request-scoped V0 KV transport through a shared filesystem.

Uses the native model runner's send/recv hooks and native paged KV tensors.
Receives populate prompt KV before the first real decode forward; they never
return fabricated hidden states or bypass decode. Each rank publishes an
atomic tensor shard, then an atomic manifest. TP/PP shards are resharded by
global layer and KV head, including replicated GQA heads. The shared root
must be visible at the same path on all serving nodes. This is a correctness
backend, not an RDMA/NCCL performance claim.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from pathlib import Path

import torch

from saturo.runtime.kv_handoff import transfer_directory
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase


def _atomic_json(path, value):
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temp.write_text(json.dumps(value, sort_keys=True))
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _head_range(total, tp, rank):
    if total >= tp:
        if total % tp:
            raise ValueError("KV heads must be divisible by tensor parallel size")
        width = total // tp
        return rank * width, (rank + 1) * width
    if tp % total:
        raise ValueError("Tensor parallel size must be divisible by KV heads")
    first = rank // (tp // total)
    return first, first + 1


def _slots(block_table, count, block_size, capacity, device):
    if count <= 0 or len(block_table) < math.ceil(count / block_size):
        raise ValueError("KV transfer has an incomplete physical block table")
    blocks = torch.tensor(block_table[:math.ceil(count / block_size)],
                          device=device, dtype=torch.long)
    if bool(((blocks < 0) | (blocks >= capacity)).any()):
        raise ValueError("KV transfer physical block ID is out of range")
    return (blocks[:, None] * block_size + torch.arange(
        block_size, device=device)[None, :]).flatten()[:count]


class SaturoSharedStorageConnector(KVConnectorBase):
    def __init__(self, rank, local_rank, config):
        extra = config.kv_transfer_config.kv_connector_extra_config
        self.root = Path(extra["directory"])
        self.namespace = extra["namespace"]
        transfer_directory(self.root, self.namespace, "validation")
        self.timeout = float(extra.get("timeout_s", 30))
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("KV transfer timeout must be positive")
        self.rank = rank
        parallel = config.parallel_config
        self.tp = parallel.tensor_parallel_size
        self.pp = parallel.pipeline_parallel_size
        self.layer_start, self.layer_end = config.model_config.get_layers_start_end_indices(parallel)
        model = config.model_config
        hf = model.hf_text_config
        sliding_window = (model.get_sliding_window() if hasattr(model, "get_sliding_window")
                          else getattr(hf, "sliding_window", None))
        if model.is_deepseek_mla or sliding_window:
            raise ValueError("Real Saturo KV transfer currently requires full-attention, non-MLA models")
        if getattr(config.cache_config, "cache_dtype", "auto") != "auto":
            raise ValueError("Real Saturo KV transfer requires unquantized KV cache")
        if getattr(config.lora_config, "max_loras", 0):
            raise ValueError("Real Saturo KV transfer does not support LoRA adapters")
        self.heads = getattr(hf, "num_key_value_heads", hf.num_attention_heads)
        self.head_start, self.head_end = _head_range(self.heads, self.tp, rank % self.tp)
        self.head_size = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
        self.block_size = config.cache_config.block_size
        # The namespace isolates runs; fingerprint additionally rejects wrong
        # models/configs/dtypes even if a caller supplies a wrong handoff key.
        identity = hf.to_dict()
        identity.pop("_name_or_path", None)
        identity.pop("torch_dtype", None)
        self.fingerprint = hashlib.sha256(json.dumps(
            {"model": model.model, "config": identity, "dtype": str(model.dtype)},
            sort_keys=True, default=str).encode()).hexdigest()
        self._loaded = {}
        self.sent_requests = 0
        self.received_requests = 0

    def close(self):
        self._loaded.clear()

    def _cache_views(self, caches):
        if len(caches) != self.layer_end - self.layer_start:
            raise ValueError("KV transfer layer count does not match local PP stage")
        views = []
        for cache in caches:
            # FLASH_ATTN NHD cache: [2, blocks, block_size, heads, head_size].
            # Reject unsupported layouts instead of silently corrupting KV.
            if (cache.ndim != 5 or cache.shape[0] != 2
                    or tuple(cache.shape[2:]) != (self.block_size,
                        self.head_end-self.head_start, self.head_size)
                    or not cache.is_contiguous()):
                raise ValueError("Real Saturo KV transfer requires contiguous FLASH_ATTN NHD KV tensors")
            views.append((cache[0].flatten(0, 1), cache[1].flatten(0, 1)))
        return views

    def _check_request(self, item):
        if item["prompt_digest_observed"] != item["prompt_digest"]:
            raise ValueError("KV handoff prompt token IDs do not match producer prefix")

    def send_kv_caches_and_hidden_states(self, model_executable, model_input,
                                        kv_caches, hidden_or_intermediate_states):
        items = [x for x in (model_input.saturo_kv_transfers or [])
                 if x["operation"] == "send"]
        if not items:
            return
        views = self._cache_views(kv_caches)
        for item in items:
            self._check_request(item)
            directory = transfer_directory(self.root, self.namespace, item["key"])
            directory.mkdir(parents=True, exist_ok=True)
            slots = _slots(item["block_table"], item["prompt_len"], self.block_size,
                           kv_caches[0].shape[1], kv_caches[0].device)
            # Synchronous D2H completes before publication and before P's block
            # allocator can free/reuse these physical blocks.
            keys = torch.stack([key.index_select(0, slots).cpu() for key, _ in views])
            values = torch.stack([value.index_select(0, slots).cpu() for _, value in views])
            metadata = dict(schema=1, fingerprint=self.fingerprint,
                prompt_digest=item["prompt_digest"], prompt_len=item["prompt_len"],
                source_tp=self.tp, source_pp=self.pp, rank=self.rank,
                layer_start=self.layer_start, layer_end=self.layer_end,
                head_start=self.head_start, head_end=self.head_end,
                dtype=str(keys.dtype), head_size=self.head_size)
            tensor_file = directory / f"rank-{self.rank}.pt"
            temp = directory / (tensor_file.name + "." + uuid.uuid4().hex + ".tmp")
            try:
                torch.save({"metadata": metadata, "keys": keys, "values": values}, temp)
                os.replace(temp, tensor_file)
                _atomic_json(directory / f"rank-{self.rank}.json", metadata)
            finally:
                temp.unlink(missing_ok=True)
            self.sent_requests += 1

    def _manifest(self, directory, rank, deadline):
        path = directory / f"rank-{rank}.json"
        while not path.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Real KV shard missing (no dummy fallback): {path}")
            time.sleep(.01)
        return json.loads(path.read_text())

    def recv_kv_caches_and_hidden_states(self, model_executable, model_input, kv_caches):
        # Completed native requests release our per-sequence import bookkeeping.
        finished = set(model_input.finished_requests_ids or [])
        if finished:
            self._loaded = {k: v for k, v in self._loaded.items() if k[0] not in finished}
        items = [x for x in (model_input.saturo_kv_transfers or [])
                 if x["operation"] == "recv"]
        if not items:
            return None, False, model_input
        views = self._cache_views(kv_caches)
        for item in items:
            self._check_request(item)
            identity = (item["request_id"], item["seq_id"])
            prefix_blocks = tuple(item["block_table"][:math.ceil(item["prompt_len"] / self.block_size)])
            signature = (item["key"], model_input.virtual_engine, prefix_blocks)
            if self._loaded.get(identity) == signature:
                continue
            directory = transfer_directory(self.root, self.namespace, item["key"])
            deadline = time.monotonic() + self.timeout
            slots = _slots(item["block_table"], item["prompt_len"], self.block_size,
                           kv_caches[0].shape[1], kv_caches[0].device)
            covered = set()
            for rank in range(item["source_tp"] * item["source_pp"]):
                meta = self._manifest(directory, rank, deadline)
                if (meta["fingerprint"] != self.fingerprint
                        or meta["prompt_digest"] != item["prompt_digest"]
                        or meta["prompt_len"] != item["prompt_len"]
                        or meta["source_tp"] != item["source_tp"]
                        or meta["source_pp"] != item["source_pp"]
                        or meta["rank"] != rank or meta["schema"] != 1
                        or meta["dtype"] != str(kv_caches[0].dtype)
                        or meta["head_size"] != self.head_size):
                    raise ValueError("Real KV shard metadata is incompatible with this request/model")
                layers = range(max(self.layer_start, meta["layer_start"]),
                               min(self.layer_end, meta["layer_end"]))
                heads = range(max(self.head_start, meta["head_start"]),
                              min(self.head_end, meta["head_end"]))
                needed = [(layer, head) for layer in layers for head in heads
                          if (layer, head) not in covered]
                if not needed:
                    continue
                payload = torch.load(directory / f"rank-{rank}.pt", map_location="cpu", weights_only=True)
                if payload["metadata"] != meta:
                    raise ValueError("Real KV manifest/tensor mismatch")
                shape = (meta["layer_end"]-meta["layer_start"], item["prompt_len"],
                         meta["head_end"]-meta["head_start"], self.head_size)
                if any(tuple(payload[name].shape) != shape or payload[name].dtype != kv_caches[0].dtype
                       for name in ("keys", "values")):
                    raise ValueError("Real KV tensor shape/dtype mismatch")
                for layer, head in needed:
                    local_layer, local_head = layer-self.layer_start, head-self.head_start
                    source_layer, source_head = layer-meta["layer_start"], head-meta["head_start"]
                    for view, name in zip(views[local_layer], ("keys", "values")):
                        view[:, local_head].index_copy_(0, slots,
                            payload[name][source_layer, :, source_head].to(view.device))
                    covered.add((layer, head))
            expected = (self.layer_end-self.layer_start) * (self.head_end-self.head_start)
            if len(covered) != expected:
                raise ValueError("Real KV shards do not cover the destination layers/heads")
            self._loaded[identity] = signature
            self.received_requests += 1
        # The next input is P's actual sampled token. Run a real forward to
        # generate its successor; there is no synthetic handoff token.
        return None, False, model_input
