"""Request metadata for real V0 KV handoff; dummy remains the default."""
from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path

CONNECTOR = "SaturoSharedStorageConnector"


def prompt_digest(tokens):
    return hashlib.sha256(json.dumps(list(tokens), separators=(",", ":")).encode()).hexdigest()


def real_handoff(options):
    if isinstance(options, dict):
        return options.get("kv_handoff", "dummy") == "real"
    return getattr(options, "kv_handoff", "dummy") == "real"


def transfer_directory(root, namespace, key):
    # Both values cross RPC boundaries; never interpret them as paths.
    for value in (namespace, key):
        if not isinstance(value, str) or not value or any(
                c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value):
            raise ValueError("Invalid KV transfer identity")
    return Path(root) / namespace / key


def new_handoff(worker, tokens):
    return {"key": uuid.uuid4().hex, "prompt_len": len(tokens),
            "prompt_digest": prompt_digest(tokens),
            "source_tp": worker.tensor_parallel_size,
            "source_pp": worker.pipeline_parallel_size}


def build_transfer_metadata(groups):
    """Carry only explicit RT handoffs, with full physical block tables.

    Export is delayed until the final prefill chunk, then gathers ALL prompt
    blocks. This works with partial prefills and mixed local BE batches.
    """
    result = []
    for group in groups:
        extra = getattr(group.sampling_params, "extra_args", None) or {}
        handoff = extra.get("saturo_kv_handoff")
        if not handoff:
            continue
        operation = handoff["operation"]
        for seq_id, data in group.seq_data.items():
            if (operation == "recv" and not group.is_prompt
                    and getattr(data, "get_output_len", lambda: 1)() > 1):
                continue  # Prefix was imported before the first decode step.
            if operation == "send":
                if (not group.is_prompt or data.get_num_computed_tokens()
                        + (group.token_chunk_size or 0) < handoff["prompt_len"]):
                    continue
            result.append(dict(handoff, request_id=group.request_id,
                seq_id=seq_id, block_table=list(group.block_tables[seq_id]),
                # At retry/recompute the actual input prefix must still match.
                prompt_digest_observed=prompt_digest(
                    data.get_prompt_token_ids()[:handoff["prompt_len"]])))
    return result


def remove_handoff(root, namespace, handoff):
    if handoff:
        shutil.rmtree(transfer_directory(root, namespace, handoff["key"]),
                      ignore_errors=True)
