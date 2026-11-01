"""Workload primitives for Saturo experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


PREFILL = "prefill"
DECODE = "decode"
TPUT = "throughput"
RT = "rt"
BE = "be"
REQUEST_TBT_DEFINITION = (
    "per-request max scheduler-receipt inter-event gap; includes P-to-D handoff"
)


def repo_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "vllm").exists() and (parent / "saturo").exists():
            return parent
    return path.parents[1]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    idx = min(len(ordered) - 1, max(0, int(math.ceil(q * len(ordered))) - 1))
    return float(ordered[idx])


def is_local_host(host: str) -> bool:
    return host in ("", "local", "localhost", "127.0.0.1")


@dataclass
class SaturoRequest:
    request_id: str
    kind: str
    prompt_len: int
    output_len: int
    arrival_offset_s: float
    prompt_token_ids: Optional[list[int]] = field(default=None, repr=False)
    prefill_submitted_s: Optional[float] = None
    prefill_finished_s: Optional[float] = None
    decode_submitted_s: Optional[float] = None
    first_token_s: Optional[float] = None
    last_token_s: Optional[float] = None
    finished_s: Optional[float] = None
    observed_tokens: int = 0
    prefill_attempts: int = 0
    decode_attempts: int = 0
    active_attempt_id: Optional[str] = None
    assigned_worker_id: Optional[int] = None
    reserved_decode_worker_id: Optional[int] = None
    decode_reservation_id: Optional[str] = None
    decode_reservation_confirmed: bool = False
    generated_token_ids: list[int] = field(default_factory=list, repr=False)
    kv_handoff: Optional[dict] = field(default=None, repr=False)
    source_id: Optional[str] = None
    dataset: Optional[str] = None
    azure_prompt_len: Optional[int] = None
    azure_output_len: Optional[int] = None
    tbt_ms: Optional[float] = None
    tbt_samples: int = 0
    tbt_unobserved_intervals: int = 0

    @property
    def ttft_ms(self) -> Optional[float]:
        if self.first_token_s is None:
            return None
        return (self.first_token_s - self.arrival_offset_s) * 1000.0

    @property
    def tbt_complete(self) -> bool:
        return (self.observed_tokens > 0
                and self.tbt_unobserved_intervals == 0
                and self.tbt_samples == self.observed_tokens - 1)

    def joint_slo_met(self, ttft_slo_ms: float,
                      tbt_slo_ms: float) -> Optional[bool]:
        if self.finished_s is None or self.observed_tokens < self.output_len:
            return False
        if (self.ttft_ms is None or not self.tbt_complete
                or (self.observed_tokens > 1 and self.tbt_ms is None)):
            return None
        return (self.ttft_ms <= ttft_slo_ms
                and (self.observed_tokens == 1 or self.tbt_ms <= tbt_slo_ms))

    @property
    def remaining_tokens(self) -> int:
        return max(self.output_len - self.observed_tokens, 0)

    @property
    def decode_prompt_len(self) -> int:
        return self.prompt_len + self.observed_tokens

    @property
    def is_finished(self) -> bool:
        return self.finished_s is not None or self.observed_tokens >= self.output_len


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_file(model: Optional[str], filename: str) -> Optional[Path]:
    if not model:
        return None
    directory = Path(model).expanduser()
    if directory.is_dir():
        path = directory / filename
        return path if path.is_file() else None
    # A Hub model ID may already have local files. Never download weights or
    # contact the Hub from the workload reader.
    from huggingface_hub import try_to_load_from_cache
    try:
        cached = try_to_load_from_cache(str(model), filename)
    except (ValueError, OSError):
        return None
    return Path(cached) if isinstance(cached, str) else None


def workload_context_limit(model=None, max_model_len=None) -> Optional[int]:
    if max_model_len is not None:
        if type(max_model_len) is not int or max_model_len <= 0:
            raise ValueError("max_model_len must be a positive integer")
        return max_model_len
    config_path = _model_file(model, "config.json")
    if config_path is not None:
        config = json.loads(config_path.read_text())
        config = config.get("text_config", config)
        value = config.get("max_position_embeddings", config.get("n_positions"))
        if type(value) is int and value > 0:
            return value
    return None


def validate_workload_manifest(path, kind, *, model=None, max_model_len=None):
    """Check prepared-data provenance locally; legacy CSVs need no manifest."""
    manifest_path = Path(str(path) + ".manifest.json")
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported workload manifest: {manifest_path}; rebuild this workload")
    expected_kinds = ("conv", "code") if kind == RT else ("be",)
    if manifest.get("kind") not in expected_kinds:
        raise ValueError(f"Workload kind mismatch in {manifest_path}: expected {kind}")
    expected_hash = manifest.get("output", {}).get("sha256")
    if not expected_hash or _sha256(Path(path)) != expected_hash:
        raise ValueError(f"Workload SHA256 mismatch: {path}")
    if model:
        files = {entry["name"]: entry["sha256"] for entry in manifest["tokenizer_files"]}
        # Model config describes the architecture, not the token-ID mapping.
        # Fast-tokenizer JSON is authoritative; optional duplicate SPM files
        # need not be present in a model-weight directory.
        if "tokenizer.json" in files:
            names = ["tokenizer.json"] + [name for name in
                     ("tokenizer_config.json", "special_tokens_map.json", "added_tokens.json") if name in files]
        else:
            names = [name for name in files if name.startswith("tokenizer") or name == "special_tokens_map.json"]
        if not names:
            raise ValueError(f"No tokenizer fingerprint in {manifest_path}")
        for name in names:
            actual = _model_file(model, name)
            if actual is None or _sha256(actual) != files[name]:
                raise ValueError(
                    f"Workload tokenizer does not match --model {model}: {name}; "
                    "use matching tokenizer files or rebuild the workload for this model")
    # RT can replay a shorter prefix; its context check uses selected rows.
    if kind == BE and max_model_len is not None:
        maximum = manifest["max_total_tokens"]
        if maximum > max_model_len:
            raise ValueError(f"{path} requires {maximum} total tokens; --max-model-len is {max_model_len}")
    return manifest


def preflight_workload_manifests(args):
    """Reject wrong tokenizer/BE context before resource profiling can start."""
    model = getattr(args, "model", None)
    limit = workload_context_limit(model, getattr(args, "max_model_len", None))
    for kind, name in ((RT, "rt_workload_csv"), (BE, "be_workload_csv")):
        path = getattr(args, name, None)
        if path:
            validate_workload_manifest(path, kind, model=model, max_model_len=limit)


def _positive_integer(value, name, location):
    try:
        number = float(value)
        if not math.isfinite(number) or number <= 0 or not number.is_integer():
            raise ValueError
        return int(number)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{location}: {name} must be a positive integer") from None


def _token_ids(value, length, name, location, vocab_size=None):
    try:
        tokens = json.loads(value)
    except (TypeError, ValueError):
        raise ValueError(f"{location}: invalid JSON in {name}") from None
    if (not isinstance(tokens, list) or len(tokens) != length
            or any(type(t) is not int or t < 0 or (vocab_size is not None and t >= vocab_size)
                   for t in tokens)):
        raise ValueError(f"{location}: invalid {name}; expected {length} nonnegative integer token IDs")
    return tokens


def _check_context(rows, limit, path):
    if limit is not None:
        for row in rows:
            total = row.prompt_len + row.output_len
            if total > limit:
                raise ValueError(f"{path}: {row.request_id} needs {total} total tokens; "
                                 f"--max-model-len is {limit}; requests are not truncated")


def load_workload_csv(path: str, kind: str, *, expected_model=None,
                      max_model_len=None, _defer_context_check=False) -> list[SaturoRequest]:
    """Read real IDs or legacy length-only rows, without retaining raw text.

    Declared token-ID columns must be populated on every row. Reference output
    IDs validate output_len but are not fed to the model or retained in memory.
    Azure target lengths are provenance only; actual token lengths drive serving.
    """
    if kind not in (RT, BE):
        raise ValueError(f"Unknown workload kind: {kind}")
    limit = workload_context_limit(expected_model, max_model_len)
    manifest = validate_workload_manifest(path, kind, model=expected_model, max_model_len=limit)
    vocab_size = manifest.get("vocab_size") if manifest else None
    csv.field_size_limit(max(csv.field_size_limit(), 16 * 1024 * 1024))
    rows, seen = [], set()
    with open(path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        if not {"prompt_len", "output_len"}.issubset(fields):
            raise ValueError(f"{path}: missing prompt_len/output_len CSV columns")
        has_prompts = "prompt_token_ids" in fields
        has_references = "output_token_ids" in fields
        if manifest and not has_prompts:
            raise ValueError(f"{path}: prepared workload is missing prompt_token_ids")
        for idx, raw in enumerate(reader):
            location = f"{path}: row {idx + 2}"
            request_id = raw.get("request_id") or f"{kind}_{idx}"
            if request_id in seen:
                raise ValueError(f"{location}: duplicate request_id {request_id}")
            seen.add(request_id)
            prompt_len = _positive_integer(raw.get("prompt_len"), "prompt_len", location)
            output_len = _positive_integer(raw.get("output_len"), "output_len", location)
            try:
                arrival = float(raw.get("arrival_offset_s") or 0.0)
            except (ValueError, TypeError):
                raise ValueError(f"{location}: invalid arrival_offset_s") from None
            if not math.isfinite(arrival) or arrival < 0:
                raise ValueError(f"{location}: arrival_offset_s must be finite and nonnegative")
            tokens = None
            if has_prompts:
                tokens = _token_ids(raw.get("prompt_token_ids"), prompt_len,
                                    "prompt_token_ids", location, vocab_size)
            elif raw.get("prompt"):
                raise ValueError(f"{location}: text prompt requires prompt_token_ids; tokenize with the workload builder")
            if has_references:
                _token_ids(raw.get("output_token_ids"), output_len,
                           "output_token_ids", location, vocab_size)
            rows.append(SaturoRequest(
                request_id=request_id, kind=kind, prompt_len=prompt_len,
                output_len=output_len, arrival_offset_s=arrival if kind == RT else 0.0,
                prompt_token_ids=tokens, source_id=raw.get("source_id") or None,
                dataset=raw.get("dataset") or None,
                azure_prompt_len=_positive_integer(raw["azure_prompt_len"], "azure_prompt_len", location)
                    if raw.get("azure_prompt_len") else None,
                azure_output_len=_positive_integer(raw["azure_output_len"], "azure_output_len", location)
                    if raw.get("azure_output_len") else None))
    if manifest and len(rows) != manifest["count"]:
        raise ValueError(f"{path}: row count differs from workload manifest")
    if kind == RT and rows:
        trim = float(os.environ.get("RT_TRIM_TOP_FRAC", "0") or 0)
        scale = float(os.environ.get("RT_LEN_SCALE", "1") or 1)
        if not math.isfinite(trim) or not 0 <= trim < 1 or not math.isfinite(scale) or scale <= 0:
            raise ValueError("RT_TRIM_TOP_FRAC must be in [0,1); RT_LEN_SCALE must be finite and positive")
        if scale != 1 and any(r.prompt_token_ids is not None for r in rows):
            raise ValueError("RT_LEN_SCALE cannot reshape real prompt_token_ids; rebuild the workload instead")
        before = len(rows)
        drop = set(sorted(range(before), key=lambda i: rows[i].prompt_len)[:int(before * trim)])
        rows = [r for i, r in enumerate(rows) if i not in drop]
        if scale != 1:
            for row in rows:
                row.prompt_len = max(1, round(row.prompt_len * scale))
        if trim or scale != 1:
            print(f"[workload] RT reshape: trim_short_frac={trim} len_scale={scale} | rows {before}->{len(rows)}")
        rows.sort(key=lambda row: row.arrival_offset_s)
    if not _defer_context_check:
        _check_context(rows, limit, path)
    return rows


def load_rt_workload_csv(
    path: str,
    *,
    num_req: int,
    target_rps: float,
    expected_model=None,
    max_model_len=None,
) -> list[SaturoRequest]:
    """Load exactly ``num_req`` RT rows and scale that prefix to target RPS."""
    from saturo.tools.trace_scaler import scale_to_target_rps_fixed_num

    # Check the context of the selected chronological prefix, not unused tail rows.
    rows = load_workload_csv(path, RT, expected_model=expected_model, _defer_context_check=True)
    scaled_arrivals = scale_to_target_rps_fixed_num(
        [row.arrival_offset_s for row in rows],
        num_req=num_req,
        target_rps=target_rps,
    )
    selected = rows[:num_req]
    _check_context(selected, workload_context_limit(expected_model, max_model_len), path)
    for row, arrival_s in zip(selected, scaled_arrivals):
        row.arrival_offset_s = arrival_s
    return selected
