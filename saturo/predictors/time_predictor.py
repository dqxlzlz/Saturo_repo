from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PiecewiseLinearModel:
    width: int = 32
    coef: Dict[int, Tuple[float, float]] = None
    fallback: Tuple[float, float] = (0.0, 0.0)

    def segment_id(self, n: float) -> int:
        return int(math.floor(n / self.width))

    def predict_one(self, n: float) -> float:
        if self.coef is None:
            return 0.0
        sid = self.segment_id(float(n))
        a, b = self.coef.get(sid, self.fallback)
        return max(0.0, a * float(n) + b)

    def repair_first_segment(self) -> None:
        # Reuse the next segment when a noisy low-token fit creates a downward
        # jump at the first boundary. Raw measurements remain unchanged.
        if not self.coef:
            return
        seg0 = self.coef.get(0)
        seg1 = self.coef.get(1)
        if seg0 is None or seg1 is None:
            return
        w = float(self.width)
        seg0_right = seg0[0] * (w - 1.0) + seg0[1]
        seg1_left = seg1[0] * w + seg1[1]
        if seg0_right > seg1_left:
            self.coef[0] = seg1

    def load_dict(self, data: dict) -> None:
        self.width = int(data.get("width", self.width))
        self.coef = {
            int(k): (float(v[0]), float(v[1]))
            for k, v in data.get("coef", {}).items()
        }
        fallback = data.get("fallback", self.fallback)
        self.fallback = (float(fallback[0]), float(fallback[1]))
        self.repair_first_segment()


@dataclass
class TileQuantizedLinearModel:
    """Measured tile lookup table for the stateless linear time.

    Tensor-core GEMMs quantize their effective M dimension into tiles.  A
    step-function is therefore a better representation of the measured
    decode-only linear time than fitting an independent slope in every small
    token interval.  The levels need not be monotonic: better GPU utilization
    can make a larger small batch faster.  Values are per transformer layer,
    just like
    :class:`PiecewiseLinearModel`.

    The table deliberately has a finite calibrated range.  Returning ``None``
    outside it lets TimePredictor fall back to the original profile instead of
    extrapolating an unmeasured plateau.
    """

    tile_width: int = 128
    max_tokens: int = 0
    token_upper_bounds: Optional[List[int]] = None
    per_layer_values_ms: Optional[List[float]] = None

    def predict_one(self, n: float) -> Optional[float]:
        if (not self.per_layer_values_ms or n <= 0
                or n > self.max_tokens):
            return None
        if self.token_upper_bounds:
            tile_id = bisect_left(self.token_upper_bounds, int(math.ceil(n)))
        else:
            tile_id = max(0, int(math.ceil(float(n) / self.tile_width)) - 1)
        if tile_id >= len(self.per_layer_values_ms):
            return None
        return max(0.0, self.per_layer_values_ms[tile_id])

    def load_dict(self, data: dict) -> None:
        tile_width = int(data.get("tile_width", self.tile_width))
        if tile_width <= 0:
            raise ValueError("linear_tile_override.tile_width must be positive")
        values = [float(value) for value in data.get("per_layer_values_ms", [])]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("linear_tile_override values must be finite")
        values = [max(0.0, value) for value in values]
        bounds = [int(value) for value in data.get("token_upper_bounds", [])]
        if bounds:
            if len(bounds) != len(values):
                raise ValueError(
                    "linear_tile_override token/value lengths differ")
            if any(right <= left for left, right in zip(bounds, bounds[1:])):
                raise ValueError(
                    "linear_tile_override bounds must strictly increase")
            if bounds[0] <= 0:
                raise ValueError(
                    "linear_tile_override bounds must be positive")
        max_tokens = int(data.get(
            "max_tokens", bounds[-1] if bounds else tile_width * len(values)))
        self.tile_width = tile_width
        self.max_tokens = min(
            max_tokens, bounds[-1] if bounds else tile_width * len(values))
        self.token_upper_bounds = bounds or None
        self.per_layer_values_ms = values


@dataclass
class RuntimeOverheadModel:
    """Empirical correction applied to one predicted engine iteration.

    New profiles should express the correction in terms of total query tokens
    so that prefill and decode work with the same token count receive the same
    correction.  The request/type-specific fields remain for compatibility
    with older profiles.
    """

    constant_ms: float = 0.0
    per_query_token_ms: float = 0.0
    query_token_hinge_threshold: float = 0.0
    per_query_token_above_hinge_ms: float = 0.0
    # Legacy profile features.
    per_prefill_request_ms: float = 0.0
    per_decode_request_ms: float = 0.0
    per_active_request_ms: float = 0.0
    per_prefill_token_ms: float = 0.0
    per_decode_token_ms: float = 0.0
    decode_request_hinge_threshold: float = 0.0
    per_decode_request_above_hinge_ms: float = 0.0

    def predict_ms(
        self,
        *,
        prefill_request_count: int,
        decode_request_count: int,
        prefill_tokens: float,
        decode_tokens: float,
    ) -> float:
        active_request_count = prefill_request_count + decode_request_count
        total_query_tokens = prefill_tokens + decode_tokens
        return max(
            0.0,
            self.constant_ms
            + self.per_query_token_ms * total_query_tokens
            + self.per_query_token_above_hinge_ms * max(
                total_query_tokens - self.query_token_hinge_threshold,
                0.0,
            )
            + self.per_prefill_request_ms * prefill_request_count
            + self.per_decode_request_ms * decode_request_count
            + self.per_active_request_ms * active_request_count
            + self.per_prefill_token_ms * prefill_tokens
            + self.per_decode_token_ms * decode_tokens
            + self.per_decode_request_above_hinge_ms * max(
                decode_request_count - self.decode_request_hinge_threshold,
                0.0,
            ),
        )

    def load_dict(self, data: dict) -> None:
        self.constant_ms = float(data.get(
            "constant_ms",
            data.get("intercept_ms", self.constant_ms),
        ))
        self.per_query_token_ms = float(data.get(
            "per_query_token_ms", self.per_query_token_ms))
        self.query_token_hinge_threshold = float(data.get(
            "query_token_hinge_threshold",
            self.query_token_hinge_threshold,
        ))
        self.per_query_token_above_hinge_ms = float(data.get(
            "per_query_token_above_hinge_ms",
            self.per_query_token_above_hinge_ms,
        ))
        self.per_prefill_request_ms = float(data.get(
            "per_prefill_request_ms", self.per_prefill_request_ms))
        self.per_decode_request_ms = float(data.get(
            "per_decode_request_ms", self.per_decode_request_ms))
        self.per_active_request_ms = float(data.get(
            "per_active_request_ms",
            data.get("per_request_ms", self.per_active_request_ms),
        ))
        self.per_prefill_token_ms = float(data.get(
            "per_prefill_token_ms", self.per_prefill_token_ms))
        self.per_decode_token_ms = float(data.get(
            "per_decode_token_ms", self.per_decode_token_ms))
        self.decode_request_hinge_threshold = float(data.get(
            "decode_request_hinge_threshold",
            self.decode_request_hinge_threshold,
        ))
        self.per_decode_request_above_hinge_ms = float(data.get(
            "per_decode_request_above_hinge_ms",
            self.per_decode_request_above_hinge_ms,
        ))

    def to_dict(self) -> Dict[str, float]:
        return {
            "constant_ms": self.constant_ms,
            "per_query_token_ms": self.per_query_token_ms,
            "query_token_hinge_threshold":
                self.query_token_hinge_threshold,
            "per_query_token_above_hinge_ms":
                self.per_query_token_above_hinge_ms,
            "per_prefill_request_ms": self.per_prefill_request_ms,
            "per_decode_request_ms": self.per_decode_request_ms,
            "per_active_request_ms": self.per_active_request_ms,
            "per_prefill_token_ms": self.per_prefill_token_ms,
            "per_decode_token_ms": self.per_decode_token_ms,
            "decode_request_hinge_threshold":
                self.decode_request_hinge_threshold,
            "per_decode_request_above_hinge_ms":
                self.per_decode_request_above_hinge_ms,
        }


STATELESS_GROUP = "stateless_linear"
STATEFUL_GROUP = "stateful_attention"
TIME_PREDICTOR_PROFILE_VERSION = 3
DEFAULT_PROFILE_WARMUP = 3
DEFAULT_PROFILE_REPEAT = 7
DEFAULT_PROFILE_NUM_LAYERS = 3
DEFAULT_PROFILE_LAYER_START = 0
DEFAULT_PROFILE_KV_CACHE_FRACTION = 0.9
DEFAULT_LINEAR_PIECEWISE_WIDTH = 32
DEFAULT_PROFILE_BLOCK_SIZE = 16
# Sweep bounds (inclusive) for the automatic profiling grid.
# Dense single-request prefill sweep used to fit the stateless-linear model.
LINEAR_PROFILE_RANGE = (2, 8192, 2)
PREFILL_REQ_LEN_RANGE = (32, 4096, 32)
PREFILL_NUM_REQ_RANGE = (1, 128, 1)
PREFILL_MAX_TOKENS = 12000
DECODE_REQ_LEN_RANGE = (32, 4096, 32)
DECODE_NUM_REQ_RANGE = (16, 512, 16)


def _clean_identity_value(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value)
    text = text.strip()
    return text or "unknown"


def _resolve_num_layers(model_executor: Any) -> int:
    """Read num_hidden_layers from the engine's model config."""
    model_config = getattr(model_executor, "model_config", None)
    if model_config is None:
        return 0
    hf_config = getattr(model_config, "hf_config", None)
    for cfg in (
        getattr(model_config, "hf_text_config", None),
        getattr(hf_config, "text_config", None),
        hf_config,
    ):
        if cfg is None:
            continue
        for name in ("num_hidden_layers", "n_layer", "num_layers"):
            value = getattr(cfg, name, None)
            if value is not None:
                return int(value)
    return 0


def _slug(value: Any, max_len: int = 96) -> str:
    text = _clean_identity_value(value).lower()
    text = re.sub(r"[^a-z0-9_.-]+", "-", text)
    text = text.strip("-._")
    if not text:
        return "unknown"
    return text[:max_len]


def _repo_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "vllm").exists() and (parent / "benchmarks").exists():
            return parent
    return path.parents[2]


def _default_cache_root() -> Path:
    return _repo_root() / "saturo" / "results" / "runtime_profiles"


def _current_gpu_identity() -> Dict[str, Any]:
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            return {"gpu_name": props.name}
    except Exception:
        pass
    return {"gpu_name": "unknown"}


@dataclass(frozen=True)
class TimePredictorProfileIdentity:
    model: str
    dtype: str
    quantization: str
    tensor_parallel_size: int
    pipeline_parallel_size: int
    gpu_name: str
    predictor_version: int = TIME_PREDICTOR_PROFILE_VERSION

    @classmethod
    def from_engine_configs(
        cls,
        *,
        model_config: Any,
        parallel_config: Any,
    ) -> "TimePredictorProfileIdentity":
        gpu = _current_gpu_identity()
        return cls(
            model=_clean_identity_value(getattr(model_config, "model", "")),
            dtype=_clean_identity_value(getattr(model_config, "dtype", "")),
            quantization=_clean_identity_value(
                getattr(model_config, "quantization", None)),
            tensor_parallel_size=int(
                getattr(parallel_config, "tensor_parallel_size", 1)),
            pipeline_parallel_size=int(
                getattr(parallel_config, "pipeline_parallel_size", 1)),
            gpu_name=_clean_identity_value(gpu["gpu_name"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "gpu_name": self.gpu_name,
            "predictor_version": self.predictor_version,
        }

    @property
    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True,
                             separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @property
    def model_slug(self) -> str:
        return _slug(self.model)

    @property
    def file_stem(self) -> str:
        parts = [
            "time_predictor",
            self.model_slug,
            f"gpu-{_slug(self.gpu_name, 40)}",
            f"tp{self.tensor_parallel_size}",
            f"pp{self.pipeline_parallel_size}",
            _slug(self.dtype, 24),
            self.digest,
        ]
        return "_".join(parts)


def time_predictor_cache_path(
    identity: TimePredictorProfileIdentity,
    cache_root: Optional[str | Path] = None,
) -> Path:
    root = Path(cache_root) if cache_root else _default_cache_root()
    return root / identity.model_slug / f"{identity.file_stem}.json"


def _identity_from_profile(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(data, dict):
        return None
    identity = data.get("profile_identity")
    if isinstance(identity, dict):
        return identity
    meta = data.get("meta")
    if isinstance(meta, dict):
        identity = meta.get("profile_identity")
        if isinstance(identity, dict):
            return identity
    nested = data.get("time_predictor")
    if isinstance(nested, dict):
        return _identity_from_profile(nested)
    return None


def _model_identity_matches(saved: Any, expected: Any) -> bool:
    """Accept a Hugging Face repo id and its local checkout as one model."""
    saved_value = _clean_identity_value(saved).replace("\\", "/")
    expected_value = _clean_identity_value(expected).replace("\\", "/")
    if saved_value == expected_value:
        return True
    saved_absolute = saved_value.startswith("/")
    expected_absolute = expected_value.startswith("/")
    if saved_absolute == expected_absolute:
        return False
    local_value = saved_value if saved_absolute else expected_value
    repo_value = expected_value if saved_absolute else saved_value
    repo_parts = [part for part in repo_value.split("/") if part]
    local_parts = [part for part in local_value.split("/") if part]
    return (len(repo_parts) >= 2
            and local_parts[-len(repo_parts):] == repo_parts)


def _profile_identity_matches(saved: Dict[str, Any],
                              expected: Dict[str, Any]) -> bool:
    for key, value in expected.items():
        if key == "model":
            if not _model_identity_matches(saved.get(key), value):
                return False
        elif saved.get(key) != value:
            return False
    return True


def profile_matches_identity(
    profile_path: str | Path,
    identity: TimePredictorProfileIdentity,
) -> bool:
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False
    saved = _identity_from_profile(data)
    if not saved:
        return False
    expected = identity.to_dict()
    return _profile_identity_matches(saved, expected)


def _configured_cache_path(
    configured_path: Optional[str],
    identity: TimePredictorProfileIdentity,
) -> Path:
    if not configured_path:
        return time_predictor_cache_path(identity)
    path = Path(configured_path)
    if path.suffix.lower() == ".json":
        if not path.exists() or profile_matches_identity(path, identity):
            return path
        fallback = time_predictor_cache_path(identity, path.parent / "auto")
        print("[TimePredictor] Ignoring configured profile with mismatched "
              f"identity: {path}; using {fallback}")
        return fallback
    return time_predictor_cache_path(identity, path)


def _group_value(row: dict, group_name: str) -> float:
    stats = row["group_per_layer_ms"][group_name]
    val = stats.get("mid3")
    if val is None or not math.isfinite(float(val)):
        val = stats["median"]
    return float(val)


def _fit_piecewise(tokens: np.ndarray,
                   values: np.ndarray,
                   width: int,
                   min_points: int = 2) -> Dict[str, Any]:
    mask = np.isfinite(tokens) & np.isfinite(values)
    tokens = tokens[mask].astype(float)
    values = values[mask].astype(float)
    if len(values) < 2:
        return {"width": width, "coef": {}, "fallback": [0.0, 0.0]}

    fallback = np.linalg.lstsq(
        np.vstack([tokens, np.ones_like(tokens)]).T,
        values,
        rcond=None,
    )[0]
    coef_by_segment = {}
    for sid in sorted(set(int(math.floor(t / width)) for t in tokens)):
        idx = np.array([int(math.floor(t / width)) == sid for t in tokens])
        if idx.sum() >= min_points:
            coef = np.linalg.lstsq(
                np.vstack([tokens[idx], np.ones(idx.sum())]).T,
                values[idx],
                rcond=None,
            )[0]
        else:
            coef = fallback
        coef_by_segment[str(sid)] = [float(coef[0]), float(coef[1])]
    return {
        "width": width,
        "coef": coef_by_segment,
        "fallback": [float(fallback[0]), float(fallback[1])],
    }


def _fit_attention(rows: List[dict], mode: str) -> Dict[str, Any]:
    selected = [r for r in rows if r.get("mode") == mode]
    selected = [r for r in selected
                if STATEFUL_GROUP in r.get("group_per_layer_ms", {})]
    if len(selected) < 2:
        return {"mode": mode, "n": len(selected)}

    n_tokens = np.array([r["n_tokens"] for r in selected], dtype=float)
    sum_context = np.array([r["sum_context"] for r in selected], dtype=float)
    y = np.array([_group_value(r, STATEFUL_GROUP) for r in selected],
                 dtype=float)
    valid = np.isfinite(n_tokens) & np.isfinite(sum_context) & np.isfinite(y)
    n_tokens, sum_context, y = n_tokens[valid], sum_context[valid], y[valid]
    if len(y) < 2:
        return {"mode": mode, "n": len(y)}
    x = np.stack([n_tokens, sum_context, np.ones_like(n_tokens)], axis=1)
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    return {
        "mode": mode,
        "features": ["n_tokens", "sum_context"],
        "coef": [float(coef[0]), float(coef[1])],
        "intercept": float(coef[2]),
    }


def _fit_profile_results(profile_results: dict,
                         num_layers: int,
                         piecewise_width: int = 32) -> Dict[str, Any]:
    rows = profile_results.get("rows", [])
    fitted = {
        "model_type": "per_layer",
        "num_layers": int(num_layers),
        "meta": profile_results.get("meta", {}),
    }

    linear_rows = [r for r in rows
                   if STATELESS_GROUP in r.get("group_per_layer_ms", {})]
    if linear_rows:
        tokens = np.array([r["n_tokens"] for r in linear_rows], dtype=float)
        values = np.array([_group_value(r, STATELESS_GROUP)
                           for r in linear_rows], dtype=float)
        fitted["linear"] = _fit_piecewise(
            tokens, values, width=piecewise_width)

    prefill = _fit_attention(rows, "prefill")
    decode = _fit_attention(rows, "decode")
    if "coef" in prefill:
        fitted["prefill_attn"] = prefill
    if "coef" in decode:
        fitted["decode_attn"] = decode
    return fitted


def _default_profile_configs(
    num_gpu_blocks: Optional[int] = None,
    block_size: int = DEFAULT_PROFILE_BLOCK_SIZE,
) -> List[Tuple[str, int, int]]:
    """Build the automatic profiling grid.

    Prefill sweeps req_len in [32, 4096] (step 32) x num_req in [1, 128]
    (step 1), dropping combinations whose total prefill tokens
    (num_req * req_len) exceed ``PREFILL_MAX_TOKENS``.

    Decode sweeps req_len in [32, 4096] (step 32) x num_req in [16, 512]
    (step 16).  When ``num_gpu_blocks`` is known (the predictor is built
    after the engine has allocated the KV cache), combinations whose
    KV-cache footprint (ceil(req_len / block_size) * num_req) exceed the
    available GPU blocks are dropped.
    """
    bs = max(1, int(block_size))
    configs: List[Tuple[str, int, int]] = []

    p_len_lo, p_len_hi, p_len_step = PREFILL_REQ_LEN_RANGE
    p_num_lo, p_num_hi, p_num_step = PREFILL_NUM_REQ_RANGE
    for req_len in range(p_len_lo, p_len_hi + 1, p_len_step):
        for num_req in range(p_num_lo, p_num_hi + 1, p_num_step):
            if num_req * req_len > PREFILL_MAX_TOKENS:
                continue
            configs.append(("prefill", req_len, num_req))

    d_len_lo, d_len_hi, d_len_step = DECODE_REQ_LEN_RANGE
    d_num_lo, d_num_hi, d_num_step = DECODE_NUM_REQ_RANGE
    for req_len in range(d_len_lo, d_len_hi + 1, d_len_step):
        for num_req in range(d_num_lo, d_num_hi + 1, d_num_step):
            if num_gpu_blocks is not None:
                blocks = math.ceil(req_len / bs) * num_req
                if blocks > num_gpu_blocks:
                    continue
            configs.append(("decode", req_len, num_req))

    return configs


def _linear_profile_configs() -> List[Tuple[str, int, int]]:
    """Dense single-request prefill sweep to fit the stateless-linear model.

    num_token = req_len sweeps LINEAR_PROFILE_RANGE (default 2..8192 step 2),
    each as a single prefill request (num_req = 1).
    """
    lo, hi, step = LINEAR_PROFILE_RANGE
    return [("prefill", n, 1) for n in range(lo, hi + 1, step)]


def _prefill_attn_profile_configs() -> List[Tuple[str, int, int]]:
    """Prefill grid used to fit the prefill-attention model."""
    configs: List[Tuple[str, int, int]] = []
    p_len_lo, p_len_hi, p_len_step = PREFILL_REQ_LEN_RANGE
    p_num_lo, p_num_hi, p_num_step = PREFILL_NUM_REQ_RANGE
    for req_len in range(p_len_lo, p_len_hi + 1, p_len_step):
        for num_req in range(p_num_lo, p_num_hi + 1, p_num_step):
            if num_req * req_len > PREFILL_MAX_TOKENS:
                continue
            configs.append(("prefill", req_len, num_req))
    return configs


def _decode_attn_profile_configs(
    num_gpu_blocks: Optional[int] = None,
    block_size: int = DEFAULT_PROFILE_BLOCK_SIZE,
) -> List[Tuple[str, int, int]]:
    """Decode grid used to fit the decode-attention model (KV-filtered)."""
    bs = max(1, int(block_size))
    configs: List[Tuple[str, int, int]] = []
    d_len_lo, d_len_hi, d_len_step = DECODE_REQ_LEN_RANGE
    d_num_lo, d_num_hi, d_num_step = DECODE_NUM_REQ_RANGE
    for req_len in range(d_len_lo, d_len_hi + 1, d_len_step):
        for num_req in range(d_num_lo, d_num_hi + 1, d_num_step):
            if num_gpu_blocks is not None:
                blocks = math.ceil(req_len / bs) * num_req
                if blocks > num_gpu_blocks:
                    continue
            configs.append(("decode", req_len, num_req))
    return configs


def _fit_split_profiles(
    linear_rows: List[dict],
    prefill_rows: List[dict],
    decode_rows: List[dict],
    num_layers: int,
    meta: dict,
    piecewise_width: int = DEFAULT_LINEAR_PIECEWISE_WIDTH,
) -> Dict[str, Any]:
    """Fit the per-layer model from three independent profiling passes.

    - linear (stateless) from the dense single-request prefill sweep,
    - prefill_attn from the prefill grid,
    - decode_attn from the decode grid.
    """
    if num_layers <= 0:
        raise ValueError("time predictor requires a positive model layer count")
    fitted: Dict[str, Any] = {
        "model_type": "per_layer",
        "num_layers": int(num_layers),
        "meta": meta,
    }

    lin = [r for r in linear_rows
           if STATELESS_GROUP in r.get("group_per_layer_ms", {})]
    if lin:
        tokens = np.array([r["n_tokens"] for r in lin], dtype=float)
        values = np.array([_group_value(r, STATELESS_GROUP) for r in lin],
                          dtype=float)
        fitted["linear"] = _fit_piecewise(tokens, values, width=piecewise_width)

    prefill = _fit_attention(prefill_rows, "prefill")
    decode = _fit_attention(decode_rows, "decode")
    if "coef" in prefill:
        fitted["prefill_attn"] = prefill
    if "coef" in decode:
        fitted["decode_attn"] = decode
    if ("linear" not in fitted or not fitted["linear"]["coef"]
            or "prefill_attn" not in fitted or "decode_attn" not in fitted):
        raise ValueError(
            "profiling requires at least two finite samples in each of the "
            "linear, prefill-attention and decode-attention passes")
    return fitted


def _write_json(path: str, payload: dict) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def build_or_load_time_predictor(
    fitted_model_path: Optional[str],
    model_executor=None,
    profile_identity: Optional[TimePredictorProfileIdentity] = None,
    num_gpu_blocks: Optional[int] = None,
    block_size: int = DEFAULT_PROFILE_BLOCK_SIZE,
) -> "TimePredictor":
    ignore_existing_profile = False
    if profile_identity is not None:
        resolved_path = _configured_cache_path(fitted_model_path,
                                               profile_identity)
        if resolved_path.exists() and not profile_matches_identity(
                resolved_path, profile_identity):
            print("[TimePredictor] WARNING: cached profile identity mismatch; "
                  f"ignoring {resolved_path}")
            ignore_existing_profile = True
        else:
            predictor = TimePredictor(
                str(resolved_path),
                warn_missing=False,
                expected_identity=profile_identity,
            )
            if predictor.available:
                return predictor
        fitted_model_path = str(resolved_path)
    elif fitted_model_path is None:
        fitted_model_path = str(_default_cache_root() /
                                "time_predictor_model.json")

    if not ignore_existing_profile:
        predictor = TimePredictor(
            fitted_model_path, warn_missing=False)
        if predictor.available:
            return predictor
    if model_executor is None:
        print(f"[TimePredictor] WARNING: fitted model not found at "
              f"{fitted_model_path}")
        return TimePredictor(
            fitted_model_path,
            warn_missing=False,
            expected_identity=profile_identity,
        )

    if os.environ.get("SATURO_SKIP_TIME_PREDICTOR_PROFILE", "0") == "1":
        print("[TimePredictor] SATURO_SKIP_TIME_PREDICTOR_PROFILE=1; skipping "
              f"live profiling. No cached profile at {fitted_model_path}; "
              "predictor will be unavailable.")
        return TimePredictor(
            fitted_model_path,
            warn_missing=False,
            expected_identity=profile_identity,
        )

    # Live profiling drives bare per-stage model forwards, which do not carry
    # the intermediate tensors pipeline stages exchange. Under PP>1 the
    # non-first stages crash, so skip live profiling entirely.
    if (profile_identity is not None
            and int(getattr(profile_identity, "pipeline_parallel_size", 1))
            > 1):
        print("[TimePredictor] pipeline_parallel_size > 1; live profiling is "
              f"unsupported under PP. No cached profile at {fitted_model_path};"
              " predictor will be unavailable.")
        return TimePredictor(
            fitted_model_path,
            warn_missing=False,
            expected_identity=profile_identity,
        )

    print(f"[TimePredictor] Profile info not found at {fitted_model_path}; "
          "profiling during engine initialization "
          "(3 passes: linear, prefill-attn, decode-attn).")
    profile_start = time.perf_counter()

    def _env_int(name, default, minimum):
        try:
            return max(minimum, int(os.environ.get(name, default)))
        except (TypeError, ValueError):
            return default

    warmup = _env_int("SATURO_PROFILE_WARMUP", DEFAULT_PROFILE_WARMUP, 1)
    repeat = _env_int("SATURO_PROFILE_REPEAT", DEFAULT_PROFILE_REPEAT, 1)
    warmup_per_config = _env_int("SATURO_PROFILE_WARMUP_PER_CONFIG", 0, 0)

    def _run_pass(name, configs):
        t0 = time.perf_counter()
        res = model_executor.profile_qwen2_module_times(
            warmup=warmup,
            repeat=repeat,
            warmup_per_config=warmup_per_config,
            configs=configs,
            profile_num_layers=DEFAULT_PROFILE_NUM_LAYERS,
            profile_layer_start=DEFAULT_PROFILE_LAYER_START,
            profile_kv_cache_fraction=DEFAULT_PROFILE_KV_CACHE_FRACTION,
        )
        rows = (res or {}).get("rows", [])
        dt = time.perf_counter() - t0
        print(f"[TimePredictor][timing] {name} pass: {len(configs)} configs -> "
              f"{len(rows)} rows in {dt:.1f}s "
              f"(warmup={warmup} repeat={repeat} "
              f"warmup_per_config={warmup_per_config})", flush=True)
        return res or {}

    try:
        linear_res = _run_pass("linear", _linear_profile_configs())
        prefill_res = _run_pass("prefill-attn", _prefill_attn_profile_configs())
        decode_res = _run_pass(
            "decode-attn",
            _decode_attn_profile_configs(num_gpu_blocks=num_gpu_blocks,
                                         block_size=block_size))
    except Exception as exc:
        print("[TimePredictor] WARNING: automatic profiling failed; "
              f"continuing without a predictor: {exc}")
        return TimePredictor(
            fitted_model_path,
            warn_missing=False,
            expected_identity=profile_identity,
        )
    profile_elapsed_s = time.perf_counter() - profile_start
    linear_rows = linear_res.get("rows", [])
    prefill_rows = prefill_res.get("rows", [])
    decode_rows = decode_res.get("rows", [])
    print(
        "[TimePredictor][timing] 3-pass profiling took "
        f"{profile_elapsed_s:.2f}s ({profile_elapsed_s / 60.0:.2f} min); "
        f"linear={len(linear_rows)} prefill={len(prefill_rows)} "
        f"decode={len(decode_rows)} rows",
        flush=True,
    )
    meta = dict(linear_res.get("meta", {}))
    meta["pass_row_counts"] = {
        "linear": len(linear_rows),
        "prefill_attn": len(prefill_rows),
        "decode_attn": len(decode_rows),
    }
    if profile_identity is not None:
        meta["profile_identity"] = profile_identity.to_dict()
        meta["profile_key"] = profile_identity.digest
    # Keep the legacy single-"rows" raw-profile format so downstream readers
    # keep working. Each row carries a "profile_pass" tag so the originating
    # pass can still be recovered if needed.
    merged_rows: List[dict] = []
    for pass_name, pass_rows in (("linear", linear_rows),
                                 ("prefill_attn", prefill_rows),
                                 ("decode_attn", decode_rows)):
        for r in pass_rows:
            r = dict(r)
            r["profile_pass"] = pass_name
            merged_rows.append(r)
    raw_profile_path = f"{fitted_model_path}.raw_profile.json"
    _write_json(raw_profile_path, {
        "meta": meta,
        "rows": merged_rows,
    })
    num_layers = _resolve_num_layers(model_executor)
    try:
        fitted = _fit_split_profiles(
            linear_rows,
            prefill_rows,
            decode_rows,
            num_layers=num_layers,
            meta=meta,
            piecewise_width=DEFAULT_LINEAR_PIECEWISE_WIDTH,
        )
    except ValueError as exc:
        print("[TimePredictor] WARNING: incomplete profiling results; "
              f"no fitted model was saved: {exc}")
        return TimePredictor(
            fitted_model_path,
            warn_missing=False,
            expected_identity=profile_identity,
        )
    if profile_identity is not None:
        fitted["profile_identity"] = profile_identity.to_dict()
        fitted["profile_key"] = profile_identity.digest
    payload = {
        "time_predictor": fitted,
        "kv_cache_block": {
            "total_gpu_blocks": (
                int(num_gpu_blocks) if num_gpu_blocks is not None else None),
            "block_size": int(block_size),
        },
    }
    _write_json(fitted_model_path, payload)
    print(f"[TimePredictor] Saved raw profile to {raw_profile_path}")
    print(f"[TimePredictor] Saved fitted model to {fitted_model_path}")
    return TimePredictor(
        fitted_model_path,
        expected_identity=profile_identity,
    )


class TimePredictor:
    """
    Predict per-iteration forward time for mixed prefill+decode batches.

    Supports two model types:
    1. iteration_level: direct prediction of iteration time in ms.
    2. per_layer: modular attention + stateless prediction per layer,
       multiplied by the number of transformer layers.

    ``num_layers`` is a fallback for legacy profiles that omit their layer
    count. A layer count stored in the profile takes precedence.
    """

    def __init__(self,
                 fitted_model_path: str,
                 warn_missing: bool = True,
                 expected_identity: Optional[
                     TimePredictorProfileIdentity] = None,
                 num_layers: Optional[int] = None):
        if num_layers is not None and num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.num_layers = int(num_layers) if num_layers is not None else 0
        self.model_type = "per_layer"
        self.available = False
        self.profile_path = fitted_model_path
        self.profile_identity: Optional[Dict[str, Any]] = None
        self.total_gpu_blocks: Optional[int] = None
        self.kv_block_size: Optional[int] = None
        self.latency_scale = 1.0
        self.latency_offset_ms = 0.0
        self.runtime_overhead = RuntimeOverheadModel()
        # Runtime-overhead correction is applied by default. Set
        # SATURO_DISABLE_RUNTIME_OVERHEAD=1, or assign False on the instance,
        # to get the uncorrected forward time instead. The environment is read
        # once here because predict_* runs in simulator hot loops.
        self.apply_runtime_overhead = os.environ.get(
            "SATURO_DISABLE_RUNTIME_OVERHEAD", "0") != "1"

        self.iter_coef = None
        self.iter_intercept = 0.0

        self.prefill_coef = (0.0, 0.0, 0.0)
        self.decode_coef = (0.0, 0.0, 0.0)
        self.decode_full_coef = None
        self.linear_model = PiecewiseLinearModel()
        self.linear_tile_model = TileQuantizedLinearModel()

        if os.path.exists(fitted_model_path):
            self._load_and_fit(fitted_model_path)
            if expected_identity is not None:
                saved = self.profile_identity
                expected = expected_identity.to_dict()
                if (not saved
                        or not _profile_identity_matches(saved, expected)):
                    print("[TimePredictor] WARNING: profile identity does not "
                          f"match this engine; skipping {fitted_model_path}")
                    return
            self._validate_profile()
            self.available = True
        elif warn_missing:
            print(f"[TimePredictor] WARNING: fitted model not found at {fitted_model_path}")

    def _load_and_fit(self, json_path: str) -> None:
        with open(json_path, "r", encoding="utf-8") as f:
            if f.readline().strip() == "version https://git-lfs.github.com/spec/v1":
                raise ValueError(
                    f"{json_path} is a Git LFS pointer; retrieve the profile "
                    "with git lfs pull before loading it")
            f.seek(0)
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("time predictor profile must be a JSON object")

        self.profile_identity = _identity_from_profile(data)
        # New nested layout separates the fitted predictor from the KV cache
        # block info: {"time_predictor": {...}, "kv_cache_block": {...}}.
        # Fall back to the legacy flat layout for older profile files.
        tp_data = data.get("time_predictor")
        if isinstance(tp_data, dict):
            kv = data.get("kv_cache_block")
            if isinstance(kv, dict):
                self.total_gpu_blocks = kv.get("total_gpu_blocks")
                self.kv_block_size = kv.get("block_size")
            data = tp_data
        self.model_type = data.get("model_type", "per_layer")
        self.latency_scale = float(data.get("latency_scale",
                                            self.latency_scale))
        self.latency_offset_ms = float(data.get(
            "latency_offset_ms", self.latency_offset_ms))
        overhead = data.get("runtime_overhead", data.get("scheduler_overhead"))
        if isinstance(overhead, dict):
            self.runtime_overhead.load_dict(overhead)
        if self.model_type == "iteration_level":
            self.iter_coef = [float(x) for x in data["coef"]]
            self.iter_intercept = float(data["intercept"])
            r2 = data.get("r2")
            mape = data.get("mape")
            quality = []
            if r2 is not None:
                quality.append(f"R2={float(r2):.4f}")
            if mape is not None:
                quality.append(f"MAPE={float(mape):.1f}%")
            suffix = f" ({', '.join(quality)})" if quality else ""
            print(f"[TimePredictor] Loaded iteration-level model{suffix}")
            return

        self.num_layers = int(data.get("num_layers", self.num_layers))

        if "prefill_attn" in data:
            pa = data["prefill_attn"]
            self.prefill_coef = (
                float(pa["coef"][0]),
                float(pa["coef"][1]),
                float(pa["intercept"]),
            )
        if "decode_attn" in data:
            da = data["decode_attn"]
            self.decode_coef = (
                float(da["coef"][0]),
                float(da["coef"][1]),
                float(da["intercept"]),
            )
        if data.get("use_decode_full_layer", False) and "decode_full_layer" in data:
            df = data["decode_full_layer"]
            self.decode_full_coef = (
                float(df["coef"][0]),
                float(df["coef"][1]),
                float(df["intercept"]),
            )
        if "linear" in data:
            self.linear_model.load_dict(data["linear"])
            tile_override = data.get("linear_tile_override")
            if isinstance(tile_override, dict):
                self.linear_tile_model.load_dict(tile_override)
            if "prefill_attn" in data and "decode_attn" in data:
                return

        if "prefill_attn_ms" in data:
            import pandas as pd

            df_p = pd.DataFrame(data["prefill_attn_ms"])
            L = df_p["req_len"].to_numpy(dtype=float)
            B = df_p["num_req"].to_numpy(dtype=float)
            n = L * B
            sum_ctx = B * (L * (L + 1.0) / 2.0)
            X = np.stack([n, sum_ctx], axis=1)
            y = df_p["ms_median"].to_numpy(dtype=float)
            mask = np.isfinite(y)
            if mask.sum() > 0:
                X_m, y_m = X[mask], y[mask]
                X_aug = np.hstack([X_m, np.ones((len(X_m), 1))])
                coef, _, _, _ = np.linalg.lstsq(X_aug, y_m, rcond=None)
                self.prefill_coef = (
                    float(coef[0]),
                    float(coef[1]),
                    float(coef[2]),
                )

        if "decode_attn_ms" in data:
            import pandas as pd

            df_d = pd.DataFrame(data["decode_attn_ms"])
            L = df_d["req_len"].to_numpy(dtype=float)
            B = df_d["num_req"].to_numpy(dtype=float)
            n = B
            sum_ctx = B * L
            X = np.stack([n, sum_ctx], axis=1)
            y = df_d["ms_median"].to_numpy(dtype=float)
            mask = np.isfinite(y)
            if mask.sum() > 0:
                X_m, y_m = X[mask], y[mask]
                X_aug = np.hstack([X_m, np.ones((len(X_m), 1))])
                coef, _, _, _ = np.linalg.lstsq(X_aug, y_m, rcond=None)
                self.decode_coef = (
                    float(coef[0]),
                    float(coef[1]),
                    float(coef[2]),
                )

        if "linear_ms" in data:
            import pandas as pd

            df_l = pd.DataFrame(data["linear_ms"])
            tokens = df_l["num_token"].to_numpy(dtype=float)
            ms_col = "ms_median" if "ms_median" in df_l.columns else "ms_mean"
            ms = df_l[ms_col].to_numpy(dtype=float)
            mask = np.isfinite(tokens) & np.isfinite(ms)
            tokens = tokens[mask]
            ms = ms[mask]

            self.linear_model.coef = {}
            if len(tokens) > 0:
                A = np.vstack([tokens, np.ones_like(tokens)]).T
                a_g, b_g = np.linalg.lstsq(A, ms, rcond=None)[0]
                self.linear_model.fallback = (float(a_g), float(b_g))

                seg_ids = np.array(
                    [self.linear_model.segment_id(t) for t in tokens],
                    dtype=int,
                )
                for sid in np.unique(seg_ids):
                    idx = np.where(seg_ids == sid)[0]
                    if len(idx) < 2:
                        continue
                    t_seg = tokens[idx]
                    y_seg = ms[idx]
                    A_seg = np.vstack([t_seg, np.ones_like(t_seg)]).T
                    a, b = np.linalg.lstsq(A_seg, y_seg, rcond=None)[0]
                    self.linear_model.coef[int(sid)] = (float(a), float(b))
            self.linear_model.repair_first_segment()

    def _validate_profile(self) -> None:
        values = [self.latency_scale, self.latency_offset_ms]
        values.extend(self.runtime_overhead.to_dict().values())
        if self.model_type == "iteration_level":
            if self.iter_coef is None or len(self.iter_coef) != 4:
                raise ValueError("iteration-level model requires four coefficients")
            values.extend(self.iter_coef)
            values.append(self.iter_intercept)
        elif self.model_type == "per_layer":
            if self.num_layers <= 0:
                raise ValueError("per-layer model requires a positive num_layers")
            if self.linear_model.width <= 0:
                raise ValueError("linear model width must be positive")
            values.extend(self.prefill_coef)
            values.extend(self.decode_coef)
            values.extend(self.decode_full_coef or ())
            values.extend(self.linear_model.fallback)
            for coefficient in (self.linear_model.coef or {}).values():
                values.extend(coefficient)
            values.extend(self.linear_tile_model.per_layer_values_ms or ())
        else:
            raise ValueError(f"unsupported time predictor model_type: {self.model_type}")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("time predictor coefficients must be finite")
        if self.latency_scale <= 0:
            raise ValueError("latency_scale must be positive")

    def predict_prefill_ms(self, req_len: int, num_req: int) -> float:
        L = float(req_len)
        B = float(num_req)
        n = L * B
        sum_ctx = B * (L * (L + 1.0) / 2.0)
        a, b, c = self.prefill_coef
        return max(0.0, a * n + b * sum_ctx + c)

    def predict_decode_ms(self, req_len: int, num_req: int) -> float:
        L = float(req_len)
        B = float(num_req)
        n = B
        sum_ctx = B * L
        a, b, c = self.decode_coef
        return max(0.0, a * n + b * sum_ctx + c)

    def predict_linear_ms(self, num_token: int) -> float:
        tile_value = self.linear_tile_model.predict_one(float(num_token))
        if tile_value is not None:
            return tile_value
        return self.linear_model.predict_one(float(num_token))

    def predict_forward_ms(
        self,
        prefill_reqs: List[Tuple[int, int]],
        decode_reqs: List[Tuple[int, int]],
    ) -> float:
        prefill_tokens = 0.0
        prefill_sum_ctx = 0.0
        for query_len, context_len in prefill_reqs:
            query_len = float(query_len)
            context_len = float(context_len)
            cached_len = max(0.0, context_len - query_len)
            prefill_tokens += query_len
            prefill_sum_ctx += (
                query_len * cached_len
                + query_len * (query_len + 1.0) / 2.0
            )

        decode_n = 0.0
        decode_sum_ctx = 0.0
        for query_len, context_len in decode_reqs:
            query_len = float(query_len)
            context_len = float(context_len)
            decode_n += query_len
            decode_sum_ctx += query_len * context_len

        return self.predict_forward_from_aggregates_ms(
            prefill_tokens=prefill_tokens,
            prefill_sum_ctx=prefill_sum_ctx,
            decode_tokens=decode_n,
            decode_sum_ctx=decode_sum_ctx,
            prefill_request_count=len(prefill_reqs),
            decode_request_count=len(decode_reqs),
        )

    def predict_forward_from_aggregates_ms(
        self,
        *,
        prefill_tokens: float,
        prefill_sum_ctx: float,
        decode_tokens: float,
        decode_sum_ctx: float,
        prefill_request_count: int,
        decode_request_count: int,
    ) -> float:
        """Predict from sufficient batch statistics.

        This is mathematically identical to :meth:`predict_forward_ms` but
        avoids rebuilding and rescanning thousands of ``(query, context)``
        tuples for every Oracle look-ahead snapshot.
        """
        if prefill_tokens == 0 and decode_tokens == 0:
            return 0.0
        if self.model_type == "iteration_level":
            raw_ms = self._predict_iter_level(
                prefill_tokens,
                prefill_sum_ctx,
                decode_tokens,
                decode_sum_ctx,
            )
            return self._add_runtime_overhead_ms(
                self._calibrate_ms(raw_ms),
                prefill_request_count=int(prefill_request_count),
                decode_request_count=int(decode_request_count),
                prefill_tokens=prefill_tokens,
                decode_tokens=decode_tokens,
            )

        raw_ms = self._predict_per_layer(
            prefill_tokens,
            prefill_sum_ctx,
            decode_tokens,
            decode_sum_ctx,
        )
        return self._add_runtime_overhead_ms(
            self._calibrate_ms(raw_ms),
            prefill_request_count=int(prefill_request_count),
            decode_request_count=int(decode_request_count),
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
        )

    def _calibrate_ms(self, raw_ms: float) -> float:
        return max(0.0, raw_ms * self.latency_scale + self.latency_offset_ms)

    def predict_forward_from_aggregates_batch_ms(
        self,
        *,
        prefill_tokens,
        prefill_sum_ctx,
        decode_tokens,
        decode_sum_ctx,
        prefill_request_count,
        decode_request_count,
    ) -> np.ndarray:
        """Predict broadcast-compatible sufficient-statistics arrays.

        The iteration-level fast path keeps the scalar multiplication,
        addition and clamp order. Legacy models and customized scalar hooks
        retain their original per-row implementation. No forecast is removed.
        """
        fields = (prefill_tokens, prefill_sum_ctx, decode_tokens,
                  decode_sum_ctx, prefill_request_count, decode_request_count)
        arrays = np.broadcast_arrays(*(np.asarray(value) for value in fields))

        def scalar_rows():
            output = np.empty(arrays[0].shape, dtype=np.float64)
            names = ("prefill_tokens", "prefill_sum_ctx", "decode_tokens",
                     "decode_sum_ctx", "prefill_request_count",
                     "decode_request_count")
            for index in np.ndindex(output.shape):
                output[index] = self.predict_forward_from_aggregates_ms(
                    **{name: value[index]
                       for name, value in zip(names, arrays)})
            return output

        hooks = (
            (self.predict_forward_from_aggregates_ms,
             TimePredictor.predict_forward_from_aggregates_ms),
            (self._predict_iter_level, TimePredictor._predict_iter_level),
            (self._calibrate_ms, TimePredictor._calibrate_ms),
            (self._add_runtime_overhead_ms,
             TimePredictor._add_runtime_overhead_ms),
        )
        if self.apply_runtime_overhead:
            hooks += ((getattr(self.runtime_overhead, "predict_ms", None),
                       RuntimeOverheadModel.predict_ms),)
        if (self.model_type != "iteration_level"
                or any(getattr(method, "__func__", None) is not expected
                       for method, expected in hooks)):
            return scalar_rows()
        # Smaller floating dtypes have different scalar promotion rules.
        # Unusual objects and very large integer sums use the exact fallback.
        for values in arrays[:4]:
            if (values.dtype.kind not in "biuf"
                    or (values.dtype.kind == "f" and values.dtype.itemsize != 8)
                    or not np.all(np.isfinite(values))
                    or (values.dtype.kind in "biu"
                        and np.any((values < -(2**51)) | (values > 2**51)))):
                return scalar_rows()
        for values in arrays[4:]:
            if (values.dtype.kind not in "biuf"
                    or not np.all(np.isfinite(values))
                    or np.any((values < -(2**31)) | (values > 2**31))):
                return scalar_rows()

        pt, pc, dt, dc = (values.astype(np.float64, copy=False)
                          for values in arrays[:4])
        # Match int(request_count), including truncation toward zero, before
        # arithmetic. The validated range keeps sums exactly representable.
        pr, dr = (values.astype(np.int64).astype(np.float64)
                  for values in arrays[4:])
        active = (pt != 0.0) | (dt != 0.0)
        def clamp_zero_first(values):
            # Python max(0.0, value) chooses +0 for NaN and signed-zero ties.
            return np.where(values > 0.0, values, 0.0)
        with np.errstate(over="ignore", invalid="ignore"):
            if self.iter_coef is None:
                raw = np.zeros(pt.shape, dtype=np.float64)
            else:
                c = self.iter_coef
                raw = clamp_zero_first(c[0] * pt + c[1] * pc + c[2] * dt
                                       + c[3] * dc + self.iter_intercept)
            forward = clamp_zero_first(
                raw * self.latency_scale + self.latency_offset_ms)
            if self.apply_runtime_overhead:
                h = self.runtime_overhead
                total_tokens = pt + dt
                active_requests = pr + dr
                overhead = clamp_zero_first(
                    h.constant_ms
                    + h.per_query_token_ms * total_tokens
                    + h.per_query_token_above_hinge_ms * np.maximum(
                        total_tokens - h.query_token_hinge_threshold, 0.0)
                    + h.per_prefill_request_ms * pr
                    + h.per_decode_request_ms * dr
                    + h.per_active_request_ms * active_requests
                    + h.per_prefill_token_ms * pt
                    + h.per_decode_token_ms * dt
                    + h.per_decode_request_above_hinge_ms * np.maximum(
                        dr - h.decode_request_hinge_threshold, 0.0))
                forward = clamp_zero_first(forward + overhead)
        return np.where(active, forward, 0.0)

    def _add_runtime_overhead_ms(
        self,
        forward_ms: float,
        *,
        prefill_request_count: int,
        decode_request_count: int,
        prefill_tokens: float,
        decode_tokens: float,
    ) -> float:
        if not self.apply_runtime_overhead:
            return max(0.0, forward_ms)
        overhead_ms = self.runtime_overhead.predict_ms(
            prefill_request_count=prefill_request_count,
            decode_request_count=decode_request_count,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
        )
        return max(0.0, forward_ms + overhead_ms)

    def _predict_iter_level(
        self,
        prefill_tokens: float,
        prefill_sum_ctx: float,
        decode_n: float,
        decode_sum_ctx: float,
    ) -> float:
        c = self.iter_coef
        if c is None:
            return 0.0
        return max(
            0.0,
            c[0] * prefill_tokens
            + c[1] * prefill_sum_ctx
            + c[2] * decode_n
            + c[3] * decode_sum_ctx
            + self.iter_intercept,
        )

    def _predict_per_layer(
        self,
        prefill_tokens: float,
        prefill_sum_ctx: float,
        decode_n: float,
        decode_sum_ctx: float,
    ) -> float:
        total_tokens = prefill_tokens + decode_n

        prefill_attn = 0.0
        if prefill_tokens > 0:
            a, b, c = self.prefill_coef
            prefill_attn = max(0.0, a * prefill_tokens + b * prefill_sum_ctx + c)

        decode_ms = 0.0
        if decode_n > 0:
            if self.decode_full_coef is not None:
                a, b, c = self.decode_full_coef
            else:
                a, b, c = self.decode_coef
            decode_ms = max(0.0, a * decode_n + b * decode_sum_ctx + c)

        linear_ms = self.predict_linear_ms(total_tokens) if total_tokens > 0 else 0.0
        if self.decode_full_coef is not None and decode_n > 0:
            linear_ms = (self.predict_linear_ms(prefill_tokens)
                         if prefill_tokens > 0 else 0.0)

        return (prefill_attn + decode_ms + linear_ms) * self.num_layers
