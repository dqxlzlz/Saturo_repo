#!/usr/bin/env bash
# Profile time_predictor for ONE model + TP config. Honors CUDA_VISIBLE_DEVICES
# (set by the caller to pin specific GPUs) and SATURO_PROFILE_* overrides.
#
# Usage:
#   MODEL=/root/models/Qwen/Qwen2.5-72B-Instruct TP=4 bash run_profile_one.sh
set -euo pipefail

MODEL="${MODEL:?set MODEL}"
TP="${TP:?set TP}"
DTYPE="${DTYPE:-float16}"
LOAD_FORMAT="${LOAD_FORMAT:-dummy}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8448}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x "${PROJECT_ROOT}/.venv/bin/python3" ]]; then
    PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python3"
fi

VENV_TORCH_LIB="$("${PYTHON_BIN}" -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))' 2>/dev/null || true)"
if [[ -n "${VENV_TORCH_LIB}" && -d "${VENV_TORCH_LIB}" ]]; then
    export LD_LIBRARY_PATH="${VENV_TORCH_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

export VLLM_USE_V1=0
export VLLM_TARGET_DEVICE=cuda
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export SATURO_SKIP_TIME_PREDICTOR_PROFILE=0
# Clock-warmup jitter fix defaults (override from caller if desired).
export SATURO_PROFILE_WARMUP="${SATURO_PROFILE_WARMUP:-5}"
export SATURO_PROFILE_REPEAT="${SATURO_PROFILE_REPEAT:-15}"
export SATURO_PROFILE_WARMUP_PER_CONFIG="${SATURO_PROFILE_WARMUP_PER_CONFIG:-3}"

echo "[run_profile_one] MODEL=${MODEL} TP=${TP} DTYPE=${DTYPE} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>}"
"${PYTHON_BIN}" benchmarks/profile_time_predictor.py \
    --model "${MODEL}" \
    --tensor-parallel-size "${TP}" \
    --dtype "${DTYPE}" \
    --load-format "${LOAD_FORMAT}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}"
