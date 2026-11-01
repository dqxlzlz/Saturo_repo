#!/usr/bin/env bash
# Trigger time_predictor profiling for each TP and validate accuracy against the
# prefill step-latency benchmark ground truth.
set -euo pipefail

MODEL="${MODEL:-/root/models/Qwen/Qwen2.5-72B-Instruct}"
LOAD_FORMAT="${LOAD_FORMAT:-dummy}"
DTYPE="${DTYPE:-float16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
TP_SIZES="${TP_SIZES:-1 2 4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8448}"
BENCH_DIR="${BENCH_DIR:-saturo/results/prefill_step_latency}"
OUTPUT_DIR="${OUTPUT_DIR:-saturo/results/prefill_step_latency/predictor_accuracy}"

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
# Ensure live profiling runs (do NOT skip).
export SATURO_SKIP_TIME_PREDICTOR_PROFILE=0

mkdir -p "${OUTPUT_DIR}"

for TP in ${TP_SIZES}; do
    echo ""
    echo "-------- profiling + validating TP=${TP} --------"
    "${PYTHON_BIN}" benchmarks/profile_and_validate_time_predictor.py \
        --model "${MODEL}" \
        --tensor-parallel-size "${TP}" \
        --dtype "${DTYPE}" \
        --load-format "${LOAD_FORMAT}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --benchmark-json "${BENCH_DIR}/prefill_step_latency_tp${TP}.json" \
        --output-json "${OUTPUT_DIR}/predictor_accuracy_tp${TP}.json"
done

echo ""
echo "All done. Accuracy reports in ${OUTPUT_DIR}"
