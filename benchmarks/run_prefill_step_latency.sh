#!/usr/bin/env bash
# Prefill step-latency sweep for Qwen2.5-72B across TP sizes.
#
# Measures the per-step forward time (CUDA-event actual_time_ms) of a single
# full prefill request at each length in PREFILL_LENS, for each TP in TP_SIZES.
# Time_predictor auto-profiling is disabled during these runs.
#
# Usage:
#   ./benchmarks/run_prefill_step_latency.sh
#   TP_SIZES="1 2 4" PREFILL_LENS="32,64,128,256,512,1024,2048,4096,8192" \
#       ./benchmarks/run_prefill_step_latency.sh
set -euo pipefail

MODEL="${MODEL:-/root/models/Qwen/Qwen2.5-72B-Instruct}"
LOAD_FORMAT="${LOAD_FORMAT:-dummy}"
DTYPE="${DTYPE:-float16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
TP_SIZES="${TP_SIZES:-1 2 4}"
PREFILL_LENS="${PREFILL_LENS:-32,64,128,256,512,1024,2048,4096,8192}"
NUM_REQUESTS="${NUM_REQUESTS:-20}"
WARMUP="${WARMUP:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-saturo/results/prefill_step_latency}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x "${PROJECT_ROOT}/.venv/bin/python3" ]]; then
    PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python3"
fi

# Match run_t_pool_saturation.sh: bind the prebuilt vLLM CUDA extensions to the
# .venv PyTorch libs (avoids libc10_cuda undefined-symbol import errors).
VENV_TORCH_LIB="$("${PYTHON_BIN}" -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))' 2>/dev/null || true)"
if [[ -n "${VENV_TORCH_LIB}" && -d "${VENV_TORCH_LIB}" ]]; then
    export LD_LIBRARY_PATH="${VENV_TORCH_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

export VLLM_USE_V1=0
export VLLM_TARGET_DEVICE=cuda
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# Disable time_predictor live profiling during the benchmark.
export SATURO_SKIP_TIME_PREDICTOR_PROFILE=1

mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "  Prefill step-latency sweep"
echo "  model        : ${MODEL}"
echo "  tp_sizes     : ${TP_SIZES}"
echo "  prefill_lens : ${PREFILL_LENS}"
echo "  num_requests : ${NUM_REQUESTS} (warmup ${WARMUP})"
echo "  load_format  : ${LOAD_FORMAT}  dtype: ${DTYPE}"
echo "  output_dir   : ${OUTPUT_DIR}"
echo "============================================"

for TP in ${TP_SIZES}; do
    echo ""
    echo "-------- TP=${TP} --------"
    "${PYTHON_BIN}" benchmarks/prefill_step_latency.py \
        --model "${MODEL}" \
        --tensor-parallel-size "${TP}" \
        --prefill-lens "${PREFILL_LENS}" \
        --num-requests "${NUM_REQUESTS}" \
        --warmup "${WARMUP}" \
        --dtype "${DTYPE}" \
        --load-format "${LOAD_FORMAT}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --output-json "${OUTPUT_DIR}/prefill_step_latency_tp${TP}.json"
done

echo ""
echo "All done. Results in ${OUTPUT_DIR}"
