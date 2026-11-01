#!/usr/bin/env bash
# Orchestrate time_predictor profiling across all model/TP configs, packing the
# 4 local GPUs: TP-4 configs run first (each uses all 4 GPUs, sequential), then
# TP-2 configs run as parallel pairs (GPU 0,1 and 2,3), then TP-1 configs run in
# parallel (GPU 0 and 1).
#
# Requires HF_TOKEN in the environment (for the gated Mistral repo). This script
# never stores the token; pass it via the environment at launch time.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

LOG_DIR="saturo/results/profile_logs"
mkdir -p "${LOG_DIR}"

QWEN72B="/root/models/Qwen/Qwen2.5-72B-Instruct"
QWEN32B="Qwen/Qwen2.5-32B-Instruct"
MISTRAL="mistralai/Mistral-Large-Instruct-2411"

# run <tag> <model> <tp> <cuda_visible_devices>
run() {
    local tag="$1" model="$2" tp="$3" cvd="$4"
    echo "[orchestrate] START ${tag} (model=${model} tp=${tp} gpus=${cvd}) $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES="${cvd}" MODEL="${model}" TP="${tp}" \
        bash benchmarks/run_profile_one.sh > "${LOG_DIR}/${tag}.log" 2>&1
    local rc=$?
    if [[ ${rc} -eq 0 ]]; then
        echo "[orchestrate] DONE  ${tag} $(date +%H:%M:%S)"
    else
        echo "[orchestrate] FAIL  ${tag} rc=${rc} (see ${LOG_DIR}/${tag}.log) $(date +%H:%M:%S)"
    fi
}

echo "===== PHASE 1: TP-4 (sequential, all 4 GPUs) ====="
run qwen72b_tp4  "${QWEN72B}" 4 "0,1,2,3"
run qwen32b_tp4  "${QWEN32B}" 4 "0,1,2,3"
run mistral_tp4  "${MISTRAL}" 4 "0,1,2,3"

echo "===== PHASE 2: TP-2 (parallel: GPU 0,1 || GPU 2,3) ====="
run qwen32b_tp2  "${QWEN32B}" 2 "0,1" &
P1=$!
run mistral_tp2  "${MISTRAL}" 2 "2,3" &
P2=$!
wait ${P1} ${P2}

echo "===== PHASE 3: TP-1 (parallel: GPU 0 || GPU 1) ====="
run qwen32b_tp1  "${QWEN32B}" 1 "0" &
P3=$!
run qwen72b_tp1  "${QWEN72B}" 1 "1" &
P4=$!
wait ${P3} ${P4}

echo "===== ALL PROFILING DONE $(date +%H:%M:%S) ====="
