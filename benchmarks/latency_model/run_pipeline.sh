#!/bin/bash
# run_pipeline.sh — Full latency modeling pipeline
#
# Steps:
#   1. Profile components (or use existing data)
#   2. Fit regression models
#   3. Validate against Azure trace (first 1000 requests)
#
# Usage:
#   bash benchmarks/latency_model/run_pipeline.sh          # full pipeline
#   SKIP_PROFILE=1 bash benchmarks/latency_model/run_pipeline.sh  # skip profiling
#   SKIP_VALIDATE=1 bash benchmarks/latency_model/run_pipeline.sh  # skip validation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

# ========================= Config =========================
MODEL="${MODEL_PATH:-/root/models/Qwen/Qwen2.5-0.5B-Instruct}"
TRACE_CSV="data/datasets/AzureLLMInferenceTrace_conv.csv"
PROFILE_OUTPUT="profile_bench_results_new.json"
FITTED_MODEL="fitted_model.json"
VALIDATION_OUTPUT="validation_results.csv"

TP=1
MAX_MODEL_LEN=8192
MAX_BATCHED_TOKENS=16384
MAX_NUM_SEQS=256

WARMUP=3
REPEAT=5
NUM_LAYER_TO_RUN=5
PIECEWISE_WIDTH=32
NUM_REQUESTS=1000

GPU="${CUDA_VISIBLE_DEVICES:-0}"
SKIP_PROFILE="${SKIP_PROFILE:-0}"
SKIP_VALIDATE="${SKIP_VALIDATE:-0}"

echo "============================================================"
echo "Latency Modeling Pipeline"
echo "  Model: $MODEL"
echo "  Trace: $TRACE_CSV"
echo "  GPU:   $GPU"
echo "============================================================"

# ========================= Step 1: Profile + Fit =========================
if [ "$SKIP_PROFILE" = "1" ]; then
    echo ""
    echo ">>> Step 1: Fitting from existing profiling data..."
    python benchmarks/latency_model/profile_and_fit.py \
        --skip-profiling \
        --profile-input "$PROFILE_OUTPUT" \
        --model-output "$FITTED_MODEL" \
        --piecewise-width $PIECEWISE_WIDTH
else
    echo ""
    echo ">>> Step 1: Profile components + Fit models..."
    CUDA_VISIBLE_DEVICES=$GPU VLLM_USE_V1=0 \
    python benchmarks/latency_model/profile_and_fit.py \
        --model "$MODEL" \
        -tp $TP \
        --enforce-eager \
        --load-format dummy \
        --max-model-len $MAX_MODEL_LEN \
        --max-num-batched-tokens $MAX_BATCHED_TOKENS \
        --max-num-seqs $MAX_NUM_SEQS \
        --warmup $WARMUP \
        --repeat $REPEAT \
        --num-layer-to-run $NUM_LAYER_TO_RUN \
        --piecewise-width $PIECEWISE_WIDTH \
        --profile-output "$PROFILE_OUTPUT" \
        --model-output "$FITTED_MODEL"
fi

# ========================= Step 2: Validate =========================
if [ "$SKIP_VALIDATE" = "1" ]; then
    echo ""
    echo ">>> Step 2: Skipped (SKIP_VALIDATE=1)"
else
    echo ""
    echo ">>> Step 2: Validate against trace ($NUM_REQUESTS requests)..."
    CUDA_VISIBLE_DEVICES=$GPU VLLM_USE_V1=0 \
    python benchmarks/latency_model/validate_trace.py \
        --model "$MODEL" \
        --fitted-model "$FITTED_MODEL" \
        --trace-csv "$TRACE_CSV" \
        --num-requests $NUM_REQUESTS \
        -tp $TP \
        --enforce-eager \
        --load-format dummy \
        --max-model-len $MAX_MODEL_LEN \
        --max-num-batched-tokens $MAX_BATCHED_TOKENS \
        --max-num-seqs $MAX_NUM_SEQS \
        --output "$VALIDATION_OUTPUT"
fi

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "  Profile data:  $PROFILE_OUTPUT"
echo "  Fitted model:  $FITTED_MODEL"
echo "  Validation:    $VALIDATION_OUTPUT"
echo "============================================================"
