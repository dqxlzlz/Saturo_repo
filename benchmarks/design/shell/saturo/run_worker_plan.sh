#!/usr/bin/env bash
set -euo pipefail

# Build a Saturo P/D/T worker plan from the resource-allocation solver.
#
# Usage examples:
#
#   bash benchmarks/design/shell/saturo/run_worker_plan.sh
#
#   TARGET_RPS=2.0 NUM_REQ=600 \
#   GPU_IDS=0,1,2,3,4,5,6,7 \
#   CONFIGS=tp1_pp1,tp2_pp1,tp4_pp1 \
#   bash benchmarks/design/shell/saturo/run_worker_plan.sh
#
#   CLUSTER_LAYOUT="local:0,1,2,3;l20d_1:0,1,2,3" \
#   WORKER_BACKEND=ray \
#   bash benchmarks/design/shell/saturo/run_worker_plan.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${PROJECT_ROOT}"

# ================= Saturo config =================
# MODEL_PATH="${MODEL_PATH:-/path/to/model}"
MODEL_PATH="/root/models/Qwen/Qwen2.5-72B-Instruct"


# If RESOURCE_PROFILE exists, Saturo reads it directly.
# If it does not exist, Saturo can collect real profiles and then generate
# allocation_profiles.json before solving. Pass --rt-workload-csv for that.
RESOURCE_PROFILE="${RESOURCE_PROFILE:-saturo/results/resource_allocation/generated/full_chain_smoke/allocation_profiles.json}"

# Candidate instance-level parallel configs. Names must be tpN_ppM.
# The default matches the bundled full_chain_smoke profile. Override this for
# real experiments, for example CONFIGS=tp1_pp1,tp2_pp1,tp4_pp1.
CONFIGS="${CONFIGS:-tp1_pp1}"

# Target RT arrival rate for this planning run. The default is intentionally
# tiny so the smoke profile is feasible. Override it for real runs.
TARGET_RPS="${TARGET_RPS:-0.1}"
NUM_REQ="${NUM_REQ:-600}"

# Single-node GPU list. Ignored when CLUSTER_LAYOUT is non-empty.
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"

# Multi-node GPU layout, e.g.:
#   local:0,1,2,3;l20d_1:0,1,2,3
# For Ray, using Ray node IPs is usually safer than aliases:
#   10.0.0.1:0,1,2,3;10.0.0.2:0,1,2,3
CLUSTER_LAYOUT="${CLUSTER_LAYOUT:-}"

# process: local process / SSH worker launcher
# ray: Ray actor launcher
WORKER_BACKEND="${WORKER_BACKEND:-process}"
RAY_ADDRESS="${RAY_ADDRESS:-auto}"

OUTPUT_DIR="${OUTPUT_DIR:-saturo/results/worker_plan_rps_${TARGET_RPS}}"

# SLO knobs used both by resource planning and the final plan metadata.
RT_TTFT_SLO_MS="${RT_TTFT_SLO_MS:-2000}"
RT_TBT_SLO_MS="${RT_TBT_SLO_MS:-100}"

# ================= Run =================
cmd=(
  python saturo/run_saturo_global_scheduler.py
  --dry-run-worker-plan
  --model "${MODEL_PATH}"
  --resource-profile "${RESOURCE_PROFILE}"
  --resource-config-filter "${CONFIGS}"
  --gpu-ids "${GPU_IDS}"
  --cluster-layout "${CLUSTER_LAYOUT}"
  --worker-backend "${WORKER_BACKEND}"
  --ray-address "${RAY_ADDRESS}"
  --num-req "${NUM_REQ}"
  --target-rps "${TARGET_RPS}"
  --rt-ttft-slo-ms "${RT_TTFT_SLO_MS}"
  --rt-tbt-slo-ms "${RT_TBT_SLO_MS}"
  --output-dir "${OUTPUT_DIR}"
)

echo "[Saturo] project root: ${PROJECT_ROOT}"
echo "[Saturo] NUM_REQ=${NUM_REQ}"
echo "[Saturo] TARGET_RPS=${TARGET_RPS}"
echo "[Saturo] GPU_IDS=${GPU_IDS}"
echo "[Saturo] CLUSTER_LAYOUT=${CLUSTER_LAYOUT:-<local>}"
echo "[Saturo] CONFIGS=${CONFIGS}"
echo "[Saturo] RESOURCE_PROFILE=${RESOURCE_PROFILE}"
echo "[Saturo] OUTPUT_DIR=${OUTPUT_DIR}"
printf '[Saturo] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"

echo "[Saturo] worker plan summary: ${OUTPUT_DIR}/resource_worker_plan.json"
