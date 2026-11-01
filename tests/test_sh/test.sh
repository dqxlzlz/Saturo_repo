#!/bin/bash

PREFILL_TP_LIST=(2 1)
PREFILL_PP_LIST=(1 1)

DECODE_TP_LIST=(1 4)
DECODE_PP_LIST=(2 1)

# 获取 Prefill 实例的数量，用于计算 Decode 的起始索引
NUM_PREFILL=${#PREFILL_TP_LIST[@]}

# =============================================================================
# 2. 预先构建策略 JSON 片段
# =============================================================================

# --- 生成 Prefill 的策略 Map (供 Decode 端 recv_target_worker_strategy 使用) ---
# 索引从 0 开始: "0":{...}, "1":{...}
PREFILL_STRATEGY_MAP=""
for j in "${!PREFILL_TP_LIST[@]}"; do
    tp=${PREFILL_TP_LIST[$j]}
    pp=${PREFILL_PP_LIST[$j]}
    # 构建 JSON 子项
    ITEM="\"$j\":{\"tp_size\":$tp,\"pp_size\":$pp}"
    # 拼接并处理逗号
    if [ -z "$PREFILL_STRATEGY_MAP" ]; then
        PREFILL_STRATEGY_MAP="$ITEM"
    else
        PREFILL_STRATEGY_MAP="$PREFILL_STRATEGY_MAP,$ITEM"
    fi
done

# --- 生成 Decode 的策略 Map (供 Prefill 端 send_target_worker_strategy 使用) ---
# 索引接在 Prefill 后面: "2":{...}, "3":{...}
DECODE_STRATEGY_MAP=""
for j in "${!DECODE_TP_LIST[@]}"; do
    global_idx=$((NUM_PREFILL + j)) # 编号偏移
    tp=${DECODE_TP_LIST[$j]}
    pp=${DECODE_PP_LIST[$j]}
    ITEM="\"$global_idx\":{\"tp_size\":$tp,\"pp_size\":$pp}"
    if [ -z "$DECODE_STRATEGY_MAP" ]; then
        DECODE_STRATEGY_MAP="$ITEM"
    else
        DECODE_STRATEGY_MAP="$DECODE_STRATEGY_MAP,$ITEM"
    fi
done

# =============================================================================
# 3. 组装最终的 ASYNC_CONFIG 字符串
# =============================================================================
# Prefill 端使用的 Config (发送给所有的 Decode)
PREFILL_ASYNC_CONFIG="{\"send_target_worker_strategy\":{$DECODE_STRATEGY_MAP}, \"recv_target_worker_strategy\":{}}"

# Decode 端使用的 Config (接收来自所有的 Prefill)
DECODE_ASYNC_CONFIG="{\"send_target_worker_strategy\":{}, \"recv_target_worker_strategy\":{$PREFILL_STRATEGY_MAP}}"

echo $PREFILL_ASYNC_CONFIG
echo $DECODE_ASYNC_CONFIG