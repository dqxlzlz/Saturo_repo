import json
import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

# ==========================================
# 1. 你原有的工具函数和类 (保持不变)
# ==========================================

def build_prefill_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    L = df["req_len"].to_numpy(dtype=float)
    B = df["num_req"].to_numpy(dtype=float)
    n = L * B
    sum_ctx = B * (L * (L + 1.0) / 2.0)
    X = np.stack([n, sum_ctx], axis=1)
    y = df["ms_median"].to_numpy(dtype=float)
    return X, y

def build_decode_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    L = df["req_len"].to_numpy(dtype=float)
    B = df["num_req"].to_numpy(dtype=float)
    n = B
    sum_ctx = B * L
    X = np.stack([n, sum_ctx], axis=1)
    y = df["ms_median"].to_numpy(dtype=float)
    return X, y

def eval_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
    }

@dataclass
class PiecewiseLinearModel:
    width: int = 32
    coef: Dict[int, Tuple[float, float]] = None
    fallback: Tuple[float, float] = (0.0, 0.0)

    def segment_id(self, n: float) -> int:
        return int(math.floor(n / self.width))

    def fit(self, tokens: np.ndarray, ms: np.ndarray, min_points_per_segment: int = 2):
        tokens = tokens.astype(float)
        ms = ms.astype(float)
        self.coef = {}

        A = np.vstack([tokens, np.ones_like(tokens)]).T
        a_g, b_g = np.linalg.lstsq(A, ms, rcond=None)[0]
        self.fallback = (float(a_g), float(b_g))

        seg_ids = np.array([self.segment_id(t) for t in tokens], dtype=int)
        for sid in np.unique(seg_ids):
            idx = np.where(seg_ids == sid)[0]
            if len(idx) < min_points_per_segment:
                continue
            t_seg = tokens[idx]
            y_seg = ms[idx]
            A_seg = np.vstack([t_seg, np.ones_like(t_seg)]).T
            a, b = np.linalg.lstsq(A_seg, y_seg, rcond=None)[0]
            self.coef[int(sid)] = (float(a), float(b))

    def predict_one(self, n: float) -> float:
        sid = self.segment_id(float(n))
        a, b = self.coef.get(sid, self.fallback)
        res = a * float(n) + b
        return max(0.0, res)  # 避免预测出负数

    def predict(self, tokens: np.ndarray) -> np.ndarray:
        return np.array([self.predict_one(t) for t in tokens], dtype=float)

# ==========================================
# 2. 新增：封装好的预测器类
# ==========================================

class ModelLatencyPredictor:
    def __init__(self, json_path: str, num_layers: int = 32):
        """
        :param json_path: bench_results.json 的路径
        :param num_layers: 模型层数 (例如 Llama2-7b 是 32, Llama2-13b 是 40)
        """
        self.num_layers = num_layers
        self.prefill_model = LinearRegression()
        self.linear_model = PiecewiseLinearModel(width=128)
        
        # 加载数据并训练
        self._train_models(json_path)

    def _train_models(self, json_path: str):
        print(f"Loading benchmark data from {json_path}...")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 1. Train Prefill Attention Model
        prefill_df = pd.DataFrame(data["prefill_attn_ms"])
        Xp, yp = build_prefill_features(prefill_df)
        self.prefill_model.fit(Xp, yp)
        print(" -> Prefill Attention model trained.")

        # 2. Train Linear/FFN Model
        linear_df = pd.DataFrame(data["linear_ms"])
        tokens = linear_df["num_token"].to_numpy(dtype=float)
        ylin = linear_df["ms_mean"].to_numpy(dtype=float)
        self.linear_model.fit(tokens, ylin)
        print(" -> Linear Piecewise model trained.")

    def predict_prefill_latency_ms(self, context_len: int, batch_size: int = 1) -> float:
        """
        预测整个模型的 Prefill 耗时 (ms)
        Logic: Layers * (Attn_Layer_Time + Linear_Layer_Time)
        """
        # A. Predict Single Layer Attention
        L = float(context_len)
        B = float(batch_size)
        n = L * B
        sum_ctx = B * (L * (L + 1.0) / 2.0)
        
        # sklearn predict expects 2D array
        attn_layer_ms = self.prefill_model.predict(np.array([[n, sum_ctx]], dtype=float))[0]
        
        # B. Predict Single Layer Linear (FFN + Projections)
        # Linear layer input is total tokens (B * L)
        linear_layer_ms = self.linear_model.predict_one(n)

        # C. Aggregate
        # 注意：这里假设 JSON 里的数据是单层的耗时
        total_ms = self.num_layers * (attn_layer_ms + linear_layer_ms)
        return max(0.0, total_ms)

# ==========================================
# 3. 主逻辑：读取 CSV 并对比
# ==========================================

def run_trace_analysis(csv_path: str, bench_json_path: str, num_layers: int = 32):
    # 1. 初始化预测器
    predictor = ModelLatencyPredictor(json_path=bench_json_path, num_layers=num_layers)

    # 2. 读取 CSV
    print(f"\nReading Trace CSV from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        df = df.head(300).copy()
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 检查必要的列
    required_cols = ["ContextTokens", "forward_time"]
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV must contain columns: {required_cols}")
        return

    print(f"Loaded {len(df)} records.")

    # 3. 遍历并预测
    predictions_ms = []
    actuals_ms = []
    
    print("Running predictions...")
    for idx, row in df.iterrows():
        ctx_len = int(row["ContextTokens"])
        
        # CSV 中的 forward_time 通常是秒，这里转换为毫秒以便对比
        # 如果你的 forward_time 已经是毫秒，请去掉 * 1000
        actual_time_s = float(row["forward_time"])
        actual_time_ms = actual_time_s * 1000.0 
        
        # 使用预测器 (默认 Batch=1)
        pred_ms = predictor.predict_prefill_latency_ms(context_len=ctx_len, batch_size=1)
        
        predictions_ms.append(pred_ms)
        actuals_ms.append(actual_time_ms)

    # 4. 将结果写回 DataFrame 以便分析
    df["predicted_ms"] = predictions_ms
    df["actual_ms"] = actuals_ms
    df["error_ms"] = df["predicted_ms"] - df["actual_ms"]
    df["abs_error_ms"] = df["error_ms"].abs()
    # 避免除以0
    df["ape"] = df["abs_error_ms"] / (df["actual_ms"] + 1e-6) 

    # 5. 计算整体指标
    y_true = np.array(actuals_ms)
    y_pred = np.array(predictions_ms)
    
    metrics = eval_regression(y_true, y_pred)
    
    print("\n" + "="*40)
    print(f"Trace Prediction Results (Num Layers = {num_layers})")
    print("="*40)
    print(f"R2 Score: {metrics['R2']:.4f}")
    print(f"MAE (ms): {metrics['MAE']:.4f}")
    print(f"MAPE (%): {metrics['MAPE'] * 100:.2f}%")
    
    # 打印部分样例对比
    print("\n--- Sample Predictions ---")
    print(df[["ContextTokens", "actual_ms", "predicted_ms", "ape"]].head(10).to_string())

    # (可选) 保存结果
    output_csv = "trace_prediction_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nDetailed results saved to {output_csv}")

if __name__ == "__main__":
    # 配置区
    BENCH_RESULT_JSON = "bench_results.json"  # 你的基准测试结果文件
    TRACE_CSV = "/data/liaojianxiong/hybrid_service/hybrid_serve_proj/data/outcome/azure_conv_forward_time_tp_1(v1).csv"              # 你的 CSV 文件路径
    
    # 重要：设置模型的层数
    # Llama-2-7b = 32, Llama-2-13b = 40, Llama-2-70b = 80
    # Qwen-72b = 80, Mixtral 8x7b = 32
    MODEL_LAYERS = 48                         
    
    run_trace_analysis(TRACE_CSV, BENCH_RESULT_JSON, num_layers=MODEL_LAYERS)
    