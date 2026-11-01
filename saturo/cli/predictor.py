#!/usr/bin/env python3
"""CLI for training and evaluating remaining-length predictors.

Usage
-----
    # 训练 + 立即评估（正确 train/test 分离，推荐）
    python -m saturo.cli.predictor run \\
        --dataset wildchat_0000 \\
        --parquet data/wildchat/0000.parquet \\
        --model-type tfidf_qrf --n-train 3000 --n-test 300

    # 只训练，保存模型
    python -m saturo.cli.predictor train \\
        --dataset wildchat_0000 \\
        --parquet data/wildchat/0000.parquet

    # 评估已有模型的 Coverage
    python -m saturo.cli.predictor eval \\
        --dataset wildchat_0000 \\
        --parquet data/wildchat/0000.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Table helper
# ---------------------------------------------------------------------------

def _tbl(rows: list[list[str]], headers: list[str]) -> None:
    ws = [max(len(h), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    sep = "+-" + "-+-".join("-" * w for w in ws) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in ws) + " |"
    print(sep); print(fmt.format(*headers)); print(sep)
    for r in rows:
        print(fmt.format(*[str(v) for v in r]))
    print(sep)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    from saturo.predictors.registry import artifact_path, train

    path = artifact_path(args.dataset, args.model_type)
    if path.exists() and not args.force:
        print(f"模型已存在: {path}")
        print("使用 --force 重新训练")
        return

    print(f"训练 {args.model_type} on {args.dataset}")
    predictor = train(
        parquet_path=args.parquet,
        model_type=args.model_type,
        max_samples=args.max_samples,
        max_steps_per_request=args.max_steps,
        quantiles=tuple(float(q) for q in args.quantiles.split(",")),
        hash_bins=args.hash_bins,
        tfidf_max_features=args.tfidf_features,
        n_estimators=args.trees,
        min_samples_leaf=args.min_leaf,
        seed=args.seed,
        verbose=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    predictor.save(path)
    print(f"\n已保存: {path}")


def cmd_run(args: argparse.Namespace) -> None:
    """正确 train/test 分离：先加载全部数据、切分，再训练和评估。"""
    from pathlib import Path

    from saturo.predictors.registry import (
        _build_tfidf_features,
        _load_wildchat_records,
        artifact_path,
        train,
    )
    from saturo.predictors.request_length_predictor import TFIDFQRFRemainingLengthPredictor

    n_train = args.n_train
    n_test = args.n_test
    total_needed = n_train + n_test

    print(f"=== TF-IDF + QRF 剩余长度预测实验 ===")
    print(f"模型: {args.model_type}  训练: {n_train} 条  验证: {n_test} 条")
    print(f"\n[1/4] 加载数据 (共需 {total_needed} 条)...")
    all_records = _load_wildchat_records(Path(args.parquet), max_samples=total_needed)
    print(f"  实际加载: {len(all_records)} 条")
    if len(all_records) < total_needed:
        n_train = int(len(all_records) * 0.9)
        n_test = len(all_records) - n_train
        print(f"  数据不足，调整为: 训练={n_train}, 验证={n_test}")

    # 按请求级别切分，确保无泄漏
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(all_records))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:n_train + n_test]
    train_records = [all_records[i] for i in train_idx]
    test_records = [all_records[i] for i in test_idx]

    ol_train = [len(o) for _, o in train_records]
    ol_test = [len(o) for _, o in test_records]
    print(f"  训练集: {len(train_records)} 条  output_len mean={np.mean(ol_train):.0f}")
    print(f"  验证集: {len(test_records)} 条  output_len mean={np.mean(ol_test):.0f}")

    # ── 训练 ──
    print(f"\n[2/4] 训练 {args.model_type}...")
    predictor = train(
        parquet_path=args.parquet,
        model_type=args.model_type,
        max_samples=n_train,
        max_steps_per_request=args.max_steps,
        quantiles=tuple(float(q) for q in args.quantiles.split(",")),
        hash_bins=args.hash_bins,
        tfidf_max_features=args.tfidf_features,
        n_estimators=args.trees,
        min_samples_leaf=args.min_leaf,
        seed=args.seed,
        verbose=True,
        records=train_records,
    )
    path = artifact_path(args.dataset, args.model_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictor.save(path)
    print(f"  模型已保存: {path}")

    # ── 在验证集上评估 ──
    print(f"\n[3/4] 在验证集 ({len(test_records)} 条) 上构建特征...")
    is_tfidf = isinstance(predictor, TFIDFQRFRemainingLengthPredictor)
    if is_tfidf:
        from scipy.sparse import hstack as sp_hstack, csr_matrix
        aug_strs, len_feats, y_te, ols_te, gls_te = _build_tfidf_features(
            test_records, max_steps_per_request=args.max_steps, seed=args.seed + 1
        )
        X_tfidf = predictor.vectorizer.transform(aug_strs)  # keep sparse
        X_te = sp_hstack([X_tfidf, csr_matrix(len_feats)], format="csr")
    else:
        from saturo.predictors.registry import _build_features
        from saturo.predictors.request_length_predictor import FEATURE_MODE_TOKEN_IDS
        X_te, y_te, ols_te, gls_te = _build_features(
            test_records, FEATURE_MODE_TOKEN_IDS, args.hash_bins, args.max_steps, seed=args.seed + 1
        )
    print(f"  验证样本数 (decode steps): {len(y_te):,}")

    # ── 评估 ──
    print(f"\n[4/4] 评估结果")
    quantiles = [float(q) for q in args.quantiles.split(",")]
    sf_te = gls_te.astype(np.float32) / np.maximum(ols_te.astype(np.float32), 1.0)

    # 整体 Coverage
    print("\n" + "=" * 60)
    print("  整体 Coverage (预测 >= 实际剩余 的比例，目标≈分位数值)")
    print("=" * 60)
    rows = []
    for eta in quantiles:
        pred = predictor.predict_quantile(X_te, eta)
        cov = float(np.mean(pred >= y_te))
        mae = float(np.mean(np.abs(pred - y_te)))
        rows.append([f"P{int(eta * 100)}", f"{eta:.2f}", f"{cov:.3f}", f"{np.mean(pred):.1f}", f"{np.mean(y_te):.1f}", f"{mae:.1f}"])
    _tbl(rows, ["Quantile", "目标", "实际Cov", "预测均值", "实际均值", "MAE"])

    # P90 分档
    if any(abs(q - 0.90) < 1e-6 for q in quantiles):
        p90 = predictor.predict_quantile(X_te, 0.90)
        print("\n" + "=" * 60)
        print("  P90 Coverage 按 output_len 分档")
        print("=" * 60)
        obins = [0, 50, 100, 200, 500, 800, 10000]
        olabels = ["<50", "50-99", "100-199", "200-499", "500-799", "800+"]
        rows = []
        for i, lbl in enumerate(olabels):
            m = (ols_te >= obins[i]) & (ols_te < obins[i + 1])
            if m.sum() < 5:
                continue
            cov = float(np.mean(p90[m] >= y_te[m]))
            rows.append([lbl, f"{m.sum():,}", f"{cov:.3f}"])
        _tbl(rows, ["output_len", "样本数", "P90 Cov"])

        # 按生成进度分层
        print("\n" + "=" * 60)
        print("  P90 Coverage 按生成进度 (generated / total_output)")
        print("=" * 60)
        sf_bins = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.01]
        sf_labels = ["0-10%", "10-25%", "25-50%", "50-75%", "75-90%", "90-100%"]
        rows = []
        for j, lbl in enumerate(sf_labels):
            mm = (sf_te >= sf_bins[j]) & (sf_te < sf_bins[j + 1])
            if mm.sum() < 5:
                continue
            cov = float(np.mean(p90[mm] >= y_te[mm]))
            med_pred = float(np.median(p90[mm]))
            med_actual = float(np.median(y_te[mm]))
            rows.append([lbl, f"{mm.sum():,}", f"{cov:.3f}", f"{med_pred:.0f}", f"{med_actual:.0f}"])
        _tbl(rows, ["进度", "样本数", "P90 Cov", "预测中位", "实际中位"])

    print("\n完成!")


def cmd_eval(args: argparse.Namespace) -> None:
    from saturo.predictors.registry import (
        _build_features,
        _build_tfidf_features,
        _load_wildchat_records,
        artifact_path,
    )
    from saturo.predictors.request_length_predictor import (
        FEATURE_MODE_TOKEN_IDS,
        TFIDFQRFRemainingLengthPredictor,
        load_length_predictor,
    )

    # cmd_run 直接传入已训练的实例，eval 独立调用时从 artifact 加载
    predictor = getattr(args, "_predictor_instance", None)
    if predictor is None:
        path = artifact_path(args.dataset, args.model_type)
        if not path.exists():
            print(f"模型不存在: {path}")
            print("请先运行: python -m saturo.cli.predictor train ...")
            sys.exit(1)
        print(f"加载模型: {path}")
        predictor = load_length_predictor(path)

    print(f"加载数据: {args.parquet}  max_samples={args.max_samples}")
    records = _load_wildchat_records(Path(args.parquet), max_samples=args.max_samples)
    print(f"  记录: {len(records)} 条  output_len mean={np.mean([len(o) for _,o in records]):.0f}")

    # 根据模型类型选择特征构建方式
    is_tfidf = isinstance(predictor, TFIDFQRFRemainingLengthPredictor)
    if is_tfidf:
        from scipy.sparse import hstack as sp_hstack, csr_matrix
        aug_strs, len_feats, y, ols, gls = _build_tfidf_features(
            records, max_steps_per_request=args.max_steps, seed=args.seed
        )
        X_tfidf = predictor.vectorizer.transform(aug_strs)  # keep sparse
        X = sp_hstack([X_tfidf, csr_matrix(len_feats)], format="csr")
    else:
        X, y, ols, gls = _build_features(
            records,
            feature_mode=FEATURE_MODE_TOKEN_IDS,
            hash_bins=args.hash_bins,
            max_steps_per_request=args.max_steps,
            seed=args.seed,
        )

    # 按请求划分 train/test（避免数据泄漏）—— test 请求的所有 decode step 都不出现在训练集
    n_records = len(records)
    rng_split = np.random.default_rng(args.seed)
    test_req_ids = set(rng_split.choice(n_records, size=n_records // 4, replace=False).tolist())

    # 重新构建，计算每行属于哪个请求
    steps_per_req = []
    for p_ids, o_ids in records:
        O = len(o_ids)
        cnt = min(O, args.max_steps) if args.max_steps else O
        steps_per_req.append(cnt)
    req_ids_per_row = np.repeat(np.arange(n_records), steps_per_req)
    te_mask = np.array([rid in test_req_ids for rid in req_ids_per_row])

    X_te = X[te_mask]; y_te = y[te_mask]
    ols_te = ols[te_mask]; gls_te = gls[te_mask]
    sf_te = gls_te.astype(np.float32) / np.maximum(ols_te, 1.0)

    quantiles = [float(q) for q in args.quantiles.split(",")]

    # ── 整体 Coverage ──
    print("\n" + "=" * 60)
    print("  整体 Coverage")
    print("=" * 60)
    rows = []
    for eta in quantiles:
        pred = predictor.predict_quantile(X_te, eta)
        cov = float(np.mean(pred >= y_te))
        rows.append([f"P{int(eta * 100)}", f"{cov:.3f}", f"{np.mean(pred):.0f}"])
    _tbl(rows, ["Quantile", "Coverage", "预测均值"])

    # ── P90 Coverage 按 output_len ──
    if any(abs(q - 0.90) < 1e-6 for q in quantiles):
        p90 = predictor.predict_quantile(X_te, 0.90)
        print("\n" + "=" * 60)
        print("  P90 Coverage 按 output_len 分档")
        print("=" * 60)
        obins = [0, 100, 200, 500, 800, 10000]
        olabels = ["<100", "100-199", "200-499", "500-799", "800+"]
        rows = []
        for i, lbl in enumerate(olabels):
            m = (ols_te >= obins[i]) & (ols_te < obins[i + 1])
            if m.sum() < 5:
                continue
            cov = float(np.mean(p90[m] >= y_te[m]))
            rows.append([lbl, f"{m.sum():,}", f"{cov:.3f}"])
        _tbl(rows, ["output_len", "样本数", "P90 Cov"])

        # ── 500+ 按生成进度 ──
        m500 = ols_te >= 500
        if m500.sum() >= 20:
            print("\n" + "=" * 60)
            print("  500+ 请求 P90 Coverage 按生成进度")
            print("=" * 60)
            sf_bins = [0.0, 0.25, 0.50, 0.75, 1.01]
            sf_labels = ["0-25%", "25-50%", "50-75%", "75-100%"]
            rows = []
            for j, lbl in enumerate(sf_labels):
                mm = m500 & (sf_te >= sf_bins[j]) & (sf_te < sf_bins[j + 1])
                if mm.sum() < 5:
                    continue
                cov = float(np.mean(p90[mm] >= y_te[mm]))
                rows.append([lbl, f"{mm.sum():,}", f"{cov:.3f}"])
            _tbl(rows, ["生成进度", "样本数", "P90 Cov"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Saturo 剩余长度预测器 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ── shared args ──
    def add_shared(p: argparse.ArgumentParser) -> None:
        p.add_argument("--dataset", required=True, help="数据集标识名，如 wildchat_0000")
        p.add_argument("--parquet", required=True, help="WildChat parquet 文件路径")
        p.add_argument("--model-type", default="tfidf_qrf",
                       choices=("tfidf_qrf", "qrf", "gbdt"),
                       help="tfidf_qrf: JITServe风格 TF-IDF+QRF（默认）; qrf: hash-bin QRF; gbdt: 分位数GBDT")
        p.add_argument("--hash-bins", type=int, default=128, help="hash-bin 模式下有效")
        p.add_argument("--tfidf-features", type=int, default=50000, help="TF-IDF 词表大小")
        p.add_argument("--max-samples", type=int, default=1000, help="最大请求数")
        p.add_argument("--max-steps", type=int, default=200, help="每条请求最多采样步数")
        p.add_argument("--quantiles", default="0.80,0.90,0.95", help="分位数，逗号分隔")
        p.add_argument("--seed", type=int, default=42)

    def add_train_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--trees", type=int, default=64, help="随机森林树数量")
        p.add_argument("--min-leaf", type=int, default=20, help="叶节点最小样本数")

    # ── run: 总是重新训练 + 立即评估 ──
    p_run = sub.add_parser("run", help="训练（总是重新）+ 立即评估，一键执行")
    add_shared(p_run)
    add_train_args(p_run)
    p_run.add_argument("--n-train", type=int, default=3000, help="训练集请求数")
    p_run.add_argument("--n-test", type=int, default=300, help="验证集请求数")

    # ── train: 只训练保存 ──
    p_train = sub.add_parser("train", help="训练并保存模型（已存在时跳过，--force 覆盖）")
    add_shared(p_train)
    add_train_args(p_train)
    p_train.add_argument("--force", action="store_true", help="强制覆盖已有模型")

    # ── eval: 加载已有模型评估 ──
    p_eval = sub.add_parser("eval", help="加载已保存模型，评估 Coverage")
    add_shared(p_eval)

    p_history = sub.add_parser(
        "train-history", help="Train a length-only QRF from Redis/CSV history")
    p_history.add_argument("--trace", required=True,
                           help="CSV path or redis-history:<key>")
    p_history.add_argument("--output", type=Path, required=True)
    p_history.add_argument("--window-s", type=float, default=1800)
    p_history.add_argument("--model", default="")
    p_history.add_argument("--max-steps", type=int, default=16)
    p_history.add_argument("--trees", type=int, default=32)
    p_history.add_argument("--min-requests", type=int, default=32)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    elif args.cmd == "train-history":
        from saturo.predictors.history_training import train_history_qrf
        if args.output.exists():
            parser.error("--output already exists; choose a new artifact path")
        train_history_qrf(
            args.trace, args.output, window_s=args.window_s, model=args.model,
            max_steps=args.max_steps, n_estimators=args.trees,
            min_requests=args.min_requests)
        print(f"Saved history QRF to {args.output}")


if __name__ == "__main__":
    main()
