"""
集成学习脚本：训练多个 GBDT 模型（LightGBM + XGBoost + CatBoost），
使用 AverageEnsemble 融合预测，对比各模型与集成效果。

用法:
    .venv\\Scripts\\python.exe scripts\\ensemble_train.py                    # 全部三个模型 + 平均集成
    .venv\\Scripts\\python.exe scripts\\ensemble_train.py --models lgbm,xgb  # 指定模型
    .venv\\Scripts\\python.exe scripts\\ensemble_train.py --data qlib_bin
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord
from qlib.model.ens.ensemble import AverageEnsemble

# ── 数据源配置（与 train.py 一致）──────────────────────────────────────
DATA_CONFIGS = {
    "qlib_bin": {
        "provider_uri": "C:/codes/qlib/qlib_bin",
        "market": "all",
        "train_start": "2020-01-01",
        "train_end": "2024-12-31",
        "valid_start": "2025-01-01",
        "valid_end": "2025-12-31",
        "test_start": "2026-01-01",
        "test_end": "2026-05-08",
    },
}

REGION = REG_CN
BENCHMARK = "SH000300"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
DOCS_DIR = Path(__file__).parent.parent / "docs"

HANDLER_MAP = {
    "alpha158": {"class": "Alpha158", "module_path": "qlib.contrib.data.handler"},
}


# ── 模型配置 ──────────────────────────────────────────────────────────
def get_model_config(model_name):
    """返回模型配置字典。"""
    configs = {
        "lgbm": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": 0.8879,
                "learning_rate": 0.2,
                "subsample": 0.8789,
                "lambda_l1": 205.6999,
                "lambda_l2": 580.9768,
                "max_depth": 8,
                "num_leaves": 210,
                "num_threads": 20,
            },
        },
        "xgb": {
            "class": "XGBModel",
            "module_path": "qlib.contrib.model.xgboost",
            "kwargs": {
                "n_estimators": 1000,
                "max_depth": 8,
                "learning_rate": 0.2,
                "subsample": 0.8789,
                "colsample_bytree": 0.8879,
                "reg_alpha": 205.6999,
                "reg_lambda": 580.9768,
                "nthread": 20,
            },
        },
        "catboost": {
            "class": "CatBoostModel",
            "module_path": "qlib.contrib.model.catboost_model",
            "kwargs": {
                "loss": "RMSE",
                "depth": 8,
                "learning_rate": 0.2,
                "iterations": 1000,
                "task_type": "CPU",
                "l2_leaf_reg": 580.9768,
                "thread_count": 20,
            },
        },
    }
    if model_name not in configs:
        raise ValueError(f"不支持的模型: {model_name}，可选: {list(configs.keys())}")
    return configs[model_name]


def get_dataset_config(data_cfg, handler="alpha158"):
    """构建 Dataset 配置。"""
    handler_def = HANDLER_MAP[handler]
    return {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                **handler_def,
                "kwargs": {
                    "start_time": data_cfg["train_start"],
                    "end_time": data_cfg["test_end"],
                    "fit_start_time": data_cfg["train_start"],
                    "fit_end_time": data_cfg["train_end"],
                    "instruments": data_cfg.get("market", "all"),
                },
            },
            "segments": {
                "train": [data_cfg["train_start"], data_cfg["train_end"]],
                "valid": [data_cfg["valid_start"], data_cfg["valid_end"]],
                "test": [data_cfg["test_start"], data_cfg["test_end"]],
            },
        },
    }


def compute_ic(pred, label_df):
    """计算 IC 和 Rank IC。"""
    # pred: pd.Series with MultiIndex (datetime, instrument)
    # label_df: DataFrame with 'label' column, same MultiIndex
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
    pred = pred.rename("score")

    merged = label_df.join(pred, how="inner")
    if merged.empty:
        return {"IC": 0, "Rank_IC": 0, "ICIR": 0, "Rank_ICIR": 0}

    # 按日期分组计算 IC
    def daily_ic(g):
        if len(g) < 5:
            return np.nan
        return g["label"].corr(g["score"], method="pearson")

    def daily_rank_ic(g):
        if len(g) < 5:
            return np.nan
        return g["label"].corr(g["score"], method="spearman")

    ic_series = merged.groupby(level="datetime").apply(daily_ic)
    rank_ic_series = merged.groupby(level="datetime").apply(daily_rank_ic)

    ic_series = ic_series.dropna()
    rank_ic_series = rank_ic_series.dropna()

    return {
        "IC": float(ic_series.mean()) if len(ic_series) > 0 else 0,
        "Rank_IC": float(rank_ic_series.mean()) if len(rank_ic_series) > 0 else 0,
        "ICIR": float(ic_series.mean() / ic_series.std()) if len(ic_series) > 1 and ic_series.std() > 0 else 0,
        "Rank_ICIR": float(rank_ic_series.mean() / rank_ic_series.std()) if len(rank_ic_series) > 1 and rank_ic_series.std() > 0 else 0,
    }


def train_single_model(model_name, model_config, dataset, data_cfg):
    """训练单个模型，返回 (model, predictions, recorder)。"""
    print(f"\n{'─' * 50}")
    print(f"训练模型: {model_name}")
    print(f"{'─' * 50}")

    experiment_name = f"ensemble_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with R.start(experiment_name=experiment_name):
        R.log_params(model=model_name, data_source="qlib_bin", ensemble=True,
                     **flatten_dict({"model": model_config}))

        model = init_instance_by_config(model_config)
        model.fit(dataset)
        R.save_objects(**{"params.pkl": model})

        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        try:
            sar = SigAnaRecord(recorder)
            sar.generate()
        except Exception as e:
            print(f"  [警告] SigAnaRecord 跳过: {e}")

    # 获取预测
    pred = model.predict(dataset)
    print(f"  {model_name} 训练完成，预测 {len(pred) if isinstance(pred, pd.Series) else 'N/A'} 条")

    return model, pred, recorder


def main():
    parser = argparse.ArgumentParser(description="Qlib 集成学习脚本 (多模型融合)")
    parser.add_argument("--data", default="qlib_bin", choices=list(DATA_CONFIGS.keys()))
    parser.add_argument("--models", default="lgbm,xgb,catboost",
                        help="模型列表，逗号分隔 (lgbm,xgb,catboost)")
    parser.add_argument("--handler", default="alpha158", choices=list(HANDLER_MAP.keys()))
    parser.add_argument("--strategy", default="average",
                        choices=["average", "rank_average"],
                        help="集成策略: average=标准化平均, rank_average=排名平均")
    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(",")]
    data_cfg = DATA_CONFIGS[args.data]

    print("=" * 60)
    print(f"Qlib 集成学习")
    print(f"模型: {model_names}")
    print(f"特征: {args.handler}")
    print(f"数据源: {args.data}")
    print(f"集成策略: {args.strategy}")
    print(f"训练: {data_cfg['train_start']} ~ {data_cfg['train_end']}")
    print(f"验证: {data_cfg['valid_start']} ~ {data_cfg['valid_end']}")
    print(f"测试: {data_cfg['test_start']} ~ {data_cfg['test_end']}")
    print("=" * 60)

    # 初始化 qlib（Windows multiprocessing 会触发 concurrent send_bytes bug，
    # 设置 joblib_backend="threading" 避免）
    qlib.init(provider_uri=data_cfg["provider_uri"], region=REGION, joblib_backend="threading")

    # 构建数据集（所有模型共享同一数据集）
    dataset_config = get_dataset_config(data_cfg, handler=args.handler)
    dataset = init_instance_by_config(dataset_config)

    # 打印数据集概要
    train_df = dataset.prepare("train")
    print(f"\n训练集: {train_df.shape}")

    # 获取标签（用于 IC 计算）
    label_df = dataset.prepare("test")
    if isinstance(label_df, pd.DataFrame) and len(label_df.columns) > 0:
        label_col = label_df.columns[-1]  # 通常是最后一列
        label_df = label_df[[label_col]].rename(columns={label_col: "label"})
    else:
        label_df = None

    # ── 训练各模型 ──────────────────────────────────────────────────
    models = {}
    predictions = {}
    recorders = {}

    for name in model_names:
        model_config = get_model_config(name)
        model, pred, recorder = train_single_model(name, model_config, dataset, data_cfg)
        models[name] = model
        predictions[name] = pred
        recorders[name] = recorder

    # ── 集成预测 ────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"集成阶段: {args.strategy}")
    print(f"{'=' * 60}")

    if args.strategy == "average":
        # 使用 qlib AverageEnsemble（标准化后取平均）
        ensemble_dict = {k: v.to_frame("score") if isinstance(v, pd.Series) else v
                         for k, v in predictions.items()}
        avg_ens = AverageEnsemble()
        ensemble_pred = avg_ens(ensemble_dict)
    elif args.strategy == "rank_average":
        # 简单排名平均（各模型预测排名后取平均）
        rank_dfs = []
        for name, pred in predictions.items():
            if isinstance(pred, pd.Series):
                rank_df = pred.groupby(level="datetime").rank(pct=True)
                rank_dfs.append(rank_df)
        if rank_dfs:
            ensemble_pred = pd.concat(rank_dfs, axis=1).mean(axis=1)

    # ── 计算 IC 指标 ────────────────────────────────────────────────
    print("\nIC 指标对比:")
    print(f"{'─' * 60}")

    results = {}
    for name, pred in predictions.items():
        if label_df is not None:
            ic_metrics = compute_ic(pred, label_df)
            results[name] = ic_metrics
            print(f"  {name:>10s}: IC={ic_metrics['IC']:+.4f}, Rank_IC={ic_metrics['Rank_IC']:+.4f}, "
                  f"ICIR={ic_metrics['ICIR']:+.4f}, Rank_ICIR={ic_metrics['Rank_ICIR']:+.4f}")

    # 集成结果
    if label_df is not None:
        ens_ic = compute_ic(ensemble_pred, label_df)
        results["ensemble"] = ens_ic
        print(f"  {'ensemble':>10s}: IC={ens_ic['IC']:+.4f}, Rank_IC={ens_ic['Rank_IC']:+.4f}, "
              f"ICIR={ens_ic['ICIR']:+.4f}, Rank_ICIR={ens_ic['Rank_ICIR']:+.4f}")

    # ── 保存结果 ────────────────────────────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 保存集成信息
    ensemble_info = {
        "models": model_names,
        "strategy": args.strategy,
        "handler": args.handler,
        "data_source": args.data,
        "ic_results": results,
        "recorder_ids": {k: v.id for k, v in recorders.items()},
        "timestamp": datetime.now().isoformat(),
    }
    info_path = REPORTS_DIR / "ensemble_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(ensemble_info, f, ensure_ascii=False, indent=2)
    print(f"\n集成结果已保存: {info_path}")

    # 更新 train_info.json 为集成模型（供 predict.py 使用最后一个模型）
    # 保留各模型的 recorder 以便后续使用
    last_model_name = model_names[-1]
    train_info = {
        "experiment_name": recorders[last_model_name].name,
        "model": f"ensemble_{'_'.join(model_names)}",
        "handler": args.handler,
        "data_source": args.data,
        "train_period": [data_cfg["train_start"], data_cfg["train_end"]],
        "valid_period": [data_cfg["valid_start"], data_cfg["valid_end"]],
        "test_period": [data_cfg["test_start"], data_cfg["test_end"]],
        "recorder_id": recorders[last_model_name].id,
        "ensemble": True,
        "ensemble_models": model_names,
        "timestamp": datetime.now().isoformat(),
    }
    # 备份原始 train_info
    orig_info_path = REPORTS_DIR / "train_info.json"
    if orig_info_path.exists():
        backup_path = REPORTS_DIR / "train_info_backup.json"
        if not backup_path.exists():
            import shutil
            shutil.copy2(orig_info_path, backup_path)
            print(f"原始 train_info.json 已备份到: {backup_path}")

    with open(orig_info_path, "w", encoding="utf-8") as f:
        json.dump(train_info, f, ensure_ascii=False, indent=2)

    # ── 生成对比报告 ────────────────────────────────────────────────
    report_path = DOCS_DIR / "analysis" / "ensemble_comparison.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# 集成学习对比报告",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**数据源**: {args.data}",
        f"**特征**: {args.handler}",
        f"**集成策略**: {args.strategy}",
        f"**模型组合**: {', '.join(model_names)}",
        "",
        "## IC 指标对比",
        "",
        "| 模型 | IC | Rank IC | ICIR | Rank ICIR |",
        "|------|:---:|:-------:|:----:|:---------:|",
    ]

    for name, metrics in results.items():
        marker = " **⭐**" if name == "ensemble" else ""
        lines.append(
            f"| {name}{marker} | {metrics['IC']:+.4f} | {metrics['Rank_IC']:+.4f} | "
            f"{metrics['ICIR']:+.4f} | {metrics['Rank_ICIR']:+.4f} |"
        )

    # 分析结论
    lines.extend(["", "## 分析结论", ""])

    best_single = max(
        [(k, v) for k, v in results.items() if k != "ensemble"],
        key=lambda x: abs(x[1]["Rank_IC"])
    )
    ens_metrics = results.get("ensemble", {})

    if ens_metrics.get("Rank_IC", 0) > best_single[1]["Rank_IC"]:
        lines.append(f"- **集成模型优于所有单模型**: Rank IC {ens_metrics['Rank_IC']:+.4f} > {best_single[0]} {best_single[1]['Rank_IC']:+.4f}")
    else:
        lines.append(f"- **最佳单模型**: {best_single[0]} (Rank IC={best_single[1]['Rank_IC']:+.4f}) 优于集成 (Rank IC={ens_metrics.get('Rank_IC', 0):+.4f})")

    lines.append(f"- 训练集: {train_df.shape[0]:,} 样本, {train_df.shape[1]} 特征")

    lines.extend(["", "---", "", "*本报告由量化模型自动生成，仅供参考，不构成任何投资建议。*"])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"对比报告已保存: {report_path}")

    print(f"\n{'=' * 60}")
    print("集成学习完成!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
