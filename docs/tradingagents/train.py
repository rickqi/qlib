# ============================================================
# Qlib 训练脚本 — tradingagents 数据
# ============================================================
# 用法:
#   cd D:\codes\qlib
#   C:\Users\szk220009\.qlib\venv\Scripts\python.exe docs\tradingagents\train.py
# ============================================================

import os
import sys
from pathlib import Path

import qlib
import pandas as pd
from qlib.constant import REG_CN
from qlib.data import D
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord

# ============================================================
# 配置区域
# ============================================================
PROVIDER_URI = "C:/Users/szk220009/.qlib/qlib_data/tradingagents"
MARKET = "all"
EXPERIMENT_NAME = "tradingagents_lgb"

# 数据时间范围
TRAIN_START = "2021-01-01"
TRAIN_END   = "2023-12-31"
VALID_START = "2024-01-01"
VALID_END   = "2024-12-31"
TEST_START  = "2025-01-01"
TEST_END    = "2026-04-30"  # 提前几天避免日历越界

# LightGBM 超参数
LGB_PARAMS = {
    "loss": "mse",
    "colsample_bytree": 0.8879,
    "learning_rate": 0.2,
    "subsample": 0.8789,
    "lambda_l1": 205.6999,
    "lambda_l2": 580.9768,
    "max_depth": 8,
    "num_leaves": 210,
    "num_threads": 4,
}

# 回测策略参数
TOPK = 10     # 持仓前 N 只
N_DROP = 2    # 每期换仓 N 只
ACCOUNT = 100000000  # 初始资金 1 亿

# ============================================================
# 1. 初始化 Qlib
# ============================================================
qlib.init(provider_uri=PROVIDER_URI, region=REG_CN)

# ============================================================
# 2. 数据探索
# ============================================================
calendar = D.calendar(start_time="2020-12-17", end_time="2026-05-08", freq="day")
instruments = D.instruments(MARKET)
inst_list = D.list_instruments(
    instruments=instruments,
    start_time="2023-01-01",
    end_time="2026-05-08",
    as_list=True,
)

print("=" * 60)
print("【数据概况】")
print(f"  交易日历: {calendar[0]} ~ {calendar[-1]}, 共 {len(calendar)} 天")
print(f"  股票数量: {len(inst_list)}")
print(f"  股票列表: {inst_list}")

# 查看可用特征
sample_fields = ["$open", "$close", "$high", "$low", "$volume"]
df_sample = D.features(
    inst_list[:3], sample_fields,
    start_time="2026-05-06", end_time="2026-05-08", freq="day",
)
print(f"\n  特征样例:")
print(df_sample.to_string())

# ============================================================
# 3. 定义 Task 配置 (Model + Dataset)
# ============================================================
task_config = {
    "model": {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": LGB_PARAMS,
    },
    "dataset": {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "start_time": TRAIN_START,
                    "end_time": TEST_END,
                    "fit_start_time": TRAIN_START,
                    "fit_end_time": TRAIN_END,
                    "instruments": MARKET,
                    "infer_processors": [
                        {
                            "class": "RobustZScoreNorm",
                            "kwargs": {"fields_group": "feature", "clip_outlier": True},
                        },
                        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                    ],
                    "learn_processors": [
                        {"class": "DropnaLabel"},
                        {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
                    ],
                    # label: T+1 到 T+2 的收益率 (即下一个交易日的收益)
                    "label": ["Ref($close, -2) / Ref($close, -1) - 1"],
                },
            },
            "segments": {
                "train": [TRAIN_START, TRAIN_END],
                "valid": [VALID_START, VALID_END],
                "test":  [TEST_START, TEST_END],
            },
        },
    },
}

# ============================================================
# 4. 创建 Model 和 Dataset
# ============================================================
print("\n" + "=" * 60)
print("【创建 Model 和 Dataset】")
model = init_instance_by_config(task_config["model"])
dataset = init_instance_by_config(task_config["dataset"])

print(f"  Model: {type(model).__name__}")
print(f"  Dataset: {type(dataset).__name__}")

df_train = dataset.prepare("train")
print(f"  训练集 shape: {df_train.shape}")

# ============================================================
# 5. 训练 + 评估 + 回测
# ============================================================
print("\n" + "=" * 60)
print("【训练 + 评估】")

port_analysis_config = {
    "executor": {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        },
    },
    "strategy": {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy.signal_strategy",
        "kwargs": {
            "signal": (model, dataset),
            "topk": TOPK,
            "n_drop": N_DROP,
        },
    },
    "backtest": {
        "start_time": TEST_START,
        "end_time": TEST_END,
        "account": ACCOUNT,
        "benchmark": inst_list[0] if inst_list else "SH600036",
        "exchange_kwargs": {
            "freq": "day",
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    },
}

with R.start(experiment_name=EXPERIMENT_NAME):
    # 记录参数
    R.log_params(**flatten_dict(task_config))

    # 训练
    model.fit(dataset)
    R.save_objects(**{"params.pkl": model})

    # 预测 + 信号记录
    recorder = R.get_recorder()
    sr = SignalRecord(model, dataset, recorder)
    sr.generate()

    # 信号分析 (IC, Rank IC 等)
    sar = SigAnaRecord(recorder)
    sar.generate()

    # 回测分析
    par = PortAnaRecord(recorder, port_analysis_config, "day")
    par.generate()

# ============================================================
# 6. 输出结果
# ============================================================
print("\n" + "=" * 60)
print("【训练完成】")
print(f"  Experiment: {EXPERIMENT_NAME}")
print(f"  Recorder ID: {recorder.id}")
print(f"  模型已保存, 可用于 predict.py 预测")
print("=" * 60)
