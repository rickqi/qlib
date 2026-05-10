# ============================================================
# Qlib 预测脚本 — tradingagents 数据
# ============================================================
# 用法:
#   cd D:\codes\qlib
#   C:\Users\szk220009\.qlib\venv\Scripts\python.exe docs\tradingagents\predict.py
#
# 可选参数 (通过环境变量):
#   TOP_N=5               推荐前 N 只 (默认 5)
#   EXPERIMENT_NAME=xxx   实验名称 (默认 tradingagents_lgb)
# ============================================================

import os
import sys
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay

import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.utils import init_instance_by_config
from qlib.workflow import R

# ============================================================
# 配置区域
# ============================================================
PROVIDER_URI = "C:/Users/szk220009/.qlib/qlib_data/tradingagents"
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "tradingagents_lgb")
TOP_N = int(os.environ.get("TOP_N", "5"))
MARKET = "all"

# 数据集配置 (需与 train.py 一致)
TRAIN_START = "2021-01-01"
TRAIN_END   = "2023-12-31"
VALID_START = "2024-01-01"
VALID_END   = "2024-12-31"
TEST_START  = "2025-01-01"
TEST_END    = "2026-05-08"

# ============================================================
# 1. 初始化 Qlib
# ============================================================
qlib.init(provider_uri=PROVIDER_URI, region=REG_CN)

# ============================================================
# 2. 日历检查
# ============================================================
calendar = D.calendar(freq="day")

print("=" * 60)
print("【日历信息】")
print(f"  最新 5 个交易日:")
for d in calendar[-5:]:
    ts = pd.Timestamp(d)
    weekday_cn = {
        "Monday": "周一", "Tuesday": "周二", "Wednesday": "周三",
        "Thursday": "周四", "Friday": "周五", "Saturday": "周六", "Sunday": "周日",
    }
    print(f"    {str(d)[:10]} ({weekday_cn.get(ts.day_name(), ts.day_name())})")

last_date = calendar[-1]
last_ts = pd.Timestamp(last_date)
# 计算下一个交易日
next_ts = last_ts + BDay(1)
weekday_cn = {
    "Monday": "周一", "Tuesday": "周二", "Wednesday": "周三",
    "Thursday": "周四", "Friday": "周五", "Saturday": "周六", "Sunday": "周日",
}
print(f"\n  数据截止: {str(last_date)[:10]} ({weekday_cn.get(last_ts.day_name(), last_ts.day_name())})")
print(f"  预测目标: {str(next_ts.date())} ({weekday_cn.get(next_ts.day_name(), next_ts.day_name())})")

# ============================================================
# 3. 加载已训练模型
# ============================================================
print("\n" + "=" * 60)
print("【加载模型】")

experiment = R.get_exp(experiment_name=EXPERIMENT_NAME)
recorders = experiment.list_recorders()

# 取最新的 recorder
latest_recorder = list(recorders.values())[-1]
model = latest_recorder.load_object("params.pkl")
print(f"  Experiment: {EXPERIMENT_NAME}")
print(f"  Recorder ID: {latest_recorder.id}")
print(f"  Model: {type(model).__name__}")

# ============================================================
# 4. 构建数据集并预测
# ============================================================
print("\n" + "=" * 60)
print("【构建数据集并预测】")

dataset_config = {
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
                "label": ["Ref($close, -2) / Ref($close, -1) - 1"],
            },
        },
        "segments": {
            "train": [TRAIN_START, TRAIN_END],
            "valid": [VALID_START, VALID_END],
            "test":  [TEST_START, TEST_END],
        },
    },
}

dataset = init_instance_by_config(dataset_config)
pred_score = model.predict(dataset, segment="test")

# ============================================================
# 5. 取最新交易日预测
# ============================================================
print("\n" + "=" * 60)
print("【预测结果】")

# pred_score 可能是 DataFrame 或 Series
if isinstance(pred_score, pd.DataFrame):
    pred_series = pred_score.iloc[:, 0]
else:
    pred_series = pred_score

# pred_series 是 MultiIndex (datetime, instrument)
last_pred_date = pred_series.index.get_level_values(0).max()
latest_pred = pred_series.loc[last_pred_date].dropna().sort_values(ascending=False)

print(f"\n  数据截止日: {str(last_pred_date)[:10]}")
print(f"  预测含义: score 越高 → 下一个交易日预期收益越高")
print(f"  预测目标日: {str(next_ts.date())}")
print()

# 打印全部排序
print(f"  {'排名':>4s}  {'股票代码':<12s}  {'预测得分':>10s}")
print("  " + "-" * 35)
for rank, (inst, score) in enumerate(latest_pred.items(), 1):
    print(f"  {rank:>4d}  {inst:<12s}  {score:>10.6f}")

# ============================================================
# 6. Top N 推荐 + 最新价格
# ============================================================
print(f"\n{'=' * 60}")
print(f"【推荐买入 Top {TOP_N}】")

for rank, (inst, score) in enumerate(latest_pred.head(TOP_N).items(), 1):
    try:
        pdf = D.features(
            [inst], ["$close", "$volume"],
            start_time="2026-05-06", end_time="2026-05-08", freq="day",
        )
        pdf = pdf.dropna()
        if not pdf.empty:
            price = pdf.iloc[-1]["$close"]
            vol = pdf.iloc[-1]["$volume"]
            print(f"  {rank}. {inst:<12s}  score={score:.6f}  收盘={price:.2f}  成交量={vol:.0f}")
        else:
            print(f"  {rank}. {inst:<12s}  score={score:.6f}  (无最新价格)")
    except Exception as e:
        print(f"  {rank}. {inst:<12s}  score={score:.6f}  (获取价格失败)")

# ============================================================
# 7. 保存预测结果
# ============================================================
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)
output_file = output_dir / f"prediction_{str(last_pred_date)[:10]}.csv"

latest_pred.to_csv(output_file, header=["score"])
print(f"\n  预测结果已保存: {output_file}")
print("=" * 60)
