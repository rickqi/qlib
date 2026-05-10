# ============================================================
# 预测指定股票下一个交易日(2026-05-11) 的价格
# ============================================================

import os
import sys
sys.path.insert(0, r"D:\codes\qlib")

import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay

import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.utils import init_instance_by_config
from qlib.workflow import R

# ============================================================
# 目标股票 (用户格式 → qlib 格式转换)
# ============================================================
RAW_CODES = [
    "688041.SH", "688256.SH", "688012.SH", "603986.SH", "688008.SH",
    "300442.SZ", "603019.SH", "688111.SH", "002230.SZ", "002837.SZ",
    "002049.SZ", "688027.SH", "300223.SZ", "301269.SZ", "002747.SZ",
    "688332.SH", "002896.SZ", "688568.SH", "300672.SZ", "300458.SZ",
]

# 转换: "688041.SH" → "SH688041"
def to_qlib_code(code):
    num, mkt = code.split(".")
    return f"{mkt.upper()}{num}"

QLIB_CODES = [to_qlib_code(c) for c in RAW_CODES]
CODE_MAP = {to_qlib_code(c): c for c in RAW_CODES}  # qlib → 用户格式

# ============================================================
# 1. 初始化
# ============================================================
provider_uri = "C:/Users/szk220009/.qlib/qlib_data/tradingagents"
qlib.init(provider_uri=provider_uri, region=REG_CN)

# ============================================================
# 2. 获取最新收盘价
# ============================================================
df_price = D.features(
    QLIB_CODES, ["$close", "$open", "$high", "$low", "$volume"],
    start_time="2026-05-06", end_time="2026-05-08", freq="day",
)
# 取最后一行 (5/8)
if isinstance(df_price.index, pd.MultiIndex):
    last_date = df_price.index.get_level_values(0).max()
    latest = df_price.loc[last_date].copy()
else:
    latest = df_price.tail(len(QLIB_CODES)).copy()

latest.columns = [c.replace("$", "") for c in latest.columns]

# ============================================================
# 3. 加载已训练模型
# ============================================================
experiment = R.get_exp(experiment_name="tradingagents_lgb")
recorders = experiment.list_recorders()
latest_recorder = list(recorders.values())[-1]
model = latest_recorder.load_object("params.pkl")

# ============================================================
# 4. 用原始标签训练一个回归模型 → 直接预测收益率
#    (原模型用 CSRankNorm, 输出是排名分不是实际收益)
# ============================================================
print("训练回归模型 (raw return label) ...")

dataset_reg_config = {
    "class": "DatasetH",
    "module_path": "qlib.data.dataset",
    "kwargs": {
        "handler": {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "start_time": "2021-01-01",
                "end_time": "2026-05-08",
                "fit_start_time": "2021-01-01",
                "fit_end_time": "2023-12-31",
                "instruments": "all",
                "infer_processors": [
                    {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                    {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                ],
                "learn_processors": [
                    {"class": "DropnaLabel"},
                ],
                # 直接预测下一交易日收益率 (不做排名归一化)
                "label": ["Ref($close, -2) / Ref($close, -1) - 1"],
            },
        },
        "segments": {
            "train": ["2021-01-01", "2025-04-30"],
            "valid": ["2025-05-01", "2026-03-31"],
            "test":  ["2026-04-01", "2026-05-08"],
        },
    },
}

dataset_reg = init_instance_by_config(dataset_reg_config)

reg_model_config = {
    "class": "LGBModel",
    "module_path": "qlib.contrib.model.gbdt",
    "kwargs": {
        "loss": "mse",
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 80,
        "num_threads": 4,
        "early_stopping_rounds": 100,
        "n_estimators": 500,
    },
}

reg_model = init_instance_by_config(reg_model_config)
reg_model.fit(dataset_reg)

# ============================================================
# 5. 预测下一交易日收益率
# ============================================================
pred_ret = reg_model.predict(dataset_reg, segment="test")

if isinstance(pred_ret, pd.DataFrame):
    pred_ret = pred_ret.iloc[:, 0]

last_pred_date = pred_ret.index.get_level_values(0).max()
pred_on_last = pred_ret.loc[last_pred_date]

# ============================================================
# 6. 合并: 最新价格 + 预测收益率 → 预测价格
# ============================================================
next_trade_day = pd.Timestamp(last_pred_date) + BDay(1)

results = []
for qlib_code in QLIB_CODES:
    user_code = CODE_MAP[qlib_code]
    try:
        cur_close = latest.loc[qlib_code, "close"]
        cur_open = latest.loc[qlib_code, "open"]
        cur_high = latest.loc[qlib_code, "high"]
        cur_low = latest.loc[qlib_code, "low"]
        cur_vol = latest.loc[qlib_code, "volume"]
    except KeyError:
        cur_close = cur_open = cur_high = cur_low = cur_vol = np.nan

    # 预测收益率
    if qlib_code in pred_on_last.index:
        pred_return = pred_on_last.loc[qlib_code]
    else:
        pred_return = np.nan

    # 预测价格 = 当前收盘价 × (1 + 预测收益率)
    pred_price = cur_close * (1 + pred_return) if not np.isnan(cur_close) and not np.isnan(pred_return) else np.nan
    pred_change = pred_return * 100 if not np.isnan(pred_return) else np.nan

    results.append({
        "code": user_code,
        "current_close": cur_close,
        "pred_return": pred_return,
        "pred_change_pct": pred_change,
        "pred_price": pred_price,
        "pred_limit_up": cur_close * 1.1 if not np.isnan(cur_close) else np.nan,
        "pred_limit_down": cur_close * 0.9 if not np.isnan(cur_close) else np.nan,
    })

df_result = pd.DataFrame(results)

# ============================================================
# 7. 输出
# ============================================================
print("\n" + "=" * 80)
print(f"  股价预测 — 数据截止: {str(last_pred_date)[:10]} → 预测目标: {str(next_trade_day.date())} (下一交易日)")
print("=" * 80)
print()
print(f"  {'股票代码':<12s}  {'5/8收盘':>10s}  {'预测涨跌幅':>10s}  {'预测价格':>10s}  {'涨停价':>10s}  {'跌停价':>10s}  {'方向':>4s}")
print("  " + "-" * 75)

for _, row in df_result.iterrows():
    code = row["code"]
    cc = f"{row['current_close']:.2f}" if not np.isnan(row["current_close"]) else "N/A"
    chg = f"{row['pred_change_pct']:+.2f}%" if not np.isnan(row["pred_change_pct"]) else "N/A"
    pp = f"{row['pred_price']:.2f}" if not np.isnan(row["pred_price"]) else "N/A"
    lu = f"{row['pred_limit_up']:.2f}" if not np.isnan(row["pred_limit_up"]) else "N/A"
    ld = f"{row['pred_limit_down']:.2f}" if not np.isnan(row["pred_limit_down"]) else "N/A"
    direction = "↑" if not np.isnan(row["pred_change_pct"]) and row["pred_change_pct"] > 0 else ("↓" if not np.isnan(row["pred_change_pct"]) else "?")
    print(f"  {code:<12s}  {cc:>10s}  {chg:>10s}  {pp:>10s}  {lu:>10s}  {ld:>10s}  {direction:>4s}")

# 保存
output_path = os.path.join(os.path.dirname(__file__), "output", f"price_pred_{str(next_trade_day.date())}.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_result.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"\n  已保存: {output_path}")
print("=" * 80)
