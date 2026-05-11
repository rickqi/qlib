"""
预测脚本：使用已训练的模型对指定股票列表进行预测，支持多数据源和多特征处理器。

用法:
    .venv\\Scripts\\python.exe scripts\\predict.py --data qlib_bin
    .venv\\Scripts\\python.exe scripts\\predict.py --data qlib_bin --latest-day
    .venv\\Scripts\\python.exe scripts\\predict.py --data qlib_bin --handler alpha360
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
from qlib.data import D
from qlib.utils import init_instance_by_config

# ── 数据源配置 ──────────────────────────────────────────────────────
PREDICT_CONFIGS = {
    "qlib_bin": {
        "provider_uri": "C:/codes/qlib/qlib_bin",
        "label": "qlib_bin (完整中国A股数据)",
        "check_start": "2026-04-01",
        "check_end": "2026-05-08",
        "predict_start": "2026-05-06",
        "predict_end": "2026-05-08",
        "fit_start": "2020-01-01",
        "fit_end": "2024-12-31",
        "handler_start": "2025-10-01",
        "train_seg": ["2020-01-01", "2024-12-31"],
        "valid_seg": ["2025-01-01", "2025-12-31"],
    },
    "cn_data": {
        "provider_uri": "~/.qlib/qlib_data/cn_data",
        "label": "cn_data (标准中国A股数据)",
        "check_start": "2020-01-01",
        "check_end": "2020-09-25",
        "predict_start": "2020-09-21",
        "predict_end": "2020-09-25",
        "fit_start": "2010-01-01",
        "fit_end": "2018-12-31",
        "handler_start": "2019-06-01",
        "train_seg": ["2010-01-01", "2018-12-31"],
        "valid_seg": ["2019-01-01", "2020-03-31"],
    },
    "tradingagents": {
        "provider_uri": "~/.qlib/qlib_data/tradingagents",
        "label": "tradingagents (自建数据)",
        "check_start": "2026-05-01",
        "check_end": "2026-05-08",
        "predict_start": "2026-05-06",
        "predict_end": "2026-05-08",
        "fit_start": "2021-01-01",
        "fit_end": "2024-12-31",
        "handler_start": "2026-04-01",
        "train_seg": ["2021-01-01", "2024-12-31"],
        "valid_seg": ["2025-01-01", "2025-06-30"],
    },
}

REGION = REG_CN
REPORTS_DIR = Path(__file__).parent.parent / "reports"
DOCS_DIR = Path(__file__).parent.parent / "docs"
RECENT_DAYS = 5

# ── Handler 映射 ────────────────────────────────────────────────────
HANDLER_MAP = {
    "alpha158": {"class": "Alpha158", "module_path": "qlib.contrib.data.handler"},
    "alpha360": {"class": "Alpha360", "module_path": "qlib.contrib.data.handler"},
}

# 用户请求的股票列表（原始格式）
REQUESTED_STOCKS = [
    "688041.SH", "688256.SH", "688012.SH", "603986.SH", "688008.SH",
    "300442.SZ", "603019.SH", "688111.SH", "002230.SZ", "002837.SZ",
    "002049.SZ", "688027.SH", "300223.SZ", "301269.SZ", "002747.SZ",
    "688332.SH", "002896.SZ", "688568.SH", "300672.SZ", "300458.SZ",
]


def convert_stock_code(code: str) -> str:
    """转换股票代码格式: 688041.SH -> SH688041 (qlib 格式)。"""
    parts = code.split(".")
    if len(parts) == 2:
        return parts[1].upper() + parts[0]
    return code.upper()


def get_available_stocks(stock_list, cfg):
    """检查股票是否在数据中可用。"""
    instruments = D.instruments("all")
    all_stocks = D.list_instruments(
        instruments=instruments,
        start_time=cfg["check_start"],
        end_time=cfg["check_end"],
        as_list=True,
    )
    available = []
    unavailable = []
    for s in stock_list:
        qlib_code = convert_stock_code(s)
        if qlib_code in all_stocks:
            available.append((s, qlib_code))
        else:
            unavailable.append(s)
    return available, unavailable


def load_trained_model():
    """从最新的训练信息中加载模型。"""
    info_path = REPORTS_DIR / "train_info.json"
    if not info_path.exists():
        print("错误: 未找到训练信息。请先运行 train.py")
        sys.exit(1)

    with open(info_path, "r", encoding="utf-8") as f:
        train_info = json.load(f)

    from qlib.workflow import R

    # 获取 recorder 并加载模型
    recorder = R.get_recorder(
        recorder_id=train_info["recorder_id"],
        experiment_name=train_info["experiment_name"],
    )
    model = recorder.load_object("params.pkl")
    return model, train_info


def predict_for_stocks(model, stock_codes_qlib, cfg, handler="alpha158"):
    """对指定股票进行预测。"""
    handler_def = HANDLER_MAP[handler]
    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                **handler_def,
                "kwargs": {
                    "start_time": cfg["handler_start"],
                    "end_time": cfg["predict_end"],
                    "fit_start_time": cfg["fit_start"],
                    "fit_end_time": cfg["fit_end"],
                    "instruments": stock_codes_qlib,
                },
            },
            "segments": {
                "train": cfg["train_seg"],
                "valid": cfg["valid_seg"],
                "test": [cfg["predict_start"], cfg["predict_end"]],
            },
        },
    }
    dataset = init_instance_by_config(dataset_config)
    pred = model.predict(dataset)
    return pred


def get_recent_data(stock_codes_qlib, days=RECENT_DAYS):
    """获取最近几天的行情数据。"""
    cal = D.calendar(freq="day")
    recent_dates = cal[-days:]
    start = str(recent_dates[0].date())
    end = str(recent_dates[-1].date())
    fields = ["$close", "$open", "$high", "$low", "$volume"]
    df = D.features(stock_codes_qlib, fields, start_time=start, end_time=end, freq="day")
    if df is not None and not df.empty:
        # 重置 index 并整理列名
        df = df.reset_index()
        df.columns = ["instrument", "datetime"] + [c.replace("$", "") for c in fields]
    return df


def generate_prediction_report(predictions, available, unavailable, recent_data, data_label, latest_day=False):
    """生成预测报告。"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── 1. 预测结果 CSV ──
    if isinstance(predictions, pd.Series):
        pred_df = predictions.reset_index()
        # qlib 的 predict 返回 MultiIndex: (datetime, instrument) 或 (instrument, datetime)
        # 按 index names 来判断列含义
        idx_names = list(predictions.index.names)
        raw_cols = list(pred_df.columns)
        # 最后一列是预测值
        raw_cols[-1] = "score"
        # 根据 index names 识别前两列
        for i in range(len(raw_cols) - 1):
            orig_name = str(idx_names[i]) if i < len(idx_names) else ""
            col_val = pred_df.iloc[0, i]
            # instrument 列值通常以 SH/SZ 开头，datetime 列值是 Timestamp
            if isinstance(col_val, pd.Timestamp) or "datetime" in orig_name.lower() or "time" in orig_name.lower():
                raw_cols[i] = "datetime"
            else:
                raw_cols[i] = "instrument"
        pred_df.columns = raw_cols
    else:
        pred_df = predictions.copy()

    # 添加原始代码映射
    code_map = {qc: orig for orig, qc in available}
    if "instrument" in pred_df.columns:
        pred_df["stock_code"] = pred_df["instrument"].astype(str).map(code_map)
        pred_df["stock_code"] = pred_df["stock_code"].fillna(pred_df["instrument"].astype(str))
    else:
        pred_df["stock_code"] = "N/A"

    # 排序（分数从高到低）
    pred_df = pred_df.sort_values("score", ascending=False).reset_index(drop=True)

    # 如果 latest_day 模式，只保留最后一个交易日的预测
    target_date_label = None
    if latest_day and "datetime" in pred_df.columns:
        latest_date = pred_df["datetime"].max()
        pred_df = pred_df[pred_df["datetime"] == latest_date].reset_index(drop=True)
        latest_str = str(pd.Timestamp(latest_date).date())
        # 从 day_future.txt 读取下一交易日
        try:
            from pathlib import Path as _P
            future_cal_path = _P("C:/codes/qlib/qlib_bin/calendars/day_future.txt")
            if future_cal_path.exists():
                with open(future_cal_path, "r") as f:
                    dates = [line.strip() for line in f if line.strip()]
                idx = dates.index(latest_str) if latest_str in dates else -1
                if idx >= 0 and idx + 1 < len(dates):
                    next_trading_day = dates[idx + 1]
                    target_date_label = f"基于 {latest_str} 收盘数据 → 预测 {next_trading_day} 涨跌"
                else:
                    target_date_label = f"基于 {latest_str} 收盘数据预测下一交易日"
            else:
                target_date_label = f"基于 {latest_str} 收盘数据预测"
        except Exception:
            target_date_label = f"基于 {latest_str} 收盘数据预测"
        print(f"[latest-day] 筛选至最新交易日: {latest_str} → {target_date_label}")

    csv_path = REPORTS_DIR / f"predictions_{timestamp}.csv"
    pred_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n预测结果已保存: {csv_path}")

    # ── 2. 分析报告 (Markdown) ──
    report_lines = [
        f"# 股票预测报告",
        f"",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**数据源**: {data_label}",
        f"**模型**: LightGBM (Alpha158)",
    ]
    if target_date_label:
        report_lines.append(f"**预测目标**: {target_date_label}")
    report_lines.extend([
        f"",
        f"## 预测概要",
        f"",
        f"- 可用股票: {len(available)} 只",
        f"- 数据缺失: {len(unavailable)} 只",
    ])

    if unavailable:
        report_lines.append(f"- 缺失股票: {', '.join(unavailable)}")

    report_lines.extend([
        f"",
        f"## 预测排名（分数从高到低）",
        f"",
        f"| 排名 | 股票代码 | 预测分数 | 建议 |",
        f"|------|----------|----------|------|",
    ])

    for i, row in pred_df.iterrows():
        score = row["score"]
        if score > 0.01:
            suggestion = "看多（偏多操作）"
        elif score > 0:
            suggestion = "轻微看多"
        elif score > -0.01:
            suggestion = "中性"
        else:
            suggestion = "看空（谨慎操作）"

        code = row.get("stock_code", row["instrument"])
        report_lines.append(f"| {i+1} | {code} | {score:.6f} | {suggestion} |")

    # 最近行情
    report_lines.extend([
        f"",
        f"## 最近行情数据",
        f"",
    ])
    if recent_data is not None and not recent_data.empty:
        report_lines.append(recent_data.to_markdown())

    report_lines.extend([
        f"",
        f"## 免责声明",
        f"",
        f"本报告由量化模型自动生成，仅供参考，不构成任何投资建议。",
        f"投资有风险，入市需谨慎。",
    ])

    report_path = DOCS_DIR / "analysis" / f"prediction_report_{timestamp}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"分析报告已保存: {report_path}")

    # ── 3. 简明摘要到 stdout ──
    print(f"\n{'=' * 60}")
    print("预测结果摘要")
    print(f"{'=' * 60}")
    for i, row in pred_df.iterrows():
        code = row.get("stock_code", row["instrument"])
        score = row["score"]
        direction = "↑" if score > 0 else "↓"
        print(f"  {i+1:2d}. {code:>12s}  分数: {score:+.6f} {direction}")
    print(f"{'=' * 60}")

    return csv_path, report_path


def main():
    parser = argparse.ArgumentParser(description="Qlib 预测脚本")
    parser.add_argument("--data", default="qlib_bin", choices=list(PREDICT_CONFIGS.keys()), help="数据源")
    parser.add_argument(
        "--stocks",
        default=None,
        help="股票列表，逗号分隔 (如: SH688041,SH688256)",
    )
    parser.add_argument("--topk", type=int, default=0, help="只预测前K只股票")
    parser.add_argument("--latest-day", action="store_true", help="只输出最后一天预测（即下一交易日预测）")
    parser.add_argument("--handler", default=None, choices=list(HANDLER_MAP.keys()), help="特征处理器（默认从 train_info.json 自动读取）")
    args = parser.parse_args()

    cfg = PREDICT_CONFIGS[args.data]

    # 初始化
    qlib.init(provider_uri=cfg["provider_uri"], region=REGION)

    # 确定股票列表
    if args.stocks:
        stock_list = [s.strip() for s in args.stocks.split(",")]
    else:
        stock_list = REQUESTED_STOCKS

    if args.topk > 0:
        stock_list = stock_list[: args.topk]

    print(f"数据源: {args.data} ({cfg['label']})")
    print(f"请求预测股票: {len(stock_list)} 只")

    # 检查可用性
    available, unavailable = get_available_stocks(stock_list, cfg)
    print(f"可用: {len(available)} 只, 缺失: {len(unavailable)} 只")
    if unavailable:
        print(f"  缺失股票: {', '.join(unavailable)}")

    if not available:
        print("没有可用的股票进行预测！")
        sys.exit(1)

    # 加载模型
    model, train_info = load_trained_model()
    # 确定使用的 handler（命令行 > train_info > 默认 alpha158）
    handler = args.handler or train_info.get("handler", "alpha158")
    print(f"已加载模型: {train_info['model']} + {handler}")

    # 预测
    qlib_codes = [qc for _, qc in available]
    pred = predict_for_stocks(model, qlib_codes, cfg, handler=handler)
    print(f"预测完成，共 {len(pred) if isinstance(pred, pd.Series) else 'N/A'} 条记录")

    # 获取最近行情
    recent_data = get_recent_data(qlib_codes, days=RECENT_DAYS)

    # 生成报告
    generate_prediction_report(pred, available, unavailable, recent_data, cfg["label"], latest_day=args.latest_day)


if __name__ == "__main__":
    main()
