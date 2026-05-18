"""
训练脚本：使用 LightGBM + Alpha158/Alpha360 模型进行训练，支持多数据源。

用法:
    .venv\\Scripts\\python.exe scripts\\train.py                          # 默认 qlib_bin + Alpha158
    .venv\\Scripts\\python.exe scripts\\train.py --handler alpha360       # Alpha360 特征
    .venv\\Scripts\\python.exe scripts\\train.py --data cn_data
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord

# ── 数据源配置 ──────────────────────────────────────────────────────
DATA_CONFIGS = {
    "qlib_bin": {
        "provider_uri": str(Path(__file__).resolve().parent.parent / "qlib_bin"),
        "market": "all",
        "train_start": "2020-01-01",
        "train_end": "2024-12-31",
        "valid_start": "2025-01-01",
        "valid_end": "2025-12-31",
        "test_start": "2026-01-01",
        "test_end": "2026-05-15",
    },
    "cn_data": {
        "provider_uri": "~/.qlib/qlib_data/cn_data",
        "market": "all",
        "train_start": "2010-01-01",
        "train_end": "2018-12-31",
        "valid_start": "2019-01-01",
        "valid_end": "2020-03-31",
        "test_start": "2020-04-01",
        "test_end": "2020-09-24",
    },
    "tradingagents": {
        "provider_uri": "~/.qlib/qlib_data/tradingagents",
        "market": "all",
        "train_start": "2021-01-01",
        "train_end": "2024-12-31",
        "valid_start": "2025-01-01",
        "valid_end": "2025-06-30",
        "test_start": "2025-07-01",
        "test_end": "2026-05-11",
    },
}

REGION = REG_CN
MARKET = "all"
BENCHMARK = "SH000300"

# ── Handler 映射 ────────────────────────────────────────────────────
HANDLER_MAP = {
    "alpha158": {"class": "Alpha158", "module_path": "qlib.contrib.data.handler"},
    "alpha360": {"class": "Alpha360", "module_path": "qlib.contrib.data.handler"},
}

# 输出目录
REPORTS_DIR = Path(__file__).parent.parent / "reports"


def get_dataset_config(data_cfg, train_end, valid_start, valid_end, test_start, test_end, handler="alpha158"):
    """构建 Dataset 配置（Alpha158/Alpha360 Handler + DatasetH）。"""
    handler_def = HANDLER_MAP[handler]
    data_handler_config = {
        "start_time": data_cfg["train_start"],
        "end_time": test_end,
        "fit_start_time": data_cfg["train_start"],
        "fit_end_time": train_end,
        "instruments": data_cfg.get("market", "all"),
    }
    return {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                **handler_def,
                "kwargs": data_handler_config,
            },
            "segments": {
                "train": [data_cfg["train_start"], train_end],
                "valid": [valid_start, valid_end],
                "test": [test_start, test_end],
            },
        },
    }


def get_model_config(model_name="lgbm"):
    """构建模型配置。"""
    if model_name == "lgbm":
        return {
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
                "num_threads": 4,
            },
        }
    raise ValueError(f"Unsupported model: {model_name}")


def get_port_analysis_config(test_start, test_end):
    """构建回测组合分析配置。"""
    return {
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
                "signal": "<PRED>",
                "topk": 50,
                "n_drop": 5,
            },
        },
        "backtest": {
            "start_time": test_start,
            "end_time": test_end,
            "account": 100000000,
            "benchmark": BENCHMARK,
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


def main():
    parser = argparse.ArgumentParser(description="Qlib 训练脚本")
    parser.add_argument("--data", default="qlib_bin", choices=list(DATA_CONFIGS.keys()), help="数据源")
    parser.add_argument("--model", default="lgbm", help="模型名称 (lgbm)")
    parser.add_argument("--handler", default="alpha158", choices=list(HANDLER_MAP.keys()), help="特征处理器 (alpha158/alpha360)")
    parser.add_argument("--market", default=None, help="覆盖市场 (all/csi300/csi500/csi800/csi1000/csiall)")
    parser.add_argument("--sequential", action="store_true", help="禁用并行处理（解决 Windows OOM 问题）")
    parser.add_argument("--train-start", default=None, help="覆盖训练起始日期（如 2023-01-01）")
    args = parser.parse_args()

    data_cfg = DATA_CONFIGS[args.data]
    # 允许命令行覆盖 market
    if args.market:
        data_cfg = {**data_cfg, "market": args.market}
    # 允许命令行覆盖 train_start（Alpha360 等大特征集可缩短训练期以降低内存）
    if args.train_start:
        data_cfg = {**data_cfg, "train_start": args.train_start}
    provider_uri = data_cfg["provider_uri"]
    train_start = data_cfg["train_start"]
    train_end = data_cfg["train_end"]
    valid_start = data_cfg["valid_start"]
    valid_end = data_cfg["valid_end"]
    test_start = data_cfg["test_start"]
    test_end = data_cfg["test_end"]

    # ── Fix label leakage ──────────────────────────────────────
    # Alpha158 label = Ref($close, -2)/Ref($close, -1) - 1
    # Last 2 training days reference validation period prices.
    # Shift train_end back by 2 trading days (~4 calendar days).
    LABEL_LEAK_DAYS = 2
    train_end_dt = datetime.strptime(train_end, "%Y-%m-%d") - timedelta(days=LABEL_LEAK_DAYS * 2)
    train_end_adj = train_end_dt.strftime("%Y-%m-%d")
    print(f"[LEAK-FIX] train_end 调整: {train_end} → {train_end_adj} (回退 {LABEL_LEAK_DAYS} 交易日)")
    train_end = train_end_adj
    # ────────────────────────────────────────────────────────────

    print(f"{'=' * 60}")
    print(f"Qlib 训练 - {args.model} + {args.handler} (数据源: {args.data})")
    print(f"Provider: {provider_uri}")
    print(f"训练: {train_start} ~ {train_end}")
    print(f"验证: {valid_start} ~ {valid_end}")
    print(f"测试: {test_start} ~ {test_end}")
    print(f"{'=' * 60}")

    # 初始化 qlib（Alpha360 在 Windows 上易 OOM，可用 --sequential 禁用并行）
    init_kwargs = {}
    if args.sequential:
        init_kwargs["kernels"] = 1
        init_kwargs["joblib_backend"] = "sequential"
        print("[INFO] 已启用 sequential 模式 (kernels=1, backend=sequential)")
    qlib.init(provider_uri=provider_uri, region=REGION, **init_kwargs)

    # ── Memory-aware training with auto-retry ───────────────────────
    model_config = get_model_config(args.model)
    dataset_config = get_dataset_config(data_cfg, train_end, valid_start, valid_end, test_start, test_end, handler=args.handler)

    FALLBACK_MARKETS = [None, "csi800", "csi500"]  # None = original market
    recorder = None
    experiment_name = None

    for attempt, fallback_market in enumerate(FALLBACK_MARKETS):
        try:
            if attempt > 0:
                # Retry with smaller market
                data_cfg_fb = {**data_cfg}
                if fallback_market:
                    data_cfg_fb["market"] = fallback_market
                print(f"\n[RETRY-{attempt}] 降级到 market={fallback_market or 'all'}...")
                dataset_config = get_dataset_config(data_cfg_fb, train_end, valid_start, valid_end, test_start, test_end, handler=args.handler)

            model = init_instance_by_config(model_config)
            dataset = init_instance_by_config(dataset_config)

            # Print dataset summary
            train_df = dataset.prepare("train")
            print(f"\n训练集: {train_df.shape}")
            print(f"特征列: {list(train_df.columns)[:5]}... (共 {len(train_df.columns)} 列)")
            del train_df  # Free memory before training

            # Port analysis config
            port_config = get_port_analysis_config(test_start, test_end)

            # Start experiment
            experiment_name = f"train_{args.data}_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with R.start(experiment_name=experiment_name):
                R.log_params(model=args.model, data_source=args.data, **flatten_dict({"model": model_config, "dataset": dataset_config}))
                model.fit(dataset)
                R.save_objects(**{"params.pkl": model})

                # Signal record
                recorder = R.get_recorder()
                sr = SignalRecord(model, dataset, recorder)
                sr.generate()

                # Signal analysis
                sar = SigAnaRecord(recorder)
                sar.generate()

                # Backtest analysis
                try:
                    par = PortAnaRecord(recorder, port_config, "day")
                    par.generate()
                except (ValueError, KeyError, IndexError) as e:
                    print(f"[警告] 回测跳过: {e}")

            break  # Success — exit retry loop

        except MemoryError:
            if attempt < len(FALLBACK_MARKETS) - 1:
                print(f"\n[OOM] MemoryError! 将重试 (attempt {attempt + 1}/{len(FALLBACK_MARKETS) - 1})")
                import gc
                gc.collect()
            else:
                print(f"\n[ERROR] 所有降级方案均失败。请使用 --market csi800 或 --sequential 参数。")
                raise
        except Exception:
            # Non-memory errors should not trigger retry
            raise
    # ─────────────────────────────────────────────────────────────────
    assert recorder is not None and experiment_name is not None, "Training failed — no recorder available"

    # 保存实验信息
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    exp_info = {
        "experiment_name": experiment_name,
        "model": args.model,
        "handler": args.handler,
        "data_source": args.data,
        "train_period": [train_start, train_end],
        "valid_period": [valid_start, valid_end],
        "test_period": [test_start, test_end],
        "recorder_id": recorder.id,
        "timestamp": datetime.now().isoformat(),
    }
    info_path = REPORTS_DIR / "train_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(exp_info, f, ensure_ascii=False, indent=2)

    print(f"\n训练完成！实验名称: {experiment_name}")
    print(f"Recorder ID: {recorder.id}")
    print(f"实验信息已保存到: {info_path}")

    return experiment_name, recorder.id


if __name__ == "__main__":
    main()
