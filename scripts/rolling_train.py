"""
滚动训练脚本：使用 Rolling Retraining 策略训练 LightGBM 模型。

将时间范围按 step（默认 20 个交易日）切分为多个滚动窗口，
每个窗口训练一个模型，最后用 RollingEnsemble 合并预测结果。

用法:
    .venv\\Scripts\\python.exe scripts\\rolling_train.py
    .venv\\Scripts\\python.exe scripts\\rolling_train.py --step 40 --horizon 10
    .venv\\Scripts\\python.exe scripts\\rolling_train.py --market csi300
"""
import argparse
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.model.ens.ensemble import RollingEnsemble
from qlib.model.trainer import TrainerR
from qlib.workflow import R
from qlib.workflow.record_temp import SigAnaRecord
from qlib.workflow.task.collect import RecorderCollector
from qlib.workflow.task.gen import RollingGen, task_generator

# 复用 train.py 的数据源和 Handler 配置
from train import DATA_CONFIGS, HANDLER_MAP, get_model_config

REGION = REG_CN
BENCHMARK = "SH000300"
REPORTS_DIR = Path(__file__).parent.parent / "reports"


def build_base_task(data_cfg, train_end, valid_start, valid_end, test_start, test_end,
                    handler="alpha158", horizon=20):
    """构建 qlib 格式的 base task dict（用于 RollingGen 切分）。"""
    handler_def = HANDLER_MAP[handler]
    handler_kwargs = {
        "start_time": data_cfg["train_start"],
        "end_time": test_end,
        "fit_start_time": data_cfg["train_start"],
        "fit_end_time": train_end,
        "instruments": data_cfg.get("market", "all"),
        "label": [f"Ref($close, -{horizon + 1}) / Ref($close, -1) - 1"],
    }

    return {
        "model": get_model_config("lgbm"),
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    **handler_def,
                    "kwargs": handler_kwargs,
                },
                "segments": {
                    "train": [data_cfg["train_start"], train_end],
                    "valid": [valid_start, valid_end],
                    "test": [test_start, test_end],
                },
            },
        },
        "record": [
            {"class": "SignalRecord", "module_path": "qlib.workflow.record_temp"},
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Qlib 滚动训练脚本")
    parser.add_argument("--data", default="qlib_bin", choices=list(DATA_CONFIGS.keys()), help="数据源")
    parser.add_argument("--handler", default="alpha158", choices=list(HANDLER_MAP.keys()), help="特征处理器")
    parser.add_argument("--market", default=None, help="覆盖市场 (all/csi300/csi500/csi800/csi1000/csiall)")
    parser.add_argument("--train-start", default=None, help="覆盖训练起始日期")
    parser.add_argument("--step", type=int, default=20, help="滚动步长（交易日数，默认 20）")
    parser.add_argument("--horizon", type=int, default=20, help="预测 horizon（交易日数，默认 20）")
    parser.add_argument("--sequential", action="store_true", help="禁用并行处理")
    args = parser.parse_args()

    data_cfg = DATA_CONFIGS[args.data]
    if args.market:
        data_cfg = {**data_cfg, "market": args.market}
    if args.train_start:
        data_cfg = {**data_cfg, "train_start": args.train_start}

    provider_uri = data_cfg["provider_uri"]
    train_start = data_cfg["train_start"]
    train_end = data_cfg["train_end"]
    valid_start = data_cfg["valid_start"]
    valid_end = data_cfg["valid_end"]
    test_start = data_cfg["test_start"]
    test_end = data_cfg["test_end"]

    print(f"{'=' * 60}")
    print(f"Qlib 滚动训练 - {args.handler} (数据源: {args.data})")
    print(f"Provider: {provider_uri}")
    print(f"训练: {train_start} ~ {train_end}")
    print(f"验证: {valid_start} ~ {valid_end}")
    print(f"测试: {test_start} ~ {test_end}")
    print(f"滚动步长: {args.step} 天, 预测 horizon: {args.horizon} 天")
    print(f"{'=' * 60}")

    # 初始化 qlib
    init_kwargs = {}
    if args.sequential:
        init_kwargs["kernels"] = 1
        init_kwargs["joblib_backend"] = "sequential"
        print("[INFO] 已启用 sequential 模式 (kernels=1)")
    qlib.init(provider_uri=provider_uri, region=REGION, **init_kwargs)

    # 1. 构建 base task
    base_task = build_base_task(
        data_cfg, train_end, valid_start, valid_end, test_start, test_end,
        handler=args.handler, horizon=args.horizon,
    )

    # 2. 生成滚动任务
    task_list = task_generator(base_task, RollingGen(step=args.step, trunc_days=args.horizon + 1))
    # 滚动任务只做信号记录，不做分析（分析推迟到最终 ensemble 后）
    for t in task_list:
        t["record"] = [{"class": "SignalRecord", "module_path": "qlib.workflow.record_temp"}]

    print(f"\n生成 {len(task_list)} 个滚动任务")
    for i, t in enumerate(task_list):
        seg = t["dataset"]["kwargs"]["segments"]
        print(f"  任务 {i + 1}: train={seg['train'][0].strftime('%Y-%m-%d')}~{seg['train'][1].strftime('%Y-%m-%d')}"
              f"  test={seg['test'][0].strftime('%Y-%m-%d')}~{seg['test'][1].strftime('%Y-%m-%d')}")

    # 3. 训练滚动模型
    rolling_exp_name = f"rolling_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"\n开始训练滚动模型（实验: {rolling_exp_name}）...")
    try:
        R.delete_exp(experiment_name=rolling_exp_name)
    except ValueError:
        pass  # 无旧实验

    trainer = TrainerR(experiment_name=rolling_exp_name)
    trainer(task_list)
    print("滚动模型训练完成")

    # 4. Ensemble 滚动预测
    print("\n合并滚动预测结果...")
    rc = RecorderCollector(
        experiment=rolling_exp_name,
        artifacts_key=["pred", "label"],
        process_list=[RollingEnsemble()],
        artifacts_path={"pred": "pred.pkl", "label": "label.pkl"},
    )
    res = rc()

    # 5. 保存最终结果并分析
    final_exp_name = f"rolling_final_{args.data}_{args.handler}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with R.start(experiment_name=final_exp_name):
        R.log_params(step=args.step, horizon=args.horizon, handler=args.handler, data=args.data)
        R.save_objects(**{"pred.pkl": res["pred"], "label.pkl": res["label"]})

        recorder = R.get_recorder()

        # 信号分析
        try:
            sar = SigAnaRecord(recorder)
            sar.generate()
        except Exception as e:
            print(f"[警告] 信号分析失败: {e}")

    print(f"\n滚动训练完成！最终实验: {final_exp_name}")
    print(f"Recorder ID: {recorder.id}")

    # 6. 保存元数据
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rolling_info = {
        "experiment_name": final_exp_name,
        "rolling_experiment": rolling_exp_name,
        "model": "lgbm",
        "handler": args.handler,
        "data_source": args.data,
        "step": args.step,
        "horizon": args.horizon,
        "train_start": train_start,
        "test_end": test_end,
        "recorder_id": recorder.id,
        "num_rolling_tasks": len(task_list),
        "timestamp": datetime.now().isoformat(),
    }
    info_path = REPORTS_DIR / "rolling_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(rolling_info, f, ensure_ascii=False, indent=2)
    print(f"滚动训练信息已保存到: {info_path}")

    return final_exp_name, recorder.id


if __name__ == "__main__":
    main()
