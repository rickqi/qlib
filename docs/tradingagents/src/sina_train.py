"""
Step 3+4: Train LightGBM + Predict prices
Must be run as __main__ (not exec) to avoid Windows multiprocessing spawn issues
"""
import sys
import os

# Must be first — before any qlib import triggers multiprocessing
sys.path.insert(0, r"D:\codes\qlib")

import pickle
import numpy as np
import pandas as pd
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.data.dataset.handler import DataHandlerLP
import qlib
from qlib.config import C

if __name__ == "__main__":
    # Load metadata
    meta_path = r"C:\Users\szk220009\AppData\Local\Temp\opencode\sina_meta.pkl"
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    prices = meta["prices"]
    last_date = meta["last_date"]
    next_td = meta["next_td"]

    # Disable proxy
    for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
        os.environ[k] = ""

    print("=" * 60)
    print("Step 3: Train + Predict")
    print("=" * 60)

    qlib.init(provider_uri="C:/Users/szk220009/.qlib/qlib_data/tradingagents", region=REG_CN)

    # Override joblib to use threading — avoids Windows multiprocessing spawn issues
    # and shape broadcast errors with sparse stock data
    C.joblib_backend = "threading"
    C.kernels = 1

    # --- Model A: CSRankNorm ---
    print("\nTraining Model A (CSRankNorm) ...")
    ds_a = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "start_time": "2020-01-01",
                    "end_time": "2026-05-08",
                    "fit_start_time": "2020-01-01",
                    "fit_end_time": "2023-12-31",
                    "instruments": "all",
                    "infer_processors": [
                        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
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
                "train": ["2020-01-01", "2023-12-31"],
                "valid": ["2024-01-01", "2024-12-31"],
                "test": ["2025-01-01", "2026-04-30"],
            },
        },
    }
    da = init_instance_by_config(ds_a)
    ma = init_instance_by_config({
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "learning_rate": 0.1,
            "max_depth": 8,
            "num_leaves": 128,
            "num_threads": 1,
            "early_stopping_rounds": 50,
            "n_estimators": 300,
            "colsample_bytree": 0.8,
            "subsample": 0.8,
        },
    })
    ma.fit(da)
    pa = ma.predict(da, segment="test")
    if isinstance(pa, pd.DataFrame):
        pa = pa.iloc[:, 0]
    la = da.prepare("test", col_set="label", data_key=DataHandlerLP.DK_R)
    if isinstance(la, pd.DataFrame):
        la = la.iloc[:, 0]
    cm = pa.index.intersection(la.index)
    ic = pa.loc[cm].corr(la.loc[cm], method="pearson")
    ric = pa.loc[cm].corr(la.loc[cm], method="spearman")
    print(f"  IC={ic:.4f}, Rank IC={ric:.4f}")
    cs_last = pa.loc[pa.index.get_level_values(0).max()]
    cs_pred = {k: float(v) for k, v in cs_last.items()}

    # --- Model B: Raw Return ---
    print("\nTraining Model B (Raw Return) ...")
    ds_b = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "start_time": "2020-01-01",
                    "end_time": "2026-05-08",
                    "fit_start_time": "2020-01-01",
                    "fit_end_time": "2023-12-31",
                    "instruments": "all",
                    "infer_processors": [
                        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                    ],
                    "learn_processors": [{"class": "DropnaLabel"}],
                    "label": ["Ref($close, -2) / Ref($close, -1) - 1"],
                },
            },
            "segments": {
                "train": ["2020-01-01", "2025-04-30"],
                "valid": ["2025-05-01", "2026-03-31"],
                "test": ["2026-04-01", "2026-05-08"],
            },
        },
    }
    db = init_instance_by_config(ds_b)
    mb = init_instance_by_config({
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "learning_rate": 0.05,
            "max_depth": 6,
            "num_leaves": 80,
            "num_threads": 1,
            "early_stopping_rounds": 100,
            "n_estimators": 500,
        },
    })
    mb.fit(db)
    pb = mb.predict(db, segment="test")
    if isinstance(pb, pd.DataFrame):
        pb = pb.iloc[:, 0]
    ld = pb.index.get_level_values(0).max()
    raw_pred = {k: float(v) for k, v in pb.loc[ld].items()}
    print(f"  Predictions: {ld}, {len(raw_pred)} stocks")

    # ============================================================
    # Output
    # ============================================================
    def to_wind_code(qc):
        """SH688041 → 688041.SH"""
        return f"{qc[2:]}.{qc[:2]}"

    # Output ALL predicted stocks (from raw_pred / cs_pred)
    all_qlib_codes = sorted(set(raw_pred.keys()) | set(cs_pred.keys()))
    print(f"\n  Total predicted stocks: {len(all_qlib_codes)}")

    rows = []
    for qc in all_qlib_codes:
        code = to_wind_code(qc)
        cur = prices.get(code, np.nan)
        ret = raw_pred.get(qc, np.nan)
        cs = cs_pred.get(qc, np.nan)
        pp = cur * (1 + ret) if not np.isnan(cur) and not np.isnan(ret) else np.nan
        chg = ret * 100 if not np.isnan(ret) else np.nan
        rows.append({
            "code": code,
            "current_close": round(cur, 2) if not np.isnan(cur) else None,
            "csrank_score": round(cs, 4) if not np.isnan(cs) else None,
            "pred_return": round(ret, 6) if not np.isnan(ret) else None,
            "pred_change_pct": round(chg, 2) if not np.isnan(chg) else None,
            "pred_price": round(pp, 2) if not np.isnan(pp) else None,
            "limit_up": round(cur * 1.1, 2) if not np.isnan(cur) else None,
            "limit_down": round(cur * 0.9, 2) if not np.isnan(cur) else None,
        })

    df = pd.DataFrame(rows).sort_values("pred_change_pct", ascending=False, na_position="last")

    print()
    print("=" * 100)
    print(f"  Price Prediction — Data through: {last_date} -> Predict: {next_td}")
    print(f"  Source: Sina Finance daily | IC={ic:.4f}  Rank IC={ric:.4f} | {meta['n_stocks']} stocks")
    print("=" * 100)
    print()
    print(f"  {'#':>4s}  {'Code':<12s}  {'Close':>10s}  {'Rank':>8s}  {'Change':>10s}  {'PredPrice':>10s}  {'LimitUp':>10s}  {'LimitDn':>10s}")
    print("  " + "-" * 85)

    # Print top 20 + bottom 5 (too many to print all 1999)
    n_valid = df["pred_change_pct"].notna().sum()
    show_top = min(20, len(df))
    show_bot = min(5, max(0, len(df) - show_top))
    for i, (idx, r) in enumerate(df.iterrows()):
        if i >= show_top and i < len(df) - show_bot:
            if i == show_top:
                print(f"  ... ({len(df) - show_top - show_bot} more stocks, see CSV for full list)")
            continue
        cc = f"{r['current_close']:.2f}" if r["current_close"] else "N/A"
        cs = f"{r['csrank_score']:.4f}" if r["csrank_score"] is not None else "N/A"
        chg = f"{r['pred_change_pct']:+.2f}%" if r["pred_change_pct"] is not None else "N/A"
        pp = f"{r['pred_price']:.2f}" if r["pred_price"] else "N/A"
        lu = f"{r['limit_up']:.2f}" if r["limit_up"] else "N/A"
        ld = f"{r['limit_down']:.2f}" if r["limit_down"] else "N/A"
        print(f"  {i+1:>4d}  {r['code']:<12s}  {cc:>10s}  {cs:>8s}  {chg:>10s}  {pp:>10s}  {lu:>10s}  {ld:>10s}")

    out_dir = r"D:\codes\qlib\docs\tradingagents\output"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"price_pred_{next_td}_sina.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  Saved: {out_path}")
    print("=" * 100)
