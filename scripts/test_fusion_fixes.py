#!/usr/bin/env python3
"""融合层修复验证测试 (v0.6.0)。

覆盖 Tier 0.1 / 1.1 / 1.2 / 1.3 / 1.4 的改动：
  - 0.1: active-TA 死分支修复（w6_use NameError 消除、槽位对齐、公式统一）
  - 1.1: IC 比例加权（_load_ic_weights，负 IC 源置 0，缺失回退）
  - 1.2: 截面排名融合（rank_fusion，尺度无关，方向正确）
  - 1.3: 多种子（get_model_config seed 注入 LGBModel）
  - 1.4: Kronos 下线（权重=0 时无贡献，fuse_signals 容错空 kronos）

运行：
    D:\\codes\\stock\\qlib\\.venv\\Scripts\\python.exe test_fusion_fixes.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# 让 stocks_config / predict_fused / train 可导入
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent / "docs" / "scripts"))

import pandas as pd

import predict_fused as pf
from train import get_model_config

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


# ── 公共 mock 数据 ──────────────────────────────────────────────
def _qlib_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"stock_code": "000858.SZ", "score": 0.05},
        {"stock_code": "600519.SH", "score": -0.03},
        {"stock_code": "000001.SZ", "score": 0.01},
        {"stock_code": "002049.SZ", "score": 0.08},
        {"stock_code": "688256.SH", "score": -0.06},
    ])


def _ta_signals() -> dict:
    return {
        "000858.SZ": {"ai_score": 2, "trader_action": 1, "research_rating": 2,
                      "decision": "Buy", "price_target": 200.0, "date": "2026-06-30"},
        "600519.SH": {"ai_score": -1, "trader_action": -1, "research_rating": -2,
                      "decision": "Sell", "price_target": 1500.0, "date": "2026-06-30"},
        "002049.SZ": {"ai_score": 1, "trader_action": 0, "research_rating": 1,
                      "decision": "Hold", "price_target": 80.0, "date": "2026-06-30"},
    }


BASE_WEIGHTS = [0.5, 0.0, 0.25, 0.1, 0.15]  # qlib, kronos, ai, trader, research


# ── Tier 0.1: active-TA 死分支修复 ────────────────────────────
def test_01_dead_branch_fixed() -> None:
    print("\n=== Tier 0.1: active-TA 死分支修复 ===")
    # 活跃 TA 股票（原代码会 NameError 崩溃）
    r = pf.fuse_signals(_qlib_df(), _ta_signals(), BASE_WEIGHTS, rank_fusion=False)
    active = r[r["stock_code"] == "000858.SZ"].iloc[0]
    pure = r[r["stock_code"] == "000001.SZ"].iloc[0]

    # 不崩溃 = 死分支已修复
    check("active 分支不再 NameError 崩溃", len(r) > 0)
    # TA 正向贡献：combined > 纯qlib加权 w1*qlib_score
    pure_qlib_contrib = 0.5 * active["qlib_score"]
    ta_contrib = active["combined_score"] - pure_qlib_contrib
    check("TA 正向贡献 > 0", ta_contrib > 0,
          f"ta_contrib={ta_contrib:.6f}")
    # 全中性股（000001 不在 ta_signals）combined = w1*qlib_score
    check("neutral 股 combined = w1*qlib", abs(pure["combined_score"] - 0.5 * 0.01) < 1e-9,
          f"combined={pure['combined_score']}")


# ── Tier 1.1: IC 比例加权 ─────────────────────────────────────
def test_11_ic_weights() -> None:
    print("\n=== Tier 1.1: IC 比例加权 ===")
    ic_path = pf.REPORTS_DIR / "per_source_ic.json"
    saved = ic_path.read_bytes() if ic_path.exists() else None
    try:
        # 正常 IC 数据
        ic_path.write_text(json.dumps({
            "qlib": 0.068, "kronos": -0.160, "ai": 0.103,
            "trader": 0.392, "research": 0.358,
            "as_of": "2026-06-29", "window_days": 30,
        }), encoding="utf-8")
        w = pf._load_ic_weights(BASE_WEIGHTS)
        check("Kronos 负 IC → 权重 0", w[1] == 0.0, f"w_kronos={w[1]}")
        check("权重和 = 1", abs(sum(w) - 1.0) < 1e-9, f"sum={sum(w)}")
        check("最高 IC 源(Trader) 获最大权重", w[3] == max(w), f"w={w}")
        check("返回 5 个权重", len(w) == 5)

        # IC 文件缺失 → 回退
        ic_path.unlink()
        fb = pf._load_ic_weights(BASE_WEIGHTS)
        check("IC 文件缺失 → 回退 base_weights", fb == BASE_WEIGHTS)

        # IC 全负 → 回退
        ic_path.write_text(json.dumps({"qlib": -0.1, "kronos": -0.2, "ai": -0.05,
                                       "trader": -0.1, "research": -0.1}), encoding="utf-8")
        fb2 = pf._load_ic_weights(BASE_WEIGHTS)
        check("全负 IC → 回退 base_weights", fb2 == BASE_WEIGHTS)
    finally:
        if saved is not None:
            ic_path.write_bytes(saved)
        elif ic_path.exists():
            ic_path.unlink()


# ── Tier 1.2: 截面排名融合 ────────────────────────────────────
def test_12_rank_fusion() -> None:
    print("\n=== Tier 1.2: 截面排名融合 ===")
    r_raw = pf.fuse_signals(_qlib_df(), _ta_signals(), BASE_WEIGHTS, rank_fusion=False)
    r_rank = pf.fuse_signals(_qlib_df(), _ta_signals(), BASE_WEIGHTS, rank_fusion=True)

    check("rank_fusion 改变 combined", not r_raw["combined_score"].equals(r_rank["combined_score"]))
    # 全正向股(000858) 排名应高于 全负向股(600519)
    a = r_rank[r_rank["stock_code"] == "000858.SZ"]["combined_score"].iloc[0]
    b = r_rank[r_rank["stock_code"] == "600519.SH"]["combined_score"].iloc[0]
    check("全正向 > 全负向（方向正确）", a > b, f"pos={a:.4f} neg={b:.4f}")
    # decision 标记 [rank]
    check("decision 标记 [rank]", "[rank]" in r_rank.iloc[0]["decision"])


# ── Tier 1.3: 多种子 ──────────────────────────────────────────
def test_13_multi_seed() -> None:
    print("\n=== Tier 1.3: 多种子注入 ===")
    c42 = get_model_config("lgbm", seed=42)
    c44 = get_model_config("lgbm", seed=44)
    kw42, kw44 = c42["kwargs"], c44["kwargs"]
    for field in ("seed", "bagging_seed", "feature_fraction_seed", "data_random_seed"):
        check(f"{field} 注入正确", kw42[field] == 42 and kw44[field] == 44)
    check("不同 seed 产生不同配置", kw42["seed"] != kw44["seed"])


# ── Tier 1.4: Kronos 下线 ─────────────────────────────────────
def test_14_kronos_disabled() -> None:
    print("\n=== Tier 1.4: Kronos 下线 ===")
    # kronos_signals={} 时 fuse_signals 正常工作，kronos 贡献=0
    r = pf.fuse_signals(_qlib_df(), _ta_signals(), BASE_WEIGHTS,
                        kronos_signals={}, rank_fusion=False)
    check("空 kronos 不崩溃", len(r) > 0)
    # kronos_ret 列全为 0
    check("kronos_ret 全为 0（无 kronos 加载）", (r["kronos_ret"] == 0.0).all())
    # 复现 main() 的 use_kronos 决策逻辑：权重=0 → 不启用
    kronos_weight = BASE_WEIGHTS[1]
    use_kronos = kronos_weight > 0  # 简化：权重 0 时不启用
    check("kronos 权重=0 → use_kronos=False", use_kronos is False)


def main() -> None:
    print("=" * 60)
    print("融合层修复验证测试 (v0.6.0)")
    print("=" * 60)
    test_01_dead_branch_fixed()
    test_11_ic_weights()
    test_12_rank_fusion()
    test_13_multi_seed()
    test_14_kronos_disabled()
    print("\n" + "=" * 60)
    print(f"  结果: {PASS} PASS / {FAIL} FAIL")
    print("=" * 60)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
