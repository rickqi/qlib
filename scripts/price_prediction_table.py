"""Generate 20-stock × 20-day actual price prediction table with trading dates."""
import csv
import os
import math

# ── Trading dates (from qlib day_future.txt) ──
TRADING_DATES = [
    "2026-05-11", "2026-05-12", "2026-05-13", "2026-05-14", "2026-05-15",
    "2026-05-18", "2026-05-19", "2026-05-20", "2026-05-21", "2026-05-22",
    "2026-05-25", "2026-05-26", "2026-05-27", "2026-05-28", "2026-05-29",
    "2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05",
]

FUSED_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "reports",
    "fused_prediction_20260511_161318.csv"
)
QLIB_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "reports",
    "qlib_only_prediction_20260511_161331.csv"
)


def read_prices(path):
    """Read prediction CSV, return {ticker: {base, name, pred: [D1..D20], ai, ta, rm, score}}"""
    stocks = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            ticker = row["stock_code"]
            preds = []
            for d in range(1, 21):
                val = float(row[f"pred_D{d}"])
                # Cap at A-share limit: ±10% per day from previous day
                if preds:
                    prev = preds[-1]
                else:
                    prev = float(row["base_close"])
                upper = prev * 1.10
                lower = prev * 0.90
                if val > upper:
                    val = upper
                elif val < lower:
                    val = lower
                preds.append(round(val, 2))
            stocks[ticker] = {
                "name": row["name"],
                "base": float(row["base_close"]),
                "preds": preds,
                "ai": int(row.get("ai_score", 0)),
                "ta": int(row.get("trader_action", 0)),
                "rm": int(row.get("research_rating", 0)),
                "score": float(row.get("combined_score", row.get("qlib_score", 0))),
                "decision": row.get("decision", ""),
            }
    return stocks


def main():
    fused = read_prices(FUSED_CSV)
    qlib = read_prices(QLIB_CSV)

    # Sort by D20 change (descending)
    order = sorted(fused.keys(), key=lambda t: fused[t]["preds"][-1] / fused[t]["base"] - 1, reverse=True)

    # ── Table 1: Fused price predictions ──
    print("=" * 200)
    print("  20只股票 × 20个交易日  融合预测价格（Qlib量化 + TradingAgents AI信号）")
    print(f"  基准价: 2026-05-08 收盘价  |  预测区间: {TRADING_DATES[0]} ~ {TRADING_DATES[-1]}")
    print(f"  融合权重: Qlib=0.50, AI=0.25, Trader=0.10, Research=0.15")
    print("=" * 200)

    # Header
    hdr = f"{'#':>2} {'代码':<12} {'名称':<8} {'基准':>8} │"
    for i, d in enumerate(TRADING_DATES):
        short = d[5:]  # MM-DD
        hdr += f" {short:>7}│"
    hdr += f" {'20日涨幅':>8} {'信号':>5}"
    print(hdr)
    print("─" * len(hdr))

    for rank, ticker in enumerate(order, 1):
        s = fused[ticker]
        base = s["base"]
        d20 = s["preds"][-1]
        chg = (d20 / base - 1) * 100

        line = f"{rank:2d} {ticker:<12} {s['name']:<8} {base:>8.2f} │"
        for p in s["preds"]:
            line += f" {p:>7.2f}│"
        sig = f"a{s['ai']:+d}t{s['ta']:+d}r{s['rm']:+d}"
        line += f" {chg:>+7.1f}% {sig:>8}"
        print(line)

    print()

    # ── Table 2: Qlib-only vs Fused comparison ──
    print("=" * 120)
    print("  对比: 纯Qlib预测 vs 融合预测 (D20 终价)")
    print("=" * 120)
    print(f"{'代码':<12} {'名称':<8} {'基准':>8} │ {'Qlib D20':>9} {'涨幅':>8} │ {'融合 D20':>9} {'涨幅':>8} │ {'Δ(pp)':>7} │ {'AI信号':>8}")
    print("─" * 120)

    for ticker in order:
        f = fused[ticker]
        q = qlib.get(ticker)
        base = f["base"]
        d20_q = q["preds"][-1] if q else base
        d20_f = f["preds"][-1]
        chg_q = (d20_q / base - 1) * 100
        chg_f = (d20_f / base - 1) * 100
        delta = chg_f - chg_q
        sig = f"a{f['ai']:+d}t{f['ta']:+d}r{f['rm']:+d}"
        print(f"{ticker:<12} {f['name']:<8} {base:>8.2f} │ {d20_q:>9.2f} {chg_q:>+7.1f}% │ {d20_f:>9.2f} {chg_f:>+7.1f}% │ {delta:>+6.1f} │ {sig:>8}")

    # ── Table 3: Key dates summary (D1, D5, D10, D20) ──
    print()
    print("=" * 100)
    print("  关键节点价格预测汇总")
    print("=" * 100)
    key_days = [0, 4, 9, 19]  # D1, D5, D10, D20
    key_labels = ["D1 (5/11)", "D5 (5/15)", "D10 (5/22)", "D20 (6/05)"]
    print(f"{'代码':<12} {'名称':<8} {'基准价':>8} │", end="")
    for label in key_labels:
        print(f" {label:>10}│", end="")
    print(f" {'评级':<14}")
    print("─" * 100)

    for ticker in order:
        s = fused[ticker]
        line = f"{ticker:<12} {s['name']:<8} {s['base']:>8.2f} │"
        for ki in key_days:
            p = s["preds"][ki]
            chg = (p / s["base"] - 1) * 100
            line += f" {p:>7.2f} {chg:>+4.1f}%│"
        line += f" {s['decision']:<14}"
        print(line)

    # ── Save as CSV ──
    out_csv = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "reports",
        "price_prediction_20stocks_20days.csv"
    )
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["代码", "名称", "基准价(5/8)"] +
                         [f"{d}" for d in TRADING_DATES] +
                         ["20日涨幅%", "AI信号", "Trader", "Research", "评级", "融合分"])
        for ticker in order:
            s = fused[ticker]
            d20 = s["preds"][-1]
            chg = round((d20 / s["base"] - 1) * 100, 2)
            writer.writerow(
                [ticker, s["name"], s["base"]] +
                s["preds"] +
                [chg, s["ai"], s["ta"], s["rm"], s["decision"], round(s["score"], 4)]
            )
    print(f"\nCSV 已保存: {out_csv}")


if __name__ == "__main__":
    main()
