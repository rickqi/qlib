"""Compare Qlib-only vs Fused predictions for 20 stocks."""
import csv, os, sys

REPORTS_DIR = r"D:\codes\stock\qlib\reports"


def read_pred(filename):
    path = os.path.join(REPORTS_DIR, filename)
    data = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            data[row["stock_code"]] = row
    return data


def main():
    qlib = read_pred("qlib_only_prediction_20260511_161331.csv")
    fused = read_pred("fused_prediction_20260511_161318.csv")

    results = []
    for t in sorted(fused.keys()):
        f = fused[t]
        q = qlib.get(t, {})
        base = float(f["base_close"])
        d20_q = float(q["pred_D20"]) if q else base
        d20_f = float(f["pred_D20"])
        chg_q = (d20_q / base - 1) * 100
        chg_f = (d20_f / base - 1) * 100
        delta = chg_f - chg_q
        ai = int(f.get("ai_score", 0))
        ta_v = int(f.get("trader_action", 0))
        rm = int(f.get("research_rating", 0))
        d1_q = float(q["pred_D1"]) if q else base
        d1_f = float(f["pred_D1"])
        results.append({
            "ticker": t, "name": f.get("name", ""), "base": base,
            "d1_q": d1_q, "d20_q": d20_q, "chg_q": chg_q,
            "d1_f": d1_f, "d20_f": d20_f, "chg_f": chg_f,
            "delta": delta, "ai": ai, "ta": ta_v, "rm": rm,
        })

    results.sort(key=lambda x: x["chg_f"], reverse=True)

    print("=" * 140)
    print("  Qlib-only vs Fused Prediction Comparison (20 stocks, 20 trading days)")
    print("=" * 140)
    print()
    hdr = f"{'#':>2} {'Ticker':<12} {'Name':<8} {'Base':>8} | {'D20_qlib':>9} {'Chg%':>8} | {'D20_fused':>9} {'Chg%':>8} | {'Delta':>8} | AI TA RM"
    print(hdr)
    print("-" * 140)

    for i, r in enumerate(results, 1):
        sign_d = "+" if r["delta"] >= 0 else ""
        print(
            f"{i:2d} {r['ticker']:<12} {r['name']:<8} {r['base']:>8.1f} | "
            f"{r['d20_q']:>9.1f} {r['chg_q']:>+7.1f}% | "
            f"{r['d20_f']:>9.1f} {r['chg_f']:>+7.1f}% | "
            f"{sign_d}{r['delta']:>+7.1f}pp | "
            f"{r['ai']:>+2d} {r['ta']:>+2d} {r['rm']:>+2d}"
        )

    # Summary
    print()
    print("-" * 140)
    bullish_q = sum(1 for r in results if r["chg_q"] > 0)
    bullish_f = sum(1 for r in results if r["chg_f"] > 0)
    avg_delta = sum(r["delta"] for r in results) / len(results)
    max_temper = max(results, key=lambda x: abs(x["delta"]))
    print(f"  Summary:")
    print(f"    Qlib-only bullish: {bullish_q}/20 | Fused bullish: {bullish_f}/20")
    print(f"    Avg delta (fused - qlib): {avg_delta:+.1f}pp")
    print(f"    Max tempered: {max_temper['ticker']} ({max_temper['name']}) delta={max_temper['delta']:+.1f}pp")

    # Direction changes
    print()
    print("  Direction changes (Qlib -> Fused):")
    for r in results:
        if (r["chg_q"] > 0) != (r["chg_f"] > 0):
            print(f"    {r['ticker']} {r['name']}: {r['chg_q']:+.1f}% -> {r['chg_f']:+.1f}%  (flipped!)")

    # Top AI impact
    print()
    print("  Top 5 AI impact (absolute delta):")
    by_impact = sorted(results, key=lambda x: abs(x["delta"]), reverse=True)
    for r in by_impact[:5]:
        print(f"    {r['ticker']} {r['name']}: {r['chg_q']:+.1f}% -> {r['chg_f']:+.1f}%  (delta={r['delta']:+.1f}pp, ai={r['ai']} ta={r['ta']} rm={r['rm']})")


if __name__ == "__main__":
    main()
