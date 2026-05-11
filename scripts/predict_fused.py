"""
方案 B 融合预测：Qlib 量化分数 + TradingAgents AI 信号 → 20 天价格预测。

用法:
    # 基本用法（从已有 Qlib 预测 + TA 日志融合）
    .venv\\Scripts\\python.exe scripts\\predict_fused.py

    # 指定 Qlib 预测文件
    .venv\\Scripts\\python.exe scripts\\predict_fused.py --qlib-pred reports/predictions_20260511_122419.csv

    # 自定义融合权重和预测天数
    .venv\\Scripts\\python.exe scripts\\predict_fused.py --weights 0.50 0.25 0.10 0.15 --days 20

    # 仅 Qlib 信号（跳过 TradingAgents，用于对比基线）
    .venv\\Scripts\\python.exe scripts\\predict_fused.py --qlib-only

信号融合公式:
    # AI 信号映射到 Qlib score 相同量级 (score_std)
    ai_mapped  = (ai_score / 2) × score_std
    ta_mapped  = trader_action × score_std × 0.5
    rr_mapped  = (research_rating / 2) × score_std

    combined = w1 × qlib_score + w2 × ai_mapped + w3 × ta_mapped + w4 × rr_mapped

    默认权重: w1=0.50, w2=0.25, w3=0.10, w4=0.15
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# 清除代理（tushare 不需要代理）
for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    os.environ.pop(key, None)

import numpy as np
import pandas as pd
import tushare as ts

# ── 配置 ──────────────────────────────────────────────────────
TUSHARE_TOKEN = '260264d1c42c2b5c47262478557e99d7f6a0769523ea19f48e09ed73'
REPORTS_DIR = Path(__file__).parent.parent / 'reports'
DOCS_DIR = Path(__file__).parent.parent / 'docs'

STOCKS = [
    "688041.SH", "688256.SH", "688012.SH", "603986.SH", "688008.SH",
    "300442.SZ", "603019.SH", "688111.SH", "002230.SZ", "002837.SZ",
    "002049.SZ", "688027.SH", "300223.SZ", "301269.SZ", "002747.SZ",
    "688332.SH", "002896.SZ", "688568.SH", "300672.SZ", "300458.SZ",
]

STOCK_NAMES = {
    "688041.SH": "海光信息", "688256.SH": "寒武纪", "688012.SH": "中微公司",
    "603986.SH": "兆易创新", "688008.SH": "澜起科技", "300442.SZ": "普丽盛",
    "603019.SH": "中科曙光", "688111.SH": "金山办公", "002230.SZ": "科大讯飞",
    "002837.SZ": "英维克", "002049.SZ": "紫光国微", "688027.SH": "天合光能",
    "300223.SZ": "北京君正", "301269.SZ": "联特科技", "002747.SZ": "尚太科技",
    "688332.SH": "联影医疗", "002896.SZ": "星帅尔", "688568.SH": "中科星图",
    "300672.SZ": "国科微", "300458.SZ": "全志科技",
}

# A股涨跌停限制
DAILY_LIMIT = 0.10  # ±10%

# 20 天衰减因子（线性衰减到 12%）
DEFAULT_DAYS = 20


def get_decay_factors(days: int) -> list[float]:
    """生成 days 天的衰减因子列表。

    策略：从 1.0 线性衰减到 0.12，确保 20 天窗口末尾仍有微弱信号。
    """
    end = 0.12
    return [1.0 - (1.0 - end) * i / (days - 1) for i in range(days)]


# ── 数据加载 ──────────────────────────────────────────────────

def load_qlib_predictions(pred_path: str | None) -> pd.DataFrame:
    """加载 Qlib 预测分数 CSV。

    如果未指定路径，自动查找 reports/ 下最新的 predictions_*.csv。
    """
    if pred_path:
        path = Path(pred_path)
    else:
        pred_files = sorted(REPORTS_DIR.glob('predictions_*.csv'))
        if not pred_files:
            print('错误: 未找到 Qlib 预测文件。请先运行 predict.py')
            sys.exit(1)
        path = pred_files[-1]

    assert isinstance(path, Path)

    print(f'加载 Qlib 预测: {path.name}')
    df = pd.read_csv(path, encoding='utf-8-sig')

    # 确保有 stock_code 和 score 列
    if 'stock_code' not in df.columns and 'instrument' in df.columns:
        # 从 qlib 格式反向转换
        df['stock_code'] = df['instrument'].apply(
            lambda x: f"{str(x)[2:]}.{str(x)[:2]}" if isinstance(x, str) and len(x) == 8 else str(x)
        )

    # 如果是 latest-day 模式，可能只有一天的数据
    return df


def load_ta_signals(signals_dir: str | None = None) -> dict[str, dict]:
    """从 TradingAgents JSON 日志加载 AI 信号。

    使用内嵌的信号提取逻辑，不依赖 tradingagents 包。

    返回: {ticker: {"ai_score": int, "trader_action": int, "research_rating": int,
                     "price_target": float, "decision": str}}
    """
    if signals_dir is None:
        signals_dir = os.path.expanduser("~/.tradingagents/logs")

    signals_dir_path = Path(signals_dir)
    if not signals_dir_path.exists():
        print(f'[WARN] TradingAgents 日志目录不存在: {signals_dir}')
        return {}

    signals = {}

    # ── 信号映射常量（内嵌，不依赖 tradingagents 包）──
    rating_map = {
        "Buy": 2, "Overweight": 1, "Hold": 0, "Underweight": -1, "Sell": -2,
    }
    action_map = {"Buy": 1, "Hold": 0, "Sell": -1}

    # ── 提取正则（与 signal_extractor.py 一致）──
    import re
    re_trader_action = re.compile(
        r"FINAL TRANSACTION PROPOSAL:\s*\*\*(BUY|HOLD|SELL)\*\*", re.IGNORECASE
    )
    re_research_rec = re.compile(
        r"\*\*Recommendation\*\*:\s*(Buy|Overweight|Hold|Underweight|Sell)", re.IGNORECASE
    )
    re_price_target = re.compile(r"\*\*Price Target\*\*:\s*([\d.]+)", re.IGNORECASE)
    re_rating_label = re.compile(r"rating.*?[:\-][\s*]*(\w+)", re.IGNORECASE)
    rating_set = {r.lower() for r in rating_map}

    def parse_rating(text: str, default: str = "Hold") -> str:
        """从文本中提取 5 级评级。"""
        for line in text.splitlines():
            m = re_rating_label.search(line)
            if m and m.group(1).lower() in rating_set:
                return m.group(1).capitalize()
        for line in text.splitlines():
            for word in line.lower().split():
                clean = word.strip("*:.,")
                if clean in rating_set:
                    return clean.capitalize()
        return default

    # ── 遍历日志文件 ──
    pattern = "**/TradingAgentsStrategy_logs/full_states_log_*.json"
    for log_file in sorted(signals_dir_path.glob(pattern)):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            ticker = data.get('company_of_interest', '')
            if not ticker:
                continue

            pm_text = data.get('final_trade_decision', '')
            trader_text = data.get('trader_investment_decision',
                                  data.get('trader_investment_plan', ''))
            research_text = data.get('investment_plan', '')

            # 提取评级
            rating = parse_rating(pm_text)
            ai_score = rating_map.get(rating, 0)

            # 提取 trader action
            action = "Hold"
            if trader_text:
                m = re_trader_action.search(trader_text)
                if m:
                    action = m.group(1).capitalize()
            trader_action = action_map.get(action, 0)

            # 提取 research rating
            research_rating = 0
            if research_text:
                m = re_research_rec.search(research_text)
                if m:
                    rec = m.group(1).capitalize()
                    research_rating = rating_map.get(rec, 0)
                else:
                    research_rating = rating_map.get(parse_rating(research_text), 0)

            # 提取 price target
            price_target = float('nan')
            if pm_text:
                m = re_price_target.search(pm_text)
                if m:
                    try:
                        price_target = float(m.group(1))
                    except ValueError:
                        pass

            signals[ticker] = {
                'ai_score': ai_score,
                'trader_action': trader_action,
                'research_rating': research_rating,
                'price_target': price_target,
                'decision': rating,
                'date': data.get('trade_date', ''),
            }
        except Exception as e:
            print(f'  [WARN] 跳过 {log_file.name}: {e}')
            continue

    print(f'加载 TradingAgents 信号: {len(signals)} 只股票')
    return signals


def get_actual_prices(date: str, stocks: list[str]) -> dict[str, float]:
    """从 tushare 获取指定日期的实际收盘价（不复权）。"""
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()

    date_fmt = date.replace('-', '')
    price_map = {}

    # tushare 一次最多查 100 只，分批
    for i in range(0, len(stocks), 100):
        batch = stocks[i:i+100]
        ts_codes = ','.join(batch)
        df = pro.daily(ts_code=ts_codes, start_date=date_fmt, end_date=date_fmt)
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                price_map[row['ts_code']] = row['close']

    print(f'获取 {date} 收盘价: {len(price_map)}/{len(stocks)} 只')
    return price_map


def get_trading_days(base_date: str, n_days: int) -> list[str]:
    """获取 base_date 之后的 n_days 个交易日。

    优先从 qlib_bin 的 day_future.txt 获取，回退到简单排除法。
    """
    future_cal = Path("C:/codes/qlib/qlib_bin/calendars/day_future.txt")
    if future_cal.exists():
        with open(future_cal, 'r') as f:
            dates = [line.strip() for line in f if line.strip()]
        try:
            idx = dates.index(base_date)
            return dates[idx+1:idx+1+n_days]
        except ValueError:
            pass

    # 回退：简单日历排除（忽略法定假日）
    from datetime import timedelta
    result = []
    current = datetime.strptime(base_date, '%Y-%m-%d')
    while len(result) < n_days:
        current += timedelta(days=1)
        if current.weekday() < 5:  # 周一~周五
            result.append(current.strftime('%Y-%m-%d'))
    return result


# ── 融合逻辑 ──────────────────────────────────────────────────

def fuse_signals(
    qlib_df: pd.DataFrame,
    ta_signals: dict[str, dict],
    weights: list[float],
    qlib_only: bool = False,
) -> pd.DataFrame:
    """将 Qlib 预测分数与 TradingAgents AI 信号融合。

    融合策略:
    - Qlib score 保持原始量级（通常在 [-0.2, +0.1] 之间），作为基础日收益率预测
    - AI 信号转换为等价收益率修正量，按权重叠加
    - combined_score 的含义是 "预测日收益率"，直接用于价格预测

    Args:
        qlib_df: Qlib 预测结果 DataFrame，需含 stock_code, score 列
        ta_signals: {ticker: {ai_score, trader_action, research_rating, ...}}
        weights: [w1, w2, w3, w4] 融合权重
        qlib_only: True 时仅使用 Qlib 信号（用于对比基线）

    Returns:
        DataFrame 含 qlib_score, ai_score, trader_action, research_rating, combined_score
    """
    w1, w2, w3, w4 = weights

    # 计算历史 Qlib score 的标准差，用于将 AI 信号映射到相同量级
    score_std = qlib_df['score'].std()
    if score_std == 0 or np.isnan(score_std):
        score_std = 0.05  # 兜底值
    print(f'Qlib score std: {score_std:.4f} (用于 AI 信号映射)')

    rows = []
    for _, row in qlib_df.iterrows():
        stock = row['stock_code']
        qlib_score = row['score']

        if qlib_only or stock not in ta_signals:
            # 纯 Qlib 模式或无 TA 信号
            combined = qlib_score
            ai_sc = 0
            ta = 0
            rr = 0
            decision = 'N/A'
        else:
            sig = ta_signals[stock]
            ai_sc = sig['ai_score']
            ta = sig['trader_action']
            rr = sig['research_rating']
            decision = sig['decision']

            # AI 信号映射到 Qlib score 的量级
            # ai_score ∈ [-2, 2] → 映射到 [-score_std, +score_std]
            ai_mapped = ai_sc / 2.0 * score_std
            # trader_action ∈ [-1, 1] → 映射到 [-score_std/2, +score_std/2]
            ta_mapped = ta / 1.0 * score_std * 0.5
            # research_rating ∈ [-2, 2] → 映射到 [-score_std, +score_std]
            rr_mapped = rr / 2.0 * score_std

            combined = w1 * qlib_score + w2 * ai_mapped + w3 * ta_mapped + w4 * rr_mapped

        rows.append({
            'stock_code': stock,
            'name': STOCK_NAMES.get(stock, ''),
            'qlib_score': round(qlib_score, 6),
            'ai_score': ai_sc,
            'trader_action': ta,
            'research_rating': rr,
            'combined_score': round(combined, 6),
            'decision': decision if not qlib_only else 'Qlib-only',
        })

    return pd.DataFrame(rows)


def predict_prices(fused_df: pd.DataFrame, price_map: dict[str, float],
                   days: int, decay: list[float], base_date: str = '2026-05-08') -> pd.DataFrame:
    """将融合分数转换为 N 天价格预测。

    Args:
        fused_df: fuse_signals() 输出
        price_map: {ticker: base_close_price}
        days: 预测天数
        decay: 衰减因子列表 (长度=days)

    Returns:
        DataFrame 含 pred_D1~DN, ret_D1~DN 列
    """
    trading_days = get_trading_days(base_date, days)

    results = []
    for _, row in fused_df.iterrows():
        stock = row['stock_code']
        combined = row['combined_score']
        base_price = price_map.get(stock)

        if base_price is None:
            print(f'  跳过 {stock}: 无基准价格')
            continue

        result = {
            'stock_code': stock,
            'name': row['name'],
            'base_close': round(base_price, 2),
            'qlib_score': row['qlib_score'],
            'ai_score': row['ai_score'],
            'trader_action': row['trader_action'],
            'research_rating': row['research_rating'],
            'combined_score': row['combined_score'],
            'decision': row['decision'],
        }

        prev_price = base_price
        for i in range(days):
            daily_return = combined * decay[i]
            daily_return = max(-DAILY_LIMIT, min(DAILY_LIMIT, daily_return))
            pred_price = round(prev_price * (1 + daily_return), 2)
            cum_return = round((pred_price / base_price - 1) * 100, 2)

            result[f'pred_D{i+1}'] = pred_price
            result[f'ret_D{i+1}'] = cum_return

            prev_price = pred_price

        results.append(result)

    return pd.DataFrame(results)


# ── 报告生成 ──────────────────────────────────────────────────

def generate_report(result_df: pd.DataFrame, days: int, weights: list[float],
                    base_date: str, qlib_only: bool) -> tuple[Path, Path]:
    """生成融合预测 CSV 和 Markdown 报告。"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    decay = get_decay_factors(days)
    mode = 'qlib_only' if qlib_only else 'fused'

    # CSV
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / f'{mode}_prediction_{timestamp}.csv'
    result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'\nCSV 已保存: {csv_path}')

    # Markdown
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    report_dir = DOCS_DIR / 'analysis'
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f'{mode}_prediction_{timestamp}.md'

    title = '纯 Qlib 量化基线' if qlib_only else 'Qlib + TradingAgents 融合'
    w_labels = ['w_qlib', 'w_ai_score', 'w_trader', 'w_research']

    lines = [
        f'# {title}预测报告 ({days} 天)',
        '',
        f'**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        f'**方案**: {"方案 B 后处理融合" if not qlib_only else "纯 Qlib 基线"}',
        f'**模型**: LightGBM + Alpha158 (IC=0.051, Rank ICIR=1.365)',
        f'**基准日**: {base_date} 收盘',
        f'**预测天数**: {days} 个交易日',
        f'**融合权重**: {", ".join(f"{l}={w:.2f}" for l, w in zip(w_labels, weights))}',
        f'**衰减策略**: Day1={decay[0]:.0%} → Day{days}={decay[-1]:.0%} (线性衰减)',
        '',
        '## 融合分数说明',
        '',
        '```',
        '# AI 信号映射到 Qlib score 相同量级（score_std）',
        'ai_mapped  = (ai_score / 2) × score_std',
        'ta_mapped  = trader_action × score_std × 0.5',
        'rr_mapped  = (research_rating / 2) × score_std',
        '',
        'combined = w1 × qlib_score        # 保持原始量级',
        '         + w2 × ai_mapped          # PM 评级映射',
        '         + w3 × ta_mapped          # Trader 行动映射',
        '         + w4 × rr_mapped          # RM 评级映射',
        '```',
        '',
        '## 预测结果',
        '',
    ]

    # 信号概览表
    headers = ['排名', '股票', '名称', '基准价', 'Qlib分', 'AI分', 'TA分', 'RM分', '融合分', '方向']
    lines.append('| ' + ' | '.join(headers) + ' |')
    lines.append('| ' + ' | '.join(['---:'] * len(headers)) + ' |')

    sorted_df = result_df.sort_values('combined_score', ascending=False)
    for i, (_, row) in enumerate(sorted_df.iterrows()):
        direction = '↑' if row['combined_score'] > 0.05 else ('↓' if row['combined_score'] < -0.05 else '—')
        lines.append(
            f'| {i+1} | {row["stock_code"]} | {row["name"]} | '
            f'{row["base_close"]:.2f} | {row["qlib_score"]:+.4f} | '
            f'{row["ai_score"]:+d} | {row["trader_action"]:+d} | '
            f'{row["research_rating"]:+d} | {row["combined_score"]:+.4f} | {direction} |'
        )

    # 价格预测表（精简：仅 D1, D5, D10, D15, D20）
    check_days = [1, 5, 10, 15, 20] if days >= 20 else list(range(1, days+1))
    lines.extend([
        '',
        '## 价格预测（关键节点）',
        '',
        '| 股票 | 名称 | 基准价 | ' + ' | '.join(f'D{d}' for d in check_days) + ' | 累计涨幅 |',
        '| --- | --- | ---: | ' + ' | '.join(['---:'] * len(check_days)) + ' | ---: |',
    ])

    for _, row in sorted_df.iterrows():
        vals = [row['stock_code'], row['name'], f'{row["base_close"]:.2f}']
        for d in check_days:
            col = f'pred_D{d}'
            vals.append(f'{row[col]:.2f}' if col in row else 'N/A')
        last_day = f'D{days}'
        vals.append(f'{row[f"ret_D{days}"]:+.2f}%')
        lines.append('| ' + ' | '.join(vals) + ' |')

    # 衰减因子表
    lines.extend([
        '',
        '## 衰减因子',
        '',
        '| 天数 | 衰减 | 天数 | 衰减 |',
        '| ---: | ---: | ---: | ---: |',
    ])
    for i in range(0, days, 2):
        left = f'| D{i+1} | {decay[i]:.2f} |'
        if i+1 < days:
            right = f' D{i+2} | {decay[i+1]:.2f} |'
        else:
            right = ' | |'
        lines.append(left + right)

    lines.extend([
        '',
        '## 免责声明',
        '',
        '本预测由量化模型 + AI 信号融合生成，仅供参考，不构成任何投资建议。',
        f'预测准确度随天数递减，D10 以后为趋势外推，置信度较低。',
        '投资有风险，入市需谨慎。',
    ])

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'报告已保存: {report_path}')

    return csv_path, report_path


def print_summary(result_df: pd.DataFrame, days: int):
    """打印摘要表格到 stdout。"""
    sorted_df = result_df.sort_values('combined_score', ascending=False)
    print(f'\n{"=" * 150}')
    print(f'  20 只股票 {days} 天融合预测')
    print(f'  融合 = w1×Qlib + w2×ai_score + w3×trader_action + w4×research_rating')
    print(f'{"=" * 150}')

    header = f'{"#":>2} {"股票":>12} {"名称":>6} {"基准":>8} {"Qlib":>8} {"AI":>3} {"TA":>3} {"RM":>3} {"融合":>7}'
    for d in [1, 5, 10, 20]:
        if d <= days:
            header += f' {"D"+str(d):>8}'
    header += f' {"涨幅":>7}'
    print(header)
    print('-' * 150)

    for i, (_, row) in enumerate(sorted_df.iterrows()):
        direction = '↑' if row['combined_score'] > 0.05 else ('↓' if row['combined_score'] < -0.05 else '—')
        line = (f'{i+1:2d} {row["stock_code"]:>12s} {row["name"]:>6s} '
                f'{row["base_close"]:>8.2f} {row["qlib_score"]:>+8.4f} '
                f'{row["ai_score"]:>+3d} {row["trader_action"]:>+3d} '
                f'{row["research_rating"]:>+3d} {row["combined_score"]:>+7.4f}')
        for d in [1, 5, 10, 20]:
            col = f'pred_D{d}'
            if d <= days and col in row:
                line += f' {row[col]:>8.2f}'
        line += f' {row[f"ret_D{days}"]:>+6.2f}%'
        print(line)

    print('=' * 150)
    print(f'  注: D{days} 以后为趋势外推，置信度较低')
    print('=' * 150)


# ── 主流程 ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='方案 B 融合预测: Qlib + TradingAgents → N天价格')
    parser.add_argument('--qlib-pred', default=None, help='Qlib 预测 CSV 路径（默认自动查找最新）')
    parser.add_argument('--signals-dir', default=None, help='TradingAgents 日志目录（默认 ~/.tradingagents/logs）')
    parser.add_argument('--weights', type=float, nargs=4, default=[0.50, 0.25, 0.10, 0.15],
                        help='融合权重 [w_qlib, w_ai, w_trader, w_research]')
    parser.add_argument('--days', type=int, default=DEFAULT_DAYS, help=f'预测天数（默认 {DEFAULT_DAYS}）')
    parser.add_argument('--base-date', default='2026-05-08', help='基准日期（默认 2026-05-08）')
    parser.add_argument('--qlib-only', action='store_true', help='仅使用 Qlib 信号（跳过 TA，用于对比基线）')
    args = parser.parse_args()

    print(f'{"=" * 60}')
    print(f'方案 B 融合预测')
    print(f'  预测天数: {args.days}')
    print(f'  融合权重: {[f"{w:.2f}" for w in args.weights]}')
    print(f'  模式: {"纯 Qlib 基线" if args.qlib_only else "Qlib + TA 融合"}')
    print(f'{"=" * 60}')

    # 1. 加载 Qlib 预测分数
    qlib_df = load_qlib_predictions(args.qlib_pred)
    print(f'Qlib 预测: {len(qlib_df)} 条记录')

    # 如果是 latest-day 模式有多个日期，取最新一天
    if 'datetime' in qlib_df.columns:
        latest = qlib_df['datetime'].max()
        qlib_df = qlib_df[qlib_df['datetime'] == latest].copy()
        print(f'筛选至最新日期: {latest}')

    # 2. 加载 TradingAgents AI 信号
    if args.qlib_only:
        ta_signals = {}
        print('[qlib-only] 跳过 TradingAgents 信号')
    else:
        ta_signals = load_ta_signals(args.signals_dir)
        if ta_signals:
            matched = sum(1 for s in STOCKS if s in ta_signals)
            print(f'信号匹配: {matched}/{len(STOCKS)} 只股票有 AI 信号')
        else:
            print('[WARN] 未找到任何 TA 信号，将退化为纯 Qlib 模式')

    # 3. 融合信号
    fused_df = fuse_signals(qlib_df, ta_signals, args.weights, qlib_only=args.qlib_only)

    # 4. 获取基准价格
    price_map = get_actual_prices(args.base_date, STOCKS)

    # 5. 生成衰减因子
    decay = get_decay_factors(args.days)

    # 6. 预测价格
    result_df = predict_prices(fused_df, price_map, args.days, decay, base_date=args.base_date)
    print(f'\n成功预测: {len(result_df)} 只股票 × {args.days} 天')

    # 7. 输出
    print_summary(result_df, args.days)
    csv_path, report_path = generate_report(
        result_df, args.days, args.weights, args.base_date, args.qlib_only
    )

    return result_df


if __name__ == '__main__':
    main()
