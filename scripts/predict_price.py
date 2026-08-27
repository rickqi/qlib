"""
预测20只股票 5.11-5.15 实际价格
使用 Qlib 模型预测分数 + tushare 实际收盘价

用法:
    .venv\Scripts\python.exe scripts\predict_price.py
"""
import os
import sys
from datetime import datetime
from pathlib import Path

# 清除代理（tushare 不需要代理）
for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    os.environ.pop(key, None)

import pandas as pd
import numpy as np
import tushare as ts

# ── 配置 ──────────────────────────────────────────────────────

def _get_tushare_token() -> str:
    """从环境变量或 .env 文件获取 Tushare token。"""
    # 1. 环境变量
    token = os.environ.get("TUSHARE_API_KEY") or os.environ.get("TUSHARE")
    if token:
        return token
    # 2. TradingAgents/.env
    env_file = Path(__file__).resolve().parent.parent.parent / "TradingAgents" / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() in ("TUSHARE_API_KEY", "TUSHARE"):
                return v.strip().strip("\"'")
    # 3. investment_data/.env
    env_file2 = Path(__file__).resolve().parent.parent.parent / "investment_data" / ".env"
    if env_file2.exists():
        for line in env_file2.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() in ("TUSHARE_API_KEY", "TUSHARE"):
                return v.strip().strip("\"'")
    raise RuntimeError(
        "未找到 Tushare token: 请设置 TUSHARE_API_KEY 环境变量"
        "或在 TradingAgents/.env 中配置"
    )

TUSHARE_TOKEN = _get_tushare_token()
REPORTS_DIR = Path(__file__).parent.parent / 'reports'

# Load shared stock config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'docs' / 'scripts'))
from stocks_config import STOCKS, STOCK_NAMES

# 交易日 (5.11-5.15 周一至周五)
TRADING_DAYS = ['2026-05-11', '2026-05-12', '2026-05-13', '2026-05-14', '2026-05-15']
BASE_DATE = '2026-05-11'

# A股涨跌停限制
DAILY_LIMIT = 0.10  # ±10%


def load_prediction_scores():
    """加载最新的 qlib 预测分数。

    只匹配时间戳命名的 ``predictions_YYYYMMDD_HHMMSS.csv``（glob
    ``predictions_[0-9]*.csv``），排除 ``predictions_test.csv`` 等非时间戳
    文件——它们按字典序排在所有时间戳文件之后，旧 glob 会误选过期数据
    （与 predict_fused.py 的同款修复一致）。
    """
    pred_files = sorted(REPORTS_DIR.glob('predictions_[0-9]*.csv'))
    if not pred_files:
        print('错误: 未找到预测文件')
        sys.exit(1)

    latest = pred_files[-1]
    print(f'加载预测文件: {latest.name}')
    df = pd.read_csv(latest, encoding='utf-8-sig')
    return df


def get_actual_prices(date, stocks):
    """从 tushare 获取指定日期的实际收盘价"""
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()

    ts_codes = ','.join(stocks)
    date_fmt = date.replace('-', '')

    print(f'获取 {date} 实际收盘价...')
    df = pro.daily(ts_code=ts_codes, start_date=date_fmt, end_date=date_fmt)

    if df is None or df.empty:
        print(f'  警告: 未获取到 {date} 的数据')
        return {}

    price_map = {}
    for _, row in df.iterrows():
        price_map[row['ts_code']] = {
            'close': row['close'],
            'open': row.get('open', None),
            'high': row.get('high', None),
            'low': row.get('low', None),
            'vol': row.get('vol', None),
        }

    print(f'  获取到 {len(price_map)} 只股票的数据')
    return price_map


def predict_prices(pred_df, price_map):
    """
    将模型预测分数转换为实际价格预测
    
    策略:
    - Day 1 (5.11): 使用模型预测分数作为日收益率
    - Day 2-5: 使用衰减的预测分数 (信心递减)
    - 每日涨跌幅限制在 ±10% (A股规则)
    """
    results = []

    # 衰减因子: 模型预测的信心随天数递减
    decay = [1.0, 0.70, 0.50, 0.35, 0.25]

    for _, pred_row in pred_df.iterrows():
        stock_code = pred_row['stock_code']
        score = pred_row['score']
        base_price = price_map.get(stock_code, {}).get('close')

        if base_price is None:
            print(f'  跳过 {stock_code}: 无基准价格')
            continue

        row = {
            'stock_code': stock_code,
            'name': STOCK_NAMES.get(stock_code, ''),
            'base_close': round(base_price, 2),
            'score': round(score, 6),
        }

        prev_price = base_price
        for i, day in enumerate(TRADING_DAYS):
            # 计算日收益率: score × 衰减因子
            daily_return = score * decay[i]

            # 限制涨跌幅
            daily_return = max(-DAILY_LIMIT, min(DAILY_LIMIT, daily_return))

            # 预测价格
            pred_price = prev_price * (1 + daily_return)
            pred_price = round(pred_price, 2)

            # 累计收益率
            cum_return = (pred_price / base_price - 1) * 100

            row[f'pred_{day}'] = pred_price
            row[f'ret_{day}'] = round(cum_return, 2)

            prev_price = pred_price

        results.append(row)

    return pd.DataFrame(results)


def generate_report(result_df):
    """生成价格预测报告"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 保存 CSV
    csv_path = REPORTS_DIR / f'price_prediction_{timestamp}.csv'
    result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'\nCSV 已保存: {csv_path}')

    # 生成 Markdown 报告
    report = [
        f'# 20只股票实际价格预测 (5.11-5.15)',
        f'',
        f'**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        f'**模型**: LightGBM + Alpha158 (IC=0.051)',
        f'**基准日**: 2026-05-08 收盘',
        f'**预测方式**: 模型分数 → 日收益率 → 实际价格',
        f'**衰减策略**: Day1=100%, Day2=70%, Day3=50%, Day4=35%, Day5=25%',
        f'',
        f'## 预测结果',
        f'',
    ]

    # 表格
    headers = ['排名', '股票', '名称', '5.8收盘', '方向']
    for d in TRADING_DAYS:
        headers.append(d)
    headers.append('周涨幅')

    report.append('| ' + ' | '.join(headers) + ' |')
    report.append('| ' + ' | '.join(['---:'] * len(headers)) + ' |')

    for i, (_, row) in enumerate(result_df.iterrows()):
        direction = '↑' if row['score'] > 0.01 else ('↓' if row['score'] < -0.01 else '—')
        vals = [
            str(i + 1),
            row['stock_code'],
            row['name'],
            f'{row["base_close"]:.2f}',
            direction,
        ]
        for d in TRADING_DAYS:
            vals.append(f'{row[f"pred_{d}"]:.2f}')
        vals.append(f'{row[f"ret_{TRADING_DAYS[-1]}"]:+.2f}%')
        report.append('| ' + ' | '.join(vals) + ' |')

    report.extend([
        '',
        '## 预测说明',
        '',
        '1. **模型分数 (score)**: Qlib LightGBM 模型预测的下交易日收益率',
        '2. **衰减因子**: 多日预测中，每日预测信心递减（100%→70%→50%→35%→25%）',
        '3. **涨跌停限制**: 每日预测涨跌幅不超过 ±10%（A股规则）',
        '4. **价格精度**: 基准为 tushare 2026-05-08 实际收盘价',
        '',
        '## 免责声明',
        '',
        '本预测由量化模型生成，仅供参考，不构成投资建议。',
        '预测准确度随天数递减，5.12-5.15 为趋势外推，置信度较低。',
        '投资有风险，入市需谨慎。',
    ])

    report_path = REPORTS_DIR.parent / 'docs' / 'analysis' / f'price_prediction_{timestamp}.md'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print(f'报告已保存: {report_path}')

    return csv_path, report_path


def print_summary(result_df):
    """打印摘要表格"""
    print('\n' + '=' * 130)
    print('  20只股票 5.11-5.15 实际价格预测')
    print('  模型: LightGBM + Alpha158 | 基准: 2026-05-08 收盘 | 衰减: 100%→70%→50%→35%→25%')
    print('=' * 130)

    header = f'{"#":>2} {"股票":>12} {"名称":>6} {"5.8收盘":>8}'
    for d in TRADING_DAYS:
        short = d[5:]  # "05-11"
        header += f' {short+":":>8}'
    header += f' {"周涨幅":>7}'
    print(header)
    print('-' * 130)

    for i, (_, row) in enumerate(result_df.iterrows()):
        direction = '↑' if row['score'] > 0.01 else ('↓' if row['score'] < -0.01 else '—')
        line = f'{i+1:2d} {row["stock_code"]:>12s} {row["name"]:>6s} {row["base_close"]:>8.2f}'
        for d in TRADING_DAYS:
            line += f' {row[f"pred_{d}"]:>8.2f}'
        line += f' {row[f"ret_{TRADING_DAYS[-1]}"]:>+6.2f}%'
        print(line)

    print('=' * 130)
    print('  注: 5.12-5.15 为趋势外推预测，置信度逐日递减')
    print('=' * 130)


def main():
    # 1. 加载预测分数
    pred_df = load_prediction_scores()
    print(f'预测记录: {len(pred_df)} 条\n')

    # 2. 获取 5.08 实际收盘价
    price_map = get_actual_prices(BASE_DATE, STOCKS)

    # 3. 计算预测价格
    result_df = predict_prices(pred_df, price_map)
    print(f'\n成功预测: {len(result_df)} 只股票')

    # 4. 显示摘要
    print_summary(result_df)

    # 5. 保存报告
    csv_path, report_path = generate_report(result_df)

    return result_df


if __name__ == '__main__':
    main()
