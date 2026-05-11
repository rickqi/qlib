"""
数据同步脚本：将 qlib_bin 二进制数据同步到本地 Dolt 数据库并推送到 DoltHub。

用法:
    .venv\\Scripts\\python.exe scripts\\sync_to_dolt.py                       # 全量同步
    .venv\\Scripts\\python.exe scripts\\sync_to_dolt.py --mode incremental    # 增量同步
    .venv\\Scripts\\python.exe scripts\\sync_to_dolt.py --stocks SH600000,SZ000001  # 指定股票
    .venv\\Scripts\\python.exe scripts\\sync_to_dolt.py --no-push             # 不同步到远程
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── 默认路径配置 ──────────────────────────────────────────────────────
DEFAULT_QLIB_DIR = "C:/codes/qlib/qlib_bin"
DEFAULT_DOLT_DIR = r"C:\codes\stock\dolt\tradingagents"
DEFAULT_DOLT_BIN = r"C:\tools\dolt\dolt-windows-amd64\bin\dolt.exe"

# qlib_bin 中的 10 个特征文件
FEATURES = ["adjclose", "amount", "change", "close", "factor", "high", "low", "open", "volume", "vwap"]

# 目标表中的列映射: CSV列名 → 顺序与 CREATE TABLE 一致
# 注意: qlib 的 "change" 在 dolt 中重命名为 "pct_change"（change 是 SQL 保留字）
CSV_COLUMNS = ["tradedate", "symbol", "open", "high", "low", "close", "volume",
               "adjclose", "factor", "pct_change", "amount", "vwap"]


def load_calendar(qlib_dir: str) -> list:
    """加载交易日历"""
    cal_path = os.path.join(qlib_dir, "calendars", "day.txt")
    with open(cal_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def get_stock_list(qlib_dir: str) -> list:
    """获取所有股票目录名"""
    features_dir = os.path.join(qlib_dir, "features")
    return sorted([d for d in os.listdir(features_dir)
                    if os.path.isdir(os.path.join(features_dir, d))])


def read_feature(qlib_dir: str, stock: str, feature: str, cal_len: int) -> np.ndarray:
    """读取单个特征二进制文件，返回与日历对齐的数组（跳过首个占位值）"""
    fpath = os.path.join(qlib_dir, "features", stock, f"{feature}.day.bin")
    if not os.path.exists(fpath):
        return np.full(cal_len, np.nan)
    arr = np.fromfile(fpath, dtype="<f")  # little-endian float32
    # 跳过 index 0 的占位值，取 index 1..cal_len
    if len(arr) < cal_len + 1:
        padded = np.full(cal_len, np.nan)
        valid_len = min(len(arr) - 1, cal_len) if len(arr) > 1 else 0
        if valid_len > 0:
            padded[:valid_len] = arr[1:1 + valid_len]
        return padded
    return arr[1:1 + cal_len]


def build_stock_df(qlib_dir: str, stock: str, calendar: list,
                   date_filter: set | None = None) -> pd.DataFrame:
    """构建单只股票的 DataFrame"""
    cal_len = len(calendar)

    # 读取所有特征
    features = {}
    for feat in FEATURES:
        features[feat] = read_feature(qlib_dir, stock, feat, cal_len)

    # 构建 DataFrame
    close = features["close"]
    valid_mask = np.isfinite(close) & (close != 0)

    # 日期过滤（增量同步时使用）
    if date_filter is not None:
        date_mask = np.array([d in date_filter for d in calendar])
        valid_mask = valid_mask & date_mask

    if not np.any(valid_mask):
        return pd.DataFrame()

    rows: dict = {
        "tradedate": [calendar[i] for i in range(cal_len) if valid_mask[i]],
        "symbol": [stock] * int(np.sum(valid_mask)),
    }
    for feat in FEATURES:
        # "change" 列在 dolt 中映射为 "pct_change"
        col_name = "pct_change" if feat == "change" else feat
        rows[col_name] = features[feat][valid_mask]

    # 重排列顺序匹配 CSV_COLUMNS
    df = pd.DataFrame(rows)
    return df.reindex(columns=CSV_COLUMNS)


def dolt_sql(dolt_bin: str, dolt_dir: str, query: str) -> str:
    """执行 dolt sql 命令"""
    result = subprocess.run(
        [dolt_bin, "sql", "--query", query],
        cwd=dolt_dir, capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        print(f"  [WARN] dolt sql error: {result.stderr.strip()}")
    return result.stdout


def dolt_table_import(dolt_bin: str, dolt_dir: str, table: str, csv_path: str,
                      mode: str = "create") -> bool:
    """执行 dolt table import。
    mode: 'create' = -c -f (创建/覆盖), 'append' = -a (追加), 'update' = -u (更新)
    """
    if mode == "create":
        cmd = [dolt_bin, "table", "import", "-c", "-f", "--pk", "tradedate,symbol", table, csv_path]
    elif mode == "append":
        cmd = [dolt_bin, "table", "import", "-a", table, csv_path]
    else:
        cmd = [dolt_bin, "table", "import", "-u", table, csv_path]
    timeout = 3600  # 大批量导入给更多时间
    result = subprocess.run(
        cmd, cwd=dolt_dir, capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0:
        print(f"  [ERROR] dolt import ({mode}) failed: {result.stderr.strip()}")
        return False
    return True


def dolt_add_commit_push(dolt_bin: str, dolt_dir: str, message: str, no_push: bool = False):
    """dolt add . && dolt commit && dolt push"""
    # add
    subprocess.run([dolt_bin, "add", "."], cwd=dolt_dir, capture_output=True, timeout=120)

    # commit
    result = subprocess.run(
        [dolt_bin, "commit", "-m", message],
        cwd=dolt_dir, capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        print(f"  [WARN] dolt commit: {result.stderr.strip()}")
        return
    # 提取 commit hash
    for line in result.stdout.splitlines():
        if "commit" in line.lower():
            print(f"  Commit: {line.strip()}")
            break

    if no_push:
        print("  (--no-push) 跳过推送")
        return

    # push
    print("  推送到 DoltHub...")
    result = subprocess.run(
        [dolt_bin, "push", "origin", "main"],
        cwd=dolt_dir, capture_output=True, text=True, timeout=1800
    )
    if result.returncode != 0:
        print(f"  [ERROR] dolt push: {result.stderr.strip()}")
    else:
        print("  推送完成")


def create_table_if_needed(dolt_bin: str, dolt_dir: str):
    """创建 qlib_stock_eod 表（如不存在）"""
    create_sql = """
    CREATE TABLE IF NOT EXISTS qlib_stock_eod (
        tradedate DATE NOT NULL,
        symbol VARCHAR(20) NOT NULL,
        open DOUBLE,
        high DOUBLE,
        low DOUBLE,
        close DOUBLE,
        volume DOUBLE,
        adjclose DOUBLE,
        factor DOUBLE,
        pct_change DOUBLE,
        amount DOUBLE,
        vwap DOUBLE,
        PRIMARY KEY (tradedate, symbol)
    )
    """
    dolt_sql(dolt_bin, dolt_dir, create_sql.strip())
    print("表 qlib_stock_eod 已就绪")


def get_max_date(dolt_bin: str, dolt_dir: str) -> str | None:
    """查询 qlib_stock_eod 中最大 tradedate"""
    output = dolt_sql(dolt_bin, dolt_dir,
                       "SELECT MAX(tradedate) as max_date FROM qlib_stock_eod")
    for line in output.splitlines():
        # 跳过分隔线和表头
        if line.strip() and "|" in line and "max_date" not in line and "tradedate" not in line and "---" not in line:
            val = line.replace("|", "").strip()
            if val and val != "NULL":
                return val
    return None


def main():
    parser = argparse.ArgumentParser(description="同步 qlib_bin 数据到 Dolt 数据库")
    parser.add_argument("--mode", choices=["full", "incremental"], default="full",
                        help="同步模式: full=全量, incremental=增量")
    parser.add_argument("--stocks", type=str, default=None,
                        help="指定股票，逗号分隔 (如 SH600000,SZ000001)")
    parser.add_argument("--chunk-size", type=int, default=100,
                        help="每批处理的股票数 (默认 100)")
    parser.add_argument("--no-push", action="store_true",
                        help="不同步推送到 DoltHub")
    parser.add_argument("--qlib-dir", type=str, default=DEFAULT_QLIB_DIR,
                        help="qlib_bin 数据目录")
    parser.add_argument("--dolt-dir", type=str, default=DEFAULT_DOLT_DIR,
                        help="Dolt 本地仓库目录")
    parser.add_argument("--dolt-bin", type=str, default=DEFAULT_DOLT_BIN,
                        help="dolt 可执行文件路径")
    args = parser.parse_args()

    start_time = time.time()

    # ── 1. 创建表 ──────────────────────────────────────────────────
    print("=" * 60)
    print("qlib_bin → Dolt 数据同步")
    print("=" * 60)
    # 全量模式: 首批 -c -f 自动创建表; 增量模式: 确保表存在
    if args.mode == "incremental":
        create_table_if_needed(args.dolt_bin, args.dolt_dir)

    # ── 2. 加载日历 ────────────────────────────────────────────────
    calendar = load_calendar(args.qlib_dir)
    print(f"交易日历: {calendar[0]} ~ {calendar[-1]} ({len(calendar)} 天)")

    # ── 3. 确定日期过滤 ────────────────────────────────────────────
    date_filter = None
    if args.mode == "incremental":
        max_date = get_max_date(args.dolt_bin, args.dolt_dir)
        if max_date:
            print(f"增量模式: 已有数据截止 {max_date}，只同步之后的数据")
            date_filter = {d for d in calendar if d > max_date}
            print(f"需要同步 {len(date_filter)} 个交易日")
            if not date_filter:
                print("无需同步，退出")
                return
        else:
            print("增量模式但表为空，回退到全量模式")

    # ── 4. 确定股票列表 ────────────────────────────────────────────
    if args.stocks:
        stocks = [s.strip() for s in args.stocks.split(",")]
        print(f"指定股票: {len(stocks)} 只")
    else:
        stocks = get_stock_list(args.qlib_dir)
        print(f"全量股票: {len(stocks)} 只")

    # ── 5. 分块处理 ────────────────────────────────────────────────
    total_rows = 0
    total_chunks = (len(stocks) + args.chunk_size - 1) // args.chunk_size
    import_mode = "create"  # 首批 create，后续 append

    for chunk_idx in range(total_chunks):
        start = chunk_idx * args.chunk_size
        end = min(start + args.chunk_size, len(stocks))
        chunk_stocks = stocks[start:end]

        print(f"\n处理第 {chunk_idx + 1}/{total_chunks} 批 (股票 {start + 1}..{end})...")

        # 构建所有股票的 DataFrame
        dfs = []
        for stock in chunk_stocks:
            df = build_stock_df(args.qlib_dir, stock, calendar, date_filter)
            if not df.empty:
                dfs.append(df)

        if not dfs:
            print(f"  本批无有效数据，跳过")
            continue

        combined = pd.concat(dfs, ignore_index=True)
        chunk_rows = len(combined)
        total_rows += chunk_rows
        print(f"  生成 {chunk_rows:,} 行，模式: {import_mode}")

        # 写入临时 CSV 并导入
        tmp_csv = os.path.join(tempfile.gettempdir(), f"qlib_sync_{chunk_idx}.csv")
        try:
            combined.to_csv(tmp_csv, index=False)
            if dolt_table_import(args.dolt_bin, args.dolt_dir, "qlib_stock_eod", tmp_csv, mode=import_mode):
                print(f"  导入成功")
                import_mode = "append"  # 首批 create 成功后，后续切 append
            else:
                print(f"  导入失败!")
        finally:
            if os.path.exists(tmp_csv):
                os.remove(tmp_csv)

    # ── 6. 提交和推送 ──────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n同步完成: {total_rows:,} 行, 耗时 {elapsed:.1f}s")

    if total_rows > 0:
        date_range = f"{calendar[0]}~{calendar[-1]}"
        if date_filter:
            date_range = f"{min(date_filter)}~{max(date_filter)}"
        msg = f"Sync qlib_bin data: {len(stocks)} stocks, {date_range}, {total_rows:,} rows"
        dolt_add_commit_push(args.dolt_bin, args.dolt_dir, msg, args.no_push)
    else:
        print("无新数据需要提交")


if __name__ == "__main__":
    main()
