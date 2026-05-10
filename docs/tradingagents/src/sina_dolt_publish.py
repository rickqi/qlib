"""
sina_dolt_publish.py — Publish qlib binary trading data to DoltHub

Reads tradingagents qlib binary data from local filesystem,
converts to CSV, and publishes to DoltHub via dolt CLI.

Target: https://www.dolthub.com/repositories/rickqi/tradingagents

Tables created:
  sina_a_stock_eod_price  — Daily OHLCV for all A-shares (PK: tradedate, symbol)
  sina_trade_calendar     — Trading calendar               (PK: trade_date)
  sina_stock_list         — Stock universe with dates      (PK: symbol)

Pattern reference: D:\\codes\\investment_data
  - daily_update.sh  : dolt table import -u <table> <csv>
  - dump_qlib_bin.sh : dolt sql-server + SQLAlchemy
  - Schema: final_a_stock_eod_price (tradedate, symbol, open, close, high, low, volume, adjclose, amount)

Usage:
  cd D:\\codes\\qlib
  python docs\\tradingagents\\src\\sina_dolt_publish.py [--no-push] [--chunk-size 500000]

Prerequisites:
  1. Dolt installed:    winget install DoltHub.Dolt
  2. Dolt authenticated: dolt login
  3. Repo exists:       https://www.dolthub.com/repositories/rickqi/tradingagents
"""

import os
import sys
import subprocess
import tempfile
import shutil
import argparse
import time

import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────
DOLT_EXE = r"C:\Program Files\Dolt\bin\dolt.exe"
DATA_DIR = r"C:\Users\szk220009\.qlib\qlib_data\tradingagents"
DOLTHUB_REMOTE = "rickqi/tradingagents"
FEATURES = ["open", "close", "high", "low", "volume"]
CHUNK_ROWS = 500_000  # rows per CSV chunk for dolt import


# ── Dolt CLI wrapper ───────────────────────────────────────────────────────
def dolt(*args, cwd=None, check=True):
    """Run a dolt CLI command, return subprocess result."""
    cmd = [DOLT_EXE] + list(args)
    print(f"  [dolt] {' '.join(args)}")
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:5]:
            print(f"         {line}")
        if r.stdout.strip().count('\n') > 5:
            print(f"         ... ({r.stdout.strip().count(chr(10))} lines total)")
    if r.returncode != 0:
        if r.stderr.strip():
            print(f"  [ERR] {r.stderr.strip()[:300]}")
        if check:
            raise RuntimeError(f"dolt exited {r.returncode}: {' '.join(args)}")
    return r


# ── Step 1: Read qlib binary data ─────────────────────────────────────────
def read_qlib_binary():
    """
    Read all qlib binary features from DATA_DIR.
    Returns:
      price_df:  DataFrame with (tradedate, symbol, open, close, high, low, volume)
      calendar:  list of date strings
      stock_list: list of dicts {symbol, start_date, end_date}
    """
    cal_path = os.path.join(DATA_DIR, "calendars", "day.txt")
    inst_path = os.path.join(DATA_DIR, "instruments", "all.txt")
    feat_dir = os.path.join(DATA_DIR, "features")

    # 1a. Read calendar
    with open(cal_path, "r") as f:
        calendar = [line.strip() for line in f if line.strip()]
    print(f"  Calendar: {calendar[0]} ~ {calendar[-1]} ({len(calendar)} days)")

    # 1b. Read instruments (stock list)
    stock_list = []
    with open(inst_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                stock_list.append({
                    "symbol": parts[0],     # e.g. "SH688001"
                    "start_date": parts[1], # e.g. "2020-07-20"
                    "end_date": parts[2],   # e.g. "2026-05-08"
                })
    print(f"  Instruments: {len(stock_list)} stocks")

    # 1c. Read all stock features (vectorized per-stock)
    stock_dirs = sorted([
        d for d in os.listdir(feat_dir)
        if os.path.isdir(os.path.join(feat_dir, d))
    ])
    print(f"  Feature dirs: {len(stock_dirs)}")
    print()

    dfs = []
    total = len(stock_dirs)
    t0 = time.time()

    for i, sdir in enumerate(stock_dirs):
        spath = os.path.join(feat_dir, sdir)

        # "sh688001" → "SH688001"
        symbol = sdir[:2].upper() + sdir[2:]

        # Read close.day.bin to get start_index and value count
        close_path = os.path.join(spath, "close.day.bin")
        if not os.path.exists(close_path):
            continue
        close_raw = np.fromfile(close_path, dtype=np.float32)
        start_idx = int(close_raw[0])
        close_vals = close_raw[1:]
        n_vals = len(close_vals)

        if n_vals == 0:
            continue

        # Date range for this stock
        end_idx = start_idx + n_vals
        if end_idx > len(calendar):
            end_idx = len(calendar)
            n_vals = end_idx - start_idx
        dates = calendar[start_idx:end_idx]

        # Read all feature arrays
        feat_arrays = {}
        for feat in FEATURES:
            fp = os.path.join(spath, f"{feat}.day.bin")
            if os.path.exists(fp):
                raw = np.fromfile(fp, dtype=np.float32)
                feat_arrays[feat] = raw[1:1 + n_vals]
            else:
                feat_arrays[feat] = np.full(n_vals, np.nan, dtype=np.float32)

        # Build stock DataFrame — only keep rows where close is not NaN (trading days)
        close_slice = close_vals[:n_vals]
        mask = ~np.isnan(close_slice)

        stock_df = pd.DataFrame({
            "tradedate": [dates[j] for j in range(n_vals) if mask[j]],
            "symbol": symbol,
            "open": feat_arrays["open"][mask],
            "close": close_slice[mask],
            "high": feat_arrays["high"][mask],
            "low": feat_arrays["low"][mask],
            "volume": feat_arrays["volume"][mask],
        })
        dfs.append(stock_df)

        if (i + 1) % 1000 == 0 or i == total - 1:
            elapsed = time.time() - t0
            rows_so_far = sum(len(d) for d in dfs)
            print(f"  Read {i+1}/{total} stocks  |  {rows_so_far:,} rows  |  {elapsed:.1f}s")

    price_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    print(f"\n  Total: {len(price_df):,} rows from {len(dfs)} stocks")

    return price_df, calendar, stock_list


# ── Step 2: Generate CSV chunks ────────────────────────────────────────────
def generate_csvs(price_df, calendar, stock_list, tmpdir, chunk_size=CHUNK_ROWS):
    """
    Export DataFrames to chunked CSV files for dolt import.
    Returns list of (table_name, csv_path, is_first_for_table) tuples.
    """
    csv_files = []

    # 2a. Stock price data — chunked
    n_total = len(price_df)
    n_chunks = max(1, (n_total + chunk_size - 1) // chunk_size)
    print(f"\n  Price data: {n_total:,} rows → {n_chunks} chunk(s) of ≤{chunk_size:,} rows")

    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n_total)
        chunk_df = price_df.iloc[start:end]
        path = os.path.join(tmpdir, f"price_{i:04d}.csv")
        chunk_df.to_csv(path, index=False)
        csv_files.append(("sina_a_stock_eod_price", path, i == 0))
        print(f"    chunk {i+1}/{n_chunks}: {len(chunk_df):,} rows → {os.path.basename(path)}")

    # 2b. Calendar
    cal_df = pd.DataFrame({"trade_date": calendar, "is_open": 1})
    cal_path = os.path.join(tmpdir, "calendar.csv")
    cal_df.to_csv(cal_path, index=False)
    csv_files.append(("sina_trade_calendar", cal_path, True))
    print(f"    calendar: {len(cal_df)} rows")

    # 2c. Stock list
    sl_df = pd.DataFrame(stock_list)
    sl_path = os.path.join(tmpdir, "stock_list.csv")
    sl_df.to_csv(sl_path, index=False)
    csv_files.append(("sina_stock_list", sl_path, True))
    print(f"    stock_list: {len(sl_df)} rows")

    return csv_files


# ── Step 3: Dolt clone + schema + import + commit + push ───────────────────
TABLE_SCHEMAS = {
    "sina_a_stock_eod_price": """
        CREATE TABLE IF NOT EXISTS sina_a_stock_eod_price (
            tradedate  DATE NOT NULL,
            symbol     VARCHAR(20) NOT NULL,
            open       DOUBLE,
            close      DOUBLE,
            high       DOUBLE,
            low        DOUBLE,
            volume     DOUBLE,
            PRIMARY KEY (tradedate, symbol)
        )
    """,
    "sina_trade_calendar": """
        CREATE TABLE IF NOT EXISTS sina_trade_calendar (
            trade_date VARCHAR(20) NOT NULL,
            is_open    INT,
            PRIMARY KEY (trade_date)
        )
    """,
    "sina_stock_list": """
        CREATE TABLE IF NOT EXISTS sina_stock_list (
            symbol     VARCHAR(20) NOT NULL,
            start_date VARCHAR(20),
            end_date   VARCHAR(20),
            PRIMARY KEY (symbol)
        )
    """,
}


def dolt_publish(csv_files, tmpdir, push=True):
    """
    Clone the DoltHub repo, create tables with explicit schemas,
    import CSV chunks, commit, and optionally push.
    """
    repo_dir = os.path.join(tmpdir, "tradingagents")

    # 3a. Init local dolt repo (remote may be empty, so init + add remote)
    print(f"\n{'='*60}")
    print("Step 3: Initialize Dolt repo")
    print(f"{'='*60}")
    if os.path.isdir(repo_dir):
        shutil.rmtree(repo_dir, ignore_errors=True)
    os.makedirs(repo_dir, exist_ok=True)

    # Try clone first; if remote is empty, fall back to init + remote add
    clone_result = dolt("clone", DOLTHUB_REMOTE, repo_dir, check=False)
    if clone_result.returncode != 0:
        # clone may have created an empty/partial directory — remove it
        if os.path.isdir(repo_dir):
            shutil.rmtree(repo_dir, ignore_errors=True)
        os.makedirs(repo_dir, exist_ok=True)
        print("  Remote empty or not clonable — initializing locally")
        dolt("init", "--name", "rickqi", "--email", "rickqi@users.noreply.github.com",
             cwd=repo_dir)
        dolt("remote", "add", "origin", f"https://doltremoteapi.dolthub.com/{DOLTHUB_REMOTE}",
             cwd=repo_dir)
    print(f"  Repo dir: {repo_dir}")

    # 3b. Create tables with explicit schema
    print(f"\n{'='*60}")
    print("Step 4: Create tables")
    print(f"{'='*60}")
    for table_name, ddl in TABLE_SCHEMAS.items():
        dolt("sql", "-q", ddl.strip(), cwd=repo_dir)
        print(f"  Created: {table_name}")

    # 3c. Import CSV chunks
    print(f"\n{'='*60}")
    print("Step 5: Import data")
    print(f"{'='*60}")
    for table_name, csv_path, is_first in csv_files:
        if is_first:
            # First chunk for this table — use -u (table already exists with schema)
            print(f"\n  Importing {os.path.basename(csv_path)} → {table_name} (first chunk)")
            dolt("table", "import", "-u", table_name, csv_path, cwd=repo_dir)
        else:
            # Subsequent chunks — update (insert new rows via PK)
            print(f"\n  Importing {os.path.basename(csv_path)} → {table_name} (update)")
            dolt("table", "import", "-u", table_name, csv_path, cwd=repo_dir)

    # 3d. Commit
    print(f"\n{'='*60}")
    print("Step 6: Commit")
    print(f"{'='*60}")
    dolt("add", "-A", cwd=repo_dir)

    status = dolt("status", cwd=repo_dir, check=False)
    if "nothing to commit" in (status.stdout or ""):
        print("  No changes to commit.")
    else:
        # Count rows for commit message
        price_rows = sum(
            1 for t, _, _ in csv_files if t == "sina_a_stock_eod_price"
        )
        dolt("commit", "-m",
             f"Sina A-share daily OHLCV data import",
             cwd=repo_dir)
        print("  Committed.")

    # 3e. Push
    if push:
        print(f"\n{'='*60}")
        print("Step 7: Push to DoltHub")
        print(f"{'='*60}")
        # Detect current branch name (dolt init defaults to 'main')
        br_result = dolt("branch", "--show-current", cwd=repo_dir, check=False)
        branch = (br_result.stdout or "").strip() or "main"
        dolt("push", "-u", "origin", branch, cwd=repo_dir)
        print(f"\n  DONE! → https://www.dolthub.com/repositories/{DOLTHUB_REMOTE}")
    else:
        print("\n  [--no-push] Skipping push. Data is committed locally at:")
        print(f"  {repo_dir}")
        print("  To push manually:")
        print(f'    cd {repo_dir}')
        br_result = dolt("branch", "--show-current", cwd=repo_dir, check=False)
        branch = (br_result.stdout or "").strip() or "main"
        print(f'    "{DOLT_EXE}" push -u origin {branch}')


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Publish qlib trading data to DoltHub"
    )
    parser.add_argument(
        "--no-push", action="store_true",
        help="Commit locally but skip push to DoltHub"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=CHUNK_ROWS,
        help=f"Rows per CSV chunk (default: {CHUNK_ROWS:,})"
    )
    parser.add_argument(
        "--keep-tmp", action="store_true",
        help="Keep temp directory after completion (for debugging)"
    )
    args = parser.parse_args()

    chunk_size = args.chunk_size

    # Verify dolt exists
    if not os.path.isfile(DOLT_EXE):
        print(f"ERROR: dolt not found at {DOLT_EXE}")
        print("Install: winget install DoltHub.Dolt")
        sys.exit(1)

    # Verify data dir exists
    if not os.path.isdir(DATA_DIR):
        print(f"ERROR: Data directory not found: {DATA_DIR}")
        print("Run sina_fetch.py first to generate qlib binary data.")
        sys.exit(1)

    tmpdir = tempfile.mkdtemp(prefix="dolt_publish_")
    print(f"Temp dir: {tmpdir}")

    try:
        # Step 1-2: Read binary + generate CSV
        print(f"{'='*60}")
        print("Step 1-2: Read qlib binary data → CSV")
        print(f"{'='*60}")
        price_df, calendar, stock_list = read_qlib_binary()
        csv_files = generate_csvs(price_df, calendar, stock_list, tmpdir, chunk_size)

        # Step 3-7: Dolt clone + import + push
        dolt_publish(csv_files, tmpdir, push=not args.no_push)

    finally:
        if not args.keep_tmp:
            shutil.rmtree(tmpdir, ignore_errors=True)
            print(f"\nCleaned up temp dir: {tmpdir}")
        else:
            print(f"\nTemp dir kept: {tmpdir}")


if __name__ == "__main__":
    main()
