"""
Step 1+2: Fetch Sina Finance data → convert to qlib binary format
No multiprocessing — safe to run directly

Stock universe: reads from C:\code\qlib\qlib_bin\instruments\all.txt
Loads ALL active SH/SZ stocks (skips BJ, indices, delisted)
Skips stocks already present in features/ directory (incremental mode)

Anti-scraping measures:
  - Random delay between requests (1.0~3.0s, exponential jitter)
  - Rotating User-Agent (10 desktop browsers)
  - Batch processing with long pauses between batches
  - Exponential backoff on retries (2s → 4s → 8s → 16s → 32s)
  - Referer header mimicking browser navigation
  - 403/429 block detection with long backoff
"""
import os, sys, json, time, random
import numpy as np
import pandas as pd

# Disable proxy
for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    os.environ[k] = ""

import requests

DATA_DIR = r"C:\Users\szk220009\.qlib\qlib_data\tradingagents"
INSTRUMENTS_FILE = r"C:\code\qlib\qlib_bin\instruments\all.txt"

# ============================================================
# Code conversion helpers
# ============================================================
def qlib_code_to_wind(code):
    """SH688001 → 688001.SH, SZ300001 → 300001.SZ"""
    market = code[:2].upper()
    number = code[2:]
    return f"{number}.{market}"

def to_qlib_dir(c):
    n, m = c.split(".")
    return f"{m.lower()}{n}"

def to_qlib_code(c):
    n, m = c.split(".")
    return f"{m.upper()}{n}"

def to_sina_code(c):
    n, m = c.split(".")
    return f"{m.lower()}{n}"

# ============================================================
# Load ALL active stocks from instruments file
# ============================================================
def load_stock_list():
    """Load all active non-BJ non-index stocks from instruments file"""
    codes = []
    skipped_bj = 0
    skipped_idx = 0
    skipped_delisted = 0
    with open(INSTRUMENTS_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            qlib_code = parts[0]
            end_date = parts[2] if len(parts) >= 3 else ""

            # Skip Beijing stocks (Sina doesn't support)
            if qlib_code.startswith("BJ"):
                skipped_bj += 1
                continue

            # Skip index codes (SH000xxx, SZ399xxx, SH8xx, SZ8xx)
            if qlib_code.startswith(("SH000", "SZ399", "SH8", "SZ8")):
                skipped_idx += 1
                continue

            # Skip delisted: end_date before 2025-01-01
            if end_date < "2025-01-01":
                skipped_delisted += 1
                continue

            codes.append(qlib_code_to_wind(qlib_code))

    print(f"  Loaded from {INSTRUMENTS_FILE}")
    print(f"  Active SH/SZ stocks: {len(codes)}")
    print(f"  Skipped: {skipped_bj} BJ, {skipped_idx} indices, {skipped_delisted} delisted")
    return codes

ALL_CODES = load_stock_list()

# ============================================================
# Check existing data → incremental skip
# ============================================================
features_dir = os.path.join(DATA_DIR, "features")
EXISTING_CODES = set()
if os.path.isdir(features_dir):
    for d in os.listdir(features_dir):
        dp = os.path.join(features_dir, d)
        if os.path.isdir(dp):
            # "sh688001" → "688001.SH"
            market = d[:2].upper()
            number = d[2:]
            EXISTING_CODES.add(f"{number}.{market}")

TO_FETCH = [c for c in ALL_CODES if c not in EXISTING_CODES]
print(f"  Already fetched: {len(EXISTING_CODES)}")
print(f"  Need to fetch:   {len(TO_FETCH)}")

if len(TO_FETCH) == 0:
    print("\n  Nothing to fetch. All stocks already in data directory.")
    print("  Run sina_train.py for training + prediction.")
    sys.exit(0)

# ============================================================
# Anti-scraping configuration
# ============================================================
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 OPR/111.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]

BATCH_SIZE = 50        # stocks per batch
BATCH_PAUSE = 15.0     # seconds pause between batches
DELAY_MIN = 1.0        # min delay between requests (seconds)
DELAY_MAX = 3.0        # max delay between requests (seconds)
MAX_RETRIES = 5        # max retries per stock
RETRY_BASE = 2.0       # base retry delay (exponential backoff)

sess = requests.Session()
sess.trust_env = False
sess.proxies = {"http": None, "https": None}

def random_delay():
    d = random.uniform(DELAY_MIN, DELAY_MAX)
    time.sleep(d)

def rotate_ua():
    sess.headers.update({
        "User-Agent": random.choice(USER_AGENTS),
        "Referer": "http://finance.sina.com.cn/",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
    })

# ============================================================
# Fetch (only new stocks)
# ============================================================
print()
print("=" * 60)
print("Step 1: Fetch from Sina Finance API")
print(f"  To fetch: {len(TO_FETCH)} | Batch: {BATCH_SIZE} | "
      f"Delay: {DELAY_MIN}~{DELAY_MAX}s | Pause: {BATCH_PAUSE}s")
print("=" * 60)

all_data = {}
failed_codes = []
n_batches = (len(TO_FETCH) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_idx in range(n_batches):
    batch_start = batch_idx * BATCH_SIZE
    batch_end = min(batch_start + BATCH_SIZE, len(TO_FETCH))
    batch = TO_FETCH[batch_start:batch_end]

    if batch_idx > 0:
        pause = BATCH_PAUSE + random.uniform(0, 5)
        print(f"\n  --- Batch {batch_idx+1}/{n_batches}: pausing {pause:.1f}s ---")
        time.sleep(pause)

    print(f"\n  Batch {batch_idx+1}/{n_batches} [{batch_start+1}~{batch_end}]:")

    for i_offset, code in enumerate(batch):
        i_global = batch_start + i_offset
        sina = to_sina_code(code)
        ok = False
        for attempt in range(MAX_RETRIES):
            rotate_ua()
            try:
                url = f"http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={sina}&scale=240&ma=no&datalen=1600"
                r = sess.get(url, timeout=15)
                if r.status_code == 403 or r.status_code == 429:
                    backoff = RETRY_BASE * (2 ** attempt) + random.uniform(1, 5)
                    print(f"    [{i_global+1}/{len(TO_FETCH)}] {code}: blocked ({r.status_code}), backoff {backoff:.1f}s")
                    time.sleep(backoff)
                    continue
                if r.status_code != 200 or not r.text.strip():
                    raise Exception(f"HTTP {r.status_code}")
                data = json.loads(r.text)
                if not data:
                    raise Exception("empty response")
                df = pd.DataFrame(data)
                df = df.rename(columns={"day": "date"})
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
                for col in ["open", "close", "high", "low", "volume"]:
                    df[col] = df[col].astype(float)
                df = df[["open", "close", "high", "low", "volume"]]
                df["factor"] = 1.0
                df = df[df.index >= "2020-01-01"]
                if len(df) < 10:
                    raise Exception(f"too few rows ({len(df)})")
                all_data[code] = df
                print(f"    [{i_global+1}/{len(TO_FETCH)}] {code}: {len(df)} rows ({df.index[0].date()} ~ {df.index[-1].date()})")
                ok = True
                break
            except Exception as e:
                err = str(e)[:60]
                if attempt < MAX_RETRIES - 1:
                    backoff = RETRY_BASE * (2 ** attempt) + random.uniform(0.5, 2)
                    print(f"    [{i_global+1}/{len(TO_FETCH)}] {code}: attempt {attempt+1}/{MAX_RETRIES} - {err}, retry in {backoff:.1f}s")
                    time.sleep(backoff)
                else:
                    print(f"    [{i_global+1}/{len(TO_FETCH)}] {code}: FAILED after {MAX_RETRIES} attempts - {err}")
        if not ok:
            failed_codes.append(code)
        random_delay()

print(f"\n  Fetched: {len(all_data)}/{len(TO_FETCH)}")
if failed_codes:
    print(f"  Failed ({len(failed_codes)}): {failed_codes[:20]}{'...' if len(failed_codes) > 20 else ''}")
if len(all_data) == 0:
    print("  ERROR: No new stocks fetched, aborting")
    sys.exit(1)

# ============================================================
# Convert to qlib binary (INCREMENTAL)
# ============================================================
import pickle

print()
print("=" * 60)
print("Step 2: Convert to qlib binary format (incremental)")
print("=" * 60)

os.makedirs(DATA_DIR, exist_ok=True)

# --- Load existing calendar or create new one ---
cal_path = os.path.join(DATA_DIR, "calendars", "day.txt")
if os.path.exists(cal_path):
    with open(cal_path, "r") as f:
        calendar_str = [line.strip() for line in f if line.strip()]
    calendar = [pd.Timestamp(d) for d in calendar_str]
    print(f"  Existing calendar: {calendar_str[0]} ~ {calendar_str[-1]} ({len(calendar_str)} days)")
else:
    calendar = []
    calendar_str = []

# Check if new stocks introduce dates outside existing calendar
date_to_idx = {d: i for i, d in enumerate(calendar)}
new_dates = set()
for df in all_data.values():
    for dt in df.index:
        if dt not in date_to_idx:
            new_dates.add(dt)

if new_dates:
    # Merge new dates into calendar and rebuild all binaries
    print(f"  WARNING: {len(new_dates)} new trading dates found. Rebuilding full calendar.")
    calendar = sorted(set(calendar) | new_dates)
    calendar_str = [d.strftime("%Y-%m-%d") for d in calendar]
    date_to_idx = {d: i for i, d in enumerate(calendar)}

    # Rewrite calendar file
    os.makedirs(os.path.join(DATA_DIR, "calendars"), exist_ok=True)
    with open(cal_path, "w") as f:
        f.write("\n".join(calendar_str) + "\n")

    # Rebuild existing stock binaries with new calendar
    n_days = len(calendar_str)
    rebuilt = 0
    for stock_dir_name in os.listdir(features_dir):
        sd_path = os.path.join(features_dir, stock_dir_name)
        if not os.path.isdir(sd_path):
            continue
        # Read existing close.day.bin to get start_index and old data
        close_path = os.path.join(sd_path, "close.day.bin")
        if not os.path.exists(close_path):
            continue
        old_data = np.fromfile(close_path, dtype=np.float32)
        old_start = int(old_data[0])
        old_vals = old_data[1:]

        # Old calendar subset
        old_cal = calendar[old_start:old_start + len(old_vals)]
        if len(old_cal) != len(old_vals):
            continue  # shouldn't happen

        new_start = old_start  # same start index (calendar only grew)

        # For each feature, remap
        for feat in ["open", "close", "high", "low", "volume", "factor"]:
            fpath = os.path.join(sd_path, f"{feat}.day.bin")
            if not os.path.exists(fpath):
                continue
            old_feat = np.fromfile(fpath, dtype=np.float32)
            old_feat_vals = old_feat[1:]

            new_arr = np.full(n_days - new_start, np.nan, dtype=np.float32)
            for j, dt in enumerate(old_cal):
                local_idx = date_to_idx[dt] - new_start
                if 0 <= local_idx < len(new_arr) and j < len(old_feat_vals):
                    new_arr[local_idx] = old_feat_vals[j]

            out = np.concatenate([[np.float32(new_start)], new_arr])
            out.tofile(fpath)
        rebuilt += 1

    print(f"  Rebuilt {rebuilt} existing stocks with expanded calendar ({n_days} days)")
else:
    n_days = len(calendar_str)
    print(f"  Calendar unchanged: {n_days} days")

# --- Write new stock features ---
os.makedirs(features_dir, exist_ok=True)
for code, df in all_data.items():
    stock_dir = os.path.join(features_dir, to_qlib_dir(code))
    os.makedirs(stock_dir, exist_ok=True)
    first_idx = date_to_idx[df.index[0]]
    n_stock_days = n_days - first_idx
    for feat in ["open", "close", "high", "low", "volume", "factor"]:
        arr = np.full(n_stock_days, np.nan, dtype=np.float32)
        for dt, row in df.iterrows():
            if dt in date_to_idx:
                local_idx = date_to_idx[dt] - first_idx
                arr[local_idx] = row[feat]
        out = np.concatenate([[np.float32(first_idx)], arr])
        out.tofile(os.path.join(stock_dir, f"{feat}.day.bin"))

# --- Append to instruments ---
inst_path = os.path.join(DATA_DIR, "instruments", "all.txt")
os.makedirs(os.path.dirname(inst_path), exist_ok=True)
with open(inst_path, "a") as f:
    for code, df in all_data.items():
        f.write(f"{to_qlib_code(code)}\t{df.index[0].strftime('%Y-%m-%d')}\t{df.index[-1].strftime('%Y-%m-%d')}\n")

# --- Update metadata pickle ---
meta_path = r"C:\Users\szk220009\AppData\Local\Temp\opencode\sina_meta.pkl"
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
else:
    meta = {"prices": {}, "last_date": "", "next_td": "", "n_stocks": 0, "n_days": 0}

meta["prices"].update({code: float(df["close"].iloc[-1]) for code, df in all_data.items()})
meta["last_date"] = calendar_str[-1]
meta["next_td"] = (pd.Timestamp(calendar_str[-1]) + pd.tseries.offsets.BDay(1)).strftime("%Y-%m-%d")
meta["n_stocks"] = meta.get("n_stocks", 0) + len(all_data)
meta["n_days"] = n_days

with open(meta_path, "wb") as f:
    pickle.dump(meta, f)

total_stocks = meta["n_stocks"]
print(f"  New stocks written: {len(all_data)}")
print(f"  Total stocks now:   {total_stocks}")
print(f"  Calendar: {calendar_str[0]} ~ {calendar_str[-1]} ({n_days} days)")
print(f"  Metadata saved: {meta_path}")
print()
print("DONE. Now run sina_train.py for training + prediction.")
