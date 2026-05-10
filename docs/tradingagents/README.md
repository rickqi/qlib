# Qlib TradingAgents 使用指南

使用 `tradingagents` 数据运行 Qlib 量化投资工作流的完整指南，包括数据采集、模型训练和股票预测。

---

## 目录

- [环境准备](#环境准备)
- [数据概况](#数据概况)
- [文件说明](#文件说明)
- [方式一：Sina 真实数据工作流（推荐）](#方式一sina-真实数据工作流推荐)
  - [数据采集 + 转换](#step-1-数据采集--转换-sina_fetchpy)
  - [训练 + 预测](#step-2-训练--预测-sina_trainpy)
- [方式二：代码化工作流（旧版模拟数据）](#方式二代码化工作流旧版模拟数据)
- [方式三：YAML 配置工作流 (qrun)](#方式三yaml-配置工作流-qrun)
- [Sina 数据获取全链路分析](#sina-数据获取全链路分析)
- [配置参数说明](#配置参数说明)
- [输出解读](#输出解读)
- [自定义扩展](#自定义扩展)
- [常见问题](#常见问题)

---

## 环境准备

### 1. 创建虚拟环境

```powershell
python -m venv C:\Users\szk220009\.qlib\venv
```

### 2. 安装 Qlib

```powershell
cd D:\codes\qlib
C:\Users\szk220009\.qlib\venv\Scripts\pip.exe install -e .
```

> 安装完成后 `qrun.exe` 位于 `C:\Users\szk220009\.qlib\venv\Scripts\`。

### 3. 验证安装

```powershell
C:\Users\szk220009\.qlib\venv\Scripts\python.exe -c "import qlib; print(qlib.__version__)"
```

---

## 数据概况

| 项目 | 值 |
|---|---|
| 数据路径 | `C:/Users/szk220009/.qlib/qlib_data/tradingagents` |
| 数据来源 | 新浪财经日线行情 API（Sina Finance） |
| 日历范围 | 2020-01-02 ~ 2026-05-08 (1535 个交易日) |
| 股票数量 | 20 只 |
| 原始特征 | open, close, high, low, volume, factor (6 个) |
| 扩展特征 | Alpha158 (158 维因子) → 训练集 159 列 |

### 数据目录结构

```
tradingagents/
├── calendars/
│   └── day.txt                  # 交易日历 (每行一个日期, 1535 天)
├── instruments/
│   └── all.txt                  # 股票池 (TSV: symbol \t start \t end)
└── features/
    ├── sh688041/                 # 每只股票一个目录
    │   ├── open.day.bin         # 二进制 float32 (小端序)
    │   ├── close.day.bin
    │   ├── high.day.bin
    │   ├── low.day.bin
    │   ├── volume.day.bin
    │   └── factor.day.bin
    ├── sh603986/
    └── ... (共 20 个目录)
```

### 股票列表

```
SZ002049  SZ002230  SZ002747  SZ002837  SZ002896
SZ300223  SZ300442  SZ300458  SZ300672  SZ301269
SH603019  SH603986  SH688008  SH688012  SH688027
SH688041  SH688111  SH688256  SH688332  SH688568
```

### 股票上市时间分布

| 类型 | 数量 | 示例 | 数据起始 |
|---|---|---|---|
| 完整 6 年 | 13 只 | 688012, 603986, 002230 等 | 2020-01-02 |
| 次新股 (~4 年) | 4 只 | 688256, 688027, 688568 | 2020-07 |
| 新股 (~4 年) | 3 只 | 688041, 301269, 688332 | 2022-07/08 |

---

## 文件说明

```
docs/tradingagents/
├── README.md                     # 本文档
├── src/                          # Sina 真实数据工作流 (推荐)
│   ├── sina_fetch.py             # 数据采集 + qlib 格式转换
│   └── sina_train.py             # 训练 + 预测
├── train.py                      # 旧版训练脚本 (模拟数据)
├── predict.py                    # 旧版预测脚本 (模拟数据)
├── predict_price.py              # 旧版价格预测脚本 (模拟数据)
├── workflow_config.yaml          # qrun YAML 配置
└── output/                       # 预测输出 (自动创建)
    ├── price_pred_2026-05-11_sina.csv     # Sina 数据预测结果
    ├── price_pred_2026-05-11_retrain.csv  # 旧版重训练结果
    └── price_pred_2026-05-11.csv          # 旧版首次预测结果
```

---

## 方式一：Sina 真实数据工作流（推荐）

从新浪财经 API 拉取真实行情数据，转换为 qlib 格式，训练 LightGBM 模型并输出股价预测。

**为什么分两个脚本**：qlib 训练内部使用 joblib 并行计算。Windows 的 multiprocessing 使用 `spawn` 模式，子进程会重新 import 主脚本。若训练和数据拉取在同一文件中，子进程会重新执行数据采集代码，导致文件锁冲突。拆分后 `sina_train.py` 有 `if __name__ == "__main__"` 守卫，避免此问题。

### Step 1: 数据采集 + 转换 (`sina_fetch.py`)

```powershell
cd D:\codes\qlib

C:\Users\szk220009\.qlib\venv\Scripts\python.exe docs\tradingagents\src\sina_fetch.py
```

**执行流程**：

```
1. 禁用代理         → 清除 HTTP_PROXY 环境变量, Session(trust_env=False)
2. 逐只股票请求     → Sina Finance API, 每只最多重试 3 次, 间隔 0.5s
3. JSON 解析        → DataFrame, 字符串转 float, 截断到 2020-01-01 起
4. 生成交易日历     → 20 只股票日期的并集, 排序后写入 calendars/day.txt
5. 生成股票池       → instruments/all.txt (TSV 格式)
6. 写入二进制特征   → features/{stock}/{feat}.day.bin (qlib 格式)
7. 保存元数据       → sina_meta.pkl (收盘价, 日期, 统计)
```

**输出示例**：

```
Step 1: Fetch from Sina Finance API
  [ 1/20] 688041.SH: 892 rows (2022-08-12 ~ 2026-05-08)
  [ 2/20] 688256.SH: 1405 rows (2020-07-20 ~ 2026-05-08)
  ...
  [20/20] 300458.SZ: 1535 rows (2020-01-02 ~ 2026-05-08)
  Success: 20/20

Step 2: Convert to qlib binary format
  Calendar: 2020-01-02 ~ 2026-05-08 (1535 days)
  Stocks: 20
```

### Step 2: 训练 + 预测 (`sina_train.py`)

```powershell
C:\Users\szk220009\.qlib\venv\Scripts\python.exe docs\tradingagents\src\sina_train.py
```

**执行流程**：

```
1. 加载元数据       → sina_meta.pkl (收盘价, 日期信息)
2. qlib.init()      → 挂载 tradingagents 数据
3. 配置 joblib      → threading 后端 (避免 Windows spawn 问题)
4. 训练 Model A     → CSRankNorm 标签 + LightGBM (排序模型)
5. 训练 Model B     → Raw Return 标签 + LightGBM (收益预测模型)
6. 合并预测         → pred_price = current_close × (1 + pred_return)
7. 输出 CSV         → 按预测涨跌幅排序, 含涨停/跌停价
```

**双模型设计**：

| 模型 | 标签处理 | 用途 | 分割策略 |
|---|---|---|---|
| Model A (CSRankNorm) | 截面排名标准化 | 提供排名分数 (csrank_score) | train 20-23, valid 24, test 25-26.04 |
| Model B (Raw Return) | 原始收益率 | 直接预测涨跌幅 (pred_return) | train 20-25.04, valid 25.05-26.03, test 26.04-26.05 |

**输出示例**：

```
  Price Prediction — Data through: 2026-05-08 -> Predict: 2026-05-11
  Source: Sina Finance daily | IC=-0.0162  Rank IC=-0.0129 | 20 stocks
  ====================================================================================================

     #  Code               Close      Rank      Change   PredPrice     LimitUp     LimitDn
  -------------------------------------------------------------------------------------
     1  002896.SZ          76.00    0.1030      +0.96%       76.73       83.60       68.40
     2  688111.SH         271.26    0.0950      +0.96%      273.87      298.39      244.13
     ...
    20  688008.SH         210.27    0.0824      -0.08%      210.11      231.30      189.24

  Saved: D:\codes\qlib\docs\tradingagents\output\price_pred_2026-05-11_sina.csv
```

---

## Sina 数据获取全链路分析

### 数据源 API

```
URL: http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData
```

| 参数 | 值 | 含义 |
|---|---|---|
| `symbol` | `sh688041` | 市场小写前缀 + 代码 |
| `scale` | `240` | K线周期 = 240 分钟 = 日线 |
| `ma` | `no` | 不返回均线数据 |
| `datalen` | `1600` | 返回最近 1600 个交易日 (~6.5 年) |

**API 响应格式** (JSON, GBK 编码):

```json
[
  {"day":"2026-05-08","open":"333.880","high":"338.560","low":"320.010","close":"323.280","volume":"49970595"}
]
```

- 6 个字段: `day`, `open`, `high`, `low`, `close`, `volume`
- 所有值均为**字符串**（需手动转 float）
- 按时间倒序返回（脚本通过 `sort_index()` 改为正序）

### 股票代码转换

脚本中定义了三种格式间的转换：

```
原始格式: "688041.SH"  (Wind/通达信格式, 代码中 RAW_CODES)
新浪格式: "sh688041"   (API 请求用, to_sina_code())
qlib目录: "sh688041"   (features 下的目录名, to_qlib_dir())
qlib代码: "SH688041"   (instruments 文件中, to_qlib_code())
```

### 网络层：代理绕过

```python
# 1. 清除环境变量
os.environ["HTTP_PROXY"] = ""

# 2. 创建无代理 Session
sess = requests.Session()
sess.trust_env = False          # 不读取系统代理设置
sess.proxies = {"http": None, "https": None}  # 显式禁用
```

系统配置了 Clash/V2Ray 代理 `127.0.0.1:7890` 但软件未运行，不绕过会导致 `ConnectionRefused`。

### qlib 二进制特征文件格式

每个 `.day.bin` 文件格式:

```
┌──────────────┬────────┬────────┬─────┬────────┐
│ start_index  │  val₀  │  val₁  │ ... │  valₙ  │
│  (float32)   │(float32)│(float32)│     │(float32)│
└──────────────┴────────┴────────┴─────┴────────┘
   4 bytes      4 bytes  4 bytes        4 bytes
```

- `start_index`: 该股票数据在日历中的起始位置 (float32 编码的整数索引)
- `val₀..valₙ`: 从 start_index 开始，日历每个位置对应的特征值
- 未上市或停牌日期填 `NaN`

**示例** — SH688041 (2022-08-12 上市):

```
文件: features/sh688041/close.day.bin
start_index = 633.0   → 日历第 633 个位置 (2022-08-12)
总长度 = 1 + (1535 - 633) = 903 个 float32
val[0] = 60.10 (上市首日收盘价)
val[901] = 323.28 (2026-05-08 收盘价)
NaN: 10 个 (停牌日)
```

**为什么需要 start_index**: qlib 的 `file_storage.py` 通过前 4 字节定位数据偏移:

```python
# qlib/data/storage/file_storage.py
index = int(np.frombuffer(fp.read(4), dtype="<f")[0])  # 读 start_index
data = np.frombuffer(fp.read(...), dtype="<f")           # 读数据
# data[i] 对应 calendar[start_index + i]
```

### 完整数据流

```
sina_fetch.py                   sina_meta.pkl              sina_train.py
┌───────────────────┐           ┌─────────────┐           ┌──────────────────┐
│ Sina API          │           │ prices{}    │           │ qlib.init()      │
│   ↓ JSON          │           │ last_date   │           │   ↓              │
│ DataFrame         │           │ next_td     │  pickle   │ Alpha158 Handler │
│   ↓ filter ≥ 2020 │           │ n_stocks    │ ◄───────► │   ↓ 158 factors  │
│ 20 DataFrames     │           │ n_days      │           │ DatasetH         │
│   ↓               │           └─────────────┘           │   ↓ segments     │
│ qlib binary:      │                                     │ LGBModel.fit()   │
│  calendars/       │                                     │   ↓ predict      │
│  instruments/     │                                     │ prices × (1+ret) │
│  features/*.bin   │                                     │   ↓              │
│   ↓               │                                     │ CSV output       │
│ sina_meta.pkl ────┼──────┐                               └──────────────────┘
└───────────────────┘      │
                           │  传递: 当前收盘价, 日期信息
                           └─────────────────────────────────►
```

---

## 方式二：代码化工作流（旧版模拟数据）

> **注意**: 此方式使用旧版模拟数据（非真实行情），数据质量较差。推荐使用方式一。

### 训练模型

```powershell
cd D:\codes\qlib

C:\Users\szk220009\.qlib\venv\Scripts\python.exe docs\tradingagents\train.py
```

**执行流程：**

```
1. qlib.init()           → 挂载 tradingagents 数据
2. 数据探索              → 打印日历/股票/特征信息
3. 创建 Alpha158 Dataset → 6 个原始特征扩展为 158 维因子
4. Model.fit()           → LightGBM 训练 (含 early stopping)
5. SignalRecord          → 生成预测信号
6. SigAnaRecord          → 计算 IC, Rank IC 等指标
7. PortAnaRecord         → 回测 (TopkDropout 策略)
8. 保存模型              → 写入 mlflow 实验记录
```

### 预测下一交易日

```powershell
C:\Users\szk220009\.qlib\venv\Scripts\python.exe docs\tradingagents\predict.py
```

---

## 方式三：YAML 配置工作流 (qrun)

一步完成训练 + 回测：

```powershell
cd D:\codes\qlib\examples

C:\Users\szk220009\.qlib\venv\Scripts\qrun.exe ..\docs\tradingagents\workflow_config.yaml
```

**YAML 配置结构：**

```yaml
qlib_init:                    # 数据初始化
    provider_uri: "C:/Users/szk220009/.qlib/qlib_data/tradingagents"
    region: cn

task:
    model:                    # 模型定义
        class: LGBModel
        ...
    dataset:                  # 数据集定义
        class: DatasetH
        handler:
            class: Alpha158   # 因子集
        segments:             # 时间分割
            train: [...]
            valid: [...]
            test:  [...]
    record:                   # 分析记录
        - SignalRecord        # 预测信号
        - SigAnaRecord        # IC 分析
        - PortAnaRecord       # 回测分析
```

---

## 配置参数说明

### 模型参数 (LightGBM)

#### Model A — CSRankNorm (排序模型)

| 参数 | 值 | 说明 |
|---|---|---|
| `loss` | `mse` | 损失函数 |
| `learning_rate` | `0.1` | 学习率 |
| `max_depth` | `8` | 树最大深度 |
| `num_leaves` | `128` | 叶节点数 |
| `n_estimators` | `300` | 最大迭代数 |
| `early_stopping_rounds` | `50` | 早停轮数 |
| `colsample_bytree` | `0.8` | 列采样比 |
| `subsample` | `0.8` | 行采样比 |

#### Model B — Raw Return (收益预测模型)

| 参数 | 值 | 说明 |
|---|---|---|
| `learning_rate` | `0.05` | 学习率 (更低, 更保守) |
| `max_depth` | `6` | 树最大深度 (更浅) |
| `num_leaves` | `80` | 叶节点数 (更少) |
| `n_estimators` | `500` | 最大迭代数 (更多) |
| `early_stopping_rounds` | `100` | 早停轮数 (更长容忍) |

### 数据处理

| 步骤 | 说明 |
|---|---|
| `Alpha158` | 从 6 个原始 OHLCV+factor 特征计算 158 个量化因子 |
| `RobustZScoreNorm` | 基于 median/IQR 的标准化, 裁剪异常值 |
| `Fillna` | 填充缺失值 (停牌日的 NaN) |
| `DropnaLabel` | 丢弃标签为空的样本 |
| `CSRankNorm` | 截面排序标准化 (每日股票间排名, 仅 Model A) |

### Label 定义

```python
label = "Ref($close, -2) / Ref($close, -1) - 1"
```

含义：T+1 收盘价到 T+2 收盘价的收益率。即基于 T 日数据，预测**下一个交易日**的收益。

---

## 输出解读

### 信号分析指标

| 指标 | 含义 | 好的范围 |
|---|---|---|
| **IC** | 预测值与实际收益的相关系数 | > 0.05 |
| **ICIR** | IC 的均值/标准差 | > 0.5 |
| **Rank IC** | 秩相关系数 | > 0.05 |
| **Rank ICIR** | Rank IC 的均值/标准差 | > 0.5 |

### 当前模型表现

| 指标 | 值 | 评价 |
|---|---|---|
| IC | -0.0162 | 接近零, 模型几乎无预测能力 |
| Rank IC | -0.0129 | 同上 |
| Early stop | Model A @ iter 1, Model B @ iter 17 | 验证集损失迅速恶化 |

**根因**: 20 只股票的截面 (cross-section) 太窄。CSRankNorm 每天只有 20 个样本做排名, Alpha158 的 158 维特征相对于训练样本量 (~20000) 严重过参数化。LightGBM 几乎无法学到有效信号。

**改进建议**:
1. 增加股票数量 (100+ 只) — 提升截面宽度
2. 使用 PyTorch 模型 (LSTM / Transformer)
3. 使用 `Alpha360` 替代 `Alpha158`
4. 减少特征维度或增加正则化

### CSV 输出字段

| 字段 | 含义 |
|---|---|
| `code` | 股票代码 (Wind 格式) |
| `current_close` | 最新收盘价 |
| `csrank_score` | Model A 截面排名分数 |
| `pred_return` | Model B 预测收益率 |
| `pred_change_pct` | 预测涨跌幅 (%) |
| `pred_price` | 预测价格 = current_close × (1 + pred_return) |
| `limit_up` | 涨停价 (当前价 × 1.1) |
| `limit_down` | 跌停价 (当前价 × 0.9) |

---

## 自定义扩展

### 使用不同模型

在 `sina_train.py` 中修改模型配置：

```python
# XGBoost
"model": {
    "class": "XGBModel",
    "module_path": "qlib.contrib.model.xgboost",
    "kwargs": {"n_estimators": 500, "max_depth": 6},
}

# MLP (需安装 pytorch)
"model": {
    "class": "MLP",
    "module_path": "qlib.contrib.model.pytorch_mlp",
    "kwargs": {"d_hidden": 128, "n_epochs": 200},
}

# LSTM (需安装 pytorch)
"model": {
    "class": "LSTM",
    "module_path": "qlib.contrib.model.pytorch_lstm_ts",
    "kwargs": {"d_hidden": 64, "n_epochs": 200},
}
```

### 使用不同因子集

```python
# Alpha360 (360 个原始价格特征, 适合深度学习模型)
"handler": {
    "class": "Alpha360",
    "module_path": "qlib.contrib.data.handler",
    ...
}
```

### 修改 Label

```python
# 预测 5 日收益
"label": ["Ref($close, -6) / Ref($close, -1) - 1"]

# 预测超额收益 (相对 benchmark)
"label": ["Ref($close, -2) / Ref($close, -1) - 1 - Ref($bench, -2) / Ref($bench, -1) + 1"]
```

### 添加更多股票

在 `sina_fetch.py` 的 `RAW_CODES` 列表中添加代码即可：

```python
RAW_CODES = [
    "688041.SH", "688256.SH", ...
    "600519.SH",  # 新增股票
    "000858.SZ",  # 新增股票
]
```

---

## 常见问题

### Q: 运行时报 `ModuleNotFoundError: No module named 'qlib.tests.config'`

**A:** 必须在 `D:\codes\qlib` 目录下运行脚本（editable install 的限制）。

```powershell
cd D:\codes\qlib
```

### Q: Sina API 请求失败 / 超时

**A:** 可能是系统代理干扰。脚本已自动绕过代理 (`trust_env=False`)，如果仍然失败：
1. 确认网络可直连 `money.finance.sina.com.cn`
2. 检查是否有防火墙拦截
3. 每只股票最多重试 3 次，间隔 2 秒

### Q: 训练报 `RuntimeError: An attempt has been made to start a new process...`

**A:** Windows multiprocessing spawn 问题。确保：
1. `sina_train.py` 有 `if __name__ == "__main__":` 守卫
2. 使用 `C.joblib_backend = "threading"` 而非默认的 `multiprocessing`
3. 不要用 `exec()` 在另一个脚本内执行训练代码

### Q: 训练报 `ValueError: cannot convert float NaN to integer`

**A:** qlib 二进制文件格式错误。每个 `.day.bin` 文件的第一个 float32 必须是有效的 start_index (日历位置), 不能是 NaN。检查 `sina_fetch.py` 中是否有 `[np.float32(first_idx)]` 前缀。

### Q: 回测报 `IndexError: index 1535 is out of bounds`

**A:** 回测 `end_time` 不能设为日历最后一天，需提前 1-2 天。

```python
TEST_END = "2026-04-30"  # 日历最后是 2026-05-08
```

### Q: 如何更新数据到最新？

**A:** 直接重新运行 `sina_fetch.py`，它会自动清除旧数据并拉取最新行情：

```powershell
cd D:\codes\qlib
C:\Users\szk220009\.qlib\venv\Scripts\python.exe docs\tradingagents\src\sina_fetch.py
C:\Users\szk220009\.qlib\venv\Scripts\python.exe docs\tradingagents\src\sina_train.py
```

### Q: 预测得分都差不多，区分度不高

**A:** 当前只有 20 只股票，数据量偏少。建议：
1. 增加股票数量（修改 `RAW_CODES` 列表）
2. 使用 PyTorch 模型 (LSTM / Transformer)
3. 调整 `learning_rate` 和 `num_leaves`
4. 使用 `Alpha360` 替代 `Alpha158`

### Q: 虚拟环境在哪里？

```
C:\Users\szk220009\.qlib\venv\
├── Scripts\
│   ├── python.exe         # Python 解释器
│   ├── pip.exe            # 包管理器
│   └── qrun.exe           # qlib 命令行工具
└── Lib\
    └── site-packages\     # 已安装的包
```
