# Qlib 量化投资工具集

基于微软 [Qlib](https://github.com/microsoft/qlib) 平台构建的量化选股与预测系统。

## 快速开始

### 环境要求

- Python 3.10+（`.venv` 虚拟环境，不使用 conda）
- pyqlib 0.9.7+
- LightGBM

### 安装

```bash
# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate

# 安装依赖
pip install numpy cython
pip install pyqlib lightgbm
```

### 数据源

| 数据源 | 路径 | 股票数 | 时间范围 | 特征数 | 说明 |
|--------|------|--------|----------|--------|------|
| **qlib_bin** (默认) | `C:/codes/qlib/qlib_bin` | 6,091 | 2000-01 ~ 2026-05-08 | 10 | 社区数据，最新最全 |
| cn_data | `~/.qlib/qlib_data/cn_data` | 3,875 | 1999-11 ~ 2020-09 | 7 | 官方标准数据（已停更） |
| tradingagents | `~/.qlib/qlib_data/tradingagents` | 280 | 2020-12 ~ 2026-05 | 6 | 自建小规模数据 |

**推荐**: 使用 `qlib_bin`，覆盖 6,091 只 A 股（含科创板），数据至 2026-05-08。

下载方式：
```bash
# 社区数据 (qlib_bin)
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
mkdir -p C:/codes/qlib/qlib_bin
tar -zxvf qlib_bin.tar.gz -C C:/codes/qlib/qlib_bin --strip-components=1
```

---

## 自定义脚本

### train.py — 模型训练

使用 LightGBM + Alpha158 特征训练股票收益预测模型。

```bash
# 默认: qlib_bin 全量 (6091 只)
.venv\Scripts\python.exe scripts\train.py

# 指定数据源
.venv\Scripts\python.exe scripts\train.py --data qlib_bin
.venv\Scripts\python.exe scripts\train.py --data cn_data
.venv\Scripts\python.exe scripts\train.py --data tradingagents
```

**参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | `qlib_bin` | 数据源: `qlib_bin` / `cn_data` / `tradingagents` |
| `--model` | `lgbm` | 模型类型（当前仅支持 `lgbm`） |

**训练配置 (qlib_bin)**:

| 阶段 | 时间范围 |
|------|----------|
| 训练集 | 2020-01-01 ~ 2024-12-31 |
| 验证集 | 2025-01-01 ~ 2025-12-31 |
| 测试集 | 2026-01-01 ~ 2026-05-08 |
| 市场 | all (6,091 只) |

**输出**:
- `reports/train_info.json` — 训练元数据（实验名、recorder_id、时间段）
- MLflow 实验记录保存在 `mlruns/`

### predict.py — 股票预测

使用已训练模型对指定股票列表进行预测。

```bash
# 预测所有 21 只目标股票 (默认 qlib_bin)
.venv\Scripts\python.exe scripts\predict.py

# 生成下一交易日预测 (5.11)
.venv\Scripts\python.exe scripts\predict.py --data qlib_bin --latest-day

# 指定股票
.venv\Scripts\python.exe scripts\predict.py --stocks "688041.SH,603986.SH"

# 使用其他数据源的模型
.venv\Scripts\python.exe scripts\predict.py --data cn_data
```

**参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | `qlib_bin` | 数据源（需与训练时一致） |
| `--stocks` | 21 只预设股票 | 逗号分隔的股票列表 |
| `--topk` | 0 (全部) | 只预测前 K 只 |
| `--latest-day` | `False` | 只输出最后一天预测（即下一交易日预测） |

**目标股票列表 (21 只)**:

> 股票列表由 `docs/scripts/stocks_config.py` 统一管理，增删只需修改该文件。

```
688041.SH  688256.SH  688012.SH  603986.SH  688008.SH
300442.SZ  603019.SH  688111.SH  002230.SZ  002837.SZ
002049.SZ  688027.SH  300223.SZ  301269.SZ  002747.SZ
688332.SH  002896.SZ  688568.SH  300672.SZ  300458.SZ
688295.SH
```

**输出**:
- `reports/predictions_YYYYMMDD_HHMMSS.csv` — 预测结果 CSV
- `docs/analysis/prediction_report_YYYYMMDD_HHMMSS.md` — 预测分析报告

---

## 项目结构

```
D:\codes\stock\qlib\
├── .venv/                    # 虚拟环境
├── scripts/
│   ├── train.py              # 模型训练脚本
│   ├── predict.py            # 股票预测脚本（Qlib 纯量化）
│   ├── predict_fused.py      # 融合预测（Qlib + AI 信号）
│   ├── predict_price.py      # 价格预测（短周期）
│   ├── compare_predictions.py # Qlib vs 融合对比
│   ├── rolling_train.py      # 滚动训练
│   ├── get_data.py           # 数据下载
│   ├── dump_bin.py           # 数据转换
│   └── data_collector/       # 数据采集器
├── reports/                  # 训练 & 预测输出
│   ├── train_info.json       # 当前模型元数据
│   ├── predictions_*.csv     # Qlib 预测结果
│   ├── fused_prediction_*.csv   # 融合预测
│   ├── qlib_only_prediction_*.csv # 纯 Qlib 基线
│   └── price_prediction_*.csv    # 价格预测表
├── docs/analysis/            # 分析报告
│   ├── comparison_three_sources.md  # 三数据源对比
│   ├── fused_prediction_*.md        # 融合预测报告
│   └── prediction_report_*.md       # 预测报告
├── mlruns/                   # MLflow 实验存储
├── AGENTS.md                 # AI 代理指引
└── qlib/                     # Qlib 源码（未修改）
```

---

## 模型性能对比

| 指标 | qlib_bin 全量 ⭐ | qlib_bin csi800 | cn_data | tradingagents |
|------|:---:|:---:|:---:|:---:|
| 训练市场 | all (6,091) | csi800 (1,993) | all (3,875) | all (280) |
| 训练样本 | 5,753,171 | 1,944,393 | 5,636,507 | 262,980 |
| IC | **0.051** | 0.028 | 0.124 | 0.001 |
| ICIR | **0.758** | 0.242 | 1.897 | 0.007 |
| 目标覆盖 | **21/21** | **21/21** | 11/21 | 11/21 |
| 数据截止 | **2026-05-08** | **2026-05-08** | 2020-09 | 2026-05 |

> 详细对比见 [`docs/analysis/comparison_three_sources.md`](docs/analysis/comparison_three_sources.md)

---

## 典型工作流

### 1. 训练模型

```bash
.venv\Scripts\python.exe scripts\train.py --data qlib_bin
```

### 2. 预测下一交易日

```bash
.venv\Scripts\python.exe scripts\predict.py --data qlib_bin --latest-day
```

### 3. 查看结果

```bash
# 预测 CSV
type reports\predictions_*.csv

# 分析报告
type docs\analysis\prediction_report_*.md
```

---

## 下一步计划

详见 [`docs/analysis/next_steps_plan.md`](docs/analysis/next_steps_plan.md)

| 阶段 | 任务 | 状态 |
|------|------|------|
| Phase 1 | 5.11 预测 | ✅ 完成 |
| Phase 2 | Alpha360 替代 Alpha158 | 📋 计划中 |
| Phase 3 | 滚动训练 (Rolling Retraining) | 📋 计划中 |
| Phase 4 | 集成学习 (多模型融合) | 📋 计划中 |

---

## 原始 Qlib 脚本

<details>
<summary>数据下载与管理（折叠）</summary>

### Download CN Data

```bash
# daily data
python get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn

# 1min data
python get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data_1min --region cn --interval 1min
```

### Download US Data

```bash
python get_data.py qlib_data --target_dir ~/.qlib/qlib_data/us_data --region us
```

### Download CN Simple Data

```bash
python get_data.py qlib_data --name qlib_data_simple --target_dir ~/.qlib/qlib_data/cn_data --region cn
```

### Using in Qlib

```python
import qlib
from qlib.constant import REG_CN

provider_uri = "~/.qlib/qlib_data/cn_data"
qlib.init(provider_uri=provider_uri, region=REG_CN)
```

### Crowd Sourced Data

<https://github.com/chenditc/investment_data/releases>

```bash
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=2
```

</details>

---

## 融合预测工作流（方案 B）

将 Qlib 量化分数与 TradingAgents AI 信号后处理融合，预测 21 支股票 20 个交易日实际价格。

### 一键运行

```bash
# 全流程（并行 2 组）
python D:\codes\stock\docs\scripts\run_full_pipeline.py --parallel 2

# 已有信号，仅重新预测
python D:\codes\stock\docs\scripts\run_full_pipeline.py --skip-ta

# 分析指定股票
python D:\codes\stock\docs\scripts\run_full_pipeline.py --only 688041.SH 688256.SH
```

### 融合公式

```
combined = w_qlib × qlib_score + w_ai × ai_mapped + w_trader × ta_mapped + w_research × rr_mapped

默认权重: w_qlib=0.50  w_ai=0.25  w_trader=0.10  w_research=0.15
20天衰减: Day1=100% → Day20=12% 线性衰减
```

### 相关脚本

| 脚本 | 说明 |
|------|------|
| `scripts/predict_fused.py` | 核心融合预测（~600 行） |
| `scripts/predict.py` | Qlib 纯量化预测 |
| `scripts/predict_price.py` | 短周期价格预测 |
| `scripts/compare_predictions.py` | 对比分析 |
| `docs/scripts/run_full_pipeline.py` | 全流程入口 |

详细使用指南见 [`docs/scripts/README.md`](../../docs/scripts/README.md)

---

## 免责声明

本项目所有预测结果由量化模型自动生成，仅供参考，不构成任何投资建议。投资有风险，入市需谨慎。
