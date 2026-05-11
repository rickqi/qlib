# AGENTS.md — Qlib 量化选股项目指引

> 本文件为 AI 代理在 `D:\codes\stock\qlib` 项目中高效工作提供高信号上下文。
> 仅收录"不看就会踩坑"的信息，通用开发知识不在此赘述。

---

## 项目概览

基于微软 Qlib（AI 量化投资平台）构建的 A 股选股与预测系统。
- 上游仓库：`microsoft/qlib`（本仓库含 qlib 源码 + 自定义脚本）
- 包名：`pyqlib`（pip），导入名：`qlib`
- Python 3.10+，Windows 环境
- 许可证：MIT

---

## 环境与安装

### 虚拟环境

```bash
# 使用 .venv（不使用 conda）
python -m venv .venv
.venv\Scripts\activate
```

**关键约束**：用户明确要求 `.venv`，不使用 conda。

### 安装依赖

```bash
# 方式一：pip 安装发布版（推荐，无需编译 Cython）
pip install numpy cython
pip install pyqlib lightgbm

# 方式二：从源码安装（开发模式，需编译 .pyx）
pip install numpy cython
pip install -e .[dev,lint,test,analysis]
```

**编译陷阱**：
- Cython `.pyx` 文件（`qlib/data/_libs/rolling.pyx`、`expanding.pyx`）必须先编译为 `.pyd`（Windows）/ `.so`（Linux）
- Windows 上 `pywinpty` 必须用 whl 安装（`--only-binary=:all:`）
- macOS M 系列芯片需 `brew install libomp` 后才能编译 LightGBM

---

## 数据源

| 数据源 | 路径 | 股票数 | 时间范围 | 特征数 | 用途 |
|--------|------|--------|----------|--------|------|
| **qlib_bin** (默认) | `C:/codes/qlib/qlib_bin` | 6,091 | 2000-01 ~ 2026-05-08 | 10 | 主力数据源 |
| cn_data | `~/.qlib/qlib_data/cn_data` | 3,875 | 1999-11 ~ 2020-09 | 7 | 基准参考（数据陈旧） |
| tradingagents | `~/.qlib/qlib_data/tradingagents` | 280 | 2020-12 ~ 2026-05 | 6 | 不推荐（数据太少） |

### qlib_bin 数据结构

```
C:/codes/qlib/qlib_bin/
├── calendars/
│   ├── day.txt          # 过去交易日历（止于 2026-05-08）
│   └── day_future.txt   # 含未来交易日（至 2026-05-29）
├── features/            # 每只股票一个目录（SH600000/、SZ000001/）
└── instruments/
    ├── all.txt          # 6,091 只
    ├── csi300.txt       # 939 只
    ├── csi500.txt       # 1,774 只
    ├── csi800.txt       # 1,993 只
    ├── csi1000.txt      # 2,739 只
    └── csiall.txt       # 5,718 只
```

### 日历与预测边界

- `day.txt` 是特征数据的实际边界，模型只能计算该日历内日期的特征
- `day_future.txt` 包含尚未有数据的未来交易日
- **预测下一交易日**：使用 `day.txt` 最后一天（如 5.8）的特征预测下一交易日（如 5.11）
- 预测脚本 `--latest-day` 参数会自动从 `day_future.txt` 查找下一交易日并标注

---

## 自定义脚本

### 目录约定

| 目录 | 用途 |
|------|------|
| `scripts/` | 所有自定义脚本（train.py、predict.py） |
| `reports/` | 训练元数据 + 预测 CSV 输出 |
| `docs/analysis/` | 分析报告（Markdown） |
| `mlruns/` | MLflow 实验存储（自动生成） |

### train.py — 模型训练

```bash
.venv\Scripts\python.exe scripts\train.py                          # 默认 qlib_bin
.venv\Scripts\python.exe scripts\train.py --data qlib_bin
.venv\Scripts\python.exe scripts\train.py --data cn_data
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | `qlib_bin` | `qlib_bin` / `cn_data` / `tradingagents` |
| `--model` | `lgbm` | 当前仅 `lgbm` |

**qlib_bin 训练配置**：
- 市场：`all`（6,091 只，全量训练）
- 训练集：2020-01-01 ~ 2024-12-31
- 验证集：2025-01-01 ~ 2025-12-31
- 测试集：2026-01-01 ~ 2026-05-08
- 特征：Alpha158（158 维技术指标）
- 输出：`reports/train_info.json`（含 recorder_id，predict.py 依赖此文件）

**已知性能**：全量训练 IC=0.051, ICIR=0.758, 5,753,171 训练样本。

### predict.py — 股票预测

```bash
.venv\Scripts\python.exe scripts\predict.py                              # 全部 3 天预测
.venv\Scripts\python.exe scripts\predict.py --latest-day                 # 仅下一交易日
.venv\Scripts\python.exe scripts\predict.py --stocks "688041.SH,603986.SH"
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | `qlib_bin` | 需与训练时一致 |
| `--stocks` | 21 只预设 | 逗号分隔 |
| `--topk` | 0 (全部) | 只预测前 K 只 |
| `--latest-day` | False | 仅输出最后一天（即下一交易日预测） |

**目标股票**（21 只）：`688041.SH, 688256.SH, 688012.SH, 603986.SH, 688008.SH, 300442.SZ, 603019.SH, 688111.SH, 002230.SZ, 002837.SZ, 002049.SZ, 688027.SH, 300223.SZ, 301269.SZ, 002747.SZ, 688332.SH, 002896.SZ, 688568.SH, 300672.SZ, 300458.SZ, 688295.SH`

**股票代码转换**：用户格式 `688041.SH` → qlib 格式 `SH688041`，`predict.py` 的 `convert_stock_code()` 处理此映射。

### 执行依赖

`predict.py` 依赖 `train.py` 的输出：
1. 先运行 `train.py` → 生成 `reports/train_info.json`
2. 再运行 `predict.py` → 从 `train_info.json` 读取 recorder_id 加载模型

---

## 实踩陷阱

### 内存

- **全量 (all 6091) 训练**：~575 万样本 × 158 特征 ≈ 3.4GB，需 8GB+ 可用内存
- 训练起点 2015-01-01 全量会 OOM，改为 2020-01-01 后成功（~7.5M→5.7M 样本）
- 若 OOM：缩短 `train_start`（如从 2015 改为 2020）或用 csi800 替代 all

### 回测

- `PortAnaRecord` 可能因日历越界抛 `IndexError`，train.py 已用 `try/except (ValueError, KeyError, IndexError)` 捕获
- 回测失败不影响模型训练和信号分析（IC/ICIR 仍正常输出）

### PowerShell 输出

- PowerShell 会将 qlib 的 `INFO` 级别日志输出到 stderr，显示为红色错误文本
- 这是 **正常的**，不影响程序运行
- 判断是否真正报错：看是否有 `Traceback` 或 `Error` 关键字

### 日历与数据边界

- `D.calendar()` 首次调用极慢（~60s），频繁调用应缓存结果
- 预测日期必须在 `day.txt` 范围内，否则特征计算失败
- `day_future.txt` 有未来日历但无数据，不能直接用于预测

### 数据源特性

- **qlib_bin**：唯一覆盖科创板 688xxx 的数据源，21/21 目标股票全覆盖
- **cn_data**：IC 最高（0.124）但数据截止 2020-09，仅覆盖 11/20 股票
- **tradingagents**：IC≈0（随机水平），仅 280 只，不推荐

---

## Qlib 上游代码结构

```
qlib/                     # 主包（未修改）
├── __init__.py           # init() / init_from_yaml_conf() 入口
├── config.py             # 全局配置单例 C
├── data/
│   ├── _libs/            # Cython: rolling.pyx, expanding.pyx
│   ├── dataset/          # DatasetH, Handler 基类
│   ├── ops.py            # 表达式算子（$close, Ref(), Mean() 等）
│   └── storage/          # 数据存储后端
├── contrib/
│   ├── data/
│   │   ├── handler.py    # Alpha158 / Alpha360 Handler
│   │   └── loader.py     # Alpha158DL / Alpha360DL 特征定义
│   ├── model/            # 模型实现（LGBModel, XGBModel, LSTM, Transformer...）
│   ├── rolling/base.py   # 滚动训练 Rolling 类
│   └── report/           # 分析报告生成
├── workflow/             # 实验管理（Recorder + ExpManager, MLflow）
├── model/ens/            # RollingEnsemble 等集成工具
└── backtest/             # 回测引擎
```

### 特征工程

| Handler | 特征数 | 内容 | 适用模型 |
|---------|--------|------|----------|
| Alpha158 | 158 | K线形态 + 价格比 + 滚动统计（ROC/MA/STD/RSV/RSI...） | GBDT |
| Alpha360 | 360 | 6 价格字段 × 60 天归一化序列 | DNN/LSTM + GBDT |

切换方式：`train.py` 中 handler 配置的 `class` 字段（`"Alpha158"` → `"Alpha360"`）。

### 模型注册

所有 qlib 模型通过 `module_path` + `class` 配置加载：
```python
{"class": "LGBModel", "module_path": "qlib.contrib.model.gbdt", "kwargs": {...}}
```
可用模型：LGBModel、XGBModel、CatBoostModel、DoubleEnsemble、Linear、LSTM、GRU、Transformer、ALSTM、GATs 等（见 `qlib/contrib/model/`）。

---

## 代码风格（上游仓库规范）

- **格式化**：Black，行宽 **120**（非默认 88）
- **Docstring**：Numpydoc
- **Pylint**：大量规则已禁用（见 Makefile），合理误报用 `# pylint: disable=XXXX`
- **Mypy**：`qlib/contrib`、`qlib/data`、`qlib/model` 等核心模块已排除（见 `.mypy.ini`），勿在这些目录修 mypy 错误
- **flake8**：忽略 E501/F541/E266/E402/W503/E731/E203
- **import**：部分文件有非顶层 import（E402），勿自动修复
- **Pre-commit**：black + flake8，安装：`pip install -e .[dev] && pre-commit install`

---

## 常用命令速查

```bash
# 训练
.venv\Scripts\python.exe scripts\train.py --data qlib_bin

# 预测下一交易日
.venv\Scripts\python.exe scripts\predict.py --data qlib_bin --latest-day

# 格式检查（上游）
black . -l 120 --check
flake8 --ignore=E501,F541,E266,E402,W503,E731,E203 qlib

# 运行 qlib 原生工作流（需在 examples/ 目录下）
cd examples && qrun benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml
```

---

## 当前模型状态

| 配置 | IC | ICIR | 训练样本 | 目标覆盖 |
|------|:---:|:---:|---------:|:---:|
| qlib_bin all (6091) | 0.051 | 0.758 | 5,753,171 | 21/21 |
| qlib_bin csi800 (1993) | 0.028 | 0.242 | 1,944,393 | 21/21 |
| cn_data all (3875) | 0.124 | 1.897 | 5,636,507 | 11/21 |
| tradingagents all (280) | 0.001 | 0.007 | 262,980 | 11/21 |

---

## CI（上游）

矩阵：`windows-latest` × `ubuntu-22.04` × `ubuntu-24.04` × `macos-14` × `macos-15`，Python 3.8–3.12。
流程：`make dev` → `make lint` → 下载数据 → 运行 LightGBM 工作流 → pytest。

---

## 已知问题

- `pandas ≥ 2.0` 的 `group_keys` 默认值变更可能导致部分脚本报错（TRA/TFT 等第三方 contrib 模块）
- 官方数据集已停更，使用社区数据源：https://github.com/chenditc/investment_data/releases
- `nbqa black` 需要 `black < 26.1`
- Makefile 使用 bash 语法，Windows 需 Git Bash 或 WSL
- RL 模块仅 Linux 受支持
