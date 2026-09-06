# CHANGELOG — qlib 自定义脚本

> 上游 microsoft/qlib 变更不入本文件；仅记录 `scripts/` 自定义脚本的修复/加固。

## 2026-09-06

### fix(predict_fused): w2==0 时强制 kronos_dir=0 —— 零权重信号不投票

- **问题**：`FUSION_WEIGHTS["kronos"]=0.00`（消融独立 IC=-0.160 归零）时，融合循环的
  agreement 投票仍按 `kronos_ret` 符号给 Kronos 计方向票。陈旧缓存（Kronos/data 旧文件）
  使 `kronos_ret` 非零概率高——一个权重为零的信号可拉低 agreement 触发 Qlib 动态降权
  ×0.6，且周一/非周一 TA 覆盖面不同造成口径漂移（docs 团队 t4 审查遗留项）。
- **修复**：`fuse_signals()` 中 `w2 == 0 → kronos_dir = 0`（与 `kronos_ret == 0` 同语义）。
  `w2 > 0`（kronos 启用）时行为不变，kronos_dir 仍正常投票。
- **验证**（rickqi-stock/scripts/tests/test_predict_fused_kronos_dir.py）：
  1. w2=0 且 kronos 缺席（v0.8.6 日常 auto-skip 形态）融合分与改前逐值一致；
  2. 构造形态 S（陈旧正 kronos + Qlib 少数派）：改前降权路径值 ≠ 改后不降权值，加固生效；
  3. w2>0 行为保持改前（kronos 仍投票、降权照常）；
  4. 历史重放：最近 14 个 `fused_prediction_*.csv`（≥10 日要求）368 活跃行降权触发
     差面 **0.00%**（全部行 kronos_ret=0），远低于 10% 降级阈值 → 可安全落地。
- **动机注记**：统一周一/非周一 TA 覆盖口径；在 docs 编排器影子对比窗口开启前落地，
  避免 legacy 融合行为中途漂移造成 diff 基线断裂。
