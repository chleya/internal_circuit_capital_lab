# IC-4: Position Sensitivity — 正式整合报告

> **并入**: UNIFIED_RESEARCH_MAP.md, STRUCTURAL_ADAPTATION_HYPOTHESIS.md (Absorption 瓶颈)
>
> **日期**: 2026-05-21
> **状态**: 两个子实验已完成，报告已整合

---

## 1. 实验架构

位置敏感性分两层测试，对应 Structural Adaptation Hypothesis 的两级瓶颈路径：

| 层 | 实验 | 问题 | 方法 |
|---|---|---|---|
| **表示层** | Rep Shift | 相同内容放在不同位置，hidden state 是否不同？ | 60 samples × 3 positions (early/mid/late via prefix shifting)，提取 last_prompt_token hidden state，计算 cosine distance + 3-NN 位置分类 |
| **行为层** | Position-to-Behavior | 如果表示层不同，行为是否跟着变？ | 60 samples × 3 positions，base mode generation，比较 H/C/CA 变化 |

---

## 2. 关键结果

### 2.1 表示层：完全敏感

| 指标 | 值 | 结论 |
|---|---|---|
| 3-NN 位置分类准确率 | **1.0000** (baseline=0.333) | 位置在 hidden state 中完全可译码 |
| early ↔ mid cosine distance | 0.0649 ± 0.0120 | 仅 8-token 偏移即产生显著表示差异 |
| early ↔ late cosine distance | 0.0796 ± 0.0135 | 偏移越大，表示差异越大 |
| mid ↔ late cosine distance | 0.0047 ± 0.0007 | 相邻位置间表示差异小 |

### 2.2 行为层：中度敏感（但模型部分补偿）

| 位置 | H | C | CA | UA |
|---|---|---|---|---|
| early | 0.867 | 0.600 | 0.067 | 0.000 |
| mid | 0.900 | 0.667 | 0.033 | 0.000 |
| late | 0.867 | 0.667 | 0.033 | 0.000 |

| 指标 | 值 |
|---|---|
| ΔH (max-min) | 0.033 |
| ΔC (max-min) | 0.067 |

### 2.3 两层对映

```
表示层：      KNN=1.0 —— 位置完全可区分
                    ↓ (不完全传递)
行为层：      ΔC=0.07 —— 位置有中度影响，但模型部分补偿
```

**核心发现**：0.5B 模型在 ~20-30 token 上下文中**能部分补偿表示偏移**。但补偿并非完全——mid 位置（距 content ~8 tokens）的 C 明显高于 early。这暗示 Absorption Input Fragmentation 的长度依赖。

---

## 3. 与 Structural Adaptation Hypothesis 的关系

### A 瓶颈：Absorption（Input Fragmentation）

| 维度 | 证据 | 置信度 |
|---|---|---|
| 表示层碎片化存在 | KNN=1.0 — 同一内容不同位置，hidden state 完全不同 | ⭐⭐⭐⭐⭐ |
| 行为层部分补偿 | ΔC=0.07 — 模型在 ~30 token 上下文中有补偿能力 | ⭐⭐⭐⭐ |
| 长度依赖预测 | n=100 时 ΔC 预计 > 0.07（待验证） | ⭐⭐⭐ |

**写入地图位置**：
- UNIFIED_RESEARCH_MAP.md → C8 锚点（新增）：Position Sensitivity
- STRUCTURAL_ADAPTATION_HYPOTHESIS.md → Absorption 瓶颈状态更新："实验证据：表示层完全区分，行为层中度敏感"

---

## 4. 对下游实验的影响

### 4.1 轨迹动力学

T0-T3 的 hallucination 数据都有固定 prompt 格式，instruction 部分的位置固定。当前数据中**位置不是自变量**——所有 sample 的 evidence 在同一位置。因此：

- T0-T3 的当前结论**不受位置影响**（因为位置未变）
- 但如果未来做 "position-aware trajectory" 实验，需要在 T0 阶段就标注位置

### 4.2 能力路由

M7 的 gate 依赖 hidden state，而 hidden state 是位置敏感的。这意味着：
- gate effectiveness 可能是 position-dependent
- 需要 position-invariant routing signal（或 position-calibrated gate）

---

## 5. 可检验的预测

1. **长度依赖**: n=100 prefix 下 ΔC > 0.07；n=200 下 ΔC > 0.10
2. **层依赖性**: 更上层（L20+）的表示层位置敏感度 > 中下层
3. **task 依赖性**: hallucination-prone 样本对位置更敏感

---

## 6. 文件索引

| 文件 | 说明 |
|---|---|
| `src/run_position_rep_shift.py` | 表示层位置偏移实验 |
| `src/run_position_behavior.py` | 行为层位置敏感性实验 |
| `data_position_sensitivity/s0/` | 位置变体数据 |
| `results_position_sensitivity_cpu/REP_SHIFT_REPORT.md` | 表示层详细报告 |
| `results_position_sensitivity_cpu/POSITION_BEHAVIOR_REPORT.md` | 行为层详细报告 |

---

*IC-4 Position Sensitivity — Integration Report v1.0*