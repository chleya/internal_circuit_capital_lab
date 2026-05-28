# IC4 P39: Layer Threshold Search — 行为控制需要多少层？

**日期**: 2026-05-26
**状态**: 完成
**耗时**: 28.6 min
**判决**: **negative_but_informative** — 存在"绝望之谷"：1-12 层训练全部恶化幻觉，仅全层训练（P15）有效

---

## 1. 实验目的

P36-P38 建立了一条清晰的证据链：

- L0-only activation 干预 → 失败（P36, P37）
- L0-only weight 干预 → 失败（P38，H 反而恶化）
- All-layer weight 干预 → 成功（P15，hall 46→34）

**P39 核心问题**: 行为控制需要多少层？是在 L0 和全层之间存在一个阈值，还是全层是唯一有效的配置？

---

## 2. 实验设计

### 扫描配置

| 配置 | 层数 | 层范围 | 可训练参数 |
|------|------|--------|-----------|
| Pre (Base) | 0 | — | 0 |
| L0 | 1 | [0] | 45K (0.009%) |
| L0-L2 | 3 | [0,1,2] | 135K (0.027%) |
| L0-L5 | 6 | [0..5] | 270K (0.054%) |
| L0-L11 | 12 | [0..11] | 541K (0.109%) |
| P15 (All) | 24 | [0..23] | ~2.4M (0.5%) |

### 统一控制

- 所有配置使用相同的 30 训练样本，3 epochs，r=8，lr=2e-4，batch=2
- 每个配置独立训练（fresh base model + LoRA + freezing non-target layers）
- 评估：model.generate() 行为分类（n=60）
- P15 数据来自 P38 实验中的实际测量

---

## 3. 结果

### 3.1 主结果表

```
              Pre     L0       L0-L2    L0-L5    L0-L11   P15(All)
              ────    ──       ─────    ─────    ──────   ────────
Hall          46      43       50       49       59       34
Abst          1       2        1        3        0        7
Mixed         4       5        3        4        0        10
Other         9       10       6        4        1        9
ΔHall vs Pre  —       -3       +4       +3       +13      -12
```

### 3.2 关键发现：绝望之谷

```
Hall rate (%)
 90% ┤                                     ● L0-L11 (59, 98.3%)
     │
 80% ┤                 ● L0-L2 (50)  ● L0-L5 (49)
     │         ● Pre (46)
 70% ┤    ● L0 (43)
     │
 60% ┤
     │                                              ● P15 All (34, 56.7%)
 50% ┤
     └─────┬─────┬─────┬─────┬─────┬─────┬─────┬──
          0      4     8     12    16    20    24
                        Layers trained
```

三条规律：

1. **1 层轻微改善** (46→43)：入口点有微弱正向效应
2. **3-12 层持续恶化** (50→59)：更多层 = 更严重的幻觉，12 层时达到 59/60（几乎全部幻觉）
3. **24 层才有效** (34)：只有训练整个模型 depth 才能抑制幻觉

### 3.3 训练 Loss 曲线

| 配置 | Epoch 1 | Epoch 2 | Epoch 3 | 收敛 |
|------|---------|---------|---------|------|
| L0 | 14.67 | 10.34 | 4.99 | 慢，高终值 |
| L0-L2 | 12.15 | 2.38 | 0.25 | 较快 |
| L0-L5 | 9.45 | 0.68 | 0.15 | 快，极低 |
| L0-L11 | 6.98 | 0.19 | 0.10 | 极快，近乎零 |

**Loss 收敛与行为完全负相关**：训练 loss 越低 → hallucination 越高！L0-L11 的 loss=0.10（几乎完美的 CE 训练），但 hall=59/60（几乎全部幻觉）。

---

## 4. 判决与分析

### 判决: **negative_but_informative**

1-12 层 LoRA 训练全部失败。不存在"中间阈值"——要么 L0 轻微改善，要么全层才有效。中间范围（3-12 层）构成一个**绝望之谷**。

### 为什么会这样？

**假说：局部路由失真**

当仅训练前 N 层时（N < 全层）：

1. 训练数据包含 answerable 样本，其正确回答是包含数字的陈述（如 "The company raised $5 million"）
2. 前 N 层学习到 "当看到数字问题时 → 输出具体数字" 的模式
3. 冻结的后续层（N+1 到 23）按照原始参数处理前 N 层的增强信号
4. 结果：前 N 层变得 **更擅长触发数字生成**，但缺乏上层对 "是否应生成数字" 的判别能力
5. 在 unanswerable 问题上，前 N 层自信地生成数字，上层无法阻止 → 更多幻觉

这解释了为什么：
- L0 稍好（仅入口点变化，上层正常处理）
- L0-L11 最差（改变了半个模型的 routing，上层无法纠正）
- P15 恢复（全层训练，上下协同学习何时拒绝）

### 完整证据链总结（P30-P39）

| 实验 | 干预 | 结果 | 含义 |
|------|------|------|------|
| P30-P35 | 探针诊断 | L0 causal entry 发现 | L0 承载因果信息 |
| P36 | L0 ablation → log-prob | Δlog-prob 4-17× | log-prob 可被干预 |
| P36b | L0 ablation → generate() | hall 恶化 | ablation = 破坏性 |
| P37 | L0 counter-vector → generate() | **零效应** | activation-level 入口无效 |
| P38 | L0 LoRA → generate() | **零效应**（LP 恶化） | weight-level 入口无效 |
| P15 | All-layer LoRA → generate() | hall 46→34 ✓ | 全层权重训练有效 |
| **P39** | **1-12 layer LoRA sweep** | **全恶化，hall 43→59** | **入口 + 中间层无效** |

**最终假说修正**:

> L0 是因果信息的入口，但不是行为控制的杠杆。行为控制需要模型 depth 的全链路协同（formation + routing + execution）。任何局部干预（无论权重级还是激活级，无论入口还是中层）都无法重定向 24 层自注意力迭代的涌现结果。

---

## 5. 方法论反思

### 实验设计优点
- 4 个递进 layer range + 2 个 baselines 覆盖完整曲线
- 所有配置统一超参，确保可比性
- Loss 曲线与行为曲线的负相关提供了对 "过拟合但无效" 的直接诊断

### 边界条件
- 仅 30 训练样本，小数据可能加剧中层过拟合
- 未测试非连续层组合（如 L0+L6+L12+L18）
- 仅 CE loss，未尝试 DPO 等对齐目标

---

## 6. 下一步建议

### 短期

**P15 的 generate() 深化分析**：P15 是唯一有效的训练实验，但其原始报告仅评估了 log-prob。需要补一份 P15 对 generate() 行为的完整分析（classification breakdown、per-position、qualitative audit）。

### 中期

鉴于 inference-time + partial-training 路径全部失败，建议转向：

1. **全层 LoRA + 更大训练数据**：将训练样本从 30 扩大到 90+，Epochs 3→5，验证 P15 的 generate 效果是稳定的
2. **DPO 对齐训练**：在 paired hallucination/abstention 数据上做 Direct Preference Optimization，直接优化 "偏好拒答 > 幻觉" 的行为目标，而非 CE 的 token matching

### 理论

"绝望之谷" 现象对 interventional interpretability 有普遍意义：
- 识别 causal entry point ≠ 找到 control lever
- 线性探针可读性 ≠ 控制能力
- 局部权重训练可能产生反效果（路由失真）
- 行为控制可能需要 full-system intervention

这些发现支持了 Masterplan Section 5 提出的 Research Law：
> Structural formation and routing interventions outperform readout repairs when latent usefulness is not already aligned with task execution.
