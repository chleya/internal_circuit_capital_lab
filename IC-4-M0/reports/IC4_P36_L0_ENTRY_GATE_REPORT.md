# IC4-P36: L0 Entry Gate — 入口级 vs 深层干预对幻觉控制的有效性

**日期**: 2026-05-26
**实验队列**: Masterplan Experiment C → P36
**前序依赖**: P30, P31, P32, P33, P34, P35
**状态**: Complete

---

## 1. Motivation

P30-P35 已系统性地建立了以下事实:
- **P30**: L0 combined ablation 对 funding token 的 lp_diff Δ=+0.339, L3 仅 +0.008 (40x)
- **P31**: 该规律在 funding 和 r_and_d_spend 上均成立 (40x, 190x)
- **P32**: L0 效应 ≈ embedding 消融 → L0 是 embedding→residual 的管道
- **P33**: Head 1 纯自聚焦 (1.000) — L0 的关键编码机制
- **P34**: multi-token tokenization 稀释信息浓度
- **P35**: L16 次高峰来自 FFN 知识检索

**未回答的核心问题**: 既然我们知道 L0 是幻觉信息的入口点，那么：
1. **在 L0 入口处干预是否比在深层（语义已形成）进行干预更有效？**
2. **log-prob 层面的效果能否转化为实际的生成行为变化？**

P36 直接测试这两个问题。

---

## 2. Experimental Design

### 2.1 干预条件 (10 种)

| 标签 | 描述 |
|------|------|
| A_baseline | 无干预，基准 |
| B_embed_ablate | Embedding 消融 at causal pos (上界) |
| C_L0_combined | L0 attn + mlp 联合消融 |
| D_L0_attn_only | L0 attention-only 消融 |
| E_L0_mlp_only | L0 MLP-only 消融 |
| F_L16_combined | L16 attn + mlp 联合消融 (深层对比) |
| G_L20_combined | L20 attn + mlp 联合消融 (更深层对比) |
| H_non_causal | 非因果位置对照 (消融邻近但非因果 token) |
| I_random_pos | 随机位置对照 |
| J_L0_L16_joint | L0 + L16 联合消融 (入口+检索双重阻断) |

### 2.2 Token 族

从数据中自动筛选出包含目标 token 的 unanswerable 样本:
- **funding**: 8 samples (base lp_diff=+0.631 → 强幻觉倾向)
- **r_and_d_spend**: 2 samples (base lp_diff=-0.134 → 弱幻觉/偏 abstain)
- **revenue**: 3 samples (base lp_diff=-0.126 → 弱幻觉/偏 abstain)

### 2.3 评估指标

**主指标**: Δlp_diff = lp_diff(干预后) - lp_diff(基准)
- lp_diff = logp(hallucination_response) - logp(abstention_response)
- Δ < 0: 幻觉减少 (hallucination tendency decreased)
- Δ > 0: 幻觉增加

**行为验证** (model.generate(), greedy): 
- hallucination rate (近似: 输出含数字)
- abstention rate (近似: 输出表明 "不知道")
- repetition score

### 2.4 运行环境

- Model: Qwen2.5-0.5B-Instruct
- Device: CPU (float32)
- 数据: 90 unique samples from train_all_s0 + test_{early,mid,late}_s0
- 耗时: ~4.5 min

---

## 3. Results

### 3.1 L0 vs L16 vs L20: lp_diff 干预效果

```
Token           |ΔL0|     |ΔL16|    Ratio     |ΔL20|     |ΔEmbed|
───────────────────────────────────────────────────────────────
funding          0.244     0.060     4.1x      0.130      0.235
r_and_d_spend    0.261     0.015     17.6x     0.011      0.121
revenue          0.019     0.002     11.9x     0.002      0.043
```

**发现 1**: L0 入口干预的 lp_diff 变化幅值在 3 个 token 族上均 >> L16 (4x-17x)。
- funding: 4.1x — 与 P30 的 40x 相比缩小了，因为 baseline mode 不同 (P30 是 hallucinated sample 在 -0.6 baseline 下的反转；此处是 +0.631 baseline 下的压制)
- r_and_d_spend: 17.6x — 与 P31 的 190x 趋势一致但绝对倍数更小
- revenue: 11.9x — 新 token 族验证成功，效应量较小但方向一致

**发现 2**: L20 干预方向不稳定。funding 上 L20 Δ=+0.130 (使幻觉更严重)，r_and_d L20 Δ=-0.011 (微弱减少)。→ 非单调的层间干预效果。

### 3.2 L0 子模块分解

```
Token           Attn Δ    MLP Δ     Combined Δ
─────────────────────────────────────────────────
funding          -0.160    -0.260    -0.244
r_and_d_spend   +0.050    -0.184    -0.261
revenue          -0.060    -0.018    -0.019
```

**发现 3**: MLP 在 L0 效应中占主导地位 (funding 上 MLP 贡献约 106% combined 效应, r_and_d 上约 70%)。

### 3.3 对照

```
Token           L0 Δ      Non-causal Δ   Random Δ     Specificity
──────────────────────────────────────────────────────────────────
funding          -0.244    +0.014          -0.077       -17.5x
r_and_d_spend   -0.261    -0.005          +0.018       -11.3x
```

**发现 4**: 非因果位置和随机位置的干预效果远小于因果 token 位置 → token-position-specific。

### 3.4 L0+L16 联合

```
Token           L0 Δ      L0+L16 Δ    Gain from L16
────────────────────────────────────────────────────
funding          -0.244    -0.232       -0.012 (negligible)
r_and_d_spend   -0.261    -0.267       -0.006 (negligible)
```

**发现 5**: L0+L16 联合消融相比于纯 L0 消融几乎没有额外收益。→ L0 是主要瓶颈，L16 的语义检索贡献在 L0 被阻断后不独立显现。

### 3.5 行为验证 (generate) — **关键阴性结果**

```
Intervention       Hall Rate   Abst Rate   Rep Score
─────────────────────────────────────────────────────────
A_baseline           4/5          1/5        0.026
B_embed_ablate       5/5          0/5        0.011
C_L0_combined        5/5          0/5        0.017
F_L16_combined       4/5          1/5        0.011
H_non_causal         3/5          2/5        0.000
J_L0_L16_joint       5/5          0/5        0.026
```

**发现 6 (关键阴性)**: **没有任何干预（包括 embedding 消融！）在 model.generate() 的行为层面产生可检测的差异。**

- 所有干预的 hallucination rate 在 3/5 到 5/5 之间，与 baseline 的 4/5 无显著差异
- Embedding 消融（B_embed_ablate）不仅没有减少幻觉，反而使 hallucination rate 从 4/5 上升到 5/5
- Non-causal control 反而显示出最低的 hallucination rate (3/5)，但这可能是小样本 (n=5) 的噪声

**解释**: log-prob 层面的敏感性不直接转化为 autoregressive 生成行为的变化。模型在生成过程中可以通过因果注意力的替代路径绕过单 token 位置的消融。

---

## 4. Discussion

### 4.1 支持 P30-P35 的延伸

P36 的 log-prob 结果与 P30-P35 的发现一致：L0 是 token 信息进入残差流的关键入口点，在此处干预对下游语义处理的影响远大于深层干预。3 个不同的 token 族（funding, r_and_d_spend, revenue）均表现出 L0 >> L16/L20 的模式，增强了结论的泛化性。

### 4.2 L20 的非单调性

L20 干预在 funding 上反而加剧了幻觉（Δ=+0.130），这是一个意外发现。可能的机制：L20 的 FFN 包含反幻觉知识，消融它移除了模型的 "自我纠错" 能力。这与 P35 的 L16 FFN 知识检索假说遥相呼应——不同深层的 FFN 可能存储不同方向的知识。

### 4.3 行为控制的失败 — 最重要的发现

> **log-prob 能证明因果性，但 log-prob 上的因果干预不一定能控制行为。**

这是 P36 最关键的击中。它与 masterplan 的判断 #4 一致："probe 能读出不等于系统能控制。"

为什么会失败？

1. **Causal attention masking 的替代路径**: 在 autoregressive generation 中，后续 token 可以通过 causal attention 重新聚合信息，绕过单 token 位置的消融
2. **迭代证据累积**: 生成是迭代过程，每个新 token 都重新计算注意力——单一位置的消融在长序列中可能被 "洗掉"
3. **Embedding 消融 ≠ 信息抹除**: 即使 zero-out embedding，模型仍能从位置信息和上下文推断所需信息（Qwen2.5 使用 RoPE，位置编码独立于 token embedding）

### 4.4 与前人工作的比较

| 实验 | L0 Δ (lp_diff) | Behavioral | 一致性? |
|------|----------------|------------|--------|
| P30 (funding) | ~+0.339 | 未测试 | N/A |
| P31 (r_and_d) | ~+0.15 | 未测试 | N/A |
| P36 (funding) | -0.244* | 5/5 hall (vs 4/5 base) | **行为不一致** |
| P36 (r_and_d) | -0.261* | 未单独测试 | N/A |

*注: P30 的 Δ 符号为正因为在负 baseline 上的 "反转"；P36 的 funding baseline 为正，Δ 为负表示 "压制"。

P36 的 key Δlp_diff 值 (0.244 nats for funding) 与 P30 的 0.339 在量级上可比，但行为测试显示这种 log-prob 级别的因果性无法转化为行为变化。

### 4.5 局限性

1. **小样本行为测试**: n=5 per intervention, total 5 interventions × 5 samples = 25 generations (有限)
2. **分类噪声**: `classify_hallucination_output` 是启发式的（数字检测 + 关键词匹配），不是人工标注
3. **Greedy decoding**: 仅使用 temperature=0，未测试不同 decoding 策略
4. **CPU 限制**: 无法进行大规模超参数搜索或更多生成样本
5. **Single-token ablation**: 可能不够强，需要 multi-position 或 continuous intervention

---

## 5. Verdict

### 一句话判决

**L0 token-entry intervention dominates L16/L20 in log-prob causal effect (4-17x), but ALL interventions fail to affect model.generate() behavior — suggesting a fundamental log-prob → behavior gap that requires generative-circuit (not logit-level) intervention.**

### 对核心判断的影响

| 判断 | P36 支持/反对/无关 |
|------|-------------------|
| 1. 世界不是预先切好的 | 无关 |
| 2. 对象 = action-consequence equivalence class | 弱支持: L0 作为 action-consequence 入口的敏感窗口 |
| 3. 表征不是第一性，结构形成和路由更关键 | **强支持: L0 entry > L16 deep** |
| 4. probe 能读出不等于系统能控制 | **强支持: lp_diff causal but behavior unchanged** |
| 5. readout-level repair 救不了坏结构 | 弱支持: token-level ablation insufficient |
| 6. formation/entry/routing > readout | **支持: L0 entry > L16/L20 deep** |
| 7. TT-SFT cosine 不要原样放大 | 无关 |

---

## 6. 下一步建议

### 最值得做的实验: **P37: Generative-Trajectory Intervention (GTI)**

P36 暴露了核心瓶颈: **log-prob 层面的因果性 ≠ 行为控制**。下一步必须解决这个 gap。

**P37 设计思路** (Masterplan Experiment F 的变体):

1. **不干预 token logits，干预生成轨迹**: 在 autoregressive generation 的每一步，对 hidden state 施加持续干预（而非仅在 causal token 位置一次性消融）
2. **对比三种 timing**:
   - T0: pre-generation (在 causal token 位置消融，= P36 的做法)
   - T1: during-generation (在生成的每个 token 上持续施加 counter-vector)
   - T2: post-generation-gate (在生成完成后对关键 token 进行 re-route)
3. **Counter-vector 设计**: 使用 abstention vs hallucination 的 hidden state difference 作为方向向量
4. **Hypothesis**: T1 (持续干预) > T0 (一次性消融) > T2 (事后纠错)

**Why this experiment**:
- 直接解决 P36 暴露的 gap
- 不需要新环境或新训练
- 可以在现有 IC-4-M0 基础设施上实现
- 如果 T1 有效，将为 generation-level control 提供初步证据
- 如果 T1 也无效，说明需要更根本的 training-time intervention（如 θ 级干预）

**备选**: **P38: Multi-Token Routing Map** — 用 P36 的框架扫描更多 token 族（10+ tokens），建立 L0 entry effect 的 token-type 谱系。回答 "什么类型的 token 在 L0 入口最敏感？"

---

## 7. 交付清单

1. **新增文件**: `src/run_p36_l0_entry_gate.py`
2. **运行命令**: `python src/run_p36_l0_entry_gate.py`
3. **结果目录**: `results_p36_l0_entry_gate/`
   - `results.json` (完整数值结果)
   - `run_log.txt` (运行日志)
4. **报告路径**: `reports/IC4_P36_L0_ENTRY_GATE_REPORT.md` (本文件)

---

*This is a minimal runnable version following the masterplan principle: do the most informative experiment with the least infrastructure, report honestly, and use negative results to inform the next step.*
