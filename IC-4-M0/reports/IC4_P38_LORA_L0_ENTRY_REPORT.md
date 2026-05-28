# IC4 P38: LoRA on L0 Entry Point — Weight-L0 vs Activation-L0

**日期**: 2026-05-26
**状态**: 完成
**耗时**: 16.5 min
**判决**: **negative_but_informative** — L0-only LoRA 无法复制 all-layer LoRA 的成功，证实行为控制需要多层协同

---

## 1. 实验目的

**前提链回顾:**

| 实验 | 发现 | 含义 |
|------|------|------|
| P30-P35 | L0 是 causal hallucination entry point | 假说：干预入口点可控制行为 |
| P15 | all-layer LoRA CE 训练 → H=0.0 (log-prob) | 权重级干预**可**改变 log-prob 偏好 |
| P36 | L0 ablation 改变 log-prob 但不改变 generate() | log-prob → behavior gap |
| P36b | L0 ablation **恶化** generate 行为 | 入口点消融是破坏性的 |
| P37 | L0 counter-vector injection → generate() **零效应** | 激活级入口干预完全无效 |

**P38 核心问题:**

P15 证明 all-layer weight-level LoRA 有效，P37 证明 activation-level L0 干预无效。那么 **weight-level L0-only LoRA** 能否成功？

即：将 LoRA 仅施加于 L0 的 q_proj/k_proj/v_proj/o_proj，能否改变 `model.generate()` 行为？

如果 P38 失败而 P15 成功 → 行为控制需要多层协同，入口点假说不成立。

如果 P38 成功 → 入口点假说仅限于 weight-level，activation-level 过于表层。

---

## 2. 实验设计

### 2.1 LoRA 配置

| 参数 | 值 |
|------|-----|
| Target modules | q_proj, k_proj, v_proj, o_proj (所有层) |
| Active layers | **仅 L0**（其余 23 层 LoRA 权重冻结） |
| Rank (r) | 8 |
| Alpha | 16 |
| Dropout | 0.05 |
| Trainable params | 45,056 / 495,114,112 = **0.01%** |
| Optimizer | AdamW, lr=2e-4 |
| Epochs | 3 |
| Batch size | 2 |

**冻结策略**: 先对全部 24 层施加 LoRA，然后遍历 `named_parameters()`，对非 L0 参数的 `requires_grad=False`。最终的 LoRA 参数分布：184 frozen（L1-L23），8 active（L0 only）。

### 2.2 训练数据

- Train: 30 samples (15 answerable + 15 unanswerable)，来自 `train_all_s0.jsonl`
- Target construction: answerable → positive_response, unanswerable → negative_response
- Loss: Standard CE，prompt tokens masked to -100

### 2.3 评估

| 指标 | 方法 | 说明 |
|------|------|------|
| **Log-prob H** | log-prob(prefer hall) / unanswerable_total | 与 P15 可比 |
| **Log-prob C** | log-prob(prefer correct) / answerable_total | 正确率是否保持 |
| **Generate Hall** | model.generate() → 行为分类器 | 主指标 |
| **Generate Abst** | model.generate() → 行为分类器 | 拒绝率 |
| **Paired flip** | 逐样本 pre vs post 行为变化 | 方向性分析 |
| **P15 compare** | 加载 P15 all-layer LoRA checkpoint 同条件对比 | 跨实验对照 |

---

## 3. 结果

### 3.1 训练曲线

| Epoch | Loss | δ |
|-------|------|-----|
| 1 | 14.67 | — |
| 2 | 10.34 | −4.33 |
| 3 | 4.99 | −5.34 |

Loss 持续收敛，训练时间 173s（仅 0.01% 参数可训练）。

### 3.2 Log-Prob 对比

| | Pre (Base) | Post (P38 L0-LoRA) | P15 (All-Layer LoRA) |
|---|---|---|---|
| **H** | 0.4667 | **0.8333** (+0.3666) | **0.1333** |
| **C** | 1.0000 | 1.0000 | 1.0000 |

**反直觉发现**: L0-only LoRA 训练后 log-prob H 从 0.47 大幅恶化到 0.83（+78%）。模型经过 CE 训练后，对幻觉响应的概率偏好**不降反升**。而 P15 all-layer LoRA 将 H 压到 0.13。

### 3.3 Generate() 行为对比 (n=60)

| | Pre (Base) | Post (P38 L0-LoRA) | P15 (All-Layer) |
|---|---|---|---|
| **Hallucination** | 46/60 (76.7%) | 43/60 (71.7%) | **34/60 (56.7%)** |
| **Abstention** | 1/60 (1.7%) | 2/60 (3.3%) | **7/60 (11.7%)** |
| **Mixed** | 4/60 | 5/60 | 10/60 |
| **Other** | 9/60 | 10/60 | 9/60 |

- P38 L0-LoRA: Δhall = −3 (n=60)，不显著
- P15 all-layer: Δhall = −12 (n=60)，显著改善

### 3.4 Paired Flip 分析

| | Better | Worse | Same |
|---|---|---|---|
| P38 L0-LoRA | 11 (18%) | 9 (15%) | 40 (67%) |

Better vs Worse ≈ 1:1，没有方向性改善。

**有价值的个别案例**:
- Sample #2: hallucination → abstention（成功学会了拒绝）
  - Pre: "The answer to this question is in the section titled SilverArc. According to the information provided, Yuki Tanaka has been with SilverArc since 2015..."
  - Post: "We don't have any information about Yuki Tanaka's role at SilverArc..."
- Sample #13: hallucination → other（切换到类拒绝模式）
- Sample #22: hallucination → other

### 3.5 三向对比总览

```
                  Pre-Base    P38 L0-LoRA    P15 All-Layer
Log-prob H:       0.47    →   0.83 (✗✗)   →   0.13 (✓✓)
Generate Hall:    46/60   →   43/60 (✗)   →   34/60 (✓)
Generate Abst:    1/60    →   2/60  (✗)   →   7/60  (✓)
Trainable:        —           0.01%            ~0.5%
Training time:    —           2.9 min         80 min
```

---

## 4. 判决与分析

### 判决: **negative_but_informative**

L0-only weight-level LoRA **失败**。它不但无法复制 P15 all-layer LoRA 的成功，还使 log-prob 偏好恶化。

### 核心证据

1. **Log-prob 恶化**: H 0.47 → 0.83，意味着模型经过训练后**更偏好幻觉响应**。这与 "入口点干预" 的预期完全相反。

2. **Generate 无实质改善**: Δhall = −3 (n=60)，better=11, worse=9，净改善几乎为零。

3. **P15 对照有效**: 同数据集同评估方法，P15 all-layer LoRA 的 generate Hall=34/60 vs P38 的 43/60 — 多层 LoRA 明显更有效。

### 为什么 L0-only LoRA 失败？

三条件对比揭示了关键模式：

| 干预 | Level | Scope | Log-Pro H | Generate Hall |
|------|-------|-------|-----------|---------------|
| Activation ablation (P36b) | Activation | L0 only | changed | **worsened** |
| Counter-vector (P37) | Activation | L0 only | N/A | **no effect** |
| LoRA L0-only (P38) | **Weight** | L0 only | **worsened** | **no effect** |
| LoRA all-layer (P15) | **Weight** | **All** | **improved** | **improved** |

规律很清晰：**Scope 决定成败，Level (weight vs activation) 不决定**。

L0 的 causal entry 角色（P30-P35 证实）意味着它承载因果信息，但这不等于它可以作为**唯一的**控制杠杆。幻觉行为是 24 层自注意力迭代的涌现属性。只在 L0 修改信号，信号在后续 23 层的非线性变换中可能被重建/抵消，甚至产生反向效果（如 log-prob H 恶化）。

类比：L0 是河源。在河源改变水流方向（LoRA 训练 L0 权重）→ 下游河道（L1-L23 已训练的 parameters）可能把水流拉回原方向，甚至造成漩涡（H 恶化）。

### 修正假说

原假说 "L0 是 causal entry point → 干预 L0 可控制行为" 需要修正为：

> **L0 承载因果信息，但行为控制需要多层协同。单点干预（无论权重级还是激活级）不足。入口点假说的 "入口" ≠ 唯一控制点。**

---

## 5. 方法论反思

### 实验设计优点
- 直接对比 P15 all-layer LoRA（最强基线）
- 双指标（log-prob + generate）避免 P15 only-log-prob 的陷阱
- Per-sample audit 可看到个别成功案例
- 训练极快（2.9 min），可快速迭代

### 边界条件
- 训练量极小（30 samples, 3 epochs），可能欠拟合
- 仅测试一个 LoRA rank (r=8)，未做 rank sweep
- 仅在 L0 施加 LoRA，未测试 L0+L1、L0+L16+L20 等组合
- 测试集 n=60，小效应可能被噪声掩盖
- CE loss 训练目标与行为控制目标不完全一致

---

## 6. 下一步建议

### 立即可做

1. **增量层训练**: 从 L0 起步，逐步增加训练层数 L0→L0+L1→L0+L1+L2→...→全层，定位"最少需要几层才能改变 generate() 行为"。这是最有信息量的 follow-up。

2. **P15 的 generate() 对比深化**: P15 报告仅报告了 log-prob H=0.0，未测 generate()。P38 首次测了 P15 的 generate 行为（hall=34/60），需要为 P15 补一份 generate 行为分析。

3. **训练规模放大**: 将训练数据从 30 扩大到 90+，epochs 从 3 到 5-10，测试 L0-only 在大数据下是否仍恶化。

### 如果 L0+X 层训练也失败

则确认结论：**inference-time 入口点干预路径到此为止**。行为控制需要类似 P15 的全层权重训练，或转向不同的控制范式（如 DPO 在抽象偏好层面而非 token-preferences 层面）。

### 理论层面积累

经过 P36→P36b→P37→P38 的完整证据链：

1. **Causal sufficiency** (L0 有因果效应) — P30-P35 证实 ✅
2. **Causal control via L0** (可通过 L0 干预控制行为) — P36-P38 否定 ❌
3. **Causal necessity** (必须通过 L0 才能控制) — P15 间接否定（all-layer LoRA 也有效，说明 L0 并非必须）

三层区分完成了对 L0 entry point 假说的全面检验。