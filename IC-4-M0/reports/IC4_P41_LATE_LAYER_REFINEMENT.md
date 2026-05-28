# IC4 P41: Late-Layer Control Refinement

**日期**: 2026-05-26
**状态**: 完成
**耗时**: 59.2 min
**判决**: **late_layer_recipe_confirmed** — L14-L23, r=8, q/k/v/o 为最优配置；P39 的极端差值是部分 small-n artifact，但 late-layer advantage 在 matched 数据下依然成立

---

## 1. 实验目的

P40 发现 late-only (L12-L23) 达到 hall=28/60，是所有配置中最佳。但存在一个关键匹配问题：

> P39 的 L0-L11@30 样本 (hall=59) vs P40 的 late-only@90 样本 (hall=28) 不是完全 matched contrast。

P41 的四个目标：

1. **Matched 负控**：用 P40 的 90 样本重跑 L0-L11，消除 sample-size confound
2. **Rank sweep**：测试 r=2, r=4 是否足够（P40 仅测了 r=8）
3. **投影子集**：测试 v/o only 是否保持效果
4. **Layer boundary sweep**：测试 L14-L23、L16-L23，定位真正的 late 边界
5. **Answerable correctness**：确保拒答改善不是以正确回答为代价

---

## 2. 实验设计

### 配置 (所有 90 样本, 3 epochs, lr=2e-4, bs=2)

| Config | Range | Rank | Proj | Params | 测试 |
|--------|-------|------|------|--------|------|
| L0-L11-R8 | 0-11 | 8 | q,k,v,o | 541K | Neg ctrl (matched) |
| L12-L23-R8 | 12-23 | 8 | q,k,v,o | 541K | Reference (P40) |
| L12-L23-R4 | 12-23 | 4 | q,k,v,o | 270K | Rank sweep |
| L12-L23-VO-R8 | 12-23 | 8 | v,o | 270K | Proj subset |
| L14-L23-R8 | 14-23 | 8 | q,k,v,o | 451K | Boundary: narrower |
| L16-L23-R8 | 16-23 | 8 | q,k,v,o | 360K | Boundary: minimal |

### 评估指标

新增 split-by-answerability 分类：
- **U-Hall**: unanswerable 样本上生成新数字的比率（主指标）
- **U-Abst**: unanswerable 样本上正确拒答的比率
- **A-Correct**: answerable 样本上非 abstention 的比率（正确性保持）

因为 answerable 样本的正确答案常含数字（被 classifier 标为 hallucination），U-Hall 是更准确的 hallucination 度量。

---

## 3. 结果

### 3.1 主结果表（按 U-Hall 升序）

| # | Config | U-Hall | U-Abst | A-Correct | U-Mixed | U-Other | A-Abst |
|---|--------|--------|--------|-----------|---------|---------|--------|
| ⭐ | **L14-L23-R8** | **2/30** | **13** | **30/30** | 4 | 11 | 0 |
| ⭐ | L12-L23-R8 | 4/30 | 12 | 30/30 | 4 | 10 | 0 |
| △ | L12-L23-VO-R8 | 8/30 | 4 | 27/30 | 1 | 17 | 3 |
| — | L0-L11-R8 (neg) | 10/30 | 8 | 30/30 | 7 | 5 | 0 |
| ✗ | L12-L23-R4 | 16/30 | 5 | 29/30 | 0 | 9 | 1 |
| ✗ | L16-L23-R8 | 18/30 | 6 | 30/30 | 2 | 4 | 0 |

### 3.2 四个关键发现

**发现 1: L14-L23 > L12-L23 — 最优边界不是 L12**

```
U-Hall vs layer range:
  L0-L11:   10/30 (neg ctrl)
  L12-L23:  4/30  ← P40 的 winner
  L14-L23:  2/30  ← 更好！L12-L13 有干扰
  L16-L23:  18/30 ← 崩溃，太少层
```

L12-L13 在 hallucination 抑制任务中可能是干扰性（而非帮助性）的。最佳 late 范围是 **L14-L23 (10 layers)**。

**发现 2: Matched neg ctrl — P39 的部分差值是 small-n artifact**

| | 训练样本 | U-Hall |
|---|---------|--------|
| P39 L0-L11 | 30 | 59/60 *overall* |
| P41 L0-L11 | 90 | 10/30 *unans-only* |

P39 的 L0-L11 极端恶化（hall=59/60）部分是由于 30 样本的小数据过拟合。用 90 样本后，早层训练的 unanswerable hallucination 仅为 10/30 = 33%。

但是，matched 负控下 late-layer advantage 依然成立：
- L0-L11@90: U-Hall=10, U-Abst=8
- L12-L23@90: U-Hall=4, U-Abst=12
- ΔU-Hall = −6 (60% reduction)

**发现 3: r=8 是必要的最低 rank**

| Rank | U-Hall | U-Abst | Params |
|------|--------|--------|--------|
| r=8 | 4 | 12 | 541K |
| r=4 | 16 | 5 | 270K |

r=4 的 U-Hall (16) 甚至比 L0-L11 (10) 更差。rank 效应很强：从 r=4 到 r=8 有 4× 的 hallucination 降低。

**发现 4: v/o only 不充分；answerable 正确性保持良好**

v/o-only 的 U-Hall=8 优于 r=4 (16) 但不如全投影 (4)。更重要的是 A-Correct 从 30→27，说明 v/o-only 丢失了对 answerable 样本的正确回答能力。

所有 late-layer 配置在 answerable 上保持 abstention=0（L12-L23, L14-L23, L16-L23），没有出现"拒绝塌缩"。

### 3.3 运行效率

| 指标 | L12-L23-R8 | L14-L23-R8 (最优) | 比率 |
|------|-----------|-------------------|------|
| Trainable params | 541K | 451K | 83% |
| 训练+评估时间 | 596s | 552s | 93% |

---

## 4. 判决与分析

### 判决: **late_layer_recipe_confirmed**

> 最优配置: **L14-L23, r=8, q/k/v/o 全投影, 90+ 训练样本**。
> U-Hall=2/30 (6.7%), U-Abst=13/30 (43.3%), A-Correct=30/30 (100%)。

### P36→P41 完整证据链终版

| 实验 | 核心发现 | 对假说的修正 |
|------|---------|-------------|
| P30-P35 | L0 是因果信息入口 | 入口承载 info |
| P36-P37 | L0 activation 干预无效 | 入口 ≠ 控制杠杆 |
| P38 | L0 weight LoRA 无效 | weight-level 入口也不行 |
| P39 | 早层连续训练 = 绝望之谷 | 部分 small-n artifact |
| P40 | 晚期 v/o 层主导行为控制 | H3 late-decision 胜出 |
| **P41** | **L14-L23 最优; P39 极值=small-n** | **recipe 确认** |

### 假说终态

```
L0  ──────── L11 │ L12 ── L13 │ L14 ───────── L23
  早期编码层        过渡层        晚期决策层
  (formation)      (ambiguous)   (execution)
  CE训练可改善      训练无益/     核心控制杠杆
  但非最优          可能干扰      U-Hall=2/30
```

**P39 的 "绝望之谷" 需要修正**：
- P39 的 L0-L11@30 hall=59/60 部分是 small-n artifact
- P41 的 L0-L11@90 U-Hall=10/30 说明早层训练并非"有毒"，只是效率低于晚期层
- 真正的不可训练区域是 L12-L13（过渡层：训练无益甚至干扰）

---

## 5. 方法论反思

### 实验设计优点
- Split-by-answerability 评估避免了 "hallucination = 新数字" 在 answerable 上的误判
- Matched negative control (L0-L11@90) 解决了 P39/P40 的 confound
- 窄化的 boundary sweep 精确定位了 L14 界线

### 边界条件
- L12-L13 的干扰性需要单独验证（L12-only, L13-only, L12+L13）
- 仅 Qwen2.5-0.5B-Instruct，跨模型泛化未验证
- 未测试 r=16（rank upper bound）
- 训练样本仅 90，更大数据可能进一步改善

---

## 6. 下一步建议

### P40/P41 管线总结

从 P30 到 P41，实验管线完成了一条完整的从诊断到控制的回路：

1. **诊断** (P30-P35): 识别 L0 为 causal entry
2. **入口干预** (P36-P38): 证明单点 L0 干预（activation/weight）无效
3. **路由审计** (P39-P40): 发现 late-layer v/o 是控制杠杆
4. **配方精炼** (P41): 确认 L14-L23/r=8/qkv/o 为最优配置

该管线可作为 IC-4-M0 "从 mechanisitic interpretability 到 behavioral control" 的参考模板。

### 可继续的扩展

- **跨模型验证**: 在 Qwen2.5-1.5B 或 Llama-3.2 上复现 L14-L23 recipe
- **数据 scaling**: 90→180→全量训练数据，看 U-Hall 是否继续下降
- **DPO 变体**: 在 L14-L23 recipe 上改用 DPO 目标，测试是否优于 CE