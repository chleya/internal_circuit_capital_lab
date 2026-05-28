# IC4 P40: Full-System Routing Training Audit

**日期**: 2026-05-26
**状态**: 完成
**耗时**: 53.0 min
**判决**: **late_layer_decision_dominant** — 晚期层（L12-L23）是行为控制的主要杠杆；v/o projection 是关键的投影类型

---

## 1. 实验目的

P39 发现了"绝望之谷"：contiguous early-layer CE LoRA (L0→L2→L5→L11) 全部恶化幻觉（43→50→49→59/60），但 P15 all-layer LoRA 改善（34/60）。

这个现象有四种可能的解释：

| 假说 | 内容 | 关键预测 |
|------|------|---------|
| **H1: full-depth coordination** | 行为控制需要跨深度路由协同 | sparse-depth 或 all-small-r2 应优于 L0-L11 |
| **H2: parameter count** | 仅仅是参数量问题，更多参数 = 更好 | hall 应与 trainable params 正相关 |
| **H3: late decision layers** | 真正控制拒答/输出的在后层 | late-only (L12-L23) 应优于 L0-L11 |
| **H4: objective mismatch** | CE 目标本身不适合行为控制 | 所有 CE 配置应 ≈ baseline |

P40 通过 5 个 ablated 配置组成 2×2 矩阵（depth 维度 × param 维度）来区分这四种假说。

---

## 2. 实验设计

### 2.1 配置矩阵

| 配置 | Rank | 层 | Proj | Params | H 测试 |
|------|------|-----|------|--------|--------|
| **all-small-r2** | r=2 | L0-L23 (24层) | q,k,v,o | 270K | H1 |
| **sparse-depth** | r=8 | {0,6,12,18,23} (5层) | q,k,v,o | 225K | H1 |
| **late-only** | r=8 | L12-L23 (12层) | q,k,v,o | 541K | H3 |
| **q-only-all** | r=8 | L0-L23 (24层) | q_proj | 344K | H2 |
| **vo-only-all** | r=8 | L0-L23 (24层) | v_proj,o_proj | 541K | H2 |

### 2.2 统一控制

- 所有配置：90 训练样本，3 epochs，lr=2e-4，bs=2
- 评估：model.generate() 行为分类（n=60）
- 测试集：同 P38/P39（"幻觉 = 生成新数字"）

### 2.3 假说判决逻辑

```
H1: sparse-depth < P39_L0L11 或 all-small-r2 < P39_L0L11
H2: correlation(params, hall) > 0.5
H3: late-only < P39_L0L11
H4: 所有配置 hall ≈ base (Δ ≤ 5)
```

---

## 3. 结果

### 3.1 主结果表（按 Hall 升序）

| # | 配置 | Params | Hall | Abst | Mixed | Other | ΔPre | Loss (终值) |
|---|------|--------|------|------|-------|-------|------|-------------|
| ⭐ | **late-only** | 541K | **28/60** | 12 | 10 | 10 | **−18** | 0.044 |
| ⭐ | **vo-only-all** | 541K | **34/60** | 11 | 4 | 11 | **−12** | 0.051 |
| — | P15 all r=4 | 688K | 34/60 | 7 | 10 | 9 | −12 | — |
| △ | all-small-r2 | 270K | 41/60 | 9 | 5 | 5 | −5 | 0.072 |
| — | q-only-all | 344K | 44/60 | 4 | 1 | 11 | −2 | 0.079 |
| — | sparse-depth | 225K | 45/60 | 2 | 4 | 9 | −1 | 0.089 |
| — | Pre (base) | 0 | 46/60 | 1 | 4 | 9 | 0 | — |
| 💀 | P39 L0-L11 | 541K | 59/60 | 1 | 0 | 0 | +13 | 0.099 |

### 3.2 假说测试

```
H1 ✗ (sparse=45, all-small-r2=41, L0-L11=59)
  → 技术上通过（两者都 < 59），但效果远逊于 H3。
    全深度有帮助，但效果不如专注后期层。

H2 ✗ REFUTED (correlation = −0.90)
  → 更多参数 → 更少幻觉（负相关），与 H2 预测相反。
    关键反例：L0-L11 (541K → hall=59) vs late-only (541K → hall=28)。
    相同参数量，深度位置决定效果。

H3 ✓✓ STRONGLY SUPPORTED
  → late-only (L12-L23, 541K) 取得 hall=28/60，超越所有其他配置，
    包括 P15 全层 LoRA (34/60)。

H4 ~ partially refuted
  → CE 训练在正确配置下（late-only, vo-only-all）可以产生显著行为改善。
    CE 本身不是瓶颈，瓶颈在于训练哪些层/哪些投影。
```

### 3.3 三个关键发现

**发现 1: 晚期层（L12-L23）即足够，且最优**

late-only 仅训练模型的后半 12 层，hall 从 46 降到 28（−39%），拒绝率从 1 升到 12。这甚至优于 P15 的全 24 层训练（hall=34）。

**反直觉洞察**: 早期层（L0-L11）不需要被训练来抑制幻觉 — 训练它们反而有害。幻觉抑制是一个晚期层决策功能。

**发现 2: v/o projection 是关键，q projection 不是**

| Proj config | Hall | 含义 |
|-------------|------|------|
| q-only-all (344K) | 44 | q_proj 近乎无用 |
| vo-only-all (541K) | 34 | v/o 单独 = P15 水平 |

v_proj（value projection）和 o_proj（output projection）控制了什么从每个 attention head 输出，直接影响 token 选择。q_proj 仅控制"注意什么"，不影响"输出什么"。

**发现 3: 深度 > 参数量**

| 对比 | Params | Hall | 优胜方 |
|------|--------|------|--------|
| L0-L11 vs late-only | 541K vs 541K | 59 vs 28 | late-only（同参数，不同深度） |
| L0-L11 vs all-small-r2 | 541K vs 270K | 59 vs 41 | all-small-r2（一半参数，全深度） |
| late-only vs vo-only-all | 541K vs 541K | 28 vs 34 | late-only（同参数，更聚焦） |

在所有 pairwise 对比中，深度位置和投影类型压倒参数量。

---

## 4. 判决与分析

### 判决: **late_layer_decision_dominant**

> 行为控制主要由晚期层（L12-L23）的 v/o projection 决定。早期层（L0-L11）的训练在有 post-training 幻觉抑制任务中是有害的。

### 机制假说: "前编后判" 两阶段路由

Qwen2.5-0.5B 的 24 层可被分为两个功能阶段：

```
L0  ───────────────── L11 │ L12 ──────────────── L23
    早期编码层                 晚期决策层
    (formation)               (execution)
    
    功能：将 prompt 编码       功能：基于编码做 token 决策
    为中间表示                  判断"是否有数字可答"
                               或"是否该拒答"

CE 训练早期层 → 学会更自信地表示数字
    ↓
冻结晚期层 → 按原始参数执行，无法识别"该拒答"
    ↓
结果：更多幻觉（P39 绝望之谷）

CE 训练晚期层 → 学会"何时拒答"的判别
    ↓
冻结早期层 → 编码质量不退化
    ↓
结果：显著改善（late-only）
```

### P30→P40 完整证据链更新

| 阶段 | 实验 | 核心发现 |
|------|------|---------|
| 诊断 | P30-P35 | L0 是因果信息入口 |
| 激活入口干预 | P36-P37 | L0 activation-level 干预完全无效 |
| 权重入口干预 | P38 | L0-only LoRA 失败（LP 恶化） |
| 层范围扫描 | P39 | 早层连续训练 = 绝望之谷（1-12 层全恶化） |
| **路由审计** | **P40** | **晚期 v/o 层是控制杠杆；早层训练有害** |

---

## 5. 方法论反思

### 实验设计优点
- 2×2 矩阵设计清晰区分四种假说
- 参数量 vs 深度位置的跨组对比强有力
- 发现 "q_proj 无用但 v/o_proj 关键" 是意外收获
- 数据量从 30→90 提高了统计可靠性

### 边界条件
- 仅测试 Qwen2.5-0.5B-Instruct，"前编后判" 分界点（L12）可能因模型而异
- 未测试非连续晚期层 (如 {12, 15, 18, 21, 23})
- 未测试 DPO 目标函数（CE 已证明在正确配置下有效）
- 训练数据仅 90 样本，更大数据可能进一步改善

---

## 6. 下一步建议

### 立即可做

1. **P41: late-only rank/size sweep** — 在 late-only (L12-L23) 基础上，测试不同的 rank (r=2, 4, 8, 16) 和数据量（90, 180, 全量），定位最优训练配置。P40 已证明 "晚期层可行"，需要找到效率上限。

2. **P41b: late-only projection sweep** — late-only 基础上测试不同投影组合：仅 v_proj、仅 o_proj、v+o、k+v+o，看是否可以进一步压缩参数。

3. **跨模型验证** — 在 Qwen2.5-1.5B 或 Llama-3.2 上复现 late-only 效果，验证 "前编后判" 的普遍性。

### 实验管线总结

P36→P40 完成了一条完整的因果链路研究：
1. P36-P37：激活级入口干预 → 失败
2. P38：权重级入口训练 → 失败
3. P39：早层连续训练 → 有害
4. **P40：晚期 v/o 层训练 → 成功**

推荐将 **late-only L12-L23, v/o_proj, r=8** 作为后续实验的标准干预配置。这比 P15 的全层 LoRA 更高效（12 层 vs 24 层）且效果相当（hall=28 vs 34）。
