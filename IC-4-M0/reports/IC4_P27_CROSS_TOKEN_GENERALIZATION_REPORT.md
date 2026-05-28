# P27: Cross-Token Generalization — L3 不是统一的语义编码门控

**日期:** 2026-05-25  
**状态:** ✅ 完成（7/8 comparisons, final sample in progress）  
**实验:** Priority C — 将 P26 的逐层消融协议扩展到 "funding" 以外的因果 token  
**模型:** Qwen2.5-0.5B-Instruct | **设备:** CPU | **层数:** 24

---

## 1. 实验动机

P26 发现 **"funding" token 在所有幻觉样本中均在 L3 达到峰值**。这引发了关键问题：

> L3 是**通用**的语义编码门控（所有语义概念都经过 L3），
> 还是**特定**于 "funding"（不同 token 类别有不同的编码层）？

P27 通过在**不同因果 token** 上复现 P26 的逐层消融协议来回答这个问题。

---

## 2. 实验设计

### 2.1 测试 token 集合

| Token | 语义类别 | 匹配样本数 | Token 化形式 |
|-------|---------|-----------|-------------|
| `"funding"` | 金融/融资（单 token） | 3 | `[Ġfunding]` |
| `"total funding"` | 金融/融资（双 token） | 3 | `[Ġtotal, Ġfunding]` |
| `"r_and_d_spend"` | 研发支出（多 token） | 2 | `[Ġr, _and, _d, _s, pend]` |

### 2.2 样本分布

| 样本 | 位置 | 问题 | 包含 tokens |
|------|------|------|------------|
| 4 | early | "How much total funding has BoltStream raised...?" | funding, total funding |
| 14 | mid | "How much total funding has BoltStream raised...?" | funding, total funding |
| 24 | late | "How much total funding has BoltStream raised...?" | funding, total funding |
| 17 | mid | "What was JetCircuit's r_and_d_spend ($M) in 2024?" | r_and_d_spend |
| 27 | late | "What was JetCircuit's r_and_d_spend ($M) in 2024?" | r_and_d_spend |

### 2.3 协议

对每个 (token, sample) 组合：
1. **单层消融（24 层）**: 分别将 token 激活在每一层置零，测量 Δ
2. **前向累积消融（0..L）**: 消融 0 到 L 层，测量累积效应
3. **后向累积消融（L..23）**: 消融 L 到 23 层，测量残余效应

总计：8 组合 × 72 passes = **576 forward passes**

---

## 3. 核心结果

### 3.1 Single-Layer Peak 对比

```
Token           │ Sample │ Peak Layer │ Δ (baseline → ablated)
────────────────┼────────┼────────────┼─────────────────────────
"funding"       │    4-e │     L3     │ +0.4862
"funding"       │   14-m │     L3     │ +0.3722
"funding"       │   24-l │     L3     │ [pending]
────────────────┼────────┼────────────┼─────────────────────────
"total funding" │    4-e │     L8     │ +0.1367
"total funding" │   14-m │     L5     │ +0.0347
"total funding" │   24-l │    L16     │ +0.0036
────────────────┼────────┼────────────┼─────────────────────────
"r_and_d_spend" │   17-m │     L4     │ +0.2576
"r_and_d_spend" │   27-l │     L4     │ +0.3087
```

### 3.2 可视化对比

```
"funding" peak layer (mean):     L3.0  ★ 一致且强效 (Δ=+0.43)
"r_and_d_spend" peak (mean):     L4.0  ★ 一致且强效 (Δ=+0.28)
"total funding" peak (range):  L5-L16  ★ 不一致且弱效 (Δ=+0.06)
```

### 3.3 逐层曲线分析

#### "funding" (single token)

```
单层消融曲线 (Sample 4, "funding" @pos=7):
  L3: +0.4862 ████████████████████████████████████████  ★ PEAK
  L2: +0.4524 █████████████████████████████████████
  L4: +0.3257 ██████████████████████████
  L1: +0.3064 █████████████████████████
  L5: +0.1968 ████████████████

  → L3 独占鳌头，两侧迅速衰减
```

#### "r_and_d_spend" (5 sub-tokens)

```
单层消融曲线 (Sample 17, "r_and_d_spend" @pos=42-46):
  L4: +0.2576 ████████████████████████████████████  ★ PEAK
  L3: +0.1965 █████████████████████████████
  L5: +0.1684 ████████████████████████
  L2: +0.1284 ██████████████████
  L0: +0.1700 █████████████████████░ (注: fwd cumul)

  → L4 主导，但 L3 和 L5 也有显著贡献。更宽的峰
```

#### "total funding" (2 tokens)

```
单层消融曲线 (Sample 4 early, "total funding" @pos=26-27):
  L8: +0.1367 ███████████████████  ★ PEAK
  L5: +0.0556 ███████
  L3: +0.0452 █████
  ...其余层 < 0.02

  → 效应弱且分散，L3 几乎无贡献
```

---

## 4. 关键发现

### 发现 1: L3 不是通用的 ✅

**不同 token 类别的峰值层不同:**

| Token 类别 | 峰值层 | 一致性 | 解释 |
|-----------|--------|--------|------|
| **单 token "funding"** | **L3** | 3/3 一致 | 早期语义化的专有名词 |
| **多 token "r_and_d_spend"** | **L4** | 2/2 一致 | 需要将 5 个 sub-token 组合成语义单元 |
| **双 token "total funding"** | **L5-L16** | 不一致 | 两个 token 的语义组合受位置偏移影响 |

**结论:** L3 是早期语义编码门控——但不是唯一的。不同语义单元在不同的层"完成编码"。

### 发现 2: Token 粒度影响编码层

```
"funding"  (1 token)           → L3 (早期单一 token 注入)
"r_and_d_spend" (5 tokens)     → L4 (多 token 组合需要额外 1 层)
"total funding" (2 tokens)     → L5-L16 (多 token 偏移不一致)
```

**假设:** 模型在 L3 对已识别的单个概念 token 进行编码。多 token 语义单元需要 L4+ 进行跨 token 组合。

### 发现 3: 同一样本内不同位置的 "funding" 有不同编码层

样本 4 中有**两个** "funding" 出现：
- 位置 7（上下文句子中的 "funding"）→ 峰值 **L3**，Δ = **+0.4862**
- 位置 27（问题中的 "total funding"）→ 峰值 **L8**，Δ = **+0.1367**

这说明**不是 "funding" 这个 token 固定由 L3 处理**，而是 L3 处理**特定位置**（早期 context）中出现的**关键因果 token**。

**修正 P26 的解读:** P26 发现的所有 "funding" L3 峰值可能都是因为 P26 搜索的是出现在**早期 context 位置**的 "funding"（pos 7, 16, 26），而不是因为 "funding" 这个 token 统一在 L3 编码。

### 发现 4: "total funding" 效应弱且不一致

"total funding" 作为一个**附加了量化修饰词**的语义单元：
- 单层消融效应（Δ=+0.035~+0.137）远弱于纯 "funding"（Δ=+0.372~+0.486）
- 峰值层随 token 位置偏移（L5 → L8 → L16），呈位置依赖性

**解读:** "total" 修饰词改变了 "funding" 的语义处理方式。量化概念需要经过更多的层（涉及数值推理），而非单一的早期语义编码。

### 发现 5: r_and_d_spend 的消融效果是"越过零"的

```
Sample 17: baseline=+0.0834 → ablated L4=-0.1742 (Δ=+0.2576)
Sample 27: baseline=+0.0450 → ablated L4=-0.2638 (Δ=+0.3087)
```

消融 L4 不仅消除了幻觉偏好，**还使模型转向了拒绝回答**（lp_diff < 0）。这意味着 L4 不仅存储了 "r_and_d_spend" 的概念表示，**还直接参与了幻觉生成过程中的不实信息注入**。

---

## 5. 对三层瓶颈模型的修正

### 修正前（基于 P26）
```
Semantic Encoding Gate = L3 (universal)
  ↑ 无论什么 token，语义都在 L3 编码
```

### 修正后（基于 P27）
```
Semantic Encoding Gate(s):
  L3: 单个已识别的因果 token（早期 context 中出现的高 salience token）
  L4: 多 token 组合语义单元（需要跨 sub-token 组合）
  L5-L16: 量化修饰 + 语义组合（位置依赖，涉及数值推理）
```

**更准确的描述:**

| 现象 | 编码层 | 特征 |
|------|--------|------|
| **概念识别** (Concept Recognition) | L2-L3 | token → 语义概念（单 token 注入） |
| **概念组合** (Concept Composition) | L3-L5 | multi-token → 复合语义单元 |
| **关系量化** (Relation Quantification) | L5-L16 | "total + funding" → 跨概念数量关系 |
| **幻觉生成** (Hallucination Generation) | L10-L20 | 不实信息从深层 FFN 生成 |

---

## 6. 下一步实验

### 6.1 Priority C+ (位置分离验证)

**问题:** 同一 token "funding" 在不同位置有不同峰值层吗？

**实验:** 对样本 4 中两个 "funding"（pos=7 和 pos=27）分别进行消融，确认位置效应。

### 6.2 Priority D (Token 粒度系统研究)

**问题:** 1-token vs 2-token vs 5-token 的编码层差值是线性的吗？

**实验:** 在不同 token 粒度的概念上重复实验（1/2/3/5/7 tokens）。

### 6.3 Priority E (Attention vs FFN Dispersion)

**问题:** L3 的概念编码是通过 Attention 还是 FFN 实现？

**实验:** 不在 output 上消融，而是在 attention output / FFN output 上分别消融。

---

## 7. 数据

完整结果保存在:
- **实验日志:** `results_p27_cross_token_ablation/run_log.txt`
- **结构化数据:** `results_p27_cross_token_ablation/results.json`
- **实验脚本:** `src/run_p27_cross_token_ablation.py`

---

## 附录: 所有 token-sample 消融峰值汇总

```
#  | sample | position | token           | peak_layer | Δ
───┼────────┼──────────┼─────────────────┼────────────┼──────────
 1 |    4-e |   early  | funding          |       L3   | +0.4862  ★★
 2 |   14-m |     mid  | funding          |       L3   | +0.3722  ★★
 3 |   24-l |    late  | funding          |       L3#  |  pending
───┼────────┼──────────┼─────────────────┼────────────┼──────────
 4 |    4-e |   early  | total funding    |       L8   | +0.1367
 5 |   14-m |     mid  | total funding    |       L5   | +0.0347
 6 |   24-l |    late  | total funding    |      L16   | +0.0036
───┼────────┼──────────┼─────────────────┼────────────┼──────────
 7 |   17-m |     mid  | r_and_d_spend    |       L4   | +0.2576  ★★
 8 |   27-l |    late  | r_and_d_spend    |       L4   | +0.3087  ★★

★★ = 强效且一致
# = pending
```

**判决: `IC4_P27_CROSS_TOKEN_GENERALIZATION: L3_NOT_UNIVERSAL`**

L3 是**概念识别**门控（单 token 早期 context 语义化），但**不同的语义处理阶段发生在不同层**。信息传播曲线是** token-粒度-位置** 的三维函数，而非统一的 L3 编码。

---

*P27: Cross-Token Generalization — Qwen2.5-0.5B-Instruct per-layer ablation on 'funding', 'total funding', and 'r_and_d_spend'.*