# IC4 P37: Generative-Trajectory Intervention Report

**日期**: 2026-05-26
**状态**: 完成
**耗时**: 21.3 min
**判决**: **negative_but_informative** — counter-vector injection 无法改变生成行为

---

## 1. 实验目的

P36 发现 L0 token-entry 的 log-prob 可以被消融显著改变（4-17×），但 `model.generate()` 行为不受影响。P36b 确认这并非 small-n artifact：消融反而使幻觉恶化。

**P37 核心问题**: 将消融（ablation）替换为 counter-vector direction injection，能否在生成轨迹层面降低 hallucination rate？

**关键改进 vs P36/P36b**:
- 使用梯度方向 counter-vector 注入替代消融
- 比较 5 种 timing mode: T0_ablation, T0_counter, T1_continuous, T2_late, T1_random
- 每步连续注入（T1_continuous），而非仅 prefill 一次性干预
- Fixed seed (42)，matched controls
- 多 token family (funding n=8, patents n=5)

---

## 2. 技术方案

### 2.1 Counter-Vector 计算（梯度法）

原始方案（hidden-state difference `h_abst - h_hall`）失败：L0 是首层，因果注意力导致同一 causal position 的 L0 输出在 hallucination/abstention 两种条件下完全相同（norm=7.1442）。

改用**梯度法**：

```
v_anti = mean( -∂L_hall/∂h_L0 + ∂L_abst/∂h_L0 )
```

即：对 hallucination 响应的 loss 求梯度取反方向，对 abstention 响应的 loss 求梯度取正方向，平均后 L2 归一化。

具体实现：
1. 对每个 paired sample (hall_resp, abst_resp)，用 hook 捕获 L0 的 hidden state `h`
2. `h.retain_grad()` 后执行 `loss.backward()`
3. 在 causal token positions 取 `h.grad` 的平均值
4. hall: 取 `-grad`（远离幻觉方向），abst: 取 `+grad`（靠近拒绝方向）
5. 所有梯度平均后 L2 归一化 → 单位向量

关键修复点：
- `h.retain_grad()`：`h` 是非叶子中间张量，PyTorch 默认不保留其 `.grad`，必须显式调用
- `model.zero_grad()`：每次 backward 前清零，防止梯度累积
- 移除重复的 `for resp_label` 循环

### 2.2 Timing Modes 设计

| Mode | 描述 | 注入时机 |
|------|------|---------|
| baseline | 无干预 | — |
| T0_ablation | L0 causal pos 归零 | prefill only (step==0) |
| T0_counter | counter-vector 注入 | prefill only (step==0) |
| T1_continuous | counter-vector 注入 | 每一步生成 |
| T2_late | counter-vector 注入 | step >= 4 |
| T1_random | 同 norm 随机方向注入 | 每一步（对照）|

注入强度：`h[0, p, :] += scale × 0.1 × cv`，scale=1.0，实际有效注入量 = 0.1（约为 hidden state norm 7.14 的 1.4%）。

### 2.3 行为分类器（与 P36b 相同）

- **hallucination**: 包含未见过的数字
- **abstention**: 包含拒绝标记（"not provided", "cannot be determined"等）
- **mixed**: 拒绝形式 + 新数字
- **other**: 过短、无模式

---

## 3. 结果

### 3.1 Counter-Vector 计算

所有 3 个 token family 都成功计算：
- funding: |cv|=1.0000 (8 samples)
- patents: |cv|=1.0000 (5 samples)
- r_and_d_spend: |cv|=1.0000 (4 samples)

注意：|cv|=1.0000 是因为 L2 归一化，无法区分真正的信号强度和噪声基底。

### 3.2 主实验

#### Token: funding (n=8)

| Mode | Hall | Abst | Mixed | Other | Mean Len | Rep |
|------|------|------|-------|-------|----------|-----|
| baseline | 5/8 | 0/8 | 2/8 | 1/8 | 64.0 | 0.027 |
| T0_ablation | **7/8** | 0/8 | 1/8 | 0/8 | 64.0 | 0.008 |
| T0_counter | 5/8 | 0/8 | 2/8 | 1/8 | 64.0 | 0.027 |
| T1_continuous | 5/8 | 0/8 | 2/8 | 1/8 | 64.0 | 0.027 |
| T2_late | 5/8 | 0/8 | 2/8 | 1/8 | 64.0 | 0.027 |
| T1_random | 5/8 | 0/8 | 2/8 | 1/8 | 64.0 | 0.027 |

**关键发现**: T0_counter, T1_continuous, T2_late, T1_random 四种 counter-vector/random 模式产生与 baseline **完全相同的 8 条文本**（逐字符 bit-identical）。Counter-vector 注入对所有 8 个样本的影响为 **零**。

T0_ablation 使幻觉从 5/8 恶化到 7/8，确认 P36b 结论。

#### Token: patents (n=5)

| Mode | Hall | Abst | Mixed | Other | Mean Len | Rep |
|------|------|------|-------|-------|----------|-----|
| baseline | 3/5 | 0/5 | 1/5 | 1/5 | 64.0 | 0.000 |
| T0_ablation | **1/5** | 0/5 | 0/5 | **4/5** | 56.2 | 0.000 |
| T0_counter | 2/5 | 0/5 | 2/5 | 1/5 | 64.0 | 0.000 |
| T1_continuous | 2/5 | 0/5 | 2/5 | 1/5 | 64.0 | 0.000 |
| T2_late | 3/5 | 0/5 | 1/5 | 1/5 | 64.0 | 0.000 |
| T1_random | 4/5 | 0/5 | 1/5 | 0/5 | 64.0 | 0.000 |

**注意**: T0_ablation 的 hall=1/5 看似改善，但 other=4/5 揭示真相——模型不再回答专利问题，转而回答职位问题（"Nathan Rhodes has held the following positions..."、"Daniel Park holds the position of Senior Engineer..."）。这不是成功的拒绝，而是**行为崩溃**：消融摧毁了 L0 的信号，模型丢失了问题意图。

T0_counter 和 T1_continuous 均为 2/5 vs baseline 3/5，仅差 1 个样本（n=5，统计不显著）。

### 3.3 Paired Flip 分析

| Token | Mode | Better | Worse | Same | Δ |
|-------|------|--------|-------|------|---|
| funding | T0_ablation | 0 | 2 | 6 | degradation |
| funding | T0_counter | 0 | 0 | 8 | **no effect** |
| funding | T1_continuous | 0 | 0 | 8 | **no effect** |
| funding | T1_random | 0 | 0 | 8 | **no effect** |
| patents | T0_ablation | 3 | 0 | 2 | → "other" (wrong Q) |
| patents | T0_counter | 1 | 0 | 4 | weak |
| patents | T1_continuous | 1 | 0 | 4 | weak |
| patents | T1_random | 0 | 0 | 5 | **no effect** |

---

## 4. 判决与分析

### 判决: **negative_but_informative**

Counter-vector injection 在 L0 causal position **不能**改变 `model.generate()` 的生成行为。

### 核心证据

1. **funding 完全 null**（最强证据）：baseline 与 T0_counter/T1_continuous/T2_late/T1_random 产生 bit-identical 的 8 条文本。即使每步连续注入（T1_continuous），模型也循完全相同的 token 序列。这不是"效果太小看不到"，而是注入被完全忽略。

2. **patents 微弱信号不可靠**：T0_counter/T1_continuous 的 3→2 差异仅为 n=5 中的 1 样本变化，不具有统计意义。且 T1_random (4/5) 比 baseline (3/5) 更差，表明专利族对随机扰动敏感但方向随机。

3. **T0_ablation 持续恶化**：确认 P36b 结论，L0 消融使幻觉更严重（funding 5→7）或导致行为崩溃（patents → 回答错误问题）。

4. **scale 不足并非根因**：即使注入强度太小（0.1 vs hidden norm 7.14 ≈ 1.4%），T0_ablation（完全归零，= 100% 干预）也是恶化而非改善，说明问题不在于"不够强"，而在于 L0 信号不是幻觉控制的正确杠杆。

### 为什么 counter-vector 无效？

综合 P36→P36b→P37 的证据链：

- **P30-P35** 建立了 L0 作为 causal hallucination information entry point（log-prob 层面）
- **P36** 发现 L0 消融能剧烈改变 log-prob 但不改变行为
- **P36b** 发现消融反而恶化行为
- **P37** 发现 counter-vector 注入了也被完全忽略

解释：L0 的 hidden state 是整个 prompt 的编码结果。幻觉生成是一个**系统级决策**，涉及所有层的交互。在入口点（L0）注入一个固定方向无法"重定向"整个生成轨迹，因为在深层 attention 的迭代中，后续层会逐步重建原始方向。

类比：在河流源头扔一块石头，几百米后水流方向不变。生成式 transformer 的轨迹由整个网络的自注意力动力学决定，而非单一位置的单一向量。

---

## 5. 方法论反思

### 实验设计优点
- Paired flip 分析提供逐样本对照
- 多 timing mode 排除"时机不对"的解释
- Random direction control (T1_random) 区分方向特异性
- T0_ablation 作为桥梁连接 P36/P36b 的消融结果

### 边界条件
- 注入强度单一（scale=0.1），未做 scale sweep
- Counter-vector 经 L2 归一化后 |cv|=1.0，无法知道原始梯度信噪比
- patents n=5 不足以检测小效应
- 仅在 L0 注入，未测试更高层（L16, L20）
- 仅测试 Qwen2.5-0.5B-Instruct，未跨模型验证

---

## 6. 下一步建议

### 短期（剩余 IC-4 实验队列）

1. **Scale sweep**: 在更大注入强度（0.5, 1.0, 2.0）下重测 T1_continuous，排除"强度不够"的可能
2. **多层注入**: 同时在 L0 + L16 + L20 注入 counter-vector，测试层间协同效应
3. **DAS (Distributed Alignment Search) 风格**: 不固定注入方向，而是搜索子空间中能改变行为的旋转矩阵

### 中期（如果 inference-time 路径继续失败）

P37 的设计 spec 中明确了分支条件：**若 T1/T3 也失败，应转向 training-time 或 theta-level intervention。**

建议方案：
- **IC-5: DPO / RLHF 微调**：在 hallucination vs abstention paired data 上做 preference optimization
- **LoRA 注入**：在 L0 token-entry 的 key/value projection 上挂 LoRA adapter，用 refusal data 训练
- **激活重新路由**：参考 Representation Engineering (RepE) 的方法，识别 refusal direction 并在推理时重路由

### 理论方向

L0 入口点控制假说的修正：L0 承载因果信息，但因果信息 ≠ 可控。需要区分：
- **causal sufficiency**（L0 有因果效应 — P30-P35 证实）
- **causal control**（可以通过 L0 干预控制行为 — P36-P37 否定）
- **causal necessity**（必须通过 L0 干预 — 未知）

这三者的区分比简单的"L0 是入口点"更精确地刻画了内部机制。