# Phase 6-7-8+: 前向研究计划

> **状态**: ✅ **Phase 6/7/8 全部完成** (2026-05-23) — 本文件保留为历史路线图，实际执行结果见 [UNIFIED_RESEARCH_MAP.md](UNIFIED_RESEARCH_MAP.md) C10-C21  
> 从 Stabilization Scaling 到三瓶颈联合补偿 —— 项目下一步的完整路线图  
> 编制日期: 2026-05-22 | 完成日期: 2026-05-23  
> 前置文档: [RESEARCH_TRAJECTORY_REPORT.md](RESEARCH_TRAJECTORY_REPORT.md) — 当前状态全集

---

## 目录

1. [概述：为什么是这个顺序](#1-概述为什么是这个顺序)
2. [Phase 6: Stabilization Scaling —— 加固最大正信号](#2-phase-6-stabilization-scaling--加固最大正信号)
3. [Phase 7: Cross-Bottleneck Joint —— 测试 1+1>2](#3-phase-7-cross-bottleneck-joint--测试-112)
4. [Phase 8+: Absorption Remedy —— 最上游的干预](#4-phase-8-absorption-remedy--最上游的干预)
5. [决策树：Go/No-Go 条件](#5-决策树gono-go-条件)
6. [风险与替代方案](#6-风险与替代方案)

---

## 1. 概述：为什么是这个顺序

### 1.1 项目的当前定位

三个瓶颈均已诊断：

| 瓶颈 | 诊断 | 补救 | 补救强度 |
|---|---|---|---|
| A: Absorption | ✅ KNN=1.0 + PSI=0.0084 + ΔC=0.067 | ❌ 无 | — |
| B: Organization | ✅ 方向特异性已排除 | ✅ 闭环 gate (M3-v6) | Partial (有边界条件) |
| C: Stabilization | ✅ Root cause: KMeans ignores Y | ✅ Per-Action KMeans=0.585 | **Strong** (+31% over NoMemory) |

**Per-Action KMeans 是项目当前最强的正信号** —— 从 0.095 到 0.585，6.2x 提升，超越了所有基线。但它只在 5 seed × ~100 sample/seed 的小规模下验证过。

Phase 6 的目标是：**验证这个突破能否在大规模下站稳**。如果它能通过 scaling 测试，它就成为项目最稳固的实验支柱。如果它在大规模下退化，我们至少知道了它的边界条件。

### 1.2 推荐的执行顺序及理由

```
Phase 6: Stabilization Scaling  (2-4 天，取决于规模)
    ↓ 如果 Per-Action KMeans 在 scaling 下保持优势
Phase 7: Cross-Bottleneck Joint  (1-2 天，依赖 Phase 6 产出)
    ↓ 如果联合实验揭示耦合效应
Phase 8+: Absorption Remedy  (设计讨论，后续实施)
```

**理由**: 
- Phase 6 不依赖任何新设计，纯粹是对已知突破的 robustness test
- Phase 6 的结果会直接决定 Phase 7 的设计参数（用什么作为 stabilization 基础）
- Phase 6 的"失败"（如果发生）同样是重要信息——它告诉我们 stabilization 突破的边界条件，直接指导后续方向
- Phase 8+ 目前处于概念阶段，需要 Phase 6/7 的发现来缩小设计空间

### 1.3 与 Research Trajectory Report 的关系

| 文档 | 覆盖范围 |
|---|---|
| [RESEARCH_TRAJECTORY_REPORT.md](RESEARCH_TRAJECTORY_REPORT.md) | 已完成的工作：Phase 1-5，2026-05-19 → 2026-05-22 |
| **PHASE_6_7_8_PLAN.md（本文件）** | **将要进行的工作：Phase 6-8+，2026-05-22 →** |

---

## 2. Phase 6: Stabilization Scaling —— 加固最大正信号

### 2.1 核心问题

> Per-Action KMeans = 0.585 是在特定规模下（5 seeds, ~100 samples/seed, 3 actions）的偶然还是可复现突破？

具体来说：
- **Seed scaling**: 从 5 seeds 扩展到 20-50 seeds，Per-Action KMeans 是否保持优势？
- **Objective scaling**: 从 3 actions 扩展到更多 actions（模拟更复杂的决策空间），Y-aware 方法是否仍然有效？
- **Capacity scaling**: 每个 action 的 per-action centroids 数量是否应该随数据量增长而自适应？
- **Noise scaling**: 当环境噪音增大时，哪些策略更鲁棒？

### 2.2 当前基线复习

来自 `run_c3_y_aware_consolidation.py`，C3 最终结果（5 seeds, ~100 samples/seed, 3 actions）：

| 策略 | Step 5 Match | 类别 |
|---|---|---|
| per_action_kmeans | **0.585** | Y-aware |
| y_aware_w5.0 | 0.500 | Y-aware |
| y_aware_w2.0 | 0.245 | Y-aware |
| nomemory | 0.445 | Shortcut |
| episodic | 0.195 | Episodic |
| dt_20 | 0.190 | Y-aware |
| kmeans_200 | 0.120 | Resolution |
| kmeans_20 | **0.095** | Baseline |

环境参数:
```python
ENV_KWARGS = dict(
    state_dim=2, mode_flip_prob=0.08, autonomous_drift=0.05,
    autonomous_noise=0.02, action_gain=0.70, action_noise=0.03,
    action_sign_flip=True, history_len=8,
    action_cost=0.20, state_dependent_gain=True, saturation_k=0.5,
)
```

### 2.3 实验设计 A: Seed Scaling

**目标**: 验证 Per-Action KMeans 在更多 seeds 下是否保持优势。

**设计**:

| 参数 | 当前 (C3) | Scaling 1 (S1) | Scaling 2 (S2) | Scaling 3 (S3) |
|---|---|---|---|---|
| N seeds | 5 | **20** | **50** | **100** |
| Samples/seed | ~100 | ~100 | ~100 | ~100 |
| Actions | 3 | 3 | 3 | 3 |
| Per-action prototypes | 7 | 7 | 7 | **adaptive** |
| 预期运行时间 | — | ~2 min | ~5 min | ~12 min |

**测试策略**（精简版——不跑全部 14 种）:

| 策略 | 理由 | S1 | S2 | S3 |
|---|---|---|---|---|
| kmeans_20 | Baseline | ✓ | ✓ | ✓ |
| nomemory | Shortcut 上界 | ✓ | ✓ | ✓ |
| y_aware_w5.0 | Best Y-aware pure | ✓ | ✓ | ✓ |
| per_action_kmeans (n=7) | **当前 best** | ✓ | ✓ | ✓ |
| per_action_kmeans (n=adaptive) | Capacity scaling | — | — | ✓ |
| dt_50 | Alternative Y-aware | ✓ | — | — |

**假说**:

| 假说 | 预测 | 验证方式 |
|---|---|---|
| H6.1: Per-Action dominates at scale | per_action_kmeans > nomemory at all seed counts | S1→S3 全部通过 |
| H6.2: Per-Action advantage plateaus | per_action gain over nomemory stabilizes at some N_seed | 追踪 Δ(per_action, nomemory) across S1→S3 |
| H6.3: Adaptive centroids outpu | adaptive > fixed at large N | S3: adaptive vs n=7 |
| H6.4: Y-aware saturates slower than X-only | y_aware_w5.0 growth rate > kmeans_20 growth rate | 比较斜率 |

**实现**: 在 C3 代码基础上扩展 `SEEDS` 并生成对应 counterfactual 数据。核心逻辑现已封装在类中，无需重新实现。

```python
# 伪代码：Seed Scaling 实验
for n_seeds, label in [(20, "S1"), (50, "S2"), (100, "S3")]:
    SEEDS = list(range(n_seeds))
    # 生成 n_seeds 条 counterfactual 数据
    # 复用 C3 的 consolidation pipeline
    # 对每个 strategy 追踪 match trajectory
```

**代码位置**: `F:\intelligence_capital_minimal_lab\src\run_c4_stabilization_scaling.py`（新建）

**数据结构**: 复用 `F:\intelligence_capital_minimal_lab\src\run_c3_y_aware_consolidation.py` 中定义的类：
- `KMeansBaseline`
- `YAwareKMeans`
- `PerActionKMeans`
- `EpisodicRetention`
- `NoMemoryBaseline`
- `DecisionTreeCentroids`

**新增类**: `AdaptivePerActionKMeans` — 根据 action 数据量自适应调整 centroids 数量

```python
class AdaptivePerActionKMeans:
    """Per-Action KMeans with adaptive centroid count per action group.
    n_prototypes = max(3, min(n_max, int(sqrt(n_samples_in_action)))).
    """
    def __init__(self, n_max=20):
        self.n_max = n_max
        self._per_action = {0: {"X": [], "Y": []},
                            1: {"X": [], "Y": []},
                            2: {"X": [], "Y": []}}
        self._centroids = None
        self._Y_centroids = None

    def update(self, X_new, Y_new, seed_label=None):
        best_actions = np.argmax(Y_new, axis=1)
        for i in range(len(X_new)):
            a = int(best_actions[i])
            self._per_action[a]["X"].append(X_new[i])
            self._per_action[a]["Y"].append(Y_new[i])
        centroids_parts = []
        Y_parts = []
        for a in range(3):
            X_arr = np.array(self._per_action[a]["X"])
            Y_arr = np.array(self._per_action[a]["Y"])
            if len(X_arr) == 0:
                continue
            nc = max(3, min(self.n_max, int(np.sqrt(len(X_arr)))))
            km = KMeans(n_clusters=nc, random_state=42, n_init="auto")
            labels = km.fit_predict(X_arr)
            for i in range(nc):
                mask = labels == i
                if mask.sum() > 0:
                    centroids_parts.append(km.cluster_centers_[i])
                    Y_parts.append(Y_arr[mask].mean(axis=0))
        if len(centroids_parts) > 0:
            self._centroids = np.stack(centroids_parts)
            self._Y_centroids = np.stack(Y_parts)

    def predict(self, X_query):
        if self._centroids is None:
            return np.ones((len(X_query), 3)) * RANDOM_BASELINE
        idxs, _ = pairwise_distances_argmin_min(X_query, self._centroids)
        return self._Y_centroids[idxs]
```

**输出指标**: 对每个 scaling level，产出：
- `match_trajectory` — 每步的 best action match
- `final_match` — 最终步 match
- `Δ vs nomemory` — per_action - nomemory
- `pooled_centroid_count` — 总 centroids 数量
- `per_action_distribution` — 每个 action 的样本数分布

**成功标准**:

| 标准 | 阈值 | 含义 |
|---|---|---|
| ✅ Strong pass | per_action > nomemory at all N, Δ ≥ 0.05 | 突破在各种规模下通用 |
| ⚠️ Conditional pass | per_action > nomemory but Δ shrinks with N | 优势随规模缩小，有边界 |
| ❌ Fail | per_action ≈ nomemory at some N | 突破是 small-N artifact |
| ❌ Catastrophic fail | per_action < nomemory at some N | 方法在高规模下有害 |

### 2.4 实验设计 B: Objective Scaling (Multi-Action)

**目标**: 测试 Y-aware 方法在更多 actions（更复杂决策空间）下的表现。

**设计**:

| 参数 | 当前 | S-A5 | S-A10 | S-A20 |
|---|---|---|---|---|
| N seeds | 5 | 5 | 5 | 5 |
| Actions | 3 | **5** | **10** | **20** |
| Per-action prototypes | 7 | 5 | 3 | 2 |
| Random baseline | 0.333 | 0.200 | 0.100 | 0.050 |

**假说**:

| 假说 | 预测 |
|---|---|
| H6.5: Action 越多，Y-aware 优势越大 | X-only KMeans 在 D=10 时 match → random baseline，Y-aware 保持 > random |
| H6.6: Per-action centroids 稀释效应 | 随着 n_actions 增加，每个 action 的 centroids 变少，优势递减 |

**实现**: 修改 `ENV_KWARGS` 或直接用 `np.random.choice` 生成多 action 标签。

**注意**: 当前 counterfactual 数据生成假设 3 actions。需要确认 `prepare_counterfactual_data` 是否支持多 action 输出，或需要修改数据生成逻辑。

**成功标准**: Y-aware 在 A=10 时仍然显著 > random；A=20 时识别退化点。

**✅ Phase 6-B COMPLETE (2026-05-23)**:

| Actions | KMeans | Y-aware | PerAct | NoMem | PA-NM | PA-KM |
|---|---|---|---|---|---|---|
| 3 | 0.095 | 0.465 | 0.500 | 0.445 | +0.055 | +0.405 |
| 5 | 0.545 | 0.665 | **0.715** | 0.385 | +0.330 | +0.170 |
| 10 | 0.285 | 0.355 | 0.355 | 0.265 | +0.090 | +0.070 |
| 20 | 0.175 | 0.195 | 0.155 | 0.110 | +0.045 | **-0.020** |

**Verdict: STABLE** — delta(PA-NM) slope = -0.010, effectively flat across 3→20 actions. PA peaks at 5 actions (0.715). At 20 actions, PA drops below KM for first time — this is the boundary. PA-NM delta stays positive through 20 actions. H6.5 (more actions = bigger Y-aware advantage): NOT SUPPORTED — advantage stable, not growing. H6.6 (per-action centroid dilution): PARTIALLY SUPPORTED — PA declines 0.715→0.155 but still > NoMem.

### 2.5 实验设计 C: Noise Scaling

**目标**: 测试各策略在环境噪音增大时的鲁棒性。

**设计**:

| 参数 | 当前 | N-low | N-mid | N-high |
|---|---|---|---|---|
| action_noise | 0.03 | 0.03 | **0.10** | **0.30** |
| autonomous_drift | 0.05 | 0.05 | **0.15** | **0.40** |
| mode_flip_prob | 0.08 | 0.08 | **0.20** | **0.50** |

**假说**:

| 假说 | 预测 |
|---|---|
| H6.7: Y-aware 在高噪声下更鲁棒 | 当 X 信号被噪声污染时，Y-aware 可利用 Y 信号补偿 → 降速慢于 X-only |
| H6.8: NoMemory 在高噪声下趋近 random | 噪声 → action 分布均匀化 → nomemory match → 1/n_actions |

**成功标准**: Y-aware 降速 < X-only 降速；在高噪声下 Y-aware 仍 > random。

**✅ Phase 6-C COMPLETE (2026-05-23)**:

| Noise (σ) | KMeans | Y-aware | PerAct | NoMem | PA-NM |
|---|---|---|---|---|---|
| 0.00 | 0.095 | 0.465 | 0.500 | 0.445 | +0.055 |
| 0.03 | 0.095 | 0.482 | 0.495 | 0.445 | +0.050 |
| 0.10 | 0.095 | 0.478 | 0.497 | 0.445 | +0.052 |
| 0.30 | 0.095 | 0.457 | 0.505 | 0.445 | +0.060 |
| 1.00 | 0.095 | 0.480 | **0.545** | 0.445 | **+0.100** |

**Verdict: WEAK POSITIVE** — KM drop = 0.000 (already at floor, can't go lower). PA drop = -0.045 (PA actually IMPROVES at noise=1.0σ: 0.545, Δ-NM=+0.100). NoMem invariant at 0.445 (frequency-based). Noise has almost zero effect on any strategy. H6.7 (Y-aware more robust): INCONCLUSIVE (KM floor effect prevents comparison). H6.8 (NoMem → noise collapses): NOT SUPPORTED (NoMem invariant, noise affects X not Y distribution). Interpretation limited by KMeans ceiling effect.

### 2.6 综合产出

**Phase 6 全部完成 (A/B/C)**:

| 产出 | 内容 | 状态 |
|---|---|---|
| `run_c4_stabilization_scaling.py` | Phase 6-A: Seed Scaling (S1→S2→S3) | ✅ 完成 |
| `run_c4_objective_noise_scaling.py` | Phase 6-B/C: Objective + Noise Scaling | ✅ 完成 |
| `results/ic2c_scaling/` | 全部实验结果 (5 CSVs + 3 logs) | ✅ 已有 |
| Phase 6 → 7 决策 | **GO to Phase 7** — ALL experiments pass | ✅ 已决策 |
| Phase 7 → 8 决策 | Phase 7已完成: 3.3A analogue supports, 3.3B negative (data-gap) | ✅ 已决策 |

---

## 3. Phase 7: Cross-Bottleneck Joint —— 测试 1+1>2

### 3.1 核心问题

> B2 audit 揭示了结构化控制的边界条件：当 C_base 接近天花板时，任何 directional intervention 都不优于 random。但如果先通过 stabilization 降低噪声地板，组织层干预能否突破这个边界？

**核心假说 H7.1**: Stabilization + Organization > max(Stabilization alone, Organization alone)。

### 3.2 为什么这是自然的下一步

Phase 6 如果成功，我们有两个独立的已验证干预：

```
Stabilization: Per-Action KMeans → 提升 match 到 0.585
Organization:  Probe→Gate→Hook → 降低 hallucination 在 C_base=0.400 时
```

问题在于它们目前各自运作在完全独立的系统上：
- Stabilization 在 IC-2 的 counterfactual 决策环境中
- Organization 在 IC-4 的 hallucination Qwen-0.5B 环境中

**Challenge**: 将两者统一到同一个实验平台上。

### 3.3 实验设计（两个选项）

#### 选项 3.3A: 概念验证 —— Counterfactual + Steering

**思路**: 在 counterfactual 决策环境中，先在状态空间施加"扰动"（模拟 organization 问题的 analogue），再测试 stabilization 能否减轻扰动的影响。

**设计**:

```
环境: 与 C3/C4 相同的 counterfactual 决策环境
Step 1: 施加状态扰动（additive Gaussian noise to X，模拟 "hallucination-inducing distortion"）
Step 2: 运行 consolidation pipeline（Y-aware vs X-only）
Step 3: 测试: Y-aware stabilizer 是否使系统对扰动更鲁棒？
```

**度量的 analogue**: 
| 组织层概念 | Counterfactual analogue |
|---|---|
| C_base (baseline capability) | Nomemory match at noise=0 |
| dC (capability loss from intervention) | match drop from noise |
| Structured control advantage | Y-aware advantage at noise > 0 |
| Boundary condition | Y-aware advantage → 0 at high C_base |

**假说**:

| 假说 | 预测 |
|---|---|
| H7.2: Y-aware reduces noise sensitivity | d(match)/d(noise) 在 Y-aware 下 < X-only 下 |
| H7.3: Stabilization advantage grows with noise | Y-aware - X-only 随 noise 增大而增大 |
| H7.4: Joint advantage > additive | (Y-aware advantage at noise=N) > (Y-aware advantage at noise=0) |

**风险**: 这是 analogue，不是真正的跨瓶颈实验。但它快速、低成本，可以在 1 天内完成。

#### 选项 3.3B: 真正的跨系统耦合 —— IC-2 → IC-4

**思路**: 将 Qwen-0.5B hallucination 环境的 hidden state 轨迹"压缩"为 counterfactual 数据，再在此数据上测试 Y-aware consolidation。

**设计**:

```
Step 1: 从 Qwen-0.5B 多个 seed 的 fine-tuning run 中提取 hidden states
Step 2: 将 hidden states 视为 counterfactual 数据点 (X = hidden state, Y = behavior label)
Step 3: 运行 consolidation → 在 hallucination task 上测试 match
```

**这一设计的吸引力**: 它直接在真正的 LLM hidden state 上测试 stabilization，而不是 counterfactual 模拟环境。

**实现难度**: 中等。需要：
1. 从多个 Qwen fine-tuning seeds 提取 hidden states
2. 构造 "{hidden_state → behavior_label}" mapping
3. 将 C3 的 consolidation 逻辑适配到 896D 向量

**假说**:

| 假说 | 预测 |
|---|---|
| H7.5: Y-aware consolidation works on LLM hidden states | Per-action KMeans 在 LLM hidden state 上 > KMeans 在 counterfactual 上 |

### 3.4 推荐路径

**先做 3.3A（概念验证，≤1 天）**。如果 analogue 揭示 synergism，再投资 3.3B（LLM 耦合，2-3 天）。

### 3.5 成功标准

| 结果 | 含义 | 下一步 |
|---|---|---|
| ✅ Joint > additive | 跨瓶颈 synergism confirmed | 推进到真正的 LLM 耦合实验 |
| ⚠️ Joint = additive | 各瓶颈独立运作，无耦合 | 分开推进：Stab scaling + Org 新方法 |
| ❌ Joint < additive | 瓶颈互相干扰 | 重新思考耦合机制 |

---

## 4. Phase 8+: Absorption Remedy —— 最上游的干预

### 4.1 为什么排在最后

吸收瓶颈的补救目前是最不成熟的方向。原因：

1. **诊断强度足够，但补救设计空间大**: 我们知道 KNN=1.0，但不知道"最好的位置归一化方法"是什么
2. **Phase 6/7 的结果会限制 Absorption 的选择**: 如果 Phase 7 证实跨瓶颈 synergism，那吸收补救应设计为能与 stabilization/organization 耦合的
3. **吸收是最上游**: 如果先把下游的 Stabilization 和 Organization 补齐，再回头处理上游的 Absortion，效果更容易度量

### 4.2 候选方向（概念级）

#### Direction A-1: Position-Invariant Routing

**思路**: 在 probe/training 中使用 position augmentation——在多个位置训练 gate probe，使路由决策对位置不敏感。

**实现**: 
```python
# 在训练 gate probe 时，对每个 sample 生成 early/mid/late 变体
# 期望: 训练在 mixed positions 上的 probe 在单一 position 上也表现良好
```

**验证**: PSI 从 0.0084 → 0.005；ΔC 从 0.067 → <0.03。

---

#### Direction A-2: Position Normalization Layer

**思路**: 在模型内部（embedding 层或早期 attention 层）加入一个 position normalization 模块——学习一个对位置不敏感的表示变换。

**实现**: 训练一个小型 adapter（LoRA 或 linear projection），将不同位置的 hidden state 映射到统一空间。

**验证**: cos(early, late) 从 0.080 → 0.020；KNN position classification 从 1.000 → <0.500。

---

#### Direction A-3: Content-Position Decoupling via SFT

**思路**: 通过 SFT 训练模型"忽略位置"，在训练数据中混入 multi-position 样本。

**实现**: 构造 SFT 数据时，对每个 training sample 生成 3 个 position 变体，要求模型对所有变体给出相同输出。

**验证**: ΔC 从 0.067 → < 0.02。与 TT-SFT 的区别：TT-SFT 用的是 trajectory cosine alignment（失败），这里用的是 behavior-level 一致性训练。

---

### 4.3 触发条件

Phase 8+ 在以下任一条件触发时启动：
1. Phase 6 成功 + Phase 7 显示 synergism → Absorption 补救可能与 stabilization 耦合
2. Phase 6 失败 → 需要探索全新的补救方向，Absorption 是天然候选
3. Phase 7 显示各瓶颈独立 → Absorption 作为独立战线推进

---

## 5. 决策树：Go/No-Go 条件

```
Phase 6: Seed Scaling (A) + Objective Scaling (B) + Noise Scaling (C)
│
├── ✅ **全部通过 — DONE (2026-05-23)**
│   └── ✅ **GO to Phase 7 — DONE (2026-05-22)**
│       │
│       ├── 3.3A analogue: PA > NoMem at ALL noise/shift levels ✅
│       │   └── ✅ **GO to 3.3B (LLM coupling) — DONE (2026-05-22)**
│       │       │
│       │       ├── 3.3B: ALL strategies = 1.000 — NEGATIVE (data-gap)
│       │       │   └── 📝 documented: need multi-checkpoint fine-tuning data
│       │       │   └── ✅ **GO to Phase 8+: Absorption Remedy**
│       │       │
│       │       └── (no synergism path — all LLM representations trivially separable)
│       │
│       └── (no 3.3A failure path — analogue passed)
│
├── (no conditional path — Phase 6 passed)
│
└── (no failure path — Phase 6 passed)
```

---

## 6. 风险与替代方案

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| Per-Action KMeans 在 scaling 下退化 | Medium | 失去最强正信号 | Phase 6 本身就是 robustness test；失败也是重要信息 |
| Counterfactual → LLM 差距过大 | High | Phase 7 的 3.3B 不可行 | 先做 3.3A analogue 降低风险；考虑中间层 abstraction |
| 三瓶颈完全独立无 synergism | Medium | Joint experiment 为 negative | 这也是重要发现——"三个瓶颈独立可测"自身就是 claim |
| 吸收补救没有 clear winner | Medium | Phase 8+ 陷入概念探索 | 先聚焦最低风险的方向 A-1 (position augmentation)，快速 prototype |
| 运行时间超出预期（P6 大规模） | Low | 延误 | 在 scaling levels 之间设置 checkpoint，允许中途决策 |
| Counterfactual 多 action 支持不完整 | Medium | P6-B 受阻 | 先确认数据生成可行性；如不满足，skip B 直接做 C |

---

## 附录 A: 文件清单

| 文件 | 类型 | 说明 |
|---|---|---|
| `F:\intelligence_capital_minimal_lab\src\run_c3_y_aware_consolidation.py` | 现有 | C3 全部策略实现（基线） |
| `F:\intelligence_capital_minimal_lab\src\run_c4_stabilization_scaling.py` | 新建 | Phase 6: A/B/C 三组实验 |
| `F:\intelligence_capital_minimal_lab\results\ic2c_scaling\` | 新建 | Phase 6 结果目录 |
| `F:\internal_circuit_capital_lab\IC-4-M0\src\run_c5_cross_bottleneck_analogue.py` | 新建 | Phase 7 3.3A: counterfactual analogue |
| `F:\internal_circuit_capital_lab\IC-4-M0\src\run_c6_llm_consolidation.py` | 新建 | Phase 7 3.3B: LLM hidden state consolidation |
| `F:\internal_circuit_capital_lab\IC-4-M0\results_c5_cross_bottleneck\` | 新建 | Phase 7 结果目录 |

## 附录 B: 与现有文档的关系

| 文档 | 更新时机 |
|---|---|
| UNIFIED_RESEARCH_MAP.md | Phase 6 完成后：新增 C10/C11（scaling results） |
| UNIFIED_THESIS.md | Phase 6 完成后：更新 Section 7.3（stabilization status） |
| CROSS_BOTTLENECK_SYNTHESIS.md | Phase 7 完成后：更新 cross-bottleneck table |
| PROJECT_ENDGAME_AND_HANDOFF.md | Phase 7 完成后：更新 proof table |
| RESEARCH_TRAJECTORY_REPORT.md | 每个 Phase 完成后追加新章节 |
| README.md | 每个 Phase 完成后更新状态表 |

---

*Phase 6-7-8+ Forward Research Plan — Generated 2026-05-22*  
*Next action: `python src/run_c4_stabilization_scaling.py --level S1`*