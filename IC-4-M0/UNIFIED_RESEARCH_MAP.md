# 总研究地图 v4.0 — Capability Routing × Structural Fidelity

**版本**: v4.0 | **日期**: 2026-05-19 | **覆盖**: `internal_circuit_capital_lab` + `intelligence_capital_minimal_lab`

---

## 位置声明：我们不在"找现象"阶段了

三句话定义当前位置：

1. **我们已有可工作的内部增强机制。** `M3-v6` 给出了完整的 reference：内部状态可读、条件化 gate 可行、正确接法可复现。

2. **我们已不是"只有一个漂亮 demo"。** `M4`、`A2`、`P1.5` 开始告诉我们机制的稳健边界在哪里、哪些失败是 artifact、哪些是小样本构造问题、哪些是更深的 routing/integration 问题。

3. **两个项目已经开始汇合。** 它们都在逼近同一个问题：**能力/信息如何被正确整合进系统默认结构。**

---

## 第一层：已钉死的锚点（不可逆）

这些结论构成了项目当前的 solid ground。后续实验若出现矛盾，应先重新审查实验方法，再考虑修正这些锚点。

### A. Internal Circuit Capital Lab（IC-4）

| 锚点 | 结论 | 证据 |
|---|---|---|
| **A1 — M3-v6** | `last_prompt_token + logistic probe + hard gate + single-pass hook + model.generate()` 是 working reference mechanism | `metrics_raw.csv`：gate H = oracle H = 0.667；rank accuracy 0.74 |
| **A2 — M4 scoped robust** | reference 在 `standard/large/hard OOD` 场景 + `α∈{-0.8,-1.0,-1.2}` 下稳定 | M4 sweep matrix：3/3 场景、3/3 alpha 通过因果分离测试 |
| **A3 — Hard gate 是当前最优** | hard gate 比 soft/no-gate/open-loop 更稳 | A2 comparative audit：hard gate 是唯一在所有条件下不翻车的方案 |
| **A4 — P1.5 小样本 artifact** | P1 的 2/5 翻车是 15A+15U 构造样本不足，非机制缺陷 | `cos(steer,shuffled)` 从 0.788→0.439（seed=2），shuffled H 从 0.667→0.900（layer=13）；30A+30U 修复 |
| **A5 — M7-Lv2 能力路由** | 小模型内部存在 verification-like latent capability，但默认路由不通 | CPU ECHO 实验：oracle routing 正确将 85.7% 样本路由到"能做"或"不能做"的正确路径 |

**当前默认配置标准**: 30A+30U construction, layer=12, alpha=-1.0, hard gate.

### B. Intelligence Capital Minimal Lab（IC-2）

| 锚点 | 结论 | 证据 |
|---|---|---|
| **B1 — Learned compression > raw memory** | IC-2b 7 种 learned compressor 全部优于 Raw/Prototype/StateOnlyMemory | ResidualCompressor best_action_match ≈ 0.60 vs raw memory ≈ 0.20 |
| **B2 — No shortcuts consistently beat learned** | LearnedStateOnly shortcut 稳定打败所有 memory 策略 | IC-2b 排名：LearnedStateOnly > learned compressors ≫ raw memory |
| **B3 — Continual consolidation = bad debt** | 跨 seed 持续 KMeans 重写产生结构性坏资本，match 0.115 < random 0.33 | IC-2c：consolidated 在所有 5 步上均最低 |
| **B4 — Bad debt 机制已拆解** | 根因不是"压缩不够"，而是跨分布平均 + centroid 失衡 + wrong readout | IC-2c.1：imbalance 2.86→7.27；centroid drift ~1.0-1.3/step；k-NN cap 0.235 |

### C. 跨项目锚点

| 锚点 | 结论 |
|---|---|
| **C1 — 能力/信息存在但默认路由错误** | IC-4 说"能力存在，但默认 generation 不调用"；IC-2 说"信息存在，但错误 consolidation 把它变坏" |
| **C2 — 小样本构造脆弱性是已知并被表征的** | 15A+15U 的不稳定性已被量化，30A+30U 是可信的最小标准 |
| **C3 — Shortcut 会赢是因为它绕过了整个 readout 问题** | NoMemory 只用 3 个 action 频率达到 0.445；episodic 用 6000 traces 只达到 0.195 — 不是信息不够，是读out不匹配 |

---

## 第二层：当前最强问题

我们已经从"找 steering vector / 找 memory mechanism"升级到了更深的问题。

不再问：
- "有没有 steering vector？" → 有，且能工作
- "压缩有没有用？" → 有，但 readout 是关键瓶颈

现在问：

### Q1: 能力路由（Capability Routing）

> 模型内部有 latent capability（如 verification-like reasoning），默认不调用。如何把这种能力**条件化、选择性、正确地**路由到 generation 中？

**子问题**：
- 路由信号从哪里来？（probe？ECHO score？entropy？）
- 路由接法是什么？（hook injection？LoRA？prompt？）
- 路由错误比无路由更差——怎么保证 routing 本身不制造新问题？

**涉及**: IC-4 M7-H/L/E 线

### Q2: 结构保真（Structural Fidelity）

> 有用经验结构在持续更新/consolidation/rewriting 中，为什么会被破坏？破坏机制是什么？如何量化？能否设计出"保真"的更新方式？

**子问题**：
- 跨分布平均化是最主要的破坏机制吗？
- 什么样的 readout 才能从 raw traces 里正确提取信息？
- 是否可以设计一种"不退化"的 episodic-to-consolidated 转换？

**涉及**: IC-2c.1 → IC-2d 线

### Q3: 统一问题（Convergence）

> 这两个问题本质上是同一个问题：**有用的结构（能力 or 信息）如何在被整合进系统默认行为时不丢失、不被破坏？**

**子问题**：
- IC-4 的 routing gate 和 IC-2 的 consolidation 是否是同一类"选择什么信息流入默认行为"的问题？
- 是否可以共享分析语言？（如：信息保真度、路由效率、bad debt ratio）
- 是否可能有一个统一的 "structural integrity under integration" 框架？

---

## 第三层：下一阶段主线

### Phase 1: 巩固当前锚点（当前 → 1 周）

| 项目 | 任务 | 状态 |
|---|---|---|
| IC-4 | 30A+30U 成为默认标准；terrain manual v3.2 已更新 | ✅ 完成 |
| IC-4 | P1.5 小样本 artifact 确认；所有 tested seed/layer 在 30A+30U 下通过 | ✅ 完成 |
| IC-2 | IC-2c.1 根因拆解（NoMemory shortcut / episodic k-NN cap / consolidated drift） | ✅ 完成 |
| IC-2 | 全量复验：30A+30U 下所有 P1 配置确认无回归 | 📋 pending |
| IC-2 | 更新 THEORY.md 的 bad debt / false capital 条目（有了实验证据） | 📋 pending |

### Phase 2: 开能力路由主线（下周起，需 GPU）

| 项目 | 任务 | 依赖 |
|---|---|---|
| IC-4 | **M7-H: LoRA 路由注入** — 用小 probe 作为 router，训练 small LoRA 做 adaptive anti-hallucination | GPU |
| IC-4 | **M7-L: ECHO 验证训练** — 将 M7-Lv2 的 routing rule 显式训练为 predictor | GPU |
| IC-4 | **A2 升级**: 用 30A+30U 重跑 A2 的 soft/hard/open-loop 对比 | CPU 可行 |

### Phase 3: 开结构保真主线（继续 CPU）

| 项目 | 任务 | 依赖 |
|---|---|---|
| IC-2 | **IC-2d: Readout-matched episodic** — 不用 Euclidean k-NN，改用 IC-2b 的 learned compressor 做 episodic 的 readout | CPU |
| IC-2 | **IC-2e: Distribution-aware consolidation** — 不让 KMeans 跨 seed 平均，给每个 seed 独立的 prototypes | CPU |
| IC-2 | **IC-2f: Structural fidelity metrics** — 定义并测量 "consolidation 的信息损失率" | CPU |

### Phase 4: 两线汇合（中期，GPU 可用后）

| 任务 | 描述 |
|---|---|
| **统一 routing-consolidation 实验** | 在一个环境里同时测试 (a) 能力路由是否正确，(b) 积累的经验结构是否在退化成 bad debt |
| **小模型大能力最小 proof** | 选一个小模型，接 M7-H routing + IC-2d learned episodic readout，看能否在 multi-task 或 OOD 环境里显著优于原始模型 |
| **统一分析框架** | 定义 `routing efficiency`、`structural fidelity`、`bad debt ratio` 为跨项目标准指标 |

---

## 小模型大能力：现状与路径

### 我们现在有什么

**还不是成品，但有三块关键积木**：

| 积木 | 证据 | 意义 |
|---|---|---|
| **1. Latent capability 确实存在** | M7-Lv2：小模型内部有 verification-like 能力，只是默认不调用 | 不是空想——能力是潜伏存在的 |
| **2. 条件化内部增强已能工作** | M3-v6：读状态 → gate → 正确注入 → oracle-level 效果 | 不是重训整个模型——可以选择性增强 |
| **3. 已知道错误做法** | P1.5 artificial/sample 翻车、IC-2c 错误 consolidation、A2 soft gate 不优于 hard | 不是瞎试——在摸边界条件 |

### 最短 proof 路径

```
M3-v6 (reference) 
  → 30A+30U 标准化 (done)
  → M7-H LoRA routing injection (need GPU)
  → IC-2d learned readout for episodic (CPU)
  → 统一环境: routing + structural fidelity 联合测试
  → "小模型大能力" 最小 proof-of-concept
```

---

## 文件地图（v4.0）

### 总地图（本文件）

| 文件 | 说明 |
|---|---|
| **`UNIFIED_RESEARCH_MAP.md`** | 本文件 — 跨项目总研究地图 |

### IC-4 核心文档

| 文件 | 说明 |
|---|---|
| `IC4_PROJECT_TERRAIN_MANUAL.md` | v3.2 — 项目地形图，定义所有阶段、机制、边界 |
| `IC4_RESEARCH_PLAN_NEXT.md` | 三层研究计划（Anchors / Near-term / Branches） |
| `reports/IC4_P1_CROSS_VALIDATION_REPORT.md` | v1.2 — 跨 seed/layer 验证 |
| `reports/IC4_P15_FAILURE_ANALYSIS_REPORT.md` | 失败模式分析 + 小样本补丁测试 |
| `reports_m4_generalization/IC4_M4_GENERALIZATION_REPORT.md` | scoped robust 证据 |
| `results_m7/IC4_M7_FINAL_REPORT_CLEAN.md` | M7 机制解释 + 能力路由发现 |
| `results_m7/M7_LV2_ECHO_CPU_REPORT.md` | M7-Lv2 能力路由实验 |

### Intelligence Capital Minimal Lab 核心文档

| 文件 | 说明 |
|---|---|
| `..\intelligence_capital_minimal_lab\THEORY.md` | 理论框架（change capital, bad debt, false capital etc.） |
| `..\intelligence_capital_minimal_lab\IC2B_LEARNED_THROTTLING_REPORT.md` | learned compressor 比较（13 机制） |
| `..\intelligence_capital_minimal_lab\IC2C_EPISODIC_VS_CONSOLIDATED_REPORT.md` | episodic vs consolidated capital 实验 |
| `..\intelligence_capital_minimal_lab\IC2C1_ROOT_CAUSE_REPORT.md` | NoMemory/Episodic/Consolidated 根因拆解 |

---

## 三个项目的统一叙事（对外版本）

> **我们研究的不是"怎么让模型少犯错"，而是"模型内部已有的能力/信息为什么没有被正确使用"。**
>
> 在一条线上（`IC-4`），我们发现小模型内部存在 latent verification capability，可以通过条件化 routing gate 激活——hard gate 在所有 tested 条件下达到 oracle 级别，稳健边界正在被逐步确认。
>
> 在另一条线上（`intelligence_capital_minimal_lab`），我们发现错误 consolidation 会把有用经验结构变成 bad debt——cross-distribution averaging + wrong readout 是主要破坏机制。
>
> 两条线正在汇合：它们都在逼近同一个问题——**有用结构（能力或信息）如何被正确整合进系统的默认行为，而不是在路由中被忽略或在中继中被破坏。**
>
> 下一步：routing injection（GPU）+ learned episodic readout（CPU）→ 小模型大能力最小 proof-of-concept。

---

## 一句话身份

> **IC-4 + intelligence_capital_minimal_lab = 一个正在成形的、关于"结构性能力/信息如何在系统中被正确路由和保真"的研究项目。**
>
> 当前阶段不是"找现象"或"做 demo"——而是"机制工程可行性验证"。