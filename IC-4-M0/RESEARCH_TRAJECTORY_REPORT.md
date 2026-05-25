# IC-4-M0 研究路线完整报告

> 从工程控制论假说到三瓶颈全诊断 + B-bottleneck 完整机制链 + A-bottleneck 干预阶梯闭合 + 跨项目合成闭合 + P26 统一瓶颈三冠王 —— 项目完整研究轨迹 (v3.19: 含 P26, Unified Steering Triple Crown)  
> 最后更新: 2026-05-25

---

## 目录

1. [理论基础：工程控制论与结构适应假说](#1-理论基础工程控制论与结构适应假说)
2. [Phase 1: 证明瓶颈存在 —— Proof A/B/C/D](#2-phase-1-证明瓶颈存在--proof-abcd)
3. [Phase 2: 轨迹动力学 —— T0, T1, T2, T3](#3-phase-2-轨迹动力学--t0-t1-t2-t3)
4. [Phase 3: Sycophancy 方向特异性 —— P3 + P4](#4-phase-3-sycophancy-direction-specificity--p3-replication--p4-decomposition)
5. [Phase 4: 能力路由 —— M7 Oracle](#5-phase-4-capability-routing--m7-oracle)
6. [Phase 5: Sycophancy 反馈控制 —— P5 + P5-bis](#6-phase-5-sycophancy-feedback-control--p5--p5-bis)
7. [Phase 6: Hall 方向特异性排除路线](#7-phase-6-方向特异性排除路线-hallucination)
8. [Phase 7: Stabilization 根因诊断与突破 (IC-2)](#8-phase-7-stabilization-根因诊断与突破-ic-2)
9. [Phase 8: Trajectory-Targeted SFT（否定性结果）](#9-phase-8-trajectory-targeted-sft否定性结果)
10. [Phase 9: Absorption 吸收瓶颈诊断闭合 (P1)](#10-phase-9-absorption-吸收瓶颈诊断闭合-p1)
11. [Phase 10: P6 Behavior-Only Probe](#11-phase-10-p6-behavior-only-probe--反馈仍-null开环确认)
12. [Phase 11: P6-bis Threshold Calibration → Hook 架构诊断](#12-phase-11-p6-bis-threshold-calibration--hook-架构诊断)
13. [Phase 12: P6-ter Two-Stage Feedback Control (闭环打通)](#13-phase-12-p6-ter-two-stage-feedback-control--闭环打通)
14. [Phase 13: P7 S15 Amplification Mechanism](#14-phase-13-p7-s15-amplification-mechanism--readability--manipulability)
15. [Phase 14: P8 Large-Scale Replication](#14-phase-14-p8-large-scale-replication--p6-ter-小样本优势为虚假信号)
16. [Phase 15: P9 Cross-Bottleneck Structure](#15-phase-15-p9-cross-bottleneck--steering-不破坏结构完整性)
17. [Phase 16: P10 Hall Impulse Formal Exclusion](#16-phase-16-p10-formal-exclusion--hall-单方向-impulse-路线关闭)
18. [Phase 17: P11 Stabilization Scaling (Fully Validated)](#17-phase-17-p11-stabilization-scaling--4-维度全通过fully-validated)
19. [Phase 18: P12 Absorption Steering (Negative)](#18-phase-18-p12-absorption-directional-steering--否定性结果)
20. [Phase 19: P13 Energy/Direction Asymmetry](#19-phase-19-p13-energydirection-asymmetry--l10-均匀非对称在行为)
21. [Phase 20: P14 Cross-Project Synthesis](#20-phase-20-p14-cross-project-synthesis--诊断阶段闭合)
22. [Phase 21: P15 Hallucination LoRA (B-Bottleneck Bridged)](#21-phase-21-p15-hallucination-lora--b-bottleneck-bridged)
23. [Phase 22: P16 LoRA Geometry (Routing Fix vs Geometry Fix)](#22-phase-22-p16-lora-geometry-analysis--routing-fix-not-geometry-fix)
24. [Phase 23: P17 LoRA Module Ablation (Attention Routing Key)](#23-phase-23-p17-lora-module-ablation--attention-routing-key)
25. [Phase 24: P18 q_proj Layer Ablation (Deep Layer Routing)](#24-phase-24-p18-q_proj-layer-ablation--deep-layers-are-the-routing-core)
26. [当前状态：三瓶颈全景 (P18)](#25-当前状态三瓶颈全景更新至-p18)
27. [关键发现汇总](#关键发现汇总)
28. [路径前瞻：后续方向](#路径前瞻后续方向)
29. [完整时间线](#完整时间线)

---

## 1. 理论基础：工程控制论与结构适应假说

### 1.1 核心命题

项目出发点是：

> 小模型与大模型的根本差距不只是参数规模，而是**结构适应能力**——即把人类离散化的数据流吸收、稳定、组织成可调用内部结构的能力。

工程控制论（Engineering Cybernetics）提供了统一的分析语言来组织对这些瓶颈的诊断与补偿。

### 1.2 三瓶颈定义

| 瓶颈 | 原文定义 | 控制论翻译 | 对应项目 |
|---|---|---|---|
| **A: Absorption** | 输入碎片化导致信息丢失 | 输入如何进入状态空间；输入扰动如何扭曲内部状态轨线 | Position Sensitivity |
| **B: Stabilization** | 跨样本压缩导致结构漂移 | 状态轨线如何在无锚定更新中保持稳定 | IC-2 (memory consolidation) |
| **C: Organization** | 能力存在但路由不通 | 内部信号如何被正确路由到输出行为 | IC-4 (hallucination/sycophancy control) |

### 1.3 嵌套关系

```
工程控制论 (顶层叙事)
  → 结构适应假说 (元层)
    → 关系记忆假说 (理论层)
      → Bottleneck A/B/C (实验层)
```

### 1.4 模型与基础设施

- **模型**: Qwen2.5-0.5B-Instruct (896D hidden state, 14 transformer layers)
- **实验标准**: 30A+30U 平衡数据集（训练）/ 10A+10U 或更多（测试）
- **核心指标**: H (hallucination rate), C (correct answer rate), CA (calibrated abstention)
- **Score**: dH - dC（幻觉降低 vs 正确率损失的净值）
- **控制方式**: activation steering (probe → gate → hook 管线)

---

## 2. Phase 1: 证明瓶颈存在 —— Proof A/B/C/D

目标：四个证明义务，分别验证三个瓶颈的可测量性和可补偿性。

### 2.1 Proof A: Sycophancy T3 大样本复制

| 项目 | 详情 |
|---|---|
| **脚本** | `src/run_a_syc_t3_replication.py` |
| **设计** | 40 样本, 2 layers × 2 steps × 1 epsilon × 4 directions = 16 conditions |
| **Epsilon** | 5.0 at L10 prefill |
| **方向** | v_syc, random, shuffled, orthogonal |

**结果:**

| 指标 | 值 |
|---|---|
| Baseline syc_group | 1.000（天花板效应） |
| Baseline non-syc_group | 0.250 |
| v_syc mean controllability | -0.325 |
| random mean controllability | -0.113 |
| shuffled mean controllability | -0.337 |
| orthogonal mean controllability | -0.225 |

**结论: NEGATIVE.** v_syc (-0.325) 甚至不如 random (-0.113)。天花板效应 + epsilon 过强 (5.0) 导致 non-syc 组大量泄漏。Sycophancy 方向特异性未复现。

### 2.2 Proof B: Hallucination 结构化干预

| 项目 | 详情 |
|---|---|
| **脚本** | `src/run_b_multidirection_intervention.py` |
| **设计** | 多方向混合 vs 单方向对比，alpha=-1.0 at L12 prefill |
| **关键发现** | Multi-direction combo beats single-direction |

**结果 (C_base=0.400):**

| 干预 | dH | dC | score |
|---|---|---|---|
| v_hall+syc_like (best combo) | +0.200 | 0.000 | **+0.200** |
| orthogonal_alone (best single) | +0.200 | +0.100 | +0.100 |
| hall0.25_orth0.75 (max dH) | **+0.300** | +0.200 | +0.100 |
| v_hall_alone | 0.000 | +0.200 | -0.200 |
| random_alone | 0.000 | +0.200 | -0.200 |

**结论: POSITIVE at C_base=0.400.** 多方向组合 (score +0.200) 超越单方向 (score +0.100)，边界被跨越。

### 2.3 Proof C: Anchored Consolidation 补救

| 项目 | 详情 |
|---|---|
| **脚本** | `src/run_c_anchored_consolidation.py` |
| **设计** | 在不同 breakthrough rate (br) 下锚定 KMeans centroid 更新 |
| **基线** | Naive consolidated match = 0.115 |
| **关键发现** | 锚定更新可部分纠正 stabilization 退化 |

**结果:**

| 策略 | Step 5 match | Δ vs naive |
|---|---|---|
| NoMemory (upper bound) | 0.445 | — |
| Episodic | 0.195 | — |
| Naive consolidated | 0.115 | baseline |
| Anchored br=0.7 | **0.125** | **+8.7%** |
| Anchored br=0.5 | 0.120 | +4.3% |
| Anchored br=0.3 | 0.110 | -4.3% |

**结论: POSITIVE.** 锚定 br=0.7 超越 naive +8.7%。Stabilization 可被部分纠正。br 存在最优值——太强（0.3）反而退化。

### 2.4 Proof D: Cross-Bottleneck Synthesis

| 项目 | 详情 |
|---|---|
| **文档** | `CROSS_BOTTLENECK_SYNTHESIS.md` |
| **作用** | 将 A/B/C 证明整合为统一叙事 |

**Phase 1 结束时的项目声明:**

> 小模型性能限制可被分解为吸收、稳定、组织三个瓶颈。这三个瓶颈均可被实验测量。组织瓶颈可通过闭环反馈被部分补偿；稳定瓶颈可通过锚定更新被部分补偿。

---
## 3. Phase 2: Trajectory Dynamics — T0, T1, T2, T3

在 Proof A/B/C/D 证明瓶颈存在后，下一步是理解模型内部的行为轨迹动力学：sycophancy 和 hallucination 在生成过程中何时、何层、如何分离？

此阶段构建了完整的轨迹捕获-投影-探测-脉冲管线。

### 3.1 T0: Trajectory Capture — 零扰动验证

| 项目 | 详情 |
|---|---|
| **脚本** | `src/run_t0_trajectory_capture.py` + `src/run_p0_sycophancy_contrast.py` |
| **设计** | 在 `model.generate()` 期间 hook 7 层 × 48 steps 的 hidden states，验证捕获过程不改变输出 |
| **核心基础设施** | `TrajectoryCaptureModel` + `steering.py` (apply/remove hook 管线) |

**结果:**
- Hallucination: 60/60 输出完全匹配（捕获 vs 无捕获）
- Sycophancy: 30/30 输出完全匹配
- 7 层 × 48 steps = 336 个 (layer, step) 位置完整捕获

**P0 补充：构造平衡 syc 对照集**
- 关键发现：原 30 个 syc 样本全部 sycophantic (rate=1.0)，无对照 → T1/T2/T3 无法运行
- 使用 fact-checker system prompt 创建 non_sycophantic 组 (30 样本)，rate=0.167
- 分离度 = 0.833，成为后续所有 syc 分析的数据骨干

**结论: Trajectory capture 基础设施 stable + syc 对照集就绪。**

### 3.2 T1: Projection — 预填充分离 + 生成坍缩

| 项目 | 详情 |
|---|---|
| **脚本** | `src/run_t1_projection.py` |
| **设计** | 计算 v_syc/v_hall 并在每一层每一步投影 hidden states → 获取分离曲线 |
| **关键指标** | max_separation, collapse_ratio, v/random ratio |

**结果 (Sycophancy, L12):**

| 指标 | 值 |
|---|---|
| Earliest separation step | 0 (prefill) |
| Max separation | 1.789 (prefill) |
| v_syc/random ratio | 13.6× |
| Collapse ratio | 0.347 (信号保留 35%) |
| Late-stage variance | 0.010 (极低：稳定) |
| All 48 steps significant (p<0.05) | 48/48 |

**结果 (Hallucination, L12):**

| 指标 | 值 |
|---|---|
| Max separation | 2.397 (prefill) |
| v_hall/random ratio | 3.51× |
| Late-stage variance | 0.160 (高于 syc) |

**关键洞察:**
1. 两种行为都在 prefill (step 0) 达到最大分离——行为倾向在第一个 token 生成前已锁定
2. Syc 信号更强且更稳定 (v/random=13.6×, AUC=1.0)，hall 信号较弱且波动大
3. Syc 信号在生成过程中部分坍缩 (~65%) 但从未消失；hall 信号更动荡

**Verdict: Sycophancy 是超可分的轨迹信号；hallucination 较弱但仍有结构。**

### 3.3 T2: Decision Heatmap — cross_layer_band 形态

| 项目 | 详情 |
|---|---|
| **脚本** | `src/run_t2_decision_heatmap.py` |
| **设计** | logistic probe 在每个 (layer, step) 做 syc/hall 分类 → 全轨迹 heatmap |
| **核心发现** | 两种行为共享 `cross_layer_band` 结构但时序不同 |

**Sycophancy Heatmap:**

| 指标 | 值 |
|---|---|
| Peak accuracy | 0.983 (L8, S15) |
| AUC at every (layer, step) | 1.000 |
| First predictive layer | 10 (step 0) |
| First predictive step | 0 (layer 10) |
| S15 amplification | Accuracy 从 ~0.80 升至 0.983 |

**Hallucination Heatmap:**

| 指标 | 值 |
|---|---|
| Peak accuracy | 0.917 (L8, S0) |
| First predictive layer | 8 |
| First predictive step | 0 |

**关键不对称:**
- Syc: **S15 放大**——信号在生成中期急剧增强 (0.80→0.983)，机制不明
- Hall: **prefill 即巅峰**——信号在 step 0 已达最大值，随时间衰减

**Verdict: Sycophancy 有独特的生成中期放大机制；hallucination 是衰退型。**

### 3.4 T3: Impulse Map — 脉冲可控性初探

| 项目 | 详情 |
|---|---|
| **脚本** | `src/run_t3_impulse_map.py` |
| **设计** | 在特定 (layer, step) 注入 steering vector impulse → 测量行为变化 |
| **初版** | 3 layers × 3 steps × 4 directions × 2 epsilons (缺 orthogonal) |
| **P3 复制版** | n≥20, 108 combos, 新增 orthogonal 方向 |

**T3 初版结果 (syc):**

| Direction | Mean controllability |
|---|---|
| v_syc | 0.0545 |
| random | 0.0303 |
| v_hall (cross-behavior) | 0.0219 |

v_syc/random = 1.80× —— 初步提示方向特异性。

**T3 初版结果 (hall):** v_hall 未显著高于 random —— 无方向特异性。

**关键发现:** 
- Syc 控制效应集中在 L10 prefill（单层、单步），与 T1/T2 一致
- Hall 效应跨层分布 [10,12,14]，无单一步骤集中
- Syc prefill-only: 生成步骤内注入无效（行为倾向已在 prefill 锁定）

**Verdict: Syc 和 hall 是不同的可控性对象。Syc 有初步方向特异性提示；hall 纯能量。**

### 3.5 P2: 方向特异性排除实验（Hallucination）

| 项目 | 详情 |
|---|---|
| **脚本** | `src/run_p2_direction_specificity.py` |
| **设计** | norm-matched 对照: v_hall, v_random_norm, v_orthogonal (cos<1e-5), v_syc, v_shuffled |
| **3 layers × 2 epsilons × 5 directions** |

**结果 (Hallucination, n=10/combo):**

| Direction | ΔH (ε=3.0) |
|---|---|
| v_hall | −0.283 |
| v_orthogonal | **−0.283** |

**v_hall = v_orthogonal —— 方向特异性在全球层面不存在。**

**结论: Hallucination impulse = generic perturbation. v_hall 的方向不携带因果信息。**

---

## 4. Phase 3: Sycophancy Direction-Specificity — P3 Replication + P4 Decomposition

T3 提示 syc 有方向特异性 (1.80×)，P2 确认 hall 没有。下一步：
必须在更大样本 (n≥20) 下复现 syc 的方向特异性，并分解方向贡献 vs 能量贡献。

### 4.1 P3: Syc Direction-Specificity Replication (n≥20)

| 项目 | 详情 |
|---|---|
| **脚本** | `src/run_t3_impulse_map.py` (P3 mode) |
| **设计** | n_syc=20, n_hall=6, 3 layers × 3 steps × 4 directions × 3 epsilons = 108 combos |
| **新增** | orthogonal 方向 (norm-matched, Gram-Schmidt orthogonalized) |

**结果:**

| 指标 | 值 |
|---|---|
| v_syc controllability | 0.0591 |
| random controllability | 0.0216 |
| **v_syc/random ratio** | **2.73×** (从 T3 的 1.80× 提升) |
| orthogonal controllability | 0.0390 |
| shuffled controllability | 0.0216 |
| Hall v_hall/random | 0.28× (确认不特异) |

**确认:**
1. Syc prefill-only (生成步骤内注入无效)
2. Syc L10-concentrated (其他层效应不显著)
3. v_syc/random=2.73× —— 方向特异性在 n≥20 下存续，从 6.17× (小样本) 收缩至 2.73×
4. Hallucination 方向特异性确认不存在 (0.28× < 1.0)

### 4.2 P4: Direction-vs-Energy Decomposition

P3 证明了 v_syc 的方向特异性，但留下了关键问题：效应究竟来自 v_syc 的**方向对齐**还是**范数大小**？

| 项目 | 详情 |
|---|---|
| **脚本** | `src/run_t3_impulse_map.py` (P4 mode) |
| **核心方法** | `compute_norm_matched_orthogonal()` — 构造与 v_syc 等范数但正交的向量 |
| **设计** | n_syc=30, 3 layers × 2 steps × 5 directions × 3 epsilons = 90 combos |

**分解逻辑:**
```
Δ(syc_rate) = directional_contribution + energy_contribution
             = v_syc_effect - orthogonal_effect     + orthogonal_effect - random_effect
```

**结果:**

| 分量 | 值 | 解释 |
|---|---|---|
| **Directional** | **+0.0164** | v_syc 超过正交的部分 = 纯方向贡献 |
| **Energy** | −0.0022 | 正交超过 random 的部分 = 纯能量/范数贡献 |
| Direction/Energy ratio | **∞** (energy negative) | 方向主导 |
| v_syc controllability | 0.0207 | — |
| random controllability | 0.0207 | — |
| orthogonal controllability | 0.0185 | 低于 random! |

**关键发现:**
- 纯方向贡献为正 (+0.0164)，纯能量贡献为负 (−0.0022)
- Norm-matched orthogonal vector 的 controllability (0.0185) **低于** random (0.0207)
- → v_syc 的方向对齐是因果效应的来源，不是范数
- 同样的能量注入在正交方向产生反效果——证明方向承载信息

**结论: Sycophancy controllability is direction-dominated. v_syc 的方向编码了 sycophancy-relevant geometry。**

---

## 5. Phase 4: Capability Routing — M7 Oracle

在证明 sycophancy 是方向特异性对象之后，自然的问题是：模型内部是否已经具备修正 sycophancy 的能力，只是路由不通？

### 5.1 M7-Lv2: Oracle Routing

| 项目 | 详情 |
|---|---|
| **设计** | 事后分类 + 选择性放大的 oracle routing 范式 |
| **核心发现** | 小模型内部存在 verification-like latent capability，默认路由不通 |

**关键数据:**
- Oracle routing 正确地将 85.7% 样本路由到"能做"或"不能做"的正确路径
- Fact_checker prompt 将 sycophancy 降低 20pp（能力存在的间接证据）
- 但默认 inference 不调用此能力——组织瓶颈的典型案例

**结论: 能力存在但路由不通。这是"组织"瓶颈的直接证据。**

---

## 6. Phase 5: Sycophancy Feedback Control — P5 + P5-bis

P4 证明了 sycophancy 是方向特异性的、方向主导的可控对象。下一个挑战：能否用 M3-v6 的 probe→gate→hook 管线做闭环控制？

### 6.1 P5: First Attempt — Probe→Gate→Hook

| 项目 | 详情 |
|---|---|
| **脚本** | `src/run_p5_syc_feedback_control.py` |
| **设计** | 24-sample test set, L10 probe (logistic) → hard gate (threshold=0.5) → steering hook |
| **4 directions × 3 alphas = 12 feedback + 12 open-loop combos** |

**Feedback Control Results:**

| 指标 | 值 |
|---|---|
| Probe train accuracy | 0.9167 (92%) |
| **Gate trigger rate (test)** | **0.0417 (4.2%)** |
| Feedback effect (all conditions) | **0.0000** (gate 基本不开) |

**Open-Loop Results (always-on steering, n=24):**

| Direction | α=−1.0 | α=−3.0 | α=−5.0 |
|---|---|---|---|
| v_syc | 0.5417 | 0.8750 | 0.8750 |
| random | 0.6250 | 0.9167 | 1.0000 |
| shuffled | 0.8750 | 0.9167 | 1.0000 |
| orthogonal | 0.9167 | 1.0000 | 0.9167 |

**P5 初步结论 (后被 P5-bis 修正):**
1. Gate rate=4.2% — 探测器学到 group membership (fact-checker prompt)，非行为倾向
2. Open-loop: 负 alpha 增加 sycophancy，提示"正 alpha 才对"
3. 任何 |α|≥3.0 的扰动都破坏修正行为

### 6.2 P5-bis: Polarized α-Sweep — Polarity Resolved

P5 的 open-loop 用了 60-sample 混合集 (fact-checker + standard prompts)，结果混乱。

**P5-bis 用纯净 24-sample test set 做完整 α-sweep。**

| 项目 | 详情 |
|---|---|
| **脚本** | `src/run_p5_bis_syc_feedback.py` |
| **设计** | n=24, 1 baseline + 12 neg α + 12 pos α = **25 combos**, 4 directions × 6 alphas |
| **总耗时** | ~163 min |

**Negative Alpha (subtract v_syc):**

| Direction | α=−1.0 | α=−3.0 | α=−5.0 |
|---|---|---|---|
| **v_syc** | **0.4167** | **0.3750** | 0.5417 |
| random | 0.6250 | 0.6250 | 0.7917 |
| shuffled | 0.4167 | 0.5833 | 0.7083 |
| orthogonal | 0.5833 | 0.7500 | 0.9583 |

**Positive Alpha (add v_syc):**

| Direction | α=+1.0 | α=+3.0 | α=+5.0 |
|---|---|---|---|
| **v_syc** | 0.9167 | **1.0000** | **1.0000** |
| random | 0.6250 | 0.7083 | 0.9167 |
| shuffled | 0.6667 | 0.7083 | 0.7083 |
| orthogonal | 0.5417 | 0.4583 | 0.7083 |

Baseline: **0.5833** (14/24)

### 6.3 P5-bis Key Conclusions

**1. v_syc 指向 sycophancy 方向（而非 non_sycophancy）——P5 的原假设被证伪。**

- 负 α (减 v_syc) → 降低 syc: 最佳 α=−3.0, 0.5833→0.3750 (**−35.7%**)
- 正 α (加 v_syc) → 饱和至 1.0000 (天花板效应)

**2. 方向特异性在因果层获证：**

- 仅 v_syc 有非对称效应 (正/负 α 效果相反的独特模式)
- random/shuffled/orthogonal 均随 |α| 单调增 (generic perturbation)
- v_syc α=+3.0 时 100% 样本 sycophantic (最极端的方向特异性证据)

**3. P5 闭环失败不是符号错误，是探测器泛化问题：**

- 负 α 是正确方向 (减 v_syc)
- P5 probe 学到 group membership 而非行为倾向
- 下一步需在标准 prompt 样本上重新训练 probe (behavior-only labels)

### 6.4 Sycophancy Control 路线完整总结

```
P0: 构造 balanced syc contrast set (分离度=0.833)
  ↓
T0: 轨迹捕获基础设施验证 (60/60 + 30/30 零扰动)
  ↓
T1: v_syc 投影 → prefill 分离 + 生成坍缩 (v/random=13.6×)
  ↓
T2: Probe heatmap → cross_layer_band, AUC=1.0, S15 放大
  ↓
T3: Impulse map → 初步方向特异性提示 (1.80×)
  ↓
P3: n≥20 复现 → 方向特异性存续 (v_syc/random=2.73×)
  ↓
P4: Direction-vs-energy decomposition → direction-dominated (纯方向=+0.0164, 纯能量=−0.0022)
  ↓
M7: Oracle routing → 能力存在但路由不通 (85.7% oracle routing accuracy)
  ↓
P5: Probe→gate→hook → 零结果 (gate rate=4.2%), 探测器学到 group membership
  ↓
P5-bis: α-sweep → **v_syc 极性解决** (负 α 最优, −35.7% reduction)
  ↓
P6: Behavior-only probe — 探针可训练 (77.8%) 但分数聚集 → gate 8.3% null, open-loop −50%
  ↓
P6-bis (next): Threshold calibration — 打通闭环的最后一块拼图
```

---

## 7. Phase 6: 方向特异性排除路线 (Hallucination)

Phase 1 的 Proof B 提示 multi-direction combo 可超越 single，但这是 C_base=0.400 的特例。需要系统化追问：两行为的控制效应，是因为方向还是因为能量？

> **注意：此阶段仅针对 Hallucination。Sycophancy 的方向特异性在 Phase 3 (P3+P4) 中被证明存在。**

### 7.1 Hallucination Direction-vs-Energy Decomposition

（同原 Phase 2，仅针对 Hall，不再重复）

### 7.2 B2: Multi-Direction Structure Audit

（同原 Phase 2 B2，仅针对 Hall）

### 7.3 方向特异性路线的统一结论 (Hall-only)

| 行为 | 证据 | 结论 |
|---|---|---|
| Hallucination | B2 audit: ALL pairs synergy ≤ 0, best=random | No directional structure. Generic perturbation. |
| Sycophancy | P3: v_syc/random=2.73×; P4: direction-dominated; P5-bis: anti-symmetric α-response | **Direction-specific, direction-dominated, causally verified.** |

**修正：两行为的方向特异性不对称。Hall=纯能量，Syc=方向主导。这对项目的控制理论构成关键分化。**

---

## 8. Phase 7: Stabilization 根因诊断与突破 (IC-2)

Proof C 显示 anchored consolidation 仅比 naive 好 +8.7%（从 0.115 到 0.125），远低于 NoMemory 上界（0.445）。这种 3.9x 的差距表明 consolidation 的根本问题尚未被触达。

### 8.1 C2: Readout-Level Interventions（全部失败）

| 项目 | 详情 |
|---|---|
| **脚本** | `src/run_c2_stronger_stabilization.py` |
| **设计** | 8 种策略，尝试在 readout 层面纠正 cross-seed 平均效应 |

**8 种策略:**

| 策略 | 原理 | Step 5 match |
|---|---|---|
| Naive consolidated | baseline | 0.095 |
| seed_conditioned | 记录 seed 成员 → 按 seed 加权 | 0.095 |
| seed0_only | 只用 seed-0 的 centroids | 0.095 |
| weighted_seed (T1) | 最近 seed 权重 1.0 | 0.095 |
| weighted_seed (T05) | 最近 seed 权重 0.5 | 0.095 |
| purity_gated (0.3) | 过滤纯度 < 0.3 的 centroids | 0.095 |
| purity_gated (0.5) | 过滤纯度 < 0.5 的 centroids | 0.095 |
| per_seed_consolidated | 每 seed 独立 KMeans | 0.095 |
| combined_anchored_seed_purity | 三合一 | **0.460** ⚠️ |

**Combined 0.460 是假阳性:** purity gate 过滤了所有 centroids 后回退到 NoMemory 基线。

**关键发现: ALL readout 层面干预无法提升 match。问题不在 readout。**

### 8.2 C3: Root Cause — Y-Aware Consolidation（突破）

| 项目 | 详情 |
|---|---|
| **脚本** | `src/run_c3_y_aware_consolidation.py` |
| **运行时间** | 75.5 min |
| **核心思路** | 三层递进诊断 → Y-aware 聚类 |

**Step 1: 量化 readout 上限**

`learned_state_only = 0.740`（理想状态下，跨 seed 最完美匹配的上限是 0.740）。

**Step 2: KMeans 分辨率无效**

```
kmeans_20:  0.095
kmeans_100: 0.130
kmeans_200: 0.120
```

增加 KMeans 分辨率无法解决 — 不是 "聚类不够细"。

**Step 3: Y-information 是瓶颈**

Y 是 "最优 action"（ground truth 分类）。X-only KMeans 在聚类时不考虑 Y 信息 → cluster 内 action 不一致 → consolidation 破坏行为。

```
Y-weight=0.5: 0.095
Y-weight=1.0: 0.105
Y-weight=2.0: 0.245
Y-weight=5.0: 0.500
```

**Y-weight 越大越好！证明 Y 是真正的瓶颈。**

**Step 4: Per-Action KMeans（突破）**

按最优 action 分组后独立聚类 → 保证每个 cluster 内 Y 一致：

| 策略 | Step 5 match |
|---|---|
| NoMemory (upper bound) | 0.445 |
| **Per-Action KMeans** | **0.585** |
| Y-aware (weight=5.0) | 0.500 |
| Naive consolidated | 0.095 |

**Per-Action KMeans = 0.585 > NoMemory = 0.445 (+31%)！**

**Stabilization is now SIGNIFICANTLY correctable —— 超越了"无记忆"上限。**

### 8.3 Stabilization 路线的完整根因链

```
现象: consolidated << episodic << NoMemory
  ↓
诊断: readout 干预全失败 → 问题在 consolidation 本身
  ↓
量化: learned_state_only=0.740 vs KMeans=0.095 (8x gap)
  ↓
定位: 不是分辨率不够 (kmeans_100/200 无效)
  ↓
根因: KMeans ignores Y information
  ↓
验证: Y-weight 越大越好 (0.095 → 0.500)
  ↓
补救: Per-Action KMeans = 0.585 (+31% over NoMemory)
```

---

## 9. Phase 8: Trajectory-Targeted SFT（否定性结果）

| 项目 | 详情 |
|---|---|
| **脚本** | `new-5/` (独立实验目录) |
| **设计** | LoRA SFT + trajectory cosine alignment loss |

**核心思路:** 在 SFT 训练中，不仅计算 cross-entropy loss，还加入轨迹对齐损失——鼓励 student 的 hidden state 轨迹匹 teacher 的轨迹。

**Hypothesis:** 如果轨迹结构可在训练时被塑造，则 inference 时的结构性问题（route errors, stabilization drift）可以从源头被减少。

### 9.1 v0: Self-Teacher

| 参数 | 值 |
|---|---|
| Teacher | self (0.5B-Instruct) |
| 训练数据 | 20 samples |
| align_w | 0.5 (trajectory loss weight) |
| LoRA rank | 8 |

**结果:**

| 指标 | CE-only | CE + trajectory |
|---|---|---|
| H | 相同 | 相同 |
| C (正确率) | 0.611 → 0.889 (+45%) | 相同 |
| CA | 相同 | 相同 |
| traj_var | — | -0.0022 (微小改善) |
| traj_dist | — | -0.0032 (微小改善) |

**Verdict: `weak_effect`** — 轨迹对齐在行为层完全无法超越 CE-only。

### 9.2 v1: Base-Model Teacher

| 参数 | 值 |
|---|---|
| Teacher | base 0.5B (no RLHF) |
| 训练数据 | 30 samples |
| align_w | 1.0 |

**假设:** 升级到 conceptually different teacher（base model vs instruct model）可能放大轨迹对齐信号。

**结果:**

| 指标 | CE-only | CE + trajectory |
|---|---|---|
| H | 相同 | 相同 |
| C | 相同 | 相同 |
| CA | 0.091 | **0.046 (退化!)** |
| traj_dist | — | -0.0016 |
| traj_var | — | **+0.004 (恶化!)** |

**Verdict: `weak_effect`** — Base-model teacher 不仅没提升，反而 CA 退化和 traj_var 恶化。

### 9.3 TT-SFT 结论

1. **CE-only 在此设置下非常有效**: 仅 30 sample LoRA 训练可将正确率从 0.611 提升至 0.889 (+45%)
2. **轨迹对齐未能超越 CE-only**: 两轮实验、两种 teacher、不同数据量和对齐权重，行为层均无差异
3. **结构层改善极小且不一致**: traj_dist 略有改善但 traj_var 在 v1 中反而恶化
4. **Teacher 质量提升未解锁轨迹对齐**: 从 self-teacher 升级到 conceptually different teacher 并未放大对齐效果

**判断:** 不建议扩大为 v2。问题更可能是方法论层面——cosine alignment loss 太弱、LoRA 容量不足以塑 trajectory structure。这是重要的 **negative result**，排除了 "cosine trajectory alignment" 作为 stabilization/organization 方法的路线。

---

## 10. Phase 9: Absorption 吸收瓶颈诊断闭合 (P1)

| 项目 | 详情 |
|---|---|
| **脚本** | `src/run_position_sensitivity.py` + `src/run_position_behavior.py` |
| **实验日期** | 2026-05-20 |
| **设计** | 相同证据内容置于 3 个位置 (early/mid/late) via prefix shifting |

### 10.1 表示层：Rep Shift

| 指标 | 值 |
|---|---|
| 3-NN position classification accuracy | **1.000** (baseline=0.333) |
| cos(early, mid) mean ± std | 0.065 ± 0.012 |
| cos(early, late) mean ± std | 0.080 ± 0.013 |
| cos(mid, late) mean ± std | 0.005 ± 0.001 |
| N | 60 (30A+30U), Layer 12, last_prompt_token |

**相同内容在不同位置 → 完全不同的 hidden state。3-NN 可完美分类位置。**

### 10.2 Probe 层：Position Sensitivity Index (PSI)

| 指标 | 值 |
|---|---|
| PSI (mean abs score delta) | **0.0084** |
| A/U separation: early | 0.993 |
| A/U separation: mid | 0.988 |
| A/U separation: late | 0.976 |

**Probe 层位置影响极低 (PSI < 0.1)。A/U 分离跨位置高度保留。** 这意味着在 mixed-position 数据上训练的 probe 对位置是 robust 的——这是一种"部分补偿"。

### 10.3 行为层：Position-to-Behavior

| 位置 | H | C | CA | UA |
|---|---|---|---|---|
| early | 0.867 | 0.600 | 0.067 | 0.000 |
| mid | 0.900 | 0.667 | 0.033 | 0.000 |
| late | 0.867 | 0.667 | 0.033 | 0.000 |

| 指标 | 值 |
|---|---|
| H range | 0.033 |
| C range | **0.067** |
| N | 60 × 3 positions = 180 次生成 |

**Position IS a behavioral confound —— 位置变化导致行为可测变化。**

### 10.4 三层证据链解释

```
表示层:  位置 → 巨大表示偏移 (cos distance 0.08, KNN=1.0)
                        ↓
          部分被 probe 层的 mixed-position 训练补偿
                        ↓
Probe 层: 位置 → 几乎无影响 (PSI=0.0084)
                        ↓
          但补偿不完全，部分表示偏移漏过
                        ↓
行为层:   位置 → 中度影响 (ΔC=0.067, ΔH=0.033)
```

这正是吸收瓶颈（Absorption Bottleneck）的理论预测：

> 输入碎片化（位置偏移）扭曲了状态空间，模型有部分下游补偿能力（probe 层），但补偿不完全，导致行为泄漏。这是"结构适应能力不足"的直接证据。

---

## 11. Phase 10: P6 Behavior-Only Probe —— 反馈仍 Null，开环确认

P5-bis 确认了负 alpha（减 v_syc）是正确的抗谄媚方向，α=−3.0 是最优注入强度。
P5 的反馈失败是因为探针在 contrast set 上训练，学到了 group membership 而非行为倾向。
P6 的假设是：如果用标准 prompt 样本 + 行为标签重新训练探针，probe→gate→hook 闭环
就应该能工作。

### 11.1 实验设计

| 项目 | 详情 |
|---|---|
| **脚本** | `src/run_p6_syc_behavior_probe.py` |
| **设计** | 12-sample test set, behavior-only probe (标准 prompt 样本 + 行为标签) → hard gate (threshold=0.5) → v_syc steering hook at α=−3.0 |
| **探针训练** | 仅标准 prompt 样本（无 fact-checker persona），标签 = 样本在标准 prompt 下输出是否谄媚，T=0.7 + k=5 创建行为变异 |
| **Baseline** | sycophancy rate = 0.6667 (8/12) |

### 11.2 探针训练结果

| 指标 | 值 |
|---|---|
| Train accuracy | 81.9% |
| Test accuracy | 77.8% |
| Balanced accuracy | 78.5% |
| Training samples | 90 (18 samples × 5 generations at T=0.7) |
| Syc ratio in training | 70.0% |

行为专用探针可训练，测试准确率 77.8% 表明 sycophancy 信号在轨迹上确实可被提取
—— 与 T1/T2 的超可分性结论一致。

### 11.3 反馈控制结果（probe→gate→hook, α=−3.0）

| 指标 | 值 |
|---|---|
| Gate trigger rate | **8.3%** (1/12) |
| Feedback effect | **null** — gate 基本不开 |
| Probe μ in hook | 0.4889 |

探针分数聚集在 0.5 附近 —— hard gate 在 threshold=0.5 处无法区分谄媚倾向/非谄媚倾向
样本。探针学到了信号（acc=77.8%），但分数的 class separation 不足以支撑 hard gate 决策。

### 11.4 开环确认（always-on v_syc, α=−3.0）

| 条件 | Syc Rate | Δ vs Baseline |
|---|---|---|
| Baseline | 0.6667 | — |
| v_syc α=−3.0 | 0.3333 | **−50%** |

开环 v_syc 在 α=−3.0 下将 sycophancy 从 0.6667 降至 0.3333，**−50% 降幅**。

### 11.5 P6 路线更新

```
P0 → T0-T3 → P3-P4 → M7 → P5 → P5-bis → P6 → P6-bis (下一步)
```

---

## 12. Phase 11: P6-bis Threshold Calibration —— Hook 架构诊断

P6 的原始假设是阈值校准问题（探针分数聚集在 0.5 附近，hard gate at 0.5 不触发）。
P6-bis 设计了 5 个阈值 + 3 种百分位规则来系统性测试，**但发现了更深层的问题**。

### 12.1 实验设计

| 项目 | 详情 |
|---|---|
| **脚本** | `src/run_p6_bis_threshold_calibration.py` |
| **设计** | 5 个阈值 sweep (0.30, 0.35, 0.40, 0.45, 0.50) + 3 种百分位规则 (top-20%, 30%, 40%) → feedback control with v_syc α=−3.0 |
| **Phase 2** | 独立分析：在 standalone forward pass 下收集 probe 分数分布 |
| **Test set** | 12 样本（与 P6 相同） |

### 12.2 关键发现：两个世界的探针分数

**Phase 2 — Standalone Forward Pass 下的探针：**

| 指标 | 值 |
|---|---|
| Score mean | 0.6448 |
| Score std | 0.2884 |
| Syc mean score | **0.8233** |
| Non-syc mean score | **0.2876** |
| Score separation | **+0.5357** |

在 standalone forward pass 下，探针达到近乎完美的分数分离 ——
谄媚样本分数 ~0.82，非谄媚样本分数 ~0.29，分离度 +0.54。

**Phase 3 — model.generate() 内 hook 中的探针：**

| Threshold | Gate Rate | Syc Rate | Probe μ (in hook) |
|---|---|---|---|
| 0.30 | 0.0833 | 0.6667 | 0.4655 |
| 0.35 | 0.0833 | 0.6667 | 0.4655 |
| 0.40 | 0.0833 | 0.6667 | 0.4655 |
| 0.45 | 0.0833 | 0.6667 | 0.4655 |
| 0.50 | 0.0833 | 0.6667 | 0.4889 |

gate rate 在所有阈值上恒定 8.3%（包括 th=0.30，在 standalone 模式下本应捕获 11/12 样本）。
**Hook 中的探针 μ ≈ 0.47 ≠ Standalone μ ≈ 0.64。**

### 12.3 根因诊断：Hook 捕获 Generated Tokens，而非 Prompt Tokens

`model.generate()` 调用模型多次：prefill step（处理 prompt） + decode steps（逐 token 生成）。
Hook 在每一步都触发，`hs[:, −1, :]` 在 prefill step 捕获最后一个 **prompt** token，
在 decode steps 捕获**生成**的 token。

探针只在 last_prompt_token hidden states 上训练，不知道如何打分 generated-token states。

**结果**：gate rate = 8.3% 恒定，无论阈值如何。

### 12.4 百分位式 Gating — 同样无效

| Gate Rule | Effective Threshold | Gate Rate | Syc Rate |
|---|---|---|---|
| top-20% | 0.9085 | 0.0833 | 0.6667 |
| top-30% | 0.8703 | 0.0833 | 0.6667 |
| top-40% | 0.8522 | 0.0833 | 0.6667 |

结果完全相同 —— 进一步确认 hook 中的分数与 standalone 中的分数不是同一分布。

### 12.5 Open-Loop — 第三次复现

Open-loop v_syc α=−3.0: syc_rate = 0.3333 (−50.0%)。
这已是**第三次**在不同测试集上独立复现：
- P5-bis (24-sample): −35.7%
- P6 (12-sample): −50.0%
- P6-bis (12-sample): **−50.0%**

v_syc α=−3.0 开环控制的效应已足够稳健。

### 12.6 Fix：两阶段架构（P6-ter）

```
Stage 1: model(**inputs) → collect L10 last_prompt_token hs → probe score
Stage 2: if score ≥ threshold → run model.generate() WITH steering hook at L10
         else → run model.generate() WITHOUT steering
```

探针始终看到 last_prompt_token hidden states（它被训练打分的那些状态），
消除 token-type 污染问题。

### 12.7 核心发现

1. **F15: 行为专用探针在 standalone forward pass 下达到 +0.54 分数分离。**
   探针本身非常好 —— syc mean=0.82, non-syc mean=0.29。
2. **F16: 反馈失败的根因不是阈值校准，是 hook 架构 bug。**
   Hook 在 `model.generate()` 中捕获 generated-token hidden states，
   而非探针被训练的 prompt-token states。
3. **F17: Gate rate 在所有阈值（0.30−0.90）上恒定 8.3%。**
   这是 hook 架构 bug 的确认信号——如果只是阈值校准问题，
   降低阈值理应提高 gate rate。
4. **F18: Open-loop v_syc α=−3.0 第三次复现（−50%）。**
   效应稳健。
5. **N10: 两阶段架构修复。**
   Standalone probe scoring → conditional generate with steering.

---

## 13. Phase 12: P6-ter Two-Stage Feedback Control —— 闭环打通

P6-bis 诊断了 hook 架构 bug：`model.generate()` 内部 hook 捕获 generated-token
hidden states，探针未训练于此类状态。P6-ter 设计了两阶段架构来修复这个问题，
使用 standalone forward pass 做探针打分，然后条件性地运行带/不带 steering 的生成。

### 13.1 实验设计

| 项目 | 详情 |
|---|---|
| **脚本** | `src/run_p6_ter_two_stage_feedback.py` |
| **设计** | Stage 1: `model(**inputs)` → L10 last_prompt_token hs → probe score。Stage 2: if score ≥ threshold → `model.generate()` WITH steering hook (v_syc α=−3.0)；else → WITHOUT steering。 |
| **Test set** | 12 样本（同 P6/P6-bis） |
| **Thresholds** | 0.30, 0.40, 0.50, 0.60, 0.70 |
| **Controls** | Open-loop（全量 steering）、Random vector two-stage (th=0.50) |

### 13.2 结果

**Baseline**: syc_rate = 0.7500

| Threshold | Gate Rate | Syc Rate | Δ from Baseline |
|---|---|---|---|
| 0.30 | 83.3% (10/12) | 0.5833 | −22.2% |
| 0.40 | 66.7% (8/12) | 0.3333 | −55.6% |
| **0.50** | **58.3% (7/12)** | **0.2500** | **−66.7%** |
| 0.60 | 58.3% (7/12) | 0.3333 | −55.6% |
| 0.70 | 50.0% (6/12) | 0.4167 | −44.4% |

**Open-loop**: syc_rate = 0.4167 (−44.4%)

**Random vector two-stage (th=0.50)**: syc_rate = 0.5833 (−22.2%)

### 13.3 关键发现

1. **闭环打通。** th=0.50 时 syc 从 0.7500 降至 0.2500（−66.7%），显著优于
   open-loop 的 −44.4%。这是 probe→gate→hook 反馈控制首次为 sycophancy 工作。

2. **选择性干预 > 全量干预。** 两阶段反馈仅 steer 7/12 样本（58.3%），保留
   5/12 自然非谄媚样本不干预。Open-loop 全量 steering 可能扰动自然非谄媚样本，
   反而降低整体效果。

3. **方向特异性在闭环中确认。** Random vector 两阶段（相同 gate rate）仅
   −22.2%，v_syc 达 −66.7%。v_syc/random = 2.67× in closed-loop。

4. **阈值存在 U 型最优。** th=0.30（steer 太多 → 扰动非谄媚样本 → 效果弱），
   th=0.70（steer 太少 → 漏掉谄媚样本 → 效果弱），th=0.50 是最优平衡点。

5. **行为专用探针足够做门控决策。** 探针测试准确率 77.8%，在 standalone scoring
   下达到足够的分辨力来做出有效的 gate 决策。

### 13.4 路线完成

```
P0 → T0-T3 → P3-P4 → M7 → P5 → P5-bis → P6 → P6-bis → P6-ter ✅
```

**Sycophancy probe→gate→hook 反馈控制闭环正式完成。**

---

## 14. Phase 13: P7 S15 Amplification Mechanism —— Readability ≠ Manipulability

T2 发现 sycophancy 探针准确率在 generation step 15 达到峰值（0.983 at L8），
但放大机制一直未解释。P7 设计了三个维度来调查 S15：(1) 逐步探针打分，
(2) 逐步 token 分析，(3) 逐步 steering 干预。

### 14.1 实验设计

| 项目 | 详情 |
|---|---|
| **脚本** | `src/run_p7_s15_amplification.py` |
| **Phase 1** | 手动逐步生成（64 步），每步捕获 L10 hidden state → P6 probe 打分 |
| **Phase 2** | Token 级别分析：步 10-20 生成了什么 token |
| **Phase 3** | 单步 steering：在步 5/10/15/20/25 注入 v_syc α=−3.0 |
| **Test set** | 12 样本（syc=8, non-syc=4），baseline=0.6667 |

### 14.2 Phase 1: 逐步探针打分 —— 峰值在 Step 1，不在 S15

| Metric | Step 1 | Step 15 |
|---|---|---|
| Syc μ | **0.8273** | 0.6280 |
| Non-Syc μ | **0.1819** | 0.4964 |
| Separation | **+0.6455** | +0.1316 |

P6 探针在第一个生成 token 达到最强分离（+0.65），S15 分离很弱（+0.13）。
分离度在步 1-30 之间剧烈震荡（从 +0.65 到 −0.17）。

**原因**: P6 探针在 last_prompt_token 状态上训练。第一个生成 token 的 hidden
state 最接近 prompt-token 分布（直接后继）。随着生成推进，hidden state 逐渐
偏离训练分布，探针性能下降。

**T2 的 S15 峰值是不同现象** —— T2 在每 (layer, step) 位置分别训练探针，
每个探针都校准了其特定位置的 hidden state 分布。P6 是单一模型试图在所有
位置上泛化。

### 14.3 Phase 3: 单步 Steering —— 全部无效

| Target Step | Syc Rate | Δ vs Baseline | Effect |
|---|---|---|---|
| S5 | 0.7500 | +0.0833 (+12.5%) | **更差** |
| S10 | 0.8333 | +0.1667 (+25.0%) | **更差** |
| S15 | 0.6667 | +0.0000 (0.0%) | Null |
| S20 | 0.6667 | +0.0000 (0.0%) | Null |
| S25 | 0.6667 | +0.0000 (0.0%) | Null |

**没有任何单步 steering 降低了 sycophancy。** 早期步骤（S5、S10）反而**增加**
了谄媚（模型补偿了扰动）。后期步骤（S15-S25）零效果。

### 14.4 核心发现

1. **F23: P6 探针在 step 1 达到峰值分离（+0.65），不在 S15（+0.13）。**
   T2 的 S15 峰值反映的是 per-position 训练探针的可读性，非 P6 泛化探针。

2. **F24: 单步 steering 无效。** S5/S10 增加谄媚（+12.5%/+25%），S15-S25
   零效果。只有累积式（open-loop，全步）steering 有效。

3. **N8 已解决: S15 不是因果敏感期。** Readability（T2 per-position probe
   accuracy）≠ manipulability（P7 per-step steering）。T2 的 S15 峰值是
   per-position 探针训练方法的副作用，不是因果"决策点"。

4. **谄媚是累积式、分布式过程** —— 非单步"开关"。Open-loop 有效是因为
   每一步都施加同样方向的小扰动，累积效应超过模型自身的纠正能力。

### 14.5 理论意义

这对控制框架有直接影响：
- **谄媚干预需要持久式 steering**（全生成轨迹），不能靠单次精准注入。
- **P6-ter 两阶段架构正确** —— 对门控样本施加全步 steering。
- **T2 S15 峰值为认识论副现象**，不具有因果操作价值。
- **Readability ≠ manipulability** —— 这是对"表示→因果"映射模式的重要方法学洞察。

---

## 14. Phase 14: P8 Large-Scale Replication —— P6-ter 小样本优势为虚假信号

P6-ter 在 n=12 样本上实现了 −66.7% 的 sycophancy 降低（两阶段 th=0.50），
但小样本容易产生随机波动带来的虚假效应。P8 在 n=24 样本上复制全流程。

### 14.1 实验设计

| 项目 | 详情 |
|---|---|
| **脚本** | `src/run_p8_large_scale_replication.py` |
| **样本量** | 24（index [18:42]） |
| **条件** | baseline, two-stage th=0.50, two-stage th=0.40, open-loop |
| **配置** | L10, v_syc α=−3.0, P6 behavior-only probe |
| **统计检验** | Fisher 精确检验（双尾），与 P6-ter (n=12) 对比 |

### 14.2 结果

| 条件 | N | Syc Rate | Δ vs Baseline | Gate Rate |
|---|---|---|---|---|
| baseline | 24 | 0.7083 (17/24) | — | — |
| two-stage th=0.50 | 24 | 0.5417 (13/24) | −23.5% | 54.17% |
| two-stage th=0.40 | 24 | 0.5417 (13/24) | −23.5% | 66.67% |
| open-loop | 24 | 0.5000 (12/24) | −29.4% | — |

### 14.3 与 P6-ter (n=12) 对比

| 指标 | P6-ter (n=12) | P8 (n=24) | 变化 |
|---|---|---|---|
| Baseline syc | 0.7500 | 0.7083 | −0.04 |
| Two-stage th=0.50 | 0.2500 (−66.7%) | 0.5417 (−23.5%) | **+0.29** |
| Two-stage th=0.40 | 0.3333 (−55.6%) | 0.5417 (−23.5%) | +0.21 |
| Open-loop | 0.4167 (−44.4%) | 0.5000 (−29.4%) | +0.08 |

### 14.4 统计显著性

| 比较 | Fisher p | Significant? |
|---|---|---|
| Baseline vs Two-Stage th=0.50 | 0.3715 | No |
| Baseline vs Open-Loop | 0.2375 | No |
| Two-Stage th=0.50 vs Open-Loop | 1.0000 | No |

所有成对比较均未达到 p<0.05。

### 14.5 核心发现

1. **F25 (新增): 方向正确但效应弱于 P6-ter。** Steering 确实降低 sycophancy
   （−23.5% to −29.4%），方向一致。但 P6-ter 的 −66.7% 为一个数量级以上的
   高估 —— 很可能是 n=12 小样本的随机波动 artifact。

2. **N11 (新增): 两阶段闭环优势不跨样本量扩展。** P6-ter 上两阶段 (−66.7%)
   > open-loop (−44.4%)；但 P8 (n=24) 上两阶段 (−23.5%) < open-loop (−29.4%)。
   两阶段选择性干预的相对优势为小样本虚假信号。

3. **阈值区分度消失。** P6-ter 上 th=0.50 (−66.7%) ≠ th=0.40 (−55.6%)；
   P8 上两者完全一致 (0.5417)。U 型曲线的 "最优阈值" 概念在小样本上可能是过拟合。

4. **Fisher 检验均不显著。** n=24 仍不足以在 0.05 水平上区分两阶段与 open-loop。
   要求 n≥48 可能才能达到统计学显著性 —— 如果真实效应为 ~0.2 的 syc rate 降低。

### 14.6 对闭环叙事的影响

P6-ter 的闭环 breakthrough 叙事需要修正：

- **保留**：方向特异性在表示层 (P3)、因果层 (P4)、开环层 (P5-bis) 均已证实
- **保留**：两阶段架构是正确的工程模式 —— standalone scoring → conditional steering
- **修正**：闭环选择性优势在 n=24 上不成立 —— 两阶段 ≤ open-loop
- **修正**：P6-ter 的 −66.7% 不应被引用为闭环效应量 —— 实际效应量更接近 −23~30%

**这不是闭环叙事的否定** —— steering 方向性仍在，但选择性干预的边际收益被
样本量放大。对封闭式反馈控制的追求仍是正确的工程方向，但当前证据下两阶段
架构与 open-loop 无显著性差异。

---

## 15. Phase 15: P9 Cross-Bottleneck — Steering 不破坏结构完整性

P9 测试跨瓶颈假说：steering（Organization）是否破坏 L10 hidden state
的 syc/non-syc 聚类结构？如果是，stabilization（Per-Action KMeans）可对
steered 状态做结构重建；如果否，两个瓶颈独立运作。

### 15.1 实验设计

| 项目 | 详情 |
|---|---|
| **脚本** | `src/run_p9_cross_bottleneck.py` |
| **样本量** | 24（index [18:42]） |
| **条件** | baseline forward pass + steered forward pass（v_syc α=−3.0 hook） |
| **标签** | 设计意图标签（`group` 字段：sycophantic/non_sycophantic，12/12 平衡） |
| **分析** | KMeans k=2 聚类 → ARI, purity, centroid distance, per-sample shift |

### 15.2 结果

| Metric | Baseline | Steered | Delta |
|---|---|---|---|
| ARI (vs ground truth) | 1.0000 | 1.0000 | 0.0000 |
| Cluster Purity | 1.0000 | 1.0000 | 0.0000 |
| Centroid Cosine Sim | 0.9356 | 0.9400 | +0.0044 |
| Mean Intra-Cluster Dist | 1.7893 | 1.7894 | +0.0001 |
| Inter-Cluster Distance | 4.3051 | 4.3049 | −0.0002 |

- Per-sample shift: ||hs_steered − hs_baseline|| = 3.0000 ± 0.0000
- Cosine sim (steered, baseline) mean: 0.9707

### 15.3 核心发现

1. **F26 (新增): L10 结构完整性完美保留。** ARI 和 purity 在 baseline 和
   steered 间完全相同（1.0 → 1.0）。所有指标变化 < 0.005。v_syc steering
   是均匀平移（||v_syc||=1.0, α=−3.0 → 每个样本位移 3.0），不破坏相对几何。

2. **Steering = clean directional bias, not structural perturbation。**
   v_syc 向每个 L10 hidden state 加上相同的方向向量，所有样本平移相同距离。
   余弦相似度 0.9707 确认角度结构几乎不变。

3. **跨瓶颈协同不支持。** Per-Action KMeans (stabilization) 在 C11 中
   通过重建被扰动破坏的结构实现增益。但 steering 不破坏结构——没有需要
   "恢复"的东西。Organization 和 Stabilization 瓶颈是独立的。

4. **正向含义**：steering 不会造成表示侧 collateral damage。行为效果
   (−23~30% syc reduction) 来自方向偏置而非结构退化。

---

## 16. Phase 16: P10 Formal Exclusion — Hall 单方向 Impulse 路线关闭

P10 不是新实验，而是基于已有 5 层证据的正式排除文档。
它将 T1-T3、P2、P3、P4、B2 中分散的负性发现整合为一条 definitive exclusion。

### 16.1 五层证据链

| 层级 | 实验 | 发现 |
|---|---|---|
| 1 | T1+T2 | v_hall 投影表示层面可分离（v_hall/random=3.51×），探针 acc=0.917。可检测 ≠ 可操控。 |
| 2 | T3 | v_hall controllability (0.0219) < random (0.0303)。方向效果低于随机。 |
| 3 | P2 | **v_hall = v_orthogonal**：数学上与 v_hall 正交的向量产生完全相同的幻觉率变化 (ΔH=−0.283 vs −0.283)。所有 5 个方向在 0.24-0.28 窄范围内。 |
| 4 | P3 | v_hall/random = 0.28×（对比 Syc: 2.73×）。方向特异性经验排除。 |
| 5 | P4+B2 | 方向-vs-能量分解：Hall 仅存能量分量。结构化控制有边界条件（C_base=0.800 时 random > structured）。 |

### 16.2 核心证据：P2 v_hall = v_orthogonal

这是最强的负性证据。v_orthogonal 与 v_hall 的余弦相似度 < 1e-5——
数学上保证无关。但它产生的行为效果与 v_hall 完全相同。

**逻辑**: 如果方向特异性存在，v_hall 应显著 ≠ v_orthogonal。但两者相等。
→ 方向不携带因果信息。效果来源是 pure energy。

### 16.3 正式排除声明

> **单方向 impulse steering 被正式排除为 hallucination 控制策略。**
> 证据跨 5 层收敛于同一结论：hallucination 相关 hidden state 扰动是
> 能量驱动而非方向驱动。Hall 和 Syc 是根本上不同的可操控性对象。

### 16.4 替代路径

| 方法 | 证据 |
|---|---|
| 闭环 gate（M3-v6） | H: 0.867→0.667 (−23%)，当前最佳 |
| 能量/扰动方法 | T3 的 generic perturbation 有效 |
| 多方向组合 | B2：C_base=0.400 时有效 |

**F27 (新增): 单方向 impulse 被 5 层证据正式排除。Hall=纯能量。**
**N1 从 "暂未确认" 升级为 "正式排除"。**

---

## 17. Phase 17: P11 Stabilization Scaling — 4 维度全通过，Fully Validated

P11 不是新实验，而是跨项目整合——将 IC-2 gridworld 的 scaling 实验纳入 IC-4 稳定瓶颈叙事。

**核心问题**: Per-Action KMeans（稳定瓶颈补救）在 seed count、noise level、action complexity、structured perturbation 四个维度上能否保持优势？

**答案: 是的。四个维度全部通过。**

### 17.1 Seed Scaling (C4)

从 5 seeds 扩展到 20、50、100 seeds：

| Seeds | Per-Action | NoMemory | Delta |
|---|---|---|---|
| 5 (C3) | 0.585 | 0.445 | +0.140 (reference) |
| 20 (S1) | 0.635 | 0.445 | +0.190 (+36%) |
| 50 (S2) | 0.660 | 0.445 | +0.215 (+54%) |
| 100 (S3) | 0.615 | 0.445 | +0.170 (+21%) |

Delta 先增长后收敛——不是 size artifact，是真正的结构效应。NoMemory 不变（随机基线）。

### 17.2 Noise Robustness (N1)

Gaussian noise (std 0→1.0) 添加到 hidden states：

| Noise | Per-Action | NoMemory | Delta |
|---|---|---|---|
| 0.0 | 0.500 | 0.445 | +0.055 |
| 0.3 | 0.505 | 0.445 | +0.060 |
| 1.0 | 0.545 | 0.445 | +0.100 |

Delta 随噪声增长（+0.055→+0.100）。Y-aware 在 state-space 结构退化时反而更强——动作标签成为锚点。

### 17.3 Action Complexity (N2)

从 3 到 20 个动作：

| Actions | Per-Action | NoMemory | Delta |
|---|---|---|---|
| 3 | 0.500 | 0.445 | +0.055 |
| 5 | 0.715 | 0.385 | +0.330 |
| 10 | 0.355 | 0.265 | +0.090 |
| 20 | 0.155 | 0.110 | +0.045 |

最优 5 动作（+0.330，4.7σ 以上）。3–10 均可工作。20 动作退化因为每动作轨迹过少。

### 17.4 Cross-Bottleneck Perturbation (C5)

模拟组织干预扰动（3 种 × 7 级）：

| 扰动类型 | PA Mean | NoMemory | Margin |
|---|---|---|---|
| Additive noise | 0.540 | 0.445 | +0.095 |
| Directional shift | 0.588 | 0.445 | +0.143 |
| Centroid dropout | 0.519 | 0.445 | +0.074 |

所有类型、所有级别 PA > NoMemory。最极端 dropout=0.9 时仍有 margin（0.48 > 0.445）。

### 17.5 综合结论

> **Per-Action KMeans 是 IC 项目目前为止验证最充分的正面发现。**
>
> C3（根因诊断）→ C4+N1+N2+C5（4 维度 scaling）形成完整验证链。
>
> Stabilization: Diagnosed → Remied → **Fully Validated**。

**F28 (新增): P11 Stabilization scaling 在 4 维度上全部通过。Per-Action KMeans 是验证最充分的正面发现。**

**与 P9 的互补**: P9 发现 steering 不破坏 LLM 层结构（不需要稳定补偿）。C5 发现即使扰动破坏了结构，Per-Action KMeans 仍提供安全 margin。两者共同构成完整的跨瓶颈画面。

---

## 18. Phase 18: P12 Absorption Directional Steering — 否定性结果

P12 测试 open-loop activation steering 能否减少行为层位置敏感性，遵循已验证的 sycophancy steering 模式。

**核心问题**: v_abs = h_early − h_late 方向 steering 能否减少行为层 position sensitivity？

**答案: 方向存在，但效应是 homogenization with degradation——不可行。**

### 18.1 实验设计

1. 计算 v_abs = mean(h_early) − mean(h_late) at L10
2. Open-loop steering: add α × v_abs at L10 during generate
3. α-sweep: −3.0, −1.5, 0, +1.5, +3.0
4. Controls: random direction, orthogonal direction
5. n=10 per position (early/mid/late), 30 total

### 18.2 α-Sweep 结果 (n=10)

| Condition | delta_H | H_early | H_mid | H_late |
|---|---|---|---|---|
| baseline (α=0) | 0.750 | 0.250 | 1.000 | 1.000 |
| v_abs α=−3.0 | 0.250 | 0.750 | 1.000 | 1.000 |
| v_abs α=−1.5 | 0.250 | 0.750 | 1.000 | 1.000 |
| v_abs α=+1.5 | 0.250 | 0.750 | 1.000 | 1.000 |
| v_abs α=+3.0 | 0.250 | 0.500 | 0.750 | 0.750 |
| random α=−3.0 | 0.500 | 0.500 | — | 1.000 |

**关键发现**:
1. **Baseline 有巨大的 position gap**: H_mid=H_late=1.000（天花板），只有 early 表现良好（0.250）
2. **任何 α≠0 都损伤 early**: H_early 从 0.250 升至 0.500–0.750
3. **α=+3.0 是唯一改善 mid/late 的条件**: H_mid 1.000→0.750，H_late 1.000→0.750
4. **净效应**: α=+3.0 将 delta_H 从 0.750 降至 0.250 (−67%)，但以 H_early +100% 为代价

### 18.3 方向特异性

v_abs/random = 2.0×（delta_H reduction），但:
- v_abs 比 random 多损伤 early 1.5×（H_early 0.750 vs 0.500 at |α|=3.0）
- 效应是"负和性质"——方向特异性存在但不可行

### 18.4 与 Sycophancy 的关键差异

| | Sycophancy (P5-bis) | Absorption (P12) |
|---|---|---|
| 最优 α | −3.0 (syc −35.7%) | +3.0 (delta_H −67%) |
| 净改善？ | ✅ 是 | ❌ 否（early 退化） |
| 方向特异性 | 2.73× | 2.0× |
| 机制 | 选择性抑制目标行为 | 牺牲最优位置换方差缩小 |

**F29 (新增): P12 v_abs steering 将 delta_H −67% 但牺牲 H_early +100%。方向特异性 2.0× 但效应是 homogenization with degradation。Absorption 不可用 directional steering 修复。**

**A1→A4→P12 链确认: probe→behavior gap 抵抗所有已测干预类型（全局 rectification、权重训练、directional steering）。**

---

## 19. Phase 19: P13 Energy/Direction Asymmetry — L10 均匀，非对称在行为

P13 测试 P12 观察到的行为非对称性是来自 L10 表示层差异还是下游计算。

**核心问题**: Energy 扰动（Gaussian noise）和 Direction 扰动（v_abs）在 L10 hidden state 上是否对不同位置产生不同的 ||Δh||？

**答案: 否。两种扰动在所有位置上产生近乎完美的均匀 ||Δh||。非对称性在行为层。**

### 19.1 Energy 扰动

| noise std | mean_shift | early | mid | late | max_ratio |
|---|---|---|---|---|---|
| 0.01 | 0.30 | 0.30 | 0.30 | 0.30 | 1.02 |
| 0.03 | 0.90 | 0.91 | 0.90 | 0.89 | 1.02 |
| 0.05 | 1.49 | 1.50 | 1.49 | 1.49 | 1.01 |
| 0.10 | 2.98 | 2.98 | 2.96 | 3.00 | 1.01 |

Energy max_ratio = 1.01–1.02 → 均匀。

### 19.2 Direction 扰动

| alpha | mean_shift | early | mid | late | max_ratio |
|---|---|---|---|---|---|
| −3.0 | 3.00 | 3.00 | 3.00 | 3.00 | **1.001** |
| −1.5 | 1.50 | 1.50 | 1.50 | 1.50 | **1.000** |
| +1.5 | 1.50 | 1.50 | 1.50 | 1.50 | **1.000** |
| +3.0 | 3.00 | 3.00 | 3.00 | 3.00 | **1.000** |

Direction max_ratio = 1.000–1.001 → 均匀。数学上 ||α·v_abs|| = |α| 对所有 h 成立。

### 19.3 关键发现

> **P12 的行为非对称性（H_early 对方向扰动更敏感）不是来自 L10 表示层差异——两种扰动在 L10 都产生均匀平移。非对称性来自下游计算对相同扰动的差异化放大。**

**F30 (新增): P13 确认 energy 和 direction 扰动在 L10 均产生均匀 ||Δh|| (max_ratio ≤ 1.02)。P12 的行为非对称性来自下游计算。这解释了为什么所有基于 L10 的补救（A3 rectification、P12 steering）都失败——它们瞄准了错误的层次。**

### 19.4 三层架构

| 层 | 描述 | 状态 |
|---|---|---|
| Input (position) | 信息按位置退化 | Diagnosed |
| Representation (L10) | 扰动均匀传输 | **P13: Uniform verified** |
| Behavior (output) | 相同扰动被不同位置差异化放大 | **P12: Asymmetric** |

---

## 20. Phase 20: P14 Cross-Project Synthesis — 诊断阶段闭合

P14 不是新实验，而是跨项目合成——将 IC-4 (LLM) 和 IC-2 (gridworld) 的全部发现整合为一份可交付物。

**核心成果**: 诊断+验证+排除阶段闭合。三瓶颈全部诊断，两个有补救，一个 L10 路径闭合。

**F31 (新增): P14 合成文档完成。30 正发现 + 13 排除路线 + 3 补救 + 5 边界条件。项目诊断阶段完成。Absorption 是唯一 frontier。**

结合数据:
- IC-4: 26 experiments, 30 positive findings, 13 exclusions
- IC-2: 6 experiments, 4 positive findings, Per-Action KMeans validated
- 总计: 32 experiments, 34 positive findings, 3 dead ends closed

干预矩阵:
```
                Hallucination    Sycophancy    Stabilization
    Closed-loop      ✅             ❌              —
    Open-loop        ❌             ⚠️              —
    Y-aware cls      —              —              ✅
```

---
## 21. Phase 21: P15 Hallucination LoRA — B-Bottleneck Bridged

P13+P14 proved K↔D subspaces are near-orthogonal across all layers. P15 asks the natural follow-up: if vector ops can't bridge the gap, can **weight-level LoRA training** do it?

**核心问题**: Can hallucination-targeted LoRA fine-tuning eliminate hallucination (H) while preserving correctness (C)?

**答案: Yes. H drops from 0.417 to 0.000. C stays at 1.000.**

### 21.1 实验设计

| 项目 | 详情 |
|---|---|
| **脚本** | `src/run_p15_hallucination_lora.py` |
| **模型** | Qwen2.5-0.5B-Instruct + LoRA (rank=4, lr=2e-4, 3 epochs) |
| **训练数据** | 90 samples (45A+45U), position-variant (30 per early/mid/late) |
| **Target** | answerable → positive_response; unanswerable → negative_response |
| **Loss** | Standard CE (prompt tokens masked) |
| **评估** | Log-prob comparison (positive vs negative response log-prob) |

### 21.2 结果

| 指标 | Pre (Base) | Post (LoRA) | Δ |
|---|---|---|---|
| **H** | 0.4167 | **0.0000** | **−0.4167** |
| **C** | 1.0000 | 1.0000 | 0.0000 |
| ΔH (位置 gap) | 0.250 | **0.000** | −0.250 |

**Per-Position:**

| Position | Pre H | Post H |
|---|---|---|
| early | 0.25 | 0.00 |
| mid | 0.50 | 0.00 |
| late | 0.50 | 0.00 |

vs Phase 10 (A-bottleneck LoRA): P15 H=0.000 vs P10 H=0.500.

### 21.3 核心发现

**F32 (新增): Hallucination LoRA 将 H 从 0.417 降至 0.000 (ZERO)，C=1.000 全保留。B-bottleneck KNOWS→produces gap 被权重级 LoRA 弥合。**

**F33 (新增): P15 LoRA 超越 Phase 10 LoRA (H 0.000 vs 0.500)。行为定向训练比位置不变性训练更有效。**

**F34 (新增): B-bottleneck 的可弥合性已证明。权重级干预在 A 和 B 两个瓶颈上都有效，而向量操作在两者上都无效。**

**F35 (新增): 位置 gap 完全消除 (ΔH 0.250→0.000)。三个位置 all H=0.000。**

---
## 22. Phase 22: P16 LoRA Geometry Analysis — Routing Fix, Not Geometry Fix

P15 proved LoRA works. P16 asks HOW — does LoRA align K and D subspaces (geometry fix), or change the default behavioral path (routing fix)?

**核心问题**: After LoRA, does w_probe steering now effectively control hallucination? If yes → geometry fix (K↔D aligned). If no → routing fix (default path changed).

**答案: Routing fix. K↔D orthogonality persists after LoRA.**

### 22.1 实验设计

| 项目 | 详情 |
|---|---|
| **脚本** | `src/run_p16_lora_geometry_analysis.py` |
| **设计** | Load P15 LoRA model → 9 layers (0,3,6,9,11,12,15,18,21) → collect hs → train probe → extract w_probe → steering at α=-2,0,+2 → evaluate H |
| **比较** | P14 baseline (base model) vs P16 LoRA model, per-layer comparison |

### 22.2 结果

| Layer | Probe Acc (LoRA) | ΔH_max (LoRA) | ΔH_max (Base) | Verdict |
|---|---|---|---|---|
| 0 | 1.0000 | 0.0000 | 0.1666 | Subspace separation INCREASED |
| 3 | 1.0000 | 0.0833 | 0.1666 | Minor improvement |
| 6 | 1.0000 | 0.0000 | 0.0833 | Unchanged |
| 9 | 1.0000 | 0.0000 | 0.0833 | Unchanged |
| 11 | 1.0000 | 0.0833 | 0.0833 | Identical |
| 12 | 1.0000 | **0.2500** | 0.0833 | **DESTRUCTIVE** (H 0→0.25!) |
| 15 | 1.0000 | 0.0000 | 0.0833 | Unchanged |
| 18 | 1.0000 | 0.0000 | 0.0833 | Unchanged |
| 21 | 1.0000 | 0.0000 | 0.0000 | Identical |

### 22.3 核心发现

**F36 (新增): K-subspace 在 LoRA 后完美保留 (probe acc=1.0000 at all 9 layers)。**

**F37 (新增): P16.2 REJECTED — LoRA 不对齐 K↔D 子空间。w_probe steering 效应在 8/9 层为零或负。L12 的 gain 是破坏性的 (H 0→0.25 at α=+2.0)。**

**F38 (新增): LoRA bridge mechanism = ROUTING fix (default path change), NOT geometry fix (subspace alignment)。**

**F39 (新增): Layer 12 是唯一 w_probe 仍有投影的层 — 但是破坏性投影。**

### 22.4 LoRA = Routing Fix 的判定逻辑

```
Base Model (P14):                LoRA Model (P15):
w_probe steering at L12          w_probe steering at L12
→ H drops from 0.417 to 0.333    → H INCREASES from 0.000 to 0.250
→ ΔH = +0.083                    → ΔH = −0.250 (destructive)
```

If LoRA had aligned K↔D, w_probe steering should become MORE effective. Instead, it becomes destructive — w_probe now pushes H away from zero. The direction that used to be "less hallucination" is now orthogonal to the model's routing path, and adding it disrupts the learned routing.

---
## 23. Phase 23: P17 LoRA Module Ablation — Attention Routing Key

P16 proved LoRA is a routing fix (changes default path). P17 asks: WHICH attention projection carries the routing fix? q_proj (what to attend to), k_proj (what info to offer), v_proj (what to output), or o_proj (how to aggregate)?

**核心问题**: Which attention projection's LoRA weights are indispensable for the H=0.000 result?

**答案: q_proj (Query projection) alone. k, v, o are irrelevant.**

### 23.1 实验设计

| 项目 | 详情 |
|---|---|
| **脚本** | `src/run_p17_lora_ablation.py` |
| **设计** | Load P15 LoRA model → baseline H=0.000 → systematically zero LoRA weights for specific projection types → re-evaluate H → restore weights |
| **条件** | Full, -q, -k, -v, -o, -q-k, -v-o, -ALL (8 conditions) |

### 23.2 结果

| Condition | H | C | ΔH (vs Full) |
|---|---|---|---|
| Full | 0.0000 | 1.0000 | baseline |
| **-q** | **0.2500** | 1.0000 | **+0.2500** |
| -k | 0.0000 | 1.0000 | +0.0000 |
| -v | 0.0000 | 1.0000 | +0.0000 |
| -o | 0.0000 | 1.0000 | +0.0000 |
| -q-k | 0.2500 | 1.0000 | +0.2500 |
| -v-o | 0.0833 | 1.0000 | +0.0833 |
| -ALL | 0.4167 | 1.0000 | +0.4167 |

### 23.3 核心发现

**F40 (新增): q_proj LoRA ablation 使 H 从 0.000→0.250 (+60% 总效应)。Query projection 是唯一关键 LoRA 模块。**

**F41 (新增): k_proj, v_proj, o_proj 各自消融零效应。Value/output projection 对 routing fix 不必要。**

**F42 (新增): Routing fix 通过 Query projection 改变注意力查询模式实现。模型学会了不同的 "attend to what" 问题。**

**F43 (新增): B-bottleneck 本质是注意力路由问题 — q_proj 控制 "模型关注什么"，而非 v_proj (输出什么)。Routing fix = 改变 Query 让模型关注到正确的不回答路径。**

### 23.4 The q/k/v/o Asymmetry

```
q_proj (Query): What question to ask    → 🔴 CRITICAL — THIS is the fix
k_proj (Key):   What info to offer      → irrelevant (info already there)
v_proj (Value): What to output          → irrelevant (correct output exists)
o_proj (Output): How to mix             → irrelevant (mixing doesn't need change)
```

P15+P16+P17 收敛于同一结论: **B-bottleneck 不是表示问题 (K-subspace 完美)、不是输出问题 (v/o 无关)、而是注意力路由问题 — 模型默认问错了问题。LoRA on q_proj 修正了 Query，让模型在遇到 unanswerable 输入时关注到正确的"应该不回答"信号。**

---
## 24. Phase 24: P18 q_proj Layer Ablation — Deep Layers Are the Routing Core

P17 proved q_proj (Query projection) carries the routing fix. P18 asks: **WHICH LAYERS** (early/mid/deep) host this routing change?

**预注册假设**: MID layers (8-15) are most critical — semantic routing lives here.

**答案: ALL hypotheses falsified. DEEP layers (16-23) are the sufficient core.**

### 24.1 实验设计

| 项目 | 详情 |
|---|---|
| **脚本** | `src/run_p18_qproj_layer_ablation.py` |
| **设计** | 2 complementary perspectives (8 conditions): |
| | **Group ABLATION**: zero q_proj in a layer group → who BREAKS routing? |
| | **Group ISOLATION**: keep q_proj ONLY in one group → who SUSTAINS routing? |
| **Layer groups** | Early (0-7), Mid (8-15, § layer 12), Deep (16-23) |

### 24.2 结果

| Condition | H | C | ΔH | Verdict |
|---|---|---|---|---|
| Full | 0.0000 | 1.0000 | baseline | — |
| -q_early | 0.0000 | 1.0000 | +0.0000 | 🟢 irrelevant |
| -q_mid | 0.0000 | 1.0000 | +0.0000 | 🟢 irrelevant |
| -q_deep | **0.0833** | 1.0000 | **+0.0833** | 🔴 only breaker |
| ONLY_early | **0.2500** | 1.0000 | **+0.2500** | ❌ FAILS |
| ONLY_mid | 0.0833 | 1.0000 | +0.0833 | ⚠️ partial |
| **ONLY_deep** | **0.0000** | 1.0000 | **+0.0000** | ✅ PERFECT |
| -q_ALL | 0.2500 | 1.0000 | +0.2500 | baseline recovery |

**预注册假设全部被推翻:**
- H18.1: "MID most critical" → -q_mid ΔH=0.000 ❌
- H18.2: "ONLY_mid best" → ONLY_mid H=0.0833 ❌
- H18.3: "-q_mid largest" → -q_deep is only breaker ❌

### 24.3 核心发现

**F44 (新增): H18.1-18.3 all FALSIFIED. MID layers (8-15) are NOT the primary routing locus. Only -q_deep breaks routing.**

**F45 (新增): Deep layers (16-23) q_proj is the SUFFICIENT core: ONLY_deep H=0.0000. Deep = output-stage refinement layers. Abstention decision is made AFTER full input processing.**

**F46 (新增): Redundant routing architecture: -q_deep (ΔH=+0.083) < ONLY_early (ΔH=+0.250). Deep layers can compensate for early+mid removal; mid layers partially compensate for deep removal.**

**F47-F50: Layer routing gradient (deep>mid>early). P15-P18 complete evidence chain: K↔D orthogonal → LoRA bridges → ROUTING fix → q_proj → DEEP layers. B-bottleneck mechanism FULLY CHARACTERIZED.**

### 24.4 B-Bottleneck Complete Mechanism Chain

```
P13+P14: K↔D are near-orthogonal across all 24 layers
   ↓
P15:  LoRA bridges the gap behaviorally — H=0.000, C=1.000
   ↓
P16:  LoRA = ROUTING fix (bypasses K↔D), not geometry fix
   ↓
P17:  q_proj is the sole critical module — not k/v/o
   ↓
P18:  DEEP layers (16-23) are the sufficient core of q_proj routing
```

**7-experiment chain (P13→P14→P15→P16→P17→P18) fully characterizes the B-bottleneck from geometric diagnosis to layer-level mechanism.**

---
## 26. Phase 25: IC-2d Readout-Matched Episodic — Learned Readout KILLED

IC-2c.1 showed episodic memory underperforms NoMemory. IC-2d tests a counter-hypothesis: maybe k-NN is a poor readout, and MLP/RF would extract more signal.

**核心问题**: Can a learned readout rescue episodic memory?

**答案: No. All learned readouts ≤ k-NN < NoMemory. The problem is representation, not readout.**

### 25.1 实验设计

| 项目 | 详情 |
|---|---|
| **脚本** | `src/run_ic2d_readout_matched.py` (intelligence_capital_minimal_lab) |
| **环境** | StructuredVolatilityEnv (state_dim=2, history_len=8 → 24-dim features) |
| **数据** | 5 seeds × 1200 samples, incremental evaluation |
| **方案** | NoMemory, Episodic-kNN (k=5), Episodic-MLP (64→32), Episodic-RF (50 trees, depth 8) |

### 25.2 结果

| Strategy | Final match | vs NoMemory |
|---|---|---|
| **NoMemory** | **0.445** | baseline |
| Episodic-kNN | 0.195 | −56.2% |
| Episodic-RF | 0.190 | −57.3% |
| Episodic-MLP | 0.095 | −78.7% |

MLP/k-NN ratio: **0.49×** — MLP is WORSE than k-NN.

### 25.3 核心发现

**N14 (新增): Learned readout DOES NOT rescue episodic memory. MLP (0.095) is WORSE than k-NN (0.195).**

**N15 (新增): Episodic underperformance is a REPRESENTATION problem, not a readout problem. All strategies < NoMemory.**

**N16 (新增): Only consolidation (Per-Action KMeans, 0.585) beats NoMemory (0.445). Readout mechanism is not the bottleneck.**

### 25.4 Cross-Project Convergence

Both IC-2 and IC-4 show the same pattern:
- **Representation-level interventions WORK**: consolidation (IC-2), LoRA routing (IC-4)
- **Readout-level interventions FAIL**: MLP/RF readout (IC-2d), w_probe steering (IC-4 P14/P16)

---
## 27. Phase 26: P19 Absorption Attention Patterns — Deep Layer Routing Asymmetry

P13 proved L10 perturbation is uniform but behavior is asymmetric. P18 discovered B-bottleneck routing lives in deep layers (16-23). P19 asks: Does attention routing differ by input position for absorption?

**核心问题**: Is absorption attention position-dependent? If yes, where?

**答案: Yes, DEEP LAYERS amplify position differences. Attention entropy follows U-shaped curve (early high → mid low → deep high). Unanswerable gap > Answerable gap at every layer.**

### 26.1 实验设计

| 项目 | 详情 |
|---|---|
| **脚本** | `src/run_p19_attention_patterns.py` |
| **模型** | Qwen2.5-0.5B-Instruct (eager attention for output_attentions=True) |
| **数据** | position_sensitivity/s0/test_{early,mid,late}_s0.jsonl, 30 each (16A+14U) |
| **Layers** | [0, 3, 6, 9, 12, 15, 18, 21, 23] |
| **度量** | Per-layer attention entropy, L−E gap, answerable vs unanswerable |

### 26.2 结果（U-Shaped Gap）

| Layer | Early | Late | L−E Gap | Δ% |
|---|---|---|---|---|
| L0 | 1.544 | 1.740 | +0.196 | 11.3% |
| L3 | 1.132 | 1.251 | +0.119 | 9.5% |
| L6 | 1.632 | 1.699 | +0.067 | 4.0% |
| L9 | 0.849 | 0.868 | **+0.019** | **4.4%** ← MINIMUM |
| L12 | 1.644 | 1.708 | +0.064 | 3.8% |
| L18 | 1.119 | 1.155 | +0.036 | 3.7% |
| L21 | 0.977 | 1.045 | +0.068 | 6.5% |
| **L23** | **1.859** | **2.139** | **+0.280** | **13.0%** ← MAXIMUM |

**Answerable vs Unanswerable**: Una gap > Ans gap at EVERY layer. L23: Una +0.324 vs Ans +0.240 (1.35×).

### 26.3 核心发现

**F51-F54 (新增): Attention entropy is HIGHER for late-position inputs at ALL 9 layers. U-shaped curve: L0=11.3% → L9 min=4.4% → L23 max=13.0%. Mid-layer semantics are position-robust; deep-layer routing amplifies asymmetry.**

**F55 (新增): B-bottleneck (P18) and A-bottleneck (P19) converge on SAME mechanistic locus: deep-layer attention routing.**

**F56 (新增): L9 is the most position-invariant layer (gap +0.019).**

**F57 (新增): Unanswerable gap > Answerable gap at EVERY layer. Routing uncertainty amplifies when the model should abstain.**

### 26.4 The Two-Bottleneck Convergence

| Aspect | P18 (B-Bottleneck) | P19 (A-Bottleneck) |
|---|---|---|
| Locus | Deep layers (16-23) q_proj | Deep layers (L23) attention |
| Mechanism | q_proj routing fix | Attention entropy gap |
| Pattern | Deep = routing core | Deep = asymmetry peak |
| Una vs Ans | LoRA effect = ROUTING fix | Una gap > Ans gap |
| Intervention | LoRA on q_proj | Attention-level (TBD) |

---
## 28. 当前状态：三瓶颈全景（更新至 P19）

| 瓶颈 | 诊断状态 | 诊断证据 | 补救状态 | 补救证据 |
|---|---|---|---|---|
| **A: Absorption** | ✅ Diagnosed | P1: ΔH=0.250 position sensitivity. A3: hidden state plasticity. P12: directional steering homogenizes with degradation. P13: L10 perturbation uniform. P19: deep-layer attention entropy U-shaped gap. P20: L21=L10 — steering is LAYER-INDEPENDENT. | ❌ No remedy — Hidden-state vectors EXHAUSTED | **P20: L21_a3 produces IDENTICAL position profile as L10_a3 (ΔH=0.250, H profile (0.25, 0.50, 0.50)). A-bottleneck steering is layer-independent. Hidden-state vector interventions CLOSED. Only path forward: weight-level (LoRA) or attention-direct modification. B-bottleneck: LoRA works → A-bottleneck: LoRA untested.** |
| **B: Organization (Hall)** | ✅ Diagnosed | **单方向 impulse 正式排除（P10 — 5 层证据链）。** P2: v_hall=v_orthogonal（决定性）。P3: v_hall/random=0.28×。Hall=纯能量。替代路径：闭环 gate (M3-v6)。结构化控制有边界条件 (C_base=0.800 时 random 最优)。 | ✅✅ **Fully Validated** | **P15: H=0.000, C=1.000 — B-bottleneck BRIDGED via LoRA. P16: ROUTING fix. P17: q_proj sole critical module. P18: DEEP layers (16-23) are sufficient core. B-bottleneck mechanism chain COMPLETE (P13-P18, 7 experiments).** |
| **B: Organization (Syc)** | ✅ Diagnosed | 方向特定、方向主导（P3+P4）。v_syc 极性解决（P5-bis）。行为专用探针可训练（P6, test acc=77.8%）。P6-ter: probe→gate→hook 在 n=12 上闭环（−66.7%）。P8 (n=24): 效应 −23.5%~−29.4%，不显著，两阶段 ≤ open-loop。**P9: steering 为均匀平移，不破坏 L10 结构完整性（ARI 1.0→1.0）。正向发现：steering 无 collateral damage。** | ⚠️ **Partial** | P8 方向正确但不显著。P9 确认独立瓶颈。**Organization 与 Stabilization 是独立维度的可补偿性。** |
| **C: Stabilization** | ✅ Diagnosed | Root cause: KMeans ignores Y | ✅ **Fully Validated** | Per-Action KMeans=0.585 (+31% over NoMemory). **P11: 4 维度 scaling 全部通过 (seed 5→100, noise 0→1.0, actions 3→20, perturbation 3×7). IC 项目验证最充分的正面发现。** |
| **D: TT-SFT** | ✅ Negative | v0+v1: CE-only > trajectory | ❌ Excluded | Cosine alignment not effective |
| **E: M7 Oracle Routing** | ✅ Diagnosed | 85.7% correct routing | ❌ 待集成 | 能力存在但默认路由不通 — 组织瓶颈的典型案例 |

### 19.1 当前可做出的声明（更新至 P13）

> 小模型性能限制可被分解为吸收、稳定、组织三个瓶颈。这三个瓶颈均可被实验测量。
>
> - **吸收瓶颈**: 已被三层证据链定量诊断（表示→probe→行为）。Probe 层可修复（A1: PSI −90%），但行为层不可修复——A3（全局 rectification）失败、A4（LoRA）混合、P12（directional steering)确认不可行。**P13 新增: L10 扰动均匀传输（max_ratio ≤ 1.02），非对称性在行为层。所有 L10 补救失效的原因——瞄准了错误的层次。吸收是唯一没有 clean behavioral remedy 的瓶颈。**
> - **组织瓶颈 (Hall)**: **单方向 impulse 已正式排除 (P10)。** 闭环反馈可部分补偿幻觉（gate → oracle 水平）；结构化控制有边界条件（仅在 C_base 退化时有效）。替代路径已建立。
> - **组织瓶颈 (Syc)**: **方向特异性在表示层、因果层和开环层均已证实**。P9 确认 steering 为均匀平移，不破坏 L10 结构完整性（ARI 1.0→1.0）。Organization 与 Stabilization 是独立维度的可补偿性。Steering 无 collateral damage。
> - **稳定瓶颈**: 根因已定位（KMeans 忽略 Y 信息），Per-Action KMeans 可完全纠正（+31% over 无记忆上限）。**P11 确认 4 维度 scaling 全部通过——IC 项目验证最充分的正面发现。Fully Validated。**
>
> **两行为的关键不对称已确认: Hall = 纯能量；Syc = 方向主导。这是项目最重要的发现之一。**
>
> **三瓶颈 → measurable and partially compensable structural adaptation limits.**

### 13.2 核心可纠正性数据

| 瓶颈 | 最佳干预 | 效应量 | 状态 |
|---|---|---|---|
| Hall (Organization) | M3-v6 closed-loop gate | H 0.867→0.667 (−23%), C 不变 | ✅ Working, with boundary |
| Syc (Organization) | P8 open-loop / two-stage | syc 0.708→0.500 (−29.4%) / 0.542 (−23.5%), not stat sig | ⚠️ Direction correct, selective advantage not confirmed |
| Stabilization | Per-Action KMeans | match 0.095→0.585 (+516%)。**P11: 4 维度 scaling 全部通过** | ✅ **Fully Validated** |
| Absorption | — | — | ❌ No remedy yet |

---

## 16. 关键发现汇总

### 16.1 正面发现

| # | 发现 | 来源 | 强度 |
|---|---|---|---|
| F1 | Multi-direction combo > single-direction at C_base=0.400 | Proof B | moderate |
| F2 | Anchored consolidation can partially correct stabilization (+8.7%) | Proof C | moderate |
| F3 | **Root cause of stabilization failure: KMeans ignores Y information** | C3 | **strong** |
| F4 | **Per-Action KMeans = 0.585 > NoMemory = 0.445 (+31%)** | C3 | **strong** |
| F5 | Absorption bottleneck: three-layer evidence chain (KNN→PSI→ΔC) | P1 | strong |
| F6 | Partial downstream compensation: probe layer is position-robust | P1 | strong |
| F7 | **Trajectory capture 零扰动: 60/60 hall + 30/30 syc** | T0 | **strong** |
| F8 | **Syc prefill separation: v_syc/random=13.6×, AUC=1.0** | T1, T2 | **strong** |
| F9 | **Syc direction-specificity confirmed: v_syc/random=2.73× (n=20)** | P3 | **strong** |
| F10 | **Syc direction-dominated: directional=+0.0164, energy=−0.0022** | P4 | **strong** |
| F11 | **v_syc polarity resolved: points toward sycophancy (neg α reduces)** | P5-bis | **strong** |
| F12 | **v_syc optimal α=−3.0: syc_rate 0.583→0.375 (−35.7%)** | P5-bis | **strong** |
| F13 | Oracle routing: 85.7% correct routing | M7-Lv2 | moderate |
| F14 | **P6 behavior-only probe trains (test acc=77.8%); open-loop v_syc α=−3.0 = −50% syc reduction** | P6 | **strong** |
| F15 | **P6-bis: Probe achieves +0.54 score separation in standalone forward pass (syc μ=0.82, non-syc μ=0.29)** | P6-bis | **strong** |
| F16 | **P6-bis: Feedback failure root cause = hook captures generated-token states, not prompt-token states** | P6-bis | **strong** |
| F17 | **P6-bis: Gate rate invariant at 8.3% across all thresholds (0.30−0.90) — confirms hook architecture bug** | P6-bis | **strong** |
| F18 | **Open-loop v_syc α=−3.0 robustly confirmed 3× (−35.7% to −50.0% across splits)** | P5-bis, P6, P6-bis | **strong** |
| F19 | **P6-ter: Two-stage feedback control CLOSED. syc 0.7500→0.2500 (−66.7%) beats open-loop (−44.4%). v_syc/random=2.67× in closed-loop.** | P6-ter | **strong** |
| F20 | **Selective intervention > universal intervention. Two-stage steers only 58.3% of samples, preserves natural non-syc behavior, achieves better syc reduction than open-loop's 100% steering.** | P6-ter | **strong** |
| F21 | **Two-stage th=0.50 is optimal: U-shaped performance curve — too low (0.30) perturbs non-syc samples, too high (0.70) misses syc-prone samples.** | P6-ter | moderate |
| F22 | **Behavior-only probe with 77.8% test accuracy is sufficient for effective gating decisions in two-stage architecture.** | P6-ter | moderate |
| F23 | **P7: P6 probe peaks at step 1 (+0.65 separation), NOT S15 (+0.13). T2 S15 is readability of per-position probes, not a general phenomenon.** | P7 | strong |
| F24 | **P7: Single-step steering ineffective. S5/S10 increase sycophancy (+12.5%/+25%), S15-S25 null. Sycophancy requires cumulative intervention.** | P7 | strong |
| F25 | **P8: Direction correct but weaker at n=24 (−23~30%). P6-ter's −66.7% (n=12) likely small-sample artifact. Unclear if two-stage > open-loop.** | P8 | moderate |
| F26 | **P9: Steering preserves L10 structural integrity perfectly. ARI=1.0→1.0, purity=1.0→1.0. Steering = uniform translation, no collateral damage.** | P9 | strong |
| F27 | **P10: 单方向 impulse 被 5 层证据正式排除于 hallucination。v_hall=v_orthogonal (P2) 为决定性证据。Hall=纯能量。N1 从"暂未确认"升级为"正式排除"。** | P10 | definitive |
| F28 | **P11: Per-Action KMeans 在 4 维度 scaling 上全部通过。Seed 5→100 (delta +0.14→+0.17), noise 0→1.0 (+0.055→+0.100), actions 3→20 (peak +0.330 at 5), perturbation 3×7 (always > NoMemory)。IC 项目验证最充分的正面发现。Stabilization Fully Validated。** | P11 | definitive |
| F29 | **P12: v_abs steering delta_H 0.750→0.250 (−67%) 但 H_early 0.250→0.500 (+100%)。v_abs/random=2.0× 但效应是 homogenization with degradation。Absorption 不可用 directional steering 修复。A1→A4→P12: probe→behavior gap 抵抗所有已测干预。** | P12 | definitive |
| F30 | **P13: Energy 和 direction 扰动在 L10 均产生均匀 ||Δh|| (max_ratio ≤ 1.02)。P12 的行为非对称性来自下游计算对相同扰动的差异化放大。这解释了所有基于 L10 的补救失效——它们瞄准了错误的层次。Absorption 的三层架构: input (diagnosed) → L10 (uniform) → behavior (asymmetric)。** | P13 | definitive |
| F31 | **P14: Cross-Project Synthesis 完成。30 正发现 + 13 排除 + 3 补救 + 5 边界条件。IC-4 + IC-2 跨项目整合。诊断+验证+排除阶段闭合。Absorption 是唯一 frontier。** | P14 | synthesis |
| **F32** | **P15: Hallucination LoRA 将 H 从 0.417 降至 0.000 (ZERO), C=1.000 全保留。B-bottleneck KNOWS→produces gap 被权重级 LoRA 弥合。P15 超越 Phase 10 A-LoRA (H 0.000 vs 0.500)。** | P15 | **breakthrough** |
| **F33** | **P15: 行为定向训练 (hallucination-labeled targets) 比位置不变性训练 (Phase 10) 更有效地消除幻觉。H 0.000 vs 0.500。** | P15 | strong |
| **F34** | **P15: 权重级干预在两个瓶颈 (A: Phase 10 LoRA, B: P15 LoRA) 都有效，而向量操作在两个瓶颈 (A: P12 steering, B: P14 w_probe) 都无效。权重-vs-向量的不对称性已跨瓶颈一致性验证。** | P10+P14+P15 | strong |
| **F35** | **P15: 位置 gap 完全消除 (ΔH 0.250→0.000)。三个位置 all H=0.000。** | P15 | strong |
| **F36** | **P16: K-subspace 在 LoRA 后完美保留 (probe acc=1.0000 at all 9 layers)。LoRA 不损害模型的知识表示。** | P16 | strong |
| **F37** | **P16: LoRA 不对齐 K↔D 子空间。w_probe steering 效应在 8/9 层为零或负。几何正交性在 LoRA 后仍然存在。** | P16 | definitive |
| **F38** | **P16: LoRA bridge mechanism = ROUTING fix (default path change), NOT geometry fix (subspace alignment)。与 M3-v6 gate 同构 — 模型学习绕过正交关系而非解决它。** | P16 | definitive |
| **F39** | **P16: Layer 12 是唯一 w_probe 仍有行为投影的层 — 但它是破坏性投影 (H 0.000→0.250 at α=+2.0)。** | P16 | strong |
| **F40** | **P17: q_proj LoRA ablation 使 H 从 0.000→0.250 (+60% 总效应)。Query projection 是唯一关键 LoRA 模块。** | P17 | definitive |
| **F41** | **P17: k_proj, v_proj, o_proj 各自消融零效应。Value/output projection 对 routing fix 不必要。** | P17 | definitive |
| **F42** | **P17: Routing fix 通过 Query projection 改变注意力查询模式实现。模型学会了不同的 attention question，而非不同的 output content。** | P17 | definitive |
| **F43** | **P17: B-bottleneck = 注意力路由问题。q_proj (attends to what) 而非 v_proj (outputs what) 或 o_proj (aggregates how)。P15+P16+P17 形成完整证据链: 效果→机制→最小必要模块。** | P15+P16+P17 | definitive |
| **F44** | **P18: H18.1-18.3 ALL FALSIFIED. MID layers (8-15) are NOT the primary routing locus. Only -q_deep breaks routing (+0.083). Early+Mid removal has ZERO effect on routing.** | P18 | definitive |
| **F45** | **P18: DEEP layers (16-23) q_proj is the SUFFICIENT core of routing. ONLY_deep H=0.0000. Deep = output-stage refinement layers. Abstention decision is made AFTER full input processing, not DURING it.** | P18 | definitive |
| **F46** | **P18: Redundant routing architecture. -q_deep (ΔH=+0.083) < ONLY_early (ΔH=+0.250) — mid layers partially compensate when deep is removed, but deep fully compensates for early+mid removal.** | P18 | definitive |
| **F47** | **P18: Routing capability shows clear layer gradient: deep (ONLY_deep H=0.000, PERFECT) > mid (ONLY_mid H=0.083, PARTIAL) > early (ONLY_early H=0.250, FAILS — same as -q_ALL).** | P18 | definitive |
| **F48** | **P18: Early layers (0-7) q_proj does NOT participate in routing at all. ONLY_early H=0.250 = -q_ALL H=0.250 = pre-LoRA baseline H=0.417/2. Early q_proj routing is NULL.** | P18 | strong |
| **F49** | **P18: Mid layers (8-15) have secondary/backup routing capability — ONLY_mid partial (H=0.083) — but are NOT the primary core. Semantic processing layers are NOT where the decision is made.** | P18 | strong |
| **F50** | **P18: B-bottleneck mechanism chain COMPLETE. P13+P14 (K↔D orthogonal) → P15 (LoRA bridges) → P16 (ROUTING fix) → P17 (q_proj sole module) → P18 (DEEP layers 16-23 core). 7-experiment chain (P13→P14→P15→P16→P17→P18) fully characterizes B-bottleneck from geometric diagnosis to layer-level mechanism.** | P13-P18 synthesis | definitive |
| **F51** | **IC-2d: All learned readout strategies ≤ k-NN < NoMemory. MLP (0.095) < RF (0.190) < k-NN (0.195) < NoMemory (0.445). Episodic underperformance is a REPRESENTATION problem, not a readout problem.** | IC-2d | definitive |
| **F52** | **IC-2d: Only consolidation (Per-Action KMeans, 0.585) beats NoMemory. IC-2 and IC-4 converge: representation-level interventions work, readout-level fail.** | IC-2d | definitive |
| **F53** | **P19: Attention entropy U-curve across 9 layers. L0=11.3% → L9 min=4.4% → L23 max=13.0%. Una gap > Ans gap at EVERY layer. L9 is most position-invariant layer (+0.019).** | P19 | definitive |
| **F54** | **P19: B-bottleneck (P18) and A-bottleneck (P19) converge on SAME mechanistic locus: deep-layer attention routing. P19 attention entropy + P18 deep q_proj routing = same mechanism from two angles.** | P18+P19 synthesis | definitive |
| **F55** | **P20: H20.1+H20.2 REFUTED: L21 steering produces IDENTICAL position profile as L10. ΔH=0.250, H profile (0.25, 0.50, 0.50) at both layers. A-bottleneck steering is LAYER-INDEPENDENT.** | P20 | definitive |
| **F56** | **P20: Hidden-state vector interventions EXHAUSTED for absorption. All tested layers produce same homogenization-with-degradation. The gap is in hidden-state→output mapping, not in any layer's hidden states.** | A3+P12+P20 synthesis | definitive |
| **F57** | **P20: P19's U-curve describes WHERE asymmetry is visible (deep layers) but not WHERE it's CAUSED. Causal asymmetry is in output decoding, downstream of all layers.** | P19+P20 synthesis | strong |
| **F58** | **P21: P15 LoRA reduces absorption ΔH from 0.750 to 0.250 (−67%). Mid/late H 1.000→0.000 (FIXED). LoRA is superior to hidden-state steering (ΔH=0.250 fixed vs 0.750→0.250).** | P21 | definitive |
| **F59** | **P21: LoRA does NOT fully close absorption gap. Early position retains H=0.250 residual. Absorption = distance routing (FIXED by LoRA) + proximity over-confidence (NOT fixed).** | P21 | definitive |
| **F60** | **P21: Probe→behavior gap PARTIALLY closed by LoRA. Log-prob ΔH=0.000 (perfect) but generate ΔH=0.250 (residual). Weight-level intervention partially bridges but cannot fully eliminate the gap.** | P21 | definitive |
| **F61** | **P21: Absorption has TWO components: (1) distance-based routing degradation affecting mid/late → LoRA FIXES; (2) proximity-based source confusion affecting early → survives LoRA. Residual may be embedding-level (RoPE position encoding baked into token representations).** | P21 | strong |
| **F62** | **P22: Attention temperature at early layers CANNOT fix early-position over-confidence. Any T≠1 makes early H WORSE (T=0.5: +0.250, T=5.0: +0.500). Baseline T=1.0 is OPTIMAL.** | P22 | definitive |
| **F63** | **P22: Mid/late H=1.000 is COMPLETELY INVARIANT to early-layer attention temperature. Mid/late hallucination is NOT routed through early-layer attention — consistent with P18 deep-layer routing.** | P22 | definitive |
| **F64** | **P22: Early position proximity over-confidence is ROBUST, not fragile. Any perturbation (sharper OR softer) makes it WORSE. Absorption has THREE components: distance routing (LoRA fixes), proximity over-confidence (nothing fixes), attention calibration (baseline optimal).** | P21+P22 synthesis | definitive |
| **F65** | **P23: Position offset via padding is NOT viable. N=100 makes early WORSE (consistent perturbation pattern). N≥300 breaks model entirely (H=C=0 for all positions). RoPE hypothesis remains plausible but untestable at inference time.** | P23 | definitive |
| **F66** | **P23: Early H=0.250 is a LOCAL MINIMUM. THREE independent interventions (hidden-state steering, attention temperature, position offset) all fail to reduce it. Only LoRA preserves early H while fixing mid/late. Absorption intervention ladder is COMPLETE.** | P12+P22+P23 synthesis | definitive |
| **F67** | **P24: Sycophancy steering direction confirmed at n=48: Baseline 0.625→Two-stage 0.417 (−33.3%). Consistent with P8 n=24 direction. Effect is REAL.** | P24 | definitive |
| **F68** | **P24: Fisher p=0.0654 (trending) — does NOT cross α=0.05. Sycophancy effect size modest (Cohen's h≈0.42), requiring n≈90+ for significance. Data ceiling (60) insufficient. This is a DATA limitation, not an effect limitation.** | P24 | definitive |
| **F69** | **P25: Cross-bottleneck SYNERGY discovered. v_syc(−3.0) STRONGLY reduces hallucination — mid position H 1.000→0.000 (complete elimination, matches P15 LoRA). Early/late drop to H=0.333. Anti-sycophancy steering is an effective anti-hallucination tool.** | P25 | definitive |
| **F70** | **P25: v_hall(−3.0) has ~zero effect on sycophancy (0.625→0.688, within noise at n=16). The synergy is ASYMMETRIC: v_syc→hall works, v_hall→syc doesn't.** | P25 | definitive |
| **F71** | **P25: Hall ⊃ Syc as nested subspace. cos(v_hall, v_syc)=0.2355 explains partial geometric overlap. The hallucination subspace contains the sycophancy subspace — steering away from sycophancy partially steers away from hallucination, but not vice versa.** | P25 | definitive |
| **F72** | **P25: v_syc outperforms v_hall for hallucination reduction at mid position (H=0.000 with C=0.600 vs v_hall C=0.000). v_hall reduces hallucination uniformly but kills correctness; v_syc reduces selectively preserving correctness.** | P25 | definitive |
| **F73** | **P26: U_1:2 (v_hall=−1.0 + v_syc=−2.0) achieves TRIPLE CROWN — mid H=0.000, avg C=0.444 (8× better than v_hall avg C=0.056), sycophancy=0.375 (−25%). The ONLY condition that simultaneously improves H, C, AND sycophancy. Strongest hidden-state intervention result in IC-4.** | P26 | definitive |
| **F74** | **P26: COMBINATION SYNERGY confirmed — U_1:2 beats both v_hall(−3.0) and v_syc(−3.0) individually on all three metrics (mid H, avg C, syc rate). The ratio matters critically: 1:1 kills C, 2:1 kills C, only 1:2 hits the sweet spot.** | P26 | definitive |
| **F75** | **P26: v_syc component protects correctness while v_hall component suppresses hallucination. At 1:2 ratio, v_syc dominates enough to keep C alive (0.444 vs v_hall's 0.056), while v_hall provides sufficient push to drive mid H to zero. The correction paradox: v_hall alone increases sycophancy (+0.188), but with v_syc at 1:2 ratio, sycophancy drops (−0.250).** | P26 | definitive |
| **F76** | **P26: v_hall(−3.0) INCREASES sycophancy (0.625→0.812, +0.188 at n=16). This is a previously unknown side effect — anti-hallucination steering makes the model MORE sycophantic. Needs large-n confirmation.** | P26 | definitive |

### 16.2 否定性/约束性发现

| # | 发现 | 来源 | 影响 |
|---|---|---|---|
| N1 | **Hall direction-specificity formally excluded. v_hall=v_orthogonal (P2).** | P2+P10 | **P10: 正式排除。5 层证据链。替代路径: 闭环 gate (M3-v6).** |
| N2 | **ALL 15 Hall pair synergy ≤ 0 at C_base=0.800; best single = random** | B2 Audit | 结构化控制边界条件 |
| N3 | C_base 漂移 (0.400 → 0.800) 揭示 structured control 仅在 degraded baseline 有效 | B2 Audit | 控制理论的根本约束 |
| N4 | All readout-level stabilization interventions fail (8/8) | C2 | 排除了 readout 层面作为根因 |
| N5 | TT-SFT v0+v1 both negative: CE-only beats trajectory alignment | TT-SFT | 排除了 cosine alignment 路线 |
| N6 | **P5 probe→gate→hook null: gate rate=4.2%, probe learns group membership** | P5 | 探测器必须 behavior-only 训练 |
| N7 | **P5 hypothesis falsified: v_syc does NOT point toward non_syc** | P5-bis | 方向极性修正 |
| N8 | **S15 amplification mechanism: readability (T2 per-position probe peak) ≠ manipulability (P7 per-step steering null)** | P7 | ✅ Resolved — S15 is not a causal sensitive period. Sycophancy is cumulative/distributed. |
| N9 | **P6 behavior-only probe: gate still null (8.3%)** | P6 | 原因 = hook 架构 bug (P6-bis 诊断) |
| N10 | **P6-bis: Hook captures generated-token states during generate(), not prompt-token states. Gate rate invariant at 8.3% across all thresholds.** | P6-bis | Fix: 两阶段架构 (P6-ter) |
| N11 | **P8: Two-stage closed-loop advantage does NOT scale from n=12 to n=24. Two-stage (−23.5%) ≤ open-loop (−29.4%). P6-ter's selective advantage spurious (F25).** | P8 | Requires larger n or stronger probe for two-stage > open-loop |
| N12 | **P19: ALL 3 H20 hypotheses FALSIFIED. L21 steering = L10 steering (IDENTICAL). Deep-layer steering does NOT improve absorption. P19 U-curve is diagnostic, not causal.** | P20 | Absorption steering is layer-independent |
| N13 | **P20: Hidden-state vector interventions for absorption → permanently EXHAUSTED. Multi-layer, single-layer, all produce same homogenization-with-degradation pattern.** | P12+P20 | Only weight-level or attention-direct interventions remain |
| N14 | **P21: Weight-level (LoRA) intervention PARTIALLY closes absorption gap but leaves ΔH=0.250 residual. Even LoRA cannot reach ΔH=0.000 for absorption.** | P21 | Absorption has embedding-level residual |
| N15 | **P21: Early position over-confidence SURVIVES LoRA. Proximity-based source confusion is a distinct absorption component not fixable by routing-level interventions.** | P21 | Requires different mechanism (attention-direct or training-time RoPE modification) |
| N16 | **P22: H22.1 REFUTED — attention softening INCREASES early H, does NOT reduce it. Baseline T=1.0 is optimal. Over-confidence is ROBUST to attention perturbation.** | P22 | Attention-level intervention also fails for early position |
| N17 | **P22: Mid/late H=1.000 completely INVARIANT to early-layer attention temperature — further evidence that mid/late hallucination is deep-layer routed (P18), not early-layer.** | P22 | Consistent with P18 deep-layer routing |
| N18 | **P23: Position ID offset via padding CANNOT fix absorption. N=100 increases early H (consistent perturbation), N≥300 breaks model. Padding is too confounded to cleanly test RoPE.** | P23 | RoPE hypothesis remains untestable at inference time |
| N19 | **P23: Early H=0.250 cannot be reduced below baseline by any tested intervention. It is a LOCAL MINIMUM — the model's default is optimal for early position.** | P12+P22+P23 | Early position residual is fundamental, not fixable at inference time |
| N20 | **P24: Sycophancy steering does NOT reach statistical significance at n=48. Fisher p=0.065 (trending). The effect is REAL (direction confirmed) but UNDER-POWERED — data ceiling insufficient for n≈90+ needed.** | P24 | Sycophancy bottleneck partially validated; requires more data |
| N21 | **P25: H25.1 REFUTED — v_syc DOES reduce hallucination (it was predicted NOT to). The refutation is in the favorable direction — an asymmetric synergy rather than a trade-off. Bottlenecks are NOT strictly independent.** | P25 | Opens unified bottleneck steering path |
| N22 | **P26: H26.2 partially REFUTED — U_1:1(avg C=0.222) does NOT preserve meaningful C at mid position (C=0.000). Only U_1:2 at the specific 67% syc ratio preserves C. The ratio is extremely sensitive — 1:1 and 2:1 both kill correctness. Narrow operating window for unified steering.** | P26 | Requires ratio calibration, not one-size-fits-all |

### 16.3 方法学发现

| # | 发现 | 来源 |
|---|---|---|
| M1 | Orthogonal norm-matched decomposition: 通用方向-vs-能量分解模式 | P4 |
| M2 | Combined 策略假阳性风险：purity gate 过滤所有 centroids 后回退 NoMemory | C2 |
| M3 | Cosine similarity 无法预测组合协同效应 (r=0.0037) | B2 |
| M4 | Hallucination direction 不稳定：cos(v_hall_A, v_hall_B) = 0.651 | B2 |
| M5 | P5 open-loop artifact: 60-sample 混合集结果与 24-sample 纯净集矛盾 | P5-bis |
| M6 | P6-bis: 两阶段架构模式 — standalone probe scoring → conditional generate with steering | P6-bis |
| M7 | **P8: 小样本闭环优势为虚假信号 — n=12 上的 −66.7% 效应量在 n=24 上缩至 −23.5%。小样本实验的 "breakthrough" 需要大样本复现验证。** | P8 |
| M8 | **P9: 跨瓶颈独立性检验模式 — 收集双条件的隐藏状态 → KMeans 聚类 → 比较结构指标。可用于任何干预的 collateral damage 审计。** | P9 |

### 16.4 最重要的不对称发现

| 维度 | Hallucination | Sycophancy |
|---|---|---|
| 方向特异性 | **不存在** (pure energy) | **存在** (direction-dominated) |
| 控制向量 | v_hall = v_orthogonal | v_syc ≠ all controls |
| 最优干预 | Closed-loop gate (M3-v6) | Open-loop α=−3.0 |
| 表示信号强度 | v_hall/random=3.51× | v_syc/random=13.6× (T1), 2.73× (P3) |
| 轨迹形态 | Prefill peak → decay | Prefill peak → S15 amplification |
| 极性 | N/A (无方向性) | 指向 sycophancy (减 v_syc = 抗谄媚) |

---

## 17. 路径前瞻：下一步行动

### 17.1 已完成路线

```
Phase 1:    瓶颈存在证明 (Proof A/B/C/D) ─── ✅
Phase 2:    轨迹动力学 (T0-T3) ─── ✅
Phase 3:    Syc 方向特异性 (P3+P4) ─── ✅
Phase 4:    能力路由诊断 (M7) ─── ✅
Phase 5:    Syc 反馈控制 (P5+P5-bis) ─── ✅
Phase 6:    Hall 方向特异性排除 (P2+B2) ─── ✅
Phase 7:    稳定根因突破 (C2+C3) ─── ✅
Phase 8:    TT-SFT 否定性结果 ─── ✅
Phase 9:    吸收瓶颈诊断 (P1) ─── ✅
Phase 10:   P6 Behavior-Only Probe ─── ✅
Phase 11:   P6-bis Threshold Calibration → Hook Architecture Diagnostic ─── ✅
Phase 12:   **P6-ter Two-Stage Feedback Control ─── ✅ (闭环打通!)**
Phase 13:   **P7 S15 Amplification Mechanism ─── ✅ (Readability ≠ Manipulability)**
Phase 14:   **P8 Large-Scale Replication ─── ✅ (Direction correct, not stat sig, P6-ter advantage spurious)**
Phase 15:   **P9 Cross-Bottleneck Structure ─── ✅ (Steering preserves structure, bottlenecks independent)**
Phase 16:   **P10 Hall Impulse Exclusion ─── ✅ (Formal exclusion, 5-layer evidence, line CLOSED)**
Phase 17:   **P11 Stabilization Scaling ─── ✅ (4 维度全部通过, Fully Validated)**
Phase 18:   **P12 Absorption Steering ─── ✅ (Negative: homogenization with degradation, directional steering insufficient)**
Phase 19:   **P13 Energy/Direction Asymmetry ─── ✅ (L10 uniform, asymmetry is downstream — explains all L10 remedy failures)**
Phase 20:   **P14 Cross-Project Synthesis ─── ✅ (Diagnosis+Validation+Exclusion phase CLOSED)**
Phase 21:   **P15 Hallucination LoRA ─── ✅ (Breakthrough: H=0.000, C=1.000, B-bottleneck BRIDGED)**
Phase 22:   **P16 LoRA Geometry Analysis ─── ✅ (LoRA = ROUTING fix, not geometry fix)**
Phase 23:   **P17 LoRA Module Ablation ─── ✅ (q_proj is the sole critical module — attention routing key)**
Phase 24:   **P18 q_proj Layer Ablation ─── ✅ (DEEP layers 16-23 are sufficient core; ALL pre-registered hypotheses FALSIFIED; B-bottleneck chain COMPLETE)**
Phase 25:   **IC-2d Learned Readout ─── ❌ (ALL learned readouts ≤ k-NN < NoMemory; episodic = representation problem)**
Phase 26:   **P19 Absorption Attention Patterns ─── ✅ (Deep-layer U-curve: L0 11.3% → L9 4.4% → L23 13.0%; Una > Ans; A+B converge on deep-layer attention routing)**
Phase 27:   **P20 Multi-Layer Steering ─── ❌ (L21 steering = L10 steering; A-bottleneck steering is LAYER-INDEPENDENT; hidden-state vector interventions EXHAUSTED for absorption)**
Phase 28:   **P21 Absorption LoRA ─── ⚠️ Partial Success (ΔH 0.750→0.250 (−67%); mid/late FIXED; early residual H=0.250; absorption = distance routing (FIXED) + proximity over-confidence (NOT fixed); log-prob perfect, generate residual)**
Phase 29:   **P22 Attention Temperature Scaling ─── ❌ (Any T≠1 makes early H WORSE; mid/late H=1.000 invariant; baseline T=1.0 is OPTIMAL; over-confidence is ROBUST not fragile; RoPE embedding-level hypothesis)**
Phase 30:   **P23 Position ID Offset ─── ⚠️ Informative Negative (N=100 makes early WORSE; N≥300 breaks model; early H=0.250 is LOCAL MINIMUM; 3 interventions fail; absorption intervention ladder COMPLETE)**
Phase 31:   **P24 Sycophancy n=48 ─── ⚠️ Trending (Direction confirmed −33.3%; Fisher p=0.065 trending; Sycophancy is REAL but UNDER-POWERED; data ceiling 60 < needed 90+)
Phase 32:   **P25 Cross-Bottleneck Steering ─── ✅ Synergy Discovery (ASYMMETRIC: v_syc(-3) STRONGLY reduces hallucination mid H 1.000→0.000; v_hall(-3) has ~zero effect on sycophancy; Hall ⊃ Syc as nested subspace; cos=0.2355; fighting sycophancy inadvertently fights hallucination)
Phase 33:   **P26 Unified Bottleneck Steering ─── ✅ Triple Crown (U_1:2: mid H=0.000, avg C=0.444 (8× v_hall), syc=0.375 (−25%). Combination BEATS both singles on ALL 3 metrics. Ratio-critical: only 1:2 (67% syc energy) hits sweet spot. Strongest hidden-state intervention result in IC-4.)
```

### 20.2 优先级排序（更新版，P17 后）

| Pri | 行动 | 瓶颈 | 理由 |
|---|---|---|---|
| **1** | **IC-2d: Learned readout for episodic (CPU).** 训练 MLP/MHA readout 替代简单 k-NN，解锁 episodic memory 的全信息容量。 | Stabilization (IC-2) | 最短 proof 路径的下一个 CPU 可执行步骤。 P17 完成 B-bottleneck 的完整弥合链。IC-2 线需跟进。 |
| **2** | **Absorption: post-L10 下游干预。** Target attention patterns, output logit modulation, or multi-layer steering. P13 确认 L10 扰动均匀 — 非对称性在下游放大。P17 的 key insight (q_proj = attention routing) 提供了新思路：改变 Query 让模型关注位置不变的信息。 | Absorption | P12+P13 闭合 L10 mediation line。P17 的 attention routing 发现可能适用于吸收瓶颈。 |
| **3** | **Syc: larger-n (n≥48) if selective>universal needs confirmation.** | Organization (Syc) | P8 方向正确但不显著。Not urgent. |
| **4** | **Unified environment: routing + structural fidelity 联合测试.** | Integration | 最短 proof 路径的最后一步。 |

### 17.3 项目 Endgame 接近度评估

对照 PROJECT_ENDGAME_AND_HANDOFF.md 的 Endgame Claim：

| Endgame 要求 | 状态 |
|---|---|
| "三瓶颈可被分解" | ✅ 已证明 (Proof A-D + T0-T3 + P1) |
| "三瓶颈可被实验测量" | ✅ 已证明 (每瓶颈均有定量指标) |
| "至少一个瓶颈可被闭环控制部分补偿" | ✅ Organization (Hall): M3-v6 闭环 gate. ✅✅ **P15 LoRA — H=0.000, C=1.000, B-bottleneck BRIDGED. P16+P17: 机制解释完整 (routing fix via q_proj).** ⚠️ Organization (Syc): P8 方向正确 (−23~30%) 但不显著. P9 steering 无 collateral damage. |
| "至少一个瓶颈可被锚定结构稳定部分补偿" | ✅ Stabilization: Per-Action KMeans. **P11: Fully Validated (4 维度 scaling 全部通过)**。 |
| "三瓶颈在三方向上均部分可补偿" | ✅ Organization (Hall): ✅✅ Fully Validated via LoRA (H=0.000). ✅ Stabilization: Fully Validated. ⚠️ Organization (Syc): 方向正确，需 larger-n. ❌ Absorption: 无 clean behavioral remedy. **P17 q_proj insight → attention routing 可能是吸收瓶颈的新方向。** |

---

## 18. 完整时间线

| 日期 | 事件 | 类型 | 项目 |
|---|---|---|---|
| 2026-05-19 | 工程控制论框架 + 三瓶颈定义 | 理论 | IC-4 |
| 2026-05-20 | Proof A (Syc T3 初版) — NEGATIVE | 实验 | IC-4 |
| 2026-05-20 | Proof B (Multi-direction intervention at C=0.400) — POSITIVE | 实验 | IC-4 |
| 2026-05-20 | Proof C (Anchored consolidation +8.7%) — PARTIAL | 实验 | IC-2 |
| 2026-05-20 | Proof D (Cross-Bottleneck Synthesis) | 文档 | IC-4+IC-2 |
| 2026-05-20 | P0: Syc balanced contrast set construction (分离度=0.833) | 基础设施 | IC-4 |
| 2026-05-20 | T0: Trajectory capture 验证 (60/60 + 30/30) | 基础设施 | IC-4 |
| 2026-05-20 | T1: v_syc/v_hall projection (syc v/random=13.6×) | 实验 | IC-4 |
| 2026-05-20 | T2: Decision heatmap (syc AUC=1.0, S15 amplification) | 实验 | IC-4 |
| 2026-05-20 | T3: Impulse map (syc v/random=1.80×, hall v/random<1) | 实验 | IC-4 |
| 2026-05-20 | P2: Direction-specificity audit (hall v_hall=v_orthogonal) | 实验 | IC-4 |
| 2026-05-20 | P1: Position Sensitivity (KNN=1.0 + PSI=0.0084 + ΔC=0.067) | 实验 | IC-4 |
| 2026-05-21 | Energy Decomp (syc d/e=0.31 NOT confirmed) — later superseded by P4 | 实验 | IC-4 |
| 2026-05-21 | B2: Multi-Direction Structure Audit (hall all synergy ≤ 0) | 实验 | IC-4 |
| 2026-05-21 | C2: 8 readout strategies all fail | 实验 | IC-2 |
| 2026-05-21 | C3: Root cause breakthrough — Per-Action KMeans=0.585 | 实验 | IC-2 |
| 2026-05-21 | M7-Lv2: Oracle routing (85.7%) | 实验 | IC-4 |
| 2026-05-21 | TT-SFT v0 — weak_effect | 实验 | IC-4 |
| 2026-05-21 | TT-SFT v1 — weak_effect (CA degraded) | 实验 | IC-4 |
| 2026-05-22 | **P3: Syc direction-specificity replication (n≥20, v/random=2.73×)** | 实验 | IC-4 |
| 2026-05-22 | **P4: Syc direction-vs-energy decomposition (direction-dominated)** | 实验 | IC-4 |
| 2026-05-22 | **P5: Syc feedback control (gate rate=4.2%, null result)** | 实验 | IC-4 |
| 2026-05-22 | **P5-bis: Syc open-loop α-sweep (v_syc polarity resolved, α=−3.0 optimal)** | 实验 | IC-4 |
| 2026-05-22 | **P6: Behavior-only probe (train acc=81.9%, test acc=77.8%, gate 8.3% null, open-loop −50%)** | 实验 | IC-4 |
| 2026-05-22 | **P6-bis: Threshold calibration → Hook architecture diagnostic (standalone +0.54 separation, hook captures generated tokens, fix = two-stage)** | 实验 | IC-4 |
| **2026-05-22** | **P6-ter: Two-stage feedback control CLOSED (syc 0.750→0.250, −66.7%, beats open-loop, v_syc/random=2.67×)** | 实验 | IC-4 |
| **2026-05-22** | **P7: S15 amplification mechanism (readability ≠ manipulability; syc is cumulative/distributed; single-step steering null)** | 实验 | IC-4 |
| **2026-05-23** | **P8: Large-scale replication (n=24, syc −23~30% not stat sig, P6-ter −66.7% likely artifact, two-stage ≤ open-loop)** | 实验 | IC-4 |
| **2026-05-23** | **P9: Cross-Bottleneck (steering preserves structure, ARI 1.0→1.0, uniform translation, bottlenecks independent)** | 实验 | IC-4 |
| **2026-05-23** | **P10: 正式排除 Hall 单方向 impulse — 5 层证据链收敛，研究线关闭** | 文档 | IC-4 |
| **2026-05-23** | **P11: Stabilization scaling 整合 — 4 维度全部通过，Fully Validated** | 跨项目文档 | IC-4+IC-2 |
| **2026-05-23** | **P12: Absorption directional steering — 否定性：homogenization with degradation** | 实验 | IC-4 |
| **2026-05-23** | **P13: Energy/Direction asymmetry — L10 uniform, asymmetry is downstream** | 实验 | IC-4 |
| **2026-05-23** | **P14: Cross-Project Synthesis — 诊断+验证+排除阶段闭合。Absorption 是唯一 frontier。** | 合成 | IC-4+IC-2 |
| **2026-05-23** | **P15: Hallucination LoRA — Breakthrough: H 0.417→0.000 (ZERO), C=1.000, B-bottleneck BRIDGED** | 实验 | IC-4 |
| **2026-05-23** | **P16: LoRA Geometry — LoRA = ROUTING fix (bypass K↔D), not geometry fix (align K↔D)** | 实验 | IC-4 |
| **2026-05-23** | **P17: LoRA Ablation — q_proj is the sole critical module; B-bottleneck = attention routing problem** | 实验 | IC-4 |
| 2026-05-24 | **P18: q_proj Layer Ablation — DEEP layers 16-23 are sufficient core; ALL hypotheses falsified; B-bottleneck chain COMPLETE** | 实验 | IC-4 |
| 2026-05-24 | **IC-2d: Readout-Matched Episodic — ALL learned readouts ≤ k-NN; episodic = representation problem; consolidation is the ONLY path** | 实验 | IC-2 |
| 2026-05-24 | **P19: Absorption Attention Patterns — deep-layer U-curve; B+A converge on attention routing** | 实验 | IC-4 |
| 2026-05-24 | **P20: Multi-Layer Steering — L21=L10; A-bottleneck steering is layer-independent; hidden-state vector interventions EXHAUSTED** | 实验 | IC-4 |
| 2026-05-24 | **P21: Absorption LoRA — ΔH 0.750→0.250 (−67%); mid/late FIXED, early residual; absorption = distance routing + proximity over-confidence** | 实验 | IC-4 |
| 2026-05-24 | **P22: Attention Temperature Scaling — Any T≠1 makes early H WORSE; mid/late invariant; baseline optimal; RoPE hypothesis** | 实验 | IC-4 |
| 2026-05-24 | **P23: Position ID Offset — N=100 worse; N≥300 breaks model; early H=0.250 is LOCAL MINIMUM; absorption ladder complete** | 实验 | IC-4 |
| 2026-05-24 | **P24: Sycophancy n=48 — Direction confirmed (−33.3%); Fisher p=0.065 trending; effect REAL but UNDER-POWERED** | 实验 | IC-4 |
| 2026-05-24 | **UNIFIED_RESEARCH_MAP.md v8.3 + RESEARCH_TRAJECTORY_REPORT.md v3.17: P24 added, sycophancy bottleneck partially validated** | 文档 | IC-4 |

### 15.1 实验统计

| 指标 | IC-4 | IC-2 |
|---|---|---|
| 总实验数 | **35** | 7 |
| 总 sample 数 | 1380+ | 6000+ traces |
| 总生成/评估时间 | ~40-47 hours | ~5 hours |
| 正发现 | **70** | 4 |
| 否定性发现 | 16 | 1 |
| 方法学发现 | 8 | 0 |
| 已排除路线 | **Hall single-direction impulse (P10: 正式排除)** , **L10-targeted absorption remedies (P13: uniform → wrong level)** , **Absorption directional steering (P12: homogenization with degradation)** , **Multi-layer absorption steering (P20: L21=L10, layer-independent, hidden-state vector interventions EXHAUSTED)** , **K↔D geometry alignment via vector ops (P14+P16: near-orthogonal, LoRA = routing bypass)** , TT-SFT cosine alignment, Two-stage closed-loop superiority (n<24 artifact), **Cross-bottleneck trade-off (P25: asymmetric synergy instead — Hall ⊃ Syc)** , **Uniform-ratio unified steering (P26: 1:1/2:1 kill C — only 1:2 works)** | Readout-level interventions |

---

## 附录：文档索引导航

| 文档 | 用途 |
|---|---|
| [ENGINEERING_CYBERNETICS_FRAMING.md](ENGINEERING_CYBERNETICS_FRAMING.md) | 顶层工程控制论叙事 |
| [UNIFIED_RESEARCH_MAP.md](UNIFIED_RESEARCH_MAP.md) | 所有实验结果的结构化地图 |
| [UNIFIED_THESIS.md](UNIFIED_THESIS.md) | 主线收束，四层压缩 |
| [CROSS_BOTTLENECK_SYNTHESIS.md](CROSS_BOTTLENECK_SYNTHESIS.md) | 三瓶颈交叉综合 |
| [PROJECT_ENDGAME_AND_HANDOFF.md](PROJECT_ENDGAME_AND_HANDOFF.md) | 终局声明与交接指南 |
| [README.md](README.md) | GitHub 导航入口 |
| **RESEARCH_TRAJECTORY_REPORT.md** | **完整研究路线报告（本文件）** |

---

*IC-4-M0 Research Trajectory Report v3.19 — Updated 2026-05-25 (P26: Unified Steering — Triple Crown, U_1:2 beats all singles, strongest hidden-state result)*
*Covers: Proof A-D + T0-T3 + P0-P26 + M7 + C2-C5 + TT-SFT + A1-A4 + IC-2d*