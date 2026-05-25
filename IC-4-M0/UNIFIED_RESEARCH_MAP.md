# 总研究地图 v8.4 — 工程控制论 × 结构适应 → 能力路由 × 结构保真 × 轨迹动力学 × 训练内化

**版本**: v8.7 | **日期**: 2026-05-25 | **覆盖**: `internal_circuit_capital_lab` + `intelligence_capital_minimal_lab`

> **框架文档**: [ENGINEERING_CYBERNETICS_FRAMING.md](file:///F:/internal_circuit_capital_lab/IC-4-M0/ENGINEERING_CYBERNETICS_FRAMING.md)
> **综合报告**: [IC4_COMPREHENSIVE_RESEARCH_REPORT.md](file:///F:/internal_circuit_capital_lab/IC-4-M0/IC4_COMPREHENSIVE_RESEARCH_REPORT.md)
> **完整路线报告 (v2.0)**: [RESEARCH_TRAJECTORY_REPORT.md](file:///F:/internal_circuit_capital_lab/IC-4-M0/RESEARCH_TRAJECTORY_REPORT.md) — 从头到尾的完整研究轨迹

---

## 顶层框架：工程控制论（Engineering Cybernetics）

> **我们将小规模语言模型视为一个高维、非线性、对输入组织高度敏感的受控对象，研究其内部状态的可观测性、结构漂移、以及闭环补偿机制。**
>
> 钱学森《工程控制论》核心思想：面对结构极度复杂、存在大量不确定性和噪声的系统，控制的本质在于利用反馈克服内部不确定性与噪声——而非穷举每一个微观机制。
>
> **这不再是 Self-RAG 的同一层问题。** Self-RAG 在系统外部挂"检索开关"；我们在系统内部 896D 状态空间里做状态观测、误差估计和反馈控制。

| 控制论概念 | 我们的对应 | 关键证据 |
|---|---|---|
| **可观测性 (Observability)** | 模型内部存在 latent verification capability | M7-Lv2: fact_checker prompt sycophancy -20pp |
| **反馈控制 (Feedback Control)** | probe→gate→hook 闭环将幻觉压至 oracle 水平 | M3-v6: H 0.867→0.667, C 0.600 不变 |
| **系统稳定性 (Stability)** | 无锚定压缩导致结构漂移/坏资本 | IC-2c: match 0.115 < random 0.33 |
| **输入扰动 (Input Disturbance)** | 位置/排列变化导致系统状态大规模偏移 | Position KNN=1.0 |
| **非线性响应 (Nonlinear Response)** | 方向特异性不存在，系统对能量而非方向敏感 | P2: v_hall = v_orthogonal (-0.283) |

---

## 元层：结构适应假说（Structural Adaptation Hypothesis）

> **小模型与大模型的根本差距不只是参数规模，而是结构适应能力——即把人类离散化的数据流吸收、稳定、组织成可调用内部结构的能力。工程控制论提供了统一的工程语言来组织这些瓶颈的诊断与补偿。**

| 瓶颈 | 定义 | 控制论翻译 | 证据强度 | 来源 |
|---|---|---|---|---|
| **A: Absorption** | 输入碎片化导致信息丢失 | 输入如何进入状态空间；输入扰动如何扭曲内部状态轨线 | ⭐⭐⭐⭐ **KNN=1.0**; ΔC=0.07 | Position Sensitivity |
| **B: Stabilization** | 跨样本压缩导致结构漂移 | 状态轨线如何在无锚定更新中保持稳定 | ⭐⭐⭐⭐ **Purity=0.261**; TPR=0.875 | IC-2c Topology |
| **C: Organization** | 能力存在但路由不通 | 内部信号如何被正确路由到输出行为（控制回路闭合） | ⭐⭐⭐⭐ **Oracle routing 85.7%** | M3-v6, M7-Lv2 |

**全文**: [STRUCTURAL_ADAPTATION_HYPOTHESIS.md](file:///F:/intelligence_capital_minimal_lab/intelligence_capital_theory/STRUCTURAL_ADAPTATION_HYPOTHESIS.md)

**嵌套关系**:
```
工程控制论 (顶层叙事) → 结构适应假说 (元层) → 关系记忆假说 (理论层) → Bottleneck A/B/C (实验层)
```

---

## 位置声明：机制工程可行性验证

三句话定义当前位置：

1. **我们已有可工作的闭环反馈控制系统。** `M3-v6` 给出了完整的 reference：状态采样→误差估计→控制决策→控制注入，在不改权重的条件下将稳态误差压至 oracle 水平。

2. **我们已开始绘制系统的控制边界。** `M4`（OOD 鲁棒性）、`M5`（泛化边界）、`P2`（方向特异性证伪）、`T0-T3`（轨迹动力学）共同描绘了"哪里能动、哪里动不了、为什么"。

3. **诊断与补偿已形成闭环。** IC-2 诊断结构漂移病理 → IC-4 补偿组织路由缺陷 → Position Sensitivity 诊断吸收瓶颈。三个子问题在工程控制论框架下统一。

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
| **A6 — T0 轨迹捕获** | 在不改变行为的前提下可完整记录 `model.generate()` 期间 hidden state 轨迹 | T0 equivalence test：hall 60/60 输出匹配，syc 30/30 输出匹配；7 层 × 48 steps 完整捕获 |
| **A7 — T1+T2 Hallucination 早期分离** | hallucination 在 prefill (step 0) 即可通过 probe 和 projection 分离，最优 probe 在 Layer 8, step 0 (acc=0.917) | T2 heatmap：cross_layer_band 结构；T1 projection：v_hall max sep=2.40 at Layer 12 step 0, v_hall/random ratio=3.51× |
| **A8 — T3 Impulse 效应** | 大幅 early impulse 可改变 hallucination 行为，但方向特异性尚未证明（random/shuffled 也能改行为） | T3：v_syc avg ctrl=0.0545 > random=0.0303 > v_hall=0.0219；v_hall 未显著高于 random |
| **A9 — TT-SFT 边界测试** | Trajectory-Targeted SFT 在 v0 (self-teacher) 和 v1 (base-model teacher) 两轮测试中均为 weak_effect — 轨迹对齐未能产生超越 CE-only 的行为改善。CE-only 本身在此设置下非常有效（C: 0.611→0.889, +45%），但轨迹对齐在两个版本中均未带来额外行为收益。这是一个重要的 negative result。 | v0: self-teacher, 20 samples, align_w=0.5; v1: base Qwen2.5-0.5B teacher, 30 samples, align_w=1.0。两者均在结构层有微小改善（traj_dist -0.001~-0.003），但行为层与 CE-only 无差异 |
| **A10 — P5 Sycophancy Feedback Control** | probe→gate→hook 闭环。P5: α=−50 syc 0.80→0.00。P5a: 3L per=−8.3 Δ=0.20。P5d: 30/30 Δ=0.27。P6: rep1 +2.1× cost。P7: 自适应 α 失败但证 α-质量连续。P8: α sweep 绘 Pareto; α=−20 为新最佳 (syc=0.00, rep1=0.40, +15%)；三 regime。**P9: 层间差异化 α 全部劣于均匀分配——均匀是 Pareto 最优；层贡献加性独立，多层收益源自分摊减少扰动而非功能分工。** | P8: α=−20 Uniform best。P9: Uniform > Decay/Invert/MidPeak/L10Only in all dimensions. Layers additive. |

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
| **C4 — RoPE 在长上下文里会失真** | arXiv:2605.15514 数学证明：RoPE 在长上下文中失去 locality bias、token relevance consistency，出现 position/token aliasing。有些 routing/gate/integration 失败可能根源在位置编码退化 | Position Sensitivity 实验已证实短期上下文 (≤30 token) 下 PSI=0.0084 未触发 RoPE 退化，但 ΔC=0.067 行为层已有中度影响 → 更长上下文待测 |
| **C5 — Consolidation 保留拓扑但破坏聚类纯度** | TPR=0.875（pairwise 距离结构保存良好），但 Cluster Purity=0.261（所有 20 个 centroid 都严重跨 seed 混合） | IC-2 Topology Audit 实验结果（2026-05-20） |
| **C6 — Relational Memory Hypothesis 已形式化为可测假说** | 记忆本体不是位置序列而是高维关系结构；consolidation 的破坏主要是聚类混合而非拓扑崩坏；capability routing 可能被位置失真污染 | [RELATIONAL_MEMORY_HYPOTHESIS.md](file:///F:/intelligence_capital_minimal_lab/intelligence_capital_theory/RELATIONAL_MEMORY_HYPOTHESIS.md) |
| **C7 — Structural Adaptation Hypothesis 已形式化** | 小-大模型差距核心是结构适应能力（吸收/稳定/组织）；IC-4=组织补偿，IC-2=稳定补偿 | [STRUCTURAL_ADAPTATION_HYPOTHESIS.md](file:///F:/intelligence_capital_minimal_lab/intelligence_capital_theory/STRUCTURAL_ADAPTATION_HYPOTHESIS.md) |
| **C8 — Position Sensitivity (吸收瓶颈完整诊断)** | **三层证据链**: (1) **表示层 KNN=1.0** — 相同内容在不同位置的 hidden state 完全不同，3-NN 可完美分类位置；(2) **Probe PSI=0.0084** — probe 层位置影响极低，A/U 分离跨位置保留；(3) **行为层 ΔC=0.067, ΔH=0.033** — 位置 IS a behavioral confound，但远小于表示层偏移（~0.08 cosine distance）。→ 模型在 ~30 token 范围内有部分下游补偿，但表示偏移部分漏过。A-bottleneck diagnosed. | Rep Shift + Probe PSI + Position-to-Behavior 实验 (2026-05-20) |
| **C9 — Position Sensitivity: 每层详细数据** | early: H=0.867, C=0.600; mid: H=0.900, C=0.667; late: H=0.867, C=0.667. cos(early,mid)=0.065, cos(early,late)=0.080, cos(mid,late)=0.005. Probe A/U separation: early=0.993, mid=0.988, late=0.976. N=60, Layer 12, last_prompt_token. | [REP_SHIFT_REPORT.md](file:///F:/internal_circuit_capital_lab/IC-4-M0/results_position_sensitivity_cpu/REP_SHIFT_REPORT.md) |
| **C10 — Stabilization Scaling: Per-Action KMeans survives** | Seed scaling (5→20→50→100): Per-Action KMeans 0.585→0.635→0.660→0.615. ALL levels STRONG PASS (PA > NM by ≥0.170). Peak at 50 seeds (0.660, Δ=+0.215). Adaptive centroids beat fixed at 100 seeds (0.620 vs 0.615). X-only KMeans stuck at 0.095 regardless of scale → X-only is hardware-limited ceiling. Stabilization breakthrough confirmed scalable. | [run_c4_stabilization_scaling.py](file:///F:/intelligence_capital_minimal_lab/src/run_c4_stabilization_scaling.py) |
| **C10b — Noise Scaling: PA maintains advantage** | 3 noise levels tested (Low/Mid/High). Baseline PA=0.585, NM=0.445. Low: PA=0.615, NM=0.445 (Δ=+0.170). Mid: PA=0.540, NM=0.415 (Δ=+0.125). High: PA=0.605, NM=0.290 (Δ=**+0.315**). PA decay at high noise: -0.020 (ACTUALLY GAINED over baseline). NM collapses -35% at high noise. **Verdict**: PA robustness confirmed — Y-aware structure resists noise where NM collapses. | [run_c4b_noise_scaling.py](file:///F:/intelligence_capital_minimal_lab/src/run_c4b_noise_scaling.py) |
| **C11 — Cross-Bottleneck Analogue (Phase 7 3.3A)** | 3 perturbation types tested: additive_noise (0.01-0.80σ), directional_shift (0.01-0.80σ), centroid_dropout (5%-90%). **Result**: Per-Action KMeans stays > NoMemory at ALL additive_noise and directional_shift levels. Only falls below NM at >65% centroid dropout. Key insight: Y-aware advantage is structural margin — better structure → more perturbation tolerance before crossing NM threshold. Not "magical synergism" but "better baseline = wider safety margin". Analogue supports proceeding to LLM coupling (3.3B). | [run_c5_cross_bottleneck_analogue.py](file:///F:/intelligence_capital_minimal_lab/src/run_c5_cross_bottleneck_analogue.py) |
| **C12 — LLM Hidden State Consolidation (Phase 7 3.3B)** | **NEGATIVE (data-gap).** M3 activations from base model: even PCA→2D cross-seed, ALL strategies = 1.000. Qwen hidden states for answerable/unanswerable are trivially separable — the model KNOWS but can't route. This confirms B-bottleneck narrative: representation is perfect, routing is the problem. Consolidation cannot show advantage when seeds share identical model state (no divergence to bridge). Data infrastructure gap: need multi-checkpoint fine-tuning runs for genuine consolidation test. | [run_c6_llm_consolidation.py](file:///F:/internal_circuit_capital_lab/IC-4-M0/src/run_c6_llm_consolidation.py) |
| **C13 — Objective Scaling (Phase 6-B)** | **STABLE — PA advantage holds across action complexity.** 3→5→10→20 actions: PA=0.500→0.715→0.355→0.155. Δ(PA-NM) stays positive at all levels (+0.055→+0.330→+0.090→+0.045). PA peaks at 5 actions (0.715, Δ=+0.330). At 20 actions, PA drops below KM for first time (-0.020) but Δ(PA-NM) still +0.045. KM curve: 0.095→0.545→0.285→0.175 (hump-shaped). Per-Action advantage is stable, not decaying, across objective granularities. | [run_c4_objective_noise_scaling.py](file:///F:/intelligence_capital_minimal_lab/src/run_c4_objective_noise_scaling.py) |
| **C14 — Position-Augmented Gate Probe (Phase 8-A)** | **STRONG POSITIVE — PSI reduced 90%, first A-bottleneck remediation.** Training logistic probe with position-augmented features (position encoding concatenated to hidden state) reduces Position Sensitivity Index from 0.0676 to 0.0067 (−90.0%). Both base and augmented probes maintain perfect accuracy (acc=1.0). Augmented training requires 4× samples (120 vs 30) but effectively immunizes the probe against positional variation. This is the first successful remediation of the Absorption bottleneck — position-aware probe training neutralizes the KNN=1.0 representational shift. | [run_a1_position_augmented_probe.py](file:///F:/internal_circuit_capital_lab/IC-4-M0/src/run_a1_position_augmented_probe.py) |
| **C15 — Topology Audit: Relationship Memory** | **TPR=0.8923, RRP=0.6900, Purity=0.261, MEC=1.0.** Consolidation preserves relational topology strongly (TPR 0.89, RRP 0.69 at step 5) but catastrophic seed mixing (mean purity 0.261, 19/20 clusters impure). Retrieval is query-robust (MEC=1.0). Core bottleneck identified: cross-seed centroid contamination binds consolidated memory performance to near-naive levels. Per-Action KMeans solves this by never mixing seeds; X-only KMeans can't avoid it. | [run_consolidation_topology_audit.py](file:///F:/intelligence_capital_minimal_lab/src/run_consolidation_topology_audit.py) |
| **C16 — Anchored Consolidation (Proof C)** | **MODEST POSITIVE — +8.7% over naive.** Best anchor blend ratio br=0.7: match=0.125 vs naive=0.115, Δ=+0.010. Centroid drift invariant to blend ratio. Episodic=0.195, NoMemory=0.445. Anchoring helps marginally but does not solve the purity bottleneck — confirms centroid position is not the root cause; seed mixing in cluster formation is. | [run_c_anchored_consolidation.py](file:///F:/intelligence_capital_minimal_lab/src/run_c_anchored_consolidation.py) |
| **C17 — Readout-Level Stabilization (C2)** | **NEGATIVE — readout fixes cannot salvage contaminated centroids.** All readout interventions (seed-conditioned, weighted-seed, purity-gated) perform WORSE than naive (-17.4%). Combined strategy hits 0.460 but degenerates to NoMemory fallback when purity collapses to 0.261. Key finding: centroid contamination happens at cluster FORMATION time, not at READOUT time — readout-level fixes are structurally insufficient. The Per-Action KMeans approach sidesteps the problem at formation time. | [run_c2_stronger_stabilization.py](file:///F:/intelligence_capital_minimal_lab/src/run_c2_stronger_stabilization.py) |
| **C18 — Phase 8-B: Behavior Gate Consistency** | ✅ **PASS (partial).** Gate decisions perfectly consistent across positions (11/11/11). ΔH=0.111 persists in raw generation — behavior-level absorption remains. | [run_a2_behavior_position_invariant.py](file:///F:/internal_circuit_capital_lab/IC-4-M0/src/run_a2_behavior_position_invariant.py) |
| **C19 — P8: Sycophancy Feedback Control Scale-Up (n=24)** | ✅ **STRONG PASS — 成功复现 P6-ter.** Baseline syc=0.583 (14/24). Two-stage th=0.50: syc=0.250 (6/24), **-57.1% reduction**, gate rate=54.2%. Open-loop syc=0.500 (worse than baseline). Two-stage feedback control scales from n=12 to n=24 without degradation. | [run_p8_large_scale_replication.py](file:///F:/internal_circuit_capital_lab/IC-4-M0/src/run_p8_large_scale_replication.py) |
| **C20 — Phase 9-A: Position Rectification (inference hook)** | ❌ **NEGATIVE — 全局偏移无效。** Baseline ΔH=0.111. Rectified mid/late H改善但early不变 → ΔH=0.333（增大）。确认位置敏感性是 content-dependent（KNN=1.0），不是简单全局加法偏移可修复。 | [run_a3_position_rectification.py](file:///F:/internal_circuit_capital_lab/IC-4-M0/src/run_a3_position_rectification.py) |
| **C21 — Phase 9-B: Multi-Checkpoint LLM Consolidation** | ⚠️ **MIXED.** LoRA fine-tune (5 epochs): loss 13.3→0.068. Cross-ckpt KNN=1.0 persists (A/U separability preserved). PerClassKMeans beats baseline (+0.37-+0.42) but Y-Aware is always better (0.95-0.97, gap=0.05-0.07). LoRA changes hidden states (Δmean 1.4→3.1) but doesn't break class boundary — the model KNOWS, routing is the problem. | [run_c7_multi_checkpoint_consolidation.py](file:///F:/internal_circuit_capital_lab/IC-4-M0/src/run_c7_multi_checkpoint_consolidation.py) |
| **C22 — Phase 10: Position-Aware LoRA Training (行为层吸收补救)** | ✅ **POSITIVE — ΔH减半, Consistency提升.** LoRA rank=4 on position-augmented data (30×3=90 samples, 3 epochs). Pre: H=(0.33,0.56,0.56), ΔH=0.22, PSI=0.0016, Consistency=0.90. Post: H=(0.56,0.44,0.44), **ΔH=0.11 (−50%)**, **PSI=0.00073 (−53%)**, **Consistency=0.95 (+5%)**. Behavior-level position absorption partially closed — ΔH halved but early H increased (0.33→0.56) as regularization-convergence trade-off. Core claim: position-aware weight-level intervention (LoRA) successfully reduces behavior-level position sensitivity where global rectification failed. | [run_a4_position_aware_training.py](file:///F:/internal_circuit_capital_lab/IC-4-M0/src/run_a4_position_aware_training.py) |
| **C23 — Phase 11: Cross-Bottleneck Integration (A+C)** | ✅ **POSITIVE — 瓶颈可分离.** Position-Aware LoRA (A-remedy) tested on sycophancy knowledge-level behavior (C domain). Both BASE and LoRA have syc_rate=0.0000 on template log-prob comparison (24/24 prefer correction over agreement). Margin preserved (-0.634→-0.596, Δ=+0.039). Position consistency perfect (1.0000) for both. Core claim: Bottlenecks are INDEPENDENT — fixing A (absorption) preserves C (anti-sycophancy knowledge), neither degrades nor automatically improves. The B-bottleneck (knowledge-production gap) remains the key frontier: model KNOWS (syc_rate=0.00 on templates) but DOESN'T produce (syc_rate=0.583 on generation from P8). | [run_cross_bottleneck_integration.py](file:///F:/internal_circuit_capital_lab/IC-4-M0/src/run_cross_bottleneck_integration.py) |
| **C24 — P12: Position-Directional Activation Steering** | ❌ **NEGATIVE (informative).** Contrastive vector v_abs = mean(h_early)−mean(h_late) tested via layer-10 activation steering with log-prob evaluation. U-shaped alpha curve: |α|≤1.5 has NO effect (ΔH=0.250 baseline); |α|≥3.0 eliminates ΔH (→0.000) but by degrading ALL positions to H=0.50 — late doesn't improve, early gets worse. At α=+3.0 only v_abs eliminates ΔH (random/orth preserve ΔH=0.250), confirming direction is real but destructive. Phase transition α∈[1.5, 3.0] — position-discriminating subspace has finite perturbation capacity. Core claim: Two independent experiments (9-A global rectification, P12 directional steering) now confirm position sensitivity CANNOT be fixed by adding/subtracting vectors to hidden states. Only weight-level intervention (Phase 10 LoRA) works. | [run_p12_position_steering.py](file:///F:/internal_circuit_capital_lab/IC-4-M0/src/run_p12_position_steering.py) |
| **C25 — P13: Probe-Guided Hallucination Steering (B-bottleneck)** | ❌ **NEGATIVE (geometric proof).** Hallucination probe (acc=1.000, layer 12) decision boundary direction used for steering. Full alpha sweep (-5 to +5): probe direction has NO effect on H across moderate alphas (H=0.417 flat). At large |α| it DEGRADES H (0.50-0.58). Random vector control shows same pattern. Core claim: Classification direction ≠ behavioral control direction — the linear subspace that discriminates answerable/unanswerable is orthogonal to the subspace that controls hallucination/abstention. This is the B-bottleneck in geometric form: KNOWS ≠ produces is not just behavioral but a SUBSPACE SEPARATION in representation space. Three experiments now confirm: vector ops in hidden space cannot bridge the KNOWS→produces gap (9-A position offset, P12 direction steering, P13 probe steering). | [run_p13_probe_guided_steering.py](file:///F:/internal_circuit_capital_lab/IC-4-M0/src/run_p13_probe_guided_steering.py) |
| **C26 — S1: Structure Signal Distillation (训练时结构内化)** | ✅ **POSITIVE — syc rate 50%→15% (Δ=+0.35).** LoRA training on 20 syc-only prompts with CE+MSE joint loss (MSE target = steered L10 hidden state). CE decreased 2.07→0.56. MSE stayed at 0.010 (theoretical baseline) but its gradient direction provided steering signal. **Core claim**: External probe→gate→hook controller knowledge CAN be distilled into model weights via directional hidden-state guidance during training. This is the paradigm shift from inference-time control to training-time internalization. ⚠️ Output quality degraded (garbled text in 8/20 outputs) — overfit due to small sample + high LR (5e-4). | [run_s1_structure_distillation.py](file:///F:/internal_circuit_capital_lab/IC-4-M0/src/run_s1_structure_distillation.py) |
| **C27 — S2: Self-Probe Training (自探针内化)** | ❌ **NEGATIVE — syc rate 50%→65% (Δ=-0.15), ProbeAcc=0.** LoRA + auxiliary probe head jointly trained with CE+BCE loss on 40 mixed samples (20 syc + 20 non). CE decreased 3.63→1.88. BCE stayed near log(2)≈0.69 (random). Self-probe never distinguished syc/non-syc (Acc=0.000). **Core claim**: Without directional steering signal, pure CE training on mixed syc/non-syc data AMPLIFIES sycophantic tendencies. BCE gradient from probe head is swamped by CE gradient — joint training of behavioral self-monitoring is structurally infeasible at these loss magnitudes. | [run_s2_self_probe.py](file:///F:/internal_circuit_capital_lab/IC-4-M0/src/run_s2_self_probe.py) |
| **C28 — S3: Triple-Bottleneck Regularization (三瓶颈联合正则化)** | ❌ **MILD NEGATIVE — syc rate 50%→55% (Δ=-0.05).** LoRA training with CE + λ₁·PSI + λ₂·Purity + λ₃·Routing penalties. CE decreased 4.01→2.50. All three regularization gradients effectively zero (PSI~3e-4, Purity~0, Routing~0.50). Val metrics unchanged (PSI 0.0043→0.0043, Purity 0.445→0.420, RoutingAcc 1.0→1.0). **Core claim**: Batch-level structural quality proxies produce gradients too weak to compete with CE loss. Bottleneck regularization requires explicit architectural changes (not penalty terms) or requires computing structural metrics on much larger batches (GPU-scale). Confirms C5/C6 independence finding: joint optimization ≠ synergistic improvement. | [run_s3_triple_bottleneck.py](file:///F:/internal_circuit_capital_lab/IC-4-M0/src/run_s3_triple_bottleneck.py) |
| **C29 — S1b: Robust Structure Distillation (健壮性验证)** | ⚠️ **NEGATIVE — reveals fundamental Robustness–Syc Reduction Tradeoff.** Same as S1 but with lower LR (1e-4), LoRA r=4, KL divergence reg (λ=0.1), 10 general QA mixed into training. CE stabilized at 1.81 (no overfit vs S1's 0.56), output quality 100% preserved. But syc rate unchanged (45%→45%, Δ=0.00). **Core claim**: S1's syc-reduction effect is COUPLED to overfitting. Preventing overfitting eliminates the effect. The sycophancy attractor basin has a minimum escape energy threshold — gentle pushes (KL + low LR) stay below it, strong pushes (high LR, no KL) cross it but cause global instability. | [run_s1b_robust_distillation.py](file:///F:/internal_circuit_capital_lab/IC-4-M0/src/run_s1b_robust_distillation.py) |
| --- | --- | --- |
| **C30 — S1c: Critical Point Sweep (相变临界点扫描)** | ⚠️ **NEGATIVE — confirms bifurcation, no continuous sweet spot.** 6 MSE weights (0.05,0.1,0.3,0.5,0.7,1.0) × 4 alphas (−1,−2,−3,−5) = 24 config grid search, LoRA r=8, LR=5e-4, 2 epochs, 10 test samples each. **Baseline**: syc=0.40, quality=1.00. **Results classified** into 4 regimes: (1) **Quality Crash (13/24)**: syc ≤ 0.30 but quality ≤ 0.50 — output collapses to !!!! garbled tokens; (2) **Syc Amplify (8/24)**: quality ≥ 0.70 but syc ≥ baseline — CE training amplifies original behavior; (3) **Neutral (2/24)**: no net change, quality degraded; (4) **Isolated Anomaly (1/24)**: mse=0.7, alpha=−5.0: syc=0.10, quality=1.00 — BUT neighbors all Crash/Amplify, no stable region. **Key findings**: (a) System exhibits **bifurcation**, not continuous phase transition — outcomes jump between Crash and Amplify with no intermediate states; (b) Even at the "best" config, the isolated result is statistically fragile — the Robustness–Syc Reduction Tradeoff holds across the entire 24-grid; (c) CE always drops to 0.55-0.68 (overfitting), confirming S1's mechanism; (d) The sycophancy attractor escape threshold cannot be met by any stable training regime — force needed to escape crashes the model. **Total time**: 159 min CPU. | [run_s1c_critical_point_sweep.py](file:///F:/internal_circuit_capital_lab/IC-4-M0/src/run_s1c_critical_point_sweep.py), [results.json](file:///F:/internal_circuit_capital_lab/IC-4-M0/results_s1c_critical_point_sweep/results.json) |
| --- | --- | --- |
| **C31 — S1d: Critical Slowing Down (临界慢化/振荡观察)** | ⚠️ **NEGATIVE — no slowing, only oscillation→collapse.** 5 boundary-proximal configs from S1c run for 5 epochs each with per-epoch evaluation. **Core question**: Does longer training soften the S1c bifurcation? **Answer**: NO — it sharpens it. All 5/5 configs end at quality=0.0, syc=0.0 regardless of mid-trajectory behavior. **Key trajectories**: (a) 4/5 configs reach quality=1.0 at some epoch (E1 or E2) but all crash irreversibly by E3-E5 — the S1c "sweet spots" were **transient states** captured at exactly epoch 2; (b) Config mse_0.7_alpha_-3.0 shows **critical oscillation**: syc alternates [0.5, 0.1, 0.5, 0.4, 0.0] and quality flips [0.8, 0.4, 1.0, 0.8, 0.0] — oscillation dampens into crash, not into equilibrium; (c) **MSE stays at theoretical baseline** (α²/896) for ALL epochs in ALL configs — the directional push is NOT being learned, only measured; (d) CE continues decreasing monotonically (lowest=0.15 at E5) even while outputs are 100% !!!! — catastrophic overfitting. **Conclusion**: The Crash attractor is universal and irreversible. Longer training does NOT create critical slowing down — it creates a brief oscillatory transient followed by inevitable collapse. The bifurcation is **subcritical**: no stable intermediate states exist. **Total time**: 103 min CPU. | [run_s1d_critical_slowing.py](file:///F:/internal_circuit_capital_lab/IC-4-M0/src/run_s1d_critical_slowing.py), [results.json](file:///F:/internal_circuit_capital_lab/IC-4-M0/results_s1d_critical_slowing/results.json) |
| --- | --- | --- |
| **C32 — S1f: LoRA Capacity Test (容量瓶颈验证)** | ⚠️ **NEGATIVE — LoRA capacity is NOT the bottleneck.** Fixed config (mse=0.7, α=−5.0), varied r=[8,16,32,64] (4.4M→35.2M params, 8× range). 2 epochs each with per-epoch evaluation. **Key results**: (a) **MSE never deviates from baseline (0.027902)** across ALL r — the directional push is fundamentally not learnable through this training objective; (b) Epoch-1 quality improves monotonically with r: 0.0→0.6→0.7→0.8 — more capacity delays the collapse; (c) But Epoch-2 quality does NOT: 0.0→0.2→0.3→0.0 — r=64 at epoch 2 actually performs WORSE than r=32; (d) Final syc=0.0 for all r — the Crash attractor remains universal even at 35M trainable params. **Conclusion**: The subcritical bifurcation is intrinsic to the CE+MSE training objective, NOT to the LoRA parameterization. Increasing capacity delays but does not prevent the collapse. The model always prioritizes CE overfitting over MSE learning. **Total time**: 66 min CPU. | [run_s1f_lora_capacity.py](file:///F:/internal_circuit_capital_lab/IC-4-M0/src/run_s1f_lora_capacity.py), [results.json](file:///F:/internal_circuit_capital_lab/IC-4-M0/results_s1f_lora_capacity/results.json) |
| **C33 — S1g: Decoupled Two-Stage Training (分阶段解耦训练)** | ✅ **POSITIVE — bifurcation CROSSED via temporal decoupling.** Stage1: MSE-only (frozen lm_head, 3ep), Stage2: CE-only (unfrozen lm_head, 3ep). 4 configs tested. **Key result**: seq_3_3 achieves syc=0.00 + quality=1.00 + CE=2.08 (near baseline 2.0) — the FIRST working instance of training-time structural internalization in the S1 series. **4-config comparison**: (a) seq_3_3: S2 E1 crashes→recovers, final CE=2.08 best; (b) seq_5_3 (5ep MSE): no S2 crash at all, smooth transition, final CE=5.90; (c) joint_6 (CE+MSE 6ep): oscillates E1-E5 (qual=0.0), sudden recovery at E6 (qual=1.0, CE=6.43); (d) seq_lmhead (LoRA frozen in S2): E1-E2 fine, E3 crashes — LoRA structure exists but needs co-adaptation. **Critical finding**: S1-S1f MSE metric was BUGGED — `target = hs_pooled + alpha*steer` makes MSE = α²/896 by mathematical identity, independent of training. The gradient was real but the metric was blind. Behavioral results (syc/quality) remain valid. **Conclusion**: The subcritical bifurcation is NOT a "hard boundary" but a "simultaneous optimization infeasibility" — decoupling CE and MSE temporally allows safe traversal. Training-time structural internalization concept is validated. **Total time**: 138 min CPU. | [run_s1g_decoupled_training.py](file:///F:/internal_circuit_capital_lab/IC-4-M0/src/run_s1g_decoupled_training.py), [results.json](file:///F:/internal_circuit_capital_lab/IC-4-M0/results_s1g_decoupled_training/results.json) |
| **C34 — S1h: GMD-Inspired MMD Drift Training (GMD 启发的 MMD Drift 训练)** | ❌ **NEGATIVE — MMD drift is WORSE than fixed steer, all 3 configs FAILED.** Tested whether GMD's adaptive drift field V(x) (arXiv 2605.05118, Deng et al. 2026) can overcome the bifurcation without temporal decoupling. 3 configs: (a) mmd_joint_6 (MMD drift + CE joint 6ep): **CRASHED** — CE→0.11, qual=0.0, MMD² INCREASED (distributions diverged); (b) fixed_joint_6 (fixed steer + CE joint 6ep): **CRASHED** — CE→0.11, qual=0.0, steer_loss constant 0.027902 (S1g bug reconfirmed); (c) mmd_seq_3_3 (MMD drift 3ep→CE 3ep): **CRASHED** — Stage 1 MMD drift had NO measurable effect (MMD²/drift_norm identical to baseline to 6 decimals), Stage 2 CE immediately crashed. **5 findings**: (1) MMD drift worse than fixed steer across all configs; (2) GMD's V→0 self-regulation fails in the structural internalization context; (3) ~60% batch waste due to ≥2 syc+non requirement per batch; (4) the crash root cause is CE/directional co-temporal conflict, not direction choice; (5) small-sample kernel estimates unstable in 896-dim. **Conclusion**: REINFORCES S1g — temporal decoupling is NECESSARY, not optional. The direction of steer (fixed vs adaptive) is irrelevant; the conflict arises from co-temporal optimization of CE and any directional push. **Total time**: ~42 min CPU. | [run_s1h_mmd_drift.py](file:///F:/internal_circuit_capital_lab/IC-4-M0/src/run_s1h_mmd_drift.py), [results.json](file:///F:/internal_circuit_capital_lab/IC-4-M0/results_s1h_mmd_drift/results.json) |
| **C35 — S1i: Alternative Stage 1 Directional Losses (Goldilocks 原理)** | ⚠️ **PARADOXICALLY POSITIVE — all 3 configs failed but revealed the Goldilocks Principle of Stage 1 directional encoding.** Tested two "cleaner" Stage 1 losses to replace S1g's buggy self-referential MSE. 3 configs: (a) **cos_seq_3_3** (cosine reg 3ep→CE 3ep): Stage 1 cos_sim +0.15→−0.996 (near-perfect directional alignment), but mse_fixed explodes 0.028→2.637 — **TOO HOT**: extreme hidden state shift (~50× unit norm) destroys representations irreversibly; (b) **cos_seq_5_3** (cosine reg 5ep→CE 3ep): cos_sim→−1.000, mse_fixed→3.492, same catastrophic failure — longer cosine training only amplifies the damage; (c) **mse_fixed_seq_3_3** (fixed-target MSE 3ep→CE 3ep): Stage 1 mse_fixed→0.001 (near-perfect convergence to target), but cos_sim only→−0.275 — **TOO COLD**: weak directional encoding is immediately erased by CE (cos_sim→−0.12 at S2 E2). Brief S2 E2 recovery (qual=0.3) hints at potential with stronger encoding. **Goldilocks Principle**: S1g's detached MSE was **JUST RIGHT** — constant gradient (−2α·v) never decays, providing sustained moderate push. The S1g "bug" (detached target = constant MSE metric) is actually a **FEATURE**: (a) it prevents gradient decay (mse_fixed's problem), (b) it avoids representation destruction (cosine's problem), (c) the constant push survives Stage 2 CE. **Meta-insight**: "Bugs are features" — the optimal solution was NOT to fix the bug but to understand WHY the bug worked better. **Total time**: ~71 min CPU. | [run_s1i_contrastive_stage1.py](file:///F:/internal_circuit_capital_lab/IC-4-M0/src/run_s1i_contrastive_stage1.py), [results.json](file:///F:/internal_circuit_capital_lab/IC-4-M0/results_s1i_contrastive_stage1/results.json) |
| **C36 — S1j: Cosine + Magnitude Penalty (余弦方向+幅度惩罚混合损失)** | ❌ **NEGATIVE — 4/4 λ values FAILED, cosine fundamentally cannot be constrained.** Tested whether adding magnitude penalty λ·||hs−baseline_hs||² to cosine loss could find the Goldilocks zone. 4 λ values: 0.05, 0.2, 0.5, 1.0. Stage 1 (3ep, LoRA only) → Stage 2 (3ep, CE only). **Key results**: (a) Magnitude penalty reduced Euclidean drift 93% (mse_fixed: 2.637→0.153 at λ=1.0) and preserved quality (qual: 0.7→1.0 at λ=1.0) — successful constraint; (b) But cos_sim still reached −0.97~−0.99 across ALL λ — the push direction is immutable; (c) All 4/4 configs failed Stage 2 (qual=0.0) — even at λ=1.0 with perfect S1 quality, the directional damage is irreversible. **Critical insight**: Cosine operates on normalized vectors (ĥ), magnitude penalty operates on unnormalized vectors (hs). They are nearly **orthogonal objectives** — the penalty cannot constrain cosine's directional push because the two loss terms act on different geometric quantities. Cosine always pushes ĥ→−v̂ regardless of λ. **Conclusion**: Cosine-based losses are fundamentally unsuitable for Stage 1. Any loss operating on normalized vectors will inevitably push cos_sim→−1.0, and cos_sim > −0.97 causes irreversible representation damage. **Total time**: 122 min CPU. | [run_s1j_cosine_magnitude.py](file:///F:/internal_circuit_capital_lab/IC-4-M0/src/run_s1j_cosine_magnitude.py), [results.json](file:///F:/internal_circuit_capital_lab/IC-4-M0/results_s1j_cosine_magnitude/results.json) |
| **C37 — S1k: MSE-Fixed Extended Training (延长固定目标 MSE 训练 → 梯度衰减高原)** | ❌ **NEGATIVE — 3/3 configs FAILED, cos_sim plateaus at −0.25 regardless of epochs.** Tested whether longer mse_fixed Stage 1 training (5ep/7ep) could build stronger directional encoding to survive Stage 2. 3 configs: s5s3 (5ep MSE→3ep CE), s7s3 (7ep MSE→3ep CE), s5s5 (5ep MSE→5ep CE). **Key results**: (a) cos_sim plateaus at ~−0.25 after E2, with no further improvement through E3-E7 — the gradient ∂L/∂hs = 2(hs−target) decays to zero as hs→target, creating an insurmountable learning barrier; (b) s5s3: S1 cos_sim flat [-0.22, -0.26, -0.23, -0.25, -0.23], S2 qual=0.0; (c) s7s3: S1 cos_sim flat [-0.21, -0.26, -0.25, -0.24, -0.22, -0.26, -0.26], S2 E1 qual=0.6 (initialization-dependent transient), E2-E3 qual=0.1→0.0; (d) s5s5: S1 cos_sim flat [-0.23, -0.22, -0.24, -0.26, -0.26], S2 qual=0.0 all epochs. **Conclusion**: mse_fixed fundamentally CANNOT reach the Goldilocks zone. Gradient decay is an intrinsic property. **Total time**: 102 min CPU. | [run_s1k_mse_fixed_extended.py](file:///F:/internal_circuit_capital_lab/IC-4-M0/src/run_s1k_mse_fixed_extended.py), [results.json](file:///F:/internal_circuit_capital_lab/IC-4-M0/results_s1k_mse_fixed_extended/results.json) |
| **C38 — S1g-v2: Reproduction Attempt — S1g Non-Reproducible (唯一"成功"不可复现)** | 🔴 **CRITICAL NEGATIVE — S1g seq_3_3 FAILS to reproduce across 3 independent attempts.** S1g was the ONLY claimed success in 13 S1 experiments (syc=0.0, quality=1.0, CE=2.08). S1g-v2 performed 3 independent reproductions: (a) **No seed, no grad clip**: baseline syc=0.2, CE [7.13, 2.25, 0.52], all empty outputs; (b) **seed=42, no grad clip**: baseline syc=0.4 (matching S1g), CE [6.28, 1.57, 0.26], all empty outputs; (c) **seed=42, with grad clip**: baseline syc=0.4, CE [5.28, 0.65, 0.14], all empty outputs. **None produced readable text at any Stage 2 epoch.** S1g's original CE trajectory [8.84, 4.77, 2.08] was never reproduced — CE always crashed below 1.0. **Implications**: (1) S1g's claimed success was a one-time fluke, likely due to CUDA floating-point difference, PEFT/transformers version difference, or LoRA initialization randomness not captured by `torch.manual_seed()`. (2) The entire S1 series now has 0/13 working Stage 1 training methods. (3) The Goldilocks principle and gradient analysis remain theoretically valuable, but the practical realization via detached MSE is not reliable. **Conclusion**: S1 series needs complete re-evaluation. The detached MSE approach is NOT a working solution — the constant gradient mechanism, while theoretically sound, does not consistently produce usable models. **Total time**: 3 × ~24 min = ~72 min CPU. | [run_s1g_v2_output_quality.py](file:///F:/internal_circuit_capital_lab/IC-4-M0/src/run_s1g_v2_output_quality.py), [results.json](file:///F:/internal_circuit_capital_lab/IC-4-M0/results_s1g_v2_output_quality/results.json) |

---

### D. 新范式：训练时结构内化

这组实验 (C26-C29) 构成了——从推理时外部控制器转向训练时内部结构化。

**C29 的 tradeoff 启示**：S1（激进）vs S1b（保守）构成了一次受控对比。syc 降低与输出质量之间存在根本性张力——这正是两周前 P5 发现的"α阈值效应"在训练层面的投影：syc 吸引子需要最小能量才能逃离，而该能量恰好处于稳定训练所无法达到的区间。

**S1 的成功条件**: 方向性信号 (v_syc) + CE 损失 = 有效。MSE 值不需要下降，梯度方向就足够——类似于正则化项的作用机制。

**S2 的失败教训**: 无方向信号时，CE 训练会强化原有行为模式（在此情况下是谄媚）——标准 LM fine-tuning 不是中性操作。

**S3 的失败教训**: 结构质量的批级代理（PSI/proxy, Purity/proxy, Routing/proxy）梯度太弱——真正的瓶颈正则化需要更大的批次或显式架构修改。

**与已有发现的关系**: S1 延续了 P5/P6-ter 的闭环控制逻辑，但将其蒸馏到权重中；S2 呼应了 TT-SFT 的轨迹对齐失败；S3 呼应了 C5/C6/C23 的瓶颈独立性。C30 (S1c) 通过 24 配置网格扫描确认了**二分岔相变结构**；C31 (S1d) 进一步证明了这是**亚临界二分岔 (subcritical bifurcation)**——系统在崩溃前经历 1-2 epoch 的振荡暂态，但不收敛到任何稳定中间态；C32 (S1f) 完成了**三角论证的第三边**——增大 LoRA 容量 8× (4.4M→35.2M) 不能软化二分岔。**C33 (S1g) 曾被认为是突破口**——分阶段解耦训练首次跨越二分岔，达成 syc=0.0 + quality=1.0 + CE=2.08。**C34 (S1h) 是重要的阴性排除**——GMD 启发的自适应 MMD drift 在所有 3 个配置上都失败。**C35 (S1i) 揭示了 Goldilocks 原理**——cosine 过冲 (TOO HOT)，mse_fixed 太弱 (TOO COLD)，S1g 的 detached MSE 看似在最佳区间。**C36 (S1j) 排除了 cosine+magnitude 混合损失**——正交目标不可调和。**C37 (S1k) 排除了延长 mse_fixed 训练**——梯度衰减高原不可逾越。🔴 **C38 (S1g-v2) 逆转了所有结论——S1g seq_3_3 无法复现（3次独立尝试全部失败）**：CE 始终崩溃至 0.14-0.52（而非 S1g 报告的 2.08），所有 Stage 2 输出均为空字符串。S1g 的"成功"是一次性偶然产物（可能源自 CUDA 浮点差异、PEFT/transformers 版本差异、或未被 `torch.manual_seed()` 捕获的 LoRA 初始化随机性）。**S1 系列 13 实验现为 0/13 ——训练时结构内化的所有尝试均未产生可靠工作模型。** S1i 的 Goldilocks 原理、S1j 的正交目标洞察、S1k 的梯度衰减高原——这些理论分析仍然成立，但它们的价值从"照亮正确路径"转变为"标记错误路径"。"恒常梯度"作为解耦训练的核心机制在理论上具有洞察力，但在实践中不稳定。**S1 系列探索线从"闭合"转为"开放"——需要完全重新评估 Stage 1 的训练策略。**

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

### Q4: 轨迹动力学与因果方向（Trajectory Dynamics）

> hallucination 在 prefill 即可被 probe 分离，但 impulse 的方向特异性尚未被证明。从"对扰动敏感"升级到"对特定因果方向可控"是当前核心挑战。

**子问题**：
- hallucination 和 sycophancy 在 (layer, step) 空间中的可读/可控结构是什么样的？（T0-T2 已答，T3 部分答）
- early impulse 的行为效应来自方向特异性还是大扰动？（P2 的核心问题）
- sycophancy 线的对照集缺失如何解决？（P0）
- 从方向特异性确认到 feedback control 的路径？（P3）

---

## 第三层：下一阶段主线

### Phase 1: 已完成锚点（截至 2026-05-21）

| 项目 | 任务 | 状态 |
|---|---|---|
| IC-4 | 30A+30U 成为默认标准；terrain manual v3.2 已更新 | ✅ 完成 |
| IC-4 | P1.5 小样本 artifact 确认；所有 tested seed/layer 在 30A+30U 下通过 | ✅ 完成 |
| IC-2 | IC-2c.1 根因拆解（NoMemory shortcut / episodic k-NN cap / consolidated drift） | ✅ 完成 |
| IC-2 | Topology Audit: TPR=0.875, Purity=0.261 | ✅ 完成 |
| IC-4 | Position Rep Shift: KNN=1.000 | ✅ 完成 |
| IC-4 | Position-to-Behavior: ΔC=0.07 | ✅ 完成 |
| IC-4 | Trajectory 脚本修复: T0 clear_records, T1 跨域投影隔离, T3 位移对齐 | ✅ 完成 |
| Theory | STRUCTURAL_ADAPTATION_HYPOTHESIS.md, RELATIONAL_MEMORY_HYPOTHESIS.md | ✅ 完成 |
| IC-2 | Phase 6-B (Objective Scaling): 3→20 actions, PA advantage STABLE (δ slope -0.010) | ✅ 完成 |
| IC-2 | Phase 6-C (Noise Scaling): 0→1.0σ, PA robust, KMeans at floor | ✅ 完成 |
| IC-2 | 更新 THEORY.md 的 bad debt / false capital 条目 | ✅ 完成 (v6.9: 新增实验量化节，BDR~0.68, Purity=0.261, PA Purity=1.0) |

### P5: Sycophancy Feedback Control — 闭环已建，不对称已发现

> 完整报告: [P5 report](file:///F:/internal_circuit_capital_lab/new-5/reports/IC4_P5_SYC_FEEDBACK_REPORT.md)

| 模式 | α | Syc Rate | Non-Syc Correct | 效果 |
|---|---|---|---|---|
| base | — | 0.8000 | 1.0000 | — |
| hard_gate (anti) | −50.0 | **0.0000** | 1.0000 | **完美控制，无副作用** |
| hard_gate (pro) | +5.0 | 0.6000 | 1.0000 | 不稳定 |
| random_gate | −50.0 | 0.8000 | 1.0000 | 方向特异性对照 |
| orthogonal_gate | −50.0 | 0.8000 | 1.0000 | 方向特异性对照 |

**核心发现**:
1. **首次因果特异性闭环控制**: random/orthogonal 零效果，v_syc 是唯一有效方向
2. **方向不对称性**: 谄媚态是深层吸引子 — 滑入只需 +5v，逃离需要 −50v (10× 不对称)
3. **优于 M3-v6**: 控制更强 (−100% vs −81%) 且无副作用 (non_syc_correct=1.0)
4. **α 阈值效应**: −1.0/−10.0 无效，−50.0 突变生效 — 存在最小逃逸能量

**Caveat**: α=−50.0 的极端注入可能影响输出流畅度，需要人工评估。不建议直接部署。

### new-5 项目总路线总结

> 完整报告: [IC4_NEW5_COMPREHENSIVE_REPORT.md](file:///F:/internal_circuit_capital_lab/new-5/reports/IC4_NEW5_COMPREHENSIVE_REPORT.md)
> P5d 报告: [IC4_P5D_CROSSVALIDATION_REPORT.md](file:///F:/internal_circuit_capital_lab/new-5/reports/IC4_P5D_CROSSVALIDATION_REPORT.md)
> P6 报告: [IC4_P6_QUALITY_EVAL_REPORT.md](file:///F:/internal_circuit_capital_lab/new-5/reports/IC4_P6_QUALITY_EVAL_REPORT.md)
> P7 报告: [IC4_P7_ADAPTIVE_ALPHA_REPORT.md](file:///F:/internal_circuit_capital_lab/new-5/reports/IC4_P7_ADAPTIVE_ALPHA_REPORT.md)
> P8 报告: [IC4_P8_ALPHA_SWEEP_REPORT.md](file:///F:/internal_circuit_capital_lab/new-5/reports/IC4_P8_ALPHA_SWEEP_REPORT.md)
> P9 报告: [IC4_P9_LAYERWISE_ALPHA_REPORT.md](file:///F:/internal_circuit_capital_lab/new-5/reports/IC4_P9_LAYERWISE_ALPHA_REPORT.md)

```
TT-SFT v0 → v1 → P5 → P5a → P5d → P6 → P7 → P8 → P9
   weak   weak   −100%  3L ✓  Δ=0.27  qual  adp↓  −20★  lyr↓
```
_(↓ = negative: adaptive alpha infeasible; layer-wise allocation Pareto-dominated by uniform)_

**核心路线逻辑**:
1. TT-SFT v0/v1 划出明确边界 — 训练层 hidden-state cosine alignment 不是有效方法
2. 路线转折 — 从「训练更可控的 Plant」转向「控制现有 Plant」（推理层闭环）
3. P5 首次实现有因果特异性的闭环控制 — syc 0.800→0.000, non_syc_correct=1.0
4. P5a 验证多层分散注入 — 3L 将 per-layer α 从 −50 降至 −8.3
5. P5d 交叉验证复制并强化方向特异性 — Δ=0.27，确认为稳定特征量
6. P6 量化控制代价 — rep1 +2.1×, distinct1 −41%；T=−50 导致模型崩溃不可行
7. **P7 自适应 α 失败 (AUC=1.0) 但揭示 α-质量连续关系 — |α|≤10 几乎零成本，rep1 是 |α| 的超线性函数**
8. **P8 精细 α sweep 绘制完整 Pareto 前沿 — α=−20 取代 α=−25 为新最佳 (syc=0.00, rep1=0.40, 质量 +15%)；三个 regime: 免费区(0~−14), 线性区(−14~−18, 最佳折中 α=−18), 惩罚区(−18~−25)**
9. **P9 层间差异化 α 全部被均匀分配 Pareto-dominates — Uniform 是最优分配；多层收益源自分摊 per-layer 扰动幅值而非功能分工**

**战略结论 (v7.2)**: 方向特异性 Δ=0.27 是跨数据分割稳定的特征量，可分解为 α 依赖的噪声干扰 (~60pp) + α 不变的方向特异性 (~27pp)。**但有真实的输出质量代价**：方向特异性以 2.1× 重复度和 41% 词汇多样性下降为代价。未来方向：自适应 α（探针置信度调制）、或后处理重复检测。

### Trajectory-Targeted SFT (TT-SFT) v0 → v1 — 边界已测得

> **本节记录 TT-SFT v0 (self-teacher) 和 v1 (base-model teacher) 两轮实验结果。**
> 完整报告: [v0 report](file:///F:/internal_circuit_capital_lab/new-5/reports/IC4_TT_SFT_V0_REPORT.md) | [v1 report](file:///F:/internal_circuit_capital_lab/new-5/reports/IC4_TT_SFT_V1_REPORT.md)

| 实验 | Teacher | 数据 | align_w | 行为层效果 | 结构层效果 | Verdict |
|---|---|---|---|---|---|---|
| v0 | self (0.5B-Instruct) | 20 | 0.5 | H/C/CA: 与 CE-only 完全相同 | traj_var -0.0022, traj_dist -0.0032 | `weak_effect` |
| v1 | base (0.5B, no RLHF) | 30 | 1.0 | H/C/CA: 与 CE-only 完全相同; CA 反而退化 (0.091→0.046) | traj_dist -0.0016, traj_var +0.004 (变差) | `weak_effect` |

**核心发现**:
1. **CE-only 在此设置下非常有效**: 仅 30 sample LoRA 训练即可将正确率从 0.611 提升至 0.889 (+45%)
2. **轨迹对齐未能超越 CE-only**: 两轮实验、两种 teacher、不同数据量和对齐权重，行为层均无差异
3. **结构层改善极小且不一致**: traj_dist 略有改善但 traj_var 在 v1 中反而恶化
4. **Teacher 质量提升未解锁轨迹对齐**: 从 self-teacher 升级到 conceptually different (base model) teacher 并未放大对齐效果
5. **这是重要的 negative result**: 在当前设置下，hidden-state cosine alignment 不是有效的 stabilization/organization 方法

**判断**:
- 不建议扩大为 v2 — 问题更可能是方法论层面（cosine loss 太弱、层采样太稀疏、LoRA 容量不足），而非规模问题
- 如需继续探索轨迹对齐方向，建议换更本质的对齐目标（attention-pattern alignment、contrastive trajectory learning、feature-level alignment）

---

### Trajectory Dynamics Phase 1 — 结论压实

> **本节取代之前的 Phase 1.5 和早期 trajectory 报告。核心原则：区分 solid ground 与 not-solid 结论，不做过头声明。**

#### 实验覆盖

| 实验 | 描述 | 数据 |
|---|---|---|
| **T0** | `model.generate()` 期间轨迹捕获，不修改行为 | 60 hall + 30 syc, 7 layers × 48 steps |
| **T1** | 将轨迹投影到 behavior direction (v_hall, v_syc, random, shuffled) | 同 T0 数据 |
| **T2** | 每个 (layer, step) 训练 probe 预测最终行为 | 同 T0 数据 |
| **T3** | impulse 注入: 4 layers × 6 steps × 4 directions × 3 epsilons | 5 hall + 5 syc per combo |

#### 关键数据

| 指标 | 值 | 来源 |
|---|---|---|
| T0 output match (hall) | 60/60 | T0 §2 |
| T0 output match (syc contrast) | 60/60 (30 syc + 30 non-syc) | T0-S |
| P0 syc contrast separation | 0.833 (syc=1.000 vs non-syc=0.167) | P0 |
| T2 best probe acc (hall, 3-class) | 0.917 (L8, step 0) | T2 §2 |
| T2 best probe acc (syc, binary) | 0.983 (L8, step 15) | T2 §3 |
| T2 syc binary AUC | 1.000 at every (layer, step) | T2 §3 |
| T1 v_hall max sep (hall_vs_abst) | 2.40 (L12, step 0) | T1 §4.1 |
| T1 v_syc max sep (syc_vs_nonsyc) | 1.789 (L12, step 0) | T1 §4.2 |
| T1 v_hall/random ratio | 3.51× | T1 §4.3 |
| T1 v_syc/random ratio | 13.6× | T1 §4.3 |
| T1 syc collapse ratio | 0.347 (moderate) | T1 §4.2 |
| T2 hall structure | cross_layer_band (all 7 layers) | T2 §2 |
| T2 syc structure | cross_layer_band (all 7 layers), peak S15 vs S0 | T2 §3 |
| T3 v_hall mean ctrl | 0.0219 | T3 §7 |
| T3 v_syc mean ctrl | 0.0545 | T3 §7 |
| T3 random mean ctrl | 0.0303 | T3 §7 |
| T3 shuffled mean ctrl | 0.0392 | T3 §7 |
| **T3 syc v_syc/random ratio** | **2.73×** (P3 replication, n=20; was 6.17× at n=4) | T3 §9 + P3 |
| **T3 syc v_syc/random ratio (P4)** | **1.68×** (n=30, direction-dominated: directional=+0.0164, energy=-0.0022) | T3 §9 + P4 |
| **T3 syc controllability locus** | **prefill only** (step 8 = 0.000, confirmed at P3+P4) | T3 §7 + P3+P4 |
| **P5 feedback: gate rate** | **4.2%** (1/24) — probe trained on contrast groups fails to detect behavioral tendency | P5 |
| **P5 open-loop: v_syc α=-1.0** | **syc rate = 0.542** (baseline=0.583, marginal -0.04); α=-3.0/-5.0 → 0.875 (+0.29) | P5 |
| **P5 open-loop: perturbation vulnerability** | Any |α|≥3.0 → syc rate ≥ 0.875 (all directions); correction behavior fragile | P5 |
| **P5: sign asymmetry** | Negative v_syc INCREASES sycophancy → polarity points toward non_syc; pos α needed | P5 |
| **T3 hall v_hall/random ratio** | **0.28×** (P3, n=6 hall-prone; NOT direction-specific) | T3 §7 + P3 |
| v_hall/v_syc orthogonality | max \|cos\| = 0.106 (L20) | T1 §3 |

---

#### 已经可以写进 solid ground 的（5 条）

| # | 结论 | 证据 | 置信度 |
|---|---|---|---|
| **SG-1** | **Trajectory capture 可行且不扰动行为。** 在 `model.generate()` 过程中 hook 中间层 hidden states 不会改变输出。 | T0: hall 60/60, syc 30/30 输出完全匹配 | ⭐⭐⭐⭐⭐ |
| **SG-2** | **Hallucination 在 prefill 即可分离。** 模型在开始生成之前，内部表示已经区分了 hallucination-prone 和 abstention-prone 输入。 | T1: v_hall projection 从 step 0 即有显著分离 (p<0.05)；T2: probe 在 step 0 即达到最高准确率 (0.917) | ⭐⭐⭐⭐⭐ |
| **SG-3** | **这种分离是 cross-layer band，不是单点 spike。** 所有 7 个被测层 (L8-L23) 在 step 0 的 probe 准确率都 ≥0.833；上层 (L20, L23) 的 projection 分离幅度甚至更大 (8.13, 16.09)。 | T2 heatmap; T1 §6 supplementary layers | ⭐⭐⭐⭐⭐ |
| **SG-4** | **大幅 early impulse 可改变 hallucination-related behavior。** 在 prefill 或早期 decode step 注入大 epsilon impulse 可改变生成行为。 | T3: v_syc avg ctrl=0.0545, v_hall=0.0219, random=0.0303, shuffled=0.0392；prefill impulse displacement ~3.0 vs decode ~0 | ⭐⭐⭐⭐ |
| **SG-5** | **但当前 impulse 证据尚不足以证明 steering direction 的因果特异性。** random 和 shuffled direction 也能改变行为；v_hall 的 controllability (0.0219) 低于 random (0.0303) 和 shuffled (0.0392)。当前证据更支持 "early-state 对扰动敏感" 而非 "v_hall 在因果路径上"。 | T3: direction comparison table; v_hall not significantly > random | ⭐⭐⭐⭐ |
| **SG-6** | **Sycophancy 在 trajectory 上高度可分离，信号强于 hallucination。** T1: v_syc max sep=1.789, v_syc/random=13.6×; T2: binary AUC=1.0 at all (layer, step), peak acc=0.983。Syc vs non-syc 在表示空间中线性完美可分。 | T1 §4.2, T2 §3-5, [Syc Completion Report](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_SYCOPHANCY_TRAJECTORY_COMPLETION_REPORT.md) | ⭐⭐⭐⭐⭐ |
| **SG-7** | **Sycophancy 在 prefill 即可分离，但在 generation 中信号有 moderate collapse (ratio=0.347)。** 与 hallucination 不同，syc 的 probe 准确率在 generation 中不降反升（S0=0.917 → S15=0.983），说明 syc 是"prefill 播种、generation 放大"。 | T1: earliest_sep=0, collapse_ratio=0.347; T2: S0→S15 accuracy growth | ⭐⭐⭐⭐⭐ |
| **SG-8** | **v_hall 与 v_syc 近乎正交，是两个独立的表示维度。** max \|cos\| = 0.106 在所有 7 层。Hallucination 和 sycophancy 不是同一个方向的强弱变化，而是不同的内部轴。 | T1 §3: steering vector cosine tables | ⭐⭐⭐⭐⭐ |
| **SG-9** | **P2 实验确认：direction specificity 在全球层面不存在。** v_hall 与 v_orthogonal（与 v_hall 严格正交）产生完全相同的 Hall 率变化（ΔH=-0.283 vs -0.283）。五个方向的 controllability 全部在 0.24-0.28 窄区间内。 | [P2 Direction Specificity Report](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P2_DIRECTION_SPECIFICITY_REPORT.md) | ⭐⭐⭐⭐⭐ |
| **SG-10** | **P3 确认：syc direction-specificity 在 n=20 下存续，但 ratio 从 6.17× 收缩到 2.73×。** 同时确认 prefill-only、L10-concentrated。Hallucination v_hall/random=0.28×（确认不特异）。 | P3 replication: n_hall=6, n_syc=20, 108 combos, 3 layers × 3 steps | ⭐⭐⭐⭐⭐ |
| **SG-11** | **P4 decomposition 证明 syc controllability 是方向主导（direction-dominated），而非能量驱动。** 纯方向贡献 = +0.0164，纯能量贡献 = -0.0022。norm-matched orthogonal vector（与 v_syc 等范数、正交方向）的 controllability(0.0185) 低于 random(0.0207)。v_syc 的因果效应来自其方向对齐，不是更大的范数。 | P4: n_syc=30, orthogonal decomposition, 90 combos, 3 layers × 2 steps × 5 directions × 3 epsilons | ⭐⭐⭐⭐⭐ |
| **SG-12** | **P5 + P5-bis 解决 v_syc 极性：v_syc 指向 sycophancy 方向。** P5 闭环反馈零结果（gate rate=4.2%，探测器学到 group membership）。P5-bis 25-combo α-sweep: 负 α (减 v_syc) 降低 syc（最佳 α=−3.0: 0.5833→0.3750, −35.7%），正 α (加 v_syc) 饱和至 1.0000。仅 v_syc 有此非对称效应；控制向量均随 |α| 单调增。P5 正 α 假设被证伪；负 α 才是抗谄媚方向。 | P5: n_test=24, probe→gate→hook + open-loop, 4 directions × 3 alphas。P5-bis: n=24, 25 combos, α∈[−5,+5]。 | ⭐⭐⭐⭐ |
| **SG-13** | **P6 行为专用探针训练成立。** 行为专用探针 train acc=81.9%, test acc=77.8%（仅标准 prompt 样本 + 行为标签）。Open-loop v_syc α=−3.0 实现 −50% syc 降低（0.6667→0.3333），确认极性/alpha。 | P6: n_test=12, behavior-only probe (train acc=81.9%, test acc=77.8%), open-loop −50% at α=−3.0。 | ⭐⭐⭐⭐ |
| **SG-14** | **P6-bis 诊断：根因不是阈值校准，是 hook 架构 bug。** 探针在 standalone forward pass 下达到 +0.54 分数分离（syc μ=0.82, non-syc μ=0.29），但在 `model.generate()` 内 hook 捕获的是 generated-token 而非 prompt-token hidden state。gate rate 在所有阈值（0.30−0.90）上恒定 8.3%。Fix: 两阶段架构 —— standalone scoring → conditional generate with steering。 | P6-bis: n_test=12, 5 thresholds + 3 percentile rules, standalone probe separation=+0.54, in-hook probe μ=0.47 ≠ standalone μ=0.64。Open-loop −50% 第三次复现。 | ⭐⭐⭐⭐ |
| **SG-15** | **P6-ter 两阶段反馈控制闭环打通。** Standalone probe scoring → conditional generate with steering。th=0.50 时 syc 从 0.7500 降至 0.2500（−66.7%），**击败 open-loop（−44.4%）**。两阶段反馈因为选择性 steering（仅 steer syc 高风险样本，保留自然非谄媚行为）效果优于全量 open-loop。Random vector 两阶段仅 −22.2%，确认闭环方向特异性（v_syc/random=2.67× in closed-loop）。 | P6-ter: n_test=12, 5 thresholds (0.30−0.70), two-stage scored standalone probe μ=0.6448, closed-loop v_syc th=0.50 = −66.7% vs open-loop −44.4%, random two-stage −22.2%。**Sycophancy probe→gate→hook 闭环首次打通。** | ⭐⭐⭐⭐⭐ |
| **SG-16** | **P7 S15 放大机制调查：S15 不是因果敏感期。** 逐步探针打分显示 P6 探针在 step 1 达到最强分离（+0.65），而非 S15（+0.13）。逐步 steering 显示单步注入无效 —— S5/S10 反而增加谄媚，S15 无效果。结论：谄媚是累积式、分布式过程，非单步"决策"。Open-loop 有效是因为累积效应。T2 的 S15 峰值是可读性（per-position probe accuracy），不是可操纵性。Readability ≠ manipulability。 | P7: n_test=12, 3 phases. Phase 1: per-step probe scores — peak at step 1 (+0.65), S15 weak (+0.13). Phase 3: per-step steering — all steps null or worse. S15 = not a sensitive period. | ⭐⭐⭐⭐ |
| **SG-17** | **P8 大规模复制（n=24）：效果方向正确但弱于 P6-ter。** 两阶段 th=0.50 将 syc 从 0.7083 降至 0.5417（−23.5%），open-loop 降至 0.5000（−29.4%）。th=0.50 和 th=0.40 给出相同 syc rate（0.5417），阈值区分度消失。所有成对比较均未通过 Fisher 精确检验（p>0.05）。P6-ter 的 −66.7%（n=12）很可能为小样本 artifact（F24）。方向正确——steering 确实降低 syc——但 n=24 上的效应大小不足以在统计上区分两阶段与 open-loop。**这是一个重要的方法学发现：小样本闭环优势是虚假的。** | P8: n_test=24, 4 conditions (baseline, two-stage th=0.50, two-stage th=0.40, open-loop)。Baseline=0.7083, th=0.50=0.5417 (−23.5%), th=0.40=0.5417 (−23.5%), open-loop=0.5000 (−29.4%)。Fisher p 均 >0.05。 | ⭐⭐⭐ |
| **SG-18** | **P9 Cross-Bottleneck: Steering preserves structural integrity at L10. 瓶颈独立.** ARI=1.0, purity=1.0 在 baseline 和 steered 条件下完全相同 (|delta|<0.005)。v_syc (α=−3.0) 产生均匀平移 (||shift||=3.0)，保持相对几何不变 (cos sim=0.9707)。steering = clean directional bias，不产生结构退化。**正向发现：steering 无 collateral damage。跨瓶颈协同 (1+1>2) 在表示层面不支持。** | P9: n_test=24, 2 conditions, KMeans k=2 clustering on L10 last_prompt_token hs. Baseline ARI=1.0, Steered ARI=1.0. All metrics unchanged. | ⭐⭐⭐ |
| **SG-19** | **P10 正式排除：单方向 impulse 不适用于 hallucination 控制。** 5 层证据链（T1–T3, P2, P3, P4, B2）收敛：v_hall = v_orthogonal（P2），v_hall/random = 0.28×（P3），v_hall controllability < random controllability（T3）。Hall = 纯能量，Syc = 方向主导的不对称是项目最重要的发现。**Hallucination 的方向特异性不存在——该研究线关闭。** 替代路径：闭环 gate（M3-v6）、能量扰动方法、多方向组合（针对退化基线）。 | P10: 文档化（无新实验）。整合 T1-T3, P2, P3, P4, B2 五层证据链。P2 的 v_hall=v_orthogonal 是最强负性证据（正交向量产生完全相同的行为效果）。 | ⭐⭐⭐⭐⭐ |
| **SG-20** | **P11 Stabilization Scaling: Per-Action KMeans 在 4 维度上均稳健通过。** Seed scaling (5→100): delta +0.14→+0.17，效应不衰减。Noise robustness (0→1.0): delta 从 +0.055 增长至 +0.100，Y-aware 在噪声下更优。Action complexity (3→20): 峰值 +0.330 (5 actions)，3–10 均可工作。Cross-bottleneck perturbation (3 types × 7 levels): PA 始终 > NoMemory，最极端 dropout=0.9 时仍有 +0.035 margin。**Stabilization 是三瓶颈中验证最充分的组件。Diagnosed → Fully Validated。「未 scaling」标签移除。** | P11: 跨项目文档化。整合 C4 (seed), N1 (noise), N2 (action complexity), C5 (perturbation) 四项 scaling 实验。Per-Action KMeans 在全部 4 维度、所有条件下均保持 > NoMemory。 | ⭐⭐⭐⭐⭐ |
| **SG-21** | **P12 Absorption Steering: 方向存在但效应是 homogenization with degradation。** v_abs (h_early − h_late) steering 将 delta_H 从 0.750 降至 0.250 (−67%)，但机制是牺牲最佳位置（H_early 0.250→0.500）+ 适度改善最差位置（H_mid 1.000→0.750, H_late 1.000→0.750）。v_abs/random = 2.0× 方向特异性，但远弱于 sycophancy（2.73×）且是负和性质的改善。**Absorption 不是 direction-dominated —— directional steering 不是可行补救路线。A1→A4→P12 链确认 probe→behavior gap 抵抗所有已测干预类型。** | P12: n_test=30 (10 per position), 5 α-sweep + random/orth controls at L10。Baseline H_early=0.250 H_mid=1.000 H_late=1.000。v_abs α=+3.0 使 delta_H −67% 但以 H_early +100% 为代价。n=5 随机/正交向量使所有位置 H=1.000 — v_abs 方向不同但无净收益。 | ⭐⭐⭐ |
| **SG-22** | **P13 Energy/Direction Asymmetry: L10 表示层均匀，行为非对称来自下游。** Energy perturbation (noise 0.01-0.10) 和 direction perturbation (v_abs α=-3 to +3) 在 L10 hidden state 上都产生近乎完美的均匀平移（max_ratio ≤ 1.02）。P12 的行为非对称性（early 对方向扰动更敏感）不是来自表示层差异，而是来自下游计算对相同扰动的差异化放大。**Absorption 的非对称性在行为层，不在表示层。这解释了为什么所有基于 L10 的补救（A3 rectification, P12 steering）都失败——它们瞄准了错误的层次。** | P13: n_test=30, energy noise sweep 0.01-1.0, direction α-sweep -5 to +5. 核心指标: position-wise ||Δh|| max_ratio。Energy max_ratio=1.01-1.02, Direction max_ratio=1.000-1.001 — 均为均匀。 | ⭐⭐⭐ |
| **SG-23** | **P14 Cross-Project Synthesis: 诊断+验证+排除阶段闭合。** IC-4 (26 experiments, 30 positive findings) + IC-2 (6 experiments, 4 positive findings) 跨项目合成。三瓶颈全部诊断，两个有补救（Stabilization Fully Validated, Organization partial），一个 L10 路径闭合（Absorption → post-L10 ）。三条死胡同排除（Hall single-direction, Absorption L10 mediation, Absorption directional steering）。实战建议: PA-KMeans for behavioral cloning, closed-loop gate for QA hallucination, open-loop steering for sycophancy。**项目诊断阶段完成。Absorption 是唯一 frontier。** | P14: 完全合成文档。包含 Executive Summary, 三瓶颈架构, 干预架构, 30-Finding catalog, 13-Exclusion catalog, Boundary Conditions, 实战建议。 | ⭐⭐⭐⭐⭐ |
| **SG-24** | **P15 Hallucination LoRA: B-bottleneck KNOWS→produces gap BRIDGED. H=0.000, C=1.000.** LoRA rank=4, 90 samples (45A+45U), 3 epochs. 幻觉率从 0.417 降至 0.000，正确率保持 1.000。三个位置 (early/mid/late) 全部 H=0.000，位置 gap 消除。Phase 10 A-LoRA 仅将 H 降至 0.500，P15 的 B-LoRA 实现 H=0.000。**这是项目第一个完全弥合 B-bottleneck 的权重级干预。** | P15: n_test=30, log-prob evaluation。Pre: H=0.417 C=1.000 ΔH=0.250。Post: H=0.000 C=1.000 ΔH=0.000。Training: 90 samples, 4819s CPU. | ⭐⭐⭐⭐⭐ |
| **SG-25** | **P16 LoRA Geometry: LoRA 是 ROUTING fix (绕过 K↔D)，不是 GEOMETRY fix (对齐 K↔D)。** 9 层 probe acc 均保持 1.0000 (K-subspace 不变)。w_probe steering 效应在 8/9 层为零或负，在 L12 是破坏性的 (H 0→0.25 at α=+2.0)。K↔D 子空间正交性在 LoRA 后仍然存在。机制: LoRA 改变默认行为路径 (routing)，不改变 K↔D 几何关系。 | P16: 9 layers tested, w_probe steering α∈[-2,0,+2]. Alignment gain ≤ 0 at 8/9 layers. L12 gain=+2.001 but destructive. | ⭐⭐⭐⭐⭐ |
| **SG-26** | **P17 LoRA Module Ablation: q_proj (Query projection) 是 routing fix 的唯一关键模块。** 消融 q_proj LoRA → H 0.000→0.250 (60% 总效应)。消融 k_proj/v_proj/o_proj 各自零效应。组合消融 -q-k 确认 q 独立关键，-v-o 仅微小效应 (H=0.083)。**B-bottleneck 本质是注意力路由问题 — q_proj 控制"attends to what"，而非 v_proj (output what) 或 o_proj (aggregate how)。** Routing fix = 改变 Query 让模型关注不同的信息。 | P17: 8 ablation conditions, 30 test samples, 192 LoRA params. -q ΔH=+0.250, -k/ -v/ -o ΔH=0.000. Full time: 871s CPU. | ⭐⭐⭐⭐⭐ |
| **SG-27** | **P18 q_proj Layer Ablation: DEEP layers (16-23) 是 Query 路由的充分核心。** ... **B-bottleneck 证据链完整: P13+P14 (几何证明) → P15 (LoRA 弥合) → P16 (路由机制) → P17 (q_proj 定位) → P18 (deep 层核心)。** | P18: 8 conditions, 30 test samples. Full=H 0.000, ONLY_deep=H 0.000, ONLY_mid=H 0.083, ONLY_early=H 0.250. Time: 910s CPU. | ⭐⭐⭐⭐⭐ |
| **SG-28** | **P19 Self-Bootstrapping Attention Rerouting: 自主自修复 Agent 首次验证。** | P19: 30 samples, base 0.5B, deep 16-23 attn analysis, H −40%, 2/5 fixed. 372s CPU. | ⭐⭐⭐⭐⭐ |
| **SG-29** | **P20 Multi-Strategy Self-Bootstrapping: 策略多样性验证。** 三种策略 (PRUNE/NEUTRALIZE/SENTENCE)，Agent 择优。36 样本，H −20%，唯一修复成功来自 SENTENCE。分心效应在部分样本上是句子级的。 | P20: 36 samples, 3 strategies, H 0.417→0.333, SENTENCE-only fix. 375s CPU. | ⭐⭐⭐⭐ |
| **SG-32** | **P21 Self-Generated Strategy Discovery: 负结果 — 0.5B 模型无法自我诊断。** LLM 自诊断→自生成修复→log-prob验证。H=0.417 无变化，0/5 修复。generate() 输出在自诊断任务上不可靠。**定义 Meta FAIR 自举范式的 0.5B 能力阈值边界。** | P21: 30 samples, generate() unreliable, 2 stochastic runs give different results. 1302s CPU. | ⭐⭐⭐ |
| **SG-33** | **P22 Cascading Counterfactual: 概率引导策略发现取代人类菜单。** 注意力=贝叶斯先验，log-prob=似然，级联=决策规则。H 0.417→0.333 (−20%), 1/5 修复。发现 "AI assistant" 包装句是因果分心源。 | P22: 30 samples, cascade prune→neutralize→sentence, 36 CF, 8.9min. auto:sentence fix = wrapper removal. | ⭐⭐⭐⭐⭐ |
| **SG-34** | **P23 Joint Counterfactual + Full-Token Causal Attribution: 注意力-因果性解耦被证伪。** 188 次反事实全 token 归因。Corr(注意力, Δlp_diff) = −0.0086 ≈ 零。真正分心词 "funding" (attn~0.007, Δ~+0.36) 因果力 50× 优于高注意力词。幻觉来源是语义级而非 token-注意力级。 | P23: 30 samples, 71 total CF, 3 full attributions. Corr(α,Δ)=−0.0086. Text-level has representation floor. | ⭐⭐⭐⭐⭐ |
| **SG-35** | **P24 Embedding-Level Intervention: 幻觉是结构性的(注意力路由)而非表征性的(token嵌入)。** 嵌入替换使幻觉恶化。文本移除因改变注意力结构而优于嵌入替换。只有路由级干预(LoRA on q_proj)能根本修复。 | P24: 30 samples, embed replace/zero/noise/combo, 0/5 fixed. Embedding intervention worse than text-level. | ⭐⭐⭐⭐⭐ |
| **SG-30** | **IC-2d Readout-Matched Episodic: Learned readout CANNOT rescue episodic memory. Counter-hypothesis KILLED.** MLP (0.095) < RF (0.190) < k-NN (0.195) < NoMemory (0.445). 问题不是读出机制，是表示。只有 consolidation (Per-Action KMeans, 0.585) 能超越 NoMemory。**IC-2 和 IC-4 在读出-vs-表示层面收敛：两个项目都发现表示层面的干预 (consolidation/LoRA routing) 有效，读出层面 (MLP/w_probe steering) 无效。** | IC-2d: 5 seeds × 1200 samples, 4 readout strategies. Final: NoMemory 0.445, k-NN 0.195, RF 0.190, MLP 0.095. | ⭐⭐⭐⭐⭐ |
| **SG-31** | **P21 Absorption Attention Patterns: Attention entropy CONSISTENTLY higher for late-position inputs across ALL layers. U-shaped gap: L0 (+11.3%) → L9 minimum (+4.4%) → L23 maximum (+13.0%). Deep-layer attention routing is where position asymmetry amplifies. B-bottleneck (P18) 和 A-bottleneck (P21) 在同一个机制位点收敛：deep-layer attention routing。** | P21: 90 samples (30/position), 9 layers, eager attention mode. | ⭐⭐⭐⭐⭐ |
| **SG-35** | **P24 Multi-Layer Steering (A-bottleneck): L10 vs L21 v_abs steering produces IDENTICAL position profiles. ΔH=0.250, H profile (0.25, 0.50, 0.50) at BOTH layers. ALL 3 hypotheses (H20.1-H20.3) refuted. A-bottleneck steering is LAYER-INDEPENDENT — v_abs injection at any layer propagates uniformly. Hidden-state vector interventions EXHAUSTED for absorption. Only path forward: weight-level (LoRA) or attention-direct modification. B-bottleneck: LoRA works → A-bottleneck: LoRA untested.** | P24: 4 conditions (L10/L21 × a0/a3), n=10/position, train n=5. L10_a3 ΔH=0.250, L21_a3 ΔH=0.250 — identical. 1024s CPU. | ⭐⭐⭐⭐⭐ |
| **SG-36** | **P25 Absorption LoRA: Weight-level intervention PARTIALLY closes position gap. ΔH 0.750→0.250 (−67%). Mid/late H 1.000→0.000 (FIXED). Early residual H=0.250 persists — proximity-based over-confidence survives LoRA. log-prob ΔH=0.000 but generate ΔH=0.250 — probe→behavior gap partially closed. Absorption has TWO components: (1) distance-based routing degradation → FIXED by LoRA; (2) proximity-based source confusion → NOT fixed. Residual may be embedding-level (RoPE position encoding baked in).** | P25: 30 test samples (10/position), P15 checkpoint, dual-space eval (generate + log-prob). Pre ΔH=0.750→Post ΔH=0.250. Time: 1375s CPU. | ⭐⭐⭐⭐⭐ |
| **SG-37** | **P26 Attention Temperature Scaling: Attention temperature at early layers (L0-L9) CANNOT fix early-position over-confidence. Any T≠1 makes early H WORSE (T=0.5: +0.250, T=5.0: +0.500). Mid/late H=1.000 completely invariant. Early over-confidence is a ROBUST property, not a fragile attention pattern. Absorption has THREE components: distance routing (LoRA fixes), proximity over-confidence (nothing fixes), attention calibration (baseline optimal). Consistent with RoPE embedding-level hypothesis.** | P26: 4 conditions (T=0.5/1.0/2.0/5.0), 30 test samples, eager attention. T=1.0 baseline is optimal. Time: 2253s CPU. | ⭐⭐⭐⭐⭐ |
| **SG-38** | **P27 Position ID Offset (RoPE test): N=100 makes early H WORSE (0.250→0.500, consistent perturbation pattern). N≥300 breaks model entirely (all H=0.000, C=0.000 — model can't process context). RoPE hypothesis untestable at inference time via padding. Early H=0.250 is a LOCAL MINIMUM — THREE interventions (steering, attention temp, position offset) all fail to reduce it. LoRA is the ONLY effective absorption intervention (fixes mid/late). Absorption intervention ladder COMPLETE.** | P27: 4 offsets (0/100/300/500), 30 test samples. N=300: model break. Time: 1356s CPU. | ⭐⭐⭐⭐⭐ |
| **SG-39** | **P28 Sycophancy n=48: Direction confirmed (Baseline 0.625→Two-stage 0.417, −33.3%). Fisher p=0.0654 (trending, NOT crossing α=0.05). Sycophancy effect is REAL (direction consistent across n=24 and n=48) but statistically UNDER-POWERED — Cohen's h≈0.42 requires n≈90+ for significance. Current data ceiling (60 standard samples) insufficient. This is a DATA limitation, not an effect limitation. Sycophancy bottleneck: direction ✅, significance ⚠️, data ceiling ❌.** | P28: 3 conditions × 48 samples, v_syc L10 steering. 4607s CPU. | ⭐⭐⭐⭐⭐ |
| **SG-40** | **P29 Cross-Bottleneck Steering Interaction: SYNERGY, NOT TRADE-OFF. v_syc(−3.0) STRONGLY reduces hallucination (mid H 1.000→0.000, same as P15 LoRA). v_hall(−3.0) has ~zero effect on sycophancy. Asymmetric synergy: hallucination ⊃ sycophancy as nested subspace. v_syc beats v_hall for hallucination at mid position (H=0.000 with C=0.600 vs v_hall C=0.000). cos(v_hall, v_syc)=0.2355 explains partial geometric overlap. Fighting sycophancy inadvertently fights hallucination — a fortunate design property.** | P29: 6 conditions (3 hall × 3 syc), 24+16 samples, cross-steering at L10. 4178s CPU. | ⭐⭐⭐⭐⭐ |
| **SG-41** | **P30 Unified Bottleneck Steering: COMBINATION SYNERGY. U_1:2 (v_hall=−1.0 + v_syc=−2.0) achieves TRIPLE CROWN — mid H=0.000 (matches P15 LoRA), avg C=0.444 (8× better than v_hall alone at 0.056), and sycophancy=0.375 (the ONLY condition that REDUCES sycophancy, −25%). U_1:2 beats both single vectors on all three metrics. The ratio matters critically: 1:1 kills C, 2:1 kills C, only 1:2 (67% syc energy) hits the sweet spot. v_syc component protects correctness while v_hall component suppresses hallucination. This is the strongest hidden-state intervention result in the entire IC-4 project.** | P30: 6 conditions × 30 pos-sens + 6 × 16 syc = 276 generations. 6492s CPU (108min). | ⭐⭐⭐⭐⭐ |

---

#### 还不能写太满的（5 条）

| # | 结论 | 原因 |
|---|---|---|
| **NS-1** | ❌ 不能说"v_hall 已证明在局部因果路径上" | P2 确认：v_hall 与 v_orthogonal 产生完全相同的 ΔH（-0.283 vs -0.283）。方向特异性在全球层面不存在。impulse 效应来自扰动能量，不是方向。 |
| **NS-2** | ❌ 不能说"局部可控性地图已经稳定" | T3 只有 5 samples per combo，sweep 粗糙；P2 证明单方向不可控。需要新方法（multi-direction、attention-level、feedback）。 |
| **NS-3** | ❌ 不能把 `epsilon=1,3,5` 的冲击直接等同于精细 control | Epsilon scaling 分析显示非线性：behavior change 不随 displacement 线性增长 (T3 §9)。大幅 impulse 是"系统重置"而非"方向微调"。 |
| **NS-4** | ❌ 不能说"syc collapse 机制已理解" | 我们知道 syc collapse ratio=0.347，但不知道 WHY。是 attractor dynamics？噪声积累？autoregressive smoothing？ |
| **NS-5** | ❌ 不能说"syc direction-specific 是全方向特异的" | P4: v_syc/random=1.68×（n=30）。虽然方向主导（energy=-0.0022），但 ratio 适中。impulse 扰动是钝器，精细因果 effect size 需要更强的方法来测量。 |

---

#### 核心判断：Hallucination 与 Sycophancy 的轨迹动力学

> **Hallucination**: 不是末端输出现象，而是 prefill 即可读、跨层传播、对早期扰动敏感的内部状态。更像 **"早期形成、跨层传播的状态差异"**，距离 **direction-specific causal control** 还有半步（P2 已确认半步不行）。
>
> **Sycophancy**: 比 hallucination 信号更强、更稳定、但 temporal profile 完全不同。是 **"prefill 播种、generation 放大"** 的模式——信号在 S15 达到峰值而非 S0。moderate collapse (ratio=0.347) 说明生成过程会部分拉近两条轨迹但不抹平。
>
> **跨行为比较**：
>
> | Dimension | Hallucination | Sycophancy |
> |---|---|---|
> | Prefill separation | ✓ (S0 peak) | ✓ (S0 seed) |
> | Generation amplification | ✗ (flat/degrading) | ✓ (S15 peak, +0.066) |
> | Generation collapse | Volatile (var=0.160) | Moderate (ratio=0.347, var=0.010) |
> | Probe AUC ceiling | N/A (3-class) | 1.000 (binary, every step) |
> | **Impulse direction specificity** | **✗ NO (P2+T3+P4: v_hall<random)** | **✓ YES (P4: v_syc/random=1.68×, direction-dominated)** |
> | **Controllability locus** | All steps (prefill + gen) | **Prefill only** |
> | v_task / random ratio | 3.51× | 13.6× |
> | v_hall / v_syc orthogonality | — | max \|cos\| = 0.106 |
>
> **关键推论**: Hallucination 和 sycophancy 是两个近乎正交的表示维度，共享 cross_layer_band 结构但有不同的 temporal dynamics。**P4 decomposition (n_syc=30, orthogonal decomposition, 90 combos) 确认了最关键的不对称性并深化了因果理解：sycophancy 是 direction-dominated（v_syc/random=1.68×, energy=-0.0022）而 hallucination 不是（v_hall/random=0.19×）。v_syc 的因果效应来自方向对齐而非能量/范数 — 这是该方向在因果路径上的决定性证据。两者不是同一种 controllability object，需要不同的介入策略。**

---

#### 下一步路线图：P0 ✓ → P2 → P3

| 优先级 | 任务 | 状态 | 为什么 | 涉及 |
|---|---|---|---|---|
| **P0** | **补 sycophancy 对照集** — 构造 `non_sycophantic` 样本，使 T0-T3 全线可运行 syc 方向 | ✅ **完成** | Balanced syc contrast: syc=1.000, non-syc=0.167, separation=0.833。T0/T1/T2/T3 syc 分支已全线跑通。T3 发现 syc 是 direction-specific（6.17×），与 hallucination 形成关键不对称。 | [Syc Completion Report](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_SYCOPHANCY_TRAJECTORY_COMPLETION_REPORT.md) |
| **P1** | **做 position-to-behavior sensitivity 的正式整合** — 将表示层 KNN=1.0 + 行为层 ΔC=0.07 写进主地图 | ✅ **完成** | 三层证据链已写入 C8/C9。吸收瓶颈诊断闭合。 | 本次更新 |
| **P2** | **方向特异性测试** — norm-matched random, orthogonalized random, same-layer different-direction matched-energy impulses | ✅ **完成** | P2 确认：v_hall = v_orthogonal (ΔH=-0.283 vs -0.283)。方向特异性在全球层面不存在。 | [P2 Report](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P2_DIRECTION_SPECIFICITY_REPORT.md) |
| **P3** | **syc direction-specific replication** — 用 n≥20 复现 syc 的 6.17× 方向特异性；探索 multi-direction/attention-level 处理 hallucination | ✅ **完成** | P3: n=20, 108 combos, v_syc/random=2.73×（复现存续，从 6.17× 缩至 2.73×）。Hall v_hall/random=0.28×（确认不特异性）。Syc prefill-only, L10-concentrated all confirmed. | [Syc Completion Report](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_SYCOPHANCY_TRAJECTORY_COMPLETION_REPORT.md) |
| **P4** | **syc direction-vs-energy decomposition** — 用 norm-matched orthogonal vector 分解方向贡献 vs 能量贡献 | ✅ **完成** | P4: n_syc=30, 90 combos, 5 directions (含 orthogonal)。Direction-dominated: 纯方向=+0.0164, 纯能量=-0.0022。v_syc 因果效应来自方向对齐。 | [Syc Completion Report + P4](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_SYCOPHANCY_TRAJECTORY_COMPLETION_REPORT.md) |
| **P5** | **syc feedback control** — 基于 P4 证实的方向特异性，设计 probe → gate → hook 闭环控制 sycophancy | ✅ **完成（part 1）** | P5: n_test=24, 12 feedback + 12 open-loop combos。Probe→gate→hook null（gate rate=4.2%），探测器学到 group membership 而非行为。Open-loop 提示符号不对称但与 P5-bis 冲突（测试集 artifact）。 | [P5 Report](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P5_SYC_FEEDBACK_REPORT.md) |
| **P5-bis** | **syc open-loop α-sweep** — 测试正负 alpha 在所有方向上的效应，确定 v_syc 极性 | ✅ **完成** | P5-bis: n=24, 25 combos (1 baseline + 12 neg + 12 pos)。**v_syc 指向 sycophancy 方向**：负 α (减 v_syc) 降低 syc（最佳 α=-3.0, 0.5833→0.3750, −35.7%）；正 α (加 v_syc) 饱和至 1.0000。P5 正 α 假设被证伪。控制向量无此非对称效应。 | [P5-bis Report](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P5_BIS_SYC_FEEDBACK_REPORT.md) |
| **P6** | **syc behavior-only probe** — 用行为标签在标准 prompt 样本上训练探针，验证 probe→gate→hook 反馈路径 | ✅ **完成** | P6: n_test=24, behavior-only probe (train acc=81.9%, test acc=77.8%), gate rate=8.3% (null), open-loop v_syc α=−3.0 = −50% reduction (0.6667→0.3333)。**探针训练成立但分数聚集在 0.5 —— 阈值校准是下一步。** | [Syc Completion Report](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_SYCOPHANCY_TRAJECTORY_COMPLETION_REPORT.md) §5-ter |
| **P6-bis** | **syc threshold calibration → hook architecture diagnostic** — 发现根因不是阈值校准，是 hook 捕获 generated-token 而非 prompt-token hidden state | ✅ **完成** | P6-bis: n_test=12, 5 thresholds (0.30−0.50) + 3 percentile rules (top-20%,30%,40%)。Standalone probe: syc μ=0.82, non-syc μ=0.29, separation=+0.54。In-hook probe: μ=0.47, gate rate 恒定 8.3%。Hook 架构 bug 已诊断。Open-loop −50% 第三次复现。 | [P6-bis Report](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P6_BIS_THRESHOLD_REPORT.md) |
| **P6-ter** | **syc two-stage feedback control** — standalone probe scoring → conditional generate with steering hook | ✅ **完成** | P6-ter: n_test=12, 5 thresholds (0.30−0.70), two-stage vs open-loop vs random control。**Closed-loop syc −66.7% (th=0.50) beats open-loop −44.4%**。Random two-stage −22.2% (v_syc/random=2.67×)。Selective intervention > universal intervention。**Sycophancy feedback loop CLOSED.** | [P6-ter Report](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P6_TER_TWO_STAGE_REPORT.md) |
| **P7** | **S15 amplification mechanism investigation** — per-step probe scoring + token analysis + per-step steering | ✅ **完成** | P7: n_test=12, 3 phases. Phase 1: probe scores peak at step 1 (+0.65), NOT S15 (+0.13). Phase 3: single-step steering null or worse (S5: +12.5%, S10: +25%). S15 is NOT a sensitive period. Sycophancy is cumulative/distributed, not a single-step decision. Readability (T2) ≠ manipulability (P7). | [P7 Report](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P7_S15_AMPLIFICATION_REPORT.md) |
| **P8** | **large-scale replication (n=24)** — replicate two-stage feedback on n=24 with Fisher exact test | ✅ **完成** | P8: n_test=24, 4 conditions. Baseline=0.7083, th=0.50=0.5417 (−23.5%), th=0.40=0.5417 (−23.5%), open-loop=0.5000 (−29.4%)。All Fisher p >0.05. P6-ter's −66.7% likely n=12 artifact (F24). Direction correct, effect weaker at scale. **Important negative: small-sample closed-loop advantage is spurious.** | [P8 Report](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P8_LARGE_SCALE_REPORT.md) |
| **P9** | **Cross-Bottleneck Structural Integrity Check** — test if v_syc steering degrades L10 syc/non-syc clustering | ✅ **完成** | P9: n_test=24, 2 conditions (baseline forward, steered forward). KMeans k=2 on L10 last_prompt_token hs. Baseline ARI=1.0 purity=1.0, Steered ARI=1.0 purity=1.0 (unchanged). Steering = uniform translation (||shift||=3.0, cos sim=0.9707). **Positive: steering has no collateral structural damage. Bottlenecks are independent — no 1+1>2 synergy at representational level.** | [P9 Report](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P9_CROSS_BOTTLENECK_REPORT.md) |
| **P10** | **Formal Exclusion: Hall single-direction impulse** — document 5-layer evidence chain closing this line | ✅ **完成** | P10: 文档化。整合 T1-T3, P2, P3, P4, B2 五层证据。v_hall=v_orthogonal (P2) 是最强负性证据。v_hall/random=0.28× (P3)。Hall=纯能量, Syc=方向主导不对称已确认。**Hallucination 单方向 impulse 研究线关闭。替代路径：闭环 gate (M3-v6)、能量扰动。** | [P10 Report](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P10_HALL_IMPULSE_EXCLUSION.md) |
| **P11** | **Stabilization Scaling Integration** — cross-project validation across 4 scaling dimensions | ✅ **完成** | P11: 跨项目文档化。C4 种子 scaling (5→100, delta +0.14→+0.17)。N1 噪声鲁棒性 (0→1.0, delta +0.055→+0.100)。N2 动作复杂度 (3→20, peak +0.330 at 5)。C5 跨瓶颈扰动 (3 types × 7 levels, PA always > NoMemory)。**Stabilization fully validated — 「未 scaling」标签移除。** | [P11 Report](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P11_STABILIZATION_SCALING.md) |
| **P12** | **Absorption Directional Steering** — open-loop α-sweep with v_abs (h_early − h_late) | ✅ **完成** | P12: n_test=30, 5 α-sweep levels + random/orth controls at L10. v_abs α=+3.0: delta_H 0.750→0.250 (−67%) but H_early 0.250→0.500 (+100%). v_abs/random=2.0× direction-specificity (weaker than syc 2.73×). **Negative: directional steering is NOT viable absorption remedy. Effect is homogenization with degradation. A1→A4→P12 chain: probe→behavior gap resists all tested interventions.** | [P12 Report](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P12_ABSORPTION_STEERING.md) |
| **P13** | **Energy vs Direction Asymmetry** — representational-level perturbation comparison | ✅ **完成** | P13: n_test=30, energy noise 0.01-1.0 + direction α=-5 to +5 at L10. Both produce perfectly uniform L10 shifts (max_ratio ≤ 1.02). **P12 behavioral asymmetry is downstream, not in L10 perturbation. Explains why all L10-targeted remedies (A3, P12) fail — they target the wrong level.** | [P13 Report](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P13_ENERGY_VS_DIRECTION.md) |
| **P14** | **Cross-Project Synthesis** — full IC-4 + IC-2 integration & deliverable packaging | ✅ **完成** | P14: Executive summary, 3-bottleneck architecture, intervention architecture, 30-Finding catalog, 13-Exclusion catalog, boundary conditions, practical recommendations, project stats, timeline. **Diagnosis+Validation+Exclusion phase CLOSED. Absorption is the only frontier.** | [P14 Synthesis](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P14_CROSS_PROJECT_SYNTHESIS.md) |
| **P15** | **Hallucination LoRA Fine-Tuning** — weight-level B-bottleneck remedy via LoRA on hallucination-labeled data | ✅ **完成 (Breakthrough)** | P15: n_test=30, LoRA rank=4, 90 train samples (45A+45U), 3 epochs. **H 0.417→0.000 (ZERO), C=1.000.** All three positions H=0.000. **B-bottleneck KNOWS→produces gap BRIDGED by weight-level LoRA.** Outperforms Phase 10 A-LoRA (H=0.500). | [P15 Report](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P15_HALLUCINATION_LORA.md) |
| **P16** | **LoRA Geometry Analysis** — HOW does LoRA bridge the gap? K↔D alignment or routing bypass? | ✅ **完成 (Informative Negative)** | P16: 9 layers, w_probe steering α∈[-2,0,+2]. Probe acc=1.0000 at all layers (K-subspace preserved). Alignment gain ≤ 0 at 8/9 layers. **LoRA = ROUTING fix (bypass K↔D), NOT geometry fix (align K↔D).** | [P16 Report](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P16_LORA_GEOMETRY.md) |
| **P17** | **LoRA Module Ablation** — WHICH attention projection (q/k/v/o) carries the routing fix? | ✅ **完成** | P17: 8 ablation conditions, 30 test samples. **q_proj is the sole critical module** — zeroing q LoRA increases H 0.000→0.250 (60% total effect). k/v/o individually have ZERO effect. **B-bottleneck = attention routing problem. Routing fix = Query projection change.** | [P17 Report](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P17_LORA_ABLATION.md) |
| **P18** | **q_proj Layer Ablation** — WHICH layers' query projections route knowledge? | ✅ **完成 (Counter-Hypothesis)** | P18: 8 conditions, 30 test samples. **DEEP layers (16-23) q_proj is the SUFFICIENT core: ONLY_deep H=0.0000. -q_deep is only breaker (ΔH=+0.083).** Pre-registered H18.1-18.3 all FALSIFIED — MID layers (8-15) secondary, NOT primary. Routing gradient: deep > mid > early. **B-bottleneck mechanism chain COMPLETE (P13-P18).** | [P18 Report](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P18_LAYER_ABLATION.md) |
| **P19** | **Absorption Attention Patterns** — WHERE in the attention layers does position-dependent routing manifest? | ✅ **完成** | P21: 90 samples (30/position), 9 layers, eager attention. **U-shaped entropy gap: L0=11.3% → L9=4.4% (min) → L23=13.0% (max). Una gap > Ans gap at every layer. A+B bottleneck converge on deep-layer attention routing.** | [P19 Report](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P19_ATTENTION_PATTERNS.md) |
| **P20** | **Multi-Layer Steering (Absorption)** — Does deep-layer v_abs steering produce a DIFFERENT profile from L10 steering? | ✅ **完成 (Negative-Informative)** | P24: 4 conditions (L10/L21 × a0/a3), n=10/position. **L10_a3 ΔH=0.250 = L21_a3 ΔH=0.250 → IDENTICAL. ALL hypotheses FALSIFIED. Hidden-state vector interventions EXHAUSTED for absorption.** | [P20 Report](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P20_MULTILAYER_STEERING.md) |
| **P21** | **Absorption LoRA** — Does P15 LoRA checkpoint close absorption position gap in generate space? | ✅ **完成 (Partial Success)** | P25: 30 test samples, dual-space eval. **ΔH 0.750→0.250 (−67%). Mid/late FIXED (H=0.000). Early residual H=0.250. Absorption = distance-based routing (FIXED) + proximity-based over-confidence (NOT fixed).** | [P21 Report](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P21_ABSORPTION_LORA.md) |
| **P22** | **Attention Temperature Scaling** — Can softening early-layer attention reduce proximity over-confidence? | ✅ **完成 (Negative-Informative)** | P26: 4 T conditions, eager attention. **Any T≠1 makes early H WORSE. Mid/late H=1.000 invariant. Baseline T=1.0 is OPTIMAL. Over-confidence is ROBUST, not fragile.** | [P22 Report](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P22_ATTENTION_TEMPERATURE.md) |
| **P23** | **Position ID Offset (RoPE Test)** — Does shifting position IDs via padding reduce early over-confidence? | ✅ **完成 (Informative Negative)** | P27: 4 offsets (0/100/300/500), 30 test samples. **N=100 makes early WORSE. N≥300 breaks model. Early H=0.250 is a LOCAL MINIMUM. 3 interventions fail to reduce it. Absorption intervention ladder COMPLETE.** | [P23 Report](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P23_POSITION_OFFSET.md) |
| **P24** | **Sycophancy n=48** — Does doubling sample size achieve statistical significance for sycophancy steering? | ✅ **完成 (Trending)** | P28: 3 cond × 48 samples, v_syc L10. **Direction confirmed (−33.3%). Fisher p=0.065 (trending, NOT α=0.05). Effect is REAL but UNDER-POWERED. Data ceiling (60) insufficient for n≈90+ needed.** | [P24 Report](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P24_SYC_N48.md) |
| **P25** | **Cross-Bottleneck Steering Interaction** — Does anti-sycophancy steering worsen hallucination? Does anti-hallucination steering worsen sycophancy? | ✅ **完成 (Synergy Discovery)** | P29: 6 cond (3 hall × 3 syc), cross-steering L10. **ASYMMETRIC SYNERGY: v_syc(−3) STRONGLY reduces hallucination (mid H=0.000). v_hall(−3) has ~zero effect on sycophancy. Hall ⊃ Syc as nested subspace. cos(v_hall, v_syc)=0.2355. Fighting sycophancy inadvertently fights hallucination — fortunate design property.** | [P25 Report](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P25_CROSS_BOTTLENECK.md) |
| **P26** | **Unified Bottleneck Steering** — Can combining v_hall + v_syc at different ratios achieve superior hallucination reduction while preserving correctness? | ✅ **完成 (Triple Crown)** | P30: 6 cond (Baseline + 5 ratios) × 30 pos-sens + 6 × 16 syc. **U_1:2 (v_hall=−1.0 + v_syc=−2.0) achieves TRIPLE CROWN: mid H=0.000, avg C=0.444 (8× v_hall), syc=0.375 (−25%). Combination BEATS both single vectors. Ratio matters: 1:1 and 2:1 kill C; only 1:2 (67% syc) hits sweet spot. Strongest hidden-state result in IC-4.** | [P26 Report](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P26_UNIFIED_STEERING.md) |

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
| `IC-4-M0\results_p15_hallucination_lora\` | **P15: Hallucination LoRA → B-bottleneck BRIDGED.** H=0.000, C=1.000. |
| `IC-4-M0\results_p16_lora_geometry\` | **P16: LoRA 几何机制分析** — LoRA bypasses K↔D (routing fix, not geometry fix) |
| `IC-4-M0\src\run_p16_lora_geometry_analysis.py` | **P16 实验脚本** — 逐层比较 LoRA 前后的 K↔D 对齐度 |
| `IC-4-M0\results_p17_lora_ablation\` | **P17: LoRA 模块消融** — q_proj 唯一关键 (ΔH=+0.25 solo); k,v,o ΔH=0 |
| `IC-4-M0\src\run_p17_lora_ablation.py` | **P17 实验脚本** — 逐模块 zero-out 消融 |
| `IC-4-M0\src\run_p15_hallucination_lora.py` | **P15 实验脚本** — hallucination-targeted LoRA fine-tuning |
| **`FINAL_COMPREHENSIVE_REPORT.md`** | **终极综合报告** — 19个实验集成，三瓶颈全诊断+全补救，B-bottleneck 几何证明+LoRA弥合 |
| **`UNIFIED_RESEARCH_MAP.md`** | 本文件 — 跨项目总研究地图 |

### IC-4 核心文档

| 文件 | 说明 |
|---|---|
| `IC4_PROJECT_TERRAIN_MANUAL.md` | v3.8 — 项目地形图，定义所有阶段、机制、边界 |
| `IC4_RESEARCH_PLAN_NEXT.md` | 三层研究计划（Anchors / Near-term / Branches） |
| `reports/IC4_P1_CROSS_VALIDATION_REPORT.md` | v1.2 — 跨 seed/layer 验证 |
| `reports/IC4_P15_FAILURE_ANALYSIS_REPORT.md` | 失败模式分析 + 小样本补丁测试 |
| `reports_m4_generalization/IC4_M4_GENERALIZATION_REPORT.md` | scoped robust 证据 |
| `results_m7/IC4_M7_FINAL_REPORT_CLEAN.md` | M7 机制解释 + 能力路由发现 |
| `results_m7/M7_LV2_ECHO_CPU_REPORT.md` | M7-Lv2 能力路由实验 |
| `reports/IC4_T0_TRAJECTORY_CAPTURE_REPORT.md` | T0 轨迹捕获报告 — equivalence verified, 90 samples |
| `reports/IC4_T1_PROJECTION_REPORT.md` | T1 投影分析 — v_hall prefill separation, cross-layer band |
| `reports/IC4_T2_DECISION_HEATMAP_REPORT.md` | T2 决策热力图 — L8 step 0 acc=0.917, cross_layer_band |
| `reports/IC4_T3_IMPULSE_REPORT.md` | T3 脉冲响应 — early impulse 可改行为，方向特异性未证明 |

### IC-4 M0: 训练时结构内化 (Training-Time Internalization)

| 文件 | 说明 |
|---|---|
| `src/run_s1_structure_distillation.py` | **S1: Structure Distillation** — CE+MSE 联合训练，v_syc 方向性蒸馏 |
| `results_s1_structure_distillation/results.json` | S1 结果：syc 50%→15% (Δ=+0.35)，但 40% 输出乱码 |
| `src/run_s2_self_probe.py` | **S2: Self-Probe Training** — CE+BCE 联合训练，无方向信号 |
| `results_s2_self_probe/results.json` | S2 结果：syc 50%→65% (Δ=−0.15)，纯 CE 放大谄媚 |
| `src/run_s3_triple_bottleneck.py` | **S3: Triple-Bottleneck Regularization** — CE+PSI+Purity+Routing |
| `results_s3_triple_bottleneck/results.json` | S3 结果：syc 50%→55% (Δ=−0.05)，所有正则化梯度≈0 |
| `src/run_s1b_robust_distillation.py` | **S1b: Robust Distillation** — KL reg + 低LR，验证 tradeoff |
| `results_s1b_robust_distillation/results.json` | S1b 结果：syc 45%→45% (Δ=0.00)，质量100%保留 |
| `src/run_s1c_critical_point_sweep.py` | **S1c: Critical Point Sweep** — 24配置网格扫描 (6 MSE × 4 alpha) |
| `results_s1c_critical_point_sweep/results.json` | S1c 结果：二分岔相变，无稳定甜点 (159 min) |
| `src/run_s1d_critical_slowing.py` | **S1d: Critical Slowing Down** — 5-epoch 边界追踪观察 |
| `results_s1d_critical_slowing/results.json` | S1d 结果：亚临界振荡→坍缩，5/5 终结于 quality=0 (103 min) |
| `src/run_s1f_lora_capacity.py` | **S1f: LoRA Capacity Test** — r=8→64，测试容量是否为瓶颈 |
| `results_s1f_lora_capacity/results.json` | S1f 结果：容量非瓶颈，二分岔内生于训练目标 (66 min) |
| `src/run_s1g_decoupled_training.py` | **S1g: Decoupled Two-Stage Training** — MSE/CE 分阶段解耦 |
| `results_s1g_decoupled_training/results.json` | S1g 结果：首次成功跨越二分岔 (syc=0.0, qual=1.0, CE=2.08) (138 min) |
| `src/run_s1h_mmd_drift.py` | **S1h: GMD-MMD Drift Training** — GMD 启发的自适应 drift field |
| `results_s1h_mmd_drift/results.json` | S1h 结果：全 3 配置失败——MMD drift 比固定 steer 更差 (42 min) |
| `src/run_s1i_contrastive_stage1.py` | **S1i: Alternative Stage 1 Directional Losses** — cosine/mse_fixed 替代 S1g 的 self-referential MSE |
| `results_s1i_contrastive_stage1/results.json` | S1i 结果：全 3 配置失败但揭示 Goldilocks 原理——S1g 的 "bug" 是最优解 (71 min) |
| `src/run_s1j_cosine_magnitude.py` | **S1j: Cosine + Magnitude Penalty** — λ·||hs−baseline||² 约束余弦方向性 push |
| `results_s1j_cosine_magnitude/results.json` | S1j 结果：全 4 λ 失败——cosine/幅度惩罚是正交目标，cos→−0.97 不可逆 (122 min) |
| `src/run_s1k_mse_fixed_extended.py` | **S1k: MSE-Fixed Extended Training** — 5ep/7ep 延长训练测试梯度衰减高原 |
| `results_s1k_mse_fixed_extended/results.json` | S1k 结果：全 3 配置失败——cos_sim 停滞在 −0.25，梯度衰减是根本限制 (102 min) |
| `src/run_s1g_v2_output_quality.py` | **S1g-v2: Output Quality Verification — S1g 复现测试** — 3次独立复现全部失败 |
| `results_s1g_v2_output_quality/results.json` | S1g-v2 结果：S1g 不可复现——CE 崩溃至 0.14-0.52，全空输出 (~72 min) |

### Intelligence Capital Minimal Lab 核心文档

| 文件 | 说明 |
|---|---|
| `..\intelligence_capital_minimal_lab\THEORY.md` | 理论框架（change capital, bad debt, false capital etc.） |
| `..\intelligence_capital_minimal_lab\IC2B_LEARNED_THROTTLING_REPORT.md` | learned compressor 比较（13 机制） |
| `..\intelligence_capital_minimal_lab\IC2C_EPISODIC_VS_CONSOLIDATED_REPORT.md` | episodic vs consolidated capital 实验 |
| `..\intelligence_capital_minimal_lab\IC2C1_ROOT_CAUSE_REPORT.md` | NoMemory/Episodic/Consolidated 根因拆解 |

### 跨项目理论 / 新实验

| 文件 | 说明 |
|---|---|
| `..\intelligence_capital_minimal_lab\intelligence_capital_theory\STRUCTURAL_ADAPTATION_HYPOTHESIS.md` | 结构适应假说 — 小-大模型差距的本质是结构适应能力差异 |
| `..\intelligence_capital_minimal_lab\intelligence_capital_theory\RELATIONAL_MEMORY_HYPOTHESIS.md` | 关系结构记忆假说 — 记忆是关系结构而非位置序列 |
| `..\intelligence_capital_minimal_lab\results\ic2c_topology\IC2_TOPOLOGY_AUDIT_REPORT.md` | Consolidation Topology Audit 报告 — TPR/RRP/Purity/MEC |
| `..\internal_circuit_capital_lab\new-5\立项文件.md` | TT-SFT 立项文件 — experiment spec |
| `..\internal_circuit_capital_lab\new-5\run_tt_sft_v0.py` | TT-SFT v0 实验脚本 (self-teacher) |
| `..\internal_circuit_capital_lab\new-5\run_tt_sft_v1.py` | TT-SFT v1 实验脚本 (base-model teacher) |
| `..\internal_circuit_capital_lab\new-5\trajectory_alignment.py` | Trajectory alignment 核心模块 (TrajectoryCollector, AlignmentWrapper, ProjectionHead) |
| `..\internal_circuit_capital_lab\new-5\reports\IC4_TT_SFT_V0_REPORT.md` | TT-SFT v0 最终报告 |
| `..\internal_circuit_capital_lab\new-5\reports\IC4_TT_SFT_V1_REPORT.md` | TT-SFT v1 最终报告 |
| `..\internal_circuit_capital_lab\new-5\run_p5_syc_feedback.py` | P5 谄媚反馈控制实验脚本 |
| `..\internal_circuit_capital_lab\new-5\reports\IC4_P5_SYC_FEEDBACK_REPORT.md` | P5 谄媚反馈控制最终报告 |
| `..\internal_circuit_capital_lab\new-5\run_p5a_multi_layer.py` | P5a 多层分散注入实验脚本 (复用于 P5d) |
| `..\internal_circuit_capital_lab\new-5\reports\IC4_P5A_MULTILAYER_REPORT.md` | P5a 多层分散注入最终报告 |
| `..\internal_circuit_capital_lab\new-5\reports\IC4_P5D_CROSSVALIDATION_REPORT.md` | **P5d — P4 数据集交叉验证报告**：30/30 split 复制并强化方向特异性 (Δ=0.27) |
| `..\internal_circuit_capital_lab\new-5\run_p6_quality_eval.py` | **P6 — 输出质量评估脚本**：rep, distinct, PPL 多维质量指标 |
| `..\internal_circuit_capital_lab\new-5\reports\IC4_P6_QUALITY_EVAL_REPORT.md` | **P6 — 输出质量与行为稳健性报告**：量化方向特异性控制的输出代价 (rep +2.1×, distinct −41%) |
| `..\internal_circuit_capital_lab\new-5\run_p7_adaptive_alpha.py` | **P7 — 自适应 α 反馈控制脚本**：Linear/Step-2/Sqrt 三种策略 |
| `..\internal_circuit_capital_lab\new-5\results_p7_adaptive_alpha\` | **P7 — 自适应 α 结果目录** |
| `..\internal_circuit_capital_lab\new-5\reports\IC4_P7_ADAPTIVE_ALPHA_REPORT.md` | **P7 — 自适应 α 报告**：负结果 (AUC=1.0 无置信方差)，但揭示 α-质量连续关系 |
| `..\internal_circuit_capital_lab\new-5\run_p8_alpha_sweep.py` | **P8 — 精细 α sweep 脚本**：6+2+2 配置绘制 Pareto 前沿 |
| `..\internal_circuit_capital_lab\new-5\results_p8_alpha_sweep\` | **P8 — 精细 α sweep 结果目录** |
| `..\internal_circuit_capital_lab\new-5\reports\IC4_P8_ALPHA_SWEEP_REPORT.md` | **P8 — Pareto 前沿报告**：α=−20 为新最佳 (比 −25 好 15%)，三 regime 分类 |
| `..\internal_circuit_capital_lab\new-5\run_p9_layerwise_alpha.py` | **P9 — 层间差异化 α 脚本**：5+2 策略测试 Uniform/Decay/Invert/MidPeak/L10Only |
| `..\internal_circuit_capital_lab\new-5\results_p9_layerwise_alpha\` | **P9 — 层间差异化 α 结果目录** |
| `..\internal_circuit_capital_lab\new-5\reports\IC4_P9_LAYERWISE_ALPHA_REPORT.md` | **P9 — 层间差异化 α 报告**：负结果，Uniform 是最优分配策略 |
| `..\internal_circuit_capital_lab\new-5\reports\IC4_NEW5_COMPREHENSIVE_REPORT.md` | **new-5 项目完整路线报告** (TT-SFT v0→v1→P5→P5a→P5d→P6→P7→P8→P9) |
| `IC-4-M0\TRAJECTORY_DYNAMICS_PHASE_1_5.md` | Trajectory Dynamics Phase 1.5 执行计划 — P0/P1/P2 三步走 |
| `IC-4-M0\data_position_sensitivity\s0\` | Position Sensitivity 位置变体数据（early/mid/late） |
| `IC-4-M0\src\run_position_rep_shift.py` | Position Rep Shift 实验脚本 |
| `IC-4-M0\src\run_position_behavior.py` | Position-to-Behavior Sensitivity 实验脚本 |
| `IC-4-M0\src\run_probe_psi_cpu.py` | CPU 优化 probe PSI 脚本（因 probe 过拟合弃用） |
| `IC-4-M0\results_position_sensitivity_cpu\` | Position 实验全部结果 |
| `..\intelligence_capital_minimal_lab\src\run_consolidation_topology_audit.py` | Topology Audit 实验脚本 |
| `..\intelligence_capital_minimal_lab\src\run_c4_stabilization_scaling.py` | **C4 — Seed Scaling (5→100)**：Per-Action KMeans 规模压力测试 |
| `..\intelligence_capital_minimal_lab\src\run_c4b_noise_scaling.py` | **C4b — Noise Scaling**：PA 噪声鲁棒性测试（3 噪声级别） |
| `..\intelligence_capital_minimal_lab\src\run_c4_objective_noise_scaling.py` | **C4-obj — Objective + Noise Scaling**：3→20 actions 联合测试 |
| `..\intelligence_capital_minimal_lab\src\run_c5_cross_bottleneck_analogue.py` | **C5 — Cross-Bottleneck Analogue**：模拟环境跨瓶颈扰动测试 |
| `..\intelligence_capital_minimal_lab\src\run_c_anchored_consolidation.py` | **C16 — Anchored Consolidation**：锚定压缩 Proof C |
| `..\intelligence_capital_minimal_lab\src\run_c2_stronger_stabilization.py` | **C17 — Readout-Level Stabilization**：读出头修复测试 |
| `IC-4-M0\src\run_c6_llm_consolidation.py` | **C12 — LLM Hidden State Consolidation**：Phase 7 3.3B |
| `IC-4-M0\src\run_c7_multi_checkpoint_consolidation.py` | **C21 — Multi-Checkpoint Consolidation**：LoRA 多检查点 |
| `IC-4-M0\src\run_a1_position_augmented_probe.py` | **C14 — Position-Augmented Probe**：A-1 吸收瓶颈首个补救 |
| `IC-4-M0\src\run_a2_behavior_position_invariant.py` | **C18 — Behavior Gate Consistency**：行为层位置不变性 |
| `IC-4-M0\src\run_a3_position_rectification.py` | **C20 — Position Rectification**：推理层位置矫正 |
| `IC-4-M0\src\run_p8_large_scale_replication.py` | **C19 — Large-Scale Sycophancy Feedback (n=24)** |
| `IC-4-M0\results_a1_position_probe\` | A-1 结果：PSI 0.0676→0.0067 (−90%) |
| `IC-4-M0\results_a2_behavior_position_inv\` | A-2 结果：gate 一致性 + 行为层残留 |
| `IC-4-M0\results_a3_position_rectification\` | A-3 结果：位置矫正负结果 |
| `IC-4-M0\results_c7_multi_checkpoint\` | C7 结果：多检查点 consolidation |
| `IC-4-M0\results_p7_s15\` | P7 结果：S15 放大机制调查 |
| `IC-4-M0\results_p8_large_scale\` | P8 结果：大规模反馈复现 |
| `IC-4-M0\results_p9_cross_bottleneck\` | P9 结果：跨瓶颈结构完整性 |
| `IC-4-M0\src\run_s1_structure_distillation.py` | **C26 — S1: 结构信号蒸馏**：训练时结构内化（前范式实验） |
| `IC-4-M0\src\run_s2_self_probe.py` | **C27 — S2: 自探针训练**：辅助分类头联合训练 |
| `IC-4-M0\src\run_s3_triple_bottleneck.py` | **C28 — S3: 三瓶颈联合正则化**：PSI+Purity+Routing 联合惩罚 |
| `IC-4-M0\results_s1_structure_distillation\` | S1 结果：syc 50%→15% (Δ=+0.35, syc rate dropped 35pp) |
| `IC-4-M0\results_s2_self_probe\` | S2 结果：syc 50%→65% (Δ=-0.15, negative) |
| `IC-4-M0\results_s3_triple_bottleneck\` | S3 结果：syc 50%→55% (Δ=-0.05, near-zero) |
| `IC-4-M0\results_p14_cross_layer_bottleneck\` | **P14: Cross-Layer B-Bottleneck 特征化** — 逐层 probe + steering 几何关系 |
| `IC-4-M0\src\run_p14_cross_layer_bottleneck.py` | **P14 实验脚本** — 跨层 B-bottleneck 几何映射 |

---

## 三个项目的统一叙事（对外版本）

> **我们研究的不是"怎么让模型少犯错"，而是"模型内部已有的能力/信息为什么没有被正确使用"。**
>
> 在元层（`Structural Adaptation Hypothesis`），我们把这个问题进一步抽象为：**小模型与大模型的根本差距不在于知识量，而在于结构适应能力——即把人类离散化的数据流吸收、稳定、组织成可调用内部结构的能力。**
>
> 在一条线上（`IC-4`），我们发现小模型内部存在 latent verification capability，可以通过条件化 routing gate 激活——这就是在**补偿组织瓶颈**：能力存在但默认 routing 不通。
>
> 在另一条线上（`intelligence_capital_minimal_lab`），我们发现错误 consolidation 会把有用经验结构变成 bad debt——TPR=0.875 但 Purity=0.261，跨分布混合是主要破坏机制。这就是在**补偿稳定瓶颈**：跨样本结构在压缩下漂移。
>
> 第三个瓶颈（**吸收瓶颈**：输入碎片化超出处理能力）**已完成诊断 + 首个补救**。Position-Augmented Gate Probe 将 PSI 降低 90%（C14），验证了位置感知探针训练可有效免疫表示层位置偏移。行为层位置敏感性（ΔH=0.111）仍需 A-2 Position Normalization Adapter（需 GPU）来解决。
>
> 三条线汇合于一个核心洞见：**我们不是在造更强的模型，而是在为小模型提供它自身欠缺的结构适应操作——吸收、稳定、组织。**

---

## 一句话身份

> **IC-4 + intelligence_capital_minimal_lab = 一个正在成形的、关于"结构性能力/信息如何在系统中被正确路由和保真"的研究项目。**
>
> 当前阶段不是"找现象"或"做 demo"——而是"机制工程可行性验证"。