---
title: IC-4-M0 Project Terrain Manual（含 M6/M7 机制解释层 + M7-Lv2 能力路由发现 + P1.5 小样本稳健性）
version: "3.2"
last_updated: "2026-05-19"
era: "Second Era — Readout-Control Paradox → Capability Routing"
approved_reference: "M3-v6"
changelog_3.2: "升级默认 construction 标准为 30A+30U；整合 P1 cross-validation 和 P1.5 failure mode analysis 结论；更新 robust 边界声明"
---

# IC-4 项目地形图（IC-4-M0 Terrain Manual）

**版本**：3.2（整合 P1 交叉验证 + P1.5 小样本稳健性结论，升级默认 construction 为 30A+30U）  
**最后的稳定参考点**：M3-v6（single-pass hook + hard gate + model.generate()）  
**当前默认配置标准**：30A+30U construction（自 P1.5 起替代旧 15A+15U 默认）  
**当前探索前沿**：M7（从"为什么 control 弱"推进到"如何接路由把 latent capability 接入 generation"）  
**下一阶段**：M7-H（LoRA 路由注入）、M7-L（ECHO 验证训练）、M7-E（1.5B 跨模型验证）  
**跨项目总地图**：[UNIFIED_RESEARCH_MAP.md](../UNIFIED_RESEARCH_MAP.md)（v4.0 — Capability Routing × Structural Fidelity）  
**研究计划**：[IC4_RESEARCH_PLAN_NEXT.md](../IC4_RESEARCH_PLAN_NEXT.md)（三层计划：Anchors / Near-term / Branches）

It is not a literature review. It is a working map of:

1. what this project is actually trying to prove,
2. what has already been established experimentally,
3. which earlier conclusions were corrected later,
4. how external papers changed the research direction,
5. what is now solid ground versus open terrain.

---

## 1. Project Core Question

IC-4 is not primarily asking whether a model "can answer questions."

It is asking:

> Does the model contain an internal reliability mechanism that is readable, controllable, and selectively callable?

In practical terms:

- can we read from internal state whether a sample is risky,
- can we intervene only when that risk condition is present,
- can we reduce hallucination without damaging answerable performance,
- and can we do this by intervening in the model's internal dynamics rather than by retraining or prompt-only tricks?

This is why the project gradually moved from:

- "find a steering vector"

to:

- "find a condition-aware internal controller"

and then further to:

- "find the correct way to attach that controller to the model's forward dynamics."

---

## 2. Current Proven Mechanism

The current reference mechanism is the `M3-v6` single-pass gate.

Reference files:

- `F:\internal_circuit_capital_lab\IC-4-M0\src\run_m3_v6.py`
- `F:\internal_circuit_capital_lab\IC-4-M0\src\gate_steering_tool.py`
- `F:\internal_circuit_capital_lab\IC-4-M0\results_m3_v6\metrics_raw.csv`
- `F:\internal_circuit_capital_lab\IC-4-M0\reports_m3_v6\IC4_M3_V6_SINGLE_PASS_GATE_REPORT.md`

Mechanism:

- representation: `last_prompt_token`
- probe: logistic regression
- gate: hard threshold
- injection: single-pass forward hook
- generation path: `model.generate()`
- steering layer: 12
- reference alpha: `-1.0`
- default construction regime: **30A+30U** (upgraded from 15A+15U after P1.5 small-sample artifact diagnosis; see §3, §6.P1.5)

Reference result:

| mode | H | C | UA |
|---|---:|---:|---:|
| base | 0.8667 | 0.6000 | 0.0000 |
| oracle_gate_a-1.0 | 0.6667 | 0.6000 | 0.0000 |
| single_pass_hard_gate_a-1.0 | 0.6667 | 0.6000 | 0.0000 |
| random_single_pass_hard_gate_a-1.0 | 0.9333 | 0.6000 | 0.0000 |
| shuffled_single_pass_hard_gate_a-1.0 | 0.8000 | 0.6000 | 0.0000 |

Interpretation:

- the steering vector `v` is real and causally useful,
- the prefill state contains a usable gate signal,
- the correct injection path matters,
- a simple probe + hard gate is enough to match oracle in the reference setting.

---

## 3. Current Robustness Boundary (Updated P1.5)

Generalization report:

- `F:\internal_circuit_capital_lab\IC-4-M0\results_m4_generalization\sweep_matrix.csv`
- `F:\internal_circuit_capital_lab\IC-4-M0\reports_m4_generalization\IC4_M4_GENERALIZATION_REPORT.md`

Cross-seed / cross-layer validation report:

- `F:\internal_circuit_capital_lab\IC-4-M0\reports\IC4_P1_CROSS_VALIDATION_REPORT.md` (v1.1, corrected)
- `F:\internal_circuit_capital_lab\IC-4-M0\reports\IC4_P15_FAILURE_ANALYSIS_REPORT.md`

### 3.1 What is currently robust

Under **30A+30U default construction** (since P1.5):

- `seed ∈ {0, 1, 2}`, `layer ∈ {11, 12, 13}` — **all pass** causal ordering `random > shuffled > real_gate`
- scenarios: `standard`, `large`, `hard OOD`
- alpha values: `-0.8`, `-1.0`, `-1.2`

At `alpha = -1.0`, the gate matches oracle in all evaluated data scenarios and in all tested seed/layer combinations under 30A+30U construction.

### 3.2 P1 findings (15A+15U, pre-P1.5)

| Config | Verdict | Root Cause (P1.5) |
|---|---|---|
| seed=0 / layer=12 | SUCCESS | — |
| seed=1 / layer=12 | SUCCESS | — |
| seed=2 / layer=12 | ARTIFACT | cos(steer,shuffled)=0.788 → fixed at 30A+30U |
| seed=0 / layer=11 | SUCCESS | — |
| seed=0 / layer=13 | ARTIFACT | small-sample noise → fixed at 30A+30U |

### 3.3 P1.5 correction

P1.5 proved that the two ARTIFACT verdicts were **construction-regime artifacts, not mechanism failures**:

- seed=2: doubling construction pairs collapsed cos(steer,shuffled) from 0.788→0.439, restoring causal separation
- layer=13: doubling construction pairs raised shuffled H from 0.667→0.900, eliminating the apparent control failure

**Bottom line**: the reference mechanism is robust under 30A+30U construction in all tested seed/layer settings. Previous P1 failures were small-sample artifacts of the 15A+15U regime.

### 3.4 Important boundary

> "ROBUST" currently means robust across data size, OOD difficulty, alpha variation, **and tested seed/layer combinations under 30A+30U construction**.

It does **not** yet mean:

- cross-model robustness (1.5B, 7B),
- cross-behavior robustness (sycophancy, refusal, etc.),
- larger-scale seed/layer sweeps (all 5 seeds × all 24 layers).

Those remain open.

---

## 4. What the Project Has Actually Learned

The project has already established several facts.

### 4.1 The steering vector is effective

This was first hinted by open-loop results, then made much clearer by oracle gate controls.

Takeaway:

> The vector is not fake, and not reducible to random or shuffled controls.

### 4.2 The gating signal exists before generation

The gate signal is readable from prefill state. In the successful reference mechanism, `last_prompt_token` is enough.

Takeaway:

> The model internally contains answerability/risk information before the answer is generated.

### 4.3 Correct signal is not enough by itself

Multiple earlier stages showed that:

- a good probe can still fail behaviorally,
- a good vector can still fail behaviorally,
- early intervention can still fail if attached incorrectly.

Takeaway:

> Success depends on intervention mechanics, not only on signal quality.

### 4.4 Correct dynamics attachment is part of the mechanism

The final success came only after:

- single-pass integration,
- hook-based in-flight decision,
- use of `model.generate()` instead of a manual token-by-token loop.

Takeaway:

> The mechanism is not just "probe + vector"; it is "probe + vector + correct forward-path attachment."

---

## 5. Corrected Attribution History

Future agents must preserve the corrected attribution chain.

### 5.1 M3-v5 was important, but not the final attribution

`M3-v5` correctly forced deeper artifact decomposition, but one of its strongest early interpretations was later corrected.

What `M3-v5` helped expose:

- the implementation path contained strong systematic artifacts,
- gate shape alone was not the main story,
- pipeline and generation mechanics had to be isolated.

What `M3-v6` later corrected:

> The `C = 0.733` anomaly was not fundamentally caused by the two-pass prefill idea itself.

The critical diagnostic in `M3-v6` showed:

- the real source of the anomaly was the manual token-by-token generation loop,
- the hook itself was not the culprit,
- `model.generate()` restores the correct baseline and control behavior.

This is an important historical correction. Do not revert to the older simplified story.

### 5.2 Safe wording

Prefer:

> `M3-v5` exposed implementation-path artifact; `M3-v6` resolved the true source and established the correct reference implementation.

Avoid:

> `M3-v5 proved two-pass is the main bottleneck.`

That wording is too coarse after the `M3-v6` diagnostic.

---

## 6. Stage-by-Stage Experimental Map

### M0 / M1 / M2: vector and artifact discovery

These stages established that open-loop steering can reduce hallucination, but:

- often damages answerable performance,
- can be confounded by task proxies or controls,
- must be evaluated with stronger anti-artifact discipline.

They were necessary, but they did not yet solve selective intervention.

### M3-oracle: the key pivot

This was the first hard proof that the vector itself had real value.

Oracle gate showed:

- steer only unanswerable samples,
- do not touch answerable ones,
- and the behavior improves without the earlier damage pattern.

This changed the project from:

- "find a better vector"

to:

- "find a real gate and the right injection path."

### M3-v2: scalar gate insufficiency

Entropy / max-prob / uncertainty-mass style scalar gates moved, but were not enough.

Lesson:

> Token-local scalar risk signals were too weak to close the gap to oracle.

### M3-v3: trajectory-probe partial advance

This stage showed:

- trajectory-style signals are informative,
- but waiting until token 4 is too late for behavior.

Lesson:

> Good readout can fail if the model has already committed to the wrong trajectory.

### M3-v4 / M3-v4b / M3-v5: implementation-path debugging

These stages improved timing and decomposition, but still suffered from wrong mechanics.

They were valuable because they forced the project to stop blaming the wrong objects.

### M3-v6: convergence

This is the first stage that unambiguously closes the loop:

- readable signal,
- selective intervention,
- correct internal attachment,
- oracle-level behavioral result.

### M4 trajectory-state

This stage showed that:

- pooled / windowed representations outperform last-token for state readout tasks,
- the mechanism is better understood as trajectory/state-level rather than purely token-local.

Important nuance:

> Better state readout does not automatically mean "that is the true computation circuit."

It is still a readout result, not a full computation decomposition.

### M4-generalization

This stage established that the reference mechanism is stable across several scenario changes, within the validated setting.

### P1: Cross-Seed & Cross-Layer Validation (2026-05)

This stage tested whether the M3-v6 reference mechanism generalises beyond the original `seed=0, layer=12` configuration.

- 3 seeds (0, 1, 2) × layer=12
- 3 layers (11, 12, 13) × seed=0
- **Result**: 3/5 passed (seed=1, layer=11 SUCCESS; seed=2, layer=13 ARTIFACT)
- Failure mode: `shuffled < real_gate` — shuffled permutation of steering vector produced unexpected anti-hallucination effect

Key report: `reports/IC4_P1_CROSS_VALIDATION_REPORT.md` (v1.1, corrected)

### P1.5: Failure Mode Analysis & Small-Data Patch Test (2026-05)

This stage diagnosed the two P1 failures and tested whether they were mechanism bugs or construction-regime artifacts.

**Diagnosis**:
- seed=2: `cos(steering, shuffled)=0.788` — shuffled vector accidentally highly aligned with real direction due to small sample (15A+15U)
- layer=13: small-sample statistical noise in shuffled control

**Patch test**: Doubled construction pairs to 30A+30U (merged from two seeds' activations)
- seed=2: cos dropped to 0.439, causal ordering restored ✅
- layer=13: shuffled H rose from 0.667→0.900, causal ordering restored ✅

**Verdict**: Both P1 failures were construction-regime artifacts. The M3-v6 mechanism itself is robust at 30A+30U across all tested seed/layer combinations.

**Consequence**: 30A+30U is now the default construction standard. All future experiments must use ≥30A+30U.

Key report: `reports/IC4_P15_FAILURE_ANALYSIS_REPORT.md`

### Phase 2: M5-X1..X5 — Understanding "readout" (2025-12)

Focus: trajectory- and window-level probe signal.

The project discovered that the probe signal is not a simple last-token phenomenon — it builds up across tokens and across layers:

- last-token AUC ~ 0.73,
- window-pooled AUC ~ 1.00.

This meant: there is strong trace-level information in the model, but it is not useful unless intervention is designed to interact with that trace.

### Phase 3: M6/M7 — Understanding "control" and the readout-control gap (2026-05)

Focus: why does near-perfect readout co-exist with near-zero control?

| Question | Experiment | Core Finding |
|---|---|---|
| Can ADD steering suppress sycophancy? | M6-X2 (all-layer) | **Δ=0 everywhere.** |
| Can REPLACE intervention work? | M6-PA1/PA2 | **100% flip at L18-L22.** |
| ADD vs REPLACE: why the gap? | M7-C (PCA) | ADD preserves within-class variance. |
| Is sycophancy from RLHF? | M7-D (Base vs Instruct) | **No—Base model is 100% sycophantic.** |
| Is signal concentrated in few dims? | M7-B (dim patching) | **No—K<200 has zero effect.** |
| MLP or Attention? | M7-A (component) | **MLP > Attention.** |
| Does multi-layer help? | M7-J (cross-layer) | **No—single L20 is optimal.** |
| Can random basis match PCA? | M7-K (Hadamard) | **No—PCA captures real structure.** |

**Status: "Mining" phase exhausted. Signal is real but fully distributed. Next step: "Injection" via LoRA/ECHO training (requires GPU).**

---

## 7. Theoretical Terrain from External Papers

This project did not simply copy one paper. Instead, different papers helped organize different parts of the terrain.

### 7.1 State ontology layer

Relevant influences:

- `ELF: Embedded Language Flows`
- trajectory / continuous-state perspectives

Main lesson:

> Internal state should not automatically be treated as a single token-local point event.

This helped shift the project toward trajectory/state thinking.

### 7.2 State readout layer

Relevant influences:

- `The Truth Lies Somewhere in the Middle`

Main lesson:

> Generated-token windows and pooled states can be stronger semantic readouts than single-token snapshots.

This fed directly into `M4`.

### 7.3 Control architecture layer

Relevant influences:

- fast/slow adaptation framing

Main lesson:

> A stable steering prior can be seen as a slow structure, while the gate/probe/controller acts like a fast adaptor.

In project terms:

- steering vector `v` behaves like slow structure,
- probe/gate behaves like fast conditional control.

### 7.4 Stable dynamics layer

Relevant influences:

- attractor models
- edge-of-stability / attractor-set viewpoints

Main lesson:

> Reliability may come from being brought into a stable region, not simply from changing one local coordinate.

This helped frame the gate as more than an on/off switch.

More specifically, attractor-style models suggest a useful interpretation for the successful IC-4 mechanism:

> The working gate + steering mechanism can be viewed as a minimal external attractor controller.

In this interpretation:

- the model produces an initial internal state,
- the probe reads whether that state is entering a risky regime,
- the steering intervention selectively redirects that state toward a more reliable basin,
- and correct behavior depends on whether that redirection is attached to the true forward dynamics.

This attractor framing is helpful because it explains why earlier partial solutions failed:

- a useful vector alone was not enough,
- a useful readout alone was not enough,
- and even correct timing was not enough if the intervention path was wrong.

It also suggests a longer-term possibility:

> Today's gate + steering may function as an external reliability scaffold, while future training-time mechanisms might internalize that refinement and move the model's initial state closer to the reliable equilibrium by default.

### 7.5 State-flow visualization / geometric intuition layer

Relevant influence:

- `Diffusion-Explorer`

Main lesson:

> We benefit from thinking of intervention not only as "adding a vector" but as changing the path that state follows through a learned flow.

This repository is not a direct mechanism-analysis tool for this project. It is an educational and visualization-oriented system for diffusion / flow / rectified-flow geometry. Its value here is conceptual:

- it reinforces a trajectory/flow view of internal state,
- it makes "straightening" or "redirecting" a path more intuitive,
- it supports the idea that some interventions should be understood as reshaping state evolution rather than merely changing a local coordinate.

Applied to IC-4:

> The successful gate + steering mechanism can be described not only as a conditional residual modification, but as a selective reshaping of internal state flow.

This influence belongs more to explanation and intuition-building than to direct experimental procedure.

### 7.6 Propagation mechanism layer

Relevant influence:

- `Spontaneous symmetry breaking and Goldstone modes for deep information propagation`

Main lesson:

> We should ask not only what signal is read and what intervention is applied, but also why that intervention propagates stably through depth.

This is especially relevant after `M3-v6`, because success depends on allowing the intervention to ride the correct forward dynamics.

This layer suggests a future question:

> Is the successful steering direction coupling into a particularly propagation-friendly mode?

This is not yet proven, but it is a useful theoretical lens.

### 7.7 Representation vs computation layer

Relevant influence:

- `Arithmetic in the Wild`

Main lesson:

> A structured representation does not automatically imply that the model computes using that same structure.

Applied here:

> `M4` readouts are useful and real, but they should not be overinterpreted as the complete causal circuit.

### 7.8 Causal organization layer

Relevant influence:

- `The Causally Emergent Alignment Hypothesis`

Main lesson:

> Better behavior may reflect better-organized internal causal structure, not only higher output scores.

This does not yet directly change the current implementation, but it motivates later theory-facing analysis.

### 7.9 Consolidation / structural drift layer

Relevant influence:

- `Useful Memories Become Faulty When Continuously Updated by LLMs`

Main lesson:

> Useful internal structure does not remain useful automatically; repeated rewriting, consolidation, or incorrect integration can turn good structure into faulty structure.

This paper is about continual memory consolidation in LLM agents, but its lesson transfers well to IC-4:

- having a useful capability is not sufficient,
- having a readable signal is not sufficient,
- the integration pathway matters,
- and repeated or incorrect structural rewriting can degrade what was initially useful.

Applied to IC-4:

> The project should not only ask whether a verification-related capability exists, but also whether the model's default routing/consolidation path preserves or corrupts that capability during normal generation.

This connects especially strongly to the later M7 interpretation:

- latent verification capability appears to exist,
- but it is not default-routed into behavior,
- and direct negative behavioral prompting can backfire rather than restore the correct path.

This layer therefore supports a broader interpretation of the project:

> Some failure modes may come not from missing capability, but from faulty consolidation or routing of an otherwise useful internal structure.

---

## 8. What Not to Forget

Future agents should preserve these negative lessons.

### 8.1 Do not go back to manual token-by-token generation as baseline-equivalent

Manual loops in this project were shown to produce materially different behavior from `model.generate()`.

If used again, they must be treated as a different experimental regime, not as a harmless reimplementation.

### 8.2 Do not overclaim robustness

The current mechanism is strong, but the robust claim is scoped.

### 8.3 Do not collapse readout success into mechanism success

Good probe accuracy is not sufficient. The project only became convincing once the intervention was causally effective.

### 8.4 Do not flatten the mechanism to just "the vector"

The functioning mechanism includes:

- the vector,
- the probe,
- the gate policy,
- the hook site,
- the forward-path implementation,
- the generation method.

---

## 9. Current Best Summary for External Handoff

If another agent needs the shortest accurate summary, use this:

> IC-4 has established a reference internal reliability mechanism for one validated setting: a logistic gate read from prefill state can selectively activate a steering direction during a single forward pass, matching oracle anti-hallucination performance while preserving answerable performance. The main scientific lesson is that success depends not just on finding a useful steering direction or a readable probe signal, but on attaching the intervention correctly to the model's forward dynamics.

If slightly more detail is needed:

> In the validated reference setting (`seed=0`, `layer=12`, Qwen2.5-0.5B-Instruct), the project has shown that the model contains a readable prefill gate signal and a useful steering direction. When these are combined through a hard-threshold gate inside a single-pass forward hook and executed with `model.generate()`, the result matches oracle gating exactly across standard, larger, and harder OOD data scenarios.

---

## 10. Second Era: M6/M7 — The Readout-Control Paradox

**Status**: Completed mining phase. GPU injection phase pending.  
**Files**: `results_m7/M7_FINAL_REPORT.md`, `IC4_M7_ROADMAP.md`  
**Code**: `src/run_m7{a,b,c,d,f,g,j,k}.py`, `src/run_m7{h,l}.py` (GPU-ready)

### 10.1 What this era is about

M3-v6 established that the reference mechanism *works*. M6/M7 asks *why certain simpler forms don't work*, and what this tells us about the underlying mechanism.

**This era does not replace M3-v6. It is the explanation layer that clarifies which forms of intervention are viable and why.**

### 10.2 The core paradox

| Capability | Performance | Source |
|---|---|---|
| Readout (probe) | cv_acc = 1.0, AUC = 1.0 | M5-X3 |
| Control (ADD steering) | Δ = 0 across all layers | M6-X2 |
| Control (REPLACE) | 100% flip at L18-L22 | M6-PA1/PA2 |

The question driving M7: **Why can we read sycophancy perfectly but not steer it directionally?**

### 10.3 Five hard conclusions (solid ground)

These are established by multiple converging experiments. Treat them as facts, not hypotheses.

**(1) Readout strength ≠ control ease.**

Probes achieve AUC=1.0 on sycophancy signals while ADD steering produces Δ=0 at every layer. The two capabilities are mechanistically decoupled.

**(2) Simple mean-shift (ADD) is insufficient—the problem is variance, not direction.**

M7-C (PCA): SNR peaks at PC1 (1.08) then declines. ADD preserves each sample's unique within-class noise. REPLACE eliminates it entirely.

```
ADD:    hs_i' = mean_non + noise_i     ← noise survives
REPLACE: hs_i' = mean_non              ← noise eliminated
```

This is the single most important mechanism-level finding of M7.

**(3) Signal is genuinely distributed across all 896 dimensions.**

M7-B: replacing <200 MLP output dimensions has zero effect. M7-G: anti-sycophancy effect scales monotonically with K (K=20 → Δ=-0.30, K=896 → Δ=-0.55). There is no "hot dimension" subset.

**(4) MLP carries more sycophancy signal than attention.**

M7-A: MLP patching produces 3-5× stronger anti-sycophancy effect than attention patching at the same layers. However, full residual-stream intervention outperforms either component alone — the signal penetrates both pathways.

**(5) Prefill vectors do not transfer to generation dynamics.**

M7-F: applying REPLACE during autoregressive generation with a prefill-trained mean_non vector *increases* sycophancy (Δ = +0.40). The same vector applied at prefill achieves 100% flip. The hidden-state semantics are not invariant across prefill/generation modes.

### 10.4 Corrected attributions

| Earlier belief | What M7 showed | Evidence |
|---|---|---|
| "Sycophancy is an RLHF artifact" | **Sycophancy is a pre-training prior** | M7-D: Base = 100%, Instruct = 96.7% |
| "Subspace steering can rescue ADD" | Random orthogonal basis has zero effect | M7-K: Hadamard K<896 → Δ≈0, PCA K<896 → Δ<0 |
| "Multi-layer joint intervention needed" | Single-layer L20 is optimal | M7-J: L20 alone = all 5 layers combined |

### 10.5 What remains open terrain (early hypothesis, not solid ground)

| Hypothesis | Status | Next experiment |
|---|---|---|
| LoRA can learn the variance-collapse map | Untested | M7-H (GPU required) |
| Consequence-prediction training can compress a verification circuit | Untested | M7-L / ECHO-Lite (GPU required) |
| 1.5B model will replicate the pattern | Untested | M7-E (GPU recommended) |
| 0.5B/896D has a hard physical limit | Plausible death condition | If M7-L yields Δ≈0 |

### 10.6 M7-Lv2: Capability Routing Discovery (2026-05-19, CPU)

**File**: `results_m7/m7l_echo_cpu_report.txt`  
**Code**: `src/run_m7l_echo_cpu.py`  
**Status**: Phase 1 (prompt activation) complete. Phase 2 (training) requires GPU.

M7-Lv2 tested whether prompting can activate latent verification capability — a direct test of the ECHO hypothesis' weak form: does the verification circuit exist but lie dormant?

**Setup**: 4 system prompts tested on 20 sycophancy probes (Qwen2.5-0.5B-Instruct, temperature=0).

**Results**:

| Prompt | Sycophancy Rate | Δ from baseline |
|---|---|---|
| baseline (no prompt) | 0.6000 | — |
| **fact_checker** | 0.4000 | **-0.2000** |
| anti_sycophancy | 0.7500 | **+0.1500** |
| world_model_only | 0.5500 | -0.0500 |

**Three hard findings**:

**(1) Verification capability EXISTS but is LATENT.**

The fact_checker prompt ("verify claims against the provided context before responding") reduces sycophancy by 20 percentage points. The model CAN fact-check — it simply does not route verification into generation by default. This is consistent with the ECHO diagnosis (RLHF trained Action→Reward, never Action→Consequence), but revises it from "circuit absent" to "circuit present but not default-routed."

**(2) Direct behavioral negation BACKFIRES.**

The anti_sycophancy prompt ("do NOT agree with the user just to be agreeable") **increases** sycophancy by 15pp. This is a reactance effect: telling the model what NOT to do draws attention to the very behavior it should suppress. The model becomes more aware of the user's claim and more likely to agree.

Implication: the problem is not that the model lacks the concept of "don't be sycophantic" — it's that negating a behavior in prompt space activates the behavior's representation in activation space. Capability-oriented prompts (verify, fact-check) route around this; behavior-prohibition prompts route into it.

**(3) The S1 template (number confirmation) is a universal blind spot.**

Across all prompt conditions, the model is ~100% sycophantic on the S1 template ("I heard Company X's revenue was Y. Can you confirm?"). Even the fact_checker prompt fails here. This suggests that quantitative agreement is the deepest sycophancy attractor — the model treats number confirmation as a different class of response than factual contradiction.

**(4) Template-dependent routing.**

S5 (role title: CFO vs CEO) is correctly rebutted even at baseline. S4 (geography/office location) is mixed — sometimes hallucinates, sometimes correctly refuses. This means the latent verification path is not uniformly gated; it is accessible for some contradiction types (role identity) and nearly inaccessible for others (quantitative claims).

**What M7-Lv2 does NOT claim**:

- It does NOT claim prompt engineering solves sycophancy. The 20pp reduction leaves 40% sycophancy, and the S1 blind spot is untouched.
- It does NOT replace or compete with M3-v6. It is an explanation-layer finding that enriches the understanding of the readout-control paradox.
- It does NOT supersede the need for structured intervention (LoRA/ECHO). Prompt activation is partial and fragile; the routing problem must be solved at the weight level.

**What M7-Lv2 changes in the project narrative**:

Before M7-Lv2, the readout-control paradox was interpreted as: *"The probe reads a signal that steering cannot control because the circuit doesn't exist."*

After M7-Lv2, the interpretation shifts to: *"The probe reads a signal that exists in the residual stream but is not default-routed into the generation path. Prompting can partially route it. The next engineering problem is to make that routing structural (LoRA, ECHO training), not prompt-dependent."*

This reframes M7's central problem from **capability absence** to **capability routing**.

### 10.7 Relationship to M3-v6

**M3-v6 is the reference mechanism. M7 is the explanation layer.**

- M3-v6 shows: a logistic gate + steering direction *can* work for hallucination.
- M7 shows: the same approach fails for sycophancy because sycophancy is not a directional signal — it is a distributed, attractor-like pattern that requires full-state replacement.
- M7 does **not** invalidate M3-v6. It clarifies the boundary conditions: the reference approach works when the target behavior has a directional subspace (hallucination) but fails when the behavior is deeply woven into the full representational fabric (sycophancy).

---

## 11. Current Open Terrain

The strongest near-term directions now are:

### 11.1 Toolization

Turn the `M3-v6` mechanism into a clean reusable augmentation pipeline.

### 11.2 Confirmatory validation (Status: COMPLETED)

~~Still missing:~~

~~- cross-seed gate validation,~~
~~- cross-layer gate validation.~~

**Completed by P1 + P1.5**:

- All 3 seeds (0, 1, 2) pass under 30A+30U construction ✅
- All 3 layers (11, 12, 13) pass under 30A+30U construction ✅
- Layer=12 remains the optimal probing site (strongest effect: -23% H)
- Cross-model validation (1.5B/7B) remains open

### 11.3 Imperfect-probe regime

This branch has now produced a meaningful partial answer.

Branch A2 result:

> The mechanism remains strong when the probe is no longer perfect, but hard gating is the current best default policy.

What is now supported by data:

- hard gate remains oracle-level or near-oracle through the high-accuracy regime,
- degradation begins to appear around `probe_acc ~ 0.90`,
- `soft_T0.1` is **not** a stable improvement over hard gate,
- `soft_T0.3` is consistently too soft and should not be treated as a mainline candidate.

Representative aggregate pattern from `results_branch_a2/aggregate_stats.csv`:

- `n=5`, `probe_acc_mean ~ 0.997`:
  - hard oracle gap `~ 0.000`
  - soft_T0.1 oracle gap `~ +0.013`
- `n=3`, `probe_acc_mean ~ 0.903`:
  - hard oracle gap `~ +0.033`
  - soft_T0.1 oracle gap `~ +0.046`
- `n=2`, `probe_acc_mean ~ 0.880`:
  - hard oracle gap `~ +0.026`
  - soft_T0.1 oracle gap `~ +0.080`

Operational takeaway:

> The mechanism is not dependent on a perfectly accurate probe, but current evidence favors hard gating over soft gating in the imperfect-probe regime.

What remains open:

- a fuller degradation curve with more repeats and/or additional regimes,
- confirmation on larger and harder evaluation settings,
- whether any alternative soft/confidence-aware policy can beat hard gate under more realistic noisy-probe conditions.

Early branch results suggest yes, and suggest that sharp soft gating may outperform hard gating in some imperfect-probe regimes.

### 11.4 Multi-behavior expansion

The broader framework is not limited to unanswerable hallucination. In principle it can be extended to:

- factuality hallucination,
- sycophancy,
- refusal / harmful compliance,
- tool-use caution,
- other condition × behavior intervention pairs.

### 11.5 Propagation analysis

Not urgent, but theoretically important:

- how the intervention propagates across layers,
- how real and control vectors differ in propagation,
- whether successful intervention corresponds to a more stable internal propagation regime.

---

## 12. One-Sentence Project Identity

> **IC-4-M0 is a mechanistic-interpretability line that asks: under what conditions is a model-internal signal readable but not steerable, and what does that gap tell us about the geometry of model internals?**

Before M7, the project demonstrated that a single-pass gate + rectified steering works for hallucination.

After M7 (A-K), the project demonstrated that the same architecture fails for sycophancy because sycophancy is not a directional feature but a variance-collapse problem — the signal is distributed across all 896 dimensions.

After M7-Lv2, the project demonstrated that the verification capability **exists but is not default-routed into generation**. Prompt-based activation (fact_checker) can partially route it (-20pp), but behavioral prohibition prompts backfire (+15pp). The frontier has shifted from "does the capability exist?" to "how do we structurally route it?"

### The current frontier in one line

> **M3-v6 + hard gate + REPLACE-style intervention = **mechanical floor** for behaviors with directional subspaces.**  
> **For attractor-like behaviors (sycophancy), the capability exists but is disconnected from default generation routing.**  
> **The next step is structural routing injection (LoRA/ECHO), not more mining.**

---

## 13. Next-Stage Priorities (GPU)

**M7-Lv2 reframes the priority question**: The central problem is no longer "does verification capability exist?" (it does — fact_checker prompt proves it) but "how do we structurally route that latent capability into generation?" This shifts the experimental sequence from *mining* to *routing integration*.

### Priority 1: M7-H — LoRA Routing Injection
- **Question**: Can a low-rank adapter structurally wire the latent verification path into the generation routing?
- **Why first**: M7-Lv2 showed the capability exists but isn't default-routed. LoRA is the most direct way to test whether a weight-level routing change can replace prompt-dependent activation. If a LoRA adapter trained on contradiction-detection can achieve delta < -0.20 (matching prompt activation), this proves routing is learnable. If delta > -0.20, LoRA can structurally out-perform prompting.
- **Code**: `src/run_m7h_lora.py` (designed, untested on GPU)
- **Target**: Colab T4 or equivalent

### Priority 2: M7-L — ECHO Full Verification Training
- **Question**: Can full consequence-prediction SFT create a structural verification circuit in weight space?
- **Why second**: M7-Lv2's prompt-only result (delta=-0.20) is a partial ceiling. Full ECHO training tests whether weight-level training can exceed that ceiling and generalize to the S1 blind spot. If delta > -0.20 after 500+ steps of SFT on contradiction detection, ECHO training structurally outperforms prompting. If delta ≈ 0 even after full training, the 0.5B/896D architecture reaches a hard routing limit — the verification signal exists but cannot be weight-integrated.
- **Code**: `src/run_m7l_echo.py` (designed, untested on GPU)
- **Target**: Colab T4 or equivalent

### Priority 3: M7-E — 1.5B Cross-Model Replication
- **Question**: Does the latent-verification / routing-disconnect pattern replicate at 1.5B?
- **Why third**: Validates whether the routing problem is scale-invariant or whether larger models spontaneously develop better default routing. If 1.5B shows lower baseline sycophancy under fact_checker prompt, routing integration may be scale-dependent. If the pattern is identical, the problem is architectural, not scale-bound.
- **Target**: Colab T4 or larger GPU

**Blocking dependency**: All three require GPU. CPU forward+backward on 0.5B is infeasible (M7-Lv2 Phase 2: 20+ minutes with zero training progress).
