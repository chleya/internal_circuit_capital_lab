# Internal Circuit Capital / IC-4-M0

> Small-model failure is not best understood as simple knowledge absence, but as a structural adaptation problem: the model struggles to absorb discretized input without distortion, stabilize useful structure under updating and compression, and organize latent capability into reliable behavior. We study whether these failures can be diagnosed and partially compensated through feedback, anchoring, and trajectory-level control.

---

## Entry points (read in this order)

| # | Document | What it is | Read when |
|---|----------|------------|-----------|
| 1 | [ENGINEERING_CYBERNETICS_FRAMING.md](ENGINEERING_CYBERNETICS_FRAMING.md) | Top-level language. Model as controlled object, hidden space as state space, probe/gate/hook as control input. | First. Sets the vocabulary for everything else. |
| 2 | [PROJECT_ENDGAME_AND_HANDOFF.md](PROJECT_ENDGAME_AND_HANDOFF.md) | Endgame specification. What "success" means, what proof obligations remain, stop/go rules. | Second. Tells you what the project is trying to prove. |
| 3 | [UNIFIED_RESEARCH_MAP.md](UNIFIED_RESEARCH_MAP.md) | Current anchor points, positions, and routes through the experiment landscape. | Third. Where are we now, what's next. |
| 4 | [UNIFIED_THESIS.md](UNIFIED_THESIS.md) | Master thesis. How IC-4, IC-2, trajectory dynamics, and the proof sequence converge to one main line. | Fourth. The unified story. |

---

## Supporting documents

| Document | Role |
|----------|------|
| [CROSS_BOTTLENECK_SYNTHESIS.md](CROSS_BOTTLENECK_SYNTHESIS.md) | Proof D. System diagram, bottleneck-symptom-diagnosis-intervention table, boundaries. |
| [**RESEARCH_TRAJECTORY_REPORT.md**](RESEARCH_TRAJECTORY_REPORT.md) | **Complete research trajectory report. Start-to-finish documentation of all experiments, findings, and the path forward.** |
| [**PHASE_6_7_8_PLAN.md**](PHASE_6_7_8_PLAN.md) | **Forward research plan. Phase 6 (Stabilization Scaling) → Phase 7 (Cross-Bottleneck) → Phase 8+ (Absorption Remedy). Go/no-go decision tree.** |
| [IC4_COMPREHENSIVE_RESEARCH_REPORT.md](IC4_COMPREHENSIVE_RESEARCH_REPORT.md) | Full experiment data from all IC-4 sub-projects. |
| [TRAJECTORY_DYNAMICS_PHASE_1_5.md](TRAJECTORY_DYNAMICS_PHASE_1_5.md) | Trajectory dynamics phase 1.5 artifact audit. |

---

## Three bottlenecks, three experimental lines

| Bottleneck | Question | Primary line | Status |
|------------|----------|-------------|--------|
| **Absorption** | How does fragmented input distort internal state? | Position sensitivity | **Diagnosed**: KNN=1.0 + PSI=0.0084 + ΔC=0.067. Not yet remedied. |
| **Stabilization** | How does useful structure survive compression and updating? | IC-2 (intelligence_capital_minimal_lab) | **Root cause found + partially fixed**: KMeans ignores Y → Per-Action KMeans=0.585 (+31% over NoMemory). |
| **Organization** | How does latent capability fail to route into behavior? | IC-4 (this repo) | **Partially compensated**: Closed-loop gate works at oracle level. **Boundary condition**: structured control only beats random at degraded C_base. Direction-specificity excluded for both hall and syc. |

---

## Proof obligations status

| Proof | Question | Result |
|-------|----------|--------|
| A | Sycophancy direction-specificity at larger n? | Negative. v_syc < random. Both behaviors on generic perturbation side. |
| B | Can multi-direction intervention beat single-direction? | Positive at C_base=0.400. **B2 audit**: boundary condition — at C_base=0.800, random beats all structured combos. |
| C | Can stabilization be remedied? | Positive. Anchored +8.7%. **Root cause breakthrough**: Per-Action KMeans=0.585 (+31% over NoMemory). |
| D | Cross-bottleneck synthesis? | Done. v5.0. All three bottlenecks diagnosed. |
| — | Absorption diagnosis (P1) | Three-layer evidence: KNN=1.0 + PSI=0.0084 + ΔC=0.067. Partial downstream compensation. |
| — | Syc energy decomposition | Pure energy, no direction-specificity. d/e ratio 0.31. |
| — | TT-SFT v0/v1 | Negative. CE-only beats trajectory cosine alignment. Route excluded. |

**4/4 proofs complete. All three bottlenecks diagnosed. Structured control boundary condition characterized.**

---

## How to run experiments

```bash
cd F:\internal_circuit_capital_lab\IC-4-M0

# Proof A: Sycophancy T3 direction-specificity (CPU ~85 min full)
python -m src.run_a_syc_t3_replication --fast

# Proof B: Multi-direction hallucination intervention (CPU ~25 min)
python -m src.run_b_multidirection_intervention --fast

# Proof C: Anchored consolidation (CPU ~5 min)
cd F:\intelligence_capital_minimal_lab
python src/run_c_anchored_consolidation.py
```

Model: `Qwen/Qwen2.5-0.5B-Instruct`

---

## For future agents

If you are a future LLM continuing this project, read the four entry points in order. Everything you need to understand the project's purpose, current state, and what to do next is in those documents.

Before proposing new work, classify it against the three bottlenecks. Prefer experiments that move an open question from ambiguous to bounded, turn a diagnosis into a remedy, or distinguish one control interpretation from another.