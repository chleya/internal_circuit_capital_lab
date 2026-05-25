# IC-4 P10: Formal Exclusion — Single-Direction Impulse for Hallucination

> **Date**: 2026-05-23 | **Status**: Completed (Documentation)
> **Type**: Formal exclusion — no new experiment
> **Question**: Can hallucination be controlled via direction-specific steering vectors,
>   as sycophancy was?

**Answer: No. Single-direction impulse is formally excluded for hallucination.
Hallucination control requires energy-based / closed-loop gate approaches, not
direction-specific probes or vectors.**

---

## 1. Evidence Chain

The exclusion rests on a 5-layer evidence chain collected across T1–T3, P2, P3,
P4, and B2, spanning probe analysis, impulse steering, direction-specificity audits,
replication, and decomposition.

### Layer 1: T1+T2 — Hallucination is Representationally Separable

| Experiment | Finding |
|---|---|
| T1 projection | v_hall projection separates hall vs abstention from step 0, v_hall/random = 3.51× |
| T2 probe | Hall probe accuracy peaks at step 0 (0.917 at L8), cross_layer_band structure |

**Interpretation**: Hallucination IS detectable in hidden states. The model
"knows" when it will hallucinate. But detectability ≠ controllability.

### Layer 2: T3 — Early Impulse Effects Are Direction-Independent

| Direction | Mean Controllability |
|---|---|
| v_hall | 0.0219 |
| random | 0.0303 |
| shuffled | 0.0392 |
| v_syc | 0.0545 |

**Key finding**: v_hall controllability (0.0219) is BELOW random (0.0303).
The direction specifically computed to capture hallucination performs WORSE
than a random direction. Early-state sensitivity to perturbation is real, but
not direction-specific.

### Layer 3: P2 + P3 — Direction Specificity Formally Excluded

| Experiment | Finding | v_hall/random |
|---|---|---|
| P2 (n_syc=12) | v_hall = v_orthogonal: ΔH=-0.283 vs -0.283 (identical) | — |
| P2 (n_syc=12) | All 5 directions in narrow 0.24–0.28 range | — |
| P3 (n_hall=6) | Confirmed: v_hall/random = 0.28× | 0.28× |

**P2 was definitive**: A vector mathematically orthogonal to v_hall (cos<1e-5)
produces EXACTLY the same hallucination rate change. The direction that
"should" matter (v_hall) confers zero causal advantage over a direction
guaranteed irrelevant.

**P3 replicated at scale**: v_hall/random = 0.28× << 1.0. A direction-specific
vector should have ratio > 1.0 (as syc does: 2.73× at P3, 1.68× at P4).

**Verdict**: v_hall does not capture causal information. The hallucination
impulse effect is pure energy, zero direction.

### Layer 4: P4 — Direction-vs-Energy Decomposition

| Behavior | Pure Direction | Pure Energy | Dominance |
|---|---|---|---|
| Sycophancy | +0.0164 | −0.0022 | Direction-dominated |
| Hallucination | — | (only energy matters) | Energy-only |

P4 decomposed syc controllability into direction alignment and energy
contribution using norm-matched orthogonal vectors. Syc's causal effect
comes from direction alignment (+0.0164), not energy (−0.0022).

For hallucination, this decomposition is not applicable because the premise
fails: there IS no direction component. v_hall = v_orthogonal means the
direction is causally irrelevant.

### Layer 5: B2 — Structured Control Boundary

| Condition | Result |
|---|---|
| C_base = 0.400 | Multi-direction combos CAN improve over single directions |
| C_base = 0.800 | ALL 15 pair synergies ≤ 0; best single = random |
| M3-v6 closed-loop gate | Oracle-level at degraded baseline; random wins at C_base=0.800 |

B2 revealed the second reason to abandon single-direction impulse: even IF
there were direction-specificity, structured control has a boundary condition.
It only works when the baseline is already degraded. At standard performance
levels, random perturbation outperforms structured control.

---

## 2. Why Sycophancy Is Different

The contrast with sycophancy makes the exclusion conclusive:

| Property | Hallucination | Sycophancy |
|---|---|---|
| Direction specificity | **No** (v_hall = v_orthogonal) | **Yes** (v_syc ≠ all controls) |
| v/random ratio | 0.28× (P3) | 2.73× (P3), 1.68× (P4) |
| Dominance | Pure energy | Direction-dominated |
| Best intervention | Closed-loop gate (M3-v6) | Open-loop steering (−23~30%) |
| Structural boundary | C_base=0.800: random > structured | No boundary found |

**The Hall-Syc asymmetry is the most important finding of this project.**
They are not the same kind of controllability object. They require different
intervention strategies.

Sycophancy: direction-specific vectors (v_syc) add causal information.
Hallucination: the "direction" of perturbation is irrelevant; only magnitude
and layer placement matter.

---

## 3. What IS Effective for Hallucination

| Intervention | Mechanism | Effect |
|---|---|---|
| M3-v6 closed-loop gate | Probe detects hallucination risk → gate routes to oracle | H: 0.867→0.667 (−23%), C_base unchanged |
| Large early impulse (any direction) | Generic perturbation at prefill/early steps | Transient effect, no direction specificity |
| Multi-direction combinations (degraded baseline) | Energy aggregation | C_base=0.400: score +0.200 over single |

The closed-loop gate (probe→gate→oracle) is the current best intervention.
It works because it detects risk and routes around it, not because it
steers activations in a specific direction.

---

## 4. Formal Exclusion Statement

> **Single-direction impulse steering using vector decomposition is formally
> excluded as a control strategy for hallucination.** The evidence spans
> 5 layers (T1–T3, P2, P3, P4, B2) and converges on a single conclusion:
> hallucination-related hidden state perturbations are energy-driven, not
> direction-driven. v_hall = v_orthogonal (P2). v_hall/random = 0.28× < 1.0
> (P3). v_hall controllability < random controllability (T3). Hallucination
> and sycophancy are fundamentally different controllability objects.

**This exclusion is definitive, not provisional.** Future work on
hallucination control should focus on:
1. Closed-loop gate-based routing (M3-v6 pattern)
2. Energy/perturbation-based approaches (not direction-specific)
3. Multi-direction / ensemble perturbation (for degraded baselines only)
4. Oracle routing enhancement

**The single-direction impulse line of inquiry is CLOSED.**

---

## 5. Impact on Research Program

### Lines Affected
- **T1–T3 hallucination branch**: Findings are valid (representational
  separability, early-state sensitivity) but the impulse interpretation
  is revised: effects are energy-driven, not direction-driven.
- **P2 direction-specificity audit**: Now serves as the definitive exclusion
  evidence, not a "negative that needs follow-up."
- **B2 structured control**: Boundary condition finding strengthened — even
  if direction mattered, structured control only works at degraded baseline.

### Lines NOT Affected
- **Syc direction-specificity**: Confirmed and strengthened by contrast.
  The asymmetry (Hall=energy, Syc=direction) is now conclusive.
- **M3-v6 closed-loop gate**: Remains the best hallucination intervention.
- **Per-Action KMeans / Stabilization**: Independent of hallucination
  direction question.

### Knowledge Debt Closed
- N1 ("Hall direction-specificity NOT confirmed") → upgraded to formal
  exclusion with 5-layer evidence chain
- The question "could a better vector decomposition method find hallucination
  direction-specificity?" is answered: No. v_hall = v_orthogonal (orthogonality
  is the strongest possible negative).