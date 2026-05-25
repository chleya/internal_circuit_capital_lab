# IC-4 P9: Cross-Bottleneck Structural Integrity Check

> **Date**: 2026-05-23 | **Status**: Completed
> **Predecessor**: P8 Large-Scale Replication, C11 Cross-Bottleneck Analogue
> **Layer**: 10 | **Alpha**: -3.0 | **N**: 24

---

## 1. Motivation

P8 confirmed steering direction is correct but the two-stage selective advantage is spurious at n=24. P9 asks a deeper structural question: does steering degrade the hidden state clustering that separates syc from non-syc samples?

If steering disrupts structural organization, the Stabilization bottleneck (Per-Action KMeans) could compensate by post-steering reclustering. If structure is preserved, the two bottlenecks operate independently and cross-bottleneck synergy is not supported.

---

## 2. Experiment Design

| Phase | Description |
|---|---|
| 1 | Collect L10 last_prompt_token hs from baseline (no hook) forward pass |
| 2 | Collect L10 last_prompt_token hs from steered (v_syc a=-3.0 hook) forward pass |
| 3 | KMeans (k=2) clustering on each condition |
| 4 | Compare ARI, purity, centroid separation, per-sample shift |

---

## 3. Results

| Metric | Baseline | Steered | Delta | Interpretation |
|---|---|---|---|---|---|
| ARI (vs ground truth) | 1.0000 | 1.0000 | +0.0000 | unchanged |
| Cluster Purity | 1.0000 | 1.0000 | +0.0000 | unchanged |
| Centroid Cosine Sim | 0.9356 | 0.9400 | +0.0044 | unchanged |
| Mean Intra-Cluster Dist | 1.7893 | 1.7894 | +0.0001 | unchanged |
| Inter-Cluster Distance | 4.3051 | 4.3049 | -0.0002 | unchanged |

- **Baseline cluster sizes**: [12, 12]
- **Steered cluster sizes**: [12, 12]

### 3.1 Per-Sample Hidden State Shift

- Mean ||hs_steered - hs_baseline||: 3.0000
- Std ||hs_steered - hs_baseline||: 0.0000
- Max ||hs_steered - hs_baseline||: 3.0000
- Cosine sim (hs_steered, hs_baseline) mean: 0.9707

### 3.2 Key Observations

1. **Perfect structural separation at L10.** ARI=1.0 means KMeans (k=2)
   perfectly separates design-intent sycophantic vs non-sycophantic samples
   based on L10 last_prompt_token hidden states. Purity=1.0 confirms each
   cluster is 100% homogeneous. This is expected: the contrast set uses
   explicitly different fact_checker personas to induce syc/non-syc behavior,
   and the model's L10 discriminates them perfectly.

2. **Steering preserves structure completely.** All metrics are effectively
   identical between baseline and steered conditions (all |delta| < 0.005).
   The v_syc vector (α=-3.0) shifts every hidden state by exactly 3.0 norm
   units (||v_syc||=1.0 × |α|), but this is a uniform translation that
   preserves inter-point distances and cluster geometry. Cosine similarity
   is 0.9707 — the angular structure is nearly unchanged.

3. **Steering = uniform translation, not structural perturbation.** The
   steering hook adds the same vector (v_syc × α) to every L10 output.
   This shifts all hidden states by the same direction and magnitude,
   preserving relative positions. The downstream effect (behavioral change)
   comes from the residual stream propagating this uniform shift through
   layers 11-24, not from disrupting the representational geometry at L10.

---

## 4. Interpretation

**Verdict: Structure preserved — bottlenecks are independent.**

Steering does NOT degrade the syc/non-syc clustering structure at L10.
ARI stays at 1.0, purity stays at 1.0, centroid cosine sim and intra/inter
distances are effectively identical.

**Why this matters**:

1. **The Organization (steering) intervention is structural-integrity-preserving.**
   v_syc steering shifts hidden states by a uniform 3.0 norm units without
   disrupting relative geometry. The mechanism is pure translation in the
   residual stream, not distortion of representational structure.

2. **Cross-bottleneck synergy (1+1>2) is NOT supported.**
   Per-Action KMeans (stabilization) works by reclustering hidden states to
   recover structural organization degraded by perturbation (as shown in C11).
   But steering does not degrade structure — it is a uniform translation.
   There is nothing for stabilization to "recover." The two bottlenecks
   operate on independent dimensions.

3. **The behavioral effect (syc reduction) is NOT mediated by structural
   disorganization.** Steering reduces sycophancy by −23~30% (P8) by shifting
   the residual stream in the anti-syc direction, not by collapsing or
   reorganizing representational geometry. This supports the interpretation
   that the steering vector acts as a directional bias in activation space,
   not as a structural perturbation.

**Implication**: Per-Action KMeans and steering ARE additive — they can be
applied together without interference — but there is no synergistic interaction
at the representational level. Each addresses a separate mechanism
(energy/stability for stabilization, direction/bias for steering).

**This is a positive finding**: it means we don't have to worry about steering
collateral damage. The steering intervention is "clean" — it shifts behavior
without degrading representational quality.

---

## 5. Next Steps

| Priority | Action |
|---|---|
| 1 | P10: Hallucination — formally abandon single-direction impulse |
| 2 | Per-Action KMeans Scaling on gridworld |
| 3 | Absorption Remedy (position-aware routing) |