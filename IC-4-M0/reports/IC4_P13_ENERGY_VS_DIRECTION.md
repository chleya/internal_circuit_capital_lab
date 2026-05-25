# IC-4 P13: Energy vs Direction Asymmetry — Uniform at Representation, Asymmetric at Behavior

> **Date**: 2026-05-23 | **Status**: Completed — Key asymmetry is downstream, not in L10 shift  
> **Type**: Experiment — representational-level perturbation comparison  
> **Predecessors**: P12 (directional steering behavioral results)

**Question**: Is the behavioral asymmetry observed in P12 (directional steering
disproportionately affects early position) reflected in L10 hidden state shift
magnitudes? Or does it arise from downstream computation?

**Answer: Both energy and directional perturbations produce nearly perfectly
uniform L10 hidden state shifts across all positions. The behavioral asymmetry
is a DOWNSTREAM effect — early position has higher behavioral sensitivity to
directional perturbation even though the hidden state shift is identical.**

---

## 1. Motivation

P12 found that directional steering (v_abs) produced behavioral homogenization
with degradation — H_early worsened more than H_mid/H_late improved. This
suggested an asymmetry: early position is more sensitive to directional
perturbation.

Two possible loci:
1. **Representational**: v_abs shifts early hidden states MORE than late
   (different ||Δh|| by position)
2. **Downstream**: v_abs shifts all positions equally at L10, but early
   position's behavioral mechanism is more fragile to directional perturbation

P13 tests locus 1 vs locus 2 by measuring ||Δh|| at L10 across positions
under two perturbation types: energy (Gaussian noise) and direction (v_abs).

---

## 2. Results

### 2.1 Energy Perturbation (Gaussian Noise at L10)

| noise std | mean_shift | early | mid | late | max_ratio | e/l ratio |
|---|---|---|---|---|---|---|
| 0.01 | 0.30 | 0.30 | 0.30 | 0.30 | 1.02 | 0.99 |
| 0.03 | 0.90 | 0.91 | 0.90 | 0.89 | 1.02 | 1.02 |
| 0.05 | 1.49 | 1.50 | 1.49 | 1.49 | 1.01 | 1.01 |
| 0.10 | 2.98 | 2.98 | 2.96 | 3.00 | 1.01 | 0.99 |

**Energy max_ratio = 1.01–1.02** → noise shifts all positions equally.

### 2.2 Direction Perturbation (v_abs at L10)

| alpha | mean_shift | early | mid | late | max_ratio |
|---|---|---|---|---|---|
| −3.0 | 3.00 | 3.00 | 3.00 | 3.00 | **1.001** |
| −1.5 | 1.50 | 1.50 | 1.50 | 1.50 | **1.000** |
| +1.5 | 1.50 | 1.50 | 1.50 | 1.50 | **1.000** |
| +3.0 | 3.00 | 3.00 | 3.00 | 3.00 | **1.000** |

**Direction max_ratio = 1.000–1.001** → steering shifts all positions equally.

### 2.3 Mathematical Confirmation

For unit-norm steering vector v_abs: ||Δh|| = ||α·v_abs|| = |α| for all h.
The direction perturbation shift is position-independent by construction.
This serves as a validation check.

---

## 3. Key Finding: Asymmetry is Downstream

| Locus | Energy | Direction | Asymmetric? |
|---|---|---|---|
| L10 hidden state shift | Uniform (max_ratio=1.01) | Uniform (max_ratio=1.00) | ❌ No |
| Behavioral sensitivity (P12) | Not tested | H_early +100% vs H_late −25% | ✅ Yes |

> **The behavioral asymmetry in P12 is NOT caused by differential hidden state
> perturbation at L10. It arises from downstream computation beyond L10.**
>
> Early position has higher **behavioral sensitivity** to directional perturbation
> even though the initial hidden state shift is identical to other positions.

### 3.1 Implications for Absorption

1. **Steering "works" at the representational level** — v_abs cleanly adds a
   uniform shift to all positions (no position-dependent bias in ||Δh||).

2. **But downstream processing amplifies the perturbation differently** —
   early position's behavioral advantage (H=0.25 vs H=1.00) is fragile to
   directional perturbation. Even though the L10 perturbation is uniform,
   early position degrades more.

3. **The probe→behavior gap extends to perturbation sensitivity** — A1 proved
   probe-level absorption is fixable, A2/A3/A4/P12 proved behavioral absorption
   resists intervention. P13 adds: even when the intervention is uniformly
   applied at L10, behavioral responses are position-dependent. This gap is
   not about intervention delivery — it's about how different computational
   paths amplify or dampen the same perturbation.

### 3.2 Three-Layer Architecture of Absorption

P13 refines the absorption picture into three layers:

| Layer | Description | Status |
|---|---|---|
| Input (position) | Information degrades by position at input | Diagnosed (PSI=0.0084) |
| Representation (L10) | Perturbations are uniformly transmitted | **P13: Uniform** |

The asymmetry emerges at the third layer — behavioral computation — which is
not directly measurable through hidden state shifts. This explains why
positional remedies (A3 rectification) and directional remedies (P12 steering)
cannot selectively improve late-position behavior: the intervention is uniform
at L10, but downstream computation produces non-uniform behavioral responses.

---

## 4. Relation to Hall-Syc Asymmetry

| Dimension | Hall | Sycophancy | Absorption |
|---|---|---|---|
| Direction-specificity | ❌ v_hall/random=0.28× | ✅ v_syc/random=2.73× | ⚠️ Uniform at L10, asymmetric at behavior |
| Energy sensitivity | ✅ Dominant | ❌ Not dominant | ✅ Uniform energy response at L10 |
| Behavioral locus | All positions uniformly | Syc-specific samples | Position-dependent sensitivity |

Absorption is closest to Hall in its energy-uniformity at the representational
level, but has unique position-dependent behavioral sensitivity that neither
Hall nor Syc exhibit. This makes it the most structurally complex bottleneck.

---

## 5. Formal Declaration

> **F30: Both energy and directional perturbations produce uniform L10 hidden
> state shifts across all positions (max_ratio ≤ 1.02). The behavioral
> asymmetry observed in P12 is a DOWNSTREAM effect — early position has
> higher behavioral sensitivity to directional perturbation despite identical
> L10 perturbation magnitude.**
>
> **Absorption is NOT energy-dominated or direction-dominated at the
> representational level — both perturbation types transmit uniformly.
> The asymmetry is in behavioral sensitivity, not in perturbation delivery.**
>
> **This explains why all tested behavioral remedies (A3: rectification,
> A4: LoRA, P12: directional steering) fail: they target the wrong level
> of the problem hierarchy.**

---

## 6. Next Steps

| Priority | Action | Rationale |
|---|---|---|
| 1 | **Absorption: downstream intervention.** Target post-L10 computation (attention patterns, output logit modulation) rather than L10 hidden states. The uniform L10 perturbation has non-uniform behavioral effects → need to modulate the amplification/dampening mechanism. | P13 shows L10 is the wrong intervention target. |
| 2 | **Cross-project synthesis.** P9–P13 complete the absorption diagnosis phase. Three bottlenecks fully characterized. | Deliverable. |
| 3 | **Syc larger-n confirmation.** | Low-hanging fruit. |

---

*IC-4-M0 P13 Report — 2026-05-23*  
*Energy vs Direction Asymmetry — Uniform at L10, Asymmetric at Behavior*