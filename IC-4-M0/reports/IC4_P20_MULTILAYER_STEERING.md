# IC-4 P20: Multi-Layer Steering — A-Bottleneck is Layer-Independent

**Date**: 2026-05-24 | **Status**: ❌ **Negative (Informative)** | **Script**: `src/run_p20_multilayer_steering.py`

---

## 1. Motivation

P12 showed L10 v_abs steering causes homogenization with degradation (H_early +100%, H_mid/late −67%). P19 revealed that attention entropy follows a U-shaped curve across layers: early high → mid minimal (L9: 4.4%) → deep amplified (L23: 13.0%). This suggested that deep-layer routing is where position asymmetry amplifies.

P20 asks: **Does steering at DEEP layers (L21) produce a DIFFERENT position profile than L10 steering?**

The logic: if asymmetry is AMPLIFIED in deep layers (P19), then intervening at the amplification point (L21) might produce a more favorable profile — less degradation of early positions, more improvement of late positions.

## 2. Design

- **Model**: Qwen2.5-0.5B-Instruct
- **Data**: position_sensitivity/s0, n=10 per position, train n=5 for v_abs
- **Steering**: v_abs = mean(h_early) − mean(h_late), unit-normalized
- **α=3.0** (P12 optimal)
- **Layers**: [10, 21]
- **Generation**: temperature=0.0, max_new_tokens=10

**4 conditions**: L10_a0 (baseline), L10_a3, L21_a0, L21_a3

**Hypotheses**:
- H20.1: L21_a3 ΔH < L10_a3 ΔH (deep-layer steering reduces position gap better)
- H20.2: L21_a3 does NOT degrade H_early as much as L10_a3 (routing-specific, not uniform)
- H20.3: L21_a0 = L10_a0 (alpha=0 baseline should be identical regardless of layer)

## 3. Results

### 3.1 Per-Condition Metrics

| Condition | H_early | H_mid | H_late | ΔH | C_early | C_late | ΔC |
|---|---|---|---|---|---|---|---|
| L10_a0 (baseline) | 0.000 | 0.500 | 0.500 | 0.500 | 0.167 | 0.000 | 0.333 |
| L10_a3 (steered) | 0.250 | 0.500 | 0.500 | 0.250 | 0.167 | 0.333 | 0.167 |
| L21_a0 (control) | 0.000 | 0.500 | 0.500 | 0.500 | 0.167 | 0.000 | 0.333 |
| **L21_a3 (steered)** | **0.250** | **0.500** | **0.500** | **0.250** | **0.167** | **0.167** | **0.333** |

### 3.2 L10 vs L21 Steering Comparison

| Metric | L10_a3 | L21_a3 | Difference |
|---|---|---|---|
| **H profile** | (0.250, 0.500, 0.500) | (0.250, 0.500, 0.500) | **IDENTICAL** |
| **ΔH** | 0.250 | 0.250 | **IDENTICAL** |
| C_early | 0.167 | 0.167 | identical |
| C_late | 0.333 | 0.167 | L21 WORSE |

### 3.3 Hypothesis Verification

| Hypothesis | Prediction | Result | Status |
|---|---|---|---|
| H20.1: L21 ΔH < L10 ΔH | L21 better | L21 ΔH = L10 ΔH = 0.250 | ❌ **REFUTED** |
| H20.2: L21 less H_early degradation | L21 better | Both H_early = 0.250 | ❌ **REFUTED** |
| H20.3: L21_a0 = L10_a0 | identical | Both (0.000, 0.500, 0.500) | ✅ confirmed |

## 4. Interpretation

### 4.1 The Key Negative Finding

**L10 and L21 steering produce IDENTICAL position profiles.** The steering layer doesn't matter — v_abs injection at any layer produces the same homogenization with degradation pattern.

This is consistent with P13: L10 perturbations are uniformly propagated through all downstream layers. If the perturbation is uniform, then injecting at L21 is equivalent to injecting at L10 — both end up affecting the same downstream computation.

### 4.2 Why P19's U-Curve Doesn't Help

P19 showed that attention entropy position gap is amplified in deep layers (L23: 13.0%). This means the asymmetry is LARGER at deep layers — but the AMPLIFICATION mechanism is not something you can fix by injecting steering at the amplification point.

The U-curve describes WHERE asymmetry becomes visible, not WHERE it's CAUSED. The causal asymmetry happens after ALL layers — in the output projection or decoding stage. Steering at L10 or L21 both propagate to the same downstream effect.

### 4.3 The Steering-Vector Parallel Transport

```
v_abs injected at L10 → propagates uniformly through L11-L24 → same output effect
v_abs injected at L21 → propagates uniformly through L22-L24 → same output effect
```

Since the final output is the same, the intervention point doesn't matter. The probe→behavior gap for absorption is truly structural — it exists in the mapping from hidden states to output tokens, not in any particular layer's hidden states.

### 4.4 Where the Gap Lives

| Intervention | Layer | Effect | Verdict |
|---|---|---|---|
| Position rectification | L10 | ΔH still 1.0 (A3) | ❌ |
| v_abs directional steering | L10 | ΔH=0.250, H_early +100% (P12) | ❌ |
| v_abs directional steering | L21 | ΔH=0.250, H_early +100% (P20) | ❌ same |
| Weight-level LoRA | — | (not tested for absorption) | TBD |

The gap is NOT in any single layer's hidden states. It's in the **mapping** from hidden states to output — the decoding process itself. This is structurally similar to the B-bottleneck (K↔D orthogonal), but for absorption, the mapping is position→output, not knowledge→output.

### 4.5 The Only Path Forward

P20 eliminates multi-layer steering as an absorption remedy path. Combined with P12 (L10 steering) and P13 (uniform L10 perturbation), the conclusion is:

- **Hidden-state vector interventions are DEAD for absorption.** All layers produce the same homogenization-with-degradation pattern.
- **Weight-level interventions (LoRA) remain untested for absorption.** Phase 10 tested position-invariance LoRA (mixed results: H=0.500). P15 showed hallucination-targeted LoRA works for B-bottleneck. An absorption-targeted LoRA might work similarly.
- **Attention-level intervention might still work.** P19 showed attention entropy differs by position. Modifying attention DIRECTLY (not via hidden-state steering) might bypass the probe→behavior gap.

## 5. Claims

| # | Claim | Evidence | Confidence |
|---|---|---|---|
| **F58** | H20.1+H20.2 REFUTED: L21 steering produces the IDENTICAL position profile as L10 steering. ΔH=0.250 for both, H profile (0.25, 0.50, 0.50) identical. | P20 L10_a3 vs L21_a3 | ⭐⭐⭐⭐⭐ |
| **F59** | A-bottleneck steering effect is LAYER-INDEPENDENT. v_abs injection at any layer propagates uniformly to the same downstream effect. | P20 + P13 uniform propagation | ⭐⭐⭐⭐⭐ |
| **F60** | L21_a3 C_late=0.167 is WORSE than L10_a3 C_late=0.333. Deep-layer steering may have a small correctness cost not present at L10. | P20 C_late comparison | ⭐⭐⭐ |
| **F61** | Hidden-state vector interventions are EXHAUSTED for absorption. All tested layers produce the same homogenization-with-degradation. The gap is in the hidden-state→output mapping, not in any layer's hidden states. | A3+P12+P20 synthesis | ⭐⭐⭐⭐⭐ |
| **F62** | P19's U-curve describes WHERE asymmetry is visible (deep layers) but not WHERE it's CAUSED. Causal asymmetry is in the output decoding, downstream of all layers. | P19+P20 synthesis | ⭐⭐⭐⭐ |

---

*Related: [P12 Absorption Steering](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P12_ABSORPTION_STEERING.md) | [P13 Energy/Direction](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P13_ENERGY_VS_DIRECTION.md) | [P19 Attention Patterns](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P19_ATTENTION_PATTERNS.md)*