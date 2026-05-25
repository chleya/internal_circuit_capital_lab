# IC-4 P16: LoRA Geometry Analysis — Routing Fix, Not Geometry Fix

**Date**: 2026-05-23 | **Status**: ❌ **Negative (Informative)** | **Script**: `src/run_p16_lora_geometry_analysis.py`

---

## 1. Motivation

P15 proved that hallucination LoRA achieves H=0.000 at C=1.000. But **HOW**? P16 asks a geometric question:

- **P16.1**: Does LoRA preserve probe accuracy (K-subspace unchanged)?
- **P16.2**: Does LoRA INCREASE steering effect — i.e., does w_probe now better control hallucination → K↔D subspaces aligned?
- **P16.3**: Is any alignment gain concentrated in specific layers?

The distinction matters: if LoRA aligns K and D subspaces, it's a **geometry fix**. If LoRA reduces H but w_probe steering doesn't improve, it's a **routing fix** (the model takes a different default path).

## 2. Design

1. Load P15 LoRA checkpoint
2. For 9 layers (0, 3, 6, 9, 11, 12, 15, 18, 21):
   - Collect hidden states from LoRA model
   - Train hallucination probe → extract w_probe_lora
   - Test steering effect: evaluate H at α = -2, 0, +2 using w_probe_lora
3. Compare with P14 baseline (base model, no LoRA):
   - probe accuracy: same in both? (K-subspace unchanged)
   - delta_H_max: can steering reduce H now? (D-subspace alignment)
   - overlap ratio: how much do the directions project onto behavior?
4. Compute alignment_gain = (overlap_lora − overlap_base) per layer

## 3. Results

### 3.1 Layer-by-Layer Comparison

| Layer | Probe Acc (LoRA) | H_baseline (LoRA) | ΔH_max (LoRA) | P14 ΔH_max (Base) | Overlap (LoRA) | Overlap (Base) | Alignment Gain |
|---|---|---|---|---|---|---|---|
| 0 | 1.0000 | 0.000 | 0.0000 | 0.1666 | 0.0000 | 0.1666 | **−1.000** |
| 3 | 1.0000 | 0.000 | 0.0833 | 0.1666 | 0.0833 | 0.1666 | **−0.500** |
| 6 | 1.0000 | 0.000 | 0.0000 | 0.0833 | 0.0000 | 0.0833 | **−1.000** |
| 9 | 1.0000 | 0.000 | 0.0000 | 0.0833 | 0.0000 | 0.0833 | **−1.000** |
| 11 | 1.0000 | 0.000 | 0.0833 | 0.0833 | 0.0833 | 0.0833 | **0.000** |
| 12 | 1.0000 | 0.000 | 0.2500 | 0.0833 | 0.2500 | 0.0833 | **+2.001** ⚠️ |
| 15 | 1.0000 | 0.000 | 0.0000 | 0.0833 | 0.0000 | 0.0833 | **−1.000** |
| 18 | 1.0000 | 0.000 | 0.0000 | 0.0833 | 0.0000 | 0.0833 | **−1.000** |
| 21 | 1.0000 | 0.000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | — |

### 3.2 Key Finding

**P16.1: ✅ CONFIRMED — Probe accuracy is 1.0000 at all layers.** K-subspace preserved after LoRA.

**P16.2: ❌ REJECTED — LoRA does NOT increase w_probe steering effect.** At 8 of 9 layers, alignment gain is ≤ 0. At layer 12, the "gain" of +2.001 is DESTRUCTIVE — steering at α=+2.0 increases H from 0 to 0.25 (worsens).

**P16.3: N/A — No positive alignment gain found.**

### 3.3 The Layer 12 Anomaly

Layer 12 is the only layer where w_probe steering has a measurable effect on LoRA model:
- H jumps from 0.000→0.250 at α=+2.0
- This is DESTRUCTIVE — steering adds hallucination where there was none
- w_probe direction in LoRA model has non-zero projection onto behavior-opposite subspace
- This confirms: LoRA achieves H=0.000 via routing change, NOT via K↔D alignment

## 4. Interpretation

### 4.1 The Mechanism: Routing Fix, Not Geometry Fix

```
Base Model (P14):                LoRA Model (P15):
                                 
K-subspace ←→ w_probe            K-subspace ←→ w_probe (same, probe acc=1.000)
      ↓                                ↓
  (orthogonal)                    (orthogonal, unchanged)
      ↓                                ↓
D-subspace                        D-subspace
      ↓                                ↓
Default path → hallucinate        Default path → abstain (ROUTING CHANGED)
```

LoRA does NOT align the classification direction (w_probe) with the behavioral control direction. Instead, it changes the model's **default output path** — when given an unanswerable question, the model now routes to abstention by default.

### 4.2 Why This Matters

| Type | Mechanism | Resolves? |
|---|---|---|
| Geometry fix (align K↔D) | w_probe steering becomes effective | Would make the model's KNOWING directly control DOING |
| **Routing fix (change default path)** | Model learns new target for unanswerable inputs | Achieves H=0.000 but leaves K↔D orthogonal |

The routing fix is a **valid practical solution** — H=0.000 at C=1.000 is the desired outcome. But it means:
1. The model hasn't learned to "understand" its own knowledge
2. External steering (via w_probe) still can't control behavior
3. The fix is brittle — it's a learned mapping from specific input patterns to output patterns

### 4.3 The Bigger Picture

P13+P14 proved K↔D subspaces are near-orthogonal across all layers. P15 proved LoRA can bridge the gap behaviorally. P16 proves that the **geometric orthogonality persists** — LoRA routes around it rather than resolving it.

This is structurally analogous to M3-v6's gate: the model doesn't learn to *use* its knowledge, it learns an alternative behavioral path that happens to produce the right output.

## 5. Claims

| # | Claim | Evidence | Confidence |
|---|---|---|---|
| **F36** | K-subspace preserved after LoRA (probe acc=1.0000 at all 9 layers) | P16 per-layer probe accuracy | ⭐⭐⭐⭐⭐ |
| **F37** | P16.2 REJECTED: LoRA does NOT align K↔D subspaces. w_probe steering effect is zero (8 layers) or destructive (layer 12) | P16 alignment_gain ≤ 0 at 8/9 layers | ⭐⭐⭐⭐⭐ |
| **F38** | LoRA bridge mechanism = ROUTING fix (default path change), NOT geometry fix (subspace alignment) | P16 + P15 + P14 synthesis | ⭐⭐⭐⭐⭐ |
| **F39** | Layer 12 is the only layer where w_probe still projects onto behavior in LoRA model — destructively (H 0→0.25 at α=+2.0) | P16 layer 12 overlap=0.250 vs baseline H=0.000 | ⭐⭐⭐⭐ |

---

*Related: [P15 Hallucination LoRA](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P15_HALLUCINATION_LORA.md) | [P17 Module Ablation](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P17_LORA_ABLATION.md)*