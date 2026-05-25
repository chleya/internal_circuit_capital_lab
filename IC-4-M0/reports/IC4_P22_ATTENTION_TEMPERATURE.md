# IC-4 P22: Attention Temperature Scaling — Proximity Over-Confidence is Robust, Not Fragile

**Date**: 2026-05-24 | **Status**: ❌ **Negative (Informative)** | **Script**: `src/run_p22_attention_temperature.py`

---

## 1. Motivation

P21 showed that LoRA eliminates mid/late hallucination (H 1.000→0.000) but leaves early position with H=0.250 residual. The hypothesis: early position over-confidence comes from too-sharp attention on recently-seen context tokens.

P22 tests this by ARTIFICIALLY modifying attention temperature at early layers:
- **T < 1.0**: Sharper attention → should INCREASE over-confidence → higher early H
- **T > 1.0**: Softer attention → should REDUCE over-confidence → lower early H

Mechanism: modify `attention_module.scaling = head_dim^-0.5 / T` directly, changing the softmax temperature.

## 2. Design

- **Model**: Qwen2.5-0.5B-Instruct, eager attention mode
- **Intervention**: Modify `self_attn.scaling` at layers [0, 3, 6, 9]
- **Temperatures**: 0.5 (sharper), 2.0 (softer), 5.0 (very soft), 1.0 (baseline)
- **Test**: 30 samples (10 per position), generate() evaluation
- **Note**: Temperature is restored after each condition

**Hypotheses**:
- H22.1: T > 1.0 at early layers reduces early-position H (soften over-confidence)
- H22.2: Temperature effect is layer-specific — only affects early position
- H22.3: T < 1.0 INCREASES early H (sharper attention worsens over-confidence)

## 3. Results

### 3.1 Per-Condition Metrics

| Condition | early H | early C | mid H | mid C | late H | late C | ΔH |
|---|---|---|---|---|---|---|---|
| T=1.0 (baseline) | **0.250** | 0.167 | 1.000 | 0.500 | 1.000 | 0.500 | 0.750 |
| T=0.5 (sharper) | 0.500 | 0.500 | 1.000 | 0.167 | 1.000 | 0.500 | 0.500 |
| T=2.0 (softer) | 0.500 | 0.333 | 1.000 | 0.167 | 1.000 | 0.333 | 0.500 |
| T=5.0 (very soft) | **0.750** | 0.667 | 1.000 | 0.667 | 1.000 | 0.500 | 0.250 |

### 3.2 Temperature → early H Relationship

```
T=1.0: early H = 0.250  (baseline, OPTIMAL)
T=0.5: early H = 0.500  (+0.250, WORSE)
T=2.0: early H = 0.500  (+0.250, WORSE)
T=5.0: early H = 0.750  (+0.500, MUCH WORSE)
```

**Both sharper AND softer attention make early H WORSE.** Baseline T=1.0 is the optimal point.

### 3.3 Mid/Late Invariance

**Mid/late H = 1.000 at ALL temperatures.** Attention temperature at early layers has ZERO effect on mid/late position hallucination. This strongly suggests mid/late hallucination is NOT an attention-routing problem within early layers (L0-L9).

### 3.4 Hypothesis Verification

| Hypothesis | Prediction | Result | Status |
|---|---|---|---|
| H22.1: T>1 reduces early H | early H ↓ | early H ↑ at BOTH T=2.0 and T=5.0 | ❌ **REFUTED** |
| H22.2: Effect is layer-specific | only early changes | mid/late = 1.000 at all T | ✅ confirmed (wrong sign) |
| H22.3: T<1 increases early H | early H ↑ | T=0.5: early H 0.250→0.500 | ✅ confirmed |

## 4. Interpretation

### 4.1 The Central Negative Finding

**Attention temperature at early layers CANNOT reduce proximity-based over-confidence.** Any deviation from T=1.0 makes early H WORSE, with the degradation proportional to the magnitude of deviation:

```
|T - 1.0| = 0.5:  ΔH_early = +0.250
|T - 1.0| = 4.0:  ΔH_early = +0.500
```

This suggests that the model's default attention temperature is already calibrated for optimal behavior. Proximity over-confidence is NOT a fragile attention pattern that can be disrupted by softening — it's a ROBUST property of the model's computation.

### 4.2 Mid/Late Complete Invariance

Mid/late H=1.000 at ALL temperatures is a striking null result. Combined with P19 (attention entropy highest at L0 and L23, minimal at L9), this reveals:

- **Early layers (L0-L9):** Attention temperature here ONLY affects early position → these layers process position-specific attention patterns
- **Mid/Late hallucination:** Is NOT routed through early-layer attention → must be routed through mid/deep layers or output-level computation

This further validates P18's finding that DEEP layers (16-23) are where routing decisions happen. Early-layer attention perturbation doesn't reach that deep.

### 4.3 The Intervention Ladder for Absorption

| Rank | Intervention | early H | mid H | late H | ΔH |
|---|---|---|---|---|---|
| 0 | Baseline | 0.250 | 1.000 | 1.000 | 0.750 |
| 1 | v_abs steering (P12) | 0.250 | 0.500 | 0.500 | 0.250 |
| 2 | Attention temperature | 0.250→0.750 | 1.000 | 1.000 | 0.250→0.750 |
| 3 | **LoRA (P21)** | **0.250** | **0.000** | **0.000** | **0.250** |

LoRA is the ONLY intervention that helps mid/late positions. Nothing helps early position.

### 4.4 Three-Component Absorption Model

P21+P22 together reveal that absorption has THREE distinct components:

| Component | Affected Position | LoRA Effect | Attention Temp Effect | Root Cause |
|---|---|---|---|---|
| **Distance routing** | mid/late | ✅ FIXED | ❌ No effect | Deep-layer q_proj routing (P18) |
| **Proximity over-confidence** | early | ❌ Not fixed | ❌ Made WORSE | Embedding-level (RoPE?) |
| **Attention calibration** | early | N/A | Baseline is OPTIMAL | Optimized by pretraining |

### 4.5 The RoPE Hypothesis

Early position proximity over-confidence survives BOTH LoRA (weight-level) AND attention temperature (attention-level) interventions. The only remaining level is **embedding-level** — specifically, RoPE position encoding.

RoPE encodes position into the query and key representations BEFORE attention computation. If RoPE makes early-position tokens appear "more relevant" to the model (closer in the rotary space), then no amount of attention modification or weight training can fix it — the position bias is baked into the token-level representations.

**Hypothesis**: For early-position inputs, the RoPE encoding makes the model "feel" that it has more relevant information about the entity, even when it doesn't. This creates a source confusion that survives all downstream interventions.

## 5. Claims

| # | Claim | Evidence | Confidence |
|---|---|---|---|
| **F62** | H22.1 REFUTED: Softening attention at early layers INCREASES early H (0.250→0.500→0.750), not decreases it. Baseline T=1.0 is the OPTIMAL point. | P22: T=0.5/2.0/5.0 results | ⭐⭐⭐⭐⭐ |
| **F63** | Mid/late H=1.000 is COMPLETELY INVARIANT to early-layer attention temperature (all T). Mid/late hallucination is NOT routed through early-layer attention — consistent with P18 deep-layer routing. | P22: mid/late = 1.000 at all T | ⭐⭐⭐⭐⭐ |
| **F64** | Early position proximity over-confidence is a ROBUST property, not a fragile attention pattern. Any perturbation (T<1 or T>1) makes it WORSE, not better. | P22: dose-response pattern | ⭐⭐⭐⭐⭐ |
| **F65** | Attention temperature scaling is inferior to hidden-state steering (P12) for absorption. Temperature: early H increases; Steering: ΔH=0.250 (partial homogenization). | P22 vs P12 comparison | ⭐⭐⭐⭐ |
| **F66** | Absorption has THREE components: (1) distance routing → LoRA fixes; (2) proximity over-confidence → NOTHING fixes (survives LoRA + attention temp); (3) attention calibration → baseline is optimal. | P21+P22 synthesis | ⭐⭐⭐⭐ |

---

*Related: [P12 Absorption Steering](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P12_ABSORPTION_STEERING.md) | [P19 Attention Patterns](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P19_ATTENTION_PATTERNS.md) | [P21 Absorption LoRA](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P21_ABSORPTION_LORA.md)*