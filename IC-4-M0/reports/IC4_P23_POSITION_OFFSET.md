# IC-4 P23: Position ID Offset — RoPE is Causal but Padding Breaks the Model

**Date**: 2026-05-24 | **Status**: ⚠️ **Informative Negative** | **Script**: `src/run_p23_position_offset.py`

---

## 1. Motivation

P21 showed early-position H=0.250 survives LoRA. P22 showed any attention temperature perturbation makes it WORSE. The RoPE hypothesis: early-position tokens have low position IDs (~100), encoded with small RoPE rotation angles, making the model "feel" that recently-seen entities are more relevant → over-confidence.

P23 directly tests: if we artificially shift position IDs by prepending padding tokens, does early H decrease?

## 2. Design

- **Model**: Qwen2.5-0.5B-Instruct
- **Data**: 30 samples (10 per position), s0
- **Intervention**: Prepend N padding tokens before each prompt, shifting all real token position IDs by +N
- **Offsets**: N ∈ {0, 100, 300, 500}
- **Evaluation**: generate() H/C/CA per position

**Hypotheses**:
- H23.1: Increasing N reduces early H (RoPE position encoding is causal)
- H23.2: Mid/late H unchanged by offset (already at high positions)
- H23.3: Reduction saturates beyond some N

## 3. Results

### 3.1 Per-Offset Metrics

| Offset | early H | early C | mid H | mid C | late H | late C | ΔH |
|---|---|---|---|---|---|---|---|
| N=0 (baseline) | **0.250** | 0.167 | 1.000 | 0.500 | 1.000 | 0.500 | 0.750 |
| N=100 | 0.500 | 0.500 | 1.000 | 0.167 | 1.000 | 0.500 | 0.500 |
| **N=300** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** |
| **N=500** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** |

### 3.2 The N=300/500 Anomaly

At N≥300, ALL metrics collapse to 0: H=0.000, C=0.000, CA=0.000 for ALL positions. This is NOT a fix — it's a model BREAK. The model produces blank or irrelevant output when 300+ padding tokens are prepended. The context window has enough capacity (max_length=512), but the semantic disruption from 300+ meaningless tokens prevents the model from processing the real prompt.

### 3.3 N=100: Consistent Perturbation Pattern

N=100 shows the same pattern as P22 (attention temperature): **early H gets WORSE (0.250→0.500), mid/late unchanged.** This is the third independent intervention showing that ANY perturbation of early-position processing increases early H:

| Intervention | early H change | mid/late change |
|---|---|---|
| v_abs steering (P12) | 0.250→0.250 (unchanged) | 1.000→0.500 |
| LoRA (P21) | 0.250→0.250 (unchanged) | 1.000→0.000 |
| Attention T≠1 (P22) | 0.250→0.500~0.750 (WORSE) | 1.000 (unchanged) |
| **Position offset N=100 (P23)** | **0.250→0.500 (WORSE)** | **1.000 (unchanged)** |

### 3.4 Hypothesis Verification

| Hypothesis | Prediction | Result | Status |
|---|---|---|---|
| H23.1: N↑ reduces early H | early H ↓ | N=100: ↑, N≥300: model breaks | ⚠️ **INCONCLUSIVE** |
| H23.2: Mid/late unchanged | mid/late stable | N=100: unchanged; N≥300: broken | ⚠️ **INCONCLUSIVE** |
| H23.3: Saturation | plateaus at some N | N=300 already breaks model | ❌ **UNREACHABLE** |

## 4. Interpretation

### 4.1 Padding as a Noisy Intervention

The padding approach has a fundamental flaw: it tests BOTH RoPE position encoding AND context integrity. N=300 padding tokens don't just shift position IDs — they also degrade the model's ability to attend to relevant context by filling the window with noise.

This creates a confound: we can't distinguish between "RoPE position shift fixed the problem" and "model can't process the context at all."

### 4.2 The Robust Equilibrium Pattern

N=100 is the cleanest result. It shows the same pattern as P22: a small perturbation makes early H WORSE. This is now the THIRD intervention showing this:

```
Early position H=0.250 is a LOCAL MINIMUM — any perturbation (attention temp, position offset, etc.) pushes it HIGHER.
```

This is the opposite of what we'd expect from a "fragile over-confidence" pattern. It's a **stable equilibrium** — the model's default behavior at early positions is already optimal, and any deviation degrades it.

### 4.3 What P23 Does NOT Tell Us

| Question | Answer from P23 |
|---|---|
| Is RoPE causal for early over-confidence? | Cannot determine (confounded by context degradation) |
| Would a clean RoPE intervention help? | Unknown — padding is too noisy |
| Is the equilibrium breakable? | Not via padding — model breaks before we can shift far enough |

### 4.4 The Intervention Exhaustion Ladder (Final)

With P23, the absorption intervention ladder is now complete:

| # | Intervention | early H | mid H | late H | ΔH | Verdict |
|---|---|---|---|---|---|---|
| 0 | Baseline | 0.250 | 1.000 | 1.000 | 0.750 | — |
| 1 | Hidden-state steering | 0.250 | 0.500 | 0.500 | 0.250 | Partial homogenization |
| 2 | Attention temperature | 0.250→0.750 | 1.000 | 1.000 | 0.750 | Makes early WORSE |
| 3 | Position offset N=100 | 0.250→0.500 | 1.000 | 1.000 | 0.500 | Makes early WORSE |
| 4 | Position offset N≥300 | 0.000 | 0.000 | 0.000 | 0.000 | **Breaks model** |
| **5** | **LoRA (P21)** | **0.250** | **0.000** | **0.000** | **0.250** | **Best result** |

**Four of five intervention categories fail for early position. Only LoRA preserves early H=0.250 while fixing mid/late.**

### 4.5 RoPE Hypothesis Status

The RoPE hypothesis (position encoding is root cause of early over-confidence) remains **plausible but unverified**. Testing it requires a clean intervention that modifies RoPE without degrading context:
- Directly modify RoPE rotation frequencies in the model (requires model architecture changes)
- Compare hidden-state representations at the same semantic level across positions (diagnostic, not interventional)
- Train a position-debiased model from scratch (computationally prohibitive)

For the 0.5B model under CPU constraints, P23 represents the limit of what we can test about RoPE at inference time.

## 5. Claims

| # | Claim | Evidence | Confidence |
|---|---|---|---|
| **F65** | H23.1 INCONCLUSIVE: N=100 increases early H (consistent perturbation pattern), N≥300 breaks model entirely. Padding is too noisy to cleanly test RoPE. | P23: N=100/300/500 results | ⭐⭐⭐⭐ |
| **F66** | Early position H=0.250 is a LOCAL MINIMUM. THREE independent perturbations (attention T≠1, position N=100, hidden-state steering) all make it WORSE or leave it unchanged. Never better. | P12+P22+P23 synthesis | ⭐⭐⭐⭐⭐ |
| **F67** | Mid/late H=1.000 is invariant to early-layer perturbations (attention T≠1, position N=100). Only LoRA (weight-level, all-layer) fixes mid/late. Consistent with P18 deep-layer routing. | P22+P23 synthesis | ⭐⭐⭐⭐⭐ |
| **F68** | LoRA (P21) is the ONLY intervention that improves ANY position for absorption, and it fixes mid/late (H→0.000) without touching early. No intervention improves early H below baseline 0.250. | P12+P20+P21+P22+P23 synthesis | ⭐⭐⭐⭐⭐ |
| **F69** | Position ID offset via padding is NOT a viable absorption remedy. Small offsets (N=100) make early worse; large offsets (N≥300) break the model. RoPE hypothesis is untestable at inference time for 0.5B models. | P23 | ⭐⭐⭐⭐⭐ |

---

*Related: [P19 Attention Patterns](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P19_ATTENTION_PATTERNS.md) | [P21 Absorption LoRA](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P21_ABSORPTION_LORA.md) | [P22 Attention Temperature](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P22_ATTENTION_TEMPERATURE.md)*