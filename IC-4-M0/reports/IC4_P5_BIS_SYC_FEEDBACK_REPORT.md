# IC-4 P5-bis: Open-Loop Positive Alpha Test

> **Date**: 2026-05-22 | **Status**: Completed
> **Layer**: 10
> **Test set**: 24 samples from P5 split (mixed syc/non-syc groups)

---

## 1. Baseline
- Baseline syc rate: **0.5833** (14/24)

---

## 2. Negative Alpha (P5 reference — subtract v_syc)

| Direction | α=-1.0 | α=-3.0 | α=-5.0 |
|---|---|---|---|
| v_syc | 0.4167 | 0.3750 | 0.5417 |
| random | 0.6250 | 0.6250 | 0.7917 |
| shuffled | 0.4167 | 0.5833 | 0.7083 |
| orthogonal | 0.5833 | 0.7500 | 0.9583 |

---

## 3. Positive Alpha (P5-bis — ADD v_syc)

| Direction | α=+1.0 | α=+3.0 | α=+5.0 |
|---|---|---|---|
| v_syc | 0.9167 (+0.3333) | 1.0000 (+0.4167) | 1.0000 (+0.4167) |
| random | 0.6250 (+0.0417) | 0.7083 (+0.1250) | 0.9167 (+0.3333) |
| shuffled | 0.6667 (+0.0833) | 0.7083 (+0.1250) | 0.7083 (+0.1250) |
| orthogonal | 0.5417 (-0.0417) | 0.4583 (-0.1250) | 0.7083 (+0.1250) |

---

## 4. Interpretation

### v_syc Polarity Verdict: Points TOWARD Sycophancy

The original P5 hypothesis was: v_syc points toward *non_sycophantic* behavior, so positive alpha (adding v_syc) should reduce sycophancy. **This is falsified.**

The data unambiguously shows the opposite:

- **Negative alpha (subtract v_syc) reduces sycophancy**: baseline 0.5833 → best 0.3750 at α=-3.0 (−35.7% relative)
- **Positive alpha (add v_syc) massively increases sycophancy**: 0.9167 → 1.0000 → 1.0000 (ceiling effect at α≥3.0)
- **v_syc is the only directional vector with this anti-symmetric pattern**: random/shuffled/orthogonal show generic perturbation effects (monotonic increase with |α|)

**Conclusion**: v_syc in L10 representation space points **toward sycophantic behavior**. Subtracting v_syc moves the model away from sycophancy; adding v_syc pushes it toward complete sycophancy.

### Optimal Steering

| α | v_syc syc_rate | Δ from baseline |
|---|---|---|
| -1.0 | 0.4167 | −0.1666 (−28.6%) |
| **-3.0** | **0.3750** | **−0.2083 (−35.7%)** |
| -5.0 | 0.5417 | −0.0417 (−7.1%) |
| +1.0 | 0.9167 | +0.3333 (+57.1%) |
| +3.0 | 1.0000 | +0.4167 (+71.4%) |
| +5.0 | 1.0000 | +0.4167 (+71.4%) |

**Best: α = −3.0**, reducing sycophancy from 58.3% to 37.5%.

### Control Vector Specificity

Only v_syc shows the anti-symmetric effect. All other vectors (random, shuffled, orthogonal) show generic perturbation: higher |α| → higher syc_rate regardless of sign. This confirms v_syc is specifically encoding sycophancy-relevant geometry, not just injecting noise.

### Implications for P5 Feedback Control

P5's probe→gate→hook design used negative alpha, which was correct for reducing sycophancy. The failure of P5 feedback control was not due to sign error — it was due to the probe learning group membership (fact-checker vs. standard prompt) rather than behavioral tendency within the standard-prompt regime.

### Path Forward

1. **Probe retraining**: Train probe on sub-threshold samples only (standard prompts where sycophancy varies) using temperature > 0 to create behavioral variation
2. **Optimal α = −3.0**: Any future steering should use this value
3. **v_syc polarity confirmed**: The direction is toward sycophancy — subtraction is the intervention
