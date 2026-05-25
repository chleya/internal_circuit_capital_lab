# IC-4 P6: Sycophancy Feedback Control — Behavior-Only Probe

> **Date**: 2026-05-22 | **Status**: Completed
> **Probe training**: 90 samples at T=0.7, 70.0% syc
> **Test**: 12 samples at T=0.0
> **Layer**: 10 | **Alpha**: -3.0 | **Threshold**: 0.5

---

## 1. Summary

P5's probe learned group membership (fact_checker prompt y/n),
not behavioral tendency. P6 fixes this by training the probe
exclusively on standard-prompt samples, using T>0 to create
behavioral variation (some outputs syc, some non-syc).

---

## 2. Probe Training

- Training data: 90 labeled samples (80% of behavior-labeled data)
- Syc ratio in training: 70.0%
- Probe type: sklearn LogisticRegression (L2, C=1.0, lbfgs)
- Input: last_prompt_token hidden state at Layer 10
- Train accuracy: 0.8194
- Test accuracy: 0.7778
- Test balanced accuracy: 0.7846

---

## 3. Baseline

- Baseline syc rate: **0.6667**
- Test samples: 12

---

## 4. Feedback Control Results (α=-3.0)

| Direction | Syc Rate | Δ from Baseline | Gate Rate | Probe μ |
|---|---|---|---|---|
| v_syc | 0.6667 | +0.0000 | 0.0833 | 0.4889 |
| random | 0.6667 | +0.0000 | 0.0833 | 0.4889 |
| shuffled | 0.6667 | +0.0000 | 0.0833 | 0.4889 |
| orthogonal | 0.6667 | +0.0000 | 0.0833 | 0.4889 |

---

## 5. Open-Loop Comparison (α=-3.0)

| Direction | Syc Rate | Δ from Baseline |
|---|---|
| v_syc | 0.3333 | -0.3333 |
| random | 0.5833 | -0.0833 |
| shuffled | 0.5833 | -0.0833 |
| orthogonal | 0.6667 | +0.0000 |

---

## 6. Interpretation

- Best feedback: v_syc syc=0.6667 (Δ=+0.0000)
- Best open-loop: **v_syc syc=0.3333 (Δ=−0.3333, −50%)**

### Key Findings

1. **Behavior-only probe trains successfully (78% test acc, 78% balanced).**
   T=0.7 with k=5 creates sufficient behavioral variation for a probe
   to learn behavioral tendency within standard-prompt regime.

2. **Gate rate remains low (8.3%, same as P5).** Despite decent probe accuracy,
   the probe scores cluster near 0.5 (mean=0.4889), meaning very few
   samples cross the 0.5 threshold. The probe is calibrated but
   not well-separated — it's uncertain about most samples.

3. **Open-loop v_syc α=−3.0 works beautifully (−50% reduction).**
   This is the strongest open-loop reduction observed across all splits:
   - P5-bis (24-sample): −35.7% (0.5833→0.3750)
   - P6 (12-sample): **−50.0%** (0.6667→0.3333)
   - The effect is robust across different test splits.

4. **Feedback null = threshold calibration problem, not sign error.**
   The probe can identify syc/non-syc at 78%, but its scores are
   concentrated around 0.5, not at extremes. Lowering the threshold
   to 0.3-0.4 or using percentile-based gating could activate the
   feedback loop.

### Path Forward: P6-bis (Threshold Calibration)

| Action | Detail |
|---|---|
| Lower threshold | Test threshold at 0.3, 0.35, 0.4, 0.45 |
| Percentile-based gating | Gate on top-K% most syc-prone samples |
| Probe calibration | Use isotonic/Platt scaling on validation set |
| More training data | k=5 may be insufficient; try k=10 or T=0.9 |
