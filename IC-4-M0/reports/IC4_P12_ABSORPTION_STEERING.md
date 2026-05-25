# IC-4 P12: Position-Directional Steering for Absorption — Negative

> **Date**: 2026-05-23 | **Status**: Completed (Negative — direction exists but effect is homogenization with degradation)
> **Type**: Experiment — open-loop activation steering α-sweep
> **Predecessors**: A1 (probe-level PSI -90%), A2 (behavior gap persists), A3 (inference rectification FAILED), A4 (LoRA mixed)

**Question**: Can open-loop activation steering with a position-direction vector
(h_early − h_late) reduce behavior-level position sensitivity, following the
proven sycophancy steering pattern?

**Answer: Partially, but with a problematic mechanism. The v_abs direction
reduces position variance (delta_H −67%) but does so by degrading the
best-performing position (early) while only moderately improving the worst
positions (mid/late). This is homogenization with degradation, not a clean
remedy.**

---

## 1. Background

### 1.1 The Absorption Remedy Chain

| Experiment | Method | Result |
|---|---|---|
| **A1** | Position-augmented probe training | ✅ PSI: 0.0676→0.0067 (−90.0%) at probe level |
| **A2** | Behavior position invariance test | ⚠️ Probe fixed (PSI −91.5%), but behavior gap persists (h_range=0.111) |
| **A3** | Inference-time position rectification | ❌ delta_H=1.0, completely ineffective |
| **A4** | LoRA position-aware training | ⚠️ delta_H −50% but H_early +67% (degradation) |

The pattern: probe-level absorption is fixable, but behavior-level position
sensitivity resists all remediation attempts. A3 (global rectification) is too
coarse; A4 (weight training) has side effects.

### 1.2 P12 Design

Following the proven sycophancy steering pattern (P5-bis open-loop α-sweep):

1. Compute v_abs = mean(h_early) − mean(h_late) at layer 10
   (points FROM late-position degradation TOWARD early-position information richness)
2. Open-loop steering during `model.generate()`: add α × v_abs to hidden states
3. α-sweep: −3.0 to +3.0 (5 levels)
4. Controls: random direction, orthogonal direction
5. Metrics: H(early), H(mid), H(late), delta_H, delta_C

---

## 2. Results

### 2.1 Steering Vector Characteristics

| Metric | Value |
|---|---|
| v_abs norm | 1.0 (unit vector) |
| ||h_early|| mean | 13.00 |
| ||h_late|| mean | 12.60 |
| ||h_early − h_late|| mean | 4.55 |

The position shift magnitude (4.55) is about 35% of the hidden state norm (~13).
This is a substantial effect — position is not a subtle perturbation.

### 2.2 α-Sweep Results (n=10 per position, 30 total)

| Condition | delta_H | delta_C | H_early | H_mid | H_late |
|---|---|---|---|---|---|
| baseline (α=0) | 0.750 | 0.333 | 0.250 | 1.000 | 1.000 |
| v_abs α=−3.0 | 0.250 | 0.333 | 0.750 | 1.000 | 1.000 |
| v_abs α=−1.5 | 0.250 | 0.333 | 0.750 | 1.000 | 1.000 |
| v_abs α=+1.5 | 0.250 | 0.500 | 0.750 | 1.000 | 1.000 |
| v_abs α=+3.0 | **0.250** | 0.167 | 0.500 | 0.750 | 0.750 |
| random α=−3.0 | 0.500 | 0.167 | 0.500 | — | 1.000 |

**Key observations:**

1. **Baseline floor/ceiling effect**: Mid and late positions are at CEILING
   (H=1.000, 100% hallucination). Only early position retains meaningful
   performance (H=0.250). This is a massive position gap.

2. **Any non-zero α degrades early**: Across ALL α≠0, H_early increases from
   0.250 to 0.500–0.750. Steering the model toward the position-direction
   uniformly damages the best-performing position.

3. **α=+3.0 is the only condition that helps mid/late**: H_mid drops from
   1.000→0.750 and H_late from 1.000→0.750. This is from "total hallucination"
   to "most hallucination" — modest but real. H_early simultaneously degrades
   from 0.250→0.500.

4. **Net effect**: α=+3.0 reduces total hallucination by 0.25 (early +0.25,
   mid −0.25, late −0.25) while reducing variance from 0.750 to 0.250.

### 2.3 Direction Specificity Comparison

| Comparison | v_abs value | Control value | Ratio |
|---|---|---|---|
| delta_H at \|α\|=3.0 | 0.250 | 0.500 (random) | **2.0×** |
| H_early at \|α\|=3.0 | 0.750 (α=−3.0) | 0.500 (random α=−3.0) | 1.5× worse |

v_abs has **2× stronger variance reduction** than random, but **1.5× more
damage to the best position**. This is direction-specificity of a problematic
kind — it "works" by more aggressively degrading the best-case scenario while
mildly helping the worst-case.

### 2.4 n=5 Ceiling Check

At n=5 (15 test samples), random and orthogonal vectors at α=+1.5 and +3.0
produce H=1.000 at ALL positions. Every single sample hallucinates. This
confirms that v_abs is directionally different from random — it does not
induce total behavioral collapse at the same perturbation magnitude.

---

## 3. Interpretation

### 3.1 The Homogenization Pattern

The v_abs steering vector's effect can be characterized as:

> **Position homogenization**: steering along the early→late direction pushes
> all positions toward a common intermediate state — degrading the best
> (early: 0.25→0.50) and improving the worst (mid/late: 1.00→0.75).

This is structurally similar to what was observed in A4's LoRA training
(delta_H improved but H_early degraded). Both methods converge the position
distribution by dragging down the best-case scenario. This is NOT a clean
remedy.

### 3.2 Comparison with Sycophancy Steering

| Dimension | Sycophancy (P5-bis) | Absorption (P12) |
|---|---|---|
| Steering vector | v_syc = h(syc) − h(non-syc) | v_abs = h(early) − h(late) |
| Optimal α | −3.0 (syc rate 0.58→0.38) | +3.0 (delta_H 0.75→0.25) |
| Clean improvement? | ✅ Yes (only syc affected) | ❌ No (early degraded) |
| Direction-specific? | ✅ v_syc/random = 2.73× | ⚠️ v_abs/random = 2.0× |
| Net benefit | −35.7% syc, C unchanged | Net −0.25 H, but early +100% |

The critical difference: sycophancy steering improved the target behavior
without collateral damage. Absorption steering "improves" by sacrificing
the best position. The former is a genuine gain; the latter is redistribution.

### 3.3 Why Does Absorption Resist Steering?

Three hypotheses:

1. **Energy-dominated, not direction-dominated**: Like hallucination (P10),
   position sensitivity may be primarily energy-based. The v_abs direction
   carries information but the system's response to perturbation is
   predominantly energy-driven, not direction-driven. The observed 2.0×
   ratio is weaker than sycophancy's 2.73× and much weaker than the initial
   6.17× found in T3 — suggesting a predominantly energy-based effect.

2. **Early-position privilege is structural**: Early position may have a
   genuine computational advantage (closer to the beginning of context
   processing). Steering along the position axis necessarily disrupts this
   advantage because it's moving states away from their naturally optimal
   configuration. You can't "steer" a structural advantage — you can only
   redistribute it.

3. **Ceiling effect contamination**: Mid/late positions at H=1.000 ceiling
   mean the only observable effect of perturbation is on early (which has room
   to degrade). The true directional effect on mid/late may be hidden behind
   the ceiling. A dataset with sub-ceiling mid/late baselines would be needed
   to properly test.

---

## 4. Formal Declaration

> **Position-directional activation steering is NOT a viable Absorption
> remedy.** While the v_abs direction exists and has a measurable 2.0×
> directional effect (vs random), the effect is homogenization with
> degradation — reducing variance by making the best position worse while
> only moderately helping the worst positions.
>
> The Absorption bottleneck remains the only bottleneck without a clean
> behavioral remedy. Probe-level absorption is fully fixable (A1: PSI −90%),
> but the probe→behavior gap resists all tested intervention types:
> global rectification (A3), weight training (A4), and directional steering
> (P12).

### Boundary Conditions

- **n=10 per position** — small sample, results may not generalize
- **Layer 10 only** — other layers not tested
- **Mid/late at ceiling** — prevents observing genuine directional improvement
  at worst positions
- **CPU inference** — generate quality may differ from GPU

---

## 5. Impact on Research Program

### Lines Affirmed
- **Absorption is the hardest bottleneck**: Unlike Stabilization (fully
  validated, P11) and Organization (partial remedies exist for both Hall
  and Syc), Absorption has no clean behavioral-level remedy.
- **Probe→behavior gap is fundamental**: A1 proved probe-level absorption is
  fixable. A2/A3/A4/P12 collectively prove behavior-level absorption resists
  all tested intervention types. This gap is not a hook architecture bug
  (unlike P6-bis) — it's a deeper structural asymmetry.

### Lines Opened
- **Absorption as energy-dominated**: If position sensitivity is energy-based
  rather than direction-based (like Hall), then directional steering is the
  wrong intervention class. Energy-based perturbations (noise, dropout,
  quantization) might be more appropriate.
- **Multi-layer or attention-level intervention**: P12 tested only L10.
  Position sensitivity may be distributed across layers (like sycophancy
  was at step 1, not S15). Testing other layers or attention heads may yield
  different results.
- **Sub-ceiling dataset needed**: The mid/late ceiling (H=1.000) limits
  observability. A more moderate baseline with sub-ceiling hallucination at
  all positions would allow proper α-sweep analysis.

---

## 6. Next Steps

| Priority | Action | Rationale |
|---|---|---|
| 1 | **Absorption: alternative intervention classes.** Energy-based methods (noise perturbation study, attention dropout), multi-layer steering, or architectural approaches (position embeddings). | Directional steering is insufficient. Need different approach class. |
| 2 | **Syc: larger-n (n≥48) confirmation.** | P8 direction correct but not significant. Low-hanging fruit. |
| 3 | **Cross-project integration & deliverable packaging.** | P9+P10+P11+P12 complete the diagnosis phase. Time to synthesize. |

---

*IC-4-M0 P12 Report — 2026-05-23*
*Absorption directional steering — Negative (homogenization with degradation)*