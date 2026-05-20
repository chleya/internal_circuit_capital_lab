# IC-4 P1: Cross-Seed & Cross-Layer Validation Report

**Status**: Complete (v1.2 — P1.5 Confirmed) | **Date**: 2026-05-19 | **Pipeline**: M3-v6 Single-Pass Hook-Based Gate

## 0. P1.5 Audit Note (2026-05-19)

**The original v1.0 of this report contained an Oracle baseline error for the reference configuration.**

- Original (incorrect): ref seed=0 layer=12, oracle H=0.533
- Corrected (v1.1): ref seed=0 layer=12, oracle H=**0.667** (from `results_m3_v6/metrics_raw.csv` and `IC4_PROJECT_TERRAIN_MANUAL.md` §2)

The value 0.533 was inadvertently copied from seed=1's oracle. The reference run's ground-truth oracle H is 0.667.
This is a **document error**, not an experimental anomaly. The reference run has always been gate H = oracle H = 0.667.

No other rows were affected — seed=1, seed=2, layer=11, and layer=13 were correctly reported from their respective run_log.txt files.

### 0.1 P1.5 Confirmation (2026-05-19, v1.2)

**P1.5 (Failure Mode Analysis + Small-Data Patch Test) has confirmed that both P1 ARTIFACT verdicts were construction-regime artifacts, not mechanism failures.**

- seed=2 / layer=12: `cos(steering, shuffled)=0.788` under 15A+15U → collapsed to 0.439 under 30A+30U → causal ordering restored ✅
- seed=0 / layer=13: shuffled H=0.667 under 15A+15U → rose to 0.900 under 30A+30U → artifact eliminated ✅

**Consequence**: The M3-v6 reference mechanism passes causal separation tests on all tested seed/layer combinations under 30A+30U construction. The two ARTIFACT verdicts in this report should be understood as small-sample artifacts, not evidence of mechanism fragility.

Full P1.5 analysis: `reports/IC4_P15_FAILURE_ANALYSIS_REPORT.md`.

## 1. Objective

Validate that the M3-v6 hard gate generalises beyond the reference configuration (seed=0, layer=12) by testing:

- **Cross-seed**: seed=1, seed=2 (same layer=12)
- **Cross-layer**: layer=11, layer=13 (same seed=0)

Criterion for SUCCESS: causal ordering `random > shuffled > real_gate` holds, and `real_gate ≈ oracle`.

## 2. Complete Results Table (Corrected)

| # | Config | seed | layer | base H | gate H | oracle H | random H | shuffled H | Causal? | Gate≈Oracle? | Verdict | Time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ref | 0 | 12 | 0.867 | 0.667 | **0.667** | 0.933 | 0.800 | ✅ R>S>G | ✅ **exact** | SUCCESS | — |
| 2 | seed=1 | 1 | 12 | 0.967 | 0.533 | 0.533 | 0.967 | 0.733 | ✅ R>S>G | ✅ exact | SUCCESS | 31.6 min |
| 3 | seed=2 | 2 | 12 | 0.900 | 0.500 | 0.500 | 0.933 | 0.467 | ❌ S<G | ✅ exact | ARTIFACT | 30.8 min |
| 4 | layer=11 | 0 | 11 | 0.867 | 0.733 | 0.733 | 1.000 | 0.833 | ✅ R>S>G | ✅ exact | SUCCESS | 29.5 min |
| 5 | layer=13 | 0 | 13 | 0.867 | 0.700 | 0.700 | 0.933 | 0.667 | ❌ S<G | ✅ exact | ARTIFACT | 29.7 min |

> **Legend**: R = random vector, S = shuffled vector, G = real gate (steering vector), O = oracle gate.
> Causal ordering: `random > shuffled > real_gate` means the real steering vector produces the lowest hallucination rate among all controls.
> **Correction**: Row #1 oracle H was originally misquoted as 0.533 (copied from seed=1). Correct value is 0.667 (from ground-truth CSV). P1.5 audit confirmed this is a document error only.

### 2.1 Per-Experiment Detail

#### #1 Reference (seed=0, layer=12)
- base H=0.867 → real_gate H=0.667 (Δ = -0.200, 23% reduction)
- Causal ordering holds: random(0.933) > shuffled(0.800) > gate(0.667)
- gate H == oracle H EXACTLY (0.667 = 0.667) — **perfect match**
- Verdict: SUCCESS — steering vector shows genuine causal effect beyond controls, and matches oracle precisely

#### #2 Seed=1
- base H=0.967 → real_gate H=0.533 (Δ = -0.434, 45% reduction)
- Strongest hallucination suppression among all experiments
- Causal ordering holds: random(0.967) > shuffled(0.733) > gate(0.533)
- gate H == oracle H EXACTLY (0.533 = 0.533) — perfect replication
- Verdict: SUCCESS — strongest replication, all criteria met perfectly

#### #3 Seed=2
- base H=0.900 → real_gate H=0.500 (Δ = -0.400, 44% reduction)
- Causal ordering VIOLATED: shuffled H=0.467 < gate H=0.500
- Shuffled vector produces LOWER hallucination than real steering vector
- This means the steering direction is not uniquely causal — a random permutation can achieve similar or better effect
- gate H == oracle H EXACTLY (0.500 = 0.500) — perfect match but meaningless due to control failure
- Verdict: ARTIFACT — shuffled control beats real steering, direction not uniquely causal

#### #4 Layer=11
- base H=0.867 → real_gate H=0.733 (Δ = -0.134, 15% reduction)
- Weaker effect than layer=12 but causal ordering holds
- random(1.000) > shuffled(0.833) > gate(0.733)
- gate H == oracle H EXACTLY (0.733 = 0.733) — perfect match
- Verdict: SUCCESS — probe direction at layer=11 shows genuine causal effect

#### #5 Layer=13
- base H=0.867 → real_gate H=0.700 (Δ = -0.167, 19% reduction)
- Causal ordering VIOLATED: shuffled H=0.667 < gate H=0.700
- Same artifact pattern as seed=2: shuffled vector beats real steering
- gate H == oracle H EXACTLY (0.700 = 0.700) — but control failure invalidates
- Verdict: ARTIFACT — shuffled control beats real steering at this layer

## 3. Cross-Seed Analysis

| seed | gate H | oracle H | shuffled H | Causal? | Reliable? |
|---|---|---|---|---|---|
| 0 | 0.667 | 0.667 | 0.800 | ✅ | ✅ |
| 1 | 0.533 | 0.533 | 0.733 | ✅ | ✅ |
| 2 | 0.500 | 0.500 | 0.467 | ❌ | ❌ |

**Cross-seed pass rate: 2/3 (67%)**

Seed=2 produces an ARTIFACT where the shuffled control underperforms shuffled baselines from other seeds (0.467 vs 0.800/0.733). This suggests seed=2's train/test split may create an activations distribution where the probe-learned direction is less orthogonal to random noise — making the shuffled perturbation accidentally align with the anti-hallucination direction.

**Implication**: The M3-v6 gate is **seed-sensitive**. With only 15A+15U training samples, the learned probe direction is not robust to different random train/test splits. This is a strong indicator that we need either (a) larger training sets per seed, or (b) cross-seed ensemble probes (train on seed=0, test on seed=1/2).

## 4. Cross-Layer Analysis

| layer | gate H | oracle H | shuffled H | Causal? | Effect size (Δ from base) |
|---|---|---|---|---|---|
| 11 | 0.733 | 0.733 | 0.833 | ✅ | -0.134 (15%) |
| 12 | 0.667 | 0.667 | 0.800 | ✅ | -0.200 (23%) |
| 13 | 0.700 | 0.700 | 0.667 | ❌ | -0.167 (19%) |

**Cross-layer pass rate: 2/3 (67%)**

Layer=12 is the sweet spot: strongest effect size (23% H reduction) with proper causal ordering. Layer=11 and layer=13 both show weaker effects, with layer=13 additionally failing the control test.

This confirms the **layer sensitivity** of the probing approach:
- **Layer=11**: Probe works but effect is weaker (earlier in the network, hallucination signal is less concentrated)
- **Layer=12**: Optimal — strongest causal effect with clean control separation
- **Layer=13**: Unstable — direction exists but not robustly separable from random permutations

## 5. Meta-Analysis

### 5.1 Oracle Matching (Corrected)

**All 5 experiments show `gate H == oracle H` exactly.** After correcting the reference run's oracle to 0.667, there is no gap between gate and oracle in any configuration.

This is strong evidence that the **single-pass hook-based hard gate** mechanically produces oracle-equivalent hallucination rates — when the probe correctly classifies a sample as `P(hallucinate) > 0.5`, steering to the anti-hallucination direction reduces H to the oracle level. No gap exists in any run.

### 5.2 Control Artifact Pattern

Both ARTIFACT verdicts (seed=2, layer=13) share the same failure mode: **shuffled H < real_gate H**. This means the shuffled permutation of the steering vector produces a stronger anti-hallucination effect than the real steering vector itself.

Possible explanations:
1. **Small sample size**: With only 30 test samples, a single lucky shuffled direction can beat the real direction by chance
2. **Weak signal**: At seed=2 and layer=13, the hallucination-direction signal in activations may be weak enough that a random direction sometimes aligns better
3. **Probe overfitting**: CV AUC=1.0 across all experiments suggests the probe may overfit to the small training set, learning spurious directions that do not generalise

### 5.3 Overall Reliability

| Dimension | Pass Rate | Interpretation |
|---|---|---|
| Cross-seed | 2/3 (67%) | Acceptable but not robust — need larger N |
| Cross-layer | 2/3 (67%) | Layer=12 is optimal; adjacent layers are borderline |
| Combined | 3/5 (60%) | Promising but fragile at current sample size |

## 6. Recommendations (Updated P1.5)

### Completed
1. ~~Increase training sample size from 15A+15U to 30A+30U~~ → **Done** (P1.5 patch test confirmed both failures resolved)
2. ~~Small-data robustness patch test~~ → **Done** (P1.5: both seed=2 and layer=13 artifacts eliminated at 30A+30U)

### Now Standard
3. **30A+30U is the new default construction regime.** All future experiments must use ≥30A+30U.
4. **Anchor on layer=12** as the default probing site.
5. **Report causal ordering** (`random > shuffled > real_gate`) for every experiment.

### Near-term (P2)
6. **LoRA routing injection (P2)**: Use the layer=12 probe as the router signal for M7-H style LoRA-based anti-hallucination (requires GPU)

### Longer-term
7. **Multi-seed probe training**: Train on concatenated activations from multiple seeds to improve robustness
8. **Layer-wise probe sweep**: Test all 24 layers at scale to map the hallucination signal topology

## 7. Appendix: Data Provenance

| Config | Run Log | Ground Truth Oracle |
|---|---|---|
| ref | `results_m3_v6/run_log.txt` | 0.667 (CSV confirmed) |
| seed=1 | `results_m3_v6_p1_seed1/run_log.txt` | 0.533 |
| seed=2 | `results_m3_v6_p1_seed2/run_log.txt` | 0.500 |
| layer=11 | `results_m3_v6_p1_layer11/run_log.txt` | 0.733 |
| layer=13 | `results_m3_v6_p1_layer13/run_log.txt` | 0.700 |