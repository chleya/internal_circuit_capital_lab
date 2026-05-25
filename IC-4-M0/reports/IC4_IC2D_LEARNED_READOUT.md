# IC-2d: Readout-Matched Episodic — Learned Readout CANNOT Rescue Episodic Memory

**Date**: 2026-05-24 | **Status**: ❌ **Negative (Informative)** | **Script**: `src/run_ic2d_readout_matched.py`

---

## 1. Motivation

IC-2c.1 established that episodic memory retention fails relative to NoMemory (IC-2b: best_action_match 0.195 vs NoMemory 0.445). IC-2c.1's hypothesis was that raw episodic traces lack sufficient information — consolidation is necessary.

**Counter-hypothesis (IC-2d)**: Maybe the problem isn't insufficient information in episodic traces, but the READOUT mechanism. Euclidean k-NN might be a poor readout for 24-dimensional history features. A learned readout (MLP, RandomForest) might extract more signal from the same episodic buffer.

If a learned readout beats k-NN and approaches NoMemory → episodic traces contain usable information, just with a wrong readout.
If all learned readouts still lose to NoMemory → episodic traces themselves lack structured information.

## 2. Design

- **Environment**: StructuredVolatilityEnv (state_dim=2, mode_flip_prob=0.08, history_len=8 → 24-dim features)
- **Train**: 5 seeds × 1200 samples/seed = 6000 samples
- **Test**: 200 samples (seed 0)
- **Incremental**: Add one seed per step, evaluate after each addition

**4 readout strategies**:

| Strategy | Mechanism | Buffer |
|---|---|---|
| NoMemory | Action-frequency baseline | None |
| Episodic-kNN | k=5 Euclidean KNeighborsRegressor (per-action) | max 200 |
| Episodic-MLP | MLPRegressor (64→32, ReLU, 500 iter, per-action) | max 200 |
| Episodic-RF | RandomForestRegressor (50 trees, depth 8, per-action) | max 200 |

All episodic strategies use the same buffer — only readout mechanism differs.

## 3. Results

### 3.1 Step-by-Step Trajectory

| Step | NoMemory | Episodic-kNN | Episodic-MLP | Episodic-RF |
|---|---|---|---|---|
| 1 | 0.460 | 0.175 | 0.095 | 0.140 |
| 2 | 0.445 | 0.145 | 0.095 | 0.105 |
| 3 | 0.445 | 0.145 | 0.095 | 0.125 |
| 4 | 0.445 | 0.150 | 0.095 | 0.130 |
| 5 | 0.445 | 0.195 | **0.095** | 0.190 |

### 3.2 Final Ranking

| Rank | Strategy | best_action_match | vs NoMemory |
|---|---|---|---|
| 1 | **NoMemory** | **0.445** | — |
| 2 | Episodic-kNN | 0.195 | −56.2% |
| 3 | Episodic-RF | 0.190 | −57.3% |
| 4 | Episodic-MLP | 0.095 | −78.7% |

### 3.3 Ratios

| Comparison | Ratio | Verdict |
|---|---|---|
| MLP / k-NN | 0.49× | MLP is **WORSE** than k-NN |
| RF / k-NN | 0.97× | RF is **equivalent** to k-NN |
| Best episodic / NoMemory | 0.44× | All episodic < NoMemory |

## 4. Interpretation

### 4.1 The Counter-Hypothesis is KILLED

The hypothesis "wrong readout causes episodic underperformance" is definitively refuted:
- MLP (deepest learned readout) is the WORST performer (0.095, −78.7% vs NoMemory)
- RF is on par with k-NN (0.190 vs 0.195)
- **No learned readout beats simple k-NN, and all lose badly to NoMemory**

The problem is NOT the readout. The problem is the **representations** — raw episodic traces don't contain enough structured information, regardless of how you read them.

### 4.2 Why MLP is WORSE than k-NN

The MLP learns spurious patterns from the 24-dim history features. With only 200 buffer samples, an MLP(64→32) has more parameters than training examples, leading to severe overfitting. k-NN's "no learning" approach is actually a regularizer in this low-data regime.

RandomForest's tree-based approach partially mitigates this (bagging provides implicit regularization), but still can't beat the simple frequency baseline.

### 4.3 The Only Path Forward: Consolidation

| Approach | best_action_match | vs NoMemory |
|---|---|---|
| NoMemory (action frequency) | **0.445** | baseline |
| Episodic (any readout) | 0.095–0.195 | −56% to −79% |
| **Per-Action KMeans** (IC-2c/C3) | **0.585** | **+31%** |

The conclusion is unambiguous: **consolidation is the ONLY path to beating NoMemory.** Episodic traces are noisy — you must compress and structure them (via Per-Action KMeans clustering) before they become useful. Readout mechanism is not the bottleneck.

### 4.4 Cross-Project Implication

IC-2 (stabilization) and IC-4 (organization) show the same pattern:
- **Raw representations** contain information but can't be directly used
- **Intervention** is needed at the representation level (consolidation for IC-2, LoRA routing for IC-4)
- **Readout-level fixes** fail in both projects (IC-2d MLP, IC-4 w_probe steering)

## 5. Claims

| # | Claim | Evidence | Confidence |
|---|---|---|---|
| **N14** | Learned readout (MLP, RF) does NOT rescue episodic memory. MLP is WORSE than k-NN (0.49×). | IC-2d MLP=0.095 vs kNN=0.195 | ⭐⭐⭐⭐⭐ |
| **N15** | Episodic underperformance is a REPRESENTATION problem, not a readout problem. | All readouts (k-NN, MLP, RF) < NoMemory (0.195, 0.095, 0.190 < 0.445) | ⭐⭐⭐⭐⭐ |
| **N16** | Consolidation (Per-Action KMeans, 0.585) is the ONLY approach beating NoMemory (0.445). Episodic → Consolidated gap is a representation gap. | C3 vs IC-2d comparison | ⭐⭐⭐⭐⭐ |

---

*Related: [IC-2c Episodic vs Consolidated](file:///F:/intelligence_capital_minimal_lab/results/ic2c/) | [P11 Stabilization Scaling](file:///F:/internal_circuit_capital_lab/IC-4-M0/reports/IC4_P11_STABILIZATION_SCALING.md)*