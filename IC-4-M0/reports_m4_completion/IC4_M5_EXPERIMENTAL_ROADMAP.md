# IC-4-M5: Post-Completion Experimental Roadmap

> Generated after M4-Completion verified cross-seed [0,1,2] × cross-layer [11,12,13] robustness.
> All 9 (seed, layer) settings achieved oracle_gap=0.000 on standard scenario.
> This document designs the next experiments organized by terrain section and priority.

---

## 0. Current Terrain State (Post M4-Completion)

### Closed Terrain ✓

| Terrain Section | Item | Status |
|---|---|---|
| 10.2 | Cross-seed gate validation (standard scenario) | **DONE** — seeds [0,1,2] at layer=12, all oracle_gap=0.000 |
| 10.2 | Cross-layer gate validation (standard scenario) | **DONE** — layers [11,12,13] at seed=0, all oracle_gap=0.000 |
| 10.2 | Cross-seed × cross-layer full matrix | **DONE** — 9/9 settings oracle_gap=0.000 |
| 10.3 | Imperfect-probe regime (standard data) | **DONE** — A2 degradation curve exists |
| — | A2 oracle_gap consistency | **FIXED** — harder_eval corrected (0.0663→0.000) |
| — | Branch B scaffold | **DONE** — framework + note created |

### Verified Facts (Post M4-Completion)

| Fact | Evidence |
|---|---|
| Oracle_gap=0.000 across seeds | s0_l12=0.000, s1_l12=0.000, s2_l12=0.000 |
| Oracle_gap=0.000 across layers | s0_l11=0.000, s0_l12=0.000, s0_l13=0.000 |
| Layer 12 is optimal | Lowest oracle H in all seeds |
| Random separation robust | 8/9 settings: random H > hard H |
| C preserved | max |ΔC| = 0.033 (within seed) |

### New Questions Raised by M4-Completion

| Question | Data Point |
|---|---|
| Why does shuffled=oracle at some (seed,layer)? | s2_l12: shuff=0.500, oracle=0.500 |
| Why does s1 get UA=0.033 at layer 11? | s1_l11: hard gate UA=0.033, only non-zero UA |
| Why do style_only_scores appear in seed=2? | s2: all non-base modes show style score 0.07-0.26 |
| Does cross-scenario hold on cross-seed? | Only standard validated; large/hard OOD not cross-validated |
| Does alpha sweep hold on cross-seed? | -0.8/-1.0/-1.2 validated only at s0_l12 |
| Why does oracle H vary so much by seed? | s0=0.667, s1=0.533, s2=0.500 at layer=12 |

---

## 1. Experiment Priority Framework

| Priority | Terrain Section | Theme | Timing |
|---|---|---|---|
| **P0** | 10.2 extension | Close remaining validation gaps (scenario × seed, alpha × seed) | Now |
| **P1** | New finding | Investigate shuffled anomaly + UA artifacts | After P0 |
| **P2** | 10.5 | Propagation analysis (layer-wise intervention mapping) | After P1 |
| **P3** | 10.4 | Branch B: Factuality hallucination (first cross-behavior) | After P2 |
| **P4** | New | Cross-model transfer | Later |

---

## 2. P0: Close Remaining Validation Gaps

### Exp-P0-A: Cross-Scenario on Cross-Seed

**Motivation:** M4-Completion validated standard scenario only. M4 generalization showed large and hard OOD work at seed=0. Need to confirm cross-seed on harder scenarios.

**Design:**

```
seeds: [0, 1, 2]
layer: 12
alpha: -1.0
scenarios: [standard, large, hard_ood]
modes: [base, oracle_gate, single_pass_hard_gate, random_hard_gate, shuffled_hard_gate]
```

**Matrix (15 settings):**

| Seed | Scenario | What we already have |
|---|---|---|
| 0 | standard | ✓ (M4-completion) |
| 0 | large | ✓ (M4-generalization) |
| 0 | hard_ood | ✓ (M4-generalization) |
| 1 | standard | ✓ (M4-completion) |
| 1 | large | **NEED** |
| 1 | hard_ood | **NEED** |
| 2 | standard | ✓ (M4-completion) |
| 2 | large | **NEED** |
| 2 | hard_ood | **NEED** |

**Expected new work:** 6 new settings × 5 modes × ~60 test samples ≈ 1800 generations (~2-3 hours CPU).

**Success criteria:**
- oracle_gap ≤ 0.05 in all 6 new settings
- random separation ≥ 0.08 in all 6 new settings
- C stability: |ΔC| ≤ 0.10 from base

**Question answered:** Does the scenario robustness (previously shown at seed=0) generalize across seeds?

---

### Exp-P0-B: Alpha Sweep on Cross-Seed

**Motivation:** M4 generalization showed -0.8, -1.0, -1.2 all work at seed=0/layer=12. Need to confirm this holds for seeds 1 and 2.

**Design:**

```
seeds: [1, 2]
layer: 12
alpha: [-0.8, -1.0, -1.2]
scenario: standard (use existing data to keep scope manageable)
modes: [base, oracle_gate, single_pass_hard_gate]
```

**Note:** -1.0 already verified for both seeds. New: -0.8 and -1.2.

**Matrix (4 new settings):**

| Seed | Alpha | oracle_gap target |
|---|---|---|
| 1 | -0.8 | ≤ 0.05 |
| 1 | -1.2 | ≤ 0.05 |
| 2 | -0.8 | ≤ 0.05 |
| 2 | -1.2 | ≤ 0.05 |

**Expected work:** 4 settings × 3 modes × 30-60 samples ≈ 540 generations (~45 min CPU).

**Question answered:** Is the alpha tolerance window seed-invariant?

---

## 3. P1: Investigate New Anomalies from M4-Completion

### Exp-P1-A: Shuffled Separation Analysis

**Motivation:** In 3/9 settings, shuffled gate H matches or beats oracle H:
- s0_l11: shuff=0.733, oracle=0.733 (zero separation)
- s2_l11: shuff=0.633, oracle=0.667 (shuffled BEATS oracle!)
- s2_l12: shuff=0.500, oracle=0.500 (zero separation)

**Hypotheses:**
1. **Subspace overlap:** At some (seed, layer) combos, the mean-diff subspace contains directionally informative components even after shuffling. The shuffling operation may preserve some subspace structure.
2. **Norm artifact:** The shuffled vector has the same L2 norm as the real vector. If any vector with sufficient norm in the right subspace has effect, shuffling won't destroy it.
3. **Layer specificity:** Early layers (11) may have a wider effective subspace, making any perturbation in that ballpark effective.

**Design (diagnostic, not validation):**

```
Phase 1: Norm-controlled comparison
  - Compare: real_v, shuffled_v, random_v, norm_matched_random_v
  - All at s0_l12 (known good separation) and s2_l12 (known zero separation)
  - This isolates whether norm alone explains the anomaly

Phase 2: Cosine similarity analysis
  - For each (seed, layer): compute cos_sim(real_v, shuffled_v)
  - If cos_sim is high at shuffled≈oracle settings, shuffling preserves too much structure
  - If cos_sim is low, the effect is due to norm/subspace rather than direction

Phase 3: Subspace projection
  - Project random vector onto mean-diff subspace
  - Project shuffled vector onto random subspace
  - Compare effective magnitude in the mean-diff direction
```

**Expected work:** ~30 min of analysis code + ~1 hour of computation.

**Question answered:** Why does shuffled gate sometimes achieve oracle level?

---

### Exp-P1-B: UA Artifact Investigation

**Motivation:** Only one setting shows non-zero UA:
- s1_l11 hard gate: UA=0.033 (1 out of 30 answerable samples unnecessarily steered)

All other 8 settings have UA=0.000. This is likely a probe edge case, not a mechanism problem.

**Design:**
1. Identify the specific sample where UA occurs
2. Check its probe score vs threshold
3. Confirm it's within expected statistical noise (1/30 ≈ 0.033, with probe cv_acc=0.933)
4. If reproducible, test if slightly higher threshold eliminates it without affecting H

**Expected work:** ~15 min analysis.

**Question answered:** Is the sole UA instance a probe artifact or a mechanism concern?

---

### Exp-P1-C: Style Score Analysis (Seed 2)

**Motivation:** Seed 2 shows non-zero style_only_score in all non-base modes:
- s2_l12 oracle: 0.196, hard: 0.196, random: 0.071, shuffled: 0.261
- Other seeds show style=0.000 for most modes

**Hypothesis:** Seed 2's entity pool happens to include company/person names that interact differently with the steering vector — the model may change surface form (not just answerability) under steering on these entities.

**Design:**
1. Extract samples with high style score in seed 2
2. Compare generated outputs base vs. steering on those samples
3. Check if style change is systematic (e.g., model uses more hedge words) or entity-specific
4. Cross-reference with M3-v6 style scores at seed=0

**Expected work:** ~30 min analysis.

**Question answered:** Is the style score increase in seed 2 a data artifact or a real steering side effect?

---

## 4. P2: Propagation Analysis (Terrain 10.5)

### Exp-P2-A: Layer-Wise Intervention Effect Mapping

**Motivation (from terrain 10.5):**
> "How the intervention propagates across layers" and "whether successful intervention corresponds to a more stable internal propagation regime."

We now have three intervention layers (11, 12, 13) fully validated. The next question: when we steer at layer L, what happens at layers L+1, L+2, ...?

**Design:**

```
seed: 0
intervention layer (steering hook): [11, 12, 13]
readout layer (probe location): [11, 12, 13, 14, 15]
modes: [base, hard_gate, oracle_gate, random_gate]
```

For each (intervention_layer, readout_layer) pair:
1. Record probe prediction at readout layer (with probe trained at that layer)
2. Record hidden state displacement ||h_steered - h_base|| at readout layer
3. Measure whether the steer signal persists through depth

**Key questions:**
- Does steering at layer 11 propagate to layer 12/13 probe readout?
- Does steering at layer 12 maintain its effect through layer 13/14/15?
- Is the propagation profile different for real vs random vectors?
- Does the oracle_gap remain 0.000 when probe is at a different layer than steering?

**Expected work:** ~4 hours CPU (probing at multiple layers without full generation).

**Question answered:** How does the steering intervention propagate through transformer depth?

---

### Exp-P2-B: Steering × Probe Layer Combinatorics

**Motivation:** All current results use probe trained at the SAME layer as steering. What happens when probe and steering are at DIFFERENT layers?

**Design:**

```
seed: 0
steering layer: 12
probe layer: [10, 11, 12, 13, 14]
scenario: standard
modes: [base, hard_gate, oracle_gate]
```

This tests whether the optimal probe layer is the same as the optimal steering layer.

**Expected work:** ~1 hour CPU.

**Question answered:** Is layer 12 optimal for both probe readout AND steering injection, or are these separate optima?

---

## 5. P3: Branch B — Factuality Hallucination (Terrain 10.4)

### Exp-P3-A: Factuality Probe Training

**Motivation (from terrain 10.4 + Branch B note):**
> "The broader framework is not limited to unanswerable hallucination."

Factuality is the most natural first extension because:
- Same steering direction (anti-hallucination) may partially transfer
- Probe training data can reuse the M3 infrastructure
- Evaluation rules are analogous

**Design (Phase 1: Probe only, no steering):**

```
1. Create factuality dataset:
   - Use M3 data generator but modify: every question IS answerable
   - Half: model answers correctly (baseline)
   - Half: the correct answer is replaced with a factually FALSE but plausible answer
   - Label: "factually_correct" vs "factually_incorrect"
   - Same company/person/location pools as M3

2. Train factuality probe:
   - Representation: last_prompt_token (align with current mechanism)
   - Layer: 12
   - Train on 30 train samples, test on 60

3. Baseline evaluation:
   - Measure how often the model produces factually false answers naturally
   - Check if the anti-hallucination steering vector also reduces factual errors
   - This tests VECTOR TRANSFER without new activation collection
```

**Expected work:** ~1-2 hours (data generation + probe training + baseline eval).

**Question answered:** Does the anti-hallucination steering vector also help with factuality hallucination?

---

### Exp-P3-B: Factuality-Specific Steering Vector

**If Exp-P3-A shows vector transfer DOESN'T work:**
1. Collect factuality activation pairs (correct vs. false responses)
2. Compute factuality-specific steering vector
3. Re-run gate with factuality-specific vector

**Expected work (if needed):** ~1 hour activation collection + 30 min eval.

---

## 6. P4: Cross-Model Transfer (New Terrain)

### Exp-P4-A: Qwen2.5-1.5B Replication

**Motivation:** All current results are on Qwen2.5-0.5B-Instruct. Is the mechanism model-specific?

**Design:**
1. Load Qwen2.5-1.5B-Instruct (or 0.5B base)
2. Run M3-v6 reference protocol: data generation → activation collection → probe training → gate eval
3. At seed=0, layer=middle (relative), standard scenario
4. Compare oracle_gap, control separation, C stability

**Note:** This is aspirational. The 1.5B model may be too slow on CPU. Start with 0.5B base (not instruct) if available, as a lighter cross-model test.

**Expected work:** ~4-6 hours (model download + full pipeline).

**Question answered:** Does the mechanism transfer to a different model?

---

## 7. Execution Sequence

```
Phase 1 (P0-A): Cross-Scenario on Cross-Seed
  ├── Generate large/hard_ood data for seeds 1, 2
  ├── Run 6 validation settings
  └── Output: results_m5_p0a/

Phase 2 (P0-B): Alpha Sweep on Cross-Seed
  ├── Run 4 validation settings (-0.8, -1.2 for seeds 1, 2)
  └── Output: results_m5_p0b/

Phase 3 (P1): Anomaly Investigation
  ├── P1-A: Shuffled separation analysis
  ├── P1-B: UA artifact check
  ├── P1-C: Style score analysis
  └── Output: reports_m5/IC4_M5_ANOMALY_REPORT.md

Phase 4 (P2): Propagation Analysis
  ├── P2-A: Layer-wise intervention effect mapping
  ├── P2-B: Steering × Probe layer combinatorics
  └── Output: reports_m5/IC4_M5_PROPAGATION_REPORT.md

Phase 5 (P3): Branch B — Factuality
  ├── P3-A: Factuality probe + vector transfer test
  ├── P3-B: Factuality-specific vector (conditional)
  └── Output: results_branch_b/factuality/

Phase 6 (P4): Cross-Model (aspirational)
  └── Output: results_m5_cross_model/
```

---

## 8. Immediate Recommendation

**Start with P0-A (Cross-Scenario on Cross-Seed).** It is the most direct extension of M4-Completion, requires no new infrastructure, and closes the last major validation gap. The script can reuse `run_m4_completion.py` with minor modifications (change scenario parameter).

After P0-A, the shuffled anomaly (P1-A) is the most scientifically interesting question — it may reveal something about the geometric structure of the effective intervention subspace that the current "mean-diff vector" interpretation doesn't fully capture.

---

## 9. What Not to Do (Reconfirmed)

- ❌ Do NOT go back to manual token-by-token generation
- ❌ Do NOT re-run open-loop alpha sweeps without gate
- ❌ Do NOT treat soft_T0.3 as a candidate
- ❌ Do NOT overclaim: robustness is cross-seed/cross-layer on standard scenario, not yet cross-scenario on cross-seed
- ❌ Do NOT flatten the mechanism to "just the vector" — the full mechanism is probe+vector+gate+hook+generate

---
*IC-4-M5: Post-Completion Experimental Roadmap*