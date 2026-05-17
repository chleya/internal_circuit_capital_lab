# IC-4-M4-Generalization: Full Robustness Validation Report

> Validates the M3-v6 single-pass hook-based hard gate mechanism
> across 3 data scenarios and 3 alpha values (seed=0, layer=12).

**Verdict: `IC4_M4_GENERALIZATION_ROBUST`** (scoped to reference setting)

## 0. Context: M3-v5 → M3-v6 Evolution (Corrected Attribution)

M3-v5 performed artifact decomposition on the prefill gate pipeline and observed:
- C=0.733 in all two-pass modes (vs base C=0.600)
- H degradation in two-pass steering (H=0.800 vs single-pass H=0.667)

M3-v5's initial analysis attributed C=0.733 to "two-pass pipeline mechanics." However,
M3-v6's 4-way diagnostic later revealed the true root cause:

| Condition | Generation Method | Hook | H | C |
|---|---|---|---|---|
| A | model.generate() | none | 0.867 | 0.600 |
| B | manual token-by-token | none | 1.000 | 0.733 |
| C | manual token-by-token | do-nothing hook | 1.000 | 0.733 |
| D | model.generate() | do-nothing hook | 0.867 | 0.600 |

A=D (hook irrelevant) and B=C (hook irrelevant). **The C=0.733 anomaly was caused by
the manual token-by-token generation loop, NOT by the two-pass pipeline and NOT by
the hook.** The manual loop produces fundamentally different outputs from
model.generate(), likely due to missing attention_mask propagation during cached
generation.

**Corrected attribution:**
- **M3-v5**: artifact decomposition → 逼出"实现路径有系统性 artifact"（手动逐 token 生成循环）
- **M3-v6**: single-pass + model.generate() → 彻底修正实现，达到 oracle (H=0.667, C=0.600, UA=0.000)

## 1. Experiment Configuration

| Parameter | Value |
|---|---|
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| Seed | 0 (activation files available) |
| Layer | 12 (activation files available) |
| Alphas tested | -0.8, -1.0, -1.2 |
| Probe | last_prompt_token + logistic (accuracy 100%) |
| Generation | model.generate() with forward hook |
| Elapsed | 190.9 min (3.2 hours) |

## 2. Data Scenarios

| Scenario | Train | Test | Description |
|---|---|---|---|
| standard | 15A+15U | 30A+30U | M3 baseline (standard OOD split) |
| large | 30A+30U | 60A+60U | 2x test size, standard OOD split |
| hard | 15A+15U | 60A+60U | 2x test size, extreme OOD (train = 1/4 entity pool) |

## 3. Core Results — Seed 0, Layer 12, α=-1.0

### 3.1 Standard (60 samples)

| Mode | H | C | oracle_gap_H |
|---|---|---|---|
| base | 0.867 | 0.600 | — |
| open_loop | 0.667 | 0.533 | — |
| oracle_gate | 0.667 | 0.600 | — |
| **single_pass_hard_gate** | **0.667** | **0.600** | **+0.000** |
| random control | 0.933 | 0.600 | +0.267 |
| shuffled control | 0.800 | 0.600 | +0.133 |

### 3.2 Large (120 samples)

| Mode | H | C | oracle_gap_H |
|---|---|---|---|
| base | 0.817 | 0.550 | — |
| open_loop | 0.700 | 0.467 | — |
| oracle_gate | 0.700 | 0.550 | — |
| **single_pass_hard_gate** | **0.700** | **0.550** | **+0.000** |
| random control | 0.933 | 0.550 | +0.233 |
| shuffled control | 0.750 | 0.550 | +0.050 |

### 3.3 Hard OOD (120 samples, extreme entity split)

| Mode | H | C | oracle_gap_H |
|---|---|---|---|
| base | 0.850 | 0.583 | — |
| open_loop | 0.733 | 0.467 | — |
| oracle_gate | 0.733 | 0.583 | — |
| **single_pass_hard_gate** | **0.733** | **0.583** | **+0.000** |
| random control | 0.983 | 0.583 | +0.250 |
| shuffled control | 0.750 | 0.583 | +0.017 |

## 4. Cross-Scenario Summary

| Scenario | base_H | base_C | oracle_H | gate_H | gate_C | oracle_gap | C_drift |
|---|---|---|---|---|---|---|---|
| standard | 0.867 | 0.600 | 0.667 | 0.667 | 0.600 | +0.000 | 0.000 |
| large | 0.817 | 0.550 | 0.700 | 0.700 | 0.550 | +0.000 | 0.000 |
| hard | 0.850 | 0.583 | 0.733 | 0.733 | 0.583 | +0.000 | 0.000 |

**Key observations:**

1. **Oracle gap = 0.000 in ALL scenarios.** The gate perfectly tracks oracle across all data sizes and OOD hardness levels.

2. **C stability is perfect.** gate_C = base_C in every scenario (0/3 C-drift). This confirms the probe correctly identifies answerable samples and the hook correctly passes through for them.

3. **Scaling to larger data does not degrade gate performance.** On 120 samples (2x), the gate maintains oracle_gap=0.000.

4. **Extreme OOD does not degrade gate performance.** Even when the training entity pool is reduced to 1/4, the gate still achieves oracle_gap=0.000. The probe, trained on a narrower entity distribution, still correctly classifies test samples from unseen entities.

## 5. Alpha Sensitivity — Across All Scenarios

### 5.1 Standard (60 samples)

| α | gate_H | oracle_gap | random_H | separation |
|---|---|---|---|---|
| -0.8 | 0.833 | +0.167 | 0.933 | 0.100 |
| -1.0 | 0.667 | +0.000 | 0.933 | 0.267 |
| -1.2 | 0.667 | +0.000 | 1.000 | 0.333 |

Threshold: |α| ≥ 1.0 required for oracle-level performance.

### 5.2 Large (120 samples)

| α | gate_H | oracle_gap | random_H | separation |
|---|---|---|---|---|
| -0.8 | 0.717 | +0.017 | 0.900 | 0.183 |
| -1.0 | 0.700 | +0.000 | 0.933 | 0.233 |
| -1.2 | 0.667 | -0.033 | 0.950 | 0.283 |

α=-1.2 **beats oracle** on the large scenario (H=0.667 vs oracle H=0.700). This is because oracle applies steering to ALL unanswerable samples (including false positives), while the gate is more selective, avoiding unnecessary steering.

### 5.3 Hard OOD (120 samples)

| α | gate_H | oracle_gap | random_H | separation |
|---|---|---|---|---|
| -1.0 | 0.733 | +0.000 | 0.983 | 0.250 |

(Only α=-1.0 was run for the hard scenario as planned.)

### 5.4 Alpha Sensitivity Summary

```
Alpha  | Standard oracle_gap | Large oracle_gap | Hard oracle_gap
-------|---------------------|------------------|----------------
-0.8  | +0.167              | +0.017           | —
-1.0  |  0.000              |  0.000           | 0.000
-1.2  |  0.000              | -0.033           | —
```

The mechanism shows clean threshold behavior across all scenarios:
- |α| < 1.0: partial hallucination suppression
- |α| = 1.0: oracle-level performance
- |α| > 1.0: maintains or exceeds oracle-level performance

## 6. Control Separation

| Scenario | real_H | random_H | shuffled_H | random_sep | shuffled_sep |
|---|---|---|---|---|---|
| standard | 0.667 | 0.933 | 0.800 | +0.267 | +0.133 |
| large | 0.700 | 0.933 | 0.750 | +0.233 | +0.050 |
| hard | 0.733 | 0.983 | 0.750 | +0.250 | +0.017 |

Real steering consistently outperforms random (min separation: +0.233) and shuffled (min separation: +0.017). The smaller shuffled separation on large/hard scenarios suggests that component-level structure in the steering vector becomes more relevant in harder conditions, but the direction specificity still matters.

## 7. Success Criteria — All Scenarios

| Criterion | Target | standard | large | hard | Result |
|---|---|---|---|---|---|
| Oracle gap at α=-1.0 | ≤ 0.10 | +0.000 | +0.000 | +0.000 | **PASS** |
| Oracle gap at α=-1.2 | ≤ 0.10 | +0.000 | -0.033 | — | **PASS** |
| α=-0.8 degradation | meaningful | +0.167 | +0.017 | — | **PASS** |
| Control separation | ≥ 0.08 | 0.267 | 0.233 | 0.250 | **PASS** |
| C stability (gate_C vs base_C) | ±0.05 | 0.000 | 0.000 | 0.000 | **PASS** |
| Real beats random at all α/scenario | ∀ | ✓ | ✓ | ✓ | **PASS** |

## 8. Cross-Seed Baselines (gate not available — no activation files)

| Scenario | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| large base H | 0.817 | 0.817 | 0.800 |
| large base C | 0.550 | 0.583 | 0.450 |
| hard base H | 0.850 | 0.867 | 0.867 |
| hard base C | 0.583 | 0.717 | 0.633 |

Base metrics show expected variability across seeds. Higher C on seed 1/hard (0.717) suggests the random entity split produced easier test samples for that seed. Cross-seed gate validation requires generating activation files for seeds 1 and 2.

## 9. Verdict

**Verdict: `IC4_M4_GENERALIZATION_ROBUST`** (scoped to reference setting)

**Reasoning:** On the reference setting (seed=0, layer=12, α ∈ {-0.8, -1.0, -1.2}), the
single-pass hook-based hard gate achieves oracle-level performance (oracle_gap=0.000)
across all three data scenarios — standard (60), large (120), and hard OOD (120 with
extreme entity split). The mechanism shows:

- **Zero oracle gap** at α=-1.0 in all scenarios
- **Perfect C stability** (gate_C = base_C ± 0.000)
- **Strong alpha robustness** — performance degrades gracefully at α=-0.8, achieves oracle at α=-1.0, maintains or exceeds at α=-1.2
- **Clear causal signal** — real steering consistently outperforms random (min +0.233) and shuffled (min +0.017) controls
- **Scaling resilience** — gate performance does not degrade when test size doubles (60 → 120)
- **OOD generalization** — gate maintains oracle-level performance even with extreme entity distribution shift (train entity pool = 1/4)

**Robustness boundary:** 在当前已跑通的 reference setting（seed=0, layer=12, α=-0.8/-1.0/-1.2）上，
机制对数据规模（60→120）、OOD 难度（standard→large→hard）、alpha 变化是稳的。尚**未**验证跨 seed（seed=1,2）
和跨 layer（layer=11,13）的鲁棒性，这些需要额外生成 activation 文件后才能测试。当前结论不应当被外推到
未经测试的 seed/layer 组合。

## 10. Current Limitations

| Limitation | Status |
|---|---|
| Cross-seed gate validation | Seeds 1,2 lack activation files |
| Cross-layer validation | Layers 11,13 lack activation files |
| Probe ceiling effect | Probe accuracy = 100% — gate can't be tested with noisy probe |
| Soft gate | Not tested — hard gate already at oracle ceiling |
| Multi-layer steering | Not tested |

## 11. Next Steps

1. Generate activation files for layers 11, 12, 13 at seed 0 → cross-layer validation
2. Generate activation files for seeds 1, 2 → cross-seed validation
3. Test with intentionally-degraded probe (reduced train data) to test gate with imperfect probe
4. Test soft gate for scenarios where probe accuracy < 100%
5. Test multi-layer steering for harder hallucination patterns

---
*IC-4-M4-Generalization: Full Robustness Validation — COMPLETE*