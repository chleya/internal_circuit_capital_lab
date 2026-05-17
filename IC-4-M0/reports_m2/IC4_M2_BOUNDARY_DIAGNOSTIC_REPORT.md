# IC-4-M2: Boundary Diagnostic & Constrained Steering Selection

## 1. M1 Recap

IC-4-M1 smoke ran 1 seed (0), 3 layers (9/12/15), 5 alphas on 60-train/120-test data.

| Metric | Value |
|---|---|
| Best Layer | 12 |
| Best Alpha | -1.25 |
| Base H | 0.900 |
| Best Steering H | 0.483 |
| Best Steering C | 0.733 |
| Best Steering UA | 0.183 |
| Control Gap vs Random | +0.367 |
| Control Gap vs Shuffled | +0.184 |

**M1 Verdict: IC4_M1_MODEL_DAMAGE** -- Real steering reduced hallucination by 46.3% (H: 0.900 -> 0.483),
clearly better than random/shuffled controls, but C dropped to 0.733 (<0.82) and UA rose to 0.183 (>0.09).
The steering vector appears to encode a caution/refusal/uncertainty direction rather than a clean
anti-hallucination circuit.

## 2. M2 Configuration

| Parameter | Value |
|---|---|
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| Train size | 60 |
| Test size | 120 |
| Seeds | [0] |
| Layers | [11, 12, 13] |
| Alphas | [-1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, 0.0] |
| Temperature | 0.0 |
| do_sample | False |
| max_new_tokens | 48 |
| Elapsed time | 31134s (518.9 min) |

## 3. Base Metrics Per Seed

| Seed | H | C | UA | CA |
|---|---|---|---|---|
| 0 | 0.9500 | 0.5667 | 0.0167 | 0.0000 |

## 4. Best Unconstrained Real Steering Points

| Seed | Layer | Alpha | H | C | UA | CA |
|---|---|---|---|---|---|---|
| 0 | 11 | -1.1000 | 0.5333 | 0.4833 | 0.1500 | 0.1500 |
| 0 | 11 | -1.0000 | 0.5667 | 0.5167 | 0.1333 | 0.1500 |
| 0 | 11 | -0.9000 | 0.5667 | 0.4667 | 0.0833 | 0.1500 |
| 0 | 13 | -1.1000 | 0.5667 | 0.4667 | 0.0667 | 0.1000 |
| 0 | 13 | -0.9000 | 0.5667 | 0.5333 | 0.0500 | 0.0833 |
| 0 | 13 | -1.0000 | 0.5833 | 0.4833 | 0.0500 | 0.0667 |
| 0 | 12 | -1.1000 | 0.5833 | 0.5000 | 0.0833 | 0.1167 |
| 0 | 13 | -0.8000 | 0.6000 | 0.5333 | 0.0500 | 0.0833 |
| 0 | 11 | -0.8000 | 0.6167 | 0.5000 | 0.0500 | 0.1500 |
| 0 | 13 | -0.7000 | 0.6333 | 0.5333 | 0.0333 | 0.0667 |

## 5. Constrained Candidates

Constraints: C >= base_C - 0.03, UA <= base_UA + 0.05, C >= 0.78, UA <= 0.08

**No constrained candidates found.** All steering points violate at least one constraint.

## 6. Candidates Rejected by Constraint

| Seed | Layer | Alpha | H | C | UA | Rejection Reasons |
|---|---|---|---|---|---|---|
| 0 | 11 | -1.1000 | 0.5333 | 0.4833 | 0.1500 | C=0.483 < 0.537; UA=0.150 > 0.067; C=0.483 < 0.78; UA=0.150 > 0.08 |
| 0 | 11 | -1.0000 | 0.5667 | 0.5167 | 0.1333 | C=0.517 < 0.537; UA=0.133 > 0.067; C=0.517 < 0.78; UA=0.133 > 0.08 |
| 0 | 11 | -0.9000 | 0.5667 | 0.4667 | 0.0833 | C=0.467 < 0.537; UA=0.083 > 0.067; C=0.467 < 0.78; UA=0.083 > 0.08 |
| 0 | 13 | -1.1000 | 0.5667 | 0.4667 | 0.0667 | C=0.467 < 0.537; C=0.467 < 0.78 |
| 0 | 13 | -0.9000 | 0.5667 | 0.5333 | 0.0500 | C=0.533 < 0.537; C=0.533 < 0.78 |
| 0 | 12 | -1.1000 | 0.5833 | 0.5000 | 0.0833 | C=0.500 < 0.537; UA=0.083 > 0.067; C=0.500 < 0.78; UA=0.083 > 0.08 |
| 0 | 13 | -1.0000 | 0.5833 | 0.4833 | 0.0500 | C=0.483 < 0.537; C=0.483 < 0.78 |
| 0 | 13 | -0.8000 | 0.6000 | 0.5333 | 0.0500 | C=0.533 < 0.537; C=0.533 < 0.78 |
| 0 | 11 | -0.8000 | 0.6167 | 0.5000 | 0.0500 | C=0.500 < 0.537; C=0.500 < 0.78 |
| 0 | 13 | -0.7000 | 0.6333 | 0.5333 | 0.0333 | C=0.533 < 0.537; C=0.533 < 0.78 |
| 0 | 12 | -1.0000 | 0.6500 | 0.4667 | 0.0667 | C=0.467 < 0.537; C=0.467 < 0.78 |
| 0 | 11 | -0.7000 | 0.6833 | 0.4667 | 0.0333 | C=0.467 < 0.537; C=0.467 < 0.78 |
| 0 | 13 | -0.6000 | 0.6833 | 0.5333 | 0.0333 | C=0.533 < 0.537; C=0.533 < 0.78 |
| 0 | 12 | -0.9000 | 0.7000 | 0.4667 | 0.0500 | C=0.467 < 0.537; C=0.467 < 0.78 |
| 0 | 12 | -0.8000 | 0.7000 | 0.4833 | 0.0333 | C=0.483 < 0.537; C=0.483 < 0.78 |
| 0 | 11 | -0.5000 | 0.7167 | 0.4833 | 0.0167 | C=0.483 < 0.537; C=0.483 < 0.78 |
| 0 | 11 | -0.6000 | 0.7333 | 0.4500 | 0.0333 | C=0.450 < 0.537; C=0.450 < 0.78 |
| 0 | 12 | -0.7000 | 0.7500 | 0.4833 | 0.0333 | C=0.483 < 0.537; C=0.483 < 0.78 |
| 0 | 12 | -0.5000 | 0.7500 | 0.5500 | 0.0167 | C=0.550 < 0.78 |
| 0 | 13 | -0.5000 | 0.7500 | 0.5667 | 0.0167 | C=0.567 < 0.78 |

## 7. Matched-Alpha Control Comparison

| Seed | Layer | Alpha | Real H | Random H | Random H OK | Shuffled H | Shuffled H OK |
|---|---|---|---|---|---|---|---|
| 0 | 11 | -1.1000 | 0.5333 | 0.7833 | PASS | 0.6667 | PASS |
| 0 | 11 | -1.0000 | 0.5667 | 0.7833 | PASS | 0.7000 | PASS |
| 0 | 11 | -0.9000 | 0.5667 | 0.8333 | PASS | 0.7167 | PASS |
| 0 | 11 | -0.8000 | 0.6167 | 0.8667 | PASS | 0.7333 | PASS |
| 0 | 11 | -0.7000 | 0.6833 | 0.9000 | PASS | 0.7667 | FAIL |
| 0 | 11 | -0.6000 | 0.7333 | 0.9000 | PASS | 0.7833 | FAIL |
| 0 | 11 | -0.5000 | 0.7167 | 0.9333 | PASS | 0.8167 | PASS |
| 0 | 12 | -1.1000 | 0.5833 | 0.9500 | PASS | 0.7667 | PASS |
| 0 | 12 | -1.0000 | 0.6500 | 0.9500 | PASS | 0.7833 | PASS |
| 0 | 12 | -0.9000 | 0.7000 | 0.9500 | PASS | 0.7833 | FAIL |
| 0 | 12 | -0.8000 | 0.7000 | 0.9167 | PASS | 0.8167 | PASS |
| 0 | 12 | -0.7000 | 0.7500 | 0.9000 | PASS | 0.8000 | FAIL |
| 0 | 12 | -0.6000 | 0.8000 | 0.8667 | FAIL | 0.8500 | FAIL |
| 0 | 12 | -0.5000 | 0.7500 | 0.8833 | PASS | 0.8833 | PASS |
| 0 | 13 | -1.1000 | 0.5667 | 0.9500 | PASS | 0.6667 | PASS |
| 0 | 13 | -1.0000 | 0.5833 | 0.9500 | PASS | 0.7000 | PASS |
| 0 | 13 | -0.9000 | 0.5667 | 0.9333 | PASS | 0.7500 | PASS |
| 0 | 13 | -0.8000 | 0.6000 | 0.9500 | PASS | 0.8000 | PASS |
| 0 | 13 | -0.7000 | 0.6333 | 0.9500 | PASS | 0.7667 | PASS |
| 0 | 13 | -0.6000 | 0.6833 | 0.9500 | PASS | 0.7833 | PASS |
| 0 | 13 | -0.5000 | 0.7500 | 0.9167 | PASS | 0.8333 | FAIL |

## 8. Best-Control Comparison

| Seed | Layer | Real H (alpha) | Best Random H | Random OK | Best Shuffled H | Shuffled OK |
|---|---|---|---|---|---|---|
| 0 | 11 | 0.5333 (a=-1.10) | 0.7833 | PASS | 0.6667 | PASS |
| 0 | 12 | 0.5833 (a=-1.10) | 0.8667 | PASS | 0.7667 | PASS |
| 0 | 13 | 0.5667 (a=-1.10) | 0.9167 | PASS | 0.6667 | PASS |

## 9. Per-Layer M2 Summary

| Layer | Best Uncon H | Best Uncon C | Best Uncon UA | Best Uncon Alpha | Best Random H | Best Shuffled H | Constrained Count |
|---|---|---|---|---|---|---|---|
| 11 | 0.5333 | 0.4833 | 0.1500 | -1.1000 | 0.7833 | 0.6667 | 0 |
| 12 | 0.5833 | 0.5000 | 0.0833 | -1.1000 | 0.8667 | 0.7667 | 0 |
| 13 | 0.5667 | 0.4667 | 0.0667 | -1.1000 | 0.9167 | 0.6667 | 0 |

## 10. Seed Stability

- Single-seed run; seed stability not assessed.

## 11. Verdict

**Verdict: `IC4_M2_MODEL_DAMAGE`**

**Reasoning:** Hallucination reduced substantially (best H=0.533 vs base H=0.950) but ALL steering points fail C>=0.78 or UA<=0.08 constraints. This suggests the steering direction is more caution/refusal than clean anti-hallucination.

### Verdict Interpretation

- Hallucination drops significantly but ALL valid points have C damage or UA excess.
- The steering direction likely encodes caution/refusal, not clean anti-hallucination.
- **Recommendation**: Consider alternative vector computation (per-sample normalization, CAA, or probing-based selection).

## 12. Next Recommendation

- **Smoke run only (1 seed).**
  - If any constrained candidate found: run full sweep with 3 seeds to validate stability.
  - If MODEL_DAMAGE or NULL: full sweep unlikely to change verdict but may reveal edge cases.

---

*IC-4-M2: Boundary Diagnostic & Constrained Steering Selection*
*Generated by report_writer*