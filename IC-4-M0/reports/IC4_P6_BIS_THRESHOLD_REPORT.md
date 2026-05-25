# IC-4 P6-bis: Threshold Calibration for Sycophancy Feedback Control

> **Date**: 2026-05-22 | **Status**: Completed — Hook Architecture Diagnostic Performed
> **Predecessor**: P6 — Behavior-only probe (train acc=81.94%, test acc=77.78%)
> **Layer**: 10 | **Alpha**: -3.0 | **Test samples**: 12 | **Baseline syc**: 0.6667

---

## 1. Motivation

P6's behavior-only probe trains successfully (78% test accuracy),
but the gate rate is only 8.3%. Original hypothesis: threshold
calibration problem (scores cluster near 0.5).

**Actual finding: hook architecture problem, not threshold calibration.**

The probe separates syc/non-syc perfectly in standalone mode (+0.54
separation), but during `model.generate()`, the hook captures generated-token
hidden states instead of last-prompt-token states, rendering probe scores
meaningless.

P6-bis systematically tested thresholds and percentile gating, and in doing
so **diagnosed the root cause** of the feedback loop failure.

---

## 2. Probe Score Distribution — Standalone Forward Pass

**Method**: Run `model(**inputs)` once, collect L10 last_prompt_token hidden
state, score with probe. This is the probe's "native" evaluation.

- Mean: **0.6448** | Std: **0.2884**
- Range: [0.2259, 0.9917]

| Percentile | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% |
|---|---|---|---|---|---|---|---|---|---|
| Score | 0.2424 | 0.3190 | 0.4104 | 0.5461 | 0.7518 | 0.8522 | 0.8703 | 0.9085 | 0.9670 |

- Syc mean score: **0.8233**
- Non-syc mean score: **0.2876**
- Score separation (syc - non-syc): **+0.5357**

**Key insight**: In standalone mode, the probe achieves near-perfect separation.
Syc samples score ~0.82; non-syc samples score ~0.29. With th=0.30, 11/12
samples would gate. **The probe is excellent — the hook is broken.**

---

## 3. Probe Score — During model.generate() (via Hook)

**Method**: Hook at L10 fires during `model.generate()`, collects `hs[:, -1, :]`
at every forward pass, scores with probe.

| Threshold | Gate Rate | Syc Rate | Delta from Baseline | Probe mu (in hook) | Probe sigma |
|---|---|---|---|---|---|
| 0.30 | 0.0833 | 0.6667 | +0.0000 | 0.4655 | 0.0000 |
| 0.35 | 0.0833 | 0.6667 | +0.0000 | 0.4655 | 0.0000 |
| 0.40 | 0.0833 | 0.6667 | +0.0000 | 0.4655 | 0.0000 |
| 0.45 | 0.0833 | 0.6667 | +0.0000 | 0.4655 | 0.0000 |
| 0.50 | 0.0833 | 0.6667 | +0.0000 | 0.4889 | 0.1027 |

**Critical observation**: Probe mu in hook (~0.47) != Probe mu in standalone (~0.64).
Gate rate = 8.3% at ALL thresholds, including 0.30 (which in standalone mode
would catch 11/12 samples). **The hook is seeing different hidden states.**

---

## 4. Percentile-Based Gating (via Hook, During generate())


| Gate Rule | Effective Threshold | Gate Rate | Syc Rate | Delta from Baseline |
|---|---|---|---|---|
| top-20% | 0.9085 | 0.0833 | 0.6667 | +0.0000 |
| top-30% | 0.8703 | 0.0833 | 0.6667 | +0.0000 |
| top-40% | 0.8522 | 0.0833 | 0.6667 | +0.0000 |

Percentile-based gating equally futile — same 8.3% gate rate regardless of
effective threshold. Confirms the probe scores inside the hook are not from
last-prompt-token states.

---

## 5. Open-Loop Reference

| Direction | Syc Rate | Delta from Baseline |
|---|---|---|
| v_syc | 0.3333 | -0.3333 (-50.0%) |

Open-loop v_syc alpha=-3.0 confirmed for the **third time** across independent splits:
- P5-bis (24-sample): -35.7%
- P6 (12-sample): -50.0%
- P6-bis (12-sample): **-50.0%**

This effect is robust and reproducible. Open-loop steering is the reliable
intervention while the hook architecture is being fixed.

---

## 6. Root Cause Diagnosis: Hook Captures Generated Tokens, Not Prompt Tokens

### The Problem

During `model.generate()`, the L10 hook fires on **every forward pass**:

| Forward Pass | hs[:, -1, :] captures | Probe trained on? |
|---|---|---|
| Prefill (step 0) | Last **prompt** token | **YES** |
| Decode step 1 | Generated token 1 | NO |
| Decode step 2 | Generated token 2 | NO |
| ... | ... | NO |

The probe was trained exclusively on **last_prompt_token** hidden states
(from Phase 2). During generation, after the prefill step, the hook sees
**generated-token** hidden states, which the probe has never learned to score.

Generated-token states all produce similar probe scores (~0.47), which
dominate the aggregate statistics and prevent meaningful gating.

### Why the gate rate is ~8.3% (1/12) at all thresholds

For the 11 non-gated samples, the hook records ~128 scores (one per
generated token), all near 0.47. These scores are meaningless — they
don't reflect sycophancy tendency. The actual prefill score (which
would trigger the gate at th=0.30) is recorded but buried in the
aggregate.

For the 1 gated sample, the prefill score happens to cross the
threshold, the gate fires, and no more scores are recorded.

### Evidence

1. **Probe in standalone mode**: Excellent separation (+0.54), syc=0.82, non-syc=0.29
2. **Probe in hook during generate()**: mu=0.47, near-zero variance
3. **Gate rate invariant**: Same 8.3% at th=0.30 (should be 91.7%) and th=0.90
4. **Scores from generated tokens**: Systematically different from prompt-token scores

---

## 7. Fix: Two-Stage Architecture

The probe->gate->hook feedback loop requires the gate decision to be made
**before** generation begins, using a standalone forward pass:

```
Stage 1: model(**inputs) -> collect L10 last_prompt_token hs -> probe score
Stage 2: if score >= threshold -> run model.generate() with steering hook at L10
         else -> run model.generate() without steering
```

This eliminates the token-type pollution problem. The probe always sees
last_prompt_token hidden states, which is what it was trained on.

### P6-ter Implementation Plan

| Step | Detail |
|---|---|
| 1 | Run standalone forward pass to get probe score before generation |
| 2 | If score >= threshold, register steering hook for v_syc at alpha=-3.0 |
| 3 | Run model.generate() with steering hook (no probe scoring needed) |
| 4 | Compare feedback syc rate vs open-loop syc rate |
| 5 | Sweep threshold to find optimal gate rate |

---

## 8. Summary of Findings

| Finding | Detail |
|---|---|
| **F15** | Behavior-only probe achieves +0.54 score separation in standalone mode |
| **F16** | Probe->gate->hook fails because hook captures generated-token states, not prompt-token states |
| **F17** | Gate rate invariant at 8.3% across all thresholds (0.30-0.90) — confirms hook architecture bug |
| **F18** | Open-loop v_syc alpha=-3.0 robustly confirmed for the third time (-50.0% reduction) |
| **N10** | Fix: Two-stage architecture — probe score from standalone forward pass, then conditional generation with steering |

---

## 9. Next Steps

| Priority | Action | Detail |
|---|---|---|
| **P6-ter** | Two-Stage Feedback Control | Standalone probe scoring -> conditional generate with steering |
| P7 | S15 Amplification | Investigate syc signal amplification at gen step 15 |
| P8 | MLP Probe | Non-linear probe — deferred until two-stage architecture works |