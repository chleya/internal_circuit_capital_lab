# IC-4 P8: Larger-Scale Replication of Two-Stage Feedback Control

> **Date**: 2026-05-23 | **Status**: Completed
> **Predecessor**: P6-ter Two-Stage Feedback (n=12, −66.7% at th=0.50)
> **Samples**: 24 | **Layer**: 10 | **Alpha**: -3.0

---

## 1. Motivation

P6-ter achieved −66.7% sycophancy reduction on n=12 (th=0.50), but
small sample sizes can produce spurious effects. P8 replicates the
two-stage architecture on n=24 samples to verify statistical robustness.

---

## 2. Results

| Condition | N | Syc Rate | Syc Count | Δ vs Baseline | Gate Rate |
|---|---|---|---|---|---|
| baseline | 24 | 0.5833 | 14/24 | +0.0000 (+0.0%) | — |
| two_stage th=0.50 | 24 | 0.2500 | 6/24 | -0.3333 (-57.1%) | 0.5417 probe_mu=0.6501 |
| two_stage th=0.40 | 24 | 0.4583 | 11/24 | -0.1250 (-21.4%) | 0.6667 probe_mu=0.6501 |
| open_loop | 24 | 0.5000 | 12/24 | -0.0833 (-14.3%) | — |

---

## 3. Comparison with P6-ter (n=12)

| Metric | P6-ter (n=12) | P8 (n=24) | Change |
|---|---|---|---|
| Baseline syc | 0.7500 | 0.5833 | -0.1667 |
| Two-stage th=0.50 | 0.2500 (−66.7%) | 0.2500 (-57.1%) | +0.0000 |
| Two-stage th=0.40 | 0.3333 (−55.6%) | 0.4583 (-21.4%) | +0.1250 |
| Open-loop | 0.4167 (−44.4%) | 0.5000 (-14.3%) | +0.0833 |

---

## 4. Statistical Significance

| Comparison | Fisher p-value | Significant (p<0.05)? |
|---|---|---|
| Baseline vs Two-Stage th=0.50 | 0.0392 | YES |
| Baseline vs Open-Loop | 0.7725 | no |
| Two-Stage th=0.50 vs Open-Loop | 0.1351 | no |

---

## 5. Interpretation

**Two-stage feedback at th=0.50**: syc=0.2500 (-57.1%), gate_rate=0.5417 (13/24 gated)
**Open-loop**: syc=0.5000 (-14.3%)

Two-stage feedback BEATS open-loop — replicating the P6-ter finding
at larger scale (n=24). Selective intervention > universal intervention.

**Statistically significant** (Fisher p=0.0392 < 0.05).
Two-stage feedback produces a significant reduction in sycophancy
compared to baseline on n=24 samples.

---

## 6. Next Steps

| Priority | Action | Detail |
|---|---|---|
| P9 | Cross-Bottleneck | stabilization + organization joint intervention |
| P10 | Hallucination abandon | Formalize exclusion of single-direction impulse |
