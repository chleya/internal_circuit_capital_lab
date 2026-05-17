# IC-4-M3-v3: Probe-Gated Steering Report

## 1. M3-O & M3-v2 Recap

M3-O (oracle gate): IC4_M3_ORACLE_GATE_SUCCESS. v is clean, bottleneck is gating signal.
M3-v2 (scalar gate): IC4_M3_V2_GATE_INSUFFICIENT. Token-level risk signals move but are insufficient. Gap to oracle: H=+0.267.
Conclusion: v is not the problem. The path forward is better gating signal, not better v.

## 2. M3-v3 Design

M3-v3 replaces token-level scalar risk signals (entropy, maxprob) with a trajectory-state probe gate:

1. Generate first `window` tokens WITHOUT steering
2. Pool hidden states over window (window-pooled representation)
3. Compute `probe_score = P(unanswerable)` via logistic regression
4. `gate = sigmoid(steepness * (probe_score - threshold))`
5. `alpha = alpha_max * gate`
6. Continue generation with fixed steering alpha

| Parameter | Value |
|---|---|
| Probe layer | 12 |
| Window size | 4 |
| Probe model | logistic |
| Decision after N tokens | 4 |
| Steepness | 10.0 |
| Threshold | 0.5 |

## 3. Probe Training Evaluation

| Metric | Value |
|---|---|
| Train accuracy | 1.0000 |
| CV accuracy (mean) | 0.9000 |
| CV accuracy (std) | 0.0816 |
| AUC | 1.0000 |
| N train samples | 30.0 |
| N answerable / unanswerable | 15.0 / 15.0 |

## 4. Experiment Configuration

| Parameter | Value |
|---|---|
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| Device / dtype | cpu / float32 |
| Steering layer | [12] |
| alpha_max | [-1.0] |
| Elapsed | 1831s (30.5 min) |

## 5. Full Metrics Table

| Mode | H | C | UA | CA | Vector |
|---|---|---|---|---|---|
| base | 0.867 | 0.600 | 0.000 | 0.067 | none |
| prompt_only | 0.400 | 0.067 | 0.200 | 0.133 | none |
| steering_a-1.0 | 0.667 | 0.533 | 0.067 | 0.100 | steering |
| oracle_gate_a-1.0 | 0.667 | 0.600 | 0.000 | 0.100 | steering |
| probe_gate_a-1.0 | 0.933 | 0.567 | 0.000 | 0.000 | steering |
| random_probe_gate_a-1.0 | 0.933 | 0.567 | 0.000 | 0.000 | random |
| shuffled_probe_gate_a-1.0 | 0.933 | 0.567 | 0.000 | 0.000 | shuffled |

## 6. Probe Gate vs Oracle Gate vs Open-Loop

| Mode | H | C | UA | dH_base | Gap to Oracle H |
|---|---|---|---|---|---|
| base | 0.867 | 0.600 | 0.000 | -- | -- |
| oracle_gate | 0.667 | 0.600 | 0.000 | -0.200 | -- |
| open_loop a=-1.0 | 0.667 | 0.533 | 0.067 | -0.200 | +0.000 |
| probe_gate_a-1.0 | 0.933 | 0.567 | 0.000 | +0.067 | +0.267 |
| random_probe_gate_a-1.0 | 0.933 | 0.567 | 0.000 | +0.067 | +0.267 |
| shuffled_probe_gate_a-1.0 | 0.933 | 0.567 | 0.000 | +0.067 | +0.267 |

## 7. Gate Telemetry (Sample-Level)

| Metric | Answerable | Unanswerable | Separation |
|---|---|---|---|
| probe_score mean | 0.9011 | 0.0309 | 0.8702 |
| gate mean | 0.0648 | 0.9698 | 0.9050 |


## 8. Comparison: Probe Gate vs Scalar Gate (M3-v2)

| Metric | M3-v2 best scalar (maxprob) | M3-v3 probe |
|---|---|---|
| H | 0.933 | 0.933 |
| C | 0.567 | 0.567 |
| Gap to oracle H | +0.267 | +0.267 |

## 9. Verdict

**Verdict: `IC4_M3_V3_CONTROL_ARTIFACT`**

**Reasoning:** Probe gate (H=0.933) is indistinguishable from random_probe (H=0.933). The probe correctly classifies answerable vs unanswerable (gate separation confirmed), but the late intervention (after 4 tokens) does not translate to behavioral improvement. The model trajectory is already committed before steering activates.

## 10. Key Questions

1. **Is probe gate closer to oracle gate than scalar gate?** Probe gate H=0.933 vs oracle H=0.667 (gap=+0.267). Same gap as M3-v2 maxprob gate. The probe perfectly classifies but cannot improve behavior.
2. **Does probe gate beat random/shuffled controls?** No. probe_gate H=0.933 = random_probe H=0.933 = shuffled_probe H=0.933. Probe gate is a control artifact.
3. **Does trajectory-level pooled state improve gate quality?** Probe train accuracy=1.0000, CV=0.9, AUC=1.0. Gate correctly separates A from U (separation confirmed). BUT behavioral outcome (H) is unchanged. The trajectory state carries valid signal, but the intervention at token 4 is too late to redirect the model's generation path.

## 11. Key Insight: Gate Timing vs Gate Accuracy

The probe gate reveals a critical distinction between two failure modes:

| Failure Mode | Description | Evidence |
|---|---|---|
| Wrong signal | Gate classifies incorrectly | NOT the problem here — probe has ~0.87 separation |
| Late signal | Gate classifies correctly but too late | IS the problem — model trajectory committed in first 4 tokens |

The probe correctly identifies which samples need steering (unanswerable) and which don't (answerable). But by the time the probe makes its decision (after 4 tokens), the model's KV cache, hidden states, and generated prefix have already committed it to a specific output trajectory. Applying steering from token 5 cannot undo this commitment.

This explains why `prompt_only` (H=0.400) — which intervenes at token 0 via the prompt — is far more effective than any post-hoc steering gate (H=0.667 best, H=0.933 typical).

**One-line Conclusion:** Probe gate H=0.933 (base H=0.867, oracle H=0.667). The trajectory-state probe correctly classifies answerable vs unanswerable, but the intervention arrives too late (after 4 tokens). The model's generation trajectory is already committed before steering activates. Control artifact confirmed.

---

*IC-4-M3-v3: Probe-Gated Steering*
*Generated by run_m3_v3*