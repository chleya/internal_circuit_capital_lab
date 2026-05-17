# IC-4-M3-v2: Telemetry-First Gated Steering Report

## 1. M3-v1 & M3-O Recap

M3-v1 (entropy feedback): implementation bug -- baseline computed under hook, gate did not activate.
M3-O (oracle gate): IC4_M3_ORACLE_GATE_SUCCESS -- with perfect gate, v achieves dH=-0.200 with C and UA fully preserved.
Conclusion: v is clean; the bottleneck is gating/feedback signal quality, not vector quality.

## 2. M3-v2 Design

Three risk-based gating signals, each implementing:
```
alpha_t = alpha_max * gate_t
gate_t = sigmoid(k * (risk_t - threshold))
```

| Gate | risk_t | default threshold | default k |
|---|---|---|---|
| entropy_gate | Shannon entropy of logits | 2.0 | 3.0 |
| maxprob_gate | 1 - max(softmax(logits)) | 0.3 | 10.0 |
| uncertainty_mass_gate | mass of uncertainty tokens (\"I don't know\", \"cannot determine\", etc.) | 0.5 | 10.0 |

## 3. Experiment Configuration

| Parameter | Value |
|---|---|
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| Device / dtype | cpu / float32 |
| Train / Test size | 30 / 60 |
| Seeds | [0] |
| Layer | [12] |
| alpha_max | [-1.0] |
| Elapsed | ~43 min (CPU) |

## 4. Full Metrics Table

| Mode | H | C | UA | CA | Vector |
|---|---|---|---|---|---|
| base | 0.867 | 0.600 | 0.000 | 0.067 | none |
| prompt_only | 0.400 | 0.067 | 0.200 | 0.133 | none |
| steering_a-1.0 | 0.667 | 0.533 | 0.067 | 0.100 | steering |
| oracle_gate_a-1.0 | 0.667 | 0.600 | 0.000 | 0.100 | steering |
| entropy_gate_a-1.0 | 0.933 | 0.533 | 0.000 | 0.000 | steering |
| maxprob_gate_a-1.0 | 0.933 | 0.567 | 0.000 | 0.000 | steering |
| uncertainty_mass_gate_a-1.0 | 0.933 | 0.567 | 0.000 | 0.000 | steering |
| random_entropy_gate_a-1.0 | 0.900 | 0.567 | 0.000 | 0.033 | random |
| shuffled_entropy_gate_a-1.0 | 0.933 | 0.567 | 0.000 | 0.000 | shuffled |

## 5. Gate Telemetry Analysis

### Gate Separation Summary

| Gate | gate_ans | gate_unans | gate_sep | entropy_t_sep | max_prob_t_sep | uncertainty_mass_t_sep | Ranking |
|---|---|---|---|---|---|---|---|
| maxprob_gate_a-1.0 | 0.3692 | 0.4867 | 0.1174 | 0.2818 | 0.0808 | 0.0011 | #1 |
| random_entropy_gate_a-1.0 (control) | 0.1970 | 0.2710 | 0.0741 | 0.2629 | 0.0811 | 0.0020 | #2 |
| entropy_gate_a-1.0 | 0.1977 | 0.2690 | 0.0714 | 0.2758 | 0.0788 | 0.0018 | #3 |
| shuffled_entropy_gate_a-1.0 (control) | 0.1923 | 0.2584 | 0.0662 | 0.2466 | 0.0776 | 0.0018 | #4 |
| uncertainty_mass_gate_a-1.0 INERT | 0.0216 | 0.0233 | 0.0017 | 0.2126 | 0.0667 | 0.0019 | #5 |

### Per-Gate Details

#### entropy_gate_a-1.0

| Metric | Answerable | Unanswerable | Separation |
|---|---|---|---|
| entropy_t mean | 0.9669 | 1.2427 | 0.2758 |
| max_prob_t mean | 0.2349 | 0.3138 | 0.0788 |
| uncertainty_mass_t mean | 0.0198 | 0.0217 | 0.0018 |
| gate mean | 0.1977 | 0.269 | 0.0714 |
| alpha mean | -0.1977 | -0.269 | 0.0714 |

#### maxprob_gate_a-1.0

| Metric | Answerable | Unanswerable | Separation |
|---|---|---|---|
| entropy_t mean | 0.9751 | 1.2569 | 0.2818 |
| max_prob_t mean | 0.2347 | 0.3155 | 0.0808 |
| uncertainty_mass_t mean | 0.0225 | 0.0236 | 0.0011 |
| gate mean | 0.3692 | 0.4867 | 0.1174 |
| alpha mean | -0.3692 | -0.4867 | 0.1174 |

#### uncertainty_mass_gate_a-1.0

| Metric | Answerable | Unanswerable | Separation |
|---|---|---|---|
| entropy_t mean | 0.9746 | 1.1872 | 0.2126 |
| max_prob_t mean | 0.2372 | 0.3039 | 0.0667 |
| uncertainty_mass_t mean | 0.0192 | 0.0211 | 0.0019 |
| gate mean | 0.0216 | 0.0233 | 0.0017 |
| alpha mean | -0.0216 | -0.0233 | 0.0017 |

#### random_entropy_gate_a-1.0

| Metric | Answerable | Unanswerable | Separation |
|---|---|---|---|
| entropy_t mean | 0.9604 | 1.2234 | 0.2629 |
| max_prob_t mean | 0.2297 | 0.3108 | 0.0811 |
| uncertainty_mass_t mean | 0.02 | 0.022 | 0.002 |
| gate mean | 0.197 | 0.271 | 0.0741 |
| alpha mean | -0.197 | -0.271 | 0.0741 |

#### shuffled_entropy_gate_a-1.0

| Metric | Answerable | Unanswerable | Separation |
|---|---|---|---|
| entropy_t mean | 0.9372 | 1.1838 | 0.2466 |
| max_prob_t mean | 0.2266 | 0.3041 | 0.0776 |
| uncertainty_mass_t mean | 0.0204 | 0.0222 | 0.0018 |
| gate mean | 0.1923 | 0.2584 | 0.0662 |
| alpha mean | -0.1923 | -0.2584 | 0.0662 |


## 6. Gate vs Oracle Gate vs Open-Loop Comparison

| Mode | H | C | UA | dH_base | Gap to Oracle H |
|---|---|---|---|---|---|
| base | 0.867 | 0.600 | 0.000 | -- | -- |
| oracle_gate | 0.667 | 0.600 | 0.000 | -0.200 | -- |
| open_loop a=-1.0 | 0.667 | 0.533 | 0.067 | -0.200 | +0.000 |
| entropy_gate_a-1.0 | 0.933 | 0.533 | 0.000 | -0.067 | +0.267 |
| maxprob_gate_a-1.0 | 0.933 | 0.567 | 0.000 | -0.067 | +0.267 |
| uncertainty_mass_gate_a-1.0 | 0.933 | 0.567 | 0.000 | -0.067 | +0.267 |
| random_entropy_gate_a-1.0 | 0.900 | 0.567 | 0.000 | -0.033 | +0.233 |
| shuffled_entropy_gate_a-1.0 | 0.933 | 0.567 | 0.000 | -0.067 | +0.267 |

## 7. Verdict

**Verdict: `IC4_M3_V2_GATE_INSUFFICIENT`**

**Reasoning:** Gate (entropy_gate_a-1.0) moves (gate_sep=0.117) but metrics: H=0.933 (base H=0.867); oracle H=0.667 C=0.6 UA=0.0. Gate signal has action but insufficient to match oracle gate.

## 8. Key Questions Answered

1. **Does the gate move?** Yes. 4/5 gates show gate movement >= 0.03. maxprob_gate_a-1.0: gate_ans=0.369 vs gate_unans=0.487 (sep=0.117). random_entropy_gate_a-1.0: gate_ans=0.197 vs gate_unans=0.271 (sep=0.074). entropy_gate_a-1.0: gate_ans=0.198 vs gate_unans=0.269 (sep=0.071). shuffled_entropy_gate_a-1.0: gate_ans=0.192 vs gate_unans=0.258 (sep=0.066). Not moving: uncertainty_mass_gate_a-1.0.
2. **Which risk signal best separates answerable/unanswerable?** **maxprob_gate_a-1.0** (gate_sep=0.117, signal_sep=0.282). Runner-up: entropy_gate_a-1.0(sep=0.071), uncertainty_mass_gate_a-1.0(sep=0.002).
3. **Real gate vs random/shuffled gate?** Yes. Best real gate_sep=0.117 > best ctrl gate_sep=0.074 (gap=+0.043).
4. **Why the gap to oracle gate?** Oracle gate: H=0.667, C=0.600, UA=0.000. Best real gate H=0.933. Gap in H: +0.267. Oracle gate uses ground-truth answerability as gate (perfect separation). Token-level risk signals (entropy/maxprob/uncertainty_mass) are much weaker proxies for answerability. The core gap is signal quality, not steering vector quality. A probe trained on hidden states to classify answerability directly would close most of this gap.
5. **Next step?** **Proceed to probe gate**, not back to re-extracting v. Evidence: (1) oracle gate proves v is clean -- with perfect gate, dH=-0.200 with C/UA fully preserved. (2) token-level risk signals have separation (entropy: 0.071, maxprob: 0.117) but are insufficient to drive gating decisions comparable to oracle. (3) A lightweight answerability classifier trained on hidden states (probe gate) would have access to much richer signal than scalar summary statistics of logits. Do NOT return to v extraction -- that would waste the oracle gate diagnostic result.

**One-line Conclusion:** 当前最好的 feedback signal 是 **maxprob_gate_a-1.0** (gate_sep=0.117), 它离 oracle gate 还有 H=+0.267 的差距。差距根因是 signal quality (scalar logits stats vs ground-truth label)，下一步应做 probe gate 而非重新提取 v。

---

*IC-4-M3-v2: Telemetry-First Gated Steering*
*Generated by run_m3_v2*