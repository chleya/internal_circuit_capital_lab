# IC-4-M3-v4: Prefill / Token-0 Probe Gate Report

> **STATUS: RETRACTED / DOWNGRADED TO `PLATFORM_INVALID`**
>
> A critical platform-level bug was discovered in `run_m3_v4.py` AFTER this report was generated:
>
> The `_generate_prefill_probe_gated` function ran prefill WITHOUT steering, extracted
> hidden states for the probe, then registered the steering hook — but **reused the
> unsteered prefill's `past_key_values` for generation**. The model was generating from
> an unsteered prefill KV cache, NOT a steered one.
>
> This means the experiment tested: **"unsteered prefill KV + post-prefill steering"**,
> which is NOT a valid test of prefill-level gated steering.
>
> **Therefore:**
> - The verdict `IC4_M3_V4_CONTROL_ARTIFACT` is **not scientifically valid**
> - This report cannot prove "prefill signal insufficient" nor falsify the timing hypothesis
> - A corrected implementation (M3-v4b) has been created with proper two-pass prefill:
>   Pass 1 (unsteered) for probe → Pass 2 (steered, from scratch) for generation
> - See `reports_m3_v4b/IC4_M3_V4B_VALID_PREFILL_GATE_REPORT.md` for the valid experiment

---

## 1. M3-v3 Recap: Why Token-4 Fails

**M3-v3 Verdict: IC4_M3_V3_CONTROL_ARTIFACT**

| Metric | base | oracle_gate | probe_gate_a-1.0 |
|---|---|---|---|
| H | 0.8667 | 0.6667 | 0.9333 |
| C | 0.6000 | 0.6000 | 0.5667 |

Key findings from M3-v3:
- Probe trained perfectly: train_acc=1.0, cv_acc=0.9, AUC=1.0
- Gate separation was strong: answerable probe_score ≈ 1, unanswerable ≈ 0
- BUT behavioral outcome was terrible: H=0.9333 = indistinguishable from random/shuffled controls
- Root cause: intervention at token 4 is too late — model trajectory is already committed

## 2. M3-v4 Hypothesis

**If M3-v3's failure is from timing (not signal quality), then moving the gate to token 0 (prefill-level decision) should significantly improve behavioral outcomes.**

M3-v4 tests this by:
1. Using prefill/prompt hidden states (NO generated tokens) as probe input
2. Making gate decision at token 0, before any generation
3. Comparing three prefill representations

## 3. M3-v4 Design

1. Run prefill forward on prompt only → capture hidden states at layer 12
2. Extract representation:
   - `last_prompt_token`: hidden state at final prompt token position
   - `mean_pooled`: average over all prompt token hidden states
   - `question_span_pooled`: average over question-text tokens only
3. `probe_score = P(answerable)` via logistic regression
4. `gate = sigmoid(steepness * (threshold - probe_score))`
   - probe_score ≈ 1 (answerable) → gate ≈ 0 → minimal steering
   - probe_score ≈ 0 (unanswerable) → gate ≈ 1 → full steering
5. `alpha = alpha_max * gate`, applied from the first generated token

| Parameter | Value |
|---|---|
| Probe layer | 12 |
| Representations | last_prompt_token, mean_pooled, question_span_pooled |
| Probe model | logistic |
| Decision timing | token 0 (prefill) |
| Steepness | 10.0 |
| Threshold | 0.5 |

## 4. Probe Training Evaluation (per Representation)

| Representation | Train Acc | CV Acc (mean) | CV Acc (std) | AUC | N pos/neg |
|---|---|---|---|---|---|
| last_prompt_token | 1.0000 | 1.0 | 0.0 | 1.0 | 15/15 |
| mean_pooled | 1.0000 | 1.0 | 0.0 | 1.0 | 15/15 |
| question_span_pooled | 1.0000 | 1.0 | 0.0 | 1.0 | 15/15 |

## 5. Experiment Configuration

| Parameter | Value |
|---|---|
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| Device / dtype | cpu / float32 |
| Steering layer | [12] |
| alpha_max | [-1.0] |
| Elapsed | 3850s (64.2 min) |

## 6. Full Metrics Table

| Mode | H | C | UA | CA | Vector |
|---|---|---|---|---|---|
| base | 0.867 | 0.600 | 0.000 | 0.067 | none |
| prompt_only | 0.400 | 0.067 | 0.200 | 0.133 | none |
| steering_a-1.0 | 0.667 | 0.533 | 0.067 | 0.100 | steering |
| oracle_gate_a-1.0 | 0.667 | 0.600 | 0.000 | 0.100 | steering |
| prefill_probe_gate_a-1.0_last_prompt_token | 1.000 | 0.733 | 0.000 | 0.000 | steering |
| random_prefill_probe_gate_a-1.0_last_prompt_token | 1.000 | 0.733 | 0.000 | 0.000 | random |
| shuffled_prefill_probe_gate_a-1.0_last_prompt_token | 0.967 | 0.733 | 0.000 | 0.033 | shuffled |
| prefill_probe_gate_a-1.0_mean_pooled | 1.000 | 0.733 | 0.000 | 0.000 | steering |
| random_prefill_probe_gate_a-1.0_mean_pooled | 1.000 | 0.733 | 0.000 | 0.000 | random |
| shuffled_prefill_probe_gate_a-1.0_mean_pooled | 0.967 | 0.733 | 0.000 | 0.033 | shuffled |
| prefill_probe_gate_a-1.0_question_span_pooled | 1.000 | 0.733 | 0.000 | 0.000 | steering |
| random_prefill_probe_gate_a-1.0_question_span_pooled | 1.000 | 0.733 | 0.000 | 0.000 | random |
| shuffled_prefill_probe_gate_a-1.0_question_span_pooled | 0.967 | 0.733 | 0.000 | 0.033 | shuffled |

## 7. Prefill Probe Gate vs Oracle Gate vs M3-v3

| Mode | H | C | UA | dH_base | Gap to Oracle | Gap to M3-v3 |
|---|---|---|---|---|---|---|
| base | 0.867 | 0.600 | 0.000 | -- | -- | -- |
| prompt_only | 0.400 | 0.067 | 0.200 | -0.467 | -0.267 | -0.533 |
| oracle_gate | 0.667 | 0.600 | 0.000 | -0.200 | -- | -0.267 |
| steering_a-1.0 | 0.667 | 0.533 | 0.067 | -0.200 | +0.000 | -0.267 |
| prefill_probe_gate_a-1.0_last_prompt_token | 1.000 | 0.733 | 0.000 | +0.133 | +0.333 | +0.067 |
| prefill_probe_gate_a-1.0_mean_pooled | 1.000 | 0.733 | 0.000 | +0.133 | +0.333 | +0.067 |
| prefill_probe_gate_a-1.0_question_span_pooled | 1.000 | 0.733 | 0.000 | +0.133 | +0.333 | +0.067 |

## 8. Gate Telemetry (per Representation)

### last_prompt_token

| Metric | Answerable | Unanswerable | Separation |
|---|---|---|---|
| probe_score (P(answerable)) mean | 0.9705 | 0.0021 | 0.9683 |
| gate mean | 0.0235 | 0.9932 | 0.9696 |

### mean_pooled

| Metric | Answerable | Unanswerable | Separation |
|---|---|---|---|
| probe_score (P(answerable)) mean | 0.9651 | 0.0024 | 0.9627 |
| gate mean | 0.0127 | 0.9931 | 0.9804 |

### question_span_pooled

| Metric | Answerable | Unanswerable | Separation |
|---|---|---|---|
| probe_score (P(answerable)) mean | 0.9865 | 0.0072 | 0.9792 |
| gate mean | 0.0082 | 0.9927 | 0.9845 |

## 9. Verdict

**Original Verdict: `IC4_M3_V4_CONTROL_ARTIFACT`** (RETRACTED — see notice at top)

**Adjusted Verdict: `IC4_M3_V4_PLATFORM_INVALID`**

**Reasoning:** The experiment tested "unsteered prefill KV + post-prefill steering", not
"prefill-level gated steering with steered prefill KV". The `past_key_values` from the
unsteered prefill were reused for generation, creating a platform-level confound. The
corrected implementation is in M3-v4b (`run_m3_v4b.py`).

## 10. Key Questions

### 10.1 Does prefill/token-0 gate outperform M3-v3 token-4 gate?
**NO.** Best prefill gate (prefill_probe_gate_a-1.0_last_prompt_token, H=1.000) is WORSE than M3-v3 token-4 gate (H=0.933). The prefill representation may carry weaker signal than the trajectory-state representation.

### 10.2 Does it beat random/shuffled controls?
**NO.** Best probe gate is indistinguishable from controls. Prefill signal alone is insufficient.

### 10.3 Does this support 'timing is the main bottleneck' hypothesis?
**NO.** Prefill gate does not close the gap to oracle. Timing is NOT the main bottleneck — the nature of the signal or the steering mechanism itself requires deeper investigation.

---

*IC-4-M3-v4: Prefill / Token-0 Probe Gate*
*Generated by run_m3_v4*