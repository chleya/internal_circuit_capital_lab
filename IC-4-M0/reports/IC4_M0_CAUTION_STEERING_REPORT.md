# IC-4-M0: Caution Steering Report

## Experiment Configuration

| Parameter | Value |
|---|---|
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| Train samples | 100 |
| Test samples | 100 |
| Total layers | 24 |
| Target layer | 12 |
| Alphas | [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0] |

## Results Summary

| mode | alpha | hallucination_rate | calibrated_abstention_rate | correct_answer_rate | unnecessary_abstention_rate | style_only_score | avg_answerable_uncertainty | avg_unanswerable_uncertainty |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 0.0000 | 0.9600 | 0.0000 | 0.8600 | 0.0400 | 0.1031 | 0.0200 | 0.0800 |
| prompt_only | 0.0000 | 0.3200 | 0.3400 | 0.4400 | 0.3000 | 0.6667 | 0.1800 | 0.0400 |
| steering_a-2.0 | -2.0000 | 0.4800 | 0.2600 | 0.7200 | 0.2800 | 0.6531 | 0.0800 | 0.2400 |
| steering_a-1.0 | -1.0000 | 0.4600 | 0.3200 | 0.8400 | 0.1600 | 0.5106 | 0.0600 | 0.1800 |
| steering_a-0.5 | -0.5000 | 0.6800 | 0.2000 | 0.7800 | 0.0400 | 0.3188 | 0.1000 | 0.1200 |
| steering_a0.0 | 0.0000 | 0.9600 | 0.0000 | 0.8600 | 0.0400 | 0.1031 | 0.0200 | 0.0800 |
| steering_a0.5 | 0.5000 | 0.8800 | 0.0800 | 0.8600 | 0.0000 | 0.0449 | 0.0200 | 0.0200 |
| steering_a1.0 | 1.0000 | 0.9400 | 0.0600 | 0.8600 | 0.0000 | 0.0421 | 0.0200 | 0.0200 |
| steering_a2.0 | 2.0000 | 0.9800 | 0.0200 | 0.8600 | 0.0600 | 0.0000 | 0.0000 | 0.0000 |
| random_a-2.0 | -2.0000 | 0.9800 | 0.0200 | 0.8400 | 0.0200 | 0.0606 | 0.0000 | 0.0600 |
| random_a-1.0 | -1.0000 | 0.9400 | 0.0600 | 0.8200 | 0.0000 | 0.0842 | 0.0400 | 0.0400 |
| random_a-0.5 | -0.5000 | 0.9200 | 0.0800 | 0.8400 | 0.0200 | 0.1290 | 0.0800 | 0.0400 |
| random_a0.5 | 0.5000 | 0.8600 | 0.0400 | 0.8600 | 0.0400 | 0.1149 | 0.0400 | 0.0600 |
| random_a1.0 | 1.0000 | 0.8400 | 0.0800 | 0.8200 | 0.0400 | 0.1412 | 0.0600 | 0.0600 |
| random_a2.0 | 2.0000 | 0.6800 | 0.1400 | 0.8400 | 0.0400 | 0.2029 | 0.0200 | 0.1200 |
| shuffled_a-2.0 | -2.0000 | 1.0000 | 0.0000 | 0.8600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| shuffled_a-1.0 | -1.0000 | 0.9200 | 0.0600 | 0.8600 | 0.0200 | 0.0000 | 0.0000 | 0.0000 |
| shuffled_a-0.5 | -0.5000 | 0.9400 | 0.0200 | 0.8800 | 0.0200 | 0.0421 | 0.0000 | 0.0400 |
| shuffled_a0.5 | 0.5000 | 0.7600 | 0.1800 | 0.8000 | 0.0400 | 0.1818 | 0.0800 | 0.0600 |
| shuffled_a1.0 | 1.0000 | 0.6600 | 0.2000 | 0.7400 | 0.0800 | 0.2388 | 0.1200 | 0.0400 |
| shuffled_a2.0 | 2.0000 | 0.1600 | 0.5400 | 0.7800 | 0.0800 | 2.2353 | 0.1200 | 0.2600 |

## Alpha Sweep Analysis

### Steering Vector

| Alpha | Hallucination Rate | Correct Answer Rate | Unnecessary Abstention | Calibrated Abstention | Style Only |
|---|---|---|---|---|---|
| -2.0000 | 0.4800 | 0.7200 | 0.2800 | 0.2600 | 0.6531 |
| -1.0000 | 0.4600 | 0.8400 | 0.1600 | 0.3200 | 0.5106 |
| -0.5000 | 0.6800 | 0.7800 | 0.0400 | 0.2000 | 0.3188 |
| 0.0000 | 0.9600 | 0.8600 | 0.0400 | 0.0000 | 0.1031 |
| 0.5000 | 0.8800 | 0.8600 | 0.0000 | 0.0800 | 0.0449 |
| 1.0000 | 0.9400 | 0.8600 | 0.0000 | 0.0600 | 0.0421 |
| 2.0000 | 0.9800 | 0.8600 | 0.0600 | 0.0200 | 0.0000 |

## Comparison Summary

| Mode | Hallucination | Correct Answer | Calibrated Abstention | Unnecessary Abstention | Style Only |
|---|---|---|---|---|---|
| base | 0.9600 | 0.8600 | 0.0000 | 0.0400 | 0.1031 |
| prompt_only | 0.3200 | 0.4400 | 0.3400 | 0.3000 | 0.6667 |
| steering_a-2.0 | 0.4800 | 0.7200 | 0.2600 | 0.2800 | 0.6531 |
| steering_a-1.0 | 0.4600 | 0.8400 | 0.3200 | 0.1600 | 0.5106 |
| steering_a-0.5 | 0.6800 | 0.7800 | 0.2000 | 0.0400 | 0.3188 |
| steering_a0.5 | 0.8800 | 0.8600 | 0.0800 | 0.0000 | 0.0449 |
| steering_a1.0 | 0.9400 | 0.8600 | 0.0600 | 0.0000 | 0.0421 |
| steering_a2.0 | 0.9800 | 0.8600 | 0.0200 | 0.0600 | 0.0000 |

### Control Vectors

| Mode | Alpha | Hallucination | Correct Answer |
|---|---|---|---|
| random_a-2.0 | -2.0000 | 0.9800 | 0.8400 |
| random_a-1.0 | -1.0000 | 0.9400 | 0.8200 |
| random_a-0.5 | -0.5000 | 0.9200 | 0.8400 |
| random_a0.5 | 0.5000 | 0.8600 | 0.8600 |
| random_a1.0 | 1.0000 | 0.8400 | 0.8200 |
| random_a2.0 | 2.0000 | 0.6800 | 0.8400 |
| shuffled_a-2.0 | -2.0000 | 1.0000 | 0.8600 |
| shuffled_a-1.0 | -1.0000 | 0.9200 | 0.8600 |
| shuffled_a-0.5 | -0.5000 | 0.9400 | 0.8800 |
| shuffled_a0.5 | 0.5000 | 0.7600 | 0.8000 |
| shuffled_a1.0 | 1.0000 | 0.6600 | 0.7400 |
| shuffled_a2.0 | 2.0000 | 0.1600 | 0.7800 |

## Evaluation Detail

- hallucination_rate: Unanswerable samples where the model fabricates concrete answers.
- calibrated_abstention_rate: Unanswerable samples where the model explicitly states insufficient information.
- correct_answer_rate: Answerable samples where the model produces the correct gold answer.
- unnecessary_abstention_rate: Answerable samples where the model refuses to answer despite having sufficient context.
- style_only_score: Measures uncertainty marker increase without corresponding hallucination reduction.

## Verdict

**Verdict: `IC4_M0_MODEL_DAMAGE`**

**Reasoning:** Hallucination reduced by 52.1% but correct_answer dropped 2.3% (>5%) or unnecessary abstention increased 12.0% (>5%).

---

*IC-4-M0: Minimal Activation Steering Anti-Hallucination Experiment*
*Generated by IC-4-M0 report_writer*