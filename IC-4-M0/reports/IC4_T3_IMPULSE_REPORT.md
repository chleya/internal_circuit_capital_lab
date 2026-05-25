# IC-4-T3: Impulse Response Map Report

> Applies small impulses at specific (layer, step, direction, epsilon)
> combinations during generation to test local controllability.
>
> This is Experiment D of the IC-4 Trajectory Dynamics Phase 1 project.

## 1. Setup

| Parameter | Value |
|---|---|
| Model | Qwen2.5-0.5B-Instruct (CPU, float32) |
| Sweep layers | [10, 12, 14] |
| Sweep steps | ['prefill', 3, 8] |
| Sweep directions | ['v_hall', 'v_syc', 'random', 'shuffled'] |
| Sweep epsilons | [1.0, 3.0, 5.0] |
| Hallucination samples | 6 |
|  — unanswerable (hall-prone) | 4 |
|  — answerable (control) | 2 |
| Sycophancy samples | 20 |
| Max new tokens | 48 |
| Total combinations | 108 |
| Elapsed | 14131s (235.5 min) |

## 2. Baseline Metrics

### Hallucination Task

| Metric | Value |
|---|---|
| Hallucination rate | 1.0000 |
| Correct answer rate | 0.5000 |
| Calibrated abstention rate | 0.0000 |
| Unnecessary abstention rate | 0.0000 |

### Sycophancy Task

| Baseline sycophancy rate | 0.5500 |

## 3. Controllability Heatmap

### Hallucination Task — Mean |behavior_change| / epsilon by (layer, step)

#### Direction: v_hall

| Layer | prefill | 3 | 8 |
|---|---|---|---|
| 10 | 0.017 | 0.000 | 0.019 |
| 12 | 0.000 | 0.000 | 0.000 |
| 14 | 0.000 | 0.030 | 0.000 |

#### Direction: v_syc

| Layer | prefill | 3 | 8 |
|---|---|---|---|
| 10 | 0.006 | 0.019 | 0.035 |
| 12 | 0.056 | 0.011 | 0.030 |
| 14 | 0.074 | 0.011 | 0.006 |

#### Direction: random

| Layer | prefill | 3 | 8 |
|---|---|---|---|
| 10 | 0.048 | 0.019 | 0.015 |
| 12 | 0.070 | 0.030 | 0.000 |
| 14 | 0.039 | 0.000 | 0.011 |

#### Direction: shuffled

| Layer | prefill | 3 | 8 |
|---|---|---|---|
| 10 | 0.028 | 0.000 | 0.000 |
| 12 | 0.020 | 0.000 | 0.000 |
| 14 | 0.085 | 0.000 | 0.000 |

### Sycophancy Task — Mean |behavior_change| / epsilon by (layer, step)

#### Direction: v_hall

| Layer | prefill | 3 | 8 |
|---|---|---|---|
| 10 | 0.066 | 0.000 | 0.000 |
| 12 | 0.059 | 0.000 | 0.000 |
| 14 | 0.034 | 0.000 | 0.000 |

#### Direction: v_syc

| Layer | prefill | 3 | 8 |
|---|---|---|---|
| 10 | 0.141 | 0.034 | 0.000 |
| 12 | 0.079 | 0.034 | 0.000 |
| 14 | 0.044 | 0.009 | 0.000 |

#### Direction: random

| Layer | prefill | 3 | 8 |
|---|---|---|---|
| 10 | 0.082 | 0.000 | 0.000 |
| 12 | 0.034 | 0.000 | 0.000 |
| 14 | 0.009 | 0.000 | 0.000 |

#### Direction: shuffled

| Layer | prefill | 3 | 8 |
|---|---|---|---|
| 10 | 0.069 | 0.034 | 0.000 |
| 12 | 0.070 | 0.014 | 0.000 |
| 14 | 0.026 | 0.000 | 0.000 |

## 4. Trajectory Displacement Analysis

Mean trajectory displacement by (layer, step, direction) across all epsilons:

### Hallucination

| Layer | Step | Direction | Mean Disp | Max Disp | Impulse Disp |
|---|---|---|---|---|---|
| 10 | 3 | random | 6.4979 | 11.4286 | 0.0000 |
| 10 | 3 | shuffled | 4.8529 | 8.5352 | 0.0000 |
| 10 | 3 | v_hall | 3.6544 | 7.5791 | 0.0000 |
| 10 | 3 | v_syc | 5.4509 | 9.6851 | 0.0000 |
| 10 | 8 | random | 3.7262 | 7.8836 | 0.0000 |
| 10 | 8 | shuffled | 4.1776 | 8.2914 | 0.0000 |
| 10 | 8 | v_hall | 4.6106 | 8.8946 | 0.0000 |
| 10 | 8 | v_syc | 3.5559 | 7.8659 | 0.0000 |
| 10 | prefill | random | 12.0692 | 18.0788 | 3.0000 |
| 10 | prefill | shuffled | 13.8632 | 19.1509 | 3.0000 |
| 10 | prefill | v_hall | 13.9436 | 19.0617 | 3.0000 |
| 10 | prefill | v_syc | 14.8908 | 19.3648 | 3.0000 |
| 12 | 3 | random | 5.0713 | 9.9008 | 0.0000 |
| 12 | 3 | shuffled | 5.7755 | 9.8704 | 0.0000 |
| 12 | 3 | v_hall | 3.9405 | 8.2746 | 0.0000 |
| 12 | 3 | v_syc | 5.3688 | 9.4766 | 0.0000 |
| 12 | 8 | random | 2.7821 | 6.6551 | 0.0000 |
| 12 | 8 | shuffled | 3.9326 | 7.8376 | 0.0000 |
| 12 | 8 | v_hall | 3.8999 | 7.6533 | 0.0000 |
| 12 | 8 | v_syc | 3.7335 | 8.2481 | 0.0000 |
| 12 | prefill | random | 11.9039 | 17.9407 | 3.0000 |
| 12 | prefill | shuffled | 12.7709 | 18.2975 | 3.0000 |
| 12 | prefill | v_hall | 15.0415 | 20.6725 | 3.0000 |
| 12 | prefill | v_syc | 13.5134 | 18.2831 | 3.0000 |
| 14 | 3 | random | 4.7360 | 8.4657 | 0.0000 |
| 14 | 3 | shuffled | 0.0625 | 3.0000 | 0.0000 |
| 14 | 3 | v_hall | 1.4102 | 5.8202 | 0.0000 |
| 14 | 3 | v_syc | 3.1447 | 7.1402 | 0.0000 |
| 14 | 8 | random | 3.6092 | 7.1933 | 0.0000 |
| 14 | 8 | shuffled | 4.4425 | 8.4845 | 0.0000 |
| 14 | 8 | v_hall | 5.2590 | 9.7262 | 0.0000 |
| 14 | 8 | v_syc | 3.5055 | 7.5356 | 0.0000 |
| 14 | prefill | random | 11.7178 | 17.9044 | 3.0000 |
| 14 | prefill | shuffled | 7.9451 | 16.2021 | 3.0000 |
| 14 | prefill | v_hall | 9.6098 | 15.4984 | 3.0000 |
| 14 | prefill | v_syc | 14.3625 | 19.5689 | 3.0000 |

### Sycophancy

| Layer | Step | Direction | Mean Disp | Max Disp | Impulse Disp |
|---|---|---|---|---|---|
| 10 | 3 | random | 2.1105 | 5.8863 | 0.0000 |
| 10 | 3 | shuffled | 1.6562 | 5.0245 | 0.0000 |
| 10 | 3 | v_hall | 1.3380 | 4.8550 | 0.0000 |
| 10 | 3 | v_syc | 1.8997 | 5.0725 | 0.0000 |
| 10 | 8 | random | 1.0252 | 4.7545 | 0.0000 |
| 10 | 8 | shuffled | 1.8788 | 6.2560 | 0.0000 |
| 10 | 8 | v_hall | 1.6031 | 5.9083 | 0.0000 |
| 10 | 8 | v_syc | 2.6689 | 7.6570 | 0.0000 |
| 10 | prefill | random | 10.0367 | 15.9149 | 3.0000 |
| 10 | prefill | shuffled | 10.7266 | 17.1392 | 3.0000 |
| 10 | prefill | v_hall | 11.2421 | 17.2446 | 3.0000 |
| 10 | prefill | v_syc | 11.7318 | 17.1710 | 3.0000 |
| 12 | 3 | random | 2.4685 | 6.4865 | 0.0000 |
| 12 | 3 | shuffled | 2.1606 | 5.4404 | 0.0000 |
| 12 | 3 | v_hall | 1.4780 | 5.0992 | 0.0000 |
| 12 | 3 | v_syc | 2.0386 | 5.2407 | 0.0000 |
| 12 | 8 | random | 1.8264 | 5.4025 | 0.0000 |
| 12 | 8 | shuffled | 1.9779 | 7.0927 | 0.0000 |
| 12 | 8 | v_hall | 2.4836 | 7.4421 | 0.0000 |
| 12 | 8 | v_syc | 2.2392 | 6.7077 | 0.0000 |
| 12 | prefill | random | 7.8464 | 16.3951 | 3.0000 |
| 12 | prefill | shuffled | 11.0461 | 17.6195 | 3.0000 |
| 12 | prefill | v_hall | 11.1697 | 18.0224 | 3.0000 |
| 12 | prefill | v_syc | 10.7505 | 16.2002 | 3.0000 |
| 14 | 3 | random | 0.3991 | 3.5440 | 0.0000 |
| 14 | 3 | shuffled | 1.9139 | 5.0528 | 0.0000 |
| 14 | 3 | v_hall | 0.8485 | 3.8357 | 0.0000 |
| 14 | 3 | v_syc | 1.8087 | 5.0160 | 0.0000 |
| 14 | 8 | random | 1.0259 | 5.3531 | 0.0000 |
| 14 | 8 | shuffled | 1.1079 | 5.6909 | 0.0000 |
| 14 | 8 | v_hall | 1.5594 | 5.6779 | 0.0000 |
| 14 | 8 | v_syc | 2.0035 | 6.5343 | 0.0000 |
| 14 | prefill | random | 8.1437 | 15.5325 | 3.0000 |
| 14 | prefill | shuffled | 7.0875 | 15.7797 | 3.0000 |
| 14 | prefill | v_hall | 8.0787 | 17.0652 | 3.0000 |
| 14 | prefill | v_syc | 9.6098 | 16.5192 | 3.0000 |

## 5. Output Degeneration

- hallucination: 0/648 (0.0000) degenerated outputs
- sycophancy: 0/2160 (0.0000) degenerated outputs

## 6. Reversibility Analysis

### Hallucination

| Reversibility | Count |
|---|---|
| no_impulse_effect | 432 |
| persistent | 190 |
| collapse_back | 25 |
| partial_collapse | 1 |

### Sycophancy

| Reversibility | Count |
|---|---|
| no_impulse_effect | 1440 |
| persistent | 579 |
| collapse_back | 124 |
| partial_collapse | 17 |

### Sycophancy: Collapse-back vs Persistent by Direction

| Direction | Collapse/Partial | Persistent | Total |
|---|---|---|---|
| v_hall | 27 | 153 | 540 |
| v_syc | 36 | 144 | 540 |
| random | 41 | 139 | 540 |
| shuffled | 37 | 143 | 540 |

## 7. Local Controllability Score Summary

### Hallucination

| Direction | Mean Ctrl | Std Ctrl | Max Ctrl |
|---|---|---|---|
| random | 0.0257 | 0.0764 | 0.3333 |
| shuffled | 0.0148 | 0.0937 | 1.0000 |
| v_hall | 0.0072 | 0.0434 | 0.3333 |
| v_syc | 0.0274 | 0.1249 | 1.0000 |

| Layer | Mean Ctrl | Std Ctrl | Max Ctrl |
|---|---|---|---|
| 10.0 | 0.0170 | 0.0677 | 0.5000 |
| 12.0 | 0.0181 | 0.0879 | 1.0000 |
| 14.0 | 0.0213 | 0.1092 | 1.0000 |

| Step | Mean Ctrl | Std Ctrl | Max Ctrl |
|---|---|---|---|
| 3 | 0.0099 | 0.0521 | 0.3333 |
| 8 | 0.0096 | 0.0478 | 0.3333 |
| prefill | 0.0369 | 0.1370 | 1.0000 |

### Sycophancy

| Direction | Mean Ctrl | Std Ctrl | Max Ctrl |
|---|---|---|---|
| random | 0.0140 | 0.0874 | 1.0000 |
| shuffled | 0.0237 | 0.1122 | 1.0000 |
| v_hall | 0.0177 | 0.0922 | 1.0000 |
| v_syc | 0.0380 | 0.1409 | 1.0000 |

| Layer | Mean Ctrl | Std Ctrl | Max Ctrl |
|---|---|---|---|
| 10.0 | 0.0356 | 0.1359 | 1.0000 |
| 12.0 | 0.0243 | 0.1104 | 1.0000 |
| 14.0 | 0.0102 | 0.0755 | 1.0000 |

| Step | Mean Ctrl | Std Ctrl | Max Ctrl |
|---|---|---|---|
| 3 | 0.0106 | 0.0768 | 1.0000 |
| 8 | 0.0000 | 0.0000 | 0.0000 |
| prefill | 0.0594 | 0.1696 | 1.0000 |

## 8. Key Questions

### Q1: Does hallucination have locally controllable hotspots?

Top 5 hallucination controllability hotspots:

| Layer | Step | Mean Controllability |
|---|---|---|
| 14 | prefill | 0.0495 |
| 12 | prefill | 0.0366 |
| 10 | prefill | 0.0245 |
| 10 | 8 | 0.0171 |
| 14 | 3 | 0.0102 |

v_hall mean controllability: 0.0072
random mean controllability: 0.0257
shuffled mean controllability: 0.0148
**Finding: v_hall does NOT show significantly higher controllability than random. Hallucination may not have clearly localized controllable hotspots.**

### Q2: Does sycophancy show 'worse after impulse / collapse back' trend?

Overall sycophancy collapse-back rate: 0.0653
v_syc collapse-back rate: 0.0667
v_syc: 56 samples became more sycophantic, 0 became less sycophantic
**Finding: Sycophancy does NOT show a strong collapse-back trend. Impulse effects may be more persistent.**

### Q3: Which directions are NOT on the causal path?

| Direction | Hall Ctrl | Syc Ctrl | On Causal Path? |
|---|---|---|---|
| v_hall | 0.0072 | 0.0177 | Likely NO |
| v_syc | 0.0274 | 0.0380 | Likely YES |
| random | 0.0257 | 0.0140 | Likely NO |
| shuffled | 0.0148 | 0.0237 | Likely NO |

**Directions likely NOT on the causal path: v_hall, random**

## 9. Sycophancy Impulse Interpretation

> With the balanced sycophancy contrast set (syc + non-syc samples),
> this section interprets what T3 impulse results tell us about sycophancy
> controllability.

### Direction comparison

| Direction | Mean Ctrl | vs Random |
|---|---|---|
| v_syc | 0.0380 | 2.73x |
| v_hall | 0.0177 | 1.27x |
| random | 0.0140 | 1.00x |
| shuffled | 0.0237 | 1.70x |

### Classification

**direction-sensitive**: v_syc controllability is significantly (>1.5x) higher than random baseline. Sycophancy behavior is specifically sensitive to the syc steering direction, suggesting v_syc captures a causally relevant axis.

**Collapse-back rate**: 0.065 (low)

### Comparison with Hallucination Impulse

| Dimension | Hallucination | Sycophancy |
|---|---|---|
| v_task mean ctrl | 0.0072 | 0.0380 |
| random mean ctrl | 0.0257 | 0.0140 |
| v_task/random ratio | 0.28x | 2.73x |
| Direction-specific? | Not confirmed | Potentially |

## 10. Epsilon Scaling Analysis

Mean controllability score by epsilon (should be roughly constant if linear):

### Hallucination

| Epsilon | Mean Ctrl Score | Mean Behavior Change | Mean Displacement |
|---|---|---|---|
| 1.0 | 0.0162 | -0.0162 | 4.3506 |
| 3.0 | 0.0239 | -0.0718 | 7.6776 |
| 5.0 | 0.0162 | -0.0810 | 8.7078 |

### Sycophancy

| Epsilon | Mean Ctrl Score | Mean Behavior Change | Mean Displacement |
|---|---|---|---|
| 1.0 | 0.0264 | 0.0181 | 2.0366 |
| 3.0 | 0.0236 | 0.0569 | 4.7590 |
| 5.0 | 0.0200 | 0.0778 | 6.4536 |

## 11. Representation / Readout Caveat

> This impulse map records `state_last` (last token hidden state) at each
> forward step. The impulse is applied as an additive perturbation to the
> full hidden state tensor at the target layer. Trajectory displacement is
> computed as the L2 norm of the difference between impulse and baseline
> `state_last` vectors at each step.
>
> **Important limitations:**
> 1. The `state_last` readout may not capture behaviorally relevant signals
>    concentrated in non-last positions (especially during prefill).
> 2. The impulse is applied uniformly across all token positions in the
>    hidden state tensor. A position-specific impulse might yield different
>    controllability results.
> 3. The controllability score |behavior_change|/epsilon assumes a roughly
>    linear relationship between impulse magnitude and behavioral effect.
>    Nonlinear regime transitions at larger epsilon values would violate this.
> 4. The step counter uses call_count // n_target_layers, which assumes
>    all target layer hooks fire in order during each forward pass. If the
>    model uses any non-standard execution order, step counting could be off.
> 5. The hallucination and sycophancy classifiers are rule-based heuristics.
>    Misclassification of generated outputs would contaminate the
>    controllability scores. A probe-based classifier would be more reliable
>    but introduces its own representation caveats.

## 12. Output Files

- `results_t3_impulse/impulse_results.csv` — Full results table
- `results_t3_impulse/steering_vectors.npz` — Computed steering vectors
- `results_t3_impulse/baseline_results.json` — Baseline generation results
- `results_t3_impulse/run_log.txt` — Execution log

---

*IC-4-T3: Impulse Response Map*
*Generated by run_t3_impulse_map.py*