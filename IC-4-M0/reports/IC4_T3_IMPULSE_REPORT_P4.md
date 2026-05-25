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
| Sweep steps | ['prefill', 8] |
| Sweep directions | ['v_hall', 'v_syc', 'random', 'shuffled', 'orthogonal'] |
| Sweep epsilons | [1.0, 3.0, 5.0] |
| Hallucination samples | 6 |
|  — unanswerable (hall-prone) | 4 |
|  — answerable (control) | 2 |
| Sycophancy samples | 30 |
| Max new tokens | 48 |
| Total combinations | 90 |
| Elapsed | 22378s (373.0 min) |

## 2. Baseline Metrics

### Hallucination Task

| Metric | Value |
|---|---|
| Hallucination rate | 1.0000 |
| Correct answer rate | 0.5000 |
| Calibrated abstention rate | 0.0000 |
| Unnecessary abstention rate | 0.0000 |

### Sycophancy Task

| Baseline sycophancy rate | 0.6333 |

## 3. Controllability Heatmap

### Hallucination Task — Mean |behavior_change| / epsilon by (layer, step)

#### Direction: v_hall

| Layer | prefill | 8 |
|---|---|---|
| 10 | 0.017 | 0.019 |
| 12 | 0.000 | 0.000 |
| 14 | 0.000 | 0.000 |

#### Direction: v_syc

| Layer | prefill | 8 |
|---|---|---|
| 10 | 0.006 | 0.035 |
| 12 | 0.000 | 0.030 |
| 14 | 0.074 | 0.006 |

#### Direction: random

| Layer | prefill | 8 |
|---|---|---|
| 10 | 0.048 | 0.015 |
| 12 | 0.070 | 0.000 |
| 14 | 0.039 | 0.011 |

#### Direction: shuffled

| Layer | prefill | 8 |
|---|---|---|
| 10 | 0.028 | 0.000 |
| 12 | 0.020 | 0.000 |
| 14 | 0.085 | 0.000 |

#### Direction: orthogonal

| Layer | prefill | 8 |
|---|---|---|
| 10 | 0.022 | 0.011 |
| 12 | 0.067 | 0.020 |
| 14 | 0.000 | 0.020 |

### Sycophancy Task — Mean |behavior_change| / epsilon by (layer, step)

#### Direction: v_hall

| Layer | prefill | 8 |
|---|---|---|
| 10 | 0.061 | 0.000 |
| 12 | 0.056 | 0.000 |
| 14 | 0.023 | 0.000 |

#### Direction: v_syc

| Layer | prefill | 8 |
|---|---|---|
| 10 | 0.117 | 0.000 |
| 12 | 0.061 | 0.000 |
| 14 | 0.032 | 0.000 |

#### Direction: random

| Layer | prefill | 8 |
|---|---|---|
| 10 | 0.067 | 0.000 |
| 12 | 0.023 | 0.000 |
| 14 | 0.035 | 0.000 |

#### Direction: shuffled

| Layer | prefill | 8 |
|---|---|---|
| 10 | 0.050 | 0.000 |
| 12 | 0.061 | 0.000 |
| 14 | 0.025 | 0.000 |

#### Direction: orthogonal

| Layer | prefill | 8 |
|---|---|---|
| 10 | 0.050 | 0.000 |
| 12 | 0.036 | 0.000 |
| 14 | 0.025 | 0.000 |

## 4. Trajectory Displacement Analysis

Mean trajectory displacement by (layer, step, direction) across all epsilons:

### Hallucination

| Layer | Step | Direction | Mean Disp | Max Disp | Impulse Disp |
|---|---|---|---|---|---|
| 10 | 8 | orthogonal | 2.3638 | 5.4838 | 0.0000 |
| 10 | 8 | random | 3.7262 | 7.8836 | 0.0000 |
| 10 | 8 | shuffled | 4.1776 | 8.2914 | 0.0000 |
| 10 | 8 | v_hall | 4.6106 | 8.8946 | 0.0000 |
| 10 | 8 | v_syc | 3.5559 | 7.8659 | 0.0000 |
| 10 | prefill | orthogonal | 11.6527 | 15.4238 | 3.0000 |
| 10 | prefill | random | 12.0692 | 18.0788 | 3.0000 |
| 10 | prefill | shuffled | 13.8632 | 19.1509 | 3.0000 |
| 10 | prefill | v_hall | 13.9436 | 19.0617 | 3.0000 |
| 10 | prefill | v_syc | 15.4523 | 19.7166 | 3.0000 |
| 12 | 8 | orthogonal | 4.0984 | 8.2437 | 0.0000 |
| 12 | 8 | random | 2.7821 | 6.6551 | 0.0000 |
| 12 | 8 | shuffled | 3.9326 | 7.8376 | 0.0000 |
| 12 | 8 | v_hall | 3.8999 | 7.6533 | 0.0000 |
| 12 | 8 | v_syc | 3.7335 | 8.2481 | 0.0000 |
| 12 | prefill | orthogonal | 11.6483 | 17.0645 | 3.0000 |
| 12 | prefill | random | 11.9039 | 17.9407 | 3.0000 |
| 12 | prefill | shuffled | 12.7709 | 18.2975 | 3.0000 |
| 12 | prefill | v_hall | 15.0415 | 20.6725 | 3.0000 |
| 12 | prefill | v_syc | 14.0144 | 18.4405 | 3.0000 |
| 14 | 8 | orthogonal | 5.3597 | 9.8265 | 0.0000 |
| 14 | 8 | random | 3.6092 | 7.1933 | 0.0000 |
| 14 | 8 | shuffled | 4.4425 | 8.4845 | 0.0000 |
| 14 | 8 | v_hall | 5.2590 | 9.7262 | 0.0000 |
| 14 | 8 | v_syc | 2.6662 | 6.3123 | 0.0000 |
| 14 | prefill | orthogonal | 8.4986 | 14.4892 | 3.0000 |
| 14 | prefill | random | 11.7178 | 17.9044 | 3.0000 |
| 14 | prefill | shuffled | 7.9451 | 16.2021 | 3.0000 |
| 14 | prefill | v_hall | 9.6098 | 15.4984 | 3.0000 |
| 14 | prefill | v_syc | 14.3179 | 19.4878 | 3.0000 |

### Sycophancy

| Layer | Step | Direction | Mean Disp | Max Disp | Impulse Disp |
|---|---|---|---|---|---|
| 10 | 8 | orthogonal | 1.2408 | 5.4698 | 0.0000 |
| 10 | 8 | random | 1.2002 | 5.4979 | 0.0000 |
| 10 | 8 | shuffled | 1.8646 | 6.2272 | 0.0000 |
| 10 | 8 | v_hall | 1.8689 | 6.5650 | 0.0000 |
| 10 | 8 | v_syc | 2.3893 | 7.2974 | 0.0000 |
| 10 | prefill | orthogonal | 9.7609 | 16.0852 | 3.0000 |
| 10 | prefill | random | 9.8784 | 16.0695 | 3.0000 |
| 10 | prefill | shuffled | 10.4936 | 17.1149 | 3.0000 |
| 10 | prefill | v_hall | 11.5783 | 17.4186 | 3.0000 |
| 10 | prefill | v_syc | 11.6305 | 17.4438 | 3.0000 |
| 12 | 8 | orthogonal | 1.4739 | 6.0566 | 0.0000 |
| 12 | 8 | random | 1.6861 | 5.6761 | 0.0000 |
| 12 | 8 | shuffled | 1.8639 | 6.7103 | 0.0000 |
| 12 | 8 | v_hall | 2.4540 | 7.2453 | 0.0000 |
| 12 | 8 | v_syc | 2.0738 | 6.3314 | 0.0000 |
| 12 | prefill | orthogonal | 8.8033 | 16.1852 | 3.0000 |
| 12 | prefill | random | 8.7755 | 17.4090 | 3.0000 |
| 12 | prefill | shuffled | 10.8221 | 17.6499 | 3.0000 |
| 12 | prefill | v_hall | 11.3193 | 17.9306 | 3.0000 |
| 12 | prefill | v_syc | 10.3789 | 16.8514 | 3.0000 |
| 14 | 8 | orthogonal | 1.5534 | 6.3292 | 0.0000 |
| 14 | 8 | random | 1.0825 | 5.5272 | 0.0000 |
| 14 | 8 | shuffled | 1.4412 | 6.2161 | 0.0000 |
| 14 | 8 | v_hall | 1.6973 | 6.0261 | 0.0000 |
| 14 | 8 | v_syc | 1.9127 | 6.1974 | 0.0000 |
| 14 | prefill | orthogonal | 6.8103 | 15.0391 | 3.0000 |
| 14 | prefill | random | 8.7531 | 16.4764 | 3.0000 |
| 14 | prefill | shuffled | 7.4306 | 16.2330 | 3.0000 |
| 14 | prefill | v_hall | 8.0171 | 17.5829 | 3.0000 |
| 14 | prefill | v_syc | 9.0808 | 16.8893 | 3.0000 |

## 5. Output Degeneration

- hallucination: 0/540 (0.0000) degenerated outputs
- sycophancy: 0/2700 (0.0000) degenerated outputs

## 6. Reversibility Analysis

### Hallucination

| Reversibility | Count |
|---|---|
| no_impulse_effect | 270 |
| persistent | 230 |
| collapse_back | 39 |
| partial_collapse | 1 |

### Sycophancy

| Reversibility | Count |
|---|---|
| no_impulse_effect | 1350 |
| persistent | 1093 |
| collapse_back | 224 |
| partial_collapse | 33 |

### Sycophancy: Collapse-back vs Persistent by Direction

| Direction | Collapse/Partial | Persistent | Total |
|---|---|---|---|
| v_hall | 37 | 233 | 540 |
| v_syc | 49 | 221 | 540 |
| random | 51 | 219 | 540 |
| shuffled | 51 | 219 | 540 |
| orthogonal | 69 | 201 | 540 |

## 7. Local Controllability Score Summary

### Hallucination

| Direction | Mean Ctrl | Std Ctrl | Max Ctrl |
|---|---|---|---|
| orthogonal | 0.0235 | 0.0600 | 0.3333 |
| random | 0.0306 | 0.0800 | 0.3333 |
| shuffled | 0.0222 | 0.1142 | 1.0000 |
| v_hall | 0.0059 | 0.0384 | 0.3333 |
| v_syc | 0.0250 | 0.1134 | 1.0000 |

| Layer | Mean Ctrl | Std Ctrl | Max Ctrl |
|---|---|---|---|
| 10.0 | 0.0200 | 0.0691 | 0.5000 |
| 12.0 | 0.0207 | 0.0652 | 0.3333 |
| 14.0 | 0.0235 | 0.1163 | 1.0000 |

| Step | Mean Ctrl | Std Ctrl | Max Ctrl |
|---|---|---|---|
| 8 | 0.0111 | 0.0470 | 0.3333 |
| prefill | 0.0317 | 0.1122 | 1.0000 |

### Sycophancy

| Direction | Mean Ctrl | Std Ctrl | Max Ctrl |
|---|---|---|---|
| orthogonal | 0.0185 | 0.1002 | 1.0000 |
| random | 0.0207 | 0.1032 | 1.0000 |
| shuffled | 0.0227 | 0.1043 | 1.0000 |
| v_hall | 0.0232 | 0.1055 | 1.0000 |
| v_syc | 0.0349 | 0.1335 | 1.0000 |

| Layer | Mean Ctrl | Std Ctrl | Max Ctrl |
|---|---|---|---|
| 10.0 | 0.0345 | 0.1335 | 1.0000 |
| 12.0 | 0.0236 | 0.1034 | 1.0000 |
| 14.0 | 0.0140 | 0.0875 | 1.0000 |

| Step | Mean Ctrl | Std Ctrl | Max Ctrl |
|---|---|---|---|
| 8 | 0.0000 | 0.0000 | 0.0000 |
| prefill | 0.0480 | 0.1519 | 1.0000 |

## 8. Key Questions

### Q1: Does hallucination have locally controllable hotspots?

Top 5 hallucination controllability hotspots:

| Layer | Step | Mean Controllability |
|---|---|---|
| 14 | prefill | 0.0396 |
| 12 | prefill | 0.0315 |
| 10 | prefill | 0.0241 |
| 10 | 8 | 0.0159 |
| 12 | 8 | 0.0100 |

v_hall mean controllability: 0.0059
random mean controllability: 0.0306
shuffled mean controllability: 0.0222
**Finding: v_hall does NOT show significantly higher controllability than random. Hallucination may not have clearly localized controllable hotspots.**

### Q2: Does sycophancy show 'worse after impulse / collapse back' trend?

Overall sycophancy collapse-back rate: 0.0952
v_syc collapse-back rate: 0.0907
v_syc: 51 samples became more sycophantic, 2 became less sycophantic
**Finding: Sycophancy does NOT show a strong collapse-back trend. Impulse effects may be more persistent.**

### Q3: Which directions are NOT on the causal path?

| Direction | Hall Ctrl | Syc Ctrl | On Causal Path? |
|---|---|---|---|
| v_hall | 0.0059 | 0.0232 | Likely NO |
| v_syc | 0.0250 | 0.0349 | Likely NO |
| random | 0.0306 | 0.0207 | Likely NO |
| shuffled | 0.0222 | 0.0227 | Likely NO |
| orthogonal | 0.0235 | 0.0185 | Likely NO |

**Directions likely NOT on the causal path: v_hall, random, shuffled, orthogonal**

## 9. Sycophancy Impulse Interpretation

> With the balanced sycophancy contrast set (syc + non-syc samples),
> this section interprets what T3 impulse results tell us about sycophancy
> controllability.

### Direction comparison

| Direction | Mean Ctrl | vs Random | Norm Source |
|---|---|---|---|
| v_syc | 0.0349 | 1.68x | steering vector |
| orthogonal | 0.0185 | 0.89x | norm-matched to v_syc |
| v_hall | 0.0232 | 1.12x | steering vector |
| shuffled | 0.0227 | 1.10x | norm-matched to v_hall |
| random | 0.0207 | 1.00x | unit norm |

### Direction-vs-Energy Decomposition

- **v_syc mean ctrl**: 0.0349
- **orthogonal mean ctrl** (same norm, orthogonal direction): 0.0185
- **random mean ctrl**: 0.0207
- **Pure directional contribution** (v_syc - orthogonal): 0.0164
- **Pure energy contribution** (orthogonal - random): -0.0022

**Direction-dominated**: the direction of v_syc matters more than its energy/norm. Sycophancy impulse effect is primarily direction-specific.

### Classification

**direction-sensitive**: v_syc controllability is significantly (>1.5x) higher than random baseline. Sycophancy behavior is specifically sensitive to the syc steering direction, suggesting v_syc captures a causally relevant axis.

**Collapse-back rate**: 0.095 (low)

### Comparison with Hallucination Impulse

| Dimension | Hallucination | Sycophancy |
|---|---|---|
| v_task mean ctrl | 0.0059 | 0.0349 |
| random mean ctrl | 0.0306 | 0.0207 |
| v_task/random ratio | 0.19x | 1.68x |
| Direction-specific? | Not confirmed | Potentially |

## 10. Epsilon Scaling Analysis

Mean controllability score by epsilon (should be roughly constant if linear):

### Hallucination

| Epsilon | Mean Ctrl Score | Mean Behavior Change | Mean Displacement |
|---|---|---|---|
| 1.0 | 0.0139 | -0.0139 | 4.5740 |
| 3.0 | 0.0259 | -0.0778 | 9.1940 |
| 5.0 | 0.0244 | -0.1167 | 10.4986 |

### Sycophancy

| Epsilon | Mean Ctrl Score | Mean Behavior Change | Mean Displacement |
|---|---|---|---|
| 1.0 | 0.0256 | 0.0056 | 2.9757 |
| 3.0 | 0.0241 | 0.0233 | 6.0145 |
| 5.0 | 0.0224 | 0.0500 | 7.9434 |

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

- `results_t3_impulse_p4/impulse_results.csv` — Full results table
- `results_t3_impulse_p4/steering_vectors.npz` — Computed steering vectors
- `results_t3_impulse_p4/baseline_results.json` — Baseline generation results
- `results_t3_impulse_p4/run_log.txt` — Execution log

---

*IC-4-T3: Impulse Response Map*
*Generated by run_t3_impulse_map.py*