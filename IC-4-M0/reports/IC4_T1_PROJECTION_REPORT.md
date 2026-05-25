# IC-4-T1: Projection Analysis Report

> Projects hidden-state trajectories onto behavior-relevant directions
> to characterize when and how behavioral divergence becomes visible
> in the model's internal representations.

## 1. Setup

| Parameter | Value |
|---|---|
| Target layers | [np.int64(8), np.int64(10), np.int64(12), np.int64(14), np.int64(16), np.int64(20), np.int64(23)] |
| Main analysis layer | 12 |
| Prefill step | 0 |
| Separation threshold | 0.5 |
| Late-stage fraction | 0.25 |
| Random baseline seeds | 5 |
| Hallucination samples | 60 |
| Sycophancy samples | 60 |
| Elapsed | 5s (0.1 min) |

## 2. Behavior Distribution

### Hallucination Task

| Behavior | Count |
|---|---|
| hallucination | 26 |
| correct | 18 |
| incorrect_answerable | 12 |
| other_unanswerable | 2 |
| abstention | 2 |

### Sycophancy Task

| Behavior | Count |
|---|---|
| sycophantic | 35 |
| non_sycophantic | 25 |

## 3. Steering Vector Properties

### Layer 8

| Direction | L2 Norm | Cosine(v_hall, v_syc) |
|---|---|---|
| v_hall | 1.000000 | — |
| v_syc | 1.000000 | — |
| random | 1.000000 | — |
| shuffled | 1.000000 | — |
| cosine(v_hall, v_syc) | — | -0.009729 |
| cosine(v_hall, random) | — | 0.006587 |
| cosine(v_hall, shuffled) | — | -0.223205 |
| cosine(v_syc, random) | — | 0.027893 |
| cosine(v_syc, shuffled) | — | 0.030284 |

### Layer 10

| Direction | L2 Norm | Cosine(v_hall, v_syc) |
|---|---|---|
| v_hall | 1.000000 | — |
| v_syc | 1.000000 | — |
| random | 1.000000 | — |
| shuffled | 1.000000 | — |
| cosine(v_hall, v_syc) | — | -0.003995 |
| cosine(v_hall, random) | — | -0.009243 |
| cosine(v_hall, shuffled) | — | -0.412365 |
| cosine(v_syc, random) | — | -0.017106 |
| cosine(v_syc, shuffled) | — | 0.018183 |

### Layer 12

| Direction | L2 Norm | Cosine(v_hall, v_syc) |
|---|---|---|
| v_hall | 1.000000 | — |
| v_syc | 1.000000 | — |
| random | 1.000000 | — |
| shuffled | 1.000000 | — |
| cosine(v_hall, v_syc) | — | 0.016296 |
| cosine(v_hall, random) | — | -0.058207 |
| cosine(v_hall, shuffled) | — | -0.525153 |
| cosine(v_syc, random) | — | 0.011863 |
| cosine(v_syc, shuffled) | — | 0.085368 |

### Layer 14

| Direction | L2 Norm | Cosine(v_hall, v_syc) |
|---|---|---|
| v_hall | 1.000000 | — |
| v_syc | 1.000000 | — |
| random | 1.000000 | — |
| shuffled | 1.000000 | — |
| cosine(v_hall, v_syc) | — | -0.018603 |
| cosine(v_hall, random) | — | -0.003405 |
| cosine(v_hall, shuffled) | — | -0.455656 |
| cosine(v_syc, random) | — | -0.020291 |
| cosine(v_syc, shuffled) | — | 0.109529 |

### Layer 16

| Direction | L2 Norm | Cosine(v_hall, v_syc) |
|---|---|---|
| v_hall | 1.000000 | — |
| v_syc | 1.000000 | — |
| random | 1.000000 | — |
| shuffled | 1.000000 | — |
| cosine(v_hall, v_syc) | — | 0.052118 |
| cosine(v_hall, random) | — | 0.032890 |
| cosine(v_hall, shuffled) | — | -0.346545 |
| cosine(v_syc, random) | — | -0.019340 |
| cosine(v_syc, shuffled) | — | -0.061259 |

### Layer 20

| Direction | L2 Norm | Cosine(v_hall, v_syc) |
|---|---|---|
| v_hall | 1.000000 | — |
| v_syc | 1.000000 | — |
| random | 1.000000 | — |
| shuffled | 1.000000 | — |
| cosine(v_hall, v_syc) | — | -0.106003 |
| cosine(v_hall, random) | — | 0.002287 |
| cosine(v_hall, shuffled) | — | -0.184936 |
| cosine(v_syc, random) | — | 0.000798 |
| cosine(v_syc, shuffled) | — | -0.119650 |

### Layer 23

| Direction | L2 Norm | Cosine(v_hall, v_syc) |
|---|---|---|
| v_hall | 1.000000 | — |
| v_syc | 1.000000 | — |
| random | 1.000000 | — |
| shuffled | 1.000000 | — |
| cosine(v_hall, v_syc) | — | 0.065257 |
| cosine(v_hall, random) | — | -0.044817 |
| cosine(v_hall, shuffled) | — | -0.154352 |
| cosine(v_syc, random) | — | -0.053490 |
| cosine(v_syc, shuffled) | — | 0.046721 |

## 4. Main Results: Layer 12

### 4.1 Hallucination vs Abstention (v_hall direction)

| Statistic | Value |
|---|---|
| Earliest visible separation step | 0 |
| Max separation step | 0 |
| Max separation value | 2.397260904312134 |
| Late-stage variance | 0.159559 |
| Valid steps | 48 |

Steps with significant separation (p<0.05): [9, 13, 22, 28, 36, 41, 42, 43, 44, 45, 46]

### 4.2 Sycophantic vs Non-Sycophantic (v_syc direction)

| Statistic | Value |
|---|---|
| Earliest visible separation step | 0 |
| Max separation step | 0 |
| Max separation value | 1.7887585163116455 |
| Late-stage variance | 0.010260 |
| Collapse ratio | 0.3466542661190033 |
| Valid steps | 48 |

Steps with significant separation (p<0.05): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

**Interpretation: Moderate collapse** — sycophancy signal weakens but persists during generation.

### 4.3 Control Directions (random, shuffled)

| Direction | Comparison | Earliest Sep | Max Sep Value | Late-Stage Var |
|---|---|---|---|---|
| random | hall_vs_abst | 21 | -1.298676609992981 | 0.096952 |
| random | syc_vs_nonsyc | None | -0.1269664317369461 | 0.000561 |
| shuffled | hall_vs_abst | 0 | -1.4816615581512451 | 0.158444 |
| shuffled | syc_vs_nonsyc | None | 0.15270251035690308 | 0.000273 |

#### Random Baseline (5 seeds)

| Metric | Mean | Std |
|---|---|---|
| Max hall separation (random dir) | 0.683585 | 0.063541 |
| Max syc separation (random dir) | 0.131502 | 0.014081 |

## 5. Key Questions

### Q1: When do hallucination vs abstention separate?

Separation is visible from the **prefill step** (step 0). The model's internal representation already distinguishes hallucination-prone from abstention-prone inputs before any generation occurs.

### Q2: Sycophancy — does syc vs non-syc separate at prefill? Does it collapse?

| Statistic | Value |
|---|---|
| Earliest visible separation step | 0 |
| Max separation step | 0 |
| Max separation value | 1.7887585163116455 |
| Late-stage variance | 0.010260 |
| Collapse ratio | 0.3466542661190033 |
| Valid steps | 48 |

Steps with significant separation (p<0.05): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

**Sycophancy separates from prefill (step 0)** — like hallucination, the model's internal representation distinguishes sycophancy-prone from correction-prone inputs before any generation occurs.
**Moderate collapse** (ratio=0.347): syc signal weakens but persists.

### Q3: Do random/shuffled directions show no structure?

**Unexpected structure detected** in control directions. This could indicate: (1) the hidden state space has low effective dimensionality, making random projections likely to capture some variance, or (2) the separation threshold is too low. Compare control separation magnitudes to v_hall/v_syc separation magnitudes to assess significance.

Quantitative comparison: v_hall max separation = 2.3973, random direction mean max separation = 0.6836, ratio = 3.51x.

## 6. Supplementary: Other Layers

### Layer 8

| Direction | Comparison | Earliest Sep | Max Sep Value | Late-Stage Var |
|---|---|---|---|---|
| v_hall | hall_vs_abst | 0 | 1.5163569450378418 | 0.076987 |
| v_syc | syc_vs_nonsyc | 0 | 1.1350384950637817 | 0.005260 |
| random | hall_vs_abst | 30 | -0.860378623008728 | 0.071748 |
| random | syc_vs_nonsyc | None | -0.14106400310993195 | 0.001106 |
| shuffled | hall_vs_abst | 2 | -0.7542492747306824 | 0.037227 |
| shuffled | syc_vs_nonsyc | None | 0.268909752368927 | 0.000675 |

### Layer 10

| Direction | Comparison | Earliest Sep | Max Sep Value | Late-Stage Var |
|---|---|---|---|---|
| v_hall | hall_vs_abst | 0 | 1.7587493658065796 | 0.137265 |
| v_syc | syc_vs_nonsyc | 0 | 2.969407320022583 | 0.007137 |
| random | hall_vs_abst | 6 | -0.9835256338119507 | 0.061453 |
| random | syc_vs_nonsyc | None | -0.19048404693603516 | 0.000265 |
| shuffled | hall_vs_abst | 0 | -0.9257146716117859 | 0.160402 |
| shuffled | syc_vs_nonsyc | None | 0.2620164752006531 | 0.000183 |

### Layer 14

| Direction | Comparison | Earliest Sep | Max Sep Value | Late-Stage Var |
|---|---|---|---|---|
| v_hall | hall_vs_abst | 0 | 3.1848344802856445 | 0.136321 |
| v_syc | syc_vs_nonsyc | 0 | 3.844142436981201 | 0.015981 |
| random | hall_vs_abst | 21 | -0.9068057537078857 | 0.050804 |
| random | syc_vs_nonsyc | None | 0.17523036897182465 | 0.001373 |
| shuffled | hall_vs_abst | 0 | -1.4511884450912476 | 0.348673 |
| shuffled | syc_vs_nonsyc | None | 0.43138518929481506 | 0.002153 |

### Layer 16

| Direction | Comparison | Earliest Sep | Max Sep Value | Late-Stage Var |
|---|---|---|---|---|
| v_hall | hall_vs_abst | 0 | 3.617392063140869 | 0.070869 |
| v_syc | syc_vs_nonsyc | 0 | 1.20371675491333 | 0.010936 |
| random | hall_vs_abst | 6 | -1.2822678089141846 | 0.149259 |
| random | syc_vs_nonsyc | None | 0.26008176803588867 | 0.001662 |
| shuffled | hall_vs_abst | 0 | 3.0281381607055664 | 0.447672 |
| shuffled | syc_vs_nonsyc | None | -0.3933233320713043 | 0.005567 |

### Layer 20

| Direction | Comparison | Earliest Sep | Max Sep Value | Late-Stage Var |
|---|---|---|---|---|
| v_hall | hall_vs_abst | 0 | 8.132556915283203 | 1.070993 |
| v_syc | syc_vs_nonsyc | 0 | 3.0764198303222656 | 0.044430 |
| random | hall_vs_abst | 2 | -2.403174877166748 | 0.516249 |
| random | syc_vs_nonsyc | None | 0.4222700893878937 | 0.000816 |
| shuffled | hall_vs_abst | 0 | 5.617788314819336 | 1.013965 |
| shuffled | syc_vs_nonsyc | 1 | -0.6058210134506226 | 0.008819 |

### Layer 23

| Direction | Comparison | Earliest Sep | Max Sep Value | Late-Stage Var |
|---|---|---|---|---|
| v_hall | hall_vs_abst | 0 | 16.08723258972168 | 7.237216 |
| v_syc | syc_vs_nonsyc | 0 | 3.898693799972534 | 0.177545 |
| random | hall_vs_abst | 0 | -3.2904908657073975 | 0.498252 |
| random | syc_vs_nonsyc | 2 | -1.0320649147033691 | 0.007225 |
| shuffled | hall_vs_abst | 0 | 14.857720375061035 | 8.865579 |
| shuffled | syc_vs_nonsyc | 1 | -2.126110792160034 | 0.343746 |

## 7. Representation / Readout Caveat

> All projections in this analysis use `state_last` (last-token hidden state)
> captured at each forward step. This readout choice has known limitations:
>
> 1. **Position bias**: The last token position may not be the most informative
>    readout for all behavioral signals. During prefill, the last token is the
>    final prompt token; during decode, it is the newly generated token.
>    These are fundamentally different positions with different informational roles.
>
> 2. **Window4 alternative**: The T0 capture also recorded `state_window4` (mean of
>    last 4 positions during prefill), which may better capture distributed signals.
>    However, during decode steps (seq_len=1), window4 is identical to state_last.
>
> 3. **Projection ≠ causation**: Finding that trajectories separate along v_hall
>    does not imply that v_hall direction *causes* hallucination. It only shows
>    that the model's internal representations encode behavior-relevant information
>    that is linearly accessible. Causal claims require intervention experiments.
>
> 4. **Normalization**: Steering vectors are L2-normalized. Projection magnitudes
>    are therefore in units of 'hidden-state component along the normalized direction'.
>    Cross-direction magnitude comparisons are valid; cross-layer comparisons require
>    caution because hidden-state norms may vary by layer.
>
> 5. **Sample size**: Small behavior-class counts (especially for sycophancy)
>    limit statistical power. Per-step t-tests should be interpreted with
>    Bonferroni or FDR correction for the number of steps tested.

## 8. Output Files

- `results_t1_projection/projections_layer*_*.npz` — per-sample projection arrays
- `results_t1_projection/steering_vectors_layer*.npz` — computed steering vectors
- `results_t1_projection/average_projection_curves.csv` — mean projection per (layer, direction, task, behavior, step)
- `results_t1_projection/separation_statistics.csv` — separation statistics summary
- `results_t1_projection/per_sample_projections.csv` — per-sample projection values
- `results_t1_projection/random_baseline_statistics.csv` — random direction baselines

---

*IC-4-T1: Projection Analysis*
*Generated by run_t1_projection_analysis.py*