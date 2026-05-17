# IC-4-M5 P0-A: Cross-Scenario on Cross-Seed

> Validates large and hard_ood scenarios across seeds [0,1,2] at layer=12 / alpha=-1.0.

## 1. Complete Matrix

| Seed | Scenario | Mode | H | C | UA | Oracle Gap |
|---|---|---:|---:|---:|---:|
| 0 | large | base | 0.833 | 0.550 | 0.000 | 0.1333 |
| 0 | large | oracle_gate_a-1.0 | 0.700 | 0.550 | 0.000 | 0.0 |
| 0 | large | real_single_pass_hard_gate_a-1.0 | 0.700 | 0.550 | 0.000 | 0.0 |
| 0 | large | random_single_pass_hard_gate_a-1.0 | 0.933 | 0.550 | 0.000 | 0.2333 |
| 0 | large | shuffled_single_pass_hard_gate_a-1.0 | 0.750 | 0.550 | 0.000 | 0.05 |
| 1 | large | base | 0.800 | 0.600 | 0.000 | 0.2333 |
| 1 | large | oracle_gate_a-1.0 | 0.567 | 0.600 | 0.000 | 0.0 |
| 1 | large | real_single_pass_hard_gate_a-1.0 | 0.567 | 0.600 | 0.000 | 0.0 |
| 1 | large | random_single_pass_hard_gate_a-1.0 | 0.867 | 0.600 | 0.000 | 0.3 |
| 1 | large | shuffled_single_pass_hard_gate_a-1.0 | 0.583 | 0.600 | 0.000 | 0.0166 |
| 2 | large | base | 0.800 | 0.433 | 0.000 | 0.2833 |
| 2 | large | oracle_gate_a-1.0 | 0.517 | 0.433 | 0.000 | 0.0 |
| 2 | large | real_single_pass_hard_gate_a-1.0 | 0.517 | 0.433 | 0.000 | 0.0 |
| 2 | large | random_single_pass_hard_gate_a-1.0 | 0.883 | 0.433 | 0.000 | 0.3666 |
| 2 | large | shuffled_single_pass_hard_gate_a-1.0 | 0.667 | 0.433 | 0.000 | 0.15 |
| 0 | hard_ood | base | 0.850 | 0.583 | 0.000 | 0.1167 |
| 0 | hard_ood | oracle_gate_a-1.0 | 0.733 | 0.583 | 0.000 | 0.0 |
| 0 | hard_ood | real_single_pass_hard_gate_a-1.0 | 0.733 | 0.583 | 0.000 | 0.0 |
| 0 | hard_ood | random_single_pass_hard_gate_a-1.0 | 0.950 | 0.583 | 0.000 | 0.2167 |
| 0 | hard_ood | shuffled_single_pass_hard_gate_a-1.0 | 0.767 | 0.583 | 0.000 | 0.0334 |
| 1 | hard_ood | base | 0.867 | 0.717 | 0.000 | 0.25 |
| 1 | hard_ood | oracle_gate_a-1.0 | 0.617 | 0.717 | 0.000 | 0.0 |
| 1 | hard_ood | real_single_pass_hard_gate_a-1.0 | 0.617 | 0.717 | 0.000 | 0.0 |
| 1 | hard_ood | random_single_pass_hard_gate_a-1.0 | 0.850 | 0.717 | 0.000 | 0.2333 |
| 1 | hard_ood | shuffled_single_pass_hard_gate_a-1.0 | 0.733 | 0.717 | 0.000 | 0.1166 |
| 2 | hard_ood | base | 0.883 | 0.617 | 0.000 | 0.2166 |
| 2 | hard_ood | oracle_gate_a-1.0 | 0.667 | 0.617 | 0.000 | 0.0 |
| 2 | hard_ood | real_single_pass_hard_gate_a-1.0 | 0.667 | 0.617 | 0.000 | 0.0 |
| 2 | hard_ood | random_single_pass_hard_gate_a-1.0 | 0.950 | 0.617 | 0.000 | 0.2833 |
| 2 | hard_ood | shuffled_single_pass_hard_gate_a-1.0 | 0.850 | 0.617 | 0.000 | 0.1833 |

## 2. Oracle Gap Summary

| Seed | Scenario | oracle H | hard H | random H | shuffled H | oracle_gap |
|---|---:|---:|---:|---:|---:|
| 0 | large | 0.7 | 0.7 | 0.9333 | 0.75 | 0.0 |
| 0 | hard_ood | 0.7333 | 0.7333 | 0.95 | 0.7667 | 0.0 |
| 1 | large | 0.5667 | 0.5667 | 0.8667 | 0.5833 | 0.0 |
| 1 | hard_ood | 0.6167 | 0.6167 | 0.85 | 0.7333 | 0.0 |
| 2 | large | 0.5167 | 0.5167 | 0.8833 | 0.6667 | 0.0 |
| 2 | hard_ood | 0.6667 | 0.6667 | 0.95 | 0.85 | 0.0 |

## 3. Control Separation

| Seed | Scenario | hard-random gap | hard-shuffled gap | random check |
|---|---:|---:|---:|
| 0 | large | 0.2333 | 0.05 | OK |
| 0 | hard_ood | 0.2167 | 0.0334 | OK |
| 1 | large | 0.3 | 0.0166 | OK |
| 1 | hard_ood | 0.2333 | 0.1166 | OK |
| 2 | large | 0.3666 | 0.15 | OK |
| 2 | hard_ood | 0.2833 | 0.1833 | OK |

## 4. Verdict

**IC4_M5_P0A_ROBUST** — Max oracle gap = 0.0000 ≤ 0.05.
