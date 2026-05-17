# IC-4-M3-v5: Artifact Decomposition for Valid Prefill Gate

> M3-v5 isolates the sources of the oracle gap and C anomaly observed in M3-v4b. 
> M3-v4b: soft prefill gate H=0.833, C=0.733 (anomaly), oracle H=0.667, C=0.600. 
> Question: Why does C rise to 0.733, and why doesn't H reach 0.667?

## 1. M3-v4b Key Gaps (Recap)

| Gap | M3-v4b Value | Baseline | Delta | Question |
|---|---|---|---|---|
| C anomaly | C=0.733 | base C=0.600 | dC=+0.133 | Why does probe gate increase C? |
| H gap vs oracle | H=0.833 | oracle H=0.667 | dH=+0.167 | Why can't prefill gate match oracle? |
| H gap vs open-loop | H=0.833 | open-loop H=0.667 | dH=+0.167 | Gate vs uniform application |

## 2. M3-v5 Design: Artifact Decomposition

**Hypothesis sources tested:**
1. Two-pass pipeline itself (vs single-pass)
2. Tiny residual alpha (~-0.02) from soft gate on answerable samples
3. Gate shape (soft sigmoid vs hard 0/1)
4. Two-pass open-loop vs single-pass open-loop (mechanics)

**New modes:**

| Mode | Description | Purpose |
|---|---|---|
| `two_pass_base_no_steering` | Two-pass, alpha=0, no steering | Isolates pipeline artifact on C/H |
| `two_pass_tiny_alpha_only` | Two-pass, alpha=-0.02 on ALL samples | Isolates tiny residual alpha effect |
| `two_pass_open_loop_full_alpha` | Two-pass, alpha=-1.0 on ALL samples | Compares two-pass vs single-pass steering |
| `soft_prefill_probe_gate` | Sigmoid gate (same as M3-v4b) | Baseline for gate comparison |
| `hard_prefill_probe_gate` | Discrete 0/1 gate (threshold 0.5) | Eliminates residual alpha on answerable |

| Parameter | Value |
|---|---|
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| Steering layer | [12] |
| alpha_max | [-1.0] |
| Probe representation | last_prompt_token |
| Gate steepness / threshold | 10.0 / 0.5 |
| Tiny alpha | -0.02 |
| Elapsed | 3693s (61.6 min) |

## 3. Probe Training

| Representation | Train Acc | CV Acc | AUC | N |
|---|---|---|---|---|
| last_prompt_token | 1.0000 | 1.0 | 1.0 | 15/15 |

## 4. Full Metrics Table

| Mode | H | C | UA | CA | Vector |
|---|---|---|---|---|---|
| base | 0.867 | 0.600 | 0.000 | 0.067 | none |
| prompt_only | 0.400 | 0.067 | 0.200 | 0.133 | none |
| steering_a-1.00 | 0.667 | 0.533 | 0.067 | 0.100 | steering |
| oracle_gate_a-1.0 | 0.667 | 0.600 | 0.000 | 0.100 | steering |
| two_pass_base_no_steering | 1.000 | 0.733 | 0.000 | 0.000 | none |
| two_pass_tiny_alpha-0.02 | 1.000 | 0.733 | 0.000 | 0.000 | steering |
| two_pass_open_loop_full_alpha-1.00 | 0.800 | 0.733 | 0.000 | 0.200 | steering |
| soft_prefill_probe_gate_a-1.0 | 0.833 | 0.733 | 0.000 | 0.167 | steering |
| random_soft_prefill_probe_gate_a-1.0 | 1.000 | 0.733 | 0.000 | 0.000 | random |
| shuffled_soft_prefill_probe_gate_a-1.0 | 1.000 | 0.733 | 0.000 | 0.000 | shuffled |
| hard_prefill_probe_gate_a-1.0 | 0.800 | 0.733 | 0.000 | 0.200 | steering |
| random_hard_prefill_probe_gate_a-1.0 | 1.000 | 0.733 | 0.000 | 0.000 | random |
| shuffled_hard_prefill_probe_gate_a-1.0 | 1.000 | 0.733 | 0.000 | 0.000 | shuffled |

## 5. Artifact Decomposition Results

### 5.1 Does two-pass pipeline itself elevate C?
- base C = 0.600
- two_pass_base_no_steering C = 0.733 (dC = +0.133)
- two_pass_base_no_steering H = 1.000 (base H = 0.867)
- **YES.** Two-pass pipeline itself shifts C by +0.133.

### 5.2 Does tiny residual alpha (~-0.02) elevate C?
- two_pass_base_no_steering C = 0.733
- two_pass_tiny_alpha_only C = 0.733 (dC = +0.000)
- two_pass_tiny_alpha_only H = 1.000
- **NO.** Tiny alpha alone does NOT significantly shift C.

### 5.3 Soft gate vs Hard gate
- soft gate: H=0.833, C=0.733
- hard gate: H=0.800, C=0.733
- dH = -0.033, dC = +0.000
- **Hard gate improves H.** Residual alpha and gate shape matter.

### 5.4 Two-pass open-loop vs single-pass open-loop
- single-pass open-loop: H=0.667, C=0.533 (steering_a-1.00)
- two-pass open-loop: H=0.800, C=0.733 (two_pass_open_loop_full_alpha-1.00)
- dH = +0.133 (two-pass is WORSE by 13.3 percentage points)
- **Two-pass mechanics significantly weaken steering.** Pipeline limits steering effectiveness by dH=+0.133.

## 6. Oracle Gap Attribution

| Source | Evidence | Magnitude | % of Gap |
|---|---|---|---|
| Pipeline C artifact | two_pass_base dC=+0.133 | 0.133 | N/A (C metric) |
| Tiny alpha C artifact | tiny dC=+0.000 | 0.000 | N/A |
| Gate shape (soft→hard) | dH=-0.033 | 0.033 | 19.8% |
| Two-pass mechanics (steering) | two-pass OL vs single OL dH=+0.133 | 0.133 | 79.6% |
| Total oracle gap (soft vs oracle) | H(0.833) - H(0.667) | 0.167 | 100% |

**Primary bottleneck for H:** two-pass steering mechanics limitation (accounts for ~80% of the oracle gap).

**Key insight:** Hard gate matches two-pass open-loop (both H=0.800), meaning even with
perfect oracle-level gate decisions, the two-pass pipeline ceiling is H=0.800 — NOT H=0.667.
Gate shape improvements can at most recover dH=-0.033. The remaining dH=+0.133 from
two-pass mechanics is fundamentally locked by the pipeline architecture.

## 7. Next Step Recommendation

**Primary: Investigate two-pass steering mechanics limitation.**
Gate shape optimization (hard gate) can recover at most dH=-0.033. The dominant bottleneck
(dH=+0.133) is the two-pass pipeline itself. Recommended directions:

1. **Single-pass prefill with delayed gate decision**: Extract probe features during
   prefill forward pass without doing a second prefill. This eliminates the two-pass
   pipeline entirely while keeping prefill-level gate timing.
   Example approach: memory-efficient hook that captures hidden states during the
   first (and only) prefill, computes probe, then applies steering if needed —
   all within one forward pass.

2. **Investigate why two-pass prefill weakens steering**: The fact that two-pass
   open-loop (H=0.800) underperforms single-pass open-loop (H=0.667) even with
   identical alpha=-1.0 is surprising. Possible factors:
   - KV cache from two-pass prefill has different key/value representations
   - Two consecutive forward passes may interact with residual connections differently
   - Model state between Pass 1 and Pass 2 may differ in subtle ways

3. **C anomaly is fully explained**: C=0.733 in all two-pass modes is a pipeline
   artifact. Not caused by gate shape or residual alpha. Two-pass prefill
   systematically shifts the model's behavior on answerable questions.

## 8. Verdict

**Primary Verdict: `IC4_M3_V5_STEERING_MECHANICS_LIMIT`**
**Secondary Verdict: `IC4_M3_V5_PIPELINE_ARTIFACT_CONFIRMED`**

**Reasoning:**

| Question | Answer | Evidence |
|---|---|---|
| C anomaly source? | Two-pass pipeline | All two-pass modes C=0.733 vs base C=0.600 |
| Tiny alpha causes C? | NO | two_pass_tiny = two_pass_base = C=0.733 |
| Hard gate beats soft gate? | YES, slightly | Hard H=0.800 < Soft H=0.833, dH=-0.033 |
| Two-pass weakens steering? | YES, significantly | Two-pass OL H=0.800 vs single-pass OL H=0.667, dH=+0.133 |
| Gate shape is the bottleneck? | NO | Hard gate = two-pass OL ceiling H=0.800 |
| Pipeline mechanics is bottleneck? | YES (~80% of gap) | dH=+0.133 from mechanics vs dH=-0.033 from gate |

**一句话结论:** M3-v4b 剩余 oracle gap 主要来自 **two-pass prefill pipeline 本身的 steering mechanics 限制** (dH=+0.133, 占 80%)，gate shape 优化最多只能恢复 dH=-0.033.

**下一步:** 应转向 steering mechanics 本身 — 用 single-pass prefill + delayed gate decision 替代 two-pass 架构，而非继续优化 gate 形状。

---

*IC-4-M3-v5: Artifact Decomposition for Valid Prefill Gate*
*Generated by run_m3_v5.py*