# M7 Clean Roadmap: From Readout-Control Paradox to Capability Routing

Date: 2026-05-19
Status: CPU mining complete. M7-Lv2 (CPU prompt activation) complete. GPU routing injection pending.
Reference: M3-v6 is the approved reference mechanism.

**Narrative shift**: M7-Lv2 moved the central question from "does verification capability exist?" to "how do we structurally route latent capability into generation?" The roadmap reflects this: priority is now on routing integration (LoRA), then capability ceiling (ECHO), then scale validation (1.5B).

---

## Completed Experiments (CPU)

| ID | Name | Core Finding | Verdict |
|---|---|---|---|
| M7-A | Component patching | MLP >> Attention | Distributed signal |
| M7-B | Dimension patching | No hot dimensions, K<200 = 0 | Bandwidth bottleneck |
| M7-C | PCA analysis | SNR peaks at PC1; ADD fails due to within-class variance | **Key mechanism** |
| M7-D | Base vs Instruct | Base=100% sycophancy | Pre-training prior, not RLHF |
| M7-F | Generation timing | Prefill vectors REVERSE effect during generation | Mode mismatch |
| M7-G | Manifold protection | K=20 gives 64% flip; K=896 needed for 100% | Monotonic scaling |
| M7-J | Cross-layer REPLACE | L20 alone = all 5 layers combined | Single-layer optimal |
| M7-K | Hadamard subspace | Random basis = zero effect; PCA = real structure | PCA is special |
| **M7-Lv2** | **Prompt activation (CPU)** | **fact_checker: delta=-0.20; anti_sycophancy: delta=+0.15** | **Latent capability exists** |

---

## Three Hypotheses — Final Verdict

| Hypothesis | Verdict | Key Evidence |
|---|---|---|
| Bandwidth bottleneck (896D overloaded) | CONFIRMED | M7-B/G/K: signal distributed across all dims |
| Circuit scarcity (no specialized heads) | CONFIRMED | M7-A: no dedicated anti-sycophancy component |
| Attractor collapse (sycophancy is low-energy) | CONFIRMED (corrected) | M7-D: pre-training prior, not RLHF artifact |

All three are true and mutually reinforcing.

---

## Core Mechanism (from M7-C)

```
ADD steering failure:
  hs_i_syc = mean_syc + noise_i
  hs_i'    = hs_i_syc + (mean_non - mean_syc) = mean_non + noise_i
  -> noise_i survives -> each sample still "looks sycophantic"

REPLACE success:
  hs_i_syc = mean_syc + noise_i
  hs_i'    = mean_non
  -> noise_i eliminated -> all samples collapse to non-sycophantic mean
```

Intervention must eliminate within-class variance, not just shift the mean.

---

## GPU-Stage Experiments (Code Ready — Routing Integration Phase)

**M7-Lv2 baseline**: fact_checker prompt achieves delta=-0.20. This is the prompt ceiling.
Each GPU experiment below tests whether weight-level routing can match or exceed this.

### Priority 1: M7-H — LoRA Routing Injection
- **File**: `src/run_m7h_lora.py`
- **Question**: Can LoRA structurally wire latent verification into generation routing?
- **Method**: Train LoRA adapter on contradiction-detection data; measure sycophancy delta post-merge
- **Why first**: M7-Lv2 showed routing is prompt-dependent and fragile (S1 blind spot untouched).
  LoRA directly tests "can we structuralize this routing." If delta < -0.20, routing is learnable
  at the weight level. Delta > -0.20 = structural routing outperforms prompting.
- **Success**: delta < -0.20 on held-out sycophancy prompts
- **Death**: delta ~ 0 — routing cannot be learned even with weight modification

### Priority 2: M7-L — ECHO Full Verification Training
- **File**: `src/run_m7l_echo.py`
- **Question**: Can full SFT on contradiction-detection exceed the prompt ceiling AND generalize to S1?
- **Method**: 500+ steps of SFT on contradiction detection; measure sycophancy on all 4 templates
- **Why second**: Tests the upper bound of weight-level routing. Critical question: does SFT
  generalize to the S1 blind spot (number confirmation, universally sycophantic across all prompts)?
  If S1 still sycophantic after 500+ steps, the routing problem has a template-specific hardness gradient.
- **Success**: delta < -0.20 AND S1 sycophancy < 0.50
- **Death**: delta ~ 0 OR S1 sycophancy = 1.0 — routing ceiling reached

### Priority 3: M7-E — 1.5B Cross-Model Replication
- **File**: Needs adaptation of M6/M7 pipeline for 1.5B
- **Question**: Does the latent-verification / routing-disconnect pattern persist at 1.5B?
- **Method**: Replicate M7-Lv2 prompt test at 1.5B; probe PCA profile
- **Why third**: If 1.5B shows same pattern (fact_checker helps, anti_sycophancy backfires),
  the routing problem is architectural, not scale-bound. If 1.5B default-routes verification
  better, routing integration success may be scale-dependent.
- **Success**: Confirms or refutes scale-invariance of routing disconnect
- **Death**: Qualitative difference renders 0.5B findings irrelevant at practical scales

---

## Design Decisions

| Decision | Rationale |
|---|---|
| Don't use ADD steering | Preserves within-class variance (M7-C) |
| Don't patch <200 dims | Signal distributed across all 896D (M7-B/G) |
| Don't intervene during generation | Prefill vectors reverse effect (M7-F) |
| Use L20 as intervention layer | Optimal single-layer (M7-J) |
| Use PCA basis, not random basis | PCA captures real structure (M7-K) |
| Use capability prompts, not behavior-prohibition | Anti-sycophancy backfires; fact_checker activates latent path (M7-Lv2) |
| Target routing integration, not more mining | Capability exists but is disconnected from generation (M7-Lv2) |

---

## File Map

| What | Where |
|---|---|
| Full M7 report (detailed) | `results_m7/M7_FINAL_REPORT.md` |
| M7 clean report (with M7-Lv2) | `results_m7/M7_FINAL_REPORT_CLEAN.md` |
| This clean roadmap | `IC4_M7_ROADMAP_CLEAN.md` |
| Next-stage research plan (3-tier) | `IC4_RESEARCH_PLAN_NEXT.md` |
| M7-Lv2 CPU report | `results_m7/m7l_echo_cpu_report.txt` |
| M7-Lv2 CPU code | `src/run_m7l_echo_cpu.py` |
| Project terrain manual (with M7-Lv2) | `reports/IC4_PROJECT_TERRAIN_MANUAL.md` |
| M7-A code | `src/run_m7a_component_patch.py` |
| M7-B code | `src/run_m7b_dim_patch.py` |
| M7-C code | `src/run_m7c_pca.py` |
| M7-D code | `src/run_m7d_attractor.py` |
| M7-F code | `src/run_m7f_timing.py` |
| M7-G code | `src/run_m7g_manifold_protect.py` |
| M7-J code | `src/run_m7j_cross_layer.py` |
| M7-K code | `src/run_m7k_hadamard.py` |
| M7-H code (GPU, Priority 1) | `src/run_m7h_lora.py` |
| M7-L code (GPU, Priority 2) | `src/run_m7l_echo.py` |
| Individual reports | `results_m7/m7*.txt` |

---

## One-Sentence Project State

M6/M7 has evolved through two phases: (1) mapping the boundary of what 0.5B/896D
can do without structured injection — the signal is real but fully distributed and
requires variance collapse (not mean shift) to control; (2) M7-Lv2 discovered that
verification capability exists but is not default-routed into generation (fact_checker
prompt: delta=-0.20, anti_sycophancy backfire: delta=+0.15). The next step is GPU-based
routing injection (LoRA/ECHO) to structurally connect latent capability to generation.