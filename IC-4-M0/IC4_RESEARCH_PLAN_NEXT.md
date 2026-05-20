# IC-4 Next-Stage Research Plan

Date: 2026-05-19
Scope: Next 1–3 weeks
Principle: Protect the trunk, build routing injection, open branches without scattering.

---

## Quick Reference Table

| Tier | Goal | Representative Work |
|---|---|---|
| **Anchors** | Irreversible conclusions | M3-v6 / M4 scoped robust / A2 imperfect probe / M7-Lv2 capability routing |
| **Near-term** | Next 1–2 weeks, highest payoff | P0: handoff consolidation / P1: cross-seed & cross-layer validation / P2: LoRA routing injection (M7-H) |
| **Branches** | Open terrain, not competing with trunk | Generation-specific routing / multi-behavior extension / propagation analysis / scale & architecture |

---

## Tier 1 — Anchors (Do Not Retreat Past These)

These four conclusions are the project's solid ground. Any future experiment that appears to contradict them must first be scrutinized for methodological error before these conclusions are revised.

### Anchor A: M3-v6 = Reference Mechanism

> `last_prompt_token` + logistic probe + hard gate + single-pass hook + `model.generate()`

This is the working prototype. All enhancements, validations, and branches should default to using it as the baseline comparison point.

**Reference files**:
- `src/run_m3_v6.py`
- `src/gate_steering_tool.py`
- `reports_m3_v6/IC4_M3_V6_SINGLE_PASS_GATE_REPORT.md`

**Reference result**:

| mode | H | C | UA |
|---|---:|---:|---:|
| base | 0.8667 | 0.6000 | 0.0000 |
| single_pass_hard_gate_a-1.0 | 0.6667 | 0.6000 | 0.0000 |

**What makes this an anchor**: This is the only mechanism in the project that unambiguously closes the loop: readable signal → selective intervention → correct forward-path attachment → oracle-level behavioral result. If something breaks, compare against this.

### Anchor B: M4 Scoped Robust

The reference mechanism is stable within a validated boundary:

- `seed = 0`
- `layer = 12`
- data scenarios: `standard`, `large`, `hard OOD`
- alpha values: `-0.8`, `-1.0`, `-1.2`

At `alpha = -1.0`, the gate matches oracle in all three evaluated data scenarios.

**What this does NOT yet mean**: cross-seed, cross-layer, cross-model, or cross-behavior robustness. Those are open questions (P1 in Near-term).

**Reference files**:
- `reports_m4_generalization/IC4_M4_GENERALIZATION_REPORT.md`
- `results_m4_generalization/sweep_matrix.csv`

### Anchor C: A2 Imperfect Probe

The mechanism is not dependent on a perfect probe, but current evidence favors hard gating:

- Hard gate remains oracle-level through `probe_acc ~ 0.997`
- Degradation begins around `probe_acc ~ 0.90`
- `soft_T0.1` is not a stable improvement over hard gate
- `soft_T0.3` is consistently too soft → **eliminated from mainline consideration**

**Reference files**:
- `results_branch_a2/aggregate_stats.csv`
- `reports_branch_a2/BRANCH_A2_DEGRADATION_REPORT.md`

### Anchor D: M7-Lv2 Capability Routing

The verification capability exists but is not default-routed into generation:

| Prompt | Sycophancy Rate | Δ from baseline |
|---|---|---|
| baseline (no prompt) | 0.6000 | — |
| fact_checker | 0.4000 | **-0.2000** |
| anti_sycophancy | 0.7500 | **+0.1500** |
| world_model_only | 0.5500 | -0.0500 |

**Four hard sub-conclusions**:

1. **Verification capability EXISTS but is LATENT.** The model can fact-check — it simply does not route verification into generation by default.
2. **Direct behavioral negation BACKFIRES.** Telling the model "don't be sycophantic" draws attention to the behavior it should suppress.
3. **S1 template (number confirmation) is a universal blind spot.** ~100% sycophantic across ALL prompt conditions. Even fact_checker cannot breach it.
4. **Template-dependent routing.** S5 (role title) is correctly rebutted at baseline; S4 (geography) is mixed. The latent verification path is not uniformly gated.

**What Anchor D means for the project narrative**:

Before M7-Lv2: *"Probes read a signal that steering cannot control because the circuit doesn't exist."*

After M7-Lv2: *"Probes read a signal that exists but is not default-routed into generation. Prompting can partially route it. The next problem is to make that routing structural."*

This is the transition from **capability absence** to **capability routing**.

**Reference files**:
- `results_m7/m7l_echo_cpu_report.txt`
- `src/run_m7l_echo_cpu.py`

---

## Tier 2 — Near-term (Next 1–2 Weeks, Ordered by Priority)

### P0: Handoff Consolidation

**Goal**: Make the project navigable by any future agent in under 10 minutes of reading.

**Specific steps**:

1. Keep terrain manual (`reports/IC4_PROJECT_TERRAIN_MANUAL.md`) as the authoritative map — update it whenever a new anchor is established
2. Maintain M7 clean report (`results_m7/M7_FINAL_REPORT_CLEAN.md`) and roadmap (`IC4_M7_ROADMAP_CLEAN.md`) as the primary M7 handoff pair
3. This document (`IC4_RESEARCH_PLAN_NEXT.md`) serves as the 1-page ultra-short compass — update priorities here, not narrative, when the plan shifts
4. Add a `README.md` at `IC-4-M0/` root that contains exactly: project identity (one sentence), link to terrain manual, link to this plan, and the quick reference table

**Success**: A new agent reading from the root can locate the correct reference mechanism, understand the anchors, and identify the next experiment to run — without reading anything else.

**Effort**: ~30 minutes of writing, no GPU.

---

### P1: Cross-Seed & Cross-Layer Validation

**Goal**: Extend the scoped robust boundary — the most conspicuous confirmatory gap in the trunk.

**Question 1 — Cross-seed**: Does the M3-v6 gate replicate at `seed = 1, 2`?

- **Method**: Run `src/run_m3_v6.py` with `--seed 1` and `--seed 2`. Compare H/C/UA against the reference (`seed=0`).
- **Success**: H/C/UA within 0.05 of reference at both seeds. Robustness boundary expands to "multi-seed."
- **Partial success**: One seed replicates, the other doesn't. Indicates seed sensitivity — investigate why.
- **Death**: Neither seed replicates. Gate is single-seed specific. This would be a significant scope limitation but does not invalidate the mechanism — it means the mechanism is seed-locked and deployment would require seed-specific calibration.

**Question 2 — Cross-layer**: Is `layer = 12` uniquely special, or do `layer = 11, 13` also work?

- **Method**: Run `src/run_m3_v6.py` at `--layer 11` and `--layer 13`. Compare H/C/UA against reference (`layer=12`).
- **Success**: All three layers match oracle within 0.05. Robustness boundary expands to "multi-layer."
- **Partial success**: Adjacent layers work but with degradation. Layer 12 is optimal but not unique.
- **Death**: Only layer 12 works. This would confirm a specific computational locus — valuable for mechanistic interpretation but limiting for deployment flexibility.

**Why P1 before P2**: This is boring but high-value. It determines whether the "scoped robust" label can be expanded without new mechanism design. If cross-seed fails, all future experiments should be designed with seed sensitivity in mind. If cross-layer fails, layer 12 becomes a hard design constraint.

**Effort**: ~1–2 hours CPU per condition. Total ~4–6 hours on CPU (4 conditions × 1h generation + evaluation).

---

### P2: LoRA Routing Injection (M7-H)

**Goal**: Test whether structural weight modification can wire latent verification into generation routing — the natural next step after M7-Lv2.

**Core question**: Can a LoRA adapter trained on contradiction-detection data achieve a sycophancy delta that matches or exceeds the prompt ceiling (delta = -0.20)?

**Method**:

1. Train a LoRA adapter (rank=8 or 16) on contradiction-detection data:
   - Input: context + sycophantic model response
   - Target: "This response contradicts the context because..."
2. Merge adapter weights into the base model
3. Evaluate on the same 4-template sycophancy test set used in M7-Lv2
4. Compare delta against the prompt ceiling:
   - delta < -0.20 → LoRA structural routing outperforms prompt activation
   - delta ≈ -0.20 → LoRA matches prompt activation but doesn't exceed it
   - delta ≈ 0 → routing cannot be learned at the weight level
5. **Critical sub-question**: Does LoRA routing generalize to the S1 blind spot?
   - If S1 sycophancy drops below 0.50 → structural routing surpasses prompting's capability boundary
   - If S1 remains at 1.0 → S1 is genuinely unreachable via current routing methods

**Why LoRA before full ECHO SFT**:

- LoRA is the minimum viable test of structural routing — if a low-rank adapter cannot wire the path, full SFT is unlikely to
- LoRA is faster to train, allowing faster iteration on routing hypotheses
- LoRA is the most deployable architecture — success here has direct practical implications
- M7-Lv2 already established the prompt ceiling; LoRA is the most direct weight-level comparison

**Success conditions**:

| Outcome | Interpretation | Next step |
|---|---|---|
| delta < -0.20, S1 < 0.50 | Structural routing definitively superior | Scale LoRA rank; test on harder OOD prompts |
| delta < -0.20, S1 ≈ 1.0 | Routing works but S1 is a hard barrier | Investigate S1 representation; try targeted S1 adapter |
| delta ≈ -0.20, S1 ≈ 1.0 | LoRA matches but doesn't exceed prompt | Try higher rank; compare against full ECHO SFT |
| delta ≈ 0 | Routing not learnable at LoRA scale | Try full ECHO SFT (M7-L); if that also fails, arch limit reached |

**Code**: `src/run_m7h_lora.py` (designed, untested)

**Effort**: ~1–2 hours on Colab T4. Requires GPU.

---

## Tier 3 — Branches (Open Terrain, Not Competing with Trunk)

These are explicitly marked as open terrain. They are not "todo items." They are directions that are scientifically valid but should not pull focus from Near-term priorities. An agent should only enter a branch if Near-term is blocked (e.g., waiting for GPU) or if a Near-term experiment produces an unexpected result that points toward a branch.

### Branch 1: Generation-Specific Routing

**Question**: Does the verification routing learned at prefill stage transfer to generation-stage dynamics?

**Why it matters**: M7-F showed that prefill vectors reverse effect during generation (delta = +0.40). If LoRA routing (P2) also fails during generation, the routing problem has a temporal component — verification must be wired into each generation step, not just the prefill state.

**Open sub-questions**:
- Do we need a separate adapter for generation-stage routing?
- Can a single adapter trained on prefill generalize to generation steps?
- Is there a hook site in the generation loop where routing is most effective?

**Relation to trunk**: This branch should only open AFTER P2 (LoRA routing) produces a prefill result. It is the natural extension of P2 into the temporal dimension.

---

### Branch 2: Multi-Behavior Extension

**Question**: Does the routing-disconnect pattern (Anchor D) generalize to other behaviors beyond sycophancy?

**Candidate behaviors**:
- Factuality hallucination (already validated with M3-v6 for unanswerable)
- Refusal / harmful compliance
- Tool-use caution
- Sycophancy in other domains (political alignment, preference mirroring)

**Why it matters**: If the routing-disconnect pattern is behavior-general, then Anchor D describes a fundamental architectural property, not a sycophancy-specific quirk. This would unify M3-v6 (hallucination) and M7 (sycophancy) under a single routing framework.

**Relation to trunk**: This branch is important for narrative unification but should not delay P1/P2. Open only after P2 produces a definitive routing result.

---

### Branch 3: Propagation Analysis

**Question**: Why does a steering intervention at layer 12 propagate correctly through the remaining layers?

**Open sub-questions**:
- What is the propagation profile of real vs. shuffled vs. random vectors?
- Does successful intervention correspond to a more stable internal propagation regime?
- Is layer 12 special because it sits at a propagation bottleneck, or because the signal at layer 12 has a particular geometric property?

**Why it matters**: This is the theoretical layer that explains WHY the mechanism works, not just THAT it works. Important for cross-architecture generalization.

**Relation to trunk**: Low urgency. Pursue when Near-term is blocked or when a mechanistic hypothesis needs testing.

---

### Branch 4: Scale & Architecture

**Question**: Do the anchors hold across model scales and architectures?

**Sub-questions**:
- M7-E: Does the routing-disconnect pattern (Anchor D) replicate at 1.5B?
- M7-E+: At 7B?
- Cross-architecture: Qwen vs. Llama vs. Mistral?

**Why it matters**: If the pattern is scale-invariant, the routing problem is architectural — it will affect all current models. If it disappears at larger scales, the 0.5B findings may be scale-specific and less practically relevant.

**Relation to trunk**: The 1.5B replication (M7-E) is the most actionable sub-question — it was Priority 3 in the original roadmap. Open after P2 produces a routing result, to avoid scattering compute across too many directions.

---

## Three-Phase Progression

This is the sequencing logic, not a schedule.

### Phase 1: Stabilize the Trunk

- Anchor documents are current and internally consistent
- P0 (handoff consolidation) is complete — any agent can navigate
- P1 (cross-seed / cross-layer) is complete — scoped robust boundary is mapped
- No new experiments are launched until this phase is clean

**Exit condition**: The four anchors are documented, the handoff files are navigable, and the robustness boundary is clearly stated (even if narrow).

---

### Phase 2: Build the Routing Mainline

- P2 (LoRA routing injection) is the primary experiment
- Compare against the prompt ceiling from Anchor D (delta = -0.20)
- Determine whether structural routing can match or exceed prompt-dependent routing
- If P2 succeeds: move to higher-rank or full ECHO SFT
- If P2 fails at LoRA scale: try full ECHO SFT (M7-L) as fallback
- If both fail: document the routing ceiling for 0.5B/896D

**Exit condition**: A definitive answer on whether weight-level routing injection is feasible. Either a working LoRA routing adapter exists, or the architecture's routing ceiling is documented.

---

### Phase 3: Extend to Other Behaviors and Scales

- Open Branches 1–4 as resources and results permit
- Priority within branches: generation routing (Branch 1) first (natural extension of P2), then multi-behavior (Branch 2) to unify the narrative, then scale (Branch 4) to validate generality, then propagation (Branch 3) as theoretical deepening
- Do NOT open all branches simultaneously — sequence them

**Exit condition**: Not defined yet. This phase should be replanned based on Phase 2 results.

---

## Design Rules (Negative Constraints)

These are things NOT to do during the plan window:

1. **Do not launch new mining experiments.** M7-A through M7-K exhausted the mining phase. The signal distribution, PCA profile, and ADD/REPLACE asymmetry are established. More patching or decomposition will not change the conclusions.
2. **Do not re-run M3-v6 variants without comparing against the reference.** The reference result is the baseline. Any variant that doesn't beat it is not progress.
3. **Do not treat prompt activation as a solution.** M7-Lv2 showed that capability-oriented prompts work and behavior-prohibition prompts backfire. This is an explanation-layer finding, not an engineering solution. The 40% residual sycophancy and S1 blind spot are unacceptable for deployment.
4. **Do not open more than one branch at a time.** Branches are for exploration when the trunk is stable, not for parallel development that competes for attention.
5. **Do not let branches overwrite anchors.** If a branch result appears to contradict an anchor, the branch methodology is suspect until proven otherwise.

---

## File Map (Key Documents for Navigation)

| Document | Purpose | When to read |
|---|---|---|
| `IC4_RESEARCH_PLAN_NEXT.md` (this file) | Compass — what to do next | First |
| `reports/IC4_PROJECT_TERRAIN_MANUAL.md` | Complete map — what we know | Second, for depth |
| `results_m7/M7_FINAL_REPORT_CLEAN.md` | M7 mechanism layer — clean version | When working on M7/routing |
| `IC4_M7_ROADMAP_CLEAN.md` | M7 experiment roadmap with priorities | When planning M7 experiments |
| `results_m7/m7l_echo_cpu_report.txt` | M7-Lv2 full findings | When referencing Anchor D |
| `reports_m3_v6/IC4_M3_V6_SINGLE_PASS_GATE_REPORT.md` | M3-v6 reference mechanism details | When working on trunk validation |
| `src/run_m3_v6.py` | Reference implementation | When running P1 or comparing against baseline |
| `src/run_m7h_lora.py` | LoRA routing injection (P2) | When running P2 |
| `src/run_m7l_echo.py` | Full ECHO SFT (P2 fallback) | When P2 LoRA fails |

---

## One-Sentence Plan Identity

> **IC-4 next stage: stabilize the trunk (anchors + cross-seed/layer), then test whether structural routing injection (LoRA) can wire latent verification capability into generation — because M7-Lv2 proved the capability exists but isn't default-routed.**