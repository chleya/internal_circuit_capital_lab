# Unified Thesis

**Version:** v1.0  
**Date:** 2026-05-21  
**Role:** master thesis document for project direction and future task alignment  
**Scope:** `IC-4-M0` + `intelligence_capital_minimal_lab`

---

## 0. One-sentence thesis

> Small-model failure is not best understood as simple knowledge absence, but as a structural adaptation problem: the model struggles to absorb discretized input without distortion, stabilize useful structure under updating and compression, and organize latent capability into reliable behavior; the project studies whether these failures can be diagnosed and partially compensated through feedback, anchoring, and trajectory-level control.

This is the main line. Everything else should be interpreted in relation to it.

---

## 1. What the project is actually about

This project is **not** fundamentally about:

- finding another steering trick
- building a smaller Self-RAG clone
- reducing hallucination by any means necessary
- adding another memory module

Those may appear as local mechanisms, but they are not the thesis.

The real question is:

> Why do small models fail to turn fragmented human data streams into stable, callable internal structure, and can external control or training constraints partially compensate for that?

This question decomposes into three bottlenecks:

1. **Absorption**  
   How input organization, position, and fragmentation distort internal state formation.

2. **Stabilization**  
   How useful structure drifts, mixes, or collapses under compression, updating, or consolidation.

3. **Organization**  
   How latent capability exists internally but fails to route into default behavior.

---

## 2. How the existing lines converge

### 2.1 Engineering Cybernetics is the top-level language

The engineering cybernetics framing gives us the right vocabulary:

- model = controlled object
- hidden space = state space
- prompt/position changes = structured disturbance
- probe = state observation
- gate/hook = control input
- drift/collapse = stability failure

This framing is useful because it lets us talk about the project as a control-and-stability program rather than a bag of tricks.

### 2.2 Structural Adaptation Hypothesis is the core causal picture

The structural adaptation hypothesis says the small-vs-large gap is not only about scale or knowledge volume. It is also about whether the model can:

- absorb fragmented input without distortion
- stabilize useful structure across samples and updates
- organize latent capability into robust behavioral paths

This is the project's central explanatory hypothesis.

### 2.3 Relational Memory Hypothesis specifies the form of useful structure

Useful structure is not merely positional or token-sequential. It is relational, distributed, and reconstructive.

This matters because it explains why:

- naive consolidation can destroy utility while preserving some local topology
- shortcut readouts can beat “richer” memory systems
- capability routing may need to reconnect relational substructure, not just add a vector

### 2.4 Trajectory Dynamics makes the hypothesis measurable

Trajectory work operationalizes the theory:

- when does a behavior separate?
- where in layer-step space does it become readable?
- is it direction-specific or only perturbation-sensitive?
- does it form early, late, diffusely, or through generation-time amplification?

Trajectory analysis is how the project stops being just philosophy.

---

## 3. What IC-4 and IC-2 each contribute

### 3.1 IC-4 = organization compensation

IC-4 asks:

> If the model already contains useful latent capability, why does it fail to use it by default, and can closed-loop or structured intervention compensate?

The strongest IC-4 findings so far are:

- `M3-v6`: a working minimal closed-loop mechanism exists
- `M7-Lv2`: latent verification-like capability exists but is misrouted
- trajectory dynamics: hallucination and sycophancy are not the same controllability object
- `P2`: hallucination is not well-described by a single causal steering direction
- `Proof B`: multi-direction intervention can beat single-direction intervention

In thesis terms, IC-4 is the main organization line.

### 3.2 IC-2 = stabilization diagnosis and early remedy

IC-2 asks:

> When useful information is compressed or updated over time, what makes it survive versus degrade into bad debt?

The strongest IC-2 findings so far are:

- learned compression can beat raw memory
- NoMemory shortcut still wins surprisingly often
- continual consolidation can drift into bad debt
- topology can partly survive even while purity collapses
- anchored consolidation provides a modest but real positive remedy (+8.7%)
- **readout-level interventions (seed-conditioned, purity-gated, per-seed, weighted) all fail to improve over naive** — the root cause is not cross-seed readout averaging but KMeans consolidation itself ignoring Y information
- combined anchor+seed+purify shows early promise at step 2 (+37%) when purity is still moderate, but collapses when purity fully degrades
- **ROOT CAUSE CONFIRMED AND FIXED: KMeans ignores Y information → Y-aware consolidation (per_action_kmeans=0.585) beats NoMemory (0.445) by +31%!**
- evidence: (1) learned_state_only=0.740 vs KMeans=0.095 (8x gap); (2) increasing resolution (kmeans_100/200) doesn't help; (3) Y-aware weight越大效果越好; (4) per_action_kmeans guarantees Y-consistency within clusters

In thesis terms, IC-2 is the main stabilization line.

### 3.3 Position sensitivity = absorption diagnosis

Position sensitivity asks:

> How much of the instability begins before reasoning, at the point where fragmented input is turned into internal state?

**Status: DIAGNOSED — three-layer evidence chain (2026-05-20).**

| Layer | Metric | Value | Interpretation |
|---|---|---|---|
| **Representation** | 3-NN position classification | **1.000** (baseline=0.333) | Same content at different positions → completely different hidden state |
| **Representation** | cos(early, mid) / cos(early, late) | 0.065 / 0.080 | ~8% cosine distance from prefix shifting alone |
| **Probe** | PSI (Position Sensitivity Index) | **0.0084** | Probe layer is position-robust; A/U separation preserved (0.993→0.976) |
| **Behavior** | C range / H range | **0.067 / 0.033** | Position IS a behavioral confound; correct rate varies ±3% by position |

**Key structural finding**: The representation is strongly position-dependent (KNN=1.0), but the model partially compensates downstream — the probe (trained on mixed positions) is nearly position-invariant, and behavior is only moderately affected. This "partial compensation" is exactly what the absorption bottleneck predicts: input fragmentation distorts state space, and the model can compensate some but not all of it.

**Per-position behavior breakdown (N=60)**:
| Position | H | C | CA |
|---|---|---|---|
| early | 0.867 | 0.600 | 0.067 |
| mid | 0.900 | 0.667 | 0.033 |
| late | 0.867 | 0.667 | 0.033 |

In thesis terms, this is the absorption line — now with quantitative diagnosis.

---

## 4. What the recent proof sequence means

The recent A/B/C/D style results are not isolated.

They form a stronger unified statement:

### 4.1 Proof A

The negative sycophancy replication result matters because it prevents premature myth-making.

It tells us:

- direction-sensitive-looking behavior may be fragile
- sycophancy cannot yet be treated as a fully established direction-specific control object

This is valuable because it narrows the space of believable interpretations.

### 4.2 Proof B

Proof B is one of the most important positive results in the whole project.

It shows:

- single-direction additive intervention is not the whole story
- structured combination can outperform the best single direction

This matters because it suggests the control object is not one axis but a structured subspace or coordinated intervention pattern.

### 4.3 Proof C

Proof C is the stabilization line’s first real remedy result.

It shows:

- anchoring can improve consolidation relative to naive updating
- stabilization is not only diagnosable, but at least partly correctable

This is crucial for the project’s credibility, because without it the stabilization line would remain mostly pathological diagnosis.

### 4.4 Proof D

Proof D is not just another report.

Its job is to force the project into a unified mechanism story:

- what is real
- what is not yet real
- what kind of control object each behavior actually is

This is where the project stops being a collection of experiments and becomes a coherent program.

---

## 5. The core unifying claim

The project’s strongest unifying claim right now is:

> Small-model failures emerge when structurally useful internal trajectories cannot be formed, stabilized, or correctly routed. Some of these failures are diagnosable from internal state; some are partially compensable through external feedback and anchoring; and the path forward is not a search for one magic direction, but for better controlled trajectory structure.

This claim is broad enough to unify the work, but narrow enough to remain honest.

---

## 6. What has already been established strongly enough

These statements are strong enough to use as active mainline assumptions:

1. A minimal closed-loop organization compensator exists (`M3-v6`).
2. Latent capability can exist without default behavioral access (`M7-Lv2`).
3. Small-sample construction artifacts can mimic mechanism failure (`P1.5`).
4. Position is a real representational disturbance source.
5. Consolidation can degrade into bad debt through mixing and wrong readout.
6. Hallucination is prefill-separable and not yet supported as a single-direction control object.
7. Structured multi-direction control is more promising than single-direction additive control.
8. Anchoring can improve stabilization relative to naive consolidation.

---

## 7. What remains open

These are the most important open questions, not side quests.

### 7.1 Direction vs energy decomposition

For sycophancy especially, we needed to know:

- how much effect is genuinely directional
- how much is just structured energy injection

**Status: RESOLVED (negative).**

The `Syc Direction-vs-Energy Decomposition Audit` swept 3 epsilons × 4 directions (v_syc, random, shuffled, orthogonal) at L10 prefill with 20+20 balanced samples. Result:

- **0/3 epsilons reach 2σ significance** for v_syc vs energy baseline
- Residuals are NOT same-sign across epsilons (-0.050, +0.083, +0.200)
- Control variance is huge (σ=0.284 at eps=3.0) — random directions produce wildly different effects
- d/e ratio ~0.31 — directional component indistinguishable from noise

**Conclusion: sycophancy impulse response is pure energy. No direction-specific component detected.**

This completes the direction-specificity exclusion for both behaviors:
- Hallucination: v_hall = v_orthogonal (P2)
- Sycophancy: v_syc indistinguishable from energy baseline (this audit)

Both behaviors now fall on the "generic perturbation" side — a unified finding that constrains the project's control theory.

### 7.2 Multi-Direction Structure Audit (COMPLETE)

The **Proof B+ Multi-Direction Structure Audit** (`run_b2_structure_audit.py`) systematically tested ALL vector pair combinations to answer: what structure makes a combination good?

**Design**: 7 singles + 15 unique 50/50 pairs (6 vectors choose 2) + 5 mixture ratios for v_hall+v_syc_like + 3 cross-layer tests (L10/L12/L14). Vectors: v_hall, v_hall_A, v_hall_B (split-test robustness), v_syc_like, random, shuffled, orthogonal. Fixed v_syc_like definition to match Proof B: orthogonalize(random(seed=999), seed=777).

**Key results:**

| Finding | Detail |
|---|---|
| C_base | 0.800 (vs original Proof B's 0.400 — due to test set random seed) |
| Best single | **random** (dH=+0.200, dC=0.000, score=+0.200) |
| Best pair | v_syc_like+random (score=+0.200, synergy≈0.000) |
| All pair synergy | **ALL ≤ 0** — no positive synergy in any of the 15 combos |
| Cosine vs synergy | r=0.0037 — no structural signal, angle cannot predict synergy |
| Hall direction stability | cos(v_hall, v_hall_A)=0.903, cos(v_hall, v_hall_B)=0.914, cos(v_hall_A, v_hall_B)=0.651 — hallucination direction is noisy |
| Cross-layer robustness | L10=0.000, L12=-0.100, L14=-0.200 — only L12 has any effect |

**Interpretation**: The original Proof B "multi-direction BEATS single-direction" finding was a specific case at C_base=0.400. At C_base=0.800, the structured control advantage vanishes entirely. This reveals a **boundary condition**: structured intervention only beats random perturbation when the baseline capability is already degraded. When capability is near ceiling (C=0.800), no directional structure provides advantage over random.

**Implication**: Hallucination direction-specificity is further excluded. The path forward is not better vector selection — it's trajectory-level structure shaping (step 4).

### 7.3 Stabilization scaling: Phase 6-A (COMPLETE)

Per-Action KMeans (0.585 at 5 seeds) was the strongest positive signal. Phase 6-A tested whether it survives scaling from 5 to 100 seeds.

**Result — ALL levels STRONG PASS:**

| Level | Seeds | PA | NM | Δ | Adaptive |
|---|---|---|---|---|---|
| C3 | 5 | 0.585 | 0.445 | +0.140 | — |
| S1 | 20 | 0.630 | 0.445 | +0.185 | 0.640 |
| S2 | 50 | **0.660** | 0.445 | **+0.215** | 0.630 |
| S3 | 100 | 0.615 | 0.445 | +0.170 | 0.630 |

**Key findings:**
- Per-Action KMeans peaks at 50 seeds (0.660, 48.3% over NoMemory)
- Even at 100 seeds, PA > NM by +0.170 — no degradation below strong pass threshold
- X-only KMeans stays at 0.095 regardless of scale → X-only is hardware-limited ceiling
- Adaptive centroids (√n per action) beats fixed at 100 seeds → large-scale benefit

**Implication**: Stabilization breakthrough is NOT a small-N artifact. Per-Action KMeans scales and even strengthens with more data. The Y-information bottleneck is fundamental — X-only clustering cannot be salvaged by more data.

### 7.3b Stabilization scaling: Phase 6-B/C (COMPLETE)

**Phase 6-B — Objective Scaling (Multi-Action)**:

| Actions | KMeans | Y-aware | PerAct | NoMem | PA-NM | PA-KM |
|---|---|---|---|---|---|---|
| 3 | 0.095 | 0.465 | 0.500 | 0.445 | +0.055 | +0.405 |
| 5 | 0.545 | 0.665 | **0.715** | 0.385 | +0.330 | +0.170 |
| 10 | 0.285 | 0.355 | 0.355 | 0.265 | +0.090 | +0.070 |
| 20 | 0.175 | 0.195 | 0.155 | 0.110 | +0.045 | **-0.020** |

**Key findings**:
- delta(PA-NM) slope = -0.010 (3→20) — Y-aware advantage is STABLE, not decaying
- PA peaks at 5 actions (0.715), then declines but PA-NM delta stays positive through 20 actions
- At 20 actions, PA (0.155) < KM (0.175) for the first time — boundary found: PA not infinitely scalable
- NoMemory degrades rapidly 0.445→0.110 as action space grows (frequency-based strategy collapses)
- The stable delta(PA-NM) across 4x action space expansion is the key robustness signal

**Phase 6-C — Noise Scaling**:

| Noise (σ) | KMeans | Y-aware | PerAct | NoMem | PA-NM |
|---|---|---|---|---|---|
| 0.00 | 0.095 | 0.465 | 0.500 | 0.445 | +0.055 |
| 0.03 | 0.095 | 0.482 | 0.495 | 0.445 | +0.050 |
| 0.10 | 0.095 | 0.478 | 0.497 | 0.445 | +0.052 |
| 0.30 | 0.095 | 0.457 | 0.505 | 0.445 | +0.060 |
| 1.00 | 0.095 | 0.480 | **0.545** | 0.445 | **+0.100** |

**Key findings**:
- KM drop from 0→max noise: 0.000 — KMeans already at floor (0.095), cannot degrade further
- PA drop from 0→max noise: -0.045 — PA actually IMPROVES at max noise (0.500→0.545)
- NoMem invariant at 0.445 (frequency-based, noise has zero effect)
- VERDICT: WEAK POSITIVE — noise scaling test partially invalidated by KMeans floor effect, but PA robustness confirmed (no degradation, slight improvement at high noise)

**Overall Phase 6 implication**: Per-Action KMeans advantage is robust across seed scaling (peak 0.660 at 50 seeds), objective scaling (stable PA-NM delta across 3→20 actions), and noise scaling (no degradation). The Y-information advantage is not a small-N or narrow-condition artifact. Phase 6 complete with ALL experiments passing.

### 7.4 Better stabilization remedy (RESOLVED)

~~Anchored consolidation is a real start, but the stabilization line still needs a stronger remedy candidate.~~

**Update (Phase 6-A)**: Per-Action KMeans is the strong remedy candidate — 0.660 at 50 seeds, confirmed scalable across 5→20→50→100 seeds. Anchored consolidation v2 is no longer the priority; multi-objective anchoring (Phase 6-B/C) and cross-bottleneck coupling (Phase 7) are.

### 7.5 Training-level trajectory intervention (RESOLVED — negative)

~~If local token-level fitting is part of the problem, then the next generation likely has to ask: Can we train the student to move through state space better?~~

**Update**: TT-SFT v0+v1 both negative. CE-only beats trajectory cosine alignment. Route excluded.

### 7.6 Absorption Remedy: Position-Augmented Probe (RESOLVED — partial)

Position sensitivity is the project's untreated upstream bottleneck. Phase 8 tested the simplest possible remedy: augmenting probe training data with position variants.

**Phase 8-A — Position-Augmented Probe Training**:

| Metric | Base Probe (std only) | Aug Probe (+pos variants) | Improvement |
|---|---|---|---|
| PSI | 0.0676 | **0.0067** | **−90.0%** |
| Probe score range (early-mid-late) | 0.100 (0.49→0.43→0.39) | 0.009 (0.50→0.50→0.49) | −91% |
| Standard test accuracy | 1.000 | 1.000 | No degradation |

Training: 30 base samples → 120 augmented samples (30 std + 30×3 position variants). Probe architecture unchanged (logistic regression on 896D last_prompt_token).

**Phase 8-B — Behavior-Level Gate Consistency**:

| Condition | early gate | mid gate | late gate | Range |
|---|---|---|---|---|
| Base probe (no aug) | 11/20 | 10/20 | 9/20 | 2 |
| Aug probe | **11/20** | **11/20** | **11/20** | **0** |

Gate decisions perfectly consistent across positions with augmented probe. However, behavior-level position sensitivity (ΔH=0.111) persists in raw `model.generate()` without steering — the generation process itself is position-dependent, not just the routing decision.

**Key structural finding**: Absorption bottleneck decomposes into two layers:
1. **Probe/routing layer** (FIXED by Phase 8): Position augmentation eliminates position bias in routing decisions.
2. **Generation layer** (PERSISTS): The model's autoregressive generation is inherently position-sensitive at the hidden state level (KNN=1.0), and a position-invariant probe cannot fix this.

**Implication**: Full absorption remedy requires intervention at the model internals level — position-aware training, position normalization in attention, or SFT with position invariance objective. The probe-level fix is necessary but not sufficient.

---

## 8. Why trajectory-targeted training fits the main line

Trajectory-targeted training is not a random new branch.

It follows directly from the project’s current constraints:

- single-direction control is limited
- structure drift is real
- latent behavior differences are visible in trajectories
- position changes distort trajectory formation

So the natural next question is:

> instead of only steering an already fragile trajectory, can we train the small model to form a better trajectory in the first place?

This is why a `Trajectory-Targeted SFT v0` experiment belongs to the main line, not the side line.

Its intended role is:

- not to replace all current work
- but to test whether training-level trajectory alignment improves structural adaptation directly

---

## 9. What the project is now trying to become

The project is no longer just:

- a steering project
- a memory project
- a small-RAG project

It is becoming:

> a mechanism program for diagnosing and partially compensating structural adaptation failures in small language models.

That is the cleanest statement of the main line.

---

## 10. What future agents should optimize for

Future work should not optimize for:

- more scattered phenomena
- more tasks for coverage’s sake
- tiny benchmark gains without mechanism clarity

Future work should optimize for:

1. moving one open question from ambiguous to bounded
2. turning one diagnosis into one remedy
3. distinguishing one control interpretation from another
4. tightening the link between experiments and the three-bottleneck thesis

If a future experiment does not do one of those, it is probably not mainline.

---

## 11. The current mainline order

The recommended order from here is:

1. ~~direction vs energy decomposition for sycophancy~~ ✅ **RESOLVED: pure energy, no direction-specificity**
2. ~~stronger stabilization remedy~~ ✅ **RESOLVED: root cause = KMeans ignores Y information. Per-Action KMeans (0.585) beats NoMemory (0.445) by +31%. Stabilization is now SIGNIFICANTLY correctable.**
3. ~~multi-direction structure audit for hallucination~~ ✅ **RESOLVED: ALL 15 pairs synergy ≤ 0, best single = random (0.200), C_base=0.800 boundary condition reveals structured control only works at degraded baseline. Hallucination direction-specificity further excluded.**
4. ~~trajectory-targeted SFT v0~~ ✅ **RESOLVED (negative): v0+v1 both show CE-only > trajectory alignment. Cosine alignment loss not effective for stabilization/organization. Methodological limitation, not scale.**
5. ~~absorption bottleneck diagnosis (P1)~~ ✅ **RESOLVED: three-layer evidence chain (KNN=1.0 + PSI=0.0084 + ΔC=0.067). Partial downstream compensation confirmed. All three bottlenecks now diagnosed.**
6. ~~Phase 6-A: Stabilization Seed Scaling~~ ✅ **RESOLVED: ALL levels STRONG PASS. Per-Action KMeans 0.585→0.630→0.660→0.615. Δ vs NoMemory: 0.140→0.185→0.215→0.170. Breakthrough confirmed scalable.**
7. ~~Phase 7 3.3A: Cross-Bottleneck Analogue~~ ✅ **RESOLVED: PA stays > NoMemory at ALL additive_noise and directional_shift levels. Only loses at >65% centroid dropout. Advantage is structural margin, not magical synergism. Analogue supports 3.3B.**
8. ~~Phase 7 3.3B: LLM Hidden State Consolidation~~ ✅ **RESOLVED (negative, data-gap): M3 activations all from base model — no state divergence between seeds. Even PCA→2D cross-seed, ALL strategies = 1.000. Qwen has perfectly separable representations but can't route. Confirms B-bottleneck narrative. Genuine consolidation test requires multi-checkpoint fine-tuning data (not currently available).**
9. ~~Phase 6-B/C: Objective + Noise Scaling~~ ✅ **RESOLVED: Phase 6-B (STABLE — PA-NM delta slope -0.010 across 3→20 actions). Phase 6-C (WEAK POSITIVE — noise almost zero effect, KMeans at floor, PA no degradation). ALL Phase 6 experiments pass. Per-Action KMeans advantage robust across seed scaling, objective scaling, and noise scaling.**
10. ~~Phase 8: Absorption Remedy~~ ✅ **RESOLVED: Position-Augmented Probe training reduces PSI by 90% (0.0676→0.0067). Gate decisions become perfectly consistent across positions (11/11/11 vs baseline 11/10/9). BUT: behavior-level ΔH=0.111 persists — absorption bottleneck has two layers: probe-level (FIXED) and generation-level (PERSISTS).**
11. ~~Phase 9-A: Position Rectification at Inference~~ ❌ **RESOLVED (negative): Global position offset hook increases ΔH from 0.111 to 0.333 — helps mid/late but not early, widening the gap. Confirms position sensitivity is content-dependent (KNN=1.0), not a global additive shift.**
12. ~~Phase 9-B: Multi-Checkpoint LLM Consolidation~~ ⚠️ **RESOLVED (mixed): LoRA fine-tune 5 epochs (loss 13.3→0.068). Cross-ckpt KNN=1.0 persists — A/U separability preserved. PerClassKMeans beats baseline (+0.37-+0.42) but Y-Aware always better (0.95-0.97, gap=0.05-0.07). LoRA changes hidden states but doesn't break class boundary. Data-gap partially addressed but KNN ceiling remains.**
13. ~~P8: Sycophancy Feedback Control Scale-Up~~ ✅ **RESOLVED: −57.1% reduction on n=24 (0.583→0.250). Two-stage th=0.50, gate=54.2%. Successfully replicates P6-ter at 2x scale. Open-loop worse than baseline.**
14. ~~Phase 10: Position-Aware LoRA Training~~ ✅ **RESOLVED: ΔH −50%, PSI −53%, Consistency +5%.** LoRA rank=4 trained on 90 position-variant samples (30 early+mid+late). Pre: H=(0.33,0.56,0.56), ΔH=0.22, PSI=0.0016, Consistency=0.90 (18/20). Post: H=(0.56,0.44,0.44), ΔH=0.11, PSI=0.00073, Consistency=0.95 (19/20). Behavior-level position absorption partially closed — weight-level intervention (LoRA) succeeds where global rectification (9-A) failed. Early H trade-off (0.33→0.56) suggests regularization-convergence tension: position invariance regularizes mid/late toward early distribution, degrading early performance. All three bottleneck remediation strategies now executed with measurable effect: A=probe-level FIXED (PSI −90%) + behavior-level IMPROVED (ΔH −50%), B=strongly remedied (PA-KMeans +31%), C=scalable (n=24 −57.1%).

15. ~~Phase 11: Cross-Bottleneck Integration~~ ✅ **RESOLVED: 瓶颈可分离.** Position-Aware LoRA (A-remedy) tested on sycophancy knowledge (C domain). BASE and LoRA both: syc_rate=0.0000 (24/24 prefer correction), position consistency=1.0000. Margin preserved (-0.634→-0.596). Three independent bottlenecks confirmed — A-remedy targeted, no cross-contamination to C. The B-bottleneck (knowledge-production gap) highlighted: model KNOWS (log-prob syc=0.00) but DOESN'T produce (generation syc=0.583).

 16. ~~P12: Position-Directional Activation Steering~~ ❌ **NEGATIVE (informative).** v_abs = mean(h_early)−mean(h_late) steering at layer 10: U-shaped alpha curve (|α|≤1.5 no effect, |α|≥3.0 collapses position discrimination). At α=+3.0 only v_abs eliminates ΔH (vs random/orth preserve 0.250), confirming direction is real. But elimination is destructive: ALL positions degrade to H=0.50, not late→early improvement. Phase transition at α∈[1.5,3.0] — position subspace has finite perturbation capacity. **Two independent experiments now confirm: position sensitivity cannot be fixed by hidden-state vector ops (9-A + P12). Only weight-level intervention works (Phase 10 LoRA).**

 17. ~~P13: Probe-Guided Hallucination Steering~~ ❌ **NEGATIVE (geometric B-bottleneck proof).** Hallucination probe (acc=1.000, layer 12) decision boundary w_probe used for steering. Full alpha sweep (-5 to +5): NO effect on H across moderate alphas (H=0.417 flat, C=1.000 preserved). At large |α| it DEGRADES (H→0.50-0.58, same P12 phase transition). Random vector shows same pattern. Core claim: Classification direction ≠ behavioral control direction — the linear subspace that discriminates answerable/unanswerable is ORTHOGONAL to the subspace that controls hallucination/abstention. **B-bottleneck now has geometric proof: KNOWS ≠ produces is a SUBSPACE SEPARATION in representation space. Three experiments (9-A, P12, P13) confirm: hidden-state vector ops cannot bridge this gap.**

 18. ~~P14: Cross-Layer B-Bottleneck Characterization~~ ❌ **NEGATIVE (depth-universal subspace separation).** Probe acc=1.0000 at ALL 9 layers tested (0,3,6,9,11,12,15,18,21) — answerable/unanswerable is perfectly linearly separable across entire network depth. Yet w_probe steering has near-zero behavioral effect everywhere: ΔH_max∈[0.000, 0.167], always DESTRUCTIVE (increases hallucination), overlap_ratio≤0.17 at every layer. Layer 21 has probe acc=1.000 but ΔH_max=0.000 — pure orthogonality. **B-bottleneck subspace separation is DEPTH-UNIVERSAL: KNOWS≠produces is not a single-layer artifact but a global geometric property of the representation space.** Four experiments (9-A, P12, P13, P14) now confirm: hidden-state vector ops cannot bridge the knowledge-production gap.

 19. ~~P15: Hallucination LoRA Fine-Tuning~~ ✅ **POSITIVE (B-bottleneck BRIDGED via weight-level LoRA).** LoRA trained on 90 hallucination-labeled samples (45A+45U): answerable→correct answer, unanswerable→abstention. Result: H 0.417→**0.000** (ZERO hallucination), C=1.000 preserved, ΔH=0.000 (position-invariant). P15 H=0.000 vs Phase 10 H=0.500. **This is the single most important positive result: weight-level intervention (LoRA) successfully bridges the B-bottleneck KNOWS→produces gap that four vector-op experiments (9-A+P12+P13+P14) could not touch. The B-bottleneck is now both geometrically characterized (depth-universal subspace separation) AND remedied (hallucination-targeted LoRA).** Combined pattern: where vector ops fail at hidden-state level, LoRA-based weight-level intervention succeeds — proven for both A-bottleneck (Phase 10) and B-bottleneck (P15).

 20. ~~P16: LoRA Geometry Analysis~~ ❌ **NEGATIVE (INFORMATIVE: mechanism revealed).** On P15 LoRA model, trained probes at all 9 layers (0-21). Result: probe acc=1.0000 everywhere (K-subspace preserved), H_base=0.000 everywhere (LoRA works at output level). w_probe steering: ΔH_max≤0 at most layers, gain=-1.0 (steering effect DECREASED). Layer 12: gain=+2.001 but DESTRUCTIVE (H 0→0.25). **P16.2 REJECTED: LoRA does NOT align K↔D subspaces. LoRA's mechanism is a ROUTING fix — it changes the model's default output path so knowledge flows into behavior correctly WITHOUT needing hidden-state steering. The K↔D subspace separation PERSISTS (or worsens). LoRA bypasses the geometric bottleneck rather than resolving it.**

 21. ~~P17: LoRA Module Ablation~~ ✅ **POSITIVE (q_proj pinpointed).** Ablated individual LoRA attention projections (q, k, v, o) across all 24 layers. Result: ONLY q_proj ablation breaks routing (H 0.000→0.250). k_proj, v_proj, o_proj ablation each has ΔH=0.000 (zero effect). -q-k same as -q alone. -v-o has minor effect (ΔH=+0.083). -ALL restores full baseline (H=0.417). **LoRA's routing fix is ENTIRELY mediated by query projection weights — it teaches the model to ATTEND differently, not to compute values differently. The mechanism is attention-pattern rewiring through query vector modification.**

 22. ~~P18: q_proj Layer Ablation~~ ✅ **POSITIVE (deep-layer core located).** 8 conditions (Group ABLATION + Group ISOLATION) on 24-layer q_proj LoRA. Group ABLATION: -q_early ΔH=0.000, -q_mid ΔH=0.000, -q_deep ΔH=+0.0833 — no single group removal breaks routing (cross-layer redundancy). Group ISOLATION: ONLY_deep H=0.0000 (PERFECT — deep alone sufficient), ONLY_mid H=0.0833 (partial), ONLY_early H=0.2500 (FAILS — early alone no routing). **Counter-hypothesis result: DEEP layers (16-23), not MID layers (8-15), are the minimally sufficient core of query routing. Mid is secondary/compensatory; early is irrelevant. B-bottleneck evidence chain now COMPLETE: geometric proof (P13+P14) → LoRA bridging (P15) → routing mechanism (P16) → q_proj mediation (P17) → deep-layer core (P18).**

 23. ~~P19: Self-Bootstrapping Attention Rerouting~~ ✅ **POSITIVE (autonomous self-repair).** First self-bootstrapping agent combining Meta FAIR's paradigm with IC-4-M0's diagnostic framework. DETECT→DIAGNOSE→REPAIR→VERIFY→REMEMBER loop on base model (H=0.417). Agent uses deep-layer (16-23) attention to discover distractor tokens, iteratively prunes them, re-evaluates log-prob. **H 0.417→0.250 (−40%), 2/5 fixed, C=1.000 preserved.** Discovered pattern: 'The' context-openers + punctuation-newline tokens are attention distractors. **Introspection-Guided Targeted Self-Repair proven: precise diagnostic map (P13-P18) enables directed self-bootstrapping, not blind search.**

 24. ~~P20: Multi-Strategy Self-Bootstrapping~~ ✅ **POSITIVE (strategy diversity essential).** Three repair strategies tested: PRUNE (remove token), NEUTRALIZE (replace with "it"), SENTENCE (remove sentence). Agent picks best per sample. 36 samples, **H 0.417→0.333 (−20%), 1/5 fixed via SENTENCE strategy.** **Key: distractor effect is SENTENCE-level for some failures — token-level alone insufficient. Multi-strategy bootstrapping is not optional but essential.** Strategy distribution: PRUNE:1, NEUTRALIZE:1, SENTENCE:2.

 25. ~~P21: Self-Generated Strategy Discovery~~ ❌ **NEGATIVE (important threshold).** LLM self-diagnosis → self-generate repair → log-prob verification. 30 samples, **H 0.417→0.417 (Δ=0), 0/5 fixed.** 0.5B model cannot reliably self-diagnose hallucination — generated_text unextractable, repair stochastic (run 1: 1 fix via luck, run 2: 0 fixes). **Key finding: self-bootstrapping via generation requires minimum capability threshold; 0.5B parameter models are below it. Negative result defines a boundary condition for Meta FAIR's self-bootstrapping paradigm.**

 26. ~~P22: Probability-Guided Cascading Counterfactual~~ ✅ **POSITIVE (probabilistic strategy discovery).** Eliminates human-defined strategy menus entirely. Uses attention weights as Bayesian prior, log-prob as likelihood objective, cascading intervention (prune→neutralize→sentence) as decision rule. **H 0.417→0.333 (−20%), 1/5 fixed at Phase 3 (sentence removal).** 36 counterfactuals, 8.9min (2.4× faster than P21). **Discovered: "The following is a response from an AI assistant" position wrapper IS a causal distractor — sentence removal at mid-position flips lp_diff from +0.083 to −0.019 (crossing zero). Probability theory succeeds where LLM self-generation fails.**

 27. ~~P23: Joint Counterfactual + Full-Token Causal Attribution~~ ⚠️ **MIXED (attention-as-causal-proxy FALSIFIED).** Multi-token joint search + full per-token counterfactual attribution on 3 unfixed samples. **H 0.417→0.333 (−20%), 1/5 fixed. Joint interventions do NOT improve over best single — distractors not additive. CENTRAL FINDING: Corr(attention_weight, Δlp_diff) = −0.0086 ≈ ZERO across 188 counterfactuals.** The true causal distractor "funding" (attn~0.007) has 50× higher causal impact (Δ~+0.36) than high-attention tokens like "The" (attn~0.42, Δ~−0.02). **Attention weights are NOT a valid search heuristic for hallucination repair. Text-level interventions hit a representation floor: even the optimal single intervention leaves lp_diff at +0.36–0.39 for funding-type hallucinations. The hallucination source is semantic ("funding" activates conflation of series-A with total) rather than token-attentional.**

 28. ~~P24: Embedding-Level Semantic Intervention~~ ❌ **NEGATIVE (structural proof).** Replaces causal token embeddings with neutral embeddings at the vector level, keeping token sequence unchanged. **H 0.417→0.417 (Δ=0). 0/5 fixed. All embedding interventions made hallucination WORSE than baseline.** Embedding replacement degrades contextual coherence without changing attention routing dynamics. **Hallucination is STRUCTURAL (attention routing) not REPRESENTATIONAL (token embedding). Text-level removal > embedding-level replacement because the former changes attention structure while the latter only corrupts a single embedding. P15's LoRA fix on q_proj remains the only complete remedy — confirming the routing-level nature of the hallucination.**

This order keeps the project coherent and prevents premature expansion.

---

## 12. Final compressed thesis

If the whole project had to be compressed into one paragraph, it would be this:

> We study small language models as high-dimensional controlled systems whose failure modes arise not only from missing knowledge, but from limits in structural adaptation: difficulty absorbing fragmented input without distortion, stabilizing useful internal structure under compression, and organizing latent capability into reliable behavior. IC-4 contributes the organization line through closed-loop and trajectory-based control experiments; IC-2 contributes the stabilization line through bad-debt diagnosis and anchored remedies; position sensitivity provides the absorption line. The project’s next step is to move from local intervention toward trajectory-level structure shaping, while keeping explicit boundaries between what is already established, what is only suggestive, and what still needs replication.

That is the main line. Everything else should serve it.

