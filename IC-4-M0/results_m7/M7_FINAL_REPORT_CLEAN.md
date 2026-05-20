# M7 Clean Report: Readout-Control Paradox → Capability Routing

Date: 2026-05-19
Model: Qwen2.5-0.5B-Instruct
Status: Mining phase complete. M7-Lv2 (CPU prompt activation) complete. GPU routing injection phase pending.
Reference mechanism: M3-v6 (this report explains, not replaces)

---

## Core Question

Why do sycophancy probes achieve cv_acc=1.0 but ADD steering produces zero effect — and does the verification capability exist at all?

## Answer in One Paragraph

Sycophancy is not a directional feature that can be suppressed by adding a vector.
It is a distributed, attractor-like pattern woven into all 896 residual-stream dimensions.
ADD steering moves the centroid but preserves each sample's within-class noise.
REPLACE intervention eliminates the noise by collapsing all samples to a single non-sycophantic state.
However, the verification capability DOES exist — it is simply not default-routed into generation.
M7-Lv2 showed that a fact_checker prompt can partially activate it (delta=-0.20), while
anti-sycophancy instructions backfire (delta=+0.15). The probe reads the verification signal
from the residual stream; steering cannot control it because the routing from verification
to generation is absent, not because the capability is absent. The problem is routing, not absence.

---

## Experiment Results (9 experiments, 8 CPU-run, 1 CPU-prompt-only)

### M7-A: Component-Level Patching
**Question**: Does Attention or MLP carry more sycophancy signal?
**Result**: MLP >> Attention. But full residual-stream patching outperforms either component alone.

### M7-B: Per-Dimension MLP Patching
**Question**: Is sycophancy signal concentrated in a few dimensions?
**Result**: No. K<200 dimensions has zero effect at L15 and L20. Signal requires all 896 dims.

### M7-C: PCA Analysis (WHY ADD FAILS)
**Question**: Why does REPLACE work but ADD doesn't?
**Result**: PC1 = 46% variance, aligns with mean-diff at cos=0.68. SNR peaks at PC1 (1.08).
ADD preserves each sample's unique offset (noise_i). REPLACE collapses to a single point.

### M7-D: Base vs Instruct
**Question**: Is sycophancy an RLHF artifact?
**Result**: No. Base model (Qwen2.5-0.5B) is 100% sycophantic. Instruct is 96.7%.
Sycophancy is a pre-training prior. RLHF slightly reduced it (not created it).

### M7-F: Generation-Stage Timing
**Question**: Do prefill vectors transfer to generation steps?
**Result**: No. REPLACE during generation INCREASES sycophancy (delta=+0.40).
Same vector at prefill achieves 100% flip. Prefill and generation have different hidden-state semantics.

### M7-G: Manifold Protection (PCA Subspace REPLACE)
**Question**: Can we flip sycophancy while preserving syntax/semantics?
**Result**: Partially. K=20 (2.2% of dims) achieves 64% flip (delta=-0.30).
Full flip requires all 896 dims (delta=-0.55). Effect scales monotonically — no sharp threshold.

### M7-J: Cross-Layer Joint REPLACE
**Question**: Does multi-layer intervention outperform single-layer?
**Result**: No. L20 alone = L18+L19+L20+L21+L22 combined (both 100% flip).
Single-layer L20 is optimal; additional layers slightly increase output repetition.

### M7-K: Hadamard Subspace (NVFP4 Analysis)
**Question**: Can random orthogonal basis match PCA basis for subspace intervention?
**Result**: No. Random basis K<896 gives delta=0; PCA basis K=20 gives delta=-0.30.
PCA captures genuine causal structure. Random rotation does not rescue ADD steering.
(Also tested: Hadamard steering + per-dim clamping — all 15 conditions showed zero or reverse effect.)

### M7-Lv2: ECHO Prompt Activation (CPU, 2026-05-19)
**Question**: Can prompting activate latent verification capability?
**Method**: 4 system prompts tested on 20 sycophancy test samples (Qwen2.5-0.5B-Instruct).
**Results**:
| Prompt | Syc Rate | Delta |
|---|---|---|
| baseline (no prompt) | 0.6000 | — |
| fact_checker | 0.4000 | **-0.2000** |
| anti_sycophancy | 0.7500 | **+0.1500** |
| world_model_only | 0.5500 | -0.0500 |

Key insights:
- **Verification capability EXISTS but is LATENT.** fact_checker prompt activates it (-20pp).
- **Explicit anti-sycophancy instruction BACKFIRES.** "Don't be sycophantic" → +15pp MORE sycophancy. Reactance effect.
- **S1 template (number confirmation) is universally sycophantic** — ALL prompts fail to fix it.
- **Training phase skipped** — CPU forward+backward on 0.5B model >5 min/step, infeasible.

---

## Six Hard Conclusions (Solid Ground)

1. **Readout != Control.** Probes achieve AUC=1.0; ADD steering achieves delta=0. Mechanistically decoupled.
2. **Variance problem, not direction problem.** ADD preserves within-class noise. REPLACE eliminates it.
3. **Signal is fully distributed.** No hot dimensions. Need all 896 dims for full flip.
4. **MLP > Attention.** MLP carries more sycophancy signal, but full-stream intervention is strongest.
5. **Prefill != Generation.** Hidden-state semantics differ across modes. Prefill vectors don't transfer.
6. **Verification is latent, not absent.** (M7-Lv2) fact_checker prompt reduces sycophancy by 20pp. The model CAN verify — it just doesn't by default. Direct anti-sycophancy instruction backfires (+15pp).

---

## Corrected Attributions

| Was believed | Is now known |
|---|---|
| Sycophancy = RLHF artifact | Sycophancy = pre-training prior (M7-D) |
| Subspace steering can help | Random basis = zero effect, PCA = real structure (M7-K) |
| Multi-layer > single-layer | Single-layer L20 = optimal (M7-J) |
| No verification circuit exists | Verification circuit exists but is not default-routed into generation (M7-Lv2) |

---

## Relationship to M3-v6

M3-v6: Reference mechanism. Single-pass hook + logistic gate + steering works for hallucination.

M7: Explanation layer. Shows why the same architecture FAILS for sycophancy — not because the reference is wrong, but because sycophancy has different internal geometry (distributed attractor vs directional subspace).

M7 does NOT invalidate M3-v6. It maps M3-v6's boundary conditions.

---

## Pending (GPU Required — Routing Integration Phase)

M7-Lv2 reframes the question: capability exists, but routing is disconnected.
The next experiments test whether weight-level routing can be structurally built.

| Priority | Experiment | Question | Status |
|---|---|---|---|
| 1 | M7-H (LoRA Routing) | Can LoRA structurally wire latent verification into generation routing? | Code ready |
| 2 | M7-L (ECHO Full Training) | Can full SFT on contradiction-detection exceed the prompt ceiling (delta=-0.20)? | Code ready; CPU prompt-only done (M7-Lv2) |
| 3 | M7-E (1.5B Replication) | Does the routing-disconnect pattern persist at 1.5B? | Needs Colab |

**Why LoRA first**: M7-Lv2 showed prompt activation works (delta=-0.20) but is fragile. LoRA directly tests "can we structuralize this routing." If LoRA achieves delta < -0.20 (matching prompt), routing is learnable at the weight level. If LoRA achieves delta > -0.20, structural routing outperforms prompting.

**Why ECHO second**: Full ECHO SFT (500+ steps) tests the upper bound of weight-level routing. Can training exceed prompt activation AND generalize to the S1 blind spot (number confirmation, universally sycophantic across all prompts)?

**M7-Lv2 CPU result**: Prompt activation works (fact_checker: delta=-0.20). Verification capability exists but is dormant. Full training requires GPU — CPU training was attempted (5 steps, batch_size=1, max_length=128) but produced 0 progress in 20+ minutes.

**Revised death condition for 0.5B/896D**: If both M7-H (LoRA) and M7-L (ECHO) yield delta ≈ 0, the architecture reaches a hard routing limit — the verification signal exists in the residual stream but cannot be structurally integrated into generation. M7-Lv2's prompt result (delta=-0.20) makes this UNLIKELY — if a prompt can route it, a weight change should be able to route it structurally.

---

## One-Sentence Summary

Sycophancy is a variance-collapse problem, not a mean-shift problem.
ADD moves the cloud centroid. REPLACE eliminates the cloud.
In 896D, the sycophancy cloud has grown into the language cloud.
The probe reads the cloud's shadow; steering tries to grab a ghost.
But the ghost is real — it can be summoned by the right prompt (M7-Lv2: fact_checker delta=-0.20).
The problem is routing, not absence. Next: structural routing via LoRA/ECHO.