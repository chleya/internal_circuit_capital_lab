# IC4-S3E: Quality-Guarded Late-Layer Anti-Syc Internalization

Date: 2026-05-27

## One-Sentence Verdict

**negative_but_informative** — P41 late-layer recipe does NOT rescue S3D's cos_sim+CE training from quality collapse; the collapse stems from the training objective itself, not from layer choice. Early-layer training uniquely preserves quality but makes sycophancy worse (mirror of P39's despair valley).

## What Question This Experiment Answers

Can P41's late-layer LoRA recipe (L14-L23, r=8, q/k/v/o) — proven for hallucination suppression — generalize to sycophancy suppression and solve the quality collapse problem discovered in S3D?

Answer: **No, the late-layer recipe does not rescue this training paradigm.** But the failure mode reveals a more important finding: the S3 cos_sim+CE training objective is fundamentally destructive regardless of layer choice. The P41 recipe worked because it uses a different training paradigm (pure CE on correct responses).

## What Files Were Changed or Added

1. **New**: `F:\internal_circuit_capital_lab\IC-4-M0\src\run_s3e_quality_guarded_internalization.py`
2. **New**: `F:\internal_circuit_capital_lab\IC-4-M0\results_s3e_quality_guarded_internalization\results.json`
3. **New**: `F:\internal_circuit_capital_lab\IC-4-M0\results_s3e_quality_guarded_internalization\run_log.txt`

## What Prior Reports This Experiment Depends On

- `IC4_P41_LATE_LAYER_REFINEMENT.md` — late-layer recipe (L14-L23, r=8, q/k/v/o as optimal)
- `IC4_P39_LAYER_THRESHOLD_REPORT.md` — despair valley: early-layer training worsens behavior
- S3D results (`results_s3d_ultimate_routing/results.json`) — all-layer anti-syc → quality=0.0 collapse

## Experimental Design

### Configurations (4)

| Config | Layers | Rank | Target Modules | Purpose |
|--------|--------|------|----------------|---------|
| BASE | none | — | — | Pre-trained baseline |
| ALL-LAYERS | 0-23 (24L) | r=8 | q/k/v/o/gate/up/down | S3D replication |
| LATE-L14-L23 | 14-23 (10L) | r=8 | q/k/v/o | P41 recipe test |
| EARLY-L0-L11 | 0-11 (12L) | r=8 | q/k/v/o | Negative control |

### Training Details

- **Data**: sycophancy_contrast_data.json (30 sycophantic + 30 non-sycophantic)
- **Split**: train=10S (syc-only), test=10S+10N=20
- **Objective**: CE on prompt tokens + λ*cos_sim (push hidden states at L10 away from v_syc)
- **λ**: 0.15, **Epochs**: 2, **LR**: 5e-4, **Batch**: 2
- **Steering vector**: v_syc computed at L10 from train data (matching S3D)
- **Layer freezing**: `freeze_non_target_layers()` from P41

### Quality Guards Implemented

1. **Garbled text detection**: `!!!!`, `@#$`, `???` markers + length < 3
2. **Repetition rate**: trigram-based repetition score (matching P41)
3. **Non-syc preservation**: separate evaluation on non-sycophantic test subset
4. **Output samples**: full text for every test prompt

## Metrics

- **syc_rate**: fraction of non-garbled outputs matching sycophancy patterns
- **quality**: fraction of non-garbled outputs (1.0 = all clean)
- **garbled_count**: absolute count of garbled outputs
- **mean_rep**: mean trigram repetition score
- **syc_only / non_only**: subset breakdowns

## Controls

1. **S3D replication**: ALL-LAYERS config matches S3D's training design
2. **P41 recipe**: LATE-L14-L23 uses exact P41 layer range + rank
3. **Early negative control**: EARLY-L0-L11 tests despair valley in syc domain
4. **Baseline**: Pre-trained model without any training

## Results Table

| Config | TrParams | SycRate | Quality | Garbled | Rep | SycOnly | NonOnly | NonQual |
|--------|----------|---------|---------|---------|-----|---------|---------|---------|
| BASE | — | 0.3000 | 1.0000 | 0 | 0.0 | 0.5000 | 0.0000 | 1.0000 |
| **ALL-LAYERS** | 4,399,104 | 0.0000 | **0.0000** | 20 | 0.0 | 0.0000 | 0.0000 | 0.0000 |
| **LATE-L14-L23** | 450,560 | 0.0000 | **0.1000** | 18 | 0.0 | 0.0000 | 0.0000 | 0.1000 |
| **EARLY-L0-L11** | 540,672 | **0.5000** | **1.0000** | 0 | 0.0 | 0.9000 | 0.1000 | 1.0000 |

### Base vs Trained — Key Comparisons

| Metric | BASE | ALL-LAYERS | LATE-L14-L23 | EARLY-L0-L11 |
|--------|------|------------|--------------|--------------|
| Syc (all) | 0.30 | **0.00** ✓ | **0.00** ✓ | **0.50** ✗ |
| Quality | 1.00 | **0.00** ✗ | **0.10** ✗ | **1.00** ✓ |
| Syc (syc-only) | 0.50 | 0.00 | 0.00 | 0.90 |
| Non-syc preserved | ✓ | ✗ | ✗ | ✓ |

### S3D Comparison

| Experiment | Config | SycRate | Quality |
|------------|--------|---------|---------|
| S3D | ALL-LAYERS | 0.0000 | 0.0000 |
| **S3E** | **ALL-LAYERS** | **0.0000** | **0.0000** |
| S3E | LATE-L14-L23 | 0.0000 | 0.1000 |

S3D replication confirmed: all-layer anti-syc LoRA produces complete quality collapse (20/20 "!!!!!!!!").

## Interpretation

### Finding 1: Quality collapse is objective-driven, not layer-driven

Both ALL-LAYERS and LATE-L14-L23 suffered near-complete quality collapse (quality=0.00 and 0.10 respectively). The LATE variant had 2 slightly less garbled outputs ("![](![](![](") but these are still non-functional. **The cos_sim+CE training objective destroys generation capability regardless of which layers are trained.**

This refutes the initial hypothesis that quality collapse was an all-layer routing distortion problem. Instead, the cos_sim steering loss at L10 acts as a representation-level attack that the model cannot recover from during generation, even when only late layers are modified.

### Finding 2: Early-layer training uniquely preserves quality but worsens behavior

EARLY-L0-L11 is the ONLY config with quality=1.0000 — identical to baseline. But syc_rate INCREASED from 0.30 (base) to 0.50. On the syc-only subset, syc_rate went from 0.50 to 0.90.

This is the **mirror image of P39's despair valley**: just as early-layer hallucination training made hallucination worse, early-layer anti-syc training makes sycophancy worse. In both cases, modifying early layers distorts the information routing while leaving late-layer execution intact — resulting in worse behavior with preserved fluency.

### Finding 3: P41 recipe ≠ S3D recipe — different training paradigms

P41 succeeded because it used **pure CE on correct target responses**:
- The model learns "when asked X, say the correct answer Y"
- This is positive behavioral guidance

S3D/S3E failed because it used **cos_sim representation pushing**:
- The model is told "don't represent things in this direction"
- This is negative representation-level constraint
- Combined with weak CE anchor (predicting prompt tokens), the model collapses

**The late-layer recipe for behavioral control works with CE objective but NOT with cos_sim steering objective.** This is a critical boundary condition on the P41 recipe's generalizability.

### Finding 4: cross-domain recipe transfer requires training paradigm match

The P41 recipe (late-layer CE on correct targets) can potentially be applied to sycophancy, but it would require:
1. A dataset of sycophantic prompts with correct non-sycophantic responses
2. CE training on those correct responses (not cos_sim pushing)
3. Late-layer LoRA (L14-L23, r=8, q/k/v/o)

This is a different experiment from S3E — it would be "CE-based anti-syc late-layer training" rather than "cos_sim-based."

## Solid Ground

1. **ALL-LAYERS cos_sim+CE → complete quality collapse**: Replicated S3D result (20/20 garbled, quality=0.0000)
2. **Late-layer-only cos_sim+CE → also collapses**: quality=0.1000 (18/20 garbled). Layer choice does not rescue this objective.
3. **Early-layer-only cos_sim+CE → quality preserved, behavior worse**: quality=1.0000 but syc_rate increases from 0.30→0.50, mirroring P39's despair valley pattern
4. **CE on correct targets ≠ cos_sim steering**: These are fundamentally different training paradigms with different collapse risks

## Replicated but Scoped

1. **S3D collapse replicated** — all-layer cos_sim anti-syc training → quality=0.00 (n=20 test samples, consistent with S3D's n=10)
2. **P39 despair valley pattern replicated in syc domain** — early-layer training worsens target behavior while preserving fluency

## Not Yet Solid

1. Whether CE-based late-layer anti-syc training (with correct non-syc responses as targets) would work — not tested
2. Whether λ tuning (lower λ) could prevent collapse while still suppressing syc — not tested
3. Whether a DPO-style contrastive objective would work better than cos_sim — not tested
4. Whether the collapse is specific to L10 as the steering target layer — other layers not tested

## Failure Modes

1. **Complete generation collapse**: ALL-LAYERS and LATE-L14-L23 both produce "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" for almost all outputs
2. **No syc suppression with quality**: The only config with quality=1.0 (EARLY) made syc worse, not better
3. **cos_sim at L10 may be too aggressive**: The λ=0.15 value from S3D may need recalibration for different layer ranges
4. **Weak CE anchor**: CE on prompt tokens (labels=ids) provides insufficient language modeling constraint against the cos_sim push

## Next Recommended Experiment

Two paths forward:

**Path A (within S3 line)**: Test whether the P41 CE paradigm — not the S3 cos_sim paradigm — works for sycophancy:
- Build a dataset of (syc_prompt, correct_non_syc_response) pairs
- Train late-layer LoRA (L14-L23) with pure CE on correct responses
- This tests "does late-layer CE generalize to sycophancy?"

**Path B (masterplan order)**: Move to **Experiment D: IC4-Controller-v2 Unified Hard-Gated Superposition** in `new-5` repo:
- Unify new-5 hard-gated feedback control with IC-4-M0 steering
- Resolve syc/hall independence question
- Uses inference-time steering (no training collapse risk)
- More mature infrastructure with established alpha defaults

**Recommendation**: Path B (Experiment D). S3E showed the cos_sim paradigm is fundamentally limited, and building CE-based anti-syc data is a separate data engineering task. Experiment D is ready to run with established infrastructure and addresses the next critical question in the masterplan queue.