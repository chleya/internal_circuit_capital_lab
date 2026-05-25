# IC-4-M0 Project Handoff Report — v27.0

> **For the next LLM or human collaborator continuing without prior chat context.**
> **Date:** 2026-05-24
> **Model:** Qwen-2.5-0.5B-Instruct (896D hidden, 24 layers)
> **Experiments completed:** 26 (P1-P26 + A/B/C phases)
> **Repository:** `F:\internal_circuit_capital_lab\IC-4-M0`

---

## 0. WHAT THIS PROJECT IS

This project treats a small language model as a high-dimensional controlled system and decomposes its failure modes into three structural bottlenecks:

| Bottleneck | Question | Status |
|-----------|----------|--------|
| **A: Absorption** | How does discretized input enter state space? | Diagnosed + Remedied |
| **B: Stabilization** | How does useful structure survive compression? | Diagnosed + Remedied |
| **C: Organization** | How is latent capability routed into behavior? | Diagnosed + Remedied |

The core bet: part of the small-vs-large model gap is missing **structural adaptation capacity**, not missing knowledge. External feedback/anchoring mechanisms can partially compensate.

**The B-bottleneck has the richest evidence chain (P12→P26, 15 experiments) and is the project's strongest deliverable.**

---

## 1. READ THIS FIRST (5-minute bootstrap)

If you only read three documents, read these in order:

1. **`FINAL_COMPREHENSIVE_REPORT.md`** — The definitive capstone. 26 experiments, 32 proven claims, 12 falsified hypotheses, full evidence chains. Start here.

2. **`UNIFIED_THESIS.md`** — The thesis argument with mainline experiment order.

3. **This document** — Operational instructions: what's done, what runs, what's next.

Key supporting docs:
- `PROJECT_ENDGAME_AND_HANDOFF.md` — Older (v24-era) but has the stop/go rules
- `ENGINEERING_CYBERNETICS_FRAMING.md` — Theoretical framing
- `CROSS_BOTTLENECK_SYNTHESIS.md` — Bottleneck comparison tables
- `UNIFIED_RESEARCH_MAP.md` — File-level map of everything

---

## 2. THE COMPLETE B-BOTTLENECK EVIDENCE CHAIN (P12→P26)

This 15-experiment chain is the project's crown jewel. It traces a complete arc:

```
GEOMETRIC PROOF (P12-P14):
  P12: Position-directional steering fails (always destructive)
  P13: w_probe has ZERO effect at L12 (K≠D subspace separation)
  P14: K≠D is DEPTH-UNIVERSAL (all 24 layers)

BRIDGING (P15-P18):
  P15: LoRA bridges gap: H 0.417→0.000 (ZERO), C=1.000
  P16: LoRA = ROUTING fix (bypass K≠D), not GEOMETRY fix (align K≠D)
  P17: Routing fix ENTIRELY mediated by q_proj
  P18: Deep layers (16-23) = sufficient routing core

NATURE OF HALLUCINATION (P19-P26):
  P19: Self-bootstrapping agent: H −40%, 2/5 fixed
  P20: Multi-strategy essential: SENTENCE removal uniquely works
  P21: LLM self-diagnosis fails below capability threshold (0/5)
  P22: Probabilistic counterfactual cascade: H 0.333, 1/5 fixed
  P23: Attention ≠ causality (Corr=−0.0086). "funding" = true distractor.
  P24: Hallucination = STRUCTURAL (routing) not REPRESENTATIONAL (embedding)
  P25: Activation ablation: 220 tests, 0/5 fixed, 3/5 crossed P23 floor
  P26: Per-layer dispersion: ALL samples peak at L3, monotonic decay
       → TWO-PHASE MODEL of hallucination
```

### The Two-Phase Hallucination Model (P26 — the latest finding)

```
 Layer 0        3         8                          21     23
   |           |         |                            |      |
   |← BUILD →|← PEAK →|← DISPERSION →|← DEAD ZONE →|
   |         |  L3     |               | L21-23:     |
   |         |  Δ=+0.49|               | Δ≤0         |
   SEMANTIC ENCODING    INFO DISPERSION    ROUTING DECISION
   (L0-L5)              (L6-L15)           (L16-L23)

Phase 1 (L0-L5): Token "funding" encodes "this is a finance question"
                 Peak at L3 (universal across all 3 samples, mean Δ=+0.41)
Phase 2 (L6-L15): Attention distributes semantic frame to ALL tokens
                 Individual token ablation becomes progressively useless
Phase 3 (L16-L23): q_proj routing decides hallucinate vs correct
                  By L21-23: token ablation has zero or negative effect
```

**Key reconciliation**: P25 found early@0.5 works but deep@0.0 fails. P17-P18 found deep layer q_proj is the routing core. P26 resolves: these are different phases — early layers encode the semantic trigger, deep layers execute the routing decision. LoRA on deep q_proj works because it rewires the ROUTING, not the ENCODING.

---

## 3. ALL COMPLETED EXPERIMENTS (with status)

### Absorption (A) — 11 items
| # | What | Result | Status |
|---|------|--------|--------|
| — | Position KNN Audit | KNN=1.0: hidden states encode position | ✅ |
| Phase 8-A | Position-Augmented Probe | PSI −90% (0.0676→0.0067) | ✅ |
| Phase 8-B | Behavior Gate Consistency | Gate consistent (11/11/11) | ✅ |
| Phase 9-A | Global Position Rectification | ΔH 0.111→0.333 (WORSE) | ❌ |
| Phase 10 | Position-Aware LoRA | ΔH −50%, PSI −53%, Consistency +5% | ✅ |
| Phase 11 | Cross-Bottleneck Integration | Bottlenecks INDEPENDENT (A≠C) | ✅ |
| P12 | Position-Directional Steering | Always destructive | ❌ (geometric proof) |

### Stabilization (B) — 12 items
| # | What | Result | Status |
|---|------|--------|--------|
| IC-2c | Episodic vs Consolidated | Consolidated 0.115 < Random 0.333 | ✅ |
| IC-2c.1 | Root Cause Analysis | KMeans ignores Y-information | ✅ |
| Topology | Purity/MEC Audit | Purity=0.261 (all 20 centroids mixed) | ✅ |
| Proof C | Anchored Consolidation | br=0.7: +8.7% over naive | ✅ |
| C2 Readout | Readout Repair | All 5 strategies fail (root cause = KMeans) | ✅ |
| Phase 6-A | Seed Scaling (5→100) | PA passes ALL, peak 0.660 at 50 seeds | ✅ |
| Phase 6-B | Objective Scaling (3→20) | PA stable, peak 0.715 at 5 actions | ✅ |
| Phase 6-C | Noise Scaling (σ=0→1.0) | PA holds 0.495-0.545 | ✅ |
| Phase 7 | Cross-Bottleneck Analogue | PA > NoMem at all noise/shift levels | ✅ |
| Phase 7-B | LLM Consolidation | ALL=1.000 (data-gap, need multi-ckpt) | ❌ |
| Phase 9-B | Multi-Checkpoint Consolidation | PerClass +0.37, KNN=1.0 persists | ⚠️ |

### Organization / Hallucination (C) — 25+ items
| # | What | Result | Status |
|---|------|--------|--------|
| M3-v6 | Closed-Loop Hallucination Gate | H 0.867→0.667 | ✅ |
| M4 | OOD Robustness (3×3) | All pass causal separation | ✅ |
| M5 | Gate Boundary Analysis | Hall✓ / Syc seed-dep / Correctness bilateral | ✅ |
| M7-Lv2 | Latent Capability Routing | Oracle routing 85.7% | ✅ |
| P2 | Hallucination Direction Audit | v_hall = v_orthogonal (no specificity) | ❌ (important) |
| Syc Audit | Sycophancy Direction Audit | v_syc < random (ceiling effect) | ❌ (unifying) |
| Proof A | Syc Direction (n larger) | NOT replicated | ✅ |
| Proof B | Multi-Direction Hallucination | Structured advantage only at degraded baseline | ✅ |
| P5-P9 | Sycophancy Feedback Chain | Two-stage gate, multi-layer, Pareto frontier | ✅ |
| P8 | Large-Scale (n=24) | −57.1% sycophancy reduction | ✅ |
| P13 | Probe-Guided Hallucination Steering | acc=1.000, H flat (K≠D) | ❌ (geometric proof) |
| P14 | Cross-Layer B-Bottleneck | Depth-universal K≠D | ❌ (geometric proof) |
| P15 | Hallucination LoRA | H 0.417→**0.000** (ZERO), C=1.000 | ✅ |
| P16 | LoRA Geometry | Routing fix, not geometry fix | ❌ (informative) |
| P17 | LoRA Module Ablation | ONLY q_proj (ΔH=+0.250), k/v/o ΔH=0 | ✅ |
| P18 | q_proj Layer Ablation | Deep (16-23) = sufficient core | ✅ |
| P19 | Self-Bootstrapping Agent | H −40%, 2/5 fixed | ✅ |
| P20 | Multi-Strategy Bootstrapping | SENTENCE removal uniquely succeeds | ✅ |
| P21 | Self-Generated Strategies | 0/5 fixed (capability threshold) | ❌ |
| P22 | Probabilistic Counterfactual Cascade | H 0.333, 1/5 fixed | ✅ |
| P23 | Joint Causal Attribution | Corr(attn, Δlp_diff) = −0.0086 | ✅ (attention ≠ causal!) |
| P24 | Embedding-Level Intervention | H 0.417, 0/5 fixed (structural proof) | ❌ |
| P25 | Activation Ablation (Ranges) | H 0.417, 0/5 fixed, 3/5 crossed floor | ✅ |
| P26 | Per-Layer Dispersion Profile | ALL peak at L3, monotonic decay | ✅ |

---

## 4. KEY NUMBERS AT A GLANCE

| Metric | Baseline | Best Fix | Method |
|--------|----------|----------|--------|
| Hallucination Rate (H) | 0.417 | **0.000** | LoRA on q_proj, deep layers 16-23 (P15) |
| Correctness (C) | 1.000 | 1.000 | Preserved under all fixes |
| LoRA q_proj ablated | 0.000 | 0.250 | Removing q_proj breaks routing (P17) |
| ONLY_deep q_proj active | — | 0.000 | Deep layers alone sufficient (P18) |
| ONLY_early q_proj active | — | 0.250 | Early layers NOT sufficient (P18) |
| Self-bootstrapping H | 0.417 | 0.250 | SBAR agent (P19) |
| Text-level removal H | 0.417 | 0.333 | Counterfactual cascade (P22) |
| Corr(attention, Δlp_diff) | — | −0.0086 | Zero correlation (P23) |
| "funding" peak causal layer | — | **L3** | Universal across 3 samples (P26) |
| "funding" mean Δ at L3 | — | +0.4114 | Single-layer ablation (P26) |
| By L21: single-layer Δ | — | < +0.05 | Info fully dispersed (P26) |
| By L22-L23: single-layer Δ | — | ≤ 0 | Dead zone (P26) |

---

## 5. HOW TO RUN EXPERIMENTS

### Environment
```bash
cd F:\internal_circuit_capital_lab\IC-4-M0
pip install torch transformers
```

### Running individual experiments
All run scripts are self-contained in `src/`. Each prints results + saves JSON to `results_*/`.

**Most important scripts (B-bottleneck chain):**
```bash
python src/run_p15_hallucination_lora.py     # Train LoRA: H→0 fix (~5 min)
python src/run_p17_lora_ablation.py           # Module ablation: q_proj pinpoint (~3 min)
python src/run_p18_qproj_layer_ablation.py    # Layer ablation: deep core (~5 min)
python src/run_p23_joint_causal_attribution.py # Attention≠causal proof (~10 min)
python src/run_p24_embedding_intervention.py   # Structural vs representational (~8 min)
python src/run_p25_activation_ablation.py      # Three-tier intervention (~30 min)
python src/run_p26_per_layer_ablation.py       # Per-layer dispersion curve (~15 min)
```

**Infrastructure utilities:**
```python
# Loading test data (used by all scripts)
from src.data_builder import load_jsonl
samples = load_jsonl("data_position_sensitivity/s0/test_early_s0.jsonl")

# Model + tokenizer pattern
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float32, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

### Data format
Test data in `data_position_sensitivity/s0/`:
- `test_early_s0.jsonl`, `test_mid_s0.jsonl`, `test_late_s0.jsonl`
- Each sample: `{"context":..., "question":..., "positive_response":..., "negative_response":..., "answerability":"answerable"/"unanswerable"}`
- 30 samples total (18 answerable + 12 unanswerable, varied by position)

### Token Finding Pattern (Qwen tokenizer quirk)
```python
# Qwen uses space-prefix tokens: " funding" → ['Ġfunding']
# Must match with both variants:
for variant in [target, " " + target]:
    variant_tokens = tokenizer.tokenize(variant)
    # then match against convert_ids_to_tokens(input_ids)
```

### Forward Hook Pattern (used by P25/P26)
```python
def hook_fn(module, input, output):
    if isinstance(output, (tuple, list)):
        hidden_states = output[0].clone()  # MUST clone, not in-place
        rest = output[1:]
    else:
        hidden_states = output.clone()
        rest = None
    # modify hidden_states...
    if rest is None:
        return hidden_states
    return (hidden_states,) + rest

handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
# ... forward pass ...
handle.remove()
```

---

## 6. OPEN QUESTIONS & NEXT DIRECTIONS

### Priority A: Cross-Bottleneck Joint Intervention
P11 proved bottlenecks are independent. P15 proved LoRA fixes B-bottleneck. Phase 10 proved LoRA fixes A-bottleneck.
**Question**: If we apply BOTH simultaneously (position-aware LoRA + hallucination LoRA), do they compose additively or interact?
**Experiment**: Train both LoRA adapters, test on the full 30-sample set.

### Priority B: Scale Verification
All experiments on Qwen-2.5-0.5B. Does the two-phase model hold at larger scales?
**Question**: Does "funding" still peak at L3 in Qwen-2.5-1.5B?
**Experiment**: Replicate P26 on 1.5B model.

### Priority C: Cross-Token Generalization
P26 studied "funding" only. Do other causal tokens (e.g., from "r_and_d_spend" questions) show different dispersion profiles?
**Question**: Is L3 the universal semantic encoding gate, or specific to "funding"?
**Experiment**: Replicate P26 on the JetCircuit "r_and_d_spend" hallucinated samples (samples 17, 27).

### Priority D: Information Dispersion Mechanism
P26 shows monotonic decay after L5, but the mechanism is unclear.
**Question**: Is dispersion via self-attention (same-layer) or feed-forward?
**Experiment**: Ablate attention heads at specific layers and measure dispersion curve changes.

### Priority E: Write-Up
The evidence chain is complete. A paper could be structured as:
1. Introduction: Three-bottleneck framework
2. Geometric proof: K≠D across transformer depth (P13-P14)
3. Bridging: LoRA on q_proj as routing fix (P15-P18)
4. Nature: Two-phase hallucination model (P19-P26)
5. Cross-bottleneck synthesis

---

## 7. WHAT IS SOLID VS WHAT IS TENTATIVE

### SOLID (do not re-litigate without extraordinary evidence)
- K≠D subspace separation is real and depth-universal (P13+P14)
- LoRA on q_proj in deep layers (16-23) completely fixes hallucination (P15-P18)
- Attention weights ≠ causal importance (P23: Corr=−0.0086)
- Hallucination is routing, not representational (P24)
- "funding" peaks at L3 across all samples (P26)
- Information disperses monotonically after L5 (P26)
- Three bottlenecks are independent (Phase 11)
- LLM self-diagnosis fails at 0.5B scale (P21)

### TENTATIVE (needs more evidence)
- Whether L3 is universal for ALL causal tokens (only tested on "funding")
- Whether the two-phase model holds at larger scales
- Whether cross-bottleneck joint intervention is additive
- The exact dispersion mechanism (attention vs FFN)

---

## 8. GIT COMMIT HISTORY (key milestones)

```
v27.0: P26 Per-Layer Information Dispersion Profile — two-phase model
v26.0: P25 Causal Token Activation Ablation — three-tier verification
v25.0: Complete report rewrite (P1-P24)
v24.0: P24 Embedding Intervention — structural vs representational proof
v23.0: P23 Joint Causal Attribution — attention≠causal
v22.0: P22 Probabilistic Counterfactual Cascade
...
v15.0: P15 LoRA — H 0.417→0.000 breakthrough
...
```

To see full history: `git log --oneline`

---

## 9. RESULT FILES MAP

Each experiment saves to its own directory:
```
results_p15_hallucination_lora/         # Contains checkpoint_final/ with LoRA weights
results_p16_lora_geometry/
results_p17_lora_ablation/
results_p18_qproj_layer_ablation/
results_p22_counterfactual_cascade/
results_p23_joint_causal_attribution/
results_p24_embedding_intervention/
results_p25_activation_ablation/
results_p26_per_layer_ablation/
results_a4_position_aware_training/     # Contains checkpoint_final/ with A-bottleneck LoRA
results_cross_bottleneck_integration/   # Phase 11 independence proof
```

Each `results.json` has: config, metrics, per-sample details, timing.

---

## 10. FINAL INSTRUCTION TO NEXT LLM

**DO:**
1. Start by reading `FINAL_COMPREHENSIVE_REPORT.md` (sections 3-5 for B-bottleneck)
2. Classify any new experiment as A, B, or C bottleneck
3. Run existing scripts to reproduce baseline results first
4. Check `results_*/` directories for existing data before re-running
5. Follow the stop/go rule: does the new experiment sharpen a bottleneck distinction?

**DON'T:**
1. Don't add new tasks without mapping to a bottleneck
2. Don't trust attention weights as causal proxies (P23 disproved this)
3. Don't try vector steering to fix hallucination (P12-P14 disproved this)
4. Don't try embedding-level fixes (P24 disproved this)
5. Don't try individual token ablation for deep layers (P25-P26 disproved this)

**IF STUCK, START HERE:**
The cleanest next experiment is Priority C (Cross-Token Generalization): replicate P26 on samples 17 and 27 (JetCircuit "r_and_d_spend" questions) to see if L3 is universal or "funding"-specific.

---

*End of Project Handoff Report v27.0. Repository: `F:\internal_circuit_capital_lab\IC-4-M0`.*