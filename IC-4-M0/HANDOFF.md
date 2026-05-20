# Ultra-Short Handoff — Capability Routing × Structural Fidelity

**Date**: 2026-05-19 | **For**: next agent / researcher taking over

---

## Anchors (Do Not Retreat)

| # | Conclusion | File |
|---|---|---|
| 1 | M3-v6 reference mechanism works: last_prompt_token + logistic probe + hard gate + single-pass hook + model.generate() | IC4_PROJECT_TERRAIN_MANUAL.md §2 |
| 2 | Under 30A+30U construction, causal ordering (random > shuffled > real_gate) holds for all tested seed/layer combos | IC4_P15_FAILURE_ANALYSIS_REPORT.md |
| 3 | P1's 2/5 ARTIFACTs were 15A+15U small-sample artifacts, not mechanism failures — P1.5 patch test proved this | same file |
| 4 | Continual consolidation (KMeans rewriting across distributions) = bad debt, not appreciation | IC2C_EPISODIC_VS_CONSOLIDATED_REPORT.md |
| 5 | Root cause: wrong readout (Euclidean k-NN) for history features; centroid imbalance 2.86→7.27 | IC2C1_ROOT_CAUSE_REPORT.md |

## Main Problem

> **Useful structure (capability or information) exists inside the system but is not correctly routed into default behavior — and wrong integration/consolidation methods actively destroy it.**

Two sides of the same coin:
- IC-4: latent capability exists → how to route it into generation?
- IC-2: useful information exists → how to read it without destroying it?

## Next Priorities

| Priority | What | Where | Needs |
|---|---|---|---|
| **P0** | 30A+30U full re-validation of all P1 success configs | IC-4, CPU | script ready, ~45 min |
| **P0** | IC-2d: learned readout for episodic retention | minimal_lab, CPU | script ready, fast |
| P1 | M7-H LoRA routing injection | IC-4 | GPU required |
| P1 | IC-2e: distribution-aware consolidation | minimal_lab, CPU | after IC-2d |

## Key Files

| File | Role |
|---|---|
| `IC-4-M0/UNIFIED_RESEARCH_MAP.md` | Cross-project master map (v4.0) |
| `IC-4-M0/reports/IC4_PROJECT_TERRAIN_MANUAL.md` | IC-4 terrain manual (v3.2) |
| `IC-4-M0/reports/IC4_P15_FAILURE_ANALYSIS_REPORT.md` | P1.5 failure mode analysis |
| `minimal_lab/IC2C1_ROOT_CAUSE_REPORT.md` | IC-2c.1 root cause analysis |
| `minimal_lab/IC2C_EPISODIC_VS_CONSOLIDATED_REPORT.md` | IC-2c episodic vs consolidated |

## ⚠️ Encoding Note

All files are UTF-8 (verified). PowerShell terminals may show mojibake — run `chcp 65001` or use `utf-8` mode. Files are not corrupted.

## One-Sentence Project Identity

> **Mechanism engineering phase: routing latent capability into default behavior while preserving structural fidelity under integration.**