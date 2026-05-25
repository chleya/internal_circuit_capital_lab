"""P6-bis: Threshold Calibration for Sycophancy Feedback Control.

P6 proved that a behavior-only probe can learn sycophancy tendency (78% test acc),
but feedback control failed because probe scores cluster near 0.5 (mean=0.4889),
causing gate rate of only 8.3%.

P6-bis systematically tests lower thresholds and percentile-based gating
to activate the probe->gate->hook feedback loop.

Design:
  Phase 1: Load existing P6 probe and test data
  Phase 2: Probe score distribution analysis on test set
  Phase 3: Threshold sweep (0.30, 0.35, 0.40, 0.45) — feedback control
  Phase 4: Percentile-based gating (top-20%, top-30%, top-40%)
  Phase 5: Open-loop reference (v_syc α=-3.0)
  Phase 6: Report generation

Usage:
  python -m src.run_p6_bis_threshold_calibration

Outputs:
  results_p6_bis_threshold/
    threshold_sweep_results.npz
    score_distribution.npz
  reports/IC4_P6_BIS_THRESHOLD_REPORT.md
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from .model_loader import load_model_and_tokenizer
from .steering import apply_steering_hook
from .run_p0_sycophancy_contrast import _is_sycophantic

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_log = logging.getLogger("p6_bis_threshold")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results_p6_bis_threshold"
REPORTS_DIR = PROJECT_ROOT / "reports"
CONTRAST_DATA_PATH = PROJECT_ROOT / "results_p0_sycophancy" / "sycophancy_contrast_data.json"
STEERING_VECTORS_PATH = PROJECT_ROOT / "results_t3_impulse_p4" / "steering_vectors.npz"
P6_PROBE_PATH = PROJECT_ROOT / "results_p6_syc_behavior" / "probe_model.pkl"
P6_RESULTS_PATH = PROJECT_ROOT / "results_p6_syc_behavior" / "p6_results.npz"

STEERING_LAYER = 10
OPTIMAL_ALPHA = -3.0
TEST_TEMPERATURE = 0.0
MAX_NEW_TOKENS = 128
RANDOM_SEED = 42

THRESHOLD_SWEEP = [0.30, 0.35, 0.40, 0.45, 0.50]
PERCENTILE_GATING = [0.20, 0.30, 0.40]


def _prompt_from_sample(sample: dict) -> str:
    prompt = sample.get("prompt", "")
    if not prompt:
        context = sample.get("context", "")
        question = sample.get("question", "")
        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
    return prompt


def _collect_last_token_hidden_state(model, tokenizer, prompt: str, layer: int) -> np.ndarray:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    hidden_states = []

    def _hook(module, inputs_tup, output):
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
        hidden_states.append(hs[:, -1, :].detach().cpu().numpy().copy())

    target_module = None
    for name, mod in model.named_modules():
        if name == f"model.layers.{layer}":
            target_module = mod
            break
    if target_module is None:
        raise ValueError(f"Layer {layer} not found")

    handle = target_module.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        handle.remove()
    return hidden_states[0][0]


def _make_feedback_hook_fn(model, probe, steering_vector, alpha, threshold, stats):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    steering_tensor = torch.tensor(steering_vector, dtype=dtype, device=device)
    gate_triggered = [False]

    def _feedback_hook(module, inputs_tup, output):
        if gate_triggered[0]:
            return output
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
        last_token_hs = hs[:, -1, :].detach().float().cpu().numpy()
        probe_score = float(probe.predict_proba(last_token_hs)[0, 1])
        stats["probe_scores"].append(probe_score)
        if probe_score >= threshold:
            gate_triggered[0] = True
            stats["gate_triggers"] += 1
            if isinstance(output, tuple):
                modified = output[0] + alpha * steering_tensor
                return (modified,) + output[1:]
            else:
                return output + alpha * steering_tensor
        stats["gate_passes"] += 1
        return output

    return _feedback_hook


def run_feedback_generation(
    model, tokenizer, test_samples, probe, steering_vector, alpha, direction_name, threshold,
):
    results = []
    stats = {"probe_scores": [], "gate_triggers": 0, "gate_passes": 0,
              "direction": direction_name, "alpha": alpha, "threshold": threshold}
    device = next(model.parameters()).device

    target_module = None
    for name, mod in model.named_modules():
        if name == f"model.layers.{STEERING_LAYER}":
            target_module = mod
            break
    if target_module is None:
        raise ValueError(f"Layer {STEERING_LAYER} not found")

    hook_fn = _make_feedback_hook_fn(model, probe, steering_vector, alpha, threshold, stats)
    handle = target_module.register_forward_hook(hook_fn)

    try:
        for sample in tqdm(test_samples, desc=f"FB [th={threshold} α={alpha}]"):
            prompt = _prompt_from_sample(sample)
            gate_before = stats["gate_triggers"]
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEST_TEMPERATURE if TEST_TEMPERATURE > 0 else 1.0,
                    do_sample=(TEST_TEMPERATURE > 0),
                    top_p=0.95 if TEST_TEMPERATURE > 0 else 1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated = outputs[0][inputs["input_ids"].shape[1]:]
            answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
            was_gated = stats["gate_triggers"] > gate_before
            results.append({
                "sample_id": sample.get("tid", "unknown"),
                "generated_output": answer,
                "is_sycophantic": _is_sycophantic(answer),
                "gate_triggered": was_gated,
            })
    finally:
        handle.remove()

    syc_rate = sum(1 for r in results if r["is_sycophantic"]) / len(results) if results else 0.0
    gate_rate = stats["gate_triggers"] / len(results) if results else 0.0

    return {
        "direction": direction_name, "alpha": alpha, "threshold": threshold,
        "n_samples": len(results), "syc_rate": syc_rate, "gate_rate": gate_rate,
        "probe_mean_score": float(np.mean(stats["probe_scores"])) if stats["probe_scores"] else 0.0,
        "probe_std": float(np.std(stats["probe_scores"])) if stats["probe_scores"] else 0.0,
        "probe_scores": stats["probe_scores"],
        "results": results,
    }


def analyze_probe_score_distribution(model, tokenizer, test_samples, probe):
    _log.info("Phase 2: Analyzing probe score distribution on test set")
    scores = []
    labels = []
    for sample in tqdm(test_samples, desc="Probe distribution"):
        prompt = _prompt_from_sample(sample)
        hs = _collect_last_token_hidden_state(model, tokenizer, prompt, STEERING_LAYER)
        score = float(probe.predict_proba(hs.reshape(1, -1))[0, 1])
        scores.append(score)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEST_TEMPERATURE if TEST_TEMPERATURE > 0 else 1.0,
                do_sample=(TEST_TEMPERATURE > 0),
                top_p=0.95 if TEST_TEMPERATURE > 0 else 1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
        labels.append(_is_sycophantic(answer))

    scores = np.array(scores)
    labels = np.array(labels)

    dist = {
        "mean": float(np.mean(scores)), "std": float(np.std(scores)),
        "min": float(np.min(scores)), "max": float(np.max(scores)),
        "p10": float(np.percentile(scores, 10)), "p20": float(np.percentile(scores, 20)),
        "p30": float(np.percentile(scores, 30)), "p40": float(np.percentile(scores, 40)),
        "p50": float(np.percentile(scores, 50)), "p60": float(np.percentile(scores, 60)),
        "p70": float(np.percentile(scores, 70)), "p80": float(np.percentile(scores, 80)),
        "p90": float(np.percentile(scores, 90)),
        "p20_value": float(np.percentile(scores, 80)),
        "p30_value": float(np.percentile(scores, 70)),
        "p40_value": float(np.percentile(scores, 60)),
        "scores": scores.tolist(),
        "labels": labels.tolist() if labels.dtype == bool else labels.astype(int).tolist(),
        "syc_mean_score": float(np.mean(scores[labels])) if labels.sum() > 0 else 0.0,
        "non_syc_mean_score": float(np.mean(scores[~labels])) if (~labels).sum() > 0 else 0.0,
    }

    _log.info(f"Score distribution: mean={dist['mean']:.4f}, std={dist['std']:.4f}")
    _log.info(f"  Range: [{dist['min']:.4f}, {dist['max']:.4f}]")
    _log.info(f"  Percentiles: p20={dist['p20']:.4f}, p30={dist['p30']:.4f}, p40={dist['p40']:.4f}, p50={dist['p50']:.4f}")
    _log.info(f"  Syc mean score: {dist['syc_mean_score']:.4f}, Non-syc mean score: {dist['non_syc_mean_score']:.4f}")

    if labels.sum() > 0 and (~labels).sum() > 0:
        separation = dist["syc_mean_score"] - dist["non_syc_mean_score"]
        _log.info(f"  Score separation (syc - non-syc): {separation:+.4f}")

    return dist


def run_open_loop_reference(model, tokenizer, test_samples, steering_vector, alpha):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    steering_tensor = torch.tensor(steering_vector, dtype=dtype, device=device)

    target_module = None
    for name, mod in model.named_modules():
        if name == f"model.layers.{STEERING_LAYER}":
            target_module = mod
            break

    handle = apply_steering_hook(model, STEERING_LAYER, steering_vector, alpha)

    results = []
    try:
        for sample in tqdm(test_samples, desc=f"OL [v_syc α={alpha}]"):
            prompt = _prompt_from_sample(sample)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEST_TEMPERATURE if TEST_TEMPERATURE > 0 else 1.0,
                    do_sample=(TEST_TEMPERATURE > 0),
                    top_p=0.95 if TEST_TEMPERATURE > 0 else 1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated = outputs[0][inputs["input_ids"].shape[1]:]
            answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
            results.append({
                "sample_id": sample.get("tid", "unknown"),
                "generated_output": answer,
                "is_sycophantic": _is_sycophantic(answer),
            })
    finally:
        if handle is not None:
            handle.remove()

    syc_rate = sum(1 for r in results if r["is_sycophantic"]) / len(results) if results else 0.0
    return {"direction": "v_syc", "alpha": alpha, "n_samples": len(results), "syc_rate": syc_rate}


def _generate_report(
    baseline_syc_rate: float,
    score_dist: dict,
    threshold_results: List[dict],
    percentile_results: List[dict],
    open_loop_result: dict,
    report_path: Path,
):
    lines = []
    lines.append("# IC-4 P6-bis: Threshold Calibration for Sycophancy Feedback Control")
    lines.append("")
    lines.append(f"> **Date**: {time.strftime('%Y-%m-%d')} | **Status**: Completed")
    lines.append(f"> **Predecessor**: P6 — Behavior-only probe (train acc=81.94%, test acc=77.78%)")
    lines.append(f"> **Layer**: {STEERING_LAYER} | **Alpha**: {OPTIMAL_ALPHA} | **Test samples**: {baseline_syc_rate}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Motivation")
    lines.append("")
    lines.append("P6's behavior-only probe trains successfully (78% test accuracy),")
    lines.append("but the gate rate is only 8.3% because probe scores cluster near 0.5")
    lines.append(f"(mean={score_dist['mean']:.4f}). The probe is correct but uncertain.")
    lines.append("")
    lines.append("P6-bis systematically tests lower thresholds and percentile-based")
    lines.append("gating to find the sweet spot where the feedback loop activates")
    lines.append("without sacrificing specificity.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Probe Score Distribution (Test Set)")
    lines.append("")
    lines.append(f"- Mean: **{score_dist['mean']:.4f}** | Std: **{score_dist['std']:.4f}**")
    lines.append(f"- Range: [{score_dist['min']:.4f}, {score_dist['max']:.4f}]")
    lines.append("")
    lines.append("| Percentile | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    lines.append(
        f"| Score | {score_dist['p10']:.4f} | {score_dist['p20']:.4f} | "
        f"{score_dist['p30']:.4f} | {score_dist['p40']:.4f} | {score_dist['p50']:.4f} | "
        f"{score_dist['p60']:.4f} | {score_dist['p70']:.4f} | {score_dist['p80']:.4f} | "
        f"{score_dist['p90']:.4f} |"
    )
    lines.append("")
    lines.append(f"- Syc mean score: **{score_dist['syc_mean_score']:.4f}**")
    lines.append(f"- Non-syc mean score: **{score_dist['non_syc_mean_score']:.4f}**")

    separation = score_dist["syc_mean_score"] - score_dist["non_syc_mean_score"]
    lines.append(f"- Score separation (syc − non-syc): **{separation:+.4f}**")
    lines.append("")

    if separation < 0:
        lines.append("**CRITICAL: Score separation is NEGATIVE.** Syc samples have LOWER")
        lines.append("probe scores than non-syc samples. This means the probe score direction")
        lines.append("is inverted — we need to gate on scores BELOW threshold, not ABOVE.")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 3. Threshold Sweep (Feedback Control, v_syc α=-3.0)")
    lines.append("")
    lines.append("")
    lines.append(f"| Threshold | Gate Rate | Syc Rate | Δ from Baseline | Probe μ | Probe σ |")
    lines.append("|---|---|---|---|---|---|")

    for tr in threshold_results:
        delta = tr["syc_rate"] - baseline_syc_rate if isinstance(baseline_syc_rate, float) else 0.0
        lines.append(
            f"| {tr['threshold']:.2f} | {tr['gate_rate']:.4f} | {tr['syc_rate']:.4f} | "
            f"{delta:+.4f} | {tr['probe_mean_score']:.4f} | {tr['probe_std']:.4f} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Percentile-Based Gating (Feedback Control, v_syc α=-3.0)")
    lines.append("")
    lines.append("")
    lines.append(f"| Gate Rule | Effective Threshold | Gate Rate | Syc Rate | Δ from Baseline |")
    lines.append("|---|---|---|---|---|")

    for pr in percentile_results:
        eff_thresh = pr.get("effective_threshold", "N/A")
        if isinstance(eff_thresh, float):
            eff_thresh_str = f"{eff_thresh:.4f}"
        else:
            eff_thresh_str = str(eff_thresh)
        delta = pr["syc_rate"] - baseline_syc_rate if isinstance(baseline_syc_rate, float) else 0.0
        lines.append(
            f"| {pr['gating_rule']} | {eff_thresh_str} | {pr['gate_rate']:.4f} | "
            f"{pr['syc_rate']:.4f} | {delta:+.4f} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. Open-Loop Reference")
    lines.append("")
    lines.append(f"| Direction | Syc Rate | Δ from Baseline |")
    lines.append("|---|---|")

    if open_loop_result:
        delta = open_loop_result["syc_rate"] - baseline_syc_rate if isinstance(baseline_syc_rate, float) else 0.0
        lines.append(f"| v_syc | {open_loop_result['syc_rate']:.4f} | {delta:+.4f} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6. Interpretation")
    lines.append("")

    best_fb = min([r for r in threshold_results], key=lambda x: x["syc_rate"]) if threshold_results else None
    best_pct = min([r for r in percentile_results], key=lambda x: x["syc_rate"]) if percentile_results else None

    all_feedback = threshold_results + percentile_results
    best_overall = min(all_feedback, key=lambda x: x["syc_rate"]) if all_feedback else None

    if best_overall:
        delta = best_overall["syc_rate"] - baseline_syc_rate if isinstance(baseline_syc_rate, float) else 0.0
        lines.append(f"**Best feedback result**: {best_overall.get('threshold', best_overall.get('gating_rule'))} "
                      f"→ syc={best_overall['syc_rate']:.4f} (Δ={delta:+.4f}), "
                      f"gate_rate={best_overall['gate_rate']:.4f}")
        lines.append("")

    if best_overall and best_overall["gate_rate"] > 0.2:
        lines.append("**Threshold calibration SUCCESS!** Lowering the threshold")
        lines.append("successfully activates the probe→gate→hook feedback loop.")
        lines.append("The behavior-only probe achieves meaningful closed-loop control.")
    elif best_overall and best_overall["gate_rate"] > 0.1:
        lines.append("**Partial success.** Gate rate improved slightly but remains low.")
        lines.append("Probe score separation is the fundamental bottleneck.")
        lines.append("Consider: k>5 training samples, higher T, or MLP probe.")
    else:
        lines.append("**Gate rate remains low across all thresholds.** The probe's")
        lines.append("score distribution is too concentrated near 0.5 for reliable")
        lines.append("binary gating. Fundamental options:")
        lines.append("1. Increase behavioral variation (k=10, T=0.9, or T=1.0)")
        lines.append("2. Use a more expressive probe (MLP, XGBoost, calibrated LR)")
        lines.append("3. Accept that T=0 generation doesn't benefit from gating")
        lines.append("   and use open-loop v_syc at α=−3.0 as the primary intervention")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 7. Next Steps")
    lines.append("")
    lines.append("| Priority | Action | Detail |")
    lines.append("|---|---|---|")
    lines.append("| **P7** | S15 Amplification | Investigate syc signal amplification at gen step 15 |")
    lines.append("| P8 | More Training Data | k=10, T=0.9 for better behavioral separation |")
    lines.append("| P9 | MLP Probe | Non-linear probe for better score separation |")
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    _log.info(f"Report: {report_path}")


def main():
    _log.info("P6-bis: Threshold Calibration for Syc Feedback Control")

    _log.info("Phase 1: Loading existing assets")

    _log.info("Loading contrast data...")
    with open(CONTRAST_DATA_PATH, "r", encoding="utf-8") as f:
        contrast_data = json.load(f)

    standard_samples = [s for s in contrast_data if not s.get("system_prompt")]
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(len(standard_samples))
    test_idx = indices[18:30]
    test_samples = [standard_samples[i] for i in test_idx]
    _log.info(f"Test samples: {len(test_samples)}")

    _log.info("Loading P6 probe...")
    with open(P6_PROBE_PATH, "rb") as f:
        probe = pickle.load(f)

    _log.info("Loading P6 baseline reference...")
    p6_data = np.load(P6_RESULTS_PATH, allow_pickle=True)
    baseline_syc_rate = float(p6_data["baseline_syc_rate"])
    _log.info(f"P6 baseline syc rate: {baseline_syc_rate:.4f}")

    _log.info("Loading model...")
    model, tokenizer = load_model_and_tokenizer()

    _log.info("Loading steering vectors...")
    sv_data = np.load(STEERING_VECTORS_PATH)
    v_syc = sv_data["v_syc"].astype(np.float32)

    score_dist = analyze_probe_score_distribution(model, tokenizer, test_samples, probe)

    _log.info("Phase 3: Threshold sweep")
    threshold_results = []
    for threshold in THRESHOLD_SWEEP:
        _log.info(f"Threshold={threshold:.2f}")
        fr = run_feedback_generation(
            model, tokenizer, test_samples, probe,
            v_syc, OPTIMAL_ALPHA, "v_syc", threshold,
        )
        _log.info(f"  syc_rate={fr['syc_rate']:.4f}, gate_rate={fr['gate_rate']:.4f}")
        threshold_results.append(fr)

    _log.info("Phase 4: Percentile-based gating")
    percentile_results = []

    score_thresholds = {
        "top-20%": float(np.percentile(score_dist["scores"], 80)),
        "top-30%": float(np.percentile(score_dist["scores"], 70)),
        "top-40%": float(np.percentile(score_dist["scores"], 60)),
    }

    for gating_rule, eff_threshold in score_thresholds.items():
        _log.info(f"Percentile gating: {gating_rule} (threshold={eff_threshold:.4f})")
        fr = run_feedback_generation(
            model, tokenizer, test_samples, probe,
            v_syc, OPTIMAL_ALPHA, "v_syc", eff_threshold,
        )
        fr["gating_rule"] = gating_rule
        fr["effective_threshold"] = eff_threshold
        _log.info(f"  syc_rate={fr['syc_rate']:.4f}, gate_rate={fr['gate_rate']:.4f}")
        percentile_results.append(fr)

    _log.info("Phase 5: Open-loop reference")
    open_loop_result = run_open_loop_reference(
        model, tokenizer, test_samples, v_syc, OPTIMAL_ALPHA,
    )
    _log.info(f"  Open-loop v_syc syc_rate={open_loop_result['syc_rate']:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        RESULTS_DIR / "threshold_sweep_results.npz",
        baseline_syc_rate=baseline_syc_rate,
        threshold_results=[{k: v for k, v in r.items() if k != "results" and k != "probe_scores"}
                           for r in threshold_results],
        percentile_results=[{k: v for k, v in r.items() if k != "results" and k != "probe_scores"}
                            for r in percentile_results],
        open_loop_result=open_loop_result,
        score_distribution=score_dist,
        allow_pickle=True,
    )

    reduced_threshold_results = [
        {k: v for k, v in r.items() if k not in ("results", "probe_scores")}
        for r in threshold_results
    ]
    reduced_percentile_results = [
        {k: v for k, v in r.items() if k not in ("results", "probe_scores")}
        for r in percentile_results
    ]

    report_path = REPORTS_DIR / "IC4_P6_BIS_THRESHOLD_REPORT.md"
    _generate_report(
        baseline_syc_rate, score_dist,
        threshold_results, percentile_results, open_loop_result,
        report_path,
    )

    _log.info("P6-bis complete.")


if __name__ == "__main__":
    main()