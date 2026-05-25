"""P8: Larger-Scale Replication of Two-Stage Feedback Control.

P6-ter achieved -66.7% sycophancy reduction on n=12 (th=0.50).
P8 verifies statistical robustness on n=24 samples.

Design:
  Phase 1: Baseline (24 samples, no steering)
  Phase 2: Two-stage feedback at th=0.50 (P6-ter optimal)
  Phase 3: Two-stage feedback at th=0.40 (P6-ter second best)
  Phase 4: Open-loop (always-on v_syc alpha=-3.0)

Usage:
  python -m src.run_p8_large_scale_replication

Outputs:
  results_p8_large_scale/
    p8_results.npz
  reports/IC4_P8_LARGE_SCALE_REPORT.md
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from .model_loader import load_model_and_tokenizer
from .steering import _make_steering_hook, _find_transformer_layer
from .run_p0_sycophancy_contrast import _is_sycophantic

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_log = logging.getLogger("p8_large_scale")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results_p8_large_scale"
REPORTS_DIR = PROJECT_ROOT / "reports"
CONTRAST_DATA_PATH = PROJECT_ROOT / "results_p0_sycophancy" / "sycophancy_contrast_data.json"
P6_PROBE_PATH = PROJECT_ROOT / "results_p6_syc_behavior" / "probe_model.pkl"
STEERING_VECTORS_PATH = PROJECT_ROOT / "results_t3_impulse_p4" / "steering_vectors.npz"

PROBE_LAYER = 10
OPTIMAL_ALPHA = -3.0
MAX_NEW_TOKENS = 128
RANDOM_SEED = 42
N_TEST = 24
THRESHOLDS = [0.40, 0.50]


def _prompt_from_sample(sample: dict) -> str:
    prompt = sample.get("prompt", "")
    if not prompt:
        context = sample.get("context", "")
        question = sample.get("question", "")
        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
    return prompt


def _get_probe_score(model, tokenizer, prompt: str, probe) -> float:
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

    target_module = _find_transformer_layer(model, PROBE_LAYER)
    handle = target_module.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        handle.remove()
    return float(probe.predict_proba(hidden_states[0])[0, 1])


def _generate_with_steering(model, tokenizer, prompt: str, steering_vector, alpha):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    hook_fn = _make_steering_hook(steering_vector, alpha, device)
    target_module = _find_transformer_layer(model, PROBE_LAYER)
    handle = target_module.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        handle.remove()
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _generate_no_steering(model, tokenizer, prompt: str):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def run_condition(
    model, tokenizer, test_samples, probe, steering_vector, alpha,
    condition_type, threshold=None, direction_name="v_syc",
):
    results = []
    scores = []
    n_gated = 0

    desc = f"{condition_type}"
    if threshold is not None:
        desc += f" [th={threshold}]"

    for sample in tqdm(test_samples, desc=desc):
        prompt = _prompt_from_sample(sample)

        if condition_type == "baseline":
            answer = _generate_no_steering(model, tokenizer, prompt)
            gated = False
        elif condition_type == "open_loop":
            answer = _generate_with_steering(model, tokenizer, prompt, steering_vector, alpha)
            gated = True
        elif condition_type == "two_stage":
            score = _get_probe_score(model, tokenizer, prompt, probe)
            scores.append(score)
            if score >= threshold:
                n_gated += 1
                answer = _generate_with_steering(model, tokenizer, prompt, steering_vector, alpha)
                gated = True
            else:
                answer = _generate_no_steering(model, tokenizer, prompt)
                gated = False
        else:
            raise ValueError(f"Unknown condition: {condition_type}")

        results.append({
            "sample_id": sample.get("tid", "unknown"),
            "generated_output": answer,
            "is_sycophantic": _is_sycophantic(answer),
            "gated": gated,
        })

    syc_rate = sum(1 for r in results if r["is_sycophantic"]) / len(results) if results else 0.0
    gate_rate = n_gated / len(results) if (condition_type == "two_stage" and results) else 0.0

    return {
        "condition": condition_type,
        "threshold": threshold,
        "direction": direction_name,
        "alpha": alpha if condition_type != "baseline" else 0.0,
        "n_samples": len(results),
        "syc_rate": syc_rate,
        "gate_rate": gate_rate,
        "n_gated": n_gated,
        "probe_mean_score": float(np.mean(scores)) if scores else None,
        "results": results,
    }


def _generate_report(
    baseline: dict, two_stage_50: dict, two_stage_40: dict, open_loop: dict, report_path: Path,
):
    lines = []
    lines.append("# IC-4 P8: Larger-Scale Replication of Two-Stage Feedback Control")
    lines.append("")
    lines.append(f"> **Date**: {time.strftime('%Y-%m-%d')} | **Status**: Completed")
    lines.append(f"> **Predecessor**: P6-ter Two-Stage Feedback (n=12, −66.7% at th=0.50)")
    lines.append(f"> **Samples**: {N_TEST} | **Layer**: {PROBE_LAYER} | **Alpha**: {OPTIMAL_ALPHA}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Motivation")
    lines.append("")
    lines.append("P6-ter achieved −66.7% sycophancy reduction on n=12 (th=0.50), but")
    lines.append("small sample sizes can produce spurious effects. P8 replicates the")
    lines.append(f"two-stage architecture on n={N_TEST} samples to verify statistical robustness.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Results")
    lines.append("")
    lines.append(f"| Condition | N | Syc Rate | Syc Count | Δ vs Baseline | Gate Rate |")
    lines.append("|---|---|---|---|---|---|")

    baseline_syc = baseline["syc_rate"]

    for cond in [baseline, two_stage_50, two_stage_40, open_loop]:
        n_syc = sum(1 for r in cond["results"] if r["is_sycophantic"]) if "results" in cond else "?"
        delta = cond["syc_rate"] - baseline_syc
        delta_pct = delta / baseline_syc * 100 if baseline_syc > 0 else 0.0
        gate_str = f"{cond['gate_rate']:.4f}" if cond.get("gate_rate") else "—"
        probe_str = ""
        if cond.get("probe_mean_score") is not None:
            probe_str = f" probe_mu={cond['probe_mean_score']:.4f}"
        lines.append(
            f"| {cond['condition']}"
            + (f" th={cond['threshold']:.2f}" if cond.get('threshold') else "")
            + f" | {cond['n_samples']} | {cond['syc_rate']:.4f} | {n_syc}/{cond['n_samples']} | "
            f"{delta:+.4f} ({delta_pct:+.1f}%) | {gate_str}{probe_str} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Comparison with P6-ter (n=12)")
    lines.append("")
    lines.append(f"| Metric | P6-ter (n=12) | P8 (n={N_TEST}) | Change |")
    lines.append("|---|---|---|---|")

    p6_baseline = 0.7500
    p6_ts50 = 0.2500
    p6_ts40 = 0.3333
    p6_ol = 0.4167

    lines.append(f"| Baseline syc | {p6_baseline:.4f} | {baseline_syc:.4f} | "
                  f"{(baseline_syc - p6_baseline):+.4f} |")
    lines.append(f"| Two-stage th=0.50 | {p6_ts50:.4f} (−66.7%) | {two_stage_50['syc_rate']:.4f} "
                  f"({(two_stage_50['syc_rate']-baseline_syc)/baseline_syc*100:+.1f}%) | "
                  f"{(two_stage_50['syc_rate'] - p6_ts50):+.4f} |")
    lines.append(f"| Two-stage th=0.40 | {p6_ts40:.4f} (−55.6%) | {two_stage_40['syc_rate']:.4f} "
                  f"({(two_stage_40['syc_rate']-baseline_syc)/baseline_syc*100:+.1f}%) | "
                  f"{(two_stage_40['syc_rate'] - p6_ts40):+.4f} |")
    lines.append(f"| Open-loop | {p6_ol:.4f} (−44.4%) | {open_loop['syc_rate']:.4f} "
                  f"({(open_loop['syc_rate']-baseline_syc)/baseline_syc*100:+.1f}%) | "
                  f"{(open_loop['syc_rate'] - p6_ol):+.4f} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Statistical Significance")
    lines.append("")

    n_b = baseline["n_samples"]
    n_ts = two_stage_50["n_samples"]

    from scipy.stats import chi2_contingency
    from scipy import stats as scipy_stats

    def _fisher_test(r1, r2):
        n_syc_b = sum(1 for r in r1 if r["is_sycophantic"])
        n_nonsyc_b = len(r1) - n_syc_b
        n_syc_ts = sum(1 for r in r2 if r["is_sycophantic"])
        n_nonsyc_ts = len(r2) - n_syc_ts
        table = [[n_syc_b, n_nonsyc_b], [n_syc_ts, n_nonsyc_ts]]
        try:
            _, p = scipy_stats.fisher_exact(table)
            return p
        except Exception:
            return None

    p_baseline_vs_ts50 = _fisher_test(baseline["results"], two_stage_50["results"])
    p_baseline_vs_ol = _fisher_test(baseline["results"], open_loop["results"])
    p_ts50_vs_ol = _fisher_test(two_stage_50["results"], open_loop["results"])

    lines.append(f"| Comparison | Fisher p-value | Significant (p<0.05)? |")
    lines.append("|---|---|---|")
    lines.append(f"| Baseline vs Two-Stage th=0.50 | {p_baseline_vs_ts50:.4f} | "
                  f"{'YES' if p_baseline_vs_ts50 and p_baseline_vs_ts50 < 0.05 else 'no'} |")
    lines.append(f"| Baseline vs Open-Loop | {p_baseline_vs_ol:.4f} | "
                  f"{'YES' if p_baseline_vs_ol and p_baseline_vs_ol < 0.05 else 'no'} |")
    lines.append(f"| Two-Stage th=0.50 vs Open-Loop | {p_ts50_vs_ol:.4f} | "
                  f"{'YES' if p_ts50_vs_ol and p_ts50_vs_ol < 0.05 else 'no'} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. Interpretation")
    lines.append("")

    ts50_delta = two_stage_50["syc_rate"] - baseline_syc
    ts50_pct = ts50_delta / baseline_syc * 100 if baseline_syc > 0 else 0.0
    ol_delta = open_loop["syc_rate"] - baseline_syc
    ol_pct = ol_delta / baseline_syc * 100 if baseline_syc > 0 else 0.0

    lines.append(f"**Two-stage feedback at th=0.50**: "
                  f"syc={two_stage_50['syc_rate']:.4f} ({ts50_pct:+.1f}%), "
                  f"gate_rate={two_stage_50['gate_rate']:.4f} "
                  f"({two_stage_50['n_gated']}/{two_stage_50['n_samples']} gated)")
    lines.append(f"**Open-loop**: syc={open_loop['syc_rate']:.4f} ({ol_pct:+.1f}%)")
    lines.append("")

    if ts50_delta < ol_delta:
        lines.append("Two-stage feedback BEATS open-loop — replicating the P6-ter finding")
        lines.append(f"at larger scale (n={N_TEST}). Selective intervention > universal intervention.")
    elif ts50_delta < -0.1:
        lines.append("Two-stage feedback achieves substantial syc reduction but does not")
        lines.append("beat open-loop at this scale. The selective intervention pattern is")
        lines.append("directionally correct but the effect size differs from P6-ter.")
    else:
        lines.append("Two-stage feedback effect is modest at this scale. The P6-ter −66.7%")
        lines.append("may have been inflated by small sample variance. The two-stage approach")
        lines.append("remains directionally correct but effect size needs calibration.")

    lines.append("")
    if p_baseline_vs_ts50 and p_baseline_vs_ts50 < 0.05:
        lines.append(f"**Statistically significant** (Fisher p={p_baseline_vs_ts50:.4f} < 0.05).")
        lines.append("Two-stage feedback produces a significant reduction in sycophancy")
        lines.append(f"compared to baseline on n={N_TEST} samples.")
    elif p_baseline_vs_ts50:
        lines.append(f"**Not statistically significant** (Fisher p={p_baseline_vs_ts50:.4f}).")
        lines.append("The effect direction is correct but does not reach p<0.05 at this")
        lines.append(f"sample size. Larger n (48+) may be needed for significance.")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6. Next Steps")
    lines.append("")
    lines.append("| Priority | Action | Detail |")
    lines.append("|---|---|---|")
    lines.append("| P9 | Cross-Bottleneck | stabilization + organization joint intervention |")
    lines.append("| P10 | Hallucination abandon | Formalize exclusion of single-direction impulse |")
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    _log.info(f"Report: {report_path}")


def main():
    _log.info(f"P8: Larger-Scale Replication (n={N_TEST})")

    _log.info("Loading contrast data...")
    with open(CONTRAST_DATA_PATH, "r", encoding="utf-8") as f:
        contrast_data = json.load(f)

    standard_samples = [s for s in contrast_data if not s.get("system_prompt")]
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(len(standard_samples))
    test_idx = indices[18:18 + N_TEST]
    test_samples = [standard_samples[i] for i in test_idx]
    _log.info(f"Test samples: {len(test_samples)}")

    _log.info("Loading P6 probe...")
    with open(P6_PROBE_PATH, "rb") as f:
        probe = pickle.load(f)

    _log.info("Loading model and steering vectors...")
    model, tokenizer = load_model_and_tokenizer()
    sv_data = np.load(STEERING_VECTORS_PATH)
    v_syc = sv_data["v_syc"].astype(np.float32)

    _log.info("Phase 1: Baseline")
    baseline = run_condition(model, tokenizer, test_samples, probe, v_syc,
                             OPTIMAL_ALPHA, "baseline")
    _log.info(f"Baseline: syc={baseline['syc_rate']:.4f} "
               f"({sum(1 for r in baseline['results'] if r['is_sycophantic'])}/{baseline['n_samples']})")

    _log.info("Phase 2: Two-stage th=0.50")
    two_stage_50 = run_condition(model, tokenizer, test_samples, probe, v_syc,
                                  OPTIMAL_ALPHA, "two_stage", threshold=0.50)
    _log.info(f"TwoStage th=0.50: syc={two_stage_50['syc_rate']:.4f}, "
               f"gate={two_stage_50['gate_rate']:.4f} ({two_stage_50['n_gated']}/{two_stage_50['n_samples']})")

    _log.info("Phase 3: Two-stage th=0.40")
    two_stage_40 = run_condition(model, tokenizer, test_samples, probe, v_syc,
                                  OPTIMAL_ALPHA, "two_stage", threshold=0.40)
    _log.info(f"TwoStage th=0.40: syc={two_stage_40['syc_rate']:.4f}, "
               f"gate={two_stage_40['gate_rate']:.4f} ({two_stage_40['n_gated']}/{two_stage_40['n_samples']})")

    _log.info("Phase 4: Open-loop")
    open_loop = run_condition(model, tokenizer, test_samples, probe, v_syc,
                               OPTIMAL_ALPHA, "open_loop")
    _log.info(f"OpenLoop: syc={open_loop['syc_rate']:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        RESULTS_DIR / "p8_results.npz",
        baseline_syc_rate=baseline["syc_rate"],
        two_stage_50_syc_rate=two_stage_50["syc_rate"],
        two_stage_40_syc_rate=two_stage_40["syc_rate"],
        open_loop_syc_rate=open_loop["syc_rate"],
        n_test=N_TEST,
        n_gated_50=two_stage_50["n_gated"],
        n_gated_40=two_stage_40["n_gated"],
        allow_pickle=True,
    )

    report_path = REPORTS_DIR / "IC4_P8_LARGE_SCALE_REPORT.md"
    _generate_report(baseline, two_stage_50, two_stage_40, open_loop, report_path)

    _log.info("P8 complete.")


if __name__ == "__main__":
    main()