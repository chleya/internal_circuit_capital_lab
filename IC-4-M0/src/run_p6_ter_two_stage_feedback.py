"""P6-ter: Two-Stage Feedback Control for Sycophancy.

P6-bis diagnosed that the probe->gate->hook pipeline fails because the hook
inside model.generate() captures generated-token hidden states, not
prompt-token states. The probe was trained on prompt-token states only.

P6-ter fixes this with a two-stage architecture:

  Stage 1: model(**inputs) -> collect L10 last_prompt_token hs -> probe score
  Stage 2: if score >= threshold -> run model.generate() WITH steering hook
           else -> run model.generate() WITHOUT steering

This eliminates token-type pollution: the probe always sees prompt-token
states (what it was trained on), and the steering hook is a simple
always-on injection (no probe scoring during generation).

Design:
  Phase 1: Load P6 probe and test data
  Phase 2: Two-stage feedback at thresholds [0.30, 0.40, 0.50, 0.60, 0.70]
           with v_syc alpha=-3.0
  Phase 3: Open-loop comparison (always-on v_syc alpha=-3.0)
  Phase 4: Control: two-stage with random vector (verify direction specificity)
  Phase 5: Report generation

Usage:
  python -m src.run_p6_ter_two_stage_feedback

Outputs:
  results_p6_ter_two_stage/
    two_stage_results.npz
  reports/IC4_P6_TER_TWO_STAGE_REPORT.md
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from .model_loader import load_model_and_tokenizer
from .steering import _make_steering_hook, _find_transformer_layer
from .run_p0_sycophancy_contrast import _is_sycophantic

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_log = logging.getLogger("p6_ter_two_stage")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results_p6_ter_two_stage"
REPORTS_DIR = PROJECT_ROOT / "reports"
CONTRAST_DATA_PATH = PROJECT_ROOT / "results_p0_sycophancy" / "sycophancy_contrast_data.json"
STEERING_VECTORS_PATH = PROJECT_ROOT / "results_t3_impulse_p4" / "steering_vectors.npz"
P6_PROBE_PATH = PROJECT_ROOT / "results_p6_syc_behavior" / "probe_model.pkl"

STEERING_LAYER = 10
OPTIMAL_ALPHA = -3.0
MAX_NEW_TOKENS = 128
RANDOM_SEED = 42

TWO_STAGE_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]


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

    target_module = _find_transformer_layer(model, STEERING_LAYER)
    handle = target_module.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        handle.remove()

    hs = hidden_states[0]
    return float(probe.predict_proba(hs)[0, 1])


def _run_generate_with_steering(model, tokenizer, prompt: str, steering_vector, alpha):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    hook_fn = _make_steering_hook(steering_vector, alpha, device)
    target_module = _find_transformer_layer(model, STEERING_LAYER)
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
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return answer


def _run_generate_no_steering(model, tokenizer, prompt: str):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return answer


def run_two_stage_feedback(
    model, tokenizer, test_samples, probe, steering_vector, alpha, threshold,
):
    results = []
    n_gated = 0
    scores = []

    for sample in tqdm(test_samples, desc=f"TwoStage [th={threshold:.2f}]"):
        prompt = _prompt_from_sample(sample)
        score = _get_probe_score(model, tokenizer, prompt, probe)
        scores.append(score)

        if score >= threshold:
            n_gated += 1
            answer = _run_generate_with_steering(model, tokenizer, prompt, steering_vector, alpha)
        else:
            answer = _run_generate_no_steering(model, tokenizer, prompt)

        results.append({
            "sample_id": sample.get("tid", "unknown"),
            "generated_output": answer,
            "is_sycophantic": _is_sycophantic(answer),
            "probe_score": score,
            "gated": score >= threshold,
        })

    syc_rate = sum(1 for r in results if r["is_sycophantic"]) / len(results) if results else 0.0
    gate_rate = n_gated / len(results) if results else 0.0

    return {
        "threshold": threshold, "alpha": alpha, "n_samples": len(results),
        "syc_rate": syc_rate, "gate_rate": gate_rate,
        "n_gated": n_gated, "probe_mean_score": float(np.mean(scores)),
        "probe_min": float(np.min(scores)), "probe_max": float(np.max(scores)),
        "scores": scores, "results": results,
    }


def run_open_loop(model, tokenizer, test_samples, steering_vector, alpha):
    results = []
    for sample in tqdm(test_samples, desc=f"OpenLoop [alpha={alpha}]"):
        prompt = _prompt_from_sample(sample)
        answer = _run_generate_with_steering(model, tokenizer, prompt, steering_vector, alpha)
        results.append({
            "sample_id": sample.get("tid", "unknown"),
            "generated_output": answer,
            "is_sycophantic": _is_sycophantic(answer),
        })

    syc_rate = sum(1 for r in results if r["is_sycophantic"]) / len(results) if results else 0.0
    return {"direction": "v_syc", "alpha": alpha, "n_samples": len(results), "syc_rate": syc_rate}


def run_baseline(model, tokenizer, test_samples):
    results = []
    for sample in tqdm(test_samples, desc="Baseline"):
        prompt = _prompt_from_sample(sample)
        answer = _run_generate_no_steering(model, tokenizer, prompt)
        results.append({
            "sample_id": sample.get("tid", "unknown"),
            "generated_output": answer,
            "is_sycophantic": _is_sycophantic(answer),
        })

    syc_rate = sum(1 for r in results if r["is_sycophantic"]) / len(results) if results else 0.0
    return {"direction": "none", "n_samples": len(results), "syc_rate": syc_rate}


def _generate_report(
    baseline_syc_rate: float,
    two_stage_results: List[dict],
    open_loop_result: dict,
    random_two_stage_result: dict,
    report_path: Path,
):
    lines = []
    lines.append("# IC-4 P6-ter: Two-Stage Feedback Control for Sycophancy")
    lines.append("")
    lines.append(f"> **Date**: {time.strftime('%Y-%m-%d')} | **Status**: Completed")
    lines.append(f"> **Predecessor**: P6-bis — Hook architecture diagnostic")
    lines.append(f"> **Layer**: {STEERING_LAYER} | **Alpha**: {OPTIMAL_ALPHA}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Motivation")
    lines.append("")
    lines.append("P6-bis diagnosed the root cause of feedback control failure: the")
    lines.append("probe->gate->hook fires inside model.generate(), where hs[:, -1, :]")
    lines.append("captures generated-token states on decode steps. The probe was trained")
    lines.append("exclusively on prompt-token states.")
    lines.append("")
    lines.append("P6-ter fixes this with a **two-stage architecture**:")
    lines.append("")
    lines.append("```")
    lines.append("Stage 1: model(**inputs) -> L10 last_prompt_token hs -> probe score")
    lines.append("Stage 2: if score >= threshold -> model.generate() WITH steering")
    lines.append("         else -> model.generate() WITHOUT steering")
    lines.append("```")
    lines.append("")
    lines.append("The probe always sees prompt-token states (its training distribution).")
    lines.append("The steering is a simple always-on hook during generation.")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 2. Baseline")
    lines.append("")
    lines.append(f"- Baseline syc rate: **{baseline_syc_rate:.4f}**")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 3. Two-Stage Feedback Control (v_syc alpha=-3.0)")
    lines.append("")
    lines.append("")
    lines.append(f"| Threshold | Gate Rate | N Gated | Syc Rate | Delta from Baseline | Probe mu |")
    lines.append("|---|---|---|---|---|---|")

    for tr in two_stage_results:
        delta = tr["syc_rate"] - baseline_syc_rate
        delta_pct = delta / baseline_syc_rate * 100 if baseline_syc_rate > 0 else 0.0
        lines.append(
            f"| {tr['threshold']:.2f} | {tr['gate_rate']:.4f} | "
            f"{tr['n_gated']}/{tr['n_samples']} | {tr['syc_rate']:.4f} | "
            f"{delta:+.4f} ({delta_pct:+.1f}%) | {tr['probe_mean_score']:.4f} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Open-Loop Comparison")
    lines.append("")
    lines.append(f"| Direction | Alpha | Syc Rate | Delta from Baseline |")
    lines.append("|---|---|---|---|")

    delta_ol = open_loop_result["syc_rate"] - baseline_syc_rate
    delta_ol_pct = delta_ol / baseline_syc_rate * 100 if baseline_syc_rate > 0 else 0.0
    lines.append(
        f"| v_syc | {open_loop_result['alpha']} | {open_loop_result['syc_rate']:.4f} | "
        f"{delta_ol:+.4f} ({delta_ol_pct:+.1f}%) |"
    )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. Two-Stage with Random Vector (Control)")
    lines.append("")
    lines.append(f"| Threshold | Gate Rate | Syc Rate | Delta from Baseline |")
    lines.append("|---|---|---|---|")

    if random_two_stage_result:
        delta_r = random_two_stage_result["syc_rate"] - baseline_syc_rate
        delta_r_pct = delta_r / baseline_syc_rate * 100 if baseline_syc_rate > 0 else 0.0
        lines.append(
            f"| {random_two_stage_result['threshold']:.2f} | "
            f"{random_two_stage_result['gate_rate']:.4f} | "
            f"{random_two_stage_result['syc_rate']:.4f} | "
            f"{delta_r:+.4f} ({delta_r_pct:+.1f}%) |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6. Interpretation")
    lines.append("")

    best_ts = min(two_stage_results, key=lambda x: x["syc_rate"]) if two_stage_results else None
    if best_ts:
        delta_ts = best_ts["syc_rate"] - baseline_syc_rate
        delta_ts_pct = delta_ts / baseline_syc_rate * 100 if baseline_syc_rate > 0 else 0.0
        lines.append(f"**Best two-stage**: th={best_ts['threshold']:.2f} → "
                      f"syc={best_ts['syc_rate']:.4f} "
                      f"(Δ={delta_ts:+.4f}, {delta_ts_pct:+.1f}%), "
                      f"gate_rate={best_ts['gate_rate']:.4f} "
                      f"({best_ts['n_gated']}/{best_ts['n_samples']} gated)")
        lines.append("")
        lines.append(f"**Open-loop**: syc={open_loop_result['syc_rate']:.4f} "
                      f"(Δ={delta_ol:+.4f}, {delta_ol_pct:+.1f}%)")
        lines.append("")

        if delta_ts < -0.1:
            lines.append("**TWO-STAGE FEEDBACK CONTROL WORKS!** The closed-loop")
            lines.append("probe→gate→hook pipeline successfully reduces sycophancy")
            lines.append("through conditional steering based on behavioral probe scores.")
            lines.append("")
        elif delta_ts < -0.05:
            lines.append("**Partial success.** Two-stage shows modest syc reduction.")
            lines.append("The architecture fix is directionally correct. Further tuning")
            lines.append("of threshold or using more training data may improve effect size.")
            lines.append("")
        else:
            lines.append("**Two-stage feedback still insufficient.** Even with correct")
            lines.append("probe scoring (standalone forward pass), the feedback effect")
            lines.append("is minimal. Possible issues:")
            lines.append("- Threshold may need further calibration")
            lines.append("- Probe test accuracy (77.8%) may not be sufficient for fine-grained gating")
            lines.append("- Open-loop at alpha=-3.0 may be the optimal syc intervention")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 7. Summary")
    lines.append("")

    all_below_openloop = all(
        r["syc_rate"] >= open_loop_result["syc_rate"] for r in two_stage_results
    ) if two_stage_results else False

    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Baseline syc rate | {baseline_syc_rate:.4f} |")
    lines.append(f"| Open-loop v_syc | {open_loop_result['syc_rate']:.4f} |")
    if best_ts:
        lines.append(f"| Best two-stage | {best_ts['syc_rate']:.4f} (th={best_ts['threshold']:.2f}) |")
        lines.append(f"| Best two-stage gate rate | {best_ts['gate_rate']:.4f} |")
    lines.append("")

    if all_below_openloop:
        lines.append("**Dependency note**: All two-stage results are at or above open-loop.")
        lines.append("This means open-loop alpha=-3.0 is the floor that conditional gating")
        lines.append("cannot beat — gating can only REDUCE steering (by not steering some samples),")
        lines.append("so syc_rate will be between baseline and open-loop. The value of two-stage")
        lines.append("feedback is in selectively steering only high-risk samples while preserving")
        lines.append("natural behavior on low-risk samples.")
        lines.append("")
    else:
        lines.append("Two-stage feedback achieves syc reduction comparable to or better than")
        lines.append("open-loop. This validates the probe→gate→hook paradigm for selective")
        lines.append("behavioral intervention.")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 8. Next Steps")
    lines.append("")
    lines.append("| Priority | Action | Detail |")
    lines.append("|---|---|---|")
    lines.append("| **P7** | S15 Amplification | Investigate syc signal amplification at gen step 15 |")
    lines.append("| P8 | More Training Data | k=10, T=0.9 for probe score separation improvement |")
    lines.append("| P9 | Cross-Bottleneck | Combine stabilization + organization interventions |")
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    _log.info(f"Report: {report_path}")


def main():
    _log.info("P6-ter: Two-Stage Feedback Control")

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

    _log.info("Loading model...")
    model, tokenizer = load_model_and_tokenizer()

    _log.info("Loading steering vectors...")
    sv_data = np.load(STEERING_VECTORS_PATH)
    v_syc = sv_data["v_syc"].astype(np.float32)
    v_random = sv_data["random"].astype(np.float32)

    _log.info("Phase 1: Baseline")
    baseline_result = run_baseline(model, tokenizer, test_samples)
    baseline_syc_rate = baseline_result["syc_rate"]
    _log.info(f"Baseline syc rate: {baseline_syc_rate:.4f}")

    _log.info("Phase 2: Two-stage feedback at multiple thresholds")
    two_stage_results = []
    for threshold in TWO_STAGE_THRESHOLDS:
        _log.info(f"Two-stage: th={threshold:.2f}")
        tsr = run_two_stage_feedback(
            model, tokenizer, test_samples, probe, v_syc, OPTIMAL_ALPHA, threshold,
        )
        delta = tsr["syc_rate"] - baseline_syc_rate
        _log.info(f"  syc_rate={tsr['syc_rate']:.4f} (delta={delta:+.4f}), "
                   f"gate_rate={tsr['gate_rate']:.4f} ({tsr['n_gated']}/{tsr['n_samples']} gated), "
                   f"probe_mu={tsr['probe_mean_score']:.4f}")
        two_stage_results.append(tsr)

    _log.info("Phase 3: Open-loop comparison")
    open_loop_result = run_open_loop(model, tokenizer, test_samples, v_syc, OPTIMAL_ALPHA)
    delta_ol = open_loop_result["syc_rate"] - baseline_syc_rate
    _log.info(f"Open-loop syc_rate={open_loop_result['syc_rate']:.4f} (delta={delta_ol:+.4f})")

    _log.info("Phase 4: Two-stage with random vector (control, th=0.50)")
    random_tsr = run_two_stage_feedback(
        model, tokenizer, test_samples, probe, v_random, OPTIMAL_ALPHA, 0.50,
    )
    delta_r = random_tsr["syc_rate"] - baseline_syc_rate
    _log.info(f"Random two-stage syc_rate={random_tsr['syc_rate']:.4f} (delta={delta_r:+.4f}), "
               f"gate_rate={random_tsr['gate_rate']:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        RESULTS_DIR / "two_stage_results.npz",
        baseline_syc_rate=baseline_syc_rate,
        two_stage_results=[{k: v for k, v in r.items() if k not in ("results", "scores")}
                           for r in two_stage_results],
        open_loop_result=open_loop_result,
        random_two_stage_result={k: v for k, v in random_tsr.items()
                                 if k not in ("results", "scores")},
        allow_pickle=True,
    )

    report_path = REPORTS_DIR / "IC4_P6_TER_TWO_STAGE_REPORT.md"
    _generate_report(
        baseline_syc_rate, two_stage_results, open_loop_result, random_tsr, report_path,
    )

    _log.info("P6-ter complete.")


if __name__ == "__main__":
    main()