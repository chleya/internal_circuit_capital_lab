""" P6: Sycophancy Feedback Control — Behavior-Only Probe.

P5's probe→gate→hook failed because the probe learned group membership
(fact_checker prompt present vs absent) rather than behavioral tendency.
P5-bis confirmed α=-3.0 is optimal for syc reduction.

P6 fixes the probe: train on standard-prompt samples only, use T>0
to create behavioral variation (some outputs syc, some non-syc).

Design:
  Phase 1: Behavior-labeled data generation
    - Take standard-prompt samples only (no fact_checker)
    - Generate k=5 outputs per sample at temperature=0.7
    - Each output gets behavioral label (is_sycophantic)
    - Collect L10 last_prompt_token hidden states

  Phase 2: Probe training
    - Train sklearn LogisticRegression on hidden_states → behavioral labels
    - Split 80/20 for train/test evaluation
    - Report: train acc, test acc, class balance

  Phase 3: Feedback control evaluation (α=-3.0)
    - Probe→gate→hook at L10: if prob>=threshold, apply v_syc at α=-3.0
    - Controls: random, shuffled, orthogonal vectors
    - Also run open-loop at α=-3.0 for comparison

Usage:
  python -m src.run_p6_syc_behavior_probe

Outputs:
  results_p6_syc_behavior/
    probe_model.pkl
    feedback_results.npz
  reports/IC4_P6_SYC_BEHAVIOR_REPORT.md (auto-generated)
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .model_loader import load_model_and_tokenizer
from .steering import apply_steering_hook, compute_norm_matched_orthogonal
from .run_p0_sycophancy_contrast import _is_sycophantic

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_log = logging.getLogger("p6_syc_behavior")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results_p6_syc_behavior"
REPORTS_DIR = PROJECT_ROOT / "reports"
CONTRAST_DATA_PATH = PROJECT_ROOT / "results_p0_sycophancy" / "sycophancy_contrast_data.json"
STEERING_VECTORS_PATH = PROJECT_ROOT / "results_t3_impulse_p4" / "steering_vectors.npz"

STEERING_LAYER = 10
OPTIMAL_ALPHA = -3.0
THRESHOLD = 0.5
MAX_NEW_TOKENS = 128
TRAIN_TEMPERATURE = 0.7
TEST_TEMPERATURE = 0.0
GENERATIONS_PER_SAMPLE = 5
RANDOM_SEED = 42


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


def _prompt_from_sample(sample: dict) -> str:
    prompt = sample.get("prompt", "")
    if not prompt:
        context = sample.get("context", "")
        question = sample.get("question", "")
        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
    return prompt


def generate_behavior_labeled_data(
    model, tokenizer, samples: List[dict], temperature: float, k: int
) -> List[dict]:
    device = next(model.parameters()).device
    labeled = []
    do_sample = temperature > 0.0

    for sample in tqdm(samples, desc=f"Behavior gen T={temperature}"):
        prompt = _prompt_from_sample(sample)
        for _ in range(k):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k_: v.to(device) for k_, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=temperature if do_sample else 1.0,
                    do_sample=do_sample,
                    top_p=0.95 if do_sample else 1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated = outputs[0][inputs["input_ids"].shape[1]:]
            answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
            is_syc = _is_sycophantic(answer)

            labeled.append({
                "sample_id": sample.get("tid", "unknown"),
                "prompt": prompt,
                "generated_output": answer,
                "is_sycophantic": is_syc,
            })

    syc_count = sum(1 for x in labeled if x["is_sycophantic"])
    _log.info(f"Behavior-labeled data: {len(labeled)} samples, {syc_count} syc ({syc_count/len(labeled):.4f})")
    return labeled


def train_behavior_probe(
    model, tokenizer, labeled_data: List[dict],
) -> Tuple[LogisticRegression, float, float, float]:
    X = []
    y = []
    for item in tqdm(labeled_data, desc="Collecting hidden states"):
        hs = _collect_last_token_hidden_state(model, tokenizer, item["prompt"], STEERING_LAYER)
        X.append(hs)
        y.append(int(item["is_sycophantic"]))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    probe = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        random_state=RANDOM_SEED,
    )
    probe.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, probe.predict(X_train))
    test_acc = accuracy_score(y_test, probe.predict(X_test))
    bal_acc = balanced_accuracy_score(y_test, probe.predict(X_test))

    _log.info(f"Probe train acc: {train_acc:.4f}, test acc: {test_acc:.4f}, bal: {bal_acc:.4f}")
    _log.info(f"Probe class distribution: pos={y.sum()}/{len(y)} ({y.mean():.4f})")
    _log.info(f"Test classification:\n{classification_report(y_test, probe.predict(X_test))}")

    return probe, train_acc, test_acc, bal_acc


def _make_feedback_hook(probe, steering_vector, alpha, threshold, stats):
    device = next(probe.coef_.__class__.__new__.__self_class__.__new__.__ne__)  # won't work, need model device
    pass


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
    model, tokenizer, test_samples, probe, steering_vector, alpha, direction_name, threshold=THRESHOLD,
):
    results = []
    stats = {"probe_scores": [], "gate_triggers": 0, "gate_passes": 0,
              "direction": direction_name, "alpha": alpha}
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
        for sample in tqdm(test_samples, desc=f"FB [{direction_name} α={alpha}]"):
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
                **sample,
                "generated_output": answer,
                "is_sycophantic": _is_sycophantic(answer),
                "gate_triggered": was_gated,
            })
    finally:
        handle.remove()

    syc_rate = sum(1 for r in results if r["is_sycophantic"]) / len(results) if results else 0.0
    gate_rate = stats["gate_triggers"] / len(results) if results else 0.0

    return {
        "direction": direction_name, "alpha": alpha, "n_samples": len(results),
        "syc_rate": syc_rate, "gate_rate": gate_rate,
        "probe_mean_score": float(np.mean(stats["probe_scores"])) if stats["probe_scores"] else 0.0,
        "results": results,
    }


def run_open_loop_generation(model, tokenizer, test_samples, steering_vector, alpha, direction_name):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    steering_tensor = torch.tensor(steering_vector, dtype=dtype, device=device)

    target_module = None
    for name, mod in model.named_modules():
        if name == f"model.layers.{STEERING_LAYER}":
            target_module = mod
            break

    handle = None
    if alpha != 0.0 and target_module is not None:
        handle = apply_steering_hook(model, STEERING_LAYER, steering_vector, alpha)

    results = []
    try:
        for sample in tqdm(test_samples, desc=f"OL [{direction_name} α={alpha}]"):
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
                **sample,
                "generated_output": answer,
                "is_sycophantic": _is_sycophantic(answer),
            })
    finally:
        if handle is not None:
            handle.remove()

    syc_rate = sum(1 for r in results if r["is_sycophantic"]) / len(results) if results else 0.0
    return {"direction": direction_name, "alpha": alpha, "n_samples": len(results), "syc_rate": syc_rate}


def _generate_report(
    baseline_syc_rate: float,
    train_n: int, train_pos: float, test_n: int,
    probe_train_acc: float, probe_test_acc: float, probe_bal_acc: float,
    feedback_results: List[dict], open_loop_results: List[dict],
    report_path: Path,
):
    lines = []
    lines.append("# IC-4 P6: Sycophancy Feedback Control — Behavior-Only Probe")
    lines.append("")
    lines.append(f"> **Date**: {time.strftime('%Y-%m-%d')} | **Status**: Completed")
    lines.append(f"> **Probe training**: {train_n} samples at T={TRAIN_TEMPERATURE}, {train_pos:.1%} syc")
    lines.append(f"> **Test**: {test_n} samples at T={TEST_TEMPERATURE}")
    lines.append(f"> **Layer**: {STEERING_LAYER} | **Alpha**: {OPTIMAL_ALPHA} | **Threshold**: {THRESHOLD}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Summary")
    lines.append("")
    lines.append("P5's probe learned group membership (fact_checker prompt y/n),")
    lines.append("not behavioral tendency. P6 fixes this by training the probe")
    lines.append("exclusively on standard-prompt samples, using T>0 to create")
    lines.append("behavioral variation (some outputs syc, some non-syc).")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Probe Training")
    lines.append("")
    lines.append(f"- Training data: {train_n} labeled samples (80% of behavior-labeled data)")
    lines.append(f"- Syc ratio in training: {train_pos:.1%}")
    lines.append(f"- Probe type: sklearn LogisticRegression (L2, C=1.0, lbfgs)")
    lines.append(f"- Input: last_prompt_token hidden state at Layer {STEERING_LAYER}")
    lines.append(f"- Train accuracy: {probe_train_acc:.4f}")
    lines.append(f"- Test accuracy: {probe_test_acc:.4f}")
    lines.append(f"- Test balanced accuracy: {probe_bal_acc:.4f}")
    lines.append("")

    if probe_test_acc < 0.55:
        lines.append("**WARNING: Probe test accuracy near chance.** Behavioral variation")
        lines.append("at T=0.7 may not produce enough separable signal. The probe is")
        lines.append("barely above random guessing. Feedback control results may be noisy.")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 3. Baseline")
    lines.append("")
    lines.append(f"- Baseline syc rate: **{baseline_syc_rate:.4f}**")
    lines.append(f"- Test samples: {test_n}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Feedback Control Results (α=-3.0)")
    lines.append("")

    header = "| Direction | Syc Rate | Δ from Baseline | Gate Rate | Probe μ |"
    sep = "|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)

    for fr in feedback_results:
        delta = fr["syc_rate"] - baseline_syc_rate
        lines.append(
            f"| {fr['direction']} | {fr['syc_rate']:.4f} | "
            f"{delta:+.4f} | {fr['gate_rate']:.4f} | {fr['probe_mean_score']:.4f} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. Open-Loop Comparison (α=-3.0)")
    lines.append("")

    header2 = "| Direction | Syc Rate | Δ from Baseline |"
    sep2 = "|---|---|"
    lines.append(header2)
    lines.append(sep2)

    for ol in open_loop_results:
        delta = ol["syc_rate"] - baseline_syc_rate
        lines.append(f"| {ol['direction']} | {ol['syc_rate']:.4f} | {delta:+.4f} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6. Interpretation")
    lines.append("")

    best_fb = min(feedback_results, key=lambda x: x["syc_rate"])
    best_fb_delta = best_fb["syc_rate"] - baseline_syc_rate
    best_ol = min(open_loop_results, key=lambda x: x["syc_rate"])
    best_ol_delta = best_ol["syc_rate"] - baseline_syc_rate

    lines.append(f"- Best feedback: {best_fb['direction']} syc={best_fb['syc_rate']:.4f} (Δ={best_fb_delta:+.4f})")
    lines.append(f"- Best open-loop: {best_ol['direction']} syc={best_ol['syc_rate']:.4f} (Δ={best_ol_delta:+.4f})")
    lines.append("")

    if probe_test_acc >= 0.55 and any(r["gate_rate"] > 0.10 for r in feedback_results):
        lines.append("**Probe→gate→hook is operational!** The behavior-only probe")
        lines.append("successfully learned behavioral tendency and gates on test samples.")
    elif probe_test_acc >= 0.55:
        lines.append("**Probe trains well but gate rarely triggers.** Consider lowering")
        lines.append("the threshold (currently 0.5) or calibrating to a lower percentile.")
    else:
        lines.append("**Behavior-only probe still insufficient.** The behavioral variation")
        lines.append("at T=0.7 is too noisy for clean separability. Consider:")
        lines.append("- Using an intermediate steering α to create stronger behavioral contrast")
        lines.append("- Using more generations per sample (k > 5)")
        lines.append("- Using a higher temperature (T > 0.7)")
        lines.append("- Alternative probe architecture (MLP, XGBoost)")

    lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    _log.info(f"Report: {report_path}")


def main():
    _log.info("P6: Behavior-Only Probe for Syc Feedback Control")

    _log.info("Loading contrast data...")
    with open(CONTRAST_DATA_PATH, "r", encoding="utf-8") as f:
        contrast_data = json.load(f)

    standard_samples = [s for s in contrast_data if not s.get("system_prompt")]
    _log.info(f"Standard-prompt samples: {len(standard_samples)}")

    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(len(standard_samples))
    train_idx = indices[:18]
    test_idx = indices[18:30]
    train_pool = [standard_samples[i] for i in train_idx]
    test_samples = [standard_samples[i] for i in test_idx]
    _log.info(f"Train pool: {len(train_pool)}, Test: {len(test_samples)}")

    _log.info("Loading model...")
    model, tokenizer = load_model_and_tokenizer()

    _log.info("Loading steering vectors...")
    sv_data = np.load(STEERING_VECTORS_PATH)
    vectors = {
        "v_syc": sv_data["v_syc"].astype(np.float32),
        "random": sv_data["random"].astype(np.float32),
        "shuffled": sv_data["shuffled"].astype(np.float32),
        "orthogonal": sv_data["orthogonal"].astype(np.float32),
    }

    _log.info(f"Phase 1: Generating behavior-labeled data (T={TRAIN_TEMPERATURE}, k={GENERATIONS_PER_SAMPLE})")
    labeled_data = generate_behavior_labeled_data(
        model, tokenizer, train_pool, TRAIN_TEMPERATURE, GENERATIONS_PER_SAMPLE
    )

    _log.info("Phase 2: Training behavior-only probe")
    probe, train_acc, test_acc, bal_acc = train_behavior_probe(model, tokenizer, labeled_data)

    _log.info("Phase 3: Baseline generation on test set")
    baseline_results = generate_behavior_labeled_data(
        model, tokenizer, test_samples, TEST_TEMPERATURE, 1
    )
    baseline_syc = sum(1 for x in baseline_results if x["is_sycophantic"])
    baseline_rate = baseline_syc / len(baseline_results) if baseline_results else 0.0
    _log.info(f"Baseline syc rate: {baseline_rate:.4f} ({baseline_syc}/{len(baseline_results)})")

    _log.info("Phase 4: Feedback control evaluation")
    feedback_results = []
    for dir_name in ["v_syc", "random", "shuffled", "orthogonal"]:
        _log.info(f"Feedback: {dir_name} α={OPTIMAL_ALPHA}")
        fr = run_feedback_generation(
            model, tokenizer, test_samples, probe,
            vectors[dir_name], OPTIMAL_ALPHA, dir_name,
        )
        _log.info(f"  syc_rate={fr['syc_rate']:.4f}, gate_rate={fr['gate_rate']:.4f}")
        feedback_results.append(fr)

    _log.info("Phase 5: Open-loop comparison")
    open_loop_results = []
    for dir_name in ["v_syc", "random", "shuffled", "orthogonal"]:
        _log.info(f"Open-loop: {dir_name} α={OPTIMAL_ALPHA}")
        ol = run_open_loop_generation(
            model, tokenizer, test_samples, vectors[dir_name],
            OPTIMAL_ALPHA, dir_name,
        )
        _log.info(f"  syc_rate={ol['syc_rate']:.4f}")
        open_loop_results.append(ol)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "probe_model.pkl", "wb") as f:
        pickle.dump(probe, f)

    fb_save = {r["direction"]: r for r in feedback_results}
    ol_save = {r["direction"]: r for r in open_loop_results}
    np.savez(
        RESULTS_DIR / "p6_results.npz",
        baseline_syc_rate=baseline_rate,
        probe_train_acc=train_acc,
        probe_test_acc=test_acc,
        probe_bal_acc=bal_acc,
        train_n=len(labeled_data),
        train_pos=sum(1 for x in labeled_data if x["is_sycophantic"]) / len(labeled_data),
        **{f"fb_{k}": v for k, v in fb_save.items()},
        **{f"ol_{k}": v for k, v in ol_save.items()},
        allow_pickle=True,
    )

    report_path = REPORTS_DIR / "IC4_P6_SYC_BEHAVIOR_REPORT.md"
    _generate_report(
        baseline_rate,
        len(labeled_data),
        sum(1 for x in labeled_data if x["is_sycophantic"]) / len(labeled_data),
        len(test_samples),
        train_acc, test_acc, bal_acc,
        feedback_results, open_loop_results,
        report_path,
    )

    _log.info("P6 complete.")


if __name__ == "__main__":
    main()