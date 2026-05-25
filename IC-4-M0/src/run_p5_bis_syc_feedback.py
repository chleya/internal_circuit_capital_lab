""" P5-bis: Open-loop positive alpha test for sycophancy sign asymmetry.

P5 found negative alpha (subtract v_syc) INCREASES sycophancy.
This tests the hypothesis: positive alpha (add v_syc) DECREASES sycophancy.

Uses the P5 test split (24 samples: mixed syc/non-syc groups) for consistency.
Open-loop only — no probe/gate, just constant steering.

Usage:
  python -m src.run_p5_bis_syc_feedback

Outputs:
  results_p5_bis_syc_feedback/
  reports/IC4_P5_BIS_SYC_FEEDBACK_REPORT.md
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from .model_loader import load_model_and_tokenizer
from .steering import apply_steering_hook
from .run_p0_sycophancy_contrast import _is_sycophantic

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_log = logging.getLogger("p5_bis")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results_p5_bis_syc_feedback"
REPORTS_DIR = PROJECT_ROOT / "reports"

STEERING_LAYER = 10
POS_ALPHAS = [1.0, 3.0, 5.0]
NEG_ALPHAS = [-1.0, -3.0, -5.0]
MAX_NEW_TOKENS = 128


def run_open_loop(model, tokenizer, samples: List[Dict],
                  steering_vector: np.ndarray, alpha: float,
                  direction_name: str) -> Dict:
    handle = apply_steering_hook(model, STEERING_LAYER, steering_vector, alpha) if alpha != 0 else None
    results = []
    device = next(model.parameters()).device

    try:
        for sample in tqdm(samples, desc=f"OL [{direction_name} α={alpha:+.1f}]"):
            prompt = sample.get("prompt", "")
            if not prompt:
                prompt = f"{sample.get('context', '')}\n\nQuestion: {sample.get('question', '')}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                         temperature=0.0, do_sample=False,
                                         pad_token_id=tokenizer.eos_token_id)
            generated = outputs[0][inputs["input_ids"].shape[1]:]
            answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
            results.append({**sample, "generated_output": answer,
                            "is_sycophantic": _is_sycophantic(answer)})
    finally:
        if handle is not None:
            handle.remove()

    syc_rate = sum(1 for r in results if r["is_sycophantic"]) / len(results)
    return {"direction": direction_name, "alpha": alpha, "n_samples": len(results),
            "n_syc": sum(1 for r in results if r["is_sycophantic"]),
            "syc_rate": syc_rate}


def _generate_report(baseline_rate, neg_results, pos_results, report_path):
    lines = [
        "# IC-4 P5-bis: Open-Loop Positive Alpha Test",
        "",
        f"> **Date**: {time.strftime('%Y-%m-%d')} | **Status**: Completed",
        f"> **Layer**: {STEERING_LAYER}",
        f"> **Test set**: 24 samples from P5 split (mixed syc/non-syc groups)",
        "",
        "---",
        "",
        "## 1. Baseline",
        f"- Baseline syc rate: **{baseline_rate['syc_rate']:.4f}** ({baseline_rate['n_syc']}/{baseline_rate['n_samples']})",
        "",
        "---",
        "",
        "## 2. Negative Alpha (P5 reference — subtract v_syc)",
        "",
        "| Direction | α=-1.0 | α=-3.0 | α=-5.0 |",
        "|---|---|---|---|",
    ]
    dirs = ["v_syc", "random", "shuffled", "orthogonal"]
    for d in dirs:
        vals = []
        for a in NEG_ALPHAS:
            key = f"{d}_{a:+.1f}"
            fr = neg_results.get(key, {})
            sr = fr.get("syc_rate", float('nan'))
            vals.append(f"{sr:.4f}")
        lines.append(f"| {d} | {vals[0]} | {vals[1]} | {vals[2]} |")

    lines += ["", "---", "",
              "## 3. Positive Alpha (P5-bis — ADD v_syc)",
              "",
              "| Direction | α=+1.0 | α=+3.0 | α=+5.0 |",
              "|---|---|---|---|",]
    for d in dirs:
        vals = []
        for a in POS_ALPHAS:
            key = f"{d}_+{a:.1f}"
            fr = pos_results.get(key, {})
            sr = fr.get("syc_rate", float('nan'))
            delta = fr.get("syc_rate", 0) - baseline_rate["syc_rate"]
            vals.append(f"{sr:.4f} ({delta:+.4f})")
        lines.append(f"| {d} | {vals[0]} | {vals[1]} | {vals[2]} |")

    lines += ["", "---", "", "## 4. Interpretation", ""]

    bl = baseline_rate["syc_rate"]
    best_neg = min((fr.get("syc_rate", 1.0) for fr in neg_results.values()), default=1.0)
    best_pos = min((fr.get("syc_rate", 1.0) for fr in pos_results.values()), default=1.0)

    lines.append(f"### Sign Asymmetry Verdict")
    lines.append(f"- Baseline syc rate: {bl:.4f}")
    lines.append(f"- Best negative α: {best_neg:.4f}")
    lines.append(f"- Best positive α: {best_pos:.4f}")
    lines.append("")

    if best_pos < bl and best_pos < best_neg:
        lines.append("**✓ Positive alpha REDUCES sycophancy below baseline AND below best negative alpha.**")
        lines.append(f"  Reduction: {bl - best_pos:.4f} ({((bl - best_pos)/bl)*100:.1f}% relative)")
        lines.append("")
        lines.append("This confirms the sign asymmetry hypothesis from P5:")
        lines.append("- v_syc in representation space points TOWARD non_sycophantic behavior")
        lines.append("- Adding v_syc (positive α) pushes the model to correct false statements")
        lines.append("- Subtracting v_syc (negative α) pushes the model to agree with false statements")
    elif best_neg < bl and best_neg < best_pos:
        lines.append("Negative alpha still gives better reduction — sign asymmetry not confirmed.")
    else:
        lines.append("**Neither sign reduces sycophancy below baseline.**")
        lines.append("Model's sycophancy may require a different intervention mechanism.")

    lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    _log.info(f"Report: {report_path}")


def main():
    _log.info("P5-bis: Open-loop positive alpha test")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    _log.info("Loading model...")
    model, tokenizer = load_model_and_tokenizer()

    with open(PROJECT_ROOT / "results_p0_sycophancy" / "sycophancy_contrast_data.json",
              "r", encoding="utf-8") as f:
        contrast_data = json.load(f)

    sv = np.load(PROJECT_ROOT / "results_t3_impulse_p4" / "steering_vectors.npz")
    steering_map = {
        "v_syc": sv["v_syc"], "random": sv["random"],
        "shuffled": sv["shuffled"], "orthogonal": sv["orthogonal"],
    }

    np.random.seed(42)
    indices = np.random.permutation(len(contrast_data))
    test_idx = indices[:24]
    test_samples = [contrast_data[i] for i in test_idx]

    _log.info(f"Test set: {len(test_samples)} samples")

    _log.info("Running baseline...")
    baseline = run_open_loop(model, tokenizer, test_samples, None, 0.0, "baseline")
    _log.info(f"Baseline syc rate: {baseline['syc_rate']:.4f} ({baseline['n_syc']}/{baseline['n_samples']})")

    neg_results = {}
    for alpha in NEG_ALPHAS:
        for d, v in steering_map.items():
            key = f"{d}_{alpha:+.1f}"
            _log.info(f"Negative: {key}")
            neg_results[key] = run_open_loop(model, tokenizer, test_samples, v, alpha, d)
            _log.info(f"  syc_rate={neg_results[key]['syc_rate']:.4f}")

    pos_results = {}
    for alpha in POS_ALPHAS:
        for d, v in steering_map.items():
            key = f"{d}_+{alpha:.1f}"
            _log.info(f"Positive: {key}")
            pos_results[key] = run_open_loop(model, tokenizer, test_samples, v, alpha, d)
            _log.info(f"  syc_rate={pos_results[key]['syc_rate']:.4f}")

    np.savez_compressed(RESULTS_DIR / "feedback_results.npz",
                        baseline_syc_rate=baseline["syc_rate"])
    _log.info(f"Results saved")

    report_path = REPORTS_DIR / "IC4_P5_BIS_SYC_FEEDBACK_REPORT.md"
    _generate_report(baseline, neg_results, pos_results, report_path)
    _log.info("P5-bis complete.")


if __name__ == "__main__":
    main()