"""
IC-4-M5 P0-A: Cross-Scenario on Cross-Seed Validation.

Validates large and hard_ood scenarios across seeds [0,1,2] at layer=12 / alpha=-1.0.
Imports core generation/probe/steering functions from run_m3_v6.py.
"""

import os, sys, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_m3_v6 import (
    _log, _find_transformer_layer, _collect_prefill_features, _train_probe,
    _generate_oracle_gated, _generate_single_pass_hard_gate,
)
from model_loader import load_model_and_tokenizer
from activation_collector import load_activations
from steering import compute_steering_vector, compute_random_vector, compute_shuffled_vector
from evaluate import generate_answers, evaluate_outputs

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_m5_p0a")
REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports_m5_p0a")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

SCENARIOS = {
    "large": {
        "train_path": "data_m4/train_test_large.jsonl",
        "test_path": "data_m4/test_large.jsonl",
        "label": "M4 large (120 samples)",
    },
    "hard_ood": {
        "train_path": "data_m4/train_test_hard.jsonl",
        "test_path": "data_m4/test_hard.jsonl",
        "label": "M4 hard OOD (extreme entity split)",
    },
}

SEEDS = [0, 1, 2]
LAYER = 12
ALPHA = -1.0
GEN_CFG = {"max_new_tokens": 48, "temperature": 0.0, "do_sample": False}
CSV_PATH = os.path.join(RESULTS_DIR, "m5_p0a_metrics.csv")


def _load_cached_m4_data(seed, base_train_path, base_test_path):
    def _inject_seed(path):
        stem, ext = os.path.splitext(path)
        return f"{stem}_s{seed}{ext}"
    train_path = _inject_seed(base_train_path)
    test_path = _inject_seed(base_test_path)
    with open(train_path, encoding="utf-8") as f:
        train = [json.loads(line) for line in f]
    with open(test_path, encoding="utf-8") as f:
        test = [json.loads(line) for line in f]
    return train, test


def _get_all_vectors(pos, neg, dim):
    real_v = compute_steering_vector(pos, neg)
    rnd_v = compute_random_vector(dim)
    shf_v = compute_shuffled_vector(pos, neg)
    return {"steering": real_v, "random": rnd_v, "shuffled": shf_v}


def _evaluate_and_add(results, seed, layer, mode, alpha, vector_type, scenario, all_rows):
    m = evaluate_outputs(results)
    row = {
        "hallucination_rate": m["hallucination_rate"],
        "calibrated_abstention_rate": m["calibrated_abstention_rate"],
        "correct_answer_rate": m["correct_answer_rate"],
        "unnecessary_abstention_rate": m["unnecessary_abstention_rate"],
        "style_only_score": m.get("style_only_score", 0.0),
        "answerable_count": m.get("answerable_count", 0),
        "unanswerable_count": m.get("unanswerable_count", 0),
        "hallucination_count": m.get("hallucination_count", 0),
        "calibrated_abstention_count": m.get("calibrated_abstention_count", 0),
        "correct_count": m.get("correct_count", 0),
        "unnecessary_abstention_count": m.get("unnecessary_abstention_count", 0),
        "avg_answerable_uncertainty": 0.0,
        "avg_unanswerable_uncertainty": 0.0,
        "seed": seed,
        "layer": layer,
        "mode": mode,
        "alpha": alpha,
        "vector_type": vector_type,
        "scenario": scenario,
    }
    all_rows.append(row)
    mode_short = mode.replace("_gate_a-1.0", "").replace("single_pass_hard_", "").replace("_single_pass_hard", "")
    _log(f"    [{scenario}]  {mode_short:12s} H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f} UA={m['unnecessary_abstention_rate']:.3f}")
    return row


def _run_one_scenario(model, tokenizer, seed, scenario_name, scenario_cfg, all_metrics):
    _log(f"\n{'='*50}")
    _log(f"SCENARIO: {scenario_name} | seed={seed} | layer={LAYER} | alpha={ALPHA:.1f}")
    _log(f"{'='*50}")

    train, test = _load_cached_m4_data(seed, scenario_cfg["train_path"], scenario_cfg["test_path"])
    n_pos = sum(1 for s in test if s.get("answerability") == "answerable")
    n_neg = sum(1 for s in test if s.get("answerability") == "unanswerable")
    _log(f"  seed={seed}: train={len(train)}, test={n_pos}A+{n_neg}U")

    base_res = generate_answers(model, tokenizer, test, mode="base", **GEN_CFG)
    _evaluate_and_add(base_res, seed, LAYER, "base", 0.0, "none", scenario_name, all_metrics)

    act_path = os.path.join("results_m3", f"activations_s{seed}_l{LAYER}.npz")
    acts = load_activations(act_path)
    all_vectors = _get_all_vectors(acts["positive"], acts["negative"], acts["positive"].shape[1])
    steering_v = all_vectors["steering"]

    oracle_res = _generate_oracle_gated(
        model, tokenizer, test, steering_v, LAYER, ALPHA,
        f"oracle_gate_a{ALPHA:+.1f}", GEN_CFG)
    _evaluate_and_add(oracle_res, seed, LAYER, f"oracle_gate_a{ALPHA:+.1f}", ALPHA,
                      "steering", scenario_name, all_metrics)

    X_train, y_train = _collect_prefill_features(model, tokenizer, train, LAYER, "last_prompt_token")
    probe_info = _train_probe(X_train, y_train, cv_folds=3)
    _log(f"    probe: train_acc={probe_info['train_acc']:.4f} cv_acc={probe_info.get('cv_acc_mean', 1.0):.4f}")
    probe_cfg = {"representation": "last_prompt_token", "threshold": 0.5, "steepness": 1.0}

    for vtype, vlabel in [("steering", "real"), ("random", "random"), ("shuffled", "shuffled")]:
        mode_name = f"{vlabel}_single_pass_hard_gate_a{ALPHA:+.1f}"
        gen_res = _generate_single_pass_hard_gate(
            model, tokenizer, test, all_vectors[vtype], LAYER,
            ALPHA, probe_info, probe_cfg, GEN_CFG, vtype)
        _evaluate_and_add(gen_res, seed, LAYER, mode_name, ALPHA,
                          vtype, scenario_name, all_metrics)

    df = pd.DataFrame(all_metrics)
    df.to_csv(CSV_PATH, index=False)


def _generate_report(all_metrics):
    if len(all_metrics) == 0:
        return "NO_DATA"
    df = pd.DataFrame(all_metrics)

    lines = [
        "# IC-4-M5 P0-A: Cross-Scenario on Cross-Seed",
        "",
        "> Validates large and hard_ood scenarios across seeds [0,1,2] at layer=12 / alpha=-1.0.",
        "",
        "## 1. Complete Matrix",
        "",
        "| Seed | Scenario | Mode | H | C | UA | Oracle Gap |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for _, row in df.iterrows():
        sc = row.get("scenario", "?")
        seed = int(row["seed"])
        mode = str(row["mode"])
        h = row["hallucination_rate"]
        c = row["correct_answer_rate"]
        ua = row["unnecessary_abstention_rate"]
        oracle_h = None
        for _, r2 in df.iterrows():
            if r2.get("scenario") == sc and int(r2["seed"]) == seed and "oracle_gate" in str(r2["mode"]):
                oracle_h = r2["hallucination_rate"]
                break
        gap = round(h - oracle_h, 4) if oracle_h is not None else "—"
        lines.append(f"| {seed} | {sc} | {mode} | {h:.3f} | {c:.3f} | {ua:.3f} | {gap} |")

    lines.append("")
    lines.append("## 2. Oracle Gap Summary")
    lines.append("")
    lines.append("| Seed | Scenario | oracle H | hard H | random H | shuffled H | oracle_gap |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    max_gap = 0.0
    for seed in SEEDS:
        for sc_name in SCENARIOS:
            sub = df[(df["seed"] == seed) & (df["scenario"] == sc_name)]
            oracle_h = hard_h = rnd_h = shf_h = None
            for _, r in sub.iterrows():
                m = str(r["mode"])
                hv = r["hallucination_rate"]
                if "oracle_gate" in m:
                    oracle_h = hv
                elif "real_single_pass" in m:
                    hard_h = hv
                elif "random_single_pass" in m:
                    rnd_h = hv
                elif "shuffled_single_pass" in m:
                    shf_h = hv
            gap = round(hard_h - oracle_h, 4) if hard_h is not None and oracle_h is not None else "—"
            if isinstance(gap, float):
                max_gap = max(max_gap, abs(gap))
            lines.append(f"| {seed} | {sc_name} | {oracle_h or '—'} | {hard_h or '—'} | {rnd_h or '—'} | {shf_h or '—'} | {gap} |")

    lines.append("")
    lines.append("## 3. Control Separation")
    lines.append("")
    lines.append("| Seed | Scenario | hard-random gap | hard-shuffled gap | random check |")
    lines.append("|---|---:|---:|---:|")

    for seed in SEEDS:
        for sc_name in SCENARIOS:
            sub = df[(df["seed"] == seed) & (df["scenario"] == sc_name)]
            hard_h = rnd_h = shf_h = None
            for _, r in sub.iterrows():
                m = str(r["mode"])
                hv = r["hallucination_rate"]
                if "real_single_pass" in m:
                    hard_h = hv
                elif "random_single_pass" in m:
                    rnd_h = hv
                elif "shuffled_single_pass" in m:
                    shf_h = hv
            hr = round(rnd_h - hard_h, 4) if rnd_h and hard_h else "—"
            hs = round(shf_h - hard_h, 4) if shf_h and hard_h else "—"
            rc = "OK" if isinstance(hr, float) and hr > 0.05 else ("WEAK" if isinstance(hr, float) and hr > 0 else "FAIL")
            lines.append(f"| {seed} | {sc_name} | {hr} | {hs} | {rc} |")

    lines.append("")
    lines.append("## 4. Verdict")
    lines.append("")
    if max_gap <= 0.05:
        lines.append(f"**IC4_M5_P0A_ROBUST** — Max oracle gap = {max_gap:.4f} ≤ 0.05.")
    elif max_gap <= 0.10:
        lines.append(f"**IC4_M5_P0A_PARTIAL** — Max oracle gap = {max_gap:.4f}.")
    else:
        lines.append(f"**IC4_M5_P0A_DEGRADED** — Max oracle gap = {max_gap:.4f} > 0.10.")
    lines.append("")

    report_path = os.path.join(REPORT_DIR, "IC4_M5_P0A_CROSS_SCENARIO_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    _log(f"Report saved to {report_path}")
    return "IC4_M5_P0A_ROBUST" if max_gap <= 0.05 else "IC4_M5_P0A_PARTIAL"


def main():
    all_metrics = []
    t_start = time.time()

    _log("=" * 60)
    _log("IC-4-M5 P0-A: Cross-Scenario on Cross-Seed")
    _log(f"Seeds: {SEEDS}, Layer: {LAYER}, Alpha: {ALPHA:.1f}")
    _log(f"Scenarios: {list(SCENARIOS.keys())}")
    _log("=" * 60)

    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")

    for scenario_name, sc in SCENARIOS.items():
        for seed in SEEDS:
            _run_one_scenario(model, tokenizer, seed, scenario_name, sc, all_metrics)

    elapsed = time.time() - t_start
    _log(f"\nAll scenarios complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    verdict = _generate_report(all_metrics)
    _log(f"\nVerdict: {verdict}")


if __name__ == "__main__":
    main()