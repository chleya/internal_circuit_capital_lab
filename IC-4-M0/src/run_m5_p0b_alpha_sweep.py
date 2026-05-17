"""
IC-4-M5 P0-B: Alpha Sweep on Cross-Seed.

Tests alpha ∈ {-0.8, -1.2} on seeds [1,2] at layer=12 (large scenario).
Determines whether oracle gap is flat w.r.t. alpha or alpha-specific.
"""

import os, sys, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_m3_v6 import (
    _log, _collect_prefill_features, _train_probe,
    _generate_oracle_gated, _generate_single_pass_hard_gate,
)
from model_loader import load_model_and_tokenizer
from activation_collector import load_activations
from steering import compute_steering_vector, compute_random_vector, compute_shuffled_vector
from evaluate import generate_answers, evaluate_outputs

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_m5_p0b")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports_m5_p0b")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

GEN_CFG = {"max_new_tokens": 48, "temperature": 0.0, "do_sample": False}
LAYER = 12
SEEDS = [1, 2]
ALPHAS = [-0.8, -1.2]
CSV_PATH = os.path.join(RESULTS_DIR, "m5_p0b_alpha_sweep.csv")

def _inject_seed(path, seed):
    stem, ext = os.path.splitext(path)
    return f"{stem}_s{seed}{ext}"

def _load_m4_data(seed, train_path_tmpl, test_path_tmpl):
    with open(_inject_seed(train_path_tmpl, seed), encoding="utf-8") as f:
        train = [json.loads(line) for line in f]
    with open(_inject_seed(test_path_tmpl, seed), encoding="utf-8") as f:
        test = [json.loads(line) for line in f]
    return train, test

def _get_all_vectors(pos, neg, dim):
    return {
        "steering": compute_steering_vector(pos, neg),
        "random": compute_random_vector(dim),
        "shuffled": compute_shuffled_vector(pos, neg),
    }

def _evaluate_and_add(results, seed, layer, mode, alpha, vector_type, scenario, all_rows):
    m = evaluate_outputs(results)
    row = {
        "hallucination_rate": m["hallucination_rate"],
        "correct_answer_rate": m["correct_answer_rate"],
        "calibrated_abstention_rate": m["calibrated_abstention_rate"],
        "unnecessary_abstention_rate": m["unnecessary_abstention_rate"],
        "style_only_score": m.get("style_only_score", 0.0),
        "seed": seed, "layer": layer, "mode": mode,
        "alpha": alpha, "vector_type": vector_type, "scenario": scenario,
    }
    all_rows.append(row)

def _run_one_alpha(model, tokenizer, seed, alpha, scenario, train, test,
                   acts, all_metrics):
    _log(f"\n  SCENARIO: {scenario} | seed={seed} | alpha={alpha:.1f}")

    steering_v = compute_steering_vector(acts["positive"], acts["negative"])
    all_vectors = _get_all_vectors(acts["positive"], acts["negative"], acts["positive"].shape[1])

    oracle_res = _generate_oracle_gated(
        model, tokenizer, test, steering_v, LAYER, alpha,
        f"oracle_gate_a{alpha:+.1f}", GEN_CFG)
    _evaluate_and_add(oracle_res, seed, LAYER, f"oracle_gate_a{alpha:+.1f}",
                      alpha, "steering", scenario, all_metrics)

    X_train, y_train = _collect_prefill_features(model, tokenizer, train, LAYER, "last_prompt_token")
    probe_info = _train_probe(X_train, y_train, cv_folds=3)
    probe_cfg = {"representation": "last_prompt_token", "threshold": 0.5, "steepness": 1.0}

    for vtype, vlabel in [("steering", "real"), ("random", "random"), ("shuffled", "shuffled")]:
        gen_res = _generate_single_pass_hard_gate(
            model, tokenizer, test, all_vectors[vtype], LAYER,
            alpha, probe_info, probe_cfg, GEN_CFG, vtype)
        mode_name = f"{vlabel}_single_pass_hard_gate_a{alpha:+.1f}"
        _evaluate_and_add(gen_res, seed, LAYER, mode_name, alpha,
                          vtype, scenario, all_metrics)

    df = pd.DataFrame(all_metrics)
    df.to_csv(CSV_PATH, index=False)

def main():
    print("=" * 56)
    print("IC-4-M5 P0-B: Alpha Sweep on Cross-Seed")
    print(f"Seeds: {SEEDS}, Alphas: {ALPHAS}, Layer: {LAYER}")
    print("Scenario: large")
    print("=" * 56)

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
    print("Model loaded.")

    TRAIN_TMPL = "data_m4/train_test_large.jsonl"
    TEST_TMPL = "data_m4/test_large.jsonl"

    all_metrics = []

    for seed in SEEDS:
        train, test = _load_m4_data(seed, TRAIN_TMPL, TEST_TMPL)
        n_pos = sum(1 for s in test if s.get("answerability") == "answerable")
        n_neg = sum(1 for s in test if s.get("answerability") == "unanswerable")
        _log(f"\n{'='*48}")
        _log(f"SEED={seed}: train={len(train)}, test={n_pos}A+{n_neg}U")
        _log(f"{'='*48}")

        base_res = generate_answers(model, tokenizer, test, mode="base", **GEN_CFG)
        _evaluate_and_add(base_res, seed, LAYER, "base", 0.0, "none", "large", all_metrics)
        bm = evaluate_outputs(base_res)
        _log(f"  [large]  base  H={bm['hallucination_rate']:.3f} C={bm['correct_answer_rate']:.3f}")

        act_path = os.path.join("results_m3", f"activations_s{seed}_l{LAYER}.npz")
        acts = load_activations(act_path)

        for alpha in ALPHAS:
            _run_one_alpha(model, tokenizer, seed, alpha, "large", train, test,
                           acts, all_metrics)

    _log("\nDONE: All alpha sweeps complete.")
    _log(f"CSV: {CSV_PATH}")

    df = pd.DataFrame(all_metrics)
    base_rows = df[df["mode"] == "base"]
    oracle_rows = df[df["mode"].str.startswith("oracle")]
    real_rows = df[df["mode"].str.startswith("real")]
    rnd_rows = df[df["mode"].str.startswith("random")]
    shf_rows = df[df["mode"].str.startswith("shuffled")]

    _log("\n=== P0-B Alpha Sweep Summary ===")
    for seed in SEEDS:
        b = base_rows[base_rows["seed"] == seed]
        _log(f"\nseed={seed}  base: H={b['hallucination_rate'].values[0]:.3f} C={b['correct_answer_rate'].values[0]:.3f}")
        for alpha in ALPHAS:
            o = oracle_rows[(oracle_rows["seed"] == seed) & (oracle_rows["alpha"] == alpha)]
            r = real_rows[(real_rows["seed"] == seed) & (real_rows["alpha"] == alpha)]
            rd = rnd_rows[(rnd_rows["seed"] == seed) & (rnd_rows["alpha"] == alpha)]
            s = shf_rows[(shf_rows["seed"] == seed) & (shf_rows["alpha"] == alpha)]
            oracle_h = o["hallucination_rate"].values[0] if len(o) else 0
            real_h = r["hallucination_rate"].values[0] if len(r) else 0
            rnd_h = rd["hallucination_rate"].values[0] if len(rd) else 0
            shf_h = s["hallucination_rate"].values[0] if len(s) else 0
            gap = real_h - oracle_h
            _log(f"  alpha={alpha:+.1f}  oracle={oracle_h:.3f} real={real_h:.3f} "
                 f"random={rnd_h:.3f} shuffled={shf_h:.3f}  gap={gap:+.3f}")

    _log("\n=== FIN ===")

if __name__ == "__main__":
    main()