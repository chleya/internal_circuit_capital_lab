"""
IC-4: Position Sensitivity Sweep — Relational Memory Hypothesis Experiment #1.

Tests whether M3-v6 probe scores and gate behavior are position-dependent.
Creates 3 position variants (early/mid/late) for each test sample and measures:
  - Probe score shift (PSI — Position Sensitivity Index)
  - Gate activation rate per position
  - If GPU available: full generation evaluation with hallucination/abstention rates

Usage:
    # Phase 1: create position-variant data files (CPU)
    python -m src.run_position_sensitivity --config configs/config_m3_v6.yaml --mode prepare_data

    # Phase 2: probe-based PSI only (needs GPU for feature extraction)
    python -m src.run_position_sensitivity --config configs/config_m3_v6.yaml --mode probe_psi

    # Phase 3: full generation evaluation (needs GPU)
    python -m src.run_position_sensitivity --config configs/config_m3_v6.yaml --mode full_eval
"""

import argparse
import json
import os
import sys
import time
import copy
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.run_m2 import load_config

FILLER_SENTENCES = [
    "Artificial intelligence continues to transform industries worldwide.",
    "The global economy has shown resilience amid recent challenges.",
]

POSITION_PREFIXES = {
    "early": "",
    "mid": "The following is a response from an AI assistant. " * 1,
    "late": "The following is a response from an AI assistant. " * 2,
}


def create_position_variants(samples, seed=42):
    rng = np.random.RandomState(seed)
    variants = {"early": [], "mid": [], "late": []}

    for sample in samples:
        evidence = sample["context"]
        question = sample["question"]

        for pos_name in ["early", "mid", "late"]:
            prefix = POSITION_PREFIXES[pos_name]
            shifted_context = prefix + evidence
            variant = copy.deepcopy(sample)
            variant["context"] = shifted_context
            variant["position"] = pos_name
            variant["original_context"] = evidence
            variants[pos_name].append(variant)

    return variants


def compute_probe_psi(model, tokenizer, probe_info, probe_cfg, variants, log_path=None):
    """Compute Position Sensitivity Index using existing probe."""
    from src.run_m3_v6 import _collect_prefill_features

    layer_idx = probe_cfg.get("layer", 12)
    representation = probe_cfg.get("representation", "last_prompt_token")

    results = {}
    for pos_name, samples in variants.items():
        X, y = _collect_prefill_features(model, tokenizer, samples, layer_idx, representation)
        scaler = probe_info["scaler"]
        clf = probe_info["classifier"]
        X_scaled = scaler.transform(X)
        probas = clf.predict_proba(X_scaled)[:, 1]
        preds = clf.predict(X_scaled)

        results[pos_name] = {
            "samples": samples,
            "features": X,
            "probe_scores": probas,
            "predictions": preds,
            "labels": y,
        }

    psi_values = []
    for pos_name in ["mid", "late"]:
        early_scores = results["early"]["probe_scores"]
        pos_scores = results[pos_name]["probe_scores"]
        delta = np.mean(np.abs(early_scores - pos_scores))
        psi_values.append({
            "comparison": f"early_vs_{pos_name}",
            "mean_abs_score_delta": float(delta),
            "early_mean_score": float(np.mean(early_scores)),
            f"{pos_name}_mean_score": float(np.mean(pos_scores)),
            "early_gate_rate": float(np.mean(early_scores < probe_cfg.get("threshold", 0.5))),
            f"{pos_name}_gate_rate": float(np.mean(pos_scores < probe_cfg.get("threshold", 0.5))),
        })

    return results, psi_values


def prepare_data(config):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)

    seed = config["steering"]["seeds"][0]
    train_path = config["data"].get("train_path", "data_m3/train.jsonl")
    test_path = config["data"].get("test_path", "data_m3/test.jsonl")
    train_final = train_path.replace(".jsonl", f"_s{seed}.jsonl")
    test_final = test_path.replace(".jsonl", f"_s{seed}.jsonl")

    from src.data_builder import load_jsonl, save_jsonl

    train_samples = load_jsonl(train_final)
    test_samples = load_jsonl(test_final)
    print(f"Loaded train={len(train_samples)}, test={len(test_samples)} (seed={seed})")

    train_variants = create_position_variants(train_samples, seed=seed)
    test_variants = create_position_variants(test_samples, seed=seed)

    out_dir = os.path.join("data_position_sensitivity", f"s{seed}")
    os.makedirs(out_dir, exist_ok=True)

    for pos_name in ["early", "mid", "late"]:
        train_path_out = os.path.join(out_dir, f"train_{pos_name}_s{seed}.jsonl")
        test_path_out = os.path.join(out_dir, f"test_{pos_name}_s{seed}.jsonl")
        save_jsonl(train_variants[pos_name], train_path_out)
        save_jsonl(test_variants[pos_name], test_path_out)
        print(f"  {pos_name}: train={len(train_variants[pos_name])}, test={len(test_variants[pos_name])}")

    all_train = train_variants["early"] + train_variants["mid"] + train_variants["late"]
    save_jsonl(all_train, os.path.join(out_dir, f"train_all_s{seed}.jsonl"))
    print(f"  train_all (mixed positions): {len(all_train)} samples")

    stats = {
        "seed": seed,
        "n_train_original": len(train_samples),
        "n_test_original": len(test_samples),
        "n_train_total": len(all_train),
        "variants": ["early", "mid", "late", "all"],
    }
    stats_path = os.path.join(out_dir, "variant_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to {stats_path}")


def probe_psi_only(config):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)
    results_dir = config["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    seed = config["steering"]["seeds"][0]
    layers = config["steering"]["layers"]
    probe_cfg = config["probe"]
    gen_cfg = config["generation"]
    representations = probe_cfg["representations"]

    from src.model_loader import load_model_and_tokenizer, get_model_layer_count
    from src.activation_collector import load_activations
    from src.steering import get_all_vectors
    from src.run_m3_v6 import _collect_prefill_features, _train_probe

    print(f"Loading model ({config['model']['name']})...")
    model, tokenizer = load_model_and_tokenizer(
        model_name=config["model"]["name"],
        device=config["model"]["device"],
        torch_dtype=config["model"].get("torch_dtype", "float32"),
    )

    train_path = config["data"].get("train_path", "data_m3/train.jsonl")
    train_final = train_path.replace(".jsonl", f"_s{seed}.jsonl")
    from src.data_builder import load_jsonl
    train = load_jsonl(train_final)

    data_dir = os.path.join("data_position_sensitivity", f"s{seed}")
    variants = {}
    for pos_name in ["early", "mid", "late"]:
        vpath = os.path.join(data_dir, f"test_{pos_name}_s{seed}.jsonl")
        variants[pos_name] = load_jsonl(vpath)

    all_psi = []

    for layer_idx in layers:
        print(f"\n{'='*60}")
        print(f"LAYER {layer_idx}")
        print(f"{'='*60}")

        for rep in representations:
            print(f"\n  Probe [{rep}]")

            X_train, y_train = _collect_prefill_features(model, tokenizer, train, layer_idx, rep)
            probe_info = _train_probe(X_train, y_train, cv_folds=probe_cfg.get("cv_folds", 3))
            print(f"    train_acc={probe_info['train_acc']:.4f}, AUC={probe_info.get('auc', 'N/A')}")

            probe_cfg_with_rep = dict(probe_cfg)
            probe_cfg_with_rep["layer"] = layer_idx
            probe_cfg_with_rep["representation"] = rep

            results, psi_vals = compute_probe_psi(model, tokenizer, probe_info, probe_cfg_with_rep, variants)

            for pv in psi_vals:
                pv["seed"] = seed
                pv["layer"] = layer_idx
                pv["representation"] = rep
                all_psi.append(pv)

            for pv in psi_vals:
                print(f"    {pv['comparison']}: dScore={pv['mean_abs_score_delta']:.4f}  "
                      f"early_gate={pv['early_gate_rate']:.3f}  {pv['comparison'].split('_')[-1]}_gate={pv[pv['comparison'].split('_')[-1] + '_gate_rate']:.3f}")

    psi_df = pd.DataFrame(all_psi)
    psi_path = os.path.join(results_dir, "position_sensitivity_index.csv")
    psi_df.to_csv(psi_path, index=False)
    print(f"\nPSI results saved to {psi_path}")

    mean_psi = psi_df["mean_abs_score_delta"].mean()
    max_psi = psi_df["mean_abs_score_delta"].max()
    print(f"\nPosition Sensitivity Index (PSI): mean={mean_psi:.4f}, max={max_psi:.4f}")
    if mean_psi > 0.3:
        print("  D9 TRIGGERED: PSI > 0.3 — position encoding is a first-order confound.")
    elif mean_psi > 0.1:
        print("  WARNING: moderate position sensitivity detected.")
    else:
        print("  OK: position sensitivity is low. RoPE ceiling not yet binding.")


def full_eval(config):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)
    results_dir = config["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "position_sensitivity_log.txt")

    def _log(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    seed = config["steering"]["seeds"][0]
    layers = config["steering"]["layers"]
    alpha_max = config["steering"]["alphas"][0]
    probe_cfg = config["probe"]
    gen_cfg = config["generation"]
    representations = probe_cfg["representations"]

    _log("=" * 60)
    _log("IC-4: POSITION SENSITIVITY — FULL GENERATION EVALUATION")
    _log("=" * 60)

    from src.model_loader import load_model_and_tokenizer, get_model_layer_count

    _log(f"Loading model ({config['model']['name']})...")
    model, tokenizer = load_model_and_tokenizer(
        model_name=config["model"]["name"],
        device=config["model"]["device"],
        torch_dtype=config["model"].get("torch_dtype", "float32"),
    )

    from src.activation_collector import load_activations
    from src.steering import get_all_vectors
    from src.run_m3_v6 import _collect_prefill_features, _train_probe, _generate_single_pass_hard_gate
    from src.evaluate import generate_answers, evaluate_outputs

    train_path = config["data"].get("train_path", "data_m3/train.jsonl")
    train_final = train_path.replace(".jsonl", f"_s{seed}.jsonl")
    from src.data_builder import load_jsonl
    train = load_jsonl(train_final)

    data_dir = os.path.join("data_position_sensitivity", f"s{seed}")
    variants = {}
    for pos_name in ["early", "mid", "late"]:
        vpath = os.path.join(data_dir, f"test_{pos_name}_s{seed}.jsonl")
        variants[pos_name] = load_jsonl(vpath)

    all_metrics = []

    for layer_idx in layers:
        _log(f"\n{'='*40}")
        _log(f"LAYER {layer_idx}")
        _log(f"{'='*40}")

        act_path = os.path.join("results_m3", f"activations_s{seed}_l{layer_idx}.npz")
        if not os.path.exists(act_path):
            _log(f"  Activations not found at {act_path}, skipping layer.")
            continue
        acts = load_activations(act_path)
        hidden_dim = acts["positive"].shape[1]
        all_vectors = get_all_vectors(acts["positive"], acts["negative"], hidden_dim)
        steering_v = all_vectors["steering"]

        for pos_name in ["early", "mid", "late"]:
            test_data = variants[pos_name]
            _log(f"\n  --- {pos_name.upper()} POSITION ---")

            _log(f"  Base...")
            base_res = generate_answers(model, tokenizer, test_data, mode="base",
                                        max_new_tokens=gen_cfg["max_new_tokens"],
                                        temperature=gen_cfg["temperature"],
                                        do_sample=gen_cfg["do_sample"])
            metrics = evaluate_outputs(base_res)
            metrics["seed"] = seed
            metrics["layer"] = layer_idx
            metrics["mode"] = "base"
            metrics["position"] = pos_name
            metrics["alpha"] = 0.0
            metrics["vector_type"] = "none"
            all_metrics.append(metrics)
            _log(f"    base: H={metrics['hallucination_rate']:.3f} C={metrics['correct_answer_rate']:.3f}")

            for rep in representations:
                _log(f"  Probe [{rep}]...")
                X_train, y_train = _collect_prefill_features(model, tokenizer, train, layer_idx, rep)
                probe_info = _train_probe(X_train, y_train, cv_folds=probe_cfg.get("cv_folds", 3))

                probe_cfg_with_rep = dict(probe_cfg)
                probe_cfg_with_rep["representation"] = rep

                _log(f"  Single-pass hard gate...")
                hard_res = _generate_single_pass_hard_gate(
                    model, tokenizer, test_data, steering_v, layer_idx,
                    alpha_max, probe_info, probe_cfg_with_rep, gen_cfg, "steering")
                metrics = evaluate_outputs(hard_res)
                metrics["seed"] = seed
                metrics["layer"] = layer_idx
                metrics["mode"] = "single_pass_hard_gate"
                metrics["position"] = pos_name
                metrics["alpha"] = alpha_max
                metrics["representation"] = rep
                metrics["vector_type"] = "steering"
                all_metrics.append(metrics)
                _log(f"    gate_{pos_name}: H={metrics['hallucination_rate']:.3f} C={metrics['correct_answer_rate']:.3f}")

    df = pd.DataFrame(all_metrics)
    cols = ["seed", "layer", "mode", "position", "alpha", "vector_type", "representation",
            "hallucination_rate", "calibrated_abstention_rate",
            "correct_answer_rate", "unnecessary_abstention_rate",
            "answerable_count", "unanswerable_count"]
    present_cols = [c for c in cols if c in df.columns]
    df = df[present_cols]
    met_path = os.path.join(results_dir, "position_sensitivity_metrics.csv")
    df.to_csv(met_path, index=False)
    _log(f"\nMetrics saved to {met_path}")

    _log("\nPOSITION SENSITIVITY SUMMARY:")
    for pos_name in ["early", "mid", "late"]:
        base_row = df[(df["mode"] == "base") & (df["position"] == pos_name)]
        gate_row = df[(df["mode"] == "single_pass_hard_gate") & (df["position"] == pos_name)]
        if len(base_row) > 0:
            _log(f"  {pos_name}: base H={base_row.iloc[0]['hallucination_rate']:.3f} C={base_row.iloc[0]['correct_answer_rate']:.3f}")
        if len(gate_row) > 0:
            _log(f"         gate H={gate_row.iloc[0]['hallucination_rate']:.3f} C={gate_row.iloc[0]['correct_answer_rate']:.3f}")

    # Compute position sensitivity
    base_rows = df[df["mode"] == "base"]
    if len(base_rows) >= 3:
        h_values = [base_rows[base_rows["position"] == p].iloc[0]["hallucination_rate"] for p in ["early", "mid", "late"] if len(base_rows[base_rows["position"] == p]) > 0]
        if len(h_values) >= 2:
            h_range = max(h_values) - min(h_values)
            _log(f"\n  Hallucination rate range across positions: {h_range:.3f}")


def main():
    parser = argparse.ArgumentParser(description="IC-4: Position Sensitivity Sweep")
    parser.add_argument("--config", type=str, default="configs/config_m3_v6.yaml")
    parser.add_argument("--mode", type=str, default="prepare_data",
                        choices=["prepare_data", "probe_psi", "full_eval"])
    args = parser.parse_args()

    config = load_config(args.config)

    if args.mode == "prepare_data":
        prepare_data(config)
    elif args.mode == "probe_psi":
        probe_psi_only(config)
    elif args.mode == "full_eval":
        full_eval(config)


if __name__ == "__main__":
    main()