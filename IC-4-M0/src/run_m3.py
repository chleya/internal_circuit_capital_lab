"""
IC-4-M3: Feedback-Controlled Steering — Smoke Test.
Compares open-loop (fixed alpha) vs closed-loop (entropy-driven per-token alpha).

Usage:
    python -m src.run_m3 --config configs/config_m3.yaml
"""

import argparse
import os
import sys
import random
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.run_m2 import load_config


def _log(msg: str, log_file: str = None):
    print(msg, flush=True)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def _build_data(seed, config, train_path, test_path, log):
    from src.data_builder import build_dataset, save_jsonl, load_jsonl

    train_final = train_path.replace(".jsonl", f"_s{seed}.jsonl")
    test_final = test_path.replace(".jsonl", f"_s{seed}.jsonl")

    if os.path.exists(train_final) and os.path.exists(test_final):
        _log(f"  Loading cached data for seed {seed}", log)
        train = load_jsonl(train_final)
        test = load_jsonl(test_final)
    else:
        _log(f"  Generating data for seed {seed}", log)
        random.seed(seed)
        np.random.seed(seed)
        data_cfg = config["data"].copy()
        train, test = build_dataset(data_cfg)
        save_jsonl(train, train_final)
        save_jsonl(test, test_final)

    num_a = sum(1 for s in train if s["answerability"] == "answerable")
    num_u = len(train) - num_a
    num_a_test = sum(1 for s in test if s["answerability"] == "answerable")
    num_u_test = len(test) - num_a_test
    _log(f"  seed={seed}: train {num_a}A+{num_u}U, test {num_a_test}A+{num_u_test}U", log)
    return train, test


def main():
    parser = argparse.ArgumentParser(description="IC-4-M3: Feedback-Controlled Steering")
    parser.add_argument("--config", type=str, default="configs/config_m3.yaml")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)

    config = load_config(args.config)
    results_dir = config["output"]["results_dir"]
    reports_dir = config["output"]["reports_dir"]
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    log_path = os.path.join(results_dir, "run_log.txt")
    _log("=" * 60, log_path)
    _log("IC-4-M3: Feedback-Controlled Steering (Smoke)", log_path)
    _log("=" * 60, log_path)

    seeds = config["steering"]["seeds"]
    layers = config["steering"]["layers"]
    alphas = config["steering"]["alphas"]
    fb_base_alpha = config["steering"]["feedback_base_alpha"]
    fb_k_values = config["steering"]["feedback_k_values"]
    gen_cfg = config["generation"]

    _log(f"\nConfig: seed={seeds}, layer={layers}, open_loop_alphas={alphas}", log_path)
    _log(f"Feedback: base_alpha={fb_base_alpha}, k={fb_k_values}", log_path)
    _log(f"Data: train={config['data']['train_size']}, test={config['data']['test_size']}", log_path)

    total_evals = (2 + len(alphas) + len(fb_k_values)) * len(seeds) * len(layers)
    _log(f"Estimated evaluations: {total_evals}", log_path)

    from src.model_loader import load_model_and_tokenizer, get_model_layer_count

    _log(f"\nLoading model ({config['model']['name']})...", log_path)
    model, tokenizer = load_model_and_tokenizer(
        model_name=config["model"]["name"],
        device=config["model"]["device"],
        torch_dtype=config["model"].get("torch_dtype", "float16"),
    )
    total_layers = get_model_layer_count(model)
    _log(f"  Total layers: {total_layers}", log_path)

    all_rows = []
    t_start = time.time()

    for seed in seeds:
        _log(f"\n{'='*40}", log_path)
        _log(f"SEED {seed}", log_path)
        _log(f"{'='*40}", log_path)

        random.seed(seed)
        np.random.seed(seed)

        train_path = config["data"].get("train_path", "data_m3/train.jsonl")
        test_path = config["data"].get("test_path", "data_m3/test.jsonl")
        train, test = _build_data(seed, config, train_path, test_path, log_path)

        from src.evaluate import generate_answers, evaluate_outputs

        _log(f"\n  Base...", log_path)
        base_results = generate_answers(model, tokenizer, test, mode="base",
                                        max_new_tokens=gen_cfg["max_new_tokens"],
                                        temperature=gen_cfg["temperature"],
                                        do_sample=gen_cfg["do_sample"])
        base_metrics = evaluate_outputs(base_results)
        base_metrics["seed"] = seed
        base_metrics["layer"] = -1
        base_metrics["mode"] = "base"
        base_metrics["alpha"] = 0.0
        all_rows.append(base_metrics)
        _log(f"    base: H={base_metrics['hallucination_rate']:.3f} "
             f"C={base_metrics['correct_answer_rate']:.3f} "
             f"UA={base_metrics['unnecessary_abstention_rate']:.3f}", log_path)

        _log(f"\n  Prompt-only...", log_path)
        po_results = generate_answers(model, tokenizer, test, mode="prompt_only",
                                      max_new_tokens=gen_cfg["max_new_tokens"],
                                      temperature=gen_cfg["temperature"],
                                      do_sample=gen_cfg["do_sample"])
        po_metrics = evaluate_outputs(po_results)
        po_metrics["seed"] = seed
        po_metrics["layer"] = -1
        po_metrics["mode"] = "prompt_only"
        po_metrics["alpha"] = 0.0
        all_rows.append(po_metrics)
        _log(f"    prompt_only: H={po_metrics['hallucination_rate']:.3f} "
             f"C={po_metrics['correct_answer_rate']:.3f} "
             f"UA={po_metrics['unnecessary_abstention_rate']:.3f}", log_path)

        from src.activation_collector import collect_pair_activations
        from src.steering import get_all_vectors
        from src.evaluate import run_generation_with_steering, run_generation_with_feedback

        for layer_idx in layers:
            layer_t0 = time.time()
            _log(f"\n  LAYER {layer_idx}", log_path)

            act_path = os.path.join(results_dir, f"activations_s{seed}_l{layer_idx}.npz")
            if os.path.exists(act_path):
                from src.activation_collector import load_activations
                _log(f"    layer={layer_idx}: loaded cached activations", log_path)
                acts = load_activations(act_path)
            else:
                _log(f"    layer={layer_idx}: collecting activations...", log_path)
                acts = collect_pair_activations(model, tokenizer, train, layer=str(layer_idx),
                                                token_position="last")
                from src.activation_collector import save_activations
                save_activations(acts, act_path)
                _log(f"    layer={layer_idx}: saved ({acts['positive'].shape[0]} pairs)", log_path)

            hidden_dim = acts["positive"].shape[1]
            vectors = get_all_vectors(acts["positive"], acts["negative"], hidden_dim)

            # ── Open-loop (fixed alpha) ──
            _log(f"\n  Open-loop steering:", log_path)
            for alpha in alphas:
                results, _ = run_generation_with_steering(
                    model, tokenizer, test, vectors["steering"], layer_idx,
                    alpha, "steering",
                    max_new_tokens=gen_cfg["max_new_tokens"],
                    temperature=gen_cfg["temperature"],
                    do_sample=gen_cfg["do_sample"],
                )
                metrics = evaluate_outputs(results)
                metrics["seed"] = seed
                metrics["layer"] = layer_idx
                metrics["mode"] = f"steering_a{alpha}"
                metrics["alpha"] = alpha
                all_rows.append(metrics)
                _log(f"    open_a{alpha:+0.2f}: H={metrics['hallucination_rate']:.3f} "
                     f"C={metrics['correct_answer_rate']:.3f} "
                     f"UA={metrics['unnecessary_abstention_rate']:.3f}", log_path)

            # ── Feedback (entropy-driven per-token alpha) ──
            _log(f"\n  Feedback steering:", log_path)
            for k in fb_k_values:
                fb_results = run_generation_with_feedback(
                    model, tokenizer, test,
                    steering_vector=vectors["steering"],
                    steering_layer=layer_idx,
                    base_alpha=fb_base_alpha,
                    k=k,
                    max_new_tokens=gen_cfg["max_new_tokens"],
                    temperature=gen_cfg["temperature"],
                    do_sample=gen_cfg["do_sample"],
                )
                fb_metrics = evaluate_outputs(fb_results)
                fb_metrics["seed"] = seed
                fb_metrics["layer"] = layer_idx
                fb_metrics["mode"] = f"feedback_a{fb_base_alpha}_k{k}"
                fb_metrics["alpha"] = fb_base_alpha
                fb_metrics["k"] = k
                all_rows.append(fb_metrics)
                _log(f"    fb_a{fb_base_alpha:+0.2f}_k{k:.1f}: H={fb_metrics['hallucination_rate']:.3f} "
                     f"C={fb_metrics['correct_answer_rate']:.3f} "
                     f"UA={fb_metrics['unnecessary_abstention_rate']:.3f}", log_path)

            _log(f"    layer {layer_idx} done in {time.time() - layer_t0:.0f}s", log_path)

    elapsed = time.time() - t_start
    _log(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)", log_path)

    # ── Save metrics ──
    df = pd.DataFrame(all_rows)
    cols_order = [
        "seed", "layer", "mode", "alpha",
        "hallucination_rate", "calibrated_abstention_rate",
        "correct_answer_rate", "unnecessary_abstention_rate",
        "answerable_count", "unanswerable_count",
        "hallucination_count", "calibrated_abstention_count",
        "correct_count", "unnecessary_abstention_count",
    ]
    present = [c for c in cols_order if c in df.columns]
    df = df[present]
    raw_path = os.path.join(results_dir, "metrics_raw.csv")
    df.to_csv(raw_path, index=False)
    _log(f"\nRaw metrics saved to {raw_path} ({len(df)} rows)", log_path)

    # ── Comparison table ──
    _log(f"\n{'='*60}", log_path)
    _log(f"COMPARISON: Open-Loop vs Feedback", log_path)
    _log(f"{'='*60}", log_path)
    _log(f"{'Mode':<25} {'H':>8} {'C':>8} {'UA':>8}", log_path)
    _log(f"{'-'*25} {'-'*8} {'-'*8} {'-'*8}", log_path)

    base_row = df[df["mode"] == "base"].iloc[0]
    _log(f"{'base':<25} {base_row['hallucination_rate']:8.3f} "
         f"{base_row['correct_answer_rate']:8.3f} "
         f"{base_row['unnecessary_abstention_rate']:8.3f}", log_path)

    for _, row in df.iterrows():
        mode = str(row.get("mode", ""))
        if mode.startswith("steering_a") and abs(row.get("alpha", 0)) > 1e-9:
            label = f"open_a{row['alpha']:+0.2f}"
            _log(f"{label:<25} {row['hallucination_rate']:8.3f} "
                 f"{row['correct_answer_rate']:8.3f} "
                 f"{row['unnecessary_abstention_rate']:8.3f}", log_path)
        elif mode.startswith("feedback_a"):
            k_val = row.get("k", "?")
            label = f"fb_a{row['alpha']:+0.2f}_k{k_val}"
            _log(f"{label:<25} {row['hallucination_rate']:8.3f} "
                 f"{row['correct_answer_rate']:8.3f} "
                 f"{row['unnecessary_abstention_rate']:8.3f}", log_path)

    # ── Delta analysis ──
    _log(f"\nDELTA ANALYSIS (vs base):", log_path)
    base_h = base_row["hallucination_rate"]
    base_c = base_row["correct_answer_rate"]
    base_ua = base_row["unnecessary_abstention_rate"]

    best_open = None
    best_fb = None
    for _, row in df.iterrows():
        mode = str(row.get("mode", ""))
        if mode.startswith("steering_a") and abs(row.get("alpha", 0)) > 1e-9:
            if best_open is None or row["hallucination_rate"] < best_open["hallucination_rate"]:
                best_open = row
        elif mode.startswith("feedback_a"):
            if best_fb is None or row["hallucination_rate"] < best_fb["hallucination_rate"]:
                best_fb = row

    if best_open is not None:
        dH_open = base_h - best_open["hallucination_rate"]
        dC_open = base_c - best_open["correct_answer_rate"]
        dUA_open = best_open["unnecessary_abstention_rate"] - base_ua
        _log(f"  Best open-loop (a={best_open['alpha']:+0.2f}): "
             f"dH=-{dH_open:.3f} dC=-{dC_open:.3f} dUA=+{dUA_open:.3f}", log_path)

    if best_fb is not None:
        dH_fb = base_h - best_fb["hallucination_rate"]
        dC_fb = base_c - best_fb["correct_answer_rate"]
        dUA_fb = best_fb["unnecessary_abstention_rate"] - base_ua
        _log(f"  Best feedback   (a={best_fb['alpha']:+0.2f} k={best_fb.get('k','?')}): "
             f"dH=-{dH_fb:.3f} dC=-{dC_fb:.3f} dUA=+{dUA_fb:.3f}", log_path)

    if best_open is not None and best_fb is not None:
        _log(f"\n  Selectivity comparison:", log_path)
        _log(f"    Open-loop:  dH/dC = {dH_open/(dC_open+1e-8):.1f}  (H reduction per C lost)", log_path)
        _log(f"    Feedback:   dH/dC = {dH_fb/(dC_fb+1e-8):.1f}  (H reduction per C lost)", log_path)
        if dC_fb + 1e-8 > 0:
            ratio = (dH_open/(dC_open+1e-8)) / (dH_fb/(dC_fb+1e-8))
            if ratio > 1.1:
                _log(f"    => Feedback is MORE selective ({ratio:.1f}x better H/C tradeoff)", log_path)
            elif ratio < 0.9:
                _log(f"    => Open-loop is MORE selective ({1/ratio:.1f}x better H/C tradeoff)", log_path)
            else:
                _log(f"    => Similar selectivity (ratio={ratio:.1f})", log_path)

    _log(f"\n{'='*60}", log_path)
    _log(f"IC-4-M3 smoke complete.", log_path)
    _log(f"{'='*60}", log_path)


if __name__ == "__main__":
    main()