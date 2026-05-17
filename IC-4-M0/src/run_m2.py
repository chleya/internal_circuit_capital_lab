"""
IC-4-M2: Boundary Diagnostic & Constrained Steering Selection.
Multi-seed, multi-layer, fine-alpha activation steering sweep
with constrained selection and control gap analysis.

Usage:
    python -m src.run_m2 --config configs/config_m2.yaml          # smoke
    python -m src.run_m2 --config configs/config_m2.yaml --full   # full sweep
"""

import argparse
import os
import sys
import json
import random
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _parse_yaml_value(val: str) -> Any:
    val = val.strip().strip('"').strip("'")
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    if val.lower() == "null" or val.lower() == "none":
        return None
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    if val.startswith("[") and val.endswith("]"):
        inner = val[1:-1]
        return [_parse_yaml_value(v) for v in inner.split(",")]
    return val


def load_config(config_path: str) -> Dict:
    config = {}
    current_section = config
    sections = [("root", config)]
    with open(config_path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip()
            if not raw or raw.lstrip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())
            stripped = raw.split("#", 1)[0].rstrip()
            if not stripped or ":" not in stripped:
                continue
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip()
            if val == "":
                sub = {}
                if indent == 0:
                    config[key] = sub
                    current_section = sub
                    sections = [("root", config), (key, sub)]
                else:
                    while len(sections) > 1 and sections[-1][0] != "root":
                        sections.pop()
                    sections[-1][1][key] = sub
                    sections.append((key, sub))
                    current_section = sub
            else:
                parsed = _parse_yaml_value(val)
                target = sections[-1][1]
                target[key] = parsed
    return config


def _log_print(msg: str, log_file: str = None):
    print(msg, flush=True)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def _build_data_for_seed(seed, config, train_path, test_path, log):
    from src.data_builder import build_dataset, save_jsonl, load_jsonl

    train_final = train_path.replace(".jsonl", f"_s{seed}.jsonl")
    test_final = test_path.replace(".jsonl", f"_s{seed}.jsonl")

    if os.path.exists(train_final) and os.path.exists(test_final):
        _log_print(f"  Loading cached data for seed {seed}", log)
        train = load_jsonl(train_final)
        test = load_jsonl(test_final)
    else:
        _log_print(f"  Generating data for seed {seed}", log)
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
    _log_print(f"  seed={seed}: train {num_a}A+{num_u}U, test {num_a_test}A+{num_u_test}U", log)
    return train, test


def _collect_acts_for_layer(model, tokenizer, train, layer, act_path, log):
    from src.activation_collector import collect_pair_activations, save_activations, load_activations

    if os.path.exists(act_path):
        _log_print(f"    layer={layer}: loaded cached activations", log)
        return load_activations(act_path)

    _log_print(f"    layer={layer}: collecting activations...", log)
    acts = collect_pair_activations(
        model, tokenizer, train, layer=str(layer), token_position="last",
    )
    save_activations(acts, act_path)
    _log_print(f"    layer={layer}: saved ({acts['positive'].shape[0]} pairs, dim={acts['positive'].shape[1]})", log)
    return acts


def _run_eval_for_mode(model, tokenizer, test, vector, layer_idx, alpha, mode, gen_config):
    from src.evaluate import run_generation_with_steering, evaluate_outputs

    max_new = gen_config.get("max_new_tokens", 48)
    temp = gen_config.get("temperature", 0.0)
    do_sample = gen_config.get("do_sample", False)

    if mode == "base" or mode == "prompt_only" or abs(alpha) < 1e-9:
        from src.evaluate import generate_answers
        results = generate_answers(
            model, tokenizer, test, mode=mode,
            max_new_tokens=max_new, temperature=temp, do_sample=do_sample,
        )
    else:
        results, handle = run_generation_with_steering(
            model, tokenizer, test,
            steering_vector=vector, steering_layer=layer_idx, alpha=alpha, mode=mode,
            max_new_tokens=max_new, temperature=temp, do_sample=do_sample,
        )

    metrics = evaluate_outputs(results)
    return results, metrics


def main():
    parser = argparse.ArgumentParser(description="IC-4-M2: Boundary Diagnostic")
    parser.add_argument("--config", type=str, default="configs/config_m2.yaml")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)

    config = load_config(args.config)

    if args.full:
        _log_print("*** FULL SWEEP MODE ***")
        config["data"]["train_size"] = 120
        config["data"]["test_size"] = 240
        config["steering"]["seeds"] = [0, 1, 2]
        config["steering"]["layers"] = [10, 11, 12, 13, 14]
        config["steering"]["alphas"] = [-1.25, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, 0.0]

    results_dir = config["output"]["results_dir"]
    reports_dir = config["output"]["reports_dir"]
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    log_path = os.path.join(results_dir, "run_log.txt")
    _log_print("=" * 60, log_path)
    _log_print("IC-4-M2: Boundary Diagnostic & Constrained Steering Selection", log_path)
    _log_print("=" * 60, log_path)
    _log_print(f"Config: {args.config}", log_path)
    _log_print(f"Model: {config['model']['name']}", log_path)

    seeds = config["steering"]["seeds"]
    layers = config["steering"]["layers"]
    alphas = config["steering"]["alphas"]
    gen_config = config["generation"]
    data_config = config["data"]

    _log_print(f"\nConfig summary:", log_path)
    _log_print(f"  seeds: {seeds}", log_path)
    _log_print(f"  layers: {layers}", log_path)
    _log_print(f"  alphas: {alphas}", log_path)
    _log_print(f"  train_size: {data_config['train_size']}", log_path)
    _log_print(f"  test_size: {data_config['test_size']}", log_path)
    _log_print(f"  max_new_tokens: {gen_config.get('max_new_tokens', 48)}", log_path)

    non_zero_alphas = [a for a in alphas if abs(a) > 1e-9]
    total_evals = len(seeds) * (2 + len(layers) * (len(alphas) + len(non_zero_alphas) * 2))
    _log_print(f"  Estimated evaluations: {total_evals}", log_path)

    from src.model_loader import load_model_and_tokenizer, get_model_layer_count

    model_cfg = config["model"]
    _log_print(f"\nLoading model ({model_cfg['name']})...", log_path)
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_cfg["name"],
        device=model_cfg["device"],
        torch_dtype=model_cfg.get("torch_dtype", "float16"),
    )
    total_layers = get_model_layer_count(model)
    _log_print(f"  Total layers: {total_layers}", log_path)

    all_rows = []
    t_start = time.time()

    for seed in seeds:
        _log_print(f"\n{'='*40}", log_path)
        _log_print(f"SEED {seed}", log_path)
        _log_print(f"{'='*40}", log_path)

        random.seed(seed)
        np.random.seed(seed)

        train_path = data_config.get("train_path", "data_m2/train.jsonl")
        test_path = data_config.get("test_path", "data_m2/test.jsonl")
        train, test = _build_data_for_seed(seed, config, train_path, test_path, log_path)

        _log_print(f"\n  Base + Prompt-only...", log_path)
        base_results, base_metrics = _run_eval_for_mode(
            model, tokenizer, test, None, 0, 0.0, "base", gen_config,
        )
        base_metrics["seed"] = seed
        base_metrics["layer"] = -1
        base_metrics["mode"] = "base"
        base_metrics["alpha"] = 0.0
        all_rows.append(base_metrics)
        _log_print(f"    base: H={base_metrics['hallucination_rate']:.3f} "
                   f"C={base_metrics['correct_answer_rate']:.3f} "
                   f"A={base_metrics['calibrated_abstention_rate']:.3f} "
                   f"UA={base_metrics['unnecessary_abstention_rate']:.3f}", log_path)

        po_results, po_metrics = _run_eval_for_mode(
            model, tokenizer, test, None, 0, 0.0, "prompt_only", gen_config,
        )
        po_metrics["seed"] = seed
        po_metrics["layer"] = -1
        po_metrics["mode"] = "prompt_only"
        po_metrics["alpha"] = 0.0
        all_rows.append(po_metrics)
        _log_print(f"    prompt_only: H={po_metrics['hallucination_rate']:.3f} "
                   f"C={po_metrics['correct_answer_rate']:.3f} "
                   f"A={po_metrics['calibrated_abstention_rate']:.3f} "
                   f"UA={po_metrics['unnecessary_abstention_rate']:.3f}", log_path)

        from src.steering import get_all_vectors

        for layer_idx in layers:
            layer_t0 = time.time()
            _log_print(f"\n  LAYER {layer_idx}", log_path)

            act_path = os.path.join(results_dir, f"activations_s{seed}_l{layer_idx}.npz")
            acts = _collect_acts_for_layer(
                model, tokenizer, train, layer_idx, act_path, log_path,
            )
            hidden_dim = acts["positive"].shape[1]

            vectors = get_all_vectors(acts["positive"], acts["negative"], hidden_dim)
            _log_print(f"    vectors: steering={np.linalg.norm(vectors['steering']):.4f} "
                       f"random={np.linalg.norm(vectors['random']):.4f} "
                       f"shuffled={np.linalg.norm(vectors['shuffled']):.4f}", log_path)

            for alpha in alphas:
                label = f"steering_a{alpha}"
                vec = vectors["steering"] if abs(alpha) > 1e-9 else None
                _, metrics = _run_eval_for_mode(
                    model, tokenizer, test, vec, layer_idx, alpha,
                    "steering" if abs(alpha) > 1e-9 else "base", gen_config,
                )
                metrics["seed"] = seed
                metrics["layer"] = layer_idx
                metrics["mode"] = label
                metrics["alpha"] = alpha
                all_rows.append(metrics)
                _log_print(f"    {label}: H={metrics['hallucination_rate']:.3f} "
                           f"C={metrics['correct_answer_rate']:.3f} "
                           f"A={metrics['calibrated_abstention_rate']:.3f} "
                           f"UA={metrics['unnecessary_abstention_rate']:.3f}", log_path)

            for alpha in non_zero_alphas:
                label = f"random_a{alpha}"
                _, metrics = _run_eval_for_mode(
                    model, tokenizer, test, vectors["random"], layer_idx, alpha,
                    "random", gen_config,
                )
                metrics["seed"] = seed
                metrics["layer"] = layer_idx
                metrics["mode"] = label
                metrics["alpha"] = alpha
                all_rows.append(metrics)
                _log_print(f"    {label}: H={metrics['hallucination_rate']:.3f} "
                           f"C={metrics['correct_answer_rate']:.3f}", log_path)

            for alpha in non_zero_alphas:
                label = f"shuffled_a{alpha}"
                _, metrics = _run_eval_for_mode(
                    model, tokenizer, test, vectors["shuffled"], layer_idx, alpha,
                    "shuffled", gen_config,
                )
                metrics["seed"] = seed
                metrics["layer"] = layer_idx
                metrics["mode"] = label
                metrics["alpha"] = alpha
                all_rows.append(metrics)
                _log_print(f"    {label}: H={metrics['hallucination_rate']:.3f} "
                           f"C={metrics['correct_answer_rate']:.3f}", log_path)

            _log_print(f"    layer {layer_idx} done in {time.time() - layer_t0:.0f}s", log_path)

    elapsed = time.time() - t_start
    _log_print(f"\nTotal sweep time: {elapsed:.0f}s ({elapsed/60:.1f} min)", log_path)

    # Save raw metrics
    df = pd.DataFrame(all_rows)
    cols_order = [
        "seed", "layer", "mode", "alpha",
        "hallucination_rate", "calibrated_abstention_rate",
        "correct_answer_rate", "unnecessary_abstention_rate",
        "style_only_score",
        "answerable_count", "unanswerable_count",
        "hallucination_count", "calibrated_abstention_count",
        "correct_count", "unnecessary_abstention_count",
        "avg_answerable_uncertainty", "avg_unanswerable_uncertainty",
    ]
    df = df[[c for c in cols_order if c in df.columns]]
    raw_path = os.path.join(results_dir, "metrics_raw.csv")
    df.to_csv(raw_path, index=False)
    _log_print(f"\nRaw metrics saved to {raw_path} ({len(df)} rows)", log_path)

    # Aggregated metrics
    group_cols = [c for c in ["layer", "mode", "alpha"] if c in df.columns]
    metric_cols = [
        "hallucination_rate", "calibrated_abstention_rate",
        "correct_answer_rate", "unnecessary_abstention_rate", "style_only_score",
    ]
    available_metric_cols = [c for c in metric_cols if c in df.columns]

    agg_parts = []
    for col in available_metric_cols:
        mean_series = df.groupby(group_cols)[col].mean().rename(f"{col}_mean")
        std_series = df.groupby(group_cols)[col].std().rename(f"{col}_std")
        agg_parts.append(mean_series)
        agg_parts.append(std_series)

    agg_df = pd.concat(agg_parts, axis=1).reset_index()
    agg_path = os.path.join(results_dir, "metrics_agg.csv")
    agg_df.to_csv(agg_path, index=False)
    _log_print(f"Aggregated metrics saved to {agg_path} ({len(agg_df)} rows)", log_path)

    # Per-layer summary
    per_layer_summary = []
    steering = df[df["mode"].str.startswith("steering_a") & (df["alpha"] != 0.0)]
    for layer_idx in layers:
        layer_data = steering[steering["layer"] == layer_idx]
        if len(layer_data) > 0:
            best = layer_data.loc[layer_data["hallucination_rate"].idxmin()]
            per_layer_summary.append({
                "layer": layer_idx,
                "best_seed": int(best["seed"]),
                "best_alpha": best["alpha"],
                "best_h": best["hallucination_rate"],
                "best_c": best["correct_answer_rate"],
                "best_ua": best["unnecessary_abstention_rate"],
                "best_ca": best["calibrated_abstention_rate"],
            })

    _log_print(f"\nPer-layer summary:", log_path)
    for entry in per_layer_summary:
        _log_print(f"  Layer {entry['layer']}: best H={entry['best_h']:.3f} "
                   f"C={entry['best_c']:.3f} UA={entry['best_ua']:.3f} "
                   f"alpha={entry['best_alpha']}", log_path)

    # Seed stability
    seed_stability = {}
    if len(seeds) > 1:
        for layer_idx in layers:
            layer_steering = steering[steering["layer"] == layer_idx]
            if len(layer_steering) > 0:
                h_by_seed = layer_steering.groupby("seed")["hallucination_rate"].min()
                seed_stability[f"layer_{layer_idx}"] = {
                    "h_range": f"{h_by_seed.min():.3f}-{h_by_seed.max():.3f}",
                    "h_mean": float(h_by_seed.mean()),
                    "h_std": float(h_by_seed.std()) if len(h_by_seed) > 1 else 0.0,
                }
    else:
        seed_stability = {"note": "Single-seed smoke run; seed stability not assessed."}

    # M2 Verdict
    _log_print(f"\nComputing M2 verdict...", log_path)
    from src.report_writer import compute_m2_verdict, generate_m2_report

    m2_result = compute_m2_verdict(df, config)
    verdict = m2_result["verdict"]
    verdict_reason = m2_result["verdict_reason"]

    _log_print(f"  Verdict: {verdict}", log_path)
    _log_print(f"  Reason: {verdict_reason}", log_path)
    if m2_result.get("constrained_candidates"):
        _log_print(f"  Constrained candidates: {len(m2_result['constrained_candidates'])}", log_path)
    if m2_result.get("rejected_candidates"):
        _log_print(f"  Rejected candidates: {len(m2_result['rejected_candidates'])}", log_path)

    # Generate report
    report_path = os.path.join(reports_dir, "IC4_M2_BOUNDARY_DIAGNOSTIC_REPORT.md")
    generate_m2_report(
        report_path=report_path,
        config=config,
        df=df,
        agg_df=agg_df,
        m2_result=m2_result,
        per_layer_summary=per_layer_summary,
        seed_stability_summary=seed_stability,
        elapsed_seconds=elapsed,
    )
    _log_print(f"Report saved to {report_path}", log_path)

    _log_print(f"\n{'=' * 60}", log_path)
    _log_print(f"IC-4-M2 complete. Verdict: {verdict}", log_path)
    _log_print(f"Reason: {verdict_reason}", log_path)
    _log_print(f"{'=' * 60}", log_path)


if __name__ == "__main__":
    main()