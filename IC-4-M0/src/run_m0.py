"""
IC-4-M0: Main run script.
Orchestrates the full activation steering experiment pipeline.

Usage:
    python -m src.run_m0 --config configs/config.yaml
"""

import argparse
import os
import sys
import json
import re
import random
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any

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


def main():
    parser = argparse.ArgumentParser(description="IC-4-M0: Activation Steering Experiment")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)

    log_path = "results/run_log.txt"
    os.makedirs("results", exist_ok=True)

    config = load_config(args.config)
    _log_print("=" * 60, log_path)
    _log_print("IC-4-M0: Minimal Activation Steering Anti-Hallucination Experiment", log_path)
    _log_print("=" * 60, log_path)
    _log_print(f"Config: {args.config}", log_path)
    _log_print(f"Model: {config['model']['name']}", log_path)

    random.seed(42)
    np.random.seed(42)

    # --- Task 1: Data ---
    _log_print("\n[Task 1] Building synthetic QA datasets...", log_path)

    from src.data_builder import build_dataset, save_jsonl, load_jsonl

    data_config = config["data"]
    train_path = data_config["train_path"]
    test_path = data_config["test_path"]

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        _log_print("Generating fresh data...", log_path)
        train, test = build_dataset(data_config)
        save_jsonl(train, train_path)
        save_jsonl(test, test_path)
        _log_print(f"  Train: {len(train)} samples -> {train_path}", log_path)
        _log_print(f"  Test:  {len(test)} samples -> {test_path}", log_path)
    else:
        _log_print(f"Loading existing data from {train_path} and {test_path}", log_path)
        train = load_jsonl(train_path)
        test = load_jsonl(test_path)

    num_answerable_train = sum(1 for s in train if s["answerability"] == "answerable")
    num_unanswerable_train = sum(1 for s in train if s["answerability"] == "unanswerable")
    num_answerable_test = sum(1 for s in test if s["answerability"] == "answerable")
    num_unanswerable_test = sum(1 for s in test if s["answerability"] == "unanswerable")
    _log_print(f"  Train: {num_answerable_train} answerable + {num_unanswerable_train} unanswerable", log_path)
    _log_print(f"  Test:  {num_answerable_test} answerable + {num_unanswerable_test} unanswerable", log_path)

    # --- Task 2: Model ---
    _log_print("\n[Task 2] Loading model...", log_path)

    from src.model_loader import load_model_and_tokenizer, get_model_layer_count, get_middle_layer_index

    model_config = config["model"]
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_config["name"],
        device=model_config["device"],
        torch_dtype=model_config.get("torch_dtype", "float16"),
    )
    total_layers = get_model_layer_count(model)
    middle_layer = get_middle_layer_index(model)
    _log_print(f"  Total layers: {total_layers}", log_path)
    _log_print(f"  Middle layer index: {middle_layer}", log_path)

    activation_config = config["activation"]
    layer_spec = activation_config.get("layer", "middle")
    if layer_spec == "middle":
        target_layer = middle_layer
    elif layer_spec == "last":
        target_layer = total_layers - 1
    else:
        target_layer = int(layer_spec)
    _log_print(f"  Target layer: {target_layer}", log_path)

    token_position = activation_config.get("token_position", "last")

    # --- Task 3: Collect activations ---
    _log_print("\n[Task 3] Collecting activations...", log_path)

    from src.activation_collector import collect_pair_activations, save_activations, load_activations

    act_path = config["output"]["results_dir"] + "/activations_train.npz"
    if not os.path.exists(act_path):
        acts = collect_pair_activations(
            model, tokenizer, train,
            layer=layer_spec,
            token_position=token_position,
        )
        save_activations(acts, act_path)
        _log_print(f"  Saved activations to {act_path}", log_path)
        _log_print(f"  Positive: {acts['positive'].shape}", log_path)
        _log_print(f"  Negative: {acts['negative'].shape}", log_path)
    else:
        acts = load_activations(act_path)
        _log_print(f"  Loaded activations from {act_path}", log_path)
        _log_print(f"  Positive: {acts['positive'].shape}", log_path)
        _log_print(f"  Negative: {acts['negative'].shape}", log_path)

    hidden_dim = acts["positive"].shape[1]

    # --- Task 4: Steering vectors ---
    _log_print("\n[Task 4] Computing steering vectors...", log_path)

    from src.steering import get_all_vectors

    vectors = get_all_vectors(acts["positive"], acts["negative"], hidden_dim)
    _log_print(f"  Steering vector norm: {np.linalg.norm(vectors['steering']):.4f}", log_path)
    _log_print(f"  Random vector norm:   {np.linalg.norm(vectors['random']):.4f}", log_path)
    _log_print(f"  Shuffled vector norm: {np.linalg.norm(vectors['shuffled']):.4f}", log_path)

    # --- Task 5 & 6: Generation & Evaluation ---
    _log_print("\n[Task 5] Running generation and evaluation...", log_path)

    from src.evaluate import (
        generate_answers,
        evaluate_outputs,
        run_generation_with_steering,
    )

    gen_config = config["generation"]
    max_new_tokens = gen_config.get("max_new_tokens", 128)
    temperature = gen_config.get("temperature", 0.0)
    do_sample = gen_config.get("do_sample", False)

    all_rows = []

    # 5a: Base (no steering, no prompt instruction)
    _log_print("\n  [5a] Base mode (no steering, no prompt instruction)...", log_path)
    base_results = generate_answers(
        model, tokenizer, test,
        mode="base",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
    )
    base_metrics = evaluate_outputs(base_results)
    base_metrics["mode"] = "base"
    base_metrics["alpha"] = 0.0
    all_rows.append(base_metrics)
    _log_print(f"    base: H={base_metrics['hallucination_rate']:.3f} "
               f"C={base_metrics['correct_answer_rate']:.3f} "
               f"A={base_metrics['calibrated_abstention_rate']:.3f} "
               f"UA={base_metrics['unnecessary_abstention_rate']:.3f}", log_path)

    # 5b: Prompt-only
    _log_print("\n  [5b] Prompt-only mode (safety instruction, no steering)...", log_path)
    po_results = generate_answers(
        model, tokenizer, test,
        mode="prompt_only",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
    )
    po_metrics = evaluate_outputs(po_results)
    po_metrics["mode"] = "prompt_only"
    po_metrics["alpha"] = 0.0
    all_rows.append(po_metrics)
    _log_print(f"    prompt_only: H={po_metrics['hallucination_rate']:.3f} "
               f"C={po_metrics['correct_answer_rate']:.3f} "
               f"A={po_metrics['calibrated_abstention_rate']:.3f} "
               f"UA={po_metrics['unnecessary_abstention_rate']:.3f}", log_path)

    # 5c: Alpha sweep with steering vector
    alphas = config["steering"]["alphas"]
    _log_print(f"\n  [5c] Alpha sweep: {alphas}", log_path)
    for alpha in alphas:
        label = f"steering_a{alpha}"
        _log_print(f"    {label}...", log_path)
        res, _ = run_generation_with_steering(
            model, tokenizer, test,
            steering_vector=vectors["steering"],
            steering_layer=target_layer,
            alpha=alpha,
            mode="steering",
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )
        metrics = evaluate_outputs(res)
        metrics["mode"] = label
        metrics["alpha"] = alpha
        all_rows.append(metrics)
        _log_print(f"    {label}: H={metrics['hallucination_rate']:.3f} "
                   f"C={metrics['correct_answer_rate']:.3f} "
                   f"A={metrics['calibrated_abstention_rate']:.3f} "
                   f"UA={metrics['unnecessary_abstention_rate']:.3f}", log_path)

    # 5d: Random vector control
    _log_print("\n  [5d] Random vector control...", log_path)
    for alpha in alphas:
        if alpha == 0.0:
            continue
        label = f"random_a{alpha}"
        _log_print(f"    {label}...", log_path)
        res, _ = run_generation_with_steering(
            model, tokenizer, test,
            steering_vector=vectors["random"],
            steering_layer=target_layer,
            alpha=alpha,
            mode="random",
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )
        metrics = evaluate_outputs(res)
        metrics["mode"] = label
        metrics["alpha"] = alpha
        all_rows.append(metrics)
        _log_print(f"    {label}: H={metrics['hallucination_rate']:.3f} "
                   f"C={metrics['correct_answer_rate']:.3f} "
                   f"A={metrics['calibrated_abstention_rate']:.3f} "
                   f"UA={metrics['unnecessary_abstention_rate']:.3f}", log_path)

    # 5e: Shuffled-label vector control
    _log_print("\n  [5e] Shuffled-label vector control...", log_path)
    for alpha in alphas:
        if alpha == 0.0:
            continue
        label = f"shuffled_a{alpha}"
        _log_print(f"    {label}...", log_path)
        res, _ = run_generation_with_steering(
            model, tokenizer, test,
            steering_vector=vectors["shuffled"],
            steering_layer=target_layer,
            alpha=alpha,
            mode="shuffled",
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )
        metrics = evaluate_outputs(res)
        metrics["mode"] = label
        metrics["alpha"] = alpha
        all_rows.append(metrics)
        _log_print(f"    {label}: H={metrics['hallucination_rate']:.3f} "
                   f"C={metrics['correct_answer_rate']:.3f} "
                   f"A={metrics['calibrated_abstention_rate']:.3f} "
                   f"UA={metrics['unnecessary_abstention_rate']:.3f}", log_path)

    # --- Save metrics ---
    df = pd.DataFrame(all_rows)
    cols = [
        "mode", "alpha",
        "hallucination_rate", "calibrated_abstention_rate",
        "correct_answer_rate", "unnecessary_abstention_rate",
        "style_only_score",
        "answerable_count", "unanswerable_count",
        "hallucination_count", "calibrated_abstention_count",
        "correct_count", "unnecessary_abstention_count",
        "avg_answerable_uncertainty", "avg_unanswerable_uncertainty",
    ]
    df = df[[c for c in cols if c in df.columns]]
    metrics_path = config["output"]["results_dir"] + "/metrics.csv"
    os.makedirs(config["output"]["results_dir"], exist_ok=True)
    df.to_csv(metrics_path, index=False)
    _log_print(f"\n[Task 6] Metrics saved to {metrics_path}", log_path)

    # --- Task 7: Report ---
    _log_print("\n[Task 7] Generating report...", log_path)

    from src.report_writer import generate_report, compute_verdict

    verdict, verdict_reason, verdict_deltas = compute_verdict(df)

    report_path = config["output"]["reports_dir"] + "/IC4_M0_CAUTION_STEERING_REPORT.md"
    generate_report(
        report_path=report_path,
        model_name=config["model"]["name"],
        train_size=len(train),
        test_size=len(test),
        target_layer=target_layer,
        total_layers=total_layers,
        alphas=alphas,
        df=df,
        verdict=verdict,
        verdict_reason=verdict_reason,
        base_metrics=base_metrics,
        po_metrics=po_metrics,
    )
    _log_print(f"  Report saved to {report_path}", log_path)

    _log_print("\n" + "=" * 60, log_path)
    _log_print(f"IC-4-M0 complete. Verdict: {verdict}", log_path)
    _log_print("=" * 60, log_path)


if __name__ == "__main__":
    main()