"""
IC-4-M3-O: Oracle-Gated Steering — Smoke Test.
Tests upper bound: if a perfect gate separates answerable/unanswerable samples,
can the current steering vector reduce hallucination without damaging correctness?

Usage:
    python -m src.run_m3_oracle --config configs/config_m3_oracle.yaml
"""

import argparse
import os
import sys
import random
import time
import json
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.run_m2 import load_config


def _log(msg: str, log_file: str = None):
    print(msg, flush=True)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def _build_data(seed, config, train_path, test_path, log):
    from src.data_builder import load_jsonl

    train_final = train_path.replace(".jsonl", f"_s{seed}.jsonl")
    test_final = test_path.replace(".jsonl", f"_s{seed}.jsonl")

    if not os.path.exists(train_final) or not os.path.exists(test_final):
        raise FileNotFoundError(f"M3-v1 cached data not found at {train_final} / {test_final}")

    _log(f"  Loading cached data for seed {seed}", log)
    train = load_jsonl(train_final)
    test = load_jsonl(test_final)

    num_a = sum(1 for s in train if s["answerability"] == "answerable")
    num_u = len(train) - num_a
    num_a_test = sum(1 for s in test if s["answerability"] == "answerable")
    num_u_test = len(test) - num_a_test
    _log(f"  seed={seed}: train {num_a}A+{num_u}U, test {num_a_test}A+{num_u_test}U", log)
    return train, test


def _generate_oracle_gated(model, tokenizer, test_data, steering_vector, layer_idx,
                            alpha, mode_label, gen_cfg):
    """
    Per-sample oracle-gated generation.

    For each sample:
      - If answerability == "answerable": NO steering hook (alpha=0)
      - If answerability == "unanswerable": steering hook with given alpha
    """
    from src.evaluate import generate_answers
    from src.steering import apply_steering_hook

    max_new = gen_cfg.get("max_new_tokens", 48)
    temp = gen_cfg.get("temperature", 0.0)
    do_sample = gen_cfg.get("do_sample", False)
    device = next(model.parameters()).device

    results = []
    answerable_samples = [s for s in test_data if s.get("answerability") == "answerable"]
    unanswerable_samples = [s for s in test_data if s.get("answerability") == "unanswerable"]

    if answerable_samples:
        _log(f"    oracle_gate: {len(answerable_samples)} answerable (no steer)", None)
        ans_results = generate_answers(
            model, tokenizer, answerable_samples, mode="base",
            max_new_tokens=max_new, temperature=temp, do_sample=do_sample,
        )
        for r in ans_results:
            r["mode"] = mode_label
            r["alpha"] = alpha
            r["alpha_applied"] = 0.0
            r["vector_type"] = "steering"
        results.extend(ans_results)

    if unanswerable_samples:
        _log(f"    oracle_gate: {len(unanswerable_samples)} unanswerable (steer a={alpha:+.2f})", None)
        handle = apply_steering_hook(model, layer_idx, steering_vector, alpha)
        unans_results = generate_answers(
            model, tokenizer, unanswerable_samples, mode="steering",
            max_new_tokens=max_new, temperature=temp, do_sample=do_sample,
        )
        handle.remove()
        for r in unans_results:
            r["mode"] = mode_label
            r["alpha"] = alpha
            r["alpha_applied"] = alpha
            r["vector_type"] = "steering"
        results.extend(unans_results)

    return results


def _generate_control_oracle_gated(model, tokenizer, test_data, control_vector, layer_idx,
                                    alpha, mode_label, control_type, gen_cfg):
    """
    Same oracle gate but with a control vector (random or shuffled).
    """
    from src.evaluate import generate_answers
    from src.steering import apply_steering_hook

    max_new = gen_cfg.get("max_new_tokens", 48)
    temp = gen_cfg.get("temperature", 0.0)
    do_sample = gen_cfg.get("do_sample", False)

    results = []
    answerable_samples = [s for s in test_data if s.get("answerability") == "answerable"]
    unanswerable_samples = [s for s in test_data if s.get("answerability") == "unanswerable"]

    if answerable_samples:
        ans_results = generate_answers(
            model, tokenizer, answerable_samples, mode="base",
            max_new_tokens=max_new, temperature=temp, do_sample=do_sample,
        )
        for r in ans_results:
            r["mode"] = mode_label
            r["alpha"] = alpha
            r["alpha_applied"] = 0.0
            r["vector_type"] = control_type
        results.extend(ans_results)

    if unanswerable_samples:
        handle = apply_steering_hook(model, layer_idx, control_vector, alpha)
        unans_results = generate_answers(
            model, tokenizer, unanswerable_samples, mode=control_type,
            max_new_tokens=max_new, temperature=temp, do_sample=do_sample,
        )
        handle.remove()
        for r in unans_results:
            r["mode"] = mode_label
            r["alpha"] = alpha
            r["alpha_applied"] = alpha
            r["vector_type"] = control_type
        results.extend(unans_results)

    return results


def _build_per_sample_csv(all_results, path):
    rows = []
    for i, r in enumerate(all_results):
        if r.get("mode", "").startswith("oracle_gate") or r.get("mode", "").startswith("random_oracle") or r.get("mode", "").startswith("shuffled_oracle"):
            pass
        rows.append({
            "sample_id": i,
            "label": r.get("answerability", "?"),
            "mode": r.get("mode", "?"),
            "alpha": r.get("alpha", 0),
            "alpha_applied": r.get("alpha_applied", 0),
            "vector_type": r.get("vector_type", "none"),
            "prediction": r.get("generated_output", ""),
            "gold_answer": r.get("gold_answer", ""),
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def _compute_oracle_verdict(df):
    base_row = df[df["mode"] == "base"]
    if len(base_row) == 0:
        return "IC4_M3_ORACLE_GATE_NULL", "No base metrics found."

    base_h = float(base_row.iloc[0]["hallucination_rate"])
    base_c = float(base_row.iloc[0]["correct_answer_rate"])
    base_ua = float(base_row.iloc[0]["unnecessary_abstention_rate"])

    oracle_rows = df[df["mode"].str.startswith("oracle_gate_a")]
    if len(oracle_rows) == 0:
        return "IC4_M3_ORACLE_GATE_NULL", "No oracle gate results found."

    best_oracle = oracle_rows.loc[oracle_rows["hallucination_rate"].idxmin()]
    oracle_h = float(best_oracle["hallucination_rate"])
    oracle_c = float(best_oracle["correct_answer_rate"])
    oracle_ua = float(best_oracle["unnecessary_abstention_rate"])
    oracle_alpha = float(best_oracle["alpha"])

    open_rows = df[df["mode"].str.startswith("steering_a")]
    best_open = open_rows.loc[open_rows["hallucination_rate"].idxmin()] if len(open_rows) > 0 else None
    best_open_h = float(best_open["hallucination_rate"]) if best_open is not None else 1.0

    h_ok = (oracle_h <= base_h - 0.15)
    c_ok = (oracle_c >= base_c - 0.03)
    ua_ok = (oracle_ua <= base_ua + 0.03)

    random_oracle_rows = df[df["mode"].str.startswith("random_oracle_gate")]
    shuffled_oracle_rows = df[df["mode"].str.startswith("shuffled_oracle_gate")]

    control_ok = True
    control_detail = ""

    if len(random_oracle_rows) > 0:
        rand_h = float(random_oracle_rows.iloc[0]["hallucination_rate"])
        rand_c = float(random_oracle_rows.iloc[0]["correct_answer_rate"])
        rand_ua = float(random_oracle_rows.iloc[0]["unnecessary_abstention_rate"])
        if not (oracle_h <= rand_h - 0.05) or abs(oracle_c - rand_c) > 0.05 or abs(oracle_ua - rand_ua) > 0.05:
            control_ok = False
            control_detail += f" random(h={rand_h:.3f} c={rand_c:.3f} ua={rand_ua:.3f})"

    if len(shuffled_oracle_rows) > 0:
        shuf_h = float(shuffled_oracle_rows.iloc[0]["hallucination_rate"])
        shuf_c = float(shuffled_oracle_rows.iloc[0]["correct_answer_rate"])
        shuf_ua = float(shuffled_oracle_rows.iloc[0]["unnecessary_abstention_rate"])
        if not (oracle_h <= shuf_h - 0.05) or abs(oracle_c - shuf_c) > 0.05 or abs(oracle_ua - shuf_ua) > 0.05:
            control_ok = False
            control_detail += f" shuffled(h={shuf_h:.3f} c={shuf_c:.3f} ua={shuf_ua:.3f})"

    if not control_ok:
        return ("IC4_M3_ORACLE_GATE_CONTROL_ARTIFACT",
                f"Oracle-gated real steering not clearly better than controls. Best oracle: "
                f"alpha={oracle_alpha:+.2f} H={oracle_h:.3f} C={oracle_c:.3f} UA={oracle_ua:.3f}.{control_detail}")

    if h_ok and c_ok and ua_ok:
        return ("IC4_M3_ORACLE_GATE_SUCCESS",
                f"Oracle gate succeeds: with perfect gate, steering vector reduces H from {base_h:.3f} to "
                f"{oracle_h:.3f} (dH=-{base_h - oracle_h:.3f}) while C stays at {oracle_c:.3f} "
                f"(base C={base_c:.3f}) and UA at {oracle_ua:.3f} (base UA={base_ua:.3f}). "
                f"Best alpha={oracle_alpha:+.2f}. Controller/gating signal route is worth pursuing.")

    if h_ok and not (c_ok and ua_ok):
        return ("IC4_M3_ORACLE_GATE_VECTOR_DIRTY",
                f"Oracle gate reduces H from {base_h:.3f} to {oracle_h:.3f} but C={oracle_c:.3f} "
                f"(base C={base_c:.3f}) or UA={oracle_ua:.3f} (base UA={base_ua:.3f}) still damaged. "
                f"The steering vector itself is not clean — even a perfect gate cannot prevent "
                f"collateral damage. Return to system identification.")

    return ("IC4_M3_ORACLE_GATE_NULL",
            f"Oracle gate h={oracle_h:.3f} vs base h={base_h:.3f}: no meaningful hallucination reduction.")


def main():
    parser = argparse.ArgumentParser(description="IC-4-M3-O: Oracle-Gated Steering")
    parser.add_argument("--config", type=str, default="configs/config_m3_oracle.yaml")
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
    _log("IC-4-M3-O: Oracle-Gated Steering (Smoke)", log_path)
    _log("=" * 60, log_path)

    seeds = config["steering"]["seeds"]
    layers = config["steering"]["layers"]
    alphas = config["steering"]["alphas"]
    gen_cfg = config["generation"]

    _log(f"\nConfig: seed={seeds}, layer={layers}, alphas={alphas}", log_path)
    _log(f"Data: train={config['data']['train_size']}, test={config['data']['test_size']}", log_path)

    total_evals = (2 + len(alphas) + len(alphas) + 2) * len(seeds) * len(layers)
    _log(f"Estimated evaluations: {total_evals} "
         f"(base+po + open_loop×{len(alphas)} + oracle_gate×{len(alphas)} + 2 controls)", log_path)

    from src.model_loader import load_model_and_tokenizer, get_model_layer_count

    _log(f"\nLoading model ({config['model']['name']})...", log_path)
    model, tokenizer = load_model_and_tokenizer(
        model_name=config["model"]["name"],
        device=config["model"]["device"],
        torch_dtype=config["model"].get("torch_dtype", "float32"),
    )
    total_layers = get_model_layer_count(model)
    _log(f"  Total layers: {total_layers}", log_path)

    all_metrics_rows = []
    all_per_sample_results = []
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

        from src.evaluate import generate_answers, evaluate_outputs, run_generation_with_steering

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
        base_metrics["vector_type"] = "none"
        all_metrics_rows.append(base_metrics)
        _log(f"    base: H={base_metrics['hallucination_rate']:.3f} "
             f"C={base_metrics['correct_answer_rate']:.3f} "
             f"UA={base_metrics['unnecessary_abstention_rate']:.3f}", log_path)
        for r in base_results:
            r["mode"] = "base"; r["alpha"] = 0.0; r["alpha_applied"] = 0.0; r["vector_type"] = "none"
        all_per_sample_results.extend(base_results)

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
        po_metrics["vector_type"] = "none"
        all_metrics_rows.append(po_metrics)
        _log(f"    prompt_only: H={po_metrics['hallucination_rate']:.3f} "
             f"C={po_metrics['correct_answer_rate']:.3f} "
             f"UA={po_metrics['unnecessary_abstention_rate']:.3f}", log_path)
        for r in po_results:
            r["mode"] = "prompt_only"; r["alpha"] = 0.0; r["alpha_applied"] = 0.0; r["vector_type"] = "none"
        all_per_sample_results.extend(po_results)

        from src.activation_collector import load_activations
        from src.steering import get_all_vectors

        for layer_idx in layers:
            layer_t0 = time.time()
            _log(f"\n  LAYER {layer_idx}", log_path)

            act_path = os.path.join("results_m3", f"activations_s{seed}_l{layer_idx}.npz")
            _log(f"    Loading activations from {act_path}", log_path)
            acts = load_activations(act_path)
            hidden_dim = acts["positive"].shape[1]
            _log(f"    Loaded: {acts['positive'].shape[0]} pairs, dim={hidden_dim}", log_path)

            all_vectors = get_all_vectors(acts["positive"], acts["negative"], hidden_dim)

            _log(f"\n  Open-loop steering:", log_path)
            for alpha in alphas:
                results, _ = run_generation_with_steering(
                    model, tokenizer, test, all_vectors["steering"], layer_idx,
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
                metrics["vector_type"] = "steering"
                all_metrics_rows.append(metrics)
                _log(f"    open_a{alpha:+0.2f}: H={metrics['hallucination_rate']:.3f} "
                     f"C={metrics['correct_answer_rate']:.3f} "
                     f"UA={metrics['unnecessary_abstention_rate']:.3f}", log_path)
                for r in results:
                    r["alpha_applied"] = alpha; r["vector_type"] = "steering"
                all_per_sample_results.extend(results)

            _log(f"\n  Oracle-gated steering:", log_path)
            for alpha in alphas:
                mode_label = f"oracle_gate_a{alpha}"
                results = _generate_oracle_gated(
                    model, tokenizer, test, all_vectors["steering"], layer_idx,
                    alpha, mode_label, gen_cfg,
                )
                metrics = evaluate_outputs(results)
                metrics["seed"] = seed
                metrics["layer"] = layer_idx
                metrics["mode"] = mode_label
                metrics["alpha"] = alpha
                metrics["vector_type"] = "steering"
                all_metrics_rows.append(metrics)
                _log(f"    oracle_a{alpha:+0.2f}: H={metrics['hallucination_rate']:.3f} "
                     f"C={metrics['correct_answer_rate']:.3f} "
                     f"UA={metrics['unnecessary_abstention_rate']:.3f}", log_path)
                all_per_sample_results.extend(results)

            _log(f"\n  Control oracle-gated:", log_path)

            mode_label = "random_oracle_gate_a-1.0"
            results = _generate_control_oracle_gated(
                model, tokenizer, test, all_vectors["random"], layer_idx,
                -1.0, mode_label, "random", gen_cfg,
            )
            metrics = evaluate_outputs(results)
            metrics["seed"] = seed
            metrics["layer"] = layer_idx
            metrics["mode"] = mode_label
            metrics["alpha"] = -1.0
            metrics["vector_type"] = "random"
            all_metrics_rows.append(metrics)
            _log(f"    random_oracle_a-1.00: H={metrics['hallucination_rate']:.3f} "
                 f"C={metrics['correct_answer_rate']:.3f} "
                 f"UA={metrics['unnecessary_abstention_rate']:.3f}", log_path)
            all_per_sample_results.extend(results)

            mode_label = "shuffled_oracle_gate_a-1.0"
            results = _generate_control_oracle_gated(
                model, tokenizer, test, all_vectors["shuffled"], layer_idx,
                -1.0, mode_label, "shuffled", gen_cfg,
            )
            metrics = evaluate_outputs(results)
            metrics["seed"] = seed
            metrics["layer"] = layer_idx
            metrics["mode"] = mode_label
            metrics["alpha"] = -1.0
            metrics["vector_type"] = "shuffled"
            all_metrics_rows.append(metrics)
            _log(f"    shuffled_oracle_a-1.00: H={metrics['hallucination_rate']:.3f} "
                 f"C={metrics['correct_answer_rate']:.3f} "
                 f"UA={metrics['unnecessary_abstention_rate']:.3f}", log_path)
            all_per_sample_results.extend(results)

            _log(f"    layer {layer_idx} done in {time.time() - layer_t0:.0f}s", log_path)

    elapsed = time.time() - t_start
    _log(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)", log_path)

    # ── Save metrics ──
    df = pd.DataFrame(all_metrics_rows)
    cols_order = [
        "seed", "layer", "mode", "alpha", "vector_type",
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

    # ── Per-sample CSV ──
    per_sample_path = os.path.join(results_dir, "per_sample_oracle.csv")
    _build_per_sample_csv(all_per_sample_results, per_sample_path)
    _log(f"Per-sample CSV saved to {per_sample_path} ({len(all_per_sample_results)} rows)", log_path)

    # ── Comparison table ──
    _log(f"\n{'='*60}", log_path)
    _log(f"COMPARISON: Open-Loop vs Oracle-Gated", log_path)
    _log(f"{'='*60}", log_path)
    _log(f"{'Mode':<28} {'H':>8} {'C':>8} {'UA':>8} {'Vector':>10}", log_path)
    _log(f"{'-'*28} {'-'*8} {'-'*8} {'-'*8} {'-'*10}", log_path)

    for _, row in df.iterrows():
        mode = str(row.get("mode", ""))
        vt = str(row.get("vector_type", ""))
        _log(f"{mode:<28} {row['hallucination_rate']:8.3f} "
             f"{row['correct_answer_rate']:8.3f} "
             f"{row['unnecessary_abstention_rate']:8.3f} "
             f"{vt:>10}", log_path)

    # ── Verdict ──
    verdict, verdict_reason = _compute_oracle_verdict(df)
    _log(f"\n{'='*60}", log_path)
    _log(f"VERDICT: {verdict}", log_path)
    _log(f"{'='*60}", log_path)
    _log(f"Reason: {verdict_reason}", log_path)

    # ── Generate report ──
    report_path = os.path.join(reports_dir, "IC4_M3_ORACLE_GATE_REPORT.md")
    _generate_report(report_path, config, df, verdict, verdict_reason, elapsed)
    _log(f"\nReport saved to {report_path}", log_path)

    _log(f"\n{'='*60}", log_path)
    _log(f"IC-4-M3-O complete. Verdict: {verdict}", log_path)
    _log(f"{'='*60}", log_path)


def _generate_report(report_path, config, df, verdict, verdict_reason, elapsed):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    base_row = df[df["mode"] == "base"].iloc[0]
    base_h = float(base_row["hallucination_rate"])
    base_c = float(base_row["correct_answer_rate"])
    base_ua = float(base_row["unnecessary_abstention_rate"])

    po_row = df[df["mode"] == "prompt_only"].iloc[0]

    lines = []
    lines.append("# IC-4-M3-O: Oracle-Gated Steering Report")
    lines.append("")

    lines.append("## 1. M3-v1 Recap")
    lines.append("")
    lines.append("M3-v1 tested entropy-driven per-token feedback steering vs open-loop at layer=12, seed=0, 30-train/60-test.")
    lines.append("")
    lines.append("| Mode | H | C | UA |")
    lines.append("|---|---|---|---|")
    lines.append(f"| base | {base_h:.3f} | {base_c:.3f} | {base_ua:.3f} |")
    lines.append(f"| prompt_only | {float(po_row['hallucination_rate']):.3f} | {float(po_row['correct_answer_rate']):.3f} | {float(po_row['unnecessary_abstention_rate']):.3f} |")
    for _, row in df.iterrows():
        mode = str(row.get("mode", ""))
        if mode.startswith("steering_a"):
            lines.append(f"| open {mode} | {row['hallucination_rate']:.3f} | {row['correct_answer_rate']:.3f} | {row['unnecessary_abstention_rate']:.3f} |")
        elif mode.startswith("feedback_a"):
            k_val = row.get("k", "?")
            lines.append(f"| fb_a-1.00_k{k_val} | {row['hallucination_rate']:.3f} | {row['correct_answer_rate']:.3f} | {row['unnecessary_abstention_rate']:.3f} |")
    lines.append("")
    lines.append("**Finding**: Entropy feedback v1 did not activate -- baseline entropy was computed under the same steering, rendering the feedback signal inert. Cannot conclude feedback *fails*; only that v1 implementation did not work.")
    lines.append("")

    lines.append("## 2. Why Oracle Gate First?")
    lines.append("")
    lines.append("Before investing in better gating signals (entropy telemetry, probe gate), we must answer a more fundamental question:")
    lines.append("")
    lines.append("> If the gate were *perfect* -- applying steering *only* to unanswerable samples and never to answerable ones -- would the current steering vector v achieve its goal?")
    lines.append("")
    lines.append("- **If yes (Oracle Gate SUCCESS)**: v is valuable. The problem is the gating/feedback signal. Proceed to build better gates (entropy telemetry v2, probe gate v3).")
    lines.append("- **If no (Oracle Gate VECTOR_DIRTY)**: v itself is not clean. Even a perfect gate cannot prevent collateral damage. Return to system identification for a better v.")
    lines.append("")

    lines.append("## 3. Experiment Configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append(f"| Model | {config['model']['name']} |")
    lines.append(f"| Device / dtype | {config['model']['device']} / {config['model']['torch_dtype']} |")
    lines.append(f"| Train / Test size | {config['data']['train_size']} / {config['data']['test_size']} |")
    lines.append(f"| Seeds | {config['steering']['seeds']} |")
    lines.append(f"| Layer | {config['steering']['layers']} |")
    lines.append(f"| Alphas | {config['steering']['alphas']} |")
    lines.append(f"| Temperature / do_sample | {config['generation']['temperature']} / {config['generation']['do_sample']} |")
    lines.append(f"| max_new_tokens | {config['generation']['max_new_tokens']} |")
    lines.append(f"| Elapsed | {elapsed:.0f}s ({elapsed/60:.1f} min) |")
    lines.append("")

    lines.append("## 4. Full Metrics Table")
    lines.append("")
    lines.append("| Mode | H | C | UA | CA | Vector |")
    lines.append("|---|---|---|---|---|---|")
    for _, row in df.iterrows():
        vt = str(row.get("vector_type", "none"))
        lines.append(f"| {row['mode']} | {row['hallucination_rate']:.3f} | {row['correct_answer_rate']:.3f} | {row['unnecessary_abstention_rate']:.3f} | {row.get('calibrated_abstention_rate', 0):.3f} | {vt} |")
    lines.append("")

    lines.append("## 5. Real vs Random/Shuffled Oracle-Gate Comparison")
    lines.append("")
    oracle_rows = df[df["mode"].str.startswith("oracle_gate_a")]
    best_idx = oracle_rows["hallucination_rate"].idxmin() if len(oracle_rows) > 0 else None

    if best_idx is not None:
        best_oracle = df.loc[best_idx]
        oracle_h = float(best_oracle["hallucination_rate"])
        oracle_c = float(best_oracle["correct_answer_rate"])
        oracle_ua = float(best_oracle["unnecessary_abstention_rate"])
        oracle_a = float(best_oracle["alpha"])

        lines.append(f"Best oracle gate: alpha={oracle_a:+.2f}, H={oracle_h:.3f}, C={oracle_c:.3f}, UA={oracle_ua:.3f}")
        lines.append("")

        rand_row = df[df["mode"] == "random_oracle_gate_a-1.0"]
        shuf_row = df[df["mode"] == "shuffled_oracle_gate_a-1.0"]

        lines.append("| Comparison | H | C | UA | Gap vs Real H | Verdict |")
        lines.append("|---|---|---|---|---|---|")

        if len(rand_row) > 0:
            rand_h = float(rand_row.iloc[0]["hallucination_rate"])
            rand_c = float(rand_row.iloc[0]["correct_answer_rate"])
            rand_ua = float(rand_row.iloc[0]["unnecessary_abstention_rate"])
            gap_rand = rand_h - oracle_h
            ok_rand = "PASS" if oracle_h <= rand_h - 0.05 else "FAIL"
            lines.append(f"| random_oracle | {rand_h:.3f} | {rand_c:.3f} | {rand_ua:.3f} | {gap_rand:+.3f} | {ok_rand} |")

        if len(shuf_row) > 0:
            shuf_h = float(shuf_row.iloc[0]["hallucination_rate"])
            shuf_c = float(shuf_row.iloc[0]["correct_answer_rate"])
            shuf_ua = float(shuf_row.iloc[0]["unnecessary_abstention_rate"])
            gap_shuf = shuf_h - oracle_h
            ok_shuf = "PASS" if oracle_h <= shuf_h - 0.05 else "FAIL"
            lines.append(f"| shuffled_oracle | {shuf_h:.3f} | {shuf_c:.3f} | {shuf_ua:.3f} | {gap_shuf:+.3f} | {ok_shuf} |")
    lines.append("")

    lines.append("## 6. Verdict")
    lines.append("")
    lines.append(f"**Verdict: `{verdict}`**")
    lines.append("")
    lines.append(f"**Reasoning:** {verdict_reason}")
    lines.append("")
    lines.append("### Interpretation")
    lines.append("")
    if verdict == "IC4_M3_ORACLE_GATE_SUCCESS":
        lines.append("- The current steering vector v, when applied only to unanswerable queries, successfully reduces hallucination without damaging correctness or causing unnecessary abstention.")
        lines.append("- The vector direction is fundamentally useful. The limiting factor is the gating/feedback mechanism.")
    elif verdict == "IC4_M3_ORACLE_GATE_VECTOR_DIRTY":
        lines.append("- Even with a perfect gate (only steered on unanswerable samples), C or UA are still damaged.")
        lines.append("- This means the steering vector v encodes caution/refusal behavior that spills into answerable cases or fails to cleanly address hallucination.")
        lines.append("- Return to system identification: the problem is in how v is computed, not in how it's gated.")
    elif verdict == "IC4_M3_ORACLE_GATE_CONTROL_ARTIFACT":
        lines.append("- Random or shuffled oracle gates perform as well as or better than the real steering vector.")
        lines.append("- The steering effect is not specific to the circuit; random perturbations achieve the same result.")
        lines.append("- Consider data leakage or degenerate vector properties.")
    elif verdict == "IC4_M3_ORACLE_GATE_NULL":
        lines.append("- Oracle gate does not meaningfully reduce hallucination beyond base levels.")
        lines.append("- The current v may simply not contain an anti-hallucination signal at this layer.")
    lines.append("")

    lines.append("## 7. Next Recommendation")
    lines.append("")
    if verdict == "IC4_M3_ORACLE_GATE_SUCCESS":
        lines.append("- **Proceed with gating/feedback development**:")
        lines.append("  - M3-v2: entropy telemetry -- fix baseline (unsteered reference), record alpha_t/gate_t/entropy_t per token")
        lines.append("  - M3-v3: probe-gated steering -- train a lightweight answerability/hallucination-risk probe on hidden states")
    elif verdict == "IC4_M3_ORACLE_GATE_VECTOR_DIRTY":
        lines.append("- **Stop PID/entropy feedback route for now.**")
        lines.append("- **Return to system identification**: re-extract a cleaner anti-hallucination vector via:")
        lines.append("  - Contrastive pairs: hallucination vs correct (rather than hallucination vs abstention)")
        lines.append("  - SAE features: dictionary learning to isolate hallucination-related directions")
        lines.append("  - Hallucination subtype vectors: named-entity errors vs relation errors vs attribute errors")
        lines.append("  - Probe-gradient steering: use a trained hallucination probe's gradient as the steering direction")
    elif verdict == "IC4_M3_ORACLE_GATE_CONTROL_ARTIFACT":
        lines.append("- Audit data construction for train/test leakage via entity pool overlap.")
        lines.append("- Test with out-of-distribution entity pools to verify circuit specificity.")
        lines.append("- Re-examine vector normalization and hook implementation for artifacts.")
    else:
        lines.append("- Test additional layers and larger training sets.")
        lines.append("- Consider alternative model architectures or generation strategies.")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*IC-4-M3-O: Oracle-Gated Steering — Upper Bound Diagnostic*")
    lines.append("*Generated by run_m3_oracle*")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()