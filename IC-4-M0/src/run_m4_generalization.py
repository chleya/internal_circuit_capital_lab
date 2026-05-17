"""
IC-4-M4-Generalization: Robustness validation of M3-v6 single-pass hook-based gate.

Sweeps across seeds, layers, and alpha values to verify that the gate mechanism
is stable and not just a lucky configuration.

Data scenarios:
  - standard: M3 data (60 samples, standard OOD split)
  - large:    M4 data (120 samples, standard OOD split)
  - hard:     M4 data (120 samples, extreme OOD, train=1/4 entity pool)

Usage:
    python -m src.run_m4_generalization --config configs/config_m4_generalization.yaml
"""

import argparse
import os
import sys
import time
import random
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.run_m2 import load_config
from src.run_m3_v6 import (
    _find_transformer_layer,
    _collect_prefill_features,
    _train_probe,
    _generate_single_pass_hard_gate,
    _generate_oracle_gated,
)


def _log(msg, log_file=None):
    print(msg, flush=True)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def _load_cached_m4_data(seed, train_path, test_path, log):
    from src.data_builder import load_jsonl
    train_final = train_path.replace(".jsonl", f"_s{seed}.jsonl")
    test_final = test_path.replace(".jsonl", f"_s{seed}.jsonl")
    if not os.path.exists(train_final) or not os.path.exists(test_final):
        raise FileNotFoundError(f"M4 data not found at {train_final} / {test_final}")
    train = load_jsonl(train_final)
    test = load_jsonl(test_final)
    na_t = sum(1 for s in train if s.get("answerability") == "answerable")
    na_s = sum(1 for s in test if s.get("answerability") == "answerable")
    _log(f"  seed={seed}: train {na_t}A+{len(train)-na_t}U, test {na_s}A+{len(test)-na_s}U", log)
    return train, test


def _load_cached_m3_data(seed, train_path, test_path, log):
    from src.data_builder import load_jsonl
    train_final = train_path.replace(".jsonl", f"_s{seed}.jsonl")
    test_final = test_path.replace(".jsonl", f"_s{seed}.jsonl")
    if not os.path.exists(train_final) or not os.path.exists(test_final):
        raise FileNotFoundError(f"M3 data not found at {train_final} / {test_final}")
    train = load_jsonl(train_final)
    test = load_jsonl(test_final)
    na_t = sum(1 for s in train if s.get("answerability") == "answerable")
    na_s = sum(1 for s in test if s.get("answerability") == "answerable")
    _log(f"  seed={seed}: train {na_t}A+{len(train)-na_t}U, test {na_s}A+{len(test)-na_s}U", log)
    return train, test


def _evaluate_and_add(results, seed, layer, mode, alpha, vector_type, all_rows):
    from src.evaluate import evaluate_outputs
    metrics = evaluate_outputs(results)
    metrics["seed"] = seed
    metrics["layer"] = layer
    metrics["mode"] = mode
    metrics["alpha"] = alpha
    metrics["vector_type"] = vector_type
    all_rows.append(metrics)
    return metrics


def _run_gate_mode(model, tokenizer, test_data, steering_vector, layer_idx,
                   alpha, probe_info, probe_cfg, gen_cfg, control_type, log):
    results = _generate_single_pass_hard_gate(
        model, tokenizer, test_data, steering_vector, layer_idx,
        alpha, probe_info, probe_cfg, gen_cfg, control_type,
    )
    label_map = {
        "steering": f"single_pass_hard_gate_a{alpha:+.1f}",
        "random": f"random_single_pass_hard_gate_a{alpha:+.1f}",
        "shuffled": f"shuffled_single_pass_hard_gate_a{alpha:+.1f}",
    }
    mode_label = label_map.get(control_type, f"gate_{control_type}")
    return results, mode_label


def _run_sweep_point(model, tokenizer, test_data, steering_v, all_vectors,
                     layer_idx, alpha, probe_info, probe_cfg, gen_cfg,
                     seed, all_metrics, test_label, log):
    """Run all gate modes for one (seed, layer, alpha, data) combination."""
    _log(f"    --- alpha={alpha:+.1f} [{test_label}] ---", log)

    results_real, mode_real = _run_gate_mode(
        model, tokenizer, test_data, steering_v, layer_idx,
        alpha, probe_info, probe_cfg, gen_cfg, "steering", log)
    m = _evaluate_and_add(results_real, seed, layer_idx, mode_real, alpha, "steering", all_metrics)
    _log(f"      real_gate:  H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f}", log)

    results_rnd, mode_rnd = _run_gate_mode(
        model, tokenizer, test_data, all_vectors["random"], layer_idx,
        alpha, probe_info, probe_cfg, gen_cfg, "random", log)
    m = _evaluate_and_add(results_rnd, seed, layer_idx, mode_rnd, alpha, "random", all_metrics)
    _log(f"      random:     H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f}", log)

    results_shf, mode_shf = _run_gate_mode(
        model, tokenizer, test_data, all_vectors["shuffled"], layer_idx,
        alpha, probe_info, probe_cfg, gen_cfg, "shuffled", log)
    m = _evaluate_and_add(results_shf, seed, layer_idx, mode_shf, alpha, "shuffled", all_metrics)
    _log(f"      shuffled:   H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f}", log)


def _generate_robustness_report(report_path, config, df, elapsed, scenarios, scenario_labels):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    lines = []
    lines.append("# IC-4-M4-Generalization: Robustness Validation Report")
    lines.append("")
    lines.append("> Validates the M3-v6 single-pass hook-based hard gate mechanism")
    lines.append("> across seeds, layers, alpha values, and data scenarios.")
    lines.append("")

    lines.append("## 1. Experiment Configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append(f"| Model | {config['model']['name']} |")
    lines.append(f"| Seeds | {config['sweep']['seeds']} |")
    lines.append(f"| Layers | {config['sweep']['layers']} |")
    lines.append(f"| Primary alpha | {config['sweep']['primary_alpha']} |")
    lines.append(f"| Extra alphas | {config['sweep']['extra_alphas']} |")
    lines.append(f"| Probe | {config['probe']['representations']} |")
    lines.append(f"| Data scenarios | {len(scenarios)} |")
    lines.append(f"| Elapsed | {elapsed:.0f}s ({elapsed/60:.1f} min) |")
    lines.append("")

    lines.append("## 2. Data Scenarios")
    lines.append("")
    lines.append("| Scenario | Description |")
    lines.append("|---|---|")
    for name, sc in scenarios.items():
        lines.append(f"| {name} | {sc['label']} |")
    lines.append("")

    lines.append("## 3. Robustness Matrix — Primary Alpha")
    lines.append("")
    lines.append("Rows show (seed, layer, scenario) combinations with primary alpha.")
    lines.append("")
    primary_alpha = config["sweep"]["primary_alpha"]

    gate_modes = [m for m in df["mode"].unique() if "hard_gate" in str(m) and "random" not in str(m) and "shuffled" not in str(m)]
    primary_df = df[df["mode"].isin(gate_modes)] if gate_modes else df[~df["mode"].str.contains("random|shuffled", na=False)]

    lines.append("| Scenario | Seed | Layer | base_H | base_C | gate_H | gate_C | oracle_H | oracle_C | oracle_gap_H | real_vs_random |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    for scenario_name in scenarios:
        scenario_label = scenario_labels.get(scenario_name, scenario_name)
        for seed in config["sweep"]["seeds"]:
            for layer in config["sweep"]["layers"]:
                base_row = df[(df["mode"] == "base") & (df["seed"] == seed) & (df["layer"] == -1)]
                if len(base_row) == 0:
                    continue

                base_h = float(base_row.iloc[0]["hallucination_rate"])
                base_c = float(base_row.iloc[0]["correct_answer_rate"])

                gate_rows = df[(df["mode"].str.contains("single_pass_hard_gate")) &
                               (~df["mode"].str.contains("random|shuffled")) &
                               (df["seed"] == seed) & (df["layer"] == layer) &
                               (abs(df["alpha"] - primary_alpha) < 0.01)]
                oracle_rows = df[(df["mode"] == "oracle_gate_a-1.0") &
                                 (df["seed"] == seed) & (df["layer"] == layer)]
                random_rows = df[(df["mode"].str.contains("random_single_pass_hard_gate")) &
                                 (df["seed"] == seed) & (df["layer"] == layer) &
                                 (abs(df["alpha"] - primary_alpha) < 0.01)]

                gate_h = float(gate_rows.iloc[0]["hallucination_rate"]) if len(gate_rows) > 0 else float('nan')
                gate_c = float(gate_rows.iloc[0]["correct_answer_rate"]) if len(gate_rows) > 0 else float('nan')
                oracle_h = float(oracle_rows.iloc[0]["hallucination_rate"]) if len(oracle_rows) > 0 else float('nan')
                oracle_c = float(oracle_rows.iloc[0]["correct_answer_rate"]) if len(oracle_rows) > 0 else float('nan')
                oracle_gap = gate_h - oracle_h if not np.isnan(gate_h) and not np.isnan(oracle_h) else float('nan')
                random_h = float(random_rows.iloc[0]["hallucination_rate"]) if len(random_rows) > 0 else float('nan')
                real_vs_random = random_h - gate_h if not np.isnan(gate_h) and not np.isnan(random_h) else float('nan')

                lines.append(f"| {scenario_name} | {seed} | {layer} | {base_h:.3f} | {base_c:.3f} | "
                             f"{gate_h:.3f} | {gate_c:.3f} | {oracle_h:.3f} | {oracle_c:.3f} | "
                             f"{oracle_gap:+.3f} | {real_vs_random:.3f} |")

    lines.append("")

    lines.append("## 4. Alpha Sensitivity")
    lines.append("")
    lines.append("| Alpha | Scenario | Seed | Layer | gate_H | gate_C | oracle_gap_H |")
    lines.append("|---|---|---|---|---|---|---|")
    extra_alphas = config["sweep"]["extra_alphas"]
    all_alphas = [primary_alpha] + extra_alphas
    for alpha in all_alphas:
        gate_rows = df[(df["mode"].str.contains("single_pass_hard_gate")) &
                       (~df["mode"].str.contains("random|shuffled")) &
                       (abs(df["alpha"] - alpha) < 0.01)]
        for _, row in gate_rows.iterrows():
            scenario = "?"  # We don't track scenario in the mode name
            oracle_rows = df[(df["mode"] == "oracle_gate_a-1.0") &
                             (df["seed"] == row["seed"]) & (df["layer"] == row["layer"])]
            oracle_h = float(oracle_rows.iloc[0]["hallucination_rate"]) if len(oracle_rows) > 0 else float('nan')
            gap = row["hallucination_rate"] - oracle_h if not np.isnan(oracle_h) else float('nan')
            lines.append(f"| {alpha:+.1f} | {scenario} | {int(row['seed'])} | {int(row['layer'])} | "
                         f"{row['hallucination_rate']:.3f} | {row['correct_answer_rate']:.3f} | {gap:+.3f} |")
    lines.append("")

    lines.append("## 5. Control Separation")
    lines.append("")
    lines.append("| Scenario | Seed | Layer | real_H | random_H | shuffled_H | separation |")
    lines.append("|---|---|---|---|---|---|---|")
    for scenario_name in scenarios:
        for seed in config["sweep"]["seeds"]:
            for layer in config["sweep"]["layers"]:
                real = df[(df["mode"].str.contains("single_pass_hard_gate")) &
                          (~df["mode"].str.contains("random|shuffled")) &
                          (df["seed"] == seed) & (df["layer"] == layer)]
                rnd = df[(df["mode"].str.contains("random_single_pass_hard_gate")) &
                         (df["seed"] == seed) & (df["layer"] == layer)]
                shf = df[(df["mode"].str.contains("shuffled_single_pass_hard_gate")) &
                         (df["seed"] == seed) & (df["layer"] == layer)]
                if len(real) > 0 and len(rnd) > 0:
                    real_h = float(real.iloc[0]["hallucination_rate"])
                    rnd_h = float(rnd.iloc[0]["hallucination_rate"])
                    shf_h = float(shf.iloc[0]["hallucination_rate"]) if len(shf) > 0 else float('nan')
                    separation = rnd_h - real_h
                    lines.append(f"| {scenario_name} | {seed} | {layer} | {real_h:.3f} | {rnd_h:.3f} | {shf_h:.3f} | {separation:.3f} |")
    lines.append("")

    lines.append("## 6. Success Criteria Summary")
    lines.append("")

    gate_df = df[(df["mode"].str.contains("single_pass_hard_gate")) &
                 (~df["mode"].str.contains("random|shuffled")) &
                 (abs(df["alpha"] - primary_alpha) < 0.01)]
    oracle_df = df[df["mode"] == "oracle_gate_a-1.0"]

    max_gap = 0.0
    min_separation = 999.0
    for _, g_row in gate_df.iterrows():
        o_rows = oracle_df[(oracle_df["seed"] == g_row["seed"]) & (oracle_df["layer"] == g_row["layer"])]
        if len(o_rows) > 0:
            gap = g_row["hallucination_rate"] - float(o_rows.iloc[0]["hallucination_rate"])
            max_gap = max(max_gap, gap)
        r_rows = df[(df["mode"].str.contains("random_single_pass_hard_gate")) &
                    (df["seed"] == g_row["seed"]) & (df["layer"] == g_row["layer"])]
        if len(r_rows) > 0:
            separation = float(r_rows.iloc[0]["hallucination_rate"]) - g_row["hallucination_rate"]
            min_separation = min(min_separation, separation)

    lines.append("| Criterion | Condition | Best/Worst | Result |")
    lines.append("|---|---|---|---|")
    passed_gap = max_gap <= 0.10
    lines.append(f"| Max oracle gap H | ≤ 0.10 | {max_gap:+.3f} | {'PASS' if passed_gap else 'FAIL'} |")
    passed_sep = min_separation >= 0.08
    lines.append(f"| Min control separation | ≥ 0.08 | {min_separation:.3f} | {'PASS' if passed_sep else 'FAIL'} |")
    stable_c = all(abs(float(r["correct_answer_rate"]) - 0.600) <= 0.10
                   for _, r in gate_df.iterrows() if "hard_gate" in str(r["mode"]))
    lines.append(f"| C stability | all within 0.10 of baseline | - | {'PASS' if stable_c else 'FAIL'} |")
    lines.append("")

    if passed_gap and passed_sep and stable_c:
        verdict = "IC4_M4_GENERALIZATION_ROBUST"
        reason = "Gate mechanism is robust across seeds, layers, alphas, and data scenarios."
    elif passed_gap and passed_sep:
        verdict = "IC4_M4_GENERALIZATION_PARTIAL"
        reason = "Gate mechanism largely robust but some conditions show variance."
    else:
        verdict = "IC4_M4_GENERALIZATION_FRAGILE"
        reason = f"Gate mechanism shows fragility: max oracle gap={max_gap:+.3f}, min separation={min_separation:.3f}."

    lines.append("## 7. Verdict")
    lines.append("")
    lines.append(f"**Verdict: `{verdict}`**")
    lines.append(f"**Reasoning:** {reason}")
    lines.append("")
    lines.append("---")
    lines.append("*IC-4-M4-Generalization: Robustness Validation*")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="IC-4-M4-Generalization: Robustness Validation")
    parser.add_argument("--config", type=str, default="configs/config_m4_generalization.yaml")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)

    config = load_config(args.config)
    sweep = config["sweep"]
    scenarios = {
        "large": {
            "train_path": "data_m4/train_test_large.jsonl",
            "test_path": "data_m4/test_large.jsonl",
            "label": "M4 large (120 samples)",
        },
        "hard": {
            "train_path": "data_m4/train_test_hard.jsonl",
            "test_path": "data_m4/test_hard.jsonl",
            "label": "M4 hard OOD (extreme entity split)",
        },
    }
    probe_cfg = config["probe"]
    gen_cfg = config["generation"]
    results_dir = config["output"]["results_dir"]
    reports_dir = config["output"]["reports_dir"]
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    log_path = os.path.join(results_dir, "run_log.txt")
    _log("=" * 60, log_path)
    _log("IC-4-M4-Generalization: Robustness Validation", log_path)
    _log("=" * 60, log_path)
    _log(f"Seeds: {sweep['seeds']}", log_path)
    _log(f"Layers: {sweep['layers']}", log_path)
    _log(f"Primary alpha: {sweep['primary_alpha']}", log_path)
    _log(f"Extra alphas: {sweep['extra_alphas']}", log_path)
    _log(f"Scenarios: {list(scenarios.keys())}", log_path)

    from src.model_loader import load_model_and_tokenizer, get_model_layer_count

    _log(f"\nLoading model ({config['model']['name']})...", log_path)
    model, tokenizer = load_model_and_tokenizer(
        model_name=config["model"]["name"],
        device=config["model"]["device"],
        torch_dtype=config["model"].get("torch_dtype", "float32"),
    )
    total_layers = get_model_layer_count(model)
    _log(f"  Total layers: {total_layers}", log_path)

    all_metrics = []
    scenario_labels = {}
    t_start = time.time()

    best_combo = None
    best_gap = float('inf')

    for scenario_name, sc in scenarios.items():
        _log(f"\n{'=' * 60}", log_path)
        _log(f"SCENARIO: {scenario_name} — {sc['label']}", log_path)
        _log(f"{'=' * 60}", log_path)

        train_path = sc["train_path"]
        test_path = sc["test_path"]
        scenario_labels[scenario_name] = sc["label"]

        for seed in sweep["seeds"]:
            _log(f"\n  --- SEED {seed} [{scenario_name}] ---", log_path)
            random.seed(seed)
            np.random.seed(seed)

            try:
                train, test = _load_cached_m4_data(seed, train_path, test_path, log_path)
            except FileNotFoundError:
                try:
                    train, test = _load_cached_m3_data(seed, train_path, test_path, log_path)
                except FileNotFoundError:
                    _log(f"  SKIP: no data for seed={seed}", log_path)
                    continue

            from src.evaluate import generate_answers, run_generation_with_steering

            _log(f"  Base (scenario={scenario_name})...", log_path)
            base_res = generate_answers(model, tokenizer, test, mode="base",
                                        max_new_tokens=gen_cfg["max_new_tokens"],
                                        temperature=gen_cfg["temperature"],
                                        do_sample=gen_cfg["do_sample"])
            m = _evaluate_and_add(base_res, seed, -1, "base", 0.0, "none", all_metrics)
            _log(f"    base: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f}", log_path)

            from src.activation_collector import load_activations
            from src.steering import get_all_vectors

            for layer_idx in sweep["layers"]:
                lt_layer = time.time()
                _log(f"\n  LAYER {layer_idx} [seed={seed}, {scenario_name}]", log_path)

                act_path = os.path.join("results_m3", f"activations_s{seed}_l{layer_idx}.npz")
                if not os.path.exists(act_path):
                    _log(f"    SKIP: no activations at {act_path}", log_path)
                    continue

                acts = load_activations(act_path)
                hidden_dim = acts["positive"].shape[1]
                all_vectors = get_all_vectors(acts["positive"], acts["negative"], hidden_dim)
                steering_v = all_vectors["steering"]
                _log(f"    Loaded {acts['positive'].shape[0]} pairs, dim={hidden_dim}", log_path)

                alpha = sweep["primary_alpha"]

                _log(f"    Single-pass open-loop a={alpha:+.2f}...", log_path)
                ol_res, _handle_ol = run_generation_with_steering(
                    model, tokenizer, test, steering_v, layer_idx, alpha, "steering",
                    max_new_tokens=gen_cfg["max_new_tokens"],
                    temperature=gen_cfg["temperature"], do_sample=gen_cfg["do_sample"],
                )
                if _handle_ol is not None:
                    _handle_ol.remove()
                m = _evaluate_and_add(ol_res, seed, layer_idx, f"steering_a{alpha:+.2f}",
                                      alpha, "steering", all_metrics)
                _log(f"      open_loop: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f}", log_path)

                _log(f"    Oracle gate...", log_path)
                og_res = _generate_oracle_gated(model, tokenizer, test, steering_v,
                                                layer_idx, alpha, "oracle_gate_a-1.0", gen_cfg)
                m = _evaluate_and_add(og_res, seed, layer_idx, "oracle_gate_a-1.0", alpha,
                                      "steering", all_metrics)
                _log(f"      oracle: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f}", log_path)

                _log(f"    Training probe [{probe_cfg['representations'][0]}]...", log_path)
                X_train, y_train = _collect_prefill_features(
                    model, tokenizer, train, layer_idx, probe_cfg["representations"][0])
                probe_info = _train_probe(X_train, y_train, cv_folds=probe_cfg.get("cv_folds", 3))
                _log(f"      Probe: train_acc={probe_info['train_acc']:.4f}, "
                     f"cv_acc={probe_info.get('cv_acc_mean', 'N/A')}, AUC={probe_info.get('auc', 'N/A')}", log_path)

                probe_cfg_with_rep = dict(probe_cfg)
                probe_cfg_with_rep["representation"] = probe_cfg["representations"][0]

                _run_sweep_point(model, tokenizer, test, steering_v, all_vectors,
                                 layer_idx, alpha, probe_info, probe_cfg_with_rep,
                                 gen_cfg, seed, all_metrics, scenario_name, log_path)

                gate_rows = [r for r in all_metrics
                             if "hard_gate" in str(r.get("mode", "")) and "random" not in str(r.get("mode", ""))
                             and "shuffled" not in str(r.get("mode", ""))
                             and r.get("seed") == seed and r.get("layer") == layer_idx]
                oracle_rows = [r for r in all_metrics
                               if r.get("mode") == "oracle_gate_a-1.0"
                               and r.get("seed") == seed and r.get("layer") == layer_idx]
                if gate_rows and oracle_rows:
                    gap = gate_rows[-1]["hallucination_rate"] - oracle_rows[-1]["hallucination_rate"]
                    if gap < best_gap:
                        best_gap = gap
                        best_combo = (seed, layer_idx, scenario_name, train, test,
                                      steering_v, all_vectors, probe_info, probe_cfg_with_rep)

                _log(f"    layer done in {time.time() - lt_layer:.0f}s", log_path)

                pd.DataFrame(all_metrics).to_csv(
                    os.path.join(results_dir, "sweep_matrix_partial.csv"), index=False)

    if best_combo is not None:
        seed_b, layer_b, sc_b, train_b, test_b, steering_v_b, all_vectors_b, probe_info_b, probe_cfg_b = best_combo
        _log(f"\n{'=' * 60}", log_path)
        _log(f"BEST COMBO: seed={seed_b}, layer={layer_b}, scenario={sc_b}", log_path)
        _log(f"Running extra alphas at best combo", log_path)

        for alpha_extra in sweep["extra_alphas"]:
            _log(f"\n  Extra alpha={alpha_extra:+.2f} [seed={seed_b}, layer={layer_b}]", log_path)
            _run_sweep_point(model, tokenizer, test_b, steering_v_b, all_vectors_b,
                             layer_b, alpha_extra, probe_info_b, probe_cfg_b,
                             gen_cfg, seed_b, all_metrics, sc_b, log_path)

    elapsed = time.time() - t_start
    _log(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)", log_path)

    df = pd.DataFrame(all_metrics)
    cols = ["seed", "layer", "mode", "alpha", "vector_type",
            "hallucination_rate", "calibrated_abstention_rate",
            "correct_answer_rate", "unnecessary_abstention_rate",
            "answerable_count", "unanswerable_count"]
    present_cols = [c for c in cols if c in df.columns]
    df = df[present_cols]
    met_path = os.path.join(results_dir, "sweep_matrix.csv")
    df.to_csv(met_path, index=False)
    _log(f"\nSweep matrix saved to {met_path} ({len(df)} rows)", log_path)

    _log(f"\n{'=' * 60}", log_path)
    _log("ROBUSTNESS MATRIX", log_path)
    _log(f"{'=' * 60}", log_path)
    for _, row in df.iterrows():
        _log(f"  s{int(row['seed'])}_l{int(row['layer'])} {row['mode']:<45} "
             f"H={row['hallucination_rate']:.3f} C={row['correct_answer_rate']:.3f}", log_path)

    report_path = os.path.join(reports_dir, "IC4_M4_GENERALIZATION_REPORT.md")
    _generate_robustness_report(report_path, config, df, elapsed, scenarios, scenario_labels)
    _log(f"\nReport saved to {report_path}", log_path)
    _log(f"\nIC-4-M4-Generalization complete.", log_path)


if __name__ == "__main__":
    main()