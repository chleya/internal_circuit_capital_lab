"""
IC-4-M4-Completion: Cross-seed + cross-layer gate validation.
Completes the M4 sweep: seeds [0,1,2] and layers [11,12,13].

Reuses M3-v6/M4 infrastructure directly for full compatibility.

Usage:
    python -u src/run_m4_completion.py                      # all stages
    python -u src/run_m4_completion.py --skip-gen --seed 1  # only validation for seed 1
    python -u src/run_m4_completion.py --report-only        # only regenerate report
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.run_m3_v6 import (
    _find_transformer_layer,
    _collect_prefill_features,
    _train_probe,
    _generate_single_pass_hard_gate,
    _generate_oracle_gated,
)
from src.model_loader import load_model_and_tokenizer, get_model_layer_count
from src.activation_collector import load_activations, collect_pair_activations
from src.steering import get_all_vectors

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_m4_completion")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports_m4_completion")
RESULTS_M3 = os.path.join(PROJECT_ROOT, "results_m3")
DATA_M3 = os.path.join(PROJECT_ROOT, "data_m3")
LOG_PATH = os.path.join(RESULTS_DIR, "run_log.txt")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(DATA_M3, exist_ok=True)
os.makedirs(RESULTS_M3, exist_ok=True)


def _log(msg):
    line = f"[M4-C] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


# ---------------------------------------------------------------------------
# Stage 1: Data Generation
# ---------------------------------------------------------------------------

def _generate_m3_data(seed):
    train_path = os.path.join(DATA_M3, f"train_s{seed}.jsonl")
    test_path = os.path.join(DATA_M3, f"test_s{seed}.jsonl")
    if os.path.exists(train_path) and os.path.exists(test_path):
        _log(f"Stage1 - Data s{seed}: already exists, skip.")
        return

    from src.data_builder import build_dataset, save_jsonl
    _log(f"Stage1 - Generating M3 data for seed={seed}...")
    random.seed(seed)
    np.random.seed(seed)
    train, test = build_dataset({"train_size": 30, "test_size": 60})
    save_jsonl(train, train_path)
    save_jsonl(test, test_path)
    na_t = sum(1 for s in train if s.get("answerability") == "answerable")
    na_s = sum(1 for s in test if s.get("answerability") == "answerable")
    _log(f"Stage1 - Data s{seed}: {len(train)} train ({na_t}A+{len(train)-na_t}U), "
         f"{len(test)} test ({na_s}A+{len(test)-na_s}U)")


# ---------------------------------------------------------------------------
# Stage 2: Activation Generation
# ---------------------------------------------------------------------------

def _generate_activations(seed, layer_idx, model, tokenizer):
    act_path = os.path.join(RESULTS_M3, f"activations_s{seed}_l{layer_idx}.npz")
    if os.path.exists(act_path):
        _log(f"Stage2 - Acts s{seed}_l{layer_idx}: already exists, skip.")
        return

    from src.data_builder import load_jsonl
    train_path = os.path.join(DATA_M3, f"train_s{seed}.jsonl")
    train = load_jsonl(train_path)

    _log(f"Stage2 - Generating activations s{seed}_l{layer_idx} ({len(train)} train)...")
    t0 = time.time()
    acts = collect_pair_activations(model, tokenizer, train, str(layer_idx),
                                    token_position="last")
    np.savez_compressed(act_path, **acts)
    _log(f"Stage2 - s{seed}_l{layer_idx}: done in {time.time()-t0:.0f}s "
         f"(pos={acts['positive'].shape}, dim={acts['positive'].shape[1]})")


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def _load_cached_m3_data(seed):
    from src.data_builder import load_jsonl
    train_path = os.path.join(DATA_M3, f"train_s{seed}.jsonl")
    test_path = os.path.join(DATA_M3, f"test_s{seed}.jsonl")
    train = load_jsonl(train_path)
    test = load_jsonl(test_path)
    na_t = sum(1 for s in train if s.get("answerability") == "answerable")
    na_s = sum(1 for s in test if s.get("answerability") == "answerable")
    _log(f"  Loaded s{seed}: train {na_t}A+{len(train)-na_t}U, test {na_s}A+{len(test)-na_s}U")
    return train, test


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Stage 3/4: Validation
# ---------------------------------------------------------------------------

GEN_CFG = {"max_new_tokens": 48, "temperature": 0.0, "do_sample": False}
PROBE_CFG = {"representations": ["last_prompt_token"], "cv_folds": 3}
PRIMARY_ALPHA = -1.0


def _run_one_setting(model, tokenizer, seed, layer_idx, all_metrics):
    _log(f"\n{'='*50}")
    _log(f"VALIDATION: seed={seed} layer={layer_idx}")
    _log(f"{'='*50}")

    random.seed(seed)
    np.random.seed(seed)

    train, test = _load_cached_m3_data(seed)

    from src.evaluate import generate_answers

    # --- Base ---
    _log(f"  base...")
    base_res = generate_answers(model, tokenizer, test, mode="base",
                                max_new_tokens=GEN_CFG["max_new_tokens"],
                                temperature=GEN_CFG["temperature"],
                                do_sample=GEN_CFG["do_sample"])
    m = _evaluate_and_add(base_res, seed, -1, "base", 0.0, "none", all_metrics)
    _log(f"    base:  H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f}")

    # Load activations & vectors
    act_path = os.path.join(RESULTS_M3, f"activations_s{seed}_l{layer_idx}.npz")
    if not os.path.exists(act_path):
        _log(f"    SKIP: no activations at {act_path}")
        return
    acts = load_activations(act_path)
    hidden_dim = acts["positive"].shape[1]
    all_vectors = get_all_vectors(acts["positive"], acts["negative"], hidden_dim)
    steering_v = all_vectors["steering"]
    _log(f"    Loaded {acts['positive'].shape[0]} activation pairs, dim={hidden_dim}")

    # --- Oracle Gate ---
    _log(f"  oracle_gate...")
    og_res = _generate_oracle_gated(model, tokenizer, test, steering_v, layer_idx,
                                     PRIMARY_ALPHA, "oracle_gate_a-1.0", GEN_CFG)
    m = _evaluate_and_add(og_res, seed, layer_idx, "oracle_gate_a-1.0", PRIMARY_ALPHA,
                          "steering", all_metrics)
    _log(f"    oracle: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f}")

    # --- Train Probe ---
    _log(f"  training probe [last_prompt_token]...")
    X_train, y_train = _collect_prefill_features(
        model, tokenizer, train, layer_idx, "last_prompt_token")
    probe_info = _train_probe(X_train, y_train, cv_folds=PROBE_CFG.get("cv_folds", 3))
    probe_cfg_wrap = {"representation": "last_prompt_token", "threshold": 0.5,
                      "steepness": 10.0, "cv_folds": 3}
    _log(f"    probe: train_acc={probe_info['train_acc']:.4f} "
         f"cv_acc={probe_info.get('cv_acc_mean','?'):.4f} auc={probe_info.get('auc','?'):.3f}")

    # --- Single-pass Hard Gate (real) ---
    _log(f"  single_pass_hard_gate (real)...")
    real_res = _generate_single_pass_hard_gate(
        model, tokenizer, test, steering_v, layer_idx,
        PRIMARY_ALPHA, probe_info, probe_cfg_wrap, GEN_CFG, "steering")
    m = _evaluate_and_add(real_res, seed, layer_idx,
                          f"single_pass_hard_gate_a{PRIMARY_ALPHA:+.1f}",
                          PRIMARY_ALPHA, "steering", all_metrics)
    _log(f"    hard:   H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f}")

    # --- Random Gate ---
    _log(f"  single_pass_hard_gate (random)...")
    rnd_res = _generate_single_pass_hard_gate(
        model, tokenizer, test, all_vectors["random"], layer_idx,
        PRIMARY_ALPHA, probe_info, probe_cfg_wrap, GEN_CFG, "random")
    m = _evaluate_and_add(rnd_res, seed, layer_idx,
                          f"random_single_pass_hard_gate_a{PRIMARY_ALPHA:+.1f}",
                          PRIMARY_ALPHA, "random", all_metrics)
    _log(f"    random: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f}")

    # --- Shuffled Gate ---
    _log(f"  single_pass_hard_gate (shuffled)...")
    shf_res = _generate_single_pass_hard_gate(
        model, tokenizer, test, all_vectors["shuffled"], layer_idx,
        PRIMARY_ALPHA, probe_info, probe_cfg_wrap, GEN_CFG, "shuffled")
    m = _evaluate_and_add(shf_res, seed, layer_idx,
                          f"shuffled_single_pass_hard_gate_a{PRIMARY_ALPHA:+.1f}",
                          PRIMARY_ALPHA, "shuffled", all_metrics)
    _log(f"    shufld: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f}")


# ---------------------------------------------------------------------------
# Stage 5: Report
# ---------------------------------------------------------------------------

def _generate_report(all_metrics):
    if len(all_metrics) == 0:
        _log("No metrics to report, skipping report generation.")
        return "IC4_M4_NO_DATA"
    df = pd.DataFrame(all_metrics)

    hard_modes = [m for m in df["mode"].unique()
                  if "single_pass_hard_gate" in str(m) and "random" not in str(m) and "shuffled" not in str(m)]
    random_modes = [m for m in df["mode"].unique() if "random_single_pass" in str(m)]
    shuffled_modes = [m for m in df["mode"].unique() if "shuffled_single_pass" in str(m)]

    lines = []
    lines.append("# IC-4-M4-Completion: Cross-Seed & Cross-Layer Validation Report")
    lines.append("")
    lines.append("> Completes the M4 generalization sweep beyond seed=0/layer=12.")
    lines.append(f"> Total trials: {len(df)}")
    lines.append("")

    # ---- Cross-Seed Section ----
    lines.append("## 1. Cross-Seed Validation (layer=12, standard)")
    lines.append("")
    lines.append("| Seed | Mode | H | C | UA | Oracle H | Oracle Gap | beats_random |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for seed in [0, 1, 2]:
        for mode_group in [["base"], ["oracle_gate_a-1.0"]] + \
                          [[m] for m in hard_modes] + \
                          [[m] for m in random_modes] + \
                          [[m] for m in shuffled_modes]:
            mode = mode_group[0]
            rows = df[(df["seed"] == seed) & (df["layer"] == 12) & (df["mode"] == mode)]
            if len(rows) == 0:
                continue
            r = rows.iloc[0]
            oracle_rows = df[(df["seed"] == seed) & (df["layer"] == 12) & (df["mode"] == "oracle_gate_a-1.0")]
            oracle_h = float(oracle_rows.iloc[0]["hallucination_rate"]) if len(oracle_rows) > 0 else float('nan')
            gap = float(r["hallucination_rate"]) - oracle_h if not np.isnan(oracle_h) else float('nan')

            rnd_rows = df[(df["seed"] == seed) & (df["layer"] == 12)
                          & (df["mode"].str.contains("random_single_pass"))]
            rnd_h = float(rnd_rows.iloc[0]["hallucination_rate"]) if len(rnd_rows) > 0 else float('nan')
            beats = rnd_h - float(r["hallucination_rate"]) if not np.isnan(rnd_h) else float('nan')
            ua = float(r.get("unnecessary_abstention_rate", 0.0))

            short_mode = mode.replace("_single_pass_hard_gate", "_hard").replace("_a-1.0", "").replace("oracle_gate", "oracle").replace("random_", "rnd_").replace("shuffled_", "shf_")
            lines.append(f"| {seed} | {short_mode} | {float(r['hallucination_rate']):.3f} | {float(r['correct_answer_rate']):.3f} | {ua:.3f} | {oracle_h:.3f} | {gap:+.3f} | {beats:+.3f} |")

    lines.append("")

    # ---- Cross-Layer Section ----
    lines.append("## 2. Cross-Layer Validation (seed=0, standard)")
    lines.append("")
    lines.append("| Layer | Mode | H | C | UA | Oracle H | Oracle Gap | beats_random |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")

    for layer in [11, 12, 13]:
        for mode_group in [["base"], ["oracle_gate_a-1.0"]] + \
                          [[m] for m in hard_modes] + \
                          [[m] for m in random_modes] + \
                          [[m] for m in shuffled_modes]:
            mode = mode_group[0]
            rows = df[(df["seed"] == 0) & (df["layer"] == layer) & (df["mode"] == mode)]
            if len(rows) == 0:
                continue
            r = rows.iloc[0]
            oracle_rows = df[(df["seed"] == 0) & (df["layer"] == layer) & (df["mode"] == "oracle_gate_a-1.0")]
            oracle_h = float(oracle_rows.iloc[0]["hallucination_rate"]) if len(oracle_rows) > 0 else float('nan')
            gap = float(r["hallucination_rate"]) - oracle_h if not np.isnan(oracle_h) else float('nan')

            rnd_rows = df[(df["seed"] == 0) & (df["layer"] == layer)
                          & (df["mode"].str.contains("random_single_pass"))]
            rnd_h = float(rnd_rows.iloc[0]["hallucination_rate"]) if len(rnd_rows) > 0 else float('nan')
            beats = rnd_h - float(r["hallucination_rate"]) if not np.isnan(rnd_h) else float('nan')
            ua = float(r.get("unnecessary_abstention_rate", 0.0))

            short_mode = mode.replace("_single_pass_hard_gate", "_hard").replace("_a-1.0", "").replace("oracle_gate", "oracle").replace("random_", "rnd_").replace("shuffled_", "shf_")
            lines.append(f"| {layer} | {short_mode} | {float(r['hallucination_rate']):.3f} | {float(r['correct_answer_rate']):.3f} | {ua:.3f} | {oracle_h:.3f} | {gap:+.3f} | {beats:+.3f} |")

    lines.append("")

    # ---- Key Questions ----
    lines.append("## 3. Key Questions")
    lines.append("")

    lines.append("### Q1: Do seeds 1/2 reproduce layer 12 success?")
    lines.append("")
    for seed in [0, 1, 2]:
        hard_rows = df[(df["seed"] == seed) & (df["layer"] == 12)
                       & (df["mode"].isin(hard_modes))]
        oracle_rows = df[(df["seed"] == seed) & (df["layer"] == 12)
                         & (df["mode"] == "oracle_gate_a-1.0")]
        if len(hard_rows) > 0 and len(oracle_rows) > 0:
            h_h = float(hard_rows.iloc[0]["hallucination_rate"])
            o_h = float(oracle_rows.iloc[0]["hallucination_rate"])
            gap = h_h - o_h
            status = "PASS (gap <= 0.05)" if abs(gap) <= 0.05 else "WARNING (gap > 0.05)"
            lines.append(f"- **seed={seed}**: hard H={h_h:.3f}, oracle H={o_h:.3f}, gap={gap:+.3f} — **{status}**")
    lines.append("")

    lines.append("### Q2: Layer comparison (seed=0)")
    lines.append("")
    for layer in [11, 12, 13]:
        hard_rows = df[(df["seed"] == 0) & (df["layer"] == layer)
                       & (df["mode"].isin(hard_modes))]
        oracle_rows = df[(df["seed"] == 0) & (df["layer"] == layer)
                         & (df["mode"] == "oracle_gate_a-1.0")]
        if len(hard_rows) > 0 and len(oracle_rows) > 0:
            h_h = float(hard_rows.iloc[0]["hallucination_rate"])
            o_h = float(oracle_rows.iloc[0]["hallucination_rate"])
            gap = h_h - o_h
            status = "PASS" if abs(gap) <= 0.05 else "WARNING"
            lines.append(f"- **layer={layer}**: hard H={h_h:.3f}, oracle H={o_h:.3f}, gap={gap:+.3f} — **{status}**")
    lines.append("")

    lines.append("### Q3: Real gate vs random/shuffled separation")
    lines.append("")
    for seed in [0, 1, 2]:
        for layer in [11, 12, 13]:
            hard_rows = df[(df["seed"] == seed) & (df["layer"] == layer)
                           & (df["mode"].isin(hard_modes))]
            rnd_rows = df[(df["seed"] == seed) & (df["layer"] == layer)
                          & (df["mode"].isin(random_modes))]
            shf_rows = df[(df["seed"] == seed) & (df["layer"] == layer)
                          & (df["mode"].isin(shuffled_modes))]
            if len(hard_rows) > 0 and len(rnd_rows) > 0:
                h_h = float(hard_rows.iloc[0]["hallucination_rate"])
                r_h = float(rnd_rows.iloc[0]["hallucination_rate"])
                s_h = float(shf_rows.iloc[0]["hallucination_rate"]) if len(shf_rows) > 0 else float('nan')
                sep = r_h - h_h
                lines.append(f"- s{seed}_l{layer}: hard H={h_h:.3f}, random H={r_h:.3f}, "
                             f"shuffled H={s_h:.3f}, separation={sep:+.3f}")
    lines.append("")

    # ---- Success Criteria ----
    lines.append("## 4. Success Criteria")
    lines.append("")

    all_hard = df[df["mode"].isin(hard_modes)]
    all_oracle = df[df["mode"] == "oracle_gate_a-1.0"]
    all_random = df[df["mode"].isin(random_modes)]

    max_gap = 0.0
    min_sep = 999.0
    for _, h_row in all_hard.iterrows():
        o_row = all_oracle[(all_oracle["seed"] == h_row["seed"]) & (all_oracle["layer"] == h_row["layer"])]
        if len(o_row) > 0:
            gap = float(h_row["hallucination_rate"]) - float(o_row.iloc[0]["hallucination_rate"])
            max_gap = max(max_gap, gap)
        r_row = all_random[(all_random["seed"] == h_row["seed"]) & (all_random["layer"] == h_row["layer"])]
        if len(r_row) > 0:
            sep = float(r_row.iloc[0]["hallucination_rate"]) - float(h_row["hallucination_rate"])
            min_sep = min(min_sep, sep)

    lines.append("| Criterion | Condition | Value | Result |")
    lines.append("|---|---|---|---|")
    gap_pass = max_gap <= 0.05
    lines.append(f"| Max oracle gap | ≤ 0.05 | {max_gap:+.3f} | {'PASS' if gap_pass else 'FAIL'} |")
    sep_pass = min_sep >= 0.08
    lines.append(f"| Min random separation | ≥ 0.08 | {min_sep:.3f} | {'PASS' if sep_pass else 'FAIL'} |")
    c_stable = all(abs(float(r["correct_answer_rate"]) - 0.600) <= 0.10 for _, r in all_hard.iterrows())
    lines.append(f"| C stability | ±0.10 from baseline | {'stable' if c_stable else 'WARN'} | {'PASS' if c_stable else 'FAIL'} |")
    lines.append("")

    # ---- Verdict ----
    lines.append("## 5. Verdict")
    lines.append("")

    if gap_pass and sep_pass and c_stable:
        verdict = "IC4_M4_COMPLETION_ROBUST"
        reason = ("Cross-seed and cross-layer validation passed. "
                  "Mechanism is robust across seeds [0,1,2] and layers [11,12,13]. "
                  "Robustness claim UPGRADED from scoped (seed=0/layer=12) to cross-seed/cross-layer.")
    elif gap_pass and sep_pass:
        verdict = "IC4_M4_COMPLETION_MOSTLY_ROBUST"
        reason = ("Gate mechanism passes oracle gap and separation checks across seeds/layers, "
                  "but C stability shows minor variance.")
    else:
        verdict = "IC4_M4_COMPLETION_PARTIAL"
        reason = (f"Gate mechanism shows some fragility: max oracle gap={max_gap:+.3f}, "
                  f"min separation={min_sep:.3f}.")

    lines.append(f"**Verdict: `{verdict}`**")
    lines.append("")
    lines.append(f"**Reasoning:** {reason}")
    lines.append("")

    lines.append("### Robustness Claim Status")
    lines.append("")
    lines.append("| Claim | Before | After |")
    lines.append("|---|---|---|")
    lines.append("| Seed coverage | seed=0 only | seeds [0,1,2] |")
    lines.append("| Layer coverage | layer=12 only | layers [11,12,13] |")
    lines.append("| Max oracle gap | 0.000 (s0_l12) | " + f"{max_gap:+.3f} (worst case)" + " |")
    lines.append("| Min separation | - | " + f"{min_sep:.3f}" + " |")
    lines.append("")
    lines.append("---")
    lines.append("*IC-4-M4-Completion: Cross-Seed & Cross-Layer Validation*")

    report_path = os.path.join(REPORTS_DIR, "IC4_M4_COMPLETION_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    _log(f"Report saved to {report_path}")
    return verdict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="M4 Completion")
    parser.add_argument("--skip-gen", action="store_true",
                        help="Skip data & activation generation.")
    parser.add_argument("--skip-validate", action="store_true",
                        help="Skip validation, only generate missing files.")
    parser.add_argument("--report-only", action="store_true",
                        help="Only regenerate report from existing CSV.")
    parser.add_argument("--seed", type=int, nargs="*", default=None,
                        help="Specific seeds to validate (default: 0,1,2).")
    parser.add_argument("--layer", type=int, nargs="*", default=None,
                        help="Specific layers to validate (default: 11,12,13).")
    args = parser.parse_args()

    seeds = args.seed if args.seed else [0, 1, 2]
    layers = args.layer if args.layer else [11, 12, 13]

    _log("=" * 60)
    _log("M4 Completion: Cross-Seed & Cross-Layer Validation")
    _log(f"Seeds: {seeds}, Layers: {layers}")
    _log("=" * 60)

    all_metrics = []

    if args.report_only:
        met_path = os.path.join(RESULTS_DIR, "m4_completion_metrics.csv")
        if os.path.exists(met_path):
            df = pd.read_csv(met_path)
            all_metrics = df.to_dict("records")
            _log(f"Loaded {len(all_metrics)} rows from {met_path}")
        verdict = _generate_report(all_metrics)
        _log(f"\nVerdict: {verdict}")
        return

    # ---- Stage 1: Data Generation ----
    if not args.skip_gen:
        _log("\n" + "=" * 50)
        _log("STAGE 1: Generate missing M3 data")
        _log("=" * 50)
        for seed in seeds:
            _generate_m3_data(seed)
        _log("Stage 1 DONE.\n")

    # ---- Stage 2: Activation Generation ----
    if not args.skip_gen:
        _log("\n" + "=" * 50)
        _log("STAGE 2: Generate missing activations")
        _log("=" * 50)
        model, tokenizer = load_model_and_tokenizer(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            device="cpu", torch_dtype="float32",
        )
        _log(f"Model loaded, hidden_dim={model.config.hidden_size}")

        needed = set()
        for seed in seeds:
            for layer in layers:
                needed.add((seed, layer))

        for seed, layer in sorted(needed):
            _generate_activations(seed, layer, model, tokenizer)
        _log("Stage 2 DONE.\n")

    # ---- Stage 3/4: Validation ----
    if not args.skip_validate:
        _log("\n" + "=" * 50)
        _log("STAGE 3/4: Cross-Seed + Cross-Layer Validation")
        _log("=" * 50)

        model, tokenizer = load_model_and_tokenizer(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            device="cpu", torch_dtype="float32",
        )
        _log(f"Model loaded, {get_model_layer_count(model)} layers total")

        t_start = time.time()

        # Cross-seed: each seed runs at layer=12 only if layer is in layers list
        # Cross-layer: seed=0 runs at each layer only if seed is in seeds list
        for seed in seeds:
            for layer in layers:
                _run_one_setting(model, tokenizer, seed, layer, all_metrics)

                # Save partial results
                pd.DataFrame(all_metrics).to_csv(
                    os.path.join(RESULTS_DIR, "m4_completion_metrics.csv"), index=False)

        elapsed = time.time() - t_start
        _log(f"\nValidation complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")

        # Save final metrics
        met_path = os.path.join(RESULTS_DIR, "m4_completion_metrics.csv")
        pd.DataFrame(all_metrics).to_csv(met_path, index=False)
        _log(f"Metrics saved to {met_path} ({len(all_metrics)} rows)")

        _log("\n" + "=" * 50)
        _log("RESULTS MATRIX")
        _log("=" * 50)
        for row in sorted(all_metrics, key=lambda r: (r.get("seed", 0), r.get("layer", 0), str(r.get("mode", "")))):
            _log(f"  s{row.get('seed','?')}_l{row.get('layer','?')} {str(row.get('mode','?')):<48} "
                 f"H={row.get('hallucination_rate',0):.3f} C={row.get('correct_answer_rate',0):.3f}")

    # ---- Stage 5: Report ----
    _log("\n" + "=" * 50)
    _log("STAGE 5: Generate Report")
    _log("=" * 50)

    met_path = os.path.join(RESULTS_DIR, "m4_completion_metrics.csv")
    if os.path.exists(met_path) and len(all_metrics) == 0:
        df = pd.read_csv(met_path)
        all_metrics = df.to_dict("records")

    verdict = _generate_report(all_metrics)
    _log(f"\nVerdict: {verdict}")

    _log("\n" + "=" * 60)
    _log("M4 Completion FINISHED.")
    _log("=" * 60)


if __name__ == "__main__":
    main()