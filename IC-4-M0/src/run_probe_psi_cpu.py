"""
IC-4: Position Sensitivity Sweep — CPU-Optimized Probe PSI.

Key optimizations for CPU:
  1. Batch feature extraction (all samples per position in one forward pass)
  2. torch.set_num_threads() tuning
  3. Reduced max_length (256 vs 512)
  4. Reuses train features extraction for probe (only 30 samples → negligible)
  5. No generation — probe scores only
  6. torch.inference_mode() for faster no-grad

Usage:
    python -m src.run_probe_psi_cpu --config configs/config_m3_v6.yaml
"""

import argparse
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.run_m2 import load_config

RESULTS_DIR = "results_position_sensitivity_cpu"


def _log(msg):
    print(msg, flush=True)


def _batched_extract_features(model, tokenizer, samples, layer_idx, representation, batch_size=16):
    """Extract probe features in batches — key CPU optimization."""
    device = next(model.parameters()).device
    X_list, y_list = [], []

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        prompts = []
        labels = []
        for s in batch:
            context = s.get("context", "")
            question = s.get("question", "")
            label = s.get("answerability", "?")
            prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
            prompts.append(prompt)
            labels.append(1 if label == "answerable" else 0)

        inputs = tokenizer(prompts, return_tensors="pt", truncation=True,
                          max_length=256, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        attention_mask = inputs["attention_mask"]

        with torch.inference_mode():
            outputs = model(**inputs)
        hs = outputs.hidden_states[layer_idx + 1]

        B, S, D = hs.shape
        for b in range(B):
            valid_len = int(attention_mask[b].sum().item())
            last_idx = valid_len - 1 if valid_len > 0 else S - 1

            if representation == "last_prompt_token":
                pooled = hs[b, last_idx, :].cpu().float().numpy()
            elif representation == "mean_pooled":
                pooled = hs[b, :valid_len, :].mean(dim=0).cpu().float().numpy() if valid_len > 0 else hs[b, 0, :].cpu().float().numpy()
            else:
                pooled = hs[b, last_idx, :].cpu().float().numpy()

            X_list.append(pooled)
            y_list.append(labels[b])

    return np.stack(X_list, axis=0), np.array(y_list, dtype=np.int32)


def _run_probe_psi_cpu(config):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    seed = config["steering"]["seeds"][0]
    layers = config["steering"]["layers"]
    probe_cfg = config["probe"]
    representations = probe_cfg["representations"]
    batch_size = int(os.environ.get("BATCH_SIZE", "16"))

    n_threads = int(os.environ.get("OMP_NUM_THREADS", str(min(8, os.cpu_count() or 4))))
    torch.set_num_threads(n_threads)
    _log(f"CPU threads: {n_threads}, batch_size: {batch_size}")

    from src.model_loader import load_model_and_tokenizer

    _log(f"Loading model ({config['model']['name']}) on CPU...")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(
        model_name=config["model"]["name"],
        device="cpu",
        torch_dtype="float32",
    )
    _log(f"  Model loaded in {time.time() - t0:.1f}s")

    data_dir = os.path.join("data_position_sensitivity", f"s{seed}")
    train_all_path = os.path.join(data_dir, f"train_all_s{seed}.jsonl")
    from src.data_builder import load_jsonl
    if os.path.exists(train_all_path):
        train = load_jsonl(train_all_path)
        _log(f"  Train (mixed positions): {len(train)} samples")
    else:
        train_path = config["data"].get("train_path", "data_m3/train.jsonl")
        train_final = train_path.replace(".jsonl", f"_s{seed}.jsonl")
        train = load_jsonl(train_final)
        _log(f"  Train (original, no position variants found): {len(train)} samples")

    data_dir = os.path.join("data_position_sensitivity", f"s{seed}")
    variants = {}
    for pos_name in ["early", "mid", "late"]:
        vpath = os.path.join(data_dir, f"test_{pos_name}_s{seed}.jsonl")
        variants[pos_name] = load_jsonl(vpath)
        _log(f"  {pos_name}: {len(variants[pos_name])} samples")

    from src.run_m3_v6 import _train_probe

    all_psi = []
    all_probe_scores = []
    t_start = time.time()

    for layer_idx in layers:
        _log(f"\n{'='*50}")
        _log(f"LAYER {layer_idx}")
        _log(f"{'='*50}")

        for rep in representations:
            _log(f"\n  Probe [{rep}]")

            t_feat = time.time()
            X_train, y_train = _batched_extract_features(
                model, tokenizer, train, layer_idx, rep, batch_size=batch_size)
            _log(f"    Train features: {X_train.shape} in {time.time() - t_feat:.1f}s")

            probe_info = _train_probe(X_train, y_train, cv_folds=probe_cfg.get("cv_folds", 3))
            _log(f"    train_acc={probe_info['train_acc']:.4f}, AUC={probe_info.get('auc', 'N/A')}")

            scaler = probe_info["scaler"]
            clf = probe_info["classifier"]
            threshold = probe_cfg.get("threshold", 0.5)

            pos_results = {}
            for pos_name in ["early", "mid", "late"]:
                t_pos = time.time()
                X_test, y_test = _batched_extract_features(
                    model, tokenizer, variants[pos_name], layer_idx, rep, batch_size=batch_size)
                X_scaled = scaler.transform(X_test)
                probas = clf.predict_proba(X_scaled)[:, 1]
                preds = clf.predict(X_scaled)

                gate_rate = float(np.mean(probas < threshold))
                gate_correct_answerable = float(np.mean((probas >= threshold) & (y_test == 1)) / max(np.sum(y_test == 1), 1))
                gate_correct_unanswerable = float(np.mean((probas < threshold) & (y_test == 0)) / max(np.sum(y_test == 0), 1))

                pos_results[pos_name] = {
                    "mean_score": float(np.mean(probas)),
                    "std_score": float(np.std(probas)),
                    "gate_rate": gate_rate,
                    "gate_correct_answerable": gate_correct_answerable,
                    "gate_correct_unanswerable": gate_correct_unanswerable,
                    "train_acc": float(probe_info["train_acc"]),
                }

                _log(f"    {pos_name}: mean_score={pos_results[pos_name]['mean_score']:.4f}, "
                     f"gate_rate={gate_rate:.3f}, "
                     f"correct_A={gate_correct_answerable:.3f}, correct_U={gate_correct_unanswerable:.3f} "
                     f"({time.time() - t_pos:.1f}s)")

                for sid in range(len(variants[pos_name])):
                    all_probe_scores.append({
                        "seed": seed,
                        "layer": layer_idx,
                        "representation": rep,
                        "position": pos_name,
                        "sample_id": sid,
                        "answerability": variants[pos_name][sid].get("answerability"),
                        "probe_score": float(probas[sid]),
                        "prediction": int(preds[sid]),
                        "label": int(y_test[sid]),
                    })

            for comp in [("early", "mid"), ("early", "late")]:
                early_scores = np.array([r["probe_score"] for r in all_probe_scores
                                         if r["position"] == comp[0] and r["seed"] == seed
                                         and r["layer"] == layer_idx and r["representation"] == rep])
                pos_scores = np.array([r["probe_score"] for r in all_probe_scores
                                       if r["position"] == comp[1] and r["seed"] == seed
                                       and r["layer"] == layer_idx and r["representation"] == rep])

                mean_delta = float(np.mean(np.abs(early_scores - pos_scores)))
                max_delta = float(np.max(np.abs(early_scores - pos_scores)))

                all_psi.append({
                    "seed": seed,
                    "layer": layer_idx,
                    "representation": rep,
                    "comparison": f"{comp[0]}_vs_{comp[1]}",
                    "mean_abs_score_delta": round(mean_delta, 6),
                    "max_abs_score_delta": round(max_delta, 6),
                    f"{comp[0]}_mean_score": pos_results[comp[0]]["mean_score"],
                    f"{comp[1]}_mean_score": pos_results[comp[1]]["mean_score"],
                    f"{comp[0]}_gate_rate": pos_results[comp[0]]["gate_rate"],
                    f"{comp[1]}_gate_rate": pos_results[comp[1]]["gate_rate"],
                    f"{comp[0]}_gate_correct_A": pos_results[comp[0]]["gate_correct_answerable"],
                    f"{comp[1]}_gate_correct_A": pos_results[comp[1]]["gate_correct_answerable"],
                    f"{comp[0]}_gate_correct_U": pos_results[comp[0]]["gate_correct_unanswerable"],
                    f"{comp[1]}_gate_correct_U": pos_results[comp[1]]["gate_correct_unanswerable"],
                })

    elapsed = time.time() - t_start
    _log(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    psi_df = pd.DataFrame(all_psi)
    psi_path = os.path.join(RESULTS_DIR, "position_sensitivity_index.csv")
    psi_df.to_csv(psi_path, index=False)
    _log(f"PSI results → {psi_path}")

    scores_df = pd.DataFrame(all_probe_scores)
    scores_path = os.path.join(RESULTS_DIR, "per_sample_probe_scores.csv")
    scores_df.to_csv(scores_path, index=False)
    _log(f"Per-sample scores → {scores_path}")

    _log(f"\n{'='*50}")
    _log("POSITION SENSITIVITY SUMMARY")
    _log(f"{'='*50}")
    for _, row in psi_df.iterrows():
        _log(f"  {row['comparison']}: dScore_mean={row['mean_abs_score_delta']:.4f}, "
             f"dScore_max={row['max_abs_score_delta']:.4f}, "
             f"early_gate={row['early_gate_rate']:.3f}, {row['comparison'].split('_')[-1]}_gate={row[row['comparison'].split('_')[-1] + '_gate_rate']:.3f}")

    mean_psi = float(psi_df["mean_abs_score_delta"].mean())
    max_psi = float(psi_df["max_abs_score_delta"].max())

    _log(f"\n  PSI_mean = {mean_psi:.4f}, PSI_max = {max_psi:.4f}")
    if mean_psi > 0.3:
        _log("  D9 TRIGGERED: PSI > 0.3 — position is a first-order confound.")
    elif mean_psi > 0.1:
        _log("  WARNING: moderate position sensitivity.")
    else:
        _log("  OK: low position sensitivity. RoPE ceiling not yet binding.")

    gate_deltas = []
    for pos_name in ["mid", "late"]:
        col = f"{pos_name}_gate_rate"
        early_col = "early_gate_rate"
        for _, row in psi_df.iterrows():
            gate_deltas.append(abs(row[col] - row[early_col]))
    mean_gate_delta = float(np.mean(gate_deltas))
    _log(f"  Mean gate rate shift across positions: {mean_gate_delta:.4f}")

    report_path = os.path.join(RESULTS_DIR, "PROBE_PSI_REPORT.md")
    _generate_report(report_path, psi_df, scores_df, config, elapsed, mean_psi, mean_gate_delta)
    _log(f"\nReport → {report_path}")

    stats = {
        "psi_mean": mean_psi,
        "psi_max": max_psi,
        "mean_gate_delta": mean_gate_delta,
        "d9_triggered": mean_psi > 0.3,
        "config": {
            "model": config["model"]["name"],
            "layer": layers[0],
            "seed": seed,
            "representation": representations[0],
        },
        "timing": {"total_s": elapsed, "total_min": elapsed / 60},
    }
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(stats, f, indent=2)


def _generate_report(report_path, psi_df, scores_df, config, elapsed, mean_psi, mean_gate_delta):
    seed = config["steering"]["seeds"][0]
    layer = config["steering"]["layers"][0]
    rep = config["probe"]["representations"][0]

    lines = []
    lines.append("# IC-4: Position Sensitivity Sweep — Probe PSI Report")
    lines.append("")
    lines.append(f"**Experiment:** Relational Memory Hypothesis — Experiment #1 (Probe PSI, CPU)  ")
    lines.append(f"**Date:** 2026-05-20  ")
    lines.append(f"**Model:** {config['model']['name']} on CPU  ")
    lines.append(f"**Elapsed:** {elapsed:.0f}s ({elapsed/60:.1f} min)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Design")
    lines.append("")
    lines.append("Same evidence content placed at 3 positions (early/mid/late) within a 5-sentence prompt. ")
    lines.append("Probe scores extracted via single forward pass (no generation). ")
    lines.append("Measures how much the probe score changes purely due to evidence position.")
    lines.append("")

    lines.append("## 2. Probe Quality")
    lines.append("")
    probe_rows = scores_df.groupby(["position"]).agg(
        n_samples=("sample_id", "count"),
        mean_score=("probe_score", "mean"),
        std_score=("probe_score", "std"),
    ).reset_index()
    lines.append("| Position | N | Mean Score | Std Score |")
    lines.append("|---|---|---|---|")
    for _, row in probe_rows.iterrows():
        lines.append(f"| {row['position']} | {int(row['n_samples'])} | {row['mean_score']:.4f} | {row['std_score']:.4f} |")
    lines.append("")

    lines.append("## 3. Position Sensitivity Index (PSI)")
    lines.append("")
    lines.append("| Comparison | dScore Mean | dScore Max | Early Gate | Late/Mid Gate | Gate dA | Gate dU |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, row in psi_df.iterrows():
        cmp_name = row["comparison"].split("_vs_")[1]
        early_gcA = row.get("early_gate_correct_A", 0)
        cmp_gcA = row.get(f"{cmp_name}_gate_correct_A", 0)
        early_gcU = row.get("early_gate_correct_U", 0)
        cmp_gcU = row.get(f"{cmp_name}_gate_correct_U", 0)
        lines.append(f"| {row['comparison']} | {row['mean_abs_score_delta']:.4f} | {row['max_abs_score_delta']:.4f} | {row['early_gate_rate']:.3f} | {row[cmp_name + '_gate_rate']:.3f} | {early_gcA:.3f}→{cmp_gcA:.3f} | {early_gcU:.3f}→{cmp_gcU:.3f} |")
    lines.append("")

    lines.append(f"**PSI (mean absolute score delta): {mean_psi:.4f}**")
    lines.append(f"**Mean gate rate shift: {mean_gate_delta:.4f}**")
    lines.append("")

    lines.append("## 4. Death Condition D9")
    lines.append("")
    if mean_psi > 0.3:
        lines.append("**D9 TRIGGERED**: PSI > 0.3. Position is a first-order confound for routing analysis. ")
        lines.append("Trajectory results must separately model position distortion.")
    elif mean_psi > 0.1:
        lines.append("**WARNING**: Moderate position sensitivity (0.1 < PSI < 0.3). ")
        lines.append("Position has a detectable but not dominant effect on probe scores.")
    else:
        lines.append("**OK**: Low position sensitivity (PSI < 0.1). ")
        lines.append("RoPE position encoding ceiling is not yet binding at this context length.")
    lines.append("")

    lines.append("## 5. Answerable vs Unanswerable Breakdown")
    lines.append("")
    for pos_name in ["early", "mid", "late"]:
        pos_mask = scores_df["position"] == pos_name
        ans_mask = scores_df["answerability"] == "answerable"
        unans_mask = scores_df["answerability"] == "unanswerable"

        ans_scores = scores_df[pos_mask & ans_mask]["probe_score"]
        unans_scores = scores_df[pos_mask & unans_mask]["probe_score"]

        lines.append(f"**{pos_name}**: answerable mean={ans_scores.mean():.4f}±{ans_scores.std():.4f}, "
                     f"unanswerable mean={unans_scores.mean():.4f}±{unans_scores.std():.4f}, "
                     f"separation={ans_scores.mean() - unans_scores.mean():.4f}")
    lines.append("")

    lines.append("---")
    lines.append("*IC-4 Position Sensitivity Sweep — Probe PSI, CPU-optimized*")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="IC-4 Position Sensitivity Sweep — CPU Optimized")
    parser.add_argument("--config", type=str, default="configs/config_m3_v6.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    _run_probe_psi_cpu(config)


if __name__ == "__main__":
    main()