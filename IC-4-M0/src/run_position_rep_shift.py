"""
IC-4: Position Representation Shift — CPU-optimized.

Instead of training a probe (which overfits with 90 samples / 896 dims),
directly measures how much the hidden state representation changes when
the SAME content appears at different positions.

Metric: cosine distance between last_prompt_token vectors for same sample
at early vs mid/late position. High cosine distance = position-sensitive
representation.

Usage:
    python -m src.run_position_rep_shift --config configs/config_m3_v6.yaml
"""

import argparse
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cosine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.run_m2 import load_config

RESULTS_DIR = "results_position_sensitivity_cpu"


def _log(msg):
    print(msg, flush=True)


def _batched_extract_vectors(model, tokenizer, samples, layer_idx, batch_size=16):
    """Extract last_prompt_token hidden states in batches."""
    device = next(model.parameters()).device
    all_vecs = []

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        prompts = []
        for s in batch:
            context = s.get("context", "")
            question = s.get("question", "")
            prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
            prompts.append(prompt)

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
            vec = hs[b, last_idx, :].cpu().float().numpy()
            all_vecs.append(vec)

    return np.stack(all_vecs, axis=0)


def _run_position_rep_shift(config):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    seed = config["steering"]["seeds"][0]
    layers = config["steering"]["layers"]
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

    from src.data_builder import load_jsonl
    variants = {}
    for pos_name in ["early", "mid", "late"]:
        vpath = os.path.join(data_dir, f"test_{pos_name}_s{seed}.jsonl")
        variants[pos_name] = load_jsonl(vpath)
        _log(f"  {pos_name}: {len(variants[pos_name])} samples")

    all_results = []

    for layer_idx in layers:
        _log(f"\n{'='*50}")
        _log(f"LAYER {layer_idx} — Direct Representation Comparison")
        _log(f"{'='*50}")

        t_feat = time.time()
        vecs = {}
        for pos_name in ["early", "mid", "late"]:
            _log(f"  Extracting {pos_name} features...")
            vecs[pos_name] = _batched_extract_vectors(
                model, tokenizer, variants[pos_name], layer_idx, batch_size=batch_size)
        _log(f"  Features extracted in {time.time() - t_feat:.1f}s")

        n_samples = len(variants["early"])
        cos_dists_early_mid = []
        cos_dists_early_late = []
        cos_dists_mid_late = []

        for i in range(n_samples):
            v_early = vecs["early"][i]
            v_mid = vecs["mid"][i]
            v_late = vecs["late"][i]

            d_em = cosine(v_early, v_mid)
            d_el = cosine(v_early, v_late)
            d_ml = cosine(v_mid, v_late)

            cos_dists_early_mid.append(d_em)
            cos_dists_early_late.append(d_el)
            cos_dists_mid_late.append(d_ml)

            all_results.append({
                "seed": seed,
                "layer": layer_idx,
                "sample_id": i,
                "answerability": variants["early"][i].get("answerability"),
                "cos_early_mid": float(d_em),
                "cos_early_late": float(d_el),
                "cos_mid_late": float(d_ml),
            })

        _log(f"\n  Cosine distance (position shift):")
        _log(f"    early ↔ mid:   mean={np.mean(cos_dists_early_mid):.6f} ± {np.std(cos_dists_early_mid):.6f}")
        _log(f"    early ↔ late:  mean={np.mean(cos_dists_early_late):.6f} ± {np.std(cos_dists_early_late):.6f}")
        _log(f"    mid   ↔ late:  mean={np.mean(cos_dists_mid_late):.6f} ± {np.std(cos_dists_mid_late):.6f}")

        delta_el_em = np.mean(np.array(cos_dists_early_late) - np.array(cos_dists_early_mid))
        _log(f"    Δ(early→late - early→mid) = {delta_el_em:.6f}")

        all_vecs_stacked = np.concatenate([vecs["early"], vecs["mid"], vecs["late"]], axis=0)
        pos_labels = np.array([0] * n_samples + [1] * n_samples + [2] * n_samples)
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(all_vecs_stacked, pos_labels)
        knn_acc = knn.score(all_vecs_stacked, pos_labels)
        _log(f"    3-NN position classification accuracy: {knn_acc:.4f} (baseline=0.333)")

    df = pd.DataFrame(all_results)
    out_path = os.path.join(RESULTS_DIR, "representation_shift.csv")
    df.to_csv(out_path, index=False)
    _log(f"\nResults → {out_path}")

    summary = {
        "model": config["model"]["name"],
        "layer": layers[0],
        "seed": seed,
        "n_samples": n_samples,
        "cos_early_mid_mean": float(np.mean(cos_dists_early_mid)),
        "cos_early_mid_std": float(np.std(cos_dists_early_mid)),
        "cos_early_late_mean": float(np.mean(cos_dists_early_late)),
        "cos_early_late_std": float(np.std(cos_dists_early_late)),
        "cos_mid_late_mean": float(np.mean(cos_dists_mid_late)),
        "cos_mid_late_std": float(np.std(cos_dists_mid_late)),
        "knn_position_acc": float(knn_acc),
        "baseline_acc": 0.333,
        "position_sensitive": float(knn_acc) > 0.5,
    }
    summary_path = os.path.join(RESULTS_DIR, "rep_shift_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    _log(f"\n{'='*50}")
    _log("INTERPRETATION")
    _log(f"{'='*50}")
    cos_scale = float(np.mean(cos_dists_early_late))
    knn_val = float(knn_acc)
    _log(f"  Cosine distance (same content, shifted position): {cos_scale:.6f}")
    _log(f"  KNN position classifiability: {knn_val:.4f} (baseline=0.333)")

    if knn_val > 0.8:
        _log("  STRONG POSITION SENSITIVITY: hidden state strongly encodes position.")
        _log("  → Routing/gate decisions may depend on WHERE content is, not WHAT it is.")
    elif knn_val > 0.5:
        _log("  MODERATE POSITION SENSITIVITY: position is detectable in hidden state.")
    elif cos_scale > 0.01:
        _log("  WEAK BUT DETECTABLE: position leaves a small but non-zero trace.")
    else:
        _log("  MINIMAL: hidden state is nearly position-invariant for same content.")
        _log("  → RoPE ceiling not yet binding at this scale.")

    report_path = os.path.join(RESULTS_DIR, "REP_SHIFT_REPORT.md")
    _generate_report(report_path, summary, df)
    _log(f"\nReport → {report_path}")


def _generate_report(report_path, summary, df):
    lines = []
    lines.append("# IC-4: Position Representation Shift Report")
    lines.append("")
    lines.append(f"**Experiment:** Relational Memory Hypothesis — Experiment #1 (Rep Shift, CPU)  ")
    lines.append(f"**Date:** 2026-05-20  ")
    lines.append(f"**Model:** {summary['model']} on CPU  ")
    lines.append(f"**Layer:** {summary['layer']}, Seed: {summary['seed']}  ")
    lines.append(f"**N samples:** {summary['n_samples']}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Method")
    lines.append("")
    lines.append("Same evidence content placed at 3 positions via prefix shifting:  ")
    lines.append("- **early**: no prefix (content at position 0)  ")
    lines.append("- **mid**: 1x prefix (content shifted right by ~8 tokens)  ")
    lines.append("- **late**: 2x prefix (content shifted right by ~16 tokens)  ")
    lines.append("")
    lines.append("Extract `last_prompt_token` hidden state vector for each variant.  ")
    lines.append("Compute **cosine distance** between same sample at different positions.  ")
    lines.append("Also: 3-NN classifier to check if position can be decoded from the hidden state.")
    lines.append("")

    lines.append("## 2. Results")
    lines.append("")
    lines.append("| Comparison | Mean Cosine Distance | Std |")
    lines.append("|---|---|---|")
    lines.append(f"| early ↔ mid | {summary['cos_early_mid_mean']:.6f} | {summary['cos_early_mid_std']:.6f} |")
    lines.append(f"| early ↔ late | {summary['cos_early_late_mean']:.6f} | {summary['cos_early_late_std']:.6f} |")
    lines.append(f"| mid ↔ late | {summary['cos_mid_late_mean']:.6f} | {summary['cos_mid_late_std']:.6f} |")
    lines.append("")
    lines.append(f"**3-NN position classification accuracy**: {summary['knn_position_acc']:.4f} (baseline = 0.333)")
    lines.append("")

    lines.append("## 3. Answerable vs. Unanswerable Breakdown")
    lines.append("")
    for ans_type in ["answerable", "unanswerable"]:
        mask = df["answerability"] == ans_type
        sub = df[mask]
        lines.append(f"**{ans_type}** (n={len(sub)}):")
        lines.append(f"  early↔mid: {sub['cos_early_mid'].mean():.6f} ± {sub['cos_early_mid'].std():.6f}")
        lines.append(f"  early↔late: {sub['cos_early_late'].mean():.6f} ± {sub['cos_early_late'].std():.6f}")
    lines.append("")

    lines.append("## 4. Interpretation")
    lines.append("")
    knn = summary['knn_position_acc']
    if knn > 0.8:
        lines.append("**STRONG POSITION SENSITIVITY**: The hidden state strongly encodes position — ")
        lines.append("a simple 3-NN can classify which position a sample came from with ")
        lines.append(f"{knn:.0%} accuracy. This means routing/gate decisions may depend on where ")
        lines.append("content is placed, not just what it says. Trajectory analysis must separately ")
        lines.append("model position distortion.")
    elif knn > 0.5:
        lines.append("**MODERATE POSITION SENSITIVITY**: Position is detectable above baseline. ")
        lines.append("Position encoding leaves a measurable trace in the hidden state. ")
        lines.append("Routing may be partially influenced by position.")
    else:
        lines.append("**LOW POSITION SENSITIVITY**: Hidden state is nearly position-invariant ")
        lines.append("for the same semantic content at this context length. ")
        lines.append("RoPE ceiling is not yet binding at this scale (~20-30 tokens).")

    lines.append("")
    lines.append("---")
    lines.append("*IC-4 Position Representation Shift — CPU-optimized*")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_m3_v6.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    _run_position_rep_shift(config)


if __name__ == "__main__":
    main()