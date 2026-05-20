"""
IC-4 P1.5: Failure Mode Analysis — Activation Geometry Diagnosis.

Analyses precomputed activation .npz files for all 5 P1 configs:
  - seed=0 / layer=12  (reference success)
  - seed=1 / layer=12  (success)
  - seed=2 / layer=12  (artifact)
  - seed=0 / layer=11  (success)
  - seed=0 / layer=13  (artifact)

Outputs a comprehensive geometry report to results_p15/.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)

sys.path.insert(0, BASE_DIR)
from src.activation_collector import load_activations
from src.steering import compute_steering_vector, compute_random_vector, compute_shuffled_vector


CONFIGS = [
    {"label": "seed0_layer12_ref",  "seed": 0, "layer": 12, "verdict": "SUCCESS",  "tag": "reference"},
    {"label": "seed1_layer12",      "seed": 1, "layer": 12, "verdict": "SUCCESS",  "tag": "cross-seed"},
    {"label": "seed2_layer12",      "seed": 2, "layer": 12, "verdict": "ARTIFACT", "tag": "cross-seed"},
    {"label": "seed0_layer11",      "seed": 0, "layer": 11, "verdict": "SUCCESS",  "tag": "cross-layer"},
    {"label": "seed0_layer13",      "seed": 0, "layer": 13, "verdict": "ARTIFACT", "tag": "cross-layer"},
]

OUT_DIR = os.path.join(BASE_DIR, "results_p15")
os.makedirs(OUT_DIR, exist_ok=True)


def analyze_activations(pos, neg):
    d = pos.shape[1]
    pos_mean = pos.mean(axis=0)
    neg_mean = neg.mean(axis=0)
    diff_vec = pos_mean - neg_mean
    diff_norm = float(np.linalg.norm(diff_vec))

    pos_var = np.trace(np.cov(pos.T))
    neg_var = np.trace(np.cov(neg.T))
    all_data = np.concatenate([pos, neg], axis=0)
    total_var = np.trace(np.cov(all_data.T))
    signal_var = diff_norm**2

    mean_separation = diff_norm
    noise_proxy = pos_var + neg_var
    snr_proxy = signal_var / max(noise_proxy, 1e-8)

    pos_cov = np.cov(pos.T)
    neg_cov = np.cov(neg.T)
    U_pos, S_pos, _ = np.linalg.svd(pos_cov)
    U_neg, S_neg, _ = np.linalg.svd(neg_cov)
    top3_pos = float(np.sum(S_pos[:3]) / max(np.sum(S_pos), 1e-8))
    top3_neg = float(np.sum(S_neg[:3]) / max(np.sum(S_neg), 1e-8))

    combined = np.concatenate([pos - pos_mean, neg - neg_mean], axis=0)
    U_comb, S_comb, _ = np.linalg.svd(combined.T @ combined / combined.shape[0])
    total_pca_rank = float(np.sum(S_comb))
    pc1_frac = float(S_comb[0] / max(total_pca_rank, 1e-8))
    pc3_frac = float(np.sum(S_comb[:3]) / max(total_pca_rank, 1e-8))

    inter_class_distance = np.linalg.norm(pos_mean - neg_mean)
    intra_pos_distance = np.mean([np.linalg.norm(p - pos_mean) for p in pos])
    intra_neg_distance = np.mean([np.linalg.norm(n - neg_mean) for n in neg])
    davies_bouldin_proxy = (intra_pos_distance + intra_neg_distance) / max(inter_class_distance, 1e-8)

    return {
        "dim": d,
        "n_pos": pos.shape[0],
        "n_neg": neg.shape[0],
        "mean_separation_L2": round(mean_separation, 6),
        "pos_total_variance": round(pos_var, 4),
        "neg_total_variance": round(neg_var, 4),
        "total_variance": round(total_var, 4),
        "signal_variance": round(signal_var, 6),
        "snr_proxy": round(snr_proxy, 6),
        "top3_pca_pos": round(top3_pos, 4),
        "top3_pca_neg": round(top3_neg, 4),
        "pc1_fraction": round(pc1_frac, 4),
        "pc3_fraction": round(pc3_frac, 4),
        "inter_class_distance": round(inter_class_distance, 4),
        "intra_pos_distance_mean": round(intra_pos_distance, 4),
        "intra_neg_distance_mean": round(intra_neg_distance, 4),
        "davies_bouldin_proxy": round(davies_bouldin_proxy, 4),
    }


def analyze_vectors(vec_steer, vec_random, vec_shuffled):
    cos_steer_random = float(np.dot(vec_steer, vec_random))
    cos_steer_shuffled = float(np.dot(vec_steer, vec_shuffled))
    cos_random_shuffled = float(np.dot(vec_random, vec_shuffled))
    return {
        "norm_steering": round(float(np.linalg.norm(vec_steer)), 6),
        "norm_random": round(float(np.linalg.norm(vec_random)), 6),
        "norm_shuffled": round(float(np.linalg.norm(vec_shuffled)), 6),
        "cos_steering_random": round(cos_steer_random, 6),
        "cos_steering_shuffled": round(cos_steer_shuffled, 6),
        "cos_random_shuffled": round(cos_random_shuffled, 6),
    }


def main():
    results = []
    for cfg in CONFIGS:
        act_path = os.path.join(BASE_DIR, "results_m3", f"activations_s{cfg['seed']}_l{cfg['layer']}.npz")
        acts = load_activations(act_path)
        pos, neg = acts["positive"], acts["negative"]
        dim = pos.shape[1]

        geom = analyze_activations(pos, neg)

        steer_v = compute_steering_vector(pos, neg)
        rand_v = compute_random_vector(dim)
        shuf_v = compute_shuffled_vector(pos, neg)
        vecs = analyze_vectors(steer_v, rand_v, shuf_v)

        per_sample_dist = []
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.metrics import roc_auc_score, accuracy_score

        X = np.concatenate([pos, neg], axis=0)
        y = np.array([1] * pos.shape[0] + [0] * neg.shape[0], dtype=np.int32)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        clf = LogisticRegression(max_iter=2000, random_state=42)
        clf.fit(X_scaled, y)
        train_acc = float(accuracy_score(y, clf.predict(X_scaled)))
        n_folds = min(3, len(y) // 2)
        cv_scores = []
        if n_folds >= 2:
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy")
        cv_acc = float(np.mean(cv_scores)) if len(cv_scores) > 0 else None
        auc = float(roc_auc_score(y, clf.predict_proba(X_scaled)[:, 1]))

        proba = clf.predict_proba(X_scaled)[:, 1]
        proba_pos = proba[y == 1]
        proba_neg = proba[y == 0]

        row = {
            "label": cfg["label"],
            "seed": cfg["seed"],
            "layer": cfg["layer"],
            "verdict": cfg["verdict"],
            "tag": cfg["tag"],
            **geom,
            **vecs,
            "train_acc": round(train_acc, 4),
            "cv_acc": round(cv_acc, 4) if cv_acc is not None else None,
            "auc": round(auc, 4),
            "proba_pos_mean": round(float(np.mean(proba_pos)), 4),
            "proba_pos_std": round(float(np.std(proba_pos)), 4),
            "proba_pos_min": round(float(np.min(proba_pos)), 4),
            "proba_pos_max": round(float(np.max(proba_pos)), 4),
            "proba_neg_mean": round(float(np.mean(proba_neg)), 4),
            "proba_neg_std": round(float(np.std(proba_neg)), 4),
            "proba_neg_min": round(float(np.min(proba_neg)), 4),
            "proba_neg_max": round(float(np.max(proba_neg)), 4),
            "proba_separation": round(float(np.mean(proba_pos) - np.mean(proba_neg)), 4),
        }
        results.append(row)

    import pandas as pd
    df = pd.DataFrame(results)
    csv_path = os.path.join(OUT_DIR, "failure_analysis_metrics.csv")
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"Saved {csv_path} ({len(df)} rows)")

    summary = {
        "generated_at": datetime.now().isoformat(),
        "source": "src/run_p15_failure_analysis.py",
        "num_configs": len(CONFIGS),
        "artifacts": [r["label"] for r in results if r["verdict"] == "ARTIFACT"],
        "successes": [r["label"] for r in results if r["verdict"] == "SUCCESS"],
    }
    json_path = os.path.join(OUT_DIR, "failure_analysis_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved {json_path}")

    print("\n" + "=" * 80)
    print("P1.5 FAILURE ANALYSIS — ACTIVATION GEOMETRY REPORT")
    print("=" * 80)

    key_cols = [
        "label", "verdict",
        "mean_separation_L2", "snr_proxy", "davies_bouldin_proxy",
        "cos_steering_shuffled", "pc1_fraction",
        "train_acc", "cv_acc", "proba_separation",
    ]
    sub = df[key_cols].copy()
    print(sub.to_string(index=False))

    print("\n--- KEY METRICS FOR FAILURE DIAGNOSIS ---")
    arti = df[df["verdict"] == "ARTIFACT"]
    succ = df[df["verdict"] == "SUCCESS"]

    def compare(metric, lower_is_worse=True):
        a_mean = float(arti[metric].mean())
        s_mean = float(succ[metric].mean())
        ratio = a_mean / max(s_mean, 1e-8)
        direction = "worse" if (lower_is_worse and ratio < 1) or (not lower_is_worse and ratio < 1) else "better" if ratio > 1 else "same"
        return a_mean, s_mean, ratio, direction

    comparisons = [
        ("mean_separation_L2", True, "Mean separation (larger = better class separability)"),
        ("snr_proxy", True, "SNR proxy (larger = stronger signal)"),
        ("davies_bouldin_proxy", False, "Davies-Bouldin proxy (smaller = better separation)"),
        ("cos_steering_shuffled", False, "cos(steering, shuffled) — larger = more overlap with control"),
        ("pc1_fraction", True, "PC1 fraction of total variance (larger = simpler structure)"),
    ]
    for metric, higher_better, desc in comparisons:
        a_m, s_m, ratio, direction = compare(metric, lower_is_worse=not higher_better)
        flag = "⚠️" if direction == "worse" else "✅"
        print(f"  {flag} {desc}:")
        print(f"       ARTIFACT mean={a_m:.4f}, SUCCESS mean={s_m:.4f}, ratio={ratio:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()