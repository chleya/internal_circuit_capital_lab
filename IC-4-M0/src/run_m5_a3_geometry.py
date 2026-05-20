"""
IC-4-M5 A3: Activation Geometry Comparison — Hallucination vs Sycophancy.

Compares the activation-space geometry of two behaviors to understand why
IC-4 steering works for hallucination but fails for sycophancy.

Metrics:
  1. Inter-class separation: ||mean(pos)-mean(neg)|| for each behavior
  2. Intra-class compactness: mean pairwise distance within each class
  3. Signal-to-noise ratio: inter_class / intra_class
  4. Cross-behavior cosine: cos(hallucination_sv, sycophancy_sv)
  5. PCA visualization: 2D projection of all activations
  6. Linear separability: probe AUC for each behavior
"""
import numpy as np
import os, sys
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

M3_DIR = "results_m3"
M5_DIR = "results_m5"
SEED, LAYER = 0, 12
OUTPUT = "results_m5/a3_geometry_report.txt"


def _log(msg):
    print(msg, flush=True)


def _load_behavior(npz_path, pos_key, neg_key):
    d = np.load(npz_path)
    return d[pos_key], d[neg_key]


def _inter_class_dist(pos, neg):
    return np.linalg.norm(pos.mean(axis=0) - neg.mean(axis=0))


def _intra_class_dist(X):
    X_c = X - X.mean(axis=0)
    return np.sqrt(np.mean(np.sum(X_c ** 2, axis=1)))


def _probe_auc(pos, neg):
    X = np.concatenate([pos, neg], axis=0)
    y = np.array([1] * len(pos) + [0] * len(neg))
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(X_s, y)
    proba = clf.predict_proba(X_s)[:, 1]
    return roc_auc_score(y, proba)


def _pairwise_cos(v1, v2):
    v1n = v1 / (np.linalg.norm(v1) + 1e-12)
    v2n = v2 / (np.linalg.norm(v2) + 1e-12)
    return np.dot(v1n, v2n)


def main():
    _log("=" * 60)
    _log("IC-4-M5 A3: Activation Geometry Comparison")
    _log("=" * 60)

    lines = []
    lines.append("IC-4-M5 A3: Activation Geometry Report")
    lines.append("=" * 60)
    lines.append("")

    hall_path = os.path.join(M3_DIR, f"activations_s{SEED}_l{LAYER}.npz")
    syc_path = os.path.join(M5_DIR, "a1_syc_activations.npz")

    if not os.path.exists(syc_path):
        _log(f"ERROR: Sycophancy activations not found at {syc_path}")
        _log("Run A1 first to generate the activations.")
        return

    _log(f"Loading hallucination activations: {hall_path}")
    hall_pos, hall_neg = _load_behavior(hall_path, "positive", "negative")
    _log(f"  hallucination: pos={hall_pos.shape} neg={hall_neg.shape}")

    _log(f"Loading sycophancy activations: {syc_path}")
    syc_pos, syc_neg = _load_behavior(syc_path, "X_syc", "X_nonsyc")
    _log(f"  sycophancy: pos={syc_pos.shape} neg={syc_neg.shape}")

    dim = hall_pos.shape[1]
    assert syc_pos.shape[1] == dim, f"Dimension mismatch: {dim} vs {syc_pos.shape[1]}"

    sv_hall = hall_pos.mean(axis=0) - hall_neg.mean(axis=0)
    sv_hall = sv_hall / (np.linalg.norm(sv_hall) + 1e-12)
    sv_syc = syc_pos.mean(axis=0) - syc_neg.mean(axis=0)
    sv_syc = sv_syc / (np.linalg.norm(sv_syc) + 1e-12)

    inter_hall = _inter_class_dist(hall_pos, hall_neg)
    inter_syc = _inter_class_dist(syc_pos, syc_neg)
    intra_hall_pos = _intra_class_dist(hall_pos)
    intra_hall_neg = _intra_class_dist(hall_neg)
    intra_syc_pos = _intra_class_dist(syc_pos)
    intra_syc_neg = _intra_class_dist(syc_neg)
    snr_hall = inter_hall / ((intra_hall_pos + intra_hall_neg) / 2 + 1e-12)
    snr_syc = inter_syc / ((intra_syc_pos + intra_syc_neg) / 2 + 1e-12)

    _log("\n--- 1. Inter-class vs Intra-class Distances ---")
    _log(f"  {'':>18s} {'Hallucination':>15s} {'Sycophancy':>15s} {'Ratio(S/H)':>12s}")
    _log(f"  {'inter_class':>18s} {inter_hall:15.4f} {inter_syc:15.4f} {inter_syc/(inter_hall+1e-12):12.4f}")
    _log(f"  {'intra_pos':>18s} {intra_hall_pos:15.4f} {intra_syc_pos:15.4f}")
    _log(f"  {'intra_neg':>18s} {intra_hall_neg:15.4f} {intra_syc_neg:15.4f}")
    _log(f"  {'SNR (inter/intra)':>18s} {snr_hall:15.4f} {snr_syc:15.4f} {snr_syc/(snr_hall+1e-12):12.4f}")

    lines.append("1. INTER-CLASS vs INTRA-CLASS DISTANCES")
    lines.append("-" * 40)
    lines.append(f"  Hallucination: inter={inter_hall:.4f} intra_pos={intra_hall_pos:.4f} intra_neg={intra_hall_neg:.4f} SNR={snr_hall:.4f}")
    lines.append(f"  Sycophancy:    inter={inter_syc:.4f} intra_pos={intra_syc_pos:.4f} intra_neg={intra_syc_neg:.4f} SNR={snr_syc:.4f}")

    auc_hall = _probe_auc(hall_pos, hall_neg)
    auc_syc = _probe_auc(syc_pos, syc_neg)

    _log(f"\n--- 2. Linear Separability (Probe AUC) ---")
    _log(f"  Hallucination AUC: {auc_hall:.4f}")
    _log(f"  Sycophancy AUC:    {auc_syc:.4f}")

    lines.append("")
    lines.append("2. LINEAR SEPARABILITY (PROBE AUC)")
    lines.append("-" * 40)
    lines.append(f"  Hallucination AUC: {auc_hall:.4f}")
    lines.append(f"  Sycophancy AUC:    {auc_syc:.4f}")

    cos_hs = _pairwise_cos(sv_hall, sv_syc)
    _log(f"\n--- 3. Cross-Behavior Steering Vector Cosine ---")
    _log(f"  cos(hallucination_SV, sycophancy_SV) = {cos_hs:+.4f}")

    pc_hall_pos = np.dot(hall_pos, sv_hall)
    pc_hall_neg = np.dot(hall_neg, sv_hall)
    pc_syc_pos_hall = np.dot(syc_pos, sv_hall)
    pc_syc_neg_hall = np.dot(syc_neg, sv_hall)
    pc_syc_pos_syc = np.dot(syc_pos, sv_syc)
    pc_syc_neg_syc = np.dot(syc_neg, sv_syc)
    pc_hall_pos_syc = np.dot(hall_pos, sv_syc)
    pc_hall_neg_syc = np.dot(hall_neg, sv_syc)

    _log(f"\n--- 4. Projection onto Steering Vectors ---")
    _log(f"  {'':>25s} {'Hallucination SV':>18s} {'Sycophancy SV':>18s}")
    _log(f"  {'hall_pos (hallucinated)':>25s} {pc_hall_pos.mean():18.4f} {pc_hall_pos_syc.mean():18.4f}")
    _log(f"  {'hall_neg (refused)':>25s} {pc_hall_neg.mean():18.4f} {pc_hall_neg_syc.mean():18.4f}")
    _log(f"  {'hall Δ (pos-neg)':>25s} {(pc_hall_pos.mean()-pc_hall_neg.mean()):18.4f} {(pc_hall_pos_syc.mean()-pc_hall_neg_syc.mean()):18.4f}")
    _log(f"  {'syc_pos (sycophantic)':>25s} {pc_syc_pos_hall.mean():18.4f} {pc_syc_pos_syc.mean():18.4f}")
    _log(f"  {'syc_neg (non-syc)':>25s} {pc_syc_neg_hall.mean():18.4f} {pc_syc_neg_syc.mean():18.4f}")
    _log(f"  {'syc Δ (pos-neg)':>25s} {(pc_syc_pos_hall.mean()-pc_syc_neg_hall.mean()):18.4f} {(pc_syc_pos_syc.mean()-pc_syc_neg_syc.mean()):18.4f}")

    lines.append("")
    lines.append("3. CROSS-BEHAVIOR STEERING VECTOR COSINE")
    lines.append("-" * 40)
    lines.append(f"  cos(hall_SV, syc_SV) = {cos_hs:+.4f}")

    _log(f"\n--- 5. PCA 2D Analysis ---")
    all_data = np.concatenate([hall_pos, hall_neg, syc_pos, syc_neg], axis=0)
    pca = PCA(n_components=2, random_state=42)
    all_2d = pca.fit_transform(all_data)
    ev_ratio = pca.explained_variance_ratio_

    n_hp, n_hn = len(hall_pos), len(hall_neg)
    n_sp, n_sn = len(syc_pos), len(syc_neg)
    idx_hp = slice(0, n_hp)
    idx_hn = slice(n_hp, n_hp + n_hn)
    idx_sp = slice(n_hp + n_hn, n_hp + n_hn + n_sp)
    idx_sn = slice(n_hp + n_hn + n_sp, n_hp + n_hn + n_sp + n_sn)

    centroids = {
        "hall_pos": all_2d[idx_hp].mean(axis=0),
        "hall_neg": all_2d[idx_hn].mean(axis=0),
        "syc_pos": all_2d[idx_sp].mean(axis=0),
        "syc_neg": all_2d[idx_sn].mean(axis=0),
    }

    _log(f"  PCA explained variance: PC1={ev_ratio[0]:.4f} PC2={ev_ratio[1]:.4f} sum={ev_ratio.sum():.4f}")
    _log(f"  Centroids:")
    for name, c in centroids.items():
        _log(f"    {name:>15s}: ({c[0]:+.4f}, {c[1]:+.4f})")

    hall_sep_2d = np.linalg.norm(centroids["hall_pos"] - centroids["hall_neg"])
    syc_sep_2d = np.linalg.norm(centroids["syc_pos"] - centroids["syc_neg"])
    cross_sep = np.linalg.norm(
        (centroids["hall_pos"] + centroids["hall_neg"]) / 2 -
        (centroids["syc_pos"] + centroids["syc_neg"]) / 2)

    _log(f"  Hallucination 2D separation: {hall_sep_2d:.4f}")
    _log(f"  Sycophancy 2D separation:    {syc_sep_2d:.4f}")
    _log(f"  Cross-behavior 2D separation: {cross_sep:.4f}")

    lines.append("")
    lines.append("4. PCA 2D SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  PCA var: PC1={ev_ratio[0]:.4f} PC2={ev_ratio[1]:.4f} sum={ev_ratio.sum():.4f}")
    lines.append(f"  Hall sep 2D: {hall_sep_2d:.4f}  Syc sep 2D: {syc_sep_2d:.4f}  Cross sep: {cross_sep:.4f}")

    _log("\n--- 6. Key Findings ---")
    findings = []
    if snr_hall > snr_syc * 1.5:
        findings.append(f"★ Hallucination SNR ({snr_hall:.3f}) is {snr_hall/(snr_syc+1e-12):.1f}x sycophancy SNR ({snr_syc:.3f}). Hallucination is MORE linearly separable.")
    elif snr_syc > snr_hall * 1.5:
        findings.append(f"★ Sycophancy SNR ({snr_syc:.3f}) is {snr_syc/(snr_hall+1e-12):.1f}x hallucination SNR ({snr_hall:.3f}). Sycophancy is MORE linearly separable (but steering fails — non-linear).")
    else:
        findings.append(f"★ SNR comparable: hallucination={snr_hall:.3f} sycophancy={snr_syc:.3f}")

    if abs(cos_hs) < 0.15:
        findings.append(f"★ Steering vectors are NEARLY ORTHOGONAL (cos={cos_hs:+.4f}). The two behaviors occupy different subspaces in activation space.")
    elif abs(cos_hs) < 0.4:
        findings.append(f"★ Steering vectors are MODERATELY correlated (cos={cos_hs:+.4f}). Partial subspace overlap.")
    else:
        findings.append(f"★ Steering vectors are STRONGLY correlated (cos={cos_hs:+.4f}). Similar directions — failure must be due to other factors.")

    if auc_hall > auc_syc:
        findings.append(f"★ Hallucination AUC ({auc_hall:.4f}) > Sycophancy AUC ({auc_syc:.4f}). Sycophancy is inherently harder to separate linearly.")

    if cross_sep > hall_sep_2d * 0.5:
        findings.append(f"★ Cross-behavior 2D separation ({cross_sep:.4f}) is significant. Hallucination and sycophancy occupy DISTINCT regions in PC space.")

    for f_text in findings:
        _log(f"  {f_text}")
    lines.append("")
    lines.append("5. KEY FINDINGS")
    lines.append("-" * 40)
    for f_text in findings:
        lines.append(f"  {f_text}")

    with open(OUTPUT, "w") as f:
        f.write("\n".join(lines))
    _log(f"\nReport saved to {OUTPUT}")
    _log("=== A3 DONE ===")


if __name__ == "__main__":
    main()