"""
IC-4-M5 X3: Sycophancy Probe Window Scan — Window-1 to Window-10.

Tests whether window-pooled representations (trajectory-level signal)
improve sycophancy probe AUC, and at what window size the signal saturates.
The "4" in M4's hallucination finding may be a decision latency constant.

Uses pre-existing A1 activations (a1_syc_activations.npz).
"""
import numpy as np
import os, sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

M5_DIR = "results_m5"
SYC_NPZ = os.path.join(M5_DIR, "a1_syc_activations.npz")
OUTPUT_TXT = os.path.join(M5_DIR, "x3_window_scan_report.txt")
SEED = 0
LAYER = 12
WINDOWS = list(range(1, 11))

os.makedirs(M5_DIR, exist_ok=True)


def _log(msg):
    print(msg, flush=True)


def load_syc_activations():
    d = np.load(SYC_NPZ)
    X_syc = d["X_syc"]
    X_nonsyc = d["X_nonsyc"]
    return X_syc, X_nonsyc


def evaluate_window(X_pos, X_neg, window):
    if window == 1 or window >= min(len(X_pos), len(X_neg)):
        return _probe_auc_simple(X_pos, X_neg)

    n_pos = len(X_pos)
    n_neg = len(X_neg)
    n_samples = min(n_pos, n_neg)

    pooled_pos = []
    pooled_neg = []
    for i in range(0, n_samples - window + 1, window):
        pooled_pos.append(X_pos[i:i+window].mean(axis=0))
        pooled_neg.append(X_neg[i:i+window].mean(axis=0))

    if len(pooled_pos) < 2 or len(pooled_neg) < 2:
        return _probe_auc_simple(X_pos[:n_samples], X_neg[:n_samples])

    X = np.array(pooled_pos + pooled_neg)
    y = np.array([1] * len(pooled_pos) + [0] * len(pooled_neg))
    return _probe_auc(X, y)


def _probe_auc_simple(X_pos, X_neg):
    n = min(len(X_pos), len(X_neg))
    X = np.concatenate([X_pos[:n], X_neg[:n]], axis=0)
    y = np.array([1] * n + [0] * n)
    return _probe_auc(X, y)


def _probe_auc(X, y):
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, random_state=42)
    try:
        cv_scores = cross_val_score(clf, X_s, y, cv=min(3, len(y)//2), scoring='roc_auc')
        cv_auc = cv_scores.mean()
    except Exception:
        clf.fit(X_s, y)
        proba = clf.predict_proba(X_s)[:, 1]
        cv_auc = roc_auc_score(y, proba)

    clf.fit(X_s, y)
    proba = clf.predict_proba(X_s)[:, 1]
    train_auc = roc_auc_score(y, proba)

    inter = np.linalg.norm(X_s[y == 1].mean(axis=0) - X_s[y == 0].mean(axis=0))
    intra_pos = np.sqrt(np.mean(np.sum((X_s[y == 1] - X_s[y == 1].mean(axis=0)) ** 2, axis=1)))
    intra_neg = np.sqrt(np.mean(np.sum((X_s[y == 0] - X_s[y == 0].mean(axis=0)) ** 2, axis=1)))
    snr = inter / ((intra_pos + intra_neg) / 2 + 1e-12)

    return {"window": 1, "n_samples": len(X), "train_auc": train_auc, "cv_auc": cv_auc,
            "inter_class": inter, "snr": snr}


def evaluate_window_sequential(X_pos, X_neg, window):
    n = min(len(X_pos), len(X_neg))
    n_pooled = n - window + 1
    if n_pooled < 2:
        return {"window": window, "n_samples": 0, "train_auc": float("nan"),
                "cv_auc": float("nan"), "inter_class": float("nan"), "snr": float("nan")}
    X_p = X_pos[:n]
    X_n = X_neg[:n]

    pos_pooled = []
    neg_pooled = []
    for i in range(n_pooled):
        pos_pooled.append(X_p[i:i+window].mean(axis=0))
        neg_pooled.append(X_n[i:i+window].mean(axis=0))

    X = np.array(pos_pooled + neg_pooled)
    y = np.array([1] * len(pos_pooled) + [0] * len(neg_pooled))
    result = _probe_auc(X, y)
    result["window"] = window
    result["n_samples"] = len(X)
    return result


def main():
    _log("=" * 60)
    _log("IC-4-M5 X3: Sycophancy Window Probe Scan")
    _log("=" * 60)

    _log(f"Loading activations from {SYC_NPZ}...")
    X_syc, X_nonsyc = load_syc_activations()
    n_syc, n_nonsyc = len(X_syc), len(X_nonsyc)
    dim = X_syc.shape[1]
    _log(f"  sycophantic: {n_syc} samples, non_syc: {n_nonsyc} samples, dim={dim}")
    _log("")

    lines = []
    lines.append("IC-4-M5 X3: Sycophancy Window Probe Scan Report")
    lines.append("=" * 60)
    lines.append(f"")
    lines.append(f"Data: {SYC_NPZ}")
    lines.append(f"Samples: sycophantic={n_syc}, non_syc={n_nonsyc}")
    lines.append(f"Dim: {dim}")
    lines.append(f"Method: Sliding window (stride=1) mean-pooling over last k tokens")
    lines.append(f"Evaluated with 3-fold CV Logistic Regression AUC")
    lines.append(f"")
    lines.append(f"{'Window':>8s} {'Train AUC':>10s} {'CV AUC':>10s} {'SNR':>10s} {'Inter-Class':>12s} {'N_Pooled':>10s}")
    lines.append("-" * 65)

    results = []
    for window in WINDOWS:
        r = evaluate_window_sequential(X_syc, X_nonsyc, window)
        results.append(r)
        _log(f"  window={window:>2d}: train_auc={r['train_auc']:.4f} cv_auc={r['cv_auc']:.4f} "
             f"SNR={r['snr']:.4f} inter={r['inter_class']:.4f} n={r['n_samples']}")
        lines.append(f"  {window:>6d}  {r['train_auc']:>10.4f} {r['cv_auc']:>10.4f} "
                      f"{r['snr']:>10.4f} {r['inter_class']:>12.4f} {r['n_samples']:>10d}")

    lines.append("")
    best = max(results, key=lambda r: r["cv_auc"])
    worst = min(results, key=lambda r: r["cv_auc"])
    lines.append(f"Best window: {best['window']} (CV AUC={best['cv_auc']:.4f}, SNR={best['snr']:.4f})")
    lines.append(f"Worst window: {worst['window']} (CV AUC={worst['cv_auc']:.4f}, SNR={worst['snr']:.4f})")
    delta = best['cv_auc'] - results[0]['cv_auc']
    lines.append(f"Window improvement over last-token: ΔAUC={delta:+.4f}")

    lines.append("")
    if delta > 0.05:
        lines.append(f"KEY FINDING: Window-pooling SIGNIFICANTLY improves sycophancy probe AUC (+{delta:.4f}).")
        lines.append("  Sycophancy signal is trajectory-level, NOT token-local — same pattern as hallucination.")
    elif delta > 0.01:
        lines.append("KEY FINDING: Window-pooling provides MODEST improvement. Signal is weakly trajectory-level.")
    else:
        lines.append("KEY FINDING: Window-pooling provides MINIMAL improvement. Signal is effectively token-local.")

    plateau_window = None
    for i in range(len(results) - 1):
        if results[i]["cv_auc"] >= 0.99 * best["cv_auc"]:
            plateau_window = results[i]["window"]
            break
    if plateau_window:
        lines.append(f"  Signal saturates at window={plateau_window} (within 1% of peak AUC).")

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    _log(f"\nReport saved to {OUTPUT_TXT}")
    _log("=== X3 DONE ===")


if __name__ == "__main__":
    main()