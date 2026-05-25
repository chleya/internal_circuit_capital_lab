"""
Phase 7 3.3B: LLM Hidden State Consolidation -- V2.
=====================================================
Cross-seed generalization with PCA difficulty scaling.

Design:
  Train on seed-0 only (60 samples), test on seed-2 (60 samples)
  PCA reduction 2D-896D to progressively increase difficulty
  Compare: Per-Class KMeans vs X-only KMeans vs Y-aware vs frequency baseline

Hypothesis:
  Per-Class KMeans should generalize better across seeds, especially in low dims.

Usage:
  cd F:\internal_circuit_capital_lab\IC-4-M0
  python src/run_c6_llm_consolidation.py
"""

import os, sys, time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ACTIVATIONS_DIR = "results_m3"
RESULTS_DIR = "results_c6_llm_consolidation"
os.makedirs(RESULTS_DIR, exist_ok=True)

LAYERS = [11, 12, 13]
DIMS = [2, 4, 8, 16, 32, 64, 128, 896]


def load_activations(layer, seeds):
    X_all, Y_all = [], []
    for s in seeds:
        path = os.path.join(ACTIVATIONS_DIR, f"activations_s{s}_l{layer}.npz")
        data = np.load(path)
        for key, y_val in [("positive", 1), ("negative", 0)]:
            X_all.append(data[key].astype(np.float32))
            Y_all.append(np.full(len(data[key]), y_val, dtype=np.int32))
    X = np.concatenate(X_all, axis=0)
    Y = np.concatenate(Y_all, axis=0)
    return X, Y


def pca_reduce(X_train, X_test, dim):
    if dim >= min(X_train.shape[1], len(X_train)):
        return X_train, X_test
    pca = PCA(n_components=dim, random_state=42)
    X_train_r = pca.fit_transform(X_train).astype(np.float32)
    X_test_r = pca.transform(X_test).astype(np.float32)
    return X_train_r, X_test_r


class PerClassKMeans:
    def __init__(self, n_prototypes_per_class=3):
        self.n_prototypes_per_class = n_prototypes_per_class
        self._centroids = None
        self._Y_centroids = None

    def fit(self, X, Y):
        centroids_parts, Y_parts = [], []
        for c in [0, 1]:
            mask = Y == c
            X_c = X[mask]
            if len(X_c) == 0:
                continue
            nc = min(self.n_prototypes_per_class, max(1, len(X_c) // 3))
            if nc < 1:
                continue
            km = KMeans(n_clusters=nc, random_state=42, n_init="auto")
            labels = km.fit_predict(X_c)
            for i in range(nc):
                m = labels == i
                if m.sum() > 0:
                    centroids_parts.append(km.cluster_centers_[i])
                    Y_parts.append(float(c))
        if centroids_parts:
            self._centroids = np.stack(centroids_parts)
            self._Y_centroids = np.array(Y_parts)

    def predict(self, X_query):
        if self._centroids is None:
            return np.ones(len(X_query)) * 0.5
        idxs, _ = pairwise_distances_argmin_min(X_query, self._centroids)
        return self._Y_centroids[idxs]


class XOnlyKMeans:
    def __init__(self, n_prototypes=5):
        self.n_prototypes = n_prototypes
        self._centroids = None
        self._Y_centroids = None

    def fit(self, X, Y):
        nc = min(self.n_prototypes, max(1, len(X) // 2))
        km = KMeans(n_clusters=nc, random_state=42, n_init="auto")
        labels = km.fit_predict(X)
        self._centroids = km.cluster_centers_
        self._Y_centroids = np.array([Y[labels == i].mean() if (labels == i).sum() > 0 else 0.5 for i in range(nc)])

    def predict(self, X_query):
        if self._centroids is None:
            return np.ones(len(X_query)) * 0.5
        idxs, _ = pairwise_distances_argmin_min(X_query, self._centroids)
        return self._Y_centroids[idxs]


class YAwareKMeansLLM:
    def __init__(self, n_prototypes=5, y_weight=2.0):
        self.n_prototypes = n_prototypes
        self.y_weight = y_weight
        self._centroids_X = None
        self._Y_centroids = None

    def fit(self, X, Y):
        nc = min(self.n_prototypes, max(1, len(X) // 2))
        dX = np.std(X, axis=0) + 1e-8
        dY = np.std(Y) + 1e-8
        joint = np.hstack([X / dX, self.y_weight * Y.reshape(-1, 1) / dY])
        km = KMeans(n_clusters=nc, random_state=42, n_init="auto")
        labels = km.fit_predict(joint)
        self._centroids_X = np.zeros((nc, X.shape[1]))
        self._Y_centroids = np.array([Y[labels == i].mean() if (labels == i).sum() > 0 else 0.5 for i in range(nc)])
        for i in range(nc):
            m = labels == i
            if m.sum() > 0:
                self._centroids_X[i] = X[m].mean(axis=0)

    def predict(self, X_query):
        if self._centroids_X is None:
            return np.ones(len(X_query)) * 0.5
        idxs, _ = pairwise_distances_argmin_min(X_query, self._centroids_X)
        return self._Y_centroids[idxs]


class FrequencyBaseline:
    def __init__(self):
        self._pos_freq = 0.5

    def fit(self, X, Y):
        self._pos_freq = Y.mean()

    def predict(self, X_query):
        return np.ones(len(X_query)) * self._pos_freq


def accuracy_at_threshold(preds_probs, Y_true, threshold=0.5):
    return np.mean((preds_probs >= threshold) == (Y_true >= threshold))


def evaluate_dim(X_train, Y_train, X_test, Y_test, dim, n_trials=3):
    results = {"pc": [], "km": [], "ya": [], "freq": []}
    for _ in range(n_trials):
        X_tr, X_te = pca_reduce(X_train, X_test, dim)

        freq = FrequencyBaseline()
        freq.fit(X_tr, Y_train)
        results["freq"].append(accuracy_at_threshold(freq.predict(X_te), Y_test))

        km = XOnlyKMeans(n_prototypes=5)
        km.fit(X_tr, Y_train)
        results["km"].append(accuracy_at_threshold(km.predict(X_te), Y_test))

        ya = YAwareKMeansLLM(n_prototypes=5, y_weight=2.0)
        ya.fit(X_tr, Y_train)
        results["ya"].append(accuracy_at_threshold(ya.predict(X_te), Y_test))

        pc = PerClassKMeans(n_prototypes_per_class=3)
        pc.fit(X_tr, Y_train)
        results["pc"].append(accuracy_at_threshold(pc.predict(X_te), Y_test))

    return {k: np.mean(v) for k, v in results.items()}


def main():
    run_id = f"c6v2_{int(time.time())}"
    log_path = os.path.join(RESULTS_DIR, "run_log_v2.txt")

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()

    log("=" * 60)
    log("Phase 7 3.3B V2: Cross-Seed + PCA Difficulty Scaling")
    log("=" * 60)
    log(f"Design: train on seed-0 (60 samples), test on seed-2 (60 samples)")
    log(f"PCA dims: {DIMS}")
    log("")

    t0 = time.time()
    all_rows = []

    for layer in LAYERS:
        log(f"{'='*40}")
        log(f"Layer {layer}")
        X_train, Y_train = load_activations(layer, [0])
        X_test, Y_test = load_activations(layer, [2])
        log(f"  Train: s0={len(X_train)} (pos={Y_train.sum():.0f}, neg={len(Y_train)-Y_train.sum():.0f})")
        log(f"  Test:  s2={len(X_test)} (pos={Y_test.sum():.0f}, neg={len(Y_test)-Y_test.sum():.0f})")

        for dim in DIMS:
            r = evaluate_dim(X_train, Y_train, X_test, Y_test, dim, n_trials=3)
            delta = r["pc"] - r["km"]
            marker = " **" if delta > 0.02 else ""
            log(f"  dim={dim:4d}  PC={r['pc']:.4f}  KM={r['km']:.4f}  YA={r['ya']:.4f}  freq={r['freq']:.4f}  d={delta:+.4f}{marker}")
            all_rows.append({
                "layer": layer, "dim": dim,
                "pc": r["pc"], "km": r["km"], "ya": r["ya"], "freq": r["freq"],
                "delta": delta,
            })
        log("")

    log("=" * 60)
    log("Summary: Per-Class KMeans advantage over X-only KMeans")
    log("=" * 60)
    log(f"  {'Layer':>6s}  {'Dim':>6s}  {'PC-KM':>8s}")
    positive_dims = []
    for row in all_rows:
        marker = " +" if row["delta"] > 0 else ""
        log(f"  {row['layer']:6d}  {row['dim']:6d}  {row['delta']:+8.4f}{marker}")
        if row["delta"] > 0.01:
            positive_dims.append((row["layer"], row["dim"], row["delta"]))

    if positive_dims:
        log(f"\n  Positive advantages: {positive_dims}")
        log("  VERDICT: CONDITIONAL PASS -- advantage appears in low-dim regime")
    else:
        log("\n  No positive advantage at any dimension.")
        log("  VERDICT: NEGATIVE -- gap between counterfactual and LLM confirmed")
        log("  The binary classification task is too easy even cross-seed in 896D.")

    elapsed = time.time() - t0
    log(f"\n  Total time: {elapsed:.0f}s")
    log("=" * 60)
    log("Phase 7 3.3B V2 complete.")
    log("=" * 60)


if __name__ == "__main__":
    main()