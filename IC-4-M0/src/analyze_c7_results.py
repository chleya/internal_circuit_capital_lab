"""
Phase 9-B Analysis: Cross-Checkpoint Consolidation Results.
============================================================
Reads saved checkpoint features and runs KNN divergence + PerClassKMeans tests.
"""
import os, json, time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

RESULTS_DIR = "results_c7_multi_checkpoint"
LOG_PATH = os.path.join(RESULTS_DIR, "analysis_log.txt")

def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()

def main():
    log("=" * 64)
    log("Phase 9-B Analysis: Cross-Checkpoint Consolidation")
    log("=" * 64)

    t0 = time.time()
    y_train = np.load(os.path.join(RESULTS_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(RESULTS_DIR, "y_test.npy"))
    log(f"Loaded labels: train={len(y_train)}, test={len(y_test)}")

    checkpoint_files = []
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if fname.startswith("X_test_checkpoint_") and fname.endswith(".npy"):
            ckpt_id = int(fname.replace("X_test_checkpoint_", "").replace(".npy", ""))
            checkpoint_files.append((ckpt_id, fname))

    checkpoint_features_train = {}
    checkpoint_features_test = {}
    for ckpt_id, _ in checkpoint_files:
        X_train = np.load(os.path.join(RESULTS_DIR, f"X_train_checkpoint_{ckpt_id}.npy"))
        X_test = np.load(os.path.join(RESULTS_DIR, f"X_test_checkpoint_{ckpt_id}.npy"))
        checkpoint_features_train[ckpt_id] = X_train
        checkpoint_features_test[ckpt_id] = X_test
        log(f"Checkpoint {ckpt_id}: X_train={X_train.shape}, X_test={X_test.shape}")

    n_ckpts = len(checkpoint_features_test)
    ids = sorted(checkpoint_features_test.keys())

    log(f"\n[Cross-Checkpoint KNN Divergence]")
    log(f"  {'Ckpt Pair':>10s} | {'KNN Acc':>8s} | {'||mean diff||':>12s} | {'CosSim':>8s}")
    log(f"  {'-'*52}")

    for i in range(n_ckpts):
        for j in range(i+1, n_ckpts):
            ci, cj = ids[i], ids[j]
            Xi = checkpoint_features_test[ci]
            Xj = checkpoint_features_test[cj]

            scaler = StandardScaler()
            Xi_s = scaler.fit_transform(Xi)
            Xj_s = scaler.transform(Xj)

            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(Xi_s, y_test)
            y_pred = knn.predict(Xj_s)
            acc = accuracy_score(y_test, y_pred)

            mean_diff = np.linalg.norm(Xi.mean(axis=0) - Xj.mean(axis=0))

            cos_sims = [np.dot(Xi[k], Xj[k]) / (np.linalg.norm(Xi[k]) * np.linalg.norm(Xj[k]) + 1e-8) for k in range(len(Xi))]
            avg_cos = np.mean(cos_sims)

            log(f"  {ci:>3d} vs {cj:<4d} | {acc:8.4f} | {mean_diff:12.4f} | {avg_cos:8.4f}")

    log(f"\n[PerClassKMeans Consolidation]")
    log(f"  {'Pair':>10s} | {'X-Only':>8s} | {'Y-Aware':>8s} | {'PerClass':>8s} | {'Freq':>8s} | {'PC-None':>8s}")
    log(f"  {'-'*65}")

    for i in range(n_ckpts):
        for j in range(i+1, n_ckpts):
            ci, cj = ids[i], ids[j]
            Xi_train = checkpoint_features_train[ci]
            Xj_test = checkpoint_features_test[cj]
            nj = len(y_test)
            n_A = int(sum(y_test))
            n_U = nj - n_A

            scaler = StandardScaler()
            Xi_s = scaler.fit_transform(Xi_train)
            Xj_s = scaler.transform(Xj_test)

            km = KMeans(n_clusters=2, random_state=42, n_init=10)
            km_labels = km.fit_predict(Xj_s)
            km_acc = max(np.mean(km_labels == y_test), np.mean(1 - km_labels == y_test))

            centroids_A = Xi_s[y_train == 1].mean(axis=0)
            centroids_U = Xi_s[y_train == 0].mean(axis=0)
            dist_A = np.linalg.norm(Xj_s - centroids_A, axis=1)
            dist_U = np.linalg.norm(Xj_s - centroids_U, axis=1)
            ya_labels = (dist_A < dist_U).astype(int)
            ya_acc = max(np.mean(ya_labels == y_test), np.mean(1 - ya_labels == y_test))

            pc_labels = np.zeros(nj, dtype=int)
            idx_A = np.where(y_train == 1)[0]
            idx_U = np.where(y_train == 0)[0]
            for k in range(nj):
                dA = np.linalg.norm(Xj_s[k] - Xi_s[idx_A], axis=1).min()
                dU = np.linalg.norm(Xj_s[k] - Xi_s[idx_U], axis=1).min()
                pc_labels[k] = 1 if dA < dU else 0
            pc_acc = max(np.mean(pc_labels == y_test), np.mean(1 - pc_labels == y_test))

            freq_acc = max(np.mean(np.random.RandomState(42).randint(0, 2, nj) == y_test), 0.5)
            no_mem = max(n_A, n_U) / nj

            log(f"  {ci:>3d}->{cj:<4d} | {km_acc:8.4f} | {ya_acc:8.4f} | {pc_acc:8.4f} | {freq_acc:8.4f} | {pc_acc-no_mem:+8.4f}")

    elapsed = time.time() - t0
    log(f"\n{'='*64}")
    log(f"Analysis Complete. ({elapsed:.1f}s)")
    log("=" * 64)

if __name__ == "__main__":
    main()