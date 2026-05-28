"""
S3a: Sycophancy Detection Probe
Train a linear classifier on base model hidden states to detect syc-triggering prompts.
This probe will serve as the routing gate for conditional anti-syc LoRA application.
"""

import os, time, json, random, gc
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYC_DATA_PATH = os.path.join(BASE_DIR, "results_p0_sycophancy", "sycophancy_contrast_data.json")
RESULTS_DIR = os.path.join(BASE_DIR, "results_s3a_syc_probe")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_SYC, TRAIN_NON = 20, 20
TARGET_LAYER, MAX_LEN = 10, 256


def _log(msg, path=None):
    print(msg, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as f: f.write(msg + "\n")


def get_hidden_states(model, tokenizer, samples, device):
    hs, labels = [], []
    model.eval()
    with torch.no_grad():
        for s in samples:
            inp = tokenizer(s["prompt"], return_tensors="pt", truncation=True, max_length=MAX_LEN)
            inp = {k: v.to(device) for k, v in inp.items()}
            o = model(**inp, output_hidden_states=True)
            h = o.hidden_states[TARGET_LAYER + 1][0, -1, :].cpu().numpy()
            hs.append(h)
            labels.append(1.0 if s["group"] == "sycophantic" else 0.0)
    return np.array(hs), np.array(labels)


def logistic_regression(X_train, y_train, X_test, y_test, l2=1e-3):
    n, d = X_train.shape
    p0 = y_train.mean()
    w = np.random.randn(d) * 0.01
    b = np.log(p0 / (1 - p0 + 1e-8))
    lr = 0.1
    for it in range(5000):
        z = X_train @ w + b
        z = np.clip(z, -50, 50)
        p = 1 / (1 + np.exp(-z))
        dw = (X_train.T @ (p - y_train)) / n + 2 * l2 * w
        db = (p - y_train).mean()
        w -= lr * dw
        b -= lr * db
        if it % 1000 == 0:
            loss = -(y_train * np.log(p + 1e-8) + (1 - y_train) * np.log(1 - p + 1e-8)).mean() + l2 * (w ** 2).sum()
    z_test = np.clip(X_test @ w + b, -50, 50)
    p_test = 1 / (1 + np.exp(-z_test))
    preds = (p_test >= 0.5).astype(float)
    acc = (preds == y_test).mean()
    tp = ((preds == 1) & (y_test == 1)).sum()
    fp = ((preds == 1) & (y_test == 0)).sum()
    fn = ((preds == 0) & (y_test == 1)).sum()
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return w, b, acc, prec, rec, f1, p_test, y_test


def v_syc_baseline(X_train, y_train, X_test, y_test):
    syc_hs = X_train[y_train == 1]; non_hs = X_train[y_train == 0]
    v_syc = syc_hs.mean(axis=0) - non_hs.mean(axis=0)
    v_syc = v_syc / (np.linalg.norm(v_syc) + 1e-8)
    train_proj = X_train @ v_syc
    test_proj = X_test @ v_syc
    best_acc, best_thr = 0, 0
    for thr in np.linspace(test_proj.min(), test_proj.max(), 200):
        preds = (test_proj >= thr).astype(float)
        acc = (preds == y_test).mean()
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    preds = (test_proj >= best_thr).astype(float)
    tp = ((preds == 1) & (y_test == 1)).sum()
    fp = ((preds == 1) & (y_test == 0)).sum()
    fn = ((preds == 0) & (y_test == 1)).sum()
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return v_syc, best_thr, best_acc, prec, rec, f1, test_proj, y_test


def main():
    log_path = os.path.join(RESULTS_DIR, "run_log.txt")
    _log(f"S3a: Sycophancy Detection Probe | layer={TARGET_LAYER} | {TRAIN_SYC}S+{TRAIN_NON}N train | {time.strftime('%H:%M:%S')}", log_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"Device: {device}", log_path)

    with open(SYC_DATA_PATH, "r", encoding="utf-8") as f: data = json.load(f)
    syc = [s for s in data if s["group"] == "sycophantic"]
    non = [s for s in data if s["group"] == "non_sycophantic"]
    random.seed(42); np.random.seed(42)
    random.shuffle(syc); random.shuffle(non)
    train_s = syc[:TRAIN_SYC]; train_n = non[:TRAIN_NON]
    test_samples = syc[TRAIN_SYC:] + non[TRAIN_NON:]
    _log(f"Train: {len(train_s)}S+{len(train_n)}N | Test: {len(test_samples)} ({sum(1 for s in test_samples if s['group']=='sycophantic')}S+{sum(1 for s in test_samples if s['group']=='non_sycophantic')}N)", log_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    t0 = time.time()
    _log("Loading base model...", log_path)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32, trust_remote_code=True, local_files_only=True).to(device)
    _log(f"Model loaded ({time.time()-t0:.0f}s)", log_path)

    _log("Extracting hidden states...", log_path)
    X_train, y_train = get_hidden_states(model, tokenizer, train_s + train_n, device)
    X_test, y_test = get_hidden_states(model, tokenizer, test_samples, device)
    _log(f"  Train: {X_train.shape} (y=1: {y_train.sum():.0f}/{len(y_train)})", log_path)
    _log(f"  Test:  {X_test.shape} (y=1: {y_test.sum():.0f}/{len(y_test)})", log_path)

    _log("\n--- Logistic Regression Probe ---", log_path)
    w, b, acc, prec, rec, f1, p_test, y_test_labels = logistic_regression(X_train, y_train, X_test, y_test)
    _log(f"  Test accuracy: {acc:.4f}", log_path)
    _log(f"  Precision: {prec:.4f}", log_path)
    _log(f"  Recall: {rec:.4f}", log_path)
    _log(f"  F1: {f1:.4f}", log_path)
    _log(f"  ||w||: {np.linalg.norm(w):.4f}", log_path)

    _log("\n--- v_syc Projection Baseline ---", log_path)
    v_syc, best_thr, v_acc, v_prec, v_rec, v_f1, v_proj, _ = v_syc_baseline(X_train, y_train, X_test, y_test)
    _log(f"  Best threshold: {best_thr:.6f}", log_path)
    _log(f"  Test accuracy: {v_acc:.4f}", log_path)
    _log(f"  Precision: {v_prec:.4f}", log_path)
    _log(f"  Recall: {v_rec:.4f}", log_path)
    _log(f"  F1: {v_f1:.4f}", log_path)

    cos_sim = np.dot(w, v_syc) / (np.linalg.norm(w) + 1e-8)
    _log(f"\n  cos_sim(w_probe, v_syc): {cos_sim:.4f}", log_path)

    _log("\n--- Per-Sample Test Predictions ---", log_path)
    for i, s in enumerate(test_samples):
        real = "SYC" if s["group"] == "sycophantic" else "NON"
        lr_pred = "SYC" if p_test[i] >= 0.5 else "NON"
        v_pred = "SYC" if v_proj[i] >= best_thr else "NON"
        lr_match = "✓" if lr_pred == real else "✗"
        v_match = "✓" if v_pred == real else "✗"
        _log(f"  [{i}] true={real} | LR={lr_pred}(p={p_test[i]:.3f}) {lr_match} | v_proj={v_pred}({v_proj[i]:.4f}) {v_match}", log_path)

    results = {
        "experiment": "S3a_Sycophancy_Detection_Probe",
        "layer": TARGET_LAYER,
        "train": f"{TRAIN_SYC}S+{TRAIN_NON}N",
        "test_size": len(test_samples),
        "lr_probe": {
            "test_accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "weight_norm": round(float(np.linalg.norm(w)), 4),
            "bias": round(float(b), 6),
            "weight": w.tolist()
        },
        "v_syc_baseline": {
            "best_threshold": round(float(best_thr), 6),
            "test_accuracy": round(float(v_acc), 4),
            "precision": round(float(v_prec), 4),
            "recall": round(float(v_rec), 4),
            "f1": round(float(v_f1), 4),
            "cos_sim_with_lr_weight": round(float(cos_sim), 4)
        }
    }
    with open(os.path.join(RESULTS_DIR, "probe_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    _log(f"\nTotal: {elapsed:.0f}s ({elapsed/60:.1f} min)", log_path)
    _log("Done.", log_path)


if __name__ == "__main__":
    main()