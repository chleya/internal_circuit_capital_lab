"""
P3 A-1: Position-Augmented Gate Probe.
========================================
Trains a probe with position augmentation (early/mid/late variants)
to make gate decisions more robust to content position in the prompt.

Hypothesis: Adding position variants to training data makes the probe
learn position-invariant features, reducing PSI from 0.0084.

Design:
  1. Load M3-v6 standard training data (30A+30U, seed=0, layer=12)
  2. Load position sensitivity training data (same content at early/mid/late)
  3. Train baseline probe (standard position only)
  4. Train augmented probe (standard + early + mid + late, 4x data)
  5. Evaluate both on position test set
  6. Compare PSI: mean |probe_score(early) - probe_score(late)| per sample

Usage:
  cd F:\\internal_circuit_capital_lab\\IC-4-M0
  python src/run_a1_position_augmented_probe.py
"""

import os, sys, time, json, random
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.run_m2 import load_config
from src.run_m3_v6 import _collect_prefill_features, _train_probe

RESULTS_DIR = "results_a1_position_probe"
os.makedirs(RESULTS_DIR, exist_ok=True)

LOG_PATH = os.path.join(RESULTS_DIR, "run_log.txt")

def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()

def load_position_variants(data_dir, seed=0):
    variants = {"early": [], "mid": [], "late": [], "all": []}
    for pos in ["early", "mid", "late", "all"]:
        train_path = os.path.join(data_dir, f"train_{pos}_s{seed}.jsonl")
        test_path = os.path.join(data_dir, f"test_{pos}_s{seed}.jsonl")
        if os.path.exists(train_path):
            from src.data_builder import load_jsonl
            variants[pos] = {
                "train": load_jsonl(train_path),
                "test": load_jsonl(test_path) if os.path.exists(test_path) else [],
            }
            na = sum(1 for s in variants[pos]["train"] if s.get("answerability") == "answerable")
            log(f"  {pos}: train {na}A+{len(variants[pos]['train'])-na}U, test {len(variants[pos]['test'])}")
    return variants

def load_m3_data(seed=0, layer=12):
    config = load_config("configs/config_m3_v6.yaml")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = config.get("data_dir", "data_m3")
    train_path = os.path.join(base_dir, data_dir, "train.jsonl")
    test_path = os.path.join(base_dir, data_dir, "test.jsonl")
    from src.data_builder import load_jsonl
    train_final = train_path.replace(".jsonl", f"_s{seed}.jsonl")
    test_final = test_path.replace(".jsonl", f"_s{seed}.jsonl")
    train = load_jsonl(train_final)
    test = load_jsonl(test_final)
    return train, test

def compute_psi(probe_info, pos_variants_test):
    scaler, clf = probe_info["scaler"], probe_info["classifier"]
    psi_scores = []
    for pos in ["early", "mid", "late"]:
        scores = []
        for s in pos_variants_test[pos]["test"]:
            content_key = s.get("variant_group", s.get("context", "")[:50])
            X = probe_info.get(f"features_{pos}", None)
            if X is not None:
                X_scaled = scaler.transform(X)
                proba = clf.predict_proba(X_scaled)[:, 1]
                scores.extend(proba.tolist())
        if scores:
            psi_scores.append(np.mean(scores))
    return psi_scores

def evaluate_psi_from_features(probe_info, features_dict):
    scaler, clf = probe_info["scaler"], probe_info["classifier"]
    scores_by_pos = {}
    for pos in ["early", "mid", "late"]:
        if pos in features_dict and features_dict[pos] is not None:
            X = features_dict[pos]
            X_scaled = scaler.transform(X)
            proba = clf.predict_proba(X_scaled)[:, 1]
            scores_by_pos[pos] = proba
    if len(scores_by_pos) >= 2:
        positions = list(scores_by_pos.keys())
        mean_abs_deltas = []
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                delta = np.abs(scores_by_pos[positions[i]] - scores_by_pos[positions[j]]).mean()
                mean_abs_deltas.append(delta)
        psi = np.mean(mean_abs_deltas)
        return psi, scores_by_pos
    return None, scores_by_pos

def main():
    log("=" * 64)
    log("P3 A-1: Position-Augmented Gate Probe")
    log("=" * 64)
    t0 = time.time()

    log("\n[Step 1] Loading model...")
    from src.model_loader import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
    device = next(model.parameters()).device
    log(f"  Model on {device}.")

    SEED = 0
    LAYER = 12
    REPR = "last_prompt_token"

    log(f"\n[Step 2] Loading M3-v6 standard data (seed={SEED})...")
    train_m3, test_m3 = load_m3_data(SEED, LAYER)
    log(f"  Standard train: {len(train_m3)} (answerable={sum(1 for s in train_m3 if s.get('answerability')=='answerable')})")
    log(f"  Standard test:  {len(test_m3)} (answerable={sum(1 for s in test_m3 if s.get('answerability')=='answerable')})")

    log(f"\n[Step 3] Loading position sensitivity data...")
    from src.data_builder import load_jsonl
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pos_dir = os.path.join(base_dir, "data_position_sensitivity")

    pos_variants = {}
    for pos in ["early", "mid", "late"]:
        train_path = os.path.join(pos_dir, "s0", f"train_{pos}_s{SEED}.jsonl")
        test_path = os.path.join(pos_dir, "s0", f"test_{pos}_s{SEED}.jsonl")
        if os.path.exists(train_path) and os.path.exists(test_path):
            pos_variants[pos] = {
                "train": load_jsonl(train_path),
                "test": load_jsonl(test_path),
            }
            na_train = sum(1 for s in pos_variants[pos]["train"] if s.get("answerability") == "answerable")
            na_test = sum(1 for s in pos_variants[pos]["test"] if s.get("answerability") == "answerable")
            log(f"  {pos}: train {na_train}A+{len(pos_variants[pos]['train'])-na_train}U, test {na_test}A+{len(pos_variants[pos]['test'])-na_test}U")

    if len(pos_variants) < 3:
        log("\n[ABORT] Position variation data incomplete. Need early/mid/late all present.")
        return

    log(f"\n[Step 4] Extracting features (baseline: standard position only)...")
    X_train_base, y_train_base = _collect_prefill_features(model, tokenizer, train_m3, LAYER, REPR)
    X_test_base, y_test_base = _collect_prefill_features(model, tokenizer, test_m3, LAYER, REPR)
    log(f"  X_train: {X_train_base.shape}, X_test: {X_test_base.shape}")

    log(f"\n[Step 5] Extracting features (position variants)...")
    X_pos = {}
    y_pos = {}
    for pos in ["early", "mid", "late"]:
        log(f"  Extracting {pos}...")
        Xp, yp = _collect_prefill_features(model, tokenizer, pos_variants[pos]["train"], LAYER, REPR)
        X_pos[pos] = Xp
        y_pos[pos] = yp
        log(f"    {pos}: {Xp.shape}")

    log(f"\n[Step 6] Training BASELINE probe (standard position only)...")
    probe_base = _train_probe(X_train_base, y_train_base)
    log(f"  Train acc: {probe_base['train_acc']:.4f}, CV: {probe_base['cv_acc_mean']}, AUC: {probe_base['auc']}")

    log(f"\n[Step 7] Training AUGMENTED probe (standard + early + mid + late)...")
    X_aug = np.vstack([X_train_base] + [X_pos[p] for p in ["early", "mid", "late"]])
    y_aug = np.concatenate([y_train_base] + [y_pos[p] for p in ["early", "mid", "late"]])
    log(f"  Augmented X: {X_aug.shape}, y: {y_aug.shape}")
    probe_aug = _train_probe(X_aug, y_aug)
    log(f"  Train acc: {probe_aug['train_acc']:.4f}, CV: {probe_aug['cv_acc_mean']}, AUC: {probe_aug['auc']}")

    log(f"\n[Step 8] Evaluating PSI on position test sets...")
    X_pos_test = {}
    for pos in ["early", "mid", "late"]:
        Xpt, ypt = _collect_prefill_features(model, tokenizer, pos_variants[pos]["test"], LAYER, REPR)
        X_pos_test[pos] = Xpt
        log(f"  {pos} test: {Xpt.shape}")

    psi_base, scores_base = evaluate_psi_from_features(probe_base, X_pos_test)
    psi_aug, scores_aug = evaluate_psi_from_features(probe_aug, X_pos_test)

    log(f"\n  BASE PSI: {psi_base:.6f}" if psi_base else "\n  BASE PSI: N/A")
    log(f"  AUG  PSI: {psi_aug:.6f}" if psi_aug else "  AUG  PSI: N/A")

    if psi_base and psi_aug:
        improvement = (psi_base - psi_aug) / psi_base * 100
        log(f"  Improvement: {improvement:+.1f}%")

    log(f"\n[Step 9] Probe score distributions by position:")
    for pos in ["early", "mid", "late"]:
        if pos in scores_base and pos in scores_aug:
            log(f"  {pos:6s}  BASE mean={scores_base[pos].mean():.4f} std={scores_base[pos].std():.4f}  |  AUG mean={scores_aug[pos].mean():.4f} std={scores_aug[pos].std():.4f}")

    log(f"\n[Step 10] Evaluating behavior impact (probe-only, no generation)...")
    y_pred_base = probe_base["classifier"].predict(probe_base["scaler"].transform(X_test_base))
    y_pred_aug = probe_aug["classifier"].predict(probe_aug["scaler"].transform(X_test_base))
    acc_base = accuracy_score(y_test_base, y_pred_base)
    acc_aug = accuracy_score(y_test_base, y_pred_aug)
    log(f"  Standard test accuracy: BASE={acc_base:.4f}, AUG={acc_aug:.4f} (Δ={acc_aug-acc_base:+.4f})")

    elapsed = time.time() - t0
    log(f"\n{'='*64}")
    log(f"P3 A-1 Complete. ({elapsed:.0f}s)")
    verdict = ""
    if psi_base and psi_aug:
        if psi_aug < psi_base:
            verdict = f"PASS — PSI reduced from {psi_base:.6f} to {psi_aug:.6f} (Δ={psi_base-psi_aug:.6f})"
        else:
            verdict = f"NO EFFECT — PSI not reduced (base={psi_base:.6f}, aug={psi_aug:.6f})"
    log(f"VERDICT: {verdict}")
    log("=" * 64)

    results = {
        "psi_base": float(psi_base) if psi_base else None,
        "psi_aug": float(psi_aug) if psi_aug else None,
        "acc_base": float(acc_base),
        "acc_aug": float(acc_aug),
        "n_base_train": int(len(y_train_base)),
        "n_aug_train": int(len(y_aug)),
        "time_s": round(elapsed, 1),
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {RESULTS_DIR}/results.json")

if __name__ == "__main__":
    main()