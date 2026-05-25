"""
Phase 8-B: Behavior-Level Position Invariance Test.
=====================================================
Simplified: NO steering hook (avoids known model.generate() hook bug).
Instead, measures probe score → behavior correlation across positions.

Core test: does position-augmented probe make the same prediction
for the same content at different positions → leading to more
consistent behavior across positions?

Design:
  1. Extract features + train probes (standard + augmented)
  2. Run model.generate() WITHOUT steering on position test samples
  3. Compare probe scores across positions per content
  4. Measure: score consistency → behavior consistency

Key metric: |C(early) - C(late)| for augmented probe decisions
vs baseline. Smaller = better position invariance.

Usage:
  cd F:\internal_circuit_capital_lab\IC-4-M0
  python src/run_a2_behavior_position_invariant.py --n 20
"""

import argparse
import os, sys, time, json, pickle
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_loader import load_model_and_tokenizer
from src.data_builder import load_jsonl
from src.evaluate import evaluate_outputs
from src.run_m3_v6 import _collect_prefill_features, _train_probe
from src.run_m2 import load_config

RESULTS_DIR = "results_a2_behavior_position_inv"
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, "run_log.txt")

def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Samples per position")
    args = parser.parse_args()
    n_per_pos = args.n

    log("=" * 64)
    log("Phase 8-B: Behavior-Level Position Invariance Test")
    log("=" * 64)
    t0 = time.time()

    log("\n[Step 1] Loading model on CPU...")
    model, tokenizer = load_model_and_tokenizer(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="cpu",
        torch_dtype="float32",
    )
    device = next(model.parameters()).device
    log(f"  Model on {device}.")

    SEED = 0
    LAYER = 12
    REPR = "last_prompt_token"

    log(f"\n[Step 2] Loading M3 standard data (seed={SEED})...")
    config = load_config("configs/config_m3_v6.yaml")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = config.get("data_dir", "data_m3")
    train_path = os.path.join(base_dir, data_dir, f"train_s{SEED}.jsonl")
    train_m3 = load_jsonl(train_path)
    log(f"  Train: {len(train_m3)}")

    log(f"\n[Step 3] Loading position test data (n={n_per_pos}/pos)...")
    pos_dir = os.path.join(base_dir, "data_position_sensitivity", "s0")
    pos_test = {}
    for pos in ["early", "mid", "late"]:
        test_path = os.path.join(pos_dir, f"test_{pos}_s{SEED}.jsonl")
        if os.path.exists(test_path):
            pos_test[pos] = load_jsonl(test_path)[:n_per_pos]
            na = sum(1 for s in pos_test[pos] if s.get("answerability") == "answerable")
            log(f"  {pos}: {na}A+{len(pos_test[pos])-na}U")

    if len(pos_test) < 3:
        log("[ABORT] Need early/mid/late all present.")
        return

    log(f"\n[Step 4] Extracting features (standard train)...")
    X_train, y_train = _collect_prefill_features(model, tokenizer, train_m3, LAYER, REPR)
    log(f"  X_train: {X_train.shape}")

    log(f"\n[Step 5] Extracting features (position train variants)...")
    X_pos_train = {}
    for pos in ["early", "mid", "late"]:
        train_path = os.path.join(pos_dir, f"train_{pos}_s{SEED}.jsonl")
        if os.path.exists(train_path):
            train_samples = load_jsonl(train_path)
            Xp, _ = _collect_prefill_features(model, tokenizer, train_samples, LAYER, REPR)
            X_pos_train[pos] = Xp
            log(f"  {pos} train: {Xp.shape}")

    log(f"\n[Step 6] Training probes...")
    probe_base = _train_probe(X_train, y_train)
    log(f"  BASE: train_acc={probe_base['train_acc']:.4f}, AUC={probe_base['auc']}")

    X_aug = np.vstack([X_train] + [X_pos_train[p] for p in ["early", "mid", "late"]])
    y_aug = np.concatenate([y_train] + [np.array([1 if s.get("answerability") == "answerable" else 0 for s in load_jsonl(os.path.join(pos_dir, f"train_{p}_s{SEED}.jsonl"))]) for p in ["early", "mid", "late"]])
    probe_aug = _train_probe(X_aug, y_aug)
    log(f"  AUG:  train_acc={probe_aug['train_acc']:.4f}, AUC={probe_aug['auc']}")

    log(f"\n[Step 7] Extracting features (position test)...")
    X_pos_test = {}
    for pos in ["early", "mid", "late"]:
        Xpt, _ = _collect_prefill_features(model, tokenizer, pos_test[pos], LAYER, REPR)
        X_pos_test[pos] = Xpt

    log(f"\n[Step 8] Computing probe scores per position...")
    scores_base = {}
    scores_aug = {}
    for pos in ["early", "mid", "late"]:
        X = X_pos_test[pos]
        scaler_b, clf_b = probe_base["scaler"], probe_base["classifier"]
        scaler_a, clf_a = probe_aug["scaler"], probe_aug["classifier"]
        scores_base[pos] = clf_b.predict_proba(scaler_b.transform(X))[:, 1]
        scores_aug[pos] = clf_a.predict_proba(scaler_a.transform(X))[:, 1]

    psi_base = 0
    psi_aug = 0
    n_pairs = 0
    positions = ["early", "mid", "late"]
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            psi_base += np.abs(scores_base[positions[i]] - scores_base[positions[j]]).mean()
            psi_aug += np.abs(scores_aug[positions[i]] - scores_aug[positions[j]]).mean()
            n_pairs += 1
    psi_base /= n_pairs
    psi_aug /= n_pairs
    log(f"  BASE PSI: {psi_base:.6f}")
    log(f"  AUG  PSI: {psi_aug:.6f}")
    log(f"  Improvement: {(psi_base - psi_aug)/psi_base*100:+.1f}%")

    log(f"\n[Step 9] Running model.generate() WITHOUT steering (behavior baseline)...")
    max_new_tokens = 48
    behavior_results = []
    t_gen = time.time()
    for pos in ["early", "mid", "late"]:
        log(f"  Generating [{pos}]...")
        for sid, sample in enumerate(pos_test[pos]):
            context = sample.get("context", "")
            question = sample.get("question", "")
            prompt = f"{context}\n\nQuestion: {question}\nAnswer:"

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated_ids = outputs[0][input_len:]
            answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            base_score = float(scores_base[pos][sid])
            aug_score = float(scores_aug[pos][sid])

            behavior_results.append({
                "position": pos,
                "sample_id": sid,
                "answerability": sample.get("answerability", "?"),
                "generated_output": answer,
                "probe_base_score": base_score,
                "probe_aug_score": aug_score,
            })
    gen_elapsed = time.time() - t_gen
    log(f"  {len(behavior_results)} generations in {gen_elapsed:.0f}s")

    log(f"\n[Step 10] Position-to-Behavior analysis...")
    log(f"\n  {'Position':8s} | {'H':>6s} | {'C':>6s} | {'Base Score Mean':>15s} | {'Aug Score Mean':>15s}")
    log(f"  {'-'*65}")

    pos_metrics = {}
    for pos in ["early", "mid", "late"]:
        pos_data = [r for r in behavior_results if r["position"] == pos]
        eval_in = [{"generated_output": r["generated_output"], "answerability": r["answerability"]} for r in pos_data]
        m = evaluate_outputs(eval_in)
        base_mean = np.mean([r["probe_base_score"] for r in pos_data])
        aug_mean = np.mean([r["probe_aug_score"] for r in pos_data])
        log(f"  {pos:8s} | {m['hallucination_rate']:.4f} | {m['correct_answer_rate']:.4f} | {base_mean:15.4f} | {aug_mean:15.4f}")
        pos_metrics[pos] = {
            "H": m["hallucination_rate"],
            "C": m["correct_answer_rate"],
            "base_score_mean": base_mean,
            "aug_score_mean": aug_mean,
        }

    h_range = max(m["H"] for m in pos_metrics.values()) - min(m["H"] for m in pos_metrics.values())
    c_range = max(m["C"] for m in pos_metrics.values()) - min(m["C"] for m in pos_metrics.values())
    base_score_range = max(m["base_score_mean"] for m in pos_metrics.values()) - min(m["base_score_mean"] for m in pos_metrics.values())
    aug_score_range = max(m["aug_score_mean"] for m in pos_metrics.values()) - min(m["aug_score_mean"] for m in pos_metrics.values())

    log(f"\n  ΔC (behavior):        {c_range:.4f}")
    log(f"  ΔH (behavior):        {h_range:.4f}")
    log(f"  Δ base_score (probe): {base_score_range:.4f}")
    log(f"  Δ aug_score (probe):  {aug_score_range:.4f}")

    log(f"\n[Step 11] Probe-Decision Consistency Analysis...")
    threshold = 0.5
    for pos in ["early", "mid", "late"]:
        pos_data = [r for r in behavior_results if r["position"] == pos]
        base_gated = sum(1 for r in pos_data if r["probe_base_score"] > threshold)
        aug_gated = sum(1 for r in pos_data if r["probe_aug_score"] > threshold)
        log(f"  {pos}: base gate={base_gated}/{len(pos_data)}, aug gate={aug_gated}/{len(pos_data)}")

    elapsed = time.time() - t0
    log(f"\n{'='*64}")
    log(f"Phase 8-B Complete. ({elapsed:.0f}s, {elapsed/60:.1f} min)")
    log(f"  PSI improvement: {(psi_base-psi_aug)/psi_base*100:+.1f}%")
    log(f"  Behavior ΔC: {c_range:.4f}")
    log(f"  Behavior ΔH: {h_range:.4f}")
    log("=" * 64)

    import pandas as pd
    pd.DataFrame(behavior_results).to_csv(os.path.join(RESULTS_DIR, "behavior_raw.csv"), index=False)

    summary = {
        "psi_base": float(psi_base),
        "psi_aug": float(psi_aug),
        "psi_improvement_pct": round((psi_base - psi_aug) / psi_base * 100, 1),
        "n_per_pos": n_per_pos,
        "c_range": round(c_range, 4),
        "h_range": round(h_range, 4),
        "base_score_range": round(base_score_range, 4),
        "aug_score_range": round(aug_score_range, 4),
        "time_s": round(elapsed, 1),
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nResults saved to {RESULTS_DIR}/")

if __name__ == "__main__":
    main()