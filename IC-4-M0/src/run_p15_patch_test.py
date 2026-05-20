"""
IC-4 P1.5: Small-Data Robustness Patch Test.

For each failed P1 config (seed=2/layer=12, seed=0/layer=13):
  1. Merge training activations from two seeds to get 30A+30U (vs original 15A+15U)
  2. Merge training prefill features from two seeds for probe training
  3. Compute new steering vectors from larger activation set
  4. Re-run hard gate + random + shuffled controls
  5. Test whether larger construction sample eliminates the 'shuffled < real' artifact

Output: results_p15/patch_test_results.csv
"""

import argparse
import os
import sys
import time
import json
import random
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from src.run_m2 import load_config
from src.model_loader import load_model_and_tokenizer, get_model_layer_count
from src.evaluate import generate_answers, evaluate_outputs, run_generation_with_steering
from src.activation_collector import load_activations
from src.steering import get_all_vectors
from src.data_builder import load_jsonl

OUT_DIR = os.path.join(BASE_DIR, "results_p15")
os.makedirs(OUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUT_DIR, "patch_test_log.txt")


def _log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()


PATCH_CONFIGS = [
    {
        "name": "seed2_layer12_30A30U",
        "test_seed": 2, "test_layer": 12,
        "train_seeds": [0, 2], "activate_layer": 12,
        "eval_seed": 2, "base_seed": 2,
    },
    {
        "name": "seed0_layer13_30A30U",
        "test_seed": 0, "test_layer": 13,
        "train_seeds": [0, 1], "activate_layer": 13,
        "eval_seed": 0, "base_seed": 0,
    },
]


def collect_prefill_features(model, tokenizer, samples, layer_idx, representation="last_prompt_token"):
    device = next(model.parameters()).device
    X_list, y_list = [], []
    for sample in samples:
        context = sample.get("context", "")
        question = sample.get("question", "")
        label = sample.get("answerability", "?")
        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        hs = outputs.hidden_states[layer_idx + 1][0]
        if representation == "last_prompt_token":
            pooled = hs[-1, :].detach().cpu().float().numpy()
        elif representation == "mean_pooled":
            pooled = hs.mean(dim=0).detach().cpu().float().numpy()
        else:
            pooled = hs[-1, :].detach().cpu().float().numpy()
        X_list.append(pooled)
        y_list.append(1 if label == "answerable" else 0)
    return np.stack(X_list, axis=0), np.array(y_list, dtype=np.int32)


def train_probe(X, y, cv_folds=3):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(X_scaled, y)
    train_acc = accuracy_score(y, clf.predict(X_scaled))
    n_folds = min(cv_folds, len(y) // 2)
    cv_scores = []
    if n_folds >= 2 and len(np.unique(y)) >= 2:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy")
    auc_score = None
    if len(np.unique(y)) >= 2:
        try:
            auc_score = roc_auc_score(y, clf.predict_proba(X_scaled)[:, 1])
        except Exception:
            pass
    return {
        "classifier": clf, "scaler": scaler,
        "train_acc": float(train_acc),
        "cv_acc_mean": float(np.mean(cv_scores)) if len(cv_scores) > 0 else None,
        "auc": float(auc_score) if auc_score is not None else None,
        "n_samples": len(y), "n_pos": int(np.sum(y == 1)), "n_neg": int(np.sum(y == 0)),
    }


def find_transformer_layer(model, layer_idx):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers[layer_idx]
    raise ValueError("Cannot locate transformer layers.")


def generate_single_pass_hard_gate(model, tokenizer, test_data, steering_vector, layer_idx,
                                    alpha_max, probe_info, rep, threshold, gen_cfg):
    max_new = gen_cfg.get("max_new_tokens", 48)
    temperature = gen_cfg.get("temperature", 0.0)
    do_sample = gen_cfg.get("do_sample", False)
    device = next(model.parameters()).device
    vec_tensor = torch.from_numpy(steering_vector).to(device).float()
    layer_module = find_transformer_layer(model, layer_idx)
    results = []

    for sid, sample in enumerate(test_data):
        context = sample.get("context", "")
        question = sample.get("question", "")
        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        probe_score = [0.5]
        gate_val = [0.0]
        effective_alpha = [0.0]
        gate_decided = [False]

        def make_hook():
            def hook(module, fn_inputs, fn_outputs):
                if isinstance(fn_outputs, tuple):
                    h_full = fn_outputs[0]
                else:
                    h_full = fn_outputs
                if not gate_decided[0]:
                    h = h_full[0] if h_full.dim() == 3 else h_full
                    if rep == "last_prompt_token":
                        pooled = h[-1, :].detach().cpu().float().numpy()
                    elif rep == "mean_pooled":
                        pooled = h.mean(dim=0).detach().cpu().float().numpy()
                    else:
                        pooled = h[-1, :].detach().cpu().float().numpy()
                    X = probe_info["scaler"].transform(pooled.reshape(1, -1))
                    proba = probe_info["classifier"].predict_proba(X)[0, 1]
                    probe_score[0] = float(proba)
                    gate_decided[0] = True
                    if probe_score[0] >= threshold:
                        gate_val[0] = 0.0
                        effective_alpha[0] = 0.0
                    else:
                        gate_val[0] = 1.0
                        effective_alpha[0] = alpha_max
                if abs(effective_alpha[0]) > 0.001:
                    v = vec_tensor.to(dtype=h_full.dtype, device=h_full.device)
                    h_modified = h_full + effective_alpha[0] * v
                    if isinstance(fn_outputs, tuple):
                        return (h_modified,) + fn_outputs[1:]
                    else:
                        return h_modified
                else:
                    return None
            return hook

        handle = layer_module.register_forward_hook(make_hook())
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new, temperature=temperature,
                do_sample=do_sample, pad_token_id=tokenizer.eos_token_id,
            )
        handle.remove()
        answer = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        results.append({
            **sample, "sample_id": sid, "generated_output": answer,
            "mode": f"single_pass_hard_gate_a{alpha_max}",
            "alpha": alpha_max, "alpha_applied": effective_alpha[0],
            "probe_score": round(probe_score[0], 6), "gate": gate_val[0],
        })
    return results


def evaluate_add(results, seed, layer, mode, alpha, vector_type, all_rows):
    metrics = evaluate_outputs(results)
    metrics["seed"] = seed
    metrics["layer"] = layer
    metrics["mode"] = mode
    metrics["alpha"] = alpha
    metrics["vector_type"] = vector_type
    all_rows.append(metrics)
    return metrics


def main():
    _log("=" * 60)
    _log("IC-4 P1.5: SMALL-DATA ROBUSTNESS PATCH TEST")
    _log("=" * 60)
    _log(f"Output dir: {OUT_DIR}")
    _log(f"Configs: {[c['name'] for c in PATCH_CONFIGS]}")

    _log("\nLoading model (Qwen/Qwen2.5-0.5B-Instruct)...")
    model, tokenizer = load_model_and_tokenizer(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="cpu",
        torch_dtype="float32",
    )
    total_layers = get_model_layer_count(model)
    _log(f"  Total layers: {total_layers}")

    gen_cfg = {"max_new_tokens": 48, "temperature": 0.0, "do_sample": False}
    threshold = 0.5

    all_results = []

    for patch_cfg in PATCH_CONFIGS:
        name = patch_cfg["name"]
        test_layer = patch_cfg["test_layer"]
        act_layer = patch_cfg["activate_layer"]
        eval_seed = patch_cfg["eval_seed"]
        train_seeds = patch_cfg["train_seeds"]

        _log(f"\n{'='*40}")
        _log(f"PATCH TEST: {name}")
        _log(f"  Construction: train seeds={train_seeds}, layer={act_layer}")
        _log(f"  Evaluation: test seed={eval_seed}")
        _log(f"{'='*40}")
        t0 = time.time()

        pos_parts, neg_parts = [], []
        for ts in train_seeds:
            ap = os.path.join(BASE_DIR, "results_m3", f"activations_s{ts}_l{act_layer}.npz")
            a = load_activations(ap)
            pos_parts.append(a["positive"])
            neg_parts.append(a["negative"])
            _log(f"    s{ts}_l{act_layer}: {a['positive'].shape[0]} pairs")

        pos_all = np.concatenate(pos_parts, axis=0)
        neg_all = np.concatenate(neg_parts, axis=0)
        _log(f"  Merged activations: {pos_all.shape[0]} pos + {neg_all.shape[0]} neg, dim={pos_all.shape[1]}")

        all_vectors = get_all_vectors(pos_all, neg_all, pos_all.shape[1])
        steering_v = all_vectors["steering"]
        _log(f"  Vector norms: steer={np.linalg.norm(steering_v):.4f}, "
             f"random={np.linalg.norm(all_vectors['random']):.4f}, "
             f"shuffled={np.linalg.norm(all_vectors['shuffled']):.4f}")
        _log(f"  cos(steer, shuffled) = {np.dot(steering_v, all_vectors['shuffled']):.4f}")

        train_path = os.path.join(BASE_DIR, "data_m3", "train.jsonl")
        test_path = os.path.join(BASE_DIR, "data_m3", "test.jsonl")
        train_all = []
        for ts in train_seeds:
            tf = train_path.replace(".jsonl", f"_s{ts}.jsonl")
            train_all.extend(load_jsonl(tf))
        _log(f"  Merged training data: {len(train_all)} samples "
             f"({sum(1 for s in train_all if s.get('answerability')=='answerable')}A + "
             f"{sum(1 for s in train_all if s.get('answerability')=='unanswerable')}U)")

        test_f = test_path.replace(".jsonl", f"_s{eval_seed}.jsonl")
        test_data = load_jsonl(test_f)
        _log(f"  Test data: {len(test_data)} samples")

        _log(f"  Collecting prefill features for probe training...")
        X_train, y_train = collect_prefill_features(model, tokenizer, train_all, act_layer)
        _log(f"    X={X_train.shape}, y pos/neg={np.sum(y_train==1)}/{np.sum(y_train==0)}")

        probe_info = train_probe(X_train, y_train)
        _log(f"    Probe: train_acc={probe_info['train_acc']:.4f}, cv_acc={probe_info['cv_acc_mean']}, AUC={probe_info['auc']}")

        probe_cfg = {"representation": "last_prompt_token", "threshold": threshold}

        alpha = -1.0

        _log(f"  Running real gate...")
        real_res = generate_single_pass_hard_gate(
            model, tokenizer, test_data, steering_v, test_layer,
            alpha, probe_info, "last_prompt_token", threshold, gen_cfg)
        m = evaluate_add(real_res, eval_seed, test_layer,
                         f"patch_{name}_real", alpha, "steering", all_results)
        _log(f"    real_gate: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f}")

        _log(f"  Running random gate...")
        rnd_res = generate_single_pass_hard_gate(
            model, tokenizer, test_data, all_vectors["random"], test_layer,
            alpha, probe_info, "last_prompt_token", threshold, gen_cfg)
        m = evaluate_add(rnd_res, eval_seed, test_layer,
                         f"patch_{name}_random", alpha, "random", all_results)
        _log(f"    random: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f}")

        _log(f"  Running shuffled gate...")
        shf_res = generate_single_pass_hard_gate(
            model, tokenizer, test_data, all_vectors["shuffled"], test_layer,
            alpha, probe_info, "last_prompt_token", threshold, gen_cfg)
        m = evaluate_add(shf_res, eval_seed, test_layer,
                         f"patch_{name}_shuffled", alpha, "shuffled", all_results)
        _log(f"    shuffled: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f}")

        elapsed = time.time() - t0
        _log(f"  Patch config done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    _log(f"\n{'='*60}")
    _log("PATCH TEST RESULTS SUMMARY")
    _log(f"{'='*60}")
    for r in all_results:
        _log(f"  {r['mode']:<45} H={r['hallucination_rate']:.3f} C={r['correct_answer_rate']:.3f} vec={r.get('vector_type','?')}")

    df = pd.DataFrame(all_results)
    csv_path = os.path.join(OUT_DIR, "patch_test_results.csv")
    df.to_csv(csv_path, index=False)
    _log(f"\nResults saved to {csv_path}")

    _log(f"\nDone. {'='*60}")


if __name__ == "__main__":
    main()