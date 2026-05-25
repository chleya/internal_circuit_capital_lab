"""IC-4 A2 Upgrade: 30A+30U soft/hard/open-loop gating comparison.

Re-runs the M4 A2 audit under 30A+30U construction standard, comparing three
gating strategies:

  - Open-loop: alpha * v applied to ALL samples, no probe
  - Soft gate:  alpha * (1 - probe_score) * v  — proportional to probe uncertainty
  - Hard gate:  alpha * v if probe_score < threshold else 0

Configs (same as re-validation):
  - seed0_layer12_ref, seed1_layer12, seed0_layer11

Expected: H(hard) <= H(soft) < H(open_loop) if gating is beneficial.
"""

import os, sys, time, json, numpy as np, pandas as pd, torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from src.model_loader import load_model_and_tokenizer, get_model_layer_count
from src.evaluate import evaluate_outputs
from src.activation_collector import load_activations
from src.steering import get_all_vectors
from src.data_builder import load_jsonl

OUT_DIR = os.path.join(BASE_DIR, "results_a2_upgrade")
os.makedirs(OUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUT_DIR, "log.txt")

GEN_CFG = {"max_new_tokens": 48, "temperature": 0.0, "do_sample": False}
THRESHOLD = 0.5
ALPHA = -1.0

CONFIGS = [
    {"name": "seed0_layer12_ref", "layer": 12, "train_seeds": [0, 1], "eval_seed": 0},
    {"name": "seed1_layer12",      "layer": 12, "train_seeds": [1, 0], "eval_seed": 1},
    {"name": "seed0_layer11",      "layer": 11, "train_seeds": [0, 1], "eval_seed": 0},
]


def _log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()


def find_transformer_layer(model, layer_idx):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers[layer_idx]
    raise ValueError("Cannot locate transformer layers.")


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score


def collect_prefill_features(model, tokenizer, samples, layer_idx):
    device = next(model.parameters()).device
    X_list, y_list = [], []
    for sample in samples:
        context = sample.get("context", "")
        question = sample.get("question", "")
        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        hs = outputs.hidden_states[layer_idx + 1][0]
        pooled = hs[-1, :].detach().cpu().float().numpy()
        X_list.append(pooled)
        y_list.append(1 if sample.get("answerability") == "answerable" else 0)
    return np.stack(X_list, axis=0), np.array(y_list, dtype=np.int32)


def train_probe(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(X_scaled, y)
    train_acc = accuracy_score(y, clf.predict(X_scaled))
    cv_scores = []
    if len(np.unique(y)) >= 2 and len(y) >= 6:
        cv = StratifiedKFold(n_splits=min(3, len(y)//2), shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy")
    return {"classifier": clf, "scaler": scaler, "train_acc": float(train_acc),
            "cv_acc": float(np.mean(cv_scores)) if len(cv_scores) > 0 else None,
            "auc": float(roc_auc_score(y, clf.predict_proba(X_scaled)[:, 1])) if len(np.unique(y))>=2 else None}


def run_open_loop(model, tokenizer, test_data, steering_v, layer_idx, alpha, gen_cfg, mode_label):
    """Apply steering to ALL samples — no gating."""
    max_new = gen_cfg.get("max_new_tokens", 48)
    temperature = gen_cfg.get("temperature", 0.0)
    do_sample = gen_cfg.get("do_sample", False)
    device = next(model.parameters()).device
    vec_tensor = torch.from_numpy(steering_v).to(device).float()
    layer_module = find_transformer_layer(model, layer_idx)
    results = []

    for sid, sample in enumerate(test_data):
        context = sample.get("context", "")
        question = sample.get("question", "")
        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        def make_hook():
            def hook(module, fn_inputs, fn_outputs):
                if isinstance(fn_outputs, tuple):
                    h_full = fn_outputs[0]
                else:
                    h_full = fn_outputs
                v = vec_tensor.to(dtype=h_full.dtype, device=h_full.device)
                h_modified = h_full + alpha * v
                if isinstance(fn_outputs, tuple):
                    return (h_modified,) + fn_outputs[1:]
                else:
                    return h_modified
            return hook

        handle = layer_module.register_forward_hook(make_hook())
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new, temperature=temperature,
                                     do_sample=do_sample, pad_token_id=tokenizer.eos_token_id)
        handle.remove()
        answer = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        results.append({**sample, "sample_id": sid, "generated_output": answer,
                         "mode": mode_label, "alpha": alpha, "alpha_applied": alpha,
                         "probe_score": 0.5, "gate": 1.0})
    return results


def run_soft_gate(model, tokenizer, test_data, steering_v, layer_idx, alpha,
                   probe_info, threshold, gen_cfg, mode_label):
    """Apply steering proportional to probe uncertainty."""
    max_new = gen_cfg.get("max_new_tokens", 48)
    temperature = gen_cfg.get("temperature", 0.0)
    do_sample = gen_cfg.get("do_sample", False)
    device = next(model.parameters()).device
    vec_tensor = torch.from_numpy(steering_v).to(device).float()
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
                    pooled = h[-1, :].detach().cpu().float().numpy()
                    X_s = probe_info["scaler"].transform(pooled.reshape(1, -1))
                    proba = probe_info["classifier"].predict_proba(X_s)[0, 1]
                    probe_score[0] = float(proba)
                    gate_decided[0] = True
                    soft_factor = max(0.0, 1.0 - proba)
                    effective_alpha[0] = alpha * soft_factor
                if abs(effective_alpha[0]) > 0.001:
                    v = vec_tensor.to(dtype=h_full.dtype, device=h_full.device)
                    h_modified = h_full + effective_alpha[0] * v
                    if isinstance(fn_outputs, tuple):
                        return (h_modified,) + fn_outputs[1:]
                    else:
                        return h_modified
                return None
            return hook

        handle = layer_module.register_forward_hook(make_hook())
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new, temperature=temperature,
                                     do_sample=do_sample, pad_token_id=tokenizer.eos_token_id)
        handle.remove()
        answer = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        results.append({**sample, "sample_id": sid, "generated_output": answer,
                         "mode": mode_label, "alpha": alpha,
                         "alpha_applied": effective_alpha[0],
                         "probe_score": round(probe_score[0], 6), "gate": 1.0})
    return results


def run_hard_gate(model, tokenizer, test_data, steering_v, layer_idx, alpha,
                   probe_info, threshold, gen_cfg, mode_label):
    """Apply steering only when probe < threshold — binary gate."""
    max_new = gen_cfg.get("max_new_tokens", 48)
    temperature = gen_cfg.get("temperature", 0.0)
    do_sample = gen_cfg.get("do_sample", False)
    device = next(model.parameters()).device
    vec_tensor = torch.from_numpy(steering_v).to(device).float()
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
                    pooled = h[-1, :].detach().cpu().float().numpy()
                    X_s = probe_info["scaler"].transform(pooled.reshape(1, -1))
                    proba = probe_info["classifier"].predict_proba(X_s)[0, 1]
                    probe_score[0] = float(proba)
                    gate_decided[0] = True
                    if proba >= threshold:
                        effective_alpha[0] = 0.0
                    else:
                        effective_alpha[0] = alpha
                if abs(effective_alpha[0]) > 0.001:
                    v = vec_tensor.to(dtype=h_full.dtype, device=h_full.device)
                    h_modified = h_full + effective_alpha[0] * v
                    if isinstance(fn_outputs, tuple):
                        return (h_modified,) + fn_outputs[1:]
                    else:
                        return h_modified
                return None
            return hook

        handle = layer_module.register_forward_hook(make_hook())
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new, temperature=temperature,
                                     do_sample=do_sample, pad_token_id=tokenizer.eos_token_id)
        handle.remove()
        answer = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        results.append({**sample, "sample_id": sid, "generated_output": answer,
                         "mode": mode_label, "alpha": alpha,
                         "alpha_applied": effective_alpha[0],
                         "probe_score": round(probe_score[0], 6)})
    return results


def main():
    _log("=" * 60)
    _log("IC-4 A2 UPGRADE: 30A+30U Soft/Hard/Open-Loop Comparison")
    _log("=" * 60)

    _log("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer(
        model_name="Qwen/Qwen2.5-0.5B-Instruct", device="cpu", torch_dtype="float32")
    _log(f"  Layers: {get_model_layer_count(model)}")

    all_metrics = []

    for cfg in CONFIGS:
        name = cfg["name"]
        layer = cfg["layer"]
        train_seeds = cfg["train_seeds"]
        eval_seed = cfg["eval_seed"]
        _log(f"\n{'='*50}")
        _log(f"Config: {name}  (layer={layer}, train={train_seeds}, eval_s{eval_seed})")
        t0 = time.time()

        pos_parts, neg_parts = [], []
        for ts in train_seeds:
            ap = os.path.join(BASE_DIR, "results_m3", f"activations_s{ts}_l{layer}.npz")
            a = load_activations(ap)
            pos_parts.append(a["positive"])
            neg_parts.append(a["negative"])
        pos_all = np.concatenate(pos_parts, axis=0)
        neg_all = np.concatenate(neg_parts, axis=0)
        _log(f"  Merged activations: {pos_all.shape[0]} pairs")

        all_vecs = get_all_vectors(pos_all, neg_all, pos_all.shape[1])
        steering_v = all_vecs["steering"]

        train_path = os.path.join(BASE_DIR, "data_m3", "train.jsonl")
        train_all = []
        for ts in train_seeds:
            tf = train_path.replace(".jsonl", f"_s{ts}.jsonl")
            train_all.extend(load_jsonl(tf))
        _log(f"  Train: {len(train_all)} samples")

        test_f = os.path.join(BASE_DIR, "data_m3", "test.jsonl").replace(".jsonl", f"_s{eval_seed}.jsonl")
        test_data = load_jsonl(test_f)
        na = sum(1 for s in test_data if s.get("answerability") == "answerable")
        _log(f"  Test: {na}A+{len(test_data)-na}U")

        _log("  Collecting prefill features...")
        X_tr, y_tr = collect_prefill_features(model, tokenizer, train_all, layer)
        probe = train_probe(X_tr, y_tr)
        _log(f"  Probe: train_acc={probe['train_acc']:.4f}, AUC={probe['auc']}")

        _log(f"  [1] Open-loop (all samples) ...")
        ol_res = run_open_loop(model, tokenizer, test_data, steering_v, layer, ALPHA,
                                GEN_CFG, f"{name}_open_loop")
        ol_m = evaluate_outputs(ol_res)
        ol_m["config"], ol_m["gate_mode"] = name, "open_loop"
        all_metrics.append(ol_m)
        _log(f"      open_loop:  H={ol_m['hallucination_rate']:.3f} C={ol_m['correct_answer_rate']:.3f}")

        _log(f"  [2] Soft gate (proportional) ...")
        sg_res = run_soft_gate(model, tokenizer, test_data, steering_v, layer, ALPHA,
                                probe, THRESHOLD, GEN_CFG, f"{name}_soft_gate")
        sg_m = evaluate_outputs(sg_res)
        sg_m["config"], sg_m["gate_mode"] = name, "soft_gate"
        all_metrics.append(sg_m)
        _log(f"      soft_gate:  H={sg_m['hallucination_rate']:.3f} C={sg_m['correct_answer_rate']:.3f}")

        _log(f"  [3] Hard gate (binary) ...")
        hg_res = run_hard_gate(model, tokenizer, test_data, steering_v, layer, ALPHA,
                                probe, THRESHOLD, GEN_CFG, f"{name}_hard_gate")
        hg_m = evaluate_outputs(hg_res)
        hg_m["config"], hg_m["gate_mode"] = name, "hard_gate"
        all_metrics.append(hg_m)
        _log(f"      hard_gate:  H={hg_m['hallucination_rate']:.3f} C={hg_m['correct_answer_rate']:.3f}")

        elapsed = (time.time() - t0) / 60
        _log(f"  Done in {elapsed:.1f} min")

    _log(f"\n{'='*60}")
    _log("RESULTS SUMMARY")
    _log(f"{'='*60}")

    for m in all_metrics:
        _log(f"  {m['config']:<25} {m['gate_mode']:<12} H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f}")

    _log(f"\nGATE MODE COMPARISON (per config):")
    for cfg_name in [c["name"] for c in CONFIGS]:
        sub = [m for m in all_metrics if m["config"] == cfg_name]
        gmap = {m["gate_mode"]: m for m in sub}
        ol_h = gmap.get("open_loop", {}).get("hallucination_rate", 1)
        sg_h = gmap.get("soft_gate", {}).get("hallucination_rate", 1)
        hg_h = gmap.get("hard_gate", {}).get("hallucination_rate", 1)
        _log(f"  {cfg_name:<25} open_loop={ol_h:.3f} > soft={sg_h:.3f} > hard={hg_h:.3f}")

    df = pd.DataFrame(all_metrics)
    csv_path = os.path.join(OUT_DIR, "a2_upgrade_30A30U.csv")
    df.to_csv(csv_path, index=False)
    _log(f"\nSaved to {csv_path}")
    _log("Done.")


if __name__ == "__main__":
    main()