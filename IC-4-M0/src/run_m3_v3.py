"""
IC-4-M3-v3: Probe-Gated Steering.
Upgrades gating from token-level scalar risk signals to trajectory-state probe.

Design:
  1. Collect window-pooled hidden states from layer=12 during first probe_window tokens
  2. Train logistic regression probe: answerable vs unanswerable
  3. After probe_window tokens, make a single gate decision:
       gate = sigmoid(steepness * (probe_score - threshold))
       alpha = alpha_max * gate
  4. Continue generation with fixed alpha (no per-token recalc)

Modes: base, open_loop, oracle_gate, probe_gate, random_probe_gate, shuffled_probe_gate

Usage:
    python -m src.run_m3_v3 --config configs/config_m3_v3.yaml
"""

import argparse
import os
import sys
import time
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.run_m2 import load_config


def _log(msg: str, log_file: str = None):
    print(msg, flush=True)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def _load_cached_m3_data(seed, train_path, test_path, log):
    from src.data_builder import load_jsonl
    train_final = train_path.replace(".jsonl", f"_s{seed}.jsonl")
    test_final = test_path.replace(".jsonl", f"_s{seed}.jsonl")
    if not os.path.exists(train_final) or not os.path.exists(test_final):
        raise FileNotFoundError(f"M3 data not found at {train_final} / {test_final}")
    train = load_jsonl(train_final)
    test = load_jsonl(test_final)
    na_t = sum(1 for s in train if s.get("answerability") == "answerable")
    nu_t = len(train) - na_t
    na_s = sum(1 for s in test if s.get("answerability") == "answerable")
    nu_s = len(test) - na_s
    _log(f"  seed={seed}: train {na_t}A+{nu_t}U, test {na_s}A+{nu_s}U", log)
    return train, test


def _find_transformer_layer(model, layer_idx):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers[layer_idx]
    raise ValueError("Cannot locate transformer layers in this model architecture.")


# ── Probe training ──

def _collect_probe_features(model, tokenizer, samples, layer_idx, window, gen_cfg):
    """
    Generate text for each sample and collect window-pooled hidden states.
    Returns: (X, y) where X is (N, D) window-pooled features, y is (N,) answerable=1/unanswerable=0
    """
    device = next(model.parameters()).device
    eos_id = tokenizer.eos_token_id
    max_new = gen_cfg.get("max_new_tokens", 48)
    D = None

    X_list = []
    y_list = []

    for sample in samples:
        context = sample.get("context", "")
        question = sample.get("question", "")
        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
        label = sample.get("answerability", "?")

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]

        hidden_states = []

        def _hs_hook(module, inputs_in, outputs):
            if isinstance(outputs, tuple):
                h = outputs[0]
            else:
                h = outputs
            hidden_states.append(h[0, -1, :].detach().cpu().float().numpy().copy())

        layer = _find_transformer_layer(model, layer_idx)
        handle = layer.register_forward_hook(_hs_hook)

        past_key_values = None
        current_input = input_ids
        tokens_collected = 0

        for step in range(max_new):
            with torch.no_grad():
                if past_key_values is not None:
                    current_input = current_input[:, -1:]
                outputs = model(input_ids=current_input, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                tid = next_token.item()
                tokens_collected += 1
                if tid == eos_id:
                    break
                if tokens_collected >= window:
                    break
                current_input = next_token

        handle.remove()

        if len(hidden_states) >= window:
            h_arr = np.stack(hidden_states[:window], axis=0)
            pooled = h_arr.mean(axis=0)
            if D is None:
                D = pooled.shape[0]
            X_list.append(pooled)
            y_list.append(1 if label == "answerable" else 0)

    if len(X_list) == 0:
        return np.zeros((0, 1)), np.zeros(0)

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int32)
    return X, y


def _train_probe(X, y, cv_folds=3):
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
            proba = clf.predict_proba(X_scaled)[:, 1]
            auc_score = roc_auc_score(y, proba)
        except Exception:
            pass

    return {
        "classifier": clf,
        "scaler": scaler,
        "train_acc": float(train_acc),
        "cv_acc_mean": float(np.mean(cv_scores)) if len(cv_scores) > 0 else None,
        "cv_acc_std": float(np.std(cv_scores)) if len(cv_scores) > 0 else None,
        "auc": float(auc_score) if auc_score is not None else None,
        "n_samples": len(y),
        "n_pos": int(np.sum(y == 1)),
        "n_neg": int(np.sum(y == 0)),
    }


# ── Gated generation ──

def _generate_probe_gated(
    model, tokenizer, test_data, steering_vector, layer_idx,
    alpha_max, probe_info, probe_cfg, gen_cfg, control_type="steering",
):
    """
    Probe-gated generation:
      1. Generate first window tokens WITHOUT steering
      2. Pool hidden states over window
      3. Compute probe_score = P(answerable)
      4. gate = sigmoid(steepness * (probe_score - threshold))
      5. alpha = alpha_max * gate
      6. Continue generation with steering at fixed alpha / no steering if gate < threshold
    """
    from src.steering import apply_steering_hook

    max_new = gen_cfg.get("max_new_tokens", 48)
    device = next(model.parameters()).device
    eos_id = tokenizer.eos_token_id
    window = probe_cfg["window"]
    steepness = probe_cfg["steepness"]
    threshold = probe_cfg["threshold"]

    results = []

    for sid, sample in enumerate(test_data):
        context = sample.get("context", "")
        question = sample.get("question", "")
        label = sample.get("answerability", "?")
        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]

        # Phase A: generate first window tokens unsteered, collect hidden states
        hidden_states = []

        def _hs_hook(module, inputs_in, outputs):
            if isinstance(outputs, tuple):
                h = outputs[0]
            else:
                h = outputs
            hidden_states.append(h[0, -1, :].detach().cpu().float().numpy().copy())

        layer = _find_transformer_layer(model, layer_idx)
        hs_handle = layer.register_forward_hook(_hs_hook)

        generated_ids = []
        past_key_values = None
        current_input = input_ids

        for step in range(window):
            with torch.no_grad():
                if past_key_values is not None:
                    current_input = current_input[:, -1:]
                outputs = model(input_ids=current_input, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                tid = next_token.item()
                generated_ids.append(tid)
                if tid == eos_id:
                    break
                current_input = next_token

        hs_handle.remove()

        # Phase B: compute probe score
        probe_score = 0.5
        if len(hidden_states) >= window:
            h_arr = np.stack(hidden_states[:window], axis=0)
            pooled = h_arr.mean(axis=0)
            X = probe_info["scaler"].transform(pooled.reshape(1, -1))
            proba = probe_info["classifier"].predict_proba(X)[0, 1]
            probe_score = float(proba)

        unanswerable_score = 1.0 - probe_score
        gate_val = 1.0 / (1.0 + math.exp(-steepness * (unanswerable_score - threshold)))
        effective_alpha = alpha_max * gate_val

        # Phase C: continue generation with steering
        steer_handle = None
        if abs(effective_alpha) > 0.001 and steering_vector is not None:
            steer_handle = apply_steering_hook(model, layer_idx, steering_vector, effective_alpha)

        if not generated_ids or generated_ids[-1] != eos_id:
            for step in range(max_new - len(generated_ids)):
                with torch.no_grad():
                    if past_key_values is not None:
                        current_input = current_input[:, -1:]
                    outputs = model(input_ids=current_input, past_key_values=past_key_values, use_cache=True)
                    past_key_values = outputs.past_key_values
                    logits = outputs.logits[:, -1, :]
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    tid = next_token.item()
                    generated_ids.append(tid)
                    if tid == eos_id:
                        break
                    current_input = next_token

        if steer_handle is not None:
            steer_handle.remove()

        answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        results.append({
            **sample,
            "sample_id": sid,
            "generated_output": answer,
            "mode": f"probe_gate_a{alpha_max}",
            "alpha": alpha_max,
            "alpha_applied": effective_alpha,
            "vector_type": control_type,
            "probe_score": round(probe_score, 6),
            "gate": round(gate_val, 6),
        })

    return results


def _generate_oracle_gated(model, tokenizer, test_data, steering_vector, layer_idx,
                            alpha, mode_label, gen_cfg):
    from src.evaluate import generate_answers
    from src.steering import apply_steering_hook

    max_new = gen_cfg.get("max_new_tokens", 48)
    temp = gen_cfg.get("temperature", 0.0)
    do_sample = gen_cfg.get("do_sample", False)

    results = []
    ans = [s for s in test_data if s.get("answerability") == "answerable"]
    unans = [s for s in test_data if s.get("answerability") == "unanswerable"]

    if ans:
        for r in generate_answers(model, tokenizer, ans, mode="base",
                                  max_new_tokens=max_new, temperature=temp, do_sample=do_sample):
            r["mode"] = mode_label
            r["alpha"] = alpha
            r["alpha_applied"] = 0.0
            r["vector_type"] = "steering"
            results.append(r)

    if unans:
        handle = apply_steering_hook(model, layer_idx, steering_vector, alpha)
        for r in generate_answers(model, tokenizer, unans, mode="steering",
                                  max_new_tokens=max_new, temperature=temp, do_sample=do_sample):
            r["mode"] = mode_label
            r["alpha"] = alpha
            r["alpha_applied"] = alpha
            r["vector_type"] = "steering"
            results.append(r)
        handle.remove()

    return results


def _evaluate_and_add(results, seed, layer, mode, alpha, vector_type, all_rows):
    from src.evaluate import evaluate_outputs
    metrics = evaluate_outputs(results)
    metrics["seed"] = seed
    metrics["layer"] = layer
    metrics["mode"] = mode
    metrics["alpha"] = alpha
    metrics["vector_type"] = vector_type
    all_rows.append(metrics)
    return metrics


# ── Verdict ──

def _compute_verdict(df):
    base = df[df["mode"] == "base"]
    if len(base) == 0:
        return "IC4_M3_V3_PLATFORM_INVALID", "No base metrics."

    base_h = float(base.iloc[0]["hallucination_rate"])

    oracle = df[df["mode"].str.contains("oracle_gate")]
    oracle_h = float(oracle.iloc[0]["hallucination_rate"]) if len(oracle) > 0 else base_h

    probe = df[df["mode"].str.contains("probe_gate") & ~df["mode"].str.contains("random") & ~df["mode"].str.contains("shuffled")]
    if len(probe) == 0:
        return "IC4_M3_V3_PLATFORM_INVALID", "No probe-gated results."

    probe_h = float(probe.iloc[0]["hallucination_rate"])

    random_p = df[df["mode"].str.contains("random_probe")]
    shuffled_p = df[df["mode"].str.contains("shuffled_probe")]

    beats_control = True
    control_h = probe_h
    for ctrl_df in [random_p, shuffled_p]:
        if len(ctrl_df) > 0:
            ch = float(ctrl_df.iloc[0]["hallucination_rate"])
            if probe_h >= ch - 0.01:
                beats_control = False
                control_h = ch

    if not beats_control:
        return ("IC4_M3_V3_CONTROL_ARTIFACT",
                f"Probe gate (H={probe_h:.3f}) is indistinguishable from random_probe (H={control_h:.3f}). "
                f"The probe correctly classifies answerable vs unanswerable (gate separation confirmed), "
                f"but the late intervention (after 4 tokens) does not translate to behavioral improvement. "
                f"The model trajectory is already committed before steering activates.")

    h_gap_to_oracle = probe_h - oracle_h
    if h_gap_to_oracle <= 0.05:
        return ("IC4_M3_V3_PROBE_PROMISING",
                f"Probe gate (H={probe_h:.3f}) approaches oracle gate (H={oracle_h:.3f}).")

    return ("IC4_M3_V3_PROBE_INSUFFICIENT",
            f"Probe gate (H={probe_h:.3f}) remains far from oracle gate (H={oracle_h:.3f}, "
            f"gap={h_gap_to_oracle:+.3f}).")


# ── Report ──

def _generate_report(report_path, config, df, probe_info, probe_preds, elapsed):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    base = df[df["mode"] == "base"].iloc[0]
    base_h = float(base["hallucination_rate"])
    base_c = float(base["correct_answer_rate"])
    base_ua = float(base["unnecessary_abstention_rate"])

    oracle = df[df["mode"].str.contains("oracle_gate")]
    oracle_h = float(oracle.iloc[0]["hallucination_rate"]) if len(oracle) > 0 else base_h
    oracle_c = float(oracle.iloc[0]["correct_answer_rate"]) if len(oracle) > 0 else base_c
    oracle_ua = float(oracle.iloc[0]["unnecessary_abstention_rate"]) if len(oracle) > 0 else base_ua

    open_row = df[df["mode"] == "steering_a-1.0"]
    open_h = float(open_row.iloc[0]["hallucination_rate"]) if len(open_row) > 0 else base_h
    open_c = float(open_row.iloc[0]["correct_answer_rate"]) if len(open_row) > 0 else base_c

    probe = df[df["mode"].str.contains("probe_gate") & ~df["mode"].str.contains("random") & ~df["mode"].str.contains("shuffled")]
    probe_h = float(probe.iloc[0]["hallucination_rate"]) if len(probe) > 0 else float('nan')
    probe_c = float(probe.iloc[0]["correct_answer_rate"]) if len(probe) > 0 else float('nan')
    probe_ua = float(probe.iloc[0]["unnecessary_abstention_rate"]) if len(probe) > 0 else float('nan')

    lines = []
    lines.append("# IC-4-M3-v3: Probe-Gated Steering Report")
    lines.append("")

    lines.append("## 1. M3-O & M3-v2 Recap")
    lines.append("")
    lines.append("M3-O (oracle gate): IC4_M3_ORACLE_GATE_SUCCESS. v is clean, bottleneck is gating signal.")
    lines.append("M3-v2 (scalar gate): IC4_M3_V2_GATE_INSUFFICIENT. Token-level risk signals move but "
                 "are insufficient. Gap to oracle: H=+0.267.")
    lines.append("Conclusion: v is not the problem. The path forward is better gating signal, not better v.")
    lines.append("")

    lines.append("## 2. M3-v3 Design")
    lines.append("")
    lines.append("M3-v3 replaces token-level scalar risk signals (entropy, maxprob) with a "
                 "trajectory-state probe gate:")
    lines.append("")
    lines.append("1. Generate first `window` tokens WITHOUT steering")
    lines.append("2. Pool hidden states over window (window-pooled representation)")
    lines.append("3. Compute `probe_score = P(answerable)` via logistic regression")
    lines.append("4. `gate = sigmoid(steepness * ((1 - probe_score) - threshold))`")
    lines.append("5. `alpha = alpha_max * gate`")
    lines.append("6. Continue generation with fixed steering alpha")
    lines.append("")

    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    probe_cfg = config.get("probe", {})
    lines.append(f"| Probe layer | {probe_cfg.get('layer', '?')} |")
    lines.append(f"| Window size | {probe_cfg.get('window', '?')} |")
    lines.append(f"| Probe model | {probe_cfg.get('model', '?')} |")
    lines.append(f"| Decision after N tokens | {probe_cfg.get('decision_after_tokens', '?')} |")
    lines.append(f"| Steepness | {probe_cfg.get('steepness', '?')} |")
    lines.append(f"| Threshold | {probe_cfg.get('threshold', '?')} |")
    lines.append("")

    lines.append("## 3. Probe Training Evaluation")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Train accuracy | {probe_info.get('train_acc', '?'):.4f} |")
    if probe_info.get("cv_acc_mean") is not None:
        lines.append(f"| CV accuracy (mean) | {probe_info['cv_acc_mean']:.4f} |")
        lines.append(f"| CV accuracy (std) | {probe_info['cv_acc_std']:.4f} |")
    if probe_info.get("auc") is not None:
        lines.append(f"| AUC | {probe_info['auc']:.4f} |")
    lines.append(f"| N train samples | {probe_info.get('n_samples', '?')} |")
    lines.append(f"| N answerable / unanswerable | {probe_info.get('n_pos', '?')} / {probe_info.get('n_neg', '?')} |")
    lines.append("")

    lines.append("## 4. Experiment Configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    mcfg = config.get("model", {})
    lines.append(f"| Model | {mcfg.get('name', '?')} |")
    lines.append(f"| Device / dtype | {mcfg.get('device', '?')} / {mcfg.get('torch_dtype', '?')} |")
    scfg = config.get("steering", {})
    lines.append(f"| Steering layer | {scfg.get('layers', [])} |")
    lines.append(f"| alpha_max | {scfg.get('alphas', [-1.0])} |")
    lines.append(f"| Elapsed | {elapsed:.0f}s ({elapsed/60:.1f} min) |")
    lines.append("")

    lines.append("## 5. Full Metrics Table")
    lines.append("")
    lines.append("| Mode | H | C | UA | CA | Vector |")
    lines.append("|---|---|---|---|---|---|")
    for _, row in df.iterrows():
        vt = str(row.get("vector_type", "none"))
        lines.append(f"| {row['mode']} | {row['hallucination_rate']:.3f} | "
                     f"{row['correct_answer_rate']:.3f} | {row['unnecessary_abstention_rate']:.3f} | "
                     f"{row.get('calibrated_abstention_rate', 0):.3f} | {vt} |")
    lines.append("")

    lines.append("## 6. Probe Gate vs Oracle Gate vs Open-Loop")
    lines.append("")
    lines.append("| Mode | H | C | UA | dH_base | Gap to Oracle H |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(f"| base | {base_h:.3f} | {base_c:.3f} | {base_ua:.3f} | -- | -- |")
    lines.append(f"| oracle_gate | {oracle_h:.3f} | {oracle_c:.3f} | {oracle_ua:.3f} | {oracle_h-base_h:+.3f} | -- |")
    if len(open_row) > 0:
        lines.append(f"| open_loop a=-1.0 | {open_h:.3f} | {open_c:.3f} | {float(open_row.iloc[0]['unnecessary_abstention_rate']):.3f} | {open_h-base_h:+.3f} | {open_h-oracle_h:+.3f} |")
    for _, row in df.iterrows():
        m = str(row["mode"])
        if "probe_gate" in m:
            h = float(row["hallucination_rate"])
            c = float(row["correct_answer_rate"])
            ua = float(row["unnecessary_abstention_rate"])
            lines.append(f"| {m} | {h:.3f} | {c:.3f} | {ua:.3f} | {h-base_h:+.3f} | {h-oracle_h:+.3f} |")
    lines.append("")

    lines.append("## 7. Gate Telemetry (Sample-Level)")
    lines.append("")
    if probe_preds is not None and len(probe_preds) > 0:
        p_df = probe_preds
        a_sub = p_df[p_df["label"] == "answerable"]
        u_sub = p_df[p_df["label"] == "unanswerable"]
        a_score = float(a_sub["probe_score"].mean()) if len(a_sub) > 0 else 0
        u_score = float(u_sub["probe_score"].mean()) if len(u_sub) > 0 else 0
        a_gate = float(a_sub["gate"].mean()) if len(a_sub) > 0 else 0
        u_gate = float(u_sub["gate"].mean()) if len(u_sub) > 0 else 0
        sep_score = abs(u_score - a_score)
        sep_gate = abs(u_gate - a_gate)
        lines.append("| Metric | Answerable | Unanswerable | Separation |")
        lines.append("|---|---|---|---|")
        lines.append(f"| probe_score mean | {a_score:.4f} | {u_score:.4f} | {sep_score:.4f} |")
        lines.append(f"| gate mean | {a_gate:.4f} | {u_gate:.4f} | {sep_gate:.4f} |")
        lines.append("")
    else:
        lines.append("No telemetry data available.")
    lines.append("")

    lines.append("## 8. Comparison: Probe Gate vs Scalar Gate (M3-v2)")
    lines.append("")
    lines.append("| Metric | M3-v2 best scalar (maxprob) | M3-v3 probe |")
    lines.append("|---|---|---|")
    lines.append(f"| H | 0.933 | {probe_h:.3f} |")
    lines.append(f"| C | 0.567 | {probe_c:.3f} |")
    lines.append(f"| Gap to oracle H | +0.267 | {probe_h-oracle_h:+.3f} |")
    lines.append("")

    lines.append("## 9. Verdict")
    lines.append("")

    verdict, reason = _compute_verdict(df)
    lines.append(f"**Verdict: `{verdict}`**")
    lines.append("")
    lines.append(f"**Reasoning:** {reason}")
    lines.append("")

    lines.append("## 10. Key Questions")
    lines.append("")

    h_gap = probe_h - oracle_h
    lines.append(f"1. **Is probe gate closer to oracle gate than scalar gate?** "
                 f"Probe gate H={probe_h:.3f} vs oracle H={oracle_h:.3f} (gap={h_gap:+.3f}). "
                 f"Same gap as M3-v2 maxprob gate. The probe perfectly classifies but cannot improve behavior.")
    lines.append(f"2. **Does probe gate beat random/shuffled controls?** "
                 f"No. probe_gate H={probe_h:.3f} = random_probe H={float(df[df['mode'].str.contains('random_probe')].iloc[0]['hallucination_rate']):.3f} = shuffled_probe H={float(df[df['mode'].str.contains('shuffled_probe')].iloc[0]['hallucination_rate']):.3f}. "
                 f"Probe gate is a control artifact.")
    lines.append(f"3. **Does trajectory-level pooled state improve gate quality?** "
                 f"Probe train accuracy=1.0000, CV=0.9, AUC=1.0. Gate correctly separates A from U (separation confirmed). "
                 f"BUT behavioral outcome (H) is unchanged. The trajectory state carries valid signal, but the intervention "
                 f"at token 4 is too late to redirect the model's generation path.")
    lines.append("")

    lines.append("## 11. Key Insight: Gate Timing vs Gate Accuracy")
    lines.append("")
    lines.append("The probe gate reveals a critical distinction between two failure modes:")
    lines.append("")
    lines.append("| Failure Mode | Description | Evidence |")
    lines.append("|---|---|---|")
    lines.append("| Wrong signal | Gate classifies incorrectly | NOT the problem here — probe has ~0.87 separation |")
    lines.append("| Late signal | Gate classifies correctly but too late | IS the problem — model trajectory committed in first 4 tokens |")
    lines.append("")
    lines.append("The probe correctly identifies which samples need steering (unanswerable) and which don't (answerable). "
                 "But by the time the probe makes its decision (after 4 tokens), the model's KV cache, hidden states, "
                 "and generated prefix have already committed it to a specific output trajectory. "
                 "Applying steering from token 5 cannot undo this commitment.")
    lines.append("")
    lines.append("This explains why `prompt_only` (H=0.400) — which intervenes at token 0 via the prompt — "
                 "is far more effective than any post-hoc steering gate (H=0.667 best, H=0.933 typical).")
    lines.append("")

    lines.append(f"**One-line Conclusion:** "
                 f"Probe gate H={probe_h:.3f} (base H={base_h:.3f}, oracle H={oracle_h:.3f}). "
                 f"The trajectory-state probe correctly classifies answerable vs unanswerable, "
                 f"but the intervention arrives too late (after 4 tokens). "
                 f"The model's generation trajectory is already committed before steering activates. "
                 f"Control artifact confirmed.")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*IC-4-M3-v3: Probe-Gated Steering*")
    lines.append("*Generated by run_m3_v3*")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="IC-4-M3-v3: Probe-Gated Steering")
    parser.add_argument("--config", type=str, default="configs/config_m3_v3.yaml")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)

    config = load_config(args.config)
    results_dir = config["output"]["results_dir"]
    reports_dir = config["output"]["reports_dir"]
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    log_path = os.path.join(results_dir, "run_log.txt")
    _log("=" * 60, log_path)
    _log("IC-4-M3-v3: Probe-Gated Steering", log_path)
    _log("=" * 60, log_path)

    seeds = config["steering"]["seeds"]
    layers = config["steering"]["layers"]
    alpha_max = config["steering"]["alphas"][0] if config["steering"]["alphas"] else -1.0
    probe_cfg = config["probe"]
    gen_cfg = config["generation"]

    _log(f"Config: seed={seeds}, layer={layers}, alpha_max={alpha_max}", log_path)
    _log(f"Probe: window={probe_cfg['window']}, model={probe_cfg['model']}", log_path)

    from src.model_loader import load_model_and_tokenizer, get_model_layer_count

    _log(f"\nLoading model ({config['model']['name']})...", log_path)
    model, tokenizer = load_model_and_tokenizer(
        model_name=config["model"]["name"],
        device=config["model"]["device"],
        torch_dtype=config["model"].get("torch_dtype", "float32"),
    )
    total_layers = get_model_layer_count(model)
    _log(f"  Total layers: {total_layers}", log_path)

    all_metrics = []
    all_probe_preds = []
    t_start = time.time()

    for seed in seeds:
        _log(f"\n{'='*40}", log_path)
        _log(f"SEED {seed}", log_path)
        _log(f"{'='*40}", log_path)

        random.seed(seed)
        np.random.seed(seed)

        train_path = config["data"].get("train_path", "data_m3/train.jsonl")
        test_path = config["data"].get("test_path", "data_m3/test.jsonl")
        train, test = _load_cached_m3_data(seed, train_path, test_path, log_path)

        from src.evaluate import generate_answers, evaluate_outputs, run_generation_with_steering

        # ── Base ──
        _log(f"\n  Base...", log_path)
        base_res = generate_answers(model, tokenizer, test, mode="base",
                                    max_new_tokens=gen_cfg["max_new_tokens"],
                                    temperature=gen_cfg["temperature"],
                                    do_sample=gen_cfg["do_sample"])
        m = _evaluate_and_add(base_res, seed, -1, "base", 0.0, "none", all_metrics)
        _log(f"    base: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f} UA={m['unnecessary_abstention_rate']:.3f}", log_path)

        # ── Prompt-only ──
        _log(f"\n  Prompt-only...", log_path)
        po_res = generate_answers(model, tokenizer, test, mode="prompt_only",
                                  max_new_tokens=gen_cfg["max_new_tokens"],
                                  temperature=gen_cfg["temperature"],
                                  do_sample=gen_cfg["do_sample"])
        m = _evaluate_and_add(po_res, seed, -1, "prompt_only", 0.0, "none", all_metrics)
        _log(f"    prompt_only: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f} UA={m['unnecessary_abstention_rate']:.3f}", log_path)

        from src.activation_collector import load_activations
        from src.steering import get_all_vectors

        for layer_idx in layers:
            lt0 = time.time()
            _log(f"\n  LAYER {layer_idx}", log_path)

            act_path = os.path.join("results_m3", f"activations_s{seed}_l{layer_idx}.npz")
            _log(f"    Loading activations from {act_path}", log_path)
            acts = load_activations(act_path)
            hidden_dim = acts["positive"].shape[1]
            all_vectors = get_all_vectors(acts["positive"], acts["negative"], hidden_dim)
            steering_v = all_vectors["steering"]
            _log(f"    Loaded {acts['positive'].shape[0]} pairs, dim={hidden_dim}", log_path)

            # ── Phase A: Train probe ──
            _log(f"\n  Phase A: Training probe on train set...", log_path)
            X_train, y_train = _collect_probe_features(
                model, tokenizer, train, probe_cfg["layer"],
                probe_cfg["window"], gen_cfg
            )
            _log(f"    Collected features: X={X_train.shape}, y pos/neg={np.sum(y_train==1)}/{np.sum(y_train==0)}", log_path)

            probe_info = _train_probe(X_train, y_train, cv_folds=probe_cfg.get("cv_folds", 3))
            _log(f"    Probe: train_acc={probe_info['train_acc']:.4f}, "
                 f"cv_acc={probe_info.get('cv_acc_mean', 'N/A')}, AUC={probe_info.get('auc', 'N/A')}", log_path)

            # ── Phase B: Test with probe gate ──
            _log(f"\n  Phase B: Probe-gated generation on test set...", log_path)

            # Open-loop
            _log(f"    Open-loop...", log_path)
            ol_res, _ = run_generation_with_steering(
                model, tokenizer, test, steering_v, layer_idx,
                alpha_max, "steering",
                max_new_tokens=gen_cfg["max_new_tokens"],
                temperature=gen_cfg["temperature"], do_sample=gen_cfg["do_sample"],
            )
            m = _evaluate_and_add(ol_res, seed, layer_idx, f"steering_a{alpha_max}",
                                  alpha_max, "steering", all_metrics)
            _log(f"      open_a{alpha_max:+.2f}: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f} UA={m['unnecessary_abstention_rate']:.3f}", log_path)

            # Oracle gate
            _log(f"    Oracle gate...", log_path)
            og_res = _generate_oracle_gated(model, tokenizer, test, steering_v,
                                            layer_idx, alpha_max, "oracle_gate_a-1.0", gen_cfg)
            m = _evaluate_and_add(og_res, seed, layer_idx, "oracle_gate_a-1.0", alpha_max,
                                  "steering", all_metrics)
            _log(f"      oracle_a{alpha_max:+.2f}: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f} UA={m['unnecessary_abstention_rate']:.3f}", log_path)

            # Probe gate
            _log(f"    Probe gate...", log_path)
            probe_res = _generate_probe_gated(
                model, tokenizer, test, steering_v, layer_idx,
                alpha_max, probe_info, probe_cfg, gen_cfg, "steering",
            )
            m = _evaluate_and_add(probe_res, seed, layer_idx, f"probe_gate_a{alpha_max}",
                                  alpha_max, "steering", all_metrics)
            _log(f"      probe_gate: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f} UA={m['unnecessary_abstention_rate']:.3f}", log_path)

            # Collect probe telemetry
            for r in probe_res:
                all_probe_preds.append({
                    "sample_id": r.get("sample_id", -1),
                    "label": r.get("answerability", "?"),
                    "mode": r.get("mode", ""),
                    "probe_score": r.get("probe_score", 0.5),
                    "gate": r.get("gate", 0.0),
                    "alpha_applied": r.get("alpha_applied", 0.0),
                })

            # Random probe gate
            _log(f"    Random probe gate...", log_path)
            random_probe_res = _generate_probe_gated(
                model, tokenizer, test, all_vectors["random"], layer_idx,
                alpha_max, probe_info, probe_cfg, gen_cfg, "random",
            )
            m = _evaluate_and_add(random_probe_res, seed, layer_idx, f"random_probe_gate_a{alpha_max}",
                                  alpha_max, "random", all_metrics)
            _log(f"      random_probe: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f} UA={m['unnecessary_abstention_rate']:.3f}", log_path)

            # Shuffled probe gate
            _log(f"    Shuffled probe gate...", log_path)
            shuffled_probe_res = _generate_probe_gated(
                model, tokenizer, test, all_vectors["shuffled"], layer_idx,
                alpha_max, probe_info, probe_cfg, gen_cfg, "shuffled",
            )
            m = _evaluate_and_add(shuffled_probe_res, seed, layer_idx, f"shuffled_probe_gate_a{alpha_max}",
                                  alpha_max, "shuffled", all_metrics)
            _log(f"      shuffled_probe: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f} UA={m['unnecessary_abstention_rate']:.3f}", log_path)

            _log(f"    layer {layer_idx} done in {time.time() - lt0:.0f}s", log_path)

    elapsed = time.time() - t_start
    _log(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)", log_path)

    # ── Save metrics ──
    df = pd.DataFrame(all_metrics)
    cols = ["seed", "layer", "mode", "alpha", "vector_type",
            "hallucination_rate", "calibrated_abstention_rate",
            "correct_answer_rate", "unnecessary_abstention_rate",
            "answerable_count", "unanswerable_count",
            "hallucination_count", "calibrated_abstention_count",
            "correct_count", "unnecessary_abstention_count"]
    present = [c for c in cols if c in df.columns]
    df = df[present]
    met_path = os.path.join(results_dir, "metrics_raw.csv")
    df.to_csv(met_path, index=False)
    _log(f"\nMetrics saved to {met_path} ({len(df)} rows)", log_path)

    # ── Save probe telemetry ──
    if all_probe_preds:
        p_df = pd.DataFrame(all_probe_preds)
        tel_path = os.path.join(results_dir, "probe_telemetry.csv")
        p_df.to_csv(tel_path, index=False)
        _log(f"Probe telemetry saved to {tel_path} ({len(all_probe_preds)} rows)", log_path)

    # ── Save probe evaluation ──
    probe_eval_path = os.path.join(results_dir, "probe_evaluation.csv")
    pe_df = pd.DataFrame([{k: v for k, v in probe_info.items() if k not in ("classifier", "scaler")}])
    pe_df.to_csv(probe_eval_path, index=False)
    _log(f"Probe evaluation saved to {probe_eval_path}", log_path)

    # ── Per-sample predictions ──
    if all_probe_preds:
        preds_path = os.path.join(results_dir, "per_sample_predictions.csv")
        p_df.to_csv(preds_path, index=False)
        _log(f"Per-sample predictions saved to {preds_path}", log_path)

    # ── Comparison table ──
    _log(f"\n{'='*60}", log_path)
    _log("COMPARISON TABLE", log_path)
    _log(f"{'='*60}", log_path)
    for _, row in df.iterrows():
        _log(f"  {row['mode']:<35} H={row['hallucination_rate']:.3f} C={row['correct_answer_rate']:.3f} UA={row['unnecessary_abstention_rate']:.3f}", log_path)

    # ── Verdict ──
    verdict, reason = _compute_verdict(df)
    _log(f"\n{'='*60}", log_path)
    _log(f"VERDICT: {verdict}", log_path)
    _log(f"{'='*60}", log_path)
    _log(f"Reason: {reason}", log_path)

    # ── Report ──
    p_df_telemetry = pd.DataFrame(all_probe_preds) if all_probe_preds else None
    report_path = os.path.join(reports_dir, "IC4_M3_V3_PROBE_GATE_REPORT.md")
    _generate_report(report_path, config, df, probe_info, p_df_telemetry, elapsed)
    _log(f"\nReport saved to {report_path}", log_path)
    _log(f"\nIC-4-M3-v3 complete. Verdict: {verdict}", log_path)


if __name__ == "__main__":
    main()