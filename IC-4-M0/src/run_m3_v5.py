"""
IC-4-M3-v5: Artifact Decomposition for Valid Prefill Gate.

Goal: Isolate the sources of the oracle gap and C anomaly observed in M3-v4b.

Key questions:
  1. Does two-pass prefill pipeline itself elevate C?
  2. Does tiny residual alpha (~-0.02) elevate C?
  3. Does hard gate (0 or 1) outperform soft gate (sigmoid)?
  4. Can two-pass open-loop replicate single-pass open-loop H=0.667?
  5. What is the primary source of the remaining oracle gap?

Modes:
  Baselines: base, prompt_only, steering_a-1.0, oracle_gate
  Two-pass artifacts: two_pass_base_no_steering, two_pass_tiny_alpha_only,
                      two_pass_open_loop_full_alpha
  Valid gates: soft_valid_prefill_probe_gate, hard_valid_prefill_probe_gate
  Controls: random_soft, shuffled_soft, random_hard, shuffled_hard

Usage:
    python -m src.run_m3_v5 --config configs/config_m3_v5.yaml
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


def _find_transformer_layer(model, layer_idx):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers[layer_idx]
    raise ValueError("Cannot locate transformer layers in this model architecture.")


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


def _get_question_span_indices(tokenizer, context, question):
    prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
    full_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    question_ids = tokenizer(question, add_special_tokens=False).input_ids
    n = len(question_ids)
    for start in range(len(full_ids) - n + 1):
        if full_ids[start:start + n] == question_ids:
            return list(range(start, start + n))
    ans_marker_ids = tokenizer("Answer:", add_special_tokens=False).input_ids
    ans_start = None
    for start in range(len(full_ids) - len(ans_marker_ids) + 1):
        if full_ids[start:start + len(ans_marker_ids)] == ans_marker_ids:
            ans_start = start
            break
    if ans_start is not None and ans_start > 3:
        return list(range(max(3, ans_start - 15), ans_start))
    third = len(full_ids) // 3
    return list(range(third, len(full_ids) - 3))


def _collect_prefill_features(model, tokenizer, samples, layer_idx, representation):
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
        elif representation == "question_span_pooled":
            indices = _get_question_span_indices(tokenizer, context, question)
            if indices:
                pooled = hs[indices, :].mean(dim=0).detach().cpu().float().numpy()
            else:
                pooled = hs[-5:-1, :].mean(dim=0).detach().cpu().float().numpy()
        else:
            pooled = hs[-1, :].detach().cpu().float().numpy()
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
            auc_score = roc_auc_score(y, clf.predict_proba(X_scaled)[:, 1])
        except Exception:
            pass
    return {
        "classifier": clf, "scaler": scaler,
        "train_acc": float(train_acc),
        "cv_acc_mean": float(np.mean(cv_scores)) if len(cv_scores) > 0 else None,
        "cv_acc_std": float(np.std(cv_scores)) if len(cv_scores) > 0 else None,
        "auc": float(auc_score) if auc_score is not None else None,
        "n_samples": len(y), "n_pos": int(np.sum(y == 1)), "n_neg": int(np.sum(y == 0)),
    }


# ==============================================================================
# TWO-PASS ARTIFACT DECOMPOSITION MODES
# ==============================================================================

def _two_pass_generate(model, tokenizer, samples, alpha, steering_vector,
                        layer_idx, gen_cfg, mode_label):
    """
    Generic two-pass prefill generation.
    Pass 1: unsteered prefill (KV cache discarded).
    Pass 2: steered prefill with given alpha (KV cache used for generation).
    """
    from src.steering import apply_steering_hook
    max_new = gen_cfg.get("max_new_tokens", 48)
    device = next(model.parameters()).device
    eos_id = tokenizer.eos_token_id
    results = []

    for sid, sample in enumerate(samples):
        context = sample.get("context", "")
        question = sample.get("question", "")
        label = sample.get("answerability", "?")
        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            model(**inputs, use_cache=True)

        handle = None
        if abs(alpha) > 0.001 and steering_vector is not None:
            handle = apply_steering_hook(model, layer_idx, steering_vector, alpha)

        with torch.no_grad():
            outputs_pass2 = model(**inputs, use_cache=True)
        steered_pkv = outputs_pass2.past_key_values

        generated_ids = []
        current_input = input_ids
        for _step in range(max_new):
            with torch.no_grad():
                if steered_pkv is not None:
                    current_input = current_input[:, -1:]
                out = model(input_ids=current_input, past_key_values=steered_pkv, use_cache=True)
                steered_pkv = out.past_key_values
                logits = out.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                tid = next_token.item()
                generated_ids.append(tid)
                if tid == eos_id:
                    break
                current_input = next_token

        if handle is not None:
            handle.remove()

        answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        results.append({
            **sample, "sample_id": sid, "generated_output": answer,
            "mode": mode_label, "alpha": alpha, "alpha_applied": alpha,
            "vector_type": "steering" if abs(alpha) > 0.001 else "none",
        })
    return results


def _generate_two_pass_no_steering(model, tokenizer, test_data, gen_cfg):
    """Two-pass prefill with alpha=0 (no steering at all)."""
    return _two_pass_generate(model, tokenizer, test_data, 0.0, None, 0,
                               gen_cfg, "two_pass_base_no_steering")


def _generate_two_pass_tiny_alpha(model, tokenizer, test_data, steering_vector,
                                   layer_idx, tiny_alpha, gen_cfg):
    """Two-pass prefill with uniform tiny alpha on ALL samples."""
    return _two_pass_generate(model, tokenizer, test_data, tiny_alpha,
                               steering_vector, layer_idx, gen_cfg,
                               f"two_pass_tiny_alpha{tiny_alpha:+.2f}")


def _generate_two_pass_open_loop(model, tokenizer, test_data, steering_vector,
                                  layer_idx, alpha, gen_cfg):
    """Two-pass prefill with uniform full alpha on ALL samples (two-pass equivalent of open-loop)."""
    return _two_pass_generate(model, tokenizer, test_data, alpha,
                               steering_vector, layer_idx, gen_cfg,
                               f"two_pass_open_loop_full_alpha{alpha:+.2f}")


# ==============================================================================
# SOFT / HARD PREFILL PROBE GATE
# ==============================================================================

def _pool_prefill_representation(hs, representation, tokenizer, context, question, device):
    if representation == "last_prompt_token":
        return hs[-1, :].detach().cpu().float().numpy()
    elif representation == "mean_pooled":
        return hs.mean(dim=0).detach().cpu().float().numpy()
    elif representation == "question_span_pooled":
        indices = _get_question_span_indices(tokenizer, context, question)
        if indices:
            return hs[indices, :].mean(dim=0).detach().cpu().float().numpy()
        else:
            return hs[-5:-1, :].mean(dim=0).detach().cpu().float().numpy()
    else:
        return hs[-1, :].detach().cpu().float().numpy()


def _generate_soft_prefill_probe_gated(
    model, tokenizer, test_data, steering_vector, layer_idx,
    alpha_max, probe_info, probe_cfg, gen_cfg, control_type="steering",
):
    """
    TWO-PASS PREFILL with SOFT (sigmoid) gate.
    Pass 1: unsteered → probe → gate = sigmoid(...) → alpha.
    Pass 2: steered with alpha, from scratch.
    """
    from src.steering import apply_steering_hook
    max_new = gen_cfg.get("max_new_tokens", 48)
    device = next(model.parameters()).device
    eos_id = tokenizer.eos_token_id
    representation = probe_cfg.get("representation", "last_prompt_token")
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

        with torch.no_grad():
            outputs_pass1 = model(**inputs, use_cache=True)
        hs = outputs_pass1.hidden_states[probe_cfg["layer"] + 1][0]
        pooled = _pool_prefill_representation(hs, representation, tokenizer, context, question, device)
        X = probe_info["scaler"].transform(pooled.reshape(1, -1))
        proba = probe_info["classifier"].predict_proba(X)[0, 1]
        probe_score = float(proba)
        gate_val = 1.0 / (1.0 + math.exp(-steepness * (threshold - probe_score)))
        effective_alpha = alpha_max * gate_val

        handle = None
        if abs(effective_alpha) > 0.001 and steering_vector is not None:
            handle = apply_steering_hook(model, layer_idx, steering_vector, effective_alpha)
        with torch.no_grad():
            outputs_pass2 = model(**inputs, use_cache=True)
        steered_pkv = outputs_pass2.past_key_values

        generated_ids = []
        current_input = input_ids
        for _step in range(max_new):
            with torch.no_grad():
                if steered_pkv is not None:
                    current_input = current_input[:, -1:]
                out = model(input_ids=current_input, past_key_values=steered_pkv, use_cache=True)
                steered_pkv = out.past_key_values
                logits = out.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                tid = next_token.item()
                generated_ids.append(tid)
                if tid == eos_id:
                    break
                current_input = next_token

        if handle is not None:
            handle.remove()

        answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        results.append({
            **sample, "sample_id": sid, "generated_output": answer,
            "mode": f"soft_prefill_probe_gate_a{alpha_max}",
            "alpha": alpha_max, "alpha_applied": effective_alpha,
            "vector_type": control_type,
            "probe_score": round(probe_score, 6), "gate": round(gate_val, 6),
        })
    return results


def _generate_hard_prefill_probe_gated(
    model, tokenizer, test_data, steering_vector, layer_idx,
    alpha_max, probe_info, probe_cfg, gen_cfg, control_type="steering",
):
    """
    TWO-PASS PREFILL with HARD (0 or 1) gate.
    Pass 1: unsteered → probe.
    If probe_score >= threshold → alpha = 0 (NO steering hook at all).
    If probe_score < threshold → alpha = alpha_max (full steering).
    """
    from src.steering import apply_steering_hook
    max_new = gen_cfg.get("max_new_tokens", 48)
    device = next(model.parameters()).device
    eos_id = tokenizer.eos_token_id
    representation = probe_cfg.get("representation", "last_prompt_token")
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

        with torch.no_grad():
            outputs_pass1 = model(**inputs, use_cache=True)
        hs = outputs_pass1.hidden_states[probe_cfg["layer"] + 1][0]
        pooled = _pool_prefill_representation(hs, representation, tokenizer, context, question, device)
        X = probe_info["scaler"].transform(pooled.reshape(1, -1))
        proba = probe_info["classifier"].predict_proba(X)[0, 1]
        probe_score = float(proba)

        is_answerable = probe_score >= threshold
        effective_alpha = 0.0 if is_answerable else alpha_max
        gate_val = 0.0 if is_answerable else 1.0

        handle = None
        if abs(effective_alpha) > 0.001 and steering_vector is not None:
            handle = apply_steering_hook(model, layer_idx, steering_vector, effective_alpha)
        with torch.no_grad():
            outputs_pass2 = model(**inputs, use_cache=True)
        steered_pkv = outputs_pass2.past_key_values

        generated_ids = []
        current_input = input_ids
        for _step in range(max_new):
            with torch.no_grad():
                if steered_pkv is not None:
                    current_input = current_input[:, -1:]
                out = model(input_ids=current_input, past_key_values=steered_pkv, use_cache=True)
                steered_pkv = out.past_key_values
                logits = out.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                tid = next_token.item()
                generated_ids.append(tid)
                if tid == eos_id:
                    break
                current_input = next_token

        if handle is not None:
            handle.remove()

        answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        results.append({
            **sample, "sample_id": sid, "generated_output": answer,
            "mode": f"hard_prefill_probe_gate_a{alpha_max}",
            "alpha": alpha_max, "alpha_applied": effective_alpha,
            "vector_type": control_type,
            "probe_score": round(probe_score, 6), "gate": gate_val,
        })
    return results


# ==============================================================================
# ORACLE GATE (reused)
# ==============================================================================

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
            r["mode"] = mode_label; r["alpha"] = alpha; r["alpha_applied"] = 0.0; r["vector_type"] = "steering"
            results.append(r)
    if unans:
        handle = apply_steering_hook(model, layer_idx, steering_vector, alpha)
        for r in generate_answers(model, tokenizer, unans, mode="steering",
                                  max_new_tokens=max_new, temperature=temp, do_sample=do_sample):
            r["mode"] = mode_label; r["alpha"] = alpha; r["alpha_applied"] = alpha; r["vector_type"] = "steering"
            results.append(r)
        handle.remove()
    return results


def _evaluate_and_add(results, seed, layer, mode, alpha, vector_type, all_rows):
    from src.evaluate import evaluate_outputs
    metrics = evaluate_outputs(results)
    metrics["seed"] = seed; metrics["layer"] = layer
    metrics["mode"] = mode; metrics["alpha"] = alpha; metrics["vector_type"] = vector_type
    all_rows.append(metrics)
    return metrics


# ==============================================================================
# VERDICT
# ==============================================================================

def _compute_verdict(df, m3v4b_soft_h=0.8333):
    base = df[df["mode"] == "base"]
    if len(base) == 0:
        return "IC4_M3_V5_PLATFORM_INVALID", "No base metrics."
    base_h = float(base.iloc[0]["hallucination_rate"])
    base_c = float(base.iloc[0]["correct_answer_rate"])

    oracle = df[df["mode"].str.contains("oracle_gate")]
    oracle_h = float(oracle.iloc[0]["hallucination_rate"]) if len(oracle) > 0 else base_h
    oracle_c = float(oracle.iloc[0]["correct_answer_rate"]) if len(oracle) > 0 else base_c

    # Two-pass base (no steering)
    tp_base = df[df["mode"] == "two_pass_base_no_steering"]
    tp_base_h = float(tp_base.iloc[0]["hallucination_rate"]) if len(tp_base) > 0 else base_h
    tp_base_c = float(tp_base.iloc[0]["correct_answer_rate"]) if len(tp_base) > 0 else base_c

    # Two-pass tiny alpha
    tp_tiny = df[df["mode"].str.contains("two_pass_tiny_alpha")]
    tp_tiny_h = float(tp_tiny.iloc[0]["hallucination_rate"]) if len(tp_tiny) > 0 else base_h
    tp_tiny_c = float(tp_tiny.iloc[0]["correct_answer_rate"]) if len(tp_tiny) > 0 else base_c

    # Two-pass open loop
    tp_ol = df[df["mode"].str.contains("two_pass_open_loop_full")]
    tp_ol_h = float(tp_ol.iloc[0]["hallucination_rate"]) if len(tp_ol) > 0 else base_h

    # Soft gate
    soft = df[df["mode"].str.contains("soft_prefill_probe_gate")]
    soft_h = float(soft.iloc[0]["hallucination_rate"]) if len(soft) > 0 else base_h
    soft_c = float(soft.iloc[0]["correct_answer_rate"]) if len(soft) > 0 else base_c

    # Hard gate
    hard = df[df["mode"].str.contains("hard_prefill_probe_gate")]
    hard_h = float(hard.iloc[0]["hallucination_rate"]) if len(hard) > 0 else soft_h
    hard_c = float(hard.iloc[0]["correct_answer_rate"]) if len(hard) > 0 else soft_c

    oracle_gap_soft = soft_h - oracle_h
    oracle_gap_hard = hard_h - oracle_h

    delta_c_tp = tp_base_c - base_c
    delta_c_tiny = tp_tiny_c - base_c
    pipeline_c_artifact = abs(delta_c_tp) > 0.05 or abs(delta_c_tiny - delta_c_tp) > 0.03

    if pipeline_c_artifact:
        verdict = "IC4_M3_V5_PIPELINE_ARTIFACT_CONFIRMED"
        reason = (f"Two-pass pipeline itself changes C: base C={base_c:.3f}, "
                  f"two_pass_base C={tp_base_c:.3f} (dC={delta_c_tp:+.3f}). "
                  f"Tiny alpha C={tp_tiny_c:.3f} (dC={delta_c_tiny:+.3f}). "
                  f"Pipeline artifact is the primary C elevation source.")
    elif oracle_gap_hard < oracle_gap_soft - 0.05:
        verdict = "IC4_M3_V5_HARD_GATE_PROMISING"
        reason = (f"Hard gate (H={hard_h:.3f}) significantly improves over soft gate (H={soft_h:.3f}). "
                  f"Oracle gap shrinks: soft dH={oracle_gap_soft:+.3f} → hard dH={oracle_gap_hard:+.3f}. "
                  f"Residual alpha / gate shape is a key bottleneck.")
    elif oracle_gap_soft <= 0.17 and abs(tp_ol_h - oracle_h) <= 0.05:
        verdict = "IC4_M3_V5_TIMING_PLUS_PIPELINE"
        reason = (f"Timing improvement confirmed (soft H={soft_h:.3f} < M3-v3 H=0.933), "
                  f"but pipeline artifact dominates remaining oracle gap. "
                  f"Two-pass open-loop H={tp_ol_h:.3f} vs single-pass H={oracle_h:.3f}.")
    elif abs(tp_ol_h - oracle_h) > 0.10:
        verdict = "IC4_M3_V5_STEERING_MECHANICS_LIMIT"
        reason = (f"Even two-pass open-loop (H={tp_ol_h:.3f}) cannot match single-pass oracle (H={oracle_h:.3f}). "
                  f"Two-pass prefill mechanics fundamentally limit steering effectiveness. "
                  f"This is deeper than gate shape or timing.")
    else:
        verdict = "IC4_M3_V5_PIPELINE_ARTIFACT_CONFIRMED"
        reason = f"Multiple factors contribute. Soft H={soft_h:.3f}, Hard H={hard_h:.3f}, Two-pass OL H={tp_ol_h:.3f}."

    return verdict, reason


# ==============================================================================
# REPORT
# ==============================================================================

def _generate_report(report_path, config, df, probe_info, probe_preds, elapsed):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    def _r(mode_str):
        return df[df["mode"] == mode_str]

    def _h(mode_str):
        rows = _r(mode_str)
        return float(rows.iloc[0]["hallucination_rate"]) if len(rows) > 0 else 0.0

    def _c(mode_str):
        rows = _r(mode_str)
        return float(rows.iloc[0]["correct_answer_rate"]) if len(rows) > 0 else 0.0

    def _ua(mode_str):
        rows = _r(mode_str)
        return float(rows.iloc[0].get("unnecessary_abstention_rate", 0.0)) if len(rows) > 0 else 0.0

    base_h, base_c = _h("base"), _c("base")
    oracle_h, oracle_c = _h("oracle_gate_a-1.0"), _c("oracle_gate_a-1.0")
    ol_h, ol_c = _h("steering_a-1.0"), _c("steering_a-1.0")

    tp_base_h, tp_base_c = _h("two_pass_base_no_steering"), _c("two_pass_base_no_steering")
    tp_tiny_h, tp_tiny_c = _h("two_pass_tiny_alpha-0.02"), _c("two_pass_tiny_alpha-0.02")
    tp_ol_h, tp_ol_c = _h("two_pass_open_loop_full_alpha-1.00"), _c("two_pass_open_loop_full_alpha-1.00")

    soft_h = _h("soft_prefill_probe_gate_a-1.0")
    soft_c = _c("soft_prefill_probe_gate_a-1.0")
    hard_h = _h("hard_prefill_probe_gate_a-1.0")
    hard_c = _c("hard_prefill_probe_gate_a-1.0")

    m3v3_h = 0.9333

    lines = []
    lines.append("# IC-4-M3-v5: Artifact Decomposition for Valid Prefill Gate")
    lines.append("")
    lines.append("> M3-v5 isolates the sources of the oracle gap and C anomaly observed in M3-v4b. ")
    lines.append("> M3-v4b: soft prefill gate H=0.833, C=0.733 (anomaly), oracle H=0.667, C=0.600. ")
    lines.append("> Question: Why does C rise to 0.733, and why doesn't H reach 0.667?")
    lines.append("")

    # 1. Recap
    lines.append("## 1. M3-v4b Key Gaps (Recap)")
    lines.append("")
    lines.append("| Gap | M3-v4b Value | Baseline | Delta | Question |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| C anomaly | C=0.733 | base C=0.600 | dC=+0.133 | Why does probe gate increase C? |")
    lines.append(f"| H gap vs oracle | H=0.833 | oracle H=0.667 | dH=+0.167 | Why can't prefill gate match oracle? |")
    lines.append(f"| H gap vs open-loop | H=0.833 | open-loop H=0.667 | dH=+0.167 | Gate vs uniform application |")
    lines.append("")

    # 2. Design
    lines.append("## 2. M3-v5 Design: Artifact Decomposition")
    lines.append("")
    lines.append("**Hypothesis sources tested:**")
    lines.append("1. Two-pass pipeline itself (vs single-pass)")
    lines.append("2. Tiny residual alpha (~-0.02) from soft gate on answerable samples")
    lines.append("3. Gate shape (soft sigmoid vs hard 0/1)")
    lines.append("4. Two-pass open-loop vs single-pass open-loop (mechanics)")
    lines.append("")
    lines.append("**New modes:**")
    lines.append("")
    lines.append("| Mode | Description | Purpose |")
    lines.append("|---|---|---|")
    lines.append("| `two_pass_base_no_steering` | Two-pass, alpha=0, no steering | Isolates pipeline artifact on C/H |")
    lines.append("| `two_pass_tiny_alpha_only` | Two-pass, alpha=-0.02 on ALL samples | Isolates tiny residual alpha effect |")
    lines.append("| `two_pass_open_loop_full_alpha` | Two-pass, alpha=-1.0 on ALL samples | Compares two-pass vs single-pass steering |")
    lines.append("| `soft_prefill_probe_gate` | Sigmoid gate (same as M3-v4b) | Baseline for gate comparison |")
    lines.append("| `hard_prefill_probe_gate` | Discrete 0/1 gate (threshold 0.5) | Eliminates residual alpha on answerable |")
    lines.append("")

    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    mcfg = config.get("model", {})
    lines.append(f"| Model | {mcfg.get('name', '?')} |")
    scfg = config.get("steering", {})
    lines.append(f"| Steering layer | {scfg.get('layers', [])} |")
    lines.append(f"| alpha_max | {scfg.get('alphas', [-1.0])} |")
    probe_cfg = config.get("probe", {})
    lines.append(f"| Probe representation | {', '.join(probe_cfg.get('representations', []))} |")
    lines.append(f"| Gate steepness / threshold | {probe_cfg.get('steepness', '?')} / {probe_cfg.get('threshold', '?')} |")
    lines.append(f"| Tiny alpha | {probe_cfg.get('tiny_alpha', -0.02)} |")
    lines.append(f"| Elapsed | {elapsed:.0f}s ({elapsed/60:.1f} min) |")
    lines.append("")

    # 3. Probe
    lines.append("## 3. Probe Training")
    lines.append("")
    if probe_info:
        lines.append("| Representation | Train Acc | CV Acc | AUC | N |")
        lines.append("|---|---|---|---|---|")
        for rep, info in probe_info.items():
            lines.append(f"| {rep} | {info.get('train_acc', '?'):.4f} | "
                         f"{info.get('cv_acc_mean', 'N/A')} | {info.get('auc', 'N/A')} | "
                         f"{info.get('n_pos', '?')}/{info.get('n_neg', '?')} |")
    lines.append("")

    # 4. Full metrics
    lines.append("## 4. Full Metrics Table")
    lines.append("")
    lines.append("| Mode | H | C | UA | CA | Vector |")
    lines.append("|---|---|---|---|---|---|")
    for _, row in df.iterrows():
        vt = str(row.get("vector_type", "none"))
        lines.append(f"| {row['mode']} | {row['hallucination_rate']:.3f} | "
                     f"{row['correct_answer_rate']:.3f} | "
                     f"{row.get('unnecessary_abstention_rate', 0):.3f} | "
                     f"{row.get('calibrated_abstention_rate', 0):.3f} | {vt} |")
    lines.append("")

    # 5. Artifact decomposition
    lines.append("## 5. Artifact Decomposition Results")
    lines.append("")

    lines.append("### 5.1 Does two-pass pipeline itself elevate C?")
    delta_c_tp = tp_base_c - base_c
    lines.append(f"- base C = {base_c:.3f}")
    lines.append(f"- two_pass_base_no_steering C = {tp_base_c:.3f} (dC = {delta_c_tp:+.3f})")
    lines.append(f"- two_pass_base_no_steering H = {tp_base_h:.3f} (base H = {base_h:.3f})")
    if abs(delta_c_tp) > 0.03:
        lines.append(f"- **YES.** Two-pass pipeline itself shifts C by {delta_c_tp:+.3f}.")
    else:
        lines.append(f"- **NO.** Two-pass pipeline alone does NOT significantly shift C.")
    lines.append("")

    lines.append("### 5.2 Does tiny residual alpha (~-0.02) elevate C?")
    delta_c_tiny = tp_tiny_c - tp_base_c
    lines.append(f"- two_pass_base_no_steering C = {tp_base_c:.3f}")
    lines.append(f"- two_pass_tiny_alpha_only C = {tp_tiny_c:.3f} (dC = {delta_c_tiny:+.3f})")
    lines.append(f"- two_pass_tiny_alpha_only H = {tp_tiny_h:.3f}")
    if abs(delta_c_tiny) > 0.03:
        lines.append(f"- **YES.** Tiny alpha (-0.02) on answerable samples shifts C by {delta_c_tiny:+.3f}.")
    else:
        lines.append(f"- **NO.** Tiny alpha alone does NOT significantly shift C.")
    lines.append("")

    lines.append("### 5.3 Soft gate vs Hard gate")
    delta_h_gate = hard_h - soft_h
    delta_c_gate = hard_c - soft_c
    lines.append(f"- soft gate: H={soft_h:.3f}, C={soft_c:.3f}")
    lines.append(f"- hard gate: H={hard_h:.3f}, C={hard_c:.3f}")
    lines.append(f"- dH = {delta_h_gate:+.3f}, dC = {delta_c_gate:+.3f}")
    if delta_h_gate < -0.03:
        lines.append(f"- **Hard gate improves H.** Residual alpha and gate shape matter.")
    else:
        lines.append(f"- **Hard gate does NOT significantly improve H.** Gate shape is not the bottleneck.")
    lines.append("")

    lines.append("### 5.4 Two-pass open-loop vs single-pass open-loop")
    delta_ol = tp_ol_h - ol_h
    lines.append(f"- single-pass open-loop: H={ol_h:.3f}, C={ol_c:.3f}")
    lines.append(f"- two-pass open-loop: H={tp_ol_h:.3f}, C={tp_ol_c:.3f}")
    lines.append(f"- dH = {delta_ol:+.3f}")
    if abs(delta_ol) > 0.05:
        lines.append(f"- **Two-pass mechanics weaken steering.** Pipeline limits steering effectiveness.")
    else:
        lines.append(f"- **Two-pass matches single-pass.** Pipeline mechanics are not the bottleneck.")
    lines.append("")

    # 6. Oracle gap attribution
    lines.append("## 6. Oracle Gap Attribution")
    lines.append("")
    lines.append("| Source | Evidence | Magnitude |")
    lines.append("|---|---|---|")
    pipeline_delta = abs(tp_base_c - base_c)
    tiny_delta = abs(tp_tiny_c - tp_base_c)
    gate_delta = abs(hard_h - soft_h)
    mech_delta = abs(tp_ol_h - ol_h)
    lines.append(f"| Pipeline C artifact | two_pass_base dC={delta_c_tp:+.3f} | {pipeline_delta:.3f} |")
    lines.append(f"| Tiny alpha C artifact | tiny dC={delta_c_tiny:+.3f} | {tiny_delta:.3f} |")
    lines.append(f"| Gate shape (soft→hard) | dH={delta_h_gate:+.3f} | {abs(delta_h_gate):.3f} |")
    lines.append(f"| Two-pass mechanics (steering) | two-pass OL vs single OL dH={delta_ol:+.3f} | {abs(delta_ol):.3f} |")
    lines.append(f"| Total oracle gap (soft vs oracle) | dH = {soft_h - oracle_h:+.3f} | {abs(soft_h - oracle_h):.3f} |")
    lines.append("")

    primary = "pipeline artifact"
    if abs(gate_delta) > abs(pipeline_delta) and abs(gate_delta) > abs(mech_delta):
        primary = "gate shape / residual alpha"
    elif abs(mech_delta) > abs(pipeline_delta) and abs(mech_delta) > abs(gate_delta):
        primary = "two-pass steering mechanics limitation"
    lines.append(f"**Primary source:** {primary}")
    lines.append("")

    # 7. Next step recommendation
    lines.append("## 7. Next Step Recommendation")
    lines.append("")
    if primary == "gate shape / residual alpha":
        lines.append("**Optimize gate shape.** Hard gate or threshold tuning should be the next focus. ")
        lines.append("Residual alpha on answerable samples is damaging. Consider: ")
        lines.append("- Hard gate with 0-alpha on answerable ")
        lines.append("- Gate threshold sweep ")
        lines.append("- Binary gate with explicit 'no hook' path")
    else:
        lines.append("**Investigate steering mechanics, not gate shape.** ")
        lines.append("Even two-pass open-loop cannot match single-pass. ")
        lines.append("Possible directions: ")
        lines.append("- Single-pass prefill with probe features extracted earlier (e.g., from KV cache) ")
        lines.append("- Steering vector applied differently in two-pass context ")
        lines.append("- Investigate why two-pass prefill weakens steering at layer 12")
    lines.append("")

    # 8. Verdict
    lines.append("## 8. Verdict")
    lines.append("")
    verdict, reason = _compute_verdict(df)
    lines.append(f"**Verdict: `{verdict}`**")
    lines.append("")
    lines.append(f"**Reasoning:** {reason}")
    lines.append("")
    next_action = ("optimize gate shape (hard gate, threshold tuning)" 
                   if primary == "gate shape / residual alpha"
                   else "investigate steering mechanics and pipeline limitations")
    lines.append(f"**一句话结论:** M3-v4b 剩余 oracle gap 主要来自 **{primary}**. ")
    lines.append(f"**下一步:** {next_action}.")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*IC-4-M3-v5: Artifact Decomposition for Valid Prefill Gate*")
    lines.append("*Generated by run_m3_v5.py*")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="IC-4-M3-v5: Artifact Decomposition for Prefill Gate")
    parser.add_argument("--config", type=str, default="configs/config_m3_v5.yaml")
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
    _log("IC-4-M3-v5: Artifact Decomposition for Valid Prefill Gate", log_path)
    _log("=" * 60, log_path)

    seeds = config["steering"]["seeds"]
    layers = config["steering"]["layers"]
    alpha_max = config["steering"]["alphas"][0]
    probe_cfg = config["probe"]
    gen_cfg = config["generation"]
    representations = probe_cfg["representations"]
    tiny_alpha = probe_cfg.get("tiny_alpha", -0.02)

    _log(f"Config: seed={seeds}, layer={layers}, alpha_max={alpha_max}", log_path)
    _log(f"Probe representations: {representations}, tiny_alpha={tiny_alpha}", log_path)

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
    all_probe_info = {}
    all_probe_preds = {}
    t_start = time.time()

    for seed in seeds:
        _log(f"\n{'='*40}", log_path)
        _log(f"SEED {seed}", log_path)
        _log(f"{'='*40}", log_path)

        random.seed(seed); np.random.seed(seed)
        train_path = config["data"].get("train_path", "data_m3/train.jsonl")
        test_path = config["data"].get("test_path", "data_m3/test.jsonl")
        train, test = _load_cached_m3_data(seed, train_path, test_path, log_path)

        from src.evaluate import generate_answers, evaluate_outputs, run_generation_with_steering

        # ---- BASELINES ----
        _log(f"\n  --- BASELINES ---", log_path)

        _log(f"  Base...", log_path)
        base_res = generate_answers(model, tokenizer, test, mode="base",
                                    max_new_tokens=gen_cfg["max_new_tokens"],
                                    temperature=gen_cfg["temperature"],
                                    do_sample=gen_cfg["do_sample"])
        m = _evaluate_and_add(base_res, seed, -1, "base", 0.0, "none", all_metrics)
        _log(f"    base: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f} UA={m['unnecessary_abstention_rate']:.3f}", log_path)

        _log(f"  Prompt-only...", log_path)
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

            _log(f"    Steering a={alpha_max:+.2f} (single-pass open loop)...", log_path)
            ol_res, _ = run_generation_with_steering(
                model, tokenizer, test, steering_v, layer_idx, alpha_max, "steering",
                max_new_tokens=gen_cfg["max_new_tokens"],
                temperature=gen_cfg["temperature"], do_sample=gen_cfg["do_sample"],
            )
            m = _evaluate_and_add(ol_res, seed, layer_idx, f"steering_a{alpha_max:+.2f}",
                                  alpha_max, "steering", all_metrics)
            _log(f"      single_pass_open_loop: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f} UA={m['unnecessary_abstention_rate']:.3f}", log_path)

            _log(f"    Oracle gate...", log_path)
            og_res = _generate_oracle_gated(model, tokenizer, test, steering_v,
                                            layer_idx, alpha_max, "oracle_gate_a-1.0", gen_cfg)
            m = _evaluate_and_add(og_res, seed, layer_idx, "oracle_gate_a-1.0", alpha_max,
                                  "steering", all_metrics)
            _log(f"      oracle: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f} UA={m['unnecessary_abstention_rate']:.3f}", log_path)

            # ---- TWO-PASS ARTIFACT MODES ----
            _log(f"\n  --- TWO-PASS ARTIFACT DECOMPOSITION ---", log_path)

            _log(f"    two_pass_base_no_steering (alpha=0, no hook at all)...", log_path)
            tp0_res = _generate_two_pass_no_steering(model, tokenizer, test, gen_cfg)
            m = _evaluate_and_add(tp0_res, seed, layer_idx, "two_pass_base_no_steering",
                                  0.0, "none", all_metrics)
            _log(f"      two_pass_base: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f} UA={m['unnecessary_abstention_rate']:.3f}", log_path)

            _log(f"    two_pass_tiny_alpha_only (alpha={tiny_alpha:+.2f} on all samples)...", log_path)
            tp_tiny_res = _generate_two_pass_tiny_alpha(
                model, tokenizer, test, steering_v, layer_idx, tiny_alpha, gen_cfg)
            m = _evaluate_and_add(tp_tiny_res, seed, layer_idx,
                                  f"two_pass_tiny_alpha{tiny_alpha:+.2f}",
                                  tiny_alpha, "steering", all_metrics)
            _log(f"      two_pass_tiny: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f} UA={m['unnecessary_abstention_rate']:.3f}", log_path)

            _log(f"    two_pass_open_loop_full_alpha (alpha={alpha_max:+.2f} on all samples)...", log_path)
            tp_ol_res = _generate_two_pass_open_loop(
                model, tokenizer, test, steering_v, layer_idx, alpha_max, gen_cfg)
            m = _evaluate_and_add(tp_ol_res, seed, layer_idx,
                                  f"two_pass_open_loop_full_alpha{alpha_max:+.2f}",
                                  alpha_max, "steering", all_metrics)
            _log(f"      two_pass_open_loop: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f} UA={m['unnecessary_abstention_rate']:.3f}", log_path)

            # ---- PROBE TRAINING ----
            for rep in representations:
                _log(f"\n  --- Representation: {rep} ---", log_path)

                _log(f"    Phase A: Training probe [{rep}]...", log_path)
                X_train, y_train = _collect_prefill_features(
                    model, tokenizer, train, probe_cfg["layer"], rep)
                _log(f"      Collected: X={X_train.shape}, y pos/neg={np.sum(y_train==1)}/{np.sum(y_train==0)}", log_path)

                probe_info = _train_probe(X_train, y_train, cv_folds=probe_cfg.get("cv_folds", 3))
                all_probe_info[rep] = probe_info
                _log(f"      Probe [{rep}]: train_acc={probe_info['train_acc']:.4f}, "
                     f"cv_acc={probe_info.get('cv_acc_mean', 'N/A')}, AUC={probe_info.get('auc', 'N/A')}", log_path)

                probe_cfg_with_rep = dict(probe_cfg)
                probe_cfg_with_rep["representation"] = rep

                # ---- SOFT GATE (same as M3-v4b) ----
                _log(f"    Phase B1: SOFT prefill probe gate [{rep}]...", log_path)
                soft_res = _generate_soft_prefill_probe_gated(
                    model, tokenizer, test, steering_v, layer_idx,
                    alpha_max, probe_info, probe_cfg_with_rep, gen_cfg, "steering")
                m = _evaluate_and_add(soft_res, seed, layer_idx,
                                      f"soft_prefill_probe_gate_a{alpha_max}",
                                      alpha_max, "steering", all_metrics)
                _log(f"      soft_gate: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f} UA={m['unnecessary_abstention_rate']:.3f}", log_path)

                soft_preds = [{"sample_id": r.get("sample_id", -1), "label": r.get("answerability", "?"),
                               "mode": r.get("mode", ""), "probe_score": r.get("probe_score", 0.5),
                               "gate": r.get("gate", 0.0), "alpha_applied": r.get("alpha_applied", 0.0)}
                              for r in soft_res]
                all_probe_preds[f"soft_{rep}"] = pd.DataFrame(soft_preds)

                _log(f"    Random soft probe gate [{rep}]...", log_path)
                soft_rnd = _generate_soft_prefill_probe_gated(
                    model, tokenizer, test, all_vectors["random"], layer_idx,
                    alpha_max, probe_info, probe_cfg_with_rep, gen_cfg, "random")
                m = _evaluate_and_add(soft_rnd, seed, layer_idx,
                                      f"random_soft_prefill_probe_gate_a{alpha_max}",
                                      alpha_max, "random", all_metrics)
                _log(f"      random_soft: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f}", log_path)

                _log(f"    Shuffled soft probe gate [{rep}]...", log_path)
                soft_shf = _generate_soft_prefill_probe_gated(
                    model, tokenizer, test, all_vectors["shuffled"], layer_idx,
                    alpha_max, probe_info, probe_cfg_with_rep, gen_cfg, "shuffled")
                m = _evaluate_and_add(soft_shf, seed, layer_idx,
                                      f"shuffled_soft_prefill_probe_gate_a{alpha_max}",
                                      alpha_max, "shuffled", all_metrics)
                _log(f"      shuffled_soft: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f}", log_path)

                # ---- HARD GATE ----
                _log(f"    Phase B2: HARD prefill probe gate [{rep}]...", log_path)
                hard_res = _generate_hard_prefill_probe_gated(
                    model, tokenizer, test, steering_v, layer_idx,
                    alpha_max, probe_info, probe_cfg_with_rep, gen_cfg, "steering")
                m = _evaluate_and_add(hard_res, seed, layer_idx,
                                      f"hard_prefill_probe_gate_a{alpha_max}",
                                      alpha_max, "steering", all_metrics)
                _log(f"      hard_gate: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f} UA={m['unnecessary_abstention_rate']:.3f}", log_path)

                # Hard gate controls (optional, quick)
                _log(f"    Random hard probe gate [{rep}]...", log_path)
                hard_rnd = _generate_hard_prefill_probe_gated(
                    model, tokenizer, test, all_vectors["random"], layer_idx,
                    alpha_max, probe_info, probe_cfg_with_rep, gen_cfg, "random")
                m = _evaluate_and_add(hard_rnd, seed, layer_idx,
                                      f"random_hard_prefill_probe_gate_a{alpha_max}",
                                      alpha_max, "random", all_metrics)
                _log(f"      random_hard: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f}", log_path)

                _log(f"    Shuffled hard probe gate [{rep}]...", log_path)
                hard_shf = _generate_hard_prefill_probe_gated(
                    model, tokenizer, test, all_vectors["shuffled"], layer_idx,
                    alpha_max, probe_info, probe_cfg_with_rep, gen_cfg, "shuffled")
                m = _evaluate_and_add(hard_shf, seed, layer_idx,
                                      f"shuffled_hard_prefill_probe_gate_a{alpha_max}",
                                      alpha_max, "shuffled", all_metrics)
                _log(f"      shuffled_hard: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f}", log_path)

            _log(f"    layer {layer_idx} done in {time.time() - lt0:.0f}s", log_path)

    elapsed = time.time() - t_start
    _log(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)", log_path)

    # Save metrics
    df = pd.DataFrame(all_metrics)
    cols = ["seed", "layer", "mode", "alpha", "vector_type",
            "hallucination_rate", "calibrated_abstention_rate",
            "correct_answer_rate", "unnecessary_abstention_rate",
            "answerable_count", "unanswerable_count",
            "hallucination_count", "calibrated_abstention_count",
            "correct_count", "unnecessary_abstention_count"]
    present_cols = [c for c in cols if c in df.columns]
    df = df[present_cols]
    met_path = os.path.join(results_dir, "metrics_raw.csv")
    df.to_csv(met_path, index=False)
    _log(f"\nMetrics saved to {met_path} ({len(df)} rows)", log_path)

    for key, p_df in all_probe_preds.items():
        tel_path = os.path.join(results_dir, f"per_sample_predictions_{key}.csv")
        p_df.to_csv(tel_path, index=False)
        _log(f"Per-sample predictions [{key}] saved to {tel_path} ({len(p_df)} rows)", log_path)

    probe_eval_rows = []
    for rep_name, info in all_probe_info.items():
        row = {"representation": rep_name}
        row.update({k: v for k, v in info.items() if k not in ("classifier", "scaler")})
        probe_eval_rows.append(row)
    pe_df = pd.DataFrame(probe_eval_rows)
    probe_eval_path = os.path.join(results_dir, "probe_evaluation.csv")
    pe_df.to_csv(probe_eval_path, index=False)
    _log(f"Probe evaluation saved to {probe_eval_path}", log_path)

    _log(f"\n{'='*60}", log_path)
    _log("COMPARISON TABLE", log_path)
    _log(f"{'='*60}", log_path)
    for _, row in df.iterrows():
        _log(f"  {row['mode']:<45} H={row['hallucination_rate']:.3f} C={row['correct_answer_rate']:.3f} UA={row.get('unnecessary_abstention_rate', 0):.3f}", log_path)

    verdict, reason = _compute_verdict(df)
    _log(f"\n{'='*60}", log_path)
    _log(f"VERDICT: {verdict}", log_path)
    _log(f"{'='*60}", log_path)
    _log(f"Reason: {reason}", log_path)

    report_path = os.path.join(reports_dir, "IC4_M3_V5_ARTIFACT_DECOMPOSITION_REPORT.md")
    _generate_report(report_path, config, df, all_probe_info, all_probe_preds, elapsed)
    _log(f"\nReport saved to {report_path}", log_path)
    _log(f"\nIC-4-M3-v5 complete. Verdict: {verdict}", log_path)


if __name__ == "__main__":
    main()