"""
IC-4-M3-v4: Prefill / Token-0 Probe Gate.
Tests whether M3-v3's failure is from late intervention timing.

Design:
  1. Run prefill forward (prompt only, no generation) to get hidden states
  2. Extract representation from prefill hidden states:
     - last_prompt_token
     - mean_pooled
     - question_span_pooled
  3. Train logistic regression probe: answerable vs unanswerable
  4. At token 0 (before any generation):
       probe_score = P(answerable)
       gate = sigmoid(steepness * (threshold - probe_score))
       alpha = alpha_max * gate
  5. Apply steering hook from the first generated token onward

Modes: base, prompt_only, steering_a-1.0, oracle_gate, prefill_probe_gate,
       random_prefill_probe_gate, shuffled_prefill_probe_gate
       (for each representation)

Key question: Does prefill/token-0 gate outperform M3-v3 token-4 gate?

Usage:
    python -m src.run_m3_v4 --config configs/config_m3_v4.yaml
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
    X_list = []
    y_list = []

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


def _generate_prefill_probe_gated(
    model, tokenizer, test_data, steering_vector, layer_idx,
    alpha_max, probe_info, probe_cfg, gen_cfg, control_type="steering",
):
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
            outputs = model(**inputs, use_cache=True)

        hs = outputs.hidden_states[probe_cfg["layer"] + 1][0]

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

        X = probe_info["scaler"].transform(pooled.reshape(1, -1))
        proba = probe_info["classifier"].predict_proba(X)[0, 1]
        probe_score = float(proba)

        gate_val = 1.0 / (1.0 + math.exp(-steepness * (threshold - probe_score)))
        effective_alpha = alpha_max * gate_val

        handle = None
        if abs(effective_alpha) > 0.001 and steering_vector is not None:
            handle = apply_steering_hook(model, layer_idx, steering_vector, effective_alpha)

        generated_ids = []
        past_key_values = outputs.past_key_values
        current_input = input_ids

        for step in range(max_new):
            with torch.no_grad():
                if past_key_values is not None:
                    current_input = current_input[:, -1:]
                out = model(input_ids=current_input, past_key_values=past_key_values, use_cache=True)
                past_key_values = out.past_key_values
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
            **sample,
            "sample_id": sid,
            "generated_output": answer,
            "mode": f"prefill_probe_gate_a{alpha_max}_{representation}",
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


def _compute_verdict(df, m3v3_probe_h=0.9333):
    base = df[df["mode"] == "base"]
    if len(base) == 0:
        return "IC4_M3_V4_PLATFORM_INVALID", "No base metrics."

    base_h = float(base.iloc[0]["hallucination_rate"])

    oracle = df[df["mode"].str.contains("oracle_gate")]
    oracle_h = float(oracle.iloc[0]["hallucination_rate"]) if len(oracle) > 0 else base_h

    probe_modes = df[df["mode"].str.contains("prefill_probe_gate") &
                     ~df["mode"].str.contains("random") &
                     ~df["mode"].str.contains("shuffled")]
    if len(probe_modes) == 0:
        return "IC4_M3_V4_PLATFORM_INVALID", "No prefill-probe gated results."

    best_probe = probe_modes.loc[probe_modes["hallucination_rate"].idxmin()]
    best_h = float(best_probe["hallucination_rate"])
    best_mode = str(best_probe["mode"])

    random_p = df[df["mode"].str.contains("random_prefill_probe")]
    shuffled_p = df[df["mode"].str.contains("shuffled_prefill_probe")]

    beats_control = True
    control_h = best_h
    for ctrl_df in [random_p, shuffled_p]:
        if len(ctrl_df) > 0:
            best_ctrl = ctrl_df.loc[ctrl_df["hallucination_rate"].idxmin()]
            ch = float(best_ctrl["hallucination_rate"])
            if best_h >= ch - 0.02:
                beats_control = False
                control_h = ch

    if not beats_control:
        return ("IC4_M3_V4_CONTROL_ARTIFACT",
                f"Best prefill probe gate ({best_mode}, H={best_h:.3f}) is indistinguishable from "
                f"random/shuffled controls (H={control_h:.3f}). "
                f"The probe signal is correct but prefill-level information is insufficient "
                f"to gate steering effectively at token 0.")

    gap_to_oracle = best_h - oracle_h
    if gap_to_oracle <= 0.05:
        return ("IC4_M3_V4_TIMING_CONFIRMED",
                f"Best prefill probe gate ({best_mode}, H={best_h:.3f}) approaches oracle gate (H={oracle_h:.3f}, "
                f"gap={gap_to_oracle:+.3f}). Timing IS the bottleneck: moving gate from token 4 to token 0 "
                f"recovers oracle-level behavior. M3-v3's failure is confirmed as a timing problem.")

    gap_vs_m3v3 = best_h - m3v3_probe_h
    if gap_vs_m3v3 < -0.05:
        return ("IC4_M3_V4_PREFILL_PROBE_PROMISING",
                f"Best prefill probe gate ({best_mode}, H={best_h:.3f}) significantly improves over "
                f"M3-v3 token-4 gate (H={m3v3_probe_h:.3f}, dH={gap_vs_m3v3:+.3f}). "
                f"Earlier intervention helps but doesn't fully close the gap to oracle (H={oracle_h:.3f}).")

    return ("IC4_M3_V4_PREFILL_PROBE_INSUFFICIENT",
            f"Best prefill probe gate ({best_mode}, H={best_h:.3f}) does not improve over "
            f"M3-v3 token-4 gate (H={m3v3_probe_h:.3f}). "
            f"Moving the gate to token 0 alone is insufficient. "
            f"The probe signal quality or the steering mechanism itself may be the bottleneck, "
            f"not just timing.")


def _generate_report(report_path, config, df, probe_info_all, probe_preds_all, elapsed):
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

    po_row = df[df["mode"] == "prompt_only"]
    po_h = float(po_row.iloc[0]["hallucination_rate"]) if len(po_row) > 0 else base_h
    po_c = float(po_row.iloc[0]["correct_answer_rate"]) if len(po_row) > 0 else base_c
    po_ua = float(po_row.iloc[0]["unnecessary_abstention_rate"]) if len(po_row) > 0 else base_ua

    probe_modes = df[df["mode"].str.contains("prefill_probe_gate") &
                     ~df["mode"].str.contains("random") &
                     ~df["mode"].str.contains("shuffled")]

    m3v3_probe_h = 0.9333

    lines = []
    lines.append("# IC-4-M3-v4: Prefill / Token-0 Probe Gate Report")
    lines.append("")

    lines.append("## 1. M3-v3 Recap: Why Token-4 Fails")
    lines.append("")
    lines.append("**M3-v3 Verdict: IC4_M3_V3_CONTROL_ARTIFACT**")
    lines.append("")
    lines.append("| Metric | base | oracle_gate | probe_gate_a-1.0 |")
    lines.append("|---|---|---|---|")
    lines.append("| H | 0.8667 | 0.6667 | 0.9333 |")
    lines.append("| C | 0.6000 | 0.6000 | 0.5667 |")
    lines.append("")
    lines.append("Key findings from M3-v3:")
    lines.append(f"- Probe trained perfectly: train_acc=1.0, cv_acc=0.9, AUC=1.0")
    lines.append(f"- Gate separation was strong: answerable probe_score ≈ 1, unanswerable ≈ 0")
    lines.append(f"- BUT behavioral outcome was terrible: H=0.9333 = indistinguishable from random/shuffled controls")
    lines.append(f"- Root cause: intervention at token 4 is too late — model trajectory is already committed")
    lines.append("")

    lines.append("## 2. M3-v4 Hypothesis")
    lines.append("")
    lines.append("**If M3-v3's failure is from timing (not signal quality), then moving the gate to token 0 "
                 "(prefill-level decision) should significantly improve behavioral outcomes.**")
    lines.append("")
    lines.append("M3-v4 tests this by:")
    lines.append("1. Using prefill/prompt hidden states (NO generated tokens) as probe input")
    lines.append("2. Making gate decision at token 0, before any generation")
    lines.append("3. Comparing three prefill representations")
    lines.append("")

    lines.append("## 3. M3-v4 Design")
    lines.append("")
    lines.append("1. Run prefill forward on prompt only → capture hidden states at layer 12")
    lines.append("2. Extract representation:")
    lines.append("   - `last_prompt_token`: hidden state at final prompt token position")
    lines.append("   - `mean_pooled`: average over all prompt token hidden states")
    lines.append("   - `question_span_pooled`: average over question-text tokens only")
    lines.append("3. `probe_score = P(answerable)` via logistic regression")
    lines.append("4. `gate = sigmoid(steepness * (threshold - probe_score))`")
    lines.append("   - probe_score ≈ 1 (answerable) → gate ≈ 0 → minimal steering")
    lines.append("   - probe_score ≈ 0 (unanswerable) → gate ≈ 1 → full steering")
    lines.append("5. `alpha = alpha_max * gate`, applied from the first generated token")
    lines.append("")

    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    probe_cfg = config.get("probe", {})
    lines.append(f"| Probe layer | {probe_cfg.get('layer', '?')} |")
    lines.append(f"| Representations | {', '.join(probe_cfg.get('representations', []))} |")
    lines.append(f"| Probe model | {probe_cfg.get('model', '?')} |")
    lines.append(f"| Decision timing | token 0 (prefill) |")
    lines.append(f"| Steepness | {probe_cfg.get('steepness', '?')} |")
    lines.append(f"| Threshold | {probe_cfg.get('threshold', '?')} |")
    lines.append("")

    lines.append("## 4. Probe Training Evaluation (per Representation)")
    lines.append("")
    lines.append("| Representation | Train Acc | CV Acc (mean) | CV Acc (std) | AUC | N pos/neg |")
    lines.append("|---|---|---|---|---|---|")
    for rep, info in probe_info_all.items():
        lines.append(f"| {rep} | {info.get('train_acc', '?'):.4f} | "
                     f"{info.get('cv_acc_mean', 'N/A')} | "
                     f"{info.get('cv_acc_std', 'N/A')} | "
                     f"{info.get('auc', 'N/A')} | "
                     f"{info.get('n_pos', '?')}/{info.get('n_neg', '?')} |")
    lines.append("")

    lines.append("## 5. Experiment Configuration")
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

    lines.append("## 6. Full Metrics Table")
    lines.append("")
    lines.append("| Mode | H | C | UA | CA | Vector |")
    lines.append("|---|---|---|---|---|---|")
    for _, row in df.iterrows():
        vt = str(row.get("vector_type", "none"))
        lines.append(f"| {row['mode']} | {row['hallucination_rate']:.3f} | "
                     f"{row['correct_answer_rate']:.3f} | {row['unnecessary_abstention_rate']:.3f} | "
                     f"{row.get('calibrated_abstention_rate', 0):.3f} | {vt} |")
    lines.append("")

    lines.append("## 7. Prefill Probe Gate vs Oracle Gate vs M3-v3")
    lines.append("")
    lines.append("| Mode | H | C | UA | dH_base | Gap to Oracle | Gap to M3-v3 |")
    lines.append("|---|---|---|---|---|---|---|")
    lines.append(f"| base | {base_h:.3f} | {base_c:.3f} | {base_ua:.3f} | -- | -- | -- |")
    lines.append(f"| prompt_only | {po_h:.3f} | {po_c:.3f} | {po_ua:.3f} | {po_h-base_h:+.3f} | {po_h-oracle_h:+.3f} | {po_h-m3v3_probe_h:+.3f} |")
    lines.append(f"| oracle_gate | {oracle_h:.3f} | {oracle_c:.3f} | {oracle_ua:.3f} | {oracle_h-base_h:+.3f} | -- | {oracle_h-m3v3_probe_h:+.3f} |")
    if len(open_row) > 0:
        lines.append(f"| steering_a-1.0 | {open_h:.3f} | {open_c:.3f} | {float(open_row.iloc[0]['unnecessary_abstention_rate']):.3f} | {open_h-base_h:+.3f} | {open_h-oracle_h:+.3f} | {open_h-m3v3_probe_h:+.3f} |")
    for _, row in df.iterrows():
        m = str(row["mode"])
        if "prefill_probe_gate" in m and "random" not in m and "shuffled" not in m:
            h = float(row["hallucination_rate"])
            c = float(row["correct_answer_rate"])
            ua = float(row["unnecessary_abstention_rate"])
            lines.append(f"| {m} | {h:.3f} | {c:.3f} | {ua:.3f} | {h-base_h:+.3f} | {h-oracle_h:+.3f} | {h-m3v3_probe_h:+.3f} |")
    lines.append("")

    lines.append("## 8. Gate Telemetry (per Representation)")
    lines.append("")
    if probe_preds_all:
        for rep, p_df in probe_preds_all.items():
            if p_df is None or len(p_df) == 0:
                continue
            a_sub = p_df[p_df["label"] == "answerable"]
            u_sub = p_df[p_df["label"] == "unanswerable"]
            a_score = float(a_sub["probe_score"].mean()) if len(a_sub) > 0 else 0
            u_score = float(u_sub["probe_score"].mean()) if len(u_sub) > 0 else 0
            a_gate = float(a_sub["gate"].mean()) if len(a_sub) > 0 else 0
            u_gate = float(u_sub["gate"].mean()) if len(u_sub) > 0 else 0
            sep_score = abs(a_score - u_score)
            sep_gate = abs(a_gate - u_gate)
            lines.append(f"### {rep}")
            lines.append("")
            lines.append("| Metric | Answerable | Unanswerable | Separation |")
            lines.append("|---|---|---|---|")
            lines.append(f"| probe_score (P(answerable)) mean | {a_score:.4f} | {u_score:.4f} | {sep_score:.4f} |")
            lines.append(f"| gate mean | {a_gate:.4f} | {u_gate:.4f} | {sep_gate:.4f} |")
            lines.append("")
    else:
        lines.append("No telemetry data available.")
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
    lines.append("### 10.1 Does prefill/token-0 gate outperform M3-v3 token-4 gate?")

    best_probe = probe_modes.loc[probe_modes["hallucination_rate"].idxmin()] if len(probe_modes) > 0 else None
    if best_probe is not None:
        best_h = float(best_probe["hallucination_rate"])
        gap = best_h - m3v3_probe_h
        if gap < -0.02:
            lines.append(f"**YES.** Best prefill gate ({best_probe['mode']}, H={best_h:.3f}) significantly "
                         f"outperforms M3-v3 token-4 gate (H={m3v3_probe_h:.3f}, dH={gap:+.3f}).")
            lines.append(f"Moving the gate from token 4 to token 0 improves behavioral outcome.")
        elif gap <= 0.02:
            lines.append(f"**MARGINAL.** Best prefill gate ({best_probe['mode']}, H={best_h:.3f}) is "
                         f"comparable to M3-v3 token-4 gate (H={m3v3_probe_h:.3f}). "
                         f"Timing alone is not the full story.")
        else:
            lines.append(f"**NO.** Best prefill gate ({best_probe['mode']}, H={best_h:.3f}) is "
                         f"WORSE than M3-v3 token-4 gate (H={m3v3_probe_h:.3f}). "
                         f"The prefill representation may carry weaker signal than the trajectory-state representation.")

    lines.append("")
    lines.append("### 10.2 Does it beat random/shuffled controls?")

    random_p = df[df["mode"].str.contains("random_prefill_probe")]
    shuffled_p = df[df["mode"].str.contains("shuffled_prefill_probe")]
    beats = True
    for ctrl_df in [random_p, shuffled_p]:
        if len(ctrl_df) > 0:
            ctrl_best = ctrl_df.loc[ctrl_df["hallucination_rate"].idxmin()]
            if best_probe is not None and float(best_probe["hallucination_rate"]) >= float(ctrl_best["hallucination_rate"]) - 0.02:
                beats = False
    if beats and best_probe is not None:
        lines.append(f"**YES.** Best probe gate (H={best_h:.3f}) beats controls. The probe signal provides real gating value.")
    else:
        lines.append(f"**NO.** Best probe gate is indistinguishable from controls. Prefill signal alone is insufficient.")

    lines.append("")
    lines.append("### 10.3 Does this support 'timing is the main bottleneck' hypothesis?")
    gap_to_oracle = best_h - oracle_h if best_probe is not None else 999
    if best_probe is not None and gap_to_oracle <= 0.05:
        lines.append(f"**YES.** Prefill gate (H={best_h:.3f}) ≈ oracle gate (H={oracle_h:.3f}). "
                     f"Timing was confirmed as the primary bottleneck.")
    elif best_probe is not None and gap_to_oracle <= 0.20:
        lines.append(f"**PARTIALLY.** Prefill gate (H={best_h:.3f}) is closer to oracle (H={oracle_h:.3f}) "
                     f"than M3-v3 (H={m3v3_probe_h:.3f}), but a gap remains. "
                     f"Timing is a bottleneck but not the only one.")
    else:
        lines.append(f"**NO.** Prefill gate does not close the gap to oracle. "
                     f"Timing is NOT the main bottleneck — the nature of the signal or the steering mechanism "
                     f"itself requires deeper investigation.")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*IC-4-M3-v4: Prefill / Token-0 Probe Gate*")
    lines.append("*Generated by run_m3_v4*")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="IC-4-M3-v4: Prefill / Token-0 Probe Gate")
    parser.add_argument("--config", type=str, default="configs/config_m3_v4.yaml")
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
    _log("IC-4-M3-v4: Prefill / Token-0 Probe Gate", log_path)
    _log("=" * 60, log_path)

    seeds = config["steering"]["seeds"]
    layers = config["steering"]["layers"]
    alpha_max = config["steering"]["alphas"][0] if config["steering"]["alphas"] else -1.0
    probe_cfg = config["probe"]
    gen_cfg = config["generation"]
    representations = probe_cfg["representations"]

    _log(f"Config: seed={seeds}, layer={layers}, alpha_max={alpha_max}", log_path)
    _log(f"Probe representations: {representations}", log_path)

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

        random.seed(seed)
        np.random.seed(seed)

        train_path = config["data"].get("train_path", "data_m3/train.jsonl")
        test_path = config["data"].get("test_path", "data_m3/test.jsonl")
        train, test = _load_cached_m3_data(seed, train_path, test_path, log_path)

        from src.evaluate import generate_answers, evaluate_outputs, run_generation_with_steering

        _log(f"\n  Base...", log_path)
        base_res = generate_answers(model, tokenizer, test, mode="base",
                                    max_new_tokens=gen_cfg["max_new_tokens"],
                                    temperature=gen_cfg["temperature"],
                                    do_sample=gen_cfg["do_sample"])
        m = _evaluate_and_add(base_res, seed, -1, "base", 0.0, "none", all_metrics)
        _log(f"    base: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f} UA={m['unnecessary_abstention_rate']:.3f}", log_path)

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

            _log(f"\n    Steering a={alpha_max:+.2f} (open loop)...", log_path)
            ol_res, _ = run_generation_with_steering(
                model, tokenizer, test, steering_v, layer_idx,
                alpha_max, "steering",
                max_new_tokens=gen_cfg["max_new_tokens"],
                temperature=gen_cfg["temperature"], do_sample=gen_cfg["do_sample"],
            )
            m = _evaluate_and_add(ol_res, seed, layer_idx, f"steering_a{alpha_max}",
                                  alpha_max, "steering", all_metrics)
            _log(f"      steering_a{alpha_max:+.2f}: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f} UA={m['unnecessary_abstention_rate']:.3f}", log_path)

            _log(f"\n    Oracle gate...", log_path)
            og_res = _generate_oracle_gated(model, tokenizer, test, steering_v,
                                            layer_idx, alpha_max, "oracle_gate_a-1.0", gen_cfg)
            m = _evaluate_and_add(og_res, seed, layer_idx, "oracle_gate_a-1.0", alpha_max,
                                  "steering", all_metrics)
            _log(f"      oracle_a{alpha_max:+.2f}: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f} UA={m['unnecessary_abstention_rate']:.3f}", log_path)

            for rep in representations:
                _log(f"\n  --- Representation: {rep} ---", log_path)

                _log(f"    Phase A: Training probe [{rep}]...", log_path)
                X_train, y_train = _collect_prefill_features(
                    model, tokenizer, train, probe_cfg["layer"], rep
                )
                _log(f"      Collected: X={X_train.shape}, y pos/neg={np.sum(y_train==1)}/{np.sum(y_train==0)}", log_path)

                probe_info = _train_probe(X_train, y_train, cv_folds=probe_cfg.get("cv_folds", 3))
                all_probe_info[rep] = probe_info
                _log(f"      Probe [{rep}]: train_acc={probe_info['train_acc']:.4f}, "
                     f"cv_acc={probe_info.get('cv_acc_mean', 'N/A')}, AUC={probe_info.get('auc', 'N/A')}", log_path)

                probe_cfg_with_rep = dict(probe_cfg)
                probe_cfg_with_rep["representation"] = rep

                _log(f"    Phase B: Prefill probe gate [{rep}]...", log_path)
                probe_res = _generate_prefill_probe_gated(
                    model, tokenizer, test, steering_v, layer_idx,
                    alpha_max, probe_info, probe_cfg_with_rep, gen_cfg, "steering",
                )
                mode_label = f"prefill_probe_gate_a{alpha_max}_{rep}"
                m = _evaluate_and_add(probe_res, seed, layer_idx, mode_label,
                                      alpha_max, "steering", all_metrics)
                _log(f"      {mode_label}: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f} UA={m['unnecessary_abstention_rate']:.3f}", log_path)

                preds = []
                for r in probe_res:
                    preds.append({
                        "sample_id": r.get("sample_id", -1),
                        "label": r.get("answerability", "?"),
                        "mode": r.get("mode", ""),
                        "probe_score": r.get("probe_score", 0.5),
                        "gate": r.get("gate", 0.0),
                        "alpha_applied": r.get("alpha_applied", 0.0),
                    })
                all_probe_preds[rep] = pd.DataFrame(preds)

                _log(f"    Random prefill probe gate [{rep}]...", log_path)
                random_res = _generate_prefill_probe_gated(
                    model, tokenizer, test, all_vectors["random"], layer_idx,
                    alpha_max, probe_info, probe_cfg_with_rep, gen_cfg, "random",
                )
                m = _evaluate_and_add(random_res, seed, layer_idx,
                                      f"random_prefill_probe_gate_a{alpha_max}_{rep}",
                                      alpha_max, "random", all_metrics)
                _log(f"      random_{rep}: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f} UA={m['unnecessary_abstention_rate']:.3f}", log_path)

                _log(f"    Shuffled prefill probe gate [{rep}]...", log_path)
                shuffled_res = _generate_prefill_probe_gated(
                    model, tokenizer, test, all_vectors["shuffled"], layer_idx,
                    alpha_max, probe_info, probe_cfg_with_rep, gen_cfg, "shuffled",
                )
                m = _evaluate_and_add(shuffled_res, seed, layer_idx,
                                      f"shuffled_prefill_probe_gate_a{alpha_max}_{rep}",
                                      alpha_max, "shuffled", all_metrics)
                _log(f"      shuffled_{rep}: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f} UA={m['unnecessary_abstention_rate']:.3f}", log_path)

            _log(f"    layer {layer_idx} done in {time.time() - lt0:.0f}s", log_path)

    elapsed = time.time() - t_start
    _log(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)", log_path)

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

    for rep, p_df in all_probe_preds.items():
        tel_path = os.path.join(results_dir, f"per_sample_predictions_{rep}.csv")
        p_df.to_csv(tel_path, index=False)
        _log(f"Per-sample predictions [{rep}] saved to {tel_path} ({len(p_df)} rows)", log_path)

    probe_eval_rows = []
    for rep, info in all_probe_info.items():
        row = {"representation": rep}
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
        _log(f"  {row['mode']:<50} H={row['hallucination_rate']:.3f} C={row['correct_answer_rate']:.3f} UA={row['unnecessary_abstention_rate']:.3f}", log_path)

    verdict, reason = _compute_verdict(df)
    _log(f"\n{'='*60}", log_path)
    _log(f"VERDICT: {verdict}", log_path)
    _log(f"{'='*60}", log_path)
    _log(f"Reason: {reason}", log_path)

    report_path = os.path.join(reports_dir, "IC4_M3_V4_PREFILL_GATE_REPORT.md")
    _generate_report(report_path, config, df, all_probe_info, all_probe_preds, elapsed)
    _log(f"\nReport saved to {report_path}", log_path)
    _log(f"\nIC-4-M3-v4 complete. Verdict: {verdict}", log_path)


if __name__ == "__main__":
    main()