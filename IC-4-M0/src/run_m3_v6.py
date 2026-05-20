"""
IC-4-M3-v6: Single-Pass Hook-Based Gate (Minimal Validation).

Eliminates the two-pass prefill artifact by performing probe extraction
and gate decision inside a forward hook during the single prefill pass.

Key design:
  - Forward hook at steering target layer L captures post-layer hidden states
  - Probe runs on `outputs[0]` (same hidden states probe was trained on)
  - Gate decision modifies layer output in-place (steering enters KV cache)
  - Single forward pass: no second prefill, no discarded KV cache

Modes (minimal set):
  - base
  - single_pass_open_loop_a-1.0
  - oracle_gate_a-1.0
  - single_pass_hard_gate_a-1.0   (hook-based, hard gate)
  - random_single_pass_hard_gate_a-1.0
  - shuffled_single_pass_hard_gate_a-1.0

Success criteria:
  - C returns to ~0.600 (base/oracle level, not 0.733)
  - H significantly better than M3-v5 hard gate H=0.800
  - Ideally approaching oracle H=0.667
  - Real gate clearly beats random/shuffled

Usage:
    python -m src.run_m3_v6 --config configs/config_m3_v6.yaml
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.run_m2 import load_config


def _log(msg, log_file=None):
    print(msg, flush=True)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()


def _find_transformer_layer(model, layer_idx):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers[layer_idx]
    raise ValueError("Cannot locate transformer layers.")


def _load_cached_m3_data(seed, train_path, test_path, log):
    from src.data_builder import load_jsonl
    train_final = train_path.replace(".jsonl", f"_s{seed}.jsonl")
    test_final = test_path.replace(".jsonl", f"_s{seed}.jsonl")
    if not os.path.exists(train_final) or not os.path.exists(test_final):
        raise FileNotFoundError(f"M3 data not found at {train_final} / {test_final}")
    train = load_jsonl(train_final)
    test = load_jsonl(test_final)
    na_t = sum(1 for s in train if s.get("answerability") == "answerable")
    na_s = sum(1 for s in test if s.get("answerability") == "answerable")
    _log(f"  seed={seed}: train {na_t}A+{len(train)-na_t}U, test {na_s}A+{len(test)-na_s}U", log)
    return train, test


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
        else:
            pooled = hs[-1, :].detach().cpu().float().numpy()
        X_list.append(pooled)
        y_list.append(1 if label == "answerable" else 0)
    return np.stack(X_list, axis=0), np.array(y_list, dtype=np.int32)


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
        "auc": float(auc_score) if auc_score is not None else None,
        "n_samples": len(y), "n_pos": int(np.sum(y == 1)), "n_neg": int(np.sum(y == 0)),
    }


# ==============================================================================
# SINGLE-PASS HOOK-BASED GATE (M3-v6 core innovation)
# ==============================================================================

def _make_single_pass_gate_hook(vec_tensor, probe_info, probe_cfg, alpha_max, representation):
    """
    Create a forward hook that, during the SINGLE prefill pass:
      1. Captures post-layer hidden states from `outputs[0]`
      2. Extracts last_prompt_token representation
      3. Runs probe → hard gate decision
      4. If gate=1 (unanswerable): adds alpha_max * v to output, steering enters KV cache
      5. If gate=0 (answerable): passes through unchanged
    """
    scaler = probe_info["scaler"]
    clf = probe_info["classifier"]
    threshold = probe_cfg["threshold"]

    def hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            h_full = outputs[0]
            h = h_full[0] if h_full.dim() == 3 else h_full
            hs = h
        else:
            h_full = outputs
            h = h_full[0] if h_full.dim() == 3 else h_full
            hs = h

        if representation == "last_prompt_token":
            pooled = hs[-1, :].detach().cpu().float().numpy()
        elif representation == "mean_pooled":
            pooled = hs.mean(dim=0).detach().cpu().float().numpy()
        else:
            pooled = hs[-1, :].detach().cpu().float().numpy()

        X = scaler.transform(pooled.reshape(1, -1))
        proba = clf.predict_proba(X)[0, 1]

        if proba >= threshold:
            return None
        else:
            v = vec_tensor.to(dtype=h_full.dtype, device=h_full.device)
            h_modified = h_full + alpha_max * v
            if isinstance(outputs, tuple):
                return (h_modified,) + outputs[1:]
            else:
                return h_modified

    return hook


def _generate_single_pass_hard_gate(
    model, tokenizer, test_data, steering_vector, layer_idx,
    alpha_max, probe_info, probe_cfg, gen_cfg, control_type="steering",
):
    """
    SINGLE-PASS PREFILL with HARD gate via forward hook.
    Uses model.generate() for proper generation (not manual loop).

    For each sample:
      1. Register forward hook at layer {layer_idx}
      2. Hook fires during model.generate()'s internal prefill
      3. Hook captures post-layer hidden states, runs probe, decides gate
      4. If gate=1: hook modifies layer output with alpha * v (→ steered KV cache)
      5. If gate=0: hook passes through (→ unsteered KV cache + generation)
      6. Hook remains registered during generation steps to sustain steering
      7. model.generate() handles all token-by-token generation internally
    """
    max_new = gen_cfg.get("max_new_tokens", 48)
    temperature = gen_cfg.get("temperature", 0.0)
    do_sample = gen_cfg.get("do_sample", False)
    device = next(model.parameters()).device
    representation = probe_cfg.get("representation", "last_prompt_token")
    threshold = probe_cfg["threshold"]
    vec_tensor = torch.from_numpy(steering_vector).to(device).float()

    layer_module = _find_transformer_layer(model, layer_idx)
    results = []

    for sid, sample in enumerate(test_data):
        context = sample.get("context", "")
        question = sample.get("question", "")
        label = sample.get("answerability", "?")
        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        probe_score = 0.5
        gate_val = 0.0
        effective_alpha = 0.0
        gate_decided = False

        def make_hook():
            nonlocal gate_decided, probe_score, gate_val, effective_alpha

            def hook(module, fn_inputs, fn_outputs):
                nonlocal gate_decided, probe_score, gate_val, effective_alpha

                if isinstance(fn_outputs, tuple):
                    h_full = fn_outputs[0]
                else:
                    h_full = fn_outputs

                if not gate_decided:
                    h = h_full[0] if h_full.dim() == 3 else h_full

                    if representation == "last_prompt_token":
                        pooled = h[-1, :].detach().cpu().float().numpy()
                    elif representation == "mean_pooled":
                        pooled = h.mean(dim=0).detach().cpu().float().numpy()
                    else:
                        pooled = h[-1, :].detach().cpu().float().numpy()

                    X = probe_info["scaler"].transform(pooled.reshape(1, -1))
                    proba = probe_info["classifier"].predict_proba(X)[0, 1]
                    probe_score = float(proba)
                    gate_decided = True

                    if probe_score >= threshold:
                        gate_val = 0.0
                        effective_alpha = 0.0
                        return None
                    else:
                        gate_val = 1.0
                        effective_alpha = alpha_max

                if abs(effective_alpha) > 0.001:
                    v = vec_tensor.to(dtype=h_full.dtype, device=h_full.device)
                    h_modified = h_full + effective_alpha * v
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
                **inputs,
                max_new_tokens=max_new,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
            )

        handle.remove()

        answer = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        results.append({
            **sample, "sample_id": sid, "generated_output": answer,
            "mode": f"single_pass_hard_gate_a{alpha_max}",
            "alpha": alpha_max, "alpha_applied": effective_alpha,
            "vector_type": control_type,
            "probe_score": round(probe_score, 6), "gate": gate_val,
        })

    return results


# ==============================================================================
# ORACLE GATE (reused, single-pass)
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

def _compute_verdict(df, m3v5_hard_h=0.800):
    base = df[df["mode"] == "base"]
    oracle = df[df["mode"].str.contains("oracle_gate")]
    hard = df[df["mode"].str.contains("single_pass_hard_gate")]
    random_hard = df[df["mode"].str.contains("random_single_pass_hard")]
    shuffled_hard = df[df["mode"].str.contains("shuffled_single_pass_hard")]

    base_c = float(base.iloc[0]["correct_answer_rate"]) if len(base) > 0 else 0.600
    oracle_h = float(oracle.iloc[0]["hallucination_rate"]) if len(oracle) > 0 else 0.667
    hard_h = float(hard.iloc[0]["hallucination_rate"]) if len(hard) > 0 else 0.800
    hard_c = float(hard.iloc[0]["correct_answer_rate"]) if len(hard) > 0 else 0.733
    random_h = float(random_hard.iloc[0]["hallucination_rate"]) if len(random_hard) > 0 else hard_h
    shuffled_h = float(shuffled_hard.iloc[0]["hallucination_rate"]) if len(shuffled_hard) > 0 else hard_h

    delta_c = hard_c - base_c
    delta_h_vs_m3v5 = hard_h - m3v5_hard_h
    delta_h_vs_oracle = hard_h - oracle_h
    beats_controls = hard_h < min(random_h, shuffled_h) - 0.05

    if abs(delta_c) <= 0.05 and delta_h_vs_oracle <= 0.05 and beats_controls:
        verdict = "IC4_M3_V6_SINGLE_PASS_SUCCESS"
        reason = (f"C anomaly eliminated (C={hard_c:.3f} vs base C={base_c:.3f}). "
                  f"H={hard_h:.3f} matches oracle H={oracle_h:.3f}. "
                  f"Single-pass hook-based gate achieves the goal.")
    elif abs(delta_c) <= 0.05 and delta_h_vs_m3v5 < -0.05 and beats_controls:
        verdict = "IC4_M3_V6_IMPROVEMENT_CONFIRMED"
        reason = (f"C anomaly eliminated (C={hard_c:.3f} vs base C={base_c:.3f}). "
                  f"H significantly improved from M3-v5 ({m3v5_hard_h:.3f}→{hard_h:.3f}, dH={delta_h_vs_m3v5:+.3f}), "
                  f"but oracle gap remains (dH={delta_h_vs_oracle:+.3f}).")
    elif abs(delta_c) <= 0.05 and not beats_controls:
        verdict = "IC4_M3_V6_CONTROL_ARTIFACT"
        reason = (f"C anomaly eliminated but no causal separation from controls: "
                  f"hard H={hard_h:.3f} vs random H={random_h:.3f} vs shuffled H={shuffled_h:.3f}.")
    elif abs(delta_c) > 0.05:
        verdict = "IC4_M3_V6_C_ANOMALY_PERSISTS"
        reason = (f"C anomaly persists: C={hard_c:.3f} vs base C={base_c:.3f} (dC={delta_c:+.3f}). "
                  f"Single-pass hook did not fully eliminate pipeline artifact.")
    else:
        verdict = "IC4_M3_V6_MIXED_RESULTS"
        reason = (f"H={hard_h:.3f} (vs M3-v5 {m3v5_hard_h:.3f}, vs oracle {oracle_h:.3f}). "
                  f"C={hard_c:.3f} (vs base {base_c:.3f}). Further investigation needed.")

    return verdict, reason


# ==============================================================================
# REPORT
# ==============================================================================

def _generate_report(report_path, config, df, probe_info, elapsed):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    def _r(mode_str):
        return df[df["mode"] == mode_str]

    def _h(mode_str):
        rows = _r(mode_str)
        return float(rows.iloc[0]["hallucination_rate"]) if len(rows) > 0 else 0.0

    def _c(mode_str):
        rows = _r(mode_str)
        return float(rows.iloc[0]["correct_answer_rate"]) if len(rows) > 0 else 0.0

    base_h, base_c = _h("base"), _c("base")
    ol_h, ol_c = _h("steering_a-1.00"), _c("steering_a-1.00")
    oracle_h, oracle_c = _h("oracle_gate_a-1.0"), _c("oracle_gate_a-1.0")
    hard_h = _h("single_pass_hard_gate_a-1.0")
    hard_c = _c("single_pass_hard_gate_a-1.0")
    rnd_h = _h("random_single_pass_hard_gate_a-1.0")
    shf_h = _h("shuffled_single_pass_hard_gate_a-1.0")

    m3v5_hard_h = 0.800
    m3v5_hard_c = 0.733
    m3v4b_soft_h = 0.833
    m3v4b_soft_c = 0.733
    m3v3_token4_h = 0.933

    lines = []
    lines.append("# IC-4-M3-v6: Single-Pass Hook-Based Gate (Minimal Validation)")
    lines.append("")
    lines.append("> Eliminates the two-pass prefill artifact by performing probe extraction")
    lines.append("> and gate decision inside a forward hook during a single prefill pass.")
    lines.append("")
    lines.append("**Architecture change from M3-v5:**")
    lines.append("- M3-v5: two-pass prefill — Pass 1 (unsteered, for probe), Pass 2 (steered, from scratch)")
    lines.append("- M3-v6: single-pass prefill — forward hook at layer L captures post-layer hiddens,")
    lines.append("  runs probe, decides gate, modifies layer output in-place. KV cache naturally")
    lines.append("  contains steering from prefill. No second forward pass.")
    lines.append("")

    lines.append("## 1. M3-v5 Pipeline Artifact (Recap)")
    lines.append("")
    lines.append("M3-v5 proved that the two-pass prefill pipeline:")
    lines.append(f"- Systematically elevates C: base C=0.600 → all two-pass C=0.733 (dC=+0.133)")
    lines.append(f"- Weakens steering effectiveness: single-pass open-loop H=0.667 → two-pass open-loop H=0.800 (dH=+0.133)")
    lines.append(f"- Hard gate at best reaches two-pass open-loop ceiling: H=0.800")
    lines.append(f"- ~80% of oracle gap attributed to two-pass pipeline mechanics")
    lines.append("")

    lines.append("## 2. M3-v6 Hypothesis")
    lines.append("")
    lines.append("**If the two-pass pipeline is the primary artifact, then switching to single-pass")
    lines.append("hook-based gate should:**")
    lines.append("1. Eliminate C anomaly (C → ~0.600)")
    lines.append("2. Improve H beyond M3-v5 ceiling of H=0.800")
    lines.append("3. Potentially approach oracle H=0.667")
    lines.append("")

    lines.append("## 3. Design")
    lines.append("")
    lines.append("**Single-pass hook-based hard gate:**")
    lines.append("1. Forward hook registered at steering target layer L")
    lines.append("2. Hook receives `outputs[0]` = post-layer-L hidden states (= same as probe training data)")
    lines.append("3. Extract `last_prompt_token` → scaler → classifier → probe_score")
    lines.append("4. Hard gate: probe_score ≥ 0.5 → answerable → no steering")
    lines.append("   probe_score < 0.5 → unanswerable → add alpha * v to layer output")
    lines.append("5. Modified output flows through subsequent layers → steered KV cache")
    lines.append("6. Single forward pass, no discarded cache, per-sample processing")
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
    lines.append(f"| Gate type | Hard (0 or 1, threshold {probe_cfg.get('threshold', '?')}) |")
    lines.append(f"| Pipeline | Single-pass, hook-based |")
    lines.append(f"| Elapsed | {elapsed:.0f}s ({elapsed/60:.1f} min) |")
    lines.append("")

    lines.append("## 4. Probe Training")
    lines.append("")
    if probe_info:
        lines.append("| Representation | Train Acc | CV Acc | AUC | N |")
        lines.append("|---|---|---|---|---|")
        for rep, info in probe_info.items():
            lines.append(f"| {rep} | {info.get('train_acc', '?'):.4f} | "
                         f"{info.get('cv_acc_mean', 'N/A')} | {info.get('auc', 'N/A')} | "
                         f"{info.get('n_pos', '?')}/{info.get('n_neg', '?')} |")
    lines.append("")

    lines.append("## 5. Full Metrics Table")
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

    lines.append("## 6. Cross-Experiment Comparison")
    lines.append("")
    lines.append("| Experiment | Mode | H | C | Pipeline |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| M3-v3 | token-4 probe gate | {m3v3_token4_h:.3f} | 0.567 | single-pass, token-4 hook |")
    lines.append(f"| M3-v4b | soft prefill gate | {m3v4b_soft_h:.3f} | {m3v4b_soft_c:.3f} | two-pass prefill |")
    lines.append(f"| M3-v5 | two-pass hard gate | {m3v5_hard_h:.3f} | {m3v5_hard_c:.3f} | two-pass prefill |")
    lines.append(f"| M3-v5 | two-pass open-loop | 0.800 | 0.733 | two-pass prefill |")
    lines.append(f"| **M3-v6** | **single-pass hard gate** | **{hard_h:.3f}** | **{hard_c:.3f}** | **single-pass hook** |")
    lines.append(f"| (ref) | base | {base_h:.3f} | {base_c:.3f} | single-pass |")
    lines.append(f"| (ref) | oracle gate | {oracle_h:.3f} | {oracle_c:.3f} | single-pass |")
    lines.append("")

    lines.append("## 7. Success Criteria Evaluation")
    lines.append("")
    delta_c = hard_c - base_c
    passed_c = abs(delta_c) <= 0.05
    lines.append(f"| Criterion | Target | Actual | Result |")
    lines.append(f"|---|---|---|---|")
    lines.append(f"| C anomaly eliminated | C ≈ {base_c:.3f} | C={hard_c:.3f} (dC={delta_c:+.3f}) | {'PASS' if passed_c else 'FAIL'} |")

    h_target = oracle_h
    h_gap = hard_h - h_target
    passed_h = h_gap <= 0.05
    lines.append(f"| H approaches oracle | H ≤ {h_target:.3f}+0.05 | H={hard_h:.3f} (dH={h_gap:+.3f}) | {'PASS' if passed_h else 'FAIL'} |")

    h_improved = hard_h < m3v5_hard_h - 0.03
    lines.append(f"| H improves over M3-v5 | H < {m3v5_hard_h-0.03:.3f} | H={hard_h:.3f} | {'PASS' if h_improved else 'FAIL'} |")

    beats = hard_h < min(rnd_h, shf_h) - 0.05
    lines.append(f"| Beats random/shuffled | H < min(random={rnd_h:.3f}, shuffled={shf_h:.3f}) | {'PASS' if beats else 'FAIL'} | {'PASS' if beats else 'FAIL'} |")
    lines.append("")

    # Oracle gap attribution
    lines.append("## 8. Oracle Gap Attribution (M3-v6)")
    lines.append("")
    remaining_gap = hard_h - oracle_h
    lines.append(f"- Oracle H: {oracle_h:.3f}")
    lines.append(f"- Single-pass hard gate H: {hard_h:.3f}")
    lines.append(f"- Remaining oracle gap: dH={remaining_gap:+.3f}")
    lines.append("")
    if abs(remaining_gap) <= 0.05:
        lines.append("**Single-pass hook-based gate achieves oracle-level performance.**")
        lines.append("Pipeline artifact confirmed as the sole bottleneck for prefill-level gating.")
    elif abs(remaining_gap) <= 0.10:
        lines.append("**Significant improvement but small gap remains.**")
        lines.append("Residual gap may come from: hook placement timing, representation choice,")
        lines.append("or an inherent ceiling of the steering vector direction itself.")
    else:
        lines.append("**Gap persists beyond pipeline artifact.**")
        lines.append("Even with single-pass, the hook-based gate cannot fully replicate oracle.")
        lines.append("This suggests additional factors beyond pipeline mechanics:")
        lines.append("- Steering vector direction effectiveness when applied mid-forward-pass vs pre-generation")
        lines.append("- Hook-based gate may alter residual stream differently than dedicated steering pass")
        lines.append("- Future work: test different hook placement, alpha values, or multi-layer steering")
    lines.append("")

    # Verdict
    verdict, reason = _compute_verdict(df)
    lines.append("## 9. Verdict")
    lines.append("")
    lines.append(f"**Verdict: `{verdict}`**")
    lines.append("")
    lines.append(f"**Reasoning:** {reason}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*IC-4-M3-v6: Single-Pass Hook-Based Gate (Minimal Validation)*")
    lines.append("*Generated by run_m3_v6.py*")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="IC-4-M3-v6: Single-Pass Hook-Based Gate")
    parser.add_argument("--config", type=str, default="configs/config_m3_v6.yaml")
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
    _log("IC-4-M3-v6: Single-Pass Hook-Based Gate (Minimal Validation)", log_path)
    _log("=" * 60, log_path)

    seeds = config["steering"]["seeds"]
    layers = config["steering"]["layers"]
    alpha_max = config["steering"]["alphas"][0]
    probe_cfg = config["probe"]
    gen_cfg = config["generation"]
    representations = probe_cfg["representations"]

    _log(f"Config: seed={seeds}, layer={layers}, alpha_max={alpha_max}", log_path)
    _log(f"Probe: {representations}, threshold={probe_cfg['threshold']}", log_path)
    _log(f"Pipeline: SINGLE-PASS HOOK-BASED GATE (no two-pass artifact)", log_path)

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

            _log(f"    Single-pass open-loop a={alpha_max:+.2f}...", log_path)
            ol_res, _ = run_generation_with_steering(
                model, tokenizer, test, steering_v, layer_idx, alpha_max, "steering",
                max_new_tokens=gen_cfg["max_new_tokens"],
                temperature=gen_cfg["temperature"], do_sample=gen_cfg["do_sample"],
            )
            m = _evaluate_and_add(ol_res, seed, layer_idx, f"steering_a{alpha_max:+.2f}",
                                  alpha_max, "steering", all_metrics)
            _log(f"      single_pass_ol: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f} UA={m['unnecessary_abstention_rate']:.3f}", log_path)

            _log(f"    Oracle gate...", log_path)
            og_res = _generate_oracle_gated(model, tokenizer, test, steering_v,
                                            layer_idx, alpha_max, "oracle_gate_a-1.0", gen_cfg)
            m = _evaluate_and_add(og_res, seed, layer_idx, "oracle_gate_a-1.0", alpha_max,
                                  "steering", all_metrics)
            _log(f"      oracle: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f} UA={m['unnecessary_abstention_rate']:.3f}", log_path)

        # Probe training (on all representations)
        for rep in representations:
            _log(f"\n  --- Probe Training: {rep} ---", log_path)
            _log(f"    Phase A: Training probe [{rep}]...", log_path)
            probe_layer_idx = layers[0]  # steer layer is the probe layer
            X_train, y_train = _collect_prefill_features(
                model, tokenizer, train, probe_layer_idx, rep)
            _log(f"      Collected: X={X_train.shape}, y pos/neg={np.sum(y_train==1)}/{np.sum(y_train==0)}", log_path)

            probe_info = _train_probe(X_train, y_train, cv_folds=probe_cfg.get("cv_folds", 3))
            all_probe_info[rep] = probe_info
            _log(f"      Probe [{rep}]: train_acc={probe_info['train_acc']:.4f}, "
                 f"cv_acc={probe_info.get('cv_acc_mean', 'N/A')}, AUC={probe_info.get('auc', 'N/A')}", log_path)

            probe_cfg_with_rep = dict(probe_cfg)
            probe_cfg_with_rep["representation"] = rep

            for layer_idx in layers:
                lt_layer = time.time()
                _log(f"\n    LAYER {layer_idx} — SINGLE-PASS HOOK-BASED HARD GATE [{rep}]", log_path)

                _log(f"    Single-pass hard gate (steering vector)...", log_path)
                hard_res = _generate_single_pass_hard_gate(
                    model, tokenizer, test, steering_v, layer_idx,
                    alpha_max, probe_info, probe_cfg_with_rep, gen_cfg, "steering")
                m = _evaluate_and_add(hard_res, seed, layer_idx,
                                      f"single_pass_hard_gate_a{alpha_max}",
                                      alpha_max, "steering", all_metrics)
                _log(f"      real_gate: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f} UA={m['unnecessary_abstention_rate']:.3f}", log_path)

                _log(f"    Single-pass hard gate (random vector)...", log_path)
                rnd_res = _generate_single_pass_hard_gate(
                    model, tokenizer, test, all_vectors["random"], layer_idx,
                    alpha_max, probe_info, probe_cfg_with_rep, gen_cfg, "random")
                m = _evaluate_and_add(rnd_res, seed, layer_idx,
                                      f"random_single_pass_hard_gate_a{alpha_max}",
                                      alpha_max, "random", all_metrics)
                _log(f"      random: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f}", log_path)

                _log(f"    Single-pass hard gate (shuffled vector)...", log_path)
                shf_res = _generate_single_pass_hard_gate(
                    model, tokenizer, test, all_vectors["shuffled"], layer_idx,
                    alpha_max, probe_info, probe_cfg_with_rep, gen_cfg, "shuffled")
                m = _evaluate_and_add(shf_res, seed, layer_idx,
                                      f"shuffled_single_pass_hard_gate_a{alpha_max}",
                                      alpha_max, "shuffled", all_metrics)
                _log(f"      shuffled: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f}", log_path)

                _log(f"    layer done in {time.time() - lt_layer:.0f}s", log_path)

    elapsed = time.time() - t_start
    _log(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)", log_path)

    df = pd.DataFrame(all_metrics)
    cols = ["seed", "layer", "mode", "alpha", "vector_type",
            "hallucination_rate", "calibrated_abstention_rate",
            "correct_answer_rate", "unnecessary_abstention_rate",
            "answerable_count", "unanswerable_count"]
    present_cols = [c for c in cols if c in df.columns]
    df = df[present_cols]
    met_path = os.path.join(results_dir, "metrics_raw.csv")
    df.to_csv(met_path, index=False)
    _log(f"\nMetrics saved to {met_path} ({len(df)} rows)", log_path)

    probe_eval_rows = []
    for rep, info in all_probe_info.items():
        row = {"representation": rep}
        row.update({k: v for k, v in info.items() if k not in ("classifier", "scaler")})
        probe_eval_rows.append(row)
    pe_df = pd.DataFrame(probe_eval_rows)
    pe_df.to_csv(os.path.join(results_dir, "probe_evaluation.csv"), index=False)

    _log(f"\n{'='*60}", log_path)
    _log("COMPARISON TABLE", log_path)
    _log(f"{'='*60}", log_path)
    for _, row in df.iterrows():
        _log(f"  {row['mode']:<45} H={row['hallucination_rate']:.3f} C={row['correct_answer_rate']:.3f}", log_path)

    verdict, reason = _compute_verdict(df)
    _log(f"\n{'='*60}", log_path)
    _log(f"VERDICT: {verdict}", log_path)
    _log(f"{'='*60}", log_path)
    _log(f"Reason: {reason}", log_path)

    report_path = os.path.join(reports_dir, "IC4_M3_V6_SINGLE_PASS_GATE_REPORT.md")
    _generate_report(report_path, config, df, all_probe_info, elapsed)
    _log(f"\nReport saved to {report_path}", log_path)
    _log(f"\nIC-4-M3-v6 complete. Verdict: {verdict}", log_path)


if __name__ == "__main__":
    main()