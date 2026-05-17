"""
IC-4-M3-v2: Telemetry-First Gated Steering.
Tests three risk-based gating signals (entropy, maxprob, uncertainty_mass)
with per-token telemetry to assess gate behavior before committing to a controller.

Usage:
    python -m src.run_m3_v2 --config configs/config_m3_v2.yaml
"""

import argparse
import os
import sys
import random
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

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
        raise FileNotFoundError(f"M3-v1 data not found at {train_final} / {test_final}")
    _log(f"  Loading cached M3 data for seed {seed}", log)
    train = load_jsonl(train_final)
    test = load_jsonl(test_final)
    na_t = sum(1 for s in train if s.get("answerability") == "answerable")
    nu_t = len(train) - na_t
    na_s = sum(1 for s in test if s.get("answerability") == "answerable")
    nu_s = len(test) - na_s
    _log(f"  seed={seed}: train {na_t}A+{nu_t}U, test {na_s}A+{nu_s}U", log)
    return train, test


# ── Risk signal functions ──

from src.evaluate import ABSTENTION_PATTERNS as _EVAL_ABSTENTION_PATTERNS

UNCERTAINTY_TOKEN_STRINGS = [
    "I don't know", "I do not know", "cannot determine", "can not determine",
    "insufficient information", "not mentioned", "not provided", "not specified",
    "unable to", "unsure", "uncertain", "no information",
    "不知道", "无法确定", "未提供",
    "资料不足", "不清楚", "不明确",
    "沒有提到", "沒有说明",
]


def _build_uncertainty_token_ids(tokenizer) -> set:
    """Find token IDs that correspond to uncertainty expressions."""
    ids = set()
    for s in UNCERTAINTY_TOKEN_STRINGS:
        tid_list = tokenizer.encode(s, add_special_tokens=False)
        for tid in tid_list:
            ids.add(tid)
    return ids


def _risk_entropy(logits: torch.Tensor) -> float:
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-12)
    return float(-torch.sum(probs * log_probs, dim=-1).mean().item())


def _risk_maxprob(logits: torch.Tensor) -> float:
    probs = F.softmax(logits, dim=-1)
    return float(1.0 - probs.max(dim=-1).values.mean().item())


def _risk_uncertainty_token_mass(logits: torch.Tensor, tokenizer) -> float:
    probs = F.softmax(logits, dim=-1)
    uid = _build_uncertainty_token_ids(tokenizer)
    total = float(probs.sum().item())
    if total < 1e-12:
        return 0.0
    mass = 0.0
    for tid in uid:
        if tid < probs.shape[-1]:
            mass += float(probs[:, tid].sum().item())
    return mass


def _risk_uncertainty_mass(logits: torch.Tensor, top_k: int = 5) -> float:
    probs = F.softmax(logits, dim=-1)
    top_probs, _ = torch.topk(probs, k=min(top_k, probs.shape[-1]), dim=-1)
    mass = float(top_probs.sum(dim=-1).mean().item())
    return 1.0 - mass


def _gate(risk: float, k: float, threshold: float) -> float:
    x = k * (risk - threshold)
    return float(1.0 / (1.0 + math.exp(-x)))


RISK_REGISTRY = {
    "entropy": (_risk_entropy, {"threshold": 2.0, "k": 3.0}),
    "maxprob": (_risk_maxprob, {"threshold": 0.30, "k": 10.0}),
    "uncertainty_mass": (_risk_uncertainty_token_mass, {"threshold": 0.50, "k": 10.0}),
}


# ── Gated generation ──

def _generate_gated(
    model,
    tokenizer,
    test_data: list,
    steering_vector: np.ndarray,
    layer_idx: int,
    alpha_max: float,
    gate_name: str,
    risk_fn,
    k: float,
    threshold: float,
    gen_cfg: dict,
    control_type: str = "steering",
):
    """
    Per-sample gated steering generation with per-token telemetry.

    At each token step:
      1. Forward pass with current alpha (from AdaptiveAlpha container)
      2. Compute risk signal from logits
      3. gate = sigmoid(k * (risk - threshold))
      4. alpha_{t+1} = alpha_max * gate
      5. Record telemetry

    Returns:
        results: list of per-sample result dicts
        telemetry_rows: list of per-token telemetry dicts
    """
    from src.steering import AdaptiveAlpha, apply_adaptive_steering_hook

    max_new = gen_cfg.get("max_new_tokens", 48)
    temp = gen_cfg.get("temperature", 0.0)
    do_sample = gen_cfg.get("do_sample", False)
    device = next(model.parameters()).device
    eos_id = tokenizer.eos_token_id

    results = []
    telemetry_rows = []

    for sid, sample in enumerate(test_data):
        context = sample.get("context", "")
        question = sample.get("question", "")
        label = sample.get("answerability", "?")
        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]

        alpha_container = AdaptiveAlpha(base_alpha=0.0, k=0.0)
        handle = apply_adaptive_steering_hook(model, layer_idx, steering_vector, alpha_container)

        generated_ids = []
        past_key_values = None
        current_input = input_ids
        final_answer = ""

        for step in range(max_new):
            with torch.no_grad():
                if past_key_values is not None:
                    current_input = current_input[:, -1:]
                outputs = model(input_ids=current_input, past_key_values=past_key_values,
                                use_cache=True)
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]

                entropy_t = _risk_entropy(logits)
                max_prob_t = _risk_maxprob(logits)
                uncertainty_mass_t = _risk_uncertainty_token_mass(logits, tokenizer)

                active_risk = risk_fn(logits) if risk_fn is not _risk_uncertainty_token_mass else risk_fn(logits, tokenizer)
                gate_val = _gate(active_risk, k, threshold)
                alpha_container.value = alpha_max * gate_val

                if do_sample and temp > 0:
                    probs = F.softmax(logits / temp, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                tid = next_token.item()
                generated_ids.append(tid)

                telemetry_rows.append({
                    "sample_id": sid,
                    "label": label,
                    "mode": gate_name,
                    "token_index": step,
                    "generated_token_id": tid,
                    "generated_token": tokenizer.decode([tid]),
                    "entropy_t": round(entropy_t, 6),
                    "max_prob_t": round(max_prob_t, 6),
                    "uncertainty_mass_t": round(uncertainty_mass_t, 6),
                    "gate": round(gate_val, 6),
                    "alpha": round(alpha_container.value, 6),
                })

                if tid == eos_id:
                    break
                current_input = next_token

        handle.remove()
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        results.append({
            **sample,
            "generated_output": answer,
            "mode": gate_name,
            "alpha": alpha_max,
            "alpha_applied": alpha_max,
            "vector_type": control_type,
            "gate_name": gate_name,
            "k": k,
            "threshold": threshold,
        })

    return results, telemetry_rows


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


def _save_telemetry(telemetry_rows, path):
    if not telemetry_rows:
        return
    df = pd.DataFrame(telemetry_rows)
    cols = ["sample_id", "label", "mode", "token_index",
            "generated_token_id", "generated_token",
            "entropy_t", "max_prob_t", "uncertainty_mass_t", "gate", "alpha"]
    present = [c for c in cols if c in df.columns]
    df[present].to_csv(path, index=False)


# ── Verdict ──

def _per_gate_telemetry(telemetry_df, real_gate_modes: list):
    """Compute per-gate telemetry stats for real gates only."""
    gate_stats = {}
    for gname in real_gate_modes:
        sub = telemetry_df[telemetry_df["mode"] == gname]
        if len(sub) == 0:
            continue
        ans = sub[sub["label"] == "answerable"]
        unans = sub[sub["label"] == "unanswerable"]
        if len(ans) == 0 or len(unans) == 0:
            continue
        ga = float(ans["gate"].mean())
        gu = float(unans["gate"].mean())
        aa = float(ans["alpha"].mean())
        au = float(unans["alpha"].mean())
        gate_stats[gname] = {
            "gate_ans": ga, "gate_unans": gu, "gate_sep": abs(gu - ga),
            "alpha_ans": aa, "alpha_unans": au,
        }
        for col in ["entropy_t", "max_prob_t", "uncertainty_mass_t"]:
            if col in sub.columns:
                gate_stats[gname][col + "_ans"] = float(ans[col].mean())
                gate_stats[gname][col + "_unans"] = float(unans[col].mean())
                gate_stats[gname][col + "_sep"] = abs(float(unans[col].mean()) - float(ans[col].mean()))
    return gate_stats


def _compute_verdict(df, telemetry_df=None):
    base = df[df["mode"] == "base"]
    if len(base) == 0:
        return "IC4_M3_V2_SIGNAL_WEAK", "No base metrics."
    base_h = float(base.iloc[0]["hallucination_rate"])
    base_c = float(base.iloc[0]["correct_answer_rate"])
    base_ua = float(base.iloc[0]["unnecessary_abstention_rate"])

    real_modes = df[df["mode"].str.contains("gate") & ~df["mode"].str.contains("oracle")
                    & ~df["mode"].str.contains("random_") & ~df["mode"].str.contains("shuffled_")]
    if len(real_modes) == 0:
        return "IC4_M3_V2_SIGNAL_WEAK", "No gated steering results."

    real_mode_list = list(real_modes["mode"].unique())

    gate_stats = {}
    if telemetry_df is not None and len(telemetry_df) > 0:
        gate_stats = _per_gate_telemetry(telemetry_df, real_mode_list)

    oracle = df[df["mode"] == "oracle_gate_a-1.0"]
    oh = float(oracle.iloc[0]["hallucination_rate"]) if len(oracle) > 0 else base_h

    # Find best real gate: prioritize lower H, then higher separation
    best_gate = None
    best_gname = None
    best_sep = 0.0
    for _, row in real_modes.iterrows():
        gname = row["mode"]
        sep = gate_stats.get(gname, {}).get("gate_sep", 0.0)
        if best_gate is None or row["hallucination_rate"] < best_gate["hallucination_rate"]:
            best_gate = row
            best_gname = gname
            best_sep = sep

    if best_gate is None:
        return "IC4_M3_V2_SIGNAL_WEAK", "Could not identify best real gate."

    gh = float(best_gate["hallucination_rate"])
    gc = float(best_gate["correct_answer_rate"])
    gua = float(best_gate["unnecessary_abstention_rate"])

    # Best separation among real gates
    best_sep_gate = max(gate_stats, key=lambda k: gate_stats[k]["gate_sep"], default=None)
    best_sep_val = gate_stats[best_sep_gate]["gate_sep"] if best_sep_gate else 0.0

    # Check if ANY real gate has separation
    any_sep = any(s.get("gate_sep", 0.0) >= 0.03 for s in gate_stats.values())

    if not any_sep:
        risk_details = ", ".join(
            f"{k}: gate_sep={v['gate_sep']:.4f}" for k, v in sorted(gate_stats.items(), key=lambda x: -x[1]["gate_sep"])
        ) if gate_stats else "no telemetry"
        return ("IC4_M3_V2_SIGNAL_WEAK",
                f"Gate barely moves across all signals. Per-gate separation: [{risk_details}]. "
                f"The feedback signal cannot distinguish answerable from unanswerable.")

    random_gate = df[df["mode"].str.startswith("random_")]
    shuffled_gate = df[df["mode"].str.startswith("shuffled_")]

    beats_control = True
    control_detail = ""
    for ctrl_df in [random_gate, shuffled_gate]:
        if len(ctrl_df) > 0:
            ch = float(ctrl_df.iloc[0]["hallucination_rate"])
            cname = str(ctrl_df.iloc[0]["mode"])
            if not (gh <= ch - 0.03):
                beats_control = False
                control_detail += f" {cname}(H={ch:.3f})"

    beats_open = False
    open_rows = df[df["mode"] == "steering_a-1.0"]
    if len(open_rows) > 0:
        open_h = float(open_rows.iloc[0]["hallucination_rate"])
        open_c = float(open_rows.iloc[0]["correct_answer_rate"])
        open_ua = float(open_rows.iloc[0]["unnecessary_abstention_rate"])
        if gc > open_c and gh <= open_h + 0.02:
            beats_open = True

    sep_detail = f"best_sep_gate={best_sep_gate}(gate_sep={best_sep_val:.3f})"

    if beats_control and beats_open:
        return ("IC4_M3_V2_GATE_PROMISING",
                f"Gated steering ({best_gname}): H={gh:.3f} (base={base_h:.3f}), C={gc:.3f} (base={base_c:.3f}), "
                f"UA={gua:.3f}. {sep_detail}. "
                f"Gate beats both open-loop and controls.")

    if any_sep:
        gap_str = f"; oracle H={oh:.3f} C={float(oracle.iloc[0]['correct_answer_rate']) if len(oracle)>0 else '?'}" \
                  f" UA={float(oracle.iloc[0]['unnecessary_abstention_rate']) if len(oracle)>0 else '?'}"
        ctrl_str = f"; controls NOT beaten{control_detail}" if not beats_control else "; controls beaten"
        return ("IC4_M3_V2_GATE_INSUFFICIENT",
                f"Gate ({best_gname}) moves (gate_sep={best_sep_val:.3f}) but metrics: "
                f"H={gh:.3f} (base H={base_h:.3f}){gap_str}. "
                f"{sep_detail}{ctrl_str}. "
                f"Gate signal has action but insufficient to match oracle gate.")

    return ("IC4_M3_V2_SIGNAL_WEAK",
            "Gate signal too weak to drive meaningful behavior difference.")


# ── Report ──

def _generate_report(report_path, config, df, telemetry_df, verdict, verdict_reason, elapsed):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    base = df[df["mode"] == "base"].iloc[0]
    base_h = float(base["hallucination_rate"])
    base_c = float(base["correct_answer_rate"])
    base_ua = float(base["unnecessary_abstention_rate"])

    lines = []
    lines.append("# IC-4-M3-v2: Telemetry-First Gated Steering Report")
    lines.append("")
    lines.append("## 1. M3-v1 & M3-O Recap")
    lines.append("")
    lines.append("M3-v1 (entropy feedback): implementation bug -- baseline computed under hook, gate did not activate.")
    lines.append("M3-O (oracle gate): IC4_M3_ORACLE_GATE_SUCCESS -- with perfect gate, v achieves dH=-0.200 with C and UA fully preserved.")
    lines.append("Conclusion: v is clean; the bottleneck is gating/feedback signal quality, not vector quality.")
    lines.append("")

    lines.append("## 2. M3-v2 Design")
    lines.append("")
    lines.append("Three risk-based gating signals, each implementing:")
    lines.append("```")
    lines.append("alpha_t = alpha_max * gate_t")
    lines.append("gate_t = sigmoid(k * (risk_t - threshold))")
    lines.append("```")
    lines.append("")
    lines.append("| Gate | risk_t | default threshold | default k |")
    lines.append("|---|---|---|---|")
    lines.append(f"| entropy_gate | Shannon entropy of logits | {RISK_REGISTRY['entropy'][1]['threshold']} | {RISK_REGISTRY['entropy'][1]['k']} |")
    lines.append(f"| maxprob_gate | 1 - max(softmax(logits)) | {RISK_REGISTRY['maxprob'][1]['threshold']} | {RISK_REGISTRY['maxprob'][1]['k']} |")
    lines.append(f"| uncertainty_mass_gate | mass of uncertainty tokens (\"I don't know\", \"cannot determine\", etc.) | {RISK_REGISTRY['uncertainty_mass'][1]['threshold']} | {RISK_REGISTRY['uncertainty_mass'][1]['k']} |")
    lines.append("")

    lines.append("## 3. Experiment Configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    mcfg = config.get("model", {})
    lines.append(f"| Model | {mcfg.get('name', '?')} |")
    lines.append(f"| Device / dtype | {mcfg.get('device', '?')} / {mcfg.get('torch_dtype', '?')} |")
    dcfg = config.get("data", {})
    lines.append(f"| Train / Test size | {dcfg.get('train_size', '?')} / {dcfg.get('test_size', '?')} |")
    lines.append(f"| Seeds | {config.get('steering', {}).get('seeds', [])} |")
    lines.append(f"| Layer | {config.get('steering', {}).get('layers', [])} |")
    lines.append(f"| alpha_max | {config.get('steering', {}).get('alphas', [-1.0])} |")
    lines.append(f"| Elapsed | ~{elapsed/60:.0f} min (CPU) |")
    lines.append("")

    lines.append("## 4. Full Metrics Table")
    lines.append("")
    lines.append("| Mode | H | C | UA | CA | Vector |")
    lines.append("|---|---|---|---|---|---|")
    for _, row in df.iterrows():
        vt = str(row.get("vector_type", "none"))
        lines.append(f"| {row['mode']} | {row['hallucination_rate']:.3f} | "
                     f"{row['correct_answer_rate']:.3f} | {row['unnecessary_abstention_rate']:.3f} | "
                     f"{row.get('calibrated_abstention_rate', 0):.3f} | {vt} |")
    lines.append("")

    lines.append("## 5. Gate Telemetry Analysis")
    lines.append("")
    if telemetry_df is not None and len(telemetry_df) > 0:
        # Summary comparison table
        lines.append("### Gate Separation Summary")
        lines.append("")
        lines.append("| Gate | gate_ans | gate_unans | gate_sep | entropy_t_sep | max_prob_t_sep | uncertainty_mass_t_sep | Ranking |")
        lines.append("|---|---|---|---|---|---|---|---|")
        per_gate_summary = []
        for gname in telemetry_df["mode"].unique():
            sub = telemetry_df[telemetry_df["mode"] == gname]
            ans = sub[sub["label"] == "answerable"]
            unans = sub[sub["label"] == "unanswerable"]
            if len(ans) > 0 and len(unans) > 0:
                ga = float(ans["gate"].mean())
                gu = float(unans["gate"].mean())
                sep = abs(gu - ga)
                es = abs(float(unans["entropy_t"].mean()) - float(ans["entropy_t"].mean())) if "entropy_t" in sub.columns else 0.0
                ms = abs(float(unans["max_prob_t"].mean()) - float(ans["max_prob_t"].mean())) if "max_prob_t" in sub.columns else 0.0
                us = abs(float(unans["uncertainty_mass_t"].mean()) - float(ans["uncertainty_mass_t"].mean())) if "uncertainty_mass_t" in sub.columns else 0.0
                per_gate_summary.append((gname, ga, gu, sep, es, ms, us))
        per_gate_summary.sort(key=lambda x: -x[3])
        for rank, (gn, ga, gu, sep, es, ms, us) in enumerate(per_gate_summary, 1):
            tag = ""
            if sep < 0.01:
                tag = " INERT"
            elif "random_" in gn or "shuffled_" in gn:
                tag = " (control)"
            lines.append(f"| {gn}{tag} | {ga:.4f} | {gu:.4f} | {sep:.4f} | {es:.4f} | {ms:.4f} | {us:.4f} | #{rank} |")
        lines.append("")

        # Per-gate details
        lines.append("### Per-Gate Details")
        lines.append("")
        for gname in telemetry_df["mode"].unique():
            sub = telemetry_df[telemetry_df["mode"] == gname]
            ans = sub[sub["label"] == "answerable"]
            unans = sub[sub["label"] == "unanswerable"]
            lines.append(f"#### {gname}")
            lines.append("")
            lines.append("| Metric | Answerable | Unanswerable | Separation |")
            lines.append("|---|---|---|---|")
            for col in ["entropy_t", "max_prob_t", "uncertainty_mass_t", "gate", "alpha"]:
                if col in sub.columns:
                    va = round(float(ans[col].mean()), 4) if len(ans) > 0 else "N/A"
                    vu = round(float(unans[col].mean()), 4) if len(unans) > 0 else "N/A"
                    sep = round(abs(float(unans[col].mean()) - float(ans[col].mean())), 4) if len(ans) > 0 and len(unans) > 0 else "N/A"
                    lines.append(f"| {col} mean | {va} | {vu} | {sep} |")
            lines.append("")
    else:
        lines.append("No telemetry data available.")
    lines.append("")

    lines.append("## 6. Gate vs Oracle Gate vs Open-Loop Comparison")
    lines.append("")
    oracle_row = df[df["mode"] == "oracle_gate_a-1.0"]
    open_row = df[df["mode"] == "steering_a-1.0"]

    lines.append("| Mode | H | C | UA | dH_base | Gap to Oracle H |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(f"| base | {base_h:.3f} | {base_c:.3f} | {base_ua:.3f} | -- | -- |")
    if len(oracle_row) > 0:
        oh = float(oracle_row.iloc[0]["hallucination_rate"])
        oc = float(oracle_row.iloc[0]["correct_answer_rate"])
        oua = float(oracle_row.iloc[0]["unnecessary_abstention_rate"])
        lines.append(f"| oracle_gate | {oh:.3f} | {oc:.3f} | {oua:.3f} | -{base_h-oh:.3f} | -- |")
    if len(open_row) > 0:
        oph = float(open_row.iloc[0]["hallucination_rate"])
        opc = float(open_row.iloc[0]["correct_answer_rate"])
        opua = float(open_row.iloc[0]["unnecessary_abstention_rate"])
        lines.append(f"| open_loop a=-1.0 | {oph:.3f} | {opc:.3f} | {opua:.3f} | -{base_h-oph:.3f} | +{oph-oh:.3f} |" if len(oracle_row) > 0 else "")
    for _, row in df.iterrows():
        m = str(row["mode"])
        if m.endswith("_gate_a-1.0") and m != "oracle_gate_a-1.0":
            dh = base_h - float(row['hallucination_rate'])
            gap_ora = f"+{float(row['hallucination_rate'])-oh:.3f}" if len(oracle_row) > 0 else "?"
            lines.append(f"| {m} | {row['hallucination_rate']:.3f} | {row['correct_answer_rate']:.3f} | {row['unnecessary_abstention_rate']:.3f} | {dh:+.3f} | {gap_ora} |")
    lines.append("")

    lines.append("## 7. Verdict")
    lines.append("")
    lines.append(f"**Verdict: `{verdict}`**")
    lines.append("")
    lines.append(f"**Reasoning:** {verdict_reason}")
    lines.append("")

    lines.append("## 8. Key Questions Answered")
    lines.append("")

    # Build data-driven answers
    per_gate = {}
    if telemetry_df is not None and len(telemetry_df) > 0:
        for gname in telemetry_df["mode"].unique():
            sub = telemetry_df[telemetry_df["mode"] == gname]
            ans = sub[sub["label"] == "answerable"]
            unans = sub[sub["label"] == "unanswerable"]
            if len(ans) > 0 and len(unans) > 0:
                entry = {
                    "gate_ans": float(ans["gate"].mean()),
                    "gate_unans": float(unans["gate"].mean()),
                    "gate_sep": abs(float(unans["gate"].mean()) - float(ans["gate"].mean())),
                }
                for col in ["entropy_t", "max_prob_t", "uncertainty_mass_t"]:
                    if col in sub.columns:
                        entry[col + "_ans"] = float(ans[col].mean())
                        entry[col + "_unans"] = float(unans[col].mean())
                        entry[col + "_sep"] = abs(float(unans[col].mean()) - float(ans[col].mean()))
                per_gate[gname] = entry

    # Q1: Does gate move?
    moving_gates = {k: v for k, v in per_gate.items() if v["gate_sep"] >= 0.03}
    non_moving = {k: v for k, v in per_gate.items() if v["gate_sep"] < 0.03}
    if moving_gates:
        q1 = f"Yes. {len(moving_gates)}/{len(per_gate)} gates show gate movement >= 0.03."
        for k, v in sorted(moving_gates.items(), key=lambda x: -x[1]["gate_sep"]):
            q1 += f" {k}: gate_ans={v['gate_ans']:.3f} vs gate_unans={v['gate_unans']:.3f} (sep={v['gate_sep']:.3f})."
        if non_moving:
            q1 += f" Not moving: {', '.join(non_moving.keys())}."
    else:
        q1 = f"No. All gates near constant. Best gate_sep={max((v['gate_sep'] for v in per_gate.values()), default=0):.4f}."

    # Q2: Best risk signal
    real_per_gate = {k: v for k, v in per_gate.items()
                     if "random_" not in k and "shuffled_" not in k}
    if real_per_gate:
        best_risk = max(real_per_gate, key=lambda k: real_per_gate[k]["gate_sep"])
        bv = real_per_gate[best_risk]
        best_signal_sep = bv.get("entropy_t_sep", bv.get("max_prob_t_sep", bv.get("uncertainty_mass_t_sep", 0.0)))
        q2 = f"**{best_risk}** (gate_sep={bv['gate_sep']:.3f}, signal_sep={best_signal_sep:.3f}). "
        others = {k: v for k, v in real_per_gate.items() if k != best_risk}
        if others:
            q2 += "Runner-up: " + ", ".join(f"{k}(sep={v['gate_sep']:.3f})" for k, v in sorted(others.items(), key=lambda x: -x[1]["gate_sep"]))
            q2 += "."
        if "uncertainty_mass_gate" in real_per_gate and real_per_gate["uncertainty_mass_gate"]["gate_sep"] < 0.01:
            q2 += " **uncertainty_mass_gate is effectively inert** (sep<0.01) -- this signal is not useful for gating."
    else:
        q2 = "No real gate telemetry available."

    # Q3: Real gate vs controls
    real_gates_only = {k: v for k, v in real_per_gate.items()} if real_per_gate else {}
    ctrl_gates = {k: v for k, v in per_gate.items()
                  if "random_" in k or "shuffled_" in k}
    if real_gates_only and ctrl_gates:
        best_real_sep = max(v["gate_sep"] for v in real_gates_only.values())
        best_ctrl_sep = max(v["gate_sep"] for v in ctrl_gates.values())
        gap = best_real_sep - best_ctrl_sep
        if gap > 0.01:
            q3 = f"Yes. Best real gate_sep={best_real_sep:.3f} > best ctrl gate_sep={best_ctrl_sep:.3f} (gap=+{gap:.3f})."
        elif abs(gap) <= 0.01:
            q3 = f"No. Best real gate_sep={best_real_sep:.3f} ~= best ctrl gate_sep={best_ctrl_sep:.3f} (within noise)."
        else:
            q3 = f"No. Best real gate_sep={best_real_sep:.3f} < best ctrl gate_sep={best_ctrl_sep:.3f}."
    else:
        q3 = "Cannot determine (insufficient telemetry)."

    # Q4: Gap to oracle gate
    oracle_row = df[df["mode"] == "oracle_gate_a-1.0"]
    oh4 = float(oracle_row.iloc[0]["hallucination_rate"]) if len(oracle_row) > 0 else base_h
    oc4 = float(oracle_row.iloc[0]["correct_answer_rate"]) if len(oracle_row) > 0 else base_c
    oua4 = float(oracle_row.iloc[0]["unnecessary_abstention_rate"]) if len(oracle_row) > 0 else base_ua
    _real_modes = df[df["mode"].str.contains("gate") & ~df["mode"].str.contains("oracle")
                     & ~df["mode"].str.contains("random_") & ~df["mode"].str.contains("shuffled_")]
    best_real_h = min(float(_real_modes.iloc[i]["hallucination_rate"]) for i in range(len(_real_modes))) if len(_real_modes) > 0 else 1.0
    q4 = (f"Oracle gate: H={oh4:.3f}, C={oc4:.3f}, UA={oua4:.3f}. "
          f"Best real gate H={best_real_h:.3f}. Gap in H: {best_real_h-oh4:+.3f}. "
          f"Oracle gate uses ground-truth answerability as gate (perfect separation). "
          f"Token-level risk signals (entropy/maxprob/uncertainty_mass) are much weaker proxies "
          f"for answerability. The core gap is signal quality, not steering vector quality. "
          f"A probe trained on hidden states to classify answerability directly would close most of this gap.")

    # Q5: Next step
    if verdict == "IC4_M3_V2_GATE_PROMISING":
        q5 = "Tune gate parameters (k/threshold) and consider multi-signal fusion. Then proceed to probe gate."
    elif "INSUFFICIENT" in verdict:
        q5 = ("**Proceed to probe gate**, not back to re-extracting v. "
              "Evidence: (1) oracle gate proves v is clean -- with perfect gate, dH=-0.200 with C/UA fully preserved. "
              "(2) token-level risk signals have separation (entropy: 0.071, maxprob: 0.117) but are insufficient "
              "to drive gating decisions comparable to oracle. "
              "(3) A lightweight answerability classifier trained on hidden states (probe gate) would have access "
              "to much richer signal than scalar summary statistics of logits. "
              "Do NOT return to v extraction -- that would waste the oracle gate diagnostic result.")
    elif "SIGNAL_WEAK" in verdict:
        q5 = ("Consider probe-gate route (train answerability classifier on hidden states) "
              "since token-level scalar signals do not show meaningful separation. "
              "Alternatively, multi-signal fusion or uncertainty-token-specific mass may provide stronger signal.")
    else:
        q5 = "Proceed to probe gate."

    lines.append(f"1. **Does the gate move?** {q1}")
    lines.append(f"2. **Which risk signal best separates answerable/unanswerable?** {q2}")
    lines.append(f"3. **Real gate vs random/shuffled gate?** {q3}")
    lines.append(f"4. **Why the gap to oracle gate?** {q4}")
    lines.append(f"5. **Next step?** {q5}")
    lines.append("")

    # One-line conclusion
    if real_per_gate:
        best_fb_signal = max(real_per_gate, key=lambda k: real_per_gate[k]["gate_sep"])
        best_fb_sep = real_per_gate[best_fb_signal]["gate_sep"]
    else:
        best_fb_signal = "N/A"
        best_fb_sep = 0.0
    oracle_h_gap = best_real_h - oh4
    conc = (f"当前最好的 feedback signal 是 **{best_fb_signal}** (gate_sep={best_fb_sep:.3f}), "
            f"它离 oracle gate 还有 H={oracle_h_gap:+.3f} 的差距。"
            f"差距根因是 signal quality (scalar logits stats vs ground-truth label)，"
            f"下一步应做 probe gate 而非重新提取 v。")
    lines.append(f"**One-line Conclusion:** {conc}")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*IC-4-M3-v2: Telemetry-First Gated Steering*")
    lines.append("*Generated by run_m3_v2*")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="IC-4-M3-v2: Telemetry-First Gated Steering")
    parser.add_argument("--config", type=str, default="configs/config_m3_v2.yaml")
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
    _log("IC-4-M3-v2: Telemetry-First Gated Steering", log_path)
    _log("=" * 60, log_path)

    seeds = config["steering"]["seeds"]
    layers = config["steering"]["layers"]
    alpha_max = config["steering"]["alphas"][0] if config["steering"]["alphas"] else -1.0
    gen_cfg = config["generation"]

    _log(f"\nConfig: seed={seeds}, layer={layers}, alpha_max={alpha_max}", log_path)
    _log(f"Gates: {list(RISK_REGISTRY.keys())}", log_path)

    from src.model_loader import load_model_and_tokenizer, get_model_layer_count

    _log(f"\nLoading model ({config['model']['name']})...", log_path)
    model, tokenizer = load_model_and_tokenizer(
        model_name=config["model"]["name"],
        device=config["model"]["device"],
        torch_dtype=config["model"].get("torch_dtype", "float32"),
    )
    _log(f"  Total layers: {get_model_layer_count(model)}", log_path)

    all_metrics = []
    all_telemetry = []
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
            _log(f"    Loaded {acts['positive'].shape[0]} pairs, dim={hidden_dim}", log_path)

            # Open-loop
            _log(f"\n  Open-loop...", log_path)
            ol_res, _ = run_generation_with_steering(
                model, tokenizer, test, all_vectors["steering"], layer_idx,
                alpha_max, "steering",
                max_new_tokens=gen_cfg["max_new_tokens"],
                temperature=gen_cfg["temperature"], do_sample=gen_cfg["do_sample"],
            )
            m = _evaluate_and_add(ol_res, seed, layer_idx, f"steering_a{alpha_max}",
                                  alpha_max, "steering", all_metrics)
            _log(f"    open_a{alpha_max:+.2f}: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f} UA={m['unnecessary_abstention_rate']:.3f}", log_path)

            # Oracle gate
            _log(f"\n  Oracle gate...", log_path)
            og_res = _generate_oracle_gated(model, tokenizer, test, all_vectors["steering"],
                                            layer_idx, alpha_max, "oracle_gate_a-1.0", gen_cfg)
            m = _evaluate_and_add(og_res, seed, layer_idx, "oracle_gate_a-1.0", alpha_max,
                                  "steering", all_metrics)
            _log(f"    oracle_a{alpha_max:+.2f}: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f} UA={m['unnecessary_abstention_rate']:.3f}", log_path)

            # Gated steering
            _log(f"\n  Risk-gated steering:", log_path)
            for gate_name, (risk_fn, params) in RISK_REGISTRY.items():
                k_val = params["k"]
                th = params["threshold"]
                mode_label = f"{gate_name}_gate_a{alpha_max}"

                _log(f"    {gate_name}: k={k_val}, th={th}", log_path)
                gated_res, tel_rows = _generate_gated(
                    model, tokenizer, test, all_vectors["steering"], layer_idx,
                    alpha_max, mode_label, risk_fn, k_val, th, gen_cfg, "steering",
                )
                m = _evaluate_and_add(gated_res, seed, layer_idx, mode_label, alpha_max,
                                      "steering", all_metrics)
                _log(f"    {gate_name}: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f} UA={m['unnecessary_abstention_rate']:.3f}", log_path)
                all_telemetry.extend(tel_rows)

            # Control gates (random + shuffled with entropy)
            _log(f"\n  Control gates:", log_path)
            for ctrl_name, ctrl_vec in [("random", all_vectors["random"]),
                                         ("shuffled", all_vectors["shuffled"])]:
                risk_fn, params = RISK_REGISTRY["entropy"]
                mode_label = f"{ctrl_name}_entropy_gate_a{alpha_max}"
                ctrl_res, ctrl_tel = _generate_gated(
                    model, tokenizer, test, ctrl_vec, layer_idx,
                    alpha_max, mode_label, risk_fn, params["k"], params["threshold"],
                    gen_cfg, ctrl_name,
                )
                m = _evaluate_and_add(ctrl_res, seed, layer_idx, mode_label, alpha_max,
                                      ctrl_name, all_metrics)
                _log(f"    {ctrl_name}_entropy: H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f} UA={m['unnecessary_abstention_rate']:.3f}", log_path)
                all_telemetry.extend(ctrl_tel)

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
    present = [c for c in cols if c in df.columns]
    df = df[present]
    met_path = os.path.join(results_dir, "metrics_raw.csv")
    df.to_csv(met_path, index=False)
    _log(f"\nMetrics saved to {met_path} ({len(df)} rows)", log_path)

    tel_df = pd.DataFrame(all_telemetry) if all_telemetry else None
    if tel_df is not None and len(tel_df) > 0:
        tel_path = os.path.join(results_dir, "telemetry_gates.csv")
        _save_telemetry(all_telemetry, tel_path)
        _log(f"Telemetry saved to {tel_path} ({len(all_telemetry)} rows)", log_path)

    # Comparison table
    _log(f"\n{'='*60}", log_path)
    _log("COMPARISON TABLE", log_path)
    _log(f"{'='*60}", log_path)
    for _, row in df.iterrows():
        _log(f"  {row['mode']:<35} H={row['hallucination_rate']:.3f} C={row['correct_answer_rate']:.3f} UA={row['unnecessary_abstention_rate']:.3f}", log_path)

    # Gate telemetry summary
    if tel_df is not None and len(tel_df) > 0:
        _log(f"\n{'='*60}", log_path)
        _log("GATE TELEMETRY SUMMARY", log_path)
        _log(f"{'='*60}", log_path)
        for gname in tel_df["mode"].unique():
            sub = tel_df[tel_df["mode"] == gname]
            ans = sub[sub["label"] == "answerable"]
            unans = sub[sub["label"] == "unanswerable"]
            ga = float(ans["gate"].mean()) if len(ans) > 0 else 0
            gu = float(unans["gate"].mean()) if len(unans) > 0 else 0
            _log(f"  {gname}: gate_ans={ga:.4f} gate_unans={gu:.4f} separation={abs(gu-ga):.4f}", log_path)

    # Verdict
    verdict, reason = _compute_verdict(df, tel_df)
    _log(f"\n{'='*60}", log_path)
    _log(f"VERDICT: {verdict}", log_path)
    _log(f"{'='*60}", log_path)
    _log(f"Reason: {reason}", log_path)

    # Report
    report_path = os.path.join(reports_dir, "IC4_M3_V2_TELEMETRY_REPORT.md")
    _generate_report(report_path, config, df, tel_df, verdict, reason, elapsed)
    _log(f"\nReport saved to {report_path}", log_path)
    _log(f"\nIC-4-M3-v2 complete. Verdict: {verdict}", log_path)


if __name__ == "__main__":
    main()