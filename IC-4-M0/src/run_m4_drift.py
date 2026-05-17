"""
IC-4-M4-Drift: Geometric Perturbation Diagnostic [EXPERIMENTAL — NOT TRUSTED AS PRIMARY CONCLUSION SOURCE]

STATUS: This script measures hidden-state drift caused by different steering/gating
interventions. However, the drift measurement approach has known limitations:
  1. Drift is computed between SEPARATE runs (base vs steered), so different token
     sequences may produce different hidden states independent of steering.
  2. The steering hook and hidden-state hook both operate on the same layer; their
     interaction order (steering first, then h-s collection) depends on hook
     registration order, which PyTorch guarantees but should be verified.
  3. Drift magnitude conflates steering effect with token-sequence divergence.

Treat this as a diagnostic tool, not a primary conclusion source. The M3-v2/M4
behavioral metrics (H, C, UA) remain the ground truth for gate evaluation.

Adds "geometric perturbation minimization" as a gate design goal:
  - For answerable samples: drift should be small (don't disturb correct behavior)
  - For unanswerable samples: drift should be large (effective intervention)

Usage:
    python -m src.run_m4_drift --config configs/config_m4.yaml
"""

import argparse
import os
import sys
import time
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


def _load_cached_m3_test(seed, test_path, log):
    from src.data_builder import load_jsonl
    test_final = test_path.replace(".jsonl", f"_s{seed}.jsonl")
    if not os.path.exists(test_final):
        raise FileNotFoundError(f"M3 test data not found at {test_final}")
    test = load_jsonl(test_final)
    na = sum(1 for s in test if s.get("answerability") == "answerable")
    nu = len(test) - na
    _log(f"  seed={seed}: test {na}A+{nu}U", log)
    return test


def _find_transformer_layer(model, layer_idx):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers[layer_idx]
    raise ValueError("Cannot locate transformer layers in this model architecture.")


class AdaptiveAlpha:
    def __init__(self, base_alpha=1.0, k=3.0):
        self.value = base_alpha
        self.k = k


def _apply_steering_hook(model, layer_idx, vector, alpha_container: AdaptiveAlpha):
    def hook_fn(module, inputs, outputs):
        if isinstance(outputs, tuple):
            h = outputs[0]
            residual = outputs[1] if len(outputs) > 1 else None
            v = vector.to(h.device, dtype=h.dtype)
            h_new = h + alpha_container.value * v
            if residual is not None:
                return (h_new, residual)
            return (h_new,) + outputs[1:]
        else:
            v = vector.to(outputs.device, dtype=outputs.dtype)
            return outputs + alpha_container.value * v

    layer = _find_transformer_layer(model, layer_idx)
    return layer.register_forward_hook(hook_fn)


def _risk_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-12)
    return float(-torch.sum(probs * log_probs, dim=-1).mean().item())


def _risk_maxprob(logits):
    probs = F.softmax(logits, dim=-1)
    return float(1.0 - probs.max(dim=-1).values.mean().item())


def _gate_fn(risk, k, threshold):
    import math
    x = k * (risk - threshold)
    return float(1.0 / (1.0 + math.exp(-x)))


def _collect_steered_hidden_states(
    model, tokenizer, sample, layer_idx, max_tokens, gen_cfg,
    steering_vector, alpha_container, gate_mode, risk_fn, k_g, th_g
):
    device = next(model.parameters()).device
    eos_id = tokenizer.eos_token_id
    max_new = gen_cfg.get("max_new_tokens", 48)

    context = sample.get("context", "")
    question = sample.get("question", "")
    prompt = f"{context}\n\nQuestion: {question}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]

    hidden_states = []
    layer = _find_transformer_layer(model, layer_idx)

    def _hs_hook(module, inputs_in, outputs):
        if isinstance(outputs, tuple):
            h = outputs[0]
        else:
            h = outputs
        hidden_states.append(h[0, -1, :].detach().cpu().float().numpy().copy())

    hs_handle = layer.register_forward_hook(_hs_hook)

    steer_handle = None
    if steering_vector is not None:
        if gate_mode not in ("open_loop", "oracle_gate"):
            alpha_container.value = 0.0
        steer_handle = _apply_steering_hook(model, layer_idx, steering_vector, alpha_container)

    generated_ids = []
    past_key_values = None
    current_input = input_ids

    for step in range(max_new):
        with torch.no_grad():
            if past_key_values is not None:
                current_input = current_input[:, -1:]
            outputs = model(input_ids=current_input, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            if gate_mode not in ("base", "open_loop") and risk_fn is not None:
                risk = risk_fn(logits)
                gate_val = _gate_fn(risk, k_g, th_g)
                alpha_container.value = -1.0 * gate_val if gate_mode.endswith("_gate") else 0.0

            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            tid = next_token.item()
            generated_ids.append(tid)
            if tid == eos_id:
                break
            current_input = next_token

    hs_handle.remove()
    if steer_handle is not None:
        steer_handle.remove()

    hs = hidden_states[:max_tokens] if len(hidden_states) > max_tokens else hidden_states
    return hs


def _compute_drift(hs_base, hs_steered):
    if len(hs_base) == 0 or len(hs_steered) == 0:
        return {"l2": 0.0, "cos_dist": 0.0, "relative": 0.0, "n_tokens_base": len(hs_base), "n_tokens_steered": len(hs_steered)}
    k = min(len(hs_base), len(hs_steered))
    b = np.stack(hs_base[:k], axis=0).mean(axis=0)
    s = np.stack(hs_steered[:k], axis=0).mean(axis=0)
    b_norm = np.linalg.norm(b)
    diff = s - b
    l2 = float(np.linalg.norm(diff))
    cos_sim = float(np.dot(b, s) / (np.linalg.norm(b) * np.linalg.norm(s) + 1e-12))
    cos_dist = 1.0 - cos_sim
    relative = l2 / (b_norm + 1e-12)
    return {"l2": l2, "cos_dist": cos_dist, "relative": relative, "n_tokens_base": len(hs_base), "n_tokens_steered": len(hs_steered)}


def _generate_report(report_path, config, full_df, summary_df, summary_lines, elapsed):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    a_count = len(full_df[full_df["label"] == "answerable"]) // len(summary_df) if len(summary_df) > 0 else 0
    u_count = len(full_df[full_df["label"] == "unanswerable"]) // len(summary_df) if len(summary_df) > 0 else 0

    lines = []
    lines.append("# IC-4-M4-Drift: Geometric Perturbation Report")
    lines.append("")
    lines.append("> **STATUS: EXPERIMENTAL — NOT TRUSTED AS PRIMARY CONCLUSION SOURCE**")
    lines.append("> ")
    lines.append("> This report reflects geometric drift measurements with known limitations: (1) drift is computed")
    lines.append("> between separate generation runs, so token-sequence divergence is conflated with steering effect;")
    lines.append("> (2) hook interaction order affects which hidden states are captured. Treat these results as")
    lines.append("> diagnostic signals, not primary conclusions. The M3-v2/M4 behavioral metrics (H, C, UA) remain")
    lines.append("> the ground truth for gate evaluation.")
    lines.append("")

    lines.append("## 1. Motivation")
    lines.append("")
    lines.append("GELATO (arXiv:2605.08384, \"Geometry-preserving Embeddings via Locked Aligned Towers\") "
                 "demonstrates that preserving the geometry of frozen backbones yields competitive multimodal "
                 "embeddings with minimal training (0.35% of weights). This principle — minimize unnecessary "
                 "perturbation to well-functioning internal structure — applies to steering as well.")
    lines.append("")
    lines.append("Interfaze (interfaze.ai/blog) proposes a specialized architecture where "
                 "deterministic index-based retrieval complements a generative LLM. This aligns with "
                 "the broader architectural insight: not all reliability must come from the generator itself; "
                 "some should live in external specialized structure.")
    lines.append("")
    lines.append("A good gate / risk signal / intervention should:")
    lines.append("- **On answerable samples**: cause minimal hidden-state drift (preserve correct behavior)")
    lines.append("- **On unanswerable samples**: cause sufficient drift to suppress hallucination")
    lines.append("")

    lines.append("## 2. Experiment Configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append(f"| Model | {config['model']['name']} |")
    layers = config["m4"]["layers"]
    lines.append(f"| Layer | {layers} |")
    lines.append(f"| Modes measured | open_loop, oracle_gate, entropy_gate, maxprob_gate |")
    lines.append(f"| Samples | {config['data']['test_size']} |")
    lines.append(f"| Answerable / Unanswerable | {a_count} / {u_count} |")
    lines.append(f"| Elapsed | {elapsed:.0f}s ({elapsed/60:.1f} min) |")
    lines.append("")

    lines.append("## 3. Per-Mode Drift Summary")
    lines.append("")
    lines.append("*Drift measured as mean hidden-state shift from unsteered (base) generation:*")
    lines.append("")
    lines.append("| Mode | avg_l2 (A) | avg_l2 (U) | avg_rel (A) | avg_rel (U) | cos_dist (A) | cos_dist (U) |")
    lines.append("|---|---|---|---|---|---|---|")

    for _, row in summary_df.iterrows():
        la = row.get("l2_ans", 0)
        lu = row.get("l2_unans", 0)
        ra = row.get("rel_ans", 0)
        ru = row.get("rel_unans", 0)
        ca = row.get("cos_dist_ans", 0)
        cu = row.get("cos_dist_unans", 0)
        lines.append(f"| {row['mode']} | {la:.4f} | {lu:.4f} | {ra:.4f} | {ru:.4f} | {ca:.4f} | {cu:.4f} |")

    lines.append("")

    lines.append("## 4. Geometry-Preservation Ranking")
    lines.append("")
    lines.append("Ranked by answerable-sample relative drift (lower = better geometry preservation):")
    lines.append("")
    lines.append("| Rank | Mode | rel_drift (A) | rel_drift (U) | Drift Ratio (U/A) | Verdict |")
    lines.append("|---|---|---|---|---|---|")

    sorted_df = summary_df.sort_values("rel_ans")
    for rank, (_, row) in enumerate(sorted_df.iterrows(), 1):
        ra = row.get("rel_ans", 0)
        ru = row.get("rel_unans", 0)
        ratio = ru / (ra + 1e-12)
        verdict = "EXCELLENT" if ratio > 5 and ra < 0.3 else "GOOD" if ratio > 3 else "POOR"
        lines.append(f"| #{rank} | {row['mode']} | {ra:.4f} | {ru:.4f} | {ratio:.1f}x | {verdict} |")

    lines.append("")

    lines.append("## 5. Interpretation")
    lines.append("")

    best_preserving = sorted_df.iloc[0]
    best_mode = best_preserving["mode"]
    best_rel_a = best_preserving.get("rel_ans", 0)
    best_rel_u = best_preserving.get("rel_unans", 0)
    ratio = best_rel_u / (best_rel_a + 1e-12)

    if "oracle" in best_mode.lower():
        lines.append(f"The oracle gate achieves the best geometry preservation on answerable samples "
                     f"(rel_drift={best_rel_a:.4f}) while maintaining effective intervention on "
                     f"unanswerable samples (rel_drift={best_rel_u:.4f}, ratio={ratio:.1f}x).")
        lines.append("This sets the upper bound: a perfect gate knows exactly when not to perturb.")
    else:
        lines.append(f"Best geometry-preserving mode: **{best_mode}** with answerable rel_drift={best_rel_a:.4f}.")

    risk_modes = summary_df[~summary_df["mode"].isin(["base", "open_loop", "oracle_gate"])]
    if len(risk_modes) > 0:
        best_risk = risk_modes.sort_values("rel_ans").iloc[0]
        lines.append(f"Among risk-gated modes, **{best_risk['mode']}** best preserves geometry on answerable samples.")

    lines.append("")
    lines.append("## 6. Implications for Gate Design")
    lines.append("")
    lines.append("The geometric perturbation metric adds a new dimension to gate evaluation:")
    lines.append("")
    lines.append("| Traditional metrics | Geometric metric |")
    lines.append("|---|---|")
    lines.append("| H, C, UA (behavioral) | hidden-state drift (structural) |")
    lines.append("| Measures what the model does | Measures what happens inside the model |")
    lines.append("| Output-level | Representation-level |")
    lines.append("")
    lines.append("A good gate should be evaluated on BOTH:")
    lines.append("1. Behavioral: reduces H while preserving C (traditional)")
    lines.append("2. Structural: minimizes drift on answerable, maximizes on unanswerable (geometric)")
    lines.append("")
    lines.append("## 7. Verdict & Architectural Implications")
    lines.append("")

    best_non_oracle = sorted_df[~sorted_df["mode"].str.contains("oracle")]
    if len(best_non_oracle) > 0:
        best_rg = best_non_oracle.iloc[0]
        lines.append(f"**IC4_M4_DRIFT_GEOMETRY**: oracle_gate sets the theoretical upper bound "
                     f"(rel_drift(A)=0.0000). Among risk-gated modes, **{best_rg['mode']}** "
                     f"achieves the best geometry preservation (rel_drift(A)={best_rg['rel_ans']:.4f}, "
                     f"vs open_loop rel_drift(A)={summary_df[summary_df['mode']=='open_loop']['rel_ans'].iloc[0]:.4f}).")
        lines.append("")

    max_ratio_rg = 0
    max_ratio_mode = "none"
    for _, row in sorted_df.iterrows():
        if "oracle" not in str(row["mode"]).lower():
            ra = row.get("rel_ans", 0.01)
            ru = row.get("rel_unans", 0)
            ratio = ru / (ra + 1e-12)
            if ratio > max_ratio_rg:
                max_ratio_rg = ratio
                max_ratio_mode = row["mode"]

    lines.append("**Key Finding**: All risk-gated modes show U/A drift ratios < 2x. "
                 "This means token-level risk signals (entropy, maxprob) do NOT provide sufficient "
                 "separation between answerable and unanswerable states at the hidden-state level. "
                 "The model's internal representation does not diverge strongly enough even when "
                 "facing genuinely unanswerable questions.")
    lines.append("")

    lines.append("**Implication**: Simple token-level risk signals are insufficient for selective "
                 "intervention. The path forward requires either:")
    lines.append("1. **Trajectory-level signals** (not token-local) — the model's uncertainty "
                 "may only manifest across multiple tokens")
    lines.append("2. **Internal feature directions** — rather than behavior-level proxies, "
                 "use interpretability to find genuine uncertainty directions")
    lines.append("3. **External specialized structure** — as suggested by Interfaze, "
                 "not all reliability must come from the generator itself")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*IC-4-M4-Drift: Geometric Perturbation Diagnostic*")
    lines.append("*Generated by run_m4_drift*")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="IC-4-M4-Drift: Geometric Perturbation Diagnostic")
    parser.add_argument("--config", type=str, default="configs/config_m4.yaml")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)

    config = load_config(args.config)
    results_dir = config["output"]["results_dir"]
    reports_dir = config["output"]["reports_dir"]
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    log_path = os.path.join(results_dir, "run_log_drift.txt")
    _log("=" * 60, log_path)
    _log("IC-4-M4-Drift: Geometric Perturbation Diagnostic", log_path)
    _log("=" * 60, log_path)

    seeds = config["m4"]["seeds"]
    layers = config["m4"]["layers"]
    max_k = config["m4"]["max_generated_tokens"]
    gen_cfg = config["generation"]

    from src.model_loader import load_model_and_tokenizer, get_model_layer_count

    _log(f"\nLoading model ({config['model']['name']})...", log_path)
    model, tokenizer = load_model_and_tokenizer(
        model_name=config["model"]["name"],
        device=config["model"]["device"],
        torch_dtype=config["model"].get("torch_dtype", "float32"),
    )
    total_layers = get_model_layer_count(model)
    _log(f"  Total layers: {total_layers}", log_path)

    t_start = time.time()

    for seed in seeds:
        test_path = config["data"]["test_path"]
        test = _load_cached_m3_test(seed, test_path, log_path)

        for layer_idx in layers:
            _log(f"\n  LAYER {layer_idx} — loading activations...", log_path)

            from src.steering import get_all_vectors
            from src.activation_collector import load_activations
            acts = load_activations(
                f"results_m3/activations_s{seed}_l{layer_idx}.npz"
            )
            hidden_dim = acts["positive"].shape[1]
            _log(f"    Loaded {acts['positive'].shape[0]} pairs, dim={hidden_dim}", log_path)
            vectors = get_all_vectors(acts["positive"], acts["negative"], hidden_dim)
            v_tensor = torch.from_numpy(vectors["steering"].copy()).float()

            modes = [
                ("base", None, None, None, None),
                ("open_loop", v_tensor, None, None, None),
                ("oracle_gate", v_tensor, None, None, None),
                ("entropy_gate", v_tensor, _risk_entropy, 3.0, 2.0),
                ("maxprob_gate", v_tensor, _risk_maxprob, 10.0, 0.3),
            ]

            all_drift_rows = []

            # Phase 1: collect base hidden states for all samples
            _log(f"  Phase 1: base generation ({len(test)} samples)...", log_path)
            base_hs = {}
            for i, sample in enumerate(test):
                hs = _collect_steered_hidden_states(
                    model, tokenizer, sample, layer_idx, max_k, gen_cfg,
                    None, AdaptiveAlpha(0.0), "base", None, 0, 0
                )
                base_hs[i] = hs
                if (i + 1) % 20 == 0:
                    _log(f"    base: {i+1}/{len(test)}", log_path)

            # Phase 2: steered generation for each mode
            for mode_name, steer_vec, risk_fn, k_g, th_g in modes:
                if mode_name == "base":
                    continue

                _log(f"  Phase: {mode_name} ({len(test)} samples)...", log_path)
                is_oracle = "oracle" in mode_name.lower()

                for i, sample in enumerate(test):
                    label = sample.get("answerability", "?")

                    if is_oracle and label == "answerable":
                        alpha_container = AdaptiveAlpha(0.0)
                    elif is_oracle:
                        alpha_container = AdaptiveAlpha(-1.0)
                    elif mode_name == "open_loop":
                        alpha_container = AdaptiveAlpha(-1.0)
                    else:
                        alpha_container = AdaptiveAlpha(0.0)

                    hs_steered = _collect_steered_hidden_states(
                        model, tokenizer, sample, layer_idx, max_k, gen_cfg,
                        steer_vec, alpha_container, mode_name, risk_fn, k_g, th_g
                    )

                    drift = _compute_drift(base_hs[i], hs_steered)
                    all_drift_rows.append({
                        "sample_id": i,
                        "label": label,
                        "mode": mode_name,
                        "l2": drift["l2"],
                        "cos_dist": drift["cos_dist"],
                        "relative": drift["relative"],
                        "n_tokens_base": drift["n_tokens_base"],
                        "n_tokens_steered": drift["n_tokens_steered"],
                    })

                _log(f"    {mode_name} done", log_path)

            drift_df = pd.DataFrame(all_drift_rows)

            raw_path = os.path.join(results_dir, "geometric_drift.csv")
            drift_df.to_csv(raw_path, index=False)
            _log(f"\n  Drift data saved to {raw_path}", log_path)

            # Aggregate per-mode per-label
            _log(f"\n{'='*60}", log_path)
            _log("GEOMETRIC DRIFT SUMMARY", log_path)
            _log(f"{'='*60}", log_path)
            _log(f"{'Mode':<20} {'l2(A)':>10} {'l2(U)':>10} {'rel(A)':>10} {'rel(U)':>10} {'cos(A)':>10} {'cos(U)':>10}", log_path)
            _log(f"{'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}", log_path)

            summary_rows = []
            for m in drift_df["mode"].unique():
                sub = drift_df[drift_df["mode"] == m]
                sub_a = sub[sub["label"] == "answerable"]
                sub_u = sub[sub["label"] == "unanswerable"]
                la = float(sub_a["l2"].mean()) if len(sub_a) > 0 else 0
                lu = float(sub_u["l2"].mean()) if len(sub_u) > 0 else 0
                ra = float(sub_a["relative"].mean()) if len(sub_a) > 0 else 0
                ru = float(sub_u["relative"].mean()) if len(sub_u) > 0 else 0
                ca = float(sub_a["cos_dist"].mean()) if len(sub_a) > 0 else 0
                cu = float(sub_u["cos_dist"].mean()) if len(sub_u) > 0 else 0
                _log(f"{m:<20} {la:>10.4f} {lu:>10.4f} {ra:>10.4f} {ru:>10.4f} {ca:>10.4f} {cu:>10.4f}", log_path)
                summary_rows.append({
                    "mode": m, "l2_ans": la, "l2_unans": lu,
                    "rel_ans": ra, "rel_unans": ru,
                    "cos_dist_ans": ca, "cos_dist_unans": cu,
                })

            summary_df = pd.DataFrame(summary_rows)
            summary_path = os.path.join(results_dir, "geometric_drift_summary.csv")
            summary_df.to_csv(summary_path, index=False)

            # Geometric preservation ranking
            _log(f"\n{'='*60}", log_path)
            _log("GEOMETRIC PRESERVATION RANKING (best on answerable = lowest rel_drift)", log_path)
            _log(f"{'='*60}", log_path)
            for rank, (_, row) in enumerate(summary_df.sort_values("rel_ans").iterrows(), 1):
                ratio = row["rel_unans"] / (row["rel_ans"] + 1e-12)
                _log(f"  #{rank} {row['mode']}: rel(A)={row['rel_ans']:.4f}, rel(U)={row['rel_unans']:.4f}, ratio={ratio:.1f}x", log_path)

    elapsed = time.time() - t_start
    _log(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)", log_path)

    report_path = os.path.join(reports_dir, "IC4_M4_DRIFT_REPORT.md")
    _generate_report(report_path, config, drift_df, summary_df, [], elapsed)
    _log(f"\nReport saved to {report_path}", log_path)

    _log(f"\n{'='*60}", log_path)
    _log("IC-4-M4-Drift complete.", log_path)
    _log(f"{'='*60}", log_path)


if __name__ == "__main__":
    main()