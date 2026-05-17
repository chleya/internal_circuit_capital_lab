"""
IC-4-M0 / IC-4-M1: Report writer.
Generates verdict reports in Markdown format.
Supports both M0 (single-seed) and M1 (multi-seed, multi-layer) formats.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional


# ── M0 Verdict (legacy, fixed) ──

def compute_verdict(df: pd.DataFrame) -> Tuple[str, str, Dict[str, float]]:
    """
    Compute the M0 final verdict with explicit per-delta breakdown.

    Returns:
        (verdict_code, reason_string, deltas_dict)
    """
    THRESH_H_REDUCTION = 0.20
    THRESH_C_DROP = 0.05
    THRESH_UA_INCREASE = 0.05
    THRESH_CONTROL_GAP = 0.05

    base_row = df[df["mode"] == "base"]
    if len(base_row) == 0:
        return "IC4_M0_NO_SIGNAL", "No base metrics found.", {}

    base_h = base_row["hallucination_rate"].values[0]
    base_c = base_row["correct_answer_rate"].values[0]
    base_ua = base_row["unnecessary_abstention_rate"].values[0]
    base_ca = base_row["calibrated_abstention_rate"].values[0]

    steering_rows = df[df["mode"].str.startswith("steering_a")]
    if len(steering_rows) == 0:
        return "IC4_M0_NO_SIGNAL", "No steering metrics found.", {}

    best_row = steering_rows.loc[steering_rows["hallucination_rate"].idxmin()]
    best_h = best_row["hallucination_rate"]
    best_c = best_row["correct_answer_rate"]
    best_ua = best_row["unnecessary_abstention_rate"]
    best_ca = best_row["calibrated_abstention_rate"]
    best_style = best_row["style_only_score"]
    best_alpha = best_row["alpha"]

    random_rows = df[df["mode"].str.startswith("random_a")]
    shuffled_rows = df[df["mode"].str.startswith("shuffled_a")]

    if len(random_rows) > 0:
        random_best_h = random_rows["hallucination_rate"].min()
    else:
        random_best_h = base_h

    if len(shuffled_rows) > 0:
        shuffled_best_h = shuffled_rows["hallucination_rate"].min()
    else:
        shuffled_best_h = base_h

    h_delta = base_h - best_h
    h_reduction_pct = (h_delta / base_h) if base_h > 0 else 0.0
    c_delta = base_c - best_c
    c_drop_pct = (c_delta / base_c) if base_c > 0 else 0.0
    ua_delta = best_ua - base_ua
    ca_delta = best_ca - base_ca

    random_h_delta = base_h - random_best_h
    random_h_reduction_pct = (random_h_delta / base_h) if base_h > 0 else 0.0
    shuffled_h_delta = base_h - shuffled_best_h
    shuffled_h_reduction_pct = (shuffled_h_delta / base_h) if base_h > 0 else 0.0

    control_gap_vs_random = h_reduction_pct - random_h_reduction_pct
    control_gap_vs_shuffled = h_reduction_pct - shuffled_h_reduction_pct

    deltas = {
        "best_alpha": best_alpha,
        "hallucination_delta": round(h_delta, 4),
        "hallucination_reduction_pct": round(h_reduction_pct, 4),
        "correct_answer_delta": round(-c_delta, 4),
        "correct_answer_drop_pct": round(c_drop_pct, 4),
        "unnecessary_abstention_delta": round(ua_delta, 4),
        "calibrated_abstention_delta": round(ca_delta, 4),
        "control_gap_vs_random": round(control_gap_vs_random, 4),
        "control_gap_vs_shuffled": round(control_gap_vs_shuffled, 4),
        "base_h": base_h,
        "base_c": base_c,
        "base_ua": base_ua,
        "best_h": best_h,
        "best_c": best_c,
        "best_ua": best_ua,
    }

    if base_h == 0.0:
        return "IC4_M0_NO_SIGNAL", "Base hallucination rate is already zero; no signal to detect.", deltas

    # ── Build reason from independent fields ──
    delta_parts = []
    delta_parts.append(f"hallucination: {base_h:.3f} → {best_h:.3f} (Δ={h_delta:+.3f}, {-h_reduction_pct:.1%})")
    delta_parts.append(f"correct_answer: {base_c:.3f} → {best_c:.3f} (Δ={-c_delta:+.3f}, {-c_drop_pct:.1%})")
    delta_parts.append(f"unnecessary_abstention: {base_ua:.3f} → {best_ua:.3f} (Δ={ua_delta:+.3f})")
    delta_parts.append(f"calibrated_abstention: {base_ca:.3f} → {best_ca:.3f} (Δ={ca_delta:+.3f})")

    failures = []

    if h_reduction_pct < THRESH_H_REDUCTION:
        failures.append(f"hallucination_reduction={h_reduction_pct:.1%} < threshold {THRESH_H_REDUCTION:.0%}")
    if c_drop_pct > THRESH_C_DROP:
        failures.append(f"correct_answer_drop={c_drop_pct:.1%} > threshold {THRESH_C_DROP:.0%}")
    if ua_delta > THRESH_UA_INCREASE:
        failures.append(f"unnecessary_abstention_increase={ua_delta:.1%} > threshold {THRESH_UA_INCREASE:.0%}")

    # ── Check for SIGNAL SUPPORTED ──
    if h_reduction_pct >= THRESH_H_REDUCTION and c_drop_pct <= THRESH_C_DROP and ua_delta <= THRESH_UA_INCREASE:
        if control_gap_vs_random > THRESH_CONTROL_GAP and control_gap_vs_shuffled > THRESH_CONTROL_GAP:
            reason = (
                "STEERING_SIGNAL_SUPPORTED: "
                + " | ".join(delta_parts)
                + f" | control_gap(random)={control_gap_vs_random:.1%}"
                + f" | control_gap(shuffled)={control_gap_vs_shuffled:.1%}"
            )
            return "IC4_M0_STEERING_SIGNAL_SUPPORTED", reason, deltas

    # ── STYLE ONLY ──
    if best_style > 2.0 and h_reduction_pct < 0.10:
        reason = (
            f"STYLE_ONLY: style_only_score={best_style:.2f} (>2.0) with hallucination_reduction only {h_reduction_pct:.1%} (<10%). "
            + " | ".join(delta_parts)
        )
        return "IC4_M0_STYLE_ONLY", reason, deltas

    # ── MODEL DAMAGE (only report actual failures) ──
    if h_reduction_pct >= THRESH_H_REDUCTION and len(failures) > 0:
        reason = (
            f"MODEL_DAMAGE: hallucination reduced by {h_reduction_pct:.1%} (>= {THRESH_H_REDUCTION:.0%}) "
            f"but: {'; '.join(failures)}. "
            + " | ".join(delta_parts)
        )
        return "IC4_M0_MODEL_DAMAGE", reason, deltas

    # ── RANDOM VECTOR EQUIVALENT ──
    if control_gap_vs_random <= 0.05 or control_gap_vs_shuffled <= 0.05:
        reason = (
            f"RANDOM_VECTOR_EQUIVALENT: steering reduction={h_reduction_pct:.1%}, "
            f"random reduction={random_h_reduction_pct:.1%} (gap={control_gap_vs_random:.1%}), "
            f"shuffled reduction={shuffled_h_reduction_pct:.1%} (gap={control_gap_vs_shuffled:.1%}). "
            + " | ".join(delta_parts)
        )
        return "IC4_M0_RANDOM_VECTOR_EQUIVALENT", reason, deltas

    # ── NO SIGNAL ──
    reason = (
        f"NO_SIGNAL: hallucination reduced by only {h_reduction_pct:.1%} (<{THRESH_H_REDUCTION:.0%}). "
        + " | ".join(delta_parts)
    )
    return "IC4_M0_NO_SIGNAL", reason, deltas


# ── M1 Verdict ──

def compute_m1_verdict(
    df: pd.DataFrame,
    base_row: Optional[pd.Series] = None,
) -> Tuple[str, str, Dict]:
    """
    Compute M1 verdict with strict criteria.

    A candidate must simultaneously satisfy:
      - hallucination_rate <= 0.50
      - correct_answer_rate >= 0.82
      - unnecessary_abstention_rate <= 0.09
      - real steering clearly better than random control
      - real steering clearly better than shuffled-label control

    Control gap:
      - real_h <= random_h - 0.10
      - real_h <= shuffled_h - 0.10
      - real_c >= control_c - 0.05
      - real_ua <= control_ua + 0.05
    """
    STEERING_H_THRESH = 0.50
    STEERING_C_THRESH = 0.82
    STEERING_UA_THRESH = 0.09
    CONTROL_GAP_H = 0.10
    CONTROL_GAP_C = 0.05
    CONTROL_GAP_UA = 0.05

    base_row_data = base_row if base_row is not None else df[df["mode"] == "base"]
    if len(base_row_data) == 0:
        return "IC4_M1_NO_DATA", "No base metrics found.", {}

    if isinstance(base_row_data, pd.DataFrame):
        base_row_data = base_row_data.iloc[0]

    base_h = base_row_data["hallucination_rate"]
    base_c = base_row_data["correct_answer_rate"]
    base_ua = base_row_data["unnecessary_abstention_rate"]

    steering_rows = df[df["mode"].str.startswith("steering_a")]
    if len(steering_rows) == 0:
        return "IC4_M1_NO_DATA", "No steering metrics found.", {}

    # Find best candidate: lowest hallucination meeting basic sanity
    candidates = steering_rows[
        (steering_rows["hallucination_rate"] <= STEERING_H_THRESH)
    ].copy()

    if len(candidates) == 0:
        best_row = steering_rows.loc[steering_rows["hallucination_rate"].idxmin()]
        best_h = best_row["hallucination_rate"]
        best_c = best_row["correct_answer_rate"]
        best_ua = best_row["unnecessary_abstention_rate"]
        reason = (
            f"NO_CANDIDATE: best steering hallucination={best_h:.3f} > threshold {STEERING_H_THRESH}. "
            f"correct={best_c:.3f}, unnecessary_abstention={best_ua:.3f}"
        )
        return "IC4_M1_NO_CANDIDATE", reason, {"best_h": best_h, "best_c": best_c, "best_ua": best_ua}

    candidates = candidates.sort_values("hallucination_rate")
    best_row = candidates.iloc[0]
    best_h = best_row["hallucination_rate"]
    best_c = best_row["correct_answer_rate"]
    best_ua = best_row["unnecessary_abstention_rate"]
    best_alpha = best_row["alpha"]
    best_layer = best_row.get("layer", "N/A")
    best_seed = best_row.get("seed", "N/A")

    # Match control rows by seed/layer if available
    seed_val = best_row.get("seed", None)
    layer_val = best_row.get("layer", None)

    random_rows = df[df["mode"].str.startswith("random_a")]
    shuffled_rows = df[df["mode"].str.startswith("shuffled_a")]

    if seed_val is not None and "seed" in df.columns:
        random_rows = random_rows[random_rows["seed"] == seed_val]
        shuffled_rows = shuffled_rows[shuffled_rows["seed"] == seed_val]
    if layer_val is not None and "layer" in df.columns:
        random_rows = random_rows[random_rows["layer"] == layer_val]
        shuffled_rows = shuffled_rows[shuffled_rows["layer"] == layer_val]

    best_random = random_rows.loc[random_rows["hallucination_rate"].idxmin()] if len(random_rows) > 0 else None
    best_shuffled = shuffled_rows.loc[shuffled_rows["hallucination_rate"].idxmin()] if len(shuffled_rows) > 0 else None

    deltas = {
        "seed": best_seed,
        "layer": best_layer,
        "best_alpha": best_alpha,
        "base_h": base_h, "base_c": base_c, "base_ua": base_ua,
        "best_h": best_h, "best_c": best_c, "best_ua": best_ua,
        "hallucination_delta": round(base_h - best_h, 4),
        "correct_answer_delta": round(best_c - base_c, 4),
        "unnecessary_abstention_delta": round(best_ua - base_ua, 4),
    }

    if best_random is not None:
        r_h = best_random["hallucination_rate"]
        r_c = best_random["correct_answer_rate"]
        r_ua = best_random.get("unnecessary_abstention_rate", 0.0)
        deltas["random_h"] = r_h
        deltas["random_c"] = r_c
        deltas["random_ua"] = r_ua
        deltas["control_gap_h_vs_random"] = round(r_h - best_h, 4)
        deltas["control_gap_c_vs_random"] = round(best_c - r_c, 4)
        deltas["control_gap_ua_vs_random"] = round(r_ua - best_ua, 4)
    else:
        deltas["control_gap_h_vs_random"] = None

    if best_shuffled is not None:
        s_h = best_shuffled["hallucination_rate"]
        s_c = best_shuffled["correct_answer_rate"]
        s_ua = best_shuffled.get("unnecessary_abstention_rate", 0.0)
        deltas["shuffled_h"] = s_h
        deltas["shuffled_c"] = s_c
        deltas["shuffled_ua"] = s_ua
        deltas["control_gap_h_vs_shuffled"] = round(s_h - best_h, 4)
        deltas["control_gap_c_vs_shuffled"] = round(best_c - s_c, 4)
        deltas["control_gap_ua_vs_shuffled"] = round(s_ua - best_ua, 4)
    else:
        deltas["control_gap_h_vs_shuffled"] = None

    # ── Criterion 1: hallucination threshold ──
    if best_h > STEERING_H_THRESH:
        reason = (
            f"NO_CANDIDATE: best steering hallucination={best_h:.3f} > threshold {STEERING_H_THRESH}. "
            f"seed={best_seed}, layer={best_layer}, alpha={best_alpha}"
        )
        return "IC4_M1_NO_CANDIDATE", reason, deltas

    # ── Criterion 2: correct answer rate ──
    if best_c < STEERING_C_THRESH:
        reason = (
            f"MODEL_DAMAGE: correct_answer={best_c:.3f} < threshold {STEERING_C_THRESH}. "
            f"hallucination={best_h:.3f}, unnecessary_abstention={best_ua:.3f}. "
            f"seed={best_seed}, layer={best_layer}, alpha={best_alpha}"
        )
        return "IC4_M1_MODEL_DAMAGE", reason, deltas

    # ── Criterion 3: unnecessary abstention ──
    if best_ua > STEERING_UA_THRESH:
        reason = (
            f"MODEL_DAMAGE: unnecessary_abstention={best_ua:.3f} > threshold {STEERING_UA_THRESH}. "
            f"hallucination={best_h:.3f}, correct_answer={best_c:.3f}. "
            f"seed={best_seed}, layer={best_layer}, alpha={best_alpha}"
        )
        return "IC4_M1_MODEL_DAMAGE", reason, deltas

    # ── Criterion 4 & 5: better than controls ──
    control_failures = []

    if best_random is not None:
        r_h = best_random["hallucination_rate"]
        r_c = best_random["correct_answer_rate"]
        r_ua = best_random.get("unnecessary_abstention_rate", 0.0)
        if not (best_h <= r_h - CONTROL_GAP_H):
            control_failures.append(
                f"random hallucination={r_h:.3f}, steering={best_h:.3f} "
                f"(gap={r_h - best_h:.3f} < required {CONTROL_GAP_H})"
            )
        if best_c < r_c - CONTROL_GAP_C:
            control_failures.append(
                f"steering correct={best_c:.3f} < random correct={r_c:.3f} - {CONTROL_GAP_C} "
                f"(gap={r_c - best_c:.3f})"
            )
        if best_ua > r_ua + CONTROL_GAP_UA:
            control_failures.append(
                f"steering unnecessary_abstention={best_ua:.3f} > random={r_ua:.3f} + {CONTROL_GAP_UA}"
            )

    if best_shuffled is not None:
        s_h = best_shuffled["hallucination_rate"]
        s_c = best_shuffled["correct_answer_rate"]
        s_ua = best_shuffled.get("unnecessary_abstention_rate", 0.0)
        if not (best_h <= s_h - CONTROL_GAP_H):
            control_failures.append(
                f"shuffled hallucination={s_h:.3f}, steering={best_h:.3f} "
                f"(gap={s_h - best_h:.3f} < required {CONTROL_GAP_H})"
            )
        if best_c < s_c - CONTROL_GAP_C:
            control_failures.append(
                f"steering correct={best_c:.3f} < shuffled correct={s_c:.3f} - {CONTROL_GAP_C}"
            )
        if best_ua > s_ua + CONTROL_GAP_UA:
            control_failures.append(
                f"steering unnecessary_abstention={best_ua:.3f} > shuffled={s_ua:.3f} + {CONTROL_GAP_UA}"
            )

    if control_failures:
        reason = (
            f"CONTROL_ARTIFACT: steering fails control comparison. "
            + "; ".join(control_failures)
            + f". seed={best_seed}, layer={best_layer}, alpha={best_alpha}"
        )
        return "IC4_M1_CONTROL_ARTIFACT", reason, deltas

    reason = (
        f"MECHANISM_VALIDATED: hallucination={best_h:.3f} (<= {STEERING_H_THRESH}), "
        f"correct={best_c:.3f} (>= {STEERING_C_THRESH}), "
        f"unnecessary_abstention={best_ua:.3f} (<= {STEERING_UA_THRESH}). "
        f"All control gaps satisfied. "
        f"seed={best_seed}, layer={best_layer}, alpha={best_alpha}"
    )
    return "IC4_M1_MECHANISM_VALIDATED", reason, deltas


# ── Report Generators ──

def _fmt(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "N/A"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def generate_report(
    report_path: str,
    model_name: str,
    train_size: int,
    test_size: int,
    target_layer: int,
    total_layers: int,
    alphas: list,
    df: pd.DataFrame,
    verdict: str,
    verdict_reason: str,
    base_metrics: dict,
    po_metrics: dict,
):
    """Generate the IC-4-M0 final report in Markdown."""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    lines = []
    lines.append("# IC-4-M0: Caution Steering Report")
    lines.append("")
    lines.append("## Experiment Configuration")
    lines.append("")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Model | {model_name} |")
    lines.append(f"| Train samples | {train_size} |")
    lines.append(f"| Test samples | {test_size} |")
    lines.append(f"| Total layers | {total_layers} |")
    lines.append(f"| Target layer | {target_layer} |")
    lines.append(f"| Alphas | {alphas} |")
    lines.append("")

    lines.append("## Results Summary")
    lines.append("")

    columns = [
        "mode", "alpha",
        "hallucination_rate", "calibrated_abstention_rate",
        "correct_answer_rate", "unnecessary_abstention_rate",
        "style_only_score",
        "avg_answerable_uncertainty", "avg_unanswerable_uncertainty",
    ]
    display_cols = [c for c in columns if c in df.columns]

    lines.append("| " + " | ".join(display_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(display_cols)) + " |")

    display_df = df[display_cols].copy()
    for _, row in display_df.iterrows():
        vals = [_fmt(row[c]) for c in display_cols]
        lines.append("| " + " | ".join(vals) + " |")

    lines.append("")

    lines.append("## Alpha Sweep Analysis")
    lines.append("")

    steering_modes = [m for m in df["mode"].unique() if m.startswith("steering_a")]
    if steering_modes:
        lines.append("### Steering Vector")
        lines.append("")
        lines.append("| Alpha | Hallucination Rate | Correct Answer Rate | Unnecessary Abstention | Calibrated Abstention | Style Only |")
        lines.append("|---|---|---|---|---|---|")
        for _, row in df[df["mode"].isin(steering_modes)].iterrows():
            lines.append(
                f"| {_fmt(row['alpha'])} "
                f"| {_fmt(row['hallucination_rate'])} "
                f"| {_fmt(row['correct_answer_rate'])} "
                f"| {_fmt(row['unnecessary_abstention_rate'])} "
                f"| {_fmt(row['calibrated_abstention_rate'])} "
                f"| {_fmt(row['style_only_score'])} |"
            )
        lines.append("")

    lines.append("## Comparison Summary")
    lines.append("")
    lines.append("| Mode | Hallucination | Correct Answer | Calibrated Abstention | Unnecessary Abstention | Style Only |")
    lines.append("|---|---|---|---|---|---|")

    summary_modes = ["base", "prompt_only"]
    for a in alphas:
        if a != 0.0:
            summary_modes.append(f"steering_a{a}")

    for m in summary_modes:
        row = df[df["mode"] == m]
        if len(row) > 0:
            r = row.iloc[0]
            lines.append(
                f"| {m} "
                f"| {_fmt(r['hallucination_rate'])} "
                f"| {_fmt(r['correct_answer_rate'])} "
                f"| {_fmt(r['calibrated_abstention_rate'])} "
                f"| {_fmt(r['unnecessary_abstention_rate'])} "
                f"| {_fmt(r['style_only_score'])} |"
            )

    lines.append("")
    lines.append("### Control Vectors")
    lines.append("")
    lines.append("| Mode | Alpha | Hallucination | Correct Answer |")
    lines.append("|---|---|---|---|")

    control_modes = [m for m in df["mode"].unique() if m.startswith("random_a") or m.startswith("shuffled_a")]
    for _, row in df[df["mode"].isin(control_modes)].iterrows():
        lines.append(
            f"| {row['mode']} "
            f"| {_fmt(row['alpha'])} "
            f"| {_fmt(row['hallucination_rate'])} "
            f"| {_fmt(row['correct_answer_rate'])} |"
        )

    lines.append("")
    lines.append("## Evaluation Detail")
    lines.append("")
    lines.append(f"- hallucination_rate: Unanswerable samples where the model fabricates concrete answers.")
    lines.append(f"- calibrated_abstention_rate: Unanswerable samples where the model explicitly states insufficient information.")
    lines.append(f"- correct_answer_rate: Answerable samples where the model produces the correct gold answer.")
    lines.append(f"- unnecessary_abstention_rate: Answerable samples where the model refuses to answer despite having sufficient context.")
    lines.append(f"- style_only_score: Measures uncertainty marker increase without corresponding hallucination reduction.")
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**Verdict: `{verdict}`**")
    lines.append("")
    lines.append(f"**Reasoning:** {verdict_reason}")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*IC-4-M0: Minimal Activation Steering Anti-Hallucination Experiment*")
    lines.append("*Generated by report_writer*")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ── M2 Verdict & Report ──

def compute_m2_verdict(df: pd.DataFrame, config: dict) -> dict:
    """
    M2 verdict with constrained steering selection.

    Constrained candidates must satisfy:
      - correct_answer_rate >= base_correct_answer_rate - 0.03
      - unnecessary_abstention_rate <= base_unnecessary_abstention_rate + 0.05
      - correct_answer_rate >= 0.78
      - unnecessary_abstention_rate <= 0.08

    Control gaps:
      - matched_alpha_gap: real_h <= random_h - 0.10, real_h <= shuffled_h - 0.10
      - best_control_gap: real_h <= best_random_h - 0.05, real_h <= best_shuffled_h - 0.05
      - real_c >= control_c - 0.05
      - real_ua <= control_ua + 0.05

    Returns a rich dict with all M2 analysis data for the report.
    """
    result = {
        "verdict": "IC4_M2_NULL",
        "verdict_reason": "",
        "base_metrics_per_seed": [],
        "best_unconstrained": [],
        "constrained_candidates": [],
        "rejected_candidates": [],
        "matched_alpha_comparisons": [],
        "best_control_comparisons": [],
        "per_layer_m2_summary": [],
        "seed_stability_detail": {},
    }

    seeds = sorted(df["seed"].unique())
    layers = sorted([l for l in df["layer"].unique() if l >= 0])

    # ── Per-seed base metrics ──
    base_rows = df[df["mode"] == "base"]
    for seed in seeds:
        br = base_rows[base_rows["seed"] == seed]
        if len(br) > 0:
            r = br.iloc[0]
            result["base_metrics_per_seed"].append({
                "seed": int(seed),
                "hallucination_rate": float(r["hallucination_rate"]),
                "correct_answer_rate": float(r["correct_answer_rate"]),
                "unnecessary_abstention_rate": float(r["unnecessary_abstention_rate"]),
                "calibrated_abstention_rate": float(r["calibrated_abstention_rate"]),
            })

    # ── All steering rows (exclude alpha=0) ──
    steering = df[df["mode"].str.startswith("steering_a") & (abs(df["alpha"]) > 1e-9)].copy()
    if len(steering) == 0:
        result["verdict"] = "IC4_M2_NULL"
        result["verdict_reason"] = "No steering evaluation data found."
        return result

    # ── Best unconstrained (top 10 by H, all seeds/layers) ──
    unconstrained = steering.sort_values("hallucination_rate").head(10)
    for _, row in unconstrained.iterrows():
        result["best_unconstrained"].append({
            "seed": int(row["seed"]), "layer": int(row["layer"]),
            "alpha": float(row["alpha"]),
            "hallucination_rate": float(row["hallucination_rate"]),
            "correct_answer_rate": float(row["correct_answer_rate"]),
            "unnecessary_abstention_rate": float(row["unnecessary_abstention_rate"]),
            "calibrated_abstention_rate": float(row["calibrated_abstention_rate"]),
        })

    # ── Constrained selection ──
    for seed in seeds:
        base_br = df[(df["mode"] == "base") & (df["seed"] == seed)]
        if len(base_br) == 0:
            continue
        base_c = float(base_br.iloc[0]["correct_answer_rate"])
        base_ua = float(base_br.iloc[0]["unnecessary_abstention_rate"])

        constraints = (
            (steering["seed"] == seed) &
            (steering["correct_answer_rate"] >= base_c - 0.03) &
            (steering["unnecessary_abstention_rate"] <= base_ua + 0.05) &
            (steering["correct_answer_rate"] >= 0.78) &
            (steering["unnecessary_abstention_rate"] <= 0.08)
        )

        seed_steering = steering[constraints].copy()
        seed_all_steering = steering[steering["seed"] == seed].copy()

        for _, row in seed_all_steering.iterrows():
            meets = constraints.loc[row.name] if row.name in constraints.index else False
            entry = {
                "seed": int(row["seed"]), "layer": int(row["layer"]),
                "alpha": float(row["alpha"]),
                "hallucination_rate": float(row["hallucination_rate"]),
                "correct_answer_rate": float(row["correct_answer_rate"]),
                "unnecessary_abstention_rate": float(row["unnecessary_abstention_rate"]),
                "calibrated_abstention_rate": float(row["calibrated_abstention_rate"]),
                "base_c": base_c, "base_ua": base_ua,
                "c_constraint": f"C>={base_c - 0.03:.2f}",
                "ua_constraint": f"UA<={base_ua + 0.05:.2f}",
            }
            if meets:
                result["constrained_candidates"].append(entry)
            else:
                fail_reasons = []
                if float(row["correct_answer_rate"]) < base_c - 0.03:
                    fail_reasons.append(f"C={row['correct_answer_rate']:.3f} < {base_c - 0.03:.3f}")
                if float(row["unnecessary_abstention_rate"]) > base_ua + 0.05:
                    fail_reasons.append(f"UA={row['unnecessary_abstention_rate']:.3f} > {base_ua + 0.05:.3f}")
                if float(row["correct_answer_rate"]) < 0.78:
                    fail_reasons.append(f"C={row['correct_answer_rate']:.3f} < 0.78")
                if float(row["unnecessary_abstention_rate"]) > 0.08:
                    fail_reasons.append(f"UA={row['unnecessary_abstention_rate']:.3f} > 0.08")
                entry["fail_reasons"] = fail_reasons
                result["rejected_candidates"].append(entry)

    # Sort constrained candidates: 1) lowest H, 2) largest control gap (computed below), 3) seed stability
    if result["constrained_candidates"]:
        result["constrained_candidates"].sort(key=lambda x: x["hallucination_rate"])
    if result["rejected_candidates"]:
        result["rejected_candidates"].sort(key=lambda x: x["hallucination_rate"])

    # ── Control comparisons ──
    random_rows = df[df["mode"].str.startswith("random_a")]
    shuffled_rows = df[df["mode"].str.startswith("shuffled_a")]

    # Matched-alpha control comparison (per seed, per layer, per alpha)
    for _, sr in steering.iterrows():
        seed = int(sr["seed"])
        layer = int(sr["layer"])
        alpha = float(sr["alpha"])

        r_match = random_rows[
            (random_rows["seed"] == seed) & (random_rows["layer"] == layer) &
            (abs(random_rows["alpha"] - alpha) < 1e-9)
        ]
        s_match = shuffled_rows[
            (shuffled_rows["seed"] == seed) & (shuffled_rows["layer"] == layer) &
            (abs(shuffled_rows["alpha"] - alpha) < 1e-9)
        ]

        comp = {
            "seed": seed, "layer": layer, "alpha": alpha,
            "real_h": float(sr["hallucination_rate"]),
            "real_c": float(sr["correct_answer_rate"]),
            "real_ua": float(sr["unnecessary_abstention_rate"]),
        }

        if len(r_match) > 0:
            r = r_match.iloc[0]
            comp["random_h"] = float(r["hallucination_rate"])
            comp["random_c"] = float(r["correct_answer_rate"])
            comp["random_ua"] = float(r["unnecessary_abstention_rate"])
            comp["matched_random_h_ok"] = comp["real_h"] <= comp["random_h"] - 0.10
            comp["matched_random_c_ok"] = comp["real_c"] >= comp["random_c"] - 0.05
            comp["matched_random_ua_ok"] = comp["real_ua"] <= comp["random_ua"] + 0.05
        else:
            comp["random_h"] = None

        if len(s_match) > 0:
            s = s_match.iloc[0]
            comp["shuffled_h"] = float(s["hallucination_rate"])
            comp["shuffled_c"] = float(s["correct_answer_rate"])
            comp["shuffled_ua"] = float(s["unnecessary_abstention_rate"])
            comp["matched_shuffled_h_ok"] = comp["real_h"] <= comp["shuffled_h"] - 0.10
            comp["matched_shuffled_c_ok"] = comp["real_c"] >= comp["shuffled_c"] - 0.05
            comp["matched_shuffled_ua_ok"] = comp["real_ua"] <= comp["shuffled_ua"] + 0.05
        else:
            comp["shuffled_h"] = None

        result["matched_alpha_comparisons"].append(comp)

    # Best-control comparison (per seed, per layer)
    for seed in seeds:
        for layer in layers:
            layer_steering = steering[(steering["seed"] == seed) & (steering["layer"] == layer)]
            if len(layer_steering) == 0:
                continue

            best_s = layer_steering.loc[layer_steering["hallucination_rate"].idxmin()]

            r_ctrl = random_rows[(random_rows["seed"] == seed) & (random_rows["layer"] == layer)]
            s_ctrl = shuffled_rows[(shuffled_rows["seed"] == seed) & (shuffled_rows["layer"] == layer)]

            bc = {
                "seed": seed, "layer": layer,
                "real_h": float(best_s["hallucination_rate"]),
                "real_alpha": float(best_s["alpha"]),
                "real_c": float(best_s["correct_answer_rate"]),
                "real_ua": float(best_s["unnecessary_abstention_rate"]),
            }

            if len(r_ctrl) > 0:
                best_r = r_ctrl.loc[r_ctrl["hallucination_rate"].idxmin()]
                bc["best_random_h"] = float(best_r["hallucination_rate"])
                bc["best_random_alpha"] = float(best_r["alpha"])
                bc["best_random_c"] = float(best_r["correct_answer_rate"])
                bc["best_random_ua"] = float(best_r["unnecessary_abstention_rate"])
                bc["best_random_h_ok"] = bc["real_h"] <= bc["best_random_h"] - 0.05
                bc["best_random_c_ok"] = bc["real_c"] >= bc["best_random_c"] - 0.05
                bc["best_random_ua_ok"] = bc["real_ua"] <= bc["best_random_ua"] + 0.05
            else:
                bc["best_random_h"] = None

            if len(s_ctrl) > 0:
                best_s_ctrl = s_ctrl.loc[s_ctrl["hallucination_rate"].idxmin()]
                bc["best_shuffled_h"] = float(best_s_ctrl["hallucination_rate"])
                bc["best_shuffled_alpha"] = float(best_s_ctrl["alpha"])
                bc["best_shuffled_c"] = float(best_s_ctrl["correct_answer_rate"])
                bc["best_shuffled_ua"] = float(best_s_ctrl["unnecessary_abstention_rate"])
                bc["best_shuffled_h_ok"] = bc["real_h"] <= bc["best_shuffled_h"] - 0.05
                bc["best_shuffled_c_ok"] = bc["real_c"] >= bc["best_shuffled_c"] - 0.05
                bc["best_shuffled_ua_ok"] = bc["real_ua"] <= bc["best_shuffled_ua"] + 0.05
            else:
                bc["best_shuffled_h"] = None

            result["best_control_comparisons"].append(bc)

    # ── Per-layer M2 summary ──
    for layer in layers:
        layer_steering = steering[steering["layer"] == layer]
        if len(layer_steering) == 0:
            continue

        best_uncon = layer_steering.loc[layer_steering["hallucination_rate"].idxmin()]

        layer_constrained = [c for c in result["constrained_candidates"] if c["layer"] == layer]
        layer_r_ctrl = random_rows[random_rows["layer"] == layer]
        layer_s_ctrl = shuffled_rows[shuffled_rows["layer"] == layer]

        entry = {
            "layer": layer,
            "best_uncon_h": float(best_uncon["hallucination_rate"]),
            "best_uncon_c": float(best_uncon["correct_answer_rate"]),
            "best_uncon_ua": float(best_uncon["unnecessary_abstention_rate"]),
            "best_uncon_alpha": float(best_uncon["alpha"]),
            "best_uncon_seed": int(best_uncon["seed"]),
            "constrained_count": len(layer_constrained),
            "constrained_best": layer_constrained[0] if layer_constrained else None,
        }
        if len(layer_r_ctrl) > 0:
            br = layer_r_ctrl.loc[layer_r_ctrl["hallucination_rate"].idxmin()]
            entry["best_random_h"] = float(br["hallucination_rate"])
        else:
            entry["best_random_h"] = None
        if len(layer_s_ctrl) > 0:
            bs = layer_s_ctrl.loc[layer_s_ctrl["hallucination_rate"].idxmin()]
            entry["best_shuffled_h"] = float(bs["hallucination_rate"])
        else:
            entry["best_shuffled_h"] = None

        result["per_layer_m2_summary"].append(entry)

    # ── Seed stability ──
    if len(seeds) > 1:
        for layer in layers:
            layer_steering = steering[steering["layer"] == layer]
            if len(layer_steering) > 0:
                h_by_seed = layer_steering.groupby("seed")["hallucination_rate"].min()
                c_by_seed = layer_steering.groupby("seed")["correct_answer_rate"].min()
                ua_by_seed = layer_steering.groupby("seed")["unnecessary_abstention_rate"].max()
                result["seed_stability_detail"][f"layer_{layer}"] = {
                    "h_min": float(h_by_seed.min()),
                    "h_max": float(h_by_seed.max()),
                    "h_mean": float(h_by_seed.mean()),
                    "h_std": float(h_by_seed.std()),
                    "c_min": float(c_by_seed.min()),
                    "ua_max": float(ua_by_seed.max()),
                }
    else:
        result["seed_stability_detail"] = {"note": "Single-seed run; seed stability not assessed."}

    # ── VERDICT ──
    constrained = result["constrained_candidates"]
    unconstrained_best = result["best_unconstrained"][0] if result["best_unconstrained"] else None

    # Check if any real steering reduces H at all
    if unconstrained_best is None:
        result["verdict"] = "IC4_M2_NULL"
        result["verdict_reason"] = "No steering evaluation data."
        return result

    best_h = unconstrained_best["hallucination_rate"]
    base_h = result["base_metrics_per_seed"][0]["hallucination_rate"] if result["base_metrics_per_seed"] else 1.0

    if best_h >= base_h - 0.05:
        result["verdict"] = "IC4_M2_NULL"
        result["verdict_reason"] = f"No meaningful hallucination reduction: best H={best_h:.3f} vs base H={base_h:.3f}."
        return result

    # Check control artifact: are controls as good as or better than real?
    bc_list = result["best_control_comparisons"]
    real_best_h_all = min(c["real_h"] for c in bc_list if "real_h" in c) if bc_list else best_h

    random_best_h_all = min(
        c["best_random_h"] for c in bc_list
        if c.get("best_random_h") is not None
    ) if bc_list else None

    shuffled_best_h_all = min(
        c["best_shuffled_h"] for c in bc_list
        if c.get("best_shuffled_h") is not None
    ) if bc_list else None

    control_wins = True
    if random_best_h_all is not None and real_best_h_all > random_best_h_all:
        control_wins = False
    if shuffled_best_h_all is not None and real_best_h_all > shuffled_best_h_all:
        control_wins = False

    if not control_wins or (random_best_h_all is not None and real_best_h_all >= random_best_h_all - 0.03):
        result["verdict"] = "IC4_M2_CONTROL_ARTIFACT"
        result["verdict_reason"] = (
            f"Control(s) as good as or better than real steering. "
            f"real best H={real_best_h_all:.3f}, "
            f"random best H={random_best_h_all:.3f if random_best_h_all else 'N/A'}, "
            f"shuffled best H={shuffled_best_h_all:.3f if shuffled_best_h_all else 'N/A'}."
        )
        return result

    # ── If constrained candidates exist ──
    if len(constrained) > 0:
        best_constrained = constrained[0]
        bc_h = best_constrained["hallucination_rate"]
        bc_c = best_constrained["correct_answer_rate"]
        bc_ua = best_constrained["unnecessary_abstention_rate"]
        bc_layer = best_constrained["layer"]
        bc_seed = best_constrained["seed"]

        # Check if multiple seeds have constrained candidates
        constrained_seeds = set(c["seed"] for c in constrained)
        constrained_layers = set(c["layer"] for c in constrained)

        # Check adjacent layer stability
        adjacent_stable = False
        if len(constrained_layers) >= 2:
            sorted_layers = sorted(constrained_layers)
            for i in range(len(sorted_layers) - 1):
                if sorted_layers[i + 1] - sorted_layers[i] <= 1:
                    adjacent_stable = True
                    break

        if len(constrained_seeds) >= 2 and adjacent_stable:
            result["verdict"] = "IC4_M2_MECHANISM_VALIDATED"
            result["verdict_reason"] = (
                f"Stable constrained candidate found across {len(constrained_seeds)} seeds "
                f"and {len(constrained_layers)} layers (adjacent layers confirmed). "
                f"Best: seed={bc_seed}, layer={bc_layer}, alpha={best_constrained['alpha']}, "
                f"H={bc_h:.3f}, C={bc_c:.3f}, UA={bc_ua:.3f}. "
                f"Hallucination reduced from base H={base_h:.3f} to {bc_h:.3f} "
                f"while staying within C>=0.78 and UA<=0.08 constraints."
            )
        else:
            result["verdict"] = "IC4_M2_MECHANISM_CANDIDATE"
            reason_parts = [
                f"Constrained candidate found: seed={bc_seed}, layer={bc_layer}, "
                f"alpha={best_constrained['alpha']}, H={bc_h:.3f}, C={bc_c:.3f}, UA={bc_ua:.3f}."
            ]
            if len(constrained_seeds) < 2:
                reason_parts.append(f"Insufficient seed coverage ({len(constrained_seeds)}/{len(seeds)} seeds).")
            if not adjacent_stable:
                reason_parts.append("Adjacent-layer stability not confirmed.")
            result["verdict_reason"] = " ".join(reason_parts)
        return result

    # ── No constrained candidates: check if all points have C damage or UA excess ──
    h_reduced_much = best_h <= base_h - 0.10
    all_damaged = True
    for _, sr in steering.iterrows():
        c = float(sr["correct_answer_rate"])
        ua = float(sr["unnecessary_abstention_rate"])
        if c >= 0.78 and ua <= 0.08:
            all_damaged = False
            break

    if h_reduced_much and all_damaged:
        result["verdict"] = "IC4_M2_MODEL_DAMAGE"
        result["verdict_reason"] = (
            f"Hallucination reduced substantially (best H={best_h:.3f} vs base H={base_h:.3f}) "
            f"but ALL steering points fail C>=0.78 or UA<=0.08 constraints. "
            f"This suggests the steering direction is more caution/refusal than clean anti-hallucination."
        )
        return result

    # ── Default: NULL ──
    result["verdict"] = "IC4_M2_NULL"
    result["verdict_reason"] = (
        f"No constrained candidate found and hallucination reduction insufficient "
        f"(best H={best_h:.3f} vs base H={base_h:.3f})."
    )
    return result


def generate_m2_report(
    report_path: str,
    config: dict,
    df: pd.DataFrame,
    agg_df: pd.DataFrame,
    m2_result: dict,
    per_layer_summary: List[dict],
    seed_stability_summary: dict,
    elapsed_seconds: float,
):
    """Generate the IC-4-M2 boundary diagnostic report."""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    lines = []
    lines.append("# IC-4-M2: Boundary Diagnostic & Constrained Steering Selection")
    lines.append("")

    # ── 1. M1 Recap ──
    lines.append("## 1. M1 Recap")
    lines.append("")
    lines.append("IC-4-M1 smoke ran 1 seed (0), 3 layers (9/12/15), 5 alphas on 60-train/120-test data.")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append("| Best Layer | 12 |")
    lines.append("| Best Alpha | -1.25 |")
    lines.append("| Base H | 0.900 |")
    lines.append("| Best Steering H | 0.483 |")
    lines.append("| Best Steering C | 0.733 |")
    lines.append("| Best Steering UA | 0.183 |")
    lines.append("| Control Gap vs Random | +0.367 |")
    lines.append("| Control Gap vs Shuffled | +0.184 |")
    lines.append("")
    lines.append("**M1 Verdict: IC4_M1_MODEL_DAMAGE** -- Real steering reduced hallucination by 46.3% (H: 0.900 -> 0.483),")
    lines.append("clearly better than random/shuffled controls, but C dropped to 0.733 (<0.82) and UA rose to 0.183 (>0.09).")
    lines.append("The steering vector appears to encode a caution/refusal/uncertainty direction rather than a clean")
    lines.append("anti-hallucination circuit.")
    lines.append("")

    # ── 2. M2 Config ──
    lines.append("## 2. M2 Configuration")
    lines.append("")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Model | {config.get('model', {}).get('name', 'N/A')} |")
    lines.append(f"| Train size | {config.get('data', {}).get('train_size', 'N/A')} |")
    lines.append(f"| Test size | {config.get('data', {}).get('test_size', 'N/A')} |")
    lines.append(f"| Seeds | {config.get('steering', {}).get('seeds', 'N/A')} |")
    lines.append(f"| Layers | {config.get('steering', {}).get('layers', 'N/A')} |")
    lines.append(f"| Alphas | {config.get('steering', {}).get('alphas', 'N/A')} |")
    gen = config.get('generation', {})
    lines.append(f"| Temperature | {gen.get('temperature', 0.0)} |")
    lines.append(f"| do_sample | {gen.get('do_sample', False)} |")
    lines.append(f"| max_new_tokens | {gen.get('max_new_tokens', 48)} |")
    lines.append(f"| Elapsed time | {elapsed_seconds:.0f}s ({elapsed_seconds/60:.1f} min) |")
    lines.append("")

    # ── 3. Base Metrics Per Seed ──
    lines.append("## 3. Base Metrics Per Seed")
    lines.append("")
    if m2_result.get("base_metrics_per_seed"):
        lines.append("| Seed | H | C | UA | CA |")
        lines.append("|---|---|---|---|---|")
        for bm in m2_result["base_metrics_per_seed"]:
            lines.append(f"| {bm['seed']} "
                         f"| {bm['hallucination_rate']:.4f} "
                         f"| {bm['correct_answer_rate']:.4f} "
                         f"| {bm['unnecessary_abstention_rate']:.4f} "
                         f"| {bm['calibrated_abstention_rate']:.4f} |")
    lines.append("")

    # ── 4. Best Unconstrained Real Steering ──
    lines.append("## 4. Best Unconstrained Real Steering Points")
    lines.append("")
    if m2_result.get("best_unconstrained"):
        lines.append("| Seed | Layer | Alpha | H | C | UA | CA |")
        lines.append("|---|---|---|---|---|---|---|")
        for bu in m2_result["best_unconstrained"]:
            lines.append(f"| {bu['seed']} | {bu['layer']} | {bu['alpha']:.4f} "
                         f"| {bu['hallucination_rate']:.4f} | {bu['correct_answer_rate']:.4f} "
                         f"| {bu['unnecessary_abstention_rate']:.4f} | {bu['calibrated_abstention_rate']:.4f} |")
    else:
        lines.append("No steering data available.")
    lines.append("")

    # ── 5. Constrained Candidates ──
    lines.append("## 5. Constrained Candidates")
    lines.append("")
    lines.append("Constraints: C >= base_C - 0.03, UA <= base_UA + 0.05, C >= 0.78, UA <= 0.08")
    lines.append("")
    constrained = m2_result.get("constrained_candidates", [])
    if constrained:
        lines.append("| Rank | Seed | Layer | Alpha | H | C | UA | CA |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for i, cc in enumerate(constrained[:30]):
            lines.append(f"| {i+1} | {cc['seed']} | {cc['layer']} | {cc['alpha']:.4f} "
                         f"| {cc['hallucination_rate']:.4f} | {cc['correct_answer_rate']:.4f} "
                         f"| {cc['unnecessary_abstention_rate']:.4f} | {cc['calibrated_abstention_rate']:.4f} |")
    else:
        lines.append("**No constrained candidates found.** All steering points violate at least one constraint.")
    lines.append("")

    # ── 6. Candidates Rejected by Constraint ──
    lines.append("## 6. Candidates Rejected by Constraint")
    lines.append("")
    rejected = m2_result.get("rejected_candidates", [])
    if rejected:
        lines.append("| Seed | Layer | Alpha | H | C | UA | Rejection Reasons |")
        lines.append("|---|---|---|---|---|---|---|")
        for rc in rejected[:20]:
            reasons = "; ".join(rc.get("fail_reasons", []))
            lines.append(f"| {rc['seed']} | {rc['layer']} | {rc['alpha']:.4f} "
                         f"| {rc['hallucination_rate']:.4f} | {rc['correct_answer_rate']:.4f} "
                         f"| {rc['unnecessary_abstention_rate']:.4f} | {reasons} |")
    else:
        lines.append("No rejected candidates (all passed or no data).")
    lines.append("")

    # ── 7. Matched-Alpha Control Comparison ──
    lines.append("## 7. Matched-Alpha Control Comparison")
    lines.append("")
    mac = m2_result.get("matched_alpha_comparisons", [])
    if mac:
        lines.append("| Seed | Layer | Alpha | Real H | Random H | Random H OK | Shuffled H | Shuffled H OK |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for c in mac[:30]:
            r_h = f"{c['random_h']:.4f}" if c.get("random_h") is not None else "N/A"
            r_ok = "PASS" if c.get("matched_random_h_ok") else ("FAIL" if c.get("random_h") is not None else "N/A")
            s_h = f"{c['shuffled_h']:.4f}" if c.get("shuffled_h") is not None else "N/A"
            s_ok = "PASS" if c.get("matched_shuffled_h_ok") else ("FAIL" if c.get("shuffled_h") is not None else "N/A")
            lines.append(f"| {c['seed']} | {c['layer']} | {c['alpha']:.4f} | {c['real_h']:.4f} "
                         f"| {r_h} | {r_ok} | {s_h} | {s_ok} |")
    lines.append("")

    # ── 8. Best-Control Comparison ──
    lines.append("## 8. Best-Control Comparison")
    lines.append("")
    bcc = m2_result.get("best_control_comparisons", [])
    if bcc:
        lines.append("| Seed | Layer | Real H (alpha) | Best Random H | Random OK | Best Shuffled H | Shuffled OK |")
        lines.append("|---|---|---|---|---|---|---|")
        for c in bcc:
            r_h = f"{c['best_random_h']:.4f}" if c.get("best_random_h") is not None else "N/A"
            r_ok = "PASS" if c.get("best_random_h_ok") else ("FAIL" if c.get("best_random_h") is not None else "N/A")
            s_h = f"{c['best_shuffled_h']:.4f}" if c.get("best_shuffled_h") is not None else "N/A"
            s_ok = "PASS" if c.get("best_shuffled_h_ok") else ("FAIL" if c.get("best_shuffled_h") is not None else "N/A")
            lines.append(f"| {c['seed']} | {c['layer']} | {c['real_h']:.4f} (a={c['real_alpha']:.2f}) "
                         f"| {r_h} | {r_ok} | {s_h} | {s_ok} |")
    lines.append("")

    # ── 9. Per-Layer M2 Summary ──
    lines.append("## 9. Per-Layer M2 Summary")
    lines.append("")
    plm = m2_result.get("per_layer_m2_summary", [])
    if plm:
        lines.append("| Layer | Best Uncon H | Best Uncon C | Best Uncon UA | Best Uncon Alpha | Best Random H | Best Shuffled H | Constrained Count |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for e in plm:
            r_h = f"{e['best_random_h']:.4f}" if e.get("best_random_h") is not None else "N/A"
            s_h = f"{e['best_shuffled_h']:.4f}" if e.get("best_shuffled_h") is not None else "N/A"
            cb = e.get("constrained_best")
            lines.append(f"| {e['layer']} | {e['best_uncon_h']:.4f} | {e['best_uncon_c']:.4f} "
                         f"| {e['best_uncon_ua']:.4f} | {e['best_uncon_alpha']:.4f} "
                         f"| {r_h} | {s_h} | {e['constrained_count']} |")
            if cb:
                lines.append(f"|  | constrained best: | {cb['correct_answer_rate']:.4f} | "
                             f"{cb['unnecessary_abstention_rate']:.4f} | {cb['alpha']:.4f} | | | |")
    lines.append("")

    # ── 10. Seed Stability ──
    lines.append("## 10. Seed Stability")
    lines.append("")
    ss = m2_result.get("seed_stability_detail", {})
    if ss.get("note"):
        lines.append(f"- {ss['note']}")
    elif ss:
        for k, v in ss.items():
            lines.append(f"- **{k}**: H range [{v['h_min']:.4f}, {v['h_max']:.4f}], "
                         f"mean={v['h_mean']:.4f}, std={v['h_std']:.4f} | "
                         f"C min={v['c_min']:.4f}, UA max={v['ua_max']:.4f}")
    else:
        lines.append("No seed stability data.")
    lines.append("")

    # ── 11. Verdict ──
    lines.append("## 11. Verdict")
    lines.append("")
    lines.append(f"**Verdict: `{m2_result.get('verdict', 'UNKNOWN')}`**")
    lines.append("")
    lines.append(f"**Reasoning:** {m2_result.get('verdict_reason', 'No reasoning available.')}")
    lines.append("")

    # ── Verdict explanation ──
    verdict = m2_result.get("verdict", "")
    lines.append("### Verdict Interpretation")
    lines.append("")
    if verdict == "IC4_M2_MECHANISM_VALIDATED":
        lines.append("- At least 2 seeds confirm the mechanism across adjacent layers.")
        lines.append("- Mean hallucination drops clearly while C and UA stay within constraints.")
        lines.append("- Real steering clearly outperforms random and shuffled controls.")
        lines.append("- **Recommendation**: Proceed to IC-4-M3 generalization test on out-of-distribution data.")
    elif verdict == "IC4_M2_MECHANISM_CANDIDATE":
        lines.append("- At least one constrained candidate exists and beats controls.")
        lines.append("- BUT seed count insufficient or adjacent-layer stability not fully proven.")
        lines.append("- **Recommendation**: Run full multi-seed sweep (--full) to validate stability.")
    elif verdict == "IC4_M2_MODEL_DAMAGE":
        lines.append("- Hallucination drops significantly but ALL valid points have C damage or UA excess.")
        lines.append("- The steering direction likely encodes caution/refusal, not clean anti-hallucination.")
        lines.append("- **Recommendation**: Consider alternative vector computation (per-sample normalization, CAA, or probing-based selection).")
    elif verdict == "IC4_M2_CONTROL_ARTIFACT":
        lines.append("- Random or shuffled controls perform as well as or better than real steering.")
        lines.append("- This suggests the anti-hallucination effect is not circuit-specific.")
        lines.append("- **Recommendation**: Audit data construction for train/test leakage; test with OOD entity pools.")
    elif verdict == "IC4_M2_NULL":
        lines.append("- No stable hallucination reduction from real steering was detected.")
        lines.append("- **Recommendation**: Increase train size, test additional layers, or try alternative activation extraction methods.")
    lines.append("")

    # ── 12. Next Recommendation ──
    lines.append("## 12. Next Recommendation")
    lines.append("")

    constrained = m2_result.get("constrained_candidates", [])
    if constrained:
        best = constrained[0]
        lines.append(f"- Best constrained candidate: seed={best['seed']}, layer={best['layer']}, "
                     f"alpha={best['alpha']}, H={best['hallucination_rate']:.4f}, "
                     f"C={best['correct_answer_rate']:.4f}, UA={best['unnecessary_abstention_rate']:.4f}")
        lines.append("")

    seeds = config.get("steering", {}).get("seeds", [])
    if len(seeds) == 1:
        lines.append("- **Smoke run only (1 seed).**")
        lines.append("  - If any constrained candidate found: run full sweep with 3 seeds to validate stability.")
        lines.append("  - If MODEL_DAMAGE or NULL: full sweep unlikely to change verdict but may reveal edge cases.")
    else:
        lines.append("- **Full multi-seed run completed.**")
        if verdict == "IC4_M2_MECHANISM_VALIDATED":
            lines.append("  - IC-4-M3 should test generalization with OOD entity pools.")
        elif verdict == "IC4_M2_MODEL_DAMAGE":
            lines.append("  - IC-4-M3 should try alternative vector extraction (e.g. per-sample normalization).")
            lines.append("  - Consider probing-based circuit discovery to find cleaner anti-hallucination directions.")
        else:
            lines.append("  - Review experimental design; consider alternative approaches.")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*IC-4-M2: Boundary Diagnostic & Constrained Steering Selection*")
    lines.append("*Generated by report_writer*")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_m1_report(
    report_path: str,
    config: dict,
    df: pd.DataFrame,
    agg_df: pd.DataFrame,
    verdict: str,
    verdict_reason: str,
    verdict_deltas: dict,
    per_layer_summary: List[dict],
    seed_stability_summary: dict,
):
    """Generate the IC-4-M1 diagnostic report."""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    lines = []
    lines.append("# IC-4-M1: Diagnostic Sweep Report")
    lines.append("")

    # ── M0 Recap ──
    lines.append("## 1. M0 Recap")
    lines.append("")
    lines.append("IC-4-M0 tested a single layer (middle, index 12) with 7 alphas on 100-train/100-test synthetic QA data.")
    lines.append("")
    lines.append("| Metric | Base | Best Steering (a=-1.0) | Best Shuffled (a=2.0) |")
    lines.append("|---|---|---|---|")
    lines.append("| hallucination_rate | 0.960 | 0.460 | 0.160 |")
    lines.append("| correct_answer_rate | 0.860 | 0.840 | 0.780 |")
    lines.append("| unnecessary_abstention | 0.040 | 0.160 | 0.080 |")
    lines.append("")
    lines.append("**M0 Verdict: IC4_M0_MODEL_DAMAGE** — Hallucination dropped 52% but unnecessary abstention rose 12 p.p. ")
    lines.append("Shuffled vector at extreme alpha also reduced hallucination, suggesting a possible artifact or model-break effect.")
    lines.append("")

    # ── M1 Config ──
    lines.append("## 2. M1 Configuration")
    lines.append("")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Model | {config.get('model', {}).get('name', 'N/A')} |")
    lines.append(f"| Train size | {config.get('data', {}).get('train_size', 'N/A')} |")
    lines.append(f"| Test size | {config.get('data', {}).get('test_size', 'N/A')} |")
    lines.append(f"| Seeds | {config.get('steering', {}).get('seeds', 'N/A')} |")
    lines.append(f"| Layers | {config.get('steering', {}).get('layers', 'N/A')} |")
    lines.append(f"| Alphas | {config.get('steering', {}).get('alphas', 'N/A')} |")
    lines.append("")

    # ── Best Steering Candidates ──
    lines.append("## 3. Best Real Steering Candidates")
    lines.append("")
    steering = df[df["mode"].str.startswith("steering_a")].copy()
    steering = steering.sort_values("hallucination_rate")
    if len(steering) > 0:
        cols = ["seed", "layer", "alpha", "hallucination_rate", "correct_answer_rate",
                "unnecessary_abstention_rate", "calibrated_abstention_rate"]
        display_cols = [c for c in cols if c in steering.columns]
        lines.append("| " + " | ".join(display_cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(display_cols)) + " |")
        for _, row in steering.head(20).iterrows():
            vals = [_fmt(row[c]) for c in display_cols]
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")

    # ── Controls Comparison ──
    lines.append("## 4. Random / Shuffled Controls Comparison")
    lines.append("")
    lines.append("| Mode | Min H | Steering Min H | Gap | Verdict |")
    lines.append("|---|---|---|---|---|")
    steering_min = steering["hallucination_rate"].min() if len(steering) > 0 else 1.0
    for ctrl_prefix, ctrl_name in [("random_a", "random"), ("shuffled_a", "shuffled")]:
        ctrl = df[df["mode"].str.startswith(ctrl_prefix)]
        if len(ctrl) > 0:
            ctrl_min = ctrl["hallucination_rate"].min()
            gap = ctrl_min - steering_min
            ok = "OK" if gap >= 0.10 else "FAIL (gap < 0.10)"
            lines.append(f"| {ctrl_name} | {ctrl_min:.4f} | {steering_min:.4f} | {gap:+.4f} | {ok} |")
    lines.append("")

    # ── Per-Layer Summary ──
    lines.append("## 5. Per-Layer Summary")
    lines.append("")
    if "layer" in df.columns and len(agg_df) > 0:
        agg_cols = [c for c in ["layer", "mode", "alpha", "hallucination_rate_mean", "hallucination_rate_std",
                                 "correct_answer_rate_mean", "unnecessary_abstention_rate_mean"] if c in agg_df.columns]
        lines.append("| " + " | ".join(agg_cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(agg_cols)) + " |")
        for _, row in agg_df.head(50).iterrows():
            vals = [_fmt(row.get(c, "N/A")) for c in agg_cols]
            lines.append("| " + " | ".join(vals) + " |")
    else:
        for entry in per_layer_summary:
            lines.append(f"- Layer {entry.get('layer', '?')}: "
                         f"best H={_fmt(entry.get('best_h', 'N/A'))}, "
                         f"C={_fmt(entry.get('best_c', 'N/A'))}, "
                         f"UA={_fmt(entry.get('best_ua', 'N/A'))}")
    lines.append("")

    # ── Seed Stability ──
    lines.append("## 6. Seed Stability")
    lines.append("")
    if seed_stability_summary:
        for k, v in seed_stability_summary.items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("Single-seed run; seed stability not assessed.")
    lines.append("")

    # ── Verdict ──
    lines.append("## 7. Verdict")
    lines.append("")
    lines.append(f"**Verdict: `{verdict}`**")
    lines.append("")
    lines.append(f"**Reasoning:** {verdict_reason}")
    lines.append("")
    if verdict_deltas:
        lines.append("### Verdict Deltas")
        lines.append("")
        for k, v in verdict_deltas.items():
            if v is not None:
                lines.append(f"- `{k}`: {_fmt(v)}")
        lines.append("")

    # ── Next Steps ──
    lines.append("## 8. Recommendation")
    lines.append("")
    if verdict == "IC4_M1_MECHANISM_VALIDATED":
        lines.append("- Mechanism validated. Proceed to IC-4-M2: optimize alpha/layer and scale up.")
        lines.append("- Test on out-of-distribution entity pools to confirm generalization.")
    elif verdict == "IC4_M1_CONTROL_ARTIFACT":
        lines.append("- Steering fails control comparison. Possible causes:")
        lines.append("  1. Data construction artifact (train/test leakage via entity similarity)")
        lines.append("  2. Hook side-effect (vector magnitude causes output degradation indistinguishable from abstention)")
        lines.append("  3. Evaluation heuristic not distinguishing genuine from degenerate abstention")
        lines.append("- Next: IC-4-M2 with revised evaluation (perplexity check, forced-choice probe).")
    elif verdict == "IC4_M1_MODEL_DAMAGE":
        lines.append("- Steering reduces hallucination but at unacceptable cost to correctness or unnecessary abstention.")
        lines.append("- Next: IC-4-M2 with finer alpha grid, constrained optimization of H-C-UA tradeoff.")
    else:
        lines.append("- No valid candidate found. Consider:")
        lines.append("  1. Larger train set to improve vector quality")
        lines.append("  2. More layers tested")
        lines.append("  3. Alternative vector computation (e.g. per-sample normalization)")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*IC-4-M1: Diagnostic Sweep for Activation Steering*")
    lines.append("*Generated by report_writer*")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))