"""
IC-4-T2: Decision Heatmap (Experiment C).

Trains lightweight probes at each (layer, step) pair to predict final behavior,
creating a heatmap of where behavior signals are readable.

Two heatmaps:
  1. Hallucination: 3-class (hallucination / abstention / correct)
  2. Sycophancy: binary (sycophantic / non_sycophantic)

Key questions:
  - When does behavior signal appear?
  - Which layer first becomes predictive?
  - Is it a local spike or cross-layer band?
  - Do hallucination and sycophancy form differently?

Usage:
    python -m src.run_t2_decision_heatmap
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = "results_t2_decision_heatmap"
REPORTS_DIR = "reports"
TRAJECTORY_DIR = "results_t0_trajectory"
TRAJECTORY_SYC_DIR = "results_t0_syc_contrast"

HALLUCINATION_3CLASS_MAP = {
    "hallucination": "hallucination",
    "abstention": "abstention",
    "other_unanswerable": "abstention",
    "correct": "correct",
    "incorrect_answerable": "correct",
    "unnecessary_abstention": "correct",
}

PROBE_MAX_ITER = 1000
PROBE_RANDOM_STATE = 42
CV_FOLDS = 5
MIN_SAMPLES_PER_CLASS = 3


def _log(msg, log_file=None):
    print(msg, flush=True)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()


def _load_trajectory_data(task_type, log, data_dir=None):
    if data_dir is None:
        data_dir = TRAJECTORY_DIR
    npz_name = f"trajectory_states_{task_type}.npz"
    csv_name = f"sample_info_{task_type}.csv"

    npz_path = os.path.join(data_dir, npz_name)
    csv_path = os.path.join(data_dir, csv_name)

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ not found: {npz_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    _log(f"  Loading {npz_path}", log)
    data = np.load(npz_path, allow_pickle=True)
    states_last = data["states_last"]
    states_window4 = data["states_window4"]
    valid_mask = data["valid_mask"]
    target_layers = data["target_layers"]

    _log(f"  Loading {csv_path}", log)
    sample_info = pd.read_csv(csv_path)

    _log(f"  states_last shape: {states_last.shape}", log)
    _log(f"  valid_mask shape:  {valid_mask.shape}", log)
    _log(f"  target_layers:     {target_layers}", log)
    _log(f"  sample_info rows:  {len(sample_info)}", log)

    return states_last, states_window4, valid_mask, target_layers, sample_info


def _build_hallucination_labels(sample_info):
    labels = []
    for _, row in sample_info.iterrows():
        raw = row["final_behavior"]
        mapped = HALLUCINATION_3CLASS_MAP.get(raw, "correct")
        labels.append(mapped)
    return np.array(labels)


def _build_sycophancy_labels(sample_info):
    labels = []
    for _, row in sample_info.iterrows():
        raw = row["final_behavior"]
        if raw == "sycophantic":
            labels.append("sycophantic")
        else:
            labels.append("non_sycophantic")
    return np.array(labels)


def _train_probe_at_position(X, y, task_name, layer, step, rep_type, results_list):
    classes = np.unique(y)
    if len(classes) < 2:
        results_list.append({
            "task": task_name,
            "layer": layer,
            "step": step,
            "rep_type": rep_type,
            "accuracy": 0.0,
            "auc": np.nan,
            "n_samples": len(y),
            "n_classes": len(classes),
            "note": "single_class",
        })
        return

    for cls in classes:
        if np.sum(y == cls) < MIN_SAMPLES_PER_CLASS:
            results_list.append({
                "task": task_name,
                "layer": layer,
                "step": step,
                "rep_type": rep_type,
                "accuracy": 0.0,
                "auc": np.nan,
                "n_samples": len(y),
                "n_classes": len(classes),
                "note": f"too_few_{cls}",
            })
            return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    is_binary = len(classes) == 2

    n_folds = min(CV_FOLDS, min(np.sum(y == c) for c in classes) // 1)
    n_folds = max(2, min(n_folds, CV_FOLDS))

    clf = LogisticRegression(
        max_iter=PROBE_MAX_ITER,
        random_state=PROBE_RANDOM_STATE,
    )

    try:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=PROBE_RANDOM_STATE)
        acc_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy")
        acc_mean = float(acc_scores.mean())
    except ValueError:
        clf.fit(X_scaled, y)
        acc_mean = float(clf.score(X_scaled, y))
        n_folds = 0

    auc_val = np.nan
    if is_binary:
        try:
            y_bin = (y == classes[0]).astype(int)
            clf_auc = LogisticRegression(
                max_iter=PROBE_MAX_ITER,
                random_state=PROBE_RANDOM_STATE,
            )
            clf_auc.fit(X_scaled, y_bin)
            if X_scaled.shape[0] >= 10:
                y_prob = clf_auc.predict_proba(X_scaled)[:, 1]
                auc_val = float(roc_auc_score(y_bin, y_prob))
            else:
                auc_val = np.nan
        except ValueError:
            auc_val = np.nan

    results_list.append({
        "task": task_name,
        "layer": layer,
        "step": step,
        "rep_type": rep_type,
        "accuracy": round(acc_mean, 4),
        "auc": round(auc_val, 4) if not np.isnan(auc_val) else np.nan,
        "n_samples": len(y),
        "n_classes": len(classes),
        "note": f"cv={n_folds}" if n_folds > 0 else "train_only",
    })


def _run_heatmap_scan(states, valid_mask, target_layers, labels, task_name, rep_type, log):
    n_samples, n_layer_idx, n_steps, hidden_dim = states.shape
    results = []

    label_arr = labels[:n_samples]

    _log(f"\n  Scanning {task_name} / {rep_type}: {n_layer_idx} layers x {n_steps} steps", log)

    for li, layer in enumerate(target_layers):
        layer = int(layer)
        for step in range(n_steps):
            sample_indices = []
            for s in range(n_samples):
                if valid_mask[s, li, step]:
                    sample_indices.append(s)

            if len(sample_indices) < 8:
                results.append({
                    "task": task_name,
                    "layer": layer,
                    "step": step,
                    "rep_type": rep_type,
                    "accuracy": 0.0,
                    "auc": np.nan,
                    "n_samples": len(sample_indices),
                    "n_classes": 0,
                    "note": "too_few_samples",
                })
                continue

            X = states[sample_indices, li, step, :]
            y = label_arr[sample_indices]

            _train_probe_at_position(X, y, task_name, layer, step, rep_type, results)

        _log(f"    Layer {layer} done", log)

    return results


def _find_first_predictive_layer(heatmap_df, task_name, acc_threshold=0.65):
    task_df = heatmap_df[heatmap_df["task"] == task_name]
    task_df = task_df[task_df["note"].isin(["cv=2", "cv=3", "cv=4", "cv=5"])]
    if len(task_df) == 0:
        return None, None

    above = task_df[task_df["accuracy"] >= acc_threshold]
    if len(above) == 0:
        return None, None

    step_groups = above.groupby("step")
    first_step = None
    first_layer = None
    for step in sorted(step_groups.groups.keys()):
        group = step_groups.get_group(step)
        best_row = group.loc[group["accuracy"].idxmax()]
        first_step = int(step)
        first_layer = int(best_row["layer"])
        break

    return first_step, first_layer


def _analyze_heatmap_structure(heatmap_df, task_name):
    task_df = heatmap_df[heatmap_df["task"] == task_name]
    task_df = task_df[task_df["note"].isin(["cv=2", "cv=3", "cv=4", "cv=5"])]

    if len(task_df) == 0:
        return {"structure": "no_valid_probes"}

    above_65 = task_df[task_df["accuracy"] >= 0.65]
    if len(above_65) == 0:
        return {"structure": "no_predictive_positions"}

    layers_with_signal = sorted(above_65["layer"].unique())
    steps_with_signal = sorted(above_65["step"].unique())

    if len(layers_with_signal) <= 2:
        structure = "local_spike"
    else:
        layer_gaps = [layers_with_signal[i+1] - layers_with_signal[i]
                      for i in range(len(layers_with_signal) - 1)]
        max_gap = max(layer_gaps) if layer_gaps else 0
        if max_gap <= 4:
            structure = "cross_layer_band"
        else:
            structure = "scattered_islands"

    best_row = task_df.loc[task_df["accuracy"].idxmax()]
    peak_layer = int(best_row["layer"])
    peak_step = int(best_row["step"])
    peak_acc = float(best_row["accuracy"])

    return {
        "structure": structure,
        "layers_with_signal": layers_with_signal,
        "steps_with_signal": steps_with_signal,
        "peak_layer": peak_layer,
        "peak_step": peak_step,
        "peak_accuracy": peak_acc,
    }


def _generate_report(report_path, heatmap_df, hall_analysis, syc_analysis,
                     hall_first, syc_first, target_layers, n_hall, n_syc, elapsed, log):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    lines = []
    lines.append("# IC-4-T2: Decision Heatmap Report")
    lines.append("")
    lines.append("> Trains lightweight probes at each (layer, step) pair to predict final")
    lines.append("> behavior, creating a heatmap of where behavior signals are readable.")
    lines.append("")

    lines.append("## 1. Setup")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append(f"| Target layers | {list(target_layers)} |")
    lines.append(f"| Probe | LogisticRegression (sklearn) |")
    lines.append(f"| Scaler | StandardScaler |")
    lines.append(f"| CV folds | {CV_FOLDS} |")
    lines.append(f"| Hallucination samples | {n_hall} |")
    lines.append(f"| Sycophancy samples | {n_syc} |")
    lines.append(f"| Elapsed | {elapsed:.0f}s ({elapsed/60:.1f} min) |")
    lines.append("")

    lines.append("## 2. Hallucination Heatmap (3-class: hallucination / abstention / correct)")
    lines.append("")

    hall_df = heatmap_df[heatmap_df["task"] == "hallucination_3class"]
    if len(hall_df) > 0:
        for rep in ["state_last", "state_window4"]:
            rep_df = hall_df[hall_df["rep_type"] == rep]
            if len(rep_df) == 0:
                continue
            lines.append(f"### {rep}")
            lines.append("")
            lines.append("| Layer | Step | Accuracy | AUC | n_samples | Note |")
            lines.append("|---|---|---|---|---|---|")
            for _, row in rep_df.sort_values(["layer", "step"]).iterrows():
                auc_str = f"{row['auc']:.4f}" if not pd.isna(row["auc"]) else "N/A"
                lines.append(f"| {int(row['layer'])} | {int(row['step'])} | {row['accuracy']:.4f} | {auc_str} | {int(row['n_samples'])} | {row['note']} |")
            lines.append("")

    if hall_analysis and hall_analysis.get("structure") != "no_valid_probes":
        lines.append("### Hallucination Structure Analysis")
        lines.append("")
        lines.append(f"- **Structure**: {hall_analysis['structure']}")
        lines.append(f"- **Peak**: layer={hall_analysis['peak_layer']}, step={hall_analysis['peak_step']}, acc={hall_analysis['peak_accuracy']:.4f}")
        lines.append(f"- **Layers with signal (acc>=0.65)**: {hall_analysis['layers_with_signal']}")
        lines.append(f"- **Steps with signal**: {hall_analysis['steps_with_signal']}")
        if hall_first[0] is not None:
            lines.append(f"- **First predictive position**: step={hall_first[0]}, layer={hall_first[1]}")
        lines.append("")
    else:
        lines.append("### Hallucination Structure Analysis")
        lines.append("")
        lines.append("No valid probe results for hallucination task.")
        lines.append("")

    lines.append("## 3. Sycophancy Heatmap (binary: sycophantic / non_sycophantic)")
    lines.append("")

    syc_df = heatmap_df[heatmap_df["task"] == "sycophancy_binary"]
    if len(syc_df) > 0:
        for rep in ["state_last", "state_window4"]:
            rep_df = syc_df[syc_df["rep_type"] == rep]
            if len(rep_df) == 0:
                continue
            lines.append(f"### {rep}")
            lines.append("")
            lines.append("| Layer | Step | Accuracy | AUC | n_samples | Note |")
            lines.append("|---|---|---|---|---|---|")
            for _, row in rep_df.sort_values(["layer", "step"]).iterrows():
                auc_str = f"{row['auc']:.4f}" if not pd.isna(row["auc"]) else "N/A"
                lines.append(f"| {int(row['layer'])} | {int(row['step'])} | {row['accuracy']:.4f} | {auc_str} | {int(row['n_samples'])} | {row['note']} |")
            lines.append("")

    if syc_analysis and syc_analysis.get("structure") != "no_valid_probes":
        lines.append("### Sycophancy Structure Analysis")
        lines.append("")
        lines.append(f"- **Structure**: {syc_analysis['structure']}")
        lines.append(f"- **Peak**: layer={syc_analysis['peak_layer']}, step={syc_analysis['peak_step']}, acc={syc_analysis['peak_accuracy']:.4f}")
        lines.append(f"- **Layers with signal (acc>=0.65)**: {syc_analysis['layers_with_signal']}")
        lines.append(f"- **Steps with signal**: {syc_analysis['steps_with_signal']}")
        if syc_first[0] is not None:
            lines.append(f"- **First predictive position**: step={syc_first[0]}, layer={syc_first[1]}")
        lines.append("")
    else:
        lines.append("### Sycophancy Structure Analysis")
        lines.append("")
        lines.append("No valid probe results for sycophancy task.")
        lines.append("")

    lines.append("## 4. Key Questions")
    lines.append("")

    lines.append("### Q1: When does behavior signal appear?")
    lines.append("")
    if hall_first[0] is not None:
        lines.append(f"- **Hallucination**: First predictive at step {hall_first[0]}, layer {hall_first[1]}")
    else:
        lines.append("- **Hallucination**: No clearly predictive position found (threshold 0.65)")
    if syc_first[0] is not None:
        lines.append(f"- **Sycophancy**: First predictive at step {syc_first[0]}, layer {syc_first[1]}")
    else:
        lines.append("- **Sycophancy**: No clearly predictive position found (threshold 0.65)")
    lines.append("")

    lines.append("### Q2: Which layer first becomes predictive?")
    lines.append("")
    if hall_first[1] is not None:
        lines.append(f"- **Hallucination**: Layer {hall_first[1]}")
    else:
        lines.append("- **Hallucination**: N/A")
    if syc_first[1] is not None:
        lines.append(f"- **Sycophancy**: Layer {syc_first[1]}")
    else:
        lines.append("- **Sycophancy**: N/A")
    lines.append("")

    lines.append("### Q3: Is it a local spike or cross-layer band?")
    lines.append("")
    if hall_analysis and hall_analysis.get("structure") not in (None, "no_valid_probes"):
        lines.append(f"- **Hallucination**: {hall_analysis['structure']}")
    else:
        lines.append("- **Hallucination**: Insufficient data")
    if syc_analysis and syc_analysis.get("structure") not in (None, "no_valid_probes"):
        lines.append(f"- **Sycophancy**: {syc_analysis['structure']}")
    else:
        lines.append("- **Sycophancy**: Insufficient data")
    lines.append("")

    lines.append("### Q4: Do hallucination and sycophancy form differently?")
    lines.append("")
    if (hall_analysis and hall_analysis.get("structure") not in (None, "no_valid_probes")
            and syc_analysis and syc_analysis.get("structure") not in (None, "no_valid_probes")):
        hall_struct = hall_analysis["structure"]
        syc_struct = syc_analysis["structure"]
        if hall_struct != syc_struct:
            lines.append(f"- **Yes**: Hallucination forms as `{hall_struct}`, sycophancy forms as `{syc_struct}`.")
            lines.append("  The two behaviors occupy different representational geometries.")
        else:
            lines.append(f"- **Same structure**: Both form as `{hall_struct}`.")
            if hall_analysis.get("peak_layer") != syc_analysis.get("peak_layer"):
                lines.append(f"  But peak layers differ: hallucination at L{hall_analysis['peak_layer']}, "
                             f"sycophancy at L{syc_analysis['peak_layer']}.")
            else:
                lines.append(f"  Peak layers also coincide at L{hall_analysis['peak_layer']}.")
    else:
        lines.append("- Insufficient data to compare formation patterns.")
    lines.append("")

    lines.append("## 5. Hallucination vs Sycophancy Formation Contrast")
    lines.append("")
    lines.append("| Dimension | Hallucination | Sycophancy |")
    lines.append("|---|---|---|")
    if hall_analysis and hall_analysis.get("structure") not in (None, "no_valid_probes"):
        lines.append(f"| Heatmap structure | {hall_analysis['structure']} | "
                     f"{syc_analysis.get('structure', 'N/A') if syc_analysis else 'N/A'} |")
        lines.append(f"| Peak (layer, step) | (L{hall_analysis.get('peak_layer', '?')}, "
                     f"S{hall_analysis.get('peak_step', '?')}) | "
                     f"(L{syc_analysis.get('peak_layer', '?') if syc_analysis else '?'}, "
                     f"S{syc_analysis.get('peak_step', '?') if syc_analysis else '?'}) |")
        lines.append(f"| Peak accuracy | {hall_analysis.get('peak_accuracy', 0):.4f} | "
             + (f"{syc_analysis.get('peak_accuracy', 0):.4f}" if syc_analysis else "N/A") + " |")
        lines.append(f"| First predictive step | {hall_first[0]} | {syc_first[0]} |")
        lines.append(f"| First predictive layer | {hall_first[1]} | {syc_first[1]} |")
        lines.append(f"| Layers with signal | {hall_analysis.get('layers_with_signal', [])} | "
                     f"{syc_analysis.get('layers_with_signal', []) if syc_analysis else []} |")
    else:
        lines.append("| | Insufficient data | Insufficient data |")
    lines.append("")

    lines.append("## 6. Representation / Readout Caveat")
    lines.append("")
    lines.append("> This experiment uses `state_last` (last-token hidden state) and")
    lines.append("> `state_window4` (mean of last 4 positions during prefill) as probe inputs.")
    lines.append("> These readout choices constrain what the probes can detect:")
    lines.append(">")
    lines.append("> 1. **Position bias**: If a behavior signal is concentrated in non-last")
    lines.append(">    positions (e.g., question tokens), `state_last` may miss it entirely.")
    lines.append(">    `state_window4` partially addresses this during prefill but is identical")
    lines.append(">    to `state_last` during decode steps (only 1 position available).")
    lines.append(">")
    lines.append("> 2. **Probe capacity**: Logistic regression is a linear probe. If the")
    lines.append(">    behavior signal is nonlinearly encoded, the probe will underreport")
    lines.append(">    readability. High accuracy implies linear separability; low accuracy")
    lines.append(">    does NOT imply absence of signal — only absence of *linearly readable*")
    lines.append(">    signal.")
    lines.append(">")
    lines.append("> 3. **Causal vs correlational**: A position being readable does not mean")
    lines.append(">    it is causally involved in generating the behavior. It may reflect")
    lines.append(">    downstream consequences rather than upstream decisions. Steering")
    lines.append(">    experiments (T3) are needed to establish causality.")
    lines.append(">")
    lines.append("> 4. **Sample size**: With small sample sizes, probe accuracy estimates")
    lines.append(">    have high variance. Cross-validation helps but does not eliminate")
    lines.append(">    this concern. Results should be treated as suggestive, not definitive.")
    lines.append("")

    lines.append("## 7. Output Files")
    lines.append("")
    lines.append(f"- `{RESULTS_DIR}/heatmap_hallucination_3class.csv`")
    lines.append(f"- `{RESULTS_DIR}/heatmap_sycophancy_binary.csv`")
    lines.append(f"- `{RESULTS_DIR}/heatmap_all.csv`")
    lines.append(f"- `{RESULTS_DIR}/run_log.txt`")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*IC-4-T2: Decision Heatmap*")
    lines.append("*Generated by run_t2_decision_heatmap.py*")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    log_path = os.path.join(RESULTS_DIR, "run_log.txt")
    _log("=" * 60, log_path)
    _log("IC-4-T2: Decision Heatmap", log_path)
    _log("=" * 60, log_path)

    t_start = time.time()

    _log("\n--- Loading hallucination trajectory data ---", log_path)
    try:
        hall_states_last, hall_states_window4, hall_valid_mask, hall_target_layers, hall_sample_info = \
            _load_trajectory_data("hallucination", log_path)
    except FileNotFoundError as e:
        _log(f"  ERROR: {e}", log_path)
        _log("  Run run_t0_trajectory_capture.py first.", log_path)
        return

    n_hall = len(hall_sample_info)
    _log(f"  Hallucination samples: {n_hall}", log_path)

    _log(f"\n--- Loading sycophancy trajectory data (balanced contrast set) ---", log_path)
    try:
        syc_states_last, syc_states_window4, syc_valid_mask, syc_target_layers, syc_sample_info = \
            _load_trajectory_data("sycophancy", log_path, data_dir=TRAJECTORY_SYC_DIR)
    except FileNotFoundError as e:
        _log(f"  ERROR: {e}", log_path)
        _log("  Run run_t0_sycophancy_contrast.py first.", log_path)
        return

    n_syc = len(syc_sample_info)
    _log(f"  Sycophancy samples: {n_syc}", log_path)

    target_layers = hall_target_layers

    _log("\n--- Building labels ---", log_path)
    hall_labels = _build_hallucination_labels(hall_sample_info)
    hall_class_counts = pd.Series(hall_labels).value_counts()
    _log(f"  Hallucination 3-class: {dict(hall_class_counts)}", log_path)

    syc_labels = _build_sycophancy_labels(syc_sample_info)
    syc_class_counts = pd.Series(syc_labels).value_counts()
    _log(f"  Sycophancy binary: {dict(syc_class_counts)}", log_path)

    all_results = []

    _log("\n" + "=" * 60, log_path)
    _log("HALLUCINATION HEATMAP (state_last)", log_path)
    _log("=" * 60, log_path)
    hall_last_results = _run_heatmap_scan(
        hall_states_last, hall_valid_mask, target_layers,
        hall_labels, "hallucination_3class", "state_last", log_path)
    all_results.extend(hall_last_results)

    _log("\n" + "=" * 60, log_path)
    _log("HALLUCINATION HEATMAP (state_window4)", log_path)
    _log("=" * 60, log_path)
    hall_win4_results = _run_heatmap_scan(
        hall_states_window4, hall_valid_mask, target_layers,
        hall_labels, "hallucination_3class", "state_window4", log_path)
    all_results.extend(hall_win4_results)

    _log("\n" + "=" * 60, log_path)
    _log("SYCOPHANCY HEATMAP (state_last)", log_path)
    _log("=" * 60, log_path)
    syc_last_results = _run_heatmap_scan(
        syc_states_last, syc_valid_mask, target_layers,
        syc_labels, "sycophancy_binary", "state_last", log_path)
    all_results.extend(syc_last_results)

    _log("\n" + "=" * 60, log_path)
    _log("SYCOPHANCY HEATMAP (state_window4)", log_path)
    _log("=" * 60, log_path)
    syc_win4_results = _run_heatmap_scan(
        syc_states_window4, syc_valid_mask, target_layers,
        syc_labels, "sycophancy_binary", "state_window4", log_path)
    all_results.extend(syc_win4_results)

    heatmap_df = pd.DataFrame(all_results)

    hall_heatmap_df = heatmap_df[heatmap_df["task"] == "hallucination_3class"]
    syc_heatmap_df = heatmap_df[heatmap_df["task"] == "sycophancy_binary"]

    hall_csv_path = os.path.join(RESULTS_DIR, "heatmap_hallucination_3class.csv")
    hall_heatmap_df.to_csv(hall_csv_path, index=False)
    _log(f"\nHallucination heatmap saved to {hall_csv_path}", log_path)

    syc_csv_path = os.path.join(RESULTS_DIR, "heatmap_sycophancy_binary.csv")
    syc_heatmap_df.to_csv(syc_csv_path, index=False)
    _log(f"Sycophancy heatmap saved to {syc_csv_path}", log_path)

    all_csv_path = os.path.join(RESULTS_DIR, "heatmap_all.csv")
    heatmap_df.to_csv(all_csv_path, index=False)
    _log(f"Combined heatmap saved to {all_csv_path}", log_path)

    _log("\n--- Analyzing heatmap structure ---", log_path)
    hall_analysis = _analyze_heatmap_structure(heatmap_df, "hallucination_3class")
    syc_analysis = _analyze_heatmap_structure(heatmap_df, "sycophancy_binary")

    _log(f"  Hallucination structure: {hall_analysis.get('structure', 'N/A')}", log_path)
    if hall_analysis.get("peak_layer") is not None:
        _log(f"    Peak: L{hall_analysis['peak_layer']} step={hall_analysis['peak_step']} acc={hall_analysis['peak_accuracy']:.4f}", log_path)

    _log(f"  Sycophancy structure: {syc_analysis.get('structure', 'N/A')}", log_path)
    if syc_analysis.get("peak_layer") is not None:
        _log(f"    Peak: L{syc_analysis['peak_layer']} step={syc_analysis['peak_step']} acc={syc_analysis['peak_accuracy']:.4f}", log_path)

    hall_first = _find_first_predictive_layer(heatmap_df, "hallucination_3class")
    syc_first = _find_first_predictive_layer(heatmap_df, "sycophancy_binary")

    _log(f"  Hallucination first predictive: step={hall_first[0]} layer={hall_first[1]}", log_path)
    _log(f"  Sycophancy first predictive: step={syc_first[0]} layer={syc_first[1]}", log_path)

    elapsed = time.time() - t_start
    _log(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)", log_path)

    _log("\n--- Summary ---", log_path)
    _log(f"{'Task':<28} {'Rep':<16} {'Best Acc':>10} {'Peak L':>8} {'Peak S':>8}", log_path)
    _log(f"{'-'*28} {'-'*16} {'-'*10} {'-'*8} {'-'*8}", log_path)
    for task in ["hallucination_3class", "sycophancy_binary"]:
        for rep in ["state_last", "state_window4"]:
            sub = heatmap_df[(heatmap_df["task"] == task) & (heatmap_df["rep_type"] == rep)]
            valid = sub[sub["note"].str.contains("cv=", na=False)]
            if len(valid) > 0:
                best = valid.loc[valid["accuracy"].idxmax()]
                _log(f"{task:<28} {rep:<16} {best['accuracy']:>10.4f} {int(best['layer']):>8} {int(best['step']):>8}", log_path)
            else:
                _log(f"{task:<28} {rep:<16} {'N/A':>10} {'N/A':>8} {'N/A':>8}", log_path)

    report_path = os.path.join(REPORTS_DIR, "IC4_T2_DECISION_HEATMAP_REPORT.md")
    _generate_report(report_path, heatmap_df, hall_analysis, syc_analysis,
                     hall_first, syc_first, target_layers, n_hall, n_syc, elapsed, log_path)
    _log(f"\nReport saved to {report_path}", log_path)
    _log(f"\nIC-4-T2 complete.", log_path)


if __name__ == "__main__":
    main()
