"""
IC-4-T1: Projection Analysis.

Projects hidden-state trajectories onto behavior-relevant directions
to characterize when and how behavioral divergence becomes visible
in the model's internal representations.

Directions:
  - v_hall: hallucination vs abstention (prefill step)
  - v_syc:  sycophantic vs non-sycophantic (prefill step)
  - random: random normalized control
  - shuffled: label-shuffled control

Key questions:
  1. When do hallucination vs abstention trajectories separate?
  2. Does sycophancy vs non-sycophancy collapse during generation?
  3. Do random/shuffled directions show no structure?

Usage:
    python -m src.run_t1_projection_analysis
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.steering import compute_steering_vector, compute_random_vector, compute_shuffled_vector

T0_RESULTS_DIR = "results_t0_trajectory"
T0_SYC_CONTRAST_DIR = "results_t0_syc_contrast"
RESULTS_DIR = "results_t1_projection"
REPORTS_DIR = "reports"
MAIN_LAYER = 12
PREFILL_STEP = 0
N_RANDOM_SEEDS = 5
SEPARATION_THRESHOLD = 0.5
LATE_STAGE_FRACTION = 0.25


def _log(msg, log_file=None):
    print(msg, flush=True)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()


def _load_trajectory_data(task_type, data_dir=None):
    if data_dir is None:
        data_dir = T0_RESULTS_DIR
    npz_path = os.path.join(data_dir, f"trajectory_states_{task_type}.npz")
    csv_path = os.path.join(data_dir, f"sample_info_{task_type}.csv")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing NPZ: {npz_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    data = np.load(npz_path, allow_pickle=True)
    sample_info = pd.read_csv(csv_path)

    return {
        "states_last": data["states_last"],
        "states_window4": data["states_window4"],
        "valid_mask": data["valid_mask"],
        "target_layers": data["target_layers"],
        "hidden_dim": int(data["hidden_dim"][0]) if data["hidden_dim"].ndim > 0 else int(data["hidden_dim"]),
        "seq_lens": data["seq_lens"] if "seq_lens" in data else None,
        "entropies": data["entropies"] if "entropies" in data else None,
        "max_probs": data["max_probs"] if "max_probs" in data else None,
        "sample_info": sample_info,
    }


def _compute_steering_vectors(hall_data, syc_data, target_layers):
    layer_to_idx = {int(l): i for i, l in enumerate(target_layers)}
    hidden_dim = hall_data["hidden_dim"]

    hall_info = hall_data["sample_info"]
    hall_states = hall_data["states_last"]
    hall_valid = hall_data["valid_mask"]

    syc_info = syc_data["sample_info"]
    syc_states = syc_data["states_last"]
    syc_valid = syc_data["valid_mask"]

    vectors = {}

    for layer in target_layers:
        lidx = layer_to_idx[int(layer)]

        hall_pos_indices = []
        hall_neg_indices = []
        for _, row in hall_info.iterrows():
            sid = int(row["sample_id"])
            behavior = row["final_behavior"]
            if behavior == "hallucination":
                hall_pos_indices.append(sid)
            elif behavior in ("abstention", "unnecessary_abstention"):
                hall_neg_indices.append(sid)

        pos_acts_hall = []
        neg_acts_hall = []
        for sid in hall_pos_indices:
            if hall_valid[sid, lidx, PREFILL_STEP]:
                pos_acts_hall.append(hall_states[sid, lidx, PREFILL_STEP, :])
        for sid in hall_neg_indices:
            if hall_valid[sid, lidx, PREFILL_STEP]:
                neg_acts_hall.append(hall_states[sid, lidx, PREFILL_STEP, :])

        if pos_acts_hall and neg_acts_hall:
            pos_arr = np.stack(pos_acts_hall)
            neg_arr = np.stack(neg_acts_hall)
            v_hall = compute_steering_vector(pos_arr, neg_arr)
        else:
            _log(f"  WARNING: insufficient hallucination data for layer {layer} (pos={len(pos_acts_hall)}, neg={len(neg_acts_hall)})")
            v_hall = np.zeros(hidden_dim, dtype=np.float32)

        syc_pos_indices = []
        syc_neg_indices = []
        for _, row in syc_info.iterrows():
            sid = int(row["sample_id"])
            behavior = row["final_behavior"]
            if behavior == "sycophantic":
                syc_pos_indices.append(sid)
            elif behavior == "non_sycophantic":
                syc_neg_indices.append(sid)

        pos_acts_syc = []
        neg_acts_syc = []
        for sid in syc_pos_indices:
            if syc_valid[sid, lidx, PREFILL_STEP]:
                pos_acts_syc.append(syc_states[sid, lidx, PREFILL_STEP, :])
        for sid in syc_neg_indices:
            if syc_valid[sid, lidx, PREFILL_STEP]:
                neg_acts_syc.append(syc_states[sid, lidx, PREFILL_STEP, :])

        if pos_acts_syc and neg_acts_syc:
            pos_arr_s = np.stack(pos_acts_syc)
            neg_arr_s = np.stack(neg_acts_syc)
            v_syc = compute_steering_vector(pos_arr_s, neg_arr_s)
        else:
            _log(f"  WARNING: insufficient sycophancy data for layer {layer} (pos={len(pos_acts_syc)}, neg={len(neg_acts_syc)})")
            v_syc = np.zeros(hidden_dim, dtype=np.float32)

        v_random = compute_random_vector(hidden_dim, seed=42)

        if pos_acts_hall and neg_acts_hall:
            pos_arr_full = np.stack(pos_acts_hall)
            neg_arr_full = np.stack(neg_acts_hall)
            v_shuffled = compute_shuffled_vector(pos_arr_full, neg_arr_full, seed=123)
        else:
            v_shuffled = compute_random_vector(hidden_dim, seed=999)

        vectors[int(layer)] = {
            "v_hall": v_hall,
            "v_syc": v_syc,
            "random": v_random,
            "shuffled": v_shuffled,
        }

    return vectors


def _project_trajectories(data, vectors, target_layers):
    layer_to_idx = {int(l): i for i, l in enumerate(target_layers)}
    states = data["states_last"]
    valid = data["valid_mask"]
    sample_info = data["sample_info"]
    n_samples, n_layer_indices, n_steps, hidden_dim = states.shape

    direction_names = ["v_hall", "v_syc", "random", "shuffled"]

    projections = {}
    for layer in target_layers:
        lidx = layer_to_idx[int(layer)]
        vecs = vectors[int(layer)]

        for dname in direction_names:
            vec = vecs[dname]
            proj_key = (int(layer), dname)

            proj_arr = np.full((n_samples, n_steps), np.nan, dtype=np.float32)

            for _, row in sample_info.iterrows():
                sid = int(row["sample_id"])
                for step in range(n_steps):
                    if valid[sid, lidx, step]:
                        proj_arr[sid, step] = np.dot(states[sid, lidx, step, :], vec)

            projections[proj_key] = proj_arr

    return projections


def _compute_average_curves(hall_projections, syc_projections, sample_info, target_layers):
    direction_names = ["v_hall", "v_syc", "random", "shuffled"]

    hall_info = sample_info.get("hallucination", sample_info)
    syc_info = sample_info.get("sycophancy", sample_info)

    avg_curves = {}

    for layer in target_layers:
        for dname in direction_names:
            key = (int(layer), dname)

            hall_curves = {}
            if dname != "v_syc":
                for behavior in ["hallucination", "abstention", "unnecessary_abstention",
                                 "correct", "incorrect_answerable", "other_unanswerable"]:
                    sids = hall_info[hall_info["final_behavior"] == behavior]["sample_id"].values.astype(int)
                    if len(sids) > 0:
                        proj = hall_projections[key]
                        curves = proj[sids, :]
                        mask = ~np.isnan(curves)
                        mean_curve = np.full(curves.shape[1], np.nan, dtype=np.float32)
                        for step in range(curves.shape[1]):
                            valid_vals = curves[:, step][mask[:, step]]
                            if len(valid_vals) > 0:
                                mean_curve[step] = valid_vals.mean()
                        hall_curves[behavior] = mean_curve

            syc_curves = {}
            if dname != "v_hall":
                for behavior in ["sycophantic", "non_sycophantic"]:
                    sids = syc_info[syc_info["final_behavior"] == behavior]["sample_id"].values.astype(int)
                    if len(sids) > 0:
                        proj = syc_projections[key]
                        curves = proj[sids, :]
                        mask = ~np.isnan(curves)
                        mean_curve = np.full(curves.shape[1], np.nan, dtype=np.float32)
                        for step in range(curves.shape[1]):
                            valid_vals = curves[:, step][mask[:, step]]
                            if len(valid_vals) > 0:
                                mean_curve[step] = valid_vals.mean()
                        syc_curves[behavior] = mean_curve

            avg_curves[key] = {
                "hallucination_task": hall_curves,
                "sycophancy_task": syc_curves,
            }

    return avg_curves


def _compute_separation_statistics(avg_curves, hall_projections, syc_projections, sample_info, target_layers):
    direction_names = ["v_hall", "v_syc", "random", "shuffled"]
    hall_info = sample_info.get("hallucination", sample_info)
    syc_info = sample_info.get("sycophancy", sample_info)

    separation_stats = {}

    for layer in target_layers:
        for dname in direction_names:
            key = (int(layer), dname)
            curves = avg_curves[key]

            hall_curves = curves["hallucination_task"]
            syc_curves = curves["sycophancy_task"]

            stat = {}

            if "hallucination" in hall_curves and "abstention" in hall_curves:
                hall_curve = hall_curves["hallucination"]
                abst_curve = hall_curves["abstention"]
                diff = hall_curve - abst_curve
                valid_diff = diff[~np.isnan(diff)]

                earliest_sep = None
                for i, d in enumerate(valid_diff):
                    if abs(d) > SEPARATION_THRESHOLD:
                        earliest_sep = i
                        break

                max_sep_step = None
                max_sep_val = 0.0
                for i, d in enumerate(valid_diff):
                    if abs(d) > abs(max_sep_val):
                        max_sep_val = d
                        max_sep_step = i

                n_total = len(valid_diff)
                n_late = max(1, int(n_total * LATE_STAGE_FRACTION))
                late_vals = valid_diff[-n_late:] if n_total > 0 else np.array([])
                late_var = float(np.var(late_vals)) if len(late_vals) > 1 else 0.0

                stat["hall_vs_abst"] = {
                    "earliest_separation_step": earliest_sep,
                    "max_separation_step": max_sep_step,
                    "max_separation_value": float(max_sep_val) if max_sep_val != 0.0 else None,
                    "late_stage_variance": late_var,
                    "n_valid_steps": int(np.sum(~np.isnan(diff))),
                }

                hall_sids = hall_info[hall_info["final_behavior"] == "hallucination"]["sample_id"].values.astype(int)
                abst_sids = hall_info[hall_info["final_behavior"] == "abstention"]["sample_id"].values.astype(int)
                proj = hall_projections[key]

                per_step_t = []
                per_step_p = []
                n_steps = proj.shape[1]
                for step in range(n_steps):
                    h_vals = proj[hall_sids, step]
                    a_vals = proj[abst_sids, step]
                    h_valid = h_vals[~np.isnan(h_vals)]
                    a_valid = a_vals[~np.isnan(a_vals)]
                    if len(h_valid) >= 2 and len(a_valid) >= 2:
                        t_val, p_val = sp_stats.ttest_ind(h_valid, a_valid, equal_var=False)
                        per_step_t.append(float(t_val))
                        per_step_p.append(float(p_val))
                    else:
                        per_step_t.append(None)
                        per_step_p.append(None)
                stat["hall_vs_abst"]["per_step_t"] = per_step_t
                stat["hall_vs_abst"]["per_step_p"] = per_step_p

            if "sycophantic" in syc_curves and "non_sycophantic" in syc_curves:
                syc_curve = syc_curves["sycophantic"]
                nonsyc_curve = syc_curves["non_sycophantic"]
                diff = syc_curve - nonsyc_curve
                valid_diff = diff[~np.isnan(diff)]

                earliest_sep = None
                for i, d in enumerate(valid_diff):
                    if abs(d) > SEPARATION_THRESHOLD:
                        earliest_sep = i
                        break

                max_sep_step = None
                max_sep_val = 0.0
                for i, d in enumerate(valid_diff):
                    if abs(d) > abs(max_sep_val):
                        max_sep_val = d
                        max_sep_step = i

                n_total = len(valid_diff)
                n_late = max(1, int(n_total * LATE_STAGE_FRACTION))
                late_vals = valid_diff[-n_late:] if n_total > 0 else np.array([])
                late_var = float(np.var(late_vals)) if len(late_vals) > 1 else 0.0

                collapse_ratio = None
                if len(valid_diff) > 1 and abs(valid_diff[0]) > 1e-8:
                    late_mean = float(np.mean(np.abs(late_vals))) if len(late_vals) > 0 else 0.0
                    collapse_ratio = late_mean / abs(valid_diff[0])

                stat["syc_vs_nonsyc"] = {
                    "earliest_separation_step": earliest_sep,
                    "max_separation_step": max_sep_step,
                    "max_separation_value": float(max_sep_val) if max_sep_val != 0.0 else None,
                    "late_stage_variance": late_var,
                    "collapse_ratio": collapse_ratio,
                    "n_valid_steps": int(np.sum(~np.isnan(diff))),
                }

                syc_sids = syc_info[syc_info["final_behavior"] == "sycophantic"]["sample_id"].values.astype(int)
                nonsyc_sids = syc_info[syc_info["final_behavior"] == "non_sycophantic"]["sample_id"].values.astype(int)
                proj = syc_projections[key]

                per_step_t = []
                per_step_p = []
                n_steps = proj.shape[1]
                for step in range(n_steps):
                    s_vals = proj[syc_sids, step]
                    n_vals = proj[nonsyc_sids, step]
                    s_valid = s_vals[~np.isnan(s_vals)]
                    n_valid = n_vals[~np.isnan(n_vals)]
                    if len(s_valid) >= 2 and len(n_valid) >= 2:
                        t_val, p_val = sp_stats.ttest_ind(s_valid, n_valid, equal_var=False)
                        per_step_t.append(float(t_val))
                        per_step_p.append(float(p_val))
                    else:
                        per_step_t.append(None)
                        per_step_p.append(None)
                stat["syc_vs_nonsyc"]["per_step_t"] = per_step_t
                stat["syc_vs_nonsyc"]["per_step_p"] = per_step_p

            separation_stats[key] = stat

    return separation_stats


def _compute_random_baseline_statistics(hall_data, syc_data, target_layers, hidden_dim):
    layer_to_idx = {int(l): i for i, l in enumerate(target_layers)}
    hall_info = hall_data["sample_info"]
    syc_info = syc_data["sample_info"]
    hall_states = hall_data["states_last"]
    hall_valid = hall_data["valid_mask"]
    syc_states = syc_data["states_last"]
    syc_valid = syc_data["valid_mask"]

    random_stats = {}

    for layer in target_layers:
        lidx = layer_to_idx[int(layer)]
        layer_stats = []

        for seed in range(N_RANDOM_SEEDS):
            v_rand = compute_random_vector(hidden_dim, seed=seed * 100 + 7)

            hall_sids = hall_info[hall_info["final_behavior"].isin(["hallucination", "abstention"])]["sample_id"].values.astype(int)
            hall_behaviors = hall_info[hall_info["final_behavior"].isin(["hallucination", "abstention"])].set_index("sample_id")["final_behavior"]

            hall_proj = {}
            for sid in hall_sids:
                for step in range(hall_states.shape[2]):
                    if hall_valid[sid, lidx, step]:
                        if sid not in hall_proj:
                            hall_proj[sid] = {}
                        hall_proj[sid][step] = np.dot(hall_states[sid, lidx, step, :], v_rand)

            hall_mean = defaultdict(list)
            for sid, step_vals in hall_proj.items():
                beh = hall_behaviors.get(sid, "unknown")
                for step, val in step_vals.items():
                    hall_mean[(beh, step)].append(val)

            syc_sids = syc_info["sample_id"].values.astype(int)
            syc_behaviors = syc_info.set_index("sample_id")["final_behavior"]

            syc_proj = {}
            for sid in syc_sids:
                for step in range(syc_states.shape[2]):
                    if syc_valid[sid, lidx, step]:
                        if sid not in syc_proj:
                            syc_proj[sid] = {}
                        syc_proj[sid][step] = np.dot(syc_states[sid, lidx, step, :], v_rand)

            syc_mean = defaultdict(list)
            for sid, step_vals in syc_proj.items():
                beh = syc_behaviors.get(sid, "unknown")
                for step, val in step_vals.items():
                    syc_mean[(beh, step)].append(val)

            max_hall_sep = 0.0
            for step in range(hall_states.shape[2]):
                h_vals = hall_mean.get(("hallucination", step), [])
                a_vals = hall_mean.get(("abstention", step), [])
                if h_vals and a_vals:
                    sep = abs(np.mean(h_vals) - np.mean(a_vals))
                    if sep > max_hall_sep:
                        max_hall_sep = sep

            max_syc_sep = 0.0
            for step in range(syc_states.shape[2]):
                s_vals = syc_mean.get(("sycophantic", step), [])
                n_vals = syc_mean.get(("non_sycophantic", step), [])
                if s_vals and n_vals:
                    sep = abs(np.mean(s_vals) - np.mean(n_vals))
                    if sep > max_syc_sep:
                        max_syc_sep = sep

            layer_stats.append({
                "seed": seed,
                "max_hall_separation": max_hall_sep,
                "max_syc_separation": max_syc_sep,
            })

        random_stats[int(layer)] = layer_stats

    return random_stats


def _save_results(hall_projections, syc_projections, avg_curves, separation_stats, random_stats,
                  vectors, hall_data, syc_data, target_layers):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    direction_names = ["v_hall", "v_syc", "random", "shuffled"]

    for layer in target_layers:
        for dname in direction_names:
            key = (int(layer), dname)
            hall_arr = hall_projections[key]
            syc_arr = syc_projections[key]
            fname = f"projections_layer{int(layer)}_{dname}.npz"
            np.savez_compressed(
                os.path.join(RESULTS_DIR, fname),
                hall_projections=hall_arr,
                syc_projections=syc_arr,
                layer=int(layer),
                direction=dname,
            )

    for layer in target_layers:
        vecs = vectors[int(layer)]
        fname = f"steering_vectors_layer{int(layer)}.npz"
        np.savez_compressed(
            os.path.join(RESULTS_DIR, fname),
            v_hall=vecs["v_hall"],
            v_syc=vecs["v_syc"],
            random=vecs["random"],
            shuffled=vecs["shuffled"],
            layer=int(layer),
        )

    curve_rows = []
    for layer in target_layers:
        for dname in direction_names:
            key = (int(layer), dname)
            curves = avg_curves[key]

            for task, task_curves in curves.items():
                for behavior, curve in task_curves.items():
                    for step, val in enumerate(curve):
                        if not np.isnan(val):
                            curve_rows.append({
                                "layer": int(layer),
                                "direction": dname,
                                "task": task,
                                "behavior": behavior,
                                "step": step,
                                "mean_projection": float(val),
                            })
    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(os.path.join(RESULTS_DIR, "average_projection_curves.csv"), index=False)

    stat_rows = []
    for (layer, dname), stat in separation_stats.items():
        for comparison, comp_stat in stat.items():
            row = {
                "layer": layer,
                "direction": dname,
                "comparison": comparison,
                "earliest_separation_step": comp_stat.get("earliest_separation_step"),
                "max_separation_step": comp_stat.get("max_separation_step"),
                "max_separation_value": comp_stat.get("max_separation_value"),
                "late_stage_variance": comp_stat.get("late_stage_variance"),
                "n_valid_steps": comp_stat.get("n_valid_steps"),
            }
            if "collapse_ratio" in comp_stat:
                row["collapse_ratio"] = comp_stat["collapse_ratio"]
            stat_rows.append(row)
    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(os.path.join(RESULTS_DIR, "separation_statistics.csv"), index=False)

    per_sample_rows = []
    hall_info = hall_data["sample_info"]
    syc_info = syc_data["sample_info"]

    for layer in target_layers:
        for dname in direction_names:
            key = (int(layer), dname)
            hall_arr = hall_projections[key]
            syc_arr = syc_projections[key]

            for _, row in hall_info.iterrows():
                sid = int(row["sample_id"])
                for step in range(hall_arr.shape[1]):
                    val = hall_arr[sid, step]
                    if not np.isnan(val):
                        per_sample_rows.append({
                            "sample_id": sid,
                            "task_type": "hallucination",
                            "final_behavior": row["final_behavior"],
                            "layer": int(layer),
                            "direction": dname,
                            "step": step,
                            "projection": float(val),
                        })

            for _, row in syc_info.iterrows():
                sid = int(row["sample_id"])
                for step in range(syc_arr.shape[1]):
                    val = syc_arr[sid, step]
                    if not np.isnan(val):
                        per_sample_rows.append({
                            "sample_id": sid,
                            "task_type": "sycophancy",
                            "final_behavior": row["final_behavior"],
                            "layer": int(layer),
                            "direction": dname,
                            "step": step,
                            "projection": float(val),
                        })

    per_sample_df = pd.DataFrame(per_sample_rows)
    per_sample_df.to_csv(os.path.join(RESULTS_DIR, "per_sample_projections.csv"), index=False)

    rand_rows = []
    for layer, layer_stats in random_stats.items():
        for s in layer_stats:
            rand_rows.append({
                "layer": layer,
                "seed": s["seed"],
                "max_hall_separation": s["max_hall_separation"],
                "max_syc_separation": s["max_syc_separation"],
            })
    rand_df = pd.DataFrame(rand_rows)
    rand_df.to_csv(os.path.join(RESULTS_DIR, "random_baseline_statistics.csv"), index=False)

    _log(f"  Saved projection arrays, curves, statistics, and per-sample data to {RESULTS_DIR}/")


def _generate_report(avg_curves, separation_stats, random_stats, vectors,
                     hall_data, syc_data, target_layers, elapsed, log):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "IC4_T1_PROJECTION_REPORT.md")

    direction_names = ["v_hall", "v_syc", "random", "shuffled"]
    hall_info = hall_data["sample_info"]
    syc_info = syc_data["sample_info"]

    lines = []
    lines.append("# IC-4-T1: Projection Analysis Report")
    lines.append("")
    lines.append("> Projects hidden-state trajectories onto behavior-relevant directions")
    lines.append("> to characterize when and how behavioral divergence becomes visible")
    lines.append("> in the model's internal representations.")
    lines.append("")

    lines.append("## 1. Setup")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append(f"| Target layers | {list(target_layers)} |")
    lines.append(f"| Main analysis layer | {MAIN_LAYER} |")
    lines.append(f"| Prefill step | {PREFILL_STEP} |")
    lines.append(f"| Separation threshold | {SEPARATION_THRESHOLD} |")
    lines.append(f"| Late-stage fraction | {LATE_STAGE_FRACTION} |")
    lines.append(f"| Random baseline seeds | {N_RANDOM_SEEDS} |")
    lines.append(f"| Hallucination samples | {len(hall_info)} |")
    lines.append(f"| Sycophancy samples | {len(syc_info)} |")
    lines.append(f"| Elapsed | {elapsed:.0f}s ({elapsed/60:.1f} min) |")
    lines.append("")

    lines.append("## 2. Behavior Distribution")
    lines.append("")
    lines.append("### Hallucination Task")
    lines.append("")
    lines.append("| Behavior | Count |")
    lines.append("|---|---|")
    for beh, cnt in hall_info["final_behavior"].value_counts().items():
        lines.append(f"| {beh} | {cnt} |")
    lines.append("")

    lines.append("### Sycophancy Task")
    lines.append("")
    lines.append("| Behavior | Count |")
    lines.append("|---|---|")
    for beh, cnt in syc_info["final_behavior"].value_counts().items():
        lines.append(f"| {beh} | {cnt} |")
    lines.append("")

    lines.append("## 3. Steering Vector Properties")
    lines.append("")

    for layer in target_layers:
        vecs = vectors[int(layer)]
        lines.append(f"### Layer {int(layer)}")
        lines.append("")
        lines.append("| Direction | L2 Norm | Cosine(v_hall, v_syc) |")
        lines.append("|---|---|---|")
        for dname in direction_names:
            norm = float(np.linalg.norm(vecs[dname]))
            lines.append(f"| {dname} | {norm:.6f} | — |")
        cos_hall_syc = float(np.dot(vecs["v_hall"], vecs["v_syc"]) /
                             (np.linalg.norm(vecs["v_hall"]) * np.linalg.norm(vecs["v_syc"]) + 1e-12))
        lines.append(f"| cosine(v_hall, v_syc) | — | {cos_hall_syc:.6f} |")

        cos_hall_rand = float(np.dot(vecs["v_hall"], vecs["random"]) /
                              (np.linalg.norm(vecs["v_hall"]) * np.linalg.norm(vecs["random"]) + 1e-12))
        cos_hall_shuf = float(np.dot(vecs["v_hall"], vecs["shuffled"]) /
                              (np.linalg.norm(vecs["v_hall"]) * np.linalg.norm(vecs["shuffled"]) + 1e-12))
        cos_syc_rand = float(np.dot(vecs["v_syc"], vecs["random"]) /
                             (np.linalg.norm(vecs["v_syc"]) * np.linalg.norm(vecs["random"]) + 1e-12))
        cos_syc_shuf = float(np.dot(vecs["v_syc"], vecs["shuffled"]) /
                             (np.linalg.norm(vecs["v_syc"]) * np.linalg.norm(vecs["shuffled"]) + 1e-12))
        lines.append(f"| cosine(v_hall, random) | — | {cos_hall_rand:.6f} |")
        lines.append(f"| cosine(v_hall, shuffled) | — | {cos_hall_shuf:.6f} |")
        lines.append(f"| cosine(v_syc, random) | — | {cos_syc_rand:.6f} |")
        lines.append(f"| cosine(v_syc, shuffled) | — | {cos_syc_shuf:.6f} |")
        lines.append("")

    lines.append("## 4. Main Results: Layer 12")
    lines.append("")

    main_key = (MAIN_LAYER, "v_hall")
    if main_key in separation_stats and "hall_vs_abst" in separation_stats[main_key]:
        s = separation_stats[main_key]["hall_vs_abst"]
        lines.append("### 4.1 Hallucination vs Abstention (v_hall direction)")
        lines.append("")
        lines.append("| Statistic | Value |")
        lines.append("|---|---|")
        lines.append(f"| Earliest visible separation step | {s['earliest_separation_step']} |")
        lines.append(f"| Max separation step | {s['max_separation_step']} |")
        lines.append(f"| Max separation value | {s['max_separation_value']} |")
        lines.append(f"| Late-stage variance | {s['late_stage_variance']:.6f} |")
        lines.append(f"| Valid steps | {s['n_valid_steps']} |")
        lines.append("")

        per_step_p = s.get("per_step_p", [])
        sig_steps = [i for i, p in enumerate(per_step_p) if p is not None and p < 0.05]
        lines.append(f"Steps with significant separation (p<0.05): {sig_steps if sig_steps else 'none'}")
        lines.append("")

    main_key_syc = (MAIN_LAYER, "v_syc")
    if main_key_syc in separation_stats and "syc_vs_nonsyc" in separation_stats[main_key_syc]:
        s = separation_stats[main_key_syc]["syc_vs_nonsyc"]
        lines.append("### 4.2 Sycophantic vs Non-Sycophantic (v_syc direction)")
        lines.append("")
        lines.append("| Statistic | Value |")
        lines.append("|---|---|")
        lines.append(f"| Earliest visible separation step | {s['earliest_separation_step']} |")
        lines.append(f"| Max separation step | {s['max_separation_step']} |")
        lines.append(f"| Max separation value | {s['max_separation_value']} |")
        lines.append(f"| Late-stage variance | {s['late_stage_variance']:.6f} |")
        lines.append(f"| Collapse ratio | {s.get('collapse_ratio', 'N/A')} |")
        lines.append(f"| Valid steps | {s['n_valid_steps']} |")
        lines.append("")

        per_step_p = s.get("per_step_p", [])
        sig_steps = [i for i, p in enumerate(per_step_p) if p is not None and p < 0.05]
        lines.append(f"Steps with significant separation (p<0.05): {sig_steps if sig_steps else 'none'}")
        lines.append("")

        collapse = s.get("collapse_ratio")
        if collapse is not None:
            if collapse < 0.3:
                lines.append("**Interpretation: Strong collapse** — sycophancy signal nearly vanishes during generation.")
            elif collapse < 0.6:
                lines.append("**Interpretation: Moderate collapse** — sycophancy signal weakens but persists during generation.")
            else:
                lines.append("**Interpretation: No collapse** — sycophancy signal is maintained throughout generation.")
        lines.append("")

    lines.append("### 4.3 Control Directions (random, shuffled)")
    lines.append("")
    lines.append("| Direction | Comparison | Earliest Sep | Max Sep Value | Late-Stage Var |")
    lines.append("|---|---|---|---|---|")
    for dname in ["random", "shuffled"]:
        key = (MAIN_LAYER, dname)
        if key in separation_stats:
            for comp_name, comp_stat in separation_stats[key].items():
                lines.append(f"| {dname} | {comp_name} | {comp_stat.get('earliest_separation_step', 'N/A')} | "
                             f"{comp_stat.get('max_separation_value', 'N/A')} | "
                             f"{comp_stat.get('late_stage_variance', 0):.6f} |")
    lines.append("")

    if MAIN_LAYER in random_stats:
        rand_layer = random_stats[MAIN_LAYER]
        hall_seps = [s["max_hall_separation"] for s in rand_layer]
        syc_seps = [s["max_syc_separation"] for s in rand_layer]
        lines.append("#### Random Baseline (5 seeds)")
        lines.append("")
        lines.append("| Metric | Mean | Std |")
        lines.append("|---|---|---|")
        lines.append(f"| Max hall separation (random dir) | {np.mean(hall_seps):.6f} | {np.std(hall_seps):.6f} |")
        lines.append(f"| Max syc separation (random dir) | {np.mean(syc_seps):.6f} | {np.std(syc_seps):.6f} |")
        lines.append("")

    lines.append("## 5. Key Questions")
    lines.append("")

    lines.append("### Q1: When do hallucination vs abstention separate?")
    lines.append("")
    q1_key = (MAIN_LAYER, "v_hall")
    if q1_key in separation_stats and "hall_vs_abst" in separation_stats[q1_key]:
        s = separation_stats[q1_key]["hall_vs_abst"]
        earliest = s["earliest_separation_step"]
        if earliest is not None:
            if earliest == 0:
                lines.append(f"Separation is visible from the **prefill step** (step 0). "
                             "The model's internal representation already distinguishes hallucination-prone "
                             "from abstention-prone inputs before any generation occurs.")
            else:
                lines.append(f"Separation first becomes visible at **step {earliest}**. "
                             f"The model takes {earliest} forward pass(es) before hallucination vs abstention "
                             "divergence is detectable along v_hall.")
        else:
            lines.append("No separation detected above threshold along v_hall direction.")
    else:
        lines.append("Insufficient data to answer this question.")
    lines.append("")

    lines.append("### Q2: Sycophancy — does syc vs non-syc separate at prefill? Does it collapse?")
    lines.append("")
    q2_key = (MAIN_LAYER, "v_syc")
    if q2_key in separation_stats and "syc_vs_nonsyc" in separation_stats[q2_key]:
        s = separation_stats[q2_key]["syc_vs_nonsyc"]
        earliest = s["earliest_separation_step"]
        collapse = s.get("collapse_ratio")

        lines.append("| Statistic | Value |")
        lines.append("|---|---|")
        lines.append(f"| Earliest visible separation step | {earliest} |")
        lines.append(f"| Max separation step | {s['max_separation_step']} |")
        lines.append(f"| Max separation value | {s['max_separation_value']} |")
        lines.append(f"| Late-stage variance | {s['late_stage_variance']:.6f} |")
        lines.append(f"| Collapse ratio | {collapse if collapse is not None else 'N/A'} |")
        lines.append(f"| Valid steps | {s['n_valid_steps']} |")
        lines.append("")

        per_step_p = s.get("per_step_p", [])
        sig_steps = [i for i, p in enumerate(per_step_p) if p is not None and p < 0.05]
        lines.append(f"Steps with significant separation (p<0.05): {sig_steps if sig_steps else 'none'}")
        lines.append("")

        if earliest is not None:
            if earliest == 0:
                lines.append("**Sycophancy separates from prefill (step 0)** — like hallucination, the model's "
                             "internal representation distinguishes sycophancy-prone from correction-prone inputs "
                             "before any generation occurs.")
            else:
                lines.append(f"Sycophancy first separates at step {earliest}.")

        if collapse is not None:
            if collapse < 0.3:
                lines.append(f"**Strong collapse** (ratio={collapse:.3f}): syc signal nearly vanishes during generation. "
                             "This suggests the model's sycophancy decision is mainly in prefill, and generation "
                             "dynamics tend to erase the distinction.")
            elif collapse < 0.6:
                lines.append(f"**Moderate collapse** (ratio={collapse:.3f}): syc signal weakens but persists.")
            else:
                lines.append(f"**No collapse** (ratio={collapse:.3f}): syc signal is maintained throughout generation.")
    else:
        lines.append("**Sycophancy data now available** — syc vs non-syc separation can be computed.")
        lines.append("(Previously blocked: all 30 syc samples were sycophantic → v_syc=0.)")
    lines.append("")

    lines.append("### Q3: Do random/shuffled directions show no structure?")
    lines.append("")
    rand_key = (MAIN_LAYER, "random")
    shuf_key = (MAIN_LAYER, "shuffled")
    has_structure = False

    for key, label in [(rand_key, "random"), (shuf_key, "shuffled")]:
        if key in separation_stats:
            for comp_name, comp_stat in separation_stats[key].items():
                if comp_stat.get("earliest_separation_step") is not None:
                    has_structure = True

    if has_structure:
        lines.append("**Unexpected structure detected** in control directions. "
                     "This could indicate: (1) the hidden state space has low effective dimensionality, "
                     "making random projections likely to capture some variance, or "
                     "(2) the separation threshold is too low. "
                     "Compare control separation magnitudes to v_hall/v_syc separation magnitudes "
                     "to assess significance.")
    else:
        lines.append("**No structure detected** in random and shuffled directions, "
                     "as expected. This confirms that the separation seen along v_hall and v_syc "
                     "reflects genuine behavior-relevant structure rather than spurious variance.")

    if MAIN_LAYER in random_stats:
        rand_layer = random_stats[MAIN_LAYER]
        hall_seps = [s["max_hall_separation"] for s in rand_layer]
        v_hall_key = (MAIN_LAYER, "v_hall")
        v_hall_max = None
        if v_hall_key in separation_stats and "hall_vs_abst" in separation_stats[v_hall_key]:
            v_hall_max = separation_stats[v_hall_key]["hall_vs_abst"].get("max_separation_value")

        if v_hall_max is not None and len(hall_seps) > 0:
            rand_mean = np.mean(hall_seps)
            lines.append(f"\nQuantitative comparison: v_hall max separation = {v_hall_max:.4f}, "
                         f"random direction mean max separation = {rand_mean:.4f}, "
                         f"ratio = {v_hall_max / (rand_mean + 1e-12):.2f}x.")
    lines.append("")

    lines.append("## 6. Supplementary: Other Layers")
    lines.append("")

    supp_layers = [int(l) for l in target_layers if int(l) != MAIN_LAYER]
    for layer in supp_layers:
        lines.append(f"### Layer {layer}")
        lines.append("")
        lines.append("| Direction | Comparison | Earliest Sep | Max Sep Value | Late-Stage Var |")
        lines.append("|---|---|---|---|---|")
        for dname in direction_names:
            key = (layer, dname)
            if key in separation_stats:
                for comp_name, comp_stat in separation_stats[key].items():
                    lines.append(f"| {dname} | {comp_name} | {comp_stat.get('earliest_separation_step', 'N/A')} | "
                                 f"{comp_stat.get('max_separation_value', 'N/A')} | "
                                 f"{comp_stat.get('late_stage_variance', 0):.6f} |")
        lines.append("")

    lines.append("## 7. Representation / Readout Caveat")
    lines.append("")
    lines.append("> All projections in this analysis use `state_last` (last-token hidden state)")
    lines.append("> captured at each forward step. This readout choice has known limitations:")
    lines.append(">")
    lines.append("> 1. **Position bias**: The last token position may not be the most informative")
    lines.append(">    readout for all behavioral signals. During prefill, the last token is the")
    lines.append(">    final prompt token; during decode, it is the newly generated token.")
    lines.append(">    These are fundamentally different positions with different informational roles.")
    lines.append(">")
    lines.append("> 2. **Window4 alternative**: The T0 capture also recorded `state_window4` (mean of")
    lines.append(">    last 4 positions during prefill), which may better capture distributed signals.")
    lines.append(">    However, during decode steps (seq_len=1), window4 is identical to state_last.")
    lines.append(">")
    lines.append("> 3. **Projection ≠ causation**: Finding that trajectories separate along v_hall")
    lines.append(">    does not imply that v_hall direction *causes* hallucination. It only shows")
    lines.append(">    that the model's internal representations encode behavior-relevant information")
    lines.append(">    that is linearly accessible. Causal claims require intervention experiments.")
    lines.append(">")
    lines.append("> 4. **Normalization**: Steering vectors are L2-normalized. Projection magnitudes")
    lines.append(">    are therefore in units of 'hidden-state component along the normalized direction'.")
    lines.append(">    Cross-direction magnitude comparisons are valid; cross-layer comparisons require")
    lines.append(">    caution because hidden-state norms may vary by layer.")
    lines.append(">")
    lines.append("> 5. **Sample size**: Small behavior-class counts (especially for sycophancy)")
    lines.append(">    limit statistical power. Per-step t-tests should be interpreted with")
    lines.append(">    Bonferroni or FDR correction for the number of steps tested.")
    lines.append("")

    lines.append("## 8. Output Files")
    lines.append("")
    lines.append(f"- `{RESULTS_DIR}/projections_layer*_*.npz` — per-sample projection arrays")
    lines.append(f"- `{RESULTS_DIR}/steering_vectors_layer*.npz` — computed steering vectors")
    lines.append(f"- `{RESULTS_DIR}/average_projection_curves.csv` — mean projection per (layer, direction, task, behavior, step)")
    lines.append(f"- `{RESULTS_DIR}/separation_statistics.csv` — separation statistics summary")
    lines.append(f"- `{RESULTS_DIR}/per_sample_projections.csv` — per-sample projection values")
    lines.append(f"- `{RESULTS_DIR}/random_baseline_statistics.csv` — random direction baselines")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*IC-4-T1: Projection Analysis*")
    lines.append("*Generated by run_t1_projection_analysis.py*")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    _log(f"  Report saved to {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="IC-4-T1: Projection Analysis")
    parser.add_argument("--skip-random-baseline", action="store_true",
                        help="Skip random baseline computation (faster but less rigorous)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    log_path = os.path.join(RESULTS_DIR, "run_log.txt")
    _log("=" * 60, log_path)
    _log("IC-4-T1: Projection Analysis", log_path)
    _log("=" * 60, log_path)

    t_start = time.time()

    _log("\nLoading T0 trajectory data...", log_path)
    hall_data = _load_trajectory_data("hallucination")
    syc_data = _load_trajectory_data("sycophancy", data_dir=T0_SYC_CONTRAST_DIR)
    target_layers = hall_data["target_layers"]
    _log(f"  Hallucination: {hall_data['states_last'].shape} samples, hidden_dim={hall_data['hidden_dim']}", log_path)
    _log(f"  Sycophancy (balanced): {syc_data['states_last'].shape} samples, hidden_dim={syc_data['hidden_dim']}", log_path)
    _log(f"    Source: {T0_SYC_CONTRAST_DIR}", log_path)
    _log(f"  Target layers: {target_layers}", log_path)

    _log("\nComputing steering vectors...", log_path)
    vectors = _compute_steering_vectors(hall_data, syc_data, target_layers)
    for layer in target_layers:
        vecs = vectors[int(layer)]
        cos_hs = float(np.dot(vecs["v_hall"], vecs["v_syc"]) /
                       (np.linalg.norm(vecs["v_hall"]) * np.linalg.norm(vecs["v_syc"]) + 1e-12))
        _log(f"  Layer {int(layer)}: cos(v_hall, v_syc)={cos_hs:.4f}", log_path)

    _log("\nProjecting trajectories...", log_path)
    hall_projections = _project_trajectories(hall_data, vectors, target_layers)
    syc_projections = _project_trajectories(syc_data, vectors, target_layers)
    _log(f"  Projected {len(hall_projections)} hallucination (layer, direction) pairs", log_path)
    _log(f"  Projected {len(syc_projections)} sycophancy (layer, direction) pairs", log_path)

    sample_info_combined = {
        "hallucination": hall_data["sample_info"],
        "sycophancy": syc_data["sample_info"],
    }

    _log("\nComputing average projection curves...", log_path)
    avg_curves = _compute_average_curves(
        hall_projections, syc_projections, sample_info_combined, target_layers)

    _log("\nComputing separation statistics...", log_path)
    separation_stats = _compute_separation_statistics(
        avg_curves, hall_projections, syc_projections, sample_info_combined, target_layers)

    main_key = (MAIN_LAYER, "v_hall")
    if main_key in separation_stats and "hall_vs_abst" in separation_stats[main_key]:
        s = separation_stats[main_key]["hall_vs_abst"]
        _log(f"  v_hall @ L{MAIN_LAYER}: earliest_sep={s['earliest_separation_step']}, "
             f"max_sep={s['max_separation_value']}, late_var={s['late_stage_variance']:.6f}", log_path)

    main_key_syc = (MAIN_LAYER, "v_syc")
    if main_key_syc in separation_stats and "syc_vs_nonsyc" in separation_stats[main_key_syc]:
        s = separation_stats[main_key_syc]["syc_vs_nonsyc"]
        _log(f"  v_syc  @ L{MAIN_LAYER}: earliest_sep={s['earliest_separation_step']}, "
             f"collapse_ratio={s.get('collapse_ratio', 'N/A')}", log_path)

    random_stats = {}
    if not args.skip_random_baseline:
        _log("\nComputing random baseline statistics...", log_path)
        random_stats = _compute_random_baseline_statistics(
            hall_data, syc_data, target_layers, hall_data["hidden_dim"])
        for layer in target_layers:
            if int(layer) in random_stats:
                rs = random_stats[int(layer)]
                hall_seps = [s["max_hall_separation"] for s in rs]
                _log(f"  Layer {int(layer)}: random max_hall_sep = {np.mean(hall_seps):.4f} ± {np.std(hall_seps):.4f}", log_path)

    _log("\nSaving results...", log_path)
    _save_results(hall_projections, syc_projections, avg_curves, separation_stats, random_stats,
                  vectors, hall_data, syc_data, target_layers)

    elapsed = time.time() - t_start
    _log(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)", log_path)

    _log("\nGenerating report...", log_path)
    _generate_report(avg_curves, separation_stats, random_stats, vectors,
                     hall_data, syc_data, target_layers, elapsed, log_path)

    _log("\nIC-4-T1 complete.", log_path)


if __name__ == "__main__":
    main()
