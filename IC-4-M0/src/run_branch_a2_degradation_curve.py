"""
Branch A2: Probe Degradation Curve and Soft-Gate Robustness.

Upgrades Branch A from single-trial stress test to statistically-backed degradation curve.
Multi-repeat resampling per train size, stratified gate comparison, aggregate statistics.

Design:
  - 6 sizes: [15, 10, 7, 5, 3, 2] per class
  - Stratified repeats: 1x (n=15,10), 2x (n=7), 5x (n=5,3,2)
  - Gate configs per size tuned for informativeness
  - Incremental CSV writes for crash-safety
  - Optional harder evaluation on hard OOD data
"""
import os, sys, time, json, random, csv, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from collections import defaultdict
from src.gate_steering_tool import GateSteeringTool
from src.model_loader import load_model_and_tokenizer
from src.data_builder import load_jsonl

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "results_branch_a2")
REPORT_DIR = os.path.join(BASE_DIR, "reports_branch_a2")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

ALPHA = -1.0
LAYER = 12
SEED = 0
ORACLE_H = 0.667
HARD_OOD_ORACLE_H = 0.733

SIZE_REPEATS = {
    15: 1,
    10: 1,
    7: 2,
    5: 5,
    3: 5,
    2: 5,
}

GATE_BY_SIZE = {
    15: ["hard", "soft_T0.1"],
    10: ["hard", "soft_T0.1"],
    7: ["hard", "soft_T0.1", "soft_T0.3"],
    5: ["hard", "soft_T0.1", "soft_T0.3", "confidence_aware"],
    3: ["hard", "soft_T0.1", "soft_T0.3", "confidence_aware"],
    2: ["hard", "soft_T0.1", "soft_T0.3"],
}

ALL_GATE_CFGS = {
    "hard": {"gate_type": "hard"},
    "soft_T0.1": {"gate_type": "soft", "soft_temperature": 0.1},
    "soft_T0.3": {"gate_type": "soft", "soft_temperature": 0.3},
    "confidence_aware": {"gate_type": "confidence_aware", "confidence_zone": (0.35, 0.65)},
}

CSV_PATH = os.path.join(OUT_DIR, "degradation_curve.csv")
LOG_PATH = os.path.join(OUT_DIR, "run_log.txt")
FIELDNAMES = [
    "n_per_class", "repeat_idx", "subsample_seed",
    "train_n_pos", "train_n_neg",
    "probe_train_acc", "probe_cv_acc", "probe_test_acc",
    "gate_type", "H", "C", "UA", "oracle_gap",
    "gate_on_count", "gate_mean", "gate_std",
    "gen_time_s", "train_time_s",
]


LOG_FILE_PATH = os.path.join(OUT_DIR, "run_log.txt")

def log(msg):
    print(msg, flush=True)
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()
    except Exception:
        pass


def subsample_by_class(data, n_per_class, rng):
    pos = [s for s in data if s.get("answerability") == "answerable"]
    neg = [s for s in data if s.get("answerability") == "unanswerable"]
    n_pos = min(n_per_class, len(pos))
    n_neg = min(n_per_class, len(neg))
    return rng.sample(pos, n_pos) + rng.sample(neg, n_neg)


def write_rows(rows, path, mode="a"):
    write_header = (mode == "w") or (not os.path.exists(path))
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def log_file(msg, path):
    with open(path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()


def run_one_trial(model, tokenizer, sv, full_train, test, n_per_class, repeat_idx,
                  subsample_seed, gate_names, log_path="", oracle_h=None):
    try:
        return _run_one_trial_inner(model, tokenizer, sv, full_train, test,
                                    n_per_class, repeat_idx, subsample_seed, gate_names, oracle_h)
    except Exception as e:
        msg = f"  !! CRASH n={n_per_class} rep={repeat_idx}: {e}"
        log(msg)
        if log_path:
            log_file(msg, log_path)
        return [], 0.0


def _run_one_trial_inner(model, tokenizer, sv, full_train, test, n_per_class, repeat_idx,
                          subsample_seed, gate_names, oracle_h=None):
    rng = random.Random(subsample_seed)
    sub_train = subsample_by_class(full_train, n_per_class, rng)
    na = sum(1 for s in sub_train if s.get("answerability") == "answerable")
    nu = len(sub_train) - na

    cv_folds = min(3, n_per_class)
    tool = GateSteeringTool(model, tokenizer, config={
        "threshold": 0.5, "cv_folds": cv_folds, "max_new_tokens": 48,
        "temperature": 0.0, "do_sample": False,
    })

    t_probe = time.time()
    probe_info = tool.train_probe(sub_train, LAYER, "last_prompt_token")
    train_time = time.time() - t_probe

    probe_eval = tool.evaluate_probe(test, probe_info)

    rows = []
    for gname in gate_names:
        gcfg = ALL_GATE_CFGS[gname]
        kwargs = dict(gcfg)
        t_gen = time.time()
        results = tool.generate_batch(test, sv, LAYER, ALPHA,
                                       probe_info=probe_info, control_type="steering",
                                       **kwargs)
        gen_time = time.time() - t_gen
        metrics = tool.evaluate(results)

        gate_vals = [r["gate"] for r in results]
        gate_on = sum(1 for g in gate_vals if g > 0.01)
        gate_mean = float(np.mean(gate_vals)) if gate_vals else 0.0
        gate_std = float(np.std(gate_vals)) if gate_vals else 0.0

        oh_ref_val = oracle_h if oracle_h is not None else ORACLE_H

        row = {
            "n_per_class": n_per_class,
            "repeat_idx": repeat_idx,
            "subsample_seed": subsample_seed,
            "train_n_pos": na,
            "train_n_neg": nu,
            "probe_train_acc": round(probe_info["train_acc"], 4),
            "probe_cv_acc": round(probe_info["cv_acc_mean"], 4) if probe_info.get("cv_acc_mean") is not None else "",
            "probe_test_acc": round(probe_eval["accuracy"], 4),
            "gate_type": gname,
            "H": round(metrics["hallucination_rate"], 4),
            "C": round(metrics["correct_answer_rate"], 4),
            "UA": round(metrics["unnecessary_abstention_rate"], 4),
            "oracle_gap": round(metrics["hallucination_rate"] - oh_ref_val, 4),
            "gate_on_count": gate_on,
            "gate_mean": round(gate_mean, 4),
            "gate_std": round(gate_std, 4),
            "gen_time_s": round(gen_time, 1),
            "train_time_s": round(train_time, 1),
        }
        rows.append(row)

    return rows, train_time


def get_completed_trials(csv_path):
    completed = set()
    if not os.path.exists(csv_path):
        return completed
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            completed.add((int(r["n_per_class"]), int(r["repeat_idx"])))
    return completed


def compute_aggregates(rows):
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["n_per_class"], r["gate_type"])
        groups[key]["probe_test_acc"].append(r["probe_test_acc"])
        groups[key]["H"].append(r["H"])
        groups[key]["C"].append(r["C"])
        groups[key]["UA"].append(r["UA"])
        groups[key]["oracle_gap"].append(r["oracle_gap"])
        groups[key]["gate_on_count"].append(r["gate_on_count"])

    agg_rows = []
    for (n_cls, gtype), vals in sorted(groups.items()):
        agg_rows.append({
            "n_per_class": n_cls,
            "gate_type": gtype,
            "n_trials": len(vals["probe_test_acc"]),
            "probe_test_acc_mean": round(np.mean(vals["probe_test_acc"]), 4),
            "probe_test_acc_std": round(np.std(vals["probe_test_acc"]), 4),
            "H_mean": round(np.mean(vals["H"]), 4),
            "H_std": round(np.std(vals["H"]), 4),
            "C_mean": round(np.mean(vals["C"]), 4),
            "C_std": round(np.std(vals["C"]), 4),
            "UA_mean": round(np.mean(vals["UA"]), 4),
            "UA_std": round(np.std(vals["UA"]), 4),
            "oracle_gap_mean": round(np.mean(vals["oracle_gap"]), 4),
            "oracle_gap_std": round(np.std(vals["oracle_gap"]), 4),
            "gate_on_mean": round(np.mean(vals["gate_on_count"]), 1),
            "gate_on_std": round(np.std(vals["gate_on_count"]), 1),
        })
    return agg_rows


def run_primary(model, tokenizer, sv, full_train, test, resume=False, n_list=None):
    log("=" * 70)
    log("PRIMARY EXPERIMENT: Standard Test Set (seed=0)")
    log("=" * 70)

    completed = get_completed_trials(CSV_PATH) if resume else set()
    all_n_values = [15, 10, 7, 5, 3, 2]
    if n_list:
        n_values = [n for n in n_list if n in all_n_values]
        log(f"Filtered n_list: {n_values}")
    else:
        n_values = all_n_values

    all_rows = []
    na_full = sum(1 for s in full_train if s.get("answerability") == "answerable")
    nu_full = len(full_train) - na_full
    na_test = sum(1 for s in test if s.get("answerability") == "answerable")
    nu_test = len(test) - na_test
    log(f"Full train: {len(full_train)} ({na_full}A+{nu_full}U)")
    log(f"Test: {len(test)} ({na_test}A+{nu_test}U)")

    if resume and completed:
        log(f"Resume mode: {len(completed)} already-completed (n,rep) found, will skip.")

    total_start = time.time()
    base_subsample_seed = 1000
    csv_exists = os.path.exists(CSV_PATH)
    skipped_count = 0

    for n_cls in n_values:
        n_repeats = SIZE_REPEATS[n_cls]
        gate_names = GATE_BY_SIZE[n_cls]
        log(f"\n{'='*50}")
        log(f"n_per_class={n_cls}, repeats={n_repeats}, gates={gate_names}")
        log(f"{'='*50}")

        for rep in range(n_repeats):
            if (n_cls, rep) in completed:
                skipped_count += 1
                log(f"  rep={rep} SKIP (already completed)")
                continue
            ss_seed = base_subsample_seed + n_cls * 100 + rep
            rows, train_t = run_one_trial(model, tokenizer, sv, full_train, test,
                                          n_cls, rep, ss_seed, gate_names,
                                          log_path=LOG_PATH)
            all_rows.extend(rows)
            write_rows(rows, CSV_PATH, mode="a" if csv_exists else "w")
            csv_exists = True

            for r in rows:
                log(f"  rep={rep} seed={ss_seed} probe_acc={r['probe_test_acc']:.3f} "
                    f"{r['gate_type']:<20} H={r['H']:.3f} C={r['C']:.3f} UA={r['UA']:.3f} "
                    f"gap={r['oracle_gap']:+.3f} gate_on={r['gate_on_count']}")

    total_elapsed = time.time() - total_start
    log(f"\nPrimary experiment complete: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    return all_rows


def run_harder_eval(model, tokenizer, sv, train_hard, test_hard):
    log("\n" + "=" * 70)
    log("HARDER EVALUATION: Hard OOD Test Set (seed=0, n_per_class=5)")
    log("=" * 70)

    na_train = sum(1 for s in train_hard if s.get("answerability") == "answerable")
    nu_train = len(train_hard) - na_train
    na_test = sum(1 for s in test_hard if s.get("answerability") == "answerable")
    nu_test = len(test_hard) - na_test
    log(f"Train: {len(train_hard)} ({na_train}A+{nu_train}U)")
    log(f"Test: {len(test_hard)} ({na_test}A+{nu_test}U)")

    gate_names = ["hard", "soft_T0.1"]
    rows, _ = run_one_trial(model, tokenizer, sv, train_hard, test_hard,
                            5, 0, 5000, gate_names, log_path=LOG_PATH, oracle_h=HARD_OOD_ORACLE_H)

    hard_csv = os.path.join(OUT_DIR, "harder_eval.csv")
    write_rows(rows, hard_csv, mode="w")

    for r in rows:
        log(f"  probe_acc={r['probe_test_acc']:.3f} "
            f"{r['gate_type']:<20} H={r['H']:.3f} C={r['C']:.3f} UA={r['UA']:.3f} "
            f"gap={r['oracle_gap']:+.3f} gate_on={r['gate_on_count']}")
    return rows


def generate_report(all_rows, harder_rows=None):
    agg = compute_aggregates(all_rows)

    log("\n" + "=" * 70)
    log("AGGREGATE STATISTICS (by n_per_class × gate_type)")
    log("=" * 70)
    header = (f"{'n':<4} {'gate':<20} {'n_tr':<5} {'p_acc':<12} "
              f"{'H':<12} {'UA':<12} {'gap':<12} {'gate_on':<10}")
    log(header)
    log("-" * len(header))
    for r in agg:
        p_acc = f"{r['probe_test_acc_mean']:.3f}±{r['probe_test_acc_std']:.3f}"
        h_str = f"{r['H_mean']:.3f}±{r['H_std']:.3f}"
        ua_str = f"{r['UA_mean']:.3f}±{r['UA_std']:.3f}"
        gap_str = f"{r['oracle_gap_mean']:+.3f}±{r['oracle_gap_std']:.3f}"
        gon_str = f"{r['gate_on_mean']:.1f}±{r['gate_on_std']:.1f}"
        log(f"{r['n_per_class']:<4} {r['gate_type']:<20} {r['n_trials']:<5} "
            f"{p_acc:<12} {h_str:<12} {ua_str:<12} {gap_str:<12} {gon_str:<10}")

    agg_csv = os.path.join(OUT_DIR, "aggregate_stats.csv")
    agg_fields = list(agg[0].keys())
    with open(agg_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields)
        w.writeheader()
        w.writerows(agg)

    report_path = os.path.join(REPORT_DIR, "BRANCH_A2_DEGRADATION_REPORT.md")
    _write_md_report(report_path, all_rows, agg, harder_rows)
    log(f"\nReport saved to {report_path}")
    log(f"Aggregate stats saved to {agg_csv}")
    log(f"Raw data saved to {CSV_PATH}")

    _write_verdict(agg, harder_rows)
    return agg


def _write_md_report(path, all_rows, agg, harder_rows):
    lines = []
    lines.append("# Branch A2: Probe Degradation Curve and Soft-Gate Robustness")
    lines.append("")
    lines.append("> Upgrades Branch A from single-trial stress test to statistically-backed")
    lines.append("> degradation curve with multi-repeat resampling per train size.")
    lines.append("")
    lines.append("## 1. Experiment Design")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append("| Model | Qwen/Qwen2.5-0.5B-Instruct |")
    lines.append("| Seed | 0 |")
    lines.append("| Layer | 12 |")
    lines.append("| Alpha | -1.0 |")
    lines.append("| Oracle H | 0.667 |")
    lines.append("| Probe type | last_prompt_token + logistic |")
    lines.append("| Test set | 60 samples (30A+30U), standard OOD split |")
    lines.append("")
    lines.append("**Stratified sampling:**")
    lines.append("")
    lines.append("| n_per_class | Repeats | Gates tested | Rationale |")
    lines.append("|---|---|---|---|")
    lines.append("| 15 | 1 | hard, soft_T0.1 | Baseline (probe ~100%) |")
    lines.append("| 10 | 1 | hard, soft_T0.1 | Baseline (probe ~100%) |")
    lines.append("| 7 | 2 | hard, soft_T0.1, soft_T0.3 | Transition zone |")
    lines.append("| 5 | 5 | hard, soft_T0.1, soft_T0.3, confidence_aware | Key degradation zone |")
    lines.append("| 3 | 5 | hard, soft_T0.1, soft_T0.3, confidence_aware | Key degradation + high variance |")
    lines.append("| 2 | 5 | hard, soft_T0.1, soft_T0.3 | Breaking point |")
    lines.append("")
    lines.append(f"**Total trials:** {len(all_rows)} (each = 60 generations)")
    lines.append("")
    lines.append("## 2. Aggregate Results")
    lines.append("")
    lines.append("| n | Gate | Trials | Probe Acc | H | UA | Oracle Gap | Gate On |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in agg:
        p_acc = f"{r['probe_test_acc_mean']:.3f}±{r['probe_test_acc_std']:.3f}"
        h_str = f"{r['H_mean']:.3f}±{r['H_std']:.3f}"
        ua_str = f"{r['UA_mean']:.3f}±{r['UA_std']:.3f}"
        gap_str = f"{r['oracle_gap_mean']:+.3f}±{r['oracle_gap_std']:.3f}"
        gon_str = f"{r['gate_on_mean']:.1f}±{r['gate_on_std']:.1f}"
        lines.append(f"| {r['n_per_class']} | {r['gate_type']} | {r['n_trials']} | {p_acc} | {h_str} | {ua_str} | {gap_str} | {gon_str} |")

    if harder_rows:
        lines.append("")
        lines.append("## 3. Harder Evaluation (Hard OOD, n=5)")
        lines.append("")
        lines.append("| Gate | Probe Acc | H | C | UA | Oracle Gap | Gate On |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in harder_rows:
            lines.append(f"| {r['gate_type']} | {r['probe_test_acc']:.3f} | {r['H']:.3f} | {r['C']:.3f} | {r['UA']:.3f} | {r['oracle_gap']:+.3f} | {r['gate_on_count']} |")

    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append("*To be filled after experiment completion.*")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_verdict(agg, harder_rows):
    log("\n" + "=" * 70)
    log("PRELIMINARY VERDICT (to be confirmed by aggregate stats)")
    log("=" * 70)

    n5_hard = [r for r in agg if r["n_per_class"] == 5 and r["gate_type"] == "hard"]
    n5_soft = [r for r in agg if r["n_per_class"] == 5 and r["gate_type"] == "soft_T0.1"]
    n3_hard = [r for r in agg if r["n_per_class"] == 3 and r["gate_type"] == "hard"]
    n3_soft = [r for r in agg if r["n_per_class"] == 3 and r["gate_type"] == "soft_T0.1"]
    n2_hard = [r for r in agg if r["n_per_class"] == 2 and r["gate_type"] == "hard"]
    n2_soft = [r for r in agg if r["n_per_class"] == 2 and r["gate_type"] == "soft_T0.1"]

    for label, hard_r, soft_r in [("n=5", n5_hard, n5_soft), ("n=3", n3_hard, n3_soft), ("n=2", n2_hard, n2_soft)]:
        if hard_r and soft_r:
            h = hard_r[0]
            s = soft_r[0]
            ua_note = ""
            if s["UA_mean"] < h["UA_mean"]:
                ua_note = " ← soft better"
            elif s["UA_mean"] > h["UA_mean"]:
                ua_note = " ← hard better"
            log(f"  {label}: hard(H={h['H_mean']:.3f}, UA={h['UA_mean']:.3f}) "
                f"vs soft_T0.1(H={s['H_mean']:.3f}, UA={s['UA_mean']:.3f}){ua_note}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["primary", "harder", "all"], default="all")
    parser.add_argument("--resume", action="store_true", help="Skip already-completed (n,rep) trials from CSV")
    parser.add_argument("--n-list", type=str, default="",
                        help="Comma-separated n_per_class list to run, e.g. '5,3'. Default: all [15,10,7,5,3,2]")
    parser.add_argument("--skip-primary", action="store_true")
    parser.add_argument("--skip-harder", action="store_true")
    args = parser.parse_args()

    log("=" * 70)
    log("Branch A2: Probe Degradation Curve and Soft-Gate Robustness")
    log("=" * 70)
    log(f"Mode: {args.mode}")

    log("\n[1/3] Loading model...")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-0.5B-Instruct", "cpu", "float32")
    log(f"  Loaded in {time.time()-t0:.0f}s")

    log("\n[2/3] Loading steering vectors...")
    vectors = GateSteeringTool.load_activation(SEED, LAYER)
    sv = vectors["steering"]
    log(f"  Steering dim: {sv.shape[0]}")

    all_rows = []
    harder_rows = None

    if args.mode in ("primary", "all") and not args.skip_primary:
        n_list = None
        if args.n_list:
            n_list = [int(x.strip()) for x in args.n_list.split(",") if x.strip()]
            log(f"n_list from args: {n_list}")
        full_train = load_jsonl("data_m3/train_s0.jsonl")
        test = load_jsonl("data_m3/test_s0.jsonl")
        all_rows = run_primary(model, tokenizer, sv, full_train, test, resume=args.resume, n_list=n_list)

        json_path = os.path.join(OUT_DIR, "degradation_curve.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, indent=2, ensure_ascii=False)
        log(f"Full dataset saved to {json_path}")

    if args.mode in ("harder", "all") and not args.skip_harder:
        train_hard = load_jsonl("data_m4/train_test_hard_s0.jsonl")
        test_hard = load_jsonl("data_m4/test_hard_s0.jsonl")
        harder_rows = run_harder_eval(model, tokenizer, sv, train_hard, test_hard)

    log("\n[3/3] Generating report...")
    if all_rows or (os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0):
        existing_rows = []
        if os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0:
            with open(CSV_PATH, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    r["n_per_class"] = int(r["n_per_class"])
                    r["repeat_idx"] = int(r["repeat_idx"])
                    r["H"] = float(r["H"])
                    r["C"] = float(r["C"])
                    r["UA"] = float(r["UA"])
                    r["oracle_gap"] = float(r["oracle_gap"])
                    r["gate_on_count"] = int(r["gate_on_count"])
                    r["probe_test_acc"] = float(r["probe_test_acc"])
                    existing_rows.append(r)
        all_full_rows = existing_rows
        generate_report(all_full_rows, harder_rows)
    elif harder_rows:
        hard_agg = compute_aggregates(harder_rows)
        generate_report(harder_rows, harder_rows)
    else:
        log("No data to report.")

    log("\nDone.")


if __name__ == "__main__":
    main()