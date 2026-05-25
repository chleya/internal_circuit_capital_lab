"""
IC-4: Position-to-Behavior Sensitivity — CPU-optimized.

Critical experiment: we've proven position shifts representation (KNN=1.0).
Now we test whether position shifts BEHAVIOR (hallucination/abstention rates).

Same evidence content at 3 positions → compare behavioral outcomes.
CPU-friendly: 180 generations (60 samples × 3 positions) on 0.5B model.

Usage:
    python -m src.run_position_behavior --n 60
"""

import argparse
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_loader import load_model_and_tokenizer
from src.data_builder import load_jsonl
from src.evaluate import evaluate_outputs

RESULTS_DIR = "results_position_sensitivity_cpu"


def _log(msg):
    print(msg, flush=True)


def _run_position_behavior(n_samples=60):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    n_threads = int(os.environ.get("OMP_NUM_THREADS", str(min(8, os.cpu_count() or 4))))
    torch.set_num_threads(n_threads)
    _log(f"CPU threads: {n_threads}")

    _log("Loading model (Qwen2.5-0.5B-Instruct) on CPU...")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="cpu",
        torch_dtype="float32",
    )
    _log(f"  Loaded in {time.time() - t0:.1f}s")

    device = next(model.parameters()).device

    seed = 0
    data_dir = os.path.join("data_position_sensitivity", f"s{seed}")
    variants = {}
    for pos_name in ["early", "mid", "late"]:
        vpath = os.path.join(data_dir, f"test_{pos_name}_s{seed}.jsonl")
        samples = load_jsonl(vpath)[:n_samples]
        variants[pos_name] = samples
        _log(f"  {pos_name}: {len(samples)} samples")
        _log(f"    A={sum(1 for s in samples if s.get('answerability')=='answerable')} "
             f"U={sum(1 for s in samples if s.get('answerability')=='unanswerable')}")

    max_new_tokens = 48
    temperature = 0.0
    do_sample = False

    all_results = []
    t_gen_start = time.time()

    for pos_name in ["early", "mid", "late"]:
        _log(f"\n--- Generating [{pos_name}] ---")
        for sid, sample in enumerate(tqdm(variants[pos_name], desc=pos_name)):
            context = sample.get("context", "")
            question = sample.get("question", "")
            prompt = f"{context}\n\nQuestion: {question}\nAnswer:"

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated_ids = outputs[0][input_len:]
            answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            all_results.append({
                "position": pos_name,
                "sample_id": sid,
                "answerability": sample.get("answerability", "?"),
                "gold_answer": sample.get("gold_answer"),
                "context": context[:100],
                "generated_output": answer,
                "n_tokens": len(generated_ids),
            })

    gen_elapsed = time.time() - t_gen_start
    _log(f"\n  {len(all_results)} generations in {gen_elapsed:.0f}s ({gen_elapsed/60:.1f} min)")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "position_behavior_raw.csv"), index=False)

    pos_metrics = {}
    for pos_name in ["early", "mid", "late"]:
        pos_results = results_df[results_df["position"] == pos_name]
        eval_input = []
        for _, row in pos_results.iterrows():
            eval_input.append({
                "generated_output": row["generated_output"],
                "answerability": row["answerability"],
                "gold_answer": row.get("gold_answer"),
            })
        metrics = evaluate_outputs(eval_input)
        pos_metrics[pos_name] = metrics
        _log(f"\n  [{pos_name}] H={metrics['hallucination_rate']:.4f} "
             f"C={metrics['correct_answer_rate']:.4f} "
             f"CA={metrics['calibrated_abstention_rate']:.4f} "
             f"UA={metrics['unnecessary_abstention_rate']:.4f}")

    h_range = max(m["hallucination_rate"] for m in pos_metrics.values()) - \
              min(m["hallucination_rate"] for m in pos_metrics.values())
    c_range = max(m["correct_answer_rate"] for m in pos_metrics.values()) - \
              min(m["correct_answer_rate"] for m in pos_metrics.values())

    _log(f"\n{'='*50}")
    _log(f"POSITION-TO-BEHAVIOR SUMMARY")
    _log(f"{'='*50}")
    _log(f"  Hallucination rate range: {h_range:.4f}")
    _log(f"  Correct rate range: {c_range:.4f}")
    _log(f"  Gate rate range: N/A (base mode)")

    if h_range > 0.1 or c_range > 0.1:
        _log("  VERDICT: Position changes BEHAVIOR — position is a first-order behavioral confound.")
        _log("  → Trajectory analysis MUST model position as an independent variable.")
        _log("  → Routing decisions dependent on hidden state are influenced by position encoding.")
    elif h_range > 0.03 or c_range > 0.03:
        _log("  VERDICT: Moderate position sensitivity — detectable but not dominant.")
        _log("  → Position should be tracked as a secondary variable in trajectory analysis.")
    else:
        _log("  VERDICT: Low position sensitivity — behavior is position-robust.")
        _log("  → RoPE ceiling not yet binding at this scale (~20-30 token context).")

    detailed = []
    for pos_name in ["early", "mid", "late"]:
        m = pos_metrics[pos_name]
        detailed.append({
            "position": pos_name,
            "hallucination_rate": round(m['hallucination_rate'], 4),
            "correct_answer_rate": round(m['correct_answer_rate'], 4),
            "calibrated_abstention_rate": round(m['calibrated_abstention_rate'], 4),
            "unnecessary_abstention_rate": round(m['unnecessary_abstention_rate'], 4),
        })

    summary = {
        "model": "Qwen2.5-0.5B-Instruct",
        "n_samples": n_samples,
        "h_range": round(h_range, 4),
        "c_range": round(c_range, 4),
        "behavior_position_sensitive": h_range > 0.05 or c_range > 0.05,
        "gen_time_s": round(gen_elapsed, 0),
        "per_position": detailed,
    }

    summary_path = os.path.join(RESULTS_DIR, "position_behavior_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    _log(f"\nSummary → {summary_path}")

    report_path = os.path.join(RESULTS_DIR, "POSITION_BEHAVIOR_REPORT.md")
    _generate_report(report_path, summary, results_df)
    _log(f"Report → {report_path}")


def _generate_report(report_path, summary, df):
    lines = []
    lines.append("# IC-4: Position-to-Behavior Sensitivity Report")
    lines.append("")
    lines.append(f"**Experiment:** Relational Memory Hypothesis — Experiment #1b (Behavior Sensitivity, CPU)  ")
    lines.append(f"**Date:** 2026-05-20  ")
    lines.append(f"**Model:** {summary['model']} on CPU  ")
    lines.append(f"**N samples:** {summary['n_samples']} × 3 positions = {summary['n_samples'] * 3} generations  ")
    lines.append(f"**Gen time:** {summary['gen_time_s']:.0f}s ({summary['gen_time_s']/60:.1f} min)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Design")
    lines.append("")
    lines.append("Same evidence content at 3 positions (early/mid/late) via prefix shifting. ")
    lines.append("Measure whether hallucination rate, correct answer rate, and abstention ")
    lines.append("rates change with position. **Base mode** (no gate, no steering).")
    lines.append("")
    lines.append("**Prior result**: Representation shift confirmed (KNN position classification = 1.0). ")
    lines.append("This experiment tests the next link: does representation shift → behavior shift?")
    lines.append("")

    lines.append("## 2. Results")
    lines.append("")
    lines.append("| Position | H | C | CA | UA |")
    lines.append("|---|---|---|---|---|")
    for d in summary["per_position"]:
        lines.append(f"| {d['position']} | {d['hallucination_rate']:.4f} | {d['correct_answer_rate']:.4f} | "
                     f"{d['calibrated_abstention_rate']:.4f} | {d['unnecessary_abstention_rate']:.4f} |")
    lines.append("")

    lines.append(f"**H range**: {summary['h_range']:.4f}")
    lines.append(f"**C range**: {summary['c_range']:.4f}")
    lines.append("")

    lines.append("## 3. Position-to-Behavior Verdict")
    lines.append("")
    if summary["behavior_position_sensitive"]:
        lines.append("**POSITION IS A BEHAVIORAL CONFOUND**: Same evidence at different positions ")
        lines.append("produces measurably different behavioral outcomes. This confirms that ")
        lines.append("representation shift (Experiment #1a) cascades into behavioral shift.")
        lines.append("")
        lines.append("**Implications**:")
        lines.append("- Trajectory analysis MUST model position as an independent variable")
        lines.append("- Routing/gate decisions dependent on hidden state are position-influenced")
        lines.append("- Position Encoding Constraint Layer is confirmed at the BEHAVIORAL level")
    else:
        lines.append("**POSITION IS NOT A BEHAVIORAL CONFOUND**: While representation shifts ")
        lines.append("with position (KNN=1.0), the behavioral output is robust to this shift ")
        lines.append("at current context lengths (~20-30 tokens).")
        lines.append("")
        lines.append("**Implications**:")
        lines.append("- Representation shift may be compensated by later layers/readout")
        lines.append("- RoPE ceiling is not yet binding at behavioral level at this scale")
    lines.append("")

    lines.append("## 4. Output Breakdown by Position")
    lines.append("")
    for pos_name in ["early", "mid", "late"]:
        pos_df = df[df["position"] == pos_name]
        lines.append(f"### {pos_name}")
        lines.append("")
        lines.append("| Answerability | Count |")
        lines.append("|---|---|")
        for ans_type in ["answerable", "unanswerable"]:
            cnt = int((pos_df["answerability"] == ans_type).sum())
            lines.append(f"| {ans_type} | {cnt} |")
        lines.append("")
        avg_tokens = pos_df["n_tokens"].mean()
        lines.append(f"Avg tokens generated: {avg_tokens:.1f}")
        lines.append("")

    lines.append("## 5. Next Step")
    lines.append("")
    if summary["behavior_position_sensitive"]:
        lines.append("- Position-aware T0/T1/T2/T3 trajectory analysis")
        lines.append("- Position-invariant routing design exploration")
    else:
        lines.append("- Scale context length until behavioral sensitivity emerges")
        lines.append("- Test with longer contexts (more prefix bytes)")
    lines.append("")

    lines.append("---")
    lines.append("*IC-4 Position-to-Behavior Sensitivity — CPU-optimized*")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60, help="Number of samples per position")
    args = parser.parse_args()
    _run_position_behavior(n_samples=args.n)


if __name__ == "__main__":
    main()