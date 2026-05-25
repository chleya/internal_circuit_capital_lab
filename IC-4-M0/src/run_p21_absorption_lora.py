"""
P21: Absorption LoRA — Does weight-level intervention close the absorption position gap?

P12: L10 v_abs steering → homogenization with degradation (H_early +100%).
P20: Multi-layer steering → LAYER-INDEPENDENT. Hidden-state vector interventions EXHAUSTED.
     "Only path forward: weight-level (LoRA) or attention-direct modification.
     B-bottleneck: LoRA works (P15) → A-bottleneck: LoRA untested."

P21 directly closes the loop:
  - P15 trained hallucination-targeted LoRA → H=0.000, C=1.000 in LOG-PROB space
  - P21 evaluates the SAME LoRA checkpoint using generate() to measure absorption
  - Does LoRA close the position gap in generate() space?

Hypotheses:
  H21.1: P15 LoRA reduces ΔH_pos (position gap in generate space) by >= 0.10
  H21.2: P15 LoRA preserves or improves H (hallucination rate) in generate space
  H21.3: Absorption gap closure in generate space is PARALLEL to B-bottleneck closure in log-prob space

Usage:
  cd F:\internal_circuit_capital_lab\IC-4-M0
  python src/run_p21_absorption_lora.py --n 10
"""

import argparse, os, sys, time, json
import numpy as np
import torch
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_loader import load_model_and_tokenizer
from src.data_builder import load_jsonl
from src.evaluate import generate_answers, evaluate_outputs

RESULTS_DIR = "results_p21_absorption_lora"
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, "run_log.txt")

def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()

def logprob_of_response(model, tokenizer, prompt, response, device):
    full_text = f"{prompt} {response}"
    full_ids = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=256)
    full_ids = {k: v.to(device) for k, v in full_ids.items()}
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    prompt_len = prompt_ids["input_ids"].shape[1]
    labels = full_ids["input_ids"].clone()
    labels[0, :prompt_len] = -100
    with torch.no_grad():
        outputs = model(**full_ids, labels=labels)
    return -outputs.loss.item()

def evaluate_behavior_logprob(model, tokenizer, samples, device):
    results = []
    for sample in samples:
        context = sample.get("context", "")
        question = sample.get("question", "")
        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
        pos_resp = sample.get("positive_response", "")
        neg_resp = sample.get("negative_response", "")
        pos_lp = logprob_of_response(model, tokenizer, prompt, pos_resp, device)
        neg_lp = logprob_of_response(model, tokenizer, prompt, neg_resp, device)
        pref_positive = pos_lp > neg_lp
        results.append({
            "answerability": sample.get("answerability", "?"),
            "pref_positive": pref_positive,
            "pos_logprob": pos_lp,
            "neg_logprob": neg_lp,
        })
    return results

def compute_lp_metrics(lp_results):
    answerable = [r for r in lp_results if r.get("answerability") == "answerable"]
    unanswerable = [r for r in lp_results if r.get("answerability") == "unanswerable"]
    n_ans = len(answerable)
    n_unans = len(unanswerable)
    hallucinations = sum(1 for r in unanswerable if r["pref_positive"])
    correct = sum(1 for r in answerable if r["pref_positive"])
    H = hallucinations / n_unans if n_unans > 0 else 0.0
    C = correct / n_ans if n_ans > 0 else 0.0
    return {"H": round(H, 4), "C": round(C, 4), "hall_count": hallucinations,
            "unans_count": n_unans, "corr_count": correct, "ans_count": n_ans}

def compute_position_metrics_generate(raw_results):
    metrics = {}
    positions = sorted(set(r.get("position", "?") for r in raw_results))
    for pos in positions:
        pos_results = [r for r in raw_results if r.get("position") == pos]
        eval_metrics = evaluate_outputs(pos_results)
        metrics[pos] = {
            "H": eval_metrics.get("hallucination_rate", 0),
            "C": eval_metrics.get("correct_answer_rate", 0),
            "CA": eval_metrics.get("calibrated_abstention_rate", 0),
            "total": len(pos_results),
        }
    return metrics

def compute_delta(metrics):
    positions = sorted(metrics.keys())
    h_vals = [metrics[p]["H"] for p in positions]
    return max(h_vals) - min(h_vals)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Test samples per position")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to LoRA checkpoint. Default: results_p15_hallucination_lora/checkpoint_final")
    args = parser.parse_args()

    log("=" * 64)
    log("P21: Absorption LoRA — Evaluating P15 LoRA Checkpoint for Position Gap")
    log(f"  n per position: {args.n}, seed: {args.seed}")
    log(f"  max_new_tokens: {args.max_new_tokens}")
    log("=" * 64)
    t0 = time.time()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chkpt_path = args.checkpoint or os.path.join(base_dir, "results_p15_hallucination_lora", "checkpoint_final")
    log(f"\nLoRA checkpoint: {chkpt_path}")
    log(f"  Exists: {os.path.isdir(chkpt_path)}")

    log("\n[Step 1] Loading base model...")
    model, tokenizer = load_model_and_tokenizer(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="cpu", torch_dtype="float32",
    )
    device = next(model.parameters()).device
    model.eval()

    pos_dir = os.path.join(base_dir, "data_position_sensitivity", f"s{args.seed}")

    log("\n[Step 2] Loading test data...")
    test_samples = []
    for pos in ["early", "mid", "late"]:
        test_path = os.path.join(pos_dir, f"test_{pos}_s{args.seed}.jsonl")
        if os.path.exists(test_path):
            for s in load_jsonl(test_path)[:args.n]:
                s["position"] = pos
                test_samples.append(s)

    n_ans = sum(1 for s in test_samples if s.get("answerability") == "answerable")
    n_unans = sum(1 for s in test_samples if s.get("answerability") != "answerable")
    log(f"  Test: {len(test_samples)} samples ({n_ans}A + {n_unans}U)")

    log("\n[Step 3] PRE-LoRA evaluation (generate space)...")
    log("  Running generate() for base model...")
    t_gen = time.time()
    pre_gen_results = generate_answers(model, tokenizer, test_samples,
                                        mode="base", max_new_tokens=args.max_new_tokens,
                                        temperature=0.0, do_sample=False)
    pre_gen_elapsed = time.time() - t_gen

    pre_gen_metrics = compute_position_metrics_generate(pre_gen_results)
    pre_delta_h = compute_delta(pre_gen_metrics)

    log(f"  Pre-LoRA generate (H per position, ΔH):")
    row = f"    "
    for pos in ["early", "mid", "late"]:
        m = pre_gen_metrics.get(pos, {})
        row += f"  {pos}: H={m.get('H',0):.3f} C={m.get('C',0):.3f} CA={m.get('CA',0):.3f} |"
    row += f"  ΔH={pre_delta_h:.3f}"
    log(row)
    log(f"  Generate time: {pre_gen_elapsed:.0f}s")

    log("\n[Step 4] PRE-LoRA evaluation (log-prob space, P15 metric)...")
    pre_lp_results = evaluate_behavior_logprob(model, tokenizer, test_samples, device)
    pre_lp_metrics = compute_lp_metrics(pre_lp_results)

    pre_lp_pos = {}
    for pos in ["early", "mid", "late"]:
        pos_results = [r for r, s in zip(pre_lp_results, test_samples) if s.get("position") == pos]
        pre_lp_pos[pos] = compute_lp_metrics(pos_results)

    h_vals_pre = [pre_lp_pos[p]["H"] for p in ["early","mid","late"]]
    pre_lp_delta_h = max(h_vals_pre) - min(h_vals_pre)

    log(f"  Pre-LoRA log-prob: H={pre_lp_metrics['H']:.4f} C={pre_lp_metrics['C']:.4f} ΔH={pre_lp_delta_h:.4f}")
    for pos in ["early", "mid", "late"]:
        m = pre_lp_pos[pos]
        log(f"    {pos}: H={m['H']:.4f} C={m['C']:.4f} (hall={m['hall_count']}/{m['unans_count']})")

    log("\n[Step 5] Loading P15 LoRA checkpoint...")
    model2, tokenizer2 = load_model_and_tokenizer(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="cpu", torch_dtype="float32",
    )
    model2 = PeftModel.from_pretrained(model2, chkpt_path)
    model2.eval()
    device2 = next(model2.parameters()).device
    log(f"  LoRA loaded. Device: {device2}")

    log("\n[Step 6] POST-LoRA evaluation (generate space)...")
    log("  Running generate() for LoRA model...")
    t_gen2 = time.time()
    post_gen_results = generate_answers(model2, tokenizer2, test_samples,
                                         mode="base", max_new_tokens=args.max_new_tokens,
                                         temperature=0.0, do_sample=False)
    post_gen_elapsed = time.time() - t_gen2

    post_gen_metrics = compute_position_metrics_generate(post_gen_results)
    post_delta_h = compute_delta(post_gen_metrics)

    log(f"  Post-LoRA generate (H per position, ΔH):")
    row = f"    "
    for pos in ["early", "mid", "late"]:
        m = post_gen_metrics.get(pos, {})
        row += f"  {pos}: H={m.get('H',0):.3f} C={m.get('C',0):.3f} CA={m.get('CA',0):.3f} |"
    row += f"  ΔH={post_delta_h:.3f}"
    log(row)
    log(f"  Generate time: {post_gen_elapsed:.0f}s")

    log("\n[Step 7] POST-LoRA evaluation (log-prob space)...")
    post_lp_results = evaluate_behavior_logprob(model2, tokenizer2, test_samples, device2)
    post_lp_metrics = compute_lp_metrics(post_lp_results)

    post_lp_pos = {}
    for pos in ["early", "mid", "late"]:
        pos_results = [r for r, s in zip(post_lp_results, test_samples) if s.get("position") == pos]
        post_lp_pos[pos] = compute_lp_metrics(pos_results)

    h_vals_post = [post_lp_pos[p]["H"] for p in ["early","mid","late"]]
    post_lp_delta_h = max(h_vals_post) - min(h_vals_post)

    log(f"  Post-LoRA log-prob: H={post_lp_metrics['H']:.4f} C={post_lp_metrics['C']:.4f} ΔH={post_lp_delta_h:.4f}")
    for pos in ["early", "mid", "late"]:
        m = post_lp_pos[pos]
        log(f"    {pos}: H={m['H']:.4f} C={m['C']:.4f} (hall={m['hall_count']}/{m['unans_count']})")

    elapsed = time.time() - t0
    log(f"\n{'='*64}")
    log("[Summary] P21 Absorption LoRA Results")

    h21_1 = post_delta_h < pre_delta_h - 0.05
    h21_2 = post_gen_metrics.get("mid", {}).get("H", 1) < pre_gen_metrics.get("mid", {}).get("H", 1)

    log(f"  H21.1 (ΔH reduce by >= 0.10): {'CONFIRMED' if h21_1 else 'REFUTED'}")
    log(f"    Pre ΔH={pre_delta_h:.3f} → Post ΔH={post_delta_h:.3f} (Δ={post_delta_h-pre_delta_h:+.3f})")
    log(f"  H21.2 (H improved in generate space): {'CONFIRMED' if h21_2 else 'REFUTED'}")
    log(f"  H21.3 (generate-space gap parallel to log-prob): TBD")

    log(f"\n  Key comparison:")
    log(f"  ─────────── ──────────────── ────────────────")
    log(f"  Space       Pre-LoRA          Post-LoRA        ")
    log(f"  ─────────── ──────────────── ────────────────")
    log(f"  log-prob    H={pre_lp_metrics['H']:.4f} C={pre_lp_metrics['C']:.4f}  ΔH={pre_lp_delta_h:.4f}   "
        f"H={post_lp_metrics['H']:.4f} C={post_lp_metrics['C']:.4f}  ΔH={post_lp_delta_h:.4f}")
    log(f"  generate    ΔH={pre_delta_h:.3f}            "
        f"ΔH={post_delta_h:.3f}            ")

    log(f"\nP21 Complete. ({elapsed:.0f}s, {elapsed/60:.1f} min)")

    summary = {
        "experiment": "P21",
        "description": "Evaluate P15 LoRA checkpoint for absorption position gap closure (generate space)",
        "n_per_position": args.n,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "checkpoint": chkpt_path,
        "pre_generate": {
            "delta_H": round(pre_delta_h, 4),
            "per_position": {p: {"H": round(m["H"], 4), "C": round(m["C"], 4),
                                  "CA": round(m["CA"], 4)}
                           for p, m in pre_gen_metrics.items()},
        },
        "post_generate": {
            "delta_H": round(post_delta_h, 4),
            "per_position": {p: {"H": round(m["H"], 4), "C": round(m["C"], 4),
                                  "CA": round(m["CA"], 4)}
                           for p, m in post_gen_metrics.items()},
        },
        "pre_logprob": {
            "H": pre_lp_metrics["H"], "C": pre_lp_metrics["C"],
            "delta_H": round(pre_lp_delta_h, 4),
            "per_position": {p: {"H": m["H"], "C": m["C"]} for p, m in pre_lp_pos.items()},
        },
        "post_logprob": {
            "H": post_lp_metrics["H"], "C": post_lp_metrics["C"],
            "delta_H": round(post_lp_delta_h, 4),
            "per_position": {p: {"H": m["H"], "C": m["C"]} for p, m in post_lp_pos.items()},
        },
        "h21_1_confirmed": h21_1,
        "h21_2_confirmed": h21_2,
        "generate_time_s": round(pre_gen_elapsed + post_gen_elapsed, 1),
        "time_s": round(elapsed, 1),
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log(f"\n  Results saved to {RESULTS_DIR}/results.json")
    log("=" * 64)

if __name__ == "__main__":
    main()