"""
P23: Position ID Offset — Testing RoPE as root cause of early-position over-confidence.

P21: LoRA fixes mid/late (H 1.000→0.000) but early H=0.250 persists.
P22: Attention temperature makes early H WORSE. Baseline T=1.0 is optimal.
     Over-confidence is ROBUST to attention perturbation.

Hypothesis (RoPE): Early-position entity tokens have low position IDs (~100),
  encoded with small RoPE rotation angles → model "feels" entity is more relevant
  → over-confidence on unanswerable questions.

P23 directly tests: if we ARTIFICIALLY shift position IDs by adding padding,
  does early H decrease? If yes → RoPE position encoding is CAUSAL.

Design:
  - For each early-position sample, prefix with N padding tokens
  - This shifts all real token position IDs by +N
  - N ∈ {0 (baseline), 100, 300, 500}
  - Mid/late as control: also shift, expect minimal change

Hypotheses:
  H23.1: Increasing position offset reduces early H (RoPE is causal)
  H23.2: Mid/late H unchanged by position offset (already at high positions)
  H23.3: The reduction saturates — beyond some N, further offset has no effect

Usage:
  cd F:\internal_circuit_capital_lab\IC-4-M0
  python src/run_p23_position_offset.py --n 10
"""

import argparse, os, sys, time, json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_loader import load_model_and_tokenizer
from src.data_builder import load_jsonl
from src.evaluate import evaluate_outputs

RESULTS_DIR = "results_p23_position_offset"
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, "run_log.txt")

def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()

def generate_with_offset(model, tokenizer, sample, offset, max_new_tokens, device):
    context = sample.get("context", "")
    question = sample.get("question", "")
    prompt_text = f"{context}\n\nQuestion: {question}\nAnswer:"

    if offset > 0:
        pad_token = tokenizer.pad_token or tokenizer.eos_token
        pad_str = " ".join([pad_token] * offset)
        prompt_text = f"{pad_str}\n{prompt_text}"

    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  temperature=0.0, do_sample=False,
                                  pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

def run_eval(model, tokenizer, samples, offset, max_new_tokens, device):
    results = []
    for sample in samples:
        output = generate_with_offset(model, tokenizer, sample, offset, max_new_tokens, device)
        results.append({**sample, "generated_output": output, "offset": offset})
    return results

def compute_position_metrics(raw_results):
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
    parser.add_argument("--offsets", type=str, default="0,100,300,500",
                        help="Position ID offsets (comma-separated)")
    args = parser.parse_args()

    offsets = [int(o.strip()) for o in args.offsets.split(",")]

    log("=" * 64)
    log("P23: Position ID Offset — RoPE Root Cause Test")
    log(f"  Offsets: {offsets}")
    log(f"  n per position: {args.n}, seed: {args.seed}")
    log("=" * 64)
    t0 = time.time()

    log("\n[Step 1] Loading model...")
    model, tokenizer = load_model_and_tokenizer(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="cpu", torch_dtype="float32",
    )
    device = next(model.parameters()).device
    model.eval()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pos_dir = os.path.join(base_dir, "data_position_sensitivity", f"s{args.seed}")

    log("\n[Step 2] Loading test data...")
    all_samples = []
    for pos in ["early", "mid", "late"]:
        test_path = os.path.join(pos_dir, f"test_{pos}_s{args.seed}.jsonl")
        if os.path.exists(test_path):
            for s in load_jsonl(test_path)[:args.n]:
                s["position"] = pos
                all_samples.append(s)
    log(f"  Test: {len(all_samples)} samples")

    all_results = {}
    for offset in offsets:
        log(f"\n[Step 3.{offset}] Offset N={offset}...")
        t1 = time.time()
        raw = run_eval(model, tokenizer, all_samples, offset, args.max_new_tokens, device)
        elapsed = time.time() - t1

        metrics = compute_position_metrics(raw)
        delta_h = compute_delta(metrics)
        label = f"N={offset}"
        all_results[label] = {"raw": raw, "metrics": metrics, "delta_h": delta_h, "time_s": elapsed}

        row = f"  {label:<10s} "
        for pos in ["early", "mid", "late"]:
            m = metrics.get(pos, {})
            row += f"  {pos}: H={m.get('H',0):.3f} C={m.get('C',0):.3f} CA={m.get('CA',0):.3f} |"
        row += f"  ΔH={delta_h:.3f}  [{elapsed:.0f}s]"
        log(row)

    baseline = all_results.get("N=0", {})
    base_metrics = baseline.get("metrics", {})
    base_early_h = base_metrics.get("early", {}).get("H", 0)
    base_delta_h = baseline.get("delta_h", 0)

    log(f"\n{'='*64}")
    log("[Summary] P23 Position Offset Results")
    log(f"  {'─'*10} {'─'*40} {'─'*10} {'─'*12}")
    log(f"  {'Offset':<10s} {'H profile (early/mid/late)':<40s} {'ΔH':<10s} {'ΔH_early':<12s}")
    log(f"  {'─'*10} {'─'*40} {'─'*10} {'─'*12}")

    for offset in offsets:
        label = f"N={offset}"
        if label in all_results:
            m = all_results[label]["metrics"]
            dh = all_results[label]["delta_h"]
            early_h = m.get("early", {}).get("H", 0)
            early_change = early_h - base_early_h
            h_prof = f"H=({early_h:.3f}, {m.get('mid',{}).get('H',0):.3f}, {m.get('late',{}).get('H',0):.3f})"
            log(f"  {label:<10s} {h_prof:<40s} {dh:<10.3f} {early_change:+.3f}")

    h23_1 = any(all_results.get(f"N={o}", {}).get("metrics", {}).get("early", {}).get("H", 1) < base_early_h - 0.05 for o in offsets if o > 0)
    log(f"\n  H23.1 (offset reduces early H): {'CONFIRMED' if h23_1 else 'REFUTED'}")

    elapsed = time.time() - t0
    log(f"\nP23 Complete. ({elapsed:.0f}s, {elapsed/60:.1f} min)")

    summary = {
        "experiment": "P23",
        "description": "Position ID offset test for RoPE root cause of early-position over-confidence",
        "n_per_position": args.n, "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "offsets": offsets,
        "baseline_early_H": round(base_early_h, 4),
        "baseline_delta_H": round(base_delta_h, 4),
        "results": {}
    }
    for label, r in all_results.items():
        m = r["metrics"]
        summary["results"][label] = {
            "delta_H": round(r["delta_h"], 4),
            "per_position": {p: {"H": round(mm["H"], 4), "C": round(mm["C"], 4),
                                  "CA": round(mm["CA"], 4)}
                           for p, mm in m.items()},
            "time_s": round(r["time_s"], 1),
        }
    summary["h23_1_confirmed"] = h23_1

    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log(f"\n  Results saved to {RESULTS_DIR}/results.json")
    log("=" * 64)

if __name__ == "__main__":
    main()