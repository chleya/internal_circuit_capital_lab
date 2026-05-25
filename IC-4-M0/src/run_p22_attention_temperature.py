"""
P22: Attention Temperature Scaling — Can softening early-layer attention reduce proximity over-confidence?

P21: LoRA closes 67% of absorption gap (ΔH 0.750→0.250). Mid/late FIXED (H=0.000).
     Early position retains H=0.250 — proximity-based over-confidence survives weight-level intervention.

P19: Attention entropy is HIGHEST at L0 (11.3% gap) and L23 (13.0% gap).
     Early layers show position-dependent attention concentration.

P22 tests: Does ARTIFICIALLY SOFTENING attention at early layers reduce the residual
         early-position hallucination?

Mechanism: Modify `attention_module.scaling = head_dim^-0.5 / temperature`.
           temperature > 1 → softer softmax → more uniform attention → less over-confidence.

Hypotheses:
  H22.1: Temperature > 1 at early layers (L0, L3) reduces early-position H
  H22.2: Temperature effect is layer-specific — early-layer temp helps, deep-layer temp doesn't
  H22.3: Temperature < 1 (sharper attention) INCREASES early-position H (makes over-confidence worse)

Usage:
  cd F:\internal_circuit_capital_lab\IC-4-M0
  python src/run_p22_attention_temperature.py --n 10
"""

import argparse, os, sys, time, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_loader import load_model_and_tokenizer
from src.data_builder import load_jsonl
from src.evaluate import generate_answers, evaluate_outputs

RESULTS_DIR = "results_p22_attention_temperature"
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, "run_log.txt")

def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()

def apply_temperature(model, layers, temperature):
    original_scalings = {}
    for layer_idx in layers:
        attn_module = model.model.layers[layer_idx].self_attn
        original_scalings[layer_idx] = attn_module.scaling
        attn_module.scaling = original_scalings[layer_idx] / temperature
    return original_scalings

def restore_temperature(model, original_scalings):
    for layer_idx, orig_scale in original_scalings.items():
        model.model.layers[layer_idx].self_attn.scaling = orig_scale

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

def run_eval(model, tokenizer, test_samples, label, args):
    t1 = time.time()
    results = generate_answers(model, tokenizer, test_samples,
                               mode="base", max_new_tokens=args.max_new_tokens,
                               temperature=0.0, do_sample=False)
    elapsed = time.time() - t1
    metrics = compute_position_metrics(results)
    delta_h = compute_delta(metrics)

    row = f"  {label:<20s} "
    for pos in ["early", "mid", "late"]:
        m = metrics.get(pos, {})
        row += f"  {pos}: H={m.get('H',0):.3f} C={m.get('C',0):.3f} CA={m.get('CA',0):.3f} |"
    row += f"  ΔH={delta_h:.3f}  [{elapsed:.0f}s]"
    log(row)
    return {"metrics": metrics, "delta_h": delta_h, "time_s": elapsed}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Test samples per position")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--temperatures", type=str, default="0.5,2.0,5.0",
                        help="Attention temperature values (comma-separated). 1.0=baseline.")
    parser.add_argument("--layers", type=str, default="0,3,6,9",
                        help="Layers to apply temperature scaling (comma-separated).")
    args = parser.parse_args()

    temperatures = [float(t.strip()) for t in args.temperatures.split(",")]
    target_layers = [int(l.strip()) for l in args.layers.split(",")]

    log("=" * 64)
    log("P22: Attention Temperature Scaling for Absorption")
    log(f"  Temperatures: {temperatures}")
    log(f"  Target layers: {target_layers}")
    log(f"  n per position: {args.n}, seed: {args.seed}")
    log("=" * 64)
    t0 = time.time()

    log("\n[Step 1] Loading model with eager attention...")
    model, tokenizer = load_model_and_tokenizer(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="cpu", torch_dtype="float32",
        attn_implementation="eager",
    )
    model.eval()
    log(f"  Attention implementation: {model.config._attn_implementation}")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pos_dir = os.path.join(base_dir, "data_position_sensitivity", f"s{args.seed}")

    log("\n[Step 2] Loading test data...")
    test_samples = []
    for pos in ["early", "mid", "late"]:
        test_path = os.path.join(pos_dir, f"test_{pos}_s{args.seed}.jsonl")
        if os.path.exists(test_path):
            for s in load_jsonl(test_path)[:args.n]:
                s["position"] = pos
                test_samples.append(s)
    log(f"  Test: {len(test_samples)} samples")

    log(f"\n[Step 3] Baseline (T=1.0, no modification)...")
    baseline_result = run_eval(model, tokenizer, test_samples, "T=1.0 (baseline)", args)

    all_results = {"T=1.0": baseline_result}

    for T in temperatures:
        if T == 1.0:
            continue
        log(f"\n[Step 4.{T}] Temperature T={T} at layers {target_layers}...")

        original_scalings = apply_temperature(model, target_layers, T)
        log(f"  Applied: scaling = head_dim^-0.5 / {T}")

        label = f"T={T} @ L{target_layers}"
        result = run_eval(model, tokenizer, test_samples, label, args)

        restore_temperature(model, original_scalings)
        log(f"  Restored original scaling.")

        all_results[label] = result

    elapsed = time.time() - t0
    log(f"\n{'='*64}")
    log("[Summary] P22 Attention Temperature Results")
    log(f"  {'─'*20} {'─'*50} {'─'*10}")
    log(f"  {'Condition':<20s} {'H profile (early/mid/late)':<50s} {'ΔH':<10s}")
    log(f"  {'─'*20} {'─'*50} {'─'*10}")

    for label, result in all_results.items():
        m = result["metrics"]
        h_prof = f"H=({m.get('early',{}).get('H',0):.3f}, {m.get('mid',{}).get('H',0):.3f}, {m.get('late',{}).get('H',0):.3f})"
        log(f"  {label:<20s} {h_prof:<50s} {result['delta_h']:<10.3f}")

    base_dh = baseline_result["delta_h"]
    base_early_h = baseline_result["metrics"].get("early", {}).get("H", 0)

    for T in temperatures:
        if T == 1.0:
            continue
        label = f"T={T} @ L{target_layers}"
        if label in all_results:
            result = all_results[label]
            dh_change = result["delta_h"] - base_dh
            early_h = result["metrics"].get("early", {}).get("H", 0)
            early_change = early_h - base_early_h
            log(f"  {label}: ΔΔH={dh_change:+.3f}, ΔH_early={early_change:+.3f}")

    log(f"\nP22 Complete. ({elapsed:.0f}s, {elapsed/60:.1f} min)")

    summary = {
        "experiment": "P22",
        "description": "Attention temperature scaling at early layers to reduce proximity over-confidence",
        "n_per_position": args.n,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "temperatures": temperatures,
        "target_layers": target_layers,
        "results": {}
    }
    for label, result in all_results.items():
        m = result["metrics"]
        summary["results"][label] = {
            "delta_H": round(result["delta_h"], 4),
            "per_position": {p: {"H": round(mm["H"], 4), "C": round(mm["C"], 4),
                                  "CA": round(mm["CA"], 4)}
                           for p, mm in m.items()},
            "time_s": round(result["time_s"], 1),
        }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log(f"\n  Results saved to {RESULTS_DIR}/results.json")
    log("=" * 64)

if __name__ == "__main__":
    main()