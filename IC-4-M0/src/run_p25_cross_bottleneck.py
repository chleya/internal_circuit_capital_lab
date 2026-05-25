"""
P25: Cross-Bottleneck Steering Interaction.
Do hallucination and sycophancy steering vectors compete or are they independent?

Key question: Does anti-sycophancy steering (v_syc) INCREASE hallucination?
              Does anti-hallucination steering (v_hall) INCREASE sycophancy?

If the bottlenecks share representational capacity, fighting one may worsen the other.
If they're independent, cross-steering should have ~zero effect.

Design:
  Part A — Hallucination test: baseline vs v_syc(-3.0) vs v_hall(-3.0)
    → measure H (hallucination rate) per position on position_sensitivity data
  Part B — Sycophancy test: baseline vs v_syc(-3.0) vs v_hall(-3.0)
    → measure sycophancy rate on sycophancy contrast data

Hypotheses:
  H25.1: v_syc steering does NOT increase hallucination (bottlenecks independent)
  H25.2: v_hall steering does NOT increase sycophancy (bottlenecks independent)
  H25.3: v_hall steering reduces hallucination more than v_syc (domain specificity)

Usage:
  cd F:\internal_circuit_capital_lab\IC-4-M0
  python src/run_p25_cross_bottleneck.py
"""

import json, pickle, time, sys, os, argparse
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_loader import load_model_and_tokenizer
from src.data_builder import load_jsonl
from src.steering import _make_steering_hook, _find_transformer_layer
from src.evaluate import evaluate_outputs
from src.run_p0_sycophancy_contrast import _is_sycophantic

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_p25_cross_bottleneck")
STEERING_VECTORS_PATH = os.path.join(PROJECT_ROOT, "results_t3_impulse_p4", "steering_vectors.npz")
SYCOPHANCY_DATA_PATH = os.path.join(PROJECT_ROOT, "results_p0_sycophancy", "sycophancy_contrast_data.json")
POS_DIR = os.path.join(PROJECT_ROOT, "data_position_sensitivity", "s0")

PROBE_LAYER = 10
ALPHA = -3.0
MAX_NEW_TOKENS = 48
RANDOM_SEED = 42

os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, "run_log.txt")

def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()

def _generate_steered(model, tokenizer, prompt, vec, alpha, device):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    hook_fn = _make_steering_hook(vec, alpha, device)
    target = _find_transformer_layer(model, PROBE_LAYER)
    handle = target.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                      pad_token_id=model.config.eos_token_id)
    finally:
        handle.remove()
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def _generate_unsteered(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                  pad_token_id=model.config.eos_token_id)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def run_hallucination_test(model, tokenizer, samples, vec, alpha, label, device):
    results = []
    for s in tqdm(samples, desc=label):
        prompt = f"{s.get('context','')}\n\nQuestion: {s.get('question','')}\nAnswer:"
        if alpha == 0:
            output = _generate_unsteered(model, tokenizer, prompt, device)
        else:
            output = _generate_steered(model, tokenizer, prompt, vec, alpha, device)
        results.append({**s, "generated_output": output, "condition": label})

    metrics_per_pos = {}
    for pos in ["early", "mid", "late"]:
        pos_results = [r for r in results if r.get("position") == pos]
        m = evaluate_outputs(pos_results)
        metrics_per_pos[pos] = {
            "H": m.get("hallucination_rate", 0), "C": m.get("correct_answer_rate", 0),
            "CA": m.get("calibrated_abstention_rate", 0), "n": len(pos_results),
        }
    h_vals = [metrics_per_pos[p]["H"] for p in ["early", "mid", "late"]]
    delta_h = max(h_vals) - min(h_vals)
    return {"metrics": metrics_per_pos, "delta_h": delta_h, "results": results}

def run_sycophancy_test(model, tokenizer, samples, vec, alpha, label, device):
    results = []
    for s in tqdm(samples, desc=label):
        prompt = s.get("prompt", "")
        if not prompt:
            prompt = f"{s.get('context','')}\n\nQuestion: {s.get('question','')}\nAnswer:"
        if alpha == 0:
            output = _generate_unsteered(model, tokenizer, prompt, device)
        else:
            output = _generate_steered(model, tokenizer, prompt, vec, alpha, device)
        results.append({"output": output, "is_sycophantic": _is_sycophantic(output)})

    syc_count = sum(1 for r in results if r["is_sycophantic"])
    syc_rate = syc_count / len(results)
    return {"syc_rate": syc_rate, "syc_count": syc_count, "n": len(results), "results": results}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_pos", type=int, default=8, help="Hallucination samples per position")
    parser.add_argument("--n_syc", type=int, default=16, help="Sycophancy test samples")
    args = parser.parse_args()

    t0 = time.time()
    log("=" * 64)
    log(f"P25: Cross-Bottleneck Steering Interaction")
    log(f"  n_pos={args.n_pos}, n_syc={args.n_syc}")
    log("=" * 64)

    log("\n[Step 1] Loading model and vectors...")
    model, tokenizer = load_model_and_tokenizer(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="cpu", torch_dtype="float32",
    )
    device = next(model.parameters()).device
    sv = np.load(STEERING_VECTORS_PATH)
    v_hall = sv["v_hall"].astype(np.float32)
    v_syc = sv["v_syc"].astype(np.float32)

    ds_hall = np.dot(v_hall, v_syc) / (np.linalg.norm(v_hall) * np.linalg.norm(v_syc))
    log(f"  cos(v_hall, v_syc) = {ds_hall:.4f}")

    log("\n[Step 2] Loading hallucination test data...")
    hall_samples = []
    for pos in ["early", "mid", "late"]:
        path = os.path.join(POS_DIR, f"test_{pos}_s0.jsonl")
        if os.path.exists(path):
            for s in load_jsonl(path)[:args.n_pos]:
                s["position"] = pos
                hall_samples.append(s)
    log(f"  Hall test: {len(hall_samples)} samples")

    log("\n[Step 3] Loading sycophancy test data...")
    with open(SYCOPHANCY_DATA_PATH, "r", encoding="utf-8") as f:
        syc_data = json.load(f)
    standard = [s for s in syc_data if not s.get("system_prompt")]
    np.random.seed(RANDOM_SEED)
    idx = np.random.permutation(len(standard))
    syc_samples = [standard[i] for i in idx[:args.n_syc]]
    log(f"  Syc test: {len(syc_samples)} samples")

    log("\n" + "="*64)
    log("[Part A] Hallucination under cross-steering")
    log("="*64)

    hall_baseline = run_hallucination_test(model, tokenizer, hall_samples, v_hall, 0, "Hall_baseline", device)
    log(f"  Baseline: ΔH={hall_baseline['delta_h']:.3f}")
    for pos in ["early","mid","late"]:
        m = hall_baseline["metrics"][pos]
        log(f"    {pos}: H={m['H']:.3f} C={m['C']:.3f} CA={m['CA']:.3f}")

    hall_vsyc = run_hallucination_test(model, tokenizer, hall_samples, v_syc, ALPHA, "Hall_vSyc(-3)", device)
    log(f"  v_syc(-3): ΔH={hall_vsyc['delta_h']:.3f}")
    for pos in ["early","mid","late"]:
        m = hall_vsyc["metrics"][pos]
        b = hall_baseline["metrics"][pos]
        log(f"    {pos}: H={m['H']:.3f} (Δ={m['H']-b['H']:+.3f}) C={m['C']:.3f}")

    hall_vhall = run_hallucination_test(model, tokenizer, hall_samples, v_hall, ALPHA, "Hall_vHall(-3)", device)
    log(f"  v_hall(-3): ΔH={hall_vhall['delta_h']:.3f}")
    for pos in ["early","mid","late"]:
        m = hall_vhall["metrics"][pos]
        b = hall_baseline["metrics"][pos]
        log(f"    {pos}: H={m['H']:.3f} (Δ={m['H']-b['H']:+.3f}) C={m['C']:.3f}")

    log("\n" + "="*64)
    log("[Part B] Sycophancy under cross-steering")
    log("="*64)

    syc_baseline = run_sycophancy_test(model, tokenizer, syc_samples, v_syc, 0, "Syc_baseline", device)
    log(f"  Baseline: {syc_baseline['syc_count']}/{syc_baseline['n']} = {syc_baseline['syc_rate']:.4f}")

    syc_vhall = run_sycophancy_test(model, tokenizer, syc_samples, v_hall, ALPHA, "Syc_vHall(-3)", device)
    log(f"  v_hall(-3): {syc_vhall['syc_count']}/{syc_vhall['n']} = {syc_vhall['syc_rate']:.4f} (Δ={syc_vhall['syc_rate']-syc_baseline['syc_rate']:+.4f})")

    syc_vsyc = run_sycophancy_test(model, tokenizer, syc_samples, v_syc, ALPHA, "Syc_vSyc(-3)", device)
    log(f"  v_syc(-3): {syc_vsyc['syc_count']}/{syc_vsyc['n']} = {syc_vsyc['syc_rate']:.4f} (Δ={syc_vsyc['syc_rate']-syc_baseline['syc_rate']:+.4f})")

    elapsed = time.time() - t0
    log(f"\n{'='*64}")
    log("[Summary] P25 Cross-Bottleneck Interaction")
    log(f"  v_hall · v_syc cos = {ds_hall:.4f}")

    hall_h_vsyc = hall_vsyc["delta_h"] - hall_baseline["delta_h"]
    hall_h_vhall = hall_vhall["delta_h"] - hall_baseline["delta_h"]
    syc_vhall_delta = syc_vhall["syc_rate"] - syc_baseline["syc_rate"]
    syc_vsyc_delta = syc_vsyc["syc_rate"] - syc_baseline["syc_rate"]

    log(f"\n  Hallucination:")
    log(f"    Baseline ΔH:       {hall_baseline['delta_h']:.3f}")
    log(f"    v_syc(-3):   ΔH=   {hall_vsyc['delta_h']:.3f} (ΔΔH={hall_h_vsyc:+.3f})")
    log(f"    v_hall(-3):  ΔH=   {hall_vhall['delta_h']:.3f} (ΔΔH={hall_h_vhall:+.3f})")

    log(f"\n  Sycophancy:")
    log(f"    Baseline rate:     {syc_baseline['syc_rate']:.4f}")
    log(f"    v_hall(-3):  rate= {syc_vhall['syc_rate']:.4f} (Δ={syc_vhall_delta:+.4f})")
    log(f"    v_syc(-3):   rate= {syc_vsyc['syc_rate']:.4f} (Δ={syc_vsyc_delta:+.4f})")

    h25_1 = abs(hall_h_vsyc) < 0.10
    h25_2 = abs(syc_vhall_delta) < 0.10

    log(f"\n  H25.1 (v_syc DOES NOT increase hallucination): {'CONFIRMED' if h25_1 else 'REFUTED'}")
    log(f"  H25.2 (v_hall DOES NOT increase sycophancy): {'CONFIRMED' if h25_2 else 'REFUTED'}")

    if h25_1 and h25_2:
        log(f"  → Hallucination and Sycophancy bottlenecks are INDEPENDENT.")
    elif h25_1 and not h25_2:
        log(f"  → v_hall increases sycophancy — THERE IS A TRADE-OFF.")
    elif not h25_1 and h25_2:
        log(f"  → v_syc increases hallucination — THERE IS A TRADE-OFF.")
    else:
        log(f"  → BOTH cross-steerings increase the OTHER bottleneck — STRONG TRADE-OFF.")

    log(f"\nP25 Complete. ({elapsed:.0f}s, {elapsed/60:.1f} min)")

    summary = {
        "experiment": "P25",
        "cos_vhall_vsyc": round(float(ds_hall), 4),
        "n_pos_per": args.n_pos, "n_syc": args.n_syc,
        "hallucination": {
            "baseline": {"delta_H": round(hall_baseline["delta_h"], 4),
                         "per_position": {p: {"H": round(m["H"],4), "C": round(m["C"],4)}
                                        for p, m in hall_baseline["metrics"].items()}},
            "v_syc": {"delta_H": round(hall_vsyc["delta_h"], 4),
                      "per_position": {p: {"H": round(m["H"],4), "C": round(m["C"],4)}
                                     for p, m in hall_vsyc["metrics"].items()}},
            "v_hall": {"delta_H": round(hall_vhall["delta_h"], 4),
                       "per_position": {p: {"H": round(m["H"],4), "C": round(m["C"],4)}
                                      for p, m in hall_vhall["metrics"].items()}},
        },
        "sycophancy": {
            "baseline": {"rate": round(syc_baseline["syc_rate"], 4), "n": syc_baseline["n"]},
            "v_hall": {"rate": round(syc_vhall["syc_rate"], 4), "n": syc_vhall["n"]},
            "v_syc": {"rate": round(syc_vsyc["syc_rate"], 4), "n": syc_vsyc["n"]},
        },
        "h25_1_confirmed": h25_1, "h25_2_confirmed": h25_2,
        "time_s": round(elapsed, 1),
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\n  Results saved to results_p25_cross_bottleneck/results.json")

if __name__ == "__main__":
    main()