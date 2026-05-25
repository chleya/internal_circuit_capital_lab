"""
P26: Unified Bottleneck Steering.
Can combining v_hall + v_syc achieve superior hallucination reduction (H=0.000
at ALL positions) while preserving correctness?

P25 discovered asymmetric synergy: v_syc(-3) reduces mid hallucination to H=0.000
(matching P15 LoRA), while v_hall(-3) reduces uniformly but kills correctness (C=0).
P26 tests whether COMBINING them yields the best of both worlds.

Design:
  Part A — Hallucination test: 6 conditions
    1. Baseline (no steering)
    2. v_hall only, α=−3.0 (reference)
    3. v_syc only, α=−3.0 (reference — P25: mid H=0.000)
    4. Unified 1:1, α_hall=−1.5, α_syc=−1.5 (energy-matched)
    5. Unified 1:2, α_hall=−1.0, α_syc=−2.0 (75% syc energy)
    6. Unified 2:1, α_hall=−2.0, α_syc=−1.0 (75% hall energy)
  Part B — Sycophancy test: baseline vs all 5 steering conditions

Hypotheses:
  H26.1: Unified steering achieves lower ΔH than either vector alone
  H26.2: Unified steering preserves C better than v_hall alone
  H26.3: Unified 1:2 (more syc) achieves best mid-position H reduction
  H26.4: Unified 2:1 (more hall) achieves best early/late H reduction

Usage:
  cd F:\internal_circuit_capital_lab\IC-4-M0
  python src/run_p26_unified_steering.py
"""

import json, time, sys, os, argparse
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
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_p26_unified_steering")
STEERING_VECTORS_PATH = os.path.join(PROJECT_ROOT, "results_t3_impulse_p4", "steering_vectors.npz")
SYCOPHANCY_DATA_PATH = os.path.join(PROJECT_ROOT, "results_p0_sycophancy", "sycophancy_contrast_data.json")
POS_DIR = os.path.join(PROJECT_ROOT, "data_position_sensitivity", "s0")

PROBE_LAYER = 10
MAX_NEW_TOKENS = 48
RANDOM_SEED = 42

os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, "run_log.txt")

def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()

def _generate_base(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                  pad_token_id=model.config.eos_token_id)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

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

def _generate_unified(model, tokenizer, prompt, v_hall, a_hall, v_syc, a_syc, device):
    v_combined = a_hall * v_hall + a_syc * v_syc
    return _generate_steered(model, tokenizer, prompt, v_combined, 1.0, device)

def run_hallucination_test(model, tokenizer, samples, generate_fn, label, device):
    results = []
    for s in tqdm(samples, desc=label):
        prompt = f"{s.get('context','')}\n\nQuestion: {s.get('question','')}\nAnswer:"
        output = generate_fn(model, tokenizer, prompt, device)
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

def run_sycophancy_test(model, tokenizer, samples, generate_fn, label, device):
    results = []
    for s in tqdm(samples, desc=label):
        prompt = s.get("prompt", "")
        if not prompt:
            prompt = f"{s.get('context','')}\n\nQuestion: {s.get('question','')}\nAnswer:"
        output = generate_fn(model, tokenizer, prompt, device)
        results.append({"output": output, "is_sycophantic": _is_sycophantic(output)})

    syc_count = sum(1 for r in results if r["is_sycophantic"])
    syc_rate = syc_count / len(results)
    return {"syc_rate": syc_rate, "syc_count": syc_count, "n": len(results), "results": results}

def make_generators(v_hall, v_syc, device):
    def unsteered(model, tokenizer, prompt, _device):
        return _generate_base(model, tokenizer, prompt, device)

    def hall_only(model, tokenizer, prompt, _device):
        return _generate_steered(model, tokenizer, prompt, v_hall, -3.0, device)

    def syc_only(model, tokenizer, prompt, _device):
        return _generate_steered(model, tokenizer, prompt, v_syc, -3.0, device)

    def unified_11(model, tokenizer, prompt, _device):
        return _generate_unified(model, tokenizer, prompt, v_hall, -1.5, v_syc, -1.5, device)

    def unified_12(model, tokenizer, prompt, _device):
        return _generate_unified(model, tokenizer, prompt, v_hall, -1.0, v_syc, -2.0, device)

    def unified_21(model, tokenizer, prompt, _device):
        return _generate_unified(model, tokenizer, prompt, v_hall, -2.0, v_syc, -1.0, device)

    return {
        "baseline": (unsteered, "Baseline"),
        "v_hall(-3)": (hall_only, "v_hall(-3.0)"),
        "v_syc(-3)": (syc_only, "v_syc(-3.0)"),
        "U_1:1": (unified_11, "Unified 1:1 (H=-1.5/S=-1.5)"),
        "U_1:2": (unified_12, "Unified 1:2 (H=-1.0/S=-2.0)"),
        "U_2:1": (unified_21, "Unified 2:1 (H=-2.0/S=-1.0)"),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_pos", type=int, default=10, help="Hallucination samples per position")
    parser.add_argument("--n_syc", type=int, default=16, help="Sycophancy test samples")
    args = parser.parse_args()

    t0 = time.time()
    log("=" * 64)
    log(f"P26: Unified Bottleneck Steering — v_hall + v_syc Combination")
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

    cos_hs = np.dot(v_hall, v_syc) / (np.linalg.norm(v_hall) * np.linalg.norm(v_syc))
    log(f"  cos(v_hall, v_syc) = {cos_hs:.4f}")
    log(f"  ||v_hall|| = {np.linalg.norm(v_hall):.4f}, ||v_syc|| = {np.linalg.norm(v_syc):.4f}")

    gens = make_generators(v_hall, v_syc, device)
    conditions = list(gens.keys())

    log("\n[Step 2] Loading hallucination test data...")
    hall_samples = []
    for pos in ["early", "mid", "late"]:
        path = os.path.join(POS_DIR, f"test_{pos}_s0.jsonl")
        if os.path.exists(path):
            for s in load_jsonl(path)[:args.n_pos]:
                s["position"] = pos
                hall_samples.append(s)
    log(f"  Hall test: {len(hall_samples)} samples ({args.n_pos}/position)")

    log("\n[Step 3] Loading sycophancy test data...")
    with open(SYCOPHANCY_DATA_PATH, "r", encoding="utf-8") as f:
        syc_data = json.load(f)
    standard = [s for s in syc_data if not s.get("system_prompt")]
    np.random.seed(RANDOM_SEED)
    idx = np.random.permutation(len(standard))
    syc_samples = [standard[i] for i in idx[:args.n_syc]]
    log(f"  Syc test: {len(syc_samples)} samples")

    log("\n" + "="*64)
    log("[Part A] Hallucination under Unified Steering (6 conditions)")
    log("="*64)

    hall_results = {}
    for key in conditions:
        gen_fn, label = gens[key]
        r = run_hallucination_test(model, tokenizer, hall_samples, gen_fn, label, device)
        hall_results[key] = r
        log(f"\n  {label}:")
        log(f"    ΔH = {r['delta_h']:.3f}")
        for pos in ["early", "mid", "late"]:
            m = r["metrics"][pos]
            log(f"    {pos}: H={m['H']:.3f}  C={m['C']:.3f}  CA={m['CA']:.3f} (n={m['n']})")

    log("\n" + "="*64)
    log("[Part B] Sycophancy under Unified Steering (6 conditions)")
    log("="*64)

    syc_results = {}
    for key in conditions:
        gen_fn, label = gens[key]
        r = run_sycophancy_test(model, tokenizer, syc_samples, gen_fn, label, device)
        syc_results[key] = r
        log(f"\n  {label}:")
        log(f"    Syc rate = {r['syc_count']}/{r['n']} = {r['syc_rate']:.4f}")

    elapsed = time.time() - t0

    log("\n" + "="*64)
    log("[Summary] P26 Unified Bottleneck Steering")
    log("="*64)

    bl_hall = hall_results["baseline"]
    bl_syc = syc_results["baseline"]

    log(f"\n  Hallucination summary (ΔH, with per-position ΔvsBaseline):")
    header = f"  {'Condition':<26} {'ΔH':>6} {'E H(Δ)':>10} {'M H(Δ)':>10} {'L H(Δ)':>10} {'avg C':>7}"
    log(header)
    log("  " + "-" * len(header))

    for key in conditions:
        r = hall_results[key]
        label = gens[key][1]
        dh = r["delta_h"]
        parts = []
        avg_c = 0
        for pos in ["early", "mid", "late"]:
            m = r["metrics"][pos]
            bm = bl_hall["metrics"][pos]
            parts.append(f"{m['H']:.3f}({m['H']-bm['H']:+.3f})")
            avg_c += m["C"]
        avg_c /= 3
        log(f"  {label:<26} {dh:>6.3f} {parts[0]:>10} {parts[1]:>10} {parts[2]:>10} {avg_c:>7.3f}")

    log(f"\n  Sycophancy summary:")
    for key in conditions:
        r = syc_results[key]
        label = gens[key][1]
        delta = r["syc_rate"] - bl_syc["syc_rate"]
        log(f"  {label:<26}  {r['syc_rate']:.4f} (Δ={delta:+.4f})")

    best_hall_key = min(conditions, key=lambda k: hall_results[k]["delta_h"])
    best_hall_label = gens[best_hall_key][1]
    best_delta_h = hall_results[best_hall_key]["delta_h"]

    log(f"\n  Best Hall ΔH: {best_hall_label} — ΔH={best_delta_h:.3f}")
    log(f"  vs v_hall(-3): ΔH={hall_results['v_hall(-3)']['delta_h']:.3f}")
    log(f"  vs v_syc(-3):  ΔH={hall_results['v_syc(-3)']['delta_h']:.3f}")

    h26_1 = best_delta_h < min(hall_results["v_hall(-3)"]["delta_h"],
                                hall_results["v_syc(-3)"]["delta_h"])
    h26_2 = hall_results["U_1:1"]["metrics"]["mid"]["C"] > hall_results["v_hall(-3)"]["metrics"]["mid"]["C"]

    log(f"\n  H26.1 (unified beats both singles): {'CONFIRMED' if h26_1 else 'REFUTED'}")
    log(f"  H26.2 (unified preserves C > v_hall alone): {'CONFIRMED' if h26_2 else 'REFUTED'}")

    log(f"\nP26 Complete. ({elapsed:.0f}s, {elapsed/60:.1f} min)")

    summary = {
        "experiment": "P26",
        "cos_vhall_vsyc": round(float(cos_hs), 4),
        "n_pos_per": args.n_pos, "n_syc": args.n_syc,
        "hallucination": {
            key: {
                "delta_H": round(r["delta_h"], 4),
                "per_position": {p: {"H": round(m["H"],4), "C": round(m["C"],4)}
                               for p, m in r["metrics"].items()}
            }
            for key, r in hall_results.items()
        },
        "sycophancy": {
            key: {"rate": round(r["syc_rate"], 4), "n": r["n"]}
            for key, r in syc_results.items()
        },
        "best_delta_h": round(float(best_delta_h), 4),
        "best_condition": best_hall_key,
        "h26_1_confirmed": h26_1,
        "h26_2_confirmed": h26_2,
        "time_s": round(elapsed, 1),
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\n  Results saved to results_p26_unified_steering/results.json")

if __name__ == "__main__":
    main()