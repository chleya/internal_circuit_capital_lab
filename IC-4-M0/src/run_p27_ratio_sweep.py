"""
P27: Ratio Sweep — Mapping the C-Collapse Boundary.

P26 discovered that unified steering ratio is extremely sensitive:
  U_1:1 (hall:syc=1:1) → C dies (avg C=0.222, mid C=0.000)
  U_1:2 (hall:syc=1:2) → C preserved (avg C=0.444, mid C=0.333)

P27 maps the precise boundary: at what ratio does C-preservation emerge?
Tests 4 ratios between 1:1 and 1:3, all energy-matched (total |alpha| = 3.0).

Design:
  Hallucination: 5 conditions (Baseline + 4 ratios), n=10/position = 30 each
  Sycophancy: same 5 conditions, n=16 each

Hypotheses:
  H27.1: C-preservation emerges GRADUALLY from 1:1 to 1:2 (continuous transition)
  H27.2: Beyond 1:2, more syc continues to improve C but worsens H reduction
  H27.3: The optimal ratio for combined H+C performance is between 1:2 and 1:3

Usage:
  cd F:\internal_circuit_capital_lab\IC-4-M0
  python src/run_p27_ratio_sweep.py
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
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_p27_ratio_sweep")
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

def define_conditions(v_hall, v_syc, device):
    """Build condition dict. All ratios energy-matched: total |alpha| = 3.0."""
    TOTAL_E = 3.0

    def make_unified(ratio_h, ratio_s, name):
        total = ratio_h + ratio_s
        a_hall = -TOTAL_E * ratio_h / total
        a_syc = -TOTAL_E * ratio_s / total
        def gen_fn(model, tokenizer, prompt, _device):
            return _generate_unified(model, tokenizer, prompt, v_hall, a_hall, v_syc, a_syc, device)
        label = f"U_1:{ratio_s/ratio_h:.0f}(H={a_hall:.3f}/S={a_syc:.3f})"
        return gen_fn, label, a_hall, a_syc

    conds = {
        "baseline": (
            lambda m, t, p, d: _generate_base(m, t, p, device),
            "Baseline", 0, 0
        ),
    }
    for r_h, r_s in [(1, 1.5), (1, 2), (1, 2.5), (1, 3)]:
        key = f"U_1:{int(r_s/r_h) if r_s/r_h==int(r_s/r_h) else r_s/r_h:.1f}"
        gen_fn, label, a_h, a_s = make_unified(r_h, r_s, key)
        conds[key] = (gen_fn, label, a_h, a_s)

    return conds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_pos", type=int, default=10, help="Hallucination samples per position")
    parser.add_argument("--n_syc", type=int, default=16, help="Sycophancy test samples")
    args = parser.parse_args()

    t0 = time.time()
    log("=" * 64)
    log(f"P27: Ratio Sweep — Mapping C-Collapse Boundary")
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

    conditions = define_conditions(v_hall, v_syc, device)
    cond_keys = list(conditions.keys())
    log(f"\n  Conditions ({len(cond_keys)}):")
    for key in cond_keys:
        _, label, a_h, a_s = conditions[key]
        log(f"    {key:<12}  {label}")

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
    log(f"[Part A] Hallucination ({len(cond_keys)} conditions)")
    log("="*64)

    hall_results = {}
    for key in cond_keys:
        gen_fn, label, a_h, a_s = conditions[key]
        r = run_hallucination_test(model, tokenizer, hall_samples, gen_fn, label, device)
        hall_results[key] = r
        log(f"\n  {label}:")
        log(f"    DH = {r['delta_h']:.3f}")
        avg_c = 0
        for pos in ["early", "mid", "late"]:
            m = r["metrics"][pos]
            avg_c += m["C"]
            log(f"    {pos}: H={m['H']:.3f}  C={m['C']:.3f}  CA={m['CA']:.3f} (n={m['n']})")
        avg_c /= 3
        log(f"    avg C = {avg_c:.3f}")

    log("\n" + "="*64)
    log(f"[Part B] Sycophancy ({len(cond_keys)} conditions)")
    log("="*64)

    syc_results = {}
    for key in cond_keys:
        gen_fn, label, a_h, a_s = conditions[key]
        r = run_sycophancy_test(model, tokenizer, syc_samples, gen_fn, label, device)
        syc_results[key] = r
        log(f"\n  {label}:")
        log(f"    Syc rate = {r['syc_count']}/{r['n']} = {r['syc_rate']:.4f}")

    elapsed = time.time() - t0

    log("\n" + "="*64)
    log("[Summary] P27 Ratio Sweep")
    log("="*64)

    bl_hall = hall_results["baseline"]
    bl_syc = syc_results["baseline"]

    log(f"\n  Hallucination (H/C per position, avg C):")
    header = f"  {'Condition':<28} {'DH':>6} {'E(H/C)':>12} {'M(H/C)':>12} {'L(H/C)':>12} {'avgC':>7}"
    log(header)
    log("  " + "-" * len(header))

    best_combo = None
    best_score = -999

    for key in cond_keys:
        _, label, a_h, a_s = conditions[key]
        r = hall_results[key]
        dh = r["delta_h"]
        avg_c = 0
        parts = []
        for pos in ["early", "mid", "late"]:
            m = r["metrics"][pos]
            parts.append(f"{m['H']:.3f}/{m['C']:.3f}")
            avg_c += m["C"]
        avg_c /= 3
        log(f"  {label:<28} {dh:>6.3f} {parts[0]:>12} {parts[1]:>12} {parts[2]:>12} {avg_c:>7.3f}")

        score = (1 - r["metrics"]["mid"]["H"]) * avg_c
        if key != "baseline" and r["metrics"]["mid"]["H"] < 1.0 and score > best_score:
            best_score = score
            best_combo = key

    log(f"\n  Sycophancy summary:")
    for key in cond_keys:
        _, label, a_h, a_s = conditions[key]
        r = syc_results[key]
        delta = r["syc_rate"] - bl_syc["syc_rate"]
        log(f"  {label:<28}  {r['syc_rate']:.4f} (D={delta:+.4f})")

    if best_combo:
        _, best_label, _, _ = conditions[best_combo]
        log(f"\n  Best combination (max (1-mid_H)*avg_C): {best_label} (score={best_score:.3f})")

    log(f"\n  C-preservation threshold analysis:")
    for key in cond_keys[1:]:
        _, label, a_h, a_s = conditions[key]
        r = hall_results[key]
        mid_c = r["metrics"]["mid"]["C"]
        mid_h = r["metrics"]["mid"]["H"]
        status = "PRESERVED" if mid_c > 0 else "COLLAPSED"
        log(f"    {label:<28}  mid C={mid_c:.3f}  mid H={mid_h:.3f}  → {status}")

    log(f"\nP27 Complete. ({elapsed:.0f}s, {elapsed/60:.1f} min)")

    summary = {
        "experiment": "P27",
        "cos_vhall_vsyc": round(float(cos_hs), 4),
        "n_pos_per": args.n_pos, "n_syc": args.n_syc,
        "conditions": {key: {"label": label, "alpha_hall": round(a_h,4), "alpha_syc": round(a_s,4)}
                       for key, (_, label, a_h, a_s) in conditions.items()},
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
        "time_s": round(elapsed, 1),
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\n  Results saved to results_p27_ratio_sweep/results.json")

if __name__ == "__main__":
    main()