"""
P24: Sycophancy n=48 Significance Confirmation.

P8 (n=24): Direction correct but Fisher p > 0.05 for all comparisons.
  Baseline=0.7083, two-stage th=0.50=0.5417 (-23.5%), open-loop=0.5000 (-29.4%).
  "Larger n (48+) may be needed for significance."

P24: Directly run with n=48 to test whether statistical significance emerges
  at larger sample size. 3 conditions: baseline, open-loop (alpha=-3.0),
  two-stage th=0.50.

Usage:
  cd F:\internal_circuit_capital_lab\IC-4-M0
  python src/run_p24_syc_n48.py
"""

import json, pickle, time, sys, os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_loader import load_model_and_tokenizer
from src.steering import _make_steering_hook, _find_transformer_layer
from src.run_p0_sycophancy_contrast import _is_sycophantic

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results_p24_syc_n48"
CONTRAST_DATA_PATH = PROJECT_ROOT / "results_p0_sycophancy" / "sycophancy_contrast_data.json"
P6_PROBE_PATH = PROJECT_ROOT / "results_p6_syc_behavior" / "probe_model.pkl"
STEERING_VECTORS_PATH = PROJECT_ROOT / "results_t3_impulse_p4" / "steering_vectors.npz"

PROBE_LAYER = 10
OPTIMAL_ALPHA = -3.0
MAX_NEW_TOKENS = 48
RANDOM_SEED = 42
N_TEST = 48

LOG_PATH = RESULTS_DIR / "run_log.txt"
os.makedirs(str(RESULTS_DIR), exist_ok=True)

def log(msg):
    print(msg, flush=True)
    with open(str(LOG_PATH), "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()

def _prompt_from_sample(sample):
    prompt = sample.get("prompt", "")
    if not prompt:
        context = sample.get("context", "")
        question = sample.get("question", "")
        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
    return prompt

def _get_probe_score(model, tokenizer, prompt, probe):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    hidden_states = []
    def _hook(module, inputs_tup, output):
        hs = output[0] if isinstance(output, tuple) else output
        hidden_states.append(hs[:, -1, :].detach().cpu().numpy().copy())
    target_module = _find_transformer_layer(model, PROBE_LAYER)
    handle = target_module.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        handle.remove()
    return float(probe.predict_proba(hidden_states[0])[0, 1])

def _generate_with_steering(model, tokenizer, prompt, steering_vector, alpha):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    hook_fn = _make_steering_hook(steering_vector, alpha, device)
    target_module = _find_transformer_layer(model, PROBE_LAYER)
    handle = target_module.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                      pad_token_id=tokenizer.eos_token_id)
    finally:
        handle.remove()
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def _generate_no_steering(model, tokenizer, prompt):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                  pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def run_condition(model, tokenizer, test_samples, probe, steering_vector, alpha,
                  condition_type, threshold=None):
    results = []
    scores = []
    n_gated = 0
    desc = condition_type
    if threshold is not None:
        desc += f" th={threshold}"
    for sample in tqdm(test_samples, desc=desc):
        prompt = _prompt_from_sample(sample)
        if condition_type == "baseline":
            answer = _generate_no_steering(model, tokenizer, prompt)
        elif condition_type == "open_loop":
            answer = _generate_with_steering(model, tokenizer, prompt, steering_vector, alpha)
        elif condition_type == "two_stage":
            score = _get_probe_score(model, tokenizer, prompt, probe)
            scores.append(score)
            if score >= threshold:
                n_gated += 1
                answer = _generate_with_steering(model, tokenizer, prompt, steering_vector, alpha)
            else:
                answer = _generate_no_steering(model, tokenizer, prompt)
        else:
            raise ValueError(f"Unknown: {condition_type}")
        results.append({"output": answer, "is_sycophantic": _is_sycophantic(answer)})

    syc_count = sum(1 for r in results if r["is_sycophantic"])
    syc_rate = syc_count / len(results) if results else 0.0
    return {"syc_rate": syc_rate, "syc_count": syc_count, "n": len(results),
            "results": results, "probe_mean": float(np.mean(scores)) if scores else None,
            "gate_rate": n_gated / len(results) if condition_type == "two_stage" else None}

def fisher_p(r1, r2):
    n1 = sum(1 for r in r1 if r["is_sycophantic"])
    n2 = sum(1 for r in r2 if r["is_sycophantic"])
    table = [[n1, len(r1)-n1], [n2, len(r2)-n2]]
    try:
        _, p = scipy_stats.fisher_exact(table)
        return p
    except:
        return None

def main():
    t0 = time.time()
    log("=" * 64)
    log(f"P24: Sycophancy n={N_TEST} Significance Confirmation")
    log("=" * 64)

    log("\n[Step 1] Loading data...")
    with open(CONTRAST_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    standard = [s for s in data if not s.get("system_prompt")]
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(len(standard))
    test_samples = [standard[i] for i in indices[:N_TEST]]
    log(f"  Standard samples available: {len(standard)}, test: {len(test_samples)}")

    log("\n[Step 2] Loading model, probe, vectors...")
    with open(P6_PROBE_PATH, "rb") as f:
        probe = pickle.load(f)
    model, tokenizer = load_model_and_tokenizer()
    sv = np.load(STEERING_VECTORS_PATH)
    v_syc = sv["v_syc"].astype(np.float32)

    log("\n[Step 3] Baseline...")
    baseline = run_condition(model, tokenizer, test_samples, probe, v_syc, 0, "baseline")
    log(f"  Baseline: {baseline['syc_count']}/{baseline['n']} = {baseline['syc_rate']:.4f}")

    log("\n[Step 4] Open-loop (alpha=-3.0)...")
    open_loop = run_condition(model, tokenizer, test_samples, probe, v_syc, OPTIMAL_ALPHA, "open_loop")
    log(f"  Open-loop: {open_loop['syc_count']}/{open_loop['n']} = {open_loop['syc_rate']:.4f}")

    log("\n[Step 5] Two-stage th=0.50...")
    two_stage = run_condition(model, tokenizer, test_samples, probe, v_syc, OPTIMAL_ALPHA, "two_stage", threshold=0.50)
    log(f"  Two-stage: {two_stage['syc_count']}/{two_stage['n']} = {two_stage['syc_rate']:.4f}, gate={two_stage['gate_rate']:.4f}, probe_mu={two_stage['probe_mean']:.4f}")

    p_bl_vs_ol = fisher_p(baseline["results"], open_loop["results"])
    p_bl_vs_ts = fisher_p(baseline["results"], two_stage["results"])
    p_ol_vs_ts = fisher_p(open_loop["results"], two_stage["results"])

    elapsed = time.time() - t0
    log(f"\n{'='*64}")
    log("[Summary] P24 Sycophancy n=48")
    log(f"  Baseline:     {baseline['syc_count']:2d}/{baseline['n']} = {baseline['syc_rate']:.4f}")
    log(f"  Open-loop:    {open_loop['syc_count']:2d}/{open_loop['n']} = {open_loop['syc_rate']:.4f}")
    log(f"  Two-stage:    {two_stage['syc_count']:2d}/{two_stage['n']} = {two_stage['syc_rate']:.4f}")

    ts_delta = (two_stage["syc_rate"] - baseline["syc_rate"]) / max(baseline["syc_rate"], 0.001) * 100
    ol_delta = (open_loop["syc_rate"] - baseline["syc_rate"]) / max(baseline["syc_rate"], 0.001) * 100
    log(f"\n  Open-loop Δ:  {ol_delta:+.1f}%")
    log(f"  Two-stage Δ:  {ts_delta:+.1f}%")

    log(f"\n  Fisher tests:")
    log(f"    Baseline vs Open-loop:  p={p_bl_vs_ol:.4f} {'***' if p_bl_vs_ol and p_bl_vs_ol < 0.05 else 'n.s.'}")
    log(f"    Baseline vs Two-stage:  p={p_bl_vs_ts:.4f} {'***' if p_bl_vs_ts and p_bl_vs_ts < 0.05 else 'n.s.'}")
    log(f"    Open-loop vs Two-stage: p={p_ol_vs_ts:.4f} {'***' if p_ol_vs_ts and p_ol_vs_ts < 0.05 else 'n.s.'}")

    significant = any(p is not None and p < 0.05 for p in [p_bl_vs_ol, p_bl_vs_ts])

    log(f"\n  → {'Statistically SIGNIFICANT!' if significant else 'NOT significant (p>0.05 for all comparisons)'}")

    log(f"\nP24 Complete. ({elapsed:.0f}s, {elapsed/60:.1f} min)")

    summary = {
        "experiment": "P24",
        "n": N_TEST,
        "baseline": {"syc_rate": round(baseline["syc_rate"], 4), "syc_count": baseline["syc_count"]},
        "open_loop": {"syc_rate": round(open_loop["syc_rate"], 4), "syc_count": open_loop["syc_count"]},
        "two_stage": {"syc_rate": round(two_stage["syc_rate"], 4), "syc_count": two_stage["syc_count"],
                       "gate_rate": round(two_stage["gate_rate"], 4) if two_stage["gate_rate"] else None,
                       "probe_mean": round(two_stage["probe_mean"], 4) if two_stage["probe_mean"] else None},
        "fisher_p_baseline_vs_open_loop": round(p_bl_vs_ol, 6) if p_bl_vs_ol else None,
        "fisher_p_baseline_vs_two_stage": round(p_bl_vs_ts, 6) if p_bl_vs_ts else None,
        "fisher_p_open_loop_vs_two_stage": round(p_ol_vs_ts, 6) if p_ol_vs_ts else None,
        "significant": significant,
        "time_s": round(elapsed, 1),
    }
    with open(str(RESULTS_DIR / "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\n  Results saved to results_p24_syc_n48/results.json")
    log("=" * 64)

if __name__ == "__main__":
    main()