"""
P12: Position-Directional Open-Loop Steering for Absorption Remedy.
===================================================================
Tests whether activation steering with a contrastive position-direction
vector can reduce behavior-level position sensitivity.

Background:
  A1: Probe-level PSI -90% (FIXED at probe level)
  A2: Behavior gap persists (h_range=0.111 despite probe fix)
  A3: Global per-position rectification FAILED (delta_h=1.0)
  A4: LoRA training mixed (delta_h -50% but H_early +67%)

P12 follows the proven sycophancy steering pattern:
  1. Compute v_abs = mean(h_early) - mean(h_late) from training data
     (points FROM late-pos degradation TOWARD early-pos information richness)
  2. Open-loop steering during model.generate() at target layer
  3. alpha-sweep to find optimal compensation magnitude
  4. Compare: baseline vs best alpha vs random control

Hypotheses:
  H12.1: Positive alpha (add v_abs) reduces H_late (late-position hallucination)
  H12.2: Positive alpha reduces delta_H = |H_late - H_early|
  H12.3: Random vector steering does NOT produce the same effect
  H12.4: There exists an optimal alpha with minimal delta_H

Usage:
  cd F:/internal_circuit_capital_lab/IC-4-M0
  python src/run_p12_absorption_steering.py --n 10 --layer 10 --alphas "-3,-2,-1,0,1,2,3"
"""

import argparse, os, sys, time, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_loader import load_model_and_tokenizer
from src.data_builder import load_jsonl
from src.evaluate import evaluate_outputs

RESULTS_DIR = "results_p12_absorption_steering"
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, "run_log.txt")


def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()


def collect_position_hidden_states(model, tokenizer, samples, layer_idx, device):
    X_list, pos_labels = [], []
    target_module = None
    for name, module in model.named_modules():
        if name.endswith(f"model.layers.{layer_idx}"):
            target_module = module
            break

    captured = []

    def _capture(module, inputs_tup, output):
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
        captured.append(hs[0, -1, :].detach().cpu().numpy().copy())

    handle = target_module.register_forward_hook(_capture)

    for sample in samples:
        context = sample.get("context", "")
        question = sample.get("question", "")
        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        captured.clear()
        with torch.no_grad():
            model(**inputs)
        if captured:
            X_list.append(captured[0])
            pos_labels.append(sample.get("position", sample.get("_position", "unknown")))

    handle.remove()
    return np.array(X_list), pos_labels


def compute_steering_vector(model, tokenizer, pos_dir, seed, layer, device):
    train_early = load_jsonl(os.path.join(pos_dir, f"train_early_s{seed}.jsonl"))
    train_late = load_jsonl(os.path.join(pos_dir, f"train_late_s{seed}.jsonl"))

    X_early, _ = collect_position_hidden_states(model, tokenizer, train_early, layer, device)
    X_late, _ = collect_position_hidden_states(model, tokenizer, train_late, layer, device)

    v_abs = X_early.mean(axis=0) - X_late.mean(axis=0)
    v_abs = v_abs / (np.linalg.norm(v_abs) + 1e-8)

    v_random = np.random.randn(*v_abs.shape).astype(np.float32)
    v_random = v_random / (np.linalg.norm(v_random) + 1e-8)
    v_orthogonal = v_random - np.dot(v_random, v_abs) * v_abs
    v_orthogonal = v_orthogonal / (np.linalg.norm(v_orthogonal) + 1e-8)

    log(f"  v_abs norm: {np.linalg.norm(v_abs):.4f}")
    log(f"  ||h_early|| mean: {np.linalg.norm(X_early, axis=1).mean():.2f}")
    log(f"  ||h_late|| mean:  {np.linalg.norm(X_late, axis=1).mean():.2f}")
    log(f"  ||h_early - h_late|| mean: {np.linalg.norm(X_early - X_late, axis=1).mean():.2f}")

    return v_abs, v_random, v_orthogonal


def make_steering_hook(steering_vector, alpha, device, dtype):
    sv = torch.tensor(steering_vector, dtype=dtype).to(device)

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
        hs_steered = hs + alpha * sv.view(1, 1, -1)
        if isinstance(output, tuple):
            return (hs_steered,) + output[1:]
        return hs_steered

    return hook_fn


def generate_with_steering(model, tokenizer, sample, layer_idx, steering_vector,
                           alpha, max_new_tokens, device):
    context = sample.get("context", "")
    question = sample.get("question", "")
    prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    target_module = None
    for name, module in model.named_modules():
        if name.endswith(f"model.layers.{layer_idx}"):
            target_module = module
            break

    model_dtype = next(model.parameters()).dtype

    handle = None
    if alpha != 0.0 and steering_vector is not None:
        handle = target_module.register_forward_hook(
            make_steering_hook(steering_vector, alpha, device, model_dtype)
        )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    if handle:
        handle.remove()

    generated_ids = outputs[0][input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def run_position_test(model, tokenizer, test_samples, layer_idx, steering_vector,
                      alpha, max_new_tokens, device):
    results = []
    for i, sample in enumerate(test_samples):
        output = generate_with_steering(
            model, tokenizer, sample, layer_idx, steering_vector,
            alpha, max_new_tokens, device
        )
        results.append({
            "sample_id": i,
            "position": sample.get("position", sample.get("_position", "unknown")),
            "answerability": sample.get("answerability", "?"),
            "gold_answer": sample.get("gold_answer", ""),
            "generated_output": output,
        })
    return results


def compute_position_metrics(raw_results):
    metrics = {}
    positions = sorted(set(r["position"] for r in raw_results))
    for pos in positions:
        pos_results = [r for r in raw_results if r["position"] == pos]
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
    if len(positions) < 2:
        return 0.0, 0.0
    h_vals = [metrics[p]["H"] for p in positions]
    c_vals = [metrics[p]["C"] for p in positions]
    delta_h = max(h_vals) - min(h_vals)
    delta_c = max(c_vals) - min(c_vals)
    return delta_h, delta_c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Samples per position")
    parser.add_argument("--layer", type=int, default=10, help="Steering layer")
    parser.add_argument("--seed", type=int, default=0, help="Data seed")
    parser.add_argument("--alphas", type=str, default="-5,-3,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,3,5",
                        help="Comma-separated alpha values")
    parser.add_argument("--max_new_tokens", type=int, default=20)
    args = parser.parse_args()

    alphas = [float(a.strip()) for a in args.alphas.split(",")]

    log("=" * 60)
    log("P12: Position-Directional Open-Loop Steering for Absorption")
    log("=" * 60)
    log(f"  Layer: {args.layer}")
    log(f"  N per position: {args.n}")
    log(f"  Alphas: {alphas}")
    log(f"  Seed: {args.seed}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"\nDevice: {device}")

    model, tokenizer = load_model_and_tokenizer()
    model.eval()

    pos_dir = f"data_position_sensitivity/s{args.seed}"

    log(f"\n[1] Computing steering vectors at layer {args.layer}...")
    t0 = time.time()
    v_abs, v_random, v_orthogonal = compute_steering_vector(
        model, tokenizer, pos_dir, args.seed, args.layer, device
    )
    log(f"  Done in {time.time() - t0:.1f}s")

    log(f"\n[2] Loading test samples...")
    test_all = []
    for pos in ["early", "mid", "late"]:
        test_path = os.path.join(pos_dir, f"test_{pos}_s{args.seed}.jsonl")
        samples = load_jsonl(test_path)[:args.n]
        for s in samples:
            s["position"] = pos
            s["_position"] = pos
        test_all.extend(samples)
    log(f"  Loaded {len(test_all)} test samples ({args.n} per position)")

    all_results = {}

    log(f"\n[3] Running alpha sweep...")
    for alpha in alphas:
        label = f"alpha={alpha:+.1f}"
        t_start = time.time()
        raw = run_position_test(
            model, tokenizer, test_all, args.layer, v_abs,
            alpha, args.max_new_tokens, device
        )
        metrics = compute_position_metrics(raw)
        delta_h, delta_c = compute_delta(metrics)
        elapsed = time.time() - t_start

        log(f"  [{label}] delta_H={delta_h:.3f} delta_C={delta_c:.3f} "
            f"H_early={metrics.get('early', {}).get('H', 0):.3f} "
            f"H_mid={metrics.get('mid', {}).get('H', 0):.3f} "
            f"H_late={metrics.get('late', {}).get('H', 0):.3f} "
            f"({elapsed:.0f}s)")

        all_results[label] = {
            "alpha": alpha,
            "delta_h": delta_h,
            "delta_c": delta_c,
            "per_position": {p: {"H": m["H"], "C": m["C"]} for p, m in metrics.items()},
            "time_s": elapsed,
        }

    log(f"\n[4] Running random vector control (key alphas)...")
    control_alphas = [a for a in alphas if a in [-3, -1.5, 0, 1.5, 3]]
    for alpha in control_alphas:
        label = f"random_alpha={alpha:+.1f}"
        t_start = time.time()
        raw = run_position_test(
            model, tokenizer, test_all, args.layer, v_random,
            alpha, args.max_new_tokens, device
        )
        metrics = compute_position_metrics(raw)
        delta_h, delta_c = compute_delta(metrics)
        elapsed = time.time() - t_start
        log(f"  [{label}] delta_H={delta_h:.3f} delta_C={delta_c:.3f} "
            f"H_early={metrics.get('early', {}).get('H', 0):.3f} "
            f"H_late={metrics.get('late', {}).get('H', 0):.3f} "
            f"({elapsed:.0f}s)")
        all_results[label] = {
            "alpha": alpha,
            "vector_type": "random",
            "delta_h": delta_h,
            "delta_c": delta_c,
            "per_position": {p: {"H": m["H"], "C": m["C"]} for p, m in metrics.items()},
            "time_s": elapsed,
        }

    log(f"\n[5] Running orthogonal vector control...")
    for alpha in control_alphas:
        label = f"orth_alpha={alpha:+.1f}"
        t_start = time.time()
        raw = run_position_test(
            model, tokenizer, test_all, args.layer, v_orthogonal,
            alpha, args.max_new_tokens, device
        )
        metrics = compute_position_metrics(raw)
        delta_h, delta_c = compute_delta(metrics)
        elapsed = time.time() - t_start
        log(f"  [{label}] delta_H={delta_h:.3f} delta_C={delta_c:.3f} "
            f"H_early={metrics.get('early', {}).get('H', 0):.3f} "
            f"H_late={metrics.get('late', {}).get('H', 0):.3f} "
            f"({elapsed:.0f}s)")
        all_results[label] = {
            "alpha": alpha,
            "vector_type": "orthogonal",
            "delta_h": delta_h,
            "delta_c": delta_c,
            "per_position": {p: {"H": m["H"], "C": m["C"]} for p, m in metrics.items()},
            "time_s": elapsed,
        }

    log(f"\n[6] Summary...")
    baseline = all_results.get("alpha=+0.0", {})
    baseline_dh = baseline.get("delta_h", 0)
    log(f"  Baseline delta_H: {baseline_dh:.3f}")

    best_key = None
    best_dh = baseline_dh
    for key, result in all_results.items():
        if key.startswith("alpha=") and result["delta_h"] < best_dh:
            best_dh = result["delta_h"]
            best_key = key

    if best_key:
        log(f"  Best: {best_key} delta_H={best_dh:.3f} "
            f"(improvement: {baseline_dh - best_dh:+.3f})")
    else:
        log(f"  No improvement over baseline")

    summary = {
        "layer": args.layer,
        "n_per_pos": args.n,
        "seed": args.seed,
        "alphas": alphas,
        "baseline_delta_h": baseline_dh,
        "best_key": best_key,
        "best_delta_h": best_dh,
        "all_results": all_results,
    }

    with open(os.path.join(RESULTS_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log(f"\nResults saved to {RESULTS_DIR}/results.json")
    log("Done.")


if __name__ == "__main__":
    main()