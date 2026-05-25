"""
P12-v2: Position-Directional Open-Loop Steering (CPU-Friendly).
================================================================
Same hypothesis as P12 but replaces autoregressive generation with
log-prob comparison of positive vs negative response for CPU efficiency.

Design:
  1. Compute v_abs = mean(h_early) - mean(h_late) from training data
     (direction FROM late-pos degradation TOWARD early-pos information)
  2. For each alpha, apply steering hook at target layer during forward pass
  3. Evaluate via log-prob comparison (positive vs negative response)
  4. Alpha-sweep to find optimal compensation magnitude
  5. Compare: baseline vs best alpha vs random/orthogonal control

Hypotheses:
  H12.1: Positive alpha (add v_abs) reduces H_late (late-position hallucination)
  H12.2: Positive alpha reduces delta_H = max(H)-min(H) across positions
  H12.3: Random/orthogonal vector steering does NOT produce the same effect
  H12.4: There exists an optimal alpha with minimal delta_H

Usage:
  cd F:\internal_circuit_capital_lab\IC-4-M0
  python src/run_p12_position_steering.py --n 10 --layer 10 --alphas "-3,-1.5,-0.5,0,0.5,1.5,3"
"""

import argparse, os, sys, time, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_loader import load_model_and_tokenizer
from src.data_builder import load_jsonl

RESULTS_DIR = "results_p12_position_steering"
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, "run_log.txt")

def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()

def collect_position_hidden_states(model, tokenizer, samples, layer_idx, device):
    X_list = []
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
        captured.append(hs[0, -1, :].detach().cpu().float().numpy().copy())

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

    handle.remove()
    return np.array(X_list)

def compute_steering_vector(model, tokenizer, pos_dir, seed, layer, device):
    train_early = load_jsonl(os.path.join(pos_dir, f"train_early_s{seed}.jsonl"))
    train_late = load_jsonl(os.path.join(pos_dir, f"train_late_s{seed}.jsonl"))

    X_early = collect_position_hidden_states(model, tokenizer, train_early, layer, device)
    X_late = collect_position_hidden_states(model, tokenizer, train_late, layer, device)

    v_abs = X_early.mean(axis=0) - X_late.mean(axis=0)
    v_abs = v_abs / (np.linalg.norm(v_abs) + 1e-8)

    v_random = np.random.RandomState(42).randn(*v_abs.shape).astype(np.float32)
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
            return (output[0] + alpha * sv.view(1, 1, -1),) + output[1:]
        return output + alpha * sv.view(1, 1, -1)
    return hook_fn

def logprob_with_steering(model, tokenizer, prompt, response, layer_idx,
                          steering_vector, alpha, device):
    full_text = f"{prompt} {response}"
    full_ids = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=256)
    full_ids = {k: v.to(device) for k, v in full_ids.items()}

    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    prompt_len = prompt_ids["input_ids"].shape[1]

    labels = full_ids["input_ids"].clone()
    labels[0, :prompt_len] = -100

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
        outputs = model(**full_ids, labels=labels)

    if handle:
        handle.remove()

    return -outputs.loss.item()

def evaluate_with_steering(model, tokenizer, samples, layer_idx, steering_vector,
                           alpha, device):
    results = []
    for sample in samples:
        context = sample.get("context", "")
        question = sample.get("question", "")
        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"

        pos_resp = sample.get("positive_response", "")
        neg_resp = sample.get("negative_response", "")

        pos_lp = logprob_with_steering(model, tokenizer, prompt, pos_resp,
                                        layer_idx, steering_vector, alpha, device)
        neg_lp = logprob_with_steering(model, tokenizer, prompt, neg_resp,
                                        layer_idx, steering_vector, alpha, device)

        pref_positive = pos_lp > neg_lp

        results.append({
            "answerability": sample.get("answerability", "?"),
            "pref_positive": pref_positive,
            "pos_logprob": pos_lp,
            "neg_logprob": neg_lp,
        })

    return results

def compute_metrics(eval_results):
    answerable = [r for r in eval_results if r.get("answerability") == "answerable"]
    unanswerable = [r for r in eval_results if r.get("answerability") == "unanswerable"]

    n_ans = len(answerable)
    n_unans = len(unanswerable)

    hallucinations = sum(1 for r in unanswerable if r["pref_positive"])
    correct = sum(1 for r in answerable if r["pref_positive"])

    H = hallucinations / n_unans if n_unans > 0 else 0.0
    C = correct / n_ans if n_ans > 0 else 0.0

    hall_count = hallucinations
    unans_count = n_unans
    corr_count = correct
    ans_count = n_ans

    margins = []
    for r in unanswerable:
        margins.append(r["pos_logprob"] - r["neg_logprob"])

    mean_margin = float(np.mean(margins)) if margins else 0.0

    return {
        "H": round(H, 4),
        "C": round(C, 4),
        "hall_count": hall_count,
        "unans_count": unans_count,
        "corr_count": corr_count,
        "ans_count": ans_count,
        "mean_margin": round(mean_margin, 4),
    }

def compute_position_consistency(eval_results_dict):
    n_consistent = 0
    n_total = 0
    for i in range(len(eval_results_dict.get("early", []))):
        if i < len(eval_results_dict.get("mid", [])) and i < len(eval_results_dict.get("late", [])):
            pref_early = eval_results_dict["early"][i]["pref_positive"]
            pref_mid = eval_results_dict["mid"][i]["pref_positive"]
            pref_late = eval_results_dict["late"][i]["pref_positive"]
            if pref_early == pref_mid == pref_late:
                n_consistent += 1
            n_total += 1
    return n_consistent / max(n_total, 1), n_consistent, n_total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Samples per position")
    parser.add_argument("--layer", type=int, default=10, help="Steering layer")
    parser.add_argument("--seed", type=int, default=0, help="Data seed")
    parser.add_argument("--alphas", type=str, default="-3,-1.5,-0.5,0,0.5,1.5,3",
                        help="Comma-separated alpha values")
    args = parser.parse_args()

    alphas = [float(a.strip()) for a in args.alphas.split(",")]

    log("=" * 64)
    log("P12-v2: Position-Directional Steering (log-prob evaluation)")
    log(f"  Layer: {args.layer}, N per position: {args.n}")
    log(f"  Alphas: {alphas}, Seed: {args.seed}")
    log("=" * 64)
    t0 = time.time()

    log("\n[Step 1] Loading model on CPU...")
    model, tokenizer = load_model_and_tokenizer(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="cpu", torch_dtype="float32",
    )
    device = next(model.parameters()).device

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pos_dir = os.path.join(base_dir, "data_position_sensitivity", f"s{args.seed}")

    log(f"\n[Step 2] Computing steering vectors at layer {args.layer}...")
    t_vec = time.time()
    v_abs, v_random, v_orthogonal = compute_steering_vector(
        model, tokenizer, pos_dir, args.seed, args.layer, device
    )
    log(f"  Done in {time.time() - t_vec:.0f}s")

    log(f"\n[Step 3] Loading test samples...")
    test_all = []
    pos_test = {}
    for pos in ["early", "mid", "late"]:
        test_path = os.path.join(pos_dir, f"test_{pos}_s{args.seed}.jsonl")
        samples = load_jsonl(test_path)[:args.n]
        for s in samples:
            s["_position"] = pos
        pos_test[pos] = samples
        test_all.extend(samples)
    log(f"  Loaded {len(test_all)} test samples ({args.n} per position)")

    all_results = {}

    log(f"\n[Step 4] Alpha sweep (v_abs: early-late direction)...")
    for alpha in alphas:
        label = f"alpha={alpha:+.1f}"
        t_start = time.time()
        pos_evals = {}
        for pos in ["early", "mid", "late"]:
            pos_evals[pos] = evaluate_with_steering(
                model, tokenizer, pos_test[pos], args.layer, v_abs, alpha, device
            )
        metrics = {pos: compute_metrics(pos_evals[pos]) for pos in ["early","mid","late"]}
        delta_h = max(metrics[p]["H"] for p in metrics) - min(metrics[p]["H"] for p in metrics)
        consistency, nc, nt = compute_position_consistency(pos_evals)

        elapsed = time.time() - t_start
        log(f"  [{label:>10s}] delta_H={delta_h:.3f} "
            f"H(early={metrics['early']['H']:.3f}, mid={metrics['mid']['H']:.3f}, "
            f"late={metrics['late']['H']:.3f}) "
            f"C={consistency:.3f} ({nc}/{nt}) "
            f"({elapsed:.0f}s)")

        all_results[label] = {
            "alpha": alpha, "vector_type": "abs",
            "delta_h": delta_h, "consistency": round(consistency, 4),
            "per_position": {p: {"H": m["H"], "C": m["C"],
                "hall_count": m["hall_count"], "corr_count": m["corr_count"]}
                for p, m in metrics.items()},
            "time_s": elapsed,
        }

    baseline = all_results.get("alpha=+0.0", {})
    baseline_dh = baseline.get("delta_h", 0)

    log(f"\n[Step 5] Random vector control...")
    control_alphas = [a for a in alphas if a in [-3, -1.5, 0, 1.5, 3]]
    for alpha in control_alphas:
        label = f"random_alpha={alpha:+.1f}"
        t_start = time.time()
        pos_evals = {}
        for pos in ["early", "mid", "late"]:
            pos_evals[pos] = evaluate_with_steering(
                model, tokenizer, pos_test[pos], args.layer, v_random, alpha, device
            )
        metrics = {pos: compute_metrics(pos_evals[pos]) for pos in ["early","mid","late"]}
        delta_h = max(metrics[p]["H"] for p in metrics) - min(metrics[p]["H"] for p in metrics)
        elapsed = time.time() - t_start
        log(f"  [{label:>18s}] delta_H={delta_h:.3f} "
            f"H(early={metrics['early']['H']:.3f}, late={metrics['late']['H']:.3f}) ({elapsed:.0f}s)")
        all_results[label] = {
            "alpha": alpha, "vector_type": "random",
            "delta_h": delta_h,
            "per_position": {p: {"H": m["H"], "C": m["C"]} for p, m in metrics.items()},
            "time_s": elapsed,
        }

    log(f"\n[Step 6] Orthogonal vector control...")
    for alpha in control_alphas:
        label = f"orth_alpha={alpha:+.1f}"
        t_start = time.time()
        pos_evals = {}
        for pos in ["early", "mid", "late"]:
            pos_evals[pos] = evaluate_with_steering(
                model, tokenizer, pos_test[pos], args.layer, v_orthogonal, alpha, device
            )
        metrics = {pos: compute_metrics(pos_evals[pos]) for pos in ["early","mid","late"]}
        delta_h = max(metrics[p]["H"] for p in metrics) - min(metrics[p]["H"] for p in metrics)
        elapsed = time.time() - t_start
        log(f"  [{label:>18s}] delta_H={delta_h:.3f} "
            f"H(early={metrics['early']['H']:.3f}, late={metrics['late']['H']:.3f}) ({elapsed:.0f}s)")
        all_results[label] = {
            "alpha": alpha, "vector_type": "orthogonal",
            "delta_h": delta_h,
            "per_position": {p: {"H": m["H"], "C": m["C"]} for p, m in metrics.items()},
            "time_s": elapsed,
        }

    log(f"\n{'='*64}")
    log(f"[Step 7] Summary...")
    log(f"  Baseline (alpha=0): delta_H={baseline_dh:.3f}")

    best_key = None
    best_dh = baseline_dh
    for key, result in all_results.items():
        if key.startswith("alpha=") and result["delta_h"] < best_dh:
            best_dh = result["delta_h"]
            best_key = key

    if best_key and best_dh < baseline_dh:
        improvement = baseline_dh - best_dh
        log(f"  Best: {best_key} delta_H={best_dh:.3f} (improvement: {improvement:+.3f}, "
            f"{-improvement/baseline_dh*100:.1f}%)")
    else:
        log(f"  No improvement over baseline. Best delta_H={best_dh:.3f}")

    log(f"\n  Direction comparison:")
    for vec_label, vec_name in [("alpha=+1.5", "v_abs"), ("random_alpha=+1.5", "random"),
                                 ("random_alpha=-1.5", "random_neg"),
                                 ("orth_alpha=+1.5", "orthogonal")]:
        if vec_label in all_results:
            r = all_results[vec_label]
            log(f"    {vec_name:>15s}: delta_H={r['delta_h']:.3f}")

    elapsed = time.time() - t0
    log(f"\nPhase 12 Complete. ({elapsed:.0f}s, {elapsed/60:.1f} min)")

    summary = {
        "layer": args.layer, "n_per_pos": args.n, "seed": args.seed,
        "alphas": alphas, "baseline_delta_h": baseline_dh,
        "best_key": best_key, "best_delta_h": best_dh,
        "all_results": all_results, "time_s": round(elapsed, 1),
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log(f"Results saved to {RESULTS_DIR}/results.json")

if __name__ == "__main__":
    main()