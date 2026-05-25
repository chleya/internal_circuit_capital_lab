"""
P13: Probe-Guided Hallucination Steering (B-bottleneck bridge).
=================================================================
The B-bottleneck: model KNOWS (KNN=1.0, probe acc~1.0) but DOESN'T produce.
This experiment tests whether the probe's decision boundary direction can
route latent knowledge into improved behavior.

Design:
  1. Train hallucination probe (answerable vs unanswerable) on position data
  2. Extract w_probe = decision boundary direction in hidden space
  3. Apply steering hook at target layer with ±alpha * w_probe
  4. Evaluate via log-prob comparison (positive vs negative response)
  5. Alpha-sweep to find optimal steering magnitude
  6. Compare with random/orthogonal control

Direction semantics:
  - w_probe points FROM unanswerable → answerable (class 0→1)
  - Applying +alpha: makes hidden states look MORE answerable
  - Applying -alpha: makes hidden states look MORE unanswerable
  - HYPOTHESIS: -alpha should reduce hallucination on unanswerable samples
    (model is steered toward "knowing it doesn't know")

Hypotheses:
  H13.1: Negative alpha reduces H on unanswerable samples (steering toward abstention)
  H13.2: Positive alpha increases H (steering toward hallucination)
  H13.3: Random/orthogonal steering does NOT show the same directional effect
  H13.4: There exists an optimal alpha with minimal H (bridging KNOWS→produces)

Usage:
  cd F:\internal_circuit_capital_lab\IC-4-M0
  python src/run_p13_probe_guided_steering.py --n 10 --layer 12 --alphas "-5,-2,-1,0,1,2,5"
"""

import argparse, os, sys, time, json, pickle
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_loader import load_model_and_tokenizer
from src.data_builder import load_jsonl
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

RESULTS_DIR = "results_p13_probe_guided_steering"
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, "run_log.txt")

def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()

def collect_hidden_states(model, tokenizer, samples, layer_idx, device):
    X_list, y_list = [], []
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
            y_list.append(1 if sample.get("answerability") == "answerable" else 0)

    handle.remove()
    return np.array(X_list), np.array(y_list)

def train_probe_and_extract_direction(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(Xs, y)
    acc = clf.score(Xs, y)
    w_probe_z = clf.coef_[0]
    w_probe_x = w_probe_z / scaler.scale_
    w_probe_x = w_probe_x / (np.linalg.norm(w_probe_x) + 1e-8)
    bias = clf.intercept_[0]
    return {
        "classifier": clf, "scaler": scaler,
        "accuracy": float(acc),
        "w_probe": w_probe_x.astype(np.float32),
        "bias": float(bias),
        "n_train": len(y),
        "n_answerable": int(sum(y)),
        "n_unanswerable": int(len(y) - sum(y)),
    }

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
    cal_abst = sum(1 for r in unanswerable if not r["pref_positive"])
    unc_abst = sum(1 for r in answerable if not r["pref_positive"])

    H = hallucinations / n_unans if n_unans > 0 else 0.0
    C = correct / n_ans if n_ans > 0 else 0.0

    return {
        "H": round(H, 4), "C": round(C, 4),
        "hall_count": hallucinations, "unans_count": n_unans,
        "corr_count": correct, "ans_count": n_ans,
        "cal_abst": cal_abst, "unc_abst": unc_abst,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Test samples per position")
    parser.add_argument("--layer", type=int, default=12, help="Probe/steering layer")
    parser.add_argument("--seed", type=int, default=0, help="Data seed")
    parser.add_argument("--alphas", type=str, default="-5,-2,-1,-0.5,0,0.5,1,2,5",
                        help="Comma-separated alpha values")
    args = parser.parse_args()

    alphas = [float(a.strip()) for a in args.alphas.split(",")]

    log("=" * 64)
    log("P13: Probe-Guided Hallucination Steering (B-bottleneck bridge)")
    log(f"  Layer: {args.layer}, Test N per position: {args.n}")
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

    log(f"\n[Step 2] Collecting training data for probe...")
    train_samples = []
    for pos in ["early", "mid", "late"]:
        path = os.path.join(pos_dir, f"train_{pos}_s{args.seed}.jsonl")
        train_samples.extend(load_jsonl(path))
    log(f"  Total training samples: {len(train_samples)}")

    log(f"\n[Step 3] Collecting hidden states at layer {args.layer}...")
    X_train, y_train = collect_hidden_states(model, tokenizer, train_samples, args.layer, device)
    n_ans = int(sum(y_train))
    log(f"  X_train.shape: {X_train.shape}, answerable: {n_ans}, unanswerable: {len(y_train)-n_ans}")

    log(f"\n[Step 4] Training probe & extracting decision boundary...")
    probe = train_probe_and_extract_direction(X_train, y_train)
    log(f"  Probe accuracy: {probe['accuracy']:.4f}")
    log(f"  w_probe norm: {np.linalg.norm(probe['w_probe']):.4f}")
    log(f"  w_probe direction: A→U (+alpha=more answerable-like)")

    v_random = np.random.RandomState(42).randn(*probe["w_probe"].shape).astype(np.float32)
    v_random = v_random / (np.linalg.norm(v_random) + 1e-8)

    log(f"\n[Step 5] Loading test samples...")
    test_samples = []
    for pos in ["early", "mid", "late"]:
        path = os.path.join(pos_dir, f"test_{pos}_s{args.seed}.jsonl")
        for s in load_jsonl(path)[:args.n]:
            s["_position"] = pos
            test_samples.append(s)
    n_unans_test = sum(1 for s in test_samples if s.get("answerability") != "answerable")
    log(f"  Test: {len(test_samples)} samples ({n_unans_test} unanswerable)")

    log(f"\n[Step 6] Alpha sweep (probe-guided steering)...")
    all_results = {}

    for alpha in alphas:
        label = f"alpha={alpha:+.1f}"
        t_start = time.time()

        eval_results = evaluate_with_steering(
            model, tokenizer, test_samples, args.layer,
            probe["w_probe"], alpha, device
        )
        metrics = compute_metrics(eval_results)

        pos_metrics = {}
        for pos in ["early", "mid", "late"]:
            pos_samples = [r for r, s in zip(eval_results, test_samples) if s.get("_position") == pos]
            pos_metrics[pos] = compute_metrics(pos_samples)

        h_values = [pos_metrics[p]["H"] for p in ["early","mid","late"]]
        delta_h = max(h_values) - min(h_values)

        elapsed = time.time() - t_start
        log(f"  [{label:>10s}] H={metrics['H']:.3f} C={metrics['C']:.3f} "
            f"ΔH={delta_h:.3f} "
            f"(hall={metrics['hall_count']}/{metrics['unans_count']}, "
            f"corr={metrics['corr_count']}/{metrics['ans_count']}) "
            f"cal_abst={metrics['cal_abst']} unc_abst={metrics['unc_abst']} "
            f"({elapsed:.0f}s)")

        all_results[label] = {
            "alpha": alpha, "vector_type": "probe",
            "H": metrics["H"], "C": metrics["C"],
            "delta_h": delta_h,
            "hall_count": metrics["hall_count"],
            "unans_count": metrics["unans_count"],
            "cal_abst": metrics["cal_abst"],
            "unc_abst": metrics["unc_abst"],
            "per_position": {p: {"H": m["H"], "C": m["C"]} for p, m in pos_metrics.items()},
            "time_s": elapsed,
        }

    baseline = all_results.get("alpha=+0.0", {})
    baseline_h = baseline.get("H", 0.5)

    log(f"\n[Step 7] Random vector control...")
    control_alphas = [a for a in alphas if a in [-5, -2, 0, 2, 5]]
    for alpha in control_alphas:
        label = f"rand_{alpha:+.1f}"
        t_start = time.time()
        eval_results = evaluate_with_steering(
            model, tokenizer, test_samples, args.layer, v_random, alpha, device
        )
        metrics = compute_metrics(eval_results)
        elapsed = time.time() - t_start
        log(f"  [{label:>12s}] H={metrics['H']:.3f} C={metrics['C']:.3f} "
            f"(hall={metrics['hall_count']}/{metrics['unans_count']}) ({elapsed:.0f}s)")
        all_results[label] = {
            "alpha": alpha, "vector_type": "random",
            "H": metrics["H"], "C": metrics["C"],
            "time_s": elapsed,
        }

    log(f"\n{'='*64}")
    log(f"[Step 8] Summary...")
    log(f"  Baseline (alpha=0): H={baseline_h:.3f}, C={baseline.get('C', 0):.3f}")

    best_key, best_h = None, baseline_h
    worst_key, worst_h = None, baseline_h
    for key, result in all_results.items():
        if key.startswith("alpha="):
            if result["H"] < best_h:
                best_h = result["H"]
                best_key = key
            if result["H"] > worst_h:
                worst_h = result["H"]
                worst_key = key

    if best_key:
        improvement = baseline_h - best_h
        log(f"  Best: {best_key} H={best_h:.3f} (Δ: {improvement:+.3f}, "
            f"{-improvement/baseline_h*100:.1f}%)")
    if worst_key:
        log(f"  Worst: {worst_key} H={worst_h:.3f} (Δ: {worst_h-baseline_h:+.3f})")

    log(f"\n  Directional comparison:")
    for key_suffix, label in [("alpha=-5.0", "probe -5"), ("rand_-5.0", "random -5"),
                               ("alpha=+5.0", "probe +5"), ("rand_+5.0", "random +5")]:
        if key_suffix in all_results:
            r = all_results[key_suffix]
            log(f"    {label:>12s}: H={r['H']:.3f}, C={r['C']:.3f}")

    elapsed = time.time() - t0
    log(f"\nP13 Complete. ({elapsed:.0f}s, {elapsed/60:.1f} min)")

    summary = {
        "layer": args.layer, "n_test_per_pos": args.n, "seed": args.seed,
        "probe_accuracy": probe["accuracy"],
        "probe_bias": probe["bias"],
        "n_train": probe["n_train"],
        "baseline_h": baseline_h,
        "best_key": best_key, "best_h": best_h,
        "worst_key": worst_key, "worst_h": worst_h,
        "all_results": all_results,
        "time_s": round(elapsed, 1),
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log(f"Results saved to {RESULTS_DIR}/results.json")

if __name__ == "__main__":
    main()