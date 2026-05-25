"""
P14: Cross-Layer B-Bottleneck Characterization.
===============================================
P13 proved that at layer 12, the hallucination probe achieves perfect classification
(acc=1.000) but its decision boundary direction has ZERO causal effect on behavior.
This is the geometric proof of the B-bottleneck as SUBSPACE SEPARATION.

P14 extends this to ALL layers, asking:
  - Is the classification-vs-control subspace separation universal across depth?
  - Are there layers where KNOWING partially overlaps with CONTROLLING?
  - What is the layer-wise geometry of the B-bottleneck?

Design:
  1. For each layer (0-23), train hallucination probe on 90 position samples
  2. Extract w_probe = decision boundary direction
  3. Test steering effect at alpha = -2.0, 0, +2.0 via log-prob comparison
  4. Measure per-layer: probe_acc, H_baseline, delta_H at ±2.0, C_baseline, C_drop
  5. Compute layer-wise alignment score = max(|delta_H_plus|, |delta_H_minus|)

Key metrics:
  - probe_acc: classification accuracy (how well the layer KNOWS)
  - delta_H_max: max behavioral change from w_probe steering (how much it CONTROLS)
  - overlap_ratio: delta_H_max / probe_acc (KNOWS→CONTROLS alignment)
  - C_drop: correctness degradation at max steering

Hypotheses:
  H14.1: probe_acc is high in middle layers, low in early/late layers
  H14.2: delta_H_max is near-zero at ALL layers (subspace separation is depth-universal)
  H14.3: If H14.2 false, there exists optimal layer with non-zero overlap_ratio
  H14.4: C_drop correlates with delta_H_max (any behavioral change is destructive)

Usage:
  cd F:\internal_circuit_capital_lab\IC-4-M0
  python src/run_p14_cross_layer_bottleneck.py --n 10 --seed 0
"""

import argparse, os, sys, time, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_loader import load_model_and_tokenizer
from src.data_builder import load_jsonl
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

RESULTS_DIR = "results_p14_cross_layer_bottleneck"
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
    if target_module is None:
        raise ValueError(f"Layer {layer_idx} not found")

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

def evaluate_at_alpha(model, tokenizer, samples, layer_idx, steering_vector,
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

    return {
        "H": round(H, 4), "C": round(C, 4),
        "hall_count": hallucinations, "unans_count": n_unans,
        "corr_count": correct, "ans_count": n_ans,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Test samples per position")
    parser.add_argument("--seed", type=int, default=0, help="Data seed")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices (default: all 0-23)")
    parser.add_argument("--step", type=int, default=3, help="Layer step size")
    args = parser.parse_args()

    if args.layers is not None:
        layers = [int(l.strip()) for l in args.layers.split(",")]
    else:
        layers = list(range(0, 24, args.step))
        if 11 not in layers:
            layers.append(11)
        if 12 not in layers:
            layers.append(12)
        layers = sorted(set(layers))

    test_alphas = [-2.0, 0.0, 2.0]

    log("=" * 64)
    log("P14: Cross-Layer B-Bottleneck Characterization")
    log(f"  Layers: {layers}")
    log(f"  Test N per position: {args.n}, Seed: {args.seed}")
    log(f"  Alphas: {test_alphas}")
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

    log(f"\n[Step 2] Loading training samples...")
    train_samples = []
    for pos in ["early", "mid", "late"]:
        path = os.path.join(pos_dir, f"train_{pos}_s{args.seed}.jsonl")
        train_samples.extend(load_jsonl(path))
    log(f"  Total training samples: {len(train_samples)}")

    log(f"\n[Step 3] Loading test samples...")
    test_samples = []
    for pos in ["early", "mid", "late"]:
        path = os.path.join(pos_dir, f"test_{pos}_s{args.seed}.jsonl")
        for s in load_jsonl(path)[:args.n]:
            s["_position"] = pos
            test_samples.append(s)
    n_unans_test = sum(1 for s in test_samples if s.get("answerability") != "answerable")
    log(f"  Test: {len(test_samples)} samples ({n_unans_test} unanswerable)")

    layer_results = []

    for layer_idx in layers:
        t_layer_start = time.time()
        log(f"\n{'─'*48}")
        log(f"[Layer {layer_idx:2d}] ...")

        log(f"  Collecting hidden states...")
        X_train, y_train = collect_hidden_states(
            model, tokenizer, train_samples, layer_idx, device
        )
        n_ans = int(sum(y_train))
        log(f"  X_train: {X_train.shape}, A:{n_ans} U:{len(y_train)-n_ans}")

        log(f"  Training probe...")
        probe = train_probe_and_extract_direction(X_train, y_train)
        log(f"  Probe acc={probe['accuracy']:.4f} bias={probe['bias']:.3f}")

        alpha_metrics = {}
        for alpha in test_alphas:
            eval_results = evaluate_at_alpha(
                model, tokenizer, test_samples, layer_idx,
                probe["w_probe"], alpha, device
            )
            metrics = compute_metrics(eval_results)

            pos_metrics = {}
            for pos in ["early", "mid", "late"]:
                pos_samples = [r for r, s in zip(eval_results, test_samples)
                              if s.get("_position") == pos]
                pos_metrics[pos] = compute_metrics(pos_samples)

            h_values = [pos_metrics[p]["H"] for p in ["early","mid","late"]]
            delta_h = max(h_values) - min(h_values)

            alpha_metrics[alpha] = {
                "H": metrics["H"], "C": metrics["C"],
                "delta_h": delta_h,
                "hall_count": metrics["hall_count"],
                "unans_count": metrics["unans_count"],
                "per_position": {p: {"H": m["H"], "C": m["C"]}
                                for p, m in pos_metrics.items()},
            }

        baseline = alpha_metrics[0.0]
        delta_H_plus = alpha_metrics[2.0]["H"] - baseline["H"]
        delta_H_minus = alpha_metrics[-2.0]["H"] - baseline["H"]
        delta_H_max = max(abs(delta_H_plus), abs(delta_H_minus))
        delta_H_effective = delta_H_plus if abs(delta_H_plus) >= abs(delta_H_minus) else delta_H_minus

        C_drop_plus = baseline["C"] - alpha_metrics[2.0]["C"]
        C_drop_minus = baseline["C"] - alpha_metrics[-2.0]["C"]
        C_drop_max = max(C_drop_plus, C_drop_minus)

        overlap_ratio = delta_H_max / max(probe["accuracy"], 0.01)

        layer_info = {
            "layer": layer_idx,
            "probe_acc": probe["accuracy"],
            "probe_bias": probe["bias"],
            "H_baseline": baseline["H"],
            "C_baseline": baseline["C"],
            "delta_h_baseline": baseline["delta_h"],
            "H_plus": alpha_metrics[2.0]["H"],
            "H_minus": alpha_metrics[-2.0]["H"],
            "delta_H_plus": round(delta_H_plus, 4),
            "delta_H_minus": round(delta_H_minus, 4),
            "delta_H_max": round(delta_H_max, 4),
            "delta_H_effective": round(delta_H_effective, 4),
            "C_plus": alpha_metrics[2.0]["C"],
            "C_minus": alpha_metrics[-2.0]["C"],
            "C_drop_plus": round(C_drop_plus, 4),
            "C_drop_minus": round(C_drop_minus, 4),
            "C_drop_max": round(C_drop_max, 4),
            "overlap_ratio": round(overlap_ratio, 4),
            "alpha_metrics": {str(k): v for k, v in alpha_metrics.items()},
        }

        elapsed = time.time() - t_layer_start
        log(f"  H baseline={baseline['H']:.3f} C={baseline['C']:.3f} ΔH={baseline['delta_h']:.3f}")
        log(f"  H(+2.0)={alpha_metrics[2.0]['H']:.3f} (Δ={delta_H_plus:+.3f}) "
            f"H(-2.0)={alpha_metrics[-2.0]['H']:.3f} (Δ={delta_H_minus:+.3f})")
        log(f"  C_drop_max={C_drop_max:.3f} overlap_ratio={overlap_ratio:.4f} "
            f"({elapsed:.0f}s)")

        layer_results.append(layer_info)

    log(f"\n{'='*64}")
    log(f"[Summary] Cross-Layer B-Bottleneck Map")
    log(f"{'='*64}")
    log(f"  {'Layer':>5s}  {'ProbeAcc':>8s}  {'H_base':>6s}  {'C_base':>6s}  "
        f"{'ΔH_max':>7s}  {'C_drop':>7s}  {'Overlap':>7s}")
    log(f"  {'─'*5}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*7}")

    best_overlap_layer = None
    best_overlap = -1
    best_probe_layer = None
    best_probe = -1

    for r in layer_results:
        log(f"  {r['layer']:5d}  {r['probe_acc']:8.4f}  {r['H_baseline']:6.3f}  "
            f"{r['C_baseline']:6.3f}  {r['delta_H_max']:+7.4f}  "
            f"{r['C_drop_max']:+7.4f}  {r['overlap_ratio']:7.4f}")
        if r["overlap_ratio"] > best_overlap:
            best_overlap = r["overlap_ratio"]
            best_overlap_layer = r["layer"]
        if r["probe_acc"] > best_probe:
            best_probe = r["probe_acc"]
            best_probe_layer = r["layer"]

    log(f"\n  Best probe layer: {best_probe_layer} (acc={best_probe:.4f})")
    log(f"  Best overlap layer: {best_overlap_layer} (ratio={best_overlap:.4f})")

    universal_zero = all(r["delta_H_max"] < 0.05 for r in layer_results)
    if universal_zero:
        log(f"\n  *** H14.2 CONFIRMED: delta_H_max < 0.05 at ALL layers ***")
        log(f"  *** B-bottleneck subspace separation is DEPTH-UNIVERSAL ***")
    else:
        nonzero_layers = [r for r in layer_results if r["delta_H_max"] >= 0.05]
        log(f"\n  *** H14.2 PARTIALLY REJECTED: {len(nonzero_layers)} layers have delta_H_max >= 0.05")
        for r in nonzero_layers:
            log(f"      Layer {r['layer']}: ΔH_max={r['delta_H_max']:.4f} overlap={r['overlap_ratio']:.4f}")

    elapsed = time.time() - t0
    log(f"\nP14 Complete. ({elapsed:.0f}s, {elapsed/60:.1f} min)")

    summary = {
        "n_test_per_pos": args.n,
        "seed": args.seed,
        "layers_tested": layers,
        "test_alphas": test_alphas,
        "best_probe_layer": best_probe_layer,
        "best_probe_acc": best_probe,
        "best_overlap_layer": best_overlap_layer,
        "best_overlap_ratio": best_overlap,
        "universal_zero": universal_zero,
        "layer_results": layer_results,
        "time_s": round(elapsed, 1),
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log(f"\nResults saved to {RESULTS_DIR}/results.json")

if __name__ == "__main__":
    main()