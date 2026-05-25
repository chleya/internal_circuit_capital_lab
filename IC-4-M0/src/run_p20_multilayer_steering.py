"""
P20: Multi-Layer Steering Comparison for Absorption.

P12: L10 v_abs steering → homogenization with degradation (H_early +100%, H_late −67%).
P19: Attention entropy U-curve: early high → mid low (L6-L18) → deep high (L21-L23).
     Position asymmetry AMPLIFIES in deep layers.

P20 asks: Does steering at different layers produce different position profiles?
  - L10 (P12 baseline): homogenization with degradation
  - L18 (end of position-INVARIANT zone, P19 gap=3.7%): should steering here behave similarly to L10?
  - L21 (deep ROUTING zone, P19 gap=6.5%): might steering here affect routing more precisely?

Hypotheses:
  H20.1: L21 steering produces LOWER delta_H than L10 (targets routing divergence)
  H20.2: L18 steering ≈ L10 (both in pre-routing zone)
  H20.3: L21 steering degrades H_early LESS than L10 (routing-specific, not uniform)

If H20.1+20.3 confirmed → absorption remedy should target deep-layer attention routing.
If all layers similar → absorption steering is layer-independent, asymmetry is structural.

Usage: python src/run_p20_multilayer_steering.py --n 10
"""

import argparse, os, sys, time, json, math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_loader import load_model_and_tokenizer
from src.data_builder import load_jsonl
from src.evaluate import evaluate_outputs

RESULTS_DIR = "results_p20_multilayer_steering"
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
    if target_module is None:
        raise ValueError(f"Layer {layer_idx} not found")

    captured = []
    def _capture(module, inputs_tup, output):
        hs = output[0] if isinstance(output, tuple) else output
        captured.append(hs[0, -1, :].detach().cpu().numpy().copy())

    handle = target_module.register_forward_hook(_capture)
    for sample in samples:
        prompt = f"{sample.get('context','')}\n\nQuestion: {sample.get('question','')}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        captured.clear()
        with torch.no_grad():
            model(**inputs)
        if captured:
            X_list.append(captured[0])
            pos_labels.append(sample.get("position", "?"))
    handle.remove()
    return np.array(X_list), pos_labels

def compute_steering_vectors(model, tokenizer, pos_dir, seed, layers, device, n_train=None):
    train_early = load_jsonl(os.path.join(pos_dir, f"train_early_s{seed}.jsonl"))
    train_late = load_jsonl(os.path.join(pos_dir, f"train_late_s{seed}.jsonl"))
    if n_train:
        train_early = train_early[:n_train]
        train_late = train_late[:n_train]
    vectors = {}
    for layer in layers:
        log(f"  Computing v_abs at L{layer} ({len(train_early)} early + {len(train_late)} late samples)...")
        X_early, _ = collect_position_hidden_states(model, tokenizer, train_early, layer, device)
        X_late, _ = collect_position_hidden_states(model, tokenizer, train_late, layer, device)
        v = X_early.mean(axis=0) - X_late.mean(axis=0)
        v = v / (np.linalg.norm(v) + 1e-8)
        vectors[layer] = v
    return vectors

def make_steering_hook(steering_vector, alpha, device, dtype):
    sv = torch.tensor(steering_vector, dtype=dtype).to(device)
    def hook_fn(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        hs_steered = hs + alpha * sv.view(1, 1, -1)
        if isinstance(output, tuple):
            return (hs_steered,) + output[1:]
        return hs_steered
    return hook_fn

def generate_with_steering(model, tokenizer, sample, layer_idx, steering_vector,
                           alpha, max_new_tokens, device):
    prompt = f"{sample.get('context','')}\n\nQuestion: {sample.get('question','')}\nAnswer:"
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
            make_steering_hook(steering_vector, alpha, device, model_dtype))

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  temperature=0.0, do_sample=False,
                                  pad_token_id=tokenizer.eos_token_id)
    if handle:
        handle.remove()
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

def run_full_test(model, tokenizer, test_samples, layer_idx, steering_vector,
                  alpha, max_new_tokens, device):
    results = []
    for i, sample in enumerate(test_samples):
        output = generate_with_steering(
            model, tokenizer, sample, layer_idx, steering_vector,
            alpha, max_new_tokens, device)
        results.append({
            "sample_id": i,
            "position": sample.get("position", "?"),
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
    h_vals = [metrics[p]["H"] for p in positions]
    c_vals = [metrics[p]["C"] for p in positions]
    return max(h_vals) - min(h_vals), max(c_vals) - min(c_vals)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--n_train", type=int, default=5, help="Training samples per position for v_abs computation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alphas", type=str, default="0,1,2,3")
    parser.add_argument("--layers", type=str, default="10,18,21")
    parser.add_argument("--max_new_tokens", type=int, default=20)
    args = parser.parse_args()

    target_layers = [int(l.strip()) for l in args.layers.split(",")]
    alphas = [float(a.strip()) for a in args.alphas.split(",")]

    log("=" * 64)
    log("P20: Multi-Layer Steering Comparison for Absorption")
    log("=" * 64)
    log(f"  Layers: {target_layers}")
    log(f"  Alphas: {alphas}")
    log(f"  N per position: {args.n} (train: {args.n_train})")
    log(f"  Seed: {args.seed}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"\nDevice: {device}")

    model, tokenizer = load_model_and_tokenizer()
    model.eval()
    pos_dir = f"data_position_sensitivity/s{args.seed}"

    log(f"\n[Step 1] Computing v_abs for layers {target_layers}...")
    t0 = time.time()
    vectors = compute_steering_vectors(model, tokenizer, pos_dir, args.seed, target_layers, device, args.n_train)
    for l, v in vectors.items():
        log(f"  L{l}: ||v_abs||={np.linalg.norm(v):.4f}")
    log(f"  Done in {time.time()-t0:.1f}s")

    log(f"\n[Step 2] Loading test data...")
    test_all = []
    for pos in ["early", "mid", "late"]:
        path = os.path.join(pos_dir, f"test_{pos}_s{args.seed}.jsonl")
        samples = load_jsonl(path)[:args.n]
        for s in samples:
            s["position"] = pos
        test_all.extend(samples)
    log(f"  Total: {len(test_all)} samples ({args.n} per position)")

    all_raw = {}
    log(f"\n[Step 3] Running steering tests...")

    for layer in target_layers:
        v = vectors[layer]
        for alpha in alphas:
            label = f"L{layer}_a{alpha:.0f}" if alpha == int(alpha) else f"L{layer}_a{alpha:.1f}"
            t1 = time.time()
            raw = run_full_test(model, tokenizer, test_all, layer, v, alpha,
                               args.max_new_tokens, device)
            all_raw[label] = raw
            metrics = compute_position_metrics(raw)
            delta_h, delta_c = compute_delta(metrics)
            h_early = metrics.get("early", {}).get("H", 0)
            h_mid = metrics.get("mid", {}).get("H", 0)
            h_late = metrics.get("late", {}).get("H", 0)
            c_early = metrics.get("early", {}).get("C", 0)
            c_late = metrics.get("late", {}).get("C", 0)
            elapsed = time.time() - t1
            log(f"  {label:>10s} H=(E={h_early:.3f}, M={h_mid:.3f}, L={h_late:.3f}) "
                f"ΔH={delta_h:.3f} ΔC={delta_c:.3f} C=(E={c_early:.3f}, L={c_late:.3f}) "
                f"[{elapsed:.0f}s]")

    log(f"\n[Step 4] Summary Matrix...")
    header = "  Alpha   "
    for layer in target_layers:
        header += f"  {'L'+str(layer)+' ':>16s}"
    log(header)
    log(f"  {'─'*8} " + " ".join([f"{'─'*16}" for _ in target_layers]))

    for alpha in alphas:
        alpha_label = f"a{alpha:.0f}" if alpha == int(alpha) else f"a{alpha:.1f}"
        row = f"  {alpha_label:<8} "
        for layer in target_layers:
            label = f"L{layer}_{alpha_label}"
            if label in all_raw:
                m = compute_position_metrics(all_raw[label])
                dh, _ = compute_delta(m)
                row += f"  ΔH={dh:.3f} H=({m.get('early',{}).get('H',0):.2f},{m.get('mid',{}).get('H',0):.2f},{m.get('late',{}).get('H',0):.2f})"
            else:
                row += f"  {'N/A':<16s}"
        log(row)

    log(f"\n[Step 5] Saving results...")
    summary = {}
    for label, raw in all_raw.items():
        metrics = compute_position_metrics(raw)
        dh, dc = compute_delta(metrics)
        summary[label] = {
            "layer": int(label.split("_")[0][1:]),
            "alpha": float(label.split("a")[1]),
            "delta_H": round(dh, 4),
            "delta_C": round(dc, 4),
            "H_early": round(metrics.get("early", {}).get("H", 0), 4),
            "H_mid": round(metrics.get("mid", {}).get("H", 0), 4),
            "H_late": round(metrics.get("late", {}).get("H", 0), 4),
            "C_early": round(metrics.get("early", {}).get("C", 0), 4),
            "C_late": round(metrics.get("late", {}).get("C", 0), 4),
            "n_per_position": args.n,
        }

    results = {
        "experiment": "P20",
        "description": "Multi-layer v_abs steering comparison for absorption (L10 vs L18 vs L21)",
        "target_layers": target_layers,
        "alphas": alphas,
        "n_per_position": args.n,
        "seed": args.seed,
        "summary": summary,
    }

    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    log(f"\n  Results saved to {RESULTS_DIR}/results.json")
    log(f"\n{'='*64}")
    log("P20 Complete.")
    log(f"{'='*64}")

if __name__ == "__main__":
    main()