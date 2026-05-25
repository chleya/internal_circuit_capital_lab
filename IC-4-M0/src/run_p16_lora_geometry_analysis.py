"""
P16: LoRA Geometry Analysis — HOW does LoRA bridge the KNOWS->DOING gap?
==========================================================================
P15 proved that hallucination LoRA reduces H from 0.417 to 0.000 at C=1.000.
But HOW? P16 asks: does LoRA change the geometric relationship between the
K-subspace (classification) and D-subspace (behavioral control)?

Design:
  1. Load P15 LoRA model checkpoint
  2. For each test layer (same as P14: 0,3,6,9,11,12,15,18,21):
     a. Collect hidden states from LoRA model
     b. Train hallucination probe → extract w_probe_lora
     c. Test steering effect: evaluate H at alpha = -2, 0, +2
  3. Compare with P14 baseline (base model, no LoRA):
     - probe_acc_base vs probe_acc_lora
     - delta_H_base vs delta_H_lora
     - overlap_ratio_base vs overlap_ratio_lora
  4. Compute alignment_gain = overlap_lora - overlap_base per layer

Key hypothesis:
  P16.1: LoRA preserves probe accuracy (K-subspace unchanged)
  P16.2: LoRA INCREASES steering effect (D-subspace alignment improved)
  P16.3: Alignment gain is concentrated in middle/deep layers

If P16.2 confirmed: LoRA bridges K<->D by aligning the classification
direction with the behavioral control direction at the representation level.

If P16.2 rejected (no steering improvement): LoRA bridges the gap through
a different mechanism (e.g., direct output-stage modification, not K<->D
subspace realignment).

Usage:
  cd F:\internal_circuit_capital_lab\IC-4-M0
  python src/run_p16_lora_geometry_analysis.py --step 3
"""

import argparse, os, sys, time, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_loader import load_model_and_tokenizer
from src.data_builder import load_jsonl
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

RESULTS_DIR = "results_p16_lora_geometry"
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
        parts = name.split(".")
        if len(parts) >= 2 and parts[-2] == "layers" and parts[-1] == str(layer_idx):
            target_module = module
            break
    if target_module is None:
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
    return {
        "accuracy": float(acc),
        "w_probe": w_probe_x.astype(np.float32),
        "bias": float(clf.intercept_[0]),
        "n_train": len(y),
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
        parts = name.split(".")
        if len(parts) >= 2 and parts[-2] == "layers" and parts[-1] == str(layer_idx):
            target_module = module
            break
    if target_module is None:
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

    return {"H": round(H, 4), "C": round(C, 4),
            "hall_count": hallucinations, "unans_count": n_unans,
            "corr_count": correct, "ans_count": n_ans}


def load_p14_baseline():
    p14_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results_p14_cross_layer_bottleneck", "results.json"
    )
    if not os.path.exists(p14_path):
        return None
    with open(p14_path, "r") as f:
        p14 = json.load(f)
    baseline = {}
    for r in p14.get("layer_results", []):
        baseline[r["layer"]] = {
            "probe_acc": r["probe_acc"],
            "H_baseline": r["H_baseline"], "C_baseline": r["C_baseline"],
            "delta_H_max": r["delta_H_max"],
            "overlap_ratio": r["overlap_ratio"],
        }
    return baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--summary-only", action="store_true",
                        help="Skip layer processing, generate summary from existing log")
    args = parser.parse_args()

    layers = list(range(0, 24, args.step))
    if 11 not in layers: layers.append(11)
    if 12 not in layers: layers.append(12)
    layers = sorted(set(layers))

    test_alphas = [-2.0, 0.0, 2.0]
    seed = 0

    log("=" * 64)
    log("P16: LoRA Geometry Analysis — HOW LoRA bridges K<->D")
    log(f"  Layers: {layers}")
    log(f"  Alphas: {test_alphas}")
    log("=" * 64)
    t0 = time.time()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    log("\n[Step 1] Loading P14 baseline results for comparison...")
    p14_baseline = load_p14_baseline()
    if p14_baseline:
        log(f"  P14 baseline loaded: {len(p14_baseline)} layers")
    else:
        log("  P14 baseline NOT FOUND — will run without comparison")

    log("\n[Step 2] Loading P15 LoRA model...")
    lora_ckpt_dir = os.path.join(base_dir, "results_p15_hallucination_lora", "checkpoint_final")
    if not os.path.isdir(lora_ckpt_dir):
        log(f"  ERROR: LoRA checkpoint not found at {lora_ckpt_dir}")
        return

    from peft import PeftModel
    base_model, tokenizer = load_model_and_tokenizer(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="cpu", torch_dtype="float32",
    )
    lora_model = PeftModel.from_pretrained(base_model, lora_ckpt_dir)
    lora_model.eval()
    device = next(lora_model.parameters()).device
    log(f"  P15 LoRA loaded successfully (device={device})")

    log("\n[Step 3] Loading test samples...")
    pos_dir = os.path.join(base_dir, "data_position_sensitivity", f"s{seed}")
    test_samples = []
    for pos in ["early", "mid", "late"]:
        path = os.path.join(pos_dir, f"test_{pos}_s{seed}.jsonl")
        if os.path.exists(path):
            for s in load_jsonl(path)[:10]:
                s["_position"] = pos
                test_samples.append(s)
    log(f"  Test: {len(test_samples)} samples")

    log("\n[Step 4] Loading training samples for probe training...")
    train_samples = []
    for pos in ["early", "mid", "late"]:
        path = os.path.join(pos_dir, f"train_{pos}_s{seed}.jsonl")
        if os.path.exists(path):
            train_samples.extend(load_jsonl(path))
    log(f"  Train: {len(train_samples)} samples")

    layer_results = []

    for layer_idx in layers:
        t_layer = time.time()
        log(f"\n{'─'*48}")
        log(f"[Layer {layer_idx:2d}] ...")

        log(f"  Collecting hidden states from LoRA model...")
        X_train, y_train = collect_hidden_states(
            lora_model, tokenizer, train_samples, layer_idx, device
        )
        log(f"  X_train: {X_train.shape}")

        log(f"  Training probe on LoRA representations...")
        probe = train_probe_and_extract_direction(X_train, y_train)
        log(f"  LoRA probe acc={probe['accuracy']:.4f} bias={probe['bias']:.3f}")

        alpha_metrics = {}
        for alpha in test_alphas:
            eval_results = evaluate_at_alpha(
                lora_model, tokenizer, test_samples, layer_idx,
                probe["w_probe"], alpha, device
            )
            metrics = compute_metrics(eval_results)
            alpha_metrics[alpha] = {"H": metrics["H"], "C": metrics["C"]}

        baseline = alpha_metrics[0.0]
        delta_H_plus = alpha_metrics[2.0]["H"] - baseline["H"]
        delta_H_minus = alpha_metrics[-2.0]["H"] - baseline["H"]
        delta_H_max = max(abs(delta_H_plus), abs(delta_H_minus))
        C_drop_plus = baseline["C"] - alpha_metrics[2.0]["C"]
        C_drop_minus = baseline["C"] - alpha_metrics[-2.0]["C"]
        C_drop_max = max(C_drop_plus, C_drop_minus)
        overlap = delta_H_max / max(probe["accuracy"], 0.01)

        p14 = p14_baseline.get(layer_idx, {}) if p14_baseline else {}
        probe_acc_p14 = p14.get("probe_acc", None)
        H_base_p14 = p14.get("H_baseline", None)
        delta_H_p14 = p14.get("delta_H_max", None)
        overlap_p14 = p14.get("overlap_ratio", None)

        alignment_gain = None
        if delta_H_p14 is not None and delta_H_p14 > 0:
            alignment_gain = (delta_H_max - delta_H_p14) / delta_H_p14
        elif delta_H_p14 is not None and delta_H_max > 0:
            alignment_gain = float("inf")

        elapsed = time.time() - t_layer
        log(f"  LoRA H_base={baseline['H']:.3f} C_base={baseline['C']:.3f}")
        log(f"  H(+2.0)={alpha_metrics[2.0]['H']:.3f} (Δ={delta_H_plus:+.3f}) "
            f"H(-2.0)={alpha_metrics[-2.0]['H']:.3f} (Δ={delta_H_minus:+.3f})")
        log(f"  C_drop_max={C_drop_max:.3f} overlap={overlap:.4f} ({elapsed:.0f}s)")

        if p14_baseline and layer_idx in p14_baseline:
            if alignment_gain is not None:
                log(f"  vs P14(base): acc {p14.get('probe_acc',0):.4f}→{probe['accuracy']:.4f} "
                    f"ΔH {p14.get('delta_H_max',0):.4f}→{delta_H_max:.4f} "
                    f"overlap {p14.get('overlap_ratio',0):.4f}→{overlap:.4f} "
                    f"(gain={alignment_gain:+.3f})")
            else:
                log(f"  vs P14(base): acc {p14.get('probe_acc',0):.4f}→{probe['accuracy']:.4f} "
                    f"ΔH {p14.get('delta_H_max',0):.4f}→{delta_H_max:.4f} "
                    f"overlap {p14.get('overlap_ratio',0):.4f}→{overlap:.4f} "
                    f"(gain=N/A)")

        layer_results.append({
            "layer": layer_idx,
            "probe_acc": probe["accuracy"],
            "H_baseline": baseline["H"], "C_baseline": baseline["C"],
            "H_plus": alpha_metrics[2.0]["H"],
            "H_minus": alpha_metrics[-2.0]["H"],
            "delta_H_plus": round(delta_H_plus, 4),
            "delta_H_minus": round(delta_H_minus, 4),
            "delta_H_max": round(delta_H_max, 4),
            "C_drop_max": round(C_drop_max, 4),
            "overlap_ratio": round(overlap, 4),
            "p14_baseline": p14,
            "alignment_gain": round(alignment_gain, 4) if alignment_gain is not None else None,
        })

    log(f"\n{'='*64}")
    log(f"[Summary] LoRA Geometry Analysis")
    log(f"{'='*64}")
    header = f"  {'Layer':>5s}  {'Acc':>6s}  {'H_base':>6s}  "
    if p14_baseline:
        header += f"{'ΔH_P14':>7s}  {'ΔH_LoRA':>7s}  {'Gain':>6s}  "
    else:
        header += f"{'ΔH_max':>7s}  "
    header += f"{'C_drop':>7s}"
    log(header)
    log(f"  {'─'*5}  {'─'*6}  {'─'*6}  " + f"{'─'*7}  {'─'*7}  {'─'*6}  {'─'*7}")

    best_gain = None
    best_gain_layer = None
    any_improvement = False

    for r in layer_results:
        if p14_baseline:
            gain = r.get("alignment_gain", 0) or 0
            p14_dh = r.get("p14_baseline", {}).get("delta_H_max", 0)
            log(f"  {r['layer']:5d}  {r['probe_acc']:6.4f}  {r['H_baseline']:6.3f}  "
                f"{p14_dh:+7.4f}  {r['delta_H_max']:+7.4f}  {gain:+6.3f}  "
                f"{r['C_drop_max']:+7.4f}")
            if gain > 0:
                any_improvement = True
            if best_gain is None or gain > best_gain:
                best_gain = gain
                best_gain_layer = r["layer"]
        else:
            log(f"  {r['layer']:5d}  {r['probe_acc']:6.4f}  {r['H_baseline']:6.3f}  "
                f"{r['delta_H_max']:+7.4f}  {r['C_drop_max']:+7.4f}")

    if p14_baseline and any_improvement:
        log(f"\n  *** P16.2 CONFIRMED: LoRA increases steering effect at layer(s) above ***")
        log(f"  *** Best alignment gain at layer {best_gain_layer}: {best_gain:+.3f} ***")
    elif p14_baseline:
        log(f"\n  *** P16.2 REJECTED: LoRA does NOT increase w_probe steering effect ***")
        log(f"  *** LoRA bridges K->D through mechanism OTHER than K<->D subspace alignment ***")

    elapsed = time.time() - t0
    log(f"\nP16 Complete. ({elapsed:.0f}s, {elapsed/60:.1f} min)")

    summary = {
        "layers_tested": layers, "alphas": test_alphas,
        "p14_baseline_available": p14_baseline is not None,
        "any_alignment_gain": any_improvement,
        "best_gain": best_gain, "best_gain_layer": best_gain_layer,
        "layer_results": layer_results,
        "time_s": round(elapsed, 1),
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nResults saved to {RESULTS_DIR}/results.json")


if __name__ == "__main__":
    main()