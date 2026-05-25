"""
P26: Per-Layer Causal Token Ablation — Tracing the Information Dispersion Curve
================================================================================
P25 found: early@0.5(funding) > deep@0.0(funding+combo). This is paradoxical because
P17-P18 showed deep-layer q_proj is the routing core.

P26 resolves this by ablating "funding" at EACH individual layer, measuring:
  1. Per-layer causal importance: which single layer has max impact?
  2. Cumulative forward (0..L): how does effect accumulate as we ablate more early layers?
  3. Cumulative backward (L..23): how much residual effect in deep layers?

These three curves together reveal the INFORMATION DISPERSION PROFILE:
  - At which layer does "funding"'s influence peak?
  - How quickly does it disperse to other tokens (making individual ablation useless)?
  - At what layer does the accumulated effect saturate?

Usage:
  python src/run_p26_per_layer_ablation.py
"""

import os, sys, time, json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_builder import load_jsonl

RESULTS_DIR = "results_p26_per_layer_ablation"
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, "run_log.txt")

N_LAYERS = 24
CAUSAL_TOKEN = "funding"
SCALE_ZERO = 0.0


def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def find_token_position(prompt, tokenizer, target):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    for variant in [target, " " + target]:
        variant_tokens = tokenizer.tokenize(variant)
        n = len(variant_tokens)
        for i in range(len(tokens) - n + 1):
            if tokens[i:i + n] == variant_tokens:
                return list(range(i, i + n))
    return None


def build_single_layer_hook(pos_set, scale):
    def hook(module, input, output):
        if isinstance(output, (tuple, list)):
            hidden_states = output[0].clone()
            rest = output[1:]
        else:
            hidden_states = output.clone()
            rest = None

        if hidden_states.dim() == 3:
            for pos in pos_set:
                if pos < hidden_states.shape[1]:
                    hidden_states[0, pos, :] *= scale
        elif hidden_states.dim() == 2:
            for pos in pos_set:
                if pos < hidden_states.shape[0]:
                    hidden_states[pos, :] *= scale

        if rest is None:
            return hidden_states
        if isinstance(output, tuple):
            return (hidden_states,) + rest
        return [hidden_states] + rest
    return hook


def logprob_with_layer_hooks(model, tokenizer, prompt, response, device,
                              layer_hooks):
    full_text = f"{prompt} {response}"
    enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"].to(device)
    prompt_enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    prompt_len = prompt_enc["input_ids"].shape[1]

    handles = []
    for layer_idx, layer in enumerate(model.model.layers):
        if layer_idx in layer_hooks:
            h = layer.register_forward_hook(layer_hooks[layer_idx])
            handles.append(h)

    try:
        labels = input_ids.clone()
        labels[0, :prompt_len] = -100
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)
        return -outputs.loss.item()
    finally:
        for h in handles:
            h.remove()


def compute_lp(model, tokenizer, prompt, pos_response, neg_response, device,
               layer_hooks=None):
    if layer_hooks is None:
        layer_hooks = {}
    lp_pos = logprob_with_layer_hooks(model, tokenizer, prompt, pos_response,
                                       device, layer_hooks)
    lp_neg = logprob_with_layer_hooks(model, tokenizer, prompt, neg_response,
                                       device, layer_hooks)
    return {
        "lp_pos": lp_pos, "lp_neg": lp_neg,
        "lp_diff": lp_pos - lp_neg,
    }


def per_layer_ablation(model, tokenizer, sample, device, sample_idx):
    ctx = sample.get("context", "")
    q = sample.get("question", "")
    prompt = f"{ctx}\n\nQuestion: {q}\nAnswer:"
    pos_resp = sample.get("positive_response", "")
    neg_resp = sample.get("negative_response", "")

    token_positions = find_token_position(prompt, tokenizer, CAUSAL_TOKEN)
    if not token_positions:
        log(f"  Sample {sample_idx}: '{CAUSAL_TOKEN}' not found in prompt")
        return None

    pos_set = set(token_positions)
    pos_str = f"positions={token_positions}"
    log(f"  Sample {sample_idx}: '{CAUSAL_TOKEN}' at {pos_str}")

    base = compute_lp(model, tokenizer, prompt, pos_resp, neg_resp, device)
    base_diff = base["lp_diff"]
    log(f"    Baseline lp_diff={base_diff:+.4f}")

    results = {
        "sample_idx": sample_idx,
        "position": sample.get("_position", "?"),
        "question": q[:80],
        "token_positions": token_positions,
        "baseline": base,
        "single_layer": {},
        "cumulative_forward": {},
        "cumulative_backward": {},
    }

    hook_fn_zero = build_single_layer_hook(pos_set, SCALE_ZERO)

    log(f"    [Single-layer ablation] 24 layers...")
    for layer_idx in range(N_LAYERS):
        result = compute_lp(model, tokenizer, prompt, pos_resp, neg_resp,
                            device, {layer_idx: hook_fn_zero})
        delta = base_diff - result["lp_diff"]
        results["single_layer"][str(layer_idx)] = {
            "lp_diff": round(result["lp_diff"], 6),
            "delta": round(delta, 6),
        }

    log(f"    [Cumulative forward ablation] layers 0..L...")
    for end_layer in range(N_LAYERS):
        hooks = {l: hook_fn_zero for l in range(0, end_layer + 1)}
        result = compute_lp(model, tokenizer, prompt, pos_resp, neg_resp,
                            device, hooks)
        delta = base_diff - result["lp_diff"]
        results["cumulative_forward"][str(end_layer)] = {
            "lp_diff": round(result["lp_diff"], 6),
            "delta": round(delta, 6),
            "n_layers_ablated": end_layer + 1,
        }

    log(f"    [Cumulative backward ablation] layers L..23...")
    for start_layer in range(N_LAYERS):
        hooks = {l: hook_fn_zero for l in range(start_layer, N_LAYERS)}
        result = compute_lp(model, tokenizer, prompt, pos_resp, neg_resp,
                            device, hooks)
        delta = base_diff - result["lp_diff"]
        results["cumulative_backward"][str(start_layer)] = {
            "lp_diff": round(result["lp_diff"], 6),
            "delta": round(delta, 6),
            "n_layers_ablated": N_LAYERS - start_layer,
        }

    best_single = max(results["single_layer"].items(),
                       key=lambda x: x[1]["delta"])
    best_forward = max(results["cumulative_forward"].items(),
                        key=lambda x: x[1]["delta"])
    best_backward = max(results["cumulative_backward"].items(),
                         key=lambda x: x[1]["delta"])

    log(f"    Best single layer:  L{best_single[0]}  "
        f"lp_diff={best_single[1]['lp_diff']:+.4f}  Δ={best_single[1]['delta']:+.4f}")
    log(f"    Best cumul forward: L{best_forward[0]}  "
        f"lp_diff={best_forward[1]['lp_diff']:+.4f}  Δ={best_forward[1]['delta']:+.4f}")
    log(f"    Best cumul backward: L{best_backward[0]}  "
        f"lp_diff={best_backward[1]['lp_diff']:+.4f}  Δ={best_backward[1]['delta']:+.4f}")

    results["summary"] = {
        "best_single_layer": int(best_single[0]),
        "best_single_delta": best_single[1]["delta"],
        "best_forward_layer": int(best_forward[0]),
        "best_forward_delta": best_forward[1]["delta"],
        "best_backward_layer": int(best_backward[0]),
        "best_backward_delta": best_backward[1]["delta"],
    }

    return results


def main():
    log("=" * 64)
    log("P26: Per-Layer Causal Token Activation Ablation")
    log("Tracing the information dispersion curve of 'funding'")
    log("=" * 64)
    t0 = time.time()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    log("\n[Step 1] Loading base model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="cpu")
    model.eval()
    device = next(model.parameters()).device
    log(f"  Device: {device}, Layers: {N_LAYERS}")

    log("\n[Step 2] Loading test samples...")
    pos_dir = os.path.join(base_dir, "data_position_sensitivity", "s0")
    test_samples = []
    for pos in ["early", "mid", "late"]:
        path = os.path.join(pos_dir, f"test_{pos}_s0.jsonl")
        if os.path.exists(path):
            for s in load_jsonl(path)[:10]:
                s["_position"] = pos
                test_samples.append(s)

    log(f"  Loaded {len(test_samples)} samples")

    log("\n[Step 3] Identifying hallucinated samples...")
    hallucinated = []
    for idx, sample in enumerate(test_samples):
        ctx = sample.get("context", "")
        q = sample.get("question", "")
        prompt = f"{ctx}\n\nQuestion: {q}\nAnswer:"
        result = compute_lp(model, tokenizer, prompt,
                            sample.get("positive_response", ""),
                            sample.get("negative_response", ""), device)
        is_hall = (result["lp_diff"] > 0 and
                   sample.get("answerability") == "unanswerable")
        if is_hall:
            hallucinated.append((idx, sample, result["lp_diff"]))
        log(f"    Sample {idx:>2d}: lp_diff={result['lp_diff']:+.4f}  "
            f"hall={'YES' if is_hall else 'no'}")

    log(f"\n  Hallucinated: {len(hallucinated)} samples")
    for idx, s, diff in hallucinated:
        log(f"    #{idx:>2d} [{s.get('_position','?'):5s}] lp_diff={diff:+.4f}  "
            f"Q: {s.get('question','')[:60]}")

    log(f"\n[Step 4] Per-layer ablation on each hallucinated sample...")
    log(f"  Total: {len(hallucinated)} samples × (24 single + 24 fwd + 24 bwd)")
    log(f"       = {len(hallucinated) * 72} forward passes\n")

    all_results = []
    for idx, sample, base_diff in hallucinated:
        log(f"\n{'='*48}")
        log(f"Sample {idx} [{sample.get('_position','?')}]"
            f" base={base_diff:+.4f}")
        result = per_layer_ablation(model, tokenizer, sample, device, idx)
        if result:
            all_results.append(result)

    elapsed = time.time() - t0

    log(f"\n{'=' * 64}")
    log(f"[Summary] P26 Per-Layer Ablation")
    log(f"  Samples analyzed: {len(all_results)}")
    log(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    summary_data = {
        "config": {
            "method": "per-layer causal token activation ablation",
            "causal_token": CAUSAL_TOKEN,
            "scale": SCALE_ZERO,
            "n_layers": N_LAYERS,
            "conditions": ["single_layer", "cumulative_forward", "cumulative_backward"],
        },
        "n_samples": len(all_results),
        "time_s": round(elapsed, 1),
        "results": [],
    }

    for r in all_results:
        single_curve = [r["single_layer"][str(l)]["delta"] for l in range(N_LAYERS)]
        fwd_curve = [r["cumulative_forward"][str(l)]["delta"] for l in range(N_LAYERS)]
        bwd_curve = [r["cumulative_backward"][str(l)]["delta"] for l in range(N_LAYERS)]

        peak_single = max(single_curve)
        peak_single_layer = single_curve.index(peak_single)

        summary_data["results"].append({
            "sample_idx": r["sample_idx"],
            "position": r["position"],
            "question": r["question"],
            "token_positions": r["token_positions"],
            "baseline_lp_diff": round(r["baseline"]["lp_diff"], 6),
            "single_layer_curve": [round(v, 6) for v in single_curve],
            "cumulative_forward_curve": [round(v, 6) for v in fwd_curve],
            "cumulative_backward_curve": [round(v, 6) for v in bwd_curve],
            "peak_single_layer": peak_single_layer,
            "peak_single_delta": round(peak_single, 6),
            "best_forward_layer": r["summary"]["best_forward_layer"],
            "best_forward_delta": round(r["summary"]["best_forward_delta"], 6),
            "best_backward_layer": r["summary"]["best_backward_layer"],
            "best_backward_delta": round(r["summary"]["best_backward_delta"], 6),
        })

        log(f"\n  Sample {r['sample_idx']} [{r['position']}]:")
        log(f"    Single-layer peak:      L{peak_single_layer}  Δ={peak_single:+.4f}")
        log(f"    Cumulative fwd peak:    L{r['summary']['best_forward_layer']}  "
            f"Δ={r['summary']['best_forward_delta']:+.4f}")
        log(f"    Cumulative bwd peak:    L{r['summary']['best_backward_layer']}  "
            f"Δ={r['summary']['best_backward_delta']:+.4f}")
        log(f"    Single-layer curve (top 5):")
        ranked = sorted(enumerate(single_curve), key=lambda x: -x[1])[:5]
        for layer_idx, delta in ranked:
            bar = "█" * max(1, int(delta * 40 / max(0.01, peak_single)))
            log(f"      L{layer_idx:>2d}: {delta:+.4f} {bar}")

    with open(os.path.join(RESULTS_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    log(f"\nResults saved to {RESULTS_DIR}/results.json")
    log(f"\nP26 Complete.")


if __name__ == "__main__":
    main()