"""
P30: Combined Sub-Module Ablation — L0 vs L3 Head-to-Head
===========================================================
P26-P28: Full-layer ablation → L3 peak (Δ=+0.41)
P29:     Single sub-module   → L0 peak (self_attn +0.19, mlp +0.33)

Gap: What happens when we ablate BOTH self_attn AND mlp simultaneously
at each layer? This removes the full sub-module contribution while
preserving only the embedding residual — the cleanest measure of where
information enters the transformer.

P30 answers: Is L0 > L3 when both sub-modules are zeroed together?

Total: 3 samples × 24 layers = 72 passes (very fast)

Usage:
  python src/run_p30_combined_submodule_ablation.py
"""

import os, sys, time, json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_builder import load_jsonl

RESULTS_DIR = "results_p30_combined_submodule"
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, "run_log.txt")

N_LAYERS = 24
SCALE_ZERO = 0.0
CAUSAL_TOKEN = "funding"


def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def find_first_token_position(prompt, tokenizer, target):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())
    for variant in [target, " " + target]:
        variant_tokens = tokenizer.tokenize(variant)
        n = len(variant_tokens)
        if n == 0:
            continue
        for i in range(len(tokens) - n + 1):
            if tokens[i:i + n] == variant_tokens:
                return list(range(i, i + n)), variant_tokens
    return None, None


def build_position_hook(pos_set, scale):
    def hook(module, input, output):
        if isinstance(output, (tuple, list)):
            hidden = output[0].clone()
            rest = output[1:]
        else:
            hidden = output.clone()
            rest = None
        if hidden.dim() == 3:
            for pos in pos_set:
                if pos < hidden.shape[1]:
                    hidden[0, pos, :] *= scale
        elif hidden.dim() == 2:
            for pos in pos_set:
                if pos < hidden.shape[0]:
                    hidden[pos, :] *= scale
        if rest is None:
            return hidden
        if isinstance(output, tuple):
            return (hidden,) + rest
        return [hidden] + rest
    return hook


def logprob_with_combined_hooks(model, tokenizer, prompt, response, device,
                                 combined_hooks):
    full_text = f"{prompt} {response}"
    enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"].to(device)
    prompt_enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    prompt_len = prompt_enc["input_ids"].shape[1]

    handles = []
    for layer_idx, (attn_hook, mlp_hook) in combined_hooks.items():
        layer = model.model.layers[layer_idx]
        handles.append(layer.self_attn.register_forward_hook(attn_hook))
        handles.append(layer.mlp.register_forward_hook(mlp_hook))

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
               combined_hooks=None):
    if combined_hooks is None:
        combined_hooks = {}
    lp_pos = logprob_with_combined_hooks(model, tokenizer, prompt, pos_response,
                                          device, combined_hooks)
    lp_neg = logprob_with_combined_hooks(model, tokenizer, prompt, neg_response,
                                          device, combined_hooks)
    return lp_pos - lp_neg


def main():
    log("=" * 64)
    log("P30: Combined Sub-Module Ablation — attn+mlp simultaneously")
    log(f"     → Finding where '{CAUSAL_TOKEN}' information truly enters")
    log("=" * 64)
    t0 = time.time()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    log("\n[Step 1] Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="cpu")
    model.eval()
    device = next(model.parameters()).device

    log("\n[Step 2] Identifying hallucinated + first-funding samples...")
    pos_dir = os.path.join(base_dir, "data_position_sensitivity", "s0")
    test_samples = []
    for pos in ["early", "mid", "late"]:
        path = os.path.join(pos_dir, f"test_{pos}_s0.jsonl")
        if os.path.exists(path):
            for s in load_jsonl(path)[:10]:
                s["_position"] = pos
                test_samples.append(s)

    targets = []
    for idx, sample in enumerate(test_samples):
        ctx = sample.get("context", "")
        q = sample.get("question", "")
        prompt = f"{ctx}\n\nQuestion: {q}\nAnswer:"
        lp_diff = compute_lp(model, tokenizer, prompt,
                             sample.get("positive_response", ""),
                             sample.get("negative_response", ""), device)
        is_hall = (lp_diff > 0 and sample.get("answerability") == "unanswerable")
        if is_hall:
            pos, tok = find_first_token_position(prompt, tokenizer, CAUSAL_TOKEN)
            if pos:
                targets.append((idx, sample, lp_diff, prompt, pos, tok))
        log(f"  Sample {idx:>2d}: hall={'YES' if is_hall else 'no '}  "
            f"funding={'YES' if is_hall and find_first_token_position(f'{ctx}\n\nQuestion: {q}\nAnswer:', tokenizer, CAUSAL_TOKEN)[0] else 'no '}")

    log(f"\n  Targets: {len(targets)}")
    for sidx, _, base, _, pos, tok in targets:
        log(f"    #{sidx} pos={pos} base={base:+.4f}")

    log(f"\n[Step 3] Combined sub-module ablation: 24 layers × {len(targets)} samples = {24*len(targets)} passes...")

    all_results = []
    hook_fn_zero = build_position_hook

    for sidx, sample, base_diff, prompt_text, positions, tok_pieces in targets:
        pos_resp = sample.get("positive_response", "")
        neg_resp = sample.get("negative_response", "")
        pos_set = set(positions)
        position_label = sample.get("_position", "?")

        log(f"\n{'='*48}")
        log(f"Sample {sidx} [{position_label}] base={base_diff:+.4f}  pos={positions}")

        curve = []
        for layer_idx in range(N_LAYERS):
            hooks = {
                layer_idx: (hook_fn_zero(pos_set, SCALE_ZERO),
                            hook_fn_zero(pos_set, SCALE_ZERO))
            }
            ablated = compute_lp(model, tokenizer, prompt_text,
                                 pos_resp, neg_resp, device,
                                 combined_hooks=hooks)
            delta = base_diff - ablated
            curve.append(round(delta, 6))

        peak = max(curve)
        peak_layer = curve.index(peak)
        l0_val = curve[0]
        l3_val = curve[3]

        log(f"    L0 (combined):  Δ={l0_val:+.4f}")
        log(f"    L3 (combined):  Δ={l3_val:+.4f}")
        log(f"    Peak:           L{peak_layer} Δ={peak:+.4f}")

        ranked = sorted(enumerate(curve), key=lambda x: -x[1])[:6]
        for li, d in ranked:
            bar = "█" * max(1, int(d * 40 / max(0.01, peak)))
            log(f"      L{li:>2d}: {d:+.4f} {bar} {'← PEAK' if li == peak_layer else ''}")

        all_results.append({
            "sample_idx": sidx,
            "position_label": position_label,
            "positions": positions,
            "baseline_lp_diff": round(base_diff, 6),
            "curve": curve,
            "peak_layer": peak_layer,
            "peak_delta": round(peak, 6),
            "l0_delta": l0_val,
            "l3_delta": l3_val,
        })

    elapsed = time.time() - t0
    log(f"\n{'=' * 64}")
    log(f"[Summary] P30 Combined Sub-Module Ablation")
    log(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    log(f"\n  === L0 vs L3 HEAD-TO-HEAD (combined attn+mlp) ===")
    for r in all_results:
        winner = "L0 >> L3" if r["l0_delta"] > r["l3_delta"] * 2 else \
                 "L0 > L3" if r["l0_delta"] > r["l3_delta"] else \
                 "L3 > L0" if r["l3_delta"] > r["l0_delta"] else "L0 ≈ L3"
        log(f"  Sample {r['sample_idx']} [{r['position_label']}]: "
            f"L0={r['l0_delta']:+.4f}  L3={r['l3_delta']:+.4f}  → {winner}")

    mean_l0 = sum(r["l0_delta"] for r in all_results) / len(all_results)
    mean_l3 = sum(r["l3_delta"] for r in all_results) / len(all_results)
    mean_ratio = mean_l0 / max(0.001, mean_l3)

    log(f"\n  Mean L0: {mean_l0:+.4f}  Mean L3: {mean_l3:+.4f}  Ratio: {mean_ratio:.1f}x")

    if mean_ratio > 1.5:
        log(f"\n  CONCLUSION: L0 dominates L3 by {mean_ratio:.1f}x")
        log(f"  → Anti-hallucination intervention should target L0 (first layer)")
    elif mean_ratio > 0.67:
        log(f"\n  CONCLUSION: L0 and L3 are comparable")
    else:
        log(f"\n  CONCLUSION: L3 dominates L0")

    peaks = [r["peak_layer"] for r in all_results]
    log(f"  Peak layers: {peaks}  mean={sum(peaks)/len(peaks):.1f}")

    summary_data = {
        "config": {
            "method": "combined sub-module ablation (self_attn + mlp simultaneously)",
            "causal_token": CAUSAL_TOKEN, "scale": SCALE_ZERO, "n_layers": N_LAYERS,
        },
        "n_results": len(all_results),
        "time_s": round(elapsed, 1),
        "head_to_head": {
            "mean_l0": round(mean_l0, 6),
            "mean_l3": round(mean_l3, 6),
            "l0_l3_ratio": round(mean_ratio, 2),
            "l0_wins": sum(1 for r in all_results if r["l0_delta"] > r["l3_delta"]),
            "l3_wins": sum(1 for r in all_results if r["l3_delta"] > r["l0_delta"]),
        },
        "results": all_results,
    }

    with open(os.path.join(RESULTS_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    log(f"\nResults saved to {RESULTS_DIR}/results.json")
    log(f"\nP30 Complete.")


if __name__ == "__main__":
    main()