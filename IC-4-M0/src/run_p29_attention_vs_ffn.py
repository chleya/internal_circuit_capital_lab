"""
P29: Attention vs FFN — Where Does L3 First-Mention Encoding Live?
====================================================================
P28 discovered: the FIRST occurrence of a causal token always encodes at L3,
with ~82% causal impact. Subsequent occurrences are "reconciled" not re-encoded.

But WHERE within L3 does this encoding happen — through Attention or FFN?

P29 answers by splitting per-layer ablation into two separate hooks:
  1. self_attn hook: zero out attention contribution at token position
  2. mlp hook:       zero out FFN contribution at token position

If one sub-module's ablation causes a much larger logprob shift, that's
where the first-mention causal information flows.

Per Qwen2.5 architecture:
  layer.self_attn → attention output → residual add
  layer.mlp       → FFN output       → residual add

Usage:
  python src/run_p29_attention_vs_ffn.py
"""

import os, sys, time, json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_builder import load_jsonl

RESULTS_DIR = "results_p29_attention_vs_ffn"
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


def logprob_with_submodule_hooks(model, tokenizer, prompt, response, device,
                                  layer_attn_hooks, layer_mlp_hooks):
    full_text = f"{prompt} {response}"
    enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"].to(device)
    prompt_enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    prompt_len = prompt_enc["input_ids"].shape[1]

    handles = []
    for layer_idx, layer in enumerate(model.model.layers):
        if layer_idx in layer_attn_hooks:
            handles.append(layer.self_attn.register_forward_hook(layer_attn_hooks[layer_idx]))
        if layer_idx in layer_mlp_hooks:
            handles.append(layer.mlp.register_forward_hook(layer_mlp_hooks[layer_idx]))

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
               attn_hooks=None, mlp_hooks=None):
    if attn_hooks is None:
        attn_hooks = {}
    if mlp_hooks is None:
        mlp_hooks = {}
    lp_pos = logprob_with_submodule_hooks(model, tokenizer, prompt, pos_response,
                                           device, attn_hooks, mlp_hooks)
    lp_neg = logprob_with_submodule_hooks(model, tokenizer, prompt, neg_response,
                                           device, attn_hooks, mlp_hooks)
    return lp_pos - lp_neg


def main():
    log("=" * 64)
    log("P29: Attention vs FFN — L3 First-Mention Encoding Source")
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

    log("\n[Step 2] Loading & identifying hallucinated samples...")
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
        log(f"  Sample {idx:>2d}: lp_diff={lp_diff:+.4f}  hall={'YES' if is_hall else 'no'}")
        if is_hall:
            pos, tok = find_first_token_position(prompt, tokenizer, CAUSAL_TOKEN)
            if pos:
                targets.append((idx, sample, lp_diff, prompt, pos, tok))

    log(f"\n  Targets: {len(targets)} samples with '{CAUSAL_TOKEN}'")
    for sidx, sample, base, prompt, pos, tok in targets:
        snipped = prompt[max(0, pos[0]-30):pos[-1]+30]
        log(f"    #{sidx} [{sample.get('_position','?'):5s}] pos={pos} \"...{snipped}...\"")

    total_passes = len(targets) * 2 * N_LAYERS
    log(f"\n  Total: {len(targets)} samples × 2 sub-modules × {N_LAYERS} layers = {total_passes} passes")

    log(f"\n[Step 3] Per-layer ablation — self_attn vs mlp...")
    all_results = []

    for sidx, sample, base_diff, prompt_text, positions, tok_pieces in targets:
        pos_resp = sample.get("positive_response", "")
        neg_resp = sample.get("negative_response", "")
        pos_set = set(positions)

        log(f"\n{'='*56}")
        log(f"Sample {sidx} [{sample.get('_position','?')}] base={base_diff:+.4f}")
        log(f"  first '{CAUSAL_TOKEN}' at pos={positions}")

        hook_fn_zero = build_position_hook(pos_set, SCALE_ZERO)

        for sub_name, hook_dict_fn in [("self_attn", lambda l: {l: hook_fn_zero}),
                                        ("mlp", lambda l: {l: hook_fn_zero})]:
            log(f"\n  --- {sub_name} ablation ---")
            curve = []
            for layer_idx in range(N_LAYERS):
                if sub_name == "self_attn":
                    ablated = compute_lp(model, tokenizer, prompt_text,
                                         pos_resp, neg_resp, device,
                                         attn_hooks=hook_dict_fn(layer_idx))
                else:
                    ablated = compute_lp(model, tokenizer, prompt_text,
                                         pos_resp, neg_resp, device,
                                         mlp_hooks=hook_dict_fn(layer_idx))
                delta = base_diff - ablated
                curve.append(round(delta, 6))

            peak = max(curve)
            peak_layer = curve.index(peak)
            log(f"    Peak: L{peak_layer} Δ={peak:+.4f}")

            ranked = sorted(enumerate(curve), key=lambda x: -x[1])[:5]
            for li, d in ranked:
                bar = "█" * max(1, int(d * 40 / max(0.01, peak)))
                log(f"      L{li:>2d}: {d:+.4f} {bar}")

            all_results.append({
                "sample_idx": sidx,
                "position_label": sample.get("_position", "?"),
                "sub_module": sub_name,
                "positions": positions,
                "baseline_lp_diff": round(base_diff, 6),
                "curve": curve,
                "peak_layer": peak_layer,
                "peak_delta": round(peak, 6),
                "l3_delta": curve[3] if len(curve) > 3 else 0,
            })

    elapsed = time.time() - t0

    log(f"\n{'=' * 64}")
    log(f"[Summary] P29 Attention vs FFN")
    log(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    log(f"\n  === L3 HEAD-TO-HEAD ===")
    for sidx in sorted(set(r["sample_idx"] for r in all_results)):
        sample_r = [r for r in all_results if r["sample_idx"] == sidx]
        attn_r = [r for r in sample_r if r["sub_module"] == "self_attn"][0]
        mlp_r = [r for r in sample_r if r["sub_module"] == "mlp"][0]
        log(f"\n  Sample {sidx} [{attn_r['position_label']}]:")
        log(f"    self_attn:  L3 Δ={attn_r['curve'][3]:+.4f}  peak=L{attn_r['peak_layer']} Δ={attn_r['peak_delta']:+.4f}")
        log(f"    mlp:        L3 Δ={mlp_r['curve'][3]:+.4f}  peak=L{mlp_r['peak_layer']} Δ={mlp_r['peak_delta']:+.4f}")
        l3_ratio = abs(attn_r['curve'][3]) / max(0.001, abs(attn_r['curve'][3]) + abs(mlp_r['curve'][3]))
        log(f"    attn_share @ L3 = {l3_ratio:.1%}")

    log(f"\n  === CROSS-SAMPLE AGGREGATE ===")
    for sub in ["self_attn", "mlp"]:
        sub_r = [r for r in all_results if r["sub_module"] == sub]
        mean_peak_layer = sum(r["peak_layer"] for r in sub_r) / len(sub_r)
        mean_peak_delta = sum(r["peak_delta"] for r in sub_r) / len(sub_r)
        mean_l3 = sum(r["l3_delta"] for r in sub_r) / len(sub_r)
        log(f"  {sub:>10s}: mean_peak=L{mean_peak_layer:.1f}  "
            f"mean_Δ={mean_peak_delta:+.4f}  mean_L3_Δ={mean_l3:+.4f}")

    attn_l3_vals = [r["curve"][3] for r in all_results if r["sub_module"] == "self_attn"]
    mlp_l3_vals = [r["curve"][3] for r in all_results if r["sub_module"] == "mlp"]
    if sum(a > m for a, m in zip(attn_l3_vals, mlp_l3_vals)) == len(attn_l3_vals):
        log(f"\n  CONCLUSION: self_attn DOMINATES at L3 (all samples)")
    elif sum(m > a for a, m in zip(attn_l3_vals, mlp_l3_vals)) == len(attn_l3_vals):
        log(f"\n  CONCLUSION: mlp DOMINATES at L3 (all samples)")
    else:
        log(f"\n  CONCLUSION: MIXED — some samples attn-dominant, some mlp-dominant")

    summary_data = {
        "config": {"method": "sub-module-separated per-layer ablation",
                   "causal_token": CAUSAL_TOKEN,
                   "sub_modules": ["self_attn", "mlp"],
                   "scale": SCALE_ZERO, "n_layers": N_LAYERS},
        "n_results": len(all_results),
        "time_s": round(elapsed, 1),
        "results": all_results,
        "l3_comparison": {
            "self_attn_l3": attn_l3_vals,
            "mlp_l3": mlp_l3_vals,
            "attn_dominant_count": sum(1 for a, m in zip(attn_l3_vals, mlp_l3_vals) if abs(a) > abs(m)),
            "mlp_dominant_count": sum(1 for a, m in zip(attn_l3_vals, mlp_l3_vals) if abs(m) > abs(a)),
        },
    }

    with open(os.path.join(RESULTS_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    log(f"\nResults saved to {RESULTS_DIR}/results.json")
    log(f"\nP29 Complete.")


if __name__ == "__main__":
    main()