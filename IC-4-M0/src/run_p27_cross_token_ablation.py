"""
P27: Cross-Token Generalization — Extending P26 beyond 'funding'
=================================================================
P26 found: 'funding' peaks at L3 across all hallucinated samples.
This raises a critical question: Is L3 the UNIVERSAL semantic-encoding gate,
or specific to 'funding'?

P27 answers this by replicating P26's per-layer ablation protocol on
DIFFERENT causal tokens:
  - Token A: 'r_and_d_spend' (JetCircuit hallucinated samples)
  - Token B: 'total funding' (BoltStream hallucinated samples, for cross-validation)
  - Token C: Any other high-importance tokens found

The key comparison: does L3 dominate across ALL tokens, or does each
token class have its own peak layer?

Usage:
  python src/run_p27_cross_token_ablation.py
"""

import os, sys, time, json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_builder import load_jsonl

RESULTS_DIR = "results_p27_cross_token_ablation"
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, "run_log.txt")

N_LAYERS = 24
SCALE_ZERO = 0.0

CAUSAL_TOKENS = [
    "r_and_d_spend",
    "total funding",
    "funding",
]


def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def find_token_position(prompt, tokenizer, target):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    for variant in [target, " " + target, " " + target + " "]:
        variant_tokens = tokenizer.tokenize(variant)
        n = len(variant_tokens)
        if n == 0:
            continue
        for i in range(len(tokens) - n + 1):
            if tokens[i:i + n] == variant_tokens:
                return list(range(i, i + n)), variant_tokens
    return None, None


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


def per_token_ablation(model, tokenizer, sample, device, sample_idx, causal_token):
    ctx = sample.get("context", "")
    q = sample.get("question", "")
    prompt = f"{ctx}\n\nQuestion: {q}\nAnswer:"
    pos_resp = sample.get("positive_response", "")
    neg_resp = sample.get("negative_response", "")

    token_positions, tokenized_pieces = find_token_position(prompt, tokenizer, causal_token)
    if not token_positions:
        return None

    pos_set = set(token_positions)
    log(f"  Sample {sample_idx}: '{causal_token}' → tokens={tokenized_pieces} positions={token_positions}")

    base = compute_lp(model, tokenizer, prompt, pos_resp, neg_resp, device)
    base_diff = base["lp_diff"]
    log(f"    Baseline lp_diff={base_diff:+.4f}")

    results = {
        "sample_idx": sample_idx,
        "causal_token": causal_token,
        "tokenized_as": tokenized_pieces,
        "position": sample.get("_position", "?"),
        "question": q[:80],
        "token_positions": token_positions,
        "baseline": base,
        "single_layer": {},
        "cumulative_forward": {},
        "cumulative_backward": {},
    }

    hook_fn_zero = build_single_layer_hook(pos_set, SCALE_ZERO)

    for layer_idx in range(N_LAYERS):
        result = compute_lp(model, tokenizer, prompt, pos_resp, neg_resp,
                            device, {layer_idx: hook_fn_zero})
        delta = base_diff - result["lp_diff"]
        results["single_layer"][str(layer_idx)] = {
            "lp_diff": round(result["lp_diff"], 6),
            "delta": round(delta, 6),
        }

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
    log("P27: Cross-Token Generalization — Per-Layer Causal Token Ablation")
    log(f"Tokens: {CAUSAL_TOKENS}")
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
                s["_source_file"] = f"test_{pos}_s0.jsonl"
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
            hallucinated.append({
                "idx": idx,
                "sample": sample,
                "base_diff": result["lp_diff"],
            })
        log(f"    Sample {idx:>2d}: lp_diff={result['lp_diff']:+.4f}  "
            f"hall={'YES' if is_hall else 'no'}")

    log(f"\n  Hallucinated: {len(hallucinated)} samples")
    for h in hallucinated:
        log(f"    #{h['idx']:>2d} [{h['sample'].get('_position','?'):5s}] "
            f"lp_diff={h['base_diff']:+.4f}  "
            f"Q: {h['sample'].get('question','')[:70]}")

    log(f"\n[Step 4] Token-by-token analysis: which causal tokens exist in each sample?")
    token_sample_map = {}
    for token in CAUSAL_TOKENS:
        token_sample_map[token] = []
        for h in hallucinated:
            ctx = h["sample"].get("context", "")
            q = h["sample"].get("question", "")
            prompt = f"{ctx}\n\nQuestion: {q}\nAnswer:"
            pos, tok_pieces = find_token_position(prompt, tokenizer, token)
            if pos:
                token_sample_map[token].append((h["idx"], h["sample"], pos, tok_pieces))
        log(f"  '{token}': found in {len(token_sample_map[token])} hallucinated samples")
        for sidx, _, pos, tok_pieces in token_sample_map[token]:
            log(f"      Sample {sidx}: tokenized={tok_pieces} positions={pos}")

    log(f"\n[Step 5] Per-layer ablation — all tokens × all matching samples")
    all_comparisons = []
    for token in CAUSAL_TOKENS:
        comparisons_this_token = len(token_sample_map[token])
        passes = comparisons_this_token * 72
        log(f"\n  Token '{token}': {comparisons_this_token} matches × 72 passes = {passes} forward passes")

    all_results = []
    for token in CAUSAL_TOKENS:
        for sidx, sample, positions, tok_pieces in token_sample_map[token]:
            log(f"\n{'='*48}")
            log(f"[Token: '{token}'] Sample {sidx} [{sample.get('_position','?')}]")
            result = per_token_ablation(model, tokenizer, sample, device, sidx, token)
            if result:
                all_results.append(result)

    elapsed = time.time() - t0

    log(f"\n{'=' * 64}")
    log(f"[Summary] P27 Cross-Token Per-Layer Ablation")
    log(f"  Samples analyzed: {len(all_results)}")
    log(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    token_summaries = {}
    for token in CAUSAL_TOKENS:
        token_results = [r for r in all_results if r["causal_token"] == token]
        if not token_results:
            continue
        peak_layers = [r["summary"]["best_single_layer"] for r in token_results]
        peak_deltas = [r["summary"]["best_single_delta"] for r in token_results]
        fwd_layers = [r["summary"]["best_forward_layer"] for r in token_results]
        bwd_layers = [r["summary"]["best_backward_layer"] for r in token_results]
        token_summaries[token] = {
            "n_samples": len(token_results),
            "single_peak_layers": peak_layers,
            "single_mean_peak": sum(peak_layers) / len(peak_layers),
            "single_max_delta": max(peak_deltas),
            "single_mean_delta": sum(peak_deltas) / len(peak_deltas),
            "fwd_peak_layers": fwd_layers,
            "bwd_peak_layers": bwd_layers,
        }
        log(f"\n  Token '{token}' [{len(token_results)} samples]:")
        log(f"    Single-layer peaks: {peak_layers}  mean={sum(peak_layers)/len(peak_layers):.1f}")
        log(f"    Single-layer deltas: max={max(peak_deltas):+.4f}  mean={sum(peak_deltas)/len(peak_deltas):+.4f}")
        log(f"    Forward peaks: {fwd_layers}  mean={sum(fwd_layers)/len(fwd_layers):.1f}")
        log(f"    Backward peaks: {bwd_layers}  mean={sum(bwd_layers)/len(bwd_layers):.1f}")

    log(f"\n  === CROSS-TOKEN COMPARISON ===")
    for token, ts in token_summaries.items():
        log(f"  '{token}': single-layer peak mean={ts['single_mean_peak']:.1f} "
            f"mean_delta={ts['single_mean_delta']:+.4f}  max_delta={ts['single_max_delta']:+.4f}")

    peaking_tokens = [(t, ts["single_mean_peak"]) for t, ts in token_summaries.items()]
    if len(peaking_tokens) > 1:
        all_same = all(abs(p[1] - peaking_tokens[0][1]) < 1.0 for p in peaking_tokens)
        log(f"\n  L3 vs other layers across tokens:")
        for token, ts in token_summaries.items():
            l3_count = sum(1 for l in ts["single_peak_layers"] if abs(l - 3) <= 1)
            log(f"    '{token}': {l3_count}/{ts['n_samples']} samples peak near L3 (L2-L4)")
        if all_same:
            log(f"\n  CONCLUSION: L3 is the UNIVERSAL semantic encoding gate across '{', '.join(CAUSAL_TOKENS)}'")
        else:
            log(f"\n  CONCLUSION: Peak layer varies across tokens — NOT universal")

    summary_data = {
        "config": {
            "method": "cross-token per-layer causal token activation ablation",
            "causal_tokens": CAUSAL_TOKENS,
            "scale": SCALE_ZERO,
            "n_layers": N_LAYERS,
            "conditions": ["single_layer", "cumulative_forward", "cumulative_backward"],
        },
        "n_total_results": len(all_results),
        "time_s": round(elapsed, 1),
        "token_summaries": {
            token: {
                "n_samples": ts["n_samples"],
                "single_peak_layers": ts["single_peak_layers"],
                "single_mean_peak": round(ts["single_mean_peak"], 2),
                "single_max_delta": round(ts["single_max_delta"], 6),
                "single_mean_delta": round(ts["single_mean_delta"], 6),
                "fwd_peak_layers": ts["fwd_peak_layers"],
                "bwd_peak_layers": ts["bwd_peak_layers"],
            }
            for token, ts in token_summaries.items()
        },
        "results": [],
    }

    for r in all_results:
        single_curve = [r["single_layer"][str(l)]["delta"] for l in range(N_LAYERS)]
        peak_single = max(single_curve)
        peak_single_layer = single_curve.index(peak_single)
        fwd_curve = [r["cumulative_forward"][str(l)]["delta"] for l in range(N_LAYERS)]
        bwd_curve = [r["cumulative_backward"][str(l)]["delta"] for l in range(N_LAYERS)]

        summary_data["results"].append({
            "sample_idx": r["sample_idx"],
            "causal_token": r["causal_token"],
            "tokenized_as": r["tokenized_as"],
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

        log(f"\n  Sample {r['sample_idx']} [{r['causal_token']} @ {r['position']}]:")
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
    log(f"\nP27 Complete.")


if __name__ == "__main__":
    main()