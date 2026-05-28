"""
P28: Position Separation Verification — Same Token, Different Positions
========================================================================
P27 found: within Sample 4, "funding" at pos=7 (context) peaks at L3,
while "funding" at pos=27 (question as "total funding") peaks at L8.

Is this a one-sample anomaly, or a systematic position-dependent effect?

P28 answers this by:
  1. Finding ALL occurrences of "funding" in each hallucinated sample
  2. Running per-layer ablation on EACH occurrence separately
  3. Comparing peak layers by position (context vs question)

Hypothesis: "funding" in the CONTEXT sentence (early position) is processed
by L3 (concept encoding), while "funding" in the QUESTION text (late position)
requires deeper layers (integration with the query intent).

Usage:
  python src/run_p28_position_separation.py
"""

import os, sys, time, json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_builder import load_jsonl

RESULTS_DIR = "results_p28_position_separation"
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, "run_log.txt")

N_LAYERS = 24
SCALE_ZERO = 0.0
CAUSAL_TOKEN = "funding"


def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def find_all_token_positions(prompt, tokenizer, target):
    """Find ALL occurrences of target in prompt, not just the first one."""
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    all_positions = []
    for variant in [target, " " + target]:
        variant_tokens = tokenizer.tokenize(variant)
        n = len(variant_tokens)
        if n == 0:
            continue
        i = 0
        while i < len(tokens) - n + 1:
            if tokens[i:i + n] == variant_tokens:
                if all(i not in pset[0] for pset in all_positions):
                    all_positions.append((list(range(i, i + n)), variant_tokens[:]))
                i += n
            else:
                i += 1
    return all_positions


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


def per_position_ablation(model, tokenizer, sample, device, sample_idx,
                          occurrence_idx, positions, tok_pieces, prompt, prompt_text):
    pos_resp = sample.get("positive_response", "")
    neg_resp = sample.get("negative_response", "")

    pos_set = set(positions)
    log(f"  Occurrence {occurrence_idx}: pos={positions} tokens={tok_pieces}")
    log(f"    Context snippet: ...{prompt_text[max(0,positions[0]-30):positions[-1]+30]}...")

    base = compute_lp(model, tokenizer, prompt, pos_resp, neg_resp, device)
    base_diff = base["lp_diff"]
    log(f"    Baseline lp_diff={base_diff:+.4f}")

    results = {
        "occurrence_idx": occurrence_idx,
        "positions": positions,
        "tokenized_as": tok_pieces,
        "context_snippet": prompt_text[max(0, positions[0]-30):positions[-1]+30],
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

    results["summary"] = {
        "best_single_layer": int(best_single[0]),
        "best_single_delta": best_single[1]["delta"],
        "best_forward_layer": int(best_forward[0]),
        "best_forward_delta": best_forward[1]["delta"],
        "best_backward_layer": int(best_backward[0]),
        "best_backward_delta": best_backward[1]["delta"],
    }

    log(f"    Best single layer:  L{best_single[0]}  "
        f"lp_diff={best_single[1]['lp_diff']:+.4f}  Δ={best_single[1]['delta']:+.4f}")
    return results


def main():
    log("=" * 64)
    log("P28: Position Separation — Same Token 'funding' at Different Positions")
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
            hallucinated.append({
                "idx": idx,
                "sample": sample,
                "base_diff": result["lp_diff"],
            })
        log(f"    Sample {idx:>2d}: lp_diff={result['lp_diff']:+.4f}  "
            f"hall={'YES' if is_hall else 'no'}")

    log(f"\n  Hallucinated: {len(hallucinated)} samples")

    log(f"\n[Step 4] Finding ALL occurrences of '{CAUSAL_TOKEN}' in each hallucinated sample...")
    sample_positions = []
    for h in hallucinated:
        ctx = h["sample"].get("context", "")
        q = h["sample"].get("question", "")
        prompt_text = f"{ctx}\n\nQuestion: {q}\nAnswer:"
        all_pos = find_all_token_positions(prompt_text, tokenizer, CAUSAL_TOKEN)
        if all_pos:
            sample_positions.append((h["idx"], h["sample"], h["base_diff"], prompt_text, all_pos))
            log(f"\n  Sample {h['idx']} [{h['sample'].get('_position','?')}]: "
                f"{len(all_pos)} occurrence(s) of '{CAUSAL_TOKEN}'")
            for oi, (pos, tok) in enumerate(all_pos):
                context_snip = prompt_text[max(0, pos[0]-30):pos[-1]+30]
                log(f"    #{oi}: pos={pos} tokens={tok} \"...{context_snip}...\"")

    total_occurrences = sum(len(occ) for _, _, _, _, occ in sample_positions)
    log(f"\n  Total occurrences to ablate: {total_occurrences}")
    total_passes = total_occurrences * 72
    log(f"  Total forward passes: {total_passes}")

    log(f"\n[Step 5] Per-layer ablation — each occurrence separately...")
    all_results = []
    for sidx, sample, base_diff, prompt_text, occurrences in sample_positions:
        log(f"\n{'='*48}")
        log(f"Sample {sidx} [{sample.get('_position','?')}] base={base_diff:+.4f}")
        log(f"  Question: {sample.get('question','')[:80]}")
        for oi, (positions, tok_pieces) in enumerate(occurrences):
            log(f"\n  --- Occurrence {oi} at pos={positions} ---")
            result = per_position_ablation(model, tokenizer, sample, device,
                                           sidx, oi, positions, tok_pieces,
                                           prompt_text, prompt_text)
            if result:
                result["sample_idx"] = sidx
                result["position_label"] = sample.get("_position", "?")
                result["question"] = sample.get("question", "")[:80]
                all_results.append(result)

    elapsed = time.time() - t0

    log(f"\n{'=' * 64}")
    log(f"[Summary] P28 Position Separation")
    log(f"  Occurrences analyzed: {len(all_results)}")
    log(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    by_sample = {}
    for r in all_results:
        key = r["sample_idx"]
        if key not in by_sample:
            by_sample[key] = []
        by_sample[key].append(r)

    log(f"\n  === POSITION COMPARISON ===")
    for sidx, results_list in sorted(by_sample.items()):
        if len(results_list) < 2:
            continue
        log(f"\n  Sample {sidx} [{results_list[0]['position_label']}]:")
        for r in results_list:
            peak = r["summary"]["best_single_layer"]
            delta = r["summary"]["best_single_delta"]
            pos = r["positions"][0]
            snippet = r["context_snippet"]
            log(f"    pos={pos:>3d} → L{peak} Δ={delta:+.4f}  \"{snippet}\"")
        peaks = [r["summary"]["best_single_layer"] for r in results_list]
        deltas = [r["summary"]["best_single_delta"] for r in results_list]
        peak_diff = abs(peaks[0] - peaks[1]) if len(peaks) >= 2 else 0
        delta_diff = abs(deltas[0] - deltas[1]) if len(deltas) >= 2 else 0
        if peak_diff >= 2:
            log(f"    → SIGNIFICANT separation: Δlayer={peak_diff}, Δdelta={delta_diff:.4f}")
        else:
            log(f"    → Same layer (±1)")

    log(f"\n  === CROSS-SAMPLE POSITION PATTERN ===")
    early_peaks = []
    late_peaks = []
    for sidx, results_list in sorted(by_sample.items()):
        sorted_by_pos = sorted(results_list, key=lambda r: r["positions"][0])
        if len(sorted_by_pos) >= 2:
            early_peaks.append(sorted_by_pos[0]["summary"]["best_single_layer"])
            late_peaks.append(sorted_by_pos[-1]["summary"]["best_single_layer"])
            log(f"  Sample {sidx}: early(pos={sorted_by_pos[0]['positions'][0]})→L{early_peaks[-1]}  "
                f"late(pos={sorted_by_pos[-1]['positions'][0]})→L{late_peaks[-1]}")

    if early_peaks and late_peaks:
        log(f"\n  Early mean peak: {sum(early_peaks)/len(early_peaks):.1f}  "
            f"Late mean peak: {sum(late_peaks)/len(late_peaks):.1f}")
        log(f"  Early peaks: {early_peaks}  Late peaks: {late_peaks}")

        all_early_late = all(abs(e - l) >= 2 for e, l in zip(early_peaks, late_peaks))
        if all_early_late:
            log(f"\n  CONCLUSION: Position-dependent encoding is SYSTEMATIC.")
            log(f"  'funding' in context → L{int(sum(early_peaks)/len(early_peaks))} (early concept recognition)")
            log(f"  'funding' in question → L{int(sum(late_peaks)/len(late_peaks))} (late query integration)")
        else:
            log(f"\n  CONCLUSION: Position effect is INCONSISTENT across samples.")

    summary_data = {
        "config": {
            "method": "position-separated per-layer causal token ablation",
            "causal_token": CAUSAL_TOKEN,
            "scale": SCALE_ZERO,
            "n_layers": N_LAYERS,
        },
        "n_occurrences": len(all_results),
        "time_s": round(elapsed, 1),
        "results": [],
        "position_comparison": {
            "early_peaks": early_peaks,
            "late_peaks": late_peaks,
            "early_mean": round(sum(early_peaks) / len(early_peaks), 2) if early_peaks else None,
            "late_mean": round(sum(late_peaks) / len(late_peaks), 2) if late_peaks else None,
        },
    }

    for r in all_results:
        single_curve = [r["single_layer"][str(l)]["delta"] for l in range(N_LAYERS)]
        peak_single = max(single_curve)
        peak_single_layer = single_curve.index(peak_single)

        summary_data["results"].append({
            "sample_idx": r["sample_idx"],
            "position_label": r["position_label"],
            "occurrence_idx": r["occurrence_idx"],
            "token_position": r["positions"][0],
            "context_snippet": r["context_snippet"],
            "token_positions": r["positions"],
            "baseline_lp_diff": round(r["baseline"]["lp_diff"], 6),
            "single_layer_curve": [round(v, 6) for v in single_curve],
            "peak_single_layer": peak_single_layer,
            "peak_single_delta": round(peak_single, 6),
            "best_forward_layer": r["summary"]["best_forward_layer"],
            "best_forward_delta": round(r["summary"]["best_forward_delta"], 6),
            "best_backward_layer": r["summary"]["best_backward_layer"],
            "best_backward_delta": round(r["summary"]["best_backward_delta"], 6),
        })

        log(f"\n  Sample {r['sample_idx']} #{r['occurrence_idx']} pos={r['positions'][0]}:")
        log(f"    Single-layer peak:      L{peak_single_layer}  Δ={peak_single:+.4f}")
        ranked = sorted(enumerate(single_curve), key=lambda x: -x[1])[:5]
        for layer_idx, delta in ranked:
            bar = "█" * max(1, int(delta * 40 / max(0.01, peak_single)))
            log(f"      L{layer_idx:>2d}: {delta:+.4f} {bar}")

    with open(os.path.join(RESULTS_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    log(f"\nResults saved to {RESULTS_DIR}/results.json")
    log(f"\nP28 Complete.")


if __name__ == "__main__":
    main()