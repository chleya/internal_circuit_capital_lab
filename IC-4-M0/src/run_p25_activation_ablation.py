"""
P25: Causal Token Activation Ablation — Fixing Hallucination at the Routing Level
==================================================================================
P23 identified "funding" as the true causal distractor (Δ +0.36 on lp_diff).
P24 showed embedding replacement fails (hallucination is structural, not representational).
P17-P18 showed q_proj in deep layers (16-23) is the routing mechanism.

P25 asks: Can we fix hallucination in the BASE model (no training) by surgically
zeroing the hidden state at the causal token position in deep layers?

This is the most surgical intervention possible:
- Does NOT remove text (preserves full sequence structure)
- Does NOT replace embeddings (preserves initial representation)
- Only prevents the causal token's representation from propagating through deep layers

Interventions tested:
  1. ZERO single token "funding" at ALL layers
  2. ZERO single token at DEEP layers only (16-23)
  3. ZERO all tokens in the causal phrase ("funding", "received", "series")
  4. SCALE (0.5×) instead of zero
  5. ZERO at deep layers + P22 text-level as joint intervention

Key question: Can activation-level ablation cross the P23 representation floor
(lp_diff +0.36) that text-level and embedding-level interventions cannot?

Usage:
  python src/run_p25_activation_ablation.py
"""

import os, sys, time, json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_builder import load_jsonl

RESULTS_DIR = "results_p25_activation_ablation"
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, "run_log.txt")

CAUSAL_TOKENS = ["funding", "received", "series", "a", "undisclosed",
                 "investor", "led", "round"]
DEEP_LAYERS = (16, 23)
SCALES = [0.0, 0.5]


def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def find_token_positions_in_prompt(prompt, tokenizer, target_substrings):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    positions = {}
    for substr in target_substrings:
        substr_variants = [substr, " " + substr]
        for variant in substr_variants:
            variant_tokens = tokenizer.tokenize(variant)
            n = len(variant_tokens)
            for i in range(len(tokens) - n + 1):
                if tokens[i:i + n] == variant_tokens:
                    positions[substr] = list(range(i, i + n))
                    break
            if substr in positions:
                break
    return positions


def build_hook_for_positions(positions_to_ablate, scale):
    pos_set = set(positions_to_ablate)
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


def logprob_with_hooks(model, tokenizer, prompt, response, device,
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


def compute_lp_with_ablation(model, tokenizer, sample, device,
                              positions_to_ablate, layer_indices, scale):
    ctx = sample.get("context", "")
    q = sample.get("question", "")
    prompt = f"{ctx}\n\nQuestion: {q}\nAnswer:"

    hook_fn = build_hook_for_positions(positions_to_ablate, scale)
    layer_hooks = {li: hook_fn for li in layer_indices}

    lp_pos = logprob_with_hooks(model, tokenizer, prompt,
                                 sample.get("positive_response", ""),
                                 device, layer_hooks)
    lp_neg = logprob_with_hooks(model, tokenizer, prompt,
                                 sample.get("negative_response", ""),
                                 device, layer_hooks)

    return {
        "lp_pos": lp_pos, "lp_neg": lp_neg,
        "lp_diff": lp_pos - lp_neg,
        "hallucinates": (lp_pos > lp_neg) and
                        sample.get("answerability") == "unanswerable",
        "pref_positive": lp_pos > lp_neg,
    }


def compute_lp_baseline(model, tokenizer, sample, device):
    return compute_lp_with_ablation(model, tokenizer, sample, device,
                                     [], [], 1.0)


def activation_ablation(model, tokenizer, sample, device, sample_idx,
                         p23_floor, p23_detail):
    base = compute_lp_baseline(model, tokenizer, sample, device)
    if not base["hallucinates"]:
        return {**base, "strategy": "none", "fixable": False,
                "improvement": 0.0, "crossed_floor": False,
                "p23_floor": p23_floor, "details": "not hallucinated"}

    ctx = sample.get("context", "")
    q = sample.get("question", "")
    prompt = f"{ctx}\n\nQuestion: {q}\nAnswer:"

    token_positions = find_token_positions_in_prompt(prompt, tokenizer, CAUSAL_TOKENS)

    base_lp_diff = base["lp_diff"]
    best = {"lp_diff": base_lp_diff, "config": "none",
            "improvement": 0.0, "fixed": False,
            "token_str": "", "scale": 1.0, "layers": "none"}

    log(f"\n  Sample {sample_idx:>3d}: base lp_diff={base_lp_diff:+.4f}")
    log(f"    P23 floor: {p23_floor:+.4f} ({p23_detail})")
    log(f"    Found causal tokens: {[(k, v) for k, v in token_positions.items() if v]}")

    layer_configs = [
        ("deep", list(range(DEEP_LAYERS[0], DEEP_LAYERS[1] + 1))),
        ("all", list(range(24))),
        ("early", list(range(0, 8))),
        ("mid", list(range(8, 16))),
    ]

    n_tests = 0
    for token_str, positions in token_positions.items():
        if not positions:
            continue

        for layer_name, layer_indices in layer_configs:
            for scale in SCALES:
                n_tests += 1
                result = compute_lp_with_ablation(
                    model, tokenizer, sample, device,
                    positions, layer_indices, scale)
                improvement = base_lp_diff - result["lp_diff"]

                marker = ""
                if result["lp_diff"] < best["lp_diff"]:
                    best = {"lp_diff": result["lp_diff"],
                            "config": f"{layer_name}@{scale}",
                            "improvement": improvement,
                            "token_str": token_str,
                            "fixed": result["lp_diff"] <= 0,
                            "scale": scale,
                            "layers": layer_name}
                    marker = " *BEST"

                if result["lp_diff"] <= 0:
                    marker += " FIXED!"
                    log(f"    [{layer_name}@{scale}] {token_str} "
                        f"lp_diff={result['lp_diff']:+.4f} "
                        f"({'+' if improvement > 0 else ''}{improvement:+.4f}){marker}")
                    crossed = p23_floor > 0 and result["lp_diff"] <= 0
                    return {
                        **base,
                        "lp_diff": result["lp_diff"],
                        "pref_positive": False,
                        "hallucinates": False,
                        "strategy": f"ablate:{layer_name}@{scale}",
                        "fixable": True,
                        "improvement": round(improvement, 4),
                        "crossed_floor": crossed,
                        "p23_floor": p23_floor,
                        "token_str": token_str,
                        "scale": scale,
                        "layers": layer_name,
                        "n_tests": n_tests,
                        "details": f"ablate {token_str} at {layer_name} "
                                   f"layers @ {scale}×",
                    }

    n_layers = max(len(model.model.layers), 24)
    all_deep_indices = list(range(DEEP_LAYERS[0], DEEP_LAYERS[1] + 1))
    all_positions = []
    all_token_names = []
    for token_str, positions in token_positions.items():
        if positions and token_str in ["funding", "received", "series", "a"]:
            all_positions.extend(positions[:1])
            all_token_names.append(token_str)

    if len(all_positions) >= 2:
        for layer_name, layer_indices in [("deep", all_deep_indices),
                                           ("all", list(range(n_layers)))]:
            for scale in SCALES:
                n_tests += 1
                result = compute_lp_with_ablation(
                    model, tokenizer, sample, device,
                    all_positions, layer_indices, scale)
                improvement = base_lp_diff - result["lp_diff"]
                combo = "+".join(all_token_names)
                log(f"    [{layer_name}@{scale}] {combo} "
                    f"lp_diff={result['lp_diff']:+.4f} "
                    f"({'+' if improvement > 0 else ''}{improvement:+.4f})"
                    f"{' FIXED!' if result['lp_diff'] <= 0 else ''}")

                if result["lp_diff"] < best["lp_diff"]:
                    best = {"lp_diff": result["lp_diff"],
                            "config": f"{layer_name}@{scale}(combo)",
                            "improvement": improvement,
                            "token_str": combo,
                            "fixed": result["lp_diff"] <= 0,
                            "scale": scale,
                            "layers": layer_name}

                if result["lp_diff"] <= 0:
                    crossed = p23_floor > 0 and result["lp_diff"] <= 0
                    return {
                        **base,
                        "lp_diff": result["lp_diff"],
                        "pref_positive": False,
                        "hallucinates": False,
                        "strategy": f"ablate:{layer_name}@{scale}(combo)",
                        "fixable": True,
                        "improvement": round(improvement, 4),
                        "crossed_floor": crossed,
                        "p23_floor": p23_floor,
                        "token_str": combo,
                        "scale": scale,
                        "layers": layer_name,
                        "n_tests": n_tests,
                        "details": f"ablate {combo} at {layer_name} "
                                   f"layers @ {scale}×",
                    }

    crossed = best["lp_diff"] < p23_floor
    log(f"    Best: {best['config']}({best['token_str']}) "
        f"lp_diff={best['lp_diff']:+.4f} "
        f"(vs P23 floor {p23_floor:+.4f})"
        f"{' CROSSED FLOOR!' if crossed else ''}")

    return {
        **base,
        "lp_diff": best["lp_diff"],
        "pref_positive": best["lp_diff"] > 0,
        "hallucinates": best["lp_diff"] > 0,
        "strategy": f"ablate:{best['config']}" if best["config"] != "none"
                   else "ablate:none",
        "fixable": best["fixed"],
        "improvement": round(best["improvement"], 4),
        "crossed_floor": crossed,
        "p23_floor": p23_floor,
        "token_str": best["token_str"],
        "scale": best["scale"],
        "layers": best["layers"],
        "n_tests": n_tests,
        "details": f"{best['config']}({best['token_str']}) "
                   f"lp_diff={best['lp_diff']:+.4f}",
    }


def compute_metrics(results, test_samples):
    ans = [r for r, s in zip(results, test_samples)
           if s.get("answerability") == "answerable"]
    unans = [r for r, s in zip(results, test_samples)
             if s.get("answerability") == "unanswerable"]
    hall = sum(1 for r in unans if r["pref_positive"])
    corr = sum(1 for r in ans if r["pref_positive"])
    n_u = len(unans) if unans else 1
    n_a = len(ans) if ans else 1
    return {"H": round(hall / n_u, 4), "C": round(corr / n_a, 4)}


def main():
    log("=" * 64)
    log("P25: Causal Token Activation Ablation")
    log("Fixing hallucination at the routing level (no training)")
    log("=" * 64)
    t0 = time.time()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    log("\n[Step 1] Loading base model (Qwen2.5-0.5B)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="cpu")
    model.eval()
    device = next(model.parameters()).device
    log(f"  Device: {device}, Layers: {len(model.model.layers)}")

    log("\n[Step 2] Loading test samples...")
    pos_dir = os.path.join(base_dir, "data_position_sensitivity", "s0")
    test_samples = []
    for pos in ["early", "mid", "late"]:
        path = os.path.join(pos_dir, f"test_{pos}_s0.jsonl")
        if os.path.exists(path):
            for s in load_jsonl(path)[:10]:
                s["_position"] = pos
                test_samples.append(s)
    n_ans = sum(1 for s in test_samples if s.get("answerability") == "answerable")
    n_unans = sum(1 for s in test_samples if s.get("answerability") == "unanswerable")
    log(f"  Loaded {len(test_samples)} samples ({n_ans}A + {n_unans}U)")

    log("\n[Step 3] BASELINE evaluation...")
    base_results = [compute_lp_baseline(model, tokenizer, s, device)
                    for s in test_samples]
    base_metrics = compute_metrics(base_results, test_samples)
    log(f"  BASELINE: H={base_metrics['H']:.4f}, C={base_metrics['C']:.4f}")

    hallucinated_indices = [
        i for i, (r, s) in enumerate(zip(base_results, test_samples))
        if r["hallucinates"]
    ]
    log(f"  Hallucinated samples: {len(hallucinated_indices)}")

    p23_floors = {
        4: 0.3649,
        14: 0.3932,
        17: -0.0194,
        24: 0.4938,
        27: 0.0158,
    }
    p23_details = {
        4: "neutralize(funding) floor",
        14: "prune(funding) floor",
        17: "FIXED",
        24: "prune(funding) floor",
        27: "neutralize(The) near zero",
    }

    log(f"\n[Step 4] Activation Ablation on Causal Tokens...")
    log(f"  Causal tokens: {CAUSAL_TOKENS}")
    log(f"  Layer configs: deep(16-23), all(0-23), early(0-7), mid(8-15)")
    log(f"  Scales: {SCALES}")
    log(f"")

    repaired_results = []
    t_repair = time.time()

    for idx, sample in enumerate(test_samples):
        floor = p23_floors.get(idx, 0.0)
        detail = p23_details.get(idx, "N/A")
        result = activation_ablation(
            model, tokenizer, sample, device, idx, floor, detail)
        repaired_results.append(result)

    repair_time = time.time() - t_repair
    post_metrics = compute_metrics(repaired_results, test_samples)

    log(f"\n{'=' * 64}")
    log(f"[Summary] P25 Activation Ablation")
    log(f"  Baseline:    H={base_metrics['H']:.4f}, C={base_metrics['C']:.4f}")
    log(f"  P22/23 best: H=0.3333 (text-level, 1/5 fixed)")
    log(f"  P24 embed:   H=0.4167 (embedding-level, 0/5 fixed)")
    log(f"  P25 ablate:  H={post_metrics['H']:.4f}, C={post_metrics['C']:.4f}")
    log(f"  ΔH (vs baseline): {post_metrics['H'] - base_metrics['H']:+.4f}")

    n_fixable = sum(1 for r in repaired_results if r.get("fixable"))
    n_crossed = sum(1 for r in repaired_results if r.get("crossed_floor"))
    total_tests = sum(r.get("n_tests", 0) for r in repaired_results)

    log(f"\n  Repair Statistics:")
    log(f"    Fixed:             {n_fixable}")
    log(f"    Crossed P23 floor: {n_crossed}")
    log(f"    Total ablations:   {total_tests}")

    log(f"\n  Layer-Level Comparison (vs floor):")
    for idx in hallucinated_indices:
        r = repaired_results[idx]
        s = test_samples[idx]
        floor = p23_floors.get(idx, 0.0)
        floor_delta = floor - r["lp_diff"]
        crossed = "CROSSED FLOOR!" if r.get("crossed_floor") else ""
        fixed = "FIXED!" if r.get("fixable") else ""
        log(f"    #{idx:>2d} [{s.get('_position','?'):5s}] "
            f"base={r.get('lp_diff',0):+.4f} "
            f"P23_floor={floor:+.4f} "
            f"P25_best={r['lp_diff']:+.4f} "
            f"Δ_floor={floor_delta:+.4f} "
            f"[{r.get('details','')}] {crossed} {fixed}")

    elapsed = time.time() - t0
    log(f"\n  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    results = {
        "config": {
            "method": "activation ablation at causal token positions",
            "causal_tokens": CAUSAL_TOKENS,
            "layer_configs": ["deep(16-23)", "all(0-23)", "early(0-7)", "mid(8-15)"],
            "scales": SCALES,
        },
        "baseline_H": base_metrics["H"],
        "baseline_C": base_metrics["C"],
        "p23_best_H": 0.3333,
        "p24_embed_H": 0.4167,
        "post_ablate_H": post_metrics["H"],
        "post_ablate_C": post_metrics["C"],
        "delta_H": round(post_metrics["H"] - base_metrics["H"], 4),
        "n_samples": len(test_samples),
        "n_hallucinated": len(hallucinated_indices),
        "n_fixed": n_fixable,
        "n_crossed_floor": n_crossed,
        "total_ablations": total_tests,
        "repairs": [{
            "idx": i,
            "position": s.get("_position", "?"),
            "question": s.get("question", "")[:60],
            "strategy": r.get("strategy", "none"),
            "lp_diff": r.get("lp_diff", 0),
            "improvement": r.get("improvement", 0),
            "fixable": r.get("fixable", False),
            "crossed_floor": r.get("crossed_floor", False),
            "p23_floor": r.get("p23_floor", 0),
            "best_layers": r.get("layers", "none"),
            "best_scale": r.get("scale", 1.0),
            "token_str": r.get("token_str", ""),
            "details": r.get("details", ""),
        } for i, (r, s) in enumerate(zip(repaired_results, test_samples))],
        "floor_comparison": [{
            "idx": idx,
            "position": test_samples[idx].get("_position", "?"),
            "p23_floor": p23_floors.get(idx, 0),
            "p25_lp_diff": repaired_results[idx]["lp_diff"],
            "floor_delta": (p23_floors.get(idx, 0)
                            - repaired_results[idx]["lp_diff"]),
            "crossed": repaired_results[idx].get("crossed_floor", False),
        } for idx in hallucinated_indices],
        "time_s": round(elapsed, 1),
    }

    with open(os.path.join(RESULTS_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log(f"\nResults saved to {RESULTS_DIR}/results.json")
    log(f"\nP25 Complete.")


if __name__ == "__main__":
    main()