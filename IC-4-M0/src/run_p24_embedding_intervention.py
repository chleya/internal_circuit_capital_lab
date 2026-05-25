"""
P24: Embedding-Level Semantic Intervention — Crossing the Representation Floor
==============================================================================
P23 revealed a representation floor: text-level interventions (removing
"funding") leave lp_diff at +0.36-0.49 for BoltStream samples.

P24 asks: Is this floor text-level or representation-level?

If we intervene at the EMBEDDING level (replace the "funding" token's
embedding vector with a neutral one, keeping the token sequence unchanged),
can we cross the floor?

Interventions tested:
  1. EMBED_REPLACE: Replace causal token embedding with neutral token
  2. EMBED_ZERO: Zero out the causal token embedding
  3. EMBED_NOISE: Add Gaussian noise to the embedding
  4. EMBED_MEAN: Replace with mean embedding of all tokens
  5. EMBED_COMBO: Replace multiple causal token embeddings simultaneously

Comparison metric: lp_diff reduction vs. P23 text-level best.

Key question: Is hallucination encoded in the token's embedding vector
(which EMBED_REPLACE changes), or in the attention/processing dynamics
that survive text-level removal?

Usage:
  python src/run_p24_embedding_intervention.py
"""

import os, sys, time, json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_builder import load_jsonl

RESULTS_DIR = "results_p24_embedding_intervention"
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, "run_log.txt")

NEUTRAL_TOKEN = "it"
NOISE_STD = 0.1


def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def logprob_of_response_embed(model, tokenizer, prompt, response, device,
                               embed_mods=None):
    """
    Compute log-prob of response given prompt, with optional embedding
    modifications.

    embed_mods: dict mapping token_index -> new_embedding or "zero" or "noise"
    """
    full_text = f"{prompt} {response}"
    enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"].to(device)

    prompt_enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    prompt_len = prompt_enc["input_ids"].shape[1]

    embed_layer = model.get_input_embeddings()
    with torch.no_grad():
        embeddings = embed_layer(input_ids)

        if embed_mods:
            for token_idx, mod in embed_mods.items():
                if token_idx >= embeddings.shape[1]:
                    continue
                if isinstance(mod, torch.Tensor):
                    embeddings[0, token_idx] = mod.to(device)
                elif mod == "zero":
                    embeddings[0, token_idx] = torch.zeros_like(
                        embeddings[0, token_idx])
                elif mod == "noise":
                    noise = torch.randn_like(embeddings[0, token_idx]) * NOISE_STD
                    embeddings[0, token_idx] = embeddings[0, token_idx] + noise

        labels = input_ids.clone()
        labels[0, :prompt_len] = -100

        outputs = model(inputs_embeds=embeddings, labels=labels)

    return -outputs.loss.item()


def compute_lp_embed(model, tokenizer, sample, device, embed_mods=None):
    ctx = sample.get("context", "")
    q = sample.get("question", "")
    prompt = f"{ctx}\n\nQuestion: {q}\nAnswer:"
    lp_pos = logprob_of_response_embed(model, tokenizer, prompt,
                                        sample.get("positive_response", ""),
                                        device, embed_mods)
    lp_neg = logprob_of_response_embed(model, tokenizer, prompt,
                                        sample.get("negative_response", ""),
                                        device, embed_mods)
    return {
        "lp_pos": lp_pos, "lp_neg": lp_neg,
        "lp_diff": lp_pos - lp_neg,
        "hallucinates": (lp_pos > lp_neg) and
                        sample.get("answerability") == "unanswerable",
        "pref_positive": lp_pos > lp_neg,
    }


def find_token_positions(prompt, target_substrings, tokenizer):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    positions = {}
    for substr in target_substrings:
        substr_tokens = tokenizer.tokenize(substr)
        n = len(substr_tokens)
        for i in range(len(tokens) - n + 1):
            if tokens[i:i + n] == substr_tokens:
                positions[substr] = list(range(i, i + n))
                break

    return positions


def embedding_intervention(model, tokenizer, sample, device, sample_idx,
                            p23_best_lp_diff, p23_detail):
    base = compute_lp_embed(model, tokenizer, sample, device)
    if not base["hallucinates"]:
        return {**base, "strategy": "none", "fixable": False,
                "improvement": 0.0, "embed_mod": "none",
                "details": "not hallucinated"}

    ctx = sample.get("context", "")
    q = sample.get("question", "")
    prompt = f"{ctx}\n\nQuestion: {q}\nAnswer:"

    causal_tokens = ["funding", "received", "series", "a", "undisclosed",
                     "investor", "led", "round", "2011"]
    token_positions = find_token_positions(prompt, causal_tokens, tokenizer)

    embed_layer = model.get_input_embeddings()
    neutral_id = tokenizer(NEUTRAL_TOKEN, add_special_tokens=False)["input_ids"][0]
    neutral_emb = embed_layer.weight[neutral_id].clone()

    base_lp_diff = base["lp_diff"]
    best = {"lp_diff": base_lp_diff, "embed_mod": "none",
            "improvement": 0.0, "token_str": "", "fixed": False}

    log(f"\n  Sample {sample_idx:>3d}: base lp_diff={base_lp_diff:+.4f}")
    log(f"    P23 best: lp_diff={p23_best_lp_diff:+.4f} ({p23_detail})")

    interventions = [
        ("embed_replace", lambda tid: neutral_emb.clone()),
        ("embed_zero", lambda tid: "zero"),
        ("embed_noise", lambda tid: "noise"),
    ]

    for mod_name, mod_fn in interventions:
        for token_str, positions in token_positions.items():
            if not positions:
                continue

            for pos in positions:
                mod = {pos: mod_fn(pos)}
                result = compute_lp_embed(model, tokenizer, sample, device, mod)
                improvement = base_lp_diff - result["lp_diff"]

                marker = ""
                if result["lp_diff"] < best["lp_diff"]:
                    best = {"lp_diff": result["lp_diff"],
                            "embed_mod": mod_name,
                            "improvement": improvement,
                            "token_str": token_str,
                            "token_pos": pos,
                            "fixed": result["lp_diff"] <= 0}
                    marker = " *BEST"

                if result["lp_diff"] <= 0:
                    marker += " FIXED!"
                    log(f"    [{mod_name}] {token_str}[{pos}] "
                        f"lp_diff={result['lp_diff']:+.4f} "
                        f"({'+' if improvement > 0 else ''}{improvement:+.4f}){marker}")

                    cross_floor = (p23_best_lp_diff > 0 and result["lp_diff"] <= 0)
                    return {
                        **base,
                        "lp_diff": result["lp_diff"],
                        "pref_positive": False,
                        "hallucinates": False,
                        "strategy": f"embed:{mod_name}",
                        "fixable": True,
                        "improvement": round(improvement, 4),
                        "embed_mod": mod_name,
                        "token_str": token_str,
                        "token_pos": pos,
                        "crossed_floor": cross_floor,
                        "p23_best_lp_diff": p23_best_lp_diff,
                        "details": f"{mod_name}({token_str}[{pos}]) "
                                   f"lp_diff={result['lp_diff']:+.4f}",
                    }

    multi_mods = {}
    multi_tokens = []
    for token_str, positions in token_positions.items():
        if positions and token_str in ["funding", "received", "series"]:
            for pos in positions[:1]:
                multi_mods[pos] = neutral_emb.clone()
                multi_tokens.append(f"{token_str}[{pos}]")

    if len(multi_mods) >= 2:
        result = compute_lp_embed(model, tokenizer, sample, device, multi_mods)
        improvement = base_lp_diff - result["lp_diff"]
        combo_str = "+".join(multi_tokens)
        log(f"    [embed_combo] {combo_str} "
            f"lp_diff={result['lp_diff']:+.4f} "
            f"({'+' if improvement > 0 else ''}{improvement:+.4f})"
            f"{' FIXED!' if result['lp_diff'] <= 0 else ''}")

        if result["lp_diff"] < best["lp_diff"]:
            best = {"lp_diff": result["lp_diff"],
                    "embed_mod": "embed_combo",
                    "improvement": improvement,
                    "token_str": combo_str,
                    "fixed": result["lp_diff"] <= 0}

        if result["lp_diff"] <= 0:
            cross_floor = (p23_best_lp_diff > 0 and result["lp_diff"] <= 0)
            return {
                **base,
                "lp_diff": result["lp_diff"],
                "pref_positive": False,
                "hallucinates": False,
                "strategy": "embed:combo",
                "fixable": True,
                "improvement": round(improvement, 4),
                "embed_mod": "embed_combo",
                "token_str": combo_str,
                "crossed_floor": cross_floor,
                "p23_best_lp_diff": p23_best_lp_diff,
                "details": f"combo({combo_str}) lp_diff={result['lp_diff']:+.4f}",
            }

    cross_floor = (best["lp_diff"] < p23_best_lp_diff)
    log(f"    Best embed: {best['embed_mod']}({best['token_str']}) "
        f"lp_diff={best['lp_diff']:+.4f} "
        f"(vs P23 floor {p23_best_lp_diff:+.4f})"
        f"{' CROSSED FLOOR!' if cross_floor else ''}")

    return {
        **base,
        "lp_diff": best["lp_diff"],
        "pref_positive": best["lp_diff"] > 0,
        "hallucinates": best["lp_diff"] > 0,
        "strategy": f"embed:{best['embed_mod']}" if best["embed_mod"] != "none"
                   else "embed:none",
        "fixable": best["fixed"],
        "improvement": round(best["improvement"], 4),
        "embed_mod": best["embed_mod"],
        "token_str": best["token_str"],
        "crossed_floor": cross_floor,
        "p23_best_lp_diff": p23_best_lp_diff,
        "details": f"{best['embed_mod']}({best['token_str']}) "
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
    log("P24: Embedding-Level Semantic Intervention")
    log("Crossing the representation floor discovered in P23")
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
    base_results = [compute_lp_embed(model, tokenizer, s, device)
                    for s in test_samples]
    base_metrics = compute_metrics(base_results, test_samples)
    log(f"  BASELINE: H={base_metrics['H']:.4f}, C={base_metrics['C']:.4f}")

    hallucinated_indices = [
        i for i, (r, s) in enumerate(zip(base_results, test_samples))
        if r["hallucinates"]
    ]
    log(f"  Hallucinated samples: {len(hallucinated_indices)}")

    p23_data = {
        4: {"best_lp_diff": 0.6063,
            "detail": "neutralize('.ĊĊ'), causal=funding(Δ+0.364)"},
        14: {"best_lp_diff": 0.7188,
             "detail": "exhausted, causal=funding(Δ+0.326)"},
        17: {"best_lp_diff": -0.0194,
             "detail": "FIXED: sentence(wrapper)"},
        24: {"best_lp_diff": 0.7188,
             "detail": "sentence(wrapper), causal=funding(Δ+0.321)"},
        27: {"best_lp_diff": 0.0158,
             "detail": "neutralize('The'), almost fixed"},
    }

    log(f"\n[Step 4] Embedding-Level Interventions...")
    log(f"  Interventions: embed_replace, embed_zero, embed_noise, embed_combo")
    log(f"  Causal tokens: funding, received, series, undisclosed, investor...")
    log(f"")

    repaired_results = []
    t_repair = time.time()

    for idx, sample in enumerate(test_samples):
        p23 = p23_data.get(idx, {"best_lp_diff": 0.0, "detail": "N/A"})
        result = embedding_intervention(
            model, tokenizer, sample, device, idx,
            p23["best_lp_diff"], p23["detail"])
        repaired_results.append(result)

    repair_time = time.time() - t_repair
    post_metrics = compute_metrics(repaired_results, test_samples)

    log(f"\n{'=' * 64}")
    log(f"[Summary] P24 Embedding-Level Semantic Intervention")
    log(f"  Baseline:    H={base_metrics['H']:.4f}, C={base_metrics['C']:.4f}")
    log(f"  After P23:   H=0.3333 (best text-level)")
    log(f"  After P24:   H={post_metrics['H']:.4f}, C={post_metrics['C']:.4f}")
    log(f"  ΔH (vs baseline): {post_metrics['H'] - base_metrics['H']:+.4f}")

    n_hall = sum(1 for r in repaired_results if r.get("hallucinates"))
    n_fixable = sum(1 for r in repaired_results if r.get("fixable"))
    n_crossed = sum(1 for r in repaired_results if r.get("crossed_floor"))

    strategy_counts = {}
    for r in repaired_results:
        s = r.get("strategy", "none")
        strategy_counts[s] = strategy_counts.get(s, 0) + 1

    log(f"\n  Repair Statistics:")
    log(f"    Hallucinated (before): {len(hallucinated_indices)}")
    log(f"    Fixed:                 {n_fixable}")
    log(f"    Crossed P23 floor:     {n_crossed}")
    log(f"    Strategy distribution: {strategy_counts}")

    log(f"\n  Embedding vs Text-Level Comparison:")
    for idx in hallucinated_indices:
        r = repaired_results[idx]
        s = test_samples[idx]
        p23 = p23_data.get(idx, {"best_lp_diff": 0.0})
        floor_delta = p23["best_lp_diff"] - r["lp_diff"]
        crossed = "CROSSED!" if r.get("crossed_floor") else ""
        log(f"    #{idx:>2d} [{s.get('_position','?'):5s}] "
            f"P23 floor={p23['best_lp_diff']:+.4f} → "
            f"P24 embed={r['lp_diff']:+.4f} "
            f"(Δ={floor_delta:+.4f}) "
            f"[{r.get('details','')}] {crossed}")

    elapsed = time.time() - t0
    log(f"\n  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    results = {
        "config": {
            "method": "embedding-level semantic intervention",
            "interventions": ["embed_replace", "embed_zero",
                              "embed_noise", "embed_combo"],
            "causal_tokens": ["funding", "received", "series",
                              "undisclosed", "investor", "led", "round"],
            "neutral_token": NEUTRAL_TOKEN,
            "noise_std": NOISE_STD,
        },
        "baseline_H": base_metrics["H"],
        "baseline_C": base_metrics["C"],
        "p23_best_H": 0.3333,
        "post_embed_H": post_metrics["H"],
        "post_embed_C": post_metrics["C"],
        "delta_H": round(post_metrics["H"] - base_metrics["H"], 4),
        "n_samples": len(test_samples),
        "n_hallucinated": len(hallucinated_indices),
        "n_fixed": n_fixable,
        "n_crossed_floor": n_crossed,
        "strategy_distribution": strategy_counts,
        "repairs": [{
            "idx": i,
            "position": s.get("_position", "?"),
            "question": s.get("question", "")[:60],
            "strategy": r.get("strategy", "none"),
            "embed_mod": r.get("embed_mod", "none"),
            "token_str": r.get("token_str", ""),
            "lp_diff": r.get("lp_diff", 0),
            "improvement": r.get("improvement", 0),
            "fixable": r.get("fixable", False),
            "crossed_floor": r.get("crossed_floor", False),
            "p23_floor": r.get("p23_best_lp_diff", 0),
            "details": r.get("details", ""),
        } for i, (r, s) in enumerate(zip(repaired_results, test_samples))],
        "floor_comparison": [{
            "idx": idx,
            "position": test_samples[idx].get("_position", "?"),
            "question": test_samples[idx].get("question", "")[:60],
            "p23_best_lp_diff": p23_data.get(idx, {}).get("best_lp_diff", 0),
            "p24_lp_diff": repaired_results[idx]["lp_diff"],
            "floor_delta": (p23_data.get(idx, {}).get("best_lp_diff", 0)
                            - repaired_results[idx]["lp_diff"]),
            "crossed": repaired_results[idx].get("crossed_floor", False),
        } for idx in hallucinated_indices],
        "time_s": round(elapsed, 1),
    }

    with open(os.path.join(RESULTS_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log(f"\nResults saved to {RESULTS_DIR}/results.json")
    log(f"\nP24 Complete.")


if __name__ == "__main__":
    main()