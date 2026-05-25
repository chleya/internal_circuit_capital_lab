"""
P23: Joint Counterfactual Search + Full-Token Causal Attribution
=================================================================
Extends P22's cascading counterfactual with two innovations:

Part A — Joint Counterfactual Search:
  When P22's single-token intervention nearly fixes hallucination
  (lp_diff close to 0), combine it with the next-best token's best
  strategy. This is a greedy combinatorial search that tests whether
  multiple distractors jointly cause the routing failure.

Part B — Full-Token Causal Attribution:
  For unfixable samples, compute the counterfactual impact of EVERY token
  (not just top-K by attention). This validates whether attention weight
  is a good proxy for causal importance — a fundamental question in
  mechanistic interpretability.

Key probability-theoretic framework:
  - Joint repair: P(hall|ctx\{t_i,t_j}) — are distractors additive?
  - Full attribution: Corr(alpha_i, Delta_lp_diff_i) — is attention causal?

Usage:
  python src/run_p23_joint_causal_attribution.py
"""

import os, sys, time, json, re
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_builder import load_jsonl

RESULTS_DIR = "results_p23_joint_causal_attribution"
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, "run_log.txt")

DEEP_LAYERS = (16, 23)
TOP_K_TOKENS = 5
MAX_JOINT_DEPTH = 3
NEUTRAL_TOKEN = "it"
FULL_ATTRIBUTION_SAMPLES = 3


def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def logprob_of_response(model, tokenizer, prompt, response, device):
    full_text = f"{prompt} {response}"
    enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=256)
    enc = {k: v.to(device) for k, v in enc.items()}
    prompt_enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    prompt_len = prompt_enc["input_ids"].shape[1]
    labels = enc["input_ids"].clone()
    labels[0, :prompt_len] = -100
    with torch.no_grad():
        outputs = model(**enc, labels=labels)
    return -outputs.loss.item()


def compute_lp_sample(model, tokenizer, sample, device):
    ctx = sample.get("context", "")
    q = sample.get("question", "")
    prompt = f"{ctx}\n\nQuestion: {q}\nAnswer:"
    lp_pos = logprob_of_response(model, tokenizer, prompt,
                                  sample.get("positive_response", ""), device)
    lp_neg = logprob_of_response(model, tokenizer, prompt,
                                  sample.get("negative_response", ""), device)
    is_unans = sample.get("answerability") == "unanswerable"
    return {
        "lp_pos": lp_pos, "lp_neg": lp_neg,
        "lp_diff": lp_pos - lp_neg,
        "hallucinates": is_unans and (lp_pos > lp_neg),
        "pref_positive": lp_pos > lp_neg,
        "prompt": prompt,
    }


def get_deep_attention(model, tokenizer, text, device):
    try:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(**enc, output_attentions=True)
        all_attn = outputs.attentions
        n_layers = len(all_attn)
        start = max(0, DEEP_LAYERS[0])
        end = min(n_layers, DEEP_LAYERS[1] + 1)
        deep_attns = [all_attn[i] for i in range(start, end)]
        if not deep_attns:
            return None, None
        stacked = torch.stack(deep_attns)
        mean_attn = stacked.mean(dim=(0, 2))
        last_to_all = mean_attn[0, -1, :]
        all_tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())
        return last_to_all.cpu().tolist(), all_tokens
    except Exception as e:
        log(f"  [WARN] Attention: {e}")
        return None, None


def get_token_char_spans(text, tokenizer):
    enc = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=256)
    offsets = enc["offset_mapping"]
    return [(s, e) for s, e in offsets]


def get_sentence_boundaries(text):
    boundaries = [0]
    for m in re.finditer(r'[.!?]\s+', text):
        boundaries.append(m.end())
    boundaries.append(len(text))
    return boundaries


def find_token_sentence(token_idx, text, tokenizer):
    spans = get_token_char_spans(text, tokenizer)
    if token_idx >= len(spans):
        return -1
    char_start = spans[token_idx][0]
    boundaries = get_sentence_boundaries(text)
    for i in range(len(boundaries) - 1):
        if boundaries[i] <= char_start < boundaries[i + 1]:
            return i
    return -1


def remove_sentence(text, sent_idx):
    boundaries = get_sentence_boundaries(text)
    if sent_idx < 0 or sent_idx >= len(boundaries) - 1:
        return text
    return (text[:boundaries[sent_idx]].rstrip() + " " +
            text[boundaries[sent_idx + 1]:].lstrip()).strip()


def remove_token_from_text(text, tokenizer, token_idx):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    token_ids = enc["input_ids"][0].tolist()
    if token_idx < 0 or token_idx >= len(token_ids):
        return text
    pruned_ids = token_ids[:token_idx] + token_ids[token_idx + 1:]
    return tokenizer.decode(pruned_ids, skip_special_tokens=True)


def neutralize_token_in_text(text, tokenizer, token_idx):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    token_ids = enc["input_ids"][0].tolist()
    if token_idx < 0 or token_idx >= len(token_ids):
        return text
    neutral_ids = tokenizer(NEUTRAL_TOKEN, add_special_tokens=False)["input_ids"]
    new_ids = token_ids[:token_idx] + neutral_ids + token_ids[token_idx + 1:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def apply_intervention(text, tokenizer, token_idx, strategy, sent_idx=0):
    if strategy == "prune":
        return remove_token_from_text(text, tokenizer, token_idx)
    elif strategy == "neutralize":
        return neutralize_token_in_text(text, tokenizer, token_idx)
    elif strategy == "sentence":
        return remove_sentence(text, sent_idx)
    return text


def apply_joint_interventions(text, tokenizer, interventions):
    working = text
    for token_idx, strategy in interventions:
        sent_idx = find_token_sentence(token_idx, text, tokenizer) if strategy == "sentence" else -1
        working = apply_intervention(working, tokenizer, token_idx, strategy, sent_idx)
    return working


def counterfactual_eval(model, tokenizer, sample, new_ctx, base_lp_diff, device):
    q = sample.get("question", "")
    prompt = f"{new_ctx}\n\nQuestion: {q}\nAnswer:"
    lp_pos = logprob_of_response(model, tokenizer, prompt,
                                  sample.get("positive_response", ""), device)
    lp_neg = logprob_of_response(model, tokenizer, prompt,
                                  sample.get("negative_response", ""), device)
    lp_diff = lp_pos - lp_neg
    return {
        "lp_diff": lp_diff,
        "improvement": base_lp_diff - lp_diff,
        "fixed": lp_diff <= 0,
    }


def rank_tokens_by_attention(attn_weights, all_tokens, n):
    token_scores = []
    for i in range(n - 1):
        tok = all_tokens[i] if i < len(all_tokens) else "?"
        if tok in ("", "<|endoftext|>", ".", ",", " ", "\n"):
            continue
        token_scores.append((i, attn_weights[i], tok))
    token_scores.sort(key=lambda x: x[1], reverse=True)
    return token_scores


def find_best_single_strategy(model, tokenizer, sample, ctx, top_tokens,
                               base_lp_diff, device):
    strategies = ["prune", "neutralize", "sentence"]
    best = {"lp_diff": base_lp_diff, "strategy": "none", "phase": 0,
            "improvement": 0.0, "token_idx": -1, "token_str": ""}
    n_cf = 0

    for phase_idx, strategy in enumerate(strategies):
        for ti, attn_w, tok_str in top_tokens:
            sent_idx = find_token_sentence(ti, ctx, tokenizer) if strategy == "sentence" else -1
            new_ctx = apply_intervention(ctx, tokenizer, ti, strategy, sent_idx)
            if new_ctx == ctx or len(new_ctx.strip()) < 10:
                continue

            n_cf += 1
            result = counterfactual_eval(model, tokenizer, sample, new_ctx, base_lp_diff, device)

            if result["lp_diff"] < best["lp_diff"]:
                best = {
                    "lp_diff": result["lp_diff"],
                    "strategy": strategy,
                    "phase": phase_idx + 1,
                    "improvement": result["improvement"],
                    "token_idx": ti,
                    "token_str": tok_str,
                    "attn_w": attn_w,
                }

            if result["fixed"]:
                return best, n_cf, True

    return best, n_cf, False


def joint_counterfactual_search(model, tokenizer, sample, ctx, top_tokens,
                                 base_lp_diff, device, sample_idx):
    best_single, n_cf_single, fixed = find_best_single_strategy(
        model, tokenizer, sample, ctx, top_tokens, base_lp_diff, device)

    if fixed:
        return {
            "lp_diff": best_single["lp_diff"],
            "strategy": f"auto:{best_single['strategy']}",
            "phase": best_single["phase"],
            "improvement": best_single["improvement"],
            "fixable": True,
            "n_counterfactuals": n_cf_single,
            "joint_depth": 1,
            "joint_tokens": [(best_single["token_idx"], best_single["token_str"],
                              best_single["strategy"])],
            "details": f"single:{best_single['strategy']} token[{best_single['token_idx']}]"
                       f"='{best_single['token_str']}'",
        }

    if best_single["strategy"] == "none":
        return {
            "lp_diff": base_lp_diff,
            "strategy": "auto:none",
            "phase": 0,
            "improvement": 0.0,
            "fixable": False,
            "n_counterfactuals": n_cf_single,
            "joint_depth": 0,
            "joint_tokens": [],
            "details": "no single strategy helped",
        }

    best_lp_diff = best_single["lp_diff"]
    best_detail = f"single:{best_single['strategy']} token[{best_single['token_idx']}]='{best_single['token_str']}'"
    best_interventions = [(best_single["token_idx"], best_single["strategy"])]
    total_cf = n_cf_single

    for depth in range(2, MAX_JOINT_DEPTH + 1):
        current_best_lp_diff = best_lp_diff
        found_better = False

        strategies = ["prune", "neutralize", "sentence"]
        for phase_idx, strategy in enumerate(strategies):
            for ti, attn_w, tok_str in top_tokens:
                if any(ti == t[0] for t in best_interventions):
                    continue

                candidate_interventions = best_interventions + [(ti, strategy)]
                new_ctx = apply_joint_interventions(ctx, tokenizer, candidate_interventions)
                if new_ctx == ctx or len(new_ctx.strip()) < 10:
                    continue

                total_cf += 1
                result = counterfactual_eval(model, tokenizer, sample, new_ctx, base_lp_diff, device)

                joint_tokens_str = " + ".join(
                    f"{s}({t})" for t, s in candidate_interventions)
                log(f"    [joint d={depth}] {joint_tokens_str} "
                    f"lp_diff={result['lp_diff']:+.4f} "
                    f"({'+' if result['improvement'] > 0 else ''}{result['improvement']:+.4f})"
                    f"{' FIXED!' if result['fixed'] else ''}")

                if result["lp_diff"] < current_best_lp_diff:
                    current_best_lp_diff = result["lp_diff"]
                    best_lp_diff = result["lp_diff"]
                    best_detail = f"joint(d={depth}): {joint_tokens_str}"
                    best_interventions = candidate_interventions
                    found_better = True

                if result["fixed"]:
                    return {
                        "lp_diff": result["lp_diff"],
                        "strategy": "auto:joint",
                        "phase": best_single["phase"],
                        "improvement": result["improvement"],
                        "fixable": True,
                        "n_counterfactuals": total_cf,
                        "joint_depth": depth,
                        "joint_tokens": [(t, s) for t, s in candidate_interventions],
                        "details": f"joint(d={depth}): {joint_tokens_str}",
                    }

        if not found_better:
            break

    return {
        "lp_diff": best_lp_diff,
        "strategy": "auto:joint" if len(best_interventions) > 1 else f"auto:{best_single['strategy']}",
        "phase": best_single["phase"],
        "improvement": base_lp_diff - best_lp_diff,
        "fixable": best_lp_diff <= 0,
        "n_counterfactuals": total_cf,
        "joint_depth": len(best_interventions),
        "joint_tokens": [(t, s) for t, s in best_interventions],
        "details": best_detail,
    }


def full_token_attribution(model, tokenizer, sample, device, sample_idx):
    base = compute_lp_sample(model, tokenizer, sample, device)
    base_lp_diff = base["lp_diff"]

    full_text = base["prompt"]
    attn_weights, all_tokens = get_deep_attention(model, tokenizer, full_text, device)
    if attn_weights is None:
        return None

    ctx = sample.get("context", "")
    n = len(attn_weights)

    attributions = []
    n_tested = 0
    for i in range(n - 1):
        tok = all_tokens[i] if i < len(all_tokens) else "?"
        if tok in ("", "<|endoftext|>", "\n"):
            continue

        sent_idx = find_token_sentence(i, ctx, tokenizer)

        for strategy in ["prune", "neutralize"]:
            new_ctx = apply_intervention(ctx, tokenizer, i, strategy, sent_idx)
            if new_ctx == ctx or len(new_ctx.strip()) < 10:
                continue

            n_tested += 1
            result = counterfactual_eval(model, tokenizer, sample, new_ctx, base_lp_diff, device)
            attributions.append({
                "token_idx": i,
                "token_str": tok,
                "attn_weight": round(attn_weights[i], 6),
                "strategy": strategy,
                "lp_diff": round(result["lp_diff"], 4),
                "improvement": round(result["improvement"], 4),
                "fixed": result["fixed"],
                "sent_idx": sent_idx,
            })

    if not attributions:
        return None

    attributions.sort(key=lambda x: x["improvement"], reverse=True)

    attn_vals = [a["attn_weight"] for a in attributions]
    imp_vals = [a["improvement"] for a in attributions]
    correlation = float(np.corrcoef(attn_vals, imp_vals)[0, 1]) if len(attn_vals) > 2 else 0.0

    log(f"\n  [Full Attribution] Sample {sample_idx}: {n_tested} counterfactuals")
    log(f"    Corr(attention, improvement) = {correlation:+.4f}")
    log(f"    Top-5 by causal impact:")
    for a in attributions[:5]:
        log(f"      token[{a['token_idx']}]='{a['token_str']}' "
            f"attn={a['attn_weight']:.4f} {a['strategy']:12s} "
            f"Δ={a['improvement']:+.4f} lp_diff={a['lp_diff']:+.4f}"
            f"{' FIXED!' if a['fixed'] else ''}")
    log(f"    Top-5 by attention weight:")
    by_attn = sorted(attributions, key=lambda x: x["attn_weight"], reverse=True)
    for a in by_attn[:5]:
        log(f"      token[{a['token_idx']}]='{a['token_str']}' "
            f"attn={a['attn_weight']:.4f} {a['strategy']:12s} "
            f"Δ={a['improvement']:+.4f}")

    return {
        "sample_idx": sample_idx,
        "question": sample.get("question", "")[:60],
        "base_lp_diff": round(base_lp_diff, 4),
        "n_counterfactuals": n_tested,
        "n_tokens": n,
        "attn_imp_correlation": round(correlation, 4),
        "top_by_impact": attributions[:10],
        "top_by_attention": by_attn[:10],
    }


def repairability_score(lp_diff_before, lp_diff_after):
    if lp_diff_after <= 0:
        return 1.0
    if lp_diff_after >= lp_diff_before:
        return 0.0
    return (lp_diff_before - lp_diff_after) / lp_diff_before


def unified_repair(model, tokenizer, sample, device, sample_idx, do_full_attribution=False):
    base = compute_lp_sample(model, tokenizer, sample, device)
    if not base["hallucinates"]:
        return {**base, "strategy": "none", "phase": 0,
                "improvement": 0.0, "fixable": False,
                "n_counterfactuals": 0, "joint_depth": 0,
                "joint_tokens": [], "attribution": None, "details": ""}

    is_unans = sample.get("answerability") == "unanswerable"
    if not is_unans:
        return {**base, "strategy": "none", "phase": 0,
                "improvement": 0.0, "fixable": False,
                "n_counterfactuals": 0, "joint_depth": 0,
                "joint_tokens": [], "attribution": None, "details": ""}

    full_text = base["prompt"]
    attn_weights, all_tokens = get_deep_attention(model, tokenizer, full_text, device)
    if attn_weights is None:
        return {**base, "strategy": "none", "phase": 0,
                "improvement": 0.0, "fixable": False,
                "n_counterfactuals": 0, "joint_depth": 0,
                "joint_tokens": [], "attribution": None, "details": "attention failed"}

    ctx = sample.get("context", "")
    n = len(attn_weights)
    all_ranked = rank_tokens_by_attention(attn_weights, all_tokens, n)
    top_tokens = all_ranked[:TOP_K_TOKENS]

    log(f"\n  Sample {sample_idx:>3d}: base lp_diff={base['lp_diff']:+.4f}")
    log(f"    Top attention: {[(t[2], f'{t[1]:.4f}') for t in top_tokens]}")

    joint_result = joint_counterfactual_search(
        model, tokenizer, sample, ctx, top_tokens, base["lp_diff"], device, sample_idx)

    attribution = None
    if do_full_attribution and not joint_result["fixable"]:
        attribution = full_token_attribution(model, tokenizer, sample, device, sample_idx)

    score = repairability_score(base["lp_diff"], joint_result["lp_diff"])

    log(f"    Result: {joint_result['details']} "
        f"lp_diff={joint_result['lp_diff']:+.4f} "
        f"score={score:.2f}"
        f"{' FIXED!' if joint_result['fixable'] else ''}")

    return {
        **base,
        "lp_diff": joint_result["lp_diff"],
        "pref_positive": joint_result["lp_diff"] > 0,
        "hallucinates": joint_result["lp_diff"] > 0,
        "strategy": joint_result["strategy"],
        "phase": joint_result["phase"],
        "improvement": round(joint_result["improvement"], 4),
        "fixable": joint_result["fixable"],
        "n_counterfactuals": joint_result["n_counterfactuals"],
        "joint_depth": joint_result["joint_depth"],
        "joint_tokens": joint_result["joint_tokens"],
        "repairability_score": round(score, 4),
        "attribution": attribution,
        "details": joint_result["details"],
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
    log("P23: Joint Counterfactual Search + Full-Token Causal Attribution")
    log("Multi-token joint repair + attention-as-causal-proxy validation")
    log("=" * 64)
    t0 = time.time()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    log("\n[Step 1] Loading base model (Qwen2.5-0.5B, eager attn)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="cpu",
        attn_implementation="eager", output_attentions=True,
        output_hidden_states=True)
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
    base_results = [compute_lp_sample(model, tokenizer, s, device) for s in test_samples]
    base_metrics = compute_metrics(base_results, test_samples)
    log(f"  BASELINE: H={base_metrics['H']:.4f}, C={base_metrics['C']:.4f}")

    hallucinated_indices = [
        i for i, (r, s) in enumerate(zip(base_results, test_samples))
        if r["hallucinates"]
    ]
    log(f"  Hallucinated samples: {len(hallucinated_indices)}")

    log(f"\n[Step 4] Joint Counterfactual Search + Full Attribution...")
    log(f"  Deep layers: {DEEP_LAYERS[0]}-{DEEP_LAYERS[1]}")
    log(f"  Top-K: {TOP_K_TOKENS}, Max joint depth: {MAX_JOINT_DEPTH}")
    log(f"  Full attribution on up to {FULL_ATTRIBUTION_SAMPLES} unfixed samples")
    log(f"")

    repaired_results = []
    t_repair = time.time()

    full_attribution_count = 0
    for idx, sample in enumerate(test_samples):
        do_full = (full_attribution_count < FULL_ATTRIBUTION_SAMPLES and
                   idx in hallucinated_indices)
        result = unified_repair(model, tokenizer, sample, device, idx, do_full)
        if result.get("attribution") is not None:
            full_attribution_count += 1
        repaired_results.append(result)

    repair_time = time.time() - t_repair
    post_metrics = compute_metrics(repaired_results, test_samples)

    log(f"\n{'=' * 64}")
    log(f"[Summary] P23 Joint Counterfactual Search")
    log(f"  Baseline:    H={base_metrics['H']:.4f}, C={base_metrics['C']:.4f}")
    log(f"  After P23:   H={post_metrics['H']:.4f}, C={post_metrics['C']:.4f}")
    log(f"  ΔH: {post_metrics['H'] - base_metrics['H']:+.4f}")

    n_hall = sum(1 for r in repaired_results if r.get("hallucinates"))
    n_fixable = sum(1 for r in repaired_results if r.get("fixable"))
    total_cf = sum(r.get("n_counterfactuals", 0) for r in repaired_results)

    strategy_counts = {}
    strategy_fixes = {}
    for r in repaired_results:
        s = r.get("strategy", "none")
        strategy_counts[s] = strategy_counts.get(s, 0) + 1
        if r.get("fixable"):
            strategy_fixes[s] = strategy_fixes.get(s, 0) + 1

    log(f"\n  Repair Statistics:")
    log(f"    Hallucinated (before): {len(hallucinated_indices)}")
    log(f"    Fixed:                 {n_fixable}")
    log(f"    Total counterfactuals: {total_cf}")
    log(f"    Strategy distribution: {strategy_counts}")
    log(f"    Strategy fixes:        {strategy_fixes}")

    log(f"\n  Per-sample repairability:")
    for idx in hallucinated_indices:
        r = repaired_results[idx]
        s = test_samples[idx]
        log(f"    #{idx:>2d} [{s.get('_position','?'):5s}] {s.get('question','')[:45]:45s} "
            f"score={r.get('repairability_score',0):.2f} "
            f"Δ={r.get('lp_diff',0):+.4f} "
            f"d={r.get('joint_depth',0)} "
            f"{'FIXED' if r.get('fixable') else ''}")

    correlations = [
        r["attribution"]["attn_imp_correlation"]
        for r in repaired_results if r.get("attribution") is not None
    ]
    if correlations:
        log(f"\n  Attention-Causality Validation:")
        log(f"    Corr(alpha, Delta_lp_diff) per sample: {[round(c,4) for c in correlations]}")
        log(f"    Mean correlation: {sum(correlations)/len(correlations):+.4f}")

    elapsed = time.time() - t0
    log(f"\n  Repair: {repair_time:.0f}s, Total: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    results = {
        "config": {
            "deep_layers": list(DEEP_LAYERS),
            "top_k_tokens": TOP_K_TOKENS,
            "max_joint_depth": MAX_JOINT_DEPTH,
            "method": "joint counterfactual search + full token attribution",
        },
        "baseline_H": base_metrics["H"],
        "baseline_C": base_metrics["C"],
        "post_repair_H": post_metrics["H"],
        "post_repair_C": post_metrics["C"],
        "delta_H": round(post_metrics["H"] - base_metrics["H"], 4),
        "n_samples": len(test_samples),
        "n_hallucinated": len(hallucinated_indices),
        "n_fixed": n_fixable,
        "total_counterfactuals": total_cf,
        "strategy_distribution": strategy_counts,
        "strategy_fixes": strategy_fixes,
        "repairs": [{
            "idx": i,
            "position": s.get("_position", "?"),
            "question": s.get("question", "")[:60],
            "strategy": r.get("strategy", "none"),
            "joint_depth": r.get("joint_depth", 0),
            "lp_diff_before": r.get("lp_diff", 0),
            "improvement": r.get("improvement", 0),
            "fixable": r.get("fixable", False),
            "repairability_score": r.get("repairability_score", 0),
            "details": r.get("details", ""),
            "n_counterfactuals": r.get("n_counterfactuals", 0),
            "attribution_correlation": r.get("attribution", {}).get("attn_imp_correlation")
            if r.get("attribution") else None,
        } for i, (r, s) in enumerate(zip(repaired_results, test_samples))
          if r.get("strategy", "none") != "none" or r.get("fixable")],
        "attributions": [r["attribution"] for r in repaired_results
                         if r.get("attribution") is not None],
        "time_s": round(elapsed, 1),
        "repair_time_s": round(repair_time, 1),
    }

    with open(os.path.join(RESULTS_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log(f"\nResults saved to {RESULTS_DIR}/results.json")
    log(f"\nP23 Complete.")


if __name__ == "__main__":
    main()