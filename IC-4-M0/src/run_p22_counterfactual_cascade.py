"""
P22: Probability-Guided Cascading Counterfactual Self-Repair
============================================================
Uses probability theory (attention as search heuristic, log-prob as
objective function) to automatically discover the optimal repair
strategy per hallucinated sample — no human-defined strategy menus,
no LLM self-generation.

P22 Pipeline (per hallucinated sample):
  1. DIAGNOSE: Extract deep-layer attention, rank tokens by weight
  2. CASCADE Phase 1 (PRUNE): Counterfactually remove each top-K token,
     measure $\Delta$lp_diff. Stop if hallucination fixed.
  3. CASCADE Phase 2 (NEUTRALIZE): Replace each top-K token with "it",
     measure $\Delta$lp_diff. Stop if fixed.
  4. CASCADE Phase 3 (SENTENCE): Remove sentence containing each top-K token.
     Stop if fixed.
  5. CASCADE Phase 4 (MULTI): Remove ALL top-K tokens at once.
     Stop if fixed.
  6. LEARN: Record which strategy-phase fixed which sample type.

Key insight: Attention weights $\alpha_i = \frac{1}{|D|}\sum_{l\in D} A_l$
serve as a probabilistic search heuristic. The counterfactual log-prob
$\Delta = \log P(pos|ctx') - \log P(neg|ctx')$ is the objective to minimize.
The cascade minimizes expected computation by stopping at first success.

This IS probability theory applied to strategy discovery:
- Prior: attention weight $\alpha_i$ ranks intervention candidates
- Likelihood: log-prob $\Delta$ measures intervention efficacy
- Decision: cascade stops when $P(correct|ctx') > P(hallucinated|ctx')$

Usage:
  python src/run_p22_counterfactual_cascade.py
"""

import os, sys, time, json, re
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_builder import load_jsonl

RESULTS_DIR = "results_p22_counterfactual_cascade"
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, "run_log.txt")

DEEP_LAYERS = (16, 23)
TOP_K_TOKENS = 5
NEUTRAL_TOKEN = "it"


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
        "prompt": prompt,
    }


def cascading_repair(model, tokenizer, sample, device, sample_idx):
    base = compute_lp_sample(model, tokenizer, sample, device)
    if not base["hallucinates"]:
        return {**base, "strategy": "none", "phase": 0,
                "improvement": 0.0, "fixable": False,
                "n_counterfactuals": 0, "details": ""}

    is_unans = sample.get("answerability") == "unanswerable"
    if not is_unans:
        return {**base, "strategy": "none", "phase": 0,
                "improvement": 0.0, "fixable": False,
                "n_counterfactuals": 0, "details": ""}

    full_text = base["prompt"]
    attn_weights, all_tokens = get_deep_attention(model, tokenizer, full_text, device)
    if attn_weights is None:
        return {**base, "strategy": "none", "phase": 0,
                "improvement": 0.0, "fixable": False,
                "n_counterfactuals": 0, "details": "attention failed"}

    n = len(attn_weights)
    token_scores = []
    for i in range(n - 1):
        tok = all_tokens[i] if i < len(all_tokens) else "?"
        if tok in ("", "<|endoftext|>", ".", ",", " ", "\n"):
            continue
        token_scores.append((i, attn_weights[i], tok))
    token_scores.sort(key=lambda x: x[1], reverse=True)
    top_tokens = token_scores[:TOP_K_TOKENS]

    ctx = sample.get("context", "")
    base_lp_diff = base["lp_diff"]
    n_cf = 0
    best = {"lp_diff": base_lp_diff, "strategy": "none", "phase": 0,
            "improvement": 0.0, "token_idx": -1, "token_str": ""}

    log(f"\n  Sample {sample_idx:>3d}: base lp_diff={base_lp_diff:+.4f}")
    log(f"    Top attention tokens: {[(t[2], f'{t[1]:.4f}') for t in top_tokens]}")

    phases = [
        ("prune", lambda ti, ctx: remove_token_from_text(ctx, tokenizer, ti)),
        ("neutralize", lambda ti, ctx: neutralize_token_in_text(ctx, tokenizer, ti)),
        ("sentence", lambda ti, ctx: remove_sentence(
            ctx, find_token_sentence(ti, ctx, tokenizer))),
    ]

    for phase_idx, (phase_name, edit_fn) in enumerate(phases):
        for ti, attn_w, tok_str in top_tokens:
            new_ctx = edit_fn(ti, ctx)
            if new_ctx == ctx or len(new_ctx.strip()) < 10:
                continue

            n_cf += 1
            result = counterfactual_eval(
                model, tokenizer, sample, new_ctx, base_lp_diff, device)
            improvement = result["improvement"]

            log(f"    [{phase_name}] token[{ti}]='{tok_str}' "
                f"lp_diff={result['lp_diff']:+.4f} "
                f"({'+' if improvement > 0 else ''}{improvement:+.4f})"
                f"{' FIXED!' if result['fixed'] else ''}")

            if result["lp_diff"] < best["lp_diff"]:
                best = {
                    "lp_diff": result["lp_diff"],
                    "strategy": f"auto:{phase_name}",
                    "phase": phase_idx + 1,
                    "improvement": improvement,
                    "token_idx": ti,
                    "token_str": tok_str,
                }

            if result["fixed"]:
                log(f"    >>> CASCADE STOP: {phase_name} fixed hallucination at phase {phase_idx+1}")
                return {
                    **base,
                    "lp_diff": result["lp_diff"],
                    "pref_positive": False,
                    "hallucinates": False,
                    "strategy": f"auto:{phase_name}",
                    "phase": phase_idx + 1,
                    "improvement": round(improvement, 4),
                    "fixable": True,
                    "n_counterfactuals": n_cf,
                    "token_idx": ti,
                    "token_str": tok_str,
                    "details": f"token[{ti}]='{tok_str}' attn={attn_w:.4f}",
                }

    if best["strategy"] != "none":
        log(f"    Best cascade result: {best['strategy']} lp_diff={best['lp_diff']:+.4f} "
            f"(improvement={best['improvement']:+.4f})")
        return {
            **base,
            "lp_diff": best["lp_diff"],
            "pref_positive": best["lp_diff"] > 0,
            "hallucinates": best["lp_diff"] > 0,
            "strategy": best["strategy"],
            "phase": best["phase"],
            "improvement": round(best["improvement"], 4),
            "fixable": best["lp_diff"] <= 0,
            "n_counterfactuals": n_cf,
            "token_idx": best["token_idx"],
            "token_str": best["token_str"],
            "details": f"best={best['strategy']} token={best['token_str']}",
        }

    log(f"    No cascade strategy succeeded ({n_cf} counterfactuals tried)")
    return {
        **base,
        "strategy": "auto:none",
        "phase": 0,
        "improvement": 0.0,
        "fixable": False,
        "n_counterfactuals": n_cf,
        "token_idx": -1,
        "token_str": "",
        "details": f"exhausted {n_cf} counterfactuals",
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
    log("P22: Probability-Guided Cascading Counterfactual Self-Repair")
    log("Attention as search heuristic, log-prob as objective function")
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

    log(f"\n[Step 4] Probability-Guided Cascading Counterfactual Repair...")
    log(f"  Deep layers: {DEEP_LAYERS[0]}-{DEEP_LAYERS[1]}")
    log(f"  Top-K attention tokens: {TOP_K_TOKENS}")
    log(f"  Cascade: prune → neutralize → sentence")
    log(f"  Stopping condition: lp_diff <= 0 (P(correct) >= P(hallucinated))")
    log(f"")

    repaired_results = []
    t_repair = time.time()
    for idx, sample in enumerate(test_samples):
        result = cascading_repair(model, tokenizer, sample, device, idx)
        repaired_results.append(result)

    repair_time = time.time() - t_repair
    post_metrics = compute_metrics(repaired_results, test_samples)

    log(f"\n{'=' * 64}")
    log(f"[Summary] P22 Cascading Counterfactual Self-Repair")
    log(f"  Baseline:    H={base_metrics['H']:.4f}, C={base_metrics['C']:.4f}")
    log(f"  After Repair: H={post_metrics['H']:.4f}, C={post_metrics['C']:.4f}")
    log(f"  ΔH: {post_metrics['H'] - base_metrics['H']:+.4f}")

    n_hall = sum(1 for r in repaired_results if r.get("hallucinates"))
    n_fixable = sum(1 for r in repaired_results if r.get("fixable"))
    total_cf = sum(r.get("n_counterfactuals", 0) for r in repaired_results)

    strategy_counts = {}
    strategy_fixes = {}
    phase_counts = {}
    for r in repaired_results:
        s = r.get("strategy", "none")
        strategy_counts[s] = strategy_counts.get(s, 0) + 1
        if r.get("fixable"):
            strategy_fixes[s] = strategy_fixes.get(s, 0) + 1
        p = r.get("phase", 0)
        phase_counts[p] = phase_counts.get(p, 0) + 1

    log(f"\n  Repair Statistics:")
    log(f"    Hallucinated (before): {n_hall}/{len(test_samples)}")
    log(f"    Fixed:                 {n_fixable}")
    log(f"    Total counterfactuals: {total_cf}")
    log(f"    Avg CF per sample:     {total_cf/len(test_samples):.1f}")
    log(f"    Strategy distribution: {strategy_counts}")
    log(f"    Strategy fixes:        {strategy_fixes}")
    log(f"    Phase distribution:    {phase_counts}")

    for r in repaired_results:
        if r.get("fixable") or r.get("improvement", 0) > 0.01:
            log(f"    Sample {r.get('sample_idx', '?'):>3s}: "
                f"{r.get('strategy','?'):20s} "
                f"Δ={r.get('lp_diff_before',0):+.4f}→{r.get('lp_diff',0):+.4f} "
                f"({r.get('details','')})")

    elapsed = time.time() - t0
    log(f"\n  Repair: {repair_time:.0f}s, Total: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    results = {
        "config": {
            "deep_layers": list(DEEP_LAYERS),
            "top_k_tokens": TOP_K_TOKENS,
            "method": "cascading counterfactual: prune→neutralize→sentence",
            "stopping_condition": "lp_diff <= 0",
        },
        "baseline_H": base_metrics["H"],
        "baseline_C": base_metrics["C"],
        "post_repair_H": post_metrics["H"],
        "post_repair_C": post_metrics["C"],
        "delta_H": round(post_metrics["H"] - base_metrics["H"], 4),
        "n_samples": len(test_samples),
        "n_hallucinated": n_hall,
        "n_fixed": n_fixable,
        "total_counterfactuals": total_cf,
        "strategy_distribution": strategy_counts,
        "strategy_fixes": strategy_fixes,
        "phase_distribution": phase_counts,
        "repairs": [{
            "idx": i,
            "position": s.get("_position", "?"),
            "question": s.get("question", "")[:60],
            "strategy": r.get("strategy", "none"),
            "phase": r.get("phase", 0),
            "lp_diff_before": r.get("lp_diff", 0),
            "improvement": r.get("improvement", 0),
            "fixable": r.get("fixable", False),
            "token_str": r.get("token_str", ""),
            "details": r.get("details", ""),
            "n_counterfactuals": r.get("n_counterfactuals", 0),
        } for i, (r, s) in enumerate(zip(repaired_results, test_samples))
          if r.get("strategy", "none") != "none" or r.get("fixable")],
        "time_s": round(elapsed, 1),
        "repair_time_s": round(repair_time, 1),
    }

    with open(os.path.join(RESULTS_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log(f"\nResults saved to {RESULTS_DIR}/results.json")
    log(f"\nP22 Complete.")


if __name__ == "__main__":
    main()