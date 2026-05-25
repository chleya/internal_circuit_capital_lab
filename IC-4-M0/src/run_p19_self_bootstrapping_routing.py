"""
P19: Self-Bootstrapping Attention Rerouting (SBAR)
===================================================
Minimal demonstration combining Meta FAIR's self-bootstrapping paradigm
with IC-4-M0's diagnostic framework (P13-P18).

The agent loop — for each hallucinated unanswerable sample:
  DETECT   → log-prob comparison: does model prefer hallucination?
  DIAGNOSE → deep-layer (16-23) attention: which tokens are most-attended?
  REPAIR   → iteratively prune high-attention tokens, re-evaluate log-prob
  VERIFY   → accept repair if hallucination gap decreases
  REMEMBER → JSON memory: which samples were fixable?

Key constraint: CPU-friendly, forward-pass only, no gradient computation.

Leverages P18 finding: DEEP layers (16-23) are the sufficient core of routing.
Search is directed to the known-effective subspace rather than blind.

Usage:
  python src/run_p19_self_bootstrapping_routing.py
"""

import os, sys, time, json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_loader import load_model_and_tokenizer
from src.data_builder import load_jsonl

RESULTS_DIR = "results_p19_self_bootstrapping"
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, "run_log.txt")

DEEP_LAYERS = (16, 23)
MAX_PRUNING_ITERATIONS = 3
TOP_K_TOKENS_PER_ITER = 3


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


def compute_h_sample(model, tokenizer, sample, device):
    ctx = sample.get("context", "")
    q = sample.get("question", "")
    prompt = f"{ctx}\n\nQuestion: {q}\nAnswer:"
    lp_pos = logprob_of_response(model, tokenizer, prompt,
                                  sample.get("positive_response", ""), device)
    lp_neg = logprob_of_response(model, tokenizer, prompt,
                                  sample.get("negative_response", ""), device)
    is_unanswerable = sample.get("answerability") == "unanswerable"
    hallucinates = is_unanswerable and (lp_pos > lp_neg)
    return {
        "lp_pos": lp_pos, "lp_neg": lp_neg,
        "lp_diff": lp_pos - lp_neg,
        "hallucinates": hallucinates,
        "pref_positive": lp_pos > lp_neg,
        "prompt": prompt,
    }


def get_deep_attention(model, tokenizer, text, device):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    enc = {k: v.to(device) for k, v in enc.items()}
    seq_len = enc["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**enc, output_attentions=True)

    all_attentions = outputs.attentions
    n_layers = len(all_attentions)
    start_layer = max(0, DEEP_LAYERS[0])
    end_layer = min(n_layers, DEEP_LAYERS[1] + 1)

    deep_attns = []
    for li in range(start_layer, end_layer):
        if li < n_layers:
            deep_attns.append(all_attentions[li])

    if not deep_attns:
        return None, None

    stacked = torch.stack(deep_attns)
    mean_attn = stacked.mean(dim=(0, 2))
    last_to_all = mean_attn[0, -1, :]

    all_tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())
    return last_to_all.cpu().tolist(), all_tokens


def prune_token_from_text(text, tokenizer, token_idx):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    token_ids = enc["input_ids"][0].tolist()
    if token_idx < 0 or token_idx >= len(token_ids):
        return text
    pruned_ids = token_ids[:token_idx] + token_ids[token_idx + 1:]
    return tokenizer.decode(pruned_ids, skip_special_tokens=True)


def self_bootstrap_sample(model, tokenizer, sample, device, sample_idx, memory):
    base = compute_h_sample(model, tokenizer, sample, device)
    if not base["hallucinates"]:
        return {**base, "pruned_count": 0, "pruned_tokens": [],
                "improvement": 0.0, "fixable": False, "attempted": False}

    is_unanswerable = sample.get("answerability") == "unanswerable"
    if not is_unanswerable:
        return {**base, "pruned_count": 0, "pruned_tokens": [],
                "improvement": 0.0, "fixable": False, "attempted": False}

    full_text = base["prompt"]
    attn_weights, all_tokens = get_deep_attention(model, tokenizer, full_text, device)
    if attn_weights is None:
        return {**base, "pruned_count": 0, "pruned_tokens": [],
                "improvement": 0.0, "fixable": False, "attempted": False}

    token_scores = []
    n = len(attn_weights)
    for i in range(n - 1):
        token_scores.append((i, attn_weights[i],
                             all_tokens[i] if i < len(all_tokens) else "?"))

    token_scores.sort(key=lambda x: x[1], reverse=True)

    best_text = full_text
    best_lp_diff = base["lp_diff"]
    pruned_idxs = []

    seen = set()
    iteration = 0
    while iteration < MAX_PRUNING_ITERATIONS:
        improved = False
        start = iteration * TOP_K_TOKENS_PER_ITER
        candidates = token_scores[start:start + TOP_K_TOKENS_PER_ITER]

        for token_idx, attn_val, token_str in candidates:
            if token_idx in seen:
                continue
            candidate_text = prune_token_from_text(best_text, tokenizer, token_idx)
            if candidate_text == best_text or len(candidate_text.strip()) < 10:
                seen.add(token_idx)
                continue

            candidate = compute_h_sample(model, tokenizer,
                                         {**sample, "context": _extract_context(candidate_text, sample)},
                                         device)
            lp_diff_new = candidate["lp_diff"]

            if lp_diff_new < best_lp_diff:
                best_lp_diff = lp_diff_new
                best_text = candidate_text
                pruned_idxs.append(token_idx)
                seen.add(token_idx)
                improved = True
                log(f"  Sample {sample_idx:>3d} iter {iteration}: pruned "
                    f"[{token_idx}]{token_str!r} "
                    f"Δ={base['lp_diff']:+.4f}→{lp_diff_new:+.4f}")
                break
            seen.add(token_idx)

        if not improved:
            break
        iteration += 1

    improvement = base["lp_diff"] - best_lp_diff
    fixable = (best_lp_diff <= 0)

    entry = {
        "sample_idx": sample_idx,
        "position": sample.get("_position", "?"),
        "question": sample.get("question", "")[:60],
        "lp_diff_before": base["lp_diff"],
        "lp_diff_after": best_lp_diff,
        "improvement": round(improvement, 4),
        "pruned_count": len(pruned_idxs),
        "pruned_indices": pruned_idxs,
        "fixable": fixable,
    }
    memory.append(entry)

    return {**base, "lp_diff": best_lp_diff, "pref_positive": best_lp_diff > 0,
            "hallucinates": best_lp_diff > 0,
            "pruned_count": len(pruned_idxs),
            "pruned_tokens": pruned_idxs,
            "improvement": round(improvement, 4),
            "fixable": fixable, "attempted": True}


def _extract_context(pruned_text, original_sample):
    q = original_sample.get("question", "")
    if f"\n\nQuestion: {q}" in pruned_text:
        return pruned_text.split(f"\n\nQuestion: {q}")[0]
    if "\n\nQuestion:" in pruned_text:
        idx = pruned_text.index("\n\nQuestion:")
        return pruned_text[:idx]
    return pruned_text


def _rebuild_prompt(ctx, sample):
    q = sample.get("question", "")
    return f"{ctx}\n\nQuestion: {q}\nAnswer:"


def main():
    log("=" * 64)
    log("P19: Self-Bootstrapping Attention Rerouting (SBAR)")
    log("Introspection-Guided Targeted Self-Repair")
    log("=" * 64)
    t0 = time.time()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    log("\n[Step 1] Loading base model (Qwen2.5-0.5B, no LoRA, eager attn)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="cpu",
        attn_implementation="eager", output_attentions=True,
        output_hidden_states=True)
    model.eval()
    device = next(model.parameters()).device
    log(f"  Device: {device}, Layers: {len(model.model.layers)}, Attn: eager")

    log("\n[Step 2] Loading test samples (position-balanced)...")
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

    log("\n[Step 3] BASELINE evaluation (no intervention)...")
    baseline_results = []
    for s in test_samples:
        baseline_results.append(compute_h_sample(model, tokenizer, s, device))

    def compute_metrics(results):
        ans = [r for (r, s) in zip(results, test_samples)
               if s.get("answerability") == "answerable"]
        unans = [r for (r, s) in zip(results, test_samples)
                 if s.get("answerability") == "unanswerable"]
        hall = sum(1 for r in unans if r["pref_positive"]) if unans else 0
        corr = sum(1 for r in ans if r["pref_positive"]) if ans else 0
        return {"H": round(hall / len(unans), 4) if unans else 0.0,
                "C": round(corr / len(ans), 4) if ans else 0.0}

    base_metrics = compute_metrics(baseline_results)
    log(f"  BASELINE: H={base_metrics['H']:.4f}, C={base_metrics['C']:.4f}")

    log(f"\n[Step 4] Self-Bootstrapping Attention Rerouting...")
    log(f"  Deep layers: {DEEP_LAYERS[0]}-{DEEP_LAYERS[1]}")
    log(f"  Max pruning iterations per sample: {MAX_PRUNING_ITERATIONS}")
    log(f"  Top-K tokens per iteration: {TOP_K_TOKENS_PER_ITER}")
    log(f"")

    memory = []
    repaired_results = []

    t_repair_start = time.time()
    for idx, sample in enumerate(test_samples):
        result = self_bootstrap_sample(model, tokenizer, sample, device, idx, memory)
        repaired_results.append(result)

    repair_time = time.time() - t_repair_start
    post_metrics = compute_metrics(repaired_results)

    log(f"\n{'=' * 64}")
    log(f"[Summary] P19 Self-Bootstrapping Attention Rerouting")
    log(f"  Deep layers used: {DEEP_LAYERS[0]}-{DEEP_LAYERS[1]}")
    log(f"  Baseline:    H={base_metrics['H']:.4f}, C={base_metrics['C']:.4f}")
    log(f"  After SBAR:  H={post_metrics['H']:.4f}, C={post_metrics['C']:.4f}")
    log(f"  ΔH: {post_metrics['H'] - base_metrics['H']:+.4f}")

    n_hall = sum(1 for r in repaired_results if r.get("hallucinates"))
    n_attempted = sum(1 for r in repaired_results if r.get("attempted"))
    n_fixable = sum(1 for r in repaired_results if r.get("fixable"))
    total_pruned = sum(r.get("pruned_count", 0) for r in repaired_results)
    avg_improvement = (sum(r.get("improvement", 0) for r in repaired_results if r.get("attempted"))
                       / max(n_attempted, 1))

    log(f"\n  Agent Statistics:")
    log(f"    Hallucinated samples: {n_hall}/{len(test_samples)}")
    log(f"    Attempted repairs:    {n_attempted}")
    log(f"    Fixable (H→0):        {n_fixable}")
    log(f"    Total tokens pruned:  {total_pruned}")
    log(f"    Avg improvement:      {avg_improvement:+.4f}")

    fixable_by_pos = {}
    for r in repaired_results:
        if r.get("fixable"):
            pos = next((s["_position"] for s in test_samples
                        if test_samples.index(s) == repaired_results.index(r)), "?")
            fixable_by_pos[pos] = fixable_by_pos.get(pos, 0) + 1
    if fixable_by_pos:
        log(f"    Fixable by position:  {fixable_by_pos}")

    elapsed = time.time() - t0
    log(f"\n  Repair time: {repair_time:.0f}s, Total: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    results = {
        "config": {
            "deep_layers": list(DEEP_LAYERS),
            "max_pruning_iterations": MAX_PRUNING_ITERATIONS,
            "top_k_per_iter": TOP_K_TOKENS_PER_ITER,
        },
        "baseline_H": base_metrics["H"],
        "baseline_C": base_metrics["C"],
        "post_repair_H": post_metrics["H"],
        "post_repair_C": post_metrics["C"],
        "delta_H": round(post_metrics["H"] - base_metrics["H"], 4),
        "n_samples": len(test_samples),
        "n_hallucinated": n_hall,
        "n_attempted": n_attempted,
        "n_fixable": n_fixable,
        "total_pruned": total_pruned,
        "avg_improvement": round(avg_improvement, 4),
        "memory": memory,
        "time_s": round(elapsed, 1),
        "repair_time_s": round(repair_time, 1),
    }

    with open(os.path.join(RESULTS_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(os.path.join(RESULTS_DIR, "memory.json"), "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)

    log(f"\nResults saved to {RESULTS_DIR}/results.json")
    log(f"Memory  saved to {RESULTS_DIR}/memory.json")
    log(f"\nP19 Complete. v20.0")


if __name__ == "__main__":
    main()