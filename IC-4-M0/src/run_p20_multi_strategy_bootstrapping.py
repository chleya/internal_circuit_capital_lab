"""
P20: Multi-Strategy Self-Bootstrapping
=====================================
Tests three repair strategies and lets the agent pick the best per sample.

Strategies:
  PRUNE    — Remove high-attention distractor token (P19 baseline)
  NEUTRALIZE — Replace distractor token with "it" (preserves structure)
  SENTENCE — Remove the entire sentence containing the distractor

Key question: Is the distractor effect at the TOKEN level or SENTENCE level?
Can the agent learn which strategy to use per sample type?

Conditions:
  NO_REPAIR     — Baseline (no intervention)
  PRUNE         — Token removal only
  NEUTRALIZE    — Token replacement only
  SENTENCE      — Sentence removal only
  AGENT_BEST    — Agent picks best strategy per sample

Usage:
  python src/run_p20_multi_strategy_bootstrapping.py
"""

import os, sys, time, json, re
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_loader import load_model_and_tokenizer
from src.data_builder import load_jsonl

RESULTS_DIR = "results_p20_multi_strategy_bootstrapping"
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, "run_log.txt")

DEEP_LAYERS = (16, 23)
TOP_K_TOKENS = 3
MAX_ITERATIONS = 2
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
        log(f"  [WARN] Attention extraction failed: {e}")
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


def apply_strategy(text, tokenizer, token_idx, strategy, sent_idx=0):
    if strategy == "PRUNE":
        return remove_token_from_text(text, tokenizer, token_idx)
    elif strategy == "NEUTRALIZE":
        return neutralize_token_in_text(text, tokenizer, token_idx)
    elif strategy == "SENTENCE":
        return remove_sentence(text, sent_idx)
    return text


def try_strategy(model, tokenizer, sample, base_lp_diff, current_text,
                 token_idx, token_str, strategy, sent_idx, device):
    if strategy == "SENTENCE" and sent_idx < 0:
        return None

    candidate_text = apply_strategy(current_text, tokenizer, token_idx, strategy, sent_idx)

    if (candidate_text == current_text or
            len(candidate_text.strip()) < 10):
        return None

    if "\n\nQuestion:" in current_text and "\n\nQuestion:" not in candidate_text:
        return None

    ctx = candidate_text.split("\n\nQuestion:")[0] if "\n\nQuestion:" in candidate_text else candidate_text
    q = sample.get("question", "")

    lp_pos = logprob_of_response(model, tokenizer,
                                  f"{ctx}\n\nQuestion: {q}\nAnswer:",
                                  sample.get("positive_response", ""), device)
    lp_neg = logprob_of_response(model, tokenizer,
                                  f"{ctx}\n\nQuestion: {q}\nAnswer:",
                                  sample.get("negative_response", ""), device)
    lp_diff_new = lp_pos - lp_neg

    return {
        "strategy": strategy,
        "token_idx": token_idx,
        "token_str": token_str,
        "lp_diff_new": lp_diff_new,
        "improvement": base_lp_diff - lp_diff_new,
        "text": candidate_text,
    }


def self_bootstrap_multi(model, tokenizer, sample, device, sample_idx, memory):
    base = compute_lp_sample(model, tokenizer, sample, device)
    if not base["hallucinates"]:
        return {**base, "strategy_used": "none", "improvement": 0.0,
                "fixable": False, "attempted": False}, memory

    is_unans = sample.get("answerability") == "unanswerable"
    if not is_unans:
        return {**base, "strategy_used": "none", "improvement": 0.0,
                "fixable": False, "attempted": False}, memory

    full_text = base["prompt"]
    attn_weights, all_tokens = get_deep_attention(model, tokenizer, full_text, device)
    if attn_weights is None:
        return {**base, "strategy_used": "none", "improvement": 0.0,
                "fixable": False, "attempted": False}, memory

    n = len(attn_weights)
    token_scores = []
    for i in range(n - 1):
        token_scores.append((i, attn_weights[i],
                             all_tokens[i] if i < len(all_tokens) else "?"))

    token_scores.sort(key=lambda x: x[1], reverse=True)

    best_result = None
    best_lp_diff = base["lp_diff"]
    best_text = full_text
    seen_tokens = set()

    for iteration in range(MAX_ITERATIONS):
        improved = False
        start = iteration * TOP_K_TOKENS
        candidates = token_scores[start:start + TOP_K_TOKENS]

        for token_idx, attn_val, token_str in candidates:
            if token_idx in seen_tokens:
                continue
            sent_idx = find_token_sentence(token_idx, full_text, tokenizer)

            for strategy in ["PRUNE", "NEUTRALIZE", "SENTENCE"]:
                if strategy == "SENTENCE" and sent_idx < 0:
                    continue

                result = try_strategy(model, tokenizer, sample,
                                      base["lp_diff"], best_text,
                                      token_idx, token_str, strategy,
                                      sent_idx, device)
                if result is None:
                    continue

                if result["lp_diff_new"] < best_lp_diff:
                    best_lp_diff = result["lp_diff_new"]
                    best_text = result["text"]
                    best_result = result
                    seen_tokens.add(token_idx)
                    improved = True
                    log(f"  Sample {sample_idx:>3d} iter {iteration}: "
                        f"{strategy:>10s} [{token_idx}]{token_str!r} "
                        f"Δ={base['lp_diff']:+.4f}→{result['lp_diff_new']:+.4f}")
                    break

            if improved:
                break

        if not improved:
            break

    if best_result is None:
        return {**base, "strategy_used": "none", "improvement": 0.0,
                "fixable": False, "attempted": True}, memory

    improvement = base["lp_diff"] - best_lp_diff
    fixable = (best_lp_diff <= 0)

    entry = {
        "sample_idx": sample_idx,
        "position": sample.get("_position", "?"),
        "question": sample.get("question", "")[:60],
        "strategy_used": best_result["strategy"],
        "token_pruned": best_result["token_str"],
        "lp_diff_before": base["lp_diff"],
        "lp_diff_after": best_lp_diff,
        "improvement": round(improvement, 4),
        "fixable": fixable,
    }
    memory.append(entry)

    return {**base, "lp_diff": best_lp_diff,
            "pref_positive": best_lp_diff > 0,
            "hallucinates": best_lp_diff > 0,
            "strategy_used": best_result["strategy"],
            "improvement": round(improvement, 4),
            "fixable": fixable, "attempted": True}, memory


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
    log("P20: Multi-Strategy Self-Bootstrapping")
    log("PRUNE | NEUTRALIZE | SENTENCE — Agent Picks Best")
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
            for s in load_jsonl(path)[:12]:
                s["_position"] = pos
                test_samples.append(s)
    n_ans = sum(1 for s in test_samples if s.get("answerability") == "answerable")
    n_unans = sum(1 for s in test_samples if s.get("answerability") == "unanswerable")
    log(f"  Loaded {len(test_samples)} samples ({n_ans}A + {n_unans}U)")

    log("\n[Step 3] BASELINE evaluation...")
    base_results = [compute_lp_sample(model, tokenizer, s, device) for s in test_samples]
    base_metrics = compute_metrics(base_results, test_samples)
    log(f"  BASELINE: H={base_metrics['H']:.4f}, C={base_metrics['C']:.4f}")

    log(f"\n[Step 4] Multi-Strategy Self-Bootstrapping...")
    log(f"  Deep layers: {DEEP_LAYERS[0]}-{DEEP_LAYERS[1]}")
    log(f"  Strategies: PRUNE | NEUTRALIZE | SENTENCE")
    log(f"")

    memory = []
    repaired_results = []

    t_repair = time.time()
    for idx, sample in enumerate(test_samples):
        result, memory = self_bootstrap_multi(model, tokenizer, sample,
                                              device, idx, memory)
        repaired_results.append(result)

    repair_time = time.time() - t_repair
    post_metrics = compute_metrics(repaired_results, test_samples)

    log(f"\n{'=' * 64}")
    log(f"[Summary] P20 Multi-Strategy Self-Bootstrapping")
    log(f"  Baseline:    H={base_metrics['H']:.4f}, C={base_metrics['C']:.4f}")
    log(f"  After SBAR:  H={post_metrics['H']:.4f}, C={post_metrics['C']:.4f}")
    log(f"  ΔH: {post_metrics['H'] - base_metrics['H']:+.4f}")

    n_hall = sum(1 for r in repaired_results if r.get("hallucinates"))
    n_attempted = sum(1 for r in repaired_results if r.get("attempted"))
    n_fixable = sum(1 for r in repaired_results if r.get("fixable"))

    strategy_counts = {}
    strategy_fixable = {}
    for r in repaired_results:
        s = r.get("strategy_used", "none")
        strategy_counts[s] = strategy_counts.get(s, 0) + 1
        if r.get("fixable"):
            strategy_fixable[s] = strategy_fixable.get(s, 0) + 1

    avg_imp = (sum(r.get("improvement", 0) for r in repaired_results if r.get("attempted"))
               / max(n_attempted, 1))

    log(f"\n  Agent Statistics:")
    log(f"    Hallucinated:  {n_hall}/{len(test_samples)}")
    log(f"    Attempted:     {n_attempted}")
    log(f"    Fixable:       {n_fixable}")
    log(f"    Avg improv:    {avg_imp:+.4f}")
    log(f"    Strategy usage: {strategy_counts}")
    log(f"    Strategy fixes: {strategy_fixable}")

    elapsed = time.time() - t0
    log(f"\n  Repair: {repair_time:.0f}s, Total: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    results = {
        "config": {
            "deep_layers": list(DEEP_LAYERS),
            "strategies": ["PRUNE", "NEUTRALIZE", "SENTENCE"],
            "max_iterations": MAX_ITERATIONS,
            "top_k_tokens": TOP_K_TOKENS,
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
        "avg_improvement": round(avg_imp, 4),
        "strategy_usage": strategy_counts,
        "strategy_fixable": strategy_fixable,
        "memory": memory,
        "time_s": round(elapsed, 1),
        "repair_time_s": round(repair_time, 1),
    }

    with open(os.path.join(RESULTS_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(RESULTS_DIR, "memory.json"), "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)

    log(f"\nResults saved to {RESULTS_DIR}/results.json")
    log(f"\nP20 Complete. v21.0")


if __name__ == "__main__":
    main()