"""
P21: Self-Generated Strategy Discovery
======================================
The LLM DISCOVERS its own repair strategies through probability-guided
self-diagnosis — not by picking from human-defined menus (P19/P20).

P21 Pipeline (per hallucinated sample):
  1. DIAGNOSE: Extract deep-layer attention, identify top-attended tokens
  2. PROMPT: Tell the LLM what went wrong and ask it to rewrite the context
  3. GENERATE: LLM produces a self-generated context repair
  4. VERIFY: log-prob comparison — did the repair reduce hallucination?
  5. LEARN: Record self-discovered strategies in probability-weighted memory

Key insight: The log-prob signal serves as a probabilistic fitness function.
The LLM explores the space of possible repairs, and probability selects
which ones survive. This is evolutionary self-bootstrapping.

Usage:
  python src/run_p21_self_generated_strategies.py
"""

import os, sys, time, json, re
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_builder import load_jsonl

RESULTS_DIR = "results_p21_self_generated_strategies"
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, "run_log.txt")

DEEP_LAYERS = (16, 23)
TOP_K_TOKENS = 5
MAX_NEW_TOKENS = 80
MAX_RETRIES = 2

SELF_DIAGNOSIS_PROMPT = """[SELF-DIAGNOSIS]
You are diagnosing your own attention routing failure.

CONTEXT: {context}

QUESTION: {question}

Your deep-layer attention (layers {deep_start}-{deep_end}) focused most on:
{attention_summary}

FAILURE: You preferred the hallucinated answer "{neg_response}"
over the correct response "{pos_response}"
(log-prob gap: {gap:+.4f})

TASK: Rewrite the CONTEXT above to fix this routing error.
Make minimal changes. Return ONLY the rewritten context, nothing else.

REWRITTEN CONTEXT:"""


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
        log(f"  [WARN] Attention failed: {e}")
        return None, None


def llm_self_diagnose(model, tokenizer, sample, attn_weights, all_tokens, base_result, device):
    ctx = sample.get("context", "")
    q = sample.get("question", "")
    pos = sample.get("positive_response", "")
    neg = sample.get("negative_response", "")

    n = len(attn_weights)
    token_scores = []
    for i in range(n - 1):
        token_scores.append((i, attn_weights[i],
                             all_tokens[i] if i < len(all_tokens) else "?"))
    token_scores.sort(key=lambda x: x[1], reverse=True)

    top_summary_lines = []
    for rank, (idx, score, tok) in enumerate(token_scores[:TOP_K_TOKENS]):
        top_summary_lines.append(
            f"  [{idx}] '{tok}' (attention={score:.4f})")

    attn_summary = "\n".join(top_summary_lines)

    prompt = SELF_DIAGNOSIS_PROMPT.format(
        context=ctx,
        question=q,
        deep_start=DEEP_LAYERS[0],
        deep_end=DEEP_LAYERS[1],
        attention_summary=attn_summary,
        neg_response=neg,
        pos_response=pos,
        gap=base_result["lp_diff"],
    )

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][enc["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return generated_text, attn_summary


def extract_context_from_generation(generated_text, original_ctx):
    if not generated_text or len(generated_text) < 10:
        return None

    if "REWRITTEN CONTEXT:" in generated_text:
        parts = generated_text.split("REWRITTEN CONTEXT:")
        if len(parts) > 1:
            generated_text = parts[-1].strip()

    lines = generated_text.strip().split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if line.startswith("[") or line.startswith("REASONING") or line.startswith("EDIT:"):
            continue
        if line.startswith("Context:") or line.startswith("CONTEXT:"):
            continue
        cleaned.append(line)

    result = " ".join(cleaned).strip()

    if not result or len(result) < 10:
        return None

    if result == original_ctx.strip():
        return None

    return result


def self_bootstrap_generated(model, tokenizer, sample, device, sample_idx, memory):
    base = compute_lp_sample(model, tokenizer, sample, device)
    if not base["hallucinates"]:
        return {**base, "strategy": "none", "generated_text": "",
                "improvement": 0.0, "fixable": False, "attempted": False}, memory

    is_unans = sample.get("answerability") == "unanswerable"
    if not is_unans:
        return {**base, "strategy": "none", "generated_text": "",
                "improvement": 0.0, "fixable": False, "attempted": False}, memory

    full_text = base["prompt"]
    attn_weights, all_tokens = get_deep_attention(model, tokenizer, full_text, device)
    if attn_weights is None:
        return {**base, "strategy": "none", "generated_text": "",
                "improvement": 0.0, "fixable": False, "attempted": True}, memory

    best_lp_diff = base["lp_diff"]
    best_ctx = sample.get("context", "")
    best_generated = ""
    best_attn_summary = ""

    for attempt in range(MAX_RETRIES):
        generated, attn_summary = llm_self_diagnose(
            model, tokenizer, sample, attn_weights, all_tokens, base, device)

        new_ctx = extract_context_from_generation(generated, sample.get("context", ""))
        if new_ctx is None:
            log(f"  Sample {sample_idx:>3d} attempt {attempt}: invalid generation, retrying")
            continue

        q = sample.get("question", "")
        lp_pos_new = logprob_of_response(model, tokenizer,
                                          f"{new_ctx}\n\nQuestion: {q}\nAnswer:",
                                          sample.get("positive_response", ""), device)
        lp_neg_new = logprob_of_response(model, tokenizer,
                                          f"{new_ctx}\n\nQuestion: {q}\nAnswer:",
                                          sample.get("negative_response", ""), device)
        lp_diff_new = lp_pos_new - lp_neg_new

        improvement = base["lp_diff"] - lp_diff_new

        original_len = len(sample.get("context", ""))
        new_len = len(new_ctx)

        log(f"  Sample {sample_idx:>3d} attempt {attempt}: "
            f"Δ={base['lp_diff']:+.4f}→{lp_diff_new:+.4f} "
            f"(ΔΔ={improvement:+.4f}) "
            f"ctx: {original_len}→{new_len} chars")

        if lp_diff_new < best_lp_diff:
            best_lp_diff = lp_diff_new
            best_ctx = new_ctx
            best_generated = generated
            best_attn_summary = attn_summary

        if lp_diff_new <= 0:
            break

    improvement = base["lp_diff"] - best_lp_diff
    fixable = (best_lp_diff <= 0)

    strategy_type = "none"
    if best_generated:
        if len(best_ctx) < len(sample.get("context", "")) * 0.8:
            strategy_type = "self:shorten"
        elif len(best_ctx) > len(sample.get("context", "")) * 1.2:
            strategy_type = "self:expand"
        else:
            strategy_type = "self:rewrite"

    entry = {
        "sample_idx": sample_idx,
        "position": sample.get("_position", "?"),
        "question": sample.get("question", "")[:60],
        "strategy": strategy_type,
        "lp_diff_before": base["lp_diff"],
        "lp_diff_after": best_lp_diff,
        "improvement": round(improvement, 4),
        "fixable": fixable,
        "generated_text": best_generated[:200] if best_generated else "",
        "ctx_len_before": len(sample.get("context", "")),
        "ctx_len_after": len(best_ctx),
        "attn_summary": best_attn_summary,
    }
    memory.append(entry)

    return {**base, "lp_diff": best_lp_diff,
            "pref_positive": best_lp_diff > 0,
            "hallucinates": best_lp_diff > 0,
            "strategy": strategy_type,
            "generated_text": best_generated[:200] if best_generated else "",
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
    log("P21: Self-Generated Strategy Discovery")
    log("LLM discovers its own repair strategies via probability feedback")
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

    log(f"\n[Step 4] Self-Generated Strategy Discovery...")
    log(f"  Deep layers: {DEEP_LAYERS[0]}-{DEEP_LAYERS[1]}")
    log(f"  Top-K attention tokens: {TOP_K_TOKENS}")
    log(f"  Max generation tokens: {MAX_NEW_TOKENS}")
    log(f"  Max retries per sample: {MAX_RETRIES}")
    log(f"  Method: LLM self-diagnosis → self-repair → log-prob verification")
    log(f"")

    memory = []
    repaired_results = []

    t_repair = time.time()
    for idx, sample in enumerate(test_samples):
        result, memory = self_bootstrap_generated(
            model, tokenizer, sample, device, idx, memory)
        repaired_results.append(result)

    repair_time = time.time() - t_repair
    post_metrics = compute_metrics(repaired_results, test_samples)

    log(f"\n{'=' * 64}")
    log(f"[Summary] P21 Self-Generated Strategy Discovery")
    log(f"  Baseline:    H={base_metrics['H']:.4f}, C={base_metrics['C']:.4f}")
    log(f"  After Self:  H={post_metrics['H']:.4f}, C={post_metrics['C']:.4f}")
    log(f"  ΔH: {post_metrics['H'] - base_metrics['H']:+.4f}")

    n_hall = sum(1 for r in repaired_results if r.get("hallucinates"))
    n_attempted = sum(1 for r in repaired_results if r.get("attempted"))
    n_fixable = sum(1 for r in repaired_results if r.get("fixable"))

    strategy_types = {}
    strategy_fixes = {}
    for r in repaired_results:
        s = r.get("strategy", "none")
        strategy_types[s] = strategy_types.get(s, 0) + 1
        if r.get("fixable"):
            strategy_fixes[s] = strategy_fixes.get(s, 0) + 1

    avg_imp = (sum(r.get("improvement", 0) for r in repaired_results if r.get("attempted"))
               / max(n_attempted, 1))

    log(f"\n  Agent Statistics:")
    log(f"    Hallucinated:  {n_hall}/{len(test_samples)}")
    log(f"    Attempted:     {n_attempted}")
    log(f"    Fixable:       {n_fixable}")
    log(f"    Avg improv:    {avg_imp:+.4f}")
    log(f"    Self-discovered strategies: {strategy_types}")
    log(f"    Strategy fixes: {strategy_fixes}")

    if memory:
        log(f"\n  Self-Discovered Repair Examples:")
        for m in memory[:5]:
            if m.get("fixable") or m.get("improvement", 0) > 0.01:
                log(f"    Sample {m['sample_idx']}: {m['strategy']} "
                    f"Δ={m['lp_diff_before']:+.4f}→{m['lp_diff_after']:+.4f} "
                    f"({m.get('question','')[:40]})")

    elapsed = time.time() - t0
    log(f"\n  Repair: {repair_time:.0f}s, Total: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    results = {
        "config": {
            "deep_layers": list(DEEP_LAYERS),
            "top_k_tokens": TOP_K_TOKENS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "max_retries": MAX_RETRIES,
            "method": "LLM self-diagnosis + self-generated repair + log-prob verification",
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
        "strategy_distribution": strategy_types,
        "strategy_fixes": strategy_fixes,
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
    log(f"\nP21 Complete. v22.0")


if __name__ == "__main__":
    main()