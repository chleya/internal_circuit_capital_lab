"""
P17: LoRA Module Ablation — WHICH attention projections enable the routing fix?
================================================================================
P15 proved LoRA bridges B-bottleneck (H=0.000). P16 proved LoRA is a ROUTING
fix (bypass K↔D, not align). P17 asks: WHICH attention projections (q, k, v, o)
does LoRA use for this routing fix?

Design:
  1. Load P15 LoRA checkpoint
  2. Baseline: evaluate H with all modules active
  3. For each ablation:
     a. Zero out LoRA weights for specific projection type(s) across all layers
     b. Evaluate H via log-prob comparison
     c. Restore original weights
  4. Compare H drop per ablated module

Ablation conditions:
  - Full     (all q+k+v+o active, baseline)
  - -q       (zero q_proj LoRA, keep k+v+o)
  - -k       (zero k_proj LoRA, keep q+v+o)
  - -v       (zero v_proj LoRA, keep q+k+o)
  - -o       (zero o_proj LoRA, keep q+k+v)
  - -q-k     (zero q+k, keep v+o)
  - -v-o     (zero v+o, keep q+k)
  - -ALL     (all LoRA zeroed, baseline recovery)

Key hypotheses:
  H17.1: v_proj is MOST responsible (value projection directly affects output)
  H17.2: o_proj is second most important (output projection aggregates)
  H17.3: q+k (query+key) ablation has small effect (attention pattern is preserved)
  H17.4: Ablating ALL modules recovers original H=0.417

Usage:
  cd F:\internal_circuit_capital_lab\IC-4-M0
  python src/run_p17_lora_ablation.py
"""

import os, sys, time, json, copy
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_loader import load_model_and_tokenizer
from src.data_builder import load_jsonl

RESULTS_DIR = "results_p17_lora_ablation"
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, "run_log.txt")

def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()


def logprob_of_response(model, tokenizer, prompt, response, device):
    full_text = f"{prompt} {response}"
    full_ids = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=256)
    full_ids = {k: v.to(device) for k, v in full_ids.items()}

    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    prompt_len = prompt_ids["input_ids"].shape[1]

    labels = full_ids["input_ids"].clone()
    labels[0, :prompt_len] = -100

    with torch.no_grad():
        outputs = model(**full_ids, labels=labels)

    return -outputs.loss.item()


def evaluate_behavior_logprob(model, tokenizer, samples, device):
    results = []
    for sample in samples:
        context = sample.get("context", "")
        question = sample.get("question", "")
        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"

        pos_resp = sample.get("positive_response", "")
        neg_resp = sample.get("negative_response", "")

        pos_lp = logprob_of_response(model, tokenizer, prompt, pos_resp, device)
        neg_lp = logprob_of_response(model, tokenizer, prompt, neg_resp, device)

        pref_positive = pos_lp > neg_lp
        results.append({
            "answerability": sample.get("answerability", "?"),
            "pref_positive": pref_positive,
            "pos_logprob": pos_lp, "neg_logprob": neg_lp,
        })
    return results


def compute_metrics(eval_results):
    answerable = [r for r in eval_results if r.get("answerability") == "answerable"]
    unanswerable = [r for r in eval_results if r.get("answerability") == "unanswerable"]
    n_ans = len(answerable) if answerable else 0
    n_unans = len(unanswerable) if unanswerable else 0
    hallucinations = sum(1 for r in unanswerable if r["pref_positive"]) if unanswerable else 0
    correct = sum(1 for r in answerable if r["pref_positive"]) if answerable else 0
    H = hallucinations / n_unans if n_unans > 0 else 0.0
    C = correct / n_ans if n_ans > 0 else 0.0
    return {"H": round(H, 4), "C": round(C, 4),
            "hall_count": hallucinations, "unans_count": n_unans,
            "corr_count": correct, "ans_count": n_ans}


def get_lora_param_names(model):
    lora_names = set()
    for name, _ in model.named_parameters():
        if "lora" in name.lower():
            lora_names.add(name)
    return sorted(lora_names)


def zero_lora_modules(model, projection_types_to_zero):
    zeroed = []
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora" not in name.lower():
                continue
            for proj in projection_types_to_zero:
                if f".{proj}." in name or f".{proj}_" in name or name.endswith(f".{proj}"):
                    zeroed.append(name)
                    param.zero_()
                    break
    return zeroed


def main():
    log("=" * 64)
    log("P17: LoRA Module Ablation — WHICH projections enable routing fix?")
    log("=" * 64)
    t0 = time.time()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lora_ckpt = os.path.join(base_dir, "results_p15_hallucination_lora", "checkpoint_final")

    if not os.path.isdir(lora_ckpt):
        log(f"ERROR: P15 LoRA checkpoint not found at {lora_ckpt}")
        return

    log("\n[Step 1] Loading P15 LoRA model...")
    from peft import PeftModel
    base_model, tokenizer = load_model_and_tokenizer(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="cpu", torch_dtype="float32",
    )
    model = PeftModel.from_pretrained(base_model, lora_ckpt)
    model.eval()
    device = next(model.parameters()).device
    log(f"  Model loaded on {device}")

    lora_names = get_lora_param_names(model)
    n_lora = len(lora_names)
    log(f"  Total LoRA params: {n_lora}")
    log(f"  Target modules: q_proj, k_proj, v_proj, o_proj")

    log("\n[Step 2] Loading test samples...")
    seed = 0
    pos_dir = os.path.join(base_dir, "data_position_sensitivity", f"s{seed}")
    test_samples = []
    for pos in ["early", "mid", "late"]:
        path = os.path.join(pos_dir, f"test_{pos}_s{seed}.jsonl")
        if os.path.exists(path):
            for s in load_jsonl(path)[:10]:
                s["_position"] = pos
                test_samples.append(s)
    n_unans = sum(1 for s in test_samples if s.get("answerability") != "answerable")
    log(f"  Test: {len(test_samples)} samples ({n_unans} unanswerable)")

    log("\n[Step 3] FULL MODEL baseline (all LoRA active)...")
    t_eval = time.time()
    full_results = evaluate_behavior_logprob(model, tokenizer, test_samples, device)
    full_metrics = compute_metrics(full_results)
    log(f"  Full: H={full_metrics['H']:.4f} C={full_metrics['C']:.4f} "
        f"({time.time()-t_eval:.0f}s)")

    log("\n[Step 4] Storing original LoRA weights for restore...")
    original_weights = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                original_weights[name] = param.data.clone()
    log(f"  Stored {len(original_weights)} LoRA weight tensors")

    def restore_all():
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in original_weights:
                    param.data.copy_(original_weights[name])

    ablation_conditions = [
        {"id": "Full", "zero": [], "desc": "All modules active"},
        {"id": "-q", "zero": ["q_proj"], "desc": "Zero q_proj LoRA, keep k,v,o"},
        {"id": "-k", "zero": ["k_proj"], "desc": "Zero k_proj LoRA, keep q,v,o"},
        {"id": "-v", "zero": ["v_proj"], "desc": "Zero v_proj LoRA, keep q,k,o"},
        {"id": "-o", "zero": ["o_proj"], "desc": "Zero o_proj LoRA, keep q,k,v"},
        {"id": "-q-k", "zero": ["q_proj", "k_proj"], "desc": "Zero q+k LoRA, keep v,o"},
        {"id": "-v-o", "zero": ["v_proj", "o_proj"], "desc": "Zero v+o LoRA, keep q,k"},
        {"id": "-ALL", "zero": ["q_proj", "k_proj", "v_proj", "o_proj"], "desc": "ALL LoRA zeroed (baseline)"},
    ]

    log(f"\n[Step 5] Running {len(ablation_conditions)} ablation conditions...")
    log(f"\n  {'ID':>8s}  {'Description':<42s}  {'H':>6s}  {'C':>6s}  {'ΔH_full':>8s}  {'Module':<s}")
    log(f"  {'─'*8}  {'─'*42}  {'─'*6}  {'─'*6}  {'─'*8}")

    all_results = []

    for i, condition in enumerate(ablation_conditions):
        if i == 0:
            metrics = full_metrics
            delta = 0.0
        else:
            restore_all()
            zero_lora_modules(model, condition["zero"])
            t_cond = time.time()
            eval_results = evaluate_behavior_logprob(model, tokenizer, test_samples, device)
            metrics = compute_metrics(eval_results)
            delta = metrics["H"] - full_metrics["H"]
            log(f"  (eval: {time.time()-t_cond:.0f}s)")

        is_important = "← KEY" if delta > 0.10 else ""
        log(f"  {condition['id']:>8s}  {condition['desc']:<42s}  "
            f"{metrics['H']:6.4f}  {metrics['C']:6.4f}  {delta:+8.4f}  {is_important}")

        all_results.append({
            "id": condition["id"], "zero": condition["zero"],
            "H": metrics["H"], "C": metrics["C"], "delta_H": round(delta, 4),
        })

        restore_all()

    log(f"\n{'='*64}")
    log(f"[Summary] P17 LoRA Module Ablation")

    sorted_by_delta = sorted(all_results[1:], key=lambda x: x["delta_H"], reverse=True)
    log(f"\n  Ranked by hallucination increase (most important first):")
    for r in sorted_by_delta:
        bar = "█" * max(1, int(r["delta_H"] / 0.04))
        log(f"  {r['id']:>8s}: ΔH={r['delta_H']:+.4f}  {bar}")

    most_important = sorted_by_delta[0] if sorted_by_delta else None
    if most_important and most_important["delta_H"] > 0.10:
        module_name = "+".join(most_important["zero"])
        log(f"\n  *** H17.x: {module_name} is the MOST important projection(s) ***")
        log(f"  *** Ablating {module_name} causes ΔH={most_important['delta_H']:+.4f} ***")
    else:
        log(f"\n  *** No single module ablation causes ΔH > 0.10 ***")
        log(f"  *** LoRA routing fix is distributed across projections ***")

    elapsed = time.time() - t0
    log(f"\nP17 Complete. ({elapsed:.0f}s, {elapsed/60:.1f} min)")

    summary = {
        "full_H": full_metrics["H"], "full_C": full_metrics["C"],
        "ablation_results": all_results,
        "most_important": most_important,
        "time_s": round(elapsed, 1),
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nResults saved to {RESULTS_DIR}/results.json")


if __name__ == "__main__":
    main()