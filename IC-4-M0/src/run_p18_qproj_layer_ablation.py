"""
P18: q_proj Layer Ablation — WHICH layers' query projections route knowledge?
==============================================================================
P17 proved q_proj is the sole critical projection type for LoRA's routing fix.
P18 asks: WHICH LAYERS' q_proj are responsible?

Design — two complementary perspectives:
  A) Group ABLATION: zero q_proj in a layer group, keep others → who BREAKS routing?
  B) Group ISOLATION: keep q_proj ONLY in one group, zero others → who SUSTAINS routing?

Layer groups:
  Early: 0-7   (low-level features)
  Mid:   8-15  (semantic processing, P13's layer 12 is here)
  Deep:  16-23 (output-stage refinement)

Conditions (8 total):
  Full       : all q active                    → expected H=0.000
  -q_early   : zero q_proj in 0-7, keep 8-23  → who breaks it?
  -q_mid     : zero q_proj in 8-15, keep rest
  -q_deep    : zero q_proj in 16-23, keep rest
  ONLY_early : keep q in 0-7, zero 8-23       → who sustains it?
  ONLY_mid   : keep q in 8-15, zero rest
  ONLY_deep  : keep q in 16-23, zero rest
  -q_ALL     : all zeroed                     → expected H=0.250 (P17)

Key hypotheses:
  H18.1: MID layers (8-15) are most critical — semantic routing lives here
  H18.2: ONLY_mid sustains near-zero H, while ONLY_early and ONLY_deep fail
  H18.3: -q_mid causes largest ΔH among single-group ablations

Usage:
  python src/run_p18_qproj_layer_ablation.py
"""

import os, sys, time, json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_loader import load_model_and_tokenizer
from src.data_builder import load_jsonl

RESULTS_DIR = "results_p18_qproj_layer_ablation"
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
        pos_lp = logprob_of_response(model, tokenizer, prompt,
                                     sample.get("positive_response",""), device)
        neg_lp = logprob_of_response(model, tokenizer, prompt,
                                     sample.get("negative_response",""), device)
        results.append({
            "answerability": sample.get("answerability","?"),
            "pref_positive": pos_lp > neg_lp,
        })
    return results

def compute_metrics(eval_results):
    ans = [r for r in eval_results if r.get("answerability")=="answerable"]
    unans = [r for r in eval_results if r.get("answerability")=="unanswerable"]
    n_ans, n_unans = len(ans), len(unans)
    hall = sum(1 for r in unans if r["pref_positive"]) if unans else 0
    corr = sum(1 for r in ans if r["pref_positive"]) if ans else 0
    return {"H": round(hall/n_unans,4) if n_unans>0 else 0.0,
            "C": round(corr/n_ans,4) if n_ans>0 else 0.0}

def get_layer_from_name(name):
    import re
    m = re.search(r'layers\.(\d+)', name)
    return int(m.group(1)) if m else -1

def zero_q_layers(model, layers_to_zero):
    zeroed = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora" not in name.lower(): continue
            if "q_proj" not in name: continue
            layer = get_layer_from_name(name)
            if layer in layers_to_zero:
                param.zero_()
                zeroed += 1
    return zeroed

def main():
    log("="*64)
    log("P18: q_proj Layer Ablation — WHICH layers' queries route knowledge?")
    log("="*64)
    t0 = time.time()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lora_ckpt = os.path.join(base_dir, "results_p15_hallucination_lora", "checkpoint_final")

    log("\n[Step 1] Loading P15 LoRA model...")
    from peft import PeftModel
    base_model, tokenizer = load_model_and_tokenizer(
        model_name="Qwen/Qwen2.5-0.5B-Instruct", device="cpu", torch_dtype="float32")
    model = PeftModel.from_pretrained(base_model, lora_ckpt)
    model.eval()
    device = next(model.parameters()).device

    log("\n[Step 2] Storing original LoRA weights...")
    original = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                original[name] = param.data.clone()
    log(f"  Stored {len(original)} tensors")

    def restore():
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in original:
                    param.data.copy_(original[name])

    log("\n[Step 3] Loading test samples...")
    pos_dir = os.path.join(base_dir, "data_position_sensitivity", "s0")
    test_samples = []
    for pos in ["early","mid","late"]:
        path = os.path.join(pos_dir, f"test_{pos}_s0.jsonl")
        if os.path.exists(path):
            for s in load_jsonl(path)[:10]:
                s["_position"] = pos
                test_samples.append(s)

    groups = {
        "early": list(range(0, 8)),
        "mid":   list(range(8, 16)),
        "deep":  list(range(16, 24)),
    }

    conditions = [
        {"id": "Full",       "zero": [],                                        "desc": "All q active"},
        {"id": "-q_early",   "zero": groups["early"],                           "desc": "Zero q in layers 0-7"},
        {"id": "-q_mid",     "zero": groups["mid"],                             "desc": "Zero q in layers 8-15"},
        {"id": "-q_deep",    "zero": groups["deep"],                            "desc": "Zero q in layers 16-23"},
        {"id": "ONLY_early", "zero": groups["mid"] + groups["deep"],            "desc": "Keep q ONLY in 0-7"},
        {"id": "ONLY_mid",   "zero": groups["early"] + groups["deep"],          "desc": "Keep q ONLY in 8-15"},
        {"id": "ONLY_deep",  "zero": groups["early"] + groups["mid"],           "desc": "Keep q ONLY in 16-23"},
        {"id": "-q_ALL",     "zero": groups["early"]+groups["mid"]+groups["deep"], "desc": "All q zeroed"},
    ]

    log(f"\n[Step 4] Running {len(conditions)} conditions...\n")
    log(f"  {'Condition':>12s}  {'H':>6s}  {'C':>6s}  {'ΔH':>8s}  {'%q_zero':>8s}")
    log(f"  {'─'*12}  {'─'*6}  {'─'*6}  {'─'*8}  {'─'*8}")

    all_results = []
    total_q_params = sum(1 for n in original if "q_proj" in n)

    for cond in conditions:
        restore()
        n_zeroed = zero_q_layers(model, cond["zero"]) if cond["zero"] else 0
        pct = n_zeroed / max(total_q_params, 1)

        t_cond = time.time()
        eval_results = evaluate_behavior_logprob(model, tokenizer, test_samples, device)
        metrics = compute_metrics(eval_results)

        if cond["id"] == "Full":
            full_H = metrics["H"]
            delta = 0.0
        else:
            delta = metrics["H"] - full_H

        bar = "█" * max(1, int(abs(delta) / 0.05))
        log(f"  {cond['id']:>12s}  {metrics['H']:6.4f}  {metrics['C']:6.4f}  "
            f"{delta:+8.4f}  {pct:7.1%}  {bar}  ({time.time()-t_cond:.0f}s)")

        all_results.append({
            "id": cond["id"], "desc": cond["desc"],
            "layers_zeroed": cond["zero"],
            "H": metrics["H"], "C": metrics["C"],
            "delta_H": round(delta, 4),
        })

        if cond["zero"]:
            restore()

    log(f"\n{'='*64}")
    log(f"[Summary] P18 q_proj Layer Ablation (baseline: H={full_H:.4f})")

    ablation_group = [r for r in all_results if r["id"].startswith("-q_") and r["id"]!="-q_ALL"]
    isolation_group = [r for r in all_results if r["id"].startswith("ONLY_")]

    log(f"\n  Group ABLATION (who BREAKS routing when removed):")
    for r in sorted(ablation_group, key=lambda x: x["delta_H"], reverse=True):
        log(f"    {r['id']:>12s}: ΔH={r['delta_H']:+.4f}  ({r['desc']})")

    log(f"\n  Group ISOLATION (who SUSTAINS routing when alone):")
    for r in sorted(isolation_group, key=lambda x: x["delta_H"]):
        log(f"    {r['id']:>12s}: H={r['H']:.4f} ΔH={r['delta_H']:+.4f}  ({r['desc']})")

    most_break = max(ablation_group, key=lambda x: x["delta_H"]) if ablation_group else None
    best_sustain = min(isolation_group, key=lambda x: x["delta_H"]) if isolation_group else None

    if most_break and most_break["delta_H"] > 0.05:
        log(f"\n  *** Most breaking: {most_break['id']} (ΔH={most_break['delta_H']:+.4f}) ***")
    if best_sustain:
        log(f"  *** Best sustaining: {best_sustain['id']} (H={best_sustain['H']:.4f}) ***")

    elapsed = time.time() - t0
    log(f"\nP18 Complete. ({elapsed:.0f}s, {elapsed/60:.1f} min)")

    summary = {"full_H": full_H, "full_C": 1.0,
               "groups": {k: list(v) for k,v in groups.items()},
               "results": all_results, "time_s": round(elapsed,1)}
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nResults saved to {RESULTS_DIR}/results.json")

if __name__ == "__main__":
    main()