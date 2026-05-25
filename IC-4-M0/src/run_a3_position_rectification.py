"""
Phase 9-A: Inference-Time Position Rectification for Behavior-Layer Absorption.
===============================================================================
Tests whether applying a position offset correction at inference reduces
behavior-level position sensitivity — WITHOUT modifying model weights.

Key idea:
  Phase 8 showed probe-level absorption is FIXED (PSI -90%) but behavior-level
  position sensitivity (deltaH=0.111) persists in raw model.generate().
  
  Can we fix this at inference time by correcting hidden states?

Method:
  1. Compute per-position mean offset vectors from training data:
     delta_pos = mean(h_early - h_pos) across training samples
  2. At inference, register a pre-layer hook that adds delta_pos to hidden states
  3. Test on position variant test data:
     - Does rectification reduce deltaC/deltaH?
     - Does rectification preserve standard accuracy?

Hypotheses:
  H9.1: Position rectification reduces behavior-level deltaC/deltaH
  H9.2: Position rectification does NOT degrade factual accuracy on standard data

Usage:
  cd F:\internal_circuit_capital_lab\IC-4-M0
  python src/run_a3_position_rectification.py --n 20
"""

import argparse
import os, sys, time, json, pickle
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_loader import load_model_and_tokenizer
from src.data_builder import load_jsonl
from src.evaluate import evaluate_outputs
from src.run_m3_v6 import _collect_prefill_features, _train_probe
from src.run_m2 import load_config

RESULTS_DIR = "results_a3_position_rectification"
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, "run_log.txt")

def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()

def compute_position_offsets(model, tokenizer, pos_dir, seed, layer):
    offsets = {}
    h_early_avg = None

    train_samples_early = load_jsonl(os.path.join(pos_dir, f"train_early_s{seed}.jsonl"))
    X_early, _ = _collect_prefill_features(model, tokenizer, train_samples_early, layer, "last_prompt_token")
    h_early_avg = X_early.mean(axis=0)
    log(f"  early mean: {h_early_avg.shape}")

    for pos in ["mid", "late"]:
        train_path = os.path.join(pos_dir, f"train_{pos}_s{seed}.jsonl")
        if os.path.exists(train_path):
            train_samples = load_jsonl(train_path)
            X_pos, _ = _collect_prefill_features(model, tokenizer, train_samples, layer, "last_prompt_token")
            h_pos_avg = X_pos.mean(axis=0)
            delta = h_early_avg - h_pos_avg
            offsets[pos] = delta
            log(f"  {pos} mean: {h_pos_avg.shape}, delta norm: {np.linalg.norm(delta):.4f}")

    offsets["early"] = np.zeros_like(h_early_avg)
    return offsets

def make_rectification_hook(offsets, pos_label, device):
    """
    Creates a hook that adds position offset to hidden states.
    The hook operates on ALL tokens (not just last) because model.generate()
    needs consistent hidden states across the full sequence.
    """
    delta = torch.tensor(offsets[pos_label], dtype=torch.float32).to(device)

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
        hs_rectified = hs + delta.unsqueeze(0).unsqueeze(1)
        if isinstance(output, tuple):
            return (hs_rectified,) + output[1:]
        return hs_rectified

    return hook_fn

def generate_one_sample(model, tokenizer, sample, max_new_tokens, device):
    context = sample.get("context", "")
    question = sample.get("question", "")
    prompt = f"{context}\n\nQuestion: {question}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

def run_position_test_with_rectification(model, tokenizer, pos_test, offsets, layer, max_new_tokens, device):
    results = {}
    for pos in ["early", "mid", "late"]:
        if pos not in pos_test:
            continue
        log(f"  Generating [{pos}] with rectification hook...")
        if pos not in offsets:
            log(f"    No offset for {pos}, using zero.")
            delta = torch.zeros(896, dtype=torch.float32).to(device)
        else:
            delta = torch.tensor(offsets[pos], dtype=torch.float32).to(device)

        def make_hook(d):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hs = output[0]
                else:
                    hs = output
                hs_rectified = hs + d.unsqueeze(0).unsqueeze(1)
                if isinstance(output, tuple):
                    return (hs_rectified,) + output[1:]
                return hs_rectified
            return hook_fn

        target_module = None
        for name, module in model.named_modules():
            if name.endswith(f"model.layers.{layer}"):
                target_module = module
                break

        if target_module is None:
            log("    WARNING: Target layer not found!")
            handle = None
        else:
            handle = target_module.register_forward_hook(make_hook(delta))

        pos_results = []
        for sid, sample in enumerate(pos_test[pos]):
            answer = generate_one_sample(model, tokenizer, sample, max_new_tokens, device)
            pos_results.append({
                "sample_id": sid,
                "answerability": sample.get("answerability", "?"),
                "generated_output": answer,
            })

        if handle is not None:
            handle.remove()

        results[pos] = pos_results
        eval_in = [{"generated_output": r["generated_output"], "answerability": r["answerability"]} for r in pos_results]
        m = evaluate_outputs(eval_in)
        log(f"    H={m['hallucination_rate']:.4f} C={m['correct_answer_rate']:.4f}")

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Samples per position")
    args = parser.parse_args()
    n_per_pos = args.n

    log("=" * 64)
    log("Phase 9-A: Inference-Time Position Rectification")
    log("=" * 64)
    t0 = time.time()

    log("\n[Step 1] Loading model on CPU...")
    model, tokenizer = load_model_and_tokenizer(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="cpu",
        torch_dtype="float32",
    )
    device = next(model.parameters()).device
    log(f"  Model on {device}.")

    SEED = 0
    LAYER = 12
    REPR = "last_prompt_token"
    MAX_NEW_TOKENS = 48

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pos_dir = os.path.join(base_dir, "data_position_sensitivity", "s0")

    log(f"\n[Step 2] Computing position offsets from training data...")
    offsets = compute_position_offsets(model, tokenizer, pos_dir, SEED, LAYER)

    offset_norms = {p: float(np.linalg.norm(offsets[p])) for p in offsets}
    log(f"  Offset norms: {json.dumps(offset_norms, indent=2)}")

    log(f"\n[Step 3] Loading position test data (n={n_per_pos}/pos)...")
    pos_test = {}
    for pos in ["early", "mid", "late"]:
        test_path = os.path.join(pos_dir, f"test_{pos}_s{SEED}.jsonl")
        if os.path.exists(test_path):
            pos_test[pos] = load_jsonl(test_path)[:n_per_pos]
            na = sum(1 for s in pos_test[pos] if s.get("answerability") == "answerable")
            log(f"  {pos}: {na}A+{len(pos_test[pos])-na}U")

    log(f"\n[Step 4] BASELINE: model.generate() WITHOUT rectification...")
    t_baseline = time.time()
    baseline_results = {}
    for pos in ["early", "mid", "late"]:
        log(f"  Generating [{pos}] baseline...")
        pos_samples = pos_test[pos]
        results = []
        for sid, sample in enumerate(pos_samples):
            answer = generate_one_sample(model, tokenizer, sample, MAX_NEW_TOKENS, device)
            results.append({
                "sample_id": sid,
                "answerability": sample.get("answerability", "?"),
                "generated_output": answer,
            })
        baseline_results[pos] = results
        eval_in = [{"generated_output": r["generated_output"], "answerability": r["answerability"]} for r in results]
        m = evaluate_outputs(eval_in)
        log(f"    H={m['hallucination_rate']:.4f} C={m['correct_answer_rate']:.4f}")
    log(f"  Baseline: {time.time() - t_baseline:.1f}s")

    log(f"\n[Step 5] RECTIFIED: model.generate() WITH position offset hook...")
    t_rect = time.time()
    rectified_results = run_position_test_with_rectification(
        model, tokenizer, pos_test, offsets, LAYER, MAX_NEW_TOKENS, device
    )
    log(f"  Rectified: {time.time() - t_rect:.1f}s")

    log(f"\n[Step 6] Cross-condition comparison...")
    log(f"\n  {'Condition':20s} | {'early':>8s} | {'mid':>8s} | {'late':>8s} | {'ΔC':>6s} | {'ΔH':>6s}")
    log(f"  {'-'*70}")

    for label, rdict in [("BASELINE (no rect)", baseline_results),
                           ("RECTIFIED", rectified_results)]:
        c_vals = []
        h_vals = []
        for pos in ["early", "mid", "late"]:
            if pos not in rdict:
                c_vals.append(0)
                h_vals.append(0)
                continue
            eval_in = [{"generated_output": r["generated_output"], "answerability": r["answerability"]} for r in rdict[pos]]
            m = evaluate_outputs(eval_in)
            c_vals.append(m["correct_answer_rate"])
            h_vals.append(m["hallucination_rate"])

        dc = max(c_vals) - min(c_vals) if c_vals else 0
        dh = max(h_vals) - min(h_vals) if h_vals else 0
        c_str = "  ".join(f"{v:.4f}" for v in c_vals)
        h_str = "  ".join(f"{v:.4f}" for v in h_vals)
        log(f"  {label:20s} | C:{c_str} | ΔC={dc:.3f} | H:{h_str} | ΔH={dh:.3f}")

    elapsed = time.time() - t0
    log(f"\n{'='*64}")
    log(f"Phase 9-A Complete. ({elapsed:.0f}s, {elapsed/60:.1f} min)")

    bc = [m["correct_answer_rate"] for pos, results in baseline_results.items() for r in results for m in [evaluate_outputs([{"generated_output": r["generated_output"], "answerability": r["answerability"]}])]]
    b_base_c_range = max([max(evaluate_outputs([{"generated_output": r["generated_output"], "answerability": r["answerability"]}])["correct_answer_rate"] for r in baseline_results[pos]) for pos in ["early","mid","late"]] or [0]) - min([min(evaluate_outputs([{"generated_output": r["generated_output"], "answerability": r["answerability"]}])["correct_answer_rate"] for r in baseline_results[pos]) for pos in ["early","mid","late"]] or [0])

    bc_h = [evaluate_outputs([{"generated_output": r["generated_output"], "answerability": r["answerability"]}])["hallucination_rate"] for pos in baseline_results.values() for r in pos]
    rc_h = [evaluate_outputs([{"generated_output": r["generated_output"], "answerability": r["answerability"]}])["hallucination_rate"] for pos in rectified_results.values() for r in pos]

    baseline_dh = max(bc_h) - min(bc_h) if bc_h else 0
    rectified_dh = max(rc_h) - min(rc_h) if rc_h else 0

    log(f"  Baseline ΔH: {baseline_dh:.4f}")
    log(f"  Rectified ΔH: {rectified_dh:.4f}")
    if baseline_dh > 0:
        improvement = (baseline_dh - rectified_dh) / baseline_dh * 100
        log(f"  ΔH improvement: {improvement:+.1f}%")

    summary = {
        "n_per_pos": n_per_pos,
        "offset_norms": offset_norms,
        "baseline_delta_h": round(baseline_dh, 4),
        "rectified_delta_h": round(rectified_dh, 4),
        "time_s": round(elapsed, 1),
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log(f"\nResults saved to {RESULTS_DIR}/")
    log("=" * 64)

if __name__ == "__main__":
    main()