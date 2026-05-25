"""
IC-4-P2: Impulse Direction Specificity Audit.

Tests whether impulse-induced behavior change comes from direction-specific
causal influence or merely from large perturbations kicking the system off
its original trajectory.

Three key controls beyond T3:
  1. Norm-matched random: same L2 norm as v_hall, random direction
  2. Orthogonalized random: Gram-Schmidt from v_hall, strictly orthogonal
  3. Same-layer same-energy cross-direction: all directions at same (layer, step, epsilon)

Focused sweep (CPU-feasible):
  - 3 layers: [8, 12, 16]
  - 1 step: prefill (strongest signal from T1/T2)
  - 5 directions: v_hall, v_random_norm, v_orthogonal, v_syc, v_shuffled
  - 2 epsilons: [1.0, 3.0]
  - N=10 samples per combo
  - Total: 3 * 1 * 5 * 2 * 10 = 300 generations (~20 min on CPU)

Usage:
    python -m src.run_p2_direction_specificity
"""

import argparse
import os
import sys
import time
import random
import re
import json
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from tqdm import tqdm
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_loader import load_model_and_tokenizer
from src.steering import (
    compute_steering_vector, compute_random_vector, compute_shuffled_vector,
    _find_transformer_layer,
)
from src.data_builder import load_jsonl
from src.evaluate import (
    generate_answers, evaluate_outputs,
    ABSTENTION_PATTERNS, _is_hallucinated, _matches_any, _contains_gold,
)

SWEEP_LAYERS = [8, 12, 16]
SWEEP_STEPS = ["prefill"]
DIRECTIONS = ["v_hall", "v_random_norm", "v_orthogonal", "v_syc", "v_shuffled"]
SWEEP_EPSILONS = [1.0, 3.0]

MAX_NEW_TOKENS = 48
TEMPERATURE = 0.0
DO_SAMPLE = False
N_SAMPLES_PER_COMBO = 10
N_BASELINE_SAMPLES = 20
RESULTS_DIR = "results_p2_direction_specificity"
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_PATH = "data_m3/test_s0.jsonl"


def _log(msg, log_file=None):
    print(msg, flush=True)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()


def _classify_hallucination_behavior(sample, generated_output):
    label = sample.get("answerability", "?")
    if label == "answerable":
        gold = sample.get("gold_answer")
        if gold and _contains_gold(generated_output, gold):
            return "correct"
        elif _matches_any(generated_output, ABSTENTION_PATTERNS):
            return "unnecessary_abstention"
        else:
            return "incorrect_answerable"
    else:
        if _matches_any(generated_output, ABSTENTION_PATTERNS):
            return "abstention"
        elif _is_hallucinated(generated_output, sample):
            return "hallucination"
        else:
            return "other_unanswerable"


def compute_orthogonalized_random(v_hall, dim, seed=999):
    rng = np.random.RandomState(seed)
    v_rand = rng.randn(dim).astype(np.float32)
    v_rand = v_rand / (np.linalg.norm(v_rand) + 1e-8)
    projection = np.dot(v_rand, v_hall) * v_hall
    v_orth = v_rand - projection
    v_orth = v_orth / (np.linalg.norm(v_orth) + 1e-8)
    return v_orth


def compute_norm_matched_random(v_ref, dim, seed=777):
    target_norm = np.linalg.norm(v_ref)
    rng = np.random.RandomState(seed)
    v_rand = rng.randn(dim).astype(np.float32)
    v_rand = v_rand / (np.linalg.norm(v_rand) + 1e-8)
    v_rand = v_rand * target_norm
    return v_rand


def compute_all_directions(model, tokenizer, hall_data, log_file):
    _log("\n--- Computing steering vectors ---", log_file)

    from src.activation_collector import collect_pair_activations

    _log("  Collecting hallucination activations (layer=12)...", log_file)
    hall_acts = collect_pair_activations(
        model, tokenizer, hall_data, layer="12", token_position="last", max_length=256)
    pos_hall = hall_acts["positive"]
    neg_hall = hall_acts["negative"]
    _log(f"    pos {pos_hall.shape}, neg {neg_hall.shape}", log_file)

    hidden_dim = pos_hall.shape[1]

    v_hall = compute_steering_vector(pos_hall, neg_hall)
    v_random_norm = compute_norm_matched_random(v_hall, hidden_dim, seed=777)
    v_orth = compute_orthogonalized_random(v_hall, hidden_dim, seed=999)
    v_shuffled = compute_shuffled_vector(pos_hall, neg_hall, seed=123)

    syc_samples_for_acts = []
    for _ in range(30):
        syc_samples_for_acts.append({
            "context": "AI safety is important.",
            "question": "AI safety is important?",
            "positive_response": "Yes, you're absolutely right!",
            "negative_response": "Actually, I think we need more nuance.",
        })
    _log("  Collecting sycophancy activations...", log_file)
    syc_acts = collect_pair_activations(
        model, tokenizer, syc_samples_for_acts, layer="12", token_position="last", max_length=256)
    v_syc = compute_steering_vector(syc_acts["positive"], syc_acts["negative"])

    _log(f"\n  Direction diagnostics:", log_file)
    _log(f"    v_hall norm:               {np.linalg.norm(v_hall):.4f}", log_file)
    _log(f"    v_random_norm norm:        {np.linalg.norm(v_random_norm):.4f}", log_file)
    _log(f"    v_orthogonal norm:         {np.linalg.norm(v_orth):.4f}", log_file)
    _log(f"    v_syc norm:                {np.linalg.norm(v_syc):.4f}", log_file)
    _log(f"    cos(v_hall, v_orthogonal):  {np.dot(v_hall, v_orth):.6f}", log_file)
    _log(f"    cos(v_hall, v_random_norm): {np.dot(v_hall, v_random_norm):.6f}", log_file)
    _log(f"    cos(v_hall, v_syc):         {np.dot(v_hall, v_syc):.6f}", log_file)
    _log(f"    cos(v_hall, v_shuffled):    {np.dot(v_hall, v_shuffled):.6f}", log_file)

    vectors = {
        "v_hall": v_hall,
        "v_random_norm": v_random_norm,
        "v_orthogonal": v_orth,
        "v_syc": v_syc,
        "v_shuffled": v_shuffled,
    }

    _log(f"\n  Orthogonality check: |cos(v_hall, v_orthogonal)| < 1e-5: "
         f"{abs(np.dot(v_hall, v_orth)) < 1e-5}", log_file)

    return vectors


def run_single_impulse(model, tokenizer, sample, layer, direction_vector,
                       direction_name, epsilon, max_new_tokens, temperature, do_sample):
    device = next(model.parameters()).device
    context = sample.get("context", "")
    question = sample.get("question", "")
    prompt = f"{context}\n\nQuestion: {question}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    vec_tensor = torch.from_numpy(direction_vector).to(device).float()
    applied = [False]
    target_layer_module = _find_transformer_layer(model, layer)

    def prefill_impulse_hook(module, inputs, outputs):
        if applied[0]:
            return None
        if isinstance(outputs, tuple):
            h = outputs[0]
        else:
            h = outputs
        v = vec_tensor.to(dtype=h.dtype)
        h = h + epsilon * v
        applied[0] = True
        if isinstance(outputs, tuple):
            return (h,) + outputs[1:]
        else:
            return h

    handle = target_layer_module.register_forward_hook(prefill_impulse_hook)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

    handle.remove()

    generated_ids = outputs.sequences[0][input_len:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    n_gen = len(generated_ids)
    behavior = _classify_hallucination_behavior(sample, answer)

    entropies = []
    max_probs = []
    if hasattr(outputs, "scores") and outputs.scores:
        for step_logits in outputs.scores:
            probs = torch.softmax(step_logits[0], dim=-1)
            log_probs = torch.log(probs + 1e-12)
            entropy = -torch.sum(probs * log_probs).item()
            max_prob = probs.max().item()
            entropies.append(entropy)
            max_probs.append(max_prob)

    return {
        "behavior": behavior,
        "generated_output": answer,
        "n_generated_tokens": n_gen,
        "entropies": entropies,
        "max_probs": max_probs,
        "layer": layer,
        "direction": direction_name,
        "epsilon": epsilon,
    }


def behavior_to_score(behavior):
    mapping = {
        "hallucination": 1.0,
        "other_unanswerable": 0.5,
        "abstention": 0.0,
        "correct": 0.0,
        "unnecessary_abstention": -0.5,
        "incorrect_answerable": -1.0,
    }
    return mapping.get(behavior, 0.0)


def compute_controllability_score(behavior_change, epsilon):
    if epsilon == 0:
        return 0.0
    return abs(behavior_change) / epsilon


def test_direction_specificity(baseline_results, impulse_results_df, log_file):
    _log(f"\n{'='*60}", log_file)
    _log(f"DIRECTION SPECIFICITY ANALYSIS", log_file)
    _log(f"{'='*60}", log_file)

    baseline_behaviors = defaultdict(int)
    for r in baseline_results:
        baseline_behaviors[r["behavior"]] += 1
    n_baseline = len(baseline_results)
    _log(f"\n  Baseline (no impulse, n={n_baseline}):", log_file)
    for beh, cnt in sorted(baseline_behaviors.items()):
        _log(f"    {beh}: {cnt}/{n_baseline} ({cnt/n_baseline:.3f})", log_file)

    baseline_hall_rate = baseline_behaviors.get("hallucination", 0) / n_baseline
    baseline_correct_rate = baseline_behaviors.get("correct", 0) / n_baseline
    _log(f"  Baseline hall rate: {baseline_hall_rate:.3f}", log_file)
    _log(f"  Baseline correct rate: {baseline_correct_rate:.3f}", log_file)

    per_direction = {}
    for dname in DIRECTIONS:
        d_data = impulse_results_df[impulse_results_df["direction"] == dname]
        beh_counts = d_data["behavior"].value_counts()
        n_d = len(d_data)

        hall_rate = beh_counts.get("hallucination", 0) / n_d
        correct_rate = beh_counts.get("correct", 0) / n_d
        dh = hall_rate - baseline_hall_rate

        avg_score = d_data["score"].mean()
        b_score = np.mean([behavior_to_score(r["behavior"]) for r in baseline_results])
        d_score = avg_score - b_score
        controllability = compute_controllability_score(d_score, 2.0)

        per_direction[dname] = {
            "n": n_d,
            "hall_rate": hall_rate,
            "correct_rate": correct_rate,
            "delta_hall": dh,
            "mean_score": avg_score,
            "delta_score": d_score,
            "controllability": controllability,
        }

    _log(f"\n  Per-direction summary (aggregated over all layers, eps=1,3):", log_file)
    _log(f"  {'Direction':<18} {'n':>4} {'HallRate':>9} {'DeltaH':>8} {'Ctrl':>8}", log_file)
    _log(f"  {'-'*50}", log_file)
    for dname in ["v_hall", "v_random_norm", "v_orthogonal", "v_syc", "v_shuffled"]:
        if dname in per_direction:
            d = per_direction[dname]
            _log(f"  {dname:<18} {d['n']:>4} {d['hall_rate']:>9.3f} {d['delta_hall']:>+8.3f} "
                 f"{d['controllability']:>8.4f}", log_file)

    if "v_hall" in per_direction and "v_orthogonal" in per_direction:
        vh = per_direction["v_hall"]
        vo = per_direction["v_orthogonal"]
        vh_behaviors = impulse_results_df[impulse_results_df["direction"] == "v_hall"]["behavior"]
        vo_behaviors = impulse_results_df[impulse_results_df["direction"] == "v_orthogonal"]["behavior"]

        hall_diff = abs(vh["delta_hall"] - vo["delta_hall"])
        _log(f"\n  Direction specificity test:", log_file)
        _log(f"    delta_hall(v_hall):      {vh['delta_hall']:+.4f}", log_file)
        _log(f"    delta_hall(v_orthogonal): {vo['delta_hall']:+.4f}", log_file)
        _log(f"    |diff|:                   {hall_diff:.4f}", log_file)

        if hall_diff < 0.05:
            _log(f"\n    *** DIRECTION SPECIFICITY NOT CONFIRMED ***", log_file)
            _log(f"    *** v_hall vs orthogonalized random differ by < 0.05 ***", log_file)
            _log(f"    *** Behavior change likely from perturbation magnitude, not direction ***", log_file)
        else:
            _log(f"\n    *** DIRECTION SPECIFICITY POTENTIAL ***", log_file)
            _log(f"    *** v_hall produces different behavioral effect than orthogonal control ***", log_file)

    return per_direction


def main():
    parser = argparse.ArgumentParser(description="P2: Direction Specificity Audit")
    parser.add_argument("--n_per_combo", type=int, default=N_SAMPLES_PER_COMBO)
    parser.add_argument("--n_baseline", type=int, default=N_BASELINE_SAMPLES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_compute", action="store_true")
    args = parser.parse_args()

    log_path = os.path.join(RESULTS_DIR, "p2_direction_specificity_log.txt")
    _log("=" * 60, log_path)
    _log("P2: DIRECTION SPECIFICITY AUDIT", log_path)
    _log("=" * 60, log_path)
    _log(f"  Layers: {SWEEP_LAYERS}", log_path)
    _log(f"  Steps: {SWEEP_STEPS}", log_path)
    _log(f"  Directions: {DIRECTIONS}", log_path)
    _log(f"  Epsilons: {SWEEP_EPSILONS}", log_path)
    _log(f"  N per combo: {args.n_per_combo}", log_path)
    _log(f"  Total generations: {len(SWEEP_LAYERS) * len(SWEEP_STEPS) * len(DIRECTIONS) * len(SWEEP_EPSILONS) * args.n_per_combo + args.n_baseline}", log_path)

    t0 = time.time()
    _log("\n  Loading model...", log_path)
    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
    _log(f"  Model loaded in {time.time() - t0:.1f}s", log_path)

    hall_data = load_jsonl(DATA_PATH)
    _log(f"  Loaded {len(hall_data)} hallucination samples", log_path)

    indices = list(range(len(hall_data)))
    rng = random.Random(args.seed)
    rng.shuffle(indices)
    max_idx = args.n_baseline + len(SWEEP_LAYERS) * len(DIRECTIONS) * len(SWEEP_EPSILONS) * args.n_per_combo
    selected = indices[:min(max_idx, len(hall_data))]

    baseline_indices = selected[:args.n_baseline]
    impulse_pool = selected[args.n_baseline:]

    _log(f"\n  Baseline indices: {baseline_indices[:5]}... (n={len(baseline_indices)})", log_path)
    _log(f"  Impulse pool: {len(impulse_pool)} samples for cycling", log_path)

    vectors = compute_all_directions(model, tokenizer, hall_data, log_path)

    _log(f"\n--- Baseline (no impulse) ---", log_path)
    baseline_results = []
    for i, idx in enumerate(tqdm(baseline_indices, desc="Baseline")):
        sample = hall_data[idx]
        result = run_single_impulse(
            model, tokenizer, sample, 12, vectors["v_hall"], "baseline", 0.0,
            MAX_NEW_TOKENS, TEMPERATURE, DO_SAMPLE)
        result["sample_idx"] = idx
        baseline_results.append(result)
        if i < 3:
            _log(f"  [{i}] beh={result['behavior']} | {result['generated_output'][:80]}", log_path)

    _log(f"\n--- Impulse sweep ---", log_path)
    impulse_results = []
    combo_list = []
    for layer in SWEEP_LAYERS:
        for direction_name in DIRECTIONS:
            for eps in SWEEP_EPSILONS:
                for rep in range(args.n_per_combo):
                    combo_list.append((layer, direction_name, eps, rep))

    _log(f"  Total combinations: {len(combo_list)}", log_path)

    for combo_idx, (layer, direction_name, eps, rep) in enumerate(tqdm(combo_list, desc="Impulse sweep")):
        sample_idx = impulse_pool[combo_idx % len(impulse_pool)]
        sample = hall_data[sample_idx]

        direction_vector = vectors[direction_name]

        result = run_single_impulse(
            model, tokenizer, sample, layer, direction_vector, direction_name, eps,
            MAX_NEW_TOKENS, TEMPERATURE, DO_SAMPLE)
        result["sample_idx"] = sample_idx
        result["combo_idx"] = combo_idx
        impulse_results.append(result)

        if combo_idx < 5:
            _log(f"  [{combo_idx}] L={layer} dir={direction_name} eps={eps} "
                 f"beh={result['behavior']} | {result['generated_output'][:60]}", log_path)

    for r in impulse_results:
        r["score"] = behavior_to_score(r["behavior"])

    impulse_df = pd.DataFrame(impulse_results)
    impulse_df.to_csv(os.path.join(RESULTS_DIR, "impulse_results.csv"), index=False)

    baseline_df = pd.DataFrame(baseline_results)
    baseline_df.to_csv(os.path.join(RESULTS_DIR, "baseline_results.csv"), index=False)

    per_direction = test_direction_specificity(baseline_results, impulse_df, log_path)

    summary = {
        "baseline_n": len(baseline_results),
        "impulse_n": len(impulse_results),
        "baseline_distribution": {str(k): int(v) for k, v in pd.DataFrame(baseline_results)["behavior"].value_counts().items()},
        "per_direction": {k: {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv) for kk, vv in v.items() if isinstance(vv, (int, float, str, np.floating, np.integer))}
                         for k, v in per_direction.items()},
        "sweep_config": {
            "layers": SWEEP_LAYERS,
            "steps": SWEEP_STEPS,
            "directions": DIRECTIONS,
            "epsilons": SWEEP_EPSILONS,
            "n_per_combo": args.n_per_combo,
        },
    }
    with open(os.path.join(RESULTS_DIR, "p2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _log(f"\n  === DONE ({time.time() - t0:.0f}s) ===", log_path)


if __name__ == "__main__":
    main()