"""
IC-4-M5 P3-B/C: Factuality Hallucination — Dedicated Steering Vector & Gate.

Computes a factuality-specific steering vector from factually_correct vs
factually_incorrect activations, then tests it with a single-pass hard gate.
Compares results with the anti-hallucination vector transfer (P3-A).
"""

import sys, os, json, numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_m3_v6 import _load_cached_m3_data, _collect_prefill_features, _train_probe
from run_m3_v6 import _generate_single_pass_hard_gate
from model_loader import load_model_and_tokenizer
from activation_collector import load_activations
from steering import compute_steering_vector, compute_random_vector, compute_shuffled_vector, apply_steering_hook
from evaluate import generate_answers, evaluate_outputs, _contains_gold

GEN_CFG = {"max_new_tokens": 48, "temperature": 0.0, "do_sample": False}
LAYER = 12
ALPHAS = [-0.8, -1.0, -1.2]
TRAIN_PATH = "data_m3/train.jsonl"
TEST_PATH = "data_m3/test.jsonl"
CSV_PATH = "results_m5/p3b_factuality_metrics.csv"
REPORT_PATH = "results_m5/p3b_factuality_report.txt"

os.makedirs("results_m5", exist_ok=True)


def _log(msg):
    print(msg, flush=True)


def _contains_gold_approx(generated, gold):
    gold_clean = gold.strip().lower().rstrip(".")
    gen_clean = generated.strip().lower()
    return gold_clean in gen_clean


def _factuality_label_from_output(result):
    return "factually_correct" if _contains_gold_approx(
        result["generated_output"], result["gold_answer"]) else "factually_incorrect"


def _get_oracle_gated_answers(model, tokenizer, samples, steering_v, layer, alpha,
                               mode_name, gen_cfg):
    max_new = gen_cfg.get("max_new_tokens", 48)
    temp = gen_cfg.get("temperature", 0.0)
    do_sample = gen_cfg.get("do_sample", False)
    results = []
    correct_s = [s for s in samples
                 if _contains_gold_approx(s.get("base_output", ""), s["gold_answer"])]
    incorrect_s = [s for s in samples
                   if not _contains_gold_approx(s.get("base_output", ""), s["gold_answer"])]
    if correct_s:
        for r in generate_answers(model, tokenizer, correct_s, mode="base",
                                  max_new_tokens=max_new, temperature=temp, do_sample=do_sample):
            r["mode"] = mode_name; r["alpha"] = alpha; r["alpha_applied"] = 0.0
            r["vector_type"] = "steering"
            results.append(r)
    if incorrect_s:
        handle = apply_steering_hook(model, layer, steering_v, alpha)
        for r in generate_answers(model, tokenizer, incorrect_s, mode="steering",
                                  max_new_tokens=max_new, temperature=temp, do_sample=do_sample):
            r["mode"] = mode_name; r["alpha"] = alpha; r["alpha_applied"] = alpha
            r["vector_type"] = "steering"
            results.append(r)
        handle.remove()
    return results


def _compute_metrics(results, label):
    n = len(results)
    n_correct = sum(1 for r in results
                    if _contains_gold_approx(r["generated_output"], r["gold_answer"]))
    factual_error = 1.0 - n_correct / n if n > 0 else 1.0
    avg_len = np.mean([len(r["generated_output"]) for r in results]) if results else 0

    return {
        "mode": label, "n": n, "n_correct": n_correct,
        "factual_error_rate": factual_error,
        "correct_answer_rate": n_correct / n if n > 0 else 0,
        "avg_output_len": avg_len,
    }


def main():
    _log("=" * 56)
    _log("IC-4-M5 P3-B/C: Factuality Steering Vector & Gate")
    _log("=" * 56)

    _log("Loading model...")
    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
    _log("Model loaded.\n")

    all_rows = []

    for seed in [0, 1, 2]:
        _log(f"\n{'='*48}")
        _log(f"SEED={seed}")
        _log(f"{'='*48}")

        train, test = _load_cached_m3_data(seed, TRAIN_PATH, TEST_PATH, None)
        train_answerable = [s for s in train if s.get("answerability") == "answerable"]
        test_answerable = [s for s in test if s.get("answerability") == "answerable"]
        _log(f"  train answerable: {len(train_answerable)}, test answerable: {len(test_answerable)}")

        base_test = generate_answers(model, tokenizer, test_answerable, mode="base", **GEN_CFG)
        for r in base_test:
            r["seed"] = seed
            r["scenario"] = "standard"
        base_metrics = _compute_metrics(base_test, "base")
        base_metrics.update({"seed": seed, "layer": LAYER, "alpha": 0.0,
                              "vector_type": "none", "scenario": "standard"})
        all_rows.append(base_metrics)
        _log(f"  BASE factual_error_rate={base_metrics['factual_error_rate']:.4f} "
             f"({base_metrics['n_correct']}/{base_metrics['n']})")

        base_train = generate_answers(model, tokenizer, train_answerable, mode="base", **GEN_CFG)
        for r in base_train:
            r["seed"] = seed

        correct_train = [r for r in base_train
                         if _contains_gold_approx(r["generated_output"], r["gold_answer"])]
        incorrect_train = [r for r in base_train
                           if not _contains_gold_approx(r["generated_output"], r["gold_answer"])]
        _log(f"  factuality labels (train): correct={len(correct_train)} incorrect={len(incorrect_train)}")

        if len(correct_train) < 3 or len(incorrect_train) < 3:
            _log("  NOT ENOUGH samples for factuality vector training, skipping seed")
            continue

        n_vec = min(len(correct_train), len(incorrect_train))
        vec_correct = correct_train[:n_vec]
        vec_incorrect = incorrect_train[:n_vec]

        X_correct, _ = _collect_prefill_features(model, tokenizer, vec_correct, LAYER, "last_prompt_token")
        X_incorrect, _ = _collect_prefill_features(model, tokenizer, vec_incorrect, LAYER, "last_prompt_token")

        factuality_sv = compute_steering_vector(X_incorrect, X_correct)
        factuality_random = compute_random_vector(X_correct.shape[1], seed=seed * 100 + 42)
        factuality_shuffled = compute_shuffled_vector(X_incorrect, X_correct, seed=seed * 100 + 123)

        fact_vectors = {"steering": factuality_sv, "random": factuality_random, "shuffled": factuality_shuffled}

        cos_real_random = np.dot(factuality_sv, factuality_random) / (
            np.linalg.norm(factuality_sv) * np.linalg.norm(factuality_random) + 1e-12)
        cos_real_shuffled = np.dot(factuality_sv, factuality_shuffled) / (
            np.linalg.norm(factuality_sv) * np.linalg.norm(factuality_shuffled) + 1e-12)
        _log(f"  factuality vector cos_sim: real-random={cos_real_random:+.4f} "
             f"real-shuffled={cos_real_shuffled:+.4f}")

        probe_train_data = []
        for r in vec_correct:
            probe_train_data.append({"context": r["context"], "question": r["question"],
                                      "gold_answer": r["gold_answer"], "label": "factually_correct"})
        for r in vec_incorrect:
            probe_train_data.append({"context": r["context"], "question": r["question"],
                                      "gold_answer": r["gold_answer"], "label": "factually_incorrect"})
        X_probe, y_probe = _collect_prefill_features(model, tokenizer, probe_train_data, LAYER, "last_prompt_token")
        y_probe = np.array([1 if s["label"] == "factually_correct" else 0 for s in probe_train_data])
        fact_probe = _train_probe(X_probe, y_probe, cv_folds=3)
        _log(f"  factuality probe: train_acc={fact_probe['train_acc']:.4f} "
             f"cv_acc={fact_probe.get('cv_acc_mean', 1.0):.4f}")
        probe_cfg = {"representation": "last_prompt_token", "threshold": 0.5, "steepness": 1.0}

        for alpha in ALPHAS:
            _log(f"\n  --- alpha={alpha:+.1f} ---")

            oracle_res = _get_oracle_gated_answers(
                model, tokenizer, test_answerable, factuality_sv, LAYER, alpha,
                f"oracle_factuality_gate_a{alpha:+.1f}", GEN_CFG)
            oracle_metrics = _compute_metrics(oracle_res, f"oracle_factuality_gate_a{alpha:+.1f}")
            oracle_metrics.update({"seed": seed, "layer": LAYER, "alpha": alpha,
                                    "vector_type": "steering", "scenario": "standard"})
            all_rows.append(oracle_metrics)
            _log(f"    ORACLE gate: factual_error_rate={oracle_metrics['factual_error_rate']:.4f} "
                 f"({oracle_metrics['n_correct']}/{oracle_metrics['n']})")

            for vtype, vlabel in [("steering", "real"), ("random", "random"), ("shuffled", "shuffled")]:
                mode_name = f"{vlabel}_factuality_gate_a{alpha:+.1f}"
                gen_res = _generate_single_pass_hard_gate(
                    model, tokenizer, test_answerable, fact_vectors[vtype], LAYER,
                    alpha, fact_probe, probe_cfg, GEN_CFG, vtype)
                gen_metrics = _compute_metrics(gen_res, mode_name)
                gen_metrics.update({"seed": seed, "layer": LAYER, "alpha": alpha,
                                     "vector_type": vtype, "scenario": "standard"})
                all_rows.append(gen_metrics)
                _log(f"    {mode_name:>30s}: factual_error_rate={gen_metrics['factual_error_rate']:.4f} "
                     f"({gen_metrics['n_correct']}/{gen_metrics['n']})")

        df = pd.DataFrame(all_rows)
        df.to_csv(CSV_PATH, index=False)
        _log(f"\n  [saved {CSV_PATH}]")

    _log("\n" + "=" * 56)
    _log("SUMMARY REPORT")
    _log("=" * 56)

    df = pd.DataFrame(all_rows)
    for seed in sorted(df["seed"].unique()):
        base_row = df[(df["seed"] == seed) & (df["mode"] == "base")]
        if len(base_row) == 0:
            continue
        base_fe = base_row["factual_error_rate"].values[0]
        _log(f"\n  Seed={seed} | BASE factual_error_rate={base_fe:.4f}")

        for alpha in sorted(df[df["seed"] == seed]["alpha"].unique()):
            if alpha == 0.0:
                continue
            subset = df[(df["seed"] == seed) & (df["alpha"] == alpha)]
            oracle_row = subset[subset["mode"].str.contains("oracle")]
            real_row = subset[subset["mode"].str.contains("real")]
            if len(oracle_row) > 0 and len(real_row) > 0:
                oracle_fe = oracle_row["factual_error_rate"].values[0]
                real_fe = real_row["factual_error_rate"].values[0]
                reduc = (base_fe - real_fe) / (base_fe - oracle_fe + 1e-12) * 100
                _log(f"    alpha={alpha:+.1f}: oracle_fe={oracle_fe:.4f} real_fe={real_fe:.4f} "
                     f"reduction={reduc:.1f}%")

    with open(REPORT_PATH, "w") as f:
        f.write("IC-4-M5 P3-B/C FACTUALITY GATE REPORT\n")
        f.write("=" * 56 + "\n\n")
        f.write(df.to_string(index=False))
    _log(f"\nReport saved to {REPORT_PATH}")
    _log("\n=== DONE ===")


if __name__ == "__main__":
    main()