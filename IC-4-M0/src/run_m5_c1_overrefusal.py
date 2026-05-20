"""
IC-4-M5 C1: Over-Refusal — Does IC-4 generalize to reducing unnecessary refusals?

Behavior: On ANSWERABLE questions, the model sometimes refuses ("I don't know").
This is the mirror image of hallucination (refusing what it should answer vs
answering what it should refuse).

Key question: Since the anti-hallucination vector pushes toward "refuse", can the
SAME vector with POSITIVE alpha reduce over-refusal? Or do we need a separate
"anti-refusal" vector?

Pipeline:
  1. Load M3 answerable data, label outputs as "refused" vs "answered_correctly"
  2. Train over-refusal probe on prefill hidden states
  3. Test anti-hallucination vector transfer (α > 0 should reduce refusal)
  4. Compute refusal-specific steering vector
  5. Build + evaluate refusal gate (oracle / real / random / shuffled)
"""
import sys, os, json, random, re, numpy as np, pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_loader import load_model_and_tokenizer
from run_m3_v6 import _collect_prefill_features, _train_probe, _generate_single_pass_hard_gate
from run_m3_v6 import _load_cached_m3_data
from evaluate import generate_answers, evaluate_outputs, _contains_gold
from steering import (compute_steering_vector, compute_random_vector,
                       compute_shuffled_vector, apply_steering_hook)
from activation_collector import load_activations

GEN_CFG = {"max_new_tokens": 48, "temperature": 0.0, "do_sample": False}
LAYER = 12
SEED = 0
ALPHAS = [+0.4, +0.8, +1.0, +1.2]
CSV_PATH = "results_m5/c1_overrefusal_metrics.csv"
REPORT_PATH = "results_m5/c1_overrefusal_report.txt"

os.makedirs("results_m5", exist_ok=True)

REFUSAL_PATTERNS = [
    r"^(i don'?t know|i'm not sure|i cannot|i can'?t|i do not have)",
    r"^(unfortunately|sorry).*(don'?t|cannot|can'?t|not able|unable)",
    r"^(there is no|no information|not enough|insufficient)",
    r"^the (provided |given |)(context|text|passage|information|data).*(does not|doesn't|do not|don't)",
    r"^(it is not|it's not).*(possible|clear|specified|mentioned|stated|provided)",
]


def _log(msg, log_file=None):
    print(msg, flush=True)
    if log_file and isinstance(log_file, str):
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def _is_refusal(output_text):
    text = output_text.strip().lower()
    for pat in REFUSAL_PATTERNS:
        if re.search(pat, text):
            return True
    return False


def _contains_gold_approx(generated, gold):
    gold_clean = gold.strip().lower().rstrip(".")
    gen_clean = generated.strip().lower()
    return gold_clean in gen_clean


def _compute_metrics(results, label):
    n = len(results)
    if n == 0:
        return {"mode": label, "n": 0, "n_correct": 0, "n_refusal": 0,
                "correct_rate": 0.0, "refusal_rate": 0.0}
    n_refusal = sum(1 for r in results if _is_refusal(r["generated_output"]))
    n_correct = sum(1 for r in results if _contains_gold_approx(
        r["generated_output"], r.get("gold_answer", "")))
    return {
        "mode": label, "n": n, "n_correct": n_correct, "n_refusal": n_refusal,
        "correct_rate": n_correct / n, "refusal_rate": n_refusal / n,
    }


def main():
    _log("=" * 56)
    _log("IC-4-M5 C1: Over-Refusal on Answerable Questions")
    _log("=" * 56)

    _log("Loading model...")
    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
    _log("Model loaded.\n")

    _log("Loading M3 answerable data...")
    train_data, test_data = _load_cached_m3_data(SEED, "data_m3/train.jsonl", "data_m3/test.jsonl", "")
    answerable_train = [s for s in train_data if s.get("answerability") == "answerable"]
    answerable_test = [s for s in test_data if s.get("answerability") == "answerable"]
    _log(f"  answerable samples: train={len(answerable_train)} test={len(answerable_test)}")

    n_test = min(30, len(answerable_test))
    test_data = answerable_test[:n_test]
    train_data = answerable_train[:min(30, len(answerable_train))]

    _log("Generating base outputs...")
    base_test = generate_answers(model, tokenizer, test_data, mode="base", **GEN_CFG)
    for r, s in zip(base_test, test_data):
        r["gold_answer"] = s.get("gold_answer", "")
    base_metrics = _compute_metrics(base_test, "base")
    base_metrics.update({"seed": SEED, "layer": LAYER, "alpha": 0.0,
                          "vector_type": "none", "scenario": "overrefusal"})
    _log(f"  BASE: correct={base_metrics['n_correct']}/{base_metrics['n']} "
         f"refusal={base_metrics['n_refusal']}/{base_metrics['n']} "
         f"(refusal_rate={base_metrics['refusal_rate']:.4f})")

    base_train = generate_answers(model, tokenizer, train_data, mode="base", **GEN_CFG)
    for r, s in zip(base_train, train_data):
        r["gold_answer"] = s.get("gold_answer", "")
        r["answerability"] = s.get("answerability", "")

    refused_train = [r for r in base_train if _is_refusal(r["generated_output"])]
    answered_train = [r for r in base_train if not _is_refusal(r["generated_output"])
                       and _contains_gold_approx(r["generated_output"], r.get("gold_answer", ""))]
    _log(f"  train: refused={len(refused_train)} answered_correctly={len(answered_train)}")

    all_rows = [base_metrics]

    if len(refused_train) < 3 or len(answered_train) < 3:
        _log("  NOT ENOUGH refused/answered samples for training, skipping.")
        df = pd.DataFrame(all_rows)
        df.to_csv(CSV_PATH, index=False)
        return

    n_vec = min(len(refused_train), len(answered_train))
    vec_refused = refused_train[:n_vec]
    vec_answered = answered_train[:n_vec]

    _log("  Collecting prefill features...")
    X_refused, _ = _collect_prefill_features(model, tokenizer, vec_refused, LAYER, "last_prompt_token")
    X_answered, _ = _collect_prefill_features(model, tokenizer, vec_answered, LAYER, "last_prompt_token")

    refusal_sv = compute_steering_vector(X_refused, X_answered)
    refusal_random = compute_random_vector(X_refused.shape[1], seed=SEED * 100 + 42)
    refusal_shuffled = compute_shuffled_vector(X_refused, X_answered, seed=SEED * 100 + 43)
    refusal_vectors = {"steering": refusal_sv, "random": refusal_random, "shuffled": refusal_shuffled}

    cos_rr = np.dot(refusal_sv, refusal_random) / (
        np.linalg.norm(refusal_sv) * np.linalg.norm(refusal_random) + 1e-12)
    cos_rs = np.dot(refusal_sv, refusal_shuffled) / (
        np.linalg.norm(refusal_sv) * np.linalg.norm(refusal_shuffled) + 1e-12)
    _log(f"  refusal vector: cos(real,random)={cos_rr:+.4f} cos(real,shuffled)={cos_rs:+.4f}")

    _log("  Training refusal probe...")
    probe_train_labels = []
    for r in vec_refused:
        probe_train_labels.append({
            "context": r["context"], "question": r["question"],
            "label": "refused"
        })
    for r in vec_answered:
        probe_train_labels.append({
            "context": r["context"], "question": r["question"],
            "label": "answered"
        })
    X_probe, y_probe = _collect_prefill_features(model, tokenizer, probe_train_labels, LAYER, "last_prompt_token")
    y_probe_arr = np.array([1 if s["label"] == "refused" else 0 for s in probe_train_labels])
    refusal_probe = _train_probe(X_probe, y_probe_arr, cv_folds=3)
    _log(f"  refusal probe: train_acc={refusal_probe['train_acc']:.4f} "
         f"cv_acc={refusal_probe.get('cv_acc_mean', 1.0):.4f}")
    probe_cfg = {"representation": "last_prompt_token", "threshold": 0.5, "steepness": 1.0}

    refused_test = [r for r in base_test if _is_refusal(r["generated_output"])]
    non_refused_test = [r for r in base_test if not _is_refusal(r["generated_output"])]
    _log(f"  oracle groups: refused={len(refused_test)} non_refused={len(non_refused_test)}")

    for alpha in ALPHAS:
        _log(f"\n  --- alpha={alpha:+.1f} (POSITIVE — push away from refusal) ---")

        oracle_results = []
        if non_refused_test:
            for r in generate_answers(model, tokenizer, non_refused_test, mode="base", **GEN_CFG):
                r["mode"] = f"oracle_refusal_a{alpha:+.1f}"
                r["alpha_applied"] = 0.0
                oracle_results.append(r)
        if refused_test:
            handle = apply_steering_hook(model, LAYER, refusal_sv, alpha)
            for r in generate_answers(model, tokenizer, refused_test, mode="steering",
                                      max_new_tokens=GEN_CFG["max_new_tokens"],
                                      temperature=GEN_CFG["temperature"],
                                      do_sample=GEN_CFG["do_sample"]):
                r["mode"] = f"oracle_refusal_a{alpha:+.1f}"
                r["alpha_applied"] = alpha
                oracle_results.append(r)
            handle.remove()
        for r in oracle_results:
            r["gold_answer"] = r.get("gold_answer", "")
        oracle_metrics = _compute_metrics(oracle_results, f"oracle_refusal_a{alpha:+.1f}")
        oracle_metrics.update({"seed": SEED, "layer": LAYER, "alpha": alpha,
                                "vector_type": "steering", "scenario": "overrefusal"})
        all_rows.append(oracle_metrics)
        _log(f"    ORACLE gate: refusal_rate={oracle_metrics['refusal_rate']:.4f} "
             f"(refused={oracle_metrics['n_refusal']}/{oracle_metrics['n']})")

        for vtype, vlabel in [("steering", "real"), ("random", "random"), ("shuffled", "shuffled")]:
            mode_name = f"{vlabel}_refusal_a{alpha:+.1f}"
            gen_res = _generate_single_pass_hard_gate(
                model, tokenizer, [{"context": r["context"], "question": r["question"]}
                                   for r in base_test],
                refusal_vectors[vtype], LAYER, alpha, refusal_probe, probe_cfg, GEN_CFG, vtype)
            for r, s in zip(gen_res, base_test):
                r["gold_answer"] = s.get("gold_answer", "")
            gen_metrics = _compute_metrics(gen_res, mode_name)
            gen_metrics.update({"seed": SEED, "layer": LAYER, "alpha": alpha,
                                 "vector_type": vtype, "scenario": "overrefusal"})
            all_rows.append(gen_metrics)
            _log(f"    {mode_name:>25s}: refusal_rate={gen_metrics['refusal_rate']:.4f} "
                 f"(refused={gen_metrics['n_refusal']}/{gen_metrics['n']})")

    df = pd.DataFrame(all_rows)
    df.to_csv(CSV_PATH, index=False)

    _log("\n" + "=" * 56)
    _log("C1 RESULTS: OVER-REFUSAL")
    _log("=" * 56)
    base_rr = base_metrics["refusal_rate"]
    for alpha in ALPHAS:
        subset = df[df["alpha"] == alpha]
        oracle_row = subset[subset["mode"].str.contains("oracle")]
        real_row = subset[subset["mode"].str.contains("real")]
        if len(oracle_row) > 0 and len(real_row) > 0:
            o_rr = oracle_row["refusal_rate"].values[0]
            r_rr = real_row["refusal_rate"].values[0]
            _log(f"  α={alpha:+.1f}: oracle_refusal={o_rr:.4f} real_refusal={r_rr:.4f} "
                 f"(base={base_rr:.4f})")

    _log(f"\nData saved to {CSV_PATH}")
    _log("=== C1 DONE ===")


if __name__ == "__main__":
    main()