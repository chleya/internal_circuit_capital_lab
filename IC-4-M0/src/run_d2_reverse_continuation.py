"""
IC-4-M5 D2-reverse continuation: finish α=+0.8, +1.0 for answer correctness.
"""
import sys, os, gc, re
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_loader import load_model_and_tokenizer
from run_m3_v6 import _collect_prefill_features, _train_probe, _load_cached_m3_data
from evaluate import generate_answers
from steering import compute_steering_vector, apply_steering_hook

GEN_CFG = {"max_new_tokens": 48, "temperature": 0.0, "do_sample": False}
SEED = 0
D2_LAYER = 12
D2_ALPHAS = [+0.6, +0.8, +1.0]
CSV_PATH = "results_m5/d1_d2_reverse_alpha.csv"

os.makedirs("results_m5", exist_ok=True)

REFUSAL_PATTERNS = [
    r"^(i don'?t know|i'm not sure|i cannot|i can'?t|i do not have)",
    r"^(unfortunately|sorry).*(don'?t|cannot|can'?t|not able|unable)",
    r"^(there is no|no information|not enough|insufficient)",
    r"^the (provided |given |)(context|text|passage|information|data).*(does not|doesn't|do not|don't)",
    r"^(it is not|it's not).*(possible|clear|specified|mentioned|stated|provided)",
]


def _log(msg):
    print(msg, flush=True)


def _is_refusal(output_text):
    for pat in REFUSAL_PATTERNS:
        if re.search(pat, output_text.strip().lower()):
            return True
    return False


def _contains_gold(generated, gold):
    return gold.strip().lower().rstrip(".") in generated.strip().lower()


def main():
    _log("=" * 56)
    _log("D2-REVERSE CONTINUATION: α=+0.6,+0.8,+1.0")
    _log("=" * 56)

    _log("Loading model...")
    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
    _log("Model loaded.\n")

    existing_df = pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else pd.DataFrame()
    d2_existing = existing_df[existing_df["experiment"] == "D2_reverse"] if len(existing_df) > 0 else pd.DataFrame()
    done_alphas = set(d2_existing["alpha"].tolist()) if len(d2_existing) > 0 else set()
    _log(f"Already done alphas: {done_alphas}")

    train_m3, test_m3 = _load_cached_m3_data(SEED, "data_m3/train.jsonl", "data_m3/test.jsonl", "")
    answerable_train_m3 = [s for s in train_m3 if s.get("answerability") == "answerable"][:30]
    answerable_test_m3 = [s for s in test_m3 if s.get("answerability") == "answerable"][:30]

    _log(f"D2: {len(answerable_train_m3)} train, {len(answerable_test_m3)} test samples")

    d2_train_base = generate_answers(model, tokenizer, answerable_train_m3, mode="base", **GEN_CFG)
    base_d2 = generate_answers(model, tokenizer, answerable_test_m3, mode="base", **GEN_CFG)
    n_correct = sum(1 for r, s in zip(base_d2, answerable_test_m3)
                    if _contains_gold(r["generated_output"], s.get("gold_answer", "")))
    d2_base_correct_rate = n_correct / len(answerable_test_m3)
    _log(f"D2 base correct_rate: {n_correct}/{len(answerable_test_m3)} = {d2_base_correct_rate:.4f}")

    correct_train = []
    incorrect_train = []
    for r, s in zip(d2_train_base, answerable_train_m3):
        gold = s.get("gold_answer", "")
        gen = r["generated_output"]
        if not _is_refusal(gen):
            if _contains_gold(gen, gold):
                correct_train.append({"context": s["context"], "question": s["question"], "gold_answer": gold})
            else:
                incorrect_train.append({"context": s["context"], "question": s["question"], "gold_answer": gold})

    _log(f"D2 train: correct={len(correct_train)} incorrect={len(incorrect_train)}")

    n_cvec = min(len(correct_train), len(incorrect_train))
    X_correct, _ = _collect_prefill_features(model, tokenizer,
        [{"context": r["context"], "question": r["question"]} for r in correct_train[:n_cvec]],
        D2_LAYER, "last_prompt_token")
    X_incorrect, _ = _collect_prefill_features(model, tokenizer,
        [{"context": r["context"], "question": r["question"]} for r in incorrect_train[:n_cvec]],
        D2_LAYER, "last_prompt_token")

    inter_d2 = np.linalg.norm(X_correct.mean(axis=0) - X_incorrect.mean(axis=0))
    intra_c = np.sqrt(np.mean(np.sum((X_correct - X_correct.mean(axis=0)) ** 2, axis=1)))
    intra_i = np.sqrt(np.mean(np.sum((X_incorrect - X_incorrect.mean(axis=0)) ** 2, axis=1)))
    snr_d2 = inter_d2 / ((intra_c + intra_i) / 2 + 1e-12)

    d2_sv = compute_steering_vector(X_correct, X_incorrect)
    X_probe_d2 = np.concatenate([X_correct, X_incorrect], axis=0)
    y_probe_d2 = np.array([1] * n_cvec + [0] * n_cvec)
    d2_probe = _train_probe(X_probe_d2, y_probe_d2, cv_folds=3)

    _log(f"D2 SNR={snr_d2:.4f} probe cv_acc={d2_probe.get('cv_acc_mean', 1.0):.4f}")

    all_rows = []
    if len(existing_df) > 0:
        all_rows = existing_df.to_dict("records")

    for alpha in D2_ALPHAS:
        if alpha in done_alphas:
            _log(f"  α={alpha:+.1f}: already done, skipping")
            continue

        incorrect_test = [r for r, s in zip(base_d2, answerable_test_m3)
                         if not _contains_gold(r["generated_output"], s.get("gold_answer", ""))]
        correct_test = [r for r, s in zip(base_d2, answerable_test_m3)
                       if _contains_gold(r["generated_output"], s.get("gold_answer", ""))]

        oracle_d2_results = []
        if correct_test:
            for r in generate_answers(model, tokenizer,
                [{"context": r["context"], "question": r["question"]} for r in correct_test],
                mode="base", **GEN_CFG):
                oracle_d2_results.append(r)
        if incorrect_test:
            handle = apply_steering_hook(model, D2_LAYER, d2_sv, alpha)
            for r in generate_answers(model, tokenizer,
                [{"context": r["context"], "question": r["question"]} for r in incorrect_test],
                mode="steering", max_new_tokens=GEN_CFG["max_new_tokens"],
                temperature=GEN_CFG["temperature"], do_sample=GEN_CFG["do_sample"]):
                oracle_d2_results.append(r)
            handle.remove()

        n_c_oracle = sum(1 for r, s in zip(oracle_d2_results, answerable_test_m3)
                        if _contains_gold(r["generated_output"], s.get("gold_answer", "")))
        oracle_cr = n_c_oracle / len(oracle_d2_results) if oracle_d2_results else 0
        delta_d2 = oracle_cr - d2_base_correct_rate

        _log(f"  D2 α={alpha:+.1f}: oracle_cr={oracle_cr:.4f} ({n_c_oracle}/{len(oracle_d2_results)}) Δ={delta_d2:+.4f}")

        all_rows.append({"experiment": "D2_reverse", "layer": D2_LAYER, "alpha": alpha,
                          "base_correct_rate": d2_base_correct_rate,
                          "oracle_correct_rate": oracle_cr,
                          "oracle_delta": delta_d2,
                          "snr": snr_d2,
                          "probe_cv_acc": d2_probe.get("cv_acc_mean", 1.0),
                          "behavior": "correctness", "seed": SEED, "mode": "oracle_only",
                          "alpha_sign": "positive"})

        gc.collect()
        df_tmp = pd.DataFrame(all_rows)
        df_tmp.to_csv(CSV_PATH, index=False)

    _log(f"\nData saved to {CSV_PATH}")
    _log("=== D2-REVERSE CONTINUATION DONE ===")


if __name__ == "__main__":
    main()