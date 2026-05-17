"""
IC-4-M5 P3-A: Factuality Hallucination — Vector Transfer Probe.

Tests whether the anti-hallucination steering vector transfers to
factuality hallucination by:
1. Labelling answerable samples as factually_correct/incorrect from base outputs
2. Training a factuality probe (correct vs incorrect generation)
3. Checking if the existing steering direction correlates with factuality
"""

import sys, os, json, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_m3_v6 import _load_cached_m3_data, _collect_prefill_features, _train_probe
from model_loader import load_model_and_tokenizer
from activation_collector import load_activations
from steering import compute_steering_vector
from evaluate import generate_answers, evaluate_outputs, _contains_gold

GEN_CFG = {"max_new_tokens": 48, "temperature": 0.0, "do_sample": False}
LAYER = 12
ALPHA = -1.0
TRAIN_PATH = "data_m3/train.jsonl"
TEST_PATH = "data_m3/test.jsonl"

def _log(msg):
    print(msg, flush=True)

def main():
    print("=" * 56)
    print("IC-4-M5 P3-A: Factuality Vector Transfer Analysis")
    print("=" * 56)

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
    print("Model loaded.")

    for seed in [0, 1, 2]:
        print(f"\n{'='*48}")
        print(f"SEED={seed}")
        print(f"{'='*48}")

        train, test = _load_cached_m3_data(seed, TRAIN_PATH, TEST_PATH, None)
        answerable = [s for s in test if s.get("answerability") == "answerable"]

        print(f"  answerable samples: {len(answerable)}/{len(test)}")

        base_res = generate_answers(model, tokenizer, answerable, mode="base", **GEN_CFG)
        correct_ct = 0
        incorrect_ct = 0
        correct_samples = []
        incorrect_samples = []
        for r in base_res:
            if _contains_gold(r["generated_output"], r["gold_answer"]):
                correct_ct += 1
                correct_samples.append(r)
            else:
                incorrect_ct += 1
                incorrect_samples.append(r)

        factual_error_rate = incorrect_ct / len(answerable)
        correct_rate = correct_ct / len(answerable)
        print(f"  base factual: correct={correct_ct} incorrect={incorrect_ct}")
        print(f"    factual_error_rate={factual_error_rate:.4f}")
        print(f"    correct_answer_rate={correct_rate:.4f}")

        # Train factuality probe
        train_answerable = [s for s in train if s.get("answerability") == "answerable"]
        train_correct = []
        train_incorrect = []
        train_base = generate_answers(model, tokenizer, train_answerable, mode="base", **GEN_CFG)
        for r in train_base:
            if _contains_gold(r["generated_output"], r["gold_answer"]):
                train_correct.append(r)
            else:
                train_incorrect.append(r)

        print(f"  probe train: {len(train_correct)} correct + {len(train_incorrect)} incorrect")

        if len(train_correct) < 5 or len(train_incorrect) < 5:
            print("  NOT ENOUGH samples for probe training, skipping")
            continue

        n = min(len(train_correct), len(train_incorrect))
        train_pos = train_correct[:n]
        train_neg = train_incorrect[:n]
        probe_train = []
        for r in train_pos:
            probe_train.append({"context": r["context"], "question": r["question"],
                                "gold_answer": r["gold_answer"], "label": "factually_correct"})
        for r in train_neg:
            probe_train.append({"context": r["context"], "question": r["question"],
                                "gold_answer": r["gold_answer"], "label": "factually_incorrect"})

        X_tr, y_tr = _collect_prefill_features(model, tokenizer, probe_train, LAYER, "last_prompt_token")
        # labels: 1 = factually_correct, 0 = factually_incorrect
        y_tr = np.array([1 if s["label"] == "factually_correct" else 0 for s in probe_train])
        fact_probe = _train_probe(X_tr, y_tr, cv_folds=3)
        print(f"  factuality probe: train_acc={fact_probe['train_acc']:.4f} cv_acc={fact_probe.get('cv_acc_mean',1.0):.4f}")

        # Check if anti-hallucination steering direction correlates with factuality
        act_path = f"results_m3/activations_s{seed}_l{LAYER}.npz"
        acts = load_activations(act_path)
        sv = compute_steering_vector(acts["positive"], acts["negative"])

        # Collect features for test answerable, project onto steering vector
        test_samples = []
        for r in base_res:
            test_samples.append({"context": r["context"], "question": r["question"],
                                 "gold_answer": r["gold_answer"], "label": "answerable"})
        X_test, _ = _collect_prefill_features(model, tokenizer, test_samples, LAYER, "last_prompt_token")

        # Project all answerable samples onto steering direction
        projections = X_test @ sv
        correct_mask = np.array([1 if _contains_gold(r["generated_output"], r["gold_answer"]) else 0
                                  for r in base_res])
        correct_proj = projections[correct_mask == 1]
        incorrect_proj = projections[correct_mask == 0]

        delta = np.mean(correct_proj) - np.mean(incorrect_proj)
        print(f"  steering projection: correct_mean={np.mean(correct_proj):+.4f} "
              f"incorrect_mean={np.mean(incorrect_proj):+.4f}")
        print(f"    delta (correct-incorrect) = {delta:+.4f}")
        if delta > 0:
            print(f"    → steering direction aligns with factual correctness (+{delta:.4f})")
        else:
            print(f"    → steering direction OPPOSES factual correctness ({delta:.4f})")

        # Factuality probe test accuracy
        n_test = min(len(correct_samples), len(incorrect_samples))
        test_pos = correct_samples[:n_test]
        test_neg = incorrect_samples[:n_test]
        probe_test = []
        for r in test_pos:
            probe_test.append({"context": r["context"], "question": r["question"],
                               "gold_answer": r["gold_answer"], "label": "factually_correct"})
        for r in test_neg:
            probe_test.append({"context": r["context"], "question": r["question"],
                               "gold_answer": r["gold_answer"], "label": "factually_incorrect"})

        X_te, y_te = _collect_prefill_features(model, tokenizer, probe_test, LAYER, "last_prompt_token")
        y_te = np.array([1 if s["label"] == "factually_correct" else 0 for s in probe_test])
        from sklearn.metrics import accuracy_score
        X_te_scaled = fact_probe["scaler"].transform(X_te)
        y_pred = fact_probe["classifier"].predict(X_te_scaled)
        test_acc = accuracy_score(y_te, y_pred)
        print(f"  factuality probe test acc: {test_acc:.4f}")

    print("\n=== DONE ===")

if __name__ == "__main__":
    main()