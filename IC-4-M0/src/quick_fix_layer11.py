"""Quick fix: re-run seed0_layer11 shuffled test only."""
import os, sys, time, numpy as np, torch
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from src.model_loader import load_model_and_tokenizer
from src.evaluate import evaluate_outputs
from src.activation_collector import load_activations
from src.steering import get_all_vectors
from src.data_builder import load_jsonl
from src.run_p15_revalidate_30A30U import (
    collect_prefill_features, train_probe, find_transformer_layer, run_hard_gate_test
)

model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-0.5B-Instruct", device="cpu", torch_dtype="float32")
gen_cfg = {"max_new_tokens": 48, "temperature": 0.0, "do_sample": False}
layer, eval_seed, train_seeds = 11, 0, [0, 1]
alpha, threshold = -1.0, 0.5

pos_parts, neg_parts = [], []
for ts in train_seeds:
    a = load_activations(os.path.join(BASE_DIR, "results_m3", f"activations_s{ts}_l{layer}.npz"))
    pos_parts.append(a["positive"]); neg_parts.append(a["negative"])
pos_all = np.concatenate(pos_parts, axis=0)
neg_all = np.concatenate(neg_parts, axis=0)
all_vecs = get_all_vectors(pos_all, neg_all, pos_all.shape[1])
shuffled_v = all_vecs["shuffled"]
print(f"cos(steer,shuffled) = {np.dot(all_vecs['steering'], shuffled_v):.4f}")

train_path = os.path.join(BASE_DIR, "data_m3", "train.jsonl")
train_all = []
for ts in train_seeds:
    train_all.extend(load_jsonl(train_path.replace(".jsonl", f"_s{ts}.jsonl")))
X_tr, y_tr = collect_prefill_features(model, tokenizer, train_all, layer)
probe = train_probe(X_tr, y_tr)

test_f = os.path.join(BASE_DIR, "data_m3", f"test_s{eval_seed}.jsonl")
test_data = load_jsonl(test_f)

t0 = time.time()
res = run_hard_gate_test(model, tokenizer, test_data, shuffled_v, layer, alpha,
                          probe, threshold, gen_cfg, "seed0_layer11_shuffled")
m = evaluate_outputs(res)
print(f"shuffled H={m['hallucination_rate']:.3f} C={m['correct_answer_rate']:.3f}")
print(f"Done in {(time.time()-t0)/60:.1f} min")
print("Verdict:  random(1.000) > shuffled({:.3f}) > gate(0.733) → {} ".format(
    m['hallucination_rate'], 'PASS' if 1.0 > m['hallucination_rate'] > 0.733 else 'FAIL'))