"""
Self-contained verification: runs GateSteeringTool and writes results to results_m4_generalization/verify_output.txt
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, torch
from src.gate_steering_tool import GateSteeringTool
from src.model_loader import load_model_and_tokenizer
from src.data_builder import load_jsonl

out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results_m4_generalization", "verify_output.txt")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

def log(msg):
    print(msg, flush=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

log("="*60)
log("GateSteeringTool Verification")
log("="*60)

log("\n[1/5] Loading model...")
model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-0.5B-Instruct", "cpu", "float32")

log("[2/5] Init GateSteeringTool...")
tool = GateSteeringTool(model, tokenizer, config={
    "threshold": 0.5, "cv_folds": 3, "max_new_tokens": 48, "temperature": 0.0, "do_sample": False,
})

log("[3/5] Loading data (seed=0)...")
train = load_jsonl("data_m3/train_s0.jsonl")
test = load_jsonl("data_m3/test_s0.jsonl")
na_t = sum(1 for s in train if s.get("answerability") == "answerable")
na_s = sum(1 for s in test if s.get("answerability") == "answerable")
log(f"  train: {na_t}A+{len(train)-na_t}U, test: {na_s}A+{len(test)-na_s}U")

log("[4/5] Training probe (layer=12, last_prompt_token)...")
probe_info = tool.train_probe(train, 12, "last_prompt_token")
log(f"  Probe: train_acc={probe_info['train_acc']:.4f}, cv_acc={probe_info.get('cv_acc_mean','N/A')}, AUC={probe_info.get('auc','N/A')}")

log("[5/5] Loading steering vector + running gate...")
vectors = GateSteeringTool.load_activation(0, 12)
sv = vectors["steering"]
log(f"  Steering dim={sv.shape[0]}")

t0 = time.time()
results_s = tool.generate_batch(test, sv, 12, -1.0, probe_info=probe_info, control_type="steering")
results_r = tool.generate_batch(test, vectors["random"], 12, -1.0, probe_info=probe_info, control_type="random")
results_f = tool.generate_batch(test, vectors["shuffled"], 12, -1.0, probe_info=probe_info, control_type="shuffled")
elapsed = time.time() - t0
log(f"  Generated 60 samples x 3 modes in {elapsed:.0f}s")

ms = tool.evaluate(results_s)
mr = tool.evaluate(results_r)
mf = tool.evaluate(results_f)

log(f"\n{'='*60}")
log("RESULTS")
log(f"{'='*60}")
log(f"  Real gate:         H={ms['hallucination_rate']:.3f} C={ms['correct_answer_rate']:.3f} UA={ms['unnecessary_abstention_rate']:.3f}")
log(f"  Random control:    H={mr['hallucination_rate']:.3f} C={mr['correct_answer_rate']:.3f}")
log(f"  Shuffled control:  H={mf['hallucination_rate']:.3f} C={mf['correct_answer_rate']:.3f}")
log(f"  Expected (M3-v6):  H=0.667 C=0.600 UA=0.000")

h_ok = abs(ms['hallucination_rate'] - 0.667) <= 0.02
c_ok = abs(ms['correct_answer_rate'] - 0.600) <= 0.02
ua_ok = abs(ms['unnecessary_abstention_rate'] - 0.000) <= 0.02
sep_ok = mr['hallucination_rate'] > ms['hallucination_rate'] + 0.10

checks = [
    ("H=0.667 +/- 0.02", h_ok, f"H={ms['hallucination_rate']:.3f}"),
    ("C=0.600 +/- 0.02", c_ok, f"C={ms['correct_answer_rate']:.3f}"),
    ("UA=0.000 +/- 0.02", ua_ok, f"UA={ms['unnecessary_abstention_rate']:.3f}"),
    ("Real beats random >0.10", sep_ok, f"sep={mr['hallucination_rate']-ms['hallucination_rate']:.3f}"),
]

log(f"\n{'='*60}")
log("CHECKS")
log(f"{'='*60}")
all_ok = True
for name, ok, val in checks:
    log(f"  {'PASS' if ok else 'FAIL'}  {name:<30} ({val})")
    if not ok:
        all_ok = False

log(f"\n{'ALL CHECKS PASSED — Tool verified against M3-v6 reference.' if all_ok else 'SOME CHECKS FAILED.'}")
log("Done.")