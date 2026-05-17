"""
Diagnostic: Isolate the source of C=0.733 anomaly in M3-v6.

Tests:
  A: model.generate() — baseline (known C=0.600)
  B: Manual token-by-token, NO hook — tests manual generation loop
  C: Manual token-by-token, WITH do-nothing hook — tests hook presence alone
  D: model.generate() WITH do-nothing hook — tests hook + generate()

All output written to results_m3_v6/diagnose_output.txt
"""
import os, sys, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results_m3_v6", "diagnose_output.txt")
os.makedirs(os.path.dirname(log_path), exist_ok=True)

log_lines = []
def log(msg):
    print(msg, flush=True)
    log_lines.append(msg)

log("=" * 60)
log("DIAGNOSTIC: C=0.733 root cause isolation")
log("=" * 60)

from src.model_loader import load_model_and_tokenizer
from src.data_builder import load_jsonl
from src.evaluate import evaluate_outputs

log("Loading model...")
model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-0.5B-Instruct", "cpu", "float32")
test = load_jsonl("data_m3/test_s0.jsonl")
max_new = 48
eos_id = tokenizer.eos_token_id
layer_module = model.model.layers[12]

# ===== A: model.generate() =====
log("\n--- A: model.generate() ---")
results_a = []
for sid, sample in enumerate(test):
    prompt = f"{sample.get('context','')}\n\nQuestion: {sample.get('question','')}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new, temperature=0.0,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                              skip_special_tokens=True).strip()
    results_a.append({**sample, "sample_id": sid, "generated_output": answer,
                       "mode": "diagnose_a_generate"})
    if sid < 5:
        log(f"  [{sid}] label={sample.get('answerability')}: {answer[:120]}")
m_a = evaluate_outputs(results_a)
log(f"  A: H={m_a['hallucination_rate']:.3f} C={m_a['correct_answer_rate']:.3f}")

# ===== B: Manual token-by-token, NO hook =====
log("\n--- B: Manual token-by-token, NO hook ---")
results_b = []
for sid, sample in enumerate(test):
    prompt = f"{sample.get('context','')}\n\nQuestion: {sample.get('question','')}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    pkv = outputs.past_key_values
    generated_ids = []
    current_input = input_ids
    for _step in range(max_new):
        with torch.no_grad():
            if pkv is not None:
                current_input = current_input[:, -1:]
            out = model(input_ids=current_input, past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values
            logits = out.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            tid = next_token.item()
            generated_ids.append(tid)
            if tid == eos_id:
                break
            current_input = next_token
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    results_b.append({**sample, "sample_id": sid, "generated_output": answer,
                       "mode": "diagnose_b_manual_no_hook"})
    if sid < 5:
        log(f"  [{sid}] label={sample.get('answerability')}: {answer[:120]}")
m_b = evaluate_outputs(results_b)
log(f"  B: H={m_b['hallucination_rate']:.3f} C={m_b['correct_answer_rate']:.3f}")

# ===== C: Manual token-by-token, WITH do-nothing hook =====
log("\n--- C: Manual token-by-token, WITH do-nothing hook ---")
results_c = []
for sid, sample in enumerate(test):
    prompt = f"{sample.get('context','')}\n\nQuestion: {sample.get('question','')}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"]

    def do_nothing_hook(module, inputs, outputs):
        return None

    handle = layer_module.register_forward_hook(do_nothing_hook)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    pkv = outputs.past_key_values
    generated_ids = []
    current_input = input_ids
    for _step in range(max_new):
        with torch.no_grad():
            if pkv is not None:
                current_input = current_input[:, -1:]
            out = model(input_ids=current_input, past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values
            logits = out.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            tid = next_token.item()
            generated_ids.append(tid)
            if tid == eos_id:
                break
            current_input = next_token
    handle.remove()
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    results_c.append({**sample, "sample_id": sid, "generated_output": answer,
                       "mode": "diagnose_c_hook_passthrough"})
    if sid < 5:
        log(f"  [{sid}] label={sample.get('answerability')}: {answer[:120]}")
m_c = evaluate_outputs(results_c)
log(f"  C: H={m_c['hallucination_rate']:.3f} C={m_c['correct_answer_rate']:.3f}")

# ===== D: model.generate() WITH do-nothing hook =====
log("\n--- D: model.generate() WITH do-nothing hook ---")
results_d = []
for sid, sample in enumerate(test):
    prompt = f"{sample.get('context','')}\n\nQuestion: {sample.get('question','')}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    def do_nothing_hook2(module, inputs, outputs):
        return None

    handle = layer_module.register_forward_hook(do_nothing_hook2)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new, temperature=0.0,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
    handle.remove()
    answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                              skip_special_tokens=True).strip()
    results_d.append({**sample, "sample_id": sid, "generated_output": answer,
                       "mode": "diagnose_d_generate_with_hook"})
    if sid < 5:
        log(f"  [{sid}] label={sample.get('answerability')}: {answer[:120]}")
m_d = evaluate_outputs(results_d)
log(f"  D: H={m_d['hallucination_rate']:.3f} C={m_d['correct_answer_rate']:.3f}")

# ===== SUMMARY =====
log("\n" + "=" * 60)
log("SUMMARY")
log("=" * 60)
log(f"  A (model.generate, no hook):          H={m_a['hallucination_rate']:.3f} C={m_a['correct_answer_rate']:.3f}")
log(f"  B (manual loop, no hook):             H={m_b['hallucination_rate']:.3f} C={m_b['correct_answer_rate']:.3f}")
log(f"  C (manual loop, do-nothing hook):     H={m_c['hallucination_rate']:.3f} C={m_c['correct_answer_rate']:.3f}")
log(f"  D (model.generate, do-nothing hook):  H={m_d['hallucination_rate']:.3f} C={m_d['correct_answer_rate']:.3f}")
log(f"  M3-v6 result (manual loop, real gate): H=0.800 C=0.733")

log("\n--- Answer-level diff A vs B (answerable only) ---")
diffs_ab = 0
for i, (ra, rb) in enumerate(zip(results_a, results_b)):
    if ra["answerability"] == "answerable":
        ca = ra["generated_output"].strip() == ra.get("correct_answer", "").strip()
        cb = rb["generated_output"].strip() == rb.get("correct_answer", "").strip()
        if ca != cb:
            diffs_ab += 1
            log(f"  [{i}] A correct={ca} B correct={cb}")
            log(f"       A: {ra['generated_output'][:120]}")
            log(f"       B: {rb['generated_output'][:120]}")
log(f"  Total answerable samples with different correctness A vs B: {diffs_ab}")

log("\n--- Answer-level diff A vs C (answerable only) ---")
diffs_ac = 0
for i, (ra, rc) in enumerate(zip(results_a, results_c)):
    if ra["answerability"] == "answerable":
        ca = ra["generated_output"].strip() == ra.get("correct_answer", "").strip()
        cc = rc["generated_output"].strip() == rc.get("correct_answer", "").strip()
        if ca != cc:
            diffs_ac += 1
            log(f"  [{i}] A correct={ca} C correct={cc}")

log(f"  Total answerable samples with different correctness A vs C: {diffs_ac}")

log("\nDone!")

with open(log_path, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
log(f"Output saved to {log_path}")