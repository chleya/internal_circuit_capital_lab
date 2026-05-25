"""
Phase 10: Position-Aware LoRA Training for Behavior-Level Absorption.
=====================================================================
Trains Qwen-0.5B on position-augmented data so that the same content
at different positions produces consistent behavior.

Key idea:
  Phase 8 showed probe-level absorption is FIXED (PSI -90%), but
  behavior-level position sensitivity (deltaH=0.111) persists.
  Phase 9-A showed global offset correction is ineffective.
  This means we need WEIGHT-LEVEL intervention.

Method:
  1. Load position sensitivity training data (same content at early/mid/late)
  2. Apply LoRA (rank=4)
  3. Train on all position variants (90 samples) with standard CE loss
  4. After training, run behavior test on position test data
  5. Measure: position consistency score, deltaC, deltaH, PSI

Behavior evaluation uses log-probability comparison (positive vs negative response)
instead of autoregressive generation for CPU efficiency (~24x faster).

Hypotheses:
  H10.1: Position-augmented training improves position consistency
  H10.2: Reduced PSI after training
  H10.3: Reduced behavior-level deltaC/deltaH after training

Usage:
  cd F:\internal_circuit_capital_lab\IC-4-M0
  python src/run_a4_position_aware_training.py --epochs 3
"""

import argparse, os, sys, time, json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from peft import LoraConfig, get_peft_model, TaskType

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_loader import load_model_and_tokenizer
from src.data_builder import load_jsonl
from src.run_m3_v6 import _train_probe
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def _collect_prefill_features_v2(model, tokenizer, samples, layer_idx, representation):
    device = next(model.parameters()).device
    X_list, y_list = [], []
    for sample in samples:
        context = sample.get("context", "")
        question = sample.get("question", "")
        label = sample.get("answerability", "?")
        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hs = outputs.hidden_states[layer_idx + 1][0]
        if representation == "last_prompt_token":
            pooled = hs[-1, :].detach().cpu().float().numpy()
        elif representation == "mean_pooled":
            pooled = hs.mean(dim=0).detach().cpu().float().numpy()
        else:
            pooled = hs[-1, :].detach().cpu().float().numpy()
        X_list.append(pooled)
        y_list.append(1 if label == "answerable" else 0)
    return np.stack(X_list, axis=0), np.array(y_list, dtype=np.int32)

RESULTS_DIR = "results_a4_position_aware_training"
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, "run_log.txt")

def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()

class PositionVariantDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=256, pos_label=""):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pos_label = pos_label

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        context = sample.get("context", "")
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
        full_text = f"{prompt} {answer}"

        inputs = self.tokenizer(full_text, truncation=True, max_length=self.max_length,
                                padding="max_length", return_tensors="pt")
        prompt_tokens = self.tokenizer(prompt, truncation=True, max_length=self.max_length,
                                       return_tensors="pt")

        labels = inputs["input_ids"].clone()
        prompt_len = prompt_tokens["input_ids"].shape[1]
        labels[0, :prompt_len] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }


def logprob_of_response(model, tokenizer, prompt, response, device):
    full_text = f"{prompt} {response}"
    full_ids = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=256)
    full_ids = {k: v.to(device) for k, v in full_ids.items()}
    full_len = full_ids["input_ids"].shape[1]

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

        pos_resp = sample.get("positive_response", "")
        neg_resp = sample.get("negative_response", "")

        pos_lp = logprob_of_response(model, tokenizer, prompt, pos_resp, device)
        neg_lp = logprob_of_response(model, tokenizer, prompt, neg_resp, device)

        pref_positive = pos_lp > neg_lp
        generated = pos_resp if pref_positive else neg_resp

        results.append({
            "generated_output": generated,
            "answerability": sample.get("answerability", "?"),
            "pref_positive": pref_positive,
            "pos_logprob": pos_lp,
            "neg_logprob": neg_lp,
        })

    return results


def compute_eval_metrics(eval_results):
    answerable = [r for r in eval_results if r.get("answerability") == "answerable"]
    unanswerable = [r for r in eval_results if r.get("answerability") == "unanswerable"]

    n_ans = len(answerable)
    n_unans = len(unanswerable)

    hallucinations = sum(1 for r in unanswerable if r["pref_positive"])
    correct = sum(1 for r in answerable if r["pref_positive"])
    calibrated_abstentions = sum(1 for r in unanswerable if not r["pref_positive"])
    unnecessary_abstentions = sum(1 for r in answerable if not r["pref_positive"])

    H = hallucinations / n_unans if n_unans > 0 else 0.0
    C = correct / n_ans if n_ans > 0 else 0.0

    return {
        "hallucination_rate": round(H, 4),
        "correct_answer_rate": round(C, 4),
        "calibrated_abstention_rate": round(calibrated_abstentions / n_unans if n_unans > 0 else 0, 4),
        "unnecessary_abstention_rate": round(unnecessary_abstentions / n_ans if n_ans > 0 else 0, 4),
        "hallucination_count": hallucinations,
        "correct_count": correct,
        "answerable_count": n_ans,
        "unanswerable_count": n_unans,
    }


def compute_psi_from_scores(scores_dict):
    positions = ["early", "mid", "late"]
    mean_abs_deltas = []
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            if positions[i] in scores_dict and positions[j] in scores_dict:
                delta = np.abs(scores_dict[positions[i]] - scores_dict[positions[j]]).mean()
                mean_abs_deltas.append(delta)
    return np.mean(mean_abs_deltas) if mean_abs_deltas else None


def compute_position_consistency(eval_results_dict):
    n_consistent = 0
    n_total = 0
    for i in range(len(eval_results_dict["early"])):
        pref_early = eval_results_dict["early"][i]["pref_positive"]
        if i < len(eval_results_dict.get("mid", [])) and i < len(eval_results_dict.get("late", [])):
            pref_mid = eval_results_dict["mid"][i]["pref_positive"]
            pref_late = eval_results_dict["late"][i]["pref_positive"]
            if pref_early == pref_mid == pref_late:
                n_consistent += 1
            n_total += 1
    return n_consistent / max(n_total, 1), n_consistent, n_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--consistency_weight", type=float, default=0.0,
                        help="Weight for hidden-state consistency loss (0=disable)")
    args = parser.parse_args()

    log("=" * 64)
    log("Phase 10: Position-Aware LoRA Training")
    log(f"  epochs={args.epochs}, rank={args.rank}, lr={args.lr}")
    log(f"  consistency_weight={args.consistency_weight}")
    log("=" * 64)
    t0 = time.time()

    log("\n[Step 1] Loading model on CPU...")
    model, tokenizer = load_model_and_tokenizer(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="cpu", torch_dtype="float32",
    )
    device = next(model.parameters()).device

    SEED, LAYER, REPR = 0, 12, "last_prompt_token"
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pos_dir = os.path.join(base_dir, "data_position_sensitivity", "s0")

    log(f"\n[Step 2] PRE-TRAINING behavior baseline (log-prob comparison)...")
    pos_test = {}
    for pos in ["early", "mid", "late"]:
        test_path = os.path.join(pos_dir, f"test_{pos}_s{SEED}.jsonl")
        if os.path.exists(test_path):
            pos_test[pos] = load_jsonl(test_path)[:20]

    pre_eval = {}
    log("  Computing log-prob preferences...")
    t_eval = time.time()
    for pos in ["early", "mid", "late"]:
        eval_results = evaluate_behavior_logprob(model, tokenizer, pos_test[pos], device)
        pre_eval[pos] = eval_results
        m = compute_eval_metrics(eval_results)
        log(f"  {pos}: H={m['hallucination_rate']:.4f} C={m['correct_answer_rate']:.4f} "
            f"(hall={m['hallucination_count']}/{m['unanswerable_count']} "
            f"corr={m['correct_count']}/{m['answerable_count']})")

    eval_time = time.time() - t_eval
    log(f"  Log-prob eval: {eval_time:.0f}s ({eval_time/60:.1f} min)")

    pre_consistency, pre_nc, pre_nt = compute_position_consistency(pre_eval)
    log(f"  Pre-training consistency: {pre_consistency:.4f} ({pre_nc}/{pre_nt})")

    log(f"\n[Step 3] Extracting PRE features + probe PSI baseline...")
    X_pre = {}
    for pos in ["early", "mid", "late"]:
        Xp, yp = _collect_prefill_features_v2(model, tokenizer, pos_test[pos], LAYER, REPR)
        X_pre[pos] = Xp

    pre_probe = _train_probe(np.vstack([X_pre[p] for p in ["early","mid","late"]]),
                             np.array([1 if s.get("answerability")=="answerable" else 0
                                       for p in ["early","mid","late"] for s in pos_test[p]]))
    pre_scores = {}
    for pos in ["early", "mid", "late"]:
        Xs = pre_probe["scaler"].transform(X_pre[pos])
        pre_scores[pos] = pre_probe["classifier"].predict_proba(Xs)[:, 1]
    pre_psi = compute_psi_from_scores(pre_scores)
    log(f"  Pre-training PSI: {pre_psi:.6f}")

    log(f"\n[Step 4] Applying LoRA (rank={args.rank})...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.rank,
        lora_alpha=args.rank * 2, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.config.output_hidden_states = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    log(f"\n[Step 5] Loading position training data...")
    datasets = []
    for pos in ["early", "mid", "late"]:
        train_path = os.path.join(pos_dir, f"train_{pos}_s{SEED}.jsonl")
        if os.path.exists(train_path):
            train_samples = load_jsonl(train_path)
            ds = PositionVariantDataset(train_samples, tokenizer, pos_label=pos)
            datasets.append(ds)
            na = sum(1 for s in train_samples if s.get("answerability")=="answerable")
            log(f"  {pos} train: {na}A+{len(train_samples)-na}U ({len(train_samples)} total)")

    combined_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True)
    log(f"  Combined: {len(combined_dataset)} samples, {len(dataloader)} batches")

    log(f"\n[Step 6] Training for {args.epochs} epochs...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    t_train = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        n_batches = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        log(f"  Epoch {epoch}: loss={avg_loss:.4f}, steps={n_batches}")

    train_elapsed = time.time() - t_train
    log(f"  Training: {train_elapsed:.0f}s ({train_elapsed/60:.1f} min)")

    log(f"\n[Step 7] POST-TRAINING behavior test (log-prob comparison)...")
    post_eval = {}
    model.eval()
    t_eval2 = time.time()
    for pos in ["early", "mid", "late"]:
        eval_results = evaluate_behavior_logprob(model, tokenizer, pos_test[pos], device)
        post_eval[pos] = eval_results
        m = compute_eval_metrics(eval_results)
        log(f"  {pos}: H={m['hallucination_rate']:.4f} C={m['correct_answer_rate']:.4f} "
            f"(hall={m['hallucination_count']}/{m['unanswerable_count']} "
            f"corr={m['correct_count']}/{m['answerable_count']})")

    eval_time2 = time.time() - t_eval2
    log(f"  Log-prob eval: {eval_time2:.0f}s")

    post_consistency, post_nc, post_nt = compute_position_consistency(post_eval)
    log(f"  Post-training consistency: {post_consistency:.4f} ({post_nc}/{post_nt})")
    if pre_consistency > 0 or post_consistency > 0:
        log(f"  Consistency change: {post_consistency - pre_consistency:+.4f}")

    log(f"\n[Step 8] POST-TRAINING feature extraction + PSI...")
    X_post = {}
    for pos in ["early", "mid", "late"]:
        Xp, yp = _collect_prefill_features_v2(model, tokenizer, pos_test[pos], LAYER, REPR)
        X_post[pos] = Xp

    post_probe = _train_probe(np.vstack([X_post[p] for p in ["early","mid","late"]]),
                              np.array([1 if s.get("answerability")=="answerable" else 0
                                        for p in ["early","mid","late"] for s in pos_test[p]]))
    post_scores = {}
    for pos in ["early", "mid", "late"]:
        Xs = post_probe["scaler"].transform(X_post[pos])
        post_scores[pos] = post_probe["classifier"].predict_proba(Xs)[:, 1]
    post_psi = compute_psi_from_scores(post_scores)
    log(f"  Post-training PSI: {post_psi:.6f}")
    if pre_psi and pre_psi > 0:
        log(f"  PSI improvement: {(pre_psi - post_psi)/pre_psi*100:+.1f}%")

    log(f"\n[Step 9] Cross-condition summary...")
    log(f"  {'Condition':20s} | {'early H':>8s} | {'mid H':>8s} | {'late H':>8s} | {'DeltaH':>6s} | {'Consistency':>12s} | {'PSI':>8s}")
    log(f"  {'-'*85}")

    def get_h(eval_dict):
        h = {}
        for pos in ["early", "mid", "late"]:
            m = compute_eval_metrics(eval_dict[pos])
            h[pos] = m["hallucination_rate"]
        return h

    pre_h = get_h(pre_eval)
    post_h = get_h(post_eval)

    pre_dh = max(pre_h.values()) - min(pre_h.values()) if pre_h else 0
    post_dh = max(post_h.values()) - min(post_h.values()) if post_h else 0

    log(f"  {'Pre-training':20s} | {pre_h['early']:8.4f} | {pre_h['mid']:8.4f} | {pre_h['late']:8.4f} | {pre_dh:.4f} | {pre_consistency:12.4f} | {pre_psi:.6f}")
    log(f"  {'Post-training':20s} | {post_h['early']:8.4f} | {post_h['mid']:8.4f} | {post_h['late']:8.4f} | {post_dh:.4f} | {post_consistency:12.4f} | {post_psi:.6f}")

    elapsed = time.time() - t0
    log(f"\n{'='*64}")
    log(f"Phase 10 Complete. ({elapsed:.0f}s, {elapsed/60:.1f} min)")

    summary = {
        "epochs": args.epochs,
        "rank": args.rank,
        "lr": args.lr,
        "consistency_weight": args.consistency_weight,
        "eval_method": "logprob_comparison",
        "pre_consistency": round(pre_consistency, 4),
        "post_consistency": round(post_consistency, 4),
        "pre_psi": float(pre_psi) if pre_psi else None,
        "post_psi": float(post_psi) if post_psi else None,
        "pre_delta_h": round(pre_dh, 4),
        "post_delta_h": round(post_dh, 4),
        "pre_h": {k: round(v, 4) for k, v in pre_h.items()},
        "post_h": {k: round(v, 4) for k, v in post_h.items()},
        "eval_time_s": round(eval_time, 1),
        "time_s": round(elapsed, 1),
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    ckpt_dir = os.path.join(RESULTS_DIR, "checkpoint_final")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    log(f"\nLoRA checkpoint saved to {ckpt_dir}")

    log("=" * 64)

if __name__ == "__main__":
    main()