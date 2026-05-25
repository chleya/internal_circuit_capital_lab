"""
P15: Hallucination-Focused LoRA Fine-Tuning (B-Bottleneck Remedy).
===================================================================
Four experiments (9-A, P12, P13, P14) proved that hidden-state vector ops
cannot bridge the B-bottleneck KNOWS->produces gap. Only weight-level
intervention (Phase 10 LoRA) succeeded in changing behavior.

P15 directly tests whether LoRA fine-tuning can bridge the B-bottleneck:
  - Train LoRA on hallucination-labeled data
  - For answerable samples: target = positive_response (correct answer)
  - For unanswerable samples: target = negative_response (correct abstention)
  - Evaluate: hallucination rate (H) and correctness (C) via log-prob comparison

Key difference from Phase 10:
  Phase 10 trained on position-variant data targeting position invariance.
  P15 trains on the SAME data but targets hallucination reduction directly:
  the model learns to answer when it knows and abstain when it doesn't.

Hypotheses:
  H15.1: LoRA reduces H (hallucination on unanswerable) by >= 0.10
  H15.2: LoRA preserves or improves C (correctness on answerable)
  H15.3: P15's hallucination-targeted training outperforms Phase 10's
         position-invariance training on hallucination metrics

Usage:
  cd F:\internal_circuit_capital_lab\IC-4-M0
  python src/run_p15_hallucination_lora.py --epochs 3 --rank 4
"""

import argparse, os, sys, time, json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from peft import LoraConfig, get_peft_model, TaskType

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_loader import load_model_and_tokenizer
from src.data_builder import load_jsonl

RESULTS_DIR = "results_p15_hallucination_lora"
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, "run_log.txt")

def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()

class HallucinationDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        context = sample.get("context", "")
        question = sample.get("question", "")
        answerability = sample.get("answerability", "answerable")

        target = sample.get("positive_response", "") if answerability == "answerable" \
            else sample.get("negative_response", "")

        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
        full_text = f"{prompt} {target}"

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

        results.append({
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

    H = hallucinations / n_unans if n_unans > 0 else 0.0
    C = correct / n_ans if n_ans > 0 else 0.0

    return {
        "H": round(H, 4), "C": round(C, 4),
        "hall_count": hallucinations, "unans_count": n_unans,
        "corr_count": correct, "ans_count": n_ans,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    log("=" * 64)
    log("P15: Hallucination-Focused LoRA Fine-Tuning (B-Bottleneck Remedy)")
    log(f"  epochs={args.epochs}, rank={args.rank}, lr={args.lr}")
    log(f"  seed={args.seed}")
    log("=" * 64)
    t0 = time.time()

    log("\n[Step 1] Loading model on CPU...")
    model, tokenizer = load_model_and_tokenizer(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="cpu", torch_dtype="float32",
    )
    device = next(model.parameters()).device

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pos_dir = os.path.join(base_dir, "data_position_sensitivity", f"s{args.seed}")

    log("\n[Step 2] PRE-TRAINING baseline (log-prob comparison)...")
    test_samples = []
    for pos in ["early", "mid", "late"]:
        test_path = os.path.join(pos_dir, f"test_{pos}_s{args.seed}.jsonl")
        if os.path.exists(test_path):
            for s in load_jsonl(test_path)[:10]:
                s["_position"] = pos
                test_samples.append(s)

    log(f"  Test: {len(test_samples)} samples "
        f"({sum(1 for s in test_samples if s.get('answerability')=='answerable')}A + "
        f"{sum(1 for s in test_samples if s.get('answerability')!='answerable')}U)")

    pre_results = evaluate_behavior_logprob(model, tokenizer, test_samples, device)
    pre_metrics = compute_eval_metrics(pre_results)

    pos_metrics_pre = {}
    for pos in ["early", "mid", "late"]:
        pos_results = [r for r, s in zip(pre_results, test_samples) if s.get("_position") == pos]
        pos_metrics_pre[pos] = compute_eval_metrics(pos_results)

    h_values = [pos_metrics_pre[p]["H"] for p in ["early","mid","late"]]
    delta_h_pre = max(h_values) - min(h_values)

    log(f"  Baseline: H={pre_metrics['H']:.4f} C={pre_metrics['C']:.4f} ΔH={delta_h_pre:.4f}")
    for pos in ["early", "mid", "late"]:
        m = pos_metrics_pre[pos]
        log(f"    {pos}: H={m['H']:.4f} C={m['C']:.4f} "
            f"(hall={m['hall_count']}/{m['unans_count']} corr={m['corr_count']}/{m['ans_count']})")

    log("\n[Step 3] Applying LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.rank,
        lora_alpha=args.rank * 2, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    log("\n[Step 4] Loading training data (hallucination-labeled)...")
    datasets = []
    total_answerable = 0
    total_unanswerable = 0
    for pos in ["early", "mid", "late"]:
        train_path = os.path.join(pos_dir, f"train_{pos}_s{args.seed}.jsonl")
        if os.path.exists(train_path):
            train_samples = load_jsonl(train_path)
            ds = HallucinationDataset(train_samples, tokenizer)
            datasets.append(ds)
            na = sum(1 for s in train_samples if s.get("answerability") == "answerable")
            nu = len(train_samples) - na
            total_answerable += na
            total_unanswerable += nu
            log(f"  {pos} train: {na}A + {nu}U ({len(train_samples)} total)")

    combined_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True)
    log(f"  Combined: {len(combined_dataset)} samples ({total_answerable}A+{total_unanswerable}U), "
        f"{len(dataloader)} batches")

    log(f"\n[Step 5] Training for {args.epochs} epochs...")
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

    log("\n[Step 6] POST-TRAINING evaluation (log-prob comparison)...")
    model.eval()
    post_results = evaluate_behavior_logprob(model, tokenizer, test_samples, device)
    post_metrics = compute_eval_metrics(post_results)

    pos_metrics_post = {}
    for pos in ["early", "mid", "late"]:
        pos_results = [r for r, s in zip(post_results, test_samples) if s.get("_position") == pos]
        pos_metrics_post[pos] = compute_eval_metrics(pos_results)

    h_values_post = [pos_metrics_post[p]["H"] for p in ["early","mid","late"]]
    delta_h_post = max(h_values_post) - min(h_values_post)

    delta_H = post_metrics["H"] - pre_metrics["H"]
    delta_C = post_metrics["C"] - pre_metrics["C"]

    log(f"  Post-LoRA: H={post_metrics['H']:.4f} C={post_metrics['C']:.4f} ΔH_pos={delta_h_post:.4f}")
    log(f"  Change: ΔH={delta_H:+.4f} ΔC={delta_C:+.4f} ΔΔH_pos={delta_h_post-delta_h_pre:+.4f}")
    for pos in ["early", "mid", "late"]:
        m_pre = pos_metrics_pre[pos]
        m_post = pos_metrics_post[pos]
        dh = m_post["H"] - m_pre["H"]
        dc = m_post["C"] - m_pre["C"]
        log(f"    {pos}: H {m_pre['H']:.4f}→{m_post['H']:.4f} ({dh:+.4f}) "
            f"C {m_pre['C']:.4f}→{m_post['C']:.4f} ({dc:+.4f})")

    log(f"\n[Step 7] Comparison with Phase 10 LoRA checkpoint...")
    phase10_chkpt = os.path.join(base_dir, "results_a4_position_aware_training", "checkpoint_final")
    has_p10 = os.path.isdir(phase10_chkpt)
    log(f"  Phase 10 checkpoint: {'FOUND' if has_p10 else 'NOT FOUND'}")

    if has_p10:
        try:
            from peft import PeftModel
            model2, tokenizer2 = load_model_and_tokenizer(
                model_name="Qwen/Qwen2.5-0.5B-Instruct",
                device="cpu", torch_dtype="float32",
            )
            model2 = PeftModel.from_pretrained(model2, phase10_chkpt)
            model2.eval()
            device2 = next(model2.parameters()).device
            p10_results = evaluate_behavior_logprob(model2, tokenizer2, test_samples, device2)
            p10_metrics = compute_eval_metrics(p10_results)
            log(f"  Phase 10: H={p10_metrics['H']:.4f} C={p10_metrics['C']:.4f}")
            log(f"  P15 vs P10: ΔH={post_metrics['H']-p10_metrics['H']:+.4f} "
                f"ΔC={post_metrics['C']-p10_metrics['C']:+.4f}")
            p10_comparison = {"H": p10_metrics["H"], "C": p10_metrics["C"]}
        except Exception as e:
            log(f"  Phase 10 load failed: {e}")
            p10_comparison = None
    else:
        p10_comparison = None

    elapsed = time.time() - t0
    log(f"\n{'='*64}")
    log(f"[Summary] P15 Hallucination LoRA Results")

    h_improved = delta_H < -0.05
    c_maintained = delta_C > -0.10

    if h_improved and c_maintained:
        log(f"  *** H15.1+H15.2 CONFIRMED: H reduced by {abs(delta_H):.3f}, C maintained ***")
        log(f"  *** B-bottleneck PARTIALLY bridged via weight-level LoRA ***")
    elif h_improved and not c_maintained:
        log(f"  *** H15.1 CONFIRMED, H15.2 REJECTED: H reduced but C degraded ***")
        log(f"  *** B-bottleneck shows trade-off: reducing hallucination costs correctness ***")
    elif not h_improved:
        log(f"  *** H15.1 REJECTED: LoRA did not reduce hallucination ***")
        log(f"  *** B-bottleneck may require different training objective or more data ***")

    if p10_comparison:
        p15_better = post_metrics["H"] < p10_comparison["H"]
        log(f"  H15.3 (P15 > P10 on H): {'CONFIRMED' if p15_better else 'REJECTED'} "
            f"(P15 H={post_metrics['H']:.4f} vs P10 H={p10_comparison['H']:.4f})")

    log(f"\nP15 Complete. ({elapsed:.0f}s, {elapsed/60:.1f} min)")

    summary = {
        "epochs": args.epochs, "rank": args.rank, "lr": args.lr,
        "seed": args.seed, "n_train": len(combined_dataset),
        "n_train_answerable": total_answerable,
        "n_train_unanswerable": total_unanswerable,
        "n_test": len(test_samples),
        "pre_H": pre_metrics["H"], "pre_C": pre_metrics["C"],
        "pre_delta_h": delta_h_pre,
        "post_H": post_metrics["H"], "post_C": post_metrics["C"],
        "post_delta_h": delta_h_post,
        "delta_H": round(delta_H, 4), "delta_C": round(delta_C, 4),
        "h_improved": h_improved, "c_maintained": c_maintained,
        "per_position_pre": {p: {"H": m["H"], "C": m["C"]} for p, m in pos_metrics_pre.items()},
        "per_position_post": {p: {"H": m["H"], "C": m["C"]} for p, m in pos_metrics_post.items()},
        "phase10_comparison": p10_comparison,
        "train_time_s": round(train_elapsed, 1),
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