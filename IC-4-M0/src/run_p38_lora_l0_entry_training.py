"""
P38: LoRA on L0 Entry Point — Weight-level training at the causal entry site
=============================================================================
P36/P37 proved inference-time L0 intervention (ablation + counter-vector)
cannot change model.generate() behavior. P15 proved all-layer LoRA can
bridge the B-bottleneck (H=0.0) on log-prob comparison.

P38 asks: Can LoRA trained ONLY on L0's attention projections change
model.generate() behavior where counter-vector injection failed?

Design:
  1. Apply LoRA (q/v/k/o_proj) to L0 ONLY (freeze all other layers' LoRA)
  2. Train on paired hallucination/abstention data
  3. Evaluate with model.generate() behavioral test (primary)
  4. Also evaluate with log-prob comparison (secondary, for P15 comparison)

Comparisons:
  - Baseline (no LoRA)
  - P38 L0-LoRA (this experiment)
  - P15 all-layer LoRA (if checkpoint available)

Key question: Weight-level L0 change vs activation-level L0 change —
  does the intervene-at-entry principle only work at weight level?

Usage:
  python src/run_p38_lora_l0_entry_training.py
"""

import os, sys, time, json, random, re, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_builder import load_jsonl

random.seed(42)
torch.manual_seed(42)

RESULTS_DIR = "results_p38_lora_l0_entry"
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, "run_log.txt")

def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


class PairedResponseDataset(Dataset):
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

        target = (sample.get("positive_response", "")
                  if answerability == "answerable"
                  else sample.get("negative_response", ""))

        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
        full_text = f"{prompt} {target}"

        inputs = self.tokenizer(full_text, truncation=True,
                                max_length=self.max_length,
                                padding="max_length", return_tensors="pt")
        prompt_tokens = self.tokenizer(prompt, truncation=True,
                                       max_length=self.max_length,
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
    full_ids = tokenizer(full_text, return_tensors="pt",
                         truncation=True, max_length=256)
    full_ids = {k: v.to(device) for k, v in full_ids.items()}

    prompt_ids = tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=256)
    prompt_len = prompt_ids["input_ids"].shape[1]

    labels = full_ids["input_ids"].clone()
    labels[0, :prompt_len] = -100

    with torch.no_grad():
        outputs = model(**full_ids, labels=labels)

    return -outputs.loss.item()


def evaluate_logprob(model, tokenizer, samples, device):
    results = []
    for sample in samples:
        ctx = sample.get("context", "")
        q = sample.get("question", "")
        prompt = f"{ctx}\n\nQuestion: {q}\nAnswer:"

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


def compute_lp_metrics(eval_results):
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


def classify_behavior(text, context, question):
    text_clean = text.strip()
    text_lower = text_clean.lower()

    abstain_markers = [
        "not provided", "not mentioned", "not stated", "not available",
        "no information", "cannot be determined", "can't be determined",
        "does not provide", "does not mention", "do not have", "don't have",
        "unclear", "insufficient information", "the passage does not",
        "the text does not", "not disclosed", "cannot answer", "can't answer",
        "unknown", "n/a", "not specified", "unspecified",
    ]
    strong_hit = [m for m in abstain_markers if m in text_lower]

    ctx_numbers = set(re.findall(r'\d[\d,.]*', context))
    text_numbers = set(re.findall(r'\d[\d,.]*', text_clean))
    new_numbers = text_numbers - ctx_numbers
    has_numeric = any(re.search(r'\d', t) for t in text_clean.split()[:30])

    if strong_hit and not new_numbers:
        return "abstention", f"markers: {', '.join(strong_hit[:3])}"
    if strong_hit and new_numbers:
        return "mixed", (f"abst_form({strong_hit[0]}) "
                         f"+ num({', '.join(list(new_numbers)[:2])})")
    if strong_hit:
        return "abstention", f"markers: {', '.join(strong_hit[:3])}"
    if has_numeric:
        return "hallucination", (f"numeric: "
                                 f"{', '.join(list(text_numbers)[:3])}")
    if len(text_clean.split()) < 5:
        return "other", "too_short"
    return "other", "no_pattern"


def detect_repetition(text):
    words = text.split()
    if len(words) < 6:
        return 0.0
    trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    if not trigrams:
        return 0.0
    return (len(trigrams) - len(set(trigrams))) / len(trigrams)


def generate_response(model, tokenizer, prompt, device, max_new=64):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_ids = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True), len(gen_ids)


def evaluate_generate(model, tokenizer, samples, device):
    results = []
    for sample in samples:
        ctx = sample.get("context", "")
        q = sample.get("question", "")
        prompt = f"{ctx}\n\nQuestion: {q}\nAnswer:"
        gen_text, gen_len = generate_response(model, tokenizer, prompt, device)
        behavior, reason = classify_behavior(gen_text, ctx, q)
        rep = detect_repetition(gen_text)
        results.append({
            "behavior": behavior, "reason": reason,
            "gen_text": gen_text, "gen_len": gen_len,
            "rep_score": round(rep, 4),
        })
    return results


def freeze_non_l0_lora(model):
    frozen_count = 0
    active_count = 0
    for name, param in model.named_parameters():
        if "lora" in name:
            if "layers.0." in name:
                param.requires_grad = True
                active_count += 1
            else:
                param.requires_grad = False
                frozen_count += 1
    return frozen_count, active_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    log("=" * 72)
    log("P38: LoRA on L0 Entry Point — Weight-level L0 training")
    log(f"  epochs={args.epochs}, rank={args.rank}, lr={args.lr}, bs={args.batch_size}")
    log("=" * 72)
    t0 = time.time()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ['HF_HOME'] = 'F:/unified-sel/topomem/data/models/hf_cache'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

    log("\n[Step 1] Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="cpu",
        attn_implementation="eager")
    model.eval()
    device = next(model.parameters()).device
    n_layers = len(model.model.layers)
    log(f"  Qwen2.5-0.5B, {n_layers} layers, device={device}")

    log("\n[Step 2] Loading data...")
    data_dir = os.path.join(base_dir, "data_position_sensitivity", "s0")
    data_files = [("train_all_s0.jsonl", "train")]
    for pos in ["early", "mid", "late"]:
        path = os.path.join(data_dir, f"test_{pos}_s0.jsonl")
        if os.path.exists(path):
            data_files.append((f"test_{pos}_s0.jsonl", pos))

    seen_keys = set()
    train_samples = []
    test_samples = []
    for fname, pos_label in data_files:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            continue
        for s in load_jsonl(path):
            key = (s.get("entity_id", -1), s.get("template_id", ""))
            if key not in seen_keys:
                seen_keys.add(key)
                s["_position"] = pos_label
                if pos_label == "train":
                    train_samples.append(s)
                else:
                    test_samples.append(s)

    n_train_ans = sum(1 for s in train_samples if s.get("answerability") == "answerable")
    n_train_unans = sum(1 for s in train_samples if s.get("answerability") != "answerable")
    n_test_ans = sum(1 for s in test_samples if s.get("answerability") == "answerable")
    n_test_unans = sum(1 for s in test_samples if s.get("answerability") != "answerable")
    log(f"  Train: {len(train_samples)} ({n_train_ans}A + {n_train_unans}U)")
    log(f"  Test:  {len(test_samples)} ({n_test_ans}A + {n_test_unans}U)")

    log("\n[Step 3] PRE-TRAINING baseline evaluation...")

    log("  [3a] Log-prob comparison baseline...")
    t_lp = time.time()
    pre_lp_results = evaluate_logprob(model, tokenizer, test_samples, device)
    pre_lp = compute_lp_metrics(pre_lp_results)
    log(f"    Pre H={pre_lp['H']:.4f} C={pre_lp['C']:.4f} "
        f"({time.time()-t_lp:.0f}s)")

    log("  [3b] Generate() behavioral baseline...")
    t_gen = time.time()
    pre_gen_results = evaluate_generate(model, tokenizer, test_samples, device)
    pre_behaviors = [r["behavior"] for r in pre_gen_results]
    pre_hall = sum(1 for b in pre_behaviors if b == "hallucination")
    pre_abst = sum(1 for b in pre_behaviors if b == "abstention")
    pre_mixed = sum(1 for b in pre_behaviors if b == "mixed")
    pre_other = sum(1 for b in pre_behaviors if b == "other")
    pre_n = len(pre_gen_results)
    log(f"    Pre generate: hall={pre_hall}/{pre_n} abst={pre_abst}/{pre_n} "
        f"mixed={pre_mixed}/{pre_n} other={pre_other}/{pre_n} "
        f"({time.time()-t_gen:.0f}s)")

    log("\n[Step 4] Applying LoRA (all layers, then freeze non-L0)...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    frozen, active = freeze_non_l0_lora(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  Frozen LoRA params (non-L0): {frozen}")
    log(f"  Active LoRA params (L0 only): {active}")
    log(f"  Trainable/Total: {trainable_params:,} / {total_params:,} "
        f"({trainable_params/total_params*100:.2f}%)")

    log("\n[Step 5] Building training dataset...")
    train_dataset = PairedResponseDataset(train_samples, tokenizer)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True)
    log(f"  {len(train_dataset)} samples, {len(dataloader)} batches")

    log(f"\n[Step 6] Training for {args.epochs} epochs...")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    t_train = time.time()
    loss_history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        n_batches = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        loss_history.append(avg_loss)
        log(f"  Epoch {epoch}: loss={avg_loss:.4f}, steps={n_batches}")

    train_elapsed = time.time() - t_train
    log(f"  Training: {train_elapsed:.0f}s ({train_elapsed/60:.1f} min)")

    log("\n[Step 7] POST-TRAINING evaluation...")
    model.eval()

    log("  [7a] Log-prob comparison post-LoRA...")
    t_lp2 = time.time()
    post_lp_results = evaluate_logprob(model, tokenizer, test_samples, device)
    post_lp = compute_lp_metrics(post_lp_results)
    delta_H_lp = post_lp["H"] - pre_lp["H"]
    delta_C_lp = post_lp["C"] - pre_lp["C"]
    log(f"    Post H={post_lp['H']:.4f} C={post_lp['C']:.4f} "
        f"ΔH={delta_H_lp:+.4f} ΔC={delta_C_lp:+.4f} "
        f"({time.time()-t_lp2:.0f}s)")

    log("  [7b] Generate() behavioral post-LoRA...")
    t_gen2 = time.time()
    post_gen_results = evaluate_generate(model, tokenizer, test_samples, device)
    post_behaviors = [r["behavior"] for r in post_gen_results]
    post_hall = sum(1 for b in post_behaviors if b == "hallucination")
    post_abst = sum(1 for b in post_behaviors if b == "abstention")
    post_mixed = sum(1 for b in post_behaviors if b == "mixed")
    post_other = sum(1 for b in post_behaviors if b == "other")
    post_n = len(post_gen_results)
    log(f"    Post generate: hall={post_hall}/{post_n} abst={post_abst}/{post_n} "
        f"mixed={post_mixed}/{post_n} other={post_other}/{post_n} "
        f"({time.time()-t_gen2:.0f}s)")

    delta_hall = post_hall - pre_hall
    delta_abst = post_abst - pre_abst
    log(f"    Δhall={delta_hall:+d} Δabst={delta_abst:+d}")

    log("\n  [7c] Per-sample behavior changes (generate)...")
    rank = {"abstention": 0, "other": 1, "mixed": 2, "hallucination": 3}
    n_better = n_worse = n_same = 0
    for pre_r, post_r in zip(pre_gen_results, post_gen_results):
        b_pre = pre_r["behavior"]
        b_post = post_r["behavior"]
        r_pre = rank.get(b_pre, 1)
        r_post = rank.get(b_post, 1)
        if r_post < r_pre:
            n_better += 1
        elif r_post > r_pre:
            n_worse += 1
        else:
            n_same += 1
    log(f"    Paired: better={n_better} worse={n_worse} same={n_same} "
        f"({n_better/max(1,post_n)*100:.0f}% better)")

    log("\n[Step 8] Comparison with P15 all-layer LoRA (if available)...")
    p15_chkpt = os.path.join(base_dir, "results_p15_hallucination_lora",
                             "checkpoint_final")
    p15_comparison = None
    if os.path.isdir(p15_chkpt):
        log(f"  P15 checkpoint FOUND, loading for comparison...")
        try:
            model2, tokenizer2 = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float32, device_map="cpu",
                attn_implementation="eager"), tokenizer
            model2 = PeftModel.from_pretrained(model2, p15_chkpt)
            model2.eval()
            device2 = next(model2.parameters()).device
            t_p15 = time.time()
            p15_lp_results = evaluate_logprob(model2, tokenizer2,
                                              test_samples, device2)
            p15_lp = compute_lp_metrics(p15_lp_results)
            log(f"    P15 LP: H={p15_lp['H']:.4f} C={p15_lp['C']:.4f} "
                f"({time.time()-t_p15:.0f}s)")
            t_p15gen = time.time()
            p15_gen_results = evaluate_generate(model2, tokenizer2,
                                                test_samples, device2)
            p15_behaviors = [r["behavior"] for r in p15_gen_results]
            p15_hall = sum(1 for b in p15_behaviors if b == "hallucination")
            p15_abst = sum(1 for b in p15_behaviors if b == "abstention")
            p15_mixed = sum(1 for b in p15_behaviors if b == "mixed")
            log(f"    P15 generate: hall={p15_hall}/{post_n} "
                f"abst={p15_abst}/{post_n} mixed={p15_mixed}/{post_n} "
                f"({time.time()-t_p15gen:.0f}s)")
            p15_comparison = {
                "lp_H": p15_lp["H"], "lp_C": p15_lp["C"],
                "gen_hall": p15_hall, "gen_abst": p15_abst,
            }
        except Exception as e:
            log(f"  P15 load failed: {e}")
    else:
        log("  P15 checkpoint NOT FOUND, skipping comparison")

    elapsed = time.time() - t0
    log("\n" + "=" * 72)
    log("P38 判决")
    log("=" * 72)

    lp_success = delta_H_lp < -0.05
    gen_success = delta_hall < 0 and delta_abst >= 0

    log(f"  Log-prob comparison: H {pre_lp['H']:.4f}→{post_lp['H']:.4f} "
        f"(Δ{delta_H_lp:+.4f}) {'✓' if lp_success else '✗'}")
    log(f"  Generate behavior: hall {pre_hall}→{post_hall} "
        f"abst {pre_abst}→{post_abst} "
        f"{'✓' if gen_success else '✗'}")

    if lp_success and gen_success:
        log("\n  VERDICT: breakthrough")
        log("  L0-only LoRA succeeds where inference-time L0 intervention "
            "failed. Weight-level change at entry point bridges the "
            "log-prob → behavior gap.")
    elif lp_success and not gen_success:
        log("\n  VERDICT: partial_lp_only")
        log("  L0-only LoRA changes log-prob preference but NOT "
            "generate() behavior. Same log-prob → behavior gap persists "
            "even with weight-level entry intervention.")
        log("  This suggests the gap is not about intervention strength "
            "but about architectural depth — behavior depends on "
            "multi-layer interactions that L0 alone cannot redirect.")
    elif not lp_success and gen_success:
        log("\n  VERDICT: unexpected_behavior_only")
        log("  Generate behavior changed without log-prob change. "
            "Possible classifier artifact or sampling effect.")
    else:
        log("\n  VERDICT: negative_but_informative")
        log("  L0-only LoRA fails on both metrics. Confirms that "
            "the entry point alone, even with weight-level training, "
            "is insufficient for behavioral control.")
        log("  Recommendation: all-layer LoRA (P15) or multi-layer "
            "training is needed for behavior change.")

    ckpt_dir = os.path.join(RESULTS_DIR, "checkpoint_final")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)

    audit_path = os.path.join(RESULTS_DIR, "per_sample_audit.jsonl")
    with open(audit_path, "w", encoding="utf-8") as f:
        for i, (pre_r, post_r) in enumerate(
                zip(pre_gen_results, post_gen_results)):
            entry = {
                "idx": i,
                "pre_behavior": pre_r["behavior"],
                "post_behavior": post_r["behavior"],
                "pre_text": pre_r["gen_text"],
                "post_text": post_r["gen_text"],
                "pre_len": pre_r["gen_len"],
                "post_len": post_r["gen_len"],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    summary = {
        "config": {
            "epochs": args.epochs, "rank": args.rank, "lr": args.lr,
            "batch_size": args.batch_size,
            "n_layers": n_layers, "seed": 42,
        },
        "train": {
            "n_samples": len(train_samples),
            "n_answerable": n_train_ans,
            "n_unanswerable": n_train_unans,
            "loss_history": [round(l, 4) for l in loss_history],
            "time_s": round(train_elapsed, 1),
        },
        "test": {
            "n_samples": len(test_samples),
            "n_answerable": n_test_ans,
            "n_unanswerable": n_test_unans,
        },
        "logprob": {
            "pre_H": pre_lp["H"], "pre_C": pre_lp["C"],
            "post_H": post_lp["H"], "post_C": post_lp["C"],
            "delta_H": round(delta_H_lp, 4),
            "delta_C": round(delta_C_lp, 4),
        },
        "generate": {
            "pre_hall": pre_hall, "pre_abst": pre_abst,
            "pre_mixed": pre_mixed, "pre_other": pre_other,
            "pre_total": pre_n,
            "post_hall": post_hall, "post_abst": post_abst,
            "post_mixed": post_mixed, "post_other": post_other,
            "post_total": post_n,
            "delta_hall": delta_hall, "delta_abst": delta_abst,
            "paired_better": n_better, "paired_worse": n_worse,
            "paired_same": n_same,
        },
        "p15_comparison": p15_comparison,
        "time_s": round(elapsed, 1),
        "time_min": round(elapsed / 60, 1),
    }
    with open(os.path.join(RESULTS_DIR, "summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log(f"\nResults saved to {RESULTS_DIR}/")
    log(f"  - summary.json")
    log(f"  - per_sample_audit.jsonl ({len(post_gen_results)} entries)")
    log(f"  - checkpoint_final/")
    log(f"\nTime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log(f"P38 Complete.")


if __name__ == "__main__":
    main()