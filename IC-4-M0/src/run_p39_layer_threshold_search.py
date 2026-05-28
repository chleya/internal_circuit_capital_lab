"""
P39: Layer Threshold Search — How many layers does behavior control need?
=========================================================================
P36-P38 established that L0-only intervention (activation OR weight) fails.
P15 established that all-layer LoRA succeeds (hall 34→46→34).

P39 asks: What is the MINIMUM layer range that changes generate() behavior?

Design:
  Train LoRA on 4 progressively larger layer ranges:
    L0 only (from P38, hall=43/60)
    L0-L2   (first 3 layers)
    L0-L5   (first 6 layers)
    L0-L11  (first half)
  Compare to P15 (all 24 layers, hall=34/60)

All configurations use:
  - Same 30 training samples, 3 epochs, r=8, lr=2e-4
  - Same test set (60 samples)
  - Same hyperparams as P38

Usage:
  python src/run_p39_layer_threshold_search.py
"""

import os, sys, time, json, random, re, argparse
import torch
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_builder import load_jsonl

random.seed(42)
torch.manual_seed(42)

RESULTS_DIR = "results_p39_layer_threshold"
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
        return "abstention"
    if strong_hit and new_numbers:
        return "mixed"
    if strong_hit:
        return "abstention"
    if has_numeric:
        return "hallucination"
    if len(text_clean.split()) < 5:
        return "other"
    return "other"


def generate_response(model, tokenizer, prompt, device, max_new=64):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                    max_length=512)
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids, max_new_tokens=max_new,
            do_sample=False, pad_token_id=tokenizer.pad_token_id,
        )
    gen_ids = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True), len(gen_ids)


def evaluate_generate(model, tokenizer, samples, device):
    results = []
    for sample in samples:
        ctx = sample.get("context", "")
        q = sample.get("question", "")
        prompt = f"{ctx}\n\nQuestion: {q}\nAnswer:"
        gen_text, gen_len = generate_response(model, tokenizer, prompt,
                                              device)
        behavior = classify_behavior(gen_text, ctx, q)
        results.append({
            "behavior": behavior, "gen_text": gen_text, "gen_len": gen_len,
        })
    return results


def freeze_non_range_lora(model, active_layers):
    """
    Freeze all LoRA params except those in `active_layers`.
    active_layers: set of layer indices (e.g. {0, 1, 2})
    """
    frozen_count = 0
    active_count = 0
    for name, param in model.named_parameters():
        if "lora" not in name:
            continue
        is_active = False
        for layer_idx in active_layers:
            if f"layers.{layer_idx}." in name:
                is_active = True
                break
        if is_active:
            param.requires_grad = True
            active_count += 1
        else:
            param.requires_grad = False
            frozen_count += 1
    return frozen_count, active_count


def train_one_epoch(model, dataloader, optimizer, device):
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
    return total_loss / max(n_batches, 1), n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    log("=" * 72)
    log("P39: Layer Threshold Search — Minimum layers for behavior control")
    log(f"  epochs={args.epochs}, rank={args.rank}, lr={args.lr}")
    log("=" * 72)
    t0 = time.time()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ['HF_HOME'] = 'F:/unified-sel/topomem/data/models/hf_cache'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

    log("\n[Step 1] Loading model & data...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
                if pos_label == "train":
                    train_samples.append(s)
                else:
                    test_samples.append(s)

    n_test_unans = sum(1 for s in test_samples
                       if s.get("answerability") != "answerable")
    log(f"  Train: {len(train_samples)}, Test: {len(test_samples)} "
        f"({n_test_unans}U)")

    train_dataset = PairedResponseDataset(train_samples, tokenizer)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True)

    layer_ranges = [
        {"label": "L0",      "layers": [0],        "desc": "L0 only (P38)"},
        {"label": "L0-L2",   "layers": [0, 1, 2],  "desc": "first 3"},
        {"label": "L0-L5",   "layers": list(range(0, 6)), "desc": "first 6"},
        {"label": "L0-L11",  "layers": list(range(0, 12)), "desc": "first half"},
    ]

    all_results = []

    for ri, lr_config in enumerate(layer_ranges):
        label = lr_config["label"]
        active_set = set(lr_config["layers"])
        n_active = len(lr_config["layers"])
        desc = lr_config["desc"]

        log(f"\n{'─' * 64}")
        log(f"  [{ri+1}/{len(layer_ranges)}] Range: {label} "
            f"({n_active} layers) — {desc}")
        log(f"{'─' * 64}")

        t_range = time.time()

        log("  Loading fresh base model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map="cpu",
            attn_implementation="eager")
        model.eval()
        device = next(model.parameters()).device

        log("  Applying LoRA + freezing non-target layers...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=args.rank,
            lora_alpha=args.rank * 2, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_config)
        frozen, active = freeze_non_range_lora(model, active_set)
        trainable = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)
        log(f"  Frozen/Active LoRA: {frozen}/{active}, "
            f"Trainable: {trainable:,}")

        log(f"  Training {args.epochs} epochs...")
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr)
        loss_history = []
        for epoch in range(1, args.epochs + 1):
            avg_loss, n_batches = train_one_epoch(
                model, dataloader, optimizer, device)
            loss_history.append(round(avg_loss, 4))
            log(f"    Epoch {epoch}: loss={avg_loss:.4f}")

        log(f"  Evaluating generate() behavior...")
        model.eval()
        gen_results = evaluate_generate(model, tokenizer, test_samples,
                                        device)
        behaviors = [r["behavior"] for r in gen_results]
        hall = sum(1 for b in behaviors if b == "hallucination")
        abst = sum(1 for b in behaviors if b == "abstention")
        mixed = sum(1 for b in behaviors if b == "mixed")
        other = sum(1 for b in behaviors if b == "other")

        elapsed = time.time() - t_range
        log(f"  Result: hall={hall}/{len(test_samples)} "
            f"abst={abst} mixed={mixed} other={other} ({elapsed:.0f}s)")

        all_results.append({
            "label": label, "n_layers": n_active,
            "loss_history": loss_history,
            "hall": hall, "abst": abst, "mixed": mixed, "other": other,
            "total": len(test_samples),
            "hall_rate": round(hall / len(test_samples), 4),
            "abst_rate": round(abst / len(test_samples), 4),
            "trainable_params": trainable,
            "time_s": round(elapsed, 1),
        })

        del model

    log("\n" + "=" * 72)
    log("P39 RESULTS: Layer threshold sweep")
    log("=" * 72)

    log(f"\n  {'Range':<10s} {'Layers':>6s} {'Trainable':>10s} "
        f"{'Hall':>6s} {'Abst':>6s} {'Mixed':>6s} {'Other':>6s} "
        f"{'Hall%':>8s}  Loss curve")
    log(f"  {'─'*10} {'─'*6} {'─'*10} {'─'*6} {'─'*6} {'─'*6} "
        f"{'─'*6} {'─'*8}  {'─'*20}")

    for r in all_results:
        loss_str = " → ".join(f"{l:.1f}" for l in r["loss_history"])
        log(f"  {r['label']:<10s} {r['n_layers']:>6d} "
            f"{r['trainable_params']:>10,d} "
            f"{r['hall']:>5d}  {r['abst']:>5d}  {r['mixed']:>5d}  "
            f"{r['other']:>5d}  {r['hall_rate']:>7.1%}  {loss_str}")

    log("\n[Step 3] Collecting baselines...")
    pre_hall = 46
    pre_abst = 1
    p15_hall = 34
    p15_abst = 7
    p38_hall = 43
    p38_abst = 2

    log(f"  Pre (base): hall={pre_hall}/60 abst={pre_abst}/60")
    log(f"  P38 (L0 LoRA): hall={p38_hall}/60 abst={p38_abst}/60")
    log(f"  P15 (All-LoRA): hall={p15_hall}/60 abst={p15_abst}/60")

    log("\n[Threshold Analysis]")
    all_points = [
        ("Pre", 0, pre_hall, pre_abst),
        ("L0", 1, p38_hall, p38_abst),
    ]
    for r in all_results:
        all_points.append((r["label"], r["n_layers"], r["hall"],
                           r["abst"]))
    all_points.append(("P15(All)", 24, p15_hall, p15_abst))

    all_points.sort(key=lambda x: x[1])

    log(f"\n  {'Config':<12s} {'Layers':>6s} {'Hall':>6s} {'Abst':>6s} "
        f"{'ΔHall vs Pre':>12s} {'ΔHall vs L0':>12s}")
    log(f"  {'─'*12} {'─'*6} {'─'*6} {'─'*6} {'─'*12} {'─'*12}")

    for name, nl, h, a in all_points:
        dh_pre = h - pre_hall
        dh_l0 = h - p38_hall
        log(f"  {name:<12s} {nl:>6d} {h:>6d} {a:>6d} "
            f"{dh_pre:>+12d} {dh_l0:>+12d}")

    log("\n[Verdict]")
    threshold_found = False
    threshold_label = None
    for r in all_results:
        if r["hall"] < pre_hall - 3:
            threshold_found = True
            threshold_label = r["label"]
            break

    if threshold_found:
        log(f"  Threshold found at {threshold_label}: "
            f"hall drops below baseline")
        log("  VERDICT: threshold_identified")
    else:
        log("  No clear threshold within tested ranges")
        log("  VERDICT: negative_but_informative")
        log("  Even L0-L11 (12 layers, half the model) insufficient. "
            "Behavior control requires >12 layers or full-model training.")

    summary = {
        "config": {
            "epochs": args.epochs, "rank": args.rank, "lr": args.lr,
            "batch_size": args.batch_size, "seed": 42,
        },
        "train": {"n_samples": len(train_samples)},
        "test": {"n_samples": len(test_samples)},
        "baselines": {
            "pre": {"hall": pre_hall, "abst": pre_abst},
            "p15_all_layer": {"hall": p15_hall, "abst": p15_abst},
            "p38_l0_only": {"hall": p38_hall, "abst": p38_abst},
        },
        "results": all_results,
        "time_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(RESULTS_DIR, "summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log(f"\nResults saved to {RESULTS_DIR}/summary.json")
    log(f"Time: {(time.time()-t0):.0f}s "
        f"({(time.time()-t0)/60:.1f} min)")
    log("P39 Complete.")


if __name__ == "__main__":
    main()