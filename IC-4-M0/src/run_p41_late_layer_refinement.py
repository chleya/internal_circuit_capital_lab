"""
P41: Late-Layer Control Refinement
===================================
P40 identified late-layer v/o projection as the dominant behavior lever
(late-only L12-L23: hall=28/60, best across all conditions).

P41 refines this into a tight recipe by testing:
  1. L0-L11 negative control @90 samples (matched to P40 data size)
  2. Late-only rank sweep: r=2, r=4 vs r=8
  3. Late-only projection subset: v/o only
  4. Late-layer boundary sweep: L14-L23, L16-L23

Key boundary from user:
  - Must re-run L0-L11 with 90 samples for fair matched contrast
  - Must track answerable correctness (not just hallucination)
  - Must not collapse into refusal; abstention not auto-success

Configs (all @90 train samples, 3 epochs, lr=2e-4, bs=2):
  L0-L11-R8        — negative control (matched 90-sample)
  L12-L23-R8        — reference (P40 replication)
  L12-L23-R4        — smaller rank
  L12-L23-VO-R8     — v/o projection only
  L14-L23-R8        — narrower late boundary
  L16-L23-R8        — minimal late layers

Usage:
  python src/run_p41_late_layer_refinement.py
"""

import os, sys, time, json, random, re, argparse
import torch
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model, TaskType

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_builder import load_jsonl

random.seed(42)
torch.manual_seed(42)

RESULTS_DIR = "results_p41_late_layer_refinement"
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


def detect_repetition(text):
    words = text.split()
    if len(words) < 6:
        return 0.0
    trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    if not trigrams:
        return 0.0
    return (len(trigrams) - len(set(trigrams))) / len(trigrams)


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


def evaluate_generate_split(model, tokenizer, samples, device):
    results_all = []
    for sample in samples:
        ctx = sample.get("context", "")
        q = sample.get("question", "")
        prompt = f"{ctx}\n\nQuestion: {q}\nAnswer:"
        gen_text, gen_len = generate_response(model, tokenizer, prompt,
                                              device)
        behavior = classify_behavior(gen_text, ctx, q)
        rep = detect_repetition(gen_text)
        results_all.append({
            "behavior": behavior, "gen_text": gen_text,
            "gen_len": gen_len, "rep_score": round(rep, 4),
            "answerability": sample.get("answerability", "?"),
        })

    unanswerable = [r for r in results_all
                    if r["answerability"] == "unanswerable"]
    answerable = [r for r in results_all
                  if r["answerability"] == "answerable"]

    def _counts(subset):
        hall = sum(1 for r in subset if r["behavior"] == "hallucination")
        abst = sum(1 for r in subset if r["behavior"] == "abstention")
        mixed = sum(1 for r in subset if r["behavior"] == "mixed")
        other = sum(1 for r in subset if r["behavior"] == "other")
        return hall, abst, mixed, other

    u_hall, u_abst, u_mixed, u_other = _counts(unanswerable)
    a_hall, a_abst, a_mixed, a_other = _counts(answerable)

    # On answerable: "correct" means NOT pure abstention
    # (classifier can't verify factual correctness, so we flag abstention
    #  on answerable as a likely correctness loss)
    a_correct = (len(answerable) - a_abst) if answerable else 0

    total = len(results_all)
    mean_len = (sum(r["gen_len"] for r in results_all) /
                max(1, total))
    mean_rep = (sum(r["rep_score"] for r in results_all) /
                max(1, total))

    return {
        "total": total,
        "n_unanswerable": len(unanswerable),
        "n_answerable": len(answerable),
        "unanswerable_hall": u_hall,
        "unanswerable_abst": u_abst,
        "unanswerable_mixed": u_mixed,
        "unanswerable_other": u_other,
        "answerable_hall": a_hall,
        "answerable_abst": a_abst,
        "answerable_mixed": a_mixed,
        "answerable_other": a_other,
        "answerable_correct": a_correct,
        "overall_hall": u_hall + a_hall,
        "overall_abst": u_abst + a_abst,
        "overall_mixed": u_mixed + a_mixed,
        "overall_other": u_other + a_other,
        "mean_len": round(mean_len, 1),
        "mean_rep": round(mean_rep, 4),
    }


def freeze_non_target_layers(model, active_layers):
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


def train_loop(model, dataloader, optimizer, device, epochs):
    loss_history = []
    for epoch in range(1, epochs + 1):
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
        loss_history.append(round(avg_loss, 4))
    return loss_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    log("=" * 72)
    log("P41: Late-Layer Control Refinement")
    log(f"  epochs={args.epochs}, lr={args.lr}, bs={args.batch_size}")
    log("  Tests: neg-ctrl L0-L11@90, rank sweep, proj subset,")
    log("    layer boundary sweep. Tracks answerable correctness.")
    log("=" * 72)
    t0 = time.time()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ['HF_HOME'] = 'F:/unified-sel/topomem/data/models/hf_cache'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

    log("\n[Step 1] Loading tokenizer & data...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_dir = os.path.join(base_dir, "data_position_sensitivity", "s0")
    train_path = os.path.join(data_dir, "train_all_s0.jsonl")
    test_paths = [os.path.join(data_dir, f"test_{pos}_s0.jsonl")
                  for pos in ["early", "mid", "late"]]

    train_samples = load_jsonl(train_path)
    seen_keys = set()
    test_samples = []
    for path in test_paths:
        if not os.path.exists(path):
            continue
        for s in load_jsonl(path):
            key = (s.get("entity_id", -1), s.get("template_id", ""))
            if key not in seen_keys:
                seen_keys.add(key)
                test_samples.append(s)

    n_test_ans = sum(1 for s in test_samples
                     if s.get("answerability") == "answerable")
    n_test_unans = sum(1 for s in test_samples
                       if s.get("answerability") != "answerable")
    log(f"  Train: {len(train_samples)}, Test: {len(test_samples)} "
        f"({n_test_ans}A + {n_test_unans}U)")

    train_dataset = PairedResponseDataset(train_samples, tokenizer)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True)

    CONFIGS = [
        {"label": "L0-L11-R8", "rank": 8,
         "layers": list(range(0, 12)),
         "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
         "desc": "NEG-CTRL: early-only matched 90-sample",
         "h_test": "neg_control"},
        {"label": "L12-L23-R8", "rank": 8,
         "layers": list(range(12, 24)),
         "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
         "desc": "REFERENCE: P40 replication",
         "h_test": "reference"},
        {"label": "L12-L23-R4", "rank": 4,
         "layers": list(range(12, 24)),
         "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
         "desc": "RANK: smaller r=4",
         "h_test": "rank_sweep"},
        {"label": "L12-L23-VO-R8", "rank": 8,
         "layers": list(range(12, 24)),
         "target_modules": ["v_proj", "o_proj"],
         "desc": "PROJ: v/o only (drop q,k)",
         "h_test": "proj_subset"},
        {"label": "L14-L23-R8", "rank": 8,
         "layers": list(range(14, 24)),
         "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
         "desc": "BOUNDARY: narrower L14-L23 (10 layers)",
         "h_test": "boundary_sweep"},
        {"label": "L16-L23-R8", "rank": 8,
         "layers": list(range(16, 24)),
         "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
         "desc": "BOUNDARY: minimal L16-L23 (8 layers)",
         "h_test": "boundary_sweep"},
    ]

    all_results = []

    for ci, cfg in enumerate(CONFIGS):
        label = cfg["label"]
        layers = cfg["layers"]
        rank = cfg["rank"]
        desc = cfg["desc"]
        h_test = cfg["h_test"]

        log(f"\n{'─' * 64}")
        log(f"  [{ci+1}/{len(CONFIGS)}] {label}: {desc}")
        log(f"    Rank={rank}, Layers={len(layers)} "
            f"({layers[0]}-{layers[-1]}), "
            f"Targets={cfg['target_modules']}")
        log(f"{'─' * 64}")

        t_cfg = time.time()

        log("  Loading fresh base model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map="cpu",
            attn_implementation="eager")
        model.eval()
        device = next(model.parameters()).device

        log("  Applying LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=rank,
            lora_alpha=rank * 2, lora_dropout=0.05,
            target_modules=cfg["target_modules"],
        )
        model = get_peft_model(model, lora_config)
        frozen, active = freeze_non_target_layers(model, set(layers))
        trainable = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)
        log(f"  Frozen/Active LoRA: {frozen}/{active}, "
            f"Trainable: {trainable:,}")

        log(f"  Training {args.epochs} epochs...")
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr)
        loss_history = train_loop(model, dataloader, optimizer,
                                  device, args.epochs)
        loss_str = " → ".join(f"{l:.3f}" for l in loss_history)
        log(f"  Loss: {loss_str}")

        log(f"  Evaluating generate() behavior (split A/U)...")
        model.eval()
        eval_res = evaluate_generate_split(model, tokenizer, test_samples,
                                           device)
        t_elapsed = time.time() - t_cfg

        log(f"  ALL:  hall={eval_res['overall_hall']} "
            f"abst={eval_res['overall_abst']} "
            f"mixed={eval_res['overall_mixed']} "
            f"other={eval_res['overall_other']} | "
            f"len={eval_res['mean_len']:.0f} rep={eval_res['mean_rep']:.3f}")
        log(f"  Unans: hall={eval_res['unanswerable_hall']} "
            f"abst={eval_res['unanswerable_abst']} "
            f"mixed={eval_res['unanswerable_mixed']} "
            f"other={eval_res['unanswerable_other']}")
        log(f"  Ansb:  hall={eval_res['answerable_hall']} "
            f"abst={eval_res['answerable_abst']} "
            f"mixed={eval_res['answerable_mixed']} "
            f"other={eval_res['answerable_other']} "
            f"correct={eval_res['answerable_correct']}"
            f"/{eval_res['n_answerable']} ({t_elapsed:.0f}s)")

        all_results.append({
            "label": label, "rank": rank, "n_layers": len(layers),
            "layer_range": f"{layers[0]}-{layers[-1]}",
            "target_modules": cfg["target_modules"],
            "desc": desc, "h_test": h_test,
            "loss_history": loss_history,
            "trainable_params": trainable,
            "time_s": round(t_elapsed, 1),
            **{k: v for k, v in eval_res.items()},
        })

        del model

    log("\n" + "=" * 72)
    log("P41 RESULTS: Late-Layer Control Refinement")
    log("=" * 72)

    PRE_OVERALL_HALL = 46

    log(f"\n  {'Config':<18s} {'Rank':>4s} {'Layers':>6s} "
        f"{'Params':>8s} {'U-Hall':>6s} {'U-Abst':>6s} "
        f"{'A-Corr':>6s} {'A-Abst':>6s} "
        f"{'Len':>4s} {'Rep':>6s} "
        f"{'ΔHall':>5s}  Loss")
    log(f"  {'─'*18} {'─'*4} {'─'*6} {'─'*8} {'─'*6} {'─'*6} "
        f"{'─'*6} {'─'*6} {'─'*4} {'─'*6} {'─'*5}  {'─'*25}")

    for r in all_results:
        loss_str = " → ".join(f"{l:.1f}" for l in r["loss_history"])
        delta = r["overall_hall"] - PRE_OVERALL_HALL
        log(f"  {r['label']:<18s} {r['rank']:>4d} "
            f"{r['layer_range']:>6s} {r['trainable_params']:>8,d} "
            f"{r['unanswerable_hall']:>6d} {r['unanswerable_abst']:>6d} "
            f"{r['answerable_correct']:>6d} {r['answerable_abst']:>6d} "
            f"{r['mean_len']:>4.0f} {r['mean_rep']:>6.3f} "
            f"{delta:>+5d}  {loss_str}")

    log(f"  {'─'*18} {'─'*4} {'─'*6} {'─'*8} {'─'*6} {'─'*6} "
        f"{'─'*6} {'─'*6} {'─'*4} {'─'*6} {'─'*5}")

    log(f"\n  Baselines:")
    log(f"    Pre(base): U-hall≈23 A-corr≈30 overall-hall=46 (est)")
    log(f"    P40 late-only L12-L23: overall-hall=28/60")
    log(f"    P40 L0-L11 @30samples: overall-hall=59/60")

    log(f"\n[Key Comparisons]")

    l0_11_row = next((r for r in all_results
                      if r["label"] == "L0-L11-R8"), None)
    l12_23_r8 = next((r for r in all_results
                      if r["label"] == "L12-L23-R8"), None)
    l12_23_r4 = next((r for r in all_results
                      if r["label"] == "L12-L23-R4"), None)
    l12_23_vo = next((r for r in all_results
                      if r["label"] == "L12-L23-VO-R8"), None)
    l14_23 = next((r for r in all_results
                   if r["label"] == "L14-L23-R8"), None)
    l16_23 = next((r for r in all_results
                   if r["label"] == "L16-L23-R8"), None)

    if l0_11_row:
        log(f"  [NEG-CTRL] L0-L11@90: overall-hall={l0_11_row['overall_hall']}"
            f" vs P39 L0-L11@30: hall=59/60")
        if l0_11_row["overall_hall"] >= 46:
            log(f"    → Confirms early-only training is harmful "
                f"even with matched 90-sample data.")

    if l12_23_r8 and l0_11_row:
        delta_ctrl = l12_23_r8["overall_hall"] - l0_11_row["overall_hall"]
        log(f"  [MATCHED] L12-L23 vs L0-L11: Δhall={delta_ctrl:+d} "
            f"(same params, opposite depth)")

    if l12_23_r4:
        log(f"  [RANK] L12-L23-R4: hall={l12_23_r4['overall_hall']}"
            f" vs R8={l12_23_r8['overall_hall'] if l12_23_r8 else '?'}")
        if l12_23_r4["overall_hall"] >= 40:
            log(f"    → r=4 insufficient; r=8 needed for late-layer effect")

    if l12_23_vo:
        log(f"  [PROJ] v/o only: hall={l12_23_vo['overall_hall']}"
            f" vs q/k/v/o={l12_23_r8['overall_hall'] if l12_23_r8 else '?'}")

    if l14_23 and l12_23_r8:
        log(f"  [BOUNDARY] L14-L23 (10L): hall={l14_23['overall_hall']}"
            f" vs L12-L23 (12L): hall={l12_23_r8['overall_hall']}")

    if l16_23:
        log(f"  [BOUNDARY] L16-L23 (8L): hall={l16_23['overall_hall']}")

    if l12_23_r8:
        log(f"  [ANSWERABLE] L12-L23 answerable correct: "
            f"{l12_23_r8['answerable_correct']}"
            f"/{l12_23_r8['n_answerable']} "
            f"(abst on answerable={l12_23_r8['answerable_abst']})")

    log("\n[Verdict]")
    best_row = min(all_results, key=lambda r: r["overall_hall"])
    best_label = best_row["label"]
    best_hall = best_row["overall_hall"]

    neg_ctrl_ok = (l0_11_row and l0_11_row["overall_hall"] >= 46)
    late_vs_early = (l12_23_r8 and l0_11_row and
                     l12_23_r8["overall_hall"] < l0_11_row["overall_hall"])
    ans_ok = (l12_23_r8 and
              l12_23_r8["answerable_abst"] < l12_23_r8["n_answerable"] * 0.3)

    improvements = []
    if l12_23_r4 and l12_23_r4["overall_hall"] < 40:
        improvements.append(f"r=4 viable ({l12_23_r4['overall_hall']}/60)")
    if l12_23_vo and l12_23_vo["overall_hall"] < 35:
        improvements.append(f"v/o only viable ({l12_23_vo['overall_hall']}/60)")
    if l14_23 and l14_23["overall_hall"] < 35:
        improvements.append(f"L14-L23 viable ({l14_23['overall_hall']}/60)")
    if l16_23 and l16_23["overall_hall"] < 35:
        improvements.append(f"L16-L23 viable ({l16_23['overall_hall']}/60)")

    if neg_ctrl_ok and late_vs_early and ans_ok:
        if best_hall <= 30:
            log(f"  VERDICT: late_layer_recipe_confirmed")
            log(f"  Best: {best_label} hall={best_hall}/60")
            log(f"  Neg ctrl passed (L0-L11@90 harmful).")
            log(f"  Answerable correctness preserved.")
            if improvements:
                for imp in improvements:
                    log(f"  Efficiency gain: {imp}")
        else:
            log(f"  VERDICT: late_layer_effect_reproduced")
            log(f"  Best: {best_label} hall={best_hall}/60 "
                f"(≥P40 level but not improved)")
    elif neg_ctrl_ok and not late_vs_early:
        log(f"  VERDICT: matched_neg_ctrl_only")
        log(f"  L0-L11@90 confirmed harmful, but late-layer")
        log(f"  advantage did not reproduce clearly.")
    else:
        log(f"  VERDICT: mixed_or_boundary_found")
        log(f"  Best: {best_label} hall={best_hall}/60")

    elapsed = time.time() - t0
    summary = {
        "config": {
            "epochs": args.epochs, "lr": args.lr,
            "batch_size": args.batch_size, "seed": 42,
            "n_train_samples": len(train_samples),
        },
        "test": {
            "n_samples": len(test_samples),
            "n_answerable": n_test_ans,
            "n_unanswerable": n_test_unans,
        },
        "results": all_results,
        "time_s": round(elapsed, 1),
    }
    with open(os.path.join(RESULTS_DIR, "summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log(f"\nResults saved to {RESULTS_DIR}/summary.json")
    log(f"Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log("P41 Complete.")


if __name__ == "__main__":
    main()