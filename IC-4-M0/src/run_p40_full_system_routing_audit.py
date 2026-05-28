"""
P40: Full-System Routing Training Audit
========================================
P39 found that contiguous early-layer LoRA produces a "valley of despair":
L0→L2→L5→L11 all worsened hallucination (43→50→49→59/60), while
P15 all-layer LoRA improved it (34/60).

P40 asks WHY by testing 4 competing hypotheses via a 2x2 design
(depth coverage × parameter count):

  Hypothesis 1: full-depth coordination
    → sparse-depth {0,6,12,18,23} or all-small-rank r=2 should work

  Hypothesis 2: parameter count
    → hallucination should correlate with trainable param count

  Hypothesis 3: late decision layers
    → late-only L12-L23 should work better than early-only L0-L11

  Hypothesis 4: objective mismatch
    → all CE-based should show similar (bad) pattern; DPO needed

Configurations (current train_all_s0 train set, 60 test, 3 epochs, lr=2e-4, bs=2):
  base           — no LoRA
  all-small-r2   — r=2, layers=[0..23], q/k/v/o  (~344K params)
  sparse-depth   — r=8, layers={0,6,12,18,23}, q/k/v/o (~287K)
  late-only      — r=8, layers=[12..23], q/k/v/o  (~688K)
  q-only-all     — r=8, layers=[0..23], only q_proj (~344K)
  vo-only-all    — r=8, layers=[0..23], v_proj+o_proj (~688K)

Reference (from P38/P39):
  Note: exact trainable counts are computed at runtime and saved in summary.json.
  The current run used 90 training samples.
  P39-L0-L11     — r=8, layers=[0..11], q/k/v/o   (~541K) [hall=59/60]
  P15-all-r4     — r=4, layers=[0..23], q/k/v/o   (~688K) [hall=34/60]

2×2 Interpretation matrix:
                         Low Param (~300K)        High Param (~600K+)
  Full Depth             all-small-r2, q-only     vo-only, P15-all
  Sparse/Late Depth      sparse-depth             late-only, L0-L11

If sparse > L0-L11: supports H1 (depth coverage matters)
If param correlates: supports H2 (parameter count)
If late > L0-L11: supports H3 (late decision)
If all ≈ base: supports H4 (CE objective mismatch)

Usage:
  python src/run_p40_full_system_routing_audit.py
"""

import os, sys, time, json, random, re, argparse
import torch
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model, TaskType

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_builder import load_jsonl

random.seed(42)
torch.manual_seed(42)

RESULTS_DIR = "results_p40_full_system_routing_audit"
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


def evaluate_generate(model, tokenizer, samples, device):
    results = []
    for sample in samples:
        ctx = sample.get("context", "")
        q = sample.get("question", "")
        prompt = f"{ctx}\n\nQuestion: {q}\nAnswer:"
        gen_text, gen_len = generate_response(model, tokenizer, prompt,
                                              device)
        behavior = classify_behavior(gen_text, ctx, q)
        rep = detect_repetition(gen_text)
        results.append({
            "behavior": behavior, "gen_text": gen_text,
            "gen_len": gen_len, "rep_score": round(rep, 4),
        })
    return results


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
        pos_lp = logprob_of_response(model, tokenizer, prompt, pos_resp,
                                     device)
        neg_lp = logprob_of_response(model, tokenizer, prompt, neg_resp,
                                     device)
        pref_positive = pos_lp > neg_lp
        results.append({
            "answerability": sample.get("answerability", "?"),
            "pref_positive": pref_positive,
            "pos_logprob": pos_lp, "neg_logprob": neg_lp,
        })
    return results


def compute_lp_metrics(eval_results):
    answerable = [r for r in eval_results
                  if r.get("answerability") == "answerable"]
    unanswerable = [r for r in eval_results
                    if r.get("answerability") == "unanswerable"]
    n_ans = len(answerable)
    n_unans = len(unanswerable)
    hallucinations = sum(1 for r in unanswerable if r["pref_positive"])
    correct = sum(1 for r in answerable if r["pref_positive"])
    H = hallucinations / n_unans if n_unans > 0 else 0.0
    C = correct / n_ans if n_ans > 0 else 0.0
    return {"H": round(H, 4), "C": round(C, 4),
            "hall_count": hallucinations, "unans_count": n_unans,
            "corr_count": correct, "ans_count": n_ans}


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
    log("P40: Full-System Routing Training Audit")
    log(f"  epochs={args.epochs}, lr={args.lr}, bs={args.batch_size}")
    log("  Testing 4 hypotheses: depth-coordination, param-count,")
    log("    late-decision, objective-mismatch")
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

    n_test_unans = sum(1 for s in test_samples
                       if s.get("answerability") != "answerable")
    n_test_ans = sum(1 for s in test_samples
                     if s.get("answerability") == "answerable")
    log(f"  Train: {len(train_samples)}, Test: {len(test_samples)} "
        f"({n_test_ans}A + {n_test_unans}U)")

    train_dataset = PairedResponseDataset(train_samples, tokenizer)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True)

    CONFIGS = [
        {"label": "all-small-r2", "rank": 2,
         "layers": list(range(0, 24)),
         "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
         "desc": "Full depth, tiny rank (r=2)",
         "h_test": "H1_full_depth"},
        {"label": "sparse-depth", "rank": 8,
         "layers": [0, 6, 12, 18, 23],
         "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
         "desc": "Sparse: {0,6,12,18,23} (5 layers, full coverage)",
         "h_test": "H1_full_depth"},
        {"label": "late-only", "rank": 8,
         "layers": list(range(12, 24)),
         "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
         "desc": "Late layers only: L12-L23",
         "h_test": "H3_late_decision"},
        {"label": "q-only-all", "rank": 8,
         "layers": list(range(0, 24)),
         "target_modules": ["q_proj"],
         "desc": "Full depth, q_proj only (1 projection)",
         "h_test": "H2_param_count"},
        {"label": "vo-only-all", "rank": 8,
         "layers": list(range(0, 24)),
         "target_modules": ["v_proj", "o_proj"],
         "desc": "Full depth, v+o only (2 projections)",
         "h_test": "H2_param_count"},
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
        log(f"    Rank={rank}, Layers={len(layers)}, "
            f"Targets={cfg['target_modules']}")
        log(f"    Tests hypothesis: {h_test}")
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
        total = sum(p.numel() for p in model.parameters())
        log(f"  Frozen/Active LoRA: {frozen}/{active}, "
            f"Trainable: {trainable:,} / {total:,} "
            f"({trainable/total*100:.2f}%)")

        log(f"  Training {args.epochs} epochs...")
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr)
        loss_history = train_loop(model, dataloader, optimizer,
                                  device, args.epochs)
        loss_str = " → ".join(f"{l:.3f}" for l in loss_history)
        log(f"  Loss: {loss_str}")

        log(f"  Evaluating generate() behavior...")
        model.eval()
        gen_results = evaluate_generate(model, tokenizer, test_samples,
                                        device)
        behaviors = [r["behavior"] for r in gen_results]
        hall = sum(1 for b in behaviors if b == "hallucination")
        abst = sum(1 for b in behaviors if b == "abstention")
        mixed = sum(1 for b in behaviors if b == "mixed")
        other = sum(1 for b in behaviors if b == "other")
        mean_len = (sum(r["gen_len"] for r in gen_results) /
                    max(1, len(gen_results)))
        mean_rep = (sum(r["rep_score"] for r in gen_results) /
                    max(1, len(gen_results)))

        t_elapsed = time.time() - t_cfg
        log(f"  hall={hall}/{len(test_samples)} abst={abst} "
            f"mixed={mixed} other={other} | "
            f"len={mean_len:.0f} rep={mean_rep:.3f} "
            f"({t_elapsed:.0f}s)")

        all_results.append({
            "label": label, "rank": rank, "n_layers": len(layers),
            "target_modules": cfg["target_modules"],
            "h_test": h_test, "desc": desc,
            "loss_history": loss_history,
            "hall": hall, "abst": abst, "mixed": mixed, "other": other,
            "total": len(test_samples),
            "hall_rate": round(hall / len(test_samples), 4),
            "abst_rate": round(abst / len(test_samples), 4),
            "mean_len": round(mean_len, 1),
            "mean_rep": round(mean_rep, 4),
            "trainable_params": trainable,
            "time_s": round(t_elapsed, 1),
        })

        del model

    log("\n" + "=" * 72)
    log("P40 RESULTS: Full-System Routing Audit")
    log("=" * 72)

    PRE_HALL = 46
    P39_L0L11_HALL = 59
    P15_ALL_HALL = 34

    log(f"\n  {'Config':<20s} {'H-test':<18s} {'Params':>8s} "
        f"{'Hall':>5s} {'Abst':>5s} {'Mix':>5s} {'Oth':>5s} "
        f"{'Len':>4s} {'Rep':>6s} {'ΔPre':>5s} {'ΔL0-11':>7s} "
        f"{'Loss'}")

    log(f"  {'─'*20} {'─'*18} {'─'*8} {'─'*5} {'─'*5} {'─'*5} "
        f"{'─'*5} {'─'*4} {'─'*6} {'─'*5} {'─'*7} {'─'*25}")

    for r in all_results:
        loss_str = " → ".join(f"{l:.1f}" for l in r["loss_history"])
        log(f"  {r['label']:<20s} {r['h_test']:<18s} "
            f"{r['trainable_params']:>8,d} "
            f"{r['hall']:>5d} {r['abst']:>5d} {r['mixed']:>5d} "
            f"{r['other']:>5d} {r['mean_len']:>4.0f} "
            f"{r['mean_rep']:>6.3f} "
            f"{r['hall']-PRE_HALL:>+5d} "
            f"{r['hall']-P39_L0L11_HALL:>+7d}  {loss_str}")

    log(f"\n  {'─'*20} {'─'*18} {'─'*8} {'─'*5} {'─'*5} {'─'*5} "
        f"{'─'*5} {'─'*4} {'─'*6} {'─'*5} {'─'*7}")
    log(f"  {'Pre (base)':<20s} {'—':<18s} {'0':>8s} "
        f"{PRE_HALL:>5d} {'1':>5s} {'4':>5s} {'9':>5s}")
    log(f"  {'P39 L0-L11':<20s} {'H1/H2/H3':<18s} {'541K':>8s} "
        f"{P39_L0L11_HALL:>5d} {'1':>5s} {'0':>5s} {'0':>5s}")
    log(f"  {'P15 all r=4':<20s} {'reference':<18s} {'688K':>8s} "
        f"{P15_ALL_HALL:>5d} {'7':>5s} {'10':>5s} {'9':>5s}")

    log("\n[Hypothesis Testing]")
    log("  H1 (full-depth coordination): "
        "sparse-depth or all-small-r2 < P39 L0-L11?")
    log("  H2 (parameter count): "
        "hall correlates with param count?")
    log("  H3 (late decision): "
        "late-only < P39 L0-L11?")
    log("  H4 (objective mismatch): "
        "all CE-based ≈ base? (DPO needed)")

    # quick numerical verdict
    sparse_row = next((r for r in all_results
                       if r["label"] == "sparse-depth"), None)
    late_row = next((r for r in all_results
                     if r["label"] == "late-only"), None)
    small_r2_row = next((r for r in all_results
                         if r["label"] == "all-small-r2"), None)
    qonly_row = next((r for r in all_results
                      if r["label"] == "q-only-all"), None)
    voonly_row = next((r for r in all_results
                       if r["label"] == "vo-only-all"), None)

    h1_evidence = False
    h2_evidence = False
    h3_evidence = False
    h4_evidence = False

    if sparse_row and sparse_row["hall"] < P39_L0L11_HALL:
        h1_evidence = True
        log(f"\n  H1 ✓: sparse-depth hall={sparse_row['hall']} "
            f"< L0-L11 hall={P39_L0L11_HALL}")
    elif small_r2_row and small_r2_row["hall"] < P39_L0L11_HALL:
        h1_evidence = True
        log(f"\n  H1 ✓: all-small-r2 hall={small_r2_row['hall']} "
            f"< L0-L11 hall={P39_L0L11_HALL}")
    else:
        log(f"\n  H1 ✗: sparse-depth and all-small-r2 both ≤ "
            f"L0-L11 ({P39_L0L11_HALL})? No")

    if late_row and late_row["hall"] < P39_L0L11_HALL:
        h3_evidence = True
        log(f"  H3 ✓: late-only hall={late_row['hall']} "
            f"< L0-L11 hall={P39_L0L11_HALL}")
    else:
        log(f"  H3 ✗: late-only not better than L0-L11")

    params_hall = [(r["trainable_params"], r["hall"])
                   for r in all_results]
    if len(params_hall) >= 3:
        xs = [p for p, _ in params_hall]
        ys = [h for _, h in params_hall]
        if len(set(xs)) > 1 and len(set(ys)) > 1:
            try:
                from statistics import correlation
                corr = correlation(xs, ys)
                log(f"  H2 corr(params, hall) = {corr:+.3f} "
                    f"({'supports' if corr > 0.5 else 'weak' if abs(corr)<0.5 else 'refutes'} "
                    f"H2)")
                if corr > 0.5:
                    h2_evidence = True
            except Exception:
                pass

    all_near_base = all(r["hall"] >= PRE_HALL - 5 for r in all_results)
    if all_near_base:
        h4_evidence = True
        log(f"  H4 ✓: all configs within 5 of base ({PRE_HALL}), "
            f"suggesting CE insufficient")

    log("\n[Verdict]")
    late_best = (
        late_row is not None
        and late_row["hall"] == min(r["hall"] for r in all_results)
    )
    if h3_evidence and late_best and not h2_evidence:
        log("  VERDICT: late_layer_decision_dominant")
        log("  Late decision layers explain most of the behavior control; "
            "depth coverage helps, but is weaker than late-layer targeting.")
    elif h1_evidence and not h2_evidence:
        log("  VERDICT: depth_coordination_dominant")
        log("  Behavior control depends on full-depth routing "
            "coordination, not parameter count.")
    elif h3_evidence:
        log("  VERDICT: late_layer_dominant")
        log("  Late (decision) layers explain a major part of the "
            "behavior control.")
    elif h2_evidence:
        log("  VERDICT: param_count_dominant")
        log("  Parameter count explains behavior control; "
            "depth structure is secondary.")
    elif h4_evidence:
        log("  VERDICT: objective_mismatch")
        log("  CE training cannot produce behavior change "
            "regardless of architecture. DPO/preference needed.")
    else:
        log("  VERDICT: mixed_or_inconclusive")
        log("  No single hypothesis clearly dominates.")

    elapsed = time.time() - t0
    summary = {
        "config": {
            "epochs": args.epochs, "lr": args.lr,
            "batch_size": args.batch_size, "seed": 42,
        },
        "train": {"n_samples": len(train_samples)},
        "test": {
            "n_samples": len(test_samples),
            "n_answerable": n_test_ans,
            "n_unanswerable": n_test_unans,
        },
        "baselines": {
            "pre": {"hall": PRE_HALL, "abst": 1},
            "p39_l0l11": {"hall": P39_L0L11_HALL},
            "p15_all": {"hall": P15_ALL_HALL, "abst": 7},
        },
        "results": all_results,
        "verdict": {
            "h1_depth_coordination": h1_evidence,
            "h2_parameter_count": h2_evidence,
            "h3_late_decision": h3_evidence,
            "h4_objective_mismatch": h4_evidence,
        },
        "time_s": round(elapsed, 1),
        "time_min": round(elapsed / 60, 1),
    }
    with open(os.path.join(RESULTS_DIR, "summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log(f"\nResults saved to {RESULTS_DIR}/summary.json")
    log(f"Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log("P40 Complete.")


if __name__ == "__main__":
    main()
