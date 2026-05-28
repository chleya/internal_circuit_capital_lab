"""
P36b: Behavior Robustness Audit — 复核 P36 的 log-prob → behavior gap
=====================================================================
核心问题: P36 发现 L0 token-entry ablation 明显改变 log-prob，
         但小样本 generate() 行为没有稳定改变。
         这个 gap 在更强行为测量下是否仍然成立？

改进（针对 P36 的 5 个边界问题）:
  1. behavior n >> 5 — 使用每个 token 族的全部可用样本
  2. fixed random seed (42) — 随机位置对照可复现
  3. 覆盖所有 token 族 — 不只 funding
  4. 每样本 paired comparison — baseline vs intervention 同 prompt
  5. 改进分类器 — 透明、可审查的决策表
  6. 输出所有 generated text — 方便人工 audit
  7. matched noncausal controls — 邻近句法区域
  8. 修正 log-prob 计算 — 使用 per-token 序列 log-prob 而非平均 loss

比较:
  - baseline
  - embedding ablation
  - L0 combined
  - L0 MLP-only
  - L16 combined
  - noncausal matched control
  - random-position fixed-seed control

Usage:
  python src/run_p36b_behavior_robustness.py
"""

import os, sys, time, json, random, re, math
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_builder import load_jsonl

random.seed(42)

RESULTS_DIR = "results_p36b_behavior_robustness"
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, "run_log.txt")
AUDIT_TABLE_PATH = os.path.join(RESULTS_DIR, "per_sample_audit.jsonl")

SCALE_ZERO = 0.0


def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def find_token_positions(prompt, tokenizer, target_words):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())
    result = {}
    for target in target_words:
        for variant in [target, " " + target, target.upper(), " " + target.upper(),
                        target.lower(), " " + target.lower()]:
            variant_tokens = tokenizer.tokenize(variant)
            n = len(variant_tokens)
            if n == 0:
                continue
            for i in range(len(tokens) - n + 1):
                if tokens[i:i + n] == variant_tokens:
                    result[target] = list(range(i, i + n))
                    break
            if target in result:
                break
    return result


def find_matched_noncausal(prompt, tokenizer, causal_positions, all_causal_tokens):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())
    causal_flat = set()
    for pos_list in causal_positions.values():
        causal_flat.update(pos_list)

    available = sorted(set(range(len(tokens))) - causal_flat)
    if not available:
        return None

    causal_centers = []
    for pos_list in causal_positions.values():
        if pos_list:
            causal_centers.append((min(pos_list) + max(pos_list)) // 2)
    if not causal_centers:
        return None
    center = sum(causal_centers) // len(causal_centers)

    best = available[0]
    best_dist = abs(best - center)
    for a in available:
        d = abs(a - center)
        if d < best_dist and a != center:
            best = a
            best_dist = d
    return [best]


def find_random_position_fixed(prompt, tokenizer, causal_positions):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())
    causal_flat = set()
    for pos_list in causal_positions.values():
        causal_flat.update(pos_list)
    all_pos = set(range(len(tokens)))
    available = sorted(all_pos - causal_flat)
    if not available:
        return None
    random.seed(hash(prompt) % (2 ** 31))
    return [random.choice(available)]


def build_pos_hook(pos_set, scale):
    def hook(module, input, output):
        if isinstance(output, (tuple, list)):
            hidden = output[0].clone()
            rest = output[1:]
        else:
            hidden = output.clone()
            rest = None
        if hidden.dim() == 3:
            for pos in pos_set:
                if pos < hidden.shape[1]:
                    hidden[0, pos, :] *= scale
        elif hidden.dim() == 2:
            for pos in pos_set:
                if pos < hidden.shape[0]:
                    hidden[pos, :] *= scale
        if rest is None:
            return hidden
        if isinstance(output, tuple):
            return (hidden,) + rest
        return [hidden] + rest
    return hook


def compute_token_logp(model, tokenizer, prompt, response, device, hooks_config):
    full_text = f"{prompt} {response}"
    enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"].to(device)
    prompt_enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    prompt_len = prompt_enc["input_ids"].shape[1]

    handles = []
    for target, hook_fn in hooks_config:
        if target == "embed":
            handles.append(model.model.embed_tokens.register_forward_hook(hook_fn))
        elif target.startswith("L"):
            parts = target.split(".")
            layer_idx = int(parts[0][1:])
            layer = model.model.layers[layer_idx]
            if len(parts) > 1 and parts[1] == "self_attn":
                handles.append(layer.self_attn.register_forward_hook(hook_fn))
            elif len(parts) > 1 and parts[1] == "mlp":
                handles.append(layer.mlp.register_forward_hook(hook_fn))
            else:
                handles.append(layer.register_forward_hook(hook_fn))

    try:
        labels = input_ids.clone()
        labels[0, :prompt_len] = -100
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)
        n_response = (labels != -100).sum().item()
        if n_response > 0:
            total_logp = -outputs.loss.item() * n_response
        else:
            total_logp = 0.0
        return total_logp
    finally:
        for h in handles:
            h.remove()


def run_generate(model, tokenizer, prompt, device, hooks_config=None, max_new=64):
    if hooks_config is None:
        hooks_config = []
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"].to(device)

    handles = []
    for target, hook_fn in hooks_config:
        if target == "embed":
            handles.append(model.model.embed_tokens.register_forward_hook(hook_fn))
        elif target.startswith("L"):
            parts = target.split(".")
            layer_idx = int(parts[0][1:])
            layer = model.model.layers[layer_idx]
            if len(parts) > 1 and parts[1] == "self_attn":
                handles.append(layer.self_attn.register_forward_hook(hook_fn))
            elif len(parts) > 1 and parts[1] == "mlp":
                handles.append(layer.mlp.register_forward_hook(hook_fn))
            else:
                handles.append(layer.register_forward_hook(hook_fn))

    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_ids = outputs[0][input_ids.shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        return gen_text, len(gen_ids)
    finally:
        for h in handles:
            h.remove()


def extract_numbers_from_text(text):
    pattern = r'(?<!\w)(\d[\d,.]*(?:\s*(?:million|billion|thousand|%|percent))?)(?!\w)'
    matches = re.findall(pattern, text)
    numbers = re.findall(r'\d[\d,.]*', text)
    return matches, numbers


def extract_numbers_from_context(ctx):
    return re.findall(r'\d[\d,.]*', ctx)


# ================================================================
# Improved classifier — transparent, auditable
# ================================================================
def classify_behavior(text, context, question, negative_response):
    text_clean = text.strip()
    text_lower = text_clean.lower()

    abstain_markers_strong = [
        "not provided", "not mentioned", "not stated", "not available",
        "no information", "cannot be determined", "can't be determined",
        "does not provide", "does not mention", "does not specify",
        "do not have", "don't have", "unclear", "not enough information",
        "insufficient information", "the passage does not",
        "the text does not", "not disclosed",
    ]
    abstain_markers_weak = [
        "cannot answer", "can't answer", "unknown", "n/a",
        "not specified", "unspecified",
    ]

    strong_hit = [
        m for m in abstain_markers_strong if m in text_lower
    ]
    weak_hit = [
        m for m in abstain_markers_weak if m in text_lower
    ]

    ctx_numbers = extract_numbers_from_context(context)
    text_num_matches, text_numbers = extract_numbers_from_text(text_clean)

    new_numbers = [n for n in text_numbers
                   if n not in ctx_numbers and n not in ctx_numbers]

    is_numeric_hall = any(
        re.search(r'\d', t) for t in text_clean.split()[:30])

    reason_parts = []

    if strong_hit and not new_numbers:
        return "abstention", f"strong: {', '.join(strong_hit[:3])}"

    if strong_hit and new_numbers:
        return "mixed", (f"abstain_form({strong_hit[0]}) "
                         f"+ numbers({', '.join(new_numbers[:2])})")

    if weak_hit and not is_numeric_hall:
        return "abstention", f"weak: {', '.join(weak_hit[:3])}"

    if is_numeric_hall:
        if not new_numbers:
            new_numbers = text_numbers[:3]
        return "hallucination", (f"numeric: {', '.join(new_numbers[:3])}"
                                 if new_numbers else "numeric_generic")

    if len(text_clean.split()) < 5:
        return "other", "too_short"

    return "other", "no_pattern_matched"


# ================================================================
# Intervention configs (simplified set from masterplan P36b)
# ================================================================
INTERVENTION_LABELS = [
    "baseline",
    "embed_ablate",
    "L0_combined",
    "L0_mlp_only",
    "L16_combined",
    "noncausal_matched",
    "random_fixed_seed",
]

INTERVENTION_CONFIGS = {
    "baseline": lambda pos_set: [],
    "embed_ablate": lambda pos_set: [
        ("embed", build_pos_hook(pos_set, SCALE_ZERO))
    ],
    "L0_combined": lambda pos_set: [
        ("L0.self_attn", build_pos_hook(pos_set, SCALE_ZERO)),
        ("L0.mlp", build_pos_hook(pos_set, SCALE_ZERO)),
    ],
    "L0_mlp_only": lambda pos_set: [
        ("L0.mlp", build_pos_hook(pos_set, SCALE_ZERO)),
    ],
    "L16_combined": lambda pos_set: [
        ("L16.self_attn", build_pos_hook(pos_set, SCALE_ZERO)),
        ("L16.mlp", build_pos_hook(pos_set, SCALE_ZERO)),
    ],
    "noncausal_matched": None,
    "random_fixed_seed": None,
}


def get_hooks(label, prompt, tokenizer, token_positions, causal_token_name,
              all_candidate_tokens):
    causal_pos = set(token_positions.get(causal_token_name, []))

    if label == "noncausal_matched":
        nc_pos = find_matched_noncausal(
            prompt, tokenizer, token_positions, all_candidate_tokens)
        if nc_pos is None:
            return []
        return INTERVENTION_CONFIGS["L0_combined"](set(nc_pos))

    if label == "random_fixed_seed":
        rand_pos = find_random_position_fixed(prompt, tokenizer, token_positions)
        if rand_pos is None:
            return []
        return INTERVENTION_CONFIGS["L0_combined"](set(rand_pos))

    if label in INTERVENTION_CONFIGS:
        return INTERVENTION_CONFIGS[label](causal_pos)

    return []


def main():
    log("=" * 72)
    log("P36b: Behavior Robustness Audit")
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
    log(f"  Model: {model_name} on {device}")

    # ================================================================
    # DATA LOADING
    # ================================================================
    log("\n[Step 2] Loading data & computing baselines...")
    data_dir = os.path.join(base_dir, "data_position_sensitivity", "s0")

    data_files = [("train_all_s0.jsonl", "train")]
    for pos in ["early", "mid", "late"]:
        path = os.path.join(data_dir, f"test_{pos}_s0.jsonl")
        if os.path.exists(path):
            data_files.append((f"test_{pos}_s0.jsonl", pos))

    seen_keys = set()
    all_samples = []
    for fname, pos_label in data_files:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            continue
        for s in load_jsonl(path):
            key = (s.get("entity_id", -1), s.get("template_id", ""))
            if key not in seen_keys:
                seen_keys.add(key)
                s["_position"] = pos_label
                s["_source"] = fname
                all_samples.append(s)

    log(f"  Loaded {len(all_samples)} unique samples")

    CANDIDATE_TOKENS = [
        "funding", "r_and_d_spend", "revenue",
        "employees", "patents", "market_share",
        "active_users", "growth_rate", "carbon_emissions",
        "customer_satisfaction", "production_volume",
    ]

    unanswerable = []
    samples_meta = []

    for idx, sample in enumerate(all_samples):
        ctx = sample.get("context", "")
        q = sample.get("question", "")
        prompt = f"{ctx}\n\nQuestion: {q}\nAnswer:"
        pos_resp = sample.get("positive_response", "")
        neg_resp = sample.get("negative_response", "")

        token_positions = find_token_positions(prompt, tokenizer, CANDIDATE_TOKENS)

        meta = {
            "idx": idx, "sample": sample, "prompt": prompt,
            "pos_resp": pos_resp, "neg_resp": neg_resp,
            "token_positions": token_positions,
        }
        samples_meta.append(meta)

        if sample.get("answerability") == "unanswerable":
            unanswerable.append(meta)

    token_samples = {}
    for token_name in CANDIDATE_TOKENS:
        samples_with_token = [
            m for m in unanswerable
            if token_name in m["token_positions"]
            and len(m["token_positions"][token_name]) > 0
        ]
        token_samples[token_name] = {
            "all": samples_with_token,
        }

    usable_tokens = []
    for token_name in CANDIDATE_TOKENS:
        ts = token_samples[token_name]
        if len(ts["all"]) >= 2:
            usable_tokens.append(token_name)

    log(f"\n  Usable tokens (>=2 unanswerable samples):")
    for token_name in usable_tokens:
        n = len(token_samples[token_name]["all"])
        log(f"    {token_name:25s}: {n} samples")

    # ================================================================
    # MAIN EXPERIMENT: Per-token, per-intervention behavioral test
    # ================================================================
    log("\n" + "=" * 72)
    log("P36b: Behavioral tests — all token families, paired")
    log("=" * 72)

    all_token_results = {}
    audit_entries = []

    for token_name in usable_tokens:
        ts = token_samples[token_name]
        samples = ts["all"]
        n_samples = len(samples)
        log(f"\n{'─' * 64}")
        log(f"  Token: {token_name} ({n_samples} samples)")
        log(f"{'─' * 64}")

        token_result = {"token": token_name, "n": n_samples,
                        "interventions": {}}

        for label in INTERVENTION_LABELS:
            results = []

            for m in samples:
                prompt = m["prompt"]
                neg_resp = m["neg_resp"]
                pos_resp = m["pos_resp"]
                ctx = m["sample"].get("context", "")
                q = m["sample"].get("question", "")
                causal_pos_set = set(m["token_positions"].get(token_name, []))

                hooks = get_hooks(label, prompt, tokenizer,
                                  m["token_positions"], token_name,
                                  CANDIDATE_TOKENS)

                gen_text, gen_len = run_generate(
                    model, tokenizer, prompt, device, hooks)
                behavior, reason = classify_behavior(
                    gen_text, ctx, q, neg_resp)

                lp_pos = compute_token_logp(
                    model, tokenizer, prompt, pos_resp, device, hooks)
                lp_neg = compute_token_logp(
                    model, tokenizer, prompt, neg_resp, device, hooks)
                lp_diff = lp_pos - lp_neg

                results.append({
                    "behavior": behavior,
                    "reason": reason,
                    "lp_diff": round(lp_diff, 4),
                    "gen_len": gen_len,
                    "gen_text": gen_text,
                })

                audit_entries.append({
                    "token": token_name,
                    "intervention": label,
                    "sample_idx": m["idx"],
                    "behavior": behavior,
                    "reason": reason,
                    "lp_diff": round(lp_diff, 4),
                    "gen_len": gen_len,
                    "gen_text": gen_text,
                    "context": ctx[:100],
                    "question": q[:100],
                })

            behaviors = [r["behavior"] for r in results]
            n_total = len(results)
            n_hall = sum(1 for b in behaviors if b == "hallucination")
            n_abst = sum(1 for b in behaviors if b == "abstention")
            n_mixed = sum(1 for b in behaviors if b == "mixed")
            n_other = sum(1 for b in behaviors if b not in
                          ("hallucination", "abstention", "mixed"))
            lp_diffs = [r["lp_diff"] for r in results]
            mean_lp = sum(lp_diffs) / max(1, n_total)
            mean_len = sum(r["gen_len"] for r in results) / max(1, n_total)

            log(f"    {label:25s}: hall={n_hall}/{n_total} "
                f"abst={n_abst}/{n_total} mixed={n_mixed}/{n_total} "
                f"other={n_other}/{n_total} | "
                f"lp_diff={mean_lp:+.2f} len={mean_len:.1f}")

            token_result["interventions"][label] = {
                "n_total": n_total,
                "hall_rate": f"{n_hall}/{n_total}",
                "abst_rate": f"{n_abst}/{n_total}",
                "mixed_rate": f"{n_mixed}/{n_total}",
                "other_rate": f"{n_other}/{n_total}",
                "mean_lp_diff": round(mean_lp, 4),
                "mean_gen_len": round(mean_len, 1),
            }

            if n_total >= 3:
                for i, r in enumerate(results):
                    log(f"      [{i}] {r['behavior']:15s} "
                        f"len={r['gen_len']:2d} "
                        f"lp_diff={r['lp_diff']:+.2f} "
                        f"\"{r['gen_text'][:70]}\"")

        all_token_results[token_name] = token_result

    # ================================================================
    # PAIRED FLIPS — baseline vs each intervention
    # ================================================================
    log("\n" + "=" * 72)
    log("Paired flips: baseline → intervention behavior changes")
    log("=" * 72)

    flip_summary = {}
    for token_name in usable_tokens:
        ts = token_samples[token_name]
        samples = ts["all"]
        token_result = all_token_results[token_name]

        baseline_results = token_result["interventions"].get(
            "baseline", {}).get("n_total", 0)
        if isinstance(baseline_results, int) and baseline_results < 2:
            continue

        flip_summary[token_name] = {}
        for label in INTERVENTION_LABELS:
            if label == "baseline":
                continue

            n_better = 0
            n_worse = 0
            n_same = 0

            for si, m in enumerate(samples):
                prompt = m["prompt"]
                neg_resp = m["neg_resp"]
                ctx = m["sample"].get("context", "")

                hooks_base = get_hooks(
                    "baseline", prompt, tokenizer,
                    m["token_positions"], token_name, CANDIDATE_TOKENS)
                hooks_intv = get_hooks(
                    label, prompt, tokenizer,
                    m["token_positions"], token_name, CANDIDATE_TOKENS)

                gen_base, _ = run_generate(
                    model, tokenizer, prompt, device, hooks_base)
                gen_intv, _ = run_generate(
                    model, tokenizer, prompt, device, hooks_intv)

                behav_base, _ = classify_behavior(
                    gen_base, ctx, m["sample"].get("question", ""), neg_resp)
                behav_intv, _ = classify_behavior(
                    gen_intv, ctx, m["sample"].get("question", ""), neg_resp)

                rank = {"abstention": 0, "other": 1, "mixed": 2,
                        "hallucination": 3}
                base_r = rank.get(behav_base, 1)
                intv_r = rank.get(behav_intv, 1)

                if intv_r < base_r:
                    n_better += 1
                elif intv_r > base_r:
                    n_worse += 1
                else:
                    n_same += 1

            flip_summary[token_name][label] = {
                "better": n_better,
                "worse": n_worse,
                "same": n_same,
                "n": n_better + n_worse + n_same,
            }

            if n_better + n_worse + n_same > 0:
                pct_better = n_better / (n_better + n_worse + n_same) * 100
                pct_worse = n_worse / (n_better + n_worse + n_same) * 100
                log(f"  {token_name:25s} {label:20s}: "
                    f"better={n_better} worse={n_worse} same={n_same} "
                    f"({pct_better:.0f}% / {pct_worse:.0f}%)")

    # ================================================================
    # SUMMARY TABLE
    # ================================================================
    log("\n" + "=" * 72)
    log("SUMMARY: Hallucination rate across all tokens & interventions")
    log("=" * 72)

    header = f"{'Token':28s}"
    for label in INTERVENTION_LABELS:
        header += f" {'hall/' + label:>18s}"
    log(header)

    for token_name in usable_tokens:
        tr = all_token_results[token_name]
        row = f"{token_name:28s}"
        for label in INTERVENTION_LABELS:
            intv = tr["interventions"].get(label, {})
            hr = intv.get("hall_rate", "N/A")
            row += f" {hr:>18s}"
        log(row)

    # ================================================================
    # VERDICT
    # ================================================================
    log("\n" + "=" * 72)
    log("P36b 判决")
    log("=" * 72)

    any_positive = False
    any_artifact = False
    token_verdicts = []

    for token_name in usable_tokens:
        tr = all_token_results[token_name]
        base_intv = tr["interventions"].get("baseline", {})
        l0_intv = tr["interventions"].get("L0_combined", {})
        embed_intv = tr["interventions"].get("embed_ablate", {})

        base_hall_str = base_intv.get("hall_rate", "0/0")
        l0_hall_str = l0_intv.get("hall_rate", "0/0")
        embed_hall_str = embed_intv.get("hall_rate", "0/0")

        if "/" in base_hall_str:
            parts = base_hall_str.split("/")
            bh = int(parts[0])
            bt = int(parts[1])
        else:
            bh, bt = 0, 0
        if "/" in l0_hall_str:
            parts = l0_hall_str.split("/")
            lh = int(parts[0])
            lt = int(parts[1])
        else:
            lh, lt = 0, 0
        if "/" in embed_hall_str:
            parts = embed_hall_str.split("/")
            eh = int(parts[0])
            et = int(parts[1])
        else:
            eh, et = 0, 0

        base_lp = base_intv.get("mean_lp_diff", 0)
        l0_lp = l0_intv.get("mean_lp_diff", 0)

        behavior_changed = (abs(bh - lh) >= max(1, bt * 0.3))
        lp_changed = abs(l0_lp - base_lp) > 0.05

        if behavior_changed:
            any_positive = True
            token_verdicts.append(
                f"  {token_name}: behavior_changed "
                f"(hall {bh}/{bt}→{lh}/{lt})")
        elif lp_changed and not behavior_changed:
            token_verdicts.append(
                f"  {token_name}: lp changed ({base_lp:+.2f}→{l0_lp:+.2f}) "
                f"but behavior unchanged ({bh}/{bt}→{lh}/{lt})")
        else:
            token_verdicts.append(
                f"  {token_name}: weak lp effect and no behavior change")

    for tv in token_verdicts:
        log(tv)

    if any_positive:
        log("\n  VERDICT: scoped_positive")
        log("  Behavior moves for some token families. "
            "P36's null was a small-n artifact.")
    elif any_artifact:
        log("\n  VERDICT: failed_due_to_artifact")
        log("  Original P36 null was due to classifier/sample/control issues.")
    else:
        log("\n  VERDICT: negative_but_informative")
        log("  Log-prob effect remains but behavior still does not "
            "reliably move under stronger measurement.")

    # ================================================================
    # SAVE
    # ================================================================
    elapsed = time.time() - t0

    with open(AUDIT_TABLE_PATH, "w", encoding="utf-8") as f:
        for entry in audit_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    output = {
        "config": {
            "interventions": INTERVENTION_LABELS,
            "tokens": usable_tokens,
            "random_seed": 42,
        },
        "time_s": round(elapsed, 1),
        "time_min": round(elapsed / 60, 1),
        "token_results": all_token_results,
        "flip_summary": flip_summary,
    }

    with open(os.path.join(RESULTS_DIR, "summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    log(f"\nResults saved to {RESULTS_DIR}/")
    log(f"  - summary.json")
    log(f"  - per_sample_audit.jsonl ({len(audit_entries)} entries)")
    log(f"  - run_log.txt")
    log(f"\nTime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log(f"\nP36b Complete.")


if __name__ == "__main__":
    main()