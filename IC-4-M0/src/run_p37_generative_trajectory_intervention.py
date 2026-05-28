"""
P37: Generative-Trajectory Intervention — counter-vector injection vs ablation
==============================================================================
核心问题: P36/P36b 证明 L0 token-entry ablation 改变 log-prob 但无法控制行为。
        换成 counter-vector direction injection 能否在生成轨迹层面
        降低 hallucination rate？

关键改进 (from P36b):
  1. 使用 counter-vector 注入 (abstention→hallucination direction) 而非消融
  2. 比较 5 种 timing: T0_ablation, T0_counter, T1_continuous, T2_late, T1_random
  3. 每步注入而非仅一次性干预
  4. Fixed seed (42), matched controls
  5. Per-sample full text audit output

Timing modes:
  T0_ablation: P36-style zeroing at causal pos, prefill only (step == 0)
  T0_counter: counter-vector at causal pos, prefill only (step == 0)
  T1_continuous: counter-vector at causal pos, ALL generation steps
  T2_late: counter-vector activates after 4 tokens generated (step >= 4)
  T1_random: same-norm random direction, T1 timing (control)
  baseline: no hook

Counter-vector:
  v_anti = mean(L0_hidden | abstention_response) - mean(L0_hidden | hallucination_response)
  Computed per-token-family from paired (hallucination, abstention) forward passes

Usage:
  python src/run_p37_generative_trajectory_intervention.py
"""

import os, sys, time, json, random, re
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_builder import load_jsonl

random.seed(42)
torch.manual_seed(42)

RESULTS_DIR = "results_p37_generative_trajectory_intervention"
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, "run_log.txt")
AUDIT_PATH = os.path.join(RESULTS_DIR, "per_sample_audit.jsonl")


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


def compute_counter_vector_grad(model, tokenizer, samples, token_name, device):
    all_grads = []

    for si, m in enumerate(samples):
        prompt = m["prompt"]
        pos_resp = m["pos_resp"]
        neg_resp = m["neg_resp"]
        causal_pos = set(m["token_positions"].get(token_name, []))

        if not causal_pos:
            continue

        enc_prompt = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=512)
        prompt_len = enc_prompt["input_ids"].shape[1]

        for resp_label in ["hall", "abst"]:
            resp = pos_resp if resp_label == "hall" else neg_resp
            full_text = f"{prompt} {resp}"
            enc = tokenizer(full_text, return_tensors="pt", truncation=True,
                            max_length=512)
            input_ids = enc["input_ids"].to(device)

            labels = input_ids.clone()
            labels[0, :prompt_len] = -100

            captured = {}
            def capture(module, input, output, _cap=captured):
                if isinstance(output, (tuple, list)):
                    _cap['h'] = output[0]
                else:
                    _cap['h'] = output

            model.zero_grad()
            handle = model.model.layers[0].register_forward_hook(capture)
            try:
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss

                if 'h' not in captured:
                    handle.remove()
                    continue
                h = captured['h']
                h.retain_grad()
                loss.backward()

                if h.grad is not None:
                    grad_at_causal = []
                    for p in causal_pos:
                        if p < h.shape[1]:
                            grad_at_causal.append(h.grad[0, p].clone())
                    if grad_at_causal:
                        avg_grad = torch.stack(grad_at_causal).mean(0)
                        if resp_label == "hall":
                            all_grads.append(-avg_grad)
                        else:
                            all_grads.append(avg_grad)
            finally:
                handle.remove()

    if len(all_grads) < 2:
        return None

    v = torch.stack(all_grads).mean(0)
    v_norm = v.norm().item()
    if v_norm < 1e-6:
        return None
    return v / v_norm


def make_hook_factory(causal_pos_set, vector_dict, mode, counter_vector, d_model):
    step_counter = [0]

    def build_hook_fn(pos_set, cv, layer_name):
        def hook(module, input, output):
            step = step_counter[0]
            if isinstance(output, (tuple, list)):
                hidden = output[0].clone()
            else:
                hidden = output.clone()

            if hidden.dim() != 3:
                return output

            current_vec = None
            current_mode = vector_dict.get(layer_name, {}).get("mode", mode)

            if current_mode == "T0_ablation":
                if step == 0:
                    for p in pos_set:
                        if p < hidden.shape[1]:
                            hidden[0, p, :] = 0

            elif current_mode in ("T0_counter", "T1_continuous", "T2_late",
                                   "T1_random"):
                if current_mode == "T0_counter":
                    should_apply = (step == 0)
                elif current_mode == "T2_late":
                    should_apply = (step >= 4)
                else:
                    should_apply = True

                if should_apply and cv is not None:
                    cv_dev = cv.to(hidden.device)
                    for p in pos_set:
                        if p < hidden.shape[1]:
                            hidden[0, p, :] = (
                                hidden[0, p, :]
                                + vector_dict.get("scale", 1.0) * 0.1 * cv_dev
                            )

            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        return hook

    return build_hook_fn, step_counter


def run_generate_hooked(model, tokenizer, prompt, device, hooks_defs, max_new=64):
    """
    hooks_defs: list of (target_str, hook_fn) pairs
    Each hook_fn uses a shared step_counter dict keyed by 'step'.
    During generate(), step_counter is incremented by a wrapper hook.
    """
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"].to(device)

    step_counter = [0]

    def step_hook(module, input, output):
        step_counter[0] += 1
        return output

    handles = []
    handle_step = model.model.layers[0].register_forward_hook(step_hook)
    handles.append(handle_step)

    for target, hook_fn in hooks_defs:
        if target == "embed":
            handles.append(
                model.model.embed_tokens.register_forward_hook(hook_fn))
        elif target.startswith("L"):
            parts = target.split(".")
            layer_idx = int(parts[0][1:])
            layer = model.model.layers[layer_idx]
            if len(parts) > 1 and parts[1] == "self_attn":
                handles.append(
                    layer.self_attn.register_forward_hook(hook_fn))
            elif len(parts) > 1 and parts[1] == "mlp":
                handles.append(
                    layer.mlp.register_forward_hook(hook_fn))
            else:
                handles.append(
                    layer.register_forward_hook(hook_fn))

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
        return gen_text, len(gen_ids), step_counter[0]
    finally:
        for h in handles:
            h.remove()


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


def main():
    log("=" * 72)
    log("P37: Generative-Trajectory Intervention (counter-vector)")
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
    d_model = model.config.hidden_size

    # ================================================================
    # DATA LOADING
    # ================================================================
    log("\n[Step 2] Loading data...")
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
                all_samples.append(s)

    CANDIDATE_TOKENS = [
        "funding", "r_and_d_spend", "revenue",
        "employees", "patents", "market_share",
        "active_users", "growth_rate", "carbon_emissions",
        "customer_satisfaction", "production_volume",
    ]

    samples_meta = []
    for idx, sample in enumerate(all_samples):
        ctx = sample.get("context", "")
        q = sample.get("question", "")
        prompt = f"{ctx}\n\nQuestion: {q}\nAnswer:"
        token_positions = find_token_positions(prompt, tokenizer, CANDIDATE_TOKENS)
        samples_meta.append({
            "idx": idx, "sample": sample, "prompt": prompt,
            "pos_resp": sample.get("positive_response", ""),
            "neg_resp": sample.get("negative_response", ""),
            "token_positions": token_positions,
        })

    unanswerable = [m for m in samples_meta
                    if m["sample"].get("answerability") == "unanswerable"]

    token_samples = {}
    for token_name in CANDIDATE_TOKENS:
        ts = [m for m in unanswerable
              if token_name in m["token_positions"]
              and len(m["token_positions"][token_name]) > 0]
        if len(ts) >= 2:
            token_samples[token_name] = ts

    primary_tokens = sorted(token_samples.keys(),
                            key=lambda t: len(token_samples[t]), reverse=True)
    log(f"  Tokens with >=2 unanswerable samples: {primary_tokens}")
    for tn in primary_tokens[:3]:
        log(f"    {tn}: {len(token_samples[tn])} samples")

    # ================================================================
    # COUNTER-VECTOR COMPUTATION
    # ================================================================
    log("\n[Step 3] Computing counter-vectors...")
    counter_vectors = {}

    for token_name in primary_tokens[:3]:
        cv = compute_counter_vector_grad(
            model, tokenizer, token_samples[token_name],
            token_name, device)
        if cv is not None:
            counter_vectors[token_name] = cv.cpu()
            log(f"  {token_name}: |cv|={cv.norm().item():.4f} "
                f"(computed from {len(token_samples[token_name])} samples)")
        else:
            log(f"  {token_name}: FAILED to compute")

    if not counter_vectors:
        log("\n[FATAL] No counter-vectors computed. Exiting.")
        return

    # ================================================================
    # INTERVENTION COMPARISON
    # ================================================================
    log("\n" + "=" * 72)
    log("P37: Timing comparison — counter-vector vs ablation")
    log("=" * 72)

    MODE_LABELS = [
        "baseline",
        "T0_ablation",
        "T0_counter",
        "T1_continuous",
        "T2_late",
        "T1_random",
    ]

    MODE_DESCRIPTIONS = {
        "baseline": "no intervention",
        "T0_ablation": "zero L0 output at causal pos, prefill only",
        "T0_counter": "counter-vector at causal pos, prefill only",
        "T1_continuous": "counter-vector at causal pos, every step",
        "T2_late": "counter-vector at causal pos, step >= 4",
        "T1_random": "same-norm random dir at causal pos, every step (control)",
    }

    all_token_results = {}
    audit_entries = []

    for token_name in primary_tokens[:2]:
        if token_name not in counter_vectors:
            continue

        samples = token_samples[token_name]
        cv = counter_vectors[token_name]
        n_max = min(len(samples), 8)
        use_samples = samples[:n_max]

        log(f"\n{'─' * 64}")
        log(f"  Token: {token_name} ({n_max} samples)")
        log(f"{'─' * 64}")

        token_result = {"token": token_name, "n": n_max, "modes": {}}

        for mode in MODE_LABELS:
            results = []
            mode_desc = MODE_DESCRIPTIONS.get(mode, mode)

            for m in use_samples:
                prompt = m["prompt"]
                ctx = m["sample"].get("context", "")
                q = m["sample"].get("question", "")
                causal_pos_set = set(
                    m["token_positions"].get(token_name, []))

                if not causal_pos_set:
                    results.append({
                        "behavior": "skip",
                        "reason": "no_causal_pos",
                        "gen_text": "",
                        "gen_len": 0,
                        "steps": 0,
                    })
                    continue

                if mode == "T1_random":
                    rand_vec = torch.randn(d_model)
                    rand_vec = rand_vec / rand_vec.norm()
                    use_cv = rand_vec
                    use_mode_for_hook = "T1_continuous"
                elif mode == "T0_ablation":
                    use_cv = None
                    use_mode_for_hook = "T0_ablation"
                elif mode == "baseline":
                    use_cv = None
                    use_mode_for_hook = "baseline"
                else:
                    use_cv = cv
                    use_mode_for_hook = mode

                if mode == "baseline":
                    hooks_defs = []
                else:
                    vector_dict = {"scale": 1.0, "mode": use_mode_for_hook}
                    factory_fn, _ = make_hook_factory(
                        causal_pos_set, vector_dict,
                        use_mode_for_hook, use_cv, d_model)
                    hook_l0 = factory_fn(causal_pos_set, use_cv, "L0")
                    hooks_defs = [("L0", hook_l0)]

                gen_text, gen_len, steps = run_generate_hooked(
                    model, tokenizer, prompt, device, hooks_defs)

                behavior, reason = classify_behavior(gen_text, ctx, q)
                rep = detect_repetition(gen_text)

                results.append({
                    "behavior": behavior,
                    "reason": reason,
                    "gen_text": gen_text,
                    "gen_len": gen_len,
                    "steps": steps,
                    "rep_score": round(rep, 4),
                })

                audit_entries.append({
                    "token": token_name,
                    "mode": mode,
                    "sample_idx": m["idx"],
                    "behavior": behavior,
                    "reason": reason,
                    "gen_text": gen_text,
                    "gen_len": gen_len,
                    "rep": round(rep, 4),
                    "steps": steps,
                })

            behaviors = [r["behavior"] for r in results]
            n_total = len([r for r in results if r["behavior"] != "skip"])
            n_hall = sum(1 for b in behaviors if b == "hallucination")
            n_abst = sum(1 for b in behaviors if b == "abstention")
            n_mixed = sum(1 for b in behaviors if b == "mixed")
            n_other = sum(1 for b in behaviors if b == "other")
            mean_len = (sum(r["gen_len"] for r in results) /
                        max(1, len([r for r in results
                                    if r["gen_len"] > 0])))
            mean_rep = (sum(r["rep_score"] for r in results) /
                        max(1, n_total))

            log(f"    {mode:18s}: hall={n_hall}/{n_total} "
                f"abst={n_abst}/{n_total} mixed={n_mixed}/{n_total} "
                f"other={n_other}/{n_total} | "
                f"len={mean_len:.0f} rep={mean_rep:.3f} "
                f"({mode_desc[:40]})")

            if n_total >= 3:
                for i, r in enumerate(results):
                    if r["behavior"] == "skip":
                        continue
                    preview = r["gen_text"][:80].replace("\n", " ")
                    log(f"      [{i}] {r['behavior']:15s} "
                        f"len={r['gen_len']:2d} st={r['steps']:2d} "
                        f"\"{preview}\"")

            token_result["modes"][mode] = {
                "n_total": n_total,
                "hall_rate": f"{n_hall}/{n_total}",
                "abst_rate": f"{n_abst}/{n_total}",
                "mixed_rate": f"{n_mixed}/{n_total}",
                "mean_len": round(mean_len, 1),
                "mean_rep": round(mean_rep, 4),
            }

        all_token_results[token_name] = token_result

    # ================================================================
    # SUMMARY TABLE
    # ================================================================
    log("\n" + "=" * 72)
    log("SUMMARY: Hallucination rate across timing modes")
    log("=" * 72)

    header = f"{'Token':20s}"
    for mode in MODE_LABELS:
        header += f" {'hall/' + mode:>18s}"
    log(header)

    for token_name in primary_tokens[:2]:
        if token_name not in all_token_results:
            continue
        tr = all_token_results[token_name]
        row = f"{token_name:20s}"
        for mode in MODE_LABELS:
            md = tr["modes"].get(mode, {})
            hr = md.get("hall_rate", "N/A")
            row += f" {hr:>18s}"
        log(row)

    # ================================================================
    # PAIRED FLIP ANALYSIS
    # ================================================================
    log("\n" + "=" * 72)
    log("Paired flips: baseline → intervention behavior changes")
    log("=" * 72)

    rank = {"abstention": 0, "other": 1, "mixed": 2, "hallucination": 3}

    for token_name in primary_tokens[:2]:
        if token_name not in all_token_results:
            continue
        samples = token_samples[token_name]
        cv = counter_vectors.get(token_name)

        for mode in MODE_LABELS:
            if mode == "baseline":
                continue

            n_better = 0
            n_worse = 0
            n_same = 0

            for m in samples[:min(len(samples), 8)]:
                prompt = m["prompt"]
                causal_pos_set = set(
                    m["token_positions"].get(token_name, []))
                if not causal_pos_set:
                    continue

                gen_base, _, _ = run_generate_hooked(
                    model, tokenizer, prompt, device, [])

                if mode == "T1_random":
                    rand_vec = torch.randn(d_model)
                    rand_vec = rand_vec / rand_vec.norm()
                    use_cv = rand_vec
                    hook_mode = "T1_continuous"
                elif mode == "T0_ablation":
                    use_cv = None
                    hook_mode = "T0_ablation"
                else:
                    use_cv = cv
                    hook_mode = mode

                vector_dict = {"scale": 1.0, "mode": hook_mode}
                factory_fn, _ = make_hook_factory(
                    causal_pos_set, vector_dict, hook_mode, use_cv, d_model)
                hook_l0 = factory_fn(causal_pos_set, use_cv, "L0")
                gen_intv, _, _ = run_generate_hooked(
                    model, tokenizer, prompt, device, [("L0", hook_l0)])

                ctx = m["sample"].get("context", "")
                q = m["sample"].get("question", "")
                b_base, _ = classify_behavior(gen_base, ctx, q)
                b_intv, _ = classify_behavior(gen_intv, ctx, q)

                r_base = rank.get(b_base, 1)
                r_intv = rank.get(b_intv, 1)

                if r_intv < r_base:
                    n_better += 1
                elif r_intv > r_base:
                    n_worse += 1
                else:
                    n_same += 1

            n_total = n_better + n_worse + n_same
            if n_total > 0:
                log(f"  {token_name:20s} {mode:18s}: "
                    f"better={n_better} worse={n_worse} same={n_same} "
                    f"({n_better/max(1,n_total)*100:.0f}% better)")

    # ================================================================
    # VERDICT
    # ================================================================
    log("\n" + "=" * 72)
    log("P37 判决")
    log("=" * 72)

    any_t1_better = False
    any_t0_better = False
    any_degradation = False
    verdict_lines = []

    for token_name in primary_tokens[:2]:
        if token_name not in all_token_results:
            continue
        tr = all_token_results[token_name]
        base_md = tr["modes"].get("baseline", {})
        t0c_md = tr["modes"].get("T0_counter", {})
        t1c_md = tr["modes"].get("T1_continuous", {})
        t0a_md = tr["modes"].get("T0_ablation", {})

        base_hr = base_md.get("hall_rate", "0/0")
        t0c_hr = t0c_md.get("hall_rate", "0/0")
        t1c_hr = t1c_md.get("hall_rate", "0/0")
        t0a_hr = t0a_md.get("hall_rate", "0/0")

        def parse_rate(r):
            parts = r.split("/") if "/" in r else ["0", "0"]
            return int(parts[0]), int(parts[1])

        bh, bt = parse_rate(base_hr)
        t0h, t0t = parse_rate(t0c_hr)
        t1h, t1t = parse_rate(t1c_hr)
        tah, tat = parse_rate(t0a_hr)

        if t1t > 0 and t1h / t1t < bh / max(1, bt):
            any_t1_better = True
            verdict_lines.append(
                f"  {token_name}: T1_counter ({t1h}/{t1t}) < "
                f"baseline ({bh}/{bt}) ✓")
        elif t1t > 0 and t1h / t1t > bh / max(1, bt):
            any_degradation = True
            verdict_lines.append(
                f"  {token_name}: T1_counter ({t1h}/{t1t}) > "
                f"baseline ({bh}/{bt}) ✗ (degradation)")

        if t0t > 0 and t0h / t0t < bh / max(1, bt):
            any_t0_better = True
            verdict_lines.append(
                f"  {token_name}: T0_counter ({t0h}/{t0t}) < "
                f"baseline ({bh}/{bt}) ✓")

    for vl in verdict_lines:
        log(vl)

    if any_t1_better or any_t0_better:
        log("\n  VERDICT: partial_evidence")
        log("  Counter-vector shows directional improvement over ablation.")
    else:
        log("\n  VERDICT: negative_but_informative")
        log("  Counter-vector injection fails to reduce hallucination "
            "behavior even with continuous T1 timing. ")
        log("  Confirms that the log-prob → behavior gap is not a timing "
            "issue but a structural one. ")
        log("  Recommendation: move to training-time / theta-level "
            "intervention. ")

    # ================================================================
    # SAVE
    # ================================================================
    elapsed = time.time() - t0

    with open(AUDIT_PATH, "w", encoding="utf-8") as f:
        for entry in audit_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    output = {
        "config": {
            "modes": MODE_LABELS,
            "tokens_tested": primary_tokens[:2],
            "random_seed": 42,
            "counter_vector_norm": {
                tn: round(cv.norm().item(), 4)
                for tn, cv in counter_vectors.items()
                if tn in primary_tokens[:2]
            },
        },
        "time_s": round(elapsed, 1),
        "time_min": round(elapsed / 60, 1),
        "token_results": all_token_results,
    }

    with open(os.path.join(RESULTS_DIR, "summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    log(f"\nResults saved to {RESULTS_DIR}/")
    log(f"  - summary.json")
    log(f"  - per_sample_audit.jsonl ({len(audit_entries)} entries)")
    log(f"\nTime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log(f"\nP37 Complete.")


if __name__ == "__main__":
    main()