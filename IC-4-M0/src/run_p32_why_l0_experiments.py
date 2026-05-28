"""
P32+P33+P34+P35: 探索"为什么 L0 是信息入口" — 四合一深度实验
=============================================================
P31 确认: L0 主导是一般规律 (funding + r_and_d_spend 皆然)
P32 回答: 为什么是 L0？→ 测试"嵌入 → L0"是否为唯一的信息编码路径
P33 回答: L0 在做什么？→ 提取 L0 self_attn 注意力模式
P34 回答: 为什么 r_and_d_spend 效应更小？→ 递进式 token 位点消融
P35 回答: 为什么 L16 是次高峰？→ 分离 L16 self_attn vs mlp

核心假设:
  H1: L0 是 embedding → residual 的唯一信息编码器
      预测: Embedding ablation ≈ L0 combined ablation
      预测: Embedding + L0 combined ≈ Embedding alone (ceiling)
  H2: L0 的 self_attn 在 causal token 上高度自聚焦
      预测: L0 attention[token_pos, token_pos] >> attention[token_pos, others]
  H3: Multi-token 效应小于 single-token 因为信息被稀释
      预测: progressive sub-token ablation 效应累加

Usage:
  python src/run_p32_why_l0_experiments.py
"""

import os, sys, time, json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_builder import load_jsonl

RESULTS_DIR = "results_p32_why_l0"
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, "run_log.txt")

N_LAYERS = 24
SCALE_ZERO = 0.0


def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def find_first_token_position(prompt, tokenizer, target):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())
    for variant in [target, " " + target]:
        variant_tokens = tokenizer.tokenize(variant)
        n = len(variant_tokens)
        if n == 0:
            continue
        for i in range(len(tokens) - n + 1):
            if tokens[i:i + n] == variant_tokens:
                return list(range(i, i + n)), variant_tokens
    return None, None


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


def run_forward(model, tokenizer, prompt, response, device, hooks_config, return_attn=False):
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
            outputs = model(input_ids=input_ids, labels=labels,
                            output_attentions=return_attn)
        if return_attn:
            return -outputs.loss.item(), outputs.attentions
        return -outputs.loss.item(), None
    finally:
        for h in handles:
            h.remove()


def compute_lp_diff(model, tokenizer, prompt, pos_resp, neg_resp, device,
                    hooks_config=None):
    if hooks_config is None:
        hooks_config = []
    lp_pos, _ = run_forward(model, tokenizer, prompt, pos_resp, device, hooks_config)
    lp_neg, _ = run_forward(model, tokenizer, prompt, neg_resp, device, hooks_config)
    return lp_pos - lp_neg


def compute_lp_diff_with_attn(model, tokenizer, prompt, pos_resp, neg_resp, device,
                               hooks_config=None):
    if hooks_config is None:
        hooks_config = []
    lp_pos, attn_pos = run_forward(model, tokenizer, prompt, pos_resp, device,
                                    hooks_config, return_attn=True)
    lp_neg, attn_neg = run_forward(model, tokenizer, prompt, neg_resp, device,
                                    hooks_config, return_attn=True)
    return lp_pos - lp_neg, attn_pos, attn_neg


def main():
    log("=" * 64)
    log("P32+P33+P34+P35: 探索'为什么 L0 是信息入口'")
    log("=" * 64)
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

    log("\n[Step 2] Loading samples & computing baselines...")
    pos_dir = os.path.join(base_dir, "data_position_sensitivity", "s0")
    all_samples = []
    for pos in ["early", "mid", "late"]:
        path = os.path.join(pos_dir, f"test_{pos}_s0.jsonl")
        if os.path.exists(path):
            for s in load_jsonl(path)[:10]:
                s["_position"] = pos
                all_samples.append(s)

    samples_meta = []
    for idx, sample in enumerate(all_samples):
        ctx = sample.get("context", "")
        q = sample.get("question", "")
        prompt = f"{ctx}\n\nQuestion: {q}\nAnswer:"
        pos_resp = sample.get("positive_response", "")
        neg_resp = sample.get("negative_response", "")
        lp_diff = compute_lp_diff(model, tokenizer, prompt, pos_resp, neg_resp, device)
        is_hall = (lp_diff > 0 and sample.get("answerability") == "unanswerable")

        f_pos, f_tok = find_first_token_position(prompt, tokenizer, "funding")
        r_pos, r_tok = find_first_token_position(prompt, tokenizer, "r_and_d_spend")

        samples_meta.append({
            "idx": idx, "sample": sample, "prompt": prompt,
            "pos_resp": pos_resp, "neg_resp": neg_resp,
            "lp_diff": lp_diff, "is_hall": is_hall,
            "funding_pos": f_pos, "funding_tok": f_tok,
            "r_and_d_pos": r_pos, "r_and_d_tok": r_tok,
        })

    funding_hall = [m for m in samples_meta if m["is_hall"] and m["funding_pos"]]
    r_and_d_hall = [m for m in samples_meta if m["is_hall"] and m["r_and_d_pos"]]

    log(f"  funding hallucinated: {len(funding_hall)}")
    log(f"  r_and_d_spend hallucinated: {len(r_and_d_hall)}")

    # ================================================================
    # P32: EMBEDDING-LEVEL ABLATION
    # ================================================================
    log("\n" + "=" * 64)
    log("P32: EMBEDDING-LEVEL ABLATION — 嵌入层 vs L0 对比")
    log("=" * 64)

    p32_results = []
    for m in funding_hall[:3]:
        prompt = m["prompt"]
        pos_resp = m["pos_resp"]
        neg_resp = m["neg_resp"]
        pos_set = set(m["funding_pos"])
        base = m["lp_diff"]

        embed_hook = build_pos_hook(pos_set, SCALE_ZERO)
        l0_combined_hooks = [
            ("L0.self_attn", build_pos_hook(pos_set, SCALE_ZERO)),
            ("L0.mlp", build_pos_hook(pos_set, SCALE_ZERO)),
        ]
        embed_plus_l0_hooks = [
            ("embed", build_pos_hook(pos_set, SCALE_ZERO)),
            ("L0.self_attn", build_pos_hook(pos_set, SCALE_ZERO)),
            ("L0.mlp", build_pos_hook(pos_set, SCALE_ZERO)),
        ]

        d_embed = base - compute_lp_diff(model, tokenizer, prompt, pos_resp, neg_resp,
                                          device, [("embed", embed_hook)])
        d_l0 = base - compute_lp_diff(model, tokenizer, prompt, pos_resp, neg_resp,
                                       device, l0_combined_hooks)
        d_embed_l0 = base - compute_lp_diff(model, tokenizer, prompt, pos_resp, neg_resp,
                                             device, embed_plus_l0_hooks)

        log(f"\n  Sample {m['idx']} [{m['sample']['_position']}] base={base:+.4f}")
        log(f"    Embedding-only:   Δembed = {d_embed:+.4f}")
        log(f"    L0-combined-only: ΔL0    = {d_l0:+.4f}")
        log(f"    Embed+L0-combined:Δboth  = {d_embed_l0:+.4f}")
        log(f"    Δembed / ΔL0     = {d_embed/max(0.0001,d_l0):.1f}x")
        log(f"    Δboth / Δembed   = {d_embed_l0/max(0.0001,d_embed):.2f}x  (ceiling check)")

        p32_results.append({
            "sample_idx": m["idx"], "position": m["sample"]["_position"],
            "baseline": round(base, 6),
            "delta_embed": round(d_embed, 6),
            "delta_l0_combined": round(d_l0, 6),
            "delta_embed_l0_combined": round(d_embed_l0, 6),
        })

    # ================================================================
    # P33: L0 ATTENTION PATTERN ANALYSIS
    # ================================================================
    log("\n" + "=" * 64)
    log("P33: L0 ATTENTION PATTERN ANALYSIS")
    log("=" * 64)

    p33_results = []
    for m in funding_hall[:1]:
        prompt = m["prompt"]
        pos_resp = m["pos_resp"]
        neg_resp = m["neg_resp"]
        pos_set = set(m["funding_pos"])
        causal_pos = list(pos_set)[0]

        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        prompt_tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())

        _, attn_pos, attn_neg = compute_lp_diff_with_attn(
            model, tokenizer, prompt, pos_resp, neg_resp, device)

        l0_attn = attn_pos[0][0].detach().cpu().numpy()
        n_heads = l0_attn.shape[0]
        seq_len = l0_attn.shape[1]

        log(f"\n  Sample {m['idx']}: causal token '{m['funding_tok']}' at pos={causal_pos}")
        log(f"  L0 attention: {n_heads} heads × {seq_len} tokens")

        for head_idx in range(n_heads):
            attn_to_self = l0_attn[head_idx, causal_pos, causal_pos]
            top5_indices = l0_attn[head_idx, causal_pos, :].argsort()[-5:][::-1]
            top5_values = l0_attn[head_idx, causal_pos, top5_indices]
            top5_str = ", ".join(
                f"pos{i}({prompt_tokens[i][:8]}:{v:.3f})"
                for i, v in zip(top5_indices, top5_values)
            )
            log(f"    Head {head_idx}: self={attn_to_self:.3f}  top5: {top5_str}")

        head_self_attn = [float(l0_attn[h, causal_pos, causal_pos]) for h in range(n_heads)]
        mean_self = sum(head_self_attn) / n_heads
        log(f"    Mean self-attention@causal_pos: {mean_self:.4f}")

        p33_results.append({
            "sample_idx": m["idx"],
            "causal_pos": causal_pos,
            "n_heads": n_heads, "seq_len": seq_len,
            "head_self_attn": head_self_attn,
            "mean_self_attn": round(mean_self, 6),
        })

    # ================================================================
    # P34: TOKENIZATION DECOMPOSITION — progressive sub-token ablation
    # ================================================================
    log("\n" + "=" * 64)
    log("P34: TOKENIZATION DECOMPOSITION — progressive sub-token ablation")
    log("=" * 64)

    p34_results = []
    for label, m_list, token_key in [
        ("funding (1 sub-token)", funding_hall[:1], "funding_pos"),
        ("r_and_d_spend (5 sub-tokens)", r_and_d_hall[:1], "r_and_d_pos"),
    ]:
        if not m_list:
            continue
        m = m_list[0]
        prompt = m["prompt"]
        pos_resp = m["pos_resp"]
        neg_resp = m["neg_resp"]
        base = m["lp_diff"]
        all_pos = m[token_key]
        token_name = "funding" if token_key == "funding_pos" else "r_and_d_spend"

        log(f"\n  [{label}] Sample {m['idx']}")
        log(f"    token positions: {all_pos} ({len(all_pos)} sub-tokens)")

        progressive = []
        for k in range(1, len(all_pos) + 1):
            subset = set(all_pos[:k])
            hooks = [
                ("L0.self_attn", build_pos_hook(subset, SCALE_ZERO)),
                ("L0.mlp", build_pos_hook(subset, SCALE_ZERO)),
            ]
            d = base - compute_lp_diff(model, tokenizer, prompt, pos_resp, neg_resp,
                                        device, hooks)
            progressive.append(round(d, 6))
            log(f"      ablate {k}/{len(all_pos)} sub-tokens → Δ={d:+.4f}")

        log(f"    full token Δ = {progressive[-1]:+.4f}")
        if len(all_pos) > 1:
            per_sub = progressive[-1] / len(all_pos)
            log(f"    per-sub-token contribution = {per_sub:+.4f}")

        p34_results.append({
            "token": token_name, "sample_idx": m["idx"],
            "n_sub_tokens": len(all_pos), "positions": all_pos,
            "progressive_deltas": progressive,
            "full_delta": progressive[-1] if progressive else 0,
        })

    # ================================================================
    # P35: L16 SELF_ATTN vs MLP SEPARATION
    # ================================================================
    log("\n" + "=" * 64)
    log("P35: L16 次高峰 — self_attn vs mlp 分离")
    log("=" * 64)

    p35_results = []
    for m in funding_hall[:3]:
        prompt = m["prompt"]
        pos_resp = m["pos_resp"]
        neg_resp = m["neg_resp"]
        pos_set = set(m["funding_pos"])
        base = m["lp_diff"]

        d_attn = base - compute_lp_diff(model, tokenizer, prompt, pos_resp, neg_resp,
                                         device, [("L16.self_attn", build_pos_hook(pos_set, SCALE_ZERO))])
        d_mlp = base - compute_lp_diff(model, tokenizer, prompt, pos_resp, neg_resp,
                                        device, [("L16.mlp", build_pos_hook(pos_set, SCALE_ZERO))])
        d_combined = base - compute_lp_diff(model, tokenizer, prompt, pos_resp, neg_resp,
                                             device, [
                                                 ("L16.self_attn", build_pos_hook(pos_set, SCALE_ZERO)),
                                                 ("L16.mlp", build_pos_hook(pos_set, SCALE_ZERO)),
                                             ])

        log(f"\n  Sample {m['idx']}: L16 self_attn={d_attn:+.4f}  mlp={d_mlp:+.4f}  combined={d_combined:+.4f}")
        dominant = "self_attn" if abs(d_attn) > abs(d_mlp) else "mlp"
        log(f"    Dominant: {dominant}  attn/mlp ratio = {abs(d_attn)/max(0.0001,abs(d_mlp)):.1f}x")

        p35_results.append({
            "sample_idx": m["idx"], "position": m["sample"]["_position"],
            "l16_self_attn": round(d_attn, 6),
            "l16_mlp": round(d_mlp, 6),
            "l16_combined": round(d_combined, 6),
            "dominant": dominant,
        })

    elapsed = time.time() - t0

    # ================================================================
    # SUMMARY
    # ================================================================
    log(f"\n{'=' * 64}")
    log(f"[P32-P35 总结] 耗时: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log(f"{'=' * 64}")

    log(f"\n  === P32: Embedding vs L0 ===")
    for r in p32_results:
        log(f"  Sample {r['sample_idx']}: Δembed={r['delta_embed']:+.4f}  "
            f"ΔL0={r['delta_l0_combined']:+.4f}  Δboth={r['delta_embed_l0_combined']:+.4f}")
    if p32_results:
        mean_embed = sum(r["delta_embed"] for r in p32_results) / len(p32_results)
        mean_l0 = sum(r["delta_l0_combined"] for r in p32_results) / len(p32_results)
        mean_both = sum(r["delta_embed_l0_combined"] for r in p32_results) / len(p32_results)
        log(f"  Mean: Δembed={mean_embed:+.4f} ΔL0={mean_l0:+.4f} Δboth={mean_both:+.4f}")
        log(f"  Δembed/ΔL0 = {mean_embed/max(0.0001,mean_l0):.1f}x")
        log(f"  Δboth/Δembed = {mean_both/max(0.0001,mean_embed):.2f}x (ceiling: close to 1.0 = perfect)")
        if abs(mean_both - mean_embed) < 0.02:
            log(f"  → Embed+L0 ≈ Embed alone → L0 adds no new info beyond embedding!")
            log(f"  → L0 is the embedding→residual processor, not an independent encoder")
        if abs(mean_embed - mean_l0) < 0.05:
            log(f"  → Embed ≈ L0 combined → confirming L0 processes raw embedding signal")

    log(f"\n  === P33: L0 Attention ===")
    if p33_results:
        r = p33_results[0]
        log(f"  Mean self-attention at causal token: {r['mean_self_attn']:.4f}")
        log(f"  Per-head: {[round(h, 4) for h in r['head_self_attn']]}")

    log(f"\n  === P34: Tokenization ===")
    for r in p34_results:
        log(f"  {r['token']}: {r['n_sub_tokens']} sub-tokens → full Δ={r['full_delta']:+.4f}")
        log(f"    progressive: {r['progressive_deltas']}")
        if r["n_sub_tokens"] > 1:
            per_sub = r["full_delta"] / r["n_sub_tokens"]
            log(f"    per-sub: {per_sub:+.4f}")

    log(f"\n  === P35: L16 Decomposition ===")
    for r in p35_results:
        log(f"  Sample {r['sample_idx']}: self_attn={r['l16_self_attn']:+.4f}  "
            f"mlp={r['l16_mlp']:+.4f}  combined={r['l16_combined']:+.4f}  "
            f"dominant={r['dominant']}")

    # ================================================================
    # VERDICT
    # ================================================================
    log(f"\n{'=' * 64}")
    log(f"综合判决: 为什么 L0 是信息入口？")
    log(f"{'=' * 64}")

    if p32_results:
        log(f"\n  [H1] Embedding ≈ L0 combined?")
        if abs(mean_embed - mean_l0) < 0.05:
            log(f"    ✓ YES — Δembed={mean_embed:+.3f} ≈ ΔL0={mean_l0:+.3f}")
            log(f"    L0 直接处理 embedding 层输出，是第一个也是唯一的信息编码层")
        else:
            log(f"    ✗ PARTIAL — Δembed={mean_embed:+.3f} vs ΔL0={mean_l0:+.3f}")

        log(f"\n  [H2] Embed+L0 ceiling ≈ Embed alone?")
        if abs(mean_both - mean_embed) < 0.02:
            log(f"    ✓ YES — Δboth={mean_both:+.3f} ≈ Δembed={mean_embed:+.3f}")
            log(f"    L0 没有超出 embedding 的额外编码能力 → 它不是独立编码器")
        else:
            log(f"    ✗ NO — Δboth={mean_both:+.3f} ≠ Δembed={mean_embed:+.3f}")

    if p34_results:
        log(f"\n  [H3] Multi-token 效应稀释?")
        if len(p34_results) >= 2:
            single = next(r for r in p34_results if r["n_sub_tokens"] == 1)
            multi = next(r for r in p34_results if r["n_sub_tokens"] > 1)
            ratio = single["full_delta"] / max(0.0001, multi["full_delta"])
            per_single = single["full_delta"]
            per_multi_sub = multi["full_delta"] / multi["n_sub_tokens"]
            log(f"    funding (1 sub-token): Δ={single['full_delta']:+.4f}")
            log(f"    r_and_d (5 sub-tokens): Δ={multi['full_delta']:+.4f}")
            log(f"    per-sub for r_and_d: {per_multi_sub:+.4f}")
            log(f"    single>multi ratio: {ratio:.1f}x")
            if ratio > 1.5:
                log(f"    → Multi-token tokenization稀释信息浓度，解释了效应差异")

    if p35_results:
        log(f"\n  [H4] L16 次高峰来源?")
        l16_attn_mean = sum(r["l16_self_attn"] for r in p35_results) / len(p35_results)
        l16_mlp_mean = sum(r["l16_mlp"] for r in p35_results) / len(p35_results)
        log(f"    L16 self_attn mean={l16_attn_mean:+.4f}  mlp mean={l16_mlp_mean:+.4f}")
        if l16_mlp_mean > l16_attn_mean * 1.5:
            log(f"    → L16 peak from FFN (知识检索阶段)")
        elif l16_attn_mean > l16_mlp_mean * 1.5:
            log(f"    → L16 peak from self_attn (上下文整合)")
        else:
            log(f"    → L16 peak from both attn+ffn (混合贡献)")

    audit_data = {
        "config": {"methods": ["P32 embedding ablation", "P33 L0 attention",
                                "P34 tokenization decomposition", "P35 L16 decomposition"]},
        "time_s": round(elapsed, 1),
        "p32_embedding_vs_l0": p32_results,
        "p33_l0_attention": p33_results,
        "p34_tokenization": p34_results,
        "p35_l16_decomposition": p35_results,
    }

    with open(os.path.join(RESULTS_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)

    log(f"\nResults saved to {RESULTS_DIR}/results.json")
    log(f"\nP32-P35 Complete.")


if __name__ == "__main__":
    main()