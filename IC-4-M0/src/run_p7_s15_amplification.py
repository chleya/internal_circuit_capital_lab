"""P7: S15 Amplification Mechanism Investigation.

T2 found that sycophancy probe accuracy peaks at generation step 15 (0.983 at L8),
amplifying from ~0.80. The mechanism behind this amplification is unknown.

P7 investigates three dimensions of S15:

Phase 1: Per-step probe scoring using the P6 behavior-only probe.
  Tracks sycophancy signal through all generation steps, identifying
  when syc vs non-syc sample trajectories diverge.

Phase 2: Token-level analysis around S15.
  What is the model generating at steps 10-20? Is there a linguistic
  pattern (e.g., first substantive content, agreement phrase)?

Phase 3: Per-step steering intervention.
  Tests if steering at S15 is a "sensitive period" — more impactful
  than steering at other steps (5, 10, 20, 25).

Usage:
  python -m src.run_p7_s15_amplification

Outputs:
  results_p7_s15/
    per_step_scores.npz
    per_step_tokens.json
    per_step_steering.npz
  reports/IC4_P7_S15_AMPLIFICATION_REPORT.md
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .model_loader import load_model_and_tokenizer
from .steering import _find_transformer_layer
from .run_p0_sycophancy_contrast import _is_sycophantic

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_log = logging.getLogger("p7_s15")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results_p7_s15"
REPORTS_DIR = PROJECT_ROOT / "reports"
CONTRAST_DATA_PATH = PROJECT_ROOT / "results_p0_sycophancy" / "sycophancy_contrast_data.json"
P6_PROBE_PATH = PROJECT_ROOT / "results_p6_syc_behavior" / "probe_model.pkl"
STEERING_VECTORS_PATH = PROJECT_ROOT / "results_t3_impulse_p4" / "steering_vectors.npz"

PROBE_LAYER = 10
OPTIMAL_ALPHA = -3.0
MAX_NEW_TOKENS = 64
RANDOM_SEED = 42
N_TEST_SAMPLES = 12

TARGET_STEPS = [5, 10, 15, 20, 25]


def _prompt_from_sample(sample: dict) -> str:
    prompt = sample.get("prompt", "")
    if not prompt:
        context = sample.get("context", "")
        question = sample.get("question", "")
        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
    return prompt


def _run_per_step_scoring(model, tokenizer, test_samples, probe):
    """Phase 1: Manual step-by-step generation, collecting:
    - hidden state at L10 for each step
    - token generated at each step
    - probe score at each step
    """
    device = next(model.parameters()).device
    results = []

    for sample in tqdm(test_samples, desc="Per-step scoring"):
        prompt = _prompt_from_sample(sample)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        step_hidden_states = []
        step_tokens = []
        step_probe_scores = []

        with torch.no_grad():
            output = model(**inputs, output_hidden_states=True, use_cache=True)
            logits = output.logits[:, -1, :]
            past_key_values = output.past_key_values
            next_token_id = torch.argmax(logits, dim=-1)

        step_tokens.append(int(next_token_id[0]))
        step_hidden_states.append(None)
        step_probe_scores.append(None)

        generated_ids = [next_token_id[0]]
        hidden_states_buffer = []

        target_module = _find_transformer_layer(model, PROBE_LAYER)

        def _capture_hook(module, inputs_tup, output):
            if isinstance(output, tuple):
                hs = output[0]
            else:
                hs = output
            hidden_states_buffer.append(hs[:, -1, :].detach().cpu().numpy().copy())

        handle = target_module.register_forward_hook(_capture_hook)

        try:
            for step_idx in range(1, MAX_NEW_TOKENS):
                current_input = next_token_id.unsqueeze(0)
                hidden_states_buffer.clear()

                with torch.no_grad():
                    output = model(
                        input_ids=current_input,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

                logits = output.logits[:, -1, :]
                past_key_values = output.past_key_values
                next_token_id = torch.argmax(logits, dim=-1)

                if hidden_states_buffer:
                    hs = hidden_states_buffer[0]
                    score = float(probe.predict_proba(hs)[0, 1])
                    step_hidden_states.append(hs)
                    step_probe_scores.append(score)
                else:
                    step_hidden_states.append(None)
                    step_probe_scores.append(None)

                step_tokens.append(int(next_token_id[0]))
                generated_ids.append(next_token_id[0])
        finally:
            handle.remove()

        full_output_ids = torch.cat([
            inputs["input_ids"][0],
            torch.tensor(generated_ids, device=device),
        ])
        full_text = tokenizer.decode(full_output_ids, skip_special_tokens=True)
        generated_text = tokenizer.decode(
            torch.tensor(generated_ids, device="cpu"), skip_special_tokens=True
        )

        is_syc = _is_sycophantic(generated_text)

        results.append({
            "sample_id": sample.get("tid", "unknown"),
            "prompt": prompt,
            "generated_text": generated_text,
            "is_sycophantic": is_syc,
            "n_steps": len(step_tokens),
            "step_tokens": step_tokens,
            "step_token_strs": [
                tokenizer.decode([tid]) if tid is not None else ""
                for tid in step_tokens
            ],
            "step_probe_scores": step_probe_scores,
            "step_hidden_states": [hs.tolist() if hs is not None else None for hs in step_hidden_states],
            "input_len": input_len,
        })

    return results


def _run_per_step_steering(model, tokenizer, test_samples, v_syc):
    """Phase 3: Test steering at specific generation steps.

    For each target step (5, 10, 15, 20, 25), apply v_syc alpha=-3.0
    injection at that step only. Compare syc_rate vs baseline and vs
    always-on open-loop.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    steering_tensor = torch.tensor(v_syc, dtype=dtype, device=device)

    all_results = {}

    for target_step in TARGET_STEPS:
        results = []
        target_module = _find_transformer_layer(model, PROBE_LAYER)

        for sample in tqdm(test_samples, desc=f"Steering S{target_step}"):
            prompt = _prompt_from_sample(sample)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            step_counter = [0]
            steered = [False]

            def _step_steering_hook(module, input_tup, output):
                step_counter[0] += 1
                if step_counter[0] == target_step + 1:
                    steered[0] = True
                    if isinstance(output, tuple):
                        modified = (output[0] + OPTIMAL_ALPHA * steering_tensor,) + output[1:]
                        return modified
                    else:
                        return output + OPTIMAL_ALPHA * steering_tensor
                return output

            handle = target_module.register_forward_hook(_step_steering_hook)

            try:
                with torch.no_grad():
                    gen_outputs = model.generate(
                        **inputs, max_new_tokens=MAX_NEW_TOKENS,
                        pad_token_id=tokenizer.eos_token_id,
                    )
            finally:
                handle.remove()

            generated = gen_outputs[0][inputs["input_ids"].shape[1]:]
            answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
            results.append({
                "sample_id": sample.get("tid", "unknown"),
                "generated_output": answer,
                "is_sycophantic": _is_sycophantic(answer),
                "steered": steered[0],
                "step_hit": step_counter[0],
                "target_step": target_step,
            })

        syc_rate = sum(1 for r in results if r["is_sycophantic"]) / len(results) if results else 0.0
        all_results[target_step] = {
            "target_step": target_step,
            "syc_rate": syc_rate,
            "n_samples": len(results),
            "results": results,
        }

    return all_results


def _analyze_per_step_scores(per_step_results, baseline_syc_rate):
    """Analyze Phase 1 results: find divergence point, S15 patterns."""
    syc_samples = [r for r in per_step_results if r["is_sycophantic"]]
    non_syc_samples = [r for r in per_step_results if not r["is_sycophantic"]]

    max_steps = max(r["n_steps"] for r in per_step_results)

    syc_scores_by_step = defaultdict(list)
    non_syc_scores_by_step = defaultdict(list)

    for r in per_step_results:
        for si in range(1, r["n_steps"]):
            score = r["step_probe_scores"][si]
            if score is not None:
                if r["is_sycophantic"]:
                    syc_scores_by_step[si].append(score)
                else:
                    non_syc_scores_by_step[si].append(score)

    per_step_stats = []
    divergence_step = None

    for s in range(1, max_steps):
        syc_vals = syc_scores_by_step.get(s, [])
        non_syc_vals = non_syc_scores_by_step.get(s, [])

        syc_mean = float(np.mean(syc_vals)) if syc_vals else None
        non_syc_mean = float(np.mean(non_syc_vals)) if non_syc_vals else None
        separation = (syc_mean - non_syc_mean) if (syc_mean is not None and non_syc_mean is not None) else None

        per_step_stats.append({
            "step": s,
            "syc_mean": syc_mean,
            "non_syc_mean": non_syc_mean,
            "separation": separation,
            "n_syc": len(syc_vals),
            "n_non_syc": len(non_syc_vals),
        })

        if separation is not None and divergence_step is None and separation > 0.15:
            divergence_step = s

    s15_stats = per_step_stats[14] if len(per_step_stats) > 14 else None

    return {
        "per_step_stats": per_step_stats,
        "divergence_step": divergence_step,
        "s15_stats": s15_stats,
        "n_syc": len(syc_samples),
        "n_non_syc": len(non_syc_samples),
    }


def _generate_report(
    baseline_syc_rate: float,
    per_step_results: List[dict],
    step_analysis: dict,
    steering_results: dict,
    report_path: Path,
):
    lines = []
    lines.append("# IC-4 P7: S15 Amplification Mechanism Investigation")
    lines.append("")
    lines.append(f"> **Date**: {time.strftime('%Y-%m-%d')} | **Status**: Completed")
    lines.append(f"> **Predecessor**: T2 Decision Heatmap — S15 peak accuracy=0.983")
    lines.append(f"> **Layer**: {PROBE_LAYER} | **Baseline syc**: {baseline_syc_rate:.4f}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Motivation")
    lines.append("")
    lines.append("T2 discovered that sycophancy probe accuracy peaks at generation step 15")
    lines.append("(0.983 at L8), amplifying from ~0.80 in earlier steps. This amplification")
    lines.append("mechanism is unexplained — P7 investigates three dimensions:")
    lines.append("")
    lines.append("1. **Per-step probe scoring**: When does the P6 behavior-only probe detect")
    lines.append("   sycophancy tendency through generation steps?")
    lines.append("2. **Token-level analysis**: What tokens are generated around S15?")
    lines.append("3. **Per-step steering**: Is S15 a \"sensitive period\" for intervention?")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 2. Phase 1: Per-Step Probe Scoring")
    lines.append("")
    lines.append(f"Test samples: {len(per_step_results)} "
                 f"(syc={step_analysis['n_syc']}, non-syc={step_analysis['n_non_syc']})")
    lines.append("")

    lines.append("### 2.1 Per-Step Syc vs Non-Syc Probe Scores")
    lines.append("")
    lines.append("| Step | Syc μ | Non-Syc μ | Separation | N Syc | N NonSyc |")
    lines.append("|---|---|---|---|---|---|")

    for ps in step_analysis["per_step_stats"]:
        if ps["step"] > 30:
            break
        syc_str = f"{ps['syc_mean']:.4f}" if ps['syc_mean'] is not None else "N/A"
        non_str = f"{ps['non_syc_mean']:.4f}" if ps['non_syc_mean'] is not None else "N/A"
        sep_str = f"{ps['separation']:+.4f}" if ps['separation'] is not None else "N/A"
        marker = " ←" if ps["step"] == 15 else ""
        lines.append(f"| {ps['step']}{marker} | {syc_str} | {non_str} | {sep_str} | {ps['n_syc']} | {ps['n_non_syc']} |")

    lines.append("")

    if step_analysis["divergence_step"] is not None:
        lines.append(f"**Divergence step**: {step_analysis['divergence_step']} "
                      f"(first step where syc−non_syc separation > 0.15)")
        lines.append("")

    s15 = step_analysis.get("s15_stats")
    if s15:
        lines.append("### 2.2 S15 Statistics")
        lines.append("")
        lines.append(f"- S15 syc mean: **{s15['syc_mean']:.4f}**" if s15['syc_mean'] else "- S15 syc mean: N/A")
        lines.append(f"- S15 non-syc mean: **{s15['non_syc_mean']:.4f}**" if s15['non_syc_mean'] else "- S15 non-syc mean: N/A")
        lines.append(f"- S15 separation: **{s15['separation']:+.4f}**" if s15['separation'] else "- S15 separation: N/A")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 3. Phase 2: Token Analysis Around S15")
    lines.append("")
    lines.append("Tokens generated at steps 10-20 for each sample, with probe scores.")
    lines.append("")

    for r in per_step_results:
        tid = r["sample_id"]
        is_syc = "🟠 SYC" if r["is_sycophantic"] else "🟢 NON"
        lines.append(f"### Sample {tid} ({is_syc})")
        lines.append("")
        lines.append("| Step | Token | Probe Score |")
        lines.append("|---|---|---|")

        for si in range(10, min(21, r["n_steps"])):
            tok_str = r["step_token_strs"][si] if si < len(r["step_token_strs"]) else "?"
            tok_display = tok_str.replace("\n", "\\n").replace("|", "\\|")[:30]
            score = r["step_probe_scores"][si] if si < len(r["step_probe_scores"]) else None
            score_str = f"{score:.4f}" if score is not None else "N/A"
            marker = " **← S15**" if si == 15 else ""
            lines.append(f"| {si}{marker} | `{tok_display}` | {score_str} |")

        lines.append("")
        lines.append(f"Full output: `{r['generated_text'][:200]}...`")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 4. Phase 3: Per-Step Steering")
    lines.append("")
    lines.append(f"| Target Step | Syc Rate | Δ from Baseline |")
    lines.append("|---|---|---|")

    for step in TARGET_STEPS:
        if step in steering_results:
            sr = steering_results[step]
            delta = sr["syc_rate"] - baseline_syc_rate
            delta_pct = delta / baseline_syc_rate * 100 if baseline_syc_rate > 0 else 0.0
            lines.append(f"| S{step} | {sr['syc_rate']:.4f} | {delta:+.4f} ({delta_pct:+.1f}%) |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. Interpretation")
    lines.append("")

    div = step_analysis.get("divergence_step")
    if div is not None:
        lines.append(f"**Divergence occurs at step {div}.** The P6 behavior-only probe")
        lines.append(f"first detects syc vs non-syc separation (>0.15) at generation step {div}.")
        if div <= 15:
            lines.append("This is consistent with T2's S15 amplification finding — the model's")
            lines.append("sycophancy tendency becomes readable in hidden states during mid-generation.")
        else:
            lines.append(f"Divergence at step {div} is later than T2's S15 peak. This may reflect")
            lines.append("differences between the P6 behavior-only probe and T2's per-position probes.")
        lines.append("")

    best_step = None
    best_delta = 0.0
    for step in TARGET_STEPS:
        if step in steering_results:
            delta = steering_results[step]["syc_rate"] - baseline_syc_rate
            if delta < best_delta:
                best_delta = delta
                best_step = step

    if best_step is not None:
        lines.append(f"**Steering is most effective at S{best_step}** (Δ={best_delta:+.4f}).")
        if best_step == 15:
            lines.append("This CONFIRMS S15 as a sensitive period — steering at this step")
            lines.append("produces the largest sycophancy reduction.")
        else:
            lines.append(f"Surprisingly, S{best_step} not S15 is the most sensitive step.")
            lines.append("S15 may be when the signal is most *readable* (T2 peak), but")
            lines.append("the causally sensitive period may be elsewhere.")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 6. Next Steps")
    lines.append("")
    lines.append("| Priority | Action | Detail |")
    lines.append("|---|---|---|")
    lines.append("| P8 | Larger-scale replication | 24-36 samples for statistical robustness |")
    lines.append("| P9 | Cross-Bottleneck | stabilization + organization joint intervention |")
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    _log.info(f"Report: {report_path}")


def main():
    _log.info("P7: S15 Amplification Mechanism Investigation")

    _log.info("Loading contrast data...")
    with open(CONTRAST_DATA_PATH, "r", encoding="utf-8") as f:
        contrast_data = json.load(f)

    standard_samples = [s for s in contrast_data if not s.get("system_prompt")]
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(len(standard_samples))
    test_idx = indices[18:18 + N_TEST_SAMPLES]
    test_samples = [standard_samples[i] for i in test_idx]
    _log.info(f"Test samples: {len(test_samples)}")

    _log.info("Loading P6 probe...")
    with open(P6_PROBE_PATH, "rb") as f:
        probe = pickle.load(f)

    _log.info("Loading model...")
    model, tokenizer = load_model_and_tokenizer()

    _log.info("Phase 1: Per-step probe scoring (manual step-by-step generation)")
    per_step_results = _run_per_step_scoring(model, tokenizer, test_samples, probe)

    syc_count = sum(1 for r in per_step_results if r["is_sycophantic"])
    baseline_syc_rate = syc_count / len(per_step_results)
    _log.info(f"Baseline syc rate: {baseline_syc_rate:.4f} ({syc_count}/{len(per_step_results)})")

    step_analysis = _analyze_per_step_scores(per_step_results, baseline_syc_rate)
    _log.info(f"Divergence step: {step_analysis['divergence_step']}")

    s15 = step_analysis.get("s15_stats")
    if s15:
        _log.info(f"S15: syc_mu={s15['syc_mean']}, non_syc_mu={s15['non_syc_mean']}, "
                   f"separation={s15['separation']}")

    _log.info("Loading steering vectors...")
    sv_data = np.load(STEERING_VECTORS_PATH)
    v_syc = sv_data["v_syc"].astype(np.float32)

    _log.info("Phase 3: Per-step steering")
    steering_results = _run_per_step_steering(model, tokenizer, test_samples, v_syc)

    for step in TARGET_STEPS:
        if step in steering_results:
            sr = steering_results[step]
            delta = sr["syc_rate"] - baseline_syc_rate
            _log.info(f"  S{step}: syc_rate={sr['syc_rate']:.4f} (delta={delta:+.4f})")

    _log.info("Saving results...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    np.savez(
        RESULTS_DIR / "per_step_scores.npz",
        per_step_stats=step_analysis["per_step_stats"],
        divergence_step=step_analysis["divergence_step"],
        baseline_syc_rate=baseline_syc_rate,
        allow_pickle=True,
    )

    per_step_tokens_data = []
    for r in per_step_results:
        per_step_tokens_data.append({
            k: v for k, v in r.items()
            if k not in ("step_hidden_states",)
        })

    with open(RESULTS_DIR / "per_step_tokens.json", "w", encoding="utf-8") as f:
        json.dump(per_step_tokens_data, f, ensure_ascii=False, indent=2)

    steering_save = {}
    for step, sr in steering_results.items():
        steering_save[str(step)] = {
            k: v for k, v in sr.items() if k != "results"
        }
    np.savez(
        RESULTS_DIR / "per_step_steering.npz",
        steering_results=steering_save,
        baseline_syc_rate=baseline_syc_rate,
        allow_pickle=True,
    )

    report_path = REPORTS_DIR / "IC4_P7_S15_AMPLIFICATION_REPORT.md"
    _generate_report(
        baseline_syc_rate, per_step_results, step_analysis, steering_results, report_path,
    )

    _log.info("P7 complete.")


if __name__ == "__main__":
    main()