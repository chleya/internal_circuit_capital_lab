"""
P13: Energy vs Direction Asymmetry for Absorption (Fast Representational Test).
===============================================================================
Uses standalone forward pass (model(**inputs)) to compare how energy and
directional perturbations differently affect hidden states at different positions.

This is ~100x faster than generation and directly tests the core hypothesis:
  Energy perturbation → uniform shift across all positions
  Direction perturbation → asymmetric shift (position-dependent)

Usage:
  cd F:/internal_circuit_capital_lab/IC-4-M0
  python src/run_p13_energy_vs_direction.py --n 10 --layer 10
"""

import argparse, os, sys, time, json, csv
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_loader import load_model_and_tokenizer
from src.data_builder import load_jsonl

RESULTS_DIR = "results_p13_energy_vs_direction"
os.makedirs(RESULTS_DIR, exist_ok=True)


def log(msg):
    print(msg, flush=True)
    with open(os.path.join(RESULTS_DIR, "run_log.txt"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()


def collect_perturbed_hidden_states(model, tokenizer, samples, layer_idx, device,
                                     pert_type, pert_value, steering_vector=None):
    target_module = None
    for name, module in model.named_modules():
        if name.endswith(f"model.layers.{layer_idx}"):
            target_module = module
            break

    model_dtype = next(model.parameters()).dtype

    X_list, pos_labels, captured = [], [], []

    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
        captured.append(hs[0, -1, :].detach().cpu().numpy().copy())

    if pert_type == "noise" and pert_value > 0.0:
        def perturb_hook(module, input, output):
            if isinstance(output, tuple):
                hs = output[0]
                hs_pert = hs + torch.randn_like(hs) * pert_value
                captured.append(hs_pert[0, -1, :].detach().cpu().numpy().copy())
                return (hs_pert,) + output[1:]
            hs_pert = output + torch.randn_like(output) * pert_value
            captured.append(hs_pert[0, -1, :].detach().cpu().numpy().copy())
            return hs_pert
        handle = target_module.register_forward_hook(perturb_hook)
    elif pert_type == "direction" and pert_value != 0.0 and steering_vector is not None:
        sv = torch.tensor(steering_vector, dtype=model_dtype).to(device)
        def perturb_hook(module, input, output):
            if isinstance(output, tuple):
                hs = output[0]
                hs_pert = hs + pert_value * sv.view(1, 1, -1)
                captured.append(hs_pert[0, -1, :].detach().cpu().numpy().copy())
                return (hs_pert,) + output[1:]
            hs_pert = output + pert_value * sv.view(1, 1, -1)
            captured.append(hs_pert[0, -1, :].detach().cpu().numpy().copy())
            return hs_pert
        handle = target_module.register_forward_hook(perturb_hook)
    else:
        handle = target_module.register_forward_hook(capture_hook)

    for sample in samples:
        context = sample.get("context", "")
        question = sample.get("question", "")
        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        captured.clear()
        with torch.no_grad():
            model(**inputs)
        if captured:
            X_list.append(captured[0])
        pos_labels.append(sample.get("position", "unknown"))

    if handle:
        handle.remove()
    return np.array(X_list), pos_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_n", type=int, default=5)
    args = parser.parse_args()

    energy_stds = [0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    direction_alphas = [-5.0, -3.0, -1.5, 0.0, 1.5, 3.0, 5.0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log("=" * 60)
    log("P13: Energy vs Direction Asymmetry (Fast Representational Test)")
    log("=" * 60)
    log(f"  Layer: {args.layer}, N: {args.n}/pos, Device: {device}")

    model, tokenizer = load_model_and_tokenizer()
    model.eval()

    pos_dir = f"data_position_sensitivity/s{args.seed}"

    log("\n[1] Loading test samples...")
    test_all = []
    for pos in ["early", "mid", "late"]:
        samples = load_jsonl(os.path.join(pos_dir, f"test_{pos}_s{args.seed}.jsonl"))[:args.n]
        for s in samples:
            s["position"] = pos
        test_all.extend(samples)
    log(f"  {len(test_all)} test samples ({args.n}/position)")

    log("\n[2] Computing v_abs direction vector...")
    train_early = load_jsonl(os.path.join(pos_dir, f"train_early_s{args.seed}.jsonl"))
    train_late = load_jsonl(os.path.join(pos_dir, f"train_late_s{args.seed}.jsonl"))

    X_early, _ = collect_perturbed_hidden_states(
        model, tokenizer, train_early[:args.train_n], args.layer, device,
        "none", 0.0
    )
    X_late, _ = collect_perturbed_hidden_states(
        model, tokenizer, train_late[:args.train_n], args.layer, device,
        "none", 0.0
    )

    v_abs = X_early.mean(axis=0) - X_late.mean(axis=0)
    v_abs = v_abs / (np.linalg.norm(v_abs) + 1e-8)
    log(f"  v_abs norm: {np.linalg.norm(v_abs):.4f}")
    log(f"  ||h_early - h_late|| mean: {np.linalg.norm(X_early - X_late, axis=1).mean():.2f}")

    all_rows = []

    log("\n[3] Collecting baseline hidden states...")
    baseline_X, _ = collect_perturbed_hidden_states(
        model, tokenizer, test_all, args.layer, device,
        "none", 0.0
    )
    log(f"  baseline ||h|| mean: {np.linalg.norm(baseline_X, axis=1).mean():.2f}")

    log("\n[4] ENERGY perturbation (Gaussian noise)...")
    for std in energy_stds:
        label = f"noise={std:.3f}"
        X, pos_labels = collect_perturbed_hidden_states(
            model, tokenizer, test_all, args.layer, device,
            "noise", std
        )
        shifts = np.linalg.norm(X - baseline_X, axis=1)

        by_pos = {"early": [], "mid": [], "late": []}
        for i, pos in enumerate(pos_labels):
            by_pos[pos].append(shifts[i])

        mean_shift = shifts.mean()
        e_shift = np.mean(by_pos["early"])
        m_shift = np.mean(by_pos["mid"])
        l_shift = np.mean(by_pos["late"])
        max_ratio = max(e_shift, m_shift, l_shift) / (min(e_shift, m_shift, l_shift) + 1e-8)
        early_vs_late_ratio = e_shift / (l_shift + 1e-8)

        log(f"  [{label}] mean_shift={mean_shift:.2f} e={e_shift:.2f} m={m_shift:.2f} "
            f"l={l_shift:.2f} max_ratio={max_ratio:.2f} e/l={early_vs_late_ratio:.2f}")

        all_rows.append({
            "perturbation_type": "energy",
            "parameter": std,
            "mean_shift": mean_shift,
            "early_shift": e_shift,
            "mid_shift": m_shift,
            "late_shift": l_shift,
            "max_ratio": max_ratio,
            "early_late_ratio": early_vs_late_ratio,
        })

    log("\n[5] DIRECTION perturbation (v_abs steering)...")
    for alpha in direction_alphas:
        label = f"alpha={alpha:+.1f}"
        X, pos_labels = collect_perturbed_hidden_states(
            model, tokenizer, test_all, args.layer, device,
            "direction", alpha, steering_vector=v_abs
        )
        shifts = np.linalg.norm(X - baseline_X, axis=1)

        by_pos = {"early": [], "mid": [], "late": []}
        for i, pos in enumerate(pos_labels):
            by_pos[pos].append(shifts[i])

        mean_shift = shifts.mean()
        e_shift = np.mean(by_pos["early"])
        m_shift = np.mean(by_pos["mid"])
        l_shift = np.mean(by_pos["late"])
        max_ratio = max(e_shift, m_shift, l_shift) / (min(e_shift, m_shift, l_shift) + 1e-8)
        early_vs_late_ratio = e_shift / (l_shift + 1e-8)

        log(f"  [{label}] mean_shift={mean_shift:.2f} e={e_shift:.2f} m={m_shift:.2f} "
            f"l={l_shift:.2f} max_ratio={max_ratio:.2f} e/l={early_vs_late_ratio:.2f}")

        all_rows.append({
            "perturbation_type": "direction",
            "parameter": alpha,
            "mean_shift": mean_shift,
            "early_shift": e_shift,
            "mid_shift": m_shift,
            "late_shift": l_shift,
            "max_ratio": max_ratio,
            "early_late_ratio": early_vs_late_ratio,
        })

    log("\n[6] ASYMMETRY ANALYSIS...")

    energy_ratios = [r["max_ratio"] for r in all_rows if r["perturbation_type"] == "energy" and r["parameter"] > 0.01]
    dir_ratios = [r["max_ratio"] for r in all_rows if r["perturbation_type"] == "direction" and r["parameter"] != 0]

    energy_avg_ratio = np.mean(energy_ratios) if energy_ratios else 0
    dir_avg_ratio = np.mean(dir_ratios) if dir_ratios else 0

    log(f"  Energy avg max_ratio: {energy_avg_ratio:.3f} (expect ~1.0 for uniform)")
    log(f"  Direction avg max_ratio: {dir_avg_ratio:.3f} (expect >1.0 for asymmetric)")
    log(f"  Direction/Energy ratio: {dir_avg_ratio/(energy_avg_ratio+1e-8):.2f}x")

    if dir_avg_ratio > energy_avg_ratio * 1.5:
        log(f"\n  VERDICT: Absorption is DIRECTION-DOMINATED at representational level")
        log(f"  (direction asymmetry > 1.5x energy asymmetry)")
    elif energy_avg_ratio > dir_avg_ratio * 1.5:
        log(f"\n  VERDICT: Absorption is ENERGY-DOMINATED at representational level")
        log(f"  (energy asymmetry > 1.5x direction asymmetry)")
    else:
        ratio = dir_avg_ratio / (energy_avg_ratio + 1e-8)
        if ratio > 1.1:
            log(f"\n  VERDICT: Absorption has WEAK directional asymmetry ({ratio:.2f}x)")
        elif ratio < 0.9:
            log(f"\n  VERDICT: Absorption has WEAK energy asymmetry ({1/ratio:.2f}x)")
        else:
            log(f"\n  VERDICT: Energy and Direction have SIMILAR asymmetry (~{ratio:.2f}x)")
            log(f"  Suggests both affect hidden states uniformly at representational level")
            log(f"  Behavioral asymmetry observed in P12 may be downstream of hidden state representation")

    with open(os.path.join(RESULTS_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump({
            "layer": args.layer, "n_per_pos": args.n, "seed": args.seed,
            "energy_stds": energy_stds, "direction_alphas": direction_alphas,
            "energy_avg_max_ratio": energy_avg_ratio,
            "direction_avg_max_ratio": dir_avg_ratio,
            "all_rows": all_rows,
        }, f, indent=2, ensure_ascii=False)

    with open(os.path.join(RESULTS_DIR, "shift_data.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    log(f"\nResults: {RESULTS_DIR}/")
    log("Done.")


if __name__ == "__main__":
    main()