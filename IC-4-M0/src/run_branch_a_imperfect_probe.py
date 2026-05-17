"""
Branch A: Imperfect Probe / Soft Gate — Robustness Under Probe Degradation.

Question: If the probe is no longer 100% accurate, does the mechanism still hold?
Compares hard gate, soft gate (two temperatures), and confidence-aware gate.

Design:
  - Gradient probe degradation: 15 → 10 → 7 → 5 → 3 train samples per class
  - Four gate types: hard, soft (T=0.1, T=0.3), confidence_aware (zone=0.35-0.65)
  - Metrics: H, C, UA, oracle_gap (vs oracle_H=0.667), gate activation stats
"""
import os, sys, time, json, random, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.gate_steering_tool import GateSteeringTool
from src.model_loader import load_model_and_tokenizer
from src.data_builder import load_jsonl

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results_branch_a")
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_SIZES = [15, 10, 7, 5, 3]
ALPHA = -1.0
LAYER = 12
SEED = 0
SUBSAMPLE_SEED = 123
ORACLE_H = 0.667

GATE_CONFIGS = [
    {"name": "hard",               "gate_type": "hard"},
    {"name": "soft_T0.1",          "gate_type": "soft",  "soft_temperature": 0.1},
    {"name": "soft_T0.3",          "gate_type": "soft",  "soft_temperature": 0.3},
    {"name": "confidence_aware",   "gate_type": "confidence_aware", "confidence_zone": (0.35, 0.65)},
]


def log(msg):
    print(msg, flush=True)


def subsample_by_class(data, n_per_class, rng):
    pos = [s for s in data if s.get("answerability") == "answerable"]
    neg = [s for s in data if s.get("answerability") == "unanswerable"]
    n_pos = min(n_per_class, len(pos))
    n_neg = min(n_per_class, len(neg))
    return rng.sample(pos, n_pos) + rng.sample(neg, n_neg)


def main():
    log("=" * 70)
    log("Branch A: Imperfect Probe / Soft Gate Experiment")
    log("=" * 70)
    log(f"Train sizes per class: {TRAIN_SIZES}")
    log(f"Gate configs: {[g['name'] for g in GATE_CONFIGS]}")
    log(f"Alpha={ALPHA}, Layer={LAYER}, Seed={SEED}, Oracle_H={ORACLE_H}")

    log("\n[1/4] Loading model...")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-0.5B-Instruct", "cpu", "float32")
    log(f"  Loaded in {time.time()-t0:.0f}s")

    log("\n[2/4] Loading data + steering vectors...")
    full_train = load_jsonl("data_m3/train_s0.jsonl")
    test = load_jsonl("data_m3/test_s0.jsonl")
    vectors = GateSteeringTool.load_activation(SEED, LAYER)
    sv = vectors["steering"]
    na_full = sum(1 for s in full_train if s.get("answerability") == "answerable")
    nu_full = len(full_train) - na_full
    na_test = sum(1 for s in test if s.get("answerability") == "answerable")
    nu_test = len(test) - na_test
    log(f"  Full train: {len(full_train)} ({na_full}A+{nu_full}U)")
    log(f"  Test: {len(test)} ({na_test}A+{nu_test}U)")
    log(f"  Steering vector dim: {sv.shape[0]}")

    log("\n[3/4] Running experiment sweep...")
    all_rows = []
    rng = random.Random(SUBSAMPLE_SEED)
    total_start = time.time()

    for n_per_class in TRAIN_SIZES:
        log(f"\n  --- Train size = {n_per_class} per class ---")

        sub_train = subsample_by_class(full_train, n_per_class, rng)
        na = sum(1 for s in sub_train if s.get("answerability") == "answerable")
        nu = len(sub_train) - na

        tool = GateSteeringTool(model, tokenizer, config={
            "threshold": 0.5, "cv_folds": min(3, n_per_class), "max_new_tokens": 48,
            "temperature": 0.0, "do_sample": False,
        })
        t_probe = time.time()
        probe_info = tool.train_probe(sub_train, LAYER, "last_prompt_token")
        train_time = time.time() - t_probe

        probe_eval = tool.evaluate_probe(test, probe_info)
        probe_test_acc = probe_eval["accuracy"]
        log(f"  Probe: train_acc={probe_info['train_acc']:.3f}, test_acc={probe_test_acc:.3f}, "
            f"train_time={train_time:.1f}s, n={na}A+{nu}U")

        for gcfg in GATE_CONFIGS:
            name = gcfg["name"]
            kwargs = {k: v for k, v in gcfg.items() if k not in ("name",)}
            t_gen = time.time()
            results = tool.generate_batch(test, sv, LAYER, ALPHA,
                                          probe_info=probe_info, control_type="steering",
                                          **kwargs)
            gen_time = time.time() - t_gen
            metrics = tool.evaluate(results)

            gate_vals = [r["gate"] for r in results]
            gate_on = sum(1 for g in gate_vals if g > 0.01)
            gate_mean = float(np.mean(gate_vals)) if gate_vals else 0.0
            gate_std = float(np.std(gate_vals)) if gate_vals else 0.0

            row = {
                "train_size_per_class": n_per_class,
                "train_n_pos": na,
                "train_n_neg": nu,
                "probe_train_acc": round(probe_info["train_acc"], 4),
                "probe_test_acc": round(probe_test_acc, 4),
                "gate_type": name,
                "gate_kwargs": str(kwargs),
                "H": round(metrics["hallucination_rate"], 4),
                "C": round(metrics["correct_answer_rate"], 4),
                "UA": round(metrics["unnecessary_abstention_rate"], 4),
                "oracle_gap": round(metrics["hallucination_rate"] - ORACLE_H, 4),
                "gate_on_count": gate_on,
                "gate_mean": round(gate_mean, 4),
                "gate_std": round(gate_std, 4),
                "gen_time_s": round(gen_time, 1),
                "train_time_s": round(train_time, 1),
            }
            all_rows.append(row)
            log(f"    {name:<20} H={row['H']:.3f} C={row['C']:.3f} UA={row['UA']:.3f} "
                f"gap={row['oracle_gap']:+.3f} gate_on={gate_on}/{len(results)} "
                f"gate_mu={gate_mean:.2f} gen={gen_time:.0f}s")

    total_elapsed = time.time() - total_start
    log(f"\n  Total sweep time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

    log("\n[4/4] Saving results...")
    csv_path = os.path.join(OUT_DIR, "sweep_results.csv")
    fieldnames = list(all_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    log(f"  Saved {len(all_rows)} rows to {csv_path}")

    json_path = os.path.join(OUT_DIR, "sweep_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False)
    log(f"  Saved to {json_path}")

    log("\n" + "=" * 70)
    log("SUMMARY TABLE")
    log("=" * 70)
    hdr = f"{'n/cls':<6} {'p_acc':<8} {'gate_type':<20} {'H':<8} {'C':<8} {'UA':<8} {'gap':<8} {'on':<6}"
    log(hdr)
    log("-" * len(hdr))
    for n in TRAIN_SIZES:
        rows_n = [r for r in all_rows if r["train_size_per_class"] == n]
        for r in rows_n:
            log(f"{r['train_size_per_class']:<6} {r['probe_test_acc']:<8.3f} {r['gate_type']:<20} "
                f"{r['H']:<8.3f} {r['C']:<8.3f} {r['UA']:<8.3f} {r['oracle_gap']:<+8.3f} "
                f"{r['gate_on_count']:<6}")

    log("\nDone.")


if __name__ == "__main__":
    main()