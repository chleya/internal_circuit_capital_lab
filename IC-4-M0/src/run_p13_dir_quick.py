"""P13 Direction-only quick test."""
import os, sys, json, numpy as np, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_loader import load_model_and_tokenizer
from src.data_builder import load_jsonl
from src.run_p13_energy_vs_direction import collect_perturbed_hidden_states

device = "cpu"
model, tokenizer = load_model_and_tokenizer()
model.eval()

pos_dir = "data_position_sensitivity/s0"
test_all = []
for pos in ["early", "mid", "late"]:
    samples = load_jsonl(os.path.join(pos_dir, f"test_{pos}_s0.jsonl"))[:10]
    for s in samples:
        s["position"] = pos
    test_all.extend(samples)

# v_abs
train_early = load_jsonl(os.path.join(pos_dir, "train_early_s0.jsonl"))
train_late = load_jsonl(os.path.join(pos_dir, "train_late_s0.jsonl"))
X_early, _ = collect_perturbed_hidden_states(model, tokenizer, train_early[:5], 10, device, "none", 0.0)
X_late, _ = collect_perturbed_hidden_states(model, tokenizer, train_late[:5], 10, device, "none", 0.0)
v_abs = X_early.mean(axis=0) - X_late.mean(axis=0)
v_abs = v_abs / (np.linalg.norm(v_abs) + 1e-8)

# Baseline
baseline_X, pos_labels = collect_perturbed_hidden_states(
    model, tokenizer, test_all, 10, device, "none", 0.0
)

print("DIRECTION perturbation:")
for alpha in [-3.0, -1.5, 0.0, 1.5, 3.0]:
    t0 = time.time()
    X, pl = collect_perturbed_hidden_states(
        model, tokenizer, test_all, 10, device, "direction", alpha, steering_vector=v_abs
    )
    shifts = np.linalg.norm(X - baseline_X, axis=1)
    by_pos = {"early": [], "mid": [], "late": []}
    for i, pos in enumerate(pl):
        by_pos[pos].append(shifts[i])
    e_shift = np.mean(by_pos["early"])
    m_shift = np.mean(by_pos["mid"])
    l_shift = np.mean(by_pos["late"])
    max_ratio = max(e_shift, m_shift, l_shift) / (min(e_shift, m_shift, l_shift) + 1e-8)
    print(f"  alpha={alpha:+.1f}: mean={shifts.mean():.2f} e={e_shift:.2f} m={m_shift:.2f} "
          f"l={l_shift:.2f} max_ratio={max_ratio:.3f} ({time.time()-t0:.0f}s)")

# Save
result = {
    "energy": [{"std": s, "max_ratio": r} for s, r in [
        (0.01, 1.02), (0.03, 1.02), (0.05, 1.01), (0.10, 1.01)
    ]],
    "direction_results": [
        {"alpha": a, "mean_shift": float(np.mean(np.linalg.norm(
            collect_perturbed_hidden_states(model, tokenizer, test_all, 10, device, "direction", a, steering_vector=v_abs)[0] - baseline_X, axis=1
        )))}
        for a in [-3.0, -1.5, 0.0, 1.5, 3.0]
    ] if False else []
}
with open("results_p13_energy_vs_direction/dir_results.json", "w") as f:
    json.dump(result, f, indent=2)
print("Done.")