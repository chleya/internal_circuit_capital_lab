"""P9: Cross-Bottleneck Structural Integrity Check.

Tests whether v_syc steering degrades the structural clustering of
syc vs non-syc hidden states. This bridges the Stabilization bottleneck
(structural organization) and the Organization bottleneck (steering).

Experiment:
  1. Collect L10 last_prompt_token hs under baseline (no hook) and
     steered (v_syc a=-3.0 hook) forward passes.
  2. Cluster each condition with KMeans (k=2).
  3. Compare structural metrics: ARI, purity, centroid distances.

Hypothesis: If steering degrades structural separation, Per-Action
KMeans (stabilization) could be applied post-steering to recover it.
If structure is preserved, bottlenecks are independent.

Usage:
  python -m src.run_p9_cross_bottleneck

Outputs:
  results_p9_cross_bottleneck/
    p9_results.npz
  reports/IC4_P9_CROSS_BOTTLENECK_REPORT.md
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

from .model_loader import load_model_and_tokenizer
from .steering import _make_steering_hook, _find_transformer_layer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_log = logging.getLogger("p9_cross_bottleneck")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results_p9_cross_bottleneck"
REPORTS_DIR = PROJECT_ROOT / "reports"
CONTRAST_DATA_PATH = PROJECT_ROOT / "results_p0_sycophancy" / "sycophancy_contrast_data.json"
STEERING_VECTORS_PATH = PROJECT_ROOT / "results_t3_impulse_p4" / "steering_vectors.npz"

STEERING_LAYER = 10
OPTIMAL_ALPHA = -3.0
RANDOM_SEED = 42
N_TEST = 24
TEST_START_IDX = 18


def _prompt_from_sample(sample: dict) -> str:
    prompt = sample.get("prompt", "")
    if not prompt:
        context = sample.get("context", "")
        question = sample.get("question", "")
        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
    return prompt


def _collect_hidden_states(
    model, tokenizer, samples: List[dict], steering_vector=None, alpha=0.0
):
    device = next(model.parameters()).device
    hidden_states = []
    labels = []

    target_module = _find_transformer_layer(model, STEERING_LAYER)

    for i, sample in enumerate(tqdm(samples, desc="collecting hs")):
        prompt = _prompt_from_sample(sample)
        is_syc = sample.get("group", "") == "sycophantic"
        labels.append(is_syc)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        captured = []

        def _capture_hook(module, inputs_tup, output):
            if isinstance(output, tuple):
                hs = output[0]
            else:
                hs = output
            captured.append(hs[0, -1, :].detach().cpu().numpy().copy())

        with torch.no_grad():
            if steering_vector is not None and alpha != 0.0:
                steer_handle = target_module.register_forward_hook(
                    _make_steering_hook(steering_vector, alpha, device)
                )
                capture_handle = target_module.register_forward_hook(_capture_hook)
                try:
                    _ = model(**inputs)
                finally:
                    steer_handle.remove()
                    capture_handle.remove()
            else:
                handle = target_module.register_forward_hook(_capture_hook)
                try:
                    _ = model(**inputs)
                finally:
                    handle.remove()

        hidden_states.append(captured[0])

    return np.stack(hidden_states), np.array(labels)


def _cluster_metrics(X: np.ndarray, y_true: np.ndarray, n_clusters=2):
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init="auto")
    y_pred = km.fit_predict(X)
    ari = adjusted_rand_score(y_true, y_pred)
    centroids = km.cluster_centers_

    labels_unique = np.unique(y_pred)
    purities = []
    for label in labels_unique:
        mask = y_pred == label
        true_labels_in_cluster = y_true[mask]
        if len(true_labels_in_cluster) > 0:
            majority_count = max(
                np.sum(true_labels_in_cluster == 0),
                np.sum(true_labels_in_cluster == 1),
            )
            purities.append(majority_count / len(true_labels_in_cluster))
    mean_purity = np.mean(purities) if purities else 0.0

    centroids_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    centroid_cos = np.dot(centroids_norm[0], centroids_norm[1])

    intra_dists = []
    for label in labels_unique:
        mask = y_pred == label
        cluster_points = X[mask]
        if len(cluster_points) > 1:
            centroid = centroids[label]
            dists = np.linalg.norm(cluster_points - centroid, axis=1)
            intra_dists.append(np.mean(dists))
    mean_intra = np.mean(intra_dists) if intra_dists else 0.0

    inter_dist = np.linalg.norm(centroids[0] - centroids[1]) if n_clusters >= 2 else 0.0

    return {
        "ari": float(ari),
        "purity": float(mean_purity),
        "centroid_cos": float(centroid_cos),
        "mean_intra_dist": float(mean_intra),
        "inter_dist": float(inter_dist),
        "cluster_sizes": [int(np.sum(y_pred == l)) for l in labels_unique],
        "y_pred": y_pred.tolist(),
    }


def _build_report(baseline_metrics, steered_metrics, shift_stats, output_path: Path):
    lines = []
    lines.append("# IC-4 P9: Cross-Bottleneck Structural Integrity Check")
    lines.append("")
    lines.append("> **Date**: 2026-05-23 | **Status**: Completed")
    lines.append("> **Predecessor**: P8 Large-Scale Replication, C11 Cross-Bottleneck Analogue")
    lines.append("> **Layer**: 10 | **Alpha**: -3.0 | **N**: 24")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Motivation")
    lines.append("")
    lines.append(
        "P8 confirmed steering direction is correct but the two-stage selective advantage "
        "is spurious at n=24. P9 asks a deeper structural question: does steering degrade "
        "the hidden state clustering that separates syc from non-syc samples?"
    )
    lines.append("")
    lines.append(
        "If steering disrupts structural organization, the Stabilization bottleneck "
        "(Per-Action KMeans) could compensate by post-steering reclustering. If structure "
        "is preserved, the two bottlenecks operate independently and cross-bottleneck "
        "synergy is not supported."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Experiment Design")
    lines.append("")
    lines.append("| Phase | Description |")
    lines.append("|---|---|")
    lines.append(
        "| 1 | Collect L10 last_prompt_token hs from baseline (no hook) forward pass |"
    )
    lines.append(
        "| 2 | Collect L10 last_prompt_token hs from steered (v_syc a=-3.0 hook) forward pass |"
    )
    lines.append("| 3 | KMeans (k=2) clustering on each condition |")
    lines.append(
        "| 4 | Compare ARI, purity, centroid separation, per-sample shift |"
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Results")
    lines.append("")
    lines.append(
        "| Metric | Baseline | Steered | Delta | Interpretation |"
    )
    lines.append("|---|---|---|---|---|")
    for key, label in [
        ("ari", "ARI (vs ground truth)"),
        ("purity", "Cluster Purity"),
        ("centroid_cos", "Centroid Cosine Sim"),
        ("mean_intra_dist", "Mean Intra-Cluster Dist"),
        ("inter_dist", "Inter-Cluster Distance"),
    ]:
        bv = baseline_metrics[key]
        sv = steered_metrics[key]
        delta = sv - bv
        if abs(delta) < 0.001:
            interp = "unchanged"
        elif key in ("centroid_cos",):
            interp = "closer" if delta > 0 else "further apart"
        elif key in ("mean_intra_dist",):
            interp = "tighter" if delta < 0 else "looser"
        elif key == "inter_dist":
            interp = "more separated" if delta > 0 else "less separated"
        else:
            interp = "better" if delta > 0 else "worse"
        lines.append(f"| {label} | {bv:.4f} | {sv:.4f} | {delta:+.4f} | {interp} |")

    lines.append("")
    lines.append(f"- **Baseline cluster sizes**: {baseline_metrics['cluster_sizes']}")
    lines.append(f"- **Steered cluster sizes**: {steered_metrics['cluster_sizes']}")
    lines.append("")
    lines.append("### 3.1 Per-Sample Hidden State Shift")
    lines.append("")
    lines.append(
        f"- Mean ||hs_steered - hs_baseline||: {shift_stats['mean_shift']:.4f}"
    )
    lines.append(
        f"- Std ||hs_steered - hs_baseline||: {shift_stats['std_shift']:.4f}"
    )
    lines.append(
        f"- Max ||hs_steered - hs_baseline||: {shift_stats['max_shift']:.4f}"
    )
    lines.append(
        f"- Cosine sim (hs_steered, hs_baseline) mean: {shift_stats['mean_cos']:.4f}"
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Interpretation")
    lines.append("")

    ari_delta = steered_metrics["ari"] - baseline_metrics["ari"]
    purity_delta = steered_metrics["purity"] - baseline_metrics["purity"]
    cos_delta = steered_metrics["centroid_cos"] - baseline_metrics["centroid_cos"]

    if abs(ari_delta) < 0.05 and abs(purity_delta) < 0.05:
        lines.append(
            "**Verdict: Structure preserved.** Steering does NOT degrade the "
            "syc/non-syc clustering structure. ARI and purity are nearly unchanged "
            "between baseline and steered conditions."
        )
        lines.append("")
        lines.append(
            "**Implication for cross-bottleneck**: Stabilization (Per-Action KMeans) "
            "and Organization (steering) operate on independent dimensions. "
            "There is no evidence that steering-induced structural degradation needs "
            "stabilization compensation. Cross-bottleneck synergy (1+1>2) is "
            "NOT supported at the representational level."
        )
        lines.append("")
        lines.append(
            "**What this means**: The two bottlenecks are additive, not synergistic. "
            "Per-Action KMeans and steering can be applied independently without "
            "interfering with each other, but combining them will not produce "
            "effects beyond their individual contributions."
        )
    else:
        lines.append(
            "**Verdict: Structure degraded.** Steering significantly changes the "
            "syc/non-syc clustering structure."
        )
        lines.append("")
        if ari_delta < -0.05:
            lines.append(
                "ARI decreased — steering makes ground-truth separation harder."
            )
        if purity_delta < -0.05:
            lines.append(
                "Purity decreased — steering mixes syc/non-syc representations."
            )
        lines.append("")
        lines.append(
            "**Implication**: Per-Action KMeans post-steering reclustering "
            "could recover structural organization. Cross-bottleneck synergy "
            "IS supported — proceed to joint testing."
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. Next Steps")
    lines.append("")
    lines.append("| Priority | Action |")
    lines.append("|---|---|")
    lines.append("| 1 | P10: Hallucination — formally abandon single-direction impulse |")
    lines.append("| 2 | Per-Action KMeans Scaling on gridworld |")
    lines.append("| 3 | Absorption Remedy (position-aware routing) |")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    _log.info(f"Report: {output_path}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    _log.info("P9: Cross-Bottleneck Structural Integrity Check")

    _log.info("Loading contrast data...")
    with open(CONTRAST_DATA_PATH, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    test_samples = all_data[TEST_START_IDX : TEST_START_IDX + N_TEST]
    _log.info(f"Test samples: {len(test_samples)}")

    _log.info("Loading model and steering vectors...")
    model, tokenizer = load_model_and_tokenizer()
    steering_data = np.load(STEERING_VECTORS_PATH, allow_pickle=True)
    v_syc = steering_data["v_syc"]

    _log.info("Phase 1: Collect baseline hidden states")
    X_base, y_true = _collect_hidden_states(model, tokenizer, test_samples)

    _log.info("Phase 2: Collect steered hidden states")
    X_steer, _ = _collect_hidden_states(
        model, tokenizer, test_samples, steering_vector=v_syc, alpha=OPTIMAL_ALPHA
    )

    _log.info("Phase 3: Clustering analysis")
    baseline_metrics = _cluster_metrics(X_base, y_true)
    steered_metrics = _cluster_metrics(X_steer, y_true)

    _log.info(f"Baseline ARI={baseline_metrics['ari']:.4f}, purity={baseline_metrics['purity']:.4f}")
    _log.info(f"Steered  ARI={steered_metrics['ari']:.4f}, purity={steered_metrics['purity']:.4f}")

    shifts = np.linalg.norm(X_steer - X_base, axis=1)
    cos_sims = np.sum(X_steer * X_base, axis=1) / (
        np.linalg.norm(X_steer, axis=1) * np.linalg.norm(X_base, axis=1) + 1e-8
    )
    shift_stats = {
        "mean_shift": float(np.mean(shifts)),
        "std_shift": float(np.std(shifts)),
        "max_shift": float(np.max(shifts)),
        "mean_cos": float(np.mean(cos_sims)),
    }

    _log.info(f"Mean hs shift: {shift_stats['mean_shift']:.4f}")
    _log.info(f"Mean hs cos sim: {shift_stats['mean_cos']:.4f}")

    np.savez(
        RESULTS_DIR / "p9_results.npz",
        X_base=X_base,
        X_steer=X_steer,
        y_true=y_true,
        baseline_metrics=baseline_metrics,
        steered_metrics=steered_metrics,
        shift_stats=shift_stats,
    )

    report_path = REPORTS_DIR / "IC4_P9_CROSS_BOTTLENECK_REPORT.md"
    _build_report(baseline_metrics, steered_metrics, shift_stats, report_path)

    _log.info("P9 complete.")


if __name__ == "__main__":
    main()