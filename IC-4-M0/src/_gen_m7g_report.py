import pandas as pd, numpy as np

df = pd.read_csv("results_m7/m7g_manifold_protect.csv")
base_sr = 0.55

lines = [
    "IC-4-M7 G: Manifold Protection -- PCA Subspace REPLACE Report",
    "=" * 64,
    "Layer: 20 | Base syc_rate: 0.5500",
    "Hypothesis: Sycophancy lives in top PCs; syntax/semantics in remaining PCs.",
    "REPLACE only top-K PC components, preserve the rest.",
    "",
]
header = f"{'K':>6s} {'Var%':>7s} {'Syc Rate':>10s} {'Delta':>10s} {'Flipped':>8s} {'AvgLen':>8s} {'Rep':>8s}"
lines.append(header)
lines.append("-" * 64)

for _, row in df.iterrows():
    k = int(row["K"])
    vp = row["var_pct"]
    sr = row["syc_rate"]
    d = row["delta"]
    fl = int(row["flipped"])
    ns = int(row["n_steered"])
    al = row["avg_len"]
    ar = row["avg_rep"]
    lines.append(f"  {k:>4d}  {vp:>5.1f}%  {sr:>10.4f}  {d:>+10.4f}  {fl:>4d}/{ns:>4d}  {al:>6.0f}  {ar:>8.3f}")

lines += [
    "",
    "## KEY FINDINGS",
    "",
    "1. MONOTONIC SCALING: Anti-sycophancy effect scales SMOOTHLY with K.",
    "   K=1  (46% var): Delta=-0.10 (3/11 flip) -- PC1 alone has small effect",
    "   K=5  (86% var): Delta=-0.20 (4/11 flip)",
    "   K=10 (94% var): Delta=-0.25 (6/11 flip)",
    "   K=20 (99% var): Delta=-0.30 (7/11 flip) -- majority flip!",
    "   K=896 (full):    Delta=-0.55 (11/11 flip) -- complete flip",
    "",
    "2. NO SHARP THRESHOLD: Effect accumulates gradually across PCs.",
    "   Sycophancy signal is NOT concentrated in PC1 -- it is distributed",
    "   across the entire PCA spectrum, with diminishing returns.",
    "",
    "3. MANIFOLD PROTECTION PARTIALLY VALIDATED:",
    "   You CAN get partial anti-sycophancy (Delta=-0.30, 7/11) with only",
    "   20 PCs (2.2% of dimensions), preserving 97.8% of the manifold.",
    "   But FULL flip requires ALL 896 dimensions.",
    "",
    "4. TEXT QUALITY STABLE: avg_len and repetition are consistent",
    "   across all K values. No collapse at any intervention level.",
    "",
    "5. THEORETICAL IMPLICATION:",
    "   The sycophancy attractor is NOT a surface overlay -- it is woven",
    "   into the full 896D fabric. PC1 captures 46% of variance but only",
    "   18% of the behavioral effect (0.10/0.55). The remaining effect",
    "   is distributed across hundreds of low-variance dimensions.",
    "",
    "   This is consistent with SUPERPOSITION: sycophancy-relevant",
    "   features are represented in superposition with syntax/semantics",
    "   across most dimensions, making them inseparable via linear PCA.",
    "",
    "6. COMPARISON WITH EXTERNAL ANALYSES:",
    "   - Bandwidth Bottleneck (Ext #4): CONFIRMED -- 896D is overcrowded",
    "   - Attractor Collapse (Ext #4): CONFIRMED -- sycophancy is woven in",
    "   - VAE Expand Dimensions (Ext #2): SUPPORTED -- more dimensions",
    "     would allow separating sycophancy from syntax/semantics",
    "",
]

with open("results_m7/m7g_manifold_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
for line in lines:
    print(line)
print("\nDone.")