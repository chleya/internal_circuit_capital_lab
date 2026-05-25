# IC-4-P2: Impulse Direction Specificity Audit Report

> **P2 实验**: 把 T3 从"脉冲存在效应"推进到"方向特异性测试"
>
> **日期**: 2026-05-21
> **状态**: 已完成 — 方向特异性未确认

---

## 1. 设计

| 参数 | 值 |
|---|---|
| Model | Qwen2.5-0.5B-Instruct |
| Sweep layers | [8, 12, 16] |
| Sweep steps | [prefill] (唯一一个 — T1/T2显示信号最强) |
| Directions | v_hall, v_random_norm, v_orthogonal, v_syc, v_shuffled |
| Epsilons | [1.0, 3.0] |
| N per combo | 10 |
| Baseline (no impulse) | 20 |
| Total generations | 320 |
| Gen time | 1070s (17.8 min) |

### 三个关键对照

| 对照 | 构造 | 目的 |
|---|---|---|
| **v_random_norm** | 与 v_hall 相同 L2 norm 的随机方向 (seed=777) | 控制扰动能量 |
| **v_orthogonal** | Gram-Schmidt: v_rand - proj(v_rand, v_hall)，|cos| < 1e-5 | 排除方向相关性（即使 cos 接近 0 的随机也可能有偶然相关） |
| **v_shuffled** | 打乱样本标签的 steering vector | 保留同分布但破坏语义 |

---

## 2. 方向诊断

| Direction | Norm | cos(v_hall, dir) | 
|---|---|---|
| v_hall | 1.0000 | 1.000000 |
| v_random_norm | 1.0000 | 0.048399 |
| v_orthogonal | 1.0000 | **-0.000000** ✓ |
| v_syc | 1.0000 | 0.115133 |
| v_shuffled | 1.0000 | 0.438955 |

**正交性确认**: v_orthogonal 与 v_hall 严格正交 (|cos| < 1e-5)。

---

## 3. 基线行为（无 impulse）

| Behavior | Count | Rate |
|---|---|---|
| hallucination | 13/20 | 0.650 |
| correct | 5/20 | 0.250 |
| incorrect_answerable | 1/20 | 0.050 |
| other_unanswerable | 1/20 | 0.050 |

---

## 4. Impulse 行为：方向间无差异

| Direction | n | Hall Rate | ΔH | Ctrl |
|---|---|---|---|---|
| v_hall | 60 | 0.367 | **-0.283** | 0.271 |
| v_random_norm | 60 | 0.367 | **-0.283** | 0.275 |
| v_orthogonal | 60 | 0.367 | **-0.283** | 0.250 |
| v_syc | 60 | 0.383 | -0.267 | 0.254 |
| v_shuffled | 60 | 0.383 | -0.267 | 0.242 |

### v_hall vs v_orthogonal (最严格的对照)

| 指标 | 值 |
|---|---|
| ΔH(v_hall) | -0.2833 |
| ΔH(v_orthogonal) | -0.2833 |
| \|diff\| | **0.0000** |

---

## 5. 结论

### 方向特异性：未确认

**v_hall 与 v_orthogonal（与 v_hall 严格正交的随机方向）产生完全相同的 Hall 率变化（-0.283 vs -0.283）。**

五个方向的 controllability 全部集中在 0.24-0.28 窄区间内。即使 cos(v_hall, v_shuffled)=0.439 的 shuffled 方向也产
生几乎相同的效应（ΔH=-0.267 vs -0.283）。

### 判定

> **当前 impulse 效应来自 early-state 的扰动敏感性（perturbation magnitude），不是方向特异性的因果控制。**
>
> 任何一个 L2 归一化的方向注入 prefilling 都能将 hallucination 率从 0.650 降低到 ~0.37。这不是 v_hall 在起作用——严格正交
> 于 v_hall 的方向产生完全相同的效果。

### 升级状态

| SG | 原表述 | 更新后 |
|---|---|---|
| SG-5 (T3) | "当前 impulse 证据尚不足以证明 steering direction 的因果特异性" | **确认**: P2 实验证明方向特异性在全球层面不存在。 |
| SG-4 (T3) | "大幅 early impulse 可改变 hallucination-related behavior" | 升级为: "大幅 early prefill impulse 可将 Hall 率降低 ~28 个百分点，但效应来自扰动能量本身而非方向" |

---

## 6. Implications

### 对 Feedback Control (P3) 的影响

如果方向特异性不存在，那么 P3 不能做 "direction-specific real-time feedback control"。
但这不意味着 feedback control 完全不可行——可以换一个策略：

> **Perturbation-magnitude control**: 在每个 step 根据 v_hall projection 的绝对值动态调整 impulse epsilon。
> 如果 projection 超过阈值，注入 calibrated perturbation 来抑制 hallucination。

另一种可能：方向特异性可能只有在**局部**才存在（某个特定 (layer, step)）。当前结果
是全球层面上的平均。可以考虑后续做逐 (layer, step) 的分析。

### 对 "early-state perturbation sensitivity" 模型的支撑

这一结果实际上**加强了** Phase 1 的核心发现——prefill 可读且对扰动高度敏感。
但如果所有方向都能达到相同效果，那么：
- hallucination suppression 不需要 "找到正确的因果方向"
- 任何足够大的 prefill perturbation 都能 "重置" 模型的 hallucination 轨迹
- 这更接近 "early-state bistability" 模型而非 "directional steering" 模型

---

## 7. 文件索引

| 文件 | 说明 |
|---|---|
| `src/run_p2_direction_specificity.py` | P2 实验脚本 |
| `results_p2_direction_specificity/p2_direction_specificity_log.txt` | 完整日志 |
| `results_p2_direction_specificity/p2_summary.json` | 结构化摘要 |
| `results_p2_direction_specificity/impulse_results.csv` | 所有 300 组 impulse 数据 |
| `results_p2_direction_specificity/baseline_results.csv` | 基线数据 |

---

*IC-4-P2 Direction Specificity Audit — 2026-05-21*