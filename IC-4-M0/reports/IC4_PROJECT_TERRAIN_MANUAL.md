# IC-4 Project Terrain Manual

This document is for future agents working in `F:\internal_circuit_capital_lab\IC-4-M0`.

It is not a literature review. It is a working map of:

1. what this project is actually trying to prove,
2. what has already been established experimentally,
3. which earlier conclusions were corrected later,
4. how external papers changed the research direction,
5. what is now solid ground versus open terrain.

---

## 1. Project Core Question

IC-4 is not primarily asking whether a model "can answer questions."

It is asking:

> Does the model contain an internal reliability mechanism that is readable, controllable, and selectively callable?

In practical terms:

- can we read from internal state whether a sample is risky,
- can we intervene only when that risk condition is present,
- can we reduce hallucination without damaging answerable performance,
- and can we do this by intervening in the model's internal dynamics rather than by retraining or prompt-only tricks?

This is why the project gradually moved from:

- "find a steering vector"

to:

- "find a condition-aware internal controller"

and then further to:

- "find the correct way to attach that controller to the model's forward dynamics."

---

## 2. Current Proven Mechanism

The current reference mechanism is the `M3-v6` single-pass gate.

Reference files:

- `F:\internal_circuit_capital_lab\IC-4-M0\src\run_m3_v6.py`
- `F:\internal_circuit_capital_lab\IC-4-M0\src\gate_steering_tool.py`
- `F:\internal_circuit_capital_lab\IC-4-M0\results_m3_v6\metrics_raw.csv`
- `F:\internal_circuit_capital_lab\IC-4-M0\reports_m3_v6\IC4_M3_V6_SINGLE_PASS_GATE_REPORT.md`

Mechanism:

- representation: `last_prompt_token`
- probe: logistic regression
- gate: hard threshold
- injection: single-pass forward hook
- generation path: `model.generate()`
- steering layer: 12
- reference alpha: `-1.0`

Reference result:

| mode | H | C | UA |
|---|---:|---:|---:|
| base | 0.8667 | 0.6000 | 0.0000 |
| oracle_gate_a-1.0 | 0.6667 | 0.6000 | 0.0000 |
| single_pass_hard_gate_a-1.0 | 0.6667 | 0.6000 | 0.0000 |
| random_single_pass_hard_gate_a-1.0 | 0.9333 | 0.6000 | 0.0000 |
| shuffled_single_pass_hard_gate_a-1.0 | 0.8000 | 0.6000 | 0.0000 |

Interpretation:

- the steering vector `v` is real and causally useful,
- the prefill state contains a usable gate signal,
- the correct injection path matters,
- a simple probe + hard gate is enough to match oracle in the reference setting.

---

## 3. Current Robustness Boundary

Generalization report:

- `F:\internal_circuit_capital_lab\IC-4-M0\results_m4_generalization\sweep_matrix.csv`
- `F:\internal_circuit_capital_lab\IC-4-M0\reports_m4_generalization\IC4_M4_GENERALIZATION_REPORT.md`

What is currently robust:

- `seed = 0`
- `layer = 12`
- scenarios:
  - `standard`
  - `large`
  - `hard OOD`
- alpha values:
  - `-0.8`
  - `-1.0`
  - `-1.2`

At `alpha = -1.0`, the gate matches oracle in all three evaluated data scenarios.

Important boundary:

> "ROBUST" currently means robust across data size, OOD difficulty, and alpha variation inside the validated reference setting.

It does **not** yet mean:

- cross-seed gate robustness,
- cross-layer gate robustness,
- cross-model robustness,
- cross-behavior robustness.

Those remain open.

---

## 4. What the Project Has Actually Learned

The project has already established several facts.

### 4.1 The steering vector is effective

This was first hinted by open-loop results, then made much clearer by oracle gate controls.

Takeaway:

> The vector is not fake, and not reducible to random or shuffled controls.

### 4.2 The gating signal exists before generation

The gate signal is readable from prefill state. In the successful reference mechanism, `last_prompt_token` is enough.

Takeaway:

> The model internally contains answerability/risk information before the answer is generated.

### 4.3 Correct signal is not enough by itself

Multiple earlier stages showed that:

- a good probe can still fail behaviorally,
- a good vector can still fail behaviorally,
- early intervention can still fail if attached incorrectly.

Takeaway:

> Success depends on intervention mechanics, not only on signal quality.

### 4.4 Correct dynamics attachment is part of the mechanism

The final success came only after:

- single-pass integration,
- hook-based in-flight decision,
- use of `model.generate()` instead of a manual token-by-token loop.

Takeaway:

> The mechanism is not just "probe + vector"; it is "probe + vector + correct forward-path attachment."

---

## 5. Corrected Attribution History

Future agents must preserve the corrected attribution chain.

### 5.1 M3-v5 was important, but not the final attribution

`M3-v5` correctly forced deeper artifact decomposition, but one of its strongest early interpretations was later corrected.

What `M3-v5` helped expose:

- the implementation path contained strong systematic artifacts,
- gate shape alone was not the main story,
- pipeline and generation mechanics had to be isolated.

What `M3-v6` later corrected:

> The `C = 0.733` anomaly was not fundamentally caused by the two-pass prefill idea itself.

The critical diagnostic in `M3-v6` showed:

- the real source of the anomaly was the manual token-by-token generation loop,
- the hook itself was not the culprit,
- `model.generate()` restores the correct baseline and control behavior.

This is an important historical correction. Do not revert to the older simplified story.

### 5.2 Safe wording

Prefer:

> `M3-v5` exposed implementation-path artifact; `M3-v6` resolved the true source and established the correct reference implementation.

Avoid:

> `M3-v5 proved two-pass is the main bottleneck.`

That wording is too coarse after the `M3-v6` diagnostic.

---

## 6. Stage-by-Stage Experimental Map

### M0 / M1 / M2: vector and artifact discovery

These stages established that open-loop steering can reduce hallucination, but:

- often damages answerable performance,
- can be confounded by task proxies or controls,
- must be evaluated with stronger anti-artifact discipline.

They were necessary, but they did not yet solve selective intervention.

### M3-oracle: the key pivot

This was the first hard proof that the vector itself had real value.

Oracle gate showed:

- steer only unanswerable samples,
- do not touch answerable ones,
- and the behavior improves without the earlier damage pattern.

This changed the project from:

- "find a better vector"

to:

- "find a real gate and the right injection path."

### M3-v2: scalar gate insufficiency

Entropy / max-prob / uncertainty-mass style scalar gates moved, but were not enough.

Lesson:

> Token-local scalar risk signals were too weak to close the gap to oracle.

### M3-v3: trajectory-probe partial advance

This stage showed:

- trajectory-style signals are informative,
- but waiting until token 4 is too late for behavior.

Lesson:

> Good readout can fail if the model has already committed to the wrong trajectory.

### M3-v4 / M3-v4b / M3-v5: implementation-path debugging

These stages improved timing and decomposition, but still suffered from wrong mechanics.

They were valuable because they forced the project to stop blaming the wrong objects.

### M3-v6: convergence

This is the first stage that unambiguously closes the loop:

- readable signal,
- selective intervention,
- correct internal attachment,
- oracle-level behavioral result.

### M4 trajectory-state

This stage showed that:

- pooled / windowed representations outperform last-token for state readout tasks,
- the mechanism is better understood as trajectory/state-level rather than purely token-local.

Important nuance:

> Better state readout does not automatically mean "that is the true computation circuit."

It is still a readout result, not a full computation decomposition.

### M4-generalization

This stage established that the reference mechanism is stable across several scenario changes, within the validated setting.

---

## 7. Theoretical Terrain from External Papers

This project did not simply copy one paper. Instead, different papers helped organize different parts of the terrain.

### 7.1 State ontology layer

Relevant influences:

- `ELF: Embedded Language Flows`
- trajectory / continuous-state perspectives

Main lesson:

> Internal state should not automatically be treated as a single token-local point event.

This helped shift the project toward trajectory/state thinking.

### 7.2 State readout layer

Relevant influences:

- `The Truth Lies Somewhere in the Middle`

Main lesson:

> Generated-token windows and pooled states can be stronger semantic readouts than single-token snapshots.

This fed directly into `M4`.

### 7.3 Control architecture layer

Relevant influences:

- fast/slow adaptation framing

Main lesson:

> A stable steering prior can be seen as a slow structure, while the gate/probe/controller acts like a fast adaptor.

In project terms:

- steering vector `v` behaves like slow structure,
- probe/gate behaves like fast conditional control.

### 7.4 Stable dynamics layer

Relevant influences:

- attractor models
- edge-of-stability / attractor-set viewpoints

Main lesson:

> Reliability may come from being brought into a stable region, not simply from changing one local coordinate.

This helped frame the gate as more than an on/off switch.

More specifically, attractor-style models suggest a useful interpretation for the successful IC-4 mechanism:

> The working gate + steering mechanism can be viewed as a minimal external attractor controller.

In this interpretation:

- the model produces an initial internal state,
- the probe reads whether that state is entering a risky regime,
- the steering intervention selectively redirects that state toward a more reliable basin,
- and correct behavior depends on whether that redirection is attached to the true forward dynamics.

This attractor framing is helpful because it explains why earlier partial solutions failed:

- a useful vector alone was not enough,
- a useful readout alone was not enough,
- and even correct timing was not enough if the intervention path was wrong.

It also suggests a longer-term possibility:

> Today's gate + steering may function as an external reliability scaffold, while future training-time mechanisms might internalize that refinement and move the model's initial state closer to the reliable equilibrium by default.

### 7.5 State-flow visualization / geometric intuition layer

Relevant influence:

- `Diffusion-Explorer`

Main lesson:

> We benefit from thinking of intervention not only as "adding a vector" but as changing the path that state follows through a learned flow.

This repository is not a direct mechanism-analysis tool for this project. It is an educational and visualization-oriented system for diffusion / flow / rectified-flow geometry. Its value here is conceptual:

- it reinforces a trajectory/flow view of internal state,
- it makes "straightening" or "redirecting" a path more intuitive,
- it supports the idea that some interventions should be understood as reshaping state evolution rather than merely changing a local coordinate.

Applied to IC-4:

> The successful gate + steering mechanism can be described not only as a conditional residual modification, but as a selective reshaping of internal state flow.

This influence belongs more to explanation and intuition-building than to direct experimental procedure.

### 7.6 Propagation mechanism layer

Relevant influence:

- `Spontaneous symmetry breaking and Goldstone modes for deep information propagation`

Main lesson:

> We should ask not only what signal is read and what intervention is applied, but also why that intervention propagates stably through depth.

This is especially relevant after `M3-v6`, because success depends on allowing the intervention to ride the correct forward dynamics.

This layer suggests a future question:

> Is the successful steering direction coupling into a particularly propagation-friendly mode?

This is not yet proven, but it is a useful theoretical lens.

### 7.7 Representation vs computation layer

Relevant influence:

- `Arithmetic in the Wild`

Main lesson:

> A structured representation does not automatically imply that the model computes using that same structure.

Applied here:

> `M4` readouts are useful and real, but they should not be overinterpreted as the complete causal circuit.

### 7.8 Causal organization layer

Relevant influence:

- `The Causally Emergent Alignment Hypothesis`

Main lesson:

> Better behavior may reflect better-organized internal causal structure, not only higher output scores.

This does not yet directly change the current implementation, but it motivates later theory-facing analysis.

---

## 8. What Not to Forget

Future agents should preserve these negative lessons.

### 8.1 Do not go back to manual token-by-token generation as baseline-equivalent

Manual loops in this project were shown to produce materially different behavior from `model.generate()`.

If used again, they must be treated as a different experimental regime, not as a harmless reimplementation.

### 8.2 Do not overclaim robustness

The current mechanism is strong, but the robust claim is scoped.

### 8.3 Do not collapse readout success into mechanism success

Good probe accuracy is not sufficient. The project only became convincing once the intervention was causally effective.

### 8.4 Do not flatten the mechanism to just "the vector"

The functioning mechanism includes:

- the vector,
- the probe,
- the gate policy,
- the hook site,
- the forward-path implementation,
- the generation method.

---

## 9. Current Best Summary for External Handoff

If another agent needs the shortest accurate summary, use this:

> IC-4 has established a reference internal reliability mechanism for one validated setting: a logistic gate read from prefill state can selectively activate a steering direction during a single forward pass, matching oracle anti-hallucination performance while preserving answerable performance. The main scientific lesson is that success depends not just on finding a useful steering direction or a readable probe signal, but on attaching the intervention correctly to the model's forward dynamics.

If slightly more detail is needed:

> In the validated reference setting (`seed=0`, `layer=12`, Qwen2.5-0.5B-Instruct), the project has shown that the model contains a readable prefill gate signal and a useful steering direction. When these are combined through a hard-threshold gate inside a single-pass forward hook and executed with `model.generate()`, the result matches oracle gating exactly across standard, larger, and harder OOD data scenarios.

---

## 10. Current Open Terrain

The strongest near-term directions now are:

### 10.1 Toolization

Turn the `M3-v6` mechanism into a clean reusable augmentation pipeline.

### 10.2 Confirmatory validation

Still missing:

- cross-seed gate validation,
- cross-layer gate validation.

### 10.3 Imperfect-probe regime

This branch has now produced a meaningful partial answer.

Branch A2 result:

> The mechanism remains strong when the probe is no longer perfect, but hard gating is the current best default policy.

What is now supported by data:

- hard gate remains oracle-level or near-oracle through the high-accuracy regime,
- degradation begins to appear around `probe_acc ~ 0.90`,
- `soft_T0.1` is **not** a stable improvement over hard gate,
- `soft_T0.3` is consistently too soft and should not be treated as a mainline candidate.

Representative aggregate pattern from `results_branch_a2/aggregate_stats.csv`:

- `n=5`, `probe_acc_mean ~ 0.997`:
  - hard oracle gap `~ 0.000`
  - soft_T0.1 oracle gap `~ +0.013`
- `n=3`, `probe_acc_mean ~ 0.903`:
  - hard oracle gap `~ +0.033`
  - soft_T0.1 oracle gap `~ +0.046`
- `n=2`, `probe_acc_mean ~ 0.880`:
  - hard oracle gap `~ +0.026`
  - soft_T0.1 oracle gap `~ +0.080`

Operational takeaway:

> The mechanism is not dependent on a perfectly accurate probe, but current evidence favors hard gating over soft gating in the imperfect-probe regime.

What remains open:

- a fuller degradation curve with more repeats and/or additional regimes,
- confirmation on larger and harder evaluation settings,
- whether any alternative soft/confidence-aware policy can beat hard gate under more realistic noisy-probe conditions.

Early branch results suggest yes, and suggest that sharp soft gating may outperform hard gating in some imperfect-probe regimes.

### 10.4 Multi-behavior expansion

The broader framework is not limited to unanswerable hallucination. In principle it can be extended to:

- factuality hallucination,
- sycophancy,
- refusal / harmful compliance,
- tool-use caution,
- other condition × behavior intervention pairs.

### 10.5 Propagation analysis

Not urgent, but theoretically important:

- how the intervention propagates across layers,
- how real and control vectors differ in propagation,
- whether successful intervention corresponds to a more stable internal propagation regime.

---

## 11. One-Sentence Project Identity

> IC-4 is a project about finding and validating an internal reliability mechanism that is readable from model state, selectively controllable, and only effective when attached correctly to the model's own forward dynamics.
