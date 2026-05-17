# Branch B: Cross-Behavior Extension Framework

> Minimal scaffold for extending the IC-4 intervention pattern from
> anti-hallucination to other behavior × condition pairs.
> Do NOT run large experiments from this scaffold in the current round.

## 1. Framework Architecture

```
BehaviorInterventionPipeline
├── BehaviorConfig         (metadata: behavior_id, labels, metrics)
├── ConditionDetector      (trainable probe / rule for condition signal)
├── SteeringVectorLoader   (loads or computes steering vector)
├── GatePolicy             (threshold / soft / confidence-aware)
└── BehaviorEvaluator      (behavior-specific evaluation rules)
```

The single-pass hook-based gate mechanism from M3-v6 is reused unchanged
as the core intervention engine. Only the detector, vector, and evaluator
vary by behavior.

## 2. Reference Example: Anti-Hallucination

The current anti-hallucination pipeline (M3-v6) is a fully concrete instance:

| Component | Implementation |
|---|---|
| `BehaviorConfig` | `behavior_id="anti_hallucination"`, condition=answerability |
| `ConditionDetector` | Logistic probe on `last_prompt_token` at layer 12 |
| `SteeringVectorLoader` | `results_m3/activations_s{seed}_l{layer}.npz` |
| `GatePolicy` | Hard threshold at 0.5 |
| `BehaviorEvaluator` | `hallucination_rate`, `correct_answer_rate`, etc. |

All concrete classes are provided in `branch_b_behavior_framework.py`
as reference implementations.

## 3. Future Behavior Candidates

### 3.1 Factuality Hallucination (Recommended Priority)

**Definition:** The model makes factually false claims when it is uncertain
about objective facts (not just "information not in context").

**Condition signal:** Factual confidence — can the model's internal state
indicate whether it is generating a confident factual claim vs guessing?

**Key difference from current:** The current mechanism deals with
"answerable vs unanswerable given context." Factuality hallucination
deals with "the model asserts false facts even when the context is adequate."

**Why this is the best first extension:**
- Closest to the current anti-hallucination mechanism
- Same steering direction may partially transfer
- Probe training data can be constructed with factually-correct vs
  factually-false answer pairs on known facts
- The M4 trajectory/state readout insights are directly applicable
- Builds on the project's strongest existing vocabulary

### 3.2 Sycophancy

**Definition:** The model agrees with a user's false premise rather than
correcting it.

**Condition signal:** User premise validity — does the model's internal
state detect that the user's claim is inconsistent with the model's
knowledge?

**Key challenge:** The condition signal is more complex than answerability
because it requires detecting conflict between the user's premise and the
model's internal world knowledge.

**Why this is worth doing but harder:**
- Sycophancy is a well-documented alignment failure mode
- Requires constructing paired data (user makes false claim → model should
  disagree vs user makes true claim → model should agree)
- The steering direction may be different from anti-hallucination
- More complex evaluation: need to measure both sycophantic agreement
  (false) and correct agreement (true)

### 3.3 Excessive Refusal

**Definition:** The model refuses benign requests that superficially
resemble harmful prompts.

**Condition signal:** Request risk assessment — is the request actually
harmful or just flagged by superficial pattern matching?

**Why this is lower priority:**
- Requires a safety harmfulness taxonomy
- Evaluation is inherently more subjective
- The current anti-hallucination mechanism already provides a calibrated
  abstention baseline — excessive refusal is a variation on that theme

## 4. Implementation Roadmap (Not for This Round)

### Phase B1: Factuality Hallucination

1. Create factuality QA dataset (correct vs false answer pairs)
2. Train factuality probe analogous to answerability probe
3. Test whether the SAME steering vector (anti-hallucination) works
4. If not, compute a new steering vector from factuality activation pairs
5. Evaluate using the BehaviorInterventionPipeline

### Phase B2: Sycophancy

1. Create sycophancy dataset (user false premise → correct/disagree pairs)
2. Train premise-validity probe
3. Compute sycophancy-specific steering vector
4. Gate policy likely needs adjustment (soft gate may be more appropriate
   for nuanced premise evaluation)

### Phase B3: Multi-Behavior Composition

Once multiple behaviors are individually validated:
- Investigate whether a single probe can detect multiple condition types
- Test orthogonal steering directions for different behaviors
- Explore whether multi-behavior intervention causes interference

## 5. Current Status

| Item | Status |
|---|---|
| Framework scaffold | Created |
| Anti-hallucination reference | Fully validated (M3-v6 + M4) |
| Factuality extension | NOT started |
| Sycophancy extension | NOT started |
| Excessive refusal extension | NOT started |
| Multi-behavior composition | NOT started |

## 6. Design Notes

### Why reuse M3-v6 single-pass hook?

The single-pass hook mechanism is proven correct for attaching a probe
+ gate to the model's forward dynamics. New behaviors should not need
a new intervention mechanism — they need new detectors, vectors, and
evaluation rules on top of the same proven infrastructure.

### Why not use manual token-by-token loop?

Already disproven by M3-v6 diagnostic. Manual loops produce C=0.733
artifact. All extensions must use `model.generate()` with forward hooks.

### Probe representation choice

`last_prompt_token` is the current default and works well. However,
M4 trajectory analysis showed that pooled/windowed representations
can be stronger for state readout. Future behaviors may benefit from
testing alternative representations (mean_pooled, question_span_pooled)
especially if the condition signal is more distributed than answerability.

### Gate policy choice

Hard gate is the current default. Branch A2 showed that soft_T0.1 is
not stably better. For new behaviors, start with hard gate and only
explore soft variants if probe accuracy is demonstrably below ~0.90.

## 7. Recommendation

**Start with factuality hallucination.** It is the most natural extension,
builds directly on existing infrastructure, and the anti-hallucination
steering vector may partially transfer, reducing the activation collection
burden.

---
*Branch B: Cross-Behavior Extension Framework — Minimal Scaffold*