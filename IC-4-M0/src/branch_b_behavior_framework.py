"""
Branch B: Cross-Behavior Extension Framework (Minimal Scaffold).

Abstracts the IC-4 intervention pattern from anti-hallucination to a general
behavior × condition interface. Not a full implementation — a scaffold that
defines the API surface for future behavior extensions.

Design principles:
  - Each behavior = (condition_detector, steering_vector, gate_policy, evaluator)
  - Anti-hallucination (current) is one concrete instantiation
  - Factuality hallucination and sycophancy are the most natural next targets
  - Do NOT duplicate the M3-v6 single-pass hook — reuse it as a strategy

Usage (NOT to be run this round):
    from src.branch_b_behavior_framework import (
        BehaviorConfig, ConditionDetector, SteeringVectorLoader,
        GatePolicy, BehaviorEvaluator, BehaviorInterventionPipeline,
    )
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np


class ConditionDetector(ABC):
    """
    Reads model state and returns a real-valued signal indicating
    whether the current context warrants intervention for this behavior.
    """

    @abstractmethod
    def train(self, model, tokenizer, train_data: List[Dict]) -> Dict:
        """
        Train the detector on labeled data.

        Returns detector metadata (train_acc, cv_acc, auc, etc.)
        """

    @abstractmethod
    def predict(self, model, tokenizer, sample: Dict) -> float:
        """
        Return a score in [0, 1] where higher values mean
        "more likely safe / less likely to need intervention".
        """

    @abstractmethod
    def get_probe_info(self) -> Dict:
        """Return trained probe information for gate use."""


class SteeringVectorLoader(ABC):
    """
    Loads or computes a steering vector for a given behavior and layer.
    """

    @abstractmethod
    def load(self, seed: int, layer: int) -> np.ndarray:
        """
        Return the steering vector (D,) for this (behavior, seed, layer).
        Must also provide random / shuffled control vectors.
        """

    @abstractmethod
    def get_control_vectors(self, dim: int) -> Dict[str, np.ndarray]:
        """Return {"random": ..., "shuffled": ...} for the given dim."""


class GatePolicy(ABC):
    """
    Decides whether and how strongly to steer, given a condition score.
    """

    @abstractmethod
    def decide(self, condition_score: float) -> Tuple[bool, float]:
        """
        Returns (should_steer, effective_alpha).

        should_steer: True = apply steering
        effective_alpha: actual steering strength (0.0 = no effect)
        """


class BehaviorEvaluator(ABC):
    """
    Evaluates model outputs for a specific behavior dimension.
    Different behaviors have different success/failure criteria.
    """

    @abstractmethod
    def evaluate(self, outputs: List[Dict]) -> Dict[str, float]:
        """
        Return behavior-specific metrics dict.
        Must include at minimum:
          - The primary failure rate for this behavior
          - Correctness on safe samples
          - Abstention / unnecessary-intervention rates
        """


@dataclass
class BehaviorConfig:
    """
    Configuration for a behavior intervention.

    Example (anti-hallucination):
        BehaviorConfig(
            behavior_id="anti_hallucination",
            description="Suppress hallucination on unanswerable questions",
            condition_label="answerability",
            condition_labels=["answerable", "unanswerable"],
            steering_direction="unanswerable → answerable (anti-hallucination)",
            evaluator_primary_metric="hallucination_rate",
            evaluator_safety_metric="correct_answer_rate",
        )
    """
    behavior_id: str
    description: str
    condition_label: str
    condition_labels: List[str]
    steering_direction: str
    evaluator_primary_metric: str
    evaluator_safety_metric: str
    alpha: float = -1.0
    layer: int = 12


class BehaviorInterventionPipeline:
    """
    Composes detector + vector + gate + evaluator into a single pipeline.

    This mirrors the structure of run_m3_v6.py but abstracts the behavior
    dimension so new behaviors can be plugged in without rewriting the
    generation/injection logic.

    The single-pass hook mechanism from M3-v6 is reused unchanged.
    Only the detector, steering vector, and evaluator vary by behavior.
    """

    def __init__(
        self,
        config: BehaviorConfig,
        detector: ConditionDetector,
        vector_loader: SteeringVectorLoader,
        gate_policy: GatePolicy,
        evaluator: BehaviorEvaluator,
    ):
        self.config = config
        self.detector = detector
        self.vector_loader = vector_loader
        self.gate_policy = gate_policy
        self.evaluator = evaluator

    def run(
        self,
        model,
        tokenizer,
        train_data: List[Dict],
        test_data: List[Dict],
        seed: int,
        layer: int,
    ) -> Dict:
        """
        Complete intervention pipeline:
          1. Train condition detector
          2. Load steering vector
          3. Run single-pass hook-based gate (reuses M3-v6)
          4. Evaluate behavior-specific metrics
        """
        detector_meta = self.detector.train(model, tokenizer, train_data)
        steering_v = self.vector_loader.load(seed, layer)
        control_vectors = self.vector_loader.get_control_vectors(steering_v.shape[0])

        results = {
            "config": self.config,
            "detector_meta": detector_meta,
            "steering_vector": steering_v,
            "control_vectors": control_vectors,
            "metrics": {},
        }
        return results


# =========================================================================
# Reference Instantiation: Anti-Hallucination (Current M3-v6)
# =========================================================================

ANTI_HALLUCINATION_CONFIG = BehaviorConfig(
    behavior_id="anti_hallucination",
    description="Suppress hallucination on unanswerable questions by steering toward calibrated uncertainty",
    condition_label="answerability",
    condition_labels=["answerable", "unanswerable"],
    steering_direction="unanswerable → answerable / calibrated abstention",
    evaluator_primary_metric="hallucination_rate",
    evaluator_safety_metric="correct_answer_rate",
    alpha=-1.0,
    layer=12,
)


class AntiHallucinationDetector(ConditionDetector):
    """
    Logistic probe on last_prompt_token representation.
    Reference: M3-v6 probe training pipeline.
    """

    def __init__(self, layer_idx: int = 12, representation: str = "last_prompt_token"):
        self.layer_idx = layer_idx
        self.representation = representation
        self._probe_info: Dict = {}

    def train(self, model, tokenizer, train_data: List[Dict]) -> Dict:
        from src.run_m3_v6 import _collect_prefill_features, _train_probe
        X, y = _collect_prefill_features(
            model, tokenizer, train_data, self.layer_idx, self.representation)
        self._probe_info = _train_probe(X, y, cv_folds=3)
        return self._probe_info

    def predict(self, model, tokenizer, sample: Dict) -> float:
        proba = self._probe_info.get("classifier")  # reuse trained probe
        return float(proba) if proba is not None else 0.5

    def get_probe_info(self) -> Dict:
        return self._probe_info


class AntiHallucinationVectorLoader(SteeringVectorLoader):
    """
    Loads steering vector from activation files.
    Reference: results_m3/activations_s{seed}_l{layer}.npz
    """

    def __init__(self, activations_dir: str = "results_m3"):
        self.activations_dir = activations_dir

    def load(self, seed: int, layer: int) -> np.ndarray:
        import os
        from src.activation_collector import load_activations
        from src.steering import compute_steering_vector
        path = os.path.join(self.activations_dir, f"activations_s{seed}_l{layer}.npz")
        acts = load_activations(path)
        return compute_steering_vector(acts["positive"], acts["negative"])

    def get_control_vectors(self, dim: int) -> Dict[str, np.ndarray]:
        from src.steering import compute_random_vector, compute_shuffled_vector
        import os
        from src.activation_collector import load_activations
        path = os.path.join(self.activations_dir, "activations_s0_l12.npz")
        acts = load_activations(path)
        return {
            "random": compute_random_vector(dim),
            "shuffled": compute_shuffled_vector(acts["positive"], acts["negative"]),
        }


class HardGatePolicy(GatePolicy):
    """
    Hard threshold gate: steer if condition_score < threshold.
    Reference: M3-v6 default gate policy.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def decide(self, condition_score: float) -> Tuple[bool, float]:
        if condition_score >= self.threshold:
            return False, 0.0
        return True, -1.0


class AntiHallucinationEvaluator(BehaviorEvaluator):
    """
    Hallucination-specific evaluation.
    Reference: src/evaluate.py evaluate_outputs.
    """

    def evaluate(self, outputs: List[Dict]) -> Dict[str, float]:
        from src.evaluate import evaluate_outputs
        return evaluate_outputs(outputs)


# =========================================================================
# Future Extension Blueprints (NOT implemented — design sketches only)
# =========================================================================

FUTURE_BEHAVIOR_FACTUALITY = BehaviorConfig(
    behavior_id="anti_factuality_hallucination",
    description="Suppress factually false claims when the model is uncertain about factual accuracy",
    condition_label="factual_confidence",
    condition_labels=["confident", "uncertain"],
    steering_direction="uncertain → calibrated refusal / evidence-bound answer",
    evaluator_primary_metric="factual_error_rate",
    evaluator_safety_metric="correct_factual_answer_rate",
)


FUTURE_BEHAVIOR_SYCOPHANCY = BehaviorConfig(
    behavior_id="anti_sycophancy",
    description="Reduce sycophantic agreement when the user proposes a false premise",
    condition_label="user_premise_validity",
    condition_labels=["valid", "false"],
    steering_direction="false premise → independent correct response",
    evaluator_primary_metric="sycophantic_agreement_rate",
    evaluator_safety_metric="correct_rejection_rate",
)


FUTURE_BEHAVIOR_REFUSAL = BehaviorConfig(
    behavior_id="anti_excessive_refusal",
    description="Reduce excessive refusal on benign requests that superficially resemble harmful ones",
    condition_label="request_risk",
    condition_labels=["benign", "harmful"],
    steering_direction="benign-but-flagged → appropriate compliance",
    evaluator_primary_metric="false_refusal_rate",
    evaluator_safety_metric="harmful_compliance_rate",
)


FUTURE_BEHAVIORS = {
    "factuality": FUTURE_BEHAVIOR_FACTUALITY,
    "sycophancy": FUTURE_BEHAVIOR_SYCOPHANCY,
    "excessive_refusal": FUTURE_BEHAVIOR_REFUSAL,
}