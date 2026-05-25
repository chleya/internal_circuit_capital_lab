"""
IC-4-M0: Steering vector computation and application.
Computes steering vectors from positive/negative activation pairs and applies
forward hooks during generation.
"""

import numpy as np
import torch
from typing import Optional, Callable


def compute_steering_vector(pos_acts: np.ndarray, neg_acts: np.ndarray) -> np.ndarray:
    """
    Compute steering vector = mean(pos_acts) - mean(neg_acts).

    Args:
        pos_acts: (N, D) positive activations.
        neg_acts: (N, D) negative activations.

    Returns:
        steering vector of shape (D,)
    """
    pos_mean = pos_acts.mean(axis=0)
    neg_mean = neg_acts.mean(axis=0)
    vector = pos_mean - neg_mean
    vector = vector / (np.linalg.norm(vector) + 1e-8)
    return vector


def compute_random_vector(dim: int, seed: int = 42) -> np.ndarray:
    """Generate a random normalized vector as a control."""
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def compute_shuffled_vector(pos_acts: np.ndarray, neg_acts: np.ndarray, seed: int = 123) -> np.ndarray:
    """
    Randomly reassign sample labels (answerable/unanswerable) while
    preserving group sizes. If the steering direction is meaningful,
    this null-hypothesis vector should be nearly orthogonal to it.
    """
    rng = np.random.RandomState(seed)
    all_acts = np.concatenate([pos_acts, neg_acts], axis=0)
    n_pos, n_neg = pos_acts.shape[0], neg_acts.shape[0]
    all_labels = np.array([1] * n_pos + [0] * n_neg, dtype=bool)
    rng.shuffle(all_labels)
    shuffled_pos = all_acts[all_labels]
    shuffled_neg = all_acts[~all_labels]
    return compute_steering_vector(shuffled_pos, shuffled_neg)


def compute_norm_matched_orthogonal(target_vector: np.ndarray, seed: int = 456) -> np.ndarray:
    """
    Generate a vector with the same norm as target_vector but strictly
    orthogonal direction. Used to decompose directional vs energetic
    contributions in impulse experiments.

    Method:
    1. Generate a random vector
    2. Gram-Schmidt: project out the target direction
    3. Renormalize to match target_vector's norm

    Args:
        target_vector: (D,) reference direction (e.g., v_syc).
        seed: random seed.

    Returns:
        orthogonal vector of shape (D,) with same L2 norm as target_vector
        and dot product ~0 with target_vector.
    """
    rng = np.random.RandomState(seed)
    dim = target_vector.shape[0]
    v_rand = rng.randn(dim).astype(np.float32)
    target_norm = np.linalg.norm(target_vector)
    target_unit = target_vector / (target_norm + 1e-8)
    proj = np.dot(v_rand, target_unit) * target_unit
    v_ortho = v_rand - proj
    v_ortho = v_ortho / (np.linalg.norm(v_ortho) + 1e-8)
    v_ortho = v_ortho * target_norm
    return v_ortho


def _make_steering_hook(vector: np.ndarray, alpha: float, target_device: torch.device):
    """Create a forward hook that adds alpha * vector to hidden states."""

    vec_tensor = torch.from_numpy(vector).to(target_device).float()

    def hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            h = outputs[0]
        else:
            h = outputs
        v = vec_tensor.to(dtype=h.dtype)
        h = h + alpha * v
        if isinstance(outputs, tuple):
            return (h,) + outputs[1:]
        else:
            return h

    return hook


def _find_transformer_layer(model, layer_idx: int):
    """
    Locate the module corresponding to a transformer layer across different model architectures.

    Returns the module that can be used with register_forward_hook.
    """

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers[layer_idx]

    raise ValueError("Cannot locate transformer layers in this model architecture.")


def apply_steering_hook(
    model,
    layer: int,
    vector: np.ndarray,
    alpha: float,
):
    """
    Register a forward hook on the specified layer that adds alpha * vector to hidden states.

    Args:
        model: The HuggingFace model.
        layer: 0-based layer index.
        vector: Steering vector (D,).
        alpha: Scaling factor.

    Returns:
        handle: Hook handle that can be used with handle.remove().
    """

    device = next(model.parameters()).device
    layer_module = _find_transformer_layer(model, layer)
    hook_fn = _make_steering_hook(vector, alpha, device)
    handle = layer_module.register_forward_hook(hook_fn)
    return handle


def get_all_vectors(
    pos_acts: np.ndarray,
    neg_acts: np.ndarray,
    dim: int,
) -> dict:
    """
    Compute all steering and control vectors.

    Returns:
        dict with keys: "steering", "random", "shuffled"
    """
    return {
        "steering": compute_steering_vector(pos_acts, neg_acts),
        "random": compute_random_vector(dim),
        "shuffled": compute_shuffled_vector(pos_acts, neg_acts),
    }


class AdaptiveAlpha:
    """
    Mutable alpha container for per-token feedback-controlled steering.

    alpha(t) = base_alpha * (1 + k * (entropy(t) / baseline_entropy - 1))
    Clamped to [min_scale * base_alpha, max_scale * base_alpha].
    """

    def __init__(self, base_alpha: float, k: float = 1.0,
                 min_scale: float = 0.2, max_scale: float = 3.0):
        self.base_alpha = base_alpha
        self.k = k
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.value = base_alpha
        self.history: list = []

    def update(self, entropy: float, baseline_entropy: float):
        if baseline_entropy < 1e-8:
            self.value = self.base_alpha
            return
        ratio = entropy / baseline_entropy
        scaled = self.base_alpha * (1.0 + self.k * (ratio - 1.0))
        min_val = self.min_scale * self.base_alpha
        max_val = self.max_scale * self.base_alpha
        if self.base_alpha < 0:
            self.value = max(min_val, min(max_val, scaled))
        else:
            self.value = min(max_val, max(min_val, scaled))
        self.history.append(self.value)


def _make_adaptive_steering_hook(vector: np.ndarray, alpha_container: AdaptiveAlpha,
                                  target_device: torch.device):
    """Create a forward hook that reads alpha dynamically from a container."""

    vec_tensor = torch.from_numpy(vector).to(target_device).float()

    def hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            h = outputs[0]
        else:
            h = outputs
        v = vec_tensor.to(dtype=h.dtype)
        alpha = alpha_container.value
        h = h + alpha * v
        if isinstance(outputs, tuple):
            return (h,) + outputs[1:]
        else:
            return h

    return hook


def apply_adaptive_steering_hook(model, layer: int, vector: np.ndarray,
                                  alpha_container: AdaptiveAlpha):
    """
    Register a forward hook with dynamically updatable alpha.

    Returns:
        handle: Hook handle for later removal.
    """
    device = next(model.parameters()).device
    layer_module = _find_transformer_layer(model, layer)
    hook_fn = _make_adaptive_steering_hook(vector, alpha_container, device)
    handle = layer_module.register_forward_hook(hook_fn)
    return handle