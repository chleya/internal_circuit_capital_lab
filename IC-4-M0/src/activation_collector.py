"""
IC-4-M0: Activation collector.
Collects hidden states for positive and negative response pairs.
"""

import torch
import numpy as np
from typing import List, Dict, Optional
from tqdm import tqdm


def _get_layer_index(model, layer_spec: str) -> int:
    """Resolve layer specification to an integer index."""
    from .model_loader import get_middle_layer_index, get_model_layer_count

    if layer_spec == "middle":
        return get_middle_layer_index(model)
    elif layer_spec == "last":
        return get_model_layer_count(model) - 1
    else:
        idx = int(layer_spec)
        total = get_model_layer_count(model)
        if idx < 0 or idx >= total:
            raise ValueError(f"Layer index {idx} out of range [0, {total - 1}]")
        return idx


def _extract_hidden_state(
    outputs, layer_idx: int, token_position: str
) -> np.ndarray:
    """
    Extract hidden state from model outputs at specified layer and token position.

    Args:
        outputs: Model output tuple with hidden_states.
        layer_idx: Index of the layer to extract from (0-based, hidden_states includes embedding layer at index 0).
        token_position: "last" or "mean".

    Returns:
        numpy array of shape (hidden_dim,)
    """
    hidden_states = outputs.hidden_states

    if isinstance(hidden_states, tuple):
        hs = hidden_states[layer_idx + 1]
    else:
        hs = hidden_states[layer_idx + 1]

    hs = hs[0]

    if token_position == "last":
        vec = hs[-1, :].detach().float().cpu().numpy()
    elif token_position == "mean":
        vec = hs.mean(dim=0).detach().float().cpu().numpy()
    else:
        pos = int(token_position)
        vec = hs[pos, :].detach().float().cpu().numpy()

    return vec


def collect_pair_activations(
    model,
    tokenizer,
    dataset: List[Dict],
    layer: str = "middle",
    token_position: str = "last",
    max_length: int = 256,
) -> Dict[str, np.ndarray]:
    """
    Collect positive and negative activations for each sample in the dataset.

    For each sample, we construct:
      - prompt = context + " " + question
      - pos_input = prompt + " " + positive_response
      - neg_input = prompt + " " + negative_response

    We run each through the model and extract the hidden state at the specified layer
    at the specified token position.

    Returns:
        dict with keys:
            "positive": np.ndarray of shape (N, hidden_dim)
            "negative": np.ndarray of shape (N, hidden_dim)
    """

    layer_idx = _get_layer_index(model, layer)

    pos_acts = []
    neg_acts = []

    for sample in tqdm(dataset, desc="Collecting activations"):
        context = sample["context"]
        question = sample["question"]
        pos_resp = sample["positive_response"]
        neg_resp = sample["negative_response"]

        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"

        pos_text = f"{prompt} {pos_resp}"
        neg_text = f"{prompt} {neg_resp}"

        pos_inputs = tokenizer(
            pos_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        neg_inputs = tokenizer(
            neg_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )

        device = next(model.parameters()).device
        pos_inputs = {k: v.to(device) for k, v in pos_inputs.items()}
        neg_inputs = {k: v.to(device) for k, v in neg_inputs.items()}

        with torch.no_grad():
            pos_outputs = model(**pos_inputs)
        pos_act = _extract_hidden_state(pos_outputs, layer_idx, token_position)
        pos_acts.append(pos_act)

        with torch.no_grad():
            neg_outputs = model(**neg_inputs)
        neg_act = _extract_hidden_state(neg_outputs, layer_idx, token_position)
        neg_acts.append(neg_act)

    return {
        "positive": np.stack(pos_acts, axis=0),
        "negative": np.stack(neg_acts, axis=0),
    }


def save_activations(activations: Dict[str, np.ndarray], path: str):
    """Save activations dictionary to .npz file."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **activations)


def load_activations(path: str) -> Dict[str, np.ndarray]:
    """Load activations from .npz file."""
    data = np.load(path)
    return {key: data[key] for key in data.files}