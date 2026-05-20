"""
IC-4-M0: Model loader.
Loads a HuggingFace causal LM with output_hidden_states=True.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional


def load_model_and_tokenizer(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    device: str = "auto",
    torch_dtype: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a causal LM and its tokenizer.

    Args:
        model_name: HuggingFace model ID or local path.
        device: "auto", "cuda", or "cpu".
        torch_dtype: "float16", "float32", "bfloat16", or None (auto).

    Returns:
        (model, tokenizer)
    """

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }

    dtype = dtype_map.get(torch_dtype, torch.float16) if torch_dtype else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "dtype": dtype,
        "trust_remote_code": True,
        "output_hidden_states": True,
    }

    model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True, **model_kwargs)

    if device == "auto":
        target = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda":
        target = "cuda"
    else:
        target = "cpu"

    model = model.to(target)

    model.eval()
    return model, tokenizer


def get_model_layer_count(model: AutoModelForCausalLM) -> int:
    """Return the number of transformer layers in the model."""

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return len(model.model.decoder.layers)

    raise ValueError("Cannot determine layer count for this model architecture.")


def get_middle_layer_index(model: AutoModelForCausalLM) -> int:
    """Return the index of the middle layer."""
    return get_model_layer_count(model) // 2