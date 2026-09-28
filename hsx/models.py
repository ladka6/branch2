"""Model loading and early-exit helpers for LayerSkip-style checkpoints."""

from copy import deepcopy
from typing import Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM


def pick_device_dtype() -> Tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.device("cuda"), dtype
    return torch.device("cpu"), torch.float32


def load_model(name: str, device: torch.device, dtype: torch.dtype):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, low_cpu_mem_usage=True)
    except TypeError:  # older transformers
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.to(device).eval()
    return tok, model


def tiny_random_model(device: torch.device, n_layers: int = 8, vocab: int = 512, seed: int = 0):
    """Small random Llama used only for offline smoke tests."""
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=vocab,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        tie_word_embeddings=True,
        initializer_range=0.2,  # large enough that depths disagree (TV ~0.3-0.45)
    )
    return LlamaForCausalLM(cfg).to(device).eval()


def make_early_exit_model(base, n_layers: int):
    """Truncate to the first n_layers, sharing parameters with base.

    Keeps the final norm and LM head, which is how LayerSkip checkpoints exit early.
    """
    memo = {id(p): p for p in base.parameters()}
    model = deepcopy(base, memo=memo)
    total = len(model.model.layers)
    if not 0 < n_layers <= total:
        raise ValueError(f"n_layers={n_layers} outside 1..{total}")
    model.model.layers = torch.nn.ModuleList(list(model.model.layers[:n_layers]))
    if hasattr(model.config, "num_hidden_layers"):
        model.config = deepcopy(model.config)
        model.config.num_hidden_layers = n_layers
    return model.eval()


def cost_units(model) -> Dict[str, float]:
    """Parameter counts used by the memory-bound cost model.

    In decoding, a forward pass is roughly bound by weight reads, so the cost of
    running k layers plus the head is proportional to k*layer_params + head_params.
    We express the head in layer-equivalents (H). For Llama-3.2-1B H is about 4,
    so an L4 early exit costs ~8 layer-units, not 4.
    """
    layer_params = sum(p.numel() for p in model.model.layers[0].parameters())
    head_params = model.lm_head.weight.numel()
    return {
        "layer_params": float(layer_params),
        "head_params": float(head_params),
        "head_in_layers": float(head_params) / float(layer_params),
        "n_layers": float(len(model.model.layers)),
    }
