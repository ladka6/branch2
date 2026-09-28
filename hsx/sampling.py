"""Autoregressive window sampling from an early-exit depth."""

import torch


@torch.inference_mode()
def sample_window(model, contexts: torch.Tensor, n_tokens: int, temperature: float,
                  generator: torch.Generator) -> torch.Tensor:
    """Sample n_tokens from `model` for a batch of equal-length contexts.

    Under any lossless stage-1 verifier, the tentative window handed to L_f is
    distributed exactly as this autoregressive sample from L_i. That is why the
    final-stage statistics can be measured without simulating stage 1.
    """
    out = model(input_ids=contexts, use_cache=True, return_dict=True)
    past, logits = out.past_key_values, out.logits[:, -1, :]
    toks = []
    for step in range(n_tokens):
        p = torch.softmax(logits.float() / temperature, dim=-1)
        nxt = torch.multinomial(p, 1, generator=generator)  # [B,1]
        toks.append(nxt)
        if step < n_tokens - 1:
            out = model(input_ids=nxt, past_key_values=past, use_cache=True, return_dict=True)
            past, logits = out.past_key_values, out.logits[:, -1, :]
    return torch.cat(toks, dim=1)
