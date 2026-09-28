"""All-depth distributions from one forward pass, and per-token statistics.

Core idea: run the full model once on context+window, capture every decoder
layer's output with hooks, and apply the shared final norm + LM head. For a
LayerSkip checkpoint this equals the early-exit model at that depth (checked in
tests/smoke_test.py), so one pass gives p_2, p_4, ..., p_L at every window
position.
"""

from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import torch

TRIPLE_FIELDS = [
    "null_up",           # sum_v p_s(v) [p_s>=p_a and p_b>=p_s]
    "null_down",         # sum_v p_s(v) [p_s<=p_a and p_b<=p_s]
    "null_overshoot",    # sum_v p_s(v) [p_s>p_a and p_b<p_s]
    "null_undershoot",   # sum_v p_s(v) [p_s<p_a and p_b>p_s]
    "acc_mass",          # P(stage-1 draft accepted) = sum_v min(p_a, p_s)
    "res_mass",          # sum_v (p_s-p_a)_+ = 1 - acc_mass, stored directly for precision
    "acc_alpha_sum",     # sum_v min(p_a,p_s)(v) * alpha_sb(v)
    "res_alpha_sum",     # sum_v (p_s-p_a)_+(v) * alpha_sb(v)
]


@torch.inference_mode()
def depth_logits(model, input_ids: torch.Tensor, depths: Sequence[int], start: int) -> Dict[int, torch.Tensor]:
    """Logits at each requested depth for positions start..end-1 (fp32).

    Position t predicts token t+1. To score a window occupying positions
    C..C+N-1 of input_ids, pass start=C-1 and input_ids[:, :C+N-1].
    """
    layers = model.model.layers
    L = len(layers)
    want = {d for d in depths if d < L}
    captured: Dict[int, torch.Tensor] = {}
    hooks = []
    for idx, layer in enumerate(layers):
        d = idx + 1
        if d in want:
            def hook(_m, _i, out, d=d):
                h = out[0] if isinstance(out, tuple) else out
                captured[d] = h[:, start:, :]
            hooks.append(layer.register_forward_hook(hook))
    try:
        out = model(input_ids=input_ids, use_cache=False, return_dict=True)
    finally:
        for h in hooks:
            h.remove()

    res = {}
    for d in depths:
        if d == L:
            res[d] = out.logits[:, start:, :].float()
        else:
            res[d] = model.lm_head(model.model.norm(captured[d])).float()
    return res


def pair_list(depths: Sequence[int]) -> List[Tuple[int, int]]:
    return list(combinations(sorted(depths), 2))


def triple_list(depths: Sequence[int], source: int) -> List[Tuple[int, int, int]]:
    lo = [d for d in depths if d < source]
    hi = [d for d in depths if d > source]
    return [(a, source, b) for a in lo for b in hi]


@torch.inference_mode()
def window_stats(
    logits: Dict[int, torch.Tensor],
    tokens: torch.Tensor,
    temperature: float,
    source: int,
) -> Dict[str, torch.Tensor]:
    """Per-token statistics for a window sampled from depth `source`.

    logits[d]: [B, N, V] fp32, row t predicts tokens[:, t].
    tokens:    [B, N]
    """
    depths = sorted(logits)
    probs, tok_logp, ent, top1, margin, amax = {}, [], [], [], [], []
    for d in depths:
        lp = torch.log_softmax(logits[d] / temperature, dim=-1)
        p = lp.exp()
        probs[d] = p
        tok_logp.append(lp.gather(-1, tokens.unsqueeze(-1)).squeeze(-1))
        ent.append(-(p * lp).sum(-1))
        t2 = p.topk(2, dim=-1)
        top1.append(t2.values[..., 0])
        margin.append(t2.values[..., 0] - t2.values[..., 1])
        amax.append(t2.indices[..., 0])
        del lp

    tv = [0.5 * (probs[a] - probs[b]).abs().sum(-1) for a, b in pair_list(depths)]

    trip = []
    for a, s, b in triple_list(depths, source):
        pa, ps, pb = probs[a], probs[s], probs[b]
        alpha = torch.clamp(pb / ps.clamp_min(1e-30), max=1.0)
        acc = torch.minimum(pa, ps)
        res = torch.clamp(ps - pa, min=0.0)
        trip.append(torch.stack([
            (ps * ((ps >= pa) & (pb >= ps))).sum(-1),
            (ps * ((ps <= pa) & (pb <= ps))).sum(-1),
            (ps * ((ps > pa) & (pb < ps))).sum(-1),
            (ps * ((ps < pa) & (pb > ps))).sum(-1),
            acc.sum(-1),
            res.sum(-1),
            (acc * alpha).sum(-1),
            (res * alpha).sum(-1),
        ], dim=-1))
        del alpha, acc, res

    B, N = tokens.shape
    out = {
        "tokens": tokens.int(),
        "tok_logp": torch.stack(tok_logp, -1),
        "entropy": torch.stack(ent, -1),
        "top1": torch.stack(top1, -1),
        "margin": torch.stack(margin, -1),
        "argmax": torch.stack(amax, -1).int(),
        "tv": torch.stack(tv, -1) if tv else torch.zeros(B, N, 0),
        "triple": torch.stack(trip, -2) if trip else torch.zeros(B, N, 0, len(TRIPLE_FIELDS)),
    }
    return {k: v.cpu() for k, v in out.items()}
