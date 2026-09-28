#!/usr/bin/env python3
"""HSD + HiSpec research harness.

Two modes are available:

1. Pair screening (L_d -> L_i)
   Compares expected tokenwise and HSD block efficiency on identical drafts.
   It supports gamma/temperature grids and several corpus presets.

2. Triple pipeline (L_d -> L_i -> L_f)
   Runs a sampling-based HiSpec composition. HSD is used only at L_d -> L_i;
   L_i -> L_f remains standard tokenwise speculative verification. L_f is
   invoked periodically after ``--ni`` tentative tokens, matching the role of
   the tentative-acceptance window in HiSpec Algorithm 2.

Examples:
  python hsd_hispec_experiments.py --pairs 4:8 \
      --gammas 2 4 6 8 12 --temperatures 0.6 0.8 1.0 --contexts 256

  python hsd_hispec_experiments.py --triples 3:6:16 4:8:16 6:10:16 \
      --contexts 128 --gamma 6 --temperature 1.0 --ni 16 \
      --pipeline-new-tokens 64

Important timing note: the triple pipeline measures this reference Python/HF
implementation. The early-exit models share weights, but this file does not yet
reuse hidden states or KV entries *across* L_d, L_i, and L_f. Consequently its
tokens/s is a correctness/diagnostic measurement, not a publication-quality
HiSpec throughput number. The program prints that limitation next to timings.
"""

import argparse
import csv
import gc
import math
import random
import time
from copy import deepcopy
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------
# Utilities
# -----------------------------


def parse_pairs(values: List[str]) -> List[Tuple[int, int]]:
    out = []
    for value in values:
        try:
            ld, li = value.split(":")
            ld, li = int(ld), int(li)
        except Exception as exc:
            raise ValueError(f"Bad pair '{value}'. Use e.g. 2:4 3:6 4:8") from exc
        if not (0 < ld < li):
            raise ValueError(f"Need 0 < Ld < Li, got {ld}:{li}")
        out.append((ld, li))
    return out


def parse_triples(values: Optional[List[str]]) -> List[Tuple[int, int, int]]:
    out: List[Tuple[int, int, int]] = []
    for value in values or []:
        try:
            ld, li, lf = (int(x) for x in value.split(":"))
        except Exception as exc:
            raise ValueError(
                f"Bad triple '{value}'. Use e.g. 3:6:16 4:8:16"
            ) from exc
        if not (0 < ld < li < lf):
            raise ValueError(f"Need 0 < Ld < Li < Lf, got {ld}:{li}:{lf}")
        out.append((ld, li, lf))
    return out


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_early_exit_model(base, n_layers: int):
    """Clone module structure but share Parameter objects with base.

    This follows the LayerSkip model-card idea: keep only the first n_layers,
    while retaining the model's final norm + LM head as the early-exit head.
    """
    memo = {id(p): p for p in base.parameters()}
    model = deepcopy(base, memo=memo)

    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError(
            "This script currently expects a Llama-like HF model with model.layers."
        )

    total = len(model.model.layers)
    if n_layers > total:
        raise ValueError(f"Requested {n_layers} layers, but model has {total}.")

    model.model.layers = torch.nn.ModuleList(list(model.model.layers[:n_layers]))
    # The actual modules define execution, but keeping config aligned avoids cache quirks.
    if hasattr(model.config, "num_hidden_layers"):
        model.config.num_hidden_layers = n_layers
    model.eval()
    return model


def safe_softmax(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    # HSD uses probability ratios. Compute softmax in fp32, then promote to fp64
    # for the small probability-mass calculations to reduce numerical error.
    if temperature <= 0:
        raise ValueError("temperature must be > 0 for lossless sampling diagnostics")
    return torch.softmax(logits.float() / temperature, dim=-1).double()


# -----------------------------
# Draft + intermediate probabilities
# -----------------------------


@torch.inference_mode()
def sample_draft_block(
    draft_model,
    input_ids: torch.Tensor,
    gamma: int,
    temperature: float,
    generator: torch.Generator,
):
    """Sample gamma tokens from q=L_d and return q distributions for each step.

    Returns:
      draft_ids: [gamma]
      q_probs:   [gamma, vocab]
    """
    out = draft_model(input_ids=input_ids, use_cache=True, return_dict=True)
    past = out.past_key_values
    logits = out.logits[:, -1, :]

    sampled: List[torch.Tensor] = []
    q_list: List[torch.Tensor] = []

    for step in range(gamma):
        probs = safe_softmax(logits, temperature)  # [1,V], fp64
        q_list.append(probs.squeeze(0))

        # multinomial does not support a CUDA generator created on CPU.
        token = torch.multinomial(
            probs.float(), num_samples=1, generator=generator
        )  # [1,1]
        sampled.append(token.squeeze(0).squeeze(0))

        if step < gamma - 1:
            out = draft_model(
                input_ids=token,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = out.past_key_values
            logits = out.logits[:, -1, :]

    draft_ids = torch.stack(sampled, dim=0)  # [gamma]
    q_probs = torch.stack(q_list, dim=0)     # [gamma,V]
    return draft_ids, q_probs


@torch.inference_mode()
def intermediate_probs_for_draft(
    intermediate_model,
    input_ids: torch.Tensor,
    draft_ids: torch.Tensor,
    temperature: float,
):
    """Compute p=L_i distributions at the same prefixes used by the draft.

    One context prefill gives p(x1 | context). Then one parallel pass over
    x1..x_{gamma-1} gives p(x2), ..., p(x_gamma).
    """
    gamma = int(draft_ids.numel())

    out = intermediate_model(input_ids=input_ids, use_cache=True, return_dict=True)
    p0 = safe_softmax(out.logits[:, -1, :], temperature).squeeze(0)

    if gamma == 1:
        return p0.unsqueeze(0)

    prefix_draft = draft_ids[:-1].view(1, -1)
    out2 = intermediate_model(
        input_ids=prefix_draft,
        past_key_values=out.past_key_values,
        use_cache=False,
        return_dict=True,
    )
    prest = safe_softmax(out2.logits.squeeze(0), temperature)  # [gamma-1,V]
    return torch.cat([p0.unsqueeze(0), prest], dim=0)


@torch.inference_mode()
def proposal_probs_with_bonus(
    model,
    input_ids: torch.Tensor,
    proposal_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Return probabilities for every proposal token plus the bonus position.

    For ``n`` proposal tokens the shape is ``[n + 1, vocab]``. Row ``j``
    predicts proposal token ``j`` and the final row predicts the token after
    the complete proposal.
    """
    out = model(input_ids=input_ids, use_cache=True, return_dict=True)
    first = safe_softmax(out.logits[:, -1, :], temperature).squeeze(0)

    if proposal_ids.numel() == 0:
        return first.unsqueeze(0)

    out2 = model(
        input_ids=proposal_ids.view(1, -1),
        past_key_values=out.past_key_values,
        use_cache=False,
        return_dict=True,
    )
    rest = safe_softmax(out2.logits.squeeze(0), temperature)
    return torch.cat([first.unsqueeze(0), rest], dim=0)


# -----------------------------
# Verification math
# -----------------------------


def tokenwise_stats(
    q_probs: torch.Tensor, p_probs: torch.Tensor, draft_ids: torch.Tensor
) -> Dict[str, float]:
    """Expected stats for standard lossless token-wise speculative verification."""
    gamma = draft_ids.numel()
    rows = torch.arange(gamma, device=draft_ids.device)
    qx = q_probs[rows, draft_ids].clamp_min(1e-300)
    px = p_probs[rows, draft_ids].clamp_min(0.0)

    alpha = torch.clamp(px / qx, max=1.0)  # per-token accept prob

    # E[tau] = sum_{k=1}^gamma P(tau >= k)
    survival = torch.cumprod(alpha, dim=0)
    expected_accept = survival.sum().item()
    full_accept = survival[-1].item()

    return {
        "expected_accept": expected_accept,
        "block_efficiency": expected_accept + 1.0,
        "full_block_accept": full_accept,
        "mean_local_alpha": alpha.mean().item(),
    }


def hsd_acceptance_probabilities(
    q_probs: torch.Tensor, p_probs: torch.Tensor, draft_ids: torch.Tensor
) -> torch.Tensor:
    """Compute HSD prefix-acceptance probabilities h_1..h_gamma.

    Implements the capped-prefix / capped-branch logic from HSD Eq. (16)-(19).
    For t < gamma, h_t is computed from the capped branch divergences over the
    FULL vocabulary of the next-token branch. For t=gamma, h_gamma is the
    capped joint prefix ratio.

    q_probs[t] and p_probs[t] are distributions for token x_{t+1}, conditioned
    on context + x_1..x_t (0-based tensor indexing).
    """
    if q_probs.shape != p_probs.shape:
        raise ValueError(f"q/p shape mismatch: {q_probs.shape} vs {p_probs.shape}")

    gamma, vocab = q_probs.shape
    if draft_ids.numel() != gamma:
        raise ValueError("draft length does not match q/p probability tensors")

    rows = torch.arange(gamma, device=draft_ids.device)
    qx = q_probs[rows, draft_ids].clamp_min(1e-300)
    px = p_probs[rows, draft_ids].clamp_min(1e-300)

    local_ratio = px / qx
    prefix_ratio = torch.cumprod(local_ratio, dim=0)  # r(X_1:t), t=1..gamma
    joint_q = torch.cumprod(qx, dim=0)

    h = torch.zeros(gamma, dtype=torch.float64, device=q_probs.device)

    # h_t for t < gamma. Prefix length t is 1..gamma-1.
    # Branch(X_1:t) enumerates all possible x_{t+1} over the vocabulary.
    for t in range(1, gamma):
        r_prefix = prefix_ratio[t - 1]
        q_prefix = joint_q[t - 1]

        # For candidate child length t+1, m() may use prefix ratios through t.
        cap = torch.maximum(
            torch.tensor(1.0, dtype=torch.float64, device=q_probs.device),
            prefix_ratio[:t].max(),
        )

        q_next = q_probs[t].clamp_min(1e-300)
        p_next = p_probs[t].clamp_min(0.0)

        # r*(X_1:t+1) = r(X_1:t+1) / max(1, max_{i<=t} r(X_1:i))
        r_star_child = (r_prefix * (p_next / q_next)) / cap
        q_joint_child = q_prefix * q_next

        deficient = torch.clamp(r_star_child - 1.0, min=0.0)
        excess = torch.clamp(1.0 - r_star_child, min=0.0)

        d_pq = (deficient * q_joint_child).sum()  # D*_Branch(p,q)
        d_qp = (excess * q_joint_child).sum()     # D*_Branch(q,p)

        # Official implementation numerically caps with max(d_pq, d_qp).
        denom = torch.maximum(d_pq, d_qp)
        if denom.item() <= 1e-30:
            # p and q are effectively identical on this branch.
            h[t - 1] = 1.0
        else:
            h[t - 1] = torch.clamp(d_pq / denom, min=0.0, max=1.0)

    # Full-sequence acceptance: h_gamma = min(r*(X_1:gamma), 1).
    if gamma == 1:
        prior_cap = torch.tensor(1.0, dtype=torch.float64, device=q_probs.device)
    else:
        prior_cap = torch.maximum(
            torch.tensor(1.0, dtype=torch.float64, device=q_probs.device),
            prefix_ratio[:-1].max(),
        )
    r_star_gamma = prefix_ratio[-1] / prior_cap
    h[-1] = torch.clamp(r_star_gamma, min=0.0, max=1.0)

    return h


def hsd_stats(
    q_probs: torch.Tensor, p_probs: torch.Tensor, draft_ids: torch.Tensor
) -> Dict[str, float]:
    """Expected HSD accepted-prefix length under the backward scan."""
    h = hsd_acceptance_probabilities(q_probs, p_probs, draft_ids)
    gamma = h.numel()

    # Backward scan gamma -> 1. At prefix length t, it is selected only if all
    # longer prefixes were rejected, then h_t succeeds.
    prob_reach = torch.tensor(1.0, dtype=torch.float64, device=h.device)
    expected_accept = torch.tensor(0.0, dtype=torch.float64, device=h.device)

    for idx in range(gamma - 1, -1, -1):
        t = idx + 1
        prob_tau_t = prob_reach * h[idx]
        expected_accept = expected_accept + t * prob_tau_t
        prob_reach = prob_reach * (1.0 - h[idx])

    return {
        "expected_accept": expected_accept.item(),
        "block_efficiency": expected_accept.item() + 1.0,
        "full_block_accept": h[-1].item(),
        "mean_h": h.mean().item(),
    }


def _sample_token(probs: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    probs32 = probs.float()
    total = probs32.sum()
    if not torch.isfinite(total) or total.item() <= 0:
        raise RuntimeError("Cannot sample from an empty/non-finite distribution")
    probs32 = probs32 / total
    return torch.multinomial(probs32, 1, generator=generator).squeeze(0)


def _uniform(device: torch.device, generator: torch.Generator) -> float:
    return torch.rand((), device=device, generator=generator).item()


def tokenwise_sample_block(
    q_probs: torch.Tensor,
    p_probs_with_bonus: torch.Tensor,
    draft_ids: torch.Tensor,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, int, bool]:
    """Realize one lossless tokenwise speculative-sampling step.

    Returns the accepted draft prefix followed by one residual/bonus token,
    the accepted-prefix length, and whether the complete draft was accepted.
    """
    gamma = int(draft_ids.numel())
    if q_probs.shape[0] != gamma or p_probs_with_bonus.shape[0] != gamma + 1:
        raise ValueError("tokenwise sampler received inconsistent probability shapes")

    accepted: List[torch.Tensor] = []
    for idx in range(gamma):
        token = draft_ids[idx]
        qx = q_probs[idx, token].clamp_min(1e-300)
        px = p_probs_with_bonus[idx, token].clamp_min(0.0)
        alpha = min(1.0, (px / qx).item())
        if _uniform(draft_ids.device, generator) <= alpha:
            accepted.append(token)
            continue

        residual = torch.clamp(p_probs_with_bonus[idx] - q_probs[idx], min=0.0)
        if residual.sum().item() <= 1e-30:
            residual = p_probs_with_bonus[idx]
        correction = _sample_token(residual, generator)
        return torch.stack(accepted + [correction]), idx, False

    bonus = _sample_token(p_probs_with_bonus[-1], generator)
    return torch.stack(accepted + [bonus]), gamma, True


def capped_resampling_distribution(
    q_probs: torch.Tensor,
    p_probs: torch.Tensor,
    draft_ids: torch.Tensor,
    accepted_prefix: int,
) -> torch.Tensor:
    """Equation (20) of HSD for the first token after an accepted prefix."""
    gamma = int(draft_ids.numel())
    tau = accepted_prefix
    if not (0 <= tau < gamma):
        raise ValueError("capped resampling is only defined after a rejection")

    if tau == 0:
        q_prefix = torch.tensor(1.0, dtype=torch.float64, device=draft_ids.device)
        r_prefix = torch.tensor(1.0, dtype=torch.float64, device=draft_ids.device)
        cap = torch.tensor(1.0, dtype=torch.float64, device=draft_ids.device)
    else:
        rows = torch.arange(tau, device=draft_ids.device)
        qx = q_probs[rows, draft_ids[:tau]].clamp_min(1e-300)
        px = p_probs[rows, draft_ids[:tau]].clamp_min(1e-300)
        prefix_ratios = torch.cumprod(px / qx, dim=0)
        q_prefix = torch.prod(qx)
        r_prefix = prefix_ratios[-1]
        cap = torch.maximum(
            torch.tensor(1.0, dtype=torch.float64, device=draft_ids.device),
            prefix_ratios.max(),
        )

    q_next = q_probs[tau].clamp_min(1e-300)
    p_next = p_probs[tau].clamp_min(0.0)
    r_star_child = (r_prefix * (p_next / q_next)) / cap
    weights = q_prefix * q_next * torch.clamp(r_star_child - 1.0, min=0.0)

    if weights.sum().item() <= 1e-30:
        # This should only be reached through floating-point degeneracy.
        weights = p_next
    return weights / weights.sum()


def hsd_sample_block(
    q_probs: torch.Tensor,
    p_probs_with_bonus: torch.Tensor,
    draft_ids: torch.Tensor,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, int, bool]:
    """Realize HSD Algorithm 2: backward scan plus one capped resample."""
    gamma = int(draft_ids.numel())
    if q_probs.shape[0] != gamma or p_probs_with_bonus.shape[0] != gamma + 1:
        raise ValueError("HSD sampler received inconsistent probability shapes")

    p_probs = p_probs_with_bonus[:-1]
    h = hsd_acceptance_probabilities(q_probs, p_probs, draft_ids)
    tau = 0
    for idx in range(gamma - 1, -1, -1):
        if _uniform(draft_ids.device, generator) <= h[idx].item():
            tau = idx + 1
            break

    prefix = [draft_ids[idx] for idx in range(tau)]
    if tau == gamma:
        bonus = _sample_token(p_probs_with_bonus[-1], generator)
        return torch.stack(prefix + [bonus]), tau, True

    correction_dist = capped_resampling_distribution(
        q_probs, p_probs, draft_ids, accepted_prefix=tau
    )
    correction = _sample_token(correction_dist, generator)
    return torch.stack(prefix + [correction]), tau, False



# -----------------------------
# Three-level novelty diagnostics
# -----------------------------


def _expected_backward_prefix_from_h(h: torch.Tensor) -> float:
    """Expected selected prefix length under an HSD-style backward scan.

    This helper is also used for *heuristic* joint h-vectors below.  Those
    heuristics are diagnostics only; they are not claimed to define a lossless
    sampler.
    """
    reach = torch.tensor(1.0, dtype=torch.float64, device=h.device)
    expected = torch.tensor(0.0, dtype=torch.float64, device=h.device)
    for idx in range(h.numel() - 1, -1, -1):
        prob_tau = reach * h[idx]
        expected = expected + (idx + 1) * prob_tau
        reach = reach * (1.0 - h[idx])
    return float(expected.item())


def _pearson(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return math.nan
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-20 or vy <= 1e-20:
        return math.nan
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def _quantile(xs: List[float], q: float) -> float:
    if not xs:
        return math.nan
    vals = sorted(xs)
    if len(vals) == 1:
        return vals[0]
    pos = max(0.0, min(1.0, q)) * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    w = pos - lo
    return vals[lo] * (1.0 - w) + vals[hi] * w


@dataclass
class NoveltyWindowRecord:
    method: str
    accepted_final: int
    full_final: int
    ni: int
    expected_final_tw: float
    mean_alpha_if: float
    top1_if: float
    tv_di: float
    tv_if: float
    tv_df: float
    consistent_fraction: float
    overshoot_fraction: float
    undershoot_fraction: float
    di_h_mean: float
    di_h_full: float
    di_h_expected: float
    if_h_mean: float
    if_h_full: float
    if_h_expected: float
    df_h_mean: float
    df_h_full: float
    df_h_expected: float
    joint_product_expected: float
    joint_min_expected: float
    joint_geom_expected: float
    mean_log_di: float
    mean_log_if: float
    mean_log_df: float
    monotonic_up_tokens: int
    monotonic_down_tokens: int
    overshoot_tokens: int
    undershoot_tokens: int
    monotonic_up_alpha_sum: float
    monotonic_down_alpha_sum: float
    overshoot_alpha_sum: float
    undershoot_alpha_sum: float


@dataclass
class AdaptiveLayerPoint:
    depth: int
    inter_h_mean: float
    inter_h_full: float
    from_ld_h_mean: float
    final_tw_expected: float
    final_hsd_expected: float
    final_top1: float
    tv_to_final: float


@dataclass
class AdaptiveWindowRecord:
    method: str
    fixed_li: int
    points: List[AdaptiveLayerPoint]


@torch.inference_mode()
def analyze_three_level_window(
    method: str,
    draft_model,
    context: torch.Tensor,
    window: torch.Tensor,
    p_li_with_bonus: torch.Tensor,
    p_lf_with_bonus: torch.Tensor,
    accepted_final: int,
    full_final: bool,
    temperature: float,
) -> Tuple[NoveltyWindowRecord, torch.Tensor]:
    """Analyze one tentative window using L_d, L_i and L_f simultaneously.

    The window is generated exactly as in the baseline HiSpec pipeline.  We then
    teacher-force that *same* window through L_d so all three distributions are
    aligned position-by-position.  This is a diagnostic: p_d on an L_i-sampled
    window is not the proposal law of the existing sampler.
    """
    gamma = int(window.numel())
    p_d = proposal_probs_with_bonus(
        draft_model, context, window, temperature
    )[:-1]
    p_i = p_li_with_bonus[:-1]
    p_f = p_lf_with_bonus[:-1]

    rows = torch.arange(gamma, device=window.device)
    pdx = p_d[rows, window].clamp_min(1e-300)
    pix = p_i[rows, window].clamp_min(1e-300)
    pfx = p_f[rows, window].clamp_min(1e-300)

    alpha_if = torch.clamp(pfx / pix, max=1.0)
    expected_final_tw = torch.cumprod(alpha_if, dim=0).sum().item()
    mean_alpha_if = alpha_if.mean().item()

    top1_if = (p_i.argmax(dim=-1) == p_f.argmax(dim=-1)).double().mean().item()
    tv_di = (0.5 * torch.abs(p_d - p_i).sum(dim=-1)).mean().item()
    tv_if = (0.5 * torch.abs(p_i - p_f).sum(dim=-1)).mean().item()
    tv_df = (0.5 * torch.abs(p_d - p_f).sum(dim=-1)).mean().item()

    di = torch.log(pix) - torch.log(pdx)
    iff = torch.log(pfx) - torch.log(pix)
    df = torch.log(pfx) - torch.log(pdx)

    # Candidate-token probability trajectory classes.  These deliberately use
    # the probability of the tentative token, not argmax identity.
    up = (pix >= pdx) & (pfx >= pix)
    down = (pix <= pdx) & (pfx <= pix)
    overshoot = (pix > pdx) & (pfx < pix)
    undershoot = (pix < pdx) & (pfx > pix)
    consistent = up | down

    h_di = hsd_acceptance_probabilities(p_d, p_i, window)
    h_if = hsd_acceptance_probabilities(p_i, p_f, window)
    h_df = hsd_acceptance_probabilities(p_d, p_f, window)

    # Candidate joint rules to study.  They are *not* used to generate tokens.
    # If one is consistently superior as a predictor/upper-bound, it is a good
    # target for a later exact coupling derivation.
    h_product = torch.clamp(h_di * h_if, min=0.0, max=1.0)
    h_min = torch.minimum(h_di, h_if)
    h_geom = torch.sqrt(torch.clamp(h_di * h_if, min=0.0, max=1.0))

    def _count(mask: torch.Tensor) -> int:
        return int(mask.sum().item())

    def _alpha_sum(mask: torch.Tensor) -> float:
        if not bool(mask.any()):
            return 0.0
        return float(alpha_if[mask].sum().item())

    record = NoveltyWindowRecord(
        method=method,
        accepted_final=int(accepted_final),
        full_final=int(bool(full_final)),
        ni=gamma,
        expected_final_tw=float(expected_final_tw),
        mean_alpha_if=float(mean_alpha_if),
        top1_if=float(top1_if),
        tv_di=float(tv_di),
        tv_if=float(tv_if),
        tv_df=float(tv_df),
        consistent_fraction=float(consistent.double().mean().item()),
        overshoot_fraction=float(overshoot.double().mean().item()),
        undershoot_fraction=float(undershoot.double().mean().item()),
        di_h_mean=float(h_di.mean().item()),
        di_h_full=float(h_di[-1].item()),
        di_h_expected=_expected_backward_prefix_from_h(h_di),
        if_h_mean=float(h_if.mean().item()),
        if_h_full=float(h_if[-1].item()),
        if_h_expected=_expected_backward_prefix_from_h(h_if),
        df_h_mean=float(h_df.mean().item()),
        df_h_full=float(h_df[-1].item()),
        df_h_expected=_expected_backward_prefix_from_h(h_df),
        joint_product_expected=_expected_backward_prefix_from_h(h_product),
        joint_min_expected=_expected_backward_prefix_from_h(h_min),
        joint_geom_expected=_expected_backward_prefix_from_h(h_geom),
        mean_log_di=float(di.mean().item()),
        mean_log_if=float(iff.mean().item()),
        mean_log_df=float(df.mean().item()),
        monotonic_up_tokens=_count(up),
        monotonic_down_tokens=_count(down),
        overshoot_tokens=_count(overshoot),
        undershoot_tokens=_count(undershoot),
        monotonic_up_alpha_sum=_alpha_sum(up),
        monotonic_down_alpha_sum=_alpha_sum(down),
        overshoot_alpha_sum=_alpha_sum(overshoot),
        undershoot_alpha_sum=_alpha_sum(undershoot),
    )
    return record, p_d


@torch.inference_mode()
def analyze_adaptive_depth_window(
    method: str,
    ld: int,
    fixed_li: int,
    candidate_models: Dict[int, Any],
    context: torch.Tensor,
    window: torch.Tensor,
    p_d: torch.Tensor,
    p_li_with_bonus: torch.Tensor,
    p_lf_with_bonus: torch.Tensor,
    temperature: float,
) -> AdaptiveWindowRecord:
    """Offline adaptive-depth study using the same tentative window.

    We score several early-exit depths and ask how early a verifier could stop.
    The stopping policy later uses only inter-level HSD statistics; L_f metrics
    here are evaluation labels, not inputs to the policy.
    """
    p_f = p_lf_with_bonus[:-1]
    fixed_pi = p_li_with_bonus[:-1]
    depths = sorted(candidate_models)
    points: List[AdaptiveLayerPoint] = []
    prev_probs = p_d

    for depth in depths:
        if depth == fixed_li:
            p_cur = fixed_pi
        else:
            p_cur = proposal_probs_with_bonus(
                candidate_models[depth], context, window, temperature
            )[:-1]

        h_inter = hsd_acceptance_probabilities(prev_probs, p_cur, window)
        h_from_ld = hsd_acceptance_probabilities(p_d, p_cur, window)
        h_to_final = hsd_acceptance_probabilities(p_cur, p_f, window)
        tw = tokenwise_stats(p_cur, p_f, window)
        top1 = (p_cur.argmax(dim=-1) == p_f.argmax(dim=-1)).double().mean().item()
        tv = (0.5 * torch.abs(p_cur - p_f).sum(dim=-1)).mean().item()
        points.append(
            AdaptiveLayerPoint(
                depth=depth,
                inter_h_mean=float(h_inter.mean().item()),
                inter_h_full=float(h_inter[-1].item()),
                from_ld_h_mean=float(h_from_ld.mean().item()),
                final_tw_expected=float(tw["expected_accept"]),
                final_hsd_expected=_expected_backward_prefix_from_h(h_to_final),
                final_top1=float(top1),
                tv_to_final=float(tv),
            )
        )
        prev_probs = p_cur

    return AdaptiveWindowRecord(method=method, fixed_li=fixed_li, points=points)


def _write_novelty_csv(path: str, rows: List[NoveltyWindowRecord]) -> None:
    if not path or not rows:
        return
    fieldnames = list(rows[0].__dataclass_fields__.keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: getattr(row, name) for name in fieldnames})


def print_novelty_diagnostics(
    triple_label: str,
    rows_by_method: Dict[str, List[NoveltyWindowRecord]],
) -> None:
    print("\n=== Three-level novelty diagnostic: Ld -> Li -> Lf ===")
    print(
        "method      N   actFinal  expTW  hDIexp  hIFexp  hDFexp  "
        "jointProd  jointMin  jointGeom  TVdi   TVif   consistent  overshoot"
    )
    print("-" * 126)
    for method, rows in rows_by_method.items():
        if not rows:
            continue
        print(
            f"{method:<10} {len(rows):4d}  "
            f"{mean([r.accepted_final for r in rows]):8.3f}  "
            f"{mean([r.expected_final_tw for r in rows]):5.2f}  "
            f"{mean([r.di_h_expected for r in rows]):6.2f}  "
            f"{mean([r.if_h_expected for r in rows]):6.2f}  "
            f"{mean([r.df_h_expected for r in rows]):6.2f}  "
            f"{mean([r.joint_product_expected for r in rows]):9.2f}  "
            f"{mean([r.joint_min_expected for r in rows]):8.2f}  "
            f"{mean([r.joint_geom_expected for r in rows]):9.2f}  "
            f"{mean([r.tv_di for r in rows]):5.3f}  "
            f"{mean([r.tv_if for r in rows]):5.3f}  "
            f"{100*mean([r.consistent_fraction for r in rows]):9.1f}%  "
            f"{100*mean([r.overshoot_fraction for r in rows]):8.1f}%"
        )

    print("\n=== Does intermediate HSD quality predict final survival? ===")
    print(
        "method      corr(hDImean,expTW)  corr(hDIfull,expTW)  "
        "corr(hDIexp,expTW)  corr(consistency,expTW)  corr(overshoot,expTW)  corr(TVif,expTW)"
    )
    print("-" * 135)
    for method, rows in rows_by_method.items():
        if not rows:
            continue
        target = [r.expected_final_tw for r in rows]
        vals = [
            _pearson([r.di_h_mean for r in rows], target),
            _pearson([r.di_h_full for r in rows], target),
            _pearson([r.di_h_expected for r in rows], target),
            _pearson([r.consistent_fraction for r in rows], target),
            _pearson([r.overshoot_fraction for r in rows], target),
            _pearson([r.tv_if for r in rows], target),
        ]
        print(
            f"{method:<10} " + "  ".join(f"{v:+20.3f}" if not math.isnan(v) else f"{'n/a':>20}" for v in vals)
        )

    print("\n=== Cross-level candidate-token trajectory classes ===")
    print("method      class            tokens   share    mean final alpha")
    print("-" * 66)
    for method, rows in rows_by_method.items():
        if not rows:
            continue
        total = sum(r.ni for r in rows)
        classes = [
            ("monotonic-up", "monotonic_up_tokens", "monotonic_up_alpha_sum"),
            ("monotonic-down", "monotonic_down_tokens", "monotonic_down_alpha_sum"),
            ("Li-overshoot", "overshoot_tokens", "overshoot_alpha_sum"),
            ("Li-undershoot", "undershoot_tokens", "undershoot_alpha_sum"),
        ]
        for label, count_name, alpha_name in classes:
            count = sum(getattr(r, count_name) for r in rows)
            alpha_sum = sum(getattr(r, alpha_name) for r in rows)
            alpha = alpha_sum / max(1, count)
            print(
                f"{method:<10} {label:<16} {count:7d}  "
                f"{100*count/max(1,total):6.1f}%   {alpha:8.3f}"
            )

    print("\n=== Intermediate-HSD bins -> final survival ===")
    bins = [(0.0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 0.90), (0.90, 1.000001)]
    for method, rows in rows_by_method.items():
        if not rows:
            continue
        print(f"{method}:")
        for lo, hi in bins:
            subset = [r for r in rows if lo <= r.di_h_mean < hi]
            if not subset:
                continue
            print(
                f"  hDImean [{lo:.2f},{min(hi,1.0):.2f}): n={len(subset):4d}, "
                f"E[final TW prefix]={mean([r.expected_final_tw for r in subset]):.3f}, "
                f"actual accepted={mean([r.accepted_final for r in subset]):.3f}, "
                f"overshoot={100*mean([r.overshoot_fraction for r in subset]):.1f}%"
            )

        # A direct quantitative version of the motivating failure mode:
        # locally strong Ld->Li windows that are weak at Li->Lf.
        di_vals = [r.di_h_expected for r in rows]
        fin_vals = [r.expected_final_tw for r in rows]
        q75 = _quantile(di_vals, 0.75)
        q25 = _quantile(fin_vals, 0.25)
        traps = [r for r in rows if r.di_h_expected >= q75 and r.expected_final_tw <= q25]
        print(
            f"  local-optimum trap: {len(traps)}/{len(rows)} "
            f"({100*len(traps)/max(1,len(rows)):.1f}%) windows are top-quartile "
            f"Ld->Li HSD but bottom-quartile final survival."
        )


def print_adaptive_depth_diagnostics(
    records_by_method: Dict[str, List[AdaptiveWindowRecord]],
    thresholds: List[float],
    quality_tolerance: float,
) -> None:
    print("\n=== Adaptive intermediate-depth diagnostic ===")
    for method, records in records_by_method.items():
        if not records:
            continue
        depths = sorted({p.depth for r in records for p in r.points})
        print(f"\n{method}: candidate depths = {depths}")
        print("depth   windows   finalTW   finalHSD  top1(Lf)  TV->Lf  interH")
        print("-" * 72)
        for depth in depths:
            pts = [p for r in records for p in r.points if p.depth == depth]
            print(
                f"L{depth:<5} {len(pts):7d}  "
                f"{mean([p.final_tw_expected for p in pts]):7.3f}  "
                f"{mean([p.final_hsd_expected for p in pts]):8.3f}  "
                f"{100*mean([p.final_top1 for p in pts]):7.2f}%  "
                f"{mean([p.tv_to_final for p in pts]):6.3f}  "
                f"{mean([p.inter_h_mean for p in pts]):6.3f}"
            )

        print("\n  HSD-stability stopping policies (evaluation uses L_f only after the choice):")
        print("  threshold  avgDepth  depth/fixed  finalTW  delta-vs-fixed  top1(Lf)")
        print("  " + "-" * 72)
        for threshold in thresholds:
            chosen: List[AdaptiveLayerPoint] = []
            fixed: List[AdaptiveLayerPoint] = []
            for rec in records:
                pts = sorted(rec.points, key=lambda p: p.depth)
                pick = pts[-1]
                for pt in pts:
                    if pt.inter_h_mean >= threshold:
                        pick = pt
                        break
                chosen.append(pick)
                fixed_pt = min(pts, key=lambda p: abs(p.depth - rec.fixed_li))
                fixed.append(fixed_pt)
            avg_depth = mean([p.depth for p in chosen])
            fixed_depth = mean([p.depth for p in fixed])
            chosen_q = mean([p.final_tw_expected for p in chosen])
            fixed_q = mean([p.final_tw_expected for p in fixed])
            print(
                f"  {threshold:8.2f}  {avg_depth:8.2f}  "
                f"{avg_depth/max(1e-9,fixed_depth):10.3f}  "
                f"{chosen_q:7.3f}  {chosen_q-fixed_q:+14.3f}  "
                f"{100*mean([p.final_top1 for p in chosen]):7.2f}%"
            )

        # Oracle lower-depth upper bound: shallowest depth whose final expected
        # tokenwise acceptance is within `quality_tolerance` tokens of fixed L_i.
        oracle: List[AdaptiveLayerPoint] = []
        fixed_pts: List[AdaptiveLayerPoint] = []
        for rec in records:
            pts = sorted(rec.points, key=lambda p: p.depth)
            fixed_pt = min(pts, key=lambda p: abs(p.depth - rec.fixed_li))
            fixed_pts.append(fixed_pt)
            target = fixed_pt.final_tw_expected - quality_tolerance
            eligible = [p for p in pts if p.final_tw_expected >= target]
            oracle.append(eligible[0] if eligible else pts[-1])
        print(
            f"  oracle (within {quality_tolerance:.2f} final tokens of fixed Li): "
            f"avg depth={mean([p.depth for p in oracle]):.2f}, "
            f"finalTW={mean([p.final_tw_expected for p in oracle]):.3f}, "
            f"fixedTW={mean([p.final_tw_expected for p in fixed_pts]):.3f}."
        )


# -----------------------------
# Dataset contexts
# -----------------------------


CORPUS_PRESETS = {
    "wikitext2": ("Salesforce/wikitext", "wikitext-2-raw-v1", "validation", "text"),
    "humaneval": ("openai/openai_humaneval", None, "test", "prompt"),
    "ultrachat": ("HuggingFaceH4/ultrachat_200k", None, "test_sft", "messages"),
}


def _record_to_text(record: Dict[str, Any], field: str, tokenizer) -> str:
    value = record.get(field)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            if hasattr(tokenizer, "apply_chat_template"):
                try:
                    return tokenizer.apply_chat_template(
                        value, tokenize=False, add_generation_prompt=False
                    )
                except Exception:
                    pass
            return "\n".join(
                f"{item.get('role', 'unknown')}: {item.get('content', '')}"
                for item in value
            )
        return "\n".join(str(item) for item in value)
    return str(value)


def build_contexts(
    tokenizer,
    n_contexts: int,
    context_tokens: int,
    seed: int,
    corpus: str = "wikitext2",
    dataset_name: Optional[str] = None,
    dataset_config: Optional[str] = None,
    dataset_split: Optional[str] = None,
    text_field: Optional[str] = None,
):
    """Build random fixed-length contexts without tokenizing one giant string.

    Tokenizing records separately avoids the misleading tokenizer warning about
    feeding a corpus longer than the model's maximum context length.
    """
    if dataset_name:
        name = dataset_name
        config = dataset_config
        split = dataset_split or "train"
        field = text_field or "text"
        corpus_label = dataset_name
    else:
        if corpus not in CORPUS_PRESETS:
            raise ValueError(
                f"Unknown corpus '{corpus}'. Choose {sorted(CORPUS_PRESETS)} "
                "or provide --dataset-name."
            )
        name, config, split, field = CORPUS_PRESETS[corpus]
        split = dataset_split or split
        field = text_field or field
        corpus_label = corpus

    print(f"Loading corpus: {corpus_label} ({name}, split={split}, field={field})")
    load_kwargs: Dict[str, Any] = {"path": name, "split": split}
    if config:
        load_kwargs["name"] = config
    ds = load_dataset(**load_kwargs)

    token_chunks: List[torch.Tensor] = []
    eos = tokenizer.eos_token_id
    for record in ds:
        text = _record_to_text(record, field, tokenizer)
        if not text or text.isspace():
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        if eos is not None:
            ids.append(eos)
        token_chunks.append(torch.tensor(ids, dtype=torch.long))

    if not token_chunks:
        raise RuntimeError(f"No usable text found in field '{field}'")
    rng = random.Random(seed)
    eligible = [chunk for chunk in token_chunks if chunk.numel() >= context_tokens + 1]
    if eligible:
        contexts = []
        order = list(range(len(eligible)))
        rng.shuffle(order)
        for idx in range(n_contexts):
            if idx > 0 and idx % len(order) == 0:
                rng.shuffle(order)
            chunk = eligible[order[idx % len(order)]]
            start = rng.randint(0, chunk.numel() - context_tokens - 1)
            contexts.append(chunk[start : start + context_tokens].clone())
        return contexts

    # Fallback for corpora made entirely of short records. EOS separators make
    # record boundaries explicit even when a context spans multiple records.
    pool = torch.cat(token_chunks)
    if pool.numel() < context_tokens + 2:
        raise RuntimeError("Dataset tokenization produced too few tokens.")
    max_start = pool.numel() - context_tokens - 1
    return [
        pool[s : s + context_tokens].clone()
        for s in (rng.randint(0, max_start) for _ in range(n_contexts))
    ]


# -----------------------------
# Evaluation
# -----------------------------


@dataclass
class PairSummary:
    ld: int
    li: int
    gamma: int
    temperature: float
    tw_accept: float
    hsd_accept: float
    tw_be: float
    hsd_be: float
    be_gain_pct: float
    tw_full: float
    hsd_full: float
    li_calls_per_100_tw: float
    li_calls_per_100_hsd: float


def mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))


@torch.inference_mode()
def evaluate_pair(
    base,
    contexts: List[torch.Tensor],
    ld: int,
    li: int,
    gamma: int,
    temperature: float,
    drafts_per_context: int,
    device: torch.device,
    seed: int,
) -> PairSummary:
    draft_model = make_early_exit_model(base, ld)
    intermediate_model = make_early_exit_model(base, li)

    draft_model.to(device)
    intermediate_model.to(device)
    draft_model.eval()
    intermediate_model.eval()

    tw_accepts, hsd_accepts = [], []
    tw_bes, hsd_bes = [], []
    tw_fulls, hsd_fulls = [], []

    # Device-specific generator so sampling is reproducible.
    gen = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
    gen.manual_seed(seed + 1000 * ld + li)

    total_blocks = len(contexts) * drafts_per_context
    done = 0

    for c_idx, context in enumerate(contexts):
        context = context.view(1, -1).to(device)

        for _ in range(drafts_per_context):
            draft_ids, q_probs = sample_draft_block(
                draft_model,
                context,
                gamma=gamma,
                temperature=temperature,
                generator=gen,
            )
            p_probs = intermediate_probs_for_draft(
                intermediate_model,
                context,
                draft_ids,
                temperature=temperature,
            )

            tw = tokenwise_stats(q_probs, p_probs, draft_ids)
            hs = hsd_stats(q_probs, p_probs, draft_ids)

            tw_accepts.append(tw["expected_accept"])
            hsd_accepts.append(hs["expected_accept"])
            tw_bes.append(tw["block_efficiency"])
            hsd_bes.append(hs["block_efficiency"])
            tw_fulls.append(tw["full_block_accept"])
            hsd_fulls.append(hs["full_block_accept"])

            done += 1
            if done % max(1, total_blocks // 10) == 0:
                print(f"  L{ld}->L{li}: {done}/{total_blocks} blocks")

            del q_probs, p_probs, draft_ids

    tw_be = mean(tw_bes)
    hsd_be = mean(hsd_bes)
    gain = 100.0 * (hsd_be / tw_be - 1.0)

    summary = PairSummary(
        ld=ld,
        li=li,
        gamma=gamma,
        temperature=temperature,
        tw_accept=mean(tw_accepts),
        hsd_accept=mean(hsd_accepts),
        tw_be=tw_be,
        hsd_be=hsd_be,
        be_gain_pct=gain,
        tw_full=mean(tw_fulls),
        hsd_full=mean(hsd_fulls),
        li_calls_per_100_tw=100.0 / tw_be,
        li_calls_per_100_hsd=100.0 / hsd_be,
    )

    del draft_model, intermediate_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def print_table(rows: List[PairSummary]) -> None:
    print("\n=== Ld -> Li verification results ===")
    header = (
        "pair       g     T   TW acc   HSD acc   TW BE   HSD BE   BE gain   "
        "TW full   HSD full   Li calls/100 (TW->HSD)"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"L{r.ld}->L{r.li:<3} {r.gamma:3d}  {r.temperature:4.2f}  "
            f"{r.tw_accept:7.3f}  {r.hsd_accept:7.3f}  "
            f"{r.tw_be:6.3f}  {r.hsd_be:6.3f}  "
            f"{r.be_gain_pct:+7.2f}%  "
            f"{100*r.tw_full:7.2f}%  {100*r.hsd_full:8.2f}%  "
            f"{r.li_calls_per_100_tw:6.2f}->{r.li_calls_per_100_hsd:6.2f}"
        )


def screening_recommendation(rows: List[PairSummary]) -> None:
    best = max(rows, key=lambda r: r.be_gain_pct)
    g = best.be_gain_pct

    print("\n=== Screening decision ===")
    print(
        f"Best pair: L{best.ld}->L{best.li}, "
        f"Block-Efficiency gain = {g:+.2f}%"
    )

    # This is intentionally a screening heuristic, not a theorem.
    if g >= 5.0:
        verdict = "STRONG SIGNAL: worth a flagship-model pilot."
    elif g >= 3.0:
        verdict = "PROMISING: worth a limited flagship-model pilot."
    elif g >= 1.0:
        verdict = (
            "BORDERLINE: first increase contexts / try a second task before paying "
            "for a flagship run."
        )
    else:
        verdict = (
            "WEAK SIGNAL: this toy test does not justify a flagship run yet. "
            "Try a better Ld/Li pair or a task with a different draft/verify gap."
        )

    print(verdict)
    print(
        "Interpretation: HSD is useful here only if it increases tentative tokens "
        "per L_i verification call enough to offset its tiny probability-mass bookkeeping."
    )
    print(
        "Important: this script does NOT benchmark L_i -> L_f, hidden-state/KV reuse, "
        "or true end-to-end tokens/sec. If this screen is positive, those are the next experiment."
    )


@dataclass
class PipelineMethodSummary:
    method: str
    tentative_tokens: int
    final_tokens: int
    li_calls: int
    lf_calls: int
    li_accepted_draft_tokens: int
    lf_accepted_tentative_tokens: int
    li_full_blocks: int
    lf_full_windows: int
    elapsed_seconds: float

    @property
    def li_be(self) -> float:
        return self.tentative_tokens / max(1, self.li_calls)

    @property
    def final_be(self) -> float:
        return self.final_tokens / max(1, self.lf_calls)

    @property
    def li_calls_per_100(self) -> float:
        return 100.0 * self.li_calls / max(1, self.final_tokens)

    @property
    def lf_calls_per_100(self) -> float:
        return 100.0 * self.lf_calls / max(1, self.final_tokens)

    @property
    def li_full_rate(self) -> float:
        return self.li_full_blocks / max(1, self.li_calls)

    @property
    def lf_full_rate(self) -> float:
        return self.lf_full_windows / max(1, self.lf_calls)

    @property
    def tokens_per_second(self) -> float:
        return self.final_tokens / max(1e-12, self.elapsed_seconds)


@dataclass
class TripleSummary:
    ld: int
    li: int
    lf: int
    gamma: int
    temperature: float
    ni: int
    tokenwise: PipelineMethodSummary
    hsd: PipelineMethodSummary
    novelty_records: Dict[str, List[NoveltyWindowRecord]] = field(default_factory=dict)
    adaptive_records: Dict[str, List[AdaptiveWindowRecord]] = field(default_factory=dict)

    @property
    def li_gain_pct(self) -> float:
        return 100.0 * (self.hsd.li_be / self.tokenwise.li_be - 1.0)

    @property
    def final_gain_pct(self) -> float:
        return 100.0 * (self.hsd.final_be / self.tokenwise.final_be - 1.0)

    @property
    def extra_token_survival(self) -> float:
        delta_i = self.hsd.li_be - self.tokenwise.li_be
        if abs(delta_i) <= 1e-12:
            return math.nan
        return (self.hsd.final_be - self.tokenwise.final_be) / delta_i


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def generate_tentative_window(
    method: str,
    draft_model,
    intermediate_model,
    context: torch.Tensor,
    gamma: int,
    ni: int,
    temperature: float,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, int, int, int]:
    """Generate exactly ``ni`` tentative L_i-distributed tokens."""
    tentative: List[torch.Tensor] = []
    li_calls = 0
    accepted_draft_tokens = 0
    full_blocks = 0

    while len(tentative) < ni:
        if tentative:
            prefix = torch.cat([context, torch.stack(tentative).view(1, -1)], dim=1)
        else:
            prefix = context

        draft_ids, q_probs = sample_draft_block(
            draft_model,
            prefix,
            gamma=gamma,
            temperature=temperature,
            generator=generator,
        )
        p_probs = proposal_probs_with_bonus(
            intermediate_model,
            prefix,
            draft_ids,
            temperature=temperature,
        )

        if method == "tokenwise":
            block, accepted, full = tokenwise_sample_block(
                q_probs, p_probs, draft_ids, generator
            )
        elif method == "hsd":
            block, accepted, full = hsd_sample_block(
                q_probs, p_probs, draft_ids, generator
            )
        else:
            raise ValueError(f"Unknown pipeline method: {method}")

        remaining = ni - len(tentative)
        tentative.extend(block[:remaining].unbind(0))
        li_calls += 1
        accepted_draft_tokens += min(accepted, remaining)
        full_blocks += int(full)

        del draft_ids, q_probs, p_probs, block

    return (
        torch.stack(tentative),
        li_calls,
        accepted_draft_tokens,
        full_blocks,
    )


@torch.inference_mode()
def evaluate_pipeline_method(
    method: str,
    draft_model,
    intermediate_model,
    full_model,
    contexts: List[torch.Tensor],
    gamma: int,
    ni: int,
    temperature: float,
    pipeline_new_tokens: int,
    device: torch.device,
    seed: int,
    ld: int,
    li: int,
    novelty_diagnostics: bool = False,
    novelty_max_windows: int = 500,
    adaptive_models: Optional[Dict[int, Any]] = None,
) -> Tuple[PipelineMethodSummary, List[NoveltyWindowRecord], List[AdaptiveWindowRecord]]:
    gen = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
    # Both methods begin with the same seed. Their random streams naturally
    # diverge after different accept/reject paths consume different draws.
    gen.manual_seed(seed)

    tentative_tokens = 0
    final_tokens = 0
    li_calls = 0
    lf_calls = 0
    li_accepted = 0
    lf_accepted = 0
    li_full = 0
    lf_full = 0
    novelty_rows: List[NoveltyWindowRecord] = []
    adaptive_rows: List[AdaptiveWindowRecord] = []

    _sync(device)
    start = time.perf_counter()

    for c_idx, raw_context in enumerate(contexts):
        context = raw_context.view(1, -1).to(device)
        produced_for_context = 0

        while produced_for_context < pipeline_new_tokens:
            window, calls, accepted, full_blocks = generate_tentative_window(
                method=method,
                draft_model=draft_model,
                intermediate_model=intermediate_model,
                context=context,
                gamma=gamma,
                ni=ni,
                temperature=temperature,
                generator=gen,
            )
            tentative_tokens += int(window.numel())
            li_calls += calls
            li_accepted += accepted
            li_full += full_blocks

            q_li = proposal_probs_with_bonus(
                intermediate_model, context, window, temperature
            )
            p_lf = proposal_probs_with_bonus(full_model, context, window, temperature)
            final_block, accepted_final, full_final = tokenwise_sample_block(
                q_li[:-1], p_lf, window, gen
            )

            # Side-channel three-level diagnostic.  It never changes the real
            # decoder state or consumes random numbers.
            if (
                novelty_diagnostics
                and (novelty_max_windows == 0 or len(novelty_rows) < novelty_max_windows)
            ):
                novelty, p_d = analyze_three_level_window(
                    method=method,
                    draft_model=draft_model,
                    context=context,
                    window=window,
                    p_li_with_bonus=q_li,
                    p_lf_with_bonus=p_lf,
                    accepted_final=accepted_final,
                    full_final=full_final,
                    temperature=temperature,
                )
                novelty_rows.append(novelty)
                if adaptive_models:
                    adaptive_rows.append(
                        analyze_adaptive_depth_window(
                            method=method,
                            ld=ld,
                            fixed_li=li,
                            candidate_models=adaptive_models,
                            context=context,
                            window=window,
                            p_d=p_d,
                            p_li_with_bonus=q_li,
                            p_lf_with_bonus=p_lf,
                            temperature=temperature,
                        )
                    )
                del p_d

            context = torch.cat([context, final_block.view(1, -1)], dim=1)
            block_len = int(final_block.numel())
            final_tokens += block_len
            produced_for_context += block_len
            lf_calls += 1
            lf_accepted += accepted_final
            lf_full += int(full_final)

            del window, q_li, p_lf, final_block

        if (c_idx + 1) % max(1, len(contexts) // 10) == 0:
            print(
                f"  {method}: {c_idx + 1}/{len(contexts)} contexts; "
                f"Li calls={li_calls}, Lf calls={lf_calls}"
            )

    _sync(device)
    elapsed = time.perf_counter() - start
    summary = PipelineMethodSummary(
        method=method,
        tentative_tokens=tentative_tokens,
        final_tokens=final_tokens,
        li_calls=li_calls,
        lf_calls=lf_calls,
        li_accepted_draft_tokens=li_accepted,
        lf_accepted_tentative_tokens=lf_accepted,
        li_full_blocks=li_full,
        lf_full_windows=lf_full,
        elapsed_seconds=elapsed,
    )
    return summary, novelty_rows, adaptive_rows


@torch.inference_mode()
def evaluate_triple(
    base,
    contexts: List[torch.Tensor],
    ld: int,
    li: int,
    lf: int,
    gamma: int,
    temperature: float,
    ni: int,
    pipeline_new_tokens: int,
    device: torch.device,
    seed: int,
    novelty_diagnostics: bool = False,
    novelty_max_windows: int = 500,
    adaptive_layers: Optional[List[int]] = None,
) -> TripleSummary:
    draft_model = make_early_exit_model(base, ld).to(device).eval()
    intermediate_model = make_early_exit_model(base, li).to(device).eval()
    full_model = base if lf == len(base.model.layers) else make_early_exit_model(base, lf)
    full_model = full_model.to(device).eval()

    adaptive_models: Dict[int, Any] = {}
    created_adaptive_models: List[Any] = []
    if novelty_diagnostics:
        requested = sorted(set(adaptive_layers or [li]))
        requested = [d for d in requested if ld < d < lf]
        if li not in requested:
            requested.append(li)
            requested.sort()
        for depth in requested:
            if depth == li:
                adaptive_models[depth] = intermediate_model
            else:
                model = make_early_exit_model(base, depth).to(device).eval()
                adaptive_models[depth] = model
                created_adaptive_models.append(model)

    methods: Dict[str, PipelineMethodSummary] = {}
    novelty_records: Dict[str, List[NoveltyWindowRecord]] = {}
    adaptive_records: Dict[str, List[AdaptiveWindowRecord]] = {}
    for method in ("tokenwise", "hsd"):
        print(f"  Running pipeline method: {method}")
        method_summary, novelty_rows, adaptive_rows = evaluate_pipeline_method(
            method=method,
            draft_model=draft_model,
            intermediate_model=intermediate_model,
            full_model=full_model,
            contexts=contexts,
            gamma=gamma,
            ni=ni,
            temperature=temperature,
            pipeline_new_tokens=pipeline_new_tokens,
            device=device,
            seed=seed + 1000 * ld + 100 * li + lf,
            ld=ld,
            li=li,
            novelty_diagnostics=novelty_diagnostics,
            novelty_max_windows=novelty_max_windows,
            adaptive_models=adaptive_models,
        )
        methods[method] = method_summary
        novelty_records[method] = novelty_rows
        adaptive_records[method] = adaptive_rows

    summary = TripleSummary(
        ld=ld,
        li=li,
        lf=lf,
        gamma=gamma,
        temperature=temperature,
        ni=ni,
        tokenwise=methods["tokenwise"],
        hsd=methods["hsd"],
        novelty_records=novelty_records,
        adaptive_records=adaptive_records,
    )

    for model in created_adaptive_models:
        del model
    del draft_model, intermediate_model
    if full_model is not base:
        del full_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def print_triple_tables(rows: List[TripleSummary]) -> None:
    print("\n=== Full Ld -> Li -> Lf pipeline results ===")
    header = (
        "triple         g     T   Ni  method      Li BE  Final BE  Li full  Lf full  "
        "Li calls/100  Lf calls/100  ref tok/s"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        for result in (row.tokenwise, row.hsd):
            print(
                f"L{row.ld}->L{row.li}->L{row.lf:<3} "
                f"{row.gamma:3d}  {row.temperature:4.2f} {row.ni:3d}  "
                f"{result.method:<10} "
                f"{result.li_be:6.3f}  {result.final_be:8.3f}  "
                f"{100*result.li_full_rate:6.2f}%  {100*result.lf_full_rate:6.2f}%  "
                f"{result.li_calls_per_100:12.2f}  "
                f"{result.lf_calls_per_100:12.2f}  "
                f"{result.tokens_per_second:9.2f}"
            )

    print("\n=== HSD survival summary ===")
    header2 = (
        "triple         g     T   Ni  Li BE gain  Final BE gain  "
        "BE-delta retention"
    )
    print(header2)
    print("-" * len(header2))
    for row in rows:
        survival = row.extra_token_survival
        survival_text = "n/a" if math.isnan(survival) else f"{100*survival:+.2f}%"
        print(
            f"L{row.ld}->L{row.li}->L{row.lf:<3} "
            f"{row.gamma:3d}  {row.temperature:4.2f} {row.ni:3d}  "
            f"{row.li_gain_pct:+10.2f}%  {row.final_gain_pct:+13.2f}%  "
            f"{survival_text:>20}"
        )

    print(
        "\nTiming warning: ref tok/s measures this unoptimized reference implementation. "
        "It does not reuse hidden states/KV caches across Ld, Li, and Lf, so do not "
        "report it as HiSpec end-to-end throughput."
    )
    print(
        "BE-delta retention is an aggregate comparison, not a token-identity "
        "survival trace, because the two exact samplers follow different random paths."
    )
    if any(any(v for v in row.novelty_records.values()) for row in rows):
        print(
            "Novelty-diagnostic timing warning: --novelty-diagnostics adds extra teacher-forced "
            "Ld/intermediate-depth passes. Li/Lf call counters still describe the baseline decoder, "
            "but ref tok/s includes diagnostic overhead and must not be compared to a non-diagnostic run."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="facebook/layerskip-llama3.2-1B",
        help="Early-exit-capable Hugging Face checkpoint.",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help="Ld:Li layer pairs for isolated screening.",
    )
    parser.add_argument(
        "--triples",
        nargs="+",
        default=None,
        help="Ld:Li:Lf triples for the periodic full-verification pipeline.",
    )
    parser.add_argument("--contexts", type=int, default=64)
    parser.add_argument("--context-tokens", type=int, default=64)
    parser.add_argument("--drafts-per-context", type=int, default=1)
    parser.add_argument("--gamma", type=int, default=6)
    parser.add_argument(
        "--gammas",
        type=int,
        nargs="+",
        default=None,
        help="Optional gamma sweep; overrides --gamma.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=None,
        help="Optional temperature sweep; overrides --temperature.",
    )
    parser.add_argument(
        "--corpus",
        choices=sorted(CORPUS_PRESETS),
        default="wikitext2",
        help="Built-in corpus preset.",
    )
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default=None)
    parser.add_argument("--text-field", default=None)
    parser.add_argument(
        "--ni",
        type=int,
        default=16,
        help="Tentative L_i tokens accumulated before each L_f verification.",
    )
    parser.add_argument(
        "--pipeline-new-tokens",
        type=int,
        default=64,
        help="Approximate final tokens generated per context in triple mode.",
    )
    parser.add_argument(
        "--novelty-diagnostics",
        action="store_true",
        help=(
            "Run three-level Ld/Li/Lf diagnostics: cross-level probability trajectories, "
            "HSD->final-survival prediction, heuristic joint-HSD scores, and adaptive-depth study."
        ),
    )
    parser.add_argument(
        "--novelty-max-windows",
        type=int,
        default=500,
        help="Maximum diagnosed Lf windows per method; 0 means all windows.",
    )
    parser.add_argument(
        "--adaptive-layers",
        type=int,
        nargs="+",
        default=[6, 8, 10, 12],
        help=(
            "Candidate early-exit verifier depths used by the adaptive-depth diagnostic. "
            "Values outside (Ld,Lf) are ignored; the configured Li is always included."
        ),
    )
    parser.add_argument(
        "--adaptive-thresholds",
        type=float,
        nargs="+",
        default=[0.60, 0.75, 0.90],
        help="Inter-level mean-HSD thresholds for simulated adaptive-depth stopping.",
    )
    parser.add_argument(
        "--adaptive-quality-tolerance",
        type=float,
        default=0.25,
        help=(
            "Oracle adaptive-depth report: allowed loss in expected final accepted tokens "
            "relative to fixed Li."
        ),
    )
    parser.add_argument(
        "--diagnostic-csv-prefix",
        default=None,
        help=(
            "Optional prefix for per-window novelty CSV files, e.g. results/novelty. "
            "One file is written per triple/method."
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    seed_everything(args.seed)
    if args.contexts <= 0 or args.context_tokens <= 0:
        raise ValueError("--contexts and --context-tokens must be positive")
    if args.drafts_per_context <= 0:
        raise ValueError("--drafts-per-context must be positive")
    if args.ni <= 0 or args.pipeline_new_tokens <= 0:
        raise ValueError("--ni and --pipeline-new-tokens must be positive")
    if args.novelty_max_windows < 0:
        raise ValueError("--novelty-max-windows must be >= 0")
    if any(not (0.0 <= t <= 1.0) for t in args.adaptive_thresholds):
        raise ValueError("--adaptive-thresholds must be in [0,1]")
    if args.adaptive_quality_tolerance < 0:
        raise ValueError("--adaptive-quality-tolerance must be >= 0")

    pair_values = args.pairs
    if pair_values is None and not args.triples:
        pair_values = ["2:4", "3:6", "4:8"]
    pairs = parse_pairs(pair_values or [])
    triples = parse_triples(args.triples)
    gammas = args.gammas or [args.gamma]
    temperatures = args.temperatures or [args.temperature]
    if any(g <= 0 for g in gammas):
        raise ValueError("All gamma values must be positive")
    if any(t <= 0 for t in temperatures):
        raise ValueError("All temperatures must be positive")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    print(f"Device: {device}; dtype: {dtype}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Loading tokenizer/model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    base.eval()

    total_layers = len(base.model.layers)
    print(f"Model layers: {total_layers}")
    for ld, li in pairs:
        if li > total_layers:
            raise ValueError(f"Pair {ld}:{li} exceeds model depth {total_layers}")
    for ld, li, lf in triples:
        if lf > total_layers:
            raise ValueError(
                f"Triple {ld}:{li}:{lf} exceeds model depth {total_layers}"
            )

    print("Building held-out contexts...")
    contexts = build_contexts(
        tokenizer,
        n_contexts=args.contexts,
        context_tokens=args.context_tokens,
        seed=args.seed,
        corpus=args.corpus,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        text_field=args.text_field,
    )

    if pairs:
        summaries: List[PairSummary] = []
        for ld, li in pairs:
            for gamma in gammas:
                for temperature in temperatures:
                    print(
                        f"\nTesting L{ld} -> L{li}; "
                        f"gamma={gamma}, T={temperature}"
                    )
                    summary = evaluate_pair(
                        base=base,
                        contexts=contexts,
                        ld=ld,
                        li=li,
                        gamma=gamma,
                        temperature=temperature,
                        drafts_per_context=args.drafts_per_context,
                        device=device,
                        seed=args.seed,
                    )
                    summaries.append(summary)

        print_table(summaries)
        screening_recommendation(summaries)

    if triples:
        triple_summaries: List[TripleSummary] = []
        for ld, li, lf in triples:
            for gamma in gammas:
                for temperature in temperatures:
                    print(
                        f"\nTesting L{ld} -> L{li} -> L{lf}; gamma={gamma}, "
                        f"T={temperature}, Ni={args.ni}"
                    )
                    summary = evaluate_triple(
                        base=base,
                        contexts=contexts,
                        ld=ld,
                        li=li,
                        lf=lf,
                        gamma=gamma,
                        temperature=temperature,
                        ni=args.ni,
                        pipeline_new_tokens=args.pipeline_new_tokens,
                        device=device,
                        seed=args.seed,
                        novelty_diagnostics=args.novelty_diagnostics,
                        novelty_max_windows=args.novelty_max_windows,
                        adaptive_layers=args.adaptive_layers,
                    )
                    triple_summaries.append(summary)

                    if args.novelty_diagnostics:
                        triple_label = f"L{ld}->L{li}->L{lf}, g={gamma}, T={temperature}, Ni={args.ni}"
                        print_novelty_diagnostics(triple_label, summary.novelty_records)
                        print_adaptive_depth_diagnostics(
                            summary.adaptive_records,
                            thresholds=args.adaptive_thresholds,
                            quality_tolerance=args.adaptive_quality_tolerance,
                        )
                        if args.diagnostic_csv_prefix:
                            prefix_path = Path(args.diagnostic_csv_prefix)
                            prefix_path.parent.mkdir(parents=True, exist_ok=True)
                            for method, diag_rows in summary.novelty_records.items():
                                csv_path = (
                                    f"{args.diagnostic_csv_prefix}_L{ld}-L{li}-L{lf}"
                                    f"_g{gamma}_T{temperature}_{method}.csv"
                                )
                                _write_novelty_csv(csv_path, diag_rows)
                                print(f"Wrote diagnostic CSV: {csv_path}")
        print_triple_tables(triple_summaries)


if __name__ == "__main__":
    main()
