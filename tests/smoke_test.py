#!/usr/bin/env python3
"""Offline smoke test on a tiny random Llama. Run: python tests/smoke_test.py"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsx.lens import depth_logits, window_stats  # noqa: E402
from hsx.models import make_early_exit_model, tiny_random_model  # noqa: E402
from hsx.sampling import sample_window  # noqa: E402


def check_lens_matches_early_exit():
    dev = torch.device("cpu")
    base = tiny_random_model(dev, n_layers=8)
    ids = torch.randint(0, base.config.vocab_size, (2, 20))
    depths = [2, 4, 6, 8]
    got = depth_logits(base, ids, depths, start=5)
    for d in depths:
        ref = make_early_exit_model(base, d)(input_ids=ids).logits[:, 5:, :].float()
        err = (got[d] - ref).abs().max().item()
        assert err < 1e-4, f"depth {d}: lens differs from early-exit model by {err}"
    print("ok: hook lens == truncated early-exit model at every depth")


def check_sampling_and_stats():
    dev = torch.device("cpu")
    base = tiny_random_model(dev, n_layers=8)
    ctx = torch.randint(0, base.config.vocab_size, (3, 10))
    drafter = make_early_exit_model(base, 4)
    g = torch.Generator().manual_seed(0)
    win = sample_window(drafter, ctx, 6, 1.0, g)
    assert win.shape == (3, 6)
    full = torch.cat([ctx, win], 1)
    logits = depth_logits(base, full[:, :-1], [2, 4, 6, 8], start=9)
    # the cached sampler and the uncached lens must agree on the drafter's distribution
    ref = drafter(input_ids=full[:, :-1]).logits[:, 9:, :].float()
    assert (logits[4] - ref).abs().max().item() < 1e-4
    st = window_stats(logits, win, 1.0, source=4)
    tr = st["triple"].numpy()  # [B,N,trip,7]
    nulls = tr[..., :4]
    # the four classes cover the mass of p_s (up/down overlap only on exact ties)
    assert np.all(nulls.sum(-1) > 0.999), nulls.sum(-1).min()
    # acc + res alpha sums equal E_{p_s}[alpha] = 1 - TV(p_s, p_b)
    p = {d: torch.softmax(logits[d], -1) for d in logits}
    for k, (a, b) in enumerate([(2, 6), (2, 8)]):
        tv = 0.5 * (p[4] - p[b]).abs().sum(-1).numpy()
        assert np.allclose(tr[..., k, 6] + tr[..., k, 7], 1 - tv, atol=1e-4)
        assert np.allclose(tr[..., k, 4] + tr[..., k, 5], 1.0, atol=1e-4)
    print("ok: sampler/lens agree, null classes and acc/res decomposition are consistent")


def run_pipeline():
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [sys.executable, str(ROOT / "experiments/collect.py"), "--tiny", "--out", tmp,
               "--contexts", "64", "--context-tokens", "12", "--window", "16",
               "--sources", "2", "4", "6", "--depths", "2", "4", "6", "8",
               "--temperatures", "1.0", "--batch", "8"]
        subprocess.run(cmd, check=True)
        subprocess.run([sys.executable, str(ROOT / "experiments/analyze.py"), "--run", tmp], check=True)
    print("ok: collect -> analyze end to end")


if __name__ == "__main__":
    check_lens_matches_early_exit()
    check_sampling_and_stats()
    run_pipeline()
