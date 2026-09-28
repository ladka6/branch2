#!/usr/bin/env python3
"""Collect per-token multi-depth statistics (GPU step).

For each source depth s and temperature T:
  1. sample an N-token window from the L_s early exit, for every context
  2. one full-model pass over context+window, read p_d at every depth d
  3. store compact per-token stats (no full distributions)

Everything downstream (acceptance curves, stage transfer, null baselines,
cost model, predictors) is computed on CPU by analyze.py from these files.

Example (LayerSkip 1B, ~15 min on one A100):
  python experiments/collect.py --out runs/wt2 --corpus wikitext2 \
      --sources 2 4 6 8 10 12 14 --depths 2 4 6 8 10 12 14 16 \
      --temperatures 0 0.6 1.0 --contexts 512 --window 32
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hsx.data import build_contexts, random_contexts  # noqa: E402
from hsx.lens import (  # noqa: E402
    PROPOSAL_FIELDS, PROPOSALS, TRIPLE_FIELDS, depth_logits, pair_list, triple_list, window_stats,
)
from hsx.models import (  # noqa: E402
    cost_units, load_model, make_early_exit_model, pick_device_dtype, tiny_random_model,
)
from hsx.sampling import sample_window  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/layerskip-llama3.2-1B")
    ap.add_argument("--tiny", action="store_true", help="random 8-layer model, no downloads (smoke test)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", default="wikitext2")
    ap.add_argument("--contexts", type=int, default=512)
    ap.add_argument("--context-tokens", type=int, default=128)
    ap.add_argument("--window", type=int, default=32, help="tokens sampled per window (max Ni studied)")
    ap.add_argument("--sources", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12, 14])
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12, 14, 16])
    ap.add_argument("--temperatures", type=float, nargs="+", default=[0.0, 0.6, 1.0], help="0 = greedy")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    device, dtype = pick_device_dtype()
    if args.tiny:
        base = tiny_random_model(device, n_layers=max(args.depths))
        tokenizer = None
        dtype = torch.float32
    else:
        tokenizer, base = load_model(args.model, device, dtype)
    L = len(base.model.layers)

    depths = sorted(set(args.depths) | set(args.sources) | {L})
    if max(depths) > L:
        raise ValueError(f"depth {max(depths)} > model layers {L}")
    sources = sorted(s for s in set(args.sources) if s < L)

    if args.tiny:
        contexts = random_contexts(args.contexts, args.context_tokens, base.config.vocab_size, args.seed)
    else:
        contexts = build_contexts(tokenizer, args.contexts, args.context_tokens, args.seed, args.corpus)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "model": "tiny-random" if args.tiny else args.model,
        "corpus": args.corpus,
        "contexts": args.contexts,
        "context_tokens": args.context_tokens,
        "window": args.window,
        "depths": depths,
        "sources": sources,
        "temperatures": args.temperatures,
        "final_depth": L,
        "pairs": pair_list(depths),
        "triples": {str(s): triple_list(depths, s) for s in sources},
        "triple_fields": TRIPLE_FIELDS,
        "proposals": PROPOSALS,
        "proposal_fields": PROPOSAL_FIELDS,
        "proposal_lower": {str(s): [d for d in depths if d < s] for s in sources},
        "cost": cost_units(base),
        "seed": args.seed,
        "dtype": str(dtype),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"device={device} dtype={dtype} layers={L} head≈{meta['cost']['head_in_layers']:.2f} layers")

    C, N = args.context_tokens, args.window
    for T in args.temperatures:
        for s in sources:
            t0 = time.time()
            drafter = make_early_exit_model(base, s)
            gen = torch.Generator(device=device.type).manual_seed(args.seed * 1000 + s + int(100 * T))
            parts = []
            for b0 in range(0, len(contexts), args.batch):
                ctx = contexts[b0 : b0 + args.batch].to(device)
                win = sample_window(drafter, ctx, N, T, gen)
                full = torch.cat([ctx, win], dim=1)
                logits = depth_logits(base, full[:, :-1], depths, start=C - 1)
                # T=0 means greedy windows; distribution stats are then taken at T=1.
                parts.append(window_stats(logits, win, T if T > 0 else 1.0, source=s))
                del logits, full, win, ctx
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            arrays = {k: torch.cat([p[k] for p in parts]).numpy() for k in parts[0]}
            path = out_dir / f"T{T}_src{s}.npz"
            np.savez_compressed(path, **arrays)
            del drafter
            print(f"T={T} source=L{s}: {len(contexts)} windows x {N} tokens -> {path.name} "
                  f"({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
