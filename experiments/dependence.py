#!/usr/bin/env python3
"""Dependent acceptance in 3-level hierarchies (CPU, uses collect.py output).

Mohri et al. (arXiv 2510.19705) derive hierarchy latency assuming acceptance is
independent across levels and positions. Section B of analyze.py shows it is not:
a token L_i took from L_d's draft survives L_f more often than a token L_i resampled.

This script simulates the lossless pipeline L_d -> L_i -> L_f (tokenwise speculative
sampling at both stages) under three acceptance models:

  iid : constant stage-1 acceptance and constant L_f acceptance (Mohri's assumption)
  pos : per-position, per-context rates; L_f acceptance ignores the stage-1 outcome
  dep : per-position rates AND L_f acceptance conditioned on the stage-1 event
        (accepted draft / residual correction / bonus token), all computed exactly
        over the vocabulary in collect.py

and three window policies (all lossless: stopping only depends on stage-1 randomness):

  fixed  : buffer Ni tentative tokens, then call L_f (Mohri / HiSpec buffer)
  flush  : also call L_f as soon as stage 1 emits a residual correction (HiSpec's
           flush-on-mismatch)
  thresh : call L_f once the predicted survival of the buffered window, using only
           stage-1 events (pooled E[alpha|event]), drops below theta

Questions answered:
  1. How wrong is the iid latency model, and does it pick the wrong hierarchy?
  2. How much of the gap is context heterogeneity (iid->pos) vs cross-level
     dependence (pos->dep)?
  3. How much does a dependence-aware policy gain over a fixed buffer, and does the
     best 3-level pipeline now beat the best 2-level one?

Approximation (same as section H): acceptance probabilities come from per-position
expectations along the collected window prefix, not from a re-sampled prefix.

Usage:
  python experiments/dependence.py --run runs/wt2_v2
  python experiments/dependence.py --run runs/wt2_8b --samples 8
"""

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from analyze import Report, Run  # same directory

MODELS = ("iid", "pos", "dep")


def stage_arrays(r: Run, d: int, i: int, model: str) -> Dict[str, np.ndarray]:
    """Per-position probabilities for the pair (d, i) with final verifier L."""
    L = r.L
    acc = r.trip(i, d, L, "acc_mass").astype(np.float64)
    res = r.trip(i, d, L, "res_mass").astype(np.float64)
    a_bon = np.clip(1.0 - r.tv(i, i, L).astype(np.float64), 0.0, 1.0)
    a_acc = np.where(acc > 1e-9, r.trip(i, d, L, "acc_alpha_sum") / np.maximum(acc, 1e-12), a_bon)
    a_res = np.where(res > 1e-9, r.trip(i, d, L, "res_alpha_sum") / np.maximum(res, 1e-12), a_bon)
    P = np.clip(acc, 0.0, 1.0)
    a_acc, a_res = np.clip(a_acc, 0, 1), np.clip(a_res, 0, 1)
    # stage-1-only predictor of L_f survival (pooled scalars)
    pred = (
        float(r.trip(i, d, L, "acc_alpha_sum").sum() / max(acc.sum(), 1e-12)),
        float(r.trip(i, d, L, "res_alpha_sum").sum() / max(res.sum(), 1e-12)),
        float(a_bon.mean()),
    )
    if model == "pos":
        a_acc = a_res = a_bon
    elif model == "iid":
        P = np.full_like(P, P.mean())
        a_acc = a_res = a_bon = np.full_like(a_bon, a_bon.mean())
    return {"P": P, "a_acc": a_acc, "a_res": a_res, "a_bon": a_bon, "pred": pred}


def sim_three(arr, gamma: int, cap: int, policy: str, theta: float,
              cost: Tuple[float, float, float], samples: int, seed: int) -> float:
    """Tokens per unit cost for L_d -> L_i -> L_f. cost = (draft/token, verify L_i, verify L_f).

    Decisions are taken only at stage-1 round boundaries and every token of a paid round
    is kept (an L_f pass costs the same for any window length, so truncating is dominated).
      fixed : stop once the window holds >= cap tokens
      flush : also stop after a round that ended in a residual correction
      thresh: also stop once the stage-1-predicted survival of the window < theta
    """
    P, a_acc, a_res, a_bon, pred = arr["P"], arr["a_acc"], arr["a_res"], arr["a_bon"], arr["pred"]
    W, N = P.shape
    rng = np.random.default_rng(seed)
    w = np.repeat(np.arange(W), samples)
    M = w.size
    p0 = np.zeros(M, dtype=np.int64)
    out_tot = np.zeros(M)
    cost_tot = np.zeros(M)
    pmax = N - 1 - cap - gamma          # window can reach cap-1+gamma+1 tokens
    c_d, c_vi, c_vf = cost
    BIG = 10 ** 9

    while True:
        idx = np.nonzero(p0 <= pmax)[0]
        if idx.size == 0:
            break
        m = idx.size
        ww, base = w[idx], p0[idx]
        ntent = np.zeros(m, dtype=np.int64)
        first_rej = np.full(m, BIG, dtype=np.int64)
        surv = np.ones(m)
        stop = np.zeros(m, dtype=bool)
        ctot = np.zeros(m)
        while not stop.all():
            live = ~stop
            ctot[live] += gamma * c_d + c_vi          # one stage-1 round
            inround = live.copy()
            rejected = np.zeros(m, dtype=bool)
            for j in range(gamma + 1):                  # gamma drafts, then a bonus if all accepted
                if not inround.any():
                    break
                pos = base + ntent
                if j < gamma:
                    accd = rng.random(m) < P[ww, pos]
                    a = np.where(accd, a_acc[ww, pos], a_res[ww, pos])
                    pr = np.where(accd, pred[0], pred[1])
                else:
                    accd = np.ones(m, dtype=bool)
                    a = a_bon[ww, pos]
                    pr = np.full(m, pred[2])
                lf_rej = inround & (rng.random(m) > a) & (first_rej == BIG)
                first_rej = np.where(lf_rej, ntent, first_rej)
                surv = np.where(inround, surv * pr, surv)
                ntent = ntent + inround
                if j < gamma:
                    ended = inround & ~accd             # residual correction ends the round
                    rejected |= ended
                    inround &= ~ended
                else:
                    inround[:] = False
            new_stop = live & (ntent >= cap)
            if policy == "flush":
                new_stop |= live & rejected
            elif policy == "thresh":
                new_stop |= live & (surv < theta)
            stop |= new_stop
        ctot += c_vf
        out = np.minimum(first_rej, ntent) + 1
        out_tot[idx] += out
        cost_tot[idx] += ctot
        p0[idx] += out
    return out_tot.sum() / max(cost_tot.sum(), 1e-12)


def sim_two(A: np.ndarray, gamma: int, c_d: float, c_v: float, samples: int, seed: int) -> float:
    """Tokens per unit cost for L_s -> L_f with per-position acceptance A[w, pos]."""
    W, N = A.shape
    rng = np.random.default_rng(seed)
    w = np.repeat(np.arange(W), samples)
    p0 = np.zeros(w.size, dtype=np.int64)
    out_tot, cost_tot = 0.0, 0.0
    offs = np.arange(gamma)
    while True:
        idx = np.nonzero(p0 <= N - gamma)[0]
        if idx.size == 0:
            break
        ok = rng.random((idx.size, gamma)) < A[w[idx, None], p0[idx, None] + offs]
        k = np.where(ok.all(1), gamma, np.argmin(ok, axis=1))
        out = k + 1
        out_tot += out.sum()
        cost_tot += idx.size * (gamma * c_d + c_v)
        p0[idx] += out
    return out_tot / max(cost_tot, 1e-12)


def run_temperature(r: Run, rep: Report, args, rows_out: list):
    H, L = r.H, r.L
    c = lambda k: k + H  # noqa: E731
    AR = c(L)
    gammas = [g for g in args.gammas]
    caps = [n for n in args.caps]
    pairs = [(d, i) for i in r.sources for d in r.sources if d < i]
    if args.min_i:
        pairs = [(d, i) for d, i in pairs if i >= args.min_i]

    for mode in ("noreuse", "reuse"):
        # ---- 2-level baselines (pos and iid) ----
        two = []
        for s in r.sources:
            A_pos = np.clip(1.0 - r.tv(s, s, L).astype(np.float64), 0, 1)
            A_iid = np.full_like(A_pos, A_pos.mean())
            c_v = c(L) if mode == "noreuse" else (L - s) + H
            for g in gammas:
                seed = hash((s, g)) % (2 ** 31)
                for name, A in (("pos", A_pos), ("iid", A_iid)):
                    sp = AR * sim_two(A, g, c(s), c_v, args.samples, seed)
                    two.append({"T": r.T, "mode": mode, "levels": 2, "config": f"L{s}->L{L}",
                                "model": name, "policy": "fixed", "gamma": g, "cap": g,
                                "theta": None, "speedup": sp})
        rows_out.extend(two)

        # ---- 3-level ----
        three = []
        for d, i in pairs:
            arrs = {m: stage_arrays(r, d, i, m) for m in MODELS}
            cost = (c(d), c(i) if mode == "noreuse" else (i - d) + H,
                    c(L) if mode == "noreuse" else (L - i) + H)
            for g, cap in itertools.product(gammas, caps):
                seed = hash((d, i, g, cap)) % (2 ** 31)   # common random numbers across policies/models
                pols = [("fixed", None), ("flush", None)] + [("thresh", t) for t in args.thetas]
                for m in MODELS:
                    for pol, th in pols:
                        sp = AR * sim_three(arrs[m], g, cap, pol, th if th is not None else 0.0,
                                            cost, args.samples, seed)
                        three.append({"T": r.T, "mode": mode, "levels": 3,
                                      "config": f"L{d}->L{i}->L{L}", "model": m, "policy": pol,
                                      "gamma": g, "cap": cap, "theta": th, "speedup": sp})
        rows_out.extend(three)
        report_mode(r, rep, mode, pd.DataFrame(two), pd.DataFrame(three))


def best_by(df, keys):
    return df.loc[df.groupby(keys)["speedup"].idxmax()]


def report_mode(r: Run, rep: Report, mode: str, two: pd.DataFrame, three: pd.DataFrame):
    rep(f"\n--- cost model: {mode} ---")
    b2 = {m: two[two.model == m].speedup.max() for m in ("iid", "pos")}
    rep(f"best 2-level: pos {b2['pos']:.3f}x | iid {b2['iid']:.3f}x")

    # Q1/Q2: fixed-buffer policy under each acceptance model, best (gamma, cap) per config
    fx = best_by(three[three.policy == "fixed"], ["config", "model"])
    piv = fx.pivot(index="config", columns="model", values="speedup")
    piv["iid_err_%"] = 100 * (piv["iid"] / piv["dep"] - 1)
    piv["hetero_%"] = 100 * (piv["pos"] / piv["iid"] - 1)
    piv["depend_%"] = 100 * (piv["dep"] / piv["pos"] - 1)
    piv = piv.sort_values("dep", ascending=False)
    rep("[1] fixed buffer, best (gamma, Ni) per config, speedup under each acceptance model")
    rep("    iid_err = iid prediction vs dep; hetero = pos vs iid; depend = dep vs pos")
    rep.table(piv.reset_index().head(10), "{:.3f}")
    rep(f"    mean over all {len(piv)} configs: iid_err {piv['iid_err_%'].mean():+.1f}%, "
        f"hetero {piv['hetero_%'].mean():+.1f}%, depend {piv['depend_%'].mean():+.1f}%")

    # selection regret: pick the best fixed config under each model, score it under dep
    dep_fx = three[(three.policy == "fixed") & (three.model == "dep")]
    true_best = dep_fx.speedup.max()
    for m in ("iid", "pos"):
        sub = three[(three.policy == "fixed") & (three.model == m)]
        pick = sub.loc[sub.speedup.idxmax()]
        true = dep_fx[(dep_fx.config == pick.config) & (dep_fx.gamma == pick.gamma)
                      & (dep_fx.cap == pick.cap)].speedup.iloc[0]
        rep(f"    {m} model picks {pick.config} g={pick.gamma} Ni={pick.cap}: predicted "
            f"{pick.speedup:.3f}x, actual {true:.3f}x; best actual fixed {true_best:.3f}x "
            f"(regret {100 * (1 - true / true_best):.1f}%)")

    # Q3: policies. A policy's gain over the fixed buffer can come from round alignment
    # (present under any model) or from cross-level dependence (present only under dep).
    # The dependence-attributable part is gain(dep) - gain(pos).
    gains = {}
    for m in MODELS:
        sub = three[three.model == m]
        pv = best_by(sub, ["config", "policy"]).pivot(index="config", columns="policy", values="speedup")
        gains[m] = pd.DataFrame({
            "flush": 100 * (pv["flush"] / pv["fixed"] - 1),
            "thresh": 100 * (pv["thresh"] / pv["fixed"] - 1),
        })
        if m == "dep":
            dep_pv = pv
    tab = dep_pv.copy()
    for pol in ("flush", "thresh"):
        tab[f"{pol}_%dep"] = gains["dep"][pol]
        tab[f"{pol}_%pos"] = gains["pos"][pol]
        tab[f"{pol}_dep_attr"] = gains["dep"][pol] - gains["pos"][pol]
    tab = tab.sort_values("fixed", ascending=False)
    rep("[2] window policy gain over the fixed buffer (best params per config).")
    rep("    %dep / %pos = gain under the dep / pos model; dep_attr = the part due to cross-level "
        "dependence (%dep - %pos)")
    rep.table(tab.reset_index().head(10), "{:.2f}")
    for pol in ("flush", "thresh"):
        rep(f"    mean over configs, {pol}: dep {gains['dep'][pol].mean():+.2f}%, "
            f"pos {gains['pos'][pol].mean():+.2f}%, iid {gains['iid'][pol].mean():+.2f}%, "
            f"dependence-attributable {(gains['dep'][pol] - gains['pos'][pol]).mean():+.2f}%")

    dep = three[three.model == "dep"]
    best3 = dep.loc[dep.speedup.idxmax()]
    rep(f"[3] best 3-level (dep): {best3.config} {best3.policy} g={best3.gamma} Ni={best3.cap} "
        f"theta={best3.theta}: {best3.speedup:.3f}x vs best 2-level {b2['pos']:.3f}x "
        f"({100 * (best3.speedup / b2['pos'] - 1):+.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--temperatures", type=float, nargs="*", default=None,
                    help="sampling temperatures only (greedy runs lack stage-1 event stats)")
    ap.add_argument("--samples", type=int, default=8, help="Monte Carlo runs per window")
    ap.add_argument("--gammas", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--caps", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--thetas", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ap.add_argument("--min-i", type=int, default=0, help="only intermediate depths >= this (speed)")
    args = ap.parse_args()

    run_dir = Path(args.run)
    meta = json.loads((run_dir / "meta.json").read_text())
    temps = [t for t in (args.temperatures or meta["temperatures"]) if t > 0]

    rep = Report()
    rep(f"run={run_dir} model={meta['model']} windows={meta['contexts']} "
        f"samples/window={args.samples}")
    rows = []
    for T in temps:
        r = Run(run_dir, T)
        if not r.data:
            continue
        rep("\n" + "=" * 100 + f"\nTemperature {T}\n" + "=" * 100)
        run_temperature(r, rep, args, rows)

    out = run_dir / "analysis"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "dependence.csv", index=False)
    (out / "dependence.txt").write_text("\n".join(rep.lines))
    print(f"\nwrote {out}/dependence.txt and dependence.csv")


if __name__ == "__main__":
    main()
