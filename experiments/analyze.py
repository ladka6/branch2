#!/usr/bin/env python3
"""Analyze collected multi-depth statistics (CPU step).

Sections:
  A. Acceptance matrix: expected accepted tokens for every drafter->verifier depth pair.
  B. Stage transfer: does the stage-1 outcome (draft accepted vs resampled at L_i)
     change the chance that L_f accepts the token? This is the exact version of
     "does L_d->L_i quality predict final survival".
  C. Trajectory classes, observed vs analytic null (checks the overshoot artifact).
  D. Final-stage acceptance vs window size Ni (how big should Ni be).
  E. Memory-bound cost model: best 2-level vs best 3-level speedup.
  F. L_f-free predictors of final acceptance (entropy, margin, lookahead TV).
  G. Position drift of L_i->L_f acceptance inside the window.
  H. Stage-1 target design: which r built from p_d, p_i maximizes final acceptance per cost.

T=0 runs are greedy: acceptance is argmax agreement and B/C are skipped.

Usage:
  python experiments/analyze.py --run runs/wt2
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


class Report:
    def __init__(self):
        self.lines: List[str] = []

    def __call__(self, *parts):
        s = " ".join(str(p) for p in parts)
        print(s)
        self.lines.append(s)

    def table(self, df: pd.DataFrame, floatfmt: str = "{:.3f}"):
        self(df.to_string(index=False, float_format=lambda x: floatfmt.format(x)))


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    label = label.astype(bool)
    n_pos, n_neg = label.sum(), (~label).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def tau_curve_from(acc: np.ndarray):
    """acc: [W, N] per-position acceptance probabilities. Returns E[accepted | n proposed]
    for n = 1..N and its standard error over windows."""
    per_window = np.cumsum(np.cumprod(acc, axis=1), axis=1)
    return per_window.mean(0), per_window.std(0) / np.sqrt(per_window.shape[0])


def expected_rounds_from(acc: np.ndarray, gamma: int, n_max: int) -> np.ndarray:
    """R[n] = expected stage-1 rounds (propose gamma, verify) to accumulate at least n
    tentative tokens. A round yields tau+1 tokens with P(tau >= k) = E[prod_{j<=k} acc_j].
    Renewal recursion, so a window always costs at least one full round."""
    surv = np.cumprod(acc[:, :gamma], axis=1).mean(0)   # P(tau>=k), k=1..gamma
    ge = np.concatenate([[1.0], surv, [0.0]])           # P(tau>=k), k=0..gamma+1
    p_tau = ge[:-1] - ge[1:]                             # P(tau=k), k=0..gamma
    R = np.zeros(n_max + 1)
    for n in range(1, n_max + 1):
        R[n] = 1.0 + sum(p_tau[k] * R[max(0, n - k - 1)] for k in range(gamma + 1))
    return R


def best_speedup(stage1_acc, final_curve, d, i, L, H, mode, n_max, gammas):
    """Best (speedup, gamma, Ni) of a 3-level pipeline under the memory-bound cost model."""
    c = lambda k: k + H  # noqa: E731
    best = (0.0, None, None)
    for g in gammas:
        rounds = expected_rounds_from(stage1_acc, g, n_max)
        stage1 = g * c(d) + (c(i) if mode == "noreuse" else (i - d) + H)
        fin = c(L) if mode == "noreuse" else (L - i) + H
        for ni in range(1, n_max + 1):
            tok = final_curve[ni - 1] + 1
            best = max(best, (c(L) * tok / (rounds[ni] * stage1 + fin), g, ni))
    return best


class Run:
    def __init__(self, run_dir: Path, T: float):
        self.meta = json.loads((run_dir / "meta.json").read_text())
        self.T = T
        self.depths: List[int] = self.meta["depths"]
        self.L: int = self.meta["final_depth"]
        self.sources: List[int] = self.meta["sources"]
        self.di = {d: k for k, d in enumerate(self.depths)}
        self.pi = {tuple(p): k for k, p in enumerate(self.meta["pairs"])}
        self.ti = {s: {tuple(t): k for k, t in enumerate(self.meta["triples"][str(s)])} for s in self.sources}
        self.tf = {f: k for k, f in enumerate(self.meta["triple_fields"])}
        self.data: Dict[int, Dict[str, np.ndarray]] = {}
        for s in self.sources:
            path = run_dir / f"T{T}_src{s}.npz"
            if path.exists():
                with np.load(path) as z:
                    self.data[s] = {k: z[k] for k in z.files}
        self.sources = sorted(self.data)
        self.H = self.meta["cost"]["head_in_layers"]
        self.greedy = T == 0

    # per-token acceptance prob of the drafted token, drafter s -> verifier v.
    # Sampling: min(1, p_v/p_s). Greedy: 1 if the verifier's argmax is the token.
    def alpha(self, s: int, v: int) -> np.ndarray:
        if self.greedy:
            D = self.data[s]
            return (D["argmax"][..., self.di[v]] == D["tokens"]).astype(np.float64)
        lp = self.data[s]["tok_logp"]
        return np.minimum(1.0, np.exp(lp[..., self.di[v]] - lp[..., self.di[s]]))

    # E[accepted tokens | proposal length n] for n = 1..N, plus standard error
    def tau_curve(self, s: int, v: int):
        return tau_curve_from(self.alpha(s, v))

    def tv(self, s: int, a: int, b: int) -> np.ndarray:
        a, b = min(a, b), max(a, b)
        return self.data[s]["tv"][..., self.pi[(a, b)]]

    def expected_rounds(self, d: int, i: int, gamma: int, n_max: int) -> np.ndarray:
        return expected_rounds_from(self.alpha(d, i), gamma, n_max)

    def prop(self, s: int, a: int, cand: str, field: str) -> np.ndarray:
        k_a = self.meta["proposal_lower"][str(s)].index(a)
        k_c = self.meta["proposals"].index(cand)
        k_f = self.meta["proposal_fields"].index(field)
        return self.data[s]["prop"][..., k_a, k_c, k_f].astype(np.float64)

    def trip(self, s: int, a: int, b: int, field: str) -> np.ndarray:
        return self.data[s]["triple"][..., self.ti[s][(a, s, b)], self.tf[field]]


def section_a(r: Run, rep: Report):
    rep("\n[A] Acceptance matrix: drafter L_s (rows) -> verifier L_v. "
        "Cells: mean per-token alpha | E[accepted] at gamma=4 | at gamma=16")
    rows = []
    for s in r.sources:
        row = {"drafter": f"L{s}"}
        for v in r.depths:
            if v <= s:
                row[f"L{v}"] = ""
                continue
            a = r.alpha(s, v).mean()
            curve, _ = r.tau_curve(s, v)
            n16 = min(16, len(curve))
            row[f"L{v}"] = f"{a:.2f}|{curve[3]:.1f}|{curve[n16 - 1]:.1f}" if len(curve) >= 4 else f"{a:.2f}"
        rows.append(row)
    rep(pd.DataFrame(rows).to_string(index=False))


def section_b(r: Run, rep: Report, rows_out: list):
    rep("\n[B] Stage transfer. For window token x ~ p_i at stage 1 (drafted by L_d):")
    rep("    P_acc    = P(L_d draft accepted by L_i)")
    rep("    a|acc    = E[alpha_i->f | draft was accepted]")
    rep("    a|res    = E[alpha_i->f | token came from L_i's residual (L_i overruled L_d)]")
    rep("    a|all    = E[alpha_i->f] = 1 - TV(p_i, p_f)")
    rep("    rho_tok  = Spearman(alpha_d->i(x), alpha_i->f(x)) on sampled tokens (original diagnostic)")
    rep("    If a|res ~ a|acc, stage 1's outcome carries no information about stage 2.")
    rows = []
    for s in r.sources:
        for (a, _, b) in r.meta["triples"][str(s)]:
            acc_mass = r.trip(s, a, b, "acc_mass")
            res_mass = r.trip(s, a, b, "res_mass")
            acc_sum = r.trip(s, a, b, "acc_alpha_sum")
            res_sum = r.trip(s, a, b, "res_alpha_sum")
            a_acc = acc_sum.sum() / acc_mass.sum() if acc_mass.sum() > 1e-6 else float("nan")
            a_res = res_sum.sum() / res_mass.sum() if res_mass.sum() > 1e-6 else float("nan")
            a_all = (1 - r.tv(s, s, b)).mean()
            check = (acc_sum + res_sum).mean()
            lp = r.data[s]["tok_logp"]
            alpha_di = np.minimum(1, np.exp(lp[..., r.di[s]] - lp[..., r.di[a]]))
            rho = spearman(alpha_di.ravel(), r.alpha(s, b).ravel())
            rows.append({"T": r.T, "d": a, "i": s, "f": b, "P_acc": acc_mass.mean(),
                         "a|acc": a_acc, "a|res": a_res, "a|all": a_all,
                         "res/acc": a_res / max(a_acc, 1e-12), "rho_tok": rho,
                         "consistency": abs(check - a_all)})
    df = pd.DataFrame(rows)
    rows_out.extend(rows)
    show = df[df.f == r.L].drop(columns=["T", "consistency"])
    rep.table(show)
    rep(f"    (internal check: max |E[acc+res] - (1-TV)| = {df.consistency.max():.2e})")


def section_c(r: Run, rep: Report, focus: List[int]):
    rep("\n[C] Trajectory classes for x ~ p_i: observed share vs analytic null. "
        "Since window tokens ARE samples from p_i, observed should equal null; "
        "a match means the 'L_i overshoot' share is a sampling artifact, not a finding.")
    rows = []
    for s in r.sources:
        for (a, _, b) in r.meta["triples"][str(s)]:
            if b != r.L or (focus and (a, s) not in focus):
                continue
            lp = r.data[s]["tok_logp"]
            la, ls, lb = lp[..., r.di[a]], lp[..., r.di[s]], lp[..., r.di[b]]
            obs = {
                "up": ((ls >= la) & (lb >= ls)).mean(),
                "down": ((ls <= la) & (lb <= ls)).mean(),
                "overshoot": ((ls > la) & (lb < ls)).mean(),
                "undershoot": ((ls < la) & (lb > ls)).mean(),
            }
            for cls in obs:
                rows.append({"triple": f"L{a}->L{s}->L{b}", "class": cls,
                             "observed": obs[cls], "null": r.trip(s, a, b, f"null_{cls}").mean()})
    rep.table(pd.DataFrame(rows))


def section_d(r: Run, rep: Report):
    rep("\n[D] Final stage: E[tokens accepted by L_f] when L_i proposes a window of Ni tokens "
        "(window ~ p_i, exact for any lossless stage 1)")
    N = r.data[r.sources[0]]["tok_logp"].shape[1]
    ns = [n for n in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32) if n <= N]
    rows = []
    for s in r.sources:
        curve, se = r.tau_curve(s, r.L)
        row = {"L_i": f"L{s}", "alpha": r.alpha(s, r.L).mean()}
        for n in ns:
            row[f"Ni={n}"] = curve[n - 1]
        row["max_se"] = se.max()
        rows.append(row)
    rep.table(pd.DataFrame(rows), "{:.2f}")


def section_e(r: Run, rep: Report, rows_out: list):
    H, L = r.H, r.L
    N = r.data[r.sources[0]]["tok_logp"].shape[1]
    gammas = range(1, min(N, 12) + 1)
    rep(f"\n[E] Memory-bound cost model. Cost of a pass through k layers = k + H, H = head size "
        f"in layer units = {H:.2f}. Autoregressive L{L} costs {L + H:.2f}/token.")
    rep("    'noreuse': every call recomputes from layer 1. 'reuse': self-speculative KV/hidden-state "
        "reuse, a verifier at depth v after depth u only pays (v-u)+H (optimistic).")
    c = lambda k: k + H  # noqa: E731
    curves = {(s, v): r.tau_curve(s, v)[0] for s in r.sources for v in r.depths if v > s}
    rows = []
    for mode in ("noreuse", "reuse"):
        # 2-level: s -> L
        for s in r.sources:
            best = (0, None)
            for g in gammas:
                tok = curves[(s, L)][g - 1] + 1
                cost = g * c(s) + (c(L) if mode == "noreuse" else (L - s) + H)
                best = max(best, (c(L) * tok / cost, g))
            rows.append({"mode": mode, "config": f"L{s}->L{L}", "levels": 2,
                         "speedup": best[0], "gamma": best[1], "Ni": None})
        # 3-level: d -> i -> L
        for i in r.sources:
            for d in r.sources:
                if d >= i:
                    continue
                best = best_speedup(r.alpha(d, i), curves[(i, L)], d, i, L, H, mode, N, gammas)
                rows.append({"mode": mode, "config": f"L{d}->L{i}->L{L}", "levels": 3,
                             "speedup": best[0], "gamma": best[1], "Ni": best[2]})
    df = pd.DataFrame(rows)
    df["T"] = r.T
    rows_out.extend(df.to_dict("records"))
    for mode in ("noreuse", "reuse"):
        sub = df[df["mode"] == mode].sort_values("speedup", ascending=False)
        rep(f"  {mode}: top configs")
        rep.table(sub.drop(columns=["mode", "T"]).head(8), "{:.2f}")
        b2 = sub[sub.levels == 2].speedup.max()
        b3 = sub[sub.levels == 3].speedup.max()
        rep(f"  {mode}: best 2-level {b2:.2f}x, best 3-level {b3:.2f}x, "
            f"3-level gain {100 * (b3 / b2 - 1):+.1f}%")


def section_f(r: Run, rep: Report, rows_out: list):
    rep("\n[F] Predicting alpha_i->f from signals available without running L_f. "
        "AUROC for 'alpha >= 0.5' and Spearman with alpha. oracle_tv uses L_f (ceiling).")
    rows = []
    for s in r.sources:
        k = r.di[s]
        D = r.data[s]
        y = r.alpha(s, r.L).ravel()
        lower = [d for d in r.depths if d < s]
        higher = [d for d in r.depths if s < d < r.L]
        feats = {
            "logp_i(x)": D["tok_logp"][..., k],
            "top1_i": D["top1"][..., k],
            "margin_i": D["margin"][..., k],
            "-entropy_i": -D["entropy"][..., k],
        }
        if lower:
            feats[f"-TV(L{lower[-1]},L{s}) free"] = -r.tv(s, lower[-1], s)
        if higher:
            feats[f"-TV(L{s},L{higher[0]}) +{higher[0] - s}L"] = -r.tv(s, s, higher[0])
        feats["-TV(i,f) oracle"] = -r.tv(s, s, r.L)
        for name, f in feats.items():
            f = f.ravel()
            rows.append({"T": r.T, "L_i": f"L{s}", "feature": name,
                         "AUROC": auroc(f, y >= 0.5), "spearman": spearman(f, y)})
    df = pd.DataFrame(rows)
    rows_out.extend(rows)
    rep.table(df.drop(columns=["T"]))


def section_g(r: Run, rep: Report):
    rep("\n[G] Mean alpha_i->f by window position (drift check)")
    N = r.data[r.sources[0]]["tok_logp"].shape[1]
    cuts = [0, 1, 2, 4, 8, 16, 32, 64]
    cuts = [c for c in cuts if c < N] + [N]
    rows = []
    for s in r.sources:
        a = r.alpha(s, r.L).mean(0)
        row = {"L_i": f"L{s}"}
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            row[f"[{lo},{hi})"] = a[lo:hi].mean()
        rows.append(row)
    rep.table(pd.DataFrame(rows))


def section_h(r: Run, rep: Report, rows_out: list):
    first = r.data[r.sources[0]]
    if "prop" not in first:
        rep("\n[H] (no proposal data in this run; recollect with the current collect.py)")
        return
    names = r.meta["proposals"]
    combos = [c for c in names if c not in ("base",) and not c.startswith("sh")]
    sharps = [c for c in names if c.startswith("sh")]
    L, H, N = r.L, r.H, first["tokens"].shape[1]
    gammas = range(1, min(N, 12) + 1)
    fin_field = "final_agree" if r.greedy else "final_acc"

    rep("\n[H] Stage-1 target design. Stage 1 does lossless speculative sampling from L_d toward a "
        "target r built from p_d and p_i (same cost as HiSpec stage 1); L_f then corrects to p_f.")
    rep("    s1  = 1-TV(p_d, r): stage-1 per-token acceptance.  fin = 1-TV(r, p_f): L_f per-token "
        "acceptance" + (" (greedy: argmax agreement)" if r.greedy else "") + ".")
    rep("    combo = best of min/prod/geo/mix/extrapolation; sharp = best temperature-only control on p_i.")
    if not r.greedy:
        rep("    speedup columns use per-position expected acceptances (approximation; see calibration).")

    rows, detail = [], []
    for s in r.sources:
        for a in r.meta["proposal_lower"][str(s)]:
            s1 = {c: r.prop(s, a, c, "stage1_acc") for c in names}
            fi = {c: r.prop(s, a, c, fin_field) for c in names}
            m1 = {c: s1[c].mean() for c in names}
            mf = {c: fi[c].mean() for c in names}
            sp = {}
            if not r.greedy:
                for mode in ("noreuse", "reuse"):
                    for c in names:
                        sp[(mode, c)] = best_speedup(s1[c], tau_curve_from(fi[c])[0], a, s, L, H,
                                                     mode, N, gammas)
            bc = max(combos, key=mf.get)
            bs = max(sharps, key=mf.get)
            row = {"d": a, "i": s, "base_s1": m1["base"], "base_fin": mf["base"],
                   "combo": bc, "combo_s1": m1[bc], "combo_fin": mf[bc],
                   "sharp": bs, "sharp_fin": mf[bs]}
            if not r.greedy:
                for mode, tag in (("noreuse", "nr"), ("reuse", "re")):
                    bsp = max(names, key=lambda c: sp[(mode, c)][0])
                    row[f"base_{tag}"] = sp[(mode, "base")][0]
                    row[f"best_{tag}"] = sp[(mode, bsp)][0]
                    row[f"by_{tag}"] = bsp
            rows.append(row)
            for c in names:
                d = {"T": r.T, "d": a, "i": s, "cand": c, "s1": m1[c], "fin": mf[c]}
                if not r.greedy:
                    for mode in ("noreuse", "reuse"):
                        v, g, ni = sp[(mode, c)]
                        d.update({f"speedup_{mode}": v, f"gamma_{mode}": g, f"Ni_{mode}": ni})
                detail.append(d)
    rows_out.extend(detail)
    rep.table(pd.DataFrame(rows))

    focus = [(2, 4), (4, 8), (6, 12)]
    ddf = pd.DataFrame(detail)
    for a, s in focus:
        sub = ddf[(ddf.d == a) & (ddf.i == s)]
        if len(sub):
            rep(f"  detail L{a}->L{s}->L{L}:")
            rep.table(sub.drop(columns=["T", "d", "i"]))

    if not r.greedy:
        rep("  calibration (base): approximate E[accepted by L_f] at Ni=4 from 1-TV vs exact from sampled windows")
        cal = []
        for s in r.sources:
            a = r.meta["proposal_lower"][str(s)]
            if not a:
                continue
            approx = tau_curve_from(r.prop(s, a[0], "base", "final_acc"))[0]
            exact = r.tau_curve(s, L)[0]
            n = min(4, N)
            cal.append({"L_i": f"L{s}", "approx": approx[n - 1], "exact": exact[n - 1]})
        rep.table(pd.DataFrame(cal))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--temperatures", type=float, nargs="*", default=None)
    args = ap.parse_args()
    run_dir = Path(args.run)
    meta = json.loads((run_dir / "meta.json").read_text())
    temps = args.temperatures or meta["temperatures"]
    out = run_dir / "analysis"
    out.mkdir(exist_ok=True)

    rep = Report()
    rep(f"run={run_dir} model={meta['model']} corpus={meta['corpus']} "
        f"windows={meta['contexts']} window_len={meta['window']}")
    trans, cost, pred, props = [], [], [], []
    for T in temps:
        r = Run(run_dir, T)
        if not r.data:
            rep(f"(no data for T={T})")
            continue
        label = "greedy (acceptance = argmax agreement)" if r.greedy else f"Temperature {T}"
        rep("\n" + "=" * 100 + f"\n{label}\n" + "=" * 100)
        section_a(r, rep)
        if not r.greedy:  # B and C are about sampling laws
            section_b(r, rep, trans)
            section_c(r, rep, focus=[])
        section_d(r, rep)
        section_e(r, rep, cost)
        section_f(r, rep, pred)
        section_g(r, rep)
        section_h(r, rep, props)

    pd.DataFrame(trans).to_csv(out / "stage_transfer.csv", index=False)
    pd.DataFrame(cost).to_csv(out / "cost_model.csv", index=False)
    pd.DataFrame(pred).to_csv(out / "predictors.csv", index=False)
    pd.DataFrame(props).to_csv(out / "proposals.csv", index=False)
    (out / "report.txt").write_text("\n".join(rep.lines))
    print(f"\nwrote {out}/report.txt and CSVs")


if __name__ == "__main__":
    main()
