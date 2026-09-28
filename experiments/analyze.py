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

    # per-token acceptance prob of the sampled token, drafter s -> verifier v
    def alpha(self, s: int, v: int) -> np.ndarray:
        lp = self.data[s]["tok_logp"]
        return np.minimum(1.0, np.exp(lp[..., self.di[v]] - lp[..., self.di[s]]))

    # E[accepted tokens | proposal length n] for n = 1..N, plus standard error
    def tau_curve(self, s: int, v: int):
        surv = np.cumprod(self.alpha(s, v), axis=1)
        per_window = np.cumsum(surv, axis=1)
        return per_window.mean(0), per_window.std(0) / np.sqrt(per_window.shape[0])

    def tv(self, s: int, a: int, b: int) -> np.ndarray:
        a, b = min(a, b), max(a, b)
        return self.data[s]["tv"][..., self.pi[(a, b)]]

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
                best = (0, None, None)
                for g in gammas:
                    b1 = curves[(d, i)][g - 1] + 1
                    stage1 = g * c(d) + (c(i) if mode == "noreuse" else (i - d) + H)
                    for ni in range(1, N + 1):
                        tok = curves[(i, L)][ni - 1] + 1
                        cost = (ni / b1) * stage1 + (c(L) if mode == "noreuse" else (L - i) + H)
                        best = max(best, (c(L) * tok / cost, g, ni))
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
    trans, cost, pred = [], [], []
    for T in temps:
        r = Run(run_dir, T)
        if not r.data:
            rep(f"(no data for T={T})")
            continue
        rep("\n" + "=" * 100 + f"\nTemperature {T}\n" + "=" * 100)
        section_a(r, rep)
        section_b(r, rep, trans)
        section_c(r, rep, focus=[])
        section_d(r, rep)
        section_e(r, rep, cost)
        section_f(r, rep, pred)
        section_g(r, rep)

    pd.DataFrame(trans).to_csv(out / "stage_transfer.csv", index=False)
    pd.DataFrame(cost).to_csv(out / "cost_model.csv", index=False)
    pd.DataFrame(pred).to_csv(out / "predictors.csv", index=False)
    (out / "report.txt").write_text("\n".join(rep.lines))
    print(f"\nwrote {out}/report.txt and CSVs")


if __name__ == "__main__":
    main()
