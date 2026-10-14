"""
fl_bridge_weight_test.py -- should the other-league bridge be weighted by how much
real data stands behind it?

Debut analysis (fl_bridge_value_deepdive.py): BB% bridge LOO R2_w is +0.22 (MiLB)
when the source rate rests on >=150 batters faced but -0.22 below that; college
+0.09 vs -0.01; K% bridge ~0.03-0.05 everywhere and NEGATIVE in held-out
Experiment 1. The shipped BB% prior blends the bridge in with a CONSTANT pseudo-
count n0_bridge (~53 PA) for everyone, and the K% bridge enters as a free OLS
term with a constant coefficient.

Variants, same Experiment 2 / Experiment 1 held-out harness (paired bootstrap on
weighted squared error vs the shipped spec); the free parameters are chosen on
the TRAIN transition only:
  BB%  shipped (constant n0)        | none (n0=0)
       scaled: n0_i = n0max * tbf_i / (tbf_i + k), k in {100, 200, 400, 800}
  K%   shipped (free bridge term)   | no bridge term
       scaled: bridge term x tbf_i / (tbf_i + k)

Usage:  python fl_bridge_weight_test.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import mlb_marcel as mm
import fl_ppa_kbb_and_contact as base
import fl_bridge_value_deepdive as dd

K_GRID = (100.0, 200.0, 400.0, 800.0)
N_BOOT = 2000


def bb_blend(df, n0_bb, lg, n0_bridge_i):
    return mm.blended_rate(df["BB_raw"], df["PA"], n0_bb, lg, n0_bridge_i, df["fl_bb_rate_bridge"])


def fit_scaled_n0(train, n0_bb, lg, k, w):
    d = train.dropna(subset=["BB_rate_actual", "command_score"])
    ww = mm.reliability_weight(d["PA_test"], n0_bb).values
    t = d["bb_bridge_tbf"].values / (d["bb_bridge_tbf"].values + k)

    def sse(n0max):
        bl = mm.blended_rate(d["BB_raw"], d["PA"], n0_bb, lg, n0max * t, d["fl_bb_rate_bridge"])
        mdl = bt.fit_weighted(d.assign(blended_BB_rate=bl), "BB_rate_actual", dd.BB_FULL, ww)
        return float(np.sum(ww * mdl.resid.values ** 2))
    r = minimize_scalar(sse, bounds=(0.0, 1000.0), method="bounded")
    return float(r.x), float(r.fun)


def boot(y, w, se_a, se_b, seed=0):
    rng = np.random.default_rng(seed); d = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        d.append((np.sum(w[i] * se_a[i]) - np.sum(w[i] * se_b[i])) / np.sum(w[i] * se_a[i]))
    return 100 * np.mean(d), 100 * np.percentile(d, 5), 100 * np.percentile(d, 95)


def report(name, rows):
    print(f"\n{name}"); print(pd.DataFrame(rows).to_string(index=False))


def run(raw, name, trf, trl, tef, tel):
    df_tr, df_te, n0_k, n0_bb = base.prep(raw, trf, trl, tef, tel)
    pool = lambda wins: pd.concat([fp.filter_period(raw, s, e) for s, e in wins], ignore_index=True)
    lg = lambda wins: (lambda t: (t["k_or_bb"] == "walk").mean())(pool(wins)[pool(wins)["is_terminal"]])
    lg_tr, lg_te = lg(trf), lg(tef)
    print(f"\n######## {name}: train n={len(df_tr)}, test n={len(df_te)}; matched share train {df_tr['has_college_match'].mean():.0%}, "
          f"median bridge TBF (matched) K {df_tr.loc[df_tr.has_college_match,'k_bridge_tbf'].median():.0f} / BB {df_tr.loc[df_tr.has_college_match,'bb_bridge_tbf'].median():.0f}")

    # ---------------- BB% ----------------
    y_all = "BB_rate_actual"
    w_tr = mm.reliability_weight(df_tr["PA_test"], n0_bb).values
    variants = {}
    # shipped: constant n0 fit by NLS (already in base.prep as blended_BB_rate)
    variants["shipped (constant n0)"] = (df_tr["blended_BB_rate"], df_te["blended_BB_rate"], None)
    variants["no bridge"] = (bb_blend(df_tr, n0_bb, lg_tr, 0.0), bb_blend(df_te, n0_bb, lg_te, 0.0), None)
    best = None
    for k in K_GRID:
        n0max, fun = fit_scaled_n0(df_tr, n0_bb, lg_tr, k, w_tr)
        variants[f"scaled k={k:.0f} (n0max={n0max:.0f})"] = (
            bb_blend(df_tr, n0_bb, lg_tr, n0max * df_tr["bb_bridge_tbf"] / (df_tr["bb_bridge_tbf"] + k)),
            bb_blend(df_te, n0_bb, lg_te, n0max * df_te["bb_bridge_tbf"] / (df_te["bb_bridge_tbf"] + k)), fun)
        if best is None or fun < best[1]:
            best = (f"scaled k={k:.0f} (n0max={n0max:.0f})", fun)
    rows = []
    te_ok = df_te.dropna(subset=[y_all, "command_score"]).index
    y = df_te.loc[te_ok, y_all].values; w = mm.reliability_weight(df_te.loc[te_ok, "PA_test"], n0_bb).values
    preds = {}
    for nm, (btr, bte, _) in variants.items():
        d_tr = df_tr.assign(blended_BB_rate=btr).dropna(subset=[y_all, "command_score"])
        m = bt.fit_weighted(d_tr, y_all, dd.BB_FULL, mm.reliability_weight(d_tr["PA_test"], n0_bb).values)
        preds[nm] = m.predict(df_te.loc[te_ok].assign(blended_BB_rate=bte.loc[te_ok])).values
    se0 = (y - preds["shipped (constant n0)"]) ** 2
    for nm, p in preds.items():
        se = (y - p) ** 2
        ci = "" if nm.startswith("shipped") else "{:+.1f} [{:+.1f}, {:+.1f}]".format(*boot(y, w, se0, se))
        rows.append({"BB% variant": nm + ("  <- train-SSE pick" if nm == best[0] else ""), "R2_w": round(mm.weighted_r2(y, p, w), 4),
                     "RMSE_w pp": round(mm.weighted_rmse(y, p, w) * 100, 3), "SSE cut vs shipped % [90% CI]": ci})
    report(f"{name} -- BB%", rows)

    # ---------------- K% ----------------
    lgk = lambda wins: (lambda t: (t["k_or_bb"] == "strikeout").mean())(pool(wins)[pool(wins)["is_terminal"]])
    lk_tr, lk_te = lgk(trf), lgk(tef)
    for df, lgv in ((df_tr, lk_tr), (df_te, lk_te)):
        for k in K_GRID:
            df[f"kb_{int(k)}"] = (df["fl_k_rate_bridge"] - lgv) * df["k_bridge_tbf"] / (df["k_bridge_tbf"] + k)
    kvars = {"shipped (free bridge term)": dd.K_FULL, "no bridge term": dd.K_NOBR}
    for k in K_GRID:
        kvars[f"bridge x tbf/(tbf+{k:.0f})"] = [c for c in dd.K_FULL if c != "fl_k_rate_bridge"] + [f"kb_{int(k)}"]
    cols = sorted({c for v in kvars.values() for c in v})
    tr = df_tr.dropna(subset=["K_rate_actual"] + cols); te = df_te.dropna(subset=["K_rate_actual"] + cols)
    y = te["K_rate_actual"].values; w = mm.reliability_weight(te["PA_test"], n0_k).values
    wt = mm.reliability_weight(tr["PA_test"], n0_k).values
    pk, train_sse = {}, {}
    for nm, regs in kvars.items():
        m = bt.fit_weighted(tr, "K_rate_actual", regs, wt)
        pk[nm] = m.predict(te).values; train_sse[nm] = float(np.sum(wt * m.resid.values ** 2))
    se0 = (y - pk["shipped (free bridge term)"]) ** 2; rows = []
    for nm, p in pk.items():
        se = (y - p) ** 2
        ci = "" if nm.startswith("shipped") else "{:+.1f} [{:+.1f}, {:+.1f}]".format(*boot(y, w, se0, se))
        rows.append({"K% variant": nm, "R2_w": round(mm.weighted_r2(y, p, w), 4), "RMSE_w pp": round(mm.weighted_rmse(y, p, w) * 100, 3),
                     "SSE cut vs shipped % [90% CI]": ci, "train SSE": round(train_sse[nm], 4)})
    report(f"{name} -- K%", rows)


if __name__ == "__main__":
    raw = fp.load_fl_pitches_cached()
    w25, w26 = (bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)
    run(raw, "Exp2 (train full-2025 -> 1H26; test 2025+1H26 -> 2H26)", [w25], [w26], [w25, w26], [(bt._DAY_AFTER_2026_CUT, bt.END_2026)])
    run(raw, "Exp1 (train 1H25 -> 2H25; test 2H25 -> 1H26)", [(bt.START_2025, bt.CUTOFF_2025)], [(bt._DAY_AFTER_2025_CUT, bt.END_2025)],
        [(bt._DAY_AFTER_2025_CUT, bt.END_2025)], [w26])
