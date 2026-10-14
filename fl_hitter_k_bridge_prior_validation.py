"""
fl_hitter_k_bridge_prior_validation.py -- EXHAUSTIVE validation of the hitter K% BRIDGE PRIOR (user 2026-10-13: "Big RMSE drop. Can we do a more exhaustive validation for it?").
Candidate: for hitters with a MiLB record before their FL debut, the K% regression shrinks toward the hitter's BRIDGE prediction instead of the league mean:
  production   pred = lg_next + (n_t dev_t + d n_p dev_p + d^2 n_pp dev_pp) / (n_t + d n_p + d^2 n_pp + n0)
  bridge prior pred = lg_next + (n_t dev_t + d n_p dev_p + d^2 n_pp dev_pp + n0 x bridge_dev) / (same denominator)
  bridge_dev = the out-of-fold bridge prediction of his FL K% deviation (leave-one-debut-class-out, fl_hitter_bridge_test), n0 = production K% n0 refit leave-one-transition-out.
Pre-specified checks (nothing is tuned on the evaluation rows):
  1  every FL hitter-transition 2023->24, 2024->25, 2025->26 of a hitter with a bridge match (not only second seasons); next-season PA >= 20 (main) and >= 1 / >= 100 (variants)
  2  breakdowns by transition and by FL history the regression uses (< 150, 150-399, 400+ PA)
  3  permutation test: bridge_dev shuffled among the hitters of the same transition (500 shuffles) -> is the gain the hitter's own MiLB information or just extra shrinkage?
  4  sensitivity (reported, not chosen): prior weight n0 x 0.5 and x 2
  5  full production path on 2025 -> 2026: Whiff% prior term (fl_hitter_pre2026 build, 2025 pitches only) added to both arms
  6  wOBA consequence: the K% change fed through the component weight (-.177) into the 50/50 wOBA blend
Scoring: level-free, standard inverse-variance weights (frozen from production's residuals), weighted RMSE and MAE, hitter-cluster bootstrap (1,000).
"""
import contextlib
import io
import warnings

import numpy as np
import pandas as pd

import fl_eval_report as er
import fl_eval_weights as ew

warnings.filterwarnings("ignore")


def build(min_next=20):
    import fl_hitter_baseline_backtest as bt
    import fl_hitter_bridge_test as hb
    import fl_hitter_depth3_test as d3
    import fl_hitter_model_dev as dev
    import fl_hitter_projection as hp
    with contextlib.redirect_stdout(io.StringIO()):
        res = hb.main()
    k = res["k"]; br = pd.DataFrame({"nm": k["dd"]["nm"].to_numpy(), "debut": k["dd"]["year"].to_numpy(), "bridge_dev": np.asarray(k["P"]["M2 pooled slope"], float)})
    p = dev.load(); q = p.copy(); q["woba"] = q["woba_n"]
    saved = bt.MIN_NEXT; bt.MIN_NEXT = min_next; d = d3.frame(q, "k"); bt.MIN_NEXT = saved
    delta = hp.SPEC["k"][0]; parts = []
    for t in sorted(d["t"].unique()):
        tr, te = d[d["t"] != t], d[d["t"] == t].copy(); te["n0"] = d3.fit_n0(tr, delta, 3); parts.append(te)
    E = pd.concat(parts); E["nm"] = E["name_clean"].map(hb.norm); E = E.merge(br, on="nm", how="inner"); E = E[E["debut"] <= E["t"]].copy()
    dev_t = E["x_t"] - E["lg_t"]; dev_p = (E["x_prev"] - E["lg_prev"].fillna(E["lg_t"])).fillna(0.0); dev_pp = (E["x_pp"] - E["lg_pp"].fillna(E["lg_t"])).fillna(0.0)
    E["num"] = E["n_t"] * dev_t + delta * E["n_prev"] * dev_p + delta ** 2 * E["n_pp"] * dev_pp; E["nh"] = E["n_t"] + delta * E["n_prev"] + delta ** 2 * E["n_pp"]
    E["hist"] = E["n_t"] + E["n_prev"] + E["n_pp"]; E = E[E["t"] >= 2023].reset_index(drop=True)
    return E


def preds(E, prior, scale=1.0):
    n0 = E["n0"] * scale; return E["lg_next"] + (E["num"] + n0 * prior) / (E["nh"] + n0)


def compare(y, n, a, b, ids):
    w = ew.weights_from_base(y, a, n)[0]; c = lambda p: p + np.average(y - p, weights=w)
    return er.compare(y, c(a), c(b), w, ids, n=1000), w


def row(lab, E, prod, alt, extra=None):
    r, _ = compare(E["x_next"].to_numpy(), E["n_next"].to_numpy(), prod.to_numpy(), alt.to_numpy(), E["name_clean"].to_numpy())
    out = {"check": lab, "n": len(E), "hitters": E["name_clean"].nunique(), "RMSE prod": r["RMSE old"], "RMSE bridge prior": r["RMSE new"], "dRMSE": r["dRMSE"], "RMSE 90%": r["RMSE 90%"],
           "P(bridge prior better)": r["P(RMSE better)"], "MAE prod": r["MAE old"], "MAE bridge prior": r["MAE new"], "dMAE": r["dMAE"], "P(MAE better)": r["P(MAE better)"]}
    out.update(extra or {}); return out


def main():
    rows = []
    E = build(20); prod = preds(E, 0.0); bp = preds(E, E["bridge_dev"])
    print(f"sample: {len(E)} hitter-transitions ({E['name_clean'].nunique()} hitters) with a MiLB bridge, next-season PA >= 20; by transition {E['t'].value_counts().sort_index().to_dict()}; "
          f"median FL history {E['hist'].median():.0f} PA; mean prior weight n0/(nh+n0) {np.mean(E['n0'] / (E['nh'] + E['n0'])):.2f}")
    rows.append(row("1 all (next PA >= 20)", E, prod, bp))
    for t in sorted(E["t"].unique()):
        m = E["t"] == t; rows.append(row(f"2 transition {t} -> {t + 1}", E[m], prod[m], bp[m]))
    for lab, m in (("2 history < 150 PA", E["hist"] < 150), ("2 history 150-399", (E["hist"] >= 150) & (E["hist"] < 400)), ("2 history 400+", E["hist"] >= 400)):
        if m.sum() >= 15:
            rows.append(row(lab, E[m], prod[m], bp[m]))
    for mn in (1, 100):
        Ev = build(mn); rows.append(row(f"1 variant: next PA >= {mn}", Ev, preds(Ev, 0.0), preds(Ev, Ev["bridge_dev"])))
    for sc in (0.5, 2.0):
        rows.append(row(f"4 sensitivity: prior weight n0 x {sc}", E, preds(E, 0.0, sc), preds(E, E["bridge_dev"], sc)))
    # 3 permutation test
    y = E["x_next"].to_numpy(); w = ew.weights_from_base(y, prod.to_numpy(), E["n_next"].to_numpy())[0]; c = lambda p: p + np.average(y - p, weights=w)
    rm = lambda p: np.sqrt(np.average((y - c(p)) ** 2, weights=w)); real = rm(bp.to_numpy()) - rm(prod.to_numpy())
    rng = np.random.default_rng(13); sh = []
    for _ in range(500):
        perm = E["bridge_dev"].copy()
        for t in E["t"].unique():
            m = (E["t"] == t).to_numpy(); perm.iloc[np.nonzero(m)[0]] = rng.permutation(perm.iloc[np.nonzero(m)[0]].to_numpy())
        sh.append(rm(preds(E, perm).to_numpy()) - rm(prod.to_numpy()))
    sh = np.array(sh)
    print(f"\n3 PERMUTATION TEST: real dRMSE {real:+.5f}; shuffled bridges (500, within transition) mean {sh.mean():+.5f}, 5-95% [{np.percentile(sh, 5):+.5f}, {np.percentile(sh, 95):+.5f}]; "
          f"share of shuffles at least as good as the real bridge = {np.mean(sh <= real):.3f}")
    # 5 full production path on 2025 -> 2026 (Whiff% term from 2025 pitches)
    import fl_hitter_pre2026 as pre
    with contextlib.redirect_stdout(io.StringIO()):
        d26, _, _ = pre.build_shifted()
    wt = d26.set_index("name_clean")["whiff_term"]; m = (E["t"] == 2025).to_numpy(); E5 = E[m].copy(); term = E5["name_clean"].map(wt).fillna(0.0)
    rows.append(row("5 full production 2025 -> 2026 (+ Whiff% term in both)", E5, (prod[m] + term).clip(0.03, 0.55), (bp[m] + term).clip(0.03, 0.55)))
    # 6 wOBA consequence (component half): delta wOBA = 0.5 x (-.177) x delta K%
    R = pd.DataFrame(rows); pd.set_option("display.width", 250)
    print("\nK% BRIDGE PRIOR vs PRODUCTION (level-free; dRMSE < 0 = bridge prior better)\n" + R.round(5).to_string(index=False))
    dk = (bp - prod); print(f"\n6 size of the change: K% mean |change| {dk.abs().mean():.4f} (max {dk.abs().max():.4f}); implied wOBA change via the component half 0.5 x -.177 x dK: mean |{0.5 * 0.177 * dk.abs().mean():.4f}| (max {0.5 * 0.177 * dk.abs().max():.4f})")
    R.to_csv("output/hitter_k_bridge_prior_validation.csv", index=False)


if __name__ == "__main__":
    main()
