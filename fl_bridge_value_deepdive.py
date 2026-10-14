"""
fl_bridge_value_deepdive.py -- how much is other-league history (College /
Pioneer / affiliated MiLB) actually worth for projecting a Frontier League
pitcher, and for WHICH pitchers?

Part A  Held-out ablation. Experiment 1 and Experiment 2 (the harness every
        other change in this project is judged on): refit the K% and BB% models
        WITH and WITHOUT the other-league terms on the train transition, score
        on the held-out period, and break the difference down by how much FL
        data the pitcher already had (feature-window PA) and by whether he had
        any bridge match at all.
Part B  The bridge for genuine FL debuts (the population it was FIT on): how
        well does each league's prior-year K%/BB% predict first-FL-year K%/BB%
        vs just guessing the league mean? Leave-one-out, PA-weighted, split by
        how much source-league data stood behind the prediction.

Usage:  python fl_bridge_value_deepdive.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import fl_college_bridge as cbridge
import fl_pioneer_bridge as pbridge
import fl_milb_bridge as mbridge
import mlb_marcel as mm

K_FULL = ["naive_K_rate", "fl_k_rate_bridge", "wh_sva_level", "spin_z_norm", "extension"]
K_NOBR = ["naive_K_rate", "wh_sva_level", "spin_z_norm", "extension"]
BB_FULL = ["blended_BB_rate", "command_score"]
BB_NOBR = ["naive_BB_rate", "command_score"]
PA_BUCKETS = [(-1, 30, "<=30 PA"), (30, 100, "31-100"), (100, 250, "101-250"), (250, 99999, ">250")]


def fit_pred(train, test, target, regs, n0, w_train_col="PA_test"):
    d = train.dropna(subset=[target] + regs)
    w = mm.reliability_weight(d[w_train_col], n0).values
    m = bt.fit_weighted(d, target, regs, w)
    t = test.dropna(subset=[target] + regs).copy()
    t["pred"] = m.predict(t).values
    return t


def bucket_report(t_full, t_nobr, target, n0, label):
    """R2_w overall + per PA bucket + per matched/unmatched, with and without bridge."""
    t = t_full[["pitcher_id", target, "PA", "PA_test", "has_college_match", "pred"]].rename(columns={"pred": "p_full"}).merge(
        t_nobr[["pitcher_id", "pred"]].rename(columns={"pred": "p_nobr"}), on="pitcher_id")
    t["w"] = mm.reliability_weight(t["PA_test"], n0).values
    rows = []

    def add(name, d):
        if len(d) < 8:
            return
        y, w = d[target].values, d["w"].values
        sse_f = np.sum(w * (y - d["p_full"]) ** 2); sse_n = np.sum(w * (y - d["p_nobr"]) ** 2)
        rows.append({"slice": name, "n": len(d),
                     "R2_w no-bridge": round(mm.weighted_r2(y, d["p_nobr"], w), 3),
                     "R2_w WITH bridge": round(mm.weighted_r2(y, d["p_full"], w), 3),
                     "RMSE_w pp (no-br -> with)": f"{mm.weighted_rmse(y, d['p_nobr'], w)*100:.2f} -> {mm.weighted_rmse(y, d['p_full'], w)*100:.2f}",
                     "SSE cut %": round(100 * (1 - sse_f / sse_n), 1)})

    add("ALL", t)
    for lo, hi, nm in PA_BUCKETS:
        add(f"feature PA {nm}", t[(t["PA"] > lo) & (t["PA"] <= hi)])
    add("bridge-matched", t[t["has_college_match"]])
    add("no bridge match", t[~t["has_college_match"]])
    out = pd.DataFrame(rows)
    print(f"\n{label}"); print(out.to_string(index=False))
    return out


def run_experiment(raw, name, train_feat, train_lab, test_feat, test_lab):
    df_tr, n0_k, n0_bb = bt.assemble(raw, train_feat, train_lab, with_changepoints=False)
    df_te, _, _ = bt.assemble(raw, test_feat, test_lab, with_changepoints=False)
    # BB blended input needs the window's league BB rate + fitted n0_bridge
    def lg_bb(windows):
        pool = pd.concat([fp.filter_period(raw, s, e) for s, e in windows], ignore_index=True)
        t = pool[pool["is_terminal"]]
        return (t["k_or_bb"] == "walk").mean()
    lg_tr, lg_te = lg_bb(train_feat), lg_bb(test_feat)
    for df, lg in ((df_tr, lg_tr), (df_te, lg_te)):
        df["BB_raw"] = df["naive_BB_rate"] * (df["PA"] + n0_bb) - n0_bb * lg
    d_bb = df_tr.dropna(subset=["BB_rate_actual", "naive_BB_rate", "fl_bb_rate_bridge", "PA", "command_score"])
    w = mm.reliability_weight(d_bb["PA_test"], n0_bb).values
    n0b = mm.fit_n0_bridge(d_bb["BB_raw"], d_bb["PA"], n0_bb, lg_tr, d_bb["fl_bb_rate_bridge"], d_bb["BB_rate_actual"], w)
    df_tr["blended_BB_rate"] = mm.blended_rate(df_tr["BB_raw"], df_tr["PA"], n0_bb, lg_tr, n0b, df_tr["fl_bb_rate_bridge"])
    df_te["blended_BB_rate"] = mm.blended_rate(df_te["BB_raw"], df_te["PA"], n0_bb, lg_te, n0b, df_te["fl_bb_rate_bridge"])
    print(f"\n==== {name}: train n={len(df_tr)}, test n={len(df_te)}; fitted n0_bridge_bb={n0b:.1f} PA-equivalents ====")
    kf = fit_pred(df_tr, df_te, "K_rate_actual", K_FULL, n0_k)
    kn = fit_pred(df_tr, df_te, "K_rate_actual", K_NOBR, n0_k)
    bucket_report(kf, kn, "K_rate_actual", n0_k, f"{name} -- K%")
    bf = fit_pred(df_tr, df_te, "BB_rate_actual", BB_FULL, n0_bb)
    bn = fit_pred(df_tr, df_te, "BB_rate_actual", BB_NOBR, n0_bb)
    bucket_report(bf, bn, "BB_rate_actual", n0_bb, f"{name} -- BB%")


# --------------------------------------------------------------------------- #
#  Part B -- debuts
# --------------------------------------------------------------------------- #
def loo_wls(d, y, x, w):
    """Leave-one-out predictions of a weighted simple regression, plus the PA-
    weighted league-mean baseline (also leave-one-out)."""
    X = np.column_stack([np.ones(len(d)), d[x].values]); Y = d[y].values; W = d[w].values
    pred, base = np.zeros(len(d)), np.zeros(len(d))
    for i in range(len(d)):
        m = np.ones(len(d), bool); m[i] = False
        Xw = X[m] * W[m, None]
        beta = np.linalg.solve(X[m].T @ Xw, Xw.T @ Y[m])
        pred[i] = X[i] @ beta
        base[i] = np.average(Y[m], weights=W[m])
    return pred, base


def debut_report(name, d, k_src, bb_src, pa_col, tbf_latest="tbf_latest", tbf_multi="tbf_multi"):
    d = d.dropna(subset=["k_rate", "bb_rate", k_src, bb_src]).copy()
    rows = []
    for metric, y, x in (("K%", "k_rate", k_src), ("BB%", "bb_rate", bb_src)):
        pred, base = loo_wls(d, y, x, pa_col)
        w = d[pa_col].values
        rows.append({"league": name, "metric": metric, "n": len(d),
                     "corr(src, FL)": round(d[x].corr(d[y]), 3),
                     "LOO R2_w vs league-mean": round(1 - np.sum(w * (d[y] - pred) ** 2) / np.sum(w * (d[y] - base) ** 2), 3),
                     "RMSE_w pp (mean -> bridge)": f"{mm.weighted_rmse(d[y], base, w)*100:.2f} -> {mm.weighted_rmse(d[y], pred, w)*100:.2f}"})
        # split by sample behind the source-league rate
        tcol = tbf_latest if metric == "K%" else tbf_multi
        if tcol in d.columns:
            for lab, mask in ((f"{tcol}<150", d[tcol] < 150), (f"{tcol}>=150", d[tcol] >= 150)):
                if mask.sum() >= 12:
                    sub = d[mask]; ps, bs = pred[mask.values], base[mask.values]; ws = sub[pa_col].values
                    rows.append({"league": name, "metric": f"  {metric} {lab}", "n": len(sub),
                                 "corr(src, FL)": round(sub[x].corr(sub[y]), 3),
                                 "LOO R2_w vs league-mean": round(1 - np.sum(ws * (sub[y] - ps) ** 2) / np.sum(ws * (sub[y] - bs) ** 2), 3),
                                 "RMSE_w pp (mean -> bridge)": f"{mm.weighted_rmse(sub[y], bs, ws)*100:.2f} -> {mm.weighted_rmse(sub[y], ps, ws)*100:.2f}"})
    return rows


if __name__ == "__main__":
    raw = fp.load_fl_pitches_cached()
    w25, w26 = (bt.START_2025, bt.END_2025), (bt.START_2026, bt.END_2026)
    print("PART A -- held-out ablation: other-league terms removed vs kept")
    run_experiment(raw, "Exp2 (train full-2025 -> 1H26; test 2025+1H26 -> 2H26)",
                   [w25], [(bt.START_2026, bt.CUTOFF_2026)],
                   [w25, (bt.START_2026, bt.CUTOFF_2026)], [(bt._DAY_AFTER_2026_CUT, bt.END_2026)])
    run_experiment(raw, "Exp1 (train 1H25 -> 2H25; test 2H25 -> 1H26)",
                   [(bt.START_2025, bt.CUTOFF_2025)], [(bt._DAY_AFTER_2025_CUT, bt.END_2025)],
                   [(bt._DAY_AFTER_2025_CUT, bt.END_2025)], [(bt.START_2026, bt.CUTOFF_2026)])

    print("\n\nPART B -- the bridge for genuine FL debuts (leave-one-out, PA-weighted)")
    cm = cbridge.fit_bridge()
    cpairs = pd.read_parquet(cbridge.BRIDGE_PATH).drop_duplicates("pitcher_id")
    print("college pair columns:", list(cpairs.columns))
    rows = []
    pm = pbridge.fit_bridge(raw); mm_ = mbridge.fit_bridge(raw)
    d = pm["transition_pairs"].copy()
    rows += debut_report("Pioneer", d, "k_pct", "bb_pct_multi", "pa")
    d = mm_["transition_pairs"].copy()
    rows += debut_report("Affiliated MiLB", d, "k_pct", "bb_pct_multi", "pa")
    # college pairs need the same latest/multi features as fit_bridge builds
    cy = cm["college_years"]
    feat_rows = []
    for _, r in cpairs.iterrows():
        f = cbridge._latest_and_multi(cy, r["name_clean"], as_of_season=r["college_year"])
        if f is not None:
            feat_rows.append({"pitcher_id": r["pitcher_id"], **f})
    dc = cpairs.merge(pd.DataFrame(feat_rows), on="pitcher_id", how="inner", suffixes=("", "_h"))
    pa_col = "pa" if "pa" in dc.columns else ("PA" if "PA" in dc.columns else None)
    if pa_col is None:
        dc["pa"] = 1.0; pa_col = "pa"
    rows += debut_report("College", dc, "k_pct", "bb_pct_multi", pa_col)
    print(pd.DataFrame(rows).to_string(index=False))
