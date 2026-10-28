"""
fl_bridge_role_test.py -- does SOURCE-LEAGUE ROLE / USAGE (innings per game, batters faced per game, share of games started) predict a debut pitcher's FL K% and BB% beyond what the unified bridge already projects?  (user 2026-10-28: "the data should have info on source league role/usage ... we should test that")
The unified bridge (fl_unified_bridge.py) uses only the source K% / BB%, batters faced (for shrinkage), the staleness gap and the league.  This adds usage features computed from the same source seasons (before the debut).
DATA: affiliated MiLB only (output/milb_pitching_2018_2026.parquet carries games, games started, innings and batters faced for every pitcher-season-level); the saved college (FanGraphs) and Pioneer (StatsCrew) pitching tables keep only batters faced, strikeouts and walks, so those two leagues cannot be tested without re-pulling games / innings.
DESIGN: the debut pairs of the unified bridge (first FL season 2025 and 2026; MiLB source: ~196 pairs); bridge prediction = leave-one-out weighted regression (as fl_rookie_value_runs.py: K% = league intercepts + slope x u_k, BB% = intercept + slope x u_bb, weights = FL PA); residual = actual - LOO prediction.  Features (over the MiLB seasons before the debut):
  ipg (innings per game), bfpg (batters faced per game), gs_share (games started / games), and each relative to the same level-season average ('_rel'; roles differ by level).  Tests: weighted regression of the residual on one feature at a time (per SD), player bootstrap t; leave-one-out add-on (coefficient refit on the other pairs) scored with the weighted RMSE / MAE and a paired bootstrap;
frozen inverse-variance weights from the variance law of the league-mean-guess residuals.   Output: output/bridge_role_test_log.txt, output/bridge_role_test.json
"""
import contextlib
import io
import json
import warnings

import numpy as np
import pandas as pd

import fl_college_bridge as cb
import fl_eval_report as er
import fl_eval_weights as ew
import fl_milb_bridge as mb
import fl_pitcher_features as fp
import fl_position_player_flag as ppf
import fl_unified_bridge as ub

warnings.filterwarnings("ignore")
rng = np.random.default_rng(12)


def ip_to_float(ip):
    ip = pd.to_numeric(ip, errors="coerce"); whole = np.floor(ip); frac = np.round((ip - whole) * 10)
    return whole + frac / 3.0


def milb_usage():
    d = pd.read_parquet(mb.MILB_PATH); d["ip"] = ip_to_float(d["inningsPitched"]); d["name_clean"] = d["player_name"].apply(mb._clean_name); d["pid"] = d["pitcher"].astype(str)
    lvl = d.groupby(["level", "season"]).agg(ip=("ip", "sum"), g=("gamesPlayed", "sum"), gs=("gamesStarted", "sum"), bf=("battersFaced", "sum")); lvl["ipg"] = lvl["ip"] / lvl["g"]; lvl["bfpg"] = lvl["bf"] / lvl["g"]; lvl["gs_share"] = lvl["gs"] / lvl["g"]
    d = d.join(lvl[["ipg", "bfpg", "gs_share"]].rename(columns=lambda c: c + "_lvl"), on=["level", "season"])
    d["ipg_lvl_w"] = d["ipg_lvl"] * d["gamesPlayed"]; d["bfpg_lvl_w"] = d["bfpg_lvl"] * d["gamesPlayed"]; d["gs_lvl_w"] = d["gs_share_lvl"] * d["gamesPlayed"]
    return d.groupby(["pid", "name_clean", "season"]).agg(g=("gamesPlayed", "sum"), gs=("gamesStarted", "sum"), ip=("ip", "sum"), bf=("battersFaced", "sum"), ipg_lvl_w=("ipg_lvl_w", "sum"), bfpg_lvl_w=("bfpg_lvl_w", "sum"), gs_lvl_w=("gs_lvl_w", "sum")).reset_index()


def build_pairs():
    raw = ppf.exclude(fp.load_fl_pitches_cached())
    with contextlib.redirect_stdout(io.StringIO()):
        mdl = ub.fit(raw, verbose=False)
    p = mdl.pairs.copy().reset_index(drop=True); w = p["pa"].to_numpy(float); leagues = sorted(p["league"].unique()); I = np.column_stack([(p["league"] == lg).astype(float) for lg in leagues])

    def loo(X, y):
        XtWX = X.T @ (X * w[:, None]); beta = np.linalg.solve(XtWX, X.T @ (w * y)); e = y - X @ beta; h = w * np.einsum("ij,jk,ik->i", X, np.linalg.inv(XtWX), X); return y - e / (1 - h), (np.sum(w * y) - w * y) / (np.sum(w) - w)
    pk, bk = loo(np.column_stack([I, p["u_k"].to_numpy()]), p["k_rate"].to_numpy()); pb, bb = loo(np.column_stack([np.ones(len(p)), p["u_bb"].to_numpy()]), p["bb_rate"].to_numpy())
    p["k_pred"], p["k_base"], p["bb_pred"], p["bb_base"] = pk, bk, pb, bb; names = raw[["pitcher_id", "pitcher"]].drop_duplicates("pitcher_id"); names["pid"] = names["pitcher_id"].astype(str); p["pid"] = p["pitcher_id"].astype(str)
    p = p.merge(names[["pid", "pitcher"]], on="pid", how="left"); p["name_clean"] = p["pitcher"].apply(cb._fl_name_clean); return p


def features(p, U):
    rows = []
    for r in p.itertuples():
        if r.league != "milb":
            rows.append({}); continue
        g = U[(U["pid"] == r.pid) | (U["name_clean"] == r.name_clean)]; g = g[g["season"] < r.fl_year]
        if g["pid"].nunique() > 1:
            g = g[g["pid"] == (r.pid if r.pid in set(g["pid"]) else g.groupby("pid")["bf"].sum().idxmax())]
        if g.empty or g["g"].sum() == 0:
            rows.append({}); continue
        G, IP, BF, GS = g["g"].sum(), g["ip"].sum(), g["bf"].sum(), g["gs"].sum()
        f = {"ipg": IP / G, "bfpg": BF / G, "gs_share": GS / G, "ipg_rel": IP / G - g["ipg_lvl_w"].sum() / G, "bfpg_rel": BF / G - g["bfpg_lvl_w"].sum() / G, "gs_share_rel": GS / G - g["gs_lvl_w"].sum() / G, "games": float(G), "n_seasons": int(len(g))}
        rows.append(f)
    return pd.DataFrame(rows, index=p.index)


def main():
    pd.set_option("display.width", 250); U = milb_usage(); p = build_pairs(); F = features(p, U); d = pd.concat([p, F], axis=1); d = d[d["league"] == "milb"].dropna(subset=["ipg"]).reset_index(drop=True)
    print(f"{len(p)} debut pairs in all; MiLB pairs with a matched usage history: {len(d)} (of {int((p['league'] == 'milb').sum())}); FL debut years {d['fl_year'].value_counts().to_dict()}")
    print("usage features (mean / SD): " + ", ".join(f"{c} {d[c].mean():.2f}/{d[c].std():.2f}" for c in ("ipg", "bfpg", "gs_share", "ipg_rel", "bfpg_rel", "gs_share_rel")) + f";  relievers (gs_share < .2): {(d['gs_share'] < .2).mean():.0%}", flush=True)
    res = {}; n = d["pa"].to_numpy(float)
    for tgt, ycol, pcol, bcol in (("K%", "k_rate", "k_pred", "k_base"), ("BB%", "bb_rate", "bb_pred", "bb_base")):
        y = d[ycol].to_numpy(); pred = d[pcol].to_numpy(); base = d[bcol].to_numpy(); a, b = ew.fit_variance_law(y - base, n); w = ew.inv_var_weights(n, a, b); e = y - pred; rows = []
        for f in ("ipg", "bfpg", "gs_share", "ipg_rel", "bfpg_rel", "gs_share_rel"):
            x = d[f].to_numpy(float); z = (x - np.average(x, weights=w)) / np.sqrt(np.average((x - np.average(x, weights=w)) ** 2, weights=w)); X = np.column_stack([np.ones(len(z)), z, pred - np.average(pred, weights=w)])
            def fit(ii):
                sw = np.sqrt(w[ii]); return np.linalg.lstsq(X[ii] * sw[:, None], e[ii] * sw, rcond=None)[0]
            b_ = fit(np.arange(len(z))); bs = np.array([fit(rng.integers(0, len(z), len(z)))[1] for _ in range(600)])
            # leave-one-out add-on
            loo = np.zeros(len(z))
            for i in range(len(z)):
                m = np.ones(len(z), bool); m[i] = False; sw = np.sqrt(w[m]); bb_ = np.linalg.lstsq(X[m][:, :2] * sw[:, None], e[m] * sw, rcond=None)[0]; loo[i] = X[i, :2] @ bb_
            old = pred + np.average(e, weights=w); new = pred + loo; new = new + np.average(y - new, weights=w); c = er.compare(y, old, new, w, d["pid"].to_numpy(), n=1000)
            rows.append({"feature": f, "coef per SD (points)": 100 * b_[1], "t": b_[1] / bs.std(), "dRMSE": c["dRMSE"], "90%": c["RMSE 90%"], "P(RMSE better)": c["P(RMSE better)"], "dMAE": c["dMAE"], "P(MAE better)": c["P(MAE better)"], "RMSE old": c["RMSE old"] if "RMSE old" in c else np.nan})
        T = pd.DataFrame(rows); print(f"\n{tgt}: residual of the bridge vs source-league usage (n = {len(d)}; LOO add-on; negative dRMSE = better)\n" + T.round(4).to_string(index=False), flush=True); res[tgt] = T.to_dict("records")
    # simple descriptive: starters vs relievers
    d["starter"] = d["gs_share"] >= 0.5; g = d.groupby("starter").apply(lambda s: pd.Series({"pairs": len(s), "mean K% miss (pts)": 100 * np.average(s["k_rate"] - s["k_pred"], weights=s["pa"]), "mean BB% miss (pts)": 100 * np.average(s["bb_rate"] - s["bb_pred"], weights=s["pa"])})); print("\nSource starters (>= half of games started) vs others: average miss of the bridge, FL PA-weighted\n" + g.round(2).to_string())
    json.dump(res, open("output/bridge_role_test.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
