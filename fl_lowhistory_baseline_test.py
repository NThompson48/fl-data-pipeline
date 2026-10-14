"""
fl_lowhistory_baseline_test.py -- are the projections for ROOKIES (bridge), LOW-HISTORY and other thin players better than NAIVE baselines?  (user 2026-10-13)
No variable selection: existing out-of-sample predictions are scored against naive predictions on IDENTICAL rows.
Scoring: weighted RMSE and MAE, standard inverse-variance weights frozen from the model's residuals (fl_eval_weights), LEVEL-FREE (every method gets its own constant
removed: the production output is recentred to the league level, and naive methods would otherwise lose just on level), player bootstrap (1,000) for model minus baseline.

A  HITTER ROOKIES, MiLB bridge (200 FL debuts 2023-26, nested leave-one-debut-class-out predictions from fl_hitter_bridge_test): wOBA, K%, BB% of the first FL season.
   Naive: league mean (debut class); RAW = the latest MiLB season's rate at face value (no level adjustment, no shrinkage); LEVEL-ADJ RAW = that rate's deviation from its own
   level-season mean, unshrunk, added to the FL league (a translation with slope 1).
B  PITCHER ROOKIES, unified bridge (FL debuts with college / Pioneer / MiLB history, leave-one-out): K%, BB% of the first FL season.  Naive: league mean; RAW (latest-season K%,
   multi-year BB%, face value); LEVEL-ADJ RAW (deviation from the source league's mean, unshrunk).
C  HITTERS WITH THIN FL HISTORY (< 150 PA over the history the model uses), transitions 2022->23 ... 2025->26, the production three-season regression with n0 fit
   leave-one-transition-out (target next season >= 100 PA, the backtest's rule): wOBA (park-neutral), K%, BB%.  Naive: league mean; last season at face value (+ league shift);
   one-season shrinkage (n0 fit on the other transitions).  Regulars (>= 150) shown alongside.
D  PITCHERS WITH THIN HISTORY, fold 2 of the production harness (features 2025 + 1H2026 -> 2H2026): K%, BB%; thin = < 150 BF in the window, rookies = no 2025 appearance.
   Naive: league mean; own window rate at face value; the model's own naive shrunk rate (naive_K_rate / naive_BB_rate, no stuff / bridge / command terms).
E  PITCHER RA9 2025 -> 2026 (fl_pwar_calibration build, >= 10 IP), thin = < 30 IP in 2025 incl. no 2025 record.  Naive: league RA9; own 2025 RA9 face value; shrunk (100 IP).
"""
import contextlib
import io
import warnings

import numpy as np
import pandas as pd

import fl_eval_report as er
import fl_eval_weights as ew

warnings.filterwarnings("ignore")
ROWS = []


def score(group, stat, y, n, model, naive: dict, ids, mask=None):
    m = np.ones(len(y), bool) if mask is None else np.asarray(mask, bool)
    y, n, model, ids = np.asarray(y, float)[m], np.asarray(n, float)[m], np.asarray(model, float)[m], np.asarray(ids)[m]
    naive = {k: np.asarray(v, float)[m] for k, v in naive.items()}
    ok = np.isfinite(y) & np.isfinite(model) & np.isfinite(n) & np.all([np.isfinite(v) for v in naive.values()], axis=0)
    y, n, model, ids = y[ok], n[ok], model[ok], ids[ok]; naive = {k: v[ok] for k, v in naive.items()}
    if len(y) < 15:
        return
    w = ew.weights_from_base(y, model, n)[0]; c = lambda p: p + np.average(y - p, weights=w)
    rm, ma = er.wmetrics(y, c(model), w)
    for k, v in naive.items():
        r = er.compare(y, c(v), c(model), w, ids, n=1000)
        ROWS.append({"group": group, "stat": stat, "n": len(y), "baseline": k, "RMSE baseline": r["RMSE old"], "RMSE model": r["RMSE new"], "dRMSE (model - base)": r["dRMSE"], "RMSE 90%": r["RMSE 90%"],
                     "P(model better)": r["P(RMSE better)"], "MAE baseline": r["MAE old"], "MAE model": r["MAE new"], "dMAE": r["dMAE"], "P(model better, MAE)": r["P(MAE better)"]})


def part_a():
    import fl_hitter_bridge_test as bt
    with contextlib.redirect_stdout(io.StringIO()):
        res = bt.main()
    m = bt.milb_table(); lvl = m.groupby(["level", "season"]).apply(lambda g: pd.Series({s: np.average(g[s], weights=g["plateAppearances"]) for s in ("woba", "k", "bbr")}))
    for s, lab in (("woba", "wOBA"), ("k", "K%"), ("bbr", "BB%")):
        r = res[s]; dd = r["dd"].reset_index(drop=True); y = np.asarray(r["y"], float); n = np.asarray(r["n"], float)
        raw, adj = [], []
        for nm, yr in zip(dd["nm"], dd["year"]):
            g = m[(m["nm"] == nm) & (m["season"] < yr)]; ls = g["season"].max(); g = g[g["season"] == ls]
            v = np.average(g[s], weights=g["plateAppearances"]); raw.append(v)
            adj.append(np.average(g[s] - [lvl.loc[(L, ls), s] for L in g["level"]], weights=g["plateAppearances"]))
        model = np.asarray(r["P"]["M2 pooled slope"], float)
        if s == "woba":                                                        # production bridge wOBA = 0.5 direct + 0.5 component-built (as in fl_rookie_value_test)
            comp = np.zeros(len(dd))
            for c_, b in {"k": -0.177, "bbr": 0.516, "hrr": 1.414}.items():
                q = res[c_]; j = {nm: i for i, nm in enumerate(q["dd"]["nm"])}; comp += b * np.array([q["P"]["M2 pooled slope"][j[nm]] if nm in j else 0.0 for nm in dd["nm"]])
            model = 0.5 * model + 0.5 * comp
        score("A hitter rookies (MiLB bridge)", lab, y, n, model, {"league mean": r["P"]["M1 debut-class mean"], "raw MiLB, face value": raw, "raw MiLB, level-adjusted": adj}, dd["nm"].to_numpy())


def part_b():
    import fl_pitcher_features as fp
    import fl_unified_bridge as ub
    raw = fp.load_fl_pitches_cached()
    with contextlib.redirect_stdout(io.StringIO()):
        mdl = ub.fit(raw, verbose=False)
    p = mdl.pairs.copy().reset_index(drop=True); w = p["pa"].to_numpy(float); leagues = sorted(p["league"].unique()); I = np.column_stack([(p["league"] == lg).astype(float) for lg in leagues])

    def loo(X, y):
        XtWX = X.T @ (X * w[:, None]); beta = np.linalg.solve(XtWX, X.T @ (w * y)); e = y - X @ beta; h = w * np.einsum("ij,jk,ik->i", X, np.linalg.inv(XtWX), X)
        return y - e / (1 - h), (np.sum(w * y) - w * y) / (np.sum(w) - w)
    pk, bk = loo(np.column_stack([I, p["u_k"].to_numpy()]), p["k_rate"].to_numpy()); pb, bb = loo(np.column_stack([np.ones(len(p)), p["u_bb"].to_numpy()]), p["bb_rate"].to_numpy())
    lk = p.groupby("league").apply(lambda g: np.average(g["k_raw_latest"], weights=g["pa"])); lb = p.groupby("league").apply(lambda g: np.average(g["bb_raw_multi"], weights=g["pa"]))
    p = p.assign(pk=pk, bk=bk, pb=pb, bb=bb); p = p.replace([np.inf, -np.inf], np.nan).sort_values("pa", ascending=False).drop_duplicates("pitcher_id")
    ids = p["pitcher_id"].astype(str).to_numpy()
    score("B pitcher rookies (bridge)", "K%", p["k_rate"], p["pa"], p["pk"], {"league mean": p["bk"], "raw source, face value": p["k_raw_latest"], "raw source, level-adjusted": p["k_raw_latest"] - p["league"].map(lk) + np.average(p["k_rate"], weights=p["pa"])}, ids)
    score("B pitcher rookies (bridge)", "BB%", p["bb_rate"], p["pa"], p["pb"], {"league mean": p["bb"], "raw source, face value": p["bb_raw_multi"], "raw source, level-adjusted": p["bb_raw_multi"] - p["league"].map(lb) + np.average(p["bb_rate"], weights=p["pa"])}, ids)


def part_c():
    import fl_hitter_baseline_backtest as bt
    import fl_hitter_depth3_test as d3
    import fl_hitter_model_dev as dev
    import fl_hitter_projection as hp
    p = dev.load(); q = p.copy(); q["woba"] = q["woba_n"]
    for s, lab in (("woba", "wOBA"), ("k", "K%"), ("bbr", "BB%")):
        delta = hp.SPEC[s][0]; d = d3.frame(q, s); one = bt.transitions(q, s); parts = []
        for t in sorted(d["t"].unique()):
            tr, te = d[d["t"] != t], d[d["t"] == t]; n0 = d3.fit_n0(tr, delta, 3); te = te.assign(pred=d3.pred(te, n0, delta, 3))
            o_tr = one[one["t"] != t]; n1 = bt.fit_n0(o_tr) if hasattr(bt, "fit_n0") else 200.0; o = one[one["t"] == t][["name_clean", "t", "x_t", "n_t", "lg_t", "lg_next"]]
            o = o.assign(naive1=o["lg_next"] + (o["x_t"] - o["lg_t"]) * o["n_t"] / (o["n_t"] + n1))[["name_clean", "t", "naive1"]]
            parts.append(te.merge(o, on=["name_clean", "t"], how="left"))
        E = pd.concat(parts); E["hist"] = E["n_t"] + E["n_prev"] + E["n_pp"]
        nv = {"league mean": E["lg_next"], "last season, face value + league shift": E["x_t"] - E["lg_t"] + E["lg_next"], "one-season shrinkage": E["naive1"]}
        for g, m in (("C hitters, thin FL history (< 150 PA)", E["hist"] < 150), ("C hitters, regulars (>= 150 PA)", E["hist"] >= 150)):
            score(g, lab, E["x_next"], E["n_next"], E["pred"], nv, (E["name_clean"] + "|" + E["t"].astype(str)).to_numpy(), m.to_numpy())


def part_d():
    import fl_halfseason_backtest as hb
    import fl_pitcher_features as fp
    import fl_position_player_flag as ppf
    import fl_posplayer_refit_check as chk
    import fl_production_projection as pp
    raw = ppf.exclude(fp.load_fl_pitches_cached())
    with contextlib.redirect_stdout(io.StringIO()):
        m_k, m_bb, n0_k, n0_bb, n0b, rho = pp.fit_frozen_model(raw); dk, db = chk.fold2(raw, m_k, m_bb, n0b)
    win = pd.concat([fp.filter_period(raw, hb.START_2025, hb.END_2025), fp.filter_period(raw, hb.START_2026, hb.CUTOFF_2026)]); term = win[win["is_terminal"]]
    rr = term.groupby("pitcher_id").agg(k=("k_or_bb", lambda s: (s == "strikeout").mean()), b=("k_or_bb", lambda s: (s == "walk").mean()))
    had25 = set(fp.filter_period(raw, hb.START_2025, hb.END_2025)["pitcher_id"].astype(str))
    for d, ac, nv_col, rcol, lab in ((dk, "K_rate_actual", "naive_K_rate", "k", "K%"), (db, "BB_rate_actual", "naive_BB_rate", "b", "BB%")):
        d = d.dropna(subset=["pred", ac, nv_col]).copy(); d["raw"] = d["pitcher_id"].map(rr[rcol]); d = d.dropna(subset=["raw"])
        ids = d["pitcher_id"].astype(str).to_numpy(); nv = {"league mean": np.full(len(d), np.average(d[ac], weights=d["PA_test"])), "own rate, face value": d["raw"], "naive shrunk own rate": d[nv_col]}
        for g, m in (("D pitchers, thin (< 150 BF in window)", d["PA"] < 150), ("D pitchers, rookies (no 2025)", ~d["pitcher_id"].astype(str).isin(had25)), ("D pitchers, regulars (>= 150 BF)", d["PA"] >= 150)):
            score(g, lab, d[ac], d["PA_test"], d["pred"], nv, ids, m.to_numpy())


def part_e():
    import fl_pwar_calibration as cal
    cal.RETURN_ALL = True
    with contextlib.redirect_stdout(io.StringIO()):
        b, war = cal.main()
    b = b[(b["ip"] >= 10) & b["pred"].notna() & b["target"].notna()].copy()
    w25 = war[war["year"] == 2025].groupby("player").agg(ip25=("ip", "sum"), ra25=("ra", "sum")); b = b.merge(w25, left_on="player", right_index=True, how="left"); b["ip25"] = b["ip25"].fillna(0)
    lg = (war[war["year"] == 2025]["ra"].sum() * 9) / war[war["year"] == 2025]["ip"].sum(); own = np.where(b["ip25"] > 0, 9 * b["ra25"] / b["ip25"].replace(0, np.nan), lg)
    nv = {"league RA9": np.full(len(b), lg), "own 2025 RA9, face value": own, "own 2025 RA9 shrunk (100 IP)": (b["ip25"] * own + 100 * lg) / (b["ip25"] + 100)}
    for g, m in (("E pitcher RA9, thin (< 30 IP in 2025, incl. none)", b["ip25"] < 30), ("E pitcher RA9, no 2025 record", b["ip25"] == 0), ("E pitcher RA9, >= 30 IP in 2025", b["ip25"] >= 30)):
        score(g, "RA9", b["target"], b["ip"], b["pred"], nv, b["pitcher_id"].astype(str).to_numpy(), m.to_numpy())


def main():
    for f in (part_a, part_b, part_c, part_d, part_e):
        try:
            f(); print(f"{f.__name__} done", flush=True)
        except Exception as ex:
            import traceback; traceback.print_exc(); print(f"{f.__name__} FAILED: {ex}", flush=True)
    pd.set_option("display.width", 260); R = pd.DataFrame(ROWS); R.to_csv("output/lowhistory_baseline_test.csv", index=False)
    print("\nMODEL vs NAIVE BASELINES (level-free; dRMSE < 0 = model better; P = player-bootstrap probability the model is better)\n" + R.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
