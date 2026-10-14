"""
fl_recent_changes_report.py -- weighted RMSE AND MAE (old vs new, frozen standard weights, cluster bootstrap) for the model changes made on 2026-10-08, to bring them to the reporting standard set the same day.
Changes: (1) third history season in the hitter preseason regression; (2) Swing% term in the hitter preseason BB% (and its effect on wOBA); (3) hitter ROS BB% in-season weight nu = .4;
(4) pitcher K% / BB% refit without position players; (5) pitcher ROS: published preseason projection as the prior (+ halved stuff / command terms).
"""
import warnings

import numpy as np
import pandas as pd

import fl_eval_report as er
import fl_eval_weights as ew

warnings.filterwarnings("ignore")


def change1():
    import fl_hitter_depth3_test as d3
    import fl_hitter_model_dev as dev
    import fl_hitter_projection as hp
    p = dev.load(); q = p.copy(); q["woba"] = q["woba_n"]; fit = hp.fit_all(p); stats = {s: d3.frame(q, s) for s in hp.SPEC if s != "hbpr"}; rows = []
    plan = [((2022, 2023), 2024), ((2022, 2023, 2024), 2025)]
    for s, d in stats.items():
        delta = hp.SPEC[s][0]; P = {2: [], 3: []}; Y = []
        for tr, te in plan:
            dtr = d[d["t"].isin(tr)]; dte = d[d["t"] == te].copy()
            for depth in (2, 3):
                P[depth].append(d3.pred(dte, d3.fit_n0(dtr, delta, depth), delta, depth).to_numpy())
            Y.append(dte)
        D = pd.concat(Y); rows.append({"stat": s, **er.compare(D["x_next"], np.concatenate(P[2]), np.concatenate(P[3]), D["w"], D["name_clean"])})
    res = {}
    for depth in (2, 3):
        pr = {}
        for s, d in stats.items():
            n0 = d3.fit_n0(d[d["t"].isin((2022, 2023, 2024))], hp.SPEC[s][0], depth); dte = d[d["t"] == 2025]; pr[s] = pd.Series(d3.pred(dte, n0, hp.SPEC[s][0], depth).to_numpy(), index=dte["name_clean"].to_numpy())
        idx = pr["woba"].index; lgn = {s: stats[s][stats[s]["t"] == 2025]["lg_next"].iloc[0] for s in stats}
        comp = sum(fit["comp_beta"][s] * (pr[s].reindex(idx) - lgn[s]) for s in ("k", "bbr", "hrr", "babip")).fillna(0.0); res[depth] = 0.5 * pr["woba"] + 0.5 * (lgn["woba"] + comp)
    dte = stats["woba"][stats["woba"]["t"] == 2025].set_index("name_clean"); ok = dte["x_next"].reindex(res[2].index).notna().to_numpy()
    rows.append({"stat": "wOBA production blend (2025->26)", **er.compare(dte["x_next"].reindex(res[2].index).to_numpy()[ok], res[2].to_numpy()[ok], res[3].to_numpy()[ok], dte["w"].reindex(res[2].index).to_numpy()[ok], res[2].index.to_numpy()[ok])})
    print("\n(1) THIRD HISTORY SEASON, hitter preseason (forward folds 2024 and 2025 for the per-stat rows; old = 2 seasons, new = 3):\n" + er.table(rows), flush=True)
    return rows


def change2():
    import fl_hitter_baseline_backtest as bt
    import fl_hitter_model_dev as dev
    import fl_hitter_projection as hp
    import fl_hitter_swing_preseason_check as sc
    p = dev.load(); q = p.copy(); q["woba"] = q["woba_n"]; fit = hp.fit_all(p); dev_sw = sc.pitches25(); preds = {}
    for s, (delta, _) in hp.SPEC.items():
        d = bt.transitions(q, s); d = d[d["t"] == 2025].copy(); d["pred"] = bt.predict(d, fit["n0"][s], delta); preds[s] = d.set_index("name_clean")
    bb = preds["bbr"]; term = sc.B * dev_sw.reindex(bb.index).fillna(0.0) * fit["n0"]["bbr"] / (bb["n_t"] + 0.5 * bb["n_prev"] + fit["n0"]["bbr"])
    y = bb["x_next"].to_numpy(); w = bb["w"].to_numpy(); fair = lambda a: a + np.average(y - a, weights=w)
    rows = [{"stat": "BB% (term added; both with a fair intercept)", **er.compare(y, fair(bb["pred"].to_numpy()), fair((bb["pred"] + term).to_numpy()), w, bb.index.to_numpy())}]
    idx = bb.index; comp = fit["comp_beta"]; lgn = {s: preds[s]["lg_next"].iloc[0] for s in preds}
    cd = lambda b: (comp["k"] * (preds["k"]["pred"].reindex(idx) - lgn["k"]) + comp["bbr"] * (b - lgn["bbr"]) + comp["hrr"] * (preds["hrr"]["pred"].reindex(idx) - lgn["hrr"]) + comp["babip"] * (preds["babip"]["pred"].reindex(idx) - lgn["babip"])).fillna(0.0)
    wo = preds["woba"].reindex(idx); ok = wo["x_next"].notna().to_numpy(); yw = wo["x_next"].to_numpy()[ok]; ww = wo["w"].to_numpy()[ok]
    b0 = (0.5 * wo["pred"] + 0.5 * (wo["lg_next"].iloc[0] + cd(bb["pred"]))).to_numpy()[ok]; b1 = (0.5 * wo["pred"] + 0.5 * (wo["lg_next"].iloc[0] + cd(bb["pred"] + term))).to_numpy()[ok]
    f2 = lambda a: a + np.average(yw - a, weights=ww)
    rows.append({"stat": "wOBA blend (term feeding the components)", **er.compare(yw, f2(b0), f2(b1), ww, idx.to_numpy()[ok])})
    print("\n(2) SWING% TERM in the hitter preseason BB% (forward 2025->26, old = without the term):\n" + er.table(rows), flush=True)
    return rows


def change3():
    import fl_hitter_ros as hr
    import fl_ros_by_day as bd
    T = hr.Tables(); rows = []
    for yr in (2026, 2025):
        t0, _ = T.season_bounds(yr); Y, O, N, W, I = [], [], [], [], []
        for d in bd.DAYS:
            f, lg = hr.features(T, yr, t0 + pd.Timedelta(days=d), min_ros_pa=30)
            if len(f) < 40:
                continue
            Y.append(f["bb_ros"].to_numpy()); O.append(hr.predict(f, lg, nu=1.0)["bbr_final"].to_numpy()); N.append(hr.predict(f, lg)["bbr_final"].to_numpy()); W.append(f["n_ros"].to_numpy()); I.append(f.index.to_numpy())
        y, o, n, nr, ids = map(np.concatenate, (Y, O, N, W, I)); w = ew.weights_from_base(y, o, nr)[0]
        rows.append({"stat": f"BB% ROS, scored on {yr} ({'forward: nu chosen on 2025' if yr == 2026 else 'reverse check'})", **er.compare(y, o, n, w, ids)})
    print("\n(3) HITTER ROS BB% in-season weight nu = .4 (old nu = 1; as-of days 7-105):\n" + er.table(rows), flush=True)
    return rows


def change4():
    import fl_pitcher_features as fp
    import fl_position_player_flag as ppf
    import fl_posplayer_refit_check as chk
    import fl_production_projection as pp
    raw = fp.load_fl_pitches_cached(); rawx = ppf.exclude(raw); fits = {"A": pp.fit_frozen_model(raw), "B": pp.fit_frozen_model(rawx)}; res = {k: chk.fold2(rawx, f[0], f[1], f[4]) for k, f in fits.items()}; rows = []
    for stat, idx, tcol in (("K%", 0, "K_rate_actual"), ("BB%", 1, "BB_rate_actual")):
        a, b = res["A"][idx], res["B"][idx]; m = a.merge(b[["pitcher_id", "pred"]], on="pitcher_id", suffixes=("_a", "_b")); y = m[tcol].to_numpy(); w = ew.weights_from_base(y, m["pred_a"].to_numpy(), m["PA_test"].to_numpy())[0]
        c = lambda p_: p_ + np.average(y - p_, weights=w)
        rows.append({"stat": stat + " (fold 2: 2H2026 targets; level-free)", **er.compare(y, c(m["pred_a"].to_numpy()), c(m["pred_b"].to_numpy()), w, m["pitcher_id"].to_numpy())})
    print("\n(4) PITCHER K% / BB% REFIT WITHOUT POSITION PLAYERS (old = fit with them; both scored on the same non-flagged pitchers):\n" + er.table(rows), flush=True)
    return rows


def change5():
    import fl_pitcher_features as fp
    import fl_pitcher_ros as pr
    import fl_pitcher_ros_stuff as pst
    import fl_ros_production as rp
    import fl_ros_published_prior as pub
    T = pr.Tables(); consts = pr.fit_constants(2025); raw = fp.load_fl_pitches_cached(); names = pst.idmap(raw); n0s = pst.reliability(raw)
    pubt = pd.read_parquet("output/pitcher_pre2026_published.parquet").dropna(subset=["name_clean"]).drop_duplicates("name_clean").set_index("name_clean"); B = rp.STUFF_B["clean"]; t0, _ = T.season_bounds(2026); acc = {"k": [], "bb": []}
    for d in pub.DAYS:
        asof = t0 + pd.Timedelta(days=d); f, lg = pr.features(T, 2026, asof)
        if len(f) < 40:
            continue
        base = pr.predict(f, lg, consts); hist = pr.predict(f, lg, consts, use_ytd=False); sf = pst.stuff_features(raw, 2026, asof, names, n0s).reindex(f.index).fillna(0.0)
        for stat, key, pcol, ycol, bi, scol in (("k", "k", "pre_k", "k_ros", 0, "wh_sva"), ("bb", "bbr", "pre_bb", "bb_ros", 1, "command_score")):
            n0 = consts[key]["n0"]; n = f["n"].to_numpy(); x_ytd = np.where(n > 0, (f["k"] if stat == "k" else f["bb"]).to_numpy() / np.where(n > 0, n, 1), 0.0)
            m0 = pubt[pcol].reindex(f.index); has = m0.notna().to_numpy(); dev0 = (m0 - np.average(m0[has], weights=np.maximum(f["n"][has], 1))).fillna(0.0).to_numpy(); n_p = hist[f"pa_eff_{key}"].to_numpy() + n0
            est = np.where(has, lg[key] + (n_p * dev0 + n * (x_ytd - lg[key])) / (n_p + n), base[key].to_numpy())
            old = base[key].to_numpy() + B[bi] * sf[scol].to_numpy() * n0 / (base[f"pa_eff_{key}"].to_numpy() + n0)
            new = est + 0.5 * B[bi] * sf[scol].to_numpy() * n0 / (hist[f"pa_eff_{key}"].to_numpy() + n + n0)
            acc[stat].append(pd.DataFrame({"y": f[ycol].to_numpy(), "n": f["n_ros"].to_numpy(), "id": f.index.to_numpy(), "old": old, "new": new}))
    rows = []
    for stat, lab in (("k", "K%"), ("bb", "BB%")):
        d = pd.concat(acc[stat]); w = ew.weights_from_base(d["y"].to_numpy(), d["old"].to_numpy(), d["n"].to_numpy())[0]; rows.append({"stat": lab + " pitcher ROS (2026 days 63-105)", **er.compare(d["y"], d["old"], d["new"], w, d["id"])})
    print("\n(5) PITCHER ROS with the published preseason projection as the prior (+ stuff / command terms x .5), old = Marcel history + full terms:\n" + er.table(rows), flush=True)
    return rows


if __name__ == "__main__":
    for f in (change1, change2, change3, change4, change5):
        f()
