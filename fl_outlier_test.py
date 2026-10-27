"""
fl_outlier_test.py -- multivariate OUTLIER detection on pitchers by pitch type: is it sticky, and does it predict performance beyond Stuff+?  (user 2026-10-27)
MEASURE: per pitch type (four-seam, sinker, cutter, slider, sweeper, curveball, changeup, splitter) a ROBUST (minimum covariance determinant, support .9) location and covariance of 7 standardised pitch characteristics (velocity, induced vertical break, arm-side horizontal break, spin rate,
extension, release height, release side; handedness-mirrored) fitted on a 10,000-pitch sample from OTHER pitchers (5 pitcher folds: each pitcher's pitches are scored against a distribution that never saw him).  Per-pitch Mahalanobis distance^2 -> log -> z-score against that type's own training distribution (median / MAD).
Pitcher indices (Mahalanobis: oi, ot; ISOLATION FOREST on the same standardised features, 300 trees, 512-pitch subsamples, anomaly score z-scored the same way: oi_if, ot_if): oi = mean z over ALL his pitches in the window (usage-weighted outlier-ness); ot = the largest pitch-type mean z among types with >= 30 of his pitches (his single most unusual pitch).  Distance is UNSIGNED: an unusual pitch can be unusual in a good or a bad way.
TESTS (pre-stated; the primary tests are P1-P3, the rest exploratory):
  S  stickiness: year-to-year r of oi and ot (indices from the 2025 window vs the 2026 window; pitchers with >= 150 / 300 / 500 pitches in each year), next to the Stuff+ (swstr_ct) level.
  P1 K%, four pairings (T0->T1b, T1->T2, T0->T2, T2->T0; fit on the first, scored on the second; frozen weights from the production model; level-free; RMSE and MAE; pitcher bootstrap 1,000): production additive K% (network swstr_ct as the stuff term) vs production + oi (controls for Stuff+) vs + ot vs + both vs oi INSTEAD of Stuff+.
  P2 BB%, same protocol: production blended BB% + command score vs + oi / + ot.
  P3 RA9 baseline (K%, BB%, innings per appearance; leave-one-pitcher-out, forward 2025 -> 2026 and reverse; n ~ 79): + oi / + ot.
  Also the correlation of oi / ot with the Stuff+ level and with the K% target.   Output: output/outlier_test.json
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest

import fl_eval_report as er
import fl_eval_weights as ew
import fl_halfseason_backtest as bt
import fl_pitcher_features as fp
import fl_position_player_flag as ppf
import fl_production_projection as pp
import fl_wh_sva_prior_xgb_robust_test as rb
import mlb_marcel as mm

warnings.filterwarnings("ignore")
FEATS = ["effective_velo", "ivb", "horz_break_as", "spin_rate", "extension", "release_pos_z", "rel_side_as"]; TYPES = ["four_seam", "sinker", "cutter", "slider", "sweeper", "curveball", "changeup", "splitter"]
ADD = pp.SPEC["K"]; BBS = pp.BB_SPEC_BLENDED; K = 5; _lv = {}; _fr = {}
SEASONS = {2025: (bt.START_2025, bt.END_2025), 2026: (bt.START_2026, bt.END_2026)}


def outlier_levels(df, seed=0):
    """pitcher-level outlier indices (oi, ot) for a window of pitches, scored against distributions fitted on other pitchers only."""
    d = df[df["pitch_type"].isin(TYPES)].dropna(subset=FEATS + ["pitcher_id"]).copy(); d["pid"] = d["pitcher_id"].astype(str); up = np.unique(d["pid"]); rng = np.random.default_rng(seed)
    fold = dict(zip(up, rng.permutation(len(up)) % K)); d["fold"] = d["pid"].map(fold); d["z"] = np.nan; d["zi"] = np.nan
    for t in TYPES:
        m = d["pitch_type"] == t
        if m.sum() < 1500:
            continue
        sub = d[m]
        for k in range(K):
            tr = sub[sub["fold"] != k]; te = sub[sub["fold"] == k]
            if len(tr) < 800 or len(te) == 0:
                continue
            samp = tr.sample(min(len(tr), 10000), random_state=seed)[FEATS]; mu, sd = samp.mean(), samp.std().replace(0, 1.0); Xs = (samp - mu) / sd
            mcd = MinCovDet(support_fraction=0.9, random_state=seed).fit(Xs); ltr = np.log(mcd.mahalanobis(Xs) + 1e-9); med = np.median(ltr); mad = 1.4826 * np.median(np.abs(ltr - med))
            Xte = (te[FEATS] - mu) / sd; d.loc[te.index, "z"] = (np.log(mcd.mahalanobis(Xte) + 1e-9) - med) / mad
            iso = IsolationForest(n_estimators=300, max_samples=512, random_state=seed).fit(Xs); atr = -iso.score_samples(Xs); amed = np.median(atr); amad = 1.4826 * np.median(np.abs(atr - amed))
            d.loc[te.index, "zi"] = (-iso.score_samples(Xte) - amed) / amad                      # isolation-forest anomaly (higher = easier to isolate = more unusual), z-scored against the type's training distribution
    d = d.dropna(subset=["z", "zi"]); n = d.groupby("pid").size().rename("n_pitches"); out = [n]
    for col, a, b in (("z", "oi", "ot"), ("zi", "oi_if", "ot_if")):
        oi = d.groupby("pid")[col].mean().rename(a); cell = d.groupby(["pid", "pitch_type"])[col].agg(["mean", "size"]).reset_index(); ot = cell[cell["size"] >= 30].groupby("pid")["mean"].max().rename(b); out += [oi, ot]
    return pd.concat(out, axis=1)


def window_levels(raw, windows):
    key = tuple(windows)
    if key not in _lv:
        _lv[key] = outlier_levels(pd.concat([fp.filter_period(raw, a, b) for a, b in windows], ignore_index=True))
    return _lv[key]


def frame(raw, name):
    if name not in _fr:
        fp.WH_SVA_ESTIMATOR = "swstr_ct"; feat, lab = rb.TR[name]; df, n0_k, n0_bb = bt.assemble(raw, feat, lab, with_changepoints=False); df = df.copy(); df["pid"] = df["pitcher_id"].astype(str)
        lv = window_levels(raw, feat); df = df.merge(lv[["oi", "ot", "oi_if", "ot_if"]], left_on="pid", right_index=True, how="left"); _fr[name] = (df, n0_k, n0_bb, rb.pk.league_k(raw, feat), _lg_bb(raw, feat))
    return _fr[name]


def _lg_bb(raw, windows):
    t = pd.concat([fp.filter_period(raw, a, b) for a, b in windows], ignore_index=True); t = t[t["is_terminal"]]; return float((t["k_or_bb"] == "walk").mean())


def stickiness(raw):
    L = {yr: window_levels(raw, [w]) for yr, w in SEASONS.items()}; n = {yr: fp.filter_period(raw, *SEASONS[yr]).groupby("pitcher_id").size() for yr in SEASONS}
    for yr in n:
        n[yr].index = n[yr].index.astype(str)
    fp.WH_SVA_ESTIMATOR = "swstr_ct"; S = {yr: fp.compute_pooled_wh_sva(fp.filter_period(raw, *SEASONS[yr]), estimator="swstr_ct").groupby("pitcher_id")["sva"].mean() for yr in SEASONS}
    for yr in S:
        S[yr].index = S[yr].index.astype(str)
    rows = []
    for lab, get in (("oi (Mahalanobis, mean z)", lambda yr: L[yr]["oi"]), ("ot (Mahalanobis, most unusual type)", lambda yr: L[yr]["ot"]), ("oi_if (isolation forest, mean z)", lambda yr: L[yr]["oi_if"]), ("ot_if (isolation forest, most unusual type)", lambda yr: L[yr]["ot_if"]), ("Stuff+ level (swstr_ct, separate fits)", lambda yr: S[yr])):
        r = {"measure": lab}
        for thr in (150, 300, 500):
            j = pd.concat([get(2025).rename("a"), get(2026).rename("b"), n[2025].rename("n25"), n[2026].rename("n26")], axis=1).dropna(); j = j[(j["n25"] >= thr) & (j["n26"] >= thr)]
            rr = float(np.corrcoef(j["a"], j["b"])[0, 1]); z = np.arctanh(rr); se = 1 / np.sqrt(len(j) - 3); r[f"r>={thr}"] = rr; r[f"CI{thr}"] = f"[{np.tanh(z - 1.96 * se):.2f}, {np.tanh(z + 1.96 * se):.2f}]"; r[f"n{thr}"] = len(j)
        rows.append(r)
    pd.set_option("display.width", 260); print("\nS. STICKINESS (year-to-year r, 2025 window vs 2026 window)\n" + pd.DataFrame(rows).round(3).to_string(index=False), flush=True)
    j = pd.concat([L[2026][["oi", "ot", "oi_if", "ot_if"]], S[2026].rename("stuff")], axis=1).dropna(); print("   2026 window, n %d: corr with the Stuff+ level: oi %.2f, ot %.2f, oi_if %.2f, ot_if %.2f;  oi vs oi_if %.2f, ot vs ot_if %.2f" % (len(j), j["oi"].corr(j["stuff"]), j["ot"].corr(j["stuff"]), j["oi_if"].corr(j["stuff"]), j["ot_if"].corr(j["stuff"]), j["oi"].corr(j["oi_if"]), j["ot"].corr(j["ot_if"])), flush=True)
    return rows, L


def compare(rows, pairing, stat, y, n_t, ids, P, base="base"):
    w = ew.weights_from_base(y, P[base], n_t)[0]; cen = lambda q: q + np.average(y - q, weights=w)
    for lab, q in P.items():
        rm, ma = er.wmetrics(y, cen(q), w); r = {"stat": stat, "pairing": pairing, "model": lab, "n": len(y), "RMSE": rm, "MAE": ma}
        if lab != base:
            c = er.compare(y, cen(P[base]), cen(q), w, ids, n=1000); r.update({"dRMSE": c["dRMSE"], "90%": c["RMSE 90%"], "P(RMSE)": c["P(RMSE better)"], "dMAE": c["dMAE"], "P(MAE)": c["P(MAE better)"]})
        rows.append(r)


def kb_tests(raw):
    rows = []; info = []
    for a, b in rb.PAIRS:
        da, n0k, n0bb, lgk_a, lgbb_a = frame(raw, a); db_, _, _, lgk_b, lgbb_b = frame(raw, b)
        # ---- K%
        dk = da.dropna(subset=["K_rate_actual"] + ADD + ["oi", "ot", "oi_if", "ot_if"]).copy(); wk = mm.reliability_weight(dk["PA_test"], n0k).values; tk = db_.dropna(subset=["K_rate_actual"] + ADD + ["oi", "ot", "oi_if", "ot_if"]).copy(); base_k = [f for f in ADD if f != "wh_sva_level"]
        specs = {"base": ADD, "+ oi": ADD + ["oi"], "+ ot": ADD + ["ot"], "+ oi + ot": ADD + ["oi", "ot"], "+ oi_if": ADD + ["oi_if"], "+ ot_if": ADD + ["ot_if"], "oi instead of Stuff+": base_k + ["oi"], "oi_if instead of Stuff+": base_k + ["oi_if"]}; P = {}; coef = {}
        for lab, spec in specs.items():
            m = bt.fit_weighted(dk, "K_rate_actual", spec, wk); P[lab] = m.predict(tk).to_numpy(); coef[lab] = {k: (round(float(m.params[k]), 4), round(float(m.tvalues[k]), 2)) for k in ("oi", "ot", "oi_if", "ot_if") if k in spec}
        compare(rows, f"{a}->{b}", "K%", tk["K_rate_actual"].to_numpy(), tk["PA_test"].to_numpy(), tk["pitcher_id"].to_numpy(), P); info.append({"pairing": f"{a}->{b}", "stat": "K%", "coef (b, t)": coef, "corr(oi, Stuff+) in the test frame": float(tk["oi"].corr(tk["wh_sva_level"])), "corr(oi, K% target)": float(tk["oi"].corr(tk["K_rate_actual"]))})
        # ---- BB%
        need = ["BB_rate_actual", "naive_BB_rate", "fl_bb_rate_bridge", "PA", "command_score", "oi", "ot", "oi_if", "ot_if"]; dbb = da.dropna(subset=need).copy(); wbb = mm.reliability_weight(dbb["PA_test"], n0bb).values
        dbb["BB_raw"] = dbb["naive_BB_rate"] * (dbb["PA"] + n0bb) - n0bb * lgbb_a; n0b = mm.fit_n0_bridge(dbb["BB_raw"], dbb["PA"], n0bb, lgbb_a, dbb["fl_bb_rate_bridge"], dbb["BB_rate_actual"], wbb); dbb["blended_BB_rate"] = mm.blended_rate(dbb["BB_raw"], dbb["PA"], n0bb, lgbb_a, n0b, dbb["fl_bb_rate_bridge"])
        tb = db_.dropna(subset=need).copy(); tb["BB_raw"] = tb["naive_BB_rate"] * (tb["PA"] + n0bb) - n0bb * lgbb_b; tb["blended_BB_rate"] = mm.blended_rate(tb["BB_raw"], tb["PA"], n0bb, lgbb_b, n0b, tb["fl_bb_rate_bridge"])
        P = {}; coef = {}
        for lab, spec in {"base": BBS, "+ oi": BBS + ["oi"], "+ ot": BBS + ["ot"], "+ oi_if": BBS + ["oi_if"], "+ ot_if": BBS + ["ot_if"]}.items():
            m = bt.fit_weighted(dbb, "BB_rate_actual", spec, wbb); P[lab] = m.predict(tb).to_numpy(); coef[lab] = {k: (round(float(m.params[k]), 4), round(float(m.tvalues[k]), 2)) for k in ("oi", "ot", "oi_if", "ot_if") if k in spec}
        compare(rows, f"{a}->{b}", "BB%", tb["BB_rate_actual"].to_numpy(), tb["PA_test"].to_numpy(), tb["pitcher_id"].to_numpy(), P); info.append({"pairing": f"{a}->{b}", "stat": "BB%", "coef (b, t)": coef})
        print(f"  {a}->{b} done", flush=True)
    T = pd.DataFrame(rows); pd.set_option("display.width", 280)
    for stat in ("K%", "BB%"):
        t = T[T["stat"] == stat]; print(f"\nP{'1' if stat == 'K%' else '2'}. {stat}: four pairings (RMSE and MAE; change vs the production model, P = bootstrap P(better))\n" + t.round(5).to_string(index=False))
        g = t.groupby("model").agg(RMSE=("RMSE", "mean"), MAE=("MAE", "mean"), dRMSE=("dRMSE", "mean"), dMAE=("dMAE", "mean")).sort_values("RMSE"); print(f"\n   {stat} mean over the four pairings\n" + g.round(5).to_string())
    print("\nCOEFFICIENTS (b, t) and overlap with Stuff+\n" + json.dumps(info, indent=1)[:3500]); return rows, info


def ra9_test(raw, L):
    from sqlalchemy import text

    import fl_ra9_trustmark_projection as ra9
    from fl_supabase import read_sql_df
    idmap = raw[["pitcher", "pitcher_id"]].drop_duplicates(); idmap["pitcher_id"] = idmap["pitcher_id"].astype(str); uniq = idmap.groupby("pitcher")["pitcher_id"].nunique(); idmap = idmap[idmap["pitcher"].isin(uniq[uniq == 1].index)].drop_duplicates("pitcher").set_index("pitcher")["pitcher_id"]
    war = read_sql_df(text("select player, year, bf, ip, so, bb, ra, ra9, g from war_pitchers where game_type='reg_season'")); rows = []
    for design, fy, ty in (("forward 2025 -> 2026", 2025, 2026), ("reverse 2026 -> 2025", 2026, 2025)):
        m = war[war["year"] == fy].merge(war[war["year"] == ty], on="player", suffixes=("_f", "_t")); m = m[(m["bf_f"] >= 100) & (m["bf_t"] >= 100)].copy()
        m["K_pred"] = m["so_f"] / m["bf_f"]; m["BB_pred"] = m["bb_f"] / m["bf_f"]; m["ip_per_g_prior"] = (m["ip_f"] + ra9.N0_IPG * ra9.LG_IPG) / (m["g_f"] + ra9.N0_IPG); m["target"] = m["ra9_t"]; m["pid"] = m["player"].map(idmap)
        for meas in ("oi", "ot", "oi_if", "ot_if"):
            d = m.merge(L[fy][[meas]].rename(columns={meas: "lvl"}), left_on="pid", right_index=True, how="inner").dropna(subset=["lvl", "target"]).copy(); d["lvl"] = (d["lvl"] - d["lvl"].mean()) / d["lvl"].std()
            y = d["target"].to_numpy(float); w = mm.reliability_weight(d["ip_t"], ra9.RA9_N0_EVAL).values; X0 = np.column_stack([np.ones(len(d)), d[["K_pred", "BB_pred", "ip_per_g_prior"]].to_numpy(float)]); X1 = np.column_stack([X0, d["lvl"].to_numpy(float)])
            def lo(X):
                A = X.T @ (X * w[:, None]); beta = np.linalg.solve(A, X.T @ (w * y)); e = y - X @ beta; h = w * np.einsum("ij,jk,ik->i", X, np.linalg.inv(A), X); return y - e / (1 - h)
            p0, p1 = lo(X0), lo(X1); wt = ew.weights_from_base(y, p0, d["ip_t"].to_numpy(float))[0]; cen = lambda q: q + np.average(y - q, weights=wt); c = er.compare(y, cen(p0), cen(p1), wt, d["pid"].to_numpy(), n=1000)
            A = X1.T @ (X1 * w[:, None]); Ai = np.linalg.inv(A); beta = Ai @ X1.T @ (w * y); e = y - X1 @ beta; t = beta[-1] / np.sqrt(Ai[-1, -1] * np.sum(w * e ** 2) / (len(y) - X1.shape[1]))
            rows.append({"design": design, "measure": meas, "n": len(d), "RMSE base": c["RMSE old"], "RMSE +idx": c["RMSE new"], "dRMSE": c["dRMSE"], "P(better)": c["P(RMSE better)"], "dMAE": c["dMAE"], "coef per SD (runs/9)": float(beta[-1]), "t": float(t)})
    print("\nP3. RA9 baseline + outlier index (leave-one-pitcher-out)\n" + pd.DataFrame(rows).round(4).to_string(index=False), flush=True); return rows


def main():
    raw = ppf.exclude(fp.load_fl_pitches_cached()); print("computing outlier indices and the stuff level ...", flush=True)
    srows, L = stickiness(raw); krows, info = kb_tests(raw); rrows = ra9_test(raw, L)
    json.dump({"stickiness": srows, "k_bb": krows, "info": info, "ra9": rrows}, open("output/outlier_test.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
