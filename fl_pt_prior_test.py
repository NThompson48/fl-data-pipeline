"""
fl_pt_prior_test.py -- should the prior LEVEL depend on playing time?  (user 2026-10-13: "in which places is a different (probably lower) level better? The players with the most
playing time are usually the best and can bias the average")
League means used as priors are playing-time weighted, so they sit near the regulars; thin-history players may be worse as a group.  One extra parameter per model:
  HITTERS (wOBA, K%, BB%, HR/PA; three-season regression, transitions 2022->23 ... 2025->26, next-season PA >= 20):
      production  pred = lg + (num) / (nh + n0)
      PT prior    pred = lg + (num + n0 x gamma x z) / (nh + n0),  z = log(1 + nh) - its training mean  (the prior mean moves with his history weight nh)
      n0 and gamma fit on the training transitions only (leave-one-transition-out).
  PITCHERS K% / BB%: the frozen production model (fit full 2025 -> 1H2026) refit with + z (z = log(1 + feature-window BF), centred), scored on fold 2 (2025 + 1H2026 -> 2H2026).
  PITCHER RA9 2025 -> 2026: production RA9 + gamma x z (z = log(1 + 2025 BF), centred), gamma cross-fitted 10-fold by pitcher.
Two weightings, both reported: PRODUCTION (standard inverse-variance weights -- grows with the target's playing time) and PER PLAYER (equal weights; gamma refit with equal weights).
Level-free (each prediction's own weighted mean error removed), player bootstrap; weighted RMSE and MAE; by history group.
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


def report(model, stat, y, p0, p1, w, ids, hist, extra=""):
    for grp, m in (("all", np.ones(len(y), bool)), ("history < 150", hist < 150), ("150-399", (hist >= 150) & (hist < 400)), ("400+", hist >= 400)):
        if m.sum() < 15:
            continue
        c = lambda p: p[m] + np.average(y[m] - p[m], weights=w[m])
        r = er.compare(y[m], c(p0), c(p1), w[m], ids[m], n=800)
        ROWS.append({"model": model, "stat": stat, "weighting": extra, "group": grp, "n": int(m.sum()), "RMSE prod": r["RMSE old"], "RMSE PT prior": r["RMSE new"], "dRMSE": r["dRMSE"],
                     "RMSE 90%": r["RMSE 90%"], "P(PT better)": r["P(RMSE better)"], "MAE prod": r["MAE old"], "MAE PT prior": r["MAE new"], "dMAE": r["dMAE"], "P(MAE better)": r["P(MAE better)"]})


def hitters():
    import fl_hitter_baseline_backtest as bt
    import fl_hitter_depth3_test as d3
    import fl_hitter_model_dev as dev
    import fl_hitter_projection as hp
    p = dev.load(); q = p.copy(); q["woba"] = q["woba_n"]
    for s, lab in (("woba", "wOBA"), ("k", "K%"), ("bbr", "BB%"), ("hrr", "HR/PA")):
        saved = bt.MIN_NEXT; bt.MIN_NEXT = 20; d = d3.frame(q, s); bt.MIN_NEXT = saved
        delta = hp.SPEC[s][0]
        dev_t = d["x_t"] - d["lg_t"]; dev_p = (d["x_prev"] - d["lg_prev"].fillna(d["lg_t"])).fillna(0.0); dev_pp = (d["x_pp"] - d["lg_pp"].fillna(d["lg_t"])).fillna(0.0)
        d["num"] = d["n_t"] * dev_t + delta * d["n_prev"] * dev_p + delta ** 2 * d["n_pp"] * dev_pp; d["nh"] = d["n_t"] + delta * d["n_prev"] + delta ** 2 * d["n_pp"]
        d["lz"] = np.log1p(d["nh"]); y = d["x_next"].to_numpy(); ids = (d["name_clean"] + "|" + d["t"].astype(str)).to_numpy(); hist = (d["n_t"] + d["n_prev"] + d["n_pp"]).to_numpy()
        for wlab in ("production", "per player"):
            p0 = np.zeros(len(d)); p1 = np.zeros(len(d)); gam = []
            for t in sorted(d["t"].unique()):
                tr, te = (d["t"] != t).to_numpy(), (d["t"] == t).to_numpy(); n0 = d3.fit_n0(d[tr], delta, 3)
                base = d["lg_next"] + d["num"] / (d["nh"] + n0); zc = d["lz"] - np.average(d.loc[tr, "lz"], weights=d.loc[tr, "w"] if wlab == "production" else None)
                x = (n0 * zc / (d["nh"] + n0)).to_numpy(); e = (y - base.to_numpy()); ww = d["w"].to_numpy() if wlab == "production" else np.ones(len(d))
                # gamma with a free level in the training fit (the scoring is level-free too)
                A = np.column_stack([np.ones(tr.sum()), x[tr]]); g = np.linalg.lstsq(A * np.sqrt(ww[tr])[:, None], e[tr] * np.sqrt(ww[tr]), rcond=None)[0][1]; gam.append(g)
                p0[te] = base.to_numpy()[te]; p1[te] = base.to_numpy()[te] + g * x[te]
            w = d["w"].to_numpy() if wlab == "production" else np.ones(len(d))
            report("hitters (history regression)", lab, y, p0, p1, w, ids, hist, wlab)
            print(f"hitters {lab:6s} {wlab:10s}: gamma by fold {np.round(gam, 4).tolist()} (prior shift per unit of log history; > 0 = thin players' prior lower)", flush=True)


def pitchers():
    import fl_halfseason_backtest as hb
    import fl_pitcher_features as fp
    import fl_position_player_flag as ppf
    import fl_production_projection as pp
    import mlb_marcel as mm
    raw = ppf.exclude(fp.load_fl_pitches_cached())
    with contextlib.redirect_stdout(io.StringIO()):
        df_tr, n0_k, n0_bb = hb.assemble(raw, [(hb.START_2025, hb.END_2025)], [(hb.START_2026, hb.CUTOFF_2026)], with_changepoints=False)
        df_te, _, _ = hb.assemble(raw, [(hb.START_2025, hb.END_2025), (hb.START_2026, hb.CUTOFF_2026)], [(hb._DAY_AFTER_2026_CUT, hb.END_2026)], with_changepoints=False)
    for d in (df_tr, df_te):
        d["lz"] = np.log1p(d["PA"].fillna(0))
    mu = float(df_tr["lz"].mean())
    for d in (df_tr, df_te):
        d["z"] = d["lz"] - mu
    t26 = fp.filter_period(raw, hb.START_2025, hb.END_2025); had25 = set(t26["pitcher_id"].astype(str))
    # K%
    tr = df_tr.dropna(subset=["K_rate_actual"] + pp.SPEC["K"]); te = df_te.dropna(subset=["K_rate_actual"] + pp.SPEC["K"]).copy()
    for wlab in ("production", "per player"):
        wt = mm.reliability_weight(tr["PA_test"], n0_k).values if wlab == "production" else np.ones(len(tr))
        m0 = hb.fit_weighted(tr, "K_rate_actual", pp.SPEC["K"], wt); m1 = hb.fit_weighted(tr, "K_rate_actual", pp.SPEC["K"] + ["z"], wt)
        y = te["K_rate_actual"].to_numpy(); p0 = m0.predict(te).to_numpy(); p1 = m1.predict(te).to_numpy()
        w = ew.weights_from_base(y, p0, te["PA_test"].to_numpy())[0] if wlab == "production" else np.ones(len(te))
        print(f"pitchers K% {wlab}: z coefficient {m1.params['z']:+.4f} (t {m1.tvalues['z']:+.2f})", flush=True)
        report("pitchers (frozen K% model)", "K%", y, p0, p1, w, te["pitcher_id"].astype(str).to_numpy(), te["PA"].to_numpy(float), wlab)
    # BB% (blended prior as in production)
    def prep(d, lg):
        d = d.dropna(subset=["BB_rate_actual", "naive_BB_rate", "fl_bb_rate_bridge", "PA", "command_score"]).copy(); d["BB_raw"] = d["naive_BB_rate"] * (d["PA"] + n0_bb) - n0_bb * lg; return d
    lg_tr = (fp.filter_period(raw, hb.START_2025, hb.END_2025).query("is_terminal")["k_or_bb"] == "walk").mean()
    pool = pd.concat([fp.filter_period(raw, hb.START_2025, hb.END_2025), fp.filter_period(raw, hb.START_2026, hb.CUTOFF_2026)]); lg_te = (pool[pool["is_terminal"]]["k_or_bb"] == "walk").mean()
    trb, teb = prep(df_tr, lg_tr), prep(df_te, lg_te)
    w0 = mm.reliability_weight(trb["PA_test"], n0_bb).values; n0b = mm.fit_n0_bridge(trb["BB_raw"], trb["PA"], n0_bb, lg_tr, trb["fl_bb_rate_bridge"], trb["BB_rate_actual"], w0)
    trb["blended_BB_rate"] = mm.blended_rate(trb["BB_raw"], trb["PA"], n0_bb, lg_tr, n0b, trb["fl_bb_rate_bridge"]); teb["blended_BB_rate"] = mm.blended_rate(teb["BB_raw"], teb["PA"], n0_bb, lg_te, n0b, teb["fl_bb_rate_bridge"])
    for wlab in ("production", "per player"):
        wt = w0 if wlab == "production" else np.ones(len(trb))
        m0 = hb.fit_weighted(trb, "BB_rate_actual", pp.BB_SPEC_BLENDED, wt); m1 = hb.fit_weighted(trb, "BB_rate_actual", pp.BB_SPEC_BLENDED + ["z"], wt)
        y = teb["BB_rate_actual"].to_numpy(); p0 = m0.predict(teb).to_numpy(); p1 = m1.predict(teb).to_numpy()
        w = ew.weights_from_base(y, p0, teb["PA_test"].to_numpy())[0] if wlab == "production" else np.ones(len(teb))
        print(f"pitchers BB% {wlab}: z coefficient {m1.params['z']:+.4f} (t {m1.tvalues['z']:+.2f})", flush=True)
        report("pitchers (frozen BB% model)", "BB%", y, p0, p1, w, teb["pitcher_id"].astype(str).to_numpy(), teb["PA"].to_numpy(float), wlab)


def ra9():
    import fl_pwar_calibration as cal
    cal.RETURN_ALL = True
    with contextlib.redirect_stdout(io.StringIO()):
        b, war = cal.main()
    b = b[(b["ip"] >= 10) & b["pred"].notna() & b["target"].notna()].copy()
    bf25 = war[war["year"] == 2025].groupby("player")["bf"].sum(); b["bf25"] = b["player"].map(bf25).fillna(0); b["lz"] = np.log1p(b["bf25"])
    y = b["target"].to_numpy(); p0 = b["pred"].to_numpy(); rng = np.random.default_rng(3)
    for wlab in ("production", "per player"):
        ww = b["ip"].to_numpy() if wlab == "production" else np.ones(len(b)); p1 = np.zeros(len(b)); fold = rng.integers(0, 10, len(b)); gs = []
        for k in range(10):
            tr, te = fold != k, fold == k; z = b["lz"].to_numpy() - np.average(b["lz"].to_numpy()[tr], weights=ww[tr]); e = y - p0
            A = np.column_stack([np.ones(tr.sum()), z[tr]]); g = np.linalg.lstsq(A * np.sqrt(ww[tr])[:, None], e[tr] * np.sqrt(ww[tr]), rcond=None)[0][1]; gs.append(g); p1[te] = p0[te] + g * z[te]
        print(f"RA9 {wlab}: gamma by fold mean {np.mean(gs):+.3f} runs/9 per unit of log BF (< 0 = thin arms' prior WORSE)", flush=True)
        report("pitchers RA9 2025 -> 2026", "RA9", y, p0, p1, ww, b["pitcher_id"].astype(str).to_numpy(), b["bf25"].to_numpy(), wlab)


def main():
    for f in (hitters, pitchers, ra9):
        try:
            f()
        except Exception:
            import traceback; traceback.print_exc()
    pd.set_option("display.width", 260); R = pd.DataFrame(ROWS); R.to_csv("output/pt_prior_test.csv", index=False)
    print("\nPLAYING-TIME-DEPENDENT PRIOR vs LEAGUE-MEAN PRIOR (level-free; dRMSE < 0 = PT prior better)\n" + R.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
