"""
fl_command_shrink_test.py -- does SHRINKING the Command Score by its own reliability improve the preseason BB% model?  (user 2026-10-28: "did we try it with shrinkage?")
Production preseason BB% = WLS on [blended_BB_rate, command_score] with the RAW pitcher-window score; only the in-season (ROS) model shrinks it.  Tested here in the project's BB% harness (fl_command_variants.bb_harness design: four pairings T0->T1b, T0->T2, T1->T2, T2->T0,
frozen weights from the production model, level-free, weighted RMSE / MAE, pitcher bootstrap 1,000):
  command_score_shrunk(n0) = mu + (score - mu) x n / (n + n0), n = the pitcher's pitches in the feature window, mu = the pitch-weighted mean of the training window's scores, n0 in {50, 95 (the measured reliability n0), 150, 300}.
Output: output/command_shrink_test_log.txt, output/command_shrink_test.json
"""
import json
import warnings

import numpy as np
import pandas as pd

import fl_command_variants as cvr
import fl_eval_report as er
import fl_eval_weights as ew
import fl_halfseason_backtest as bt
import fl_pitcher_features as fp
import fl_position_player_flag as ppf
import mlb_marcel as mm

warnings.filterwarnings("ignore")
N0S = (50, 95, 150, 300)


def frame(raw, name):
    feat, lab = cvr.WINDOWS[name]; df, n0k, n0bb = bt.assemble_features(raw, feat, with_changepoints=False, with_surprise_levels=False); df = df.merge(bt.actual_rate(raw, lab), on="pitcher_id", how="inner")
    cs = fp.compute_command_score(pd.concat([fp.filter_period(raw, a, b) for a, b in feat], ignore_index=True))[["pitcher_id", "command_score", "n_pitches"]]; df = df.merge(cs, on="pitcher_id", how="left")
    ok = df["command_score"].notna(); mu = float(np.average(df.loc[ok, "command_score"], weights=df.loc[ok, "n_pitches"]))
    df["command_score"] = df["command_score"].fillna(mu); df["n_pitches"] = df["n_pitches"].fillna(0.0)
    for n0 in N0S:
        df[f"cs_s{n0}"] = mu + (df["command_score"] - mu) * df["n_pitches"] / (df["n_pitches"] + n0)
    return df, n0bb, cvr.lg_bb(raw, feat)


def main():
    pd.set_option("display.width", 250); raw = ppf.exclude(fp.load_fl_pitches_cached()); fr = {n: frame(raw, n) for n in cvr.WINDOWS}; labels = [f"cs_s{n0}" for n0 in N0S]; rows = []; ncs = {n: fr[n][0]["n_pitches"].describe().round(0).to_dict() for n in fr}
    print("pitches behind the score (training windows):", {k: (v["25%"], v["50%"], v["75%"]) for k, v in ncs.items()}, flush=True)
    for a, b in cvr.PAIRS:
        da, n0, lga = fr[a]; db, _, lgb = fr[b]; need = ["BB_rate_actual", "naive_BB_rate", "fl_bb_rate_bridge", "PA"]; ta = da.dropna(subset=need).copy(); tb = db.dropna(subset=need).copy(); wtr = mm.reliability_weight(ta["PA_test"], n0).values
        ta["BB_raw"] = ta["naive_BB_rate"] * (ta["PA"] + n0) - n0 * lga; n0b = mm.fit_n0_bridge(ta["BB_raw"], ta["PA"], n0, lga, ta["fl_bb_rate_bridge"], ta["BB_rate_actual"], wtr); ta["blended_BB_rate"] = mm.blended_rate(ta["BB_raw"], ta["PA"], n0, lga, n0b, ta["fl_bb_rate_bridge"])
        tb["BB_raw"] = tb["naive_BB_rate"] * (tb["PA"] + n0) - n0 * lgb; tb["blended_BB_rate"] = mm.blended_rate(tb["BB_raw"], tb["PA"], n0, lgb, n0b, tb["fl_bb_rate_bridge"]); P = {}; coef = {}
        for c in ["command_score"] + labels:
            m = bt.fit_weighted(ta, "BB_rate_actual", ["blended_BB_rate", c], wtr); P[c] = m.predict(tb).to_numpy(); coef[c] = float(m.params[c])
        y = tb["BB_rate_actual"].to_numpy(); n_t = tb["PA_test"].to_numpy(); ids = tb["pitcher_id"].to_numpy(); base = P["command_score"]; w = ew.weights_from_base(y, base, n_t)[0]; cen = lambda q: q + np.average(y - q, weights=w)
        for c in ["command_score"] + labels:
            rm, ma = er.wmetrics(y, cen(P[c]), w); r = {"pairing": f"{a}->{b}", "n": len(y), "model": c, "RMSE": rm, "MAE": ma, "coef": coef[c]}
            if c != "command_score":
                q = er.compare(y, cen(base), cen(P[c]), w, ids, n=1000); r.update({"dRMSE": q["dRMSE"], "90%": q["RMSE 90%"], "P(RMSE)": q["P(RMSE better)"], "dMAE": q["dMAE"], "P(MAE)": q["P(MAE better)"]})
            rows.append(r)
        print(f"  {a}->{b} done", flush=True)
    T = pd.DataFrame(rows); print("\nBB%: per pairing (change vs the production RAW command_score; negative = better)\n" + T[T["model"] != "command_score"].round(5).to_string(index=False))
    g = T.groupby("model").agg(RMSE=("RMSE", "mean"), MAE=("MAE", "mean")); g["dRMSE vs production"] = g["RMSE"] - g.loc["command_score", "RMSE"]; g["dMAE vs production"] = g["MAE"] - g.loc["command_score", "MAE"]; g["pairings better (RMSE)"] = T[T["model"] != "command_score"].groupby("model")["dRMSE"].apply(lambda s: int((s < 0).sum()))
    print("\nMEAN OVER THE FOUR PAIRINGS\n" + g.round(5).to_string()); json.dump({"bb": T.to_dict("records"), "bb_mean": g.reset_index().to_dict("records")}, open("output/command_shrink_test.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
