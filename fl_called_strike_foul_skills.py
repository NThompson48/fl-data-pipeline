"""
fl_called_strike_foul_skills.py -- cleaner "skill" versions of the pitch-count
signal inside `ppa_resid`, per the user's suggestion (called strikes / foul
balls):

  csae       called strikes ABOVE EXPECTED per take, where "expected" is a
             location model P(called strike | take, plate_x, plate_z within
             the batter's zone, batter side) fit out-of-fold (GroupKFold by
             game). Removes pure location (a pitcher who throws more
             borderline strikes is not the same as one who gets them called).
  csae_adj   the same with the CATCHER'S framing/umpire-zone effect removed
             (catcher mean residual, shrunk by n/(n+400) toward 0, subtracted
             pitch-by-pitch) -- so what is left is the pitcher's own called-
             strike edge (deception / pitch shape at the edge), not the
             receiver's glove.
  foul_share fouls / (fouls + balls in play): how much of a pitcher's contact
             dies as a foul rather than a ball in play.
  whiff_sw   swinging strikes / swings.

For each: (1) YoY stability 2025->2026, split by team changers (a real
pitcher skill should travel; a catcher/ump effect should not), (2) incremental
out-of-sample RA9 value added to the base spec (K%, BB%, park, IP/G) forward
and reversed, vs the shipped ppa_resid.

Usage:  python fl_called_strike_foul_skills.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import GroupKFold
from sqlalchemy import text
from xgboost import XGBClassifier

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import fl_ra9_trustmark_projection as ra9
import fl_ra9_input_robustness as rob
import fl_ppa_decomposition as dec
from fl_supabase import read_sql_df

FOULS = dec.FOULS
CATCHER_SHRINK_N = 400.0


def take_model_residuals(raw: pd.DataFrame) -> pd.DataFrame:
    """Out-of-fold (by game) expected called-strike probability on takes."""
    t = raw[raw["pitch_call"].isin(["strike_called", "ball_called"])].copy()
    t = t.dropna(subset=["plate_x", "plate_z", "sz_top", "sz_bot", "batter_side"])
    t["zn"] = (t["plate_z"] - t["sz_bot"]) / (t["sz_top"] - t["sz_bot"])
    t["bs"] = (t["batter_side"].astype(str).str.lower().str.startswith("l")).astype(int)
    t["y"] = (t["pitch_call"] == "strike_called").astype(int)
    X = t[["plate_x", "zn", "bs"]].values; y = t["y"].values
    oof = np.zeros(len(t))
    for tr, te in GroupKFold(5).split(X, y, groups=t["game_id"]):
        m = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, subsample=0.8, tree_method="hist",
                          eval_metric="logloss", random_state=0, n_jobs=-1)
        m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    t["exp"] = oof; t["res"] = t["y"] - t["exp"]
    # catcher effect, shrunk
    cg = t.groupby("catcher_id")["res"].agg(["mean", "size"])
    cg["eff"] = cg["mean"] * cg["size"] / (cg["size"] + CATCHER_SHRINK_N)
    t["res_adj"] = t["res"] - t["catcher_id"].map(cg["eff"]).fillna(0.0)
    print(f"take model: n={len(t):,}, base CS rate {t['y'].mean():.3f}, OOF logloss-free check: mean |res| {t['res'].abs().mean():.3f}; "
          f"catcher effect sd (n>=400 takes) = {cg.loc[cg['size']>=400,'eff'].std():.4f}, n catchers={len(cg)}")
    return t[["pitcher_id", "date", "res", "res_adj", "y"]]


def pitcher_skills(raw, takes, start, end):
    p = fp.filter_period(raw, start, end)
    tk = takes[(takes["date"] >= pd.Timestamp(start)) & (takes["date"] <= pd.Timestamp(end))]
    g = tk.groupby("pitcher_id").agg(takes=("y", "size"), csae=("res", "mean"), csae_adj=("res_adj", "mean")).reset_index()
    pc = p.groupby("pitcher_id")["pitch_call"]
    cnt = pd.DataFrame({"foul": pc.apply(lambda s: s.isin(FOULS).sum()), "inplay": pc.apply(lambda s: (s == "in_play").sum()),
                        "whiff": pc.apply(lambda s: (s == "strike_swinging").sum()),
                        "swings": p.groupby("pitcher_id")["is_swing"].sum()}).reset_index()
    cnt["foul_share"] = cnt["foul"] / (cnt["foul"] + cnt["inplay"]).clip(lower=1)
    cnt["whiff_sw"] = cnt["whiff"] / cnt["swings"].clip(lower=1)
    team = p.groupby("pitcher_id")["pitcher_team"].agg(lambda s: s.mode().iloc[0]).rename("team").reset_index()
    out = g.merge(cnt, on="pitcher_id").merge(team, on="pitcher_id")
    out["pitcher_id"] = out["pitcher_id"].astype(str)
    return out[out["takes"] >= 300]


def loocv_extra(m, feat_cols):
    n = len(m); preds = np.full(n, np.nan)
    spec = "target ~ K_pred + BB_pred + ppf + ip_per_g_prior" + "".join(f" + {c}" for c in feat_cols)
    for i in range(n):
        tr = m.drop(index=i); te = m.loc[[i]].copy()
        mod = smf.wls(spec, data=tr, weights=tr["w"]).fit()
        for c in ("ppf", "ip_per_g_prior"):
            te[c] = te[c].clip(tr[c].min(), tr[c].max())
        preds[i] = float(mod.predict(te).iloc[0])
    return preds


if __name__ == "__main__":
    raw = fp.load_fl_pitches_cached(); raw["date"] = pd.to_datetime(raw["date"])
    war = read_sql_df(text("select player, year, team, bf, ip, so, bb, ra9, ppf, g from war_pitchers where game_type='reg_season'"))
    takes = take_model_residuals(raw)
    s25 = pitcher_skills(raw, takes, bt.START_2025, bt.END_2025)
    s26 = pitcher_skills(raw, takes, bt.START_2026, bt.END_2026)
    FEATS = ["csae", "csae_adj", "foul_share", "whiff_sw"]
    both = s25.merge(s26, on="pitcher_id", suffixes=("_25", "_26"))
    both = both[(both["foul_25"] + both["inplay_25"] >= 60) & (both["foul_26"] + both["inplay_26"] >= 60)]
    same = both["team_25"] == both["team_26"]
    print(f"\n[1] YoY stability (n={len(both)}; team changers n={int((~same).sum())})")
    rows = []
    for f in FEATS:
        a, b = both[f + "_25"], both[f + "_26"]
        rows.append({"skill": f, "YoY corr": round(a.corr(b), 3), "same team": round(a[same].corr(b[same]), 3),
                     "changed team": round(a[~same].corr(b[~same]), 3),
                     "ICC(team) 2026": round(dec.icc_oneway(s26, f, "team"), 3)})
    print(pd.DataFrame(rows).to_string(index=False))

    base_ids = pd.DataFrame({"pitcher_id": raw["pitcher_id"].astype(str).unique(), "K_oof": 0.0, "BB_oof": 0.0})
    for lab, y_in, y_tgt, win, sin in (("FORWARD 2025 -> 2026", 2025, 2026, (bt.START_2025, bt.END_2025), s25),
                                       ("REVERSED 2026 -> 2025", 2026, 2025, (bt.START_2026, bt.END_2026), s26)):
        m = rob.build_frame(war, raw, base_ids, y_in, y_tgt, win)
        m["pitcher_id"] = m["pitcher_id"].astype(str)
        m = m.merge(sin[["pitcher_id"] + FEATS], on="pitcher_id", how="inner").reset_index(drop=True)
        m["K_pred"], m["BB_pred"] = m["K_raw"], m["BB_raw"]
        for f in FEATS:   # standardize so coefficients are per-SD
            m[f] = (m[f] - m[f].mean()) / m[f].std()
        variants = [("shipped ppa_resid", None), ("base (no pitch-count term)", [])] + [(f, [f]) for f in FEATS] + \
                   [("csae_adj + foul_share", ["csae_adj", "foul_share"])]
        res = []
        for nm, cols in variants:
            p = dec.loocv_shipped(m) if cols is None else loocv_extra(m, cols)
            res.append({"variant": nm, **{k: v for k, v in rob.summarize(m, p, nm).items() if k in ("R2_w", "RMSE_w", "calib_slope")}})
        print(f"\n[2] RA9 LOOCV {lab} (n={len(m)})"); print(pd.DataFrame(res).to_string(index=False))
        full = smf.wls("target ~ K_pred + BB_pred + ppf + ip_per_g_prior + " + " + ".join(FEATS), data=m, weights=m["w"]).fit()
        print("    coefficients per +1 SD (runs/9): " + ", ".join(f"{f} {full.params[f]:+.2f} (p={full.pvalues[f]:.2f})" for f in FEATS))

    js = s26.merge(raw[["pitcher_id", "pitcher"]].drop_duplicates("pitcher_id").assign(pitcher_id=lambda d: d["pitcher_id"].astype(str)), on="pitcher_id")
    j = js[js["pitcher"] == "Smith, Jackson"]
    if len(j):
        print("\n[3] Jackson Smith 2026 percentiles among pitchers with >=300 takes:")
        print({f: int(round(100 * (s26[f] < j[f].iloc[0]).mean())) for f in FEATS}, "values:", j[FEATS].round(3).to_dict("records"))
