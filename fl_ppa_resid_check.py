"""
fl_ppa_resid_check.py -- is `ppa_resid` (pitches per PA beyond what K%/BB%/ppf
imply) a stable PITCHER trait, or does it mostly reflect the team context
(catcher framing, umpires, scoring/tracking conventions) the pitcher happens to
be in?  Prompted by Jackson Smith: a 4th-percentile ppa_resid (-0.24 pitches/PA)
contributes +0.51 runs/9 to his RA9 -- bigger than his K%, BB% and IP/G
contributions combined -- so it needs to be a real trait to deserve that weight.

Checks (BF>=100 pitchers):
  1. year-over-year stability of ppa_resid (2025 -> 2026), overall and for
     pitchers who did NOT change teams vs those who did.
  2. share of ppa_resid variance that is TEAM-level (ICC, one-way random effects)
     within each season, and team-fixed-effect-adjusted stability.
  3. Smith-specific: his teammates' ppa_resid, his 2025 vs 2026 value.
  4. Out-of-sample value of the term: LOOCV forward + reversed with the shipped
     spec vs the spec WITHOUT ppa_resid, and vs a version where ppa_resid is
     shrunk 50% toward 0.

Usage: python fl_ppa_resid_check.py
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sqlalchemy import text

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import mlb_marcel as mm
import fl_ra9_trustmark_projection as ra9
from fl_supabase import read_sql_df


def per_year_frame(war, raw, year):
    w = war[(war["year"] == year) & (war["bf"] >= 100)].copy()
    win = (bt.START_2025, bt.END_2025) if year == 2025 else (bt.START_2026, bt.END_2026)
    ppa = ra9.pitches_per_pa(raw, *win).rename(columns={"pitches_per_pa": "ppa"})
    w = w.merge(ppa, left_on="player", right_on="pitcher", how="inner")
    w["K"] = w["so"] / w["bf"]
    w["BB"] = w["bb"] / w["bf"]
    return w


def icc_oneway(df, val, grp):
    g = df.groupby(grp)[val]
    n_i = g.size()
    k = len(n_i)
    n = n_i.sum()
    grand = df[val].mean()
    ssb = (n_i * (g.mean() - grand) ** 2).sum()
    ssw = ((df[val] - g.transform("mean")) ** 2).sum()
    msb, msw = ssb / (k - 1), ssw / (n - k)
    n0 = (n - (n_i ** 2).sum() / n) / (k - 1)
    return float((msb - msw) / (msb + (n0 - 1) * msw))


def loocv_variant(m, drop_ppa=False, ppa_scale=1.0):
    n = len(m)
    preds = np.full(n, np.nan)
    spec = "target ~ K_pred + BB_pred + ppf + ip_per_g_prior" + ("" if drop_ppa else " + ppa_resid")
    for i in range(n):
        tr = m.drop(index=i).copy(); te = m.loc[[i]].copy()
        aux = smf.ols(ra9.PPA_RESID_SPEC, data=tr).fit()
        tr["ppa_resid"] = aux.resid
        mod = smf.wls(spec, data=tr, weights=tr["w"]).fit()
        te["ppa_resid"] = (te["pitches_per_pa_prior"] - aux.predict(te)) * ppa_scale
        for c in ra9.CONTINUOUS_REGRESSORS:
            if c in te:
                te[c] = te[c].clip(tr[c].min(), tr[c].max())
        preds[i] = float(mod.predict(te).iloc[0])
    return preds


if __name__ == "__main__":
    raw = fp.load_fl_pitches_cached()
    war = read_sql_df(text("select player, year, team, bf, ip, so, bb, ra9, ppf, g from war_pitchers where game_type='reg_season'"))
    y25, y26 = per_year_frame(war, raw, 2025), per_year_frame(war, raw, 2026)
    # residualize ppa against K, BB, ppf EXACTLY as production does (one aux model per year, own K/BB/ppf)
    for y in (y25, y26):
        aux = smf.ols("ppa ~ K + BB + ppf", data=y).fit()
        y["ppa_resid"] = aux.resid
    both = y25.merge(y26, on="player", suffixes=("_25", "_26"))
    print(f"[1] stability, n={len(both)}: corr(ppa_resid 2025, 2026) = {both['ppa_resid_25'].corr(both['ppa_resid_26']):.3f};  "
          f"raw ppa corr = {both['ppa_25'].corr(both['ppa_26']):.3f}")
    same = both[both["team_25"] == both["team_26"]]; diff = both[both["team_25"] != both["team_26"]]
    print(f"    same-team n={len(same)} corr={same['ppa_resid_25'].corr(same['ppa_resid_26']):.3f};  "
          f"changed-team n={len(diff)} corr={diff['ppa_resid_25'].corr(diff['ppa_resid_26']):.3f}")
    for y, lab in ((y25, "2025"), (y26, "2026")):
        print(f"[2] {lab}: ICC(team) of ppa_resid = {icc_oneway(y, 'ppa_resid', 'team'):.3f} "
              f"(teams={y['team'].nunique()}, n={len(y)}); ICC of ppa_resid among BF>=100 for K%={icc_oneway(y, 'K', 'team'):.3f}")
    t26 = y26.groupby("team")["ppa_resid"].agg(["mean", "count"]).round(3).sort_values("mean")
    print("    2026 team mean ppa_resid (lowest first):"); print(t26.to_string())
    js = y26[y26["player"] == "Smith, Jackson"]
    print("[3] Jackson Smith 2026:", js[["team", "bf", "ppa", "ppa_resid"]].round(3).to_dict("records"))
    js25 = y25[y25["player"] == "Smith, Jackson"]
    print("    Jackson Smith 2025:", js25[["team", "bf", "ppa", "ppa_resid"]].round(3).to_dict("records") or "BF<100 (excluded)")
    mates = y26[y26["team"] == "Mississippi Mud Monsters"]["ppa_resid"]
    print(f"    Mississippi teammates 2026 (n={len(mates)}): mean {mates.mean():.3f}, sd {mates.std():.3f}")

    # [4] out-of-sample value of the term, both directions
    import fl_ra9_input_robustness as rob
    out = {}
    for lab, y_in, y_tgt, win in (("forward 2025->2026", 2025, 2026, (bt.START_2025, bt.END_2025)),
                                  ("reversed 2026->2025", 2026, 2025, (bt.START_2026, bt.END_2026))):
        m = rob.build_frame(war, raw, pd.DataFrame({"pitcher_id": raw["pitcher_id"].unique(), "K_oof": 0.0, "BB_oof": 0.0}),
                            y_in, y_tgt, win)
        m["K_pred"], m["BB_pred"] = m["K_raw"], m["BB_raw"]
        res = []
        for nm, kw in (("shipped (full ppa_resid)", {}), ("ppa_resid x0.5", {"ppa_scale": 0.5}), ("no ppa_resid", {"drop_ppa": True})):
            p = loocv_variant(m, **kw)
            res.append({"variant": nm, **rob.summarize(m, p, nm)})
        print(f"\n[4] {lab} (n={len(m)})")
        print(pd.DataFrame(res)[["variant", "R2_w", "RMSE_w", "calib_slope"]].to_string(index=False))
