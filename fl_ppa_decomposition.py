"""
fl_ppa_decomposition.py -- decompose `ppa_resid` (pitches per PA beyond what
K%/BB%/park imply) into the things that actually extend or shorten a plate
appearance, and find out which part is pitcher skill and which part carries the
RA9 signal.

Identity (exact): every PA has exactly one terminal pitch, so
    pitches/PA - 1 = fouls/PA + called strikes(non-terminal)/PA
                     + swinging strikes(non-terminal)/PA + balls(non-terminal)/PA.
Each component is regressed on the same (K%, BB%, ppf) the production
auxiliary model uses; because OLS residuals are linear and the regressors are
identical, the four component residuals SUM EXACTLY to ppa_resid.

Questions asked of each component residual:
  1. Is it a stable PITCHER trait?  (year-over-year corr, 2025 -> 2026; and for
     team-changers separately, since a trait should travel with the pitcher.)
  2. How much of it is TEAM context (catcher framing, umpire zones, tracking)?
     (one-way ICC by team.)
  3. Does it improve out-of-sample RA9 prediction, forward and reversed?
     Variants: shipped ppa_resid / none / foul only / called-strike only /
     foul + called-strike / all four.

Usage:  python fl_ppa_decomposition.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sqlalchemy import text

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import mlb_marcel as mm
import fl_ra9_trustmark_projection as ra9
import fl_ra9_input_robustness as rob
from fl_supabase import read_sql_df

BALLS = {"ball_called", "ball_in_dirt", "ball_intentional", "automatic_ball"}
FOULS = {"foul_not_fieldable", "foul_fieldable", "foul_ball"}
COMPS = ["foul_pa", "cs_pa", "sw_pa", "ball_pa"]
LABEL = {"foul_pa": "foul balls", "cs_pa": "called strikes", "sw_pa": "swinging strikes", "ball_pa": "balls"}


def components(raw: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    p = fp.filter_period(raw, start, end)
    # same corrupted-game guard as ra9.pitches_per_pa
    g = p.groupby(["pitcher_id", "game_id"]).agg(n=("pitch_call", "size"), t=("is_terminal", "sum")).reset_index()
    bad = g[(g["n"] >= ra9.CORRUPT_GAME_MIN_PITCHES) & (g["t"] == 0)]
    if len(bad):
        keys = set(zip(bad["pitcher_id"], bad["game_id"]))
        p = p[[k not in keys for k in zip(p["pitcher_id"], p["game_id"])]]
    nt = p[~p["is_terminal"]]
    cnt = pd.DataFrame({
        "foul": nt.groupby("pitcher_id")["pitch_call"].apply(lambda s: s.isin(FOULS).sum()),
        "cs": nt.groupby("pitcher_id")["pitch_call"].apply(lambda s: (s == "strike_called").sum()),
        "sw": nt.groupby("pitcher_id")["pitch_call"].apply(lambda s: (s == "strike_swinging").sum()),
        "ball": nt.groupby("pitcher_id")["pitch_call"].apply(lambda s: s.isin(BALLS).sum()),
        "other_nt": nt.groupby("pitcher_id")["pitch_call"].apply(lambda s: (~s.isin(FOULS | BALLS | {"strike_called", "strike_swinging"})).sum()),
    })
    pa = p[p["is_terminal"]].groupby("pitcher_id").size().rename("PA")
    out = cnt.join(pa, how="inner").reset_index()
    for c in ("foul", "cs", "sw", "ball", "other_nt"):
        out[c + "_pa"] = out[c] / out["PA"]
    out["ppa_check"] = 1 + out[["foul_pa", "cs_pa", "sw_pa", "ball_pa", "other_nt_pa"]].sum(axis=1)
    team = p.groupby("pitcher_id")["pitcher_team"].agg(lambda s: s.mode().iloc[0]).rename("team")
    name = p.groupby("pitcher_id")["pitcher"].first().rename("pitcher")
    out = out.merge(team, on="pitcher_id").merge(name, on="pitcher_id")
    out["pitcher_id"] = out["pitcher_id"].astype(str)
    return out


def icc_oneway(df, val, grp):
    g = df.groupby(grp)[val]
    n_i = g.size(); k = len(n_i); n = n_i.sum()
    grand = df[val].mean()
    ssb = (n_i * (g.mean() - grand) ** 2).sum(); ssw = ((df[val] - g.transform("mean")) ** 2).sum()
    msb, msw = ssb / (k - 1), ssw / (n - k)
    n0 = (n - (n_i ** 2).sum() / n) / (k - 1)
    return float((msb - msw) / (msb + (n0 - 1) * msw))


def loocv_comps(m, resid_comps, ppa_scale=1.0):
    """LOOCV of RA9 ~ K + BB + ppf + ip_per_g_prior + <component residuals>.
    Each component's residual is built from an auxiliary OLS on the TRAINING
    fold only (component ~ K_pred + BB_pred + ppf), then applied to the held-out
    pitcher -- same discipline as production's ppa_resid."""
    n = len(m); preds = np.full(n, np.nan)
    terms = " + ".join(f"{c}_r" for c in resid_comps)
    spec = "target ~ K_pred + BB_pred + ppf + ip_per_g_prior" + (f" + {terms}" if resid_comps else "")
    for i in range(n):
        tr = m.drop(index=i).copy(); te = m.loc[[i]].copy()
        for c in resid_comps:
            aux = smf.ols(f"{c} ~ K_pred + BB_pred + ppf", data=tr).fit()
            tr[f"{c}_r"] = aux.resid
            te[f"{c}_r"] = (te[c] - aux.predict(te))
        mod = smf.wls(spec, data=tr, weights=tr["w"]).fit()
        for c in ("ppf", "ip_per_g_prior"):
            te[c] = te[c].clip(tr[c].min(), tr[c].max())
        preds[i] = float(mod.predict(te).iloc[0])
    return preds


def loocv_shipped(m):
    n = len(m); preds = np.full(n, np.nan)
    for i in range(n):
        tr = m.drop(index=i).copy(); te = m.loc[[i]].copy()
        aux = smf.ols(ra9.PPA_RESID_SPEC, data=tr).fit()
        tr["ppa_resid"] = aux.resid
        te["ppa_resid"] = te["pitches_per_pa_prior"] - aux.predict(te)
        mod = smf.wls(ra9.RA9_SPEC, data=tr, weights=tr["w"]).fit()
        for c in ra9.CONTINUOUS_REGRESSORS:
            te[c] = te[c].clip(tr[c].min(), tr[c].max())
        preds[i] = float(mod.predict(te).iloc[0])
    return preds


if __name__ == "__main__":
    raw = fp.load_fl_pitches_cached(); raw["date"] = pd.to_datetime(raw["date"])
    war = read_sql_df(text("select player, year, team, bf, ip, so, bb, ra9, ppf, g from war_pitchers where game_type='reg_season'"))
    c25 = components(raw, bt.START_2025, bt.END_2025); c26 = components(raw, bt.START_2026, bt.END_2026)
    print(f"identity check (max |ppa - recon| over pitchers, should be ~0 up to 'other' pitches): "
          f"{(c26['ppa_check'] - (c26[['foul_pa','cs_pa','sw_pa','ball_pa','other_nt_pa']].sum(axis=1) + 1)).abs().max():.1e}; "
          f"mean other_nt_pa={c26['other_nt_pa'].mean():.4f}")

    # attach K/BB/ppf (war_pitchers is keyed by name) and residualize each component per season
    def with_war(c, year):
        w = war[(war["year"] == year) & (war["bf"] >= 100)].copy()
        w["K"] = w["so"] / w["bf"]; w["BB"] = w["bb"] / w["bf"]
        uniq = c[~c["pitcher"].duplicated(keep=False)]
        d = uniq.merge(w[["player", "K", "BB", "ppf", "ra9", "ip", "bf"]], left_on="pitcher", right_on="player")
        for comp in COMPS:
            d[comp + "_r"] = smf.ols(f"{comp} ~ K + BB + ppf", data=d).fit().resid
        d["ppa_r_sum"] = d[[c_ + "_r" for c_ in COMPS]].sum(axis=1)
        return d
    d25, d26 = with_war(c25, 2025), with_war(c26, 2026)
    print(f"\n[0] variance of ppa_resid explained by each component residual (2026, n={len(d26)}): "
          + ", ".join(f"{LABEL[c]} sd={d26[c+'_r'].std():.3f}" for c in COMPS) + f";  total sd={d26['ppa_r_sum'].std():.3f}")
    cov = d26[[c + "_r" for c in COMPS]].cov(); tot = d26["ppa_r_sum"].var()
    print("    share of Var(ppa_resid): " + ", ".join(f"{LABEL[c]} {cov[c+'_r'].sum()/tot:.0%}" for c in COMPS))

    both = d25.merge(d26, on="pitcher_id", suffixes=("_25", "_26"))
    print(f"\n[1] year-over-year stability of each residual component (n={len(both)} pitchers with BF>=100 both years)")
    rows = []
    for c in COMPS + ["ppa_r_sum"]:
        a, b = both[c + "_r_25" if c != "ppa_r_sum" else "ppa_r_sum_25"], both[c + "_r_26" if c != "ppa_r_sum" else "ppa_r_sum_26"]
        same = both["team_25"] == both["team_26"]
        rows.append({"component": LABEL.get(c, "ppa_resid total"), "YoY corr all": round(a.corr(b), 3),
                     "same team": round(a[same].corr(b[same]), 3), f"changed team (n={int((~same).sum())})": round(a[~same].corr(b[~same]), 3),
                     "ICC(team) 2026": round(icc_oneway(d26, c + "_r" if c != "ppa_r_sum" else "ppa_r_sum", "team"), 3)})
    print(pd.DataFrame(rows).to_string(index=False))

    # [2] RA9 value, forward + reversed
    base_ids = pd.DataFrame({"pitcher_id": raw["pitcher_id"].astype(str).unique(), "K_oof": 0.0, "BB_oof": 0.0})
    for lab, y_in, y_tgt, win, cin in (("FORWARD 2025 -> 2026", 2025, 2026, (bt.START_2025, bt.END_2025), c25),
                                       ("REVERSED 2026 -> 2025", 2026, 2025, (bt.START_2026, bt.END_2026), c26)):
        m = rob.build_frame(war, raw, base_ids, y_in, y_tgt, win)
        m["pitcher_id"] = m["pitcher_id"].astype(str)
        m = m.merge(cin[["pitcher_id"] + [c for c in COMPS]], on="pitcher_id", how="inner").reset_index(drop=True)
        m["K_pred"], m["BB_pred"] = m["K_raw"], m["BB_raw"]
        variants = [("shipped ppa_resid", None), ("no pitch-count term", []), ("foul balls only", ["foul_pa"]),
                    ("called strikes only", ["cs_pa"]), ("foul + called strikes", ["foul_pa", "cs_pa"]),
                    ("all four components", COMPS)]
        res = []
        for nm, comps in variants:
            p = loocv_shipped(m) if comps is None else loocv_comps(m, comps)
            res.append({"variant": nm, **{k: v for k, v in rob.summarize(m, p, nm).items() if k in ("R2_w", "RMSE_w", "calib_slope")}})
        print(f"\n[2] RA9 LOOCV {lab}  (n={len(m)})"); print(pd.DataFrame(res).to_string(index=False))
        # coefficients on the full sample for the multi-component variant
        trd = m.copy()
        for c in COMPS:
            trd[c + "_r"] = smf.ols(f"{c} ~ K_pred + BB_pred + ppf", data=trd).fit().resid
        mod = smf.wls("target ~ K_pred + BB_pred + ppf + ip_per_g_prior + " + " + ".join(c + "_r" for c in COMPS),
                      data=trd, weights=trd["w"]).fit()
        print("    full-sample coefficients (runs/9 per +1.0 pitch/PA of that component's residual): "
              + ", ".join(f"{LABEL[c]} {mod.params[c+'_r']:+.2f} (p={mod.pvalues[c+'_r']:.2f})" for c in COMPS))

    # [3] Jackson Smith's profile
    js = d26[d26["pitcher"] == "Smith, Jackson"]
    if len(js):
        print("\n[3] Jackson Smith 2026 residual components (pitches/PA vs. what his K%/BB%/park imply):")
        print(js[[c + "_r" for c in COMPS] + ["ppa_r_sum"]].round(3).to_string(index=False))
        print("    percentile among BF>=100 pitchers (low = fewer than expected): "
              + ", ".join(f"{LABEL[c]} {100*(d26[c+'_r'] < js[c+'_r'].iloc[0]).mean():.0f}" for c in COMPS))
