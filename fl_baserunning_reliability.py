"""
fl_baserunning_reliability.py -- how STICKY is baserunning?  2026 split-half reliability from play-by-play  (user 2026-10-10: "We are not able to do baserunning WAR for 2025. Can we check 2026 reliability / split-half stickiness?")
Source: pbp_events (2026 only: that is why baserunning WAR exists only for 2026).  Components rebuilt per runner / batter:
  STEALS    opportunity = a runner on 1B with 2B open during a plate appearance (one per runner x PA); attempt (STEAL_PLAY from 1B); success (SAFE) / caught (OUT);  runs = rv_sb x SB + rv_cs x CS - league runs per opportunity x opportunities
  EXTRA BASES  hits (single / double, no HR) with a runner on 1B / 2B: extra base taken = ended beyond the minimum advance (single: 1B -> 3B or home, 2B -> home; double: 1B -> home); out on the bases = ended out;  (taken / out rate - league rate for that hit type, start base and outs) x run value
  GDP       batter ground balls with a runner on 1B and < 2 outs: double plays vs the league rate for those states x run value
Run values are approximations (SB +0.20, CS -0.45, extra base +0.27, out on bases -0.65, GDP -0.44): they only weight the composite; the component reliabilities do not depend on them.  Published components (war_players 2026: sb, cs, wsb_runs, ubr_runs, wgdp_runs, bsr_runs) are used to validate the rebuild.
Reliability: random split not used; ALTERNATING GAMES (game order by date, odd vs even) and FIRST vs SECOND HALF of the schedule; correlation across runners with enough opportunities in both halves, Spearman-Brown corrected to a full season (r_full = 2r/(1+r)), implied share of true-talent variance, SD of the observed and the true component in runs per 600 PA.
"""
import warnings

import numpy as np
import pandas as pd
from sqlalchemy import text

import fl_innings_projection as ip
from fl_supabase import read_sql_df

warnings.filterwarnings("ignore")
RV = {"sb": 0.20, "cs": -0.45, "ebt": 0.27, "oob": -0.65, "gdp": -0.44}


def load():
    cols = ("game_guid, date, seq, event_name, pa_uid, outs_before, base_state_before, runner_1b_id, runner_2b_id, runner_3b_id, runner_1b_after_id, runner_2b_after_id, runner_3b_after_id, scorer_ids, pa_result, batted_ball_type, outs_on_play, batter_id, steal_from, steal_to, steal_result, is_pa_end")
    e = read_sql_df(text(f"select {cols} from pbp_events where extract(year from date) = 2026 and event_name in ('PITCH', 'BIP', 'STEAL_PLAY')")); e["date"] = pd.to_datetime(e["date"])
    pl = read_sql_df(text("select player_id, name from pbp_players")); nm = dict(zip(pl["player_id"], pl["name"])); return e, nm


def build(e):
    g = e[["game_guid", "date"]].drop_duplicates().sort_values(["date", "game_guid"]).reset_index(drop=True); g["gi"] = np.arange(len(g)); gi = dict(zip(g["game_guid"], g["gi"])); e["gi"] = e["game_guid"].map(gi); e["odd"] = e["gi"] % 2; e["late"] = (e["gi"] >= len(g) / 2).astype(int)
    rows = []   # (runner id, game idx, component, opp, x, expected key)
    # steals: opportunities from PITCH events with a runner on 1B and 2B open, one per (pa_uid, runner)
    p = e[(e["event_name"] == "PITCH") & e["base_state_before"].isin(["1__", "1_3"]) & e["runner_1b_id"].notna()].drop_duplicates(["pa_uid", "runner_1b_id"])
    opp = pd.DataFrame({"rid": p["runner_1b_id"], "gi": p["gi"], "odd": p["odd"], "late": p["late"], "comp": "sb_opp", "v": 1.0})
    s = e[(e["event_name"] == "STEAL_PLAY") & (e["steal_from"] == "1B") & (e["steal_to"] == "2B") & e["steal_result"].isin(["SAFE", "OUT"])].copy(); s["rid"] = s["runner_1b_id"]; s = s.dropna(subset=["rid"])
    att = pd.concat([pd.DataFrame({"rid": s["rid"], "gi": s["gi"], "odd": s["odd"], "late": s["late"], "comp": "sb_att", "v": 1.0}), pd.DataFrame({"rid": s["rid"], "gi": s["gi"], "odd": s["odd"], "late": s["late"], "comp": np.where(s["steal_result"] == "SAFE", "sb", "cs"), "v": 1.0})])
    # extra bases on hits
    b = e[(e["event_name"] == "BIP") & e["pa_result"].isin(["1B", "2B"])].copy(); out = []
    for base, rcol in (("1B", "runner_1b_id"), ("2B", "runner_2b_id")):
        x = b[b[rcol].notna()].copy()
        if base == "2B":
            x = x[x["pa_result"] == "1B"]                                               # runner on 2B: only singles are scored
        x["rid"] = x[rcol]; sc = x["scorer_ids"].fillna("").astype(str)
        end = np.where([r in t for r, t in zip(x["rid"], sc)], 4, np.where(x["rid"] == x["runner_3b_after_id"], 3, np.where(x["rid"] == x["runner_2b_after_id"], 2, np.where(x["rid"] == x["runner_1b_after_id"], 1, 0))))
        start = 1 if base == "1B" else 2
        x["end"] = end; x["start"] = start; x["minend"] = np.where(x["pa_result"] == "1B", start + 1, start + 2).clip(max=4); x["ebt"] = (x["end"] > x["minend"]).astype(float); x["oob"] = (x["end"] == 0).astype(float); x["key"] = x["pa_result"] + "|" + base + "|" + x["outs_before"].astype(str)
        out.append(x[["rid", "gi", "odd", "late", "key", "ebt", "oob"]])
    eb = pd.concat(out)
    gd = e[(e["event_name"] == "BIP") & (e["batted_ball_type"] == "GB") & e["base_state_before"].str.startswith("1") & (e["outs_before"] < 2) & e["batter_id"].notna()].copy(); gd["gdp"] = ((gd["outs_on_play"] >= 2) & gd["pa_result"].isin(["OUT", "FC"])).astype(float); gd["key"] = gd["base_state_before"] + "|" + gd["outs_before"].astype(str)
    pa = e[e["is_pa_end"].astype(bool) & e["batter_id"].notna()].drop_duplicates(["pa_uid"])
    real = lambda d, c="rid": d[~d[c].astype(str).str.startswith("__")]       # drop the automatic extra-innings runner (rid "__PLACED__"): it is "out" on every hit by construction
    return real(opp), real(att), real(eb[eb["key"].str.split("|").str[2].astype(int) <= 2]), real(gd, "batter_id"), pa


def table(opp, att, eb, gd, pa, half):
    """per runner (and half if half in {'odd','late'}) component totals"""
    def agg(df, col, name, cat=None):
        k = ["rid"] + ([half] if half else []); return df.groupby(k)[col].sum().rename(name)
    parts = [opp.groupby(["rid"] + ([half] if half else []))["v"].sum().rename("sb_opp")]
    for c in ("sb_att", "sb", "cs"):
        parts.append(att[att["comp"] == c].groupby(["rid"] + ([half] if half else []))["v"].sum().rename(c))
    lg = eb.groupby("key")[["ebt", "oob"]].mean(); eb = eb.assign(ebt_x=eb["ebt"] - eb["key"].map(lg["ebt"]), oob_x=eb["oob"] - eb["key"].map(lg["oob"])); k = ["rid"] + ([half] if half else [])
    parts += [eb.groupby(k)["ebt_x"].sum().rename("ebt_x"), eb.groupby(k).size().rename("ebt_opp"), eb.groupby(k)["oob_x"].sum().rename("oob_x")]
    lgg = gd.groupby("key")["gdp"].mean(); gd = gd.assign(gdp_x=gd["gdp"] - gd["key"].map(lgg)); gd["rid"] = gd["batter_id"]; parts += [gd.groupby(k)["gdp_x"].sum().rename("gdp_x"), gd.groupby(k).size().rename("gdp_opp")]
    pa = pa.assign(rid=pa["batter_id"]); parts.append(pa.groupby(k).size().rename("pa"))
    T = pd.concat(parts, axis=1).fillna(0.0); return T


def main():
    e, nm = load(); opp, att, eb, gd, pa = build(e); print(f"2026 events: {len(e):,}; steal opportunities {len(opp):,}, attempts {int((att['comp'] == 'sb_att').sum()):,}; extra-base opportunities {len(eb):,}; GDP opportunities {len(gd):,}")
    for df in (opp, att):
        pass
    T = table(opp, att, eb, gd, pa, None); lg_sb_per_opp = (T["sb"].sum() * RV["sb"] + T["cs"].sum() * RV["cs"]) / T["sb_opp"].sum()
    T["wsb"] = RV["sb"] * T["sb"] + RV["cs"] * T["cs"] - lg_sb_per_opp * T["sb_opp"]; T["ubr"] = RV["ebt"] * T["ebt_x"] + RV["oob"] * T["oob_x"]; T["wgdp"] = -RV["gdp"] * 0 + RV["gdp"] * T["gdp_x"]; T["bsr_rebuilt"] = T["wsb"] + T["ubr"] + T["wgdp"]
    T["name"] = T.index.map(nm); T["name_clean"] = T["name"].map(lambda s: " ".join(str(s).lower().replace(".", "").replace("'", "").split()) if isinstance(s, str) else None)
    war = read_sql_df(text("select player, pa, sb, cs, wsb_runs, ubr_runs, wgdp_runs, bsr_runs from war_players where year = 2026 and game_type = 'reg_season'")); war["name_clean"] = war["player"].apply(ip.lastfirst_to_clean)
    V = T.dropna(subset=["name_clean"]).groupby("name_clean").sum(numeric_only=True).join(war.groupby("name_clean")[["pa", "sb", "cs", "wsb_runs", "ubr_runs", "wgdp_runs", "bsr_runs"]].sum(), rsuffix="_pub", how="inner"); V = V[V["pa_pub"] >= 100]
    print(f"\nREBUILD vs PUBLISHED 2026 (n = {len(V)} hitters with >= 100 PA): corr SB {V['sb'].corr(V['sb_pub']):.2f}, CS {V['cs'].corr(V['cs_pub']):.2f}; wSB-lite vs published wsb_runs {V['wsb'].corr(V['wsb_runs']):.2f}; UBR-lite vs ubr_runs {V['ubr'].corr(V['ubr_runs']):.2f}; wGDP-lite vs wgdp_runs {V['wgdp'].corr(V['wgdp_runs']):.2f}; "
          f"composite vs bsr_runs {V['bsr_rebuilt'].corr(V['bsr_runs']):.2f}; SD of rebuilt composite {V['bsr_rebuilt'].std():.2f} runs vs published {V['bsr_runs'].std():.2f}")
    res = []
    for split in ("odd", "late"):
        H = table(opp, att, eb, gd, pa, split).reset_index(); A = H[H[split] == 0].set_index("rid"); B = H[H[split] == 1].set_index("rid"); J = A.join(B, lsuffix="_a", rsuffix="_b", how="inner")
        sb_opp = lg_sb_per_opp
        for lab, need, fa, fb, scale_col in (("steal attempt rate (attempts / opportunity)", ("sb_opp", 15), lambda d, s: d["sb_att" + s] / d["sb_opp" + s].replace(0, np.nan), None, None),
                                             ("steal runs per opportunity (wSB-lite)", ("sb_opp", 15), lambda d, s: (RV["sb"] * d["sb" + s] + RV["cs"] * d["cs" + s] - lg_sb_per_opp * d["sb_opp" + s]) / d["sb_opp" + s].replace(0, np.nan), None, None),
                                             ("steal success rate (SB / attempts), >= 4 attempts per half", ("sb_att", 4), lambda d, s: d["sb" + s] / d["sb_att" + s].replace(0, np.nan), None, None),
                                             ("extra bases taken above expected per opportunity", ("ebt_opp", 8), lambda d, s: d["ebt_x" + s] / d["ebt_opp" + s].replace(0, np.nan), None, None),
                                             ("outs on the bases above expected per opportunity", ("ebt_opp", 8), lambda d, s: d["oob_x" + s] / d["ebt_opp" + s].replace(0, np.nan), None, None),
                                             ("GDP above expected per opportunity, >= 6 per half", ("gdp_opp", 6), lambda d, s: d["gdp_x" + s] / d["gdp_opp" + s].replace(0, np.nan), None, None),
                                             ("COMPOSITE baserunning runs per PA (all three), >= 100 PA per half", ("pa", 100), lambda d, s: (RV["sb"] * d["sb" + s] + RV["cs"] * d["cs" + s] - lg_sb_per_opp * d["sb_opp" + s] + RV["ebt"] * d["ebt_x" + s] + RV["oob"] * d["oob_x" + s] + RV["gdp"] * d["gdp_x" + s]) / d["pa" + s].replace(0, np.nan), None, None)):
            col, thr = need; m = (J[col + "_a"] >= thr) & (J[col + "_b"] >= thr); x, y = fa(J[m], "_a"), fa(J[m], "_b"); ok = x.notna() & y.notna(); x, y = x[ok], y[ok]
            if len(x) < 15:
                continue
            r = np.corrcoef(x, y)[0, 1]; rf = 2 * r / (1 + r); sd_obs = np.sqrt((x.var() + y.var()) / 2); res.append({"split": "alternating games" if split == "odd" else "first vs second half", "component": lab, "runners": len(x), "split-half r": r, "full-season reliability": rf, "implied n0 (opps)": np.nan})
    R = pd.DataFrame(res); print("\nSPLIT-HALF RELIABILITY (2026)\n" + R.drop(columns="implied n0 (opps)").round(3).to_string(index=False))
    T2 = V.copy(); T2["pa600"] = 600 / T2["pa_pub"]; print(f"\nSEASON-LEVEL SPREAD (hitters >= 100 PA, runs per 600 PA): rebuilt composite SD {(T2['bsr_rebuilt'] * T2['pa600']).std():.2f}, published bsr SD {(T2['bsr_runs'] * T2['pa600']).std():.2f}; steals {(T2['wsb'] * T2['pa600']).std():.2f}, extra bases {(T2['ubr'] * T2['pa600']).std():.2f}, GDP {(T2['wgdp'] * T2['pa600']).std():.2f}")
    R.to_csv("output/baserunning_reliability_2026.csv", index=False)


if __name__ == "__main__":
    main()
