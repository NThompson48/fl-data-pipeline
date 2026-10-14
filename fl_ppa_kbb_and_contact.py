"""
fl_ppa_kbb_and_contact.py -- user hypothesis: pitches-per-PA information is more
believable if it feeds the K% and BB% projections (where a plate-discipline /
pitch-count skill would show up) rather than entering the RA9 model directly --
unless it reflects QUALITY OF CONTACT.  Two tests:

Part 1  Does pitch-count information predict FUTURE K% / BB% beyond the shipped
        K%/BB% models?  Experiment 2 and Experiment 1 held-out harness (same as
        fl_bridge_value_deepdive.py): add pitches/PA, or its four components
        (fouls, called strikes, swinging strikes, balls per PA), to the K% and
        BB% specs, refit on the train transition, score held-out; paired
        bootstrap on the weighted squared-error difference vs the base spec.
Part 2  Does ppa_resid (and each component) track CONTACT QUALITY allowed?  Per
        pitcher-season (>=40 batted balls): exit velo, hard-hit% (>=95 mph),
        xwOBA on contact, actual wOBA on contact, wOBA-xwOBA gap, air%.
        Same-year (partial on K%, BB%) and predictive (prior-year component ->
        next-year metric, controlling for the metric's own prior value).

Usage:  python fl_ppa_kbb_and_contact.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sqlalchemy import text

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import mlb_marcel as mm
import fl_ppa_decomposition as dec
import fl_bridge_value_deepdive as dd
from fl_supabase import read_sql_df

N_BOOT = 2000
PC = ["foul_pa", "cs_pa", "sw_pa", "ball_pa"]


def prep(raw, train_feat, train_lab, test_feat, test_lab):
    df_tr, n0_k, n0_bb = bt.assemble(raw, train_feat, train_lab, with_changepoints=False)
    df_te, _, _ = bt.assemble(raw, test_feat, test_lab, with_changepoints=False)

    def lg_bb(windows):
        pool = pd.concat([fp.filter_period(raw, s, e) for s, e in windows], ignore_index=True)
        t = pool[pool["is_terminal"]]
        return (t["k_or_bb"] == "walk").mean()
    lg_tr, lg_te = lg_bb(train_feat), lg_bb(test_feat)
    for df, lg in ((df_tr, lg_tr), (df_te, lg_te)):
        df["BB_raw"] = df["naive_BB_rate"] * (df["PA"] + n0_bb) - n0_bb * lg
    d_bb = df_tr.dropna(subset=["BB_rate_actual", "naive_BB_rate", "fl_bb_rate_bridge", "PA", "command_score"])
    w = mm.reliability_weight(d_bb["PA_test"], n0_bb).values
    n0b = mm.fit_n0_bridge(d_bb["BB_raw"], d_bb["PA"], n0_bb, lg_tr, d_bb["fl_bb_rate_bridge"], d_bb["BB_rate_actual"], w)
    df_tr["blended_BB_rate"] = mm.blended_rate(df_tr["BB_raw"], df_tr["PA"], n0_bb, lg_tr, n0b, df_tr["fl_bb_rate_bridge"])
    df_te["blended_BB_rate"] = mm.blended_rate(df_te["BB_raw"], df_te["PA"], n0_bb, lg_te, n0b, df_te["fl_bb_rate_bridge"])
    # pitch-count features over each frame's FEATURE window
    for df, windows in ((df_tr, train_feat), (df_te, test_feat)):
        pool = pd.concat([fp.filter_period(raw, s, e) for s, e in windows], ignore_index=True)
        c = dec.components(pool, None, None)[["pitcher_id", "ppa_check"] + PC].rename(columns={"ppa_check": "ppa"})
        df["pitcher_id"] = df["pitcher_id"].astype(str)
        df[["ppa"] + PC] = df[["pitcher_id"]].merge(c, on="pitcher_id", how="left")[["ppa"] + PC].values
    return df_tr, df_te, n0_k, n0_bb


def eval_variants(df_tr, df_te, target, base, variants, n0, label, seed=0):
    cols = sorted({c for v in [base] + list(variants.values()) for c in v})
    tr = df_tr.dropna(subset=[target] + cols); te = df_te.dropna(subset=[target] + cols).copy()
    preds = {}
    for nm, regs in {"base": base, **variants}.items():
        w = mm.reliability_weight(tr["PA_test"], n0).values
        m = bt.fit_weighted(tr, target, regs, w)
        preds[nm] = m.predict(te).values
        if nm != "base":
            tv = {k: round(v, 2) for k, v in m.tvalues.items() if k in regs and k not in base}
            preds[nm + "_t"] = tv
    y = te[target].values; w = mm.reliability_weight(te["PA_test"], n0).values
    rng = np.random.default_rng(seed)
    se_base = (y - preds["base"]) ** 2
    rows = []
    for nm in ["base"] + list(variants):
        se = (y - preds[nm]) ** 2
        diffs = []
        if nm != "base":
            for _ in range(N_BOOT):
                i = rng.integers(0, len(y), len(y))
                diffs.append((np.sum(w[i] * se_base[i]) - np.sum(w[i] * se[i])) / np.sum(w[i] * se_base[i]))
        rows.append({"variant": nm, "R2_w": round(mm.weighted_r2(y, preds[nm], w), 4),
                     "RMSE_w pp": round(mm.weighted_rmse(y, preds[nm], w) * 100, 3),
                     "SSE cut vs base % [90% CI]": "" if nm == "base" else
                     f"{100*np.mean(diffs):+.1f} [{100*np.percentile(diffs,5):+.1f}, {100*np.percentile(diffs,95):+.1f}]",
                     "train t-values (new terms)": "" if nm == "base" else preds[nm + "_t"]})
    print(f"\n{label}  (train n={len(tr)}, test n={len(te)})"); print(pd.DataFrame(rows).to_string(index=False))


def contact_table(raw):
    b = pd.read_csv("output/batted_ball_event_residuals.csv").dropna(subset=["pitcher_id", "exit_speed"])
    b["pitcher_id"] = b["pitcher_id"].astype("int64").astype(str)
    b["hh"] = b["exit_speed"] >= 95
    b["air"] = b["angle"] >= 10
    b["gap"] = b["woba"] - b["xwoba"]
    g = b.groupby(["pitcher_id", "year"]).agg(bip=("hh", "size"), ev=("exit_speed", "mean"), hh=("hh", "mean"),
                                              xwobacon=("xwoba", "mean"), wobacon=("woba", "mean"), gap=("gap", "mean"),
                                              air=("air", "mean")).reset_index()
    return g[g["bip"] >= 40]


def season_frame(raw, war, year, win):
    c = dec.components(raw, *win)
    w = war[(war["year"] == year) & (war["bf"] >= 100)].copy()
    w["K"] = w["so"] / w["bf"]; w["BB"] = w["bb"] / w["bf"]
    uniq = c[~c["pitcher"].duplicated(keep=False)]
    d = uniq.merge(w[["player", "K", "BB", "ppf"]], left_on="pitcher", right_on="player")
    for comp in PC:
        d[comp + "_r"] = smf.ols(f"{comp} ~ K + BB + ppf", data=d).fit().resid
    d["ppa_r"] = d[[c_ + "_r" for c_ in PC]].sum(axis=1)
    d["year"] = year
    return d


if __name__ == "__main__":
    raw = fp.load_fl_pitches_cached(); raw["date"] = pd.to_datetime(raw["date"])
    w25, w26 = (bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)

    print("PART 1 -- do pitch counts predict FUTURE K% / BB% beyond the shipped models?")
    for name, trf, trl, tef, tel in (
            ("Exp2 (train full-2025 -> 1H26; test 2025+1H26 -> 2H26)", [w25], [w26], [w25, w26], [(bt._DAY_AFTER_2026_CUT, bt.END_2026)]),
            ("Exp1 (train 1H25 -> 2H25; test 2H25 -> 1H26)", [(bt.START_2025, bt.CUTOFF_2025)], [(bt._DAY_AFTER_2025_CUT, bt.END_2025)],
             [(bt._DAY_AFTER_2025_CUT, bt.END_2025)], [w26])):
        df_tr, df_te, n0_k, n0_bb = prep(raw, trf, trl, tef, tel)
        eval_variants(df_tr, df_te, "K_rate_actual", dd.K_FULL,
                      {"+ pitches/PA": dd.K_FULL + ["ppa"], "+ 4 components": dd.K_FULL + PC,
                       "+ fouls/PA": dd.K_FULL + ["foul_pa"], "+ called strikes/PA": dd.K_FULL + ["cs_pa"]},
                      n0_k, f"{name} -- K%")
        eval_variants(df_tr, df_te, "BB_rate_actual", dd.BB_FULL,
                      {"+ pitches/PA": dd.BB_FULL + ["ppa"], "+ 4 components": dd.BB_FULL + PC,
                       "+ balls/PA": dd.BB_FULL + ["ball_pa"], "+ fouls/PA": dd.BB_FULL + ["foul_pa"]},
                      n0_bb, f"{name} -- BB%")

    print("\n\nPART 2 -- does the pitch-count signal track CONTACT QUALITY?")
    war = read_sql_df(text("select player, year, team, bf, so, bb, ppf from war_pitchers where game_type='reg_season'"))
    ct = contact_table(raw)
    f25 = season_frame(raw, war, 2025, (bt.START_2025, bt.END_2025)); f26 = season_frame(raw, war, 2026, (bt.START_2026, bt.END_2026))
    METR = ["ev", "hh", "xwobacon", "wobacon", "gap", "air"]
    rows = []
    for f, yr in ((f25, 2025), (f26, 2026)):
        d = f.merge(ct[ct["year"] == yr], on="pitcher_id")
        for mt in METR:   # metric residualized on K%, BB% (partial)
            d[mt + "_p"] = smf.ols(f"{mt} ~ K + BB", data=d).fit().resid
        for x in PC + ["ppa"]:
            xr = (x + "_r") if x != "ppa" else "ppa_r"
            rows.append({"year": yr, "n": len(d), "component resid": x.replace("_pa", "") if x != "ppa" else "ppa_resid total",
                         **{mt: round(d[xr].corr(d[mt + "_p"]), 2) for mt in METR}})
    print("[2a] SAME-YEAR partial correlation (contact metric residualized on K%, BB%) with each pitch-count residual")
    print(pd.DataFrame(rows).to_string(index=False))

    both = f25.merge(ct[ct["year"] == 2025], on="pitcher_id").merge(
        ct[ct["year"] == 2026], on="pitcher_id", suffixes=("_25", "_26"))
    print(f"\n[2b] PREDICTIVE: 2025 pitch-count residual -> 2026 contact metric, controlling for the metric's own 2025 value (n={len(both)})")
    rows = []
    for x in PC + ["ppa"]:
        xr = (x + "_r") if x != "ppa" else "ppa_r"
        r = {"component resid": x.replace("_pa", "") if x != "ppa" else "ppa_resid total"}
        for mt in METR:
            m = smf.ols(f"{mt}_26 ~ {mt}_25 + {xr}", data=both).fit()
            r[mt] = f"t={m.tvalues[xr]:+.1f}"
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))
    print("(|t| > 2 would be a real predictive link; contact metrics: ev=exit velo, hh=hard-hit%, xwobacon, wobacon, gap=wOBA-xwOBA, air=air%)")
