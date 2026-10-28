"""
fl_pstrike_network.py -- P(strike | stuff, location, count, handedness) network, and does it line up with team-wide trends / team over- and under-performance against the projections?  (user 2026-10-28)
MODEL: per pitch, strike = called strike, swinging strike or foul; ball = called ball or ball in the dirt (balls in play, hit batters, intentional and automatic calls excluded, as the project's other strike networks).  Inputs: the 7 stuff measures (velocity, induced vertical break, arm-side horizontal break, spin, extension, release height,
release side), location (arm-side-signed plate_x, plate_z), balls, strikes, pitcher hand, batter side and a same-hand flag; no pitch-type input (the request named stuff, location, count and handedness).  3-seed PyTorch MLP (the project's fl_wh_sva_torch), 5-fold cross-fit BY PITCHER, so a pitcher's own pitches never inform his own prediction and no team identity is an input.
A second network without the stuff measures (location + count + handedness) separates what the location alone earns from what the stuff adds (stuff value = P_full - P_loc).
ANALYSIS: (a) calibration / AUC; (b) pitcher-level persistence 2025 -> 2026 of expected strike rate, actual strike rate, residual (actual - expected) and stuff value; (c) team tables (pitching team, 2025 and 2026): expected, actual, residual, stuff value, and the residual at HOME parks vs on the ROAD (a park / umpire effect shows up in both
teams' pitchers at that park; a team effect follows the team); (d) alignment of the 2026 team measures with the team K%, BB% and RA9 misses against the preseason projections (output/team_surprise_components_2026.csv, output/team_surprise_2026.csv), Spearman / Pearson across the 18 teams.
Usage: python fl_pstrike_network.py    (log: output/pstrike_network_log.txt)    Output: output/pstrike_pitch_level.parquet, output/pstrike_team_2026.csv, output/pstrike_network.json
"""
import torch  # MUST come first: on this Windows machine importing xgboost / scikit-learn before torch breaks torch's DLL load
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

import fl_pitcher_features as fp
import fl_position_player_flag as ppf
import fl_wh_sva_torch as ft

warnings.filterwarnings("ignore")
SEEDS = (1, 2, 3); OUT = "output/pstrike_pitch_level.parquet"
HAND = ["throws_r", "bat_l", "same_hand"]


def _job(tag, cfg, seed, tr, te, X, y, g):
    torch.set_num_threads(2); p, ep, best = ft.fit_predict(X[tr], y[tr], X[te], g[tr], seed=seed, cfg=cfg); return tag, seed, te, p


def build_population():
    raw = ppf.exclude(fp.load_fl_pitches_cached()); d = fp.add_count(raw[raw["pitch_call"] != "undefined"]); d = d[d["pitch_call"].isin(fp._STRIKE_CALLS | fp._BALL_CALLS)].copy(); d["target"] = d["pitch_call"].isin(fp._STRIKE_CALLS).astype(int)
    d = d[d["pitcher_throws"].isin(["left", "right"]) & d["batter_side"].isin(["left", "right"])].copy(); d["throws_r"] = (d["pitcher_throws"] == "right").astype(float); d["bat_l"] = (d["batter_side"] == "left").astype(float); d["same_hand"] = (d["pitcher_throws"] == d["batter_side"]).astype(float)
    need = fp.FULL_COLS + ["balls", "strikes"] + HAND; pop = d.dropna(subset=need).reset_index(drop=True); print(f"{len(pop):,} decision pitches (strike rate {pop['target'].mean():.4f}); years {pop['year'].value_counts().to_dict()}", flush=True); return pop


def fit_networks(pop):
    y = pop["target"].to_numpy(int); g = pop["pitcher_id"].to_numpy(); cnt = ["balls", "strikes"] + HAND; Xf = pop[fp.FULL_COLS + cnt].astype(float).to_numpy(); Xl = pop[fp.LOC_COLS + cnt].astype(float).to_numpy()
    folds = list(GroupKFold(n_splits=5).split(Xf, y, g)); jobs = []
    for tag, X, cfg in (("full", Xf, {}), ("loc", Xl, {"hidden": (64, 64)})):
        jobs += [delayed(_job)(tag, cfg, s, tr, te, X, y, g) for s in SEEDS for tr, te in folds]
    O = {}
    for tag, s, te, p in Parallel(n_jobs=8, backend="loky")(jobs):
        O.setdefault((tag, s), np.zeros(len(y)))[te] = p
    return np.mean([O[("full", s)] for s in SEEDS], axis=0), np.mean([O[("loc", s)] for s in SEEDS], axis=0)


def main():
    pd.set_option("display.width", 250); res = {}
    if os.path.exists(OUT):
        pop = pd.read_parquet(OUT)
    else:
        t0 = time.time(); pop = build_population(); pf, pl = fit_networks(pop); pop["p_full"] = pf; pop["p_loc"] = pl; pop.to_parquet(OUT, index=False); print(f"networks done in {time.time() - t0:.0f}s", flush=True)
    y = pop["target"].to_numpy(); base = y.mean()
    for lab, c in (("full (stuff + location + count + handedness)", "p_full"), ("location + count + handedness only", "p_loc")):
        p = pop[c].to_numpy(); print(f"{lab}: AUC {roc_auc_score(y, p):.4f}, log loss {log_loss(y, p):.4f} (base-rate guess {log_loss(y, np.full(len(y), base)):.4f})"); res[lab] = {"auc": float(roc_auc_score(y, p)), "logloss": float(log_loss(y, p))}
    pop["bin"] = pd.qcut(pop["p_full"], 10, duplicates="drop"); cal = pop.groupby("bin").agg(n=("target", "size"), predicted=("p_full", "mean"), actual=("target", "mean")); print("\nCALIBRATION by predicted-probability decile\n" + cal.round(3).to_string())
    pop["resid"] = pop["target"] - pop["p_full"]; pop["stuff_value"] = pop["p_full"] - pop["p_loc"]; pop["pid"] = pop["pitcher_id"].astype(str)
    # (b) pitcher persistence
    pit = pop.groupby(["pid", "year"]).agg(n=("target", "size"), expected=("p_full", "mean"), actual=("target", "mean"), resid=("resid", "mean"), stuff=("stuff_value", "mean"), loc_expected=("p_loc", "mean")).reset_index()
    a, b = pit[pit["year"] == 2025].set_index("pid"), pit[pit["year"] == 2026].set_index("pid"); rows = []
    for thr in (150, 300, 500):
        j = a.join(b, lsuffix="_25", rsuffix="_26", how="inner"); j = j[(j["n_25"] >= thr) & (j["n_26"] >= thr)]; rows.append({"min decision pitches per year": thr, "pitchers": len(j), **{f"r {m}": float(np.corrcoef(j[f"{m}_25"], j[f"{m}_26"])[0, 1]) for m in ("expected", "actual", "resid", "stuff", "loc_expected")}})
    P = pd.DataFrame(rows); print("\nPITCHER-LEVEL YEAR-TO-YEAR CORRELATION (2025 vs 2026)\n" + P.round(3).to_string(index=False)); res["pitcher_persistence"] = P.to_dict("records")
    # (c) team tables
    tabs = {}
    for yr in (2025, 2026):
        d = pop[pop["year"] == yr]; g = d.groupby("pitcher_team").agg(pitches=("target", "size"), expected=("p_full", "mean"), actual=("target", "mean"), resid=("resid", "mean"), stuff=("stuff_value", "mean"), loc_expected=("p_loc", "mean"))
        d = d.assign(home=(d["pitcher_team"] == d["home_team"])); h = d[d["home"]].groupby("pitcher_team")["resid"].mean().rename("resid_home"); r = d[~d["home"]].groupby("pitcher_team")["resid"].mean().rename("resid_road"); tabs[yr] = g.join(h).join(r)
        T = tabs[yr].sort_values("resid", ascending=False); print(f"\nTEAM TABLE {yr} (pitching team; strike rates in percentage points; resid = actual - expected)\n" + (T * np.array([1, 100, 100, 100, 100, 100, 100, 100])).round(2).to_string())
    j = tabs[2025].join(tabs[2026], lsuffix="_25", rsuffix="_26", how="inner"); print("\nTEAM-LEVEL YEAR-TO-YEAR CORRELATION 2025 vs 2026 (18 teams): " + ", ".join(f"{m} {np.corrcoef(j[m + '_25'], j[m + '_26'])[0, 1]:+.2f}" for m in ("expected", "actual", "resid", "stuff", "loc_expected")))
    pk = pop.assign(home_team_=pop["home_team"]).groupby(["home_team_", "year"]).agg(resid=("resid", "mean"), n=("target", "size")).reset_index(); pw = pk.pivot(index="home_team_", columns="year", values="resid"); print("PARK / UMPIRE effect (mean residual of ALL pitchers at that home park): correlation 2025 vs 2026 across parks " + f"{np.corrcoef(pw[2025].dropna(), pw[2026].reindex(pw[2025].dropna().index).fillna(pw[2026].mean()))[0, 1]:+.2f}; SD of park means 2026: {100 * pw[2026].std():.2f} points")
    # (d) alignment with the 2026 team misses
    c = pd.read_csv("output/team_surprise_components_2026.csv"); w = pd.read_csv("output/team_surprise_2026.csv")
    def pick(df, key, col, grp):
        m = df[df["table"].astype(str).str.contains(key)] if "table" in df.columns else df[df["group"] == grp]; return m.set_index("team")[col]
    pk_ = pick(c, "PITCHERS K%", "points vs projection", None); pb_ = pick(c, "PITCHERS BB%", "points vs projection", None); pr_ = w[w["group"] == "pitchers"].set_index("team")["runs vs projection"]
    A = tabs[2026].join(pk_.rename("K% miss (pts)")).join(pb_.rename("BB% miss (pts)")).join(pr_.rename("RA9 runs saved vs proj")); print("\nTEAM 2026: network measures next to the misses against the preseason projections\n" + A.assign(**{c_: 100 * A[c_] for c_ in ("expected", "actual", "resid", "stuff", "loc_expected", "resid_home", "resid_road")}).round(2).sort_values("resid", ascending=False).to_string())
    rows = []
    for m in ("expected", "actual", "resid", "stuff", "loc_expected", "resid_home", "resid_road"):
        for tgt in ("K% miss (pts)", "BB% miss (pts)", "RA9 runs saved vs proj"):
            ok = A[[m, tgt]].dropna(); rows.append({"measure": m, "vs": tgt, "n": len(ok), "Pearson": pearsonr(ok[m], ok[tgt])[0], "Spearman": spearmanr(ok[m], ok[tgt])[0]})
    R = pd.DataFrame(rows).pivot(index="measure", columns="vs", values="Spearman"); print("\nALIGNMENT (Spearman across 18 teams, 2026)\n" + R.round(2).to_string()); print(pd.DataFrame(rows).pivot(index="measure", columns="vs", values="Pearson").round(2).to_string())
    A.to_csv("output/pstrike_team_2026.csv"); json.dump(res, open("output/pstrike_network.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
