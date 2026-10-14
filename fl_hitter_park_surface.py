"""
fl_hitter_park_surface.py -- HITTER Trustmark translation built on the batted-ball residual (wOBA - xwOBA) surface  (user 2026-10-10: "this is the basis we should use for our park effects ... it's the best control on competition level, since we are conditioning on the quality of the batted balls")
Definition: the park effect = what Trustmark does to a ball of a GIVEN exit speed / launch angle (/ side / spray), relative to the other parks: effect(x) = f_TM(x) - f_other(x) with f the mean of the hitter-adjusted residual res_adj = (wOBA - xwOBA) - the hitter's own non-Trustmark mean residual
(shrunk n0 150), fitted on VISITING hitters' balls in play only (home hitters are selected for the park), by gradient boosting with strong regularisation (depth 3, 40 balls per leaf).  Contact-QUALITY shifts (hitters making harder contact at Trustmark) are deliberately NOT part of the park effect.
Hitter shift (wOBA per ball in play) = mean of effect(x) over the hitter's own balls (Trustmark excluded), shrunk toward the league mean of effect by n/(n + n0); per PA = x (balls in play per PA); strikeouts, walks and the rest stay global.
features(): 'evla' = (EV, LA); 'evla_side' = + batter side; 'evla_side_dir' = + field direction (spray).
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import fl_hitter_ros as hr

warnings.filterwarnings("ignore")
TM = "TrustmarkPark"; RES_N0 = 150.0
FEATS = {"evla": ["ev", "la"], "evla_side": ["ev", "la", "left"], "evla_side_dir": ["ev", "la", "left", "dir"]}


def load_balls():
    d = pd.read_parquet("2025_2026_pitches.parquet", columns=["game_id", "pa_id", "year", "stadium", "home_team", "batter_team", "is_terminal_pitch", "exit_speed", "angle", "direction", "xwoba", "woba", "play_result", "batter_side"])
    b = d[d["is_terminal_pitch"].astype(str).isin(["1", "True", "true"]) & d["exit_speed"].notna() & d["angle"].notna() & d["woba"].notna() & d["xwoba"].notna() & d["play_result"].isin(["out", "single", "double", "triple", "error", "fielders_choice", "sacrifice", "home_run"]) & d["batter_side"].isin(["left", "right"])].copy()
    pa = hr.Tables().pa[["game_id", "pa_id", "name_clean"]].drop_duplicates(["game_id", "pa_id"]); b = b.merge(pa, on=["game_id", "pa_id"], how="left").dropna(subset=["name_clean"]).reset_index(drop=True)
    b["tm"] = (b["stadium"] == TM).astype(int); b["vis"] = (b["batter_team"] != b["home_team"]).astype(int); b["left"] = (b["batter_side"] == "left").astype(int); b["ev"] = b["exit_speed"]; b["la"] = b["angle"]; b["dir"] = b["direction"].fillna(0.0)
    b["res"] = b["woba"] - b["xwoba"]; return b


def hitter_baseline(b, col="res", n0=RES_N0):
    """each hitter's own NON-Trustmark mean of `col` (shrunk toward 0 by n0): removes systematic hitter-level over / under-performance vs expected (speed, spray, ...)."""
    non = b[b["tm"] == 0].groupby("name_clean")[col].agg(["sum", "size"]); return (non["sum"] / (non["size"] + n0))


def fit_effect(train, feats, seed=0):
    """effect surface from VISITING hitters: returns predict(X) = f_TM - f_other."""
    v = train[train["vis"] == 1]; fm = FEATS[feats]; kw = dict(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=40, l2_regularization=1.0, random_state=seed)
    f1 = HistGradientBoostingRegressor(**kw).fit(v.loc[v["tm"] == 1, fm], v.loc[v["tm"] == 1, "res_adj"]); f0 = HistGradientBoostingRegressor(**kw).fit(v.loc[v["tm"] == 0, fm], v.loc[v["tm"] == 0, "res_adj"])
    return lambda X: f1.predict(X[fm]) - f0.predict(X[fm])


def hitter_shift(balls_profile, effect, feats, n0=60.0):
    """per-hitter mean effect over his non-Trustmark balls (wOBA per ball in play), shrunk toward the league mean of the effect; returns Series by name_clean, plus the league mean."""
    x = balls_profile[balls_profile["tm"] == 0].copy(); x["eff"] = effect(x); g = x.groupby("name_clean")["eff"].agg(["mean", "size"]); mu = float(np.average(g["mean"], weights=g["size"])); return mu + (g["mean"] - mu) * g["size"] / (g["size"] + n0), mu


if __name__ == "__main__":
    b = load_balls(); b["res_adj"] = b["res"] - b["name_clean"].map(hitter_baseline(b)).fillna(0.0); print(f"{len(b):,} balls in play; visiting at Trustmark {int(((b['tm'] == 1) & (b['vis'] == 1)).sum()):,}")
    for f in FEATS:
        eff = fit_effect(b, f); sh, mu = hitter_shift(b, eff, f); print(f"{f}: league-mean effect {mu:+.4f}; hitter shift SD {sh.std():.4f}; range [{sh.min():+.3f}, {sh.max():+.3f}] over {len(sh)} hitters")
