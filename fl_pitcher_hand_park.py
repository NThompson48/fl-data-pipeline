"""
fl_pitcher_hand_park.py -- does PITCHER HANDEDNESS matter at Trustmark under the batted-ball residual-surface approach?  (user 2026-10-10: "even if just looking at launch angle / Air%")
A  PROFILE by hand: non-Trustmark balls allowed, pitcher level (>= 100 balls): Air%, mean launch angle, fly+pop-up share, hard-hit share, hard-hit AIR share (EV >= 90, 10-40 deg), mean exit speed, share of balls from left-handed batters; LHP minus RHP with a Welch t.
B  SURFACE SHIFT by hand: the pitcher's mean surface effect over the balls he allows, split into the CONTACT channel (surface evaluated with batter side set to the league mix) and the BATTER-SIDE channel (the rest); LHP minus RHP.
C  DIRECT TEST at Trustmark: hitter-adjusted residual (wOBA - xwOBA - hitter's away mean), Trustmark minus away, by pitcher hand and by pitcher hand x batter side (same-hand / opposite-hand), game-cluster bootstrap SEs; also wOBA on contact (total) and the Air% / hard-air-share composition of the Trustmark balls by pitcher hand.
"""
import warnings

import numpy as np
import pandas as pd

import fl_hitter_park_ms_check as ms
import fl_pitcher_park_surface_test as pt
import fl_park_play_the_park as pp

warnings.filterwarnings("ignore")


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); d = a.mean() - b.mean(); se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)); return d, se, d / se


def main():
    b = pt.load(); raw = pd.read_parquet("2025_2026_pitches.parquet", columns=["game_id", "pa_id", "is_terminal_pitch", "pitcher_throws"]); raw = raw[raw["is_terminal_pitch"].astype(str).isin(["1", "True", "true"])].drop_duplicates(["game_id", "pa_id"])
    b = b.merge(raw[["game_id", "pa_id", "pitcher_throws"]], on=["game_id", "pa_id"], how="left"); b = b[b["pitcher_throws"].astype(str).str.lower().str[0].isin(["l", "r"])].copy(); b["plefty"] = (b["pitcher_throws"].astype(str).str.lower().str[0] == "l").astype(int)
    b["same_hand"] = (b["plefty"] == b["left"]).astype(int); print(f"{len(b):,} balls; lefty-pitcher share {b['plefty'].mean():.3f}; at Trustmark {int(b['tm'].sum()):,}")
    # A
    x = b[b["tm"] == 0].copy(); x["air"] = (x["la"] >= 10).astype(float); x["fbpu"] = (x["la"] >= 25).astype(float); x["hard"] = (x["ev"] >= 90).astype(float); x["hardair"] = ((x["ev"] >= 90) & x["la"].between(10, 39.999)).astype(float)
    g = x.groupby("pid").agg(n=("la", "size"), hand=("plefty", "mean"), air=("air", "mean"), la=("la", "mean"), fbpu=("fbpu", "mean"), hard=("hard", "mean"), hardair=("hardair", "mean"), ev=("ev", "mean"), lhb=("left", "mean")); g = g[g["n"] >= 100]; L, R = g[g["hand"] > 0.5], g[g["hand"] <= 0.5]
    rows = []
    for c, lab in (("air", "Air% (>= 10)"), ("la", "mean launch angle (deg)"), ("fbpu", "fly + pop-up share"), ("hard", "hard-hit share (>= 90 mph)"), ("hardair", "hard-hit AIR share (EV >= 90, 10-40 deg)"), ("ev", "mean exit speed"), ("lhb", "share of balls from left-handed batters")):
        d, se, t = welch(L[c], R[c]); rows.append({"profile": lab, "LHP mean": L[c].mean(), "RHP mean": R[c].mean(), "LHP - RHP": d, "SE": se, "t": t})
    print(f"\nA. BATTED-BALL PROFILE ALLOWED BY HAND (non-Trustmark; {len(L)} LHP, {len(R)} RHP with >= 100 balls)\n" + pd.DataFrame(rows).round(4).to_string(index=False))
    # B
    eff = ms.fit_effect_pooled(b, "evla_side"); fm = ["ev", "la", "left"]; x["eff"] = eff(x)
    xs = x.copy(); xs["left"] = 0; e0 = eff(xs); xs["left"] = 1; e1 = eff(xs); p_left = x["left"].mean(); x["eff_contact"] = (1 - p_left) * np.asarray(e0) + p_left * np.asarray(e1)
    g2 = x.groupby("pid").agg(n=("eff", "size"), hand=("plefty", "mean"), eff=("eff", "mean"), contact=("eff_contact", "mean")); g2 = g2[g2["n"] >= 100]; g2["side_channel"] = g2["eff"] - g2["contact"]; L2, R2 = g2[g2["hand"] > 0.5], g2[g2["hand"] <= 0.5]; rows = []
    for c, lab in (("eff", "total surface shift (wOBA per ball)"), ("contact", "contact channel (batter side at league mix)"), ("side_channel", "batter-side channel")):
        d, se, t = welch(L2[c], R2[c]); rows.append({"shift": lab, "LHP": L2[c].mean(), "RHP": R2[c].mean(), "LHP - RHP": d, "SE": se, "t": t, "= runs/9 at half exposure": d * 0.671 * 39.9 * 0.5})
    print("\nB. SURFACE SHIFT BY HAND (negative = more park penalty)\n" + pd.DataFrame(rows).round(4).to_string(index=False))
    # C
    tm = (b["tm"] == 1).to_numpy(); bt = pp.Boot(b["game_id"].to_numpy(), B=300); res = b["res_adj"].to_numpy(); wo = b["woba"].to_numpy(); rows = []
    cells = [("all balls", np.ones(len(b), bool))] + [("LHP", (b["plefty"] == 1).to_numpy()), ("RHP", (b["plefty"] == 0).to_numpy()), ("LHP vs LHH (same hand)", ((b["plefty"] == 1) & (b["left"] == 1)).to_numpy()), ("LHP vs RHH", ((b["plefty"] == 1) & (b["left"] == 0)).to_numpy()),
                                                      ("RHP vs RHH (same hand)", ((b["plefty"] == 0) & (b["left"] == 0)).to_numpy()), ("RHP vs LHH", ((b["plefty"] == 0) & (b["left"] == 1)).to_numpy())]
    for lab, m in cells:
        e, s = bt.diff(res, tm, m); ew_, sw = bt.diff(wo, tm, m); a = b.loc[m & tm, "la"]; rows.append({"cell": lab, "TM balls": int((m & tm).sum()), "residual TM - other": e, "SE": s, "t": e / s, "wOBAcon TM - other": ew_, "TM Air%": (a >= 10).mean(), "other Air%": (b.loc[m & ~tm, "la"] >= 10).mean(),
                                                                                                    "TM hard-air share": ((b.loc[m & tm, "ev"] >= 90) & a.between(10, 39.999)).mean(), "other hard-air share": ((b.loc[m & ~tm, "ev"] >= 90) & b.loc[m & ~tm, "la"].between(10, 39.999)).mean()})
    print("\nC. TRUSTMARK EFFECT ON SAME-QUALITY CONTACT BY PITCHER HAND (all balls at Trustmark vs away; residual = wOBA - xwOBA, hitter-adjusted)\n" + pd.DataFrame(rows).round(3).to_string(index=False))
    d_, s_ = bt.diff(res, tm, (b["plefty"] == 1).to_numpy()); r_, sr_ = bt.diff(res, tm, (b["plefty"] == 0).to_numpy()); print(f"\nLHP minus RHP residual penalty at Trustmark: {d_ - r_:+.4f} (independent-SE approximation {np.hypot(s_, sr_):.4f})")


if __name__ == "__main__":
    main()
