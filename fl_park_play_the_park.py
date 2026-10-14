"""
fl_park_play_the_park.py -- how does the ball actually play at Trustmark, and what does that mean for who pitches / how we hit / where we stand?  (user 2026-10-09: "run the follow-up ... curious about how to best play at our park even outside of the projection setting")
Data: 2025_2026_pitches.parquet, every tracked ball in play (terminal pitch with exit speed and a wOBA value, home runs included), 2025-26 regular season.
Expected value WITHOUT park information: cross-fitted (5-fold by game) gradient boosting of the ball's wOBA value on exit speed, launch angle, field direction, batter side.  Residual = wOBA value - expected.  So 'Trustmark residual' = what the park adds to a
ball of that quality and direction relative to the league's average park.  Game-cluster bootstrap SEs (B = 400) for Trustmark-minus-other-parks.
A  WHERE the park gives / takes: (1) batted-ball type (ground < 10, line drive 10-25, fly 25-40, pop-up > 40) x spray relative to the hitter (pull / middle / oppo, +-15 deg), (2) airborne balls by FIELD sector (LF line .. RF line) with the HR and non-HR parts, for each batter side.
B  PITCHERS (the follow-up): a pitcher's batted-ball mix allowed OUTSIDE Trustmark (4 types x 3 spray cells) times the cell residuals above = his predicted Trustmark wOBA shift per ball in play (cell residuals re-estimated leaving the pitcher out).
   Then: how spread out is it across pitchers, and does it predict what visiting pitchers actually allowed at Trustmark relative to their own other-park results (slope 1 = structural story right)?
C  OUTFIELD POSITIONING: depth (x) and width (z) of LF/CF/RF at Trustmark vs the other parks, home vs visiting defences, and whether depth explains the outfield hit residual inside Trustmark.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
TM = "TrustmarkPark"
TYPES = [(-91, 10, "ground"), (10, 25, "line drive"), (25, 40, "fly ball"), (40, 91, "pop-up")]
SPRAY = [("pull", 15, 999), ("middle", -15, 15), ("oppo", -999, -15)]
SECTORS = [(-90, -30, "LF line"), (-30, -10, "LF gap"), (-10, 10, "CF"), (10, 30, "RF gap"), (30, 90, "RF line")]


def load():
    d = pd.read_parquet("2025_2026_pitches.parquet")
    b = d[d["is_terminal_pitch"].astype(str).isin(["1", "True", "true"]) & d["exit_speed"].notna() & d["angle"].notna() & d["direction"].notna() & d["woba"].notna()
          & d["play_result"].isin(["out", "single", "double", "triple", "error", "fielders_choice", "sacrifice", "home_run"]) & d["batter_side"].isin(["left", "right"])].copy().reset_index(drop=True)
    b["tm"] = (b["stadium"] == TM).astype(int); b["is_hr"] = (b["play_result"] == "home_run").astype(float); b["hit_nohr"] = b["play_result"].isin(["single", "double", "triple"]).astype(float)
    b["pull_deg"] = np.where(b["batter_side"] == "right", -b["direction"], b["direction"]); b["left"] = (b["batter_side"] == "left").astype(int)
    b["of_fielded"] = b["fielded_by_pos"].isin(["LF", "CF", "RF"]).astype(int); b["field_home"] = (b["pitcher_team"] == b["home_team"]).astype(int)
    b["type"] = pd.cut(b["angle"], [t[0] for t in TYPES] + [91], labels=[t[2] for t in TYPES], right=False)
    b["spray"] = np.select([b["pull_deg"] > 15, b["pull_deg"] < -15], ["pull", "oppo"], "middle")
    b["sector"] = pd.cut(b["direction"], [s[0] for s in SECTORS] + [90.001], labels=[s[2] for s in SECTORS], right=False)
    return b


def cross_fit(b, target):
    X = pd.DataFrame({"ev": b["exit_speed"], "angle": b["angle"], "dir": b["direction"], "side": b["left"]}); y = b[target].to_numpy(float); p = np.zeros(len(b))
    for tr, te in GroupKFold(5).split(X, y, groups=b["game_id"]):
        p[te] = HistGradientBoostingRegressor(max_depth=5, learning_rate=0.05, max_iter=300, min_samples_leaf=50).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    return p


class Boot:
    """game-cluster bootstrap multiplicities"""
    def __init__(self, games, B=400, seed=7):
        codes, uniq = pd.factorize(games); rng = np.random.default_rng(seed); self.codes = codes; self.W = [np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[codes] for _ in range(B)]

    def diff(self, v, tm, mask):
        """Trustmark-minus-other mean of v within mask: (estimate, SE)"""
        a, o = (mask & tm), (mask & ~tm); est = v[a].mean() - v[o].mean(); bs = []
        for w in self.W:
            wa, wo = w * a, w * o
            if wa.sum() > 0 and wo.sum() > 0:
                bs.append((wa * v).sum() / wa.sum() - (wo * v).sum() / wo.sum())
        return est, float(np.std(bs))


def part_a(b, bt):
    tm = (b["tm"] == 1).to_numpy(); rows = []
    for (_, _, tl) in TYPES:
        for sp, _, _ in SPRAY:
            m = ((b["type"] == tl) & (b["spray"] == sp)).to_numpy(); e, s = bt.diff(b["res_woba"].to_numpy(), tm, m); eh, sh = bt.diff(b["res_hr"].to_numpy(), tm, m); en, sn = bt.diff(b["res_hit"].to_numpy(), tm, m)
            rows.append({"type": tl, "spray": sp, "n at TM": int((m & tm).sum()), "share of TM balls": (m & tm).sum() / tm.sum(), "wOBA effect": e, "SE": s, "t": e / s if s else np.nan, "HR/ball": eh, "non-HR hit/ball": en})
    t = pd.DataFrame(rows); print("A1. TRUSTMARK EFFECT PER BALL IN PLAY (wOBA value vs a same-quality ball at the average park); spray is relative to the hitter\n" + t.round(3).to_string(index=False))
    rows = []
    air = (b["angle"] >= 10).to_numpy()
    for side in ("right", "left"):
        for (_, _, sl) in SECTORS:
            m = air & ((b["sector"] == sl) & (b["batter_side"] == side)).to_numpy(); e, s = bt.diff(b["res_woba"].to_numpy(), tm, m); eh, sh = bt.diff(b["res_hr"].to_numpy(), tm, m); en, sn = bt.diff(b["res_hit"].to_numpy(), tm, m)
            rows.append({"batter": side, "field sector": sl, "n at TM": int((m & tm).sum()), "wOBA effect": e, "SE": s, "HR/ball": eh, "HR SE": sh, "non-HR hit/ball": en, "non-HR SE": sn})
    print("\nA2. AIRBORNE BALLS (>= 10 deg) BY FIELD SECTOR AND BATTER SIDE (HR and non-HR hit parts)\n" + pd.DataFrame(rows).round(3).to_string(index=False))


def part_b(b, bt):
    tm = (b["tm"] == 1).to_numpy(); cell = (b["type"].astype(str) + "|" + b["spray"]).to_numpy(); cells = sorted(set(cell)); ci = pd.Categorical(cell, categories=cells).codes; res = b["res_woba"].to_numpy()
    pit = b["pitcher_id"].astype(str).to_numpy(); ok = pd.notna(b["pitcher_id"]).to_numpy()
    # pitcher mix outside Trustmark
    mix = pd.crosstab(pit[ok & ~tm], ci[ok & ~tm]).reindex(columns=range(len(cells)), fill_value=0); n_out = mix.sum(axis=1); mixf = mix.div(n_out, axis=0)
    # cell residual totals (Trustmark / other), for leave-one-pitcher-out cell effects
    S_tm = np.bincount(ci[tm], weights=res[tm], minlength=len(cells)); N_tm = np.bincount(ci[tm], minlength=len(cells)); S_ot = np.bincount(ci[~tm], weights=res[~tm], minlength=len(cells)); N_ot = np.bincount(ci[~tm], minlength=len(cells))
    full_eff = S_tm / N_tm - S_ot / N_ot
    pm_tm = pd.DataFrame({"p": pit[ok & tm], "c": ci[ok & tm], "r": res[ok & tm], "vis": (b["field_home"].to_numpy()[ok & tm] == 0)}); pm_ot = pd.DataFrame({"p": pit[ok & ~tm], "c": ci[ok & ~tm], "r": res[ok & ~tm]})
    g_tm = pm_tm.groupby(["p", "c"])["r"].agg(["sum", "size"]); g_ot = pm_ot.groupby(["p", "c"])["r"].agg(["sum", "size"])
    # (1) spread of the predicted shift among pitchers with >= 150 balls outside Trustmark
    elig = n_out[n_out >= 150].index; pred = (mixf.loc[elig].to_numpy() * full_eff).sum(axis=1); sd_pred = pred.std()
    print(f"\nB1. PREDICTED TRUSTMARK wOBA SHIFT PER BALL IN PLAY from the pitcher's batted-ball mix (n = {len(elig)} pitchers with >= 150 balls elsewhere): mean {pred.mean():+.4f}, SD {sd_pred:.4f}, 10th-90th pct [{np.percentile(pred, 10):+.4f}, {np.percentile(pred, 90):+.4f}]")
    # in wOBA per batter faced: balls in play per BF ~ .68 (league), so per-BF = per-ball x .68 -> compare with the K%-allowed profile term (+.0115 wOBA per SD of K% allowed, per full exposure)
    print(f"   = SD {sd_pred * 0.68:.4f} wOBA per batter faced at full Trustmark exposure; the current profile's K%-allowed term is +.0115 per SD (also per PA at full exposure)")
    # (2) does it predict what visiting pitchers allowed at Trustmark vs elsewhere?  leave-one-pitcher-out cell effects
    rows = []
    for p in elig:
        if (p, 0) not in g_tm.index and p not in g_tm.index.get_level_values(0):
            continue
        t_rows = g_tm.loc[p] if p in g_tm.index.get_level_values(0) else None
        o_rows = g_ot.loc[p] if p in g_ot.index.get_level_values(0) else None
        if t_rows is None or o_rows is None:
            continue
        vis = pm_tm[pm_tm["p"] == p]["vis"].mean()
        if vis < 0.999:
            continue                                                         # visiting pitchers only (home pitchers carry roster fit)
        st = np.zeros(len(cells)); nt = np.zeros(len(cells)); so = np.zeros(len(cells)); no = np.zeros(len(cells)); st[t_rows.index] = t_rows["sum"]; nt[t_rows.index] = t_rows["size"]; so[o_rows.index] = o_rows["sum"]; no[o_rows.index] = o_rows["size"]
        loo = (S_tm - st) / np.maximum(N_tm - nt, 1) - (S_ot - so) / np.maximum(N_ot - no, 1)
        pred_p = float((mixf.loc[p].to_numpy() * loo).sum()); n_t = nt.sum(); rows.append({"p": p, "pred": pred_p, "n_tm": n_t, "act_tm": st.sum() / n_t, "act_ot": so.sum() / no.sum()})
    q = pd.DataFrame(rows); q["obs_diff"] = q["act_tm"] - q["act_ot"]; q = q[q["n_tm"] >= 8]; w = q["n_tm"].to_numpy()
    A = np.column_stack([np.ones(len(q)), q["pred"]]); beta = np.linalg.solve(A.T @ (A * w[:, None]), A.T @ (w * q["obs_diff"].to_numpy())); rng = np.random.default_rng(3); bs = []
    for _ in range(1000):
        ii = rng.integers(0, len(q), len(q)); Ab = A[ii]; wb = w[ii]; bs.append(np.linalg.solve(Ab.T @ (Ab * wb[:, None]), Ab.T @ (wb * q["obs_diff"].to_numpy()[ii]))[1])
    print(f"B2. VISITING PITCHERS (n = {len(q)}, >= 8 balls at Trustmark; leave-one-pitcher-out cell effects): observed Trustmark-minus-own-other-parks result on predicted shift: slope {beta[1]:.2f} (bootstrap SE {np.std(bs):.2f}; 1 = structural story exact, 0 = no information), "
          f"intercept {beta[0]:+.4f}; SD of predicted {q['pred'].std():.4f}")
    # (3) simple labelled pitcher types: fly-ball / ground-ball
    gb = mixf.loc[elig, [i for i, c in enumerate(cells) if c.startswith("ground")]].sum(axis=1); fb = mixf.loc[elig, [i for i, c in enumerate(cells) if c.startswith("fly") or c.startswith("pop")]].sum(axis=1)
    t = pd.DataFrame({"pred": pred, "gb_share": gb.values, "fb_share": fb.values}, index=elig); print("B3. predicted shift by pitcher type (terciles of ground-ball share among balls in play):")
    t["gb_tercile"] = pd.qcut(t["gb_share"], 3, labels=["low GB (fly-ball type)", "middle", "high GB"]); print(t.groupby("gb_tercile").agg(n=("pred", "size"), gb_share=("gb_share", "mean"), fb_pop_share=("fb_share", "mean"), predicted_shift_per_ball=("pred", "mean")).round(4).to_string())
    return q


def part_c(b, bt):
    tm = (b["tm"] == 1).to_numpy(); print("\nC1. OUTFIELD POSITIONING (feet; x = depth from the plate, z = width from the centre line), mean over balls in play where positions are recorded")
    rows = []
    for lab, m in (("Trustmark, home defence", tm & (b["field_home"].to_numpy() == 1)), ("Trustmark, visiting defence", tm & (b["field_home"].to_numpy() == 0)), ("other parks, home defence", ~tm & (b["field_home"].to_numpy() == 1)), ("other parks, visiting defence", ~tm & (b["field_home"].to_numpy() == 0))):
        g = b[m]; rows.append({"group": lab, "n": len(g), "LF depth": g["pos_lf_x"].mean(), "CF depth": g["pos_cf_x"].mean(), "RF depth": g["pos_rf_x"].mean(), "LF width": g["pos_lf_z"].mean(), "CF width": g["pos_cf_z"].mean(), "RF width": g["pos_rf_z"].mean()})
    print(pd.DataFrame(rows).round(1).to_string(index=False))
    # does depth matter for OF-fielded balls at Trustmark?  hit residual traits-only (cross-fit already in res_hit for hits non-HR) vs fielder depth, within Trustmark
    of = b[(b["of_fielded"] == 1) & (b["tm"] == 1) & (b["play_result"] != "home_run")].copy(); of["depth"] = np.select([of["fielded_by_pos"] == "LF", of["fielded_by_pos"] == "CF"], [of["pos_lf_x"], of["pos_cf_x"]], of["pos_rf_x"])
    of["depth_rel"] = of["depth"] - of.groupby("fielded_by_pos")["depth"].transform("mean"); of = of[of["depth"].notna()]
    X = np.column_stack([np.ones(len(of)), of["depth_rel"]]); beta = np.linalg.lstsq(X, of["res_hit"].to_numpy(), rcond=None)[0]; codes = pd.factorize(of["game_id"])[0]; bs = []
    rng = np.random.default_rng(2); by = pd.Series(np.arange(len(of))).groupby(codes).apply(np.array); ug = by.index.to_numpy()
    for _ in range(400):
        ii = np.concatenate([by[u] for u in rng.choice(ug, len(ug))]); bs.append(np.linalg.lstsq(X[ii], of["res_hit"].to_numpy()[ii], rcond=None)[0][1])
    print(f"C2. At Trustmark, outfield-fielded non-HR balls (n = {len(of)}): hit residual per +10 ft of fielder depth {beta[1] * 10:+.4f} (SE {np.std(bs) * 10:.4f}); SD of depth among these plays {of['depth_rel'].std():.1f} ft")
    for lab, m in (("shallow-ish balls (< 250 ft)", of["distance"] < 250), ("deep balls (>= 300 ft)", of["distance"] >= 300)):
        o = of[m]; Xo = np.column_stack([np.ones(len(o)), o["depth_rel"]]); bo = np.linalg.lstsq(Xo, o["res_hit"].to_numpy(), rcond=None)[0]; print(f"   {lab}: n {len(o)}, hit residual per +10 ft depth {bo[1] * 10:+.4f}")


def main():
    b = load(); print(f"{len(b):,} tracked balls in play (HR included); at Trustmark {int(b['tm'].sum()):,}")
    b["exp_woba"] = cross_fit(b, "woba"); b["res_woba"] = b["woba"] - b["exp_woba"]
    b["exp_hr"] = cross_fit(b, "is_hr"); b["res_hr"] = b["is_hr"] - b["exp_hr"]; b["exp_hit"] = cross_fit(b[["exit_speed", "angle", "direction", "left", "game_id", "hit_nohr", "is_hr"]].assign(hit_nohr=b["hit_nohr"]), "hit_nohr"); b["res_hit"] = b["hit_nohr"] - b["exp_hit"]
    tm = (b["tm"] == 1).to_numpy(); print(f"overall Trustmark effect per ball in play: wOBA {b.loc[tm, 'res_woba'].mean() - b.loc[~tm, 'res_woba'].mean():+.4f}, HR {b.loc[tm, 'res_hr'].mean() - b.loc[~tm, 'res_hr'].mean():+.4f}, non-HR hit {b.loc[tm, 'res_hit'].mean() - b.loc[~tm, 'res_hit'].mean():+.4f}")
    bt = Boot(b["game_id"].to_numpy()); part_a(b, bt); q = part_b(b, bt); part_c(b, bt)


if __name__ == "__main__":
    main()
