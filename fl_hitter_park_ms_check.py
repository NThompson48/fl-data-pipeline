"""
fl_hitter_park_ms_check.py -- include the MUD MONSTERS' hitters in the batted-ball residual park effect  (user 2026-10-10: "it's important to consider the mud monsters hitters too, because they play against a bunch of different defenses to help randomize that effect")
Why: every VISITING hitter at Trustmark faces the SAME defence and staff (Mississippi's); Mississippi's own hitters at home face 17 different visiting defences, and their road balls give each hitter a baseline.  So the visitors-only effect is confounded with one team's defence / pitching.
PART A  Do the two groups agree?  hitter-adjusted residual (wOBA - xwOBA - the hitter's own away mean) at Trustmark for VISITING hitters vs MISSISSIPPI hitters AT HOME, by launch-angle band, exit-speed band, batter side, with game-cluster bootstrap SEs, and the difference.
        'Away' reference (the zero of the scale) = every hitter's balls away from his own park, so both groups are measured against the same baseline.
PART B  Surface rebuilt from visitors + Mississippi hitters at home; held-out-season validation (fit on one season, score the other) for ALL units, for the MISSISSIPPI units (the ones the translation is for) and for visitors alone, vs the profile model and a level-only baseline.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import fl_eval_report as er
import fl_eval_weights as ew
import fl_hitter_park_surface as ps
import fl_hitter_park_surface_test as st

warnings.filterwarnings("ignore")
MS = "Mississippi Mud Monsters"; LAB = [(-91, 0, "LA < 0"), (0, 10, "LA 0-10"), (10, 20, "LA 10-20"), (20, 30, "LA 20-30"), (30, 40, "LA 30-40"), (40, 91, "LA > 40")]; EVB = [(0, 90, "EV < 90"), (90, 100, "EV 90-100"), (100, 130, "EV 100+")]


def prep():
    b = ps.load_balls(); b["res_adj"] = b["res"] - b["name_clean"].map(ps.hitter_baseline(b)).fillna(0.0); b["ms_home"] = ((b["batter_team"] == MS) & (b["vis"] == 0)).astype(int); b["away"] = (b["vis"] == 1).astype(int); return b


def part_a(b):
    codes, uniq = pd.factorize(b["game_id"]); rng = np.random.default_rng(6); W = [np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[codes] for _ in range(300)]; res = b["res_adj"].to_numpy()
    gv = ((b["tm"] == 1) & (b["vis"] == 1)).to_numpy(); gm = (b["ms_home"] == 1).to_numpy() & (b["tm"] == 1); ga = ((b["tm"] == 0) & (b["vis"] == 1)).to_numpy()

    def wm(w, m):
        return (w * m * res).sum() / (w * m).sum() if (w * m).sum() > 0 else np.nan

    def line(lab, mask):
        mv, mm, ma = gv & mask, gm & mask, ga & mask; ev = res[mv].mean() - res[ma].mean(); em = res[mm].mean() - res[ma].mean(); bs_v = [wm(w, mv) - wm(w, ma) for w in W]; bs_m = [wm(w, mm) - wm(w, ma) for w in W]; bs_d = [x - y for x, y in zip(bs_m, bs_v)]
        return {"group": lab, "visitors n": int(mv.sum()), "visitors effect": f"{ev:+.3f} ({np.nanstd(bs_v):.3f})", "MS home n": int(mm.sum()), "MS home effect": f"{em:+.3f} ({np.nanstd(bs_m):.3f})", "MS - visitors": f"{em - ev:+.3f} ({np.nanstd(bs_d):.3f})"}
    rows = [line("ALL balls in play", np.ones(len(b), bool))]
    for lo, hi, lab in LAB:
        rows.append(line(lab, ((b["la"] >= lo) & (b["la"] < hi)).to_numpy()))
    for lo, hi, lab in EVB:
        rows.append(line(lab, ((b["ev"] >= lo) & (b["ev"] < hi)).to_numpy()))
    rows.append(line("EV >= 90 and LA 10-40", ((b["ev"] >= 90) & (b["la"] >= 10) & (b["la"] < 40)).to_numpy())); rows.append(line("LHH", (b["left"] == 1).to_numpy())); rows.append(line("RHH", (b["left"] == 0).to_numpy()))
    rows.append(line("EV >= 90, LA < 10 (hard grounders), LHH", ((b["ev"] >= 90) & (b["la"] < 10) & (b["left"] == 1)).to_numpy())); rows.append(line("EV >= 90, LA < 10 (hard grounders), RHH", ((b["ev"] >= 90) & (b["la"] < 10) & (b["left"] == 0)).to_numpy()))
    print("PART A.  HITTER-ADJUSTED RESIDUAL (wOBA - xwOBA - own away mean), Trustmark minus away baseline: VISITING hitters (same MS defence) vs MISSISSIPPI hitters AT HOME (17 different defences); (bootstrap SE)\n" + pd.DataFrame(rows).to_string(index=False))


def fit_effect_pooled(train, feats, seed=0):
    fm = ps.FEATS[feats]; kw = dict(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=40, l2_regularization=1.0, random_state=seed)
    t1 = train[(train["tm"] == 1) & ((train["vis"] == 1) | (train["ms_home"] == 1))]; t0 = train[(train["tm"] == 0) & (train["vis"] == 1)]
    f1 = HistGradientBoostingRegressor(**kw).fit(t1[fm], t1["res_adj"]); f0 = HistGradientBoostingRegressor(**kw).fit(t0[fm], t0["res_adj"]); return lambda X: f1.predict(X[fm]) - f0.predict(X[fm])


def units_pooled(b, season, woba_base):
    v = b[(b["tm"] == 1) & ((b["vis"] == 1) | (b["ms_home"] == 1)) & (b["year"] == season)]; g = v.groupby("name_clean").agg(n=("res", "size"), t1=("res_adj", "mean"), woba=("woba", "mean"), ms=("ms_home", "max")); g = g[g["n"] >= st.MIN_N].copy(); g["t2"] = g["woba"] - g.index.map(woba_base).to_numpy(); return g.drop(columns="woba")


def part_b(b):
    non = b[b["tm"] == 0].groupby("name_clean")["woba"].agg(["sum", "size"]); lgw = b.loc[b["tm"] == 0, "woba"].mean(); woba_base = (non["sum"] + 100 * lgw) / (non["size"] + 100); allu = []
    for tr_y, te_y in ((2025, 2026), (2026, 2025)):
        train = b[b["year"] == tr_y]; Utr = units_pooled(b, tr_y, woba_base); Ute = units_pooled(b, te_y, woba_base); P = st.profile_table(train); Ute = Ute.join(P, how="left"); Utr = Utr.join(P, how="left"); ok = Utr["z_air"].notna(); out = Ute.copy()
        for tgt in ("t1", "t2"):
            w = Utr.loc[ok, "n"].to_numpy(float); y = Utr.loc[ok, tgt].to_numpy(float); out[f"{tgt}_NONE"] = np.average(y, weights=w)
            X = np.column_stack([np.ones(ok.sum()), Utr.loc[ok, ["z_air", "z_hr", "left"]].to_numpy(float)]); XtWX = X.T @ (X * w[:, None]); beta = np.linalg.solve(XtWX, X.T @ (w * y)); e = y - X @ beta; s2 = np.sum(w * e ** 2) / (len(y) - 4); se = np.sqrt(np.diag(np.linalg.inv(XtWX)) * s2 * np.mean(w)); t = beta / se
            sh = np.where(np.abs(t) > 1, np.maximum(0.0, 1 - 1 / t ** 2), 0.0); sh[0] = 1.0; Xt = np.column_stack([np.ones(len(out)), out[["z_air", "z_hr", "left"]].fillna({"z_air": 0, "z_hr": 0, "left": train["left"].mean()}).to_numpy(float)]); out[f"{tgt}_OLD"] = Xt @ (beta * sh)
        for f in ps.FEATS:
            eff = fit_effect_pooled(train, f); sh_, mu = ps.hitter_shift(train, eff, f); out[f"NEW_{f}"] = out.index.map(sh_).to_numpy(); out[f"NEW_{f}"] = out[f"NEW_{f}"].fillna(mu)
        allu.append(out)
    U = pd.concat(allu)
    for lab, sel in (("ALL units", np.ones(len(U), bool)), ("MISSISSIPPI hitters at home", (U["ms"] == 1).to_numpy()), ("VISITING hitters", (U["ms"] == 0).to_numpy())):
        Us = U[sel]
        if len(Us) < 15:
            print(f"\n{lab}: only {len(Us)} units"); continue
        for tgt, title in (("t1", "T1 park effect on same-quality contact"), ("t2", "T2 total change in wOBA on contact")):
            y = Us[tgt].to_numpy(float); n = Us["n"].to_numpy(float); base = Us[f"{tgt}_NONE"].to_numpy(float); a, b_ = ew.fit_variance_law(y - base, n); w = ew.inv_var_weights(n, a, b_); ids = Us.index.to_numpy()
            models = {"NONE (level only)": base, "OLD profile (Air%, power, left)": Us[f"{tgt}_OLD"].to_numpy(float)}
            for f in ps.FEATS:
                models[f"NEW surface: {f}"] = Us[f"NEW_{f}"].to_numpy(float)
            old = models["OLD profile (Air%, power, left)"]; rows = []
            for k, v in models.items():
                rm, ma = er.wmetrics(y, v, w); row = {"model": k, "RMSE": rm, "MAE": ma, "mean pred": np.average(v, weights=w)}
                if k != "OLD profile (Air%, power, left)":
                    rep = er.compare(y, old, v, w, ids, n=1000); row.update({"dRMSE vs OLD": rep["dRMSE"], "P(RMSE better)": rep["P(RMSE better)"], "dMAE vs OLD": rep["dMAE"], "P(MAE better)": rep["P(MAE better)"]})
                rows.append(row)
            print(f"\nPART B.  {lab} ({len(Us)} hitter-seasons), {title}; weighted mean target {np.average(y, weights=w):+.4f}\n" + er.table(rows, 5))


def main():
    b = prep(); print(f"{len(b):,} balls in play; visiting at Trustmark {int(((b['tm'] == 1) & (b['vis'] == 1)).sum()):,}; Mississippi hitters at home {int(b['ms_home'].sum()):,}; Mississippi hitters on the road {int(((b['batter_team'] == MS) & (b['vis'] == 1)).sum()):,}\n")
    part_a(b); part_b(b)


if __name__ == "__main__":
    main()
