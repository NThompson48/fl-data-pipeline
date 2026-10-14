"""
fl_pitcher_park_surface_test.py -- does the batted-ball residual-surface approach work for PITCHERS?  (user 2026-10-10: "quality of contact is so random but at least Air% seems predictable from pitcher traits")
The Trustmark ball set is the same for both sides (visiting pitchers' balls allowed = Mississippi hitters' balls; Mississippi pitchers' balls allowed at home = visiting hitters' balls), so the SAME surface applies: effect(x) = f_TM(x) - f_away(x) on hitter-adjusted residuals
(fl_hitter_park_surface.py / fl_hitter_park_ms_check.fit_effect_pooled; features exit speed, launch angle, batter side).  A pitcher's shift = the surface averaged over the balls HE ALLOWS.
PART 1  How much do pitchers differ?  SD of the shift across pitchers (per ball, per batter faced, runs/9 at half exposure); split-half reliability and season-to-season correlation of the shift; how well pitcher TRAITS predict it (pitcher-grouped 5-fold ridge on fastball VAA, velocity, IVB, release height, ..., the same trait set as fl_pitcher_fbshare_sticky).
PART 2  Held-out-season validation against pitchers' actual Trustmark results.  Unit = pitcher-season with >= 8 balls allowed at Trustmark (visiting pitchers + Mississippi pitchers at home).  Target T1 = mean hitter-adjusted residual (wOBA - xwOBA) allowed at Trustmark minus the pitcher's own AWAY mean
        (shrunk n0 150).  Everything (surface, history, traits) from the TRAIN season, target from the TEST season, both directions pooled.  Models: NONE (level only = the production global-only shift's structure); HIST (surface averaged over his own balls allowed, shrunk n0 60 toward the league mean, deviations
        added to the train level); TRAITS (ridge-predicted shift from traits, deviations + train level); HIST->TRAITS (his own history shrunk toward the trait prediction).  Weighted RMSE / MAE (inverse-variance weights from NONE, frozen), pitcher-cluster bootstrap.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

import fl_eval_report as er
import fl_eval_weights as ew
import fl_hitter_park_ms_check as ms
import fl_hitter_park_surface as ps
import fl_pitcher_fbshare_sticky as st

warnings.filterwarnings("ignore")
MS = "Mississippi Mud Monsters"; MIN_N = 8


def load():
    b = ms.prep()                                                                         # hitter-adjusted residuals res_adj, vis / ms_home flags (hitter perspective), features
    d = pd.read_parquet("2025_2026_pitches.parquet", columns=["game_id", "pa_id", "pitcher_id", "pitcher_team"]).drop_duplicates(["game_id", "pa_id"]); import fl_hitter_ros as hr
    pa = hr.Tables().pa[["game_id", "pa_id", "name_clean"]].drop_duplicates(["game_id", "pa_id"])
    raw = pd.read_parquet("2025_2026_pitches.parquet", columns=["game_id", "pa_id", "is_terminal_pitch", "pitcher_id", "pitcher_team"]); raw = raw[raw["is_terminal_pitch"].astype(str).isin(["1", "True", "true"])].drop_duplicates(["game_id", "pa_id"])
    b = b.merge(pa.merge(raw[["game_id", "pa_id", "pitcher_id", "pitcher_team"]], on=["game_id", "pa_id"], how="left"), on=["game_id", "pa_id", "name_clean"], how="left")
    b = b.dropna(subset=["pitcher_id"]).copy(); b["pid"] = b["pitcher_id"].astype(str); b["pit_vis"] = (b["pitcher_team"] != b["home_team"]).astype(int); b["pit_ms_home"] = ((b["pitcher_team"] == MS) & (b["pit_vis"] == 0)).astype(int); return b.reset_index(drop=True)


def pitcher_baseline(b, n0=150.0):
    a = b[(b["tm"] == 0) & (b["pit_vis"] == 1)].groupby("pid")["res_adj"].agg(["sum", "size"]); return a["sum"] / (a["size"] + n0)


def shifts(balls, effect, n0=60.0):
    x = balls[balls["tm"] == 0].copy(); x["eff"] = effect(x); g = x.groupby("pid")["eff"].agg(["mean", "size"]); mu = float(np.average(g["mean"], weights=g["size"])); return mu + (g["mean"] - mu) * g["size"] / (g["size"] + n0), g["mean"], g["size"], mu


def trait_pred(train_shift_raw, train_n, T, year, others_idx):
    """ridge from traits (train season) to the pitcher's mean effect; returns Series by pid (cross-fitted inside the train season for pitchers in it, full-fit for others)"""
    Ty = T.xs(year, level="year"); Ty.index = Ty.index.astype(str); feats = [c for c in Ty.columns if c != "n_pitches"]; D = Ty.join(pd.DataFrame({"y": train_shift_raw, "n": train_n}), how="inner"); D = D[(D["n"] >= 30) & (Ty.loc[D.index, "n_pitches"] >= 150)]
    X = Ty[feats].copy(); med = X.median(); X = X.fillna(med); mu_, sd_ = X.mean(), X.std(); X = (X - mu_) / sd_; pred = pd.Series(np.nan, index=X.index)
    tr_idx = D.index; Xt = X.loc[tr_idx]; yt = D["y"].to_numpy(); wt = D["n"].to_numpy(float)
    for tr, te in GroupKFold(5).split(Xt, yt, groups=tr_idx):
        m = RidgeCV(alphas=np.logspace(-1, 3, 12)).fit(Xt.iloc[tr], yt[tr], sample_weight=wt[tr]); pred.loc[tr_idx[te]] = m.predict(Xt.iloc[te])
    m = RidgeCV(alphas=np.logspace(-1, 3, 12)).fit(Xt, yt, sample_weight=wt); rest = pred.index[pred.isna()]; pred.loc[rest] = m.predict(X.loc[rest]); return pred, float(np.average(yt, weights=wt))


def part1(b):
    eff = ms.fit_effect_pooled(b, "evla_side"); sh, raw, n, mu = shifts(b, eff); ok = n >= 150; BF_PER_BIP = 1 / 0.671
    print(f"PART 1.  surface fitted on all balls (2025-26); league-mean effect {mu:+.4f} per ball in play")
    print(f"  pitchers with >= 150 non-Trustmark balls allowed: {int(ok.sum())}; SD of the shrunk shift {sh[ok].std():.4f} per ball in play; = {sh[ok].std() * 0.671:.4f} wOBA per batter faced; = {sh[ok].std() * 0.671 * 39.9 * 0.5:.3f} runs/9 at half exposure (compare the hitter shift SD .0117 per ball)")
    x = b[b["tm"] == 0].copy(); x["eff"] = eff(x); rng = np.random.default_rng(2); x["h"] = rng.integers(0, 2, len(x)); g = x.groupby(["pid", "h"])["eff"].agg(["mean", "size"]).unstack("h").dropna(); g = g[(g[("size", 0)] >= 40) & (g[("size", 1)] >= 40)]
    r = np.corrcoef(g[("mean", 0)], g[("mean", 1)])[0, 1]; rf = 2 * r / (1 + r); print(f"  split-half reliability of a pitcher's mean effect (random halves of his balls, >= 40 each; n = {len(g)}): r {r:.2f} -> full-length {rf:.2f}; implied n0 {(g[('size', 0)] + g[('size', 1)]).mean() * (1 - rf) / rf:.0f} balls")
    yy = x.groupby(["pid", "year"])["eff"].agg(["mean", "size"]).unstack("year").dropna(); yy = yy[(yy[("size", 2025)] >= 60) & (yy[("size", 2026)] >= 60)]; print(f"  season-to-season correlation of the mean effect (>= 60 balls both seasons, n = {len(yy)}): {np.corrcoef(yy[('mean', 2025)], yy[('mean', 2026)])[0, 1]:.2f}")
    T = st.trait_table(); y25 = b[b["year"] == 2025]; sh25, raw25, n25, mu25 = shifts(y25, ms.fit_effect_pooled(y25, "evla_side")); pr, lev = trait_pred(raw25, n25, T, 2025, None); j = pd.DataFrame({"raw": raw25, "n": n25, "pred": pr}).dropna(); j = j[j["n"] >= 30]
    print(f"  traits -> mean effect (2025 pitchers, cross-fitted by pitcher, n = {len(j)}): corr {np.corrcoef(j['pred'], j['raw'])[0, 1]:.2f}; weighted R^2 {1 - np.average((j['raw'] - j['pred']) ** 2, weights=j['n']) / np.average((j['raw'] - np.average(j['raw'], weights=j['n'])) ** 2, weights=j['n']):.2f}")


def units(b, season, pbase):
    v = b[(b["tm"] == 1) & ((b["pit_vis"] == 1) | (b["pit_ms_home"] == 1)) & (b["year"] == season)]; g = v.groupby("pid").agg(n=("res", "size"), t1=("res_adj", "mean"), ms=("pit_ms_home", "max")); g = g[g["n"] >= MIN_N].copy(); g["t1"] = g["t1"] - g.index.map(pbase).fillna(0.0).to_numpy(); return g


def part2(b):
    pbase = pitcher_baseline(b); T = st.trait_table(); allu = []
    for tr_y, te_y in ((2025, 2026), (2026, 2025)):
        train = b[b["year"] == tr_y]; Utr = units(b, tr_y, pbase); Ute = units(b, te_y, pbase); eff = ms.fit_effect_pooled(train, "evla_side"); sh, raw, n, mu = shifts(train, eff); pr, lev = trait_pred(raw, n, T, tr_y, None)
        level = np.average(Utr["t1"], weights=Utr["n"]); out = Ute.copy(); out["NONE"] = level
        h = out.index.map(sh).to_numpy(); out["HIST"] = level + np.where(np.isnan(h), 0.0, h - mu); tp = out.index.map(pr).to_numpy(); out["TRAITS"] = level + np.where(np.isnan(tp), 0.0, tp - lev)
        nn = out.index.map(n).to_numpy(); nn = np.where(np.isnan(nn), 0.0, nn); hraw = out.index.map(raw).to_numpy(); comb = np.where(np.isnan(tp), np.where(np.isnan(hraw), mu, hraw), tp); hr_ = np.where(np.isnan(hraw), comb, hraw)
        shr = comb + (hr_ - comb) * nn / (nn + 60.0); out["HIST_TO_TRAITS"] = level + (shr - np.where(np.isnan(tp), mu, lev)); allu.append(out)
    U = pd.concat(allu)
    for lab, sel in (("ALL units", np.ones(len(U), bool)), ("MISSISSIPPI pitchers at home", (U["ms"] == 1).to_numpy()), ("VISITING pitchers", (U["ms"] == 0).to_numpy())):
        Us = U[sel]
        if len(Us) < 15:
            print(f"\n{lab}: only {len(Us)} units"); continue
        y = Us["t1"].to_numpy(float); n = Us["n"].to_numpy(float); base = Us["NONE"].to_numpy(float); a, b_ = ew.fit_variance_law(y - base, n); w = ew.inv_var_weights(n, a, b_); ids = Us.index.to_numpy(); rows = []
        for k in ("NONE", "HIST", "TRAITS", "HIST_TO_TRAITS"):
            v = Us[k].to_numpy(float); rm, ma = er.wmetrics(y, v, w); row = {"model": k, "RMSE": rm, "MAE": ma, "SD of prediction": np.std(v)}
            if k != "NONE":
                rep = er.compare(y, base, v, w, ids, n=1000); row.update({"dRMSE vs NONE": rep["dRMSE"], "RMSE 90%": rep["RMSE 90%"], "P(RMSE better)": rep["P(RMSE better)"], "dMAE": rep["dMAE"], "P(MAE better)": rep["P(MAE better)"]})
            rows.append(row)
        print(f"\nPART 2.  {lab} ({len(Us)} pitcher-seasons; weighted mean target {np.average(y, weights=w):+.4f}; variance law a {a:.5f}, b {b_:.3f})\n" + er.table(rows, 5))


def main():
    b = load(); print(f"{len(b):,} balls with pitcher ids; Trustmark balls allowed by visiting pitchers {int(((b['tm'] == 1) & (b['pit_vis'] == 1)).sum()):,}, by Mississippi pitchers at home {int(((b['tm'] == 1) & (b['pit_ms_home'] == 1)).sum()):,}\n"); part1(b); part2(b)


if __name__ == "__main__":
    main()
