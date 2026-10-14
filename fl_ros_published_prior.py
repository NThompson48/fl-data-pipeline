"""
fl_ros_published_prior.py -- should the pitcher ROS update start from the PUBLISHED preseason K% / BB% (the production projection model, incl. bridges / arsenal / stuff features) instead of the
Marcel history of the ROS model?  Test on 2026 only (the only season with a preseason projection built without that season's data), as-of days 63-105 (>= 2026-07-09: after the 2026-07-07 cut-off
of the frozen model's training targets, so the ROS targets are out of sample for the projection's coefficients).

Update with a published prior (deviation form, so the league level is the CURRENT one, like the Marcel update):
    dev0  = pre - weighted mean of pre over the population (weights = this season's BF to date)      (recentering, as the production projection does)
    x     = lg_now + (n_p dev0 + nu n_ytd (x_ytd - lg_now)) / (n_p + nu n_ytd),      n_p = phi (W_hist + n0)
W_hist + n0 is exactly the strength the Marcel update gives its own history, so phi = 1 means "same prior weight, better prior mean" -- zero free parameters.  phi in {.5 .. 2} is shown as sensitivity.
Pitchers without a published projection (no 2025 FL record; the 2026-style projection has no bridge for them) keep the Marcel history prior (= league).
Stuff / command terms: the published model already contains 2025 wh_sva_level / command_score, so adding the full term double counts; scales 0 / .25 / .5 / 1 of the production coefficient are compared.
"""
import warnings

import numpy as np
import pandas as pd

import fl_eval_weights as ew
import fl_pitcher_features as fp
import fl_pitcher_ros as pr
import fl_pitcher_ros_stuff as pst
import fl_ros_production as rp

warnings.filterwarnings("ignore")
rng = np.random.default_rng(88)
DAYS = list(range(63, 106, 7)); PHIS = (0.5, 1.0, 1.5, 2.0); TS = (0.0, 0.25, 0.5, 1.0)


def wr(y, p, w):
    return float(np.sqrt(np.average((y - p) ** 2, weights=w)))


def main():
    T = pr.Tables(); consts = pr.fit_constants(2025); raw = fp.load_fl_pitches_cached(); names = pst.idmap(raw); n0s = pst.reliability(raw)
    pub = pd.read_parquet("output/pitcher_pre2026_published.parquet").dropna(subset=["name_clean"]).drop_duplicates("name_clean").set_index("name_clean")
    B = rp.STUFF_B["clean"]; t0, _ = T.season_bounds(2026); out = []
    for d in DAYS:
        asof = t0 + pd.Timedelta(days=d); f, lg = pr.features(T, 2026, asof)
        if len(f) < 40:
            continue
        base = pr.predict(f, lg, consts); hist = pr.predict(f, lg, consts, use_ytd=False)
        sf = pst.stuff_features(raw, 2026, asof, names, n0s).reindex(f.index).fillna(0.0)
        for stat, key, pcol, ycol, bi, scol in (("k", "k", "pre_k", "k_ros", 0, "wh_sva"), ("bb", "bbr", "pre_bb", "bb_ros", 1, "command_score")):
            n0 = consts[key]["n0"]; lgn = lg[key]; x_ytd = np.where(f["n"] > 0, (f["k"] if stat == "k" else f["bb"]) / f["n"].replace(0, np.nan), 0.0)
            m0 = pub[pcol].reindex(f.index); has = m0.notna().to_numpy()
            dev0 = (m0 - np.average(m0[has], weights=np.maximum(f["n"][has], 1))).fillna(0.0).to_numpy()
            n_hist = hist[f"pa_eff_{key}"].to_numpy() + n0
            df = pd.DataFrame({"y": f[ycol].to_numpy(), "n": f["n_ros"].to_numpy(), "id": f.index.to_numpy(), "day": d, "stat": stat, "has_pub": has, "w_hist": hist[f"pa_eff_{key}"].to_numpy()})
            fade = n0 / (base[f"pa_eff_{key}"].to_numpy() + n0)
            for ts in TS:
                df[f"marcel|{ts}"] = base[key].to_numpy() + ts * B[bi] * sf[scol].to_numpy() * fade
            for phi in PHIS:
                n_p = phi * n_hist; pub_pred = lgn + (n_p * dev0 + f["n"].to_numpy() * (x_ytd - lgn)) / (n_p + f["n"].to_numpy())
                est = np.where(has, pub_pred, base[key].to_numpy())
                fade_p = n0 / (phi * hist[f"pa_eff_{key}"].to_numpy() + f["n"].to_numpy() + n0)
                for ts in TS:
                    df[f"pub{phi}|{ts}"] = est + ts * B[bi] * sf[scol].to_numpy() * fade_p
            out.append(df)
    D = pd.concat(out, ignore_index=True)
    for stat, lab in (("k", "K%"), ("bb", "BB%")):
        d = D[D["stat"] == stat].reset_index(drop=True); ref = "marcel|1.0"
        w = ew.weights_from_base(d["y"].to_numpy(), d[ref].to_numpy(), d["n"].to_numpy())[0]; y = d["y"].to_numpy(); ids = d["id"].to_numpy()
        by = pd.Series(np.arange(len(d))).groupby(ids).apply(np.array); ug = by.index.to_numpy()
        print(f"\n{'=' * 100}\nPITCHERS {lab}: ROS as-of days {DAYS[0]}-{DAYS[-1]} of 2026 ({len(d)} pitcher-dates, {d['has_pub'].mean():.0%} with a published preseason projection)\n{'=' * 100}")
        cols = [c for c in d.columns if "|" in c]; rows = []
        for c in cols:
            r = {"model": c, "RMSE": wr(y, d[c].to_numpy(), w), "RMSE (with published)": wr(y[d["has_pub"]], d[c].to_numpy()[d["has_pub"]], w[d["has_pub"]])}
            if c != ref:
                dd = []
                for _ in range(300):
                    ii = np.concatenate([by[u] for u in rng.choice(ug, len(ug))]); dd.append(wr(y[ii], d[c].to_numpy()[ii], w[ii]) - wr(y[ii], d[ref].to_numpy()[ii], w[ii]))
                dd = np.array(dd); r["vs production (marcel|1.0)"] = np.mean(dd); r["90% interval"] = f"[{np.percentile(dd, 5):+.5f}, {np.percentile(dd, 95):+.5f}]"; r["P(better)"] = float((dd < 0).mean())
            rows.append(r)
        print(pd.DataFrame(rows).round(5).to_string(index=False))
        thin = (d["w_hist"] < d.loc[d["has_pub"], "w_hist"].median()).to_numpy()
        print("  by 2025 history: thin / rich (RMSE) production vs pub1.0|0.5: " + ", ".join(
            f"{nm}: {wr(y[m & d['has_pub'].to_numpy()], d[ref].to_numpy()[m & d['has_pub'].to_numpy()], w[m & d['has_pub'].to_numpy()]):.5f} vs {wr(y[m & d['has_pub'].to_numpy()], d['pub1.0|0.5'].to_numpy()[m & d['has_pub'].to_numpy()], w[m & d['has_pub'].to_numpy()]):.5f}" for nm, m in (("thin", thin), ("rich", ~thin))))


if __name__ == "__main__":
    main()
