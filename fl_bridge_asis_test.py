"""
fl_bridge_asis_test.py -- for players with thin FL history AND an other-league record, is the production projection better than the BRIDGE AS IS (the other-league translation alone,
ignoring the FL data)?  (user 2026-10-13: "only as good as the league average is fine, is it better than just using the bridge as is?")

HITTERS: FL debut hitters 2023-25 with MiLB history (fl_hitter_bridge_test pairs) who play >= 20 PA in their second FL season (the fl_hitter_bridge_prior_test sample); target = second-season rate
  minus that season's FL league mean.  Arms: PRODUCTION-like = debut FL season shrunk to the league mean with the production n0 (the hitter model has no bridge term);
  BRIDGE AS IS = the out-of-fold bridge prediction (leave-one-debut-class-out); COMBINED = debut season shrunk toward the bridge (the 10-09 candidate, not adopted).
  wOBA = 0.5 direct + 0.5 component-built in every arm.
PITCHERS: fold 2 of the production harness (features 2025 + 1H2026 -> 2H2026), pitchers with an other-league match (k_bridge_tbf / bb_bridge_tbf > 0), split by FL BF in the window;
  arms: PRODUCTION (frozen K% / BB% model) vs BRIDGE AS IS (fl_k_rate_bridge / fl_bb_rate_bridge).
Scoring: level-free, frozen standard weights (from production's residuals), player bootstrap, weighted RMSE and MAE.
"""
import contextlib
import io
import warnings

import numpy as np
import pandas as pd

import fl_eval_report as er
import fl_eval_weights as ew

warnings.filterwarnings("ignore")
ROWS = []


def cmp(group, stat, y, n, prod, alts, ids):
    ok = np.isfinite(y) & np.isfinite(prod) & np.all([np.isfinite(a) for a in alts.values()], axis=0)
    y, n, prod, ids = y[ok], n[ok], prod[ok], ids[ok]; alts = {k: v[ok] for k, v in alts.items()}
    w = ew.weights_from_base(y, prod, n)[0]; c = lambda p: p + np.average(y - p, weights=w)
    for k, v in alts.items():
        r = er.compare(y, c(v), c(prod), w, ids, n=1000)
        ROWS.append({"group": group, "stat": stat, "n": int(len(y)), "alternative": k, "RMSE alternative": r["RMSE old"], "RMSE production": r["RMSE new"], "dRMSE (prod - alt)": r["dRMSE"], "RMSE 90%": r["RMSE 90%"],
                     "P(production better)": r["P(RMSE better)"], "MAE alternative": r["MAE old"], "MAE production": r["MAE new"], "P(production better, MAE)": r["P(MAE better)"]})


def hitters():
    import fl_hitter_bridge_test as bt
    from fl_hitter_bridge_prior_test import BETA, N0
    with contextlib.redirect_stdout(io.StringIO()):
        res = bt.main()
    h = pd.read_parquet("output/fl_hitter_panel.parquet"); h["nm"] = h["name_clean"].map(bt.norm); h["k"] = h["so"] / h["pa"]; h["bbr"] = h["bb"] / h["pa"]; h["hrr"] = h["hr"] / h["pa"]
    lg = {s: h.dropna(subset=[s]).groupby("year").apply(lambda g: np.average(g[s], weights=g["pa"])) for s in bt.STATS}
    nxt = h[["nm", "year", "pa"] + list(bt.STATS)].copy(); nxt["year"] = nxt["year"] - 1; nxt = nxt.drop_duplicates(["nm", "year"]).set_index(["nm", "year"])
    r0 = res["woba"]["dd"]; keep = np.array([i for i, (nm, yr) in enumerate(zip(r0["nm"], r0["year"])) if yr <= 2025 and (nm, yr) in nxt.index and nxt.loc[(nm, yr), "pa"] >= 20])
    names = r0["nm"].to_numpy()[keep]; yrs = r0["year"].to_numpy()[keep]; n_deb = r0["pa"].to_numpy(float)[keep]; n_next = np.array([nxt.loc[(a, b), "pa"] for a, b in zip(names, yrs)], float)
    P, B, C, Y = {}, {}, {}, {}
    for s in bt.STATS:
        q = res[s]; idx = {nm: i for i, nm in enumerate(q["dd"]["nm"])}; sel = np.array([idx[nm] for nm in names]); dev = q["dd"][f"y_{s}"].to_numpy()[sel]; prior = q["P"]["M2 pooled slope"][sel]; sh = n_deb / (n_deb + N0[s])
        P[s] = sh * dev; B[s] = prior; C[s] = sh * dev + (1 - sh) * prior; Y[s] = np.array([nxt.loc[(a, b), s] - lg[s].loc[b + 1] for a, b in zip(names, yrs)])
    for arm in (P, B, C):
        arm["woba"] = 0.5 * arm["woba"] + 0.5 * sum(BETA[s] * arm[s] for s in BETA)
    print(f"hitters: {len(keep)} debut hitters with MiLB history and >= 20 PA in their second FL season; debut-season PA median {np.median(n_deb):.0f}")
    for s, lab in (("woba", "wOBA"), ("k", "K%"), ("bbr", "BB%")):
        cmp("hitters: 2nd FL season, debut-season history + MiLB", lab, Y[s], n_next, P[s], {"bridge as is": B[s], "debut season shrunk toward the bridge": C[s]}, names)


def pitchers():
    import fl_pitcher_features as fp
    import fl_position_player_flag as ppf
    import fl_posplayer_refit_check as chk
    import fl_production_projection as pp
    raw = ppf.exclude(fp.load_fl_pitches_cached())
    with contextlib.redirect_stdout(io.StringIO()):
        m_k, m_bb, n0_k, n0_bb, n0b, rho = pp.fit_frozen_model(raw); dk, db = chk.fold2(raw, m_k, m_bb, n0b)
    for d, ac, br, tb, lab in ((dk, "K_rate_actual", "fl_k_rate_bridge", "k_bridge_tbf", "K%"), (db, "BB_rate_actual", "fl_bb_rate_bridge", "bb_bridge_tbf", "BB%")):
        d = d.dropna(subset=["pred", ac, br]).copy()
        if tb not in d.columns:
            print(f"  {tb} not in the fold-2 frame; columns: {[c for c in d.columns if 'bridge' in c]}"); return
        d = d[d[tb].fillna(0) > 0]
        for g, m in (("pitchers with a bridge match, < 150 FL BF", d["PA"] < 150), ("pitchers with a bridge match, 150+ FL BF", d["PA"] >= 150), ("pitchers with a bridge match, all", d["PA"] >= 0)):
            x = d[m]
            if len(x) >= 15:
                cmp(g, lab, x[ac].to_numpy(float), x["PA_test"].to_numpy(float), x["pred"].to_numpy(float), {"bridge as is": x[br].to_numpy(float)}, x["pitcher_id"].astype(str).to_numpy())
        print(f"pitchers {lab}: {len(d)} with a bridge match; < 150 FL BF {int((d['PA'] < 150).sum())}")


def main():
    hitters(); pitchers()
    pd.set_option("display.width", 250); R = pd.DataFrame(ROWS); R.to_csv("output/bridge_asis_test.csv", index=False)
    print("\nPRODUCTION vs BRIDGE AS IS (level-free; dRMSE < 0 = production better)\n" + R.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
