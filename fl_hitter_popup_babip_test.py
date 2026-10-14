"""
fl_hitter_popup_babip_test.py -- is Pop-Up rate (share of tracked batted balls with launch angle > 50 deg) related to BABIP, and does it help predict it?
(1) Same season (hitter-seasons >= 150 PA and >= 60 tracked balls, 2025-26): distribution, correlation with BABIP, with Air% / GB%, and BABIP on the NON-pop-up tracked balls (what is left when the near-automatic outs are removed).
(2) Reliability (random-half split of balls) -> shrinkage n0 for the shrunk deviation; persistence 2025 -> 2026 of PU% vs BABIP; PU%_25 vs BABIP_26.
(3) Forward 2025 -> 2026: production BABIP regression + b x shrunk PU% deviation, alone, and together with the Air% prior; reference = same regression with a fitted intercept (10-fold CV intercept in both); weighted RMSE and MAE with
    hitter-cluster bootstrap intervals (reporting standard).
"""
import warnings

import numpy as np
import pandas as pd

import fl_eval_report as er
import fl_hitter_air_babip_test as ab
import fl_hitter_depth3_test as d3
import fl_hitter_ev as hev
import fl_hitter_model_dev as dev

warnings.filterwarnings("ignore"); rng = np.random.default_rng(14)


def main():
    b = hev.load_bbe(); b["pu"] = (b["angle"] > 50).astype(int); b["air"] = (b["angle"] >= 10).astype(int); b["hit"] = b["play_result"].isin(["single", "double", "triple"]).astype(int); b["hr"] = (b["play_result"] == "home_run").astype(int)
    g = b.groupby(["name_clean", "year"]).agg(n=("pu", "size"), pu=("pu", "sum"), air=("air", "sum")).reset_index()
    nb = b[(b["hr"] == 0)]; h = nb.groupby(["name_clean", "year"]).agg(nip=("hit", "size"), hits=("hit", "sum")).reset_index(); h2 = nb[nb["pu"] == 0].groupby(["name_clean", "year"]).agg(nip_np=("hit", "size"), hits_np=("hit", "sum")).reset_index()
    g = g.merge(h, on=["name_clean", "year"], how="left").merge(h2, on=["name_clean", "year"], how="left"); g["pu_pct"] = g["pu"] / g["n"]; g["air_pct"] = g["air"] / g["n"]; g["babip_np"] = g["hits_np"] / g["nip_np"]
    p = dev.load(); t = p[(p["year"] >= 2025) & (p["pa"] >= 150)][["name_clean", "year", "pa", "babip"]].merge(g, on=["name_clean", "year"]); t = t[t["n"] >= 60]
    lg = {y: gg["pu"].sum() / gg["n"].sum() for y, gg in g.groupby("year")}
    print(f"(1) SAME SEASON: n = {len(t)} hitter-seasons; league pop-up rate {lg[2025]:.3f} / {lg[2026]:.3f} (2025 / 2026); hitter SD {t['pu_pct'].std():.3f}; mean BABIP {t['babip'].mean():.3f}")
    print(t[["pu_pct", "air_pct", "babip", "babip_np"]].corr().round(3).loc[["pu_pct"]].to_string())
    sl = np.polyfit(t["pu_pct"], t["babip"], 1)[0]; print(f"   slope: BABIP per +1 point of pop-up rate = {sl * 0.01:+.4f}")
    # reliability by random halves of balls
    hh = []
    for _ in range(20):
        x = b.assign(h=rng.integers(0, 2, len(b))).groupby(["name_clean", "year", "h"]).agg(n=("pu", "size"), pu=("pu", "sum")).reset_index(); x = x.pivot_table(index=["name_clean", "year"], columns="h", values=["n", "pu"]).dropna()
        x = x[(x[("n", 0)] >= 20) & (x[("n", 1)] >= 20)]; hh.append((np.corrcoef(x[("pu", 0)] / x[("n", 0)], x[("pu", 1)] / x[("n", 1)])[0, 1], (x[("n", 0)].mean() + x[("n", 1)].mean()) / 2))
    r, mh = np.mean([a for a, _ in hh]), np.mean([c for _, c in hh]); n0 = mh * (1 - r) / r; print(f"(2) RELIABILITY (random halves, >= 20 balls per half): r = {r:.2f} at {mh:.0f} balls per half -> n0 = {n0:.0f} tracked balls")
    a25 = t[t.year == 2025].set_index("name_clean"); a26 = t[t.year == 2026].set_index("name_clean"); j = a25.join(a26, lsuffix="_25", rsuffix="_26", how="inner")
    print(f"   2025 -> 2026 (n = {len(j)}): PU% persistence {j['pu_pct_25'].corr(j['pu_pct_26']):.2f}, BABIP {j['babip_25'].corr(j['babip_26']):.2f}, Air% {j['air_pct_25'].corr(j['air_pct_26']):.2f}; PU%_25 vs BABIP_26 {j['pu_pct_25'].corr(j['babip_26']):+.2f}; Air%_25 vs BABIP_26 {j['air_pct_25'].corr(j['babip_26']):+.2f}")
    # forward predictive
    q = p.copy(); q["woba"] = q["woba_n"]; d = d3.frame(q, "babip"); n0b = d3.fit_n0(d[d["t"].isin((2022, 2023, 2024))], 0.5, 3); te = d[d["t"] == 2025].copy(); te["base"] = d3.pred(te, n0b, 0.5, 3)
    s = g[g["year"] == 2025].drop_duplicates("name_clean").set_index("name_clean").reindex(te["name_clean"]); ok = s["n"].notna().to_numpy() & (s["n"].to_numpy() > 0); te = te[ok]; s = s[ok]
    te["pu"] = ((s["pu_pct"] - lg[2025]) * s["n"] / (s["n"] + n0)).to_numpy(); te["air"] = ((s["air_pct"] - g[g.year == 2025]["air"].sum() / g[g.year == 2025]["n"].sum()) * s["n"] / (s["n"] + ab.N0_AIR)).to_numpy()
    y = te["x_next"].to_numpy(); w = te["w"].to_numpy(); base = te["base"].to_numpy(); ids = te["name_clean"].to_numpy(); ref = ab.cvfit(y, base, None, w); rows = []
    for lab, cols in (("Pop-up% alone", ["pu"]), ("Air% alone", ["air"]), ("Pop-up% + Air%", ["pu", "air"])):
        X = te[cols].to_numpy(); bb = np.linalg.lstsq(np.column_stack([np.ones(len(y)), X]) * np.sqrt(w)[:, None], (y - base) * np.sqrt(w), rcond=None)[0]
        rows.append({"term": lab, "coef(s)": ", ".join(f"{c:+.3f}" for c in bb[1:]), **er.compare(y, ref, ab.cvfit(y, base, X, w), w, ids, n=500)})
    print(f"\n(3) FORWARD 2025 -> 2026 next-year BABIP (n = {len(y)}); old = production BABIP regression + intercept only\n" + er.table(rows, 5))


if __name__ == "__main__":
    main()
