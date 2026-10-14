"""
fl_hitter_oppo_babip_test.py -- is Oppo% (share of tracked batted balls hit to the opposite-field third, spray direction beyond 15 deg toward the opposite side) related to BABIP, and does it help predict it?
Direction convention checked on the data: right-handed batters average -7.6 deg (pull = negative), left-handed +8.2 (pull = positive); pull-signed direction s = -direction (RHB) / +direction (LHB); oppo: s < -15, pull: s > 15.
Same tests as fl_hitter_popup_babip_test.py: (1) same season; (2) reliability and 2025 -> 2026 persistence; (3) forward 2025 -> 2026 prediction of BABIP beyond the production BABIP regression, alone and combined with the Air% and
Pop-up% priors (reference = same regression with a fitted intercept; 10-fold CV intercept in both; weighted RMSE and MAE with hitter-cluster bootstrap intervals).
"""
import warnings

import numpy as np
import pandas as pd

import fl_eval_report as er
import fl_hitter_air_babip_test as ab
import fl_hitter_depth3_test as d3
import fl_hitter_ev as hev
import fl_hitter_model_dev as dev

warnings.filterwarnings("ignore"); rng = np.random.default_rng(15)


def main():
    b = hev.load_bbe(); b = b[b["batter_side"].isin(["left", "right"]) & b["direction"].between(-90, 90)].copy()
    b["s"] = np.where(b["batter_side"] == "right", -b["direction"], b["direction"]); b["oppo"] = (b["s"] < -15).astype(int); b["pull"] = (b["s"] > 15).astype(int)
    b["pu"] = (b["angle"] > 50).astype(int); b["air"] = (b["angle"] >= 10).astype(int)
    g = b.groupby(["name_clean", "year"]).agg(n=("oppo", "size"), oppo=("oppo", "sum"), pull=("pull", "sum"), pu=("pu", "sum"), air=("air", "sum")).reset_index()
    for c in ("oppo", "pull", "pu", "air"):
        g[c + "_pct"] = g[c] / g["n"]
    p = dev.load(); t = p[(p["year"] >= 2025) & (p["pa"] >= 150)][["name_clean", "year", "pa", "babip", "k", "bbr"]].merge(g, on=["name_clean", "year"]); t = t[t["n"] >= 60]
    lg = {y: {c: gg[c].sum() / gg["n"].sum() for c in ("oppo", "pull", "pu", "air")} for y, gg in g.groupby("year")}
    print(f"(1) SAME SEASON: n = {len(t)} hitter-seasons; league Oppo% {lg[2025]['oppo']:.3f} / {lg[2026]['oppo']:.3f} (2025 / 2026), Pull% {lg[2025]['pull']:.3f}; hitter SD of Oppo% {t['oppo_pct'].std():.3f}")
    print(t[["oppo_pct", "pull_pct", "air_pct", "pu_pct", "k", "babip"]].corr().round(3).loc[["oppo_pct"]].to_string()); print(f"   slope: BABIP per +1 point of Oppo% = {np.polyfit(t['oppo_pct'], t['babip'], 1)[0] * 0.01:+.4f}")
    hh = []
    for _ in range(20):
        x = b.assign(h=rng.integers(0, 2, len(b))).groupby(["name_clean", "year", "h"]).agg(n=("oppo", "size"), o=("oppo", "sum")).reset_index().pivot_table(index=["name_clean", "year"], columns="h", values=["n", "o"]).dropna()
        x = x[(x[("n", 0)] >= 20) & (x[("n", 1)] >= 20)]; hh.append((np.corrcoef(x[("o", 0)] / x[("n", 0)], x[("o", 1)] / x[("n", 1)])[0, 1], (x[("n", 0)].mean() + x[("n", 1)].mean()) / 2))
    r, mh = np.mean([a for a, _ in hh]), np.mean([c for _, c in hh]); n0 = mh * (1 - r) / r; print(f"(2) RELIABILITY (random halves): r = {r:.2f} at {mh:.0f} balls per half -> n0 = {n0:.0f} tracked balls")
    a25 = t[t.year == 2025].set_index("name_clean"); a26 = t[t.year == 2026].set_index("name_clean"); j = a25.join(a26, lsuffix="_25", rsuffix="_26", how="inner")
    print(f"   2025 -> 2026 (n = {len(j)}): Oppo% persistence {j['oppo_pct_25'].corr(j['oppo_pct_26']):.2f}, BABIP {j['babip_25'].corr(j['babip_26']):.2f}; Oppo%_25 vs BABIP_26 {j['oppo_pct_25'].corr(j['babip_26']):+.2f}")
    q = p.copy(); q["woba"] = q["woba_n"]; d = d3.frame(q, "babip"); n0b = d3.fit_n0(d[d["t"].isin((2022, 2023, 2024))], 0.5, 3); te = d[d["t"] == 2025].copy(); te["base"] = d3.pred(te, n0b, 0.5, 3)
    s = g[g["year"] == 2025].drop_duplicates("name_clean").set_index("name_clean").reindex(te["name_clean"]); ok = s["n"].notna().to_numpy() & (s["n"].to_numpy() > 0); te = te[ok]; s = s[ok]
    for c, nn in (("oppo", n0), ("pu", 135.0), ("air", ab.N0_AIR)):
        te[c] = ((s[c + "_pct"] - lg[2025][c]) * s["n"] / (s["n"] + nn)).to_numpy()
    y = te["x_next"].to_numpy(); w = te["w"].to_numpy(); base = te["base"].to_numpy(); ids = te["name_clean"].to_numpy(); ref = ab.cvfit(y, base, None, w); rows = []
    for lab, cols in (("Oppo% alone", ["oppo"]), ("Air% alone", ["air"]), ("Pop-up% alone", ["pu"]), ("Oppo% + Air%", ["oppo", "air"]), ("Oppo% + Air% + Pop-up%", ["oppo", "air", "pu"])):
        X = te[cols].to_numpy(); bb = np.linalg.lstsq(np.column_stack([np.ones(len(y)), X]) * np.sqrt(w)[:, None], (y - base) * np.sqrt(w), rcond=None)[0]
        rows.append({"term": lab, "coef(s)": ", ".join(f"{c:+.2f}" for c in bb[1:]), **er.compare(y, ref, ab.cvfit(y, base, X, w), w, ids, n=500)})
    print(f"\n(3) FORWARD 2025 -> 2026 next-year BABIP (n = {len(y)}); old = production BABIP regression + intercept only\n" + er.table(rows, 5))


if __name__ == "__main__":
    main()
