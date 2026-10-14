"""
fl_hitter_resid_babip_test.py -- how does the wOBA - xwOBA residual relate to BABIP?
(1) Same season (hitter-seasons >= 150 PA, 2025-26 Trackman): correlation of (wOBA - xwOBA) with BABIP and with the other outcomes.
(2) Persistence 2025 -> 2026: is the residual a stable skill (spray / speed / hands) or luck?  corr(resid_25, resid_26), corr(BABIP_25, BABIP_26), corr(resid_25, BABIP_26).
(3) Forward 2025 -> 2026: does last year's (shrunk) residual predict next year's BABIP beyond the production BABIP regression?  Alone, and together with the Air% prior; reference = the same regression with a fitted intercept
    (10-fold CV intercept in both, standard weights), weighted RMSE and MAE with hitter-cluster bootstrap intervals (reporting standard).  Shrinkage of the residual: n0 in {150, 300, 600} PA (sensitivity).
"""
import warnings

import numpy as np
import pandas as pd

import fl_eval_report as er
import fl_hitter_air_babip_test as ab
import fl_hitter_depth3_test as d3
import fl_hitter_model_dev as dev

warnings.filterwarnings("ignore")


def main():
    p = dev.load(); t = p[(p["year"] >= 2025) & (p["pa"] >= 150) & p["xwoba"].notna() & p["woba_raw"].notna()].copy(); t["resid"] = t["woba_raw"] - t["xwoba"]; t["air"] = t["air_n"] / t["bbe"]; t["hrr"] = t["hr"] / t["pa"]
    print(f"(1) SAME SEASON, hitter-seasons >= 150 PA with Trackman xwOBA: n = {len(t)}; mean resid {t['resid'].mean():+.4f}, SD {t['resid'].std():.4f}")
    print(t[["resid", "babip", "k", "bbr", "hrr", "air", "woba_raw", "xwoba"]].corr().round(2).loc[["resid"]].to_string())
    a25 = t[t.year == 2025].set_index("name_clean"); a26 = t[t.year == 2026].set_index("name_clean"); j = a25.join(a26, lsuffix="_25", rsuffix="_26", how="inner")
    print(f"\n(2) PERSISTENCE 2025 -> 2026 (n = {len(j)}): resid {j['resid_25'].corr(j['resid_26']):+.2f}, BABIP {j['babip_25'].corr(j['babip_26']):+.2f}, xwOBA {j['xwoba_25'].corr(j['xwoba_26']):+.2f}, wOBA {j['woba_raw_25'].corr(j['woba_raw_26']):+.2f}; "
          f"resid_25 -> BABIP_26 {j['resid_25'].corr(j['babip_26']):+.2f}; xwOBA_25 -> BABIP_26 {j['xwoba_25'].corr(j['babip_26']):+.2f}; BABIP_25 -> resid_26 {j['babip_25'].corr(j['resid_26']):+.2f}")
    # forward predictive test on the production BABIP regression
    q = p.copy(); q["woba"] = q["woba_n"]; d = d3.frame(q, "babip"); n0 = d3.fit_n0(d[d["t"].isin((2022, 2023, 2024))], 0.5, 3)
    te = d[d["t"] == 2025].copy(); te["base"] = d3.pred(te, n0, 0.5, 3); s25 = p[(p["year"] == 2025) & p["xwoba"].notna() & p["woba_raw"].notna()].drop_duplicates("name_clean").set_index("name_clean")
    te = te[te["name_clean"].isin(s25.index)].copy(); s = s25.reindex(te["name_clean"]); te["resid"] = (s["woba_raw"] - s["xwoba"]).to_numpy(); te["pa25"] = s["pa"].to_numpy()
    lgair = p[p.year == 2025]["air_n"].sum() / p[p.year == 2025]["bbe"].sum(); bb = s["bbe"].to_numpy(); an = s["air_n"].to_numpy()
    te["air"] = np.where(bb > 0, (an / np.where(bb > 0, bb, 1) - lgair) * bb / (bb + ab.N0_AIR), 0.0); te = te.dropna(subset=["resid", "air"])
    y = te["x_next"].to_numpy(); w = te["w"].to_numpy(); base = te["base"].to_numpy(); ids = te["name_clean"].to_numpy(); ref = ab.cvfit(y, base, None, w); rows = []
    print(f"\n(3) FORWARD 2025 -> 2026 next-year BABIP (n = {len(te)}): coefficient and held-out change vs the intercept-only reference")
    for n0r in (150, 300, 600):
        r = (te["resid"] * te["pa25"] / (te["pa25"] + n0r)).to_numpy()
        for lab, X in ((f"resid (n0 {n0r})", r[:, None]), (f"resid (n0 {n0r}) + Air%", np.column_stack([r, te["air"].to_numpy()]))):
            if n0r != 300 and "Air" in lab:
                continue
            new = ab.cvfit(y, base, X, w); b = np.linalg.lstsq(np.column_stack([np.ones(len(y)), X]) * np.sqrt(w)[:, None], (y - base) * np.sqrt(w), rcond=None)[0]
            rows.append({"term": lab, "coef(s)": ", ".join(f"{c:+.3f}" for c in b[1:]), **er.compare(y, ref, new, w, ids, n=500)})
    rows.append({"term": "Air% alone", "coef(s)": "", **er.compare(y, ref, ab.cvfit(y, base, te["air"].to_numpy()[:, None], w), w, ids, n=500)})
    print(er.table(rows, 5))


if __name__ == "__main__":
    main()
