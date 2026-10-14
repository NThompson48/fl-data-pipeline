"""
fl_baserunning_term_test.py -- MODEL-CHANGE REPORT for adding a shrunk BASERUNNING term to projected hitter WAR  (user 2026-10-10: "I think we can add the shrinkage term")
No second season exists, so the test is inside 2026 on the play-by-play rebuild (fl_baserunning_reliability.py): predict a runner's composite baserunning runs PER PA in one half of the season from the other half.
  NONE   = the league mean per PA (what the projection effectively assumes today: no baserunning term)
  SHRUNK = mean + (own half rate - mean) x PA / (PA + n0)
n0 is chosen on the ALTERNATING-GAMES split (grid) and the result is reported on the FIRST-HALF -> SECOND-HALF split (and, as a check, the reverse direction and the alternating split itself).  Weights: 1/(a + b/n), n = PA in the target half, law fitted on NONE's residuals and frozen;
weighted RMSE and MAE with a runner-cluster bootstrap.  Also prints the implied scale of the term in WAR per 600 PA on the PUBLISHED baserunning scale.
"""
import warnings

import numpy as np
import pandas as pd

import fl_baserunning_reliability as br
import fl_eval_report as er
import fl_eval_weights as ew

warnings.filterwarnings("ignore")


def comp(H, s):
    lg = br.RV
    return (lg["sb"] * H["sb" + s] + lg["cs"] * H["cs" + s] + lg["ebt"] * H["ebt_x" + s] + lg["oob"] * H["oob_x" + s] + lg["gdp"] * H["gdp_x" + s])


def halves(opp, att, eb, gd, pa, split, lg_sb_per_opp):
    H = br.table(opp, att, eb, gd, pa, split).reset_index(); A = H[H[split] == 0].set_index("rid"); B = H[H[split] == 1].set_index("rid"); J = A.join(B, lsuffix="_a", rsuffix="_b", how="inner")
    for s in ("_a", "_b"):
        J["runs" + s] = comp(J, s) - lg_sb_per_opp * J["sb_opp" + s]
    J = J[(J["pa_a"] >= 30) & (J["pa_b"] >= 30)].copy(); J["r_a"] = J["runs_a"] / J["pa_a"]; J["r_b"] = J["runs_b"] / J["pa_b"]; return J


def predict(J, n0, direction):
    a, b = ("_a", "_b") if direction == "AB" else ("_b", "_a"); mu = np.average(J["r" + a], weights=J["pa" + a]); return mu + (J["r" + a] - mu) * J["pa" + a] / (J["pa" + a] + n0), mu, J["r" + b].to_numpy(), J["pa" + b].to_numpy(float)


def main():
    e, nm = br.load(); opp, att, eb, gd, pa = br.build(e); T = br.table(opp, att, eb, gd, pa, None); lg_sb = (T["sb"].sum() * br.RV["sb"] + T["cs"].sum() * br.RV["cs"]) / T["sb_opp"].sum()
    Jalt = halves(opp, att, eb, gd, pa, "odd", lg_sb); Jlate = halves(opp, att, eb, gd, pa, "late", lg_sb); print(f"runners with >= 30 PA in both halves: alternating games {len(Jalt)}, first/second half {len(Jlate)}")
    grid = (50, 100, 150, 200, 300, 400, 600, 900, 1500); best = None; rows = []
    for n0 in grid:
        err = 0.0
        for d in ("AB", "BA"):
            p, mu, y, n = predict(Jalt, n0, d); base = np.full(len(y), mu); a, b = ew.fit_variance_law(y - base, n); w = ew.inv_var_weights(n, a, b); err += np.sum(w * (y - p) ** 2)
        rows.append((n0, err)); best = (n0, err) if best is None or err < best[1] else best
    print("n0 grid on the alternating-games split (frozen-weight SSE, both directions): " + ", ".join(f"{n}: {e_:.1f}" for n, e_ in rows) + f"  -> n0 = {best[0]} PA")
    N0 = best[0]; out = []
    for lab, J, d in (("first half -> second half", Jlate, "AB"), ("second half -> first half", Jlate, "BA"), ("alternating games (odd -> even)", Jalt, "AB"), ("alternating games (even -> odd)", Jalt, "BA")):
        p, mu, y, n = predict(J, N0, d); base = np.full(len(y), mu); a, b = ew.fit_variance_law(y - base, n); w = ew.inv_var_weights(n, a, b); rep = er.compare(y, base, p, w, J.index.to_numpy(), n=1000)
        out.append({"direction": lab, "runners": len(y), "RMSE none": rep["RMSE old"], "RMSE shrunk": rep["RMSE new"], "dRMSE": rep["dRMSE"], "RMSE 90%": rep["RMSE 90%"], "P(RMSE better)": rep["P(RMSE better)"], "MAE none": rep["MAE old"], "MAE shrunk": rep["MAE new"], "dMAE": rep["dMAE"], "P(MAE better)": rep["P(MAE better)"], "in-split n0": N0})
    print(f"\nHELD-OUT BASERUNNING per PA (runs, rebuilt composite), n0 = {N0}\n" + er.table(out, 5))
    p, mu, y, n = predict(Jlate, N0, "AB"); print(f"first -> second half: SD of the shrunk prediction {np.std(p) * 600:.2f} runs per 600 PA (rebuilt scale), SD of the target {np.std(y) * 600:.2f}; slope of actual on predicted {np.polyfit(p, y, 1)[0]:.2f}")
    print(f"published scale: SD of published baserunning ~ 8.6 runs per 600 PA; with n0 = {N0} and a 450-PA regular, shrink weight {450 / (450 + N0):.2f} -> SD of the projected term ~ {8.6 * 0.9 * 450 / (450 + N0) * np.sqrt(450 / (450 + N0)) / 11.98 if False else 8.6 * np.sqrt(450 / (450 + N0)) * 450 / (450 + N0) / 11.98:.2f} WAR per 600 PA")


if __name__ == "__main__":
    main()
