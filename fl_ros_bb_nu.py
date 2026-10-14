"""
fl_ros_bb_nu.py -- follow-up to fl_ros_robust.py: hitter BB% preferred the lowest in-season weight tried (nu = 0.5, 100% of bootstrap draws) and the 21-day half-life.  Extend the nu grid downward, cross it with recency
weighting, and test it forward (choose on one season, score on the other).  Hitter BB% only (wOBA and K% were best at nu ~ 1, see the cutoff/nu section of the robustness study).
"""
import warnings

import numpy as np
import pandas as pd

import fl_eval_weights as ew
import fl_hitter_ros as hr

warnings.filterwarnings("ignore")
rng = np.random.default_rng(31)
C8 = [round(x, 2) for x in np.arange(0.10, 0.801, 0.10)]
NUS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0)
HLS = (None, 90, 45, 21)


def wr(y, p, w):
    return float(np.sqrt(np.average((y - p) ** 2, weights=w)))


def main():
    T = hr.Tables(); rows = {}
    for hl in HLS:
        for y in (2025, 2026):
            t0, t1 = T.season_bounds(y)
            for c in C8:
                f, lg = hr.features(T, y, t0 + (t1 - t0) * c, min_ros_pa=30, half_life=hl)
                for nu in NUS:
                    o = hr.predict(f, lg, nu=nu)
                    rows[(hl, nu, y, c)] = pd.DataFrame({"y": f["bb_ros"].to_numpy(), "n": f["n_ros"].to_numpy(), "id": (f.index + "|" + str(y)).to_numpy(), "p": o["bbr_final"].to_numpy(), "year": y, "cut": c})
    ref = pd.concat([rows[(None, 1.0, y, c)] for y in (2025, 2026) for c in C8], ignore_index=True)
    w = ew.weights_from_base(ref["y"].to_numpy(), ref["p"].to_numpy(), ref["n"].to_numpy())[0]; y = ref["y"].to_numpy(); yr = ref["year"].to_numpy(); ids = ref["id"].to_numpy()

    def P(hl, nu):
        return np.concatenate([rows[(hl, nu, yy, c)]["p"].to_numpy() for yy in (2025, 2026) for c in C8])

    tab = pd.DataFrame({f"hl={hl}": {nu: wr(y, P(hl, nu), w) for nu in NUS} for hl in HLS}); tab.index.name = "nu"
    print("HITTER BB% ROS RMSE (production = nu 1.0, no recency)\n", tab.round(5).to_string())
    for yy in (2025, 2026):
        m = yr == yy
        print(f"  {yy} only:\n", pd.DataFrame({f"hl={hl}": {nu: wr(y[m], P(hl, nu)[m], w[m]) for nu in NUS} for hl in HLS}).round(5).to_string())
    # forward selection: choose (hl, nu) on one season, score on the other
    cells = [(hl, nu) for hl in HLS for nu in NUS]
    base = P(None, 1.0)
    for tr, te in ((2025, 2026), (2026, 2025)):
        mtr, mte = yr == tr, yr == te
        best = min(cells, key=lambda k: wr(y[mtr], P(*k)[mtr], w[mtr]))
        pb = P(*best); by = pd.Series(np.arange(len(y))[mte]).groupby(ids[mte]).apply(np.array); ug = by.index.to_numpy(); d = []
        for _ in range(500):
            ii = np.concatenate([by[u] for u in rng.choice(ug, len(ug))]); d.append(wr(y[ii], pb[ii], w[ii]) - wr(y[ii], base[ii], w[ii]))
        d = np.array(d)
        print(f"  select on {tr} -> {best}; scored on {te}: change vs production {np.mean(d):+.5f} [{np.percentile(d, 5):+.5f}, {np.percentile(d, 95):+.5f}]")
    # which of the two matters: is nu alone enough?
    for lab, k in (("nu .3 alone", (None, 0.3)), ("nu .5 alone", (None, 0.5)), ("hl 21 alone", (21, 1.0)), ("hl 21 + nu .5", (21, 0.5))):
        pb = P(*k); by = pd.Series(np.arange(len(y))).groupby(ids).apply(np.array); ug = by.index.to_numpy(); d = []
        for _ in range(500):
            ii = np.concatenate([by[u] for u in rng.choice(ug, len(ug))]); d.append(wr(y[ii], pb[ii], w[ii]) - wr(y[ii], base[ii], w[ii]))
        d = np.array(d); print(f"  {lab}: change vs production {np.mean(d):+.5f} [{np.percentile(d, 5):+.5f}, {np.percentile(d, 95):+.5f}]")


if __name__ == "__main__":
    main()
