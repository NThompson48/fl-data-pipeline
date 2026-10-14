"""
fl_2027_preregistered_checks.py -- checks FIXED BEFORE 2027 outcomes exist (pre-registered 2026-10-13, user: "yep").  Nothing here may be re-tuned on 2027 data.

  python fl_2027_preregistered_checks.py --freeze     (run ONCE, now: fits the two fixed coefficients on 2025 + 2026 and writes output/preregistered_2027.json)
  python fl_2027_preregistered_checks.py --run        (after the 2027 season: evaluates every check against the frozen file; not runnable until 2027 data are loaded)

Success rule for every model-change check (the user's reporting standard): held-out weighted RMSE AND weighted MAE both better, frozen standard inverse-variance weights,
player bootstrap P(better) >= .90.  Calibration checks: stated below.
"""
import json
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
OUT = "output/preregistered_2027.json"

CHECKS = {
    "C1 pitcher K% + fastball velocity": {"type": "model change (candidate, NOT in production)", "form": "proj_K += b x (fb_velo_shrunk - population mean); constant; single term",
        "trait": "mean effective velocity of four-seam / sinker in the feature window, shrunk by n / (n + 8.4) fastballs", "coefficient": "frozen, fit on ROS 2025 + ROS 2026 pooled (see 'frozen')",
        "samples": ["2026 -> 2027 preseason (production K%)", "2027 in-season, as of mid-season -> rest of 2027"], "success": "RMSE and MAE both better, P(better) >= .90, in both samples"},
    "C2 hitter BB% + barrel%": {"type": "model change (candidate, NOT in production)", "form": "proj_BB += b x (barrel_pct_shrunk - population mean); constant; single term",
        "trait": "barrels per tracked ball in play in the feature window, shrunk by the split-half n0", "coefficient": "frozen, fit on ROS 2025 + ROS 2026 pooled",
        "samples": ["2026 -> 2027 preseason", "2027 in-season, as of mid-season"], "success": "RMSE and MAE both better, P(better) >= .90, in both samples"},
    "C3 hitter K% bridge prior (IN production since 2026-10-13)": {"type": "confirmation", "sample": "the 2026 FL debut class with a MiLB record: second-season (2027) K%",
        "compare": "production (bridge prior) vs the same regression shrinking to the league mean", "success": "RMSE and MAE better, P(better) >= .90; if worse with P(worse) >= .90, remove"},
    "C4 thin-history hitter survivorship": {"type": "decision", "sample": "2026 -> 2027 hitters with effective history < 100 PA, PA-weighted target",
        "test": "mean error (actual - projected wOBA); pooled with the four earlier transitions", "success": "apply the +.0046 step if the 5-transition interval excludes 0"},
    "C5 pitcher P(WAR>0) calibration by projected IP": {"type": "calibration", "groups": ["< 22 projected IP", "22-45", "45+"],
        "success": "predicted mean inside each group's 90% Wilson interval; the < 22 group was ~7 points optimistic in 2026"},
    "C6 hitter WAR range coverage": {"type": "calibration", "test": "share of realised 2027 WAR (batting + position + replacement) inside the 50 / 80 / 90% ranges",
        "success": "each within 2 binomial SE of nominal"},
    "C7 baserunning persistence": {"type": "data", "test": "2026 -> 2027 correlation of published baserunning runs per PA (>= 150 PA both years) and the shrunk term's held-out RMSE",
        "success": "decides whether baserunning returns to WAR (currently out of production)"},
    "C8 Trustmark surfaces": {"type": "validation", "test": "2027 balls at Trustmark (visitors + Mississippi at home): observed (wOBA - xwOBA) and HR vs the 2025-26 surface predictions, per hitter and overall",
        "success": "calibration slope interval contains 1; level inside the 2025-26 bootstrap range"},
    "C9 fielding / position with two tracked seasons": {"type": "data", "test": "2026 -> 2027 persistence of fielding runs by position (infield if the data exist) and of position runs",
        "success": "decides whether fielding returns to WAR (currently out of production)"},
}


def freeze():
    import pickle
    import fl_error_analysis_common as ea
    from fl_error_analysis_select import load
    frozen = {}
    for kind, tg, trait in (("p", "k", "fb_velo"), ("h", "bb", "barrel_pct")):
        st, S, n0t, _ = load(kind); e, w, x = [], [], []
        for s in ("ROS25", "ROS26"):
            F = S[s]["targets"][tg]; v, n = S[s]["traits"][trait]; xs = ea.shrink(v, n, st.loc[trait, "n0"]).reindex(F.index)
            mu = np.average(xs.dropna(), weights=F.loc[xs.notna(), "w"]); xs = xs.fillna(mu)
            e.append(F["e"].to_numpy()); w.append(F["w"].to_numpy()); x.append(xs.to_numpy() - mu)
        e, w, x = map(np.concatenate, (e, w, x)); b = float(np.sum(w * x * e) / np.sum(w * x * x))
        frozen[f"{kind}:{tg}:{trait}"] = {"b": b, "n0_trait": float(st.loc[trait, "n0"]), "n_rows": int(len(e))}
        print(f"{kind} {tg} {trait}: b = {b:+.5f} per unit of the shrunk trait (n = {len(e)}; trait n0 {st.loc[trait, 'n0']:.1f})")
    json.dump({"frozen_on": "2026-10-13", "checks": CHECKS, "frozen": frozen}, open(OUT, "w"), indent=1); print("saved", OUT)


def run():
    spec = json.load(open(OUT))
    raise SystemExit("2027 outcomes are not loaded yet. When they are: implement each check strictly as written in " + OUT + " (frozen " + spec["frozen_on"] + ").")


if __name__ == "__main__":
    freeze() if "--freeze" in sys.argv else run()
