"""
fl_hitter_hr_term_test.py -- a contact-quality / swing-path TERM for the hitter HR/PA projection, judged on HR/PA and end-to-end on wOBA  (user 2026-10-27: "focus on the HR/PA part"; a BABIP term is dropped: the wOBA gain is too small to justify the complexity)
Production HR/PA = the regressed component (no contact-quality term).  Candidates (the HR/PA shortlist of fl_hitter_battrack_hrbabip.py, which includes the user's estimates): zcontact%, whiff%, two-strike whiff%, measured bat speed, avg EV, hard-hit%, barrel%, pull%, oppo%, xwOBAcon, bat_speed_est, pad_lac.
  A  FORWARD SELECTION (fl_error_analysis_select.forward: add the trait that lowers the held-out SSE in BOTH directions ROS25 -> ROS26 and ROS26 -> ROS25; <= 4 traits; faded and constant forms), then the selected set and the named sets
     [barrel%], [barrel% + pad_lac], [bat_speed_est + pad_lac], [avg EV], [avg EV + pad_lac], [bat_speed (measured)] scored on HR/PA with the reporting standard (weighted RMSE / MAE, frozen weights, player bootstrap; overall / low-history / rookies):
     ROS25 -> ROS26, ROS26 -> ROS25, ROS25 -> PRE (and CV within PRE).
  B  END-TO-END wOBA: the HR/PA term enters the component-built half of wOBA (production: wOBA = half direct + half built from K%, BB%, HR/PA with HR weight 1.414), so wOBA_new = wOBA + .5 x 1.414 x (HR/PA term), b fit on the HR/PA residuals of one sample
     and applied to the wOBA frames of the other (ROS25 <-> ROS26, ROS25 -> PRE); the production wOBA already carries the EV90 prior and the xwOBA add-on, so any overlap shows up as a small gain.  Reported: wOBA RMSE / MAE change, overall / low-history / rookies.
Usage: python fl_hitter_hr_term_test.py   (log: output/hr_term_log.txt)    Output: output/hr_term_test.json
"""
import json
import pickle
import warnings

import numpy as np
import pandas as pd

import fl_error_analysis_common as ea
import fl_error_analysis_hitters as eh
import fl_error_analysis_select as sel
import fl_hitter_battrack_test as bk

warnings.filterwarnings("ignore")
CANDS = ["zcontact_pct", "whiff_pct", "two_strike_whiff", "bat_speed", "avg_ev", "hardhit_pct", "barrel_pct", "pull_pct", "oppo_pct", "xwobacon", "bat_speed_est", "pad_lac"]
NAMED = {"barrel%": ["barrel_pct"], "barrel% + pad_lac": ["barrel_pct", "pad_lac"], "bat_speed_est + pad_lac": ["bat_speed_est", "pad_lac"], "avg EV": ["avg_ev"], "avg EV + pad_lac": ["avg_ev", "pad_lac"], "measured bat speed": ["bat_speed"], "hard-hit%": ["hardhit_pct"]}
N0T = {"woba": 434.2, "k": 79.7, "bb": 146.7, "hr": 136.8, "babip": 851.6}
HR_W = 0.5 * 1.414                                                      # weight of an HR/PA deviation in production wOBA (half direct, half built; HR component weight 1.414)


def e2e(L, S, st, fade, which):
    """HR/PA term (b fit on one sample's HR/PA residuals) applied to the wOBA frames; returns report rows."""
    rows = []
    for a, b in which:
        if a == "CV":
            continue
        FA = S[a]["targets"]["hr"]; bb = sel.fit(FA, sel.X(S[a], "hr", L, st, N0T, fade)); Fw = S[b]["targets"]["woba"]
        Xw = sel.X(S[b], "woba", L, st, {**N0T, "woba": N0T["hr"]}, fade); e1 = Fw["e"].to_numpy() - HR_W * (Xw @ bb)
        rows += sel.report(Fw, e1, f"wOBA via HR/PA term {a} -> {b}")
    return rows


def main():
    pd.set_option("display.width", 270); t = bk.load_tracking_tolerant(); t["key"] = t["bat"]
    st, S = pickle.load(open("output/_battrack_samples.pkl", "rb")); S2 = pickle.load(open("output/_battrack_hrbabip_samples.pkl", "rb"))
    for name, s_ in S.items():
        s_["targets"].update(S2[name]); s_["window"] = eh.window(t, name, s_.get("asof")); s_["traits"] = bk.build_all(s_["window"].assign(key=s_["window"]["bat"]), "h")
    _, which = sel.designs("hr", S); out = {"forward": [], "hr": [], "woba": []}
    for fade in (True, False):
        L, path = sel.forward("hr", CANDS, S, st, N0T, fade); form = "faded" if fade else "constant"
        print(f"\nFORWARD SELECTION, HR/PA, {form} form: {L if L else 'nothing added'}")
        if path:
            print(pd.DataFrame(path).to_string(index=False))
        out["forward"].append({"form": form, "selected": L, "path": path}); sets = dict(NAMED)
        if L:
            sets["SELECTED: " + " + ".join(L)] = L
        for lab, S_ in sets.items():
            for r in sel.evaluate(S_, "hr", S, st, N0T, fade, which):
                out["hr"].append({"set": lab, "form": form, **r})
            for r in e2e(S_, S, st, fade, which):
                out["woba"].append({"set": lab, "form": form, **r})
            bsel = {s: sel.fit(S[s]["targets"]["hr"], sel.X(S[s], "hr", S_, st, N0T, fade)) for s in ("ROS25", "ROS26", "PRE")}
            print(f"   coefficients ({form}) {lab}: " + "; ".join(f"{s}: " + ", ".join(f"{k} {v:+.5f}" for k, v in zip(S_, b)) for s, b in bsel.items()))
    for key, title in (("hr", "HR/PA"), ("woba", "wOBA (end-to-end through the HR/PA term)")):
        A = pd.DataFrame(out[key]); A = A[A["group"] == "all"]
        cols = ["set", "form", "test", "n", "RMSE old", "dRMSE", "RMSE 90%", "P(RMSE better)", "dMAE", "P(MAE better)"]
        print(f"\n{title}: change vs production (negative = better)\n" + A[cols].round(5).to_string(index=False))
    json.dump(out, open("output/hr_term_test.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
