"""
fl_error_analysis_select.py -- step 3 of the error analysis: do the shortlisted traits IMPROVE the projections, overall and for low-history players / rookies?
(design in fl_error_analysis_common.py; samples and shortlists from fl_error_analysis_hitters.py / fl_error_analysis_pitchers.py, read from their caches / screen CSVs)

For each target with a shortlist:
  A  single-trait tests: b fit on one sample, applied to another: ROS25 -> ROS26, ROS26 -> ROS25 (disjoint outcome seasons) and ROS25 -> PRE (transfer to the preseason
     model; ROS26 -> PRE is NOT used: same outcome season); plus 10-fold-by-player x 10 cross-validation inside PRE.  RA9 (no disjoint pair): CV inside RA9-pre and inside ROS26.
  B  forward selection (correlated traits): start from production; at each step add the shortlisted trait that most lowers the held-out weighted SSE summed over the two
     cross-season directions (each relative to its own baseline), jointly re-fitting all selected terms (weighted LS, no intercept); accept only if BOTH directions improve;
     at most 4 traits.  RA9: the same with the two CV samples.  Run for the faded form (primary) and the constant form.
  C  the selected set scored with the reporting standard: weighted RMSE and MAE, frozen weights, player bootstrap (600), overall / low-history (< 150) / rookies.
Usage:  python fl_error_analysis_select.py [h|p]
"""
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

import fl_error_analysis_common as ea
import fl_eval_report as er

warnings.filterwarnings("ignore")
NB = 600


def load(kind):
    t = ea.load_tracking()
    if kind == "h":
        from fl_error_analysis_hitters import CACHE, N0_T, window
        st, S = pickle.load(open(CACHE, "rb")); n0t = N0_T; key = "bat"
    else:
        from fl_error_analysis_pitchers import CACHE, window
        st, S, n0t = pickle.load(open(CACHE, "rb")); key = None
    for name, s in S.items():
        w = window(t, name, s.get("asof")).copy(); w["key"] = w[key or s["key"]]; s["traits"] = ea.build(w, kind)
    scr = pd.read_csv(f"output/error_analysis_{'hitters' if kind == 'h' else 'pitchers'}_screen.csv", index_col=[0, 1])
    short = {tg: list(scr.loc[tg].index[scr.loc[tg]["shortlist"].astype(bool)]) for tg in scr.index.get_level_values(0).unique()}
    return st, S, n0t, short


def X(s, tg, L, st, n0t, fade):
    F = s["targets"][tg]; cols = []
    for k in L:
        v, n = s["traits"][k]; x = ea.shrink(v, n, st.loc[k, "n0"]).reindex(F.index)
        mu = np.average(x.dropna(), weights=F.loc[x.notna(), "w"]) if x.notna().any() else 0.0; x = x.fillna(mu)
        f = n0t[tg] / (F["hist"] + n0t[tg]) if fade else 1.0; cols.append(((x - mu) * f).to_numpy())
    return np.column_stack(cols) if cols else np.zeros((len(F), 0))


def fit(F, Xm):
    w = F["w"].to_numpy(); return np.linalg.lstsq(Xm * np.sqrt(w)[:, None], F["e"].to_numpy() * np.sqrt(w), rcond=None)[0]


def rel_sse(F, e1):
    w = F["w"].to_numpy(); e0 = F["e"].to_numpy(); c = lambda e: e - np.average(e, weights=w)
    return np.sum(w * c(e1) ** 2) / np.sum(w * c(e0) ** 2)


def cv_terms(F, Xm, folds=10, reps=10, seed=5):
    rng = np.random.default_rng(seed); out = np.zeros(len(F))
    for _ in range(reps):
        fid = rng.integers(0, folds, len(F))
        for k in range(folds):
            tr, te = fid != k, fid == k; b = fit(F[tr], Xm[tr]); out[te] += Xm[te] @ b / reps
    return out


def report(F, e1, lab):
    w = F["w"].to_numpy(); e0 = F["e"].to_numpy(); rows = []
    for grp, m in (("all", np.ones(len(F), bool)), ("low-history", (F["hist"] < 150).to_numpy()), ("rookies", F["rookie"].to_numpy().astype(bool))):
        if m.sum() < 15:
            continue
        y0 = e0[m] - np.average(e0[m], weights=w[m]); y1 = e1[m] - np.average(e1[m], weights=w[m])
        r = er.compare(np.zeros(m.sum()), -y0, -y1, w[m], F.index.to_numpy()[m], n=NB)
        rows.append({"test": lab, "group": grp, "n": int(m.sum()), "RMSE old": r["RMSE old"], "RMSE new": r["RMSE new"], "dRMSE": r["dRMSE"], "RMSE 90%": r["RMSE 90%"], "P(RMSE better)": r["P(RMSE better)"],
                     "MAE old": r["MAE old"], "MAE new": r["MAE new"], "dMAE": r["dMAE"], "P(MAE better)": r["P(MAE better)"]})
    return rows


def designs(tg, S):
    if tg == "ra9":
        return [], [("CV", "RA9"), ("CV", "ROS26")]
    pairs = [("ROS25", "ROS26"), ("ROS26", "ROS25")]
    return pairs, pairs + [("ROS25", "PRE"), ("CV", "PRE")]


def evaluate(L, tg, S, st, n0t, fade, which):
    rows = []
    for a, b in which:
        if a == "CV":
            F = S[b]["targets"][tg]; Xm = X(S[b], tg, L, st, n0t, fade); e1 = F["e"].to_numpy() - cv_terms(F, Xm); lab = f"CV within {b}"
        else:
            FA, FB = S[a]["targets"][tg], S[b]["targets"][tg]; bb = fit(FA, X(S[a], tg, L, st, n0t, fade)); e1 = FB["e"].to_numpy() - X(S[b], tg, L, st, n0t, fade) @ bb; F = FB; lab = f"{a} -> {b}"
        rows += report(F, e1, lab)
    return rows


def forward(tg, cands, S, st, n0t, fade, max_k=4):
    sel_pairs, _ = designs(tg, S); path = []
    def score(L):
        out = []
        if tg == "ra9":
            for s in ("RA9", "ROS26"):
                F = S[s]["targets"][tg]; out.append(rel_sse(F, F["e"].to_numpy() - cv_terms(F, X(S[s], tg, L, st, n0t, fade))))
        else:
            for a, b in sel_pairs:
                FA, FB = S[a]["targets"][tg], S[b]["targets"][tg]; bb = fit(FA, X(S[a], tg, L, st, n0t, fade)); out.append(rel_sse(FB, FB["e"].to_numpy() - X(S[b], tg, L, st, n0t, fade) @ bb))
        return np.array(out)
    cur = []; cur_s = np.ones(2)
    while len(cur) < max_k:
        best = None
        for c in cands:
            if c in cur:
                continue
            sc = score(cur + [c])
            if np.all(sc < cur_s) and (best is None or sc.sum() < best[1].sum()):
                best = (c, sc)
        if best is None:
            break
        cur.append(best[0]); cur_s = best[1]; path.append({"step": len(cur), "added": best[0], "held-out SSE ratio (dir 1 / dir 2)": " / ".join(f"{v:.4f}" for v in cur_s)})
    return cur, path


def main(kind):
    st, S, n0t, short = load(kind)
    single, sel_rows, final = [], [], []
    for tg, cands in short.items():
        if not cands:
            print(f"\n{tg}: no shortlisted trait"); continue
        _, which = designs(tg, S)
        for c in cands:
            for fade in (True, False):
                for r in evaluate([c], tg, S, st, n0t, fade, which):
                    single.append({"target": tg, "trait": c, "form": "faded" if fade else "constant", **r})
        for fade in (True, False):
            L, path = forward(tg, cands, S, st, n0t, fade)
            print(f"\nFORWARD SELECTION, target {tg}, {'faded' if fade else 'constant'} form: {L if L else 'nothing added'}")
            if path:
                print(pd.DataFrame(path).to_string(index=False))
            sel_rows.append({"target": tg, "form": "faded" if fade else "constant", "selected": ", ".join(L)})
            if L:
                for r in evaluate(L, tg, S, st, n0t, fade, which):
                    final.append({"target": tg, "form": "faded" if fade else "constant", "traits": " + ".join(L), **r})
                # coefficients on the full disjoint samples, for the record
                for s in (["RA9", "ROS26"] if tg == "ra9" else ["ROS25", "ROS26", "PRE"]):
                    b = fit(S[s]["targets"][tg], X(S[s], tg, L, st, n0t, fade)); print(f"   coefficients fit on {s}: " + ", ".join(f"{k} {v:+.5f}" for k, v in zip(L, b)) + "  (per unit of the shrunk trait" + (" x fade)" if fade else ")"))
    pd.set_option("display.width", 250)
    if single:
        A = pd.DataFrame(single); A.to_csv(f"output/error_analysis_{kind}_single_trait_tests.csv", index=False)
        print("\nSINGLE-TRAIT TESTS, overall rows (full table incl. low-history / rookies in the CSV)\n" + A[A["group"] == "all"].drop(columns=["group"]).round(5).to_string(index=False))
    if final:
        B = pd.DataFrame(final); B.to_csv(f"output/error_analysis_{kind}_selected_tests.csv", index=False)
        print("\nSELECTED SETS, all groups\n" + B.round(5).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "h")
