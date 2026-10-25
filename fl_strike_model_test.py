"""
fl_strike_model_test.py -- STRIKE stuff models: is P(strike | stuff, location) - P(strike | location) and the count version stable / sticky and predictive?  (user 2026-10-25)
MODELS (network, 3 seeds, 5-fold GroupKFold by pitcher; fl_wh_sva_strike_worker.py): strike_loc = P(strike | 9 stuff + location inputs) - P(strike | location);  strike_ct = the same with balls and strikes in both.
'strike' = called strike, swinging strike or foul vs ball called / in dirt (balls in play, hit batters, intentional and automatic calls excluded).  No pitch-type input.  Level = mean over all of a pitcher's pitches in the population.
REFERENCES (same code path): torch3 = swing-based wh_sva network; swstr_ct = per-pitch swinging-strike with pitch type and count.
(1) STABLE / STICKY: split-half reliability (alternating games, pitchers with >= 150 thrown pitches per half, Spearman-Brown) in 2025 and 2026 separately; YEAR-TO-YEAR correlation of the pitcher level (2025 fit vs 2026 fit, pitchers with >= 300 pitches each year); correlations among the measures.
(2) PREDICTIVE: the standard K% harness (fit full 2025 -> 1H2026, score fold 2 -> 2H2026, level-free, frozen inverse-variance weights from the PUBLISHED model, pitcher bootstrap) and the same for BB% (blended BB rate + command score, the production BB model):
    K%: published | additive swstr_ct | strike_loc replaces the stuff term | strike_ct replaces the stuff term | swstr_ct + strike_ct;   BB%: published | + swstr_ct level | + strike_loc level | + strike_ct level | + swstr_ct + strike_ct.
Output: output/strike_model_test.json
"""
import json
import warnings

import numpy as np
import pandas as pd

import fl_eval_report as er
import fl_eval_weights as ew
import fl_halfseason_backtest as bt
import fl_k_prior_production_test as pt
import fl_pitcher_features as fp
import fl_position_player_flag as ppf
import fl_production_projection as pp
import fl_wh_sva_prior_k_test as pk
import mlb_marcel as mm

warnings.filterwarnings("ignore")
EST = {"network (swing)": "torch3", "swstr_ct": "swstr_ct", "strike_loc": "strike_loc", "strike_ct": "strike_ct"}
ADD = pp.SPEC["K"]; BBS = pp.BB_SPEC_BLENDED


def season_df(raw, yr):
    return fp.filter_period(raw, bt.START_2025 if yr == 2025 else bt.START_2026, bt.END_2025 if yr == 2025 else bt.END_2026)


def stability(raw):
    out = {}; L = {}
    for yr in (2025, 2026):
        s = season_df(raw, yr); g = s.drop_duplicates(["game_id", "pitcher_id"])[["game_id", "pitcher_id", "date"]].sort_values("date"); g["half"] = g.groupby("pitcher_id").cumcount() % 2
        thrown = s.merge(g[["game_id", "pitcher_id", "half"]], on=["game_id", "pitcher_id"]).groupby(["pitcher_id", "half"]).size().unstack(); n_all = s.groupby("pitcher_id").size()
        for lab, est in EST.items():
            sv = fp.compute_pooled_wh_sva(s, estimator=est); d = sv.merge(g[["game_id", "pitcher_id", "half"]], on=["game_id", "pitcher_id"]); lv = d.groupby(["pitcher_id", "half"])["sva"].mean().unstack()
            ok = (thrown.reindex(lv.index).min(axis=1) >= 150) & lv.notna().all(axis=1); r = float(np.corrcoef(lv.loc[ok, 0], lv.loc[ok, 1])[0, 1])
            out.setdefault(lab, {})[yr] = {"split_half_r": r, "S-B": 2 * r / (1 + r), "pitchers": int(ok.sum()), "n_pitches_sva": int(len(sv))}; L[(lab, yr)] = sv.groupby("pitcher_id")["sva"].mean().to_frame("lv").join(n_all.rename("n"))
        print(f"  {yr} done", flush=True)
    for lab in EST:
        a, b = L[(lab, 2025)], L[(lab, 2026)]; j = a.join(b, lsuffix="_25", rsuffix="_26", how="inner"); j = j[(j["n_25"] >= 300) & (j["n_26"] >= 300)]
        out[lab]["year_to_year_r"] = float(np.corrcoef(j["lv_25"], j["lv_26"])[0, 1]); out[lab]["year_to_year_n"] = int(len(j))
    for yr in (2025, 2026):
        M = pd.DataFrame({lab: L[(lab, yr)]["lv"] for lab in EST}).dropna(); print(f"\n  {yr}: correlation of pitcher levels among the measures (n = {len(M)})\n" + M.corr().round(3).to_string()); out[f"corr_{yr}"] = M.corr().round(4).to_dict()
    return out


def lg_rate(raw, windows, what):
    t = pd.concat([fp.filter_period(raw, a, b) for a, b in windows], ignore_index=True); t = t[t["is_terminal"]]; return float((t["k_or_bb"] == what).mean())


def predictive(raw):
    F = {e: pt.frames(raw, e) for e in ("xgb", "swstr_ct", "strike_loc", "strike_ct")}; tr, n0_k, f2, n0_k2 = F["xgb"]; lvl = {"sw": "swstr_ct", "sl": "strike_loc", "sc": "strike_ct"}
    def merged(frame, which):
        d = frame.copy()
        for k, e in lvl.items():
            d = d.merge(F[e][which][["pitcher_id", "wh_sva_level"]].rename(columns={"wh_sva_level": k}), on="pitcher_id", how="left")
        return d
    dtr, dte = merged(tr, 0).copy(), merged(f2, 2).copy()
    # blended BB rate on both frames (as production: n0_bridge fit on the training transition)
    lg_bb_tr = lg_rate(raw, [(bt.START_2025, bt.END_2025)], "walk"); lg_bb_f2 = lg_rate(raw, [(bt.START_2025, bt.END_2025), (bt.START_2026, bt.CUTOFF_2026)], "walk"); n0_bb_tr = F["xgb"][1]
    _, _, n0_bb = bt.assemble(raw, [(bt.START_2025, bt.END_2025)], [(bt.START_2026, bt.CUTOFF_2026)], with_changepoints=False)
    dbb = dtr.dropna(subset=["BB_rate_actual", "naive_BB_rate", "fl_bb_rate_bridge", "PA", "command_score", "sw", "sl", "sc"]).copy(); wbb = mm.reliability_weight(dbb["PA_test"], n0_bb).values
    dbb["BB_raw"] = dbb["naive_BB_rate"] * (dbb["PA"] + n0_bb) - n0_bb * lg_bb_tr; n0b = mm.fit_n0_bridge(dbb["BB_raw"], dbb["PA"], n0_bb, lg_bb_tr, dbb["fl_bb_rate_bridge"], dbb["BB_rate_actual"], wbb)
    dbb["blended_BB_rate"] = mm.blended_rate(dbb["BB_raw"], dbb["PA"], n0_bb, lg_bb_tr, n0b, dbb["fl_bb_rate_bridge"])
    tb = dte.dropna(subset=["BB_rate_actual", "naive_BB_rate", "fl_bb_rate_bridge", "PA", "command_score", "sw", "sl", "sc"]).copy(); tb["BB_raw"] = tb["naive_BB_rate"] * (tb["PA"] + n0_bb) - n0_bb * lg_bb_f2
    tb["blended_BB_rate"] = mm.blended_rate(tb["BB_raw"], tb["PA"], n0_bb, lg_bb_f2, n0b, tb["fl_bb_rate_bridge"])
    dk = dtr.dropna(subset=["K_rate_actual"] + ADD + ["sw", "sl", "sc"]).copy(); wk = mm.reliability_weight(dk["PA_test"], n0_k).values; tk = dte.dropna(subset=["K_rate_actual"] + ADD + ["sw", "sl", "sc"]).copy()
    base_k = [f for f in ADD if f != "wh_sva_level"]; res = {}
    for stat, dtrain, wtr, dtest, target, specs in (
            ("K%", dk, wk, tk, "K_rate_actual", {"published": ADD, "additive swstr_ct": base_k + ["sw"], "strike_loc": base_k + ["sl"], "strike_ct": base_k + ["sc"], "swstr_ct + strike_ct": base_k + ["sw", "sc"]}),
            ("BB%", dbb, wbb, tb, "BB_rate_actual", {"published": BBS, "+ swstr_ct": BBS + ["sw"], "+ strike_loc": BBS + ["sl"], "+ strike_ct": BBS + ["sc"], "+ swstr_ct + strike_ct": BBS + ["sw", "sc"]})):
        P = {}; coefs = {}
        for lab, spec in specs.items():
            m = bt.fit_weighted(dtrain, target, spec, wtr); P[lab] = m.predict(dtest).to_numpy(); coefs[lab] = {k: (round(float(m.params[k]), 4), round(float(m.tvalues[k]), 2)) for k in spec if k in ("sw", "sl", "sc", "wh_sva_level")}
        y = dtest[target].to_numpy(); n_t = dtest["PA_test"].to_numpy(); ids = dtest["pitcher_id"].to_numpy(); base = P["published"]; w = ew.weights_from_base(y, base, n_t)[0]; cen = lambda q: q + np.average(y - q, weights=w); rows = []
        for lab, q in P.items():
            rm, ma = er.wmetrics(y, cen(q), w); r = {"model": lab, "RMSE": rm, "MAE": ma}
            if lab != "published":
                c = er.compare(y, cen(base), cen(q), w, ids, n=1000); r.update({"dRMSE": c["dRMSE"], "RMSE 90%": c["RMSE 90%"], "P(RMSE better)": c["P(RMSE better)"], "dMAE": c["dMAE"], "MAE 90%": c["MAE 90%"], "P(MAE better)": c["P(MAE better)"]})
            rows.append(r)
        pd.set_option("display.width", 260); print(f"\n{stat}: fold 2, n = {len(y)}, level-free, frozen weights (coefficients (b, t): {coefs})\n" + er.table(rows, 5)); res[stat] = {"n": len(y), "table": rows, "coefs": coefs}
    return res


def main():
    raw = ppf.exclude(fp.load_fl_pitches_cached()); print("STABILITY", flush=True); st = stability(raw)
    print("\nSTABILITY SUMMARY (split-half S-B 2025 | 2026; year-to-year r)")
    for lab, v in st.items():
        if lab in EST:
            print(f"  {lab:16s} {v[2025]['S-B']:.4f} | {v[2026]['S-B']:.4f}; year-to-year r {v['year_to_year_r']:.3f} (n = {v['year_to_year_n']})")
    print("\nPREDICTIVE", flush=True); pr = predictive(raw); json.dump({"stability": st, "predictive": pr}, open("output/strike_model_test.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
