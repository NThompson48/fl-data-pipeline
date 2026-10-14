"""
fl_ra9_input_robustness.py -- does the shipped RA9 baseline hold up when it is
trained on RAW prior-season K%/BB% but SCORED on shrunk model projections?

Background (DECISION_LOG.md, "RA/9 pivot: production-projected K%/BB% beat raw
rates"): exploration found projected K%/BB% were the best RA9 inputs, but the
shipped fit_ra9_baseline_model (fl_ra9_trustmark_projection.py) trains on raw
so/bf and bb/bf from the prior season while the roster is scored with the
frozen K%/BB% model's SHRUNK projections -- an input-type mismatch that can
double-shrink the coefficients (raw inputs are noisy -> attenuated betas ->
applied to already-shrunk inputs).

This script makes the projected K%/BB% genuinely OUT-OF-SAMPLE with respect to
each pitcher's own outcome (K-fold cross-fitting of the K%/BB% model on its
training transition), then runs LOOCV of the RA9 baseline under three
train/score input combinations:
    raw -> raw        what the old LOOCV validated
    raw -> projected  what is actually SHIPPED (train raw, score projected)
    proj -> projected the consistent alternative
in BOTH directions (2025 -> 2026 forward, 2026 -> 2025 reversed, same
robustness convention the log already uses for the RA9 regressors).

Reports weighted R2/RMSE, calibration slope (actual on predicted; <1 means
over-dispersed predictions, >1 means compressed), prediction SD vs actual SD,
and a paired bootstrap on the RMSE difference.

Usage:  python fl_ra9_input_robustness.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import KFold

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import mlb_marcel as mm
import fl_ra9_trustmark_projection as ra9
from fl_production_projection import SPEC, BB_SPEC_BLENDED
from fl_supabase import read_sql_df

N_FOLDS = 10
SEED = 0
N_BOOT = 2000


# --------------------------------------------------------------------------- #
#  Out-of-fold K%/BB% projections
# --------------------------------------------------------------------------- #
def oof_kbb_projections(raw, feat_windows, label_windows):
    """Project K%/BB% for every pitcher in `feat_windows` using the SAME model
    form production ships (WLS OLS K spec; BB with blended informative prior),
    fit on feat_windows -> label_windows. Pitchers that appear in the training
    transition get an OUT-OF-FOLD prediction (the model that scored them never
    saw their own label); others get the full-sample fit."""
    feat, n0_k, n0_bb = bt.assemble_features(raw, feat_windows, with_changepoints=False)
    actual = bt.actual_rate(raw, label_windows)
    pool = pd.concat([fp.filter_period(raw, s, e) for s, e in feat_windows], ignore_index=True)
    term = pool[pool["is_terminal"]]
    lg_bb = (term["k_or_bb"] == "walk").mean()
    feat["BB_raw"] = feat["naive_BB_rate"] * (feat["PA"] + n0_bb) - n0_bb * lg_bb

    tr = feat.merge(actual, on="pitcher_id", how="inner")
    tr_k = tr.dropna(subset=["K_rate_actual"] + SPEC["K"]).reset_index(drop=True)
    tr_bb = tr.dropna(subset=["BB_rate_actual", "naive_BB_rate", "fl_bb_rate_bridge", "PA", "command_score"]).reset_index(drop=True)

    def fit_k(d):
        w = mm.reliability_weight(d["PA_test"], n0_k).values
        return bt.fit_weighted(d, "K_rate_actual", SPEC["K"], w)

    def fit_bb(d):
        w = mm.reliability_weight(d["PA_test"], n0_bb).values
        n0b = mm.fit_n0_bridge(d["BB_raw"], d["PA"], n0_bb, lg_bb, d["fl_bb_rate_bridge"], d["BB_rate_actual"], w)
        d = d.copy()
        d["blended_BB_rate"] = mm.blended_rate(d["BB_raw"], d["PA"], n0_bb, lg_bb, n0b, d["fl_bb_rate_bridge"])
        return bt.fit_weighted(d, "BB_rate_actual", BB_SPEC_BLENDED, w), n0b

    def pred_bb(m, n0b, d):
        d = d.copy()
        d["blended_BB_rate"] = mm.blended_rate(d["BB_raw"], d["PA"], n0_bb, lg_bb, n0b, d["fl_bb_rate_bridge"])
        return m.predict(d)

    m_k_full = fit_k(tr_k)
    m_bb_full, n0b_full = fit_bb(tr_bb)
    out = feat[["pitcher_id"]].copy()
    ok = feat.dropna(subset=SPEC["K"])
    out = out.merge(pd.DataFrame({"pitcher_id": ok["pitcher_id"], "K_proj": m_k_full.predict(ok).values}),
                    on="pitcher_id", how="left")
    okb = feat.dropna(subset=["naive_BB_rate", "fl_bb_rate_bridge", "PA", "command_score"])
    out = out.merge(pd.DataFrame({"pitcher_id": okb["pitcher_id"], "BB_proj": pred_bb(m_bb_full, n0b_full, okb).values}),
                    on="pitcher_id", how="left")

    kf = KFold(N_FOLDS, shuffle=True, random_state=SEED)
    oof_k, oof_bb = {}, {}
    for tri, tei in kf.split(tr_k):
        m = fit_k(tr_k.iloc[tri])
        for pid, v in zip(tr_k.iloc[tei]["pitcher_id"], m.predict(tr_k.iloc[tei]).values):
            oof_k[pid] = v
    for tri, tei in kf.split(tr_bb):
        m, n0b = fit_bb(tr_bb.iloc[tri])
        for pid, v in zip(tr_bb.iloc[tei]["pitcher_id"], pred_bb(m, n0b, tr_bb.iloc[tei]).values):
            oof_bb[pid] = v
    out["K_oof"] = out["pitcher_id"].map(oof_k).fillna(out["K_proj"])
    out["BB_oof"] = out["pitcher_id"].map(oof_bb).fillna(out["BB_proj"])
    print(f"  OOF K%/BB% projections: n_train K={len(tr_k)} BB={len(tr_bb)}; "
          f"corr(oof, in-sample) K={np.corrcoef(out['K_oof'].dropna(), out.loc[out['K_oof'].notna(), 'K_proj'])[0,1]:.3f}")
    return out


# --------------------------------------------------------------------------- #
#  RA9 frame for one direction
# --------------------------------------------------------------------------- #
def build_frame(war, raw, oof, y_in, y_tgt, win_in):
    a = war[war["year"] == y_in].copy()
    b = war[war["year"] == y_tgt].copy()
    m = a.merge(b, on="player", suffixes=("_in", "_tgt"))
    m = m[(m["bf_in"] >= 100) & (m["bf_tgt"] >= 100)].copy()
    m["K_raw"] = m["so_in"] / m["bf_in"]
    m["BB_raw"] = m["bb_in"] / m["bf_in"]
    m["ppf"] = m["ppf_tgt"]
    m["ip_per_g_prior"] = (m["ip_in"] + ra9.N0_IPG * ra9.LG_IPG) / (m["g_in"] + ra9.N0_IPG)
    m["target"] = m["ra9_tgt"]
    ppa = ra9.pitches_per_pa(raw, *win_in).rename(columns={"pitches_per_pa": "pitches_per_pa_prior"})
    m = m.merge(ppa, left_on="player", right_on="pitcher", how="left").drop(columns="pitcher")
    # name -> unique pitcher_id (drop colliding names rather than guess)
    names = raw[["pitcher_id", "pitcher"]].drop_duplicates("pitcher_id")
    uniq = names[~names["pitcher"].duplicated(keep=False)]
    m = m.merge(uniq.rename(columns={"pitcher": "player"}), on="player", how="inner")
    m = m.merge(oof[["pitcher_id", "K_oof", "BB_oof"]], on="pitcher_id", how="left")
    m = m.dropna(subset=["K_oof", "BB_oof", "pitches_per_pa_prior", "ppf", "target"]).reset_index(drop=True)
    m["w"] = mm.reliability_weight(m["ip_tgt"], ra9.RA9_N0_EVAL).values
    return m


def loocv(m, k_tr, bb_tr, k_sc, bb_sc):
    """LOOCV of the shipped RA9 spec. Trains with (k_tr, bb_tr) input columns,
    scores the held-out pitcher with (k_sc, bb_sc). Mirrors production:
    ppa_resid from an auxiliary OLS on the TRAINING inputs, regressors clipped
    to the training fold's range."""
    n = len(m)
    preds = np.full(n, np.nan)
    for i in range(n):
        tr = m.drop(index=i).copy()
        te = m.loc[[i]].copy()
        trd = tr.assign(K_pred=tr[k_tr], BB_pred=tr[bb_tr])
        aux = smf.ols(ra9.PPA_RESID_SPEC, data=trd).fit()
        trd["ppa_resid"] = aux.resid
        mod = smf.wls(ra9.RA9_SPEC, data=trd, weights=tr["w"]).fit()
        ted = te.assign(K_pred=te[k_sc], BB_pred=te[bb_sc])
        ted["ppa_resid"] = ted["pitches_per_pa_prior"] - aux.predict(ted)
        for c in ra9.CONTINUOUS_REGRESSORS:
            ted[c] = ted[c].clip(trd[c].min(), trd[c].max())
        preds[i] = float(mod.predict(ted).iloc[0])
    return preds


def summarize(m, preds, label):
    y, w = m["target"].values, m["w"].values
    ok = ~np.isnan(preds)
    y, p, w = y[ok], preds[ok], w[ok]
    slope = smf.wls("y ~ p", data=pd.DataFrame({"y": y, "p": p}), weights=w).fit().params["p"]
    sd = lambda x: float(np.sqrt(np.cov(x, aweights=w)))
    return {"label": label, "n": int(ok.sum()), "R2_w": round(mm.weighted_r2(y, p, w), 4),
            "RMSE_w": round(mm.weighted_rmse(y, p, w), 4), "calib_slope": round(float(slope), 3),
            "sd_pred": round(sd(p), 3), "sd_actual": round(sd(y), 3)}


def boot_rmse_diff(m, p_a, p_b, seed=1):
    y, w = m["target"].values, m["w"].values
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        d.append(mm.weighted_rmse(y[idx], p_a[idx], w[idx]) - mm.weighted_rmse(y[idx], p_b[idx], w[idx]))
    d = np.array(d)
    return float(np.mean(d)), float(np.percentile(d, 5)), float(np.percentile(d, 95))


def run_direction(raw, war, name, y_in, y_tgt, feat_windows, label_windows, win_in):
    print(f"\n=== {name} ===")
    oof = oof_kbb_projections(raw, feat_windows, label_windows)
    m = build_frame(war, raw, oof, y_in, y_tgt, win_in)
    print(f"  RA9 sample n={len(m)}  corr(K_raw,K_oof)={m['K_raw'].corr(m['K_oof']):.3f} "
          f"corr(BB_raw,BB_oof)={m['BB_raw'].corr(m['BB_oof']):.3f}  "
          f"sd K_raw={m['K_raw'].std():.4f} sd K_oof={m['K_oof'].std():.4f} "
          f"sd BB_raw={m['BB_raw'].std():.4f} sd BB_oof={m['BB_oof'].std():.4f}")
    combos = {"raw->raw": ("K_raw", "BB_raw", "K_raw", "BB_raw"),
              "raw->proj (SHIPPED)": ("K_raw", "BB_raw", "K_oof", "BB_oof"),
              "proj->proj": ("K_oof", "BB_oof", "K_oof", "BB_oof")}
    res, P = [], {}
    for lab, cols in combos.items():
        P[lab] = loocv(m, *cols)
        res.append(summarize(m, P[lab], lab))
    print(pd.DataFrame(res).to_string(index=False))
    for a, b in [("raw->proj (SHIPPED)", "proj->proj")]:
        mean, lo, hi = boot_rmse_diff(m, P[a], P[b])
        print(f"  RMSE_w({a}) - RMSE_w({b}): mean {mean:+.4f}  90% bootstrap CI [{lo:+.4f}, {hi:+.4f}]")
    return res


if __name__ == "__main__":
    raw = fp.load_fl_pitches_cached()
    war = read_sql_df("select player, year, bf, ip, so, bb, ra, ra9, rar, ppf, g "
                      "from war_pitchers where game_type='reg_season'")
    w25, w26 = (bt.START_2025, bt.END_2025), (bt.START_2026, bt.END_2026)
    run_direction(raw, war, "FORWARD 2025 inputs -> 2026 RA9", 2025, 2026,
                  [w25], [(bt.START_2026, bt.CUTOFF_2026)], w25)
    run_direction(raw, war, "REVERSED 2026 inputs -> 2025 RA9 (non-causal robustness check)", 2026, 2025,
                  [w26], [w25], w26)
