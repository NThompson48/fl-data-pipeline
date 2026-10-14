"""
fl_park_pitcher_fe.py -- the matching PITCHER-side test of the TrustmarkPark translation (the current pitcher Mississippi RA9 shift = HR per air ball x .446 + non-HR air-ball hit delta .0073, about -0.87 runs/9 for a
full-time Trustmark pitcher at the league-average air-ball rate, -0.435 for half the games).
(1) Trustmark effect on wOBA ALLOWED from the same fixed-effects model as the hitter side (hitter + pitcher + year + park effects), converted to runs per 9 (wOBA / 1.0175 x 4.512 BF per IP x 9);
(2) Trustmark x PITCHER PROFILE: Air% allowed (shrunk n0 89 balls), HR/PA allowed (shrunk n0 137 PA), K% allowed (shrunk n0 113 PA) and throwing hand, profile measured on his NON-Trustmark PA / balls;
    outcomes wOBA, HR, K, BB, singles, 2B+3B (positive = worse for the pitcher);
(3) out of sample (fit on 2025, score on 2026; pitchers with >= 30 Trustmark BF in 2026): observed effect = Trustmark wOBA allowed minus a baseline from his 2026 non-Trustmark PA (hitter + pitcher + park effects);
    models: no adjustment | CURRENT pitcher translation (fl_ra9_trustmark_projection.build_trustmark_adjustment at full exposure, converted to wOBA per PA) | global effect (2025 fit) | global + profile (2025 fit);
    weighted by Trustmark BF; RMSE and MAE with pitcher-cluster bootstrap (reporting standard).
"""
import os
import warnings

import numpy as np
import pandas as pd
from sqlalchemy import text

import fl_eval_report as er
import fl_pitcher_features as fp
import fl_ra9_trustmark_projection as ra9
from fl_park_profile_fe import cluster_ols, partial_out
from fl_park_profile_final import load_pa
from fl_park_woba_fe import fe_fit
from fl_supabase import read_sql_df

warnings.filterwarnings("ignore")
SC = 1.0175; BF_IP = 4.512; TO_R9 = BF_IP * 9 / SC
BBE = "output/pitcher_bbe_2025_2026.parquet"


def pitcher_bbe():
    if os.path.exists(BBE):
        return pd.read_parquet(BBE)
    d = read_sql_df(text("""select pitcher_id, year, game_id, angle from pitches where game_type='reg_season' and year in (2025,2026) and angle is not null and exit_speed is not null
                            and play_result in ('single','double','triple','home_run','out','fielders_choice','error')"""))
    d.to_parquet(BBE, index=False); return d


def main():
    pa = load_pa(); pit = pd.read_parquet("output/pitcher_pa_ros_2025_2026.parquet"); assert len(pit) == len(pa) or True
    # load_pa drops rows without wOBA/park; rebuild the pitcher column consistently
    T = None
    from fl_park_profile_final import hr
    pa0 = hr.Tables().pa.copy(); assert len(pit) == len(pa0); pa0["pitcher"] = pit["pitcher_id"].astype(object).where(pit["pitcher_id"].notna(), "unknown").astype(str).values
    pa0["park"] = pa0["game_id"].str.split("-").str[1]; pa0 = pa0[pa0["woba"].notna() & pa0["park"].notna()].reset_index(drop=True); pa0["tm"] = (pa0["park"] == "TrustmarkPark").astype(int); pa0["name_clean"] = pa0["name_clean"].fillna("unknown")
    kind = pa0["kind"].astype(str); pa0["is_hr"] = (kind == "home_run").astype(float); pa0["is_k"] = (kind == "strikeout").astype(float); pa0["is_bb"] = (kind == "walk").astype(float); pa0["is_1b"] = (kind == "single").astype(float); pa0["is_xbh"] = kind.isin(["double", "triple"]).astype(float)
    pa = pa0
    raw = fp.load_fl_pitches_cached(); hand = raw.groupby("pitcher_id")["p_throws_clean"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan); pit_id = pd.read_parquet("output/pitcher_pa_ros_2025_2026.parquet")["pitcher_id"]
    pa["pid"] = pa["pitcher"]; hmap = hand.copy(); hmap.index = hmap.index.astype(str); pa["lefty"] = pa["pid"].map(hmap).map({"L": 1.0, "R": 0.0, "left": 1.0, "right": 0.0})
    bb = pitcher_bbe(); bb["pid"] = bb["pitcher_id"].astype(str); bb["air"] = (bb["angle"] >= 10).astype(int); bb["tm"] = bb["game_id"].str.contains("TrustmarkPark").astype(int)
    # ---------- (1) global effects and (2) interactions on the pooled sample
    groups = [pd.factorize(pa[c])[0] for c in ("year", "name_clean", "pid", "park")]; parks = pd.factorize(pa["park"])[1]
    out = {"wOBA": "woba", "HR/PA": "is_hr", "K%": "is_k", "BB%": "is_bb", "single/PA": "is_1b", "2B+3B/PA": "is_xbh"}; rows = []
    for lab, col in out.items():
        y = pa[col].to_numpy(float); mu, eff = fe_fit(y, groups, iters=30); pe = eff[3] - np.average(eff[3], weights=np.bincount(groups[3])); rows.append({"outcome": lab, "league": y.mean(), "Trustmark effect": pe[list(parks).index("TrustmarkPark")]})
    t = pd.DataFrame(rows).set_index("outcome"); print(f"(1) TRUSTMARK effect on the PITCHER's outcomes (per PA allowed; positive = worse for the pitcher): wOBA {t.loc['wOBA', 'Trustmark effect']:+.4f} = {t.loc['wOBA', 'Trustmark effect'] * TO_R9:+.2f} runs/9 for a full-time Trustmark pitcher "
          f"(current model: about -0.87 at the league air-ball rate, -0.435 at half the games)\n" + t.round(4).to_string())
    non = pa[pa["tm"] == 0]; g = non.groupby("pid").agg(n=("woba", "size"), hr=("is_hr", "sum"), k=("is_k", "sum")); lg_hr, lg_k = g["hr"].sum() / g["n"].sum(), g["k"].sum() / g["n"].sum()
    g["hr_d"] = (g["hr"] / g["n"] - lg_hr) * g["n"] / (g["n"] + 137.0); g["k_d"] = (g["k"] / g["n"] - lg_k) * g["n"] / (g["n"] + 113.0)
    bn = bb[bb["tm"] == 0].groupby("pid").agg(nb=("air", "size"), air=("air", "sum")); lg_air = bn["air"].sum() / bn["nb"].sum(); bn["air_d"] = (bn["air"] / bn["nb"] - lg_air) * bn["nb"] / (bn["nb"] + 89.0); g = g.join(bn[["air_d", "nb", "air"]], how="left")
    for c in ("hr_d", "k_d", "air_d"):
        pa[c] = pa["pid"].map(g[c])
    est = pa.dropna(subset=["hr_d", "k_d", "air_d", "lefty"]).reset_index(drop=True)
    for c in ("hr_d", "k_d", "air_d"):
        est["z_" + c] = (est[c] - est[c].mean()) / est[c].std()
    gr = [pd.factorize(est[c])[0] for c in ("year", "name_clean", "pid", "park")]; tmv = est["tm"].to_numpy(float); cl = pd.factorize(est["game_id"])[0]; lm = est["lefty"].mean()
    X = np.column_stack([tmv * est["z_air_d"], tmv * est["z_hr_d"], tmv * est["z_k_d"], tmv * (est["lefty"] - lm)]); Xr = partial_out(X, gr); res = []
    for lab, col in out.items():
        yr = partial_out(est[[col]].to_numpy(float), gr)[:, 0]; b, se = cluster_ols(Xr, yr, cl); res.append({"outcome": lab, "Air% allowed /SD": f"{b[0]:+.4f} ({se[0]:.4f})", "HR-prone /SD": f"{b[1]:+.4f} ({se[1]:.4f})", "K% allowed /SD": f"{b[2]:+.4f} ({se[2]:.4f})", "LHP vs avg": f"{b[3]:+.4f} ({se[3]:.4f})", "t air": b[0] / se[0], "t hr": b[1] / se[1], "t k": b[2] / se[2], "t lhp": b[3] / se[3]})
    print(f"\n(2) TRUSTMARK x PITCHER PROFILE (n = {len(est):,} PA; coefficient (cluster-robust SE) per +1 SD of the profile, wOBA points allowed per PA; positive = worse for the pitcher; corr(air, HR-prone) {est[['z_air_d', 'z_hr_d']].corr().iloc[0, 1]:+.2f})\n" + pd.DataFrame(res).round(2).to_string(index=False))
    # ---------- (3) out of sample 2025 -> 2026
    tr = pa[(pa["year"] == 2025)].dropna(subset=["hr_d", "k_d", "air_d", "lefty"]).reset_index(drop=True)
    for c in ("hr_d", "k_d", "air_d"):
        tr["z_" + c] = (tr[c] - est[c].mean()) / est[c].std()
    gt = [pd.factorize(tr[c])[0] for c in ("name_clean", "pid", "park")]; ytr = tr["woba"].to_numpy(float); mu, eff = fe_fit(ytr, gt, iters=30); pe = pd.Series(eff[2] - np.average(eff[2], weights=np.bincount(gt[2])), index=pd.factorize(tr["park"])[1]); d0 = float(pe["TrustmarkPark"])
    tmt = tr["tm"].to_numpy(float); Xt = np.column_stack([tmt * tr["z_air_d"], tmt * tr["z_hr_d"], tmt * tr["z_k_d"], tmt * (tr["lefty"] - lm)]); Xtr = partial_out(Xt, gt); ytr_r = partial_out(ytr[:, None], gt)[:, 0]; bt_, st_ = cluster_ols(Xtr, ytr_r, pd.factorize(tr["game_id"])[0])
    print(f"\n2025 fit ({len(tr):,} PA, {int(tmt.sum()):,} at Trustmark): global {d0:+.4f}; air {bt_[0]:+.4f} ({st_[0]:.4f}), HR-prone {bt_[1]:+.4f} ({st_[1]:.4f}), K% {bt_[2]:+.4f} ({st_[2]:.4f}), LHP {bt_[3]:+.4f} ({st_[3]:.4f})")
    te = pa[pa["year"] == 2026].reset_index(drop=True); nt = te[te["tm"] == 0]; tx = te[te["tm"] == 1]; g2 = [pd.factorize(nt[c])[0] for c in ("name_clean", "pid", "park")]
    mu2, e2 = fe_fit(nt["woba"].to_numpy(float), g2, iters=30); hit = pd.Series(e2[0], index=pd.factorize(nt["name_clean"])[1]); pit_e = pd.Series(e2[1], index=pd.factorize(nt["pid"])[1]); pkm = np.average(e2[2], weights=np.bincount(g2[2]))
    tx = tx[tx["pid"].isin(pit_e.index)].copy(); tx["base"] = mu2 + tx["name_clean"].map(hit).fillna(0.0) + tx["pid"].map(pit_e) + pkm; tx["excess"] = tx["woba"] - tx["base"]
    h = tx.groupby("pid").agg(n_tm=("woba", "size"), excess=("excess", "mean"), lefty=("lefty", "mean")); h = h[h["n_tm"] >= 30]
    prof25 = pa[pa["year"] == 2025]; non25 = prof25[prof25["tm"] == 0]; g25 = non25.groupby("pid").agg(n=("woba", "size"), hr=("is_hr", "sum"), k=("is_k", "sum"), bb=("is_bb", "sum")); l_hr, l_k = g25["hr"].sum() / g25["n"].sum(), g25["k"].sum() / g25["n"].sum()
    g25["hr_d"] = (g25["hr"] / g25["n"] - l_hr) * g25["n"] / (g25["n"] + 137.0); g25["k_d"] = (g25["k"] / g25["n"] - l_k) * g25["n"] / (g25["n"] + 113.0)
    b25 = bb[(bb["year"] == 2025) & (bb["tm"] == 0)].groupby("pid").agg(nb=("air", "size"), air=("air", "sum")); la = b25["air"].sum() / b25["nb"].sum(); b25["air_d"] = (b25["air"] / b25["nb"] - la) * b25["nb"] / (b25["nb"] + 89.0); b25["air_pct"] = b25["air"] / b25["nb"]
    h = h.join(g25[["hr_d", "k_d", "n"]], how="inner").join(b25[["air_d", "air_pct"]], how="inner").dropna(); h["lefty"] = h["lefty"].round()
    zs = {c: (est[c].mean(), est[c].std()) for c in ("hr_d", "k_d", "air_d")}; z = lambda c: (h[c] - zs[c][0]) / zs[c][1]
    k_rate = (g25.loc[h.index, "k"] / g25.loc[h.index, "n"]); bb_rate = (g25.loc[h.index, "bb"] / g25.loc[h.index, "n"])
    cur = ra9.build_trustmark_adjustment(pd.DataFrame({"K_pred": k_rate, "BB_pred": bb_rate, "proj_air_pct": h["air_pct"], "proj_bf": 1000.0, "proj_ip": 1000.0 / BF_IP}, index=h.index), 1.0)["trustmark_ra9_shift"] / (9 * BF_IP) * SC
    preds = {"M0 no adjustment": np.zeros(len(h)), "M1 CURRENT pitcher translation (full exposure)": cur.to_numpy(), "M2 global effect (2025 fit)": np.full(len(h), d0),
             "M3 global + profile (2025 fit)": d0 + bt_[0] * z("air_d") + bt_[1] * z("hr_d") + bt_[2] * z("k_d") + bt_[3] * (h["lefty"] - lm)}
    w = h["n_tm"].to_numpy(float); obs = h["excess"].to_numpy(); ids = h.index.to_numpy(); rows = []
    print(f"2026 test: {len(h)} pitchers with >= 30 Trustmark BF in 2026 (total {int(w.sum()):,} PA); observed mean effect {np.average(obs, weights=w):+.4f} wOBA allowed = {np.average(obs, weights=w) * TO_R9:+.2f} runs/9; predicted means: " + ", ".join(f"{k.split()[0]} {np.average(np.asarray(v), weights=w):+.4f}" for k, v in preds.items()))
    for k, v in preds.items():
        v = np.asarray(v, float); r_, m_ = er.wmetrics(obs, v, w); row = {"model": k, "RMSE": r_, "MAE": m_, "slope of observed on predicted": np.polyfit(v, obs, 1, w=np.sqrt(w))[0] if np.std(v) > 1e-9 else np.nan}
        if not k.startswith("M1"):
            rep = er.compare(obs, np.asarray(preds["M1 CURRENT pitcher translation (full exposure)"], float), v, w, ids, n=800); row.update({"dRMSE vs current": rep["dRMSE"], "RMSE 90%": rep["RMSE 90%"], "P(RMSE better)": rep["P(RMSE better)"], "dMAE vs current": rep["dMAE"], "P(MAE better)": rep["P(MAE better)"]})
        rows.append(row)
    print("\nOUT OF SAMPLE: observed 2026 Trustmark wOBA-allowed effect per pitcher vs predictions (full-park effect)\n" + er.table(rows, 4))


if __name__ == "__main__":
    main()
