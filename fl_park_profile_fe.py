"""
fl_park_profile_fe.py -- what does TrustmarkPark do to a hitter, by outcome and by hitter PROFILE (Air%, power, handedness)?  (2025-26 regular season, 135,382 PA)
Fixed-effects design (hitter + pitcher + year + park effects absorbed by iterative demeaning, Frisch-Waugh-Lovell for the interaction terms; cluster-robust SEs by game):
 (1) PARK EFFECT BY OUTCOME for every park (per PA, in points of the rate): HR, K, BB, single, double+triple, and wOBA; the Trustmark row shows where the global wOBA effect comes from.
 (2) TRUSTMARK x PROFILE: y = FE + d1 x TM x z_air + d2 x TM x z_power + d3 x TM x left + d4 x TM x switch-hitter-vs-... (the TM main effect is absorbed by the park FE, so d_k is the profile-specific deviation from the Trustmark
     average).  Profile variables are measured on the hitter's NON-Trustmark plate appearances / balls (so the park cannot create its own profile): shrunk Air% (n0 89 balls), shrunk HR/PA (n0 137 PA), standardised; handedness is the
     side the hitter batted from in THAT plate appearance (first pitch's batter_side; switch hitters therefore enter by the side they used).
 Outputs: output/park_outcome_fe_2025_26.csv, and the profile table printed; compare with the current translation (HR factor .446, hit delta .0073).
"""
import warnings

import numpy as np
import pandas as pd

import fl_hitter_ev as hev
import fl_hitter_ros as hr
import fl_hitter_whiff as hwf
from fl_park_woba_fe import fe_fit

warnings.filterwarnings("ignore")


def partial_out(cols, groups, iters=30):
    """Residualise each column of `cols` (n x m) on the fixed effects `groups`."""
    out = np.empty_like(cols, dtype=float)
    for j in range(cols.shape[1]):
        mu, eff = fe_fit(cols[:, j].astype(float), groups, iters=iters); out[:, j] = cols[:, j] - mu - sum(e[g] for e, g in zip(eff, groups))
    return out


def cluster_ols(X, y, cl):
    b = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X @ b; XtXi = np.linalg.inv(X.T @ X); meat = np.zeros((X.shape[1], X.shape[1]))
    df = pd.DataFrame(X * e[:, None]); s = df.groupby(cl).sum().to_numpy(); meat = s.T @ s; V = XtXi @ meat @ XtXi * (len(np.unique(cl)) / (len(np.unique(cl)) - 1)); return b, np.sqrt(np.diag(V))


def main():
    T = hr.Tables(); pa = T.pa.copy(); pit = pd.read_parquet("output/pitcher_pa_ros_2025_2026.parquet"); assert len(pit) == len(pa) and (pit["game_id"].values == pa["game_id"].values).all()
    pa["pitcher"] = pit["pitcher_id"].astype(object).where(pit["pitcher_id"].notna(), "unknown").astype(str).values; pa["park"] = pa["game_id"].str.split("-").str[1]; pa["name_clean"] = pa["name_clean"].fillna("unknown")
    pa = pa[pa["woba"].notna() & pa["park"].notna()].reset_index(drop=True); pa["tm"] = (pa["park"] == "TrustmarkPark").astype(int)
    # PA-level batter side from the pitch table (first pitch of the plate appearance)
    pi = hwf.load(); pi = pi[pi["year"].isin([2025, 2026]) & pi["pa_id"].notna() & pi["batter_side"].isin(["left", "right"])].copy(); pi["name_clean"] = pi["batter"].map(lambda x: x)
    import fl_innings_projection as ip
    pi["name_clean"] = pi["batter"].apply(ip.lastfirst_to_clean); side = pi.sort_values("pitch_no").groupby(["game_id", "pa_id", "name_clean"])["batter_side"].first().rename("side").reset_index()
    pa = pa.merge(side, on=["game_id", "pa_id", "name_clean"], how="left"); print(f"PA with a known batting side: {pa['side'].notna().mean():.1%}; left share {(pa['side'] == 'left').mean():.1%}")
    pa["left"] = (pa["side"] == "left").astype(float); pa.loc[pa["side"].isna(), "left"] = np.nan
    kind = pa["kind"].astype(str); pa["is_hr"] = (kind == "home_run").astype(float); pa["is_k"] = (kind == "strikeout").astype(float); pa["is_bb"] = (kind == "walk").astype(float)
    pa["is_1b"] = (kind == "single").astype(float); pa["is_xbh"] = kind.isin(["double", "triple"]).astype(float)
    groups = [pd.factorize(pa[c])[0] for c in ("year", "name_clean", "pitcher", "park")]; parks = pd.factorize(pa["park"])[1]; cl = pd.factorize(pa["game_id"])[0]
    # (1) park effect by outcome
    outc = {"HR/PA": "is_hr", "K%": "is_k", "BB%": "is_bb", "single/PA": "is_1b", "2B+3B/PA": "is_xbh", "wOBA": "woba"}; rows = []
    P = np.column_stack([(pa["park"] == p).astype(float).to_numpy() for p in parks])
    for lab, col in outc.items():
        y = pa[col].to_numpy(float); mu, eff = fe_fit(y, groups, iters=30); pe = eff[3] - np.average(eff[3], weights=np.bincount(groups[3])); rows.append({"outcome": lab, "league rate": y.mean(), **{"effect: " + p: pe[i] for i, p in enumerate(parks)}})
    t = pd.DataFrame(rows).set_index("outcome"); t.to_csv("output/park_outcome_fe_2025_26.csv")
    tm = pd.DataFrame({"league rate": t["league rate"], "TrustmarkPark effect (points)": t["effect: TrustmarkPark"], "as % of league rate": 100 * t["effect: TrustmarkPark"] / t["league rate"]})
    print("\n(1) TRUSTMARKPARK effect by outcome (per PA; hitter + pitcher + year fixed effects; vs the league-average park)\n" + tm.round(4).to_string())
    pk = t[[c for c in t.columns if c.startswith("effect: ")]].T; pk.index = [i.replace("effect: ", "") for i in pk.index]
    print("\n    HR/PA effect by park (the HR column sorted), for context:\n" + pk[["HR/PA", "wOBA"]].sort_values("HR/PA").round(4).to_string())
    # (2) Trustmark x profile, profile measured away from Trustmark
    non = pa[pa["tm"] == 0]; g = non.groupby("name_clean").agg(pa_n=("woba", "size"), hr=("is_hr", "sum")); lg_hr = g["hr"].sum() / g["pa_n"].sum(); g["power"] = (g["hr"] / g["pa_n"] - lg_hr) * g["pa_n"] / (g["pa_n"] + 137.0)
    b = hev.load_bbe(); b["air"] = (b["angle"] >= 10).astype(int); b = b.merge(pa[pa["tm"] == 1][["game_id"]].drop_duplicates().assign(tm_game=1), how="left", on="game_id") if "game_id" in b.columns else b
    bb = hev.load_bbe(); bb["air"] = (bb["angle"] >= 10).astype(int); ga = bb.groupby("name_clean").agg(n=("air", "size"), air=("air", "sum")); lg_air = ga["air"].sum() / ga["n"].sum(); ga["air_dev"] = (ga["air"] / ga["n"] - lg_air) * ga["n"] / (ga["n"] + 89.0)
    pa["power_z"] = pa["name_clean"].map(g["power"]); pa["air_z"] = pa["name_clean"].map(ga["air_dev"]); pa = pa.dropna(subset=["power_z", "air_z", "left"]).reset_index(drop=True)
    pa["power_z"] = (pa["power_z"] - pa["power_z"].mean()) / pa["power_z"].std(); pa["air_z"] = (pa["air_z"] - pa["air_z"].mean()) / pa["air_z"].std()
    groups2 = [pd.factorize(pa[c])[0] for c in ("year", "name_clean", "pitcher", "park")]; cl2 = pd.factorize(pa["game_id"])[0]; tmv = pa["tm"].to_numpy(float)
    print(f"\n(2) TRUSTMARK x PROFILE: {len(pa):,} PA with a profile (hitters with non-Trustmark PA and tracked balls); correlation of Air% and power profiles {pa[['air_z', 'power_z']].corr().iloc[0, 1]:+.2f}")
    specs = {"HR/PA": "is_hr", "K%": "is_k", "BB%": "is_bb", "single/PA": "is_1b", "2B+3B/PA": "is_xbh", "wOBA": "woba"}; res = []
    X = np.column_stack([tmv * pa["air_z"], tmv * pa["power_z"], tmv * (pa["left"] - pa["left"].mean())]); Xr = partial_out(X, groups2)
    for lab, col in specs.items():
        yr = partial_out(pa[[col]].to_numpy(float), groups2)[:, 0]; bcoef, se = cluster_ols(Xr, yr, cl2); res.append({"outcome": lab, "Air% (per SD)": f"{bcoef[0]:+.4f} ({se[0]:.4f})", "power (per SD)": f"{bcoef[1]:+.4f} ({se[1]:.4f})", "LEFT-handed vs average": f"{bcoef[2]:+.4f} ({se[2]:.4f})", "t Air": bcoef[0] / se[0], "t power": bcoef[1] / se[1], "t left": bcoef[2] / se[2]})
    print("    Trustmark-specific effect per +1 SD of the profile (coefficient (cluster-robust SE) in points per PA); the Trustmark average effect itself is in table (1)\n" + pd.DataFrame(res).round(2).to_string(index=False))
    # handedness separately: Trustmark effect for LHB vs RHB
    rows = []
    for lab, col in specs.items():
        y = pa[col].to_numpy(float); out = {"outcome": lab}
        for sd_, nm in ((1.0, "LHB"), (0.0, "RHB")):
            pass
        mu, eff = fe_fit(y, [pd.factorize(pa[c])[0] for c in ("year", "name_clean", "pitcher")] + [pd.factorize(pa["park"].astype(str) + "|" + np.where(pa["left"] == 1, "L", "R"))[0]], iters=30)
        labs = pd.factorize(pa["park"].astype(str) + "|" + np.where(pa["left"] == 1, "L", "R"))[1]; pe = pd.Series(eff[3], index=labs); w = pd.Series(np.bincount(pd.factorize(pa["park"].astype(str) + "|" + np.where(pa["left"] == 1, "L", "R"))[0]), index=labs)
        for s_ in ("L", "R"):
            sel = [i for i in labs if i.endswith("|" + s_)]; base = np.average(pe[sel], weights=w[sel]); out[f"Trustmark, {s_}HB"] = pe[f"TrustmarkPark|{s_}"] - base
        rows.append(out)
    print("\n(3) TRUSTMARK effect by batting side (per PA, each side measured against its own league-average park)\n" + pd.DataFrame(rows).set_index("outcome").round(4).to_string())


if __name__ == "__main__":
    main()
