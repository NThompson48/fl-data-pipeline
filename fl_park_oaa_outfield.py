"""
fl_park_oaa_outfield.py -- Trustmark outfield defence and fly-ball hits  (user 2026-10-09: "our OFs had a huge OAA at home and so did opposing outfielders at Trustmark")
Data: 2025_2026_pitches.parquet terminal pitches with an OAA value (outfield-fielded balls only, n ~ 11.9k; OAA per play in [-1, 1]: catch = 1 - p, miss = -p, p = the tracking vendor's catch probability, park-blind).
(1) OAA per outfield ball by park, by home vs visiting FIELDERS (a park property would lift both; roster quality only the home side), game-cluster bootstrap for Trustmark vs the other parks.
(2) Is it a lower HIT rate on comparable balls?  Cross-fitted (5-fold) gradient-boosting expected hit probability from batted-ball traits only (exit speed, launch angle, spray direction, distance, hang time, fielder position,
    batter side; NO park / team information); residual = hit - expected, by park and by fielder side.
(3) Does it depend on the hitter's air-ball profile?  Among VISITING hitters, residual hit rate on outfield balls regressed on the shrunk Air% profile z_air (per +1 SD), at Trustmark vs the other parks (interaction),
    and the same split of the OAA value itself.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
TM = "TrustmarkPark"


def load():
    d = pd.read_parquet("2025_2026_pitches.parquet")
    b = d[d["oaa"].notna() & d["is_terminal_pitch"].astype(str).isin(["1", "True", "true"])].copy()
    b["hit"] = b["play_result"].isin(["single", "double", "triple"]).astype(float)
    b = b[b["play_result"].isin(["out", "sacrifice", "single", "double", "triple", "error", "fielders_choice"])].reset_index(drop=True)
    b["field_home"] = (b["pitcher_team"] == b["home_team"]).astype(int); b["tm"] = (b["stadium"] == TM).astype(int)
    pa = __import__("fl_hitter_ros").Tables().pa[["game_id", "pa_id", "name_clean"]].drop_duplicates(["game_id", "pa_id"])
    b = b.merge(pa, on=["game_id", "pa_id"], how="left")
    prof = pd.read_parquet("output/hitter_park_profile.parquet"); b["z_air"] = b["name_clean"].map(prof["z_air"]); b["z_power"] = b["name_clean"].map(prof["z_power"])
    return b


def expected_hit(b):
    X = pd.DataFrame({"ev": b["exit_speed"], "angle": b["angle"], "dir": b["direction"], "dist": b["distance"], "hang": b["hang_time"], "pos": b["fielded_by_pos"].astype("category").cat.codes, "side": (b["batter_side"].astype(str) == "Left").astype(int)})
    p = np.zeros(len(b)); y = b["hit"].to_numpy()
    for tr, te in GroupKFold(5).split(X, y, groups=b["game_id"]):
        m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=250, min_samples_leaf=40).fit(X.iloc[tr], y[tr]); p[te] = m.predict_proba(X.iloc[te])[:, 1]
    return p


def boot_diff(b, col, mask_a, mask_b, B=400, seed=3):
    rng = np.random.default_rng(seed); g = b["game_id"].to_numpy(); by = pd.Series(np.arange(len(b))).groupby(g).apply(np.array); ug = by.index.to_numpy(); v = b[col].to_numpy(float); a = mask_a.to_numpy(); c = mask_b.to_numpy(); out = []
    for _ in range(B):
        ii = np.concatenate([by[u] for u in rng.choice(ug, len(ug))]); x = a[ii]; z = c[ii]; out.append(v[ii][x].mean() - v[ii][z].mean())
    return np.array(out)


def main():
    b = load(); print(f"{len(b):,} outfield-fielded balls in play with OAA (hit rate {b['hit'].mean():.3f}); named hitter profile on {b['z_air'].notna().mean():.0%}")
    b["p_hat"] = expected_hit(b); b["resid"] = b["hit"] - b["p_hat"]
    print(f"expected-hit model (traits only, cross-fitted): Brier {np.mean((b['hit'] - b['p_hat']) ** 2):.4f} vs base {b['hit'].var():.4f}")
    # (1) + (2) by park
    t = b.groupby("stadium").agg(n=("oaa", "size"), OAA_per_ball=("oaa", "mean"), hit_rate=("hit", "mean"), expected_hit=("p_hat", "mean"), hit_minus_expected=("resid", "mean")).sort_values("OAA_per_ball")
    print("\n(1)-(2) BY PARK: outfield balls\n" + t.round(3).to_string()); print(f"  correlation across parks, OAA per ball vs hit-minus-expected: {t['OAA_per_ball'].corr(t['hit_minus_expected']):+.2f}")
    tm, ot = b["tm"] == 1, b["tm"] == 0
    for col, lab in (("oaa", "OAA per ball"), ("resid", "hit minus expected")):
        dd = boot_diff(b, col, tm, ot); est = b.loc[tm, col].mean() - b.loc[ot, col].mean(); print(f"  Trustmark minus other parks, {lab}: {est:+.4f} (bootstrap SE {dd.std():.4f}, 90% [{np.percentile(dd, 5):+.4f}, {np.percentile(dd, 95):+.4f}])")
    print("\nTrustmark by FIELDER side (home vs visiting outfielders)")
    s = b[tm].groupby("field_home").agg(n=("oaa", "size"), OAA_per_ball=("oaa", "mean"), hit_minus_expected=("resid", "mean")); print(s.round(4).to_string())
    for col in ("oaa", "resid"):
        dd = boot_diff(b, col, tm & (b["field_home"] == 1), ot & (b["field_home"] == 1)); dv = boot_diff(b, col, tm & (b["field_home"] == 0), ot & (b["field_home"] == 0))
        print(f"  {col}: home fielders at Trustmark vs home fielders elsewhere {b.loc[tm & (b['field_home'] == 1), col].mean() - b.loc[ot & (b['field_home'] == 1), col].mean():+.4f} (SE {dd.std():.4f}); visiting fielders at Trustmark vs visiting elsewhere "
              f"{b.loc[tm & (b['field_home'] == 0), col].mean() - b.loc[ot & (b['field_home'] == 0), col].mean():+.4f} (SE {dv.std():.4f})")
    # by fielder position at Trustmark
    print("\nTrustmark vs other parks by outfield position (hit minus expected | OAA per ball)")
    for pos in ("LF", "CF", "RF"):
        m = b["fielded_by_pos"] == pos; print(f"  {pos}: Trustmark n {int((m & tm).sum())}: resid {b.loc[m & tm, 'resid'].mean():+.4f}, OAA {b.loc[m & tm, 'oaa'].mean():+.4f} | other parks resid {b.loc[m & ot, 'resid'].mean():+.4f}, OAA {b.loc[m & ot, 'oaa'].mean():+.4f}")
    # (3) air-ball profile among VISITING hitters
    v = b[(b["field_home"] == 1) & b["z_air"].notna()].copy(); print(f"\n(3) VISITING hitters, outfield balls with a profile: n = {len(v):,} ({int(v['tm'].sum())} at Trustmark)")
    for col, lab in (("resid", "hit minus expected"), ("oaa", "OAA per ball (as the FIELDER's value; negative = hitter beats the defence)")):
        rows = []
        for nm, m in (("Trustmark", v["tm"] == 1), ("other parks", v["tm"] == 0)):
            x = v.loc[m, "z_air"].to_numpy(); y = v.loc[m, col].to_numpy(); sl = np.polyfit(x, y, 1)[0]; bs = []
            gg = v.loc[m, "game_id"].to_numpy(); by = pd.Series(np.arange(m.sum())).groupby(gg).apply(np.array); ug = by.index.to_numpy(); rng = np.random.default_rng(5)
            for _ in range(400):
                ii = np.concatenate([by[u] for u in rng.choice(ug, len(ug))]); bs.append(np.polyfit(x[ii], y[ii], 1)[0])
            rows.append({"park group": nm, "n": int(m.sum()), "slope per +1 SD Air%": sl, "bootstrap SE": float(np.std(bs))})
        r = pd.DataFrame(rows); diff = r["slope per +1 SD Air%"].iloc[0] - r["slope per +1 SD Air%"].iloc[1]; se = float(np.sqrt((r["bootstrap SE"] ** 2).sum())); print(f"  {lab}\n" + r.round(4).to_string(index=False) + f"\n  Trustmark minus other: {diff:+.4f} (SE ~ {se:.4f}, t {diff / se:+.1f})")


if __name__ == "__main__":
    main()
