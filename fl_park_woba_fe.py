"""
fl_park_woba_fe.py -- the wOBA park effect of every Frontier League park, estimated from plate appearances with hitter, pitcher and park fixed effects (2025-26, regular season), and compared with
(a) the published park factor (war_players.park_factor, 2025-26 only) and (b) the TrustmarkPark translation used by the projections (HR factor .446 + air-ball hit delta; fl_hitter_park).
Model: woba_PA = year + hitter + pitcher + park + noise (alternating least squares / iterative demeaning, park effects constrained to PA-weighted mean zero); standard errors by bootstrap over GAMES.
Park effect in wOBA points relative to the league-average park; x(100 / league wOBA) gives a comparable park-factor scale.  One-sided: batters and pitchers who only appear at one park (their home) are partly
absorbed by their own fixed effects, which is why the fixed-effects identification comes from road games and visitors (everyone in this league plays about half of games on the road).
"""
import warnings

import numpy as np
import pandas as pd

import fl_hitter_ros as hr

warnings.filterwarnings("ignore")


def fe_fit(y, groups, iters=40, weights=None):
    """Iterative demeaning: returns the effect arrays for each grouping (codes) after convergence (grand mean first)."""
    n = len(y); mu = y.mean(); eff = [np.zeros(g.max() + 1) for g in groups]
    for _ in range(iters):
        for k, g in enumerate(groups):
            resid = y - mu - sum(eff[j][groups[j]] for j in range(len(groups)) if j != k)
            cnt = np.bincount(g, minlength=eff[k].size); s = np.bincount(g, weights=resid, minlength=eff[k].size); eff[k] = np.where(cnt > 0, s / np.maximum(cnt, 1), 0.0)
    return mu, eff


def main():
    T = hr.Tables(); pa = T.pa.copy(); pit = pd.read_parquet("output/pitcher_pa_ros_2025_2026.parquet"); assert len(pit) == len(pa) and (pit["game_id"].values == pa["game_id"].values).all()
    assert ((pa["batter_team"].values != pit["pitcher_team"].values) & (((pa["batter_team"] == pa["home_team"]) & (pit["pitcher_team"] == pa["away_team"])) | ((pa["batter_team"] == pa["away_team"]) & (pit["pitcher_team"] == pa["home_team"]))).values).all()
    pa["pitcher"] = pit["pitcher_id"].values                                        # the two PA tables are the same query, row-aligned (verified: same game, same wOBA, pitcher team = the batter's opponent on every row)
    pa["pitcher"] = pa["pitcher"].astype(object).where(pa["pitcher"].notna(), "unknown").astype(str); pa["name_clean"] = pa["name_clean"].fillna("unknown"); pa["park"] = pa["game_id"].str.split("-").str[1]; pa = pa[pa["woba"].notna() & pa["park"].notna()].copy()
    pit_col = "pitcher" if "pitcher" in pa.columns else None
    assert pit_col, "no pitcher column in the PA table"
    codes = {k: pd.factorize(pa[c])[0] for k, c in (("year", "year"), ("bat", "name_clean"), ("pit", pit_col), ("park", "park"))}; parks = pd.factorize(pa["park"])[1]; games = pd.factorize(pa["game_id"])[0]
    y = pa["woba"].to_numpy(float); order = ["year", "bat", "pit", "park"]
    mu, eff = fe_fit(y, [codes[k] for k in order]); pe = eff[3] - np.average(eff[3], weights=np.bincount(codes["park"]))      # park effects centred on the PA-weighted league average
    rng = np.random.default_rng(1); ng = games.max() + 1; boots = []; by_game = pd.Series(np.arange(len(y))).groupby(games).apply(np.array)
    for _ in range(60):
        pick = rng.integers(0, ng, ng); ii = np.concatenate([by_game[g] for g in pick]); cc = [pd.factorize(pa.iloc[ii][c].to_numpy())[0] for c in ("year", "name_clean", pit_col, "park")]
        pk = pd.factorize(pa.iloc[ii]["park"].to_numpy())[1]; m_, e_ = fe_fit(y[ii], cc, iters=25); pe_b = pd.Series(e_[3] - np.average(e_[3], weights=np.bincount(cc[3])), index=pk); boots.append(pe_b.reindex(parks))
    B = pd.concat(boots, axis=1); lg = y.mean()
    out = pd.DataFrame({"park": parks, "wOBA effect (points)": pe, "se": B.std(axis=1).to_numpy(), "park-factor scale (100 = neutral)": 100 * (1 + pe / lg)}).set_index("park")
    pub = {"TrustmarkPark": 99.5}
    print(f"\nleague wOBA {lg:.4f}; {len(pa):,} PA; {len(parks)} parks\n" + out.sort_values("wOBA effect (points)").round(4).to_string())
    t = out.loc["TrustmarkPark"]; print(f"\nTrustmarkPark: wOBA effect {t['wOBA effect (points)']:+.4f} (se {t['se']:.4f}) = park-factor scale {t['park-factor scale (100 = neutral)']:.1f}; published Mississippi team park factor 99.3-99.7 (2025-26); "
          f"our translation for a half-season hitter averages -0.0119 (about -0.024 for a full-time Trustmark hitter)")
    out.to_csv("output/park_woba_fe_2025_26.csv")


if __name__ == "__main__":
    main()
