"""
fl_hitter_park_surface_final.py -- FINAL FIT of the hitter Trustmark translation on the batted-ball residual surface  (user 2026-10-10)
effect(x) = f_TM(x) - f_away(x) on hitter-adjusted residuals (wOBA - xwOBA - own away mean), features = exit speed, launch angle, batter side (depth-3 gradient boosting, 40 balls per leaf), TM balls = VISITING hitters + MISSISSIPPI hitters at home (the latter face 17 different
defences; fl_hitter_park_ms_check.py shows the two groups agree), away balls = hitters away from their own park (visitors elsewhere, Mississippi on the road).  Fitted on 2025 + 2026.
Hitter shift per ball in play = mean effect over his non-Trustmark balls, shrunk toward the league mean by n/(n + 60).  Uncertainty: game-cluster bootstrap (B = 100, surface refit each time) -> per-hitter SD.
Outputs: output/hitter_park_surface_effect.parquet (name_clean, n_balls, shift_bip, shift_bip_sd), output/hitter_park_surface_coefs.json (league mean effect per ball in play, balls in play per PA, global total effect, surface description).
Per-PA effect at full park (used by fl_hitter_park.effect): global_total + BIP_PER_PA x (shift_bip - league_mean), global_total = the fixed-effects global Trustmark wOBA effect from visiting hitters and pitchers, shrunk (0.000).
"""
import json
import warnings

import numpy as np
import pandas as pd

import fl_hitter_park_ms_check as ms
import fl_hitter_park_surface as ps

warnings.filterwarnings("ignore")
BIP_PER_PA = 0.671; B = 100; FEAT = "evla_side"


def shifts(b, seed=0):
    eff = ms.fit_effect_pooled(b, FEAT, seed=seed); return ps.hitter_shift(b, eff, FEAT)


def main():
    b = ms.prep(); print(f"{len(b):,} balls; Trustmark training balls: visiting {int(((b['tm'] == 1) & (b['vis'] == 1)).sum()):,} + Mississippi at home {int(b['ms_home'].sum()):,}")
    sh, mu = shifts(b); n = b[b["tm"] == 0].groupby("name_clean").size(); games = b["game_id"].unique(); by_game = {g: d for g, d in b.groupby("game_id")}; rng = np.random.default_rng(7); draws = []
    for i in range(B):
        sub = pd.concat([by_game[g] for g in rng.choice(games, len(games))]); sub["res_adj"] = sub["res"] - sub["name_clean"].map(ps.hitter_baseline(sub)).fillna(0.0)
        try:
            s_b, mu_b = shifts(sub, seed=i + 1); draws.append(s_b.reindex(sh.index) - mu_b)
        except Exception:
            continue
        if (i + 1) % 25 == 0:
            print(f"  bootstrap {i + 1}/{B}", flush=True)
    D = pd.concat(draws, axis=1); sd = D.std(axis=1); out = pd.DataFrame({"name_clean": sh.index, "n_balls": n.reindex(sh.index).to_numpy(), "shift_bip": sh.to_numpy(), "shift_bip_sd": sd.to_numpy()}); out.to_parquet("output/hitter_park_surface_effect.parquet", index=False)
    old = json.load(open("output/trustmark_profile_coefs.json"))["coefs"]["global"]["final"]; glob = float(old)
    json.dump({"model": "residual surface (wOBA - xwOBA), features " + FEAT, "league_mean_effect_per_bip": float(mu), "bip_per_pa": BIP_PER_PA, "global_total_effect_per_pa": glob, "n0_shrink_balls": 60, "bootstrap_draws": int(D.shape[1]), "training": "visiting hitters + Mississippi hitters at home, 2025-26"}, open("output/hitter_park_surface_coefs.json", "w"), indent=1)
    eff_pa = glob + BIP_PER_PA * (out["shift_bip"] - mu)
    print(f"\nleague mean effect per ball in play {mu:+.4f}; hitters {len(out)}; shift per BIP SD {out['shift_bip'].std():.4f} (range {out['shift_bip'].min():+.3f} to {out['shift_bip'].max():+.3f}); per-PA effect SD {eff_pa.std():.4f} (range {eff_pa.min():+.3f} to {eff_pa.max():+.3f}); median bootstrap SD per BIP {out['shift_bip_sd'].median():.4f}")


if __name__ == "__main__":
    main()
