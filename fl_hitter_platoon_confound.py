"""
fl_hitter_platoon_confound.py -- is the Trustmark 'same-hand matchup' penalty a PARK effect or Mississippi's PITCHING STAFF?  (follows fl_hitter_profile_screen_bip.py: same-hand x Trustmark -.059 wOBA on contact, t -2.9)
Platoon coefficient (wOBA on contact, same-handed minus opposite-handed matchup; hitter + pitcher fixed effects inside each group; game-cluster SEs) in groups of balls in play:
 (a) visiting hitters at Trustmark (facing the Mississippi staff)      (b) Mississippi pitchers on the ROAD (facing home hitters elsewhere)
 (c) Mississippi hitters at home (facing visiting pitchers)             (d) Mississippi hitters on the road
 (e) everything else (neither team Mississippi), park fixed effects
A park effect would show in (a) AND (c); a Mississippi-staff effect in (a) and (b); a Mississippi-lineup effect in (c) and (d).
"""
import warnings
import numpy as np
import pandas as pd
import fl_hitter_profile_screen_bip as sb
from fl_park_profile_fe import cluster_ols, partial_out
warnings.filterwarnings("ignore")
MS = "Mississippi Mud Monsters"

est = sb.build(); est = est[est["same_hand"].notna()].reset_index(drop=True); bt, ht, at = est["batter_team"], est["home_team"], est["away_team"]
ms_pitch_home = (ht == MS) & (bt == at); ms_pitch_road = (at == MS) & (bt == ht)
groups = {"(a) visiting hitters at Trustmark (vs MS staff)": (est["tm"] == 1) & (bt != MS), "(b) MS pitchers on the road": ms_pitch_road, "(c) MS hitters at home (vs visiting pitchers)": (bt == MS) & (bt == ht), "(d) MS hitters on the road": (bt == MS) & (bt != ht),
          "(e) all other games (no MS, not Trustmark)": (bt != MS) & (ht != MS) & (at != MS)}
rows = []
for lab, m in groups.items():
    s = est[m.to_numpy()].reset_index(drop=True)
    g = [pd.factorize(s["name_clean"])[0], pd.factorize(s["pitcher"])[0], pd.factorize(s["year"])[0]] + ([pd.factorize(s["park"])[0]] if lab.startswith("(e)") else [])
    x = partial_out(s["same_hand"].to_numpy(float)[:, None], g, iters=30)[:, 0]; y = partial_out(s["y"].to_numpy(float)[:, None], g, iters=30)[:, 0]; hr = partial_out(s["o_hr"].to_numpy(float)[:, None], g, iters=30)[:, 0]; cl = pd.factorize(s["game_id"])[0]
    b, se = cluster_ols(x[:, None], y, cl); bh, sh = cluster_ols(x[:, None], hr, cl)
    rows.append({"group": lab, "balls": len(s), "same-hand share": s["same_hand"].mean(), "wOBAcon: same minus opposite": b[0], "SE": se[0], "t": b[0] / se[0], "HR/BIP: same minus opposite": bh[0], "HR SE": sh[0]})
print(pd.DataFrame(rows).round(4).to_string(index=False))
# the pitcher-handedness mix of the MS staff vs visitors
ph = est[(est["tm"] == 1) & (bt != MS)]; print(f"\nMS staff at home: share of balls thrown by lefties {ph['p_lefty'].mean():.3f}; same-hand share vs visiting hitters {ph['same_hand'].mean():.3f}; league-wide same-hand share {est['same_hand'].mean():.3f}")
