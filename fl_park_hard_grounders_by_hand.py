"""fl_park_hard_grounders_by_hand.py -- hard-hit grounders / low liners (EV >= 90, LA < 10) at Trustmark by batter side and spray: residual (wOBA - xwOBA) TM minus other parks, hit rate and fielder position.  Game-cluster bootstrap B = 300."""
import warnings
import numpy as np
import pandas as pd
import fl_park_play_the_park as pp
warnings.filterwarnings("ignore")
d = pd.read_parquet("2025_2026_pitches.parquet", columns=["game_id", "stadium", "is_terminal_pitch", "exit_speed", "angle", "direction", "xwoba", "woba", "play_result", "batter_side", "fielded_by_pos"])
b = d[d["is_terminal_pitch"].astype(str).isin(["1", "True", "true"]) & d["exit_speed"].notna() & d["angle"].notna() & d["direction"].notna() & d["woba"].notna() & d["xwoba"].notna() & d["play_result"].isin(["out", "single", "double", "triple", "error", "fielders_choice", "sacrifice"]) & d["batter_side"].isin(["left", "right"])].reset_index(drop=True)
b["tm"] = (b["stadium"] == "TrustmarkPark").astype(int); b["res"] = b["woba"] - b["xwoba"]; b["pull_deg"] = np.where(b["batter_side"] == "right", -b["direction"], b["direction"]); b["hit"] = b["play_result"].isin(["single", "double", "triple"]).astype(float)
b["spray"] = np.select([b["pull_deg"] > 15, b["pull_deg"] < -15], ["pull", "oppo"], "middle"); z = ((b["exit_speed"] >= 90) & (b["angle"] < 10)).to_numpy(); tm = (b["tm"] == 1).to_numpy(); bt = pp.Boot(b["game_id"].to_numpy(), B=300); rows = []
for side in ("right", "left"):
    for sp in ("pull", "middle", "oppo"):
        m = z & ((b["batter_side"] == side) & (b["spray"] == sp)).to_numpy(); e, s = bt.diff(b["res"].to_numpy(), tm, m); h, hs = bt.diff(b["hit"].to_numpy(), tm, m)
        rows.append({"batter": side, "spray": sp, "TM balls": int((m & tm).sum()), "hit rate TM": b.loc[m & tm, "hit"].mean(), "hit rate other": b.loc[m & ~tm, "hit"].mean(), "hit-rate diff": h, "SE ": hs, "residual wOBA diff": e, "SE": s})
print("HARD-HIT GROUNDERS / LOW LINERS (EV >= 90, LA < 10)\n" + pd.DataFrame(rows).round(3).to_string(index=False))
print("\nfielder position on these balls at Trustmark (share), by side:"); t = b[z & tm]; print(pd.crosstab(t["batter_side"], t["fielded_by_pos"].fillna("unk"), normalize="index").round(2).to_string())
