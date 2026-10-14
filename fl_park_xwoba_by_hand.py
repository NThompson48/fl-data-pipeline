"""fl_park_xwoba_by_hand.py -- residual (wOBA - xwOBA) at Trustmark minus other parks, by batter side, in the key exit-speed / launch-angle zones (all hitters; game-cluster bootstrap B = 300)."""
import warnings
import numpy as np
import pandas as pd
import fl_park_play_the_park as pp
from fl_park_xwoba_surface import load
warnings.filterwarnings("ignore")
b = load(); tm = (b["tm"] == 1).to_numpy(); bt = pp.Boot(b["game_id"].to_numpy(), B=300); res = b["res"].to_numpy(); xw = b["xwoba"].to_numpy(); rows = []
zones = [("EV >= 90, LA 10-20", (b["exit_speed"] >= 90) & b["angle"].between(10, 19.999)), ("EV >= 90, LA 20-30", (b["exit_speed"] >= 90) & b["angle"].between(20, 29.999)), ("EV >= 90, LA 30-40", (b["exit_speed"] >= 90) & b["angle"].between(30, 39.999)),
         ("EV >= 90, LA < 10 (grounders / low liners)", (b["exit_speed"] >= 90) & (b["angle"] < 10)), ("EV < 90 (all angles)", b["exit_speed"] < 90), ("ALL balls in play", b["exit_speed"] > 0)]
for lab, z in zones:
    z = z.to_numpy(); row = {"zone": lab}
    for side, name in ((0, "RHH"), (1, "LHH")):
        m = z & (b["left"] == side).to_numpy(); e, s = bt.diff(res, tm, m); ex, _ = bt.diff(xw, tm, m); row[name] = f"{e:+.3f} ({s:.3f}) n{int((m & tm).sum())}"; row[name + " xwOBA TM-other"] = f"{ex:+.3f}"
    rows.append(row)
print("RESIDUAL wOBA - xwOBA, TRUSTMARK MINUS OTHER PARKS, by batter side (SE) and Trustmark balls; xwOBA difference = contact quality\n" + pd.DataFrame(rows).to_string(index=False))
