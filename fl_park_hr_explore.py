"""
fl_park_hr_explore.py -- the Trustmark HOME RUN suppression, explored  (user 2026-10-09: "The Mud Monsters have such a hugely suppressed HR park factor that I want to explore this")
Data: 2025_2026_pitches.parquet, every tracked ball in play (80k, HR included, 2025-26 regular season).  Two expected-HR benchmarks that know nothing about the park:
  (T) traits model: cross-fitted (5-fold by game) gradient boosting of HR on exit speed, launch angle, field direction, batter side;
  (M) MLB-park count: the tracking data's own 'in how many of 17 MLB parks would this ball be a home run', mapped to an FL-average-park HR probability by isotonic regression fitted on the OTHER 17 FL parks only.
1  HR factor by park (actual / expected), game-cluster bootstrap for Trustmark; HR per 100 balls; HR lost per Trustmark game.
2  WHERE it is lost: by field sector x batter side; by exit-speed x launch-angle zone; by projected distance.
3  EFFECTIVE FENCE: per sector, the projected distance at which a hard-hit (EV >= 92) airborne (15-45 deg) ball is a HR 50% of the time, Trustmark vs the other parks (logistic in distance), in feet.
4  WHAT HAPPENS to the would-be home runs at Trustmark (balls with MLB-park share >= .7 / .3-.7): outcome mix and wOBA vs the other parks.
5  THE MUD MONSTERS: HR vs expected for MS hitters at home / on the road and visiting hitters at Trustmark (the same balls seen from the pitching side), i.e. has the roster adapted?
6  WEATHER: temperature and wind coverage at Trustmark; carry (distance given exit speed and angle).
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
TM = "TrustmarkPark"; MS = "Mississippi Mud Monsters"
SECTORS = [(-90, -30, "LF line"), (-30, -10, "LF gap"), (-10, 10, "CF"), (10, 30, "RF gap"), (30, 90, "RF line")]


def load():
    d = pd.read_parquet("2025_2026_pitches.parquet")
    b = d[d["is_terminal_pitch"].astype(str).isin(["1", "True", "true"]) & d["exit_speed"].notna() & d["angle"].notna() & d["direction"].notna() & d["play_result"].isin(["out", "single", "double", "triple", "error", "fielders_choice", "sacrifice", "home_run"]) & d["batter_side"].isin(["left", "right"])].copy().reset_index(drop=True)
    b["tm"] = (b["stadium"] == TM).astype(int); b["hr"] = (b["play_result"] == "home_run").astype(float); b["left"] = (b["batter_side"] == "left").astype(int); b["mlb"] = (b["num_parks_hr"] / b["num_parks_hr_denom"]).fillna(0.0)
    b["sector"] = pd.cut(b["direction"], [s[0] for s in SECTORS] + [90.001], labels=[s[2] for s in SECTORS], right=False)
    b["batter_home"] = (b["batter_team"] == b["home_team"]).astype(int); return b


def expected(b):
    X = pd.DataFrame({"ev": b["exit_speed"], "angle": b["angle"], "dir": b["direction"], "side": b["left"]}); y = b["hr"].to_numpy(); pt = np.zeros(len(b))
    for tr, te in GroupKFold(5).split(X, y, groups=b["game_id"]):
        pt[te] = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.05, max_iter=300, min_samples_leaf=50).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    b["exp_t"] = pt; nt = (b["tm"] == 0).to_numpy(); pm = np.zeros(len(b)); iso_all = IsotonicRegression(out_of_bounds="clip").fit(b.loc[nt, "mlb"], b.loc[nt, "hr"]); pm[~nt] = 0
    for tr, te in GroupKFold(5).split(b[nt], groups=b.loc[nt, "game_id"]):
        idx = np.where(nt)[0]; m = IsotonicRegression(out_of_bounds="clip").fit(b.loc[idx[tr], "mlb"], b.loc[idx[tr], "hr"]); pm[idx[te]] = m.predict(b.loc[idx[te], "mlb"])
    pm[~nt] = iso_all.predict(b.loc[~nt, "mlb"]); b["exp_m"] = pm; return b


class Boot:
    def __init__(self, games, B=400, seed=11):
        codes, uniq = pd.factorize(games); rng = np.random.default_rng(seed); self.W = [np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[codes] for _ in range(B)]

    def ratio(self, a, e, mask):
        est = a[mask].sum() / e[mask].sum(); bs = [(w * a * mask).sum() / (w * e * mask).sum() for w in self.W if (w * e * mask).sum() > 0]; return est, float(np.std(bs))


def part1(b, bt):
    rows = []
    for st, g in b.groupby("stadium"):
        m = (b["stadium"] == st).to_numpy(); rt, st_t = bt.ratio(b["hr"].to_numpy(), b["exp_t"].to_numpy(), m); rm, st_m = bt.ratio(b["hr"].to_numpy(), b["exp_m"].to_numpy(), m)
        rows.append({"park": st, "balls": len(g), "HR per 100": 100 * g["hr"].mean(), "expected (traits)": 100 * g["exp_t"].mean(), "factor (traits)": rt, "SE": st_t, "expected (MLB count)": 100 * g["exp_m"].mean(), "factor (MLB count)": rm, "SE ": st_m})
    t = pd.DataFrame(rows).sort_values("factor (traits)"); print("1. HR FACTOR BY PARK (actual / expected; 1.00 = the average park)\n" + t.round(3).to_string(index=False))
    tm = b[b["tm"] == 1]; games = tm["game_id"].nunique(); lost = (tm["exp_t"].sum() - tm["hr"].sum()); lost_m = (tm["exp_m"].sum() - tm["hr"].sum())
    print(f"\n   Trustmark: {int(tm['hr'].sum())} HR in {len(tm):,} tracked balls vs {tm['exp_t'].sum():.0f} expected (traits) / {tm['exp_m'].sum():.0f} (MLB count) -> {lost:.0f} / {lost_m:.0f} home runs 'lost' = {lost / games:.2f} / {lost_m / games:.2f} per tracked game ({games} games)")


def part2(b, bt):
    tm = (b["tm"] == 1).to_numpy(); hr, e = b["hr"].to_numpy(), b["exp_t"].to_numpy(); rows = []
    for side in ("right", "left"):
        for _, _, sl in SECTORS:
            m = tm & ((b["sector"] == sl) & (b["batter_side"] == side)).to_numpy(); r, s = bt.ratio(hr, e, m); rows.append({"batter": side, "sector": sl, "balls at TM": int(m.sum()), "HR": int(hr[m].sum()), "expected": e[m].sum(), "ratio": r, "SE": s, "HR lost": e[m].sum() - hr[m].sum()})
    print("\n2a. BY FIELD SECTOR AND BATTER SIDE (traits expectation)\n" + pd.DataFrame(rows).round(3).to_string(index=False))
    rows = []
    for ev_lo, ev_hi in ((0, 95), (95, 100), (100, 105), (105, 130)):
        for a_lo, a_hi in ((15, 25), (25, 30), (30, 35), (35, 50)):
            m = tm & ((b["exit_speed"] >= ev_lo) & (b["exit_speed"] < ev_hi) & (b["angle"] >= a_lo) & (b["angle"] < a_hi)).to_numpy()
            if e[m].sum() >= 1.5:
                r, s = bt.ratio(hr, e, m); rows.append({"exit speed": f"{ev_lo}-{ev_hi}" if ev_hi < 130 else f"{ev_lo}+", "launch angle": f"{a_lo}-{a_hi}", "balls at TM": int(m.sum()), "HR": int(hr[m].sum()), "expected": e[m].sum(), "ratio": r, "SE": s, "HR lost": e[m].sum() - hr[m].sum()})
    print("\n2b. BY EXIT SPEED x LAUNCH ANGLE (cells with >= 1.5 expected HR)\n" + pd.DataFrame(rows).round(3).to_string(index=False))
    rows = []
    for lo, hi in ((0, 300), (300, 330), (330, 360), (360, 380), (380, 400), (400, 600)):
        m = tm & ((b["distance"] >= lo) & (b["distance"] < hi)).to_numpy(); o = (~tm) & ((b["distance"] >= lo) & (b["distance"] < hi)).to_numpy(); air = (b["angle"] >= 15).to_numpy(); mm, oo = m & air, o & air
        rows.append({"projected distance": f"{lo}-{hi}", "TM airborne balls": int(mm.sum()), "TM HR share": hr[mm].mean() if mm.sum() else np.nan, "other-park airborne balls": int(oo.sum()), "other HR share": hr[oo].mean() if oo.sum() else np.nan})
    print("\n2c. HR SHARE OF AIRBORNE (>= 15 deg) BALLS BY PROJECTED DISTANCE\n" + pd.DataFrame(rows).round(3).to_string(index=False))


def d50(sub):
    if sub["hr"].sum() < 5 or (1 - sub["hr"]).sum() < 5:
        return np.nan
    lr = LogisticRegression(C=1e4, max_iter=500).fit(sub[["distance"]], sub["hr"]); return -lr.intercept_[0] / lr.coef_[0][0]


def part3(b):
    s = b[(b["angle"] >= 15) & (b["angle"] <= 45) & (b["exit_speed"] >= 92) & b["distance"].between(280, 480)].copy(); rows = []; rng = np.random.default_rng(2); games = s["game_id"].unique()
    by = {k: g for k, g in s.groupby("game_id")}
    for _, _, sl in SECTORS:
        a = s[(s["sector"] == sl) & (s["tm"] == 1)]; o = s[(s["sector"] == sl) & (s["tm"] == 0)]; est = d50(a) - d50(o); bs = []
        for _ in range(150):
            sub = pd.concat([by[g] for g in rng.choice(games, len(games))]); sa = sub[(sub["sector"] == sl) & (sub["tm"] == 1)]; so = sub[(sub["sector"] == sl) & (sub["tm"] == 0)]; bs.append(d50(sa) - d50(so))
        bs = np.array(bs, float); rows.append({"sector": sl, "TM balls / HR": f"{len(a)} / {int(a['hr'].sum())}", "D50 Trustmark (ft)": d50(a), "D50 other parks (ft)": d50(o), "Trustmark deeper by (ft)": est, "bootstrap SE": np.nanstd(bs), "bootstrap n ok": int(np.isfinite(bs).sum())})
    print("\n3. EFFECTIVE FENCE: projected distance where a hard-hit (EV >= 92), 15-45 deg ball is a HR 50% of the time\n" + pd.DataFrame(rows).round(1).to_string(index=False))


def part4(b):
    rows = []
    for lab, lo, hi in (("would be HR in >= 70% of MLB parks", 0.7, 1.01), ("30-70%", 0.3, 0.7), ("5-30%", 0.05, 0.3)):
        for park, m0 in (("Trustmark", b["tm"] == 1), ("other FL parks", b["tm"] == 0)):
            g = b[m0 & (b["mlb"] >= lo) & (b["mlb"] < hi)]; pr = g["play_result"]
            rows.append({"ball": lab, "park": park, "n": len(g), "HR": (pr == "home_run").mean(), "2B/3B": pr.isin(["double", "triple"]).mean(), "single": (pr == "single").mean(), "out/other": 1 - pr.isin(["home_run", "double", "triple", "single"]).mean(), "mean wOBA": g["woba"].mean()})
    t = pd.DataFrame(rows); print("\n4. WHAT HAPPENS TO WOULD-BE HOME RUNS (outcome mix)\n" + t.round(3).to_string(index=False))
    for lab, lo, hi in (("would be HR in >= 70% of MLB parks", 0.7, 1.01),):
        a = t[(t["ball"] == lab) & (t["park"] == "Trustmark")].iloc[0]; o = t[(t["ball"] == lab) & (t["park"] == "other FL parks")].iloc[0]; print(f"   cost per such ball at Trustmark: {a['mean wOBA'] - o['mean wOBA']:+.3f} wOBA")


def part5(b, bt):
    hr, e = b["hr"].to_numpy(), b["exp_t"].to_numpy(); bh = b["batter_home"].to_numpy() == 1; tm = b["tm"].to_numpy() == 1; msb = (b["batter_team"] == MS).to_numpy(); msp = (b["pitcher_team"] == MS).to_numpy(); rows = []
    for lab, m in (("MS hitters at home (facing visiting pitchers)", msb & tm & bh), ("visiting hitters at Trustmark (facing MS pitchers)", ~msb & tm & ~bh), ("MS hitters on the road", msb & ~bh), ("MS pitchers on the road (other teams' hitters at home)", msp & ~tm & (b["batter_team"] != MS).to_numpy()),
                   ("all other balls outside Trustmark", ~tm)):
        r, s = bt.ratio(hr, e, m); rows.append({"group": lab, "balls": int(m.sum()), "HR": int(hr[m].sum()), "expected": e[m].sum(), "HR/expected": r, "SE": s})
    print("\n5. THE MUD MONSTERS: HR vs expected (traits benchmark)\n" + pd.DataFrame(rows).round(3).to_string(index=False))
    for lab, m in (("MS hitters at home", msb & tm & bh), ("visiting hitters at Trustmark", ~msb & tm & ~bh)):
        g = b[m]; print(f"   {lab}: mean launch angle {g['angle'].mean():.1f}, exit speed {g['exit_speed'].mean():.1f}, share of airborne >= 25 deg {(g['angle'] >= 25).mean():.3f}, pull share (direction toward own pull side) {((np.where(g['left'] == 1, g['direction'], -g['direction'])) > 15).mean():.3f}")


def part6(b):
    tm = b[b["tm"] == 1]; print("\n6. WEATHER: temp coverage at Trustmark %.0f%% (mean %.0f F); wind coverage %.0f%%" % (100 * tm["temp"].notna().mean(), tm["temp"].mean(), 100 * tm["wind"].notna().mean()))
    w = tm["wind"].dropna().astype(str); sp = w.str.extract(r"(\d+)\s*MPH")[0].astype(float); dr = np.where(w.str.contains(" out"), "out", np.where(w.str.contains(" in"), "in", np.where(w.str.contains("cross|to R|to L", regex=True), "cross", "calm")))
    t = pd.DataFrame({"speed": sp, "dir": dr}); print("   wind (Trustmark games with a reading): " + ", ".join(f"{k} {v}" for k, v in t["dir"].value_counts().items()) + f"; mean speed {t['speed'].mean():.1f} mph")


def main():
    b = expected(load()); print(f"{len(b):,} tracked balls in play; HR {int(b['hr'].sum()):,} ({100 * b['hr'].mean():.2f}% of balls); Trustmark {int(b['tm'].sum()):,} balls\n")
    bt = Boot(b["game_id"].to_numpy()); part1(b, bt); part2(b, bt); part3(b); part4(b); part5(b, bt); part6(b)


if __name__ == "__main__":
    main()
