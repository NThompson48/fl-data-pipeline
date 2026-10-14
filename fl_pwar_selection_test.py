"""
fl_pwar_selection_test.py -- is the pitcher P(WAR>0) optimism for low-innings arms a SURVIVORSHIP / selection artifact of the backtest sample?  (user 2026-10-09: "some survivorship here, maybe?")
Backtest 2025 -> 2026 (fl_pwar_calibration build: out-of-sample projections, P = P(RA9 < replacement) with SD sqrt(185.16 / projected IP)).  The calibration sample only contains pitchers with >= 10 IP in 2026, so pitchers who
were released / pulled after bad outings (or never pitched) are missing: if bad outcomes lead to fewer innings, the sample is selected on the outcome.  Vary the inclusion rule and look at how the observed beat-replacement rate
for LOW projected-IP arms (< 22 projected IP) changes: >= 1 IP, >= 5, >= 10 (production backtest), >= 20, >= 30; also report how many projected pitchers have NO 2026 innings at all (the unobserved).
Selection on the OUTCOME would move the observed rate monotonically with the threshold; no movement means the threshold is not what drives the gap.
"""
import warnings
import numpy as np
import pandas as pd
import fl_pwar_calibration as cal

warnings.filterwarnings("ignore")
cal.RETURN_ALL = True
b, war = cal.main()
b = b.copy(); low = b["ip_proj"] < 22
print(f"projected pitchers (with a 2025-based projection): {len(b)}; with any 2026 innings {int((b['ip'] > 0).sum())}; with >= 10 IP {int((b['ip'] >= 10).sum())}")
rows = []
for thr in (0.1, 1, 5, 10, 20, 30):
    for lab, m in (("all projected IP", np.ones(len(b), bool)), ("< 22 projected IP", low.to_numpy()), (">= 22 projected IP", ~low.to_numpy())):
        d = b[(b["ip"] >= thr) & m]
        if len(d) < 15:
            continue
        rows.append({"min 2026 IP": thr, "group": lab, "n": len(d), "mean predicted P": d["P"].mean(), "observed share beating replacement": d["y"].mean(), "gap (pred - obs)": d["P"].mean() - d["y"].mean(), "mean 2026 IP": d["ip"].mean()})
print(pd.DataFrame(rows).round(3).to_string(index=False))
# IP-weighted view (a pitcher with 3 IP counts little): removes the arbitrary threshold
for lab, m in (("< 22 projected IP", low.to_numpy()), (">= 22 projected IP", ~low.to_numpy())):
    d = b[(b["ip"] > 0) & m]; print(f"IP-weighted, any 2026 IP, {lab}: n = {len(d)}, predicted {np.average(d['P'], weights=d['ip']):.3f} vs observed {np.average(d['y'], weights=d['ip']):.3f}")
# projected IP vs realised IP for the low group (is the shortfall in innings?)
d = b[low & (b["ip"] > 0)]; print(f"low projected-IP arms with any 2026 IP: projected IP {d['ip_proj'].mean():.1f}, realised IP {d['ip'].mean():.1f}, share with < 10 IP {np.mean(d['ip'] < 10):.0%}")

# ---- by REALIZED innings bin, low-projected arms: where does the gap live?  (bootstrap over pitchers)
print("\nLOW projected-IP arms (< 22 projected IP), by realized 2026 IP")
d = b[low].copy(); d["bin"] = pd.cut(d["ip"], [0, 20, 30, 1000], labels=["10-20 IP", "20-30 IP", "30+ IP"], right=False)
rng = np.random.default_rng(5); rows = []
for lab, g in d.groupby("bin", observed=True):
    bs = [ (lambda s: s["P"].mean() - s["y"].mean())(g.iloc[rng.integers(0, len(g), len(g))]) for _ in range(2000)]
    rows.append({"realized IP": lab, "n": len(g), "predicted": g["P"].mean(), "observed": g["y"].mean(), "gap": g["P"].mean() - g["y"].mean(), "gap 90% (bootstrap)": f"[{np.percentile(bs, 5):+.3f}, {np.percentile(bs, 95):+.3f}]", "mean RA9": g["target"].mean(), "mean predicted RA9": g["pred"].mean()})
print(pd.DataFrame(rows).round(3).to_string(index=False))
# informative exposure: among low-projected arms, does realized RA9 (relative to the prediction) correlate with innings actually pitched?
d["miss"] = d["target"] - d["pred"]; print(f"corr(RA9 miss, log realized IP) among low-projected arms: {np.corrcoef(d['miss'], np.log(d['ip']))[0, 1]:+.2f}  (negative = arms who pitched worse than predicted pitched fewer innings)")
for lab, m in (("< 22 projected IP", low), (">= 22 projected IP", ~low)):
    g = b[m]; print(f"  {lab}: corr {np.corrcoef(g['target'] - g['pred'], np.log(g['ip']))[0, 1]:+.2f} (n = {len(g)})")
