"""
fl_hitter_wobacon_test.py -- replace the HR/PA and BABIP components with wOBA on contact (wOBAcon)?

Identity (exact, from the wOBA definition):
    wOBA = (wBB*uBB + wHBP*HBP + contact * wOBAcon) / den
    contact = AB - SO + SF   (balls put in play, home runs included)       wOBAcon = (w1B*1B + w2B*2B + w3B*3B + wHR*HR) / contact
    den     = AB + uBB + SF + HBP
So if K%, BB%, HBP% and wOBAcon are projected, wOBA follows with NO fitted component weights.

Compared (leave-one-transition-out on 2022-26, next-year PA >= 100, park-neutral wOBA, all with the same regression-to-the-league machinery):
   direct            regressed wOBA                         (shipped half a)
   components        K%, BB%, HR/PA, BABIP -> fitted weights (shipped half b)
   shipped blend     50/50 of the two
   wOBAcon identity  K%, BB%, HBP%, wOBAcon -> wOBA by the identity
   blends            identity + direct, identity + shipped blend
Then 2025 -> 2026 only: does xwOBAcon (mean xwOBA on contact, Trackman) predict next-year wOBAcon better than wOBAcon itself, and does it help the final wOBA?

Usage:  python fl_hitter_wobacon_test.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sqlalchemy import text

import fl_eval_weights as ew
import fl_hitter_baseline_backtest as bt
import fl_hitter_model_dev as dv
import fl_hitter_panel as hp
from fl_supabase import read_sql_df

warnings.filterwarnings("ignore")


def add_wcon(p):
    w = hp.woba_weights()
    p = p.copy()
    p["sf"] = p["sf"].fillna(0); p["sh"] = p["sh"].fillna(0); p["hbp"] = p["hbp"].fillna(0); p["ibb"] = p["ibb"].fillna(0)
    p["contact"] = p["ab"] - p["so"] + p["sf"]
    p["wcon_num"] = w["1B"] * p["x1b"] + w["2B"] * p["x2b"] + w["3B"] * p["x3b"] + w["HR"] * p["hr"]
    p["wcon"] = p["wcon_num"] / p["contact"].replace(0, np.nan)
    p["den"] = p["ab"] + (p["bb"] - p["ibb"]) + p["sf"] + p["hbp"]
    # identity check
    ident = (w["BB"] * (p["bb"] - p["ibb"]) + w["HBP"] * p["hbp"] + p["wcon_num"]) / p["den"].replace(0, np.nan)
    p["woba_ident"] = ident
    # park-neutral wOBAcon: the wOBA add-back (per PA) re-based to contact
    p["wcon_n"] = p["wcon"] + (p["woba_n"] - p["woba_raw"]) * p["den"] / p["contact"].replace(0, np.nan)
    p["con_r"] = p["contact"] / p["pa"]; p["den_r"] = p["den"] / p["pa"]
    p["ubb_r"] = (p["bb"] - p["ibb"]) / p["pa"]
    return p, w


def project(q, stat, delta, n0_by_t, t):
    d = bt.transitions(q, stat)
    return d


def loto_pred(q, stat, delta):
    """LOTO regressed prediction for every (name, t) row of this stat; n0 fit on the other transitions."""
    d = bt.transitions(q, stat)
    E, n0s = bt.loto(d, delta)
    return E.set_index(["name_clean", "t"]), n0s


def main():
    p = dv.load()
    p, w = add_wcon(p)
    ok = p["woba_raw"].notna() & p["woba_ident"].notna()
    print(f"identity check: max |wOBA - rebuilt| = {(p.loc[ok,'woba_raw'] - p.loc[ok,'woba_ident']).abs().max():.4f}; mean {(p.loc[ok,'woba_raw'] - p.loc[ok,'woba_ident']).abs().mean():.5f}  "
          f"(2022-24 IBB-only-in-2022 and sac-bunt rows explain the rest)")
    q = p.copy(); q["woba"] = q["woba_n"]; q["wcon"] = q["wcon_n"]
    bt.DEN["wcon"] = "contact"
    print("\nREGRESSION CONSTANTS (pooled; PA for rates, balls in play for wOBAcon)")
    rel = []
    for s, delta in (("wcon", 0.5), ("hrr", 0.5), ("babip", 0.5), ("k", 0.75), ("bbr", 0.5), ("hbpr", 0.5)):
        d = bt.transitions(q, s)
        rel.append({"stat": s, "n": len(d), "y/y corr": np.corrcoef(d["x_t"] - d["lg_t"], d["x_next"] - d["lg_next"])[0, 1], "n0": bt.fit_n0(d, delta),
                    "RMSE league mean": bt.rmse(d, d["lg_next"]), "RMSE regressed": bt.rmse(*[bt.loto(d, delta)[0]] * 1, bt.loto(d, delta)[0]["pred"]) if False else np.nan})
    # RMSE regressed via loto
    for r in rel:
        d = bt.transitions(q, r["stat"]); delta = {"k": .75}.get(r["stat"], .5)
        E, _ = bt.loto(d, delta); r["RMSE regressed"] = bt.rmse(E, E["pred"]); r["RMSE last-year raw"] = bt.rmse(d, d["x_t"])
    print(pd.DataFrame(rel).round(4).to_string(index=False))

    # ---- LOTO predictions of every piece, then rebuild wOBA ----------------------------------------------------------
    P = {}
    for s, delta in (("woba", 0.75), ("wcon", 0.5), ("hrr", 0.5), ("babip", 0.5), ("k", 0.75), ("bbr", 0.5), ("hbpr", 0.5)):
        E, _ = loto_pred(q, s, delta)
        P[s] = E
    base = P["woba"][["x_next", "n_next", "lg_next", "w", "pred"]].rename(columns={"pred": "direct"})
    # league constants for the identity (target-year league values, as for the other pieces)
    lgs = {s: bt.league(q, s) for s in ("k", "bbr", "hbpr")}
    lg_sh = (q["sh"].sum() / q["pa"].sum())
    J = base.copy()
    for s in ("wcon", "k", "bbr", "hbpr", "hrr", "babip"):
        J = J.join(P[s][["pred"]].rename(columns={"pred": f"{s}_pred"}), how="inner")
    J = J.dropna().reset_index()
    ubb_over_bb = (q["bb"] - q["ibb"]).sum() / q["bb"].sum()
    J["ubb_pred"] = J["bbr_pred"] * ubb_over_bb
    J["ibb_r"] = (q["ibb"].sum() / q["pa"].sum()); J["sh_r"] = lg_sh
    J["den_r"] = 1 - J["ibb_r"] - J["sh_r"]
    J["con_r"] = (1 - J["bbr_pred"] - J["hbpr_pred"] - J["sh_r"] - J["k_pred"]).clip(lower=0.2)
    J["identity"] = (w["BB"] * J["ubb_pred"] + w["HBP"] * J["hbpr_pred"] + J["con_r"] * J["wcon_pred"]) / J["den_r"]
    # shipped component model (fitted weights, deviation form), fit LOTO
    comp = ["k", "bbr", "hrr", "babip"]
    out = []
    for t in sorted(J["t"].unique()):
        tr, te = J[J["t"] != t], J[J["t"] == t]
        lgn = {s: bt.league(q, s) for s in comp}
        def X(df):
            return np.column_stack([df[f"{s}_pred"] - df["t"].map(lambda tt, s=s: lgn[s][tt + 1]) for s in comp])
        wts = tr["w"].to_numpy()
        yv = (tr["x_next"] - tr["lg_next"]).to_numpy()
        beta = np.linalg.lstsq(X(tr) * np.sqrt(wts)[:, None], yv * np.sqrt(wts), rcond=None)[0]
        out.append(te.assign(comp_model=te["lg_next"] + X(te) @ beta))
    J = pd.concat(out)
    J["shipped_blend"] = 0.5 * (J["direct"] + J["comp_model"])
    J["id_direct"] = 0.5 * (J["identity"] + J["direct"])
    J["id_blend"] = 0.5 * (J["identity"] + J["shipped_blend"])
    J["three"] = (J["identity"] + J["direct"] + J["comp_model"]) / 3
    print(f"\nWHOLE-wOBA ACCURACY, leave-one-transition-out, n={len(J)} hitter-transitions (next-year PA >= {bt.MIN_NEXT})")
    rows = []
    for lab in ("direct", "comp_model", "shipped_blend", "identity", "id_direct", "id_blend", "three"):
        r = {"model": lab, "RMSE": bt.rmse(J, J[lab])}
        for t in sorted(J["t"].unique()):
            g = J[J["t"] == t]; r[f"t={t}"] = bt.rmse(g, g[lab])
        rows.append(r)
    rows.append({"model": "league mean", "RMSE": bt.rmse(J, J["lg_next"])})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    # paired bootstrap: identity vs shipped blend
    rng = np.random.default_rng(0); wts = J["w"].to_numpy()
    e1 = (J["x_next"] - J["shipped_blend"]).to_numpy() ** 2; e2 = (J["x_next"] - J["identity"]).to_numpy() ** 2; e3 = (J["x_next"] - J["id_blend"]).to_numpy() ** 2
    for lab, e in (("identity", e2), ("identity+shipped blend", e3)):
        diffs = []
        for _ in range(2000):
            i = rng.integers(0, len(J), len(J)); diffs.append(np.sqrt(np.average(e1[i], weights=wts[i])) - np.sqrt(np.average(e[i], weights=wts[i])))
        print(f"   RMSE gain of {lab} over the shipped blend: {np.mean(diffs):+.5f}  90% CI [{np.percentile(diffs,5):+.5f}, {np.percentile(diffs,95):+.5f}]")

    # ---- xwOBAcon, 2025 -> 2026 ---------------------------------------------------------------------------------------
    xc = read_sql_df(text("""select batter as player, year, count(*) n, avg(xwoba) xwobacon from pitches
                             where game_type='reg_season' and year in (2025, 2026) and exit_speed is not null and xwoba is not null
                               and play_result in ('single','double','triple','home_run','out','fielders_choice','error') group by 1, 2"""))
    xc["year"] = xc["year"].astype(int); import fl_innings_projection as ip
    xc["name_clean"] = xc["player"].apply(ip.lastfirst_to_clean)
    xc = xc.groupby(["name_clean", "year"]).apply(lambda g: pd.Series({"n_x": g["n"].sum(), "xwobacon": np.average(g["xwobacon"], weights=g["n"])})).reset_index()
    a = q[q["year"] == 2025].set_index("name_clean"); b = q[q["year"] == 2026].set_index("name_clean")
    j = a[["wcon", "contact", "woba", "pa"]].join(b[["wcon", "contact", "woba", "pa"]], lsuffix="_25", rsuffix="_26", how="inner")
    j = j.join(xc[xc["year"] == 2025].set_index("name_clean")[["xwobacon", "n_x"]].rename(columns={"xwobacon": "xcon_25"}), how="left")
    j = j[(j["contact_25"] >= 50) & (j["contact_26"] >= 80)].dropna(subset=["xcon_25", "wcon_25", "wcon_26"])
    lg25 = np.average(j["wcon_25"], weights=j["contact_25"]); lg26 = np.average(j["wcon_26"], weights=j["contact_26"]); lgx = np.average(j["xcon_25"], weights=j["contact_25"])
    print(f"\nxwOBAcon, 2025 -> 2026 (n={len(j)} hitters; >= 50 balls in play in 2025, >= 80 in 2026)")
    print(f"   corr: wOBAcon 25->26 {j['wcon_25'].corr(j['wcon_26']):.3f}; xwOBAcon 25 -> wOBAcon 26 {j['xcon_25'].corr(j['wcon_26']):.3f}; xwOBAcon 25 -> 26 {j['xcon_25'].corr(j['xcon_25']):.0f} (n/a)")
    y = j["wcon_26"].to_numpy()
    _ref = lg26 + ((j["contact_25"] * j["wcon_25"] + 200 * lg25) / (j["contact_25"] + 200) - lg25).to_numpy()
    ww, _a, _b = ew.weights_from_base(y, _ref, j["contact_26"].to_numpy())
    print(f"   variance law for next-year wOBAcon: a={_a:.5f}, b={_b:.3f} (n0_eval = {_b/max(_a,1e-12):.0f})")
    rm = lambda pr: float(np.sqrt(np.average((y - pr) ** 2, weights=ww)))
    res = [{"predictor of 2026 wOBAcon": "league mean", "RMSE": rm(np.full(len(y), lg26))}, {"predictor of 2026 wOBAcon": "last year raw", "RMSE": rm(j["wcon_25"].to_numpy() + lg26 - lg25)}]
    for n0 in (100, 200, 400, 800):
        res.append({"predictor of 2026 wOBAcon": f"wOBAcon shrunk n0={n0}", "RMSE": rm(lg26 + ((j["contact_25"] * j["wcon_25"] + n0 * lg25) / (j["contact_25"] + n0) - lg25).to_numpy())})
    for n0 in (100, 200, 400):
        res.append({"predictor of 2026 wOBAcon": f"xwOBAcon shrunk n0={n0}", "RMSE": rm(lg26 + ((j["contact_25"] * j["xcon_25"] + n0 * lgx) / (j["contact_25"] + n0) - lgx).to_numpy())})
    from sklearn.linear_model import LinearRegression
    for lab, cols in (("fit: wOBAcon + xwOBAcon", ["wcon_25", "xcon_25"]), ("fit: xwOBAcon", ["xcon_25"])):
        Xm = j[cols].to_numpy(); pr = np.zeros(len(y))
        for rep in range(20):
            for tri, tei in KFold(10, shuffle=True, random_state=rep).split(Xm):
                m = LinearRegression().fit(Xm[tri], y[tri], sample_weight=ww[tri]); pr[tei] += m.predict(Xm[tei]) / 20
        res.append({"predictor of 2026 wOBAcon": lab + " (10-fold)", "RMSE": rm(pr)})
    print(pd.DataFrame(res).round(4).to_string(index=False))
    return J, q, xc, w


def fit_comp(tr, cols, lgd):
    wts = tr["w"].to_numpy()
    X = np.column_stack([tr[f"{s}_pred"] - tr["t"].map(lambda tt, s=s: lgd[s][tt + 1]) for s in cols])
    yv = (tr["x_next"] - tr["lg_next"]).to_numpy()
    return np.linalg.lstsq(X * np.sqrt(wts)[:, None], yv * np.sqrt(wts), rcond=None)[0]


def comp_pred(df, cols, beta, lgd):
    X = np.column_stack([df[f"{s}_pred"] - df["t"].map(lambda tt, s=s: lgd[s][tt + 1]) for s in cols])
    return df["lg_next"].to_numpy() + X @ beta


def main2(J, q, xc, w):
    print("\n" + "=" * 100)
    print("HYBRID: fitted component weights with wOBAcon in place of HR/PA + BABIP (leave-one-transition-out)")
    lgd = {s: bt.league(q, s) for s in ("k", "bbr", "hbpr", "hrr", "babip", "wcon")}
    sets = {"shipped: K, BB, HR, BABIP": ["k", "bbr", "hrr", "babip"], "K, BB, wOBAcon": ["k", "bbr", "wcon"], "K, BB, HBP, wOBAcon": ["k", "bbr", "hbpr", "wcon"],
            "K, BB, HR, wOBAcon": ["k", "bbr", "hrr", "wcon"], "BB, wOBAcon": ["bbr", "wcon"]}
    rows = []
    for lab, cols in sets.items():
        pr = pd.Series(index=J.index, dtype=float)
        for t in sorted(J["t"].unique()):
            tr, te = J[J["t"] != t], J[J["t"] == t]
            pr.loc[te.index] = comp_pred(te, cols, fit_comp(tr, cols, lgd), lgd)
        J[f"c_{lab}"] = pr
        b = fit_comp(J, cols, lgd)
        r = {"component set": lab, "RMSE (comp only)": bt.rmse(J, pr), "RMSE 50/50 with direct": bt.rmse(J, 0.5 * (pr + J["direct"])), "weights (all data)": ", ".join(f"{s} {v:+.2f}" for s, v in zip(cols, b))}
        rows.append(r)
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print("\n2025 -> 2026 with Trackman: does xwOBAcon (instead of whole-ball xwOBA) improve the final wOBA?")
    T = J[J["t"] == 2025].copy()
    x25 = xc[xc["year"] == 2025].set_index("name_clean")["xwobacon"].rename("xcon25")
    xall = q[q["year"] == 2025].set_index("name_clean")[["xwoba", "contact", "pa"]].rename(columns={"xwoba": "xw25", "contact": "con25", "pa": "pa25"})
    T = T.join(x25, on="name_clean").join(xall, on="name_clean").dropna(subset=["xcon25", "xw25"])
    T = T[T["pa25"] >= 50]
    lgxc = np.average(T["xcon25"], weights=T["con25"]); lgxw = np.average(T["xw25"], weights=T["pa25"])
    yv = T["x_next"].to_numpy(); wt = T["w"].to_numpy()
    rm = lambda pr: float(np.sqrt(np.average((yv - pr) ** 2, weights=wt)))

    def cvg(base, z):
        pr = np.zeros(len(yv)); gs = []
        for rep in range(20):
            for tri, tei in KFold(10, shuffle=True, random_state=rep).split(z):
                g = np.sum(wt[tri] * z[tri] * (yv[tri] - base[tri])) / np.sum(wt[tri] * z[tri] ** 2); gs.append(g); pr[tei] += (base[tei] + g * z[tei]) / 20
        return pr, float(np.mean(gs))
    xw_s = ((T["pa25"] * T["xw25"] + 200 * lgxw) / (T["pa25"] + 200)).to_numpy()
    xc_s = ((T["con25"] * T["xcon25"] + 150 * lgxc) / (T["con25"] + 150)).to_numpy()
    shipped = T["shipped_blend"].to_numpy(); hyb = T["c_K, BB, wOBAcon"].to_numpy(); hyb4 = T["c_K, BB, HBP, wOBAcon"].to_numpy()
    res = [{"model": "direct", "RMSE": rm(T["direct"].to_numpy()), "gamma": np.nan}, {"model": "shipped blend (no xwOBA)", "RMSE": rm(shipped), "gamma": np.nan}]
    pr, g = cvg(shipped, xw_s - shipped); res.append({"model": "shipped blend + gamma(xwOBA - blend)  [as shipped]", "RMSE": rm(pr), "gamma": g})
    res.append({"model": "hybrid K,BB,wOBAcon (no x)", "RMSE": rm(hyb), "gamma": np.nan})
    res.append({"model": "avg(hybrid, direct) (no x)", "RMSE": rm(0.5 * (hyb + T["direct"].to_numpy())), "gamma": np.nan})
    pr, g = cvg(hyb, xw_s - hyb); res.append({"model": "hybrid + gamma(xwOBA - hybrid)", "RMSE": rm(pr), "gamma": g})
    b2 = 0.5 * (hyb + T["direct"].to_numpy())
    pr, g = cvg(b2, xw_s - b2); res.append({"model": "avg(hybrid, direct) + gamma(xwOBA - .)", "RMSE": rm(pr), "gamma": g})
    # xwOBAcon routed THROUGH the contact component: replace the regressed wOBAcon by regressed + g(xwOBAcon_s - regressed), then rebuild with the fitted hybrid weights
    cols = ["k", "bbr", "wcon"]; beta = fit_comp(J, cols, lgd)
    wc = T["wcon_pred"].to_numpy(); zc = xc_s - wc
    yc = None
    # g for the contact component chosen by 10-fold on the FINAL wOBA error
    for g in (0.0, 0.25, 0.5, 0.75, 1.0):
        T2 = T.copy(); T2["wcon_pred"] = wc + g * zc
        pr = comp_pred(T2, cols, beta, lgd)
        res.append({"model": f"hybrid with xwOBAcon routed through contact, g={g}", "RMSE": rm(pr), "gamma": g})
    print(f"   n = {len(T)} hitters; league wOBA RMSE reference {rm(T['lg_next'].to_numpy()):.4f}")
    print(pd.DataFrame(res).round(4).to_string(index=False))


if __name__ == "__main__":
    out = main()
    main2(*out)
