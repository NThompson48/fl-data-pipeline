"""
fl_unified_bridge.py -- ONE other-league bridge (College / Pioneer / affiliated
MiLB -> Frontier League K% and BB%), replacing the three league-specific bridges
(fl_college_bridge / fl_pioneer_bridge / fl_milb_bridge) in production.

Why (DECISION_LOG.md "Other-league bridge: recency and league audit", "Unified vs
by-league bridge: head-to-head"): on 513 FL-debut pairs the shipped bridges used
different recency rules per league, no shrinkage of thin samples, and a mapping that
ignored how stale the source season was. Shrinking by batters faced was the dominant
lever; league identity mattered little once rates were treated consistently.

Policy (identical for all three leagues):
  * Aggregate a pitcher's source seasons with TBF x decay^lag weights
    (K%: decay 0.25 -- close to latest-season; BB%: decay 1.0 -- flat), then SHRINK
    toward that SOURCE LEAGUE's mean rate by N0 = 50 batters faced.
  * Damp the K% deviation by RHO_K = 0.7 per extra year of staleness
    (gap = projection year - latest source season; BB% undamped, RHO_BB = 1.0).
    u = (shrunk rate - league mean) x damping.
  * One pooled mapping to FL:   K%  = league intercept + slope x u
                                BB% = intercept + slope x u
    (K% keeps league intercepts -- the best pooled K% structure; BB% a single line.)
  * Several leagues for one pitcher: take the source with the highest reliability
    sumW/(sumW + N0) x rho^(gap-1) per metric (no league-R2 weighting).
  * MiLB matches by MLBAM id when the FL pitcher_id is one, else by name; College by
    name (FanGraphs upid used to split same-name people); Pioneer by name.

Validation (leave-one-out, 513 debut pairs; out-of-year too): K% error cut vs the
league-mean guess 3.9% -> 7.8%, BB% 11.2% -> 20.2%; parameters picked on one debut
year still beat the shipped bridge on the other (K% 1.1 -> 4.6 / 3.9 -> 4.4,
BB% 4.1 -> 21.6 / 16.8 -> 19.5).

Usage:
    import fl_unified_bridge as ub
    model = ub.fit(raw)
    feat  = model.match(pitcher_ids, pitcher_names, current_year=2026)   # roster
    tbls  = model.prospect_tables(current_year=2026)                      # lookup pages
    fl_k, fl_bb = model.translate("milb", k_pct=.30, bb_pct=.08, tbf_latest=300, tbf_multi=600, gap=1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import fl_pitcher_features as fp
import fl_halfseason_backtest as bt
import fl_college_bridge as cb
import fl_pioneer_bridge as pb
import fl_milb_bridge as mb

N0 = 50.0
DECAY = {"k": 0.25, "bb": 1.0}
RHO = {"k": 0.7, "bb": 1.0}
LEAGUES = ["college", "pioneer", "milb"]
LEAGUE_LABEL = {"college": "College", "pioneer": "Pioneer League", "milb": "Affiliated MiLB"}
DEBUT_YEARS = [(2025, bt.START_2025, bt.END_2025), (2026, bt.START_2026, bt.END_2026)]


# --------------------------------------------------------------------------- #
#  source data -> common schema
# --------------------------------------------------------------------------- #
def source_tables() -> dict[str, pd.DataFrame]:
    cy = cb.load_all_college_years()
    college = pd.DataFrame({"kid": cy["upid"].astype(str), "name": cy["name"], "name_clean": cy["name_clean"],
                            "season": cy["season"].astype(int), "tbf": cy["tbf"].astype(float),
                            "k": cy["k_pct"], "bb": cy["bb_pct"]})
    py = pb.load_all_pioneer_years()
    pioneer = pd.DataFrame({"kid": py["name_clean"], "name": py["player"], "name_clean": py["name_clean"],
                            "season": py["year"].astype(int), "tbf": py["tbf"].astype(float), "k": py["k_pct"], "bb": py["bb_pct"]})
    my = mb.load_all_milb_years()
    milb = pd.DataFrame({"kid": my["pitcher"].astype(str), "name": my["player"], "name_clean": my["name_clean"],
                         "season": my["year"].astype(int), "tbf": my["tbf"].astype(float), "k": my["k_pct"], "bb": my["bb_pct"]})
    out = {"college": college, "pioneer": pioneer, "milb": milb}
    for lg, d in out.items():
        out[lg] = d.dropna(subset=["k", "bb", "tbf"]).query("tbf > 0").reset_index(drop=True)
    return out


def _weights(tbf, lag, decay):
    return tbf * (np.where(lag == 0, 1.0, decay ** lag) if decay > 0 else (lag == 0).astype(float))


def hist_features(g: pd.DataFrame, lg: dict, projection_year: int) -> dict:
    """Shrunk, decayed, staleness-damped inputs for one person's source seasons."""
    season = g["season"].to_numpy(); tbf = g["tbf"].to_numpy(float)
    latest = int(season.max()); lag = latest - season
    gap = max(projection_year - latest, 1)
    out = {"latest_season": latest, "gap": gap, "n_years": int(len(g)), "tbf_multi": float(tbf.sum()),
           "tbf_latest": float(tbf[season == latest].sum())}
    for m in ("k", "bb"):
        w = _weights(tbf, lag, DECAY[m])
        x = (np.sum(w * g[m].to_numpy(float)) + N0 * lg[m]) / (np.sum(w) + N0)
        damp = RHO[m] ** (gap - 1)
        out[f"x_{m}"] = float(x); out[f"u_{m}"] = float((x - lg[m]) * damp)
        out[f"rel_{m}"] = float(np.sum(w) / (np.sum(w) + N0) * damp)
    out["k_raw_latest"] = float(np.average(g.loc[g["season"] == latest, "k"], weights=g.loc[g["season"] == latest, "tbf"]))
    out["bb_raw_multi"] = float(np.average(g["bb"], weights=g["tbf"]))
    return out


# --------------------------------------------------------------------------- #
class UnifiedBridge:
    def __init__(self, srcs: dict[str, pd.DataFrame]):
        self.srcs = srcs
        self.lgm = {lg: {"k": float(np.average(s["k"], weights=s["tbf"])), "bb": float(np.average(s["bb"], weights=s["tbf"]))}
                    for lg, s in srcs.items()}
        self.by_name = {lg: {n: g for n, g in s.groupby("name_clean")} for lg, s in srcs.items()}
        self.by_kid_milb = {k: g for k, g in srcs["milb"].groupby("kid")}
        self.k_intercepts: dict[str, float] = {}
        self.k_slope = np.nan
        self.bb_intercept = self.bb_slope = np.nan
        self.fit_summary: dict = {}

    # -- matching -------------------------------------------------------------
    def history(self, lg: str, pitcher_id, name_clean: str, current_year: int):
        g = None
        if lg == "milb" and pitcher_id is not None and str(pitcher_id) in self.by_kid_milb:
            g = self.by_kid_milb[str(pitcher_id)]
        if g is None:
            g = self.by_name[lg].get(name_clean)
        if g is None:
            return None
        g = g[g["season"] <= current_year]
        if g.empty:
            return None
        if g["kid"].nunique() > 1:    # same-name collision -> the person whose history is most recent (ties: total TBF)
            lat = g.groupby("kid").agg(ls=("season", "max"), tb=("tbf", "sum")).sort_values(["ls", "tb"], ascending=False)
            g = g[g["kid"] == lat.index[0]]
        return g

    def map_k(self, lg: str, u: float) -> float:
        return self.k_intercepts[lg] + self.k_slope * u

    def map_bb(self, u: float) -> float:
        return self.bb_intercept + self.bb_slope * u

    def features_for(self, pitcher_id, pitcher_name: str, current_year: int) -> dict:
        nc = cb._fl_name_clean(pitcher_name)
        best = {"k": (-1.0, None, None), "bb": (-1.0, None, None)}
        for lg in LEAGUES:
            g = self.history(lg, pitcher_id, nc, current_year)
            if g is None:
                continue
            f = hist_features(g, self.lgm[lg], current_year + 1)
            for m in ("k", "bb"):
                if f[f"rel_{m}"] > best[m][0]:
                    best[m] = (f[f"rel_{m}"], lg, f)
        return best

    def match(self, pitcher_ids, pitcher_names, current_year: int) -> pd.DataFrame:
        """Same columns as fl_bridge_matching.recency_weighted_match (+ TBF behind each pick)."""
        rows = []
        for pid, nm in zip(pd.Series(pitcher_ids).astype(str).values, pd.Series(pitcher_names).values):
            best = self.features_for(pid, nm, current_year)
            r = {"pitcher_id": pid, "pitcher": nm}
            kb, bbb = best["k"], best["bb"]
            r["fl_k_rate_bridge"] = self.map_k(kb[1], kb[2]["u_k"]) if kb[1] else np.nan
            r["fl_bb_rate_bridge"] = self.map_bb(bbb[2]["u_bb"]) if bbb[1] else np.nan
            r["k_bridge_source"] = kb[1] or "none"; r["bb_bridge_source"] = bbb[1] or "none"
            r["k_bridge_tbf"] = kb[2]["tbf_latest"] if kb[1] else 0.0
            r["bb_bridge_tbf"] = bbb[2]["tbf_multi"] if bbb[1] else 0.0
            rows.append(r)
        out = pd.DataFrame(rows)
        out["has_college_match"] = out["fl_k_rate_bridge"].notna() | out["fl_bb_rate_bridge"].notna()
        return out

    # -- fitting ----------------------------------------------------------------
    def build_pairs(self, raw: pd.DataFrame) -> pd.DataFrame:
        seen, rows = set(), []
        names = raw[["pitcher_id", "pitcher"]].drop_duplicates("pitcher_id")
        for fl_year, s, e in DEBUT_YEARS:
            k = fp.build_period_kbb(fp.filter_period(raw, s, e), min_pa=1)
            k = k[~k["pitcher_id"].isin(seen)]; seen |= set(k["pitcher_id"])
            k = k.merge(names, on="pitcher_id", how="left")
            k["name_clean"] = k["pitcher"].apply(cb._fl_name_clean)
            k = k.sort_values("PA", ascending=False).drop_duplicates("name_clean", keep="first")
            for _, r in k.iterrows():
                for lg in LEAGUES:
                    g = self.history(lg, str(r["pitcher_id"]), r["name_clean"], fl_year - 1)
                    if g is None:
                        continue
                    f = hist_features(g, self.lgm[lg], fl_year)
                    rows.append({"league": lg, "pitcher_id": str(r["pitcher_id"]), "fl_year": fl_year, "pa": float(r["PA"]),
                                 "k_rate": r["K_rate"], "bb_rate": r["BB_rate"], **f})
        return pd.DataFrame(rows)

    def fit(self, raw: pd.DataFrame, verbose: bool = True):
        pairs = self.build_pairs(raw)
        self.pairs = pairs
        w = pairs["pa"].to_numpy(float)
        I = np.column_stack([(pairs["league"] == lg).astype(float) for lg in LEAGUES])

        def wls(X, y):
            XtWX = X.T @ (X * w[:, None]); beta = np.linalg.solve(XtWX, X.T @ (w * y))
            e = y - X @ beta; h = w * np.einsum("ij,jk,ik->i", X, np.linalg.inv(XtWX), X)
            loo = y - e / (1 - h); base = (np.sum(w * y) - w * y) / (np.sum(w) - w)
            return beta, 1 - np.sum(w * (y - loo) ** 2) / np.sum(w * (y - base) ** 2)

        bk, r2k = wls(np.column_stack([I, pairs["u_k"].to_numpy()]), pairs["k_rate"].to_numpy())
        self.k_intercepts = {lg: float(bk[i]) for i, lg in enumerate(LEAGUES)}; self.k_slope = float(bk[3])
        bb, r2b = wls(np.column_stack([np.ones(len(pairs)), pairs["u_bb"].to_numpy()]), pairs["bb_rate"].to_numpy())
        self.bb_intercept, self.bb_slope = float(bb[0]), float(bb[1])
        self.fit_summary = {"n_pairs": int(len(pairs)), "pairs_by_league": pairs["league"].value_counts().to_dict(),
                            "loo_r2_k": float(r2k), "loo_r2_bb": float(r2b)}
        if verbose:
            print(f"[fl_unified_bridge] {len(pairs)} debut pairs {self.fit_summary['pairs_by_league']}; "
                  f"K%: intercepts {{{', '.join(f'{k} {v:.4f}' for k, v in self.k_intercepts.items())}}}, slope {self.k_slope:.3f} "
                  f"(LOO R2_w {r2k:.3f});  BB%: {self.bb_intercept:.4f} + {self.bb_slope:.3f} u (LOO R2_w {r2b:.3f})")
        return self

    # -- calculator / lookup helpers ----------------------------------------------
    def translate(self, lg: str, k_pct: float, bb_pct: float, tbf_latest: float, tbf_multi: float, gap: int = 1):
        """Same arithmetic as the roster path, for typed-in numbers (K% = latest season, BB% = multi-year)."""
        lgm = self.lgm[lg]
        xk = (tbf_latest * k_pct + N0 * lgm["k"]) / (tbf_latest + N0)
        xb = (tbf_multi * bb_pct + N0 * lgm["bb"]) / (tbf_multi + N0)
        uk = (xk - lgm["k"]) * RHO["k"] ** (max(gap, 1) - 1)
        ub_ = (xb - lgm["bb"]) * RHO["bb"] ** (max(gap, 1) - 1)
        return self.map_k(lg, uk), self.map_bb(ub_)

    def prospect_tables(self, current_year: int) -> dict[str, pd.DataFrame]:
        """One row per person per source league (the person with the most recent history when a name repeats),
        projected as of the NEXT season: gap = current_year + 1 - latest season."""
        out = {}
        for lg in LEAGUES:
            s = self.srcs[lg]; s = s[s["season"] <= current_year]
            rows = []
            for kid, g in s.groupby("kid"):
                f = hist_features(g, self.lgm[lg], current_year + 1)
                rows.append({"kid": kid, "name": g["name"].iloc[-1], "name_clean": g["name_clean"].iloc[0], **f})
            d = pd.DataFrame(rows)
            d = d.sort_values(["latest_season", "tbf_multi"]).groupby("name_clean").tail(1).reset_index(drop=True)
            d["fl_k_rate_bridge"] = [self.map_k(lg, u) for u in d["u_k"]]
            d["fl_bb_rate_bridge"] = [self.map_bb(u) for u in d["u_bb"]]
            d["fl_kbb_rate_bridge"] = d["fl_k_rate_bridge"] - d["fl_bb_rate_bridge"]
            d = d.rename(columns={"k_raw_latest": "k_pct", "bb_raw_multi": "bb_pct_multi"})
            out[lg] = d[["name", "name_clean", "latest_season", "gap", "n_years", "tbf_latest", "tbf_multi", "k_pct", "bb_pct_multi",
                         "rel_k", "rel_bb", "fl_k_rate_bridge", "fl_bb_rate_bridge", "fl_kbb_rate_bridge"]] \
                .sort_values("fl_kbb_rate_bridge", ascending=False).reset_index(drop=True)
        return out

    def coef_dict(self, current_year: int) -> dict:
        return {"n0": N0, "decay_k": DECAY["k"], "decay_bb": DECAY["bb"], "rho_k": RHO["k"], "rho_bb": RHO["bb"],
                "league_means": self.lgm, "k_intercepts": self.k_intercepts, "k_slope": self.k_slope,
                "bb_intercept": self.bb_intercept, "bb_slope": self.bb_slope, "projection_year": current_year + 1,
                "n_pairs": self.fit_summary.get("n_pairs"), "loo_r2_k": self.fit_summary.get("loo_r2_k"),
                "loo_r2_bb": self.fit_summary.get("loo_r2_bb")}


def fit(raw: pd.DataFrame, verbose: bool = True) -> UnifiedBridge:
    return UnifiedBridge(source_tables()).fit(raw, verbose)


if __name__ == "__main__":
    raw = fp.load_fl_pitches_cached()
    m = fit(raw)
    t = m.prospect_tables(2026)
    for lg, d in t.items():
        print(f"\n{lg}: {len(d)} people; top 5 by projected FL K-BB%:")
        print(d.head(5).round(3).to_string(index=False))
