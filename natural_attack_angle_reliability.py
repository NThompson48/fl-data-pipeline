"""
Natural Attack Angle — Reliability Analysis & Tuned Function
=============================================================
Schema: exit_speed, angle, batter_id, batter, year  (FL/TrackMan column names)

Sections
--------
  1. Corrected compute_natural_attack_angle()   (schema fix + bug fixes)
  2. split_half_reliability()                   (odd/even split, Spearman-Brown corrected)
  3. reliability_curve()                        (reliability vs BBE threshold)
  4. method_audit()                             (how often each fallback fires)
  5. stability_zscore()                         (league-normalised signal distribution)
  6. run_full_analysis()                        (runs & plots everything)

Drop-in usage
-------------
    from natural_attack_angle_reliability import run_full_analysis
    run_full_analysis(df)          # df must contain exit_speed, angle, batter_id, year
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.neighbors import NearestNeighbors

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CORRECTED  compute_natural_attack_angle()
# ─────────────────────────────────────────────────────────────────────────────

def compute_natural_attack_angle(
    df,
    bins=np.linspace(-30, 60, 13),   # 7.5° bins
    top_k=6,
    K=5,
    curvature_threshold=0.0005,
    clamp_range=(-5, 40),
    max_domain=(-10, 45),
    min_bbe=10,                       # raised from 6; below this the envelope is meaningless
    min_envelope_pts=4,               # minimum distinct filled bins required
):
    """
    Natural Attack Angle — schema-corrected, bug-fixed version.

    Key fixes vs. original
    ──────────────────────
    • Column names: exit_speed / angle / batter_id  (FL/TrackMan schema)
    • k_dynamic scope bug: was leaked from the inner loop into the outer
      envelope-size check.  Now captured as `last_k_dynamic` and compared
      against `min_envelope_pts` (a stable parameter) instead.
    • min_bbe raised to 10 — below 6 the envelope had only 1-2 filled bins,
      making the quadratic fit degenerate.
    • Adaptive bin count: for batters with 10–30 BBE, coarser bins (9 instead
      of 13) are used so more bins have enough mass.
    • EV-weighted fallback now clips angle to clamp_range so it can't itself
      produce a clamped result.
    • method column included on every row for auditability.
    """

    results = []

    for (batter_id, batter_name, year), g in df.groupby(
        ["batter_id", "batter", "year"]
    ):
        g = g.dropna(subset=["angle", "exit_speed"])
        bbe = len(g)

        def _append(naa, ev, method):
            results.append({
                "batter_id": batter_id,
                "batter":    batter_name,
                "year":      year,
                "Natural_Attack_Angle": naa,
                "Natural_Max_EV":       ev,
                "BBE":                  bbe,
                "method":               method,
            })

        if bbe < min_bbe:
            _append(np.nan, np.nan, "insufficient_data")
            continue

        g = g.copy()

        # ── Adaptive binning: coarser for thin samples ──────────────────────
        if bbe < 30:
            active_bins = np.linspace(-30, 60, 9)   # 11.25° bins
        else:
            active_bins = bins

        g["angle_bin"] = pd.cut(g["angle"], active_bins, include_lowest=True)

        # ── Top-k envelope ──────────────────────────────────────────────────
        envelope_rows = []
        for _, group in g.groupby("angle_bin", observed=True):
            if len(group) == 0:
                continue
            k_dyn = max(1, min(top_k, int(np.ceil(0.65 * len(group)))))
            top_grp = group.nlargest(k_dyn, "exit_speed")
            envelope_rows.append({
                "angle":      top_grp["angle"].mean(),
                "exit_speed": top_grp["exit_speed"].mean(),
            })

        env = pd.DataFrame(envelope_rows)

        if len(env) < min_envelope_pts:          # ← fixed: stable threshold
            _append(np.nan, np.nan, "insufficient_envelope")
            continue

        x = env["angle"].to_numpy(dtype=float).reshape(-1, 1)
        y = env["exit_speed"].to_numpy(dtype=float)

        # ── KNN weights ────────────────────────────────────────────────────
        nbrs = NearestNeighbors(
            n_neighbors=min(K, len(x)), algorithm="ball_tree"
        ).fit(x)
        distances, _ = nbrs.kneighbors(x)
        mean_dist = distances.mean(axis=1)
        weights = 1.0 / (mean_dist + 1e-6)
        weights /= weights.sum()

        # ── Weighted quadratic fit ─────────────────────────────────────────
        coeffs = np.polyfit(x.flatten(), y, deg=2, w=weights)
        poly   = np.poly1d(coeffs)
        a, b, _ = coeffs

        # ── EV-weighted fallback angle (clipped to valid range) ────────────
        ev_weighted_angle = float(np.clip(
            np.average(g["angle"], weights=g["exit_speed"]),
            clamp_range[0], clamp_range[1]
        ))

        # ── Tier 1: analytic maximum ───────────────────────────────────────
        analytic_max   = -b / (2 * a) if a != 0 else np.nan
        valid_quad     = (a < 0) and (abs(a) > curvature_threshold)
        plausible_dom  = max_domain[0] <= analytic_max <= max_domain[1]

        if valid_quad and plausible_dom:
            attack_angle = analytic_max
            max_ev       = poly(attack_angle)
            method       = "analytic_max"

        else:
            # Tier 2: frontier / EV-weighted
            best_idx      = env["exit_speed"].idxmax()
            frontier_angle = env.loc[best_idx, "angle"]
            envelope_flat  = (env["exit_speed"].max() - env["exit_speed"].min()) < 1.5

            if len(env) < 4 or envelope_flat:
                attack_angle = ev_weighted_angle
                max_ev       = g["exit_speed"].max()
                method       = "ev_weighted_fallback"
            else:
                attack_angle = frontier_angle
                max_ev       = poly(frontier_angle)
                method       = "frontier_fallback"

        # ── Final rule: avoid hard clamp ──────────────────────────────────
        if attack_angle <= clamp_range[0] or attack_angle >= clamp_range[1]:
            angle_radius   = 5.0
            max_angle_spread = 3.0
            max_ev_spread    = 2.0
            min_neighbors    = 3

            supported = []
            for _, row in env.iterrows():
                ang = row["angle"]
                ev  = row["exit_speed"]
                local = env[np.abs(env["angle"] - ang) <= angle_radius]
                if (
                    len(local) >= min_neighbors
                    and (local["angle"].max() - local["angle"].min()) <= max_angle_spread
                    and (local["exit_speed"].max() - local["exit_speed"].min()) <= max_ev_spread
                ):
                    supported.append((ang, ev))

            if supported:
                supported.sort(key=lambda t: t[1], reverse=True)
                attack_angle, max_ev = supported[0]
                method = "clamp_supported_empirical"
            else:
                attack_angle = ev_weighted_angle
                max_ev       = g["exit_speed"].max()
                method       = "clamp_ev_weighted"

        _append(attack_angle, max_ev, method)

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SPLIT-HALF RELIABILITY
# ─────────────────────────────────────────────────────────────────────────────

def split_half_reliability(df, n_splits=200, seed=42, min_bbe=10):
    """
    Odd/even random split-half reliability for Natural Attack Angle.

    For each batter-year with ≥ min_bbe BBE:
      • Randomly shuffle their BBE and split into two halves
      • Compute NAA on each half independently
      • Pearson-correlate the two halves across all batter-years
      • Spearman-Brown correct to full-sample reliability:
            r_full = 2r / (1 + r)

    Returns a DataFrame with one row per split: r_half, r_full, n_batters.
    """
    rng = np.random.default_rng(seed)
    records = []

    for split_i in range(n_splits):
        half1_rows, half2_rows = [], []

        for (bid, bname, yr), g in df.groupby(["batter_id", "batter", "year"]):
            g_clean = g.dropna(subset=["angle", "exit_speed"])
            n = len(g_clean)
            if n < min_bbe * 2:          # need enough for both halves
                continue
            idx = rng.permutation(n)
            mid = n // 2
            h1 = g_clean.iloc[idx[:mid]].copy()
            h2 = g_clean.iloc[idx[mid:]].copy()
            h1["batter_id"] = bid; h1["batter"] = bname; h1["year"] = yr
            h2["batter_id"] = bid; h2["batter"] = bname; h2["year"] = yr
            half1_rows.append(h1)
            half2_rows.append(h2)

        if not half1_rows:
            continue

        df1 = compute_natural_attack_angle(pd.concat(half1_rows, ignore_index=True))
        df2 = compute_natural_attack_angle(pd.concat(half2_rows, ignore_index=True))

        merged = df1.merge(
            df2,
            on=["batter_id", "year"],
            suffixes=("_h1", "_h2")
        ).dropna(subset=["Natural_Attack_Angle_h1", "Natural_Attack_Angle_h2"])

        if len(merged) < 10:
            continue

        r_half = np.corrcoef(
            merged["Natural_Attack_Angle_h1"],
            merged["Natural_Attack_Angle_h2"]
        )[0, 1]
        r_full = (2 * r_half) / (1 + r_half)   # Spearman-Brown

        records.append({
            "split":    split_i,
            "r_half":   r_half,
            "r_full":   r_full,
            "n_batters": len(merged),
        })

        if split_i % 50 == 0:
            print(f"  split {split_i}/{n_splits}  r_half={r_half:.3f}  "
                  f"r_full={r_full:.3f}  n={len(merged)}")

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  RELIABILITY CURVE  (reliability vs. BBE threshold)
# ─────────────────────────────────────────────────────────────────────────────

def reliability_curve(df, bbe_thresholds=None, n_splits=100, seed=0):
    """
    Run split-half at each BBE threshold in `bbe_thresholds` and return
    median Spearman-Brown r_full per threshold.

    This is the standard "reliability vs. sample size" diagnostic used
    throughout the project (same pattern as bat speed).
    """
    if bbe_thresholds is None:
        bbe_thresholds = [10, 15, 20, 25, 30, 40, 50, 75, 100]

    rows = []
    for thr in bbe_thresholds:
        # Restrict to batter-years with at least 2×thr BBE
        bbe_counts = (
            df.dropna(subset=["angle", "exit_speed"])
              .groupby(["batter_id", "year"])
              .size()
              .reset_index(name="bbe")
        )
        eligible = bbe_counts[bbe_counts["bbe"] >= thr * 2]
        sub = df.merge(eligible[["batter_id", "year"]], on=["batter_id", "year"])

        if len(sub) < 20:
            rows.append({"bbe_threshold": thr, "r_full": np.nan, "n_batters": 0})
            continue

        rel = split_half_reliability(sub, n_splits=n_splits, seed=seed, min_bbe=thr)
        med_r = rel["r_full"].median() if len(rel) else np.nan
        n_bat = rel["n_batters"].median() if len(rel) else 0

        print(f"  BBE≥{thr:3d}  →  r_full(median)={med_r:.3f}  n_batters≈{n_bat:.0f}")
        rows.append({"bbe_threshold": thr, "r_full": med_r, "n_batters": int(n_bat)})

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  METHOD AUDIT
# ─────────────────────────────────────────────────────────────────────────────

def method_audit(naa_df):
    """
    Summarise how often each computation method fires.
    High rates of frontier_fallback or ev_weighted suggest the quadratic
    fit is failing too often — either bins are too sparse or curvature
    threshold is too aggressive.
    """
    counts = (
        naa_df.groupby("method")
              .agg(n=("batter_id", "count"),
                   mean_bbe=("BBE", "mean"),
                   mean_naa=("Natural_Attack_Angle", "mean"))
              .reset_index()
              .sort_values("n", ascending=False)
    )
    counts["pct"] = (counts["n"] / counts["n"].sum() * 100).round(1)
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SIGNAL STABILITY  (z-score distribution & year-to-year if available)
# ─────────────────────────────────────────────────────────────────────────────

def stability_zscore(naa_df, min_bbe=20):
    """
    Z-score NAA within each year.  A well-behaved metric produces a
    near-normal distribution with σ ≈ 1.  Heavy tails or multi-modality
    suggest fallback instability.

    If multiple years are present, also reports year-over-year Pearson r
    for returning batters (≥ min_bbe in both years).
    """
    out = naa_df[naa_df["BBE"] >= min_bbe].copy()
    out["naa_z"] = out.groupby("year")["Natural_Attack_Angle"].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-9)
    )

    yoy = None
    years = sorted(out["year"].unique())
    if len(years) >= 2:
        pairs = []
        for i in range(len(years) - 1):
            y1, y2 = years[i], years[i + 1]
            a = out[out["year"] == y1][["batter_id", "naa_z"]].rename(
                columns={"naa_z": f"naa_z_{y1}"}
            )
            b = out[out["year"] == y2][["batter_id", "naa_z"]].rename(
                columns={"naa_z": f"naa_z_{y2}"}
            )
            m = a.merge(b, on="batter_id").dropna()
            if len(m) >= 5:
                r, p = stats.pearsonr(m[f"naa_z_{y1}"], m[f"naa_z_{y2}"])
                pairs.append({"years": f"{y1}→{y2}", "r": round(r, 3),
                              "p": round(p, 4), "n": len(m)})
        yoy = pd.DataFrame(pairs)

    return out, yoy


# ─────────────────────────────────────────────────────────────────────────────
# 6.  FULL ANALYSIS  (runs everything, produces one composite figure)
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "analytic_max":            "#2c7bb6",
    "frontier_fallback":       "#f46d43",
    "ev_weighted_fallback":    "#d73027",
    "clamp_supported_empirical": "#74add1",
    "clamp_ev_weighted":       "#fdae61",
    "insufficient_data":       "#cccccc",
    "insufficient_envelope":   "#aaaaaa",
}


def run_full_analysis(
    raw_df,
    min_bbe_reliable=20,
    n_splits_curve=80,
    output_path=None,
):
    """
    Parameters
    ----------
    raw_df          : DataFrame with columns  batter_id, batter, year,
                      exit_speed, angle  (FL/TrackMan schema)
    min_bbe_reliable: minimum BBE for a batter-year to appear in signal plots
    n_splits_curve  : number of split-half iterations per BBE threshold
    output_path     : if given, saves the figure to this path

    Steps
    -----
    1. Run compute_natural_attack_angle on the full dataset
    2. Method audit
    3. NAA distribution by year
    4. Reliability curve (BBE thresholds 10–75)
    5. YoY stability (if ≥2 years)
    """
    print("── Step 1: computing Natural Attack Angle ──────────────────────")
    naa_df = compute_natural_attack_angle(raw_df)
    print(f"   {len(naa_df):,} batter-years  |  "
          f"{naa_df['Natural_Attack_Angle'].notna().sum():,} with valid NAA")

    print("\n── Step 2: method audit ─────────────────────────────────────────")
    audit = method_audit(naa_df)
    print(audit.to_string(index=False))

    print("\n── Step 3: reliability curve ────────────────────────────────────")
    thresholds = [10, 15, 20, 25, 30, 40, 50, 75]
    rel_curve = reliability_curve(raw_df, bbe_thresholds=thresholds,
                                  n_splits=n_splits_curve)
    print(rel_curve.to_string(index=False))

    print("\n── Step 4: stability / YoY ──────────────────────────────────────")
    z_df, yoy = stability_zscore(naa_df, min_bbe=min_bbe_reliable)
    if yoy is not None:
        print(yoy.to_string(index=False))
    else:
        print("   (only one year present — skipping YoY)")

    # ── Figure ─────────────────────────────────────────────────────────────
    years = sorted(naa_df["year"].unique())
    n_year_panels = len(years)

    fig = plt.figure(figsize=(16, 10), facecolor="#0f1117")
    fig.suptitle("Natural Attack Angle — Reliability Analysis",
                 color="white", fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(
        2, 3 + n_year_panels,
        figure=fig,
        hspace=0.42, wspace=0.35,
        left=0.06, right=0.97, top=0.92, bottom=0.08
    )

    ax_style = dict(facecolor="#1a1d27", labelcolor="white",
                    titlecolor="white")

    def style_ax(ax, title="", xlabel="", ylabel=""):
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.set_title(title, color="white", fontsize=9, pad=6)
        ax.set_xlabel(xlabel, color="#aaa", fontsize=8)
        ax.set_ylabel(ylabel, color="#aaa", fontsize=8)

    # Panel A — method breakdown (bar)
    ax_m = fig.add_subplot(gs[0, 0])
    audit_valid = audit[~audit["method"].str.startswith("insufficient")]
    bar_colors = [COLORS.get(m, "#888") for m in audit_valid["method"]]
    ax_m.barh(audit_valid["method"], audit_valid["pct"],
              color=bar_colors, edgecolor="none")
    for i, (pct, n) in enumerate(zip(audit_valid["pct"], audit_valid["n"])):
        ax_m.text(pct + 0.3, i, f"{pct:.1f}%  (n={n})",
                  va="center", color="white", fontsize=7)
    style_ax(ax_m, "Method Distribution", "% of batter-years", "")
    ax_m.set_xlim(0, audit_valid["pct"].max() * 1.35)

    # Panel B — reliability curve
    ax_r = fig.add_subplot(gs[0, 1:3])
    valid_rc = rel_curve.dropna(subset=["r_full"])
    ax_r.plot(valid_rc["bbe_threshold"], valid_rc["r_full"],
              "o-", color="#2c7bb6", linewidth=2, markersize=6)
    ax_r.axhline(0.70, color="#f46d43", linestyle="--", linewidth=1,
                 label="r = 0.70 (decent)")
    ax_r.axhline(0.80, color="#74c476", linestyle="--", linewidth=1,
                 label="r = 0.80 (good)")
    for _, row in valid_rc.iterrows():
        ax_r.annotate(f"n≈{int(row['n_batters'])}",
                      (row["bbe_threshold"], row["r_full"]),
                      textcoords="offset points", xytext=(0, 8),
                      ha="center", color="#aaa", fontsize=7)
    style_ax(ax_r, "Split-Half Reliability (Spearman-Brown corrected)",
             "Minimum BBE", "r_full")
    ax_r.set_ylim(0, 1.05)
    ax_r.legend(fontsize=8, facecolor="#1a1d27", labelcolor="white",
                edgecolor="#444")

    # Panel C — NAA distribution by year
    for i, yr in enumerate(years):
        ax_d = fig.add_subplot(gs[0, 3 + i])
        yr_data = naa_df[(naa_df["year"] == yr) &
                         naa_df["Natural_Attack_Angle"].notna() &
                         (naa_df["BBE"] >= min_bbe_reliable)]
        ax_d.hist(yr_data["Natural_Attack_Angle"], bins=25,
                  color="#2c7bb6", alpha=0.8, edgecolor="none")
        med = yr_data["Natural_Attack_Angle"].median()
        ax_d.axvline(med, color="white", linestyle="--", linewidth=1.2,
                     label=f"median={med:.1f}°")
        style_ax(ax_d, f"{yr}  NAA distribution\n(BBE≥{min_bbe_reliable})",
                 "Attack Angle (°)", "Count")
        ax_d.legend(fontsize=7, facecolor="#1a1d27", labelcolor="white",
                    edgecolor="#444")

    # Panel D — NAA z-score distribution (all years combined)
    ax_z = fig.add_subplot(gs[1, 0])
    z_vals = z_df["naa_z"].dropna()
    ax_z.hist(z_vals, bins=30, color="#74add1", alpha=0.8, edgecolor="none",
              density=True)
    xr = np.linspace(-4, 4, 200)
    ax_z.plot(xr, stats.norm.pdf(xr), color="white", linewidth=1.2,
              linestyle="--", label="N(0,1)")
    style_ax(ax_z, "NAA z-score distribution\n(should ≈ N(0,1))",
             "z-score", "density")
    ax_z.legend(fontsize=7, facecolor="#1a1d27", labelcolor="white",
                edgecolor="#444")
    ax_z.set_xlim(-4.5, 4.5)

    # Panel E — BBE vs NAA scatter (coloured by method)
    ax_s = fig.add_subplot(gs[1, 1:3])
    valid_naa = naa_df[naa_df["Natural_Attack_Angle"].notna()]
    for method, grp in valid_naa.groupby("method"):
        ax_s.scatter(grp["BBE"], grp["Natural_Attack_Angle"],
                     s=18, alpha=0.5, label=method,
                     color=COLORS.get(method, "#888"), linewidths=0)
    ax_s.axhline(0,  color="#555", linewidth=0.8, linestyle=":")
    ax_s.axhline(15, color="#555", linewidth=0.8, linestyle=":")
    style_ax(ax_s, "NAA vs BBE — coloured by method",
             "BBE (balls in play)", "Natural Attack Angle (°)")
    ax_s.set_xlim(left=0)
    ax_s.set_ylim(-15, 50)
    ax_s.legend(fontsize=7, facecolor="#1a1d27", labelcolor="white",
                edgecolor="#444", markerscale=1.5, ncol=2)

    # Panel F — YoY scatter (if available)
    ax_y = fig.add_subplot(gs[1, 3:3 + n_year_panels])
    if yoy is not None and len(yoy):
        for _, pair_row in yoy.iterrows():
            y1_str, y2_str = pair_row["years"].split("→")
            y1, y2 = int(y1_str), int(y2_str)
            a = naa_df[(naa_df["year"] == y1) & (naa_df["BBE"] >= min_bbe_reliable)][
                ["batter_id", "Natural_Attack_Angle"]
            ].rename(columns={"Natural_Attack_Angle": "naa_y1"})
            b = naa_df[(naa_df["year"] == y2) & (naa_df["BBE"] >= min_bbe_reliable)][
                ["batter_id", "Natural_Attack_Angle"]
            ].rename(columns={"Natural_Attack_Angle": "naa_y2"})
            merged_yy = a.merge(b, on="batter_id").dropna()
            if len(merged_yy) >= 5:
                ax_y.scatter(merged_yy["naa_y1"], merged_yy["naa_y2"],
                             s=22, alpha=0.6, color="#74add1", linewidths=0)
                m, intercept, r, *_ = stats.linregress(
                    merged_yy["naa_y1"], merged_yy["naa_y2"]
                )
                xfit = np.linspace(merged_yy["naa_y1"].min(),
                                   merged_yy["naa_y1"].max(), 100)
                ax_y.plot(xfit, m * xfit + intercept,
                          color="white", linewidth=1.2)
                ax_y.set_title(
                    f"YoY  {pair_row['years']}  r={pair_row['r']:.3f}  "
                    f"n={pair_row['n']}",
                    color="white", fontsize=9, pad=6
                )
        style_ax(ax_y, "", f"NAA  {years[0]}  (°)", f"NAA  {years[-1]}  (°)")
    else:
        ax_y.text(0.5, 0.5, "Only one season —\nYoY not available",
                  ha="center", va="center", color="#888",
                  transform=ax_y.transAxes, fontsize=9)
        style_ax(ax_y, "Year-over-Year Stability", "", "")

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor="#0f1117")
        print(f"\nFigure saved → {output_path}")

    plt.show()

    return {
        "naa_df":     naa_df,
        "audit":      audit,
        "rel_curve":  rel_curve,
        "z_df":       z_df,
        "yoy":        yoy,
    }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK-START
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Replace this block with your actual data pull ─────────────────────
    # Example (Supabase / psycopg2 pattern from the project):
    #
    #   import sqlalchemy
    #   engine = sqlalchemy.create_engine(SUPABASE_URL)
    #   raw_df = pd.read_sql("""
    #       SELECT batter_id, batter, year,
    #              exit_speed, angle
    #       FROM pitches
    #       WHERE exit_speed IS NOT NULL
    #         AND angle IS NOT NULL
    #         AND year IN (2025, 2026)
    #   """, engine)
    #
    # Then:
    #   results = run_full_analysis(raw_df)
    #   results["naa_df"].to_csv("naa_output.csv", index=False)

    print("Import this module and call run_full_analysis(your_df).")
    print("See the docstring at the top for column requirements.")
