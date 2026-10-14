"""
validate_bearing.py
===================
Validates:
  1. Sign convention on bearing — does negative = pull for RHH?
  2. Whether 15° is the right pull/oppo cutoff

Run this before using bearing in any matchup logic.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────────────────────────────────────

ENV_PATH = r"C:/Users/215nt/PycharmProjects/FLStuff/FLPitching/FLCommand/.env"

def get_engine():
    load_dotenv(ENV_PATH)
    host     = os.getenv("SUPABASE_HOST", "aws-1-us-east-1.pooler.supabase.com")
    port     = os.getenv("SUPABASE_PORT", "6543")
    dbname   = os.getenv("SUPABASE_DB",   "postgres")
    user     = os.getenv("SUPABASE_USER")
    password = os.getenv("SUPABASE_PASSWORD")
    return create_engine(
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}?sslmode=require"
    )

# ─────────────────────────────────────────────────────────────────────────────
# PULL
# ─────────────────────────────────────────────────────────────────────────────

SQL = """
SELECT
    batter,
    batter_side,
    bearing,
    exit_speed,
    angle
FROM pitches
WHERE pitch_call  = 'in_play'
  AND exit_speed  IS NOT NULL
  AND angle       IS NOT NULL
  AND bearing     IS NOT NULL
  AND batter_side IS NOT NULL
"""

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_bearing(df: pd.DataFrame):

    rh = df[df["batter_side"].str.lower() == "right"]
    lh = df[df["batter_side"].str.lower() == "left"]

    print("── Summary stats by batter_side ────────────────────────────────")
    for label, sub in [("RHH", rh), ("LHH", lh)]:
        print(f"\n  {label}  (n={len(sub):,})")
        print(f"    bearing  mean={sub['bearing'].mean():.2f}°  "
              f"median={sub['bearing'].median():.2f}°  "
              f"std={sub['bearing'].std():.2f}°")
        print(f"    range    [{sub['bearing'].min():.1f},  {sub['bearing'].max():.1f}]")

    # ── Sign check ────────────────────────────────────────────────────────────
    # For TrackMan: if negative = pull for RHH, then:
    #   RHH pull side mean bearing should be < 0
    #   LHH pull side mean bearing should be > 0
    # (or vice versa — this is what we're checking)
    rhh_mean = rh["bearing"].mean()
    lhh_mean = lh["bearing"].mean()

    print("\n── Sign convention check ───────────────────────────────────────")
    print(f"  RHH mean bearing: {rhh_mean:+.2f}°")
    print(f"  LHH mean bearing: {lhh_mean:+.2f}°")

    if rhh_mean < 0 and lhh_mean > 0:
        print("\n  Convention: NEGATIVE = pull for RHH, POSITIVE = pull for LHH")
        print("  (TrackMan standard — negative toward 1B side)")
    elif rhh_mean > 0 and lhh_mean < 0:
        print("\n  Convention: POSITIVE = pull for RHH, NEGATIVE = pull for LHH")
        print("  (Flipped — may need sign inversion before using cutoffs)")
    else:
        print("\n  ⚠  Means are same sign — something unexpected. "
              "Check batter_side values and sample sizes.")

    # ── Percentile breakdown at candidate cutoffs ─────────────────────────────
    print("\n── Pull % by cutoff candidate (RHH: bearing < -X) ────────────")
    print(f"  {'Cutoff':>8}  {'RHH pull %':>12}  {'LHH pull %':>12}")
    for cutoff in [10, 12, 15, 17, 20]:
        # If negative = pull for RHH
        rhh_pull_pct = (rh["bearing"] < -cutoff).mean() * 100
        lhh_pull_pct = (lh["bearing"] >  cutoff).mean() * 100
        # Flip if convention is opposite
        if rhh_mean > 0:
            rhh_pull_pct = (rh["bearing"] >  cutoff).mean() * 100
            lhh_pull_pct = (lh["bearing"] < -cutoff).mean() * 100
        print(f"  {cutoff:>8}°  {rhh_pull_pct:>11.1f}%  {lhh_pull_pct:>11.1f}%")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9), facecolor="#0f1117")
    fig.suptitle("Bearing / Spray Angle Validation",
                 color="white", fontsize=13, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.4, wspace=0.32,
                           left=0.06, right=0.97, top=0.92, bottom=0.08)

    def style_ax(ax, title="", xlabel="", ylabel=""):
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.set_title(title, color="white", fontsize=9, pad=6)
        ax.set_xlabel(xlabel, color="#aaa", fontsize=8)
        ax.set_ylabel(ylabel, color="#aaa", fontsize=8)

    # Panel 1: Bearing distribution by side (overlaid)
    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.linspace(-60, 60, 60)
    ax1.hist(rh["bearing"], bins=bins, alpha=0.6, color="#2c7bb6",
             label="RHH", density=True)
    ax1.hist(lh["bearing"], bins=bins, alpha=0.6, color="#d7191c",
             label="LHH", density=True)
    for cutoff, ls in [(15, "--"), (20, ":")]:
        ax1.axvline(-cutoff, color="#aaa", linewidth=0.9, linestyle=ls)
        ax1.axvline( cutoff, color="#aaa", linewidth=0.9, linestyle=ls)
    ax1.axvline(0, color="white", linewidth=0.8, linestyle="-", alpha=0.4)
    style_ax(ax1, "Bearing distribution by handedness",
             "Bearing (°)", "Density")
    ax1.legend(fontsize=8, facecolor="#1a1d27", labelcolor="white",
               edgecolor="#444")

    # Panel 2: Bearing vs Exit Speed (RHH)
    ax2 = fig.add_subplot(gs[0, 1])
    sc = ax2.scatter(rh["bearing"], rh["exit_speed"],
                     s=4, alpha=0.3, c=rh["angle"],
                     cmap="RdYlGn", vmin=-10, vmax=40, linewidths=0)
    plt.colorbar(sc, ax=ax2, label="Launch Angle (°)").ax.yaxis.label.set_color("white")
    ax2.axvline(-15, color="#f46d43", linewidth=1.2, linestyle="--", label="±15°")
    ax2.axvline( 15, color="#f46d43", linewidth=1.2, linestyle="--")
    style_ax(ax2, "RHH: Bearing vs Exit Speed\n(coloured by launch angle)",
             "Bearing (°)", "Exit Speed (mph)")
    ax2.legend(fontsize=8, facecolor="#1a1d27", labelcolor="white",
               edgecolor="#444")

    # Panel 3: Bearing vs Exit Speed (LHH)
    ax3 = fig.add_subplot(gs[0, 2])
    sc3 = ax3.scatter(lh["bearing"], lh["exit_speed"],
                      s=4, alpha=0.3, c=lh["angle"],
                      cmap="RdYlGn", vmin=-10, vmax=40, linewidths=0)
    plt.colorbar(sc3, ax=ax3, label="Launch Angle (°)").ax.yaxis.label.set_color("white")
    ax3.axvline(-15, color="#f46d43", linewidth=1.2, linestyle="--", label="±15°")
    ax3.axvline( 15, color="#f46d43", linewidth=1.2, linestyle="--")
    style_ax(ax3, "LHH: Bearing vs Exit Speed\n(coloured by launch angle)",
             "Bearing (°)", "Exit Speed (mph)")
    ax3.legend(fontsize=8, facecolor="#1a1d27", labelcolor="white",
               edgecolor="#444")

    # Panel 4: Mean EV by bearing bin (RHH)
    ax4 = fig.add_subplot(gs[1, 0])
    bin_edges = np.linspace(-50, 50, 21)
    bin_mids  = (bin_edges[:-1] + bin_edges[1:]) / 2
    rh_binned = pd.cut(rh["bearing"], bin_edges)
    lh_binned = pd.cut(lh["bearing"], bin_edges)
    rh_ev = rh.groupby(rh_binned, observed=True)["exit_speed"].mean()
    lh_ev = lh.groupby(lh_binned, observed=True)["exit_speed"].mean()
    ax4.plot(bin_mids, rh_ev.values, color="#2c7bb6", label="RHH", linewidth=1.8)
    ax4.plot(bin_mids, lh_ev.values, color="#d7191c", label="LHH", linewidth=1.8)
    ax4.axvline(-15, color="#aaa", linewidth=0.9, linestyle="--")
    ax4.axvline( 15, color="#aaa", linewidth=0.9, linestyle="--")
    ax4.axvline(0, color="white", linewidth=0.8, linestyle="-", alpha=0.3)
    style_ax(ax4, "Mean EV by bearing bin\n(peak = pull side confirmation)",
             "Bearing (°)", "Mean Exit Speed (mph)")
    ax4.legend(fontsize=8, facecolor="#1a1d27", labelcolor="white",
               edgecolor="#444")

    # Panel 5: Bearing percentile CDF by side
    ax5 = fig.add_subplot(gs[1, 1])
    for label, sub, color in [("RHH", rh, "#2c7bb6"), ("LHH", lh, "#d7191c")]:
        sorted_b = np.sort(sub["bearing"])
        cdf = np.arange(1, len(sorted_b) + 1) / len(sorted_b)
        ax5.plot(sorted_b, cdf, color=color, label=label, linewidth=1.5)
    for cutoff in [15, 20]:
        ax5.axvline(-cutoff, color="#aaa", linewidth=0.8, linestyle=":")
        ax5.axvline( cutoff, color="#aaa", linewidth=0.8, linestyle=":")
    style_ax(ax5, "CDF of bearing by handedness\n(use to pick pull/oppo percentile thresholds)",
             "Bearing (°)", "Cumulative %")
    ax5.legend(fontsize=8, facecolor="#1a1d27", labelcolor="white",
               edgecolor="#444")

    # Panel 6: Hard-hit rate (EV >= 95) by bearing zone and side
    ax6 = fig.add_subplot(gs[1, 2])
    cutoff = 15
    for label, sub, color in [("RHH", rh, "#2c7bb6"), ("LHH", lh, "#d7191c")]:
        # Pull, Middle, Oppo — sign depends on convention, shown for both
        pull_sign = -1 if rhh_mean < 0 else 1
        if label == "LHH":
            pull_sign = -pull_sign
        zones = {
            "Pull":   sub["bearing"] * pull_sign >  cutoff,
            "Middle": sub["bearing"].abs()        <= cutoff,
            "Oppo":   sub["bearing"] * pull_sign < -cutoff,
        }
        hhr = {z: sub[mask]["exit_speed"].ge(95).mean() * 100
               for z, mask in zones.items()}
        xs = np.arange(3)
        offset = -0.2 if label == "RHH" else 0.2
        ax6.bar(xs + offset, hhr.values(), width=0.35,
                color=color, alpha=0.8, label=label, edgecolor="none")
    ax6.set_xticks([0, 1, 2])
    ax6.set_xticklabels(["Pull", "Middle", "Oppo"], color="white")
    style_ax(ax6, f"Hard-hit rate (EV≥95) by zone\n(cutoff=±{cutoff}°)",
             "", "Hard-Hit %")
    ax6.legend(fontsize=8, facecolor="#1a1d27", labelcolor="white",
               edgecolor="#444")

    plt.savefig(r"C:\Users\215nt\PycharmProjects\FLStuff\bearing_validation.png",
                dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.show()
    print("\nFigure saved → bearing_validation.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text(SQL), conn)
    print(f"Pulled {len(df):,} BBE rows\n")
    validate_bearing(df)
