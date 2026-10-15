"""
run_all.py -- rebuild every production output and publish it to the dashboard, in dependency order, stopping at the first failure.  (user 2026-10-13: one-command rebuild before deployment)

  python run_all.py                 full rebuild (panel from Supabase, pitchers, park surfaces, bridges, hitters, rest-of-season), publish, then the dashboard tests
  python run_all.py --fast          skip the slow steps that rarely change (park surfaces with bootstrap, rest-of-season snapshots)
  python run_all.py --no-publish    build only (outputs stay in output/)
  python run_all.py --from STEP     start at step STEP (1-based), e.g. after fixing a failure

Order and why:
  1 hitter panel           Supabase -> output/fl_hitter_panel.parquet (every hitter step reads it)
  2 position-player flags  pitchers who are really position players (excluded by the pitcher pipeline)
  3 pitcher K% / BB%       frozen model + other-league prospect tables (fl_production_projection)
  4 pitcher RA9 / WAR      fl_ra9_trustmark_projection (reads step 3)
  5 park: wOBA surface     slow (100 bootstrap refits); --fast skips
  6 park: HR surface       --fast skips
  7 hitter projection      reads the panel, park surfaces and the K% bridge coefficients from step 8's previous run (bridge coefficients depend only on MiLB data)
  8 hitter MiLB bridge     coefficients + prospect table (reads league constants from step 7)
  9 hitter Pioneer bridge  provisional (MiLB mapping)
 10 college bridge        FanGraphs college exports in input/fangraphs_college/ (fl_hitter_bridge_college.py --build)
 11 rest-of-season         2026 as-of snapshots; --fast skips (not a priority until 2027 in-season)
 then: copy the published files (MANIFEST) to ../projections-dashboard/data and run its tests (pytest).
Logs: output/run_logs/<timestamp>/<NN>_<step>.log
"""
import datetime as dt
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DASH = os.path.join(HERE, "..", "projections-dashboard")
STEPS = [("hitter panel", "fl_hitter_panel.py", False), ("position-player flags", "fl_position_player_flag.py", False), ("pitcher K% and BB%", "fl_production_projection.py", False),
         ("pitcher RA9 and WAR", "fl_ra9_trustmark_projection.py", False), ("park: wOBA surface", "fl_hitter_park_surface_final.py", True), ("park: HR surface", "fl_hitter_park_hr_surface.py", True),
         ("hitter projection", "fl_hitter_projection.py", False), ("hitter MiLB bridge", "fl_hitter_bridge.py", False), ("hitter Pioneer bridge", "fl_hitter_bridge_pioneer.py", False), ("hitter college bridge", "fl_hitter_bridge_college.py --build", False),
         ("rest-of-season snapshots", "fl_ros_build.py", True)]
MANIFEST = ["fl_2027_hitter_projections.parquet", "fl_hitter_history.parquet", "fl_hitter_model_coefs.json", "fl_2027_projections.parquet", "fl_ra9_trustmark_projection.parquet",
            "fl_projection_model_coefs.json", "fl_ra9_model_coefs.json", "position_player_pitchers.parquet", "fl_hitter_bridge_coefs.json", "fl_hitter_milb_prospects.parquet", "fl_hitter_milb_history.parquet",
            "fl_hitter_pioneer_prospects.parquet", "fl_hitter_pioneer_history.parquet", "fl_hitter_college_prospects.parquet", "fl_hitter_college_history.parquet", "ros_hitters_2026.parquet", "ros_pitchers_2026.parquet",
            "fl_college_prospect_lookup.parquet", "fl_college_prospect_projections.parquet", "fl_pioneer_prospect_lookup.parquet", "fl_pioneer_prospect_projections.parquet",
            "fl_milb_prospect_lookup.parquet", "fl_milb_prospect_projections.parquet"]


def main(argv):
    fast, publish = "--fast" in argv, "--no-publish" not in argv
    start = int(argv[argv.index("--from") + 1]) if "--from" in argv else 1
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S"); logdir = os.path.join(HERE, "output", "run_logs", stamp); os.makedirs(logdir, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for i, (name, script, slow) in enumerate(STEPS, 1):
        if i < start or (fast and slow):
            print(f"[{i:2d}] {name:28s} skipped"); continue
        t0 = dt.datetime.now(); log = os.path.join(logdir, f"{i:02d}_{os.path.splitext(script.split()[0])[0]}.log")
        with open(log, "w", encoding="utf-8") as fh:
            r = subprocess.run([sys.executable, *script.split()], cwd=HERE, stdout=fh, stderr=subprocess.STDOUT, env=env)
        secs = (dt.datetime.now() - t0).total_seconds()
        print(f"[{i:2d}] {name:28s} {'OK' if r.returncode == 0 else 'FAILED'}  ({secs:.0f}s)  {log}")
        if r.returncode != 0:
            print(f"Stopped. Fix the failure, then resume with: python run_all.py --from {i}" + (" --fast" if fast else "")); return 1
    if not publish:
        print("built (not published)"); return 0
    missing = [f for f in MANIFEST if not os.path.exists(os.path.join(HERE, "output", f))]
    if missing:
        print("missing outputs, not published:", missing); return 1
    for f in MANIFEST:
        shutil.copy(os.path.join(HERE, "output", f), os.path.join(DASH, "data", f))
    print(f"published {len(MANIFEST)} files to {os.path.normpath(os.path.join(DASH, 'data'))}")
    r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"], cwd=DASH, env=env)
    print("dashboard tests:", "PASSED" if r.returncode == 0 else "FAILED"); return r.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
