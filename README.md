# FLHitting -- Frontier League projection pipeline

Builds the 2027 Frontier League projections (pitchers and hitters), rest-of-season snapshots and other-league (college / Pioneer / MiLB) translations, and publishes them to the
Streamlit dashboard in `../projections-dashboard`. Data files and model outputs are not in this repository (see `.gitignore`); they are pulled from the database and rebuilt.

## Run it
```
python run_all.py                 # full rebuild, publish to ../projections-dashboard/data, run the dashboard tests
python run_all.py --fast          # skip the slow park surfaces and rest-of-season snapshots
python run_all.py --from 3        # resume at step 3 (steps and what each does are listed at the top of run_all.py)
python run_all.py --skip-check    # bypass the data-freshness pre-flight (fl_data_freshness.py)
```
Needs a `secrets.toml` with the database connection (never committed) and Python 3.11 with the packages used by the scripts (pandas, numpy, scikit-learn, statsmodels, xgboost, torch).
On Windows, `import torch` must come before xgboost / scikit-learn in any script that uses both (see ROLL_FORWARD.md, section 3b).

## What is where
| | |
|---|---|
| `run_all.py` | the one-command rebuild, in dependency order |
| `fl_production_projection.py`, `fl_ra9_trustmark_projection.py` | pitcher K% / BB%, RA9 and WAR |
| `fl_pitcher_features.py` + `fl_wh_sva_*_worker.py` | pitch-level stuff measures (the network estimators run in separate worker processes) |
| `fl_hitter_projection.py`, `fl_hitter_bridge*.py` | hitter wOBA / K% / BB% / WAR and the other-league bridges |
| `fl_ros_*.py`, `fl_pitcher_ros*.py`, `fl_hitter_ros.py` | rest-of-season snapshots |
| `fl_data_freshness.py` | stops a rebuild when local data lag the database |
| `fl_*_test.py` | the validation tests behind each modelling decision |
| `DECISION_LOG.md` | every trade-off and its evidence, newest last |
| `ROLL_FORWARD.md` | what to change when a new season's data arrive |
| `prereg/` | the pre-registered 2027 checks (`fl_2027_preregistered_checks.py`) |

## Conventions
Report weighted RMSE and MAE (frozen inverse-variance weights from `fl_eval_weights.py`) with a pitcher bootstrap for every model change; regular-season games only; one fit per measure for stability tests.
