#!/usr/bin/env python3
"""build_table1_numbers.py - descriptive numbers for Table 1 of the journal paper (journal/getfast-heat-journal.tex).

Prints the counts and distributions that journal/numbers.tex quotes for the cohorts. Estimates nothing; writes
nothing (stdout only). Read-only on every input.

Inputs:
  /weather/data/activities_model_clean_v3.parquet   training-run corpus (one row per run)
  /weather/data/athlinks_persons.parquet            linked official finishes (one row per finish)
  /weather/data/athlinks_edition_effects_v2.parquet + edition_weather_official_v2.parquet   analysis editions (v2)
  /weather/data/fade/hrrr_extension_20261028/build_efforts_v3.json   race-distance effort set (counts only)

Run: /home/bb/test_env/bin/python3 journal/build_table1_numbers.py
"""
import json
import numpy as np, pandas as pd

D = "/weather/data"

# ---- training-run corpus: the filter as coded in the analysis (paper_open_items_20261109/scripts/
#      corpus_count_check.py): pace, WBGT and every baseline feature present (mean HR, 30-day fitness pace, distance,
#      climb per km, climb), mean HR strictly between 60 and 210 bpm, pace strictly between 2.5 and 11.5 min/km, run
#      WBGT at or below 28 C. Canonical count 403,719 runs, 2,420 runners (numbers.json corpus_runs).
cols = ["user_id", "start_dt", "distance_km", "duration_s", "pace_min_km", "avg_hr", "wbgt", "cc", "fit_pace_30d",
        "elev_gain_per_km", "elev_gain_m"]
d = pd.read_parquet(f"{D}/activities_model_clean_v3.parquet", columns=cols)
d = d.dropna(subset=["pace_min_km", "wbgt", "avg_hr", "fit_pace_30d", "distance_km", "elev_gain_per_km", "elev_gain_m"])
m = (d.avg_hr > 60) & (d.avg_hr < 210) & (d.pace_min_km > 2.5) & (d.pace_min_km < 11.5) & (d.wbgt <= 28)
x = d[m]
runs_per = x.groupby("user_id").size()
print("== training-run corpus")
print(f"   runs {len(x):,}; runners {x.user_id.nunique():,}; runs per runner median {runs_per.median():.0f} "
      f"(IQR {runs_per.quantile(.25):.0f}-{runs_per.quantile(.75):.0f})")
print(f"   dates {x.start_dt.min():%Y-%m-%d} to {x.start_dt.max():%Y-%m-%d}; run-location countries {x.cc.nunique()}")
print("   top countries (% of runs):", (100 * x.cc.value_counts(normalize=True).head(5)).round(1).to_dict())
q = lambda s: f"{s.median():.1f} ({s.quantile(.25):.1f}-{s.quantile(.75):.1f})"
print(f"   distance km {q(x.distance_km)}; duration min {q(x.duration_s / 60)}; pace min/km {q(x.pace_min_km)}; "
      f"mean HR bpm {q(x.avg_hr)}")
print(f"   run WBGT C median {x.wbgt.median():.1f}, p10 {x.wbgt.quantile(.1):.1f}, p90 {x.wbgt.quantile(.9):.1f}; "
      f">= 18 C {100 * (x.wbgt >= 18).mean():.1f} %; >= 22 C {100 * (x.wbgt >= 22).mean():.1f} %; "
      f">= 25 C {100 * (x.wbgt >= 25).mean():.1f} %")

# ---- official results: analysis editions of the v2 specification (identified first stage, v2 weather table) -------
E = pd.read_parquet(f"{D}/athlinks_edition_effects_v2.parquet")[["course", "year", "n_pairs", "n_repeaters"]]
W = pd.read_parquet(f"{D}/edition_weather_official_v2.parquet")
W["course"] = np.where(W.band == "half", W.race + " [half]", W.race)
W["deviation"] = W.wbgt_mean - W.climate60_wbgt
A = E.merge(W, on=["course", "year"], how="inner")
A = A[~((A.course == "London") & (A.year == 2020)) & (A.n_pairs >= 30)].dropna(subset=["wbgt_mean", "deviation"])
print("== official results, analysis editions (v2)")
for band, g in A.groupby("band"):
    print(f"   {band}: editions {len(g)}, races {g.course.nunique()}, years {g.year.min()}-{g.year.max()}, "
          f"race-window WBGT median {g.wbgt_mean.median():.1f} (range {g.wbgt_mean.min():.1f} to {g.wbgt_mean.max():.1f}), "
          f"median finish h median {g.median_finish_h.median():.2f}, repeat pairs {int(g.n_pairs.sum()):,}")
# finishes behind the analysis editions, from the linked persons table (one row per finish)
P = pd.read_parquet(f"{D}/athlinks_persons.parquet", columns=["person_id", "course", "band", "year"])
print(f"   persons table: finishes {len(P):,}; persons {P.person_id.nunique():,}; "
      f"persons with 2+ finishes {(P.groupby('person_id').size() >= 2).sum():,}; courses {P.course.nunique()}")
for band, g in A.groupby("band"):                              # both tables label halves "<race> [half]"
    sub = P[P.band.eq(band)].merge(g[["course", "year"]], on=["course", "year"])
    print(f"   {band}: finishes in analysis editions {len(sub):,}; persons {sub.person_id.nunique():,}")

# ---- race-distance effort set v3 ------------------------------------------------------------------------------------
B = json.load(open(f"{D}/fade/hrrr_extension_20261028/build_efforts_v3.json"))
print("== race-distance efforts (v3):", B)
