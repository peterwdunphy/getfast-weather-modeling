#!/usr/bin/env python3
"""derived_numbers.py - numbers in journal/numbers.tex that are arithmetic on stored estimates rather than read off a
JSON field directly. Prints each with the inputs it uses. Read-only; writes nothing (stdout only).

  (1) Table 2: % of time and minutes slower than at WBGT 10 C implied by a scale s on the learned curve, for the
      training-run curve (s = 1), long training runs (the > 150 min within-runner scale), the half marathon (2:00
      finish) and the marathon (4:00 finish). Log-unit convention of the identified first stage: % = 100 x
      (exp(s x c_log(W) / 100) - 1), c_log = 100 x log(1 + c(W)/100). Intervals carry the scale's interval only.
  (2) Hometown climate: 95% intervals from the stored coefficient and SE (est +/- 1.96 SE), and an illustration.
  (3) Seconds per kilometre equivalents used in the Discussion.

Run: /home/bb/test_env/bin/python3 journal/derived_numbers.py
"""
import json
import numpy as np

D = "/weather/data"
a = np.load(f"{D}/heat_percentile_averaged_v3.npz", allow_pickle=True)
w = a["wsweep"].astype(float); pc = a["pctl_curves"].astype(float)[49]; pop = pc - np.interp(10.0, w, pc)
c_pct = lambda x: float(np.interp(x, w, pop, left=0.0, right=pop[-1]))
c_log = lambda x: 100 * np.log1p(c_pct(x) / 100)
pct = lambda s, x: 100 * np.expm1(s * c_log(x) / 100)

# ---- (1) Table 2: one row per condition; cells "% (min)" at WBGT 16-22 C ------------------------------------------------
# scale sources: training curve (1), R3 long-run band, short races (RunSignup, course-clustered CI,
# short_race_alpha_20261109/alpha_table.csv), half and marathon (official, identified first stage)
import csv
SRT = {r["group"]: r for r in csv.DictReader(open("/weather/data/fade/short_race_alpha_20261109/alpha_table.csv"))}
R3 = json.load(open(f"{D}/effort_heat_tests_round2.json"))["R3"]
long_run = R3["training_scale_by_expected_duration"]["training_without_race_distance_runs"][4]["scale"]
REF = json.load(open(f"{D}/edition_tests_all_official_v2_fs2.json"))
sr = lambda g: (float(SRT[g]["alpha"]), float(SRT[g]["ci_cl_lo"]), float(SRT[g]["ci_cl_hi"]))
ROWS = [("TabTr", "Training runs, all", (1.0, None, None), None),
        ("TabLong", "Training runs over 150 min", (long_run["coef"], *long_run["ci"]), None),
        ("TabFiveK", "5K", sr("5k"), 30.0),
        ("TabTenK", "10K", sr("10k"), 60.0),
        ("TabFifteen", "15K and 10 miles", sr("15k+10mi"), 90.0),
        ("TabHalf", "Half marathon", tuple(REF["half"]["alpha"]), 120.0),
        ("TabMar", "Marathon", tuple(REF["marathon"]["alpha"]), 240.0)]
print("% Table 2 rows (derived_numbers.py part 1): scale [95% CI] & then % (min) at WBGT 16, 18, 20, 22 C")
for mac, lab, (s_, lo, hi), ref in ROWS:
    head = "1" if lo is None else f"{s_:.2f} [{lo:.2f}, {hi:.2f}]"
    cells = []
    for W_ in (16, 18, 20, 22):
        p = pct(s_, W_)
        cells.append(f"{p:.1f}" if ref is None else f"{p:.1f} ({p * ref / 100:.1f})")
    print(f"\\newcommand{{\\{mac}}}{{{head} & {' & '.join(cells)}}}   % {lab}" + ("" if ref is None else f", reference finish {ref:.0f} min"))

# ---- (2) hometown climate ------------------------------------------------------------------------------------------
H = json.load(open(f"{D}/hometown_climate_test_official_v2.json"))
print("\n% hometown climate (derived_numbers.py part 2): est [est - 1.96 SE, est + 1.96 SE]")
for grp in ["all editions", "spring editions", "autumn editions", "half: all editions"]:
    g = H[grp]
    for term in ["cxpre", "cxyear"]:
        e, se = g[term]["coef_pct_per_curvepct_per_C"], g[term]["se"]
        print(f"   {grp:20s} {term:7s} {e:+.3f} [{e - 1.96 * se:+.3f}, {e + 1.96 * se:+.3f}] p={g[term]['p']:.1e}")
    e, se = g["cxpre_alone"]["coef"], g["cxpre_alone"]["se"]
    print(f"   {grp:20s} recent alone {e:+.3f} [{e - 1.96 * se:+.3f}, {e + 1.96 * se:+.3f}] p={g['cxpre_alone']['p']:.1e}; "
          f"n {g['n']:,} finishes, {g['persons']:,} persons, {g['editions']} editions")
# illustration: an edition at WBGT 22 C (curve value in %), recent home climate 5 C warmer than the course's
cv = c_pct(22.0)
for lab, grp in [("marathon", "all editions"), ("half", "half: all editions")]:
    e = H[grp]["cxpre_alone"]["coef"]
    print(f"   illustration {lab}, recent alone, WBGT 22 (c = {cv:.2f} %), +5 C home: {e * cv * 5:+.2f} % of finish time"
          f" = {e * cv * 5 * (240 if lab == 'marathon' else 120) / 100:+.1f} min on a {'4' if lab == 'marathon' else '2'}-h finish")

# ---- (3) long training run at WBGT 20 and 22 ----------------------------------------------------------------------
print("\n% long training runs (> 150 min) at the within-runner scale:",
      {x: round(pct(long_run["coef"], x), 2) for x in (20, 22)})
