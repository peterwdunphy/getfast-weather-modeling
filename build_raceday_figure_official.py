#!/usr/bin/env python3
"""build_raceday_figure_official.py - Figure: race-day scale of the learned heat curve, marathon and half marathon,
from the OFFICIAL-data rebuild of the race-day weather (official date, first-mass start, start line, window from the
first mass start to that start plus the edition's median finish time).

Supersedes build_raceday_figure.py for the manuscript (that script and figures/fig_raceday_scale.pdf are kept).

Inputs (same as /weather/scripts/edition_tests_all_official.py, whose numbers this script must reproduce):
  /weather/data/athlinks_edition_effects.parquet   within-person edition effects (% of finish time, per course/year,
                                                   person-clustered SEs); halves carry " [half]" in the course label
  /weather/data/edition_weather_official.parquet   official race-window weather, include_weather editions only
  /weather/data/heat_percentile_averaged_v3.npz    the learned population curve (training-run scale)
  /weather/data/edition_tests_all_official.json    the reference estimates (checked against, not read for plotting)
Model per band (as in the text): effect_e = course_c + year_y + alpha * c_DL(WBGT_e), WLS with weights 1/SE^2,
editions with fewer than 30 repeat pairs dropped.
Each panel shows the edition effect net of the course and year fixed effects against the curve value, with the
fitted alpha line. Titles state what is plotted and the numbers only.
Writes figures/fig_raceday_scale_official.pdf and prints every number the text quotes from this model.
"""
import json, os, numpy as np, pandas as pd, statsmodels.api as sm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = "/weather/data"

# --- the learned curve: median-percentile sweep, zeroed at 10 C, held flat beyond the sweep (same as the test script)
a = np.load(f"{D}/heat_percentile_averaged_v3.npz", allow_pickle=True)
w = a["wsweep"].astype(float); pc = a["pctl_curves"].astype(float)[49]; pop = pc - np.interp(10.0, w, pc)
curve = lambda x: np.interp(x, w, pop, left=0.0, right=pop[-1])

# --- edition table: identical merge and filters to edition_tests_all_official.py
E = pd.read_parquet(f"{D}/athlinks_edition_effects.parquet")
W = pd.read_parquet(f"{D}/edition_weather_official.parquet")
W["deviation"] = W.wbgt_mean - W.climate60_wbgt
W["course"] = np.where(W.band == "half", W.race + " [half]", W.race)     # edition effects label halves this way
E = E.merge(W, on=["course", "year"], how="inner")
E = E[~((E.course == "London") & (E.year == 2020))]                      # virtual mass field (already not in W)
E = E[E.n_pairs >= 30].dropna(subset=["wbgt_mean", "deviation"]).reset_index(drop=True)
E["c"] = curve(E.wbgt_mean.values)

REF = json.load(open(f"{D}/edition_tests_all_official.json"))            # reference values to reproduce
MAJORS = ["New York City", "Chicago", "Boston", "London", "Big Sur"]
MK = {"New York City": "o", "Chicago": "s", "Boston": "^", "London": "D", "Big Sur": "v"}
NAME = {"marathon": "Marathon", "half": "Half marathon"}

fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.0), sharey=True)
for ax, band in zip(axes, ["marathon", "half"]):
    d = E[E.band == band].reset_index(drop=True)
    wts = 1.0 / d["se_%"] ** 2                                           # precision weights
    X = pd.get_dummies(d.course).astype(float).join(pd.get_dummies(d.year, prefix="y", drop_first=True).astype(float))
    X["c"] = d.c.values
    m = sm.WLS(d["edition_effect_%"].values, X, weights=wts).fit()
    alpha, (lo, hi) = m.params["c"], m.conf_int().loc["c"]
    # guard: the figure must show the same model the text quotes
    assert len(d) == REF[band]["editions"] and abs(alpha - REF[band]["alpha"][0]) < 1e-6, (band, len(d), alpha)
    print(f"{band}: editions {len(d)}, courses {d.course.nunique()}, years {d.year.min()}-{d.year.max()}, "
          f"alpha {alpha:.2f} [{lo:.2f}, {hi:.2f}], R2 {m.rsquared:.2f}, WBGT range {d.wbgt_mean.min():.1f}..{d.wbgt_mean.max():.1f}")
    # calendar-year effects (relative to the first year in the band), printed for the text
    yfx = m.params[[k for k in m.params.index if k.startswith("y_")]]
    print("   year effects (% vs first year):", {k[2:]: round(v, 2) for k, v in yfx.items()})
    # partial residual: the edition effect minus everything in the model except the curve term
    d["net"] = d["edition_effect_%"].values - X.drop(columns="c").values @ m.params.drop("c").values
    print("   hottest editions:"); print(d.nlargest(5, "wbgt_mean")[["course", "year", "wbgt_mean", "c", "edition_effect_%", "net"]].round(2).to_string(index=False))

    oth = d[~d.course.isin(MAJORS)]
    ax.errorbar(oth.c, oth.net, yerr=1.96 * oth["se_%"], fmt="o", ms=3, lw=0.4, color="0.45", alpha=0.55, capsize=0,
                label=f"editions (n = {len(oth)})" if band == "marathon" else f"editions (n = {len(d)})")
    if band == "marathon":                                               # the five-course subset reported in the text
        for c in MAJORS:
            g = d[d.course == c]
            ax.errorbar(g.c, g.net, yerr=1.96 * g["se_%"], fmt=MK[c], ms=5, lw=0.8, capsize=0, label=f"{c} (n = {len(g)})")
    xs = np.linspace(0, d.c.max() * 1.05, 50)
    ax.plot(xs, alpha * xs, "k-", lw=1.3, label=rf"fitted $\alpha$ = {alpha:.2f} [{lo:.2f}, {hi:.2f}]")
    ax.plot(xs, xs, "k:", lw=1, label=r"$\alpha$ = 1 (training-run scale)")
    # secondary axis: the WBGT that maps to each curve value (curve is monotone above 10 C)
    sec = ax.secondary_xaxis("top", functions=(lambda c: np.interp(c, pop, w), lambda t: curve(t)))
    sec.set_xticks([10, 16, 18, 20, 22, 24]); sec.set_xlabel("race-window mean WBGT (°C)", fontsize=9)
    ax.set_xlabel("learned curve at the edition's WBGT, $c(\\bar{W})$\n(% of pace, training-run scale)", fontsize=9)
    ax.set_title(f"{NAME[band]}: n = {len(d)} editions, {d.course.nunique()} courses, {d.year.min()}–{d.year.max()}\n"
                 f"$\\alpha$ = {alpha:.2f}, 95% CI [{lo:.2f}, {hi:.2f}], $R^2$ = {m.rsquared:.2f}\n"
                 "WLS (weights 1/SE²), course + calendar-year fixed effects", fontsize=8.5)
    ax.axhline(0, color="grey", lw=0.5); ax.grid(alpha=0.3); ax.legend(fontsize=7.5, loc="lower right")
axes[0].set_ylabel("edition effect net of course and year effects\n(% of finish time, within-person)")
fig.tight_layout(); out = os.path.join(HERE, "figures", "fig_raceday_scale_official.pdf"); fig.savefig(out); print("wrote", out)
print("curve at WBGT 16/18/20/21/22/24/28:", {t: round(float(curve(t)), 2) for t in [16, 18, 20, 21, 22, 24, 28]})

# ----------------------------------------------------------------------------------------------------------------
# Numbers quoted in the text that are not in a JSON: printed here so each one traces to this script and its inputs.
# ----------------------------------------------------------------------------------------------------------------
# (a) measurement sensitivity: old inputs (list date, city geocode, fixed window) vs official inputs, for the
#     editions that enter the regressions. Flags are those of compare_edition_weather_official.py:
#     date_changed = official race day differs from the list date; loc_changed = start line > 5 km from the old
#     geocode; start_changed = first mass start > 30 min from the old window start; d_wbgt = official - old mean WBGT.
C = pd.read_csv(f"{D}/edition_weather_official_changes.csv")
C = C.merge(E[["course", "year"]], on=["course", "year"])               # restrict to the analysis editions
print(f"\nmeasurement changes, analysis editions n = {len(C)} (of 656 in the changes table)")
for k in ["date_changed", "loc_changed", "start_changed", "any_changed"]:
    g = C[C[k]]; print(f"   {k:14s} n = {len(g):3d}; mean |d_wbgt| {g.d_wbgt.abs().mean():.2f} C, median |d_wbgt| {g.d_wbgt.abs().median():.2f} C")
print(f"   all            mean |d_wbgt| {C.d_wbgt.abs().mean():.2f} C, median {C.d_wbgt.abs().median():.2f} C, mean signed {C.d_wbgt.mean():+.2f} C; "
      f"|d_wbgt| > 2 C in {(C.d_wbgt.abs() > 2).sum()} editions")
# (b) finishes behind the analysis editions (linked finish rows in those course-years)
P = pd.read_parquet(f"{D}/athlinks_persons.parquet", columns=["course", "year", "band"])
print("finish rows in the persons table:", len(P), "| in the analysis editions:", P.merge(E[["course", "year"]], on=["course", "year"]).groupby("band").size().to_dict())
print("courses with edition effects:", pd.read_parquet(f"{D}/athlinks_edition_effects.parquet").course.nunique())
