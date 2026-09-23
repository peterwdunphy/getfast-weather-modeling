#!/usr/bin/env python3
"""build_raceday_figure.py - Figure: race-day scale of the learned heat curve, from official results.

Inputs (produced in /weather/scripts, see athlinks_*.py):
  /weather/data/athlinks_edition_effects.parquet   within-person edition effects (% of finish time, per course/year,
                                                   with person-clustered SEs) from same-course repeat finishers
  /weather/data/edition_weather_features.parquet   race-window weather per edition (mean outdoor WBGT etc.)
  /weather/data/heat_percentile_averaged_v3.npz    the learned population curve (training-run scale)
Model (as in the text): effect_e = course_c + year_y + alpha * c_DL(WBGT_e), weighted by 1/SE^2, London 2020
(virtual mass race) excluded. The plot shows the effect net of course and year fixed effects against the
curve value, with the fitted alpha line; the caption states what is plotted and the numbers only.
Writes figures/fig_raceday_scale.pdf and prints the coefficients used in the text.
"""
import os, numpy as np, pandas as pd, statsmodels.api as sm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
a = np.load("/weather/data/heat_percentile_averaged_v3.npz", allow_pickle=True)
w = a["wsweep"].astype(float); pc = a["pctl_curves"].astype(float)[49]; pop = pc - np.interp(10.0, w, pc)
curve = lambda x: np.interp(x, w, pop, left=0.0, right=pop[-1])

E = pd.read_parquet("/weather/data/athlinks_edition_effects.parquet").merge(pd.read_parquet("/weather/data/edition_weather_features.parquet"), on=["course", "year"])
E = E[~((E.course == "London") & (E.year == 2020))].reset_index(drop=True)
E["c"] = curve(E.wbgt_mean.values); wts = 1.0 / E["se_%"] ** 2
X = pd.get_dummies(E.course).astype(float).join(pd.get_dummies(E.year, prefix="y", drop_first=True).astype(float)); X["c"] = E.c
m = sm.WLS(E["edition_effect_%"], X, weights=wts).fit()
X0 = pd.get_dummies(E.course).astype(float); X0["c"] = E.c
m0 = sm.WLS(E["edition_effect_%"], X0, weights=wts).fit()
alpha, lo, hi = m.params["c"], *m.conf_int().loc["c"]
print(f"editions {len(E)} | alpha course+year FE {alpha:.2f} [{lo:.2f}, {hi:.2f}] R2 {m.rsquared:.2f} | course FE only {m0.params['c']:.2f} [{m0.conf_int().loc['c',0]:.2f}, {m0.conf_int().loc['c',1]:.2f}]")
# partial residual: effect minus everything except the curve term
fe = X.drop(columns="c").values @ m.params.drop("c").values
E["net"] = E["edition_effect_%"] - fe

fig, ax = plt.subplots(figsize=(6.6, 4.4))
mk = {"New York City": "o", "Chicago": "s", "Boston": "^", "London": "D", "Big Sur": "v"}
for c, g in E.groupby("course"):
    ax.errorbar(g.c, g.net, yerr=1.96 * g["se_%"], fmt=mk[c], ms=5, lw=0.8, capsize=0, label=c, alpha=0.9)
xs = np.linspace(0, E.c.max() * 1.05, 50)
ax.plot(xs, alpha * xs, "k-", lw=1.2, label=rf"$\alpha$ = {alpha:.2f} [{lo:.2f}, {hi:.2f}]")
ax.plot(xs, 1.0 * xs, "k:", lw=1, label=r"$\alpha$ = 1 (training-run scale)")
top = np.arange(0, 1.36, 0.25)                                      # secondary axis: the WBGT that maps to each curve value
sec = ax.secondary_xaxis("top", functions=(lambda c: np.interp(c, pop, w), lambda t: curve(t)))
sec.set_xticks([10, 16, 18, 19, 20, 21]); sec.set_xlabel("race-window mean WBGT (°C)", fontsize=9)
ax.set_xlabel("learned curve at the edition's WBGT, $c(\\bar{W})$ (% of pace, training-run scale)")
ax.set_ylabel("edition effect net of course and year effects\n(% of finish time, within-person)")
ax.set_title(f"{len(E)} marathon editions, 2015–2025, five courses; WLS, course + calendar-year fixed effects", fontsize=9)
ax.axhline(0, color="grey", lw=0.5); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); out = os.path.join(HERE, "figures", "fig_raceday_scale.pdf"); fig.savefig(out); print("wrote", out)
