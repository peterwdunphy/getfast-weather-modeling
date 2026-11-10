#!/usr/bin/env python3
"""build_journal_figures.py - composite figures for the journal-length paper (journal/getfast-heat-journal.tex).

Writes three PDFs into journal/figures/ and prints every derived number that journal/numbers.tex quotes, so each
number traces to this script and its inputs. Nothing outside journal/ is written; no input file is modified.

  fig1_dose_response.pdf   A: the training-run heat curve c(W) and the same curve at the race-day scales
                           B: within-runner scale on c(W) by expected training-run duration, with the official
                              race-day scales placed at the median finish time of each distance
  fig2_raceday_scale.pdf   within-person edition effects against c(W) at each edition's race-window WBGT
                           (marathon, half marathon), with the fitted race-day scale
  fig4_between_runner.pdf  D: race-to-race persistence of the race-day scale in official results (split halves)
                           A: shrunk per-runner heart-rate heat slopes with the three tier cut-points
                           B: held-out (later-half) slope by tier set on the earlier half
                           C: race-day heat cost by thirds, thirds set from the training slope or from earlier hot
                              race-distance efforts

Figure 3 of the paper reuses ../figures/fig_travel_hr.png unchanged (no script here).

Specification switch (the race-day first stage is an open item in the long manuscript): FIRST_STAGE=v2 (default)
uses the identified first stage (athlinks_edition_effects_v2.parquet, log units, no drift term) and the v2 weather
table (one outlier edition removed); FIRST_STAGE=v1 uses the original first stage and weather table. The script
asserts that it reproduces the stored estimate of the chosen specification before plotting.

Inputs (read only):
  /weather/data/heat_percentile_averaged_v3.npz        learned population curve (training-run scale)
  /weather/data/effort_heat_tests_round2.json          R3: within-runner scale by expected duration, race alpha
  /weather/data/athlinks_edition_effects{,_v2}.parquet within-person edition effects
  /weather/data/edition_weather_official{,_v2}.parquet official race-window weather
  /weather/data/edition_tests_all_official{_v2_fs2,}.json  stored estimates (reproduction check)
  /weather/data/fade/resilience_buckets_20261028/resilience_buckets.json   tiers (histogram, held-out slopes)
  /weather/data/fade/combined_placement_20261109/{calibration_ra0,followups}.json   race-day heat cost by thirds
                                                       (combined placement model, miss-trait correlation fixed at 0)

Run: /home/bb/test_env/bin/python3 journal/build_journal_figures.py
"""
import json, os
import numpy as np, pandas as pd, statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")                                            # file output only, no display
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures"); os.makedirs(OUT, exist_ok=True)
D = "/weather/data"
FS = os.environ.get("FIRST_STAGE", "v2")                         # which race-day specification to draw

# ---- shared style: categorical slots 1-3 of the reference palette (validated all-pairs), grey for context ----------
BLUE, ORANGE, AQUA, GREY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "lines.linewidth": 1.6, "pdf.fonttype": 42})


def panel_letter(ax, s):
    """bold panel letter in the top-left corner, outside the plot area."""
    ax.text(-0.13, 1.04, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


# =================================================================================================================
# The learned curve. Median-runner sweep (percentile index 49), zeroed at WBGT 10 C, 0 below 10, held flat above 28.
# Identical construction to edition_tests_all_official_v2.py, so alpha multiplies the same quantity.
# =================================================================================================================
a = np.load(f"{D}/heat_percentile_averaged_v3.npz", allow_pickle=True)
w = a["wsweep"].astype(float); pc = a["pctl_curves"].astype(float)[49]
pop = pc - np.interp(10.0, w, pc)                                # % slower than at WBGT 10 C (arithmetic)
curve_pct = lambda x: np.interp(np.asarray(x, float), w, pop, left=0.0, right=pop[-1])
curve_log = lambda x: 100 * np.log1p(curve_pct(x) / 100)        # the same curve in 100 x log points
LOG_UNITS = FS == "v2"                                           # v2 works in log points end to end
curve = curve_log if LOG_UNITS else curve_pct

print("== learned training-run curve, % slower than at WBGT 10 C")
for t in [14, 16, 18, 20, 22, 24, 26, 28]:
    print(f"   c({t}) = {float(curve_pct(t)):.2f} %   ({float(curve_log(t)):.3f} log points)")


def race_pct(alpha, wbgt):
    """% of finish time slower than at WBGT 10 C when the race-day scale alpha multiplies the curve."""
    if LOG_UNITS:
        return 100 * np.expm1(alpha * curve_log(wbgt) / 100)
    return alpha * curve_pct(wbgt)


# =================================================================================================================
# Race-day scale: reproduce the stored edition regression for the chosen specification.
# effect_e = course_c + calendar-year_y + alpha * c(WBGT_e), WLS with weights 1 / SE^2, editions with >= 30 pairs.
# =================================================================================================================
if FS == "v2":
    EFF, WX, REF = (f"{D}/athlinks_edition_effects_v2.parquet", f"{D}/edition_weather_official_v2.parquet",
                    f"{D}/edition_tests_all_official_v2_fs2.json")
else:
    EFF, WX, REF = (f"{D}/athlinks_edition_effects.parquet", f"{D}/edition_weather_official.parquet",
                    f"{D}/edition_tests_all_official.json")
E = pd.read_parquet(EFF)
if "effect_log" in E:                                            # v2 stores log points: put them on the 100 x log scale
    E = E.assign(**{"edition_effect_%": 100 * E.effect_log, "se_%": 100 * E.se_log})
E = E[["course", "year", "edition_effect_%", "se_%", "n_pairs"]]
W = pd.read_parquet(WX)
W["deviation"] = W.wbgt_mean - W.climate60_wbgt
W["course"] = np.where(W.band == "half", W.race + " [half]", W.race)   # first-stage labels carry " [half]"
E = E.merge(W, on=["course", "year"], how="inner")
E = E[~((E.course == "London") & (E.year == 2020))]              # virtual mass field
E = E[E.n_pairs >= 30].dropna(subset=["wbgt_mean", "deviation"]).reset_index(drop=True)
E["c"] = curve(E.wbgt_mean.values)
REFJ = json.load(open(REF))

fits = {}
for band in ["marathon", "half"]:
    d = E[E.band == band].reset_index(drop=True)
    X = pd.get_dummies(d.course).astype(float).join(pd.get_dummies(d.year, prefix="y", drop_first=True).astype(float))
    X["c"] = d.c.values
    m = sm.WLS(d["edition_effect_%"].values, X, weights=1.0 / d["se_%"] ** 2).fit()
    # partial R^2 of the curve: share of the weighted variance left by the course and year effects that the curve
    # explains (the overall R^2 is inflated by the calendar trend the year effects absorb under the identified stage)
    m0 = sm.WLS(d["edition_effect_%"].values, X.drop(columns="c"), weights=1.0 / d["se_%"] ** 2).fit()
    pr2 = 1 - m.ssr / m0.ssr
    al, (lo, hi) = m.params["c"], m.conf_int().loc["c"]
    assert len(d) == REFJ[band]["editions"] and abs(al - REFJ[band]["alpha"][0]) < 1e-6, (band, len(d), al)
    d["net"] = d["edition_effect_%"].values - X.drop(columns="c").values @ m.params.drop("c").values  # partial residual
    fits[band] = dict(d=d, alpha=al, lo=lo, hi=hi, r2=m.rsquared, pr2=pr2)
    print(f"== {band} ({FS}): editions {len(d)}, courses {d.course.nunique()}, years {d.year.min()}-{d.year.max()}, "
          f"alpha {al:.2f} [{lo:.2f}, {hi:.2f}], R2 {m.rsquared:.2f} (partial, curve {pr2:.2f}), race-window WBGT {d.wbgt_mean.min():.1f} to "
          f"{d.wbgt_mean.max():.1f} C, editions >= 18 C {int((d.wbgt_mean >= 18).sum())}, >= 22 C {int((d.wbgt_mean >= 22).sum())}, "
          f"median of edition median finish {60 * d.median_finish_h.median():.0f} min")

# ---- implied race-day cost: % of finish time and minutes for a reference runner (4:00 marathon, 2:00 half) ---------
REF_MIN = {"marathon": 240.0, "half": 120.0}
print("== implied race-day cost relative to WBGT 10 C (alpha interval only; the curve is treated as fixed)")
for band in ["marathon", "half"]:
    f = fits[band]
    for t in [14, 16, 18, 20, 21, 22]:
        p, pl, ph = (float(race_pct(x, t)) for x in (f["alpha"], f["lo"], f["hi"]))
        print(f"   {band} WBGT {t}: {p:.1f} % [{pl:.1f}, {ph:.1f}] = {p * REF_MIN[band] / 100:.1f} min "
              f"[{pl * REF_MIN[band] / 100:.1f}, {ph * REF_MIN[band] / 100:.1f}] on a {REF_MIN[band] / 60:.0f}-h finish")
    p_tr = float(curve_pct(20))
    print(f"   training-run curve at WBGT 20: {p_tr:.2f} % = {p_tr * REF_MIN[band] / 100:.1f} min on a {REF_MIN[band] / 60:.0f}-h finish")

# =================================================================================================================
# R3: within-runner scale on c(W) by expected training-run duration (race-distance runs removed).
# =================================================================================================================
R3 = json.load(open(f"{D}/effort_heat_tests_round2.json"))["R3"]
bands = R3["training_scale_by_expected_duration"]["training_without_race_distance_runs"]
print("== within-runner training scale on c(W) by expected duration (R3)")
for b in bands:
    print(f"   {b['band_min']:>7s} min (median {b['median_expected_min']:.0f}): scale {b['scale']['coef']:.2f} "
          f"[{b['scale']['ci'][0]:.2f}, {b['scale']['ci'][1]:.2f}], n {b['n']:,} runs, {b['runners']:,} runners, "
          f">22 C {b['n_above_22C']:,}")
cont = R3["continuous_scale_x_log_duration"]["per_log_unit"]
print(f"   per log unit of duration: {cont['coef']:.2f} [{cont['ci'][0]:.2f}, {cont['ci'][1]:.2f}]")
for band in ["half", "marathon"]:
    rf = R3["remaining_factor"][band]["alpha_over_base"]
    print(f"   {band}: race alpha / duration-matched training scale = {rf['ratio']:.2f} [{rf['ci'][0]:.2f}, {rf['ci'][1]:.2f}]"
          f"; median finish {R3['alpha'][band]['median_finish_min']:.0f} min")
plc = R3["placebo_homogeneous_effect"]
print(f"   placebo (common true scale {plc['true_common_scale']:.2f}): band means "
      + ", ".join(f"{k} {v:.2f}" for k, v in plc["mean_scale_by_band"].items()))

# =================================================================================================================
# FIGURE 1
# Short races (5K to 10 miles, plus RunSignup half and marathon as checks) come from the official-results study
# /weather/data/fade/short_race_alpha_20261109 (same first stage and second stage; course-clustered intervals).
# =================================================================================================================
SR = pd.read_csv("/weather/data/fade/short_race_alpha_20261109/alpha_table.csv").set_index("group")
FIT = json.load(open("/weather/data/fade/short_race_alpha_20261109/alpha_vs_duration_fit.json"))
RSU_BANDS = ["5k", "4mi", "5mi", "8k", "10k", "12k", "15k", "10mi", "half", "marathon"]   # the ten bands of the fit
print("== short races (RunSignup official results; course-clustered 95% CI)")
for g in ["5k", "10k", "15k+10mi", "15k", "10mi", "half", "marathon"]:
    r = SR.loc[g]
    print(f"   {g:9s} alpha {r.alpha:.2f} [{r.ci_cl_lo:.2f}, {r.ci_cl_hi:.2f}] (classic [{r.ci_lo:.2f}, {r.ci_hi:.2f}]), editions {int(r.editions):,}, "
          f"courses {int(r.courses):,}, >= 18 C {int(r.n_wbgt_ge18):,}, median finish {60 * r.median_finish_h:.0f} min")
fa, ft = FIT["all_race_bands"], FIT["training"]
print(f"   race bands: alpha = {fa['intercept']:.2f} + {fa['slope_per_log_min']:.2f} ln(min), R2 {fa['r2']:.2f}, per doubling {fa['per_doubling']:.2f}; "
      f"training bands: per doubling {ft['per_doubling']:.2f}; race / training at matched duration "
      + ", ".join(f"{r['g']} {r['race_over_training']:.2f}" for r in FIT["race_over_training_at_same_duration"]))

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1.0, 1.05]})
# A: curves
xs = np.linspace(0, 28, 281)
axA.plot(xs, curve_pct(xs), color=GREY, lw=1.8, label="training runs (scale 1)")
r5 = SR.loc["5k"]
axA.fill_between(xs, race_pct(r5.ci_cl_lo, xs), race_pct(r5.ci_cl_hi, xs), color=BLUE, alpha=0.18, lw=0)
axA.plot(xs, race_pct(r5.alpha, xs), color=BLUE, label=f"5K races (scale {r5.alpha:.2f})")
for band, col, lab in [("half", AQUA, "half marathon"), ("marathon", ORANGE, "marathon")]:
    f = fits[band]; xm = np.linspace(0, f["d"].wbgt_mean.max(), 240)   # drawn only to the hottest edition observed
    axA.fill_between(xm, race_pct(f["lo"], xm), race_pct(f["hi"], xm), color=col, alpha=0.18, lw=0)
    axA.plot(xm, race_pct(f["alpha"], xm), color=col, label=f"{lab} races (scale {f['alpha']:.2f})")
axA.set_xlabel("WBGT (°C)"); axA.set_ylabel("slower than at WBGT 10 °C (% of time)")
axA.set_xlim(0, 28); axA.set_ylim(-0.3, 16)
axA.legend(loc="upper left", frameon=False)
axA.set_title("Heat curve at training-run and race-day scales")
panel_letter(axA, "A")
# B: scale by duration
mid = np.array([b["median_expected_min"] for b in bands]); sc = np.array([b["scale"]["coef"] for b in bands])
lo_ = np.array([b["scale"]["ci"][0] for b in bands]); hi_ = np.array([b["scale"]["ci"][1] for b in bands])
axB.errorbar(mid, sc, yerr=[sc - lo_, hi_ - sc], fmt="o", ms=5, color=GREY, ecolor=GREY, lw=1.2, capsize=0,
             label="training runs, by expected duration")
tt = np.linspace(18, 190, 50); axB.plot(tt, ft["intercept"] + ft["slope_per_log_min"] * np.log(tt), color=GREY, lw=0.9, ls="--")
m_ = 60 * SR.loc[RSU_BANDS, "median_finish_h"].values; a_ = SR.loc[RSU_BANDS, "alpha"].values
l_ = SR.loc[RSU_BANDS, "ci_cl_lo"].values; h_ = SR.loc[RSU_BANDS, "ci_cl_hi"].values
axB.errorbar(m_, a_, yerr=[a_ - l_, h_ - a_], fmt="o", ms=4, mfc="white", color=BLUE, lw=0.9, capsize=0,
             label="races 5K to marathon (RunSignup)")
tr = np.linspace(30, 290, 50); axB.plot(tr, fa["intercept"] + fa["slope_per_log_min"] * np.log(tr), color=BLUE, lw=0.9, ls="--")
for band, col, mk, lab in [("half", AQUA, "D", "half marathon (main source)"), ("marathon", ORANGE, "s", "marathon (main source)")]:
    f = fits[band]; tmin = R3["alpha"][band]["median_finish_min"]
    axB.errorbar([tmin], [f["alpha"]], yerr=[[f["alpha"] - f["lo"]], [f["hi"] - f["alpha"]]], fmt=mk, ms=6,
                 color=col, lw=1.2, capsize=0, label=lab)
axB.axhline(1, color=INK, lw=0.8, ls=":")
axB.set_xscale("log"); axB.set_xticks([20, 35, 60, 90, 135, 260]); axB.set_xticklabels(["20", "35", "60", "90", "135", "260"])
axB.minorticks_off()
axB.set_xlabel("duration (min, log scale)")
axB.set_ylabel("scale on the training-run curve")
axB.set_ylim(0, 6.6); axB.legend(loc="upper left", frameon=False, fontsize=6.5)
axB.set_title("Scale by duration: training runs and races")
panel_letter(axB, "B")
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_dose_response.pdf"); plt.close(fig)

# =================================================================================================================
# FIGURE 2
# =================================================================================================================
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7), sharey=True)
unit = "100 × log ratio" if LOG_UNITS else "%"
for ax, band, col, name in zip(axes, ["marathon", "half"], [ORANGE, AQUA], ["Marathon", "Half marathon"]):
    f = fits[band]; d = f["d"]
    ax.errorbar(d.c, d.net, yerr=1.96 * d["se_%"], fmt="o", ms=2.6, lw=0.4, color=GREY, alpha=0.6, capsize=0,
                label=f"editions (n = {len(d)})")
    xx = np.linspace(0, d.c.max() * 1.05, 50)
    ax.plot(xx, f["alpha"] * xx, color=col, lw=1.8, label=f"fitted scale {f['alpha']:.2f} [{f['lo']:.2f}, {f['hi']:.2f}]")
    ax.plot(xx, xx, color=INK, lw=0.9, ls=":", label="scale 1 (training-run curve)")
    ax.axhline(0, color=GREY, lw=0.5)
    sec = ax.secondary_xaxis("top", functions=(lambda c: np.interp(c, curve(w), w), lambda t: curve(t)))
    sec.set_xticks([16, 18, 20, 22] + ([24] if band == "half" else [])); sec.set_xlabel("race-window mean WBGT (°C)")
    ax.set_xlabel(f"training-run curve at the edition's WBGT ({unit})")
    ax.set_title(f"{name}: {len(d)} editions, {d.course.nunique()} races,\n{d.year.min()}–{d.year.max()}, "
                 f"partial $R^2$ of the curve = {f['pr2']:.2f}", pad=6, y=1.18)
    ax.legend(loc="upper left", frameon=False)
axes[0].set_ylabel(f"edition effect net of race and year effects\n(finish time, {unit})")
panel_letter(axes[0], "A"); panel_letter(axes[1], "B")
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_raceday_scale.pdf"); plt.close(fig)

# =================================================================================================================
# FIGURE 4: between-runner differences
# =================================================================================================================
RB = json.load(open(f"{D}/fade/resilience_buckets_20261028/resilience_buckets.json"))
CAL = json.load(open(f"{D}/fade/combined_placement_20261109/calibration_ra0.json"))   # thirds of the posterior trait
FU = json.load(open(f"{D}/fade/combined_placement_20261109/followups.json"))          # thirds of the training trait
hist = RB["A_buckets"]["histogram"]; edges = np.array(hist["edges"], float); cnt = np.array(hist["shrunk"], float)
cuts = RB["A_buckets"]["3"]["edges"]; mu = RB["params"]["hr"]["mu"]
ho = RB["B_holdout"]["3"]
print("== tiers: cut-points", cuts, "; persons", int(cnt.sum()), "; population mean", mu)
print("   held-out later-half HR slope by tier:", [(b["late_hr_slope"], b["late_hr_ci95"], b["runners"]) for b in ho["buckets"]],
      "top minus bottom", ho["top_minus_bottom_hr"])
print("   median posterior probability of own tier: K=3", RB["A_buckets"]["3"]["median_confidence"],
      "K=4", RB["A_buckets"]["4"]["median_confidence"], "K=5", RB["A_buckets"]["5"]["median_confidence"])

fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.8))
a1, a2, a3, a4 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]
# A: histogram of shrunk slopes
a1.bar(edges[:-1], cnt, width=np.diff(edges), align="edge", color=BLUE, alpha=0.75, edgecolor="white", lw=0.5)
for cpt in cuts:
    a1.axvline(cpt, color=INK, lw=0.9, ls="--")
a1.axvline(mu, color=ORANGE, lw=1.2)                            # mean of the person slopes (value in the legend text)
a1.set_xlim(-0.1, 1.2); a1.set_xlabel("HR heat slope above WBGT 15 °C\n(bpm per °C, shrunk)")
a1.set_ylabel("runners"); a1.set_title(f"Per-runner slopes (n = {int(cnt.sum()):,})")
panel_letter(a1, "A")
# B: held-out later-half slope by tier
lab3 = ["lower", "middle", "upper"]
for i, b in enumerate(ho["buckets"]):
    a2.errorbar([i], [b["late_hr_slope"]], yerr=[[b["late_hr_slope"] - b["late_hr_ci95"][0]], [b["late_hr_ci95"][1] - b["late_hr_slope"]]],
                fmt="o", ms=5, color=BLUE, lw=1.2, capsize=0)
a2.set_xticks(range(3)); a2.set_xticklabels(lab3); a2.set_xlim(-0.5, 2.5)
a2.set_xlabel("tier set on earlier half"); a2.set_ylabel("later-half HR heat slope (bpm per °C)")
a2.set_title(f"Held-out separation\n(n = {sum(b['runners'] for b in ho['buckets']):,} runners)")
a2.set_ylim(0, 0.8); panel_letter(a2, "B")
# C: race-day heat cost by thirds
# observed multiplier = (loss - personal miss estimated from earlier data) / typical cost, hot efforts with an earlier
# hot effort; thirds set only from data dated before each effort; runner-bootstrap intervals
T = FU["training_trait_thirds"]["with earlier hot"]; C = CAL["thirds_with_earlier_hot"]
for k, (src, col, mk, lab) in enumerate([(T, GREY, "o", "set from training heat slopes"),
                                        (C, ORANGE, "s", "set from earlier hot efforts")]):
    for i, g in enumerate(src["groups"]):
        y, (l, h) = g["observed_rho"], g["observed_ci95_runners"]
        a3.errorbar([i + (k - 0.5) * 0.22], [y], yerr=[[y - l], [h - y]], fmt=mk, ms=5, color=col, lw=1.2, capsize=0,
                    label=lab if i == 0 else None)
    print(f"   race-day thirds {lab}: " + ", ".join(f"{g['group']} {g['observed_rho']:.2f} {g['observed_ci95_runners']} (n {g['efforts']})" for g in src["groups"])
          + f"; gap {src['sensitive_minus_resilient']:.2f} {src['gap_ci95_runners']}")
a3.axhline(1, color=INK, lw=0.8, ls=":")
a3.set_xticks(range(3)); a3.set_xticklabels(["lower", "middle", "upper"]); a3.set_xlim(-0.5, 2.5)
a3.set_xlabel("third of the pre-race heat trait")
a3.set_ylabel("race-day heat cost / typical cost")
a3.set_title(f"Hot race-distance efforts\n(n = {sum(g['efforts'] for g in T['groups'])} and {sum(g['efforts'] for g in C['groups'])})")
a3.set_ylim(0, 1.7); a3.legend(loc="lower right", frameon=False, fontsize=6.5)
panel_letter(a3, "C")
# D: race-to-race persistence of the race-day scale in official results (race_resilience_persistence_20261109).
# tau^2 = covariance of a runner's heat slopes from two halves of their races (alpha units^2, course-jackknife 95% CI);
# hollow markers: placebo P5 (which year was hot shuffled within course). REML: random-slope model, all repeat runners.
PS = json.load(open(f"{D}/fade/race_resilience_persistence_20261109/summary.json"))
pm, ph = PS["marathon"], PS["half"]
rows = [("shared courses\n(naive)", pm["S0_naive"], pm["S0_naive"]["P5"]),
        ("course-\ndisjoint", pm["S6_course_disjoint_within"], pm["S6_course_disjoint_within"]["P5"]),
        ("+ personal\ntrends", pm["S7_course_disjoint_within_trend"], pm["S7_course_disjoint_within_trend"]["P5"]),
        ("+ year-\ndisjoint", dict(tau2=pm["period_check"]["S8_course_and_year_disjoint"]["tau2"],
                                  ci=pm["period_check"]["S8_course_and_year_disjoint"]["ci95"]), None)]
reml = [h for h in pm["hier"] if h["x"] == "c_within" and h["hetero"] and not h["trend"] and h["persons"] == 562388][0]
for i, (lab, r, p5) in enumerate(rows):
    a4.errorbar([i], [r["tau2"]], yerr=[[r["tau2"] - r["ci"][0]], [r["ci"][1] - r["tau2"]]], fmt="o", ms=5, color=ORANGE, lw=1.2, capsize=0,
                label="marathon, split-half" if i == 0 else None)
    if p5 is not None:
        a4.plot([i + 0.18], [p5], marker="o", ms=5, mfc="white", color=GREY, ls="none", label="placebo (hot years shuffled)" if i == 0 else None)
a4.errorbar([4], [reml["tau2"]], yerr=[[0], [reml["tau2_ci95_profile"][1] - reml["tau2"]]], fmt="s", ms=5, color=INK, lw=1.2, capsize=0,
            label="marathon, random-slope model")
h6 = ph["S6_course_disjoint_within"]
a4.errorbar([5], [h6["tau2"]], yerr=[[h6["tau2"] - h6["ci"][0]], [h6["ci"][1] - h6["tau2"]]], fmt="D", ms=5, color=AQUA, lw=1.2, capsize=0,
            label="half marathon, course-disjoint")
a4.axhline(0, color=GREY, lw=0.6)
a4.set_xticks(range(6)); a4.set_xticklabels([r[0] for r in rows] + ["random-\nslope", "half,\ndisjoint"], fontsize=6.3)
a4.set_ylabel("between-runner variance of the\nrace-day scale (scale units$^2$)")
a4.set_ylim(-30, 100); a4.legend(loc="upper right", frameon=False, fontsize=6.2)
a4.set_title(f"Race-to-race persistence, official results\n(marathon: {int(round(pm['S6_course_disjoint_within']['persons'])):,} runners, course-disjoint)")
panel_letter(a4, "D")
print("== race-to-race persistence (summary.json): S6", round(pm["S6_course_disjoint_within"]["tau2"], 2), [round(v, 2) for v in pm["S6_course_disjoint_within"]["ci"]],
      "minus P5", round(pm["S6_course_disjoint_within"]["tau2_minus_P5"], 2), [round(v, 2) for v in pm["S6_course_disjoint_within"]["tau2_minus_P5_ci"]],
      "| S7", round(pm["S7_course_disjoint_within_trend"]["tau2"], 2), [round(v, 2) for v in pm["S7_course_disjoint_within_trend"]["ci"]],
      "| S8", round(pm["period_check"]["S8_course_and_year_disjoint"]["tau2"], 2), [round(v, 2) for v in pm["period_check"]["S8_course_and_year_disjoint"]["ci95"]],
      "| naive", round(pm["S0_naive"]["tau2"], 1), "P5", round(pm["S0_naive"]["P5"], 1), "| REML", reml["tau2"], reml["tau2_ci95_profile"],
      "| half S6", round(h6["tau2"], 2), [round(v, 2) for v in h6["ci"]])
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_between_runner.pdf"); plt.close(fig)
print("wrote", ", ".join(sorted(os.listdir(OUT))))
