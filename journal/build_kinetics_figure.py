#!/usr/bin/env python3
"""build_kinetics_figure.py - Figure 3 of the journal paper: how heat acclimation builds and fades, read from heart
rate (HR) at matched load. Replaces the earlier reuse of ../figures/fig_travel_hr.png. Estimates nothing: it plots the
stored aggregates of /weather/data/fade/acclimation_kinetics_20261109 and prints the numbers the text quotes.

Outcome in every panel: on a run, HR minus the runner's load model (pace, duration, climb, fitted on cool runs), the
start-hour effect and the population heat cost; "excess" is that residual relative to each design's own baseline.
Heat is counted from WBGT 22 C (a hot run lasts 20 min or more at WBGT 22 C or above).

  A  onset: hot-run excess by the number of earlier hot-run days since the first hot run after 45+ days without one
     (episode fixed effects, days-since-onset trend, control for a hot run the day before)   onset_ydayctrl.json
  B  trips > 500 km from a rolling home: excess on away runs minus the pre-trip home mean, by day since arrival, for
     hot (+5 C or more), neutral (within 2 C) and cool (-5 C or less) trips; the exponential fitted to hot trips
                                                                                                travel.json
  C  breaks after a heat block: excess on the first hot run back minus the block's last week, by break length
                                                                                                breaks.json
Intervals: estimate +/- 1.96 person-clustered SE (A, C) or SE across runs (B), as stored.

Run: /home/bb/test_env/bin/python3 journal/build_kinetics_figure.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")                                            # file output only
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
K = "/weather/data/fade/acclimation_kinetics_20261109"
BLUE, ORANGE, GREY, INK = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "lines.linewidth": 1.4, "pdf.fonttype": 42})

O = json.load(open(f"{K}/onset_ydayctrl.json"))
T = json.load(open(f"{K}/travel.json"))
B = json.load(open(f"{K}/breaks.json"))
S = json.load(open(f"{K}/summary.json"))

fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.2, 3.0))

# ---- A: onset, excess by earlier hot-run days (k) ---------------------------------------------------------------
fe = O["fe_hot_by_k_ctrl_yday"]
labs = ["0", "1", "2", "3-4", "5-7", "8-12", "13+"]
est = np.array([fe[f"hk_{l}"][0] for l in labs]); se = np.array([fe[f"hk_{l}"][1] for l in labs])
x = np.arange(len(labs))
a1.errorbar(x, est, yerr=1.96 * se, fmt="o", ms=4, color=ORANGE, lw=1.1, capsize=0)
a1.axhline(0, color=GREY, lw=0.6)
a1.set_xticks(x); a1.set_xticklabels(labs)
a1.set_xlabel("earlier hot-run days since onset")
a1.set_ylabel("HR excess on hot runs (bpm)")
ob = S["build_onset_ydayctrl"]
a1.set_title(f"Onset after 45+ days without heat\n({ob['episodes']:,} episodes, {ob['persons']:,} persons)")
print(f"A onset: episodes {ob['episodes']}, persons {ob['persons']}, half-life {ob['half_life_exposures']:.2f} exposures "
      f"{[round(v, 2) for v in ob['half_life_exposures_ci95']]}, decline {ob['decline_bpm']:.2f} {[round(v, 2) for v in ob['decline_ci95']]}, "
      f"count beats calendar {ob['share_count_beats_calendar']:.3f}, count beats ambient {ob['share_count_beats_ambient']:.3f}")
print("   joint count vs days (coef, SE):", ob["joint_count_vs_days"]["coef_d"], "; count vs ambient:", ob["joint_count_vs_ambient"]["coef_a"])

# ---- B: trips, by day since arrival -------------------------------------------------------------------------------
for grp, col, mk, lab in [("hot", ORANGE, "o", "hot trips"), ("neutral", GREY, "s", "neutral trips"), ("cool", BLUE, "^", "cool trips")]:
    g = T["A_build"][grp]; days = sorted(int(d) for d in g["by_day"])
    m = np.array([g["by_day"][str(d)]["mean"] for d in days]); s = np.array([g["by_day"][str(d)]["se"] for d in days])
    off = {"hot": 0.0, "neutral": -0.18, "cool": 0.18}[grp]
    a2.errorbar(np.array(days) + off, m, yerr=1.96 * s, fmt=mk, ms=3, color=col, lw=0.7, capsize=0, alpha=0.85,
                label=f"{lab} (n = {g['n_trips']:,})")
e = T["A_build"]["hot"]["expo"]
tt = np.linspace(0, 14, 200)
a2.plot(tt, e["E_inf"] + (e["E_0"] - e["E_inf"]) * np.exp(-tt / e["tau"]), color=ORANGE, lw=1.4)
a2.axhline(0, color=GREY, lw=0.6)
a2.set_xlabel("day since arrival"); a2.set_ylabel("HR excess vs pre-trip home runs (bpm)")
a2.set_title("Trips over 500 km from home\n(line: exponential fit, hot trips)")
a2.legend(loc="lower left", frameon=False, fontsize=6.3)
bt = S["build_travel"]
print(f"B trips: hot {bt['hot_trips']['n_trips']} trips / {bt['hot_trips']['n_persons']} persons, half-life "
      f"{bt['hot_trips']['half_life_days']:.2f} {[round(v, 2) for v in bt['hot_trips']['half_life_ci95']]}, drop {bt['hot_trips']['drop_bpm']:.2f} "
      f"{[round(v, 2) for v in bt['hot_trips']['drop_ci95']]}")
for grp, w in [("hot", bt["hot_trips"]["within_trip_slope_d0_7"]), ("neutral", bt["placebo_slopes"]["neutral"]), ("cool", bt["placebo_slopes"]["cool"])]:
    print(f"   within-trip slope days 0-7, {grp}: {w['coef']:+.3f} [{w['coef'] - 1.96 * w['se']:+.3f}, {w['coef'] + 1.96 * w['se']:+.3f}], trips {w['n_trips']}")
h0, n0 = T["A_build"]["hot"]["day0_2"]["mean"], T["A_build"]["neutral"]["day0_2"]["mean"]
print(f"   days 0-2: hot {h0:+.2f}, neutral {n0:+.2f}, hot minus neutral {h0 - n0:+.2f}; neutral trips n {T['A_build']['neutral']['n_trips']}, "
      f"cool trips n {T['A_build']['cool']['n_trips']}")

# ---- C: breaks, first hot run back minus block end, by break length -----------------------------------------------
Ls = ["1", "2-3", "4-7", "8-14", "15-28", "29-56", "57-120"]
xm = {"1": 1, "2-3": 2.5, "4-7": 5.5, "8-14": 11, "15-28": 21.5, "29-56": 42.5, "57-120": 88.5}
m = np.array([B["by_L"][l]["d_hot"]["mean"] for l in Ls]); s = np.array([B["by_L"][l]["d_hot"]["se"] for l in Ls])
a3.errorbar([xm[l] for l in Ls], m, yerr=1.96 * s, fmt="o", ms=4, color=ORANGE, lw=1.1, capsize=0)
a3.axhline(0, color=GREY, lw=0.6)
a3.set_xscale("log"); a3.set_xticks([1, 2.5, 5.5, 11, 21.5, 42.5, 88.5]); a3.set_xticklabels(Ls, rotation=45, fontsize=6.5)
a3.minorticks_off()
a3.set_xlabel("days without a hot run (break length)")
a3.set_ylabel("first hot run back minus block end (bpm)")
a3.set_title(f"Breaks after a heat block\n({B['breaks']:,} breaks, {B['persons']:,} persons)")
db = S["decay_breaks"]
print(f"C breaks: {db['breaks']} breaks / {db['persons']} persons; half-life {db['half_life_days']:.2f} {[round(v, 2) for v in db['half_life_ci95']]}"
      f" (80% {[round(v, 2) for v in db['half_life_ci80']]}); full loss {db['full_loss_bpm']:.2f} {[round(v, 2) for v in db['full_loss_ci95']]}")
for l in Ls:
    v = B["by_L"][l]["d_hot"]; print(f"   L {l:>6s}: {v['mean']:+.2f} [{v['mean'] - 1.96 * v['se']:+.2f}, {v['mean'] + 1.96 * v['se']:+.2f}], n {v['n']}")
dn = S["decay_breaks"]["did_nonhot"]["L_15-28"]
print(f"   non-hot difference-in-differences at 15-28 d: {dn[0]:+.2f} (SE {dn[1]:.2f}, p {dn[2]:.2f}); n breaks {S['decay_breaks']['did_nonhot_n']}")
og = O["state_grid_onset_boot"]["decay_half_life_quantiles"]; kp = S["kinetic"]["_ydayctrl"]["hot22"]["decay_half_life_q"]
print(f"   state-model decay half-life: onset episodes median {og['50']:.1f} (80% {og['10']:.1f}-{og['90']:.1f}); "
      f"all runs {S['kinetic']['_ydayctrl']['hot22']['decay_half_life']:.1f} (80% {kp['10']:.1f}-{kp['90']:.1f})")
lv = S["kinetic"]["_ydayctrl"]["hot22"]; print("   future placebo, heat-specific:", lv["placebo_future_best"], "level:", lv["placebo_future_level"])

for ax, lt in zip((a1, a2, a3), "ABC"):
    ax.text(-0.2, 1.17, lt, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
fig.tight_layout()
out = os.path.join(HERE, "figures", "fig3_acclimation_kinetics.pdf"); fig.savefig(out); print("wrote", out)
