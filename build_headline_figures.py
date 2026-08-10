"""build_headline_figures.py

Two headline figures for the Results section, both built to answer "so what?"
in units a runner recognises rather than in model-error units.

  1. fig_heat_cost   -- the learned population heat curve, with the runner-to-runner
     spread as a band, and a second axis giving the cost in minutes for a runner
     whose cool-weather marathon is 3:30. Annotated at three reference conditions.

  2. fig_bias_fix    -- the paper's central practical result: the weather-blind
     model predicts hot runs too fast, and adding heat information removes that
     systematic optimism. Drawn as bias against WBGT with the optimistic region
     shaded, plus a bar inset in finish-time minutes for the hottest band.

Curve source: /weather/data/heat_percentile_averaged.npz (population sweep and
per-runner percentile curves, both anchored at 10 C).
Bias source: notebook 10 held-out bin table, transcribed in BINS below.

Run: /home/bb/test_env/bin/python3 build_headline_figures.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT_PDF = "/home/bb/weather/getfast-weather-modeling/figures"
OUT_PNG = "/tmp/claude-1000/-home-bb-weather/599857c7-9ec0-49e5-8a67-0c0c8644ad80/scratchpad/paperfigs"
os.makedirs(OUT_PNG, exist_ok=True)

# Okabe-Ito, fixed roles across every figure in the paper.
INK, MUTE, GRID = "#1a1a1a", "#5c5b57", "#e6e5df"
BLUE, VERM, GREEN, ORANGE, GREY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#8a897f"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.edgecolor": MUTE, "axes.linewidth": 0.8,
    "xtick.color": MUTE, "ytick.color": MUTE, "text.color": INK,
    "axes.labelcolor": INK, "axes.titlecolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8.6, "figure.dpi": 150,
})

REF = ((210.0, "3:30"), (240.0, "4:00"))   # reference finish times for the minute conversions
REF_MIN = REF[0][0]                        # 3:30, used for the right-hand axis


def finish(fig, name):
    fig.savefig(f"{OUT_PDF}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT_PNG}/{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved", name)


# =========================================================================== #
# FIGURE 1 -- what the heat costs, in percent and in minutes
# =========================================================================== #
def fig_heat_cost():
    a = np.load("/weather/data/heat_percentile_averaged.npz", allow_pickle=True)
    w = a["wsweep"].astype(float)
    pc = a["pctl_curves"].astype(float)
    anc = lambda r: r - np.interp(10.0, w, r)      # re-anchor every curve at 10 C
    med, p10, p90 = anc(pc[49]), anc(pc[9]), anc(pc[89])

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    # runner-to-runner spread first, so the median sits on top of it
    ax.fill_between(w, p10, p90, color=BLUE, alpha=0.16, lw=0,
                    label="runner-to-runner spread (10th--90th percentile)")
    ax.plot(w, med, color=BLUE, lw=3.0, label="population heat curve (median runner)")
    ax.axhline(0, color=GREY, lw=0.8)

    # three reference conditions, labelled in both units
    for x in (20.0, 24.0, 28.0):
        y = float(np.interp(x, w, med))
        ax.plot([x, x], [0, y], color=MUTE, lw=0.7, ls=(0, (2, 2)), zorder=1)
        ax.plot([x], [y], "o", color=BLUE, ms=6, mec="white", mew=0.8, zorder=4)
        mins = "  /  ".join(f"+{y/100*t:.0f} min ({lab})" for t, lab in REF)
        ax.annotate(f"{y:.1f}%\n{mins}", (x, y),
                    xytext=(-4, 10), textcoords="offset points",
                    ha="right", va="bottom", fontsize=8.2, color=INK, linespacing=1.3)

    ax.set_xlim(10, 28)
    ax.set_ylim(-0.3, max(p90) * 1.18)
    ax.set_xlabel("wet-bulb globe temperature during the run (°C)")
    ax.set_ylabel("predicted pace slowdown vs 10°C (%)")

    # right-hand axis converts the same quantity into minutes on a 3:30 marathon
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim()[0] / 100 * REF_MIN, ax.get_ylim()[1] / 100 * REF_MIN)
    ax2.set_ylabel("added time for a 3:30 marathon (min)")
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)

    ax.set_title("Predicted slowdown as conditions warm", loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    finish(fig, "fig_heat_cost")


# =========================================================================== #
# FIGURE 2 -- the systematic optimism on hot days, and its correction
# =========================================================================== #
def fig_bias_fix():
    # (wbgt_mid, n, bias_weatherblind, bias_population, bias_personalized)
    BINS = np.array([
        [0.82, 1951, -2.061, -1.660, -1.205],
        [7.81, 2673, -2.295, -1.924, -1.471],
        [12.57, 3888, -2.093, -1.666, -1.242],
        [16.49, 2621, -1.467, -1.404, -0.964],
        [18.99, 1529, -1.013, -1.238, -0.709],
        [21.01, 1431, -0.473, -1.537, -0.962],
        [23.37, 1799, 0.755, -1.336, -0.715],
    ])
    x, n = BINS[:, 0], BINS[:, 1]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.2, 4.4),
                                   gridspec_kw={"width_ratios": [1.75, 1]})

    # --- A: bias across the temperature range --------------------------------
    axA.axhline(0, color=INK, lw=1.0)
    # the region above zero is where a model predicts a run faster than it was run
    axA.fill_between([-1, 26], 0, 1.4, color=VERM, alpha=0.07, lw=0)
    axA.annotate("predicted too fast", (-0.4, 1.15), fontsize=8.4, color=VERM, style="italic")
    for j, (c, lab, lw) in enumerate([(GREY, "weather-blind", 2.4),
                                      (BLUE, "+ population curve", 1.8),
                                      (GREEN, "+ personalized score", 1.8)]):
        axA.plot(x, BINS[:, 2 + j], "o-", color=c, lw=lw, ms=5, label=lab, zorder=3 - j)
    axA.set_xlim(-1, 26)
    axA.set_xlabel("wet-bulb globe temperature (°C)")
    axA.set_ylabel("signed bias (%)")
    axA.set_title("A. Prediction bias across conditions", loc="left", fontsize=10.5)
    axA.legend(loc="lower left")

    # --- B: the hottest band, in minutes -------------------------------------
    labs = ["weather-blind", "+ population\ncurve", "+ personalized\nscore"]
    pct = BINS[-1, 2:5]
    cols = [GREY, BLUE, GREEN]
    y = np.arange(3)[::-1]
    axB.axvline(0, color=INK, lw=1.0)
    # paired bars: one per reference finish time
    for k, (t, lab) in enumerate(REF):
        off = 0.19 if k == 0 else -0.19
        axB.barh(y + off, pct / 100 * t, color=cols, height=0.34,
                 alpha=1.0 if k == 0 else 0.55)
    for yi, v in zip(y, pct):
        a330, a400 = v / 100 * REF[0][0], v / 100 * REF[1][0]
        axB.annotate(f"{a330:+.1f} / {a400:+.1f} min", (max(a330, a400) if v > 0 else min(a330, a400), yi),
                     xytext=(8 if v > 0 else -8, 0), textcoords="offset points",
                     va="center", ha="left" if v > 0 else "right",
                     fontsize=8.4, fontweight="bold", color=INK)
    axB.set_yticks(y); axB.set_yticklabels(labs, fontsize=8.6)
    axB.set_xlim(-7.6, 5.2)
    axB.set_xlabel("bias in minutes (3:30 solid, 4:00 pale)")
    axB.set_title(f"B. Hottest band (22--25°C, n = {int(n[-1]):,})", loc="left", fontsize=10.5)
    axB.grid(axis="y", visible=False)

    fig.tight_layout()
    finish(fig, "fig_bias_fix")


# =========================================================================== #
# FIGURE 3 -- added time as a function of cool-weather finish time and WBGT
# =========================================================================== #
def fig_heat_nomogram():
    """A lookup surface: find your cool-weather time on the x axis and the
    forecast WBGT on the y axis, read the added minutes off the contours.

    The penalty is multiplicative, so added time is (cool-weather minutes) x
    f(WBGT)/100 with f the population curve. Contours are therefore hyperbolas,
    which is itself the point: the same conditions cost a slower runner more
    absolute time than a faster one."""
    a = np.load("/weather/data/heat_percentile_averaged.npz", allow_pickle=True)
    w = a["wsweep"].astype(float)
    med = a["pctl_curves"].astype(float)[49]
    med = med - np.interp(10.0, w, med)

    # the sweep is stored on a coarse 37-point grid whose plateaus show up as
    # staircases once contoured; PCHIP re-interpolates smoothly without
    # introducing overshoot or breaking monotonicity
    from scipy.interpolate import PchipInterpolator
    k = np.ones(5) / 5.0
    sm = np.convolve(np.pad(med, 2, mode="edge"), k, mode="valid")
    sm = np.maximum.accumulate(sm - sm[0])          # monotone, anchored at 0
    # edge padding drags the steep upper end down ~0.4 pp; rescale so the surface
    # reproduces the 28 C value reported in the text exactly
    sm *= float(np.interp(28.0, w, med)) / float(np.interp(28.0, w, sm))
    f = PchipInterpolator(w, sm)

    T = np.linspace(150, 330, 400)          # cool-weather finish, 2:30 to 5:30
    W = np.linspace(10, 28, 400)            # forecast WBGT
    TT, WW = np.meshgrid(T, W)
    ADD = TT * f(WW) / 100.0

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    levels = np.arange(0, 26, 1.0)
    cf = ax.contourf(TT, WW, ADD, levels=levels, cmap="YlOrRd", extend="max")
    # labelled contours at the round values a runner would actually use
    cs = ax.contour(TT, WW, ADD, levels=[2, 5, 10], colors="#40342b",
                    linewidths=1.0, alpha=0.85)
    ax.clabel(cs, fmt=lambda v: f"+{v:.0f} min", fontsize=8.6, inline=True,
              manual=[(300, 18.6), (296, 22.6), (268, 26.6)])

    # the two reference runners used elsewhere in the paper
    for t, lab in REF:
        ax.axvline(t, color=INK, lw=1.0, ls=(0, (4, 3)), alpha=0.75)
        ax.annotate(lab, (t, 10.4), rotation=90, ha="right", va="bottom",
                    fontsize=8.6, color=INK, fontweight="bold")

    ax.set_xticks([150, 180, 210, 240, 270, 300, 330])
    ax.set_xticklabels(["2:30", "3:00", "3:30", "4:00", "4:30", "5:00", "5:30"])
    ax.set_xlabel("cool-weather finish time (10\u00b0C reference)")
    ax.set_ylabel("wet-bulb globe temperature on race day (\u00b0C)")
    ax.set_title("Predicted added finish time", loc="left")
    ax.grid(False)
    cb = fig.colorbar(cf, ax=ax, pad=0.02)
    cb.set_label("added time (min)")
    cb.outline.set_visible(False)
    fig.tight_layout()
    finish(fig, "fig_heat_nomogram")


if __name__ == "__main__":
    print("building headline figures ...")
    fig_heat_cost()
    fig_bias_fix()
    fig_heat_nomogram()
    print("done ->", OUT_PDF)
