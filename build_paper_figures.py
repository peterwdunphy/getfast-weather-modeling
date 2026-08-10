"""build_paper_figures.py

Figure suite for the modeling-and-results paper. The Results report the analysis
in notebooks 09/10 (population and personalized deep-learning heat features added
to a weather-blind XGBoost pace model). Values are the authoritative held-out
numbers from notebook 10 (15,892 activities, 104 runners, 3 XGB seeds, robust
p99-clipped MAPE and 1st/99th-clipped signed bias).

Design rules (project conventions): titles state WHAT is plotted (interpretation
lives in the LaTeX captions), one colorblind-safe palette (Okabe-Ito) in fixed
roles, recessive axes, confidence intervals where an estimate has them.

Outputs vector PDFs into figures/ and PNGs into a scratch dir for review.
Run: /home/bb/test_env/bin/python3 build_paper_figures.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_PDF = "/home/bb/weather/getfast-weather-modeling/figures"
OUT_PNG = "/tmp/claude-1000/-home-bb-weather/599857c7-9ec0-49e5-8a67-0c0c8644ad80/scratchpad/paperfigs"
os.makedirs(OUT_PNG, exist_ok=True)

# Okabe-Ito palette, fixed roles.
INK, MUTE, GRID = "#1a1a1a", "#5c5b57", "#e6e5df"
BLUE, VERM, GREEN, ORANGE, PURPLE, GREY = ("#0072B2","#D55E00","#009E73",
                                           "#E69F00","#CC79A7","#8a897f")
plt.rcParams.update({
    "figure.facecolor":"white","axes.facecolor":"white",
    "font.size":10,"axes.titlesize":11,"axes.labelsize":10,
    "axes.edgecolor":MUTE,"axes.linewidth":0.8,
    "xtick.color":MUTE,"ytick.color":MUTE,"text.color":INK,
    "axes.labelcolor":INK,"axes.titlecolor":INK,
    "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.7,
    "axes.spines.top":False,"axes.spines.right":False,
    "legend.frameon":False,"legend.fontsize":8.6,"figure.dpi":150,
})
def finish(fig, name):
    fig.savefig(f"{OUT_PDF}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT_PNG}/{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("  saved", name)

# =========================================================================== #
# FIGURE 1 -- main result: MAPE ladder + gain-vs-population with bootstrap CIs
# =========================================================================== #
def fig_mape_ladder():
    models = ["weather-blind\nXGB", "+ population\nDL curve", "+ personalized\nDL score",
              "shuffled\ncontrol"]
    mape   = [7.3381, 7.2936, 7.1304, 7.2996]
    cols   = [GREY, BLUE, GREEN, VERM]
    # gain vs the reference each is compared against (points + 95% runner bootstrap)
    gains  = [("population DL\nvs weather-blind", 0.0445, -0.0092, 0.1011, BLUE),
              ("personalized DL\nvs population",  0.1632,  0.0390, 0.3075, GREEN),
              ("shuffled control\nvs population", -0.0060, -0.0514, 0.0339, VERM)]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 3.8),
                                   gridspec_kw={"width_ratios":[1.05,1]})
    # A: robust MAPE by model
    y = np.arange(len(models))[::-1]
    axA.barh(y, mape, color=cols, height=0.62)
    axA.set_yticks(y); axA.set_yticklabels(models, fontsize=8.6)
    axA.set_xlim(7.0, 7.40); axA.set_xlabel("robust MAPE (%)")
    axA.set_title("A. Held-out error by model", loc="left", fontsize=10.5)
    axA.grid(axis="y", visible=False)
    for yi, m in zip(y, mape):
        axA.annotate(f"{m:.3f}", (m, yi), xytext=(-4,0), textcoords="offset points",
                     va="center", ha="right", color="white", fontsize=8.4, fontweight="bold")
    # B: gain vs population with bootstrap CI
    yb = np.arange(len(gains))[::-1]
    axB.axvline(0, color=GREY, lw=1, ls=(0,(4,3)))
    for yi,(lab,g,lo,hi,c) in zip(yb, gains):
        axB.hlines(yi, lo, hi, color=c, lw=2.4)
        axB.plot(g, yi, "o", color=c, ms=8, markeredgecolor="white", markeredgewidth=0.8)
    axB.set_yticks(yb); axB.set_yticklabels([g[0] for g in gains], fontsize=8.4)
    axB.set_xlabel("MAPE improvement (points, 95% runner bootstrap)")
    axB.set_title("B. Incremental gain", loc="left", fontsize=10.5)
    axB.grid(axis="y", visible=False)
    fig.tight_layout(); finish(fig, "fig_mape_ladder")

# =========================================================================== #
# FIGURE 2 -- error by WBGT bin (MAPE and signed bias), three models
# =========================================================================== #
def fig_error_by_temp():
    # (wbgt_mid, mape_pre, mape_pop, mape_strict, bias_pre, bias_pop, bias_strict)
    B = np.array([
        [ 0.82, 7.312,7.181,6.962, -2.061,-1.660,-1.205],
        [ 7.81, 7.408,7.316,7.106, -2.295,-1.924,-1.471],
        [12.57, 7.367,7.329,7.171, -2.093,-1.666,-1.242],
        [16.49, 7.412,7.399,7.198, -1.467,-1.404,-0.964],
        [18.99, 7.153,7.162,7.067, -1.013,-1.238,-0.709],
        [21.01, 7.096,7.113,6.942, -0.473,-1.537,-0.962],
        [23.37, 7.489,7.440,7.342,  0.755,-1.336,-0.715],
    ])
    x = B[:,0]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for j,(c,lab) in enumerate([(GREY,"weather-blind"),(BLUE,"+ population DL"),
                                (GREEN,"+ personalized DL")]):
        axA.plot(x, B[:,1+j], "o-", color=c, lw=1.6, ms=5, label=lab)
        axB.plot(x, B[:,4+j], "o-", color=c, lw=1.6, ms=5, label=lab)
    axA.set_xlabel("WBGT (°C)"); axA.set_ylabel("robust MAPE (%)")
    axA.set_title("A. Error by WBGT", loc="left", fontsize=10.5); axA.legend(loc="upper left")
    axB.axhline(0, color=GREY, lw=0.8)
    axB.set_xlabel("WBGT (°C)"); axB.set_ylabel("signed bias (%),  + = predicted too fast")
    axB.set_title("B. Bias by WBGT", loc="left", fontsize=10.5)
    fig.tight_layout(); finish(fig, "fig_error_by_temp")

# =========================================================================== #
# FIGURE 4 -- affine calibration: population scale (alpha) vs personalized (gamma)
# =========================================================================== #
def fig_gamma():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.2, 3.2),
                                   gridspec_kw={"width_ratios":[1,1.35]})
    # A: population curve scale alpha, reference = 1 (compatible)
    axA.axvline(1, color=GREY, lw=1, ls=(0,(4,3)))
    axA.hlines(0, 0.823, 1.344, color=BLUE, lw=2.6)
    axA.plot(1.078, 0, "o", color=BLUE, ms=9, markeredgecolor="white", markeredgewidth=0.8)
    axA.set_yticks([0]); axA.set_yticklabels(["population\ncurve scale α"], fontsize=8.8)
    axA.set_xlim(0.4, 1.7); axA.set_xlabel("calibrated scale (1 = compatible)")
    axA.set_title("A. Population scale", loc="left", fontsize=10.5)
    axA.grid(axis="y", visible=False)
    # B: personalized gamma by WBGT subset, reference = 0
    subs = [("all eligible",-1.654,-2.545,-0.865),
            ("WBGT > 10°C", -1.627,-2.528,-0.792),
            ("WBGT > 15°C", -1.452,-2.320,-0.602),
            ("WBGT > 20°C", -0.961,-1.827,-0.067)]
    y = np.arange(len(subs))[::-1]
    axB.axvline(0, color=GREY, lw=1, ls=(0,(4,3)))
    for yi,(lab,g,lo,hi) in zip(y, subs):
        axB.hlines(yi, lo, hi, color=VERM, lw=2.4)
        axB.plot(g, yi, "o", color=VERM, ms=8, markeredgecolor="white", markeredgewidth=0.8)
    axB.set_yticks(y); axB.set_yticklabels([s[0] for s in subs], fontsize=8.8)
    axB.set_xlabel("personalized residual coefficient γ  (0 = no effect)")
    axB.set_title("B. Personalized deviation (reversed)", loc="left", fontsize=10.5)
    axB.grid(axis="y", visible=False)
    fig.tight_layout(); finish(fig, "fig_gamma")

if __name__ == "__main__":
    print("building 09/10 paper figures ...")
    fig_mape_ladder()
    fig_error_by_temp()
    fig_gamma()
    print("done ->", OUT_PDF)
