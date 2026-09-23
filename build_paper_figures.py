"""build_paper_figures.py

Figure suite for the modeling-and-results paper. The Results report the analysis
in notebooks 09/10 (population and personalized deep-learning heat features added
to a weather-blind XGBoost pace model). Values are the authoritative held-out
numbers from notebook 10 rerun on v3 with the 5M-run checkpoint (notebook 19b:
24,311 activities, 154 runners, 3 XGB seeds, holdout ceiling 28 C, robust
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
    mape   = [7.5574, 7.4976, 7.3293, 7.4841]      # v3 / 5M checkpoint (nb 19b)
    cols   = [GREY, BLUE, GREEN, VERM]
    # gain vs the reference each is compared against (points + 95% runner bootstrap)
    gains  = [("population DL\nvs weather-blind", 0.0598,  0.0027, 0.1183, BLUE),
              ("personalized DL\nvs population",  0.1683,  0.0620, 0.2826, GREEN),
              ("shuffled control\nvs population",  0.0135, -0.0190, 0.0425, VERM)]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 3.8),
                                   gridspec_kw={"width_ratios":[1.05,1]})
    # A: robust MAPE by model
    y = np.arange(len(models))[::-1]
    axA.barh(y, mape, color=cols, height=0.62)
    axA.set_yticks(y); axA.set_yticklabels(models, fontsize=8.6)
    axA.set_xlim(7.2, 7.60); axA.set_xlabel("robust MAPE (%)")
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
    # v3 / 5M checkpoint, notebook 19b cell "Error by temperature" (bins up to the 28 C ceiling)
    B = np.array([
        [ 0.13, 7.189,7.089,6.927, -2.109,-1.559,-0.979],
        [ 7.79, 7.834,7.697,7.464, -2.646,-2.073,-1.612],
        [12.59, 7.720,7.639,7.452, -2.490,-1.920,-1.431],
        [16.51, 7.705,7.693,7.529, -1.968,-1.687,-1.217],
        [18.99, 7.607,7.622,7.476, -1.580,-1.605,-1.168],
        [21.01, 7.188,7.223,7.024, -0.761,-1.438,-0.951],
        [23.44, 7.155,7.118,7.050,  0.360,-1.118,-0.651],
        [26.24, 7.911,7.747,7.636,  1.836,-1.076,-0.814],
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
    axA.hlines(0, 0.914, 1.303, color=BLUE, lw=2.6)
    axA.plot(1.098, 0, "o", color=BLUE, ms=9, markeredgecolor="white", markeredgewidth=0.8)
    axA.set_yticks([0]); axA.set_yticklabels(["population\ncurve scale α"], fontsize=8.8)
    axA.set_xlim(0.4, 1.7); axA.set_xlabel("calibrated scale (1 = compatible)")
    axA.set_title("A. Population scale", loc="left", fontsize=10.5)
    axA.grid(axis="y", visible=False)
    # B: personalized gamma by WBGT subset, reference = 0
    subs = [("all eligible",-0.963,-1.712,-0.100),     # v3 / 5M checkpoint, nb 19b section 9
            ("WBGT > 10°C", -0.946,-1.719,-0.080),
            ("WBGT > 15°C", -0.906,-1.663, 0.003),
            ("WBGT > 20°C", -0.652,-1.463, 0.332)]
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
