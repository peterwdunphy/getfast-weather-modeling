"""build_lit_comparison.py

Literature-comparison figure for the manuscript:
  1. fig_heat_effects_lit  -- our estimated heat-slowdown curve against the published
     functional forms and slopes. Marker shape encodes the study population:
     DIAMOND = elite field, SQUARE = amateur / mass field.

All heat-effect curves are expressed as percent pace slowdown versus a 10 C anchor on a
common temperature axis. Slopes reported in seconds per degree are converted with a
representative finish time (elite 130 min, amateur 240 min); studies native to air
temperature rather than WBGT are plotted on the same axis, so placement above ~18 C is
approximate. These conversions are stated in the manuscript caption.

Run: /home/bb/test_env/bin/python3 build_lit_comparison.py
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

INK, MUTE, GRID = "#1a1a1a", "#5c5b57", "#e6e5df"
BLUE, VERM, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#8a897f"
SKY = "#56B4E9"   # Okabe-Ito sky blue: core temperature, paired with HR
ELITE_C, AMATEUR_C, OURS_C = "#2c6fb0", "#d1662b", "#111111"   # elite / amateur / ours
plt.rcParams.update({
    "figure.facecolor":"white","axes.facecolor":"white",
    "font.size":10,"axes.titlesize":11,"axes.labelsize":10,
    "axes.edgecolor":MUTE,"axes.linewidth":0.8,
    "xtick.color":MUTE,"ytick.color":MUTE,"text.color":INK,
    "axes.labelcolor":INK,"axes.titlecolor":INK,
    "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.7,
    "axes.spines.top":False,"axes.spines.right":False,
    "legend.frameon":False,"legend.fontsize":8.4,"figure.dpi":150,
})
def finish(fig,name):
    fig.savefig(f"{OUT_PDF}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT_PNG}/{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("  saved",name)

# --- representative finish times for s/degC -> %/degC conversion ------------
ELITE_S, AMATEUR_S = 130*60.0, 240*60.0   # 2:10 and 4:00 marathons, in seconds

# =========================================================================== #
# FIGURE 1 -- heat-slowdown: our curve vs the published forms
# =========================================================================== #
def fig_heat_effects_lit():
    a = np.load("/weather/data/heat_percentile_averaged.npz", allow_pickle=True)
    w = a["wsweep"].astype(float); pc = a["pctl_curves"].astype(float)
    dl = pc[49] - np.interp(10, w, pc[49])                 # our DL population median, anchored at 10C
    T = np.linspace(10, 30, 200)
    lin = lambda slope, bp: slope*np.maximum(0.0, T-bp)    # linear form: %/degC above a breakpoint

    # Martin 1999 quadratic (elite men, NYC), T in degF, expressed as % vs the 10C anchor
    tf = T*9/5+32; wt = 148.51 - 0.713*tf + 0.00657*tf**2
    martin = wt/np.interp(50.0, tf, wt)*100 - 100

    # (label, curve, %/degC note).  Slopes: %/degC directly, or s/degC / finish time.
    elite = [
        ("Martin 1999",            martin),
        ("Mantzios 2022",          lin(0.20, 15)),                      # elite marathon, WBGT
        ("Ely 2007 (25th)",  lin(0.22, 10)),                      # 1.1%/5C WBGT
        ("Guy 2014",               np.minimum(lin(0.18, 8), 0.18*17)),  # 3.1% by >25C, capped
        ("Nikolaidis 2019 (top-10)", lin(38.0/ELITE_S*100, 10)),       # 38 s/C air temp
        ("Knechtle 2019 (win.)",  lin(20.0/ELITE_S*100, 10)),       # 20 s/C air temp
    ]
    amateur = [
        ("Vernon 2021 (London mass)", lin(0.56, 12)),                  # 2.8%/5C above 12C
        ("Ely 2007 (300th)",    lin(0.64, 10)),                  # 3.2%/5C WBGT
        ("Nikolaidis 2019 (all)",     lin(113.0/AMATEUR_S*100, 10)),   # 1:53/C air temp
        ("Knechtle 2019 (all)",       lin(107.0/AMATEUR_S*100, 10)),   # 1:47/C air temp
        ("Knechtle 2021b (masters)",  lin(0.28, 10)),                  # ~8 min hottest vs coldest tertile
    ]
    mark_T = np.array([15, 20, 25, 30])                                # where population markers sit

    fig, ax = plt.subplots(figsize=(9.4, 6.4))
    ends = []   # (y_at_right_edge, label, color, weight) for staggered right-edge labels
    def draw(group, color, shape):
        for lab, y in group:
            ax.plot(T, y, color=color, lw=1.1, alpha=0.5, zorder=2)
            ax.plot(mark_T, np.interp(mark_T, T, y), linestyle="none", marker=shape,
                    ms=6, mfc=color, mec="white", mew=0.5, alpha=0.9, zorder=3)
            ends.append([y[-1], lab, color, "normal"])
    draw(elite, ELITE_C, "D")           # diamonds = elite
    draw(amateur, AMATEUR_C, "s")       # squares  = amateur
    # our curve (bold, distinct)
    ax.plot(w, dl, color=OURS_C, lw=3.2, zorder=5)
    ax.plot(w[-1], dl[-1], "o", color=OURS_C, ms=7, mec="white", mew=0.8, zorder=6)
    ends.append([dl[-1], "This study", OURS_C, "bold"])

    # stagger the right-edge labels so they do not overlap (repel upward, min gap)
    ends.sort(key=lambda e: e[0])
    gap, ypos = 0.62, None
    for e in ends:
        ypos = e[0] if ypos is None else max(e[0], ypos + gap)
        e.append(ypos)
    for y_end, lab, color, weight, ylab in ends:
        ax.plot([30, 30.4], [y_end, ylab], color=color, lw=0.5, alpha=0.6, clip_on=False)
        ax.annotate(lab, (30.4, ylab), xytext=(2, 0), textcoords="offset points",
                    va="center", fontsize=7.3, color=color, fontweight=weight, annotation_clip=False)

    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xlim(10, 30); ax.set_ylim(-0.5, 16.5)
    ax.set_xlabel("temperature during the run (°C, WBGT or air per study)")
    ax.set_ylabel("pace slowdown vs 10°C (%)")
    ax.set_title("Marathon heat-slowdown: this study versus published forms", loc="left")
    handles = [Line2D([0],[0], color=ELITE_C, lw=1.6, marker="D", mfc=ELITE_C, mec="white", ms=7, label="Elite field"),
               Line2D([0],[0], color=AMATEUR_C, lw=1.6, marker="s", mfc=AMATEUR_C, mec="white", ms=7, label="Amateur / mass field"),
               Line2D([0],[0], color=OURS_C, lw=3.0, label="This study (deep-model median)")]
    ax.legend(handles=handles, loc="upper left")
    fig.subplots_adjust(right=0.75)
    finish(fig, "fig_heat_effects_lit")

if __name__ == "__main__":
    print("building literature-comparison figure ...")
    fig_heat_effects_lit()
    print("done ->", OUT_PDF)
