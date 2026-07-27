"""build_paper_figures.py

Purpose-built figure suite for the modeling-and-results paper. Every figure is
drawn fresh from the result artifacts in one consistent, print-oriented style so
the paper reads as one system. Design rules (from the project conventions):
  - Titles state WHAT is plotted; interpretation lives in the LaTeX caption/prose.
  - No on-figure editorial claims (no "over-correction gone", etc.).
  - Colorblind-safe Okabe-Ito palette, assigned in a fixed order, never cycled.
  - Recessive grid/axes, thin marks, confidence bands where the estimate has one.
Outputs vector PDFs into figures/ (used by LaTeX) and PNGs into a scratch dir
(for visual review only; not committed).

Run: /home/bb/test_env/bin/python3 build_paper_figures.py
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# --------------------------------------------------------------------------- #
# Shared style
# --------------------------------------------------------------------------- #
OUT_PDF = "/home/bb/weather/getfast-weather-modeling/figures"
OUT_PNG = "/tmp/claude-1000/-home-bb-weather/599857c7-9ec0-49e5-8a67-0c0c8644ad80/scratchpad/paperfigs"
os.makedirs(OUT_PNG, exist_ok=True)
DATA = "/weather/data"
RES  = "/weather/results"

# Okabe-Ito: colorblind-safe by construction. Fixed roles, never cycled.
INK   = "#1a1a1a"      # primary text
MUTE  = "#5c5b57"      # secondary text
GRID  = "#e6e5df"      # recessive grid
BLUE  = "#0072B2"      # GetFast / primary estimate
VERM  = "#D55E00"      # contrast / "before" / coach
GREEN = "#009E73"      # third series / "after-personalized"
ORANGE= "#E69F00"      # literature / fourth
PURPLE= "#CC79A7"      # literature / fifth
GREY  = "#8a897f"      # reference / neutral line

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.edgecolor": MUTE, "axes.linewidth": 0.8,
    "xtick.color": MUTE, "ytick.color": MUTE, "text.color": INK,
    "axes.labelcolor": INK, "axes.titlecolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8.6,
    "figure.dpi": 150,
})

def finish(fig, name):
    """Save vector PDF (for LaTeX) + PNG (for review)."""
    fig.savefig(f"{OUT_PDF}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT_PNG}/{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")

def wins(a, lo=1, hi=99):
    """Winsorize an array to its [lo, hi] percentiles (outlier control)."""
    a = np.asarray(a, float)
    l, h = np.percentile(a, [lo, hi])
    return np.clip(a, l, h)

PACE0_SPK = 347.78     # runner-average pace in s/km (for s/km -> % conversion)

# =========================================================================== #
# FIGURE 1 -- within-runner pace and HR response to WBGT (two panels)
# =========================================================================== #
def fig_heat_response():
    m3 = json.load(open(f"{RES}/model3_results.json"))
    # Panel A: empirical within-runner pace deviation vs WBGT (the U-curve).
    b = [x for x in m3["binned_response"] if x["wbgt_mid"] >= 0]        # drop the <0 outlier bin
    xw  = np.array([x["wbgt_mid"] for x in b])
    pct = np.array([x["spk_vs_norm"] for x in b]) / PACE0_SPK * 100     # s/km deviation -> %
    nb  = np.array([x["n"] for x in b], float)
    pct = pct - pct.min()                                              # anchor the U at its optimum

    # Panel B: within-runner HR deviation vs WBGT, computed from source.
    am = pd.read_parquet(f"{DATA}/activities_model_clean.parquet",
                         columns=["user_id","wbgt","avg_hr","pace_min_km","distance_km"])
    am = am.dropna(subset=["wbgt","avg_hr","pace_min_km"])
    am = am[(am.avg_hr>60)&(am.avg_hr<210)&(am.wbgt>-10)&(am.wbgt<40)]
    # subtract each runner's own means (runner fixed effects) so we see within-runner movement
    am["hr_w"]   = am.avg_hr  - am.groupby("user_id").avg_hr.transform("mean")
    am["wbgt_r"] = am.wbgt    - am.groupby("user_id").wbgt.transform("mean")
    am["wbin"]   = (am.wbgt/2).round()*2                               # 2 C bins on absolute WBGT
    g = am.groupby("wbin").agg(hr=("hr_w","mean"), n=("hr_w","size")).reset_index()
    g = g[(g.wbin>=4)&(g.wbin<=32)&(g.n>=300)]
    g["hr"] = g.hr - g.hr.min()                                        # anchor to the coolest bin

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.4, 3.7))
    # A
    axA.plot(xw, pct, "-", color=BLUE, lw=1.4, zorder=2)
    axA.scatter(xw, pct, s=np.sqrt(nb)/2.0, color=BLUE, edgecolor="white",
                linewidth=0.6, zorder=3)
    axA.axvline(11, color=GREY, lw=1, ls=(0,(4,3)))
    axA.annotate("optimum ~11°C", (11, axA.get_ylim()[1]), xytext=(2,-2),
                 textcoords="offset points", fontsize=7.6, color=MUTE, va="top")
    axA.set_xlabel("WBGT (°C)"); axA.set_ylabel("pace slowdown vs optimum (%)")
    axA.set_title("A. Pace response (within-runner)", loc="left", fontsize=10.5)
    axA.set_xlim(0, 40)
    # B
    axB.plot(g.wbin, g.hr, "-", color=VERM, lw=1.4, zorder=2)
    axB.scatter(g.wbin, g.hr, s=22, color=VERM, edgecolor="white", linewidth=0.6, zorder=3)
    axB.set_xlabel("WBGT (°C)"); axB.set_ylabel("heart-rate rise vs coolest (bpm)")
    axB.set_title("B. Heart-rate response (within-runner)", loc="left", fontsize=10.5)
    axB.set_xlim(0, 34)
    fig.tight_layout()
    finish(fig, "fig_heat_response")

# =========================================================================== #
# FIGURE 2 -- GetFast population g and published marathon fits (6 curves)
# =========================================================================== #
def fig_g_vs_literature():
    W = np.linspace(10, 30, 400)
    def anchor(y): return y - np.interp(10, W, y)
    # GetFast shipped two-hinge population g: flat<15, mild arm to 24, steeper above 24
    def gg(w):
        return np.where(w<15, 0.0,
               np.where(w<24, 0.2222*(w-15), 2.0 + 0.65*(w-24)))
    getfast = gg(W)
    # coach benchmark table (deg F -> C), anchored at 10 C
    cf = np.arange(50.,96.); cp = np.array([0,.1,.1,.2,.3,.4,.4,.5,.6,.7,.7,.9,1,1.2,1.3,1.5,
        1.6,1.8,1.9,2.1,2.2,2.5,2.7,3,3.3,3.5,3.8,4,4.3,4.6,4.8,5.2,5.5,5.8,6.2,6.5,6.8,7.2,
        7.5,7.8,8.2,8.6,9.1,9.5,9.9,10.4])
    coach = anchor(np.interp(W, (cf-32)*5/9, cp))
    ely   = anchor(0.64*np.maximum(0, W-10))
    vernon= anchor(0.56*np.maximum(0, W-12))
    mantz = anchor(0.20*np.maximum(0, W-15))
    tf = W*9/5+32; wt = 148.51-0.713*tf+0.00657*tf**2
    martin= anchor(wt/wt[np.argmin(np.abs(W-10))]*100-100)

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    series = [(getfast,"GetFast population g", BLUE, 3.0, "-"),
              (coach,  "Coach benchmark",      VERM, 2.2, (0,(5,2.5))),
              (ely,    "Ely 2007 (mass)",      ORANGE,1.8,"-"),
              (vernon, "Vernon 2021 (London mass)", GREEN,1.8,"-"),
              (mantz,  "Mantzios 2022 (elite)",GREY, 1.8,"-"),
              (martin, "Martin 1999 (quadratic)",PURPLE,1.8,(0,(1,1.4)))]
    for y,lab,c,lw,ls in series:
        ax.plot(W, y, color=c, lw=lw, ls=ls, solid_capstyle="round")
        ax.annotate(lab, (30, y[-1]), xytext=(6,0), textcoords="offset points",
                    va="center", fontsize=8.4, color=c,
                    fontweight="bold" if "GetFast" in lab else "normal")
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xlim(10, 30); ax.set_ylim(-0.4, max(ely[-1], coach[-1])+0.6)
    ax.set_xlabel("WBGT (°C)"); ax.set_ylabel("pace slowdown vs 10°C (%)")
    ax.set_title("Population heat curve versus published marathon fits", loc="left")
    ax.xaxis.set_major_locator(MultipleLocator(5))
    fig.subplots_adjust(right=0.74)
    finish(fig, "fig_g_vs_literature")

# =========================================================================== #
# FIGURE 3 -- heat penalty by race distance (Model 4 hierarchical posterior)
# =========================================================================== #
def fig_distance_scaling():
    m4 = json.load(open(f"{RES}/model4_hierarchical.json"))
    order = ["10k","half","marathon"]; labels=["10 km","half","marathon"]
    est = [m4["slope_above27_at"][k][0] for k in order]
    lo  = [m4["slope_above27_at"][k][1] for k in order]
    hi  = [m4["slope_above27_at"][k][2] for k in order]
    y = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.hlines(y, lo, hi, color=BLUE, lw=2.4)
    ax.plot(est, y, "o", color=BLUE, ms=8, markeredgecolor="white", markeredgewidth=0.8)
    for yi, e, h in zip(y, est, hi):
        ax.annotate(f"{e:.1f}", (h, yi), xytext=(7,0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("hot-end slope above WBGT 27 (s/km per °C, 94% interval)")
    ax.set_ylabel("")
    ax.set_title("Hot-end heat slope by race distance (hierarchical posterior)", loc="left")
    ax.set_xlim(0, max(hi)+1.3); ax.grid(axis="y", visible=False)
    mp = m4["marathon_wbgt30_penalty"]
    ax.annotate(f"marathon at WBGT 30 vs 21:  +{mp['pct'][0]:.1f}%  "
                f"(+{mp['min_at_3h30'][0]:.0f} min on 3:30)",
                (0, -0.75), xycoords=("data","axes fraction"), fontsize=8.4, color=MUTE)
    fig.tight_layout()
    finish(fig, "fig_distance_scaling")

# =========================================================================== #
# FIGURE 4 -- predicted-pace bias by WBGT and by tolerance tier (validation)
# =========================================================================== #
def fig_validation():
    df = pd.read_parquet(f"{DATA}/oof_v5_preds.parquet")
    def bias(pred):                                    # + = predicted too fast
        return wins((df.pace_min_km - df[pred])/df.pace_min_km*100)
    df["b0"]  = bias("y_hat0")      # weather-blind
    df["bp"]  = bias("pred_pop")    # + population g
    df["bh"]  = bias("pred_hr")     # + HR-personalized
    # Panel A: bias vs WBGT, weather-blind vs population
    df["wbin"] = pd.cut(df.wbgt, bins=[-5,10,15,18,21,24,27,40])
    ga = df.groupby("wbin", observed=True).agg(w=("wbgt","mean"),
            b0=("b0","mean"), bp=("bp","mean")).dropna()
    # Panel B: bias by tolerance tier, on HOT runs only (the tier scaling acts only in heat)
    tiers=["tolerant","mid","intolerant"]
    hot = df[df.wbgt > 20]
    gb = hot.groupby("tier", observed=True)[["b0","bp","bh"]].mean().reindex(tiers)
    gap0 = gb.loc["intolerant","b0"] - gb.loc["tolerant","b0"]
    gaph = gb.loc["intolerant","bh"] - gb.loc["tolerant","bh"]
    print(f"    [validation] hot-run tier gap (intolerant-tolerant): weather-blind {gap0:+.2f} "
          f"-> +HR-personalized {gaph:+.2f}  ({100*(1-gaph/gap0):.0f}% closed)")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    axA.axhline(0, color=GREY, lw=0.8)
    axA.plot(ga.w, ga.b0, "o-", color=VERM, lw=1.6, ms=6, label="weather-blind")
    axA.plot(ga.w, ga.bp, "o-", color=BLUE, lw=1.6, ms=6, label="+ population g")
    axA.set_xlabel("WBGT (°C)"); axA.set_ylabel("signed bias (%),  + = predicted too fast")
    axA.set_title("A. Bias by WBGT", loc="left", fontsize=10.5); axA.legend(loc="lower left")
    # B: grouped bars
    x = np.arange(len(tiers)); w = 0.26
    axB.axhline(0, color=GREY, lw=0.8)
    axB.bar(x-w, gb.b0, w, color=VERM,  label="weather-blind")
    axB.bar(x,   gb.bp, w, color=BLUE,  label="+ population g")
    axB.bar(x+w, gb.bh, w, color=GREEN, label="+ HR-personalized")
    axB.set_xticks(x); axB.set_xticklabels(tiers)
    axB.set_ylabel("hot-run signed bias (%)"); axB.set_xlabel("heat-tolerance tier")
    axB.set_title("B. Bias by tolerance tier", loc="left", fontsize=10.5)
    axB.legend(loc="upper center", ncol=1); axB.grid(axis="x", visible=False)
    fig.tight_layout()
    finish(fig, "fig_validation")

    # print MAPE (flat) for the caption
    def mape(p): return wins(np.abs(df.pace_min_km-df[p])/df.pace_min_km*100, hi=99).mean()
    print(f"    [validation] MAPE weather-blind {mape('y_hat0'):.2f} | "
          f"+pop {mape('pred_pop'):.2f} | +hr {mape('pred_hr'):.2f}")

# =========================================================================== #
# FIGURE 5 -- deep-model per-runner heat response (population + spread)
# =========================================================================== #
def fig_dl_response():
    a = np.load(f"{DATA}/heat_percentile_averaged.npz", allow_pickle=True)
    w = a["wsweep"].astype(float); pc = a["pctl_curves"].astype(float)
    def anc(row): return row - np.interp(10, w, row)
    med = anc(pc[49]); p10 = anc(pc[9]); p90 = anc(pc[89])
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.fill_between(w, p10, p90, color=BLUE, alpha=0.15, lw=0, label="runner p10–p90")
    ax.plot(w, med, color=BLUE, lw=2.6, label="population median")
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xlim(10, 28); ax.set_xlabel("WBGT (°C)")
    ax.set_ylabel("pace slowdown vs 10°C (%)")
    ax.set_title("Deep-model heat response and between-runner spread", loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    finish(fig, "fig_dl_response")

# =========================================================================== #
# FIGURE 6 -- pace response to training-climate deviation (acclimation)
# =========================================================================== #
def fig_acclimation():
    am = pd.read_parquet(f"{DATA}/activities_model_clean.parquet",
                         columns=["user_id","start_dt","wbgt","pace_min_km",
                                  "fit_pace_30d","distance_km"])
    am = am.dropna(subset=["wbgt","pace_min_km","fit_pace_30d","start_dt"])
    am = am[(am.pace_min_km>2.5)&(am.pace_min_km<11.5)].sort_values(["user_id","start_dt"])
    # recent training climate = trailing 30-day EWMA of a runner's own run WBGT (leakage-free: shifted)
    am["recent"] = (am.groupby("user_id", group_keys=False)
                      .apply(lambda d: d.wbgt.shift(1).ewm(halflife=30, min_periods=3).mean()))
    am = am.dropna(subset=["recent"])
    am["dev"] = am.wbgt - am.recent                       # + = racing hotter than trained
    # within-runner log-pace residual (remove runner mean + fitness), then bin by deviation
    am["lp"] = np.log(am.pace_min_km)
    am["lp_w"] = am.lp - am.groupby("user_id").lp.transform("mean")
    am["fit_w"] = am.fit_pace_30d - am.groupby("user_id").fit_pace_30d.transform("mean")
    # crude fitness partial-out via a single slope, enough for a descriptive figure
    slope = np.polyfit(am.fit_w, am.lp_w, 1)[0]
    am["resid_pct"] = (am.lp_w - slope*am.fit_w) * 100    # log-pace residual in %
    am["dbin"] = (am.dev/2).round()*2
    g = am.groupby("dbin").agg(r=("resid_pct","mean"), n=("resid_pct","size")).reset_index()
    g = g[(g.dbin>=-12)&(g.dbin<=14)&(g.n>=200)].reset_index(drop=True)
    g["r"] = g.r - g.r.iloc[int(g.dbin.abs().values.argmin())]        # anchor at deviation ~ 0

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axvline(0, color=GREY, lw=1, ls=(0,(4,3)))
    ax.plot(g.dbin, g.r, "-", color=BLUE, lw=1.4, zorder=2)
    ax.scatter(g.dbin, g.r, s=np.sqrt(g.n)/1.4, color=BLUE, edgecolor="white",
               linewidth=0.6, zorder=3)
    ax.set_xlabel("race-day WBGT minus recent training climate (°C)")
    ax.set_ylabel("within-runner pace residual (%)")
    ax.set_title("Within-runner pace residual by training-climate deviation", loc="left")
    ax.annotate("trained hotter\n(cooler race)", (-11, ax.get_ylim()[1]),
                fontsize=7.6, color=MUTE, va="top")
    ax.annotate("trained cooler\n(hotter race)", (13, ax.get_ylim()[1]),
                fontsize=7.6, color=MUTE, va="top", ha="right")
    fig.tight_layout()
    finish(fig, "fig_acclimation")

# =========================================================================== #
# FIGURE 7 -- between-runner heat tolerance: distribution + test-retest
# =========================================================================== #
def fig_tolerance():
    am = pd.read_parquet(f"{DATA}/activities_model_clean.parquet",
                         columns=["user_id","wbgt","avg_hr","pace_min_km","distance_km"])
    am = am.dropna(subset=["wbgt","avg_hr","pace_min_km","distance_km"])
    am = am[(am.avg_hr>60)&(am.avg_hr<210)&(am.wbgt>-10)&(am.wbgt<40)]
    # per-runner HR-heat slope = coef of wbgt in avg_hr ~ wbgt + pace + distance (matched load)
    def hr_slope(d):
        if len(d) < 25 or d.wbgt.std() < 2: return np.nan
        X = np.column_stack([np.ones(len(d)), d.wbgt, d.pace_min_km, d.distance_km])
        try: return np.linalg.lstsq(X, d.avg_hr.values, rcond=None)[0][1]
        except Exception: return np.nan
    # Split-half reliability uses an INTERLEAVED (odd/even) split, not first/second half,
    # so the two halves cover the same seasons and fitness: this isolates measurement
    # noise in the estimate rather than genuine drift over a runner's history.
    slopes, sA, sB = [], [], []
    for _, d in am.groupby("user_id"):
        if len(d) < 40 or d.wbgt.std() < 2: continue           # need enough runs per half
        d = d.reset_index(drop=True)
        s = hr_slope(d)
        if np.isnan(s): continue
        sa, sb = hr_slope(d.iloc[1::2]), hr_slope(d.iloc[0::2])
        if np.isnan(sa) or np.isnan(sb): continue
        slopes.append(s); sA.append(sa); sB.append(sb)
    slopes = np.array(slopes)
    ab = pd.DataFrame({"a":sA,"b":sB}).dropna()
    r = ab.a.corr(ab.b)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.4, 3.8))
    axA.hist(wins(slopes, 1, 99), bins=40, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.4)
    axA.axvline(np.median(slopes), color=INK, lw=1.2, ls=(0,(4,3)))
    axA.set_xlabel("per-runner HR–heat slope (bpm per °C)"); axA.set_ylabel("runners")
    axA.set_title(f"A. Distribution (n={len(slopes):,} runners)", loc="left", fontsize=10.5)
    axB.scatter(wins(ab.a,1,99), wins(ab.b,1,99), s=6, color=BLUE, alpha=0.25, edgecolor="none")
    lim = np.percentile(np.r_[wins(ab.a,1,99), wins(ab.b,1,99)], [1,99])
    axB.plot(lim, lim, color=GREY, lw=1, ls=(0,(4,3)))
    axB.set_xlabel("slope, odd runs (bpm/°C)"); axB.set_ylabel("slope, even runs (bpm/°C)")
    axB.set_title(f"B. Split-half reliability  (r = {r:.2f})", loc="left", fontsize=10.5)
    axB.set_xlim(*lim); axB.set_ylim(*lim)
    fig.tight_layout()
    finish(fig, "fig_tolerance")
    print(f"    [tolerance] n_runners={len(slopes)} split-half r={r:.3f} "
          f"median slope={np.median(slopes):.3f} bpm/C")

if __name__ == "__main__":
    print("building paper figures ...")
    fig_heat_response()
    fig_g_vs_literature()
    fig_distance_scaling()
    fig_validation()
    fig_dl_response()
    fig_acclimation()
    fig_tolerance()
    print("done. PDFs ->", OUT_PDF)
