"""build_compensability_figures.py

Illustrations for the compensability appendix (Appendix F, tommy_appendix.tex).
Five two-panel figures, in the order the appendix uses them:

  1. fig_heat_ceiling  -- the heat balance as a speed limit. Panel (a) splits the
     dissipation ceiling into its dry and evaporative parts against wet-bulb
     temperature and draws heat production across it; panel (b) turns the
     resulting head into the running speed the environment permits, against the
     runner's own aerobic ceiling.

  2. fig_wetbulb_isopleth -- what a line of constant wet-bulb temperature is.
     Panel (a) draws the isopleths on the air-temperature/humidity plane with
     the tabulated one highlighted; panel (b) walks that line and shows the dry
     and evaporative routes trading off exactly while the strain rises anyway.

  3. fig_two_links -- the chain the balance describes. Panel (a) is a schematic
     of the two links in series, muscle to skin by blood and skin to air by
     evaporation, and the boundary they share. Panel (b) shows the two demands
     against that boundary, moving in opposite directions.

  4. fig_duration_bank -- why duration matters. Panel (a) draws core temperature
     against distance covered for a runner holding pace at several required
     wettednesses; panel (b) shows the storage allowance as a share of heat
     production against race distance, which crosses unity at a fixed distance
     independent of both body mass and pace.

  5. fig_drybulb_axis -- the response on the dry-bulb axis at fixed humidity.
     Panel (a) is required wettedness, linear in dry air and increasingly convex
     as humidity rises; panel (b) re-expresses the learned WBGT curve as a
     function of air temperature at four humidities.

Every physical constant matches verify_heat_convexity.py, which is the
self-checking source of truth for the numbers quoted in the appendix text.

Design rules (project conventions): titles state WHAT is plotted, interpretation
lives in the LaTeX caption; one colorblind-safe palette (Okabe-Ito) in fixed
roles; recessive axes.

Run: python3 build_compensability_figures.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, "figures")
os.makedirs(OUT_PDF, exist_ok=True)

# Okabe-Ito, fixed roles across every figure in the paper.
INK, MUTE, GRID = "#1a1a1a", "#5c5b57", "#e6e5df"
BLUE, VERM, GREEN, ORANGE, PURPLE, GREY = ("#0072B2", "#D55E00", "#009E73",
                                           "#E69F00", "#CC79A7", "#8a897f")
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


def finish(fig, name):
    fig.savefig(f"{OUT_PDF}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  saved", name)


# ========================================================================== #
# Physics. Identical constants and forms to verify_heat_convexity.py.
# ========================================================================== #
T_SKIN = 31.0        # C, mean skin temperature in outdoor competition
A_BODY = 1.85        # m2
MASS = 70.0          # kg
ECON = 4184.0        # J/kg/km
EFF = 0.79           # fraction of metabolic turnover appearing as heat
V_REF = 3.33         # m/s, the reference runner's marathon speed
LR = 124.0 / 8.3     # 14.9 K/kPa, Taylor (2006)
LR_PSY = 15.04
KAPPA = EFF * ECON * MASS / 1000.0     # J of heat per metre travelled
C_BODY = 3474.0      # J/kg/K, Burton (1935)
DT_TOL = 2.5         # K, the core-temperature rise a runner will spend
T_CORE_0 = 37.0      # C, core temperature at the gun
BANK = MASS * C_BODY * DT_TOL          # J of storable heat
D_BANK = BANK / KAPPA                  # m; = C_BODY*DT_TOL/(EFF*ECON/1000)
RHO_CB = 3770.0      # J/L/K, volumetric heat capacity of blood
T_CORE = 39.0        # C, core temperature held during a hot marathon


def psat(T):
    """Buck (1981), kPa."""
    return 0.61121 * math.exp((18.678 - T / 234.5) * (T / (257.14 + T)))


def hc(v):
    return 8.3 * math.sqrt(v)


def he(v):
    return LR * hc(v)


def H_prod(v=V_REF, m=MASS):
    return EFF * ECON * m * (v / 1000.0)


def head_wb(Twb, T_sk=T_SKIN):
    """Thermal head Phi, in kelvin-equivalents, on the wet-bulb axis."""
    return (T_sk - Twb) + LR * (psat(T_sk) - psat(Twb))


def head_ta(Ta, rh, T_sk=T_SKIN):
    """Thermal head on the dry-bulb axis at fixed relative humidity."""
    return (T_sk - Ta) + LR * (psat(T_sk) - rh * psat(Ta))


def v_thermal(head):
    """Speed at which production equals capacity, given h ~ sqrt(v).

    kappa*v = A*8.3*sqrt(v)*Phi  =>  v = (8.3*A/kappa)^2 * Phi^2.
    """
    return (8.3 * A_BODY / KAPPA) ** 2 * max(head, 0.0) ** 2


def twb(Ta, rh, lewis=LR_PSY):
    lo, hi = -40.0, Ta
    for _ in range(200):
        m = (lo + hi) / 2
        if (m - Ta) + lewis * (psat(m) - rh * psat(Ta)) < 0:
            lo = m
        else:
            hi = m
    return lo


def wbgt(Ta, rh, solar, wind=1.0):
    tg = Ta + 1.6 * (solar / 1000.0) ** 0.6 * (2.0 / max(wind, 0.3)) ** 0.4 * 10
    return 0.7 * twb(Ta, rh) + 0.2 * tg + 0.1 * Ta


def wreq(Ta, rh, T_sk=T_SKIN, v=V_REF, solar=0.0):
    dry = hc(v) * (T_sk - Ta) * A_BODY - solar * A_BODY
    return (H_prod(v) - dry) / (he(v) * (psat(T_sk) - rh * psat(Ta)) * A_BODY)


def e_max(Ta, rh, T_sk=T_SKIN, v=V_REF):
    """Maximum evaporative capacity of the environment, W over the whole body."""
    return he(v) * (psat(T_sk) - rh * psat(Ta)) * A_BODY


def Ta_vapour_wall(rh, T_sk=T_SKIN):
    """Air temperature at which rh*Ps(Ta) reaches Ps(T_sk).

    Above it the skin-to-air vapour gradient reverses, E_max turns negative and
    w_req changes sign. Any search over air temperature has to stop below it or
    a bisection will walk straight through the singularity.
    """
    lo, hi = T_sk, 90.0
    for _ in range(200):
        m = (lo + hi) / 2
        if rh * psat(m) < psat(T_sk):
            lo = m
        else:
            hi = m
    return lo


def Ta_for_wreq(target, rh=0.60, solar_abs=50.0):
    """Air temperature at which the reference runner faces a given w_req."""
    lo, hi = 5.0, Ta_vapour_wall(rh) - 0.05
    for _ in range(200):
        m = (lo + hi) / 2
        if wreq(m, rh, solar=solar_abs) < target:
            lo = m
        else:
            hi = m
    return lo


def storage_rate(w, Ta, rh=0.60):
    """Heat stored per second once the skin is fully wet, W.

    Beyond w_req = 1 the skin cannot get wetter, so evaporation saturates at
    E_max and the excess is stored: S = E_req - E_max = E_max*(w_req - 1).
    """
    return max(w - 1.0, 0.0) * e_max(Ta, rh)


# The learned population curve, at the anchor points reported in Section 4.
LEARNED = {10: 0.0, 12: 0.05, 14: 0.15, 16: 0.30, 20: 1.20, 24: 2.40, 28: 5.02}
LEARNED_MAX = 28.0
SOLAR = 200.0        # W/m2, the convention used throughout the appendix
WIND = 1.0           # m/s


def learned_penalty(w):
    """Linear interpolation of the learned curve; None beyond its support."""
    xs = sorted(LEARNED)
    if w < xs[0]:
        return 0.0
    if w > LEARNED_MAX:
        return None
    for a, b in zip(xs, xs[1:]):
        if a <= w <= b:
            f = (w - a) / (b - a)
            return LEARNED[a] + f * (LEARNED[b] - LEARNED[a])
    return None


# ========================================================================== #
# FIGURE 1 -- the dissipation ceiling and the speed it permits
# ========================================================================== #
def fig_heat_ceiling():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.8, 3.9))

    tw = np.linspace(4, 31, 400)
    dry = np.array([hc(V_REF) * (T_SKIN - t) for t in tw])
    evap = np.array([he(V_REF) * (psat(T_SKIN) - psat(t)) for t in tw])
    total = dry + evap
    prod = H_prod() / A_BODY

    # ---- A: the ceiling, split into its two routes
    axA.fill_between(tw, 0, dry, color=BLUE, alpha=0.30, lw=0,
                     label="dry (convective)")
    axA.fill_between(tw, dry, total, color=GREEN, alpha=0.30, lw=0,
                     label="evaporative")
    axA.plot(tw, total, color=INK, lw=1.6, label=r"ceiling $Q_{\max}$")
    axA.axhline(prod, color=VERM, lw=1.5, ls="--",
                label="heat production at 5:00/km")

    # where the ceiling crosses production
    cross = tw[np.argmin(np.abs(total - prod))]
    axA.plot([cross], [prod], "o", color=VERM, ms=6, zorder=6)
    axA.annotate("balance impossible\nbeyond here", xy=(cross, prod),
                 xytext=(24.4, 690), fontsize=8.4, color=VERM, ha="center",
                 arrowprops=dict(arrowstyle="->", color=VERM, lw=0.9),
                 bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    axA.annotate(r"$Q_{\max}\!\to\!0$ where $T_{wb}\!=\!T_{sk}$",
                 xy=(30.8, 8), xytext=(17.0, 120), fontsize=8.4, color=MUTE,
                 arrowprops=dict(arrowstyle="->", color=MUTE, lw=0.9),
                 bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))

    axA.set_xlabel("Wet-bulb temperature (°C)")
    axA.set_ylabel("Heat flux (W m$^{-2}$)")
    axA.set_title("(a)  The dissipation ceiling falls, and falls faster", loc="left")
    axA.set_xlim(4, 31)
    axA.set_ylim(0, 1300)
    axA.legend(loc="upper right")

    # ---- B: the speed the ceiling permits
    axB.set_xlim(4, 31)
    axB.set_ylim(0, 4.3)

    v_th = np.array([v_thermal(head_wb(t)) for t in tw])
    v_sus = np.minimum(v_th, V_REF)
    t_bind = tw[np.where(v_th < V_REF)[0][0]]

    axB.fill_between(tw, 0, v_sus, color=ORANGE, alpha=0.16, lw=0)
    axB.plot(tw, v_th, color=MUTE, lw=1.2, ls=(0, (4, 3)),
             label="thermal ceiling alone")
    axB.plot(tw, v_sus, color=INK, lw=2.0,
             label=r"sustainable speed $=\min$(aerobic, thermal)")
    axB.axhline(V_REF, color=VERM, lw=1.3, ls="--",
                label="aerobic ceiling (3:31 marathon runner)")
    axB.axvline(t_bind, color=MUTE, lw=0.9, ls=":")
    axB.annotate(f"thermal ceiling takes over\nat $T_{{wb}}$ = {t_bind:.1f} °C",
                 xy=(t_bind, V_REF), xytext=(15.4, 2.10), fontsize=8.4,
                 color=INK, ha="center",
                 arrowprops=dict(arrowstyle="->", color=MUTE, lw=0.9),
                 bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    axB.text(17.5, 0.45, "flat, then a steep fall", fontsize=8.4, color=MUTE,
             ha="center")

    # pace equivalents on the right, after the limits are fixed
    axB2 = axB.twinx()
    axB2.set_ylim(axB.get_ylim())
    axB2.grid(False)
    ticks = [1000.0 / (p * 60.0) for p in (4, 5, 6, 8, 12)]
    axB2.set_yticks(ticks)
    axB2.set_yticklabels([f"{p}:00" for p in (4, 5, 6, 8, 12)], fontsize=8.4)
    axB2.set_ylabel("pace (min km$^{-1}$)", fontsize=9, color=MUTE)
    axB2.spines["top"].set_visible(False)

    axB.set_xlabel("Wet-bulb temperature (°C)")
    axB.set_ylabel("Sustainable speed (m s$^{-1}$)")
    axB.set_title("(b)  A concave speed limit, so a convex penalty", loc="left")
    axB.legend(loc="upper left")

    fig.tight_layout()
    finish(fig, "fig_heat_ceiling")


# ========================================================================== #
# FIGURE 2 -- what a wet-bulb isopleth is
# ========================================================================== #
def fig_wetbulb_isopleth():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.8, 3.9),
                                   gridspec_kw={"width_ratios": [1.06, 1]})

    # ---- A: isopleths on the air-temperature / humidity plane
    ta = np.linspace(10, 45, 220)
    rh = np.linspace(0.10, 1.0, 200)
    TA, RH = np.meshgrid(ta, rh)
    TW = np.vectorize(twb)(TA, RH)

    levels = [10, 14, 18, 26, 30]
    cs = axA.contour(TA, RH * 100, TW, levels=levels, colors=[MUTE], linewidths=0.9)

    def ta_on_isopleth(level, rh_pct):
        """Air temperature at which T_wb hits `level` for this humidity."""
        lo, hi = 0.0, 60.0
        for _ in range(120):
            m = (lo + hi) / 2
            if twb(m, rh_pct / 100.0) < level:
                lo = m
            else:
                hi = m
        return lo

    label_rh = {10: 62, 14: 62, 18: 62, 26: 72, 30: 82}
    axA.clabel(cs, fmt=lambda v: f"{v:.0f}°C", fontsize=8, inline=True,
               manual=[(ta_on_isopleth(L, label_rh[L]), label_rh[L])
                       for L in levels])
    cs22 = axA.contour(TA, RH * 100, TW, levels=[22], colors=[VERM], linewidths=2.2)

    rows = [(23, 92), (29, 54), (31, 45), (35, 32), (41, 18)]
    axA.plot([r[0] for r in rows], [r[1] for r in rows], "o", color=VERM,
             ms=6, mec="white", mew=1.0, zorder=5)
    axA.annotate("cool and humid", xy=(23, 92), xytext=(25.2, 93),
                 fontsize=8.4, color=INK,
                 arrowprops=dict(arrowstyle="->", color=MUTE, lw=0.9))
    axA.annotate("hot and dry", xy=(41, 18), xytext=(33.0, 22),
                 fontsize=8.4, color=INK,
                 arrowprops=dict(arrowstyle="->", color=MUTE, lw=0.9))
    axA.text(31.5, 43, "$T_{wb}$ = 22 °C", fontsize=9, color=VERM, rotation=-22,
             bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.5))
    axA.text(11.0, 16, "every point on one line is\nthe same thermal environment",
             fontsize=8.4, color=MUTE,
             bbox=dict(fc="white", ec="none", alpha=0.85, pad=2.0))

    axA.set_xlabel("Air temperature (°C)")
    axA.set_ylabel("Relative humidity (%)")
    axA.set_title("(a)  Lines of constant wet-bulb temperature", loc="left")
    axA.set_xlim(10, 45)
    axA.set_ylim(10, 100)

    # ---- B: walking the highlighted isopleth
    labels = [f"{t}°C\n{h}%" for t, h in rows]
    dryv, evapv, wr, sweat = [], [], [], []
    for Ta, RHp in rows:
        Pa = psat(22.0) + (22.0 - Ta) / LR
        rh_i = Pa / psat(Ta)
        h_c = hc(V_REF)
        dryv.append(h_c * (T_SKIN - Ta))
        evapv.append(LR * h_c * (psat(T_SKIN) - Pa))
        wr.append(wreq(float(Ta), rh_i))
        e_req = H_prod() - h_c * (T_SKIN - Ta) * A_BODY
        eff = 1.0 - wr[-1] ** 2 / 2.0            # ISO 7933 sweating efficiency
        sweat.append(e_req / max(eff, 0.5) / 2426.0 * 3.6)   # L/h

    x = np.arange(len(rows))
    axB.bar(x, dryv, width=0.62, color=BLUE, alpha=0.75, label="dry")
    axB.bar(x, evapv, width=0.62, bottom=np.maximum(dryv, 0), color=GREEN,
            alpha=0.75, label="evaporative")
    axB.plot(x, np.array(dryv) + np.array(evapv), "-o", color=INK, lw=1.4, ms=5,
             label=r"total $Q_{\max}$ (invariant)")
    axB.axhline(0, color=MUTE, lw=0.8)
    for i, d in enumerate(dryv):
        if d < 0:
            axB.text(i, d - 25, "heat gained", fontsize=7.6, color=VERM,
                     ha="center", va="top")

    axB.set_xticks(x)
    axB.set_xticklabels(labels, fontsize=8.4)
    axB.set_ylabel("Heat-loss capacity (W m$^{-2}$)")
    axB.set_ylim(-300, 1020)
    axB.set_title("(b)  Along the isopleth the routes trade off exactly", loc="left")

    axB2 = axB.twinx()
    axB2.grid(False)
    axB2.plot(x, wr, "-s", color=PURPLE, lw=1.5, ms=5,
              label=r"required wettedness $w_{req}$")
    axB2.set_ylabel(r"$w_{req}$", color=PURPLE)
    axB2.tick_params(axis="y", colors=PURPLE)
    axB2.set_ylim(0, 1.55)
    axB2.spines["top"].set_visible(False)
    axB2.spines["right"].set_color(PURPLE)

    hb, lb = axB.get_legend_handles_labels()
    h2, l2 = axB2.get_legend_handles_labels()
    axB.legend(hb + h2, lb + l2, loc="upper left", ncol=2, columnspacing=1.1)

    fig.tight_layout()
    finish(fig, "fig_wetbulb_isopleth")


# ========================================================================== #
# FIGURE 3 -- the two links of the chain, and the boundary they share
# ========================================================================== #
CHAIN_TA = 30.0      # C, the environment panel (b) is drawn in
CHAIN_RH = 0.60
CHAIN_SOLAR_ABS = 50.0


def skin_blood_flow(T_sk, T_core=T_CORE):
    """Minimum skin blood flow to carry the whole production, L/min.

    Eq. (skbf) in the appendix. T_core enters only here, and only as the
    location of the vertical asymptote: the demand diverges as T_sk -> T_core.
    """
    return H_prod() / (RHO_CB * (T_core - T_sk)) * 60.0


def wreq_of_tsk(T_sk, Ta=CHAIN_TA, rh=CHAIN_RH, solar_abs=CHAIN_SOLAR_ABS):
    """Required wettedness as a function of skin temperature, environment fixed."""
    dry = hc(V_REF) * (T_sk - Ta) * A_BODY - solar_abs * A_BODY
    return (H_prod() - dry) / (he(V_REF) * (psat(T_sk) - rh * psat(Ta)) * A_BODY)


def _box(ax, x, y, w, h, title, sub, face, edge):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=face, edgecolor=edge,
                               lw=1.4, zorder=3, joinstyle="round"))
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
            fontsize=9.2, color=INK, zorder=4)
    ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
            fontsize=8.6, color=MUTE, zorder=4)


def fig_two_links():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.8, 3.9))

    # ---- A: the chain, and the boundary the two links share
    axA.set_xlim(0, 10)
    axA.set_ylim(0, 6)
    axA.axis("off")
    axA.grid(False)

    _box(axA, 0.0, 4.1, 2.1, 1.2, "Working muscle", r"$T_{core}\approx39$ °C",
         "#f4f3ee", MUTE)
    _box(axA, 4.1, 4.1, 1.8, 1.2, "Skin", r"$T_{sk}$", "#fdf1e6", ORANGE)
    _box(axA, 7.9, 4.1, 2.1, 1.2, "Air", r"$T_{wb}$", "#f4f3ee", MUTE)

    for x0, x1, col in ((2.15, 4.05, VERM), (5.95, 7.85, BLUE)):
        axA.annotate("", xy=(x1, 4.70), xytext=(x0, 4.70),
                     arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2,
                                     shrinkA=0, shrinkB=0))
    axA.text(3.10, 5.45, "transport", ha="center", fontsize=8.8, color=VERM)
    axA.text(3.10, 4.02, r"$T_{core}\!-\!T_{sk}$", ha="center", va="top",
             fontsize=8.4, color=VERM)
    axA.text(6.90, 5.45, "surface", ha="center", fontsize=8.8, color=BLUE)
    axA.text(6.90, 4.02, r"$\Phi(T_{wb})$", ha="center", va="top",
             fontsize=8.4, color=BLUE)

    axA.annotate("shared boundary", xy=(5.0, 4.1), xytext=(5.0, 3.20),
                 ha="center", fontsize=8.6, color=ORANGE,
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.0))

    axA.text(5.0, 2.35, r"Raise $T_{sk}$ and the two gaps move oppositely:",
             ha="center", fontsize=9.0, color=INK)
    axA.text(3.25, 1.70, "gap narrows,\ntransport tightens", ha="center",
             va="top", fontsize=8.6, color=VERM)
    axA.text(6.75, 1.70, "gap widens,\nsurface eases", ha="center", va="top",
             fontsize=8.6, color=BLUE)

    axA.plot([5.0, 5.0], [1.02, 0.72], color=GREY, lw=1.1, ls=(0, (3, 2)))
    axA.text(5.0, 0.58, "the same sweat loads both links: the fluid that\n"
                        "evaporates at the surface leaves the circulation",
             ha="center", va="top", fontsize=8.2, color=MUTE)

    axA.set_title("(a)  Two links in series, one shared boundary", loc="left")

    # ---- B: the two demands against the boundary they share
    t = np.linspace(28.0, 38.0, 400)
    flow = np.array([skin_blood_flow(x) for x in t])
    w = np.array([wreq_of_tsk(x) for x in t])

    axB.plot(t, flow, color=VERM, lw=2.0)
    axB.set_xlim(28, 38)
    axB.set_ylim(0, 8)
    axB.set_xlabel("Mean skin temperature (°C)")
    axB.set_ylabel("Required skin blood flow (L min$^{-1}$)", color=VERM)
    axB.tick_params(axis="y", colors=VERM)

    axB2 = axB.twinx()
    axB2.plot(t, w, color=BLUE, lw=2.0)
    axB2.set_ylim(0, 2)
    axB2.set_ylabel(r"Required wettedness $w_{req}$", color=BLUE)
    axB2.tick_params(axis="y", colors=BLUE)
    axB2.grid(False)
    axB2.spines["top"].set_visible(False)
    axB2.axhline(1.0, color=BLUE, lw=1.0, ls="--", alpha=0.55)
    axB2.text(28.2, 1.03, r"$w_{req}=1$: skin fully wet", fontsize=8.0,
              color=BLUE, va="bottom")

    axB.axvline(31.0, color=MUTE, lw=1.0, ls=":")
    axB.annotate("this appendix\nholds $T_{sk}$ here", xy=(31.0, 6.6),
                 xytext=(31.5, 6.9), fontsize=8.2, color=MUTE, va="top",
                 arrowprops=dict(arrowstyle="->", color=MUTE, lw=0.9))

    axB.annotate("tightens", xy=(36.6, skin_blood_flow(36.6)),
                 xytext=(-6, 16), textcoords="offset points", fontsize=8.6,
                 color=VERM, ha="right",
                 arrowprops=dict(arrowstyle="->", color=VERM, lw=1.0))
    axB2.annotate("eases", xy=(35.0, wreq_of_tsk(35.0)),
                  xytext=(2, -22), textcoords="offset points", fontsize=8.6,
                  color=BLUE, ha="left",
                  arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0))
    axB.text(35.5, 7.25, r"$\to\infty$ as $T_{sk}\to T_{core}$", fontsize=8.2,
             color=VERM, ha="left")
    axB.text(28.15, 0.98,
             "held fixed:  $T_a$ = 30 °C,  RH 60%,\n"
             r"$T_{core}$ = 39 °C,  5:00 min km$^{-1}$",
             fontsize=8.0, color=MUTE, ha="left", va="top",
             bbox=dict(fc="white", ec=GRID, lw=0.7, pad=3.0))

    axB.set_title("(b)  They pull on that boundary in opposite directions",
                  loc="left")

    fig.tight_layout()
    finish(fig, "fig_two_links")


# ========================================================================== #
# FIGURE 4 -- the storage bank, and the distance over which it lasts
# ========================================================================== #
EVENTS = [("800 m", 800.0), ("1500 m", 1500.0), ("5000 m", 5000.0),
          ("10 km", 10000.0), ("half", 21097.5), ("marathon", 42195.0)]


def fig_duration_bank():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.8, 3.9))

    # ---- A: core temperature against distance, at several wettednesses
    ladder = [(1.05, BLUE), (1.10, GREEN), (1.25, ORANGE), (1.50, VERM)]
    d = np.linspace(0, 42195, 600)
    t_lim = T_CORE_0 + DT_TOL

    axA.axhline(t_lim, color=INK, lw=1.3, ls="--", zorder=1)
    axA.text(41700, t_lim + 0.07, f"tolerable ceiling, +{DT_TOL:.1f} °C",
             fontsize=8.4, color=INK, ha="right", va="bottom")

    # the compensable case: balance closes, so core plateaus and never arrives
    axA.plot([0, 42195], [38.0, 38.0], color=GREY, lw=1.5, ls=(0, (5, 3)),
             zorder=1)
    axA.text(500, 37.83, r"$w_{req}\leq 1$: core plateaus",
             fontsize=8.4, color=MUTE, ha="left", va="top")

    for w, col in ladder:
        Ta = Ta_for_wreq(w)
        S = storage_rate(w, Ta)
        rate = S / (MASS * C_BODY)                   # K per second
        temp = T_CORE_0 + rate * d / V_REF
        axA.plot(d, np.minimum(temp, 41.0), color=col, lw=1.9, zorder=4,
                 label=rf"$w_{{req}}$ = {w:.2f}")
        d_star = (DT_TOL / rate) * V_REF
        if d_star <= 42195:
            axA.plot([d_star], [t_lim], "o", color=col, ms=6, mec="white",
                     mew=0.9, zorder=6)
            axA.annotate(f"{d_star/1000:.1f} km", xy=(d_star, t_lim),
                         xytext=(7, -12), textcoords="offset points",
                         fontsize=8.0, color=col, ha="left")
        else:
            axA.annotate(f"{d_star/1000:.0f} km", xy=(42195, T_CORE_0 + DT_TOL
                                                      * 42195 / d_star),
                         xytext=(-6, 6), textcoords="offset points",
                         fontsize=8.0, color=col, ha="right")

    axA.set_xlabel("Distance covered (km)")
    axA.set_ylabel("Core temperature (°C)")
    axA.set_title("(a)  Overshooting the ceiling is a stopwatch", loc="left")
    axA.set_xlim(0, 42195)
    axA.set_xticks([0, 10000, 20000, 30000, 42195])
    axA.set_xticklabels(["0", "10", "20", "30", "42.2"])
    axA.set_ylim(36.6, 40.6)
    axA.legend(loc="lower right", borderpad=0.2, labelspacing=0.35,
               handlelength=1.6)

    # ---- B: the bank as a share of production, against race distance
    dd = np.logspace(math.log10(100.0), math.log10(42195.0), 400)
    share = 100.0 * D_BANK / dd

    axB.fill_between(dd, 100.0, np.maximum(share, 100.0), color=BLUE,
                     alpha=0.13, lw=0)
    axB.fill_between(dd, np.minimum(share, 100.0), 100.0, color=VERM,
                     alpha=0.13, lw=0)
    axB.plot(dd, share, color=INK, lw=2.0)
    axB.axhline(100.0, color=MUTE, lw=1.1, ls="--")
    axB.axvline(D_BANK, color=VERM, lw=1.0, ls=":")

    for lab, dist in EVENTS:
        y = 100.0 * D_BANK / dist
        axB.plot([dist], [y], "o", color=INK, ms=4.5, mec="white", mew=0.8,
                 zorder=6)
        off = (-2, 14) if dist > 30000 else (11, 8)
        axB.annotate(lab, xy=(dist, y), xytext=off, textcoords="offset points",
                     fontsize=7.8, color=MUTE,
                     ha="right" if dist > 30000 else "left")

    axB.annotate(f"storage alone covers the whole race out to\n{D_BANK/1000:.1f} km,"
                 " whatever the pace and whoever runs it",
                 xy=(D_BANK, 100.0), xytext=(118, 6.2), fontsize=8.4,
                 color=INK, ha="left", va="bottom",
                 arrowprops=dict(arrowstyle="->", color=MUTE, lw=0.9),
                 bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    axB.text(118, 170, "heat is bankable:\nthe race ends first", fontsize=8.6,
             color=BLUE, va="bottom")
    axB.text(40000, 55, "heat binds", fontsize=8.6, color=VERM, ha="right")

    axB.set_xscale("log")
    axB.set_yscale("log")
    axB.set_xlabel("Race distance (m)")
    axB.set_ylabel("Storage allowance (% of heat production)")
    axB.set_title("(b)  Where the allowance runs out", loc="left")
    axB.set_xlim(100, 42195)
    axB.set_ylim(4, 3000)
    axB.set_xticks([100, 1000, 10000, 42195])
    axB.set_xticklabels(["100", "1000", "10 000", "42 195"])
    axB.set_yticks([10, 100, 1000])
    axB.set_yticklabels(["10%", "100%", "1000%"])

    fig.tight_layout()
    finish(fig, "fig_duration_bank")


# ========================================================================== #
# FIGURE 5 -- the response on the dry-bulb axis
# ========================================================================== #
def fig_drybulb_axis():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.8, 3.9))

    # ---- A: required wettedness against air temperature, by humidity
    ta = np.linspace(8, 34, 300)
    hums = [(0.0, GREY, "0% (dry air)"), (0.30, BLUE, "30%"),
            (0.50, GREEN, "50%"), (0.70, ORANGE, "70%"), (0.90, VERM, "90%")]
    for rh, col, lab in hums:
        w = np.array([wreq(t, rh) for t in ta])
        w = np.where(w < 1.6, w, np.nan)
        axA.plot(ta, w, color=col, lw=1.9 if rh in (0.0, 0.9) else 1.5,
                 ls="--" if rh == 0.0 else "-", label=lab)
    axA.axhline(1.0, color=MUTE, lw=1.0, ls=":")
    axA.text(33.6, 1.03, "skin fully wetted", fontsize=8.2, color=MUTE, ha="right")
    axA.annotate("dry air: exactly linear,\nno curvature at all",
                 xy=(31.0, wreq(31.0, 0.0)), xytext=(24.0, 0.30), fontsize=8.4,
                 color=INK, ha="center",
                 arrowprops=dict(arrowstyle="->", color=MUTE, lw=0.9),
                 bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))

    axA.set_xlabel("Air temperature (°C)")
    axA.set_ylabel(r"Required wettedness $w_{req}$")
    axA.set_title("(a)  All curvature is a humidity effect", loc="left")
    axA.set_xlim(8, 34)
    axA.set_ylim(0, 1.6)
    axA.legend(loc="upper left", title="relative humidity",
               title_fontsize=8.4)

    # ---- B: the learned curve re-expressed on the dry-bulb axis
    for rh, col, lab in hums[1:]:
        xs, ys = [], []
        for t in np.linspace(8, 38, 400):
            w = wbgt(float(t), rh, SOLAR, WIND)
            p = learned_penalty(w)
            if p is None:
                break
            xs.append(t)
            ys.append(p)
        axB.plot(xs, ys, color=col, lw=1.8, label=lab)
        if xs:
            axB.plot([xs[-1]], [ys[-1]], "o", color=col, ms=4.5, mec="white",
                     mew=0.8)

    axB.text(37.6, 0.22,
             "each line stops where WBGT leaves the\nrange the model was evaluated over (28 °C)",
             fontsize=8.0, color=MUTE, ha="right",
             bbox=dict(fc="white", ec="none", alpha=0.85, pad=2.0))
    axB.set_xlabel("Air temperature (°C)")
    axB.set_ylabel("Predicted pace penalty (%)")
    axB.set_title("(b)  The learned penalty, on the thermometer's axis", loc="left")
    axB.set_xlim(8, 38)
    axB.set_ylim(0, 5.4)
    axB.legend(loc="upper left", title="relative humidity", title_fontsize=8.4)

    fig.tight_layout()
    finish(fig, "fig_drybulb_axis")


if __name__ == "__main__":
    print("building compensability figures into", OUT_PDF)
    fig_heat_ceiling()
    fig_wetbulb_isopleth()
    fig_two_links()
    fig_duration_bank()
    fig_drybulb_axis()
    print("done")
