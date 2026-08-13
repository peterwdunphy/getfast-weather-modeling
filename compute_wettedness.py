"""compute_wettedness.py

Physical numbers behind the "why is the heat penalty convex" argument in Results.

Everything here uses the heat-exchange coefficients reported by Taylor & Cotter (2006),
which the manuscript already cites, rather than generic textbook values:

    convective    C = 8.3  (Tsk - Ta) sqrt(v)   W m^-2 degC^-1
    radiant       R = 5.2  (Tsk - Tmrt)         W m^-2 degC^-1
    evaporative   E = 124  (Psk - Pa) sqrt(v)   W m^-2 kPa^-1

Three claims are checked:
  1. wet-bulb temperature is a sufficient statistic for evaporative + dry capacity
     (dry and wet shares trade off exactly; the total is invariant along a Twb isopleth)
  2. required skin wettedness w_req = E_req / E_max is CONVEX in wet-bulb, because the
     numerator is ~linear while the denominator is concave (Clausius-Clapeyron)
  3. the convexity predicted by (2) is compared to the curvature of the LEARNED curve

Run: /home/bb/test_env/bin/python3 compute_wettedness.py
"""
import numpy as np

# --- saturation vapour pressure, kPa (Tetens). Convex in T: this is the whole story. ---
def psat(T):
    return 0.6108 * np.exp(17.27 * T / (T + 237.3))

# Taylor & Cotter coefficients. Their ratio he/hc sets the psychrometric constant.
HC, HE, HR = 8.3, 124.0, 5.2
RATIO = HE / HC                      # degC per kPa
print(f"he/hc from Taylor & Cotter = {RATIO:.2f} degC/kPa"
      f"  -> psychrometric constant {1/RATIO:.4f} kPa/degC (standard sea-level value ~0.067)")

def twb_from(Ta, rh):
    """Invert the psychrometer equation Ta + ratio*Pa = Twb + ratio*Ps(Twb) for Twb."""
    Pa = psat(Ta) * rh / 100.0
    lhs = Ta + RATIO * Pa
    g = np.linspace(-20, Ta, 40001)          # Twb <= Ta always
    return g[np.argmin(np.abs(g + RATIO * psat(g) - lhs))]

def rh_for_twb(Ta, twb):
    """Relative humidity that puts a given air temperature on a target wet-bulb isopleth."""
    Pa = (twb + RATIO * psat(twb) - Ta) / RATIO
    return 100.0 * Pa / psat(Ta)

# =====================================================================
# 1. Wet-bulb as sufficient statistic: walk one Twb isopleth
# =====================================================================
TSK, V, TWB0 = 35.0, 3.35, 22.0          # skin 35 C, 3:30-marathon speed 3.35 m/s
sq = np.sqrt(V)
print(f"\n[1] one wet-bulb isopleth, Twb = {TWB0} degC, skin {TSK} degC, speed {V} m/s")
print(f"{'Ta':>6} {'RH%':>7} {'dry W/m2':>10} {'evap W/m2':>11} {'total':>9}")
for Ta in (23.0, 29.0, 35.0, 41.0):
    rh = rh_for_twb(Ta, TWB0)
    dry = HC * (TSK - Ta) * sq                       # convective only (Tmrt = Ta, no sun)
    ev  = HE * (psat(TSK) - psat(Ta) * rh / 100) * sq
    print(f"{Ta:6.1f} {rh:7.1f} {dry:10.1f} {ev:11.1f} {dry+ev:9.1f}")
print("   total is invariant along the isopleth -> Twb carries all the information")

# =====================================================================
# 2. Required wettedness vs wet-bulb
# =====================================================================
# metabolic heat production for a 70 kg runner at 3:30 marathon pace
MASS, BSA = 70.0, 1.85
speed_kmh = 42.195 / 3.5
Hprod = 0.80 * (MASS * speed_kmh * 1.0) * 1000 * 4.184 / 3600 / BSA   # ~80% of turnover is heat
print(f"\n[2] metabolic heat production {Hprod:.0f} W/m2 (70 kg, 3:30 marathon, 80% as heat)")

def wreq(twb, rh=60.0, solar=0.0):
    """Required skin wettedness at a given wet-bulb temperature."""
    # recover the air temperature sitting on this isopleth at the stated humidity
    g = np.linspace(twb, twb + 25, 6001)
    Ta = g[np.argmin(np.abs(np.array([rh_for_twb(t, twb) for t in g]) - rh))]
    Pa = psat(Ta) * rh / 100.0
    dry = HC * (TSK - Ta) * sq + HR * (TSK - Ta) - solar
    Emax = HE * (psat(TSK) - Pa) * sq
    return (Hprod - dry) / Emax, Ta, Emax

print(f"{'Twb':>6} {'Ta':>6} {'Emax':>8} {'w_req':>8} {'d w_req/degC':>13}")
grid = np.array([10.7, 15.0, 19.3, 23.7, 28.1])
prev = None
for t in grid:
    w, Ta, Em = wreq(t)
    inc = "" if prev is None else f"{(w-prev[0])/(t-prev[1]):13.4f}"
    print(f"{t:6.1f} {Ta:6.1f} {Em:8.0f} {w:8.3f} {inc}")
    prev = (w, t)

# Clausius-Clapeyron: fractional growth of Ps per kelvin, and capacity loss per degree
for t in (12.0, 26.0):
    d = (psat(t + 0.5) - psat(t - 0.5)) / psat(t) * 100
    cap = HE * (psat(TSK) - psat(t)) * sq
    cap2 = HE * (psat(TSK) - psat(t + 1)) * sq
    print(f"   at Twb {t:4.1f}: Ps grows {d:.1f}%/K, evaporative ceiling falls {cap-cap2:.0f} W/m2 per degree")

# =====================================================================
# 3. Predicted vs observed curvature, 20-24 vs 24-28 C
# =====================================================================
print("\n[3] steepening between the 20-24 and 24-28 degC bands")
a = np.load("/weather/data/heat_percentile_averaged.npz", allow_pickle=True)
w_ax, curve = a["wsweep"].astype(float), a["pctl_curves"].astype(float)[49]
def band(lo, hi, x, y):
    return (np.interp(hi, x, y) - np.interp(lo, x, y)) / (hi - lo)
s1 = band(20, 24, w_ax, curve); s2 = band(24, 28, w_ax, curve)
print(f"   learned curve      : {s1:.3f} %/degC -> {s2:.3f} %/degC   ratio {s2/s1:.2f}x")
for rh, solar, lab in [(40, 0, "40% RH, no sun"), (70, 200, "70% RH, 200 W/m2"), (85, 0, "85% RH, no sun")]:
    v1 = (wreq(24, rh, solar)[0] - wreq(20, rh, solar)[0]) / 4
    v2 = (wreq(28, rh, solar)[0] - wreq(24, rh, solar)[0]) / 4
    print(f"   w_req {lab:18}: {v1:.4f} -> {v2:.4f}          ratio {v2/v1:.2f}x")

# =====================================================================
# 4. Storage allowance scales as 1/duration
# =====================================================================
print("\n[4] heat-storage allowance for a 2.5 degC core rise (70 kg, 3.47 kJ/degC/kg)")
Q = MASS * 3.47 * 2.5                                  # kJ
for dur_s, lab in [(10, "100 m"), (15*60, "5 km"), (3*3600, "marathon 3:00"), (4.5*3600, "marathon 4:30")]:
    print(f"   {lab:16}: {Q*1000/dur_s:9.0f} W  ({Q*1000/dur_s/BSA:7.0f} W/m2)")
print(f"   metabolic heat production for comparison: {Hprod:.0f} W/m2")
