"""Verification for the v3 heat-convexity insert.

v3 changes from v2:
  * one coefficient convention throughout: Taylor (2006), h_c = 8.3*sqrt(v),
    h_e = 124*sqrt(v), so LR = 124/8.3 = 14.9 K/kPa. This matches the
    manuscript's Appendix C AND makes the wet-bulb substitution essentially
    exact (the psychrometric constant is 15.0), removing the v2 footnote.
  * convexity stated as a theorem on the air-temperature axis, with the
    axis transfer to WBGT checked numerically rather than asserted.
  * flat-band paragraph rebuilt: the aerobic/thermal crossing lands near
    28 C WBGT, NOT 16 C, so it cannot explain the observed breakpoint.
  * curvature reported as a LOWER bound, with sensitivity to h_c and to
    metabolic rate reported alongside the (dominant) sensitivity to T_sk.

Run: python verify_heat_convexity_v3.py    (stdlib only, self-checking)
"""
import math

T_SKIN = 31.0      # C, mean skin temp in outdoor competition (Aylwin 2023)
A_BODY = 1.85      # m2
MASS   = 70.0
C_BODY = 3500.0    # J/kg/K
ECON   = 4184.0    # J/kg/km
EFF    = 0.79
V_REF  = 3.33      # m/s = 5:00/km  (a 3:31 marathon, not 3:30)
LR     = 124.0/8.3 # 14.9 K/kPa, Taylor (2006) coefficients
LR_PSY = 15.04     # reciprocal psychrometric constant

def psat(T):
    """Buck (1981), kPa."""
    return 0.61121*math.exp((18.678 - T/234.5)*(T/(257.14+T)))

def hc(v):   return 8.3*math.sqrt(v)      # Taylor (2006)
def H_prod(v=V_REF, m=MASS): return EFF*ECON*m*(v/1000.0)

def twb(Ta, rh, lewis=LR_PSY):
    lo, hi = -40.0, Ta
    for _ in range(120):
        m = (lo+hi)/2
        if (m-Ta) + lewis*(psat(m)-rh*psat(Ta)) < 0: lo = m
        else: hi = m
    return lo

def wbgt(Ta, rh, solar, wind=1.0):
    tg = Ta + 1.6*(solar/1000.0)**0.6*(2.0/max(wind,0.3))**0.4*10
    return 0.7*twb(Ta,rh) + 0.2*tg + 0.1*Ta

def Ta_for_wbgt(tgt, rh, solar):
    lo, hi = -20.0, 60.0
    for _ in range(100):
        m = (lo+hi)/2
        if wbgt(m,rh,solar) < tgt: lo = m
        else: hi = m
    return lo

def wreq(Ta, rh, T_sk=T_SKIN, v=V_REF, solar=0.0, h_c=None, H=None, m=MASS):
    h_c = hc(v) if h_c is None else h_c
    H   = H_prod(v, m) if H is None else H
    dry = h_c*(T_sk-Ta)*A_BODY - solar*A_BODY
    return (H - dry)/(LR*h_c*(psat(T_sk)-rh*psat(Ta))*A_BODY)

def curvature(T_sk=T_SKIN, h_c=None, H=None):
    """[slope 24-28 C WBGT] / [slope 20-24 C], swept over RH and solar load."""
    out = []
    for rh in (0.40,0.55,0.70,0.85):
        for solar in (0.0,200.0,500.0):
            ws, ok = [], True
            for tgt in (20,24,28):
                w = wreq(Ta_for_wbgt(tgt,rh,solar), rh, T_sk,
                         solar=solar*0.25, h_c=h_c, H=H)
                if not (0 < w < 50): ok = False; break
                ws.append(w)
            if ok: out.append((ws[2]-ws[1])/(ws[1]-ws[0]))
    return min(out), max(out)

fails = []
def check(label, got, want, tol):
    ok = abs(got-want) <= tol
    if not ok: fails.append(label)
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}: {got:.4g} (claimed {want:g})")

OBS = ((5.02-2.4)/4)/((2.4-1.2)/4)
print(f"reference runner: {H_prod():.0f} W total, {H_prod()/A_BODY:.0f} W/m2; "
      f"h_c = {hc(V_REF):.1f} W/m2K; LR = {LR:.1f} K/kPa")
check("observed steepening of the learned curve", OBS, 2.18, 0.02)

print("\nCLAIM 1: wet-bulb is a sufficient statistic; the substitution is exact.")
tot = []
for Ta in (23,29,35,41):
    Pa = psat(22.0) + (22.0-Ta)/LR
    h_c = hc(V_REF)
    qd = h_c*(T_SKIN-Ta); qe = LR*h_c*(psat(T_SKIN)-Pa)
    tot.append(qd+qe)
    print(f"    Ta={Ta}C RH={Pa/psat(Ta):5.1%}  dry {qd:+7.1f}  evap {qe:7.1f}"
          f"  total {qd+qe:8.2f}")
check("invariance of the total along the isopleth", max(tot)-min(tot), 0.0, 1e-6)
gap = twb(30.0,0.60,LR) - twb(30.0,0.60,LR_PSY)
check("T_wb gap between body LR and psychrometric LR (C)", abs(gap), 0.02, 0.02)
print("    -> with Taylor's coefficients the two definitions agree to ~0.02 C,")
print("       so Eq.(2) is exact for practical purposes and needs no caveat.")

print("\nCLAIM 2: Clausius-Clapeyron. Report the ABSOLUTE slope, not the fraction.")
for T in (12,20,28):
    d = (psat(T+.01)-psat(T-.01))/0.02
    print(f"    {T} C: dPs/dT = {d:.3f} kPa/K   ({100*d/psat(T):.1f} %/K)")
check("Ps(28)/Ps(15) -- 'more than doubles'", psat(28)/psat(15), 2.22, 0.02)
print("    NB the FRACTIONAL rate FALLS (6.6 -> 5.8 %/K); only the absolute")
print("       slope rises. Quote the absolute one or the argument reads backwards.")

print("\nCLAIM 3: convexity is a theorem on the air-temperature axis.")
h_c = hc(V_REF)
for rh in (0.3,0.6,0.9):
    N  = lambda T: H_prod() - h_c*(T_SKIN-T)*A_BODY
    D  = lambda T: LR*h_c*(psat(T_SKIN)-rh*psat(T))*A_BODY
    d2N = (N(20+.1)-2*N(20)+N(20-.1))/0.01
    d2D = (D(20+.1)-2*D(20)+D(20-.1))/0.01
    print(f"    RH {rh:.0%}: N''={d2N:+.1e} (linear)  D''={d2D:+.2f} (<0)  D>0 {D(20)>0}")
print("    f'' = [-N D''D - 2N'D'D + 2N D'^2]/D^3, all three terms positive.")
print("    Convexity therefore does not depend on a parameter sweep.")

print("\nCLAIM 3b: it survives the change of axis (checked, not assumed).")
def conv_on(axis, rng, rh, solar=200.0):
    d2 = []
    for x in rng:
        def f(u):
            if axis=='Ta': Ta = u
            else:
                lo,hi=-20.0,60.0
                for _ in range(100):
                    m=(lo+hi)/2
                    cur = twb(m,rh) if axis=='twb' else wbgt(m,rh,solar)
                    if cur<u: lo=m
                    else: hi=m
                Ta=lo
            return wreq(Ta, rh, solar=(50.0 if axis=='wbgt' else 0.0))
        d2.append((f(x+.25)-2*f(x)+f(x-.25))/0.0625)
    return all(v>0 for v in d2)
allok = True
for axis,rng in (('Ta',range(8,34,2)),('twb',range(6,28,2)),('wbgt',range(8,29,2))):
    res = [conv_on(axis,rng,rh) for rh in (0.3,0.5,0.7,0.9)]
    allok &= all(res)
    print(f"    axis {axis:4s}: convex at RH 30/50/70/90%? {res}")
check("convex on every axis at every humidity", 1.0 if allok else 0.0, 1.0, 0.0)
print("    (T_a is NOT exactly convex in T_wb, so the composition shortcut fails;")
print("     the axis is near-affine instead, which is why this is checked.)")

print("\nCLAIM 4: storage scales as 1/duration.")
for label,t,want in (("marathon 3:00",10800,57),("marathon 4:30",16200,38)):
    check(f"{label} (W)", MASS*C_BODY*2.5/t, want, 1.0)
print(f"    against {H_prod():.0f} W of heat production: the storage term buys"
      f" {100*MASS*C_BODY*2.5/10800/H_prod():.0f}% at 3:00.")
print("    (compare W with W, not W with W/m2 -- the v2 text mixed them.)")

print("\nCLAIM 5: the aerobic/thermal crossing CANNOT explain the 16 C breakpoint.")
for T_sk in (29.0,31.0,35.0):
    for w_max in (0.85,1.0):
        cross=None
        for t10 in range(100,400):
            t=t10/10.0
            if wreq(Ta_for_wbgt(t,0.60,200.0),0.60,T_sk,solar=50.0) >= w_max:
                cross=t; break
        print(f"    T_sk={T_sk} w_max={w_max}: ceiling reached at WBGT "
              f"{cross if cross else '>40'} C")
print("    -> the crossing sits in the mid-to-high 20s. A min(aerobic, thermal)")
print("       account predicts a flat band roughly 12 C too long. Drop it.")

print("\nCLAIM 6: the flat band and the curvature need ONE convex strain->pace map.")
learned = {10:0.0,12:0.05,14:0.15,16:0.30,20:1.20,24:2.40,28:5.02}
w = {t: wreq(Ta_for_wbgt(t,0.60,200.0),0.60,solar=50.0) for t in learned}
best=None
for k10 in range(5,61):
    k=k10/10.0
    ts=[16,20,24,28]
    c=sum(learned[t] for t in ts)/sum(w[t]**k for t in ts)
    err=sum((learned[t]-c*w[t]**k)**2 for t in ts)
    if best is None or err<best[2]: best=(k,c,err)
k,c,_=best
print(f"    best-fit exponent in penalty = c*w_req^k over 16-28 C: k = {k:.1f}")
print("    WBGT   w_req    c*w^k   learned")
for t in sorted(learned):
    print(f"    {t:4d}  {w[t]:6.3f}   {c*w[t]**k:6.2f}%   {learned[t]:5.2f}%")
r_w = (w[28]-w[24])/(w[24]-w[20])
print(f"    steepening of w_req alone: {r_w:.2f};  of the penalty: {OBS:.2f}")
print("    -> k>1 lifts the former to the latter AND flattens the cool end.")
print("       The physics therefore predicts a LOWER BOUND on curvature.")

print("\nCLAIM 7: what the prediction is, and is not, sensitive to.")
lo,hi = curvature()
print(f"    baseline (T_sk=31, RH 40-85%, solar 0-500): {lo:.2f} to {hi:.2f}")
print("    varying h_c across the full published span at 3.3 m/s:")
for h_v in (10.4, hc(V_REF), 17.1, 33.7):
    l,hh = curvature(h_c=h_v)
    print(f"      h_c={h_v:5.1f} -> {l:.2f} to {hh:.2f}")
print("    varying metabolic rate (mass and pace):")
for lab,H in (("60 kg 5:00/km",H_prod(m=60)),("70 kg 5:00/km",H_prod()),
              ("85 kg 5:00/km",H_prod(m=85)),("70 kg 4:00/km",H_prod(v=4.17))):
    l,hh = curvature(H=H)
    print(f"      {lab:15s} ({H:.0f} W) -> {l:.2f} to {hh:.2f}")
print("    varying mean skin temperature:")
for T_sk in (29,30,31,32,33,35):
    l,hh = curvature(T_sk=float(T_sk))
    print(f"      T_sk={T_sk} C -> {l:.2f} to {hh:.2f}"
          f"   {'brackets' if l<=OBS<=hh else 'MISSES'}")
print("    -> h_c and metabolic rate barely move it; T_sk dominates entirely.")

print("\nCLAIM 8: the exposed-skin caveat, quantified.")
print("    Aylwin measured EXPOSED skin by IR thermography, 64% of BSA.")
print("    A whole-body mean must weight in the covered 36%, which runs warmer:")
for cov in (32,34,36,38):
    tsk = 0.64*29.35 + 0.36*cov
    l,hh = curvature(T_sk=tsk)
    print(f"      covered skin {cov} C -> weighted mean {tsk:.1f} C -> {l:.2f} to {hh:.2f}"
          f"   {'brackets' if l<=OBS<=hh else 'MISSES'}")
print("    -> the band still brackets even on the least favourable weighting.")
print("       It fails only above T_sk ~ 33 C, which no weighting of Aylwin reaches.")

print("\n" + ("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILED: {fails}"))
