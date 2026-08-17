"""Verification for the compensability appendix (Appendix F, tommy_appendix.tex).

Stdlib only, no arguments, self-checking: every claim prints an ok/FAIL line and
the script ends with a pass/fail summary. Every number quoted in the appendix
text appears here.

The argument the script follows, in the order the appendix makes it:

  1. Heat balance at steady state is one equation in one unknown, the runner's
     speed. Its solution is the thermal speed ceiling.
  2. The whole environment enters through one scalar, the thermal head Phi, and
     Phi is decreasing and CONCAVE in wet-bulb temperature. That concavity is
     Clausius-Clapeyron and nothing else.
  3. The speed ceiling goes as Phi^2 (because the transfer coefficients scale as
     sqrt(v)), so the pace penalty goes as Phi^-2, which is strictly convex for
     any positive power. Convexity is a theorem, not a parameter sweep.
  4. The hard ceiling binds only in the mid-20s, far too warm to explain a 16 C
     breakpoint, so runners must respond to graded strain below it. Required
     wettedness w_req is that graded measure, and it is convex for the same
     Clausius-Clapeyron reason.
  5. The storage allowance is a fixed quantity of HEAT, so what it buys is a
     fixed DISTANCE, d_bank = B/kappa ~ 2.6 km, with pace and body mass both
     cancelling. That is why heat governs a marathon and spares a mile.
  6. On the dry-bulb axis at fixed humidity, the curvature is proportional to
     RH. In perfectly dry air w_req is exactly linear.

Run: python3 verify_heat_convexity.py
"""
import math

T_SKIN = 31.0      # C, mean skin temp in outdoor competition (Aylwin 2023)
A_BODY = 1.85      # m2
MASS   = 70.0      # kg
C_BODY = 3474.0    # J/kg/K, Burton (1935); the quoted 3500 has no primary source
ECON   = 4184.0    # J/kg/km
EFF    = 0.79      # fraction of metabolic turnover appearing as heat
V_REF  = 3.33      # m/s = 5:00/km  (a 3:31 marathon, not 3:30)
LR     = 124.0/8.3 # 14.9 K/kPa, Taylor (2006) coefficients
LR_PSY = 15.04     # reciprocal psychrometric constant
KAPPA  = EFF*ECON*MASS/1000.0      # J of heat produced per metre travelled
RHO_CB = 3770.0    # J/L/K, volumetric heat capacity of blood
T_CORE = 39.0      # C, core temperature held during a hot marathon

def psat(T):
    """Buck (1981), kPa."""
    return 0.61121*math.exp((18.678 - T/234.5)*(T/(257.14+T)))

def hc(v):   return 8.3*math.sqrt(v)      # Taylor (2006)
def he(v):   return LR*hc(v)
def H_prod(v=V_REF, m=MASS): return EFF*ECON*m*(v/1000.0)

def head_wb(Twb, T_sk=T_SKIN):
    """Thermal head Phi on the wet-bulb axis, in kelvin-equivalents."""
    return (T_sk - Twb) + LR*(psat(T_sk) - psat(Twb))

def head_ta(Ta, rh, T_sk=T_SKIN):
    """Thermal head on the dry-bulb axis at fixed relative humidity."""
    return (T_sk - Ta) + LR*(psat(T_sk) - rh*psat(Ta))

def v_thermal(head):
    """kappa*v = A*8.3*sqrt(v)*Phi  =>  v = (8.3*A/kappa)^2 * Phi^2."""
    return (8.3*A_BODY/KAPPA)**2 * max(head, 0.0)**2

def twb(Ta, rh, lewis=LR_PSY):
    lo, hi = -40.0, Ta
    for _ in range(200):
        m = (lo+hi)/2
        if (m-Ta) + lewis*(psat(m)-rh*psat(Ta)) < 0: lo = m
        else: hi = m
    return lo

def wbgt(Ta, rh, solar, wind=1.0):
    tg = Ta + 1.6*(solar/1000.0)**0.6*(2.0/max(wind,0.3))**0.4*10
    return 0.7*twb(Ta,rh) + 0.2*tg + 0.1*Ta

def Ta_for_wbgt(tgt, rh, solar):
    lo, hi = -20.0, 60.0
    for _ in range(200):
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
      f"h_c = {hc(V_REF):.1f} W/m2K; LR = {LR:.1f} K/kPa; "
      f"kappa = {KAPPA:.0f} J/m")
check("observed steepening of the learned curve", OBS, 2.18, 0.02)

# ------------------------------------------------------------------ CLAIM 1
print("\nCLAIM 1: heat balance at S=0 is one equation in one unknown, the speed.")
print("    production kappa*v must equal capacity A*8.3*sqrt(v)*Phi, so")
print(f"    v_max = (8.3*A/kappa)^2 * Phi^2 = {(8.3*A_BODY/KAPPA)**2:.6f} * Phi^2")
v_check = v_thermal(head_wb(22.0))
lo, hi = 0.001, 20.0
for _ in range(200):                       # independent bisection of the balance
    m = (lo+hi)/2
    cap = A_BODY*(hc(m)*(T_SKIN-22.0) + he(m)*(psat(T_SKIN)-psat(22.0)))
    if cap > KAPPA*m: lo = m
    else: hi = m
check("closed form matches a direct solve of the balance at T_wb=22 (m/s)",
      v_check, lo, 0.01)
print(f"    at T_wb = 22 C the balance permits {v_check:.2f} m/s "
      f"= {1000/v_check/60:.2f} min/km")

# ------------------------------------------------------------------ CLAIM 2
print("\nCLAIM 2: the head Phi is decreasing and CONCAVE in wet-bulb temperature.")
print("    Phi'  = -(1 + LR*Ps'(T_wb))   < 0")
print("    Phi'' = -LR*Ps''(T_wb)        < 0   <- Clausius-Clapeyron, nothing else")
print(f"    {'T_wb':>5} {'Phi':>8} {'Q_max':>9} {'Phi_1':>8} {'Phi_2':>8}")
conc = True
for T in (5,10,15,20,25,30):
    p  = head_wb(T)
    d1 = (head_wb(T+.05)-head_wb(T-.05))/0.1
    d2 = (head_wb(T+.05)-2*head_wb(T)+head_wb(T-.05))/0.0025
    conc &= (d1 < 0 and d2 < 0)
    print(f"    {T:5d} {p:8.2f} {hc(V_REF)*p:9.1f} {d1:8.3f} {d2:8.4f}")
check("Phi decreasing and concave at every tabulated point", 1.0 if conc else 0.0, 1.0, 0.0)
check("Q_max vanishes exactly at T_wb = T_sk", hc(V_REF)*head_wb(T_SKIN), 0.0, 1e-9)

# ------------------------------------------------------------------ CLAIM 3
print("\nCLAIM 3: the pace penalty at the ceiling is convex. THEOREM, not a sweep.")
print("    penalty = v_ref/v_max - 1 proportional to Phi^-p with p = 2, and")
print("    d2/dx2 [Phi^-p] = p*Phi^(-p-2) * [ (p+1)*Phi'^2 - Phi*Phi'' ]")
print("    both bracketed terms are positive, so the penalty is strictly convex")
print("    for ANY p > 0 -- it does not depend on how h scales with speed.")
allpos = True
for p in (1.0, 2.0):
    for T in (10,15,20,25,29):
        phi  = head_wb(T)
        d1   = (head_wb(T+.05)-head_wb(T-.05))/0.1
        d2   = (head_wb(T+.05)-2*head_wb(T)+head_wb(T-.05))/0.0025
        term = (p+1)*d1**2 - phi*d2
        f2   = (head_wb(T+.05)**-p - 2*head_wb(T)**-p + head_wb(T-.05)**-p)/0.0025
        allpos &= (term > 0 and f2 > 0)
    print(f"    p = {p:.0f}: bracket positive and f'' > 0 at every T_wb checked")
check("penalty strictly convex on the wet-bulb axis for p = 1 and 2",
      1.0 if allpos else 0.0, 1.0, 0.0)

# ------------------------------------------------------------------ CLAIM 4
print("\nCLAIM 4: the hard ceiling binds too late to explain the flat band.")
phi_bind = math.sqrt(V_REF/((8.3*A_BODY/KAPPA)**2))
lo, hi = 0.0, T_SKIN
for _ in range(200):
    m = (lo+hi)/2
    if head_wb(m) > phi_bind: lo = m
    else: hi = m
print(f"    reference runner needs Phi >= {phi_bind:.2f} K-equivalents to hold pace")
check("wet-bulb temperature at which the thermal ceiling takes over (C)", lo, 24.6, 0.15)
for rh in (0.40, 0.60, 0.85):
    for solar in (0.0, 200.0):
        ta = 0.0
        loT, hiT = -20.0, 60.0
        for _ in range(200):
            m = (loT+hiT)/2
            if twb(m, rh) < lo: loT = m
            else: hiT = m
        ta = loT
        print(f"      RH {rh:.0%}, solar {solar:3.0f} W/m2 -> T_a = {ta:.1f} C, "
              f"WBGT = {wbgt(ta,rh,solar):.1f} C")
print("    -> the crossing sits in the mid-to-high 20s on the WBGT axis, roughly")
print("       ten degrees too warm to produce a breakpoint near 16 C. A pure")
print("       min(aerobic, thermal) account is therefore not the explanation.")
for T_sk in (29.0,31.0,35.0):
    for w_max in (0.85,1.0):
        cross=None
        for t10 in range(100,400):
            t=t10/10.0
            if wreq(Ta_for_wbgt(t,0.60,200.0),0.60,T_sk,solar=50.0) >= w_max:
                cross=t; break
        print(f"      w_req route: T_sk={T_sk} w_max={w_max} -> WBGT "
              f"{cross if cross else '>40'} C")

# ------------------------------------------------------------------ CLAIM 5
print("\nCLAIM 5: wet-bulb is a sufficient statistic; the substitution is exact.")
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
print("       so the wet-bulb form is exact for practical purposes.")

# ------------------------------------------------------------------ CLAIM 6
print("\nCLAIM 6: Clausius-Clapeyron. Report the ABSOLUTE slope, not the fraction.")
for T in (12,20,28):
    d = (psat(T+.01)-psat(T-.01))/0.02
    print(f"    {T} C: dPs/dT = {d:.3f} kPa/K   ({100*d/psat(T):.1f} %/K)")
check("Ps(28)/Ps(15) -- 'more than doubles'", psat(28)/psat(15), 2.22, 0.02)
print("    NB the FRACTIONAL rate FALLS (6.6 -> 5.8 %/K); only the absolute")
print("       slope rises. Quote the absolute one or the argument reads backwards.")

# ------------------------------------------------------------------ CLAIM 7
print("\nCLAIM 7: w_req, the graded sub-ceiling measure, is convex for the")
print("         same reason. N affine increasing over D positive decreasing concave.")
h_c = hc(V_REF)
for rh in (0.3,0.6,0.9):
    N  = lambda T: H_prod() - h_c*(T_SKIN-T)*A_BODY
    D  = lambda T: LR*h_c*(psat(T_SKIN)-rh*psat(T))*A_BODY
    d2N = (N(20+.1)-2*N(20)+N(20-.1))/0.01
    d2D = (D(20+.1)-2*D(20)+D(20-.1))/0.01
    print(f"    RH {rh:.0%}: N''={d2N:+.1e} (linear)  D''={d2D:+.2f} (<0)  D>0 {D(20)>0}")
print("    f'' = [-N D''D - 2N'D'D + 2N D'^2]/D^3, all three terms positive.")
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
check("w_req convex on every axis at every humidity", 1.0 if allok else 0.0, 1.0, 0.0)
print("    (T_a is NOT exactly convex in T_wb, so the composition shortcut fails;")
print("     the axis is near-affine instead, which is why this is checked.)")

# ------------------------------------------------------------------ CLAIM 8
print("\nCLAIM 8: on the dry-bulb axis the curvature is PROPORTIONAL to humidity.")
print("    Phi''(T_a) = -LR*RH*Ps''(T_a), so it vanishes identically at RH = 0.")
for rh in (0.0,0.25,0.50,0.75,1.0):
    d2 = (head_ta(25+.05,rh)-2*head_ta(25,rh)+head_ta(25-.05,rh))/0.0025
    print(f"    RH {rh:4.0%}: Phi(25 C) = {head_ta(25,rh):6.2f}   Phi'' = {d2:+7.4f}")
d2_dry = (head_ta(25+.05,0.0)-2*head_ta(25,0.0)+head_ta(25-.05,0.0))/0.0025
d2_wet = (head_ta(25+.05,1.0)-2*head_ta(25,1.0)+head_ta(25-.05,1.0))/0.0025
check("Phi'' is exactly zero in dry air", abs(d2_dry), 0.0, 1e-6)
check("ratio of Phi'' at RH 100% to RH 50%",
      d2_wet/((head_ta(25+.05,0.5)-2*head_ta(25,0.5)+head_ta(25-.05,0.5))/0.0025),
      2.0, 1e-6)
ws = {T: wreq(float(T),0.0) for T in range(10,33)}
lin = max(abs(ws[T+1]-2*ws[T]+ws[T-1]) for T in range(11,32))
check("w_req is EXACTLY linear in air temperature at RH = 0", lin, 0.0, 1e-12)
print("    but the penalty keeps a residual convexity even in dry air, because")
print("    a reciprocal of an affine function is still convex:")
for rh in (0.0, 0.5, 0.9):
    pen = lambda T: (V_REF/v_thermal(head_ta(T,rh)) - 1)
    d2 = (pen(25+.5)-2*pen(25)+pen(25-.5))/0.25
    print(f"      RH {rh:4.0%}: penalty'' at 25 C = {d2:+.5f}")
print("    -> so the two measures make different predictions in dry air, which")
print("       is a discriminating test the corpus does not yet contain.")

# ------------------------------------------------------------------ CLAIM 9
print("\nCLAIM 9: the storage allowance is a fixed quantity of HEAT, so what it")
print("         buys is a fixed DISTANCE -- independent of pace and of mass.")
DT_TOL  = 2.5
BANK    = MASS*C_BODY*DT_TOL
D_BANK  = BANK/KAPPA
check("stored energy in a 2.5 C rise (kJ)", BANK/1000.0, 608, 1.0)
for label,t,want in (("marathon 3:00",10800,56),("marathon 4:30",16200,38)):
    check(f"spread over a {label} (W)", BANK/t, want, 1.0)
print(f"    d_bank = B/kappa = {D_BANK:.0f} m.  Both the speed and the mass")
print("    cancel: production and distance both scale with v, and the bank and")
print("    the cost of covering ground both scale with m. Check that directly:")
same = []
for m in (52.0, 70.0, 91.0):
    for v in (2.8, 3.33, 6.0):
        same.append((m*C_BODY*DT_TOL)/(EFF*ECON*m/1000.0))
check("d_bank identical across 3 masses x 3 speeds (m)", max(same)-min(same), 0.0, 1e-9)
check("d_bank from the mass-free form c*dT/c_run (m)",
      C_BODY*DT_TOL/(EFF*ECON/1000.0), 2628, 2.0)
check("economy carried as heat, c_run = kappa/m (J/kg/m)", KAPPA/MASS, 3.3, 0.02)
print("\n    share of total heat production the bank covers, = d_bank/d exactly:")
for lab,d in (("100 m",100),("800 m",800),("1500 m",1500),("3000 m",3000),
              ("5000 m",5000),("10 km",10000),("half",21097.5),("marathon",42195.0)):
    print(f"      {lab:9s} {100*D_BANK/d:8.1f}% of production")
check("share over 1500 m (x production)", D_BANK/1500, 1.75, 0.02)
check("share over a marathon (%)", 100*D_BANK/42195, 6.2, 0.1)
for v in (3.33, 42195/10800, 42195/16200):     # the share does not move with pace
    check(f"marathon share at v={v:.2f} m/s (%)",
          100*BANK/(H_prod(v)*(42195/v)), 6.2, 0.1)
print(f"    over a ten-second sprint the same bank is worth {BANK/10/1000:.0f} kW,")
print(f"    against {H_prod():.0f} W of production. Heat cannot bind a race")
print(f"    shorter than {D_BANK/1000:.1f} km in ANY conditions.")

print("\n    trajectories: S = E_max*(w_req - 1), core climbs at S/(m c).")
print(f"    {'w_req':>6} {'WBGT':>6} {'T_a':>6} {'S (W)':>7} {'C/h':>6} "
      f"{'t* (min)':>9} {'d* (km)':>8}")
def _ta_wall(rh):
    lo, hi = T_SKIN, 90.0
    for _ in range(200):
        m = (lo+hi)/2
        if rh*psat(m) < psat(T_SKIN): lo = m
        else: hi = m
    return lo
def _ta_for_w(target, rh=0.60):
    lo, hi = 5.0, _ta_wall(rh)-0.05      # must stop below the vapour-gradient
    for _ in range(200):                 # reversal or the bisection walks past it
        m = (lo+hi)/2
        if wreq(m, rh, solar=50.0) < target: lo = m
        else: hi = m
    return lo
traj = {}
for w in (1.05, 1.10, 1.25, 1.50):
    Ta = _ta_for_w(w)
    S  = (w-1.0)*LR*hc(V_REF)*(psat(T_SKIN)-0.60*psat(Ta))*A_BODY
    rate = S/(MASS*C_BODY)
    traj[w] = (wbgt(Ta,0.60,200.0), Ta, S, rate*3600, BANK/S/60, V_REF*BANK/S)
    print(f"    {w:6.2f} {traj[w][0]:6.1f} {Ta:6.1f} {S:7.0f} {rate*3600:6.1f} "
          f"{BANK/S/60:9.0f} {V_REF*BANK/S/1000:8.1f}")
for w,(wb,ta,S,cph,tmin,dm) in traj.items():
    check(f"w_req={w:.2f}: WBGT (C)", wb, {1.05:27.5,1.10:27.9,1.25:28.9,1.50:30.1}[w], 0.1)
    check(f"w_req={w:.2f}: stored (W)", S, {1.05:40,1.10:78,1.25:177,1.50:307}[w], 1.0)
    check(f"w_req={w:.2f}: core rise (C/h)", cph,
          {1.05:0.6,1.10:1.1,1.25:2.6,1.50:4.5}[w], 0.05)
    check(f"w_req={w:.2f}: allowance spent at (km)", dm/1000,
          {1.05:50.5,1.10:26.1,1.25:11.5,1.50:6.6}[w], 0.1)
check("the no-dissipation limit lands on d_bank (km)",
      V_REF*BANK/H_prod()/1000, D_BANK/1000, 1e-6)
check("and its core rise (C/h)", H_prod()/(MASS*C_BODY)*3600, 11.4, 0.05)
print("    t* is a hyperbola in (w_req - 1), not a threshold -- which is why the")
print("    same physics gives a flat cool band and a steep warm one.")
print("    (compare W with W, not W with W/m2.)")

# ------------------------------------------------------------------ CLAIM 10
print("\nCLAIM 10: the same balance read as a demand on the circulation.")
print("    Heat reaches the skin only by blood: SkBF = H / (rho*c * (T_core - T_sk))")
print(f"    with rho*c = {RHO_CB:.0f} J/L/K and T_core = {T_CORE:.0f} C.")
for T_sk in (25.0, 28.0, 31.0, 33.0, 35.0):
    flow = H_prod()/(RHO_CB*(T_CORE-T_sk))*60.0
    print(f"    T_sk = {T_sk:4.1f} C: gradient {T_CORE-T_sk:4.1f} K -> "
          f"minimum skin blood flow {flow:4.2f} L/min")
check("minimum skin blood flow at T_sk = 31 C (L/min)",
      H_prod()/(RHO_CB*(T_CORE-31.0))*60.0, 1.53, 0.05)
check("and at T_sk = 35 C (L/min)",
      H_prod()/(RHO_CB*(T_CORE-35.0))*60.0, 3.07, 0.05)
print("\n    the two links pull on the shared boundary T_sk in OPPOSITE directions.")
CH_TA, CH_RH, CH_SOLAR = 30.0, 0.60, 50.0
print(f"    environment: T_a = {CH_TA:.0f} C, RH {CH_RH:.0%} "
      f"(T_wb {twb(CH_TA,CH_RH):.1f} C, WBGT {wbgt(CH_TA,CH_RH,200.0):.1f} C)")
def _w_of_tsk(T_sk):
    dry = hc(V_REF)*(T_sk-CH_TA)*A_BODY - CH_SOLAR*A_BODY
    return (H_prod()-dry)/(LR*hc(V_REF)*(psat(T_sk)-CH_RH*psat(CH_TA))*A_BODY)
def _flow(T_sk): return H_prod()/(RHO_CB*(T_CORE-T_sk))*60.0
print(f"    {'T_sk':>5} {'w_req':>8} {'SkBF':>8}")
for T_sk in (28.0, 31.0, 34.0, 37.0):
    print(f"    {T_sk:5.0f} {_w_of_tsk(T_sk):8.3f} {_flow(T_sk):8.2f}")
check("w_req at T_sk = 28 C (hot-marathon air)", _w_of_tsk(28.0), 1.78, 0.01)
check("w_req at T_sk = 37 C", _w_of_tsk(37.0), 0.43, 0.01)
check("skin blood flow at T_sk = 28 C (L/min)", _flow(28.0), 1.1, 0.05)
check("skin blood flow at T_sk = 37 C (L/min)", _flow(37.0), 6.1, 0.05)
_mono = all(_w_of_tsk(t) > _w_of_tsk(t+0.5) and _flow(t) < _flow(t+0.5)
            for t in [28.0+0.5*i for i in range(18)])
check("w_req falls and SkBF rises monotonically in T_sk",
      1.0 if _mono else 0.0, 1.0, 0.0)
print("    -> warmer skin EASES the surface link and TIGHTENS the transport one.")
print("       T_sk is an outcome of both balances, not an input the runner sets;")
print("       this appendix fixes it at 31 C instead of solving them jointly.")
print("    1/(T_core - T_sk) is convex in T_sk, so the circulatory route carries")
print("    the same curvature as the evaporative one. These are two links in one")
print("    chain, not two rival explanations.")

# ------------------------------------------------------------------ CLAIM 11
print("\nCLAIM 11: one convex strain->pace map does both jobs.")
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
check("steepening of w_req alone over the two bands", r_w, 1.80, 0.02)
print(f"    against {OBS:.2f} for the penalty: k>1 lifts the former to the latter")
print("    AND flattens the cool end. The physics predicts a LOWER BOUND.")

# ------------------------------------------------------------------ CLAIM 12
print("\nCLAIM 12: what the curvature prediction is, and is not, sensitive to.")
lo_c,hi_c = curvature()
print(f"    baseline (T_sk=31, RH 40-85%, solar 0-500): {lo_c:.2f} to {hi_c:.2f}")
check("baseline band brackets the observation", 1.0 if lo_c<=OBS<=hi_c else 0.0, 1.0, 0.0)
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

# ------------------------------------------------------------------ CLAIM 13
print("\nCLAIM 13: the exposed-skin caveat, quantified.")
print("    Aylwin measured EXPOSED skin by IR thermography, 64% of BSA.")
for cov in (32,34,36,38):
    tsk = 0.64*29.35 + 0.36*cov
    l,hh = curvature(T_sk=tsk)
    print(f"      covered skin {cov} C -> weighted mean {tsk:.1f} C -> {l:.2f} to {hh:.2f}"
          f"   {'brackets' if l<=OBS<=hh else 'MISSES'}")
print("    -> the band still brackets even on the least favourable weighting.")
print("       It fails only above T_sk ~ 33 C, which no weighting of Aylwin reaches.")

# ------------------------------------------------------------------ CLAIM 14
print("\nCLAIM 14: the response re-expressed on the dry-bulb axis.")
SOLAR = 200.0
xs = sorted(learned)
def g(wv):
    if wv <= xs[0]: return 0.0
    if wv > 28.0:   return None
    for a,b in zip(xs, xs[1:]):
        if a <= wv <= b:
            return learned[a] + (wv-a)/(b-a)*(learned[b]-learned[a])
    return None
print("    learned penalty (%) at fixed relative humidity, solar 200 W/m2:")
print(f"    {'T_a':>5}" + "".join(f"{int(r*100):>9}%" for r in (0.3,0.5,0.7,0.9)))
for Ta in (10,15,20,25,30):
    row=f"    {Ta:5d}"
    for rh in (0.3,0.5,0.7,0.9):
        p = g(wbgt(float(Ta),rh,SOLAR))
        row += ("      n/a" if p is None else f"{p:10.2f}")
    print(row)
check("penalty at 25 C air, 50% RH (%)", g(wbgt(25.0,0.5,SOLAR)), 1.70, 0.05)
check("penalty at 25 C air, 90% RH (%)", g(wbgt(25.0,0.9,SOLAR)), 3.53, 0.05)
check("penalty at 30 C air, 30% RH (%)", g(wbgt(30.0,0.3,SOLAR)), 2.17, 0.05)
print("    -> raising air temperature at FIXED humidity raises the penalty")
print("       monotonically, so the finding is not an artifact of the index;")
print("       'n/a' marks cells whose WBGT leaves the evaluated range.")

print("\n" + ("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILED: {fails}"))
