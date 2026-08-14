"""build_interaction_fig.py

Two products for the manuscript:
  1. fig_hr_wbgt_interaction  -- the three-way structure between heart rate, pace and heat.
     The heat penalty is not a single slope: it steepens with the effort a runner is holding.
     Estimated WITHIN runner (each runner's own means removed) so it is not a comparison of
     fast runners to slow ones, and with distance/elevation/fitness controlled.
  2. the data-support numbers behind the 28 C ceiling on all reported curves.
Run: /home/bb/test_env/bin/python3 build_interaction_fig.py
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

INK,MUTE,GRID="#1a1a1a","#5c5b57","#e6e5df"
BLUE,VERM,GREEN,GREY="#0072B2","#D55E00","#009E73","#9a998f"
plt.rcParams.update({"figure.facecolor":"white","axes.facecolor":"white","font.size":9.5,
 "axes.edgecolor":MUTE,"axes.linewidth":0.8,"xtick.color":MUTE,"ytick.color":MUTE,
 "text.color":INK,"axes.labelcolor":INK,"axes.titlecolor":INK,"axes.grid":True,
 "grid.color":GRID,"grid.linewidth":0.7,"axes.spines.top":False,"axes.spines.right":False,
 "legend.frameon":False,"legend.fontsize":8.2})

d=pd.read_parquet("/weather/data/activities_model_clean.parquet")
d=d.dropna(subset=["pace_min_km","wbgt","avg_hr","rel_effort","distance_km",
                   "elev_gain_per_km","fit_pace_30d"])
d=d[(d.avg_hr.between(60,210))&(d.pace_min_km.between(2.5,11.5))&(d.wbgt<=30)]
d["lp"]=np.log(d.pace_min_km)
d["h"]=np.maximum(0.0,d.wbgt-15.0)              # heat above the flat band

# ---- within-runner demeaning: removes "who is fast" and any fixed runner trait ----
g=d.groupby("user_id")
for c in ["lp","h","avg_hr","distance_km","elev_gain_per_km","fit_pace_30d","wbgt"]:
    d[c+"_w"]=d[c]-g[c].transform("mean")
d["hr_z"]=(d.avg_hr-d.avg_hr.mean())/d.avg_hr.std()
d["hr_z_w"]=d.hr_z-g["hr_z"].transform("mean")
d["h_x_hr"]=d.h_w*d.hr_z_w                       # the interaction of interest

X=np.column_stack([np.ones(len(d)),d.h_w,d.hr_z_w,d.h_x_hr,
                   d.distance_km_w,d.elev_gain_per_km_w,d.fit_pace_30d_w])
beta,*_=np.linalg.lstsq(X,d.lp_w.values,rcond=None)
res=d.lp_w.values-X@beta
# runner-clustered SEs
uid=d.user_id.astype(str).values; XtX_inv=np.linalg.inv(X.T@X); meat=np.zeros_like(XtX_inv)
for u in np.unique(uid):
    m=uid==u; Xu=X[m]; ru=res[m]; s=Xu.T@ru; meat+=np.outer(s,s)
V=XtX_inv@meat@XtX_inv; se=np.sqrt(np.diag(V))
print("within-runner, log pace (percent per degree above WBGT 15):")
print(f"  heat slope at mean HR        {beta[1]*100:+.4f} %/C   se {se[1]*100:.4f}")
print(f"  heat x HR interaction        {beta[3]*100:+.4f} %/C per HR sd   se {se[3]*100:.4f}"
      f"   t={beta[3]/se[3]:.1f}")

# ---- Panel A: heat slope estimated separately within HR quintiles ----
d["hrq"]=pd.qcut(d.avg_hr,5,labels=False)
qs=[];
for q in range(5):
    s=d[d.hrq==q]
    Xq=np.column_stack([np.ones(len(s)),s.h_w,s.distance_km_w,s.elev_gain_per_km_w,s.fit_pace_30d_w])
    bq,*_=np.linalg.lstsq(Xq,s.lp_w.values,rcond=None)
    rq=s.lp_w.values-Xq@bq; uq=s.user_id.astype(str).values
    A=np.linalg.inv(Xq.T@Xq); M=np.zeros_like(A)
    for u in np.unique(uq):
        m=uq==u; sv=Xq[m].T@rq[m]; M+=np.outer(sv,sv)
    seq=np.sqrt(np.diag(A@M@A))
    qs.append((s.avg_hr.mean(),bq[1]*100,seq[1]*100,len(s)))
    print(f"  HR quintile {q+1} (mean {s.avg_hr.mean():5.1f} bpm, n={len(s):6,}): "
          f"{bq[1]*100:+.4f} %/C  se {seq[1]*100:.4f}")

fig,axes=plt.subplots(1,2,figsize=(11.6,4.6))
ax=axes[0]
hr=[q[0] for q in qs]; sl=[q[1] for q in qs]; er=[1.96*q[2] for q in qs]
ax.errorbar(hr,sl,yerr=er,color=BLUE,lw=2,marker="o",ms=6,mfc=BLUE,mec="white",
            mew=0.8,capsize=3,elinewidth=1)
ax.axhline(0,color=INK,lw=0.9)
ax.set_xlabel("mean heart rate during the run (bpm)")
ax.set_ylabel("pace penalty per °C above WBGT 15 (%)")
ax.set_title("A.  Heat slope by effort, within runner", loc="left", fontsize=10)
ax.annotate(f"interaction {beta[3]*100:+.3f} %/°C per HR sd (t={beta[3]/se[3]:.1f})",
            (0.03,0.94), xycoords="axes fraction", fontsize=8, color=MUTE)

# ---- Panel B: cumulative penalty implied by the DIRECTLY ESTIMATED quintile slopes.
# Deliberately not a smooth surface from the linear interaction term: that coefficient is
# not distinguishable from zero (t=0.8) because the steepening is concentrated in the top
# quintile rather than being linear in HR. Plotting only what was estimated.
ax=axes[1]
W=np.linspace(15,28,100)
for (hrm,slope,sem,nq),col,lab in zip([qs[0],qs[2],qs[4]],[BLUE,GREEN,VERM],
        ["easiest fifth","middle fifth","hardest fifth"]):
    y=slope*np.maximum(0,W-15)
    ax.plot(W,y,color=col,lw=2.2,label=f"{lab} ({hrm:.0f} bpm)")
    ax.fill_between(W,(slope-1.96*sem)*np.maximum(0,W-15),
                      (slope+1.96*sem)*np.maximum(0,W-15),color=col,alpha=0.13,lw=0)
ax.set_xlabel("WBGT (°C)"); ax.set_ylabel("pace penalty vs WBGT 15 (%)")
ax.set_title("B.  Penalty by effort level", loc="left", fontsize=10)
ax.legend(loc="upper left")
fig.tight_layout()
for e in ("pdf","png"): fig.savefig(f"figures/fig_hr_wbgt_interaction.{e}",bbox_inches="tight",dpi=150)
print("\nsaved figures/fig_hr_wbgt_interaction.{pdf,png}")

# ---- data support behind the 28 C ceiling ----
raw=pd.read_parquet("/weather/data/activities_model_clean.parquet").dropna(subset=["wbgt"])
print("\ndata support by WBGT (unfiltered corpus, n=%d):" % len(raw))
for t in (24,26,28,30,32):
    m=raw.wbgt>t
    print(f"  > {t} C : {m.sum():7,} runs ({m.mean()*100:5.3f}%)  {raw[m].user_id.nunique():5,} runners")
