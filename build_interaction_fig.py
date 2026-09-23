"""build_interaction_fig.py

fig_hr_wbgt_interaction -- the three-way structure between heat, effort and pace.

METHOD, and why it is not a fitted surface. The linear heart-rate-by-heat interaction is not
distinguishable from zero (t=0.4 on the v3 corpus) because the steepening is concentrated in the hardest efforts
rather than being linear in heart rate. Rendering a smooth surface from that coefficient would
be drawing a shape the data does not support. So the surface here is MEASURED, not fitted:

  1. residualize log pace within runner on distance, elevation and fitness. Heart rate is
     deliberately NOT controlled for, since it is an axis of the plot.
  2. bin the residuals on a (WBGT x heart rate) grid.
  3. express each cell relative to that heart-rate row's own value at WBGT 15, so the surface
     shows the heat penalty AT a given effort rather than the pace level of that effort
     (harder efforts are faster, which would otherwise dominate the picture).
  4. mask cells with fewer than MIN_N observations, and show the support separately, so the
     reader can see where the surface is real and where the data thins out.

Run: /home/bb/test_env/bin/python3 build_interaction_fig.py
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

INK,MUTE,GRID="#1a1a1a","#5c5b57","#e6e5df"
BLUE,VERM,GREEN="#0072B2","#D55E00","#009E73"
plt.rcParams.update({"figure.facecolor":"white","axes.facecolor":"white","font.size":9.5,
 "axes.edgecolor":MUTE,"axes.linewidth":0.8,"xtick.color":MUTE,"ytick.color":MUTE,
 "text.color":INK,"axes.labelcolor":INK,"axes.titlecolor":INK,
 "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.7,
 "axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False,"legend.fontsize":8.2})

MIN_N = 150          # cells thinner than this are not drawn
import os
# v3 rerun: corpus overridable (INTERACTION_ACT) so the surface can be rebuilt on the enlarged table
d=pd.read_parquet(os.environ.get("INTERACTION_ACT","/weather/data/activities_model_clean_v3.parquet"))
d=d.dropna(subset=["pace_min_km","wbgt","avg_hr","distance_km","elev_gain_per_km","fit_pace_30d"])
d=d[(d.avg_hr.between(60,210))&(d.pace_min_km.between(2.5,11.5))&(d.wbgt.between(2,30))]
d["lp"]=np.log(d.pace_min_km)

# --- 1. within-runner residual, controlling everything EXCEPT heat and heart rate ---
g=d.groupby("user_id")
for c in ["lp","distance_km","elev_gain_per_km","fit_pace_30d"]:
    d[c+"_w"]=d[c]-g[c].transform("mean")
X=np.column_stack([np.ones(len(d)),d.distance_km_w,d.elev_gain_per_km_w,d.fit_pace_30d_w])
beta,*_=np.linalg.lstsq(X,d.lp_w.values,rcond=None)
d["resid"]=(d.lp_w.values-X@beta)*100          # percent pace, within runner

# --- 2. bin on the (WBGT, HR) grid ---
WE=np.arange(6,30.1,2.0)                        # WBGT edges
HE=np.array([110,125,135,145,155,165,185])      # HR edges
d["wi"]=np.digitize(d.wbgt,WE)-1
d["hi"]=np.digitize(d.avg_hr,HE)-1
d=d[(d.wi>=0)&(d.wi<len(WE)-1)&(d.hi>=0)&(d.hi<len(HE)-1)]
cell=d.groupby(["hi","wi"]).resid.agg(["mean","size"]).reset_index()
Z=np.full((len(HE)-1,len(WE)-1),np.nan); N=np.zeros_like(Z)
for _,r in cell.iterrows():
    Z[int(r.hi),int(r.wi)]=r["mean"]; N[int(r.hi),int(r.wi)]=r["size"]

# --- 3. re-base each HR row to its own value at the WBGT 14-16 bin => heat penalty at that effort
ref=np.argmin(np.abs((WE[:-1]+WE[1:])/2-15))
for i in range(Z.shape[0]):
    Z[i]-=Z[i,ref]
Z[N<MIN_N]=np.nan
wc=(WE[:-1]+WE[1:])/2; hc=(HE[:-1]+HE[1:])/2
print(f"grid {Z.shape}, cells kept {np.isfinite(Z).sum()} of {Z.size} (min n={MIN_N})")
print(f"penalty at WBGT {wc[-1]:.0f}: easiest row {Z[0,-1]:+.2f}%, hardest row {Z[-1,-1]:+.2f}%")

fig=plt.figure(figsize=(13.0,5.6))

# ================= A: the measured surface, in 3D =================
ax=fig.add_subplot(1,2,1,projection="3d")
WW,HH=np.meshgrid(wc,hc)
Zm=np.ma.masked_invalid(Z)
vmin,vmax=np.nanmin(Z),np.nanmax(Z)
surf=ax.plot_surface(WW,HH,Zm,cmap=cm.YlOrRd,rstride=1,cstride=1,
                     linewidth=0.3,edgecolors="white",antialiased=True,
                     vmin=vmin,vmax=vmax,alpha=0.96)
# shadow contours on the floor make the shape legible without adding any new claim
ax.contourf(WW,HH,Zm,zdir="z",offset=vmin-1.6,levels=10,cmap=cm.YlOrRd,alpha=0.55)
ax.scatter(WW[np.isfinite(Z)],HH[np.isfinite(Z)],Z[np.isfinite(Z)],
           s=6,color=INK,alpha=0.40,depthshade=False)
ax.set_zlim(vmin-1.6,vmax+0.4)
ax.set_xlabel("WBGT (°C)",labelpad=8)
ax.set_ylabel("mean heart rate (bpm)",labelpad=8)
ax.set_zlabel("pace penalty vs WBGT 15 (%)",labelpad=6)
ax.set_title("A.  Measured penalty surface, within runner",loc="left",fontsize=10.5)
ax.view_init(elev=27,azim=-119)
for pane in (ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane): pane.set_alpha(0.02)
ax.tick_params(labelsize=8)
fig.colorbar(surf,ax=ax,shrink=0.52,pad=0.11,label="% slower than the same effort at WBGT 15")

# ================= B: support =================
ax=fig.add_subplot(1,2,2)
Nm=np.ma.masked_where(N==0,N)
im=ax.pcolormesh(WE,HE,Nm,cmap="Blues",
                 norm=matplotlib.colors.LogNorm(vmin=50,vmax=np.nanmax(N)))
for i in range(len(hc)):
    for j in range(len(wc)):
        if N[i,j]==0 or N[i,j]>=MIN_N: continue
        ax.plot(wc[j],hc[i],marker="x",ms=10,color=VERM,mew=2.0)
        ax.text(wc[j],hc[i]-4,f"{int(N[i,j])}",ha="center",va="top",fontsize=7,color=VERM)
ax.set_xlabel("WBGT (°C)"); ax.set_ylabel("mean heart rate (bpm)")
ax.set_title(f"B.  Runs per cell (× = under {MIN_N}, omitted from A)",loc="left",fontsize=10.5)
ax.grid(False)
fig.colorbar(im,ax=ax,shrink=0.85,pad=0.02,label="runs in cell")
fig.tight_layout()
for e in ("pdf","png"):
    fig.savefig(f"figures/fig_hr_wbgt_interaction.{e}",bbox_inches="tight",dpi=150)
print("saved figures/fig_hr_wbgt_interaction.{pdf,png}")
print(f"hot+hard corner support: {int(N[-1,-1])} runs")
