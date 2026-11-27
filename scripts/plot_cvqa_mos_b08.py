#!/usr/bin/env python3
"""Regenerate B08 release illustrations and an offline research-only explorer.

Usage: python scripts/plot_cvqa_mos_b08.py
Requires matplotlib/numpy for generation only. No raw research data or network access.
The historical B07 coefficients below are used only for the comparison figure.
"""
from pathlib import Path
import json
import sys
import math
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from pixvqa_cvqahdr import cvqa_mos_b08 as m

OUT = ROOT / "docs/dev/cvqa_mos_b08"
ROUTES = [
    ("pq", 2160, "HDR PQ 2160p", "#174b91"),
    ("sdr", 2160, "SDR 2160p", "#16806c"),
    ("pq", 1080, "HDR PQ 1080p", "#d08023"),
    ("sdr", 1080, "SDR 1080p", "#8b52a7"),
]
X = np.linspace(0, 15, 1501)
WEIGHT = m.WEIGHT

# Historical, superseded full-range B07 scenario (not the localized 15% proposal).
# Frozen provenance: H1_HUMAN7_FULLRANGE_ALL4_EXTRAPOLATION_20261124.
B07_THETA = (-0.1387460215555705, 0.6449941229765033, -0.43841398782780305)
B07_BIAS = (-0.2622676545260202, -0.11600537071745345, -0.909486807431632,
            0.7286214904189257, 0.2991035559648164, 0.26003478629136356)
B07_SIGMA = 0.6977632247313065
B07_RL, B07_SLOPE = -0.3356560871471081, 0.23201992953580938

def sigmoid(z):
    return 1/(1+math.exp(-z)) if z >= 0 else math.exp(z)/(1+math.exp(z))

def b07_expected(x):
    # The original B07 comparison used unclipped A2 above its observed support.
    # Here use the canonical bounded A2 baseline for a like-for-like comparison.
    a = m.baseline(x, "pq", 2160)
    mu = a + B07_THETA[0] + B07_THETA[1]*(x-9.15) + B07_THETA[2]*max(0,x-9.25)
    return 1 + sum(sigmoid((mu+b-k)/B07_SIGMA) for b in B07_BIAS
                   for k in (1.5,2.5,3.5,4.5))/len(B07_BIAS)

B07_D10 = b07_expected(10)-m.baseline(10,"pq",2160)

def b07_ratio(x):
    if x < m.L:
        return B07_RL+B07_SLOPE*math.expm1(x-m.L)
    if x == 10:
        return 0.
    a = m.baseline(x, "pq", 2160)
    return (b07_expected(x)-a-B07_D10)/(a-1)

def bounded(y):
    return np.clip(y,1,5)

def save(fig, name):
    for suffix in ("png","svg"):
        fig.savefig(OUT / (name+"."+suffix), dpi=190, bbox_inches="tight",
                    facecolor="white", metadata={"Date":None} if suffix=="svg" else None)
    svg = OUT / (name+".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n", encoding="utf-8")
    plt.close(fig)

def format_ax(ax, ylabel="Estimated MOS (1–5)", ylim=None):
    ax.axvspan(m.L,m.H,color="#c8d9ed",alpha=.32,zorder=0)
    ax.axvline(10,color="#68717b",ls=":",lw=1)
    ax.set(xlim=(0,15),xlabel="Canonical clip-level CVQA",ylabel=ylabel)
    if ylim is not None: ax.set_ylim(*ylim)
    ax.grid(alpha=.18)
    ax.spines[["top","right"]].set_visible(False)

def main():
    m.verified_digest()
    OUT.mkdir(parents=True,exist_ok=True)
    data = json.loads((OUT/"validation_summary.json").read_text())
    sweep=data["primary"]
    bases = [np.array([m.baseline(float(x),t,h) for x in X]) for t,h,_,_ in ROUTES]
    b08s = [np.array([m.predict_mos(float(x),transfer=t,encoded_height=h) for x in X])
            for t,h,_,_ in ROUTES]
    b07r=np.array([b07_ratio(float(x)) for x in X])
    b07s=[bounded(1+(a-1)*(1+WEIGHT*b07r)) for a in bases]
    rho=np.array([m.rho(float(x)) for x in X])

    plt.rcParams.update({"font.size":11,"axes.titlesize":12,"figure.titlesize":16,
                         "legend.frameon":False,"font.family":"DejaVu Sans","svg.hashsalt":m.VERSION})
    fig,ax=plt.subplots(figsize=(11,6.3),layout="constrained")
    for (_,_,label,color),a,b in zip(ROUTES,bases,b08s):
        ax.plot(X,b,color=color,lw=2.5,label=label)
        ax.plot(X,a,color=color,lw=1.1,ls="--",alpha=.7)
    format_ax(ax,ylim=(1,5.08))
    ax.set_title("Canonical B08 — all four routes, w = 0.05",loc="left",weight="bold")
    ax.legend(loc="upper left",ncol=2)
    ax.text(.015,.02,"Solid: B08    Dashed: original A2    Shaded: H1 HDR2160 score interval\n"
            "H1 evidence does not directly validate the other routes or extrapolated scores.",
            transform=ax.transAxes,fontsize=9,color="#344454",va="bottom")
    save(fig,"B08_four_curves")

    fig,axs=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True,layout="constrained")
    for ax,(_,_,label,color),a,b,c in zip(axs.flat,ROUTES,bases,b07s,b08s):
        ax.plot(X,a-a,ls="--",color="#4c5761",lw=1.5,label="Original A2")
        ax.plot(X,b-a,color="#ad7d32",ls="--",lw=2,label="B07, w=0.05")
        ax.plot(X,c-a,color=color,lw=2.5,label="B08, w=0.05")
        format_ax(ax,ylabel="Δ MOS relative to A2")
        ax.set_title(label,loc="left",weight="bold")
        ax.legend(loc="lower right",fontsize=9)
    fig.suptitle("A2 / B07 / B08 — matched 5% correction\n"
                 "Shading: observed H1 HDR2160 CVQA interval; anchor at CVQA=10")
    save(fig,"A2_B07_B08_comparison")

    fig,axs=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True,layout="constrained")
    for ax,(_,_,label,color),a,b,c in zip(axs.flat,ROUTES,bases,b07s,b08s):
        ax.plot(X,a,ls=":",color="#4c5761",lw=1.8,label="Original A2")
        ax.plot(X,b,color="#ad7d32",ls="--",lw=2,label="B07, w=0.05")
        ax.plot(X,c,color=color,lw=2.3,label="B08, w=0.05")
        format_ax(ax,ylim=(1,5.05))
        ax.set_title(label,loc="left",weight="bold")
        ax.legend(loc="upper left",fontsize=9)
    fig.suptitle("Absolute estimated-MOS curves — A2 / B07 / B08")
    save(fig,"A2_B07_B08_absolute")

    weights=sorted(map(float,sweep))
    rows=[sweep[str(w) if w else "0"] for w in weights]
    zero=rows[0]
    fig,axs=plt.subplots(1,2,figsize=(12,5.6),layout="constrained")
    for t,h,label,color in ROUTES:
        key=f"{t}-{h}"
        axs[0].plot(np.array(weights)*100,[100*(r[key]/zero[key]-1) for r in rows],
                    marker="o",ms=4,color=color,label=label)
    axs[0].plot(np.array(weights)*100,[100*(r["overall"]/zero["overall"]-1) for r in rows],
                marker="s",ms=3,ls="--",color="#32383f",label="Overall (116 clips)")
    axs[0].axhline(1,color="#a64c38",ls=":",lw=1.3,label="Approx. +1% guardrail")
    for key,label,color in [("h1_individual","Individual ACR","#174b91"),("h1_clip_mean","Clip-mean ACR","#16806c")]:
        axs[1].plot(np.array(weights)*100,[100*(r[key]/zero[key]-1) for r in rows],
                    marker="o",ms=4,color=color,label=label)
    for ax,title in zip(axs,["Historical TV123 (0–100 targets)","H1 direct ACR (1–5 targets)"]):
        ax.axhline(0,color="#8d98a3",lw=1)
        ax.axvline(5,color="#303840",ls="--",lw=1.5,label="Selected 5%")
        ax.set(title=title,xlabel="H1 correction weight (%)",ylabel="Source-balanced MAE change vs A2 (%)")
        ax.grid(alpha=.18)
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(fontsize=8.5,loc="best")
    fig.suptitle("Joint source-held-out trade-off — seven screened ballots from eight collected\n"
                 "Lower is better; 5% chosen despite the SDR1080 +1.36% regression")
    save(fig,"B08_weight_tradeoff")

    # Exact tabulated evaluations; no raw observer or source-level data.
    with (OUT/"B08_curve_samples.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(["cvqa","rho"]+[f"{t}_{h}_{a}" for t,h,_,_ in ROUTES for a in ("A2","B07_w005","B08_w005")])
        for i,x in enumerate(X):
            writer.writerow([float(x),float(rho[i])]+[float(v[i]) for triple in zip(bases,b07s,b08s) for v in triple])
    payload={"version":m.VERSION,"fixedWeight":WEIGHT,"minX":0,"maxX":15,"step":0.01,
             "support":[m.L,m.H],"rho":rho.tolist(),
             "routes":[{"label":label,"color":color,"base":a.tolist(),"canonical":b.tolist()}
                       for (_,_,label,color),a,b in zip(ROUTES,bases,b08s)]}
    html=(ROOT/"scripts/cvqa_mos_b08_explorer_template.html").read_text(encoding="utf-8")
    (OUT/"B08_curve_explorer.html").write_text(html.replace("__B08_DATA__",json.dumps(payload,separators=(",",":"))),encoding="utf-8")
    print("Generated",OUT)

if __name__=="__main__":
    main()
