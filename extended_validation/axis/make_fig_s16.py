"""Supp. Fig. S16 - the innate-hyperactive axis."""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps

matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8,
                            "axes.linewidth": .6, "xtick.major.width": .6,
                            "ytick.major.width": .6})

S = "/Users/yc4229/Downloads/Journal of Autoimmunity/Supplementary Tables.xlsx"
sc = pd.read_csv("axis_scores.csv")
load = pd.read_csv("axis_loadings.csv", index_col=0).iloc[:, 0]
clin = pd.read_csv("axis_clinical_correlations.csv")
st2 = pd.read_excel(S, "ST2_immunotype_clinical").set_index("name")

CIT = {"IT1": "#d1495b", "IT2": "#e3a72f", "IT3": "#5d76cb"}
CAB = {"Jo-1": "#8e5ea2", "PL-12": "#2a9d8f", "EJ": "#e76f51", "PL-7": "#264653"}

fig = plt.figure(figsize=(7.2, 7.4))
gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.0, 1.0], hspace=.72, wspace=.34)


def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# (a) patients ranked along the axis --------------------------------------
ax = fig.add_subplot(gs[0, :])
d = sc.sort_values("axis_PC1")
y = np.arange(len(d))
ax.barh(y, d.axis_PC1, color=[CIT[i] for i in d.immunotype], height=.62, zorder=3)
for i, (_, r) in enumerate(d.iterrows()):
    off = .35 if r.axis_PC1 > 0 else -.35
    ax.text(r.axis_PC1 + off, i, f"{r.patient} ({r.antibody})",
            va="center", ha="left" if r.axis_PC1 > 0 else "right", fontsize=6.3)
ax.axvline(0, color="#888", lw=.7)
ax.set_yticks([]); ax.set_xlim(-9.5, 11.5)
ax.set_xlabel("innate-hyperactive axis  (PC1 of the 63-feature patient matrix, 34.2% of variance)",
              fontsize=7.4)
ax.set_title("a  Patients occupy a continuum, not three discrete groups", fontsize=7.9, pad=4)
h = [plt.Rectangle((0, 0), 1, 1, color=CIT[k]) for k in CIT]
ax.legend(h, [f"{k} region" for k in CIT], fontsize=6.2, frameon=False,
          loc="lower right", ncol=3)
despine(ax)

# (b) orthogonality to serology -------------------------------------------
ax = fig.add_subplot(gs[1, 0])
order = ["Jo-1", "PL-12", "EJ", "PL-7"]
rng = np.random.default_rng(1)
for i, a in enumerate(order):
    v = sc[sc.antibody == a].axis_PC1.values
    ax.scatter(np.full(len(v), i) + rng.uniform(-.13, .13, len(v)), v,
               s=26, c=CAB[a], zorder=3, linewidths=0)
    ax.hlines(np.median(v), i - .28, i + .28, color="k", lw=1.3, zorder=4)
jo = sc[sc.antibody == "Jo-1"].axis_PC1
ax.annotate("", xy=(-.30, jo.max()), xytext=(-.30, jo.min()),
            arrowprops=dict(arrowstyle="<->", lw=1.0, color="#8e5ea2"))
ax.text(-.42, jo.mean(), "anti-Jo-1 spans\n94% of the axis", rotation=90, va="center",
        ha="center", fontsize=5.9, color="#8e5ea2")
ax.set_xticks(range(4)); ax.set_xticklabels(order, fontsize=6.8)
ax.set_ylabel("axis score", fontsize=7.4)
ax.set_xlim(-0.95, 3.4)
ax.axhline(0, color="#ddd", lw=.5, ls=":")
ax.set_title("b  Axis is orthogonal to serology\n     Kruskal–Wallis P=0.95", fontsize=7.9, pad=4)
despine(ax)

# (c) bootstrap rank stability --------------------------------------------
ax = fig.add_subplot(gs[1, 1])
d2 = sc.sort_values("axis_PC1", ascending=False)
xx = np.arange(len(d2))
ax.errorbar(xx, d2.boot_mean_rank, yerr=d2.boot_rank_sd, fmt="o", ms=4,
            lw=1.0, capsize=2.4, color="#333", zorder=3)
ax.plot(xx, np.arange(len(d2), 0, -1), ls="--", lw=.8, color="#bbb", zorder=1,
        label="observed order")
ax.set_xticks(xx); ax.set_xticklabels(d2.patient, rotation=90, fontsize=6.2)
ax.set_ylabel("bootstrap rank on the axis", fontsize=7.4)
ax.set_title("c  Rank order is stable (B=2,000)\n     mean SD 1.37 vs 2.87 if random",
             fontsize=7.9, pad=4)
ax.legend(fontsize=6.0, frameon=False, loc="upper right")
despine(ax)

# (d,e) clinical correlations ---------------------------------------------
for k, (v, lab) in enumerate([("ESR", "ESR (mm h⁻¹)"), ("DA_score", "composite disease activity")]):
    ax = fig.add_subplot(gs[2, k])
    yv = st2[v].reindex(sc.set_index("patient").index).astype(float)
    xv = sc.set_index("patient").axis_PC1
    ok = yv.notna()
    row = clin[clin.variable == v].iloc[0]
    for pat in xv[ok].index:
        ax.scatter(xv[pat], yv[pat], s=30, zorder=3, linewidths=0,
                   c=CIT[sc.set_index("patient").immunotype[pat]])
    b, a0 = np.polyfit(xv[ok], yv[ok], 1)
    xs = np.linspace(xv[ok].min(), xv[ok].max(), 20)
    ax.plot(xs, a0 + b * xs, color="#666", lw=1.0, ls="--", zorder=2)
    ax.set_xlabel("innate-hyperactive axis", fontsize=7.4)
    ax.set_ylabel(lab, fontsize=7.4)
    ax.set_title(f"{'de'[k]}  ρ={row.rho:+.2f}, P={row.p:.2f}, n={int(row.n)}\n"
                 f"     LOO range {row.loo_min:+.2f} to {row.loo_max:+.2f}, sign-stable",
                 fontsize=7.9, pad=4)
    despine(ax)

fig.suptitle("Supp. Fig. S16  The innate-hyperactive axis: a continuous ordering of patients "
             "that is\northogonal to autoantibody class", fontsize=8.6, y=.985)
fig.savefig("SuppFig_S16_axis.pdf", bbox_inches="tight", dpi=300)
fig.savefig("SuppFig_S16_axis.png", bbox_inches="tight", dpi=220)
print("wrote SuppFig_S16_axis.pdf/.png")
