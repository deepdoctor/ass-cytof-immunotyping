"""
Revised Fig. 2 (revision R1) -- the innate-hyperactive axis.

Identical in content to the submitted Fig. 2 except that panel b now reports the
effect size of autoantibody class on the axis (eta^2 with 95% CI, Reviewer 4,
comment 4) alongside the Kruskal-Wallis P value.

Inputs
  cytof_output_nature/SuppTable_feature_matrix.csv   (= Supp. Table ST1)
  revision_R1/r1_analyses_results.json               (eta^2 and CI from r1_analyses.py)
  Clinical values (LDH, ESR) as in Table 1 of the manuscript.

Run from the repository root:
  python revision_R1/fig2_axis_revised.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

OUT = "revision_R1"
plt.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})
CIT = {"IT1": "#C0443C", "IT2": "#E0A458", "IT3": "#3C6CA8"}
CAB = {"Jo-1": "#7858A4", "PL-12": "#289C8C", "EJ": "#E0785C", "PL-7": "#2C4858"}
INNATE = ["Classical Monocyte", "Inflammatory Monocyte", "Non-classical Monocyte",
          "NK cell", "mDC", "pDC", "Activation_score", "Cytotoxic_score"]

# ── axis (PC1 of the z-scored 66-feature matrix; as in r1_analyses.py) ──────
feat = pd.read_csv("cytof_output_nature/SuppTable_feature_matrix.csv")
pid = "P" + feat["pid"].astype(int).astype(str).str.zfill(2)
X = feat[list(feat.columns[5:])].set_axis(pid)
Z = (X - X.mean()) / X.std(ddof=1)
Zc = Z.values - Z.values.mean(0)
U, s, Vt = np.linalg.svd(Zc, full_matrices=False)
var1 = (s ** 2 / (s ** 2).sum())[0]
axis = pd.Series(U[:, 0] * s[0], index=pid)
if pd.Series(Vt[0], index=Z.columns)[INNATE].mean() < 0:
    axis = -axis
ab = pd.Series(feat["antibody"].values, index=pid)
region = pd.Series(feat["immunotype"].values, index=pid)

# ── bootstrap rank stability (2,000 replicates; procedure of build_axis.py) ─
rng = np.random.default_rng(20260727)
B = 2000
ranks = np.zeros((B, len(axis)))
for b in range(B):
    idx = rng.choice(len(axis), len(axis), replace=True)
    Xb = Z.values[idx] - Z.values[idx].mean(0)
    sd = Xb.std(0, ddof=1); sd[sd == 0] = 1
    _, _, Vtb = np.linalg.svd(Xb / sd, full_matrices=False)
    proj = Zc @ Vtb[0]
    if np.corrcoef(proj, axis.values)[0, 1] < 0:
        proj = -proj
    ranks[b] = sps.rankdata(proj)
rank_mean = pd.Series(ranks.mean(0), index=pid)
rank_sd = pd.Series(ranks.std(0), index=pid)
rand_sd = np.sqrt((len(axis) ** 2 - 1) / 12)

# ── clinical values (Table 1) ───────────────────────────────────────────────
LDH = {"P01": 315, "P02": 374, "P03": 203, "P04": 195, "P05": 147,
       "P06": 202, "P07": 343, "P08": 755, "P09": 321, "P10": 492}
ESR = {"P01": 56, "P02": 13, "P03": 24, "P04": 7, "P05": 2,
       "P06": 8, "P07": 30, "P08": 36, "P09": 13, "P10": 28}
ldh = np.log10(pd.Series(LDH)[axis.index])
esr = pd.Series(ESR)[axis.index].astype(float)

# ── serology statistics ─────────────────────────────────────────────────────
H, pKW = sps.kruskal(*[axis[ab == a].values for a in ["Jo-1", "PL-12", "EJ", "PL-7"]])
ser = json.load(open(f"{OUT}/r1_analyses_results.json"))["serology"]
e2, (lo, hi) = ser["eta2"], ser["eta2_CI95"]
jo = axis[ab == "Jo-1"]
span = (jo.max() - jo.min()) / (axis.max() - axis.min())

stats_out = {
    "PC1_var_pct": round(100 * var1, 1), "P08": round(axis["P08"], 2), "P04": round(axis["P04"], 2),
    "KW_H": round(H, 2), "KW_P": round(pKW, 2), "eta2": e2, "eta2_CI95": [lo, hi],
    "antiJo1_span_pct": round(100 * span), "mean_rank_sd": round(rank_sd.mean(), 2),
    "random_rank_sd": round(rand_sd, 2),
    "LDH": [round(v, 3) for v in sps.spearmanr(axis, ldh)],
    "ESR": [round(v, 3) for v in sps.spearmanr(axis, esr)],
}
print(stats_out)


def despine(ax, left=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)


def panel(ax, letter, dx=-0.10, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            ha="left", va="bottom")


fig = plt.figure(figsize=(7.2, 7.2 * 3.398 / 6.0))
gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.0], hspace=0.62, wspace=1.6,
                      left=0.07, right=0.985, top=0.93, bottom=0.10)

# (a) ranked axis scores
ax = fig.add_subplot(gs[0, :4])
order = axis.sort_values()
y = np.arange(len(order))
ax.barh(y, order.values, color=[CIT[region[p]] for p in order.index], height=0.68, zorder=3)
for yi, p in zip(y, order.index):
    v = order[p]
    ax.text(v + (0.25 if v > 0 else -0.25), yi, f"{p}  {ab[p]}", va="center",
            ha="left" if v > 0 else "right", fontsize=6.2)
ax.axvline(0, color="black", lw=0.6, zorder=4)
ax.set_yticks([])
ax.set_xlim(-10, 11)
ax.set_xticks([-7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10])
ax.set_xticklabels(["−7.5", "−5.0", "−2.5", "0.0", "2.5", "5.0", "7.5", "10.0"])
ax.set_xlabel(f"Innate-hyperactive axis (PC1, {100 * var1:.1f}% of variance)")
despine(ax, left=False)
handles = [plt.Rectangle((0, 0), 1, 1, color=CIT[k]) for k in CIT]
ax.legend(handles, list(CIT), loc="lower right", frameon=False, fontsize=6, ncol=3,
          handlelength=1.2, columnspacing=1.0)
panel(ax, "a", dx=-0.02)

# (b) axis by autoantibody class
ax = fig.add_subplot(gs[0, 4:])
classes = ["Jo-1", "PL-12", "EJ", "PL-7"]
jit = np.random.default_rng(1)
for i, a in enumerate(classes):
    v = axis[ab == a].values
    ax.scatter(np.full(len(v), i) + jit.uniform(-0.12, 0.12, len(v)), v, s=11,
               color=CAB[a], zorder=3, linewidths=0)
    ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color="black", lw=1.1, zorder=4)
ax.axhline(0, color="#999999", lw=0.5, ls=":")
ax.set_xticks(range(4)); ax.set_xticklabels(classes)
ax.set_xlim(-0.5, 4.0)
ax.set_ylabel("Axis score")
ax.set_title(f"Kruskal–Wallis P = {pKW:.2f}\nη² = {e2:.2f} (95% CI {lo:.0f}–{hi:.2f})",
             fontsize=6.6, pad=4)
xa = 3.65
ax.annotate("", xy=(xa, jo.max()), xytext=(xa, jo.min()),
            arrowprops=dict(arrowstyle="<->", color=CAB["Jo-1"], lw=0.7))
ax.text(xa + 0.12, (jo.max() + jo.min()) / 2, f"anti-Jo-1 spans\n{100 * span:.0f}% of the axis",
        rotation=270, va="center", ha="left", fontsize=5.6, color=CAB["Jo-1"])
despine(ax)
panel(ax, "b", dx=-0.32)

# (c) bootstrap rank stability
ax = fig.add_subplot(gs[1, :2])
o = axis.sort_values(ascending=False).index
xx = np.arange(len(o))
ax.errorbar(xx, rank_mean[o], yerr=rank_sd[o], fmt="o", ms=3.2, color="#222222",
            ecolor="#222222", elinewidth=0.7, capsize=1.8, zorder=3)
ax.plot(xx, np.arange(len(o), 0, -1), ls="--", color="#999999", lw=0.6, zorder=1)
ax.set_xticks(xx); ax.set_xticklabels(o, rotation=90, fontsize=5.8)
ax.set_ylabel("Bootstrap rank")
ax.set_title(f"Mean rank s.d. {rank_sd.mean():.2f} ({rand_sd:.2f} if random)", fontsize=6.6, pad=4)
despine(ax)
panel(ax, "c", dx=-0.30)


def scatter(ax, yv, ylabel, letter):
    r, p = sps.spearmanr(axis, yv)
    ax.scatter(axis, yv, s=11, color=[CIT[region[q]] for q in axis.index], zorder=3, linewidths=0)
    m, c = np.polyfit(axis.values, yv.values, 1)
    xs = np.linspace(axis.min() - 0.5, axis.max() + 0.5, 50)
    ax.plot(xs, m * xs + c, ls="--", color="#555555", lw=0.7, zorder=2)
    ax.set_xlabel("Innate-hyperactive axis")
    ax.set_ylabel(ylabel)
    ax.set_title(f"ρ = {r:+.2f}, P = {p:.2f}, n = {len(yv)}", fontsize=6.6, pad=4)
    ax.set_xticks([-5, -2.5, 0, 2.5, 5, 7.5])
    ax.set_xticklabels(["−5.0", "−2.5", "0.0", "2.5", "5.0", "7.5"])
    despine(ax)
    panel(ax, letter, dx=-0.30)


scatter(fig.add_subplot(gs[1, 2:4]), ldh, "log$_{10}$ LDH (U l$^{-1}$)", "d")
scatter(fig.add_subplot(gs[1, 4:]), esr, "ESR (mm h$^{-1}$)", "e")

fig.savefig(f"{OUT}/Fig2_revised.pdf")
fig.savefig(f"{OUT}/Fig2_revised.png", dpi=300)
fig.savefig(f"{OUT}/Fig2_revised.tiff", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
print("wrote", f"{OUT}/Fig2_revised.pdf/.png/.tiff")
