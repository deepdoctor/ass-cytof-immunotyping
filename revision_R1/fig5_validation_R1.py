"""
Revision R1 -- manuscript Fig. 5 (GSE220915 external validation), redrawn from
the corrected-mapping outputs (41/41 IT1 genes; see verify_gse220915_mapping.py)
in the layout and palette of the submitted figure.

Inputs (written by cytof_external_validation.py):
  cytof_output_nature/S4_validation_scores.csv     per-biopsy IT1 scores
  cytof_output_nature/S4_validation_stats.csv      Kruskal-Wallis / Mann-Whitney
  cytof_output_nature/S5_AS_IT1high_vs_IT1low_DE.csv
  cytof_output_nature/S5_AS_pathway_enrichment.csv
Pathway gene sets are those of cytof_within_AS.py.

Run from the repository root:  python revision_R1/fig5_validation_R1.py
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats as sps

warnings.filterwarnings("ignore")
IN_DIR = "cytof_output_nature"
OUT_DIR = "revision_R1/figs_R1"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 6.5, "axes.titlesize": 6.5, "axes.labelsize": 7.0,
    "xtick.labelsize": 6.0, "ytick.labelsize": 6.0, "legend.fontsize": 5.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})
PANEL = dict(fontsize=9, ha="left", va="bottom")

GROUPS = ["NT", "IMNM", "IBM", "DM", "ASyS"]
GCOL = {"NT": "#8d99ae", "IMNM": "#3d6da8", "IBM": "#2a9d8f", "DM": "#e0a458", "ASyS": "#c1443c"}
PATHWAYS = {   # cytof_within_AS.py
    "IFN-γ signalling": {
        "STAT1", "IRF1", "IRF7", "IRF8", "IRF9", "GBP1", "GBP5", "CXCL9", "CXCL10", "CXCL11",
        "HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "CD74", "PSMB8", "PSMB9",
        "TAP1", "TAP2", "MX1", "ISG15", "OAS1", "OAS2", "OAS3", "IFI44", "IFIT1", "IFIT3",
        "IFITM1", "IFITM3"},
    "Complement cascade": {
        "C1QA", "C1QB", "C1QC", "C1R", "C1S", "C2", "C3", "C4A", "C4B", "C5", "CFB",
        "C3AR1", "C5AR1", "ITGAM", "ITGAX", "CR1", "VSIG4"},
    "Cytolytic effector": {
        "GZMA", "GZMB", "GZMK", "GZMH", "GZMM", "PRF1", "GNLY", "NKG7", "KLRD1", "KLRK1",
        "FGFBP2", "FASLG"},
    "Monocyte / myeloid": {
        "CD14", "CD68", "CD163", "VCAN", "S100A8", "S100A9", "S100A12", "FCGR1A", "FCGR3A",
        "CSF1R", "LYZ", "MS4A6A", "MS4A7", "CLEC10A"},
    "Antibody / plasma cell": {
        "IGHG1", "IGHG3", "IGHG4", "IGHA1", "IGHA2", "JCHAIN", "MZB1", "XBP1", "PRDM1", "CD38"},
    "T cell activation": {
        "CD3D", "CD3E", "CD3G", "CD4", "CD8A", "CD8B", "LCK", "ZAP70", "IL2RA", "CD69", "ICOS"},
}
PCOL = {"IFN-γ signalling": "#c1443c", "Complement cascade": "#3d6da8",
        "Cytolytic effector": "#7b5aa6", "Monocyte / myeloid": "#e0a458",
        "Antibody / plasma cell": "#2a9d8f", "T cell activation": "#2f4858"}
PW_FILE_NAME = {"IFN-γ signalling (IRDS core)": "IFN-γ signalling",
                "Complement (Mammen 2023 primary finding)": "Complement cascade"}


def sci(p, digits=1):
    e = int(np.floor(np.log10(p)))
    c = p / 10 ** e
    if round(c, digits - 1 if digits > 1 else 0) >= 10:
        c, e = c / 10, e + 1
    cs = f"{c:.{digits - 1}f}" if digits > 1 else f"{c:.0f}"
    return cs, e


def p_label(p):
    if p < 0.001:
        c, e = sci(p, 1)
        return rf"P = {c}$\times$10$^{{{e}}}$"
    return f"P = {p:.2f}"


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return (gt - lt) / (len(a) * len(b))


# ── data ─────────────────────────────────────────────────────────────────────
scores = pd.read_csv(f"{IN_DIR}/S4_validation_scores.csv", index_col=0)
st = pd.read_csv(f"{IN_DIR}/S4_validation_stats.csv").set_index("test")
de = pd.read_csv(f"{IN_DIR}/S5_AS_IT1high_vs_IT1low_DE.csv")
pw = pd.read_csv(f"{IN_DIR}/S5_AS_pathway_enrichment.csv")
pw["name"] = pw["pathway"].map(lambda s: PW_FILE_NAME.get(s, s))
by = {g: scores.loc[scores.diagnosis == g, "IT1_score"].values for g in GROUPS}

fig = plt.figure(figsize=(501.525 / 72, 379.983 / 72))
# axes boxes measured from the submitted figure (figure fractions)
COLS = [(0.065, 0.300), (0.440, 0.640), (0.775, 0.990)]
ROWS = [(0.617, 0.930), (0.090, 0.405)]
axs = np.array([[fig.add_axes([x0, y0, x1 - x0, y1 - y0]) for (x0, x1) in COLS]
                for (y0, y1) in ROWS])
LBL = {"a": (0.008, 0.965), "b": (0.326, 0.965), "c": (0.700, 0.965),
       "d": (0.008, 0.440), "e": (0.282, 0.440), "f": (0.700, 0.440)}
for k, (x, y_) in LBL.items():
    fig.text(x, y_, k, **PANEL)
rng = np.random.default_rng(0)

# (a) IT1 score by diagnostic group
ax = axs[0, 0]
for i, g in enumerate(GROUPS):
    v = ax.violinplot(by[g], positions=[i], widths=0.8, showextrema=False)
    for b in v["bodies"]:
        b.set_facecolor(GCOL[g]); b.set_alpha(0.28); b.set_edgecolor("none")
    ax.scatter(i + rng.uniform(-0.17, 0.17, len(by[g])), by[g], s=3.5, color=GCOL[g],
               alpha=0.85, lw=0, zorder=3)
    ax.plot([i - 0.25, i + 0.25], [np.median(by[g])] * 2, color="#222", lw=1.1, zorder=4)
ax.set_xticks(range(len(GROUPS)))
ax.set_xticklabels([f"{g}\nn={len(by[g])}" for g in GROUPS])
ax.set_ylabel("IT1 signature score")
H, Pk = st.loc["KW 5-group", "stat"], st.loc["KW 5-group", "p"]
c, e = sci(Pk, 2)
ax.set_title(rf"Kruskal–Wallis H = {H:.2f}, P = {c} $\times$ 10$^{{{e}}}$", pad=4)

# (b) pairwise Cliff's delta, ASyS vs each comparator, 2,000-resample bootstrap CI
ax = axs[0, 1]
comps = ["NT", "IMNM", "IBM", "DM"]
brng = np.random.default_rng(2024)
for k, g in enumerate(comps):
    a, b = by["ASyS"], by[g]
    d = cliffs_delta(a, b)
    boots = [cliffs_delta(brng.choice(a, len(a)), brng.choice(b, len(b))) for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    y = len(comps) - 1 - k
    ax.plot([lo, hi], [y, y], color=GCOL[g], lw=1.1, solid_capstyle="round")
    ax.scatter([d], [y], s=14, color=GCOL[g], zorder=3, lw=0)
    ax.text(d, y + 0.22, p_label(st.loc[f"MW ASyS vs {g}", "p"]), va="bottom", ha="center",
            fontsize=5.2, color="#333", zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
ax.axvline(0, color="#222", lw=0.6)
ax.set_yticks(range(len(comps)))
ax.set_yticklabels([f"ASyS vs {g}" for g in comps[::-1]], fontsize=5.6)
ax.set_xlim(-0.25, 1.0)
ax.set_ylim(-0.6, len(comps) - 0.4)
ax.set_xlabel("Cliff's δ (95% bootstrap CI)")
ax.set_title("Pairwise effect size", pad=4)

# (c) within-ASyS heterogeneity, median split
ax = axs[0, 2]
asy = np.sort(by["ASyS"])
med = np.median(asy)
ax.bar(range(len(asy)), asy, width=0.72, color=["#c1443c" if v > med else "#8d99ae" for v in asy])
for val, col, ls, lab in [(med, "#222", "--", f"ASyS median {med:.2f}"),
                          (np.median(by["DM"]), "#e0a458", ":", "DM median"),
                          (np.median(by["NT"]), "#8d99ae", ":", "NT median")]:
    ax.axhline(val, color=col, ls=ls, lw=0.9)
    right = lab == "NT median"
    ax.text(len(asy) - 0.6 if right else -0.4, val + 0.03, lab, fontsize=5.5,
            color=col if col != "#222" else "#333", va="bottom", ha="right" if right else "left")
ax.set_xticks([])
ax.set_xlabel(f"{len(asy)} ASyS biopsies, ranked")
ax.set_ylabel("IT1 signature score")
ax.set_title("IT1-high (red) versus IT1-low (grey)", pad=4)

# (d) volcano, IT1-high vs IT1-low (Welch), BH FDR < 0.10 in red
ax = axs[1, 0]
sig = de.q_fdr < 0.10
y = -np.log10(de.p)
ax.scatter(de.log2fc[~sig], y[~sig], s=1.2, color="#c8ccd3", lw=0, rasterized=True)
ax.scatter(de.log2fc[sig], y[sig], s=1.6, color="#c1443c", alpha=0.75, lw=0, rasterized=True)
p_thr = de.loc[sig, "p"].max()
ax.axhline(-np.log10(p_thr), color="#222", ls=":", lw=0.9)
ax.text(de.log2fc.min(), -np.log10(p_thr) + 0.05, "BH q = 0.10", fontsize=5.5, color="#333",
        va="bottom", bbox=dict(facecolor="white", edgecolor="none", pad=0.4, alpha=0.85))
from adjustText import adjust_text
lab = pd.concat([de[sig].nlargest(5, "log2fc"), de[sig & (de.log2fc > 0)].nsmallest(2, "p")])
texts = [ax.text(r.log2fc, -np.log10(r.p), r.symbol, fontsize=5.0, color="#333")
         for _, r in lab.drop_duplicates("symbol").iterrows()]
ax.set_xlim(de.log2fc.min() - 0.3, de.log2fc.max() + 0.4)
adjust_text(texts, ax=ax, expand=(1.3, 1.6), ensure_inside_axes=True,
            arrowprops=dict(arrowstyle="-", color="#666", lw=0.4))
ax.set_xlabel(r"log$_2$ fold change (IT1-high vs IT1-low)")
ax.set_ylabel(r"$-$log$_{10}$ P (Welch)")
n_hi = int((scores.loc[scores.diagnosis == "ASyS", "IT1_score"] > med).sum())
ax.set_title(f"{int(sig.sum()):,} of {len(de):,} genes at FDR < 0.10 "
             f"(n = {n_hi} vs {len(asy) - n_hi})", pad=4)

# (e) pathway enrichment among genes up in IT1-high
ax = axs[1, 1]
e_ = pw.sort_values("hi_qBH", ascending=False).reset_index(drop=True)
xv = -np.log10(e_.hi_qBH)
ax.barh(range(len(e_)), xv, height=0.62, color=[PCOL[n] for n in e_.name], alpha=0.9)
for i, r in e_.iterrows():
    ax.text(xv[i] + 0.3, i, f"{r.hi_hits}/{r.K_in_bg}", va="center", fontsize=5.5, color="#333")
ax.set_yticks(range(len(e_)))
ax.set_yticklabels(e_.name, fontsize=5.6)
ax.set_xlim(0, xv.max() * 1.22)
ax.set_xlabel(r"$-$log$_{10}$ BH q (one-sided hypergeometric)")
ax.set_title("Enrichment among genes up in IT1-high", pad=4)

# (f) the fourteen pathway-annotated genes (FDR < 0.10, up) with the largest fold change
ax = axs[1, 2]
g2p = {g: p for p, gs in PATHWAYS.items() for g in gs}
up = de[sig & (de.log2fc > 0) & de.symbol.isin(g2p)].sort_values("log2fc", ascending=False)
top = up.drop_duplicates("symbol").head(14).iloc[::-1]
ax.barh(range(len(top)), top.log2fc, height=0.72, color=[PCOL[g2p[s]] for s in top.symbol],
        alpha=0.9)
ax.set_yticks(range(len(top)))
ax.set_yticklabels(top.symbol)
ax.set_xlim(0, top.log2fc.max() * 1.35)
ax.set_xlabel(r"log$_2$ fold change (IT1-high vs IT1-low)")
ax.set_title("Leading pathway-annotated genes", pad=4)
ax.legend(handles=[Patch(color=c, label=n) for n, c in PCOL.items()], loc="lower right",
          frameon=False, fontsize=4.8, handlelength=0.9, handleheight=0.9, labelspacing=0.25,
          borderaxespad=0.1)

for ext in ("pdf", "png"):
    fig.savefig(f"{OUT_DIR}/Fig5_revised.{ext}", dpi=300, facecolor="white")
fig.savefig(f"{OUT_DIR}/Fig5_revised.tiff", dpi=300, facecolor="white",
            pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print("panel f genes:", top.symbol.tolist()[::-1])
print("ASyS median", round(med, 3), "| DEGs", int(sig.sum()), "| P threshold", p_thr)
