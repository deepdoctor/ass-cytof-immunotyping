"""
Native unified vector rebuild of Main Figure 1 — atlas.

Layout: 5 panels in a clean 2-row arrangement with all UMAP panels
uniform-sized.

  Row 1 (3 equal-sized square UMAPs): (a) group · (b) sample · (c) cluster
  Row 2 (1 UMAP + 1 dotplot, dotplot spans 2 cols): (d) cell type · (e) dotplot

The by-sample UMAP (panel b) serves as batch-correction verification —
its retention here lets a reader confirm Harmony integration directly.

All elements are vector primitives in the output PDF. Dense UMAP scatters
in panels (a–d) are explicitly rasterized at 200 dpi to keep file size
compact (~1 MB rather than ~6 MB); all other elements (text, axes,
legends, dotplot dots) remain vector and editable.
"""
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as mplcm
import matplotlib.colors as mcolors

OUT_DIR = "./cytof_output_nature"
H5AD    = "/Users/yichen/Desktop/collaboration/raw/cytof_output_v3/cytof_analyzed_v3.h5ad"

CLUSTER_ANNOTATION = {
    "0":"Classical Monocyte","1":"Naive B","2":"CD8 Effector T",
    "3":"CD4 Naive T","4":"CD4 Central Memory T","5":"Non-classical Monocyte",
    "6":"NK cell","7":"CD4 Th1-like","8":"Classical Monocyte",
    "9":"CD4 Effector Memory T","10":"Inflammatory Monocyte",
    "11":"CD8 Naive T","12":"γδ T cell","13":"mDC","14":"pDC",
    "15":"T-Myeloid doublets","16":"CD16+ Granulocyte",
}
ARTIFACTS = {"T-Myeloid doublets","CD16+ Granulocyte"}

CT_ORDER = [
    "CD4 Naive T","CD4 Central Memory T","CD4 Effector Memory T","CD4 Th1-like",
    "CD8 Naive T","CD8 Effector T","γδ T cell",
    "Classical Monocyte","Inflammatory Monocyte","Non-classical Monocyte",
    "mDC","pDC","NK cell","Naive B",
]
CT_PALETTE = {
    "CD4 Naive T":"#7DC1E8","CD4 Central Memory T":"#1F77B4",
    "CD4 Effector Memory T":"#5B9BD5","CD4 Th1-like":"#3A7CA5",
    "CD8 Naive T":"#C5B0D5","CD8 Effector T":"#9467BD",
    "γδ T cell":"#F1C40F",
    "Classical Monocyte":"#E67E22","Inflammatory Monocyte":"#D35400",
    "Non-classical Monocyte":"#F39C12",
    "mDC":"#E74C3C","pDC":"#C0392B","NK cell":"#27AE60","Naive B":"#16A085",
}
ARTIFACT_COL = "#BBBBBB"

GROUP_PALETTE = {"Jo-1":"#C0392B","PL-12":"#2471A3","EJ":"#1A7A4A","PL-7":"#7D3C98"}
GROUP_ORDER = ["Jo-1","PL-12","EJ","PL-7"]
GROUP_LABEL = {"Jo-1":"Jo-1","PL-12":"PL-12","EJ":"EJ","PL-7":"PL-7"}  # plain ASCII hyphen

SAMPLE_PALETTE_10 = [
    "#E41A1C","#377EB8","#4DAF4A","#FF7F00","#984EA3",
    "#FFFF33","#A65628","#F781BF","#999999","#1B9E77",
]

LEIDEN_PALETTE = [
    "#E64B35","#4DBBD5","#00A087","#3C5488","#F39B7F",
    "#8491B4","#91D1C2","#DC0000","#7E6148","#B09C85",
    "#3B9AB2","#78B7C5","#EBCC2A","#E1AF00","#F21A00",
    "#E2D200","#46ACC8",
]

CANONICAL = [
    "CD3","CD4","CD8a","CD45RA","CD45RO","CCR7","CD27","CD127",
    "CXCR5","ICOS","GranzymeB","PD-1","TIGIT","Tim-3","TCF1",
    "CD19","CD20","IgD","IgM","CD38",
    "CD14","CD16","CD11c","CD123","HLA-DR","CD56","TCRgd","Ki-67",
]

plt.rcParams.update({
    "font.family":         "sans-serif",
    "font.sans-serif":     ["Arial","Helvetica","DejaVu Sans"],
    "font.size":           9.5,
    "axes.titlesize":      11.0,
    "axes.labelsize":      9.5,
    "xtick.labelsize":     8.5,
    "ytick.labelsize":     8.5,
    "legend.fontsize":     8.5,
    "axes.linewidth":      0.8,
    "xtick.major.width":   0.7,
    "ytick.major.width":   0.7,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "pdf.fonttype":        42,
    "ps.fonttype":         42,
    "savefig.dpi":         300,
    "figure.dpi":          300,
})


# ─── load h5ad ──────────────────────────────────────────────────────────────
print("Loading h5ad …")
a = ad.read_h5ad(H5AD)
a.obs_names_make_unique()
a.obs["cell_type"] = a.obs["leiden"].astype(str).map(CLUSTER_ANNOTATION).fillna("Unannotated")
X = a.obsm["X_umap"]
samples = sorted(a.obs["sample"].unique())
print(f"  {a.n_obs:,} cells, {len(samples)} samples")


# ════════════════════════════════════════════════════════════════════════════
# LAYOUT
#  Row 1 (3 cols): [a group][b sample][c cluster] — three same-sized UMAPs
#  Row 2 (3 cols): [d celltype][e dotplot spans cols 1 + 2]
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14.0, 11.5), facecolor="white")
outer = gridspec.GridSpec(
    2, 3, figure=fig,
    left=0.045, right=0.985, top=0.965, bottom=0.040,
    wspace=0.20, hspace=0.22,
    width_ratios=[1, 1, 1],
    height_ratios=[1, 1.02],
)

xlim = (X[:,0].min() - 1.0, X[:,0].max() + 1.0)
ylim = (X[:,1].min() - 1.0, X[:,1].max() + 1.0)


def render_umap_panel(outer_cell, scatter_func, title=None,
                      legend_handles=None, legend_ncol=4,
                      legend_height=0.55):
    """Render a UMAP into outer_cell, with optional title and bottom legend.

    scatter_func(ax) draws scatter on the axes. Returns the data ax.
    """
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer_cell,
        height_ratios=[5.0, legend_height], hspace=0.05,
    )
    ax = fig.add_subplot(inner[0, 0])
    scatter_func(ax)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect("equal")
    if title:
        ax.text(0.02, 0.98, title, transform=ax.transAxes,
                ha="left", va="top", fontsize=10, color="#444",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                          boxstyle="round,pad=0.20"))
    if legend_handles is not None:
        ax_leg = fig.add_subplot(inner[1, 0]); ax_leg.axis("off")
        ax_leg.legend(handles=legend_handles, loc="upper center",
                       bbox_to_anchor=(0.5, 1.0), fontsize=9.5,
                       frameon=False, ncol=legend_ncol,
                       handletextpad=0.4, columnspacing=1.4,
                       labelspacing=0.45, borderpad=0.2)
    return ax


# ── (a) UMAP by autoantibody group ───────────────────────────────────────
def _scatter_a(ax):
    for g in GROUP_ORDER:
        mask = a.obs["group"].values == g
        ax.scatter(X[mask,0], X[mask,1], s=1.4,
                    c=GROUP_PALETTE[g], alpha=0.85,
                    edgecolor="none", linewidths=0, rasterized=True)
group_handles = [Line2D([0],[0], marker="o", color="w",
                          markerfacecolor=GROUP_PALETTE[g], markersize=10,
                          label=GROUP_LABEL[g])
                 for g in GROUP_ORDER]
render_umap_panel(outer[0, 0], _scatter_a,
                   title="Autoantibody group",
                   legend_handles=group_handles, legend_ncol=4)
fig.text(0.038, 0.962, "a", fontsize=24, fontweight="bold", ha="left", va="top")


# ── (b) UMAP by individual sample (batch-correction verification) ────────
def _scatter_b(ax):
    for i, s in enumerate(samples):
        mask = a.obs["sample"].values == s
        ax.scatter(X[mask,0], X[mask,1], s=1.4,
                    c=SAMPLE_PALETTE_10[i % 10], alpha=0.78,
                    edgecolor="none", linewidths=0, rasterized=True)
sample_handles = [Line2D([0],[0], marker="o", color="w",
                           markerfacecolor=SAMPLE_PALETTE_10[i % 10],
                           markersize=8,
                           label=s.replace("L01912_",""))
                  for i, s in enumerate(samples)]
render_umap_panel(outer[0, 1], _scatter_b,
                   title="Individual sample",
                   legend_handles=sample_handles, legend_ncol=5,
                   legend_height=0.55)
fig.text(0.355, 0.962, "b", fontsize=24, fontweight="bold", ha="left", va="top")


# ── (c) UMAP by Leiden cluster ───────────────────────────────────────────
def _scatter_c(ax):
    for cl in sorted(a.obs["leiden"].astype(int).unique()):
        mask = a.obs["leiden"].astype(int).values == cl
        ax.scatter(X[mask,0], X[mask,1], s=1.4,
                    c=LEIDEN_PALETTE[cl % len(LEIDEN_PALETTE)],
                    alpha=0.85, edgecolor="none", linewidths=0,
                    rasterized=True)
        cx, cy = np.median(X[mask,0]), np.median(X[mask,1])
        ax.text(cx, cy, str(cl), ha="center", va="center",
                 fontsize=10, fontweight="bold", color="#111",
                 bbox=dict(boxstyle="circle,pad=0.18",
                           facecolor="white", edgecolor="#444", lw=0.7),
                 zorder=10)
render_umap_panel(outer[0, 2], _scatter_c,
                   title="Leiden clusters\n(resolution = 0.8)",
                   legend_handles=None, legend_height=0.05)
fig.text(0.670, 0.962, "c", fontsize=24, fontweight="bold", ha="left", va="top")


# ── (d) UMAP by cell type (row 2, col 0) ──────────────────────────────────
def _scatter_d(ax):
    for ct in ARTIFACTS:
        mask = a.obs["cell_type"].values == ct
        if mask.sum():
            ax.scatter(X[mask,0], X[mask,1], s=1.4, c=ARTIFACT_COL,
                        alpha=0.55, edgecolor="none", linewidths=0,
                        rasterized=True)
    for ct in CT_ORDER:
        if ct in ARTIFACTS: continue
        mask = a.obs["cell_type"].values == ct
        if mask.sum():
            ax.scatter(X[mask,0], X[mask,1], s=1.4,
                        c=CT_PALETTE[ct], alpha=0.85,
                        edgecolor="none", linewidths=0, rasterized=True)

ct_handles = [Line2D([0],[0], marker="o", color="w",
                       markerfacecolor=CT_PALETTE[ct], markersize=8,
                       label=ct)
              for ct in CT_ORDER if ct not in ARTIFACTS]
render_umap_panel(outer[1, 0], _scatter_d,
                   title="Cell-type annotation",
                   legend_handles=ct_handles, legend_ncol=2,
                   legend_height=1.10)
fig.text(0.038, 0.485, "d", fontsize=24, fontweight="bold", ha="left", va="top")


# ── (e) Canonical-marker dotplot (row 2, cols 1 + 2 spanning) ────────────
inner_e_outer = gridspec.GridSpecFromSubplotSpec(
    1, 1, subplot_spec=gridspec.GridSpecFromSubplotSpec(
        1, 1, subplot_spec=outer[1, 1:3]
    )[0, 0],
)
# direct GridSpecFromSubplotSpec on a sliced outer cell
inner_e = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outer[1, 1:3],
    width_ratios=[8.5, 1.0], wspace=0.04,
)
ax_e = fig.add_subplot(inner_e[0, 0])
HEAT_DOT = LinearSegmentedColormap.from_list("dot",
    ["#1F618D","#5DADE2","#F4D03F","#E67E22","#C0392B"], N=256)

mask_keep = a.obs["cell_type"].isin(CT_ORDER) & (~a.obs["cell_type"].isin(ARTIFACTS))
a_real = a[mask_keep].copy()
markers_present = [m for m in CANONICAL if m in a_real.var_names]

mean_mat = np.zeros((len(CT_ORDER), len(markers_present)))
pct_mat  = np.zeros((len(CT_ORDER), len(markers_present)))
for i, ct in enumerate(CT_ORDER):
    if ct in ARTIFACTS: continue
    idx = np.where(a_real.obs["cell_type"].values == ct)[0]
    if len(idx) == 0: continue
    Xsub = a_real.X[idx]
    Xsub = Xsub.toarray() if hasattr(Xsub,"toarray") else np.asarray(Xsub)
    for j, mk in enumerate(markers_present):
        col_idx = list(a_real.var_names).index(mk)
        vals = Xsub[:, col_idx]
        mean_mat[i, j] = vals.mean()
        pct_mat[i, j]  = (vals > 0.5).mean() * 100

z = (mean_mat - mean_mat.mean(axis=0, keepdims=True)) / \
    (mean_mat.std(axis=0, keepdims=True) + 1e-9)
vmin, vmax = -2.0, 2.0
z_clip = np.clip(z, vmin, vmax)

size_min, size_max = 18, 220
sizes = size_min + (pct_mat / 100.0) * (size_max - size_min)

n_ct = len(CT_ORDER); n_mk = len(markers_present)
for i in range(n_ct):
    for j in range(n_mk):
        col = HEAT_DOT((z_clip[i, j] - vmin) / (vmax - vmin))
        ax_e.scatter(j, n_ct - 1 - i, s=sizes[i, j], c=[col],
                      edgecolor="white", linewidth=0.4)
ax_e.set_xticks(range(n_mk))
ax_e.set_xticklabels(markers_present, rotation=45, ha="right", fontsize=8.5)
ax_e.set_yticks(range(n_ct))
ax_e.set_yticklabels(CT_ORDER[::-1], fontsize=8.8)
ax_e.set_xlim(-0.6, n_mk - 0.4)
ax_e.set_ylim(-0.6, n_ct - 0.4)
ax_e.grid(color="#eee", lw=0.3); ax_e.set_axisbelow(True)
ax_e.text(0.005, 1.025, "Canonical marker expression by cell type",
           transform=ax_e.transAxes, ha="left", va="bottom",
           fontsize=10, color="#444")

# right column: colorbar (top half) + size legend (bottom half), no overlap
ax_e_legend = fig.add_subplot(inner_e[0, 1]); ax_e_legend.axis("off")

# colorbar in upper half
sm = mplcm.ScalarMappable(cmap=HEAT_DOT,
                          norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
cbar = fig.colorbar(sm, ax=ax_e_legend, fraction=0.7, pad=0.0,
                     shrink=0.45, anchor=(0.0, 1.0),
                     panchor=(0.0, 1.0))
cbar.set_label("Z-score", fontsize=9.0)
cbar.ax.tick_params(labelsize=8.0); cbar.outline.set_linewidth(0.4)

# size legend in lower half
size_legend_pcts = [10, 30, 60, 90]
size_handles = [
    Line2D([0],[0], marker="o", color="w",
           markerfacecolor="#888", markeredgecolor="white",
           markersize=np.sqrt(size_min + (pct/100.0)*(size_max-size_min))/1.5,
           label=f"{pct}%")
    for pct in size_legend_pcts
]
ax_e_legend.legend(handles=size_handles,
                    loc="lower left", bbox_to_anchor=(-0.05, 0.02),
                    frameon=False, title="% expressing",
                    title_fontsize=8.5, fontsize=8.5,
                    handletextpad=0.5, labelspacing=0.55, borderpad=0.2)

fig.text(0.355, 0.485, "e", fontsize=24, fontweight="bold", ha="left", va="top")


# ─── save ────────────────────────────────────────────────────────────────
out_pdf = f"{OUT_DIR}/MainFig1_atlas.pdf"
out_png = f"{OUT_DIR}/MainFig1_atlas.png"
fig.savefig(out_pdf, dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
fig.savefig(out_png, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Saved (vector): {out_pdf}")
print(f"Saved (raster): {out_png}")
