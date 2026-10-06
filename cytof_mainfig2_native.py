"""
Native unified vector rebuild of Main Figure 2 — functional-state characterization.

All four panels rendered as matplotlib primitives. Heatmaps use pcolormesh.
Dense UMAP overlays are subsampled to 25k cells per panel to keep them vector
(matplotlib auto-rasterizes scatter > ~50k points for performance).
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
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as mplcm
import matplotlib.colors as mcolors

OUT_DIR  = "./cytof_output_nature"
H5AD     = "/Users/yichen/Desktop/collaboration/raw/cytof_output_v3/cytof_analyzed_v3.h5ad"

CLUSTER_ANNOTATION = {
    "0":"Classical Monocyte","1":"Naive B","2":"CD8 Effector T",
    "3":"CD4 Naive T","4":"CD4 Central Memory T","5":"Non-classical Monocyte",
    "6":"NK cell","7":"CD4 Th1-like","8":"Classical Monocyte",
    "9":"CD4 Effector Memory T","10":"Inflammatory Monocyte",
    "11":"CD8 Naive T","12":"γδ T cell","13":"mDC","14":"pDC",
    "15":"T-Myeloid doublets","16":"CD16+ Granulocyte",
}
ARTIFACTS = {"T-Myeloid doublets","CD16+ Granulocyte"}
GROUP_PALETTE = {"Jo-1":"#C0392B","PL-12":"#2471A3","EJ":"#1A7A4A","PL-7":"#7D3C98"}
GROUP_ORDER = ["Jo-1","PL-12","EJ","PL-7"]
GROUP_LABEL_NB = {"Jo-1":"Jo-1","PL-12":"PL-12","EJ":"EJ","PL-7":"PL-7"}  # plain ASCII hyphen

plt.rcParams.update({
    "font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "font.size":10.5,"axes.titlesize":12.0,"axes.labelsize":10.5,
    "xtick.labelsize":9.5,"ytick.labelsize":9.5,"legend.fontsize":9.5,
    "axes.linewidth":0.9,"axes.spines.top":False,"axes.spines.right":False,
    "pdf.fonttype":42,"ps.fonttype":42,"savefig.dpi":300,"figure.dpi":300,
})

def fmt_p(p):
    if p == 0 or p < 1e-300: return r"$P < 10^{-300}$"
    if p < 1e-4:
        e = int(np.floor(np.log10(p))); c = p/10**e
        return rf"$P = {c:.1f}\times 10^{{{e}}}$"
    if p < 1e-2: return f"P = {p:.1e}"
    return f"P = {p:.3f}"

# ─── load ──────────────────────────────────────────────────────────────────
print("Loading h5ad …")
a = ad.read_h5ad(H5AD)
a.obs_names_make_unique()
a.obs["cell_type"] = a.obs["leiden"].astype(str).map(CLUSTER_ANNOTATION).fillna("Unannotated")
a = a[~a.obs["cell_type"].isin(ARTIFACTS)].copy()
X_umap = a.obsm["X_umap"]


# ════════════════════════════════════════════════════════════════════════════
# FIGURE LAYOUT — 14 × 13 outer 2×2
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(15.5, 14.5), facecolor="white")
outer = gridspec.GridSpec(
    2, 2, figure=fig,
    left=0.045, right=0.985, top=0.955, bottom=0.030,
    wspace=0.20, hspace=0.26,
)


# ── (a) UMAP overlays — 6 functional markers (subsampled for vector) ─────
SELECTED_MK = ["PD-1","GranzymeB","CD38","Ki-67","HLA-DR","CXCR3"]
present_mk = [m for m in SELECTED_MK if m in a.var_names]

# Subsample to 5k cells (random) — at this size matplotlib's PDF backend
# produces vector scatter rather than auto-rasterizing for performance.
SUB_N = 5000
rng_a = np.random.default_rng(42)
sub_idx = rng_a.choice(a.n_obs, size=min(SUB_N, a.n_obs), replace=False)
X_sub = X_umap[sub_idx]

SEQ_CMAP = LinearSegmentedColormap.from_list("mk",
    ["#E0E0E0","#FAD7A0","#F39C12","#E67E22","#C0392B","#7B241C"], N=256)

inner_a = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer[0, 0],
                                            wspace=0.30, hspace=0.30)
for ai, mk in enumerate(present_mk):
    ax = fig.add_subplot(inner_a[ai // 3, ai % 3])
    j = list(a.var_names).index(mk)
    expr = a.X[:, j]
    expr = expr.toarray().ravel() if hasattr(expr,"toarray") else np.asarray(expr).ravel()
    expr_sub = expr[sub_idx]
    vmax = np.percentile(expr_sub, 99)
    expr_clip = np.clip(expr_sub, 0, vmax)
    order = np.argsort(expr_clip)
    sc = ax.scatter(X_sub[order,0], X_sub[order,1], s=2.5,
                     c=expr_clip[order], cmap=SEQ_CMAP, alpha=0.85,
                     edgecolor="none", linewidths=0,
                     vmin=0, vmax=vmax,
                     rasterized=False)
    ax.set_title(mk, fontsize=12, fontweight="bold", pad=3)
    ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(sc, ax=ax, shrink=0.60, pad=0.014, fraction=0.050)
    cbar.ax.tick_params(labelsize=8); cbar.outline.set_linewidth(0.4)
fig.text(0.025, 0.962, "a", fontsize=24, fontweight="bold", ha="left", va="top")


# ── (b) Top 15 differential markers heatmap (pcolormesh = vector) ────────
ax_b = fig.add_subplot(outer[0, 1])
de_g = pd.read_csv(f"{OUT_DIR}/06_differential_markers.csv")
top15 = de_g.sort_values("KW_stat", ascending=False).head(15).copy()
mean_cols = [f"mean_{g}" for g in GROUP_ORDER]
M_mean = top15[mean_cols].values
M_z = (M_mean - M_mean.mean(axis=1, keepdims=True)) / \
      (M_mean.std(axis=1, keepdims=True) + 1e-9)

from scipy.cluster.hierarchy import linkage, leaves_list
Z = linkage(M_z, method="average")
order = leaves_list(Z)
M_z = M_z[order]
markers_b = top15["marker"].iloc[order].values
padjs_b = top15["padj"].iloc[order].values

HEAT_DIV = LinearSegmentedColormap.from_list("hd",
    ["#1A3A6B","#2980B9","#7FB3D3","#F4D03F","#E67E22","#C0392B","#7B241C"], N=256)

nrows_b, ncols_b = M_z.shape
im_b = ax_b.pcolormesh(
    np.arange(ncols_b + 1) - 0.5, np.arange(nrows_b + 1) - 0.5,
    M_z, cmap=HEAT_DIV, vmin=-2, vmax=2,
    edgecolors="white", lw=0.5, rasterized=False, shading="flat",
)
ax_b.invert_yaxis()
# group brackets above (placed clearly above the heatmap; title space reserved)
for i, g in enumerate(GROUP_ORDER):
    ax_b.add_patch(Rectangle((i - 0.45, -0.85), 0.90, 0.40,
                              facecolor=GROUP_PALETTE[g], edgecolor="white", lw=0.5,
                              clip_on=False))
    ax_b.text(i, -1.10, GROUP_LABEL_NB[g], ha="center", va="bottom",
               fontsize=11.5, fontweight="bold", color=GROUP_PALETTE[g], clip_on=False)
ax_b.set_xticks(range(len(GROUP_ORDER)))
ax_b.set_xticklabels([])
ax_b.set_yticks(range(len(markers_b)))
ax_b.set_yticklabels(markers_b, fontsize=11)

# annotate q on right
for i, q in enumerate(padjs_b):
    if q == 0 or q < 1e-300: q_str = r"$P < 10^{-300}$"
    elif q < 1e-4:
        e = int(np.floor(np.log10(q))); c = q/10**e
        q_str = rf"$P = {c:.1f}\times 10^{{{e}}}$"
    else: q_str = f"P = {q:.1e}"
    ax_b.text(len(GROUP_ORDER) - 0.4, i, q_str, ha="left", va="center",
               fontsize=8.5, color="#444")

cbar_b = fig.colorbar(im_b, ax=ax_b, fraction=0.04, pad=0.20, shrink=0.85)
cbar_b.set_label("Z-score (mean arcsinh)", fontsize=10)
cbar_b.ax.tick_params(labelsize=9); cbar_b.outline.set_linewidth(0.4)
# Inline title placed inside the panel area, not overlapping the brackets above
ax_b.text(0.5, 1.18, "Top 15 differential markers (KW)",
           transform=ax_b.transAxes, ha="center", va="bottom",
           fontsize=11.5, color="#222")
fig.text(0.515, 0.962, "b", fontsize=24, fontweight="bold", ha="left", va="top")


# ── (c) 6 top-marker violins ──────────────────────────────────────────────
inner_c = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer[1, 0],
                                            wspace=0.40, hspace=0.55)
top6 = de_g.sort_values("KW_stat", ascending=False).head(6)
rng_c = np.random.default_rng(0)
for ci, (_, r) in enumerate(top6.iterrows()):
    ax_c = fig.add_subplot(inner_c[ci // 3, ci % 3])
    mk = r["marker"]
    if mk not in a.var_names: continue
    j = list(a.var_names).index(mk)
    Xj = a.X[:, j]
    Xj = Xj.toarray().ravel() if hasattr(Xj,"toarray") else np.asarray(Xj).ravel()
    data_list = [Xj[a.obs["group"].values == g] for g in GROUP_ORDER]
    parts = ax_c.violinplot(data_list, positions=range(len(GROUP_ORDER)),
                              widths=0.72, showmedians=True, showextrema=False)
    for body, g in zip(parts["bodies"], GROUP_ORDER):
        body.set_facecolor(GROUP_PALETTE[g]); body.set_alpha(0.65)
        body.set_edgecolor(GROUP_PALETTE[g]); body.set_linewidth(0.6)
    parts["cmedians"].set_color("#222"); parts["cmedians"].set_linewidth(0.8)
    for k, g in enumerate(GROUP_ORDER):
        v = data_list[k]
        if len(v) > 500: v = rng_c.choice(v, 500, replace=False)
        jit = rng_c.uniform(-0.10, 0.10, len(v))
        ax_c.scatter(k + jit, v, c=GROUP_PALETTE[g], s=2.0, alpha=0.25,
                     zorder=3, linewidths=0)
    ax_c.set_xticks(range(len(GROUP_ORDER)))
    ax_c.set_xticklabels([GROUP_LABEL_NB[g] for g in GROUP_ORDER], fontsize=10)
    ax_c.set_ylabel("arcsinh expr.", fontsize=10)
    ax_c.set_title(f"{mk}    {fmt_p(float(r['padj']))}", fontsize=11.5,
                    fontweight="bold", pad=3)
fig.text(0.025, 0.475, "c", fontsize=24, fontweight="bold", ha="left", va="top")


# ── (d) Marker co-expression modules (Spearman heatmap, Ward modules) ────
ax_d = fig.add_subplot(outer[1, 1])
# Build patient-level marker correlation matrix
samples = sorted(a.obs["sample"].unique())
markers_all = list(a.var_names)
mfi = pd.DataFrame(index=samples, columns=markers_all, dtype=float)
for s in samples:
    mask = a.obs["sample"].values == s
    Xs = a.X[mask]
    Xs = Xs.toarray() if hasattr(Xs,"toarray") else np.asarray(Xs)
    mfi.loc[s] = Xs.mean(axis=0)
corr = mfi.corr(method="spearman")

# Ward modules from 1-|r|
import scipy.spatial.distance as ssd
dist = 1 - corr.abs()
condensed = ssd.squareform(dist.values, checks=False)
Zm = linkage(condensed, method="average")
order_m = leaves_list(Zm)
markers_o = [markers_all[i] for i in order_m]
corr_o = corr.iloc[order_m, order_m]

n_m = len(markers_o)
RB = LinearSegmentedColormap.from_list("rb",
    ["#1A3A6B","#2980B9","#FFFFFF","#E67E22","#C0392B"], N=256)
im_d = ax_d.pcolormesh(
    np.arange(n_m + 1) - 0.5, np.arange(n_m + 1) - 0.5,
    corr_o.values, cmap=RB, vmin=-1, vmax=1,
    edgecolors="none", rasterized=False, shading="flat",
)
ax_d.invert_yaxis()
ax_d.set_xticks(range(n_m))
ax_d.set_yticks(range(n_m))
ax_d.set_xticklabels(markers_o, rotation=90, fontsize=8)
ax_d.set_yticklabels(markers_o, fontsize=8)
ax_d.set_aspect("equal")

cbar_d = fig.colorbar(im_d, ax=ax_d, fraction=0.04, pad=0.025, shrink=0.85)
cbar_d.set_label("Spearman ρ (patient-level marker corr.)", fontsize=10)
cbar_d.ax.tick_params(labelsize=9); cbar_d.outline.set_linewidth(0.4)
ax_d.text(0.5, 1.02,
          "Patient-level marker co-expression modules (Spearman, Ward)",
          transform=ax_d.transAxes, ha="center", va="bottom",
          fontsize=11.5, color="#222")
fig.text(0.515, 0.475, "d", fontsize=24, fontweight="bold", ha="left", va="top")


# ─── suptitle + save ─────────────────────────────────────────────────────
# Figure title removed (journal convention).
out_pdf = f"{OUT_DIR}/MainFig2_marker_characterization.pdf"
out_png = f"{OUT_DIR}/MainFig2_marker_characterization.png"
fig.savefig(out_pdf, dpi=300, bbox_inches="tight",
             facecolor="white", edgecolor="none")
fig.savefig(out_png, dpi=300, bbox_inches="tight",
             facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Saved (vector): {out_pdf}")
print(f"Saved (raster): {out_png}")
