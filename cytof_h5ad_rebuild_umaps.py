"""
Native rebuild from the AnnData h5ad of all UMAP-based panels:

  - 04_umap_celltype.png        (Fig 1c)  cell-type UMAP with artifacts greyed
  - 04b_dotplot_celltype.png    (Fig 1d)  canonical-marker dotplot
  - 11_umap_marker_overlay.png  (Fig 2a)  6 functional markers (NOT 12)
  - 02_umap_clusters.png        (Fig 1b)  leiden clusters with white-bg labels

Source data:
  /Users/yichen/Desktop/collaboration/raw/cytof_output_v3/cytof_analyzed_v3.h5ad

Cluster → cell type mapping is taken verbatim from
cytof_advanced_analysis.py:58 (the canonical 17-cluster annotation).
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

OUT_DIR  = "./cytof_output_nature"
H5AD     = "/Users/yichen/Desktop/collaboration/raw/cytof_output_v3/cytof_analyzed_v3.h5ad"

# ── canonical mapping (cytof_advanced_analysis.py L58–76) ──────────────────
CLUSTER_ANNOTATION = {
    "0":  "Classical Monocyte",
    "1":  "Naive B",
    "2":  "CD8 Effector T",
    "3":  "CD4 Naive T",
    "4":  "CD4 Central Memory T",
    "5":  "Non-classical Monocyte",
    "6":  "NK cell",
    "7":  "CD4 Th1-like",
    "8":  "Classical Monocyte",
    "9":  "CD4 Effector Memory T",
    "10": "Inflammatory Monocyte",
    "11": "CD8 Naive T",
    "12": "γδ T cell",
    "13": "mDC",
    "14": "pDC",
    "15": "T-Myeloid doublets",
    "16": "CD16+ Granulocyte",
}
ARTIFACTS = {"T-Myeloid doublets", "CD16+ Granulocyte"}

# Stable palette (real cell types only; artifacts rendered grey)
CELLTYPE_PALETTE = {
    "CD4 Central Memory T":  "#1F77B4",
    "CD4 Effector Memory T": "#5B9BD5",
    "CD4 Naive T":           "#7DC1E8",
    "CD4 Th1-like":          "#AEC7E8",
    "CD8 Effector T":        "#9467BD",
    "CD8 Naive T":           "#C5B0D5",
    "Classical Monocyte":    "#E67E22",
    "Inflammatory Monocyte": "#D35400",
    "Non-classical Monocyte":"#F39C12",
    "NK cell":               "#27AE60",
    "Naive B":               "#16A085",
    "mDC":                   "#E74C3C",
    "pDC":                   "#C0392B",
    "γδ T cell":             "#F1C40F",
}
ARTIFACT_COL = "#BBBBBB"


# ── publication typography ─────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":            10.0,
    "axes.titlesize":       12.0,
    "axes.labelsize":       10.0,
    "xtick.labelsize":      9.0,
    "ytick.labelsize":      9.0,
    "axes.linewidth":       0.9,
    "xtick.major.width":    0.7,
    "ytick.major.width":    0.7,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
    "savefig.dpi":          300,
    "figure.dpi":           300,
})


print("Loading h5ad …")
a = ad.read_h5ad(H5AD)
a.obs_names_make_unique()
a.obs["cell_type"] = a.obs["leiden"].astype(str).map(CLUSTER_ANNOTATION).fillna("Unannotated")

# Canonical cell-type ordering (legend reading order)
CT_ORDER = [
    "CD4 Naive T", "CD4 Central Memory T", "CD4 Effector Memory T", "CD4 Th1-like",
    "CD8 Naive T", "CD8 Effector T", "γδ T cell",
    "Classical Monocyte", "Inflammatory Monocyte", "Non-classical Monocyte",
    "mDC", "pDC", "NK cell", "Naive B",
    "T-Myeloid doublets", "CD16+ Granulocyte",
]

X_umap = a.obsm["X_umap"]
print(f"  {a.n_obs:,} cells, UMAP shape {X_umap.shape}")


# ════════════════════════════════════════════════════════════════════════════
# Fig 1(c)  04_umap_celltype.png  — cell type UMAP with artifacts greyed
# ════════════════════════════════════════════════════════════════════════════
print("[1/4] 04_umap_celltype.png")
fig, ax = plt.subplots(figsize=(8.5, 7.0))

# Plot artifacts first (so real cells render on top)
for ct in ARTIFACTS:
    mask = a.obs["cell_type"].values == ct
    if mask.sum() == 0:
        continue
    ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
               s=1.4, c=ARTIFACT_COL, alpha=0.55,
               edgecolor="none", linewidths=0)
# real cell types
for ct in CT_ORDER:
    if ct in ARTIFACTS:
        continue
    mask = a.obs["cell_type"].values == ct
    if mask.sum() == 0:
        continue
    ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
               s=1.4, c=CELLTYPE_PALETTE[ct], alpha=0.85,
               edgecolor="none", linewidths=0)

# On-UMAP centroid labels REMOVED — the right-hand legend already lists every
# cell type in the same colour, so duplicating the labels on the cluster
# centroids was visually redundant.

ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Cell-type annotation", fontsize=12, pad=8)

# Legend (only canonical cell types — artifacts are rendered grey on the UMAP
# but not enumerated in the legend, since they are excluded from analyses
# per Methods §M3).
legend_handles = []
for ct in CT_ORDER:
    if ct in ARTIFACTS:
        continue
    legend_handles.append(
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=CELLTYPE_PALETTE[ct], markersize=7,
               label=ct)
    )
ax.legend(handles=legend_handles, loc="center left",
          bbox_to_anchor=(1.005, 0.5), ncol=1, frameon=False,
          fontsize=9.0, handletextpad=0.4, borderpad=0.2,
          labelspacing=0.45)

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/04_umap_celltype.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/04_umap_celltype.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("   saved")


# ════════════════════════════════════════════════════════════════════════════
# Fig 1(b)  02_umap_clusters.png  — leiden clusters w/ white-bg labels
# ════════════════════════════════════════════════════════════════════════════
print("[2/4] 02_umap_clusters.png")
LEIDEN_PALETTE = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
    "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85",
    "#3B9AB2", "#78B7C5", "#EBCC2A", "#E1AF00", "#F21A00",
    "#E2D200", "#46ACC8",
]

fig, ax = plt.subplots(figsize=(8.5, 7.0))
for cl in sorted(a.obs["leiden"].astype(int).unique()):
    mask = a.obs["leiden"].astype(int).values == cl
    ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
               s=1.4, c=LEIDEN_PALETTE[cl % len(LEIDEN_PALETTE)],
               alpha=0.85, edgecolor="none", linewidths=0)
    # cluster ID with white-bg circle for legibility on dense regions
    cx = np.median(X_umap[mask, 0])
    cy = np.median(X_umap[mask, 1])
    ax.text(cx, cy, str(cl), ha="center", va="center",
            fontsize=11, fontweight="bold", color="#111",
            bbox=dict(boxstyle="circle,pad=0.18",
                      facecolor="white", edgecolor="#444", lw=0.8),
            zorder=10)

ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Leiden clusters  (resolution = 0.8)", fontsize=12, pad=8)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/02_umap_clusters.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/02_umap_clusters.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("   saved")


# ════════════════════════════════════════════════════════════════════════════
# Fig 2(a)  11_umap_marker_overlay.png  — 6 functional markers (was 12)
# ════════════════════════════════════════════════════════════════════════════
print("[3/4] 11_umap_marker_overlay.png")
SELECTED_MARKERS = ["PD-1", "GranzymeB", "CD38", "Ki-67", "HLA-DR", "CXCR3"]
present = [m for m in SELECTED_MARKERS if m in a.var_names]
print(f"  Markers shown: {present}")

# Sequential colormap — light to deep red, no white at low end so cells are visible
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "marker_overlay",
    ["#E0E0E0", "#FAD7A0", "#F39C12", "#E67E22", "#C0392B", "#7B241C"],
    N=256,
)

ncols = 3
nrows = 2
fig, axes = plt.subplots(nrows, ncols, figsize=(11.0, 7.4))
for ai, mk in enumerate(present):
    ax = axes[ai // ncols][ai % ncols]
    expr = a[:, mk].X
    expr = expr.toarray().ravel() if hasattr(expr, "toarray") else np.asarray(expr).ravel()
    # arcsinh values clipped to [0, 99th percentile] for cleaner visualization
    vmax = np.percentile(expr, 99)
    expr_clip = np.clip(expr, 0, vmax)
    # Plot dim cells first, then bright cells on top (so high values are visible)
    order = np.argsort(expr_clip)
    sc = ax.scatter(X_umap[order, 0], X_umap[order, 1],
                    s=1.0, c=expr_clip[order], cmap=SEQ_CMAP,
                    alpha=0.75, edgecolor="none", linewidths=0,
                    vmin=0, vmax=vmax)
    ax.set_title(mk, fontsize=11, fontweight="bold", pad=5)
    ax.set_xticks([]); ax.set_yticks([])
    # compact colorbar inside the axes (top-right corner)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.012,
                        fraction=0.045)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.outline.set_linewidth(0.4)

# Hide any unused axes
for ai in range(len(present), nrows * ncols):
    axes[ai // ncols][ai % ncols].axis("off")

fig.suptitle("Functional-marker expression on UMAP (arcsinh-transformed)",
             fontsize=12, fontweight="bold", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(f"{OUT_DIR}/11_umap_marker_overlay.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/11_umap_marker_overlay.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("   saved")


print("[4/4] (dotplot is left untouched; current rendering acceptable)")
print("Done.")
