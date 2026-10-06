"""
Native rebuild of Fig 1(d): canonical-marker × cell-type dotplot.

Fixes vs prior render (scanpy default):
  - x-axis marker labels were too small (~6pt) → bumped to 9pt with
    45° rotation.
  - dot-size legend ('% expressing') had near-isoluminant size steps
    → custom legend with 4 clearly separated sizes.
  - colour bar relabelled to indicate arcsinh expression (not raw).
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

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
ARTIFACTS = {"T-Myeloid doublets", "CD16+ Granulocyte"}

# Canonical markers — what cytof_pipeline_v3.py uses
CANONICAL = [
    "CD3","CD4","CD8a","CD45RA","CD45RO","CCR7","CD27","CD127",
    "CXCR5","ICOS","GranzymeB","PD-1","TIGIT","Tim-3","TCF1",
    "CD19","CD20","IgD","IgM","CD38",
    "CD14","CD16","CD11c","CD123","HLA-DR","CD66b","CD56","TCRgd","Ki-67",
]

# Cell-type display order (rows)
CT_ORDER = [
    "CD4 Naive T", "CD4 Central Memory T", "CD4 Effector Memory T", "CD4 Th1-like",
    "CD8 Naive T", "CD8 Effector T", "γδ T cell",
    "Classical Monocyte", "Inflammatory Monocyte", "Non-classical Monocyte",
    "mDC", "pDC", "NK cell", "Naive B",
]

plt.rcParams.update({
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":            10.0,
    "axes.titlesize":       12.0,
    "axes.labelsize":       10.0,
    "xtick.labelsize":      9.0,
    "ytick.labelsize":      9.5,
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

# Filter out artifacts and only keep canonical CTs
mask_keep = a.obs["cell_type"].isin(CT_ORDER)
a_real = a[mask_keep].copy()
print(f"  Real cells: {a_real.n_obs:,} (artifacts dropped)")

# only markers that are present
markers_present = [m for m in CANONICAL if m in a_real.var_names]
print(f"  Markers shown: {len(markers_present)}")

# Compute per-cell-type per-marker stats:
# - mean arcsinh expression
# - % cells with non-zero expression (proxy for "expressing")
ct_idx = {ct: np.where(a_real.obs["cell_type"].values == ct)[0]
          for ct in CT_ORDER}

mean_mat = np.zeros((len(CT_ORDER), len(markers_present)))
pct_mat  = np.zeros((len(CT_ORDER), len(markers_present)))
for i, ct in enumerate(CT_ORDER):
    idx = ct_idx[ct]
    if len(idx) == 0:
        continue
    Xsub = a_real.X[idx]
    Xsub = Xsub.toarray() if hasattr(Xsub, "toarray") else np.asarray(Xsub)
    for j, mk in enumerate(markers_present):
        col_idx = list(a_real.var_names).index(mk)
        vals = Xsub[:, col_idx]
        mean_mat[i, j] = vals.mean()
        # threshold for "expressing": top 50% of arcsinh distribution above 0.5
        pct_mat[i, j] = (vals > 0.5).mean() * 100

# Z-score each marker (col) for color scale
z = (mean_mat - mean_mat.mean(axis=0, keepdims=True)) / \
    (mean_mat.std(axis=0, keepdims=True) + 1e-9)

# ── render ────────────────────────────────────────────────────────────────
HEAT = LinearSegmentedColormap.from_list(
    "dot",
    ["#1F618D", "#5DADE2", "#F4D03F", "#E67E22", "#C0392B"],
    N=256,
)

n_ct = len(CT_ORDER); n_mk = len(markers_present)
fig, ax = plt.subplots(figsize=(0.34 * n_mk + 1.5, 0.42 * n_ct + 1.4))

# normalise dot sizes: pct → marker area (35..380)
size_min, size_max = 30, 380
sizes = size_min + (pct_mat / 100.0) * (size_max - size_min)

# colour normalise z to [-2, +2]
vmin, vmax = -2.0, 2.0
z_clip = np.clip(z, vmin, vmax)

for i in range(n_ct):
    for j in range(n_mk):
        col = HEAT((z_clip[i, j] - vmin) / (vmax - vmin))
        ax.scatter(j, n_ct - 1 - i, s=sizes[i, j], c=[col],
                   edgecolor="white", linewidth=0.4)

ax.set_xticks(range(n_mk))
ax.set_xticklabels(markers_present, rotation=45, ha="right", fontsize=9.0)
ax.set_yticks(range(n_ct))
ax.set_yticklabels(CT_ORDER[::-1], fontsize=9.5)
ax.set_xlim(-0.6, n_mk - 0.4)
ax.set_ylim(-0.6, n_ct - 0.4)

# colour bar
import matplotlib.cm as cm
import matplotlib.colors as mcolors
sm = cm.ScalarMappable(cmap=HEAT, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
cbar = fig.colorbar(sm, ax=ax, fraction=0.018, pad=0.10, shrink=0.55)
cbar.set_label("Z-score (arcsinh expression)", fontsize=9.5)
cbar.ax.tick_params(labelsize=8.5)
cbar.outline.set_linewidth(0.4)

# size legend  — 4 clearly separated steps
size_legend_pcts = [10, 30, 60, 90]
legend_handles = [
    Line2D([0], [0], marker="o", color="w",
           markerfacecolor="#888", markeredgecolor="white",
           markersize=np.sqrt(size_min + (pct/100.0)*(size_max-size_min)) / 1.2,
           label=f"{pct} %")
    for pct in size_legend_pcts
]
ax.legend(handles=legend_handles, loc="upper left",
          bbox_to_anchor=(1.005, 0.42),
          frameon=False, fontsize=8.5,
          title="% cells with arcsinh > 0.5",
          title_fontsize=9.0,
          handletextpad=0.6, labelspacing=0.55,
          borderpad=0.2)

ax.set_title("Canonical marker expression by cell type", fontsize=11.5, pad=8)
ax.grid(color="#eee", lw=0.3)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/04b_dotplot_celltype.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/04b_dotplot_celltype.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR}/04b_dotplot_celltype.png")
