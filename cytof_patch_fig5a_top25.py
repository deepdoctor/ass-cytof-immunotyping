"""
Rebuild 08_within_celltype_markers.png as a publication-grade
top-25-marker heatmap.

The original render packs 14 cell types × all 42 markers (588 cells) with
~5 pt labels and substantial 'not tested (n<20/group)' grey area, which
diluted the real signal and was illegible at composite scale.

This rebuild:
  - selects the top 25 (cell_type × marker) pairs by -log10(padj),
    dedup-bound to up to 25 unique markers shown
  - renders cell_type on rows (compact), marker on columns
  - rotates marker labels 45° for clarity
  - caps the colorbar at 50 (-log10 q ≥ 50 saturates), avoiding the
    300-vs-100 visual flattening
  - drops the grey 'not tested' rectangles entirely
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "./cytof_output_nature"

plt.rcParams.update({
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":            7.0,
    "axes.titlesize":       8.0,
    "axes.labelsize":       7.0,
    "xtick.labelsize":      6.5,
    "ytick.labelsize":      6.5,
    "axes.linewidth":       0.7,
    "xtick.major.width":    0.6,
    "ytick.major.width":    0.6,
    "xtick.major.size":     2.5,
    "ytick.major.size":     2.5,
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
    "savefig.dpi":          300,
    "figure.dpi":           300,
})

CELL_TYPES = [
    "CD4 Central Memory T", "CD4 Effector Memory T", "CD4 Naive T",
    "CD4 Th1-like", "CD8 Effector T", "CD8 Naive T",
    "Classical Monocyte", "Inflammatory Monocyte", "Non-classical Monocyte",
    "NK cell", "Naive B", "mDC", "pDC", "γδ T cell",
]

de = pd.read_csv(f"{OUT_DIR}/08_within_celltype_markers.csv")
print(f"Loaded {len(de)} rows; {de['padj'].notna().sum()} with valid padj")

# Filter to canonical cell types (drop artifacts)
de = de[de["cell_type"].isin(CELL_TYPES)].copy()

# Effective -log10(padj), capped at 50 to avoid visual saturation
de["padj_clip"] = de["padj"].clip(lower=1e-50, upper=1.0)
de["neglog_q"]  = -np.log10(de["padj_clip"])

# Pick top 25 unique markers ranked by max -log10(padj) across cell types
mk_top = (de.groupby("marker")["neglog_q"].max()
            .sort_values(ascending=False).head(25).index.tolist())

# Build matrix: cell type × marker
M = de.pivot_table(index="cell_type", columns="marker",
                    values="neglog_q", aggfunc="max")
M = M.reindex(index=CELL_TYPES, columns=mk_top)

# Diverging-style sequential colormap: white-low to deep red-high (no grey)
from matplotlib.colors import LinearSegmentedColormap
HEAT = LinearSegmentedColormap.from_list(
    "fdr_seq",
    ["#FAEBD7", "#F4A261", "#E76F51", "#C0392B", "#7B241C"],
    N=256,
)

fig, ax = plt.subplots(figsize=(7.0, 4.4))
im = ax.imshow(M.values, cmap=HEAT, vmin=0, vmax=50, aspect="auto")

# Threshold marker for q=0.05 in the colorbar
THR_NEGLOG = -np.log10(0.05)  # ≈ 1.30

ax.set_xticks(np.arange(len(mk_top)))
ax.set_xticklabels(mk_top, rotation=45, ha="right", fontsize=6.5)
ax.set_yticks(np.arange(len(CELL_TYPES)))
ax.set_yticklabels(CELL_TYPES, fontsize=6.5)

# Light grid lines between cells
ax.set_xticks(np.arange(-0.5, len(mk_top), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(CELL_TYPES), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=0.6)
ax.tick_params(which="minor", length=0)

# Annotate "n.s." on cells where padj > 0.05 (very few expected at top 25)
for i, ct in enumerate(CELL_TYPES):
    for j, mk in enumerate(mk_top):
        v = M.values[i, j]
        if np.isnan(v):
            ax.text(j, i, "·", ha="center", va="center",
                    fontsize=5.5, color="#888")
        elif v < THR_NEGLOG:
            ax.text(j, i, "ns", ha="center", va="center",
                    fontsize=4.8, color="#444")

cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.012, shrink=0.85)
cbar.set_label(r"$-\log_{10}$  q (BH-FDR; capped at 50)", fontsize=6.5)
cbar.ax.tick_params(labelsize=5.8)
cbar.outline.set_linewidth(0.4)

ax.set_title("Within-cell-type differential markers — top 25 markers "
             "(Kruskal–Wallis, BH-FDR; full 42-marker matrix in Supp. Fig. S12)",
             fontsize=7.5, pad=6)

fig.tight_layout()
out_pdf = f"{OUT_DIR}/08_within_celltype_markers.pdf"
out_png = f"{OUT_DIR}/08_within_celltype_markers.png"
fig.savefig(out_pdf, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
fig.savefig(out_png, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
