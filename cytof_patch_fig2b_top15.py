"""
Native rebuild of Fig 2(b): global differential-marker heatmap restricted
to the top 15 markers by KW statistic (was: all 42 markers, illegible at
composite scale).

Reads precomputed group means and KW statistics from
06_differential_markers.csv. Z-scores each marker across the four
autoantibody groups and renders a top-15 heatmap with autoantibody
colour brackets above the matrix and clear annotation.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

OUT_DIR = "./cytof_output_nature"

plt.rcParams.update({
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":            10.0,
    "axes.titlesize":       12.0,
    "axes.labelsize":       10.0,
    "xtick.labelsize":      9.5,
    "ytick.labelsize":      9.5,
    "axes.linewidth":       0.9,
    "xtick.major.width":    0.7,
    "ytick.major.width":    0.7,
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
    "savefig.dpi":          300,
    "figure.dpi":           300,
})

GROUP_PALETTE = {"Jo-1": "#C0392B", "PL-12": "#2471A3",
                 "EJ":   "#1A7A4A", "PL-7":  "#7D3C98"}
GROUP_ORDER = ["Jo-1", "PL-12", "EJ", "PL-7"]

de = pd.read_csv(f"{OUT_DIR}/06_differential_markers.csv")
print(f"Loaded {len(de)} markers; padj distribution: "
      f"{(de['padj']<0.05).sum()} at FDR<0.05")

# Top 15 by KW statistic
top15 = de.sort_values("KW_stat", ascending=False).head(15).copy()

# Build z-score matrix: marker × group
mean_cols = [f"mean_{g}" for g in GROUP_ORDER]
M_mean = top15[mean_cols].values
# row-wise z-score
M_z = (M_mean - M_mean.mean(axis=1, keepdims=True)) / (M_mean.std(axis=1, keepdims=True) + 1e-9)

# Diverging colormap (no white)
HEAT = LinearSegmentedColormap.from_list(
    "heat_div",
    ["#1A3A6B", "#2980B9", "#7FB3D3", "#F4D03F", "#E67E22", "#C0392B", "#7B241C"],
    N=256,
)

# Cluster markers by similarity (optional: hierarchical sort)
from scipy.cluster.hierarchy import linkage, leaves_list
Z = linkage(M_z, method="average")
order = leaves_list(Z)
M_z = M_z[order]
markers = top15["marker"].iloc[order].values
padjs   = top15["padj"].iloc[order].values

# ── render ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.6, 6.4))
im = ax.imshow(M_z, cmap=HEAT, vmin=-2, vmax=2, aspect="auto")

# group brackets along top
# Using a unicode non-breaking hyphen (U+2011) avoids the bold-Arial visual
# kerning collapse where '−1' renders as a 'T' shape at small sizes.
GROUP_LABEL = {"Jo-1":  "Jo‑1",
               "PL-12": "PL‑12",
               "EJ":    "EJ",
               "PL-7":  "PL‑7"}
for i, g in enumerate(GROUP_ORDER):
    ax.add_patch(Rectangle((i - 0.45, -0.85), 0.90, 0.45,
                            facecolor=GROUP_PALETTE[g],
                            edgecolor="white", lw=0.5,
                            transform=ax.transData, clip_on=False))
    ax.text(i, -1.18, GROUP_LABEL[g], ha="center", va="bottom",
            fontsize=10.5, fontweight="bold",
            color=GROUP_PALETTE[g], clip_on=False)

ax.set_xticks(range(len(GROUP_ORDER)))
ax.set_xticklabels([])  # the bracket above is the label
ax.set_yticks(range(len(markers)))
ax.set_yticklabels(markers, fontsize=9.5)

# annotate q on right of each row using mathtext (no Unicode superscript glyphs)
for i, q in enumerate(padjs):
    if q == 0 or q < 1e-300:
        q_str = r"$P < 10^{-300}$"
    elif q < 1e-4:
        exp = int(np.floor(np.log10(q)))
        coef = q / 10**exp
        q_str = rf"$P = {coef:.1f}\times 10^{{{exp}}}$"
    else:
        q_str = f"P = {q:.1e}"
    ax.text(len(GROUP_ORDER) - 0.4, i, q_str, ha="left", va="center",
            fontsize=7.5, color="#444")

# colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.20, shrink=0.85)
cbar.set_label("Z-score (mean arcsinh)", fontsize=9.5)
cbar.ax.tick_params(labelsize=8.5)
cbar.outline.set_linewidth(0.4)

ax.set_title("Top 15 differential markers across\n"
             "autoantibody groups (Kruskal–Wallis)",
             fontsize=11, pad=28)

# light grid lines between cells
ax.set_xticks(np.arange(-0.5, len(GROUP_ORDER), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(markers), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=0.7)
ax.tick_params(which="minor", length=0)

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/06_differential_marker_heatmap.pdf",
            dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/06_differential_marker_heatmap.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR}/06_differential_marker_heatmap.png")
