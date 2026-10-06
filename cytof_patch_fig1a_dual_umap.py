"""
Native rebuild of Fig 1(a): dual UMAP coloured by autoantibody group (left)
and by individual sample (right), equal-sized, with a clearly distinguishable
10-sample palette.

Fixes vs prior render:
  - left/right UMAPs were unequal sizes (manual scaling artefact) → now in
    a uniform GridSpec with shared aspect ratio.
  - sample legend used tab10/Set3 with multiple near-isoluminant pairs
    (FH0013/FH0014, FH0011/FH0009) → replaced with a 10-colour palette
    selected for max perceptual contrast (custom ColorBrewer-derived).
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

OUT_DIR  = "./cytof_output_nature"
H5AD     = "/Users/yichen/Desktop/collaboration/raw/cytof_output_v3/cytof_analyzed_v3.h5ad"

GROUP_PALETTE = {"Jo-1":  "#C0392B", "PL-12": "#2471A3",
                 "EJ":    "#1A7A4A", "PL-7":  "#7D3C98"}
GROUP_ORDER = ["Jo-1", "PL-12", "EJ", "PL-7"]

# 10 visually distinct colours (no near-isoluminant pairs).
SAMPLE_PALETTE_10 = [
    "#E41A1C",  # red
    "#377EB8",  # blue
    "#4DAF4A",  # green
    "#FF7F00",  # orange
    "#984EA3",  # purple
    "#FFFF33",  # yellow
    "#A65628",  # brown
    "#F781BF",  # pink
    "#999999",  # grey
    "#1B9E77",  # teal
]

plt.rcParams.update({
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":            10.0,
    "axes.titlesize":       12.0,
    "axes.labelsize":       10.0,
    "axes.linewidth":       0.9,
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
    "savefig.dpi":          300,
    "figure.dpi":           300,
})

print("Loading h5ad …")
a = ad.read_h5ad(H5AD)
a.obs_names_make_unique()
X = a.obsm["X_umap"]
samples = sorted(a.obs["sample"].unique())
print(f"  {a.n_obs:,} cells; {len(samples)} samples; {a.obs['group'].nunique()} groups")

# Square-ish aspect ratio so the panel fits a 2×2 composite cell well.
fig = plt.figure(figsize=(9.0, 8.4), facecolor="white")
gs = gridspec.GridSpec(
    2, 2, figure=fig,
    left=0.06, right=0.985, top=0.94, bottom=0.085,
    wspace=0.06, hspace=0.06,
    height_ratios=[5.0, 0.95],
)

# ── (left)  by group ────────────────────────────────────────────────────────
ax_g = fig.add_subplot(gs[0, 0])
for g in GROUP_ORDER:
    mask = a.obs["group"].values == g
    ax_g.scatter(X[mask, 0], X[mask, 1], s=1.4,
                 c=GROUP_PALETTE[g], alpha=0.85,
                 edgecolor="none", linewidths=0,
                 label=g)
ax_g.set_xticks([]); ax_g.set_yticks([])
ax_g.set_title("Autoantibody group", fontsize=13, pad=4)
ax_g.set_xlabel("UMAP 1", fontsize=11)
ax_g.set_ylabel("UMAP 2", fontsize=11)
ax_g.set_aspect("equal", adjustable="datalim")

# ── (right) by sample ──────────────────────────────────────────────────────
ax_s = fig.add_subplot(gs[0, 1])
for i, s in enumerate(samples):
    short = s.replace("L01912_", "")
    mask = a.obs["sample"].values == s
    ax_s.scatter(X[mask, 0], X[mask, 1], s=1.4,
                 c=SAMPLE_PALETTE_10[i % 10], alpha=0.78,
                 edgecolor="none", linewidths=0,
                 label=short)
ax_s.set_xticks([]); ax_s.set_yticks([])
ax_s.set_title("Individual sample", fontsize=13, pad=4)
ax_s.set_xlabel("UMAP 1", fontsize=11)
ax_s.set_ylabel("", fontsize=11)
ax_s.set_aspect("equal", adjustable="datalim")

# share x/y limits for visual parity
xlim = (min(ax_g.get_xlim()[0], ax_s.get_xlim()[0]),
        max(ax_g.get_xlim()[1], ax_s.get_xlim()[1]))
ylim = (min(ax_g.get_ylim()[0], ax_s.get_ylim()[0]),
        max(ax_g.get_ylim()[1], ax_s.get_ylim()[1]))
ax_g.set_xlim(xlim); ax_g.set_ylim(ylim)
ax_s.set_xlim(xlim); ax_s.set_ylim(ylim)

# ── horizontal legends in dedicated bottom row ────────────────────────────
ax_leg_g = fig.add_subplot(gs[1, 0])
ax_leg_g.axis("off")
group_handles = [
    Line2D([0], [0], marker="o", color="w",
           markerfacecolor=GROUP_PALETTE[g], markersize=11, label=g)
    for g in GROUP_ORDER
]
ax_leg_g.legend(
    handles=group_handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
    fontsize=10.5, frameon=False, ncol=4,
    handletextpad=0.4, borderpad=0.2, columnspacing=1.6,
)

ax_leg_s = fig.add_subplot(gs[1, 1])
ax_leg_s.axis("off")
sample_handles = [
    Line2D([0], [0], marker="o", color="w",
           markerfacecolor=SAMPLE_PALETTE_10[i % 10], markersize=11,
           label=s.replace("L01912_", ""))
    for i, s in enumerate(samples)
]
ax_leg_s.legend(
    handles=sample_handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
    fontsize=8.5, frameon=False, ncol=5,
    handletextpad=0.3, borderpad=0.2, columnspacing=0.7,
    labelspacing=0.4,
)

fig.savefig(f"{OUT_DIR}/01_umap_group_sample.pdf",
            dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/01_umap_group_sample.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR}/01_umap_group_sample.png")
