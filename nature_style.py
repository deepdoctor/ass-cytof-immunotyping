"""
nature_style.py — Nature-quality matplotlib configuration for CyTOF figures
============================================================================
- No white/gray cells in heatmaps
- Publication-ready typography, sizing, and color palettes
- All figures exported at 300 DPI as PDF + PNG
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL RCPARAMS  (Nature-style)
# ══════════════════════════════════════════════════════════════════════════════

def apply_nature_style():
    rcParams.update({
        # Font — large; designed to remain readable after the
        # source-PNG → composite → Word-down-scaling pipeline.
        "font.family":          "sans-serif",
        "font.sans-serif":      ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":            22,
        "axes.titlesize":       26,
        "axes.labelsize":       22,
        "xtick.labelsize":      20,
        "ytick.labelsize":      20,
        "legend.fontsize":      20,
        "legend.title_fontsize":22,

        # Lines & patches (thicker to match the larger fonts)
        "axes.linewidth":       2.0,
        "xtick.major.width":    2.0,
        "ytick.major.width":    2.0,
        "xtick.major.size":     7.0,
        "ytick.major.size":     7.0,
        "lines.linewidth":      2.2,
        "patch.linewidth":      1.4,

        # Axes
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.edgecolor":       "#333333",
        "xtick.color":          "#333333",
        "ytick.color":          "#333333",
        "axes.labelcolor":      "#333333",
        "text.color":           "#333333",

        # Figure
        "figure.dpi":           300,
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.05,
        "figure.facecolor":     "white",
        "axes.facecolor":       "white",
        "savefig.facecolor":    "white",

        # Legend
        "legend.frameon":       False,
        "legend.borderpad":     0.3,
        "legend.handlelength":  1.0,

        # PDF font embedding
        "pdf.fonttype":         42,
        "ps.fonttype":          42,
    })

apply_nature_style()

# ══════════════════════════════════════════════════════════════════════════════
# COLOR PALETTES
# ══════════════════════════════════════════════════════════════════════════════

# Group palette — muted, Nature-journal style
GROUP_PALETTE = {
    "Jo-1":  "#C0392B",   # deep red
    "PL-12": "#2471A3",   # steel blue
    "EJ":    "#1A7A4A",   # forest green
    "PL-7":  "#7D3C98",   # purple
}

# Cluster palette (tab20 but shifted to avoid grays)
CLUSTER_PALETTE = [
    "#E64B35","#4DBBD5","#00A087","#3C5488","#F39B7F",
    "#8491B4","#91D1C2","#DC0000","#7E6148","#B09C85",
    "#3B9AB2","#78B7C5","#EBCC2A","#E1AF00","#F21A00",
    "#E2D200","#46ACC8","#E58601","#B40F20","#0B775E",
    "#35274A","#F2300F","#ECCBAE","#D69C4E",
]

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM COLORMAPS  — no white, no gray
# ══════════════════════════════════════════════════════════════════════════════

# Heatmap: deep blue → gold → deep red  (diverging, never white)
HEATMAP_DIVERG = LinearSegmentedColormap.from_list(
    "nature_heatmap",
    ["#1A3A6B", "#2471A3", "#5DADE2", "#F7DC6F", "#E67E22", "#C0392B", "#7B241C"],
    N=256
)

# Heatmap: for single-direction data (e.g. mean expression, always positive)
HEATMAP_SEQ = LinearSegmentedColormap.from_list(
    "nature_seq",
    ["#1C2833", "#1A5276", "#2E86C1", "#7FB3D3", "#F9E79F", "#F39C12", "#922B21"],
    N=256
)

# Subtle background for UMAP density — switched to pure white per submission spec
UMAP_BG = "#FFFFFF"


def save(fig, path_no_ext, out_dir):
    """Save figure as both PDF and PNG."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(f"{out_dir}/{path_no_ext}.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
