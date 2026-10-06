"""
cytof_replot_nature.py
======================
Loads the already-processed h5ad and regenerates ALL figures in Nature quality.
Run this AFTER cytof_pipeline_v3.py has completed.

Usage:
    cd <folder containing this script>
    python cytof_replot_nature.py

To add cell type annotation:
    1. Fill CLUSTER_ANNOTATION below
    2. Re-run
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")

try:
    import anndata as ad
    import scanpy as sc
    from cytof_plots_nature import (
        plot_cell_counts, plot_umap, plot_umap_clusters,
        plot_cluster_heatmap, plot_celltype_umap, plot_dotplot,
        plot_differential_abundance, plot_differential_markers,
        plot_within_celltype_markers,
        plot_stacked_bar, plot_sample_clustering, plot_umap_markers,
        plot_pairwise_abundance, plot_pairwise_volcano,
        plot_exhaustion_activation,
    )
except ImportError as e:
    sys.exit(f"[ERROR] {e}")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

H5AD_PATH = "./cytof_output_v3/cytof_analyzed_v3.h5ad"   # from pipeline_v3
OUT_DIR   = "./cytof_output_nature"

# ── Fill after reviewing 03_cluster_marker_heatmap.pdf ───────────────────────
# ── Annotation based on 11-cluster Harmony-corrected solution ────────────
# Based on 03_cluster_marker_heatmap.pdf (post-Harmony, resolution=0.8, 17 clusters)
CLUSTER_ANNOTATION = {
    "0":  "Classical Monocyte",       # CD11b++, CD11c++, CD14+, HLA-DR++, CD38++, CD39++
    "1":  "Naive B",                  # CD19++, CD20++, CD22++, IgD++, IgM++, HLA-DR++
    "2":  "CD8 Effector T",           # CD8a++, CD3++, GranzymeB+, CD95++
    "3":  "CD4 Naive T",              # CD4++, CD3++, CCR7++, CD27++, CD45RB++, CD127+
    "4":  "CD4 Central Memory T",     # CD4++, CD3++, CD95++, CD45RO+, CD127+, CCR4+
    "5":  "Non-classical Monocyte",   # CD11b++, CD56++, CD11c++, CD14+, CD38++
    "6":  "NK cell",                  # CD56++, CD16++, GranzymeB++, CD38++
    "7":  "CD4 Th1-like",             # CXCR3++, CCR4++, CD3+, CD4+, CCR6+, GranzymeB+
    "8":  "Classical Monocyte",       # CD11b++, CD11c++, CD14+, HLA-DR++ (sub-cluster of 0)
    "9":  "CD4 Effector Memory T",    # CD4++, CD95++, CCR4++, CD27++, ICOS+, CD25+
    "10": "Inflammatory Monocyte",    # CD11b++, CCR4++, CXCR3++, CD11c++, HLA-DR++, TIGIT+
    "11": "CD8 Naive T",              # CD8a++, CD3++, CCR7++, CD27++, CD45RA++, CD127+
    "12": "γδ T cell",                # CD3++, TCRgd++, GranzymeB+
    "13": "mDC",                      # HLA-DR++, CD11c++, CD16+, Tim-3+, CD38++, CD39++
    "14": "pDC",                      # CD123++, CD38++
    "15": "T-Myeloid doublets",       # n=101, CD3++CD4++CD14++CD11b++ mixed profile
    "16": "CD16+ Granulocyte",        # n=80, CD16++, TCF1++, CD11b+
}

# Canonical markers for dotplot (in display order)
CANONICAL_MARKERS = [
    "CD3", "CD4", "CD8a",
    "CD45RA", "CD45RO", "CCR7", "CD95",
    "CD25", "CD127",           # Treg
    "CXCR5", "ICOS",           # Tfh
    "CXCR3", "CCR4", "CCR6",  # Th1/2/17
    "GranzymeB", "PD-1", "TIGIT", "Tim-3", "TCF1",  # CD8 subsets
    "TCRgd",
    "CD19", "CD20", "IgD", "IgM", "CD38", "CD27", "CD24",
    "CD14", "CD16", "CD11c", "CD11b", "HLA-DR", "CD123", "CD66b",
    "CD56", "Ki-67",
]

# Biological marker order for heatmap (grouped by lineage)
MARKER_ORDER = [
    "CD45", "CD3", "CD4", "CD8a", "CD19", "CD20", "CD14",
    "CD56", "CD11c", "CD11b", "CD66b",
    "CCR7", "CD45RA", "CD45RO", "CD45RB", "CD95",
    "CD25", "CD127", "CD69", "CD27",
    "CXCR5", "ICOS", "CCR4", "CCR6", "CXCR3", "CD73",
    "PD-1", "TIGIT", "Tim-3", "CD39", "TCF1",
    "GranzymeB", "TCRgd", "CD16",
    "CD22", "IgD", "IgM", "CD38", "CD24", "HLA-DR", "CD123",
    "Ki-67",
]

# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[Loading] {H5AD_PATH} …")
if not os.path.exists(H5AD_PATH):
    sys.exit(f"[ERROR] File not found: {H5AD_PATH}\n"
             "Run cytof_pipeline_v3.py first.")
adata = ad.read_h5ad(H5AD_PATH)
os.makedirs(OUT_DIR, exist_ok=True)
print(f"  {adata.n_obs:,} cells  |  {adata.n_vars} markers  |  "
      f"{adata.obs['leiden'].nunique()} clusters")

# ══════════════════════════════════════════════════════════════════════════════
# APPLY ANNOTATION (if provided)
# ══════════════════════════════════════════════════════════════════════════════

if CLUSTER_ANNOTATION:
    adata.obs["cell_type"] = (
        adata.obs["leiden"].astype(str).map(CLUSTER_ANNOTATION).fillna("Unannotated")
    )
    print(f"  Cell types annotated: {sorted(adata.obs['cell_type'].unique())}")
else:
    adata.obs["cell_type"] = "Cluster_" + adata.obs["leiden"].astype(str)
    print("  [INFO] Using cluster IDs as cell types (fill CLUSTER_ANNOTATION to annotate)")

# ══════════════════════════════════════════════════════════════════════════════
# GENERATE ALL FIGURES
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[Plotting] Output → {OUT_DIR}/\n")

print("  Fig 0: Cell counts …")
plot_cell_counts(adata, OUT_DIR)

print("  Fig 1: UMAP group/sample …")
plot_umap(adata, OUT_DIR)

print("  Fig 2: UMAP clusters …")
plot_umap_clusters(adata, OUT_DIR)

print("  Fig 3: Cluster heatmap …")
plot_cluster_heatmap(adata, OUT_DIR, marker_order=MARKER_ORDER)

print("  Fig 4: Cell type UMAP …")
plot_celltype_umap(adata, OUT_DIR)

print("  Fig 4b: Dotplot …")
plot_dotplot(adata, CANONICAL_MARKERS, OUT_DIR)

print("  Fig 5: Differential abundance …")
plot_differential_abundance(adata, OUT_DIR)

print("  Fig 6/7: Differential markers …")
plot_differential_markers(adata, OUT_DIR)

print("  Fig 8: Within-cell-type differential markers …")
plot_within_celltype_markers(adata, OUT_DIR)

print("  Fig 9: Stacked bar composition …")
plot_stacked_bar(adata, OUT_DIR)

print("  Fig 10: Sample clustering heatmap …")
plot_sample_clustering(adata, OUT_DIR)

print("  Fig 11: UMAP marker overlays …")
plot_umap_markers(adata, OUT_DIR)

print("  Fig 12: Pairwise differential abundance …")
plot_pairwise_abundance(adata, OUT_DIR)

print("  Fig 13: Pairwise volcano plots …")
plot_pairwise_volcano(adata, OUT_DIR)

print("  Fig 14: Exhaustion/activation scores …")
plot_exhaustion_activation(adata, OUT_DIR)

print(f"""
══════════════════════════════════════════════════════════
  All Nature-quality figures saved to: {OUT_DIR}/
══════════════════════════════════════════════════════════
  00_cell_counts                    .pdf / .png
  01_umap_group_sample              .pdf / .png
  02_umap_clusters                  .pdf / .png
  03_cluster_marker_heatmap         .pdf / .png
  04_umap_celltype                  .pdf / .png
  04b_dotplot_celltype              .pdf / .png
  05_differential_abundance         .pdf / .png
  06_differential_marker_heatmap    .pdf / .png
  07_top_marker_violins             .pdf / .png
  08_within_celltype_markers        .pdf / .png
  09_stacked_bar_composition        .pdf / .png
  10_sample_clustering_heatmap      .pdf / .png
  11_umap_marker_overlay            .pdf / .png
  12_pairwise_abundance_forest      .pdf / .png
  13_pairwise_volcano               .pdf / .png
  14_exhaustion_activation_scores   .pdf / .png
""")
