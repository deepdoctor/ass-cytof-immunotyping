"""
cytof_composite_figures.py
==========================
Build 5 Main Figure composites (6 panels each, image-tiling layout):
  Fig 1 — Cohort + atlas
  Fig 2 — Marker characterization & functional states
  Fig 3 — Compositional heterogeneity
  Fig 4 — Three immunotypes + clinical recapitulation
  Fig 5 — Differential markers

Note: Figure 6 (external validation) is built natively in cytof_mainfig6_rebuild.py
— it is NOT a composite of pre-rendered PNGs. Figure 7 (prospective experimental
validation) was removed; the planned-experiment content was moved to a Discussion
paragraph in MANUSCRIPT_FULL.md.

Panel layout: 3 columns × 2 rows (6 panels) per figure.

Run:
    cd /Users/yichen/Desktop/collaboration/raw
    /Users/yichen/anaconda3/bin/conda run -n cytof python cytof_composite_figures.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

OUT_DIR = "./cytof_output_nature"

# Each Main Fig: list of 4 panels (post-Nature-style restructure).
# Panels dropped from earlier 6-panel composites are noted; redundant or
# QC-only content has been moved to Supplementary, and the most informative
# panels now drive each main figure.
MAIN_FIGURES = [
    {
        "name":  "MainFig1_atlas",
        "title": "Figure 1 · Cohort and high-dimensional single-cell atlas of antisynthetase syndrome",
        # Dropped: (a) cell counts → Supp Fig S1 (QC-only);
        #          (f) cluster × marker heatmap → Supp Fig S6 (redundant with dotplot).
        "panels": [
            ("a", "01_umap_group_sample.png",      "UMAP by autoantibody group / by sample"),
            ("b", "02_umap_clusters.png",          "UMAP by Leiden cluster (resolution 0.8)"),
            ("c", "04_umap_celltype.png",          "UMAP by 16 annotated cell types"),
            ("d", "04b_dotplot_celltype.png",      "Canonical lineage marker dotplot"),
        ],
    },
    {
        "name":  "MainFig2_marker_characterization",
        "title": "Figure 2 · Functional-state characterization across immune populations",
        # Dropped: (d) exhaustion/activation scores → Supp (all KW p > 0.2, no signal);
        #          (f) trajectory + PAGA → Supp (independent story; doesn't fit panel).
        "panels": [
            ("a", "11_umap_marker_overlay.png",         "Functional markers on UMAP"),
            ("b", "06_differential_marker_heatmap.png", "Global differential-marker heatmap (KW + BH-FDR)"),
            ("c", "07_top_marker_violins.png",          "Top-discriminatory-marker violins"),
            ("d", "28_marker_coexpression_modules.png", "Patient-level marker co-expression modules"),
        ],
    },
    # Figure 3 is rendered natively (not as a composite of PNGs).
    # See cytof_mainfig3_rebuild.py.
    {
        "name":  "MainFig4_immunotypes",
        "title": "Figure 4 · Three serology-orthogonal immunotypes recapitulate clinical phenotype (core finding)",
        # Dropped: (d) signature score heatmap → Supp (redundant with integrated heatmap);
        #          (e) Spearman correlation matrix → Supp (exploratory at n=10).
        # The bootstrap (formerly Fig 6a) is now panel (d) of this figure — it
        # belongs with the immunotype discovery, not with external validation.
        "panels": [
            ("a", "21_immunotype_discovery.png",          "Ward clustering → IT1 / IT2 / IT3"),
            ("b", "23_immunotype_clinical_severity.png",  "Clinical phenotype per immunotype"),
            ("c", "15_clinical_immune_heatmap.png",       "Integrated clinical-immune heatmap"),
            ("d", "24_bootstrap_stability.png",           "Bootstrap co-clustering stability (B = 1,000)"),
        ],
    },
    {
        "name":  "MainFig5_DE",
        "title": "Figure 5 · Differential expression across autoantibody classes",
        # Restructure: promote the patient-level cluster-bootstrap forest to a
        # core position (it is the key statistical contribution of this study);
        # drop (c) pairwise cell-level volcano (redundant with (d) per-cell-type
        # volcano) and (f) abundance forest (now in Fig 3d).
        "panels": [
            ("a", "08_within_celltype_markers.png",      "Within-cell-type differential expression"),
            ("b", "08b_within_celltype_violins.png",     "Within-cell-type marker violins"),
            ("c", "27b_DE_bootstrap_CI_forest.png",      "Patient-level cluster-bootstrap Cohen's d (B = 2,000)"),
            ("d", "20_percelltype_volcano_EJvsPL7.png",  "Per-cell-type volcano (EJ vs PL-7)"),
        ],
    },
    # Figure 6 is rendered natively (not as a composite of PNGs).
    # See cytof_mainfig6_rebuild.py.
]


def build_main_composite(spec, out_dir, dpi=300):
    """Build a composite figure from N source PNGs.
    Layout adapts to panel count:
      - 4 panels → 2×2 grid (figsize 11×11)
      - 6 panels → 3×2 grid (figsize 11×14.5; legacy)
    Compact total size ensures source-text remains readable when the composite
    is embedded at ~6.8 in width in a Word document.
    """
    n = len(spec["panels"])
    if n == 4:
        # Bigger panels, tighter layout for 4-panel composites
        nrows, ncols, figsize = 2, 2, (13.0, 11.5)
    elif n == 6:
        nrows, ncols, figsize = 3, 2, (11.0, 14.5)
    else:
        nrows = (n + 1) // 2
        ncols = 2
        figsize = (11.0, 4.8 * nrows + 0.8)
    fig = plt.figure(figsize=figsize, facecolor="white")
    # Tighter spacing — less margin around panels and between panels
    gs  = gridspec.GridSpec(nrows, ncols, figure=fig,
                            hspace=0.06 if nrows == 2 else 0.18,
                            wspace=0.03,
                            left=0.025, right=0.998,
                            top=0.965 if nrows == 2 else 0.965,
                            bottom=0.005)
    for i, (lbl, png, _caption) in enumerate(spec["panels"][:nrows * ncols]):
        r, c = i // ncols, i % ncols
        ax = fig.add_subplot(gs[r, c])
        path = os.path.join(out_dir, png)
        if os.path.exists(path):
            ax.imshow(mpimg.imread(path), interpolation="hanning")
        else:
            ax.text(0.5, 0.5, f"missing\n{png}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=16, color="#888")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_facecolor("white")
        # Panel label inside the panel's top-left corner so it doesn't
        # consume vertical space outside the imshow.
        ax.text(0.012, 0.985, lbl, transform=ax.transAxes,
                fontsize=30, fontweight="bold", color="#111",
                ha="left", va="top", family="sans-serif",
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.85, boxstyle="round,pad=0.10"))
    fig.suptitle(spec["title"], fontsize=20, fontweight="bold",
                 color="#111", y=0.998)
    out_pdf = os.path.join(out_dir, spec["name"] + ".pdf")
    out_png = os.path.join(out_dir, spec["name"] + ".png")
    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"   {spec['name']}.pdf / .png  saved")


if __name__ == "__main__":
    print("Building Main Figure composites …")
    for spec in MAIN_FIGURES:
        print(f"  {spec['name']}")
        build_main_composite(spec, OUT_DIR)
    print("\nFigure 6 must be built separately by running:")
    print("    python cytof_mainfig6_rebuild.py")
    print("\nDone. Composites in:", OUT_DIR)
