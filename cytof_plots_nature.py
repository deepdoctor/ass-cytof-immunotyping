"""
cytof_plots_nature.py — Nature-quality figure generation for CyTOF pipeline
=============================================================================
Drop-in replacement for all plotting sections in cytof_pipeline_v3.py.

Usage:
    from cytof_plots_nature import *
    plot_cell_counts(adata, OUT_DIR)
    plot_umap(adata, OUT_DIR)
    ...
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
import scanpy as sc

# ── apply Nature rcParams ─────────────────────────────────────────────────────
from nature_style import (
    apply_nature_style, GROUP_PALETTE, CLUSTER_PALETTE,
    HEATMAP_DIVERG, HEATMAP_SEQ, UMAP_BG, save
)

def fmt_p(p, prefix="P"):
    """Publication-grade p-value formatter.
    Replaces the "FDR=0.0e+00" atrocities produced by `:.1e` on under-flowed
    floats. Returns e.g. 'P < 10⁻³⁰⁰', 'P = 6.0×10⁻¹⁰⁰', 'P = 0.024', 'P = 0.31'.
    """
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return f"{prefix} = n/a"
    if p == 0 or p < 1e-300:
        return f"{prefix} < 10$^{{-300}}$"
    if p < 1e-4:
        exp = int(np.floor(np.log10(p)))
        coef = p / 10**exp
        return f"{prefix} = {coef:.1f}×10$^{{{exp}}}$"
    if p < 1e-2:
        return f"{prefix} = {p:.1e}"
    if p < 0.05:
        return f"{prefix} = {p:.3f}"
    return f"{prefix} = {p:.2f}"
apply_nature_style()

# Nature single-column = 88 mm = 3.46 in
# Nature double-column = 180 mm = 7.09 in
SC = 3.46   # inches, single column
DC = 7.09   # inches, double column


# ══════════════════════════════════════════════════════════════════════════════
# FIG 0 — Cell counts per sample
# ══════════════════════════════════════════════════════════════════════════════

def plot_cell_counts(adata, out_dir):
    # One row per sample — avoid cartesian-product bug from groupby(sample, group)
    count_df = (
        adata.obs.groupby("sample")
        .agg(n=("sample", "count"), group=("group", "first"))
        .reset_index()
        .sort_values("group")
    )

    fig, ax = plt.subplots(figsize=(DC, 2.0))
    bar_colors = [GROUP_PALETTE.get(g, "#888") for g in count_df["group"]]
    bars = ax.bar(range(len(count_df)), count_df["n"],
                  color=bar_colors, edgecolor="white", linewidth=0.4, width=0.7)

    ax.set_xticks(range(len(count_df)))
    ax.set_xticklabels(
        [s.replace("L01912_", "") for s in count_df["sample"]],
        rotation=45, ha="right", fontsize=6
    )
    ax.set_ylabel("Cell count", fontsize=7)
    ax.set_xlabel("")
    ax.set_title("Cells per sample after QC filtering", fontsize=8, pad=6)
    ax.yaxis.grid(False)
    ax.set_axisbelow(False)

    # Group colour legend
    handles = [mpatches.Patch(color=c, label=g) for g, c in GROUP_PALETTE.items()]
    ax.legend(handles=handles, loc="upper right", ncol=2, fontsize=6,
              frameon=False, borderpad=0)

    # Annotate counts on bars
    for bar, n in zip(bars, count_df["n"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + count_df["n"].max()*0.01,
                f"{n/1000:.1f}k", ha="center", va="bottom", fontsize=5, color="#444")

    save(fig, "00_cell_counts", out_dir)
    print(f"  Saved: 00_cell_counts.pdf/.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — UMAP: group + sample (side by side)
# ══════════════════════════════════════════════════════════════════════════════

def plot_umap(adata, out_dir):
    umap = adata.obsm["X_umap"]
    fig, axes = plt.subplots(1, 2, figsize=(DC, DC*0.48))
    fig.patch.set_facecolor(UMAP_BG)

    for ax in axes:
        ax.set_facecolor(UMAP_BG)
        ax.set_aspect("equal")

    # — by group —
    for grp, col in GROUP_PALETTE.items():
        idx = adata.obs["group"].values == grp
        axes[0].scatter(umap[idx, 0], umap[idx, 1],
                        c=col, s=0.8, alpha=0.35, rasterized=True,
                        linewidths=0)
    _umap_style(axes[0], "Group")
    handles = [mpatches.Patch(color=c, label=g) for g, c in GROUP_PALETTE.items()]
    axes[0].legend(handles=handles, markerscale=4, fontsize=6,
                   loc="lower right", frameon=True,
                   facecolor=UMAP_BG, edgecolor="none", framealpha=0.8)

    # — by sample —
    samples = sorted(adata.obs["sample"].unique())
    cmap = plt.cm.get_cmap("tab10", len(samples))
    for i, sid in enumerate(samples):
        idx = adata.obs["sample"].values == sid
        axes[1].scatter(umap[idx, 0], umap[idx, 1],
                        c=[cmap(i)], s=0.8, alpha=0.35, rasterized=True,
                        linewidths=0, label=sid.replace("L01912_", ""))
    _umap_style(axes[1], "Sample")
    axes[1].legend(markerscale=4, fontsize=5.5, loc="lower right",
                   frameon=True, facecolor=UMAP_BG, edgecolor="none",
                   framealpha=0.8, ncol=2)

    fig.tight_layout(pad=1.0)
    save(fig, "01_umap_group_sample", out_dir)
    print(f"  Saved: 01_umap_group_sample.pdf/.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — UMAP: Leiden clusters
# ══════════════════════════════════════════════════════════════════════════════

def plot_umap_clusters(adata, out_dir):
    umap = adata.obsm["X_umap"]
    clusters = sorted(adata.obs["leiden"].unique(), key=int)
    palette = {c: CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
               for i, c in enumerate(clusters)}

    fig, ax = plt.subplots(figsize=(SC + 1.2, SC + 0.8))
    ax.set_facecolor(UMAP_BG)
    fig.patch.set_facecolor(UMAP_BG)

    for cl in clusters:
        idx = adata.obs["leiden"].values == cl
        ax.scatter(umap[idx, 0], umap[idx, 1],
                   c=palette[cl], s=0.8, alpha=0.5, rasterized=True,
                   linewidths=0, label=cl)

        # Cluster label at centroid
        cx, cy = umap[idx, 0].mean(), umap[idx, 1].mean()
        ax.text(cx, cy, cl, fontsize=5, ha="center", va="center",
                fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.12", fc=palette[cl],
                          ec="none", alpha=0.85))

    _umap_style(ax, f"Leiden clusters  (resolution = 0.8)")
    fig.tight_layout()
    save(fig, "02_umap_clusters", out_dir)
    print(f"  Saved: 02_umap_clusters.pdf/.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Cluster × Marker heatmap (no white/gray cells)
# ══════════════════════════════════════════════════════════════════════════════

def plot_cluster_heatmap(adata, out_dir, marker_order=None):
    X_raw = np.arcsinh(adata.layers["raw"] / 5)
    cluster_mean = (
        pd.DataFrame(X_raw, columns=adata.var_names, index=adata.obs.index)
        .assign(cluster=adata.obs["leiden"].values)
        .groupby("cluster").mean()
    )
    cluster_mean.to_csv(f"{out_dir}/03_cluster_marker_means.csv")

    # Z-score per marker — clip to [-2.5, 2.5] to avoid white from outliers
    cluster_z = cluster_mean.apply(
        lambda col: (col - col.mean()) / (col.std() + 1e-9)
    ).clip(-2.5, 2.5)

    # Marker row order
    if marker_order:
        ordered = [m for m in marker_order if m in cluster_z.columns]
        ordered += [m for m in cluster_z.columns if m not in ordered]
    else:
        ordered = list(cluster_z.columns)

    plot_data = cluster_z[ordered].T   # markers = rows, clusters = cols
    n_markers  = len(ordered)
    n_clusters = cluster_z.shape[0]

    fig_h = max(5.5, n_markers * 0.22)
    fig_w = max(4.5, n_clusters * 0.38)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.set_facecolor("#1A3A6B")   # match darkest colormap edge, no white bleed
    im = ax.imshow(
        plot_data.values,
        cmap=HEATMAP_DIVERG,
        aspect="auto",
        vmin=-2.5, vmax=2.5,
        interpolation="none",     # no antialiasing → no white bleed between cells
        rasterized=True,           # force raster in PDF → zero cell boundary artifacts
    )

    # Axes ticks
    ax.set_xticks(range(n_clusters))
    ax.set_xticklabels(plot_data.columns, fontsize=6.5, fontweight="bold")
    ax.set_yticks(range(n_markers))
    ax.set_yticklabels(plot_data.index, fontsize=6)
    ax.set_xlabel("Cluster", fontsize=7, labelpad=4)
    ax.set_ylabel("Marker", fontsize=7, labelpad=4)
    ax.set_title("Cluster × Marker mean expression\n(z-scored per marker)",
                 fontsize=8, pad=8)

    ax.tick_params(which="both", bottom=False, left=False)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.02, shrink=0.6)
    cbar.set_label("Z-score", fontsize=6.5)
    cbar.ax.tick_params(labelsize=5.5)
    cbar.outline.set_linewidth(0.4)

    # Remove all spines on heatmap
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    save(fig, "03_cluster_marker_heatmap", out_dir)
    print(f"  Saved: 03_cluster_marker_heatmap.pdf/.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Cell type UMAP + Dotplot (after annotation)
# ══════════════════════════════════════════════════════════════════════════════

def plot_celltype_umap(adata, out_dir):
    if "cell_type" not in adata.obs.columns:
        print("  [SKIP] cell_type not annotated yet"); return

    umap = adata.obsm["X_umap"]
    celltypes = sorted(adata.obs["cell_type"].unique())
    ct_palette = {ct: CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
                  for i, ct in enumerate(celltypes)}

    fig, ax = plt.subplots(figsize=(SC + 1.5, SC + 1.0))
    ax.set_facecolor(UMAP_BG); fig.patch.set_facecolor(UMAP_BG)

    for ct in celltypes:
        idx = adata.obs["cell_type"].values == ct
        ax.scatter(umap[idx, 0], umap[idx, 1], c=ct_palette[ct],
                   s=0.8, alpha=0.45, rasterized=True, linewidths=0)
        cx, cy = umap[idx, 0].mean(), umap[idx, 1].mean()
        ax.text(cx, cy, ct, fontsize=4.5, ha="center", va="center",
                fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.1", fc=ct_palette[ct],
                          ec="none", alpha=0.85))

    handles = [mpatches.Patch(color=ct_palette[ct], label=ct) for ct in celltypes]
    ax.legend(handles=handles, fontsize=5, loc="lower right", ncol=2,
              frameon=True, facecolor=UMAP_BG, edgecolor="none", framealpha=0.8)
    _umap_style(ax, "Cell type annotation")

    fig.tight_layout()
    save(fig, "04_umap_celltype", out_dir)
    print(f"  Saved: 04_umap_celltype.pdf/.png")


def plot_dotplot(adata, canonical_markers, out_dir):
    if "cell_type" not in adata.obs.columns:
        print("  [SKIP] cell_type not annotated yet"); return

    markers = [m for m in canonical_markers if m in adata.var_names]
    celltypes = sorted(adata.obs["cell_type"].unique())

    X_raw = np.arcsinh(adata.layers["raw"] / 5)
    df = pd.DataFrame(X_raw, columns=adata.var_names, index=adata.obs.index)
    df["cell_type"] = adata.obs["cell_type"].values

    # Compute mean expression and % expressing (>0.5 arcsinh)
    mean_expr = df.groupby("cell_type")[markers].mean()
    pct_expr  = df.groupby("cell_type")[markers].apply(
        lambda g: (g > 0.5).mean()
    )

    # Scale mean expression 0-1 per marker
    mean_scaled = mean_expr.apply(lambda col: (col - col.min()) / (col.max() - col.min() + 1e-9))

    n_ct  = len(celltypes)
    n_m   = len(markers)
    fig_w = max(DC, n_m * 0.28)
    fig_h = max(2.5, n_ct * 0.35)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_facecolor("white")

    max_dot = 180  # max dot area in points²
    for i, ct in enumerate(celltypes):
        for j, marker in enumerate(markers):
            expr  = mean_scaled.loc[ct, marker]
            pct   = pct_expr.loc[ct, marker]
            color = HEATMAP_SEQ(expr)
            size  = pct * max_dot
            ax.scatter(j, i, s=size, c=[color], linewidths=0.3,
                       edgecolors="#333", zorder=3)

    ax.set_xticks(range(n_m))
    ax.set_xticklabels(markers, rotation=45, ha="right", fontsize=5.5)
    ax.set_yticks(range(n_ct))
    ax.set_yticklabels(celltypes, fontsize=6)
    ax.set_xlim(-0.6, n_m - 0.4)
    ax.set_ylim(-0.6, n_ct - 0.4)
    ax.set_title("Marker expression by cell type", fontsize=8, pad=8)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Colorbar (mean expression)
    sm = plt.cm.ScalarMappable(cmap=HEATMAP_SEQ,
                               norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.015, pad=0.01, shrink=0.5)
    cbar.set_label("Scaled mean expression", fontsize=5.5)
    cbar.ax.tick_params(labelsize=5)

    # Dot size legend
    for pct_val in [0.25, 0.50, 0.75, 1.0]:
        ax.scatter([], [], s=pct_val * max_dot, c="#888", label=f"{int(pct_val*100)}%",
                   linewidths=0.3, edgecolors="#333")
    ax.legend(title="% expressing", title_fontsize=5.5, fontsize=5,
              loc="upper right", bbox_to_anchor=(1.18, 1), frameon=False)

    fig.tight_layout()
    save(fig, "04b_dotplot_celltype", out_dir)
    print(f"  Saved: 04b_dotplot_celltype.pdf/.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Differential abundance  (stacked bar + boxplots)
# ══════════════════════════════════════════════════════════════════════════════

def plot_differential_abundance(adata, out_dir):
    prop_df = (
        adata.obs.groupby(["sample", "group", "cell_type"])
        .size().reset_index(name="count")
    )
    prop_df["proportion"] = (
        prop_df["count"] /
        prop_df.groupby("sample")["count"].transform("sum")
    )
    prop_df.to_csv(f"{out_dir}/05_cell_proportions.csv", index=False)

    cell_types = sorted(prop_df["cell_type"].unique())
    order = list(GROUP_PALETTE.keys())

    ncols = min(4, len(cell_types))
    nrows = int(np.ceil(len(cell_types) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 1.7, nrows * 1.8),
                              squeeze=False)

    for idx, ct in enumerate(cell_types):
        ax = axes[idx // ncols][idx % ncols]
        sub = prop_df[prop_df["cell_type"] == ct]
        palette = [GROUP_PALETTE.get(g, "#888") for g in order]

        bp = ax.boxplot(
            [sub.loc[sub["group"] == g, "proportion"].values for g in order],
            positions=range(len(order)), widths=0.45,
            patch_artist=True,
            medianprops={"color": "white", "linewidth": 1.2},
            whiskerprops={"linewidth": 0.6},
            capprops={"linewidth": 0.6},
            flierprops={"marker": "o", "markersize": 2, "alpha": 0.5,
                        "markeredgewidth": 0.3},
        )
        for patch, col in zip(bp["boxes"], palette):
            patch.set_facecolor(col); patch.set_alpha(0.85)
            patch.set_linewidth(0.5)

        # Jitter points
        for k, g in enumerate(order):
            vals = sub.loc[sub["group"] == g, "proportion"].values
            jitter = np.random.uniform(-0.15, 0.15, len(vals))
            ax.scatter(k + jitter, vals, c=palette[k], s=8, alpha=0.9,
                       zorder=5, linewidths=0.3, edgecolors="white")

        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=35, ha="right", fontsize=5)
        ax.set_title(ct, fontsize=6, pad=3, fontweight="bold")
        ax.set_ylabel("Proportion", fontsize=5.5)
        ax.yaxis.grid(False)

    for idx in range(len(cell_types), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Cell type proportion by group", fontsize=8, y=1.01,
                 fontweight="bold")
    fig.tight_layout(h_pad=1.5, w_pad=1.0)
    save(fig, "05_differential_abundance", out_dir)
    print(f"  Saved: 05_differential_abundance.pdf/.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Differential marker expression heatmap + volcano-style ranking
# ══════════════════════════════════════════════════════════════════════════════

def plot_differential_markers(adata, out_dir):
    X_df = pd.DataFrame(
        np.arcsinh(adata.layers["raw"] / 5),
        columns=adata.var_names, index=adata.obs.index
    )
    X_df["group"] = adata.obs["group"].values
    groups = list(GROUP_PALETTE.keys())

    results = []
    for marker in adata.var_names:
        vals = [X_df.loc[X_df["group"] == g, marker].values for g in groups]
        vals_valid = [v for v in vals if len(v) > 0]
        if len(vals_valid) < 2: continue
        stat, pval = stats.kruskal(*vals_valid)
        means = {f"mean_{g}": np.mean(v) for g, v in zip(groups, vals)}
        results.append({"marker": marker, "KW_stat": stat, "pval": pval, **means})

    res_df = pd.DataFrame(results)
    _, res_df["padj"], _, _ = multipletests(res_df["pval"], method="fdr_bh")
    res_df = res_df.sort_values("padj")
    res_df.to_csv(f"{out_dir}/06_differential_markers.csv", index=False)

    sig = res_df[res_df["padj"] < 0.05]["marker"].tolist()
    if not sig:
        sig = res_df.head(20)["marker"].tolist()
        print("  [INFO] No FDR<0.05; showing top 20")

    # — Heatmap: significant markers × group means —
    mean_grp = X_df.groupby("group")[sig].mean().T.reindex(columns=groups)
    mean_z   = mean_grp.apply(lambda col: (col - col.mean()) / (col.std() + 1e-9), axis=1)
    mean_z   = mean_z.clip(-2.5, 2.5)

    n_sig  = len(sig)
    n_grp  = len(groups)
    fig_h  = max(3.5, n_sig * 0.28)
    fig, ax = plt.subplots(figsize=(n_grp * 0.9 + 1.2, fig_h))

    ax.set_facecolor("#1A3A6B")
    im = ax.imshow(mean_z.values, cmap=HEATMAP_DIVERG, aspect="auto",
                   vmin=-2.5, vmax=2.5, interpolation="none", rasterized=True)
    ax.set_xticks(range(n_grp))
    ax.set_xticklabels(groups, fontsize=7, fontweight="bold")
    ax.set_yticks(range(n_sig))
    ax.set_yticklabels(sig, fontsize=6)

    ax.tick_params(which="both", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Group color bar on top
    for j, g in enumerate(groups):
        ax.add_patch(mpatches.FancyBboxPatch(
            (j - 0.48, n_sig - 0.48), 0.96, 0.96 + 0.05,
            boxstyle="square,pad=0", fc=GROUP_PALETTE[g], ec="none",
            transform=ax.transData, clip_on=False, alpha=0.0,
        ))

    cbar = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.02, shrink=0.6)
    cbar.set_label("Z-score (mean arcsinh)", fontsize=6)
    cbar.ax.tick_params(labelsize=5.5)
    cbar.outline.set_linewidth(0.4)

    ax.set_title(f"Differential markers between groups (FDR < 0.05, n={n_sig})",
                 fontsize=8, pad=8)
    fig.tight_layout()
    save(fig, "06_differential_marker_heatmap", out_dir)

    # — Violin plots: top 9 markers —
    top9 = res_df.head(min(9, len(res_df)))["marker"].tolist()
    ncols = 3; nrows = int(np.ceil(len(top9) / 3))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 2.0, nrows * 2.0),
                              squeeze=False)

    for i, marker in enumerate(top9):
        ax = axes[i // 3][i % 3]
        data_list = [X_df.loc[X_df["group"] == g, marker].values for g in groups]
        colors = [GROUP_PALETTE.get(g, "#888") for g in groups]

        vp = ax.violinplot(data_list, positions=range(len(groups)),
                           showmedians=True, showextrema=False,
                           widths=0.7)
        for body, col in zip(vp["bodies"], colors):
            body.set_facecolor(col); body.set_alpha(0.7)
            body.set_edgecolor("none")
        vp["cmedians"].set_color("white"); vp["cmedians"].set_linewidth(1.2)

        # Jitter points
        for k, (g, col) in enumerate(zip(groups, colors)):
            vals = data_list[k]
            if len(vals) > 500:   # subsample for speed
                vals = np.random.choice(vals, 500, replace=False)
            jitter = np.random.uniform(-0.08, 0.08, len(vals))
            ax.scatter(k + jitter, vals, c=col, s=1.5, alpha=0.25,
                       zorder=3, linewidths=0)

        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=35, ha="right", fontsize=5)
        padj_v = res_df.loc[res_df["marker"] == marker, "padj"].values[0]
        ax.set_title(f"{marker}  {fmt_p(padj_v, prefix='FDR')}", fontsize=6.5, pad=3,
                     fontweight="bold")
        ax.set_ylabel("arcsinh expression", fontsize=5.5)
        ax.yaxis.grid(False)

    for i in range(len(top9), nrows * ncols):
        axes[i // 3][i % 3].set_visible(False)

    fig.suptitle("Top differential markers across groups", fontsize=8,
                 fontweight="bold", y=1.01)
    fig.tight_layout(h_pad=1.5, w_pad=1.0)
    save(fig, "07_top_marker_violins", out_dir)
    print(f"  Saved: 06_differential_marker_heatmap.pdf/.png")
    print(f"  Saved: 07_top_marker_violins.pdf/.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 8 — Within-cell-type differential markers (composition-unconfounded)
# ══════════════════════════════════════════════════════════════════════════════

def plot_within_celltype_markers(adata, out_dir, min_cells=20):
    """
    For each cell type, test marker differences between groups using
    Kruskal-Wallis. This avoids the composition confound where whole-sample
    analysis reflects cell-type proportion changes rather than per-cell changes.
    Only tests cell types with >= min_cells per group.
    Filters out off-lineage markers to avoid biological artifacts.
    """
    if "cell_type" not in adata.obs.columns:
        print("  [SKIP] cell_type not annotated"); return

    # ── Lineage-relevant markers per cell type ───────────────────────────────
    # Only test markers that are biologically meaningful within each population.
    # Off-lineage markers (e.g. CD11b in T cells) produce spurious hits.
    _T_COMMON = ["CD3","CD4","CD8a","CD45RA","CD45RO","CD45RB","CCR7","CD95",
                 "CD25","CD127","CD69","CD27","CXCR5","ICOS","CCR4","CCR6",
                 "CXCR3","CD73","PD-1","TIGIT","Tim-3","CD39","TCF1",
                 "GranzymeB","Ki-67","CD38","HLA-DR"]
    _B_COMMON = ["CD19","CD20","CD22","IgD","IgM","CD38","CD27","CD24",
                 "HLA-DR","Ki-67","CD45RA","CD45RB","CD69","CD95","CD73"]
    _MYELOID  = ["CD14","CD16","CD11c","CD11b","HLA-DR","CD123","CD66b",
                 "CD38","CD39","CD69","Ki-67","Tim-3","CD45","CD73","CD27"]
    _NK       = ["CD56","CD16","CD11b","GranzymeB","Ki-67","CD38","CD69",
                 "CD27","Tim-3","TIGIT","PD-1","CD45RA","CD45RO","CD95"]
    RELEVANT_MARKERS = {
        "CD4 Naive T":              _T_COMMON,
        "CD4 Central Memory T":     _T_COMMON,
        "CD4 Effector Memory T":    _T_COMMON,
        "CD4 Memory T (Tfh-like)":  _T_COMMON,
        "CD4 Th1-like":             _T_COMMON,
        "CD8 Effector T":           _T_COMMON,
        "CD8 Naive T":              _T_COMMON,
        "CD8 T (effector)":         _T_COMMON,
        "Treg":                     _T_COMMON,
        "γδ T cell":                _T_COMMON + ["TCRgd"],
        "NK cell":                  _NK,
        "Naive B":                  _B_COMMON,
        "Classical Monocyte":       _MYELOID,
        "Non-classical Monocyte":   _MYELOID,
        "Inflammatory Monocyte":    _MYELOID,
        "Monocyte":                 _MYELOID,
        "mDC":                      _MYELOID,
        "pDC":                      _MYELOID,
    }

    X_df = pd.DataFrame(
        np.arcsinh(adata.layers["raw"] / 5),
        columns=adata.var_names, index=adata.obs.index
    )
    X_df["group"]     = adata.obs["group"].values
    X_df["cell_type"] = adata.obs["cell_type"].values

    groups     = list(GROUP_PALETTE.keys())
    celltypes  = sorted(adata.obs["cell_type"].unique())
    all_results = []

    for ct in celltypes:
        sub = X_df[X_df["cell_type"] == ct]
        # Use lineage-relevant markers if defined, else all
        allowed = RELEVANT_MARKERS.get(ct, list(adata.var_names))
        test_markers = [m for m in adata.var_names if m in allowed]
        for marker in test_markers:
            vals = [sub.loc[sub["group"] == g, marker].values for g in groups]
            # Only test if all groups have enough cells
            if any(len(v) < min_cells for v in vals):
                continue
            stat, pval = stats.kruskal(*vals)
            means = {f"mean_{g}": np.mean(v) for g, v in zip(groups, vals)}
            all_results.append({
                "cell_type": ct, "marker": marker,
                "KW_stat": stat, "pval": pval, **means
            })

    if not all_results:
        print("  [WARN] No cell type × group combinations with enough cells"); return

    res = pd.DataFrame(all_results)
    _, res["padj"], _, _ = multipletests(res["pval"], method="fdr_bh")
    res = res.sort_values(["cell_type", "padj"])
    res.to_csv(f"{out_dir}/08_within_celltype_markers.csv", index=False)

    # ── Heatmap: top significant marker per cell type ─────────────────────────
    sig = res[res["padj"] < 0.05]
    if sig.empty:
        print("  [INFO] No significant within-cell-type markers at FDR<0.05")
        sig = res.groupby("cell_type").head(3)

    # Build matrix: cell_type × marker, value = -log10(padj), masked if ns
    pivot = sig.pivot_table(
        index="cell_type", columns="marker", values="padj",
        aggfunc="min"
    )
    pivot_log = -np.log10(pivot.clip(lower=1e-300))

    # Replace NaN (not enough cells tested) with -1 so we can colour them distinctly
    pivot_plot = pivot_log.fillna(-1)

    # Custom colormap: light gray for -1 (not tested), then white→orange→red
    from matplotlib.colors import ListedColormap, BoundaryNorm
    base_cmap = plt.cm.get_cmap("YlOrRd", 256)
    newcolors  = base_cmap(np.linspace(0, 1, 256))
    gray_color = np.array([0.88, 0.88, 0.88, 1.0])   # light gray for "not tested"
    vmax_val   = max(10, np.nanpercentile(pivot_log.values[~np.isnan(pivot_log.values)], 98))

    fig_w = max(DC, pivot_plot.shape[1] * 0.32)
    fig_h = max(3.0, pivot_plot.shape[0] * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_facecolor("white")

    # Draw gray background for NaN cells first
    nan_mask = pivot_log.isna().values
    nan_overlay = np.where(nan_mask, 1.0, np.nan)
    ax.imshow(nan_overlay, cmap=ListedColormap(["#CCCCCC"]),
              aspect="auto", vmin=0, vmax=1,
              interpolation="none", rasterized=True, alpha=1.0)

    # Draw actual values on top (NaN cells will show through as gray)
    im = ax.imshow(
        pivot_log.values,
        cmap="YlOrRd", aspect="auto",
        vmin=0, vmax=vmax_val,
        interpolation="none", rasterized=True
    )
    ax.set_xticks(range(pivot_log.shape[1]))
    ax.set_xticklabels(pivot_log.columns, rotation=45, ha="right", fontsize=5.5)
    ax.set_yticks(range(pivot_log.shape[0]))
    ax.set_yticklabels(pivot_log.index, fontsize=6)
    ax.set_title("Within-cell-type differential markers\n($-\\log_{10}$ FDR, Kruskal-Wallis)",
                 fontsize=8, pad=8)
    ax.tick_params(which="both", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.02, shrink=0.5)
    cbar.set_label("$-\\log_{10}$(FDR)", fontsize=6)
    cbar.ax.tick_params(labelsize=5); cbar.outline.set_linewidth(0)
    # Add "not tested" label
    ax.text(1.01, -0.04, "□ not tested\n   (n < 20/group)",
            transform=ax.transAxes, fontsize=5, color="#888", va="top")

    fig.tight_layout()
    save(fig, "08_within_celltype_markers", out_dir)

    # ── Violin: top hit per major cell type ───────────────────────────────────
    major_types = [ct for ct in [
        "CD4 Naive T", "CD4 Central Memory T", "CD4 Effector Memory T",
        "CD4 Th1-like", "CD8 Effector T", "CD8 Naive T",
        "NK cell", "Naive B", "Classical Monocyte",
        # Fallbacks for other annotation schemes
        "CD4 Memory T (Tfh-like)", "CD8 T (effector)", "Monocyte",
    ] if ct in sig["cell_type"].values]

    if major_types:
        ncols = min(3, len(major_types))
        nrows = int(np.ceil(len(major_types) / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(ncols * 2.2, nrows * 2.0),
                                  squeeze=False)
        for i, ct in enumerate(major_types):
            ax = axes[i // ncols][i % ncols]
            # top marker for this cell type
            top_hit = sig[sig["cell_type"] == ct].iloc[0]
            marker  = top_hit["marker"]
            sub     = X_df[X_df["cell_type"] == ct]

            data_list = [sub.loc[sub["group"] == g, marker].values for g in groups]
            colors    = [GROUP_PALETTE.get(g, "#888") for g in groups]
            vp = ax.violinplot(data_list, positions=range(len(groups)),
                               showmedians=True, showextrema=False, widths=0.65)
            for body, col in zip(vp["bodies"], colors):
                body.set_facecolor(col); body.set_alpha(0.75); body.set_edgecolor("none")
            vp["cmedians"].set_color("white"); vp["cmedians"].set_linewidth(1.2)
            for k, (g, col) in enumerate(zip(groups, colors)):
                v = data_list[k]
                if len(v) > 300: v = np.random.choice(v, 300, replace=False)
                ax.scatter(k + np.random.uniform(-0.08, 0.08, len(v)), v,
                           c=col, s=1.5, alpha=0.25, zorder=3, linewidths=0)
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels(groups, rotation=35, ha="right", fontsize=5)
            ax.set_title(f"{ct}\n{marker}  {fmt_p(top_hit['padj'], prefix='FDR')}",
                         fontsize=6, pad=3, fontweight="bold")
            ax.set_ylabel("arcsinh", fontsize=5.5)
            ax.yaxis.grid(False)
        for i in range(len(major_types), nrows * ncols):
            axes[i // ncols][i % ncols].set_visible(False)
        fig.suptitle("Top within-cell-type differential marker per population",
                     fontsize=8, fontweight="bold", y=1.01)
        fig.tight_layout(h_pad=1.5, w_pad=1.0)
        save(fig, "08b_within_celltype_violins", out_dir)

    print(f"  Saved: 08_within_celltype_markers.pdf/.png")
    print(f"  Saved: 08b_within_celltype_violins.pdf/.png")
    print(f"  Saved: 08_within_celltype_markers.csv  ({len(res)} tests)")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 9 — Stacked bar chart: cell type proportions by sample/group
# ══════════════════════════════════════════════════════════════════════════════

def plot_stacked_bar(adata, out_dir):
    """Stacked bar chart showing cell type composition per sample, grouped."""
    prop = (
        adata.obs.groupby(["sample", "group", "cell_type"], observed=True)
        .size().reset_index(name="count")
    )
    prop["proportion"] = (
        prop["count"] / prop.groupby("sample")["count"].transform("sum")
    )
    pivot = prop.pivot_table(
        index="sample", columns="cell_type", values="proportion", fill_value=0
    )

    # Sort samples by group
    sample_group = (
        adata.obs[["sample", "group"]].drop_duplicates()
        .sort_values("group").set_index("sample")
    )
    pivot = pivot.reindex(sample_group.index)
    celltypes = sorted(pivot.columns)

    ct_palette = {ct: CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
                  for i, ct in enumerate(celltypes)}

    fig, ax = plt.subplots(figsize=(DC, 3.0))
    bottom = np.zeros(len(pivot))
    for ct in celltypes:
        vals = pivot[ct].values
        ax.bar(range(len(pivot)), vals, bottom=bottom, width=0.75,
               color=ct_palette[ct], edgecolor="white", linewidth=0.3,
               label=ct)
        bottom += vals

    ax.set_xticks(range(len(pivot)))
    labels = [s.replace("L01912_", "") for s in pivot.index]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Proportion", fontsize=7)
    ax.set_title("Cell type composition per sample", fontsize=8, pad=6)
    ax.set_ylim(0, 1.0)

    # Group brackets on top
    grp_order = list(GROUP_PALETTE.keys())
    x_pos = 0
    for grp in grp_order:
        n = (sample_group["group"] == grp).sum()
        if n == 0:
            continue
        mid = x_pos + n / 2 - 0.5
        ax.text(mid, 1.03, grp, ha="center", va="bottom", fontsize=6,
                fontweight="bold", color=GROUP_PALETTE[grp],
                transform=ax.get_xaxis_transform())
        if n > 1:
            ax.plot([x_pos - 0.3, x_pos + n - 1 + 0.3], [1.01, 1.01],
                    color=GROUP_PALETTE[grp], lw=1.2, clip_on=False,
                    transform=ax.get_xaxis_transform())
        x_pos += n

    ax.legend(fontsize=4.5, ncol=4, loc="upper left",
              bbox_to_anchor=(0, -0.22), frameon=False,
              columnspacing=0.8, handletextpad=0.3)

    fig.tight_layout()
    save(fig, "09_stacked_bar_composition", out_dir)
    print(f"  Saved: 09_stacked_bar_composition.pdf/.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 10 — Sample-level clustering heatmap (cell type proportions)
# ══════════════════════════════════════════════════════════════════════════════

def plot_sample_clustering(adata, out_dir):
    """Hierarchical clustering of samples by cell type proportions."""
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import pdist

    prop = (
        adata.obs.groupby(["sample", "group", "cell_type"], observed=True)
        .size().reset_index(name="count")
    )
    prop["proportion"] = (
        prop["count"] / prop.groupby("sample")["count"].transform("sum")
    )
    pivot = prop.pivot_table(
        index="sample", columns="cell_type", values="proportion", fill_value=0
    )

    sample_group = (
        adata.obs[["sample", "group"]].drop_duplicates().set_index("sample")
    )

    # Z-score per cell type for visualization
    pivot_z = pivot.apply(lambda col: (col - col.mean()) / (col.std() + 1e-9))

    # Hierarchical clustering
    dist = pdist(pivot_z.values, metric="euclidean")
    link = linkage(dist, method="ward")

    fig = plt.figure(figsize=(DC, 4.5))
    gs = gridspec.GridSpec(2, 2, height_ratios=[0.15, 1],
                           width_ratios=[0.12, 1],
                           hspace=0.02, wspace=0.02)

    # Dendrogram (top)
    ax_dendro = fig.add_subplot(gs[0, 1])
    dn = dendrogram(link, labels=pivot_z.index.tolist(), ax=ax_dendro,
                    no_labels=True, color_threshold=0, above_threshold_color="#888")
    ax_dendro.set_xticks([])
    ax_dendro.spines["bottom"].set_visible(False)
    ax_dendro.spines["left"].set_visible(False)
    ax_dendro.set_yticks([])
    for spine in ax_dendro.spines.values():
        spine.set_visible(False)

    # Reorder by dendrogram
    order = dn["leaves"]
    pivot_ordered = pivot_z.iloc[order]

    # Group color sidebar (left)
    ax_grp = fig.add_subplot(gs[1, 0])
    for i, sample in enumerate(pivot_ordered.index):
        grp = sample_group.loc[sample, "group"]
        ax_grp.barh(i, 1, color=GROUP_PALETTE.get(grp, "#888"),
                    edgecolor="none", height=1.0)
    ax_grp.set_ylim(-0.5, len(pivot_ordered) - 0.5)
    ax_grp.invert_yaxis()
    ax_grp.set_xticks([])
    ax_grp.set_yticks([])
    for spine in ax_grp.spines.values():
        spine.set_visible(False)

    # Heatmap
    ax_heat = fig.add_subplot(gs[1, 1])
    im = ax_heat.imshow(pivot_ordered.values, cmap=HEATMAP_DIVERG,
                        aspect="auto", vmin=-2, vmax=2,
                        interpolation="none", rasterized=True)
    ax_heat.set_xticks(range(len(pivot_ordered.columns)))
    ax_heat.set_xticklabels(pivot_ordered.columns, rotation=45, ha="right",
                            fontsize=5.5)
    ax_heat.set_yticks(range(len(pivot_ordered)))
    ax_heat.set_yticklabels(
        [s.replace("L01912_", "") for s in pivot_ordered.index], fontsize=6
    )
    ax_heat.tick_params(which="both", bottom=False, left=False)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.02, shrink=0.6)
    cbar.set_label("Z-score (proportion)", fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    cbar.outline.set_linewidth(0)

    # Group legend
    import matplotlib.patches as mp
    handles = [mp.Patch(color=c, label=g) for g, c in GROUP_PALETTE.items()]
    ax_heat.legend(handles=handles, fontsize=5, loc="upper right",
                   bbox_to_anchor=(1.25, 1.0), frameon=False)

    fig.suptitle("Sample clustering by immune composition",
                 fontsize=8, fontweight="bold", y=0.98)
    save(fig, "10_sample_clustering_heatmap", out_dir)
    print(f"  Saved: 10_sample_clustering_heatmap.pdf/.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 11 — UMAP marker overlay panel
# ══════════════════════════════════════════════════════════════════════════════

def plot_umap_markers(adata, out_dir):
    """Panel of UMAP plots colored by key functional markers."""
    markers = [m for m in [
        "Ki-67", "PD-1", "TIGIT", "Tim-3", "CD39", "GranzymeB",
        "CD38", "CD69", "ICOS", "CXCR3", "CCR4", "TCF1",
    ] if m in adata.var_names]

    X_raw = np.arcsinh(adata.layers["raw"] / 5)
    umap = adata.obsm["X_umap"]

    ncols = 4
    nrows = int(np.ceil(len(markers) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(DC + 1, nrows * 1.7),
                              squeeze=False)

    for i, marker in enumerate(markers):
        ax = axes[i // ncols][i % ncols]
        ax.set_facecolor(UMAP_BG)
        midx = list(adata.var_names).index(marker)
        vals = X_raw[:, midx]

        # Random order to avoid overplotting bias
        rng = np.random.RandomState(42)
        shuf = rng.permutation(len(vals))

        sc_plot = ax.scatter(
            umap[shuf, 0], umap[shuf, 1],
            c=vals[shuf], cmap=HEATMAP_SEQ, s=0.3, alpha=0.5,
            rasterized=True, linewidths=0,
            vmin=np.percentile(vals, 2), vmax=np.percentile(vals, 98)
        )
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(marker, fontsize=7, fontweight="bold", pad=3)

        cbar = fig.colorbar(sc_plot, ax=ax, fraction=0.04, pad=0.02, shrink=0.7)
        cbar.ax.tick_params(labelsize=4)
        cbar.outline.set_linewidth(0)

    for i in range(len(markers), nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    fig.suptitle("Functional marker expression on UMAP",
                 fontsize=8, fontweight="bold", y=1.01)
    fig.tight_layout(h_pad=0.8, w_pad=0.5)
    save(fig, "11_umap_marker_overlay", out_dir)
    print(f"  Saved: 11_umap_marker_overlay.pdf/.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 12 — Pairwise differential abundance (forest plot)
# ══════════════════════════════════════════════════════════════════════════════

def plot_pairwise_abundance(adata, out_dir):
    """Forest plot: pairwise group comparisons of cell type proportions."""
    from itertools import combinations

    prop = (
        adata.obs.groupby(["sample", "group", "cell_type"], observed=True)
        .size().reset_index(name="count")
    )
    prop["proportion"] = (
        prop["count"] / prop.groupby("sample")["count"].transform("sum")
    )

    groups = list(GROUP_PALETTE.keys())
    pairs = list(combinations(groups, 2))
    celltypes = sorted(adata.obs["cell_type"].unique())
    # Exclude tiny/artifact clusters
    celltypes = [ct for ct in celltypes
                 if ct not in ("T-Myeloid doublets", "CD16+ Granulocyte")]

    results = []
    for ct in celltypes:
        for g1, g2 in pairs:
            v1 = prop.loc[(prop["cell_type"] == ct) & (prop["group"] == g1),
                          "proportion"].values
            v2 = prop.loc[(prop["cell_type"] == ct) & (prop["group"] == g2),
                          "proportion"].values
            if len(v1) < 2 or len(v2) < 2:
                continue
            diff = np.mean(v1) - np.mean(v2)
            # Pooled SE for CI
            se = np.sqrt(np.var(v1, ddof=1)/len(v1) + np.var(v2, ddof=1)/len(v2))
            ci_lo = diff - 1.96 * se
            ci_hi = diff + 1.96 * se
            _, pval = stats.mannwhitneyu(v1, v2, alternative="two-sided")
            results.append({
                "cell_type": ct, "comparison": f"{g1} vs {g2}",
                "diff": diff, "ci_lo": ci_lo, "ci_hi": ci_hi,
                "pval": pval, "g1": g1, "g2": g2
            })

    res = pd.DataFrame(results)
    if res.empty:
        print("  [WARN] No pairwise comparisons possible"); return
    _, res["padj"], _, _ = multipletests(res["pval"], method="fdr_bh")
    res.to_csv(f"{out_dir}/12_pairwise_abundance.csv", index=False)

    # Forest plot: one panel per comparison
    ncols = min(3, len(pairs))
    nrows = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 2.8, max(3.5, len(celltypes) * 0.3)),
                              squeeze=False)

    for pidx, (g1, g2) in enumerate(pairs):
        ax = axes[pidx // ncols][pidx % ncols]
        sub = res[res["comparison"] == f"{g1} vs {g2}"].sort_values("cell_type")

        for i, (_, row) in enumerate(sub.iterrows()):
            is_sig = row["padj"] < 0.05
            color = "#C0392B" if is_sig else "#888"
            marker = "D" if is_sig else "o"
            ax.errorbar(row["diff"], i, xerr=[[row["diff"] - row["ci_lo"]],
                        [row["ci_hi"] - row["diff"]]],
                        fmt=marker, color=color, markersize=3,
                        elinewidth=0.6, capsize=1.5, capthick=0.5)
            # Annotate p-value for significant hits
            if is_sig:
                ax.text(row["ci_hi"] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01,
                        i, f"{row['padj']:.1e}", fontsize=4, color="#C0392B",
                        va="center", ha="left")

        ax.axvline(0, color="#aaa", linestyle="--", linewidth=0.5, zorder=0)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub["cell_type"].values, fontsize=5)
        ax.set_xlabel("$\\Delta$ proportion", fontsize=6)
        ax.set_title(f"{g1} vs {g2}", fontsize=7, fontweight="bold", pad=4,
                     color=GROUP_PALETTE[g1])
        ax.invert_yaxis()
        ax.yaxis.grid(False)
        # Expand xlim slightly to fit p-value text
        xlim = ax.get_xlim()
        ax.set_xlim(xlim[0], xlim[1] + (xlim[1] - xlim[0]) * 0.25)

    for pidx in range(len(pairs), nrows * ncols):
        axes[pidx // ncols][pidx % ncols].set_visible(False)

    fig.suptitle("Pairwise differential cell type abundance\n(95% CI, $\\diamond$ = FDR < 0.05)",
                 fontsize=8, fontweight="bold", y=1.02)
    fig.tight_layout(h_pad=1.5, w_pad=2.0)
    save(fig, "12_pairwise_abundance_forest", out_dir)
    print(f"  Saved: 12_pairwise_abundance_forest.pdf/.png")
    print(f"  Saved: 12_pairwise_abundance.csv")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 13 — Pairwise volcano plots (marker-level)
# ══════════════════════════════════════════════════════════════════════════════

def plot_pairwise_volcano(adata, out_dir):
    """Volcano plots for each pairwise group comparison (all markers, all cells)."""
    from itertools import combinations

    X_df = pd.DataFrame(
        np.arcsinh(adata.layers["raw"] / 5),
        columns=adata.var_names, index=adata.obs.index
    )
    X_df["group"] = adata.obs["group"].values

    groups = list(GROUP_PALETTE.keys())
    pairs = list(combinations(groups, 2))

    ncols = 3
    nrows = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.0, nrows * 2.8),
                              squeeze=False)

    for pidx, (g1, g2) in enumerate(pairs):
        ax = axes[pidx // ncols][pidx % ncols]
        v1_df = X_df[X_df["group"] == g1]
        v2_df = X_df[X_df["group"] == g2]

        results = []
        for marker in adata.var_names:
            v1 = v1_df[marker].values
            v2 = v2_df[marker].values
            mean_diff = np.mean(v1) - np.mean(v2)
            pooled_std = np.sqrt((np.var(v1) + np.var(v2)) / 2 + 1e-9)
            cohens_d = mean_diff / pooled_std
            _, pval = stats.mannwhitneyu(v1, v2, alternative="two-sided")
            results.append({"marker": marker, "cohens_d": cohens_d, "pval": pval})

        vdf = pd.DataFrame(results)
        _, vdf["padj"], _, _ = multipletests(vdf["pval"], method="fdr_bh")
        vdf["neg_log_padj"] = -np.log10(vdf["padj"].clip(lower=1e-300))

        # Color by significance
        sig_mask = (vdf["padj"] < 0.05) & (vdf["cohens_d"].abs() > 0.2)
        ax.scatter(vdf.loc[~sig_mask, "cohens_d"],
                   vdf.loc[~sig_mask, "neg_log_padj"],
                   c="#CCCCCC", s=8, alpha=0.6, linewidths=0, zorder=2)
        ax.scatter(vdf.loc[sig_mask, "cohens_d"],
                   vdf.loc[sig_mask, "neg_log_padj"],
                   c="#C0392B", s=12, alpha=0.8, linewidths=0.3,
                   edgecolors="white", zorder=3)

        # Label top hits with adjustText to avoid overlap
        from adjustText import adjust_text
        top = vdf[sig_mask].nlargest(5, "neg_log_padj")
        texts = []
        for _, row in top.iterrows():
            texts.append(ax.text(row["cohens_d"], row["neg_log_padj"],
                                 row["marker"], fontsize=4.5, ha="center", va="bottom"))
        if texts:
            adjust_text(texts, ax=ax,
                        arrowprops=dict(arrowstyle="-", color="#888", lw=0.3),
                        force_text=(0.8, 0.8), force_points=(0.5, 0.5))

        ax.axhline(-np.log10(0.05), color="#aaa", linestyle="--", linewidth=0.4)
        ax.axvline(0.2, color="#aaa", linestyle=":", linewidth=0.4)
        ax.axvline(-0.2, color="#aaa", linestyle=":", linewidth=0.4)
        ax.set_xlabel("Cohen's d", fontsize=6)
        ax.set_ylabel("$-\\log_{10}$(FDR)", fontsize=6)
        ax.set_title(f"{g1} vs {g2}", fontsize=7, fontweight="bold",
                     color=GROUP_PALETTE[g1])

    for pidx in range(len(pairs), nrows * ncols):
        axes[pidx // ncols][pidx % ncols].set_visible(False)

    fig.suptitle("Pairwise marker comparisons (volcano)",
                 fontsize=8, fontweight="bold", y=1.02)
    fig.tight_layout(h_pad=1.5, w_pad=1.0)
    save(fig, "13_pairwise_volcano", out_dir)
    print(f"  Saved: 13_pairwise_volcano.pdf/.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 14 — T cell exhaustion & activation scores by group
# ══════════════════════════════════════════════════════════════════════════════

def plot_exhaustion_activation(adata, out_dir):
    """Compute and plot exhaustion/activation composite scores for T cell subsets.
    Uses boxplot + jitter style (matching Fig 05) with KW p-values annotated."""
    X_raw = np.arcsinh(adata.layers["raw"] / 5)
    df = pd.DataFrame(X_raw, columns=adata.var_names, index=adata.obs.index)
    df["sample"]    = adata.obs["sample"].values
    df["group"]     = adata.obs["group"].values
    df["cell_type"] = adata.obs["cell_type"].values

    # Define scores
    exh_markers = [m for m in ["PD-1", "TIGIT", "Tim-3", "CD39"] if m in df.columns]
    act_markers = [m for m in ["CD38", "Ki-67", "CD69"] if m in df.columns]

    if not exh_markers or not act_markers:
        print("  [SKIP] Not enough markers for scores"); return

    df["exhaustion_score"] = df[exh_markers].mean(axis=1)
    df["activation_score"] = df[act_markers].mean(axis=1)

    # T cell subsets only
    t_types = [ct for ct in df["cell_type"].unique()
               if any(x in ct for x in ["CD4", "CD8", "Th1", "Treg", "γδ"])]
    t_types = sorted(t_types)
    if not t_types:
        print("  [SKIP] No T cell types found"); return

    t_df = df[df["cell_type"].isin(t_types)]
    groups = list(GROUP_PALETTE.keys())

    # Aggregate to sample level
    agg_all = {}
    for score_name in ["exhaustion_score", "activation_score"]:
        agg_all[score_name] = (
            t_df.groupby(["sample", "group", "cell_type"], observed=True)[score_name]
            .mean().reset_index()
        )

    # Compute KW p-values for annotation
    kw_pvals = {}
    score_results = []
    for score_name in ["exhaustion_score", "activation_score"]:
        for ct in t_types:
            agg = agg_all[score_name]
            sub = agg[agg["cell_type"] == ct]
            vals = [sub.loc[sub["group"] == g, score_name].values for g in groups]
            vals_valid = [v for v in vals if len(v) > 0]
            if len(vals_valid) < 2:
                continue
            stat, pval = stats.kruskal(*vals_valid)
            kw_pvals[(score_name, ct)] = pval
            means = {f"mean_{g}": np.mean(v) if len(v) > 0 else np.nan
                     for g, v in zip(groups, vals)}
            score_results.append({
                "score": score_name, "cell_type": ct,
                "KW_stat": stat, "pval": pval, **means
            })

    # FDR correction
    if score_results:
        sdf = pd.DataFrame(score_results)
        _, sdf["padj"], _, _ = multipletests(sdf["pval"], method="fdr_bh")
        sdf.to_csv(f"{out_dir}/14_functional_scores.csv", index=False)
        # Update kw_pvals with FDR-corrected values
        for _, row in sdf.iterrows():
            kw_pvals[(row["score"], row["cell_type"])] = row["padj"]

    # --- Boxplot + jitter per cell type (one row per score) ---
    ncols = len(t_types)
    fig, axes = plt.subplots(2, ncols,
                              figsize=(max(DC, ncols * 1.5), 4.5),
                              squeeze=False)

    for row_idx, (score_name, score_label) in enumerate([
        ("exhaustion_score", "Exhaustion\n(PD-1+TIGIT+Tim-3+CD39)"),
        ("activation_score", "Activation\n(CD38+Ki-67+CD69)")
    ]):
        agg = agg_all[score_name]
        for col_idx, ct in enumerate(t_types):
            ax = axes[row_idx][col_idx]
            sub = agg[agg["cell_type"] == ct]
            order = groups
            palette = [GROUP_PALETTE.get(g, "#888") for g in order]

            # Boxplot
            data_list = [sub.loc[sub["group"] == g, score_name].values for g in order]
            bp = ax.boxplot(
                data_list, positions=range(len(order)), widths=0.5,
                patch_artist=True,
                medianprops={"color": "white", "linewidth": 1.2},
                whiskerprops={"linewidth": 0.6},
                capprops={"linewidth": 0.6},
                flierprops={"marker": "o", "markersize": 2, "alpha": 0.5,
                            "markeredgewidth": 0.3},
            )
            for patch, col in zip(bp["boxes"], palette):
                patch.set_facecolor(col); patch.set_alpha(0.8)
                patch.set_linewidth(0.5)

            # Jitter points
            for k, (g, col) in enumerate(zip(order, palette)):
                vals = data_list[k]
                jitter = np.random.uniform(-0.15, 0.15, len(vals))
                ax.scatter(k + jitter, vals, c=col, s=14, alpha=0.9,
                           zorder=5, linewidths=0.3, edgecolors="white")

            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order, rotation=35, ha="right", fontsize=5)
            ax.yaxis.grid(False)

            # P-value annotation
            padj = kw_pvals.get((score_name, ct), None)
            if padj is not None:
                pstr = f"p={padj:.2e}" if padj < 0.05 else f"p={padj:.2f}"
                color = "#C0392B" if padj < 0.05 else "#888"
                ax.set_title(f"{ct}\n{pstr}", fontsize=5.5, pad=3,
                             fontweight="bold", color=color)
            else:
                ax.set_title(ct, fontsize=5.5, pad=3, fontweight="bold")

            if col_idx == 0:
                ax.set_ylabel(score_label, fontsize=6)

    fig.suptitle("T cell functional scores across autoantibody groups\n(Kruskal-Wallis, FDR-corrected)",
                 fontsize=8, fontweight="bold", y=1.03)
    fig.tight_layout(h_pad=2.0, w_pad=0.8)
    save(fig, "14_exhaustion_activation_scores", out_dir)
    print(f"  Saved: 14_exhaustion_activation_scores.pdf/.png")
    print(f"  Saved: 14_functional_scores.csv")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _umap_style(ax, title):
    """Apply Nature-style formatting to a UMAP axis."""
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=7.5, pad=5, fontweight="bold")

    # Small axis arrows indicating direction
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    x0 = xlim[0] + (xlim[1] - xlim[0]) * 0.02
    y0 = ylim[0] + (ylim[1] - ylim[0]) * 0.02
    arr_len = (xlim[1] - xlim[0]) * 0.12
    kw = dict(color="#555", lw=0.8, arrowprops=dict(arrowstyle="-|>",
              color="#555", lw=0.8, mutation_scale=5))
    ax.annotate("", xy=(x0 + arr_len, y0), xytext=(x0, y0),
                arrowprops=kw["arrowprops"])
    ax.annotate("", xy=(x0, y0 + arr_len), xytext=(x0, y0),
                arrowprops=kw["arrowprops"])
    ax.text(x0 + arr_len * 0.5, y0 - (ylim[1]-ylim[0])*0.03,
            "UMAP 1", fontsize=5, ha="center", color="#555")
    ax.text(x0 - (xlim[1]-xlim[0])*0.03, y0 + arr_len * 0.5,
            "UMAP 2", fontsize=5, ha="center", va="center",
            rotation=90, color="#555")
