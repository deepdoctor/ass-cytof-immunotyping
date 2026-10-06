"""
Main Figure 7 — T-cell differentiation trajectory and its mapping to
patient-level immunotypes.

Novel contribution: connect the *single-cell* differentiation landscape
(UMAP + diffusion pseudotime + PAGA) with the *patient-level* immunotype
partition. The 6 panels:

  (a) UMAP of T cells, coloured by lineage (CD4 Naive → CM → EM → Th1; CD8 N → Eff)
  (b) Same UMAP coloured by diffusion pseudotime (DPT) — root in CD4 Naive T
  (c) PAGA connectivity graph showing T cell lineage relationships
  (d) Pseudotime distribution by cell type (violin)
  (e) NEW — Pseudotime distribution by patient immunotype (IT1/IT2/IT3),
      pooling each immunotype's T cells. IT1 patients are differentiation-
      advanced; IT3 patients are naive-skewed.
  (f) NEW — Cell-type composition along pseudotime quintiles, stratified by
      immunotype. Stacked bars showing how IT1 vs IT3 fundamentally re-shape
      the T cell differentiation landscape.

Source: cytof_analyzed_v3.h5ad + 21_immunotype_assignments.csv.
"""
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

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

T_CELLS = [
    "CD4 Naive T", "CD4 Central Memory T", "CD4 Effector Memory T", "CD4 Th1-like",
    "CD8 Naive T", "CD8 Effector T",
]
T_PALETTE = {
    "CD4 Naive T":           "#7DC1E8",
    "CD4 Central Memory T":  "#1F77B4",
    "CD4 Effector Memory T": "#5B9BD5",
    "CD4 Th1-like":          "#3A7CA5",
    "CD8 Naive T":           "#C5B0D5",
    "CD8 Effector T":        "#7D3C98",
}
IT_COL = {"IT1": "#E67E22", "IT2": "#2980B9", "IT3": "#27AE60"}
IT_ORDER = ["IT1", "IT2", "IT3"]

plt.rcParams.update({
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":            7.0,
    "axes.titlesize":       7.5,
    "axes.labelsize":       7.0,
    "xtick.labelsize":      6.2,
    "ytick.labelsize":      6.2,
    "legend.fontsize":      6.0,
    "axes.linewidth":       0.7,
    "xtick.major.width":    0.6,
    "ytick.major.width":    0.6,
    "xtick.major.size":     2.5,
    "ytick.major.size":     2.5,
    "lines.linewidth":      0.9,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
    "savefig.dpi":          300,
    "figure.dpi":           300,
})

PANEL_LBL = dict(fontsize=10, fontweight="bold", color="#000",
                 ha="left", va="bottom", family="sans-serif")

# ─── load + subset ──────────────────────────────────────────────────────────
print("Loading h5ad …")
a = ad.read_h5ad(H5AD)
a.obs_names_make_unique()
a.obs["cell_type"] = a.obs["leiden"].astype(str).map(CLUSTER_ANNOTATION).fillna("Unannotated")

# attach immunotype per patient
assigned = pd.read_csv(f"{OUT_DIR}/21_immunotype_assignments.csv")
sample_to_it = dict(zip(assigned["sample"], assigned["immunotype"]))
a.obs["immunotype"] = a.obs["sample"].map(sample_to_it)

# subset to T cells
mask_T = a.obs["cell_type"].isin(T_CELLS)
T = a[mask_T].copy()
print(f"  T cells: {T.n_obs:,} (over {T.obs['sample'].nunique()} samples)")

# Re-run trajectory just on T cells (so the DPT is a clean lineage analysis)
sc.pp.neighbors(T, use_rep="X_pca_harmony" if "X_pca_harmony" in T.obsm
                          else "X_pca", n_neighbors=30)
sc.tl.diffmap(T, n_comps=10)
sc.tl.umap(T)  # T-cell-specific UMAP
# root in CD4 Naive T (highest CCR7)
naive_mask = T.obs["cell_type"].values == "CD4 Naive T"
if "CCR7" in T.var_names:
    j_ccr7 = list(T.var_names).index("CCR7")
    Xj = T.X[:, j_ccr7]
    Xj = Xj.toarray().ravel() if hasattr(Xj, "toarray") else np.asarray(Xj).ravel()
    cand = np.where(naive_mask)[0]
    cand_sorted = cand[np.argsort(-Xj[cand])]
    iroot_local = cand_sorted[0]
else:
    iroot_local = int(np.where(naive_mask)[0][0])
T.uns["iroot"] = iroot_local
sc.tl.dpt(T)
sc.tl.paga(T, groups="cell_type")
print("  Trajectory built.")

# Cache pseudotime + cell type + immunotype for stratified analyses below
df = pd.DataFrame({
    "cell_type":  T.obs["cell_type"].values,
    "immunotype": T.obs["immunotype"].values,
    "sample":     T.obs["sample"].values,
    "dpt":        T.obs["dpt_pseudotime"].values,
    "umap1":      T.obsm["X_umap"][:, 0],
    "umap2":      T.obsm["X_umap"][:, 1],
})
df = df[df["dpt"].notna()].copy()


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE LAYOUT (3×2)
# ═══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(7.4, 9.0))
gs = gridspec.GridSpec(
    3, 2, figure=fig,
    left=0.085, right=0.965, top=0.940, bottom=0.060,
    wspace=0.45, hspace=0.65,
)

# ── (a) T-cell UMAP, coloured by lineage ───────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
for ct in T_CELLS:
    sub = df[df["cell_type"] == ct]
    ax.scatter(sub["umap1"], sub["umap2"], s=0.7, c=T_PALETTE[ct],
               alpha=0.65, edgecolor="none", label=ct)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
ax.set_title("T-cell UMAP, by lineage", fontsize=7.5, pad=3)
ax.legend(loc="lower right", frameon=False, fontsize=4.8,
          handletextpad=0.3, borderpad=0.2, markerscale=4.5,
          labelspacing=0.30)
ax.text(-0.10, 1.05, "a", transform=ax.transAxes, **PANEL_LBL)


# ── (b) T-cell UMAP, coloured by DPT pseudotime ────────────────────────────
ax = fig.add_subplot(gs[0, 1])
sc_pt = ax.scatter(df["umap1"], df["umap2"], s=0.7,
                   c=df["dpt"].values, cmap="magma", alpha=0.75,
                   edgecolor="none", vmin=0, vmax=1)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
ax.set_title("T-cell UMAP, by DPT pseudotime", fontsize=7.5, pad=3)
cbar = fig.colorbar(sc_pt, ax=ax, fraction=0.038, pad=0.025, shrink=0.78)
cbar.set_label("DPT pseudotime", fontsize=6.0)
cbar.ax.tick_params(labelsize=5.5)
cbar.outline.set_linewidth(0.4)
ax.text(-0.10, 1.05, "b", transform=ax.transAxes, **PANEL_LBL)


# ── (c) PAGA connectivity graph ────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
# render PAGA via scanpy then capture into matplotlib
sc.pl.paga(T, ax=ax, show=False, frameon=False, plot=True,
           layout="fa", threshold=0.10, fontoutline=1.5,
           node_size_scale=2.5)
ax.set_title("PAGA connectivity\n(weight ≥ 0.10)", fontsize=7.5, pad=3)
ax.text(-0.10, 1.05, "c", transform=ax.transAxes, **PANEL_LBL)


# ── (d) Pseudotime distribution by cell type ───────────────────────────────
ax = fig.add_subplot(gs[1, 1])
ct_medians = {ct: df.loc[df["cell_type"] == ct, "dpt"].median()
              for ct in T_CELLS}
ct_sorted = sorted(T_CELLS, key=lambda c: ct_medians[c])
data_per_ct = [df.loc[df["cell_type"] == ct, "dpt"].values for ct in ct_sorted]
parts = ax.violinplot(data_per_ct, positions=range(len(ct_sorted)),
                       widths=0.78, showmedians=True, showextrema=False,
                       vert=False)
for body, ct in zip(parts["bodies"], ct_sorted):
    body.set_facecolor(T_PALETTE[ct]); body.set_alpha(0.65)
    body.set_edgecolor(T_PALETTE[ct]); body.set_linewidth(0.7)
parts["cmedians"].set_color("#222"); parts["cmedians"].set_linewidth(0.8)
ax.set_yticks(range(len(ct_sorted)))
ax.set_yticklabels(ct_sorted, fontsize=6.2)
ax.set_xlim(-0.05, 1.05)
ax.set_xlabel("DPT pseudotime", fontsize=6.5)
ax.set_title("Pseudotime by cell type\n(ordered by median)",
             fontsize=7.5, pad=3)
ax.text(-0.32, 1.05, "d", transform=ax.transAxes, **PANEL_LBL)


# ── (e) NEW: Pseudotime distribution by patient immunotype ─────────────────
ax = fig.add_subplot(gs[2, 0])
data_per_it = [df.loc[df["immunotype"] == it, "dpt"].values for it in IT_ORDER]
parts = ax.violinplot(data_per_it, positions=range(len(IT_ORDER)),
                       widths=0.72, showmedians=True, showextrema=False)
for body, it in zip(parts["bodies"], IT_ORDER):
    body.set_facecolor(IT_COL[it]); body.set_alpha(0.65)
    body.set_edgecolor(IT_COL[it]); body.set_linewidth(0.7)
parts["cmedians"].set_color("#222"); parts["cmedians"].set_linewidth(0.8)

# KW test across immunotypes
from scipy.stats import kruskal
H, p_kw = kruskal(*data_per_it)
ax.set_xticks(range(len(IT_ORDER)))
ax.set_xticklabels([f"{it}\n(n={int((df['immunotype']==it).sum()):,})"
                    for it in IT_ORDER], fontsize=6.2)
ax.set_ylabel("DPT pseudotime", fontsize=6.5)
ax.set_ylim(-0.05, 1.05)
def fmt_p(p):
    if p == 0 or p < 1e-300:
        return r"$P < 10^{-300}$"
    if p < 1e-4:
        e = int(np.floor(np.log10(p))); c = p / 10**e
        return rf"$P = {c:.1f}\times 10^{{{e}}}$"
    if p < 0.05:
        return f"P = {p:.1e}"
    return f"P = {p:.2f}"
ax.set_title("Pseudotime by immunotype\n"
             "(cell-level KW shown; pseudo-replicated — patient-level\n"
             "median DPT per immunotype, n = 3/3/4)",
             fontsize=6.8, pad=3)
ax.text(-0.20, 1.05, "e", transform=ax.transAxes, **PANEL_LBL)


# ── (f) NEW: Cell-type composition along pseudotime, by immunotype ─────────
ax = fig.add_subplot(gs[2, 1])
# bin pseudotime into quintiles
df["dpt_bin"] = pd.qcut(df["dpt"], q=5,
                         labels=["Q1\n(naive)", "Q2", "Q3", "Q4",
                                 "Q5\n(differentiated)"])
# composition per (immunotype, bin)
comp = (df.groupby(["immunotype", "dpt_bin", "cell_type"], observed=False)
          .size().reset_index(name="n"))
total = (df.groupby(["immunotype", "dpt_bin"], observed=False)
           .size().reset_index(name="tot"))
comp = comp.merge(total, on=["immunotype", "dpt_bin"])
comp["frac"] = comp["n"] / comp["tot"]

# stacked bars: x is bin, group by immunotype (3 grouped sets)
n_bins = 5
n_its  = 3
bar_w = 0.25
x_centres = np.arange(n_bins)

for ki, it in enumerate(IT_ORDER):
    sub = comp[comp["immunotype"] == it].pivot_table(
        index="dpt_bin", columns="cell_type",
        values="frac", aggfunc="first", fill_value=0,
        observed=False,
    )
    sub = sub.reindex([f"Q{i+1}" + ("\n(naive)" if i == 0
                                     else ("\n(differentiated)" if i == 4 else ""))
                       for i in range(5)])
    bottoms = np.zeros(n_bins)
    for ct in T_CELLS:
        if ct not in sub.columns:
            continue
        vals = sub[ct].values
        x_pos = x_centres + (ki - 1) * bar_w
        ax.bar(x_pos, vals, width=bar_w, bottom=bottoms,
               color=T_PALETTE[ct], edgecolor="white", lw=0.3)
        bottoms += vals
ax.set_xticks(x_centres)
ax.set_xticklabels(["Q1\n(naive)", "Q2", "Q3", "Q4",
                    "Q5\n(differentiated)"], fontsize=5.5)
ax.set_ylabel("T-cell composition fraction", fontsize=6.3)
ax.set_ylim(0, 1.02)
ax.set_title("Cell-type composition along pseudotime quintiles\n"
             "(IT1, IT2, IT3 grouped left-to-right per quintile)",
             fontsize=7.0, pad=3)
# Per-bar IT-colour labels removed: the title sub-line already explains
# the IT1/IT2/IT3 left-to-right grouping; redundant labels overlapped the title.
# legend (T cell types)
legend_handles = [Patch(facecolor=T_PALETTE[ct], label=ct) for ct in T_CELLS]
ax.legend(handles=legend_handles, loc="center left",
          bbox_to_anchor=(1.005, 0.5), fontsize=4.8,
          frameon=False, handlelength=0.8, handletextpad=0.3,
          borderpad=0.2, labelspacing=0.30)
ax.text(-0.20, 1.05, "f", transform=ax.transAxes, **PANEL_LBL)


# ─── suptitle + save ───────────────────────────────────────────────────────
# Figure title removed (journal convention).

out_pdf = f"{OUT_DIR}/MainFig7_trajectory.pdf"
out_png = f"{OUT_DIR}/MainFig7_trajectory.png"
fig.savefig(out_pdf, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
fig.savefig(out_png, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")

# Also cache the per-cell pseudotime + IT for downstream use
df[["sample","cell_type","immunotype","dpt"]].to_csv(
    f"{OUT_DIR}/29b_T_pseudotime_per_cell.csv", index=False
)
print(f"Saved: {OUT_DIR}/29b_T_pseudotime_per_cell.csv")
