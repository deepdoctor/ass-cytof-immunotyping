"""
cytof_advanced_analysis.py
==========================
High-end CyTOF analyses integrating patient clinical metadata for a
JIF>8 submission. Generates figures 15-22:

  15 – Clinical-immune integrated heatmap (patient × celltype) with
        clinical annotation tracks
  16 – Patient-level PCA / MDS on composition and global marker MFI
  17 – Immune–clinical correlation heatmap (Spearman) with FDR dots
  18 – Cell-type co-variation correlation matrix
  19 – Composite immune signature scores across cell types × groups
  20 – Per-cell-type pairwise volcano panel (EJ vs PL-7 contrast)
  21 – Patient immunotype discovery (hierarchical clustering + dendrogram)
  22 – Cell-type ratio analyses (CD4:CD8, Cl:NonCl Mono, Naive:Memory, NK:T)

Run:
    cd /Users/yichen/Desktop/collaboration/raw
    /Users/yichen/anaconda3/bin/conda run -n cytof python cytof_advanced_analysis.py
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, ListedColormap
import seaborn as sns
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
import anndata as ad

from nature_style import (
    apply_nature_style, GROUP_PALETTE, CLUSTER_PALETTE,
    HEATMAP_DIVERG, HEATMAP_SEQ, save
)
apply_nature_style()

SC, DC = 3.46, 7.09

# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════
H5AD_PATH = "./cytof_output_v3/cytof_analyzed_v3.h5ad"
OUT_DIR   = "./cytof_output_nature"

# ── cluster → cell type (match cytof_replot_nature.py exactly) ──────────────
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
DROP_CELLTYPES = {"T-Myeloid doublets", "CD16+ Granulocyte"}

# ── patient clinical metadata (de-identified; from the clinical sample sheet) ──
# Sample barcode ↔ patient ID ↔ autoantibody group:
#   FH0002 = P08 (Jo-1)
#   FH0003 = P07 (Jo-1)
#   FH0009 = P06 (PL-12)
#   FH0011 = P05 (EJ)
#   FH0012 = P04 (PL-7)
#   FH0013 = P03 (PL-12)
#   FH0014 = P02 (Jo-1)
#   FH0015 = P01 (PL-12)
#   FH0016 = P10 (EJ)
#   FH0017 = P09 (PL-7)
PATIENT_META = pd.DataFrame([
    # sample,          pid, name,     ab,    age, myositis, arthritis, rash, ILD, CK,     LDH,    ESR,    CRP,    FVC,    DLCO
    ["L01912_FH0002",   8, "P08", "Jo-1",  57,   1,        0,         0,    1,  3838,   755,    11,     1.71,   np.nan, np.nan],
    ["L01912_FH0003",   7, "P07", "Jo-1",  52,   1,        1,         0,    1,  469,    343,    30,     0.73,   39.4,   27.9 ],
    ["L01912_FH0009",   6, "P06", "PL-12", 50,   1,        0,         0,    1,  26,     202,    np.nan, 0.25,   np.nan, np.nan],
    ["L01912_FH0011",   5, "P05", "EJ",    67,   1,        0,         0,    1,  26,     147,    2,      np.nan, np.nan, np.nan],
    ["L01912_FH0012",   4, "P04",   "PL-7",  63,   0,        1,         0,    1,  41,     195,    7,      np.nan, 86.5,   77.3 ],
    ["L01912_FH0013",   3, "P03", "PL-12", 67,   1,        1,         1,    0,  13,     318,    13,     4.6,    np.nan, np.nan],
    ["L01912_FH0014",   2, "P02", "Jo-1",  60,   1,        0,         0,    1,  np.nan, 374,    6,      np.nan, np.nan, np.nan],
    ["L01912_FH0015",   1, "P01", "PL-12", 83,   0,        0,         0,    1,  43,     315,    2,      0.25,   np.nan, np.nan],
    ["L01912_FH0016",  10, "P10", "EJ",    66,   1,        1,         0,    1,  np.nan, np.nan, 28,     np.nan, np.nan, np.nan],
    ["L01912_FH0017",   9, "P09", "PL-7",  52,   0,        0,         1,    1,  15,     np.nan, 13,     0.25,   65,     46.1 ],
], columns=["sample","pid","name","antibody","age","myositis","arthritis","rash",
            "ILD","CK","LDH","ESR","CRP","FVC","DLCO"])
PATIENT_META["log10_CK"]  = np.log10(PATIENT_META["CK"].astype(float))
PATIENT_META["log10_LDH"] = np.log10(PATIENT_META["LDH"].astype(float))

GROUP_ORDER = ["Jo-1", "PL-12", "EJ", "PL-7"]

# ── immune signature definitions (marker → functional module) ───────────────
SIGNATURES = {
    "Th1":         ["CXCR3", "CCR6"],              # CXCR3+CCR6-
    "Th17":        ["CCR6", "CCR4"],               # CCR6+CCR4+
    "Tfh":         ["CXCR5", "ICOS", "PD-1"],
    "Treg":        ["CD25", "CD127"],              # CD25+CD127lo (inv CD127)
    "Cytotoxic":   ["GranzymeB"],
    "Exhaustion":  ["PD-1", "TIGIT", "Tim-3", "CD39"],
    "Activation":  ["CD38", "HLA-DR", "CD69", "Ki-67"],
    "Naive":       ["CCR7", "CD45RA", "CD27", "TCF1"],
    "Memory":      ["CD45RO", "CD95"],
    "Migration":   ["CCR4", "CXCR3", "CCR6"],
}

# ════════════════════════════════════════════════════════════════════════════
# LOAD
# ════════════════════════════════════════════════════════════════════════════
print(f"\n[Loading] {H5AD_PATH}")
adata = ad.read_h5ad(H5AD_PATH)
adata.obs_names_make_unique()
adata.obs["cell_type"] = (
    adata.obs["leiden"].astype(str).map(CLUSTER_ANNOTATION).fillna("Unannotated")
)
adata.obs["group"] = pd.Categorical(adata.obs["group"], categories=GROUP_ORDER, ordered=True)

# keep real biological cell types only
adata = adata[~adata.obs["cell_type"].isin(DROP_CELLTYPES)].copy()
print(f"  After dropping spurious clusters: {adata.n_obs:,} cells "
      f"× {adata.n_vars} markers × {adata.obs['cell_type'].nunique()} cell types")

os.makedirs(OUT_DIR, exist_ok=True)

# ── per-sample cell-type proportions (pivot) ────────────────────────────────
prop_long = (
    adata.obs.groupby(["sample", "cell_type"], observed=True).size()
    .groupby(level=0).transform(lambda x: x / x.sum())
    .rename("prop").reset_index()
)
prop_wide = prop_long.pivot(index="sample", columns="cell_type", values="prop").fillna(0)
# add group + clinical metadata
meta_idx = PATIENT_META.set_index("sample")
prop_wide = prop_wide.loc[meta_idx.index]           # align order
prop_wide = prop_wide * 100.0                       # %

# per-sample per-celltype marker mean (sample × celltype × marker)
def per_sample_celltype_marker_mean(adata):
    X = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
    X["sample"]    = adata.obs["sample"].values
    X["cell_type"] = adata.obs["cell_type"].values
    return X.groupby(["sample","cell_type"], observed=True).mean()

SCT = per_sample_celltype_marker_mean(adata)   # MultiIndex DF

# per-sample global mean
GLOBAL = (
    pd.DataFrame(adata.X, columns=adata.var_names)
    .assign(sample=adata.obs["sample"].values)
    .groupby("sample").mean()
    .loc[meta_idx.index]
)

print("  Pre-computation complete.")

# ════════════════════════════════════════════════════════════════════════════
# FIG 15 — Clinical-immune integrated heatmap
# ════════════════════════════════════════════════════════════════════════════
print("\n  Fig 15: Clinical-immune integrated heatmap …")

# Z-score cell type proportions across samples
zmat = (prop_wide - prop_wide.mean()) / prop_wide.std(ddof=0).replace(0, 1)

# hierarchical clustering on samples
link_samples = hierarchy.linkage(pdist(zmat.values, metric="euclidean"),
                                 method="average")
link_cells   = hierarchy.linkage(pdist(zmat.values.T, metric="euclidean"),
                                 method="average")
sample_order = hierarchy.leaves_list(link_samples)
cell_order   = hierarchy.leaves_list(link_cells)
zmat_ord = zmat.iloc[sample_order, cell_order]

# build clinical annotation block (samples × features)
clin_cols = ["antibody","age","ILD","myositis","arthritis","rash",
             "log10_CK","log10_LDH","ESR","CRP"]
clin = meta_idx.loc[zmat_ord.index, clin_cols].copy()

fig = plt.figure(figsize=(DC+0.7, 5.6))
gs = gridspec.GridSpec(
    2, 3, figure=fig,
    width_ratios=[0.55, 1.35, 6.0],
    height_ratios=[0.9, 8.0],
    wspace=0.04, hspace=0.02,
)

# column dendrogram (cell types) — top
ax_dtop = fig.add_subplot(gs[0, 2])
hierarchy.dendrogram(link_cells, ax=ax_dtop, color_threshold=0, above_threshold_color="#555",
                     no_labels=True)
ax_dtop.axis("off")

# row dendrogram (samples) — left
ax_dleft = fig.add_subplot(gs[1, 0])
hierarchy.dendrogram(link_samples, ax=ax_dleft, orientation="left",
                     color_threshold=0, above_threshold_color="#555",
                     no_labels=True)
ax_dleft.axis("off")

# clinical annotation strip
ax_clin = fig.add_subplot(gs[1, 1])
# header row above clinical columns
ax_clin_hdr = fig.add_subplot(gs[0, 1])
ax_clin_hdr.axis("off")
# render as multi-column discrete/continuous — use separate mini-heatmaps per column
ax_clin.axis("off")
nrows = len(clin)
ncol = len(clin_cols)
cell_w, cell_h = 1.0 / ncol, 1.0 / nrows

def _text_color(bg):
    r, g, b = bg[:3]
    return "white" if (0.299*r + 0.587*g + 0.114*b) < 0.55 else "#222"

for ci, col in enumerate(clin_cols):
    v = clin[col]
    if col == "antibody":
        for ri, val in enumerate(v):
            c = GROUP_PALETTE.get(val, "#ccc")
            ax_clin.add_patch(plt.Rectangle((ci*cell_w, 1 - (ri+1)*cell_h),
                                            cell_w, cell_h, facecolor=c,
                                            edgecolor="white", linewidth=0.4))
    elif col in {"ILD","myositis","arthritis","rash"}:
        for ri, val in enumerate(v):
            if pd.isna(val):
                c, ec = "#eeeeee", "#cccccc"
            elif val == 1:
                c, ec = "#34495E", "white"
            else:
                c, ec = "#FFFFFF", "#cccccc"
            ax_clin.add_patch(plt.Rectangle((ci*cell_w, 1 - (ri+1)*cell_h),
                                            cell_w, cell_h, facecolor=c,
                                            edgecolor=ec, linewidth=0.5))
    else:
        vals = v.astype(float).values
        if np.isfinite(vals).sum() >= 2:
            vmin, vmax = np.nanmin(vals), np.nanmax(vals)
            rng = vmax - vmin if vmax > vmin else 1.0
        else:
            vmin, vmax, rng = 0, 1, 1
        cmap = HEATMAP_SEQ
        for ri, val in enumerate(vals):
            if np.isnan(val):
                c = "#eeeeee"
            else:
                c = cmap((val - vmin) / rng)
            ax_clin.add_patch(plt.Rectangle((ci*cell_w, 1 - (ri+1)*cell_h),
                                            cell_w, cell_h, facecolor=c,
                                            edgecolor="white", linewidth=0.4))
    # header in separate axis to avoid overlap
    ax_clin_hdr.text(ci*cell_w + cell_w/2, 0.02, col, rotation=60,
                     ha="left", va="bottom", fontsize=6, color="#222")

ax_clin.set_xlim(0, 1); ax_clin.set_ylim(0, 1)
ax_clin_hdr.set_xlim(0, 1); ax_clin_hdr.set_ylim(0, 1)

# main z-score heatmap
ax_hm = fig.add_subplot(gs[1, 2])
vlim = np.nanpercentile(np.abs(zmat_ord.values), 99)
im = ax_hm.imshow(zmat_ord.values, aspect="auto", cmap=HEATMAP_DIVERG,
                  vmin=-vlim, vmax=vlim)
ax_hm.set_xticks(range(zmat_ord.shape[1]))
ax_hm.set_xticklabels(zmat_ord.columns, rotation=60, ha="right", fontsize=6)
# move sample labels to right side so they don't collide with clinical strip
row_labels = [f"P{int(meta_idx.loc[s,'pid']):02d} · {meta_idx.loc[s,'antibody']}"
              for s in zmat_ord.index]
ax_hm.set_yticks(range(zmat_ord.shape[0]))
ax_hm.set_yticklabels([""] * zmat_ord.shape[0])
ax_hm.tick_params(axis="y", length=0)
ax_hm.tick_params(axis="x", length=0)
ax_hm.yaxis.tick_right()
ax_hm.set_yticklabels(row_labels, fontsize=6)
for tick in ax_hm.yaxis.get_majorticklabels():
    tick.set_verticalalignment("center")

# colorbar — move to the left side to avoid collision with right-side sample labels
cax = fig.add_axes([0.02, 0.25, 0.012, 0.22])
cb = fig.colorbar(im, cax=cax)
cb.set_label("Proportion (Z-score)", fontsize=6)
cb.ax.tick_params(labelsize=5)

fig.suptitle("Integrated clinical-immune landscape of antisynthetase syndrome",
             fontsize=8.5, y=0.98)
save(fig, "15_clinical_immune_heatmap", OUT_DIR)
print("    Saved: 15_clinical_immune_heatmap.pdf/.png")

# ════════════════════════════════════════════════════════════════════════════
# FIG 16 — Patient PCA / MDS
# ════════════════════════════════════════════════════════════════════════════
print("\n  Fig 16: Patient PCA + MDS …")

fig, axes = plt.subplots(1, 3, figsize=(DC, 2.5))

# (a) PCA on cell-type composition
Xc = StandardScaler().fit_transform(prop_wide.values)
pca = PCA(n_components=2).fit(Xc)
Zc = pca.transform(Xc)
ax = axes[0]
for g in GROUP_ORDER:
    idx = [i for i, s in enumerate(prop_wide.index) if meta_idx.loc[s,"antibody"] == g]
    ax.scatter(Zc[idx,0], Zc[idx,1], c=GROUP_PALETTE[g], s=55,
               edgecolor="white", linewidth=0.9, label=g, zorder=3)
for i, s in enumerate(prop_wide.index):
    ax.text(Zc[i,0], Zc[i,1]+0.08, f"P{int(meta_idx.loc[s,'pid']):02d}",
            fontsize=5.5, ha="center", color="#333", zorder=4)
ax.set_xlabel(f"PC1  ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2  ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("PCA · cell composition", fontsize=7.5)
ax.legend(fontsize=5.5, loc="best", frameon=False, ncol=2)
ax.axhline(0, color="#ddd", lw=0.4, zorder=1)
ax.axvline(0, color="#ddd", lw=0.4, zorder=1)

# (b) PCA on global marker MFIs
Xg = StandardScaler().fit_transform(GLOBAL.values)
pca2 = PCA(n_components=2).fit(Xg)
Zg = pca2.transform(Xg)
ax = axes[1]
for g in GROUP_ORDER:
    idx = [i for i, s in enumerate(GLOBAL.index) if meta_idx.loc[s,"antibody"] == g]
    ax.scatter(Zg[idx,0], Zg[idx,1], c=GROUP_PALETTE[g], s=55,
               edgecolor="white", linewidth=0.9, zorder=3)
for i, s in enumerate(GLOBAL.index):
    ax.text(Zg[i,0], Zg[i,1]+0.15, f"P{int(meta_idx.loc[s,'pid']):02d}",
            fontsize=5.5, ha="center", color="#333", zorder=4)
ax.set_xlabel(f"PC1  ({pca2.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2  ({pca2.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("PCA · global marker MFI", fontsize=7.5)
ax.axhline(0, color="#ddd", lw=0.4, zorder=1)
ax.axvline(0, color="#ddd", lw=0.4, zorder=1)

# (c) MDS on combined (composition + MFI)
comb = np.hstack([StandardScaler().fit_transform(prop_wide.values),
                  StandardScaler().fit_transform(GLOBAL.values)])
D = squareform(pdist(comb, metric="euclidean"))
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0,
          n_init=20, normalized_stress="auto")
Zm = mds.fit_transform(D)
ax = axes[2]
for g in GROUP_ORDER:
    idx = [i for i, s in enumerate(prop_wide.index) if meta_idx.loc[s,"antibody"] == g]
    ax.scatter(Zm[idx,0], Zm[idx,1], c=GROUP_PALETTE[g], s=55,
               edgecolor="white", linewidth=0.9, zorder=3)
for i, s in enumerate(prop_wide.index):
    ax.text(Zm[i,0], Zm[i,1]+0.06, f"P{int(meta_idx.loc[s,'pid']):02d}",
            fontsize=5.5, ha="center", color="#333", zorder=4)
ax.set_xlabel("MDS-1")
ax.set_ylabel("MDS-2")
ax.set_title("MDS · composition + MFI", fontsize=7.5)
ax.axhline(0, color="#ddd", lw=0.4, zorder=1)
ax.axvline(0, color="#ddd", lw=0.4, zorder=1)

fig.suptitle("Unsupervised patient-level embedding by autoantibody group",
             fontsize=8.5, y=1.02)
fig.tight_layout()
save(fig, "16_patient_pca_mds", OUT_DIR)
print("    Saved: 16_patient_pca_mds.pdf/.png")

# ════════════════════════════════════════════════════════════════════════════
# FIG 17 — Immune ↔ clinical correlation heatmap
# ════════════════════════════════════════════════════════════════════════════
print("\n  Fig 17: Immune-clinical correlation heatmap …")

clin_feat = ["age","log10_CK","log10_LDH","ESR","CRP",
             "myositis","arthritis","rash","ILD"]
# immune features: cell type proportions + a few global scores
global_scores = pd.DataFrame(index=GLOBAL.index)
for name, mks in SIGNATURES.items():
    use = [m for m in mks if m in GLOBAL.columns]
    if name == "Treg":
        # CD25 high, CD127 low
        if "CD25" in GLOBAL.columns and "CD127" in GLOBAL.columns:
            global_scores["Treg_score"] = GLOBAL["CD25"] - GLOBAL["CD127"]
    else:
        if use:
            global_scores[f"{name}_score"] = GLOBAL[use].mean(axis=1)

immune_block = pd.concat([prop_wide, global_scores], axis=1)

rho_mat = pd.DataFrame(index=immune_block.columns, columns=clin_feat, dtype=float)
p_mat   = pd.DataFrame(index=immune_block.columns, columns=clin_feat, dtype=float)
for feat in clin_feat:
    x = meta_idx.loc[immune_block.index, feat].astype(float)
    for imm in immune_block.columns:
        y = immune_block[imm].astype(float)
        mask = x.notna() & y.notna()
        if mask.sum() >= 4:
            r, p = stats.spearmanr(x[mask], y[mask])
            rho_mat.loc[imm, feat] = r
            p_mat.loc[imm, feat]   = p
# FDR across all tests
flat_p = p_mat.values.flatten()
flat_p_finite = flat_p[~np.isnan(flat_p)]
if flat_p_finite.size > 0:
    rej, q, _, _ = multipletests(flat_p_finite, alpha=0.1, method="fdr_bh")
    q_full = np.full_like(flat_p, np.nan)
    q_full[~np.isnan(flat_p)] = q
    q_mat = pd.DataFrame(q_full.reshape(p_mat.shape),
                         index=p_mat.index, columns=p_mat.columns)
else:
    q_mat = p_mat.copy()

fig, ax = plt.subplots(figsize=(SC+1.7, 5.5))
vlim = 1.0
im = ax.imshow(rho_mat.astype(float).values, aspect="auto",
               cmap=HEATMAP_DIVERG, vmin=-vlim, vmax=vlim)

# significance dots
for i in range(rho_mat.shape[0]):
    for j in range(rho_mat.shape[1]):
        q = q_mat.iloc[i, j]
        p = p_mat.iloc[i, j]
        if pd.notna(q) and q < 0.1:
            ax.scatter(j, i, s=22, facecolor="white", edgecolor="black", linewidth=0.6, zorder=5)
        elif pd.notna(p) and p < 0.05:
            ax.scatter(j, i, s=8, color="white", zorder=5)

ax.set_xticks(range(len(clin_feat)))
ax.set_xticklabels(clin_feat, rotation=45, ha="right", fontsize=6)
ax.set_yticks(range(len(rho_mat.index)))
ax.set_yticklabels(rho_mat.index, fontsize=5.5)
ax.tick_params(length=0)

# separator between proportion and score blocks
ax.axhline(prop_wide.shape[1] - 0.5, color="#222", lw=0.6)

cax = fig.add_axes([0.92, 0.30, 0.018, 0.28])
cb = fig.colorbar(im, cax=cax)
cb.set_label("Spearman ρ", fontsize=6)
cb.ax.tick_params(labelsize=5)

ax.set_title("Immune features × clinical variables\n(● p<0.05    ○ FDR q<0.10)",
             fontsize=7.5, pad=6)
save(fig, "17_clinical_correlation", OUT_DIR)
print("    Saved: 17_clinical_correlation.pdf/.png")
rho_mat.to_csv(f"{OUT_DIR}/17_clinical_correlation_rho.csv")
p_mat.to_csv(f"{OUT_DIR}/17_clinical_correlation_pval.csv")

# ════════════════════════════════════════════════════════════════════════════
# FIG 18 — Cell-type co-variation correlation matrix
# ════════════════════════════════════════════════════════════════════════════
print("\n  Fig 18: Cell-type co-variation correlation matrix …")

cell_corr = prop_wide.corr(method="spearman")
link = hierarchy.linkage(pdist(cell_corr.values, metric="euclidean"), method="average")
order = hierarchy.leaves_list(link)
cell_corr = cell_corr.iloc[order, order]

fig, ax = plt.subplots(figsize=(SC+1.3, SC+1.3))
im = ax.imshow(cell_corr.values, aspect="auto", cmap=HEATMAP_DIVERG,
               vmin=-1, vmax=1)

# overlay significance from bootstrapping
# simple: highlight |rho| > 0.7 with black border
for i in range(cell_corr.shape[0]):
    for j in range(cell_corr.shape[1]):
        if i != j and abs(cell_corr.iloc[i, j]) >= 0.7:
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                       fill=False, edgecolor="black", linewidth=0.5))

ax.set_xticks(range(len(cell_corr)))
ax.set_xticklabels(cell_corr.columns, rotation=60, ha="right", fontsize=6)
ax.set_yticks(range(len(cell_corr)))
ax.set_yticklabels(cell_corr.index, fontsize=6)
ax.tick_params(length=0)

cax = fig.add_axes([0.94, 0.28, 0.018, 0.28])
cb = fig.colorbar(im, cax=cax)
cb.set_label("Spearman ρ", fontsize=6)
cb.ax.tick_params(labelsize=5)

ax.set_title("Co-variation of cell-type abundances\n(boxes: |ρ|≥0.70)",
             fontsize=7.5, pad=6)
save(fig, "18_celltype_covariation", OUT_DIR)
print("    Saved: 18_celltype_covariation.pdf/.png")

# ════════════════════════════════════════════════════════════════════════════
# FIG 19 — Immune signature scores across cell types × groups
# ════════════════════════════════════════════════════════════════════════════
print("\n  Fig 19: Immune signature scores …")

# Relevant cell types to score (focus on T / NK)
SIG_CELLTYPES = [
    "CD4 Naive T", "CD4 Central Memory T", "CD4 Effector Memory T",
    "CD4 Th1-like", "CD8 Naive T", "CD8 Effector T",
    "γδ T cell", "NK cell",
]
SIG_ORDER = ["Naive","Memory","Th1","Th17","Tfh","Treg",
             "Cytotoxic","Exhaustion","Activation","Migration"]

def compute_signature(df_marker, name):
    mks = SIGNATURES[name]
    use = [m for m in mks if m in df_marker.columns]
    if name == "Treg":
        if "CD25" in df_marker.columns and "CD127" in df_marker.columns:
            return df_marker["CD25"] - df_marker["CD127"]
    if use:
        return df_marker[use].mean(axis=1)
    return None

# build matrix: (celltype × signature) of group-mean z-scores, with KW p across groups
z_records = []   # long format for plotting
kw_records = []
for ct in SIG_CELLTYPES:
    # per-sample marker means for this cell type
    sub = SCT.xs(ct, level="cell_type")                  # samples × markers
    sub = sub.loc[sub.index.isin(meta_idx.index)]
    for sig in SIG_ORDER:
        vals = compute_signature(sub, sig)
        if vals is None:
            continue
        # z-score across samples
        z = (vals - vals.mean()) / (vals.std(ddof=0) + 1e-9)
        # group means
        groups = meta_idx.loc[z.index, "antibody"].values
        kw_groups = [z[meta_idx.loc[z.index,"antibody"].values == g].values for g in GROUP_ORDER]
        kw_groups = [g for g in kw_groups if len(g) >= 1]
        try:
            kws, kwp = stats.kruskal(*kw_groups)
        except Exception:
            kws, kwp = np.nan, np.nan
        kw_records.append({"cell_type": ct, "signature": sig, "kw_p": kwp})
        for g in GROUP_ORDER:
            mask = groups == g
            if mask.any():
                z_records.append({
                    "cell_type": ct, "signature": sig, "group": g,
                    "mean_z": float(z[mask].mean()),
                })

zdf = pd.DataFrame(z_records)
kwdf = pd.DataFrame(kw_records)
# FDR across all tests
if kwdf["kw_p"].notna().any():
    rej, q, _, _ = multipletests(kwdf["kw_p"].fillna(1.0), alpha=0.05, method="fdr_bh")
    kwdf["fdr_q"] = q
else:
    kwdf["fdr_q"] = np.nan

# plot: for each group, one small-multiples heatmap (cell types × signatures) of mean z
fig, axes = plt.subplots(1, len(GROUP_ORDER), figsize=(DC+1.0, 2.9),
                         sharey=False,
                         gridspec_kw={"wspace": 0.08,
                                      "width_ratios": [1.55, 1.0, 1.0, 1.0]})
vlim = 1.5
for axi, g in enumerate(GROUP_ORDER):
    ax = axes[axi]
    mat = (zdf[zdf["group"] == g]
           .pivot(index="cell_type", columns="signature", values="mean_z")
           .reindex(index=SIG_CELLTYPES, columns=SIG_ORDER))
    im = ax.imshow(mat.values, aspect="auto", cmap=HEATMAP_DIVERG,
                   vmin=-vlim, vmax=vlim)
    ax.set_xticks(range(len(SIG_ORDER)))
    ax.set_xticklabels(SIG_ORDER, rotation=60, ha="right", fontsize=6)
    ax.set_yticks(range(len(SIG_CELLTYPES)))
    if axi == 0:
        ax.set_yticklabels(SIG_CELLTYPES, fontsize=6.3)
    else:
        ax.set_yticklabels([])
    ax.set_title(g, fontsize=7.5, color=GROUP_PALETTE[g], pad=2)
    ax.tick_params(length=0)
    # mark cells with FDR q<0.10 (across groups, for this cell type × signature)
    for i, ct in enumerate(SIG_CELLTYPES):
        for j, sig in enumerate(SIG_ORDER):
            row = kwdf[(kwdf["cell_type"] == ct) & (kwdf["signature"] == sig)]
            if not row.empty and pd.notna(row.iloc[0]["fdr_q"]) and row.iloc[0]["fdr_q"] < 0.1:
                ax.plot(j, i, marker="o", markersize=3.5,
                        markerfacecolor="none", markeredgecolor="black",
                        markeredgewidth=0.6)

cax = fig.add_axes([0.93, 0.25, 0.013, 0.30])
cb = fig.colorbar(im, cax=cax)
cb.set_label("Mean Z-score", fontsize=6)
cb.ax.tick_params(labelsize=5)

fig.suptitle("Composite immune signature scores by cell type × autoantibody group "
             "(○ FDR q<0.10 across groups)", fontsize=8, y=1.02)
save(fig, "19_immune_signatures", OUT_DIR)
print("    Saved: 19_immune_signatures.pdf/.png")
zdf.to_csv(f"{OUT_DIR}/19_immune_signatures.csv", index=False)
kwdf.to_csv(f"{OUT_DIR}/19_immune_signatures_KW.csv", index=False)

# ════════════════════════════════════════════════════════════════════════════
# FIG 20 — Per-cell-type pairwise volcano panel (EJ vs PL-7)
# ════════════════════════════════════════════════════════════════════════════
print("\n  Fig 20: Per-cell-type pairwise volcano panel …")

from adjustText import adjust_text

TARGET_CTS = [
    "CD4 Naive T", "CD4 Central Memory T", "CD4 Th1-like",
    "CD8 Effector T", "NK cell", "Classical Monocyte",
    "Non-classical Monocyte", "Naive B", "mDC",
]
# biologically relevant markers (lineage-appropriate per CT)
RELEVANT = {
    "CD4 Naive T":          ["CD3","CD4","CD45RA","CCR7","CD27","CD127","CD95","CD45RO","CXCR3","CCR4","CCR6","CD38","CD69","HLA-DR","Ki-67","PD-1","TIGIT","Tim-3","CD39","GranzymeB","ICOS","TCF1"],
    "CD4 Central Memory T": ["CD3","CD4","CD45RA","CD45RO","CCR7","CCR4","CCR6","CXCR3","CD27","CD127","CD95","CD38","CD69","HLA-DR","Ki-67","PD-1","TIGIT","Tim-3","CD39","GranzymeB","ICOS","CXCR5","TCF1","CD25"],
    "CD4 Th1-like":         ["CD3","CD4","CXCR3","CCR4","CCR6","CD45RO","CD95","CD38","CD69","HLA-DR","PD-1","TIGIT","Tim-3","CD39","GranzymeB","Ki-67","ICOS","TCF1","CD27","CD127"],
    "CD8 Effector T":       ["CD3","CD8a","CD45RA","CD45RO","CCR7","CD27","CD95","CD38","CD69","HLA-DR","PD-1","TIGIT","Tim-3","CD39","GranzymeB","Ki-67","TCF1","CD127","CXCR3","CCR4"],
    "NK cell":              ["CD56","CD16","CD38","HLA-DR","GranzymeB","Ki-67","TIGIT","Tim-3","PD-1","CD69"],
    "Classical Monocyte":   ["CD14","CD16","CD11b","CD11c","HLA-DR","CD38","CD39","CCR4","CXCR3","Tim-3","TIGIT","CD69","Ki-67"],
    "Non-classical Monocyte":["CD14","CD16","CD11b","CD11c","HLA-DR","CD38","CD39","CD56","CCR4","CXCR3","Tim-3","TIGIT"],
    "Naive B":              ["CD19","CD20","CD22","IgD","IgM","CD24","CD27","CD38","HLA-DR","CXCR5","Ki-67"],
    "mDC":                  ["HLA-DR","CD11c","CD16","CD38","CD39","Tim-3","CD11b"],
}

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2: return np.nan
    s2 = ((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx+ny-2)
    sd = np.sqrt(s2)
    return (np.mean(x) - np.mean(y)) / sd if sd > 0 else np.nan

def per_cell_volcano(ax, ct, gA, gB):
    cells = adata[(adata.obs["cell_type"] == ct) &
                  (adata.obs["group"].isin([gA, gB]))].copy()
    if cells.n_obs < 20:
        ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center")
        ax.set_axis_off()
        return
    mks = [m for m in RELEVANT.get(ct, list(adata.var_names)) if m in adata.var_names]
    X = pd.DataFrame(cells.X, columns=adata.var_names)[mks]
    grp = cells.obs["group"].values
    records = []
    for m in mks:
        xA = X.loc[grp == gA, m].values
        xB = X.loc[grp == gB, m].values
        if len(xA) < 10 or len(xB) < 10:
            continue
        try:
            _, p = stats.mannwhitneyu(xA, xB, alternative="two-sided")
        except Exception:
            p = 1.0
        d = cohens_d(xA, xB)
        records.append({"marker": m, "d": d, "p": p})
    dfv = pd.DataFrame(records)
    if dfv.empty:
        ax.set_axis_off(); return
    _, q, _, _ = multipletests(dfv["p"].fillna(1), method="fdr_bh")
    dfv["q"] = q
    dfv["neglog10q"] = -np.log10(dfv["q"].clip(1e-300))
    # plot
    ax.axhline(-np.log10(0.05), color="#888", lw=0.4, ls="--")
    ax.axvline(0, color="#888", lw=0.4)
    for _, r in dfv.iterrows():
        if r["q"] < 0.05 and abs(r["d"]) > 0.3:
            c = GROUP_PALETTE[gA] if r["d"] > 0 else GROUP_PALETTE[gB]
            ax.scatter(r["d"], r["neglog10q"], s=11, c=c, edgecolor="white", linewidth=0.3)
        else:
            ax.scatter(r["d"], r["neglog10q"], s=7, c="#bbbbbb", edgecolor="none")
    texts = []
    top = dfv.reindex(dfv["neglog10q"].sort_values(ascending=False).index).head(6)
    for _, r in top.iterrows():
        t = ax.text(r["d"], r["neglog10q"], r["marker"], fontsize=5, color="#222")
        texts.append(t)
    try:
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="#888", lw=0.3),
                    expand_points=(1.1, 1.2), expand_text=(1.1, 1.2))
    except Exception:
        pass
    ax.set_title(ct, fontsize=6.5, pad=2)
    ax.set_xlabel("Cohen's d", fontsize=5.5)
    ax.set_ylabel(r"$-\log_{10}(q)$", fontsize=5.5)
    ax.tick_params(labelsize=5)

gA, gB = "EJ", "PL-7"
nr, nc = 3, 3
fig, axes = plt.subplots(nr, nc, figsize=(DC, 6.2))
axes = axes.flatten()
for i, ct in enumerate(TARGET_CTS):
    per_cell_volcano(axes[i], ct, gA, gB)
fig.suptitle(f"Per-cell-type differential marker expression: {gA}  vs  {gB}",
             fontsize=8.5, y=1.0)
# group legend
h1 = mpatches.Patch(color=GROUP_PALETTE[gA], label=f"↑ in {gA}")
h2 = mpatches.Patch(color=GROUP_PALETTE[gB], label=f"↑ in {gB}")
fig.legend(handles=[h1, h2], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01),
           fontsize=7, frameon=False)
fig.tight_layout()
save(fig, "20_percelltype_volcano_EJvsPL7", OUT_DIR)
print("    Saved: 20_percelltype_volcano_EJvsPL7.pdf/.png")

# ════════════════════════════════════════════════════════════════════════════
# FIG 21 — Immunotype discovery
# ════════════════════════════════════════════════════════════════════════════
print("\n  Fig 21: Immunotype discovery …")

# feature matrix: composition + global marker MFI + signature scores
feat = pd.concat([prop_wide,
                  GLOBAL.add_prefix("MFI_"),
                  global_scores], axis=1)
Xf = StandardScaler().fit_transform(feat.values)
link = hierarchy.linkage(Xf, method="ward")
# cut at k=3 (empirical)
k = 3
groups = hierarchy.fcluster(link, t=k, criterion="maxclust")

fig = plt.figure(figsize=(DC, 3.5))
gs = gridspec.GridSpec(2, 2, figure=fig,
                        width_ratios=[1, 2.2], height_ratios=[1, 0.2],
                        wspace=0.1, hspace=0.08)

# dendrogram
ax_d = fig.add_subplot(gs[0, 0])
dd = hierarchy.dendrogram(
    link, labels=[f"P{int(meta_idx.loc[s,'pid']):02d}" for s in feat.index],
    orientation="left", ax=ax_d, color_threshold=link[-k+1, 2] - 1e-6,
    above_threshold_color="#888",
)
ax_d.tick_params(axis="y", labelsize=6)
ax_d.set_xlabel("Ward distance", fontsize=6)
ax_d.spines["top"].set_visible(False); ax_d.spines["right"].set_visible(False)

# aligned metadata heatmap (antibody + immunotype + key clinical)
leaf_order = dd["leaves"]
labels_in_order = [feat.index[i] for i in leaf_order][::-1]  # dendrogram orientation
ax_m = fig.add_subplot(gs[0, 1])
meta_cols = ["antibody","immunotype","ILD","myositis","arthritis","rash"]
M = meta_idx.loc[labels_in_order].copy()
M["immunotype"] = [f"IT{g}" for g in groups[[feat.index.get_loc(s) for s in labels_in_order]]]
IT_PALETTE = {"IT1": "#E67E22", "IT2": "#1F618D", "IT3": "#117A65"}
ax_m.axis("off")
nrows, ncols = len(M), len(meta_cols)
cw, ch = 1/ncols, 1/nrows
for ci, col in enumerate(meta_cols):
    for ri, val in enumerate(M[col].values):
        if col == "antibody":
            c = GROUP_PALETTE.get(val, "#ccc")
        elif col == "immunotype":
            c = IT_PALETTE[val]
        else:
            if pd.isna(val):
                c, ec = "#eee", "#cccccc"
            elif val == 1:
                c, ec = "#34495E", "white"
            else:
                c, ec = "#FFFFFF", "#cccccc"
        ax_m.add_patch(plt.Rectangle((ci*cw, 1-(ri+1)*ch), cw, ch,
                                     facecolor=c, edgecolor=ec, linewidth=0.5))
        if col == "antibody":
            ax_m.text(ci*cw + cw/2, 1-(ri+0.5)*ch, val,
                      ha="center", va="center", fontsize=5,
                      color=_text_color(matplotlib.colors.to_rgb(c)))
        elif col == "immunotype":
            ax_m.text(ci*cw + cw/2, 1-(ri+0.5)*ch, val,
                      ha="center", va="center", fontsize=5,
                      color=_text_color(matplotlib.colors.to_rgb(c)))
    ax_m.text(ci*cw + cw/2, 1.01, col, rotation=45, ha="left", va="bottom",
              fontsize=6)
ax_m.set_xlim(0, 1); ax_m.set_ylim(0, 1)

# legend panel
ax_l = fig.add_subplot(gs[1, :])
ax_l.axis("off")
handles = [mpatches.Patch(color=c, label=g) for g, c in GROUP_PALETTE.items()]
handles += [mpatches.Patch(color=c, label=k) for k, c in IT_PALETTE.items()]
handles += [mpatches.Patch(color="#34495E", label="present"),
            mpatches.Patch(color="#FFFFFF", edgecolor="#888", linewidth=0.5, label="absent")]
ax_l.legend(handles=handles, loc="center", ncol=5, fontsize=6, frameon=False)

fig.suptitle("Immunotype discovery by unsupervised patient clustering",
             fontsize=8.5, y=1.0)
save(fig, "21_immunotype_discovery", OUT_DIR)
# save assignment table
it_df = pd.DataFrame({
    "sample": feat.index,
    "pid":    [int(meta_idx.loc[s,"pid"]) for s in feat.index],
    "antibody":[meta_idx.loc[s,"antibody"] for s in feat.index],
    "immunotype": [f"IT{g}" for g in groups],
})
it_df.to_csv(f"{OUT_DIR}/21_immunotype_assignments.csv", index=False)
print("    Saved: 21_immunotype_discovery.pdf/.png")

# ════════════════════════════════════════════════════════════════════════════
# FIG 22 — Cell-type ratio analyses
# ════════════════════════════════════════════════════════════════════════════
print("\n  Fig 22: Cell-type ratio analyses …")

def ratio(num_list, den_list, name):
    num = prop_wide[[c for c in num_list if c in prop_wide.columns]].sum(axis=1)
    den = prop_wide[[c for c in den_list if c in prop_wide.columns]].sum(axis=1).replace(0, np.nan)
    r = (num / den).rename(name)
    return r

RATIOS = [
    ("CD4:CD8",
        ["CD4 Naive T","CD4 Central Memory T","CD4 Effector Memory T","CD4 Th1-like"],
        ["CD8 Naive T","CD8 Effector T"]),
    ("Classical : Non-classical Mono",
        ["Classical Monocyte"], ["Non-classical Monocyte"]),
    ("Memory : Naive T (CD4)",
        ["CD4 Central Memory T","CD4 Effector Memory T","CD4 Th1-like"],
        ["CD4 Naive T"]),
    ("NK : T",
        ["NK cell"],
        ["CD4 Naive T","CD4 Central Memory T","CD4 Effector Memory T","CD4 Th1-like",
         "CD8 Naive T","CD8 Effector T","γδ T cell"]),
    ("B : T",
        ["Naive B"],
        ["CD4 Naive T","CD4 Central Memory T","CD4 Effector Memory T","CD4 Th1-like",
         "CD8 Naive T","CD8 Effector T","γδ T cell"]),
    ("Inflammatory : Classical Mono",
        ["Inflammatory Monocyte"], ["Classical Monocyte"]),
]

fig, axes = plt.subplots(2, 3, figsize=(DC, 4.5))
axes = axes.flatten()
for i, (name, num, den) in enumerate(RATIOS):
    ax = axes[i]
    r = ratio(num, den, name).dropna()
    df = pd.DataFrame({"r": r.values,
                       "group": [meta_idx.loc[s,"antibody"] for s in r.index]})
    # box + jitter
    positions = np.arange(len(GROUP_ORDER))
    data = [df[df["group"] == g]["r"].values for g in GROUP_ORDER]
    bp = ax.boxplot(data, positions=positions, widths=0.55,
                    patch_artist=True, showcaps=False,
                    medianprops=dict(color="black", lw=0.8),
                    boxprops=dict(edgecolor="black", lw=0.5),
                    whiskerprops=dict(color="black", lw=0.5),
                    flierprops=dict(marker=""))
    for patch, g in zip(bp["boxes"], GROUP_ORDER):
        patch.set_facecolor(GROUP_PALETTE[g] + "66")
    for j, g in enumerate(GROUP_ORDER):
        y = df[df["group"] == g]["r"].values
        x = np.random.normal(j, 0.07, size=len(y))
        ax.scatter(x, y, s=18, color=GROUP_PALETTE[g],
                   edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xticks(positions); ax.set_xticklabels(GROUP_ORDER, fontsize=6)
    ax.set_title(name, fontsize=7.2, pad=2)
    ax.set_ylabel("Ratio", fontsize=6)
    try:
        valid = [d for d in data if len(d) > 0]
        if len(valid) >= 2:
            kws, kwp = stats.kruskal(*valid)
            txt = f"KW p = {kwp:.2g}"
        else:
            txt = ""
    except Exception:
        txt = ""
    ax.text(0.98, 0.97, txt, transform=ax.transAxes,
            ha="right", va="top", fontsize=5.5, color="#555")

fig.suptitle("Canonical immune cell-type ratios across autoantibody groups",
             fontsize=8.5, y=1.01)
fig.tight_layout()
save(fig, "22_celltype_ratios", OUT_DIR)
print("    Saved: 22_celltype_ratios.pdf/.png")

# ════════════════════════════════════════════════════════════════════════════
# FIG 23 — Immunotype ↔ clinical disease severity (the "killer" figure)
# ════════════════════════════════════════════════════════════════════════════
print("\n  Fig 23: Immunotype × clinical severity …")

# pull immunotype assignments (already computed above — reuse `groups`, `feat`)
it_assign = pd.Series([f"IT{g}" for g in groups], index=feat.index, name="immunotype")
clin_all = meta_idx.join(it_assign)

# composite clinical disease activity score (z-summed over log10_CK, log10_LDH, ESR, CRP)
for col in ["log10_CK", "log10_LDH", "ESR", "CRP"]:
    clin_all[col + "_z"] = (clin_all[col] - clin_all[col].mean()) / clin_all[col].std(ddof=0)
clin_all["DA_score"] = clin_all[["log10_CK_z","log10_LDH_z","ESR_z","CRP_z"]].mean(axis=1)

IT_ORDER = ["IT1", "IT2", "IT3"]
IT_PALETTE = {"IT1": "#E67E22", "IT2": "#1F618D", "IT3": "#117A65"}

# panels: log10_CK, log10_LDH, ESR, CRP, Age, DA_score (continuous)
#          myositis, arthritis, rash, ILD (binary → prevalence bar)
fig = plt.figure(figsize=(DC+1.5, 5.2))
gs = gridspec.GridSpec(2, 4, figure=fig, wspace=0.55, hspace=0.60)

CONT_PANELS = [
    ("log10_CK",  "log₁₀ CK (U/L)",        gs[0, 0]),
    ("log10_LDH", "log₁₀ LDH (U/L)",       gs[0, 1]),
    ("ESR",       "ESR (mm/hr)",           gs[0, 2]),
    ("CRP",       "CRP (mg/L)",            gs[0, 3]),
    ("age",       "Age (years)",           gs[1, 0]),
    ("DA_score",  "Composite disease\nactivity (z)", gs[1, 1]),
]
BIN_PANELS = [
    ("myositis",  "Clinical myositis",     gs[1, 2]),
    ("arthritis", "Arthritis",             gs[1, 3]),
]

def annotate_p(ax, p, prefix="KW"):
    if pd.isna(p):
        return
    if p < 0.001: txt = f"{prefix} p<0.001"
    elif p < 0.05: txt = f"{prefix} p={p:.3f}"
    else: txt = f"{prefix} p={p:.2f}"
    ax.text(0.97, 1.02, txt, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6,
            color=("#C0392B" if p < 0.05 else "#666"))

# --- continuous variable panels (box + jitter + KW) ---
for col, ylab, pos in CONT_PANELS:
    ax = fig.add_subplot(pos)
    data = [clin_all.loc[clin_all["immunotype"] == it, col].dropna().values
            for it in IT_ORDER]
    positions = np.arange(len(IT_ORDER))
    bp = ax.boxplot(data, positions=positions, widths=0.55,
                    patch_artist=True, showcaps=False,
                    medianprops=dict(color="black", lw=0.9),
                    boxprops=dict(edgecolor="black", lw=0.5),
                    whiskerprops=dict(color="black", lw=0.5),
                    flierprops=dict(marker=""))
    for patch, it in zip(bp["boxes"], IT_ORDER):
        patch.set_facecolor(IT_PALETTE[it] + "55")
        patch.set_edgecolor(IT_PALETTE[it])
    for j, it in enumerate(IT_ORDER):
        y = data[j]
        x = np.random.normal(j, 0.08, size=len(y))
        ax.scatter(x, y, s=22, color=IT_PALETTE[it],
                   edgecolor="white", linewidth=0.5, zorder=3)
        # label P08 (outlier in CK/LDH)
        if col in {"log10_CK", "log10_LDH", "DA_score"}:
            sub = clin_all[clin_all["immunotype"] == it].dropna(subset=[col])
            for _, row in sub.iterrows():
                if (col == "log10_CK"  and row[col] > 3.0) or \
                   (col == "log10_LDH" and row[col] > 2.7) or \
                   (col == "DA_score" and row[col] > 1.5):
                    ax.text(j + 0.13, row[col], f"P{int(row['pid']):02d}",
                            fontsize=5, color="#222", va="center")
    ax.set_xticks(positions); ax.set_xticklabels(IT_ORDER, fontsize=6.5)
    ax.set_ylabel(ylab, fontsize=6.5)
    ax.set_title("", fontsize=7)
    try:
        valid = [d for d in data if len(d) >= 1]
        if len(valid) >= 2 and sum(len(d) for d in valid) >= 4:
            _, kwp = stats.kruskal(*valid)
        else: kwp = np.nan
    except Exception:
        kwp = np.nan
    if pd.notna(kwp): annotate_p(ax, kwp)

# --- binary variable panels (prevalence bar + Fisher) ---
for col, ylab, pos in BIN_PANELS:
    ax = fig.add_subplot(pos)
    prev = []
    n_per = []
    for it in IT_ORDER:
        sub = clin_all.loc[clin_all["immunotype"] == it, col].dropna()
        n_per.append(len(sub))
        prev.append(sub.mean() if len(sub) > 0 else np.nan)
    positions = np.arange(len(IT_ORDER))
    for j, it in enumerate(IT_ORDER):
        ax.bar(j, prev[j]*100, width=0.55,
               color=IT_PALETTE[it]+"AA", edgecolor=IT_PALETTE[it], linewidth=0.6)
        ax.text(j, prev[j]*100 + 3, f"{int(prev[j]*n_per[j])}/{n_per[j]}",
                ha="center", va="bottom", fontsize=6, color="#222")
    ax.set_xticks(positions); ax.set_xticklabels(IT_ORDER, fontsize=6.5)
    ax.set_ylabel(ylab + "\nprevalence (%)", fontsize=6.5)
    ax.set_ylim(0, 120)
    ax.set_yticks([0, 50, 100])
    # Fisher-exact for 3xk: use chi-square test with small-sample warning
    try:
        table = np.array([[int(prev[j]*n_per[j]), n_per[j]-int(prev[j]*n_per[j])]
                          for j in range(len(IT_ORDER))])
        chi2, p_chi, _, _ = stats.chi2_contingency(table.T, correction=False)
    except Exception:
        p_chi = np.nan
    if pd.notna(p_chi): annotate_p(ax, p_chi)

# add IT legend once
handles = [mpatches.Patch(color=IT_PALETTE[it], label=it) for it in IT_ORDER]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02),
           ncol=3, fontsize=7, frameon=False)

fig.suptitle("Immunotype stratification recapitulates clinical disease severity",
             fontsize=9, y=1.06)
save(fig, "23_immunotype_clinical_severity", OUT_DIR)
print("    Saved: 23_immunotype_clinical_severity.pdf/.png")

# also export a per-patient table
clin_all[["pid","name","antibody","immunotype",
          "age","myositis","arthritis","rash","ILD",
          "CK","LDH","ESR","CRP","FVC","DLCO","DA_score"]].to_csv(
    f"{OUT_DIR}/23_immunotype_clinical_table.csv", index=False)

# ════════════════════════════════════════════════════════════════════════════
# FIG 24 — Bootstrap stability of immunotype assignment (1000 iterations)
# ════════════════════════════════════════════════════════════════════════════
print("\n  Fig 24: Bootstrap stability of immunotype clustering …")

from sklearn.metrics import adjusted_rand_score

N_BOOT   = 1000
rng      = np.random.default_rng(42)
n_pat    = Xf.shape[0]                 # 10 patients
pair_co  = np.zeros((n_pat, n_pat))    # numerator: times co-clustered
pair_tot = np.zeros((n_pat, n_pat))    # denominator: times both sampled

for b in range(N_BOOT):
    idx = rng.integers(0, n_pat, size=n_pat)    # sample with replacement
    X_b = Xf[idx]
    # Ward clustering on bootstrap sample
    try:
        link_b = hierarchy.linkage(X_b, method="ward")
        clust_b = hierarchy.fcluster(link_b, t=k, criterion="maxclust")
    except Exception:
        continue
    # for each pair of distinct original patients that both appear in idx,
    # register whether they were clustered together on any of their bootstrap copies
    present = {i: np.where(idx == i)[0] for i in range(n_pat)}
    for i in range(n_pat):
        if len(present[i]) == 0: continue
        for j in range(i, n_pat):
            if len(present[j]) == 0: continue
            pair_tot[i, j] += 1; pair_tot[j, i] += 1
            # any co-cluster pair (i_copy, j_copy)?
            coc = False
            for ii in present[i]:
                for jj in present[j]:
                    if ii != jj and clust_b[ii] == clust_b[jj]:
                        coc = True
                        break
                    elif i == j and ii == jj:
                        coc = True
                        break
                if coc: break
            if coc or (i == j):
                pair_co[i, j] += 1; pair_co[j, i] += 1

co_mat = np.divide(pair_co, pair_tot,
                   out=np.zeros_like(pair_co), where=pair_tot > 0)
np.fill_diagonal(co_mat, 1.0)

# order by original IT assignment
labels = [f"P{int(meta_idx.loc[s,'pid']):02d}" for s in feat.index]
it_labels = [f"IT{g}" for g in groups]
order_idx = sorted(range(n_pat), key=lambda i: (it_labels[i], labels[i]))
co_mat_ord   = co_mat[np.ix_(order_idx, order_idx)]
labels_ord   = [labels[i] for i in order_idx]
it_labs_ord  = [it_labels[i] for i in order_idx]
ab_labs_ord  = [meta_idx.loc[feat.index[i], "antibody"] for i in order_idx]

# per-patient stability: mean co-cluster probability with other members of own IT
stab_per_pat = []
for i in range(n_pat):
    same = [j for j in range(n_pat) if j != i and it_labels[j] == it_labels[i]]
    if same:
        stab_per_pat.append(co_mat[i, same].mean())
    else:
        stab_per_pat.append(np.nan)
stab_df = pd.DataFrame({
    "pid": [int(meta_idx.loc[s,'pid']) for s in feat.index],
    "sample": feat.index,
    "antibody": [meta_idx.loc[s,"antibody"] for s in feat.index],
    "immunotype": it_labels,
    "within_IT_stability": stab_per_pat,
})
stab_df.to_csv(f"{OUT_DIR}/24_bootstrap_stability_per_patient.csv", index=False)

fig = plt.figure(figsize=(DC, 3.8))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.6, 1.0], wspace=0.35)

# (a) co-clustering matrix
ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(co_mat_ord, cmap=HEATMAP_SEQ, vmin=0, vmax=1, aspect="equal")
ax.set_xticks(range(n_pat)); ax.set_yticks(range(n_pat))
ax.set_xticklabels([f"{labels_ord[i]}\n{ab_labs_ord[i]}"
                    for i in range(n_pat)], fontsize=5.5, rotation=0)
ax.set_yticklabels([f"{labels_ord[i]} · {ab_labs_ord[i]} · {it_labs_ord[i]}"
                    for i in range(n_pat)], fontsize=5.5)
ax.tick_params(length=0)
# IT boundary lines
from itertools import groupby as _gb
counts = [len(list(g)) for _, g in _gb(it_labs_ord)]
cum = 0
for c in counts[:-1]:
    cum += c
    ax.axhline(cum - 0.5, color="#000", lw=0.7)
    ax.axvline(cum - 0.5, color="#000", lw=0.7)
# numeric co-cluster probabilities in cells
for i in range(n_pat):
    for j in range(n_pat):
        v = co_mat_ord[i, j]
        col = "white" if v > 0.55 else "#222"
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                fontsize=4.5, color=col)
cax = fig.add_axes([0.395, 0.16, 0.011, 0.25])
cb = fig.colorbar(im, cax=cax)
cb.set_label("Co-clustering frequency\n(1,000 bootstraps)", fontsize=5.5)
cb.ax.tick_params(labelsize=5)
ax.set_title("Bootstrap co-clustering matrix", fontsize=7.5, pad=4)

# (b) per-patient stability bar
ax2 = fig.add_subplot(gs[0, 1])
order_b = sorted(range(n_pat), key=lambda i: stab_per_pat[i])
y = np.arange(n_pat)
colors = [IT_PALETTE[it_labels[i]] for i in order_b]
ax2.barh(y, [stab_per_pat[i] for i in order_b], color=colors,
         edgecolor="white", linewidth=0.5, height=0.7)
ax2.set_yticks(y)
ax2.set_yticklabels([f"{labels[i]} · {it_labels[i]}" for i in order_b], fontsize=6)
ax2.set_xlim(0, 1)
ax2.set_xlabel("Mean co-cluster probability\nwith own immunotype", fontsize=6)
ax2.axvline(0.5, color="#888", ls="--", lw=0.5)
for i_rank, i in enumerate(order_b):
    ax2.text(stab_per_pat[i] + 0.01, i_rank,
             f"{stab_per_pat[i]:.2f}",
             va="center", fontsize=5.5, color="#333")
ax2.set_title("Per-patient stability", fontsize=7.5, pad=4)

fig.suptitle("Bootstrap validation of immunotype assignment "
             f"(n = 10 patients, B = {N_BOOT})", fontsize=8.5, y=1.01)
save(fig, "24_bootstrap_stability", OUT_DIR)
print("    Saved: 24_bootstrap_stability.pdf/.png")

# ════════════════════════════════════════════════════════════════════════════
# FIG 25 — Leave-one-out stability
# ════════════════════════════════════════════════════════════════════════════
print("\n  Fig 25: Leave-one-out immunotype stability …")

loo_records = []
loo_assign_matrix = {}   # for each left-out patient, store remaining patients' new cluster

orig_labels = np.array(it_labels)          # 10 labels, IT1/IT2/IT3
for i_out in range(n_pat):
    keep = [j for j in range(n_pat) if j != i_out]
    X_loo = Xf[keep]
    link_loo = hierarchy.linkage(X_loo, method="ward")
    clust_loo = hierarchy.fcluster(link_loo, t=k, criterion="maxclust")
    # align LOO labels to original labels by majority matching
    # --- build best permutation of LOO labels → ITn to maximise agreement
    orig_sub = orig_labels[keep]
    best_map = {}
    # iterate LOO cluster ids; for each, find the IT that dominates
    for c in np.unique(clust_loo):
        mask = clust_loo == c
        if mask.sum() == 0: continue
        best_it, _ = max(pd.Series(orig_sub[mask]).value_counts().items())
        best_map[c] = best_it
    loo_mapped = np.array([best_map.get(c, "IT?") for c in clust_loo])
    ari = adjusted_rand_score(orig_sub, clust_loo)
    agree = (loo_mapped == orig_sub).mean()
    loo_records.append({
        "left_out_pid": int(meta_idx.loc[feat.index[i_out], "pid"]),
        "left_out_sample": feat.index[i_out],
        "left_out_IT": orig_labels[i_out],
        "ARI": ari,
        "label_agreement": agree,
    })
    loo_assign_matrix[feat.index[i_out]] = dict(zip(
        [feat.index[j] for j in keep], loo_mapped))

loo_df = pd.DataFrame(loo_records)
loo_df.to_csv(f"{OUT_DIR}/25_loo_stability.csv", index=False)

# plot
fig = plt.figure(figsize=(DC, 3.0))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.0], wspace=0.35)

# (a) ARI per dropped patient
ax = fig.add_subplot(gs[0, 0])
ord_loo = loo_df.sort_values("ARI", ascending=True).reset_index(drop=True)
y = np.arange(len(ord_loo))
colors = [IT_PALETTE[it] for it in ord_loo["left_out_IT"]]
ax.barh(y, ord_loo["ARI"], color=colors,
        edgecolor="white", linewidth=0.4, height=0.7)
ax.set_yticks(y)
ax.set_yticklabels(
    [f"P{int(r.left_out_pid):02d} · {r.left_out_IT}"
     for r in ord_loo.itertuples(index=False)], fontsize=6)
ax.set_xlabel("Adjusted Rand Index\n(vs. full-cohort assignment)", fontsize=6)
ax.axvline(1.0, color="#000", lw=0.5, ls="--")
for i, v in enumerate(ord_loo["ARI"]):
    ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=5.5)
ax.set_xlim(0, 1.15)
ax.set_title("Clustering preserved after removing each patient",
             fontsize=7.5, pad=4)

# (b) fraction of remaining patients whose IT label is preserved
ax2 = fig.add_subplot(gs[0, 1])
ord_loo = loo_df.sort_values("label_agreement", ascending=True).reset_index(drop=True)
y = np.arange(len(ord_loo))
colors = [IT_PALETTE[it] for it in ord_loo["left_out_IT"]]
ax2.barh(y, ord_loo["label_agreement"] * 100, color=colors,
         edgecolor="white", linewidth=0.4, height=0.7)
ax2.set_yticks(y)
ax2.set_yticklabels(
    [f"P{int(r.left_out_pid):02d} · {r.left_out_IT}"
     for r in ord_loo.itertuples(index=False)], fontsize=6)
ax2.set_xlabel("% of remaining 9 patients\nretaining original IT label",
               fontsize=6)
ax2.axvline(100, color="#000", lw=0.5, ls="--")
ax2.set_xlim(0, 115)
for i, v in enumerate(ord_loo["label_agreement"]):
    ax2.text(v*100 + 1, i, f"{v*100:.0f}%", va="center", fontsize=5.5)
ax2.set_title("Label-level stability", fontsize=7.5, pad=4)

fig.suptitle("Leave-one-out validation of immunotype assignment",
             fontsize=8.5, y=1.03)
save(fig, "25_leave_one_out_stability", OUT_DIR)
print("    Saved: 25_leave_one_out_stability.pdf/.png")

# ════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY TABLE — full 65-dim feature matrix per patient
# ════════════════════════════════════════════════════════════════════════════
print("\n  Supp Table: full feature matrix per patient …")
feat_supp = feat.copy()
feat_supp.insert(0, "immunotype", it_labels)
feat_supp.insert(0, "antibody",
                 [meta_idx.loc[s, "antibody"] for s in feat_supp.index])
feat_supp.insert(0, "name",
                 [meta_idx.loc[s, "name"] for s in feat_supp.index])
feat_supp.insert(0, "pid",
                 [int(meta_idx.loc[s, "pid"]) for s in feat_supp.index])
feat_supp.index.name = "sample"
feat_supp.to_csv(f"{OUT_DIR}/SuppTable_feature_matrix.csv")
print(f"    Saved: SuppTable_feature_matrix.csv "
      f"({feat_supp.shape[0]} patients × {feat_supp.shape[1]-4} features)")

print(f"\n[Bootstrap] summary: "
      f"mean within-IT stability = {np.nanmean(stab_per_pat):.2f}, "
      f"min = {np.nanmin(stab_per_pat):.2f} "
      f"({labels[int(np.nanargmin(stab_per_pat))]})")
print(f"[LOO]       summary: mean ARI = {loo_df['ARI'].mean():.2f}, "
      f"mean label-agreement = {loo_df['label_agreement'].mean()*100:.0f}%")

print(f"""
══════════════════════════════════════════════════════════
  Advanced analyses complete.  New figures in: {OUT_DIR}/
    15_clinical_immune_heatmap
    16_patient_pca_mds
    17_clinical_correlation
    18_celltype_covariation
    19_immune_signatures
    20_percelltype_volcano_EJvsPL7
    21_immunotype_discovery
    22_celltype_ratios
    23_immunotype_clinical_severity  ← clinical validation
    24_bootstrap_stability           ← reviewer-proof stability
    25_leave_one_out_stability       ← reviewer-proof stability
  + SuppTable_feature_matrix.csv
══════════════════════════════════════════════════════════
""")
