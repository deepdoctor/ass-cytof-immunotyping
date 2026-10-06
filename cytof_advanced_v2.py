"""
cytof_advanced_v2.py
====================
Additional high-end analyses for JIF>8 submission — Part II:

  26 – FlowSOM + Phenograph robustness check vs Leiden clustering
  27 – Bootstrap 95% CI for cell-type abundance and DE marker effect sizes
  28 – Patient-level marker co-expression modules (WGCNA-style)
  29 – T cell diffusion map + PAGA pseudotime trajectory

Run:
    cd /Users/yichen/Desktop/collaboration/raw
    /Users/yichen/anaconda3/bin/conda run -n cytof python cytof_advanced_v2.py
"""

import os, sys, warnings, multiprocessing as mp
warnings.filterwarnings("ignore")

# Force fork start method on macOS so that phenograph's multiprocessing.Pool
# does NOT re-execute the whole script in child workers (which otherwise
# causes infinite re-import recursion under `python -u …`).
try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import anndata as ad
import scanpy as sc

from nature_style import (
    apply_nature_style, GROUP_PALETTE, CLUSTER_PALETTE,
    HEATMAP_DIVERG, HEATMAP_SEQ, save
)
apply_nature_style()
SC, DC = 3.46, 7.09

# ════════════════════════════════════════════════════════════════════════════
# CONFIG — same as cytof_advanced_analysis.py
# ════════════════════════════════════════════════════════════════════════════
H5AD_PATH = "./cytof_output_v3/cytof_analyzed_v3.h5ad"
OUT_DIR   = "./cytof_output_nature"

CLUSTER_ANNOTATION = {
    "0":  "Classical Monocyte", "1":  "Naive B", "2":  "CD8 Effector T",
    "3":  "CD4 Naive T", "4":  "CD4 Central Memory T",
    "5":  "Non-classical Monocyte", "6":  "NK cell", "7":  "CD4 Th1-like",
    "8":  "Classical Monocyte", "9":  "CD4 Effector Memory T",
    "10": "Inflammatory Monocyte", "11": "CD8 Naive T",
    "12": "γδ T cell", "13": "mDC", "14": "pDC",
    "15": "T-Myeloid doublets", "16": "CD16+ Granulocyte",
}
DROP_CELLTYPES = {"T-Myeloid doublets", "CD16+ Granulocyte"}

PATIENT_META = pd.DataFrame([
    ["L01912_FH0002",   8, "P08", "Jo-1",  57,   1,0,0,1, 3838, 755, 11, 1.71, np.nan, np.nan],
    ["L01912_FH0003",   7, "P07", "Jo-1",  52,   1,1,0,1, 469, 343, 30, 0.73, 39.4, 27.9 ],
    ["L01912_FH0009",   6, "P06", "PL-12", 50,   1,0,0,1, 26, 202, np.nan, 0.25, np.nan, np.nan],
    ["L01912_FH0011",   5, "P05", "EJ",    67,   1,0,0,1, 26, 147, 2, np.nan, np.nan, np.nan],
    ["L01912_FH0012",   4, "P04",   "PL-7",  63,   0,1,0,1, 41, 195, 7, np.nan, 86.5, 77.3 ],
    ["L01912_FH0013",   3, "P03", "PL-12", 67,   1,1,1,0, 13, 318, 13, 4.6, np.nan, np.nan],
    ["L01912_FH0014",   2, "P02", "Jo-1",  60,   1,0,0,1, np.nan, 374, 6, np.nan, np.nan, np.nan],
    ["L01912_FH0015",   1, "P01", "PL-12", 83,   0,0,0,1, 43, 315, 2, 0.25, np.nan, np.nan],
    ["L01912_FH0016",  10, "P10", "EJ",    66,   1,1,0,1, np.nan, np.nan, 28, np.nan, np.nan, np.nan],
    ["L01912_FH0017",   9, "P09", "PL-7",  52,   0,0,1,1, 15, np.nan, 13, 0.25, 65, 46.1 ],
], columns=["sample","pid","name","antibody","age","myositis","arthritis","rash",
            "ILD","CK","LDH","ESR","CRP","FVC","DLCO"])
meta_idx = PATIENT_META.set_index("sample")
GROUP_ORDER = ["Jo-1", "PL-12", "EJ", "PL-7"]

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
adata = adata[~adata.obs["cell_type"].isin(DROP_CELLTYPES)].copy()
print(f"  {adata.n_obs:,} cells × {adata.n_vars} markers × "
      f"{adata.obs['cell_type'].nunique()} cell types")

os.makedirs(OUT_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# FIG 26 — FlowSOM + Phenograph robustness check
# ════════════════════════════════════════════════════════════════════════════
print("\n  Fig 26: FlowSOM + Phenograph robustness check …")

import phenograph
from minisom import MiniSom

X = adata.obsm["X_pca_harmony"]
leiden_labels = adata.obs["leiden"].astype(str).values

# --- Phenograph ---
print("    Running Phenograph (k=30)…")
pg_labels, _, _ = phenograph.cluster(X, k=30, n_jobs=1, seed=0)
print(f"    Phenograph → {len(np.unique(pg_labels))} clusters")

# --- FlowSOM via MiniSom + meta-clustering ---
print("    Running FlowSOM (SOM 10×10 + meta-clustering)…")
som = MiniSom(10, 10, X.shape[1], sigma=1.0, learning_rate=0.5,
              neighborhood_function="gaussian", random_seed=0)
som.random_weights_init(X)
som.train(X, 30000, verbose=False)
som_nodes = np.array([som.winner(x) for x in X])      # (N, 2)
som_id    = som_nodes[:,0]*10 + som_nodes[:,1]        # 0..99

# meta-cluster SOM nodes by hierarchical clustering of their prototype vectors
weights = som.get_weights().reshape(100, -1)           # 100 nodes × 20 PCs
link_nodes = hierarchy.linkage(weights, method="average")
k_meta = 17                                            # match Leiden cluster count
node_meta = hierarchy.fcluster(link_nodes, t=k_meta, criterion="maxclust")
flowsom_labels = node_meta[som_id].astype(str)
print(f"    FlowSOM → {len(np.unique(flowsom_labels))} meta-clusters")

# --- confusion matrices ---
def confusion(a, b):
    cats_a = sorted(np.unique(a), key=lambda x: int(x) if str(x).isdigit() else x)
    cats_b = sorted(np.unique(b), key=lambda x: int(x) if str(x).isdigit() else x)
    M = pd.DataFrame(0, index=cats_a, columns=cats_b, dtype=float)
    for ai, bi in zip(a, b):
        M.loc[ai, bi] += 1
    # row-normalise
    M = M.div(M.sum(axis=1).replace(0, 1), axis=0)
    return M

conf_pg = confusion(leiden_labels, pg_labels.astype(str))
conf_fs = confusion(leiden_labels, flowsom_labels)

# ARI / NMI
ari_pg = adjusted_rand_score(leiden_labels, pg_labels)
nmi_pg = normalized_mutual_info_score(leiden_labels, pg_labels)
ari_fs = adjusted_rand_score(leiden_labels, flowsom_labels)
nmi_fs = normalized_mutual_info_score(leiden_labels, flowsom_labels)
print(f"    Leiden × Phenograph  ARI={ari_pg:.2f}, NMI={nmi_pg:.2f}")
print(f"    Leiden × FlowSOM      ARI={ari_fs:.2f}, NMI={nmi_fs:.2f}")

# --- reorder columns to maximise diagonal for display ---
def reorder_confusion(M):
    # Hungarian-style reordering: for each row, place argmax as close as possible to diagonal
    col_order = []
    used = set()
    for i, row in enumerate(M.index):
        ranked = M.loc[row].sort_values(ascending=False).index.tolist()
        for c in ranked:
            if c not in used:
                col_order.append(c); used.add(c)
                break
    # append remaining columns
    for c in M.columns:
        if c not in used:
            col_order.append(c)
    return M[col_order]

conf_pg_ord = reorder_confusion(conf_pg)
conf_fs_ord = reorder_confusion(conf_fs)

fig = plt.figure(figsize=(DC+0.5, 5.0))
gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.32, hspace=0.55,
                       height_ratios=[1.0, 1.0])

def plot_confusion(ax, M, title, metric_txt):
    im = ax.imshow(M.values, aspect="auto", cmap=HEATMAP_SEQ, vmin=0, vmax=1)
    ax.set_xticks(range(M.shape[1])); ax.set_yticks(range(M.shape[0]))
    ax.set_xticklabels(M.columns, fontsize=5, rotation=0)
    # annotate Leiden cluster with cell type
    yt_labels = [f"{c} · {CLUSTER_ANNOTATION.get(c, '?')[:18]}" for c in M.index]
    ax.set_yticklabels(yt_labels, fontsize=5.5)
    ax.tick_params(length=0)
    # value annotations only on >0.3
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.iloc[i, j]
            if v >= 0.3:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=4, color=("white" if v > 0.55 else "#222"))
    ax.set_title(title + "\n" + metric_txt, fontsize=7, pad=3)
    return im

ax1 = fig.add_subplot(gs[0, 0])
im1 = plot_confusion(ax1, conf_pg_ord,
                     "Leiden (rows) × Phenograph (cols)",
                     f"ARI = {ari_pg:.2f}   NMI = {nmi_pg:.2f}")
ax1.set_xlabel("Phenograph cluster", fontsize=6)

ax2 = fig.add_subplot(gs[0, 1])
im2 = plot_confusion(ax2, conf_fs_ord,
                     "Leiden (rows) × FlowSOM (cols)",
                     f"ARI = {ari_fs:.2f}   NMI = {nmi_fs:.2f}")
ax2.set_xlabel("FlowSOM meta-cluster", fontsize=6)

# colorbar
cax = fig.add_axes([0.92, 0.58, 0.011, 0.26])
cb = fig.colorbar(im1, cax=cax)
cb.set_label("Row-normalised fraction", fontsize=6)
cb.ax.tick_params(labelsize=5)

# --- UMAP panels coloured by alternative clusterings ---
ax3 = fig.add_subplot(gs[1, 0])
Z = adata.obsm["X_umap"]
# subsample for speed
idx = np.random.default_rng(0).choice(adata.n_obs, size=min(20000, adata.n_obs),
                                       replace=False)
pg_cmap = plt.cm.get_cmap("tab20", len(np.unique(pg_labels)))
pg_col = pg_cmap(pg_labels[idx].astype(int) % pg_cmap.N)
ax3.scatter(Z[idx,0], Z[idx,1], s=0.3, c=pg_col, alpha=0.5, edgecolor="none")
ax3.set_xticks([]); ax3.set_yticks([])
ax3.set_xlabel("UMAP-1", fontsize=6); ax3.set_ylabel("UMAP-2", fontsize=6)
ax3.set_title("UMAP · Phenograph clusters", fontsize=7)

ax4 = fig.add_subplot(gs[1, 1])
fs_unique = sorted(np.unique(flowsom_labels), key=int)
fs_map = {c: i for i, c in enumerate(fs_unique)}
fs_idx = np.array([fs_map[c] for c in flowsom_labels])
fs_cmap = plt.cm.get_cmap("tab20", len(fs_unique))
fs_col = fs_cmap(fs_idx[idx] % fs_cmap.N)
ax4.scatter(Z[idx,0], Z[idx,1], s=0.3, c=fs_col, alpha=0.5, edgecolor="none")
ax4.set_xticks([]); ax4.set_yticks([])
ax4.set_xlabel("UMAP-1", fontsize=6); ax4.set_ylabel("UMAP-2", fontsize=6)
ax4.set_title("UMAP · FlowSOM clusters", fontsize=7)

fig.suptitle("Robustness of Leiden clustering: cross-validation with Phenograph and FlowSOM",
             fontsize=8.5, y=0.99)
save(fig, "26_clustering_robustness", OUT_DIR)
print("    Saved: 26_clustering_robustness.pdf/.png")

# save tables
pd.DataFrame({"metric":["ARI","NMI"],
              "Leiden_vs_Phenograph":[ari_pg,nmi_pg],
              "Leiden_vs_FlowSOM":   [ari_fs,nmi_fs]}).to_csv(
    f"{OUT_DIR}/26_clustering_robustness_metrics.csv", index=False)

# ════════════════════════════════════════════════════════════════════════════
# FIG 27 — Bootstrap 95% CI for cell abundances and DE markers
# ════════════════════════════════════════════════════════════════════════════
print("\n  Fig 27: Bootstrap 95% CI for cell abundances and key markers …")

# --- per-sample cell-type proportions ---
prop_long = (
    adata.obs.groupby(["sample", "cell_type"], observed=True).size()
    .groupby(level=0).transform(lambda x: x/x.sum())
    .rename("prop").reset_index()
)
prop_wide = prop_long.pivot(index="sample", columns="cell_type", values="prop").fillna(0)*100
prop_wide = prop_wide.loc[meta_idx.index]

# bootstrap samples within each group, B=2000
B = 2000
rng = np.random.default_rng(42)
ab_series = meta_idx["antibody"]

abundance_ci = []
for ct in prop_wide.columns:
    for g in GROUP_ORDER:
        gs_samps = [s for s in prop_wide.index if ab_series[s] == g]
        vals = prop_wide.loc[gs_samps, ct].values
        if len(vals) == 0: continue
        boots = np.array([rng.choice(vals, size=len(vals), replace=True).mean()
                          for _ in range(B)])
        abundance_ci.append({
            "cell_type": ct, "group": g, "n": len(vals),
            "mean": vals.mean(),
            "ci_low":  np.percentile(boots, 2.5),
            "ci_high": np.percentile(boots, 97.5),
        })
ab_ci = pd.DataFrame(abundance_ci)
ab_ci.to_csv(f"{OUT_DIR}/27_abundance_bootstrap_CI.csv", index=False)

# forest plot — 2 columns of small multiples
ct_order = (ab_ci.groupby("cell_type")["mean"].max()
                 .sort_values(ascending=False).index.tolist())

fig = plt.figure(figsize=(DC+0.5, 5.6))
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.45)

for ci_col, ct_sub in enumerate([ct_order[:len(ct_order)//2],
                                  ct_order[len(ct_order)//2:]]):
    ax = fig.add_subplot(gs[0, ci_col])
    y_pos = []
    labels = []
    for i, ct in enumerate(ct_sub):
        for j, g in enumerate(GROUP_ORDER):
            row = ab_ci[(ab_ci["cell_type"] == ct) & (ab_ci["group"] == g)]
            if row.empty: continue
            r = row.iloc[0]
            y = -(i*5 + j)        # stack groups within cell type, inverted
            ax.plot([r["ci_low"], r["ci_high"]], [y, y],
                    color=GROUP_PALETTE[g], lw=1.3)
            ax.plot(r["mean"], y, "o", color=GROUP_PALETTE[g],
                    markersize=3.5, markeredgecolor="white", markeredgewidth=0.4)
            if j == 0: labels.append((y - 1.5, ct))
    ax.set_yticks([l[0] for l in labels])
    ax.set_yticklabels([l[1] for l in labels], fontsize=6)
    ax.set_xlabel("Proportion (%)", fontsize=6)
    ax.axvline(0, color="#ddd", lw=0.4)
    ax.grid(axis="x", color="#eee", lw=0.3)
    ax.set_axisbelow(True)
    ax.set_xlim(left=0)

# legend
handles = [mpatches.Patch(color=GROUP_PALETTE[g], label=g) for g in GROUP_ORDER]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.01),
           ncol=4, fontsize=6.5, frameon=False)
fig.suptitle(f"Cell-type abundance with bootstrap 95% CI (B = {B})",
             fontsize=8.5, y=1.04)
save(fig, "27a_abundance_bootstrap_forest", OUT_DIR)
print("    Saved: 27a_abundance_bootstrap_forest.pdf/.png")

# --- part (b): bootstrap CI on key DE marker effect sizes (Cohen's d) for EJ vs PL-7 ---
print("    Computing bootstrap CI for EJ vs PL-7 marker effect sizes …")
# focus on a curated list of key markers across 6 principal cell types
KEY_TESTS = [
    ("CD4 Central Memory T", "GranzymeB"),
    ("CD4 Central Memory T", "HLA-DR"),
    ("CD4 Naive T",          "CD27"),
    ("CD4 Naive T",          "CXCR3"),
    ("CD8 Effector T",       "CD38"),
    ("CD8 Effector T",       "HLA-DR"),
    ("NK cell",              "HLA-DR"),
    ("NK cell",              "GranzymeB"),
    ("Classical Monocyte",   "CD69"),
    ("Classical Monocyte",   "HLA-DR"),
    ("Naive B",              "CXCR5"),
    ("mDC",                  "CD11b"),
]
gA, gB = "EJ", "PL-7"

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2: return np.nan
    s2 = ((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx+ny-2)
    sd = np.sqrt(s2)
    return (np.mean(x) - np.mean(y)) / sd if sd > 0 else np.nan

records = []
X_mat = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
for ct, m in KEY_TESTS:
    cells = adata[(adata.obs["cell_type"] == ct) &
                  (adata.obs["group"].isin([gA, gB]))]
    vals = X_mat.loc[cells.obs_names, m].values
    grp  = cells.obs["group"].values
    xA = vals[grp == gA]; xB = vals[grp == gB]
    if len(xA) < 10 or len(xB) < 10: continue
    d_pt = cohens_d(xA, xB)
    # bootstrap cells (stratified)
    boots = []
    for _ in range(B):
        bA = rng.choice(xA, size=len(xA), replace=True)
        bB = rng.choice(xB, size=len(xB), replace=True)
        boots.append(cohens_d(bA, bB))
    records.append({
        "cell_type": ct, "marker": m, "d": d_pt,
        "ci_low":  np.percentile(boots, 2.5),
        "ci_high": np.percentile(boots, 97.5),
        "crosses_zero": not ((np.percentile(boots, 2.5) > 0) or
                              (np.percentile(boots, 97.5) < 0)),
    })

d_df = pd.DataFrame(records)
d_df.to_csv(f"{OUT_DIR}/27b_DE_bootstrap_CI.csv", index=False)
d_df = d_df.sort_values("d").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(DC-0.5, 3.5))
y = np.arange(len(d_df))
for i, r in d_df.iterrows():
    c = GROUP_PALETTE[gA] if r["d"] > 0 else GROUP_PALETTE[gB]
    ax.plot([r["ci_low"], r["ci_high"]], [i, i], color=c,
            lw=1.4, alpha=0.9 if not r["crosses_zero"] else 0.5)
    ax.plot(r["d"], i, "o", color=c, markersize=4.5,
            markeredgecolor="white", markeredgewidth=0.5)
ax.axvline(0, color="#333", lw=0.6)
ax.set_yticks(y)
ax.set_yticklabels([f"{r['marker']} · {r['cell_type']}"
                    for _, r in d_df.iterrows()], fontsize=6)
ax.set_xlabel(r"Cohen's d  (EJ − PL-7)", fontsize=7)
ax.set_title(f"Bootstrap 95% CI of EJ vs PL-7 effect size (B = {B})",
             fontsize=8, pad=4)
# annotate direction
ax.text(0.02, 0.98, "↑ in PL-7", transform=ax.transAxes,
        va="top", fontsize=6, color=GROUP_PALETTE[gB])
ax.text(0.98, 0.98, "↑ in EJ", transform=ax.transAxes,
        va="top", ha="right", fontsize=6, color=GROUP_PALETTE[gA])
save(fig, "27b_DE_bootstrap_CI_forest", OUT_DIR)
print("    Saved: 27b_DE_bootstrap_CI_forest.pdf/.png")

# ════════════════════════════════════════════════════════════════════════════
# FIG 28 — Patient-level marker co-expression modules
# ════════════════════════════════════════════════════════════════════════════
print("\n  Fig 28: Patient-level marker co-expression modules …")

# per-sample global marker mean
GLOBAL = (
    pd.DataFrame(adata.X, columns=adata.var_names)
    .assign(sample=adata.obs["sample"].values)
    .groupby("sample").mean().loc[meta_idx.index]
)

# Spearman correlation of markers across patients
rho = GLOBAL.corr(method="spearman")

# hierarchical clustering on markers
link_m = hierarchy.linkage(pdist(rho.values, metric="correlation"), method="average")
# choose k_modules by cutting at a reasonable distance
k_mod = 6
mod_ids = hierarchy.fcluster(link_m, t=k_mod, criterion="maxclust")
mk_order = hierarchy.leaves_list(link_m)
rho_ord = rho.iloc[mk_order, mk_order]
mod_ord = mod_ids[mk_order]

# build module dataframe
module_df = pd.DataFrame({"marker": rho.index,
                          "module": [f"M{m}" for m in mod_ids]})
module_df.to_csv(f"{OUT_DIR}/28_marker_modules.csv", index=False)

fig = plt.figure(figsize=(DC+0.5, 5.0))
gs = gridspec.GridSpec(2, 2, figure=fig,
                       width_ratios=[3.8, 2.2], height_ratios=[5.0, 2.0],
                       wspace=0.35, hspace=0.45)

# (a) correlation heatmap with module bars
ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(rho_ord.values, aspect="auto", cmap=HEATMAP_DIVERG,
               vmin=-1, vmax=1)
ax.set_xticks(range(len(rho_ord)))
ax.set_xticklabels(rho_ord.columns, rotation=60, ha="right", fontsize=5)
ax.set_yticks(range(len(rho_ord)))
ax.set_yticklabels(rho_ord.index, fontsize=5)
ax.tick_params(length=0)

# module boundaries
module_colors = plt.cm.get_cmap("tab10", k_mod)
cum = 0
boundaries = []
for m in sorted(np.unique(mod_ord), key=lambda x: list(mod_ord).index(x)):
    idx = np.where(mod_ord == m)[0]
    if len(idx) == 0: continue
    start, end = idx.min(), idx.max()
    ax.add_patch(plt.Rectangle((start-0.5, start-0.5), end-start+1, end-start+1,
                               fill=False, edgecolor="black", linewidth=1.0))
    boundaries.append((m, start, end))

cax = fig.add_axes([0.62, 0.58, 0.012, 0.25])
cb = fig.colorbar(im, cax=cax)
cb.set_label("Spearman ρ", fontsize=6)
cb.ax.tick_params(labelsize=5)

ax.set_title("Patient-level marker co-expression (Spearman)",
             fontsize=7.5, pad=3)

# (b) module membership table
ax2 = fig.add_subplot(gs[0, 1])
ax2.axis("off")
# display modules
y_pos = 0.98
ax2.text(0, y_pos, "Marker modules (M1–M6):",
         fontsize=7.5, fontweight="bold", transform=ax2.transAxes)
y_pos -= 0.07
mod_palette = {}
for k, m in enumerate(sorted(np.unique(mod_ids))):
    mks_in_mod = module_df[module_df["module"] == f"M{m}"]["marker"].tolist()
    col = matplotlib.colors.to_hex(module_colors(k % k_mod))
    mod_palette[f"M{m}"] = col
    ax2.text(0, y_pos, f"M{m} (n={len(mks_in_mod)}):",
             fontsize=6.5, color=col, fontweight="bold",
             transform=ax2.transAxes)
    y_pos -= 0.05
    ax2.text(0.02, y_pos, ", ".join(mks_in_mod[:10]) +
             ("…" if len(mks_in_mod) > 10 else ""),
             fontsize=5.8, color="#222", transform=ax2.transAxes,
             wrap=True)
    y_pos -= 0.10

# (c) module score × group heatmap
ax3 = fig.add_subplot(gs[1, :])
mod_scores = pd.DataFrame(index=meta_idx.index)
for m in sorted(np.unique(mod_ids)):
    mks = module_df[module_df["module"] == f"M{m}"]["marker"].tolist()
    mks = [x for x in mks if x in GLOBAL.columns]
    if mks: mod_scores[f"M{m}"] = GLOBAL[mks].mean(axis=1)

# z-score within modules across samples
mod_scores_z = (mod_scores - mod_scores.mean()) / mod_scores.std(ddof=0)
# group means
group_mod = pd.DataFrame(index=GROUP_ORDER, columns=mod_scores_z.columns, dtype=float)
for g in GROUP_ORDER:
    ss = [s for s in meta_idx.index if meta_idx.loc[s,"antibody"] == g]
    group_mod.loc[g] = mod_scores_z.loc[ss].mean()

im = ax3.imshow(group_mod.values, aspect="auto", cmap=HEATMAP_DIVERG,
                vmin=-1.2, vmax=1.2)
ax3.set_xticks(range(group_mod.shape[1]))
ax3.set_xticklabels(group_mod.columns, fontsize=6)
ax3.set_yticks(range(group_mod.shape[0]))
ax3.set_yticklabels(group_mod.index, fontsize=6)
ax3.tick_params(length=0)
for i in range(group_mod.shape[0]):
    for j in range(group_mod.shape[1]):
        v = group_mod.iloc[i, j]
        col = "white" if abs(v) > 0.6 else "#111"
        ax3.text(j, i, f"{v:.2f}", ha="center", va="center",
                 fontsize=5.5, color=col)
ax3.set_title("Module z-score by autoantibody group", fontsize=7.5, pad=3)
cax2 = fig.add_axes([0.92, 0.13, 0.012, 0.14])
cb2 = fig.colorbar(im, cax=cax2)
cb2.set_label("Mean z-score", fontsize=6)
cb2.ax.tick_params(labelsize=5)

fig.suptitle("Data-driven marker co-expression modules recapitulate canonical\nimmune functional programs",
             fontsize=8.5, y=1.03)
save(fig, "28_marker_coexpression_modules", OUT_DIR)
print("    Saved: 28_marker_coexpression_modules.pdf/.png")

# ════════════════════════════════════════════════════════════════════════════
# FIG 29 — T cell diffusion map + PAGA pseudotime
# ════════════════════════════════════════════════════════════════════════════
print("\n  Fig 29: T cell diffusion map + PAGA pseudotime …")

T_CELLS = [
    "CD4 Naive T", "CD4 Central Memory T", "CD4 Effector Memory T",
    "CD4 Th1-like", "CD8 Naive T", "CD8 Effector T", "γδ T cell",
]
tdata = adata[adata.obs["cell_type"].isin(T_CELLS)].copy()
tdata.obs["cell_type"] = pd.Categorical(tdata.obs["cell_type"], categories=T_CELLS)
print(f"    T cell subset: {tdata.n_obs:,} cells")

# use harmony PCs directly; compute neighbors within T cell subset
sc.pp.neighbors(tdata, use_rep="X_pca_harmony", n_neighbors=30, random_state=0)
sc.tl.diffmap(tdata, n_comps=15)

# PAGA on cell_type
sc.tl.paga(tdata, groups="cell_type")

# Root = CD4 Naive T, which is the canonical "start" of CD4 differentiation.
# Use cell at the extreme of DC1 within CD4 Naive T as root
naive_mask = (tdata.obs["cell_type"] == "CD4 Naive T").values
dc1 = tdata.obsm["X_diffmap"][:, 1]
# pick direction so that naive cells are on the extreme; use the argmax/min in naive
if dc1[naive_mask].mean() > dc1.mean():
    tdata.uns["iroot"] = int(np.argmax(dc1[naive_mask]) + np.where(naive_mask)[0][0])
else:
    tdata.uns["iroot"] = np.where(naive_mask)[0][int(np.argmin(dc1[naive_mask]))]

sc.tl.dpt(tdata, n_branchings=0)

# also recompute UMAP within T subset (optional for visualisation)
sc.tl.umap(tdata, random_state=0)

# --- plotting ---
fig = plt.figure(figsize=(DC+0.5, 5.4))
gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.32, hspace=0.35,
                       height_ratios=[1.0, 1.0])

TCELL_PAL = {
    "CD4 Naive T":          "#E64B35",
    "CD4 Central Memory T": "#4DBBD5",
    "CD4 Effector Memory T":"#00A087",
    "CD4 Th1-like":         "#F39B7F",
    "CD8 Naive T":          "#3C5488",
    "CD8 Effector T":       "#8491B4",
    "γδ T cell":            "#7E6148",
}
Z_umap = tdata.obsm["X_umap"]
Z_dmap = tdata.obsm["X_diffmap"][:, 1:3]       # components 1 & 2
dpt    = tdata.obs["dpt_pseudotime"].values

idx = np.random.default_rng(0).choice(tdata.n_obs,
        size=min(15000, tdata.n_obs), replace=False)

# (a) UMAP by cell type
ax = fig.add_subplot(gs[0, 0])
for ct in T_CELLS:
    m = (tdata.obs["cell_type"].values == ct)
    mi = idx[m[idx]]
    ax.scatter(Z_umap[mi,0], Z_umap[mi,1], s=1.0,
               c=TCELL_PAL[ct], alpha=0.5, edgecolor="none", label=ct)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("UMAP-1", fontsize=6); ax.set_ylabel("UMAP-2", fontsize=6)
ax.set_title("T cell UMAP", fontsize=7.5)
ax.legend(fontsize=5, markerscale=4, loc="upper right",
          frameon=False, handlelength=1, handletextpad=0.2)

# (b) Diffusion map DC1 vs DC2 by cell type
ax = fig.add_subplot(gs[0, 1])
for ct in T_CELLS:
    m = (tdata.obs["cell_type"].values == ct)
    mi = idx[m[idx]]
    ax.scatter(Z_dmap[mi,0], Z_dmap[mi,1], s=1.0,
               c=TCELL_PAL[ct], alpha=0.5, edgecolor="none")
ax.set_xlabel("DC1", fontsize=6); ax.set_ylabel("DC2", fontsize=6)
ax.set_title("Diffusion map · cell type", fontsize=7.5)
ax.set_xticks([]); ax.set_yticks([])

# (c) Diffusion map coloured by pseudotime
ax = fig.add_subplot(gs[0, 2])
sc_im = ax.scatter(Z_dmap[idx,0], Z_dmap[idx,1], s=1.0,
                    c=dpt[idx], cmap="plasma",
                    alpha=0.55, edgecolor="none")
ax.set_xlabel("DC1", fontsize=6); ax.set_ylabel("DC2", fontsize=6)
ax.set_title("Diffusion map · pseudotime", fontsize=7.5)
ax.set_xticks([]); ax.set_yticks([])
cax = fig.add_axes([0.92, 0.58, 0.011, 0.22])
cb = fig.colorbar(sc_im, cax=cax)
cb.set_label("DPT pseudotime", fontsize=6)
cb.ax.tick_params(labelsize=5)

# (d) PAGA graph
ax = fig.add_subplot(gs[1, 0:2])
import networkx as nx
conn = tdata.uns["paga"]["connectivities"].toarray()
group_names = tdata.obs["cell_type"].cat.categories.tolist()
G = nx.Graph()
for i, n in enumerate(group_names):
    G.add_node(i, name=n)
for i in range(len(group_names)):
    for j in range(i+1, len(group_names)):
        if conn[i, j] > 0.05:
            G.add_edge(i, j, weight=float(conn[i, j]))
pos_dict = nx.spring_layout(G, seed=0, k=1.2)
pos = np.array([pos_dict[i] for i in range(len(group_names))])
# edges
for i in range(len(group_names)):
    for j in range(i+1, len(group_names)):
        w = conn[i, j]
        if w > 0.05:
            ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]],
                    color="#888", lw=0.3+3*w, alpha=0.7, zorder=1)
# nodes sized by abundance
sizes = tdata.obs["cell_type"].value_counts().reindex(group_names).values
sizes_sc = 30 + sizes / sizes.max() * 400
for i, ct in enumerate(group_names):
    ax.scatter(pos[i,0], pos[i,1], s=sizes_sc[i],
               c=TCELL_PAL[ct], edgecolor="black", linewidth=0.6, zorder=3)
    ax.text(pos[i,0], pos[i,1]+0.08, ct, ha="center", fontsize=5.5,
            color="#222", zorder=4)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("PAGA connectivity graph",
             fontsize=7.5, pad=3)
# remove axis lines
for spine in ax.spines.values():
    spine.set_visible(False)

# (e) Pseudotime distribution by cell type (ridgeline approximated with violin)
ax = fig.add_subplot(gs[1, 2])
parts_data = []
parts_labels = []
for ct in T_CELLS[::-1]:
    m = (tdata.obs["cell_type"].values == ct)
    parts_data.append(dpt[m])
    parts_labels.append(ct)
parts = ax.violinplot(parts_data, positions=np.arange(len(parts_labels)),
                      vert=False, widths=0.9, showmeans=False, showmedians=True)
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(TCELL_PAL[parts_labels[i]] + "AA")
    pc.set_edgecolor(TCELL_PAL[parts_labels[i]])
    pc.set_linewidth(0.6)
for partname in ["cmedians","cbars","cmins","cmaxes"]:
    if partname in parts:
        parts[partname].set_color("#333"); parts[partname].set_lw(0.5)
ax.set_yticks(range(len(parts_labels)))
ax.set_yticklabels(parts_labels, fontsize=6)
ax.set_xlabel("DPT pseudotime", fontsize=6)
ax.set_title("Pseudotime by cell type", fontsize=7.5)

fig.suptitle("T cell differentiation trajectory: diffusion map + PAGA + pseudotime",
             fontsize=8.5, y=1.00)
save(fig, "29_T_cell_pseudotime", OUT_DIR)
print("    Saved: 29_T_cell_pseudotime.pdf/.png")

# save pseudotime distribution summary
pt_summary = pd.DataFrame({
    "cell_type": T_CELLS,
    "n_cells":   [int((tdata.obs["cell_type"] == ct).sum()) for ct in T_CELLS],
    "mean_pseudotime": [float(dpt[tdata.obs["cell_type"] == ct].mean()) for ct in T_CELLS],
    "median_pseudotime": [float(np.median(dpt[tdata.obs["cell_type"] == ct])) for ct in T_CELLS],
})
pt_summary.to_csv(f"{OUT_DIR}/29_pseudotime_summary.csv", index=False)

print(f"""
══════════════════════════════════════════════════════════
  Advanced v2 analyses complete.  New figures in: {OUT_DIR}/
    26_clustering_robustness           (Leiden × Phenograph / FlowSOM)
    27a_abundance_bootstrap_forest     (cell abundance 95% CI)
    27b_DE_bootstrap_CI_forest         (DE marker effect size 95% CI)
    28_marker_coexpression_modules     (data-driven functional modules)
    29_T_cell_pseudotime               (PAGA + DPT)
══════════════════════════════════════════════════════════
""")
