"""
cytof_advanced_v2_fix.py
========================
Targeted fixes for three reviewer-risk figures in cytof_advanced_v2.py
required for IF>8 submission:

  27b – DE effect size with PATIENT-LEVEL (cluster) bootstrap,
        symmetric top-hits in both directions, BH-FDR.
  28  – Marker-module figure with clean, non-overlapping module legend
        (textwrap + dynamic y-advance).
  29  – T-cell trajectory with γδ removed from the diffusion/PAGA
        analysis (shown as off-trajectory outgroup), and biologically
        motivated root selection (most-naive CD4 Naive T cell).

Run:
    cd /Users/yichen/Desktop/collaboration/raw
    /Users/yichen/anaconda3/bin/conda run -n cytof python cytof_advanced_v2_fix.py
"""

import os, warnings, textwrap
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
import anndata as ad
import scanpy as sc

from nature_style import (
    apply_nature_style, GROUP_PALETTE,
    HEATMAP_DIVERG, HEATMAP_SEQ, save
)
apply_nature_style()
SC, DC = 3.46, 7.09

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
    ["L01912_FH0002", "Jo-1" ], ["L01912_FH0003", "Jo-1" ],
    ["L01912_FH0009", "PL-12"], ["L01912_FH0011", "EJ"   ],
    ["L01912_FH0012", "PL-7" ], ["L01912_FH0013", "PL-12"],
    ["L01912_FH0014", "Jo-1" ], ["L01912_FH0015", "PL-12"],
    ["L01912_FH0016", "EJ"   ], ["L01912_FH0017", "PL-7" ],
], columns=["sample","antibody"]).set_index("sample")
GROUP_ORDER = ["Jo-1", "PL-12", "EJ", "PL-7"]

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

# how many patients per group
n_per_group = adata.obs.groupby("group", observed=True)["sample"].nunique()
print("  patients per group:", n_per_group.to_dict())

os.makedirs(OUT_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# FIG 27b — FULL SCAN + PATIENT-LEVEL CLUSTER BOOTSTRAP, SYMMETRIC TOP HITS
# ════════════════════════════════════════════════════════════════════════════
print("\n  Fig 27b (fixed): patient-level cluster bootstrap …")

gA, gB = "EJ", "PL-7"
B      = 2000
MIN_CELLS_PER_PATIENT = 20      # skip (patient, ct) pairs with too few cells
TOP_K_PER_DIR = 6               # show top-6 in each direction

# per-patient × per-(cell_type, marker) mean expression
X_df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
obs  = adata.obs[["sample","cell_type","group"]].copy()
obs["sample"] = obs["sample"].astype(str)

# cells of interest: only EJ or PL-7 patients
sel = obs["group"].isin([gA, gB]).values
obs_sel = obs.loc[sel].copy()
X_sel   = X_df.loc[obs_sel.index]

samples_EJ  = [s for s in PATIENT_META.index if PATIENT_META.loc[s,"antibody"] == gA]
samples_PL7 = [s for s in PATIENT_META.index if PATIENT_META.loc[s,"antibody"] == gB]
print(f"    {gA} n={len(samples_EJ)}, {gB} n={len(samples_PL7)}")

cell_types = [c for c in sorted(obs_sel["cell_type"].unique())
              if c not in ("Unannotated",)]
markers    = list(adata.var_names)

# pre-compute per-(sample, cell_type) cell-index arrays, so cluster bootstrap
# can be done by re-indexing
print("    Indexing per-(patient, cell-type) cell arrays …")
cell_idx = {}   # (sample, ct) -> row indices into X_sel
for (s, ct), grp in obs_sel.groupby(["sample","cell_type"], observed=True):
    cell_idx[(s, ct)] = grp.index.values

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2: return np.nan
    s2 = ((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx+ny-2)
    sd = np.sqrt(s2)
    return (np.mean(x) - np.mean(y)) / sd if sd > 0 else np.nan

rng = np.random.default_rng(42)
records = []

print(f"    Scanning {len(cell_types)}×{len(markers)} = "
      f"{len(cell_types)*len(markers)} (ct × marker) pairs …")

for ct in cell_types:
    # make sure both groups have ≥2 patients with at least MIN_CELLS_PER_PATIENT cells
    usable_A = [s for s in samples_EJ  if (s, ct) in cell_idx
                and len(cell_idx[(s, ct)]) >= MIN_CELLS_PER_PATIENT]
    usable_B = [s for s in samples_PL7 if (s, ct) in cell_idx
                and len(cell_idx[(s, ct)]) >= MIN_CELLS_PER_PATIENT]
    if len(usable_A) < 2 or len(usable_B) < 2:
        continue

    for m in markers:
        # point estimate: pool across patients within group
        xA_all = X_sel.loc[np.concatenate([cell_idx[(s, ct)] for s in usable_A]), m].values
        xB_all = X_sel.loc[np.concatenate([cell_idx[(s, ct)] for s in usable_B]), m].values
        d_pt   = cohens_d(xA_all, xB_all)
        if np.isnan(d_pt):
            continue

        # cluster (patient-level) bootstrap
        boots = np.empty(B)
        for b in range(B):
            sA = rng.choice(usable_A, size=len(usable_A), replace=True)
            sB = rng.choice(usable_B, size=len(usable_B), replace=True)
            xA = X_sel.loc[np.concatenate([cell_idx[(s, ct)] for s in sA]), m].values
            xB = X_sel.loc[np.concatenate([cell_idx[(s, ct)] for s in sB]), m].values
            boots[b] = cohens_d(xA, xB)

        boots = boots[~np.isnan(boots)]
        if len(boots) < B*0.8:
            continue

        ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
        # two-sided empirical p value: fraction of bootstraps on the wrong side of 0
        # (conservative definition; accounts for small n)
        p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())

        records.append({
            "cell_type": ct, "marker": m,
            "n_A": len(usable_A), "n_B": len(usable_B),
            "d": d_pt, "ci_low": ci_low, "ci_high": ci_high,
            "boot_p": p,
            "crosses_zero": (ci_low < 0 < ci_high),
        })

d_df = pd.DataFrame(records)

# BH-FDR adjustment on boot_p
from statsmodels.stats.multitest import multipletests
if len(d_df) > 0:
    _, q, _, _ = multipletests(d_df["boot_p"].values, method="fdr_bh")
    d_df["q_fdr"] = q
else:
    d_df["q_fdr"] = np.nan
d_df.to_csv(f"{OUT_DIR}/27b_DE_bootstrap_CI.csv", index=False)
print(f"    Scanned {len(d_df)} viable (ct × marker) pairs")
print(f"    Significant at FDR < 0.10: {(d_df['q_fdr']<0.10).sum()}")
print(f"    CI does not cross 0       : {(~d_df['crosses_zero']).sum()}")

# select symmetric top hits
top_A = d_df[d_df["d"] > 0].sort_values("d", ascending=False).head(TOP_K_PER_DIR)
top_B = d_df[d_df["d"] < 0].sort_values("d", ascending=True ).head(TOP_K_PER_DIR)
plot_df = pd.concat([top_B, top_A], axis=0).reset_index(drop=True)
# order for plotting: largest PL-7↑ at bottom → largest EJ↑ at top
plot_df = plot_df.sort_values("d").reset_index(drop=True)

# ---- plot ----
fig, ax = plt.subplots(figsize=(DC-0.2, 3.8))
y = np.arange(len(plot_df))
for i, r in plot_df.iterrows():
    col = GROUP_PALETTE[gA] if r["d"] > 0 else GROUP_PALETTE[gB]
    alpha_bar = 0.95 if not r["crosses_zero"] else 0.45
    ax.plot([r["ci_low"], r["ci_high"]], [i, i], color=col,
            lw=1.5, alpha=alpha_bar, solid_capstyle="round")
    ax.plot(r["d"], i, "o", color=col, markersize=4.8,
            markeredgecolor="white", markeredgewidth=0.6, zorder=3)
    # FDR annotation
    if r["q_fdr"] < 0.10:
        sig = "**" if r["q_fdr"] < 0.01 else "*"
        xend = r["ci_high"] + 0.05 if r["d"] > 0 else r["ci_low"] - 0.05
        ha   = "left"           if r["d"] > 0 else "right"
        ax.text(xend, i, sig, fontsize=7, color=col, ha=ha, va="center",
                fontweight="bold")

ax.axvline(0, color="#222", lw=0.7, zorder=1)
ax.set_yticks(y)
ax.set_yticklabels([f"{r['marker']} · {r['cell_type']}"
                    for _, r in plot_df.iterrows()], fontsize=6)
ax.set_xlabel(f"Cohen's d  ({gA} − {gB})", fontsize=7)
ax.set_title(f"EJ vs PL-7 marker expression — patient-level cluster bootstrap\n"
             f"(B = {B},  top {TOP_K_PER_DIR} per direction,  * FDR<0.10  ** FDR<0.01)",
             fontsize=7.5, pad=5)
ax.text(0.015, 0.985, f"↑ in {gB}", transform=ax.transAxes,
        va="top", fontsize=6, color=GROUP_PALETTE[gB], fontweight="bold")
ax.text(0.985, 0.985, f"↑ in {gA}", transform=ax.transAxes,
        va="top", ha="right", fontsize=6, color=GROUP_PALETTE[gA], fontweight="bold")
# show n
ax.text(0.5, -0.14, f"n(patients): {gA} = {len(samples_EJ)}, {gB} = {len(samples_PL7)}  ·  "
                    f"cluster bootstrap resamples patients (not cells) within group",
        transform=ax.transAxes, ha="center", fontsize=5.5, color="#555")
# pad x limits for asterisks
xmin, xmax = ax.get_xlim()
pad = 0.12*(xmax - xmin)
ax.set_xlim(xmin - pad, xmax + pad)

save(fig, "27b_DE_bootstrap_CI_forest", OUT_DIR)
print("    Saved: 27b_DE_bootstrap_CI_forest.pdf/.png")

# ════════════════════════════════════════════════════════════════════════════
# FIG 28 — MARKER CO-EXPRESSION MODULES, CLEAN LEGEND
# ════════════════════════════════════════════════════════════════════════════
print("\n  Fig 28 (fixed): marker co-expression modules with clean legend …")

# per-sample global marker mean
GLOBAL = (
    pd.DataFrame(adata.X, columns=adata.var_names)
    .assign(sample=adata.obs["sample"].values)
    .groupby("sample").mean().loc[PATIENT_META.index]
)

rho = GLOBAL.corr(method="spearman")
link_m = hierarchy.linkage(pdist(rho.values, metric="correlation"), method="average")
k_mod = 6
mod_ids  = hierarchy.fcluster(link_m, t=k_mod, criterion="maxclust")
mk_order = hierarchy.leaves_list(link_m)
rho_ord  = rho.iloc[mk_order, mk_order]
mod_ord  = mod_ids[mk_order]

module_df = pd.DataFrame({"marker": rho.index,
                          "module": [f"M{m}" for m in mod_ids]})
module_df.to_csv(f"{OUT_DIR}/28_marker_modules.csv", index=False)

# Figure layout: wider, give legend column more breathing room.
fig = plt.figure(figsize=(DC+1.0, 5.8))
gs = gridspec.GridSpec(2, 2, figure=fig,
                       width_ratios=[3.6, 2.6], height_ratios=[5.2, 2.0],
                       wspace=0.30, hspace=0.55)

# (a) correlation heatmap with module boxes
ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(rho_ord.values, aspect="auto", cmap=HEATMAP_DIVERG,
               vmin=-1, vmax=1)
ax.set_xticks(range(len(rho_ord)))
ax.set_xticklabels(rho_ord.columns, rotation=60, ha="right", fontsize=5)
ax.set_yticks(range(len(rho_ord)))
ax.set_yticklabels(rho_ord.index, fontsize=5)
ax.tick_params(length=0)

module_colors = plt.cm.get_cmap("tab10", k_mod)
# map original module ids (1..k) to the order they first appear along the diagonal
first_appear = {}
for k, m in enumerate(mod_ord):
    if m not in first_appear:
        first_appear[m] = len(first_appear)
# boundaries
for m in np.unique(mod_ord):
    idx = np.where(mod_ord == m)[0]
    start, end = idx.min(), idx.max()
    ax.add_patch(plt.Rectangle((start-0.5, start-0.5),
                               end-start+1, end-start+1,
                               fill=False, edgecolor="black", linewidth=1.2))

cax = fig.add_axes([0.535, 0.58, 0.010, 0.23])
cb = fig.colorbar(im, cax=cax)
cb.set_label("Spearman ρ", fontsize=6)
cb.ax.tick_params(labelsize=5)
ax.set_title("Patient-level marker co-expression (Spearman)",
             fontsize=7.5, pad=3)

# (b) CLEAN module membership list with textwrap
ax2 = fig.add_subplot(gs[0, 1])
ax2.axis("off")

# order modules by first appearance in diagonal for consistency
mod_order_sorted = sorted(first_appear, key=lambda m: first_appear[m])
mod_palette = {f"M{m}": matplotlib.colors.to_hex(module_colors(first_appear[m] % k_mod))
               for m in mod_order_sorted}

# figure out how much text we can fit
WRAP_WIDTH = 32              # characters per line
LINE_HEIGHT = 0.040          # y-axis fraction per line
BLOCK_GAP   = 0.025          # extra gap between modules
HEADER_GAP  = 0.050

y_pos = 0.985
ax2.text(0, y_pos, "Marker modules (M1–M6):",
         fontsize=8, fontweight="bold", transform=ax2.transAxes)
y_pos -= HEADER_GAP + 0.015

for m in mod_order_sorted:
    col = mod_palette[f"M{m}"]
    mks = module_df[module_df["module"] == f"M{m}"]["marker"].tolist()
    # header
    header = f"M{m}  (n={len(mks)})"
    ax2.text(0.00, y_pos, "■", fontsize=9, color=col,
             va="center", transform=ax2.transAxes)
    ax2.text(0.055, y_pos, header, fontsize=7, color=col,
             fontweight="bold", va="center", transform=ax2.transAxes)
    y_pos -= LINE_HEIGHT

    # wrapped member list
    members = ", ".join(mks)
    wrapped = textwrap.wrap(members, width=WRAP_WIDTH,
                            break_long_words=False, break_on_hyphens=False)
    for line in wrapped:
        ax2.text(0.055, y_pos, line, fontsize=6.2, color="#222",
                 va="center", transform=ax2.transAxes)
        y_pos -= LINE_HEIGHT
    y_pos -= BLOCK_GAP

# (c) module score × group heatmap
ax3 = fig.add_subplot(gs[1, :])
mod_scores = pd.DataFrame(index=PATIENT_META.index)
for m in np.unique(mod_ids):
    mks = module_df[module_df["module"] == f"M{m}"]["marker"].tolist()
    mks = [x for x in mks if x in GLOBAL.columns]
    if mks: mod_scores[f"M{m}"] = GLOBAL[mks].mean(axis=1)

mod_scores_z = (mod_scores - mod_scores.mean()) / mod_scores.std(ddof=0)
group_mod = pd.DataFrame(index=GROUP_ORDER, columns=mod_scores_z.columns, dtype=float)
for g in GROUP_ORDER:
    ss = [s for s in PATIENT_META.index if PATIENT_META.loc[s,"antibody"] == g]
    group_mod.loc[g] = mod_scores_z.loc[ss].mean()

# re-order columns by module_order_sorted
col_order = [f"M{m}" for m in mod_order_sorted if f"M{m}" in group_mod.columns]
group_mod = group_mod[col_order]

im3 = ax3.imshow(group_mod.values, aspect="auto", cmap=HEATMAP_DIVERG,
                 vmin=-1.2, vmax=1.2)
ax3.set_xticks(range(group_mod.shape[1]))
ax3.set_xticklabels(group_mod.columns, fontsize=7)
# color x-ticks to match modules
for tl, c in zip(ax3.get_xticklabels(), col_order):
    tl.set_color(mod_palette[c]); tl.set_fontweight("bold")
ax3.set_yticks(range(group_mod.shape[0]))
ax3.set_yticklabels(group_mod.index, fontsize=7)
ax3.tick_params(length=0)
for i in range(group_mod.shape[0]):
    for j in range(group_mod.shape[1]):
        v = group_mod.iloc[i, j]
        col = "white" if abs(v) > 0.6 else "#111"
        ax3.text(j, i, f"{v:.2f}", ha="center", va="center",
                 fontsize=6, color=col)
ax3.set_title("Module z-score by autoantibody group", fontsize=7.5, pad=3)
cax2 = fig.add_axes([0.92, 0.13, 0.011, 0.16])
cb2 = fig.colorbar(im3, cax=cax2)
cb2.set_label("Mean z-score", fontsize=6)
cb2.ax.tick_params(labelsize=5)

fig.suptitle("Data-driven marker co-expression modules recapitulate canonical "
             "immune functional programs", fontsize=8.5, y=1.00)
save(fig, "28_marker_coexpression_modules", OUT_DIR)
print("    Saved: 28_marker_coexpression_modules.pdf/.png")

# ════════════════════════════════════════════════════════════════════════════
# FIG 29 — T-cell trajectory WITHOUT γδ, biologically motivated root
# ════════════════════════════════════════════════════════════════════════════
print("\n  Fig 29 (fixed): T cell trajectory (γδ as off-trajectory outgroup) …")

T_CELLS_TRAJ = [
    "CD4 Naive T", "CD4 Central Memory T", "CD4 Effector Memory T",
    "CD4 Th1-like", "CD8 Naive T", "CD8 Effector T",
]
T_CELLS_ALL = T_CELLS_TRAJ + ["γδ T cell"]

tdata_all = adata[adata.obs["cell_type"].isin(T_CELLS_ALL)].copy()
tdata = adata[adata.obs["cell_type"].isin(T_CELLS_TRAJ)].copy()
tdata.obs["cell_type"] = pd.Categorical(tdata.obs["cell_type"],
                                        categories=T_CELLS_TRAJ)
print(f"    T subset for trajectory: {tdata.n_obs:,} cells (γδ removed)")
print(f"    γδ T cells kept aside   : {(tdata_all.obs['cell_type']=='γδ T cell').sum():,}")

sc.pp.neighbors(tdata, use_rep="X_pca_harmony", n_neighbors=30, random_state=0)
sc.tl.diffmap(tdata, n_comps=15)
sc.tl.paga(tdata, groups="cell_type")

# ---- biologically motivated root: most-naive CD4 Naive T cell ----
# score = CCR7 + CD45RA + CD27 − CD45RO − HLA-DR  (classic naive signature)
naive_markers_pos = ["CCR7", "CD45RA", "CD27"]
naive_markers_neg = ["CD45RO", "HLA-DR"]
Xn_pos = tdata[:, [m for m in naive_markers_pos if m in tdata.var_names]].X
Xn_neg = tdata[:, [m for m in naive_markers_neg if m in tdata.var_names]].X
naive_score = np.asarray(Xn_pos).mean(axis=1) - np.asarray(Xn_neg).mean(axis=1)
naive_mask  = (tdata.obs["cell_type"] == "CD4 Naive T").values
# root = CD4 Naive T cell with the single highest naive score
cand_idx   = np.where(naive_mask)[0]
root_local = cand_idx[int(np.argmax(naive_score[cand_idx]))]
tdata.uns["iroot"] = int(root_local)
print(f"    Root cell index: {root_local}  "
      f"(naive score = {naive_score[root_local]:.3f})")

sc.tl.dpt(tdata, n_branchings=0)
sc.tl.umap(tdata, random_state=0)

# separate UMAP for all T cells including γδ (for panel a)
sc.pp.neighbors(tdata_all, use_rep="X_pca_harmony", n_neighbors=30, random_state=0)
sc.tl.umap(tdata_all, random_state=0)

# ---- plotting ----
fig = plt.figure(figsize=(DC+1.2, 6.2))
gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.32, hspace=0.55,
                       height_ratios=[1.0, 1.15])

TCELL_PAL = {
    "CD4 Naive T":          "#E64B35",
    "CD4 Central Memory T": "#4DBBD5",
    "CD4 Effector Memory T":"#00A087",
    "CD4 Th1-like":         "#F39B7F",
    "CD8 Naive T":          "#3C5488",
    "CD8 Effector T":       "#8491B4",
    "γδ T cell":            "#7E6148",
}
Z_umap_all = tdata_all.obsm["X_umap"]
Z_dmap = tdata.obsm["X_diffmap"][:, 1:3]
dpt    = tdata.obs["dpt_pseudotime"].values

idx_all = np.random.default_rng(0).choice(
    tdata_all.n_obs, size=min(15000, tdata_all.n_obs), replace=False)
idx = np.random.default_rng(1).choice(
    tdata.n_obs, size=min(15000, tdata.n_obs), replace=False)

# (a) UMAP of all T cells — shows γδ is separate
ax = fig.add_subplot(gs[0, 0])
for ct in T_CELLS_ALL:
    m = (tdata_all.obs["cell_type"].values == ct)
    mi = idx_all[m[idx_all]]
    is_gd = (ct == "γδ T cell")
    ax.scatter(Z_umap_all[mi,0], Z_umap_all[mi,1],
               s=1.2 if is_gd else 1.0,
               c=TCELL_PAL[ct],
               alpha=0.75 if is_gd else 0.5,
               edgecolor="none",
               label=ct + (" (off-trajectory)" if is_gd else ""))
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("UMAP-1", fontsize=6); ax.set_ylabel("UMAP-2", fontsize=6)
ax.set_title("T cell UMAP", fontsize=7.5)
ax.legend(fontsize=4.8, markerscale=3, loc="upper right",
          frameon=False, handlelength=1, handletextpad=0.3,
          borderpad=0.2, labelspacing=0.25)

# (b) Diffusion map by cell type (γδ already removed)
ax = fig.add_subplot(gs[0, 1])
for ct in T_CELLS_TRAJ:
    m = (tdata.obs["cell_type"].values == ct)
    mi = idx[m[idx]]
    ax.scatter(Z_dmap[mi,0], Z_dmap[mi,1], s=1.0,
               c=TCELL_PAL[ct], alpha=0.5, edgecolor="none")
# mark root
ax.scatter(Z_dmap[root_local,0], Z_dmap[root_local,1], s=60,
           facecolor="none", edgecolor="#C0392B", lw=1.4, zorder=4)
ax.annotate("root", xy=(Z_dmap[root_local,0], Z_dmap[root_local,1]),
            xytext=(12, 12), textcoords="offset points", fontsize=5.8,
            color="#C0392B",
            arrowprops=dict(arrowstyle="-", color="#C0392B", lw=0.5))
ax.set_xlabel("DC1", fontsize=6); ax.set_ylabel("DC2", fontsize=6)
ax.set_title("Diffusion map · cell type", fontsize=7.5)
ax.set_xticks([]); ax.set_yticks([])

# (c) Diffusion map coloured by pseudotime
ax = fig.add_subplot(gs[0, 2])
sc_im = ax.scatter(Z_dmap[idx,0], Z_dmap[idx,1], s=1.0,
                    c=dpt[idx], cmap="plasma",
                    alpha=0.7, edgecolor="none")
ax.set_xlabel("DC1", fontsize=6); ax.set_ylabel("DC2", fontsize=6)
ax.set_title("Diffusion map · pseudotime", fontsize=7.5)
ax.set_xticks([]); ax.set_yticks([])
cax = fig.add_axes([0.92, 0.58, 0.010, 0.22])
cb = fig.colorbar(sc_im, cax=cax)
cb.set_label("DPT pseudotime", fontsize=6)
cb.ax.tick_params(labelsize=5)

# (d) PAGA graph
ax = fig.add_subplot(gs[1, 0:2])
import networkx as nx
conn = tdata.uns["paga"]["connectivities"].toarray()
group_names = tdata.obs["cell_type"].cat.categories.tolist()
G = nx.Graph()
for i in range(len(group_names)):
    G.add_node(i)
for i in range(len(group_names)):
    for j in range(i+1, len(group_names)):
        if conn[i, j] > 0.05:
            G.add_edge(i, j, weight=float(conn[i, j]))
pos_dict = nx.spring_layout(G, seed=0, k=2.2, iterations=200)
pos = np.array([pos_dict[i] for i in range(len(group_names))])
# rescale to [-1, 1] and add horizontal padding for labels
pos = pos - pos.mean(0)
pos = pos / max(np.abs(pos).max(), 1e-8) * 0.82
# edges
for i in range(len(group_names)):
    for j in range(i+1, len(group_names)):
        w = conn[i, j]
        if w > 0.05:
            ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]],
                    color="#888", lw=0.4+4*w, alpha=0.75, zorder=1)
sizes = tdata.obs["cell_type"].value_counts().reindex(group_names).values
sizes_sc = 120 + sizes / sizes.max() * 650
for i, ct in enumerate(group_names):
    ax.scatter(pos[i,0], pos[i,1], s=sizes_sc[i],
               c=TCELL_PAL[ct], edgecolor="black", linewidth=0.7, zorder=3)
# place labels in quadrant-aware positions so they never overlap the nodes
for i, ct in enumerate(group_names):
    x, y = pos[i]
    # offset direction based on position (push labels outward from centroid)
    dx = 0.0
    dy = 0.17 if y >= 0 else -0.17
    ha = "center"
    va = "bottom" if y >= 0 else "top"
    ax.text(x + dx, y + dy, ct, ha=ha, va=va, fontsize=6.5,
            color="#111", zorder=5, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none",
                      alpha=0.82, pad=1.2))

# add γδ as an isolated annotation chip (off-trajectory), placed safely
gd_n = int((tdata_all.obs['cell_type'] == "γδ T cell").sum())
ax.scatter([1.22], [0.0], s=180, c=TCELL_PAL["γδ T cell"],
           edgecolor="black", linewidth=0.7, zorder=3)
ax.text(1.22, 0.20, f"γδ T\n(n = {gd_n:,})\noff-trajectory",
        fontsize=6, color=TCELL_PAL["γδ T cell"], ha="center", va="bottom",
        fontweight="bold", zorder=5)
# dashed "excluded" line between lineage cloud and γδ chip
ax.plot([0.95, 1.10], [0.0, 0.0], ls="--", color="#bbb", lw=0.6, zorder=1)

ax.set_xticks([]); ax.set_yticks([])
ax.set_xlim(-1.15, 1.55)
ax.set_ylim(-1.15, 1.15)
ax.set_title("PAGA connectivity (γδ excluded as independent lineage)",
             fontsize=7.5, pad=3)
for spine in ax.spines.values():
    spine.set_visible(False)

# (e) Pseudotime violin by cell type — ordered by median pseudotime
ax = fig.add_subplot(gs[1, 2])
order_by_pt = sorted(T_CELLS_TRAJ,
                     key=lambda c: np.median(dpt[tdata.obs["cell_type"] == c]))
parts_data, parts_labels = [], []
for ct in order_by_pt:
    m = (tdata.obs["cell_type"].values == ct)
    parts_data.append(dpt[m]); parts_labels.append(ct)
parts = ax.violinplot(parts_data, positions=np.arange(len(parts_labels)),
                      vert=False, widths=0.9, showmeans=False, showmedians=True)
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(TCELL_PAL[parts_labels[i]])
    pc.set_alpha(0.6)
    pc.set_edgecolor(TCELL_PAL[parts_labels[i]])
    pc.set_linewidth(0.6)
for partname in ["cmedians","cbars","cmins","cmaxes"]:
    if partname in parts:
        parts[partname].set_color("#333"); parts[partname].set_lw(0.5)
ax.set_yticks(range(len(parts_labels)))
ax.set_yticklabels(parts_labels, fontsize=6)
for tl, lbl in zip(ax.get_yticklabels(), parts_labels):
    tl.set_color(TCELL_PAL[lbl])
ax.set_xlabel("DPT pseudotime", fontsize=6)
ax.set_title("Pseudotime by cell type\n(ordered by median)", fontsize=7.5)

fig.suptitle("T cell differentiation trajectory: diffusion map + PAGA + DPT "
             "(γδ T as off-trajectory outgroup)",
             fontsize=8.5, y=1.00)
save(fig, "29_T_cell_pseudotime", OUT_DIR)
print("    Saved: 29_T_cell_pseudotime.pdf/.png")

pt_summary = pd.DataFrame({
    "cell_type":        T_CELLS_TRAJ,
    "n_cells":          [int((tdata.obs["cell_type"] == ct).sum()) for ct in T_CELLS_TRAJ],
    "mean_pseudotime":  [float(dpt[tdata.obs["cell_type"] == ct].mean())   for ct in T_CELLS_TRAJ],
    "median_pseudotime":[float(np.median(dpt[tdata.obs["cell_type"] == ct])) for ct in T_CELLS_TRAJ],
})
pt_summary = pt_summary.sort_values("median_pseudotime").reset_index(drop=True)
pt_summary.to_csv(f"{OUT_DIR}/29_pseudotime_summary.csv", index=False)
print("    Pseudotime order:")
print(pt_summary.to_string(index=False))

print(f"""
══════════════════════════════════════════════════════════
  v2 fixes complete.
    27b_DE_bootstrap_CI_forest        → patient-level cluster bootstrap + FDR
    28_marker_coexpression_modules    → clean legend (no overlap)
    29_T_cell_pseudotime              → γδ excluded; biologically motivated root
══════════════════════════════════════════════════════════
""")
