"""
Main Figure 8 — Cell-cell co-activation network and per-immunotype
activation profiles.

Methodological note: at n=3-4 patients per immunotype, Spearman
correlation saturates at ±1/±0.5 and per-IT 'coupling networks' are
statistically uninformative. We therefore use a two-pronged framing:

  (1) GLOBAL co-activation network on n=10 patients (panel b–c) — this is
      the cell-cell coupling biology that holds across the cohort.
  (2) PER-IT mean activation profiles per cell type (panel a, d) — at
      n=3-4 the patient-mean activation score is a stable summary
      statistic, even when correlation networks are not.

The 4 panels:

  (a) Patient × cell-type × module score heatmap — rows grouped by IT.
  (b) Global Spearman co-activation matrix on the full cohort (n=10).
  (c) Global activation network (force layout, |ρ|≥0.6).
  (d) Per-IT mean activation profile per cell type (heatmap of cell-type
      × IT, M3+M6 modules combined). IT1 = activation high across innate
      compartments; IT3 = uniformly low.
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
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr
import networkx as nx

OUT_DIR = "./cytof_output_nature"
H5AD    = "/Users/yichen/Desktop/collaboration/raw/cytof_output_v3/cytof_analyzed_v3.h5ad"

CLUSTER_ANNOTATION = {
    "0":"Classical Monocyte","1":"Naive B","2":"CD8 Effector T",
    "3":"CD4 Naive T","4":"CD4 Central Memory T","5":"Non-classical Monocyte",
    "6":"NK cell","7":"CD4 Th1-like","8":"Classical Monocyte",
    "9":"CD4 Effector Memory T","10":"Inflammatory Monocyte",
    "11":"CD8 Naive T","12":"γδ T cell","13":"mDC","14":"pDC",
    "15":"T-Myeloid doublets","16":"CD16+ Granulocyte",
}
ARTIFACTS = {"T-Myeloid doublets", "CD16+ Granulocyte"}
CT_ORDER = [
    "CD4 Naive T", "CD4 Central Memory T", "CD4 Effector Memory T", "CD4 Th1-like",
    "CD8 Naive T", "CD8 Effector T", "γδ T cell",
    "Classical Monocyte", "Inflammatory Monocyte", "Non-classical Monocyte",
    "mDC", "pDC", "NK cell", "Naive B",
]
CT_PALETTE = {
    "CD4 Naive T":           "#7DC1E8", "CD4 Central Memory T":  "#1F77B4",
    "CD4 Effector Memory T": "#5B9BD5", "CD4 Th1-like":          "#3A7CA5",
    "CD8 Naive T":           "#C5B0D5", "CD8 Effector T":        "#9467BD",
    "γδ T cell":             "#F1C40F",
    "Classical Monocyte":    "#E67E22", "Inflammatory Monocyte": "#D35400",
    "Non-classical Monocyte":"#F39C12",
    "mDC": "#E74C3C", "pDC": "#C0392B",
    "NK cell": "#27AE60", "Naive B": "#16A085",
}
IT_COL = {"IT1": "#E67E22", "IT2": "#2980B9", "IT3": "#27AE60"}
IT_ORDER = ["IT1", "IT2", "IT3"]

plt.rcParams.update({
    "font.family":"sans-serif",
    "font.sans-serif":["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":7.0, "axes.titlesize":7.5, "axes.labelsize":7.0,
    "xtick.labelsize":6.2, "ytick.labelsize":6.2, "legend.fontsize":6.0,
    "axes.linewidth":0.7, "axes.spines.top":False, "axes.spines.right":False,
    "pdf.fonttype":42, "ps.fonttype":42, "savefig.dpi":300, "figure.dpi":300,
})
PANEL_LBL = dict(fontsize=10, fontweight="bold", color="#000",
                 ha="left", va="bottom", family="sans-serif")

# ─── load + prepare ────────────────────────────────────────────────────────
print("Loading h5ad …")
a = ad.read_h5ad(H5AD)
a.obs_names_make_unique()
a.obs["cell_type"] = a.obs["leiden"].astype(str).map(CLUSTER_ANNOTATION).fillna("Unannotated")
a = a[~a.obs["cell_type"].isin(ARTIFACTS)].copy()
assigned = pd.read_csv(f"{OUT_DIR}/21_immunotype_assignments.csv")
sample_to_it = dict(zip(assigned["sample"], assigned["immunotype"]))
a.obs["immunotype"] = a.obs["sample"].map(sample_to_it)

mod_df = pd.read_csv(f"{OUT_DIR}/28_marker_modules.csv")
mod_df = mod_df[mod_df["marker"].isin(a.var_names)]
modules = sorted(mod_df["module"].unique())
print(f"  {len(modules)} modules: {modules}")

def mean_module_score(adata, sample, ct, markers):
    mask = (adata.obs["sample"].values == sample) & \
           (adata.obs["cell_type"].values == ct)
    if mask.sum() < 20:
        return np.nan
    X_sub = adata.X[mask]
    X_sub = X_sub.toarray() if hasattr(X_sub, "toarray") else np.asarray(X_sub)
    cols = [list(adata.var_names).index(m) for m in markers if m in adata.var_names]
    if not cols:
        return np.nan
    return float(X_sub[:, cols].mean())

samples = sorted(a.obs["sample"].unique())
print("Computing patient × cell-type × module score matrix …")
records = []
for s in samples:
    it = sample_to_it.get(s)
    for ct in CT_ORDER:
        if ct in ARTIFACTS:
            continue
        for m in modules:
            mks = mod_df[mod_df["module"] == m]["marker"].tolist()
            score = mean_module_score(a, s, ct, mks)
            records.append({"sample": s, "immunotype": it,
                            "cell_type": ct, "module": m,
                            "score": score})
score_df = pd.DataFrame(records)
score_df.to_csv(f"{OUT_DIR}/30_module_score_per_sample_celltype.csv", index=False)

# z-score within (cell_type × module) across patients
score_df["z"] = score_df.groupby(["cell_type","module"], group_keys=False, observed=False)\
                         .apply(lambda g: pd.Series(
                             (g["score"].values - np.nanmean(g["score"].values)) /
                             (np.nanstd(g["score"].values) + 1e-9), index=g.index))


# ════════════════════════════════════════════════════════════════════════════
# (a) heatmap rows = (patient × cell type), cols = M1..M6, grouped by IT
# ════════════════════════════════════════════════════════════════════════════
heat = (score_df.pivot_table(index=["immunotype","sample","cell_type"],
                              columns="module", values="z", aggfunc="first",
                              observed=False)
                .reset_index())
heat["immunotype"] = pd.Categorical(heat["immunotype"], categories=IT_ORDER, ordered=True)
heat["cell_type"]  = pd.Categorical(heat["cell_type"], categories=CT_ORDER, ordered=True)
heat = heat.sort_values(["immunotype","sample","cell_type"]).reset_index(drop=True)
M_a = heat[modules].values

# ════════════════════════════════════════════════════════════════════════════
# (b/c) global co-activation matrix on n=10 patients (M3 module)
# ════════════════════════════════════════════════════════════════════════════
m3 = score_df[score_df["module"] == "M3"]
pv = m3.pivot_table(index="sample", columns="cell_type", values="z",
                     aggfunc="first", observed=False)
cts_b = [c for c in CT_ORDER if c in pv.columns]
pv = pv[cts_b]
n_ct = len(cts_b)
M_glob = np.full((n_ct, n_ct), np.nan)
for i, c1 in enumerate(cts_b):
    for j, c2 in enumerate(cts_b):
        v1, v2 = pv[c1].values, pv[c2].values
        ok = ~(np.isnan(v1) | np.isnan(v2))
        if ok.sum() >= 4:
            r, _ = spearmanr(v1[ok], v2[ok])
            M_glob[i, j] = r
np.fill_diagonal(M_glob, 1.0)

# Hierarchical sort on global correlation matrix for heatmap legibility
from scipy.cluster.hierarchy import linkage, leaves_list
Z = linkage(np.nan_to_num(M_glob), method="average")
order = leaves_list(Z)
M_glob_o = M_glob[order][:, order]
cts_b_o = [cts_b[i] for i in order]


# ════════════════════════════════════════════════════════════════════════════
# (d) per-IT mean activation score per cell type, M3+M6 combined
# ════════════════════════════════════════════════════════════════════════════
m36 = score_df[score_df["module"].isin(["M3","M6"])].copy()
m36_mean = (m36.groupby(["immunotype","cell_type"], observed=False)["z"]
               .mean()
               .reset_index())
heatmap_d = m36_mean.pivot_table(index="cell_type", columns="immunotype",
                                 values="z", aggfunc="first",
                                 observed=False)
heatmap_d = heatmap_d.reindex(index=CT_ORDER)[IT_ORDER]


# ════════════════════════════════════════════════════════════════════════════
# FIGURE
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(7.5, 9.5))
gs = gridspec.GridSpec(
    3, 2, figure=fig,
    left=0.10, right=0.965, top=0.940, bottom=0.060,
    wspace=0.50, hspace=0.55,
    height_ratios=[1.5, 1.05, 1.0],
)

# ── (a) score heatmap ─────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, :])
HEAT = LinearSegmentedColormap.from_list("div",
    ["#1A3A6B","#2980B9","#7FB3D3","#F4D03F","#E67E22","#C0392B","#7B241C"], N=256)
n_rows = len(heat)
n_mod  = len(modules)
im = ax.pcolormesh(
    np.arange(n_mod + 1) - 0.5, np.arange(n_rows + 1) - 0.5,
    np.where(np.isnan(M_a), 0, M_a),
    cmap=HEAT, vmin=-2, vmax=2,
    edgecolors="none", rasterized=False, shading="flat",
)
ax.invert_yaxis()
ax.set_xticks(range(len(modules)))
ax.set_xticklabels(modules, fontsize=8, fontweight="bold")
strip_x = -0.65
for i in range(n_rows):
    ax.add_patch(Rectangle((strip_x, i - 0.5), 0.4, 1,
                            facecolor=IT_COL[heat["immunotype"].iloc[i]],
                            edgecolor="white", lw=0.3, clip_on=False))
ax.set_yticks([])
ax.set_xlim(-0.85, len(modules) - 0.5)
cbar = fig.colorbar(im, ax=ax, fraction=0.020, pad=0.025, shrink=0.55)
cbar.set_label("Z-score (within cell type × module)", fontsize=6.0)
cbar.ax.tick_params(labelsize=5.5); cbar.outline.set_linewidth(0.4)
# IT colour key annotation on left
ax.text(strip_x + 0.20, -2, "IT", ha="center", va="bottom",
        fontsize=6.0, color="#444", clip_on=False)
ax.set_title("Patient × cell-type × module score heatmap "
             "(rows: 10 patients × 14 cell types; grouped by IT — "
             "orange=IT1, blue=IT2, green=IT3)", fontsize=7.5, pad=4)
ax.text(-0.045, 1.02, "a", transform=ax.transAxes, **PANEL_LBL)


# ── (b) global Spearman co-activation matrix (M3) ─────────────────────────
ax = fig.add_subplot(gs[1, 0])
n_glob = M_glob_o.shape[0]
im2 = ax.pcolormesh(
    np.arange(n_glob + 1) - 0.5, np.arange(n_glob + 1) - 0.5,
    M_glob_o, cmap="RdBu_r", vmin=-1, vmax=1,
    edgecolors="none", rasterized=False, shading="flat",
)
ax.invert_yaxis()
ax.set_aspect("equal")
ax.set_xticks(range(n_ct))
ax.set_yticks(range(n_ct))
ax.set_xticklabels(cts_b_o, rotation=90, fontsize=5.0)
ax.set_yticklabels(cts_b_o, fontsize=5.5)
cbar2 = fig.colorbar(im2, ax=ax, fraction=0.040, pad=0.025, shrink=0.85)
cbar2.set_label("Spearman ρ (M3 activation, n=10)", fontsize=6.0)
cbar2.ax.tick_params(labelsize=5.5); cbar2.outline.set_linewidth(0.4)
ax.set_title("Global cell-cell co-activation matrix\n(M3 module, n = 10 patients pooled)",
             fontsize=7.0, pad=4)
ax.text(-0.30, 1.05, "b", transform=ax.transAxes, **PANEL_LBL)


# ── (c) global activation network (force layout, |ρ|≥0.6) ────────────────
ax = fig.add_subplot(gs[1, 1])
G = nx.Graph()
for ct in cts_b:
    G.add_node(ct)
threshold = 0.6
for i, c1 in enumerate(cts_b):
    for j, c2 in enumerate(cts_b):
        if j <= i: continue
        rho = M_glob[i, j]
        if not np.isnan(rho) and abs(rho) >= threshold:
            G.add_edge(c1, c2, weight=rho)

pos = nx.spring_layout(G, seed=7, k=0.85, iterations=80)
# edges
for u, v, d in G.edges(data=True):
    rho = d["weight"]
    col = "#C0392B" if rho > 0 else "#2980B9"
    x1, y1 = pos[u]; x2, y2 = pos[v]
    ax.plot([x1, x2], [y1, y2], color=col,
            lw=0.5 + 1.6 * (abs(rho) - threshold), alpha=0.65,
            zorder=2)
# nodes
for n in G.nodes():
    x, y = pos[n]
    ax.scatter(x, y, s=80, c=CT_PALETTE.get(n, "#888"),
               edgecolor="white", linewidth=0.6, zorder=4)
    ax.text(x, y - 0.07, n, ha="center", va="top",
            fontsize=4.6, color="#222", fontweight="bold")

ax.set_title(f"Global activation network\n"
             f"(|ρ|≥{threshold}, {G.number_of_edges()} edges)",
             fontsize=7.5, pad=4)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_visible(False)
legend_handles = [
    Patch(facecolor="#C0392B", label="ρ > 0  (co-active)"),
    Patch(facecolor="#2980B9", label="ρ < 0  (anti-coupled)"),
]
ax.legend(handles=legend_handles, loc="upper right",
          fontsize=5.5, frameon=False,
          handlelength=0.8, handletextpad=0.3, borderpad=0.2)
ax.text(-0.15, 1.05, "c", transform=ax.transAxes, **PANEL_LBL)


# ── (d) per-IT mean activation per cell type (M3+M6) ─────────────────────
ax = fig.add_subplot(gs[2, :])
HEAT2 = LinearSegmentedColormap.from_list("div2",
    ["#1A3A6B","#2980B9","#FFFFFF","#E67E22","#C0392B"], N=256)
M_d = heatmap_d.values.T  # rows = IT, cols = cell type (transposed for layout)
n_it_d, n_ct_d = M_d.shape
im3 = ax.pcolormesh(
    np.arange(n_ct_d + 1) - 0.5, np.arange(n_it_d + 1) - 0.5,
    M_d, cmap=HEAT2, vmin=-1.2, vmax=1.2,
    edgecolors="white", lw=0.4, rasterized=False, shading="flat",
)
ax.invert_yaxis()
ax.set_aspect("auto")
ax.set_yticks(range(len(IT_ORDER)))
ax.set_yticklabels(IT_ORDER, fontsize=8, fontweight="bold")
for tick, it in zip(ax.get_yticklabels(), IT_ORDER):
    tick.set_color(IT_COL[it])
ax.set_xticks(range(len(CT_ORDER)))
ax.set_xticklabels(CT_ORDER, rotation=45, ha="right", fontsize=6.5)
# annotate cell values
for i in range(len(IT_ORDER)):
    for j in range(len(CT_ORDER)):
        v = M_d[i, j]
        if np.isnan(v): continue
        text_col = "white" if abs(v) > 0.7 else "#333"
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                fontsize=5.5, color=text_col)
cbar3 = fig.colorbar(im3, ax=ax, fraction=0.025, pad=0.012, shrink=0.85)
cbar3.set_label("Mean (M3+M6) activation Z-score", fontsize=6.0)
cbar3.ax.tick_params(labelsize=5.5); cbar3.outline.set_linewidth(0.4)
ax.set_title("Per-immunotype mean activation profile (M3 + M6) per cell type:\n"
             "IT1 = innate-myeloid + cytotoxic activation; IT3 = uniformly low",
             fontsize=7.2, pad=4)
ax.text(-0.045, 1.05, "d", transform=ax.transAxes, **PANEL_LBL)


# ─── suptitle + save ─────────────────────────────────────────────────────
# Figure title removed (journal convention).

out_pdf = f"{OUT_DIR}/MainFig8_network.pdf"
out_png = f"{OUT_DIR}/MainFig8_network.png"
fig.savefig(out_pdf, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
fig.savefig(out_png, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
print(f"Global network: {G.number_of_edges()} edges at |ρ| ≥ {threshold}")
