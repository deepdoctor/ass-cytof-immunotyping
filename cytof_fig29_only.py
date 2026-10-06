"""Re-render Fig 29 only (PAGA layout fix)."""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import anndata as ad
import scanpy as sc

from nature_style import apply_nature_style, save
apply_nature_style()
DC = 7.09

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
DROP = {"T-Myeloid doublets", "CD16+ Granulocyte"}

print("[Loading]", H5AD_PATH)
adata = ad.read_h5ad(H5AD_PATH)
adata.obs_names_make_unique()
adata.obs["cell_type"] = (adata.obs["leiden"].astype(str)
                          .map(CLUSTER_ANNOTATION).fillna("Unannotated"))
adata = adata[~adata.obs["cell_type"].isin(DROP)].copy()

T_CELLS_TRAJ = [
    "CD4 Naive T", "CD4 Central Memory T", "CD4 Effector Memory T",
    "CD4 Th1-like", "CD8 Naive T", "CD8 Effector T",
]
T_CELLS_ALL = T_CELLS_TRAJ + ["γδ T cell"]

tdata_all = adata[adata.obs["cell_type"].isin(T_CELLS_ALL)].copy()
tdata = adata[adata.obs["cell_type"].isin(T_CELLS_TRAJ)].copy()
tdata.obs["cell_type"] = pd.Categorical(tdata.obs["cell_type"],
                                        categories=T_CELLS_TRAJ)
print(f"  trajectory cells: {tdata.n_obs:,}  γδ: {(tdata_all.obs['cell_type']=='γδ T cell').sum():,}")

sc.pp.neighbors(tdata, use_rep="X_pca_harmony", n_neighbors=30, random_state=0)
sc.tl.diffmap(tdata, n_comps=15)
sc.tl.paga(tdata, groups="cell_type")

naive_mk_pos = [m for m in ["CCR7","CD45RA","CD27"]  if m in tdata.var_names]
naive_mk_neg = [m for m in ["CD45RO","HLA-DR"]       if m in tdata.var_names]
Xp = np.asarray(tdata[:, naive_mk_pos].X).mean(1)
Xn = np.asarray(tdata[:, naive_mk_neg].X).mean(1)
naive_score = Xp - Xn
naive_mask  = (tdata.obs["cell_type"] == "CD4 Naive T").values
cand_idx    = np.where(naive_mask)[0]
root_local  = int(cand_idx[int(np.argmax(naive_score[cand_idx]))])
tdata.uns["iroot"] = root_local
print("  root idx:", root_local, "score:", float(naive_score[root_local]))

sc.tl.dpt(tdata, n_branchings=0)
sc.tl.umap(tdata, random_state=0)

sc.pp.neighbors(tdata_all, use_rep="X_pca_harmony", n_neighbors=30, random_state=0)
sc.tl.umap(tdata_all, random_state=0)

# ---- plot ----
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
Z_all = tdata_all.obsm["X_umap"]
Z_dmap = tdata.obsm["X_diffmap"][:, 1:3]
dpt = tdata.obs["dpt_pseudotime"].values
rng0 = np.random.default_rng(0)
idx_all = rng0.choice(tdata_all.n_obs, size=min(15000, tdata_all.n_obs), replace=False)
idx     = np.random.default_rng(1).choice(tdata.n_obs, size=min(15000, tdata.n_obs), replace=False)

# (a) UMAP all
ax = fig.add_subplot(gs[0, 0])
for ct in T_CELLS_ALL:
    m = (tdata_all.obs["cell_type"].values == ct)
    mi = idx_all[m[idx_all]]
    is_gd = (ct == "γδ T cell")
    ax.scatter(Z_all[mi,0], Z_all[mi,1],
               s=1.2 if is_gd else 1.0, c=TCELL_PAL[ct],
               alpha=0.75 if is_gd else 0.5, edgecolor="none",
               label=ct + (" (off-traj.)" if is_gd else ""))
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("UMAP-1", fontsize=6); ax.set_ylabel("UMAP-2", fontsize=6)
ax.set_title("T cell UMAP", fontsize=7.5)
ax.legend(fontsize=4.8, markerscale=3, loc="upper right", frameon=False,
          handlelength=1, handletextpad=0.3, borderpad=0.2, labelspacing=0.25)

# (b) diffusion map by cell type
ax = fig.add_subplot(gs[0, 1])
for ct in T_CELLS_TRAJ:
    m = (tdata.obs["cell_type"].values == ct)
    mi = idx[m[idx]]
    ax.scatter(Z_dmap[mi,0], Z_dmap[mi,1], s=1.0,
               c=TCELL_PAL[ct], alpha=0.5, edgecolor="none")
ax.scatter(Z_dmap[root_local,0], Z_dmap[root_local,1], s=70,
           facecolor="none", edgecolor="#C0392B", lw=1.5, zorder=4)
ax.annotate("root", xy=(Z_dmap[root_local,0], Z_dmap[root_local,1]),
            xytext=(12, 12), textcoords="offset points", fontsize=6,
            color="#C0392B",
            arrowprops=dict(arrowstyle="-", color="#C0392B", lw=0.6))
ax.set_xlabel("DC1", fontsize=6); ax.set_ylabel("DC2", fontsize=6)
ax.set_title("Diffusion map · cell type", fontsize=7.5)
ax.set_xticks([]); ax.set_yticks([])

# (c) diffusion map by pseudotime
ax = fig.add_subplot(gs[0, 2])
sc_im = ax.scatter(Z_dmap[idx,0], Z_dmap[idx,1], s=1.0,
                    c=dpt[idx], cmap="plasma",
                    alpha=0.7, edgecolor="none")
ax.set_xlabel("DC1", fontsize=6); ax.set_ylabel("DC2", fontsize=6)
ax.set_title("Diffusion map · pseudotime", fontsize=7.5)
ax.set_xticks([]); ax.set_yticks([])
cax = fig.add_axes([0.92, 0.60, 0.010, 0.22])
cb = fig.colorbar(sc_im, cax=cax); cb.set_label("DPT pseudotime", fontsize=6)
cb.ax.tick_params(labelsize=5)

# (d) PAGA — FIXED layout
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
pos = pos - pos.mean(0)
pos = pos / max(np.abs(pos).max(), 1e-8) * 0.82

for i in range(len(group_names)):
    for j in range(i+1, len(group_names)):
        w = conn[i, j]
        if w > 0.05:
            ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]],
                    color="#888", lw=0.4+4*w, alpha=0.75, zorder=1)
sizes = tdata.obs["cell_type"].value_counts().reindex(group_names).values
sizes_sc = 150 + sizes / sizes.max() * 700
for i, ct in enumerate(group_names):
    ax.scatter(pos[i,0], pos[i,1], s=sizes_sc[i],
               c=TCELL_PAL[ct], edgecolor="black", linewidth=0.7, zorder=3)
# offset labels outward
centroid = pos.mean(0)
for i, ct in enumerate(group_names):
    x, y = pos[i]
    # direction: push away from centroid
    vx, vy = x - centroid[0], y - centroid[1]
    n = np.sqrt(vx*vx + vy*vy) + 1e-8
    ox, oy = vx/n * 0.16, vy/n * 0.16
    # fallback if node is at centroid
    if n < 0.05:
        ox, oy = 0.0, 0.18
    ax.text(x + ox, y + oy, ct, ha="center",
            va="center", fontsize=6.5, color="#111",
            zorder=5, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none",
                      alpha=0.85, pad=1.5))

# γδ off-trajectory chip
gd_n = int((tdata_all.obs['cell_type'] == "γδ T cell").sum())
ax.scatter([1.22], [0.0], s=200, c=TCELL_PAL["γδ T cell"],
           edgecolor="black", linewidth=0.7, zorder=3)
ax.text(1.22, 0.24, f"γδ T\n(n = {gd_n:,})\noff-trajectory",
        fontsize=6, color=TCELL_PAL["γδ T cell"], ha="center", va="bottom",
        fontweight="bold", zorder=5)
ax.plot([0.95, 1.10], [0.0, 0.0], ls="--", color="#bbb", lw=0.6, zorder=1)

ax.set_xticks([]); ax.set_yticks([])
ax.set_xlim(-1.18, 1.58)
ax.set_ylim(-1.18, 1.18)
ax.set_title("PAGA connectivity (γδ excluded as independent lineage)",
             fontsize=7.5, pad=3)
for s in ax.spines.values(): s.set_visible(False)

# (e) pseudotime violin
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
    pc.set_facecolor(TCELL_PAL[parts_labels[i]]); pc.set_alpha(0.6)
    pc.set_edgecolor(TCELL_PAL[parts_labels[i]]); pc.set_linewidth(0.6)
for pn in ["cmedians","cbars","cmins","cmaxes"]:
    if pn in parts:
        parts[pn].set_color("#333"); parts[pn].set_lw(0.5)
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
print("Saved 29_T_cell_pseudotime.pdf/.png")
