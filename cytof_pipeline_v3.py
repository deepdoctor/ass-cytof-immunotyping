"""
CyTOF Pipeline v3 — Correct panel mapping + smart channel filtering
=====================================================================
Panel: 42 biological markers (see panel.csv)
Groups: Jo-1 / PL-12 / EJ / PL-7
"""

import os, sys, warnings, re
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

try:
    import fcsparser, anndata as ad, scanpy as sc
except ImportError as e:
    sys.exit(f"[ERROR] {e}\nRun: pip install -r requirements.txt\n")

sc.settings.verbosity = 1
sc.settings.set_figure_params(dpi=150, facecolor="white")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

FCS_DIR      = "/Users/yichen/Desktop/collaboration/raw"
PANEL_CSV    = "/Users/yichen/Desktop/collaboration/cytof/zip_cytof_res/Panel_csv/panel.csv"
OUT_DIR      = "./cytof_output_v3"
COFACTOR     = 5
N_CELLS_MAX  = 10_000
RESOLUTION   = 0.8
RANDOM_STATE = 42

SAMPLE_GROUP = {
    "L01912_FH0002": "Jo-1",
    "L01912_FH0003": "Jo-1",
    "L01912_FH0014": "Jo-1",
    "L01912_FH0009": "PL-12",
    "L01912_FH0013": "PL-12",
    "L01912_FH0015": "PL-12",
    "L01912_FH0011": "EJ",
    "L01912_FH0016": "EJ",
    "L01912_FH0012": "PL-7",
    "L01912_FH0017": "PL-7",
}

GROUP_PALETTE = {
    "Jo-1":  "#E64B35",
    "PL-12": "#4DBBD5",
    "EJ":    "#00A087",
    "PL-7":  "#3C5488",
}

# ── Panel: 42 biological markers ─────────────────────────────────────────────
# channel → clean marker name
PANEL = {
    "89Y":   "CD45",
    "115In": "CD3",
    "141Pr": "Ki-67",
    "142Nd": "CD22",
    "143Nd": "IgD",
    "144Nd": "CD14",
    "145Nd": "CD20",
    "146Nd": "CD38",
    "147Sm": "CCR7",
    "148Nd": "CXCR5",
    "149Sm": "CD25",
    "150Nd": "CD56",
    "151Eu": "ICOS",
    "152Sm": "CD11c",
    "153Eu": "Tim-3",
    "154Sm": "TIGIT",
    "155Gd": "CD45RB",
    "156Gd": "TCRgd",
    "157Gd": "CD39",
    "158Gd": "CCR4",
    "159Tb": "CD16",
    "160Gd": "TCF1",
    "161Dy": "CD73",
    "162Dy": "CD123",
    "163Dy": "CXCR3",
    "164Dy": "CD45RA",
    "165Ho": "CD95",
    "166Er": "CD69",
    "167Er": "CD27",
    "168Er": "IgM",
    "169Tm": "CD45RO",
    "170Er": "CD127",
    "171Yb": "PD-1",
    "172Yb": "CD19",
    "173Yb": "GranzymeB",
    "174Yb": "CCR6",
    "175Lu": "CD24",
    "176Yb": "HLA-DR",
    "195Pt": "CD66b",
    "197Au": "CD4",
    "198Pt": "CD8a",
    "209Bi": "CD11b",
}
BIO_MARKERS = list(PANEL.values())   # 42 clean names

os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# HELPER: match FCS column names → clean marker names
# ══════════════════════════════════════════════════════════════════════════════

def normalize(s):
    """Lowercase, strip non-alphanumeric for fuzzy matching."""
    return re.sub(r"[^a-z0-9]", "", s.lower())

# CD number aliases used in some FCS files instead of protein names
CD_ALIASES = {
    "CD183": "CXCR3",
    "CD185": "CXCR5",
    "CD194": "CCR4",
    "CD196": "CCR6",
    "CD197": "CCR7",
    "CD278": "ICOS",
    "CD279": "PD-1",
    "CD366": "Tim-3",
}

def build_rename_map(fcs_columns):
    """
    Match FCS column names → clean marker names using:
      1. Direct marker name match (strip _plt suffix, handle underscores)
      2. CD number alias match (e.g. CD183 → CXCR3)
      3. Channel code match (e.g. '89Y' in column name)
    Returns: {old_col: clean_marker_name}
    """
    rename = {}
    # Build a lookup: normalized marker name → marker
    marker_lookup = {normalize(v): v for v in PANEL.values()}

    # Build channel code lookup (both forward and reversed notation)
    channel_lookup = {}
    for ch, marker in PANEL.items():
        ch_norm = normalize(ch)
        channel_lookup[ch_norm] = marker
        ch_rev = re.sub(r"^(\d+)([a-z]+)$", r"\2\1", ch_norm)
        if ch_rev != ch_norm:
            channel_lookup[ch_rev] = marker

    # Build CD alias lookup: normalized alias → marker
    alias_lookup = {normalize(k): v for k, v in CD_ALIASES.items()}

    for col in fcs_columns:
        # Skip known technical/barcode channels
        col_up = col.upper()
        if any(x in col_up for x in ["BARCODE", "DNA", "CISPLATIN",
                                       "EVENT", "TIME", "CENTER",
                                       "OFFSET", "WIDTH", "RESIDUAL",
                                       "CELL_ID"]):
            continue

        # Clean the column name: strip _plt suffix, replace _ with nothing
        col_clean = re.sub(r"_plt$", "", col)
        col_norm = normalize(col_clean)

        matched = None

        # 1. Exact marker name match (e.g. 'cd45ra' → 'CD45RA')
        if col_norm in marker_lookup:
            matched = marker_lookup[col_norm]

        # 2. CD alias match (e.g. 'cd183' → 'CXCR3')
        if matched is None and col_norm in alias_lookup:
            matched = alias_lookup[col_norm]

        # 3. Special cases: underscored names → marker
        if matched is None:
            # Handle names like "Granzyme_B_Recombinant" → "granzymeb"
            # and "HLA_DR" → "hladr", "Ki_67" → "ki67", "TCR_g_d_plt" → "tcrgd"
            for m_norm, marker in marker_lookup.items():
                if col_norm == m_norm:
                    matched = marker
                    break
            # Try removing common suffixes: "recombinant", etc.
            if matched is None:
                col_stripped = re.sub(r"(recombinant|plt)$", "", col_norm)
                col_stripped = col_stripped.rstrip("_")
                if col_stripped in marker_lookup:
                    matched = marker_lookup[col_stripped]

        # 4. Channel code match (e.g. column "89Y_CD45" → channel "89Y")
        if matched is None:
            for ch_norm, marker in channel_lookup.items():
                if ch_norm in col_norm:
                    # Verify this isn't a Pd barcode channel falsely matching
                    if len(ch_norm) >= 3:  # Only match if channel code is specific enough
                        matched = marker
                        break

        if matched is not None and col not in rename:
            rename[col] = matched

    return rename

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1. LOAD + RENAME COLUMNS
# ══════════════════════════════════════════════════════════════════════════════

def load_all_fcs(fcs_dir):
    fcs_files = sorted(f for f in os.listdir(fcs_dir) if f.lower().endswith(".fcs"))
    if not fcs_files:
        sys.exit(f"[ERROR] No .fcs files in {fcs_dir}")

    print(f"\n[Step 1] Loading {len(fcs_files)} FCS files …")
    adatas, rename_map_global = [], {}

    for fname in fcs_files:
        sid = fname.replace(".exported.FCS3.fcs","").replace(".fcs","")
        fpath = os.path.join(fcs_dir, fname)
        try:
            _, data = fcsparser.parse(fpath, reformat_meta=True)
        except Exception as ex:
            print(f"  [WARN] {fname}: {ex}"); continue

        data = data.select_dtypes(include=[np.number])

        # Build rename map from first file, reuse for rest
        if not rename_map_global:
            rename_map_global = build_rename_map(data.columns.tolist())

            print(f"\n  Channel → Marker mapping ({len(rename_map_global)} matched):")
            for old_col, new_marker in sorted(rename_map_global.items()):
                print(f"    {old_col:35s} → {new_marker}")
            unmapped = [c for c in data.columns if c not in rename_map_global]
            missing_markers = set(PANEL.values()) - set(rename_map_global.values())
            print(f"\n  Unmapped columns: {unmapped}")
            print(f"  Missing markers:  {sorted(missing_markers)}\n")

        # Keep only panel columns, rename
        keep = {k: v for k, v in rename_map_global.items() if k in data.columns}
        data = data[list(keep.keys())].rename(columns=keep)

        # Remove duplicate marker columns (keep first)
        data = data.loc[:, ~data.columns.duplicated()]

        if N_CELLS_MAX and len(data) > N_CELLS_MAX:
            data = data.sample(N_CELLS_MAX, random_state=RANDOM_STATE)

        adata = ad.AnnData(X=data.values.astype(np.float32),
                           var=pd.DataFrame(index=data.columns))
        adata.obs["sample"] = sid
        adata.obs["group"]  = SAMPLE_GROUP.get(sid, "Unknown")
        adatas.append(adata)
        print(f"  {sid}: {len(data):,} cells → {SAMPLE_GROUP.get(sid,'?')}")

    adata_all = ad.concat(adatas, join="inner")
    adata_all.var_names_make_unique()
    print(f"\n  Total: {adata_all.n_obs:,} cells | {adata_all.n_vars} markers")
    return adata_all

adata = load_all_fcs(FCS_DIR)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2. QC + arcsinh
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 2] QC + arcsinh …")
adata.layers["raw"] = adata.X.copy()
adata.obs["total_signal"] = adata.X.sum(axis=1)
lo = np.percentile(adata.obs["total_signal"], 1)
hi = np.percentile(adata.obs["total_signal"], 99.5)
n_b = adata.n_obs
adata = adata[(adata.obs["total_signal"] >= lo) & (adata.obs["total_signal"] <= hi)].copy()
print(f"  QC: {n_b - adata.n_obs:,} removed → {adata.n_obs:,} retained")
adata.X = np.arcsinh(adata.X / COFACTOR)

# Cell count bar plot
count_df = adata.obs.groupby(["sample","group"]).size().reset_index(name="n")
fig, ax = plt.subplots(figsize=(10, 4))
bar_cols = [GROUP_PALETTE.get(g,"gray") for g in count_df["group"]]
ax.bar(count_df["sample"], count_df["n"], color=bar_cols, edgecolor="k", linewidth=0.5)
ax.set_ylabel("Cell count"); ax.set_title("Cells per sample after QC")
plt.xticks(rotation=45, ha="right", fontsize=8)
from matplotlib.patches import Patch
ax.legend([Patch(color=c, label=g) for g, c in GROUP_PALETTE.items()],
          [g for g in GROUP_PALETTE], title="Group", bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/00_cell_counts.pdf", bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3. PCA → UMAP → LEIDEN
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 3] PCA → UMAP → Leiden …")
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, n_comps=min(30, adata.n_vars-1), random_state=RANDOM_STATE)

# ── Harmony batch correction (removes inter-sample technical variation) ────
try:
    import harmonypy
    print("  Running Harmony batch correction on 'sample' …")
    ho = harmonypy.run_harmony(
        adata.obsm["X_pca"], adata.obs, "sample",
        max_iter_harmony=20, random_state=RANDOM_STATE
    )
    adata.obsm["X_pca_harmony"] = np.asarray(ho.Z_corr)
    use_rep = "X_pca_harmony"
    print("  Harmony done.")
except ImportError:
    print("  [WARN] harmonypy not installed — skipping batch correction.")
    print("         Install with: pip install harmonypy")
    use_rep = "X_pca"

sc.pp.neighbors(adata, n_neighbors=30, use_rep=use_rep, random_state=RANDOM_STATE)
sc.tl.umap(adata, random_state=RANDOM_STATE)
sc.tl.leiden(adata, resolution=RESOLUTION, random_state=RANDOM_STATE, key_added="leiden")
n_clusters = adata.obs["leiden"].nunique()
print(f"  Clusters: {n_clusters}")

umap = adata.obsm["X_umap"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for grp, col in GROUP_PALETTE.items():
    idx = adata.obs["group"] == grp
    axes[0].scatter(umap[idx,0], umap[idx,1], c=col, s=1.5, alpha=0.4,
                    label=grp, rasterized=True)
axes[0].set_title("UMAP — Group"); axes[0].axis("off")
axes[0].legend(markerscale=6, frameon=False)
cmap = plt.cm.get_cmap("tab10", adata.obs["sample"].nunique())
for i, sid in enumerate(sorted(adata.obs["sample"].unique())):
    idx = adata.obs["sample"] == sid
    axes[1].scatter(umap[idx,0], umap[idx,1], c=[cmap(i)], s=1.5, alpha=0.4,
                    label=sid, rasterized=True)
axes[1].set_title("UMAP — Sample"); axes[1].axis("off")
axes[1].legend(markerscale=5, frameon=False, fontsize=7)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_umap_group_sample.pdf", bbox_inches="tight", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(7, 6))
sc.pl.umap(adata, color="leiden", ax=ax, show=False, palette="tab20",
           title=f"Leiden clusters (res={RESOLUTION})")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_umap_clusters.pdf", bbox_inches="tight", dpi=150)
plt.close()
print("  Saved: 01 + 02 UMAP plots")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4. CLUSTER HEATMAP — clean layout, markers as rows
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 4] Cluster marker heatmap …")

X_raw = np.arcsinh(adata.layers["raw"] / COFACTOR)
cluster_mean = (
    pd.DataFrame(X_raw, columns=adata.var_names, index=adata.obs.index)
    .assign(cluster=adata.obs["leiden"].values)
    .groupby("cluster").mean()
)
cluster_mean.to_csv(f"{OUT_DIR}/03_cluster_marker_means.csv")

# Z-score per marker for better contrast
cluster_mean_z = cluster_mean.apply(
    lambda col: (col - col.mean()) / (col.std() + 1e-9)
)

# Sort markers by biological category for readability
marker_order = [
    # Lineage
    "CD45","CD3","CD4","CD8a","CD19","CD20","CD14","CD56","CD11c","CD11b","CD66b",
    # T cell
    "CCR7","CD45RA","CD45RO","CD45RB","CD95","CD25","CD127","CD69","CD27",
    # Tfh / activation
    "CXCR5","ICOS","CCR4","CCR6","CXCR3","CD73",
    # Checkpoint / exhaustion
    "PD-1","TIGIT","Tim-3","CD39","TCF1",
    # Cytotoxic
    "GranzymeB","TCRgd","CD16",
    # B cell
    "CD22","IgD","IgM","CD38","CD24","HLA-DR","CD123",
    # Other
    "Ki-67",
]
marker_order = [m for m in marker_order if m in cluster_mean_z.columns]
# append any remaining
marker_order += [m for m in cluster_mean_z.columns if m not in marker_order]

plot_data = cluster_mean_z[marker_order].T   # markers as rows, clusters as cols

n_markers = len(marker_order)
fig, ax = plt.subplots(figsize=(max(10, n_clusters*0.6), max(10, n_markers*0.35)))
ax.set_facecolor("#1A3A6B")
im = ax.imshow(plot_data.values, cmap="RdYlBu_r", aspect="auto",
               vmin=-2.5, vmax=2.5, interpolation="none", rasterized=True)
ax.set_xticks(range(n_clusters))
ax.set_xticklabels(plot_data.columns, fontsize=9, fontweight="bold")
ax.set_yticks(range(n_markers))
ax.set_yticklabels(plot_data.index, fontsize=8)
ax.set_xlabel("Cluster", fontsize=10)
ax.set_ylabel("Marker", fontsize=10)
ax.set_title("Cluster × Marker  [biological markers only, z-scored by marker]",
             fontsize=12, pad=12)
ax.tick_params(which="both", bottom=False, left=False)
for spine in ax.spines.values():
    spine.set_visible(False)
cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.02, shrink=0.6)
cbar.set_label("Z-score (arcsinh)", fontsize=9)
cbar.ax.tick_params(labelsize=8)
cbar.outline.set_linewidth(0)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_cluster_marker_heatmap.pdf", bbox_inches="tight", dpi=300)
plt.close()
print("  Saved: 03_cluster_marker_heatmap.pdf")
print("         03_cluster_marker_means.csv")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5. ANNOTATION
# ══════════════════════════════════════════════════════════════════════════════
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  Review 03_cluster_marker_heatmap.pdf, then fill in the dict below.        │
# │                                                                             │
# │  Your panel covers these cell types (key markers):                         │
# │                                                                             │
# │  T CELLS (CD3+)                                                             │
# │    CD4 Naive T         CD3+ CD4+  CD45RA+ CCR7+  CD95-                    │
# │    CD4 Central Memory  CD3+ CD4+  CD45RO+ CCR7+  CD95+                    │
# │    CD4 Effector Memory CD3+ CD4+  CD45RO+ CCR7-                           │
# │    Treg                CD3+ CD4+  CD25+   CD127-  CD45RO+                 │
# │    Tfh                 CD3+ CD4+  CXCR5+  ICOS+   CCR7-                   │
# │    Th1                 CD3+ CD4+  CXCR3+  CCR6-   CCR4-                   │
# │    Th2                 CD3+ CD4+  CCR4+   CXCR3-                          │
# │    Th17                CD3+ CD4+  CCR6+   CXCR3-                          │
# │    CD8 Naive T         CD3+ CD8+  CD45RA+ CCR7+                           │
# │    CD8 Effector T      CD3+ CD8+  GranzymeB+ CD69+                        │
# │    CD8 Exhausted T     CD3+ CD8+  PD-1+   TIGIT+ Tim-3+                   │
# │    CD8 Stem-like T     CD3+ CD8+  TCF1+   PD-1+  CD45RA+                  │
# │    γδ T cell           CD3+ TCRgd+                                         │
# │                                                                             │
# │  NK CELLS (CD3-)                                                            │
# │    CD56bright NK       CD3- CD56hi  CD16-  CD11b-                          │
# │    CD56dim NK          CD3- CD56lo  CD16+  CD11b+  GranzymeB+             │
# │                                                                             │
# │  B CELLS (CD19+)                                                            │
# │    Naive B             CD19+ CD20+ IgD+  IgM+  CD27-                      │
# │    Memory B            CD19+ CD20+ IgD-  CD27+                             │
# │    Plasma cell         CD19+ CD38hi  CD20-  CD27+                          │
# │    Transitional B      CD19+ CD20+ CD38+  CD24hi                           │
# │                                                                             │
# │  MYELOID (CD3- CD19-)                                                       │
# │    Classical Mono      CD14+  CD16-  HLA-DR+ CD11c+                       │
# │    Non-classical Mono  CD14lo CD16+  HLA-DR+ CD11b+                       │
# │    mDC                 HLA-DR+ CD11c+  CD14-  CD123-                      │
# │    pDC                 HLA-DR+ CD123+  CD11c- CD14-                       │
# │    Granulocyte         CD66b+  CD11b+                                      │
# └─────────────────────────────────────────────────────────────────────────────┘

CLUSTER_ANNOTATION = {
    # "0":  "CD4 Naive T",
    # "1":  "CD4 Central Memory T",
    # "2":  "CD8 Effector T",
    # "3":  "Naive B",
    # "4":  "Classical Monocyte",
    # "5":  "NK",
    # ...
}

if CLUSTER_ANNOTATION:
    adata.obs["cell_type"] = (
        adata.obs["leiden"].map(CLUSTER_ANNOTATION).fillna("Unannotated")
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    sc.pl.umap(adata, color="cell_type", ax=ax, show=False,
               title="UMAP — Cell type annotation")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/04_umap_celltype.pdf", bbox_inches="tight", dpi=150)
    plt.close()

    # Dotplot with canonical markers
    canonical = [m for m in [
        "CD3","CD4","CD8a","CD45RA","CD45RO","CCR7","CD25","CD127",
        "CXCR5","ICOS","GranzymeB","PD-1","TIGIT","Tim-3","TCF1",
        "CD19","CD20","IgD","IgM","CD38","CD27",
        "CD14","CD16","CD11c","CD123","HLA-DR","CD66b","CD56","TCRgd","Ki-67"
    ] if m in adata.var_names]
    sc.pl.dotplot(adata, var_names=canonical, groupby="cell_type",
                  show=False, figsize=(16, 7),
                  title="Canonical marker expression by cell type")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/04b_dotplot_celltype.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved: 04_umap_celltype.pdf + 04b_dotplot_celltype.pdf")
else:
    print("  [INFO] Fill CLUSTER_ANNOTATION and re-run from Step 5")
    adata.obs["cell_type"] = "Cluster_" + adata.obs["leiden"].astype(str)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6. DIFFERENTIAL ABUNDANCE
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 6] Differential abundance …")
prop_df = (
    adata.obs.groupby(["sample","group","cell_type"])
    .size().reset_index(name="count")
)
prop_df["proportion"] = prop_df["count"] / \
    prop_df.groupby("sample")["count"].transform("sum")
prop_df.to_csv(f"{OUT_DIR}/05_cell_proportions.csv", index=False)

cell_types = sorted(prop_df["cell_type"].unique())
ncols = min(4, len(cell_types))
nrows = int(np.ceil(len(cell_types) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3.5, nrows*3.5), squeeze=False)
order = list(GROUP_PALETTE.keys())
for idx, ct in enumerate(cell_types):
    ax = axes[idx//ncols][idx%ncols]
    sub = prop_df[prop_df["cell_type"] == ct]
    palette = [GROUP_PALETTE.get(g,"gray") for g in order]
    sns.boxplot(data=sub, x="group", y="proportion", order=order,
                palette=palette, ax=ax, width=0.5)
    sns.stripplot(data=sub, x="group", y="proportion", order=order,
                  color="black", size=4, ax=ax, alpha=0.8)
    ax.set_title(ct, fontsize=9); ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30, labelsize=7)
for idx in range(len(cell_types), nrows*ncols):
    axes[idx//ncols][idx%ncols].set_visible(False)
plt.suptitle("Cell proportion by group", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/05_differential_abundance.pdf", bbox_inches="tight")
plt.close()
print("  Saved: 05_differential_abundance.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7. DIFFERENTIAL MARKER EXPRESSION  (Kruskal-Wallis, FDR)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 7] Differential marker expression …")
X_df = pd.DataFrame(np.arcsinh(adata.layers["raw"] / COFACTOR),
                    columns=adata.var_names, index=adata.obs.index)
X_df["group"] = adata.obs["group"].values
groups = list(GROUP_PALETTE.keys())
results = []
for marker in adata.var_names:
    vals = [X_df.loc[X_df["group"]==g, marker].values for g in groups]
    vals = [v for v in vals if len(v) > 0]
    if len(vals) < 2: continue
    stat, pval = stats.kruskal(*vals)
    means = {f"mean_{g}": X_df.loc[X_df["group"]==g, marker].mean() for g in groups}
    results.append({"marker": marker, "KW_stat": stat, "pval": pval, **means})
res_df = pd.DataFrame(results)
_, res_df["padj"], _, _ = multipletests(res_df["pval"], method="fdr_bh")
res_df = res_df.sort_values("padj")
res_df.to_csv(f"{OUT_DIR}/06_differential_markers.csv", index=False)

sig = res_df[res_df["padj"] < 0.05]["marker"].tolist() or res_df.head(20)["marker"].tolist()
mean_grp = X_df.groupby("group")[sig].mean().T.reindex(columns=groups)
fig, ax = plt.subplots(figsize=(8, max(6, len(sig)*0.35)))
n_sig_markers = len(sig)
n_grps = len(groups)
ax.set_facecolor("#1A3A6B")
im2 = ax.imshow(mean_grp.values, cmap="RdYlBu_r", aspect="auto",
                vmin=mean_grp.values.min(), vmax=mean_grp.values.max(),
                interpolation="none", rasterized=True)
ax.set_xticks(range(n_grps))
ax.set_xticklabels(groups, fontsize=10, fontweight="bold")
ax.set_yticks(range(n_sig_markers))
ax.set_yticklabels(sig, fontsize=8)
ax.set_title("Differential markers between groups (FDR<0.05)", fontsize=11)
ax.tick_params(which="both", bottom=False, left=False)
for spine in ax.spines.values():
    spine.set_visible(False)
cbar2 = fig.colorbar(im2, ax=ax, fraction=0.06, pad=0.02, shrink=0.6)
cbar2.set_label("Mean arcsinh", fontsize=9)
cbar2.outline.set_linewidth(0)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/06_differential_marker_heatmap.pdf", bbox_inches="tight", dpi=300)
plt.close()

top9 = res_df.head(9)["marker"].tolist()
fig, axes = plt.subplots(3, 3, figsize=(13, 10))
for i, marker in enumerate(top9):
    ax = axes[i//3][i%3]
    palette = [GROUP_PALETTE.get(g,"gray") for g in order]
    sns.violinplot(data=X_df[["group",marker]], x="group", y=marker,
                   order=order, palette=palette, ax=ax, inner="box", cut=0)
    padj_v = res_df.loc[res_df["marker"]==marker,"padj"].values[0]
    ax.set_title(f"{marker}\nFDR={padj_v:.2e}", fontsize=9)
    ax.set_xlabel(""); ax.tick_params(axis="x", rotation=30, labelsize=7)
for i in range(len(top9), 9):
    axes[i//3][i%3].set_visible(False)
plt.suptitle("Top differential markers", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/07_top_marker_violins.pdf", bbox_inches="tight")
plt.close()
print("  Saved: 06 + 07 differential marker plots")

# SAVE
adata.write_h5ad(f"{OUT_DIR}/cytof_analyzed_v3.h5ad")

print(f"""
══════════════════════════════════════════════════════════
  Done!  Output: {OUT_DIR}/
══════════════════════════════════════════════════════════
  00_cell_counts.pdf
  01_umap_group_sample.pdf
  02_umap_clusters.pdf
  03_cluster_marker_heatmap.pdf   ← review this for annotation
  03_cluster_marker_means.csv
  05_differential_abundance.pdf
  06_differential_markers.csv
  06_differential_marker_heatmap.pdf
  07_top_marker_violins.pdf
  cytof_analyzed_v3.h5ad

  NEXT: fill CLUSTER_ANNOTATION dict (~line 165) → re-run
""")
