"""
Native rebuild of Fig 4(c): integrated clinical-immune heatmap with
bilateral dendrograms.

Fixes vs prior render:
  - bottom cell-type labels were horizontal and crammed together → rotated
    30° with sufficient padding so 14 cell types are individually readable.
  - patient labels formatted as 'P08 (Jo-1)' instead of 'P08_Jo-1' (no
    underscores in publication labels).
  - left annotation strips: autoantibody + immunotype, with white separator.

Builds the patient feature vector ('14 cell-type proportions + 9 signature
scores') and the clinical-variable matrix ('age, CK, LDH, ESR, CRP, FVC,
DLCO'), z-scores each, and clusters rows (patients) and columns (features)
with Ward linkage.
"""
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from sklearn.preprocessing import StandardScaler

OUT_DIR = "./cytof_output_nature"

plt.rcParams.update({
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":            10.0,
    "axes.titlesize":       11.5,
    "axes.labelsize":       10.0,
    "xtick.labelsize":      9.0,
    "ytick.labelsize":      9.5,
    "axes.linewidth":       0.9,
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
    "savefig.dpi":          300,
    "figure.dpi":           300,
})

GROUP_PALETTE = {"Jo-1":  "#C0392B", "PL-12": "#2471A3",
                 "EJ":    "#1A7A4A", "PL-7":  "#7D3C98"}
IT_COL = {"IT1": "#E67E22", "IT2": "#2980B9", "IT3": "#27AE60"}

CT_FEATURES = [
    "CD4 Central Memory T", "CD4 Effector Memory T", "CD4 Naive T", "CD4 Th1-like",
    "CD8 Effector T", "CD8 Naive T",
    "Classical Monocyte", "Inflammatory Monocyte", "Non-classical Monocyte",
    "NK cell", "Naive B", "mDC", "pDC", "γδ T cell",
]
SIGNATURE_FEATURES = [
    "Th1_score", "Th17_score", "Tfh_score", "Treg_score",
    "Cytotoxic_score", "Exhaustion_score", "Activation_score",
    "Naive_score", "Memory_score",
]
CLINICAL_FEATURES = [
    "age", "CK", "LDH", "ESR", "CRP", "FVC", "DLCO",
]

# ─── load ───────────────────────────────────────────────────────────────────
feat = pd.read_csv(f"{OUT_DIR}/SuppTable_feature_matrix.csv")
feat = feat.drop(columns=[c for c in feat.columns if c == "name"])
feat["pid_label"] = feat["pid"].apply(lambda x: f"P{int(x):02d}")
clin = pd.read_csv(f"{OUT_DIR}/23_immunotype_clinical_table.csv")
clin = clin.drop(columns=[c for c in clin.columns if c == "name"])
clin = clin.set_index("pid")

# Patient × feature matrix (proportions + signature scores + clinical)
features_avail = (
    [c for c in CT_FEATURES if c in feat.columns]
    + [c for c in SIGNATURE_FEATURES if c in feat.columns]
)
M_immune = feat.set_index("pid")[features_avail].copy()

# Clinical features per patient (CK/LDH log-transformed)
M_clin = clin[CLINICAL_FEATURES].copy()
for c in ["CK", "LDH"]:
    if c in M_clin.columns:
        M_clin[c] = np.log10(M_clin[c].astype(float) + 1)

# z-score each column independently (NaN-safe)
def col_zscore(df):
    out = df.copy().astype(float)
    for c in out.columns:
        v = out[c].values
        mu = np.nanmean(v); sd = np.nanstd(v)
        out[c] = (v - mu) / (sd + 1e-9)
    return out

Z_immune = col_zscore(M_immune)
Z_clin   = col_zscore(M_clin)

# Combine: concatenate columns
M_all = pd.concat([Z_immune, Z_clin], axis=1)
print(f"Patients: {M_all.shape[0]}; features: {M_all.shape[1]} "
      f"({Z_immune.shape[1]} immune + {Z_clin.shape[1]} clinical)")

# Cluster rows (patients) and columns (features) with Ward linkage
# Replace NaN with 0 just for clustering (patient-level rare missing)
M_for_cluster = M_all.fillna(0).values
Z_row = linkage(M_for_cluster, method="ward")
Z_col = linkage(M_for_cluster.T, method="ward")
row_order = leaves_list(Z_row)
col_order = leaves_list(Z_col)

M_disp = M_all.iloc[row_order, col_order]

# Patient/feature labels
pids = list(M_disp.index)
features = list(M_disp.columns)

# Patient annotations
ab_strip = [feat.set_index("pid").loc[p, "antibody"] for p in pids]
it_strip = [feat.set_index("pid").loc[p, "immunotype"] for p in pids]
patient_labels = [f"P{int(p):02d} ({ab})" for p, ab in zip(pids, ab_strip)]

# Feature labels: distinguish immune vs clinical (color tag)
def is_clinical(f):
    return f in CLINICAL_FEATURES
def is_signature(f):
    return f in SIGNATURE_FEATURES

# ─── figure ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12.0, 7.4), facecolor="white")
gs = gridspec.GridSpec(
    2, 4, figure=fig,
    left=0.06, right=0.985, top=0.945, bottom=0.275,
    wspace=0.020, hspace=0.020,
    height_ratios=[0.7, 5.0],
    width_ratios=[1.4, 0.18, 0.18, 7.5],
)

ax_col_dn = fig.add_subplot(gs[0, 3])
ax_row_dn = fig.add_subplot(gs[1, 0])
ax_ab     = fig.add_subplot(gs[1, 1])
ax_it     = fig.add_subplot(gs[1, 2])
ax_h      = fig.add_subplot(gs[1, 3])

# ── column dendrogram (top) ───────────────────────────────────────────────
dendrogram(Z_col, ax=ax_col_dn, no_labels=True, color_threshold=0,
           above_threshold_color="#666")
for s in ax_col_dn.spines.values(): s.set_visible(False)
ax_col_dn.set_xticks([]); ax_col_dn.set_yticks([])

# ── row dendrogram (left) ─────────────────────────────────────────────────
dendrogram(Z_row, ax=ax_row_dn, orientation="left", no_labels=True,
           color_threshold=0, above_threshold_color="#666")
ax_row_dn.invert_yaxis()
for s in ax_row_dn.spines.values(): s.set_visible(False)
ax_row_dn.set_xticks([]); ax_row_dn.set_yticks([])

# ── annotation strips ─────────────────────────────────────────────────────
n_pat = len(pids)
for ax_strip, vals, palette, label in [
    (ax_ab, ab_strip, GROUP_PALETTE, "antibody"),
    (ax_it, it_strip, IT_COL,        "immunotype"),
]:
    for i, v in enumerate(vals):
        ax_strip.add_patch(Rectangle((0, i), 1, 1,
                                      facecolor=palette.get(v, "#888"),
                                      edgecolor="white", lw=0.6))
    ax_strip.set_xlim(0, 1)
    ax_strip.set_ylim(0, n_pat)
    ax_strip.invert_yaxis()
    ax_strip.set_xticks([])
    ax_strip.set_yticks([])
    for s in ax_strip.spines.values(): s.set_visible(False)
    # column header
    ax_strip.text(0.5, -0.6, label, ha="center", va="bottom",
                  fontsize=8.5, color="#333", rotation=30,
                  rotation_mode="anchor")

# ── main heatmap ──────────────────────────────────────────────────────────
HEAT = LinearSegmentedColormap.from_list(
    "heat_div",
    ["#1A3A6B", "#2980B9", "#7FB3D3", "#F4D03F", "#E67E22", "#C0392B", "#7B241C"],
    N=256,
)
im = ax_h.imshow(M_disp.values, cmap=HEAT, vmin=-2, vmax=2,
                 aspect="auto")
ax_h.set_xticks(range(len(features)))
xtick_labels = []
for f in features:
    if is_clinical(f):
        xtick_labels.append(f)  # plain
    elif is_signature(f):
        xtick_labels.append(f.replace("_score", " score"))
    else:
        xtick_labels.append(f)
ax_h.set_xticklabels(xtick_labels, rotation=30, ha="right",
                      fontsize=8.7)
# colour-code feature labels
for i, (f, tick) in enumerate(zip(features, ax_h.get_xticklabels())):
    if is_clinical(f):
        tick.set_color("#000")
        tick.set_fontweight("bold")
    elif is_signature(f):
        tick.set_color("#7D3C98")
    else:
        tick.set_color("#1F618D")
ax_h.set_yticks(range(n_pat))
ax_h.set_yticklabels(patient_labels, fontsize=9.5)
# move y-axis labels to the right
ax_h.yaxis.tick_right()
ax_h.tick_params(left=False)

# colour bar (above figure / outside)
cbar_ax = fig.add_axes([0.92, 0.92, 0.06, 0.022])
cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
cbar.set_label("Z-score", fontsize=8.5)
cbar.ax.tick_params(labelsize=7.5)
cbar.outline.set_linewidth(0.4)

# Combined legend: rows (immunotype + autoantibody) and columns (feature type)
# rendered as a single coherent block so a reader sees what every colour means.
# Use Line2D for feature-label colour key (small dashes match the actual styling).
from matplotlib.lines import Line2D

row_handles = [
    Patch(facecolor="white", edgecolor="white", label=r"$\bf{Patient\ rows:}$"),
    *[Patch(facecolor=GROUP_PALETTE[g], label=f"Ab class: {g}")
      for g in ["Jo-1", "PL-12", "EJ", "PL-7"]],
    *[Patch(facecolor=IT_COL[t], label=f"Immunotype: {t}")
      for t in ["IT1", "IT2", "IT3"]],
]
col_handles = [
    Patch(facecolor="white", edgecolor="white",
          label=r"$\bf{Feature\ columns\ (label\ colour):}$"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#1F618D",
           markersize=10, label="cell-type proportion (blue label)"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#7D3C98",
           markersize=10, label="composite signature (purple label)"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#000",
           markersize=10, label="clinical variable (black bold label)"),
]
fig.legend(handles=row_handles, loc="lower left",
           bbox_to_anchor=(0.03, 0.005), frameon=False,
           ncol=4, fontsize=8.8, handlelength=1.0, handletextpad=0.4,
           columnspacing=1.0, borderpad=0.2)
fig.legend(handles=col_handles, loc="lower right",
           bbox_to_anchor=(0.985, 0.005), frameon=False,
           ncol=2, fontsize=8.8, handlelength=1.0, handletextpad=0.4,
           columnspacing=1.0, borderpad=0.2)

fig.suptitle("Integrated clinical-immune heatmap with bilateral dendrograms",
             fontsize=12, fontweight="bold", y=0.995)

out_pdf = f"{OUT_DIR}/15_clinical_immune_heatmap.pdf"
out_png = f"{OUT_DIR}/15_clinical_immune_heatmap.png"
fig.savefig(out_pdf, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
fig.savefig(out_png, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
