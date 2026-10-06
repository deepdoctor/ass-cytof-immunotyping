"""
Native rebuild of Fig 4(a): immunotype discovery dendrogram + clinical strip.

The original render had two cosmetic issues exposed at composite scale:
  1. patient IDs P01..P10 sat on top of the antibody color box and were
     visually truncated to 'P0_' once the figure was downsized into the
     MainFig4 composite (3-char label crammed into a 2-char visual slot).
  2. the leaf labels were also too small at composite resolution.

This rebuild:
  - recomputes Ward linkage on the standardised 65-D feature vector,
    matching the immunotype assignments in 21_immunotype_assignments.csv
  - puts patient IDs OUTSIDE the clinical annotation block, with
    explicit horizontal padding so they cannot be visually clipped
  - renders the dendrogram with immunotype-coloured branches
    (IT1 orange, IT2 blue, IT3 green)
  - stacks the clinical-annotation strip to the RIGHT of the dendrogram
  - saves over 21_immunotype_discovery.png so the existing composite
    pipeline picks it up.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, Patch
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

OUT_DIR = "./cytof_output_nature"

plt.rcParams.update({
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":            10.0,
    "axes.titlesize":       12.0,
    "axes.labelsize":       10.0,
    "xtick.labelsize":      9.0,
    "ytick.labelsize":      9.0,
    "legend.fontsize":      9.0,
    "axes.linewidth":       1.0,
    "xtick.major.width":    0.9,
    "ytick.major.width":    0.9,
    "xtick.major.size":     4.0,
    "ytick.major.size":     4.0,
    "lines.linewidth":      1.4,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
    "savefig.dpi":          300,
    "figure.dpi":           300,
})

# ─── load ───────────────────────────────────────────────────────────────────
feat = pd.read_csv(f"{OUT_DIR}/SuppTable_feature_matrix.csv")
feat = feat.drop(columns=[c for c in feat.columns if c == "name"])
clin = pd.read_csv(f"{OUT_DIR}/23_immunotype_clinical_table.csv")
clin = clin.drop(columns=[c for c in clin.columns if c == "name"])

# Build 65-D feature vector
meta_cols = ["sample", "pid", "antibody", "immunotype"]
X = feat.drop(columns=meta_cols).values
Xs = StandardScaler().fit_transform(X)

# Ward linkage; cut at k = 3
Z = linkage(Xs, method="ward")
labels = fcluster(Z, t=3, criterion="maxclust")
# Map cluster index → IT name (anchor by patient assignment in CSV)
assigned = pd.read_csv(f"{OUT_DIR}/21_immunotype_assignments.csv")
pid_to_it = dict(zip(assigned["pid"], assigned["immunotype"]))
feat["IT"] = feat["pid"].map(pid_to_it)
# Sanity check: cluster index → IT mode
fc_to_it = {}
for cid in np.unique(labels):
    pids = feat["pid"].values[labels == cid]
    its = [pid_to_it[p] for p in pids]
    fc_to_it[cid] = max(set(its), key=its.count)
print("cluster→IT mapping:", fc_to_it)

# Dendrogram — leaf order
dd = dendrogram(Z, no_plot=True)
leaf_order = dd["leaves"]
print("Leaf order:", leaf_order, "→ pids:", feat["pid"].iloc[leaf_order].tolist())

# ─── colour palettes (consistent with rest of figures) ─────────────────────
IT_COL = {"IT1": "#E67E22", "IT2": "#2980B9", "IT3": "#27AE60"}
AB_COL = {"Jo-1": "#C0392B", "PL-12": "#2471A3",
          "EJ": "#1A7A4A", "PL-7": "#7D3C98"}
EVENT_COL_PRES = "#2C3E50"
EVENT_COL_ABS  = "#FFFFFF"
EVENT_BORDER   = "#999999"

# ─── figure layout ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10.5, 6.0), facecolor="white")
gs = gridspec.GridSpec(
    1, 3, figure=fig,
    left=0.04, right=0.985, top=0.93, bottom=0.18,
    wspace=0.020,
    width_ratios=[2.6, 0.45, 3.0],
)

# Subplot for the dendrogram (left)
ax_dn = fig.add_subplot(gs[0, 0])
# Subplot for the patient-ID column (middle, narrow)
ax_id = fig.add_subplot(gs[0, 1])
# Subplot for the clinical annotation strip (right)
ax_an = fig.add_subplot(gs[0, 2])

# ─── dendrogram ────────────────────────────────────────────────────────────
def link_color(cid):
    """Colour function for dendrogram links: lookup leaf cluster IDs."""
    if cid >= len(labels):
        return "#888"
    it = fc_to_it.get(int(labels[cid]), None)
    return IT_COL.get(it, "#888")

# Custom colouring: walk linkage matrix and assign per-link colour
n = len(labels)
link_cols = {}
for i, (a, b, dist, cnt) in enumerate(Z):
    a, b = int(a), int(b)
    ca = labels[a] if a < n else link_cols.get(a, None)
    cb = labels[b] if b < n else link_cols.get(b, None)
    if ca == cb and ca is not None:
        link_cols[n + i] = ca
    else:
        link_cols[n + i] = -1  # mixed → grey

def color_func(cid):
    cl = link_cols.get(cid, -1)
    if cl == -1:
        return "#888888"
    return IT_COL.get(fc_to_it.get(cl, None), "#888888")

dendrogram(
    Z,
    ax=ax_dn,
    orientation="right",
    no_labels=True,
    color_threshold=None,
    above_threshold_color="#888888",
    link_color_func=color_func,
)
# Match leaf y-positions to the annotation strip (1-indexed * 10)
ax_dn.set_xlabel("Ward linkage distance")
ax_dn.spines["left"].set_visible(False)
ax_dn.set_yticks([])
ax_dn.invert_xaxis()  # root on left, leaves on right (so it points toward IDs)
ax_dn.set_title("Immunotype discovery by unsupervised patient clustering",
                fontsize=12, pad=8, loc="left")

# ─── patient-ID column ─────────────────────────────────────────────────────
# Leaves of dendrogram come in order leaf_order; each occupies y in [10*i+5,...]
n_pat = len(leaf_order)
ax_id.set_xlim(0, 1)
ax_id.set_ylim(0, n_pat * 10)
ax_id.invert_yaxis()
for s in ax_id.spines.values():
    s.set_visible(False)
ax_id.set_xticks([]); ax_id.set_yticks([])

# Y-coordinate convention: in scipy.dendrogram with orientation="right", leaves
# are spaced at 5, 15, 25, ... bottom-to-top (or top-to-bottom after invert).
# After ax_dn invert_xaxis, leaves are in default spacing, but we mirror them
# manually using the y-tick locations from the dendrogram.
ytick_locs = [5 + 10 * i for i in range(n_pat)]
# leaf_order[i] is the dataframe-row index for the leaf at position i
# (bottom-to-top in default scipy, so we reverse)
order_rows = leaf_order[::-1]  # top-to-bottom for our display
for i, row_ix in enumerate(order_rows):
    pid = int(feat["pid"].iloc[row_ix])
    ax_id.text(0.5, ytick_locs[i], f"P{pid:02d}",
               ha="center", va="center",
               fontsize=10, fontweight="bold", color="#222")

# Align dendrogram y-axis to match
ax_dn.set_ylim(ax_id.get_ylim())  # both inverted

# ─── clinical annotation strip ─────────────────────────────────────────────
# Columns: antibody | immunotype | ILD | myositis | arthritis | rash
# Use clin (10 patients, may be in different order)
clin_by_pid = clin.set_index("pid")
event_cols = ["ILD", "myositis", "arthritis", "rash"]

ax_an.set_xlim(0, 6)
ax_an.set_ylim(0, n_pat * 10)
ax_an.invert_yaxis()
for s in ax_an.spines.values():
    s.set_visible(False)
ax_an.set_yticks([])

# Column header positions (top, above plot area)
col_headers = ["antibody", "immunotype"] + event_cols
col_x_centres = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]

for x_c, hdr in zip(col_x_centres, col_headers):
    ax_an.text(x_c, -1.5, hdr, ha="center", va="bottom",
               fontsize=9.5, color="#333", rotation=30, rotation_mode="anchor")

# Each row's annotation
ROW_H = 9.0  # of 10 unit slot (small gap between rows)
for i, row_ix in enumerate(order_rows):
    pid = int(feat["pid"].iloc[row_ix])
    ab  = feat["antibody"].iloc[row_ix]
    it  = feat["IT"].iloc[row_ix]
    cy  = ytick_locs[i]
    y0  = cy - ROW_H / 2
    # antibody box
    ax_an.add_patch(Rectangle((0.10, y0), 0.80, ROW_H,
                               facecolor=AB_COL.get(ab, "#888"),
                               edgecolor="white", lw=0.6))
    ax_an.text(0.5, cy, ab, ha="center", va="center",
               fontsize=9, fontweight="bold", color="white")
    # immunotype box
    ax_an.add_patch(Rectangle((1.10, y0), 0.80, ROW_H,
                               facecolor=IT_COL.get(it, "#888"),
                               edgecolor="white", lw=0.6))
    ax_an.text(1.5, cy, it, ha="center", va="center",
               fontsize=9, fontweight="bold", color="white")
    # event boxes
    for j, ev in enumerate(event_cols):
        present = (clin_by_pid.loc[pid, ev] == 1
                   if pid in clin_by_pid.index else 0)
        col = EVENT_COL_PRES if present else EVENT_COL_ABS
        ax_an.add_patch(Rectangle((2.10 + j, y0), 0.80, ROW_H,
                                   facecolor=col,
                                   edgecolor=EVENT_BORDER, lw=0.6))

ax_an.set_xticks([])

# legend (bottom of figure)
legend_handles = [
    *[Patch(facecolor=AB_COL[g], label=g) for g in ["Jo-1", "PL-12", "EJ", "PL-7"]],
    *[Patch(facecolor=IT_COL[t], label=t) for t in ["IT1", "IT2", "IT3"]],
    Patch(facecolor=EVENT_COL_PRES, label="present"),
    Patch(facecolor=EVENT_COL_ABS, edgecolor=EVENT_BORDER, label="absent"),
]
fig.legend(handles=legend_handles, ncol=9, loc="lower center",
           bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=9,
           handlelength=1.0, handletextpad=0.4, columnspacing=0.9)

out_pdf = f"{OUT_DIR}/21_immunotype_discovery.pdf"
out_png = f"{OUT_DIR}/21_immunotype_discovery.png"
fig.savefig(out_pdf, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
fig.savefig(out_png, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
