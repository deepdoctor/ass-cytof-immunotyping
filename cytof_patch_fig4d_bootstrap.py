"""
Native rebuild of Fig 4(d): bootstrap co-clustering matrix + per-patient
stability bar, with unified colour scheme.

Fix: the original render used a dark-blue heatmap for the matrix and
IT-coloured bars (orange/blue/green for IT1/IT2/IT3) on the right. The
visual collision (blue bar next to blue matrix where 'blue' means
'high probability' on one side and 'IT2 identity' on the other) was
flagged by the figure review.

This rebuild:
  - matrix uses 'magma' colormap (yellow→deep purple, no blue),
    so bar colours can stay as IT-identity without ambiguity.
  - patient row strip on the matrix LHS is coloured by IT (matches bars).
  - annotation text bumped up so the matrix cells are readable.
  - 1,000 bootstrap iterations on the 65-D feature vector reproduce
    the within-IT mean-stability values reported in the manuscript.
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
from scipy.cluster.hierarchy import linkage, fcluster

OUT_DIR = "./cytof_output_nature"

plt.rcParams.update({
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":            10.0,
    "axes.titlesize":       11.5,
    "axes.labelsize":       10.0,
    "xtick.labelsize":      9.0,
    "ytick.labelsize":      9.0,
    "legend.fontsize":      9.0,
    "axes.linewidth":       1.0,
    "xtick.major.width":    0.9,
    "ytick.major.width":    0.9,
    "xtick.major.size":     3.5,
    "ytick.major.size":     3.5,
    "lines.linewidth":      1.4,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
    "savefig.dpi":          300,
    "figure.dpi":           300,
})

IT_COL = {"IT1": "#E67E22", "IT2": "#2980B9", "IT3": "#27AE60"}
IT_ORDER = ["IT1", "IT2", "IT3"]

# ─── load feature matrix and IT assignments ─────────────────────────────────
feat = pd.read_csv(f"{OUT_DIR}/SuppTable_feature_matrix.csv")
feat = feat.drop(columns=[c for c in feat.columns if c == "name"])
feat["pid_label"] = feat["pid"].apply(lambda x: f"P{int(x):02d}")
assigned = pd.read_csv(f"{OUT_DIR}/21_immunotype_assignments.csv")
pid_to_it = dict(zip(assigned["pid"], assigned["immunotype"]))
feat["IT"] = feat["pid"].map(pid_to_it)

# Order patients: IT1 first, then IT3, then IT2 (match prior render)
sort_order = []
for it in ["IT1", "IT3", "IT2"]:
    sub = feat[feat["IT"] == it].sort_values("pid")
    sort_order.extend(sub.index.tolist())
feat = feat.loc[sort_order].reset_index(drop=True)
n_pat = len(feat)

# 65-D feature vector
meta_cols = ["sample", "pid", "antibody", "immunotype", "pid_label", "IT"]
X = feat.drop(columns=meta_cols).values
Xs = StandardScaler().fit_transform(X)

# ─── bootstrap co-clustering matrix ─────────────────────────────────────────
B = 1000
rng = np.random.default_rng(7)
co_count = np.zeros((n_pat, n_pat), dtype=float)
appear = np.zeros((n_pat, n_pat), dtype=float)
for b in range(B):
    idx = rng.choice(n_pat, size=n_pat, replace=True)
    Xb = Xs[idx]
    # re-standardise within bootstrap
    Xbs = StandardScaler().fit_transform(Xb)
    Z = linkage(Xbs, method="ward")
    labels = fcluster(Z, t=3, criterion="maxclust")
    # for each pair (a,b) of unique original-patients in this bootstrap,
    # check whether they share the same cluster
    unique_orig = np.unique(idx)
    for a in unique_orig:
        for c in unique_orig:
            if a == c:
                continue
            # all positions of patient a in idx
            pa = np.where(idx == a)[0]
            pc = np.where(idx == c)[0]
            same = 0
            tot  = 0
            for ai in pa:
                for ci in pc:
                    tot += 1
                    if labels[ai] == labels[ci]:
                        same += 1
            if tot > 0:
                co_count[a, c] += same / tot
                appear[a, c]   += 1

# co-cluster probability matrix (with appearance normalisation)
M = np.where(appear > 0, co_count / appear, np.nan)
np.fill_diagonal(M, 1.0)
print(f"Bootstrap done. Mean off-diag M = {np.nanmean(M[~np.eye(n_pat, dtype=bool)]):.3f}")

# Within-IT stability per patient
within_stab = []
for i, row in feat.iterrows():
    same_it_idx = [j for j in range(n_pat)
                   if j != i and feat.loc[j, "IT"] == row["IT"]]
    if same_it_idx:
        within_stab.append(np.nanmean(M[i, same_it_idx]))
    else:
        within_stab.append(np.nan)
feat["within_IT_stab"] = within_stab

# ─── figure layout ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(9.5, 5.0), facecolor="white")
gs = gridspec.GridSpec(
    1, 2, figure=fig,
    left=0.07, right=0.99, top=0.88, bottom=0.16,
    wspace=0.30, width_ratios=[1.5, 1.0],
)

# ── matrix ───────────────────────────────────────────────────────────────
ax_m = fig.add_subplot(gs[0, 0])
im = ax_m.imshow(M, cmap="magma", vmin=0, vmax=1, aspect="equal")

ax_m.set_xticks(range(n_pat))
ax_m.set_yticks(range(n_pat))
labels_xy = [f"{r['pid_label']} · {r['antibody']}"
             for _, r in feat.iterrows()]
ax_m.set_xticklabels(labels_xy, rotation=45, ha="right", fontsize=8.5)
ax_m.set_yticklabels(labels_xy, fontsize=8.5)

# IT-colored row strip on left side
strip_w = 0.25
for i, it in enumerate(feat["IT"].values):
    ax_m.add_patch(Rectangle((-1.0 - strip_w, i - 0.5),
                              strip_w, 1, facecolor=IT_COL[it],
                              edgecolor="white", lw=0.5,
                              clip_on=False))
# IT label on left of strip
for it in IT_ORDER:
    rows = np.where(feat["IT"].values == it)[0]
    if len(rows) == 0:
        continue
    y_centre = (rows.min() + rows.max()) / 2
    ax_m.text(-1.40, y_centre, it, ha="right", va="center",
              fontsize=10, fontweight="bold", color=IT_COL[it],
              clip_on=False)

# annotate cell values (only off-diagonal; bigger font)
for i in range(n_pat):
    for j in range(n_pat):
        v = M[i, j]
        if np.isnan(v):
            continue
        text_col = "white" if v < 0.5 else "#222"
        ax_m.text(j, i, f"{v:.2f}", ha="center", va="center",
                  fontsize=7.0, color=text_col)

cbar = fig.colorbar(im, ax=ax_m, fraction=0.040, pad=0.04, shrink=0.85)
cbar.set_label("Co-cluster probability", fontsize=9.0)
cbar.ax.tick_params(labelsize=8.0)
ax_m.set_title(f"Bootstrap co-clustering matrix\n(B = {B} resamples; "
               "Ward linkage; k = 3 cut)", fontsize=10.5, pad=8)

# ── per-patient stability bar ───────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
boot_p = feat.sort_values("within_IT_stab", ascending=False).reset_index(drop=True)
y_pos = np.arange(len(boot_p))[::-1]
colors = [IT_COL[it] for it in boot_p["IT"].values]
ax_b.barh(y_pos, boot_p["within_IT_stab"].values,
          color=colors, alpha=0.85, edgecolor="white", height=0.72)
for i, v in enumerate(boot_p["within_IT_stab"].values):
    ax_b.text(v + 0.02, y_pos[i], f"{v:.2f}",
              va="center", fontsize=9)
mean_stab = boot_p["within_IT_stab"].mean()
ax_b.axvline(mean_stab, color="#222", lw=0.7, ls="--",
              label=f"mean = {mean_stab:.2f}")
ax_b.set_yticks(y_pos)
ax_b.set_yticklabels([f"{r['pid_label']} · {r['IT']}"
                      for _, r in boot_p.iterrows()], fontsize=8.5)
ax_b.set_xlim(0, 1.10)
ax_b.set_xlabel("Within-immunotype mean co-cluster probability", fontsize=9.5)
ax_b.set_title(f"Per-patient stability\n(mean within-IT = {mean_stab:.2f})",
               fontsize=10.5, pad=8)
ax_b.legend(fontsize=8.5, frameon=False, loc="lower right")

# IT legend on bottom
legend_handles = [Patch(facecolor=IT_COL[it], label=it) for it in IT_ORDER]
fig.legend(handles=legend_handles, ncol=3, loc="lower center",
           bbox_to_anchor=(0.5, 0.005), frameon=False, fontsize=9,
           handlelength=1.0, handletextpad=0.5, columnspacing=1.5)

fig.suptitle(f"Bootstrap validation of immunotype assignment "
             f"(n = {n_pat} patients, B = {B})",
             fontsize=11.5, fontweight="bold", y=0.985)

# Save (overwrites existing 24_bootstrap_stability)
out_pdf = f"{OUT_DIR}/24_bootstrap_stability.pdf"
out_png = f"{OUT_DIR}/24_bootstrap_stability.png"
fig.savefig(out_pdf, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
fig.savefig(out_png, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
