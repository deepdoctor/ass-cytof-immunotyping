"""
cytof_mainfig3_rebuild.py
=========================
Native 4-panel rebuild of Main Figure 3 — compositional heterogeneity within
and between autoantibody classes.

Layout: panel (a) spans the full top row (stacked bars need horizontal real
estate), panels (b)(c)(d) share the bottom row at equal width.

Replaces the prior 6-panel composite that suffered three label-overlap render
disasters in the source PNGs.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import kruskal

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
    "axes.linewidth":       0.9,
    "xtick.major.width":    0.7,
    "ytick.major.width":    0.7,
    "xtick.major.size":     3.0,
    "ytick.major.size":     3.0,
    "lines.linewidth":      1.0,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.edgecolor":       "#222222",
    "xtick.color":          "#222222",
    "ytick.color":          "#222222",
    "axes.labelcolor":      "#222222",
    "text.color":           "#222222",
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
    "savefig.dpi":          300,
    "figure.dpi":           300,
})

GROUP_PALETTE = {"Jo-1": "#C0392B", "PL-12": "#2471A3",
                 "EJ":   "#1A7A4A", "PL-7":  "#7D3C98"}
GROUP_ORDER = ["Jo-1", "PL-12", "EJ", "PL-7"]

CELL_TYPES = [
    "CD4 Central Memory T", "CD4 Effector Memory T", "CD4 Naive T", "CD4 Th1-like",
    "CD8 Effector T", "CD8 Naive T",
    "Classical Monocyte", "Inflammatory Monocyte", "Non-classical Monocyte",
    "NK cell", "Naive B", "mDC", "pDC", "γδ T cell",
]
CELLTYPE_PALETTE = [
    "#1F77B4", "#5B9BD5", "#7DC1E8", "#AEC7E8",
    "#9467BD", "#C5B0D5",
    "#E67E22", "#D35400", "#F39C12",
    "#27AE60", "#16A085",
    "#E74C3C", "#C0392B",
    "#F1C40F",
]

# ─────────────────────────────────────────────────────────────────────────────
feat = pd.read_csv(f"{OUT_DIR}/SuppTable_feature_matrix.csv")
prop = pd.read_csv(f"{OUT_DIR}/05_cell_proportions.csv")
boot = pd.read_csv(f"{OUT_DIR}/27_abundance_bootstrap_CI.csv")

feat = feat.drop(columns=[c for c in feat.columns if c == "name"])
feat["pid_label"] = feat["pid"].apply(lambda x: f"P{int(x):02d}")
feat = feat.sort_values(["antibody", "pid"]).reset_index(drop=True)
prop["pid"] = prop["sample"].map(dict(zip(feat["sample"], feat["pid"])))
prop["pid_label"] = prop["pid"].apply(lambda x: f"P{int(x):02d}")

# CRITICAL: 05_cell_proportions.csv contains rows for every (sample × all 4
# autoantibody groups) — i.e. each sample appears 4× with each group label,
# not just its own. Filter to only the rows where 'group' matches the
# sample's actual antibody class (from feat).
sample_to_ab = dict(zip(feat["sample"], feat["antibody"]))
prop["true_group"] = prop["sample"].map(sample_to_ab)
prop = prop[prop["group"] == prop["true_group"]].copy()
prop = prop.drop(columns=["true_group"])

# ═════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14.5, 11.5))
gs = gridspec.GridSpec(
    2, 3, figure=fig,
    left=0.060, right=0.835, top=0.920, bottom=0.110,
    wspace=0.55, hspace=0.55,
    height_ratios=[1.05, 1.0],
    width_ratios=[1.0, 1.15, 1.0],
)
PANEL_LBL = dict(fontsize=22, fontweight="bold", color="#000",
                 ha="left", va="bottom", family="sans-serif")


# ── (a) Stacked-bar composition per sample (full top row) ───────────────────
ax = fig.add_subplot(gs[0, :])

prop_pv = prop.pivot_table(index=["pid_label", "group", "sample"],
                            columns="cell_type", values="proportion",
                            aggfunc="sum", fill_value=0.0).reset_index()
prop_pv = prop_pv[["pid_label", "group", "sample"] +
                  [c for c in CELL_TYPES if c in prop_pv.columns]]
prop_pv["group"] = pd.Categorical(prop_pv["group"], categories=GROUP_ORDER,
                                   ordered=True)
prop_pv = prop_pv.sort_values(["group", "pid_label"]).reset_index(drop=True)

# Compute x-positions with explicit gaps between autoantibody groups
# so a reader sees that each group has only its own patients (no empty slots).
GROUP_GAP = 0.7  # extra spacing between groups
xs = []
running = 0.0
prev_group = None
for _, row in prop_pv.iterrows():
    if prev_group is not None and row["group"] != prev_group:
        running += GROUP_GAP
    xs.append(running)
    running += 1.0
    prev_group = row["group"]
xs = np.array(xs)
bottoms = np.zeros(len(prop_pv))
for ct, color in zip(CELL_TYPES, CELLTYPE_PALETTE):
    if ct not in prop_pv.columns:
        continue
    vals = prop_pv[ct].values
    ax.bar(xs, vals, bottom=bottoms, width=0.78, color=color,
           edgecolor="white", linewidth=0.4, label=ct)
    bottoms += vals

ax.set_xticks(xs)
ax.set_xticklabels(prop_pv["pid_label"].values, rotation=45, ha="right",
                    fontsize=10.0)
ax.set_xlim(xs[0] - 0.65, xs[-1] + 0.65)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Cell-type proportion", fontsize=11.0)
ax.tick_params(axis="y", labelsize=9.5)

# group brackets — anchored to actual x-positions of group members
groups = prop_pv["group"].astype(str).values
y_bracket = 1.035
for g in GROUP_ORDER:
    idx = np.where(groups == g)[0]
    if len(idx) == 0:
        continue
    x0, x1 = xs[idx.min()] - 0.40, xs[idx.max()] + 0.40
    ax.plot([x0, x1], [y_bracket, y_bracket],
            color=GROUP_PALETTE[g], lw=3.0,
            solid_capstyle="butt", clip_on=False)
    ax.text((x0 + x1) / 2, y_bracket + 0.025, g,
            ha="center", va="bottom", fontsize=11.5, fontweight="bold",
            color=GROUP_PALETTE[g], clip_on=False)

# 14-cell-type legend on the right of the panel (outside axes)
ax.legend(loc="center left", bbox_to_anchor=(1.005, 0.50),
          ncol=1, fontsize=9.0, frameon=False,
          handlelength=0.9, handletextpad=0.4, borderpad=0.2,
          labelspacing=0.45,
          title="Cell type", title_fontsize=10.0)

ax.text(-0.045, 1.12, "a", transform=ax.transAxes, **PANEL_LBL)


# ── (b) PCA on patient feature vector ───────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])

meta_cols = ["sample", "pid", "antibody", "immunotype", "pid_label"]
feat_num = feat.drop(columns=meta_cols)
X = StandardScaler().fit_transform(feat_num.values)
pca = PCA(n_components=2)
Xp = pca.fit_transform(X)
v1, v2 = pca.explained_variance_ratio_ * 100

for g in GROUP_ORDER:
    mask = feat["antibody"].values == g
    ax.scatter(Xp[mask, 0], Xp[mask, 1], s=85, c=GROUP_PALETTE[g],
               edgecolor="white", linewidth=0.7,
               label=f"{g} (n={int(mask.sum())})", zorder=3, alpha=0.92)
for i, lbl in enumerate(feat["pid_label"].values):
    ax.annotate(lbl, (Xp[i, 0], Xp[i, 1]),
                xytext=(5, 4), textcoords="offset points",
                fontsize=8.5, color="#222")

ax.axhline(0, color="#bbb", lw=0.5, ls="--", zorder=0)
ax.axvline(0, color="#bbb", lw=0.5, ls="--", zorder=0)
ax.set_xlabel(f"PC1 ({v1:.1f}% variance)", fontsize=11.0)
ax.set_ylabel(f"PC2 ({v2:.1f}% variance)", fontsize=11.0)
ax.legend(fontsize=9.0, loc="lower right", frameon=False,
          handletextpad=0.4, borderpad=0.3)
ax.set_title("Patient-level PCA on 65-D\nimmune feature vector",
             fontsize=11.5, pad=4)
xpad = (Xp[:, 0].max() - Xp[:, 0].min()) * 0.13
ypad = (Xp[:, 1].max() - Xp[:, 1].min()) * 0.13
ax.set_xlim(Xp[:, 0].min() - xpad, Xp[:, 0].max() + xpad * 1.35)
ax.set_ylim(Xp[:, 1].min() - ypad, Xp[:, 1].max() + ypad * 1.15)
ax.text(-0.30, 1.05, "b", transform=ax.transAxes, **PANEL_LBL)


# ── (c) Canonical immune ratios by autoantibody group ───────────────────────
ax = fig.add_subplot(gs[1, 1])

ratio_specs = [
    ("CD4:CD8",
     lambda r: (r["CD4 Central Memory T"] + r["CD4 Effector Memory T"] +
                r["CD4 Naive T"] + r["CD4 Th1-like"]) /
                max(r["CD8 Effector T"] + r["CD8 Naive T"], 1e-3)),
    ("Inf-Mono:Class-Mono",
     lambda r: r["Inflammatory Monocyte"] /
                max(r["Classical Monocyte"], 1e-3)),
    ("NK:T",
     lambda r: r["NK cell"] /
                max(r["CD4 Central Memory T"] + r["CD4 Effector Memory T"] +
                    r["CD4 Naive T"] + r["CD4 Th1-like"] +
                    r["CD8 Effector T"] + r["CD8 Naive T"], 1e-3)),
    ("B:T",
     lambda r: r["Naive B"] /
                max(r["CD4 Central Memory T"] + r["CD4 Effector Memory T"] +
                    r["CD4 Naive T"] + r["CD4 Th1-like"] +
                    r["CD8 Effector T"] + r["CD8 Naive T"], 1e-3)),
]

ratio_df = pd.DataFrame({"antibody": feat["antibody"].values})
for name, fn in ratio_specs:
    ratio_df[name] = feat.apply(fn, axis=1).values

n_ratios = len(ratio_specs)
n_groups = len(GROUP_ORDER)
slot_per_ratio = n_groups + 1.4   # in-ratio (n_groups) + gap

# compute positions
positions, ratio_centres = [], []
for ri in range(n_ratios):
    base = ri * slot_per_ratio
    grp_pos = [base + gi for gi in range(n_groups)]
    positions.extend(grp_pos)
    ratio_centres.append(base + (n_groups - 1) / 2)

rng = np.random.default_rng(1)
for ri, (name, _) in enumerate(ratio_specs):
    for gi, g in enumerate(GROUP_ORDER):
        pos = ri * slot_per_ratio + gi
        vals = ratio_df.loc[ratio_df["antibody"] == g, name].values
        ax.boxplot(
            [vals], positions=[pos], widths=0.72, patch_artist=True,
            showfliers=False,
            whiskerprops=dict(lw=0.5, color="#444"),
            capprops=dict(lw=0.5, color="#444"),
            medianprops=dict(lw=0.9, color="#222"),
            boxprops=dict(lw=0.4, edgecolor="#444",
                          facecolor=GROUP_PALETTE[g], alpha=0.55),
        )
        xs_j = pos + rng.uniform(-0.20, 0.20, size=len(vals))
        ax.scatter(xs_j, vals, s=8, c=GROUP_PALETTE[g],
                   edgecolor="white", linewidth=0.3, zorder=3)

ax.set_yscale("log")
ax.set_xticks(ratio_centres)
ax.set_xticklabels([s[0] for s in ratio_specs], fontsize=10.0,
                    rotation=20, ha="right")
ax.set_ylabel("Ratio (log scale)", fontsize=11.0)
ax.tick_params(axis="y", labelsize=9.5)
ax.set_xlim(-0.9, n_ratios * slot_per_ratio - 0.4)

# expand log-axis upper limit so KW-p annotations clear box tops
ymin_log, ymax_log = ax.get_ylim()
ax.set_ylim(ymin_log, ymax_log * 4.0)
y_anno = ymax_log * 2.0
for ri, (name, _) in enumerate(ratio_specs):
    groups_vals = [ratio_df.loc[ratio_df["antibody"] == g, name].values
                   for g in GROUP_ORDER]
    try:
        _, p_kw = kruskal(*groups_vals)
    except Exception:
        p_kw = np.nan
    p_str = f"P = {p_kw:.2f}" if not np.isnan(p_kw) else "n/a"
    ax.text(ratio_centres[ri], y_anno, p_str,
            ha="center", va="center", fontsize=9.0, color="#555")

# group legend (above panel area to avoid box overlap)
legend_handles = [Patch(facecolor=GROUP_PALETTE[g], edgecolor="none",
                        label=g, alpha=0.7) for g in GROUP_ORDER]
ax.legend(handles=legend_handles, fontsize=9.5, loc="lower center",
          bbox_to_anchor=(0.5, -0.45), frameon=False, ncol=4,
          handlelength=1.0, handletextpad=0.4, columnspacing=1.4,
          borderpad=0.3)
ax.set_title("Canonical immune ratios\n(KW P; no ratio reaches P < 0.05)",
             fontsize=11.5, pad=4)
ax.text(-0.30, 1.05, "c", transform=ax.transAxes, **PANEL_LBL)


# ── (d) Top-5 cell-type bootstrap forest ────────────────────────────────────
ax = fig.add_subplot(gs[1, 2])

mean_pv = boot.pivot_table(index="cell_type", columns="group",
                            values="mean", aggfunc="first")
spread = (mean_pv.max(axis=1) - mean_pv.min(axis=1)).sort_values(ascending=False)
top5 = spread.head(5).index.tolist()

sub = boot[boot["cell_type"].isin(top5)].copy()
sub["cell_type"] = pd.Categorical(sub["cell_type"], categories=top5[::-1],
                                   ordered=True)
sub["group"] = pd.Categorical(sub["group"], categories=GROUP_ORDER,
                               ordered=True)
sub = sub.sort_values(["cell_type", "group"]).reset_index(drop=True)

ct_to_y = {ct: yi for yi, ct in enumerate(top5[::-1])}
group_offsets = {g: (gi - (len(GROUP_ORDER) - 1) / 2) * 0.18
                 for gi, g in enumerate(GROUP_ORDER)}
for _, r in sub.iterrows():
    yi = ct_to_y[r["cell_type"]] + group_offsets[r["group"]]
    col = GROUP_PALETTE[r["group"]]
    ax.plot([r["ci_low"], r["ci_high"]], [yi, yi],
            color=col, lw=1.6, alpha=0.85, solid_capstyle="butt")
    ax.plot([r["mean"]], [yi], "o", color=col, markersize=6.5,
            markeredgecolor="white", markeredgewidth=0.5, zorder=3)

ax.set_yticks(list(range(len(top5))))
ax.set_yticklabels(top5[::-1], fontsize=10.0)
ax.tick_params(axis="x", labelsize=9.5)
ax.axvline(0, color="#222", lw=0.6)
ax.set_xlabel("Mean abundance %  (95% bootstrap CI)", fontsize=11.0)
ax.set_title("Top 5 most-divergent\ncell types (max-min spread)",
             fontsize=11.5, pad=4)
legend_handles = [Patch(facecolor=GROUP_PALETTE[g], edgecolor="none",
                        label=g) for g in GROUP_ORDER]
ax.legend(handles=legend_handles, fontsize=9.5, loc="lower center",
          bbox_to_anchor=(0.5, -0.45), frameon=False, ncol=4,
          handlelength=1.0, handletextpad=0.4, columnspacing=1.4,
          borderpad=0.3)
ax.grid(axis="x", color="#eee", lw=0.3); ax.set_axisbelow(True)
ax.set_xlim(left=0)
ax.text(-0.40, 1.05, "d", transform=ax.transAxes, **PANEL_LBL)


# ─────────────────────────────────────────────────────────────────────────────
# Figure title removed (journal convention).

out_pdf = f"{OUT_DIR}/MainFig3_composition.pdf"
out_png = f"{OUT_DIR}/MainFig3_composition.png"
fig.savefig(out_pdf, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
fig.savefig(out_png, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
