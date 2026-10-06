"""
Native unified vector rebuild of Main Figure 4 — three immunotypes recapitulate
clinical phenotype.

All four panels rendered as matplotlib primitives (no imshow, no pre-rendered
PNGs) so the output PDF is fully editable in Illustrator/Acrobat Pro.
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
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, leaves_list
from sklearn.preprocessing import StandardScaler
from scipy.stats import kruskal, spearmanr

OUT_DIR = "./cytof_output_nature"

plt.rcParams.update({
    "font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "font.size":10.5,"axes.titlesize":12.0,"axes.labelsize":10.5,
    "xtick.labelsize":9.5,"ytick.labelsize":9.5,"legend.fontsize":9.5,
    "axes.linewidth":0.9,"axes.spines.top":False,"axes.spines.right":False,
    "pdf.fonttype":42,"ps.fonttype":42,"savefig.dpi":300,"figure.dpi":300,
})

GROUP_PALETTE = {"Jo-1":"#C0392B","PL-12":"#2471A3","EJ":"#1A7A4A","PL-7":"#7D3C98"}
IT_COL = {"IT1":"#E67E22","IT2":"#2980B9","IT3":"#27AE60"}
IT_ORDER = ["IT1","IT2","IT3"]

CT_FEATURES = [
    "CD4 Central Memory T", "CD4 Effector Memory T", "CD4 Naive T", "CD4 Th1-like",
    "CD8 Effector T", "CD8 Naive T",
    "Classical Monocyte", "Inflammatory Monocyte", "Non-classical Monocyte",
    "NK cell", "Naive B", "mDC", "pDC", "γδ T cell",
]
SIGNATURE_FEATURES = [
    "Th1_score","Th17_score","Tfh_score","Treg_score",
    "Cytotoxic_score","Exhaustion_score","Activation_score",
    "Naive_score","Memory_score",
]
CLINICAL_FEATURES = ["age","CK","LDH","ESR","CRP","FVC","DLCO"]
EVENT_COL_PRES = "#2C3E50"; EVENT_COL_ABS = "#FFFFFF"; EVENT_BORDER = "#999999"

# ─── load all data ─────────────────────────────────────────────────────────
feat = pd.read_csv(f"{OUT_DIR}/SuppTable_feature_matrix.csv")
feat = feat.drop(columns=[c for c in feat.columns if c == "name"])
feat["pid_label"] = feat["pid"].apply(lambda x: f"P{int(x):02d}")
clin = pd.read_csv(f"{OUT_DIR}/23_immunotype_clinical_table.csv")
clin = clin.drop(columns=[c for c in clin.columns if c == "name"])
assigned = pd.read_csv(f"{OUT_DIR}/21_immunotype_assignments.csv")
pid_to_it = dict(zip(assigned["pid"], assigned["immunotype"]))
feat["IT"] = feat["pid"].map(pid_to_it)

meta_cols = ["sample","pid","antibody","immunotype","pid_label","IT"]
X = feat.drop(columns=meta_cols).values
Xs = StandardScaler().fit_transform(X)

# ─── pre-compute everything needed for each panel ──────────────────────────
# Panel A: Ward linkage, leaf order
Z_pat = linkage(Xs, method="ward")
labels_3 = fcluster(Z_pat, t=3, criterion="maxclust")
fc_to_it = {}
for cid in np.unique(labels_3):
    pids = feat["pid"].values[labels_3 == cid]
    its = [pid_to_it[p] for p in pids]
    fc_to_it[cid] = max(set(its), key=its.count)
dd = dendrogram(Z_pat, no_plot=True)
leaf_order_a = dd["leaves"]
order_rows_a = leaf_order_a[::-1]

# Panel C: bilateral dendrogram heatmap
clin_by_pid = clin.set_index("pid")
features_avail = ([c for c in CT_FEATURES if c in feat.columns] +
                   [c for c in SIGNATURE_FEATURES if c in feat.columns])
M_immune = feat.set_index("pid")[features_avail].copy()
M_clin = clin_by_pid[CLINICAL_FEATURES].copy()
for c in ["CK","LDH"]:
    if c in M_clin.columns:
        M_clin[c] = np.log10(M_clin[c].astype(float) + 1)
def col_zscore(df):
    out = df.copy().astype(float)
    for c in out.columns:
        v = out[c].values
        out[c] = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)
    return out
M_all = pd.concat([col_zscore(M_immune), col_zscore(M_clin)], axis=1)
M_for_cluster = M_all.fillna(0).values
Z_row_c = linkage(M_for_cluster, method="ward")
Z_col_c = linkage(M_for_cluster.T, method="ward")
row_order_c = leaves_list(Z_row_c)
col_order_c = leaves_list(Z_col_c)
M_disp_c = M_all.iloc[row_order_c, col_order_c]

# Panel D: bootstrap (recompute or load cached)
print("Computing bootstrap co-cluster matrix (B=1000) for Panel D …")
sort_d = []
for it in ["IT1","IT3","IT2"]:
    sub = feat[feat["IT"] == it].sort_values("pid")
    sort_d.extend(sub.index.tolist())
feat_d = feat.loc[sort_d].reset_index(drop=True)
n_pat = len(feat_d)
X_d = feat_d.drop(columns=meta_cols).values
Xs_d = StandardScaler().fit_transform(X_d)

B = 1000
rng = np.random.default_rng(7)
co_count = np.zeros((n_pat, n_pat)); appear = np.zeros((n_pat, n_pat))
for b in range(B):
    idx = rng.choice(n_pat, size=n_pat, replace=True)
    Xb = StandardScaler().fit_transform(Xs_d[idx])
    Z = linkage(Xb, method="ward")
    labels = fcluster(Z, t=3, criterion="maxclust")
    unique_orig = np.unique(idx)
    for ai_orig in unique_orig:
        for cj_orig in unique_orig:
            if ai_orig == cj_orig: continue
            pa = np.where(idx == ai_orig)[0]
            pc = np.where(idx == cj_orig)[0]
            same = sum(1 for ai in pa for ci in pc if labels[ai] == labels[ci])
            tot = len(pa) * len(pc)
            if tot > 0:
                co_count[ai_orig, cj_orig] += same / tot
                appear[ai_orig, cj_orig]   += 1
M_boot = np.where(appear > 0, co_count / appear, np.nan)
np.fill_diagonal(M_boot, 1.0)

within_stab = []
for i, row in feat_d.iterrows():
    same_it = [j for j in range(n_pat) if j != i and feat_d.loc[j,"IT"] == row["IT"]]
    within_stab.append(np.nanmean(M_boot[i, same_it]) if same_it else np.nan)
feat_d["within_IT_stab"] = within_stab
mean_stab = np.nanmean(within_stab)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE LAYOUT — 13.5 × 13 outer 2×2
# ════════════════════════════════════════════════════════════════════════════
# Wide landscape canvas — 24×13 in, panels each ~11×6 in. Generous wspace/hspace
# means every panel has clear breathing room, no need for cramped tricks.
fig = plt.figure(figsize=(18.0, 11.0), facecolor="white")
outer = gridspec.GridSpec(
    2, 2, figure=fig,
    left=0.030, right=0.995, top=0.960, bottom=0.075,
    wspace=0.22, hspace=0.32,
)

# ── (a) Dendrogram + clinical annotation strips ──────────────────────────
inner_a = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=outer[0, 0],
    width_ratios=[2.4, 0.45, 3.2], wspace=0.020,
)
ax_a_dn = fig.add_subplot(inner_a[0, 0])
ax_a_id = fig.add_subplot(inner_a[0, 1])
ax_a_an = fig.add_subplot(inner_a[0, 2])

# dendrogram (right-orientation, inverted x for tree-towards-leaves)
def color_func_a(cid):
    n = len(labels_3)
    link_cols = {}
    for i, (a_, b_, *_) in enumerate(Z_pat):
        a_, b_ = int(a_), int(b_)
        ca = labels_3[a_] if a_ < n else link_cols.get(a_, None)
        cb = labels_3[b_] if b_ < n else link_cols.get(b_, None)
        link_cols[n + i] = ca if ca == cb else -1
    cl = link_cols.get(cid, -1)
    return "#888888" if cl == -1 else IT_COL.get(fc_to_it.get(cl), "#888")
dendrogram(Z_pat, ax=ax_a_dn, orientation="right", no_labels=True,
           color_threshold=None, above_threshold_color="#888",
           link_color_func=color_func_a)
ax_a_dn.set_xlabel("Ward linkage distance", fontsize=10.5)
ax_a_dn.spines["left"].set_visible(False)
ax_a_dn.set_yticks([]); ax_a_dn.invert_xaxis()
ax_a_dn.set_title("Immunotype discovery", fontsize=12.5, pad=6, loc="left")

# patient ID column
n_pat_a = len(order_rows_a)
ytick_a = [5 + 10*i for i in range(n_pat_a)]
ax_a_id.set_xlim(0, 1); ax_a_id.set_ylim(0, n_pat_a*10); ax_a_id.invert_yaxis()
for s in ax_a_id.spines.values(): s.set_visible(False)
ax_a_id.set_xticks([]); ax_a_id.set_yticks([])
for i, row_ix in enumerate(order_rows_a):
    pid = int(feat["pid"].iloc[row_ix])
    ax_a_id.text(0.5, ytick_a[i], f"P{pid:02d}", ha="center", va="center",
                  fontsize=11, fontweight="bold")
ax_a_dn.set_ylim(ax_a_id.get_ylim())

# annotation strip
event_cols = ["ILD","myositis","arthritis","rash"]
ax_a_an.set_xlim(0, 6); ax_a_an.set_ylim(0, n_pat_a*10); ax_a_an.invert_yaxis()
for s in ax_a_an.spines.values(): s.set_visible(False)
ax_a_an.set_xticks([]); ax_a_an.set_yticks([])
col_x_centres = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
for x_c, hdr in zip(col_x_centres, ["antibody","immunotype"] + event_cols):
    ax_a_an.text(x_c, -0.5, hdr, ha="left", va="bottom",
                  fontsize=11.0, color="#333", rotation=45,
                  rotation_mode="anchor", fontweight="bold")
ROW_H = 9.0
clin_by_pid_a = clin.set_index("pid")
for i, row_ix in enumerate(order_rows_a):
    pid = int(feat["pid"].iloc[row_ix])
    ab = feat["antibody"].iloc[row_ix]; it = feat["IT"].iloc[row_ix]
    cy = ytick_a[i]; y0 = cy - ROW_H/2
    ax_a_an.add_patch(Rectangle((0.10, y0), 0.80, ROW_H,
                                  facecolor=GROUP_PALETTE.get(ab,"#888"),
                                  edgecolor="white", lw=0.6))
    ax_a_an.text(0.5, cy, ab, ha="center", va="center",
                  fontsize=10.0, fontweight="bold", color="white")
    ax_a_an.add_patch(Rectangle((1.10, y0), 0.80, ROW_H,
                                  facecolor=IT_COL.get(it,"#888"),
                                  edgecolor="white", lw=0.6))
    ax_a_an.text(1.5, cy, it, ha="center", va="center",
                  fontsize=10.0, fontweight="bold", color="white")
    for j, ev in enumerate(event_cols):
        present = clin_by_pid_a.loc[pid, ev] == 1 if pid in clin_by_pid_a.index else 0
        col = EVENT_COL_PRES if present else EVENT_COL_ABS
        ax_a_an.add_patch(Rectangle((2.10 + j, y0), 0.80, ROW_H,
                                      facecolor=col, edgecolor=EVENT_BORDER, lw=0.6))
fig.text(0.020, 0.962, "a", fontsize=24, fontweight="bold", ha="left", va="top")


# ── (b) 4 clinical phenotype panels ──────────────────────────────────────
inner_b = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[0, 1],
                                           wspace=0.55)
ax_b_ck = fig.add_subplot(inner_b[0, 0])
ax_b_age = fig.add_subplot(inner_b[0, 1])
ax_b_my = fig.add_subplot(inner_b[0, 2])
ax_b_dlco = fig.add_subplot(inner_b[0, 3])

def kw_p(values_per_group):
    cleaned = [v[~np.isnan(v)] for v in values_per_group]
    if any(len(v) < 2 for v in cleaned): return np.nan
    try:
        _, p = kruskal(*cleaned); return p
    except: return np.nan

positions = np.arange(len(IT_ORDER))
rng_b = np.random.default_rng(0)

def render_box(ax, key, ylabel, title_fmt, log=False, scatter_size=22):
    groups = [clin.loc[clin["immunotype"]==it, key].dropna().values for it in IT_ORDER]
    parts = ax.boxplot(groups, positions=positions, widths=0.6,
                       patch_artist=True, showfliers=False,
                       whiskerprops=dict(lw=0.7, color="#444"),
                       capprops=dict(lw=0.7, color="#444"),
                       medianprops=dict(lw=1.0, color="#222"),
                       boxprops=dict(lw=0.5, edgecolor="#444"))
    for box, it in zip(parts["boxes"], IT_ORDER):
        box.set_facecolor(IT_COL[it]); box.set_alpha(0.55)
    for i, it in enumerate(IT_ORDER):
        v = clin.loc[clin["immunotype"]==it, key].dropna().values
        xs = i + rng_b.uniform(-0.13, 0.13, len(v))
        ax.scatter(xs, v, s=scatter_size, c=IT_COL[it], edgecolor="white",
                   linewidth=0.4, zorder=3)
    if log: ax.set_yscale("log")
    ax.set_xticks(positions); ax.set_xticklabels(IT_ORDER, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10.5)
    p = kw_p(groups)
    ax.set_title(title_fmt.format(p=p), fontsize=10.5, pad=5)

render_box(ax_b_ck, "CK", "CK (U/L, log)",
           "CK\n(KW P = {p:.2f})", log=True)
# annotate P08 outlier
v_it1 = clin.loc[clin["immunotype"]=="IT1","CK"].dropna().values
if len(v_it1) and v_it1.max() > 1000:
    idx_max = np.argmax(v_it1)
    ax_b_ck.annotate("P08", (0 + 0.05, v_it1[idx_max]),
                     xytext=(8, 0), textcoords="offset points", fontsize=9.5)

render_box(ax_b_age, "age", "Age (yr)", "Age\n(KW P = {p:.2f}; IT3 oldest)")

# myositis bar
myo_pct = [100*clin[clin["immunotype"]==it]["myositis"].mean() for it in IT_ORDER]
bars = ax_b_my.bar(positions, myo_pct,
                    color=[IT_COL[it] for it in IT_ORDER], alpha=0.78,
                    edgecolor="white", lw=0.6, width=0.7)
for i, v in enumerate(myo_pct):
    ax_b_my.text(i, v + 3, f"{v:.0f}%", ha="center", va="bottom",
                  fontsize=10.5, fontweight="bold")
ax_b_my.set_xticks(positions); ax_b_my.set_xticklabels(IT_ORDER, fontsize=11)
ax_b_my.set_ylabel("Myositis prev. (%)", fontsize=10.5)
ax_b_my.set_ylim(0, 115)
ax_b_my.set_title("Myositis prevalence\n(IT2 = 100%)", fontsize=10.5, pad=5)

render_box(ax_b_dlco, "DLCO", "DLCO (% pred)",
           "DLCO % predicted\n({p_str})".format(
                p_str=f"KW P = {kw_p([clin.loc[clin['immunotype']==it,'DLCO'].dropna().values for it in IT_ORDER]):.2f}"
                if not np.isnan(kw_p([clin.loc[clin['immunotype']==it,'DLCO'].dropna().values for it in IT_ORDER])) else "n/a"
           ).replace("{p}","{p:.2f}"))
fig.text(0.515, 0.962, "b", fontsize=24, fontweight="bold", ha="left", va="top")


# ── (c) Integrated clinical-immune heatmap with bilateral dendrograms ────
inner_c = gridspec.GridSpecFromSubplotSpec(
    2, 4, subplot_spec=outer[1, 0],
    height_ratios=[0.7, 5.0], width_ratios=[1.4, 0.18, 0.18, 7.5],
    wspace=0.025, hspace=0.025,
)
ax_c_col_dn = fig.add_subplot(inner_c[0, 3])
ax_c_row_dn = fig.add_subplot(inner_c[1, 0])
ax_c_ab     = fig.add_subplot(inner_c[1, 1])
ax_c_it     = fig.add_subplot(inner_c[1, 2])
ax_c_h      = fig.add_subplot(inner_c[1, 3])

dendrogram(Z_col_c, ax=ax_c_col_dn, no_labels=True, color_threshold=0,
           above_threshold_color="#666")
for s in ax_c_col_dn.spines.values(): s.set_visible(False)
ax_c_col_dn.set_xticks([]); ax_c_col_dn.set_yticks([])

dendrogram(Z_row_c, ax=ax_c_row_dn, orientation="left", no_labels=True,
           color_threshold=0, above_threshold_color="#666")
ax_c_row_dn.invert_yaxis()
for s in ax_c_row_dn.spines.values(): s.set_visible(False)
ax_c_row_dn.set_xticks([]); ax_c_row_dn.set_yticks([])

pids_c = list(M_disp_c.index)
ab_strip = [feat.set_index("pid").loc[p,"antibody"] for p in pids_c]
it_strip = [feat.set_index("pid").loc[p,"immunotype"] for p in pids_c]
n_pc = len(pids_c)

for ax_strip, vals, palette, hdr in [
    (ax_c_ab, ab_strip, GROUP_PALETTE, "antibody"),
    (ax_c_it, it_strip, IT_COL,        "immunotype"),
]:
    for i, v in enumerate(vals):
        ax_strip.add_patch(Rectangle((0, i), 1, 1,
                                      facecolor=palette.get(v,"#888"),
                                      edgecolor="white", lw=0.5))
    ax_strip.set_xlim(0,1); ax_strip.set_ylim(0, n_pc); ax_strip.invert_yaxis()
    ax_strip.set_xticks([]); ax_strip.set_yticks([])
    for s in ax_strip.spines.values(): s.set_visible(False)
    ax_strip.text(0.5, -0.5, hdr, ha="center", va="bottom",
                   fontsize=10.0, rotation=45, rotation_mode="anchor",
                   fontweight="bold")

HEAT_DIV = LinearSegmentedColormap.from_list("hd",
    ["#1A3A6B","#2980B9","#7FB3D3","#F4D03F","#E67E22","#C0392B","#7B241C"], N=256)
features_c = list(M_disp_c.columns)
patient_labels = [f"P{int(p):02d} ({a})" for p, a in zip(pids_c, ab_strip)]
# pcolormesh + rasterized=False = true vector heatmap (each cell is a polygon)
nrows_c, ncols_c = M_disp_c.shape
im_c = ax_c_h.pcolormesh(
    np.arange(ncols_c + 1) - 0.5, np.arange(nrows_c + 1) - 0.5,
    M_disp_c.values, cmap=HEAT_DIV, vmin=-2, vmax=2,
    edgecolors="none", rasterized=False, shading="flat",
)
ax_c_h.invert_yaxis()
ax_c_h.set_xlim(-0.5, ncols_c - 0.5)
ax_c_h.set_ylim(nrows_c - 0.5, -0.5)
ax_c_h.set_aspect("auto")
ax_c_h.set_xticks(range(len(features_c)))
xtick_labels = [f.replace("_score"," score") if f.endswith("_score") else f
                 for f in features_c]
ax_c_h.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=9.5)
for i, (f, tick) in enumerate(zip(features_c, ax_c_h.get_xticklabels())):
    if f in CLINICAL_FEATURES:
        tick.set_color("#000"); tick.set_fontweight("bold")
    elif f in SIGNATURE_FEATURES: tick.set_color("#7D3C98")
    else: tick.set_color("#1F618D")
ax_c_h.set_yticks(range(n_pc))
ax_c_h.set_yticklabels(patient_labels, fontsize=10.0)
ax_c_h.yaxis.tick_right(); ax_c_h.tick_params(left=False)
fig.text(0.020, 0.480, "c", fontsize=24, fontweight="bold", ha="left", va="top")


# ── (d) Bootstrap matrix + per-patient stability bar ────────────────────
inner_d = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1, 1],
                                            width_ratios=[1.5, 1.0], wspace=0.30)
ax_d_m = fig.add_subplot(inner_d[0, 0])
ax_d_b = fig.add_subplot(inner_d[0, 1])

im_d = ax_d_m.pcolormesh(
    np.arange(n_pat + 1) - 0.5, np.arange(n_pat + 1) - 0.5,
    M_boot, cmap="magma", vmin=0, vmax=1,
    edgecolors="none", rasterized=False, shading="flat",
)
ax_d_m.invert_yaxis()
ax_d_m.set_xlim(-0.5, n_pat - 0.5)
ax_d_m.set_ylim(n_pat - 0.5, -0.5)
ax_d_m.set_aspect("equal")
ax_d_m.set_xticks(range(n_pat))
ax_d_m.set_yticks(range(n_pat))
labels_xy = [f"P{int(r['pid']):02d}·{r['antibody']}" for _, r in feat_d.iterrows()]
ax_d_m.set_xticklabels(labels_xy, rotation=45, ha="right", fontsize=9.0)
ax_d_m.set_yticklabels(labels_xy, fontsize=9.0)
# IT colour strip placed FAR LEFT of the matrix so it never collides with
# the patient row labels (which extend ~1.4 units leftward from x=-0.5).
# IT colour strip and IT label sit cleanly to the left of patient row
# labels; patient labels then sit cleanly to the left of the matrix.
strip_w = 0.35
strip_x = -2.50
for i, it in enumerate(feat_d["IT"].values):
    ax_d_m.add_patch(Rectangle((strip_x, i - 0.5), strip_w, 1,
                                 facecolor=IT_COL[it], edgecolor="white",
                                 lw=0.5, clip_on=False))
for it in IT_ORDER:
    rows = np.where(feat_d["IT"].values == it)[0]
    if len(rows):
        ax_d_m.text(strip_x - 0.15, (rows.min()+rows.max())/2, it,
                     ha="right", va="center",
                     fontsize=11.5, fontweight="bold", color=IT_COL[it],
                     clip_on=False)
ax_d_m.set_xlim(strip_x - 1.10, n_pat - 0.5)
for i in range(n_pat):
    for j in range(n_pat):
        v = M_boot[i,j]
        if not np.isnan(v):
            tc = "white" if v < 0.5 else "#222"
            ax_d_m.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=7.5, color=tc)
cbar_d = fig.colorbar(im_d, ax=ax_d_m, fraction=0.038, pad=0.030, shrink=0.85)
cbar_d.set_label("Co-cluster prob.", fontsize=10.5)
cbar_d.ax.tick_params(labelsize=9); cbar_d.outline.set_linewidth(0.4)
ax_d_m.set_title(f"Bootstrap (B = {B:,})", fontsize=11.5, pad=5)

# stability bar
boot_p = feat_d.sort_values("within_IT_stab", ascending=False).reset_index(drop=True)
y_pos = np.arange(len(boot_p))[::-1]
colors_d = [IT_COL[it] for it in boot_p["IT"].values]
ax_d_b.barh(y_pos, boot_p["within_IT_stab"].values, color=colors_d,
             alpha=0.85, edgecolor="white", height=0.72)
for i, v in enumerate(boot_p["within_IT_stab"].values):
    ax_d_b.text(v + 0.02, y_pos[i], f"{v:.2f}", va="center", fontsize=9.5)
ax_d_b.axvline(mean_stab, color="#222", lw=0.7, ls="--",
                label=f"mean = {mean_stab:.2f}")
ax_d_b.set_yticks(y_pos)
ax_d_b.set_yticklabels([f"P{int(r['pid']):02d}·{r['IT']}"
                         for _, r in boot_p.iterrows()], fontsize=9.5)
ax_d_b.set_xlim(0, 1.20)
ax_d_b.set_xlabel("Within-IT mean prob.", fontsize=10.5)
ax_d_b.set_title(f"Per-patient stability\n(mean = {mean_stab:.2f})",
                  fontsize=11.5, pad=5)
ax_d_b.legend(fontsize=9.5, frameon=False, loc="lower right")
fig.text(0.515, 0.480, "d", fontsize=24, fontweight="bold", ha="left", va="top")


# ─── suptitle + save ─────────────────────────────────────────────────────
# Figure title removed (journal convention).
out_pdf = f"{OUT_DIR}/MainFig4_immunotypes.pdf"
out_png = f"{OUT_DIR}/MainFig4_immunotypes.png"
fig.savefig(out_pdf, dpi=300, bbox_inches="tight",
             facecolor="white", edgecolor="none")
fig.savefig(out_png, dpi=300, bbox_inches="tight",
             facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Saved (vector): {out_pdf}")
print(f"Saved (raster): {out_png}")
