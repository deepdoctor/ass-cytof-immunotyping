"""
Native unified vector rebuild of Main Figure 5 — differential expression.

All four panels rendered as matplotlib primitives (no imshow, no pre-rendered
PNGs) so the output PDF is fully editable in Illustrator/Acrobat Pro. Heatmap
panels use pcolormesh+rasterized=False instead of imshow.
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
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

OUT_DIR  = "./cytof_output_nature"
H5AD     = "/Users/yichen/Desktop/collaboration/raw/cytof_output_v3/cytof_analyzed_v3.h5ad"

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
    "CD4 Central Memory T", "CD4 Effector Memory T", "CD4 Naive T",
    "CD4 Th1-like", "CD8 Effector T", "CD8 Naive T",
    "Classical Monocyte", "Inflammatory Monocyte", "Non-classical Monocyte",
    "NK cell", "Naive B", "mDC", "pDC", "γδ T cell",
]
GROUP_PALETTE = {"Jo-1":"#C0392B","PL-12":"#2471A3","EJ":"#1A7A4A","PL-7":"#7D3C98"}
GROUP_ORDER = ["Jo-1","PL-12","EJ","PL-7"]
GROUP_LABEL_NB = {"Jo-1":"Jo-1","PL-12":"PL-12","EJ":"EJ","PL-7":"PL-7"}  # plain ASCII hyphen
EJ_COL  = "#1A7A4A"; PL7_COL = "#7D3C98"

plt.rcParams.update({
    "font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "font.size":10.5,"axes.titlesize":12.0,"axes.labelsize":10.5,
    "xtick.labelsize":9.5,"ytick.labelsize":9.5,"legend.fontsize":9.5,
    "axes.linewidth":0.9,"axes.spines.top":False,"axes.spines.right":False,
    "pdf.fonttype":42,"ps.fonttype":42,"savefig.dpi":300,"figure.dpi":300,
})

def fmt_p(p):
    if p == 0 or p < 1e-300: return r"$P < 10^{-300}$"
    if p < 1e-4:
        e = int(np.floor(np.log10(p))); c = p/10**e
        return rf"$P = {c:.1f}\times 10^{{{e}}}$"
    if p < 1e-2: return f"P = {p:.1e}"
    return f"P = {p:.3f}"

# ─── load h5ad + DE table ──────────────────────────────────────────────────
print("Loading h5ad …")
a = ad.read_h5ad(H5AD)
a.obs_names_make_unique()
a.obs["cell_type"] = a.obs["leiden"].astype(str).map(CLUSTER_ANNOTATION).fillna("Unannotated")
a = a[~a.obs["cell_type"].isin(ARTIFACTS)].copy()

de_w = pd.read_csv(f"{OUT_DIR}/08_within_celltype_markers.csv")
de_w = de_w[~de_w["cell_type"].isin(ARTIFACTS)].copy()


# ════════════════════════════════════════════════════════════════════════════
# FIGURE LAYOUT — 14 × 14 outer 2×2
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18.0, 11.0), facecolor="white")
outer = gridspec.GridSpec(
    2, 2, figure=fig,
    left=0.030, right=0.992, top=0.960, bottom=0.060,
    wspace=0.22, hspace=0.32,
)


# ── (a) Top-25 within-cell-type heatmap (pcolormesh = vector) ────────────
ax_a = fig.add_subplot(outer[0, 0])
de_w["padj_clip"] = de_w["padj"].clip(lower=1e-50, upper=1.0)
de_w["neglog_q"] = -np.log10(de_w["padj_clip"])
mk_top = (de_w.groupby("marker")["neglog_q"].max()
              .sort_values(ascending=False).head(25).index.tolist())
M_a = de_w.pivot_table(index="cell_type", columns="marker",
                        values="neglog_q", aggfunc="max")
M_a = M_a.reindex(index=CT_ORDER, columns=mk_top)

HEAT_FDR = LinearSegmentedColormap.from_list("fdr",
    ["#FAEBD7","#F4A261","#E76F51","#C0392B","#7B241C"], N=256)
nrows_a, ncols_a = M_a.shape
im_a = ax_a.pcolormesh(
    np.arange(ncols_a + 1) - 0.5, np.arange(nrows_a + 1) - 0.5,
    np.where(np.isnan(M_a.values), -1, M_a.values),
    cmap=HEAT_FDR, vmin=0, vmax=50, edgecolors="white", lw=0.4,
    rasterized=False, shading="flat",
)
ax_a.invert_yaxis()
ax_a.set_xticks(np.arange(ncols_a))
ax_a.set_xticklabels(mk_top, rotation=45, ha="right", fontsize=10.0)
ax_a.set_yticks(np.arange(nrows_a))
ax_a.set_yticklabels(CT_ORDER, fontsize=10.5)

THR_NEGLOG = -np.log10(0.05)
for i, ct in enumerate(CT_ORDER):
    for j, mk in enumerate(mk_top):
        v = M_a.values[i, j]
        if np.isnan(v):
            ax_a.text(j, i, "·", ha="center", va="center", fontsize=8.0, color="#888")
        elif v < THR_NEGLOG:
            ax_a.text(j, i, "ns", ha="center", va="center", fontsize=7.5, color="#444")
cbar_a = fig.colorbar(im_a, ax=ax_a, fraction=0.022, pad=0.012, shrink=0.85)
cbar_a.set_label(r"$-\log_{10}$ q (BH-FDR; capped 50)", fontsize=10.5)
cbar_a.ax.tick_params(labelsize=9); cbar_a.outline.set_linewidth(0.4)
ax_a.set_title("Within-cell-type differential markers — top 25\n"
                "(Kruskal–Wallis BH-FDR; full 42-marker matrix in Supp. Fig. S12)",
                fontsize=12, pad=6)
fig.text(0.020, 0.962, "a", fontsize=24, fontweight="bold", ha="left", va="top")


# ── (b) Top differential marker per cell-type — 6 violins ───────────────
WITHIN_CT_TOP = (de_w.sort_values(["cell_type","padj"])
                    .groupby("cell_type", as_index=False).first()
                    .sort_values("padj").head(6))

inner_b = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer[0, 1],
                                            wspace=0.40, hspace=0.55)
rng_b = np.random.default_rng(0)
for bi, (_, r) in enumerate(WITHIN_CT_TOP.iterrows()):
    ax_b = fig.add_subplot(inner_b[bi // 3, bi % 3])
    ct, mk, padj = r["cell_type"], r["marker"], r["padj"]
    if mk not in a.var_names: continue
    j = list(a.var_names).index(mk)
    mask = a.obs["cell_type"].values == ct
    Xj = a.X[mask, j]
    Xj = Xj.toarray().ravel() if hasattr(Xj, "toarray") else np.asarray(Xj).ravel()
    grp_vals = a.obs["group"].values[mask]
    data_list = [Xj[grp_vals == g] for g in GROUP_ORDER]
    parts = ax_b.violinplot(data_list, positions=range(len(GROUP_ORDER)),
                             widths=0.72, showmedians=True, showextrema=False)
    for body, g in zip(parts["bodies"], GROUP_ORDER):
        body.set_facecolor(GROUP_PALETTE[g]); body.set_alpha(0.65)
        body.set_edgecolor(GROUP_PALETTE[g]); body.set_linewidth(0.6)
    parts["cmedians"].set_color("#222"); parts["cmedians"].set_linewidth(0.8)
    for k, (g, col) in enumerate(zip(GROUP_ORDER,
                                       [GROUP_PALETTE[g] for g in GROUP_ORDER])):
        v = data_list[k]
        if len(v) > 500:
            v = rng_b.choice(v, 500, replace=False)
        jitter = rng_b.uniform(-0.10, 0.10, len(v))
        ax_b.scatter(k + jitter, v, c=col, s=2.0, alpha=0.25,
                     zorder=3, linewidths=0)
    ax_b.set_xticks(range(len(GROUP_ORDER)))
    ax_b.set_xticklabels([GROUP_LABEL_NB[g] for g in GROUP_ORDER], fontsize=10.0)
    ax_b.set_ylabel("arcsinh expr.", fontsize=10.0)
    ax_b.set_title(f"{ct}\n{mk}    {fmt_p(padj)}", fontsize=11.0,
                    fontweight="bold", pad=4)
fig.text(0.515, 0.962, "b", fontsize=24, fontweight="bold", ha="left", va="top")


# ── (c) Patient-level cluster bootstrap forest (EJ vs PL-7) ─────────────
ax_c = fig.add_subplot(outer[1, 0])
df_c = pd.read_csv(f"{OUT_DIR}/27b_DE_bootstrap_CI.csv")
ci_clear = ~df_c["crosses_zero"].astype(bool)
df_clear = df_c[ci_clear]
top_ej  = df_clear[df_clear["d"] > 0].nlargest(6, "d")
top_pl7 = df_clear[df_clear["d"] < 0].nsmallest(6, "d")
sub_c = pd.concat([top_ej, top_pl7]).sort_values("d").reset_index(drop=True)

n_c = len(sub_c); y_c = np.arange(n_c)
for i, r in sub_c.iterrows():
    col = EJ_COL if r["d"] > 0 else PL7_COL
    ax_c.plot([r["ci_low"], r["ci_high"]], [y_c[i], y_c[i]],
               color=col, lw=1.6, alpha=0.85, solid_capstyle="butt")
    cap_h = 0.18
    for x_c_pt in [r["ci_low"], r["ci_high"]]:
        ax_c.plot([x_c_pt, x_c_pt], [y_c[i]-cap_h, y_c[i]+cap_h],
                   color=col, lw=1.0, alpha=0.85)
    ax_c.plot(r["d"], y_c[i], "D", color=col, markersize=8,
               markeredgecolor="white", markeredgewidth=0.6, zorder=4)
    if r["q_fdr"] < 0.10:
        stars = "*" if r["q_fdr"] >= 0.01 else ("**" if r["q_fdr"] >= 0.001 else "***")
        ax_c.text(r["ci_high"] + 0.05, y_c[i], stars, ha="left", va="center",
                   fontsize=13, color=col, fontweight="bold")
labels_c = [f"{r['marker']}  ·  {r['cell_type']}" for _, r in sub_c.iterrows()]
ax_c.set_yticks(y_c); ax_c.set_yticklabels(labels_c, fontsize=10.5)
ax_c.axvline(0, color="#222", lw=0.7)
ax_c.set_xlim(-2.5, 2.5)
ax_c.set_xlabel("Cohen's d  (EJ − PL-7)  [patient-level cluster bootstrap, B = 2,000]",
                fontsize=11.0)
ax_c.tick_params(axis="x", labelsize=10)
ax_c.text(-2.4, n_c+0.1, "↑ in PL-7", ha="left", va="bottom",
           fontsize=12, color=PL7_COL, fontweight="bold")
ax_c.text(2.4, n_c+0.1, "↑ in EJ", ha="right", va="bottom",
           fontsize=12, color=EJ_COL, fontweight="bold")
ax_c.set_title("Patient-level cluster-bootstrap effect sizes (EJ vs PL-7)\n"
                "top 6 per direction; * FDR<0.10, ** <0.01, *** <0.001",
                fontsize=12, pad=8)
ax_c.grid(axis="x", color="#eee", lw=0.4); ax_c.set_axisbelow(True)
ax_c.set_ylim(-0.6, n_c + 0.6)
fig.text(0.020, 0.475, "c", fontsize=24, fontweight="bold", ha="left", va="top")


# ── (d) Per-cell-type EJ vs PL-7 volcanos (4 panels) ────────────────────
SELECTED = ["CD4 Central Memory T","CD8 Effector T","Classical Monocyte","NK cell"]

# subset h5ad to EJ+PL-7
mask_grp = a.obs["group"].isin(["EJ","PL-7"])
a_sub = a[mask_grp].copy()

def compute_de(adata_sub, ct):
    sel = adata_sub.obs["cell_type"].values == ct
    asub = adata_sub[sel]
    grp_ej  = asub.obs["group"].values == "EJ"
    grp_pl7 = asub.obs["group"].values == "PL-7"
    if grp_ej.sum() < 30 or grp_pl7.sum() < 30:
        return None
    Xe = asub.X[grp_ej]; Xp = asub.X[grp_pl7]
    Xe = Xe.toarray() if hasattr(Xe,"toarray") else np.asarray(Xe)
    Xp = Xp.toarray() if hasattr(Xp,"toarray") else np.asarray(Xp)
    n_ej, n_pl = Xe.shape[0], Xp.shape[0]
    rows = []
    for jj, mk in enumerate(asub.var_names):
        ve, vp = Xe[:, jj], Xp[:, jj]
        try: u, p = mannwhitneyu(ve, vp, alternative="two-sided")
        except: continue
        delta = (2.0 * u / (n_ej * n_pl)) - 1.0
        rows.append({"marker": mk, "delta": delta, "p": p,
                     "n_ej": n_ej, "n_pl": n_pl})
    res = pd.DataFrame(rows)
    if len(res) == 0: return res
    _, padj, _, _ = multipletests(res["p"].values, method="fdr_bh")
    res["padj"] = padj
    res["neglog_q"] = -np.log10(np.clip(res["padj"].values, 1e-300, 1.0))
    return res

inner_d = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[1, 1],
                                            wspace=0.30, hspace=0.40)
for di, ct in enumerate(SELECTED):
    ax_d = fig.add_subplot(inner_d[di // 2, di % 2])
    res = compute_de(a_sub, ct)
    if res is None or len(res) == 0:
        ax_d.text(0.5, 0.5, f"{ct}\n(insufficient n)", ha="center", va="center",
                   transform=ax_d.transAxes, fontsize=12, color="#888")
        continue
    sig = res["padj"] < 0.05
    ax_d.scatter(res.loc[~sig,"delta"], res.loc[~sig,"neglog_q"],
                  s=18, c="#cccccc", alpha=0.55, edgecolor="none",
                  linewidths=0, zorder=1)
    sig_ej = sig & (res["delta"] > 0); sig_pl = sig & (res["delta"] < 0)
    ax_d.scatter(res.loc[sig_ej,"delta"], res.loc[sig_ej,"neglog_q"],
                  s=28, c=EJ_COL, alpha=0.92, edgecolor="white",
                  linewidths=0.4, zorder=3)
    ax_d.scatter(res.loc[sig_pl,"delta"], res.loc[sig_pl,"neglog_q"],
                  s=28, c=PL7_COL, alpha=0.92, edgecolor="white",
                  linewidths=0.4, zorder=3)
    ax_d.axhline(-np.log10(0.05), color="#888", lw=0.5, ls="--")
    ax_d.axvline(0, color="#888", lw=0.5)
    top = res[sig].assign(absd=lambda d: d["delta"].abs()).nlargest(8, "absd")
    # Use adjustText for proper non-overlapping label placement with leader lines
    from adjustText import adjust_text
    texts = []
    for _, row in top.iterrows():
        t = ax_d.text(row["delta"], row["neglog_q"], row["marker"],
                       fontsize=9.0, color="#222", fontweight="bold")
        texts.append(t)
    adjust_text(
        texts, ax=ax_d,
        expand=(1.6, 1.8),
        arrowprops=dict(arrowstyle="-", color="#888", lw=0.4, alpha=0.8),
        force_text=(0.8, 1.0),
        force_pull=(0.05, 0.05),
        max_move=(30, 50),
        min_arrow_len=2,
    )
    ax_d.set_xlim(-1.05, 1.05)
    ax_d.set_xlabel("Cliff's δ  (+ = higher in EJ)", fontsize=10.5)
    ax_d.set_ylabel(r"$-\log_{10}$ q (BH-FDR)", fontsize=10.5)
    ax_d.tick_params(labelsize=9.5)
    n_sig = int(sig.sum())
    ax_d.set_title(f"{ct}   ({n_sig}/{len(res)} markers, FDR<0.05)",
                    fontsize=11.5, pad=4)
fig.text(0.515, 0.475, "d", fontsize=24, fontweight="bold", ha="left", va="top")


# ─── suptitle + save ─────────────────────────────────────────────────────
# Figure title removed (journal convention).
out_pdf = f"{OUT_DIR}/MainFig5_DE.pdf"
out_png = f"{OUT_DIR}/MainFig5_DE.png"
fig.savefig(out_pdf, dpi=300, bbox_inches="tight",
             facecolor="white", edgecolor="none")
fig.savefig(out_png, dpi=300, bbox_inches="tight",
             facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Saved (vector): {out_pdf}")
print(f"Saved (raster): {out_png}")
