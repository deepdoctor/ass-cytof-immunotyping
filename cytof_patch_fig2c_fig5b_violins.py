"""
Native rebuild of Fig 2(c) and Fig 5(b) — top differential marker violins,
trimmed from 9 to 6 panels per the figure review.

Fig 2(c) — global top 6 markers by KW statistic across the 4 antibody groups.
Fig 5(b) — top differential marker per cell-type (lineage), 6 most-different
            populations, derived from h5ad cell-level data.

Both panels render KW p-values with mathtext superscripts (no Unicode
glyph issues), with selection rationale documented per panel.
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
from scipy.stats import kruskal
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

GROUP_PALETTE = {"Jo-1": "#C0392B", "PL-12": "#2471A3",
                 "EJ":   "#1A7A4A", "PL-7":  "#7D3C98"}
GROUP_ORDER = ["Jo-1", "PL-12", "EJ", "PL-7"]
GROUP_LABEL_NB = {"Jo-1": "Jo‑1", "PL-12": "PL‑12",
                  "EJ":   "EJ",   "PL-7":  "PL‑7"}

plt.rcParams.update({
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":            10.0,
    "axes.titlesize":       11.0,
    "axes.labelsize":       9.5,
    "xtick.labelsize":      9.0,
    "ytick.labelsize":      9.0,
    "axes.linewidth":       0.9,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
    "savefig.dpi":          300,
    "figure.dpi":           300,
})

def fmt_p(p):
    if p == 0 or p < 1e-300:
        return r"$P < 10^{-300}$"
    if p < 1e-4:
        exp = int(np.floor(np.log10(p)))
        coef = p / 10**exp
        return rf"$P = {coef:.1f}\times 10^{{{exp}}}$"
    if p < 1e-2:
        return f"P = {p:.1e}"
    return f"P = {p:.3f}"


def render_violin_grid(panels, save_name, suptitle, out_dir,
                       n_cells_subsample=500, fig_w=11.0, fig_h=4.5):
    """panels: list of dicts with keys {title, vals_per_group, padj}."""
    n = len(panels)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h * nrows))
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)

    rng = np.random.default_rng(0)
    for ai, panel in enumerate(panels):
        ax = axes[ai // ncols][ai % ncols]
        data_list = panel["vals_per_group"]
        colors = [GROUP_PALETTE[g] for g in GROUP_ORDER]

        vp = ax.violinplot(data_list, positions=range(len(GROUP_ORDER)),
                           showmedians=True, showextrema=False,
                           widths=0.72)
        for body, col in zip(vp["bodies"], colors):
            body.set_facecolor(col); body.set_alpha(0.65)
            body.set_edgecolor(col); body.set_linewidth(0.6)
        vp["cmedians"].set_color("#222"); vp["cmedians"].set_linewidth(0.9)

        for k, (g, col) in enumerate(zip(GROUP_ORDER, colors)):
            vals = data_list[k]
            if len(vals) > n_cells_subsample:
                vals = rng.choice(vals, n_cells_subsample, replace=False)
            jitter = rng.uniform(-0.10, 0.10, len(vals))
            ax.scatter(k + jitter, vals, c=col, s=2.5, alpha=0.25,
                       zorder=3, linewidths=0)

        ax.set_xticks(range(len(GROUP_ORDER)))
        ax.set_xticklabels([GROUP_LABEL_NB[g] for g in GROUP_ORDER],
                           rotation=0, fontsize=8.5)
        ax.set_ylabel("arcsinh expression", fontsize=8.5)
        ax.set_title(f"{panel['title']}    {fmt_p(panel['padj'])}",
                     fontsize=10, fontweight="bold", pad=4)
        ax.yaxis.grid(False)

    # hide extra panels if any
    for ai in range(n, nrows * ncols):
        axes[ai // ncols][ai % ncols].axis("off")

    fig.suptitle(suptitle, fontsize=11.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=1.5, w_pad=0.8)
    fig.savefig(f"{out_dir}/{save_name}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_dir}/{save_name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir}/{save_name}.png")


# ── load h5ad once ────────────────────────────────────────────────────────
print("Loading h5ad …")
a = ad.read_h5ad(H5AD)
a.obs_names_make_unique()
a.obs["cell_type"] = a.obs["leiden"].astype(str).map(CLUSTER_ANNOTATION).fillna("Unannotated")
# drop artifacts
a = a[~a.obs["cell_type"].isin(ARTIFACTS)].copy()
print(f"  {a.n_obs:,} cells after dropping artifacts")

# ════════════════════════════════════════════════════════════════════════════
# Fig 2(c) — top 6 global markers by KW H statistic
# ════════════════════════════════════════════════════════════════════════════
print("[1/2] Fig 2(c) — global top 6 violins")
de_g = pd.read_csv(f"{OUT_DIR}/06_differential_markers.csv")
top6_global = de_g.sort_values("KW_stat", ascending=False).head(6)
print("  selected:", top6_global["marker"].tolist())

panels_g = []
for _, r in top6_global.iterrows():
    mk = r["marker"]
    if mk not in a.var_names:
        continue
    j = list(a.var_names).index(mk)
    Xj = a.X[:, j]
    Xj = Xj.toarray().ravel() if hasattr(Xj, "toarray") else np.asarray(Xj).ravel()
    vals = [Xj[a.obs["group"].values == g] for g in GROUP_ORDER]
    panels_g.append({"title": mk, "vals_per_group": vals,
                     "padj": float(r["padj"])})

render_violin_grid(panels_g, "07_top_marker_violins",
                   "Top 6 differential markers across autoantibody groups",
                   OUT_DIR, fig_w=10.5, fig_h=4.0)


# ════════════════════════════════════════════════════════════════════════════
# Fig 5(b) — top differential marker per the 6 most-divergent cell types
# ════════════════════════════════════════════════════════════════════════════
print("[2/2] Fig 5(b) — within-cell-type top 6 violins")
de_w = pd.read_csv(f"{OUT_DIR}/08_within_celltype_markers.csv")

# keep only canonical cell types
de_w = de_w[~de_w["cell_type"].isin(ARTIFACTS)].copy()

# Pick top marker per cell type (smallest padj per cell type), then keep
# the 6 cell types with the smallest top-marker padj.
top_per_ct = (de_w.sort_values(["cell_type", "padj"])
                  .groupby("cell_type", as_index=False).first())
top6_ct = top_per_ct.sort_values("padj").head(6)
print("  selected (cell_type → top marker):")
for _, r in top6_ct.iterrows():
    print(f"    {r['cell_type']} → {r['marker']}  (padj={r['padj']:.2e})")

panels_w = []
for _, r in top6_ct.iterrows():
    ct = r["cell_type"]; mk = r["marker"]
    mask = (a.obs["cell_type"].values == ct)
    if mask.sum() < 30 or mk not in a.var_names:
        continue
    j = list(a.var_names).index(mk)
    Xj = a.X[mask, j]
    Xj = Xj.toarray().ravel() if hasattr(Xj, "toarray") else np.asarray(Xj).ravel()
    grp_vals = a.obs["group"].values[mask]
    vals = [Xj[grp_vals == g] for g in GROUP_ORDER]
    panels_w.append({"title": f"{ct}\n{mk}", "vals_per_group": vals,
                     "padj": float(r["padj"])})

render_violin_grid(panels_w, "08b_within_celltype_violins",
                   "Top differential marker per cell-type (6 most-divergent)",
                   OUT_DIR, fig_w=10.5, fig_h=4.4)
