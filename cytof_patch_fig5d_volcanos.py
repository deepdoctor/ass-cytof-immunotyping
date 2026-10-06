"""
Native rebuild of Fig 5(d): per-cell-type EJ vs PL-7 volcano, restricted
to 4 most-informative populations (was 9, with overlapping labels).

Cell types selected per the figure review:
  - CD4 Central Memory T
  - CD8 Effector T
  - Classical Monocyte
  - NK cell

Source: cytof_analyzed_v3.h5ad (cell-level arcsinh expression).
Per (cell type × marker) pair: KW between EJ and PL-7 (Mann-Whitney U
on cells), Cliff's δ effect size, BH-FDR across the 42 markers within
each cell type. Label only top 8 per panel by |t| with collision-aware
placement.
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
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

OUT_DIR  = "./cytof_output_nature"
H5AD     = "/Users/yichen/Desktop/collaboration/raw/cytof_output_v3/cytof_analyzed_v3.h5ad"

CLUSTER_ANNOTATION = {
    "0":  "Classical Monocyte",   "1":  "Naive B",            "2":  "CD8 Effector T",
    "3":  "CD4 Naive T",          "4":  "CD4 Central Memory T","5": "Non-classical Monocyte",
    "6":  "NK cell",              "7":  "CD4 Th1-like",       "8":  "Classical Monocyte",
    "9":  "CD4 Effector Memory T","10": "Inflammatory Monocyte","11":"CD8 Naive T",
    "12": "γδ T cell",            "13": "mDC",                "14": "pDC",
    "15": "T-Myeloid doublets",   "16": "CD16+ Granulocyte",
}

SELECTED = ["CD4 Central Memory T", "CD8 Effector T",
            "Classical Monocyte",    "NK cell"]

plt.rcParams.update({
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":            10.0,
    "axes.titlesize":       11.5,
    "axes.labelsize":       10.0,
    "xtick.labelsize":      9.5,
    "ytick.labelsize":      9.5,
    "legend.fontsize":      9.0,
    "axes.linewidth":       0.9,
    "xtick.major.width":    0.7,
    "ytick.major.width":    0.7,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
    "savefig.dpi":          300,
    "figure.dpi":           300,
})

print("Loading h5ad …")
a = ad.read_h5ad(H5AD)
a.obs_names_make_unique()
a.obs["cell_type"] = a.obs["leiden"].astype(str).map(CLUSTER_ANNOTATION).fillna("Unannotated")

# subset to EJ vs PL-7 + 4 cell types
mask_grp = a.obs["group"].isin(["EJ", "PL-7"])
a_sub = a[mask_grp].copy()
print(f"  EJ/PL-7 subset: {a_sub.n_obs:,} cells")

def compute_de(adata_sub, ct):
    """Per-marker MW (EJ vs PL-7) + Cliff's δ + BH-FDR for one cell type."""
    sel = adata_sub.obs["cell_type"].values == ct
    asub = adata_sub[sel]
    grp_ej   = asub.obs["group"].values == "EJ"
    grp_pl7  = asub.obs["group"].values == "PL-7"
    # need both groups
    if grp_ej.sum() < 30 or grp_pl7.sum() < 30:
        return None
    Xe = asub.X[grp_ej]
    Xp = asub.X[grp_pl7]
    Xe = Xe.toarray() if hasattr(Xe, "toarray") else np.asarray(Xe)
    Xp = Xp.toarray() if hasattr(Xp, "toarray") else np.asarray(Xp)
    n_ej, n_pl = Xe.shape[0], Xp.shape[0]
    rows = []
    for j, mk in enumerate(asub.var_names):
        ve = Xe[:, j]; vp = Xp[:, j]
        try:
            u, p = mannwhitneyu(ve, vp, alternative="two-sided")
        except Exception:
            continue
        # Cliff's δ from MW U
        delta = (2.0 * u / (n_ej * n_pl)) - 1.0
        diff  = ve.mean() - vp.mean()  # positive = higher in EJ
        rows.append({"marker": mk, "delta": delta,
                     "diff": diff, "p": p,
                     "n_ej": n_ej, "n_pl": n_pl})
    res = pd.DataFrame(rows)
    if len(res) == 0:
        return res
    _, padj, _, _ = multipletests(res["p"].values, method="fdr_bh")
    res["padj"] = padj
    res["neglog_q"] = -np.log10(np.clip(res["padj"].values, 1e-300, 1.0))
    return res

# Compute DE for each selected cell type
results = {}
for ct in SELECTED:
    print(f"  DE for {ct} …")
    res = compute_de(a_sub, ct)
    if res is None:
        print(f"    [skip] insufficient cells")
        continue
    print(f"    {(res['padj']<0.05).sum()} markers at FDR<0.05")
    results[ct] = res

# ── plot 4 volcanos in 2×2 grid ───────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.4))
axes_flat = axes.flatten()

# Diverging color: blue = up in PL-7, red = up in EJ
EJ_COL  = "#1A7A4A"  # green per group palette
PL7_COL = "#7D3C98"

for ai, ct in enumerate(SELECTED):
    ax = axes_flat[ai]
    if ct not in results:
        ax.text(0.5, 0.5, f"{ct}\n(insufficient n)", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="#888")
        ax.set_xticks([]); ax.set_yticks([])
        continue
    r = results[ct]
    sig = r["padj"] < 0.05
    # NS points: grey
    ax.scatter(r.loc[~sig, "delta"], r.loc[~sig, "neglog_q"],
               s=22, c="#cccccc", alpha=0.55, edgecolor="none",
               linewidths=0, zorder=1)
    # Up in EJ
    sig_ej = sig & (r["delta"] > 0)
    ax.scatter(r.loc[sig_ej, "delta"], r.loc[sig_ej, "neglog_q"],
               s=32, c=EJ_COL, alpha=0.92, edgecolor="white",
               linewidths=0.4, zorder=3)
    # Up in PL-7
    sig_pl = sig & (r["delta"] < 0)
    ax.scatter(r.loc[sig_pl, "delta"], r.loc[sig_pl, "neglog_q"],
               s=32, c=PL7_COL, alpha=0.92, edgecolor="white",
               linewidths=0.4, zorder=3)
    # threshold lines
    ax.axhline(-np.log10(0.05), color="#888", lw=0.5, ls="--")
    ax.axvline(0, color="#888", lw=0.5)

    # Label top 8 by |delta| among significant
    top = r[sig].assign(absd=lambda d: d["delta"].abs()) \
                .nlargest(8, "absd")
    seen_pos = []
    for _, row in top.iterrows():
        x, y = row["delta"], row["neglog_q"]
        # collision avoidance
        dy = 0
        for px, py in seen_pos:
            if abs(x - px) < 0.18 and abs(y - py) < 5.0:
                dy += 5.0
        ax.annotate(row["marker"], (x, y),
                    xytext=(3.5, 1.5 + dy), textcoords="offset points",
                    fontsize=7.5, color="#222",
                    bbox=dict(boxstyle="round,pad=0.10",
                              fc="white", ec="none", alpha=0.7))
        seen_pos.append((x, y + dy))

    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("Cliff's δ  (+ = higher in EJ)", fontsize=9.5)
    ax.set_ylabel(r"$-\log_{10}$ q (BH-FDR)", fontsize=9.5)
    n_sig = int(sig.sum())
    ax.set_title(f"{ct}   ({n_sig}/{len(r)} markers, FDR < 0.05)",
                 fontsize=10.5, pad=4)

# legend on figure level
from matplotlib.patches import Patch
fig.legend(
    handles=[
        Patch(facecolor=EJ_COL,  label="up in EJ (cliff's δ > 0)"),
        Patch(facecolor=PL7_COL, label="up in PL-7 (cliff's δ < 0)"),
        Patch(facecolor="#cccccc", label="not significant"),
    ],
    loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3,
    frameon=False, fontsize=9, handlelength=1.0, handletextpad=0.4,
    columnspacing=1.5,
)
fig.suptitle(
    "Per-cell-type marker differential expression: EJ vs PL-7\n"
    "(4 representative cell types; full 9-cell-type panel in Supp. Fig. S12)",
    fontsize=11.5, fontweight="bold", y=0.995,
)
fig.tight_layout(rect=[0, 0.04, 1, 0.95])

out_pdf = f"{OUT_DIR}/20_percelltype_volcano_EJvsPL7.pdf"
out_png = f"{OUT_DIR}/20_percelltype_volcano_EJvsPL7.png"
fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
