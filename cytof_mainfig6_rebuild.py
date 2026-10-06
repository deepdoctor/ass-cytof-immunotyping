"""
cytof_mainfig6_rebuild.py
=========================
Native 6-panel rebuild of Main Figure 6 — pure cross-cohort external validation.

Replaces the broken composite (which nested entire prior figures inside subpanels)
with a single coherent figure built directly from CSV inputs:

  (a) IT1 score across 5 muscle-disease diagnoses (NT/IMNM/IBM/DM/AS), n=165
      KW + per-pair MW with Cliff's δ effect sizes.
  (b) Cliff's δ forest plot — AS vs each comparator (5-group pairwise).
  (c) Within-AS heterogeneity — 18 AS patients ranked by IT1, NT/IMNM 5-95%
      reference bands.
  (d) IT1-high vs IT1-low within-AS volcano (3,266 genes at FDR<0.10).
  (e) Pathway enrichment in IT1-high — IFN-γ, complement, cytolytic, myeloid,
      antibody, T cell activation (hypergeometric, BH-q).
  (f) Top 30 DE genes ranked by |t|, colored by direction, pathway category strip.

Output: MainFig6_external_validation.{pdf,png}
"""
import os, warnings, ast
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

OUT_DIR = "./cytof_output_nature"

# ─────────────────────────────────────────────────────────────────────────────
# Figure-local typography (overrides the 22pt nature_style defaults that are
# tuned for downscaled composites — this figure is consumed natively at 7.09").
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":            10.5,
    "axes.titlesize":       12.0,
    "axes.labelsize":       10.5,
    "xtick.labelsize":      9.5,
    "ytick.labelsize":      9.5,
    "legend.fontsize":      9.5,
    "axes.linewidth":       0.9,
    "xtick.major.width":    0.7,
    "ytick.major.width":    0.7,
    "xtick.major.size":     3.0,
    "ytick.major.size":     3.0,
    "lines.linewidth":      1.1,
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

DX_ORDER = ["NT", "IMNM", "IBM", "DM", "AS"]
DX_COL = {
    "NT":   "#7F8C8D",
    "IMNM": "#2980B9",
    "IBM":  "#8E44AD",
    "DM":   "#E67E22",
    "AS":   "#C0392B",
}
PATHWAY_COL = {
    "IFN-γ":      "#C0392B",
    "Complement": "#1F618D",
    "Cytolytic":  "#7D3C98",
    "Myeloid":    "#E67E22",
    "Antibody":   "#117A65",
    "T cell":     "#B7950B",
    "Other":      "#7F8C8D",
}

PATHWAYS = {
    "IFN-γ": {"STAT1","IRF1","IRF7","IRF8","IRF9","GBP1","GBP5","CXCL9","CXCL10",
              "CXCL11","HLA-DRA","HLA-DRB1","HLA-DPA1","HLA-DPB1","CD74",
              "PSMB8","PSMB9","TAP1","TAP2","MX1","ISG15","OAS1","OAS2","OAS3",
              "IFI44","IFIT1","IFIT3","IFITM1","IFITM3"},
    "Complement": {"C1QA","C1QB","C1QC","C1R","C1S","C2","C3","C4A","C4B","C5",
                   "CFB","C3AR1","C5AR1","ITGAM","ITGAX","CR1","VSIG4"},
    "Cytolytic": {"GZMA","GZMB","GZMK","GZMH","GZMM","PRF1","GNLY","NKG7",
                  "KLRD1","KLRK1","FGFBP2","FASLG"},
    "Myeloid":   {"CD14","CD68","CD163","VCAN","S100A8","S100A9","S100A12",
                  "FCGR1A","FCGR3A","CSF1R","LYZ","MS4A6A","MS4A7","CLEC10A"},
    "Antibody":  {"IGHG1","IGHG3","IGHG4","IGHA1","IGHA2","JCHAIN","MZB1","XBP1",
                  "PRDM1","CD38"},
    "T cell":    {"CD3D","CD3E","CD3G","CD4","CD8A","CD8B","LCK","ZAP70","IL2RA",
                  "CD69","ICOS"},
}

def gene_pathway(symbol):
    for p, gset in PATHWAYS.items():
        if symbol in gset:
            return p
    return "Other"


def fmt_p(p):
    """Publication-grade P-value formatter — uniform mathtext superscript."""
    if p == 0 or p < 1e-300:
        return r"$P < 10^{-300}$"
    if p < 1e-2:
        # Always render as coef × 10^exp via mathtext (uniform across magnitudes)
        exp = int(np.floor(np.log10(p)))
        coef = p / 10**exp
        return rf"$P = {coef:.1f}\times 10^{{{exp}}}$"
    if p < 0.05:
        return f"P = {p:.3f}"
    return f"P = {p:.2f}"


def sig_stars(p):
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 0.05: return "*"
    return "ns"


# ═════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═════════════════════════════════════════════════════════════════════════════
scores  = pd.read_csv(f"{OUT_DIR}/S4_validation_scores.csv", index_col=0)
stats   = pd.read_csv(f"{OUT_DIR}/S4_validation_stats.csv")
pw      = pd.read_csv(f"{OUT_DIR}/S5_AS_pathway_enrichment.csv")
de      = pd.read_csv(f"{OUT_DIR}/S5_AS_IT1high_vs_IT1low_DE.csv")

print(f"Loaded: {len(scores)} samples, {len(de)} DE rows, "
      f"{len(de[de['q_fdr']<0.10])} at FDR<0.10")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE — 7.09 × 9.0 in (Nature double-column page-fit)
# ═════════════════════════════════════════════════════════════════════════════
# Landscape canvas (18×10 in) — 6 panels in a 2×3 grid.
fig = plt.figure(figsize=(18.0, 10.0))
gs = gridspec.GridSpec(
    2, 3, figure=fig,
    left=0.040, right=0.992, top=0.960, bottom=0.060,
    wspace=0.30, hspace=0.32,
)

PANEL_LBL = dict(
    fontsize=24, fontweight="bold", color="#000",
    ha="left", va="bottom", family="sans-serif",
)


# ── (a) IT1 score by 5-group diagnosis (violin + strip) ─────────────────────
ax = fig.add_subplot(gs[0, 0])
groups = [scores.loc[scores["diagnosis"] == g, "IT1_score"].values for g in DX_ORDER]
parts = ax.violinplot(groups, positions=np.arange(len(DX_ORDER)),
                      widths=0.78, showmeans=False, showmedians=True,
                      showextrema=False)
for i, body in enumerate(parts["bodies"]):
    body.set_facecolor(DX_COL[DX_ORDER[i]])
    body.set_edgecolor(DX_COL[DX_ORDER[i]])
    body.set_alpha(0.55)
    body.set_linewidth(0.6)
parts["cmedians"].set_color("#222"); parts["cmedians"].set_linewidth(0.8)

rng = np.random.default_rng(0)
for i, g in enumerate(DX_ORDER):
    yv = scores.loc[scores["diagnosis"] == g, "IT1_score"].values
    xs = i + rng.uniform(-0.13, 0.13, size=len(yv))
    ax.scatter(xs, yv, s=5, c=DX_COL[g], alpha=0.85,
               edgecolor="white", linewidth=0.25, zorder=3)
ax.axhline(0, color="#bbb", lw=0.4, ls="--", zorder=0)

# significance bars: AS vs each
y_top = scores["IT1_score"].max()
y_bot = scores["IT1_score"].min()
y_pad = (y_top - y_bot) * 0.08
ax.set_ylim(y_bot - 0.12, y_top + y_pad * 5.5)
as_i = DX_ORDER.index("AS")
for j, g in enumerate(["NT", "IMNM", "IBM", "DM"]):
    gi = DX_ORDER.index(g)
    row = stats[stats["test"] == f"MW AS vs {g}"]
    if row.empty: continue
    p = float(row.iloc[0]["p"])
    y_bar = y_top + y_pad * (j + 1)
    ax.plot([gi, as_i], [y_bar, y_bar], color="#444", lw=0.5)
    ax.text((gi + as_i) / 2, y_bar + y_pad * 0.10, sig_stars(p),
            ha="center", va="bottom", fontsize=10.5)

ax.set_xticks(range(len(DX_ORDER)))
ax.set_xticklabels([f"{g}\nn={int((scores['diagnosis']==g).sum())}"
                    for g in DX_ORDER], fontsize=10.0)
ax.set_ylabel("IT1 innate-hyperactive score (z)", fontsize=11.0)
kw_row = stats[stats["test"] == "KW 5-group"].iloc[0]
ax.set_title(f"GSE220915 muscle-biopsy RNA-seq (n=165)\n"
             f"Kruskal–Wallis H = {kw_row['stat']:.1f}, "
             f"{fmt_p(float(kw_row['p']))}",
             fontsize=12.0, pad=3)
ax.text(-0.18, 1.05, "a", transform=ax.transAxes, **PANEL_LBL)


# ── (b) Cliff's δ forest — AS vs each comparator ────────────────────────────
ax = fig.add_subplot(gs[0, 1])
comp_order = ["NT", "IMNM", "IBM", "DM"]
y_pos = np.arange(len(comp_order))
for i, g in enumerate(comp_order):
    row = stats[stats["test"] == f"MW AS vs {g}"].iloc[0]
    d = float(row["effect"]); p = float(row["p"])
    col = "#C0392B" if d > 0 else "#2980B9"
    ax.barh(i, d, height=0.55, color=col, alpha=0.85, edgecolor="white", lw=0.5)
    # annotation: effect on bar end, p-value to the right
    # δ on bar end
    ax.text(d + (0.020 if d > 0 else -0.020), i - 0.20,
            f"δ={d:+.2f}",
            va="center", ha="left" if d > 0 else "right",
            fontsize=9.5, color="#222", fontweight="bold")
    # significance + p-value below
    ax.text(d + (0.020 if d > 0 else -0.020), i + 0.22,
            f"{sig_stars(p)}  {fmt_p(p)}",
            va="center", ha="left" if d > 0 else "right",
            fontsize=9.0, color="#444")
ax.axvline(0, color="#222", lw=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"AS  vs  {g}" for g in comp_order], fontsize=10.5)
ax.invert_yaxis()
ax.set_xlim(-0.50, 1.20)
ax.set_xlabel("Cliff's δ  (positive = IT1 score higher in AS)", fontsize=11.0)
ax.set_title("Pairwise effect size: AS vs each diagnosis",
             fontsize=12.0, pad=3)
ax.grid(axis="x", color="#eee", lw=0.3); ax.set_axisbelow(True)
ax.text(-0.22, 1.05, "b", transform=ax.transAxes, **PANEL_LBL)


# ── (c) Within-AS heterogeneity (18 patients ranked) ────────────────────────
ax = fig.add_subplot(gs[0, 2])
as_df  = scores.loc[scores["diagnosis"] == "AS"].sort_values("IT1_score")
nt_df  = scores.loc[scores["diagnosis"] == "NT"]
imnm_df = scores.loc[scores["diagnosis"] == "IMNM"]
nt_lo, nt_hi = np.percentile(nt_df["IT1_score"], [5, 95])
im_lo, im_hi = np.percentile(imnm_df["IT1_score"], [5, 95])

ax.axhspan(nt_lo, nt_hi, color=DX_COL["NT"], alpha=0.18, lw=0, zorder=0,
           label=f"NT 5–95% ile (n={len(nt_df)})")
ax.axhspan(im_lo, im_hi, color=DX_COL["IMNM"], alpha=0.18, lw=0, zorder=0,
           label=f"IMNM 5–95% ile (n={len(imnm_df)})")
xs = np.arange(1, len(as_df) + 1)
ax.plot(xs, as_df["IT1_score"].values, color=DX_COL["AS"],
        lw=0.6, alpha=0.55, zorder=2)
ax.scatter(xs, as_df["IT1_score"].values, s=22, c=DX_COL["AS"],
           edgecolor="white", linewidth=0.4, zorder=3)
# median split annotation
median_it1 = as_df["IT1_score"].median()
n_lo = (as_df["IT1_score"] <= median_it1).sum()
n_hi = (as_df["IT1_score"] >  median_it1).sum()
ax.axvline(n_lo + 0.5, color="#444", ls=":", lw=0.6, zorder=1)
ax.text(n_lo + 0.5 - 0.3, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else as_df["IT1_score"].max(),
        f"IT1-low\n(n={n_lo})", ha="right", va="top", fontsize=9.5, color="#555")
ax.text(n_lo + 0.5 + 0.3, as_df["IT1_score"].max(),
        f"IT1-high\n(n={n_hi})", ha="left", va="top", fontsize=9.5, color="#555")
ax.axhline(0, color="#bbb", lw=0.4, ls="--", zorder=0)
ax.set_xlabel("18 AS patients (ranked by IT1 score)", fontsize=11.0)
ax.set_ylabel("IT1 innate-hyperactive score (z)", fontsize=11.0)
ax.set_title("Within-AS IT1 heterogeneity:\n"
             "AS spans NT-quiescent → IMNM-intermediate → inflammatory range",
             fontsize=12.0, pad=3)
ax.legend(fontsize=9.0, loc="lower right", frameon=False, handletextpad=0.4,
          borderpad=0.2)
ax.text(-0.18, 1.05, "c", transform=ax.transAxes, **PANEL_LBL)


# ── (d) IT1-high vs IT1-low volcano within-AS ───────────────────────────────
ax = fig.add_subplot(gs[1, 0])
de_plot = de.dropna(subset=["log2fc", "q_fdr"]).copy()
de_plot["neg_log10_q"] = -np.log10(de_plot["q_fdr"].clip(lower=1e-30))
# direction-aware coloring
de_plot["pathway"] = de_plot["symbol"].astype(str).map(gene_pathway)
sig_mask = de_plot["q_fdr"] < 0.10
ns = de_plot[~sig_mask]
sig = de_plot[sig_mask]

ax.scatter(ns["log2fc"], ns["neg_log10_q"],
           s=2, c="#cccccc", alpha=0.45, edgecolor="none", zorder=1)
# sig points: gray base, then highlight pathway members
sig_other = sig[sig["pathway"] == "Other"]
ax.scatter(sig_other["log2fc"], sig_other["neg_log10_q"],
           s=4, c="#888", alpha=0.55, edgecolor="none", zorder=2)
for pname in ["IFN-γ","Complement","Cytolytic","Myeloid","Antibody","T cell"]:
    sub = sig[sig["pathway"] == pname]
    ax.scatter(sub["log2fc"], sub["neg_log10_q"],
               s=10, c=PATHWAY_COL[pname], alpha=0.9,
               edgecolor="white", linewidth=0.25, zorder=3,
               label=f"{pname} (n={len(sub)})")

# Label top 6 IT1-high-up genes as a side legend (no in-plot overlap).
top_hits = sig[sig["log2fc"] > 0].nlargest(6, "t").reset_index(drop=True)
y_max_data = sig["neg_log10_q"].max()
# Highlight the top 6 hits with larger edge markers
ax.scatter(top_hits["log2fc"], top_hits["neg_log10_q"],
           s=22, facecolors="none", edgecolors="#000",
           linewidths=0.7, zorder=4)
# Numerical key by each highlighted point (1, 2, 3, 4, 5, 6)
for i, r in top_hits.iterrows():
    ax.text(r["log2fc"], r["neg_log10_q"], str(i + 1),
            fontsize=9.0, color="#000", fontweight="bold",
            ha="center", va="center", zorder=5,
            bbox=dict(boxstyle="circle,pad=0.06",
                      fc="white", ec="#000", lw=0.4))
# Compact legend in upper-right reading order: '1: CD48 ...'
gene_legend = "\n".join(
    f"{i+1}. {r['symbol']}  (t={r['t']:.1f})"
    for i, r in top_hits.iterrows()
)
ax.text(0.985, 0.98, "Top 6 IT1-high-up:\n" + gene_legend,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8.5, color="#222",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.32",
                  fc="white", ec="#888", lw=0.4, alpha=0.94))
ax.axhline(-np.log10(0.10), color="#888", ls="--", lw=0.4)
ax.axvline(0, color="#888", lw=0.4)
ax.set_xlabel("log$_2$ fold-change  (IT1-high vs IT1-low, within AS)", fontsize=11.0)
ax.set_ylabel("−log$_{10}$ q (BH-FDR)", fontsize=11.0)
n_sig = int(sig_mask.sum())
ax.set_title(f"Within-AS IT1-stratified DE\n"
             f"{n_sig:,} genes at FDR<0.10  "
             f"(n=18 AS patients, median split)",
             fontsize=12.0, pad=3)
ax.legend(fontsize=9.0, loc="upper left", frameon=False, ncol=1,
          handletextpad=0.3, borderpad=0.2)
ax.text(-0.18, 1.05, "d", transform=ax.transAxes, **PANEL_LBL)


# ── (e) Pathway enrichment in IT1-high AS ───────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
pw_short = {
    "IFN-γ signalling (IRDS core)":            "IFN-γ",
    "Complement (Mammen 2023 primary finding)":"Complement",
    "Cytolytic effector":                       "Cytolytic",
    "Monocyte / myeloid":                       "Myeloid",
    "Antibody / plasma cell":                   "Antibody",
    "T cell activation":                        "T cell",
}
pw["short"] = pw["pathway"].map(pw_short).fillna(pw["pathway"])
pw["color"] = pw["short"].map(PATHWAY_COL).fillna("#888")
pw_sorted = pw.sort_values("hi_p")
y = np.arange(len(pw_sorted))[::-1]
neglog_q = -np.log10(pw_sorted["hi_qBH"].clip(lower=1e-20))
ax.barh(y, neglog_q.values, color=pw_sorted["color"].values,
        alpha=0.85, edgecolor="white", lw=0.5, height=0.65)
for i, (_, r) in enumerate(pw_sorted.iterrows()):
    yi = y[i] if isinstance(y, np.ndarray) else i
    ax.text(neglog_q.iloc[i] + 0.4, yi,
            f"{r['hi_hits']}/{r['K_in_bg']}  (q={r['hi_qBH']:.1e})",
            fontsize=9.0, va="center", color="#333")
ax.axvline(-np.log10(0.05), color="#444", ls="--", lw=0.5,
           label="q = 0.05")
ax.axvline(-np.log10(0.001), color="#444", ls=":", lw=0.5,
           label="q = 0.001")
ax.set_yticks(y)
ax.set_yticklabels(pw_sorted["short"].values, fontsize=10.5, fontweight="bold")
# color-code the y-tick labels to match
for tick, c in zip(ax.get_yticklabels(), pw_sorted["color"].values):
    tick.set_color(c)
ax.set_xlabel(r"$-\log_{10}$  q (BH, hypergeometric)", fontsize=11.0)
ax.set_xlim(0, max(neglog_q.max() * 1.30, -np.log10(0.001) * 1.05))
ax.set_title("Pathway enrichment in IT1-high AS\n"
             "(recapitulates the primary finding of Mammen/Pinal-Fernández 2023)",
             fontsize=12.0, pad=3)
ax.legend(fontsize=9.0, loc="lower right", frameon=False, handletextpad=0.4)
ax.grid(axis="x", color="#eee", lw=0.3); ax.set_axisbelow(True)
ax.text(-0.18, 1.05, "e", transform=ax.transAxes, **PANEL_LBL)


# ── (f) Top 30 DE genes ranked by t-statistic, pathway-coloured ─────────────
ax = fig.add_subplot(gs[1, 2])
de_sym = de.dropna(subset=["symbol", "t", "log2fc"]).copy()
de_sym = de_sym[de_sym["symbol"].astype(str) != ""]
de_sym["pathway"] = de_sym["symbol"].astype(str).map(gene_pathway)
# select top per pathway then fill remainder with global |t| top
top_per_path = []
for p_name in ["IFN-γ", "Complement", "Cytolytic", "Myeloid", "Antibody", "T cell"]:
    sub = de_sym[(de_sym["pathway"] == p_name) & (de_sym["q_fdr"] < 0.10)]
    sub = sub.sort_values("t", ascending=False).head(5)
    top_per_path.append(sub)
top_df = pd.concat(top_per_path).drop_duplicates("symbol")
top_df = top_df.sort_values("t", ascending=False)

n_show = len(top_df)
yv = np.arange(n_show)[::-1]
colors = top_df["pathway"].map(PATHWAY_COL).fillna("#888").values
ax.barh(yv, top_df["t"].values, color=colors, alpha=0.88,
        edgecolor="white", lw=0.4, height=0.72)
ax.axvline(0, color="#222", lw=0.5)
ax.set_yticks(yv)
ax.set_yticklabels(top_df["symbol"].values, fontsize=9.0)
for tick, c in zip(ax.get_yticklabels(), colors):
    tick.set_color(c)
    tick.set_fontweight("bold")
ax.set_xlabel("t-statistic  (IT1-high vs IT1-low, within AS)", fontsize=11.0)
ax.set_title("Top pathway-annotated genes\n"
             "(top 5 per programme, ranked by |t|)",
             fontsize=12.0, pad=3)
# pathway legend
legend_handles = [Patch(facecolor=PATHWAY_COL[p], edgecolor="none", label=p)
                  for p in ["IFN-γ","Complement","Cytolytic","Myeloid","Antibody","T cell"]]
ax.legend(handles=legend_handles, fontsize=9.0, loc="lower right",
          frameon=False, ncol=2, handlelength=0.8, handletextpad=0.3,
          columnspacing=0.6, borderpad=0.2)
ax.grid(axis="x", color="#eee", lw=0.3); ax.set_axisbelow(True)
ax.text(-0.28, 1.05, "f", transform=ax.transAxes, **PANEL_LBL)


# ─────────────────────────────────────────────────────────────────────────────
# Figure title removed (journal convention).

# Save (overwrites the broken composite-of-composites)
out_pdf = f"{OUT_DIR}/MainFig6_external_validation.pdf"
out_png = f"{OUT_DIR}/MainFig6_external_validation.png"
fig.savefig(out_pdf, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
fig.savefig(out_png, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
