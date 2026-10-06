"""
cytof_within_AS.py
==================
B — Within-antisynthetase-syndrome sub-analysis on GSE220915.

  (i)   Stratify the 18 AS samples by IT1 score (median split →
        IT1-high AS vs IT1-low AS).
  (ii)  Unsupervised clustering of the 18 AS transcriptomes on the
        full variable-gene space to check whether sub-structure
        exists *within* the AS arm independently of IT1.
  (iii) Differential expression IT1-high vs IT1-low AS (Welch-t +
        BH-FDR on log2-CPM, n≈9 vs n≈9) — realistic effect sizes,
        not cell-level pseudo-replication.
  (iv)  GO / pathway enrichment of the top DE genes using a minimal
        hypergeometric test against the Reactome "innate immune",
        "interferon signalling" and "cytolysis" gene sets.
  (v)   A combined 4-panel figure.

Run:
    cd /Users/yichen/Desktop/collaboration/raw
    /Users/yichen/anaconda3/bin/conda run -n cytof python cytof_within_AS.py
"""

import os, gzip, warnings, json, urllib.request
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from nature_style import apply_nature_style, HEATMAP_DIVERG, save
apply_nature_style()
SC, DC = 3.46, 7.09

VAL_DIR = "./validation"
OUT_DIR = "./cytof_output_nature"
COUNTS  = f"{VAL_DIR}/counts.tsv.gz"
SERIES  = f"{VAL_DIR}/series.txt.gz"

print("[1/5] Reading counts + series matrix …")
with gzip.open(SERIES, "rt") as f:
    lines = f.readlines()
titles, diagnoses = [], []
for ln in lines:
    if ln.startswith("!Sample_title"):
        titles = [x.strip().strip('"') for x in ln.split("\t")[1:]]
    elif ln.startswith("!Sample_characteristics_ch1") and "diagnosis:" in ln:
        diagnoses = [x.strip().strip('"').replace("diagnosis: ", "")
                     for x in ln.split("\t")[1:]]
sample_ids = [t.rsplit("_", 1)[0] for t in titles]
def short(d):
    d = d.strip().upper()
    if "IMNM" in d: return "IMNM"
    if "IBM"  in d: return "IBM"
    if "AS"   in d: return "AS"
    if "DM"   in d: return "DM"
    if "NORMAL" in d: return "NT"
    return d[:6]
dx = [short(d) for d in diagnoses]
meta = pd.DataFrame({"sample": sample_ids, "diagnosis": dx}).set_index("sample")

cnt = pd.read_csv(COUNTS, sep="\t", compression="gzip", index_col=0)
cnt.index = cnt.index.str.split(".").str[0]
meta = meta.loc[cnt.columns]

lib  = cnt.sum(0)
cpm  = cnt.div(lib, axis=1) * 1e6
lcpm = np.log2(cpm + 1.0)

# reload IT1 scores from previous step
scores = pd.read_csv(f"{OUT_DIR}/S4_validation_scores.csv", index_col=0)
meta   = meta.join(scores[["IT1_score","IT3_score"]])

AS_samples = meta.index[meta["diagnosis"] == "AS"].tolist()
print(f"  AS samples: {len(AS_samples)}")

# ════════════════════════════════════════════════════════════════════════════
# (i) Median split by IT1
# ════════════════════════════════════════════════════════════════════════════
print("\n[2/5] IT1 median split within AS …")
as_meta = meta.loc[AS_samples].sort_values("IT1_score")
median_it1 = as_meta["IT1_score"].median()
as_meta["IT1_strat"] = np.where(as_meta["IT1_score"] > median_it1,
                                "AS-IT1high", "AS-IT1low")
print(as_meta[["IT1_score","IT3_score","IT1_strat"]].to_string())

hi = as_meta.index[as_meta["IT1_strat"] == "AS-IT1high"].tolist()
lo = as_meta.index[as_meta["IT1_strat"] == "AS-IT1low" ].tolist()
print(f"  IT1-high AS n={len(hi)}  /  IT1-low AS n={len(lo)}")

# ════════════════════════════════════════════════════════════════════════════
# (ii) Unsupervised clustering of the 18 AS transcriptomes
# ════════════════════════════════════════════════════════════════════════════
print("\n[3/5] Unsupervised clustering of 18 AS transcriptomes …")
as_lcpm = lcpm[AS_samples]
# keep only genes with some expression
keep = (as_lcpm > 1.0).sum(axis=1) >= 6
as_lcpm = as_lcpm.loc[keep]
# top 3000 most variable genes
top_var = as_lcpm.var(axis=1).sort_values(ascending=False).head(3000).index
M = as_lcpm.loc[top_var]
# standardise per gene
Mz = M.sub(M.mean(axis=1), axis=0).div(M.std(axis=1, ddof=0), axis=0).fillna(0)

# PCA
pcs = PCA(n_components=5, random_state=0).fit_transform(Mz.T)
as_pca = pd.DataFrame(pcs, index=M.columns,
                      columns=[f"PC{i+1}" for i in range(5)])
as_pca = as_pca.join(as_meta[["IT1_strat","IT1_score"]])

# Ward clustering into 2 groups
link = hierarchy.linkage(pcs, method="ward")
clust2 = hierarchy.fcluster(link, t=2, criterion="maxclust")
as_pca["cluster_k2"] = [f"C{c}" for c in clust2]

# concordance: IT1 strat vs unsupervised cluster
tab = pd.crosstab(as_pca["IT1_strat"], as_pca["cluster_k2"])
print("  Concordance table (IT1-strat × unsupervised cluster):")
print(tab.to_string())
# simple ARI-like agreement rate
best_match = max((tab.values[0,0] + tab.values[1,1]),
                 (tab.values[0,1] + tab.values[1,0]))
concord = best_match / tab.values.sum()
print(f"  Best-match label concordance: {concord:.2f}")

# ════════════════════════════════════════════════════════════════════════════
# (iii) DE: IT1-high AS vs IT1-low AS
# ════════════════════════════════════════════════════════════════════════════
print("\n[4/5] DE IT1-high AS vs IT1-low AS (Welch-t, BH) …")
# restrict to expressed genes
expressed = (lcpm[AS_samples] > 1.0).sum(axis=1) >= 6
X_as = lcpm.loc[expressed, AS_samples]
print(f"  {X_as.shape[0]:,} expressed genes to test")

hi_vals = X_as[hi].values
lo_vals = X_as[lo].values
t_stat, p_vals = stats.ttest_ind(hi_vals, lo_vals, axis=1, equal_var=False)
log2fc = X_as[hi].mean(axis=1).values - X_as[lo].mean(axis=1).values
_, q_vals, _, _ = multipletests(p_vals, method="fdr_bh")

de = pd.DataFrame({
    "gene_id":  X_as.index,
    "t":        t_stat,
    "log2fc":   log2fc,
    "p":        p_vals,
    "q_fdr":    q_vals,
})
print(f"  Significant at FDR<0.10: {(de['q_fdr']<0.10).sum()}")
print(f"  Significant at FDR<0.05: {(de['q_fdr']<0.05).sum()}")
print(f"  Significant at FDR<0.01: {(de['q_fdr']<0.01).sum()}")

# map to symbols via local GENCODE v46 annotation (built beforehand)
ens2sym_df = pd.read_csv(f"{VAL_DIR}/ensg2sym.csv", index_col="ensg")
ens2sym    = ens2sym_df["symbol"].to_dict()
de["symbol"] = de["gene_id"].map(ens2sym).fillna("")
de.to_csv(f"{OUT_DIR}/S6_AS_IT1high_vs_IT1low_DE.csv", index=False)
print(f"    Mapped {sum(de['symbol']!='')}/{len(de)} gene symbols from local GENCODE v46")

# ════════════════════════════════════════════════════════════════════════════
# (iv) Minimal hypergeometric enrichment for a few curated gene sets
# ════════════════════════════════════════════════════════════════════════════
print("\n[5/5] Hypergeometric enrichment of curated pathways …")

# manually curated, conservative pathway sets (HGNC symbols)
PATHWAYS = {
    "IFN-γ signalling (IRDS core)": {
        "STAT1","IRF1","IRF7","IRF8","IRF9","GBP1","GBP5","CXCL9","CXCL10","CXCL11",
        "HLA-DRA","HLA-DRB1","HLA-DPA1","HLA-DPB1","CD74","PSMB8","PSMB9",
        "TAP1","TAP2","MX1","ISG15","OAS1","OAS2","OAS3","IFI44","IFIT1","IFIT3",
        "IFITM1","IFITM3",
    },
    "Complement (Mammen 2023 primary finding)": {
        "C1QA","C1QB","C1QC","C1R","C1S","C2","C3","C4A","C4B","C5","CFB",
        "C3AR1","C5AR1","ITGAM","ITGAX","CR1","VSIG4",
    },
    "Cytolytic effector": {
        "GZMA","GZMB","GZMK","GZMH","GZMM","PRF1","GNLY","NKG7","KLRD1","KLRK1",
        "FGFBP2","FASLG",
    },
    "Monocyte / myeloid": {
        "CD14","CD68","CD163","VCAN","S100A8","S100A9","S100A12","FCGR1A","FCGR3A",
        "CSF1R","LYZ","MS4A6A","MS4A7","CLEC10A",
    },
    "Antibody / plasma cell": {
        "IGHG1","IGHG3","IGHG4","IGHA1","IGHA2","JCHAIN","MZB1","XBP1","PRDM1","CD38",
    },
    "T cell activation": {
        "CD3D","CD3E","CD3G","CD4","CD8A","CD8B","LCK","ZAP70","IL2RA","CD69","ICOS",
    },
}

# background = detected genes with symbols
bg_symbols = set(de.loc[de["symbol"] != "", "symbol"].values)
print(f"  Background (detected w/ symbol): {len(bg_symbols)}")

def hyper_enrich(hit_symbols, bg_symbols, pathway):
    pw = pathway & bg_symbols
    if len(pw) == 0: return np.nan, 0, 0
    K = len(pw)                         # in pathway & in background
    n = len(hit_symbols)                # DEGs in foreground
    N = len(bg_symbols)                 # total background
    k = len(hit_symbols & pw)           # overlap
    if k == 0: return 1.0, k, K
    # sf = 1 - cdf(k-1) = P(X >= k)
    p = stats.hypergeom.sf(k-1, N, K, n)
    return p, k, K

# define foreground: up in IT1-high (log2fc>0, q<0.20)  |  up in IT1-low (log2fc<0, q<0.20)
up_hi  = set(de.loc[(de["log2fc"] >  0.5) & (de["q_fdr"] < 0.20) & (de["symbol"] != ""),
                    "symbol"].tolist())
up_lo  = set(de.loc[(de["log2fc"] < -0.5) & (de["q_fdr"] < 0.20) & (de["symbol"] != ""),
                    "symbol"].tolist())
print(f"  up in IT1-high: {len(up_hi)} genes  |  up in IT1-low: {len(up_lo)} genes")

enrich_rows = []
for pname, pset in PATHWAYS.items():
    p_hi, k_hi, K_hi = hyper_enrich(up_hi, bg_symbols, pset)
    p_lo, k_lo, K_lo = hyper_enrich(up_lo, bg_symbols, pset)
    enrich_rows.append({
        "pathway": pname, "K_in_bg": K_hi,
        "hi_hits": k_hi, "hi_p": p_hi,
        "lo_hits": k_lo, "lo_p": p_lo,
    })
enrich = pd.DataFrame(enrich_rows)
enrich["hi_qBH"] = multipletests(enrich["hi_p"].fillna(1), method="fdr_bh")[1]
enrich["lo_qBH"] = multipletests(enrich["lo_p"].fillna(1), method="fdr_bh")[1]
enrich.to_csv(f"{OUT_DIR}/S6_AS_pathway_enrichment.csv", index=False)
print(enrich.to_string(index=False))

# ════════════════════════════════════════════════════════════════════════════
# PLOT
# ════════════════════════════════════════════════════════════════════════════
print("\n[Plotting] 4-panel within-AS figure …")
# Override the global "large-print" Nature style for this multi-panel
# diagnostic so lines/ticks/fonts render at small-multiples-appropriate
# weight. No further figures are drawn after S5 in this script, so we set
# rcParams in-place rather than wrapping the long plotting block.
plt.rcParams.update({
    "font.size":          7,
    "axes.titlesize":     7.5,
    "axes.labelsize":     7,
    "xtick.labelsize":    6,
    "ytick.labelsize":    6,
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.major.size":   2.5,
    "ytick.major.size":   2.5,
    "lines.linewidth":    1.0,
    "patch.linewidth":    0.4,
    "legend.fontsize":    5.5,
})
fig = plt.figure(figsize=(DC+1.2, 6.4))
gs  = gridspec.GridSpec(2, 2, figure=fig, wspace=0.40, hspace=0.52,
                        height_ratios=[1.0, 1.2])

STRAT_COL = {"AS-IT1high": "#C0392B", "AS-IT1low": "#2980B9"}
CLU_COL   = {"C1": "#8E44AD", "C2": "#F39C12"}

def _panel_label(ax, lbl, x=-0.20, y=1.08):
    ax.text(x, y, lbl, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")

# (a) AS PCA by IT1 stratum + cluster concordance
ax = fig.add_subplot(gs[0, 0])
_panel_label(ax, "a", x=-0.18)
for s, row in as_pca.iterrows():
    ax.scatter(row["PC1"], row["PC2"], s=60,
               c=STRAT_COL[row["IT1_strat"]],
               edgecolor=CLU_COL[row["cluster_k2"]],
               linewidth=1.6, zorder=3, alpha=0.9)
    ax.annotate(s.split("_")[-1], xy=(row["PC1"], row["PC2"]),
                xytext=(7, 5), textcoords="offset points",
                fontsize=4.8, color="#555")
ax.set_xlabel("PC1 ({:.0f}% var)".format(
    PCA(n_components=5, random_state=0).fit(Mz.T).explained_variance_ratio_[0]*100),
    fontsize=6.5)
ax.set_ylabel("PC2", fontsize=6.5)
ax.set_title(f"18 AS transcriptomes in PCA space\n"
             f"(fill = IT1 strat;  edge = Ward k=2 cluster;  concord. = {concord:.2f})",
             fontsize=7, pad=4)
# legends
from matplotlib.lines import Line2D
leg1 = [Line2D([0],[0], marker="o", color="w",
              markerfacecolor=c, markersize=7, markeredgecolor="#333",
              label=k) for k, c in STRAT_COL.items()]
leg2 = [Line2D([0],[0], marker="o", color="w",
              markerfacecolor="#eee", markersize=7,
              markeredgecolor=c, markeredgewidth=1.8, label=k)
       for k, c in CLU_COL.items()]
l1 = ax.legend(handles=leg1, loc="upper right", fontsize=5.2,
               frameon=False, handletextpad=0.3,
               title="IT1 stratum", title_fontsize=5.5)
ax.add_artist(l1)
ax.legend(handles=leg2, loc="lower right", fontsize=5.2, frameon=False,
          handletextpad=0.3, title="Ward k=2", title_fontsize=5.5)

# (b) volcano of DE
ax = fig.add_subplot(gs[0, 1])
_panel_label(ax, "b", x=-0.18)
neg_log_q = -np.log10(de["q_fdr"].clip(lower=1e-12))
col = np.where((de["q_fdr"] < 0.10) & (de["log2fc"] >  0.5), "#C0392B",
      np.where((de["q_fdr"] < 0.10) & (de["log2fc"] < -0.5), "#2980B9", "#bbbbbb"))
ax.scatter(de["log2fc"].values, neg_log_q.values,
           s=5, c=col, alpha=0.55, edgecolor="none")
ax.axhline(-np.log10(0.10), color="#333", ls="--", lw=0.5)
ax.axvline(+0.5, color="#333", ls="--", lw=0.5)
ax.axvline(-0.5, color="#333", ls="--", lw=0.5)
# headroom above the data so labels can stack out of the dense tip
neg_log_q_max = float(neg_log_q.max())
ax.set_ylim(-0.05, neg_log_q_max + 1.4)
# label top hits, repulsion to avoid overlap
from adjustText import adjust_text
texts = []
top_lab = de.loc[de["symbol"] != ""].nlargest(8, columns="t")
for _, r in top_lab.iterrows():
    texts.append(ax.text(
        r["log2fc"], -np.log10(max(r["q_fdr"], 1e-12)),
        r["symbol"], fontsize=5, color="#C0392B"))
top_lab_lo = de.loc[de["symbol"] != ""].nsmallest(4, columns="t")
for _, r in top_lab_lo.iterrows():
    texts.append(ax.text(
        r["log2fc"], -np.log10(max(r["q_fdr"], 1e-12)),
        r["symbol"], fontsize=5, color="#2980B9"))
adjust_text(texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color="#888", lw=0.3),
            expand=(2.0, 2.4),
            force_text=(0.8, 1.4),
            force_static=(0.4, 0.6))
ax.set_xlabel(r"log$_2$ FC (IT1-high AS  −  IT1-low AS)", fontsize=6.5)
ax.set_ylabel(r"−log$_{10}$ q (BH)", fontsize=6.5)
ax.set_title(f"IT1-high vs IT1-low AS — differential expression\n"
             f"(n={len(hi)} vs n={len(lo)};  "
             f"{int((de['q_fdr']<0.10).sum())} genes at FDR<0.10)",
             fontsize=7, pad=4)

# (c) pathway enrichment double bar
ax = fig.add_subplot(gs[1, 0])
_panel_label(ax, "c", x=-0.40)
y = np.arange(len(enrich))
bar_h = 0.4
ax.barh(y - bar_h/2, -np.log10(enrich["hi_p"].clip(lower=1e-20)),
        height=bar_h, color=STRAT_COL["AS-IT1high"], alpha=0.8, label="up in IT1-high")
ax.barh(y + bar_h/2, -np.log10(enrich["lo_p"].clip(lower=1e-20)),
        height=bar_h, color=STRAT_COL["AS-IT1low"], alpha=0.8, label="up in IT1-low")
ax.set_yticks(y); ax.set_yticklabels(enrich["pathway"].values, fontsize=6)
ax.axvline(-np.log10(0.05), color="#333", ls="--", lw=0.5)
ax.set_xlabel(r"−log$_{10}$ hypergeometric p", fontsize=6.5)
ax.set_title("Pathway enrichment of top DEGs (curated sets,\nbackground = detected AS-expressed genes)",
             fontsize=7, pad=4)
ax.legend(fontsize=5, frameon=False, loc="lower right",
          handletextpad=0.3)
for i, r in enrich.iterrows():
    ax.text(-np.log10(max(r["hi_p"],1e-20))+0.1, i - bar_h/2,
            f"{r['hi_hits']}/{r['K_in_bg']}", fontsize=5, va="center",
            color=STRAT_COL["AS-IT1high"])
    ax.text(-np.log10(max(r["lo_p"],1e-20))+0.1, i + bar_h/2,
            f"{r['lo_hits']}/{r['K_in_bg']}", fontsize=5, va="center",
            color=STRAT_COL["AS-IT1low"])
ax.invert_yaxis()

# (d) heatmap of top DE genes × AS samples
ax = fig.add_subplot(gs[1, 1])
_panel_label(ax, "d", x=-0.18)
top_n = 25
# absolute t-value ordering
de_sym = de.loc[de["symbol"] != ""].copy()
de_sym["abs_t"] = de_sym["t"].abs()
top_genes = de_sym.sort_values("abs_t", ascending=False).head(top_n)
hm_ensg = top_genes["gene_id"].tolist()
hm = lcpm.loc[hm_ensg, AS_samples]
# row-z
hm = hm.sub(hm.mean(axis=1), axis=0).div(hm.std(axis=1, ddof=0), axis=0).fillna(0)
# order columns by IT1 score
col_order = as_meta.sort_values("IT1_score").index.tolist()
hm = hm[col_order]
# rename rows to symbol
sym = [top_genes.set_index("gene_id").loc[e, "symbol"] for e in hm.index]
hm.index = sym

_nr, _nc = hm.shape
im = ax.pcolormesh(np.arange(_nc + 1) - 0.5, np.arange(_nr + 1) - 0.5,
                    hm.values, cmap=HEATMAP_DIVERG, vmin=-2, vmax=2,
                    edgecolors="none", rasterized=False, shading="flat")
ax.invert_yaxis(); ax.set_aspect("auto")
ax.set_yticks(range(hm.shape[0]))
ax.set_yticklabels(hm.index, fontsize=5.5)
ax.set_xticks(range(hm.shape[1]))
ax.set_xticklabels([s.split("_")[-1] for s in hm.columns],
                    fontsize=5, rotation=90)
ax.tick_params(length=0)
# IT1 strat colour strip
strip_h = 0.9
strat_colors = [STRAT_COL[as_meta.loc[s,"IT1_strat"]] for s in col_order]
for i, c in enumerate(strat_colors):
    ax.add_patch(plt.Rectangle((i-0.5, -1.6), 1, strip_h,
                                color=c, lw=0, clip_on=False))
ax.set_ylim(hm.shape[0]-0.5, -2.5)
ax.set_title(f"Top {top_n} DE genes × 18 AS samples\n(z-score; sorted by IT1 score)",
             fontsize=7, pad=4)
cax = fig.add_axes([0.94, 0.08, 0.010, 0.16])
cb  = fig.colorbar(im, cax=cax); cb.set_label("gene z-score", fontsize=6)
cb.ax.tick_params(labelsize=5)

# Figure title removed (journal convention).
save(fig, "S6_within_AS_substructure", OUT_DIR)
print("   Saved: S6_within_AS_substructure.pdf/.png")

# save all DE results
de.sort_values("q_fdr").to_csv(f"{OUT_DIR}/S6_AS_IT1high_vs_IT1low_DE.csv", index=False)

print(f"""
══════════════════════════════════════════════════════════
  Within-AS analysis complete.
    S6_within_AS_substructure.pdf/.png
    S6_AS_IT1high_vs_IT1low_DE.csv
    S6_AS_pathway_enrichment.csv
══════════════════════════════════════════════════════════
""")
