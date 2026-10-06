"""Revision R1 copy of cytof_external_validation.py used to redraw Supp. Fig. S8 with the
ASyS label; outputs go to revision_R1/figs_R1/S8.  Run from the repository root."""
"""
cytof_external_validation.py
============================
External validation of the IT1 "innate-hyperactive" immunotype signature on
the independent GSE220915 muscle-biopsy RNA-seq cohort
  (Mammen / Pinal-Fernandez 2023, n=165 muscle biopsies;
   ASS=18, DM=44, IMNM=54, IBM=16, normal muscle=33).

Pipeline:
  (1) Derive a gene-level IT1 signature from the CyTOF IT1 definition
      (innate myeloid + cytotoxic + activation programs).
  (2) For contrast: a lymphoid-quiescent IT3 signature.
  (3) Score each GSE220915 sample on both signatures using GSVA-style
      z-scored mean expression.
  (4) Test:
        - IT1 score differs across 5 diagnosis groups (Kruskal-Wallis)
        - IT1 score differs ASyS vs NT (normal muscle)
        - IT1 score differs ASyS vs IMNM (the low-IFN-γ myositis subtype)
        - IT1 and IT3 should anti-correlate
  (5) Plot a single publication-ready panel.

Run:
    cd /Users/yichen/Desktop/collaboration/raw
    /Users/yichen/anaconda3/bin/conda run -n cytof python cytof_external_validation.py
"""

import os, json, gzip, warnings, urllib.request
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

from nature_style import apply_nature_style, HEATMAP_DIVERG, save
apply_nature_style()
SC, DC = 3.46, 7.09

VAL_DIR = "./validation"
OUT_DIR = "./revision_R1/figs_R1/S8"
os.makedirs(OUT_DIR, exist_ok=True)

COUNTS  = f"{VAL_DIR}/counts.tsv.gz"
SERIES  = f"{VAL_DIR}/series.txt.gz"

# ════════════════════════════════════════════════════════════════════════════
# IT1  =  innate-hyperactive immunotype signature
# Derived from the CyTOF IT1 definition (↑ monocyte, ↑ NK, ↑ mDC/pDC, ↑ HLA-DR,
#    ↑ CD38, ↑ CD69, ↑ GranzymeB) translated to cognate transcripts.
# ════════════════════════════════════════════════════════════════════════════
IT1_GENES = {
    # myeloid / monocyte
    "CD14", "ITGAM", "ITGAX", "FCGR3A", "FCGR1A", "CD68", "CD86",
    "S100A8", "S100A9", "S100A12", "VCAN", "LYZ",
    # inflammatory cytokines / chemokines
    "IL1B", "TNF", "CCL2", "CXCL10", "CXCL9",
    # dendritic cells (mDC + pDC)
    "CD1C", "CLEC4C", "IRF8", "IRF7",
    # NK / cytotoxic
    "NKG7", "GNLY", "GZMB", "GZMA", "GZMK", "PRF1", "KLRD1",
    "KLRK1", "NCAM1", "FCGR3B",
    # activation
    "HLA-DRA", "HLA-DRB1", "CD38", "CD69", "CD274",  # CD274 = PD-L1
    # interferon signalling (matches Mammen 2023 story)
    "IFNG", "STAT1", "ISG15", "MX1", "OAS1",
}

# IT3 = lymphoid-quiescent signature (naive T, B cell, regulatory)
IT3_GENES = {
    "CCR7", "SELL", "LEF1", "TCF7", "CD27", "IL7R",
    "FOXP3", "CTLA4", "KLRG1",
    "MS4A1", "CD19", "CD79A", "IGHM", "IGHD",  # naive B
    "CD4",   # CD4 baseline
}

# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — parse series matrix → diagnosis per sample
# ════════════════════════════════════════════════════════════════════════════
print("[1/6] Parsing series matrix for diagnosis labels …")
with gzip.open(SERIES, "rt") as f:
    lines = f.readlines()

titles   = []
diagnoses = []
for ln in lines:
    if ln.startswith("!Sample_title"):
        titles = [x.strip().strip('"') for x in ln.split("\t")[1:]]
    elif ln.startswith("!Sample_characteristics_ch1") and "diagnosis:" in ln:
        diagnoses = [x.strip().strip('"').replace("diagnosis: ", "")
                     for x in ln.split("\t")[1:]]

# sample_X_suffix → sample_X (match counts file column names)
sample_ids = [t.rsplit("_", 1)[0] for t in titles]
# quick diagnosis cleanup
def short(d):
    d = d.strip().upper()
    if "IMNM" in d:           return "IMNM"
    if "IBM"  in d:           return "IBM"
    if "AS" in d: return "ASyS"
    if "DM"   in d:           return "DM"
    if "NORMAL" in d:         return "NT"
    return d[:6]
dx = [short(d) for d in diagnoses]
meta = pd.DataFrame({"sample": sample_ids, "diagnosis": dx})
print("   sample counts per diagnosis:")
print(meta["diagnosis"].value_counts().to_string())

# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — query mygene.info for symbol → Ensembl ID mapping
# ════════════════════════════════════════════════════════════════════════════
print("\n[2/6] Resolving gene symbols → Ensembl IDs …")
all_symbols = sorted(IT1_GENES | IT3_GENES)

def query_mygene(symbols):
    url  = "https://mygene.info/v3/query"
    data = f"q={','.join(symbols)}&scopes=symbol&fields=ensembl.gene,symbol&species=human"
    req  = urllib.request.Request(url, data=data.encode(), method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=30).read().decode())

resp = []  # R1: online lookup unused (local GENCODE map below)
sym2ens = {}
for r in resp:
    if r.get("notfound"): continue
    ens = r.get("ensembl")
    if ens is None: continue
    if isinstance(ens, list):
        # prefer any ENSG that is a protein-coding principal gene — take the first
        ens = ens[0]
    gene_id = ens.get("gene")
    sym     = r.get("symbol", r.get("query"))
    if gene_id:
        sym2ens[sym] = gene_id
print(f"   resolved {len(sym2ens)}/{len(all_symbols)} symbols")

missing = sorted(set(all_symbols) - set(sym2ens))
if missing:
    print("   WARNING missing:", missing)

# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — load counts, strip version suffix, subset to signature genes
# ════════════════════════════════════════════════════════════════════════════
print("\n[3/6] Loading counts matrix …")
cnt = pd.read_csv(COUNTS, sep="\t", compression="gzip", index_col=0)
# strip .version suffix
cnt.index = cnt.index.str.split(".").str[0]
print(f"   counts shape: {cnt.shape}  ({cnt.columns[:3].tolist()}…)")

# verify column order matches meta order
meta = meta.set_index("sample").loc[cnt.columns]
print(f"   metadata aligned: {meta.shape}")

# Collapse counts ENSG → symbol via local mapping (matches cytof_mainfig9_multicohort.py;
# this is the biologically correct lookup that catches all 41 IT1 symbols — mygene.info
# alone misses TNF/IRF7/HLA-DRB1 because it returns patch/alt ENSGs not in this cohort)
ens_map_local = pd.read_csv(f"{VAL_DIR}/ensg2sym.csv")
_ec = "ensg" if "ensg" in ens_map_local.columns else ens_map_local.columns[0]
_sc = "symbol" if "symbol" in ens_map_local.columns else ens_map_local.columns[1]
_ensg_to_sym_local = dict(zip(ens_map_local[_ec], ens_map_local[_sc]))
_sym_idx = cnt.index.map(_ensg_to_sym_local)
_keep = pd.notna(_sym_idx)
cnt_sym = cnt.loc[_keep].copy()
cnt_sym.index = _sym_idx[_keep]
cnt_sym = cnt_sym.groupby(level=0).sum()    # collapse duplicate ENSGs to symbol level
print(f"   counts collapsed by symbol: {cnt_sym.shape}")

# normalisation: CPM then log2  (both ENSG- and symbol-indexed)
lib  = cnt_sym.sum(0)
cpm  = cnt_sym.div(lib, axis=1) * 1e6
lcpm = np.log2(cpm + 1.0)                  # now SYMBOL-indexed, not ENSG-indexed
print(f"   median library size: {lib.median():,.0f}")

# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — score IT1 and IT3 signatures
# ════════════════════════════════════════════════════════════════════════════
print("\n[4/6] Scoring signatures per sample …")

def z_score(x):
    mu, sd = np.nanmean(x), np.nanstd(x)
    return (x - mu) / sd if sd > 0 else x - mu

def score_signature(lcpm, sym_set, sym2ens=None):
    """Mean z-score across detected signature genes.

    lcpm is now SYMBOL-indexed (post-collapse), so we look up symbols directly.
    sym2ens kept for backwards compat but unused.
    """
    detected = sorted({s for s in sym_set if s in lcpm.index})
    gene_block = lcpm.loc[detected]
    z = gene_block.apply(z_score, axis=1)       # (g × n) z-scored per gene
    return z.mean(axis=0), detected

it1_score, it1_hits = score_signature(lcpm, IT1_GENES)
it3_score, it3_hits = score_signature(lcpm, IT3_GENES)
print(f"   IT1: {len(it1_hits)}/{len(IT1_GENES)} genes detected")
print(f"   IT3: {len(it3_hits)}/{len(IT3_GENES)} genes detected")

meta["IT1_score"] = it1_score
meta["IT3_score"] = it3_score
meta.to_csv(f"{OUT_DIR}/S5_validation_scores.csv")

# ════════════════════════════════════════════════════════════════════════════
# STEP 5 — statistical tests
# ════════════════════════════════════════════════════════════════════════════
print("\n[5/6] Statistical tests …")

DX_ORDER = ["NT", "IMNM", "IBM", "DM", "ASyS"]
groups = [meta.loc[meta["diagnosis"]==g, "IT1_score"].values for g in DX_ORDER]

# 5-group KW
kw_h, kw_p = stats.kruskal(*groups)
# primary comparisons vs AS
mw_res = {}
for g in DX_ORDER:
    if g == "ASyS": continue
    u, p = stats.mannwhitneyu(
        meta.loc[meta["diagnosis"]=="ASyS",  "IT1_score"],
        meta.loc[meta["diagnosis"]==g,     "IT1_score"],
        alternative="two-sided",
    )
    # Cliff's delta
    xA = meta.loc[meta["diagnosis"]=="ASyS", "IT1_score"].values
    xG = meta.loc[meta["diagnosis"]==g,    "IT1_score"].values
    d_cliff = np.mean([np.sign(a - b) for a in xA for b in xG])
    mw_res[g] = {"u": u, "p": p, "cliff_d": d_cliff,
                  "median_AS": np.median(xA), "median_G": np.median(xG)}

# IT1 vs IT3 correlation
rho_it1_it3, p_it1_it3 = stats.spearmanr(meta["IT1_score"], meta["IT3_score"])

print(f"   Kruskal-Wallis (5-group):  H={kw_h:.2f}  p={kw_p:.2e}")
for g, r in mw_res.items():
    print(f"   MW  ASyS vs {g:<5}  p={r['p']:.2e}  Cliff's δ={r['cliff_d']:+.2f}  "
          f"(median AS={r['median_AS']:+.2f}  {g}={r['median_G']:+.2f})")
print(f"   IT1 × IT3 Spearman:  ρ={rho_it1_it3:+.3f}  p={p_it1_it3:.2e}")

# ════════════════════════════════════════════════════════════════════════════
# STEP 6 — plot
# ════════════════════════════════════════════════════════════════════════════
print("\n[6/6] Plotting external-validation panel …")

DX_COL = {
    "NT":   "#95A5A6",
    "IMNM": "#2980B9",
    "IBM":  "#8E44AD",
    "DM":   "#F39C12",
    "ASyS":   "#C0392B",
}

# Override the global "large-print" Nature style for this multi-panel
# diagnostic so lines/ticks/fonts render at small-multiples-appropriate
# weight. No further figures are drawn after S4 in this script, so we set
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

fig = plt.figure(figsize=(DC+1.2, 8.8))
gs  = gridspec.GridSpec(3, 3, figure=fig, wspace=0.42, hspace=0.62,
                        height_ratios=[1.0, 1.0, 1.75],
                        width_ratios=[1.0, 1.0, 1.1])

def _panel_label(ax, lbl, x=-0.20, y=1.12):
    ax.text(x, y, lbl, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")

# (a) IT1 score by diagnosis — violin + strip
ax = fig.add_subplot(gs[0, 0])
_panel_label(ax, "a", x=-0.28)
parts = ax.violinplot([meta.loc[meta["diagnosis"]==g, "IT1_score"].values
                        for g in DX_ORDER],
                       positions=np.arange(len(DX_ORDER)),
                       widths=0.82, showmeans=False, showmedians=True)
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(DX_COL[DX_ORDER[i]]); pc.set_alpha(0.55)
    pc.set_edgecolor(DX_COL[DX_ORDER[i]]); pc.set_linewidth(0.7)
for k in ["cmedians","cbars","cmins","cmaxes"]:
    if k in parts: parts[k].set_color("#333"); parts[k].set_lw(0.5)
# strip
rng = np.random.default_rng(0)
for i, g in enumerate(DX_ORDER):
    y = meta.loc[meta["diagnosis"]==g, "IT1_score"].values
    x = i + rng.uniform(-0.14, 0.14, size=len(y))
    ax.scatter(x, y, s=7, c=DX_COL[g], alpha=0.8,
               edgecolor="white", linewidth=0.3, zorder=3)
ax.set_xticks(range(len(DX_ORDER)))
ax.set_xticklabels([f"{g}\nn={int((meta['diagnosis']==g).sum())}"
                    for g in DX_ORDER], fontsize=6)
ax.set_ylabel("IT1 innate-hyperactive score\n(mean z-score across sig. genes)", fontsize=6.5)
ax.axhline(0, color="#bbb", lw=0.4, ls="--")
ax.set_title(f"IT1 signature in GSE220915 muscle biopsies\n"
             f"Kruskal-Wallis: H={kw_h:.1f}, p={kw_p:.1e}", fontsize=7, pad=4)
# significance bars vs AS
def pstr(p):
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 0.05: return "*"
    return "ns"
y_top = meta["IT1_score"].max()
as_i = DX_ORDER.index("ASyS")
bar_step = 0.55          # vertical gap between successive sig bars
bar_base = y_top + 0.25  # lowest bar
ax.set_ylim(meta["IT1_score"].min() - 0.2,
            bar_base + bar_step * 4 + 0.55)   # headroom for top star (R1: more, clears the title)
for g in ["NT","IMNM","IBM","DM"]:
    gi = DX_ORDER.index(g)
    y_bar = bar_base + bar_step * abs(gi - as_i)
    ax.plot([gi, as_i], [y_bar, y_bar], color="#555", lw=0.6)
    ax.text((gi+as_i)/2, y_bar + 0.04, pstr(mw_res[g]["p"]),
            ha="center", va="bottom", fontsize=8)

# (b) IT1 vs IT3 scatter — reframed as "tissue-level concordance"
ax = fig.add_subplot(gs[0, 1])
_panel_label(ax, "b", x=-0.20)
for g in DX_ORDER:
    sub = meta[meta["diagnosis"] == g]
    ax.scatter(sub["IT1_score"], sub["IT3_score"], s=22,
               c=DX_COL[g], alpha=0.8, edgecolor="white",
               linewidth=0.4, label=g, zorder=3)
m, b = np.polyfit(meta["IT1_score"], meta["IT3_score"], 1)
xs = np.linspace(meta["IT1_score"].min(), meta["IT1_score"].max(), 100)
ax.plot(xs, m*xs + b, color="#333", lw=0.8, alpha=0.6, zorder=2)
ax.set_xlabel("IT1 score (innate program)", fontsize=6.5)
ax.set_ylabel("IT3 score (lymphoid program)", fontsize=6.5)
ax.set_title(f"Tissue-level immune infiltrate concordance\n"
             f"Spearman ρ = {rho_it1_it3:+.2f}  "
             f"(muscle: both ↑ together)",
             fontsize=7, pad=4)
ax.axhline(0, color="#ddd", lw=0.3); ax.axvline(0, color="#ddd", lw=0.3)
ax.legend(fontsize=5, frameon=False, loc="upper left",
          markerscale=0.9, ncol=2, handletextpad=0.2, borderpad=0.2)

# (c) Effect-size (Cliff's d) forest — ASyS vs each other group
ax = fig.add_subplot(gs[0, 2])
_panel_label(ax, "c", x=-0.30)
comp_order = ["NT","IMNM","IBM","DM"]
y = np.arange(len(comp_order))
for i, g in enumerate(comp_order):
    d = mw_res[g]["cliff_d"]
    p = mw_res[g]["p"]
    col = "#C0392B" if d > 0 else "#2980B9"
    ax.barh(i, d, color=col, alpha=0.85, edgecolor="white")
    ax.text(d + (0.01 if d > 0 else -0.01), i,
            f"  {pstr(p)}  p={p:.1e}",
            va="center", ha="left" if d > 0 else "right",
            fontsize=5.5, color="#333")
ax.axvline(0, color="#333", lw=0.5)
ax.set_yticks(y); ax.set_yticklabels([f"ASyS vs {g}" for g in comp_order], fontsize=6)
ax.set_xlabel("Cliff's δ  (+ = IT1 higher in ASyS)", fontsize=6.5)
ax.set_title("Effect size of IT1-score elevation\nin ASyS vs each group",
             fontsize=7, pad=4)
ax.set_xlim(-1.05, 1.05)
ax.invert_yaxis()
ax.grid(axis="x", color="#eee", lw=0.3); ax.set_axisbelow(True)

# (d) WITHIN-AS heterogeneity of IT1 score — validates our core hypothesis
ax = fig.add_subplot(gs[1, 0])
_panel_label(ax, "d", x=-0.22)
as_scores = meta.loc[meta["diagnosis"]=="ASyS", "IT1_score"].sort_values()
# reference distributions
nt_scores  = meta.loc[meta["diagnosis"]=="NT",   "IT1_score"].values
imnm_scores = meta.loc[meta["diagnosis"]=="IMNM","IT1_score"].values
# AS fan plot
ax.axhspan(np.percentile(nt_scores, 5), np.percentile(nt_scores, 95),
           color=DX_COL["NT"], alpha=0.18, lw=0, zorder=0,
           label="NT 5–95 %ile")
ax.axhspan(np.percentile(imnm_scores, 5), np.percentile(imnm_scores, 95),
           color=DX_COL["IMNM"], alpha=0.18, lw=0, zorder=0,
           label="IMNM 5–95 %ile")
xs = np.arange(1, len(as_scores)+1)
ax.scatter(xs, as_scores.values, s=28, c=DX_COL["ASyS"],
           edgecolor="white", linewidth=0.5, zorder=3)
ax.plot(xs, as_scores.values, color=DX_COL["ASyS"], lw=0.6, alpha=0.6, zorder=2)
# annotate range
as_range = float(as_scores.max() - as_scores.min())
ax.text(0.05, 0.97,
        f"ASyS within-cohort range = {as_range:.2f}\n"
        f"IQR / std = {as_scores.std():.2f}",
        transform=ax.transAxes, va="top", fontsize=6, color=DX_COL["ASyS"])
ax.axhline(0, color="#bbb", lw=0.4, ls="--")
ax.set_xlabel("ASyS patients (ranked by IT1 score)", fontsize=6.5)
ax.set_ylabel("IT1 score", fontsize=6.5)
ax.set_title("Within-ASyS IT1 heterogeneity\n"
             "(18 ASyS patients span NT→IMNM→high-inflam. range)",
             fontsize=7, pad=4)
ax.legend(fontsize=5, frameon=False, loc="lower right",
          handletextpad=0.3, borderpad=0.2)

# (e) Test: does within-ASyS IT1 variance exceed what random sampling gives?
ax = fig.add_subplot(gs[1, 1])
_panel_label(ax, "e", x=-0.20)
# permutation: shuffle 18 labels 5000 times across all 165 samples, record variance
rng = np.random.default_rng(1)
obs_var = float(as_scores.var(ddof=1))
perm_vars = []
all_scores = meta["IT1_score"].values
for _ in range(5000):
    perm = rng.choice(all_scores, size=18, replace=False)
    perm_vars.append(float(perm.var(ddof=1)))
perm_vars = np.array(perm_vars)
p_var = (perm_vars >= obs_var).mean()
ax.hist(perm_vars, bins=40, color="#95A5A6", alpha=0.7,
        edgecolor="white", linewidth=0.3)
ax.axvline(obs_var, color=DX_COL["ASyS"], lw=1.8)
ax.text(obs_var, ax.get_ylim()[1]*0.92,
        f" observed\n ASyS variance\n p = {p_var:.3f}",
        color=DX_COL["ASyS"], fontsize=6, va="top", fontweight="bold")
ax.set_xlabel("Variance of IT1 score (random 18-sample draw)", fontsize=6.5)
ax.set_ylabel("count", fontsize=6.5)
ax.set_title("Permutation test: is ASyS more heterogeneous\n"
             "than a random 18-sample subset? (5,000 perms)",
             fontsize=7, pad=4)

# (f) Signature robustness — leave-one-gene-out of IT1
ax = fig.add_subplot(gs[1, 2])
_panel_label(ax, "f", x=-0.22)
as_idx = meta["diagnosis"] == "ASyS"
nt_idx = meta["diagnosis"] == "NT"
logo_deltas = []
logo_labels = []
# lcpm is now SYMBOL-indexed (post symbol-collapse), so look up by symbol directly
it1_sym_all = [s for s in IT1_GENES if s in lcpm.index]
for drop_sym in it1_sym_all:
    kept = [g for g in it1_sym_all if g != drop_sym]
    block = lcpm.loc[kept].apply(z_score, axis=1)
    sc_loo = block.mean(axis=0)
    d = sc_loo[as_idx].median() - sc_loo[nt_idx].median()
    logo_deltas.append(d); logo_labels.append(drop_sym)
logo_df = pd.DataFrame({"gene_dropped": logo_labels,
                         "AS_vs_NT_median_diff": logo_deltas}
                       ).sort_values("AS_vs_NT_median_diff")
# reference: full signature delta
full_d = float(meta.loc[as_idx,"IT1_score"].median() - meta.loc[nt_idx,"IT1_score"].median())

ax.scatter(logo_df["AS_vs_NT_median_diff"].values,
           np.arange(len(logo_df)), s=14, c=DX_COL["ASyS"],
           alpha=0.8, edgecolor="white", linewidth=0.3)
ax.axvline(full_d, color="#333", lw=0.8, ls="--",
           label=f"full sig. Δ = {full_d:.2f}")
ax.set_yticks(np.arange(len(logo_df))[::3])
ax.set_yticklabels(logo_df["gene_dropped"].values[::3], fontsize=5.2)
ax.set_xlabel("ASyS − NT median IT1 score\n(leave-one-gene-out)", fontsize=6.5)
ax.set_title("Signature robustness to gene removal\n"
             "(no single gene drives the ASyS>NT gap)",
             fontsize=7, pad=4)
ax.legend(fontsize=5, frameon=False, loc="lower right")

# (g) Heatmap of top IT1 genes × samples (sorted by IT1 score within diagnosis)
ax = fig.add_subplot(gs[2, :])
_panel_label(ax, "g", x=-0.05, y=1.18)
# pick top 25 IT1 genes with the largest between-group SD
# lcpm is SYMBOL-indexed (post symbol-collapse), so look up by symbol directly
it1_syms = sorted(s for s in IT1_GENES if s in lcpm.index)
it1_expr = lcpm.loc[it1_syms].apply(z_score, axis=1)
# pick top 25 by SD of log2 CPM (R1: before z-scoring, after which every gene has SD 1
# and the choice was arbitrary)
top_genes = lcpm.loc[it1_syms].std(axis=1).sort_values(ascending=False).head(25).index
it1_expr = it1_expr.loc[top_genes]

# order samples: by diagnosis (in DX_ORDER), then by IT1_score
ordered_samples = []
for g in DX_ORDER:
    sub = meta[meta["diagnosis"]==g].sort_values("IT1_score")
    ordered_samples.extend(sub.index.tolist())
hm = it1_expr[ordered_samples]

_nr_s4, _nc_s4 = hm.shape
im = ax.pcolormesh(np.arange(_nc_s4 + 1) - 0.5, np.arange(_nr_s4 + 1) - 0.5,
                    hm.values, cmap=HEATMAP_DIVERG, vmin=-2.0, vmax=2.0,
                    edgecolors="none", rasterized=False, shading="flat")
ax.invert_yaxis(); ax.set_aspect("auto")
ax.set_yticks(range(hm.shape[0]))
ax.set_yticklabels(hm.index, fontsize=5.5)
ax.set_xticks([]); ax.tick_params(length=0)

# diagnosis colour strip at top
ycoord = -0.04
for g in DX_ORDER:
    idxs = [i for i, s in enumerate(ordered_samples)
            if meta.loc[s, "diagnosis"] == g]
    if not idxs: continue
    start, end = min(idxs), max(idxs)
    ax.add_patch(plt.Rectangle((start-0.5, -3.0), end-start+1, 1.8,
                                color=DX_COL[g], lw=0, clip_on=False))
    ax.text((start+end)/2, -4.2, f"{g} (n={end-start+1})", ha="center",
            va="bottom", fontsize=6, color=DX_COL[g],
            fontweight="bold", clip_on=False)

ax.set_ylim(hm.shape[0]-0.5, -5.0)    # give headroom for diagnosis strip
ax.set_title("Top 25 IT1 signature genes across GSE220915 samples\n"
             "(z-score; samples sorted by IT1 score within each diagnosis)",
             fontsize=7, pad=14)
cax = fig.add_axes([0.92, 0.08, 0.010, 0.20])
cb = fig.colorbar(im, cax=cax); cb.set_label("gene z-score", fontsize=6)
cb.ax.tick_params(labelsize=5)

# Figure title removed (journal convention).

save(fig, "S5_external_validation_GSE220915", OUT_DIR)
print("   Saved: S5_external_validation_GSE220915.pdf/.png")

# final summary
result = pd.DataFrame([
    {"test": "KW 5-group", "stat": kw_h, "p": kw_p, "effect": np.nan},
] + [
    {"test": f"MW ASyS vs {g}",
     "stat": mw_res[g]["u"], "p": mw_res[g]["p"],
     "effect": mw_res[g]["cliff_d"]}
    for g in comp_order
] + [
    {"test": "IT1 × IT3 Spearman", "stat": rho_it1_it3,
     "p": p_it1_it3, "effect": rho_it1_it3},
])
result.to_csv(f"{OUT_DIR}/S5_validation_stats.csv", index=False)

print(f"""
══════════════════════════════════════════════════════════
  External validation complete.
    S5_external_validation_GSE220915.pdf/.png
    S5_validation_scores.csv    (per-sample IT1 / IT3 scores)
    S5_validation_stats.csv     (all test results)
══════════════════════════════════════════════════════════
""")
