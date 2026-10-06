"""
Revision R1 copy of cytof_mainfig9_multicohort.py (manuscript Fig. 7), harmonised
with extended_validation/ (Results 3.12) so that both sections report identical
per-cohort and pooled effects:
  - GSE220915: CPM over all Ensembl genes, highest-mean Ensembl ID per symbol;
  - GSE128470: GEO GPL96.annot.gz ('Gene symbol'), single-gene probes only,
    highest-mean probe per gene (validation_extra/GPL96.annot.gz, downloaded from
    https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL96/annot/);
  - panel f excludes genes invariant in either cohort (FCGR1A in GSE128470).
Outputs go to revision_R1/figs_R1.  Run from the repository root.
"""
"""
Main Figure 9 — Multi-cohort meta-analysis of the IT1 innate-hyperactive
signature.

Cohorts:
  1. GSE220915 (Mammen / Pinal-Fernandez 2023): bulk RNA-seq, n = 165
     muscle biopsies (AS / DM / IMNM / IBM / NT).
  2. GSE128470 (Greenberg 2019):  microarray (Affymetrix HG-U133A,
     GPL96), n = 77 muscle biopsies (DM / IBM / NM / NS / PM / NL).

The 41-gene IT1 signature, derived from CyTOF IT1 patients only, is
projected onto each cohort. We compute per-cohort effect sizes and
combine across cohorts (random-effects meta-analysis on Cohen's d).

The 6 panels:

  (a) Per-cohort cohort-summary box: n, diagnostic groups, modality.
  (b) GSE220915 violin (re-rendered, 5 groups).
  (c) GSE128470 violin (6 groups).
  (d) Cohort × diagnostic-group effect-size forest (Cohen's d for
      'inflammatory myopathy vs normal' in each cohort).
  (e) Random-effects meta-analysis: pooled Cohen's d with 95 % CI
      across the 2 cohorts (and within-cohort I² heterogeneity).
  (f) Cross-cohort signature-gene heatmap: top 25 IT1 genes' z-score
      direction concordance.

Output: MainFig9_multicohort.pdf/png.
"""
import warnings
warnings.filterwarnings("ignore")
import os, gzip
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu

OUT_DIR  = "./revision_R1/figs_R1"
COHORT1  = "validation/counts.tsv.gz"          # GSE220915
COHORT1_S = "validation/series.txt.gz"
COHORT1_E = "validation/ensg2sym.csv"
COHORT2  = "validation_extra/GSE128470_series.txt.gz"
GPL96    = "validation_extra/GPL96.tsv"

# IT1 signature (innate myeloid + cytotoxic + interferon)
IT1_GENES = {
    "CD14","ITGAM","ITGAX","FCGR3A","FCGR1A","CD68","CD86",
    "S100A8","S100A9","S100A12","VCAN","LYZ",
    "IL1B","TNF","CCL2","CXCL10","CXCL9",
    "CD1C","CLEC4C","IRF8","IRF7",
    "NKG7","GNLY","GZMB","GZMA","GZMK","PRF1","KLRD1","KLRK1","NCAM1","FCGR3B",
    "HLA-DRA","HLA-DRB1","CD38","CD69","CD274",
    "IFNG","STAT1","ISG15","MX1","OAS1",
}

plt.rcParams.update({
    "font.family":"sans-serif",
    "font.sans-serif":["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":10.5, "axes.titlesize":12.0, "axes.labelsize":10.5,
    "xtick.labelsize":9.5, "ytick.labelsize":9.5, "legend.fontsize":9.5,
    "axes.linewidth":0.9, "xtick.major.width":0.7, "ytick.major.width":0.7,
    "axes.spines.top":False, "axes.spines.right":False,
    "pdf.fonttype":42, "ps.fonttype":42,
    "savefig.dpi":300, "figure.dpi":300,
})
PANEL_LBL = dict(fontsize=22, fontweight="bold", color="#000",
                 ha="left", va="bottom", family="sans-serif")

# ─── Cohort 1: GSE220915 ────────────────────────────────────────────────────
print("[1/3] Loading GSE220915 (cohort 1) …")
# parse series for diagnosis
with gzip.open(COHORT1_S, "rt") as f:
    lines = f.readlines()
titles = []; diagnoses = []
for ln in lines:
    if ln.startswith("!Sample_title"):
        titles = [x.strip().strip('"') for x in ln.split("\t")[1:]]
    elif ln.startswith("!Sample_characteristics_ch1") and "diagnosis:" in ln:
        diagnoses = [x.strip().strip('"').replace("diagnosis: ", "")
                     for x in ln.split("\t")[1:]]
sample_ids_c1 = [t.rsplit("_", 1)[0] for t in titles]
def short_dx(d):
    d = d.strip().upper()
    if "IMNM"   in d: return "IMNM"
    if "IBM"    in d: return "IBM"
    if "AS"     in d: return "ASyS"
    if "DM"     in d: return "DM"
    if "NORMAL" in d: return "NT"
    return d[:6]
dx_c1 = [short_dx(d) for d in diagnoses]
meta_c1 = pd.DataFrame({"sample": sample_ids_c1, "diagnosis": dx_c1})
print(f"  {len(meta_c1)} samples; {meta_c1['diagnosis'].value_counts().to_dict()}")

# Load counts (ENSG-indexed) → gene symbol
counts_c1 = pd.read_csv(COHORT1, sep="\t", index_col=0)
counts_c1.columns = [c.split("_")[0] + "_" + c.split("_")[1]
                     if c.startswith("sample_") else c
                     for c in counts_c1.columns]
ens_map = pd.read_csv(COHORT1_E)
# Build ENSG (no version) → symbol mapping
ensg_col = "ensg" if "ensg" in ens_map.columns else \
           "ensembl_gene_id" if "ensembl_gene_id" in ens_map.columns else \
           ens_map.columns[0]
sym_col  = "symbol" if "symbol" in ens_map.columns else ens_map.columns[1]
ensg_to_sym = dict(zip(ens_map[ensg_col], ens_map[sym_col]))

# Strip version suffix from counts row IDs (e.g. ENSG00000000003.15 → ENSG00000000003)
counts_c1.index = counts_c1.index.str.split(".").str[0]

# Map to symbols
# R1: same normalisation as extended_validation/load_cohorts.py -- CPM over all
# Ensembl genes, then the highest-mean Ensembl ID kept for each symbol.
cpm_all = counts_c1.div(counts_c1.sum(axis=0), axis=1) * 1e6
log_all = np.log2(cpm_all + 1.0)
sym_idx = pd.Series(log_all.index.map(ensg_to_sym), index=log_all.index)
log_all = log_all.loc[sym_idx.notna().values]
log_all = log_all.assign(_s=sym_idx.dropna().values,
                         _m=log_all.mean(axis=1)).sort_values("_m", ascending=False)
log_cpm = (log_all.drop_duplicates("_s").set_index("_s").drop(columns="_m"))
log_cpm.index.name = None

# IT1 score = mean z-score of signature genes (symbol-collapsed counts)
it1_genes_present = [g for g in IT1_GENES if g in log_cpm.index]
print(f"  IT1 genes detected in C1: {len(it1_genes_present)}/{len(IT1_GENES)}")
sig_mat = log_cpm.loc[it1_genes_present]
sig_z = (sig_mat.sub(sig_mat.mean(axis=1), axis=0)
                .div(sig_mat.std(axis=1), axis=0))
it1_c1 = sig_z.mean(axis=0)

# Map sample columns to diagnoses (strip _suffix from columns to align with meta)
sample_to_dx_c1 = dict(zip(meta_c1["sample"], meta_c1["diagnosis"]))
# Try short-id matching: split by _ to find matching base
common_ids = []
for col in it1_c1.index:
    base = col.split("_")[0] + "_" + col.split("_")[1] if "_" in col else col
    if base in sample_to_dx_c1:
        common_ids.append(base)
# Direct mapping
df_c1 = pd.DataFrame({"sample": it1_c1.index})
df_c1["short"] = df_c1["sample"].apply(
    lambda c: c.split("_")[0] + "_" + c.split("_")[1] if "_" in c else c
)
df_c1["diagnosis"] = df_c1["short"].map(sample_to_dx_c1)
df_c1["IT1"] = it1_c1.values
df_c1 = df_c1.dropna(subset=["diagnosis"])
print(f"  C1 samples scored: {len(df_c1)}; "
      f"diagnoses: {df_c1['diagnosis'].value_counts().to_dict()}")

# ─── Cohort 2: GSE128470 ────────────────────────────────────────────────────
print("[2/3] Loading GSE128470 (cohort 2) …")
# parse series matrix metadata
with gzip.open(COHORT2, "rt") as f:
    lines = f.readlines()
titles_c2 = []; dx_c2_raw = []
for ln in lines:
    if ln.startswith("!Sample_title"):
        titles_c2 = [x.strip().strip('"') for x in ln.split("\t")[1:]]
    elif ln.startswith("!Sample_characteristics_ch1") and "disease state" in ln:
        dx_c2_raw = [x.strip().strip('"').replace("disease state: ", "")
                     for x in ln.split("\t")[1:]]

def short_dx_c2(d):
    d = d.strip().lower()
    if "dermat" in d:                return "DM"
    if "inclusion" in d:             return "IBM"
    if "necrotizing" in d:           return "IMNM"  # = NM in this cohort
    if "polymyositis" in d:          return "PM"
    if "nonspecific" in d:           return "NS"
    if "normal" in d:                return "NT"
    return d[:5]

dx_c2 = [short_dx_c2(d) for d in dx_c2_raw]
print(f"  {len(dx_c2)} samples; {pd.Series(dx_c2).value_counts().to_dict()}")

# extract data section
data_start = next(i for i, ln in enumerate(lines) if "!series_matrix_table_begin" in ln)
data_end   = next(i for i, ln in enumerate(lines) if "!series_matrix_table_end" in ln)
data_lines = lines[data_start + 1:data_end]
expr_c2 = pd.read_csv(
    pd.io.common.StringIO("".join(data_lines)),
    sep="\t", index_col=0, na_values=["null", ""],
)
# Strip quotes from column headers
expr_c2.columns = [c.strip('"') for c in expr_c2.columns]
print(f"  Probe matrix: {expr_c2.shape}")

# Map probes → gene symbols via GPL96
def parse_gpl96(gpl_path):
    """Extract probe ID → gene symbol mapping from GPL96 SOFT format."""
    with open(gpl_path) as f:
        in_table = False
        rows = []
        header = None
        for ln in f:
            if "!platform_table_begin" in ln:
                in_table = True
                continue
            if "!platform_table_end" in ln:
                break
            if in_table:
                if header is None:
                    header = ln.strip().split("\t")
                    continue
                parts = ln.strip().split("\t")
                if len(parts) >= len(header):
                    rows.append(parts[:len(header)])
        df = pd.DataFrame(rows, columns=header)
    sym_col = "Gene Symbol" if "Gene Symbol" in df.columns else \
              [c for c in df.columns if "symbol" in c.lower()][0]
    return dict(zip(df["ID"], df[sym_col]))

print("  Building GPL96 probe → symbol map …")
# R1: same annotation and rule as extended_validation/load_cohorts.py
# (GEO GPL96.annot.gz, 'Gene symbol' column, single-gene probes only).
probe_to_sym = {}
with gzip.open("validation_extra/GPL96.annot.gz", "rt", errors="replace") as fh:
    started = False
    for L in fh:
        if L.startswith("!platform_table_begin"):
            started = True; next(fh); continue
        if not started or L.startswith("!platform_table_end"):
            continue
        p_ = L.rstrip("\n").split("\t")
        if len(p_) > 2 and p_[2] and "///" not in p_[2]:
            probe_to_sym[p_[0]] = p_[2]
# Map each probe.  R1: harmonised with extended_validation/load_cohorts.py --
# multi-gene probes ('A /// B') are dropped and, where several probes map to one
# symbol, the probe with the highest mean expression is kept (previously all
# probes were averaged and multi-gene probes assigned to their first symbol).
expr_c2.index = expr_c2.index.map(probe_to_sym)
expr_c2 = expr_c2.loc[~expr_c2.index.isna()].copy()
expr_c2 = expr_c2.loc[[("///" not in str(s)) and str(s).strip() != "" for s in expr_c2.index]]
expr_c2 = expr_c2.assign(_m=expr_c2.mean(axis=1)).sort_values("_m", ascending=False)
expr_c2 = expr_c2[~expr_c2.index.duplicated(keep="first")].drop(columns="_m")
print(f"  Gene-level matrix: {expr_c2.shape}")

# IT1 score on cohort 2 (already log-transformed by GEO)
it1_c2_genes = [g for g in IT1_GENES if g in expr_c2.index]
print(f"  IT1 genes detected in C2: {len(it1_c2_genes)}/{len(IT1_GENES)}")
sig2 = expr_c2.loc[it1_c2_genes]
sig2_z = (sig2.sub(sig2.mean(axis=1), axis=0)
              .div(sig2.std(axis=1) + 1e-9, axis=0))
it1_c2 = sig2_z.mean(axis=0)

df_c2 = pd.DataFrame({"sample": it1_c2.index, "diagnosis": dx_c2,
                       "IT1": it1_c2.values})
print(f"  C2 samples scored: {len(df_c2)}; "
      f"diagnoses: {df_c2['diagnosis'].value_counts().to_dict()}")

# ─── meta-analysis ────────────────────────────────────────────────────────
def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    s_pool = np.sqrt(((len(a) - 1) * a.var(ddof=1) +
                      (len(b) - 1) * b.var(ddof=1)) /
                     (len(a) + len(b) - 2))
    if s_pool == 0:
        return 0.0
    return (a.mean() - b.mean()) / s_pool

def d_se(a, b, d):
    """Standard error of Cohen's d (Hedges 1981)."""
    n1, n2 = len(a), len(b)
    return np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))

# For each cohort, compute "disease vs control" Cohen's d
records = []
# Cohort 1: ASyS vs NT, IMNM vs NT, etc.
for grp in ["ASyS", "DM", "IMNM", "IBM"]:
    a = df_c1.loc[df_c1["diagnosis"] == grp, "IT1"].values
    b = df_c1.loc[df_c1["diagnosis"] == "NT", "IT1"].values
    if len(a) >= 3 and len(b) >= 3:
        d = cohens_d(a, b)
        se = d_se(a, b, d)
        records.append({"cohort": "GSE220915", "group": grp,
                         "d": d, "se": se, "n_a": len(a), "n_b": len(b)})
# Cohort 2: DM vs NL, IBM vs NL, IMNM vs NL, PM vs NL, NS vs NL
for grp in ["DM", "IBM", "IMNM", "PM", "NS"]:
    a = df_c2.loc[df_c2["diagnosis"] == grp, "IT1"].values
    b = df_c2.loc[df_c2["diagnosis"] == "NT", "IT1"].values
    if len(a) >= 3 and len(b) >= 3:
        d = cohens_d(a, b)
        se = d_se(a, b, d)
        records.append({"cohort": "GSE128470", "group": grp,
                         "d": d, "se": se, "n_a": len(a), "n_b": len(b)})
meta_df = pd.DataFrame(records)
print("\nMeta-analysis effect sizes:")
print(meta_df.to_string(index=False))

# Random-effects meta-analysis (DerSimonian-Laird) per disease group
def re_meta(d_arr, se_arr):
    w = 1.0 / se_arr**2
    d_bar_fe = (w * d_arr).sum() / w.sum()
    # heterogeneity Q
    Q = (w * (d_arr - d_bar_fe)**2).sum()
    df = len(d_arr) - 1
    if df > 0:
        c = w.sum() - (w**2).sum() / w.sum()
        tau2 = max((Q - df) / c, 0.0) if c > 0 else 0.0
        I2 = max((Q - df) / Q, 0.0) * 100 if Q > 0 else 0.0
    else:
        tau2 = 0.0
        I2 = 0.0
    w_re = 1.0 / (se_arr**2 + tau2)
    d_re = (w_re * d_arr).sum() / w_re.sum()
    se_re = np.sqrt(1.0 / w_re.sum())
    return d_re, se_re, tau2, I2

# Collapse "myositis" by computing an overall pooled effect across cohorts
# but only for groups with both cohorts (DM, IBM, IMNM)
pooled = []
for grp in ["DM", "IBM", "IMNM"]:
    sub = meta_df[meta_df["group"] == grp]
    if len(sub) >= 2:
        d_re, se_re, tau2, I2 = re_meta(sub["d"].values, sub["se"].values)
        pooled.append({"group": grp, "d_re": d_re, "se_re": se_re,
                       "tau2": tau2, "I2": I2, "k": len(sub)})
# Add AS only in cohort 1
sub = meta_df[meta_df["group"] == "ASyS"]
if len(sub):
    pooled.append({"group": "ASyS", "d_re": sub.iloc[0]["d"],
                   "se_re": sub.iloc[0]["se"], "tau2": 0,
                   "I2": np.nan, "k": 1})
pooled_df = pd.DataFrame(pooled)
print("\nPooled (random-effects):")
print(pooled_df.to_string(index=False))

# Save outputs
meta_df.to_csv(f"{OUT_DIR}/30_multicohort_per_group_effects.csv", index=False)
pooled_df.to_csv(f"{OUT_DIR}/30_multicohort_pooled_effects.csv", index=False)

# Compute concordance: which IT1 genes are up in disease vs normal in both cohorts
# (gene-level mean per group, t-stat sign concordance)
def gene_t_stat(expr, dx, target, control):
    """Welch t-stat per gene, target vs control."""
    a = expr.loc[:, dx == target]
    b = expr.loc[:, dx == control]
    if a.shape[1] < 3 or b.shape[1] < 3:
        return None
    mu_a, mu_b = a.mean(axis=1), b.mean(axis=1)
    sd_a, sd_b = a.std(axis=1) + 1e-9, b.std(axis=1) + 1e-9
    n_a, n_b = a.shape[1], b.shape[1]
    se = np.sqrt(sd_a**2 / n_a + sd_b**2 / n_b)
    return (mu_a - mu_b) / se

# Cohort 1 gene-level t (any disease vs normal — use IBM as reference)
def assemble_t(expr, dx_arr, comparator):
    """t-stat for IT1 genes per disease group."""
    out = pd.DataFrame(index=[g for g in IT1_GENES if g in expr.index])
    dx_arr = pd.Series(dx_arr, index=expr.columns)
    for grp in dx_arr.unique():
        if grp == "NT" or grp == comparator: continue
        t = gene_t_stat(expr, dx_arr, grp, comparator)
        if t is not None:
            out[grp] = t.reindex(out.index).values
    return out

# Cohort 1 dx mapping
sample_to_dx_c1_full = dict(zip(df_c1["sample"], df_c1["diagnosis"]))
dx_arr_c1 = pd.Series([sample_to_dx_c1_full.get(s, np.nan)
                        for s in log_cpm.columns], index=log_cpm.columns)
log_cpm_used = log_cpm.loc[:, dx_arr_c1.notna()]
dx_arr_c1 = dx_arr_c1.dropna()
t_c1 = assemble_t(log_cpm_used, dx_arr_c1, "NT")

# Cohort 2 dx mapping (use NT = Normal)
dx_arr_c2 = pd.Series(dx_c2, index=expr_c2.columns)
t_c2 = assemble_t(expr_c2, dx_arr_c2, "NT")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18.0, 10.0))
gs = gridspec.GridSpec(
    2, 3, figure=fig,
    left=0.040, right=0.992, top=0.945, bottom=0.085,
    wspace=0.32, hspace=0.42,
)

DX_COL = {"NT":"#7F8C8D","IMNM":"#2980B9","IBM":"#8E44AD",
          "DM":"#E67E22","ASyS":"#C0392B","PM":"#1A7A4A","NS":"#7D3C98"}

# ── (a) cohort composition — proper visualisation ────────────────────────
# Stacked horizontal bars showing each cohort's diagnostic-group composition.
# The bar length = total cohort n, segments coloured by diagnostic group,
# with per-segment count annotated. Modality + IT1-gene-detection rate
# annotated as concise inline text.
ax = fig.add_subplot(gs[0, 0])

# Counts per diagnosis per cohort (preserve a stable diagnostic-group order)
COHORT_INFO = [
    ("GSE220915\n(Casal-Dominguez 2023)\nbulk RNA-seq",
     df_c1, ["NT","IMNM","IBM","DM","ASyS"],
     len(it1_genes_present), len(IT1_GENES)),
    ("GSE128470\n(Greenberg 2019)\nAffymetrix microarray",
     df_c2, ["NT","NS","PM","IMNM","DM","IBM"],
     len(it1_c2_genes), len(IT1_GENES)),
]

bar_h = 0.55
y_positions = [1.05, 0.0]  # Cohort 1 on top, Cohort 2 below

for y, (label, df_x, order_x, n_genes, n_total_genes) in zip(y_positions, COHORT_INFO):
    counts = [int((df_x["diagnosis"] == g).sum()) for g in order_x]
    n_total = sum(counts)
    # Stacked horizontal segments
    left = 0
    for g, n in zip(order_x, counts):
        if n == 0: continue
        ax.barh(y, n, left=left, height=bar_h,
                color=DX_COL[g], edgecolor="white", lw=0.8)
        # Always render count INSIDE segment in white. Group name placed
        # via a small annotation below the bar (with leader line if needed)
        # — handled in a separate loop below to keep adjacent labels apart.
        if n >= 4:
            ax.text(left + n/2, y, f"{n}",
                    ha="center", va="center",
                    fontsize=10.0, fontweight="bold", color="white")
        left += n
    # Cohort label on the left
    ax.text(-3, y, label, ha="right", va="center",
            fontsize=10.5, fontweight="bold", color="#222")
    # Total + IT1 gene detection on the right
    pct_genes = 100 * n_genes / n_total_genes
    ax.text(n_total + 4, y,
            f"n = {n_total}\nIT1 genes: {n_genes}/{n_total_genes} ({pct_genes:.0f}%)",
            ha="left", va="center", fontsize=10.0, color="#444")

    # Group-name labels BELOW the bar (R1): labels are pushed apart to a minimum
    # spacing and joined to their segment by a short leader line, so narrow
    # adjacent segments never produce overlapping names.
    y_label = y - bar_h/2 - 0.16
    cxs, left = [], 0
    for g, n in zip(order_x, counts):
        cxs.append((g, left + n / 2)); left += n
    lx_prev = -np.inf
    for g, cx in cxs:
        lx = max(cx, lx_prev + 21)
        if abs(lx - cx) > 0.5:
            ax.plot([cx, lx], [y - bar_h/2 - 0.01, y_label + 0.01], color=DX_COL[g], lw=0.7)
        ax.text(lx, y_label, g, ha="center", va="top",
                fontsize=9.5, fontweight="bold", color=DX_COL[g])
        lx_prev = lx

# Formatting — extra bottom margin for two-row group-name annotation rows
ax.set_xlim(-65, 250)
ax.set_ylim(-0.80, 1.45)
ax.set_xlabel("Number of biopsies", fontsize=10.5, labelpad=10)
ax.set_xticks([0, 50, 100, 150, 200])
ax.tick_params(axis="x", labelsize=9.5)
ax.set_yticks([])
for s in ax.spines.values(): s.set_visible(False)
ax.spines["bottom"].set_visible(True)
ax.spines["bottom"].set_color("#444")
ax.spines["bottom"].set_linewidth(0.8)
ax.set_title("Cohort composition by diagnostic group",
             fontsize=12, pad=6, loc="left", fontweight="bold")
ax.text(-0.05, 1.08, "a", transform=ax.transAxes, **PANEL_LBL)


# ── (b) GSE220915 violin ─────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
order_c1 = ["NT","IMNM","IBM","DM","ASyS"]
data_c1 = [df_c1.loc[df_c1["diagnosis"]==g, "IT1"].values for g in order_c1]
parts = ax.violinplot(data_c1, positions=range(len(order_c1)),
                      widths=0.78, showmedians=True, showextrema=False)
for body, g in zip(parts["bodies"], order_c1):
    body.set_facecolor(DX_COL[g]); body.set_alpha(0.55)
    body.set_edgecolor(DX_COL[g]); body.set_linewidth(0.6)
parts["cmedians"].set_color("#222"); parts["cmedians"].set_linewidth(0.8)
rng = np.random.default_rng(0)
for i, g in enumerate(order_c1):
    yv = data_c1[i]
    xs = i + rng.uniform(-0.13, 0.13, size=len(yv))
    ax.scatter(xs, yv, s=4, c=DX_COL[g], alpha=0.85,
               edgecolor="white", linewidth=0.25, zorder=3)
ax.set_xticks(range(len(order_c1)))
ax.set_xticklabels([f"{g}\n(n={int((df_c1['diagnosis']==g).sum())})"
                    for g in order_c1], fontsize=9.5)
ax.set_ylabel("IT1 score", fontsize=10.5)
ax.set_title("Cohort 1: GSE220915  (n = 165)", fontsize=11.5, pad=4)
ax.axhline(0, color="#bbb", lw=0.4, ls="--")
ax.text(-0.18, 1.05, "b", transform=ax.transAxes, **PANEL_LBL)


# ── (c) GSE128470 violin ─────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
order_c2 = ["NT","NS","PM","IMNM","DM","IBM"]
data_c2 = [df_c2.loc[df_c2["diagnosis"]==g, "IT1"].values for g in order_c2]
parts = ax.violinplot(data_c2, positions=range(len(order_c2)),
                      widths=0.78, showmedians=True, showextrema=False)
for body, g in zip(parts["bodies"], order_c2):
    body.set_facecolor(DX_COL[g]); body.set_alpha(0.55)
    body.set_edgecolor(DX_COL[g]); body.set_linewidth(0.6)
parts["cmedians"].set_color("#222"); parts["cmedians"].set_linewidth(0.8)
for i, g in enumerate(order_c2):
    yv = data_c2[i]
    xs = i + rng.uniform(-0.13, 0.13, size=len(yv))
    ax.scatter(xs, yv, s=8, c=DX_COL[g], alpha=0.85,
               edgecolor="white", linewidth=0.25, zorder=3)
ax.set_xticks(range(len(order_c2)))
ax.set_xticklabels([f"{g}\n(n={int((df_c2['diagnosis']==g).sum())})"
                    for g in order_c2], fontsize=9.5)
ax.set_ylabel("IT1 score", fontsize=10.5)
ax.set_title("Cohort 2: GSE128470  (n = 77)", fontsize=11.5, pad=4)
ax.axhline(0, color="#bbb", lw=0.4, ls="--")
ax.text(-0.18, 1.05, "c", transform=ax.transAxes, **PANEL_LBL)


# ── (d) per-cohort effect sizes forest ───────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
y_pos = np.arange(len(meta_df))[::-1]
for i, (idx, r) in enumerate(meta_df.iterrows()):
    yi = y_pos[i]
    col = "#C0392B" if r["d"] > 0 else "#2980B9"
    lo, hi = r["d"] - 1.96 * r["se"], r["d"] + 1.96 * r["se"]
    ax.plot([lo, hi], [yi, yi], color=col, lw=1.2, alpha=0.85)
    ax.plot(r["d"], yi, "D", color=col, markersize=5,
            markeredgecolor="white", markeredgewidth=0.5, zorder=3)
ax.axvline(0, color="#222", lw=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{r['cohort']}\n{r['group']} vs NT (n={r['n_a']}/{r['n_b']})"
                    for _, r in meta_df.iterrows()],
                   fontsize=9.0)
ax.set_xlabel("Cohen's d  (disease − control)", fontsize=10.5)
ax.set_xlim(-0.6, 5.5)
ax.set_title("Per-cohort effect sizes",
             fontsize=11.5, pad=4)
ax.text(-0.40, 1.05, "d", transform=ax.transAxes, **PANEL_LBL)


# ── (e) random-effects meta-analysis ─────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
y_pos2 = np.arange(len(pooled_df))[::-1]
for i, (idx, r) in enumerate(pooled_df.iterrows()):
    yi = y_pos2[i]
    col = "#C0392B" if r["d_re"] > 0 else "#2980B9"
    lo, hi = r["d_re"] - 1.96 * r["se_re"], r["d_re"] + 1.96 * r["se_re"]
    ax.plot([lo, hi], [yi, yi], color=col, lw=1.4, alpha=0.85)
    ax.plot(r["d_re"], yi, "D", color=col, markersize=7,
            markeredgecolor="white", markeredgewidth=0.6, zorder=4)
    label = f"{r['group']} (k={int(r['k'])})"
    if not np.isnan(r["I2"]):
        label += f"  I²={r['I2']:.0f}%"
    ax.text(hi + 0.10, yi, label, va="center", fontsize=10.5, color="#333")
ax.axvline(0, color="#222", lw=0.5)
ax.set_yticks(y_pos2)
ax.set_yticklabels([f"{r['group']} pooled" for _, r in pooled_df.iterrows()],
                   fontsize=7)
ax.set_xlabel("Random-effects pooled Cohen's d  (95 % CI)", fontsize=10.5)
ax.set_xlim(-0.5, 6.0)
ax.set_title("Random-effects meta-analysis",
             fontsize=11.5, pad=4)
ax.text(-0.20, 1.05, "e", transform=ax.transAxes, **PANEL_LBL)


# ── (f) cross-cohort gene-level concordance heatmap ──────────────────────
ax = fig.add_subplot(gs[1, 2])
# Combine t-stats across cohorts (any disease vs NT) — average per gene per cohort
# t_c1 has columns for each disease in C1
# t_c2 has columns for each disease in C2
mean_t_c1 = t_c1.mean(axis=1)
mean_t_c2 = t_c2.mean(axis=1)
combined = pd.DataFrame({
    "C1 (avg disease vs NT)": mean_t_c1,
    "C2 (avg disease vs NT)": mean_t_c2,
})
# Keep genes with values in both
combined = combined.dropna()
# R1: drop genes that are invariant (not measurable) in either cohort -- e.g.
# FCGR1A, whose only single-gene GPL96 probe sits at the array floor in all
# GSE128470 samples (t = 0); extended_validation likewise drops invariant genes.
combined = combined[(combined != 0).all(axis=1)]
# Sort by mean across cohorts
combined["sum"] = combined.sum(axis=1)
combined = combined.sort_values("sum", ascending=False).head(25)
combined = combined.drop(columns=["sum"])

from matplotlib.colors import LinearSegmentedColormap
HEAT_T = LinearSegmentedColormap.from_list("dvdv",
    ["#1A3A6B","#2980B9","#FFFFFF","#E67E22","#C0392B"], N=256)
nrows_c9, ncols_c9 = combined.shape
im = ax.pcolormesh(
    np.arange(ncols_c9 + 1) - 0.5, np.arange(nrows_c9 + 1) - 0.5,
    combined.values, cmap=HEAT_T, vmin=-6, vmax=6,
    edgecolors="white", lw=0.3, rasterized=False, shading="flat",
)
ax.invert_yaxis()
ax.set_aspect("auto")
ax.set_xticks(range(combined.shape[1]))
ax.set_xticklabels(combined.columns, rotation=20, ha="right",
                    fontsize=9.5)
ax.set_yticks(range(combined.shape[0]))
ax.set_yticklabels(combined.index, fontsize=9.5)
cbar = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.025, shrink=0.85)
cbar.set_label("t-statistic (disease vs NT)", fontsize=9.5)
cbar.ax.tick_params(labelsize=5.5); cbar.outline.set_linewidth(0.4)
n_concordant = int(((combined > 0).all(axis=1) | (combined < 0).all(axis=1)).sum())
print(f"  Panel f: {int((combined > 0).all(axis=1).sum())}/{len(combined)} genes up in both cohorts; "
      f"min C2 t = {combined.iloc[:, 1].min():.2f}")
ax.set_title("Top 25 IT1 genes — cross-cohort concordance",
             fontsize=12.0, pad=4)
ax.text(-0.30, 1.05, "f", transform=ax.transAxes, **PANEL_LBL)


# ─── suptitle + save ─────────────────────────────────────────────────────
# Figure title removed (journal convention).

out_pdf = f"{OUT_DIR}/MainFig9_multicohort.pdf"
out_png = f"{OUT_DIR}/MainFig9_multicohort.png"
fig.savefig(out_pdf, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
fig.savefig(out_png, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
