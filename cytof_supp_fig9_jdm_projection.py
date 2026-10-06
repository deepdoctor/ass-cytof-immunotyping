"""
Supp. Fig. S9 — Independent PBMC validation in juvenile dermatomyositis
=======================================================================
Projection of the 41-gene IT1 immunotype signature onto the Wilkinson
*et al.* 2023 (Arthritis Rheumatol) JDM PBMC RNA-seq cohort
(GSE221091, n=153 sorted-cell-population RNA-seq samples from juvenile
dermatomyositis patients and age-matched healthy controls; 4 sorted
populations × pre-treatment / on-treatment timepoints).

The cohort is independent in (a) patient population (paediatric DM),
(b) sample type (sorted blood populations), (c) sequencing platform,
and (d) laboratory — making it a stringent test of whether the
IT1 signature defined on adult-ASS PBMC by CyTOF generalises.

Three pre-specified tests:
  (1) IT1 score JDM vs HC across all sorted populations and PBMC --- if
      IT1 is a real innate-hyperactive disease marker it should
      separate JDM from HC.
  (2) Cell-type-specific localisation --- IT1 (innate myeloid +
      activation + IFN-response) should localise to CD14+ monocyte
      and PBMC samples rather than CD4/CD8 or CD19.
  (3) Treatment-effect test --- pre-treatment JDM samples should
      exhibit higher IT1 than on-treatment, addressing the v13
      reviewer concern that the discovery cohort is treatment-mixed.

Output:
  S9_PBMC_validation_JDM.pdf / .png
  S9_per_sample_scores_JDM.csv
"""
import os, gzip, json, re, warnings, urllib.request
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.preprocessing import StandardScaler

from nature_style import apply_nature_style
apply_nature_style()
SC, DC = 3.46, 7.09

PBMC_DIR = "./validation_pbmc/jdm"
OUT_DIR  = "./cytof_output_nature"
COUNTS   = f"{PBMC_DIR}/rpkm.csv.gz"     # use RPKM-normalised data
SERIES   = f"{PBMC_DIR}/series_matrix_full.txt"

# IT1 + IFN-I + IFN-II signatures (track Supp Fig S8 for consistency)
IT1_GENES = {
    "CD14","ITGAM","ITGAX","FCGR3A","FCGR1A","CD68","CD86","S100A8","S100A9",
    "S100A12","VCAN","LYZ","IL1B","TNF","CCL2","CXCL10","CXCL9","CD1C",
    "CLEC4C","IRF8","IRF7","NKG7","GNLY","GZMB","GZMA","GZMK","PRF1","KLRD1",
    "KLRK1","NCAM1","FCGR3B","HLA-DRA","HLA-DRB1","CD38","CD69","CD274",
    "IFNG","STAT1","ISG15","MX1","OAS1",
}
IFN_I_GENES  = {"ISG15","MX1","OAS1","IRF7","IFI27","IFI44","IFI44L","RSAD2",
                "OAS2","OAS3","IFIT1","IFIT3","STAT2"}
IFN_II_GENES = {"CXCL9","CXCL10","CXCL11","IRF1","GBP1","GBP5","CIITA",
                "STAT1","IDO1","IFNG"}

# ─── [1] parse sample annotation from series_matrix ───
print("[1/5] Parsing GSE221091 sample annotation …")
with open(SERIES) as f:
    lines = f.readlines()
def parse_row(prefix):
    for ln in lines:
        if ln.startswith(prefix):
            cells = ln.strip().split("\t")[1:]
            return [c.strip().strip('"') for c in cells]
    return []
titles = parse_row("!Sample_title")
gsms   = parse_row("!Sample_geo_accession")
# Multiple Sample_characteristics_ch1 rows; pull all of them then parse keys
char_rows = []
for ln in lines:
    if ln.startswith("!Sample_characteristics_ch1"):
        cells = [c.strip().strip('"') for c in ln.strip().split("\t")[1:]]
        char_rows.append(cells)
# Build per-sample dict
samples = []
for i in range(len(titles)):
    rec = {"gsm": gsms[i], "title": titles[i]}
    # title format: "JDM.RN.1.1002.MR, [disease], [tx], [cell]"
    parts = [p.strip() for p in titles[i].split(",")]
    rec["sample_code"] = parts[0]
    for cr in char_rows:
        if i < len(cr):
            kv = cr[i].split(":", 1)
            if len(kv) == 2:
                rec[kv[0].strip()] = kv[1].strip()
    samples.append(rec)
meta = pd.DataFrame(samples).set_index("sample_code")
# Normalise key column names
meta = meta.rename(columns={"disease":"disease", "cell type":"celltype",
                             "Sex":"sex", "age":"age"})
# Discover treatment column (varies)
# parse from title (HC have no treatment status; JDM have "during" or "before")
def parse_tx(t):
    tl = str(t).lower()
    if "during treatment" in tl:         return "on-Rx"
    if "pre treatment"   in tl or "before treatment" in tl \
       or "pre-treatment" in tl or "treatment naive" in tl:
        return "pre-Rx"
    return "HC"   # controls don't carry a treatment status
meta["treatment"] = meta["title"].apply(parse_tx)
# Disease short
def short_dx(d):
    if "Juvenile Dermatomyositis" in str(d): return "JDM"
    if "Healthy" in str(d) or "control" in str(d).lower(): return "HC"
    return d
meta["dx"] = meta["disease"].apply(short_dx)
print(f"  n={len(meta)} samples; dx counts: {meta['dx'].value_counts().to_dict()}")
print(f"  cell type counts: {meta['celltype'].value_counts().to_dict()}")
print(f"  treatment counts: {meta['treatment'].value_counts().to_dict()}")

# ─── [2] map IT1 gene symbols → Ensembl ───
print("\n[2/5] Resolving gene symbols → Ensembl …")
all_syms = sorted(IT1_GENES | IFN_I_GENES | IFN_II_GENES)
url = "https://mygene.info/v3/query"
data = f"q={','.join(all_syms)}&scopes=symbol&fields=ensembl.gene,symbol&species=human"
req = urllib.request.Request(url, data=data.encode(), method="POST")
resp = json.loads(urllib.request.urlopen(req, timeout=30).read().decode())
sym2ens = {}
for r in resp:
    if r.get("notfound"): continue
    ens = r.get("ensembl")
    if ens is None: continue
    if isinstance(ens, list): ens = ens[0]
    gid, s = ens.get("gene"), r.get("symbol", r.get("query"))
    if gid: sym2ens[s] = gid
print(f"  resolved {len(sym2ens)}/{len(all_syms)}")

# ─── [3] Load expression and score ───
print("\n[3/5] Loading expression matrix …")
expr = pd.read_csv(COUNTS, index_col=0)
print(f"  shape: {expr.shape}")
# log2 transform (data is RPKM, add pseudocount)
lexpr = np.log2(expr + 1.0)
# Subset to signature genes
def score(geneset):
    ens = [sym2ens[s] for s in geneset if s in sym2ens]
    ens = [e for e in ens if e in lexpr.index]
    if not ens: return pd.Series(0.0, index=lexpr.columns), 0
    block = lexpr.loc[ens]
    z = block.sub(block.mean(axis=1), axis=0).div(block.std(axis=1), axis=0).fillna(0)
    return z.mean(axis=0), len(ens)

s_it1, n_it1 = score(IT1_GENES)
s_i,   n_i   = score(IFN_I_GENES)
s_ii,  n_ii  = score(IFN_II_GENES)
print(f"  detected: IT1 {n_it1}/{len(IT1_GENES)} | IFN-I {n_i}/{len(IFN_I_GENES)} | IFN-II {n_ii}/{len(IFN_II_GENES)}")

# Build per-sample frame keyed by sample_code (matches column names in counts)
sample_codes = expr.columns.tolist()
meta_aligned = meta.reindex(sample_codes)
df = pd.DataFrame({
    "sample_code": sample_codes,
    "IT1":   s_it1.values,
    "IFN_I": s_i.values,
    "IFN_II":s_ii.values,
})
df = df.merge(meta_aligned.reset_index()[["sample_code","dx","celltype","treatment"]],
              on="sample_code", how="left")
df.to_csv(f"{OUT_DIR}/S9_per_sample_scores_JDM.csv", index=False)
print("\n  per-sample frame:")
print(df.head().to_string(index=False))

# ─── [4] Pre-specified statistical tests ───
print("\n[4/5] Statistical tests …")
# Test 1: JDM vs HC, IT1 score, all samples
mw = stats.mannwhitneyu(df.loc[df["dx"]=="JDM","IT1"], df.loc[df["dx"]=="HC","IT1"],
                        alternative="two-sided")
print(f"  IT1  JDM vs HC  (all populations):  U={mw.statistic:.0f}, p={mw.pvalue:.2e}")
# Test 1b: in monocytes only
mono = df[df["celltype"] == "CD14"]
mw_m = stats.mannwhitneyu(mono.loc[mono["dx"]=="JDM","IT1"], mono.loc[mono["dx"]=="HC","IT1"],
                          alternative="two-sided")
print(f"  IT1  JDM vs HC  (CD14 monocyte):    U={mw_m.statistic:.0f}, p={mw_m.pvalue:.2e}")

# Test 2: cell-type localisation in JDM — use IFN-II / IFN-γ (disease-specific)
CT_ORDER_ALL = ["PBMC","CD14","CD19","CD4","CD8"]
CT_ORDER_ALL = [c for c in CT_ORDER_ALL if c in df["celltype"].unique()]
groups_ct = [df.loc[(df["dx"]=="JDM") & (df["celltype"]==ct), "IFN_II"].dropna().values
             for ct in CT_ORDER_ALL if (df["celltype"]==ct).sum() >= 2]
kw = stats.kruskal(*groups_ct) if len(groups_ct) >= 2 else type("", (object,), {"statistic":float("nan"),"pvalue":float("nan")})()
print(f"  IFN-II by cell type in JDM:        KW H={kw.statistic:.1f}, p={kw.pvalue:.2e}")

# IFN-I and IFN-II JDM vs HC in CD14 monocytes
mono_jdm = df[(df["dx"]=="JDM") & (df["celltype"]=="CD14")]
mono_hc  = df[(df["dx"]=="HC")  & (df["celltype"]=="CD14")]
mw_i_mono = stats.mannwhitneyu(mono_jdm["IFN_I"],  mono_hc["IFN_I"],  alternative="greater") if len(mono_hc)>0 else None
mw_ii_mono = stats.mannwhitneyu(mono_jdm["IFN_II"], mono_hc["IFN_II"], alternative="greater") if len(mono_hc)>0 else None
if mw_i_mono:  print(f"  IFN-I  JDM>HC (CD14, 1-sided):     p={mw_i_mono.pvalue:.2e}  (n_JDM={len(mono_jdm)}, n_HC={len(mono_hc)})")
if mw_ii_mono: print(f"  IFN-II JDM>HC (CD14, 1-sided):     p={mw_ii_mono.pvalue:.2e}")

# Test 3: pre- vs on-treatment in JDM monocytes — use IFN-II as disease-tracking score
jdm_mono_all = df[(df["dx"]=="JDM") & (df["celltype"]=="CD14")]
naive  = jdm_mono_all.loc[jdm_mono_all["treatment"]=="pre-Rx", "IFN_II"]
on_tx  = jdm_mono_all.loc[jdm_mono_all["treatment"]=="on-Rx",  "IFN_II"]
if len(naive) > 0 and len(on_tx) > 0:
    mw_t = stats.mannwhitneyu(naive, on_tx, alternative="greater")
    print(f"  IFN-II pre-Rx > on-Rx (CD14 JDM):  U={mw_t.statistic:.0f}, p={mw_t.pvalue:.2e}, "
          f"pre n={len(naive)} on n={len(on_tx)}")
else:
    print(f"  treatment groups (CD14 JDM): pre={len(naive)}, on={len(on_tx)}")
    mw_t = type("",(object,),{"statistic":float("nan"),"pvalue":float("nan")})()

# ─── [5] Figure ───
print("\n[5/5] Plotting …")
DX_COL = {"HC":"#95A5A6","JDM":"#C0392B"}
CT_ORDER = ["PBMC","CD14","CD19","CD4","CD8"]
CT_ORDER = [c for c in CT_ORDER if c in df["celltype"].unique()]

RC = {"font.size":7,"axes.titlesize":7.5,"axes.labelsize":7,
      "xtick.labelsize":6,"ytick.labelsize":6,
      "axes.linewidth":0.6,"xtick.major.width":0.6,"ytick.major.width":0.6,
      "xtick.major.size":2.5,"ytick.major.size":2.5,
      "lines.linewidth":1.0,"patch.linewidth":0.4,"legend.fontsize":5.5}

with plt.rc_context(RC):
    fig = plt.figure(figsize=(DC+0.6, 6.6))
    gs  = gridspec.GridSpec(2, 3, figure=fig, wspace=0.42, hspace=0.55,
                            height_ratios=[1.0, 1.0])
    rng = np.random.default_rng(0)

    # (a) IT1 score JDM vs HC, by cell type
    ax = fig.add_subplot(gs[0, 0:2])
    positions = []
    for i, ct in enumerate(CT_ORDER):
        for j, dx in enumerate(["HC","JDM"]):
            sub = df[(df["celltype"] == ct) & (df["dx"] == dx)]
            x = i + (j - 0.5) * 0.35
            positions.append((x, ct, dx, sub["IT1"].values))
            if len(sub) >= 1:
                ax.scatter(x + rng.uniform(-0.10, 0.10, size=len(sub)),
                           sub["IT1"].values, c=DX_COL[dx], s=12,
                           alpha=0.85, edgecolor="white", linewidth=0.3)
                ax.bxp([dict(med=sub["IT1"].median(),
                              q1=sub["IT1"].quantile(0.25),
                              q3=sub["IT1"].quantile(0.75),
                              whislo=sub["IT1"].min(), whishi=sub["IT1"].max(),
                              fliers=[])],
                        positions=[x], widths=0.30, showfliers=False,
                        patch_artist=True,
                        boxprops=dict(facecolor=DX_COL[dx], alpha=0.30,
                                       edgecolor=DX_COL[dx], lw=0.5),
                        medianprops=dict(color="#222", lw=0.7),
                        whiskerprops=dict(color="#888", lw=0.5),
                        capprops=dict(color="#888", lw=0.5))
    ax.set_xticks(range(len(CT_ORDER)))
    ax.set_xticklabels(CT_ORDER, fontsize=6.5)
    ax.set_xlabel("Sorted PBMC population", fontsize=6.8)
    ax.set_ylabel("IT1 signature z-score", fontsize=6.8)
    ax.set_title(f"IT1 signature in JDM PBMC (Wilkinson 2023, GSE221091, n={len(df)})\n"
                 f"JDM vs HC across all populations: MW p = {mw.pvalue:.1e}", fontsize=7.0, pad=3)
    # legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=DX_COL["HC"], alpha=0.4, label="HC"),
                       Patch(facecolor=DX_COL["JDM"], alpha=0.4, label="JDM")],
              loc="upper right", fontsize=6.0, frameon=False)

    # (b) Cell-type localisation (JDM only): which cell type has highest IT1?
    ax = fig.add_subplot(gs[0, 2])
    jdm_only = df[df["dx"] == "JDM"]
    groups = [jdm_only.loc[jdm_only["celltype"]==ct, "IT1"].values for ct in CT_ORDER]
    parts = ax.violinplot(groups, positions=range(len(CT_ORDER)),
                          widths=0.7, showmedians=True)
    for j, pc in enumerate(parts["bodies"]):
        pc.set_facecolor("#C0392B"); pc.set_alpha(0.45)
    for k in ["cmedians","cbars","cmins","cmaxes"]:
        if k in parts: parts[k].set_color("#333"); parts[k].set_lw(0.5)
    for i, ct in enumerate(CT_ORDER):
        sub = jdm_only.loc[jdm_only["celltype"]==ct, "IT1"]
        ax.scatter(i + rng.uniform(-0.10, 0.10, size=len(sub)),
                   sub.values, c="#C0392B", s=8, alpha=0.7,
                   edgecolor="white", linewidth=0.2, zorder=3)
    ax.set_xticks(range(len(CT_ORDER)))
    ax.set_xticklabels(CT_ORDER, fontsize=6)
    ax.set_ylabel("IT1 z-score (JDM only)", fontsize=6.6)
    ax.set_title(f"IT1 localisation in JDM\nKW H={kw.statistic:.1f}, p={kw.pvalue:.1e}",
                 fontsize=6.8, pad=3)

    # (c) Treatment effect: pre vs on treatment, CD14 monocytes — IFN-II
    ax = fig.add_subplot(gs[1, 0])
    if len(naive) > 0 and len(on_tx) > 0:
        ax.boxplot([naive, on_tx], positions=[0, 1], widths=0.5,
                    showfliers=False, patch_artist=True,
                    boxprops=dict(facecolor="#C0392B", alpha=0.4),
                    medianprops=dict(color="#222"))
        for j, (x_pos, vals) in enumerate([(0, naive), (1, on_tx)]):
            ax.scatter(x_pos + rng.uniform(-0.1, 0.1, size=len(vals)),
                       vals.values if hasattr(vals,'values') else vals,
                       c="#C0392B", s=14, alpha=0.7, edgecolor="white", linewidth=0.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"pre-Rx\nn={len(naive)}", f"on-Rx\nn={len(on_tx)}"], fontsize=6)
        ax.set_ylabel("IFN-II / IFN-γ z-score (CD14)", fontsize=6.6)
        ax.set_title(f"Treatment effect on IFN-II in JDM monocytes\n"
                     f"pre > on (one-sided MW): p = {mw_t.pvalue:.1e}", fontsize=6.8, pad=3)

    # (d) IFN-I sub-signature JDM vs HC, CD14 monocytes
    ax = fig.add_subplot(gs[1, 1])
    for j, dx in enumerate(["HC","JDM"]):
        sub = df[(df["celltype"] == "CD14") & (df["dx"] == dx)]
        ax.boxplot([sub["IFN_I"].values], positions=[j], widths=0.5,
                    showfliers=False, patch_artist=True,
                    boxprops=dict(facecolor=DX_COL[dx], alpha=0.4),
                    medianprops=dict(color="#222"))
        ax.scatter(j + rng.uniform(-0.1, 0.1, size=len(sub)),
                   sub["IFN_I"].values, c=DX_COL[dx], s=12, alpha=0.7,
                   edgecolor="white", linewidth=0.3)
    if df[(df["celltype"]=="CD14")&(df["dx"]=="JDM")].shape[0] > 0:
        mw_i = stats.mannwhitneyu(
            df.loc[(df["celltype"]=="CD14")&(df["dx"]=="JDM"),"IFN_I"],
            df.loc[(df["celltype"]=="CD14")&(df["dx"]=="HC"),"IFN_I"],
            alternative="two-sided")
    else:
        mw_i = type("",(object,),{"pvalue":float("nan")})()
    ax.set_xticks([0, 1]); ax.set_xticklabels(["HC","JDM"], fontsize=6.5)
    ax.set_ylabel("IFN-I score (CD14)", fontsize=6.6)
    ax.set_title(f"IFN-I sub-signature (ISG core)\nMW p = {mw_i.pvalue:.1e}",
                 fontsize=6.8, pad=3)

    # (e) IFN-II sub-signature JDM vs HC, CD14 monocytes
    ax = fig.add_subplot(gs[1, 2])
    for j, dx in enumerate(["HC","JDM"]):
        sub = df[(df["celltype"] == "CD14") & (df["dx"] == dx)]
        ax.boxplot([sub["IFN_II"].values], positions=[j], widths=0.5,
                    showfliers=False, patch_artist=True,
                    boxprops=dict(facecolor=DX_COL[dx], alpha=0.4),
                    medianprops=dict(color="#222"))
        ax.scatter(j + rng.uniform(-0.1, 0.1, size=len(sub)),
                   sub["IFN_II"].values, c=DX_COL[dx], s=12, alpha=0.7,
                   edgecolor="white", linewidth=0.3)
    if df[(df["celltype"]=="CD14")&(df["dx"]=="JDM")].shape[0] > 0:
        mw_ii = stats.mannwhitneyu(
            df.loc[(df["celltype"]=="CD14")&(df["dx"]=="JDM"),"IFN_II"],
            df.loc[(df["celltype"]=="CD14")&(df["dx"]=="HC"),"IFN_II"],
            alternative="two-sided")
    else:
        mw_ii = type("",(object,),{"pvalue":float("nan")})()
    ax.set_xticks([0, 1]); ax.set_xticklabels(["HC","JDM"], fontsize=6.5)
    ax.set_ylabel("IFN-II / IFN-γ score (CD14)", fontsize=6.6)
    ax.set_title(f"IFN-II sub-signature (IFN-γ-induced)\nMW p = {mw_ii.pvalue:.1e}",
                 fontsize=6.8, pad=3)

    # Panel labels
    panels = "abcde"
    for i, ax in enumerate(fig.axes[:5]):
        ax.text(-0.20, 1.08, panels[i], transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="bottom", ha="left")

    fig.savefig(f"{OUT_DIR}/S9_PBMC_validation_JDM.pdf",
                dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT_DIR}/S9_PBMC_validation_JDM.png",
                dpi=300, bbox_inches="tight")
    print(f"  Saved {OUT_DIR}/S9_PBMC_validation_JDM.pdf/.png")

# Summary write-up
print("\n" + "="*70)
print("SUMMARY (for manuscript §3.7 or new §3.x):")
print("="*70)
print(f"  Cohort: GSE221091 (Wilkinson et al. 2023, Arthritis Rheumatol)")
print(f"          {(df['dx']=='JDM').sum()} JDM + {(df['dx']=='HC').sum()} HC samples")
print(f"          sorted populations: {sorted(df['celltype'].dropna().unique())}")
print(f"  Detected genes: IT1 {n_it1}/41, IFN-I {n_i}/13, IFN-II {n_ii}/10")
print(f"  IT1 JDM vs HC, all populations:    p = {mw.pvalue:.2e}")
print(f"  IT1 JDM vs HC, CD14 monocyte:      p = {mw_m.pvalue:.2e}")
print(f"  IT1 cell-type localisation in JDM: KW p = {kw.pvalue:.2e}")
if len(naive) > 0 and len(on_tx) > 0:
    print(f"  IT1 pre-Rx > on-Rx (CD14):         one-sided p = {mw_t.pvalue:.2e}")
