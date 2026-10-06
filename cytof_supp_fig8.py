"""
Supp. Fig. S8 — IT1 composite vs classical type-I / type-II IFN sub-signatures
==============================================================================
Reviewer critique #3 + #4: IT1 mixes type-I and type-II IFN response,
and discriminates AS from NT/IMNM but not from DM. Dissect IT1 into its
IFN-α/β-responsive and IFN-γ-responsive sub-components and benchmark each
(plus a classical 4-gene IFN score) in a head-to-head ROC analysis across
the GSE220915 5 diagnostic groups.
"""
import os, gzip, json, urllib.request, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score
from nature_style import apply_nature_style
apply_nature_style()
SC, DC = 3.46, 7.09

VAL_DIR, OUT_DIR = "./validation", "./cytof_output_nature"
COUNTS, SERIES = f"{VAL_DIR}/counts.tsv.gz", f"{VAL_DIR}/series.txt.gz"

# Composite (the 41-gene IT1)
IT1_GENES = {
    "CD14","ITGAM","ITGAX","FCGR3A","FCGR1A","CD68","CD86","S100A8","S100A9",
    "S100A12","VCAN","LYZ","IL1B","TNF","CCL2","CXCL10","CXCL9","CD1C",
    "CLEC4C","IRF8","IRF7","NKG7","GNLY","GZMB","GZMA","GZMK","PRF1","KLRD1",
    "KLRK1","NCAM1","FCGR3B","HLA-DRA","HLA-DRB1","CD38","CD69","CD274",
    "IFNG","STAT1","ISG15","MX1","OAS1",
}
# Sub-signatures
IFN_I_GENES = {"ISG15","MX1","OAS1","IRF7","IFI27","IFI44","IFI44L","RSAD2",
               "OAS2","OAS3","IFIT1","IFIT3","STAT2"}
IFN_II_GENES = {"CXCL9","CXCL10","CXCL11","IRF1","GBP1","GBP5","CIITA",
                "STAT1","IDO1","IFNG"}
CLASSIC_IFN_GENES = {"MX1","ISG15","OAS1","IFI27"}

print(f"  IT1 size:      {len(IT1_GENES)}")
print(f"  IFN-I core:    {len(IFN_I_GENES)}")
print(f"  IFN-II core:   {len(IFN_II_GENES)}")
print(f"  Classical IFN: {len(CLASSIC_IFN_GENES)}")

# ── [1] Parse series matrix ──
print("\n[1/4] Parsing series matrix …")
with gzip.open(SERIES, "rt") as f:
    lines = f.readlines()
titles, diagnoses = [], []
for ln in lines:
    if ln.startswith("!Sample_title"):
        titles = [x.strip().strip('"') for x in ln.split("\t")[1:]]
    elif ln.startswith("!Sample_characteristics_ch1") and "diagnosis:" in ln:
        diagnoses = [x.strip().strip('"').replace("diagnosis: ","")
                     for x in ln.split("\t")[1:]]
sample_ids = [t.rsplit("_", 1)[0] for t in titles]
def short(d):
    d = d.strip().upper()
    if "IMNM" in d: return "IMNM"
    if "IBM" in d:  return "IBM"
    if "AS" in d:   return "AS"
    if "DM" in d:   return "DM"
    if "NORMAL" in d: return "NT"
    return d[:6]
dx = [short(d) for d in diagnoses]
meta = pd.DataFrame({"sample": sample_ids, "diagnosis": dx})
print(f"  group counts: {meta['diagnosis'].value_counts().to_dict()}")

# ── [2] symbol → ENSG ──
print("\n[2/4] Resolving symbols via mygene.info …")
all_syms = sorted(IT1_GENES | IFN_I_GENES | IFN_II_GENES | CLASSIC_IFN_GENES)

def query_mygene(symbols):
    url = "https://mygene.info/v3/query"
    data = f"q={','.join(symbols)}&scopes=symbol&fields=ensembl.gene,symbol&species=human"
    req = urllib.request.Request(url, data=data.encode(), method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=30).read().decode())

resp = query_mygene(all_syms)
sym2ens = {}
for r in resp:
    if r.get("notfound"): continue
    ens = r.get("ensembl")
    if ens is None: continue
    if isinstance(ens, list): ens = ens[0]
    gid, s = ens.get("gene"), r.get("symbol", r.get("query"))
    if gid: sym2ens[s] = gid
print(f"  resolved {len(sym2ens)}/{len(all_syms)}")

# ── [3] Load counts + score ──
print("\n[3/4] Loading counts + scoring …")
cnt = pd.read_csv(COUNTS, sep="\t", compression="gzip", index_col=0)
cnt.index = cnt.index.str.split(".").str[0]
meta = meta.set_index("sample").loc[cnt.columns]
lcpm = np.log2(cnt.div(cnt.sum(0), axis=1) * 1e6 + 1.0)

def z(x):
    mu, sd = np.nanmean(x), np.nanstd(x)
    return (x - mu) / sd if sd > 0 else x - mu

def score(geneset):
    ens = [sym2ens[s] for s in geneset if s in sym2ens]
    ens = [e for e in ens if e in lcpm.index]
    if not ens: return pd.Series(0.0, index=lcpm.columns), 0
    block = lcpm.loc[ens].apply(z, axis=1)
    return block.mean(axis=0), len(ens)

s_it1,    n1 = score(IT1_GENES)
s_ifn_i,  n2 = score(IFN_I_GENES)
s_ifn_ii, n3 = score(IFN_II_GENES)
s_class,  n4 = score(CLASSIC_IFN_GENES)
print(f"  detected genes: IT1 {n1}/{len(IT1_GENES)} | "
      f"IFN-I {n2}/{len(IFN_I_GENES)} | "
      f"IFN-II {n3}/{len(IFN_II_GENES)} | "
      f"classical {n4}/{len(CLASSIC_IFN_GENES)}")

meta["IT1"]           = s_it1
meta["IFN_I"]         = s_ifn_i
meta["IFN_II"]        = s_ifn_ii
meta["Classical_IFN"] = s_class
meta.to_csv(f"{OUT_DIR}/S8_subscore_per_sample.csv")

# ── [4] Figure ──
print("\n[4/4] Plotting …")
DX_ORDER = ["NT","IMNM","IBM","DM","AS"]
DX_COL   = {"NT":"#95A5A6","IMNM":"#2980B9","IBM":"#8E44AD",
            "DM":"#F39C12","AS":"#C0392B"}
SCORES = [("IT1",           f"IT1 composite  ({n1} genes, mixed I+II)"),
          ("IFN_I",         f"IFN-I response  ({n2} ISG core genes)"),
          ("IFN_II",        f"IFN-II / IFN-γ  ({n3} IFN-γ-induced genes)"),
          ("Classical_IFN", f"Classical IFN score  ({n4}-gene)")]

S8_RC = {
    "font.size":7,"axes.titlesize":7.5,"axes.labelsize":7,
    "xtick.labelsize":6,"ytick.labelsize":6,
    "axes.linewidth":0.6,"xtick.major.width":0.6,"ytick.major.width":0.6,
    "xtick.major.size":2.5,"ytick.major.size":2.5,
    "lines.linewidth":1.0,"patch.linewidth":0.4,"legend.fontsize":5.5,
}
with plt.rc_context(S8_RC):
    fig = plt.figure(figsize=(DC+1.2, 5.6))
    gs  = gridspec.GridSpec(2, 4, figure=fig, wspace=0.42, hspace=0.55)

    # ── Top row: violin/strip per diagnosis ──
    rng = np.random.default_rng(0)
    for i, (col, label) in enumerate(SCORES):
        ax = fig.add_subplot(gs[0, i])
        groups = [meta.loc[meta["diagnosis"]==g, col].values for g in DX_ORDER]
        parts = ax.violinplot(groups, positions=np.arange(len(DX_ORDER)),
                              widths=0.82, showmeans=False, showmedians=True)
        for j, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(DX_COL[DX_ORDER[j]]); pc.set_alpha(0.55)
            pc.set_edgecolor(DX_COL[DX_ORDER[j]]); pc.set_linewidth(0.7)
        for k in ["cmedians","cbars","cmins","cmaxes"]:
            if k in parts:
                parts[k].set_color("#333"); parts[k].set_lw(0.5)
        for j, g in enumerate(DX_ORDER):
            y = meta.loc[meta["diagnosis"]==g, col].values
            x = j + rng.uniform(-0.14, 0.14, size=len(y))
            ax.scatter(x, y, s=5, c=DX_COL[g], alpha=0.75,
                       edgecolor="white", linewidth=0.2, zorder=3)
        h, p = stats.kruskal(*groups)
        ax.set_xticks(range(len(DX_ORDER)))
        ax.set_xticklabels(DX_ORDER, fontsize=6)
        ax.axhline(0, color="#bbb", lw=0.4, ls="--")
        ax.set_title(f"{label}\nKW H={h:.1f}, p={p:.1e}",
                     fontsize=6.6, pad=3)
        if i == 0:
            ax.set_ylabel("Sub-score (z, mean over signature)", fontsize=6.6)

    # ── Bottom row: ROC AS vs each comparator ──
    for i, (col, label) in enumerate(SCORES):
        ax = fig.add_subplot(gs[1, i])
        ax.plot([0,1],[0,1], color="#bbb", lw=0.4, ls="--")
        for comp in ["NT","IMNM","IBM","DM"]:
            sub = meta[meta["diagnosis"].isin(["AS", comp])].copy()
            y_true = (sub["diagnosis"] == "AS").astype(int).values
            y_score = sub[col].values
            if y_true.sum() == 0 or y_true.sum() == len(y_true):
                continue
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)
            ax.plot(fpr, tpr, color=DX_COL[comp], lw=1.0,
                    label=f"AS vs {comp}  AUC={auc:.2f}")
        ax.set_xlim(0,1); ax.set_ylim(0,1.02)
        ax.set_xlabel("False positive rate", fontsize=6.6)
        if i == 0: ax.set_ylabel("True positive rate", fontsize=6.6)
        ax.legend(loc="lower right", fontsize=5.2, frameon=False,
                  handlelength=1.2, handletextpad=0.4, borderpad=0.2)
        ax.set_title(label, fontsize=6.6, pad=3)

    def _panel(ax, lbl, x=-0.22, y=1.10):
        ax.text(x, y, lbl, transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="bottom", ha="left")
    for i, ax in enumerate(fig.axes[:4]): _panel(ax, "abcd"[i])
    for i, ax in enumerate(fig.axes[4:8]): _panel(ax, "efgh"[i])

    fig.savefig(f"{OUT_DIR}/S8_IT1_vs_IFN_subscores.pdf",
                dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT_DIR}/S8_IT1_vs_IFN_subscores.png",
                dpi=300, bbox_inches="tight")
    print(f"  Saved {OUT_DIR}/S8_IT1_vs_IFN_subscores.pdf/.png")

# ── AUC table for manuscript ──
rows = []
for col, label in SCORES:
    for comp in ["NT","IMNM","IBM","DM"]:
        sub = meta[meta["diagnosis"].isin(["AS", comp])].copy()
        y_true = (sub["diagnosis"] == "AS").astype(int).values
        if y_true.sum() == 0 or y_true.sum() == len(y_true): continue
        auc = roc_auc_score(y_true, sub[col].values)
        rows.append({"signature": col, "label": label.split("  ")[0],
                     "comparator": comp,
                     "n_AS": (sub["diagnosis"]=="AS").sum(),
                     "n_comp": (sub["diagnosis"]==comp).sum(),
                     "AUC": round(auc, 3)})
auc_df = pd.DataFrame(rows)
auc_df.to_csv(f"{OUT_DIR}/S8_AUC_summary.csv", index=False)
print("\n  AUC summary (AS vs each comparator):")
print(auc_df.pivot_table(index="label", columns="comparator", values="AUC").to_string())
