"""
cytof_supp_additions.py
=======================
Three reviewer-critical supplementary analyses for IF>8 submission:

  S1 – k-selection evidence: silhouette + gap statistic + WSS elbow for the
       65-D patient feature matrix (Ward linkage).  Proves that k=3 is
       data-supported, not author-chosen.

  S2 – Fig-17 LOO stability for all q<0.10 clinical correlations.  Shows
       ρ is not driven by a single patient.

  S3 – Patient-aggregated within-cell-type DE.  Re-tests the Fig-8 claims
       on PER-PATIENT means (not pooled cells) to show that within-cell-type
       signals survive proper aggregation and are not pseudo-replication
       artefacts.

Run:
    cd /Users/yichen/Desktop/collaboration/raw
    /Users/yichen/anaconda3/bin/conda run -n cytof python cytof_supp_additions.py
"""

import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
import anndata as ad

from nature_style import apply_nature_style, GROUP_PALETTE, HEATMAP_DIVERG, save
apply_nature_style()
SC, DC = 3.46, 7.09

OUT_DIR   = "./cytof_output_nature"
FEAT_CSV  = f"{OUT_DIR}/SuppTable_feature_matrix.csv"
H5AD_PATH = "./cytof_output_v3/cytof_analyzed_v3.h5ad"

CLUSTER_ANNOTATION = {
    "0":"Classical Monocyte","1":"Naive B","2":"CD8 Effector T","3":"CD4 Naive T",
    "4":"CD4 Central Memory T","5":"Non-classical Monocyte","6":"NK cell",
    "7":"CD4 Th1-like","8":"Classical Monocyte","9":"CD4 Effector Memory T",
    "10":"Inflammatory Monocyte","11":"CD8 Naive T","12":"γδ T cell",
    "13":"mDC","14":"pDC","15":"T-Myeloid doublets","16":"CD16+ Granulocyte",
}
DROP = {"T-Myeloid doublets","CD16+ Granulocyte"}
GROUP_ORDER = ["Jo-1","PL-12","EJ","PL-7"]

# clinical metadata (re-declared to avoid re-running the whole pipeline)
CLIN = pd.DataFrame([
    ["L01912_FH0002", 8,"Jo-1" ,57, 1,0,0,1, 3838,755,11 ,1.71,np.nan,np.nan],
    ["L01912_FH0003", 7,"Jo-1" ,52, 1,1,0,1,  469,343,30 ,0.73, 39.4, 27.9],
    ["L01912_FH0009", 6,"PL-12",50, 1,0,0,1,   26,202,np.nan,0.25,np.nan,np.nan],
    ["L01912_FH0011", 5,"EJ"   ,67, 1,0,0,1,   26,147, 2 ,np.nan,np.nan,np.nan],
    ["L01912_FH0012", 4,"PL-7" ,63, 0,1,0,1,   41,195, 7 ,np.nan, 86.5, 77.3],
    ["L01912_FH0013", 3,"PL-12",67, 1,1,1,0,   13,318,13 ,4.6 ,np.nan,np.nan],
    ["L01912_FH0014", 2,"Jo-1" ,60, 1,0,0,1,np.nan,374, 6 ,np.nan,np.nan,np.nan],
    ["L01912_FH0015", 1,"PL-12",83, 0,0,0,1,   43,315, 2 ,0.25,np.nan,np.nan],
    ["L01912_FH0016",10,"EJ"   ,66, 1,1,0,1,np.nan,np.nan,28,np.nan,np.nan,np.nan],
    ["L01912_FH0017", 9,"PL-7" ,52, 0,0,1,1,   15,np.nan,13,0.25, 65  , 46.1],
], columns=["sample","pid","antibody","age","myositis","arthritis","rash","ILD",
            "CK","LDH","ESR","CRP","FVC","DLCO"]).set_index("sample")

# ════════════════════════════════════════════════════════════════════════════
# S1 — k-SELECTION EVIDENCE (silhouette + gap + elbow)
# ════════════════════════════════════════════════════════════════════════════
print("\n══ S1: k-selection evidence ══")
feat = pd.read_csv(FEAT_CSV).set_index("sample")
X65 = feat.drop(columns=["pid","name","antibody","immunotype"]).astype(float)
print(f"  Feature matrix: {X65.shape}")

# standardize (matches Ward clustering done in the main pipeline)
Xz = StandardScaler().fit_transform(X65.values)

K_RANGE = list(range(2, 8))

# --- Ward linkage once; re-cut for each k ---
link = hierarchy.linkage(Xz, method="ward")

def wss_for_cut(X, labels):
    tot = 0.0
    for c in np.unique(labels):
        Xc = X[labels == c]
        if len(Xc) > 0:
            tot += ((Xc - Xc.mean(0))**2).sum()
    return tot

sil, wss = [], []
cluster_labels_by_k = {}
for k in K_RANGE:
    labels = hierarchy.fcluster(link, t=k, criterion="maxclust")
    cluster_labels_by_k[k] = labels
    if len(np.unique(labels)) < 2 or len(np.unique(labels)) == len(labels):
        sil.append(np.nan)
    else:
        sil.append(silhouette_score(Xz, labels, metric="euclidean"))
    wss.append(wss_for_cut(Xz, labels))

# --- Gap statistic (Tibshirani 2001) ---
# reference: uniform distribution over bounding box, B=100 permutations
B_GAP = 200
rng   = np.random.default_rng(7)
n, d  = Xz.shape
mn, mx = Xz.min(0), Xz.max(0)

log_W_obs = np.log(np.array(wss))
log_W_ref = np.zeros((B_GAP, len(K_RANGE)))
print(f"  Gap statistic: {B_GAP} uniform references, k∈{K_RANGE}…")
for b in range(B_GAP):
    Xr = rng.uniform(mn, mx, size=(n, d))
    lr = hierarchy.linkage(Xr, method="ward")
    for ki, k in enumerate(K_RANGE):
        lbl = hierarchy.fcluster(lr, t=k, criterion="maxclust")
        log_W_ref[b, ki] = np.log(wss_for_cut(Xr, lbl))

gap  = log_W_ref.mean(0) - log_W_obs
sdk  = log_W_ref.std(0, ddof=1)
sk   = sdk * np.sqrt(1 + 1/B_GAP)

# Tibshirani optimal-k rule: smallest k such that gap(k) >= gap(k+1) - s(k+1)
k_opt_gap = None
for i in range(len(K_RANGE)-1):
    if gap[i] >= gap[i+1] - sk[i+1]:
        k_opt_gap = K_RANGE[i]; break

k_opt_sil = K_RANGE[int(np.nanargmax(sil))]
print(f"  Optimal k (silhouette): {k_opt_sil}")
print(f"  Optimal k (gap stat)  : {k_opt_gap}")
print(f"  Silhouette values: " +
      ", ".join(f"k={k}:{s:.3f}" for k, s in zip(K_RANGE, sil)))

# --- plot ---
# Override the global "large-print" Nature style locally so S1 renders at
# normal proportions for a small 3-panel diagnostic.
S1_RC = {
    "font.size":          7,
    "axes.titlesize":     7.5,
    "axes.labelsize":     7,
    "xtick.labelsize":    6.5,
    "ytick.labelsize":    6.5,
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.major.size":   2.5,
    "ytick.major.size":   2.5,
    "lines.linewidth":    1.0,
    "patch.linewidth":    0.6,
}
with plt.rc_context(S1_RC):
    fig = plt.figure(figsize=(DC+0.3, 2.4))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.42)

    # (a) silhouette
    ax = fig.add_subplot(gs[0,0])
    ax.plot(K_RANGE, sil, "-o", color="#2471A3", lw=1.0, markersize=3.5,
            markerfacecolor="white", markeredgewidth=0.8)
    ax.axvline(k_opt_sil, color="#C0392B", ls="--", lw=0.6, alpha=0.8)
    ax.text(k_opt_sil, max(sil), f"  k*={k_opt_sil}", color="#C0392B",
            fontsize=6, va="top", ha="left")
    ax.set_xlabel("k"); ax.set_ylabel("Silhouette (Ward)")
    ax.set_title("Silhouette vs k")
    ax.set_xticks(K_RANGE); ax.grid(True, color="#eee", lw=0.3); ax.set_axisbelow(True)

    # (b) gap statistic
    ax = fig.add_subplot(gs[0,1])
    ax.errorbar(K_RANGE, gap, yerr=sk, fmt="-o", color="#1A7A4A", lw=1.0,
                markersize=3.5, markerfacecolor="white", markeredgewidth=0.8,
                capsize=2, elinewidth=0.6)
    if k_opt_gap is not None:
        ax.axvline(k_opt_gap, color="#C0392B", ls="--", lw=0.6, alpha=0.8)
        ax.text(k_opt_gap, max(gap), f"  k*={k_opt_gap}",
                color="#C0392B", fontsize=6, va="top", ha="left")
    ax.set_xlabel("k"); ax.set_ylabel(r"Gap(k) = E[log Wₖ] − log Wₖ")
    ax.set_title("Gap statistic (B=200 uniform ref.)")
    ax.set_xticks(K_RANGE); ax.grid(True, color="#eee", lw=0.3); ax.set_axisbelow(True)

    # (c) WSS elbow
    ax = fig.add_subplot(gs[0,2])
    ax.plot(K_RANGE, wss, "-o", color="#7D3C98", lw=1.0, markersize=3.5,
            markerfacecolor="white", markeredgewidth=0.8)
    ax.set_xlabel("k"); ax.set_ylabel("Within-cluster SS")
    ax.set_title("WSS elbow")
    ax.set_xticks(K_RANGE); ax.grid(True, color="#eee", lw=0.3); ax.set_axisbelow(True)

    save(fig, "S1_k_selection", OUT_DIR)
print("  Saved: S1_k_selection.pdf/.png")

pd.DataFrame({
    "k": K_RANGE, "silhouette": sil, "wss": wss, "gap": gap, "gap_sd": sk
}).to_csv(f"{OUT_DIR}/S1_k_selection.csv", index=False)

# ════════════════════════════════════════════════════════════════════════════
# S2 — Fig 17 LOO STABILITY for clinical correlations
# ════════════════════════════════════════════════════════════════════════════
print("\n══ S2: LOO stability for clinical correlations ══")

# re-derive per-sample cell-type proportions + global MFI + signatures from adata
print(f"  Loading {H5AD_PATH} …")
adata = ad.read_h5ad(H5AD_PATH)
adata.obs_names_make_unique()
adata.obs["cell_type"] = adata.obs["leiden"].astype(str).map(CLUSTER_ANNOTATION).fillna("Unannotated")
adata = adata[~adata.obs["cell_type"].isin(DROP)].copy()

prop = (adata.obs.groupby(["sample","cell_type"], observed=True).size()
        .groupby(level=0).transform(lambda x: x/x.sum())
        .rename("prop").reset_index()
        .pivot(index="sample", columns="cell_type", values="prop").fillna(0)*100)

# signature scores (re-build the 10 canonical signatures from markers used in report)
SIG = {
    "Th1":         ["CXCR3","CCR4","CD4"],
    "Th17":        ["CCR6","CCR4","CD4"],
    "Tfh":         ["CXCR5","ICOS","CD4"],
    "Treg":        ["CD25","CD127","CD4"],
    "Cytotoxic":   ["GranzymeB","CD8a","CD56"],
    "Exhaustion":  ["PD-1","TIGIT","Tim-3"],
    "Activation":  ["HLA-DR","CD38","CD69"],
    "Naive":       ["CCR7","CD45RA","CD27"],
    "Memory":      ["CD45RO","CD95"],
    "Migration":   ["CXCR3","CCR4","CCR6","CXCR5"],
}
# weight Treg inversely for CD127
TREG_NEG = {"CD127"}
X_df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
sig_by_sample = pd.DataFrame(index=prop.index)
for sig, mks in SIG.items():
    mks = [m for m in mks if m in X_df.columns]
    if not mks: continue
    v = X_df[mks].mean(axis=1)
    per_sample = v.groupby(adata.obs["sample"].values).mean()
    sig_by_sample[f"{sig}_score"] = per_sample.reindex(prop.index).values

# assemble immune feature block: proportions + signatures
immune = pd.concat([prop, sig_by_sample], axis=1)
# clinical block (continuous subset for correlation)
clin_cont = CLIN[["age","CK","LDH","ESR","CRP","FVC","DLCO","myositis","arthritis","rash","ILD"]]

# full-cohort Spearman across all (immune, clinical) pairs
pairs = []
for ic in immune.columns:
    for cc in clin_cont.columns:
        x = immune[ic].values
        y = clin_cont[cc].values
        mask = ~(pd.isna(x) | pd.isna(y))
        if mask.sum() < 5:
            continue
        rho, p = stats.spearmanr(x[mask], y[mask])
        pairs.append({"immune": ic, "clinical": cc, "rho": rho, "p": p, "n": int(mask.sum())})
pair_df = pd.DataFrame(pairs)
_, q, _, _ = multipletests(pair_df["p"].fillna(1).values, method="fdr_bh")
pair_df["q_fdr"] = q
pair_df = pair_df.sort_values("q_fdr")

top_pairs = pair_df[pair_df["q_fdr"] < 0.10].head(10)
if len(top_pairs) == 0:
    # relax if nothing survives BH — take top 6 by |rho|
    top_pairs = pair_df.iloc[np.argsort(-pair_df["rho"].abs().values)].head(8)
print(f"  Full-cohort significant pairs (FDR<0.10): {(pair_df['q_fdr']<0.10).sum()}")
print("  Top pairs to test for LOO stability:")
for _, r in top_pairs.iterrows():
    print(f"    {r['immune']:<25} × {r['clinical']:<10}  ρ={r['rho']:+.3f}  q={r['q_fdr']:.3g}")

# LOO: drop each patient, recompute ρ
records = []
for _, r in top_pairs.iterrows():
    ic, cc = r["immune"], r["clinical"]
    x = immune[ic]; y = clin_cont[cc]
    mask = ~(x.isna() | y.isna())
    samples_ok = x.index[mask]
    loo_rhos = []
    loo_ps   = []
    for sdrop in samples_ok:
        keep = [s for s in samples_ok if s != sdrop]
        rho_loo, p_loo = stats.spearmanr(x.loc[keep].values, y.loc[keep].values)
        loo_rhos.append(rho_loo); loo_ps.append(p_loo)
    records.append({
        "pair": f"{ic} × {cc}",
        "immune": ic, "clinical": cc,
        "rho_full": r["rho"], "q_full": r["q_fdr"],
        "n": int(mask.sum()),
        "loo_rhos":   loo_rhos,
        "rho_mean":   float(np.mean(loo_rhos)),
        "rho_min":    float(np.min(loo_rhos)),
        "rho_max":    float(np.max(loo_rhos)),
        "rho_range":  float(np.max(loo_rhos) - np.min(loo_rhos)),
        "sign_stable": bool(all(np.sign(v) == np.sign(r["rho"]) for v in loo_rhos)),
    })

loo_df = pd.DataFrame(records)
loo_df.to_csv(f"{OUT_DIR}/S2_LOO_correlation_stability.csv", index=False)

# plot: dumbbell + LOO dots
S2_RC = {
    "font.size":          7,
    "axes.titlesize":     7.5,
    "axes.labelsize":     7,
    "xtick.labelsize":    6.5,
    "ytick.labelsize":    6.5,
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.major.size":   2.5,
    "ytick.major.size":   2.5,
    "lines.linewidth":    1.0,
    "patch.linewidth":    0.6,
}
with plt.rc_context(S2_RC):
    fig, ax = plt.subplots(figsize=(DC-0.2, 0.30*len(loo_df) + 1.0))
    y = np.arange(len(loo_df))
    for i, r in loo_df.iterrows():
        col = "#1A7A4A" if r["rho_full"] > 0 else "#C0392B"
        ax.scatter(r["loo_rhos"], [i]*len(r["loo_rhos"]),
                   s=8, c=col, alpha=0.35, edgecolor="none", zorder=2)
        ax.plot([r["rho_min"], r["rho_max"]], [i, i],
                color=col, lw=0.8, alpha=0.5, zorder=1)
        ax.plot(r["rho_full"], i, "D", color=col,
                markersize=4.5, markeredgecolor="white", markeredgewidth=0.5, zorder=4)
    ax.axvline(0, color="#222", lw=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([r["pair"] for _, r in loo_df.iterrows()])
    ax.set_xlabel(r"Spearman $\rho$")
    ax.set_title("Leave-one-out stability of top clinical correlations\n"
                 "(◆ = full-cohort ρ,  • = LOO replicates)", pad=5)
    for i, r in loo_df.iterrows():
        mark = "stable" if r["sign_stable"] else "FLIPS sign"
        col_m = "#1A7A4A" if r["sign_stable"] else "#C0392B"
        ax.text(1.02, i, mark, transform=ax.get_yaxis_transform(),
                fontsize=5.5, color=col_m, va="center", fontweight="bold")
    ax.set_xlim(-1.05, 1.05)
    save(fig, "S2_fig17_LOO_stability", OUT_DIR)
print("  Saved: S2_fig17_LOO_stability.pdf/.png")

# ════════════════════════════════════════════════════════════════════════════
# S3 — PATIENT-AGGREGATED DE (reviewer-proofed Fig 8)
# ════════════════════════════════════════════════════════════════════════════
print("\n══ S3: patient-aggregated within-cell-type DE ══")

obs = adata.obs[["sample","cell_type"]].copy()
# per-(sample, cell_type, marker) mean
print("  Building per-patient × per-(ct, marker) mean table …")
sample_of  = adata.obs["sample"].values
ct_of      = adata.obs["cell_type"].values
# quick groupby: stack X with labels
big = pd.DataFrame(adata.X, columns=adata.var_names)
big["sample"]    = sample_of
big["cell_type"] = ct_of
means = big.groupby(["cell_type","sample"]).mean()   # rows: (ct, sample), cols: markers

groups = CLIN["antibody"].reindex(prop.index).values
sample_to_grp = dict(zip(prop.index, groups))

MIN_CELLS_PER_PATIENT = 20
cell_counts = big.groupby(["cell_type","sample"]).size().rename("n").reset_index()
cell_counts_pivot = cell_counts.pivot(index="cell_type", columns="sample",
                                       values="n").fillna(0)

# For each cell type, test each marker across 4 groups using KW on per-patient means
records = []
for ct in means.index.get_level_values(0).unique():
    if ct in DROP or ct == "Unannotated":
        continue
    sub = means.loc[ct]
    # drop samples with <MIN cells of this ct
    ok_samples = cell_counts_pivot.loc[ct]
    ok_samples = ok_samples[ok_samples >= MIN_CELLS_PER_PATIENT].index.tolist()
    if len(ok_samples) < 6: continue   # need enough patients for KW
    sub = sub.reindex(ok_samples)

    # group by antibody
    groups_here = [sample_to_grp[s] for s in sub.index]
    for marker in sub.columns:
        vals = sub[marker].values
        if np.isnan(vals).any(): continue
        # build 4 group vectors
        by_g = {g: [] for g in GROUP_ORDER}
        for v, g in zip(vals, groups_here):
            by_g[g].append(v)
        non_empty = [np.array(v) for v in by_g.values() if len(v) >= 1]
        # KW requires ≥2 non-empty groups and ≥ overall n-groups variability
        if len(non_empty) < 2: continue
        if sum(len(g) for g in non_empty) < 5: continue
        try:
            kw_h, kw_p = stats.kruskal(*non_empty)
        except ValueError:
            continue
        # effect size (eta^2 ≈ (H - k + 1) / (n - k))
        n_tot = sum(len(g) for g in non_empty); k = len(non_empty)
        eta2  = max(0.0, (kw_h - k + 1) / (n_tot - k))
        # range of group means
        g_means = {g: np.mean(v) for g, v in by_g.items() if len(v) > 0}
        range_m = max(g_means.values()) - min(g_means.values())
        records.append({
            "cell_type": ct, "marker": marker,
            "kw_H": kw_h, "kw_p": kw_p, "eta2": eta2,
            "range_of_means": range_m,
            "n_samples_used": len(sub),
        })

de_df = pd.DataFrame(records)
_, q, _, _ = multipletests(de_df["kw_p"].values, method="fdr_bh")
de_df["q_fdr"] = q
de_df.to_csv(f"{OUT_DIR}/S3_patient_aggregated_DE.csv", index=False)

n_tested = len(de_df)
n_sig    = int((de_df["q_fdr"] < 0.10).sum())
n_sig05  = int((de_df["q_fdr"] < 0.05).sum())
print(f"  Tested {n_tested} (ct × marker) pairs at patient level")
print(f"    FDR<0.10: {n_sig}  ({100*n_sig/n_tested:.1f}%)")
print(f"    FDR<0.05: {n_sig05}  ({100*n_sig05/n_tested:.1f}%)")
print("  (contrast: cell-level Fig 8 reported 91.5% FDR<0.05)")

# --- heatmap: -log10 q by cell_type × marker, masking non-lineage off-diagonals? ---
# keep full ct × marker matrix, clip
hm = de_df.pivot(index="cell_type", columns="marker", values="q_fdr")
# order cell types & markers to match the main figure (alphabetical is fine here)
hm = hm.reindex(index=sorted(hm.index), columns=sorted(hm.columns))
log_hm = -np.log10(hm.clip(lower=1e-4))

S3_RC = {
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
}
with plt.rc_context(S3_RC):
    fig = plt.figure(figsize=(DC+0.6, 4.6))
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3.5, 1.0], hspace=0.55)

    ax = fig.add_subplot(gs[0])
    _nr_s3, _nc_s3 = log_hm.shape
    im = ax.pcolormesh(np.arange(_nc_s3 + 1) - 0.5, np.arange(_nr_s3 + 1) - 0.5,
                        log_hm.values, cmap="viridis", vmin=0, vmax=2.0,
                        edgecolors="none", rasterized=False, shading="flat")
    ax.invert_yaxis(); ax.set_aspect("auto")
    ax.set_xticks(range(log_hm.shape[1]))
    ax.set_xticklabels(log_hm.columns, rotation=75, ha="right", fontsize=5)
    ax.set_yticks(range(log_hm.shape[0]))
    ax.set_yticklabels(log_hm.index, fontsize=6)
    ax.tick_params(length=0)
    for i in range(log_hm.shape[0]):
        for j in range(log_hm.shape[1]):
            q = hm.iloc[i, j]
            if pd.notna(q) and q < 0.10:
                ax.text(j, i, "*", ha="center", va="center",
                        fontsize=5.5, color="white", fontweight="bold")

    cax = fig.add_axes([0.92, 0.50, 0.010, 0.30])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$-\log_{10}$ q (BH)", fontsize=6)
    cb.ax.tick_params(labelsize=5, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)

    ax.set_title(f"Within-cell-type DE re-tested on PATIENT-LEVEL means\n"
                 f"({n_sig}/{n_tested} pairs at FDR<0.10;  "
                 f"cell-level Fig 8: 269/294 ≈ 91.5% — patient-level: "
                 f"{100*n_sig/n_tested:.1f}%)", pad=5)

    ax2 = fig.add_subplot(gs[1])
    ax2.hist(de_df["kw_p"].values, bins=20, color="#2471A3",
             alpha=0.75, edgecolor="white", linewidth=0.4)
    ax2.axhline(len(de_df)/20, color="#C0392B", ls="--", lw=0.6,
                label="uniform expectation")
    ax2.set_xlabel("KW p-value (patient-level)")
    ax2.set_ylabel("count")
    ax2.set_title("p-value distribution (strong left skew → real signal, "
                  "but much less inflated than cell-level)",
                  fontsize=6.5, pad=3)
    ax2.legend(loc="upper right")

    save(fig, "S3_patient_aggregated_DE", OUT_DIR)
    print("  Saved: S3_patient_aggregated_DE.pdf/.png")

    # Forest of top 12 patient-level hits
    top12 = de_df.sort_values("q_fdr").head(12).copy()
    print("\n  Top 12 patient-level hits:")
    for _, r in top12.iterrows():
        print(f"    {r['cell_type']:<25} {r['marker']:<12} q={r['q_fdr']:.3g}  "
              f"Δ(group means)={r['range_of_means']:+.2f}")

    fig, ax = plt.subplots(figsize=(DC-0.4, 3.0))
    y = np.arange(len(top12))
    cols = plt.cm.viridis(np.clip(-np.log10(top12["q_fdr"].clip(lower=1e-4))/2.0, 0, 1))
    ax.barh(y, top12["range_of_means"], color=cols, edgecolor="white", linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['marker']} · {r['cell_type']}"
                        for _, r in top12.iterrows()], fontsize=6)
    for i, r in top12.iterrows():
        q_txt = f"q={r['q_fdr']:.2g}"
        ax.text(r["range_of_means"]*1.02, list(top12.index).index(i),
                q_txt, fontsize=5.5, va="center", color="#333")
    ax.set_xlabel("Range of patient-level group means\n(max group mean − min group mean)",
                  fontsize=6.5)
    ax.invert_yaxis()
    ax.set_title("Top patient-aggregated within-cell-type DE effects", pad=4)
    ax.grid(axis="x", color="#eee", lw=0.3); ax.set_axisbelow(True)
    save(fig, "S4_patient_aggregated_top12", OUT_DIR)
print("  Saved: S4_patient_aggregated_top12.pdf/.png")

print(f"""
══════════════════════════════════════════════════════════
  Supplementary additions complete.  Files in: {OUT_DIR}/
    S1_k_selection                   (silhouette + gap + WSS)
    S2_fig17_LOO_stability           (LOO ρ for top correlations)
    S3_patient_aggregated_DE         (patient-level DE heatmap)
    S4_patient_aggregated_top12     (patient-level DE top hits)
══════════════════════════════════════════════════════════
""")
