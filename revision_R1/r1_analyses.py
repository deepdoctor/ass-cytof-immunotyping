"""
Revision R1 (Frontiers in Immunology, manuscript 1954309) -- additional analyses
requested by Reviewer 4.

  A. Comment 6  -- provenance of the per-patient marker intensities: were the 42
     global marker means (and the ten composites built from them) computed on
     Harmony-corrected values?  Recompute them from the stored expression matrix
     and from plain arcsinh(x/5) values, re-derive the axis, and compare.
  B. Comment 4  -- "serology-orthogonality": effect size of autoantibody class on
     the axis with a confidence interval, an exact permutation test, and the power
     of the n = 3/3/2/2 Kruskal-Wallis test, so the claim can be stated with its
     power limitation.
  C. Minor 7    -- detection of the 15-gene IT3 transcriptomic signature in GSE220915.

Inputs (local only; patient-level FCS/h5ad are not distributed):
  cytof_output_v3/cytof_analyzed_v3.h5ad
  cytof_output_nature/SuppTable_feature_matrix.csv   (= Supp. Table ST1)
  validation/counts.tsv.gz, validation/ensg2sym.csv  (GSE220915)

Run from the repository root:
  python revision_R1/r1_analyses.py
"""
import itertools
import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.optimize import brentq

warnings.filterwarnings("ignore")
OUT = "revision_R1"
RNG = np.random.default_rng(20261003)
COFACTOR = 5.0

feat = pd.read_csv("cytof_output_nature/SuppTable_feature_matrix.csv")
FEATS = list(feat.columns[5:])
assert len(FEATS) == 66, len(FEATS)
PROPS = FEATS[:14]
MFI = [c for c in FEATS if c.startswith("MFI_")]
COMP = [c for c in FEATS if c.endswith("_score")]
assert (len(PROPS), len(MFI), len(COMP)) == (14, 42, 10)

SIGNATURES = {            # as in cytof_advanced_analysis.py
    "Th1": ["CXCR3", "CCR6"], "Th17": ["CCR6", "CCR4"],
    "Tfh": ["CXCR5", "ICOS", "PD-1"], "Treg": ["CD25", "CD127"],
    "Cytotoxic": ["GranzymeB"], "Exhaustion": ["PD-1", "TIGIT", "Tim-3", "CD39"],
    "Activation": ["CD38", "HLA-DR", "CD69", "Ki-67"],
    "Naive": ["CCR7", "CD45RA", "CD27", "TCF1"], "Memory": ["CD45RO", "CD95"],
    "Migration": ["CCR4", "CXCR3", "CCR6"],
}
INNATE = ["Classical Monocyte", "Inflammatory Monocyte", "Non-classical Monocyte",
          "NK cell", "mDC", "pDC", "Activation_score", "Cytotoxic_score"]


def axis_pc1(X):
    """PC1 of the z-scored patient x feature matrix, oriented innate-high."""
    Z = (X - X.mean()) / X.std(ddof=1)
    Z = Z.loc[:, Z.notna().all()]
    Zc = Z.values - Z.values.mean(0)
    U, s, Vt = np.linalg.svd(Zc, full_matrices=False)
    pc1 = U[:, 0] * s[0]
    load = pd.Series(Vt[0], index=Z.columns)
    if load[[c for c in INNATE if c in load.index]].mean() < 0:
        pc1, load = -pc1, -load
    return pd.Series(pc1, index=X.index), (s ** 2 / (s ** 2).sum())[0], load


report = {}
pid = "P" + feat["pid"].astype(int).astype(str).str.zfill(2)
X0 = feat[FEATS].set_axis(pid)
ab = pd.Series(feat["antibody"].values, index=pid)
region = pd.Series(feat["immunotype"].values, index=pid)
axis0, var0, load0 = axis_pc1(X0)
report["primary_axis"] = {
    "PC1_var_pct": round(100 * var0, 1),
    "scores": axis0.round(2).sort_values(ascending=False).to_dict(),
}

# ════════════════════════════════════════════════════════════════════════════
# A. Harmony provenance of the marker-intensity features
# ════════════════════════════════════════════════════════════════════════════
import anndata as ad

adata = ad.read_h5ad("cytof_output_v3/cytof_analyzed_v3.h5ad")
ANNOT_DROP = {"15", "16"}          # T-myeloid doublets, CD16+ granulocytes
keep = ~adata.obs["leiden"].astype(str).isin(ANNOT_DROP).values
sample = adata.obs["sample"].astype(str).values[keep]
X_stored = np.asarray(adata.X)[keep]                         # per-marker standardised arcsinh
X_arcsinh = np.arcsinh(np.asarray(adata.layers["raw"])[keep] / COFACTOR)   # plain arcsinh
markers = list(adata.var_names)
report["harmony"] = {
    "n_cells_h5ad": int(adata.n_obs), "n_cells_after_artefact_drop": int(keep.sum()),
    "harmony_input": "first %d PCs of the standardised 42-marker matrix" % adata.obsm["X_pca"].shape[1],
    "harmony_output_dims": int(adata.obsm["X_pca_harmony"].shape[1]),
    "expression_matrix_is_harmony_corrected": False,
}
order = feat["sample"].values


def patient_features(Xcell):
    g = pd.DataFrame(Xcell, columns=markers).assign(sample=sample).groupby("sample").mean().loc[order]
    out = pd.DataFrame(index=g.index)
    for m in markers:
        out["MFI_" + m] = g[m].values
    for name, mks in SIGNATURES.items():
        out[name + "_score"] = (g["CD25"] - g["CD127"]) if name == "Treg" else g[mks].mean(axis=1)
    return out


F_stored = patient_features(X_stored)
F_arcsinh = patient_features(X_arcsinh)
# do the deposited ST1 marker features equal the stored (non-Harmony) expression means?
dev = (F_stored[MFI + COMP].values - feat[MFI + COMP].values)
report["harmony"]["max_abs_dev_ST1_vs_recomputed_stored"] = float(np.abs(dev).max())

X_arc = X0.copy()
X_arc[MFI + COMP] = F_arcsinh[MFI + COMP].values
axis_arc, var_arc, _ = axis_pc1(X_arc)
r_p = float(np.corrcoef(axis0, axis_arc)[0, 1])
r_s = float(sps.spearmanr(axis0, axis_arc).statistic)
H_arc, p_arc = sps.kruskal(*[axis_arc[ab == a].values for a in ab.unique()])
jo1 = axis_arc[ab == "Jo-1"]
report["harmony"]["unscaled_arcsinh_axis"] = {
    "PC1_var_pct": round(100 * var_arc, 1),
    "pearson_r_vs_primary": round(r_p, 4), "spearman_rho_vs_primary": round(r_s, 4),
    "identical_rank_order": bool((axis0.rank() == axis_arc.rank()).all()),
    "rank_order": list(axis_arc.sort_values(ascending=False).index),
    "primary_rank_order": list(axis0.sort_values(ascending=False).index),
    "serology_KW_H": round(float(H_arc), 2), "serology_KW_P": round(float(p_arc), 2),
    "antiJo1_span_pct": round(100 * (jo1.max() - jo1.min()) / (axis_arc.max() - axis_arc.min())),
}
# per-feature agreement of the two marker-feature versions across patients
per_feat_r = [np.corrcoef(F_stored[c], F_arcsinh[c])[0, 1] for c in MFI + COMP]
report["harmony"]["feature_r_stored_vs_arcsinh_min"] = round(float(np.nanmin(per_feat_r)), 4)
report["harmony"]["feature_r_stored_vs_arcsinh_median"] = round(float(np.nanmedian(per_feat_r)), 4)
del adata, X_stored, X_arcsinh

# ════════════════════════════════════════════════════════════════════════════
# B. Serology: effect size, CI, exact permutation and power
# ════════════════════════════════════════════════════════════════════════════
y = axis0.values
lab = ab.values
classes = ["Jo-1", "PL-12", "EJ", "PL-7"]
N, k = len(y), len(classes)


def eta2(y, lab):
    m = y.mean()
    ssb = sum((lab == c).sum() * (y[lab == c].mean() - m) ** 2 for c in np.unique(lab))
    return ssb / ((y - m) ** 2).sum()


e2 = eta2(y, lab)
H, pKW = sps.kruskal(*[y[lab == c] for c in classes])
eps2 = H / (N - 1)                                   # rank-based epsilon-squared
F = (e2 / (k - 1)) / ((1 - e2) / (N - k))
df1, df2 = k - 1, N - k


def nc_bound(q):
    """noncentrality lambda such that P(F' <= F_obs | lambda) = q"""
    f = lambda lam: sps.ncf.cdf(F, df1, df2, lam) - q
    if f(0) < 0:            # even lambda=0 is too large
        return 0.0
    return brentq(f, 0, 500)


def ci_eta2(level):
    a = (1 - level) / 2
    lo, hi = nc_bound(1 - a), nc_bound(a)
    return lo / (lo + N), hi / (hi + N)


ci95, ci90 = ci_eta2(0.95), ci_eta2(0.90)

# exact permutation over every distinct assignment of the 3/3/2/2 class labels
idx = np.arange(N)
perm_e2 = []
for jo in itertools.combinations(idx, 3):
    r1 = [i for i in idx if i not in jo]
    for pl12 in itertools.combinations(r1, 3):
        r2 = [i for i in r1 if i not in pl12]
        for ej in itertools.combinations(r2, 2):
            L = np.empty(N, dtype=object)
            L[list(jo)], L[list(pl12)], L[list(ej)] = "Jo-1", "PL-12", "EJ"
            L[[i for i in r2 if i not in ej]] = "PL-7"
            perm_e2.append(eta2(y, L))
perm_e2 = np.array(perm_e2)
p_perm = float((perm_e2 >= e2 - 1e-12).mean())
pct_rank = float((perm_e2 < e2).mean())

# Within-class share: how much of the axis range lies within single classes
rng_all = y.max() - y.min()
within_range = {c: round(100 * (y[lab == c].max() - y[lab == c].min()) / rng_all) for c in classes}

# Power of the KW test at n = 3/3/2/2 as a function of the true eta^2
ns = [3, 3, 2, 2]


def power_kw(eta2_true, pattern, B=4000):
    # pattern: unit-free group means; scaled so the population eta^2 equals eta2_true
    mu = np.array(pattern, float)
    w = np.array(ns) / sum(ns)
    mu = mu - (w * mu).sum()
    var_b = (w * mu ** 2).sum()
    if eta2_true == 0:
        mu = np.zeros_like(mu)
    else:
        mu = mu * np.sqrt(eta2_true / (1 - eta2_true) / var_b)
    hits = 0
    for _ in range(B):
        groups = [RNG.normal(mu[i], 1.0, ns[i]) for i in range(k)]
        hits += sps.kruskal(*groups).pvalue < 0.05
    return hits / B


patterns = {"one class shifted (Jo-1 vs rest)": [1, 0, 0, 0],
            "equally spaced class means": [3, 2, 1, 0]}
grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
power = {name: {g: power_kw(g, pat) for g in grid} for name, pat in patterns.items()}


def mde(curve, target=0.8):
    for g in grid:
        if curve[g] >= target:
            return g
    return None


# eta^2 of the regions along the axis, for scale (expected to be high by construction)
e2_region = eta2(y, region.values)

report["serology"] = {
    "KW_H": round(float(H), 2), "KW_P": round(float(pKW), 2),
    "eta2": round(float(e2), 3), "epsilon2_rank": round(float(eps2), 3),
    "eta2_CI95": [round(ci95[0], 3), round(ci95[1], 3)],
    "eta2_CI90": [round(ci90[0], 3), round(ci90[1], 3)],
    "exact_permutation_P": round(p_perm, 3), "n_permutations": int(len(perm_e2)),
    "observed_eta2_percentile_of_null": round(100 * pct_rank, 1),
    "within_class_range_pct_of_axis": within_range,
    "KW_power_by_true_eta2": {n: {str(g): round(v, 3) for g, v in c.items()} for n, c in power.items()},
    "min_eta2_for_80pct_power": {n: mde(c) for n, c in power.items()},
    "eta2_regions_on_axis": round(float(e2_region), 3),
}

# ════════════════════════════════════════════════════════════════════════════
# C. IT3 transcriptomic signature detection in GSE220915
# ════════════════════════════════════════════════════════════════════════════
IT3_GENES = {"CCR7", "SELL", "LEF1", "TCF7", "CD27", "IL7R", "FOXP3", "CTLA4", "KLRG1",
             "MS4A1", "CD19", "CD79A", "IGHM", "IGHD", "CD4"}
cnt = pd.read_csv("validation/counts.tsv.gz", sep="\t", compression="gzip", index_col=0)
cnt.index = cnt.index.str.split(".").str[0]
emap = pd.read_csv("validation/ensg2sym.csv")
sym = cnt.index.map(dict(zip(emap.iloc[:, 0], emap.iloc[:, 1])))
detected = sorted(set(sym.dropna()) & IT3_GENES)
report["IT3_transcriptomic"] = {"n_genes": len(IT3_GENES), "detected_GSE220915": len(detected),
                                "missing": sorted(IT3_GENES - set(detected))}

with open(f"{OUT}/r1_analyses_results.json", "w") as fh:
    json.dump(report, fh, indent=2, ensure_ascii=False, default=float)
print(json.dumps(report, indent=2, ensure_ascii=False, default=float))

# ════════════════════════════════════════════════════════════════════════════
# Supplementary Tables ST8e / ST8f
# ════════════════════════════════════════════════════════════════════════════
st8e = pd.DataFrame({
    "patient": axis0.index,
    "autoantibody": ab.values,
    "region": region.values,
    "axis_primary (ST1 features)": axis0.values.round(3),
    "axis_unstandardised_arcsinh": axis_arc.values.round(3),
    "rank_primary": axis0.rank(ascending=False).astype(int).values,
    "rank_unstandardised_arcsinh": axis_arc.rank(ascending=False).astype(int).values,
}).sort_values("rank_primary")
st8e_summary = pd.DataFrame([
    ("Harmony input", report["harmony"]["harmony_input"]),
    ("Harmony-corrected quantity", "30-dimensional PCA embedding only (neighbour graph, Leiden, UMAP, T-cell pseudotime)"),
    ("Expression matrix used for marker features", "arcsinh(x/5), standardised per marker across cells; not batch-corrected"),
    ("Max |ST1 - recomputed| over 52 marker features x 10 patients", f"{report['harmony']['max_abs_dev_ST1_vs_recomputed_stored']:.1e}"),
    ("PC1 variance, primary / unstandardised arcsinh (%)", f"{report['primary_axis']['PC1_var_pct']} / {report['harmony']['unscaled_arcsinh_axis']['PC1_var_pct']}"),
    ("Pearson r, axis scores", report["harmony"]["unscaled_arcsinh_axis"]["pearson_r_vs_primary"]),
    ("Spearman rho, axis scores", report["harmony"]["unscaled_arcsinh_axis"]["spearman_rho_vs_primary"]),
    ("Identical ranking of all ten patients", report["harmony"]["unscaled_arcsinh_axis"]["identical_rank_order"]),
    ("Serology Kruskal-Wallis H / P (unstandardised)", f"{report['harmony']['unscaled_arcsinh_axis']['serology_KW_H']} / {report['harmony']['unscaled_arcsinh_axis']['serology_KW_P']}"),
], columns=["item", "value"])
s = report["serology"]
st8f_summary = pd.DataFrame([
    ("Kruskal-Wallis H (df=3)", s["KW_H"]), ("Kruskal-Wallis P", s["KW_P"]),
    ("Exact permutation P (eta^2; 25,200 label assignments)", s["exact_permutation_P"]),
    ("eta^2 (proportion of axis variance explained by class)", s["eta2"]),
    ("eta^2 95% CI (noncentral F inversion)", f"{s['eta2_CI95'][0]}-{s['eta2_CI95'][1]}"),
    ("eta^2 90% CI", f"{s['eta2_CI90'][0]}-{s['eta2_CI90'][1]}"),
    ("epsilon^2 (rank-based, H/(n-1))", s["epsilon2_rank"]),
    ("Observed eta^2 percentile of permutation null (%)", s["observed_eta2_percentile_of_null"]),
    ("For scale: eta^2 of regions IT1-IT3 on the axis (by construction)", s["eta2_regions_on_axis"]),
] + [(f"Within-class range, {c} (% of full axis range)", v) for c, v in s["within_class_range_pct_of_axis"].items()],
    columns=["quantity", "value"])
pw = pd.DataFrame(s["KW_power_by_true_eta2"]).rename_axis("true eta^2").reset_index()
pw.insert(0, "note", "")
pw.loc[0, "note"] = ("Simulated Kruskal-Wallis power (alpha=0.05; n=3/3/2/2; normal errors; 4,000 replicates per "
                     "cell). The asymptotic test is conservative at these group sizes.")
with pd.ExcelWriter(f"{OUT}/Supp_Table_ST8e-ST8g.xlsx") as xw:
    st8e_summary.to_excel(xw, "ST8e_summary", index=False)
    st8e.to_excel(xw, "ST8e_axis_unstandardised", index=False)
    st8f_summary.to_excel(xw, "ST8f_serology_effect", index=False)
    pw.to_excel(xw, "ST8f_KW_power", index=False)
print(f"wrote {OUT}/Supp_Table_ST8e-ST8g.xlsx")

# ════════════════════════════════════════════════════════════════════════════
# D. Fig. 6 re-check -- T-cell pseudotime by region (per-cell output of
#    cytof_mainfig7_trajectory.py, i.e. the data plotted in Fig. 6)
# ════════════════════════════════════════════════════════════════════════════
pt = pd.read_csv("cytof_output_nature/29b_T_pseudotime_per_cell.csv")
pid_of = dict(zip(feat["sample"], "P" + feat["pid"].astype(int).astype(str).str.zfill(2)))
pt["patient"] = pt["sample"].map(pid_of)
pt["quintile"] = pd.qcut(pt["dpt"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
REG = ["IT1", "IT2", "IT3"]
cell_med = pt.groupby("immunotype")["dpt"].median()
H_cell = sps.kruskal(*[pt.loc[pt.immunotype == r, "dpt"] for r in REG]).statistic
pmed = pt.groupby(["patient", "immunotype"])["dpt"].median().reset_index()
pmed["axis"] = pmed["patient"].map(axis0)
H_pat, P_pat = sps.kruskal(*[pmed.loc[pmed.immunotype == r, "dpt"] for r in REG])
rho_ax, p_ax = sps.spearmanr(pmed["axis"], pmed["dpt"])
eff = pt["cell_type"].isin(["CD8 Effector T", "CD4 Th1-like"])
d_rep = {
    "n_T_cells": int(len(pt)), "cell_types": sorted(pt["cell_type"].unique()),
    "cell_level_median_dpt": {r: round(float(cell_med[r]), 4) for r in REG},
    "cell_level_KW_H": round(float(H_cell), 1),
    "share_in_Q5": {r: round(float((pt.loc[pt.immunotype == r, "quintile"] == "Q5").mean()), 4) for r in REG},
    "share_CD8eff_plus_Th1": {r: round(float(eff[pt.immunotype == r].mean()), 4) for r in REG},
    "median_dpt_within_CD8_effector": {r: round(float(pt.loc[(pt.immunotype == r) & (pt.cell_type == "CD8 Effector T"), "dpt"].median()), 4) for r in REG},
    "patient_level_median_of_medians": {r: round(float(pmed.loc[pmed.immunotype == r, "dpt"].median()), 4) for r in REG},
    "patient_level_KW_H": round(float(H_pat), 2), "patient_level_KW_P": round(float(P_pat), 3),
    "spearman_axis_vs_patient_median_dpt": [round(float(rho_ax), 3), round(float(p_ax), 3)],
}
report["pseudotime_fig6"] = d_rep
with open(f"{OUT}/r1_analyses_results.json", "w") as fh:
    json.dump(report, fh, indent=2, ensure_ascii=False, default=float)
st8g = pmed.rename(columns={"immunotype": "region", "dpt": "median_DPT"}).sort_values("axis", ascending=False)
st8g["median_DPT"] = st8g["median_DPT"].round(4); st8g["axis"] = st8g["axis"].round(3)
st8g_sum = pd.DataFrame([(k, json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
                         for k, v in d_rep.items()], columns=["quantity", "value"])
with pd.ExcelWriter(f"{OUT}/Supp_Table_ST8e-ST8g.xlsx", mode="a", engine="openpyxl") as xw:
    st8g_sum.to_excel(xw, "ST8g_pseudotime_summary", index=False)
    st8g.to_excel(xw, "ST8g_pseudotime_patient", index=False)
print(json.dumps(d_rep, indent=1))
