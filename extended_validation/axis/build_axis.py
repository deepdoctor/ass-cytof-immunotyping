"""Recast the immunotype partition as a continuous innate-hyperactive axis.

Input is the deposited per-patient feature matrix (Supp. Table ST1) and the
clinical table (ST2) -- no raw FCS data required, so this is fully reproducible
from the supplementary workbook alone.

The axis is PC1 of the z-scored patient x feature matrix: entirely unsupervised,
computed without reference to the immunotype labels, so using it to order
patients is not circular.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats as sps

RNG = np.random.default_rng(20260727)
S = "/Users/yc4229/Downloads/Journal of Autoimmunity/Supplementary Tables.xlsx"

st1 = pd.read_excel(S, "ST1_feature_matrix")
st2 = pd.read_excel(S, "ST2_immunotype_clinical")
EXCLUDE = ["Tfh_score", "Memory_score", "Migration_score"]   # marker sets not documented
FEAT = [c for c in st1.columns[5:] if c not in EXCLUDE]
X = st1[FEAT].astype(float)
Z = (X - X.mean()) / X.std(ddof=1)
Z = Z.loc[:, Z.notna().all()]                      # drop invariant/NaN features
print(f"feature matrix: {Z.shape[0]} patients x {Z.shape[1]} features")

# ---------------------------------------------------------------- PCA
Zc = Z.values - Z.values.mean(0)
U, s, Vt = np.linalg.svd(Zc, full_matrices=False)
var = s**2 / (s**2).sum()
pc = U * s
axis = pd.Series(pc[:, 0], index=st1["name"], name="axis")

# Orient so that higher = more innate-hyperactive: the axis must load positively
# on the innate compartments and activation programmes.
innate = [c for c in Z.columns if c in
          ("Classical Monocyte", "Inflammatory Monocyte", "Non-classical Monocyte",
           "NK cell", "mDC", "pDC")] + \
         [c for c in Z.columns if c in ("Activation_score", "Cytotoxic_score")]
load = pd.Series(Vt[0], index=Z.columns)
if load[innate].mean() < 0:
    axis, load = -axis, -load
    pc[:, 0] = -pc[:, 0]

print(f"PC1 variance explained: {var[0]*100:.1f}%   (PC2 {var[1]*100:.1f}%)")
print("\npatients ordered along the innate-hyperactive axis:")
tab = pd.DataFrame({"axis": axis.round(2),
                    "immunotype": st1["immunotype"].values,
                    "antibody": st1["antibody"].values}).sort_values("axis", ascending=False)
print(tab.to_string())

print("\ntop +10 loadings:"); print(load.sort_values(ascending=False).head(10).round(2).to_string())
print("\ntop -10 loadings:"); print(load.sort_values().head(10).round(2).to_string())

# --------------------------------------------- does the axis recover IT1>IT2>IT3?
it = pd.Series(st1["immunotype"].values, index=st1["name"])
grp = {g: axis[it[it == g].index].values for g in ("IT1", "IT2", "IT3")}
print("\naxis by immunotype (mean +/- sd):")
for g in ("IT1", "IT2", "IT3"):
    print(f"  {g}: {grp[g].mean():+.2f} +/- {grp[g].std(ddof=1):.2f}   n={len(grp[g])}")
H, pH = sps.kruskal(*grp.values())
print(f"  Kruskal-Wallis across the three immunotypes: H={H:.2f}, P={pH:.4f}")
rho_it, p_it = sps.spearmanr(axis.values, it.map({"IT1": 3, "IT2": 2, "IT3": 1}).values)
print(f"  Spearman(axis, immunotype rank) = {rho_it:+.3f}, P={p_it:.2e}")

# --------------------------------------------- orthogonality to serology
ab = pd.Series(st1["antibody"].values, index=st1["name"])
gs = [axis[ab[ab == a].index].values for a in ab.unique()]
Hs, ps = sps.kruskal(*gs)
print(f"\naxis by autoantibody class: Kruskal-Wallis H={Hs:.2f}, P={ps:.3f} "
      f"({'ns -> orthogonal' if ps > .05 else 'SIGNIFICANT'})")
jo1 = axis[ab[ab == "Jo-1"].index]
print(f"  the three anti-Jo-1 patients span {jo1.min():+.2f} to {jo1.max():+.2f} "
      f"({(jo1.max()-jo1.min())/(axis.max()-axis.min())*100:.0f}% of the full axis range)")

# --------------------------------------------- continuous clinical correlation
cl = st2.set_index("name")
CLIN = ["CK", "LDH", "ESR", "CRP", "FVC", "DLCO", "age", "DA_score"]
rows = []
for v in CLIN:
    y = cl[v].reindex(axis.index).astype(float)
    x = axis.reindex(y.index)
    ok = y.notna() & x.notna()
    if ok.sum() < 5:
        continue
    yy = np.log10(y[ok]) if v in ("CK", "LDH", "CRP") else y[ok]
    r, p = sps.spearmanr(x[ok], yy)
    # leave-one-out stability of the correlation
    loo = []
    idx = list(x[ok].index)
    for drop in idx:
        k = [i for i in idx if i != drop]
        loo.append(sps.spearmanr(x[k], yy[k]).statistic)
    loo = np.array(loo)
    rows.append(dict(variable=("log10 " + v if v in ("CK", "LDH", "CRP") else v),
                     n=int(ok.sum()), rho=r, p=p,
                     loo_min=loo.min(), loo_max=loo.max(),
                     sign_stable=bool(np.all(np.sign(loo) == np.sign(r)))))
res = pd.DataFrame(rows).sort_values("p")
print("\n=== continuous axis vs clinical variables (Spearman) ===")
print(res.to_string(index=False, float_format=lambda z: f"{z:.3f}"))

# --------------------------------------------- power comparison vs 3-group test
print("\n=== same variables, 3-group Kruskal-Wallis on the immunotype labels ===")
rows2 = []
for v in CLIN:
    y = cl[v].reindex(axis.index).astype(float)
    yy = np.log10(y) if v in ("CK", "LDH", "CRP") else y
    gg = [yy[it[it == g].index].dropna().values for g in ("IT1", "IT2", "IT3")]
    if min(len(g) for g in gg) < 2:
        continue
    H, p = sps.kruskal(*gg)
    rows2.append(dict(variable=v, H=H, p=p))
print(pd.DataFrame(rows2).to_string(index=False, float_format=lambda z: f"{z:.3f}"))

# --------------------------------------------- bootstrap stability of the axis
B = 2000
ranks = np.zeros((B, len(axis)))
for b in range(B):
    idx = RNG.choice(len(axis), len(axis), replace=True)
    Xb = Z.values[idx]
    Xb = Xb - Xb.mean(0)
    sd = Xb.std(0, ddof=1); sd[sd == 0] = 1
    _, _, Vtb = np.linalg.svd(Xb / sd, full_matrices=False)
    proj = (Z.values - Z.values.mean(0)) @ Vtb[0]
    if np.corrcoef(proj, axis.values)[0, 1] < 0:
        proj = -proj
    ranks[b] = sps.rankdata(proj)
rank_sd = ranks.std(0)
print("\n=== bootstrap stability of each patient's rank on the axis (B=2,000) ===")
out = pd.DataFrame({"patient": axis.index, "axis": axis.values.round(2),
                    "mean_rank": ranks.mean(0).round(1), "rank_sd": rank_sd.round(2),
                    "immunotype": it.values}).sort_values("axis", ascending=False)
print(out.to_string(index=False))
print(f"\nmean rank SD across patients: {rank_sd.mean():.2f} positions "
      f"(a random ordering would give ~{np.sqrt((len(axis)**2-1)/12):.2f})")

pd.DataFrame({"patient": axis.index, "axis_PC1": axis.values,
              "immunotype": it.values, "antibody": ab.values,
              "boot_mean_rank": ranks.mean(0), "boot_rank_sd": rank_sd}
             ).to_csv("axis_scores.csv", index=False)
res.to_csv("axis_clinical_correlations.csv", index=False)
load.sort_values(ascending=False).to_csv("axis_loadings.csv")
np.save("axis_pc.npy", pc[:, :2])
print("\nwrote axis_scores.csv, axis_clinical_correlations.csv, axis_loadings.csv")
