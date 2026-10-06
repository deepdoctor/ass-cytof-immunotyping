"""
Revision R1 -- verification of the GSE220915 gene-mapping correction.

`cytof_external_validation.py` originally resolved signature symbols to Ensembl
IDs through mygene.info, which returns patch/alt ENSGs for TNF, IRF7 and
HLA-DRB1 that are absent from this cohort, so those three genes were silently
dropped.  The script now collapses the count matrix from ENSG to symbol with the
local GENCODE map instead, which recovers all 41 IT1 genes.

The submitted Fig. 5 and the statistics quoted in Results 3.7 were computed
before that correction; Methods 2.9 and Results 3.7 nevertheless state that
41/41 genes were detected, and the extended validation of Results 3.12 uses the
corrected pipeline.  This script re-derives the affected statistics from the raw
counts so the figure and text can be brought onto the corrected pipeline.

Inputs (local only):
  validation/counts.tsv.gz, validation/ensg2sym.csv, validation/series.txt.gz

Run from the repository root:
  python revision_R1/verify_gse220915_mapping.py
"""
import gzip
import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sps

warnings.filterwarnings("ignore")
OUT = "revision_R1"

IT1_GENES = {
    "CD14", "ITGAM", "ITGAX", "FCGR3A", "FCGR1A", "CD68", "CD86",
    "S100A8", "S100A9", "S100A12", "VCAN", "LYZ",
    "IL1B", "TNF", "CCL2", "CXCL10", "CXCL9",
    "CD1C", "CLEC4C", "IRF8", "IRF7",
    "NKG7", "GNLY", "GZMB", "GZMA", "GZMK", "PRF1", "KLRD1", "KLRK1", "NCAM1", "FCGR3B",
    "HLA-DRA", "HLA-DRB1", "CD38", "CD69", "CD274",
    "IFNG", "STAT1", "ISG15", "MX1", "OAS1",
}
DROPPED_BY_MYGENE = {"TNF", "IRF7", "HLA-DRB1"}
GROUPS = ["NT", "IMNM", "IBM", "DM", "ASyS"]

# ── load GSE220915 and collapse ENSG -> symbol (the corrected mapping) ───────
cnt = pd.read_csv("validation/counts.tsv.gz", sep="\t", index_col=0)
cnt.index = cnt.index.str.split(".").str[0]
emap = pd.read_csv("validation/ensg2sym.csv")
ens2sym = dict(zip(emap.iloc[:, 0], emap.iloc[:, 1]))
sym = cnt.index.map(ens2sym)
keep = pd.notna(sym)
cs = cnt.loc[keep].copy()
cs.index = sym[keep]
cs = cs.groupby(level=0).sum()
lcpm = np.log2(cs.div(cs.sum(0), axis=1) * 1e6 + 1.0)

lines = gzip.open("validation/series.txt.gz", "rt").readlines()
titles = [x.strip().strip('"') for l in lines if l.startswith("!Sample_title")
          for x in l.split("\t")[1:]]
diagnoses = [x.strip().strip('"').replace("diagnosis: ", "") for l in lines
             if l.startswith("!Sample_characteristics_ch1") and "diagnosis:" in l
             for x in l.split("\t")[1:]]


def short(d):
    d = d.strip().upper()
    for key, val in (("IMNM", "IMNM"), ("IBM", "IBM"), ("AS", "ASyS"),
                     ("DM", "DM"), ("NORMAL", "NT")):
        if key in d:
            return val
    return d[:6]


meta = pd.Series([short(d) for d in diagnoses],
                 index=[t.rsplit("_", 1)[0] for t in titles]).reindex(cs.columns)


def score(genes):
    g = sorted(set(genes) & set(lcpm.index))
    z = lcpm.loc[g].apply(lambda r: (r - r.mean()) / r.std() if r.std() > 0 else r - r.mean(),
                          axis=1)
    return z.mean(axis=0), g


def summarise(genes, label):
    s, g = score(genes)
    by = {k: s[meta == k].values for k in GROUPS}
    H, p = sps.kruskal(*by.values())
    out = {"label": label, "n_genes": len(g),
           "missing": sorted(set(genes) - set(g)),
           "KW_H": round(float(H), 2), "KW_P": float(p)}
    for comp in ("NT", "IMNM", "DM"):
        U, pu = sps.mannwhitneyu(by["ASyS"], by[comp])
        n1, n2 = len(by["ASyS"]), len(by[comp])
        out[f"ASyS_vs_{comp}"] = {"cliffs_delta": round(2 * U / (n1 * n2) - 1, 3),
                                  "auc": round(U / (n1 * n2), 3), "P": float(pu)}
    return out


corrected = summarise(IT1_GENES, "corrected mapping (41/41)")
legacy = summarise(IT1_GENES - DROPPED_BY_MYGENE, "legacy mygene mapping (38/41)")

# ── two-cohort random-effects pooling, as in extended_validation ────────────
def dl_pool(ds, ns):
    ds = np.asarray(ds, float)
    v = np.array([(n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2))
                  for d, (n1, n2) in zip(ds, ns)])
    w = 1 / v
    Q = float((w * (ds - (w * ds).sum() / w.sum()) ** 2).sum())
    k = len(ds)
    C = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0
    wr = 1 / (v + tau2)
    pooled = float((wr * ds).sum() / wr.sum())
    se = float(np.sqrt(1 / wr.sum()))
    return {"d": round(pooled, 2), "lo": round(pooled - 1.96 * se, 2),
            "hi": round(pooled + 1.96 * se, 2),
            "I2_pct": round(max(0.0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0.0)}


# per-cohort Cohen's d from the corrected pipeline (extended_validation/validation_summary.json)
PER_COHORT = {   # group: [(d, n_case, n_ctrl) for GSE220915, GSE128470]
    "DM":   [(1.8019, 44, 33), (2.6336, 12, 12)],
    "IBM":  [(2.2662, 16, 33), (2.2072, 26, 12)],
    "IMNM": [(1.4778, 54, 33), (2.1501, 6, 12)],
}
pooled2 = {g: dl_pool([d for d, _, _ in v], [(a, b) for _, a, b in v])
           for g, v in PER_COHORT.items()}

report = {"gse220915_projection": {"corrected": corrected, "legacy": legacy},
          "two_cohort_pooled_corrected": pooled2,
          "published_two_cohort_legacy": {"DM": {"d": 2.19, "I2_pct": 62},
                                          "IBM": {"d": 2.19, "I2_pct": 0},
                                          "IMNM": {"d": 1.66, "I2_pct": 25}}}
with open(f"{OUT}/gse220915_mapping_check.json", "w") as fh:
    json.dump(report, fh, indent=2, ensure_ascii=False)
print(json.dumps(report, indent=2, ensure_ascii=False))
