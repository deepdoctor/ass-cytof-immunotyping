"""
Supp. Fig. S9 — Independent PBMC validation of the IT1 immunotype
==================================================================
Projection of the 41-marker IT1 immunotype signature onto the Wilfong
*et al.* 2022 (Front Immunol) myositis PBMC CyTOF dataset (DM/PM/IMNM,
n = 36) and/or Galindo-Feria *et al.* 2020 (DM/PM PBMC scRNA-seq,
n = 14).

Pipeline:
  (1) Download FCS files from FlowRepository (Wilfong) or counts from
      GEO (Galindo-Feria) into ./validation_pbmc/
  (2) Load + arcsinh-transform CyTOF data (Wilfong) or log2 CPM
      (Galindo-Feria); harmonise marker / gene names against the
      41-marker IT1 panel below.
  (3) Compute per-cell signature scores via mean of z-scored marker
      expression; aggregate to per-patient median.
  (4) Per-diagnostic-group violin + strip (DM vs PM vs IMNM); within-
      group IT1-high/IT1-low Ward stratification to test patient-level
      partition recovery.
  (5) Companion split: project just the IFN-I sub-signature (ISG-core
      orthologues) and IFN-II sub-signature (CXCL9/10 etc.) — confirm
      Supp Fig S8 finding (DM > AS for type-I IFN; IMNM lowest IFN-γ)
      reproduces in PBMC.

Output: S9_PBMC_validation.pdf/.png + S9_per_patient_scores.csv

----------------------------------------------------------------------
USAGE — fill in the dataset accession and uncomment the loader below
once you have downloaded the public files. The analysis logic
(scoring, stratification, plotting) is dataset-agnostic.
----------------------------------------------------------------------
"""

import os, warnings, glob
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.cluster import hierarchy
from sklearn.preprocessing import StandardScaler

from nature_style import apply_nature_style
apply_nature_style()
SC, DC = 3.46, 7.09
PBMC_DIR  = "./validation_pbmc"
OUT_DIR   = "./cytof_output_nature"
os.makedirs(PBMC_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# IT1 marker / gene signature (41 members) and IFN sub-signatures
# Tracks Supp. Fig. S8 sub-signature definitions for consistency.
# ─────────────────────────────────────────────────────────────────────
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

# CyTOF protein → mRNA mapping for Wilfong (CyTOF). Add panel-specific
# entries as the Wilfong panel becomes available; HLA-DR & many CD
# markers map directly. CyTOF surface markers correspond to subset of
# the IT1 list below.
CYTOF_TO_GENE = {
    "CD14":"CD14","CD11b":"ITGAM","CD11c":"ITGAX","CD16":"FCGR3A",
    "CD68":"CD68","CD86":"CD86","HLA-DR":"HLA-DRA","CD38":"CD38",
    "CD69":"CD69","CD274":"CD274","CD1c":"CD1C","NKG7":"NKG7",
    "GranzymeB":"GZMB","GZMA":"GZMA","CD56":"NCAM1","CD66b":"FCGR3B",
    "PD-L1":"CD274","CD83":"CD83",
}

# ─────────────────────────────────────────────────────────────────────
# 1) DATA LOADER — uncomment the relevant block once downloaded.
# ─────────────────────────────────────────────────────────────────────
def load_wilfong_cytof():
    """Load Wilfong 2022 FCS files from ./validation_pbmc/wilfong/*.fcs.
    Expects a metadata.csv with columns: file, sample_id, diagnosis.
    Returns a long-format DataFrame with one row per cell and columns
    sample_id, diagnosis, plus arcsinh-transformed marker channels.

    Download instructions:
      1. Go to https://flowrepository.org/ and search "Wilfong 2022".
      2. Request the FCS deposit (typically FR-FCM-Zxxx).
      3. Save FCS files to ./validation_pbmc/wilfong/ ; provide
         metadata.csv mapping each filename to diagnostic group.
    """
    import fcsparser
    meta = pd.read_csv(f"{PBMC_DIR}/wilfong/metadata.csv")
    rows = []
    for _, m in meta.iterrows():
        fname = f"{PBMC_DIR}/wilfong/{m['file']}"
        if not os.path.exists(fname):
            print(f"  missing: {fname}")
            continue
        hdr, df = fcsparser.parse(fname)
        df = df.rename(columns={c: c.split("_")[-1] for c in df.columns})
        df["sample_id"] = m["sample_id"]; df["diagnosis"] = m["diagnosis"]
        rows.append(df)
    cells = pd.concat(rows, axis=0, ignore_index=True)
    # arcsinh transform (cofactor 5; standard for CyTOF)
    for col in cells.columns:
        if col not in {"sample_id","diagnosis"}:
            cells[col] = np.arcsinh(cells[col] / 5.0)
    return cells

def load_galindo_feria_scrna():
    """Load Galindo-Feria 2020 DM/PM PBMC scRNA-seq from GSE150681.
    Expects ./validation_pbmc/galindo_feria/{counts.h5ad, metadata.csv}.
    """
    import anndata as ad
    adata = ad.read_h5ad(f"{PBMC_DIR}/galindo_feria/counts.h5ad")
    meta = pd.read_csv(f"{PBMC_DIR}/galindo_feria/metadata.csv")
    adata.obs = adata.obs.join(meta.set_index("cell_id"), how="left")
    return adata

# ─────────────────────────────────────────────────────────────────────
# 2) SCORING — CyTOF (protein) and scRNA (gene)
# ─────────────────────────────────────────────────────────────────────
def score_cytof_per_sample(cells, marker_set, sample_col="sample_id"):
    """Per-sample mean of z-scored marker expression."""
    detected = [m for m in marker_set if m in cells.columns]
    if not detected:
        return pd.Series(dtype=float), []
    # z-score each marker across all cells
    cells_z = cells.copy()
    for m in detected:
        mu, sd = cells_z[m].mean(), cells_z[m].std()
        if sd > 0:
            cells_z[m] = (cells_z[m] - mu) / sd
    # mean marker across z-scored, then per-sample median
    per_cell = cells_z[detected].mean(axis=1)
    per_sample = pd.DataFrame({"score": per_cell, sample_col: cells[sample_col]})\
                   .groupby(sample_col)["score"].median()
    return per_sample, detected

def score_scrna_per_sample(adata, gene_set):
    """Per-sample mean z-scored gene expression (pseudo-bulk)."""
    pres = [g for g in gene_set if g in adata.var_names]
    if not pres: return pd.Series(dtype=float), []
    expr = adata[:, pres].X.toarray() if hasattr(adata[:, pres].X, "toarray") \
                                       else adata[:, pres].X
    # log1p if raw counts
    expr = np.log1p(expr) if expr.max() > 50 else expr
    z = StandardScaler().fit_transform(expr)
    per_cell = z.mean(axis=1)
    df = pd.DataFrame({"score": per_cell, "sample_id": adata.obs["sample_id"].values})
    return df.groupby("sample_id")["score"].median(), pres

# ─────────────────────────────────────────────────────────────────────
# 3) FIGURE
# ─────────────────────────────────────────────────────────────────────
def make_supp_fig_s9(per_sample_scores, meta, dx_col="diagnosis"):
    """per_sample_scores: DataFrame indexed by sample_id with columns
    [IT1, IFN_I, IFN_II, Classical_IFN] ; meta: DataFrame with sample_id, diagnosis."""

    df = per_sample_scores.merge(meta[["sample_id", dx_col]],
                                  left_index=True, right_on="sample_id")
    DX_ORDER = ["NT","IMNM","PM","DM","AS"]
    DX_ORDER = [d for d in DX_ORDER if d in df[dx_col].unique()]
    DX_COL = {"NT":"#95A5A6","IMNM":"#2980B9","PM":"#16A085",
              "IBM":"#8E44AD","DM":"#F39C12","AS":"#C0392B"}
    SCORES = [("IT1","IT1 composite"),
              ("IFN_I","IFN-I (ISG core)"),
              ("IFN_II","IFN-II / IFN-γ"),
              ("Classical_IFN","Classical IFN")]

    RC = {"font.size":7,"axes.titlesize":7.5,"axes.labelsize":7,
          "xtick.labelsize":6,"ytick.labelsize":6,
          "axes.linewidth":0.6,"lines.linewidth":1.0}
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(DC+0.5, 5.6))
        gs = gridspec.GridSpec(2, 4, figure=fig, wspace=0.40, hspace=0.55)
        rng = np.random.default_rng(0)
        for i, (col, lab) in enumerate(SCORES):
            ax = fig.add_subplot(gs[0, i])
            groups = [df.loc[df[dx_col]==g, col].values for g in DX_ORDER]
            parts = ax.violinplot(groups, positions=range(len(DX_ORDER)),
                                   widths=0.82, showmedians=True)
            for j, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(DX_COL.get(DX_ORDER[j], "#999")); pc.set_alpha(0.55)
            for j, g in enumerate(DX_ORDER):
                y = df.loc[df[dx_col]==g, col].values
                x = j + rng.uniform(-0.12, 0.12, size=len(y))
                ax.scatter(x, y, s=6, c=DX_COL.get(g, "#999"), alpha=0.8,
                           edgecolor="white", linewidth=0.2, zorder=3)
            h, p = stats.kruskal(*groups) if len(groups)>1 else (0, 1)
            ax.set_xticks(range(len(DX_ORDER)))
            ax.set_xticklabels(DX_ORDER, fontsize=6)
            ax.set_title(f"{lab}\nKW H={h:.1f}, p={p:.2e}",
                         fontsize=6.6, pad=3)

        # Within-disease IT1-high/IT1-low partition
        for i, dx in enumerate(["DM","IMNM","PM","AS"][:4]):
            ax = fig.add_subplot(gs[1, i])
            sub = df[df[dx_col]==dx].copy()
            if len(sub) < 4:
                ax.text(0.5, 0.5, f"{dx}\nn={len(sub)} insufficient",
                        ha="center", va="center", transform=ax.transAxes)
                continue
            sub = sub.sort_values("IT1").reset_index(drop=True)
            med = sub["IT1"].median()
            sub["stratum"] = np.where(sub["IT1"] >= med, "IT1-high", "IT1-low")
            colors = ["#C0392B" if s == "IT1-high" else "#2980B9"
                      for s in sub["stratum"]]
            ax.scatter(range(len(sub)), sub["IT1"].values, c=colors,
                       s=22, edgecolor="white", linewidth=0.4)
            ax.axhline(med, color="#888", ls="--", lw=0.6)
            ax.set_xlabel(f"{dx} samples (IT1-sorted)", fontsize=6.6)
            ax.set_ylabel("IT1 score", fontsize=6.6)
            ax.set_title(f"Within-{dx} IT1-high/low\nmedian-split partition "
                         f"(n={len(sub)})", fontsize=6.6, pad=3)

        for i, ax in enumerate(fig.axes[:8]):
            ax.text(-0.20, 1.08, "abcdefgh"[i], transform=ax.transAxes,
                    fontsize=10, fontweight="bold", va="bottom", ha="left")

        out = f"{OUT_DIR}/S9_PBMC_validation.pdf"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        fig.savefig(out.replace(".pdf",".png"), dpi=300, bbox_inches="tight")
        print(f"  saved {out}")

# ─────────────────────────────────────────────────────────────────────
# 4) DRIVER (uncomment after downloading)
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if os.path.exists(f"{PBMC_DIR}/wilfong/metadata.csv"):
        print("[Wilfong] Loading CyTOF cells …")
        cells = load_wilfong_cytof()
        scores = pd.DataFrame(index=cells["sample_id"].unique())
        scores["IT1"],    _ = score_cytof_per_sample(cells,
                              {CYTOF_TO_GENE[m] for m in CYTOF_TO_GENE if CYTOF_TO_GENE[m] in IT1_GENES})
        scores["IFN_I"],  _ = score_cytof_per_sample(cells, IFN_I_GENES & set(cells.columns))
        scores["IFN_II"], _ = score_cytof_per_sample(cells, IFN_II_GENES & set(cells.columns))
        meta = cells[["sample_id","diagnosis"]].drop_duplicates()
        scores.to_csv(f"{OUT_DIR}/S9_per_patient_scores_wilfong.csv")
        make_supp_fig_s9(scores, meta)
    elif os.path.exists(f"{PBMC_DIR}/galindo_feria/counts.h5ad"):
        print("[Galindo-Feria] Loading scRNA-seq …")
        adata = load_galindo_feria_scrna()
        scores = pd.DataFrame(index=adata.obs["sample_id"].unique())
        scores["IT1"], _   = score_scrna_per_sample(adata, IT1_GENES)
        scores["IFN_I"], _ = score_scrna_per_sample(adata, IFN_I_GENES)
        scores["IFN_II"], _= score_scrna_per_sample(adata, IFN_II_GENES)
        meta = adata.obs[["sample_id","diagnosis"]].drop_duplicates()
        scores.to_csv(f"{OUT_DIR}/S9_per_patient_scores_galindo.csv")
        make_supp_fig_s9(scores, meta)
    else:
        print("==> No PBMC dataset downloaded yet.\n"
              "    Wilfong:       request FCS deposit from FlowRepository\n"
              "                   (search 'Wilfong 2022 myositis CyTOF'),\n"
              "                   save to ./validation_pbmc/wilfong/.\n"
              "    Galindo-Feria: download GSE150681 from GEO,\n"
              "                   save counts.h5ad + metadata.csv to\n"
              "                   ./validation_pbmc/galindo_feria/.\n"
              "    Then re-run this script.")
