"""Loaders for the two newly added external validation cohorts.

GSE125977 -- adult idiopathic inflammatory myopathy WHOLE BLOOD RNA-seq
             (Hanna et al.; whole blood, Illumina HiSeq 4000)
             5 healthy controls, 7 DM, 7 PM, 5 IBM  = 24 mRNA samples,
             plus a nested anti-Jo-1-positive contrast (5 Jo-1+ vs 5 HC).

GSE143323 -- muscle biopsy RNA-seq, DM n=39 vs normal muscle n=20.
"""
import gzip
import numpy as np
import pandas as pd
import xlrd

# ---------------------------------------------------------------- GSE125977

# The DM / PM / IBM contrast tables all share the same five control samples.
CTRL = ["JP11_S11", "JP16_S16", "JP20_S20", "JP3_S3", "JP8_S8"]
GROUPS_125977 = {
    "DM":  ["JP10_S10", "JP15_S15", "JP1_S1", "JP22_S22", "JP4_S4", "JP7_S7", "JP9_S9"],
    "PM":  ["JP14_S14", "JP17_S17", "JP19_S19", "JP21_S21", "JP23_S23", "JP2_S2", "JP6_S6"],
    "IBM": ["JP12_S12", "JP13_S13", "JP18_S18", "JP24_S24", "JP5_S5"],
}
TABLES_125977 = {
    "DM":  "GSE125977_mRNA_star_DM_vis_CONTROL_Wald_DESeq2.xls",
    "PM":  "GSE125977_mRNA_star_PM_vis_CONTROL_Wald_DESeq2.xls",
    "IBM": "GSE125977_mRNA_star_IBM_vis_CONTROL_Wald_DESeq2.xls",
}
JO1_TABLE = "GSE125977_mRNA_all_jo1_vs_controls_DESeq2.xls"


def _read_deseq_xls(path):
    """Return (normalised_counts_df, stats_df).

    Layout: ID | <raw counts> | <normalised counts, same names> | baseMean ...
    | gene_type | gene_name | description.  The two count blocks are equal in
    width; the second is DESeq2 size-factor normalised.
    """
    sh = xlrd.open_workbook(path).sheet_by_index(0)
    hdr = [sh.cell_value(0, c) for c in range(sh.ncols)]
    i_base = hdr.index("baseMean")
    n_samp = (i_base - 1) // 2
    raw_cols = list(range(1, 1 + n_samp))
    norm_cols = list(range(1 + n_samp, 1 + 2 * n_samp))
    names = [hdr[c] for c in raw_cols]

    rows, stats = [], []
    i_gn, i_gt = hdr.index("gene_name"), hdr.index("gene_type")
    i_l2, i_p, i_padj = hdr.index("log2FoldChange"), hdr.index("pvalue"), hdr.index("padj")
    for r in range(1, sh.nrows):
        gn = sh.cell_value(r, i_gn)
        if not isinstance(gn, str) or not gn:
            continue
        rows.append([gn] + [sh.cell_value(r, c) for c in norm_cols])

        def num(c):
            v = sh.cell_value(r, c)
            return float(v) if isinstance(v, (int, float)) else np.nan
        stats.append([gn, sh.cell_value(r, i_gt), num(i_base), num(i_l2), num(i_p), num(i_padj)])

    counts = pd.DataFrame(rows, columns=["gene_name"] + names)
    counts = counts.apply(pd.to_numeric, errors="ignore")
    # collapse duplicate symbols by highest mean expression
    counts["_m"] = counts[names].mean(axis=1)
    counts = (counts.sort_values("_m", ascending=False)
                    .drop_duplicates("gene_name")
                    .set_index("gene_name")[names])
    st = pd.DataFrame(stats, columns=["gene_name", "gene_type", "baseMean",
                                      "log2FoldChange", "pvalue", "padj"])
    st = (st.sort_values("baseMean", ascending=False)
            .drop_duplicates("gene_name").set_index("gene_name"))
    return counts, st


def load_gse125977_wholeblood():
    """Unified 24-sample adult whole-blood matrix (log2 normalised counts)."""
    mats, stats = {}, {}
    for grp, tbl in TABLES_125977.items():
        counts, st = _read_deseq_xls(tbl)
        mats[grp] = counts
        stats[grp] = st

    genes = set(mats["DM"].index)
    for g in ("PM", "IBM"):
        genes &= set(mats[g].index)
    genes = sorted(genes)

    # controls taken once, from the DM table
    out = mats["DM"].loc[genes, CTRL].copy()
    labels = {c: "HC" for c in CTRL}
    for grp in ("DM", "PM", "IBM"):
        cols = [c for c in GROUPS_125977[grp] if c in mats[grp].columns]
        out = pd.concat([out, mats[grp].loc[genes, cols]], axis=1)
        labels.update({c: grp for c in cols})

    expr = np.log2(out.astype(float) + 1.0)
    meta = pd.Series(labels)[expr.columns].rename("group")
    return expr, meta, stats


def load_gse125977_jo1():
    """Nested anti-Jo-1-positive contrast: 5 Jo-1+ vs the same 5 controls."""
    counts, st = _read_deseq_xls(JO1_TABLE)
    ctrl = [c for c in counts.columns if c.upper().startswith("CONTROL")]
    jo1 = [c for c in counts.columns if "jo1" in c.lower()]
    expr = np.log2(counts[ctrl + jo1].astype(float) + 1.0)
    meta = pd.Series({**{c: "HC" for c in ctrl}, **{c: "Jo1" for c in jo1}})[expr.columns]
    return expr, meta.rename("group"), st


# ---------------------------------------------------------------- GSE143323

def load_gse143323(path="GSE143323_net_gene_fpkm.csv.gz"):
    """Muscle-biopsy FPKM matrix: DM n=39 vs normal muscle (NT) n=20."""
    with gzip.open(path, "rt") as fh:
        df = pd.read_csv(fh, index_col=0)
    df.index = [str(i).replace("_AS1", "-AS1") for i in df.index]
    # the deposited matrix uses underscores for HLA genes
    df.index = pd.Index([i.replace("HLA_", "HLA-") for i in df.index])
    expr = np.log2(df.astype(float) + 1.0)
    grp = ["DM" if c.startswith("DM") else "NT" for c in expr.columns]
    return expr, pd.Series(grp, index=expr.columns, name="group")


# ------------------------------------------------- GSE220915 (original cohort)

def load_gse220915(counts="GSE220915_anonym_counts_hg38.tsv.gz",
                   soft="GSE220915.soft.gz",
                   ens2sym="gencode_map.tsv"):
    """Muscle RNA-seq n=165. Sample titles are 'sample_N_<group>'."""
    import re
    with gzip.open(soft, "rt", errors="replace") as fh:
        t = fh.read()
    lab = {}
    for b in t.split("^SAMPLE = ")[1:]:
        ti = re.search(r"!Sample_title = (.*)", b).group(1).strip()
        dg = re.search(r"diagnosis: (.*)", b).group(1).strip()
        col = "_".join(ti.split("_")[:2])            # sample_N
        g = ("NT" if "normal" in dg.lower()
             else re.search(r"\((\w+)\)", dg).group(1).upper())
        lab[col] = g
    with gzip.open(counts, "rt") as fh:
        df = pd.read_csv(fh, sep="\t", index_col=0)
    cpm = df.div(df.sum(axis=0), axis=1) * 1e6
    expr = np.log2(cpm + 1.0)
    expr.index = [str(i).split(".")[0] for i in expr.index]
    sym = pd.read_csv(ens2sym, sep="\t", index_col=0)["symbol"]
    expr = expr.loc[expr.index.intersection(sym.index)]
    expr = expr.assign(_s=sym.reindex(expr.index).values)
    expr["_m"] = expr.drop(columns="_s").mean(axis=1)
    expr = (expr.sort_values("_m", ascending=False).dropna(subset=["_s"])
                .drop_duplicates("_s").set_index("_s").drop(columns="_m"))
    expr.index.name = None
    meta = pd.Series({c: lab[c] for c in expr.columns if c in lab}, name="group")
    return expr[meta.index], meta


# ------------------------------------------------- GSE128470 (original cohort)

GSE128470_MAP = {"Dermatomyositis": "DM", "Inclusion body myositis": "IBM",
                 "Necrotizing myopathy": "IMNM", "Polymyositis": "PM",
                 "Nonspecific myositis": "NSM", "Normal": "NT"}


def load_gse128470(mat="GSE128470_series_matrix.txt.gz", annot="GPL96.annot.gz"):
    """Affymetrix HG-U133A muscle, n=77. Series matrix is already log2 RMA."""
    import re
    with gzip.open(mat, "rt", errors="replace") as fh:
        txt = fh.read()
    gsms = [v.strip('"') for v in
            re.search(r"!Sample_geo_accession\t(.*)", txt).group(1).split("\t")]
    dis = [v.strip('"').replace("disease state: ", "") for v in
           re.search(r"!Sample_characteristics_ch1\t(.*)", txt).group(1).split("\t")]
    body = txt.split("!series_matrix_table_begin\n")[1].split("!series_matrix_table_end")[0]
    from io import StringIO
    df = pd.read_csv(StringIO(body), sep="\t", index_col=0)
    df.index = [str(i).strip('"') for i in df.index]
    df.columns = [str(c).strip('"') for c in df.columns]

    ann = {}
    with gzip.open(annot, "rt", errors="replace") as fh:
        started = False
        for L in fh:
            if L.startswith("!platform_table_begin"):
                started = True
                next(fh)
                continue
            if not started or L.startswith("!platform_table_end"):
                continue
            p = L.rstrip("\n").split("\t")
            if len(p) > 2 and p[2] and "///" not in p[2]:
                ann[p[0]] = p[2]
    df = df.loc[df.index.intersection(ann.keys())]
    df = df.assign(_s=[ann[i] for i in df.index])
    df["_m"] = df.drop(columns="_s").mean(axis=1)
    df = (df.sort_values("_m", ascending=False).drop_duplicates("_s")
            .set_index("_s").drop(columns="_m"))
    df.index.name = None
    meta = pd.Series({g: GSE128470_MAP.get(d, d) for g, d in zip(gsms, dis)}, name="group")
    meta = meta[[g for g in df.columns if g in meta.index]]
    return df[meta.index].astype(float), meta
