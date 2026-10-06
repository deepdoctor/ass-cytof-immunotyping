# Extended external validation and the innate-hyperactive axis

Analysis code for the revised version of *Single-cell immunophenotyping identifies a
continuous, serology-orthogonal innate-hyperactive axis recapitulating the
interferon–complement axis in antisynthetase syndrome*.

Everything here runs from **public data only** — the GEO accessions below plus the
deposited supplementary workbook — so all results are reproducible without the
discovery-cohort FCS files.

## Layout

```
it1_signature.py     41-gene IT1 signature, the 13-gene ISG core and 10-gene IFN-II
                     sub-signatures, the clinical IFN-4 score, and symbol aliasing
load_cohorts.py      loaders for GSE125977, GSE143323, GSE220915, GSE128470
score_and_test.py    signature scoring, Cohen's d with bootstrap CI, AUC,
                     DerSimonian-Laird pooling, gene-level concordance tests
run_validation.py    reproduces Results 3.12, Supp. Fig. S13 (labelled S15 in earlier drafts) and Supp. Table ST7

axis/build_axis.py   derives the innate-hyperactive axis from Supp. Table ST1;
                     reproduces Results 3.3-3.4 and Supp. Table ST8
axis/make_fig_s16.py draft axis figure (63-feature sensitivity version; the manuscript's Fig. 2 is
                     drawn by revision_R1/fig2_axis_revised.py from the primary 66-feature axis)

hc/analyze_with_hc.py  pre-specified analysis for a future healthy-control CyTOF
                       group (see HC_ANALYSIS_PLAN.md); not used in the manuscript
```

## Requirements

```
python>=3.9
pandas>=2.0
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
openpyxl>=3.1
xlrd==1.2.0        # required to read the legacy .xls tables deposited with GSE125977
```

`xlrd` must be 1.2.x: version 2.x dropped `.xls` support, and pandas cannot read these
files, so `load_cohorts.py` calls `xlrd` directly.

## Input data

Download into the working directory before running `run_validation.py`:

| file | source |
|---|---|
| `GSE125977_mRNA_*_DESeq2.xls.gz` (4 files) | GEO GSE125977 supplementary |
| `GSE143323_net_gene_fpkm.csv.gz` | GEO GSE143323 supplementary |
| `GSE220915_anonym_counts_hg38.tsv.gz` + `GSE220915_family.soft.gz` | GEO GSE220915 |
| `GSE128470_series_matrix.txt.gz` | GEO GSE128470 matrix |
| `GPL96.annot.gz` | `ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL96/annot/` |
| `gencode_map.tsv` | built by the snippet at the end of `load_cohorts.py` usage notes |

`axis/build_axis.py` needs only `Supplementary Tables.xlsx` (sheets ST1 and ST2).

## Reproducing the manuscript

```bash
gunzip -k GSE125977_*.xls.gz          # the .xls readers expect uncompressed files
python run_validation.py              # -> Supp. Fig. S13, Supp. Table ST7
cd axis && python build_axis.py && python make_fig_s16.py
```

## Verification performed

`run_validation.py` re-scores GSE220915 and GSE128470 from raw data with the same
pipeline as the newly added cohorts. This reproduces the previously published
per-cohort values (ASyS Cohen's d = 2.50; 41/41 and 36/41 signature genes detected),
confirming that all four cohorts enter the meta-analysis on a common footing.

`build_axis.py` excludes the Tfh, Memory and Migration composites (63 features; the sensitivity analysis of Methods 2.5) and gives PC1 = 34.2% and PC2 = 23.7% of variance. With all 66 features (the primary axis) PC1 = 35.8% and PC2 = 22.7%, as reported in the manuscript; the 66-feature axis is reproduced by `revision_R1/r1_analyses.py`.

## Determinism

All bootstrap procedures use fixed seeds (`numpy.random.default_rng(20250726)` in
`score_and_test.py`, `20260727` in `build_axis.py`), so reported confidence intervals
are exactly reproducible.

## Licence

MIT, as for the parent repository.
