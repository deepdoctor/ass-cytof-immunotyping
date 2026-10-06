# ass-cytof-immunotyping

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Analysis code accompanying:

> **Single-cell immunophenotyping reveals an innate-hyperactive immune continuum
> spanning autoantibody classes in antisynthetase syndrome that recapitulates the
> interferon–complement axis.** Zhao J\*, Guo L\*, Sun S, Chen Z, Fu Y, Wang K#.
> *Manuscript under review (revision R1), 2026.*

This repository contains all Python scripts used to (i) preprocess the
42-marker CyTOF discovery cohort (n = 10 ASyS patients), (ii) derive the
continuous innate-hyperactive axis and the descriptive regions IT1-IT3 along
it, (iii) project the 41-gene IT1 signature onto three independent
muscle-biopsy cohorts (GSE220915, GSE128470, GSE143323), an adult whole-blood
cohort containing healthy controls (GSE125977) and a juvenile-DM PBMC cohort
(GSE221091), and (iv) generate every main and supplementary figure in the
manuscript as editable vector PDFs.

Code for the extended external validation and the axis analysis lives under
`extended_validation/`; see the README in that directory. Analyses and figures
added in revision R1 (Supp. Tables ST8e–ST8g, the revised Fig. 2 and the re-plotted
Supp. Fig. S12) live under `revision_R1/`:

| script | output |
|---|---|
| `revision_R1/r1_analyses.py` | marker features recomputed without batch correction (ST8e); effect size, exact permutation test and power for autoantibody class on the axis (ST8f); T-cell pseudotime by region and patient (ST8g) |
| `revision_R1/fig2_axis_revised.py` | Fig. 2 with η² and its 95% CI in panel b |
| `revision_R1/suppfig12_jdm_replot.py` | Supp. Fig. S12 re-plotted from the 106 annotated GSE221091 samples |
| `revision_R1/verify_gse220915_mapping.py` | re-derives the GSE220915 projection under the legacy (mygene) and corrected (GENCODE) symbol mappings, and the two-cohort pooled effects |

Patient identifiers in this repository are study IDs (P01–P10) and sample
barcodes only.

---

## Repository scope

This repo contains **analysis code only**. Per IRB protocol LY2023-284-C
(Renji Hospital Ethics Committee), patient-identifiable raw data are not
distributed via this repository. To reproduce the analyses end-to-end:

| Data | Source | How to obtain |
|---|---|---|
| Discovery-cohort FCS files (10 patients) | FlowRepository | Will be deposited at FlowRepository upon publication; ID to be added |
| GSE220915 — bulk RNA-seq (n=165) | NCBI GEO | `GEOquery` or direct download |
| GSE128470 — Affymetrix microarray (n=77) | NCBI GEO | `GEOquery` or direct download |
| GSE221091 — JDM PBMC RNA-seq | NCBI GEO | `GEOquery` or direct download |

---

## Installation

```bash
# Recommended: fresh conda env
conda create -n cytof python=3.10 -y
conda activate cytof
pip install -r requirements.txt
```

Tested under macOS 14 (Apple Silicon) with Python 3.10.

---

## Reproducibility pipeline

Scripts are designed to be executed sequentially. Intermediate output is
written to `cytof_output_nature/` (which the scripts create on first run).

### Stage 1 — Preprocessing and atlas

```bash
python cytof_pipeline_v3.py             # Parse FCS, arcsinh, QC gating, Harmony,
                                        # Leiden, UMAP, cell-type annotation,
                                        # writes .h5ad + feature matrix
python cytof_h5ad_rebuild_umaps.py      # Re-renders UMAP plots from .h5ad
```

### Stage 2 — Patient-level immunotyping

```bash
python cytof_advanced_analysis.py       # 65-D feature vector, Ward clustering,
                                        # bootstrap + LOO stability
python cytof_within_AS.py               # Within-AS heterogeneity statistics
python cytof_fig10_treatment_sensitivity.py
                                        # Treatment-intensity sensitivity panels
```

### Stage 3 — External validation and meta-analysis

```bash
python cytof_external_validation.py     # GSE220915 IT1 projection, pathway
                                        # enrichment, within-AS DE
python cytof_supp_fig9_jdm_projection.py
                                        # GSE221091 JDM treatment-response
python cytof_supp_fig9_wilfong_projection.py
                                        # Wilfong 2022 secondary validation
```

### Stage 4 — Figure generation (vector PDFs)

Each `cytof_mainfig*_native.py` / `cytof_mainfig*_rebuild.py` script
generates one main-text figure. Each `cytof_supp_*.py` script generates a
supplementary figure. All outputs are 300 dpi PDFs with `pdf.fonttype=42`
(editable text in Illustrator / Inkscape).

```bash
python cytof_mainfig1_native.py         # Fig 1 — atlas
python cytof_mainfig2_native.py         # Fig 2 — marker characterisation
python cytof_mainfig3_rebuild.py        # Fig 3 — composition
python cytof_mainfig4_native.py         # Fig 4 — immunotypes
python cytof_mainfig5_native.py         # Fig 5 — differential expression
python cytof_mainfig6_rebuild.py        # Fig 6 — external validation
python cytof_mainfig7_trajectory.py     # Fig 7 — T-cell pseudotime
python cytof_mainfig8_network.py        # Fig 8 — cell-cell co-activation
python cytof_mainfig9_multicohort.py    # Fig 9 — multi-cohort meta-analysis
python cytof_supp_additions.py          # Supp Fig 1 (k-selection) + others
python cytof_supp_tier2.py              # Supp Fig 7 (LOO prediction + WGCNA)
python cytof_supp_fig8.py               # Supp Fig 8 (IFN-I vs IFN-II ROC)
```

---

## Figure-to-script map

| Manuscript figure | Generating script |
|---|---|
| Fig. 1 — Cohort and single-cell atlas | `cytof_mainfig1_native.py` |
| Fig. 2 — Functional-marker characterisation | `cytof_mainfig2_native.py` |
| Fig. 3 — Compositional heterogeneity | `cytof_mainfig3_rebuild.py` |
| Fig. 4 — Three serology-orthogonal immunotypes | `cytof_mainfig4_native.py` |
| Fig. 5 — Treatment-intensity sensitivity | `cytof_fig10_treatment_sensitivity.py` |
| Fig. 6 — Differential expression | `cytof_mainfig5_native.py` |
| Fig. 7 — External validation on GSE220915 | `cytof_mainfig6_rebuild.py` |
| Fig. 8 — T-cell pseudotime | `cytof_mainfig7_trajectory.py` |
| Fig. 9 — Cell–cell co-activation | `cytof_mainfig8_network.py` |
| Fig. 10 — Multi-cohort meta-analysis | `cytof_mainfig9_multicohort.py` |
| Supp Fig S1 — k-selection metrics | `cytof_supp_additions.py` |
| Supp Fig S2 — LOO stability | `cytof_supp_additions.py` |
| Supp Fig S3–S4 — Patient-aggregated DE | `cytof_supp_additions.py` |
| Supp Fig S5 — External-validation supplementary panels | `cytof_supp_additions.py` |
| Supp Fig S6 — Within-ASS heatmaps | `cytof_external_validation.py` |
| Supp Fig S7 — LOO prediction + WGCNA | `cytof_supp_tier2.py` |
| Supp Fig S8 — IFN-I vs IFN-II head-to-head ROC | `cytof_supp_fig8.py` |
| Supp Fig S9 — JDM PBMC projection | `cytof_supp_fig9_jdm_projection.py` |

---

## Statistical framework

- **Effect-size-first** reporting throughout — bootstrap 95% CIs
  (B = 2,000) rather than P-values where n per group is small.
- **Patient-level inference** is primary; cell-level Kruskal–Wallis results
  are reported only as exploratory because cells within a patient are not
  biologically independent units.
- **Multi-cohort meta-analysis** uses DerSimonian–Laird random-effects
  pooling of per-cohort Cohen's d, with I² heterogeneity statistic.

---

## Dependencies

See `requirements.txt`. Core libraries:

- `scanpy >= 1.10`, `anndata >= 0.10`
- `harmony-pytorch`, `leidenalg`, `umap-learn`, `phenograph`, `minisom`
- `scipy`, `scikit-learn`, `statsmodels`, `lifelines`
- `matplotlib`, `seaborn`, `adjustText`

---

## Citation

If you use this code, please cite the manuscript (full citation to be added
on acceptance) and this repository:

```bibtex
@software{zhao2026asys_cytof,
  author  = {Zhao, Jiangfeng and Guo, Li and Sun, Shuhui and Chen, Zhiwei
            and Fu, Yakai and Wang, Kaiwen},
  title   = {ass-cytof-immunotyping: analysis code for the serology-orthogonal
            innate-hyperactive axis in antisynthetase syndrome},
  year    = {2026},
  url     = {https://github.com/deepdoctor/ass-cytof-immunotyping}
}
```

---

## License

MIT License — see `LICENSE`.

---

## Corresponding authors

- **Kaiwen Wang** — Rheumatology and Immunology Laboratory, Jiading Branch,
  Renji Hospital, Shanghai Jiao Tong University School of Medicine, Shanghai,
  China. wangkaiwen_1983@163.com
- **Shuang Ye** — Department of Rheumatology, Renji Hospital, Shanghai
  Jiao Tong University School of Medicine, Shanghai, China.
  ye_shuang2000@163.com

Issues / questions about this code: please open a GitHub Issue.
