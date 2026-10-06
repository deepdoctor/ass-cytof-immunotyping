"""
Supp. Fig. S12 (GSE221091 juvenile-DM projection) -- re-plot for revision R1.

The deposited GSE221091 RPKM table carries 47 empty trailing columns
("Unnamed: 107" ...). They score as NaN and do not enter any test, but they were
counted in the panel-a title (n=153). This script re-plots the figure from the
saved per-sample scores restricted to the 106 annotated samples (88 JDM, 18 HC),
re-using the statistics and plotting code of cytof_supp_fig9_jdm_projection.py
unchanged, so every score, test and point is identical to the submitted figure.

Run from the repository root:
  python revision_R1/suppfig12_jdm_replot.py
"""
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.getcwd())          # nature_style.py lives in the repository root
from nature_style import apply_nature_style  # noqa: E402

warnings.filterwarnings("ignore")
apply_nature_style()

SRC = "cytof_supp_fig9_jdm_projection.py"
OUT = "revision_R1"

df = pd.read_csv("cytof_output_nature/S9_per_sample_scores_JDM.csv")
df = df[df["dx"].notna()].reset_index(drop=True)
assert (df["dx"] == "JDM").sum() == 88 and (df["dx"] == "HC").sum() == 18

code = open(SRC, encoding="utf-8").read()
start = code.index("# ─── [4] Pre-specified statistical tests ───")
stop = code.index("# Summary write-up")
ns = {"np": np, "pd": pd, "plt": plt, "gridspec": gridspec, "stats": stats,
      "df": df, "SC": 3.46, "DC": 7.09, "OUT_DIR": OUT}
exec(compile(code[start:stop], SRC, "exec"), ns)

for ext in ("pdf", "png"):
    os.replace(f"{OUT}/S9_PBMC_validation_JDM.{ext}", f"{OUT}/SuppFig_S12_JDM_revised.{ext}")
print(f"n annotated samples = {len(df)}; wrote {OUT}/SuppFig_S12_JDM_revised.pdf/.png")
