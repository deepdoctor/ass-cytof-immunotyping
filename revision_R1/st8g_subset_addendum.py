"""
Revision R1 -- addendum to Supp. Table ST8g: T-cell pseudotime by subset and
lineage (Fig. 6).  Adds sheet 'ST8g_pseudotime_by_subset' to the ST8e-ST8g
workbook written by r1_analyses.py.  Run from the repository root.
"""
import sys
import pandas as pd

WB = sys.argv[1] if len(sys.argv) > 1 else "revision_R1/Supp_Table_ST8e-ST8g.xlsx"
REG = ["IT1", "IT2", "IT3"]
pt = pd.read_csv("cytof_output_nature/29b_T_pseudotime_per_cell.csv")
pt["quintile"] = pd.qcut(pt["dpt"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
pt["lineage"] = pt["cell_type"].str[:3]

rows = []
for ct, sub in pt.groupby("cell_type"):
    r = {"cell_type": ct, "n_cells": len(sub),
         "median_DPT_all": round(sub["dpt"].median(), 4),
         "share_of_Q5_cells": round((pt.loc[pt.quintile == "Q5", "cell_type"] == ct).mean(), 4)}
    for reg in REG:
        s = sub[sub.immunotype == reg]
        r[f"median_DPT_{reg}"] = round(s["dpt"].median(), 4)
        r[f"share_of_{reg}_T_cells"] = round(len(s) / (pt.immunotype == reg).sum(), 4)
    rows.append(r)
for lin in ("CD4", "CD8"):
    sub = pt[pt.lineage == lin]
    r = {"cell_type": f"all {lin}", "n_cells": len(sub),
         "median_DPT_all": round(sub["dpt"].median(), 4),
         "share_of_Q5_cells": round((pt.loc[pt.quintile == "Q5", "lineage"] == lin).mean(), 4)}
    for reg in REG:
        s = sub[sub.immunotype == reg]
        r[f"median_DPT_{reg}"] = round(s["dpt"].median(), 4)
        r[f"share_of_{reg}_T_cells"] = round(len(s) / (pt.immunotype == reg).sum(), 4)
    rows.append(r)
out = pd.DataFrame(rows)
with pd.ExcelWriter(WB, mode="a", engine="openpyxl", if_sheet_exists="replace") as xw:
    out.to_excel(xw, sheet_name="ST8g_pseudotime_by_subset", index=False)
print(out.to_string(index=False))
