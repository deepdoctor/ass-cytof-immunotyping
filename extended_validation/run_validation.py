"""Extended external validation of the 41-gene IT1 signature.

Adds two cohorts to the two already reported in the manuscript:
  GSE143323  muscle biopsy RNA-seq, DM n=39 vs normal muscle n=20   (3rd muscle cohort)
  GSE125977  ADULT WHOLE BLOOD RNA-seq, 5 healthy controls vs 7 DM / 7 PM / 5 IBM,
             plus a nested anti-Jo-1-positive contrast (5 Jo-1+ vs 5 HC)
             -> the first healthy-control-anchored, blood-compartment validation.

Also recomputes GSE220915 and GSE128470 from raw data so that every cohort is
scored by an identical pipeline and the meta-analysis is internally consistent.

Outputs: results tables (CSV) + Supp. Fig. S15 (PDF/PNG).
"""
import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from it1_signature import (IT1_41, IT1_MODULES, ISG_CORE_13, IFN_II_10,
                           CLINICAL_IFN4, resolve)
import load_cohorts as LC
import score_and_test as ST

matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                            "font.size": 8, "axes.linewidth": 0.6,
                            "xtick.major.width": 0.6, "ytick.major.width": 0.6})

OUT = {}

# ------------------------------------------------------------------ load all
print("loading cohorts ...")
e915, m915 = LC.load_gse220915()
e847, m847 = LC.load_gse128470()
e433, m433 = LC.load_gse143323()
eWB, mWB, _ = LC.load_gse125977_wholeblood()
eJO, mJO, stJO = LC.load_gse125977_jo1()

COHORTS = {
    "GSE220915 (muscle, RNA-seq)":   (e915, m915, "NT"),
    "GSE128470 (muscle, array)":     (e847, m847, "NT"),
    "GSE143323 (muscle, RNA-seq)":   (e433, m433, "NT"),
    "GSE125977 (whole blood, RNA-seq)": (eWB, mWB, "HC"),
}

# ------------------------------------------------- per-cohort IT1 contrasts
rows = []
scores = {}
for name, (expr, meta, ctrl) in COHORTS.items():
    sc, used, miss = ST.signature_score(expr, IT1_41, resolve)
    scores[name] = (sc, meta, ctrl)
    for g in [x for x in meta.unique() if x != ctrl]:
        if (meta == g).sum() < 3:
            continue
        r = ST.contrast(sc, meta, g, ctrl=ctrl)
        r.update(cohort=name, n_genes=len(used))
        rows.append(r)
res = pd.DataFrame(rows)
OUT["per_cohort_contrasts"] = res
print("\n=== per-cohort IT1 contrasts ===")
print(res[["cohort", "case", "n_case", "n_ctrl", "n_genes", "d", "d_lo", "d_hi",
           "auc", "p_two"]].to_string(index=False, float_format=lambda x: f"{x:.3g}"))

# --------------------------------------------------- muscle meta-analysis
print("\n=== random-effects meta-analysis (muscle cohorts only) ===")
meta_rows = []
mus = res[res.cohort.str.contains("muscle")]
for dz in ["DM", "IBM", "IMNM", "AS"]:
    sub = mus[mus.case == dz]
    if len(sub) == 0:
        continue
    pool = ST.dl_pool(sub.d.values, list(zip(sub.n_case, sub.n_ctrl)))
    meta_rows.append(dict(group=dz, k=pool["k"], d=pool["d"], lo=pool["lo"],
                          hi=pool["hi"], I2=pool["I2"], p_Q=pool["p_Q"],
                          per_cohort="; ".join(f"{c.split()[0]}={d:.2f}"
                                               for c, d in zip(sub.cohort, sub.d))))
meta_df = pd.DataFrame(meta_rows)
OUT["muscle_meta"] = meta_df
print(meta_df.to_string(index=False, float_format=lambda x: f"{x:.3g}"))

# ------------------------------------- module breakdown in the blood cohort
MODS = dict(IT1_MODULES)
MODS["ISG core (IFN-I, 13)"] = ISG_CORE_13
MODS["IFN-II / IFNg (10)"] = IFN_II_10
MODS["clinical IFN-4"] = CLINICAL_IFN4

mIIM = mWB.replace({"DM": "IIM", "PM": "IIM", "IBM": "IIM"})
mod_rows = []
for nm, gs in MODS.items():
    s_wb, u_wb, _ = ST.signature_score(eWB, gs, resolve)
    r_wb = ST.contrast(s_wb, mIIM, "IIM")
    s_jo, u_jo, _ = ST.signature_score(eJO, gs, resolve)
    r_jo = ST.contrast(s_jo, mJO, "Jo1")
    gc = ST.gene_concordance(stJO, gs, resolve)
    mod_rows.append(dict(module=nm, n_genes=len(u_wb),
                         iim_d=r_wb["d"], iim_lo=r_wb["d_lo"], iim_hi=r_wb["d_hi"],
                         iim_auc=r_wb["auc"], iim_p1=r_wb["p_one"],
                         jo1_d=r_jo["d"], jo1_auc=r_jo["auc"], jo1_p1=r_jo["p_one"],
                         jo1_up=f"{gc['n_up']}/{gc['n_tested']}",
                         jo1_med_l2fc=gc["median_l2fc"], jo1_p_comp=gc["p_competitive"]))
mod_df = pd.DataFrame(mod_rows)
OUT["blood_modules"] = mod_df
print("\n=== module breakdown, adult whole blood ===")
print(mod_df.to_string(index=False, float_format=lambda x: f"{x:.3g}"))

# ------------------------------------------- whole-signature blood contrasts
sc_wb, _, _ = ST.signature_score(eWB, IT1_41, resolve)
sc_jo, _, _ = ST.signature_score(eJO, IT1_41, resolve)
blood_rows = [ST.contrast(sc_wb, mIIM, "IIM"), ST.contrast(sc_jo, mJO, "Jo1")]
for r, lab in zip(blood_rows, ["all IIM vs HC", "anti-Jo-1+ vs HC"]):
    r["label"] = lab
OUT["blood_composite"] = pd.DataFrame(blood_rows)
print("\n=== composite IT1 in blood ===")
print(OUT["blood_composite"][["label", "n_case", "n_ctrl", "d", "d_lo", "d_hi",
                              "auc", "p_two", "p_one"]].to_string(index=False,
                              float_format=lambda x: f"{x:.3g}"))

gc41 = ST.gene_concordance(stJO, IT1_41, resolve)
OUT["jo1_gene_table"] = gc41["table"]
print(f"\nanti-Jo-1 gene level: {gc41['n_up']}/{gc41['n_tested']} up, "
      f"binom p={gc41['p_binom']:.3g}, competitive p={gc41['p_competitive']:.3g}, "
      f"median log2FC={gc41['median_l2fc']:.2f}, padj<0.05 up={gc41['n_sig_up']}")

# ------------------------------------------------------------------- figure
fig = plt.figure(figsize=(7.2, 8.6))
gs = fig.add_gridspec(4, 2, height_ratios=[1.05, 1.0, 1.0, 1.15],
                      hspace=0.75, wspace=0.36)
C = {"NT": "#9aa3ad", "HC": "#9aa3ad", "DM": "#d1495b", "PM": "#e3a72f",
     "IBM": "#5b8c5a", "IMNM": "#5d76cb", "AS": "#8e5ea2", "NSM": "#7f7f7f",
     "IIM": "#d1495b", "Jo1": "#8e5ea2"}


def strip(ax, sc, meta, order, title, ylab="IT1 score"):
    rng = np.random.default_rng(0)
    for i, g in enumerate(order):
        v = sc[meta[meta == g].index].values
        ax.scatter(np.full(len(v), i) + rng.uniform(-.16, .16, len(v)), v, s=9,
                   c=C.get(g, "#555"), alpha=.85, linewidths=0, zorder=3)
        ax.hlines(np.median(v), i - .3, i + .3, color="k", lw=1.4, zorder=4)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{g}\nn={(meta==g).sum()}" for g in order], fontsize=6.5)
    ax.set_ylabel(ylab, fontsize=7.5)
    ax.set_title(title, fontsize=7.8, pad=4)
    ax.axhline(0, color="#cccccc", lw=.5, ls=":", zorder=1)
    ax.tick_params(labelsize=6.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# (a) new muscle cohort
ax = fig.add_subplot(gs[0, 0])
s433, _, _ = ST.signature_score(e433, IT1_41, resolve)
strip(ax, s433, m433, ["NT", "DM"], "a  GSE143323 muscle (new 3rd cohort)")
r = res[(res.cohort.str.startswith("GSE143323")) & (res.case == "DM")].iloc[0]
ax.text(.5, .96, f"d={r.d:.2f} [{r.d_lo:.2f},{r.d_hi:.2f}]  AUC={r.auc:.3f}",
        transform=ax.transAxes, ha="center", va="top", fontsize=6.4)

# (b) adult whole blood, all groups
ax = fig.add_subplot(gs[0, 1])
strip(ax, sc_wb, mWB, ["HC", "DM", "PM", "IBM"], "b  GSE125977 adult whole blood (new)")
rb = OUT["blood_composite"].iloc[0]
ax.text(.5, .96, f"IIM vs HC d={rb.d:.2f}  AUC={rb.auc:.2f}  P={rb.p_two:.2f} (ns)",
        transform=ax.transAxes, ha="center", va="top", fontsize=6.4)

# (c) muscle forest plot, per cohort + pooled
ax = fig.add_subplot(gs[1, :])
XMAX = 4.0
ticks = []          # (y, label) pairs, kept in lockstep with the plotted rows
y = 0.0
for dz in ["AS", "DM", "IBM", "IMNM"]:
    sub = mus[mus.case == dz]
    for _, rr in sub.iterrows():
        hi_clip = min(rr.d_hi, XMAX)
        ax.errorbar(rr.d, y, xerr=[[rr.d - rr.d_lo], [hi_clip - rr.d]], fmt="o",
                    ms=3.4, lw=.9, color=C.get(dz, "#555"), capsize=1.6)
        if rr.d_hi > XMAX:      # CI runs off-scale: mark with an arrow
            ax.annotate("", xy=(XMAX + .07, y), xytext=(hi_clip, y),
                        arrowprops=dict(arrowstyle="-|>", lw=.9, color=C.get(dz, "#555")))
        ticks.append((y, f"{dz}  {rr.cohort.split()[0]} ({rr.n_case}v{rr.n_ctrl})"))
        y += 1
    mr = meta_df[meta_df.group == dz]
    if len(mr):
        mr = mr.iloc[0]
        ax.errorbar(mr.d, y, xerr=[[mr.d - mr.lo], [mr.hi - mr.d]], fmt="D",
                    ms=4.6, lw=1.6, color="k", capsize=2.2)
        tag = "single cohort" if int(mr.k) == 1 else f"k={int(mr.k)}, I²={mr.I2:.0f}%"
        ticks.append((y, f"{dz}  POOLED ({tag})"))
        y += 1
    y += .6
ax.set_yticks([t[0] for t in ticks])
ax.set_yticklabels([t[1] for t in ticks], fontsize=6.2)
ax.set_ylim(y - .4, -0.7)
ax.set_xlim(-0.05, XMAX + .18)
ax.axvline(0, color="#bbbbbb", lw=.6, ls=":")
ax.set_xlabel("Cohen's d  (disease vs normal muscle);  arrow = 95% CI extends beyond axis",
              fontsize=7.2)
ax.set_title("c  Muscle meta-analysis, now three independent cohorts",
             fontsize=7.8, pad=4)
ax.tick_params(labelsize=6.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# (d) anti-Jo-1 vs HC, composite
ax = fig.add_subplot(gs[2, 0])
strip(ax, sc_jo, mJO, ["HC", "Jo1"], "d  anti-Jo-1⁺ whole blood vs HC")
ax.set_xticklabels([f"HC\nn=5", f"anti-Jo-1⁺\nn=5"], fontsize=6.5)
rj = OUT["blood_composite"].iloc[1]
ax.text(.5, .96, f"composite d={rj.d:.2f}  P={rj.p_two:.2f} (ns)",
        transform=ax.transAxes, ha="center", va="top", fontsize=6.4)

# (e) interferon module in anti-Jo-1
ax = fig.add_subplot(gs[2, 1])
s_ifn, _, _ = ST.signature_score(eJO, ISG_CORE_13, resolve)
strip(ax, s_ifn, mJO, ["HC", "Jo1"], "e  Type-I ISG core (13 genes), anti-Jo-1⁺",
      ylab="ISG-core score")
ax.set_xticklabels(["HC\nn=5", "anti-Jo-1⁺\nn=5"], fontsize=6.5)
rm = mod_df[mod_df.module == "ISG core (IFN-I, 13)"].iloc[0]
ax.text(.5, .96, f"d={rm.jo1_d:.2f}  AUC={rm.jo1_auc:.2f}  P={rm.jo1_p1:.3f}",
        transform=ax.transAxes, ha="center", va="top", fontsize=6.4)

# (f) module-resolved effect sizes in blood
ax = fig.add_subplot(gs[3, :])
order = ["ISG core (IFN-I, 13)", "clinical IFN-4", "IFN-II / IFNg (10)",
         "interferon_extra", "myeloid_monocyte", "activation",
         "inflammatory_mediator", "nk_cytotoxic", "dendritic_cell"]
PRETTY = {"interferon_extra": "IFN genes\nin IT1 (5)",
          "ISG core (IFN-I, 13)": "ISG core\nIFN-I (13)",
          "clinical IFN-4": "clinical\nIFN-4 score",
          "IFN-II / IFNg (10)": "IFN-II /\nIFN-\u03b3 (10)",
          "myeloid_monocyte": "myeloid /\nmonocyte (12)", "activation": "activation\n(5)",
          "inflammatory_mediator": "inflammatory\nmediator (5)",
          "nk_cytotoxic": "NK /\ncytotoxic (10)", "dendritic_cell": "dendritic\ncell (4)"}
md = mod_df.set_index("module").loc[order]
xx = np.arange(len(order))
ax.bar(xx - .2, md.iim_d, .38, color="#d1495b", label="all adult IIM vs HC (19 v 5)")
ax.bar(xx + .2, md.jo1_d, .38, color="#8e5ea2", label="anti-Jo-1⁺ vs HC (5 v 5)")
for i, (a, b) in enumerate(zip(md.iim_p1, md.jo1_p1)):
    if a < .05:
        ax.text(i - .2, md.iim_d.iloc[i] + .09, "*", ha="center", fontsize=8)
    if b < .05:
        ax.text(i + .2, md.jo1_d.iloc[i] + .09, "*", ha="center", fontsize=8)
ax.axhline(0, color="k", lw=.6)
lo_y = min(md.jo1_d.min(), md.iim_d.min()) - .35
hi_y = max(md.jo1_d.max(), md.iim_d.max()) + .55
ax.set_ylim(lo_y, hi_y)
ax.axvspan(6.5, 8.5, color="#f4f4f4", zorder=0)
ax.text(7.5, hi_y * .62,
        "do not replicate\nin whole blood",
        ha="center", fontsize=5.8, color="#666666", style="italic")
ax.set_xticks(xx)
ax.set_xticklabels([PRETTY[o] for o in order], fontsize=6.0)
ax.set_ylabel("Cohen's d vs healthy control", fontsize=7.5)
ax.set_title("f  Which arms of IT1 replicate in blood against healthy controls "
             "(* one-sided P<0.05)", fontsize=7.8, pad=9)
ax.legend(fontsize=6.2, frameon=False, loc="upper right")
ax.tick_params(labelsize=6.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.suptitle("Supp. Fig. S15  Extended external validation: a third muscle cohort and "
             "the first\nhealthy-control-anchored blood cohort", fontsize=8.6, y=.985)
fig.savefig("SuppFig_S15_extended_validation.pdf", bbox_inches="tight", dpi=300)
fig.savefig("SuppFig_S15_extended_validation.png", bbox_inches="tight", dpi=220)
print("\nwrote SuppFig_S15_extended_validation.pdf/.png")

# ------------------------------------------------------------------ save
with pd.ExcelWriter("ST7_extended_validation.xlsx") as xw:
    OUT["per_cohort_contrasts"].to_excel(xw, "per_cohort_IT1", index=False)
    OUT["muscle_meta"].to_excel(xw, "muscle_meta_analysis", index=False)
    OUT["blood_modules"].to_excel(xw, "blood_module_breakdown", index=False)
    OUT["blood_composite"].to_excel(xw, "blood_composite", index=False)
    OUT["jo1_gene_table"].to_excel(xw, "antiJo1_41gene_DE")
print("wrote ST7_extended_validation.xlsx")

summary = {k: (v.to_dict("records") if isinstance(v, pd.DataFrame) else v)
           for k, v in OUT.items() if k != "jo1_gene_table"}
with open("validation_summary.json", "w") as fh:
    json.dump(summary, fh, indent=1, default=float)
print("wrote validation_summary.json")
