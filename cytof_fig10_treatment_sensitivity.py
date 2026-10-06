"""
Main Fig 10 — Treatment-confounding sensitivity analysis
=========================================================
Reviewer critique #1: all 10 patients on glucocorticoid + 8/10 on
second-line. Promote the partial-correlation sensitivity analysis from
the supplement into a main figure to defuse the obvious confound:

  (a) IT1 score vs prednisolone dose, per patient, with Spearman ρ
  (b) IT2 score vs prednisolone dose
  (c) IT3 score vs prednisolone dose
  (d) IT1 score vs second-line agent class (boxplot)
  (e) Partial-correlation table: IT1/2/3 × clinical (CK, LDH, ESR) with
      and without controlling for prednisolone dose — shows whether the
      clinical correlations survive treatment adjustment.

The principal claim is that IT1 high vs low is *not* driven by being on
more or less treatment — i.e., the IT1 signal we see is *residual*
above the immunosuppressive background, not a treatment artefact.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.preprocessing import StandardScaler
from nature_style import apply_nature_style
apply_nature_style()
SC, DC = 3.46, 7.09
OUT_DIR = "./cytof_output_nature"

# ── Build per-patient frame from Table 1 + 65-D feature matrix ──
feat = pd.read_csv(f"{OUT_DIR}/SuppTable_feature_matrix.csv")
imt  = pd.read_csv(f"{OUT_DIR}/21_immunotype_assignments.csv")
print("feature matrix:", feat.shape, "  immunotype assigns:", imt.shape)

# Map per-patient prednisolone dose + second-line class from Table 1
TREATMENT = {
    1:  {"pred":12.5, "second":"none"},
    2:  {"pred":60,   "second":"DMARD"},      # iguratimod + nintedanib
    3:  {"pred":40,   "second":"JAKi"},       # tofacitinib
    4:  {"pred":7.5,  "second":"IL6Ri"},      # tocilizumab
    5:  {"pred":40,   "second":"IL6Ri"},      # tocilizumab + nintedanib
    6:  {"pred":50,   "second":"CTX"},
    7:  {"pred":60,   "second":"MMF"},
    8:  {"pred":60,   "second":"CTX"},
    9:  {"pred":50,   "second":"CTX"},
    10: {"pred":60,   "second":"CNI+JAKi"},   # tacrolimus + tofa + pirfenidone
}
feat["pred_mg"]  = feat["pid"].map(lambda p: TREATMENT[int(p)]["pred"])
feat["second"]   = feat["pid"].map(lambda p: TREATMENT[int(p)]["second"])

# Compute per-patient z-scored sub-signature averages corresponding to
# the three immunotypes' defining axes. IT1 = innate hyperactive
# (monocyte/NK/DC + activation MFI); IT2 = CD4 memory (CD4 Central
# Memory + CD4 Effector Memory + Memory_score); IT3 = lymphoid-quiescent
# (Naive B + CD4 Naive T + Naive_score).
def pick_cols(df, patterns):
    cols = []
    for pat in patterns:
        cols += [c for c in df.columns if pat.lower() in c.lower()]
    return list(dict.fromkeys(cols))

it1_cols = pick_cols(feat, ["Inflammatory Monocyte","Classical Monocyte",
                            "Non-classical Monocyte","NK cell","mDC","pDC",
                            "MFI_HLA-DR","MFI_CD38","MFI_CD69","MFI_GranzymeB"])
it2_cols = pick_cols(feat, ["CD4 Central Memory","CD4 Effector Memory",
                            "MFI_CD45RO","Memory_score"])
it3_cols = pick_cols(feat, ["Naive B","CD4 Naive T","CD8 Naive T",
                            "MFI_CCR7","Naive_score"])

# Z-score each column then average
def zmean(df, cols):
    Z = StandardScaler().fit_transform(df[cols].values)
    return Z.mean(axis=1)

feat["IT1_score"] = zmean(feat, it1_cols)
feat["IT2_score"] = zmean(feat, it2_cols)
feat["IT3_score"] = zmean(feat, it3_cols)

# Merge in clinical metadata for partial correlations
clin = pd.read_csv(f"{OUT_DIR}/23_immunotype_clinical_table.csv")
merged = feat[["pid","sample","antibody","immunotype",
               "IT1_score","IT2_score","IT3_score",
               "pred_mg","second"]].merge(
    clin[["pid","CK","LDH","ESR","CRP","FVC","DLCO","age","DA_score"]],
    on="pid", how="left")
merged["logCK"]  = np.log10(merged["CK"].fillna(merged["CK"].median()) + 1)
merged["logLDH"] = np.log10(merged["LDH"].fillna(merged["LDH"].median()) + 1)
merged.to_csv(f"{OUT_DIR}/FIG10_treatment_sensitivity_per_patient.csv", index=False)

# ── Partial correlation: r(IT, clinical | pred_mg) ──
def partial_spearman(x, y, z):
    """Spearman partial correlation of x and y controlling for z, by ranks."""
    df = pd.DataFrame({"x": x, "y": y, "z": z}).dropna()
    if len(df) < 4: return np.nan, np.nan
    rx = df["x"].rank(); ry = df["y"].rank(); rz = df["z"].rank()
    # residualise rx and ry against rz (OLS on ranks → Spearman partial)
    res_x = rx - rx.mean() - (rz - rz.mean()) * np.cov(rx, rz)[0,1] / np.var(rz, ddof=1)
    res_y = ry - ry.mean() - (rz - rz.mean()) * np.cov(ry, rz)[0,1] / np.var(rz, ddof=1)
    r, p = stats.spearmanr(res_x, res_y)
    return float(r), float(p)

IT_SCORES = ["IT1_score","IT2_score","IT3_score"]
CLIN_VARS = ["logCK","logLDH","ESR","DA_score"]
rows = []
for it in IT_SCORES:
    for cv in CLIN_VARS:
        r_raw, p_raw = stats.spearmanr(merged[it], merged[cv], nan_policy="omit")
        r_par, p_par = partial_spearman(merged[it], merged[cv], merged["pred_mg"])
        rows.append({"score": it, "clinical": cv,
                     "rho_raw": round(float(r_raw),3), "p_raw": round(float(p_raw),3),
                     "rho_partial": round(r_par,3), "p_partial": round(p_par,3)})
partial_df = pd.DataFrame(rows)
partial_df.to_csv(f"{OUT_DIR}/FIG10_partial_correlations.csv", index=False)
print("Partial correlations:")
print(partial_df.to_string(index=False))

# Spearman of each IT score against pred_mg
print("\nIT score vs Pred dose (Spearman):")
for it in IT_SCORES:
    r, p = stats.spearmanr(merged[it], merged["pred_mg"])
    print(f"  {it} ↔ pred_mg  ρ = {r:+.3f},  p = {p:.3f}")

# ── Figure ──
IT_COL = {"IT1":"#E67E22","IT2":"#2980B9","IT3":"#27AE60"}
ANTI_COL = {"Jo-1":"#C0392B","PL-12":"#2471A3","EJ":"#1A7A4A","PL-7":"#7D3C98"}

RC = {
    "font.size":7,"axes.titlesize":7.5,"axes.labelsize":7,
    "xtick.labelsize":6,"ytick.labelsize":6,
    "axes.linewidth":0.6,"xtick.major.width":0.6,"ytick.major.width":0.6,
    "xtick.major.size":2.5,"ytick.major.size":2.5,
    "lines.linewidth":1.0,"patch.linewidth":0.4,"legend.fontsize":5.5,
}

with plt.rc_context(RC):
    fig = plt.figure(figsize=(DC+0.5, 5.6))
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.42, hspace=0.55,
                            height_ratios=[1.0, 1.2])

    # ── (a-c) IT score vs prednisolone dose, colored by antibody ──
    for i, it_name in enumerate(IT_SCORES):
        ax = fig.add_subplot(gs[0, i])
        for ab, col in ANTI_COL.items():
            sub = merged[merged["antibody"] == ab]
            ax.scatter(sub["pred_mg"], sub[it_name],
                       s=42, c=col, edgecolor="white", linewidth=0.5,
                       label=ab, zorder=3)
        # OLS line
        x = merged["pred_mg"].values; y = merged[it_name].values
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, m*xs+b, color="#555", lw=0.8, ls="--", alpha=0.5)
        r, p = stats.spearmanr(x, y)
        # patient labels
        for _, row in merged.iterrows():
            ax.annotate(f"P{int(row['pid']):02d}",
                        (row["pred_mg"], row[it_name]),
                        xytext=(4, 1), textcoords="offset points",
                        fontsize=5.0, color="#333")
        ax.set_xlabel("Prednisolone dose (mg/day)", fontsize=6.8)
        ax.set_ylabel(f"{it_name.replace('_score','')} signature z-score", fontsize=6.8)
        ax.set_title(f"{it_name.replace('_score','')} score vs treatment intensity\n"
                     f"Spearman ρ = {r:+.2f},  p = {p:.2f}",
                     fontsize=6.8, pad=3)
        if i == 2:
            ax.legend(fontsize=5.0, loc="upper right",
                      frameon=False, handletextpad=0.3)
        # panel label
        ax.text(-0.20, 1.08, "abc"[i], transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="bottom", ha="left")

    # ── (d) IT1 by second-line agent class ──
    ax = fig.add_subplot(gs[1, 0])
    AGENT_ORDER = ["none","DMARD","IL6Ri","JAKi","MMF","CTX","CNI+JAKi"]
    agent_data = [merged.loc[merged["second"]==a, "IT1_score"].values
                  for a in AGENT_ORDER]
    bp = ax.boxplot(agent_data, positions=range(len(AGENT_ORDER)),
                     widths=0.5, showfliers=False, patch_artist=True,
                     medianprops=dict(color="#222", lw=0.8))
    for box in bp["boxes"]:
        box.set_facecolor("#E67E22"); box.set_alpha(0.4); box.set_edgecolor("#E67E22")
    for whisker in bp["whiskers"]: whisker.set_color("#888"); whisker.set_lw(0.6)
    for cap in bp["caps"]:         cap.set_color("#888"); cap.set_lw(0.6)
    for i, a in enumerate(AGENT_ORDER):
        sub = merged[merged["second"]==a]
        ax.scatter(np.full(len(sub), i), sub["IT1_score"].values,
                   c=[ANTI_COL[ab] for ab in sub["antibody"]],
                   s=28, edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xticks(range(len(AGENT_ORDER)))
    ax.set_xticklabels(AGENT_ORDER, fontsize=5.5, rotation=30, ha="right")
    ax.set_ylabel("IT1 signature z-score", fontsize=6.8)
    ax.set_title("IT1 score by second-line immunomodulator\n"
                 "(JAK-inhibited patient P03/P10 still IT2-spectrum, not IT3)",
                 fontsize=6.6, pad=3)
    ax.text(-0.20, 1.08, "d", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")

    # ── (e) Partial correlation table panel ──
    ax = fig.add_subplot(gs[1, 1:])
    ax.axis("off")
    # 3 (IT) × 4 (clinical) = 12 cells, each with raw vs adjusted
    table = partial_df.pivot_table(index="score",
                                    columns="clinical",
                                    values=["rho_raw","rho_partial"])
    # nicer: build a clean grouped table
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    # Header
    ax.text(5.0, 9.6, "IT score × clinical correlations:  raw  →  partial  (controlling for prednisolone dose)",
            ha="center", va="center", fontsize=7.0, fontweight="bold")
    headers = ["IT score","clinical","ρ raw","p","ρ partial","p"]
    xs = [0.4, 2.2, 4.0, 5.0, 6.2, 7.4]
    for x, h in zip(xs, headers):
        ax.text(x, 8.6, h, ha="left", va="center",
                fontsize=6.8, fontweight="bold", color="#222")
    ax.plot([0.3, 9.7], [8.30, 8.30], color="#888", lw=0.5)
    # rows — tighter spacing
    row_step = 0.48
    for r_i, row in partial_df.iterrows():
        y = 8.0 - r_i * row_step
        ax.text(xs[0], y, row["score"].replace("_score",""),
                ha="left", va="center", fontsize=6.2, family="monospace")
        ax.text(xs[1], y, row["clinical"],
                ha="left", va="center", fontsize=6.2, family="monospace")
        ax.text(xs[2], y, f"{row['rho_raw']:+.2f}",
                ha="left", va="center", fontsize=6.2, family="monospace",
                color="#C0392B" if abs(row["rho_raw"])>=0.5 else "#333")
        ax.text(xs[3], y, f"{row['p_raw']:.2f}",
                ha="left", va="center", fontsize=6.0, family="monospace", color="#666")
        ax.text(xs[4], y, f"{row['rho_partial']:+.2f}",
                ha="left", va="center", fontsize=6.2, family="monospace",
                color="#1A7A4A" if abs(row["rho_partial"])>=0.5 else "#333")
        ax.text(xs[5], y, f"{row['p_partial']:.2f}",
                ha="left", va="center", fontsize=6.0, family="monospace", color="#666")
    bottom_y = 8.0 - len(partial_df)*row_step + 0.20
    ax.plot([0.3, 9.7], [bottom_y, bottom_y], color="#CCC", lw=0.4)
    # interpretation footnote — placed below the table line with breathing room
    ax.text(5.0, bottom_y - 0.85,
            "Spearman partial correlation by rank-residualisation against\n"
            "prednisolone (mg/day). Red |ρ|≥0.5 raw; green |ρ|≥0.5 partial.\n"
            "n = 10 throughout.",
            ha="center", va="center", fontsize=5.6, color="#555",
            style="italic", linespacing=1.2)
    ax.text(-0.04, 1.08, "e", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")

    fig.savefig(f"{OUT_DIR}/MainFig10_treatment_sensitivity.pdf",
                dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT_DIR}/MainFig10_treatment_sensitivity.png",
                dpi=300, bbox_inches="tight")
    print(f"\nSaved MainFig10_treatment_sensitivity.pdf/.png")
