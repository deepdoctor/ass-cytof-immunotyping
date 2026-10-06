"""
Main Figure 6 — Robustness, clinical correlations, and external validation.
Integrates panels from Fig 24, Fig 25, Fig 17 (top pairs), and S4 validation
into a single publication-ready 6-panel composite.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from scipy import stats

from nature_style import apply_nature_style, GROUP_PALETTE, HEATMAP_DIVERG, save
apply_nature_style()
DC = 7.09
OUT_DIR = "./cytof_output_nature"

# Load cached scores / stats
scores_df   = pd.read_csv(f"{OUT_DIR}/S4_validation_scores.csv", index_col=0)
val_stats   = pd.read_csv(f"{OUT_DIR}/S4_validation_stats.csv")
loo_corr    = pd.read_csv(f"{OUT_DIR}/S2_LOO_correlation_stability.csv")
immuno_tbl  = pd.read_csv(f"{OUT_DIR}/21_immunotype_assignments.csv")

print("Figure 6 inputs loaded.")

# reconstruct a bootstrap co-clustering matrix from cached files if present
def try_read_bootstrap():
    cand = [f"{OUT_DIR}/24_bootstrap_cocluster.csv",
            f"{OUT_DIR}/24_bootstrap_stability_per_patient.csv"]
    for p in cand:
        if os.path.exists(p):
            df = pd.read_csv(p, index_col=0)
            return df
    return None
boot_mat = try_read_bootstrap()

# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(DC+1.5, 8.6))
gs = gridspec.GridSpec(3, 3, figure=fig,
                       wspace=0.45, hspace=0.60,
                       height_ratios=[1.05, 1.0, 1.05],
                       width_ratios=[1.0, 1.0, 1.1])

DX_ORDER = ["NT", "IMNM", "IBM", "DM", "AS"]
DX_COL = {"NT":"#95A5A6","IMNM":"#2980B9","IBM":"#8E44AD",
          "DM":"#F39C12","AS":"#C0392B"}

# ── (a) Per-patient bootstrap within-IT stability (from cached CSV) ────────
ax = fig.add_subplot(gs[0, 0])
boot_p = pd.read_csv(f"{OUT_DIR}/24_bootstrap_stability_per_patient.csv")
boot_p = boot_p.sort_values("within_IT_stability")
palette_IT = {"IT1":"#C0392B","IT2":"#F39C12","IT3":"#2980B9"}
colors = [palette_IT.get(x, "#888") for x in boot_p["immunotype"]]
y_pos = np.arange(len(boot_p))
ax.barh(y_pos, boot_p["within_IT_stability"].values,
        color=colors, alpha=0.85, edgecolor="white")
for i, v in enumerate(boot_p["within_IT_stability"].values):
    ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=5)
ax.axvline(0.62, color="#222", lw=0.6, ls="--", label="mean 0.62")
ax.set_yticks(y_pos)
ax.set_yticklabels([f"P{int(p)} · {ab} · {it}"
                     for p, ab, it in zip(boot_p["pid"],
                                          boot_p["antibody"],
                                          boot_p["immunotype"])],
                    fontsize=5)
ax.set_xlim(0, 1.05)
ax.set_xlabel("within-immunotype co-cluster probability\n(B = 1,000 bootstrap)", fontsize=6)
ax.set_title("(a) Bootstrap within-IT stability\n(mean 0.62 ± 0.18)", fontsize=7, pad=4)
ax.legend(fontsize=5, frameon=False, loc="lower right")

# ── (b) LOO stability bar (ARI) ───────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
# use summary numbers: mean ARI 0.78, 86% label agreement; plot per-patient if
# cached CSV is available
loo_df = pd.read_csv(f"{OUT_DIR}/25_loo_stability.csv")
# sort by ARI (smallest first, so P01/P03 end up at top with red flag)
loo_df = loo_df.sort_values("ARI").reset_index(drop=True)
xs = np.arange(len(loo_df))
bar_h = 0.38
colors_it = {"IT1":"#C0392B","IT2":"#F39C12","IT3":"#2980B9"}
for i, r in loo_df.iterrows():
    col = colors_it.get(r["left_out_IT"], "#888")
    ax.barh(i - bar_h/2, r["ARI"], height=bar_h, color=col, alpha=0.85)
    ax.barh(i + bar_h/2, r["label_agreement"], height=bar_h,
            color=col, alpha=0.45)
ax.axvline(0.78, color="#222", lw=0.7, ls="--", label="mean ARI = 0.78")
ax.set_yticks(xs)
ax.set_yticklabels([f"drop P{int(p)} · {it}"
                     for p, it in zip(loo_df["left_out_pid"],
                                      loo_df["left_out_IT"])],
                    fontsize=5)
ax.set_xlim(0, 1.05)
ax.set_xlabel("ARI  (solid) · label agreement (faded)", fontsize=6)
ax.set_title("(b) Leave-one-out stability\n(mean ARI = 0.78, agreement = 86 %)", fontsize=7, pad=4)
ax.legend(fontsize=5, frameon=False, loc="lower right")

# ── (c) Top clinical correlations (LOO-stable) ─────────────────────────────
ax = fig.add_subplot(gs[0, 2])
# loo_corr has: pair, rho_full, rho_min, rho_max, sign_stable, q_full
y = np.arange(len(loo_corr))[::-1]   # reverse so largest-ρ is on top
for i, row in loo_corr.iterrows():
    col = "#1A7A4A" if row["rho_full"] > 0 else "#C0392B"
    yi  = y[i]
    ax.plot([row["rho_min"], row["rho_max"]], [yi, yi], color=col, lw=1.2, alpha=0.55)
    # LOO jitter dots
    loo_list = eval(row["loo_rhos"]) if isinstance(row["loo_rhos"], str) else row["loo_rhos"]
    ax.scatter(loo_list, [yi]*len(loo_list), s=10, c=col, alpha=0.35,
               edgecolor="none", zorder=2)
    ax.plot(row["rho_full"], yi, "D", color=col, markersize=6,
            markeredgecolor="white", markeredgewidth=0.6, zorder=4)
ax.axvline(0, color="#222", lw=0.6)
ax.set_yticks(y); ax.set_yticklabels(loo_corr["pair"].values, fontsize=5.5)
ax.set_xlabel("Spearman ρ (LOO + full)", fontsize=6.5)
ax.set_title("(c) Clinical correlations —\nLOO sign-stable", fontsize=7, pad=4)
ax.set_xlim(-1.05, 1.05)

# ── (d) External validation: IT1 score by diagnosis (VIOLIN + STRIP) ───────
ax = fig.add_subplot(gs[1, 0])
groups = [scores_df.loc[scores_df["diagnosis"]==g, "IT1_score"].values
          for g in DX_ORDER]
parts = ax.violinplot(groups, positions=np.arange(len(DX_ORDER)),
                      widths=0.82, showmeans=False, showmedians=True)
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(DX_COL[DX_ORDER[i]]); pc.set_alpha(0.55)
    pc.set_edgecolor(DX_COL[DX_ORDER[i]]); pc.set_linewidth(0.7)
for k in ["cmedians","cbars","cmins","cmaxes"]:
    if k in parts:
        parts[k].set_color("#333"); parts[k].set_lw(0.5)
rng = np.random.default_rng(0)
for i, g in enumerate(DX_ORDER):
    yv = scores_df.loc[scores_df["diagnosis"]==g, "IT1_score"].values
    xs = i + rng.uniform(-0.13, 0.13, size=len(yv))
    ax.scatter(xs, yv, s=6, c=DX_COL[g], alpha=0.8,
               edgecolor="white", linewidth=0.3, zorder=3)
ax.set_xticks(range(len(DX_ORDER)))
ax.set_xticklabels([f"{g}\nn={int((scores_df['diagnosis']==g).sum())}"
                    for g in DX_ORDER], fontsize=5.8)
ax.axhline(0, color="#bbb", lw=0.4, ls="--")
ax.set_ylabel("IT1 innate-hyperactive score", fontsize=6.5)

# pull KW p-value from stats table
kw_row = val_stats[val_stats["test"]=="KW 5-group"].iloc[0]
ax.set_title(f"(d) External validation: IT1 score in GSE220915\n"
             f"Kruskal-Wallis H={kw_row['stat']:.1f}, p={kw_row['p']:.1e}",
             fontsize=7, pad=4)
# significance bars vs AS
def pstr(p):
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 0.05: return "*"
    return "ns"
y_top = scores_df["IT1_score"].max()
ax.set_ylim(scores_df["IT1_score"].min() - 0.2, y_top + 0.6)
as_i = DX_ORDER.index("AS")
for g in ["NT","IMNM","IBM","DM"]:
    gi = DX_ORDER.index(g)
    prow = val_stats[val_stats["test"]==f"MW AS vs {g}"]
    if prow.empty: continue
    p = float(prow.iloc[0]["p"])
    y_bar = y_top + 0.10 + 0.12*abs(gi - as_i)
    ax.plot([gi, as_i], [y_bar, y_bar], color="#555", lw=0.5)
    ax.text((gi+as_i)/2, y_bar + 0.02, pstr(p),
            ha="center", va="bottom", fontsize=5.5)

# ── (e) Cliff's δ forest AS vs others ──────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
comp_order = ["NT","IMNM","IBM","DM"]
y = np.arange(len(comp_order))
for i, g in enumerate(comp_order):
    row = val_stats[val_stats["test"]==f"MW AS vs {g}"].iloc[0]
    d   = float(row["effect"])
    p   = float(row["p"])
    col = "#C0392B" if d > 0 else "#2980B9"
    ax.barh(i, d, color=col, alpha=0.85, edgecolor="white")
    ax.text(d + (0.015 if d > 0 else -0.015), i,
            f"  {pstr(p)}  p={p:.1e}",
            va="center", ha="left" if d > 0 else "right",
            fontsize=5.5, color="#333")
ax.axvline(0, color="#333", lw=0.5)
ax.set_yticks(y); ax.set_yticklabels([f"AS vs {g}" for g in comp_order], fontsize=6)
ax.set_xlabel("Cliff's δ  (+ = IT1 higher in AS)", fontsize=6.5)
ax.set_title("(e) Pairwise effect sizes\n(AS is significantly higher than NT and IMNM)",
             fontsize=7, pad=4)
ax.set_xlim(-1.05, 1.05)
ax.invert_yaxis()
ax.grid(axis="x", color="#eee", lw=0.3); ax.set_axisbelow(True)

# ── (f) Within-AS heterogeneity (n=18 ranked) ──────────────────────────────
ax = fig.add_subplot(gs[1, 2])
as_df = scores_df.loc[scores_df["diagnosis"]=="AS"].sort_values("IT1_score")
nt_df = scores_df.loc[scores_df["diagnosis"]=="NT"]
imnm_df = scores_df.loc[scores_df["diagnosis"]=="IMNM"]
ax.axhspan(np.percentile(nt_df["IT1_score"], 5),
           np.percentile(nt_df["IT1_score"], 95),
           color=DX_COL["NT"], alpha=0.16, lw=0, zorder=0,
           label="NT 5–95 %ile")
ax.axhspan(np.percentile(imnm_df["IT1_score"], 5),
           np.percentile(imnm_df["IT1_score"], 95),
           color=DX_COL["IMNM"], alpha=0.16, lw=0, zorder=0,
           label="IMNM 5–95 %ile")
xs = np.arange(1, len(as_df)+1)
ax.scatter(xs, as_df["IT1_score"].values, s=26, c=DX_COL["AS"],
           edgecolor="white", linewidth=0.4, zorder=3)
ax.plot(xs, as_df["IT1_score"].values, color=DX_COL["AS"],
        lw=0.55, alpha=0.55, zorder=2)
ax.axhline(0, color="#bbb", lw=0.4, ls="--")
ax.set_xlabel("18 AS patients (ranked by IT1 score)", fontsize=6.5)
ax.set_ylabel("IT1 score", fontsize=6.5)
ax.set_title("(f) Within-AS IT1 heterogeneity\n(18 AS patients span NT→IMNM→inflam.)",
             fontsize=7, pad=4)
ax.legend(fontsize=5, frameon=False, loc="lower right",
          handletextpad=0.3, borderpad=0.2)

# ── (g) Pathway enrichment in IT1-high AS ─────────────────────────────────
ax = fig.add_subplot(gs[2, 0:2])
pw = pd.read_csv(f"{OUT_DIR}/S5_AS_pathway_enrichment.csv")
# keep only those with any hit on hi-side (the programme of interest)
pw = pw.sort_values("hi_p")
y = np.arange(len(pw))[::-1]
ax.barh(y, -np.log10(pw["hi_p"].clip(lower=1e-20)),
        color="#C0392B", alpha=0.85, edgecolor="white")
ax.set_yticks(y); ax.set_yticklabels(pw["pathway"].values, fontsize=6.5)
ax.axvline(-np.log10(0.05), color="#333", ls="--", lw=0.6,
           label="nominal p=0.05")
ax.axvline(-np.log10(0.001), color="#333", ls=":", lw=0.6,
           label="p=0.001")
ax.set_xlabel(r"$-\log_{10}$ hypergeometric p   (up in IT1-high AS)", fontsize=6.5)
ax.set_title("(g) Within-AS IT1-high transcriptomic programme: "
             "IFN-γ + complement + cytolytic + myeloid "
             "(recapitulates the primary finding of GSE220915 source study)",
             fontsize=7, pad=4)
for i, r in pw.iterrows():
    yi = y[i] if isinstance(y, np.ndarray) else i
    xv = -np.log10(max(r["hi_p"], 1e-20))
    ax.text(xv + 0.3, yi,
            f"{r['hi_hits']}/{r['K_in_bg']}  (q={r['hi_qBH']:.1e})",
            fontsize=5.5, va="center", color="#555")
ax.legend(fontsize=5.5, frameon=False, loc="lower right")
ax.grid(axis="x", color="#eee", lw=0.3); ax.set_axisbelow(True)

# ── (h) Key numbers summary box ────────────────────────────────────────────
ax = fig.add_subplot(gs[2, 2])
ax.axis("off")
summary = (
    "Cross-cohort validation summary\n"
    "──────────────────────────────\n\n"
    "Discovery (CyTOF PBMC, n=10):\n"
    "  • 16 immune populations, 98,319 cells\n"
    "  • 3 immunotypes, stable under:\n"
    "     – bootstrap (within-IT 0.62)\n"
    "     – leave-one-out (ARI 0.78)\n\n"
    "External validation (bulk RNA-seq,\n"
    "GSE220915, n=165):\n"
    "  • 5-group KW H=68, p=6×10⁻¹⁴\n"
    "  • AS vs NT:   δ=+0.93  p=6×10⁻⁸\n"
    "  • AS vs IMNM: δ=+0.56  p=4×10⁻⁴\n\n"
    "Within-AS sub-structure (n=18):\n"
    "  • IT1 median split ↔ unsupervised\n"
    "     k=2 concordance = 0.83\n"
    "  • 3,266 DEGs at FDR<0.10\n"
    "  • IFN-γ (IRDS) p=6×10⁻¹⁷ (29/29)\n"
    "  • Complement p=2×10⁻⁸ (16/17)\n\n"
    "→ signature derived from n=10\n"
    "     PBMC CyTOF recapitulates the\n"
    "     primary muscle-RNA-seq finding"
)
ax.text(0, 1, summary, transform=ax.transAxes,
        va="top", ha="left", fontsize=6.5, family="monospace",
        color="#222",
        bbox=dict(facecolor="#FAFAFA", edgecolor="#DDD",
                  boxstyle="round,pad=0.6", lw=0.6))

fig.suptitle("Figure 6 — Robustness, clinical integration, and cross-cohort external validation",
             fontsize=9, y=1.00)

save(fig, "FIG6_combined_validation", OUT_DIR)
print("Saved FIG6_combined_validation.pdf/.png")
