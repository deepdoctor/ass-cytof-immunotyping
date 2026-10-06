"""
Native rebuild of Fig 4(b): clinical phenotype per immunotype, simplified
from 8 to 4 most-informative panels (CK, age, myositis%, DLCO%) per the
Figure-4 review.

The earlier 8-panel layout listed every clinical variable with its KW p-value;
because all p-values exceed 0.2 at n=10, the 8-panel grid effectively
amplified the figure's weakest signal. This rebuild keeps the four panels
that show the most directional, clinically interpretable contrast and
explicitly frames them as 'directional patterns; none individually
significant at this cohort size'.

Saves over 23_immunotype_clinical_severity.png so the existing composite
pipeline picks it up.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.stats import kruskal

OUT_DIR = "./cytof_output_nature"

plt.rcParams.update({
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":            10.0,
    "axes.titlesize":       11.0,
    "axes.labelsize":       10.0,
    "xtick.labelsize":      9.5,
    "ytick.labelsize":      9.5,
    "legend.fontsize":      9.0,
    "axes.linewidth":       1.0,
    "xtick.major.width":    0.9,
    "ytick.major.width":    0.9,
    "xtick.major.size":     3.5,
    "ytick.major.size":     3.5,
    "lines.linewidth":      1.4,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
    "savefig.dpi":          300,
    "figure.dpi":           300,
})

IT_COL = {"IT1": "#E67E22", "IT2": "#2980B9", "IT3": "#27AE60"}
IT_ORDER = ["IT1", "IT2", "IT3"]

clin = pd.read_csv(f"{OUT_DIR}/23_immunotype_clinical_table.csv")
clin = clin.drop(columns=[c for c in clin.columns if c == "name"])

def kw_p(values_per_group):
    """Kruskal–Wallis p; safe to NaNs."""
    cleaned = [v[~np.isnan(v)] for v in values_per_group]
    if any(len(v) < 2 for v in cleaned):
        return np.nan
    try:
        _, p = kruskal(*cleaned)
        return p
    except Exception:
        return np.nan

# ── figure layout: 1 row × 4 panels ───────────────────────────────────────
fig = plt.figure(figsize=(11.0, 4.0), facecolor="white")
gs = gridspec.GridSpec(1, 4, figure=fig,
                       left=0.06, right=0.985, top=0.83, bottom=0.20,
                       wspace=0.45)

# ── (b1) CK (log10 scale; box+jitter) ─────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
groups_ck = [clin.loc[clin["immunotype"] == it, "CK"].dropna().values
             for it in IT_ORDER]
positions = np.arange(len(IT_ORDER))
parts = ax.boxplot(groups_ck, positions=positions, widths=0.6,
                   patch_artist=True, showfliers=False,
                   whiskerprops=dict(lw=0.7, color="#444"),
                   capprops=dict(lw=0.7, color="#444"),
                   medianprops=dict(lw=1.0, color="#222"),
                   boxprops=dict(lw=0.5, edgecolor="#444"))
for box, it in zip(parts["boxes"], IT_ORDER):
    box.set_facecolor(IT_COL[it])
    box.set_alpha(0.55)
rng = np.random.default_rng(0)
for i, it in enumerate(IT_ORDER):
    vals = clin.loc[clin["immunotype"] == it, "CK"].dropna().values
    xs = i + rng.uniform(-0.15, 0.15, size=len(vals))
    ax.scatter(xs, vals, s=30, c=IT_COL[it], edgecolor="white",
               linewidth=0.5, zorder=3, alpha=0.95)
    # annotate the CK-extreme outlier in IT1 (P08)
    if it == "IT1" and vals.max() > 1000:
        idx_max = np.argmax(vals)
        ax.annotate("P08", (xs[idx_max], vals[idx_max]),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=8, color="#222")
ax.set_yscale("log")
ax.set_xticks(positions)
ax.set_xticklabels(IT_ORDER)
ax.set_ylabel("CK (U/L, log scale)", fontsize=10)
p = kw_p(groups_ck)
ax.set_title(f"CK\n(KW P = {p:.2f}; IT1 contains CK-extreme P08)",
             fontsize=10, pad=6)

# ── (b2) age ──────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
groups_age = [clin.loc[clin["immunotype"] == it, "age"].dropna().values
              for it in IT_ORDER]
parts = ax.boxplot(groups_age, positions=positions, widths=0.6,
                   patch_artist=True, showfliers=False,
                   whiskerprops=dict(lw=0.7, color="#444"),
                   capprops=dict(lw=0.7, color="#444"),
                   medianprops=dict(lw=1.0, color="#222"),
                   boxprops=dict(lw=0.5, edgecolor="#444"))
for box, it in zip(parts["boxes"], IT_ORDER):
    box.set_facecolor(IT_COL[it])
    box.set_alpha(0.55)
for i, it in enumerate(IT_ORDER):
    vals = clin.loc[clin["immunotype"] == it, "age"].dropna().values
    xs = i + rng.uniform(-0.15, 0.15, size=len(vals))
    ax.scatter(xs, vals, s=30, c=IT_COL[it], edgecolor="white",
               linewidth=0.5, zorder=3, alpha=0.95)
ax.set_xticks(positions); ax.set_xticklabels(IT_ORDER)
ax.set_ylabel("Age (years)", fontsize=10)
p = kw_p(groups_age)
ax.set_title(f"Age\n(KW P = {p:.2f}; IT3 oldest)",
             fontsize=10, pad=6)

# ── (b3) Myositis prevalence (% bar) ──────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
myo_pct = []
for it in IT_ORDER:
    sub = clin[clin["immunotype"] == it]["myositis"]
    myo_pct.append(100 * sub.mean())
bars = ax.bar(positions, myo_pct,
              color=[IT_COL[it] for it in IT_ORDER],
              alpha=0.78, edgecolor="white", linewidth=0.6, width=0.7)
for i, (bar, val) in enumerate(zip(bars, myo_pct)):
    ax.text(bar.get_x() + bar.get_width() / 2,
            val + 3, f"{val:.0f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(positions); ax.set_xticklabels(IT_ORDER)
ax.set_ylabel("Clinical myositis prevalence (%)", fontsize=10)
ax.set_ylim(0, 115)
# binary KW (proportions test approximation)
groups_myo = [clin.loc[clin["immunotype"] == it, "myositis"].dropna().values
              for it in IT_ORDER]
p = kw_p(groups_myo)
p_str = f"P = {p:.2f}" if not np.isnan(p) else "n/a"
ax.set_title(f"Myositis prevalence\n"
             f"(IT2 = 100 %; KW {p_str})",
             fontsize=10, pad=6)

# ── (b4) DLCO % ───────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 3])
groups_dlco = [clin.loc[clin["immunotype"] == it, "DLCO"].dropna().values
               for it in IT_ORDER]
parts = ax.boxplot(groups_dlco, positions=positions, widths=0.6,
                   patch_artist=True, showfliers=False,
                   whiskerprops=dict(lw=0.7, color="#444"),
                   capprops=dict(lw=0.7, color="#444"),
                   medianprops=dict(lw=1.0, color="#222"),
                   boxprops=dict(lw=0.5, edgecolor="#444"))
for box, it in zip(parts["boxes"], IT_ORDER):
    box.set_facecolor(IT_COL[it])
    box.set_alpha(0.55)
for i, it in enumerate(IT_ORDER):
    vals = clin.loc[clin["immunotype"] == it, "DLCO"].dropna().values
    if len(vals) == 0:
        continue
    xs = i + rng.uniform(-0.15, 0.15, size=len(vals))
    ax.scatter(xs, vals, s=30, c=IT_COL[it], edgecolor="white",
               linewidth=0.5, zorder=3, alpha=0.95)
ax.set_xticks(positions); ax.set_xticklabels(IT_ORDER)
ax.set_ylabel("DLCO (% predicted)", fontsize=10)
p = kw_p(groups_dlco)
p_str = f"KW P = {p:.2f}" if not np.isnan(p) else "n/a (insufficient n)"
ax.set_title(f"DLCO % predicted\n({p_str})",
             fontsize=10, pad=6)

# Suptitle indicating directional framing
fig.suptitle(
    "Clinical phenotype per immunotype "
    "(directional patterns at n = 10; full 8-panel matrix in Supp. Fig. S10)",
    fontsize=11, fontweight="bold", y=0.99,
)

# Save (overwrites the 8-panel composite source)
out_pdf = f"{OUT_DIR}/23_immunotype_clinical_severity.pdf"
out_png = f"{OUT_DIR}/23_immunotype_clinical_severity.png"
fig.savefig(out_pdf, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
fig.savefig(out_png, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
