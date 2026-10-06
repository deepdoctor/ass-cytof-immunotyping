"""
Native rebuild of Fig 5(c): patient-level cluster-bootstrap Cohen's d
forest plot for EJ vs PL-7, top 6 per direction.

Fixes vs prior render:
  - x-axis was -4 to +4 with all data in [-2, +2] → constrained to ±2.5
    so the data fills the panel.
  - composite labels 'GranzymeB · CD8 Naive T' were rendered too small
    → bumped to 9 pt with two-line wrap, bolded marker name.
  - 95 % CI bar + diamond at point estimate, asterisk for FDR < 0.10.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "./cytof_output_nature"

plt.rcParams.update({
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":            10.0,
    "axes.titlesize":       12.0,
    "axes.labelsize":       10.0,
    "xtick.labelsize":      9.5,
    "ytick.labelsize":      9.0,
    "axes.linewidth":       0.9,
    "xtick.major.width":    0.7,
    "ytick.major.width":    0.7,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
    "savefig.dpi":          300,
    "figure.dpi":           300,
})

EJ_COL  = "#1A7A4A"
PL7_COL = "#7D3C98"

df = pd.read_csv(f"{OUT_DIR}/27b_DE_bootstrap_CI.csv")
print(f"Loaded {len(df)} (cell_type × marker) pairs")

# Top 6 by direction (largest |d| with CI not crossing zero)
ci_clear = ~df["crosses_zero"].astype(bool)
df_clear = df[ci_clear].copy()
top_ej  = df_clear[df_clear["d"] > 0].nlargest(6, "d")
top_pl7 = df_clear[df_clear["d"] < 0].nsmallest(6, "d")
sub = pd.concat([top_ej, top_pl7]).reset_index(drop=True)
sub = sub.sort_values("d").reset_index(drop=True)

n = len(sub)
y = np.arange(n)

fig, ax = plt.subplots(figsize=(7.4, 0.42 * n + 1.4))

for i, r in sub.iterrows():
    col = EJ_COL if r["d"] > 0 else PL7_COL
    # CI bar
    ax.plot([r["ci_low"], r["ci_high"]], [y[i], y[i]],
            color=col, lw=1.6, alpha=0.85, solid_capstyle="butt")
    # caps
    cap_h = 0.18
    for x_c in [r["ci_low"], r["ci_high"]]:
        ax.plot([x_c, x_c], [y[i] - cap_h, y[i] + cap_h],
                color=col, lw=1.0, alpha=0.85)
    # diamond at point estimate
    ax.plot(r["d"], y[i], "D", color=col, markersize=8,
            markeredgecolor="white", markeredgewidth=0.6, zorder=4)
    # FDR significance asterisk
    if r["q_fdr"] < 0.10:
        stars = "*" if r["q_fdr"] >= 0.01 else ("**" if r["q_fdr"] >= 0.001 else "***")
        ax.text(r["ci_high"] + 0.05, y[i], stars,
                ha="left", va="center", fontsize=11, color=col,
                fontweight="bold")

# y-axis labels: marker · cell type (plain text, larger font)
labels = [f"{r['marker']}  ·  {r['cell_type']}" for _, r in sub.iterrows()]
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9.5)

# x-axis constrained to ±2.5
ax.axvline(0, color="#222", lw=0.7, zorder=1)
ax.set_xlim(-2.5, 2.5)
ax.set_xlabel("Cohen's d  (EJ − PL-7)  [patient-level cluster bootstrap, B = 2,000]",
              fontsize=10)

# direction headers above
ax.text(-2.4, n + 0.1, "↑ in PL-7", ha="left", va="bottom",
        fontsize=10, color=PL7_COL, fontweight="bold")
ax.text(2.4, n + 0.1, "↑ in EJ", ha="right", va="bottom",
        fontsize=10, color=EJ_COL, fontweight="bold")

ax.set_title(
    "Patient-level cluster-bootstrap effect sizes (EJ vs PL-7)\n"
    "top 6 per direction; * FDR < 0.10, ** < 0.01, *** < 0.001",
    fontsize=11, pad=8,
)
ax.grid(axis="x", color="#eee", lw=0.4)
ax.set_axisbelow(True)
ax.set_ylim(-0.6, n - 0.4 + 0.6)

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/27b_DE_bootstrap_CI_forest.pdf",
            dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/27b_DE_bootstrap_CI_forest.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR}/27b_DE_bootstrap_CI_forest.png")
