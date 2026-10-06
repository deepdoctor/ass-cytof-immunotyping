"""
Combined Supplementary Figure (Tier 2) covering:

  D. Patient-level outcome prediction model — leave-one-patient-out
     logistic regression on the 65-D feature vector predicting
     disease-activity (DA_score) high vs low; ROC + per-feature
     coefficients.
  E. WGCNA-style correlation network on the 65-D patient feature
     vector — module detection by hierarchical clustering of
     correlation distance.
  F. Mediation analysis: IT1 → CK → DA_score path. EXPLORATORY ONLY
     at n = 10; reported with explicit caveat that the cohort cannot
     identify a causal effect, but the framework is shown for follow-up.

Output: Supp_Fig_Tier2.pdf/png + per-component CSVs.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, Rectangle, FancyBboxPatch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import LeaveOneOut
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.stats import spearmanr, pearsonr

OUT_DIR = "./cytof_output_nature"

plt.rcParams.update({
    "font.family":"sans-serif",
    "font.sans-serif":["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":7.0, "axes.titlesize":7.5, "axes.labelsize":7.0,
    "xtick.labelsize":6.2, "ytick.labelsize":6.2, "legend.fontsize":6.0,
    "axes.linewidth":0.7, "axes.spines.top":False, "axes.spines.right":False,
    "pdf.fonttype":42, "ps.fonttype":42, "savefig.dpi":300, "figure.dpi":300,
})
PANEL_LBL = dict(fontsize=10, fontweight="bold", color="#000",
                 ha="left", va="bottom", family="sans-serif")

def _panel_label(ax, lbl, x=-0.18, y=1.06):
    ax.text(x, y, lbl, transform=ax.transAxes, **PANEL_LBL)

IT_COL = {"IT1": "#E67E22", "IT2": "#2980B9", "IT3": "#27AE60"}

# ─── load ──────────────────────────────────────────────────────────────────
feat = pd.read_csv(f"{OUT_DIR}/SuppTable_feature_matrix.csv")
feat = feat.drop(columns=[c for c in feat.columns if c == "name"])
feat["pid_label"] = feat["pid"].apply(lambda x: f"P{int(x):02d}")
clin = pd.read_csv(f"{OUT_DIR}/23_immunotype_clinical_table.csv")
clin = clin.drop(columns=[c for c in clin.columns if c == "name"])

meta_cols = ["sample","pid","antibody","immunotype","pid_label"]
X = feat.drop(columns=meta_cols).values
feat_names = list(feat.drop(columns=meta_cols).columns)
Xs = StandardScaler().fit_transform(X)

# Bind to clinical
clin_by_pid = clin.set_index("pid")
da = pd.Series([clin_by_pid.loc[p, "DA_score"] for p in feat["pid"]],
                index=feat["pid"].values)
ck = pd.Series([clin_by_pid.loc[p, "CK"] for p in feat["pid"]],
                index=feat["pid"].values)
da = da.astype(float); ck = ck.astype(float)
# Median split DA → high/low
da_med = da.median()
y_da = (da > da_med).astype(int).values  # 1 = high DA
print(f"DA median split: {y_da.sum()} high, {(1-y_da).sum()} low")


# ════════════════════════════════════════════════════════════════════════════
# D. Outcome prediction (LOO logistic regression on top-K features)
# ════════════════════════════════════════════════════════════════════════════
print("[D] Outcome prediction model …")
# At n=10 with 65 features, we MUST aggressively regularize. Use L1 logistic
# with leave-one-patient-out CV. Alternative: use a small subset of features.
n = len(Xs)
loo_pred  = np.zeros(n)
loo_proba = np.zeros(n)
sel_features = []
for i in range(n):
    mask_train = np.ones(n, dtype=bool); mask_train[i] = False
    Xtr, ytr = Xs[mask_train], y_da[mask_train]
    Xte = Xs[i:i+1]
    # L1 logistic with internal CV (n=9 → 3-fold inner)
    clf = LogisticRegressionCV(
        Cs=10, penalty="l1", solver="liblinear", cv=3,
        scoring="roc_auc", max_iter=5000, random_state=0,
    )
    clf.fit(Xtr, ytr)
    loo_proba[i] = clf.predict_proba(Xte)[0, 1]
    loo_pred[i]  = int(clf.predict(Xte)[0])
    # store which features were selected
    nonzero = np.where(clf.coef_.ravel() != 0)[0]
    sel_features.append(nonzero)

# Compute ROC curve from LOO probabilities
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, _ = roc_curve(y_da, loo_proba)
auc = roc_auc_score(y_da, loo_proba)
print(f"  LOO ROC-AUC = {auc:.3f}")

# Feature stability: how often each feature was selected across LOO folds
n_sel = np.zeros(len(feat_names), dtype=int)
for sel in sel_features:
    for j in sel:
        n_sel[j] += 1
stability = pd.DataFrame({"feature": feat_names,
                           "selection_freq": n_sel / n})\
              .sort_values("selection_freq", ascending=False)
stability.to_csv(f"{OUT_DIR}/30_supp_predict_feature_stability.csv", index=False)


# ════════════════════════════════════════════════════════════════════════════
# E. WGCNA-style correlation network on 65-D features
# ════════════════════════════════════════════════════════════════════════════
print("[E] Feature correlation network …")
Xdf = pd.DataFrame(Xs, columns=feat_names)
# Pearson correlation across patients (n=10 — exploratory)
corr = Xdf.corr(method="pearson")
# distance = 1 - |corr|
dist = 1 - corr.abs()
# Hierarchical clustering on distance (avg linkage)
import scipy.spatial.distance as ssd
condensed = ssd.squareform(dist.values, checks=False)
Z_feat = linkage(condensed, method="average")
modules = fcluster(Z_feat, t=0.4, criterion="distance")
n_modules = len(np.unique(modules))
print(f"  Detected {n_modules} feature modules (cut at 1-|r|=0.4)")
mod_df = pd.DataFrame({"feature": feat_names, "module_E": modules})\
          .sort_values("module_E")
mod_df.to_csv(f"{OUT_DIR}/30_supp_feature_modules.csv", index=False)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE — three panels: a (ROC), b (feature stability), c (correlation)
# Mediation panel removed: at n = 8 it cannot identify a causal pathway
# and the diagram is therefore not scientifically informative.
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(7.4, 8.0))
gs = gridspec.GridSpec(
    2, 2, figure=fig,
    left=0.080, right=0.965, top=0.940, bottom=0.060,
    wspace=0.42, hspace=0.50,
    height_ratios=[1.0, 1.9],   # taller correlation heatmap
)

# ── (D1) ROC curve ─────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
ax.plot([0, 1], [0, 1], color="#888", lw=0.5, ls="--")
ax.plot(fpr, tpr, color="#C0392B", lw=1.4)
ax.fill_between(fpr, tpr, alpha=0.15, color="#C0392B")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.set_xlabel("False positive rate", fontsize=7.0)
ax.set_ylabel("True positive rate", fontsize=7.0)
ax.set_title(f"LOO logistic regression\n"
             f"DA-high vs DA-low  (LOO AUC = {auc:.2f}, n = {n})",
             fontsize=7.5, pad=4)
_panel_label(ax, "a")


# ── (D2) Feature selection stability bar ──────────────────────────────
ax = fig.add_subplot(gs[0, 1])
top_stab = stability.head(12)
y_pos = np.arange(len(top_stab))[::-1]
bars = ax.barh(y_pos, top_stab["selection_freq"].values,
                color="#2980B9", alpha=0.85, edgecolor="white")
ax.set_yticks(y_pos)
ax.set_yticklabels(top_stab["feature"].values, fontsize=6.0)
for i, v in enumerate(top_stab["selection_freq"].values):
    ax.text(v + 0.02, y_pos[i], f"{v:.0%}", va="center",
            fontsize=5.5)
ax.set_xlim(0, 1.15)
ax.set_xlabel("LOO selection frequency", fontsize=7.0)
ax.set_title("Top features by LOO stability\n(L1 logistic, 10 LOO folds)",
             fontsize=7.5, pad=4)
_panel_label(ax, "b")


# ── (E) Feature correlation network heatmap with module annotation ───
ax = fig.add_subplot(gs[1, :])
order = leaves_list(Z_feat)
corr_o = corr.iloc[order, order]
mod_o = modules[order]

from matplotlib.colors import LinearSegmentedColormap
RB = LinearSegmentedColormap.from_list("rb",
    ["#1A3A6B","#2980B9","#FFFFFF","#E67E22","#C0392B"], N=256)
_nr_t2, _nc_t2 = corr_o.shape
im = ax.pcolormesh(np.arange(_nc_t2 + 1) - 0.5, np.arange(_nr_t2 + 1) - 0.5,
                    corr_o.values, cmap=RB, vmin=-1, vmax=1,
                    edgecolors="none", rasterized=False, shading="flat")
ax.invert_yaxis(); ax.set_aspect("auto")
ax.set_xticks(range(len(order)))
ax.set_yticks(range(len(order)))
ax.set_xticklabels([feat_names[i] for i in order], rotation=90, fontsize=3.6)
ax.set_yticklabels([feat_names[i] for i in order], fontsize=3.6)
ax.tick_params(axis="both", which="major", pad=1, length=1.5, width=0.4)
# module annotation strip on top
mod_palette = ["#C0392B","#2980B9","#27AE60","#F39C12","#7D3C98",
               "#E67E22","#1A7A4A","#7B241C","#888888"]
strip_y = -1.5
for i, m in enumerate(mod_o):
    ax.add_patch(Rectangle((i - 0.5, strip_y), 1, 1,
                            facecolor=mod_palette[(int(m) - 1) % len(mod_palette)],
                            edgecolor="white", lw=0.2, clip_on=False))
cbar = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.012, shrink=0.85)
cbar.set_label("Pearson r (feature × feature)", fontsize=6.0)
cbar.ax.tick_params(labelsize=5.5); cbar.outline.set_linewidth(0.4)
ax.set_title(f"WGCNA-style feature correlation network "
             f"({n_modules} modules at d = 1−|r| = 0.4; n=10 patients, exploratory)",
             fontsize=7.0, pad=4)
_panel_label(ax, "c", x=-0.06, y=1.025)


# ─── save ─────────────────────────────────────────────────────────────────
# Figure title removed (journal convention).
out_pdf = f"{OUT_DIR}/S7_tier2_ancillary.pdf"
out_png = f"{OUT_DIR}/S7_tier2_ancillary.png"
fig.savefig(out_pdf, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
fig.savefig(out_png, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
