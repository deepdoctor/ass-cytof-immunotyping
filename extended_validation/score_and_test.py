"""Signature scoring and effect-size statistics, matching Methods 2.6/2.9.

Per-sample signature score = mean of per-gene z-scores (z across all samples of
that cohort), i.e. the same construction used for the muscle cohorts.
"""
import numpy as np
from scipy import stats as sps

RNG = np.random.default_rng(20250726)


def zscore_genes(expr, min_var=1e-9):
    """Z-score each gene across samples; drop invariant genes."""
    m = expr.mean(axis=1)
    s = expr.std(axis=1, ddof=1)
    keep = s > min_var
    return expr.loc[keep].sub(m[keep], axis=0).div(s[keep], axis=0)


def signature_score(expr, symbols, resolver):
    """Mean per-gene z-score over the resolvable members of `symbols`."""
    found, missing = resolver(symbols, expr.index)
    z = zscore_genes(expr.loc[sorted(set(found.values()))])
    return z.mean(axis=0), sorted(found), missing


def cohens_d(a, b):
    """Cohen's d with pooled SD (a vs b; positive = a higher)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def hedges_g(a, b):
    d = cohens_d(a, b)
    n = len(a) + len(b)
    return d * (1 - 3 / (4 * n - 9))


def boot_ci_d(a, b, B=10000, alpha=0.05):
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = np.empty(B)
    for i in range(B):
        out[i] = cohens_d(RNG.choice(a, len(a), replace=True),
                          RNG.choice(b, len(b), replace=True))
    out = out[np.isfinite(out)]
    return np.percentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def auc(a, b):
    """AUC for separating a from b (0.5 = no separation)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    u = sps.mannwhitneyu(a, b, alternative="two-sided").statistic
    return u / (len(a) * len(b))


def contrast(scores, meta, case, ctrl="HC", B=10000):
    a = scores[meta[meta == case].index].values
    b = scores[meta[meta == ctrl].index].values
    mw2 = sps.mannwhitneyu(a, b, alternative="two-sided")
    mw1 = sps.mannwhitneyu(a, b, alternative="greater")
    lo, hi = boot_ci_d(a, b, B=B)
    return dict(case=case, ctrl=ctrl, n_case=len(a), n_ctrl=len(b),
                mean_case=a.mean(), mean_ctrl=b.mean(),
                d=cohens_d(a, b), g=hedges_g(a, b), d_lo=lo, d_hi=hi,
                auc=auc(a, b), p_two=mw2.pvalue, p_one=mw1.pvalue)


def dl_pool(ds, ns):
    """DerSimonian-Laird random-effects pooling of Cohen's d.

    ns: list of (n_case, n_ctrl) per study.
    Returns dict with pooled d, 95% CI, tau2, I2, Q, p_Q.
    """
    ds = np.asarray(ds, float)
    v = np.array([(n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2))
                  for d, (n1, n2) in zip(ds, ns)])
    w = 1 / v
    fixed = (w * ds).sum() / w.sum()
    Q = (w * (ds - fixed) ** 2).sum()
    k = len(ds)
    C = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0
    wr = 1 / (v + tau2)
    pooled = (wr * ds).sum() / wr.sum()
    se = np.sqrt(1 / wr.sum())
    I2 = max(0.0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0.0
    return dict(d=pooled, lo=pooled - 1.96 * se, hi=pooled + 1.96 * se,
                tau2=tau2, I2=I2, Q=Q, k=k,
                p_Q=1 - sps.chi2.cdf(Q, k - 1) if k > 1 else np.nan)


def gene_concordance(stats_df, symbols, resolver):
    """Direction concordance + competitive rank test on DESeq2 log2FC."""
    found, missing = resolver(symbols, stats_df.index)
    sub = stats_df.loc[sorted(set(found.values()))].dropna(subset=["log2FoldChange"])
    up = int((sub["log2FoldChange"] > 0).sum())
    n = len(sub)
    binom = sps.binomtest(up, n, 0.5, alternative="greater").pvalue if n else np.nan
    others = stats_df.drop(index=sub.index, errors="ignore").dropna(subset=["log2FoldChange"])
    comp = sps.mannwhitneyu(sub["log2FoldChange"], others["log2FoldChange"],
                            alternative="greater")
    return dict(n_tested=n, n_up=up, frac_up=up / n if n else np.nan,
                p_binom=binom, median_l2fc=float(sub["log2FoldChange"].median()),
                mean_l2fc=float(sub["log2FoldChange"].mean()),
                p_competitive=comp.pvalue, missing=missing,
                n_sig_up=int(((sub["log2FoldChange"] > 0) & (sub["padj"] < 0.05)).sum()),
                table=sub.sort_values("log2FoldChange", ascending=False))
