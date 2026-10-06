#!/usr/bin/env python3
"""Healthy-control analysis for the innate-hyperactive axis.

Run this once the control CyTOF samples exist. Input is a per-patient feature
matrix in exactly the Supp. Table ST1 layout:

    sample, pid, name, antibody, immunotype, <66 feature columns...>

with control rows appended and `immunotype` set to "HC". Optionally supply an
`age` column (or let the script pull ages from ST2 for the patients).

Implements the pre-specified plan in HC_ANALYSIS_PLAN.md:
  2.1  project controls onto the existing patient-derived axis + 3-outcome test
  2.2  HC-referenced z-scores (deviation from normal)
  2.3  per-compartment patient-vs-control effect sizes
  2.4  IT3 vs HC
  2.5  age / immunosenescence
  2.6  IT1 region vs HC on innate + activation modules

Usage:
    python analyze_with_hc.py --matrix matrix_with_hc.csv [--outdir hc_results]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

RNG = np.random.default_rng(20260727)
META = ["sample", "pid", "name", "antibody", "immunotype"]
INNATE = ["Classical Monocyte", "Inflammatory Monocyte", "Non-classical Monocyte",
          "NK cell", "mDC", "pDC"]


# --------------------------------------------------------------------- helpers
def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def boot_ci_d(a, b, B=10000, alpha=.05):
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = np.array([cohens_d(RNG.choice(a, len(a), True), RNG.choice(b, len(b), True))
                    for _ in range(B)])
    out = out[np.isfinite(out)]
    if out.size == 0:
        return (np.nan, np.nan)
    return tuple(np.percentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def bh(p):
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    q = np.full(p.shape, np.nan)
    pv = p[ok]
    n = pv.size
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


def contrast(a, b, label):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return dict(what=label, n_a=len(a), n_b=len(b), d=np.nan, lo=np.nan,
                    hi=np.nan, auc=np.nan, p=np.nan)
    u = sps.mannwhitneyu(a, b, alternative="two-sided")
    lo, hi = boot_ci_d(a, b)
    return dict(what=label, n_a=len(a), n_b=len(b), d=cohens_d(a, b), lo=lo, hi=hi,
                auc=u.statistic / (len(a) * len(b)), p=u.pvalue)


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True,
                    help="ST1-layout CSV/XLSX including HC rows (immunotype == 'HC')")
    ap.add_argument("--loadings", default=None,
                    help="axis loadings CSV from build_axis.py (default: recompute "
                         "from the patient rows in --matrix)")
    ap.add_argument("--outdir", default="hc_results")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = (pd.read_excel(args.matrix) if args.matrix.endswith((".xlsx", ".xls"))
          else pd.read_csv(args.matrix))
    missing = [c for c in META if c not in df.columns]
    if missing:
        sys.exit(f"input is missing required columns: {missing}")
    feats = [c for c in df.columns if c not in META and c != "age"]
    is_hc = df["immunotype"].astype(str).str.upper().eq("HC")
    pat, hc = df[~is_hc].reset_index(drop=True), df[is_hc].reset_index(drop=True)
    print(f"loaded {len(pat)} patients and {len(hc)} healthy controls, "
          f"{len(feats)} features")
    if len(hc) == 0:
        sys.exit("no rows with immunotype == 'HC' found")

    # ---- axis, derived from PATIENTS ONLY, then controls projected onto it ----
    mu, sd = pat[feats].mean(), pat[feats].std(ddof=1)
    sd = sd.replace(0, np.nan)
    keep = [f for f in feats if np.isfinite(sd[f]) and sd[f] > 0]
    Zp = ((pat[keep] - mu[keep]) / sd[keep])

    if args.loadings:
        w = pd.read_csv(args.loadings, index_col=0).iloc[:, 0].reindex(keep).fillna(0)
        load = w.values
        var1 = np.nan
    else:
        Xc = Zp.values - Zp.values.mean(0)
        _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        load = Vt[0]
        var1 = (s ** 2 / (s ** 2).sum())[0]
    # orient: higher = more innate / activated
    probe = [f for f in keep if f in INNATE or f in ("Activation_score", "Cytotoxic_score")]
    if probe and pd.Series(load, index=keep)[probe].mean() < 0:
        load = -load

    centre = Zp.values.mean(0)
    axis_p = (Zp.values - centre) @ load
    Zh = ((hc[keep] - mu[keep]) / sd[keep]).values          # patient-referenced scaling
    axis_h = (Zh - centre) @ load

    pat_axis = pd.Series(axis_p, index=pat["name"])
    hc_axis = pd.Series(axis_h, index=hc["name"])
    if np.isfinite(var1):
        print(f"axis: PC1 of the patient matrix, {var1*100:.1f}% of variance")

    # ---- 2.1 primary: where do controls sit? --------------------------------
    print("\n=== 2.1 axis position of healthy controls ===")
    rows = [contrast(pat_axis.values, hc_axis.values, "all patients vs HC")]
    for reg in sorted(pat["immunotype"].unique()):
        v = pat_axis[pat.loc[pat.immunotype == reg, "name"]].values
        rows.append(contrast(v, hc_axis.values, f"{reg} vs HC"))
    prim = pd.DataFrame(rows)
    print(prim.to_string(index=False, float_format=lambda x: f"{x:.3g}"))

    lo_reg = pat.loc[pat.immunotype == sorted(pat.immunotype.unique())[-1], "name"]
    lo_axis = pat_axis[lo_reg].mean() if len(lo_reg) else np.nan
    hi_axis = pat_axis.max()
    hc_med = np.median(hc_axis)
    print(f"\nHC median axis = {hc_med:+.2f}; lowest patient region mean = {lo_axis:+.2f}; "
          f"highest patient = {hi_axis:+.2f}")
    if hc_med <= lo_axis + .5 * abs(lo_axis or 1):
        verdict = ("OUTCOME A - controls sit at the quiescent end. The axis is a "
                   "disease-activity axis and the low region is near-normal.")
    elif hc_med < pat_axis.median():
        verdict = ("OUTCOME B - controls sit MID-AXIS, above the quiescent region. The "
                   "quiescent patients are suppressed BELOW normal; rewrite that region as "
                   "therapeutically suppressed, not constitutionally quiescent.")
    else:
        verdict = ("OUTCOME C - controls at or above the patient median. Check batch "
                   "anchors before interpreting; the axis may carry technical variance.")
    print("\n>>> " + verdict)

    # ---- 2.2 HC-referenced z-scores -----------------------------------------
    print("\n=== 2.2 deviation from normal (HC-referenced) ===")
    hmu, hsd = hc[keep].mean(), hc[keep].std(ddof=1).replace(0, np.nan)
    Zp_hc = ((pat[keep] - hmu) / hsd).dropna(axis=1)
    Xc2 = Zp_hc.values - Zp_hc.values.mean(0)
    _, _, Vt2 = np.linalg.svd(Xc2, full_matrices=False)
    l2 = Vt2[0]
    pr2 = [f for f in Zp_hc.columns if f in INNATE or f in ("Activation_score", "Cytotoxic_score")]
    if pr2 and pd.Series(l2, index=Zp_hc.columns)[pr2].mean() < 0:
        l2 = -l2
    axis_hcref = pd.Series(Xc2 @ l2, index=pat["name"])
    rho, pv = sps.spearmanr(pat_axis.values, axis_hcref.values)
    print(f"HC-referenced axis vs original axis: Spearman rho={rho:+.3f}, P={pv:.2g} "
          f"({'concordant' if rho > .8 else 'DISCORDANT - investigate'})")
    dev = pd.DataFrame({"patient": pat["name"], "axis": pat_axis.values,
                        "axis_HCreferenced": axis_hcref.values,
                        "mean_abs_z_vs_HC": Zp_hc.abs().mean(axis=1).values,
                        "immunotype": pat["immunotype"].values}
                       ).sort_values("axis", ascending=False)
    print(dev.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # ---- 2.3 per-compartment ------------------------------------------------
    print("\n=== 2.3 per-feature patient vs HC (top 20 by |d|) ===")
    recs = []
    for f in keep:
        r = contrast(pat[f].values, hc[f].values, f)
        recs.append(r)
    per = pd.DataFrame(recs)
    per["q"] = bh(per["p"].values)
    per["kind"] = ["proportion" if f in INNATE or not (f.startswith("MFI_") or f.endswith("_score"))
                   else ("MFI" if f.startswith("MFI_") else "score") for f in per["what"]]
    print(per.reindex(per.d.abs().sort_values(ascending=False).index)
             .head(20).to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    print(f"\nfeatures at BH-FDR<0.05: {(per.q < .05).sum()} / {len(per)}")

    # ---- 2.4 / 2.6 region-level ---------------------------------------------
    print("\n=== 2.4 & 2.6 region-level contrasts vs HC (effect sizes; under-powered) ===")
    mods = {"innate compartments": [f for f in INNATE if f in keep],
            "Activation_score": [f for f in ["Activation_score"] if f in keep],
            "Cytotoxic_score": [f for f in ["Cytotoxic_score"] if f in keep],
            "Naive_score": [f for f in ["Naive_score"] if f in keep]}
    rr = []
    for reg in sorted(pat["immunotype"].unique()):
        sel = pat.immunotype == reg
        for mname, cols in mods.items():
            if not cols:
                continue
            a = pat.loc[sel, cols].mean(axis=1).values
            b = hc[cols].mean(axis=1).values
            rr.append({**contrast(a, b, f"{reg} vs HC : {mname}")})
    print(pd.DataFrame(rr).to_string(index=False, float_format=lambda x: f"{x:.3g}"))

    # ---- 2.5 age ------------------------------------------------------------
    if "age" in df.columns and hc["age"].notna().sum() >= 4:
        print("\n=== 2.5 immunosenescence ===")
        r, p = sps.spearmanr(hc["age"], hc_axis.values)
        print(f"within HC: axis vs age Spearman rho={r:+.3f}, P={p:.3f} (n={len(hc)})")
        b, a0 = np.polyfit(hc["age"].astype(float), hc_axis.values, 1)
        if "age" in pat.columns:
            exp = a0 + b * pat["age"].astype(float)
            resid = pat_axis.values - exp.values
            out = pd.DataFrame({"patient": pat["name"], "age": pat["age"],
                                "axis": pat_axis.values,
                                "age_expected": exp.values, "residual": resid}
                               ).sort_values("residual", ascending=False)
            print(out.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    else:
        print("\n=== 2.5 skipped: no usable 'age' column for the controls ===")

    # ---- save ---------------------------------------------------------------
    prim.to_csv(f"{args.outdir}/primary_axis_vs_HC.csv", index=False)
    dev.to_csv(f"{args.outdir}/deviation_from_normal.csv", index=False)
    per.to_csv(f"{args.outdir}/per_feature_vs_HC.csv", index=False)
    pd.DataFrame(rr).to_csv(f"{args.outdir}/region_modules_vs_HC.csv", index=False)
    pd.DataFrame({"name": list(pat["name"]) + list(hc["name"]),
                  "group": ["patient"] * len(pat) + ["HC"] * len(hc),
                  "axis": list(pat_axis.values) + list(hc_axis.values)}
                 ).to_csv(f"{args.outdir}/axis_scores_all.csv", index=False)
    with open(f"{args.outdir}/VERDICT.txt", "w") as fh:
        fh.write(verdict + "\n")
    print(f"\nwrote results to {args.outdir}/")


if __name__ == "__main__":
    main()
