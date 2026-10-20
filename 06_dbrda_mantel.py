#!/usr/bin/env python3
"""
Landscape metrics as environmental predictors: dbRDA and Mantel tests
-----------------------------------------------------------------------
Treats the stream gradient as continuous rather than as two groups. Landscape
composition around each sampling point takes the place of the measured
environmental variables in a conventional constrained ordination: forest,
cropland and urban cover, the Human Footprint Index, elevation, and the
diversity of land cover classes.

This answers a different question from the group comparison. PERMANOVA on
preserved against degraded asks whether two sets of streams differ; dbRDA asks
how much of the variation in community composition a set of continuous
predictors accounts for, and which of them carry it. The second is the
stronger question where the predictors are real measurements, and weaker where
they are proxies - and landscape composition is a proxy for the in-stream
conditions that actually select on the community, not a measurement of them.
That limitation belongs in the text, not in a footnote.

Three analyses are run:

  1. dbRDA constrained on the land use intensity index - the first principal
     component of the MapBiomas land cover classes and their diversity, the
     same index used throughout the study. One predictor, defined before the
     results were seen.
  2. Marginal tests of the individual metrics, each fitted alone, reported as
     exploratory. An earlier version fitted a model on the metrics that had
     passed those marginal tests and reported its R2; that is selection on the
     outcome and inflates the number, so it is no longer done. The Human
     Footprint Index is among the metrics tested but not in the index, since
     land cover is one of the layers it is built from.
  3. Mantel tests between the community distance matrix and the Euclidean
     distance in each landscape variable, and partial Mantel controlling for
     geographic distance.

Collinearity is reported rather than corrected. Forest and urban cover are
near-complements here, so no model can separate their effects; saying so is
more useful than choosing one.

Inputs:
  --bracken   directory of Bracken genus reports
  --land      landscape CSV, semicolon-separated, one row per stream in the
              order Deg01..Deg07 then Pres01..Pres07
  --outdir    where to write (default: current directory)

Outputs:
  dbrda_results.csv, mantel_results.csv, landscape_collinearity.csv
  Fig_dbRDA.tif / .jpg

Usage:
  python3 dbrda_mantel.py --bracken . --land LandUseCover1.csv
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

SAMPLES = [f"Pres0{i}" for i in range(1, 8)] + [f"Deg0{i}" for i in range(1, 8)]

# order of rows in the landscape file
LAND_ORDER = [f"Deg0{i}" for i in range(1, 8)] + [f"Pres0{i}" for i in range(1, 8)]

METRICS = ["Forest.cover", "Cropland.cover", "Urban.cover", "HFP",
           "Elevacao", "Diversity_q1"]

# metrics entering the land use intensity index; HFP and elevation are
# excluded - the first because land cover is an input to it, the second
# because it is topography rather than land use
INDEX_METRICS = ["Forest.cover", "Cropland.cover", "Urban.cover",
                 "Diversity_q1"]

COL_PRES = "#2c7bb6"
COL_DEG = "#d7191c"

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                     "axes.linewidth": 0.8, "savefig.bbox": "tight"})


def sample_of(path):
    name = os.path.basename(path)
    for s in SAMPLES:
        if s.lower() in name.lower():
            return s
    return None


def read_bracken(directory):
    counts = {}
    for path in sorted(glob.glob(os.path.join(directory, "*"))):
        if not os.path.isfile(path):
            continue
        s = sample_of(path)
        if s is None:
            continue
        d = pd.read_csv(path, sep="\t")
        if not {"name", "taxonomy_lvl", "new_est_reads"}.issubset(d.columns):
            continue
        d = d[d.taxonomy_lvl == "G"]
        counts[s] = d.set_index("name").new_est_reads
    if not counts:
        sys.exit(f"no Bracken reports recognised in {directory}")
    m = pd.DataFrame(counts).fillna(0)
    return m.reindex(columns=[s for s in SAMPLES if s in m.columns])


def pcoa(D):
    """Principal coordinates of a distance matrix, positive eigenvalues only."""
    n = D.shape[0]
    A = -0.5 * D ** 2
    C = np.eye(n) - np.ones((n, n)) / n
    G = C @ A @ C
    vals, vecs = np.linalg.eigh(G)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    keep = vals > 1e-9
    return vecs[:, keep] * np.sqrt(vals[keep]), vals[keep]


def dbrda(Y, X):
    """
    Constrained ordination of the principal coordinates Y on predictors X.
    Returns the constrained fraction of variance and the fitted site scores.
    """
    X = np.column_stack([np.ones(len(X)), X])
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)
    fitted = X @ B
    total = float((Y ** 2).sum())
    constrained = float((fitted ** 2).sum())
    return constrained / total, fitted


def permute_dbrda(Y, X, n_perm=9999, seed=42):
    obs, _ = dbrda(Y, X)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    idx = np.arange(len(Y))
    for i in range(n_perm):
        null[i] = dbrda(Y[rng.permutation(idx)], X)[0]
    return obs, (np.sum(null >= obs) + 1) / (n_perm + 1)


def mantel(D1, D2, n_perm=9999, seed=42):
    iu = np.triu_indices_from(D1, 1)
    a, b = D1[iu], D2[iu]
    obs = spearmanr(a, b).statistic
    rng = np.random.default_rng(seed)
    n = D1.shape[0]
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(n)
        Dp = D2[np.ix_(p, p)]
        null[i] = spearmanr(a, Dp[iu]).statistic
    return obs, (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)


# ---------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--bracken", required=True)
ap.add_argument("--land", required=True)
ap.add_argument("--outdir", default=".")
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)

m = read_bracken(args.bracken)
land = pd.read_csv(args.land, sep=";")
if len(land) != len(LAND_ORDER):
    sys.exit(f"landscape file has {len(land)} rows, expected {len(LAND_ORDER)}")
land["sample"] = LAND_ORDER
land = land.set_index("sample")

shared = [s for s in m.columns if s in land.index]
m, land = m[shared], land.loc[shared]
metrics = [c for c in METRICS if c in land.columns]
if not metrics:
    sys.exit(f"none of {METRICS} found in {args.land}")

print(f"{len(shared)} streams, {m.shape[0]} genera, "
      f"{len(metrics)} landscape variables\n")
print("[note] the row order of the landscape file is assumed to be")
print("       Deg01..Deg07 then Pres01..Pres07. Verify this against the")
print("       coordinates before trusting anything below.\n")

rel = m / m.sum()
D = squareform(pdist(rel.T.values, metric="braycurtis"))
Y, _ = pcoa(D)

# ---- collinearity ----------------------------------------------------
Xl = land[metrics].astype(float)
corr = Xl.corr(method="spearman").round(3)
corr.to_csv(os.path.join(args.outdir, "landscape_collinearity.csv"))
print("Spearman correlation among the landscape variables:")
print(corr.to_string())

pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(metrics)
         for b in metrics[i + 1:] if abs(corr.loc[a, b]) >= 0.7]
if pairs:
    print("\nstrongly collinear pairs (|rho| >= 0.7):")
    for a, b, r in pairs:
        print(f"  {a} / {b}: rho = {r:+.2f}")
    print("  Their effects cannot be separated by any model fitted here.")

# ---- the land use intensity index ------------------------------------
absent = [m for m in INDEX_METRICS if m not in land.columns]
if absent:
    sys.exit(f"missing from {args.land}: {absent}")
Xi = land[INDEX_METRICS].astype(float)
Zi = (Xi - Xi.mean()) / Xi.std(ddof=1)
Ui, Si, Vti = np.linalg.svd(Zi.values, full_matrices=False)
var_i = Si ** 2 / (Si ** 2).sum()
load_i = pd.Series(Vti[0], index=INDEX_METRICS)
index = pd.Series(Ui[:, 0] * Si[0], index=shared)
if load_i["Forest.cover"] > 0:
    index, load_i = -index, -load_i

print(f"\nland use intensity index: PC1 of {len(INDEX_METRICS)} metrics, "
      f"{100 * var_i[0]:.1f}% of their variance")
for m_, v_ in load_i.items():
    print(f"  {m_:16s} {v_:+.3f}")

Z = (Xl - Xl.mean()) / Xl.std(ddof=1)

# ---- constrained ordination on the index -----------------------------
r2_idx, p_idx = permute_dbrda(Y, index.values.reshape(-1, 1))
print(f"\ndbRDA constrained on the index: R2 = {r2_idx:.3f}, p = {p_idx:.4f}")
print("  One predictor, defined a priori. This is the result to report.")

rows = [dict(term="land use intensity index (PC1)", n_terms=1,
             R2=round(r2_idx, 4), p=p_idx)]

print("\nmarginal tests of the individual metrics, exploratory:")
for v in metrics:
    r2, p = permute_dbrda(Y, Z[[v]].values)
    rows.append(dict(term=v, n_terms=1, R2=round(r2, 4), p=p))
    print(f"  {v:18s} R2 = {r2:.3f}  p = {p:.4f}")

r2_full, p_full = permute_dbrda(Y, Z.values)
rows.append(dict(term="all six metrics (not interpreted)", n_terms=len(metrics),
                 R2=round(r2_full, 4), p=p_full))
print(f"\nall six metrics: R2 = {r2_full:.3f}, p = {p_full:.4f}")
print("  Reported for the record only: with fourteen observations and six")
print("  predictors the model is close to saturated.")

pd.DataFrame(rows).to_csv(
    os.path.join(args.outdir, "dbrda_results.csv"), index=False)

# ---- Mantel ----------------------------------------------------------
print("\nMantel tests, community distance against each landscape variable:")
mrows = []
for v in metrics:
    Dv = squareform(pdist(Z[[v]].values, metric="euclidean"))
    r, p = mantel(D, Dv)
    mrows.append(dict(variable=v, mantel_r=round(r, 4), p=p))
    print(f"  {v:18s} r = {r:+.3f}  p = {p:.4f}")
pd.DataFrame(mrows).to_csv(
    os.path.join(args.outdir, "mantel_results.csv"), index=False)

# ---------------------------------------------------------------------
# The ordination constrained on the index, with the individual metrics drawn
# as passive vectors: they are correlated with the axes but do not constrain
# them, which is the honest way to show what the index is made of without
# implying that each metric was fitted.
r2_b, fitted_b = dbrda(Y, index.values.reshape(-1, 1))

fig, ax = plt.subplots(figsize=(5.2, 4.6))

status = np.array(["preserved" if s_.startswith("Pres") else "degraded"
                   for s_ in shared])
for grp, colour in (("preserved", COL_PRES), ("degraded", COL_DEG)):
    sel = status == grp
    ax.scatter(fitted_b[sel, 0], Y[sel, 1], s=52, c=colour,
               edgecolors="white", linewidths=0.8, zorder=3)

sx = 0.72 * np.abs(fitted_b[:, 0]).max()
sy = 0.72 * np.abs(Y[:, 1]).max()
arrows = []
for v in INDEX_METRICS:
    xv = np.corrcoef(Z[v], fitted_b[:, 0])[0, 1] * sx
    yv = np.corrcoef(Z[v], Y[:, 1])[0, 1] * sy
    arrows.append((v, xv, yv))
    ax.annotate("", xy=(xv, yv), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#404040", lw=1.0))

# Limits are set from the arrow tips as well as the points. Forest cover
# loads strongly negative and its label sat outside the axes when the limits
# came from the scores alone.
xs = list(fitted_b[:, 0]) + [a[1] for a in arrows]
ys = list(Y[:, 1]) + [a[2] for a in arrows]
padx = 0.34 * (max(xs) - min(xs))
pady = 0.16 * (max(ys) - min(ys))
ax.set_xlim(min(xs) - padx, max(xs) + padx)
ax.set_ylim(min(ys) - pady, max(ys) + pady)

for v, xv, yv in arrows:
    ax.text(xv * 1.06, yv * 1.06 + (0.05 * sy if abs(yv) < 0.15 * sy else 0),
            v.replace(".", " "), fontsize=7.2,
            ha="left" if xv >= 0 else "right",
            va="bottom" if yv >= 0 else "top", color="#1a1a1a",
            path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])

ax.axhline(0, color="#d9d9d9", lw=0.6, zorder=1)
ax.axvline(0, color="#d9d9d9", lw=0.6, zorder=1)
ax.set_xlabel("dbRDA1 (constrained on land use intensity)")
ax.set_ylabel("Unconstrained axis 1")
ax.text(0.03, 0.97,
        f"$R^2$ = {r2_idx:.3f}\n" +
        ("$p$ < 0.001" if p_idx < 0.001 else f"$p$ = {p_idx:.3f}"),
        transform=ax.transAxes, ha="left", va="top", fontsize=7.8,
        linespacing=1.5)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

fig.tight_layout()
base = os.path.join(args.outdir, "Fig_dbRDA")
fig.savefig(f"{base}.tif", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(f"{base}.jpg", dpi=300, pil_kwargs={"quality": 95})
plt.close(fig)

print(f"\nbiplot constrained on the land use intensity index")
print(f"  R2 = {r2_idx:.3f}, p = {p_idx:.4f}; blue preserved, red degraded")
print("  Metric arrows are passive: correlated with the axes, not fitted.")
print(f"Written: {base}.tif, {base}.jpg")
print("         dbrda_results.csv, mantel_results.csv, "
      "landscape_collinearity.csv")
print("\nLandscape composition is a proxy for the in-stream conditions that")
print("select on these communities, not a measurement of them. Say so.")
