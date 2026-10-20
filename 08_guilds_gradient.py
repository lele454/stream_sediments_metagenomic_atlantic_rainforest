#!/usr/bin/env python3
"""
Do the guilds that differ between groups also track the landscape gradient?
-----------------------------------------------------------------------------
Every guild is tested, against a single composite index of land use intensity.

Testing all guilds against all six landscape metrics is 66 comparisons, and
after correction nothing survives even where the correlation reaches
rho = -0.89: the penalty, not the evidence, decides the outcome. Restricting
the predictor rather than the guilds keeps the correction affordable without
deciding in advance which guilds are allowed to respond.

The predictor is the first principal component of the landscape metrics rather
than one of them. Forest, cropland and urban cover and the Human Footprint
Index correlate at |rho| >= 0.73 and measure one axis three ways, so any single
choice among them is arbitrary - and a choice made after seeing which metric
gives the lowest p value is worse than arbitrary. The first component takes
the shared axis without privileging a metric, and its sign is set so the index
increases with land use intensity.

Elevation is excluded from the index. It is a topographic variable, not a
measure of land use, and belongs in a model of land use intensity no more than
it belongs in a definition of it. That excluding it also raises the variance
explained is a consequence rather than the reason.

Correlations for the individual metrics are written to file, uncorrected, for
reference.

The line through each panel is a Theil-Sen fit: the median of the slopes of
every pair of points. Like Spearman it is rank-based and resistant to a single
outlying stream, so the line and the statistic beside it rest on the same
assumptions - which a least-squares line would not, and which a running median
does not either. A running median also wanders up and down between points and
invites the reader to see structure in seven samples that is not there.

Inputs:
  --guilds    guilds_redox.csv from fig_guilds_redox.py
  --tests     guilds_redox_tests.csv from the same script
  --land      landscape CSV
  --metric    single landscape variable to test instead of the PCA index,
              for sensitivity checks; the index is used when omitted
  --outdir    where to write (default: current directory)

Outputs:
  guilds_gradient.csv                 every guild against the metric
  guilds_vs_landscape_all.csv         every guild against every metric,
                                      uncorrected, for reference
  Fig_guilds_gradient.tif / .jpg

Usage:
  python3 guilds_gradient.py --guilds guilds_redox.csv \\
      --tests guilds_redox_tests.csv --land LandUseCover1.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, theilslopes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

LAND_ORDER = [f"Deg0{i}" for i in range(1, 8)] + [f"Pres0{i}" for i in range(1, 8)]
METRICS = ["Forest.cover", "Cropland.cover", "Urban.cover", "HFP",
           "Diversity_q1", "Elevacao"]

COL_PRES = "#2c7bb6"
COL_DEG = "#d7191c"

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                     "axes.linewidth": 0.8, "savefig.bbox": "tight"})


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.empty_like(p)
    r = p[o] * len(p) / (np.arange(len(p)) + 1)
    q[o] = np.minimum.accumulate(r[::-1])[::-1]
    return np.clip(q, 0, 1)


ap = argparse.ArgumentParser()
ap.add_argument("--guilds", required=True)
ap.add_argument("--tests", required=True)
ap.add_argument("--land", required=True)
ap.add_argument("--metric", default=None)
ap.add_argument("--all-width", type=float, default=None,
                help="width in inches of Fig_guilds_gradient_all, so it can "
                     "be generated to match a panel it will be stacked with "
                     "instead of being rescaled afterwards")
ap.add_argument("--outdir", default=".")
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)

g = pd.read_csv(args.guilds, index_col=0)
t = pd.read_csv(args.tests, index_col=0)
land = pd.read_csv(args.land, sep=";")
if len(land) != len(LAND_ORDER):
    sys.exit(f"landscape file has {len(land)} rows, expected {len(LAND_ORDER)}")
land["sample"] = LAND_ORDER
land = land.set_index("sample")
# Metrics entering the index. The Human Footprint Index is deliberately not
# among them: it is itself built from land cover, population density, roads,
# night lights and accessibility, so cropland and urban cover are inputs to it
# rather than variables independent of it. Including all four would weight the
# same information twice and inflate the share of variance the first component
# appears to explain. The index is therefore built from the MapBiomas land
# cover classes and the diversity of those classes - quantities measured
# directly and auditable - and the Human Footprint Index is used afterwards to
# check the index against a published global one.
INDEX_METRICS = ["Forest.cover", "Cropland.cover", "Urban.cover",
                 "Diversity_q1"]
VALIDATION_METRIC = "HFP"
if args.metric is not None and args.metric not in land.columns:
    sys.exit(f"{args.metric} not in {args.land}")

shared = [c for c in g.columns if c in land.index]
if len(shared) < 6:
    sys.exit(f"only {len(shared)} samples shared between the two files")

# ---- the land use intensity index -----------------------------------
if args.metric is None:
    missing = [m for m in INDEX_METRICS if m not in land.columns]
    if missing:
        sys.exit(f"missing from {args.land}: {missing}")
    X = land.loc[shared, INDEX_METRICS].astype(float)
    Zs = (X - X.mean()) / X.std(ddof=1)
    U, S, Vt = np.linalg.svd(Zs.values, full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    loadings = pd.Series(Vt[0], index=INDEX_METRICS)
    index = pd.Series(U[:, 0] * S[0], index=shared)
    pc2 = pd.Series(U[:, 1] * S[1], index=shared)
    load2 = pd.Series(Vt[1], index=INDEX_METRICS)
    # orient so the index rises with land use intensity, which is the
    # direction the sign of the eigenvector does not fix
    if loadings["Forest.cover"] > 0:
        index, loadings = -index, -loadings
    predictor_name = "Land use intensity (PC1)"
    print(f"land use intensity index from {len(INDEX_METRICS)} metrics: "
          f"PC1 explains {100 * var[0]:.1f}% (PC2 {100 * var[1]:.1f}%)")
    print("loadings:")
    for m, v in loadings.items():
        print(f"  {m:16s} {v:+.3f}")
    pd.DataFrame({"sample": shared, "PC1": index.values}).assign(
        **{m: X[m].values for m in INDEX_METRICS}).to_csv(
        os.path.join(args.outdir, "land_use_index.csv"), index=False)
    pd.DataFrame({"metric": INDEX_METRICS, "PC1_loading": loadings.values,
                  "variance_explained": [var[0]] + [np.nan] * (len(INDEX_METRICS) - 1)
                  }).to_csv(os.path.join(args.outdir,
                                         "land_use_index_loadings.csv"),
                            index=False)

    # agreement with the published index, as a check that the locally built
    # one measures the same gradient
    if VALIDATION_METRIC in land.columns:
        rv, pv = spearmanr(index.values,
                           land.loc[shared, VALIDATION_METRIC].astype(float))
        print(f"agreement with {VALIDATION_METRIC}: rho = {rv:+.3f}, "
              f"p = {pv:.4f}")
        print("  The Human Footprint Index is not part of the index; this is a")
        print("  check that the two describe the same gradient.")
    print()
else:
    index = land.loc[shared, args.metric].astype(float)
    predictor_name = args.metric.replace(".", " ")
    loadings, var = None, None

# ---- every guild against the chosen metric --------------------------
# a guild with no variation across samples has no defined correlation
chosen = [gd for gd in g.index if g.loc[gd, shared].std() > 0]
dropped = [gd for gd in g.index if gd not in chosen]
if dropped:
    print(f"[note] constant across samples, not tested: {dropped}")

print(f"{len(shared)} streams, predictor {predictor_name}, "
      f"{len(chosen)} guilds\n")

rows = []
for gd in chosen:
    rho, p = spearmanr(index.values, g.loc[gd, shared])
    rows.append(dict(guild=gd, predictor=predictor_name, rho=rho, p=p,
                     group_q=(t.loc[gd, "q"] if gd in t.index and "q" in t.columns
                              else np.nan)))
c = pd.DataFrame(rows)
c["q"] = bh(c.p)
c = c.sort_values("rho")

print(f"corrected over {len(c)} comparisons "
      f"(group-comparison q shown for reference):")
for _, r in c.iterrows():
    mark = " *" if r.q < 0.05 else ("  ." if r.q < 0.10 else "   ")
    gq = "   --" if pd.isna(r.group_q) else f"{r.group_q:.3f}"
    print(f" {mark} {r.guild:30s} rho = {r.rho:+.3f}  p = {r.p:.5f}  "
          f"q = {r.q:.4f}   group q = {gq}")
print(f"\n{(c.q < 0.05).sum()} of {len(c)} at q < 0.05")
c.to_csv(os.path.join(args.outdir, "guilds_gradient.csv"), index=False)

# ---- everything, uncorrected, for reference -------------------------
allrows = []
for gd in g.index:
    if g.loc[gd, shared].std() == 0:
        continue
    for m in [x for x in METRICS if x in land.columns]:
        rho, p = spearmanr(land.loc[shared, m], g.loc[gd, shared])
        allrows.append(dict(guild=gd, metric=m, rho=rho, p=p))
allr = pd.DataFrame(allrows)
allr.to_csv(os.path.join(args.outdir, "guilds_vs_landscape_all.csv"),
            index=False)
print(f"\nall {len(allr)} guild-by-metric correlations written uncorrected to")
print("guilds_vs_landscape_all.csv. They are exploratory: corrected over that")
print("many tests nothing survives, including correlations above 0.85.")

# ---------------------------------------------------------------------
# only the guilds that survived correction are plotted: eleven panels would
# be unreadable, and the full set of correlations is in the output table
plotted = c[c.q < 0.05]
if plotted.empty:
    plotted = c.reindex(c.p.sort_values().index[:2])
    print("\n[note] no guild survived correction; the two lowest p values are "
          "plotted")

n = len(plotted)
fig, axes = plt.subplots(1, n, figsize=(2.15 * n + 0.8, 3.2), squeeze=False)
axes = axes[0]

x = index.values
status = np.array(["preserved" if s.startswith("Pres") else "degraded"
                   for s in shared])

for letter, ax, (_, r) in zip("abcdefgh", axes, plotted.iterrows()):
    y = g.loc[r.guild, shared].astype(float).values
    for grp, colour in (("preserved", COL_PRES), ("degraded", COL_DEG)):
        sel = status == grp
        ax.scatter(x[sel], y[sel], s=34, c=colour, edgecolors="white",
                   linewidths=0.6, zorder=3)

    # Theil-Sen: the median of all pairwise slopes. Rank-based like the
    # Spearman beside it, and not pulled by one stream the way least squares
    # would be.
    slope, intercept, _, _ = theilslopes(y, x)
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, intercept + slope * xs, color="#757575", lw=1.1, zorder=2)

    ax.text(-0.30, 1.06, f"({letter})", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left")
    ax.set_title(r.guild.replace(" / ", "\n").replace(" (", "\n("),
                 fontsize=6.8, pad=8, linespacing=1.3)
    # corner chosen per guild: see STAT_CORNER below, defined once and used
    # by both figures
    if r.guild in ("Fe(III) reducer",):
        tx, ty, ha, va = 0.96, 0.05, "right", "bottom"
    else:
        tx, ty, ha, va = 0.04, 0.95, "left", "top"
    ax.text(tx, ty,
            f"$\\rho$ = {r.rho:.2f}\n" +
            ("$q$ < 0.001" if r.q < 0.001 else f"$q$ = {r.q:.3f}"),
            transform=ax.transAxes, ha=ha, va=va, fontsize=7.0,
            linespacing=1.5)
    ax.set_xlabel(predictor_name, fontsize=7.8)
    if letter == "a":
        ax.set_ylabel("% within domain", fontsize=8)
    ax.tick_params(labelsize=7.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

fig.tight_layout()
base = os.path.join(args.outdir, "Fig_guilds_gradient")
fig.savefig(f"{base}.tif", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(f"{base}.jpg", dpi=300, pil_kwargs={"quality": 95})
plt.close(fig)

# ---------------------------------------------------------------------
# A biplot of the ordination the index comes from. The index is a derived
# variable, and quoting the percentage of variance it explains without showing
# the ordination leaves that number unsupported: the reader has to see how
# much of the structure sits on the first axis, what the axis is made of, and
# where the streams fall along it.
if loadings is not None:
    fig2, ax = plt.subplots(figsize=(5.2, 4.6))

    status2 = np.array(["preserved" if s_.startswith("Pres") else "degraded"
                        for s_ in shared])
    for grp, colour in (("preserved", COL_PRES), ("degraded", COL_DEG)):
        sel = status2 == grp
        ax.scatter(index.values[sel], pc2.values[sel], s=52, c=colour,
                   edgecolors="white", linewidths=0.8, zorder=3)

    # sample labels alternate above and below the point so neighbours in a
    # tight cluster do not overprint each other
    for k2, (s_, xv, yv) in enumerate(zip(shared, index.values, pc2.values)):
        dy = -11 if k2 % 2 == 0 else 8
        ax.annotate(s_, (xv, yv), textcoords="offset points",
                    xytext=(0, dy), ha="center", fontsize=6.0,
                    color="#595959")

    # loading arrows scaled to the spread of the scores on each axis
    sx = 0.72 * np.abs(index.values).max()
    sy = 0.72 * np.abs(pc2.values).max()
    for mname in INDEX_METRICS:
        xv, yv = loadings[mname] * sx * 1.5, load2[mname] * sy * 1.5
        ax.annotate("", xy=(xv, yv), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="#404040", lw=1.0))
        # the label is pushed clear of the arrow tip along both axes, and
        # given a light halo so it stays readable where it meets a point
        ax.text(xv * 1.13, yv * 1.13 + (0.06 * sy if abs(yv) < 0.2 * sy else 0),
                mname.replace(".", " "), fontsize=7.2,
                ha="left" if xv >= 0 else "right",
                va="bottom" if yv >= 0 else "top", color="#1a1a1a",
                path_effects=[pe.withStroke(linewidth=2.4,
                                            foreground="white")])

    ax.axhline(0, color="#d9d9d9", lw=0.6, zorder=1)
    ax.axvline(0, color="#d9d9d9", lw=0.6, zorder=1)
    ax.set_xlabel(f"PC1, land use intensity ({100 * var[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({100 * var[1]:.1f}%)")
    ax.margins(0.30)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig2.tight_layout()
    base2 = os.path.join(args.outdir, "Fig_land_use_pca")
    fig2.savefig(f"{base2}.tif", dpi=300,
                 pil_kwargs={"compression": "tiff_lzw"})
    fig2.savefig(f"{base2}.jpg", dpi=300, pil_kwargs={"quality": 95})
    plt.close(fig2)
    print(f"\nWritten: {base2}.tif, {base2}.jpg")
    print(f"         PCA of the landscape metrics; PC1 {100 * var[0]:.1f}%, "
          f"PC2 {100 * var[1]:.1f}%")

# ---------------------------------------------------------------------
# Only the guilds that survive correction are drawn. The correlations for
# every guild are in guilds_gradient.csv, which is where a reader checks
# whether the plotted set was cherry-picked; panels for associations the
# correction rejects would put them on the same visual footing as the ones
# it supports.
order_all = c[c.q < 0.05].sort_values("rho", ascending=False)
if order_all.empty:
    order_all = c.reindex(c.p.sort_values().index[:2])
    print("\n[note] no guild survived correction; the two lowest p values "
          "are plotted")

ncol = min(4, len(order_all))
nrow = int(np.ceil(len(order_all) / ncol))
# Upscaling a raster afterwards costs resolution, so when this panel is going
# to be stacked under a wider one it is drawn at that width from the start.
w3 = args.all_width if args.all_width else 2.05 * ncol
fig3, axes3 = plt.subplots(nrow, ncol, figsize=(w3, 2.15 * nrow),
                           squeeze=False)

# Where the statistics block sits is set per guild rather than globally: the
# points climb to the right in every panel, so a fixed corner collides with
# the data in some of them. A guild whose highest points are on the right
# gets the block bottom right, the others top left.
STAT_CORNER = {
    "Fe(III) reducer": "bottom right",
}

for k, (_, r) in enumerate(order_all.iterrows()):
    ax = axes3[k // ncol][k % ncol]
    y = g.loc[r.guild, shared].astype(float).values
    for grp, colour in (("preserved", COL_PRES), ("degraded", COL_DEG)):
        sel = np.array([s_.startswith("Pres") == (grp == "preserved")
                        for s_ in shared])
        ax.scatter(index.values[sel], y[sel], s=22, c=colour,
                   edgecolors="white", linewidths=0.5, zorder=3)

    xa = index.values
    slope, intercept, _, _ = theilslopes(y, xa)
    xs = np.array([xa.min(), xa.max()])
    ax.plot(xs, intercept + slope * xs, color="#757575", lw=1.1, zorder=2)

    ax.set_title(r.guild.replace(" / ", "\n").replace(" (", "\n("),
                 fontsize=6.6, pad=5, linespacing=1.25)

    corner = STAT_CORNER.get(r.guild, "top left")
    if corner == "bottom right":
        tx, ty, ha, va = 0.96, 0.05, "right", "bottom"
    else:
        tx, ty, ha, va = 0.04, 0.95, "left", "top"
    ax.text(tx, ty,
            f"$\\rho$ = {r.rho:.2f}\n" +
            ("$q$ < 0.001" if r.q < 0.001 else f"$q$ = {r.q:.3f}"),
            transform=ax.transAxes, ha=ha, va=va, fontsize=6.4,
            linespacing=1.4)
    ax.tick_params(labelsize=6.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

for k in range(len(order_all), nrow * ncol):
    axes3[k // ncol][k % ncol].axis("off")

fig3.supxlabel("Land use intensity (PC1)" if loadings is not None
               else predictor_name, fontsize=8.5)
fig3.supylabel("% within domain", fontsize=8.5)
fig3.tight_layout()
base3 = os.path.join(args.outdir, "Fig_guilds_gradient_all")
fig3.savefig(f"{base3}.tif", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
fig3.savefig(f"{base3}.jpg", dpi=300, pil_kwargs={"quality": 95})
plt.close(fig3)
print(f"\nWritten: {base3}.tif, {base3}.jpg")
print("         the guilds that survived correction, against the index")

print(f"\nWritten: {base}.tif, {base}.jpg")
print("         guilds_gradient.csv, guilds_vs_landscape_all.csv")
print("         land_use_index.csv, land_use_index_loadings.csv")
print("\nBlue preserved, red degraded; the grey line is a Theil-Sen fit, the")
print("median of all pairwise slopes, which is rank-based like the Spearman")
print("coefficient beside it.")
