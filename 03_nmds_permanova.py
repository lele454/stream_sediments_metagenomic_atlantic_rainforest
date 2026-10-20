#!/usr/bin/env python3
"""
Figure: NMDS of taxonomic and functional composition, side by side
--------------------------------------------------------------------
Four panels: genus composition from the Bracken reports and KEGG orthologue
composition from the KO table, each ordinated twice.

The top row is Bray-Curtis on proportions, non-metric MDS, as before. The
bottom row is Aitchison distance - Euclidean distance between centred
log-ratios - shown as principal coordinates, which for that distance is the
same as a principal component analysis of the log-ratios and gives axes with
a readable share of variance.

Both rows are drawn because they do not agree, and which one is right is not
a matter of taste. These tables are compositions: every sample sums to a
fixed total, so a part that grows does so at the expense of the others, and
Bray-Curtis takes that arithmetic at face value. A log-ratio treats each part
relative to the geometric mean of the sample and is unaffected by the total.
Where the two disagree, the Bray-Curtis result includes an effect of the
closure that the Aitchison result does not.

The two panels answer different questions and are worth showing together. A
taxonomic difference between the stream groups says the assemblages differ; a
functional difference says the gene content differs, which need not follow.
Where taxonomy separates and function does not, different organisms are doing
comparable things.

Each panel carries its PERMANOVA, its PERMDISP and, for the NMDS panels, the
stress. PERMDISP is on the panel rather than in the console because the two
tests are read together: PERMANOVA asks whether the groups sit in different
places, PERMDISP whether they are differently spread, and a significant
PERMANOVA with a significant PERMDISP cannot be attributed to location. The
dashed ellipse at 1.96 standard deviations helps the eye group the points and
is not a test; with seven samples per group it should not be read as one, and
the ellipses often differ in size where PERMDISP finds no difference at all.

Abundances are converted to proportions within each sample before the
distances are computed. Sequencing depth differs severalfold between samples
and Bray-Curtis on raw counts would track depth.

Inputs:
  --bracken   directory of Bracken genus reports, one per sample
  --ko        KO abundance table (.xlsx or .csv), KEGG_KO in the first column
  --outdir    where to write (default: current directory)

Outputs:
  Fig_NMDS_taxonomic_functional.tif / .jpg
  nmds_scores.csv        sample scores of both ordinations
  permanova_results.csv

Usage:
  python3 fig_nmds.py --bracken bracken/ --ko Table_S4_KO_abundance.xlsx
"""

import os
import re
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

try:
    from sklearn.manifold import MDS
except ImportError:
    sys.exit("scikit-learn is required for NMDS:\n"
             "  conda install -c conda-forge scikit-learn")

SAMPLES = [f"Pres0{i}" for i in range(1, 8)] + [f"Deg0{i}" for i in range(1, 8)]

COL_PRES = "#2c7bb6"
COL_DEG = "#d7191c"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------
def sample_of(path):
    name = os.path.basename(path)
    for s in SAMPLES:
        if s.lower() in name.lower():
            return s
    return None


def read_bracken(directory):
    """Genus-by-sample counts from Bracken reports."""
    counts = {}
    for path in sorted(glob.glob(os.path.join(directory, "*"))):
        if not os.path.isfile(path):
            continue
        s = sample_of(path)
        if s is None:
            continue
        d = pd.read_csv(path, sep="\t")
        if not {"name", "taxonomy_lvl", "new_est_reads"}.issubset(d.columns):
            print(f"[skipped] {os.path.basename(path)}: unexpected columns")
            continue
        d = d[d.taxonomy_lvl == "G"]
        counts[s] = d.set_index("name").new_est_reads
    if not counts:
        sys.exit(f"no Bracken reports recognised in {directory}")
    m = pd.DataFrame(counts).fillna(0)
    return m.reindex(columns=[s for s in SAMPLES if s in m.columns])


def read_ko(path):
    d = (pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls"))
         else pd.read_csv(path))
    d = d.set_index(d.columns[0])
    cols = [c for c in SAMPLES if c in d.columns]
    if not cols:
        sys.exit(f"no sample columns of the expected names found in {path}")
    return d[cols]


def permanova(D, groups, n_perm=9999, seed=42):
    """
    One-way PERMANOVA on a distance matrix. Implemented here rather than
    imported so the script has no dependency beyond scipy: the statistic is
    the ratio of between- to within-group sums of squares from the distances,
    and its null distribution comes from permuting the labels.
    """
    g = np.asarray(groups)
    n = len(g)
    levels = np.unique(g)
    a = len(levels)

    def ss_within(labels):
        w = 0.0
        for lv in levels:
            idx = np.where(labels == lv)[0]
            if len(idx) == 0:
                return np.nan
            sub = D[np.ix_(idx, idx)]
            w += (sub ** 2).sum() / (2 * len(idx))
        return w

    total = (D ** 2).sum() / (2 * n)

    def F_of(labels):
        w = ss_within(labels)
        return ((total - w) / (a - 1)) / (w / (n - a))

    obs = F_of(g)
    r2 = 1 - ss_within(g) / total

    rng = np.random.default_rng(seed)
    null = np.array([F_of(rng.permutation(g)) for _ in range(n_perm)])
    p = (np.sum(null >= obs) + 1) / (n_perm + 1)
    return dict(R2=r2, F=obs, p=p, n_perm=n_perm)


def dispersion(D, groups):
    """Mean within-group distance, to be read alongside the PERMANOVA."""
    g = np.asarray(groups)
    out = {}
    for lv in np.unique(g):
        idx = np.where(g == lv)[0]
        sub = D[np.ix_(idx, idx)]
        out[lv] = float(sub[np.triu_indices(len(idx), 1)].mean())
    return out


def gower_centre(D):
    A = -0.5 * D ** 2
    n = A.shape[0]
    one = np.ones((n, n)) / n
    return A - one @ A - A @ one + one @ A @ one


def permdisp(D, groups, n_perm=9999, seed=42):
    """
    Homogeneity of multivariate dispersions (Anderson 2006), the betadisper
    and permutest of vegan. Each sample's distance to its group centroid is
    measured in principal coordinate space, with the correction that subtracts
    the contribution of the imaginary axes a non-Euclidean distance produces;
    the difference between the group means of those distances is then tested
    by permuting the labels.
    """
    g = np.asarray(groups)
    w, V = np.linalg.eigh(gower_centre(D))
    keep = w > 1e-10
    real = V[:, keep] * np.sqrt(w[keep])
    neg = w < -1e-10
    imag = V[:, neg] * np.sqrt(-w[neg]) if neg.any() else None

    def to_centroid(labels):
        d = np.empty(len(labels))
        for lv in np.unique(labels):
            idx = np.where(labels == lv)[0]
            sq = ((real[idx] - real[idx].mean(axis=0)) ** 2).sum(axis=1)
            if imag is not None:
                sq = sq - ((imag[idx] - imag[idx].mean(axis=0)) ** 2).sum(axis=1)
            d[idx] = np.sqrt(np.abs(sq))
        return d

    d = to_centroid(g)
    lv = np.unique(g)
    obs = abs(d[g == lv[0]].mean() - d[g == lv[1]].mean())
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        gg = rng.permutation(g)
        dd = to_centroid(gg)
        null[i] = abs(dd[gg == lv[0]].mean() - dd[gg == lv[1]].mean())
    p = (np.sum(null >= obs) + 1) / (n_perm + 1)
    return {l: float(d[g == l].mean()) for l in lv}, p


def multiplicative_replacement(P, frac=0.65):
    """
    Zeros cannot be logged, and in a count table they are values below
    detection rather than true absences. Each is replaced by a fraction of
    the smallest observed proportion and the remaining parts are rescaled so
    the sample still sums to one (Martin-Fernandez et al. 2003).
    """
    out = P.copy()
    delta = frac * P.values[P.values > 0].min()
    for c in out.columns:
        v = out[c].values.astype(float)
        z = v == 0
        if z.any():
            v[z] = delta
            v[~z] = v[~z] * (1.0 - z.sum() * delta)
            out[c] = v
    return out


def clr(P):
    """Centred log-ratio, per sample."""
    L = np.log(P.values)
    return pd.DataFrame(L - L.mean(axis=0, keepdims=True),
                        index=P.index, columns=P.columns)


def pcoa(D):
    """Principal coordinates, with the variance each axis carries."""
    w, V = np.linalg.eigh(gower_centre(D))
    order = w.argsort()[::-1]
    w, V = w[order], V[:, order]
    keep = w > 1e-10
    coords = V[:, keep] * np.sqrt(w[keep])
    var = 100 * w[keep] / w[keep].sum()
    return coords, var


def confidence_ellipse(ax, x, y, colour, n_std=1.96):
    """
    Ellipse covering the group at the stated multiple of the standard
    deviation, drawn from the covariance of the two ordination axes. With
    seven points per group this is a visual aid to reading the panel, not an
    inference: the PERMANOVA is what tests the grouping.
    """
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    if np.any(vals <= 0):
        return
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * n_std * np.sqrt(vals)
    e = Ellipse((x.mean(), y.mean()), width=w, height=h, angle=angle,
                facecolor=colour, alpha=0.12, edgecolor=colour,
                linewidth=1.0, linestyle="--", zorder=2)
    ax.add_patch(e)


def nmds(D, seed=42):
    """Non-metric MDS on a precomputed distance matrix, two dimensions."""
    # scikit-learn is renaming these arguments; the call is built to suit the
    # installed version so the script does not emit deprecation warnings or
    # break on the rename
    import inspect
    sig = inspect.signature(MDS.__init__).parameters
    kw = dict(n_components=2, random_state=seed, n_init=50, max_iter=1000,
              normalized_stress=True)
    if "metric_mds" in sig:
        # current naming: metric_mds switches off the metric variant and
        # metric names the distance. Both exist for a while, with
        # dissimilarity deprecated, so the new pair is used wherever it is
        # available rather than whichever is found first.
        kw["metric_mds"] = False
        kw["metric"] = "precomputed"
    else:
        kw["metric"] = False
        kw["dissimilarity"] = "precomputed"
    if "init" in sig:
        # pin it: the default is scheduled to change from random to
        # classical_mds, which would silently alter the ordination
        kw["init"] = "random"
    mds = MDS(**kw)
    coords = mds.fit_transform(D)
    return coords, mds.stress_


# ---------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--bracken", required=True)
ap.add_argument("--ko", required=True)
ap.add_argument("--outdir", default=".")
args = ap.parse_args()

os.makedirs(args.outdir, exist_ok=True)

tax = read_bracken(args.bracken)
fun = read_ko(args.ko)

shared = [s for s in SAMPLES if s in tax.columns and s in fun.columns]
if len(shared) < 6:
    sys.exit(f"only {len(shared)} samples present in both tables")
tax, fun = tax[shared], fun[shared]
tax = tax.loc[tax.sum(axis=1) > 0]
fun = fun.loc[fun.sum(axis=1) > 0]

status = np.array(["preserved" if s.startswith("Pres") else "degraded"
                   for s in shared])

print(f"{len(shared)} samples: {int((status=='preserved').sum())} preserved, "
      f"{int((status=='degraded').sum())} degraded")
print(f"taxonomy: {tax.shape[0]} genera")
print(f"function: {fun.shape[0]} KEGG orthologues\n")

panels = []
records = []
for label, table in [("Taxonomic (genus)", tax), ("Functional (KEGG KO)", fun)]:
    rel = table / table.sum()

    # Bray-Curtis on proportions, ordinated by non-metric MDS
    D = squareform(pdist(rel.T.values, metric="braycurtis"))
    res = permanova(D, status)
    disp = dispersion(D, status)
    dmeans, dp = permdisp(D, status)
    coords, stress = nmds(D)

    print(f"--- {label}, Bray-Curtis ---")
    print(f"  PERMANOVA  R2 = {res['R2']:.3f}  F = {res['F']:.2f}  "
          f"p = {res['p']:.4f}  ({res['n_perm']} permutations)")
    print(f"  PERMDISP   preserved {dmeans['preserved']:.4f}, "
          f"degraded {dmeans['degraded']:.4f}  p = {dp:.4f}")
    print(f"  mean within-group distance: preserved {disp['preserved']:.3f}, "
          f"degraded {disp['degraded']:.3f}")
    print(f"  NMDS stress {stress:.4f}\n")

    panels.append(dict(label=label, distance="Bray-Curtis", coords=coords,
                       stress=float(stress), res=res, permdisp_p=dp,
                       axis_labels=("NMDS1", "NMDS2")))
    records.append(dict(dataset=label, distance="Bray-Curtis on proportions",
                        ordination="NMDS", n_features=table.shape[0],
                        R2=round(res["R2"], 4), F=round(res["F"], 3),
                        p=res["p"], stress=round(float(stress), 4),
                        disp_preserved=round(dmeans["preserved"], 4),
                        disp_degraded=round(dmeans["degraded"], 4),
                        permdisp_p=dp))

    # Aitchison, ordinated by principal coordinates
    C = clr(multiplicative_replacement(rel))
    Da = squareform(pdist(C.T.values, metric="euclidean"))
    resa = permanova(Da, status)
    ameans, ap_ = permdisp(Da, status)
    acoords, avar = pcoa(Da)

    print(f"--- {label}, Aitchison ---")
    print(f"  PERMANOVA  R2 = {resa['R2']:.3f}  F = {resa['F']:.2f}  "
          f"p = {resa['p']:.4f}")
    print(f"  PERMDISP   preserved {ameans['preserved']:.3f}, "
          f"degraded {ameans['degraded']:.3f}  p = {ap_:.4f}")
    print(f"  PC1 {avar[0]:.1f}%, PC2 {avar[1]:.1f}%\n")

    panels.append(dict(label=label, distance="Aitchison", coords=acoords,
                       stress=None, res=resa, permdisp_p=ap_,
                       axis_labels=(f"PC1 ({avar[0]:.1f}%)",
                                    f"PC2 ({avar[1]:.1f}%)")))
    records.append(dict(dataset=label, distance="Aitchison (CLR)",
                        ordination="PCoA", n_features=table.shape[0],
                        R2=round(resa["R2"], 4), F=round(resa["F"], 3),
                        p=resa["p"], stress=np.nan,
                        disp_preserved=round(ameans["preserved"], 4),
                        disp_degraded=round(ameans["degraded"], 4),
                        permdisp_p=ap_))

pd.DataFrame(records).to_csv(
    os.path.join(args.outdir, "permanova_results.csv"), index=False)

if any(not np.isnan(r["stress"]) and r["stress"] > 0.2 for r in records):
    print("[note] a stress above 0.2 means the two dimensions represent the")
    print("       distances poorly; read such an ordination with caution.\n")

# ---------------------------------------------------------------------
# rows are the two distances, columns the two datasets, so a reader compares
# down a column and not across unrelated panels
ORDER = [("Taxonomic (genus)", "Bray-Curtis"), ("Functional (KEGG KO)", "Bray-Curtis"),
         ("Taxonomic (genus)", "Aitchison"), ("Functional (KEGG KO)", "Aitchison")]
SHORT_TITLE = {"Taxonomic (genus)": "Taxonomic (genus)",
               "Functional (KEGG KO)": "Functional (KEGG KO)"}
ROW_LABEL = {"Bray-Curtis": "Bray-Curtis on proportions",
             "Aitchison": "Aitchison distance (CLR)"}
by_key = {(p["label"], p["distance"]): p for p in panels}

fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.0))
axes = axes.ravel()

scores = []
for panel_letter, ax, key in zip("abcd", axes, ORDER):
    pan = by_key[key]
    coords, res = pan["coords"], pan["res"]
    for grp, colour in (("preserved", COL_PRES), ("degraded", COL_DEG)):
        sel = status == grp
        confidence_ellipse(ax, coords[sel, 0], coords[sel, 1], colour)
        ax.scatter(coords[sel, 0], coords[sel, 1], s=46, c=colour,
                   edgecolors="white", linewidths=0.8, zorder=3)

    ax.axhline(0, color="#d9d9d9", linewidth=0.6, zorder=1)
    ax.axvline(0, color="#d9d9d9", linewidth=0.6, zorder=1)

    # p is reported as an inequality at the permutation floor: with 9999
    # permutations no smaller value is resolvable, and printing one would
    # imply precision the test does not have
    floor = 1 / (res["n_perm"] + 1)
    pstr = (f"$p$ < {floor:.4f}" if res["p"] <= floor
            else f"$p$ = {res['p']:.3f}")
    lines = [f"$R^2$ = {res['R2']:.3f}", pstr]
    if pan["stress"] is not None:
        lines.append(f"stress = {pan['stress']:.3f}")
    lines.append(f"PERMDISP $p$ = {pan['permdisp_p']:.3f}")
    ax.text(0.97, 0.97, "\n".join(lines), transform=ax.transAxes,
            ha="right", va="top", fontsize=7.5, linespacing=1.5,
            color="#1a1a1a")

    # panel letter outside the axes, top left, so it never sits on a point.
    # The title carries only the dataset: the distance is the row, and naming
    # it in every title pushed the title into the panel letter once the equal
    # aspect had narrowed the axes.
    ax.text(-0.14, 1.04, f"({panel_letter})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")
    ax.set_title(SHORT_TITLE[pan["label"]], fontsize=8.5, pad=8)

    ax.set_xlabel(pan["axis_labels"][0])
    ax.set_ylabel(pan["axis_labels"][1])
    # Headroom for the statistics block, added before the aspect is fixed.
    # An ordination has to be drawn with equal scales or the distances it
    # shows are not the distances that were tested, and with
    # adjustable="datalim" matplotlib reaches that by overriding the limits
    # set here - which is what discarded the headroom and let the text sit on
    # a point. adjustable="box" reshapes the axes instead and keeps both.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.26 * (hi - lo))
    ax.set_aspect("equal", adjustable="box")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    for s_, (x, y) in zip(shared, coords[:, :2]):
        scores.append(dict(dataset=pan["label"], distance=pan["distance"],
                           sample=s_,
                           status="preserved" if s_.startswith("Pres") else "degraded",
                           axis1=round(float(x), 4), axis2=round(float(y), 4)))

pd.DataFrame(scores).to_csv(
    os.path.join(args.outdir, "nmds_scores.csv"), index=False)

fig.tight_layout()
# The equal aspect reshapes each axes at draw time, so the row labels are
# positioned from the boxes as they actually end up rather than from where
# the layout manager left them.
fig.canvas.draw()
for row, dist in enumerate(["Bray-Curtis", "Aitchison"]):
    box = axes[row * 2].get_position()
    fig.text(0.012, box.y0 + box.height / 2, ROW_LABEL[dist],
             rotation=90, ha="left", va="center", fontsize=8.5)
fig.subplots_adjust(left=0.115)

base = os.path.join(args.outdir, "Fig_NMDS_taxonomic_functional")
fig.savefig(f"{base}.tif", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(f"{base}.jpg", dpi=300, pil_kwargs={"quality": 95})
plt.close(fig)

print(f"Written: {base}.tif, {base}.jpg")
print("         nmds_scores.csv, permanova_results.csv")
print("\nEach panel carries its PERMANOVA, PERMDISP and, for the NMDS panels,")
print("the stress. Top row is Bray-Curtis on proportions, bottom row Aitchison")
print("distance on centred log-ratios; left column taxonomic, right functional.")
print("Blue is preserved and red degraded. Where the two rows disagree, the")
print("Bray-Curtis result carries an effect of the closure and the Aitchison")
print("one does not: say which you are reporting, and why.")
