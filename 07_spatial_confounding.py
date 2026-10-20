#!/usr/bin/env python3
"""
Is the landscape effect separable from geographic distance?
-------------------------------------------------------------
The preserved streams here sit inside one protected reserve and the degraded
ones are spread across a coastal region, so landscape composition and spatial
position are not independent. Any association between community composition
and land cover could therefore be distance decay wearing a landscape label.
This tests which it is, as far as the design allows.

Three tests, in increasing strictness:

  1. Mantel between community distance and geographic distance. Establishes
     whether distance decay is present at all.
  2. Mantel between community distance and each landscape variable, and
     between geographic distance and each landscape variable. If the second is
     strong, the two predictors are confounded by construction.
  3. Partial Mantel of community against landscape, controlling for geography,
     and against geography, controlling for landscape. This is the test that
     asks whether either predictor carries information the other does not.

A partial Mantel cannot separate predictors that are themselves strongly
correlated - it will simply report both as weak - so its correlations are read
alongside test 2 rather than on their own. Where the design confounds two
variables, no statistic recovers the distinction, and saying so is the honest
result.

Geographic distance is computed on the sphere from latitude and longitude
rather than in degrees, since a degree of longitude is shorter than a degree
of latitude at this latitude and Euclidean distance in degrees would distort
the east-west axis.

Inputs:
  --bracken   directory of Bracken genus reports
  --land      landscape CSV with Latitude and Longitude columns
  --outdir    where to write (default: current directory)

Outputs:
  spatial_confounding.csv
  Fig_distance_decay.tif / .jpg

Usage:
  python3 spatial_confounding.py --bracken . --land LandUseCover1.csv
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

SAMPLES = [f"Pres0{i}" for i in range(1, 8)] + [f"Deg0{i}" for i in range(1, 8)]
LAND_ORDER = [f"Deg0{i}" for i in range(1, 8)] + [f"Pres0{i}" for i in range(1, 8)]

METRICS = ["Forest.cover", "Cropland.cover", "Urban.cover", "HFP",
           "Elevacao", "Diversity_q1"]

EARTH_RADIUS_KM = 6371.0

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


def haversine_matrix(lat, lon):
    """Great-circle distance in km between every pair of points."""
    la, lo = np.radians(np.asarray(lat)), np.radians(np.asarray(lon))
    dla = la[:, None] - la[None, :]
    dlo = lo[:, None] - lo[None, :]
    a = (np.sin(dla / 2) ** 2
         + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin(dlo / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def lower(D):
    return D[np.triu_indices_from(D, 1)]


def mantel(D1, D2, n_perm=9999, seed=42):
    a, b = lower(D1), lower(D2)
    obs = spearmanr(a, b).statistic
    rng = np.random.default_rng(seed)
    n = D1.shape[0]
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(n)
        null[i] = spearmanr(a, lower(D2[np.ix_(p, p)])).statistic
    return obs, (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)


def partial_mantel(D1, D2, D3, n_perm=9999, seed=42):
    """
    Correlation of D1 and D2 with the effect of D3 removed from both, tested
    by permuting D2. The statistic is the partial Spearman correlation on the
    ranked lower triangles.
    """
    def rank(x):
        return pd.Series(x).rank().values

    def partial(a, b, c):
        a, b, c = rank(a), rank(b), rank(c)
        rab = np.corrcoef(a, b)[0, 1]
        rac = np.corrcoef(a, c)[0, 1]
        rbc = np.corrcoef(b, c)[0, 1]
        denom = np.sqrt((1 - rac ** 2) * (1 - rbc ** 2))
        return np.nan if denom == 0 else (rab - rac * rbc) / denom

    a, c = lower(D1), lower(D3)
    obs = partial(a, lower(D2), c)
    rng = np.random.default_rng(seed)
    n = D1.shape[0]
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(n)
        null[i] = partial(a, lower(D2[np.ix_(p, p)]), c)
    null = null[~np.isnan(null)]
    return obs, (np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1)


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

if not {"Latitude", "Longitude"}.issubset(land.columns):
    sys.exit("Latitude and Longitude columns are required")

shared = [s for s in m.columns if s in land.index]
m, land = m[shared], land.loc[shared]
metrics = [c for c in METRICS if c in land.columns]

rel = m / m.sum()
Dcom = squareform(pdist(rel.T.values, metric="braycurtis"))
Dgeo = haversine_matrix(land.Latitude.values, land.Longitude.values)

print(f"{len(shared)} streams")
print(f"pairwise distance: {lower(Dgeo).min():.1f} to "
      f"{lower(Dgeo).max():.1f} km (median {np.median(lower(Dgeo)):.1f})\n")

# within and between group distances, to show the confounding directly
status = np.array([s.startswith("Pres") for s in shared])
iu = np.triu_indices_from(Dgeo, 1)
same = status[iu[0]] == status[iu[1]]
print(f"mean distance within a group:  {lower(Dgeo)[same].mean():.1f} km")
print(f"mean distance between groups:  {lower(Dgeo)[~same].mean():.1f} km")
print("If these differ substantially, group and space are confounded and the")
print("tests below cannot fully separate them.\n")

rows = []

r, p = mantel(Dcom, Dgeo)
rows.append(dict(test="community ~ geography", statistic=round(r, 4), p=p))
print(f"community vs geographic distance: r = {r:+.3f}, p = {p:.4f}")
print("  This is distance decay. A strong result here means any landscape")
print("  association has to be read against it.\n")

print("landscape variable, three tests each:")
print("  (i) community ~ variable   (ii) geography ~ variable"
      "   (iii) community ~ variable | geography\n")
for v in metrics:
    Dv = squareform(pdist(((land[[v]] - land[[v]].mean())
                           / land[[v]].std(ddof=1)).values))
    r1, p1 = mantel(Dcom, Dv)
    r2, p2 = mantel(Dgeo, Dv)
    r3, p3 = partial_mantel(Dcom, Dv, Dgeo)
    rows += [
        dict(test=f"community ~ {v}", statistic=round(r1, 4), p=p1),
        dict(test=f"geography ~ {v}", statistic=round(r2, 4), p=p2),
        dict(test=f"community ~ {v} | geography", statistic=round(r3, 4), p=p3)]
    print(f"  {v:16s} (i) r = {r1:+.3f} p = {p1:.4f}"
          f"   (ii) r = {r2:+.3f} p = {p2:.4f}"
          f"   (iii) r = {r3:+.3f} p = {p3:.4f}")

r, p = partial_mantel(Dcom, Dgeo, squareform(pdist(
    ((land[metrics] - land[metrics].mean())
     / land[metrics].std(ddof=1)).values)))
rows.append(dict(test="community ~ geography | all landscape",
                 statistic=round(r, 4), p=p))
print(f"\ncommunity vs geography, controlling for all landscape variables:"
      f" r = {r:+.3f}, p = {p:.4f}")

pd.DataFrame(rows).to_csv(
    os.path.join(args.outdir, "spatial_confounding.csv"), index=False)

# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.8, 3.8))
x, y = lower(Dgeo), lower(Dcom)
for lbl, sel, colour in [("within group", same, "#2c7bb6"),
                         ("between groups", ~same, "#d7191c")]:
    ax.scatter(x[sel], y[sel], s=20, c=colour, alpha=0.75,
               edgecolors="white", linewidths=0.4, zorder=3, label=lbl)

rho, pv = spearmanr(x, y)
ax.set_xlabel("Geographic distance (km)")
ax.set_ylabel("Bray-Curtis dissimilarity")
ax.text(0.03, 0.97,
        f"Mantel $r$ = {rho:+.3f}\n" +
        ("$p$ < 0.001" if pv < 0.001 else f"$p$ = {pv:.3f}"),
        transform=ax.transAxes, ha="left", va="top", fontsize=7.8,
        linespacing=1.5)
ax.legend(frameon=False, fontsize=7.4, loc="lower right")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

fig.tight_layout()
base = os.path.join(args.outdir, "Fig_distance_decay")
fig.savefig(f"{base}.tif", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(f"{base}.jpg", dpi=300, pil_kwargs={"quality": 95})
plt.close(fig)

print(f"\nWritten: {base}.tif, {base}.jpg, spatial_confounding.csv")
print("\nRead (ii) before (iii): where geography and a landscape variable are")
print("themselves strongly correlated, the partial test reports both as weak")
print("and the design, not the statistic, is the limit.")
