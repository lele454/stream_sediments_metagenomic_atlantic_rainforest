#!/usr/bin/env python3
"""
PERMANOVA with a test of dispersion homogeneity for the resistome
-----------------------------------------------------------------
A significant PERMANOVA can come from a shift in location, from a difference
in spread, or from both, and the test itself does not tell them apart. PERMDISP
(Anderson 2006), the betadisper/permutest of vegan, tests the spread.

Each matrix is tested with three distances, each under PERMANOVA and PERMDISP:

  Bray-Curtis on proportions       the published analysis
  Bray-Curtis on log10(RPKM+1)     what the Methods section describes
  Aitchison on proportions         the compositional treatment, where every
                                   ratio carries equal weight and no single
                                   dominant category can drive the result

The three do not agree, which is the point of printing them together. The
matrix is tested whole and with the categories named in --drop removed.

Usage
  python3 resistome_dispersion.py --table Resistome_functional_RPKM_by_sample.csv
  python3 resistome_dispersion.py --table t.csv --drop "Efflux / multidrug,Unclassified"
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from scipy.spatial.distance import pdist, squareform

SEED, N_PERM = 42, 9999

ap = argparse.ArgumentParser()
ap.add_argument("--table", required=True)
ap.add_argument("--drop", default="Efflux / multidrug,Unclassified",
                help="categories removed for the second analysis, comma separated")
ap.add_argument("--outdir", default=".")
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)

df = pd.read_csv(a.table)
df = df.rename(columns={df.columns[0]: "Function"}).set_index("Function")
df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
df = df.loc[df.sum(axis=1) > 0]
PRES = sorted([c for c in df.columns if c.lower().startswith("pres")])
DEGR = sorted([c for c in df.columns if c.lower().startswith("deg")])
df = df[PRES + DEGR]
grp = np.array(["P"] * len(PRES) + ["D"] * len(DEGR))
print(f"{df.shape[0]} ARG categories, {len(PRES)} preserved and {len(DEGR)} degraded\n")

tot = df.sum(axis=0)
u, p = mannwhitneyu(tot[PRES], tot[DEGR], alternative="two-sided")
print(f"total resistome (sum of RPKM): preserved median {np.median(tot[PRES]):,.0f}, "
      f"degraded {np.median(tot[DEGR]):,.0f}, p = {p:.4f}")
DROP = [c.strip() for c in a.drop.split(",") if c.strip() in df.index]
for c in DROP:
    sh = df.loc[c] / tot * 100
    print(f"'{c}' is {sh.mean():.1f}% of the resistome "
          f"(range {sh.min():.1f}-{sh.max():.1f}%)")
print()


def gower(D):
    A = -0.5 * D ** 2
    n = A.shape[0]
    o = np.ones((n, n)) / n
    return A - o @ A - A @ o + o @ A @ o


def permanova(D, g, n_perm=N_PERM):
    rs = np.random.default_rng(SEED)
    G, n, lev = gower(D), D.shape[0], np.unique(g)

    def st(gg):
        H = np.zeros((n, n))
        for l in lev:
            i = np.where(gg == l)[0]
            H[np.ix_(i, i)] = 1.0 / len(i)
        t = np.trace(G); aa = np.trace(H @ G @ H)
        return ((aa / (len(lev) - 1)) / ((t - aa) / (n - len(lev))), aa / t)

    F, R2 = st(g)
    null = np.array([st(rs.permutation(g))[0] for _ in range(n_perm)])
    return F, R2, (np.sum(null >= F) + 1) / (n_perm + 1)


def permdisp(D, g, n_perm=N_PERM):
    rs = np.random.default_rng(SEED)
    G = gower(D)
    w, V = np.linalg.eigh(G)
    k = w > 1e-10
    co = V[:, k] * np.sqrt(w[k])
    nk = w < -1e-10
    ic = V[:, nk] * np.sqrt(-w[nk]) if nk.any() else None

    def dist(gg):
        d = np.empty(len(gg))
        for l in np.unique(gg):
            i = np.where(gg == l)[0]
            sq = ((co[i] - co[i].mean(axis=0)) ** 2).sum(axis=1)
            if ic is not None:
                sq = sq - ((ic[i] - ic[i].mean(axis=0)) ** 2).sum(axis=1)
            d[i] = np.sqrt(np.abs(sq))
        return d

    d = dist(g)
    lev = np.unique(g)
    obs = abs(d[g == lev[0]].mean() - d[g == lev[1]].mean())
    null = np.empty(n_perm)
    for i in range(n_perm):
        gg = rs.permutation(g); dd = dist(gg)
        null[i] = abs(dd[gg == lev[0]].mean() - dd[gg == lev[1]].mean())
    return {l: d[g == l].mean() for l in lev}, (np.sum(null >= obs) + 1) / (n_perm + 1)


def clr(P):
    L = np.log(P.values)
    return pd.DataFrame(L - L.mean(axis=0, keepdims=True), index=P.index, columns=P.columns)


def repl(d, frac=0.65):
    P = (d / d.sum(axis=0)).copy()
    delta = frac * P.values[P.values > 0].min()
    for c in P.columns:
        v = P[c].values.astype(float); z = v == 0
        if z.any():
            v[z] = delta; v[~z] = v[~z] * (1 - z.sum() * delta); P[c] = v
    return P


def hill(p, q):
    p = p[p > 0]
    if q == 0: return float(len(p))
    if abs(q - 1) < 1e-9: return float(np.exp(-(p * np.log(p)).sum()))
    return float((p ** q).sum() ** (1 / (1 - q)))


rows = []
for tag, d in (("whole resistome", df),
               ("without " + " and ".join(f"'{c}'" for c in DROP), df.drop(index=DROP))):
    print("=" * 70); print(tag); print("=" * 70)
    prop = (d / d.sum(axis=0)).T.values
    mats = {"Bray-Curtis on proportions (published)":
                squareform(pdist(prop, metric="braycurtis")),
            "Bray-Curtis on log10(RPKM+1) (as in Methods)":
                squareform(pdist(np.log10(d.T.values + 1), metric="braycurtis")),
            "Aitchison on proportions (compositional)":
                squareform(pdist(clr(repl(d)).T.values, metric="euclidean"))}
    for name, D in mats.items():
        F, R2, pp = permanova(D, grp)
        means, pdp = permdisp(D, grp)
        verdict = ("location and/or dispersion - not separable"
                   if pp < 0.05 and pdp < 0.05 else
                   "location effect; dispersions homogeneous" if pp < 0.05 else
                   "no difference detected")
        print(f"\n{name}")
        print(f"  PERMANOVA  pseudo-F = {F:.3f}  R2 = {R2:.3f}  p = {pp:.4f}")
        print(f"  PERMDISP   preserved {means['P']:.4f}, degraded {means['D']:.4f}  p = {pdp:.4f}")
        print(f"  -> {verdict}")
        rows.append(dict(matrix=tag, distance=name, pseudo_F=round(F, 3), R2=round(R2, 3),
                         permanova_p=pp, disp_P=round(means["P"], 4),
                         disp_D=round(means["D"], 4), permdisp_p=pdp, reading=verdict))

    P = d / d.sum(axis=0)
    print("\n  Hill numbers on ARG categories")
    for q in (0, 1, 2):
        h = np.array([hill(P[c].values, q) for c in P.columns])
        _, pq = mannwhitneyu(h[grp == "P"], h[grp == "D"], alternative="two-sided")
        print(f"    q{q}: preserved {np.median(h[grp=='P']):7.2f}   "
              f"degraded {np.median(h[grp=='D']):7.2f}   p = {pq:.4f}")
    print()

pd.DataFrame(rows).to_csv(os.path.join(a.outdir, "Table_resistome_dispersion.csv"), index=False)
print(f"written to {a.outdir}/Table_resistome_dispersion.csv")
