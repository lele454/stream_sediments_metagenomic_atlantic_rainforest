#!/usr/bin/env python3
"""
Compositional reanalysis of the Atlantic Rainforest stream dataset
------------------------------------------------------------------
Genera, guilds and KEGG orthologs are all proportions, so they sum to a
constant and an apparent increase in one part can be the arithmetic
complement of a decrease in another. Every test here is run twice: once as
published, on proportions, and once on log-ratios, which the closure does not
touch.

For each of the three levels:

  PERMANOVA on Bray-Curtis (proportions) and on Aitchison distance (Euclidean
  on centred log-ratios), each with PERMDISP, so a difference in location is
  not confused with a difference in spread.

  Per-feature Mann-Whitney with Benjamini-Hochberg correction, on proportions
  and on CLR values.

  Hill numbers of orders 0, 1 and 2 on proportions.

And two tests that exist only to answer the closure objection:

  The whole analysis is repeated with rhizobia removed and the table reclosed.
  Anything produced by rhizobia giving up percentage points disappears there.
  Two rhizobia definitions are used, a strict one and a broad one, so the
  conclusion does not rest on where the boundary of the group is drawn.

  At guild level, every pairwise log-ratio between guilds is tested. A ratio
  is invariant to the total, so a shift in the balance between two guilds is
  a statement about them and not about the rest of the community.

Usage
  python3 atlantica_compositional.py \
      --genus Table_S6_genus_abundance.xlsx \
      --guilds TableS3_guild_assignments.csv \
      --ko Table_S4_KO_abundance.xlsx \
      --outdir .

Any of the three inputs may be omitted; the levels given are the levels run.
The guild file is read in the wide layout of Table S3: one row per guild, with
the member genera in a column of semicolon-separated names. Guilds whose name
contains "excluded" are skipped.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from scipy.spatial.distance import pdist, squareform

SEED, N_PERM = 42, 9999

# Rhizobia, under two definitions. Strict: the genera a reader would call
# rhizobia. Broad: the strict set plus close relatives in Nitrobacteraceae /
# Bradyrhizobiaceae. Nitrobacter sits in the broad set but is also a member of
# the NOB guild, which is why the two definitions are kept apart.
RHIZOBIA_STRICT = ["Bradyrhizobium", "Rhizobium", "Mesorhizobium", "Sinorhizobium",
                   "Ensifer", "Neorhizobium", "Azorhizobium", "Allorhizobium",
                   "Pararhizobium"]
RHIZOBIA_BROAD = RHIZOBIA_STRICT + ["Nitrobacter", "Afipia", "Rhodopseudomonas"]

ap = argparse.ArgumentParser()
ap.add_argument("--genus")
ap.add_argument("--guilds")
ap.add_argument("--ko")
ap.add_argument("--archaea", help="archaeal genus matrix, where archaeal guilds "
                                 "are expressed as a fraction of Archaea")
ap.add_argument("--preserved", default="Pres01,Pres02,Pres03,Pres04,Pres05,Pres06,Pres07")
ap.add_argument("--degraded", default="Deg01,Deg02,Deg03,Deg04,Deg05,Deg06,Deg07")
ap.add_argument("--outdir", default=".")
A = ap.parse_args()
os.makedirs(A.outdir, exist_ok=True)
if not any([A.genus, A.guilds, A.ko]):
    sys.exit("nothing to do: give at least --genus, --guilds or --ko")

PRES = [s.strip() for s in A.preserved.split(",")]
DEGR = [s.strip() for s in A.degraded.split(",")]
GROUPS = np.array(["P"] * len(PRES) + ["D"] * len(DEGR))


# ---- input ---------------------------------------------------------------
def read_any(path, sheet=0):
    if str(path).lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path, sheet_name=sheet)
    return pd.read_csv(path)


def load_matrix(path, what):
    d = read_any(path)
    d = d.rename(columns={d.columns[0]: what}).set_index(what)
    pres, degr = PRES, DEGR
    miss = [s for s in pres + degr if s not in d.columns]
    if miss:
        # the tables in this study label the same streams two ways,
        # P01/D01 and Pres01/Deg01. Accept either.
        alt = {"P": "Pres", "D": "Deg", "Pres": "P", "Deg": "D"}
        def swap(n):
            for a, b in alt.items():
                if n.startswith(a) and n[len(a):].isdigit():
                    return b + n[len(a):]
            return n
        cand_p, cand_d = [swap(x) for x in pres], [swap(x) for x in degr]
        if all(c in d.columns for c in cand_p + cand_d):
            pres, degr = cand_p, cand_d
            print(f"  {os.path.basename(path)}: using the {pres[0]}/{degr[0]} labels")
        else:
            sys.exit(f"{os.path.basename(path)}: columns not found: {miss}\n"
                     f"available: {list(d.columns)}")
    d = d[pres + degr].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return d.loc[d.sum(axis=1) > 0]


def load_guilds(path, domain=None):
    """Table S3 in its published wide layout -> {guild: [genera]}.

    Bacterial and archaeal guilds live in different matrices, because the
    archaeal percentages in this study are fractions of Archaea, not of the
    whole community. `domain` selects one of them."""
    g = read_any(path)
    gcol = next((c for c in g.columns if c.strip().lower() == "guild"), g.columns[0])
    dcol = next((c for c in g.columns if c.strip().lower() == "domain"), None)
    lcol = next((c for c in g.columns
                 if g[c].astype(str).str.contains(";").sum() > len(g) / 2), g.columns[-1])
    out = {}
    for _, row in g.iterrows():
        name = str(row[gcol]).strip()
        if "exclud" in name.lower():
            continue
        if domain and dcol and str(row[dcol]).strip().lower() != domain.lower():
            continue
        members = [x.strip() for x in str(row[lcol]).split(";") if x.strip()]
        if members:
            out[name] = members
    return out


# ---- compositional machinery --------------------------------------------
def close(df):
    return df / df.sum(axis=0)


def repl(df, frac=0.65):
    """Multiplicative replacement of zeros (Martin-Fernandez et al. 2003)."""
    P = close(df).copy()
    delta = frac * P.values[P.values > 0].min()
    for c in P.columns:
        v = P[c].values.astype(float)
        z = v == 0
        if z.any():
            v[z] = delta
            v[~z] = v[~z] * (1.0 - z.sum() * delta)
            P[c] = v
    return P


def clr(P):
    L = np.log(P.values)
    return pd.DataFrame(L - L.mean(axis=0, keepdims=True), index=P.index, columns=P.columns)


def gower(D):
    Aa = -0.5 * D ** 2
    n = Aa.shape[0]
    o = np.ones((n, n)) / n
    return Aa - o @ Aa - Aa @ o + o @ Aa @ o


def permanova(D, g, n_perm=N_PERM):
    rs = np.random.default_rng(SEED)
    G, n, lev = gower(D), D.shape[0], np.unique(g)

    def st(gg):
        H = np.zeros((n, n))
        for l in lev:
            i = np.where(gg == l)[0]
            H[np.ix_(i, i)] = 1.0 / len(i)
        t = np.trace(G)
        aa = np.trace(H @ G @ H)
        return (aa / (len(lev) - 1)) / ((t - aa) / (n - len(lev))), aa / t

    F, R2 = st(g)
    null = np.array([st(rs.permutation(g))[0] for _ in range(n_perm)])
    return F, R2, (np.sum(null >= F) + 1) / (n_perm + 1)


def permdisp(D, g, n_perm=N_PERM):
    """Anderson (2006): distance to the group centroid in principal coordinate
    space, with the correction for imaginary axes, then a permutation test."""
    rs = np.random.default_rng(SEED)
    w, V = np.linalg.eigh(gower(D))
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
        gg = rs.permutation(g)
        dd = dist(gg)
        null[i] = abs(dd[gg == lev[0]].mean() - dd[gg == lev[1]].mean())
    return {l: d[g == l].mean() for l in lev}, (np.sum(null >= obs) + 1) / (n_perm + 1)


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(np.argsort(p)[::-1]):
        prev = min(prev, p[i] * n / (n - rank))
        q[i] = prev
    return q


def feature_tests(mat, label):
    rows = []
    for name, v in mat.iterrows():
        a, b = v[GROUPS == "P"].values, v[GROUPS == "D"].values
        if np.allclose(np.concatenate([a, b]), a[0]):
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        rows.append(dict(level=label, feature=name, median_P=np.median(a),
                         median_D=np.median(b), U=float(u), p=p))
    out = pd.DataFrame(rows)
    if len(out):
        out["q"] = bh(out.p.values)
        out = out.sort_values("p").reset_index(drop=True)
    return out


def hill(p, q):
    p = p[p > 0]
    if q == 0:
        return float(len(p))
    if abs(q - 1) < 1e-9:
        return float(np.exp(-(p * np.log(p)).sum()))
    return float((p ** q).sum() ** (1.0 / (1.0 - q)))


# ---- one level, one table ------------------------------------------------
STATS, TESTS, HILLS = [], [], []


def analyse(df, level, tag, show=8):
    P = close(df)
    C = clr(repl(df))
    print(f"\n{'-' * 72}\n{level} | {tag}  ({df.shape[0]} features)\n{'-' * 72}")

    for dname, D in (("Bray-Curtis on proportions", squareform(pdist(P.T.values, "braycurtis"))),
                     ("Aitchison (CLR)", squareform(pdist(C.T.values, "euclidean")))):
        F, R2, p = permanova(D, GROUPS)
        m, pdp = permdisp(D, GROUPS)
        reading = ("location and/or dispersion, not separable"
                   if p < 0.05 and pdp < 0.05 else
                   "location effect, dispersions homogeneous" if p < 0.05 else
                   "no difference detected")
        print(f"  {dname:30s} F = {F:6.3f}  R2 = {R2:.3f}  p = {p:.4f}   "
              f"| PERMDISP p = {pdp:.4f}  -> {reading}")
        STATS.append(dict(level=level, table=tag, distance=dname, pseudo_F=round(F, 3),
                          R2=round(R2, 3), permanova_p=p, disp_P=round(m["P"], 4),
                          disp_D=round(m["D"], 4), permdisp_p=pdp, reading=reading))

    # the matrix arrives in the units the manuscript reports (per cent of the
    # community, or ORFs per 100,000). Re-closing it here would silently turn a
    # guild's share of the community into its share of the guild set, so the
    # tests run on the values as given; only the log-ratio version is derived.
    for mat, lab in ((df, "as reported"), (C, "CLR")):
        t = feature_tests(mat, f"{level} | {tag} | {lab}")
        if not len(t):
            continue
        TESTS.append(t)
        nsig = int((t.q < 0.05).sum())
        print(f"  Mann-Whitney on {lab:10s}: {nsig} with q < 0.05"
              + ("" if nsig else f"   (lowest q = {t.q.min():.3f})"))
        if nsig:
            print(t[t.q < 0.05].head(show).to_string(index=False, float_format="%.4f"))

    h = pd.DataFrame([dict(level=level, table=tag, sample=s, group=g,
                           q0=hill(P[s].values, 0), q1=hill(P[s].values, 1),
                           q2=hill(P[s].values, 2))
                      for s, g in zip(P.columns, GROUPS)])
    HILLS.append(h)
    line = []
    for q in ("q0", "q1", "q2"):
        a, b = h.loc[h.group == "P", q].values, h.loc[h.group == "D", q].values
        _, p = mannwhitneyu(a, b, alternative="two-sided")
        line.append(f"{q}: {np.median(a):.2f} -> {np.median(b):.2f} (p = {p:.4f})")
    print("  Hill  " + "   ".join(line))


# ---- genera ---------------------------------------------------------------
genus = None
if A.genus:
    genus = load_matrix(A.genus, "Genus")
    print(f"\n{'=' * 72}\nGENERA\n{'=' * 72}")
    print(f"{genus.shape[0]} genera, {(genus == 0).sum().sum() / genus.size * 100:.1f}% zeros")
    analyse(genus, "genus", "full table")
    for name, lst in (("strict", RHIZOBIA_STRICT), ("broad", RHIZOBIA_BROAD)):
        present = [g for g in lst if g in genus.index]
        share = 100 * close(genus).loc[present].sum().mean()
        print(f"\nrhizobia removed, {name} definition: {len(present)} genera, "
              f"{share:.1f}% of the average sample")
        analyse(genus.drop(index=present), "genus", f"rhizobia removed ({name})")

# ---- guilds ---------------------------------------------------------------
def guild_block(base, gmap, domain_label, rhizobia=True):
    rows, miss = {}, {}
    for guild, members in gmap.items():
        found = [m for m in members if m in base.index and base.loc[m].sum() > 0]
        miss[guild] = len(members) - len(found)
        if found:
            rows[guild] = base.loc[found].sum(axis=0)
    guilds = pd.DataFrame(rows).T
    print(f"\n{'=' * 72}\nGUILDS, {domain_label}\n{'=' * 72}")
    print(f"{guilds.shape[0]} guilds retained")
    for k, v in miss.items():
        if v:
            print(f"  {k}: {v} assigned genera absent from the matrix")
    share = 100 * guilds.sum(axis=0) / base.sum(axis=0)
    print(f"together they are {share.min():.1f}% to {share.max():.1f}% of the matrix")

    print("\n(a) each guild as a fraction of the matrix, as published")
    analyse(guilds.div(base.sum(axis=0), axis=1) * 100, "guild", f"{domain_label}, as published")

    if rhizobia:
        for name, lst in (("strict", RHIZOBIA_STRICT), ("broad", RHIZOBIA_BROAD)):
            present = [g for g in lst if g in base.index]
            if not present:
                continue
            sub = base.drop(index=present)
            rows2 = {k: sub.loc[[m for m in v if m in sub.index]].sum(axis=0)
                     for k, v in gmap.items()
                     if any(m in sub.index for m in v)}
            gg = pd.DataFrame(rows2).T
            print(f"\n(b) the same, with rhizobia removed and the matrix reclosed "
                  f"({name} definition)")
            analyse(gg.div(sub.sum(axis=0), axis=1) * 100, "guild",
                    f"{domain_label}, rhizobia removed ({name})")

    print("\n(c) the balance among the guilds themselves, which no total can affect")
    analyse(guilds, "guild", f"{domain_label}, closed set")

    print("\n(d) every pairwise log-ratio between guilds")
    Gr = repl(guilds)
    names = list(Gr.index)
    rows3 = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            lr = np.log(Gr.iloc[i].values / Gr.iloc[j].values)
            _, p = mannwhitneyu(lr[GROUPS == "P"], lr[GROUPS == "D"], alternative="two-sided")
            rows3.append(dict(domain=domain_label, ratio=f"{names[i]} / {names[j]}",
                              median_P=np.median(lr[GROUPS == "P"]),
                              median_D=np.median(lr[GROUPS == "D"]), p=p))
    lrt = pd.DataFrame(rows3)
    lrt["q"] = bh(lrt.p.values)
    lrt = lrt.sort_values("p").reset_index(drop=True)
    nsig = int((lrt.q < 0.05).sum())
    print(f"  {len(lrt)} ratios, {nsig} with q < 0.05")
    print(lrt.head(10).to_string(index=False, float_format="%.4f"))
    return lrt


LR = []
if A.guilds and genus is not None:
    LR.append(guild_block(genus, load_guilds(A.guilds, "Bacteria"), "Bacteria"))
if A.guilds and A.archaea:
    arch = load_matrix(A.archaea, "Genus")
    print(f"\narchaeal matrix: {arch.shape[0]} genera")
    LR.append(guild_block(arch, load_guilds(A.guilds, "Archaea"), "Archaea", rhizobia=False))
if LR:
    pd.concat(LR).to_csv(os.path.join(A.outdir, "Table_guild_logratios.csv"), index=False)

# ---- KEGG orthologs -------------------------------------------------------
if A.ko:
    ko = load_matrix(A.ko, "KO")
    print(f"\n\n{'=' * 72}\nKEGG ORTHOLOGS\n{'=' * 72}")
    print(f"{ko.shape[0]} orthologs, {(ko == 0).sum().sum() / ko.size * 100:.1f}% zeros")
    analyse(ko, "KO", "full table", show=12)

# ---- output ---------------------------------------------------------------
pd.DataFrame(STATS).to_csv(os.path.join(A.outdir, "Table_permanova_permdisp.csv"), index=False)
if TESTS:
    pd.concat(TESTS).to_csv(os.path.join(A.outdir, "Table_feature_tests.csv"), index=False)
if HILLS:
    pd.concat(HILLS).to_csv(os.path.join(A.outdir, "Table_hill_numbers.csv"), index=False)
print(f"\nwritten to {A.outdir}/: Table_permanova_permdisp.csv, Table_feature_tests.csv, "
      "Table_hill_numbers.csv" + (", Table_guild_logratios.csv" if LR else ""))
