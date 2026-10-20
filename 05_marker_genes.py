#!/usr/bin/env python3
"""
Figure: marker genes for the metabolic guilds, by stream
----------------------------------------------------------
A heatmap of KEGG orthologues that mark each guild's pathway, one column per
stream, grouped by guild and ordered by the electron acceptor sequence.

Streams are ordered along the land use gradient rather than by group, so a
trend across the panel can be read directly instead of being inferred from
two blocks. The group of each stream is still shown, by the colour of its
label.

The ordering variable is the land use intensity index - the first principal
component of the MapBiomas land cover classes and their diversity - computed
here exactly as it is for the guild analysis, so that one gradient runs
through the whole study. The Human Footprint Index orders the streams almost
as well (the denitrification markers follow it at rho = 0.72 against 0.67 for
the index) but it is built from land cover among other layers, which would
make cropland and urban cover both predictors and inputs. Using the index
everywhere costs little and leaves one axis to explain rather than two. Pass
--order-by with a column name to order by that column instead.

Why this is worth having alongside the taxonomic guilds: those are assigned
from the dominant metabolism of each genus, which is an inference from
taxonomy rather than a measurement of function. A marker gene measures the
pathway directly. Where gene and taxon agree, the guild assignment is
supported by something independent of it.

Three guilds cannot be checked this way, and the gaps are stated rather than
filled with a weak proxy:

  Fe(III) reduction has no universal marker. The outer-membrane cytochromes
  that carry it - omcB in Geobacter, mtrC in Shewanella - are lineage
  specific and have no shared orthologue, so the guild with the strongest
  gradient result in this dataset is the one gene content cannot confirm.

  Ammonia oxidation shares its orthologues with methane oxidation. AmoA and
  PmoA are homologous enzymes and KEGG assigns them the same identifiers
  (K10944-46), so an abundance at those KOs cannot be attributed to
  nitrifiers or to methanotrophs. They are plotted once, labelled as shared,
  and read as neither.

  Nitrite oxidation shares K00370 with nitrate reduction, for the same
  reason: NxrA and NarG are the same enzyme family running in opposite
  directions. K00370 is plotted under denitrification, where the rest of the
  pathway supports the reading.

Values are as supplied in the KO table, per 100,000 annotated ORFs. Each row
is z-scored across streams so that rows of different magnitude are comparable
in one panel; the raw values are written to file, since the z-score hides how
rare a marker is. Markers detected in fewer than half the samples are kept and
marked, because their rarity is itself the result for methanogenesis.

Inputs:
  --ko        KO abundance table (.xlsx or .csv), KEGG_KO in the first column
  --outdir    where to write (default: current directory)

Outputs:
  Fig_marker_genes.tif / .jpg
  marker_genes_raw.csv        values as supplied
  marker_genes_tests.csv      Mann-Whitney per marker, BH corrected

Usage:
  python3 fig_marker_genes.py --ko Table_S4_KO_abundance.xlsx
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAMPLES = [f"Pres0{i}" for i in range(1, 8)] + [f"Deg0{i}" for i in range(1, 8)]

# Markers by guild, in the order the electron acceptors are consumed. Each
# entry is (label, KO). Labels carry the gene name, since a KO number alone
# is unreadable in a figure.
MARKERS = [
    ("Ammonia / methane oxidation (shared)", [
        ("amoA / pmoA", "K10944"),
        ("amoB / pmoB", "K10945"),
        ("amoC / pmoC", "K10946"),
        ("hao", "K10535"),
    ]),
    ("Denitrification", [
        ("narG / nxrA", "K00370"),
        ("napA", "K02567"),
        ("nirK", "K00368"),
        ("nirS", "K15864"),
        ("norB", "K04561"),
        ("nosZ", "K00376"),
    ]),
    ("Sulfate reduction", [
        ("sat", "K00958"),
        ("aprA", "K00394"),
        ("dsrA", "K11180"),
        ("dsrB", "K11181"),
    ]),
    ("Sulfur oxidation", [
        ("soxB", "K17224"),
        ("fccB", "K17229"),
    ]),
    ("Methanogenesis", [
        ("fwdA", "K00200"),
        ("mtrA", "K00577"),
        ("mcrA", "K00399"),
        ("mcrB", "K00401"),
        ("mcrG", "K00402"),
        ("mtaB", "K04480"),
    ]),
]

# guilds with no usable marker, named in the figure so the gap is visible
NO_MARKER = ["Fe(III) reduction: no universal marker"]

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
ap.add_argument("--ko", required=True)
ap.add_argument("--land", default=None,
                help="landscape CSV; columns are ordered along the land use "
                     "gradient when given, by group otherwise")
ap.add_argument("--order-by", default=None,
                help="landscape column to order the streams by; the land use "
                     "intensity index (PC1) is used when omitted")
ap.add_argument("--outdir", default=".")
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)

d = (pd.read_excel(args.ko) if args.ko.lower().endswith((".xlsx", ".xls"))
     else pd.read_csv(args.ko))
d = d.set_index(d.columns[0])
d = d[~d.index.duplicated()]
cols = [c for c in SAMPLES if c in d.columns]
if len(cols) < 6:
    sys.exit(f"only {len(cols)} of the expected sample columns found")
d = d[cols]

pres = [c for c in cols if c.startswith("Pres")]
degr = [c for c in cols if c.startswith("Deg")]

# ---- column order ----------------------------------------------------
LAND_ORDER = [f"Deg0{i}" for i in range(1, 8)] + [f"Pres0{i}" for i in range(1, 8)]
gradient = None
if args.land:
    land = pd.read_csv(args.land, sep=";")
    if len(land) != len(LAND_ORDER):
        sys.exit(f"landscape file has {len(land)} rows, "
                 f"expected {len(LAND_ORDER)}")
    land["sample"] = LAND_ORDER
    land = land.set_index("sample")
    have = [c for c in cols if c in land.index]

    if args.order_by is None:
        # the same index as the guild analysis: first principal component of
        # the land cover classes, oriented to rise with land use intensity
        INDEX_METRICS = ["Forest.cover", "Cropland.cover", "Urban.cover",
                         "Diversity_q1"]
        absent = [m for m in INDEX_METRICS if m not in land.columns]
        if absent:
            sys.exit(f"missing from {args.land}: {absent}")
        Xi = land.loc[have, INDEX_METRICS].astype(float)
        Zi = (Xi - Xi.mean()) / Xi.std(ddof=1)
        Ui, Si, Vti = np.linalg.svd(Zi.values, full_matrices=False)
        var_i = Si ** 2 / (Si ** 2).sum()
        load_i = pd.Series(Vti[0], index=INDEX_METRICS)
        gradient = pd.Series(Ui[:, 0] * Si[0], index=have)
        if load_i["Forest.cover"] > 0:
            gradient, load_i = -gradient, -load_i
        order_label = f"land use intensity index (PC1, {100*var_i[0]:.1f}%)"
    else:
        if args.order_by not in land.columns:
            sys.exit(f"{args.order_by} not in {args.land}")
        gradient = land.loc[have, args.order_by].astype(float)
        order_label = args.order_by

    order = list(gradient.sort_values().index)
    missing_order = [c for c in cols if c not in order]
    order += missing_order
    print(f"streams ordered by {order_label}, least to most impacted:")
    print("  " + " < ".join(order))
    if missing_order:
        print(f"  [note] no landscape row for {missing_order}; placed last")
    print()
else:
    order = pres + degr
    print("streams ordered by group; pass --land to order them along the "
          "land use gradient\n")

rows, labels, guild_of, missing = [], [], {}, []
for guild, markers in MARKERS:
    for label, ko in markers:
        if ko not in d.index:
            missing.append((guild, label, ko))
            continue
        rows.append(d.loc[ko, order].astype(float).values)
        labels.append(label)
        guild_of[label] = guild

if missing:
    print("not present in the KO table, omitted:")
    for guild, label, ko in missing:
        print(f"  {guild}: {label} ({ko})")
    print()

m = pd.DataFrame(rows, index=labels, columns=order)
m.round(4).to_csv(os.path.join(args.outdir, "marker_genes_raw.csv"))

print(f"{len(m)} markers across {len(cols)} samples "
      f"({len(pres)} preserved, {len(degr)} degraded)\n")

# ---- per-marker test -------------------------------------------------
recs = []
for label in m.index:
    a, b = m.loc[label, pres], m.loc[label, degr]
    u, p = mannwhitneyu(a, b)
    recs.append(dict(marker=label, guild=guild_of[label],
                     detected_in=int((m.loc[label] > 0).sum()),
                     preserved_median=a.median(), degraded_median=b.median(),
                     log2FC=float(np.log2((b.mean() + 0.01) / (a.mean() + 0.01))),
                     p=p))
t = pd.DataFrame(recs)
t["q"] = bh(t.p)
t.round(4).to_csv(os.path.join(args.outdir, "marker_genes_tests.csv"),
                  index=False)

print(t[["marker", "guild", "detected_in", "preserved_median",
         "degraded_median", "log2FC", "q"]].to_string(
    index=False, float_format=lambda v: f"{v:.3f}"))
print(f"\n{(t.q < 0.05).sum()} of {len(t)} markers at q < 0.05")

sparse = t[t.detected_in < len(cols) / 2]
if len(sparse):
    print(f"\ndetected in fewer than half the samples: "
          f"{', '.join(sparse.marker)}")
    print("Their rarity is itself informative and they are kept in the figure,")
    print("marked with an open circle.")

# ---- z-score per row -------------------------------------------------
# rows differ in magnitude by two orders of magnitude, so a shared colour
# scale on raw values would show only which marker is abundant
z = m.sub(m.mean(axis=1), axis=0).div(m.std(axis=1, ddof=1).replace(0, np.nan),
                                      axis=0)

fig, ax = plt.subplots(figsize=(0.46 * len(m.columns) + 4.2,
                                0.30 * len(m) + 2.0))

vmax = float(np.nanmax(np.abs(z.values)))
im = ax.imshow(z.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
               aspect="auto", interpolation="nearest")

ax.set_xticks(np.arange(len(m.columns)))
ax.set_xticklabels(m.columns, rotation=90, fontsize=7.4)
for tick, s in zip(ax.get_xticklabels(), m.columns):
    tick.set_color(COL_PRES if s.startswith("Pres") else COL_DEG)

ax.set_yticks(np.arange(len(m)))
ax.set_yticklabels(m.index, fontsize=7.4)

# separator after the last row of each guild, and the guild name at the right
edges, names = [], []
prev = None
for i, label in enumerate(m.index):
    if prev is not None and guild_of[label] != prev:
        edges.append(i - 0.5)
    prev = guild_of[label]
for e in edges:
    ax.axhline(e, color="white", lw=2.2)
if gradient is None:
    ax.axvline(len(pres) - 0.5, color="white", lw=2.2)

start = 0
for i, label in enumerate(list(m.index) + [None]):
    g = guild_of[label] if label is not None else None
    if g != guild_of[m.index[start]]:
        mid = (start + i - 1) / 2
        ax.text(len(m.columns) - 0.35, mid,
                guild_of[m.index[start]].replace(" (shared)", "\n(shared)"),
                ha="left", va="center", fontsize=7.0, linespacing=1.3)
        start = i

# mark the sparse markers
for i, label in enumerate(m.index):
    if (m.loc[label] > 0).sum() < len(cols) / 2:
        ax.plot(-0.72, i, marker="o", ms=3.4, mfc="none", mec="#404040",
                mew=0.8, clip_on=False)

cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.20)
cb.set_label("z-score across streams", fontsize=7.6)
cb.ax.tick_params(labelsize=7.0)

ax.set_xlim(-1.1, len(m.columns) - 0.5)
for side in ("top", "right", "left", "bottom"):
    ax.spines[side].set_visible(False)
ax.tick_params(length=0)

fig.tight_layout()
base = os.path.join(args.outdir, "Fig_marker_genes")
fig.savefig(f"{base}.tif", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(f"{base}.jpg", dpi=300, pil_kwargs={"quality": 95})
plt.close(fig)

print(f"\nWritten: {base}.tif, {base}.jpg")
print("         marker_genes_raw.csv, marker_genes_tests.csv")
print("\nGuilds without a usable marker:")
for n in NO_MARKER:
    print(f"  {n}")
if gradient is not None:
    print(f"\nColumns run left to right from the lowest to the highest "
          f"{order_label}.")
print("\nSample labels are coloured by group; rows are z-scored, so the colour")
print("shows where a marker is high relative to its own mean, not how")
print("abundant it is. Open circles mark rows detected in fewer than half")
print("the samples.")
