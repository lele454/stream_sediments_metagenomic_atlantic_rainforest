#!/usr/bin/env python3
"""
Figure: relative abundance of the twenty most abundant genera per stream
--------------------------------------------------------------------------
Stacked bars, one per sample, with the twenty most abundant genera across the
dataset and everything else pooled as "Other". Samples are grouped by stream
status with a gap between the groups.

The twenty are chosen by mean relative abundance across all samples, not per
sample. Picking the top twenty of each sample separately would give each bar a
different set of genera and make the bars incomparable, which is the one thing
a stacked bar chart is for.

"Other" is usually the largest single block in a metagenome and that is worth
seeing rather than hiding: it says what fraction of the community the named
genera actually account for. It is drawn in grey at the top of each bar.

Genus names are italicised. Ranks above genus that Bracken reports at the
genus level - order or family names for reads it could not place further - are
left upright, since they are not genus names.

Inputs:
  --bracken   directory of Bracken genus reports, one per sample
  --top       how many genera to name (default 20)
  --outdir    where to write (default: current directory)

Outputs:
  Fig_genus_barplot.tif / .jpg
  genus_relative_abundance.csv    the full matrix, all genera
  genus_top_abundance.csv         the plotted matrix, top N plus Other

Usage:
  python3 fig_genus_barplot.py --bracken .
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SAMPLES = [f"Pres0{i}" for i in range(1, 8)] + [f"Deg0{i}" for i in range(1, 8)]

OTHER = "Other"
COL_OTHER = "#d9d9d9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "savefig.bbox": "tight",
})


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


def is_genus(name):
    """
    A single capitalised word is taken as a genus. Bracken labels some reads
    with an order or family name at the genus level when it cannot place them
    further; those end in -ales or -aceae and should not be italicised, and
    neither should the pooled Other category.
    """
    n = str(name)
    if n == OTHER or " " in n or not n[:1].isupper():
        return False
    return not (n.endswith("ales") or n.endswith("aceae") or
                n.endswith("ineae") or n.endswith("aria"))


def palette(n):
    """
    Twenty categorical colours: tab20 alone repeats hues in adjacent pairs,
    which is hard to follow in a stacked bar, so it is interleaved with tab20b
    to separate neighbouring blocks.
    """
    a = plt.get_cmap("tab20").colors
    b = plt.get_cmap("tab20b").colors
    pool = [c for pair in zip(a[::2], a[1::2]) for c in pair] + list(b)
    return pool[:n]


# ---------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--bracken", required=True)
ap.add_argument("--top", type=int, default=20)
ap.add_argument("--outdir", default=".")
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)

m = read_bracken(args.bracken)
rel = 100 * m / m.sum()
rel.round(5).to_csv(os.path.join(args.outdir, "genus_relative_abundance.csv"))

print(f"{rel.shape[0]} genera, {rel.shape[1]} samples")

# ranked by mean across samples, so every bar shows the same genera
top = rel.mean(axis=1).sort_values(ascending=False).index[:args.top]
plotted = rel.loc[top].copy()
plotted.loc[OTHER] = rel.drop(index=top).sum()
plotted.round(5).to_csv(os.path.join(args.outdir, "genus_top_abundance.csv"))

print(f"\nthe {args.top} named genera account for "
      f"{plotted.loc[top].sum().min():.1f}% to "
      f"{plotted.loc[top].sum().max():.1f}% of the community per sample")
print(f"Other ranges from {plotted.loc[OTHER].min():.1f}% to "
      f"{plotted.loc[OTHER].max():.1f}%\n")
print(pd.DataFrame({"mean %": rel.loc[top].mean().round(3)}).T.to_string())
print("\nnamed genera, mean relative abundance:")
for g in top:
    print(f"  {g:28s} {rel.loc[g].mean():6.2f}%")

# ---------------------------------------------------------------------
pres = [c for c in plotted.columns if c.startswith("Pres")]
degr = [c for c in plotted.columns if c.startswith("Deg")]
order = pres + degr

# a gap between the groups, so the two are read as blocks
x = np.arange(len(order), dtype=float)
x[len(pres):] += 0.8

colours = dict(zip(top, palette(len(top))))
colours[OTHER] = COL_OTHER

fig, ax = plt.subplots(figsize=(8.2, 4.4))

bottom = np.zeros(len(order))
for g in list(top) + [OTHER]:
    vals = plotted.loc[g, order].values
    ax.bar(x, vals, bottom=bottom, width=0.78, color=colours[g],
           edgecolor="white", linewidth=0.4,
           label=g, zorder=2)
    bottom += vals

ax.set_xticks(x)
ax.set_xticklabels(order, rotation=90, fontsize=8)
ax.set_ylabel("Relative abundance (%)")
ax.set_ylim(0, 100)
ax.set_xlim(x[0] - 0.7, x[-1] + 0.7)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

# group labels below the sample names
ax.annotate("Preserved", xy=(np.mean(x[:len(pres)]), -0.17),
            xycoords=("data", "axes fraction"), ha="center", va="top",
            fontsize=9, annotation_clip=False)
ax.annotate("Degraded", xy=(np.mean(x[len(pres):]), -0.17),
            xycoords=("data", "axes fraction"), ha="center", va="top",
            fontsize=9, annotation_clip=False)

handles = [Patch(facecolor=colours[g], edgecolor="white", linewidth=0.4,
                 label=g) for g in list(top) + [OTHER]]
leg = ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
                frameon=False, fontsize=7.6, handlelength=1.1,
                handleheight=1.1, labelspacing=0.45)
for text, g in zip(leg.get_texts(), list(top) + [OTHER]):
    if is_genus(g):
        text.set_style("italic")

fig.tight_layout()
base = os.path.join(args.outdir, "Fig_genus_barplot")
fig.savefig(f"{base}.tif", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(f"{base}.jpg", dpi=300, pil_kwargs={"quality": 95})
plt.close(fig)

print(f"\nWritten: {base}.tif, {base}.jpg")
print("         genus_relative_abundance.csv, genus_top_abundance.csv")
