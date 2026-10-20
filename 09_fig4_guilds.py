#!/usr/bin/env python3
"""
Figure 4: metabolic guilds along the redox sequence and the land use gradient
-------------------------------------------------------------------------------
One figure, drawn in one pass. Panel (a) places every guild in the order its
electron acceptor is consumed and shows the change between stream groups; the
remaining panels show the guilds that survive correction, first as a group
comparison and then against the land use intensity index.

Drawn together rather than assembled from two images. Two figures glued one
above the other either stretch the narrower row until its panels tower over
the rest, or leave a block of white where it ends, and in both cases the
panels no longer share a scale. A single gridspec gives every panel the same
size and the same type, which is the only way the letters mean anything.

The right-hand block is two rows of three. With four guilds surviving
correction and two of them also tracking the gradient, six panels fill it
exactly; a layout of four over two would leave half a row empty.

Inputs, all written by 04_guilds_redox.py and 08_guilds_gradient.py:
  --guilds    guilds_redox.csv          within-domain proportions
  --tests     guilds_redox_tests.csv    group comparison, with q
  --gradient  guilds_gradient.csv       correlation with the index, with q
  --index     land_use_index.csv        the index itself, per stream

Usage
  python3 fig4_guilds.py --guilds guilds_redox.csv --tests guilds_redox_tests.csv \
      --gradient guilds_gradient.csv --index land_use_index.csv --outdir Figures
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.stats import theilslopes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COL_PRES, COL_DEG, COL_NS = "#2c7bb6", "#d7191c", "#bdbdbd"

# the order the electron acceptors are consumed, and the domain of each guild
GUILD_ORDER = [
    ("Methanotroph", "B"), ("AOA (ammonia oxidiser)", "A"),
    ("AOB (ammonia oxidiser)", "B"), ("NOB (nitrite oxidiser)", "B"),
    ("Denitrifiers", "B"), ("Fe(III) reducer", "B"),
    ("Sulfate reducer", "B"), ("Sulfur oxidiser", "B"),
    ("Methanogen (H2/CO2)", "A"), ("Methanogen (acetate)", "A"),
    ("Methanogen (methyl)", "A"), ("ANME (anaerobic methane ox.)", "A"),
]

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                     "axes.linewidth": 0.8, "savefig.bbox": "tight"})

ap = argparse.ArgumentParser()
ap.add_argument("--guilds", required=True)
ap.add_argument("--tests", required=True)
ap.add_argument("--gradient", required=True)
ap.add_argument("--index", required=True)
ap.add_argument("--outdir", default=".")
ap.add_argument("--name", default="Fig4_guilds")
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)

g = pd.read_csv(a.guilds, index_col=0)
t = pd.read_csv(a.tests, index_col=0)
grad = pd.read_csv(a.gradient).set_index("guild")
idx = pd.read_csv(a.index).set_index("sample")

samples = [c for c in g.columns if c in idx.index]
if len(samples) < 6:
    sys.exit(f"only {len(samples)} samples shared between {a.guilds} and {a.index}")
pres = [s for s in samples if s.startswith("Pres")]
degr = [s for s in samples if s.startswith("Deg")]
x = idx.loc[samples, "PC1"].astype(float).values
is_pres = np.array([s.startswith("Pres") for s in samples])

t = t.reindex([gd for gd, _ in GUILD_ORDER if gd in t.index])
sig = t[t.q < 0.05]
sig_grad = grad[grad.q < 0.05]
print(f"{len(t)} guilds; {len(sig)} differ between groups, "
      f"{len(sig_grad)} track the index")

n_right = len(sig) + len(sig_grad)
ncol = 3
nrow = int(np.ceil(n_right / ncol))
print(f"right-hand block: {nrow} rows of {ncol}, {n_right} panels")

fig = plt.figure(figsize=(4.0 + 1.95 * ncol, 2.3 * nrow + 0.6))
outer = fig.add_gridspec(1, 2, width_ratios=[4.0, 1.95 * ncol], wspace=0.22)
axA = fig.add_subplot(outer[0, 0])
inner = outer[0, 1].subgridspec(nrow, ncol, wspace=0.55, hspace=0.62)

# ---- (a) the whole sequence ------------------------------------------
plotted = list(t.index)[::-1]
domain_of = dict(GUILD_ORDER)
for i, guild in enumerate(plotted):
    r = t.loc[guild]
    colour = (COL_DEG if r.log2FC > 0 else COL_PRES) if r.q < 0.05 else COL_NS
    marker = "o" if domain_of.get(guild, "B") == "B" else "s"
    axA.plot([0, r.log2FC], [i, i], color=colour, lw=1.4, zorder=2)
    axA.scatter(r.log2FC, i, s=62, c=colour, marker=marker,
                edgecolors="white", linewidths=0.8, zorder=3)
    if r.q < 0.05:
        axA.text(0.16, i + 0.30, f"$q$ = {r.q:.3f}", fontsize=6.8,
                 va="center", ha="left", color="#404040")

axA.text(-0.62, 1.02, "(a)", transform=axA.transAxes, fontsize=11,
         fontweight="bold", va="bottom", ha="left")
axA.axvline(0, color="#1a1a1a", lw=0.8, zorder=1)
axA.set_yticks(np.arange(len(plotted)))
axA.set_yticklabels(plotted, fontsize=7.6)
axA.set_xlabel("log$_2$ fold change, degraded / preserved", fontsize=8)
lo, hi = axA.get_xlim()
axA.set_xlim(lo - 0.25, hi + 0.60)
axA.set_ylim(-0.8, len(plotted) - 0.2)
for side in ("top", "right", "left"):
    axA.spines[side].set_visible(False)
axA.tick_params(axis="y", length=0)
axA.legend(handles=[
    Line2D([], [], marker="o", linestyle="", color="#595959", markersize=6,
           label="Bacteria"),
    Line2D([], [], marker="s", linestyle="", color="#595959", markersize=6,
           label="Archaea")],
    loc="lower right", frameon=False, fontsize=7.4, handletextpad=0.4)

# ---- the right-hand block --------------------------------------------
LETTERS = "bcdefghij"
rng = np.random.default_rng(42)
k = 0


def title_of(name):
    return name.replace(" / ", "\n").replace(" (", "\n(")


for guild in sig.index:
    ax = fig.add_subplot(inner[k // ncol, k % ncol])
    groups = [g.loc[guild, pres].values, g.loc[guild, degr].values]
    bp = ax.boxplot(groups, widths=0.55, patch_artist=True,
                    medianprops=dict(color="#1a1a1a", linewidth=1.1),
                    whiskerprops=dict(color="#595959", linewidth=0.7),
                    capprops=dict(color="#595959", linewidth=0.7),
                    flierprops=dict(marker="", linestyle="none"))
    for patch, colour in zip(bp["boxes"], (COL_PRES, COL_DEG)):
        patch.set_facecolor(colour)
        patch.set_alpha(0.26)
        patch.set_edgecolor(colour)
        patch.set_linewidth(0.9)
    for i, (vals, colour) in enumerate(zip(groups, (COL_PRES, COL_DEG)), start=1):
        ax.scatter(rng.normal(i, 0.055, len(vals)), vals, s=22, c=colour,
                   edgecolors="white", linewidths=0.5, zorder=3)
    q = t.loc[guild, "q"]
    ax.text(0.04, 0.96, f"$q$ = {q:.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.6)
    ax.text(-0.34, 1.12, f"({LETTERS[k]})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["P", "D"], fontsize=8)
    ax.set_title(title_of(guild), fontsize=7.0, pad=7, linespacing=1.3)
    if k % ncol == 0:
        ax.set_ylabel("% within domain", fontsize=8)
    ax.tick_params(labelsize=7.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    k += 1

for guild, r in sig_grad.sort_values("rho", ascending=False).iterrows():
    ax = fig.add_subplot(inner[k // ncol, k % ncol])
    y = g.loc[guild, samples].astype(float).values
    for sel, colour in ((is_pres, COL_PRES), (~is_pres, COL_DEG)):
        ax.scatter(x[sel], y[sel], s=22, c=colour, edgecolors="white",
                   linewidths=0.5, zorder=3)
    # Theil-Sen: the median of all pairwise slopes, rank-based like the
    # Spearman coefficient printed beside it
    slope, intercept, _, _ = theilslopes(y, x)
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, intercept + slope * xs, color="#757575", lw=1.1, zorder=2)
    ax.text(0.04, 0.96,
            f"$\\rho$ = {r.rho:.2f}\n" +
            ("$q$ < 0.001" if r.q < 0.001 else f"$q$ = {r.q:.3f}"),
            transform=ax.transAxes, ha="left", va="top", fontsize=6.6,
            linespacing=1.4)
    ax.text(-0.34, 1.12, f"({LETTERS[k]})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")
    ax.set_title(title_of(guild), fontsize=7.0, pad=7, linespacing=1.3)
    # kept short: the full name runs past the right edge of the last column
    ax.set_xlabel("Land use (PC1)", fontsize=7.6)
    if k % ncol == 0:
        ax.set_ylabel("% within domain", fontsize=8)
    ax.tick_params(labelsize=7.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    k += 1

base = os.path.join(a.outdir, a.name)
fig.savefig(f"{base}.tif", dpi=300, bbox_inches="tight", pad_inches=0.06,
            pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(f"{base}.jpg", dpi=300, bbox_inches="tight", pad_inches=0.06,
            pil_kwargs={"quality": 95})
plt.close(fig)
from PIL import Image
w, h = Image.open(f"{base}.tif").size
print(f"\nwritten: {base}.tif / .jpg   {w} x {h} px "
      f"({w/300:.1f} x {h/300:.1f} in at 300 dpi)")
print("(a) every guild in the order its acceptor is consumed; "
      f"({LETTERS[0]}-{LETTERS[len(sig)-1]}) the guilds that differ between "
      f"groups; ({LETTERS[len(sig)]}-{LETTERS[k-1]}) those that track the index.")
