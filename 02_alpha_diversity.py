#!/usr/bin/env python3
"""
Figure: alpha diversity of the stream communities
---------------------------------------------------
Boxplots of Hill numbers q = 0, 1 and 2 with Shannon entropy, compared between
preserved and degraded streams, in two rows: the whole community, and the
community with the rhizobia removed and the table reclosed.

The second row exists because proportions sum to a constant. Rhizobia are the
dominant group of the preserved streams and lose about ten percentage points
in the degraded ones; whatever they give up has to reappear somewhere, and an
evenness index will register that as a more even community whether or not
anything else changed. Removing them and recomputing answers the question the
first row cannot: is the rest of the community differently structured, or was
the difference one group of taxa all along.

Each panel carries the Mann-Whitney p value for that index. Shannon entropy H and the Hill number of
order one are the same quantity on different scales, q1 = exp(H), so they will
always tell the same story and are shown together only because both are
conventionally reported. The information is in the series across orders: q0
counts genera regardless of abundance, q1 weights them by frequency, q2
weights the common ones more heavily still. A difference at q0 but not at q2
means rare genera differ while the dominant ones do not, which is a different
result from a difference across all three.

Richness is sensitive to sequencing depth, and depth differs severalfold here.
Counts are therefore rarefied to the smallest sample total, averaged over
repeated draws, before q0 is computed. q1 and q2 are computed on proportions,
where the effect of depth is weaker but not absent.

Inputs:
  --bracken   directory of Bracken genus reports, one per sample
  --outdir    where to write (default: current directory)
  --n-rare    rarefaction draws (default 100)

Outputs:
  Fig_alpha_diversity.tif / .jpg
  alpha_diversity.csv
  alpha_diversity_tests.csv

Usage:
  python3 fig_alpha_diversity.py --bracken bracken/
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAMPLES = [f"Pres0{i}" for i in range(1, 8)] + [f"Deg0{i}" for i in range(1, 8)]

# Rhizobia and close relatives in Nitrobacteraceae. Nitrobacter, Afipia and
# Rhodopseudomonas sit in the same family as Bradyrhizobium but are not
# rhizobia in the usual sense, so they are kept out of the default set and
# can be added with --rhizobia-broad as a sensitivity check.
RHIZOBIA_STRICT = ["Bradyrhizobium", "Rhizobium", "Mesorhizobium",
                   "Sinorhizobium", "Ensifer", "Neorhizobium", "Azorhizobium",
                   "Allorhizobium", "Pararhizobium"]
RHIZOBIA_BROAD = RHIZOBIA_STRICT + ["Nitrobacter", "Afipia", "Rhodopseudomonas"]

COL_PRES = "#2c7bb6"
COL_DEG = "#d7191c"

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
            print(f"[skipped] {os.path.basename(path)}: unexpected columns")
            continue
        d = d[d.taxonomy_lvl == "G"]
        counts[s] = d.set_index("name").new_est_reads
    if not counts:
        sys.exit(f"no Bracken reports recognised in {directory}")
    m = pd.DataFrame(counts).fillna(0)
    return m.reindex(columns=[s for s in SAMPLES if s in m.columns])


def rarefied_richness(counts, depth, n_draws, rng):
    """Mean genera observed in repeated draws of `depth` reads."""
    counts = counts[counts > 0].astype(int)
    total = counts.sum()
    if total <= depth:
        return float((counts > 0).sum())
    p = counts.values / total
    obs = [int((rng.multinomial(depth, p) > 0).sum()) for _ in range(n_draws)]
    return float(np.mean(obs))


def hill(p, q):
    p = p[p > 0]
    if q == 1:
        return float(np.exp(-(p * np.log(p)).sum()))
    return float((p ** q).sum() ** (1 / (1 - q)))


# ---------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--bracken", required=True)
ap.add_argument("--outdir", default=".")
ap.add_argument("--n-rare", type=int, default=100)
ap.add_argument("--rhizobia-broad", action="store_true",
                help="include Nitrobacter, Afipia and Rhodopseudomonas among "
                     "the genera removed in the second row")
args = ap.parse_args()

os.makedirs(args.outdir, exist_ok=True)
rng = np.random.default_rng(42)

m = read_bracken(args.bracken)
depth = int(m.sum().min())
print(f"{m.shape[0]} genera, {m.shape[1]} samples")
print(f"reads per sample: {int(m.sum().min())} to {int(m.sum().max())}")
print(f"rarefying to {depth} reads, {args.n_rare} draws\n")

RHIZOBIA = RHIZOBIA_BROAD if args.rhizobia_broad else RHIZOBIA_STRICT
present = [g for g in RHIZOBIA if g in m.index]
share = 100 * (m.loc[present].sum() / m.sum())
print(f"rhizobia removed in the second row: {len(present)} genera "
      f"({', '.join(present)})")
print(f"they are {share.min():.1f}% to {share.max():.1f}% of a sample "
      f"(mean {share.mean():.1f}%)\n")

TABLES = [("full", m), ("no_rhizobia", m.drop(index=present))]


def diversity_table(counts, depth, label):
    out = []
    for s_ in counts.columns:
        col = counts[s_]
        p = (col / col.sum()).values
        out.append(dict(
            table=label,
            sample=s_,
            status="preserved" if s_.startswith("Pres") else "degraded",
            reads=int(col.sum()),
            q0_rarefied=round(rarefied_richness(col, depth, args.n_rare, rng), 1),
            q0_observed=int((col > 0).sum()),
            q1=round(hill(p, 1), 2),
            q2=round(hill(p, 2), 2),
            shannon=round(float(np.log(hill(p, 1))), 3)))
    return pd.DataFrame(out)


frames = {lab: diversity_table(tab, depth, lab) for lab, tab in TABLES}
d = frames["full"]
dd = pd.concat(frames.values(), ignore_index=True)
dd.to_csv(os.path.join(args.outdir, "alpha_diversity.csv"), index=False)
print(dd.to_string(index=False))

# ---------------------------------------------------------------------
INDICES = [("q0_rarefied", "$^{0}D$ (rarefied)"),
           ("q1", "$^{1}D$"),
           ("q2", "$^{2}D$")]

TABLE_TITLE = {"full": "whole community",
               "no_rhizobia": "rhizobia removed, table reclosed"}

tests = {}
trows = []
for lab, frame in frames.items():
    print(f"\n--- preserved vs degraded, {TABLE_TITLE[lab]} ---")
    for key in [k for k, _ in INDICES] + ["shannon"]:
        a = frame.loc[frame.status == "preserved", key]
        b = frame.loc[frame.status == "degraded", key]
        u, p = mannwhitneyu(a, b)
        tests[(lab, key)] = dict(table=lab, index=key,
                                 preserved_median=round(a.median(), 3),
                                 degraded_median=round(b.median(), 3),
                                 U=float(u), p=round(p, 4))
        trows.append(tests[(lab, key)])
        note = "" if key in dict(INDICES) else "   (table only)"
        print(f"  {key:14s} preserved {a.median():9.2f}  degraded {b.median():9.2f}"
              f"   U = {u:5.1f}  p = {p:.4f}{note}")

pd.DataFrame(trows).to_csv(
    os.path.join(args.outdir, "alpha_diversity_tests.csv"), index=False)

print("\nShannon is in the table but not plotted: H = ln(q1), so its test is"
      "\nidentical to that of q1 by construction.")

# ---------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(6.8, 6.0))
letters = iter("abcdef")

for row, (lab, frame) in enumerate(frames.items()):
    for col, (key, ylabel) in enumerate(INDICES):
        ax = axes[row, col]
        groups = [frame.loc[frame.status == g, key].values
                  for g in ("preserved", "degraded")]

        bp = ax.boxplot(groups, widths=0.55, patch_artist=True,
                        medianprops=dict(color="#1a1a1a", linewidth=1.2),
                        whiskerprops=dict(color="#595959", linewidth=0.8),
                        capprops=dict(color="#595959", linewidth=0.8),
                        flierprops=dict(marker="", linestyle="none"))
        for patch, colour in zip(bp["boxes"], (COL_PRES, COL_DEG)):
            patch.set_facecolor(colour)
            patch.set_alpha(0.28)
            patch.set_edgecolor(colour)
            patch.set_linewidth(1.0)

        # the points matter more than the box with seven per group
        for i, (vals, colour) in enumerate(zip(groups, (COL_PRES, COL_DEG)),
                                           start=1):
            x = rng.normal(i, 0.055, size=len(vals))
            ax.scatter(x, vals, s=24, c=colour, edgecolors="white",
                       linewidths=0.6, zorder=3)

        # the p value sits above a bracket spanning the two groups, placed in
        # axis coordinates so it clears the highest point whatever the scale
        pv = tests[(lab, key)]["p"]
        lo, hi = ax.get_ylim()
        span = hi - lo
        ax.set_ylim(lo, hi + 0.16 * span)
        y = hi + 0.03 * span
        ax.plot([1, 1, 2, 2], [y, y + 0.02 * span, y + 0.02 * span, y],
                color="#595959", linewidth=0.8, clip_on=False)
        ax.text(1.5, y + 0.045 * span,
                "$p$ < 0.001" if pv < 0.001 else f"$p$ = {pv:.3f}",
                ha="center", va="bottom", fontsize=7.5, color="#1a1a1a")

        ax.text(-0.30, 1.04, f"({next(letters)})", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="bottom", ha="left")

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["P", "D"])
        ax.set_ylabel(ylabel)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

fig.tight_layout()
# The row label is placed after the layout is fixed, from the real position of
# the row, so it never lands on the y axis of a four-digit richness scale the
# way an offset in axes coordinates would.
for row, lab in enumerate(frames):
    box = axes[row, 0].get_position()
    fig.text(0.012, box.y0 + box.height / 2, TABLE_TITLE[lab],
             rotation=90, ha="left", va="center", fontsize=8.5)
fig.subplots_adjust(left=0.135)

base = os.path.join(args.outdir, "Fig_alpha_diversity")
fig.savefig(f"{base}.tif", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(f"{base}.jpg", dpi=300, pil_kwargs={"quality": 95})
plt.close(fig)

print(f"\nWritten: {base}.tif, {base}.jpg")
print("         alpha_diversity.csv, alpha_diversity_tests.csv")
print("\nEach panel carries its Mann-Whitney p value. Top row (a-c) is the")
print("whole community, bottom row (d-f) the same after removing rhizobia and")
print("reclosing the table; columns are q0, q1 and q2.")
print("P is preserved and D is degraded; medians and U statistics are in the")
print("tests file.")
