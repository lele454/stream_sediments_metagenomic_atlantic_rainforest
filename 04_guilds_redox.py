#!/usr/bin/env python3
"""
Figure: metabolic guilds along the redox sequence, bacteria and archaea
-------------------------------------------------------------------------
Panel (a) places every guild in the order its electron acceptor is consumed as
a sediment turns anoxic, from methane oxidation through nitrate, Fe(III) and
sulfate to carbon dioxide, and plots the change between preserved and degraded
streams. The remaining panels show the guilds that survive correction.

The sequence is the argument. Two guilds rising in the degraded streams is two
results; several rising in the order in which a sediment loses its electron
acceptors is one, and methanogenesis is the last step of that sequence and is
archaeal. A figure restricted to bacteria stops in the middle of the gradient.

Proportions are within each domain: a bacterial guild as a percentage of the
bacterial community, an archaeal guild as a percentage of the archaeal
community. Archaea are a fraction of a percent of the reads here, so plotting
both against the whole community would compress every archaeal bar to the
axis and invite a comparison of magnitudes that the depths do not support.
Within-domain proportions ask the same question of both: how much of this
domain does this metabolism account for.

Two archaeal groups are excluded before the proportions are computed.
Halophiles and thermophiles have no plausible population in a freshwater
tropical stream and are read as misclassification against a database whose
archaeal representation is dominated by cultured extremophiles; they
nonetheless took up a large and variable share of the archaeal reads, which
distorted every other proportion. The exclusion is stated rather than silent,
and the script reports what it removed.

Inputs:
  --bracken   directory of Bracken genus reports, one per sample
  --outdir    where to write (default: current directory)

Outputs:
  Fig_guilds_redox.tif / .jpg
  guilds_redox.csv              within-domain proportions
  guilds_redox_tests.csv
  archaeal_depth.csv            archaeal reads per sample, and what was dropped

Usage:
  python3 fig_guilds_redox.py --bracken .
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
from matplotlib.lines import Line2D

SAMPLES = [f"Pres0{i}" for i in range(1, 8)] + [f"Deg0{i}" for i in range(1, 8)]

COL_PRES = "#2c7bb6"
COL_DEG = "#d7191c"
COL_NS = "#bdbdbd"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "savefig.bbox": "tight",
})

# Guilds in the order their electron acceptor is consumed. Bacterial and
# archaeal guilds are interleaved by that order, not grouped by domain: the
# sequence is what the panel shows, and separating the domains would hide it.
# Each entry is (guild, domain).
GUILD_ORDER = [
    ("Methanotroph", "B"),
    ("AOA (ammonia oxidiser)", "A"),
    ("AOB (ammonia oxidiser)", "B"),
    ("NOB (nitrite oxidiser)", "B"),
    ("Denitrifiers", "B"),
    ("Fe(III) reducer", "B"),
    ("Sulfate reducer", "B"),
    ("Sulfur oxidiser", "B"),
    ("Methanogen (H2/CO2)", "A"),
    ("Methanogen (acetate)", "A"),
    ("Methanogen (methyl)", "A"),
    ("ANME (anaerobic methane ox.)", "A"),
]

DOMAIN_NAME = {"B": "Bacteria", "A": "Archaea"}

# archaeal groups excluded before computing within-domain proportions
ARCHAEA_EXCLUDED = ["Halophile", "Thermophile"]

GENUS_GUILD = {}


def assign(names, guild):
    for n in names.split():
        GENUS_GUILD[n] = guild


# ---- bacteria --------------------------------------------------------
assign("""Methylobacter Methylomicrobium Methylovulum Methylotuvimicrobium
Methylococcus Methylomonas Methylosarcina Methylocaldum Methylosphaera
Methylocystis Methylosinus Methylocella Methylocapsa Methyloferula
Crenothrix Methylacidiphilum""", "Methanotroph")

assign("Nitrosomonas Nitrosospira Nitrosococcus Nitrosovibrio Nitrosolobus",
       "AOB (ammonia oxidiser)")
assign("Nitrobacter Nitrospira Nitrospina Nitrococcus Nitrotoga Nitrolancea",
       "NOB (nitrite oxidiser)")

# Named for the metabolism, not for a taxon: the guild is defined by the
# electron acceptor, and a name that mixes the two invites the question of
# why one order appears in it. Rhodocyclales genera are here because
# denitrification is conserved across the order, which Table S3 records.
assign("""Dechloromonas Thauera Azoarcus Azospira Zoogloea Quatrionicoccus
Propionivibrio Denitratisoma Sulfuritalea Georgfuchsia Rhodocyclus Azonexus
Uliginosibacterium Aromatoleum Oryzomicrobium Ideonella Aquincola
Rhizobacter""", "Denitrifiers")

assign("""Geobacter Anaeromyxobacter Desulfuromonas Pelobacter Geothrix
Rhodoferax Albidiferax Shewanella Ferribacterium""", "Fe(III) reducer")

assign("""Desulfovibrio Desulfobacter Desulfobulbus Desulfococcus
Desulfomicrobium Desulfotomaculum Desulfobacterium Desulfatibacillum
Desulfosporosinus Desulfomonile Syntrophobacter Desulfarculus Desulfonema
Desulfobacca Desulfatiglans""", "Sulfate reducer")

assign("""Thiobacillus Sulfurimonas Sulfuricurvum Thiothrix Beggiatoa
Thiomonas Sulfurovum Thioalkalivibrio Halothiobacillus Acidithiobacillus
Sulfuritortus Thiovirga""", "Sulfur oxidiser")

# ---- archaea ---------------------------------------------------------
ARCHAEAL_GUILD = {}


def assign_a(names, guild):
    for n in names.split(";"):
        n = n.strip()
        if n:
            ARCHAEAL_GUILD[n] = guild


assign_a("""Nitrosopumilus; Candidatus Nitrosopumilus; Nitrososphaera;
Candidatus Nitrososphaera; Candidatus Nitrosotenuis;
Candidatus Nitrosoarchaeum; Nitrosotalea; Candidatus Nitrosotalea;
Candidatus Nitrosocosmicus; Cenarchaeum""", "AOA (ammonia oxidiser)")

assign_a("""Methanobacterium; Methanobrevibacter; Methanosphaera;
Methanothermobacter; Methanothermus; Methanococcus; Methanothermococcus;
Methanocaldococcus; Methanotorris; Methanomicrobium; Methanogenium;
Methanoculleus; Methanolacinia; Methanoplanus; Methanofollis;
Methanocorpusculum; Methanospirillum; Methanolinea; Methanoregula;
Candidatus Methanoregula; Methanosphaerula; Methanocalculus; Methanocella;
Candidatus Methanoflorens; Methanosarcina""", "Methanogen (H2/CO2)")

assign_a("Methanosaeta; Methanothrix", "Methanogen (acetate)")

assign_a("""Methanolobus; Methanococcoides; Methanohalophilus; Methanohalobium;
Methanomethylovorans; Methanosalsum; Methermicoccus; Methanomassiliicoccus;
Candidatus Methanomassiliicoccales;
Candidatus Methanomethylophilus""", "Methanogen (methyl)")

assign_a("Candidatus Methanoperedens", "ANME (anaerobic methane ox.)")

assign_a("""Halobacterium; Haloarcula; Haloferax; Halorubrum; Halomicrobium;
Haloquadratum; Halogeometricum; Natrialba; Natronomonas; Natronococcus;
Haloterrigena; Halopiger; Natrinema; Halalkalicoccus; Halorhabdus;
Haladaptatus; Halosimplex; Halovivax; Salinarchaeum; Halobellus; Halolamina;
Halogranum; Halopenitus; Halovenus""", "Halophile")

assign_a("""Thermococcus; Pyrococcus; Palaeococcus; Archaeoglobus; Ferroglobus;
Geoglobus; Sulfolobus; Saccharolobus; Metallosphaera; Acidianus;
Sulfurisphaera; Stygiolobus; Thermoproteus; Pyrobaculum; Caldivirga;
Vulcanisaeta; Thermofilum; Desulfurococcus; Staphylothermus; Aeropyrum;
Ignicoccus; Hyperthermus; Pyrolobus; Thermoplasma; Ferroplasma; Picrophilus;
Cuniculiplasma""", "Thermophile")


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


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = np.empty_like(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q[order] = np.minimum.accumulate(ranked[::-1])[::-1]
    return np.clip(q, 0, 1)


# ---------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--bracken", required=True)
ap.add_argument("--outdir", default=".")
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)

m = read_bracken(args.bracken)
samples = list(m.columns)
pres = [c for c in samples if c.startswith("Pres")]
degr = [c for c in samples if c.startswith("Deg")]
order_s = pres + degr

archaeal_genera = [g for g in m.index if g in ARCHAEAL_GUILD]
arch = m.loc[archaeal_genera]

dropped = arch.loc[[g for g in arch.index
                    if ARCHAEAL_GUILD[g] in ARCHAEA_EXCLUDED]]
arch_kept = arch.drop(index=dropped.index)

depth = pd.DataFrame({
    "archaeal_reads_total": arch.sum(),
    "excluded_reads": dropped.sum(),
    "retained_reads": arch_kept.sum()})
depth["percent_excluded"] = (100 * depth.excluded_reads
                             / depth.archaeal_reads_total).round(1)
depth["percent_of_all_reads"] = (100 * depth.archaeal_reads_total
                                 / m.sum()).round(3)
depth = depth.loc[order_s]
depth.to_csv(os.path.join(args.outdir, "archaeal_depth.csv"))

print(f"{m.shape[0]} genera, {len(samples)} samples")
print(f"\narchaeal reads: {int(depth.archaeal_reads_total.min())} to "
      f"{int(depth.archaeal_reads_total.max())} per sample "
      f"({depth.percent_of_all_reads.min():.2f}% to "
      f"{depth.percent_of_all_reads.max():.2f}% of classified reads)")
print(f"excluded as halophile or thermophile: "
      f"{depth.percent_excluded.min():.0f}% to "
      f"{depth.percent_excluded.max():.0f}% of archaeal reads\n")
if depth.retained_reads.min() < 100:
    print(f"[note] {int(depth.retained_reads.min())} archaeal reads in the")
    print("       shallowest sample after exclusion; the archaeal panels rest")
    print("       on thin data and should be read as such.\n")

# within-domain proportions
bact = m.drop(index=archaeal_genera)
rows = {}
for guild, domain in GUILD_ORDER:
    if domain == "B":
        members = [g for g in bact.index if GENUS_GUILD.get(g) == guild]
        rows[guild] = 100 * bact.loc[members].sum() / bact.sum()
    else:
        members = [g for g in arch_kept.index
                   if ARCHAEAL_GUILD.get(g) == guild]
        rows[guild] = 100 * arch_kept.loc[members].sum() / arch_kept.sum()

g = pd.DataFrame(rows).T[order_s].fillna(0)
g.round(4).to_csv(os.path.join(args.outdir, "guilds_redox.csv"))

recs = []
for guild, domain in GUILD_ORDER:
    a, b = g.loc[guild, pres], g.loc[guild, degr]
    if a.sum() + b.sum() == 0:
        continue
    u, p = mannwhitneyu(a, b)
    # mean with SD for the text, median for the figure: the test is on ranks,
    # so the median is what it compares, but a reader reporting a percentage
    # wants the spread as well
    recs.append(dict(guild=guild, domain=DOMAIN_NAME[domain],
                     preserved_mean=a.mean(), preserved_sd=a.std(ddof=1),
                     degraded_mean=b.mean(), degraded_sd=b.std(ddof=1),
                     preserved_median=a.median(), degraded_median=b.median(),
                     preserved_q1=a.quantile(.25), preserved_q3=a.quantile(.75),
                     degraded_q1=b.quantile(.25), degraded_q3=b.quantile(.75),
                     log2FC=float(np.log2((b.mean() + 1e-9) / (a.mean() + 1e-9))),
                     U=float(u), p=p))
t = pd.DataFrame(recs)
t["q"] = bh(t.p)
t = t.set_index("guild")
t = t.reindex([gd for gd, _ in GUILD_ORDER if gd in t.index])
t.round(4).to_csv(os.path.join(args.outdir, "guilds_redox_tests.csv"))

print(t.round(4).to_string())
print(f"\n{(t.q < 0.05).sum()} guilds with q < 0.05")

# Which summary to quote depends on the shape of the distribution. Where the
# standard deviation exceeds the mean, the mean is being pulled by one or two
# samples and quoting it with an SD reads as a symmetric spread that is not
# there; the median with its interquartile range is both honest and closer to
# what the rank test actually compares.
print("\n--- as they would be written ---")
for guild, r in t.iterrows():
    skewed = (r.preserved_sd > r.preserved_mean
              or r.degraded_sd > r.degraded_mean)
    if skewed:
        summary = (f"{r.preserved_median:6.2f} [{r.preserved_q1:.2f}"
                   f"\u2013{r.preserved_q3:.2f}]  \u2192  "
                   f"{r.degraded_median:6.2f} [{r.degraded_q1:.2f}"
                   f"\u2013{r.degraded_q3:.2f}]   median [IQR]")
    else:
        summary = (f"{r.preserved_mean:6.2f} \u00b1 {r.preserved_sd:5.2f}"
                   f"  \u2192  {r.degraded_mean:6.2f} \u00b1 "
                   f"{r.degraded_sd:5.2f}   mean \u00b1 SD")
    flag = " *" if r.q < 0.05 else "  "
    print(f" {flag} {guild:30s} {summary}, q = {r.q:.3f}")

print("\nWhere SD exceeds the mean the median and IQR are given instead: the")
print("mean is being carried by one or two samples and an SD would imply a")
print("symmetric spread that the data do not have.")

# ---------------------------------------------------------------------
sig = t[t.q < 0.05]
n_box = max(1, len(sig))

# the boxplot strip widens with the number of significant guilds so the
# individual panels keep a sensible aspect ratio instead of being squeezed
fig = plt.figure(figsize=(7.0 + 1.35 * n_box, 4.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.42 * n_box], wspace=0.30)
axA = fig.add_subplot(gs[0, 0])
gsB = gs[0, 1].subgridspec(1, n_box, wspace=0.62)

plotted = list(t.index)[::-1]
domain_of = dict(GUILD_ORDER)

for i, guild in enumerate(plotted):
    r = t.loc[guild]
    colour = (COL_DEG if r.log2FC > 0 else COL_PRES) if r.q < 0.05 else COL_NS
    marker = "o" if domain_of[guild] == "B" else "s"
    axA.plot([0, r.log2FC], [i, i], color=colour, linewidth=1.4, zorder=2)
    axA.scatter(r.log2FC, i, s=62, c=colour, marker=marker,
                edgecolors="white", linewidths=0.8, zorder=3)
    if r.q < 0.05:
        axA.text(0.16, i + 0.28, f"$q$ = {r.q:.3f}", fontsize=6.8,
                 va="center", ha="left", color="#404040")

axA.text(-0.60, 1.03, "(a)", transform=axA.transAxes, fontsize=11,
         fontweight="bold", va="bottom", ha="left")
axA.axvline(0, color="#1a1a1a", linewidth=0.8, zorder=1)
axA.set_yticks(np.arange(len(plotted)))
axA.set_yticklabels(plotted, fontsize=7.6)
axA.set_xlabel("log$_2$ fold change, degraded / preserved")
lo, hi = axA.get_xlim()
axA.set_xlim(lo - 0.25, hi + 0.60)
axA.set_ylim(-0.8, len(plotted) - 0.2)
for side in ("top", "right", "left"):
    axA.spines[side].set_visible(False)
axA.tick_params(axis="y", length=0)

axA.legend(handles=[
    Line2D([], [], marker="o", linestyle="", color="#595959",
           markersize=6, label="Bacteria"),
    Line2D([], [], marker="s", linestyle="", color="#595959",
           markersize=6, label="Archaea")],
    loc="lower right", frameon=False, fontsize=7.4, handletextpad=0.4)

rng = np.random.default_rng(42)
LETTERS = "bcdefghij"
for j, guild in enumerate(sig.index):
    ax = fig.add_subplot(gsB[0, j])
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
    ax.text(-0.34, 1.10, f"({LETTERS[j]})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["P", "D"], fontsize=8)
    # guild names are long: break on the separators rather than let
    # neighbouring titles overlap
    ax.set_title(guild.replace(" / ", "\n").replace(" (", "\n("),
                 fontsize=7.2, pad=8, linespacing=1.3)
    if j == 0:
        ax.set_ylabel("% within domain", fontsize=8)
    ax.tick_params(labelsize=7.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

base = os.path.join(args.outdir, "Fig_guilds_redox")
fig.savefig(f"{base}.tif", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(f"{base}.jpg", dpi=300, pil_kwargs={"quality": 95})
plt.close(fig)

letters = ", ".join(f"({LETTERS[j]}) {gd}" for j, gd in enumerate(sig.index))
print(f"\nWritten: {base}.tif, {base}.jpg")
print("         guilds_redox.csv, guilds_redox_tests.csv, archaeal_depth.csv")
print("\n(a) guilds in the order their electron acceptor is consumed; circles")
print("are bacterial, squares archaeal, grey is q >= 0.05.")
print(f"Then {letters}; P preserved, D degraded.")
print("\nProportions are within domain. Archaeal and bacterial values are not")
print("comparable in magnitude and must not be added.")
