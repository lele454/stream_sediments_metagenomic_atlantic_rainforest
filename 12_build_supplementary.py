#!/usr/bin/env python3
"""
Assemble the supplementary tables into one workbook per table
---------------------------------------------------------------
Twenty loose CSVs become six or seven files, each with the parts of one
supplementary table on its own sheet. A reader opening Table S10 sees six
tabs rather than hunting for six downloads, and the numbering stops drifting
between the manuscript and the folder.

Files are grouped by the number in their name, in any of the spellings this
project has used - TableS10_x.csv, Table_S10_x.csv, Table_10_x.csv - and the
rest of the name becomes the sheet name. Sheet order follows SHEET_ORDER
where a table is listed there and is alphabetical otherwise, so the first tab
a reader lands on is the one that explains the rest.

Table S11 is the per-feature test table and runs to tens of thousands of
rows, which no supplement should carry whole. It is filtered to the features
reaching --p-cutoff in at least one of its levels, and the sheet records how
many rows were dropped and why.

Usage
  python3 build_supplementary.py --indir saida --outdir Tabelas_suplementares
  python3 build_supplementary.py --indir saida --outdir T --p-cutoff 0.01
"""

import os
import re
import sys
import glob
import argparse
import pandas as pd

# The first sheet of each workbook should be the one that frames the others.
SHEET_ORDER = {
    "S2": ["LandUseCover1", "land_use_index", "land_use_index_loadings"],
    "S7": ["permanova_results", "permanova_permdisp", "nmds_scores"],
    "S4": ["marker_genes", "marker_genes_raw", "marker_genes_tests"],
    "S5": ["Resistome_functional_RPKM_by_sample", "resistome_dispersion"],
    "S8": ["alpha_diversity", "alpha_diversity_tests", "hill_numbers"],
    "S9": ["guilds_redox", "guilds_redox_tests", "guilds_gradient",
           "guilds_vs_landscape_all", "guild_logratios", "archaeal_depth"],
    "S10": ["dbrda_results", "mantel_results", "spatial_confounding",
            "landscape_collinearity"],
}

TITLES = {
    "S1": "Physicochemical and limnological variables",
    "S2": "Land use variables and the riparian land use intensity index",
    "S3": "Definition of the metabolic guilds",
    "S4": "Marker genes for each guild",
    "S5": "Resistome structure and its ordination",
    "S6": "Relative abundance of genera",
    "S7": "Ordinations: PERMANOVA, PERMDISP and sample scores",
    "S8": "Alpha diversity, with and without rhizobia",
    "S9": "Metabolic guilds along the gradient",
    "S10": "Landscape, space and their confounding",
    "S11": "Per-feature tests on centred log-ratios",
}

PATTERN = re.compile(r"^Table_?S?_?(\d+)_(.+)\.csv$", re.IGNORECASE)

ap = argparse.ArgumentParser()
ap.add_argument("--indir", required=True)
ap.add_argument("--outdir", required=True)
ap.add_argument("--p-cutoff", type=float, default=0.05,
                help="rows kept in Table S12 (default 0.05)")
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)

groups = {}
unmatched = []
for path in sorted(glob.glob(os.path.join(args.indir, "*.csv"))):
    m = PATTERN.match(os.path.basename(path))
    if not m:
        unmatched.append(os.path.basename(path))
        continue
    groups.setdefault(f"S{int(m.group(1))}", []).append((m.group(2), path))

if unmatched:
    print("not named Table[_]S<n>_<name>.csv, skipped:")
    for u in unmatched:
        print(f"  {u}")
    print()

if not groups:
    sys.exit(f"nothing to assemble in {args.indir}")


def sheet_name(raw):
    """Excel allows 31 characters and no []:*?/\\ in a sheet name."""
    s = re.sub(r"[\[\]:*?/\\]", "-", raw)
    return s[:31]


def order_of(key, parts):
    wanted = SHEET_ORDER.get(key)
    if not wanted:
        return sorted(parts)
    rank = {w.lower(): i for i, w in enumerate(wanted)}
    return sorted(parts, key=lambda p: (rank.get(p[0].lower(), len(rank)), p[0]))


def trim_feature_tests(d, cutoff):
    """Keep every row of a feature that reaches the cutoff anywhere."""
    if "p" not in d.columns or "feature" not in d.columns:
        return d, 0
    keep = set(d.loc[d.p < cutoff, "feature"])
    out = d[d.feature.isin(keep)]
    return out, len(d) - len(out)

written = []
for key in sorted(groups, key=lambda k: int(k[1:])):
    parts = order_of(key, groups[key])
    out = os.path.join(args.outdir, f"Table_{key}.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        for raw, path in parts:
            d = pd.read_csv(path)
            note = ""
            if key == "S11":
                d, dropped = trim_feature_tests(d, args.p_cutoff)
                if dropped:
                    note = (f"  [{dropped} rows dropped: features with no "
                            f"p < {args.p_cutoff} at any level]")
            d.to_excel(xl, sheet_name=sheet_name(raw), index=False)
            print(f"  {key}  {sheet_name(raw):34s} {len(d):>7} rows{note}")
    size = os.path.getsize(out) / 1024
    written.append((key, out, len(parts), size))
    print(f"-> {os.path.basename(out)}  ({len(parts)} sheets, {size:.0f} KB)\n")

print("=" * 64)
for key, out, n, size in written:
    print(f"Table {key:4s} {TITLES.get(key, ''):52s} {n} sheet(s)")

absent = [k for k in TITLES if k not in groups]
if absent:
    print("\nnot assembled, no file found:")
    for k in sorted(absent, key=lambda x: int(x[1:])):
        print(f"  Table {k:4s} {TITLES[k]}")
