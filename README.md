# Land-use intensity and the sediment microbiome of Atlantic Rainforest streams

Analysis code for the manuscript. Fourteen sediment metagenomes, seven from
streams inside the Juréia-Itatins Ecological Station and seven from
peri-urban and agricultural catchments in the Baixada Santista and Vale do
Ribeira, São Paulo, Brazil.

## Running it

```bash
pip install -r requirements.txt
bash run_all.sh BRACKEN_DIR DATA_DIR
```

Both arguments are optional and default to `data/`. Figures are written to
`results/figures` as TIFF at 300 dpi with LZW compression, and as JPEG to
check on screen; tables go to `results/tables`. Seeds are fixed, so a rerun
reproduces the published numbers.

## Scripts

| # | script | produces |
|---|--------|----------|
| 01 | `01_genus_barplot.py` | Figure 1, genus composition |
| 02 | `02_alpha_diversity.py` | Figure 2, Hill numbers with and without rhizobia |
| 03 | `03_nmds_permanova.py` | Figure 3, ordinations on Bray-Curtis and Aitchison |
| 04 | `04_guilds_redox.py` | guild abundances and group tests |
| 05 | `05_marker_genes.py` | Figure S3, marker gene heatmap |
| 06 | `06_dbrda_mantel.py` | Figure 5, constrained ordination and Mantel tests |
| 07 | `07_spatial_confounding.py` | Figure S4, distance decay and partial Mantel |
| 08 | `08_guilds_gradient.py` | Figure S2, guilds against the land use index |
| 09 | `09_fig4_guilds.py` | Figure 4, drawn in one pass from 04 and 08 |
| 10 | `10_compositional_analysis.py` | Tables S7, S8 and S11 |
| 11 | `11_resistome_dispersion.py` | Table S5, resistome PERMANOVA and PERMDISP |
| 12 | `12_build_supplementary.py` | assembles the loose CSVs into one workbook per table |

Scripts 09 to 12 read what 04 and 08 write, so `run_all.sh` keeps the order.

## What the analysis decides, and why

**The tables are compositions, and the analysis says so.** Every sample sums
to a fixed total, so a group that grows does so at the expense of the others
and an index of evenness records that redistribution whether or not anything
else changed. Two controls run throughout. Every diversity index and group
comparison is repeated with the rhizobia removed and the table reclosed, and
every ordination is repeated on Aitchison distance, the Euclidean distance
between centred log-ratios, which the total cannot move. Where the two
treatments disagree, both are reported: the difference between them is the
part of the Bray-Curtis result that the closure contributes.

**PERMANOVA is never reported without PERMDISP.** A significant PERMANOVA
with unequal dispersions reflects spread as much as location, and the test
itself does not tell them apart.

**Proportions are within domain.** A bacterial guild is a percentage of the
bacterial community and an archaeal guild a percentage of the archaeal
community. Archaea are under 1% of the reads here; plotting both against the
whole community would compress every archaeal value to the axis and invite a
comparison of magnitudes the sequencing depth does not support. The archaeal
panels rest on 116 to 709 archaeal reads per sample and the script prints
that count rather than burying it.

**Limitations are in the code, not only in the paper.** The limnological
variables were measured on a disturbed water-sediment mixture and cannot
represent in-situ redox conditions. Taxonomic abundance comes from
k-mer classification and is not an organism count, so richness is computed
but not interpreted. Neither is worked around; both are stated where the
analysis touches them.

## Data

| file | contents |
|------|----------|
| Bracken reports | genus abundances per sample, one file per stream |
| `Table_S4_KO_abundance.xlsx` | KEGG Ortholog abundances per 100,000 annotated ORFs |
| `TableS3_guild_assignments.csv` | which genera make up each guild |
| `TableS6_genus_relative_abundance.csv` | the genus matrix as percentages |
| `Resistome_functional_RPKM_by_sample.csv` | ARG abundance by functional category |
| `LandUseCover1.csv` | MapBiomas cover classes, elevation, HFP and coordinates |

## Data availability

Raw sequence data: [BioProject accession].

## Licence

[To be chosen before the repository is made public.]
