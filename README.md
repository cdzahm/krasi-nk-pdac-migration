# krasi-nk-pdac-migration

Code for the 3D NK-cell migration and morphology analysis in

> Dodhiawala PB, Zahm CD, et al. Inhibition of the KRAS oncoprotein sensitizes
> pancreatic cancer to NK cell immunotherapy. *Cancer Cell* (2026). [DOI]

It takes time-lapse multiphoton stacks of NK cells migrating in collagen
with pancreatic tumor cells, and produces the per-cell measures, statistics
and panels in Figures 3, 4, S3 and S5.

## Panels

| Figure | Measure | Cells | Panel name |
| --- | --- | --- | --- |
| 3B | Motility coefficient, speed | MIA-PaCa-2, IL-15 vs KRASi | `Fig3B_motility`, `Fig3B_speed` |
| 3D (right) | Motility coefficient | HPAF-II, IL-15 vs KRASi | `Fig3D_motility` |
| 3E | Arrest coefficient, elongated-fast and round-slow occupancy, cell area | MIA-PaCa-2, IL-15 vs KRASi | `Fig3E_*` |
| 3F | Arrest coefficient | HPAF-II, IL-15 vs KRASi | `Fig3F_arrest` |
| 4G (left) | Motility coefficient | MIA-PaCa-2, IL-15 vs TriKE | `Fig4G_motility` |
| 4H | Motility coefficient | HPAF-II, IL-15 / TriKE / TriKE+KRASi | `Fig4H_motility` |
| 4J | Elongated-fast occupancy | HPAF-II, IL-15 / TriKE / TriKE+KRASi | `Fig4J_elongated_fast` |
| S3B | Motility coefficient, speed | IL-15 titration (0, 1, 10 ng) | `FigS3B_*` |
| S3C | Arrest, elongated-fast, round-slow, cell area | IL-15 titration | `FigS3C_*` |
| S3D | Motility coefficient | MIA-PaCa-2, no IL-15 vs IL-15 | `FigS3D_motility` |
| S3E | Arrest, elongated-fast, round-slow, cell area | MIA-PaCa-2, no IL-15 vs IL-15 | `FigS3E_*` |
| S3F | Elongated-fast and round-slow occupancy | HPAF-II, IL-15 vs KRASi | `FigS3F_*` |
| S5G | Motility coefficient | HPAC, IL-15 / TriKE / TriKE+KRASi | `FigS5G_motility` |

The panel definitions are in [`src/nkmigration/panels.py`](src/nkmigration/panels.py).
Track overlays, time-lapse frames and the other panels in these figures come
from separate analyses and are not part of this repository.

## Installation

Python 3.11. With conda:

```bash
conda env create -f environment.yml
conda activate nkmigration
```

or with pip, into an existing Python 3.11 environment:

```bash
pip install -e ".[segmentation,test]"
```

Versions are pinned to the ones used for the paper. Segmentation (step 2)
needs Cellpose and PyTorch, and a GPU is strongly recommended; every other
step runs on a laptop.

## Data

Each recording is one folder of calibrated TZYX stacks, one per channel, as
written by [PPP Stack Organizer](https://github.com/cdzahm/PPP_Stack_Organizer)
from Bruker PrairieView time series:

```
<data root>/
    MIAPaCa2_A_IL15/
        nk.tif
        tumor.tif
        collagen.tif
    ...
```

Voxel size and frame interval are read from each stack's own metadata. For
stacks without calibration, `scripts/00_add_calibration.py` writes it from
the PrairieView acquisition XML.

[Data availability statement to be added.]

## The recordings

[`config/recordings.csv`](config/recordings.csv) lists the 27 recordings and
every per-recording setting. Donors are labeled A, B, C within each cell
line.

| Column | Meaning |
| --- | --- |
| `experiment` | One donor with one tumor line, or the IL-15 titration. State cutoffs are set per experiment. |
| `stimulus`, `il15_ng`, `kras_inhibitor` | Treatment arm. |
| `drop_first_frames` | Frames removed from the start before analysis. |
| `segment_first_z_planes` | Segment only the bottom planes. Used for the four MIA-PaCa-2 donor B recordings: in one of them the upper planes are nearly identical to one another, consistent with the Z drive sticking, so all four were cropped to the same depth. |
| `drift_method` | `reference_frame`, `windowed` (+/-3 frames), `windowed_z` (Z only, +/-3 frames) or `none`. |
| `drift_channel` | Registration input: `collagen`, `collagen_enhanced` (background-subtracted, CLAHE) or `tumor_minus_nk` (tumor channel with NK signal masked). |

Some recordings appear in no panel, such as MIA-PaCa-2 donor A with no IL-15
plus KRASi. They are included because state cutoffs are medians over every
recording in an experiment.

## Running the analysis

Set the two roots once:

```bash
export NKMIG_DATA=/path/to/data
export NKMIG_OUT=/path/to/outputs
```

Then run the steps in order. Steps 1 to 4 take recording IDs, and run on
every recording when given none.

| Step | Script | Writes |
| --- | --- | --- |
| 1 | `scripts/01_correct_drift.py` | `drift/<id>/shifts.csv`, drift-corrected `nk.tif` and `tumor.tif` |
| 2 | `scripts/02_segment.py` | `segmentation/<id>/objects.csv` and one label stack per Z slab |
| 3 | `scripts/03_detect.py` | `detections/<id>.csv`: one row per cell per frame, with its Z position |
| 4 | `scripts/04_track.py` | `tracks/<id>.csv` |
| 5 | `scripts/05_measure.py` | `measures/per_cell.csv`, `prwm_fits.csv`, `state_cutoffs.csv` |
| 6 | `scripts/06_make_panels.py` | `panels/<panel>.pdf` and `panel_stats.csv` |
| 7 | `scripts/07_screen_measures.py` | `screens/`: the measure-selection screens |

For example, to redo one recording up to tracking and then rebuild everything
downstream:

```bash
python scripts/01_correct_drift.py HPAC_A_IL15
python scripts/02_segment.py HPAC_A_IL15
python scripts/03_detect.py HPAC_A_IL15
python scripts/04_track.py HPAC_A_IL15
python scripts/05_measure.py
python scripts/06_make_panels.py
```

`measures/per_cell.csv` has one row per tracked cell. A measure is filled in
only when the cell meets that measure's inclusion rule (see below), so
`per_cell.csv` on its own is enough to redraw every panel and rerun every
test.

## Methods in brief

**Drift.** A rigid translation per frame, from phase cross-correlation
(10x subpixel) on isotropically resampled volumes. Shifts are applied to the
NK and tumor channels by linear interpolation.

**Segmentation.** Cellpose-SAM (built-in `cpsam_v2` weights, flow threshold
0.4, minimum size 5 px) on maximum-intensity projections of overlapping
7-plane Z slabs at a 4-plane stride. Objects in adjacent slabs are chained
when their masks overlap with IoU >= 0.3 (one-to-one, highest IoU first), and
the object with the largest area x mean intensity is kept. Each cell's Z
position is the intensity-weighted centroid of the NK signal under its 2D
mask, through the full stack depth. Cells touching the XY border are tracked
but left out of shape measures.

**Tracking.** LapTrack linear assignment on (x, y, z), at most 15 um per
frame, with no gap closing, splitting or merging.

**Per-cell measures.** One value per cell. Motion measures use XY steps
between consecutive frames and need at least 3 frames; shape measures need at
least 3 frames off the border.

- Speed: median step length per minute.
- Arrest coefficient: fraction of steps shorter than one pixel.
- Cell area: median 2D mask area.
- Motility coefficient: a persistent random walk model fit per axis
  ([`prwm.py`](src/nkmigration/prwm.py)), mu = S^2 P summed over x, y and z,
  with P bounded by the track's duration. Lags need at least 4 displacement
  pairs and fits at least 4 lags, so fitted tracks have 7 or more frames.
- Morphodynamic states: each cell-timepoint is round or elongated and slow or
  fast, splitting aspect ratio and step length at their medians within an
  experiment. Occupancy is the fraction of a cell's classified frames in a
  state, for cells with at least 10 classified frames.

**Statistics.** Cells are pooled by cell line across the donors that have
every arm in a comparison. Two groups: two-sided Mann-Whitney U. Three
groups: Kruskal-Wallis, then Dunn's test with Holm correction, with pairs
shown as not significant unless the Kruskal-Wallis p < 0.05. Motility
coefficients above 300 um^2/min are drawn at 300 and kept in every
statistic.

## How the measures were chosen

The pipeline computes 36 candidate per-cell measures of NK-cell shape and
motion ([`percell.py`](src/nkmigration/percell.py)). Step 7 runs the two
screens that guided which of them are reported:

1. **Redundancy.** For every donor and comparison (KRASi, TriKE or
   TriKE+KRASi, each against IL-15), each measure's effect is Cliff's delta.
   Measures whose effects correlate at |Spearman rho| >= 0.8 across these
   units are grouped by average-linkage clustering
   (`screens/redundancy_groups.csv`).
2. **Donor consistency.** For each measure and comparison, whether every
   donor's effect has the same sign, and whether every donor's 95% bootstrap
   interval excludes zero (`screens/donor_consistency.csv`).

From these we chose interpretable measures, guided by which changed in the
same direction across donors. The screens informed that choice; they did not
make it.

## Tests

```bash
pytest
```

The tests use synthetic data: known shifts recovered by each drift method, a
persistent random walk recovered by the PRWM fit, slab merging, the state
definitions and the inclusion rules.

## Citation

If you use this code, please cite the paper above and this repository
([Zenodo DOI]). See [`CITATION.cff`](CITATION.cff).

## License

Copyright (C) 2026 Regents of the University of Minnesota.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. It is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See [`LICENSE`](LICENSE) for details.

## Use of AI tools

The analysis code was developed with the help of AI coding assistants
(Claude Code and Claude, Anthropic; Gemini, Google). The authors directed the
analysis, reviewed the code, and checked intermediate results before use.
