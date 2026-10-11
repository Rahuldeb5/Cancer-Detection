# Session S10: pancreatic-duct atlas (2026-10-03)

**Answer.** The stacked `pancreatic_duct` masks of 6061 PanTS glands form one continuous, curved ridge that follows
the gland's bend. It sits ~3.5–5.5 mm anterior to the tissue centreline through neck/body/tail. It is densest in the
head/neck and fades toward the tail: 56% of cases have duct at u = 0.2, 8% at u = 0.8. In the axial view the head end
splits into two lobes (posterior near u 0.1, ap −0.1, plus the main ridge rising anteriorly). Annotation is
fragmentary (median p5–p95 extent 27.6 mm), so the darkness of every panel is coverage × anatomy.

Deliverables: `results/duct_atlas/{duct_atlas.png, duct_atlas_mm.png, README.md}`, plus the optional
`duct_atlas_by_tumor.png`. Every number in the README is printed in `work/duct_atlas/full/report*.txt`.

## What was built (`src/duct_atlas/`, importable; no existing script edited)

| file | role |
|---|---|
| `frame.py` | `build_frame`: envelope centroid, PC1 signed head→tail, world-anterior / world-superior axes made orthogonal to PC1; normalized and mm coordinates |
| `grid.py` | the two atlas grids; `splat` deposits voxel volume into the bin of each voxel centre and returns the off-grid mass |
| `case.py` | loads the 4 masks via `tumorlib.io.load_mask` as voxel-coordinate lists (one full volume in memory at a time), builds a bit-flag crop for the unions, applies the exclusions and splats. The head affine is used for all four masks |
| `phantom.py` | random-pose ellipsoid glands with head/body/tail thirds and an axial duct |
| `run.py` | chunked (50 cases), resumable, 4-worker run. Sparse per-chunk checkpoints in `work/duct_atlas/chunks/` (1.2 GB), merge into `work/duct_atlas/<tag>/`. Options: `--limit`, `--cases`, `--ids-file` |
| `plot.py` | the figures, QC image (`work/duct_atlas/full/qc_cases.png`), the report numbers, and the `--split` tumor figure |
| `flip_audit.py` | aorta-posterior test for y-mirrored gland affines |
| `tests/test_duct_atlas.py` | 10 tests (3 oracles + frame sign + exclusions + union + 1 real-data load), passing before any real run |

Reproduce:
```
PYTHONPATH=src .venv/bin/python -m duct_atlas.run                 # ~23 min
PYTHONPATH=src .venv/bin/python -m duct_atlas.plot                # figures + report
PYTHONPATH=src .venv/bin/python -m duct_atlas.run --ids-file work/duct_atlas/tumor_pos_ids.txt --tag tumorpos
PYTHONPATH=src .venv/bin/python -m duct_atlas.plot --split tumorpos
```

PC only, CPU only, 4 workers. Full run 22.9 min; worker peak RSS ≤ 602 MB. Worst case PanTS_00008854: 3.8 s, 460 MB.
The server was not touched. No masks were pulled: all 9901 head/body/tail/duct masks are real files. `pancreas.nii.gz`
is still a stub for 8293 cases, and the frame does not use it.

## Checks (all requested ones)

- **Oracles.**
  - 50 random-pose phantoms: PC1 within 1° of the true axis; section centroids within 0.02 of their analytic u;
    over 99% of duct mass within |ap|, |si| < 0.03.
  - Conservation exact on phantoms. On real data the duct in the atlas is 0.9967 (normalized) / 0.9997 (mm) of the
    per-case sum.
  - Splat cube lands in the expected bins.
- **Order on u.** Head < tail holds 6061/6061. Head < body < tail holds 6036/6051; the 15 violators have mislabelled
  section masks and are kept (the frame uses head/tail only).
- **Median section boundaries.** 0.364 and 0.670 vs the lesion map's 0.38 and 0.68: both within 0.02.
- **Cohort duct count.** 1031 vs 1063 MPD-present (−32). All 32 are 2–7.6 mm³ ducts under the 8 mm³ floor.

## Process notes

- **Two oracle failures were my expectations, not the code.**
  1. A 2 mm duct lying in the plane of 5 mm slices is only voxelized where a slice centre falls within 2 mm of it.
     One pose lost half its duct. The phantom now uses r = 3 mm, at least half a slice.
  2. I had asserted that the head's own 1st percentile sits at u = 0. But u = 0 is the *envelope's* 1st percentile,
     and the head holds ~26% of the envelope, so the head reaches slightly below 0. The test now compares against
     the analytic value. The per-pose ridge tolerance is likewise half a slice, in gland lengths.
- **Non-cohort cases have no CT on the PC.** `tumorlib.io.load_mask` therefore prints a "missing ct.nii.gz" line per
  call and skips its CT-grid check. `case.py` captures stdout and drops only those lines. Any other loader message is
  kept in `cases.csv:msgs`. Only the 9 gotcha-4 cases produced one.
- **The duct label is disjoint from head/body/tail** in every case (0 shared voxels), like the lesion carved out of
  `pancreas.nii.gz`. That is why tissue must be head ∪ body ∪ tail ∪ duct.

## Surprises

1. **"3 corrupted duct affines" is really 21, and they are all the same corruption: a y-axis (A↔P) flip** with
   spacing unchanged. Over 9901 cases, 21 duct affines differ from the head affine. The 8 cohort head affines that
   differ from the CT are the same flip (`L,P,S` vs `L,A,S`).
2. **The CT is not a reliable arbiter.** An aorta-posterior test (median −35.8 mm; flagged cases +15 to +76 mm, with
   a clean gap) finds 13/1608 glands mirrored in world space. 4 of them match their CT. Among included cases that
   could be tested, 8/1200 (0.7%) are mirrored. That is negligible for the atlas picture, but any per-case `ap`
   analysis needs the test. It cannot be run on most non-cohort cases, whose aorta (and kidney) masks are LFS stubs.
3. **Tumor+ glands carry ~4× more annotated duct** (median 405 vs 108 mm³ per case), reaching into body and tail.
   Tumor− duct mass is 72% in the head. The split mostly shows where annotators drew duct, not anatomy.
4. **The axial head end is bimodal** (ap peaks at −0.07 and +0.05 at u 0.15–0.25), not a fan. The coronal view shows
   one band.

## Not done / open

- Correcting mirrored cases. This would need aorta (or kidney) masks for the non-cohort cases: an LFS pull outside
  this session's pre-approval. Ask first.
- The 9 included cases whose head→tail points to the patient's right were not investigated (an x-mirror does not
  change u/ap/si).
- CONTEXT.md gotcha 4 could be updated: the corruption is a y-flip; 21 duct-vs-head disagreements across PanTS; and
  the aorta test catches mirrored glands that the CT check misses. Not edited, pending your OK.
