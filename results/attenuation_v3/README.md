# Attenuation v3 (session S3): is 40.7% "iso" real or a measurement artifact?

**Answer: mostly real, at least as far as HU measurement goes.** The whole-mask median does lose contrast to partial volume, but fixing it only moves iso from 40.7% to 37.3% (all lesions).

**Among venous-phase lesions ≥10 mm, iso stays above ~30%.** v3 gives 30.8% (167/543, 95% Wilson CI 27.0–34.8%), against 31.9% for v2. Every v3 variant lands between 29.5% and 31.0%, and the figure is 31.0% with the 70 separated lesions excluded. The lower CI bound dips just under 30%. The whole interval, though, is far above a 5–15% literature range. That range is *carried over* from CONTEXT.md and was not verified here.

Most of the headline 40.7% comes from contrast phase: iso is 48.7% in non-contrast scans (a third of the lesions) and 30.3% in venous ones.

Everything below was computed in S3 unless it is marked *carried over*.

```
.venv/bin/python src/attenuation-v3/regress_v2.py   # step 0: v2 regression   (~35 min, 2 workers)
.venv/bin/python src/attenuation-v3/phantom.py      # step 1: phantom, fixes core params (~5 min)
.venv/bin/python src/attenuation-v3/measure_v3.py   # step 2: v3 measurements  (~5 min, 2 workers, peak 1.2 GB/worker)
.venv/bin/python src/attenuation-v3/analyze_v3.py   # step 3: join + tables -> attenuation_labels_v3.csv
```
Intermediates (logs, v2 rerun, phantom trials, `analysis.md` with every table) are in `work/attenuation_v3/`, which is git-ignored. The v2 files in `results/attenuation_results/` were not touched.

## 0. Regression: v2 reproduced exactly
`src/attenuation-labeling/main.py` was imported unchanged and rerun on the 981 tumor+ cases. v2 returns no rows for negatives, so skipping them changes nothing. Against the committed `attenuation_labels.csv`:
- all 1235 (case_id, lesion_id) rows match
- 0 mismatches in `n_vox`, `n_panc_vox`, `delta_hu`, the medians and `attenuation` (max float difference 1e-14)
- counts are **hypo 561 / iso 503 / hyper 168 / unknown 3**

## 1. Phantom: partial-volume bias of the whole-mask median
**Setup.** A sphere of known contrast sits in uniform 100 HU parenchyma.
- **Scanner model.** It is simulated on a 0.25 mm grid: in-plane PSF FWHM 1.2 mm, z PSF FWHM 1.0 mm, and the voxel box (0.75 mm in-plane, slice thickness in z). The result is sampled with a random sub-voxel offset.
- **Noise.** Gaussian, 18 HU per voxel, correlated in-plane.
- **Mask.** Every voxel whose centre lies inside the sphere, the way an annotator would draw it.
- **Trials.** 100 per condition. For r=2 mm on 5 mm slices, 24/100 masks were empty and were skipped.

Mean bias (estimate − truth, HU) for a true dHU of −30. A positive value means the lesion looks less hypo, i.e. is pushed toward iso:

| r (diameter) | slice | v2 whole-mask median | v3 core median |
|---|---|---|---|
| 2 mm (4 mm) | 0.75 / 2.5 / 5 mm | +9.9 / +9.4 / +16.8 | +7.2 / +9.4 / +16.8 |
| 4 mm (8 mm) | 0.75 / 2.5 / 5 mm | +4.8 / +5.9 / +8.1 | 0.0 / +0.3 / +2.7 |
| 8 mm (16 mm) | 0.75 / 2.5 / 5 mm | +2.3 / +2.8 / +4.3 | 0.0 / −0.1 / −0.3 |

- **Size of the bias.** The whole-mask median keeps 73–84% of the true contrast at r=4 and 86–92% at r=8. The bias is proportional to dHU, so it only moves lesions whose true |dHU| is within a few HU of the cutoff.
- **The core fixes it for r ≥ 4 mm.** With a true dHU of −15 at r=4 on 5 mm slices, the lesion is called iso 40% of the time with v2 and 28% with the core. That 28% is noise, not bias.
- **r=2 mm cannot be recovered.** On 5 mm slices the brightest voxel carries only 59% of the contrast (noise-free check). Sub-5 mm dHU values should not be interpreted.
- **Cost.** For r=8 on 5 mm slices the core is noisier than the whole mask (RMSE 5.2 vs 4.6 HU), because it uses about 84 voxels instead of 760.
- **Noise estimator.** It returns 18.00 HU for white noise with a true SD of 18. For the correlated noise it returns 8.5. So `sigma_hu` is a relative high-frequency noise index, not the voxel SD. Use CNR for ranking only.

**How the core parameters were chosen, on the phantom only, before any real data.** The candidates were depth D ∈ {0.5, 1, 2} mm and minimum voxel count N ∈ {10, 20, 40, 80}.
- Pooled RMSE kept improving as N grew, because a larger N turns the core back into the whole mask. That makes it the wrong criterion for a bias audit.
- The rule used instead: the smallest N whose RMSE beats v2 at r=4 on every slice thickness. That gives **D=0.5 mm, N=40**.
- N=80 brings the bias back (+7.1 HU at r=4 / 5 mm). N=20 is noisier than v2 at r=4 / 2.5 mm.

## 2. What v3 measures (columns of `attenuation_labels_v3.csv`)
All of it runs on the bbox crop of lesion ∪ envelope before any EDT. Key is `(case_id, lesion_id)` from `tumorlib.label_lesions`. `gt_vox` equals the master table for 1235/1235 rows (asserted).

| column | meaning |
|---|---|
| `gt_vox` | lesion voxel count, same as the master table |
| `n_core`, `core_fallback`, `core_min_depth_mm`, `max_depth_mm` | Tumor core. **Depth** is the mm EDT of the lesion mask after removing its outer voxel layer (26-connected erosion). That equals the distance from a voxel to the edge minus the voxel's own size in that direction, so a thick-slice voxel on the top slice counts as boundary even though its centre is 5 mm from the outside. The core is every voxel with depth ≥ 0.5 mm. If fewer than 40 voxels qualify, it is the 40 deepest voxels (`core_fallback`), or the whole lesion if it has ≤40 voxels. |
| `tumor_whole_median`, `tumor_core_median`, `tumor_core_trim` | HU: whole-mask median (v2's tumor statistic), core median, and core 20%-trimmed mean |
| `ref_global_*`, `n_ref_global` | **Global pool** = hole-filled envelope (pancreas ∪ head ∪ body ∪ tail, 3D `binary_fill_holes`; no closing, which would pull fat in from between lobules) − (all lesions + 5 mm) − (pancreatic duct ∪ common bile duct). Median and trimmed mean. |
| `ref_local_*`, `n_ref_local` | **Local ring** = global pool within 5–15 mm of *this* lesion. The pool already excludes every lesion + 5 mm and the ducts. NaN if fewer than 50 voxels (42 lesions, 41 of them separated). |
| `dHU_v2` | carried over from v2, unchanged |
| `dHU_whole_global` | whole-mask median − global median. This isolates the change of reference. |
| **`dHU_core_global`** | **core median − global median: the primary v3 value** |
| `dHU_coretrim_global`, `dHU_core_local`, `dHU_coretrim_local` | the other estimator/reference combinations |
| `sigma_hu`, `n_sigma` | noise index: 1.4826·MAD of the in-plane Laplacian residual (x − mean of 4 in-plane neighbours) over the global pool, divided by √1.25. Slice axis taken from the master table. |
| `cnr` | `dHU_core_global / sigma_hu` |
| `attenuation_v2`, `attenuation_v3`, `attenuation_v3_local`, `attenuation_v3_trim` | hypo < −10 ≤ iso ≤ +10 < hyper, from `dHU_v2`, `dHU_core_global`, `dHU_core_local`, `dHU_coretrim_global`. An exact ±10 counts as iso, as in v2. |

Join onto `results/master_lesion_table/master_lesions.csv` for phase, thickness, size, tier and Dice.

## 3. Results

### Class counts (±10 HU)
| | v2 | whole\|global | **core\|global (v3)** | coretrim\|global | core\|local | coretrim\|local |
|---|---|---|---|---|---|---|
| hypo | 561 | 567 | **585** | 578 | 603 | 601 |
| iso | 503 | 505 | **461** | 446 | 457 | 442 |
| hyper | 168 | 163 | **189** | 211 | 133 | 150 |
| unknown | 3 | 0 | **0** | 0 | 42 | 42 |

### Which lesions change class (v2 → v3), 134/1235
| v2 \ v3 | hypo | iso | hyper |
|---|---|---|---|
| hypo | 532 | 28 | 1 |
| iso | **50** | 417 | **36** |
| hyper | 2 | 14 | 152 |
| unknown | 1 | 2 | 0 |

- Iso loses 86 lesions and gains 42, so net −42. The losses go both ways (50 to hypo, 36 to hyper), which fits partial volume pulling contrast toward 0 whatever its sign.
- The 3 v2 unknowns (an empty `pancreas.nii.gz` pool) now have a reference.
- Excluding the 70 separated lesions gives the same picture (1165 lesions: 49 iso→hypo, 35 iso→hyper).

**Where the shift comes from.**
- **Reference change.** Swapping v2's `pancreas.nii.gz` − 2.5 mm pool for the envelope − 5 mm − ducts pool changes dHU by a median of 0.0 HU in every size bin.
- **Tumor core.** For lesions with |dHU| ≥ 10 it raises |dHU| by a median of +2 HU (10–20 mm), +3 HU (20–40 mm) and +1 HU (≥40 mm). That is 7–10% more contrast, which matches the phantom's prediction.
- **Local ring vs global pool.** The ring reads about 2 HU brighter than the global pool, a median shift of −2 HU in dHU.

### Iso fraction by contrast phase
| phase | n | v2 | **v3** | core\|local |
|---|---|---|---|---|
| Venous | 557 | 31.4% | **30.3%** | 30.2% |
| Arterial | 260 | 38.2% | **34.2%** | 35.2% |
| Non-contrast | 413 | 55.2% | **48.7%** | 51.1% |
| Delay | 4 | 50% | 50% | 50% |

Non-contrast scans hold 33% of the lesions, with half of them iso. On an unenhanced scan that is expected: tumor and gland only separate once contrast is given. Excluding separated lesions, the v3 values are venous 30.6%, arterial 35.9% and non-contrast 50.1%.

### Iso fraction by slice thickness (v3)
| thickness | all phases | venous only |
|---|---|---|
| ≤2 mm | 41.6% (n=550) | 31.7% (126) |
| 2–4 mm | 28.8% (468) | 26.6% (320) |
| >4 mm | 44.7% (217) | 39.6% (111) |

- The ≤2 mm bin is 48% non-contrast (264/549), which is why its all-phase value is high.
- Within every phase, >4 mm slices give the highest iso fraction: venous 39.6%, arterial 50%, non-contrast 49%. Thick slices cost contrast that even the core does not fully recover. The phantom shows a +2.7 HU residual at r=4 mm on 5 mm slices, and real lesions are not spheres.

### Iso fraction by size (Feret, v3)
| | <5 | 5-10 | 10-20 | 20-40 | ≥40 |
|---|---|---|---|---|---|
| n | 49 | 37 | 275 | 597 | 277 |
| v2 | 22.4% | 45.9% | 41.5% | 42.7% | 38.8% |
| **v3** | 22.4% | 32.4% | **36.0%** | 39.2% | 37.9% |
| venous, v3 | 1/5 | 1/9 | 32.0% (97) | 30.4% (316) | 30.8% (130) |

Iso is nearly flat with size from 10 mm up. If it were a partial-volume artifact it would climb steeply as lesions shrink. The <5 mm bin reads low only because those estimates are unreliable (phantom r=2), not because the lesions are hypo.

### Sensitivity to the cutoff (v3, core|global)
| cohort | n | ±5 HU | ±10 HU | ±15 HU |
|---|---|---|---|---|
| all | 1235 | 22.0% | 37.3% | 51.1% |
| venous ≥10 mm | 543 | 17.3% | **30.8%** | 41.1% |
| venous ≥10 mm, not separated | 509 | 17.7% | 31.0% | 41.8% |
| arterial ≥10 mm | 246 | 15.9% | 33.7% | 49.2% |
| non-contrast ≥10 mm | 355 | 34.6% | 52.4% | 70.4% |

For v2 the venous ≥10 mm figures were 19.0 / 31.9 / 42.9%. Even at ±5 HU, one venous lesion in six is within 5 HU of its gland. **272 of the 461 v3-iso lesions have |dHU| ≤ 5 HU**, so this is not a pile-up of borderline ±9 HU cases. Of the 167 venous ≥10 mm iso lesions, 88 have negative, 12 zero and 67 positive dHU.

### Diagnosis field
None exists. S1 checked the metadata and all 9901 structured reports and found no diagnosis, pathology or lesion-type field (*carried over*). So iso cannot be split by PDAC vs other lesion types. The best candidate explanation for "too many iso" is that `pancreatic_lesion` is not PDAC-only, and it stays untested.

### Noise and CNR
- The median `sigma_hu` falls with slice thickness: 18.6 HU (≤2 mm), 12.3 (2–4), 9.3 (>4), as it should.
- The median CNR is −2.3 for v3-hypo, −0.05 for v3-iso and +1.3 for v3-hyper.
- Only 121 of 461 iso lesions lie within 2 white-noise standard errors of 0 (`1.2533·sigma/√n_core`). That SE is optimistic because the noise is correlated, so this number is soft. The iso class is still clearly not just noise-limited measurements.

## Caveats
- **The core helps less on small lesions.** It falls back to the 40 deepest voxels in 235/1235 lesions: 39% of 10–20 mm lesions and ≥89% below 10 mm. At 10–20 mm the phantom residual stays ≤2.7 HU. Below 10 mm, and especially below 5 mm, v3 dHU is still biased toward 0.
- **Uniform phantom.** It has no fat edge, no heterogeneity (necrotic centre with an enhancing rim, where a median can land at parenchyma level) and no mask errors. Masks that over-segment into fat or vessels are not modelled.
- **All masks are PanTS GT.** The envelope, ducts and CBD are imperfect (CONTEXT.md gotcha 5). The hole fill adds any enclosed non-duct structure, such as a vessel, to the reference pool; medians limit the damage.
- **Phase labels are metadata, not verified.** The median reference HU is 61 for "Arterial", 81 for "Venous" and 41 for "Non-contrast". Pancreatic-phase parenchyma is usually brighter than venous, so many "arterial" scans may be early-arterial. Treat phase as a noisy label.
- **Aorta and liver noise were not computed.** It was optional, and those masks are un-pulled git-LFS stubs on the PC (1300/1300 checked). `sigma_hu` is relative only (phantom: 0.47× the voxel SD under correlated noise).
- **Separated lesions.** 70 lie outside the gland, where a gland-referenced dHU is nominal. Results are given with and without them where it matters.
- **Not a clinical label.** HU contrast ≠ radiologist attenuation, and the literature iso rate is quoted from memory (*carried over*, unverified).

## What surprised me
1. **Partial volume is real but small.** Real 10–40 mm lesions gained 7–10% contrast in the core, exactly as the phantom predicted. Yet only 134 lesions changed class, and iso fell by just 3.4 points.
2. **Non-contrast scans are a third of the tumor+ cohort,** and they carry half the iso lesions (201 of 461).
3. **Iso is flat with size from 10 mm up (36–39%),** which argues against an artifact.
4. **"Arterial" parenchyma (median 61 HU) is darker than "venous" (81 HU).** This points to early-arterial timing or unreliable phase labels.
5. **Thick slices (>4 mm) have the highest iso fraction in every phase.** That residual thickness effect is the one part that still looks like measurement.
