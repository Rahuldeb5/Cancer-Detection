# Contrast phase from the scan itself (session S12)

**Answer: the metadata phase label is wrong for about 3 scans in 10, and the negatives were not really phase-matched.**
- **Rule.** The estimate comes from two vessels in each scan: median aorta HU and median portal/splenic-vein HU, classified with a fixed 3-cluster Gaussian mixture.
- **Agreement with metadata.** 69.0% (95% CI 66.3–71.5%; 849/1231 confident estimates that also have a 3-class metadata label).
- **Independent check.** Gland HU never enters the rule, yet it follows the *estimated* phase, not the metadata phase.
- **Metadata "Non-contrast" scans the rule calls venous** (108 scans) have an 80 HU gland.
- **Metadata "Venous" scans the rule calls non-contrast** (27 scans) have a 30 HU gland.

Main consequences:
- **S3's "arterial gland is darker than venous" puzzle is explained.** Metadata-"Arterial" is a mixture. It holds 34.8% venous-looking scans, 32.3% early arterial and 12.8% non-contrast. Only 16.0% are late-arterial (pancreatic phase), and in those the gland reads **93 HU, brighter than venous (81 HU)**, as physiology predicts.
- **Tumor+ and tumor− differ in phase under the new labels.** Venous is 58.6% of tumor+ vs 41.6% of tumor− (χ² p = 8×10⁻⁶). Under the metadata labels they look matched (p = 0.94), but the matching was done on a noisy label.

Everything below was computed in S12 unless marked *carried over*.

```
PYTHONPATH=src .venv/bin/python -m phase.oracle        # step 3 oracle (~3 min)
PYTHONPATH=src .venv/bin/python -m phase.affine_check  # step 1 check: masks align on the CT grid (~2 min)
PYTHONPATH=src .venv/bin/python -m phase.measure       # step 2: 1308 cases, 4 workers, 7.2 min
PYTHONPATH=src .venv/bin/python -m phase.analyze       # steps 4-5: GMM, tables, PNGs (~1 min)
```
- **Code.** `src/phase/`. All rule constants are frozen in `core.py`.
- **Intermediates.** In git-ignored `work/phase/`: the measurement CSV and log, oracle results, the affine check, and `analysis.md` with every table quoted here.
- **Figures.** In `notes/LoG_notes/SESSION_12_scatter_{metadata,estimated}.png`.

## 1. Which masks
- **Aorta.** `segmentations/aorta.nii.gz`.
- **Portal/splenic vein.** `segmentations/veins.nii.gz`. PanTS has no file named "portal vein".
  - **Why `veins` is the portal system** (12 random cases): it is 4.6–49 mL and has 0 voxels shared with the aorta. It is disjoint from `postcava` (the IVC; ≤4 shared voxels). It reaches beyond both ends of the pancreas, toward the splenic and liver hila, and 38–91% of it lies within 10 mm of the gland.
  - **It likely also contains the SMV and intrahepatic branches.** The middle-80%-slices rule trims those ends.
  - **Two masks not used:** `celiac_artery` and `superior_mesenteric_artery` are arteries, and `postcava` is used for QC only.
- **Availability.** Both files are real (not LFS stubs) for **1308/1308** cases, so the 5% STOP rule was not triggered. Every one of the 2616 files uses encoding 1 (raw {0,1}, slope 1). All are loaded with `tumorlib.io.load_mask`.
- **Empty masks.** `veins` is empty in 13 cases. Every mask (aorta, veins, pancreas, lesion) is empty in 3 more.
- **Affines.** The aorta affine differs from the CT affine in **367/1308** cases:

  | aorta mask vs CT | cases |
  |---|---|
  | LPS vs RAS | 300 |
  | translation only | 55 |
  | LPS vs LAS | 7 |
  | LPS vs LAI | 5 |

  `veins` differs only in the 9 cases already known (CONTEXT gotcha 4).
- **Checked (`affine_check.py`): the voxel arrays still line up with the CT.** In 17 sampled flipped cases, reading the aorta on the CT grid gave IQR 21–129 HU and 0 voxels shared with the veins. Remapping it through its own affine raised IQR to 28–370 HU, and in 10 of those cases it landed on the veins mask. The 6 translation-only cases sampled read identical voxels either way. So the CT grid is used throughout, as `load_mask` already does.

## 2. Measurements (columns of `phase_estimate.csv`)
**Vessel HU (`aorta_hu`, `pv_hu`).**
- **Erosion.** One-voxel erosion, 6-connected: every voxel that touches the outside through a face is dropped.
- **Slices.** Only the middle 80% of the slices that contain the vessel, along the CT slice axis (S1's rule).
- **Statistic.** The median HU of what remains.
- **Voxel counts.** `aorta_n_vox` and `pv_n_vox` are the eroded voxels actually used.
- **`low_vox_flag`.** Either vessel has fewer than 200 voxels.
  - 22 cases are flagged. 14 of them have 0 voxels: 13 empty `veins` masks, plus 1 vein on 5 mm slices that erodes away.
  - The other 8 come from small vein annotations on thin-slice scans, not thick slices.
- **`aorta_iqr` and `pv_iqr`.** Spread inside the vessel, for QC.

**Gland HU (`gland_hu`).**
- The S3 v3 global pool, reused through `attn_v3`'s constants and functions: hole-filled envelope − (lesions + 5 mm) − ducts.
- **Regression check:** it equals S3's `ref_global_median` for **981/981** tumor+ cases (max difference 0.015 HU, from S3's 4-significant-digit CSV).
- For negatives there is no lesion shell to remove.
- NaN for 12 cases with an empty envelope.
- **Descriptive only: it is not an input to the rule.**

**Phase columns.**
| column | meaning |
|---|---|
| `est_phase` | non-contrast / arterial / venous; `uncertain` if the max posterior < 0.8; `unmeasurable` if aorta or PV is NaN (17 cases, not in the fit) |
| `posterior` | max GMM posterior |
| `argmax_phase` | the most likely phase, even when uncertain |
| `pv_enh_frac` | (PV − 39.2) / (162.7 − 39.2): how far the PV is from the non-contrast centroid toward the venous centroid |
| `early_arterial` | estimated arterial and `pv_enh_frac` < 0.5 (definition fixed in `core.py` before the scatter was looked at) |
| `metadata_phase` | metadata `ct phase`, untouched |
| `slice_thickness_mm` | spacing along the slice axis |
| `qc_note` | low voxels / aorta–postcava overlap (10 cases, >1000 voxels; an annotation overlap) / mask affine ≠ CT |

## 3. Oracle (before any real data): passes for the aorta, fails the strict tolerance for a small vein
**Setup.**
- Synthetic cylinders: vessel 300 HU in fat −100 HU, with a partial-volume voxel box, PSF blur and 15 HU noise.
- Written to disk as fake cases in **both** mask encodings and read back with `load_mask`.
- The mask is shifted by 1 voxel along x, y, z or diagonally.
- Each shape was tested on 0.8, 2.5 and 5 mm slices.

**Results.**
| shape | slices | worst error (1-voxel shift) | verdict |
|---|---|---|---|
| aorta-like, r = 11 mm, along z | 0.8 / 2.5 / 5 mm | −2.7 HU | pass |
| vein-like, r = 5 mm, in-plane | 0.8 mm | −5.7 HU (diagonal shift) | **fail** (tolerance 5) |
| vein-like, r = 5 mm, in-plane | 2.5 mm | −7.1 / −7.2 HU (shift along the slice axis / diagonal) | **fail** |
| vein-like, r = 5 mm, in-plane | 5 mm | 0 voxels survive → NaN | unmeasurable, never a wrong value |

- **The two encodings gave identical numbers.**
- **I did not change the erosion to pass.** The worst error is 1.8% of the 400 HU vessel–fat contrast, and it scales down with real contrast. The three cluster centroids are ≥ 38 HU apart in PV and ≥ 95 HU apart in aorta.
- **A 2-voxel shift is not survivable.** On 2.5 mm slices the vein then reads −6.5 HU, i.e. fat, with 4505 voxels. This was informational only, but it shows the method assumes masks are off by at most one voxel.

## 4. Phase rule
**The fit.** A 3-component full-covariance GMM on (aorta HU, PV HU) over the 1291 measurable cases (n_init 20, seed 0). Clusters are named from the centroids only. **No lesion mask, lesion HU, tumor status or metadata phase enters the fit.**

| cluster | aorta mean (SD) | PV mean (SD) | weight |
|---|---|---|---|
| non-contrast | 40.9 (6.4) | 39.2 (7.0) | 0.22 |
| venous | 148.0 (37.7) | 162.7 (42.5), corr 0.95 | 0.56 |
| arterial | 242.9 (92.0) | 124.9 (54.2) | 0.23 |

**Stability.**
- Argmax labels are identical across seeds 1–10.
- Refitting without the 8 measured low-voxel cases changes 1 label (99.92% agreement).
- The median posterior is 0.978. **53 cases (4.1%) are uncertain.**

**Is there a 4th group?** BIC keeps falling up to k=6 (ΔBIC vs k=3: −18 at k=4, −58 at k=5, −76 at k=6). That is expected when clusters are elongated arms rather than Gaussian blobs, so it is not evidence for discrete phases.
- **k=4** splits off 26 very bright arterial scans (centroid 285 / 131).
- **k=5** splits the arterial fan into an early-arterial-like group (274 / 70, ~119 cases) and a late-arterial-like group (210 / 157, ~155 cases). It also splits the venous arm along its length.

The scatter shows a **continuum**:
- **Arterial fan:** runs from early arterial (bright aorta, unopacified PV ≈ 50–80 HU) to pancreatic phase (PV approaching the aorta).
- **Venous arm:** runs along aorta ≈ PV, from about 90 to 250 HU.

The 3-class rule is kept. Early vs late arterial is reported with the a priori `pv_enh_frac` < 0.5 split.

## 5. Results
### Metadata (rows) vs estimated (columns)
| metadata \ estimated | non-contrast | arterial | venous | uncertain | unmeasurable | total |
|---|---|---|---|---|---|---|
| Non-contrast | **214** | 35 | 108 | 14 | 9 | 380 |
| Arterial | 36 | **136** | 98 | 8 | 4 | 282 |
| Venous | 27 | 77 | **500** | 31 | 4 | 639 |
| Delay | 1 | 0 | 5 | 0 | 0 | 6 |
| missing | 0 | 1 | 0 | 0 | 0 | 1 |
| total | 278 | 249 | 711 | 53 | 17 | 1308 |

- **Non-contrast:** 143/380 metadata-"Non-contrast" scans (37.6%) look contrast-enhanced.
- **Venous:** metadata-"Venous" is the most reliable label (500/639 = 78.2%).
- **Delay:** the 6 "Delay" scans look venous (5) or unenhanced (1).

### How many metadata-"Arterial" scans look venous or early arterial?
| metadata Arterial (n=282) | n | % (95% CI) |
|---|---|---|
| looks venous | 98 | 34.8 (29.4–40.5) |
| early arterial (`pv_enh_frac` < 0.5) | 91 | 32.3 (27.1–37.9) |
| late arterial / pancreatic | 45 | 16.0 (12.1–20.7) |
| looks non-contrast | 36 | 12.8 (9.4–17.2) |
| uncertain / unmeasurable | 8 / 4 | 2.8 / 1.4 |

**Venous or early arterial: 189/282 = 67.0% (61.3–72.2%).**

### Gland HU (S3 v3 pool; descriptive, not in the rule)
| estimated phase | n | median (IQR) |
|---|---|---|
| non-contrast | 277 | 33 (20–40) |
| arterial: all | 248 | 78.5 (57–98) |
| arterial: early | 114 | 64 (50–83) |
| arterial: late | 134 | **93** (73–108) |
| venous | 711 | 81 (64–100) |
| uncertain | 53 | 73 (42–92) |

By metadata phase: Non-contrast 40 (27–73), Arterial 68 (47–93), Venous 81 (62–100).
- **The wide metadata-Non-contrast IQR (27–73)** is the 143 enhanced scans hiding in it.
- **Gland HU follows the estimate, whatever the metadata says.** Gland median by estimated phase within each metadata row:
  - non-contrast: 30–36 HU
  - venous: 80–85 HU (the 5 "Delay" scans: 60 HU)
- **Only 0.4% of estimated non-contrast scans have a gland above 60 HU.** 7.5% of estimated venous scans have a gland below 40 HU, likely fatty glands; not investigated.
- **Per-lesion vs per-case.** S3's per-lesion figures were 61 (arterial) vs 81 (venous) (*carried over*); per case they are 68 vs 81.

### Tumor+ vs tumor− phase mix: were negatives phase-matched?
| | tumor+ (n=981) | tumor− (n=327) |
|---|---|---|
| **estimated** non-contrast | 19.9% | 25.4% |
| **estimated** arterial | 17.2% | 24.5% |
| **estimated** venous | **58.6%** (55.5–61.7) | **41.6%** (36.4–47.0) |
| uncertain / unmeasurable | 3.9% / 0.4% | 4.6% / 4.0% |
| metadata Non-contrast / Arterial / Venous | 28.8 / 21.5 / 49.1% | 29.7 / 21.7 / 48.0% |

**Under the metadata labels the groups look matched; under the estimated labels they are not.**
| labels | χ² (3 classes) | p |
|---|---|---|
| metadata | 0.12 | 0.94 |
| estimated | 23.4 | 8×10⁻⁶ |

**The label disagreement rate is the same in both groups** (31.0% tumor+ vs 30.6% tumor−). What differs is its direction:
| metadata label | estimate | tumor+ | tumor− |
|---|---|---|---|
| Non-contrast | venous | 94/283 (33%) | 14/97 (14%) |
| Venous | arterial | 45/482 (9%) | 32/157 (20%) |

13 of the 17 unmeasurable cases are negatives, mostly empty `veins` masks.

### Slice thickness by estimated phase (mm)
| estimated | n | median (IQR) | ≤2 mm | 2–4 mm | >4 mm |
|---|---|---|---|---|---|
| non-contrast | 278 | **0.80** (0.80–1.25) | 84% | 8% | 8% |
| arterial | 249 | 1.25 (1.0–2.5) | 65% | 20% | 14% |
| venous | 711 | 2.5 (2.0–2.5) | 27% | 55% | 18% |
| uncertain | 53 | 2.5 (1.25–3.0) | 40% | 38% | 23% |

- **Phase and thickness are strongly confounded:** true non-contrast scans are almost all thin-sliced.
- **Within estimated venous**, the median thickness is 1.5 mm for tumor− vs 2.5 mm for tumor+. So the S1 thickness confound (*carried over*) persists within phase.
- Study type cannot explain the thin non-contrast group: it is blank for most cases.

## Caveats
- **No ground truth.** The only independent check is gland HU. It agrees with the estimate, but gland HU is a function of the same contrast bolus, so this validates "enhanced vs not" and "how enhanced". It does not validate the protocol name.
  - Split-bolus protocols or delayed scans would look venous here. The 6 "Delay" scans did.
- **The arterial GMM component is very wide and grabs some outliers.**
  - 10/249 estimated-arterial scans have PV ≥ aorta. 8 look venous/delayed; 2 have an unenhanced aorta (43–44 HU) with a 127 HU "vein" and a 40 HU gland, probably a mask problem.
  - 29/53 uncertain cases sit *above* the diagonal (PV > aorta), where the wide arterial component overlaps the venous one.
  - The rule was frozen before the scatter was seen, so these are reported, not fixed. `aorta_hu > pv_hu` is a cheap extra filter for analyses that need clean arterial scans.
- **Oracle.** It fails the 5 HU tolerance for a 5 mm-radius vein shifted along the slice axis (≤ −7.2 HU, biased toward fat). Misregistration of 2 voxels or more would give fat-level values; nothing detects that except `pv_iqr`.
- **`veins` = portal/splenic vein is inferred from geometry,** not from PanTS documentation.
- **Early arterial (`pv_enh_frac` < 0.5) is a convention,** with its cut-point fixed a priori. The k=5 GMM split the arterial fan at a similar place, but the boundary is a continuum.
- **Aorta–postcava mask overlap** above 1000 voxels in 10 cases. Measured anyway (8/10 have posterior > 0.95) and flagged in `qc_note`.
- **Plot range.** 3 cases with aorta > 484 HU are outside the plot range, and stated on the figure.

## What surprised me
1. **Metadata "Non-contrast" is wrong more than a third of the time** (143/380 enhanced, gland 80 HU). I expected the arterial/venous boundary to be the noisy part, not contrast vs no contrast.
2. **The S3 puzzle was mixture, not physiology.** True late-arterial glands (93 HU) are brighter than venous ones (81 HU). Metadata-"Arterial" is only 16% late arterial.
3. **Negatives are short on venous scans by 17 points** (41.6% vs 58.6%), though the metadata says they are perfectly matched. This touches every tumor+ vs tumor− comparison so far: specificity, S2 negative search-region peaks, S4 FP-source (a) and the S9 bank matching.
4. **True non-contrast scans are nearly all ≤ 1.25 mm** (median 0.8 mm), so "phase" and "thickness" are close to the same variable in this cohort.
5. **The 3-cluster picture is really a blob plus two arms.** The arterial "cluster" is a fan from early arterial to pancreatic phase, not a group.
