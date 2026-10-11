# Session S12: contrast phase from the scan itself (2026-10-03)

Full write-up and tables: `results/phase_estimate/README.md`. Code: `src/phase/`. Intermediates: `work/phase/` (git-ignored).
Figures: `SESSION_12_scatter_metadata.png` and `SESSION_12_scatter_estimated.png` (this folder).
PC only, CPU only, 4 workers. The server was not touched.

## Scoreboard
| # | question | answer | good / bad |
|---|---|---|---|
| 1 | Do aorta and portal/splenic-vein masks exist for the cohort? | `aorta.nii.gz` and `veins.nii.gz` are real for 1308/1308 (0 LFS stubs; all encoding 1). `veins` is the portal/splenic system by geometry, likely plus the SMV. Empty in 13 (+3 all-empty cases). | good |
| 2 | Can the masks be used despite affine mismatches? | Yes. The aorta affine differs from the CT in 367/1308 cases (300 LPS-vs-RAS), but the voxels align on the CT grid. Remapping by the affine puts the aorta on the veins (`affine_check.py`). | good (new gotcha) |
| 3 | Oracle: 1-voxel-shifted mask within 5 HU of truth, both encodings? | Aorta-like: yes (≤2.7 HU), both encodings identical. Vein-like r=5 mm: **no**, −5.7 to −7.2 HU in 3 of 8 measurable shift conditions (per encoding). On 5 mm slices it is unmeasurable (NaN). | bad (small: ≤1.8% of contrast; method not tuned) |
| 4 | Vessel HU measurable? | Aorta 1305/1308, PV 1291/1308. 8 more measured with <200 voxels (flagged). | good |
| 5 | Gland HU reuses S3 exactly? | Yes. 981/981 tumor+ equal S3 `ref_global_median` (max 0.015 HU, CSV rounding). | good |
| 6 | Is the 3-cluster rule stable? | Yes. Identical across 10 seeds, 99.92% without the low-voxel cases; median posterior 0.978; 53 uncertain (4.1%). | good |
| 7 | Is there a 4th group? | BIC keeps falling (k=4 −18, k=5 −58). Clusters are arms, not blobs. k=5 splits arterial into early (274/70) and late (210/157). Kept 3 classes. | neutral (continuum) |
| 8 | Does metadata phase agree with the estimate? | 69.0% (66.3–71.5). 143/380 metadata "Non-contrast" look enhanced. | **bad for the metadata** |
| 9 | Is the estimate right where they disagree? | Gland HU (not in the rule) follows the estimate: est-NC 30–36 HU and est-venous 80–85 HU, whatever the metadata row says. | good |
| 10 | Metadata "Arterial" that look venous or early arterial? | 98 venous (34.8%) + 91 early arterial (32.3%) = 189/282 (67.0%, CI 61–72). Only 45 (16%) late arterial; 36 (12.8%) non-contrast. | bad for the metadata |
| 11 | Why is "arterial" gland darker than venous (S3)? | It is a mixture. Est late-arterial gland 93 HU > venous 81 > early-arterial 64 > NC 33. | good (puzzle solved) |
| 12 | Were negatives really phase-matched? | **No.** Est venous is 58.6% tumor+ vs 41.6% tumor− (p 8e-6); matched only on the noisy metadata label (p 0.94). | **bad** |
| 13 | Slice thickness by estimated phase? | NC median 0.8 mm (84% ≤2 mm), arterial 1.25, venous 2.5. Within venous: tumor− 1.5 vs tumor+ 2.5. | bad (phase ≈ thickness confound) |
| 14 | Any rule artifacts? | The wide arterial component claims 10 scans with PV ≥ aorta. 29/53 uncertain lie above the diagonal. Reported, not fixed (rule frozen). | minor bad |

## Steps
1. **Masks.** Listed all 28 mask files per case; real vs stub per structure. Identified `veins` as the portal system by volume, IVC/aorta disjointness, extent and proximity to the gland. Classified the affine mismatches and showed the CT-grid reading is correct.
2. **Oracle.** Ran before any real data (`oracle.py`). Synthetic cylinders with PV, PSF and noise, written in both encodings, read with `load_mask`. Outcome is in the scoreboard; the erosion was kept as specified.
3. **Measurement.** `measure.py`: 1308 cases in 7.2 min. Eroded middle-80%-slice medians, plus the S3 gland pool with a regression assert.
4. **Rule.** `analyze.py`: 3-component GMM on (aorta, PV) only; names from centroids; posterior < 0.8 → uncertain. The early-arterial split (`pv_enh_frac` < 0.5) was written into `core.py` before the scatter was plotted.
5. **Report.** Confusion table, gland HU, tumor± mix, thickness, scatters.

## Code added (new files only; no existing script changed)
| file | role |
|---|---|
| `src/phase/__init__.py` | package docstring |
| `src/phase/core.py` | frozen constants; vessel stats, gland pool (S3 reuse), cluster naming |
| `src/phase/oracle.py` | step 3 oracle |
| `src/phase/affine_check.py` | affine tally + CT-grid vs world-mapped aorta check |
| `src/phase/measure.py` | per-case HU → `work/phase/measure.csv` |
| `src/phase/analyze.py` | GMM, tables, PNGs → `results/phase_estimate/` |

## Process notes
- **The oracle first tripped tumorlib's LFS-stub guard.** An 80 mm phantom gzips below 1 KB. Fixed by using a realistic 200 mm in-plane FOV, not by changing tumorlib.
- **The oracle failure was not "fixed".** A 2-voxel erosion or in-plane-only erosion would probably pass, but changing the method to meet a gate is the tuning the rules forbid. The bias is small next to the cluster gaps.
- **Not done (not asked):** lesion-level tables by estimated phase, e.g. iso fraction or nnU-Net Dice per estimated phase. The rule is frozen now, so S8 can join `phase_estimate.csv` on `case_id` directly.
- **CONTEXT.md updated (2026-10-05, after the session):** session status; cohort phase-matching claim corrected; S3 arterial-gland line explained; new "Contrast phase (S12)" section; gotcha 4 extended to the 367 aorta affine mismatches.
