# Session S3: attenuation audit (2026-09-28)

Full write-up, tables and caveats: `results/attenuation_v3/README.md`. Code: `src/attenuation-v3/`.

## Steps and outcome
0. **Regression.** v2 (`src/attenuation-labeling/main.py`, imported unchanged) reproduced hypo 561 / iso 503 / hyper 168 / unknown 3, with 0 mismatches on 1235 rows.
1. **Phantom.** On a sphere in noisy anisotropic volumes, the whole-mask median loses 16–27% of the contrast at r=4 mm and 8–14% at r=8 mm. The PV-aware core fixes r ≥ 4 mm; r=2 mm cannot be recovered. Core parameters (D=0.5 mm, N=40) were fixed on the phantom before any real data.
2. **v3 measurements.** Core median and trimmed mean against (i) a global envelope pool and (ii) a local 5–15 mm ring. Noise σ from the in-plane Laplacian MAD, plus CNR. Every step runs on a crop (peak 1.2 GB per worker, 2 workers).
3. **Analysis.**
   - 134/1235 lesions change class; iso goes from 40.7% to 37.3%.
   - Among venous lesions ≥10 mm, iso stays above ~30%: 30.8% (CI 27.0–34.8%).
   - Phase is the main driver: non-contrast scans are 49% iso.
   - There is no diagnosis field (S1).

## Code added (only new files; no existing script changed)
| file | role |
|---|---|
| `attn_v3.py` | pure functions: `core_mask`, `noise_sigma`, `classify`, frozen parameters |
| `regress_v2.py` | step 0 |
| `phantom.py` | step 1, core-parameter selection |
| `measure_v3.py` | step 2 → `work/attenuation_v3/measure_v3.csv` |
| `analyze_v3.py` | step 3 → deliverable CSV + `work/attenuation_v3/analysis.md` |

CONTEXT.md: the Attenuation section was updated with the v3 findings.

## Process notes
- **The first selection rule was wrong.** Pooled RMSE over all phantom conditions kept picking the largest N at the edge of the grid (40, then 80), because a larger N turns the core back into the whole mask. I switched to a bias-first rule: the smallest N that beats v2's RMSE at r=4 on every thickness. The rule was changed on phantom data only.
- **σ is a relative index.** The Laplacian-MAD estimate is exact for white noise but reads 0.47× the voxel SD under the phantom's correlated noise.
