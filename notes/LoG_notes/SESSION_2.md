# Session S2: does nnU-Net "whisper" inside the tumors it misses? (2026-09-28)

Full results, definitions and tables: `results/nnunet_subthreshold/README.md`. Every number there was computed this
session. This note covers what was done around it.

## What was computed
- Step 0 oracle on 3 cases, then the same checks over all 1308 cases:
  - argmax equals the saved seg (0 voxel mismatches)
  - p > 0.5 equals the saved seg
  - channels sum to 1 within 8.8e-8
  - the transpose is right: flipping any axis kills the tumor overlap
- Step 1 regression reproduced every carried-over number exactly: 631/288/62, 0/31.8/65.6/85.6% by largest
  lesion, and lesion detect@any 2.0/0.0/25.8/59.8/83.0%.
- Steps 2-4: the per-lesion table, the FROC ladder (2 hit rules × size bins) and the saturation report.
- Step 5: float16 crops (prob + clipped logit) for 1297 cases and 9 component tables, synced to the PC at
  `~/research/nnunet_probs/`.

## How
- Code lives in `src/nnunet-subthreshold/`:
  - `common.py`: paths, loaders, logit helpers
  - `oracle.py`: step 0
  - `case_pass.py`: phantom oracle, then the per-case pass
  - `summarize.py`: asserts, deliverables, `summary.txt`
- It runs on the server with 4 workers, each handling one case at a time. The pass took 32 min; peak RSS was 3.0 GB
  per worker on PanTS_00008854.
- IO is kept separate from the analysis, so the phantom exercises the exact function (`analyze`) used on real data.
- The server has no CTs. tumorlib's CT grid points at a symlink farm of `gt_segmentations` headers
  (`work/nnunet_subthreshold/ctgrid`) through `TUMORLIB_CT_ROOT`. That spacing matches S1's CT-affine spacing
  within 5.4e-8 mm. tumorlib itself was not changed.

## Server repo hygiene (user-requested force sync)
- The server repo was at `b942917`: 39 commits behind `origin/main`, with uncommitted edits to the old MedFormer
  `src/training/*` and `src/config/*`, which no longer exist in main. There were also untracked copies of
  duct-cutoff, nnunet and tumorlib files.
- **Checked first:** the only untracked files that are also tracked in main were 10, all byte-identical.
- **Backup** in `~/repo_backup_pre_S2_2026-09-28/`, outside the repo: `uncommitted_tracked.patch`, `old_HEAD.txt`,
  and `files.tgz` holding the modified files plus `percase_eval.*`, `results/`, `src/duct-cutoff`, `src/nnunet`
  and `src/tumorlib`.
- Then ran `git reset --hard origin/main` and **no `git clean`**. Untracked `exp/` (7 GB MedFormer checkpoints),
  `log/`, `percase_eval.*` and the untracked results dirs are still there. Nothing under `~/research/`
  (nnU-Net results, preprocessed data, masks) was touched.
- `master_lesions.csv` was also scp'd as asked. Its md5 equals the local copy.

## Caveats
Listed in the results README. The main ones:
- The SR and GT anatomy are ground truth.
- Negatives are thinner-sliced than positives.
- Comparing a lesion's peak with an SR-wide peak is size-asymmetric.
- The strict peak hit rule is non-monotone in t.

## Surprises
1. The probabilities never reach the float32 ceiling. The max is 0.9999875 over all volumes. The histogram has a
   soft step near logit 6 (p ≈ 0.9975) instead, so the upper clip never binds.
2. 11 negatives have all four pancreas masks empty, so they have no search region. The same holds on the PC masks.
3. Missed <10 mm lesions: median peak p in lesion + 2 mm is 1.3e-4. 82% of negatives' SR peaks sit at or above
   the median missed lesion's peak. Most (76/85) sit at or below their own case's SR 99th percentile.
