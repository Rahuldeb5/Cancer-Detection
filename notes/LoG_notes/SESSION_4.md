# Session S4: does a scale-space blob map separate small tumors from clutter? (2026-09-29)

**Answer: no, in every stratum and at every threshold.** Full results, tables and caveats:
`results/log_candidates/README.md`. Code: `src/log-candidates/`. Every number there was computed
this session; carried-over columns are labelled.

The headline, for the group the brief defined as realistic (10-20 mm, hypo/hyper on v3,
contrast-enhanced, n=106): the lesion's own strongest blob response is beaten by a median 76.5% of
same-size random patches of the *same* pancreas. Ten candidates per case (8.3 false peaks/case)
finds 2.8% of them (CI 1.0-8.0%). At a tolerable false-peak rate (≤3/case, z ≥ 6) sensitivity is 0%.

## What was built

| file | role |
|---|---|
| `logblob.py` | pure array functions: σ²·LoG response, per-(case, scale) MAD normalization, local maxima + 5 mm greedy thinning, cross-scale pooling, Hessian shape terms. No I/O, so the phantom exercises exactly what the cohort run calls. |
| `phantom.py` | step 0: the four known-answer checks |
| `case_pass.py` | per-case pass → `work/log_candidates/cases/*.pkl` (resumable) |
| `summarize.py` | oracles, the three deliverable CSVs, `work/log_candidates/summary.txt` |
| `tests/test_logblob.py` | 16 fast unit tests (full suite now 47 passed, 2 slow deselected) |

Run on the PC, 4 workers, one case per worker at a time: 1308 cases in **33 min**, no full-volume
1 mm array ever cached. No existing script was modified.

## Steps and outcome

0. **Phantoms, all four passed before any real data.** Scale law on the raw σ²·LoG within one ladder
   step of r/√3 in 66/66 runs; (0.8, 0.8, 5) mm → 1 mm keeps in-plane localization ≤1 mm in 20/20
   trials with the centre offset randomized (z error median 0.38, max 1.04 mm — measured, not
   assumed); noise-only null gives ~5-7 false peaks per 200 cm³ at z ≥ 4 and ~0 at z ≥ 5, with
   counts below z ≈ 3.4 censored by K = 50; `blobness` separates a 2 mm ball from a 2 mm tube
   completely (ball ≥ 0.503, tube ≤ 0.153 → cut frozen at 0.33).
1. **Coverage ceiling.** 73/1235 lesions lie entirely outside the search region (5.9% loss before any
   detector runs), concentrated in the `abutting` tier (mean coverage 0.358). Reported, not tuned.
2. **Ranks.** Target group: 14.2% have an in-lesion peak anywhere in the 50-candidate list, 0% at
   rank 1, median rank 36.
3. **False peaks from both sources.** Source (b) (positives' non-tumor region) is the headline.
4. **Random same-size patches**, so a small lesion is never compared against a whole-region maximum.
5. **Stratified** by size, v3 class, phase, region, thickness, tier and nnU-Net detected/missed.
6. **Structures**: top false peaks sit on veins (30.4%) and duodenum (24.9%), median 3.5 mm away.

## The methodological finding (more useful than the negative one)

**The a-priori design of pooling scales by noise-normalized score is strictly dominated.** Ranking
by z = response/MAD sends **80.6% of the candidate list — and 82.3% of its rank-1 peaks — to the
finest scale, σ = 1.5 mm.** A single fixed σ = 5 mm produces only 6.5 candidates per case and still
gets 20.8% of the target group into the top 10, versus the pooled list's 2.8% at 50 candidates:
~7x the sensitivity at ~1/8 the cost.

Mechanism: the response MAD **rises** with σ on real scans (16.9 → 22.8 HU across the ladder)
because inside a real pancreas it measures vessels, ducts and bowel wall, not photon noise. In the
noise-only phantom it **falls** with σ. So the normalization the phantom validated penalizes exactly
the scales that match lesion size.

I did **not** adopt σ = 4 or 5: choosing it from this table would be tuning on evaluation lesions
(CONTEXT rule 3). The whole ladder is reported so nothing is selected. It also would not rescue the
result — the best row is 22.6% top-10, and its median rank of 8 out of 16 candidates means ~7
non-lesion peaks in the same pancreas still outscore the lesion.

## Process notes

- **The phantom caught a real bug.** The cross-scale pooling used `all()` over an empty kept-list,
  which is `True`, so it rejected every candidate and the pooled list came back empty. Check A passed
  anyway because it reads a probe rather than the peak list; check B failed immediately.
- **My first scale-law assertion was wrong, on the phantom only.** I asserted the law on z. With no
  noise the response MAD is ~0, z collapses to (raw / floor) for every scale, and the argmax lands on
  the ball's sharp edge. Fixing it meant asserting on the raw response and *reporting* the z shift
  separately — which is how the real-data finding above got noticed at all.
- **My first z-limitation claim was untested.** I wrote "a 5 mm grid cannot localize better than
  ~2.5 mm" as an assertion; it is a property of the grid, not of what the detector achieves. I
  randomized the sub-voxel offset per trial and measured it instead: median 0.38, max 1.04 mm,
  because the partial-volume profile carries sub-slice information.
- **Two unit-test failures were my expectations, not the code**: a flat background is a plateau of
  local maxima (so a short peak list gets padded with zero-response peaks — harmless, they sort last
  and fall far below any usable threshold), and the noiseless-detect expectation had the same
  MAD-collapse problem as above. Both are now documented where the behaviour lives.
- **Verdict rule rewritten once.** My first version keyed only on the patch comparison, which labelled
  *every* stratum "featureless" including ≥40 mm lesions that are found 62% of the time. Split it
  into two axes — "is there a response above the phantom null floor" and "can it be separated from
  same-size patches" — and ≥40 mm correctly becomes clutter-dominated while ≤20 mm stays featureless.
- **`cat src/data/fold_*_ids.txt` merges IDs across files** (no trailing newlines) and silently gave
  1304 instead of 1308. Used `tumorlib.io.case_ids()` instead. This is CONTEXT gotcha territory and
  cost a restarted download.

## Data fetched (flagging: this changed the dataset on disk)

Step 6 needed vessel/bowel masks, and **all of them were un-pulled git-LFS stubs** cohort-wide —
only pancreas, lesion, pancreatic_duct and common_bile_duct had ever been pulled. I ran
`git lfs pull -I <1308 cohort paths>` once per structure for duodenum, aorta, veins, superior
mesenteric artery, IVC, celiac artery, stomach and gall bladder: **~2 GB, 270 GB free afterwards**,
~4 min total. One pull per structure keeps each include string under the 131072-byte limit (gotcha
7); the longest was 95483 bytes. Nothing was reset or committed in that repo. Without this, step 6
would have been duct-only and the "clutter is vessels and bowel" finding unavailable.

## Cross-session regressions (all passed)

- Search-region coverage reproduces S2's independently computed `in_sr_frac` to max |diff| 1.6e-3,
  mean 1.4e-6 — different machine, different mask root.
- Lesion volumes recomputed from the masks match `master_lesions.csv` to 4.7e-6 relative (that file
  is stored at 6 significant figures, which is the tightest check available).
- The 11 cases with an empty pancreas envelope are exactly S2's list.
- The nearest-neighbour label map onto the 1 mm grid preserves lesion volume (median ratio 1.0000).
- My Wilson interval reproduces S3's published "venous ≥10 mm 30.8% (CI 27.0-34.8%)" exactly.

## Surprises

1. **The phantom's noise model had the wrong sign** relative to real data (MAD vs σ). A phantom can
   validate a formula and still mislead about which regime you are in.
2. **The shape term points the wrong way.** In-lesion peaks are *less* blob-like than false ones
   (median blobness 0.086 vs 0.123); the phantom-frozen gate keeps only 6.9% of true in-lesion peaks.
   The most obvious clutter suppressor suppresses tumors harder.
3. **The negatives understate false peaks by ~1/3**, and reweighting for slice thickness barely
   changes it (z ≥ 6: 3.60 → 3.82 per case). The thickness confound is real but the (a)/(b) gap is
   mostly *not* thickness — a pancreas containing a tumor is a more cluttered organ. **Worth adopting
   source (b) as the default FP denominator in later sessions, not just this one.**
4. **Thicker slices give more false peaks, not fewer** (z ≥ 6, source b: 2.90 at ≤1.5 mm vs 4.44 at
   2.5-5 mm).
5. **Contrast class hardly matters below 20 mm.** Iso within ±5 HU is no worse than hypo/hyper, whose
   median CNR is −2.33. The limit at these sizes is not contrast — there is no blob-shaped response
   above the noise floor to threshold.

## What this closes, and what it does not

Closed: a plain scale-normalized LoG/Hessian blob map, pancreas-restricted, is not a viable candidate
generator for 10-20 mm lesions on this cohort. Per CONTEXT's framing ("the paper only exists if the
blob map finds small tumors that nnU-Net misses at EVERY nnU-Net threshold"), this is the negative
result, and on the `nnU-Net missed ∩ TARGET` subset (n=74) it is 12.2% hit / 0% at rank 1 — the blob
map does not find what nnU-Net misses.

Not closed, and not attempted here (no attention or model injection was built, as instructed):
whether a **differently normalized** scale-space ranking is worth one more pass. The honest version
would fix the ranking (raw response, or MAD measured per scale on a *clutter-free* reference), select
σ out-of-fold, and re-run sections 2-4. Section 2b bounds the upside at ~22% top-10 for the target
group, so my recommendation is that this is not worth a session unless the goal shifts from
"standalone candidate generator" to "one weak feature among several".
