# Session S15: gland shape profile pilot (in progress; started 2026-10-05)

**Status: DONE (2026-10-08). Part C ran EXPLORATORY ("tool failed its insertion check"): missed targets bulge at
chance (top-3 39% vs 41% chance); detected same-size lesions bulge above chance (68% vs 50%).**

## Plain scoreboard (final)

| # | question | answer | good / bad |
|---|---|---|---|
| 1 | Does the tool find synthetic bulges? (core, round 5) | 8 mm any length and 5 mm 10-30 mm: 100% located and top-ranked, both grids | good |
| 2 | Is it fooled by a bend? (noise-free) | 2.5 mm grid +0.004 z; iso grid **−0.498 z (gate 0.25, two-sided)** — the bend scores lower, not higher | fail as written |
| 3 | Does an inserted 20 mm bulge on a normal gland stand out? | median within-scan percentile **87.8** (gate ≥ 95) | **fail** → Part C exploratory |
| 4 | Do missed 20-40 mm targets bulge? (n = 46 evaluable) | percentile 78.8; **top-3 39% vs chance 41%** | **bad: at chance** |
| 5 | ... in the head / elsewhere | head 52% vs 51% chance; elsewhere 18% vs 23% | bad |
| 6 | ... enhanced / non-contrast; iso / not | 50% vs 41% / 22% vs 40%; 44% vs 50% / 33% vs 30% | at or near chance |
| 7 | Do detected 20-40 mm lesions bulge? (n = 455) | 96.2; top-3 68% vs 50% chance | good (but echo, see 9) |
| 8 | Control: detected ≥ 40 mm (n = 246) | 100.0; 85% vs 73% | good |
| 9 | Echo: score vs 502 lesion probability | per-case ρ median 0.285; targets ρ 0.458 with their faint 502 probability | shape partly repeats 502 |
| 10 | Case level, tumor+ vs tumor− | AUC max score 0.566 (0.530-0.604); atrophy 0.560 | weak |
| 11 | Atrophy (2 mm tail shrink on normals) | not detected (2% vs 2%) | bad |
Secondary signs of a tumor on Dataset502's predicted pancreas region (channel 0 > 0.5): a contour **bulge** where the
tumor sits, and **atrophy** (thinning) upstream of it. GT pancreas / lesion masks are used only for evaluation. No
cohort lesion has been scored.

## Scoreboard (so far)

| step | result | good / bad |
|---|---|---|
| Tool (outline → centerline → sections → rays → scores) | built, `src/glandshape/`; 14 unit tests pass | good |
| Known-answer v1 (brief's parameters) | bulge PASS (2.5 mm grid, just) / FAIL (iso); atrophy FAIL (both); bend PASS / FAIL; dent PASS | bad → design changed |
| Known-answer v2 (Rahul's changes below) | atrophy, bend, dent PASS on both grids; bulge FAIL on both, only at h 3 mm × 30/40 mm long (located 70/83% and 63/83% vs 90%) | mostly good; one gate open |
| Known-answer v3 (width, centerline origin) | atrophy, dent PASS both grids; bend PASS 2.5 mm, **FAIL iso (15%)**; bulge FAIL both grids at h 3 × 30/40 mm (resolution limit, pre-declared) **and h 5 × 40 mm (80 / 83%)** | stop (rule 3) |
| Known-answer v4 (located = middle half; 200-pair bend; noise-free bend gate) | **all CORE PASS on both grids**; no non-core fail (3 mm × 30-40 mm now located 97-100% under the middle-half rule; 4 mm rule 47-80%) | good → Part B |
| Part B normal reference (bank, 502 5-fold ensemble) | 298/300 kept (2 < 5 cm³); 249 reference / 49 held out | good |
| Part B insertion gate (20 mm sphere, median within-scan percentile ≥ 95) | **84.3 (residual primary) → FAIL**; raw 85.5; 15 mm 79.7; 30 mm 89.1 | bad |
| Part B held-out top cells, residual primary | **47% at u < 0.1 (head end cap)**; raw width 12% | bad (round-4 cap artifact, real) |
| Part B tail shrink 2 mm | flagged 2% = unmodified 2% | bad (atrophy blind at 2 mm) |
| S14 target footprints (v3, counts only) | primary 47 kept: 1 not evaluable, 15 fold-heavy, 2 with a censored width; secondary 46: 0, 8, 1 | — |
| Cohort outlines (1308) | 12 empty, 7 < 5 cm³, 5 > 250 cm³ → 24 dropped; 1284 kept (31 fragmented, flagged) | — |
| S14 targets after QC | 20-40 mm: 1/48 dropped (> 250 cm³), 2 flagged, 45 clean. 10-20 mm: 0/46 dropped, 46 clean | good |
| Aorta test (v2 band) | 5 flipped, 34 inconclusive (all but 2 between −10 and −0.5 mm), 1256 normal, 1 missing | — |
| Head→tail sign (CT affine, tail to patient left) | agrees with S10's GT-signed PC1 in 1005/1006 (PanTS_00007757 disagrees) | good |

## Part A round 1 (v1, the brief's parameters; 2026-10-05)

Every number: `work/gland_shape/partA/known_answer_v1.txt`, `diagnose_a_v1.txt`, `cohort_qc_v1.txt`.
- **Bulge** (±10 mm running median, signed z): 10 mm-long bulges 100% located and top-ranked on both grids. 20 mm-long
  bulges failed on the iso grid (top-ranked 80%). Diagnostic: bulges **30-40 mm long were located in 0-40%**: a
  21 mm median window treats a longer bulge as the gland's own width. The S15 target lesions are 20-40 mm.
- **Atrophy** (whole-side medians): flagged 100% of step tubes but peaked ~48 mm from the step. On any steadily tapering
  gland a step changes the whole-side ratio by the same amount at every position (a plateau), so it cannot localize.
- **Bend** (90°, radius 15 mm): PASS on 2.5 mm, FAIL on iso (12% vs ≤ 10%): the smoothed centerline cuts the corner, so
  the outer side read as a bulge (+0.75 z noiseless).
- **Dent** sign: PASS.

## Changed after Part A known-answer failure (synthetic only)

Decided by Rahul on 2026-10-06, before any cohort lesion was scored. v1 values remain in `src/glandshape/core.py`
marked "v1 (superseded)".
1. **Bulge window → multi-scale.** Running-median half-windows ±15, ±30, ±45 mm; at each scale
   z = residual / the gland's own MAD of residuals at that scale; **primary = max over scales of the positive z**, then
   the 3 stops × 3 directions smoothing. Bulges 30 and 40 mm long (heights 3/5/8) added to the gates with the same
   pass rule as 10 and 20 mm.
2. **"Located"** = within 4 mm along the centerline only (the brief's rule). The 45° direction check is a secondary
   number, not a gate.
3. **Atrophy**: keep the whole-side median score, case level only. New gate: step tubes flagged ≥ 90% AND smoothly
   tapering tubes without a step flagged ≤ 10% (same threshold). The ±20 mm windowed version is a secondary localizer.
   Implementation choice: threshold = 95th percentile of the 40 reference tubes' case atrophy, each scored leave-one-out.
4. **Bend**: 12% counted as a fail. Rays are now cast from the **centroid of each cross-section's gland region**
   (falls back to the centerline point if the centroid is outside the region; flagged). Bend limit unchanged (≤ 10%).
5. **Gland QC** (fixed now): drop empty outlines, volume < 5 or > 250 cm³, centerline failures (length > 1.5 × the
   outline's bounding-box diagonal, or the centerline / profile raises). Fragmented outlines are kept (largest
   component) and flagged; Part C reports targets with and without them.
6. **Aorta test**: |aorta − gland y| ≤ 10 mm = inconclusive, treated as not flipped.
7. **Head→tail sign from the CT affine** (tail toward patient left): approved.

## Part A round 2 (v2; 2026-10-06)

Criteria written into `src/glandshape/known_answer.py` before the run. Every number:
`work/gland_shape/partA/known_answer.txt`, `cohort_qc.txt`.

| test | 0.8 × 0.8 × 2.5 mm | 0.8 mm iso |
|---|---|---|
| Bulge h 5 / 8, lengths 10-40 mm | located and top-ranked 97-100% | 90-100% |
| Bulge h 3, 10 / 20 mm | 100 / 100% | 100 / 97% |
| Bulge h 3, 30 / 40 mm | **83 / 70% → FAIL** | **83 / 63% → FAIL** |
| Atrophy: step flagged / no-step flagged | 100% / 0% → PASS | 100% / 0% → PASS |
| Atrophy, ±20 mm localizer (secondary) | 100% within 10 mm | 100% |
| Bend: bent above straight P95 | 0% → PASS | 2% → PASS |
| Dent sign | 100% → PASS | 100% → PASS |

**Side effect of the centroid origin (secondary direction check).** A large bulge pulls the section centroid toward
itself (≈ 4 mm for an 8 × 20 mm bulge), which lengthens the opposite rays by the same amount. The bulge signal is
split between the two sides. The peak is at the right position along the centerline, but points at the bulge in 0%
of 30-40 mm bulges, and the median direction for 8 × 20 mm is 172° (the opposite side). On real glands the centroid
also jumps where a section region changes between stops, leaving vertical seams in the radius map
(`work/gland_shape/partA/smoke_profiles.png`). This matters for Part C, whose lesion footprint is defined by the rays
that pass through the lesion.

## Round 3: radius -> width (origin-independent), decided after round 2 synthetic results

Decided by Rahul on 2026-10-06 (synthetic only; no cohort lesion scored). Criteria written into `known_answer.py`
before the run. Every number: `work/gland_shape/partA/known_answer.txt`, `diagnose_r3.txt`, `cohort_qc.txt`,
`target_footprints.csv`, `smoke_profiles.png` (round-2 files suffixed `_v2`).

**Changes.**
1. **WIDTH is the primary measurement**: 12 axes 15° apart, w = r(θ) + r(θ + 180°), first exit, 40 mm cap per
   half-ray. Ray origin back to the centerline point; if that point is outside its plane's gland, the nearest inside
   sample (flagged; the smoothed-centroid idea was dropped, not implemented). bulge_local = the round-2 multi-scale
   rule on width (3 stops × 3 axes smoothing); bulge_global on width with a (u, axis) reference. The 24-direction
   radius map is kept as a secondary output (which side), never a score. Area / atrophy unchanged.
2. Gates unchanged from round 2 (3 mm bulges at every length; located = within 4 mm along the centerline).
   Secondary: the top-ranked axis within 45° of the bulge direction.
3. Rule fixed before the run: if only the 3 mm × 30-40 mm cells fail, record them as the tool's resolution limit (a
   long 3 mm bulge against ±1 mm lobulation), report them as FAILs, and proceed to Part B. Anything else → stop.
4. **Part C footprint** (fixed, `src/glandshape/footprint.py`, not run on scores): the (stop, axis) cells where
   either half of that axis's diameter passes through the GT lesion; chance = footprint cells / scoreable cells,
   same cell definition. No footprint cell → not evaluable.
5. Fold-over stops are kept in the profile and the scoring, and flagged. Lesions with > half of their footprint
   stops fold-over ("fold-heavy") will be reported with and without.
6. A width with a capped half-ray is censored (flagged).

**Known-answer results (v3).**

| test | 0.8 × 0.8 × 2.5 mm | 0.8 mm iso |
|---|---|---|
| Bulge h 3: 10 / 20 / 30 / 40 mm long, located | 100 / 100 / **80 / 63%** | 100 / 100 / **70 / 53%** |
| Bulge h 5: 10 / 20 / 30 / 40 mm | 100 / 100 / 100 / **80%** | 100 / 100 / 100 / **83%** |
| Bulge h 8: 10-40 mm | 100% | 100% |
| Atrophy: step / no-step flagged | 100 / 3% → PASS | 100 / 0% → PASS |
| Bend: bent above straight P95 | 0% → PASS | **15% → FAIL** |
| Dent sign | 100% → PASS | 100% → PASS |

Top-ranked equals located in every cell except iso h 3 × 40 (47%). Secondary axis check: 100% within 45° for
10-20 mm bulges; for long bulges it drops (h 8 × 40 mm: 0%, median 82°).

**Rule 3 outcome: STOP.** Besides the pre-declared resolution-limit cells (h 3 × 30/40, recorded as FAILs), bulge
h 5 × 40 mm fails on both grids and the bend fails on the iso grid.

**Diagnostics (`diagnose_r3.py`, read-only on the saved run; nothing changed).**
- *Long bulges sit on a flat crest.* In every h 5 × 40 mm "miss", the top cell is still within the middle half of the
  bulge (|ds| ≤ 10 mm: 100% on both grids; worst 6.9 / 7.3 mm). The phantom's crest 4-8 mm from the centre is only
  0.1-0.4 mm below the apex, under the ±1 mm lobulation, so the 4 mm rule here tests where noise puts the peak
  along the crest rather than detection. Same for h 3 × 30 on 2.5 mm (100% within the middle half). True misses
  exist only at h 3 × 40 on iso (13% outside the bulge; worst 81 mm away).
- *Bend.* Noiseless bent vs straight 2.00 vs 1.98 (iso); paired median difference +0.09 z. Under a paired
  label-swap null (no bend effect), this gate's statistic exceeds 10% in 17% (iso) / 22% (2.5 mm) of draws;
  the observed 15% has p = 0.115. Not evidence of a bend artifact, but the fixed gate fails.

**Cohort counts (v3; no scores).** 1284 glands kept (unchanged QC). Censored widths in 197/1284 glands.
Origin fallbacks: 529 stops in 514 glands, shift median 0.0 mm, P95 0.5 mm (mostly at the extended tips).
- S14 primary target, 47 remaining: 1 not evaluable (no footprint cell), **15 fold-heavy**, 2 with a censored width;
  footprint stops median 10, footprint fraction of the map median 0.081.
- S14 secondary, 46: 0 not evaluable, **8 fold-heavy**, 1 with a censored width; footprint stops median 7, fraction
  0.041.

**Smoke maps (item 7).** Same 6 negatives (they already include PanTS_00007401 and PanTS_00000708).
- Width removes the v2 mirror-splitting: PanTS_00001524's two symmetric radius patches (100° and 280°) become one
  width patch on axis 100°.
- **The vertical seams are NOT gone**: PanTS_00007401 at s ≈ 68 mm and PanTS_00000708 at s ≈ 65 mm show in the width
  map too. They are real anatomy measured over 3-5 stops, not origin artifacts: 7401 is a neck pinch (area 524 →
  296 → 446 mm² over s 58-78 mm); 708 is the head-to-body step (966 → 350 mm² over ~12 mm, where the fold-over
  stops end). The ±30/45 mm scales may read the head side of such a step as an outward bulge — a Part C risk at
  the head end.

## Round 4: located rule, bend rerun, head-body step, residual primary (decided after round 3; 2026-10-06)

Decided by Rahul (synthetic + normal-gland checks only; no cohort lesion scored). Criteria written into
`known_answer.py` before the run; outputs `work/gland_shape/partA/known_answer.txt` (round-3 files now `*_v3`).
Pre-committed: after this round go to Part B unless a CORE test fails (8 mm bulges any length; 5 mm bulges
10-30 mm; atrophy step / no-step; dent sign; noise-free bend). Other cells are reported and do not block.

**Changes.**
1. **Located rule changed after round 3 to match Part C's footprint question**: a bulge is located if the
   top-scoring cell lies within the middle half of the bulge along the centerline (|s − s0| ≤ length / 4). The old
   4 mm rule is reported next to it. 3 mm × 30-40 mm stays a reported resolution-limit cell.
2. Bend: one rerun with 200 new pairs (lobulation seeds 5000-5199), same ≤ 10% limit; plus a deterministic gate:
   noise-free bent vs straight median score in the bend window differ by < 0.25 z on both grids.
3. New report-only synthetic test: head-body step (width 30 → 14 mm over 12 mm at 0.3 L), without a bulge and with
   a 5 × 20 mm bulge on the head side, body side, or at the step; raw width and width minus a 30-tube step reference.
4. **Design change for Part B onward**: primary score = multi-scale bulge_local on the RESIDUAL width (width − the
   250 reference bank glands' median width at the same u bin 0.05 and axis); raw-width bulge_local is secondary.
5. Part C definitions written into `src/glandshape/partc.py` (not run): within-scan percentile with footprint-bbox
   windows (axes wrap, stops do not; windows overlapping the footprint excluded; ties half); top-3 peaks after
   non-max suppression (≥ 10 mm OR ≥ 45° apart); top-3 chance = max(uniform exact placement, empirical from the 50
   held-out normal glands mapped by u). Unit-tested (exact uniform chance vs Monte Carlo).
6. Part B addition: held-out normal glands' top-cell u / axis distribution under both scores.

**Results (both grids).**

| test | 0.8 × 0.8 × 2.5 mm | 0.8 mm iso |
|---|---|---|
| CORE bulges (h 8 all lengths; h 5 at 10-30 mm), located / top | 100 / 100% every cell | 100 / 100% every cell |
| h 5 × 40 (non-core) | 100 / 100% (4 mm rule: 80%) | 100 / 100% (4 mm: 83%) |
| h 3 × 10 / 20 (non-core) | 100 / 100% | 100 / 100% |
| h 3 × 30 / 40 (resolution-limit cells) | 100 / 97% located (4 mm: 80 / 63%) | 100 / 100% (4 mm: 70 / 53%) |
| CORE atrophy: step / no-step flagged | 100 / 3% | 100 / 0% |
| non-core bend, 200 pairs: bent above P95 | 0.0% | 3.0% |
| CORE noise-free bend: median diff | −0.027 z | **−0.246 z (limit 0.25; bent scores lower)** |
| CORE dent sign | 100% | 100% |

Under the middle-half rule the resolution-limit cells pass; they are kept labelled as such, per the decision.

**Head-body step (report only).**
- Raw width: with no bulge, the top cell sits on the head side next to the step (73-90% within 6-30 mm headward);
  a bulge AT the step is never located (0%). Head-side and body-side bulges are located 100%.
- Residual (step reference): the step itself disappears (bulge at the step located 97%, top 93-97%), but with no
  bulge the top cell moves to the far head end (90-93% more than 30 mm headward; median −47 mm ≈ the head tip),
  and head-side bulges are top-ranked only 47-63%.

**Glaring issue found (reported, not acted on): the residual score has a false peak at the head end cap.**
Re-reading the saved run with the residual score on ordinary tubes (synthetic normal reference): short bulges lose
top rank (h 5 × 10: 47-73% vs raw 100%; h 3 × 10: 10-13%), and **every such miss puts the top cell at u 0.03-0.04**.
Cause: the first u bin (0-0.05, ~7-9 mm) spans the rounded end cap, where width rises from 0 to ~24 mm; the bin
median (17.6 mm) is far below the cap's later stops, so they read as a +6 mm residual. Elsewhere adjacent bins differ
by only ~0.67 mm. Real glands also end in a cap, so the same artifact is expected in Part B (item 6 will show it).

## Part B: normal reference from the S9 bank (2026-10-06/07)

Every number: `work/gland_shape/partB/{bank_phase.txt, bank_profile.txt, diagnose_b.txt}`; tables `bank_pass1.csv`,
`bank_split.csv`, `insertion.csv`, `heldout_top_cells.csv`, `heldout_peaks.json` (Part C empirical chance).
Profiles and predictions outside the repo: `~/research/gland_profiles/{bank_pred, bank, bank_reference.npz}`.

1. **Phase (S12 estimator, unchanged, `bank_phase.py`).** The cohort refit reproduces S12's labels for 1291/1291
   measurable cohort cases. Bank: venous 152, non-contrast 63, late arterial 46, early arterial 28, uncertain 8,
   unmeasurable 3 (cohort, carried over: venous 54%, non-contrast 21%). Metadata "Arterial" bank scans: 17 venous,
   11 non-contrast by the estimate. Bank median slice 2.5 mm.
2. **Inference (server, GPUs 0,1).** 300 CTs copied (9.3 GB; 396 GB free before). Dataset502 5-fold ensemble,
   default settings (mirroring TTA, as the cohort's OOF predictions), `bank_predict.sh`, ~7.4 h (20:23 → 03:46).
   `bank_extract.py` keeps channel 0 (pancreas region) as float16 crops (353 MB for 300); the full nnU-Net npz files
   are still on the server in `~/research/gland_shape_bank/pred/` (delete when agreed).
3. **Split and QC.** rng(15) permutation before QC → 250 reference / 50 held out. Dropped: 2 (outline < 5 cm³; one
   per group). Kept 249 / 49; fragmented-flagged 8 / 1. Aorta: 288 normal, 10 inconclusive, 0 flipped. Kept glands:
   median length 181 mm; fold flags in 251/298; censored widths in 44/298. Reference: all 240 (u bin, axis) cells
   filled; median width by u bin 15 24 27 29 27 23 20 20 21 22 22 22 23 23 22 22 21 20 19 13 mm; atrophy LOO P95
   threshold |z| 4.55.
4. **Held-out top cells (item 6).** Residual primary: u deciles [23, 3, 9, 4, 1, 0, 1, 0, 1, 7] → **47% of top
   cells at u < 0.1 (the head end cap)**, 14% at u > 0.9. Raw width: [6, 17, 8, 5, 2, 1, 1, 1, 4, 4] (12% < 0.1).
   No pile-up at the head-body transition itself: u 0.2-0.45 holds 29% of top cells under both scores vs 25% of the
   scoreable cells. This is the round-4 head-cap artifact, now confirmed on real normal glands.
5. **Insertion (item 9).** One surface point per held-out gland (rng(1500 + index); u 0.10-0.89, median 0.46).
   | sphere | primary (residual) percentile median (IQR) | raw | top-3 hit primary / raw | uniform chance |
   |---|---|---|---|---|
   | 15 mm | 79.7 (60.5-91.6) | 84.0 | 35% / 57% | 13.4% |
   | 20 mm | **84.3 (71.9-92.7)** | 85.5 | 67% / 84% | 25.0% |
   | 30 mm | 89.1 (76.9-96.9) | 100.0 | 80% / 94% | 40.9% |
   n = 48 / 48 / 47 with a percentile (1-2 glands have no window that avoids the footprint).
   **Gate (20 mm median ≥ 95): FAIL (84.3).**
   *Diagnostic (`diagnose_b.py`, read-only; reproduces 84.3 / 85.5 exactly).* Windows are the footprint's box:
   median 11 stops × all 12 axes, ~64 window positions per gland. Of the windows that beat the sphere, 318/624
   (residual) and 250/587 (raw) touch an end cap (u < 0.1 or > 0.9). Counterfactual only, not a result: without
   end-cap windows the medians would be 94.3 (residual) and 95.4 (raw). The residual score also lowers the sphere's
   own score (median z 4.36 vs 5.45 raw).
6. **Tail shrink 2 mm.** Flagged (|z| > 4.55, z < 0) 2% vs 2% of the unmodified glands; median percentile of the
   shrunk case among unmodified 63. Case-level atrophy does not see a 2 mm tail shrink (~5100 voxels removed).

## Round 5: raw-width primary, tip stops excluded, fixed outcome rule (decided 2026-10-08, BEFORE any run)

Decided by Rahul after Part B (insertion used normal glands only; no cohort lesion scored). Written here and into
the code (`core.py`, `known_answer.py --core`, `bank_round5.py`, `partc_run.py`) before running.
1. **Primary score = multi-scale bulge_local on RAW width** (±15/±30/±45 mm, as round 3). The residual score (width
   minus the normal median) becomes secondary, reported only. Reason: on the 49 held-out normal glands it was worse
   than raw width on every insertion measure and put 47% of top cells at u < 0.1.
2. **Extrapolated tip stops are excluded** from scoring AND from the comparison windows: stops on the centerline
   extension past the skeleton path's ends (recorded per case; head and tail lengths swap when the curve is
   reversed to run head → tail). Implementation: their width is set to NaN before any score, so they are neither
   scored nor in any window, footprint or chance. No other stop is excluded by u; no tip-distance indexing.
3. Rerun once, same seeds, same held-out 49: Part A core tests (8 mm bulges any length, 5 mm bulges 10-30 mm, dent
   sign, noise-free bend; both grids) and the Part B insertion (15/20/30 mm, same surface points) and held-out
   top-cell checks.
4. Counts only, before Part C: how many of the 46 evaluable primary targets and 46 secondary targets lose footprint
   cells to step 2, and how many lose all of them (they become not evaluable).
5. **Outcome rule (fixed now):**
   - Core tests pass AND 20 mm insertion median ≥ 95 → run Part C with the existing gates unchanged.
   - Otherwise → run Part C anyway, labelled **EXPLORATORY ("tool failed its insertion check")** in the results
     README, these notes and the CSV. Exploratory results cannot pass a gate and are not used for the case-level
     classifier.
   - Either way, no further rule changes.
6. Atrophy: the 2 mm tail shrink was not detected (2% vs 2%); case-level atrophy stays in Part C as a reported
   number only.
Part C then runs on the fixed definitions (`partc.py`: footprint-box window percentile; top-3 after non-max
suppression; chance = larger of uniform and empirical), with and without fold-heavy and fragmented-flagged
targets, enhanced vs non-contrast, iso vs not, the 461 detected-and-included reference, and the ≥ 40 mm control.

### Round 5 results (run once; outputs `work/gland_shape/partA/known_answer_v5.txt`, `work/gland_shape/partB/bank_round5.txt`)

| check | 0.8 × 0.8 × 2.5 mm | 0.8 mm iso |
|---|---|---|
| CORE bulges h 8 (10-40 mm), h 5 (10-30 mm): located / top | 100 / 100% every cell | 100 / 100% every cell |
| CORE dent sign | 100% | 100% |
| CORE noise-free bend (|diff| < 0.25 z) | +0.004 → PASS | **−0.498 → FAIL** (bent 0.369 < straight 0.867: no false bulge, but the gate is two-sided) |

Part B rerun (same 49 held-out glands, same surface points):
- Insertion, v5 primary: 15 mm 83.4, **20 mm 87.8**, 30 mm 100.0 (Part B v4: 79.7 / 84.3 / 89.1). Top-3 hit 53 / 82 /
  94% vs uniform chance 14.5 / 24.9 / 42.6%. Residual (secondary): 81.8 / 84.8 / 91.3.
  **Insertion check (20 mm median ≥ 95): FAIL (87.8).**
- Held-out top cells, v5 primary: u < 0.1 6% (was 47% under the residual), but **41% (20/49) at u 0.1-0.2**, the head
  just past the excluded tip; head-body band 0.2-0.45 29% vs its 28% share of cells. Residual: 24% at u < 0.1.
  Extrapolated tip stops per gland: median 10.

**Outcome (rule fixed above): core FAIL (iso noise-free bend) and insertion FAIL → Part C runs EXPLORATORY ("tool
failed its insertion check"). It cannot pass a gate and is not used for the case-level classifier.**

## Part C (EXPLORATORY; 2026-10-08)

Every number: `work/gland_shape/partC/partc_summary.txt`; deliverables `results/gland_shape/{per_lesion_shape.csv,
README.md}` (both labelled EXPLORATORY); 12 QC PNGs in `work/gland_shape/partC/qc/` (4 head-heavy, 2 target misses,
2 control, 2 reference, 2 tumor−; picked by a fixed rule before drawing). Run: 309 s, 4 workers, 0 errors.
- **Glands:** 1284 kept (12 empty, 7 < 5 cm³, 5 > 250 cm³ dropped). gt_vox asserted equal to S14 for all 1235
  lesions; group sizes reproduce S14 (48 / 46 / 461 / 253).
- **Round-5 item 4 (tip exclusion):** target 10/46 lose ≥ 1 footprint cell, 0 lose all (46 evaluable); secondary
  11/46 lose cells, 1 loses all (45 evaluable).
- **Results:** see the scoreboard above and the README table. Target, no fold-heavy: 65.6 / 23% vs 33%. Target, no
  flag at all: 65.7 / 24% vs 33%. Secondary 10-20 mm: 66.7 / 27% vs 21%. Residual (secondary score): target 78.3 /
  35%; reference 92.7 / 60%; control 100 / 81%.
- **Head vs elsewhere** (footprint starts at u < 0.3): target 93.3 / 52% vs 51% chance vs 65.6 / 18% vs 23%;
  reference 100 / 79% vs 58% vs 59.7 / 42% vs 31%.
- **Echo:** per-case Spearman ρ (primary vs 502 lesion probability in the same cells) median 0.285 (IQR 0.130-0.428,
  961 tumor+ glands). Targets: lesion score vs faint 502 probability in the footprint ρ 0.458 (median 0.008).
- **Case level (reported only):** AUC max primary 0.566 (0.530-0.604); residual 0.530; atrophy 0.560; within thickness
  bins 0.54-0.59. Atrophy flags 7.0% tumor+ vs 3.6% tumor−.

### What surprised me
1. **The head is the dominant "bulge" in normal glands.** The wide head next to the narrow neck reads as outward at
   the ±30/45 mm scales; 41% of normal top cells sit at u 0.1-0.2. Chance measured on normal glands absorbs it, and
   once it does, missed targets are at chance in the head (52% vs 51%) and below it elsewhere (18% vs 23%).
2. **Detected tumors bulge, missed ones do not.** Same size bin, same gland, same tool: 68% vs 50% for detected,
   39% vs 41% for missed. 502 seems to miss exactly the tumors that do not change the gland's outline, which is
   consistent with S14 (misses are swallowed as gland: iso or unenhanced).
3. **Enhanced targets look better than non-contrast ones** (50% vs 41% chance, ratio 1.21, vs 22% vs 40%), but with
   n = 28 / 18 the CIs overlap chance.
4. **The fixed gates were strict in useful ways.** The insertion check showed, before any lesion was scored, that a
   20 mm bulge is not the most unusual shape in a normal gland often enough (87.8 vs 95) — the same head effect.

## Not done / open
- CONTEXT.md not edited. Suggested S15 block: exploratory negative; missed targets at chance; head = normal bulge;
  detected tumors bulge above chance; tip stops / width / multi-scale design notes.
- Server: the full nnU-Net bank outputs (~20 GB) are still in `~/research/gland_shape_bank/pred/`; delete on approval.
- Per the outcome rule, no classifier uses these shape features.

## Process notes
- Unit-test expectations I had wrong (code unchanged): per-ray discretization of a binary 1 mm outline (±1 mm, not
  ±0.6); a θ index can differ by one between mirrored grids; my own slice count; a degenerate U-turn phantom; HU
  oracle still cast from the old origin; aorta offsets in the inconclusive-band test.
- v1 / v2 known-answer and cohort outputs were renamed `*_v1` / `*_v2` and kept; unsuffixed files are round 3.
- Round 3 test fixes (code unchanged): wrong hole coordinates in the nearest-inside-origin test.

## Not done / open
- Process: `pgrep -f` inside an ssh command matched its own command line (same trap as S9's `pkill -f`), so a wait
  loop believed inference was still running; the extraction ran ~10 min late, no harm. pandas `df.flags` is a
  DataFrame attribute, not a column (fixed in bank_profile).
- Part B (normal bank through 502 on the server), Part C (cohort). Deliverables in `results/gland_shape/` not created.
