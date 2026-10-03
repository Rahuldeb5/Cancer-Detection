"""S9: normative ("what does a normal pancreas look like") anomaly model, pilot.

Import with `src/` on sys.path (pytest.ini does this; scripts run as
`PYTHONPATH=src .venv/bin/python -m normative.<module>`). Modules:

  common     paths, constants fixed a priori, small shared helpers (thickness bins, phase groups)
  pool       PHASE 0: normal-bank candidate pool, filters, stratified sample, LFS/tar member lists
  features   PHASE 1: pure array functions (patch grid, rotation-invariant patch features, gland
             frame u, sphere insertion). No I/O, never sees a lesion mask.
  case_pass  PHASE 1: per-case I/O -> cached feature tables in work/normative/
  model      PHASE 2: standardize + PCA + strata + k-means (score A) + k-NN (score B), fit on the bank
  evaluate   PHASES 3-4: oracles, cohort evaluation, deliverables
"""
