"""S12: estimate each scan's contrast phase from the scan itself (aorta vs portal/splenic vein HU),
without the metadata `ct phase` label. CPU only, PC only.

Import with `src/` on sys.path (scripts run as `PYTHONPATH=src .venv/bin/python -m phase.<module>`).
Modules:

  core     constants fixed a priori (2026-10-03, before any real-data run) and pure array functions:
           eroded middle-80%-slice vessel median, S3 v3 gland pool, GMM phase naming
  oracle   synthetic known-HU volumes written to disk in both PanTS mask encodings, read back with
           tumorlib.io.load_mask, measured with the same core functions (step 3; must pass first)
  measure  per-case aorta / portal-vein / gland HU for the 1308 cohort -> work/phase/measure.csv
           (gland pool regression-checked against results/attenuation_v3 for every tumor+ case)
  analyze  3-component GMM phase rule, tables, scatter PNGs -> results/phase_estimate/
"""
