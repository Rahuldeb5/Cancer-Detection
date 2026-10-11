"""S11: rescue check. For lesions the baseline nnU-Net misses at t = 0.5, would an existing signal
have flagged them? Pure analysis of existing outputs; no training, CPU only, PC only.

Import with `src/` on sys.path (pytest.ini does this; scripts run as
`PYTHONPATH=src .venv/bin/python -m rescue_check.<module>`). Modules:

  core          constants and operating points fixed before any result was seen, pure functions
                (Wilson CI, component ranking / touch hits, signal logic, overlap counts, gate)
  nnunet_table  --pred_source -> one per-lesion nnU-Net table + per-case FP table in work/rescue_check/
                (the only nnU-Net-dependent step; rerunning on another model is one command)
  analyze       joins master table + nnU-Net table + S9 + duct outputs, writes results/rescue_check/
  ds502_fold0   optional: Dataset501 vs Dataset502 fold-0 missed-set comparison (reads the server over ssh)
"""
