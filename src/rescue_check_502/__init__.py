"""S14: describe the lesions Dataset502 still misses, and check whether an existing signal flags them beyond chance.
PC only, CPU only. Nothing is trained; no existing script is edited (S11's rescue_check.nnunet_table is run unchanged
with --pred_source, S11 / S8b / S13 functions are imported).

Import with `src/` on sys.path (pytest.ini does this; scripts run as
`PYTHONPATH=src .venv/bin/python -m rescue_check_502.<module>`). Modules:

  core        constants fixed before any result, pure functions (placement chance by FFT, predicted-pancreas
              lesion metrics, category rules)
  export_s8   S8's 502 per-case pickles -> an S11 --pred_source directory (component tables in S2's schema)
  case_pass   per case: 502 + D (S13) lesion scores, predicted-pancreas metrics, chance rates of both signals
  analyze     joins everything, writes results/rescue_check_502/{missed_lesions.csv, README tables in summary}
"""
