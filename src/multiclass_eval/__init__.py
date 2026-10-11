"""S8: Dataset502 (multi-class pancreas + lesion) vs Dataset501 (binary lesion), identical protocol, lesion
channel only.

Import with `src/` on sys.path (pytest.ini does this; scripts run as
`PYTHONPATH=src python -m multiclass_eval.<module>`). Modules:

  core        pure per-case evaluation: lesion mask from either model's saved segmentation, the region-npz
              consistency scan, prepare_excluded_eval's ignore-region semantics in nibabel order, and
              evaluate_case(), which runs S2's analyze() (threshold ladder) and evaluate.py's own metric
              functions on one case. No IO.
  case_pass   server: one pickle per (model, case) from the real OOF exports (resumable, multiprocessing)
  stats       paired case-level bootstrap (pure)
  summarize   PC: regression asserts, deliverables in results/nnunet_multiclass/
"""
