"""Shared, tested building blocks for the small-tumor / blob-map work on PanTS.

Import with `src/` on sys.path (pytest.ini does this for tests; scripts can run
with `PYTHONPATH=src`). Modules:

  io        -- case lists, mask/CT loading, spacing (nibabel x,y,z order everywhere)
  lesions   -- label_lesions(): THE definition of (case_id, lesion_id); feret_mm()
  envelope  -- pancreas envelope + hole-filled search region (never reads the lesion)
  resample  -- bbox-crop-then-resample to isotropic mm (masks and CT)
"""
