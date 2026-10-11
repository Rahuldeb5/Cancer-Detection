"""S10 duct atlas: every PanTS pancreas aligned to one gland frame, pancreatic_duct masks stacked.

Modules:
  frame    -- per-case gland frame (centroid, PC1 head->tail, world-anterior / world-superior axes)
  grid     -- atlas grids (normalized gland units and physical mm) and the voxel-centre splat
  case     -- one case: load masks as voxel coordinates, build the frame, splat duct/tissue/sections
  phantom  -- synthetic ellipsoid glands with a duct, in random poses (oracle tests)
  run      -- chunked, resumable multi-worker run over all 9901 cases -> work/duct_atlas/
  plot     -- deliverable figures + per-case QC image

Run with PYTHONPATH=src (tests get it from pytest.ini).
"""
