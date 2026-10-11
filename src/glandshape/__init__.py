"""S15 gland shape profile: contour bulge and upstream atrophy on Dataset502's predicted pancreas region.

Pipeline (one case):
  outline     502 channel 0 > 0.5 -> largest 3D component -> hole fill -> crop bbox + 10 mm -> 1 mm isotropic
  centerline  3D skeleton -> longest geodesic path -> smoothing spline -> extended to the boundary -> head->tail
              -> stops every 2 mm of arc length, rotation-minimizing (parallel-transport) frame, angle 0 = anterior
  profile     perpendicular cross-sections (0.5 mm), 24 rays (first exit, max 40 mm) -> r(s, theta), A(s), HU rays
  scores      bulge_local (primary, outward), bulge_global, atrophy

GT pancreas and GT lesion masks are never read by any module here except `evaluate` code in later parts.
Modules: core (frozen constants), outline, centerline, profile, scores, orient, phantom, known_answer, qc_cohort.
"""
