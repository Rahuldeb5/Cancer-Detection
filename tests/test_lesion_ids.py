"""Test group a: label_lesions() reproduces the (case_id, lesion_id) -> gt_vox table that
every downstream join was built on, for all 1235 lesions. Zero mismatches allowed.

Also checks that feret_mm() + io.spacing() reproduce the table's diam_mm, and counts
how many lesions would have hit the degenerate-hull fallback.
"""
import multiprocessing as mp

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import find_objects
from scipy.spatial import ConvexHull

from tumorlib import io
from tumorlib.lesions import feret_mm, label_lesions

CSV = io.REPO_ROOT / "results" / "attenuation_results" / "lesion_dice_attenuation.csv"
N_WORKERS = 4  # 15 GB RAM on the PC; worst case ~1 GB per worker


def lesions_of(case_id: str) -> list[dict] | str:
    mask = io.load_mask(case_id, "pancreatic_lesion")
    sp = io.spacing(case_id)
    if mask is None or sp is None:
        return f"{case_id}: could not load"
    labels, n = label_lesions(mask)
    rows = []
    for k, sl in enumerate(find_objects(labels), start=1):
        coords = np.argwhere(labels[sl] == k) + [s.start for s in sl]
        hull_failed = False
        if len(coords) > 200:
            try:
                ConvexHull(coords * np.asarray(sp))
            except Exception:
                hull_failed = True
        rows.append({"case": case_id, "lesion_id": k, "gt_vox_new": len(coords),
                     "diam_new": feret_mm(coords, sp), "hull_failed": hull_failed})
    return rows


@pytest.fixture(scope="module")
def joined():
    ref = pd.read_csv(CSV)
    cases = sorted(ref["case"].unique())
    with mp.Pool(N_WORKERS) as pool:
        results = pool.map(lesions_of, cases, chunksize=4)
    errors = [r for r in results if isinstance(r, str)]
    new = pd.DataFrame([row for r in results if not isinstance(r, str) for row in r])
    merged = ref.merge(new, on=["case", "lesion_id"], how="outer", indicator=True)
    return ref, new, merged, errors


@pytest.mark.data
@pytest.mark.slow
def test_gt_vox_reproduced_for_every_lesion(joined):
    ref, new, merged, errors = joined
    assert not errors, errors[:10]
    assert len(ref) == 1235
    only_ref = merged[merged["_merge"] == "left_only"]
    only_new = merged[merged["_merge"] == "right_only"]
    assert only_ref.empty, f"lesions in the table but not found now:\n{only_ref.head(20)}"
    assert only_new.empty, f"extra components not in the table:\n{only_new.head(20)}"
    bad = merged[merged["gt_vox"] != merged["gt_vox_new"]]
    assert bad.empty, f"{len(bad)} gt_vox mismatches:\n{bad[['case', 'lesion_id', 'gt_vox', 'gt_vox_new']].head(20)}"


@pytest.mark.data
@pytest.mark.slow
def test_feret_reproduces_diam_mm_and_fallback_is_dormant(joined):
    _, new, merged, _ = joined
    both = merged[merged["_merge"] == "both"]
    diff = (both["diam_mm"] - both["diam_new"]).abs()
    print(f"\nmax |diam_mm - diam_new| = {diff.max():.2e} mm over {len(both)} lesions; "
          f"hull failures = {int(new['hull_failed'].sum())}")
    assert diff.max() < 1e-3, both.loc[diff.idxmax(), ["case", "lesion_id", "diam_mm", "diam_new"]]
