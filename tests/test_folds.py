"""Test group d: every cohort case is in exactly one fold file."""
from collections import Counter

from tumorlib import io


def test_each_case_in_exactly_one_fold():
    counts = Counter(c for k in range(1, io.N_FOLDS + 1) for c in io.fold_ids(k))
    dupes = {c: n for c, n in counts.items() if n != 1}
    assert not dupes, f"cases in more than one fold (or repeated within one): {dupes}"
    assert len(counts) == 1308
    assert io.case_ids() == sorted(counts)
