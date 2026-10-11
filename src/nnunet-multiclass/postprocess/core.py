"""S13 core: cheap post-processing of Dataset502's lesion channel. Pure functions, no IO, no tuning.

A case is reduced once (build_case) to a small table of "atoms", and every grid setting is then scored by arithmetic on
those atoms (score_grid, evaluate). The reduction is exact because:
  - kept components are distinct 26-connected components of p > t_low, so they are never 26-adjacent, and each one's
    mask is a subset of it. The connected components (CCs) of the final mask are therefore exactly the union of the
    CCs of each kept component's own mask;
  - evaluate.py's lesion Dice is Dice(L, union of predicted CCs that overlap L), its case Dice is Dice(all GT, mask),
    and an FP component is a predicted CC overlapping no GT lesion. All three are sums over CCs.
tests/test_postprocess.py checks this against a dense reference implementation + evaluate.py on synthetic volumes, and
the real-data run asserts that the no-op setting reproduces S8's 502 numbers case by case.

Settings (fixed before any real-data run):
  A  filter      keep a component iff |comp ∩ dilate(predicted pancreas, d mm)| >= 1 voxel ('any') or
                 >= 50% of the component ('half'); predicted pancreas = npz channel 0 > 0.5; dilation = Euclidean
                 distance in mm <= d (voxel centres, affine spacing). None = no filter.
  B  components  26-connected components of p > t_low over the whole volume; rank by peak p (desc), ties by size
                 (desc) then the peak voxel's flat index (asc); keep the first, then the next ones while peak >= p_min,
                 up to k in total (k None = no cap).
  D  mask        per kept component: voxels with p > t_seg inside it; if none, voxels with p >= peak / 2 inside it.
                 B's fixed rule is D with t_seg = 0.5.
  require_core   D-alone family: keep exactly the components that contain a voxel > 0.5 (the saved segmentation's own
                 objects, regrown to t_seg inside their t_low footprint); k and p_min are then unused.
The no-op (Setting()) is: no filter, t_low 0.5, keep all, t_seg 0.5 = the saved 502 segmentation (seg == 2).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np
from scipy.ndimage import distance_transform_edt, label as cc_label

CC_STRUCT = np.ones((3, 3, 3), bool)  # 26-connectivity, = tumorlib.lesions.CC_STRUCT
FLOOR = 1e-4
T_LOWS = (0.5, 0.1, 0.01, 0.001, 1e-4)
T_SEGS = (0.5, 0.3, 0.2, 0.1)
D_MM = (0.0, 5.0, 10.0)
F_RULES = ("any", "half")
KS = (None, 1, 2, 3, 5)
P_MINS = (None, 0.1, 0.3)
THRS = tuple(sorted(set(T_LOWS) | set(T_SEGS), reverse=True))


@dataclass(frozen=True)
class Setting:
    filt: tuple[float, str] | None = None
    t_low: float = 0.5
    k: int | None = None
    p_min: float | None = None
    t_seg: float = 0.5
    require_core: bool = False

    def label(self) -> str:
        f = "none" if self.filt is None else f"d{self.filt[0]:g}mm/{self.filt[1]}"
        if self.require_core:
            sel = "core>0.5"
        else:
            sel = f"k={'all' if self.k is None else self.k},pmin={'none' if self.p_min is None else f'{self.p_min:g}'}"
        return f"A={f} tlow={self.t_low:g} {sel} tseg={self.t_seg:g}"


BASELINE = Setting()
FILTERS = (None, *[(d, f) for d in D_MM for f in F_RULES])
SELECTIONS = [(None, None)] + [(k, pm) for k in KS if k is not None for pm in (P_MINS if k > 1 else (None,))] \
    + [(None, pm) for pm in P_MINS if pm is not None]


def family(name: str) -> list[Setting]:
    """Grid of one family, in a fixed order (the first maximum wins in tuning, so the no-op comes first)."""
    b_sel = [Setting(t_low=t, k=k, p_min=pm) for t in T_LOWS for k, pm in SELECTIONS]
    out = {
        "baseline": [BASELINE],
        "A": [Setting(filt=f) for f in FILTERS],
        "B": b_sel,
        # t_low = 0.5 adds nothing to D (every footprint voxel is already > 0.5), so its rows are the no-op
        "D": [BASELINE] + [Setting(t_low=t, t_seg=s, require_core=True) for t in T_LOWS[1:] for s in T_SEGS],
        "BD": [replace(s, t_seg=ts) for s in b_sel for ts in T_SEGS],
        "AB": [replace(s, filt=f) for f in FILTERS for s in b_sel],
        "ABD": [replace(s, filt=f, t_seg=ts) for f in FILTERS for s in b_sel for ts in T_SEGS],
        # added after the first results: C as specified cannot express D's core-only selection, so its grid is not a
        # superset of D's. ABDc = ABD + (every filter x D's core-only selection); still tuned leave-one-fold-out.
        "ABDc": [replace(s, filt=f, t_seg=ts) for f in FILTERS for s in b_sel for ts in T_SEGS]
                + [Setting(filt=f, t_low=t, t_seg=ts, require_core=True) for f in FILTERS for t in T_LOWS[1:] for ts in T_SEGS],
    }[name]
    assert len(set(out)) == len(out), name
    return out


# ------------------------------------------------------------------ per-case reduction
def _first_per_label(lab: np.ndarray, *keys) -> tuple[np.ndarray, np.ndarray]:
    """For labels > 0: (labels, position of the first element per label after lexsort by (lab, *keys))."""
    order = np.lexsort(tuple(reversed((lab, *keys))))
    ls = lab[order]
    first = np.r_[True, ls[1:] != ls[:-1]]
    keep = first & (ls > 0)
    return ls[keep], order[keep]


def _cc_records(lab_sparse: np.ndarray, gl: np.ndarray, n: int) -> tuple[np.ndarray, list[dict]]:
    """Sizes and lesion intersections of CCs 1..n given each sparse voxel's CC label and GT lesion label."""
    size = np.bincount(lab_sparse, minlength=n + 1)
    m = (lab_sparse > 0) & (gl > 0)
    inter = [dict() for _ in range(n + 1)]
    if m.any():
        pairs, cnt = np.unique(np.stack([lab_sparse[m], gl[m]], 1), axis=0, return_counts=True)
        for (c, l), k in zip(pairs, cnt):
            inter[int(c)][int(l)] = int(k)
    return size, inter


def build_case(shape, sp, p_idx, p_val, panc_idx, gt_idx, gt_lab) -> dict:
    """Reduce one case to its atoms. All index arrays are flat C-order indices into the (x, y, z) volume.
    p_idx/p_val must hold every voxel with p > FLOOR (more is fine, fewer is not)."""
    shape = tuple(int(s) for s in shape)
    sp = np.asarray(sp, float)
    gt_lab = np.asarray(gt_lab, np.int64)
    n_les = int(gt_lab.max()) if gt_lab.size else 0
    les_vox = np.bincount(gt_lab, minlength=n_les + 1)[1:]
    out = dict(shape=shape, sp=tuple(sp), vox_mm3=float(np.prod(sp)), n_lesions=n_les, les_vox=les_vox,
               gt_vox=int(gt_lab.size), comps={}, atoms={})
    keep = p_val > FLOOR
    p_idx, p_val = np.asarray(p_idx, np.int64)[keep], np.asarray(p_val, np.float32)[keep]
    if p_idx.size == 0:
        for t in T_LOWS:
            out["comps"][t] = dict(n_vox=np.zeros(0, int), peak=np.zeros(0), peak_flat=np.zeros(0, int),
                                   panc_n={d: np.zeros(0, int) for d in D_MM}, has_core=np.zeros(0, bool),
                                   touch=[])
            for s in T_SEGS:
                out["atoms"][(t, s)] = []
        out["n_sparse"] = 0
        out["dist"] = np.zeros(0)
        return out
    xyz = np.stack(np.unravel_index(p_idx, shape), 1)
    lo, hi = xyz.min(0), xyz.max(0) + 1
    cxyz = tuple((xyz - lo).T)
    crop_shape = tuple(int(v) for v in hi - lo)
    pc = np.zeros(crop_shape, np.float32)
    pc[cxyz] = p_val

    # GT lesion label of each sparse voxel
    gl = np.zeros(p_idx.size, np.int64)
    if gt_idx is not None and len(gt_idx):
        o = np.argsort(gt_idx)
        gs = np.asarray(gt_idx, np.int64)[o]
        pos = np.searchsorted(gs, p_idx)
        pos_c = np.minimum(pos, gs.size - 1)
        hit = gs[pos_c] == p_idx
        gl[hit] = gt_lab[o][pos_c[hit]]

    # distance (mm) from each sparse voxel to the nearest predicted-pancreas voxel; exact up to max(D_MM)
    dist = np.full(p_idx.size, np.inf)
    if panc_idx is not None and len(panc_idx):
        pxyz = np.stack(np.unravel_index(np.asarray(panc_idx, np.int64), shape), 1)
        pad = np.ceil(max(D_MM) / sp).astype(int) + 1
        blo, bhi = np.maximum(pxyz.min(0) - pad, 0), np.minimum(pxyz.max(0) + 1 + pad, shape)
        box = np.ones(tuple(int(v) for v in bhi - blo), bool)
        box[tuple((pxyz - blo).T)] = False
        dt = distance_transform_edt(box, sampling=sp)
        inb = np.all((xyz >= blo) & (xyz < bhi), axis=1)
        dist[inb] = dt[tuple((xyz[inb] - blo).T)]
        del box, dt
    out["dist"] = dist  # kept for tests only (popped by the case pass)

    # CC labels of p > thr for every threshold, as per-sparse-voxel vectors
    L, N, CCS = {}, {}, {}
    for thr in THRS:
        lab, n = cc_label(pc > thr, structure=CC_STRUCT)
        L[thr] = lab[cxyz].astype(np.int64)
        N[thr] = n
        CCS[thr] = _cc_records(L[thr], gl, n)
        del lab
    del pc

    for t in T_LOWS:
        lab, n = L[t], N[t]
        labs, first = _first_per_label(lab, -p_val, p_idx)
        assert np.array_equal(labs, np.arange(1, n + 1))
        peak = p_val[first].astype(float)
        nv = np.bincount(lab, minlength=n + 1)[1:]
        o = np.argsort(lab, kind="stable")
        members = np.split(o, np.cumsum(np.bincount(lab, minlength=n + 1)))[:n + 1]  # members[c]: sparse idx of comp c
        out["comps"][t] = dict(
            n_vox=nv, peak=peak, peak_flat=p_idx[first],
            panc_n={d: np.bincount(lab[(lab > 0) & (dist <= d)], minlength=n + 1)[1:] for d in D_MM},
            has_core=peak > 0.5,
            touch=[set(CCS[t][1][c]) for c in range(1, n + 1)])
        for s in T_SEGS:
            thr = max(t, s)
            size, inter = CCS[thr]
            # parent component (at t_low) of every CC at thr: any of its voxels
            ccl, cfirst = _first_per_label(L[thr])
            parent = lab[cfirst]
            by_comp = [[] for _ in range(n + 1)]
            for c, pa in zip(ccl, parent):
                by_comp[int(pa)].append((int(size[c]), inter[c]))
            atoms = []
            for c in range(1, n + 1):
                if by_comp[c]:
                    atoms.append(by_comp[c])
                    continue
                # fallback: no voxel > t_seg in this component -> voxels >= peak / 2 inside it
                v = members[c]
                v = v[p_val[v] >= peak[c - 1] / 2]
                sub = xyz[v]
                slo = sub.min(0)
                box = np.zeros(tuple(int(v) for v in sub.max(0) - slo + 1), bool)
                box[tuple((sub - slo).T)] = True
                bl, bn = cc_label(box, structure=CC_STRUCT)
                bsz, binter = _cc_records(bl[tuple((sub - slo).T)].astype(np.int64), gl[v], bn)
                atoms.append([(int(bsz[j]), binter[j]) for j in range(1, bn + 1)])
            out["atoms"][(t, s)] = atoms
    out["n_sparse"] = int(p_idx.size)
    return out


# ------------------------------------------------------------------ selection
def filter_pass(comps: dict, filt) -> np.ndarray:
    if filt is None:
        return np.ones(comps["n_vox"].size, bool)
    d, f = filt
    pn = comps["panc_n"][d]
    return pn >= 1 if f == "any" else 2 * pn >= comps["n_vox"]


def ranked(comps: dict, filt) -> np.ndarray:
    """Surviving component indices (0-based), best first."""
    ok = np.nonzero(filter_pass(comps, filt))[0]
    o = np.lexsort((comps["peak_flat"][ok], -comps["n_vox"][ok], -comps["peak"][ok]))
    return ok[o]


def n_kept(peaks_ranked: np.ndarray, k, p_min) -> int:
    """Length of the kept prefix: the first, then the following while peak >= p_min, up to k."""
    n = peaks_ranked.size
    if n == 0:
        return 0
    m = n if k is None else min(k, n)
    if p_min is not None and m > 1:
        below = np.nonzero(peaks_ranked[1:m] < p_min)[0]
        if below.size:
            m = 1 + int(below[0])
    return m


def select(case: dict, s: Setting) -> np.ndarray:
    comps = case["comps"][s.t_low]
    r = ranked(comps, s.filt)
    if s.require_core:
        return r[comps["has_core"][r]]
    return r[:n_kept(comps["peak"][r], s.k, s.p_min)]


def kept_ccs(case: dict, s: Setting) -> list[tuple[int, dict]]:
    atoms = case["atoms"][(s.t_low, s.t_seg)]
    return [cc for c in select(case, s) for cc in atoms[c]]


# ------------------------------------------------------------------ scoring
def objective_value(case: dict, pred_vox: int, inter_vox: int) -> float:
    """Per-case lesion Dice: tumor+ 2|P∩G| / (|P| + |G|) (0 if nothing predicted); tumor- 1 if nothing predicted."""
    if case["gt_vox"] == 0:
        return 1.0 if pred_vox == 0 else 0.0
    return 2.0 * inter_vox / (case["gt_vox"] + pred_vox)


def evaluate(case: dict, s: Setting) -> tuple[dict, list[dict]]:
    """Case row and per-lesion rows (lesion_id = tumorlib numbering) for one setting."""
    ccs = kept_ccs(case, s)
    pred = sum(sz for sz, _ in ccs)
    inter = sum(sum(d.values()) for _, d in ccs)
    n_fp = sum(1 for _, d in ccs if not d)
    row = dict(objective=objective_value(case, pred, inter), pred_vox=pred, inter_vox=inter, n_cc=len(ccs),
               n_fp_cc=n_fp, seg_any=pred > 0, localized=inter > 0,
               dice=(2.0 * inter / (case["gt_vox"] + pred)) if case["gt_vox"] else np.nan,
               pred_vol_mm3=pred * case["vox_mm3"], ref_vol_mm3=case["gt_vox"] * case["vox_mm3"])
    les = []
    for j in range(1, case["n_lesions"] + 1):
        lv = int(case["les_vox"][j - 1])
        tc = [(sz, d) for sz, d in ccs if j in d]
        u = sum(sz for sz, _ in tc)
        i = sum(d[j] for _, d in tc)
        les.append(dict(lesion_id=j, gt_vox=lv, lesion_dice=(2.0 * i / (lv + u)) if tc else 0.0,
                        lesion_iou=(i / (lv + u - i)) if tc else 0.0, detected_any=int(i > 0),
                        n_touching_cc=len(tc), pred_vol_ratio=u / lv))
    return row, les


def score_grid(case: dict, settings: list[Setting]) -> np.ndarray:
    """Objective for every setting (fast path; equals evaluate()['objective'], checked in the tests)."""
    G = case["gt_vox"]
    cache = {}
    out = np.empty(len(settings))
    for i, s in enumerate(settings):
        key = (s.filt, s.t_low, s.t_seg, s.require_core)
        if key not in cache:
            comps = case["comps"][s.t_low]
            r = ranked(comps, s.filt)
            if s.require_core:
                r = r[comps["has_core"][r]]
            atoms = case["atoms"][(s.t_low, s.t_seg)]
            sz = np.array([sum(a for a, _ in atoms[c]) for c in r], float)
            it = np.array([sum(sum(d.values()) for _, d in atoms[c]) for c in r], float)
            cache[key] = (comps["peak"][r], np.r_[0, np.cumsum(sz)], np.r_[0, np.cumsum(it)])
        pk, csz, cit = cache[key]
        m = pk.size if s.require_core else n_kept(pk, s.k, s.p_min)
        out[i] = (1.0 if m == 0 else 0.0) if G == 0 else 2.0 * cit[m] / (G + csz[m])
    return out


def all_settings() -> list[Setting]:
    """Union of every family's grid, de-duplicated, fixed order."""
    seen, out = set(), []
    for fam in ("baseline", "A", "B", "D", "BD", "AB", "ABD", "ABDc"):
        for s in family(fam):
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out


def tie_at_cut(case: dict, s: Setting) -> bool:
    """True when the last kept and the first dropped component (by k) share the same peak p."""
    if s.require_core or s.k is None:
        return False
    comps = case["comps"][s.t_low]
    r = ranked(comps, s.filt)
    m = n_kept(comps["peak"][r], s.k, s.p_min)
    return 0 < m < r.size and comps["peak"][r[m - 1]] == comps["peak"][r[m]]

