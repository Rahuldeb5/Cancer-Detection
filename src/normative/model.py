"""PHASE 2: the normal model, fit on bank patches ONLY. One model per patch radius (the feature
vectors of the two radii have different lengths and describe different scales).

  1. standardize every feature (bank mean / std), PCA fit on the bank keeping 95% of the variance
  2. strata = phase group x thickness group (<=2.5 / >2.5 mm) x u tercile (tercile edges = bank
     patch u quantiles). Merge rule for strata with < 2000 bank patches (`merge_strata`):
       a. per (phase, u tercile): if either thickness cell is short, merge the two thickness cells
       b. per phase: if any stratum is still short, merge the u terciles (keeping the thickness
          split if step a left it intact everywhere in that phase, else merging all of the phase);
          if a merged (phase, thickness) cell is still short, merge the whole phase
       Phases are never merged with each other. A phase that is short even when fully merged is kept
       and reported.
  3. score A (Rahul's method): k-means with K = 16 per stratum in PCA space; each cluster keeps a
     diagonal covariance (its members' per-dimension variance, shrunk toward the stratum's variance
     with weight VAR_SHRINK_N patches so a tiny cluster cannot have a near-zero variance). Anomaly =
     the diagonal Mahalanobis distance to the nearest centroid, "nearest" measured in that same metric
     (i.e. the minimum over the stratum's 16 clusters).
  4. score B (reference): mean Euclidean distance to the k = 5 nearest bank patches of the stratum in
     PCA space.
  Patches whose phase group or u is unknown are scored against the UNION of every stratum compatible
  with what is known (min over the union's clusters for A, k-NN over the union's patches for B).

Score C (within-gland z) is a per-case transform of A or B and lives in evaluate.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from normative import common as C

PHASES = ("noncontrast", "arterial", "venous")
THICKS = ("thin", "thick")
UTERS = ("u1", "u2", "u3")


def u_tercile(u: np.ndarray, edges: tuple[float, float]) -> np.ndarray:
    out = np.full(len(u), "unknown", dtype=object)
    ok = np.isfinite(u)
    out[ok] = np.where(u[ok] < edges[0], "u1", np.where(u[ok] < edges[1], "u2", "u3"))
    return out


def merge_strata(counts: dict[tuple[str, str, str], int], min_n: int = C.MIN_STRATUM) -> dict:
    """Map every fine key (phase, thick, uter) -> merged stratum name, following the module rule.
    Merged components are written as '*'."""
    out = {}
    for ph in PHASES:
        n = lambda t, u: counts.get((ph, t, u), 0)  # noqa: E731
        # a. thickness, per u tercile
        merged_t = {u: min(n("thin", u), n("thick", u)) < min_n for u in UTERS}
        strata = {}
        for u in UTERS:
            for t in THICKS:
                strata[(t, u)] = ("*" if merged_t[u] else t, u)
        size = {}
        for k, s in strata.items():
            size[s] = size.get(s, 0) + n(*k)
        # b. u terciles, per phase
        if any(v < min_n for v in size.values()):
            if not any(merged_t.values()):
                strata = {(t, u): (t, "*") for t, u in strata}
                size = {}
                for k, s in strata.items():
                    size[s] = size.get(s, 0) + n(*k)
                if any(v < min_n for v in size.values()):
                    strata = {k: ("*", "*") for k in strata}
            else:
                strata = {k: ("*", "*") for k in strata}
        for (t, u), (mt, mu) in strata.items():
            out[(ph, t, u)] = f"{ph}|{mt}|{mu}"
    return out


@dataclass
class Stratum:
    name: str
    n: int
    centroids: np.ndarray            # (K, d)
    var: np.ndarray                  # (K, d) shrunk diagonal variances
    X: np.ndarray                    # (n, d) bank patches in PCA space (for k-NN)
    tree: cKDTree = field(repr=False, default=None)


@dataclass
class NormalModel:
    radius: float
    names: list[str]
    mean: np.ndarray
    std: np.ndarray
    pca: PCA
    u_edges: tuple[float, float]
    key_to_stratum: dict
    strata: dict[str, Stratum]
    fit_cases: list[str]
    report: dict

    # ---------------------------------------------------------- transforms
    def project(self, F: np.ndarray) -> np.ndarray:
        return self.pca.transform((F - self.mean) / self.std)

    def strata_for(self, phase: str, thick: str, u: np.ndarray) -> np.ndarray:
        """Resolved stratum NAME SET per patch, as a '+'-joined string (one name if fully known)."""
        ut = u_tercile(np.asarray(u, float), self.u_edges)
        phs = PHASES if phase not in PHASES else (phase,)
        ths = THICKS if thick not in THICKS else (thick,)
        cache = {}
        out = np.empty(len(ut), dtype=object)
        for i, x in enumerate(ut):
            if x not in cache:
                us = UTERS if x == "unknown" else (x,)
                names = {self.key_to_stratum[(p, t, uu)] for p in phs for t in ths for uu in us}
                names &= set(self.strata)          # a phase with no bank patches has no fitted stratum
                cache[x] = "+".join(sorted(names or self.strata))
            out[i] = cache[x]
        return out

    # ---------------------------------------------------------- scores
    def score(self, F: np.ndarray, phase: str, thick: str, u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(score_A, score_B, stratum label) for raw feature rows F of one case."""
        Z = self.project(np.asarray(F, np.float64))
        lab = self.strata_for(phase, thick, u)
        a = np.full(len(Z), np.nan)
        b = np.full(len(Z), np.nan)
        for s in np.unique(lab):
            m = lab == s
            members = [self.strata[n] for n in s.split("+")]
            a[m] = score_a(Z[m], members)
            b[m] = score_b(Z[m], members)
        return a, b, lab


def score_a(Z: np.ndarray, members: list[Stratum]) -> np.ndarray:
    cen = np.concatenate([s.centroids for s in members])
    var = np.concatenate([s.var for s in members])
    best = np.full(len(Z), np.inf)
    for c, v in zip(cen, var):
        d = np.sqrt((((Z - c) ** 2) / v).sum(1))
        np.minimum(best, d, out=best)
    return best


def score_b(Z: np.ndarray, members: list[Stratum], k: int = C.KNN_K) -> np.ndarray:
    ds = []
    for s in members:
        if s.tree is None:
            s.tree = cKDTree(s.X)
        d, _ = s.tree.query(Z, k=min(k, s.n))
        ds.append(np.asarray(d).reshape(len(Z), -1))
    d = np.sort(np.concatenate(ds, 1), 1)[:, :k]
    return d.mean(1)


def fit_cluster_model(X: np.ndarray, name: str, seed: int = C.SEED, k: int = C.K_MEANS) -> Stratum:
    km = KMeans(n_clusters=min(k, len(X)), n_init=4, random_state=seed).fit(X)
    v_s = X.var(0) + 1e-9
    var = np.empty_like(km.cluster_centers_)
    for j in range(len(km.cluster_centers_)):
        Xi = X[km.labels_ == j]
        nj = len(Xi)
        vj = Xi.var(0) if nj > 1 else np.zeros(X.shape[1])
        var[j] = (nj * vj + C.VAR_SHRINK_N * v_s) / (nj + C.VAR_SHRINK_N)
    return Stratum(name=name, n=len(X), centroids=km.cluster_centers_, var=var, X=X.astype(np.float64))


def fit(bank: pd.DataFrame, names: list[str], radius: float, seed: int = C.SEED) -> NormalModel:
    """bank: one row per bank patch with columns names + ['case_id', 'phase_group', 'thick_group', 'u']."""
    F = bank[names].to_numpy(np.float64)
    mean, std = F.mean(0), F.std(0)
    std[std < 1e-9] = 1.0
    pca = PCA(n_components=C.PCA_VAR, svd_solver="full", random_state=seed).fit((F - mean) / std)
    Z = pca.transform((F - mean) / std)
    u = bank.u.to_numpy(float)
    edges = tuple(float(x) for x in np.nanquantile(u, [1 / 3, 2 / 3]))
    ut = u_tercile(u, edges)
    keys = list(zip(bank.phase_group, bank.thick_group, ut))
    fit_ok = np.array([p in PHASES and t in THICKS and x in UTERS for p, t, x in keys])
    counts: dict = {}
    for k_, ok in zip(keys, fit_ok):
        if ok:
            counts[k_] = counts.get(k_, 0) + 1
    k2s = merge_strata(counts)
    lab = np.array([k2s[k_] if ok else "" for k_, ok in zip(keys, fit_ok)], dtype=object)
    strata = {}
    for s in sorted(set(lab) - {""}):
        strata[s] = fit_cluster_model(Z[lab == s], s, seed)
    rep = dict(n_bank_patches=len(bank), n_fit=int(fit_ok.sum()), n_dropped_unknown=int((~fit_ok).sum()),
               n_components=int(pca.n_components_),
               explained=float(pca.explained_variance_ratio_.sum()),
               fine_counts={"|".join(k_): v for k_, v in sorted(counts.items())},
               strata_sizes={s: v.n for s, v in strata.items()},
               short_strata=[s for s, v in strata.items() if v.n < C.MIN_STRATUM])
    return NormalModel(radius=radius, names=names, mean=mean, std=std, pca=pca, u_edges=edges,
                       key_to_stratum=k2s, strata=strata, fit_cases=sorted(bank.case_id.unique()), report=rep)


def within_case_z(s: np.ndarray) -> np.ndarray:
    """Score C: each patch's score z-scored against all OTHER patches of the same pancreas
    (leave-one-out mean and SD, computed exactly from running sums)."""
    s = np.asarray(s, float)
    n = len(s)
    if n < 3:
        return np.full(n, np.nan)
    S, Q = s.sum(), (s * s).sum()
    m = (S - s) / (n - 1)
    var = (Q - s * s - (n - 1) * m * m) / (n - 2)
    return (s - m) / np.sqrt(np.maximum(var, 1e-12))
