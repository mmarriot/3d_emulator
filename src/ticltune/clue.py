"""Numpy emulation of the CMSSW CLUEstering trackster building, on layer clusters or on rechits.

The points are the layer clusters passing the CLUE3DHigh mask (`level="lc"`, what CMSSW clusters), all layer clusters
including the single-hit ones (`level="lc_all"`: the mask with min_cluster_size = 1) or the HGCAL rechits
(`level="rh"`, no 2D step); the algorithm and its parameters are the same. It follows, line by line where it
matters for the result:
  RecoHGCal/TICL/plugins/alpaka/TracksterCLUEsteringAlgoWrapper.dev.cc
      gnomonic coordinates x/|z|, y/|z| in units of the smallest sigmaT; z = layer / layerScale with CMSSW's
      layer index and float rounding (see _pairs); cylinder metric max(|dxy| * 2 / (rel_i + rel_j), |dz|);
      rhoc_i = rhoc * (pivot / r_i)^alpha; tie-break tag = the point's DetId (layer cluster: its seed);
      the two endcaps never interact.
  CLUEstering 2.12 core/detail/ClusteringKernels.hpp, FlatKernel(0.5)
      rho_i = sum_{d(i,j) <= dc} k_ij E_j with k_ii = 1 and k_ij = 0.5 otherwise; nearest higher among j with
      rho_j > rho_i (or equal, rho_j > 0, tag_j > tag_i) within seedingDistance if rho_i >= rhoc_i else
      outlierDistance; on equal distance the higher rho, then the higher tag; seed = no nearest higher and
      rho_i >= rhoc_i; a chain that ends on a non-seed is an outlier.
  RecoHGCal/TICL/plugins/PatternRecognitionbyCLUEstering.cc
      masked points are left out; tracksters with fewer than minNumLayerCluster points are dropped.

Defaults are the CMSSW configuration (RecoHGCal/TICL/python/CLUE3DHighStep_cff.py, ticl_dev). The metric is
invariant under the choice of the reference sigmaT, so the emulation is exact whichever reference CMSSW uses.
"""
from dataclasses import dataclass, replace, asdict
from typing import Tuple

import numpy as np
from scipy.spatial import cKDTree

from .data import SIGMA_INDEX, HGCAL_DETECTORS

LAST_LAYER = 47  # layers per endcap (rhtools lastLayer, D128): the + side offset of the rechit-SoA layer index


@dataclass(frozen=True)
class Params:
    dc: float = 1.0
    rhoc: float = 0.8
    outlierDistance: float = 2.8
    seedingDistance: float = 2.8
    sigmaT: Tuple[float, float, float] = (0.003, 0.006, 0.012)  # EE, HSi, HSci
    layerScale: float = 6.0
    rhocEtaExponent: float = 2.0
    rhocPivotRadius: float = 0.42
    minNumLayerCluster: int = 2     # minimum number of points of a trackster
    f32: bool = True                # coordinates, distances and rho in float32, as CMSSW
    fma: bool = False               # layer coordinate as one fused multiply-add (GPU) or product then sum (CPU)

    def with_(self, **kw):
        return replace(self, **kw)

    def as_dict(self):
        return asdict(self)


DEFAULT = Params()


@dataclass
class Points:
    """The points of one level: per-point arrays and `active` = the points the step clusters."""
    level: str
    E: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    layer: np.ndarray
    side: np.ndarray
    det: np.ndarray
    tag: np.ndarray
    active: np.ndarray
    cache: dict

    @property
    def n(self):
        return len(self.E)


LEVELS = ("lc", "lc_all", "rh")


def points(ev, level):
    """The points of an event at `level` (LEVELS), cached on the event."""
    key = ("points", level)
    if key not in ev._cache:
        if level in ("lc", "lc_all"):
            lc = ev.lc
            active = ev.lc_eligible()
            if level == "lc_all":  # filteredLayerClusters with min_cluster_size = 1: the size cut gone
                active = active | (np.isin(lc["det"], HGCAL_DETECTORS) & np.isin(lc["algo"], (6, 7, 8)))
            ev._cache[key] = Points(level, lc["E"], lc["x"], lc["y"], lc["z"], lc["layer"], lc["side"], lc["det"],
                                    lc["seed"].astype(np.int64), active, {})
        elif level == "rh":
            rh = ev.rh
            active = np.isin(rh["det"], HGCAL_DETECTORS) & (rh["E"] > 0)
            ev._cache[key] = Points("rh", rh["E"], rh["x"], rh["y"], rh["z"], rh["layer"],
                                    (rh["z"] > 0).astype(np.int64), rh["det"], rh["id"].astype(np.int64), active, {})
        else:
            raise ValueError(f"level must be one of {LEVELS}, not {level!r}")
    return ev._cache[key]


def _f(a, f32):
    return np.asarray(a, np.float32) if f32 else np.asarray(a, np.float64)


def _candidates(X, Y, Z, rel, rmax):
    """Candidate pairs (i, j) for the metric distance max(|dxy| * 2 / (rel_i + rel_j), |dz|) <= rmax: a superset,
    filtered exactly by the caller. Points are grouped by relative sigma (one group per detector), and every pair of
    groups is searched with the box |dx|, |dy| <= rmax * (rel_a + rel_b) / 2, |dz| <= rmax (Chebyshev distance with
    z rescaled), which holds the disc of the metric with a margin for float32 rounding."""
    groups = np.unique(rel)
    members = [np.nonzero(rel == g)[0] for g in groups]
    trees = {}
    ii, jj = [], []
    for a in range(len(groups)):
        for b in range(a, len(groups)):
            r0 = rmax * (float(groups[a]) + float(groups[b])) / 2
            zs = r0 / rmax                 # |dz| <= rmax  <=>  |dz| * zs <= r0
            rT = r0 * (1 + 1e-5) + 1e-6    # margin for float32 rounding, on every axis
            for g in (a, b):
                if (g, zs) not in trees:
                    m = members[g]
                    P = np.column_stack([np.asarray(X[m], float), np.asarray(Y[m], float),
                                         np.asarray(Z[m], float) * zs])
                    trees[(g, zs)] = cKDTree(P)
            ta, tb = trees[(a, zs)], trees[(b, zs)]
            if a == b:
                pr = ta.query_pairs(rT, p=np.inf, output_type="ndarray")
                ii.append(members[a][pr[:, 0]])
                jj.append(members[a][pr[:, 1]])
            else:
                sd = ta.sparse_distance_matrix(tb, rT, p=np.inf, output_type="ndarray")
                ii.append(members[a][sd["i"]])
                jj.append(members[b][sd["j"]])
    if not ii:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(ii), np.concatenate(jj)


def _pairs(pts, idx, p, key):
    """All ordered pairs (I, J) of the points `idx` (one endcap) within metric distance rmax, with their distances.
    Cached per coordinate setup, since only the thresholds change between most settings."""
    setup = (p.sigmaT, p.layerScale, p.f32, p.fma, max(p.outlierDistance, p.seedingDistance, p.dc),
             max(p.dc, p.outlierDistance))
    hit = pts.cache.get(key)
    if hit is not None and hit[0] == setup:
        return hit[1]
    f32 = p.f32
    x, y, z = (_f(a[idx], f32) for a in (pts.x, pts.y, pts.z))
    layer, side = pts.layer[idx], pts.side[idx]
    inv_absz = _f(1.0, f32) / np.abs(z)
    sigma = _f([p.sigmaT[SIGMA_INDEX[d]] for d in pts.det[idx]], f32)
    sref = _f(min(p.sigmaT), f32)
    rel = sigma / sref  # >= 1, as in CMSSW
    r = np.sqrt(x * x + y * y) * inv_absz  # gnomonic radius as CMSSW computes it
    inv_sref = _f(1.0, f32) / sref         # CMSSW multiplies by the inverse: same rounding
    X, Y = x * inv_absz * inv_sref, y * inv_absz * inv_sref
    # CMSSW's layer coordinate, bit for bit: the rechit-SoA layer (0-based, + side offset by the last layer) /
    # layerScale, plus a gap on the + side: one fused multiply-add on the GPU, a rounded product then a rounded sum
    # on the CPU serial backend (`fma`). dc * layerScale is a whole number of layers, so whether a neighbour exactly
    # dc away counts depends on this rounding.
    zl = _f(layer - 1 + side * LAST_LAYER, f32)
    gap = np.where(side == 1, _f(2.0, f32) * _f(max(p.dc, p.outlierDistance), f32), _f(0.0, f32))
    inv_scale = _f(1.0, f32) / _f(p.layerScale, f32)
    if p.fma:
        Z = _f(zl.astype(np.float64) * np.float64(inv_scale) + gap, f32)  # the float64 product of float32 is exact
    else:
        Z = _f(zl * inv_scale, f32) + gap
    rmax = max(p.outlierDistance, p.seedingDistance, p.dc)
    i, j = _candidates(X, Y, Z, rel, rmax)
    dx, dy, dz = X[i] - X[j], Y[i] - Y[j], Z[i] - Z[j]
    dT = np.sqrt(dx * dx + dy * dy) * _f(2.0, f32) / (rel[i] + rel[j])
    d = np.maximum(dT, np.abs(dz))
    keep = d <= rmax
    i, j, d = i[keep].astype(np.int32), j[keep].astype(np.int32), d[keep]
    out = dict(I=np.concatenate([i, j]), J=np.concatenate([j, i]), D=np.concatenate([d, d]), r=r)
    pts.cache[key] = (setup, out)
    return out


def _cluster_side(pts, idx, p, key):
    """Labels (0..n_clusters-1, -1 outlier) of the points `idx` of one endcap, before the minimum-size cut.
    Returns (labels, n_clusters)."""
    n = len(idx)
    if n == 0:
        return np.zeros(0, np.int64), 0
    dt = np.float32 if p.f32 else np.float64
    pr = _pairs(pts, idx, p, key)
    I, J, D = pr["I"], pr["J"], pr["D"]
    E = _f(pts.E[idx], p.f32)
    tag = pts.tag[idx]
    EJ = E.astype(float)  # density: self weight 1, neighbours within dc weighted 0.5 (FlatKernel)
    within = D <= dt(p.dc)
    rho = (EJ + np.bincount(I, weights=EJ[J] * 0.5 * within, minlength=n)).astype(dt)
    # pow in float64, rounded once: what powf returns, independent of numpy's SIMD float32 power, which differs by
    # an ulp on some CPUs (AVX-512) and would make the seed decision CPU dependent
    q = (dt(p.rhocPivotRadius) / pr["r"]).astype(dt)
    rhoc_i = (dt(p.rhoc) * np.power(q.astype(np.float64), np.float64(dt(p.rhocEtaExponent))).astype(dt)).astype(dt)
    dense = rho >= rhoc_i
    eff = np.where(dense, dt(p.seedingDistance), dt(p.outlierDistance))
    higher = (rho[J] > rho[I]) | ((rho[J] == rho[I]) & (rho[J] > 0) & (tag[J] > tag[I]))
    cand = higher & (D <= eff[I])
    Ic, Jc, Dc = I[cand], J[cand], D[cand]
    nh = np.full(n, -1, dtype=np.int64)
    if len(Ic):
        order = np.lexsort((-tag[Jc], -rho[Jc], Dc, Ic))     # nearest; then the higher rho, then the higher tag
        first = np.unique(Ic[order], return_index=True)[1]
        nh[Ic[order][first]] = Jc[order][first]
    is_seed = (nh == -1) & dense
    parent = np.where(nh >= 0, nh, np.arange(n))             # follow the chain to its root by pointer jumping
    while True:
        gp = parent[parent]
        if np.array_equal(gp, parent):
            break
        parent = gp
    seeds = np.nonzero(is_seed)[0]
    cid = np.full(n, -1, dtype=np.int64)
    cid[seeds] = np.arange(len(seeds))
    return np.where(is_seed[parent], cid[parent], -1), len(seeds)


def cluster(pts, p: Params = DEFAULT, drop_small=True):
    """Trackster label of every point of the level (-1: not active, outlier or in a dropped small trackster).
    Labels are unique across the two endcaps. drop_small=False returns the raw CLUEstering assignment, comparable to
    CMSSW's ticlTrackstersCLUEsteringAssignment."""
    labels = np.full(pts.n, -1, dtype=np.int64)
    offset = 0
    for side in (0, 1):
        idx = np.nonzero(pts.active & (pts.side == side))[0]
        lab, nclu = _cluster_side(pts, idx, p, side)
        if drop_small and nclu:
            small = np.bincount(lab[lab >= 0], minlength=nclu) < p.minNumLayerCluster
            lab = np.where((lab >= 0) & ~small[np.maximum(lab, 0)], lab, -1)
        labels[idx] = np.where(lab >= 0, lab + offset, -1)
        offset += nclu
    return labels
