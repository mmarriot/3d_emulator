"""Numpy emulation of the CMSSW CLUEstering trackster building, returning one label per layer cluster.

Follows, line by line where it matters for the result:
  RecoHGCal/TICL/plugins/alpaka/TracksterCLUEsteringAlgoWrapper.dev.cc
      gnomonic coordinates x/|z|, y/|z| in units of the smallest sigmaT; z = layer / layerScale with
      CMSSW's layer index and float rounding (see _pairs);
      cylinder metric max(|dxy| * 2 / (rel_i + rel_j), |dz|); rhoc_i = rhoc * (pivot / r_i)^alpha;
      tie-break tag = the layer cluster's seed DetId; the two endcaps never interact.
  CLUEstering 2.12 core/detail/ClusteringKernels.hpp, FlatKernel(0.5)
      rho_i = sum_{d(i,j) <= dc} k_ij E_j with k_ii = 1 and k_ij = 0.5 otherwise;
      nearest higher among j with rho_j > rho_i (or equal, rho_j > 0, tag_j > tag_i) within
      seedingDistance if rho_i >= rhoc_i else outlierDistance; on equal distance the higher rho,
      then the higher tag; seed = no nearest higher and rho_i >= rhoc_i; a chain that ends on a
      non-seed is an outlier.
  RecoHGCal/TICL/plugins/PatternRecognitionbyCLUEstering.cc
      masked layer clusters are left out; tracksters with fewer than minNumLayerCluster LCs dropped.

Defaults are the CMSSW configuration (RecoHGCal/TICL/python/CLUE3DHighStep_cff.py, ticl_dev).
The metric is invariant under the choice of the reference sigmaT (coordinates and ratios scale
together), so the emulation is exact whichever reference CMSSW uses.

Research variants (all off by default, so the default IS the CMSSW algorithm) are carried over from
the original clue_emu.py; see `Params`.
"""
from dataclasses import dataclass, replace, asdict
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .data import SIGMA_INDEX, HGCAL_DETECTORS


@dataclass(frozen=True)
class Params:
    # --- the CMSSW parameters (TrackstersCLUEsteringProducer + plugin) ---
    dc: float = 1.0
    rhoc: float = 0.8
    outlierDistance: float = 2.8
    seedingDistance: float = 2.8
    sigmaT: Tuple[float, float, float] = (0.003, 0.006, 0.012)  # EE, HSi, HSci
    layerScale: float = 6.0
    rhocEtaExponent: float = 2.0
    rhocPivotRadius: float = 0.42
    minNumLayerCluster: int = 2
    f32: bool = True  # carry coordinates, distances and rho in float32, as CMSSW does
    # --- research variants (off = CMSSW) ---
    metric: str = "cyl"            # "cyl" (CMSSW) or "sphere"
    kernel: str = "flat"           # "flat" (CMSSW), "gauss", "exp"
    k_flat: float = 0.5
    k_std: float = 0.5
    k_amp: float = 0.5
    k_avg: float = 1.0
    kmetric: Optional[str] = None  # metric of the kernel only; None = same as `metric`
    sparse_to_dense_only: bool = False
    dense_frac: float = 1.0
    dense_exp: Optional[float] = None
    max_sparse_run: int = 0
    r_max: float = 0.0
    eta_shrink: float = 0.0
    eta_shrink_pivot: float = 0.42
    same_layer: bool = True
    max_dlayer: int = 0
    self_density: float = 0.0
    nh_order: str = "metric"       # "metric" (CMSSW) or "transverse"
    sigma_cm: float = 0.0
    depth: Optional[Tuple[float, ...]] = None  # per-layer longitudinal coordinate; None = layer index

    def with_(self, **kw):
        return replace(self, **kw)

    def as_dict(self):
        return asdict(self)


DEFAULT = Params()

# layers per endcap (rhtools lastLayer, D128): the + side offset of the rechit-SoA layer index
LAST_LAYER = 47


def _f(a, f32):
    return np.asarray(a, np.float32) if f32 else np.asarray(a, np.float64)


def _pairs(ev, side_idx, p, cache_key):
    """All ordered pairs (I, J) of points of one endcap within metric distance rmax, with their
    distances. Cached per event and coordinate setup, since only the thresholds change between
    most settings."""
    key = (cache_key, p.sigmaT, p.layerScale, p.eta_shrink, p.eta_shrink_pivot, p.sigma_cm, p.depth, p.f32,
           max(p.outlierDistance, p.seedingDistance, p.dc), max(p.dc, p.outlierDistance))
    hit = ev._cache.get(("pairs", cache_key))
    if hit is not None and hit[0] == key:
        return hit[1]
    lc = ev.lc
    f32 = p.f32
    x, y, z = (_f(lc[k][side_idx], f32) for k in ("x", "y", "z"))
    det = lc["det"][side_idx]
    layer = lc["layer"][side_idx]
    absz = np.abs(z)
    inv_absz = _f(1.0, f32) / absz
    gx, gy = x * inv_absz, y * inv_absz
    if p.sigma_cm:
        sigma = _f(p.sigma_cm, f32) / absz
    else:
        sigma = _f([p.sigmaT[SIGMA_INDEX[d]] for d in det], f32)
    sref = _f(sigma.min() if p.sigma_cm else min(p.sigmaT), f32)
    rel = sigma / sref  # >= 1, as in CMSSW
    r = np.hypot(gx, gy)
    if p.eta_shrink:
        rel = rel * np.minimum(1.0, (r / p.eta_shrink_pivot) ** p.eta_shrink).astype(rel.dtype)
    inv_sref = _f(1.0, f32) / sref  # CMSSW multiplies by the inverse: same rounding
    X, Y = gx * inv_sref, gy * inv_sref
    if p.depth is None:
        # CMSSW's layer coordinate, bit for bit: the rechit-SoA layer (0-based, + side offset by the last
        # layer) / layerScale, plus a gap on the + side, fused into one multiply-add on the GPU. dc *
        # layerScale is a whole number of layers, so whether a neighbour exactly dc away counts depends
        # on this rounding.
        side = lc["side"][side_idx]
        zl = _f(layer - 1 + side * LAST_LAYER, f32)
        gap = np.where(side == 1, _f(2.0, f32) * _f(max(p.dc, p.outlierDistance), f32), _f(0.0, f32))
        # float32 fma: the float64 product of two float32 is exact
        Z = _f(zl.astype(np.float64) * np.float64(_f(1.0, f32) / _f(p.layerScale, f32)) + gap, f32)
    else:
        Z = _f(np.asarray(p.depth, float)[layer - 1], f32) * _f(1.0 / p.layerScale, f32)
    rmax = max(p.outlierDistance, p.seedingDistance, p.dc)
    # Superset search in coordinates where the metric never shrinks a distance: scale the transverse
    # axes by the largest relative sigma, so |dxy|/relmax <= metric transverse distance.
    relmax = float(rel.max()) if len(rel) else 1.0
    P = np.column_stack([np.asarray(X, float) / relmax, np.asarray(Y, float) / relmax, np.asarray(Z, float)])
    tree = cKDTree(P)
    pr = tree.query_pairs(rmax * np.sqrt(2.0) + 1e-6, output_type="ndarray")
    i, j = pr[:, 0], pr[:, 1]
    dx, dy, dz = X[i] - X[j], Y[i] - Y[j], Z[i] - Z[j]
    two = _f(2.0, f32)
    dT = np.sqrt(dx * dx + dy * dy) * two / (rel[i] + rel[j])
    dZ = np.abs(dz)
    d = np.maximum(dT, dZ)
    ds = np.sqrt(dT * dT + dZ * dZ)
    keep = (d <= rmax) | (ds <= rmax)
    i, j, d, ds, dT = i[keep], j[keep], d[keep], ds[keep], dT[keep]
    out = dict(I=np.concatenate([i, j]), J=np.concatenate([j, i]), D=np.concatenate([d, d]),
               DS=np.concatenate([ds, ds]), DT=np.concatenate([dT, dT]), X=X, Y=Y, rel=rel, r=r,
               layer=layer)
    ev._cache[("pairs", cache_key)] = (key, out)
    return out


def _cluster_side(ev, side_idx, p, cache_key):
    """Labels (0..n_clusters-1, -1 outlier) for the points `side_idx` of one endcap, before the
    minimum-size cut. Returns (labels, n_clusters)."""
    n = len(side_idx)
    if n == 0:
        return np.zeros(0, np.int64), 0
    f32 = p.f32
    dt = np.float32 if f32 else np.float64
    pr = _pairs(ev, side_idx, p, cache_key)
    I, J = pr["I"], pr["J"]
    E = _f(ev.lc["E"][side_idx], f32)
    tag = ev.lc["seed"][side_idx].astype(np.int64)
    layer = pr["layer"]
    dL = np.abs(layer[I] - layer[J])
    ok = np.ones(len(I), bool)
    if not p.same_layer:
        ok &= dL > 0
    if p.max_dlayer:
        ok &= dL <= p.max_dlayer
    D = pr["D"] if p.metric == "cyl" else pr["DS"]
    DK = D if (p.kmetric is None or p.kmetric == p.metric) else (pr["D"] if p.kmetric == "cyl" else pr["DS"])
    # density: self weight 1, neighbours within dc weighted by the kernel
    if p.kernel == "flat":
        w = np.full(len(DK), p.k_flat)
    elif p.kernel == "gauss":
        w = p.k_amp * np.exp(-DK.astype(float) ** 2 / (2 * p.k_std ** 2))
    elif p.kernel == "exp":
        w = p.k_amp * np.exp(-p.k_avg * DK.astype(float))
    else:
        raise ValueError(p.kernel)
    within = (D <= dt(p.dc)) & ok
    rho = (E.astype(float) + np.bincount(I, weights=E[J].astype(float) * w * within, minlength=n)).astype(dt)
    r = pr["r"]
    rhoc_i = (dt(p.rhoc) * (dt(p.rhocPivotRadius) / r) ** dt(p.rhocEtaExponent)).astype(dt)
    dense = rho >= rhoc_i
    eff = np.where(dense, dt(p.seedingDistance), dt(p.outlierDistance))
    higher = (rho[J] > rho[I]) | ((rho[J] == rho[I]) & (rho[J] > 0) & (tag[J] > tag[I]))
    cand = higher & (D <= eff[I]) & ok
    dexp = p.rhocEtaExponent if p.dense_exp is None else p.dense_exp
    rhoc_dense = p.dense_frac * p.rhoc * (p.rhocPivotRadius / r) ** dexp
    if p.sparse_to_dense_only:
        core = rho >= rhoc_dense
        cand &= core[J] | dense[I]
    Ic, Jc, Dc = I[cand], J[cand], D[cand]
    Dr = pr["DT"][cand] if p.nh_order == "transverse" else Dc
    nh = np.full(n, -1, dtype=np.int64)
    if len(Ic):
        # nearest; on equal distance the higher rho, then the higher tag
        order = np.lexsort((-tag[Jc], -rho[Jc], Dr, Ic))
        first = np.unique(Ic[order], return_index=True)[1]
        nh[Ic[order][first]] = Jc[order][first]
    if p.max_sparse_run:
        core = rho >= rhoc_dense
        run_len = np.zeros(n, dtype=np.int64)
        big = 10 ** 6
        for k in np.argsort(-rho.astype(float), kind="stable"):  # parents (uphill) first
            if core[k]:
                continue
            run_len[k] = 1 + (run_len[nh[k]] if nh[k] >= 0 else big)
        nh[(run_len > p.max_sparse_run) & ~core] = -1
    is_seed = (nh == -1) & dense
    if p.self_density:
        with np.errstate(divide="ignore", invalid="ignore"):
            is_seed &= np.where(rho > 0, E / rho, np.inf) > p.self_density
    # follow the chain to its root by pointer jumping
    parent = np.where(nh >= 0, nh, np.arange(n))
    while True:
        gp = parent[parent]
        if np.array_equal(gp, parent):
            break
        parent = gp
    seeds = np.nonzero(is_seed)[0]
    cid = np.full(n, -1, dtype=np.int64)
    cid[seeds] = np.arange(len(seeds))
    labels = np.where(is_seed[parent], cid[parent], -1)
    if p.r_max:
        X, Y, rel = pr["X"], pr["Y"], pr["rel"]
        s_ = parent
        dts = np.hypot(X - X[s_], Y - Y[s_]) * 2.0 / (rel + rel[s_])
        labels[(labels >= 0) & (dts > p.r_max)] = -1
    return labels, len(seeds)


def cluster(ev, p: Params = DEFAULT, drop_small=True):
    """Trackster label of every layer cluster of the event (-1: masked, outlier or in a dropped
    small trackster). Labels are unique across the two endcaps. drop_small=False returns the raw
    CLUEstering assignment, comparable to CMSSW's ticlTrackstersCLUEsteringAssignment."""
    labels = np.full(ev.n_lc, -1, dtype=np.int64)
    elig = ev.eligible()
    offset = 0
    for side in (0, 1):
        idx = np.nonzero(elig & (ev.lc["side"] == side))[0]
        lab, nclu = _cluster_side(ev, idx, p, cache_key=side)
        if drop_small and nclu:
            size = np.bincount(lab[lab >= 0], minlength=nclu)
            small = size < p.minNumLayerCluster
            lab = np.where((lab >= 0) & ~small[np.maximum(lab, 0)], lab, -1)
        labels[idx] = np.where(lab >= 0, lab + offset, -1)
        offset += nclu
    return labels
