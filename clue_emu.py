#!/usr/bin/env python3
"""A numpy emulation of TrackstersCLUEsteringProducer + PatternRecognitionbyCLUEstering, run on the
layer clusters stored in the campaign nano, so that algorithm variants can be scored in seconds
instead of a 12-minute cmsRun. The emulation follows the code literally:

  TracksterCLUEsteringAlgoWrapper.dev.cc   gnomonic coordinates x/|z|, y/|z| in units of the largest
                                           sigmaT; z = layer/layerScale; cylinder metric
                                           max(transverse * 2/(ratio_i+ratio_j), |dz|);
                                           rhoc(point) = rhoc * (pivot/r)^alpha
  CLUEstering ClusteringKernels.hpp        rho_i = E_i + 0.5 * sum_{d<=dc, j!=i} E_j  (FlatKernel(0.5), self weight 1)
                                           nearest higher within sd (rho >= rhoc_i) or od (rho < rhoc_i)
                                           seed: no higher & rho >= rhoc_i; outlier: no higher & rho < rhoc_i
                                           followers chain to the seed
  PatternRecognitionbyCLUEstering.cc       tracksters with < minNumLayerCluster (2) LCs dropped
  filteredLayerClustersCLUE3DHigh          mask: algo in {6,7,8} and (nHits >= 2 or scintillator)

Only the endcap that holds the CaloParticle is clustered (the two are independent in the real
algorithm too, separated by endcapGap).

Variants (all off by default, so the default setting IS the frozen algorithm):
  sparse_to_dense_only  a point below its seed threshold ("sparse") may only take a nearest higher
                        that is itself above  dense_frac * rhoc  ("dense"): DBSCAN-like border points,
                        no chaining through pile-up
  max_sparse_run        the chain from a point may pass through at most N sparse points (itself
                        included) before reaching a dense one; longer runs are dropped as outliers
  r_max                 a follower further than r_max (metric units) transversely from its seed is
                        dropped
  eta_shrink            per-point transverse scale multiplied by (r/pivot)^beta for r < pivot, i.e.
                        all radii shrink at high eta (needs the wrapper's ratio to become per point)
  od may be < sd        (needs the library's search box to be max(od, sd) instead of od)

CLUE3D-matching variants (match3d/, all off by default too). CLUE3D is the same CLUE, so these are
the places where the two implementations of it differ rather than new algorithms:
  same_layer=False      drop the dL == 0 pairs from the density AND the nearest-higher search, as
                        CLUE3D does with densityOnSameLayer / nearestHigherOnSameLayer = False
  max_dlayer            hard window of +-N layers for both, independent of dc/sd/od. The cylinder
                        metric ties the longitudinal reach to the transverse one (|dL| <= d*L), so
                        a sparse point searching out to od reaches od*L layers where CLUE3D still
                        stops at densitySiblingLayers
  self_density          CLUE3D's criticalSelfDensity: a seed must hold E_i/rho_i of the density
  nh_order="transverse" pick the nearest higher by transverse distance alone, as CLUE3D does,
                        instead of by the full cylinder metric (same candidate set, different parent)
  sigma_cm              per-point transverse scale sigma_cm/|z_i| instead of a fixed angle per
                        sub-detector: CLUE3D's cut is a fixed number of cm at the cluster's own z
"""
import os, sys, json, argparse, time
import numpy as np
import awkward as ak
import uproot
from scipy.spatial import cKDTree

SIM = "ticlSimTrackstersfromCPs"
RECO = "ticlTrackstersCLUE3DHigh"
SIGMA_T = {6: 0.003, 7: 0.006, 8: 0.012}
DEFAULT = dict(dc=1.0, rhoc=0.8, od=3.6, sd=2.8, sigmaT=(0.003, 0.006, 0.012), layerScale=4.0,
               rhocEtaExponent=2.0, rhocPivotRadius=0.42, minNumLayerCluster=2,
               sparse_to_dense_only=False, dense_frac=1.0, dense_exp=None, max_sparse_run=0, r_max=0.0,
               f32=False,       # emulate CMSSW single precision: round the distances and rho to float32
               eta_shrink=0.0, eta_shrink_pivot=0.42,
               metric="cyl", kernel="flat", k_flat=0.5, k_std=0.5, k_amp=0.5, k_avg=1.0,
               kmetric=None,   # metric used for the KERNEL only; None = the same as metric (the CMSSW behaviour)
               same_layer=True, max_dlayer=0, self_density=0.0, nh_order="metric", sigma_cm=0.0,
               depth=None)     # per-layer longitudinal coordinate (47 values, layer 1 first) that layerScale divides; None = layer index

BR = ["event", f"n{SIM}", f"{SIM}_raw_energy", f"{SIM}_barycenter_z",
      f"{SIM}vertices_vertices", f"{SIM}vertices_vertex_mult",
      "HGCalLayerClusters_algoID", "HGCalLayerClusters_nHits", "HGCalLayerClusters_energy",
      "HGCalLayerClusters_position_x", "HGCalLayerClusters_position_y", "HGCalLayerClusters_position_z"]
BR_REAL = [f"{RECO}_raw_energy", f"{RECO}_n{RECO}vertices", f"{RECO}_o{RECO}vertices", f"{RECO}vertices_vertices",
           f"SimCP2{RECO}ByHitsLinks_index", f"SimCP2{RECO}ByHitsLinks_sharedEnergy"]

_LAYER_Z = None


def layer_table(absz):
    zs = np.sort(np.unique(np.round(absz, 1)))
    groups = [[zs[0]]]
    for z in zs[1:]:
        (groups[-1].append(z) if z - groups[-1][-1] < 0.5 else groups.append([z]))
    c = np.array([np.mean(g) for g in groups])
    assert len(c) == 47, len(c)
    return c


def layer_of(absz):
    global _LAYER_Z
    if _LAYER_Z is None:
        _LAYER_Z = layer_table(absz)
    return 1 + np.argmin(np.abs(absz[:, None] - _LAYER_Z[None, :]), axis=1)


def sigma_eff(x):
    x = np.asarray(x, float); x = np.sort(x[np.isfinite(x)]); n = len(x)
    if n < 10:
        return float("nan")
    w = max(int(round(0.6827 * n)), 2)
    return float(np.min(x[w - 1:] - x[:n - w + 1]) / 2)


class Event:
    """Masked layer clusters of the CP's endcap, in the algorithm's coordinates, plus the truth."""

    def __init__(self, a, i, real=None):
        E = ak.to_numpy(a["HGCalLayerClusters_energy"][i]).astype(np.float64)
        x = ak.to_numpy(a["HGCalLayerClusters_position_x"][i]).astype(np.float64)
        y = ak.to_numpy(a["HGCalLayerClusters_position_y"][i]).astype(np.float64)
        z = ak.to_numpy(a["HGCalLayerClusters_position_z"][i]).astype(np.float64)
        algo = ak.to_numpy(a["HGCalLayerClusters_algoID"][i])
        nh = ak.to_numpy(a["HGCalLayerClusters_nHits"][i])
        self.sim_e = float(a[f"{SIM}_raw_energy"][i][0])
        side = np.sign(float(a[f"{SIM}_barycenter_z"][i][0]))
        mask = np.isin(algo, (6, 7, 8)) & ((nh >= 2) | (algo == 8)) & (np.sign(z) == side)
        self.idx = np.nonzero(mask)[0]                     # merged-collection index of every point
        self.E, self.algo = E[mask], algo[mask]
        absz = np.abs(z[mask])
        self.gx, self.gy = x[mask] / absz, y[mask] / absz  # gnomonic
        self.r = np.hypot(self.gx, self.gy)
        self.layer = layer_of(absz).astype(np.float64)
        f = np.zeros(len(E))
        sv = ak.to_numpy(a[f"{SIM}vertices_vertices"][i]); sm = ak.to_numpy(a[f"{SIM}vertices_vertex_mult"][i])
        f[sv] = 1.0 / sm
        self.f = f[mask]
        self.n = len(self.E)
        self.real_best = None
        if real is not None:
            sh = ak.to_numpy(real[f"SimCP2{RECO}ByHitsLinks_sharedEnergy"][i])
            if len(sh):
                ib = int(np.argmax(sh)); best = int(real[f"SimCP2{RECO}ByHitsLinks_index"][i][ib])
                n_v = int(real[f"{RECO}_n{RECO}vertices"][i][best]); o_v = int(real[f"{RECO}_o{RECO}vertices"][i][best])
                self.real_best = ak.to_numpy(real[f"{RECO}vertices_vertices"][i])[o_v:o_v + n_v]
                self.real_ntrk = int(len(real[f"{RECO}_raw_energy"][i]))
        self._pairs = None

    def pairs(self, rmax, layerScale, sigmaT, eta_shrink, pivot, sigma_cm=0.0, depth=None):
        """All ordered pairs (I, J) with metric distance D <= rmax, for the given coordinate setup."""
        depth = None if depth is None else tuple(depth)
        key = (rmax, layerScale, tuple(sigmaT), eta_shrink, pivot, sigma_cm, depth)
        if self._pairs is not None and self._pairs[0] == key:
            return self._pairs[1]
        if sigma_cm:
            # CLUE3D's transverse cut is sigma_cm centimetres measured at the cluster's own z, i.e.
            # the angular scale sigma_cm/|z| -- the widest of them is the reference, so ratio <= 1.
            absz = self.absz
            sigma = sigma_cm / absz
            sref = float(sigma.max())
            ratio = sigma / sref
        else:
            sref = max(sigmaT)
            ratio = np.array([sigmaT[{6: 0, 7: 1, 8: 2}[a]] / sref for a in self.algo])
        if eta_shrink:
            ratio = ratio * np.minimum(1.0, (self.r / pivot) ** eta_shrink)
        # longitudinal coordinate: the layer index (CMSSW), or any per-layer depth table (z, material...)
        zl = self.layer if depth is None else np.asarray(depth, float)[self.layer.astype(np.int64) - 1]
        X, Y, Z = self.gx / sref, self.gy / sref, zl / layerScale
        P = np.column_stack([X, Y, Z])
        tree = cKDTree(P)
        pr = tree.query_pairs(rmax * np.sqrt(2.0) + 1e-6, output_type="ndarray")
        i, j = pr[:, 0], pr[:, 1]
        dT = np.hypot(X[i] - X[j], Y[i] - Y[j]) * 2.0 / (ratio[i] + ratio[j])
        dZ = np.abs(Z[i] - Z[j])
        d = np.maximum(dT, dZ)                       # cylinder metric (the production one)
        keep = d <= rmax                              # the sphere distance is >= the cylinder one, so this superset holds both
        i, j, d, dT, dZ = i[keep], j[keep], d[keep], dT[keep], dZ[keep]
        ds = np.hypot(dT, dZ)                         # spherical (Euclidean) metric in the same scaled coordinates
        out = dict(I=np.concatenate([i, j]), J=np.concatenate([j, i]), D=np.concatenate([d, d]), DS=np.concatenate([ds, ds]),
                   DT=np.concatenate([dT, dT]), X=X, Y=Y, ratio=ratio)
        self._pairs = (key, out)
        return out


def run(ev, s, return_labels=False):
    """Cluster one event with setting s (dict, DEFAULT overridden). Returns per-event metrics
    (and, with return_labels, also the per-point cluster label array (-1 = outlier) and the number of seeds)."""
    p = dict(DEFAULT); p.update(s)
    rmax = max(p["od"], p["sd"], p["dc"])
    pr = ev.pairs(rmax, p["layerScale"], tuple(p["sigmaT"]), p["eta_shrink"], p["eta_shrink_pivot"], p["sigma_cm"], p["depth"])
    I, J = pr["I"], pr["J"]
    # CLUE3D keeps the layer window fixed and excludes the reference layer; CLUEstering does neither
    dL = np.abs(ev.layer[I] - ev.layer[J])
    ok = np.ones(len(I), bool)
    if not p["same_layer"]:
        ok &= dL > 0
    if p["max_dlayer"]:
        ok &= dL <= p["max_dlayer"]
    D = pr["D"] if p["metric"] == "cyl" else pr["DS"]
    if p["f32"]:
        D = D.astype(np.float32).astype(np.float64)
    # The kernel may use a different norm from the cuts. In CMSSW one template parameter serves both,
    # so kmetric=None reproduces it; the split exists to test whether the density wants a separable
    # (ellipsoidal, sphere) profile while the nearest-higher search keeps the cylinder's full
    # transverse reach across the layer window. Not available in cmsRun without a C++ patch.
    if p["kmetric"] is None or p["kmetric"] == p["metric"]:
        DK = D
    else:
        DK = pr["D"] if p["kmetric"] == "cyl" else pr["DS"]
        if p["f32"]:
            DK = DK.astype(np.float32).astype(np.float64)
    n, E = ev.n, ev.E
    # density: self weight 1, neighbours within dc weighted by the kernel (CLUEstering ConvolutionalKernel.hpp)
    if p["kernel"] == "flat":
        w = np.full(len(DK), p["k_flat"])
    elif p["kernel"] == "gauss":
        w = p["k_amp"] * np.exp(-DK * DK / (2 * p["k_std"] ** 2))
    elif p["kernel"] == "exp":
        w = p["k_amp"] * np.exp(-p["k_avg"] * DK)
    else:
        raise ValueError(p["kernel"])
    rho = E + np.bincount(I, weights=E[J] * w * (D <= p["dc"]) * ok, minlength=n)
    # CMSSW carries rho and the distances as floats; two values that differ below the float32
    # resolution are an exact tie there, broken by index, while float64 sees an ordering. That flips
    # whole chains between "follower" and "outlier" in ~3 % of PU200 events.
    if p["f32"]:
        rho = rho.astype(np.float32).astype(np.float64)
    rhoc = p["rhoc"] * (p["rhocPivotRadius"] / ev.r) ** p["rhocEtaExponent"]
    dense = rho >= rhoc
    eff = np.where(dense, p["sd"], p["od"])
    # nearest higher
    higher = (rho[J] > rho[I]) | ((rho[J] == rho[I]) & (rho[J] > 0) & (J > I))
    cand = higher & (D <= eff[I]) & ok
    dexp = p["rhocEtaExponent"] if p["dense_exp"] is None else p["dense_exp"]
    rhoc_dense = p["dense_frac"] * p["rhoc"] * (p["rhocPivotRadius"] / ev.r) ** dexp
    if p["sparse_to_dense_only"]:
        core = rho >= rhoc_dense
        cand &= core[J] | dense[I]
    Ic, Jc, Dc = I[cand], J[cand], D[cand]
    # CLUE3D ranks the candidates by transverse distance alone, CLUEstering by the full metric
    Dr = pr["DT"][cand] if p["nh_order"] == "transverse" else Dc
    nh = np.full(n, -1, dtype=np.int64)
    if len(Ic):
        order = np.lexsort((-Jc, -rho[Jc], Dr, Ic))
        first = np.unique(Ic[order], return_index=True)[1]
        nh[Ic[order][first]] = Jc[order][first]
    if p["max_sparse_run"]:
        core = rho >= rhoc_dense
        run_len = np.zeros(n, dtype=np.int64)
        big = 10 ** 6
        for k in np.argsort(-rho):           # chains go uphill, so parents are processed first
            if core[k]:
                continue
            run_len[k] = 1 + (run_len[nh[k]] if nh[k] >= 0 else big)
        nh[(run_len > p["max_sparse_run"]) & ~core] = -1
    is_seed = (nh == -1) & dense
    if p["self_density"]:
        with np.errstate(divide="ignore", invalid="ignore"):
            is_seed &= np.where(rho > 0, E / rho, np.inf) > p["self_density"]
    # chain to the root by pointer jumping
    parent = np.where(nh >= 0, nh, np.arange(n))
    while True:
        gp = parent[parent]
        if np.array_equal(gp, parent):
            break
        parent = gp
    cluster = np.full(n, -1, dtype=np.int64)
    seeds = np.nonzero(is_seed)[0]
    cluster[seeds] = np.arange(len(seeds))
    cluster = np.where(is_seed[parent], cluster[parent], -1)
    if p["r_max"]:
        X, Y, ratio = pr["X"], pr["Y"], pr["ratio"]
        s_ = parent
        dts = np.hypot(X - X[s_], Y - Y[s_]) * 2.0 / (ratio + ratio[s_])
        cluster[(cluster >= 0) & (dts > p["r_max"])] = -1
    # drop tracksters with fewer than minNumLayerCluster LCs
    sz = np.bincount(cluster[cluster >= 0], minlength=len(seeds))
    small = sz < p["minNumLayerCluster"]
    if len(seeds):
        cluster[(cluster >= 0) & small[np.maximum(cluster, 0)]] = -1
    n_trk = int((~small).sum())
    # best trackster = highest shared (signal) energy
    sig = E * ev.f
    shared = np.bincount(cluster[cluster >= 0], weights=sig[cluster >= 0], minlength=len(seeds))
    out = dict(n_trk=n_trk, n_seed=int(is_seed.sum()), n_out=int((cluster < 0).sum()), linked=0,
               C=np.nan, K=np.nan, purity=np.nan, n_lc=0, jac=np.nan)
    if len(shared) and shared.max() > 0:
        b = int(np.argmax(shared)); m = cluster == b
        e_best = E[m].sum(); c = sig[m].sum() / ev.sim_e
        out.update(linked=1, C=float(c), K=float((e_best - sig[m].sum()) / ev.sim_e),
                   purity=float(sig[m].sum() / e_best), n_lc=int(m.sum()))
        if ev.real_best is not None:
            A, B = set(ev.idx[m].tolist()), set(ev.real_best.tolist())
            out["jac"] = len(A & B) / len(A | B)
    if return_labels:
        return out, cluster, len(seeds)
    return out


def load_point(files, real_files=None):
    """Events of a point (list of nano files, concatenated). real_files: the arm whose trackster
    vertex lists are used for validation (same events, same order)."""
    evs = []
    for k, f in enumerate(files):
        a = uproot.open(f)["Events"].arrays(BR, library="ak")
        real = uproot.open(real_files[k])["Events"].arrays(BR_REAL + ["event"], library="ak") if real_files else None
        if real is not None:
            # align on event number (the arms may write events in a different order)
            re = ak.to_numpy(real["event"]); ae = ak.to_numpy(a["event"])
            pos = {e: i for i, e in enumerate(re)}
            real = real[np.array([pos[e] for e in ae])]
        for i in range(len(a)):
            if a[f"n{SIM}"][i] == 1:
                evs.append(Event(a, i, real))
    return evs


def _work(args):
    ev, settings = args
    return [run(ev, s) for s in settings]


def score(evs, settings, jobs=16):
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(jobs) as pool:
        res = pool.map(_work, [(ev, settings) for ev in evs], chunksize=1)
    per = {s["name"]: [r[k] for r in res] for k, s in enumerate(settings)}
    sim_e = np.array([ev.sim_e for ev in evs])
    return per, sim_e, time.time() - t0


def summarise(per, sim_e, ref=""):
    rows = {}
    names = list(per)
    good = {nm: np.array([bool(r["linked"]) and r["purity"] > 0.5 for r in per[nm]], dtype=bool) for nm in names}
    refgood = good[ref] if ref in good else good[names[0]]
    for nm in names:
        rs = per[nm]
        linked = np.array([r["linked"] for r in rs], bool)
        C = np.array([r["C"] for r in rs]); K = np.array([r["K"] for r in rs]); R = C + K
        n = len(rs)
        rows[nm] = dict(
            eff=float((linked & (C > 0.5)).sum() / n), nolink=float((~linked).sum() / n),
            f_good=float(good[nm].sum() / n), n_ref=int(refgood.sum()),
            C_med=float(np.nanmedian(C[refgood])), K_med=float(np.nanmedian(K[refgood])),
            K_med_all=float(np.nanmedian(K[linked])), K_sig=sigma_eff(K[refgood]),
            R_sig=sigma_eff(R[refgood]), R_sig_all=sigma_eff(R[linked]), R_med=float(np.nanmedian(R[refgood])),
            k_gev=float(np.nanmedian((K * sim_e)[refgood])),
            n_trk=float(np.mean([r["n_trk"] for r in rs])), n_lc=float(np.median([r["n_lc"] for r in rs if r["linked"]])),
            jac=float(np.nanmedian([r["jac"] for r in rs])) if not all(np.isnan(r["jac"]) for r in rs) else float("nan"))
    return rows


def print_rows(rows, title):
    print(f"\n=== {title} (reference set: {next(iter(rows.values()))['n_ref']} CPs good in the reference setting) ===")
    hdr = ("setting", "eff", "nolink", "f_good", "C_med", "K_med", "sK", "R_med", "sR", "sR_all", "K[GeV]", "ntrk", "nLC", "jac")
    print("%-16s %5s %6s %6s %6s %6s %5s %6s %5s %6s %6s %5s %4s %5s" % hdr)
    for nm, r in rows.items():
        print("%-16s %5.2f %6.2f %6.2f %6.3f %6.3f %5.3f %6.3f %5.3f %6.3f %6.1f %5.0f %4.0f %5.2f" % (
            nm, r["eff"], r["nolink"], r["f_good"], r["C_med"], r["K_med"], r["K_sig"], r["R_med"], r["R_sig"], r["R_sig_all"],
            r["k_gev"], r["n_trk"], r["n_lc"], r["jac"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="emulate CLUEstering on the layer clusters of campaign nanos")
    ap.add_argument("files", nargs="+", help="offline nanos (any arm: the layer clusters are the same)")
    ap.add_argument("--settings", required=True, help="json list of {name:..., <DEFAULT key>: value, ...}")
    ap.add_argument("--real", nargs="*", default=None,
                    help="the same chunks of the arm to validate against (Jaccard of the trackster LC sets)")
    ap.add_argument("--ref", default="", help="setting whose good CPs define the reference set (default: first)")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--nev", type=int, default=0)
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()
    settings = json.load(open(args.settings))
    evs = load_point(args.files, args.real)
    if args.nev:
        evs = evs[:args.nev]
    print(f"{len(evs)} events from {len(args.files)} file(s); {len(settings)} settings")
    per, sim_e, dt = score(evs, settings, args.jobs)
    rows = summarise(per, sim_e, args.ref)
    print_rows(rows, f"[{dt:.0f} s]")
    if args.out:
        json.dump(dict(rows=rows, per_event=per, sim_e=sim_e.tolist()), open(args.out, "w"))
