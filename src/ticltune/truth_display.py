"""Export for the truth viewer (TruthMetrics/Display/viewer/truth_targets.html): the truth graph of each event (from
the TruthGraphJsonDumper JSON written next to the ntuple, truthgraph_<event>.json), its base particles with their
deposits, and the targets of the ideal-clustering test on layer clusters and on rechits. No reconstruction.
Base particles are joined to the graph by graph index (the ntuple's bp_id = the JSON's gid); with pileup the JSON
holds only a cone around the signal, and base particles outside it have no graph node (bp.g = -1).

The tuning objectives are shown on the emulated CLUEstering tracksters (default parameters): the truth groups
(signal targets joined by the tracksters that claim them), every term of C and P per group and per trackster
(`metrics.ideal_scores`), and the trackster and group ideal cluster of each layer cluster.

With pileup only the cells within dR < roi of a signal (gun) particle are written; the targets are always those of
the whole event. Colours are decided here: base particles that share a layer cluster or are within NEAR_CM of each
other get different colours (hardest first), and a target takes the colour of its hardest base particle, so a target
that is one particle looks the same in both panes. Two objects with one colour are never one object; only the
colours are reused far apart.
"""
import glob
import json
import os

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from . import clue, data, metrics, truth
from .data import Tracksters

N_COLOURS = 20  # the viewer's palette
NEAR_CM = 30.0  # display only: objects closer than this (projected to the front face) get different colours
GRAPH_FIELDS = ("pdg", "E", "pt", "eta", "phi", "gen", "sim", "status", "levels", "bs", "bx", "evt", "prod", "dec")
DEFAULT_ROI = 0.5  # dR around each signal particle of the cells written for events with pileup


def _r(a, nd=4):
    """Rounded list, NaN -> None (JSON has no NaN)."""
    a = np.round(np.asarray(a, float), nd)
    return [None if not np.isfinite(v) else v for v in a.tolist()]


def _i(a):
    return np.asarray(a).astype(np.int64).tolist()


def _dominant(W, cellE):
    """Per cell (row of W): the column with most energy (-1 if none), its share of the cell's energy, and the
    share with no in-time truth."""
    W = W.tocsr()
    top = W.max(axis=1).toarray().ravel()
    arg = np.where(top > 0, np.asarray(W.argmax(axis=1)).ravel(), -1)
    tot = np.asarray(W.sum(axis=1)).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        return arg, np.where(cellE > 0, top / cellE, 0.0), np.where(cellE > 0, np.clip(1 - tot / cellE, 0, 1), 0.0)


def _greedy_colours(S, E, x, y, z, rank_E=None):
    """Colour index per column of S (n_cells, n_obj; cell positions x, y, z): the first colour that no harder
    neighbour has (sharing a cell, or within NEAR_CM projected to the front face, same endcap), else the one whose
    hardest holder is softest; the N_COLOURS objects of each endcap with most rank_E (default E: e.g. the energy
    in the displayed cone) get one colour each. Display only."""
    A = ((S.T @ S) > 0).tocsr()
    with np.errstate(divide="ignore", invalid="ignore"):
        St = S.T.tocsr()
        az = np.abs(z)
        cx, cy = 322.0 * (St @ (x / az)) / E, 322.0 * (St @ (y / az)) / E
        side = np.sign(St @ np.sign(z))
    ok = np.nonzero(E > 0)[0]
    i, j = cKDTree(np.column_stack([cx[ok] + 1e5 * side[ok], cy[ok]])).query_pairs(NEAR_CM, output_type="ndarray").T
    A = (A + sp.csr_matrix((np.ones(len(i)), (ok[i], ok[j])), shape=A.shape) + sp.csr_matrix((np.ones(len(i)), (ok[j], ok[i])), shape=A.shape)).tocsr()
    col = np.full(S.shape[1], -1, np.int64)
    rank_E = E if rank_E is None else rank_E
    for sd in (-1, 1):  # the N_COLOURS most visible objects of each endcap: one colour each
        top = [u for u in np.argsort(-rank_E, kind="stable") if rank_E[u] > 0 and side[u] == sd][:N_COLOURS]
        col[top] = np.arange(len(top))
    for u in np.argsort(-E, kind="stable"):
        if col[u] >= 0:
            continue
        nb = A.indices[A.indptr[u]:A.indptr[u + 1]]
        nb = nb[(col[nb] >= 0) & (nb != u)]
        held = np.full(N_COLOURS, -1.0)
        np.maximum.at(held, col[nb], E[nb])
        free = np.nonzero(held < 0)[0]
        col[u] = free[0] if len(free) else np.argmin(held)
    return col


def _in_roi(x, y, z, axes, roi):
    """Cells within dR < roi of any (eta, phi) axis, same endcap. roi <= 0: all."""
    if roi <= 0 or not axes:
        return np.ones(len(x), bool)
    eta, phi = np.arcsinh(z / np.maximum(np.hypot(x, y), 1e-9)), np.arctan2(y, x)
    ok = np.zeros(len(x), bool)
    for ae, ap in axes:
        dphi = np.angle(np.exp(1j * (phi - ap)))
        ok |= (np.sign(eta) == np.sign(ae)) & (np.hypot(eta - ae, dphi) < roi)
    return ok


def _scoring(ev, frac, params, cfg):
    """The objectives on the emulated tracksters: event summary, truth groups with their terms, per-trackster
    terms, and the trackster and group ideal cluster of every layer cluster."""
    tg = truth.build(ev, "lc", frac)
    lab = clue.cluster(ev, params)
    ts = Tracksters.from_labels(lab, ev.lc_energy())
    d = metrics.evaluate(ev, ts, tg, cfg, keep_details=True)
    o, sm = d["ideal"], metrics.summary(d["sums"])
    lc_ts = np.full(ev.n_lc, -1, np.int64)
    lc_ts[ts.lc] = ts.owner()
    return lc_ts, o["dom"], dict(
        params=params.as_dict(), select_energy=cfg.objective_energy, signal_frac=cfg.signal_frac, claim_frac=cfg.claim_frac,
        summary={k: (None if not np.isfinite(v) else round(float(v), 5)) for k, v in sm.items()
                 if k in ("C", "P", "F") or k.startswith(("obj_", "signal_"))},
        targets=dict(signal_frac=_r(o["signal_frac"], 3), group=_i(o["target_group"]), objective=_i(o["objective"])),
        groups=dict(scored=_i(o["group_scored"]), E=_r(o["group_E"]), n_targets=_i(o["group_n_targets"]),
                    reach=_r(o["reach"]), Kall=_r(o["Kall"]), credit=_r(o["credit"]), completeness=_r(o["completeness"]),
                    purity=_r(o["purity"]), n_tracksters=_i(o["n_pieces"]), avoid=_r(o["avoid"]), tracksters_E=_r(o["pieces_E"]),
                    nfrag=_r(o["nfrag"], 3)),
        tracksters=dict(main=_i(o["ts_main"]), group=_i(o["ts_group"]), E=_r(o["ts_E"]), credit=_r(o["ts_credit"]),
                        unavoid=_r(o["ts_unavoid"]), avoid=_r(o["ts_avoid"]), own_outside=_r(o["ts_own_outside"]),
                        n_lc=_i(np.diff(ts.offsets))))


def _level(ev, level, frac, bp_col, axes, roi, lc_ts=None, lc_group=None):
    tg = truth.build(ev, level, frac)
    if level == "lc":
        c, elig = ev.lc, ev.eligible()
        cellE = c["recE"].astype(float)
    else:
        c, elig = ev.rh, ev.rh_eligible()
        cellE = c["E"].astype(float)
    inroi = elig & _in_roi(c["x"].astype(float), c["y"].astype(float), c["z"].astype(float), axes, roi)
    keep = np.nonzero(inroi)[0]
    p, ps, nt = _dominant(tg.S, cellE)
    t, tsh, _ = _dominant(tg.Tc, cellE)
    members = [np.nonzero(tg.member_of == k)[0] for k in range(tg.n)]
    E_bp = np.asarray(tg.S.sum(axis=0)).ravel()
    hardest = np.array([m[np.argmax(E_bp[m])] for m in members], np.int64)
    cells = dict(x=_r(c["x"][keep], 2), y=_r(c["y"][keep], 2), z=_r(c["z"][keep], 2), E=_r(cellE[keep]),
                 layer=_i(c["layer"][keep]), p=_i(p[keep]), ps=_r(ps[keep], 2), t=_i(t[keep]), ts=_r(tsh[keep], 2),
                 nt=_r(nt[keep], 2))
    if lc_ts is not None:
        cells["k"] = _i(lc_ts[keep])     # emulated trackster of each layer cluster (-1: none)
        cells["g"] = _i(lc_group[keep])  # truth group whose ideal cluster holds it (-1: another object or no truth)
    return dict(
        cells=cells,
        targets=dict(E=_r(tg.E), roiE=_r(np.asarray(tg.Tc[keep].sum(axis=0)).ravel()), members=[_i(m) for m in members],
                     hardest=_i(hardest), col=_i(bp_col[hardest]),
                     completeness=_r(tg.completeness, 3), purity=_r(tg.purity, 3), sig=_i(tg.is_signal)),
        bp=dict(E=_r(E_bp), roiE=_r(np.asarray(tg.S[keep].sum(axis=0)).ravel()), t=_i(tg.member_of),
                merged_at=_i(tg.merged_at), fail_c=_r(tg.fail_completeness, 3), fail_p=_r(tg.fail_purity, 3),
                partner=_i(tg.partner)),
        n_iterations=tg.n_iterations, frac=frac,
    )


def export_event(ev, graph, frac=truth.DEFAULT_FRAC, roi=DEFAULT_ROI, params=clue.DEFAULT, cfg=metrics.MetricConfig()):
    S_lc = truth.cells(ev, "lc")[1]
    P = graph["particles"]
    local = {p.get("gid", p["id"]): k for k, p in enumerate(P)}  # graph index -> row of this file
    inter = ev.bp["bx"].astype(np.int64) * 100000 + ev.bp["evt"].astype(np.int64)
    n_inter = len(np.unique(inter))
    roi = roi if n_inter > 1 else 0.0
    # the signal (gun) particles: generator final state of the signal interaction
    axes = [(p["eta"], p["phi"]) for p in P if p["gen"] and p["status"] == 1 and p.get("bx", 0) == 0 and p.get("evt", 0) == 0]
    xyz = [ev.lc[c].astype(float) for c in ("x", "y", "z")]
    in_view = S_lc[np.nonzero(_in_roi(*xyz, axes, roi))[0]]
    bp_col = _greedy_colours(S_lc, np.asarray(S_lc.sum(axis=0)).ravel(), *xyz, np.asarray(in_view.sum(axis=0)).ravel())
    lc_ts, lc_group, scoring = _scoring(ev, frac, params, cfg)
    return dict(
        run=ev.run, lumi=ev.lumi, event=ev.event, levels=graph["levels"], n_interactions=n_inter, roi=roi,
        graph=dict(particles={k: [p.get(k, 0) for p in P] for k in GRAPH_FIELDS} | dict(cb=[p.get("cb") for p in P]),
                   vertices={k: [v[k] for v in graph["vertices"]] for k in ("x", "y", "z", "reason")}),
        bp=dict(id=_i(ev.bp["id"]), g=[local.get(int(g), -1) for g in ev.bp["id"]], pdg=_i(ev.bp["pdg"]),
                Ein=_r(ev.bp["E"], 3), sig=_i(ev.bp["signal"]), kind=_i(ev.bp["kind"]),
                side=_i(np.where(ev.bp["eta"] < 0, -1, 1)), col=_i(bp_col)),
        lc=_level(ev, "lc", frac, bp_col, axes, roi, lc_ts, lc_group),
        scoring=scoring,
        rh=_level(ev, "rh", frac, bp_col, axes, roi) if ev.rh is not None else None,
    )


def export(files, outdir, frac=truth.DEFAULT_FRAC, roi=DEFAULT_ROI):
    """One JSON per event; the truth graph is read from truthgraph_<event>.json next to each ntuple, and the
    events are grouped by the ntuple's sample directory (e.g. pion/job_0 -> pion)."""
    os.makedirs(outdir, exist_ok=True)
    index = []
    for f in files:
        d = os.path.dirname(os.path.abspath(f))
        group = os.path.basename(os.path.dirname(d)) or "events"
        graphs = {json.load(open(g))["event"]: g for g in glob.glob(os.path.join(d, "truthgraph_*.json"))}
        for ev in sorted(data.load(f), key=lambda e: e.event):
            if ev.event not in graphs:
                print(f"{f}: no truth graph JSON for event {ev.event}, skipped")
                continue
            name = f"{group}_{ev.lumi}_{ev.event}.json"
            with open(os.path.join(outdir, name), "w") as out:
                json.dump(export_event(ev, json.load(open(graphs[ev.event])), frac, roi), out, separators=(",", ":"))
            index.append(dict(file=name, group=group, label=f"{group}: lumi {ev.lumi} evt {ev.event}"))
    with open(os.path.join(outdir, "index.json"), "w") as out:
        json.dump(dict(source=list(files), frac=frac, events=index), out, separators=(",", ":"))
    return index
