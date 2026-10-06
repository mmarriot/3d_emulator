"""The truth of tuning/V3_TRUTH_AND_METRICS.md (section 4): atoms -> units -> targets, on rechits.

Atoms are the truth-graph particles with energy in HGCAL rechits (inside the calorimeter included); s(c, a), their
energy in rechit c, is in the ntuple. Each atom belongs to a unit (the ntuplizer's rule: the nearest
reconstructableFinalState ancestor, the decay daughter below a pi0, else the root), the coarsest object downstream wants
as one object.

Targets: the units, merged by the ideal-clustering test on ALL HGCAL rechits (the same targets for rechit-level and
layer-cluster-level clustering). The ideal clustering of a set of objects gives every rechit, whole, to the object
with the most energy in it. An object passes if its ideal cluster holds more than `frac` of its energy
(completeness) and more than `frac` of the energy in that cluster is its own (purity). A failing object is merged
with the object of the SAME interaction it is most confused with (its energy in the other's ideal cluster plus the
other's in its own) and the test is repeated until every object passes; a failing object with no same-interaction
partner stays a target of its own, flagged unreachable. Never merging across interactions keeps unresolvable pileup
inside signal objects a contamination, and stops collecting pileup from counting as efficiency.

Also here: the ideal clusterings at both levels (the reference values of the metrics) and the natural pieces of each
target (the same test on its atoms, a failing piece joining the piece of its parent atom).
"""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .data import EM_PDG

DEFAULT_FRAC = 0.5  # strictly more than half, both ways


@dataclass
class Targets:
    frac: float
    n: int
    unit_target: np.ndarray    # unit row -> target
    T: sp.csr_matrix           # (n_rh, n) energy of every target in every rechit
    E: np.ndarray              # deposited energy of every target (all HGCAL rechits)
    signal: np.ndarray         # bool: from the hard-scatter interaction
    em: np.ndarray             # bool: class EM (the unit with the most energy is e, gamma or a pi0 daughter), else HAD
    unreachable: np.ndarray    # bool: failed the test with no same-interaction partner
    n_units: np.ndarray
    completeness: np.ndarray   # of the final ideal clustering (> frac unless unreachable)
    purity: np.ndarray
    merged_at: np.ndarray      # per unit: iteration at which its object first failed and merged (-1: never)
    partner: np.ndarray        # per unit: the hardest unit of the object it merged with (-1: never)
    n_iterations: int
    no_truth: np.ndarray       # (n_rh,) rechit energy with no in-time sim energy
    n_pieces: np.ndarray       # natural pieces per target (diagnostic)


def ideal(Tc):
    """Ideal clustering of objects with cell energies Tc (n_cells, n_obj): every cell to its largest object.
    Returns (dominant object per cell, -1 if none; completeness; purity; K) with K[o, t] = energy of object t in the
    ideal cluster of object o."""
    Tc = Tc.tocsr()
    n_obj = Tc.shape[1]
    top = Tc.max(axis=1).toarray().ravel()
    dom = np.where(top > 0, np.asarray(Tc.argmax(axis=1)).ravel(), -1)
    has = np.nonzero(dom >= 0)[0]
    owner = sp.csr_matrix((np.ones(len(has)), (dom[has], has)), shape=(n_obj, Tc.shape[0]))
    K = (owner @ Tc).tocsr()
    own = K.diagonal()
    E = np.asarray(Tc.sum(axis=0)).ravel()
    inK = np.asarray(K.sum(axis=1)).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        comp = np.where(E > 0, own / E, 0.0)
        pur = np.where(inK > 0, own / inK, 0.0)
    return dom, comp, pur, K


def _heads(lab, n, E_base):
    """The hardest base column of every object."""
    o = np.lexsort((-E_base, lab))
    first = o[np.r_[True, np.diff(lab[o]) != 0]] if len(o) else o
    head = np.full(n, -1, np.int64)
    head[lab[first]] = first
    return head


def separate(S, group, frac=DEFAULT_FRAC, partner_of=None, max_iterations=100000):
    """Merge the columns of S (n_cells, n_base) by the ideal-clustering test.

    group: per base column; merges only within a group (the interaction; for natural pieces, the target).
    partner_of: optional per base column, a preferred partner column (natural pieces: the parent atom), used for a
    failing object whose hardest column has one in another object of the same group; otherwise the partner is the
    object of the same group it is most confused with. A failing object with no partner is flagged unreachable and
    not retested (the flag is lost if another object merges into it).
    Returns (label per column, completeness, purity, unreachable per object, merged_at, partner, iterations)."""
    n_base = S.shape[1]
    E_base = np.asarray(S.sum(axis=0)).ravel()
    group = np.asarray(group)
    label = np.arange(n_base)
    flagged = np.zeros(n_base, bool)  # indexed by label value
    merged_at = np.full(n_base, -1, np.int64)
    partner = np.full(n_base, -1, np.int64)
    S = S.tocsc()
    for it in range(max_iterations):
        uniq, lab = np.unique(label, return_inverse=True)
        n = len(uniq)
        flag = flagged[uniq]
        G = sp.csr_matrix((np.ones(n_base), (np.arange(n_base), lab)), shape=(n_base, n))
        _, comp, pur, K = ideal(S @ G)
        grp = np.zeros(n, group.dtype)
        grp[lab] = group
        fail = np.nonzero(((comp <= frac) | (pur <= frac)) & ~flag)[0]
        if len(fail) == 0:
            return lab, comp, pur, flag, merged_at, partner, it
        C = (K + K.T).tocsr()[fail].tocoo()
        ok = (grp[C.col] == grp[fail[C.row]]) & (C.col != fail[C.row]) & (C.data > 0)
        best = np.full(len(fail), -1, np.int64)
        if ok.any():
            r, c, v = C.row[ok], C.col[ok], C.data[ok]
            order = np.lexsort((c, -v, r))  # the largest confusion, then the lower index
            first = np.unique(r[order], return_index=True)[1]
            best[r[order][first]] = c[order][first]
        head = _heads(lab, n, E_base)
        if partner_of is not None:
            pref = partner_of[head[fail]]
            pobj = np.where(pref >= 0, lab[np.maximum(pref, 0)], -1)
            use = (pobj >= 0) & (pobj != fail)
            use[use] &= grp[pobj[use]] == grp[fail[use]]
            best = np.where(use, pobj, best)
        lone = best < 0
        if lone.all():  # only new unreachable objects
            flagged[uniq[fail]] = True
            continue
        f_obj, p_obj = fail[~lone], best[~lone]
        where = np.full(n, -1, np.int64)
        where[f_obj] = np.arange(len(f_obj))
        first = (where[lab] >= 0) & (merged_at < 0)
        merged_at[first] = it
        partner[first] = head[p_obj[where[lab[first]]]]
        _, cc = connected_components(sp.csr_matrix((np.ones(len(f_obj)), (f_obj, p_obj)), shape=(n, n)), directed=False)
        keep = flag.copy()
        keep[fail[lone]] = True
        alone = np.bincount(cc)[cc] == 1  # objects not merged keep (or get) their flag
        label = cc[lab]
        flagged = np.zeros(n_base, bool)
        flagged[cc[alone]] = keep[alone]
    raise RuntimeError("ideal-clustering test did not converge")


def unit_cells(ev):
    """(n_rh, n_units) energy of every unit in every rechit."""
    tra = ev.tra
    return sp.csr_matrix((tra["E"], (tra["rh"], ev.at["unit"][tra["at"]])), shape=(ev.n_rh, len(ev.un["id"])))


def build(ev, frac=DEFAULT_FRAC, pieces=True):
    """Targets of an event (cached on the event)."""
    key = ("targets", frac, pieces)
    if key in ev._cache:
        return ev._cache[key]
    un = ev.un
    S = unit_cells(ev)
    inter = un["bx"].astype(np.int64) * 1_000_000 + un["evt"].astype(np.int64)
    lab, comp, pur, flag, merged_at, partner, it = separate(S, inter, frac)
    n = int(lab.max()) + 1 if len(lab) else 0
    G = sp.csr_matrix((np.ones(len(lab)), (np.arange(len(lab)), lab)), shape=(len(lab), n))
    T = (S @ G).tocsr()
    E = np.asarray(T.sum(axis=0)).ravel()
    signal = np.zeros(n, bool)
    signal[lab[un["signal"] > 0]] = True
    lead = _heads(lab, n, np.asarray(S.sum(axis=0)).ravel())  # the unit with the most energy
    em = np.isin(np.abs(un["pdg"][lead]), EM_PDG) | (un["kind"][lead] == 1)
    t = Targets(frac=frac, n=n, unit_target=lab, T=T, E=E, signal=signal, em=em, unreachable=flag,
                n_units=np.bincount(lab, minlength=n), completeness=comp, purity=pur, merged_at=merged_at,
                partner=partner, n_iterations=it, no_truth=np.asarray(ev.rh["noTruthE"], float),
                n_pieces=natural_pieces(ev, lab, frac) if pieces else np.ones(n, np.int64))
    ev._cache[key] = t
    return t


def natural_pieces(ev, unit_target, frac=DEFAULT_FRAC):
    """Number of natural pieces of every target: the ideal-clustering test on the target's atoms, on its own energy
    only (cells = (rechit, target) pairs), a failing piece joining the piece of its parent atom (else the piece it is
    most confused with)."""
    tra, at = ev.tra, ev.at
    tgt_of_atom = unit_target[at["unit"]]
    n_t = int(unit_target.max()) + 1 if len(unit_target) else 0
    cell = tra["rh"].astype(np.int64) * max(n_t, 1) + tgt_of_atom[tra["at"]]
    uc, cell_idx = np.unique(cell, return_inverse=True)
    A = sp.csr_matrix((tra["E"], (cell_idx, tra["at"])), shape=(len(uc), len(at["id"])))
    parent = at["parent"].copy()
    parent[(parent >= 0) & (tgt_of_atom[np.maximum(parent, 0)] != tgt_of_atom)] = -1
    lab = separate(A, tgt_of_atom, frac, partner_of=parent)[0]
    pieces = np.zeros(n_t, np.int64)
    np.add.at(pieces, tgt_of_atom[np.unique(lab, return_index=True)[1]], 1)
    return pieces


def ideal_labels(ev, t, level):
    """The ideal clustering at `level` as labels: every rechit ("rh") or every layer cluster ("lc", its truth from
    its rechit fractions) to its dominant target; -1 if it holds no truth energy."""
    Tc = t.T if level == "rh" else (ev.lc_to_rh() @ t.T).tocsr()
    return ideal(Tc)[0]
