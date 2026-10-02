"""Truth targets for trackster building: which calorimeter-boundary particles could an ideal clustering
reconstruct on their own?

Base particles (from the ntuple) are the calorimeter-boundary particles of the signal and of every in-time
pileup interaction: what the truth graph says arrived at the calorimeter. Each owns the sim energy of everything
it produced inside, rechit weighted per input cell: s(c, b). The cells are the layer clusters (what trackster
building receives, `level="lc"`) or the rechits (`level="rh"`, what the whole chain starts from). Only cells the
trackster-building step may use count (`Event.eligible`); energy elsewhere is reported as unreachable.

Ideal-clustering test. The ideal clustering of a set of objects gives every cell, whole, to the object with the
most energy in it. Object t passes if its ideal cluster K_t holds more than `frac` of its energy (completeness)
and more than `frac` of the truth energy in K_t is its own (purity): the metrics' "individual" outcome
(`metrics.MetricConfig.individual_frac`), for the best clustering any algorithm could make of these cells.

An object that fails cannot be reconstructed on its own by any algorithm working on these cells, so it is merged
with the object it is most confused with: the o maximising s_t(K_o) + s_o(K_t) (its energy in o's ideal cluster
plus o's in its own). Merging changes the ideal clustering, so the test is repeated on the merged objects until
every object passes. The targets are the final objects: no geometry, no thresholds on distance or overlap, and
no energy threshold (which targets the objectives score is decided afterwards, `MetricConfig.select_energy`).
"""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

DEFAULT_FRAC = 0.5  # as metrics.MetricConfig.individual_frac: strictly more than half, both ways


@dataclass
class Targets:
    level: str                   # "lc" or "rh": the cells the test was run on
    frac: float
    n: int
    member_of: np.ndarray        # base-particle row -> target index (-1: no reachable energy)
    T: sp.csr_matrix             # (n_lc, n) energy of each target in each layer cluster (eligible only): what metrics use
    Tc: sp.csr_matrix            # (n_cells, n) the same in the cells of the test
    S: sp.csr_matrix             # (n_cells, n_bp) base-particle energies in the cells of the test (eligible only)
    E: np.ndarray                # reachable energy per target
    completeness: np.ndarray     # of each target's ideal cluster (> frac by construction)
    purity: np.ndarray
    E_unreachable: np.ndarray    # energy in cells the step may not use, per target
    E_signal: np.ndarray
    E_pileup: np.ndarray
    is_signal: np.ndarray        # holds a signal base particle
    n_members: np.ndarray
    n_interactions: np.ndarray
    n_origins: np.ndarray
    origin_pdg: np.ndarray       # the origin's pdg id if the target has a single origin, else 0
    merged_at: np.ndarray        # per base particle: iteration at which its object first failed (-1: never)
    fail_completeness: np.ndarray  # ... that object's completeness and purity then
    fail_purity: np.ndarray
    partner: np.ndarray          # ... and the hardest base particle of the object it was merged with (-1: never)
    n_iterations: int = 0
    E_untargeted: float = 0.0    # energy of base particles with no reachable energy

    def describe(self, t):
        return (f"target {t}: E={self.E[t]:.3f} GeV ({'signal' if self.is_signal[t] else 'pileup'}), "
                f"{self.n_members[t]} particles, {self.n_interactions[t]} interaction(s), {self.n_origins[t]} origin(s)")


def ideal(Tc):
    """Ideal clustering of objects with cell energies Tc (n_cells, n_obj): every cell to its largest object.
    Returns (dominant object per cell, -1 if no truth energy; completeness; purity; K) with
    K[o, t] = energy of object t in the ideal cluster of object o."""
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


def separate(S, frac=DEFAULT_FRAC, max_iterations=1000):
    """Target label per column of S (n_cells, n_bp; columns with no energy must be removed beforehand) and the
    merge record, by the ideal-clustering test (module docstring)."""
    n_bp = S.shape[1]
    E_bp = np.asarray(S.sum(axis=0)).ravel()
    label = np.arange(n_bp)
    merged_at = np.full(n_bp, -1, np.int64)
    fail_c, fail_p = np.full(n_bp, np.nan), np.full(n_bp, np.nan)
    partner = np.full(n_bp, -1, np.int64)
    for it in range(max_iterations):
        uniq, lab = np.unique(label, return_inverse=True)
        n = len(uniq)
        G = sp.csr_matrix((np.ones(n_bp), (np.arange(n_bp), lab)), shape=(n_bp, n))
        _, comp, pur, K = ideal(S @ G)
        fail = np.nonzero((comp <= frac) | (pur <= frac))[0]
        if len(fail) == 0:
            return lab, comp, pur, merged_at, fail_c, fail_p, partner, it
        C = K + K.T  # confusion of every pair of objects
        C = (C - sp.diags(C.diagonal())).tocsr()[fail]
        part = np.asarray(C.argmax(axis=1)).ravel()
        assert (C.max(axis=1).toarray().ravel() > 0).all()  # a failing object always shares energy with another
        o = np.lexsort((-E_bp, lab))  # hardest base particle of each object, to name the partner
        head = o[np.r_[True, np.diff(lab[o]) != 0]]
        hardest = np.empty(n, np.int64)
        hardest[lab[head]] = head
        first = np.isin(lab, fail) & (merged_at < 0)
        fpos = np.searchsorted(fail, lab[first])
        merged_at[first] = it
        fail_c[first], fail_p[first] = comp[lab[first]], pur[lab[first]]
        partner[first] = hardest[part[fpos]]
        A = sp.csr_matrix((np.ones(len(fail)), (fail, part)), shape=(n, n))
        _, cc = connected_components(A, directed=False)
        label = cc[lab]
    raise RuntimeError("ideal-clustering test did not converge")


def _per_target_unique(inv, values, n):
    """Number of distinct values per target, and the value where there is only one."""
    pairs_ = np.unique(np.stack([inv, values]), axis=1)
    count = np.bincount(pairs_[0], minlength=n)
    single = np.zeros(n, np.int64)
    single[pairs_[0]] = pairs_[1]
    return count, np.where(count == 1, single, 0)


def cells(ev, level):
    """(full, eligible) base-particle energies in the cells of a level: (n_cells, n_bp) CSR matrices."""
    if level == "lc":
        full = sp.csr_matrix((ev.tr_E, (ev.tr_lc, ev.tr_bp)), shape=(ev.n_lc, ev.n_bp))
        elig = ev.eligible()
    elif level == "rh":
        if ev.rh is None:
            raise ValueError("this ntuple has no rechit truth table (rh_*, trh_*)")
        full = sp.csr_matrix((ev.trh_E, (ev.trh_rh, ev.trh_bp)), shape=(len(ev.rh["E"]), ev.n_bp))
        elig = ev.rh_eligible()
    else:
        raise ValueError(f"level must be 'lc' or 'rh', not {level!r}")
    S = (sp.diags(elig.astype(float)) @ full).tocsr()
    S.eliminate_zeros()
    return full, S


def build(ev, level="lc", frac=DEFAULT_FRAC):
    """Targets of one event (cached per setting)."""
    key = ("targets", level, frac)
    if key in ev._cache:
        return ev._cache[key]
    full, S = cells(ev, level)
    Eb = np.asarray(S.sum(axis=0)).ravel()
    Eb_un = np.asarray(full.sum(axis=0)).ravel() - Eb
    reach = np.nonzero(Eb > 0)[0]
    lab, comp, pur, m_at, f_c, f_p, part, n_it = separate(S[:, reach], frac)
    n = int(lab.max()) + 1 if len(lab) else 0
    member_of = np.full(ev.n_bp, -1, np.int64)
    member_of[reach] = lab
    merged_at = np.full(ev.n_bp, -1, np.int64)
    merged_at[reach] = m_at
    fail_c, fail_p = np.full(ev.n_bp, np.nan), np.full(ev.n_bp, np.nan)
    fail_c[reach], fail_p[reach] = f_c, f_p
    partner = np.full(ev.n_bp, -1, np.int64)
    partner[reach] = np.where(part >= 0, reach[np.maximum(part, 0)], -1)
    G = sp.csr_matrix((np.ones(len(reach)), (reach, lab)), shape=(ev.n_bp, n))
    _, S_lc = cells(ev, "lc")
    sig = ev.bp["signal"][reach].astype(bool)
    E = np.bincount(lab, weights=Eb[reach], minlength=n)
    E_sig = np.bincount(lab, weights=(Eb[reach] * sig), minlength=n)
    is_sig = np.zeros(n, bool)
    np.logical_or.at(is_sig, lab, sig)
    inter = ev.bp["bx"].astype(np.int64) * 100000 + ev.bp["evt"].astype(np.int64)
    n_int, _ = _per_target_unique(lab, inter[reach], n)
    n_org, _ = _per_target_unique(lab, ev.bp["origin"][reach].astype(np.int64), n)
    _, pdg1 = _per_target_unique(lab, ev.bp["originPdg"][reach].astype(np.int64), n)
    out = Targets(level=level, frac=frac, n=n, member_of=member_of, T=(S_lc @ G).tocsr(), Tc=(S @ G).tocsr(), S=S,
                  E=E, completeness=comp, purity=pur, E_unreachable=np.bincount(lab, weights=Eb_un[reach], minlength=n),
                  E_signal=E_sig, E_pileup=E - E_sig, is_signal=is_sig, n_members=np.bincount(lab, minlength=n),
                  n_interactions=n_int, n_origins=n_org, origin_pdg=np.where(n_org == 1, pdg1, 0),
                  merged_at=merged_at, fail_completeness=fail_c, fail_purity=fail_p, partner=partner,
                  n_iterations=n_it, E_untargeted=float(Eb_un[Eb <= 0].sum()))
    ev._cache[key] = out
    return out
