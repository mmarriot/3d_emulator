"""Truth targets for trackster building.

Base particles (from the ntuple) are the calorimeter-boundary particles of the signal and of every
in-time pileup interaction. Their energy in each layer cluster, s(l, b), is rechit-energy weighted.
Only the layer clusters the trackster-building step may use (`Event.eligible`) count: energy in
masked layer clusters is unreachable for this step and is reported separately.

Inseparability of two base particles a, b, measured on the layer clusters the step receives:

    M(a, b) = sum_l min(s(l,a), s(l,b)) / min(E_a, E_b)

M = 1: the smaller particle sits entirely in layer clusters shared with the other, in at least
equal measure. Pairs with M >= tau are linked; connected components are the targets. One tau
for every pair, whatever the family tree or interaction: separability is a property of the
layer clusters. A target records which interactions and origins it spans.

Targets have no energy threshold: every reachable base particle is linked. Which targets the
objectives score (`metrics.MetricConfig.select_energy`) is decided afterwards, on the target, so a
soft particle inseparable from a hard one is inside the hard target, not contamination of it.
"""
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

DEFAULT_TAU = 0.9


@dataclass
class Targets:
    tau: float
    n: int
    member_of: np.ndarray        # base-particle row -> target index (-1: no reachable energy)
    T: sp.csr_matrix             # (n_lc, n) energy of each target in each layer cluster (eligible only)
    S: sp.csr_matrix             # (n_lc, n_bp) base-particle energies (eligible only)
    E: np.ndarray                # reachable energy per target
    E_unreachable: np.ndarray    # energy in masked layer clusters per target
    E_signal: np.ndarray         # part of E from signal base particles
    E_pileup: np.ndarray         # part of E from pileup base particles
    is_signal: np.ndarray        # target contains a signal base particle
    n_members: np.ndarray
    n_interactions: np.ndarray
    n_origins: np.ndarray
    origin_pdg: np.ndarray       # the origin's pdg id if the target has a single origin, else 0
    pairs: Dict[str, np.ndarray] = field(default_factory=dict)  # a, b, M of every LC-sharing pair

    def describe(self, t):
        return (f"target {t}: E={self.E[t]:.3f} GeV ({'signal' if self.is_signal[t] else 'pileup'}), "
                f"{self.n_members[t]} particles, {self.n_interactions[t]} interaction(s), {self.n_origins[t]} origin(s)")


def _row_pairs(indptr):
    """Indices (into the CSR data) of every unordered pair of entries sharing a row."""
    counts = np.diff(indptr)
    left, right = [], []
    for k in np.unique(counts[counts >= 2]):
        rows = np.nonzero(counts == k)[0]
        a, b = np.triu_indices(k, 1)
        base = indptr[rows][:, None]
        left.append((base + a[None, :]).ravel())
        right.append((base + b[None, :]).ravel())
    if not left:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(left), np.concatenate(right)


def inseparability(S):
    """M(a, b) for every pair of base particles sharing at least one layer cluster.
    S: (n_lc, n_bp) CSR of energies. Returns (a, b, M) with a < b."""
    S = S.tocsr()
    S.sort_indices()
    Eb = np.asarray(S.sum(axis=0)).ravel()
    i, j = _row_pairs(S.indptr)
    if len(i) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    a, b = S.indices[i], S.indices[j]
    m = np.minimum(S.data[i], S.data[j])
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = lo.astype(np.int64) * S.shape[1] + hi
    uk, inv = np.unique(key, return_inverse=True)
    shared = np.bincount(inv, weights=m)
    a, b = uk // S.shape[1], uk % S.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        M = shared / np.minimum(Eb[a], Eb[b])
    return a, b, np.nan_to_num(M)


def build(ev, tau=DEFAULT_TAU):
    """Targets of one event at threshold tau (cached per tau)."""
    key = ("targets", tau)
    if key in ev._cache:
        return ev._cache[key]
    nlc, nbp = ev.n_lc, ev.n_bp
    elig = ev.eligible()
    full = sp.csr_matrix((ev.tr_E, (ev.tr_lc, ev.tr_bp)), shape=(nlc, nbp))
    S = sp.diags(elig.astype(float)) @ full
    S = S.tocsr()
    S.eliminate_zeros()
    Eb = np.asarray(S.sum(axis=0)).ravel()
    Eb_un = np.asarray(full.sum(axis=0)).ravel() - Eb
    a, b, M = inseparability(S)
    link = M >= tau
    reach = Eb > 0
    adj = sp.csr_matrix((np.ones(int(link.sum())), (a[link], b[link])), shape=(nbp, nbp))
    _, comp = connected_components(adj, directed=False)
    # renumber over reachable particles only
    member_of = np.full(nbp, -1, np.int64)
    uc, inv = np.unique(comp[reach], return_inverse=True)
    member_of[reach] = inv
    n = len(uc)
    G = sp.csr_matrix((np.ones(int(reach.sum())), (np.nonzero(reach)[0], inv)), shape=(nbp, n))
    T = (S @ G).tocsr()
    sig = ev.bp["signal"].astype(bool)
    E = np.bincount(inv, weights=Eb[reach], minlength=n)
    E_sig = np.bincount(inv, weights=(Eb * sig)[reach], minlength=n)
    # unreachable energy of a target = that of its members (particles with no reachable energy at
    # all are not targets; their energy is reported by the metrics as unreachable, untargeted)
    E_un = np.bincount(inv, weights=Eb_un[reach], minlength=n)
    rows = np.nonzero(reach)[0]
    inter = ev.bp["bx"].astype(np.int64) * 100000 + ev.bp["evt"].astype(np.int64)
    n_int = np.array([len(np.unique(inter[rows[inv == t]])) for t in range(n)]) if n else np.zeros(0, int)
    origins = ev.bp["origin"]
    n_org = np.array([len(np.unique(origins[rows[inv == t]])) for t in range(n)]) if n else np.zeros(0, int)
    org_pdg = np.array([ev.bp["originPdg"][rows[inv == t]][0] if n_org[t] == 1 else 0 for t in range(n)],
                       dtype=np.int64) if n else np.zeros(0, np.int64)
    out = Targets(tau=tau, n=n, member_of=member_of, T=T, S=S, E=E, E_unreachable=E_un, E_signal=E_sig,
                  E_pileup=E - E_sig, is_signal=E_sig > 0 if n else np.zeros(0, bool),
                  n_members=np.bincount(inv, minlength=n), n_interactions=n_int, n_origins=n_org,
                  origin_pdg=org_pdg, pairs=dict(a=a, b=b, M=M))
    # a target is signal if it holds any signal particle, even one with little energy
    sig_member = np.zeros(n, bool)
    np.logical_or.at(sig_member, inv, sig[reach])
    out.is_signal = sig_member
    ev._cache[key] = out
    return out
